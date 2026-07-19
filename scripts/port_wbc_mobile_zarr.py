#!/usr/bin/env python3
"""Port WBC mobile-manipulation raw HDF5 episodes to a ManiFlow point-cloud zarr.

The 3D sibling of ``scripts/port_wbc_mobile_hdf5.py``. Both read the SAME raw takes
(``wbc_vr_robot.py --record``, 10 Hz, flat ``<raw-dir>/episode_*.hdf5``) and this script
REUSES that porter's episode loading, windowing and state/action assembly verbatim, so
the ``state``/``action`` written here are bit-identical to the LeRobot dataset's
``observation.state``/``action``. Only the observation differs: instead of RGB video, a
world-frame head point cloud.

Emitted per split (``<out-root>/<name>_<split>.zarr``), the layout ManiFlow's
``ReplayBuffer.copy_from_path`` expects::

    data/point_cloud   (T, num_points, 6) float32   world XYZ + RGB[0,1]
    data/state         (T, 32)            float32   agent_pos
    data/action        (T, 29)            float32
    data/env_state     (T, object_nums*3) float32   only with --positions-dir; see below
    data/env_dino      (T, object_nums*D) float32   only with --dino: frame-0 DINOv3
                                                    region features (task-slot order),
                                                    constant per episode
    data/point_mask    (T, num_points)    int8      only with --masks-dir: tracked object
                                                    label per stored point (0 bg,
                                                    1 box/src, 2 cloth/dst)
    meta/episode_ends  (E,)               int64     cumulative frame counts

Point cloud (see ``omniteleop.wbc_pointcloud``): head depth is unprojected through the
static intrinsic and the FK head pose composed with the odometry base pose, giving points
in the ENGAGE-ORIGIN WORLD frame -- the same frame as the 29-D ``action`` targets, the
``state[29:32]`` odometry anchor, and the SceneDiff object positions. Points are cropped
to the measured task volume, uniformly pooled, then farthest-point-sampled to
``--num-points``. Colours ride along because ManiFlow slices ``[..., :3]`` when
``use_pc_color=False``, making colour a train-time flag rather than a reconversion.

``--state-frame`` defaults to ``world`` (NOT the LeRobot porter's ``base``): the
observation is world-frame geometry, so the EEF/head blocks of ``agent_pos`` are composed
to world too and the policy sees the hand in the same coordinates as the points and the
action. ``base`` reproduces the LeRobot porter's egocentric state.

``--positions-dir`` (optional) enables position conditioning: the SceneDiff object
positions under ``<positions-dir>/<source>/episode_<N>.npz`` are arranged into the
per-episode-constant [s1_src, s1_dst, ...] world-frame vector -- the SAME loading,
validation and (source, raw_index) keying as the LeRobot porter, via its helpers -- and
broadcast to every frame as ``data/env_state``. All npz files are validated up front;
one missing/bad episode aborts the run before anything is written.

``--dino`` (needs ``--positions-dir``) additionally reads each episode's DINO sidecar
``episode_<N>_dino.npz`` (scene_diff/scripts/extract_object_dino_feats.py): frame-0
DINOv3 region features, size-slot ordered, re-ordered here by the npz ``order`` into
task slots and broadcast as ``data/env_dino``. The sidecar's ``base_sha256`` must match
the positions npz it was computed from -- a stage-3 re-run invalidates stale sidecars.

``--masks-dir`` (optional) attaches tracked object masks
(scene_diff/scripts/build_wbc_masks.py NPZs, full RAW episode length) as
``data/point_mask``: each stored point's label, selected through the EXACT same
valid-depth -> crop -> pool -> FPS row lineage as its XYZRGB (wbc_pointcloud index
propagation) -- never re-projected. ``data/point_cloud`` and the rng draw sequence are
bit-identical with and without masks; verify_split_zarr proves it by recomputing
frame 0 from scratch.

Splits reuse ``<raw-dir>/split.csv`` (rows ``episode, split``; unlisted -> train), each
written to its own zarr. ManiFlow carves *validation* out of the train zarr itself via
its dataset ``val_ratio``, so a ``val`` split here is optional.

A ``<name>_<split>.meta.json`` sidecar records the crop/depth/FPS parameters plus an
ordered episode manifest. Each zarr episode maps to its source HDF5, full-resolution
mask NPZ, kept raw-frame window and zarr-frame window, all with half-open ``[start,end)``
ranges. The live rollout MUST rebuild its cloud with those exact values -- a
train/deploy mismatch in the frame, crop or depth scale silently poisons the policy.

Environment: ``dexmate`` or ``dexmate_maniflow`` (both have pinocchio/pink + fpsample).
Writes zarr **format 2** stores (``numcodecs.Blosc``) so ManiFlow's ``ReplayBuffer`` can
read them under either zarr 2.x or 3.x. No GPU -- the downsampler is fpsample's CPU
bucket-FPS.

Usage:
  python scripts/port_wbc_mobile_zarr.py \
    --raw-dir ~/Dexmate/data/raw_data \
    --out-root ~/Dexmate/data/processed_wbc/maniflow \
    --name dexmate_wbc --num-points 1024 \
    --include-recovery-data ~/Dexmate/data/raw_data/recovery --overwrite \
    --positions-dir ~/Dexmate/data/scene_diff/positions
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

from omniteleop.wbc_pointcloud import (
    CROP_MAX,
    CROP_MIN,
    MAX_DEPTH_M,
    MIN_DEPTH_M,
    POINT_CHANNELS,
    build_world_cloud,
    resample_clouds,
    sampler_signature,
    world_t_head_from_state,
)
from omniteleop.wbc_policy_format import ACTION_AXES, STATE_AXES, WBCPolicyFK


def _load_hdf5_porter():
    """Import the sibling HDF5 porter by path (``scripts/`` is not a package).

    Everything episode-level -- discovery, split parsing, the leading-NaN-gripper drop,
    the recovery trim, and the state/action assembly -- is reused from it so the two
    datasets can never drift apart.
    """
    path = Path(__file__).resolve().parent / "port_wbc_mobile_hdf5.py"
    spec = importlib.util.spec_from_file_location("port_wbc_mobile_hdf5", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import the HDF5 porter at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


porter = _load_hdf5_porter()

DEFAULT_OUT_ROOT = Path("~/Dexmate/data/processed_wbc/maniflow").expanduser()
DEFAULT_NAME = "dexmate_wbc"
DEFAULT_NUM_POINTS = 1024
# Uniform pool before FPS, bounding its cost. The measured crop keeps >13k points per
# frame; below the pool size the subsample simply draws with replacement.
DEFAULT_POOL_SIZE = 16384
# 32 frames * 1024 * 6 * 4 B ~= 0.8 MB/chunk: small enough for the sampler's random
# short-window reads, large enough to compress well.
DEFAULT_CHUNK_FRAMES = 32
DEFAULT_STATE_FRAME = "world"

# data/point_mask labels, fixed by the scene_diff mask builder (build_wbc_masks.py).
MASK_LEGEND = "0=background 1=box(src) 2=cloth(dst)"


def load_episode_masks(masks_dir: Path, source: str, raw_index: int,
                       frame_count: int, image_hw: tuple[int, int]) -> np.ndarray:
    """``<masks-dir>/<source>/episode_<N>.npz`` -> ``(T_raw, H, W)`` uint8, validated.

    The masks MUST cover the FULL raw episode (the builder tracks from raw frame 0; the
    porter applies its own window), match the recorder's image size, and carry the
    exact label legend this porter writes into ``data/point_mask``.
    """
    path = Path(masks_dir) / source / f"episode_{raw_index}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"masks file for {source}/episode_{raw_index} not found: {path} "
            "(run scene_diff/scripts/build_wbc_masks.py first, or drop --masks-dir; "
            "note the builder auto-REJECTS episodes that fail its QC gate)")
    with np.load(path, allow_pickle=False) as z:
        for key in ("masks", "legend", "n_frames"):
            if key not in z.files:
                raise ValueError(f"{path}: missing '{key}' (regenerate with build_wbc_masks.py)")
        legend = str(z["legend"])
        if legend != MASK_LEGEND:
            raise ValueError(f"{path}: legend {legend!r} != {MASK_LEGEND!r}")
        n_frames = int(z["n_frames"])
        masks = np.asarray(z["masks"])
    if masks.dtype != np.uint8 or masks.shape != (frame_count, *image_hw):
        raise ValueError(
            f"{path}: masks {masks.shape} {masks.dtype} != ({frame_count}, "
            f"{image_hw[0]}, {image_hw[1]}) uint8 -- built from a different take?")
    if n_frames != frame_count:
        raise ValueError(f"{path}: n_frames {n_frames} != raw frame count {frame_count}")
    if masks.max() > 2:
        raise ValueError(f"{path}: label {masks.max()} outside the legend")
    return masks


def validate_all_episode_masks(masks_dir: Path, work: list[dict],
                               positions_dir: Path | None) -> None:
    """Fail fast (before any zarr is written) on missing/contract-less masks NPZs, and
    tie each one to the CURRENT positions NPZ: the seed's task-slot ordering came from
    that file, so a positions re-run must invalidate stale masks or point_mask labels
    could silently disagree with env_state. Full length/size validation happens per
    episode at load."""
    for item in work:
        path = Path(masks_dir) / item["source"] / f"episode_{item['raw_index']}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"masks file missing for {item['source']}/episode_{item['raw_index']}: "
                f"{path} (REJECTED by the builder's QC gate, or the batch is incomplete)")
        with np.load(path, allow_pickle=False) as z:
            missing = [k for k in ("masks", "legend", "n_frames") if k not in z.files]
            stored_sha = str(z["positions_sha256"]) if "positions_sha256" in z.files else None
        if missing:
            raise ValueError(f"{path}: missing fields {missing}")
        if positions_dir is not None:
            if stored_sha is None:
                raise ValueError(
                    f"{path}: no positions_sha256 -- rebuild the masks (or patch the "
                    "field) so they are provably tied to the current positions NPZ")
            base = Path(positions_dir) / item["source"] / f"episode_{item['raw_index']}.npz"
            if stored_sha != hashlib.sha256(base.read_bytes()).hexdigest():
                raise ValueError(
                    f"{path}: positions_sha256 does not match {base.name} -- the "
                    "positions NPZ changed after tracking; re-run build_wbc_masks.py")
    logging.info("Masks ON: %d episode NPZ(s) present under %s%s", len(work), masks_dir,
                 "" if positions_dir is None else " (positions provenance verified)")


def load_episode_dino(positions_dir: Path, source: str, raw_index: int,
                      object_nums: int) -> np.ndarray:
    """DINO sidecar -> task-slot-arranged ``(object_nums * D,)`` float32.

    Cross-checks the sidecar against its base positions npz: ``base_sha256`` (a stage-3
    re-run invalidates stale sidecars), ``scene_diff_obj_ids`` and ``order`` equality,
    and ``dino_valid`` all-ones. Additionally requires the ``[0,1]`` marker in
    ``dino_source``: sidecars from before the 2026-07-18 input-scale fix share their
    base npz with corrected ones, so only this marker tells them apart. Features are
    stored size-slot ordered; arranged here by ``order`` exactly like the positions
    themselves.
    """
    base_path = Path(positions_dir) / source / f"episode_{raw_index}.npz"
    path = base_path.with_name(base_path.stem + "_dino.npz")
    if not path.exists():
        raise FileNotFoundError(
            f"DINO sidecar for {source}/episode_{raw_index} not found: {path} "
            "(run scene_diff/scripts/extract_object_dino_feats.py first, or drop --dino)")
    with np.load(base_path, allow_pickle=False) as base:
        base_order = np.asarray(base["order"])
        base_ids = np.asarray(base["scene_diff_obj_ids"])
    with np.load(path, allow_pickle=False) as z:
        for key in ("dino_region_feats", "dino_valid", "base_sha256",
                    "scene_diff_obj_ids", "order"):
            if key not in z.files:
                raise ValueError(f"{path}: missing '{key}' (regenerate the sidecar)")
        feats = np.asarray(z["dino_region_feats"], dtype=np.float32)
        # base_sha256 can't catch this one: sidecars computed on uint8 [0,255] input
        # (2026-07-18 scale bug) share the base with corrected [0,1] ones. The source
        # string is the only marker distinguishing them.
        source_tag = str(z["dino_source"]) if "dino_source" in z.files else "<missing>"
        if "[0,1]" not in source_tag:
            raise ValueError(
                f"{path}: dino_source '{source_tag}' predates the [0,1] input-scale "
                "fix; re-run extract_object_dino_feats.py")
        if str(z["base_sha256"]) != hashlib.sha256(base_path.read_bytes()).hexdigest():
            raise ValueError(
                f"{path}: base_sha256 does not match {base_path.name} -- the positions "
                "npz changed after the sidecar was computed; re-run "
                "extract_object_dino_feats.py")
        if not np.array_equal(np.asarray(z["order"]), base_order) or \
                not np.array_equal(np.asarray(z["scene_diff_obj_ids"]), base_ids):
            raise ValueError(f"{path}: order/scene_diff_obj_ids disagree with the base npz")
        if not np.all(np.asarray(z["dino_valid"]) == 1.0):
            raise ValueError(f"{path}: dino_valid {z['dino_valid']} has invalid slots")
    if feats.ndim != 2 or feats.shape[0] != object_nums:
        raise ValueError(f"{path}: dino_region_feats {feats.shape} != ({object_nums}, D)")
    if not np.all(np.isfinite(feats)) or not feats.any():
        raise ValueError(f"{path}: dino_region_feats non-finite or all-zero")
    return feats[base_order].reshape(-1)


def validate_all_episode_dino(positions_dir: Path, work: list[dict],
                              object_nums: int) -> dict[tuple[str, int], np.ndarray]:
    """Load + validate EVERY episode's DINO vector up front; uniform D enforced."""
    out = {}
    dims = set()
    for item in work:
        vec = load_episode_dino(positions_dir, item["source"], item["raw_index"],
                                object_nums)
        dims.add(vec.shape[0])
        out[(item["source"], item["raw_index"])] = vec
    if len(dims) != 1:
        raise ValueError(f"inconsistent env_dino widths across episodes: {sorted(dims)}")
    return out


def episode_arrays(hdf5_path: Path, fk: WBCPolicyFK, fps: int, *,
                   trim: tuple[int, int] | None, num_points: int, pool_size: int,
                   state_frame: str, crop_min: tuple[float, float, float],
                   crop_max: tuple[float, float, float], min_depth: float,
                   max_depth: float, rng: np.random.Generator,
                   masks_dir: Path | None = None, source: str | None = None,
                   raw_index: int | None = None,
                   ) -> tuple[
                       np.ndarray, np.ndarray, np.ndarray, int, int, int,
                       np.ndarray | None,
                   ]:
    """Build one raw take's policy arrays and raw-frame window.

    Returns ``(point_cloud (T',N,6), state (T',32), action (T',29), T_raw,
    raw_start, raw_end_exclusive, point_mask (T',N) int8 | None)``.

    ``T'`` is the kept window: the leading-NaN-gripper drop for a raw episode, or the
    absolute inclusive ``trim`` segment for a recovery episode -- identical windowing to
    the LeRobot porter, via its own helpers.

    With ``masks_dir`` set, each stored point's tracked-object label rides the exact
    valid->crop->pool->FPS row lineage of its XYZRGB (index propagation, never
    re-projection). The cloud values and the rng draw sequence are identical with and
    without masks.
    """
    ep = porter.load_and_validate_episode(hdf5_path, fps)
    start, end = porter._resolve_window(ep["frame_count"], ep["t0"], trim, hdf5_path)
    kept = end - start
    intrinsic = ep["intrinsic"]
    rgb, depth = ep["rgb"], ep["depth"]

    mask_frames = None
    if masks_dir is not None:
        mask_frames = load_episode_masks(
            masks_dir, source, raw_index, ep["frame_count"], depth.shape[1:3])

    states = np.empty((kept, len(STATE_AXES)), dtype=np.float32)
    actions = np.empty((kept, len(ACTION_AXES)), dtype=np.float32)
    clouds: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    with h5py.File(hdf5_path, "r") as raw:
        for i, t in enumerate(range(start, end)):
            action_eef, action_head = porter.load_action_targets_world(raw, t)
            state, action = porter.compute_frame_state_action(
                raw, t, fk, action_eef, action_head, state_frame=state_frame)
            states[i], actions[i] = state, action
            if mask_frames is None:
                clouds.append(build_world_cloud(
                    depth[t], rgb[t], intrinsic,
                    world_t_head_from_state(state, state_frame),
                    crop_min=crop_min, crop_max=crop_max,
                    min_depth=min_depth, max_depth=max_depth))
            else:
                cloud, pix_idx = build_world_cloud(
                    depth[t], rgb[t], intrinsic,
                    world_t_head_from_state(state, state_frame),
                    crop_min=crop_min, crop_max=crop_max,
                    min_depth=min_depth, max_depth=max_depth,
                    return_pixel_index=True)
                clouds.append(cloud)
                labels.append(mask_frames[t].ravel()[pix_idx])

    point_mask = None
    if mask_frames is None:
        point_cloud = resample_clouds(clouds, num_points, pool_size=pool_size, rng=rng)
    else:
        point_cloud, chosen = resample_clouds(
            clouds, num_points, pool_size=pool_size, rng=rng, return_indices=True)
        point_mask = np.empty((kept, num_points), dtype=np.int8)
        for i in range(kept):
            point_mask[i] = labels[i][chosen[i]].astype(np.int8)

    logging.info("  %s: %d frames [%d,%d) of %d, crop kept %d..%d pts/frame%s",
                 hdf5_path.name, kept, start, end, ep["frame_count"],
                 min(c.shape[0] for c in clouds), max(c.shape[0] for c in clouds),
                 "" if point_mask is None else
                 f", obj pts/frame {int((point_mask > 0).sum(1).min())}.."
                 f"{int((point_mask > 0).sum(1).max())}")
    return point_cloud, states, actions, ep["frame_count"], start, end, point_mask


def _zarr_blosc():
    """Blosc compressor that works on both zarr 2.x (``zarr.Blosc``) and 3.x (numcodecs only)."""
    import numcodecs  # noqa: PLC0415

    return numcodecs.Blosc(cname="zstd", clevel=3, shuffle=1)


def _zarr_writable_root(zarr_path: Path):
    """Open a writable group; always emit **format 2** so ManiFlow can ``zarr.open`` it.

    zarr 3.x dropped ``DirectoryStore`` / ``create_dataset`` / ``zarr.Blosc``; writing with
    ``zarr_format=2`` keeps the on-disk layout compatible with ManiFlow's ReplayBuffer.
    """
    import zarr  # noqa: PLC0415

    major = int(zarr.__version__.split(".", 1)[0])
    if major >= 3:
        return zarr.group(store=str(zarr_path), overwrite=True, zarr_format=2), major
    return zarr.group(store=zarr.DirectoryStore(str(zarr_path)), overwrite=True), major


def _zarr_create_array(group, name: str, *, major: int, data=None, **kwargs):
    """``create_dataset`` (zarr 2) / ``create_array`` (zarr 3).

    zarr 3 rejects passing ``data`` and ``dtype`` together; drop ``dtype`` when ``data`` is set.
    """
    create = group.create_array if major >= 3 else group.create_dataset
    if data is not None:
        if major >= 3:
            kwargs.pop("dtype", None)
        return create(name, data=data, **kwargs)
    return create(name, **kwargs)


def write_split_zarr(zarr_path: Path, items: list[dict], fk: WBCPolicyFK, args,
                     rng: np.random.Generator,
                     env_state_by_key: dict[tuple[str, int], np.ndarray] | None,
                     env_dino_by_key: dict[tuple[str, int], np.ndarray] | None,
                     ) -> tuple[int, int, list[dict]]:
    """Write one split incrementally; return episode/frame counts and raw provenance."""
    compressor = _zarr_blosc()
    root, major = _zarr_writable_root(zarr_path)
    data, meta = root.create_group("data"), root.create_group("meta")
    chunk = min(args.chunk_frames, 100)
    pc_ds = _zarr_create_array(
        data, "point_cloud", major=major,
        shape=(0, args.num_points, POINT_CHANNELS),
        chunks=(chunk, args.num_points, POINT_CHANNELS), dtype="float32", compressor=compressor)
    state_ds = _zarr_create_array(
        data, "state", major=major,
        shape=(0, len(STATE_AXES)), chunks=(100, len(STATE_AXES)),
        dtype="float32", compressor=compressor)
    action_ds = _zarr_create_array(
        data, "action", major=major,
        shape=(0, len(ACTION_AXES)), chunks=(100, len(ACTION_AXES)),
        dtype="float32", compressor=compressor)
    env_ds = None
    if env_state_by_key is not None:
        env_dim = args.object_nums * 3
        env_ds = _zarr_create_array(
            data, "env_state", major=major,
            shape=(0, env_dim), chunks=(100, env_dim),
            dtype="float32", compressor=compressor)
    dino_ds = None
    if env_dino_by_key is not None:
        dino_dim = next(iter(env_dino_by_key.values())).shape[0]
        dino_ds = _zarr_create_array(
            data, "env_dino", major=major,
            shape=(0, dino_dim), chunks=(100, dino_dim),
            dtype="float32", compressor=compressor)
    mask_ds = None
    if args.masks_dir is not None:
        mask_ds = _zarr_create_array(
            data, "point_mask", major=major,
            shape=(0, args.num_points), chunks=(chunk, args.num_points),
            dtype="int8", compressor=compressor)

    episode_ends: list[int] = []
    episode_manifest: list[dict] = []
    total = 0
    for zarr_episode_index, item in enumerate(items):
        point_cloud, states, actions, _, raw_start, raw_end, point_mask = episode_arrays(
            item["path"], fk, args.fps, trim=item["trim"], num_points=args.num_points,
            pool_size=args.pool_size, state_frame=args.state_frame,
            crop_min=tuple(args.crop_min), crop_max=tuple(args.crop_max),
            min_depth=args.min_depth, max_depth=args.max_depth, rng=rng,
            masks_dir=args.masks_dir, source=item["source"],
            raw_index=item["raw_index"])
        if raw_end - raw_start != len(states):
            raise RuntimeError(
                f"{item['path']}: manifest raw window [{raw_start},{raw_end}) has "
                f"{raw_end - raw_start} frames, zarr episode has {len(states)}"
            )
        zarr_start = total
        pc_ds.append(point_cloud)
        state_ds.append(states)
        action_ds.append(actions)
        if env_ds is not None:
            env_vec = env_state_by_key[(item["source"], item["raw_index"])]
            env_ds.append(np.tile(env_vec, (len(states), 1)))
        if dino_ds is not None:
            dino_vec = env_dino_by_key[(item["source"], item["raw_index"])]
            dino_ds.append(np.tile(dino_vec, (len(states), 1)))
        if mask_ds is not None:
            mask_ds.append(point_mask)
        total += len(states)
        episode_ends.append(total)
        mask_path = None
        if args.masks_dir is not None:
            mask_path = (
                Path(args.masks_dir) / item["source"] /
                f"episode_{item['raw_index']}.npz"
            ).resolve()
        episode_manifest.append({
            "zarr_episode_index": zarr_episode_index,
            "source": item["source"],
            "raw_index": int(item["raw_index"]),
            "hdf5_path": str(Path(item["path"]).resolve()),
            "mask_npz_path": None if mask_path is None else str(mask_path),
            "raw_frame_start": int(raw_start),
            "raw_frame_end_exclusive": int(raw_end),
            "zarr_frame_start": int(zarr_start),
            "zarr_frame_end_exclusive": int(total),
        })

    _zarr_create_array(
        meta, "episode_ends", major=major,
        data=np.asarray(episode_ends, dtype=np.int64),
        dtype="int64", compressor=compressor)
    return len(items), total, episode_manifest


def verify_split_zarr(zarr_path: Path, first_item: dict, fk: WBCPolicyFK, args) -> None:
    """Reopen a written zarr and cross-check it against a fresh recomputation."""
    import zarr  # noqa: PLC0415

    root = zarr.open(str(zarr_path), mode="r")
    required = ["data/point_cloud", "data/state", "data/action", "meta/episode_ends"]
    for flag, key in ((args.positions_dir, "env_state"), (args.dino, "env_dino"),
                      (args.masks_dir, "point_mask")):
        if flag:
            required.append(f"data/{key}")
        elif key in root["data"]:
            raise RuntimeError(f"{zarr_path}: has data/{key} but its flag is unset")
    for key in required:
        if key not in root:
            raise RuntimeError(f"{zarr_path}: missing required array {key}")
    point_cloud, state, action = root["data/point_cloud"], root["data/state"], root["data/action"]
    episode_ends = np.asarray(root["meta/episode_ends"])

    frames = point_cloud.shape[0]
    if point_cloud.shape[1:] != (args.num_points, POINT_CHANNELS):
        raise RuntimeError(f"{zarr_path}: point_cloud {point_cloud.shape[1:]} != "
                           f"({args.num_points}, {POINT_CHANNELS})")
    if state.shape != (frames, len(STATE_AXES)) or action.shape != (frames, len(ACTION_AXES)):
        raise RuntimeError(f"{zarr_path}: state {state.shape} / action {action.shape} "
                           f"disagree with {frames} frames")
    for name, arr in (("point_cloud", point_cloud), ("state", state), ("action", action)):
        if arr.dtype != np.float32:
            raise RuntimeError(f"{zarr_path}: {name} dtype {arr.dtype} != float32")
    if episode_ends.ndim != 1 or episode_ends.size == 0:
        raise RuntimeError(f"{zarr_path}: episode_ends must be non-empty 1-D")
    if int(episode_ends[-1]) != frames:
        raise RuntimeError(f"{zarr_path}: episode_ends[-1]={int(episode_ends[-1])} != {frames}")
    if np.any(np.diff(episode_ends) <= 0):
        raise RuntimeError(f"{zarr_path}: episode_ends is not strictly increasing")

    # Frame 0 of the split's first episode must equal an independent rebuild -- states,
    # actions, the CLOUD ITSELF, and (with masks) the point labels. episode_arrays
    # consumes rng draws sequentially from a per-split default_rng(seed), so a fresh
    # generator's first draw reproduces frame 0 of the first episode exactly.
    ep = porter.load_and_validate_episode(first_item["path"], args.fps)
    start, _ = porter._resolve_window(ep["frame_count"], ep["t0"], first_item["trim"],
                                      first_item["path"])
    with h5py.File(first_item["path"], "r") as raw:
        action_eef, action_head = porter.load_action_targets_world(raw, start)
        exp_state, exp_action = porter.compute_frame_state_action(
            raw, start, fk, action_eef, action_head, state_frame=args.state_frame)
    if not np.allclose(state[0], exp_state, atol=1e-4):
        raise RuntimeError(f"{zarr_path}: state[0] != recompute at raw frame {start}")
    if not np.allclose(action[0], exp_action, atol=1e-4):
        raise RuntimeError(f"{zarr_path}: action[0] != recompute at raw frame {start}")

    world_t_cam = world_t_head_from_state(exp_state, args.state_frame)
    exp_cloud, exp_pix = build_world_cloud(
        ep["depth"][start], ep["rgb"][start], ep["intrinsic"], world_t_cam,
        crop_min=tuple(args.crop_min), crop_max=tuple(args.crop_max),
        min_depth=args.min_depth, max_depth=args.max_depth, return_pixel_index=True)
    exp_pc, exp_chosen = resample_clouds(
        [exp_cloud], args.num_points, pool_size=args.pool_size,
        rng=np.random.default_rng(args.seed), return_indices=True)
    if not np.array_equal(np.asarray(point_cloud[0]), exp_pc[0]):
        raise RuntimeError(f"{zarr_path}: point_cloud[0] != bit-exact recompute -- the "
                           "cloud path changed (index propagation must not alter it)")

    stored = np.asarray(point_cloud[0])
    xyz, rgb = stored[:, :3], stored[:, 3:]
    lo, hi = np.asarray(args.crop_min), np.asarray(args.crop_max)
    if not np.all((xyz > lo) & (xyz < hi)):
        raise RuntimeError(f"{zarr_path}: point_cloud[0] escapes the workspace crop")
    if not (0.0 <= rgb.min() and rgb.max() <= 1.0):
        raise RuntimeError(f"{zarr_path}: point_cloud[0] rgb outside [0,1]")
    if len(np.unique(xyz, axis=0)) != args.num_points:
        raise RuntimeError(f"{zarr_path}: point_cloud[0] has duplicate points "
                           "(FPS should return distinct picks)")

    if args.masks_dir is not None:
        point_mask = root["data/point_mask"]
        if point_mask.shape != (frames, args.num_points) or point_mask.dtype != np.int8:
            raise RuntimeError(f"{zarr_path}: point_mask {point_mask.shape} "
                               f"{point_mask.dtype} != ({frames}, {args.num_points}) int8")
        values = np.asarray(point_mask[0])
        if values.min() < 0 or values.max() > 2:
            raise RuntimeError(f"{zarr_path}: point_mask[0] outside the label legend")
        mask_frames = load_episode_masks(
            args.masks_dir, first_item["source"], first_item["raw_index"],
            ep["frame_count"], ep["depth"].shape[1:3])
        exp_labels = mask_frames[start].ravel()[exp_pix][exp_chosen[0]].astype(np.int8)
        if not np.array_equal(values, exp_labels):
            raise RuntimeError(f"{zarr_path}: point_mask[0] != recompute through the "
                               "same pixel lineage")

    if args.dino:
        env_dino = root["data/env_dino"]
        exp_dino = load_episode_dino(args.positions_dir, first_item["source"],
                                     first_item["raw_index"], args.object_nums)
        if env_dino.shape != (frames, exp_dino.shape[0]) or env_dino.dtype != np.float32:
            raise RuntimeError(f"{zarr_path}: env_dino {env_dino.shape} {env_dino.dtype} "
                               f"!= ({frames}, {exp_dino.shape[0]}) float32")
        first_ep_dino = np.asarray(env_dino[:int(episode_ends[0])])
        if not np.array_equal(first_ep_dino,
                              np.tile(exp_dino, (len(first_ep_dino), 1))):
            raise RuntimeError(f"{zarr_path}: env_dino of episode 0 is not the constant "
                               "sidecar vector")

    if args.positions_dir is not None:
        env_state = root["data/env_state"]
        env_dim = args.object_nums * 3
        if env_state.shape != (frames, env_dim) or env_state.dtype != np.float32:
            raise RuntimeError(f"{zarr_path}: env_state {env_state.shape} {env_state.dtype} "
                               f"!= ({frames}, {env_dim}) float32")
        first_ep = np.asarray(env_state[:int(episode_ends[0])])
        exp_env = porter.load_episode_positions(
            args.positions_dir, first_item["source"], first_item["raw_index"],
            args.object_nums)
        if not np.array_equal(first_ep, np.tile(exp_env, (len(first_ep), 1))):
            raise RuntimeError(f"{zarr_path}: env_state of episode 0 is not the constant "
                               f"npz vector for {first_item['source']}/"
                               f"episode_{first_item['raw_index']}")

    logging.info("verify[%s] OK: point_cloud=%s state=%s action=%s episodes=%d frames=%d%s",
                 zarr_path.name, point_cloud.shape, state.shape, action.shape,
                 len(episode_ends), frames,
                 "" if args.positions_dir is None else " env_state=" + str(
                     tuple(root["data/env_state"].shape)))


def write_meta_sidecar(zarr_path: Path, split: str, args, episodes: int, frames: int,
                       episode_manifest: list[dict]) -> None:
    """Record the exact preprocessing the live rollout must reproduce."""
    if len(episode_manifest) != episodes:
        raise RuntimeError(
            f"episode manifest has {len(episode_manifest)} entries for {episodes} episodes"
        )
    expected_start = 0
    for index, entry in enumerate(episode_manifest):
        zarr_start = entry["zarr_frame_start"]
        zarr_end = entry["zarr_frame_end_exclusive"]
        raw_kept = entry["raw_frame_end_exclusive"] - entry["raw_frame_start"]
        if (entry["zarr_episode_index"] != index or zarr_start != expected_start or
                zarr_end <= zarr_start or raw_kept != zarr_end - zarr_start):
            raise RuntimeError(f"invalid episode manifest entry {index}: {entry}")
        expected_start = zarr_end
    if expected_start != frames:
        raise RuntimeError(f"episode manifest terminates at {expected_start}, expected {frames}")
    meta = {
        "schema": "wbc_maniflow_pointcloud_v2",
        "split": split,
        "num_points": args.num_points,
        "point_channels": ["x", "y", "z", "r", "g", "b"],
        "point_frame": "world (engage-origin, same as action targets)",
        "crop_min": list(args.crop_min), "crop_max": list(args.crop_max),
        "min_depth_m": args.min_depth, "max_depth_m": args.max_depth,
        "pool_size": args.pool_size, "downsample": "uniform pool -> farthest point sample",
        # Exact downsampler identity; the live rollout refuses a mismatch (see
        # wbc_pointcloud.sampler_signature -- an unpinned start_idx would silently skew
        # the deploy cloud away from the one the policy trained on).
        **sampler_signature(),
        "state_axes": list(STATE_AXES), "state_frame": args.state_frame,
        "action_axes": list(ACTION_AXES),
        "action_frame": "world (verbatim ik.solve targets, engage-origin)",
        "fps": args.fps, "episodes": episodes, "frames": frames,
        "seed": args.seed,
        "episode_manifest": episode_manifest,
    }
    if args.positions_dir is not None:
        meta["env_state"] = {
            "object_nums": args.object_nums,
            "axes": porter.env_state_axes(args.object_nums),
            "frame": "world (engage-origin, same as point_cloud/action)",
            "positions_dir": str(args.positions_dir),
            "constant_per_episode": True,
        }
    if args.dino:
        meta["env_dino"] = {
            "object_nums": args.object_nums,
            "layout": "task-slot order [src, dst], D per slot; frame-0 DINOv3 "
                      "region features (see extract_object_dino_feats.py)",
            "constant_per_episode": True,
            "dino_input_scale": "[0,1]",
        }
    if args.masks_dir is not None:
        meta["point_mask"] = {
            "legend": MASK_LEGEND,
            "masks_dir": str(args.masks_dir),
            "tracker": "sam3.1_multiplex (scene_diff build_wbc_masks.py)",
            "selection": "index propagation through valid->crop->pool->FPS "
                         "(point_cloud bit-identical to a maskless build)",
        }
    zarr_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def convert(args) -> None:
    work = porter.build_work_list(args.raw_dir, args.include_recovery_data)

    explicit_split = args.split_csv is not None
    split_csv = args.split_csv if explicit_split else args.raw_dir / "split.csv"
    if split_csv.exists():
        split_map = porter.parse_split_csv(split_csv)
        logging.info("Using split csv %s (%d listed episode(s))", split_csv, len(split_map))
    elif explicit_split:
        raise RuntimeError(f"--split-csv not found: {split_csv}")
    else:
        logging.info("No split csv at %s; all episodes -> train", split_csv)
        split_map = {}
    splits = porter.assign_splits(work, split_map, args.include_recovery_data)

    n_recovery = sum(item["source"] == "recovery" for item in work)
    logging.info("Port plan: %d episode(s) = %d raw + %d recovery -> {%s} "
                 "(state_frame=%s, num_points=%d)", len(work), len(work) - n_recovery,
                 n_recovery, ", ".join(f"{s}:{len(v)}" for s, v in splits.items()),
                 args.state_frame, args.num_points)

    # Position conditioning: load + validate EVERY episode's env-state before any zarr is
    # created/overwritten, exactly like the LeRobot porter (same helpers, same keying).
    env_state_by_key = None
    if args.positions_dir is not None:
        env_state_by_key = porter.validate_all_episode_positions(
            args.positions_dir, work, args.object_nums)
        logging.info("Position conditioning ON: %d env-state vector(s) (%d-D, world frame) "
                     "from %s", len(env_state_by_key), args.object_nums * 3,
                     args.positions_dir)
    env_dino_by_key = None
    if args.dino:
        env_dino_by_key = validate_all_episode_dino(
            args.positions_dir, work, args.object_nums)
        logging.info("DINO conditioning ON: %d env-dino vector(s) (%d-D) from the "
                     "positions sidecars", len(env_dino_by_key),
                     next(iter(env_dino_by_key.values())).shape[0])
    if args.masks_dir is not None:
        validate_all_episode_masks(args.masks_dir, work, args.positions_dir)

    # Fail fast on pre-existing outputs BEFORE any heavy work.
    args.out_root.mkdir(parents=True, exist_ok=True)
    zarr_paths = {s: args.out_root / f"{args.name}_{s}.zarr" for s in splits}
    for split, path in zarr_paths.items():
        if path.exists():
            if not args.overwrite:
                raise RuntimeError(f"{path} already exists. Use --overwrite to replace it.")
            logging.info("Removing existing %s (--overwrite)", path)
            shutil.rmtree(path)

    fk = WBCPolicyFK()
    for split, items in splits.items():
        path = zarr_paths[split]
        logging.info("Writing %s (%d episode(s))", path, len(items))
        rng = np.random.default_rng(args.seed)
        episodes, frames, episode_manifest = write_split_zarr(
            path, items, fk, args, rng, env_state_by_key, env_dino_by_key
        )
        write_meta_sidecar(path, split, args, episodes, frames, episode_manifest)
        verify_split_zarr(path, items[0], fk, args)
        logging.info("Split %s finalized: %d episode(s), %d frames", split, episodes, frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=porter.DEFAULT_RAW_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--name", type=str, default=DEFAULT_NAME,
                        help=f"zarr basename; writes <out-root>/<name>_<split>.zarr "
                             f"(default {DEFAULT_NAME})")
    parser.add_argument("--fps", type=int, default=porter.DEFAULT_FPS)
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS,
                        help=f"points per frame stored in the zarr (default "
                             f"{DEFAULT_NUM_POINTS}, matching RoboTwin; the encoder "
                             f"farthest-point-samples this down to visual_cond_len). Use "
                             f"4096 for the cluttered-scene ablation.")
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE,
                        help=f"uniform pool before FPS (default {DEFAULT_POOL_SIZE})")
    parser.add_argument("--chunk-frames", type=int, default=DEFAULT_CHUNK_FRAMES)
    parser.add_argument("--state-frame", choices=("world", "base"), default=DEFAULT_STATE_FRAME,
                        help="frame of the agent_pos EEF/head blocks. Default 'world' so "
                             "agent_pos shares the frame of the point cloud and the action; "
                             "'base' reproduces the LeRobot porter's egocentric state.")
    parser.add_argument("--crop-min", type=float, nargs=3, default=list(CROP_MIN),
                        metavar=("X", "Y", "Z"),
                        help=f"world-frame workspace lower bound (default {CROP_MIN})")
    parser.add_argument("--crop-max", type=float, nargs=3, default=list(CROP_MAX),
                        metavar=("X", "Y", "Z"),
                        help=f"world-frame workspace upper bound (default {CROP_MAX})")
    parser.add_argument("--min-depth", type=float, default=MIN_DEPTH_M)
    parser.add_argument("--max-depth", type=float, default=MAX_DEPTH_M)
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the uniform pool subsample (FPS itself is deterministic)")
    parser.add_argument("--include-recovery-data", "--include_recovery_data",
                        dest="include_recovery_data", type=Path, default=None,
                        help="directory of extra recovery episode_<N>.hdf5 takes, APPENDED "
                             "after the raw episodes. Requires <dir>/trim.csv, exactly as "
                             "scripts/port_wbc_mobile_hdf5.py.")
    parser.add_argument("--split-csv", "--split_csv", dest="split_csv", type=Path, default=None,
                        help="CSV assigning episodes to train/val/test (default "
                             "<raw-dir>/split.csv). Each split gets its own zarr.")
    parser.add_argument("--positions-dir", "--positions_dir", dest="positions_dir",
                        type=Path, default=None,
                        help="SceneDiff positions root (<dir>/<source>/episode_<N>.npz, "
                             "source in {raw, recovery}). Enables data/env_state; every "
                             "episode MUST have a valid npz.")
    parser.add_argument("--object-nums", "--object_nums", dest="object_nums", type=int,
                        default=porter.DEFAULT_OBJECT_NUMS,
                        help="conditioned objects per episode; env_state is "
                             "(object_nums*3,) [s1_src, s1_dst, ...] "
                             f"(default {porter.DEFAULT_OBJECT_NUMS})")
    parser.add_argument("--masks-dir", "--masks_dir", dest="masks_dir", type=Path,
                        default=None,
                        help="tracked-mask NPZ root (<dir>/<source>/episode_<N>.npz, "
                             "from scene_diff/scripts/build_wbc_masks.py). Enables "
                             "data/point_mask; every episode MUST have a valid NPZ.")
    parser.add_argument("--dino", action="store_true",
                        help="also write data/env_dino from the positions DINO sidecars "
                             "(episode_<N>_dino.npz; needs --positions-dir)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.num_points < 1:
        raise SystemExit(f"--num-points must be >= 1, got {args.num_points}")
    if args.pool_size < args.num_points:
        raise SystemExit(f"--pool-size {args.pool_size} < --num-points {args.num_points}")
    if args.dino and args.positions_dir is None:
        raise SystemExit("--dino needs --positions-dir (the sidecars live next to the "
                         "positions NPZs)")

    try:
        convert(args)
    except Exception as exc:  # CLI boundary: name the failure, exit 1
        logging.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
