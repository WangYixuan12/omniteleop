#!/usr/bin/env python3
"""Port WBC mobile-manipulation raw HDF5 episodes to MoF training safetensors.

The MoF sibling of ``scripts/port_wbc_mobile_hdf5.py`` (LeRobot) and
``scripts/port_wbc_mobile_zarr.py`` (ManiFlow). All three read the SAME raw takes
(``wbc_vr_robot.py --record``, 10 Hz) and this script REUSES the HDF5 porter's episode
loading, windowing, split and state/action assembly verbatim, so the 29-D ``action``
written here is bit-identical to the LeRobot / ManiFlow datasets and the per-entity
observation poses come from the same ``state_frame="world"`` blocks as ManiFlow's
``agent_pos``.

Per split, one safetensors file per episode plus one shared sidecar::

    <out-root>/<name>/train/episode_<i>.safetensors
    <out-root>/<name>/test/episode_<i>.safetensors
    <out-root>/<name>/meta.json      <- source of truth train + deploy re-read

Episode file schema (T = kept frames)::

    head_image              (T, 224, 224, 3) uint8   anisotropic squash (full FOV kept)
    wrist_image             (T, 224, 224, 3) uint8   left wrist, same squash
    left_ee_pos/quat        (T,3)/(T,4) f32   ENGAGE-ORIGIN WORLD, quat wxyz (pytorch3d)
    right_ee_pos/quat       (T,3)/(T,4) f32
    head_pos/quat           (T,3)/(T,4) f32
    base_pos/base_quat      (T,3)/(T,4) f32   planar odometry pose lifted to 3-D
    proprioception_grippers (T,2) f32          raw FC03 readings (state dims 9/19)
    action                  (T,29) f32         verbatim world targets, COLUMN rot6d
    env_state               (T, object_nums*3) f32   SceneDiff positions, constant

The action keeps the recorded column-major rot6d UNCHANGED (meta.json:
``rotation_convention: column``); the mofpo WBC dataset must NOT apply the BiGym
row->column boundary conversion. Obs quaternions are REAL-FIRST (wxyz), matching
mofpo's pytorch3d RotationTransformer("quaternion").

SceneDiff positions are ALWAYS attached (a position-conditioned run is then a
train-config flip, no re-port); every episode must have its npz -- validated up
front, aborting before anything is written. Splits reuse ``<raw-dir>/split.csv``
and recovery takes keep their trim windows, exactly like the sibling porters.

Environment: ``dexmate_lerobot`` (safetensors + cv2 + pinocchio FK).

Usage:
  python scripts/port_wbc_mobile_mof.py \
    --raw-dir ~/Dexmate/data/box2cloth/raw_data \
    --out-root ~/Dexmate/data/box2cloth/processed_wbc/mof \
    --name dexmate_wbc_mof \
    --include-recovery-data ~/Dexmate/data/box2cloth/raw_data/recovery \
    --positions-dir ~/Dexmate/data/box2cloth/scene_diff/positions
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
import types
from argparse import Namespace
from pathlib import Path

import cv2
import h5py
import numpy as np
from safetensors.numpy import save_file

from omniteleop.wbc_policy_format import ACTION_AXES, STATE_AXES, WBCPolicyFK


def _load_hdf5_porter() -> "types.ModuleType":
    """Import the sibling HDF5 porter by path (``scripts/`` is not a package)."""
    path = Path(__file__).resolve().parent / "port_wbc_mobile_hdf5.py"
    spec = importlib.util.spec_from_file_location("port_wbc_mobile_hdf5", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import the HDF5 porter at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


porter = _load_hdf5_porter()

DEFAULT_RAW_DIR = Path("~/Dexmate/data/box2cloth/raw_data").expanduser()
DEFAULT_OUT_ROOT = Path("~/Dexmate/data/box2cloth/processed_wbc/mof").expanduser()
DEFAULT_POSITIONS_DIR = Path("~/Dexmate/data/box2cloth/scene_diff/positions").expanduser()
DEFAULT_NAME = "dexmate_wbc_mof"
IMAGE_HW = (224, 224)

# The world-frame 32-D state's per-entity blocks (see wbc_policy_format.STATE_AXES).
_ENTITY_BLOCKS = {
    "left_ee": (slice(0, 3), slice(3, 9)),
    "right_ee": (slice(10, 13), slice(13, 19)),
    "head": (slice(20, 23), slice(23, 29)),
}
_GRIPPER_STATE_DIMS = (9, 19)


def _rot6d_columns_to_quat_wxyz(rot6d: np.ndarray) -> np.ndarray:
    """(N, 6) column-major rot6d (two orthonormal columns) -> (N, 4) wxyz quats.

    The state's rot6d blocks are the exact first two columns of a rotation
    matrix (``mat_to_pos6d``), so the third column is their cross product --
    no Gram-Schmidt re-orthogonalisation is applied.
    """
    from scipy.spatial.transform import Rotation

    c1, c2 = rot6d[:, 0:3], rot6d[:, 3:6]
    c3 = np.cross(c1, c2)
    R = np.stack([c1, c2, c3], axis=-1)  # columns
    xyzw = Rotation.from_matrix(R).as_quat()
    return np.concatenate([xyzw[:, 3:4], xyzw[:, 0:3]], axis=1).astype(np.float32)


def _squash_images(frames: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) uint8 -> (T, 224, 224, 3) uint8 anisotropic INTER_AREA resize."""
    out = np.empty((frames.shape[0], *IMAGE_HW, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        out[i] = cv2.resize(frame, (IMAGE_HW[1], IMAGE_HW[0]), interpolation=cv2.INTER_AREA)
    return out


def episode_arrays(
    hdf5_path: Path,
    fk: WBCPolicyFK,
    fps: int,
    *,
    trim: tuple[int, int] | None,
    env_state: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    """One raw take -> the per-episode safetensors dict + episode meta.

    Windowing is identical to the sibling porters: the leading-NaN-gripper drop
    for a raw episode, the absolute inclusive ``trim`` segment for a recovery one.
    """
    ep = porter.load_and_validate_episode(hdf5_path, fps)
    start, end = porter._resolve_window(ep["frame_count"], ep["t0"], trim, hdf5_path)  # noqa: SLF001
    kept = end - start

    states = np.empty((kept, len(STATE_AXES)), dtype=np.float32)
    actions = np.empty((kept, len(ACTION_AXES)), dtype=np.float32)
    with h5py.File(hdf5_path, "r") as raw:
        for i, t in enumerate(range(start, end)):
            action_eef, action_head = porter.load_action_targets_world(raw, t)
            states[i], actions[i] = porter.compute_frame_state_action(
                raw, t, fk, action_eef, action_head, state_frame="world"
            )

    arrays: dict[str, np.ndarray] = {
        "head_image": _squash_images(ep["rgb"][start:end]),
        "wrist_image": _squash_images(ep["wrist_rgb"][start:end]),
        "action": actions,
        "proprioception_grippers": states[:, list(_GRIPPER_STATE_DIMS)],
        "env_state": np.repeat(env_state[None].astype(np.float32), kept, axis=0),
    }
    for name, (pos_sl, rot_sl) in _ENTITY_BLOCKS.items():
        arrays[f"{name}_pos"] = states[:, pos_sl]
        arrays[f"{name}_quat"] = _rot6d_columns_to_quat_wxyz(states[:, rot_sl])
    # Planar odometry base pose lifted to a 3-D world pose (z = 0, yaw about z).
    x, y, yaw = states[:, 29], states[:, 30], states[:, 31]
    arrays["base_pos"] = np.stack([x, y, np.zeros_like(x)], axis=1).astype(np.float32)
    half = (yaw / 2.0).astype(np.float64)
    arrays["base_quat"] = np.stack(
        [np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)], axis=1
    ).astype(np.float32)

    meta = {
        "num_frames_before": int(ep["frame_count"]),
        "num_frames_after": int(kept),
        "window": [int(start), int(end)],
        "source_image_hw": [int(ep["rgb"].shape[1]), int(ep["rgb"].shape[2])],
    }
    return arrays, meta


def _git_sha(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def build_args(**overrides: object) -> Namespace:
    """Defaults mirroring the CLI; tests override paths."""
    args = Namespace(
        raw_dir=DEFAULT_RAW_DIR,
        out_root=DEFAULT_OUT_ROOT,
        name=DEFAULT_NAME,
        fps=porter.DEFAULT_FPS,
        include_recovery_data=None,
        split_csv=None,
        positions_dir=DEFAULT_POSITIONS_DIR,
        object_nums=porter.DEFAULT_OBJECT_NUMS,
        overwrite=False,
    )
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise TypeError(f"unknown porter arg {key!r}")
        setattr(args, key, value)
    return args


def convert(args: Namespace) -> None:
    work = porter.build_work_list(args.raw_dir, args.include_recovery_data)

    explicit_split = args.split_csv is not None
    split_csv = args.split_csv if explicit_split else Path(args.raw_dir) / "split.csv"
    if split_csv.exists():
        split_map = porter.parse_split_csv(split_csv)
        logging.info("Using split csv %s (%d listed episode(s))", split_csv, len(split_map))
    elif explicit_split:
        raise RuntimeError(f"--split-csv not found: {split_csv}")
    else:
        logging.info("No split csv at %s; all episodes -> train", split_csv)
        split_map = {}
    splits = porter.assign_splits(work, split_map, args.include_recovery_data)

    # env_state is MANDATORY for this port: validate every episode up front and
    # abort before anything is written (a conditioned run later is config-only).
    env_state_by_key = porter.validate_all_episode_positions(
        args.positions_dir, work, args.object_nums
    )

    n_recovery = sum(item["source"] == "recovery" for item in work)
    logging.info(
        "Port plan: %d episode(s) = %d raw + %d recovery -> {%s}",
        len(work),
        len(work) - n_recovery,
        n_recovery,
        ", ".join(f"{s}:{len(v)}" for s, v in splits.items()),
    )

    out_dir = Path(args.out_root) / args.name
    if out_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"{out_dir} already exists. Use --overwrite to replace it.")
        logging.info("Removing existing %s (--overwrite)", out_dir)
        shutil.rmtree(out_dir)

    fk = WBCPolicyFK()
    episode_meta: list[dict] = []
    tensor_schema: dict[str, dict] | None = None
    total_frames = 0
    for split, items in splits.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for processed_index, item in enumerate(items):
            key = (item["source"], item["raw_index"])
            arrays, ep_meta = episode_arrays(
                item["path"], fk, args.fps, trim=item["trim"], env_state=env_state_by_key[key]
            )
            if tensor_schema is None:
                tensor_schema = {
                    k: {"dtype": str(v.dtype), "frame_shape": list(v.shape[1:])}
                    for k, v in sorted(arrays.items())
                }
            file_name = f"episode_{processed_index}.safetensors"
            save_file(arrays, str(split_dir / file_name))
            total_frames += ep_meta["num_frames_after"]
            episode_meta.append(
                {
                    "split": split,
                    "source": item["source"],
                    "raw_index": item["raw_index"],
                    "processed_index": processed_index,
                    "file": f"{split}/{file_name}",
                    **ep_meta,
                }
            )
            logging.info(
                "  %s/%s: %d frames (of %d) from %s",
                split,
                file_name,
                ep_meta["num_frames_after"],
                ep_meta["num_frames_before"],
                item["path"].name,
            )

    meta = {
        "schema": "wbc_mof_safetensors_v1",
        # Authoritative per-episode tensor contract: key -> dtype + per-frame
        # shape (all tensors are (T, *frame_shape)). Consumers validate against
        # this instead of guessing.
        "tensors": tensor_schema,
        "fps": args.fps,
        "image_resolution": list(IMAGE_HW),
        "image_resize": "anisotropic",
        "image_interpolation": "cv2.INTER_AREA",
        "cameras": {"head_image": "head_left_rgb", "wrist_image": "left_wrist_rgb"},
        "rotation_convention": "column",
        "quaternion_convention": "wxyz",
        "state_frame": "world",
        "obs_frame": "world (engage-origin; base = planar odometry pose lifted to 3-D)",
        "action_axes": list(ACTION_AXES),
        "action_frame": "world (verbatim ik.solve targets, engage-origin)",
        "env_state": {
            "object_nums": args.object_nums,
            "axes": porter.env_state_axes(args.object_nums),
            "frame": "world (engage-origin, same as action)",
            "positions_dir": str(args.positions_dir),
            "constant_per_episode": True,
        },
        "episodes": episode_meta,
        "total_frames": total_frames,
        "omniteleop_git_sha": _git_sha(Path(__file__).resolve().parents[1]),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logging.info("Wrote %s: %d episode(s), %d frames", out_dir, len(episode_meta), total_frames)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_NAME,
        help=f"dataset dir name under --out-root (default {DEFAULT_NAME})",
    )
    parser.add_argument("--fps", type=int, default=porter.DEFAULT_FPS)
    parser.add_argument(
        "--include-recovery-data",
        "--include_recovery_data",
        dest="include_recovery_data",
        type=Path,
        default=None,
        help="directory of recovery episode_<N>.hdf5 takes (requires "
        "<dir>/trim.csv), exactly as the sibling porters",
    )
    parser.add_argument(
        "--split-csv",
        "--split_csv",
        dest="split_csv",
        type=Path,
        default=None,
        help="default <raw-dir>/split.csv",
    )
    parser.add_argument(
        "--positions-dir",
        "--positions_dir",
        dest="positions_dir",
        type=Path,
        default=DEFAULT_POSITIONS_DIR,
        help="SceneDiff positions root; REQUIRED to exist -- env_state "
        "is always attached (training decides whether to use it)",
    )
    parser.add_argument(
        "--object-nums",
        "--object_nums",
        dest="object_nums",
        type=int,
        default=porter.DEFAULT_OBJECT_NUMS,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        convert(args)
    except Exception as exc:  # CLI boundary: name the failure, exit 1
        logging.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
