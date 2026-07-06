#!/usr/bin/env python3
"""Port WBC mobile-manipulation raw HDF5 episodes to a LeRobotDataset.

Consumes the raw takes written by ``scripts/wbc_vr_robot.py --record`` (10 Hz,
EpisodeRecorder schema, flat ``<raw-dir>/episode_*.hdf5``) and emits the WBC
policy schema of ``omniteleop.wbc_policy_format`` (see PLAN.md Conventions):

  observation.state (32,) float32
      base-frame ACHIEVED left/right EEF pos3+rot6 (+ raw FC03 gripper) and
      zed_depth_frame head pos3+rot6, from WBC FK on the MEASURED ``obs/joint``
      with the base at zero -- computed on the SAME model the follower solves
      with (``WBCPolicyFK``; never a different URDF / KinHelper) -- then the
      measured odometry base pose ``obs/base/pose`` (x, y, yaw, engage-origin
      world): the policy's only absolute world anchor.
  action (29,) float32
      the WORLD-frame targets given VERBATIM to ``ik.solve()`` at the record
      tick, read straight from ``action/eef/{left,right}`` / ``action/head``
      (+ binarized commanded gripper). NO FK, NO base composition: offline
      action values are bit-identical to the online solver inputs.

Required raw datasets (RuntimeError if absent -- re-record with the updated
recorder; the porter never substitutes ``episode_*_debug.hdf5`` data):
``action/eef/{left,right}`` (T,4,4), ``action/head`` (T,4,4),
``obs/base/pose`` (T,3), plus the usual ``obs/joint``, ``obs/gripper``,
``action/gripper``, and ``obs/images/{head_left_rgb,head_depth,left_wrist_rgb,
intrinsic}`` (intrinsic is STATIC (3,3) in this recorder).

Gripper policy: ``obs/gripper`` is NaN until the first FC03 reply, so leading
frames with a NaN gripper are DROPPED; a NaN after the first finite value is
data corruption and raises. Action grippers are binarized at 0.5 like the
tabletop porter.

Sidecars per episode (not in the parquet):
``debug/depth/episode_XXXXXX.npz``  key ``depth``   (T', resize_h, resize_w) uint16
``debug/calib/episode_XXXXXX.npz``  keys ``extrinsic`` (world_T_zed, from
    world_T_base(obs/base/pose) @ base_T_zed), ``base_extrinsic`` (base_T_zed
    from the state FK), ``intrinsic`` (static intrinsic rescaled to the output
    size, broadcast (T',3,3)).

Episode ordering & splits: raw ``episode_<N>.hdf5`` are ported in NUMERIC index
order (the recorder writes non-zero-padded names, so a lexicographic sort is
wrong). ``--split-csv`` (default ``<raw-dir>/split.csv``) assigns episodes to
train/val/test; each split is written to its OWN ``<root>/<split>/<repo_id>/``
dataset. Rows are ``episode, split`` where ``episode`` is ``<N>`` (raw) or
``recovery/<N>``; every episode NOT listed defaults to ``train``. If the DEFAULT
split.csv is absent every episode goes to ``train``; an explicitly-passed missing
``--split-csv`` errors. Each split dataset is re-opened and cross-checked by
``verify()`` right after it is written.

Recovery episodes (``--include_recovery_data <dir>``, default off): extra takes
in ``<dir>/episode_<N>.hdf5`` are APPENDED after the raw episodes (within their
split), each trimmed to a segment named in the REQUIRED ``<dir>/trim.csv`` (one
header row, then ``episode, frame_start, frame_end`` per row; ABSOLUTE INCLUSIVE
raw-frame indices, ``frame_end == -1`` -> last frame). Every recovery file must
have exactly one trim row and vice-versa (else RuntimeError). The recovery dir is
only READ, never modified.

Index log (``--log_index <csv>``, default off): one row per emitted episode --
``split``, ``source`` (raw/recovery), ``raw_data_index`` (the N in
``episode_<N>.hdf5``), ``processed_data_index`` (0-based LeRobot episode index
WITHIN its split), ``num_frames_before`` (raw take length), ``num_frames_after``
(frames actually written after the leading-NaN drop or the trim).

Environments: the unit tests need only the dexmate env (pinocchio/pink); the
actual conversion + ``verify()`` additionally need ``lerobot`` and ``tqdm``
(``dexmate_lerobot`` env) -- both imports are lazy.

Usage:
  python scripts/port_wbc_mobile_hdf5.py \
    --raw-dir ~/Dexmate/data/raw_data --root ~/Dexmate/data/processed_wbc \
    --fps 10 --resize-h 120 --resize-w 160 --repo-id dexmate_wbc_eef_head \
    --split-csv ~/Dexmate/data/raw_data/split.csv \
    --include_recovery_data ~/Dexmate/data/raw_data/recovery \
    --log_index ~/Dexmate/data/raw_data/log_index.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

import cv2
import h5py
import numpy as np

from omniteleop.wbc_policy_format import (
    ACTION_AXES,
    GRIPPER_DIMS,
    RELATIVE_EXCLUDE_DIMS,
    SKIP_NORMALIZATION_DIMS,
    STATE_AXES,
    STATE_FRAMES,
    WBCPolicyFK,
    base_pose_to_mat,
    build_state_vector,
    mat_to_pos6d,
    pos6d_to_mat,
)

# Silence the per-encode libsvtav1 banner (matches the tabletop porter).
os.environ.setdefault("SVT_LOG", "1")

DEFAULT_RAW_DIR = Path("~/Dexmate/data/raw_data").expanduser()
DEFAULT_ROOT = Path("~/Dexmate/data/processed_wbc").expanduser()
DEFAULT_FPS = 10
DEFAULT_RESIZE_H = 120
DEFAULT_RESIZE_W = 160
DEFAULT_TASK = "dexmate_wbc"
DEFAULT_REPO_ID = "dexmate_wbc_eef_head"
GRIPPER_BINARY_THRESHOLD = 0.5
ROBOT_TYPE = "dexmate_vega_wbc"
# Splits are written to <root>/<split>/<repo_id>/ (tabletop porter layout).
ALL_SPLITS = ("train", "val", "test")
DEFAULT_SPLIT = "train"  # episodes not named in split.csv land here
# Rotation-column sanity for RECORDED action targets (float32 leader poses).
_ROT_COL_NORM_TOL = 1e-2

_REQUIRED_ACTION_TARGETS = ("action/eef/left", "action/eef/right", "action/head")

# Strict ``episode_<int>.hdf5`` name; excludes any ``episode_*_debug.hdf5`` sidecar.
_EPISODE_RE = re.compile(r"^episode_(\d+)\.hdf5$")


def episode_index_from_path(path: Path) -> int:
    """Integer N parsed from an ``episode_<N>.hdf5`` filename (for numeric sort)."""
    match = _EPISODE_RE.match(Path(path).name)
    if match is None:
        raise RuntimeError(
            f"episode filename does not match episode_<int>.hdf5: {Path(path).name}"
        )
    return int(match.group(1))


def find_episode_files(raw_dir: Path) -> list[Path]:
    """``episode_<N>.hdf5`` in ``raw_dir``, sorted by the NUMERIC index N.

    The recorder writes non-zero-padded names (``episode_1``..``episode_61``), so a
    plain lexicographic sort is wrong (``episode_11`` before ``episode_2``); we sort by
    the parsed integer. Names that are not ``episode_<int>.hdf5`` (e.g. any
    ``episode_*_debug.hdf5``) are ignored, never ported.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise RuntimeError(f"raw dir not found: {raw_dir}")
    episode_files = sorted(
        (p for p in raw_dir.glob("episode_*.hdf5") if _EPISODE_RE.match(p.name)),
        key=episode_index_from_path,
    )
    if not episode_files:
        raise RuntimeError(f"no episode_*.hdf5 found in {raw_dir}")
    return episode_files


def load_trim_spec(recovery_dir: Path) -> dict[int, tuple[int, int]]:
    """Parse ``<recovery_dir>/trim.csv`` into ``{episode_index: (frame_start, frame_end)}``.

    Format (one header row, then one row per episode)::

        episode, frame_start, frame_end
        25, 409, -1

    ``frame_start`` / ``frame_end`` are ABSOLUTE, INCLUSIVE raw-frame indices into
    ``episode_<idx>.hdf5``; ``frame_end == -1`` means "through the last frame". Raises
    on a missing file, a malformed row, a non-integer field, a negative index, an
    inverted range, or a duplicate episode. The file is only READ, never modified.
    """
    trim_path = Path(recovery_dir) / "trim.csv"
    if not trim_path.exists():
        raise RuntimeError(
            f"--include_recovery_data requires the trim file {trim_path} (not found)"
        )
    spec: dict[int, tuple[int, int]] = {}
    first_row = True
    with open(trim_path, newline="") as handle:
        for lineno, row in enumerate(csv.reader(handle), start=1):
            cells = [c.strip() for c in row]
            if not cells or all(c == "" for c in cells):
                continue  # blank line
            if len(cells) != 3:
                raise RuntimeError(
                    f"{trim_path}:{lineno}: expected 3 columns "
                    f"'episode, frame_start, frame_end', got {len(cells)}: {row}"
                )
            if first_row and not cells[0].lstrip("-").isdigit():
                first_row = False
                continue  # header row (e.g. 'episode, frame_start, frame_end')
            first_row = False
            try:
                episode, frame_start, frame_end = (int(c) for c in cells)
            except ValueError as exc:
                raise RuntimeError(f"{trim_path}:{lineno}: non-integer field in {row}") from exc
            if episode < 0:
                raise RuntimeError(f"{trim_path}:{lineno}: negative episode index {episode}")
            if frame_start < 0:
                raise RuntimeError(
                    f"{trim_path}:{lineno}: frame_start must be >= 0, got {frame_start}"
                )
            if frame_end < -1:
                raise RuntimeError(
                    f"{trim_path}:{lineno}: frame_end must be >= 0 or -1, got {frame_end}"
                )
            if frame_end != -1 and frame_end < frame_start:
                raise RuntimeError(
                    f"{trim_path}:{lineno}: frame_end {frame_end} < frame_start {frame_start}"
                )
            if episode in spec:
                raise RuntimeError(f"{trim_path}:{lineno}: duplicate episode {episode}")
            spec[episode] = (frame_start, frame_end)
    if not spec:
        raise RuntimeError(f"{trim_path}: no trim rows found")
    return spec


def _resolve_window(frame_count: int, t0: int, trim: tuple[int, int] | None,
                    source: str | Path) -> tuple[int, int]:
    """The half-open ``[start, end)`` frame window an episode contributes.

    ``trim is None`` (raw episode): ``[t0, frame_count)`` -- the porter's usual
    leading-NaN-gripper drop. ``trim=(frame_start, frame_end)`` (recovery episode):
    the ABSOLUTE, INCLUSIVE ``[frame_start, frame_end]`` (``frame_end == -1`` -> last
    frame), independent of ``t0``; a NaN obs gripper inside the window is caught
    per-frame downstream (``compute_frame_state_action``) and raises there.
    """
    if trim is None:
        return t0, frame_count
    frame_start, frame_end = trim
    end_incl = frame_count - 1 if frame_end == -1 else frame_end
    if not 0 <= frame_start <= end_incl <= frame_count - 1:
        raise RuntimeError(
            f"{source}: trim [{frame_start}, {frame_end}] out of range for T={frame_count} "
            f"(resolved last frame {end_incl}; need 0 <= start <= end <= T-1)"
        )
    return frame_start, end_incl + 1


def _validate_finite(name: str, arr: np.ndarray, source: str | Path,
                     *, check_finite: bool = True) -> None:
    if arr.size == 0:
        raise RuntimeError(f"{source}: {name} is empty (shape={arr.shape})")
    if arr.dtype.kind not in "biufc":
        raise RuntimeError(f"{source}: {name} has non-numeric dtype {arr.dtype}")
    if check_finite and arr.dtype.kind in "fc" and not np.all(np.isfinite(arr)):
        idx = np.argwhere(~np.isfinite(arr))[0]
        raise RuntimeError(
            f"{source}: {name} is non-finite at index {idx.tolist()} "
            f"(shape={arr.shape}, dtype={arr.dtype})"
        )


def read_required_array(f: h5py.File, key: str, source: str | Path,
                        *, check_finite: bool = True) -> np.ndarray:
    """Load a required dataset; RuntimeError names the missing path (PLAN.md)."""
    if key not in f:
        raise RuntimeError(
            f"{source}: missing required HDF5 dataset {key} -- re-record with the "
            "updated wbc_vr_robot.py recorder (debug files are never a substitute)"
        )
    node = f[key]
    if not isinstance(node, h5py.Dataset):
        raise RuntimeError(f"{source}: {key} is not an HDF5 dataset")
    arr = np.asarray(node[()])
    _validate_finite(key, arr, source, check_finite=check_finite)
    return arr


def _validate_target_batch(name: str, mats: np.ndarray, source: str | Path) -> None:
    """(T,4,4) finite pose batch whose rotation columns are unit within tolerance."""
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise RuntimeError(f"{source}: {name} must be (T,4,4), got {mats.shape}")
    _validate_finite(name, mats, source)
    for col in (0, 1):
        norms = np.linalg.norm(mats[:, :3, col], axis=1)
        bad = np.abs(norms - 1.0) > _ROT_COL_NORM_TOL
        if bad.any():
            t = int(np.argwhere(bad)[0][0])
            raise RuntimeError(
                f"{source}: {name} rotation column {col} is not unit at frame {t} "
                f"(norm={norms[t]:.6f}) -- corrupt target"
            )


def first_valid_frame(gripper_left: np.ndarray, gripper_right: np.ndarray,
                      source: str | Path) -> int:
    """First index where BOTH obs grippers are finite; leading NaNs are dropped.

    ``obs/gripper`` is NaN until the FC03 monitor's first reply. A NaN AFTER the
    first finite value cannot happen in a healthy take (the recorder holds the
    last reply), so it raises instead of being repaired.
    """
    finite = np.isfinite(gripper_left) & np.isfinite(gripper_right)
    if not finite.any():
        raise RuntimeError(f"{source}: obs/gripper never becomes finite (no FC03 reply)")
    t0 = int(np.argmax(finite))
    for name, g in (("obs/gripper/left", gripper_left), ("obs/gripper/right", gripper_right)):
        later = ~np.isfinite(g[t0:])
        if later.any():
            t_bad = t0 + int(np.argmax(later))
            raise RuntimeError(
                f"{source}: {name} is NaN at frame {t_bad}, after the first finite "
                f"frame {t0} -- corrupt take (mid-take NaN is never interpolated)"
            )
    return t0


def load_action_targets_world(raw: h5py.File, t: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """The frame-``t`` WORLD-frame solver targets, VERBATIM from the raw file.

    Returns ``({"left": (4,4), "right": (4,4)}, head (4,4))``. RuntimeError on a
    missing dataset (named), a bad frame index, a wrong shape, or a non-finite /
    non-rotation transform.
    """
    source = raw.filename
    for key in _REQUIRED_ACTION_TARGETS:
        if key not in raw:
            raise RuntimeError(
                f"{source}: missing required HDF5 dataset {key} -- re-record with the "
                "updated wbc_vr_robot.py recorder (debug files are never a substitute)"
            )
    out: dict[str, np.ndarray] = {}
    for name, key in (("left", "action/eef/left"), ("right", "action/eef/right"),
                      ("head", "action/head")):
        ds = raw[key]
        if ds.ndim != 3 or ds.shape[1:] != (4, 4):
            raise RuntimeError(f"{source}: {key} must be (T,4,4), got {ds.shape}")
        if not 0 <= t < ds.shape[0]:
            raise RuntimeError(f"{source}: frame index {t} out of range for {key} "
                               f"(T={ds.shape[0]})")
        mat = np.asarray(ds[t], dtype=np.float64)
        _validate_target_batch(key, mat[None], source)
        out[name] = mat
    return {"left": out["left"], "right": out["right"]}, out["head"]


def compute_frame_state_action(
    raw: h5py.File,
    t: int,
    fk: WBCPolicyFK,
    action_eef: Mapping[str, np.ndarray],
    action_head: np.ndarray,
    *,
    state_frame: str = "base",
) -> tuple[np.ndarray, np.ndarray]:
    """One frame's ``(observation.state (32,), action (29,))`` float32 vectors.

    State: achieved FK poses of the measured ``obs/joint`` (base zero in q), in the
    ``state_frame`` frame (``"base"`` = egocentric, absolute-action default;
    ``"world"`` = composed to the engage-origin world via ``obs/base/pose``, REQUIRED
    for use_relative_actions so the state shares the world action frame), + raw obs
    grippers + ``obs/base/pose`` (dims 29-31, world anchor, same in both frames).
    Action: ``mat_to_pos6d`` of the given world-frame targets (verbatim from raw) +
    binarized action grippers.
    """
    source = raw.filename

    def at(key: str) -> np.ndarray:
        arr = np.asarray(raw[key][t], dtype=np.float64) if key in raw else None
        if arr is None:
            raise RuntimeError(f"{source}: missing required HDF5 dataset {key}")
        _validate_finite(f"{key}[{t}]", arr, source)
        return arr

    poses = fk.base_frame_poses(
        at("obs/joint/torso"), at("obs/joint/left_arm"),
        at("obs/joint/right_arm"), at("obs/joint/head"),
    )
    base_pose = at("obs/base/pose")
    if base_pose.shape != (3,):
        raise RuntimeError(f"{source}: obs/base/pose[{t}] must be (3,), got {base_pose.shape}")
    grip_obs_l, grip_obs_r = float(at("obs/gripper/left")), float(at("obs/gripper/right"))
    grip_act_l = float(float(at("action/gripper/left")) >= GRIPPER_BINARY_THRESHOLD)
    grip_act_r = float(float(at("action/gripper/right")) >= GRIPPER_BINARY_THRESHOLD)

    state = build_state_vector(poses, base_pose, grip_obs_l, grip_obs_r, state_frame=state_frame)
    action = np.concatenate([
        mat_to_pos6d(action_eef["left"]), [grip_act_l],
        mat_to_pos6d(action_eef["right"]), [grip_act_r],
        mat_to_pos6d(action_head),
    ]).astype(np.float32)
    if action.shape != (len(ACTION_AXES),):
        raise RuntimeError(
            f"{source}: frame {t} built action {action.shape}, "
            f"expected ({len(ACTION_AXES)},)"
        )
    return state, action


def load_and_validate_episode(hdf5_path: Path, fps: int) -> dict:
    """Read + cross-validate every array the port needs; shared with verify().

    Returns arrays plus ``t0`` (first frame with finite obs grippers). All frame
    counts must agree; the required policy datasets must exist; the record
    cadence must match ``fps`` at the median.
    """
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        raise RuntimeError(f"episode file not found: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        rgb = read_required_array(f, "obs/images/head_left_rgb", hdf5_path)
        wrist_rgb = read_required_array(f, "obs/images/left_wrist_rgb", hdf5_path)
        depth = read_required_array(f, "obs/images/head_depth", hdf5_path)
        intrinsic = read_required_array(f, "obs/images/intrinsic", hdf5_path)
        joints_obs = {
            grp: read_required_array(f, f"obs/joint/{grp}", hdf5_path)
            for grp in ("torso", "left_arm", "right_arm", "head")
        }
        base_pose = read_required_array(f, "obs/base/pose", hdf5_path)
        gripper_obs = {
            side: read_required_array(f, f"obs/gripper/{side}", hdf5_path,
                                      check_finite=False)
            for side in ("left", "right")
        }
        gripper_act = {
            side: read_required_array(f, f"action/gripper/{side}", hdf5_path)
            for side in ("left", "right")
        }
        eef_left = read_required_array(f, "action/eef/left", hdf5_path)
        eef_right = read_required_array(f, "action/eef/right", hdf5_path)
        head_target = read_required_array(f, "action/head", hdf5_path)
        timestamp_ns = read_required_array(f, "timestamp_ns", hdf5_path)

    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise RuntimeError(f"{hdf5_path}: expected (T,H,W,3) uint8 head RGB, got "
                           f"{rgb.shape} {rgb.dtype}")
    if wrist_rgb.ndim != 4 or wrist_rgb.shape[-1] != 3 or wrist_rgb.dtype != np.uint8:
        raise RuntimeError(f"{hdf5_path}: expected (T,H,W,3) uint8 wrist RGB, got "
                           f"{wrist_rgb.shape} {wrist_rgb.dtype}")
    if depth.ndim != 3 or depth.dtype != np.uint16:
        raise RuntimeError(f"{hdf5_path}: expected (T,H,W) uint16 depth, got "
                           f"{depth.shape} {depth.dtype}")
    if depth.shape[1:] != rgb.shape[1:3]:
        raise RuntimeError(f"{hdf5_path}: depth/rgb spatial mismatch: "
                           f"{depth.shape[1:]} vs {rgb.shape[1:3]}")
    # This recorder stores the head intrinsic ONCE (static (3,3)), not per frame.
    if intrinsic.shape != (3, 3):
        raise RuntimeError(
            f"{hdf5_path}: expected STATIC (3,3) obs/images/intrinsic (wbc_vr_robot "
            f"recorder), got {intrinsic.shape}"
        )

    frame_count = rgb.shape[0]
    checks: list[tuple[str, np.ndarray]] = [
        ("obs/images/left_wrist_rgb", wrist_rgb), ("obs/images/head_depth", depth),
        ("obs/base/pose", base_pose), ("timestamp_ns", timestamp_ns),
        ("action/eef/left", eef_left), ("action/eef/right", eef_right),
        ("action/head", head_target),
    ]
    checks += [(f"obs/joint/{g}", a) for g, a in joints_obs.items()]
    checks += [(f"obs/gripper/{s}", a) for s, a in gripper_obs.items()]
    checks += [(f"action/gripper/{s}", a) for s, a in gripper_act.items()]
    for name, arr in checks:
        if arr.shape[0] != frame_count:
            raise RuntimeError(
                f"{hdf5_path}: frame counts disagree: head_left_rgb={frame_count}, "
                f"{name}={arr.shape[0]}"
            )
    expected_joint_dims = {"torso": 3, "left_arm": 7, "right_arm": 7, "head": 3}
    for grp, dim in expected_joint_dims.items():
        if joints_obs[grp].ndim != 2 or joints_obs[grp].shape[1] != dim:
            raise RuntimeError(f"{hdf5_path}: obs/joint/{grp} must be (T,{dim}), got "
                               f"{joints_obs[grp].shape}")
    if base_pose.ndim != 2 or base_pose.shape[1] != 3:
        raise RuntimeError(f"{hdf5_path}: obs/base/pose must be (T,3), got {base_pose.shape}")
    for name, arr in (("action/eef/left", eef_left), ("action/eef/right", eef_right),
                      ("action/head", head_target)):
        _validate_target_batch(name, arr, hdf5_path)

    # Record cadence: the recorder throttles to --record-rate (= dataset fps), but
    # hold windows leave gaps. The median tick must match fps; gaps are warned.
    ts = timestamp_ns.astype(np.int64)
    if frame_count > 1:
        dt = np.diff(ts) / 1e9
        if np.any(dt <= 0):
            raise RuntimeError(f"{hdf5_path}: timestamp_ns is not strictly increasing")
        median_dt = float(np.median(dt))
        if abs(median_dt - 1.0 / fps) > 0.15 / fps:
            raise RuntimeError(
                f"{hdf5_path}: median frame dt {median_dt * 1000:.1f} ms does not match "
                f"--fps {fps} ({1000.0 / fps:.1f} ms); re-record or fix --fps"
            )
        gaps = int(np.sum(dt > 1.5 / fps))
        if gaps:
            logging.warning("%s: %d hold gap(s) > %.0f ms in the 10 Hz stream",
                            hdf5_path.name, gaps, 1500.0 / fps)

    t0 = first_valid_frame(gripper_obs["left"], gripper_obs["right"], hdf5_path)
    if t0 > 0:
        logging.info("%s: dropping %d leading frame(s) with NaN obs/gripper",
                     hdf5_path.name, t0)

    return {
        "rgb": rgb, "wrist_rgb": wrist_rgb, "depth": depth, "intrinsic": intrinsic,
        "base_pose": base_pose, "frame_count": frame_count, "t0": t0,
    }


def build_features(resize_h: int, resize_w: int) -> dict:
    return {
        "observation.images.head_rgb": {
            "dtype": "video",
            "shape": (resize_h, resize_w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist_rgb": {
            "dtype": "video",
            "shape": (resize_h, resize_w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_AXES),),
            "names": {"axes": list(STATE_AXES)},
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_AXES),),
            "names": {"axes": list(ACTION_AXES)},
        },
    }


def _sidecar_path(variant_root: Path, kind: str, episode_idx: int) -> Path:
    return Path(variant_root) / "debug" / kind / f"episode_{episode_idx:06d}.npz"


def _scaled_intrinsic(intrinsic: np.ndarray, raw_hw: tuple[int, int],
                      out_hw: tuple[int, int]) -> np.ndarray:
    """Static (3,3) intrinsic rescaled from the raw frame size to the output size."""
    k = np.asarray(intrinsic, dtype=np.float32).copy()
    k[0, :] *= out_hw[1] / raw_hw[1]   # fx, 0, cx scale with width
    k[1, :] *= out_hw[0] / raw_hw[0]   # 0, fy, cy scale with height
    return k


def add_episode(dataset, hdf5_path: Path, fk: WBCPolicyFK, resize_h: int,
                resize_w: int, task: str, variant_root: Path, episode_idx: int,
                fps: int, state_frame: str = "base",
                *, trim: tuple[int, int] | None = None) -> tuple[int, int]:
    """Port one raw episode; returns ``(num_frames_before, num_frames_after)``.

    ``num_frames_before`` is the raw take length; ``num_frames_after`` is the number
    of frames written -- after the leading-NaN-gripper drop (``trim is None``) or after
    the ABSOLUTE INCLUSIVE ``trim=(frame_start, frame_end)`` recovery segment.
    """
    if not task:
        raise ValueError("task must be a non-empty string")
    ep = load_and_validate_episode(hdf5_path, fps)
    rgb, wrist_rgb, depth = ep["rgb"], ep["wrist_rgb"], ep["depth"]
    frame_count = ep["frame_count"]
    start, end = _resolve_window(frame_count, ep["t0"], trim, hdf5_path)
    kept = end - start

    depth_resized = np.empty((kept, resize_h, resize_w), dtype=np.uint16)
    extrinsic = np.empty((kept, 4, 4), dtype=np.float32)
    base_extrinsic = np.empty((kept, 4, 4), dtype=np.float32)

    with h5py.File(hdf5_path, "r") as raw:
        for i, t in enumerate(range(start, end)):
            action_eef, action_head = load_action_targets_world(raw, t)
            state, action = compute_frame_state_action(
                raw, t, fk, action_eef, action_head, state_frame=state_frame
            )

            # Calib sidecar: recover base_T_zed and world_T_zed from the state head
            # block, which is base_T_zed in "base" frame and world_T_zed in "world"
            # frame -- so derive both via the odometry base pose regardless of frame.
            head_mat = pos6d_to_mat(state[20:29])
            world_T_base = base_pose_to_mat(state[29:32])
            if state_frame == "world":
                world_t_zed = head_mat
                base_t_zed = np.linalg.inv(world_T_base) @ head_mat
            else:
                base_t_zed = head_mat
                world_t_zed = world_T_base @ head_mat
            base_extrinsic[i] = base_t_zed.astype(np.float32)
            extrinsic[i] = world_t_zed.astype(np.float32)

            rgb_resized = cv2.resize(rgb[t], (resize_w, resize_h),
                                     interpolation=cv2.INTER_AREA)
            wrist_resized = cv2.resize(wrist_rgb[t], (resize_w, resize_h),
                                       interpolation=cv2.INTER_AREA)
            # INTER_NEAREST keeps uint16 millimeter depth exact across edges.
            depth_resized[i] = cv2.resize(depth[t], (resize_w, resize_h),
                                          interpolation=cv2.INTER_NEAREST)

            dataset.add_frame({
                "observation.images.head_rgb": rgb_resized,
                "observation.images.wrist_rgb": wrist_resized,
                "observation.state": state,
                "action": action,
                "task": task,
            })

    dataset.save_episode()

    depth_path = _sidecar_path(variant_root, "depth", episode_idx)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(depth_path, depth=depth_resized)

    intrinsic_resized = _scaled_intrinsic(
        ep["intrinsic"], rgb.shape[1:3], (resize_h, resize_w)
    )
    calib_path = _sidecar_path(variant_root, "calib", episode_idx)
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        calib_path,
        extrinsic=extrinsic,
        base_extrinsic=base_extrinsic,
        intrinsic=np.broadcast_to(intrinsic_resized, (kept, 3, 3)).copy(),
    )
    logging.info("Ported %s: %d frames [%d,%d) of %d, sidecars %s / %s",
                 hdf5_path.name, kept, start, end, frame_count,
                 depth_path.name, calib_path.name)
    return frame_count, kept


def build_work_list(raw_dir: Path, include_recovery_data: Path | None) -> list[dict]:
    """Ordered port plan: raw episodes (numeric-sorted), then -- if requested -- the
    trimmed recovery episodes (numeric-sorted, APPENDED last).

    Each item is ``{"source": "raw"|"recovery", "path": Path, "raw_index": int,
    "trim": (frame_start, frame_end) | None}``; ``processed_data_index`` is the item's
    position in the returned list. Enforces a strict 1:1 match between the recovery
    ``episode_<N>.hdf5`` files and the ``trim.csv`` rows.
    """
    work = [
        {"source": "raw", "path": p, "raw_index": episode_index_from_path(p), "trim": None}
        for p in find_episode_files(raw_dir)
    ]
    if include_recovery_data is None:
        return work

    recovery_dir = Path(include_recovery_data)
    if not recovery_dir.is_dir():
        raise RuntimeError(f"--include_recovery_data dir not found: {recovery_dir}")
    if recovery_dir.resolve() == Path(raw_dir).resolve():
        raise RuntimeError(
            f"--include_recovery_data must differ from --raw-dir (both {recovery_dir})"
        )
    trim_spec = load_trim_spec(recovery_dir)
    recovery_files = find_episode_files(recovery_dir)
    file_indices = {episode_index_from_path(p) for p in recovery_files}
    missing_rows = sorted(file_indices - set(trim_spec))
    missing_files = sorted(set(trim_spec) - file_indices)
    if missing_rows or missing_files:
        raise RuntimeError(
            f"{recovery_dir / 'trim.csv'} does not match the recovery episodes 1:1: "
            f"episode files without a trim row={missing_rows}, "
            f"trim rows without a file={missing_files}"
        )
    for p in recovery_files:
        idx = episode_index_from_path(p)
        work.append({"source": "recovery", "path": p, "raw_index": idx,
                     "trim": trim_spec[idx]})
    return work


def _parse_episode_ref(ref: str, source: Path, lineno: int) -> tuple[str, int]:
    """``'61'`` -> ``('raw', 61)``; ``'recovery/56'`` -> ``('recovery', 56)``.

    The key matches the ``(source, raw_index)`` of a :func:`build_work_list` item.
    """
    if "/" in ref:
        prefix, _, rest = ref.partition("/")
        if prefix not in ("raw", "recovery") or not rest.isdigit():
            raise RuntimeError(
                f"{source}:{lineno}: bad episode ref {ref!r}; expected '<int>' or 'recovery/<int>'"
            )
        return prefix, int(rest)
    if not ref.isdigit():
        raise RuntimeError(
            f"{source}:{lineno}: bad episode ref {ref!r}; expected '<int>' or 'recovery/<int>'"
        )
    return "raw", int(ref)


def parse_split_csv(split_csv: Path) -> dict[tuple[str, int], str]:
    """Parse ``split.csv`` into ``{(source, raw_index): split}``.

    Format (one header row, then one row per NON-``train`` episode)::

        episode, split
        61, test
        recovery/56, test

    ``episode`` is a bare raw index ``<N>`` (source ``"raw"``) or ``recovery/<N>``
    (source ``"recovery"``); ``split`` is one of :data:`ALL_SPLITS`. Episodes NOT
    listed default to :data:`DEFAULT_SPLIT` (see :func:`assign_splits`). Raises on a
    missing file, a malformed row, an unknown split, or a duplicate episode. Only
    READ, never modified.
    """
    split_csv = Path(split_csv)
    if not split_csv.exists():
        raise RuntimeError(f"split csv not found: {split_csv}")
    spec: dict[tuple[str, int], str] = {}
    with open(split_csv, newline="") as handle:
        for lineno, row in enumerate(csv.reader(handle), start=1):
            cells = [c.strip() for c in row]
            if not cells or all(c == "" for c in cells):
                continue  # blank line
            if cells[0].lower() == "episode":
                continue  # header row (e.g. 'episode, split')
            if len(cells) != 2:
                raise RuntimeError(
                    f"{split_csv}:{lineno}: expected 2 columns 'episode, split', "
                    f"got {len(cells)}: {row}"
                )
            episode_ref, split = cells
            key = _parse_episode_ref(episode_ref, split_csv, lineno)
            if split not in ALL_SPLITS:
                raise RuntimeError(
                    f"{split_csv}:{lineno}: unknown split {split!r}; expected one of {ALL_SPLITS}"
                )
            if key in spec:
                raise RuntimeError(f"{split_csv}:{lineno}: duplicate episode {episode_ref!r}")
            spec[key] = split
    return spec


def assign_splits(work: list[dict], split_map: Mapping[tuple[str, int], str],
                  include_recovery_data: Path | None = None) -> dict[str, list[dict]]:
    """Partition the ordered work list into ``{split: [items...]}`` per ``split_map``.

    Items not named in ``split_map`` go to :data:`DEFAULT_SPLIT` (``"train"``). Within
    each split the raw-then-recovery numeric order of ``work`` is preserved, and empty
    splits are dropped. Raises if ``split_map`` names an episode absent from ``work``
    (a stale split.csv -- e.g. a ``recovery/<N>`` row without ``--include_recovery_data``).
    """
    keys = {(w["source"], w["raw_index"]) for w in work}
    unknown = sorted(set(split_map) - keys)
    if unknown:
        refs = ", ".join(f"recovery/{i}" if s == "recovery" else str(i) for s, i in unknown)
        hint = ("" if include_recovery_data is not None
                else " (recovery episodes require --include_recovery_data)")
        raise RuntimeError(f"split csv references episodes not in the port plan: {refs}{hint}")
    buckets: dict[str, list[dict]] = {s: [] for s in ALL_SPLITS}
    for w in work:
        buckets[split_map.get((w["source"], w["raw_index"]), DEFAULT_SPLIT)].append(w)
    return {s: buckets[s] for s in ALL_SPLITS if buckets[s]}


def write_log_index(log_index_path: Path, rows: list[dict]) -> None:
    """Write the raw->processed episode index map as CSV (with a header).

    Columns: ``split``, ``source`` (raw/recovery), ``raw_data_index`` (N in
    ``episode_<N>.hdf5``), ``processed_data_index`` (0-based LeRobot episode index
    WITHIN its split), ``num_frames_before`` (raw take length), ``num_frames_after``
    (frames written after the drop/trim).
    """
    log_index_path = Path(log_index_path)
    log_index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_index_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "source", "raw_data_index", "processed_data_index",
                         "num_frames_before", "num_frames_after"])
        for r in rows:
            writer.writerow([r["split"], r["source"], r["raw_index"], r["processed_index"],
                             r["num_before"], r["num_after"]])
    logging.info("Wrote index log %s (%d episode(s))", log_index_path, len(rows))


def _write_meta(variant_root: Path, split: str, state_frame: str, fps: int,
                resize_h: int, resize_w: int, task: str) -> None:
    """Write the ``dexmate_meta.json`` sidecar describing this split's schema."""
    relative_ready = state_frame == "world"
    (variant_root / "dexmate_meta.json").write_text(json.dumps({
        "schema": "wbc_eef_head_v1",
        "split": split,
        "state_axes": list(STATE_AXES), "action_axes": list(ACTION_AXES),
        # "base": egocentric achieved FK (absolute-action default). "world": composed
        # to engage-origin world so state shares the action frame -- for use_relative_actions.
        "state_frame": state_frame,
        "action_frame": "world (verbatim ik.solve targets, engage-origin)",
        # This dataset is only usable for use_relative_actions=true when state_frame="world"
        # (LeRobot's relative step subtracts observation.state[:29] from the world action).
        "relative_actions_ready": relative_ready,
        "relative_exclude_dims": RELATIVE_EXCLUDE_DIMS if relative_ready else None,
        # 6-D rotation dims ride raw under QUANTILES normalization (README step 3).
        "skip_normalization_dims": SKIP_NORMALIZATION_DIMS,
        "fps": fps, "resize_h": resize_h, "resize_w": resize_w,
        "task": task, "robot_type": ROBOT_TYPE,
        "gripper_binary_threshold": GRIPPER_BINARY_THRESHOLD,
    }, indent=2))


def verify(dataset, first_item: dict, fk: WBCPolicyFK, resize_h: int, resize_w: int,
           fps: int, state_frame: str, variant_root: Path, split: str) -> None:
    """Re-open a written split dataset and cross-check frame 0 against a fresh build.

    Confirms ``observation.state``/``action`` for the split's first episode first kept
    frame match an independent recomputation (SAME FK model); the action EEF/head blocks
    are the VERBATIM world targets (no composition); grippers are binary; depth/calib are
    NOT parquet features; and both sidecars round-trip. Needs lerobot -- only the real
    conversion calls it (the dexmate-only unit tests never do).
    """
    meta = dataset.meta
    logging.info("verify[%s]: fps=%s episodes=%s frames=%s robot=%s", split,
                 meta.fps, meta.total_episodes, meta.total_frames, meta.robot_type)

    for forbidden in ("observation.images.head_depth", "observation.images.depth",
                      "observation.extrinsic", "observation.intrinsic",
                      "observation.base_extrinsic"):
        if forbidden in meta.features:
            raise RuntimeError(
                f"verify[{split}]: {forbidden} must NOT be a dataset feature -- it belongs "
                "in the debug sidecar under <variant_root>/debug/"
            )

    item = dataset[0]
    head_rgb, wrist_rgb = item["observation.images.head_rgb"], item["observation.images.wrist_rgb"]
    state = item["observation.state"].cpu().numpy()
    action = item["action"].cpu().numpy()
    for cam_key, cam in (("head_rgb", head_rgb), ("wrist_rgb", wrist_rgb)):
        if tuple(cam.shape) != (3, resize_h, resize_w):
            raise RuntimeError(f"verify[{split}]: {cam_key} shape {tuple(cam.shape)} != "
                               f"(3, {resize_h}, {resize_w})")
    if state.shape != (len(STATE_AXES),):
        raise RuntimeError(f"verify[{split}]: state shape {state.shape} != ({len(STATE_AXES)},)")
    if action.shape != (len(ACTION_AXES),):
        raise RuntimeError(f"verify[{split}]: action shape {action.shape} != ({len(ACTION_AXES)},)")

    # Recompute the first stored frame independently (same FK) and pull the raw verbatim
    # targets to confirm the action is a straight copy (no base composition).
    ep = load_and_validate_episode(first_item["path"], fps)
    start, _ = _resolve_window(ep["frame_count"], ep["t0"], first_item["trim"], first_item["path"])
    with h5py.File(first_item["path"], "r") as raw:
        action_eef, action_head = load_action_targets_world(raw, start)
        exp_state, exp_action = compute_frame_state_action(
            raw, start, fk, action_eef, action_head, state_frame=state_frame)
        raw_left = mat_to_pos6d(np.asarray(raw["action/eef/left"][start], dtype=np.float64))
        raw_head = mat_to_pos6d(np.asarray(raw["action/head"][start], dtype=np.float64))
        raw_base = np.asarray(raw["obs/base/pose"][start], dtype=np.float64)
        depth_start = np.asarray(raw["obs/images/head_depth"][start])
    if not np.allclose(state, exp_state, atol=1e-4):
        raise RuntimeError(f"verify[{split}]: stored state[0] != recompute at frame {start}")
    if not np.allclose(action, exp_action, atol=1e-4):
        raise RuntimeError(f"verify[{split}]: stored action[0] != recompute at frame {start}")
    if not np.allclose(action[:9], raw_left, atol=1e-4):
        raise RuntimeError(f"verify[{split}]: action left block is not the verbatim world target")
    if not np.allclose(action[20:29], raw_head, atol=1e-4):
        raise RuntimeError(f"verify[{split}]: action head block is not the verbatim world target")
    if not np.allclose(state[29:32], raw_base, atol=1e-5):
        raise RuntimeError(f"verify[{split}]: state base dims != obs/base/pose[{start}]")
    for idx in GRIPPER_DIMS:
        if float(action[idx]) not in (0.0, 1.0):
            raise RuntimeError(f"verify[{split}]: action gripper dim {idx} is not binary "
                               f"({float(action[idx])})")

    # Depth sidecar roundtrip (episode 0, frame 0 == the window start).
    depth_path = _sidecar_path(variant_root, "depth", 0)
    if not depth_path.exists():
        raise RuntimeError(f"verify[{split}]: depth sidecar missing: {depth_path}")
    with np.load(depth_path) as data:
        depth_stack = data["depth"]
    if depth_stack.dtype != np.uint16 or depth_stack.shape[1:] != (resize_h, resize_w):
        raise RuntimeError(f"verify[{split}]: depth sidecar {depth_stack.dtype} "
                           f"{depth_stack.shape[1:]} != uint16 ({resize_h}, {resize_w})")
    expected_depth = cv2.resize(depth_start, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
    if not np.array_equal(depth_stack[0], expected_depth):
        raise RuntimeError(f"verify[{split}]: depth sidecar[0] roundtrip mismatch")

    # Calib sidecar: keys/shapes + the static intrinsic rescale.
    calib_path = _sidecar_path(variant_root, "calib", 0)
    if not calib_path.exists():
        raise RuntimeError(f"verify[{split}]: calib sidecar missing: {calib_path}")
    with np.load(calib_path) as data:
        extrinsic_sc, base_extrinsic_sc, intrinsic_sc = (
            data["extrinsic"], data["base_extrinsic"], data["intrinsic"])
    if extrinsic_sc.shape[1:] != (4, 4) or base_extrinsic_sc.shape[1:] != (4, 4):
        raise RuntimeError(f"verify[{split}]: calib extrinsics not (T,4,4): "
                           f"{extrinsic_sc.shape}, {base_extrinsic_sc.shape}")
    if intrinsic_sc.shape[1:] != (3, 3):
        raise RuntimeError(f"verify[{split}]: calib intrinsic not (T,3,3): {intrinsic_sc.shape}")
    expected_intrinsic = _scaled_intrinsic(ep["intrinsic"], ep["rgb"].shape[1:3], (resize_h, resize_w))
    if not np.allclose(intrinsic_sc[0], expected_intrinsic, atol=1e-4):
        raise RuntimeError(f"verify[{split}]: calib intrinsic[0] != rescaled static intrinsic")

    logging.info(
        "verify[%s] OK: head_rgb=%s wrist_rgb=%s state=%s action=%s depth=%s "
        "calib=extrinsic%s+base%s+intrinsic%s", split,
        tuple(head_rgb.shape), tuple(wrist_rgb.shape), tuple(state.shape), tuple(action.shape),
        tuple(depth_stack.shape), tuple(extrinsic_sc.shape), tuple(base_extrinsic_sc.shape),
        tuple(intrinsic_sc.shape),
    )


def convert_dataset(raw_dir: Path, repo_id: str, fps: int, resize_h: int,
                    resize_w: int, task: str, root: Path, overwrite: bool,
                    state_frame: str = "base", include_recovery_data: Path | None = None,
                    log_index: Path | None = None,
                    split_map: Mapping[tuple[str, int], str] | None = None) -> dict:
    """Full conversion (needs lerobot + tqdm -- run in the dexmate_lerobot env).

    Splits the episodes per ``split_map`` (from split.csv) into
    ``<root>/<split>/<repo_id>/`` datasets (unlisted -> :data:`DEFAULT_SPLIT`); one tqdm
    bar spans every episode across splits, and each split is ``verify()``-ed after it is
    written. Returns ``{split: LeRobotDataset}``.
    """
    from lerobot.datasets import LeRobotDataset, recompute_stats  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415

    work = build_work_list(raw_dir, include_recovery_data)
    splits = assign_splits(work, split_map or {}, include_recovery_data)
    n_recovery = sum(item["source"] == "recovery" for item in work)
    logging.info(
        "Port plan: %d episode(s) = %d raw + %d recovery -> splits {%s} (state_frame=%s)",
        len(work), len(work) - n_recovery, n_recovery,
        ", ".join(f"{s}:{len(v)}" for s, v in splits.items()), state_frame,
    )

    # Fail fast on pre-existing split dirs BEFORE any heavy work / partial writes.
    variant_roots = {s: Path(root) / s / repo_id for s in splits}
    for split, variant_root in variant_roots.items():
        if variant_root.exists():
            if not overwrite:
                raise RuntimeError(f"{variant_root} already exists. Use --overwrite to replace it.")
            logging.info("Removing existing %s (--overwrite)", variant_root)
            shutil.rmtree(variant_root)

    fk = WBCPolicyFK()
    datasets: dict[str, object] = {}
    log_rows: list[dict] = []
    grand_total = 0
    with tqdm(total=len(work), desc="Porting episodes", unit="ep") as pbar:
        for split, items in splits.items():
            variant_root = variant_roots[split]
            dataset = LeRobotDataset.create(
                repo_id=repo_id, fps=fps, features=build_features(resize_h, resize_w),
                root=variant_root, robot_type=ROBOT_TYPE, use_videos=True,
            )
            split_total = 0
            for processed_idx, item in enumerate(items):
                num_before, num_after = add_episode(
                    dataset, item["path"], fk, resize_h, resize_w, task,
                    variant_root=variant_root, episode_idx=processed_idx, fps=fps,
                    state_frame=state_frame, trim=item["trim"])
                split_total += num_after
                log_rows.append({"split": split, "source": item["source"],
                                 "raw_index": item["raw_index"], "processed_index": processed_idx,
                                 "num_before": num_before, "num_after": num_after})
                pbar.update(1)
            dataset.finalize()
            dataset = recompute_stats(dataset, skip_image_video=True)
            _write_meta(variant_root, split, state_frame, fps, resize_h, resize_w, task)
            verify(dataset, items[0], fk, resize_h, resize_w, fps, state_frame,
                   variant_root, split)
            datasets[split] = dataset
            grand_total += split_total
            logging.info("Split %s finalized: %d episode(s), %d frames", split,
                         len(items), split_total)

    logging.info("Done: %d split(s), %d episode(s), %d frames total",
                 len(splits), len(work), grand_total)
    if log_index is not None:
        write_log_index(log_index, log_rows)
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--resize-h", type=int, default=DEFAULT_RESIZE_H)
    parser.add_argument("--resize-w", type=int, default=DEFAULT_RESIZE_W)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--repo-id", type=str, default=None,
                        help=f"dataset repo-id (default {DEFAULT_REPO_ID}, or "
                             f"{DEFAULT_REPO_ID}_rel with --relative-actions).")
    parser.add_argument("--relative-actions", action="store_true",
                        help="emit observation.state EEF/head in the WORLD frame "
                             "(composed via obs/base/pose) so the dataset supports "
                             "use_relative_actions=true training (LeRobot subtracts "
                             "observation.state[:29] from the world action). Default: "
                             "base-frame state (absolute-action, egocentric).")
    parser.add_argument("--include-recovery-data", "--include_recovery_data",
                        dest="include_recovery_data", type=Path, default=None,
                        help="directory of extra recovery episode_<N>.hdf5 takes to "
                             "APPEND after the raw episodes. Requires <dir>/trim.csv "
                             "('episode, frame_start, frame_end' rows; ABSOLUTE INCLUSIVE "
                             "raw-frame indices, frame_end=-1 -> last frame). Every "
                             "recovery file must have exactly one trim row and vice-versa. "
                             "The recovery dir is only read, never modified.")
    parser.add_argument("--log-index", "--log_index", dest="log_index", type=Path,
                        default=None,
                        help="write a CSV mapping raw episode index -> processed dataset "
                             "index (columns: split, source, raw_data_index, "
                             "processed_data_index, num_frames_before, num_frames_after).")
    parser.add_argument("--split-csv", "--split_csv", dest="split_csv", type=Path,
                        default=None,
                        help="CSV assigning episodes to train/val/test (default "
                             "<raw-dir>/split.csv). Rows 'episode, split' where episode is "
                             "'<N>' (raw) or 'recovery/<N>'; unlisted episodes -> train. Each "
                             "split is written to <root>/<split>/<repo_id>/. If the DEFAULT "
                             "file is absent all episodes -> train; an explicitly-passed "
                             "missing file errors.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    state_frame = "world" if args.relative_actions else "base"
    if state_frame not in STATE_FRAMES:  # defensive; STATE_FRAMES is the source of truth
        raise SystemExit(f"invalid state_frame {state_frame!r}")
    repo_id = args.repo_id or (
        f"{DEFAULT_REPO_ID}_rel" if args.relative_actions else DEFAULT_REPO_ID
    )

    # Resolve the split map. An explicit --split-csv must exist; the default
    # <raw-dir>/split.csv is optional (absent -> every episode goes to DEFAULT_SPLIT).
    explicit_split = args.split_csv is not None
    split_csv = args.split_csv if explicit_split else args.raw_dir / "split.csv"

    try:
        if split_csv.exists():
            split_map = parse_split_csv(split_csv)
            logging.info("Using split csv %s (%d listed episode(s))", split_csv, len(split_map))
        elif explicit_split:
            raise RuntimeError(f"--split-csv not found: {split_csv}")
        else:
            logging.info("No split csv at %s; all episodes -> %s", split_csv, DEFAULT_SPLIT)
            split_map = {}
        convert_dataset(args.raw_dir, repo_id, args.fps, args.resize_h,
                        args.resize_w, args.task, args.root, args.overwrite,
                        state_frame=state_frame,
                        include_recovery_data=args.include_recovery_data,
                        log_index=args.log_index, split_map=split_map)
    except Exception as exc:  # CLI boundary: name the failure, exit 1
        logging.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
