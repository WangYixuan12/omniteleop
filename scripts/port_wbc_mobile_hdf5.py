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

Environments: ``--dry-run`` and the unit tests need only the dexmate env
(pinocchio/pink); the actual conversion additionally needs ``lerobot``
(``dexmate_lerobot`` env) -- lerobot imports are lazy.

Usage:
  python scripts/port_wbc_mobile_hdf5.py \
    --raw-dir ~/Dexmate/data/raw_data --root ~/Dexmate/data/processed_wbc \
    --fps 10 --resize-h 240 --resize-w 320 --repo-id dexmate_wbc_eef_head
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

import cv2
import h5py
import numpy as np

from omniteleop.wbc_policy_format import (
    ACTION_AXES,
    RELATIVE_EXCLUDE_DIMS,
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
DEFAULT_RESIZE_H = 240
DEFAULT_RESIZE_W = 320
DEFAULT_TASK = "dexmate_wbc"
DEFAULT_REPO_ID = "dexmate_wbc_eef_head"
GRIPPER_BINARY_THRESHOLD = 0.5
ROBOT_TYPE = "dexmate_vega_wbc"
# Rotation-column sanity for RECORDED action targets (float32 leader poses).
_ROT_COL_NORM_TOL = 1e-2

_REQUIRED_ACTION_TARGETS = ("action/eef/left", "action/eef/right", "action/head")


def find_episode_files(raw_dir: Path) -> list[Path]:
    """Flat ``episode_*.hdf5`` listing (the wbc_vr_robot recorder layout)."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise RuntimeError(f"raw dir not found: {raw_dir}")
    episode_files = sorted(raw_dir.glob("episode_*.hdf5"))
    if not episode_files:
        raise RuntimeError(f"no episode_*.hdf5 found in {raw_dir}")
    return episode_files


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
    """Read + cross-validate every array the port needs; shared with --dry-run.

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
                fps: int, state_frame: str = "base") -> int:
    """Port one raw episode; returns the number of frames written."""
    if not task:
        raise ValueError("task must be a non-empty string")
    ep = load_and_validate_episode(hdf5_path, fps)
    rgb, wrist_rgb, depth = ep["rgb"], ep["wrist_rgb"], ep["depth"]
    t0, frame_count = ep["t0"], ep["frame_count"]
    kept = frame_count - t0

    depth_resized = np.empty((kept, resize_h, resize_w), dtype=np.uint16)
    extrinsic = np.empty((kept, 4, 4), dtype=np.float32)
    base_extrinsic = np.empty((kept, 4, 4), dtype=np.float32)

    with h5py.File(hdf5_path, "r") as raw:
        for i, t in enumerate(range(t0, frame_count)):
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
    logging.info("Ported %s: %d frames (dropped %d leading), sidecars %s / %s",
                 hdf5_path.name, kept, t0, depth_path.name, calib_path.name)
    return kept


def convert_dataset(raw_dir: Path, repo_id: str, fps: int, resize_h: int,
                    resize_w: int, task: str, root: Path, overwrite: bool,
                    state_frame: str = "base"):
    """Full conversion (needs lerobot -- run in the dexmate_lerobot env)."""
    from lerobot.datasets import LeRobotDataset, recompute_stats  # noqa: PLC0415

    episode_files = find_episode_files(raw_dir)
    logging.info("Found %d episode file(s) in %s (state_frame=%s)",
                 len(episode_files), raw_dir, state_frame)

    variant_root = Path(root) / repo_id
    if variant_root.exists():
        if not overwrite:
            raise RuntimeError(f"{variant_root} already exists. Use --overwrite to replace it.")
        logging.info("Removing existing %s (--overwrite)", variant_root)
        shutil.rmtree(variant_root)

    fk = WBCPolicyFK()
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=build_features(resize_h, resize_w),
        root=variant_root, robot_type=ROBOT_TYPE, use_videos=True,
    )
    total = 0
    for episode_idx, ep_path in enumerate(episode_files):
        total += add_episode(dataset, ep_path, fk, resize_h, resize_w, task,
                             variant_root=variant_root, episode_idx=episode_idx, fps=fps,
                             state_frame=state_frame)
    dataset.finalize()
    logging.info("Finalized dataset: %d episodes, %d frames", len(episode_files), total)
    dataset = recompute_stats(dataset, skip_image_video=True)

    relative_ready = state_frame == "world"
    (variant_root / "dexmate_meta.json").write_text(json.dumps({
        "schema": "wbc_eef_head_v1",
        "state_axes": list(STATE_AXES), "action_axes": list(ACTION_AXES),
        # "base": egocentric achieved FK (absolute-action default). "world": composed
        # to engage-origin world so state shares the action frame -- for use_relative_actions.
        "state_frame": state_frame,
        "action_frame": "world (verbatim ik.solve targets, engage-origin)",
        # This dataset is only usable for use_relative_actions=true when state_frame="world"
        # (LeRobot's relative step subtracts observation.state[:29] from the world action).
        "relative_actions_ready": relative_ready,
        "relative_exclude_dims": RELATIVE_EXCLUDE_DIMS if relative_ready else None,
        "fps": fps, "resize_h": resize_h, "resize_w": resize_w,
        "task": task, "robot_type": ROBOT_TYPE,
        "gripper_binary_threshold": GRIPPER_BINARY_THRESHOLD,
    }, indent=2))
    return dataset


def dry_run(raw_dir: Path, fps: int, state_frame: str = "base") -> None:
    """Validate every episode end-to-end (no lerobot, no dataset written)."""
    episode_files = find_episode_files(raw_dir)
    logging.info("Dry run over %d episode file(s) in %s (state_frame=%s)",
                 len(episode_files), raw_dir, state_frame)
    fk = WBCPolicyFK()
    for ep_path in episode_files:
        ep = load_and_validate_episode(ep_path, fps)
        with h5py.File(ep_path, "r") as raw:
            for t in (ep["t0"], ep["frame_count"] - 1):
                action_eef, action_head = load_action_targets_world(raw, t)
                state, action = compute_frame_state_action(
                    raw, t, fk, action_eef, action_head, state_frame=state_frame
                )
        logging.info(
            "%s OK: %d frames (t0=%d), state dim %d, action dim %d, rgb %sx%s",
            ep_path.name, ep["frame_count"], ep["t0"], state.shape[0], action.shape[0],
            ep["rgb"].shape[1], ep["rgb"].shape[2],
        )
    logging.info("Dry run passed: state (%d,), action (%d,), state_frame=%s",
                 len(STATE_AXES), len(ACTION_AXES), state_frame)


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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate shapes/dims only; no dataset is created")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    state_frame = "world" if args.relative_actions else "base"
    if state_frame not in STATE_FRAMES:  # defensive; STATE_FRAMES is the source of truth
        raise SystemExit(f"invalid state_frame {state_frame!r}")
    repo_id = args.repo_id or (
        f"{DEFAULT_REPO_ID}_rel" if args.relative_actions else DEFAULT_REPO_ID
    )

    try:
        if args.dry_run:
            dry_run(args.raw_dir, args.fps, state_frame=state_frame)
        else:
            convert_dataset(args.raw_dir, repo_id, args.fps, args.resize_h,
                            args.resize_w, args.task, args.root, args.overwrite,
                            state_frame=state_frame)
    except Exception as exc:  # CLI boundary: name the failure, exit 1
        logging.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
