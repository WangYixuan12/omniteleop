"""Compare ``action`` vs ``observation.state`` from processed LeRobotDataset variants in rerun.

Sibling to ``scripts/vis_episode_online.py`` — same rendering structure, but
sourced entirely from processed parquet variants produced by
``examples/port_datasets/port_dexmate_hdf5.py`` (no raw HDF5 input).

Two variants of the same source episode are loaded in lockstep. The viewer handles
BOTH bimanual and single-arm datasets: the active arm side(s) and per-block column
offsets are read from each variant's stored axis names, so dims scale automatically
(joint 8-D, eef 10-D per arm). The shapes below describe the BIMANUAL case — each
vector is the LEFT arm's block followed by the RIGHT arm's block; a single-arm dataset
carries just the one selected side's block (half the dims), and any arm absent from the
data is pinned to its INIT_*_ARM_JOINTS in the URDF view:

- ``--marker_and_plot_dir`` (optional; the ``dexmate_eef_eef`` variant root).
  Omit to visualize only ``--urdf_joint_motion_dir`` (URDF + joint grippers;
  cameras/depth use the joint variant sidecars). When set:
    * ``observation.state`` (N, 20) — per arm: FK on the OBSERVED arm joints
      (pos6d + raw gripper); the EEF pose the robot actually reached. Left =
      ``[0:10]``, Right = ``[10:20]``; each block = ``[tx,ty,tz, R6(6), grip]``.
    * ``action`` (N, 20) — per arm: FK on the COMMANDED arm joints (pos6d +
      binary gripper); the FK image of the dexmate_joint_joint action.
    * Camera scene: ``observation.images.head_rgb`` + depth sidecar drive the
      head RGB / depth / point-cloud panels; ``observation.images.wrist_rgb``
      drives a plain 2D wrist panel (no calibration/depth for the wrist).

- ``--urdf_joint_motion_dir`` (the ``dexmate_joint_joint`` variant root):
    * ``observation.state`` (N, 16) — per arm: 7 joints + raw gripper. The left
      arm ``[0:7]`` and right arm ``[8:15]`` drive both URDF arms.

What this shows:

- **EEF position** — per arm, red markers = ``action`` pos; blue markers =
  ``observation.state`` pos (eef variant).
- **Position condition** — per frame, orange/green markers on the head RGB
  panel show the selected 6-D ``observation.environment_state`` chunk:
  before XYZ and after XYZ. Selection uses ``observation.pos_condition_mask``
  exactly like training (``[1, 0]`` -> ``environment_state[:6]``,
  ``[0, 1]`` -> ``environment_state[6:12]``).
- **Robot URDF** — pose follows ``observation.state`` from the joint_joint
  variant (left arm ``[0:7]`` + right arm ``[8:15]``; torso/head pinned INIT_*).
- **Gripper** — two plots (left, right). In each: red = ``action`` gripper
  (binary commanded); blue = ``observation.state`` gripper (raw observed).
  Left gripper = dim 9, right gripper = dim 19.
- **Matplotlib EEF xyz** — per arm (left + right), red = ``action``, blue =
  ``observation.state``; per-axis cross-correlation lag annotated.

A small per-frame divergence is expected (controller tracking lag); large
divergence indicates a port-time or controller bug.

Camera calibration (per-frame extrinsic and intrinsic) is read from the
sidecar at ``<variant_root>/debug/calib/episode_<idx>.npz``:

- ``"extrinsic"`` : ``(T, 4, 4)`` float32 — camera-in-world SE(3) pose.
- ``"intrinsic"`` : ``(T, 3, 3)`` float32 — already rescaled to the stored
  ``resize_h × resize_w`` frame; use directly without further scaling.

The porter (``port_dexmate_hdf5.py``) writes this sidecar alongside the
depth sidecar.

Depth: NOT in the parquet (would be auto-picked-up as a VISUAL policy input).
Read instead from the sidecar at
``<variant_root>/debug/depth/episode_<idx>.npz`` (key ``"depth"``, shape
``(T, H, W)`` uint16). The porter writes it in lockstep with the parquet.

Usage::

    # Full comparison (eef markers + joint URDF):
    python scripts/vis_episode_processed.py \\
        --marker_and_plot_dir /home/yixuan/omniteleop/Dexmate/data/processed_data/train/dexmate_eef_eef \\
        --urdf_joint_motion_dir /home/yixuan/omniteleop/Dexmate/data/processed_data/train/dexmate_joint_joint \\
        --episode_index 0

    # Joint URDF only:
    python scripts/vis_episode_processed.py \\
        --urdf_joint_motion_dir /home/yixuan/omniteleop/Dexmate/data/processed_data/train/dexmate_joint_joint \\
        --episode_index 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from yixuan_utilities.kinematics_helper import KinHelper
from yixuan_utilities.robot_mesh_generator import RobotMeshGenerator

from omniteleop.common.vr_mode_const import (
    INIT_HEAD_JOINTS,
    INIT_LEFT_ARM_JOINTS,
    INIT_RIGHT_ARM_JOINTS,
    INIT_TORSO_JOINTS,
)

_ROBOT_NAME = "vega_no_effector"
_RIGHT_ARM_EEF_LINK = "R_ee"
_TORSO_JOINTS = ("torso_j1", "torso_j2", "torso_j3")
_HEAD_JOINTS = ("head_j1", "head_j2", "head_j3")
_LEFT_ARM_JOINTS = tuple(f"L_arm_j{i}" for i in range(1, 8))
_RIGHT_ARM_JOINTS = tuple(f"R_arm_j{i}" for i in range(1, 8))

_WHEEL_NAMES = ["B_wheel_j1", "B_wheel_j2", "R_wheel_j1", "R_wheel_j2", "L_wheel_j1", "L_wheel_j2"]
_OBS_NAMES = (
    list(_TORSO_JOINTS) + list(_LEFT_ARM_JOINTS) + list(_RIGHT_ARM_JOINTS) + list(_HEAD_JOINTS)
)

_COLOR_ACTION = (255, 0, 0)   # red  — ``action`` markers + binary gripper
_COLOR_STATE  = (0, 80, 255)  # blue — ``observation.state`` markers + raw gripper
_COLOR_CONDITION_BEFORE = (255, 180, 0)  # orange — selected condition before XYZ
_COLOR_CONDITION_AFTER = (0, 220, 120)   # green  — selected condition after XYZ
_COLOR_CONDITION_LINE = (255, 255, 255)  # white  — before->after condition vector

_OBS_ENV_STATE_KEY = "observation.environment_state"
_OBS_POS_CONDITION_MASK_KEY = "observation.pos_condition_mask"

# State/action layout (porter output). Per-arm block: eef = 10-D
# [tx,ty,tz, R6(6), gripper]; joint = 8-D [7 joints, gripper]. Bimanual is the LEFT
# block then the RIGHT block; single-arm is just one side's block. The active arm
# side(s) and their per-block column offsets are recovered from each variant's stored
# axis names below, so the same viewer handles bimanual and single-arm datasets.
_INIT_ARM_JOINTS = {"left": INIT_LEFT_ARM_JOINTS, "right": INIT_RIGHT_ARM_JOINTS}


def arm_sides_from_axes(axes: list[str]) -> tuple[str, ...]:
    """Ordered (left-then-right) arm sides present in a state/action axis-name list.

    The porter names every dim ``{side}_*`` (e.g. ``left_tx`` / ``right_arm_j0``),
    so the active arm(s) are recovered from the prefixes. Raises if none are found.
    """
    seen = [str(ax).split("_", 1)[0] for ax in axes]
    sides = tuple(s for s in ("left", "right") if s in seen)
    if not sides:
        raise ValueError(f"could not parse any arm side (left/right) from axes {axes}")
    return sides


def eef_index_maps(
    arm_sides: tuple[str, ...],
) -> tuple[dict[str, int], dict[str, int]]:
    """Per-side start column + gripper column for the 10-D eef blocks."""
    base = {side: i * 10 for i, side in enumerate(arm_sides)}
    grip = {side: i * 10 + 9 for i, side in enumerate(arm_sides)}
    return base, grip


def joint_index_maps(
    arm_sides: tuple[str, ...],
) -> tuple[dict[str, slice], dict[str, int]]:
    """Per-side 7-joint slice + gripper column for the 8-D joint blocks."""
    arm_slice = {side: slice(i * 8, i * 8 + 7) for i, side in enumerate(arm_sides)}
    grip = {side: i * 8 + 7 for i, side in enumerate(arm_sides)}
    return arm_slice, grip

_GRIPPER_Y_MIN = -0.2
_GRIPPER_Y_MAX = 1.1

# Default rerun playback rate for the sequence timeline (frames/sec at 1× speed).
_PLAYBACK_FPS = 15.0


def gram_schmidt_6d_to_R(r6: np.ndarray) -> np.ndarray:
    """Recover (3, 3) rotation from Zhou et al. 6-D representation."""
    a1, a2 = r6[:3], r6[3:6]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def project_world_to_pixel(
    pts_world: np.ndarray, K: np.ndarray, world_t_cam: np.ndarray, z_min: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Project (N, 3) world points into pixel coords using the same convention
    as ``unproject_depth`` (camera frame: +x right, +y down, +z forward).

    Returns ``(uv, valid)`` where ``uv`` is (N, 2) float32 pixel coords and
    ``valid`` is a boolean mask of points in front of the camera.
    """
    pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
    cam_t_world = np.linalg.inv(world_t_cam)
    pts_homog = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    pts_cam = (cam_t_world @ pts_homog.T).T[:, :3]
    z = pts_cam[:, 2]
    valid = z > z_min
    z_safe = np.where(valid, z, 1.0)
    u = K[0, 0] * pts_cam[:, 0] / z_safe + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / z_safe + K[1, 2]
    return np.stack([u, v], axis=-1).astype(np.float32), valid


def select_position_conditions(
    env_state: np.ndarray,
    pos_condition_mask: np.ndarray | None,
    *,
    condition_dim: int = 6,
) -> np.ndarray:
    """Select per-frame position condition with the same slot logic as training."""
    if condition_dim <= 0:
        raise ValueError(f"condition_dim must be positive, got {condition_dim}")

    env = np.asarray(env_state, dtype=np.float32)
    if env.ndim != 2:
        raise ValueError(
            f"{_OBS_ENV_STATE_KEY} must be 2-D (T, D), got shape {env.shape}"
        )
    if not np.all(np.isfinite(env)):
        raise ValueError(f"{_OBS_ENV_STATE_KEY} must contain only finite values")

    frame_count, env_dim = env.shape
    if env_dim == condition_dim:
        return env.copy()
    if env_dim % condition_dim != 0:
        raise ValueError(
            f"{_OBS_ENV_STATE_KEY} last dimension must be {condition_dim} or a "
            f"multiple of it, got shape {env.shape}"
        )
    if pos_condition_mask is None:
        raise ValueError(
            f"{_OBS_POS_CONDITION_MASK_KEY} is required to select a {condition_dim}-D "
            f"condition from {_OBS_ENV_STATE_KEY} shape {env.shape}"
        )

    object_count = env_dim // condition_dim
    mask = np.asarray(pos_condition_mask, dtype=np.float32)
    if mask.shape != (frame_count, object_count):
        raise ValueError(
            f"{_OBS_POS_CONDITION_MASK_KEY} shape must be {(frame_count, object_count)} "
            f"to match {_OBS_ENV_STATE_KEY} shape {env.shape}, got {mask.shape}"
        )
    if not np.all(np.isfinite(mask)):
        raise ValueError(
            f"{_OBS_POS_CONDITION_MASK_KEY} must contain finite one-hot values"
        )

    is_binary = np.isclose(mask, 0.0, atol=1e-4, rtol=0) | np.isclose(
        mask, 1.0, atol=1e-4, rtol=0
    )
    if not np.all(is_binary) or not np.allclose(
        mask.sum(axis=-1), 1.0, atol=1e-4, rtol=0
    ):
        raise ValueError(
            f"{_OBS_POS_CONDITION_MASK_KEY} must be one-hot along the last dimension"
        )

    env_blocks = env.reshape(frame_count, object_count, condition_dim)
    return np.sum(env_blocks * mask[:, :, None], axis=1, dtype=np.float32)


def unproject_depth(
    depth_m: np.ndarray, K: np.ndarray, world_t_cam: np.ndarray, max_depth: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a depth image to world-frame XYZ. Returns (pts, mask)."""
    H, W = depth_m.shape
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    mask = (depth_m > 0.05) & (depth_m < max_depth)
    z = depth_m[mask]
    x = (u[mask] - K[0, 2]) / K[0, 0] * z
    y = (v[mask] - K[1, 2]) / K[1, 1] * z
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)
    pts_world = (world_t_cam @ pts_cam.T).T[:, :3]
    return pts_world.astype(np.float32), mask


def voxel_downsample(
    pts: np.ndarray, colors: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-average downsample on CUDA when available; random subsample otherwise."""
    if pts.shape[0] == 0:
        return pts, colors
    try:
        pts_t = torch.from_numpy(pts).cuda()
        cols_t = torch.from_numpy(colors.astype(np.float32)).cuda()
        voxels = torch.floor(pts_t / voxel_size).int()
        keys = voxels[:, 0] * (1 << 20) + voxels[:, 1] * (1 << 10) + voxels[:, 2]
        _, inv = torch.unique(keys, return_inverse=True)
        out_pts = torch.zeros(inv.max() + 1, 3, device="cuda")
        out_cols = torch.zeros(inv.max() + 1, 3, device="cuda")
        cnt = torch.zeros(inv.max() + 1, device="cuda")
        out_pts.scatter_add_(0, inv.unsqueeze(1).expand(-1, 3), pts_t)
        out_cols.scatter_add_(0, inv.unsqueeze(1).expand(-1, 3), cols_t)
        cnt.scatter_add_(0, inv, torch.ones(len(inv), device="cuda"))
        out_pts = (out_pts / cnt.unsqueeze(1)).cpu().numpy()
        out_cols = (out_cols / cnt.unsqueeze(1)).cpu().numpy().astype(np.uint8)
        return out_pts, out_cols
    except Exception:
        idx = np.random.choice(len(pts), min(len(pts), 50000), replace=False)
        return pts[idx], colors[idx]


class _Kin:
    """Shared KinHelper + name→index maps for FK / robot-mesh logic."""

    def __init__(self) -> None:
        self.kin = KinHelper(_ROBOT_NAME)
        self.eef_idx = self.kin.link_name_to_idx[_RIGHT_ARM_EEF_LINK]
        active = self.kin.sapien_robot.get_active_joints()
        self.name_to_idx = {j.name: i for i, j in enumerate(active)}
        self.dof = self.kin.sapien_robot.dof

    def _qpos(self, torso: np.ndarray, l_arm: np.ndarray, r_arm: np.ndarray,
              head: np.ndarray) -> np.ndarray:
        qpos = np.zeros(self.dof, dtype=np.float64)
        for vals, names in (
            (torso, _TORSO_JOINTS),
            (l_arm, _LEFT_ARM_JOINTS),
            (r_arm, _RIGHT_ARM_JOINTS),
            (head, _HEAD_JOINTS),
        ):
            for v, n in zip(vals, names, strict=True):
                if n in self.name_to_idx:
                    qpos[self.name_to_idx[n]] = v
        return qpos

    def fk_link(self, link_idx: int, torso, l_arm, r_arm, head) -> np.ndarray:
        qpos = self._qpos(torso, l_arm, r_arm, head)
        return self.kin.compute_fk_from_link_idx(qpos, [link_idx])[0]


def _rgb_to_hwc(t) -> np.ndarray:
    """Normalize an RGB observation to (H, W, 3) uint8."""
    a = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    if a.ndim == 3 and a.shape[0] == 3:
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    return a


def _format_lag_ms(lag_ms: float, *, include_unit: bool = False) -> str:
    if not np.isfinite(lag_ms):
        return "n/a"
    text = f"{lag_ms:.1f}"
    return f"{text} ms" if include_unit else text


def estimate_lag_ms(
    cmd: np.ndarray,
    actual: np.ndarray,
    dt_s: float,
    *,
    max_lag_ms: float = 1000.0,
    min_corr: float = 0.25,
    min_overlap_frac: float = 0.5,
) -> float:
    """Velocity cross-correlation peak lag in ms (positive = ``actual`` lags ``cmd``).

    Verbatim copy of ``vis_teleop_curves.estimate_lag_ms`` so plots share the
    identical lag definition with the raw-HDF5 viewer.
    """
    if dt_s <= 0 or not np.isfinite(dt_s):
        return float("nan")
    cmd = np.asarray(cmd, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if cmd.ndim != 1 or actual.ndim != 1 or len(cmd) != len(actual) or len(cmd) < 4:
        return float("nan")
    if not (np.all(np.isfinite(cmd)) and np.all(np.isfinite(actual))):
        return float("nan")
    a = np.diff(cmd)
    b = np.diff(actual)
    n = len(a)
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    min_overlap = max(3, int(np.ceil(n * min_overlap_frac)))
    max_lag_samples = round(max_lag_ms / (dt_s * 1000.0))
    max_lag_samples = min(max_lag_samples, n - min_overlap)
    if max_lag_samples < 1:
        return float("nan")
    lags = np.arange(-max_lag_samples, max_lag_samples + 1, dtype=np.float64)
    scores = np.full(lags.shape, np.nan, dtype=np.float64)
    for i, lag in enumerate(lags.astype(int)):
        if lag >= 0:
            aw = a[: n - lag]
            bw = b[lag:]
        else:
            aw = a[-lag:]
            bw = b[: n + lag]
        aw = aw - aw.mean()
        bw = bw - bw.mean()
        denom = np.linalg.norm(aw) * np.linalg.norm(bw)
        if denom > 1e-12:
            scores[i] = float(np.dot(bw, aw) / denom)
    if np.all(np.isnan(scores)):
        return float("nan")
    best_i = int(np.nanargmax(scores))
    best_corr = float(scores[best_i])
    if best_corr < min_corr:
        return float("nan")
    lag_samples = float(lags[best_i])
    if 0 < best_i < len(scores) - 1 and np.all(np.isfinite(scores[best_i - 1 : best_i + 2])):
        y0, y1, y2 = scores[best_i - 1 : best_i + 2]
        denom = y0 - (2.0 * y1) + y2
        if abs(denom) > 1e-12:
            offset = 0.5 * (y0 - y2) / denom
            lag_samples += float(np.clip(offset, -1.0, 1.0))
    return lag_samples * dt_s * 1000.0


def _build_eef_xyz_figs(
    t: np.ndarray,
    action_xyz: np.ndarray,
    state_xyz: np.ndarray,
    dt_s: float,
    prefix: str = "",
) -> list[plt.Figure]:
    """Three figures (one per axis) comparing ``action`` vs ``observation.state``.

    Per-axis velocity-domain cross-correlation lag (positive = state lags
    action — controller tracking lag) is shown in the upper-left of each
    subplot. Peak-difference annotations (Δ in mm at local extrema) live on
    the raw-HDF5 viewer ``vis_teleop_curves.fig_eef_xyz(show_extrema_gaps=True)``
    instead; this viewer keeps the processed-dataset curves uncluttered.
    """
    color_action = (1.0, 0.0, 0.0)         # red  — action
    color_state  = (0.0, 80 / 255.0, 1.0)  # blue — observation.state
    figs: list[plt.Figure] = []
    for axis_idx, axis_name in enumerate("xyz"):
        act = action_xyz[:, axis_idx]
        st  = state_xyz[:, axis_idx]
        lag_ms = estimate_lag_ms(act, st, dt_s)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(t, act, color=color_action, lw=1.4, label="action")
        ax.plot(t, st,  color=color_state,  lw=1.2, label="observation.state")

        ax.text(
            0.01, 0.95,
            f"lag = {_format_lag_ms(lag_ms, include_unit=True)}   "
            f"(positive = observation.state lags action)",
            transform=ax.transAxes, fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
        )
        ax.set_xlabel("t (s)")
        ax.set_ylabel(f"eef_{axis_name} (m)")
        ax.set_title(
            f"{prefix}EEF {axis_name}"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        figs.append(fig)
    return figs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--marker_and_plot_dir",
        type=str,
        help="Path to the dexmate_eef_eef variant root. Provides "
             "observation.state (eef) + action for the rerun markers, gripper "
             "curves, matplotlib EEF xyz figures, and the RGB / depth / "
             "point-cloud camera scene.",
    )
    parser.add_argument(
        "--urdf_joint_motion_dir",
        type=str,
        required=True,
        help="Path to the dexmate_joint_joint variant root. Provides "
             "observation.state (joint) so state[:7] drives the URDF mesh "
             "right-arm pose.",
    )
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument(
        "--voxel", type=float, default=0.008, help="Voxel size for PCD downsample (m)."
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Log every Nth frame (>=1). Robot mesh geometry is logged once "
        "regardless; this trims the per-frame image/depth/pointcloud cost, "
        "which dominates memory after the mesh fix.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="If set, stream to this .rrd file headlessly instead of spawning a "
        "viewer (useful over SSH: generate on the server, open later with "
        "`rerun FILE.rrd`). Default: spawn the viewer (single-script workflow).",
    )
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError(f"--stride must be >= 1, got {args.stride}")

    eef_enabled = args.marker_and_plot_dir is not None
    joint_root = Path(args.urdf_joint_motion_dir)
    if not joint_root.is_dir():
        raise FileNotFoundError(f"--urdf_joint_motion_dir not a directory: {joint_root}")
    eef_root: Path | None = None
    if eef_enabled:
        eef_root = Path(args.marker_and_plot_dir)
        if not eef_root.is_dir():
            raise FileNotFoundError(f"--marker_and_plot_dir not a directory: {eef_root}")

    # return_uint8=True: video frames default to float32 [0, 1]; we need raw
    # uint8 for direct rerun logging (and to avoid the all-zero astype trap).
    dataset_joint = LeRobotDataset(
        repo_id=joint_root.name,
        root=joint_root,
        episodes=[args.episode_index],
        return_uint8=True,
    )
    dataset_eef: LeRobotDataset | None = None
    if eef_enabled:
        assert eef_root is not None
        dataset_eef = LeRobotDataset(
            repo_id=eef_root.name,
            root=eef_root,
            episodes=[args.episode_index],
            return_uint8=True,
        )
        N = len(dataset_eef)
        if N == 0:
            raise ValueError(f"Episode {args.episode_index} is empty under {eef_root}.")
        if len(dataset_joint) != N:
            raise ValueError(
                f"frame count mismatch for episode {args.episode_index}: "
                f"marker_and_plot_dir={N}, urdf_joint_motion_dir={len(dataset_joint)}"
            )
    else:
        N = len(dataset_joint)
        if N == 0:
            raise ValueError(f"Episode {args.episode_index} is empty under {joint_root}.")

    # Cameras / depth sidecars live on whichever variant supplies RGB (eef when set).
    camera_root = eef_root if eef_enabled else joint_root
    dataset_rgb = dataset_eef if eef_enabled else dataset_joint

    # Active arm side(s) + per-block column maps, read from the joint variant's stored
    # axis names (self-describing: bimanual -> (left, right); single-arm -> one side).
    arm_sides = arm_sides_from_axes(
        dataset_joint.meta.features["observation.state"]["names"]["axes"]
    )
    n_arms = len(arm_sides)
    joint_arm_slice, joint_grip_idx = joint_index_maps(arm_sides)
    eef_base, eef_grip_idx = eef_index_maps(arm_sides)
    if eef_enabled:
        assert dataset_eef is not None
        eef_sides = arm_sides_from_axes(
            dataset_eef.meta.features["observation.state"]["names"]["axes"]
        )
        if eef_sides != arm_sides:
            raise ValueError(
                f"arm sides differ between variants: urdf_joint_motion_dir={arm_sides}, "
                f"marker_and_plot_dir={eef_sides}; pass the same episode's variants."
            )
    print(f"arm mode: {'bimanual' if n_arms == 2 else 'single-arm'} (sides={arm_sides})")

    state_joint = np.stack([dataset_joint[i]["observation.state"].numpy() for i in range(N)])  # (N, 8*n_arms)
    action_joint = np.stack([dataset_joint[i]["action"].numpy() for i in range(N)])  # (N, 8*n_arms)

    state_eef: np.ndarray | None = None
    action: np.ndarray | None = None
    position_condition: np.ndarray | None = None
    if eef_enabled:
        assert dataset_eef is not None
        state_eef = np.stack([dataset_eef[i]["observation.state"].numpy() for i in range(N)])  # (N, 10*n_arms)
        action = np.stack([dataset_eef[i]["action"].numpy() for i in range(N)])  # (N, 10*n_arms)
        missing_position_features = [
            key for key in (_OBS_ENV_STATE_KEY, _OBS_POS_CONDITION_MASK_KEY)
            if key not in dataset_eef.meta.features
        ]
        if missing_position_features:
            raise ValueError(
                "--marker_and_plot_dir is missing position-conditioning feature(s): "
                f"{missing_position_features}. Re-port with --positions-dir so the viewer "
                "can overlay the selected 6-D position condition on the head RGB."
            )
        env_state = np.stack([dataset_eef[i][_OBS_ENV_STATE_KEY].numpy() for i in range(N)])
        pos_condition_mask = np.stack([
            dataset_eef[i][_OBS_POS_CONDITION_MASK_KEY].numpy() for i in range(N)
        ])
        position_condition = select_position_conditions(env_state, pos_condition_mask)  # (N, 6)
        if position_condition.shape != (N, 6):
            raise ValueError(
                f"selected position condition must have shape {(N, 6)}, got "
                f"{position_condition.shape}"
            )

    # Depth sidecar (NOT in the parquet — see docstring).
    depth_sidecar = (
        camera_root / "debug" / "depth" / f"episode_{args.episode_index:06d}.npz"
    )
    if not depth_sidecar.exists():
        raise FileNotFoundError(
            f"depth sidecar missing: {depth_sidecar}. Re-run port_dexmate_hdf5.py "
            "with --overwrite to regenerate."
        )
    with np.load(depth_sidecar) as data:
        depth_stack = data["depth"]  # (T, H, W) uint16
    if depth_stack.shape[0] != N:
        raise ValueError(
            f"depth sidecar frame count {depth_stack.shape[0]} != dataset frame count {N}"
        )
    if depth_stack.dtype != np.uint16:
        raise ValueError(f"depth sidecar dtype {depth_stack.dtype}, expected uint16")

    # Calibration sidecar: per-frame extrinsic (T, 4, 4) and intrinsic (T, 3, 3).
    # Intrinsic is already rescaled to the stored resize_h × resize_w frame.
    calib_sidecar = (
        camera_root / "debug" / "calib" / f"episode_{args.episode_index:06d}.npz"
    )
    if not calib_sidecar.exists():
        raise FileNotFoundError(
            f"calib sidecar missing: {calib_sidecar}. Re-run port_dexmate_hdf5.py "
            "with --overwrite to regenerate."
        )
    with np.load(calib_sidecar) as data:
        extrinsic_stack = data["extrinsic"].astype(np.float64)   # (T, 4, 4) world_T_cam
        intrinsic_stack = data["intrinsic"].astype(np.float64)   # (T, 3, 3)
    if extrinsic_stack.shape[0] != N:
        raise ValueError(
            f"calib sidecar extrinsic frame count {extrinsic_stack.shape[0]} != {N}"
        )
    if intrinsic_stack.shape[0] != N:
        raise ValueError(
            f"calib sidecar intrinsic frame count {intrinsic_stack.shape[0]} != {N}"
        )

    if state_joint.shape[1] != 8 * n_arms:
        raise ValueError(
            f"--urdf_joint_motion_dir observation.state: expected (N, {8 * n_arms}) "
            f"for arm_sides={arm_sides}, got {state_joint.shape}"
        )
    if eef_enabled:
        assert state_eef is not None and action is not None
        if state_eef.shape[1] != 10 * n_arms:
            raise ValueError(
                f"--marker_and_plot_dir observation.state: expected (N, {10 * n_arms}) "
                f"for arm_sides={arm_sides}, got {state_eef.shape}"
            )
        if action.shape[1] != 10 * n_arms:
            raise ValueError(
                f"--marker_and_plot_dir action: expected (N, {10 * n_arms}) "
                f"for arm_sides={arm_sides}, got {action.shape}"
            )

    if "observation.images.wrist_rgb" not in dataset_rgb.meta.features:
        raise ValueError(
            f"{camera_root.name} is missing observation.images.wrist_rgb; "
            "re-port with port_dexmate_hdf5.py (always stores head + wrist)."
        )

    action_pos: dict[str, np.ndarray] = {}
    state_eef_pos: dict[str, np.ndarray] = {}
    if eef_enabled:
        assert state_eef is not None and action is not None
        action_pos = {
            side: action[:, eef_base[side]:eef_base[side] + 3] for side in arm_sides
        }
        state_eef_pos = {
            side: state_eef[:, eef_base[side]:eef_base[side] + 3] for side in arm_sides
        }

    # ── EEF xyz figures (eef variant only) ───────────────────────────────────
    dt_s = 1.0 / float(dataset_rgb.fps) if dataset_rgb.fps else 1.0 / 30.0
    eef_figs: list[plt.Figure] = []
    if eef_enabled:
        assert state_eef is not None and action is not None
        t_s = np.arange(N, dtype=np.float64) * dt_s
        for side in arm_sides:
            eef_figs += _build_eef_xyz_figs(
                t_s, action_pos[side], state_eef_pos[side], dt_s, prefix=f"{side} "
            )
        print(
            f"built {len(eef_figs)} EEF xyz figures (left+right) "
            f"(dt = {dt_s*1000:.1f} ms, fps = {1.0/dt_s:.1f} Hz)"
        )
        object_count = env_state.shape[1] // 6
        mask_switches = (
            np.flatnonzero(np.diff(np.argmax(pos_condition_mask, axis=1)) != 0) + 1
        )
        print(
            f"position condition: env_state {env_state.shape[1]}D over "
            f"{object_count} object slots -> selected 6D; mask switches at frames "
            f"{mask_switches.tolist()}"
        )
    else:
        print("joint-only mode: skipping EEF markers, position condition, and matplotlib EEF plots")

    # ── RGB dimensions (constant across frames) ──────────────────────────────
    sample_rgb = _rgb_to_hwc(dataset_rgb[0]["observation.images.head_rgb"])
    H, W = sample_rgb.shape[:2]
    print(f"episode {args.episode_index}: {N} frames  |  rgb: {H}x{W}  |  calib from sidecar")

    # ── FK for per-frame robot meshes ─────────────────────────────────────────
    # Camera pose and intrinsics now come from the calib sidecar (per-frame).
    robot_mesh_gen = RobotMeshGenerator(_ROBOT_NAME)

    # ── rerun setup ──────────────────────────────────────────────────────────
    rr.init("vis_episode_processed", spawn=args.save is None)
    if args.save is not None:
        rr.save(args.save)
        print(f"Streaming to {args.save} (headless; no viewer spawned)")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Explicit layout (auto-views are disabled once a blueprint is sent). Default
    # panels: 3D world (pointcloud + robot + both-arm EEF markers), head RGB/depth
    # (pinhole, under world/camera), wrist RGB (plain 2D, no calib/depth), and the
    # left + right gripper time series.
    lookback = max(400, min(N, 5000))
    lookahead = min(80, max(1, N // 50 + 5))

    def _gripper_view(origin: str, name: str) -> rrb.TimeSeriesView:
        return rrb.TimeSeriesView(
            origin=origin,
            name=name,
            axis_y=rrb.ScalarAxis(range=(_GRIPPER_Y_MIN, _GRIPPER_Y_MAX), zoom_lock=True),
            time_ranges=[
                rrb.VisibleTimeRange(
                    "frame",
                    start=rrb.TimeRangeBoundary.cursor_relative(seq=-lookback),
                    end=rrb.TimeRangeBoundary.cursor_relative(seq=lookahead),
                )
            ],
        )

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="world", name="World (pointcloud + robot)"),
                    rrb.Spatial2DView(origin="world/camera", name="Head RGB / depth"),
                    rrb.Spatial2DView(origin="wrist", name="Wrist RGB"),
                    column_shares=[2.0, 1.3, 1.3],
                ),
                rrb.Horizontal(
                    *[
                        _gripper_view(f"plots/gripper_{s}", f"{s.capitalize()} gripper")
                        for s in arm_sides
                    ],
                ),
                row_shares=[2.2, 1.0],
            ),
            rrb.TimePanel(timeline="frame", fps=_PLAYBACK_FPS),
        )
    )

    # One gripper plot per arm; each shares two named series:
    #   action[grip]            (red)  — binarized commanded gripper
    #   observation.state[grip] (blue) — raw obs gripper /obs/gripper/{side}
    # Left gripper = dim 9, right gripper = dim 19.
    for side in arm_sides:
        rr.log(
            f"plots/gripper_{side}",
            rr.SeriesLines(
                colors=[_COLOR_ACTION, _COLOR_STATE],
                names=[f"{side} action", f"{side} observation.state"],
                widths=[2.0, 2.0],
            ),
            static=True,
        )

    # Camera transform and pinhole are logged per-frame in the loop below
    # because the extrinsic/intrinsic can change each frame.

    # ── robot link geometry: log ONCE as static, then animate per-frame via
    #    Transform3D in the loop below. Re-logging full Mesh3D every frame costs
    #    ~42 MB/frame (39 links, 1.85M verts); static geometry collapses that to
    #    a one-time cost. Pixel identical — verified 0 m vertex diff.
    robot_link_names = list(robot_mesh_gen.meshes.keys())
    for name in robot_link_names:
        m = robot_mesh_gen.meshes[name]
        rr.log(
            f"world/robot/{name}",
            rr.Mesh3D(
                vertex_positions=np.asarray(m.vertices, dtype=np.float32),
                triangle_indices=np.asarray(m.faces, dtype=np.uint32),
            ),
            static=True,
        )

    # ── Per-frame loop ───────────────────────────────────────────────────────
    # Sequence (idx) timeline; playback rate is set by ``TimePanel(fps=...)``
    # above so the viewer advances at ``_PLAYBACK_FPS`` at 1× speed. The trailing
    # ``[N]`` index drives the end-of-episode marker clear regardless of stride.
    for idx in list(range(0, N, args.stride)) + [N]:
        rr.set_time("frame", sequence=idx)

        if idx == N:
            if eef_enabled:
                rr.log("world/action", rr.Clear(recursive=True))
                rr.log("world/state", rr.Clear(recursive=True))
                rr.log("world/camera/eef_state_2d", rr.Clear(recursive=False))
                rr.log("world/camera/position_condition_2d", rr.Clear(recursive=False))
                rr.log("world/camera/position_condition_vector_2d", rr.Clear(recursive=False))
            continue

        frame = dataset_rgb[idx]

        # Per-frame camera pose and intrinsics from calib sidecar.
        world_t_cam = extrinsic_stack[idx]          # (4, 4) world_T_cam
        K = intrinsic_stack[idx]                     # (3, 3) already at stored resolution
        K32 = K.astype(np.float32)
        rr.log(
            "world/camera",
            rr.Transform3D(translation=world_t_cam[:3, 3], mat3x3=world_t_cam[:3, :3]),
        )
        rr.log(
            "world/camera",
            rr.Pinhole(image_from_camera=K32, width=W, height=H),
        )

        if eef_enabled:
            assert action is not None and state_eef is not None
            # Future-trail markers + current pose frames, per arm (action red, state blue).
            for side in arm_sides:
                b = eef_base[side]
                rr.log(
                    f"world/action/{side}/marker",
                    rr.Points3D(action_pos[side][idx:], colors=[_COLOR_ACTION], radii=0.006),
                )
                rr.log(
                    f"world/action/{side}/frame",
                    rr.Transform3D(
                        translation=action_pos[side][idx],
                        mat3x3=gram_schmidt_6d_to_R(action[idx, b + 3:b + 9]),
                    ),
                )
                rr.log(
                    f"world/state/{side}/marker",
                    rr.Points3D(state_eef_pos[side][idx:], colors=[_COLOR_STATE], radii=0.006),
                )
                rr.log(
                    f"world/state/{side}/frame",
                    rr.Transform3D(
                        translation=state_eef_pos[side][idx],
                        mat3x3=gram_schmidt_6d_to_R(state_eef[idx, b + 3:b + 9]),
                    ),
                )

        # Head RGB from parquet; depth from sidecar (both under world/camera).
        rgb = _rgb_to_hwc(frame["observation.images.head_rgb"])
        depth_mm = depth_stack[idx]
        rr.log("world/camera/rgb", rr.Image(rgb))
        rr.log("world/camera/depth", rr.DepthImage(depth_mm, meter=1000.0))

        # Wrist RGB: plain 2D panel (no calibration/depth for the wrist camera).
        wrist_rgb = _rgb_to_hwc(frame["observation.images.wrist_rgb"])
        rr.log("wrist/rgb", rr.Image(wrist_rgb))

        if eef_enabled:
            assert position_condition is not None
            uv_pts = []
            for side in arm_sides:
                uv, uv_valid = project_world_to_pixel(
                    state_eef_pos[side][idx : idx + 1], K, world_t_cam
                )
                if uv_valid[0]:
                    uv_pts.append(uv[0])
            if uv_pts:
                rr.log(
                    "world/camera/eef_state_2d",
                    rr.Points2D(
                        np.asarray(uv_pts, dtype=np.float32), colors=[_COLOR_STATE], radii=2.0
                    ),
                )
            else:
                rr.log("world/camera/eef_state_2d", rr.Clear(recursive=False))

            condition_xyz = position_condition[idx].reshape(2, 3)
            condition_uv, condition_valid = project_world_to_pixel(
                condition_xyz, K, world_t_cam
            )
            valid_condition_idx = np.flatnonzero(condition_valid)
            if len(valid_condition_idx) > 0:
                condition_colors = np.asarray(
                    [_COLOR_CONDITION_BEFORE, _COLOR_CONDITION_AFTER], dtype=np.uint8
                )[valid_condition_idx]
                rr.log(
                    "world/camera/position_condition_2d",
                    rr.Points2D(
                        condition_uv[valid_condition_idx],
                        colors=condition_colors,
                        radii=np.full(len(valid_condition_idx), 3.0, dtype=np.float32),
                    ),
                )
            else:
                rr.log("world/camera/position_condition_2d", rr.Clear(recursive=False))
            if np.all(condition_valid):
                rr.log(
                    "world/camera/position_condition_vector_2d",
                    rr.LineStrips2D(
                        [condition_uv.astype(np.float32)],
                        colors=[_COLOR_CONDITION_LINE],
                        radii=1.5,
                    ),
                )
            else:
                rr.log(
                    "world/camera/position_condition_vector_2d",
                    rr.Clear(recursive=False),
                )

        # Point cloud (manual unproject + voxel downsample).
        depth_m = depth_mm.astype(np.float32) / 1000.0
        pts, mask = unproject_depth(depth_m, K, world_t_cam)
        cols = rgb[mask]
        pts, cols = voxel_downsample(pts, cols, args.voxel)
        rr.log("world/pcd", rr.Points3D(pts, colors=cols, radii=0.003))

        # Robot pose: each PRESENT arm follows the joint variant; an arm absent from a
        # single-arm dataset is pinned to its INIT_*_ARM_JOINTS (like torso/head) so the
        # full URDF still renders. Geometry is static (logged once above); here we emit
        # only per-link transforms.
        def _arm_vals(side: str) -> list[float]:
            if side in arm_sides:
                return state_joint[idx, joint_arm_slice[side]].tolist()
            return list(_INIT_ARM_JOINTS[side])

        joint_vals = (
            [0.0] * len(_WHEEL_NAMES)
            + list(INIT_TORSO_JOINTS)
            + _arm_vals("left")
            + _arm_vals("right")
            + list(INIT_HEAD_JOINTS)
        )
        joint_names = _WHEEL_NAMES + _OBS_NAMES
        qpos = robot_mesh_gen.convert_to_sapien_joint_order(
            np.array(joint_vals), joint_names
        )
        link_tf = robot_mesh_gen.compute_fk_from_link_names(
            qpos, robot_link_names, in_obj_frame=True
        )
        for name in robot_link_names:
            tf = link_tf[name]
            rr.log(
                f"world/robot/{name}",
                rr.Transform3D(translation=tf[:3, 3], mat3x3=tf[:3, :3]),
            )

        # Gripper scalars per arm: [action[grip], observation.state[grip]].
        for side in arm_sides:
            if eef_enabled:
                assert action is not None and state_eef is not None
                g = eef_grip_idx[side]
                rr.log(
                    f"plots/gripper_{side}",
                    rr.Scalars([float(action[idx, g]), float(state_eef[idx, g])]),
                )
            else:
                g = joint_grip_idx[side]
                rr.log(
                    f"plots/gripper_{side}",
                    rr.Scalars([float(action_joint[idx, g]), float(state_joint[idx, g])]),
                )

    if eef_figs:
        plt.show()


if __name__ == "__main__":
    main()
