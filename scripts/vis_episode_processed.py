"""Compare ``action`` vs ``observation.state`` from processed LeRobotDataset variants in rerun.

Sibling to ``scripts/vis_episode_online.py`` — same rendering structure, but
sourced entirely from processed parquet variants produced by
``examples/port_datasets/port_dexmate_hdf5.py`` (no raw HDF5 input).

Two variants of the same source episode are loaded in lockstep:

- ``--marker_and_plot_dir`` (the ``dexmate_eef_eef`` variant root):
    * ``observation.state`` (N, 10) — FK on the OBSERVED arm joints (pos6d +
      raw gripper); the EEF pose the robot actually reached.
    * ``action`` (N, 10) — FK on the COMMANDED arm joints (pos6d + binary
      gripper); identical to the joint_eef action by construction.
    * Camera scene: ``observation.images.left_rgb`` + depth sidecar drive
      the rerun RGB / depth / point-cloud panels.

- ``--urdf_joint_motion_dir`` (the ``dexmate_joint_eef`` variant root):
    * ``observation.state`` (N, 8) — right-arm joints + raw gripper;
      ``state[:7]`` drives the URDF mesh visualization.

What this shows (matches the README "What ``vis_episode_processed.py`` shows" block):

- **EEF position** — red markers = ``action``[:3]; blue markers =
  ``observation.state``[:3] (eef variant).
- **Robot URDF** — pose follows ``observation.state`` from the joint_eef
  variant (``state[:7]`` feeds right-arm joints, the rest pinned at INIT_*).
- **Gripper** — red curve = ``action`` gripper (binary commanded); blue
  curve = ``observation.state`` gripper (raw observed).
- **Matplotlib EEF xyz** — red = ``action``, blue = ``observation.state``;
  per-axis cross-correlation lag (positive = state lags action) annotated.

A small per-frame divergence is expected (controller tracking lag); large
divergence indicates a port-time or controller bug.

Camera pose/intrinsics are NOT read from the parquet (this script predates
the extrinsic/intrinsic columns and still uses the static fallback): hardcoded
ZED intrinsics + FK on ``zed_depth_frame`` using
``INIT_TORSO_JOINTS / INIT_LEFT_ARM_JOINTS / INIT_HEAD_JOINTS``.

Depth: NOT in the parquet (would be auto-picked-up as a VISUAL policy input).
Read instead from the sidecar at
``<variant_root>/debug/depth/episode_<idx>.npz`` (key ``"depth"``, shape
``(T, H, W)`` uint16). The porter writes it in lockstep with the parquet.

Usage::

    python scripts/vis_episode_processed.py \\
        --marker_and_plot_dir /home/yixuan/omniteleop/Dexmate/data/processed_data/train/dexmate_eef_eef \\
        --urdf_joint_motion_dir /home/yixuan/omniteleop/Dexmate/data/processed_data/train/dexmate_joint_eef \\
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
    INIT_TORSO_JOINTS,
)

_ROBOT_NAME = "vega_no_effector"
_RIGHT_ARM_EEF_LINK = "R_ee"
_CAM_LINK = "zed_depth_frame"
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
        self.cam_idx = self.kin.link_name_to_idx[_CAM_LINK]
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
            f"EEF {axis_name}"
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
        required=True,
        help="Path to the dexmate_eef_eef variant root. Provides "
             "observation.state (eef) + action for the rerun markers, gripper "
             "curves, matplotlib EEF xyz figures, and the RGB / depth / "
             "point-cloud camera scene.",
    )
    parser.add_argument(
        "--urdf_joint_motion_dir",
        type=str,
        required=True,
        help="Path to the dexmate_joint_eef variant root. Provides "
             "observation.state (joint) so state[:7] drives the URDF mesh "
             "right-arm pose.",
    )
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument(
        "--voxel", type=float, default=0.008, help="Voxel size for PCD downsample (m)."
    )
    args = parser.parse_args()

    eef_root   = Path(args.marker_and_plot_dir)
    joint_root = Path(args.urdf_joint_motion_dir)
    if not eef_root.is_dir():
        raise FileNotFoundError(f"--marker_and_plot_dir not a directory: {eef_root}")
    if not joint_root.is_dir():
        raise FileNotFoundError(f"--urdf_joint_motion_dir not a directory: {joint_root}")

    # ── Load the requested episode from both variants in lockstep ────────────
    # return_uint8=True: video frames default to float32 [0, 1]; we need raw
    # uint8 for direct rerun logging (and to avoid the all-zero astype trap).
    dataset_eef = LeRobotDataset(
        repo_id=eef_root.name,
        root=eef_root,
        episodes=[args.episode_index],
        return_uint8=True,
    )
    dataset_joint = LeRobotDataset(
        repo_id=joint_root.name,
        root=joint_root,
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

    # Pre-collect numeric trails so future-trail markers are O(1) per frame.
    state_eef   = np.stack([dataset_eef[i]["observation.state"].numpy()   for i in range(N)])  # (N, 10)
    action      = np.stack([dataset_eef[i]["action"].numpy()              for i in range(N)])  # (N, 10)
    state_joint = np.stack([dataset_joint[i]["observation.state"].numpy() for i in range(N)])  # (N, 8)

    # Depth sidecar (NOT in the parquet — see docstring). Read from
    # marker_and_plot_dir; both variants write the same sidecar content per
    # (variant, episode).
    depth_sidecar = (
        eef_root / "debug" / "depth" / f"episode_{args.episode_index:06d}.npz"
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

    if state_joint.shape[1] != 8:
        raise ValueError(
            f"--urdf_joint_motion_dir observation.state: expected (N, 8), got {state_joint.shape}"
        )
    if state_eef.shape[1] != 10:
        raise ValueError(
            f"--marker_and_plot_dir observation.state: expected (N, 10), got {state_eef.shape}"
        )
    if action.shape[1] != 10:
        raise ValueError(
            f"--marker_and_plot_dir action: expected (N, 10), got {action.shape}"
        )

    action_pos    = action[:, :3]
    state_eef_pos = state_eef[:, :3]

    # ── EEF xyz figures: action vs observation.state with lag ────────────────
    # Built before the rerun loop so they pop up immediately. ``plt.show()`` at
    # the end of main blocks until the user closes the windows; the spawned
    # rerun viewer is a separate process and is unaffected.
    dt_s = 1.0 / float(dataset_eef.fps) if dataset_eef.fps else 1.0 / 30.0
    t_s = np.arange(N, dtype=np.float64) * dt_s
    eef_figs = _build_eef_xyz_figs(t_s, action_pos, state_eef_pos, dt_s)
    print(
        f"built {len(eef_figs)} EEF xyz figures "
        f"(dt = {dt_s*1000:.1f} ms, fps = {1.0/dt_s:.1f} Hz)"
    )

    # ── Camera intrinsics ────────────────────────────────────────────────────
    # Hardcoded ZED calibration is for the *raw* 600×960 HDF5 frame (verified
    # against the stored /obs/images/intrinsic). The porter resizes to 240×320,
    # so we rescale fx,cx by W/RAW_W and fy,cy by H/RAW_H. Without this the
    # unprojected point cloud is ~3× too narrow and drifts away from the
    # robot mesh in world space.
    RAW_H, RAW_W = 600, 960
    sample_rgb = _rgb_to_hwc(dataset_eef[0]["observation.images.left_rgb"])
    H, W = sample_rgb.shape[:2]
    sx, sy = W / RAW_W, H / RAW_H
    fx = (770.1868 / 2.0) * sx
    fy = (770.1868 / 2.0) * sy
    cx = (990.2711 / 2.0) * sx
    cy = (637.7721 / 2.0) * sy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    K32 = K.astype(np.float32)
    print(f"episode {args.episode_index}: {N} frames  |  rgb: {H}x{W}  |  K rescaled by ({sx:.3f}, {sy:.3f})")

    # ── FK for static camera pose and per-frame robot meshes ─────────────────
    kin = _Kin()
    robot_mesh_gen = RobotMeshGenerator(_ROBOT_NAME)
    torso_init = np.asarray(INIT_TORSO_JOINTS, dtype=np.float64)
    l_arm_init = np.asarray(INIT_LEFT_ARM_JOINTS, dtype=np.float64)
    head_init  = np.asarray(INIT_HEAD_JOINTS,  dtype=np.float64)

    # Camera lives on the torso/head chain (both held at INIT_*) → constant.
    world_t_cam = kin.fk_link(
        kin.cam_idx, torso_init, l_arm_init, state_joint[0, :7].astype(np.float64), head_init
    )

    # ── rerun setup ──────────────────────────────────────────────────────────
    rr.init("vis_episode_processed", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Explicit layout: default auto-views are disabled once we send a blueprint,
    # so we must include a 2D view for ``world/camera/rgb`` (and depth) in
    # addition to the 3D world and the gripper time series.
    lookback = max(400, min(N, 5000))
    lookahead = min(80, max(1, N // 50 + 5))
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="world", name="World"),
                    rrb.Spatial2DView(
                        origin="world/camera",
                        name="Camera (RGB / depth)",
                    ),
                    column_shares=[2.0, 1.4],
                ),
                rrb.TimeSeriesView(
                    origin="plots/gripper",
                    name="Gripper",
                    axis_y=rrb.ScalarAxis(
                        range=(_GRIPPER_Y_MIN, _GRIPPER_Y_MAX),
                        zoom_lock=True,
                    ),
                    time_ranges=[
                        rrb.VisibleTimeRange(
                            "frame",
                            start=rrb.TimeRangeBoundary.cursor_relative(
                                seq=-lookback
                            ),
                            end=rrb.TimeRangeBoundary.cursor_relative(seq=lookahead),
                        )
                    ],
                ),
                row_shares=[2.2, 1.0],
            ),
            rrb.TimePanel(timeline="frame", fps=_PLAYBACK_FPS),
        )
    )

    # Two named gripper series sharing one plot entity.
    #   action[9]              (red)  — binarized commanded gripper
    #   observation.state[9]   (blue) — raw obs gripper /obs/gripper/right
    rr.log(
        "plots/gripper",
        rr.SeriesLines(
            colors=[_COLOR_ACTION, _COLOR_STATE],
            names=["action", "observation.state"],
            widths=[2.0, 2.0],
        ),
        static=True,
    )

    rr.log(
        "world/camera",
        rr.Transform3D(translation=world_t_cam[:3, 3], mat3x3=world_t_cam[:3, :3]),
        static=True,
    )
    rr.log(
        "world/camera",
        rr.Pinhole(image_from_camera=K32, width=W, height=H),
        static=True,
    )

    # ── Per-frame loop ───────────────────────────────────────────────────────
    # Sequence (idx) timeline; playback rate is set by ``TimePanel(fps=...)``
    # above so the viewer advances at ``_PLAYBACK_FPS`` at 1× speed.
    for idx in range(N + 1):
        rr.set_time("frame", sequence=idx)

        if idx == N:
            rr.log("world/action", rr.Clear(recursive=True))
            rr.log("world/state",  rr.Clear(recursive=True))
            rr.log("world/camera/eef_state_2d", rr.Clear(recursive=False))
            continue

        frame = dataset_eef[idx]

        # Future-trail markers + current pose frames.
        rr.log(
            "world/action/marker",
            rr.Points3D(action_pos[idx:], colors=[_COLOR_ACTION], radii=0.006),
        )
        rr.log(
            "world/action/frame",
            rr.Transform3D(
                translation=action_pos[idx], mat3x3=gram_schmidt_6d_to_R(action[idx, 3:9])
            ),
        )
        rr.log(
            "world/state/marker",
            rr.Points3D(state_eef_pos[idx:], colors=[_COLOR_STATE], radii=0.006),
        )
        rr.log(
            "world/state/frame",
            rr.Transform3D(
                translation=state_eef_pos[idx], mat3x3=gram_schmidt_6d_to_R(state_eef[idx, 3:9])
            ),
        )

        # RGB from parquet; depth from sidecar (camera pose set statically above).
        rgb = _rgb_to_hwc(frame["observation.images.left_rgb"])
        depth_mm = depth_stack[idx]
        rr.log("world/camera/rgb", rr.Image(rgb))
        rr.log("world/camera/depth", rr.DepthImage(depth_mm, meter=1000.0))

        # Project current observation.state EEF position onto the camera image.
        uv_state, uv_valid = project_world_to_pixel(
            state_eef_pos[idx : idx + 1], K, world_t_cam
        )
        if uv_valid[0]:
            rr.log(
                "world/camera/eef_state_2d",
                rr.Points2D(uv_state, colors=[_COLOR_STATE], radii=2.0),
            )
        else:
            rr.log("world/camera/eef_state_2d", rr.Clear(recursive=False))

        # Point cloud (manual unproject + voxel downsample).
        depth_m = depth_mm.astype(np.float32) / 1000.0
        pts, mask = unproject_depth(depth_m, K, world_t_cam)
        cols = rgb[mask]
        pts, cols = voxel_downsample(pts, cols, args.voxel)
        rr.log("world/pcd", rr.Points3D(pts, colors=cols, radii=0.003))

        # Robot mesh: right arm from state_joint[:7] (dexmate_joint_eef variant),
        # rest pinned at INIT_*. Matches the README "Robot URDF: pose follows
        # observation.state (dexmate_joint_eef)" bullet.
        joint_vals = (
            [0.0] * len(_WHEEL_NAMES)
            + list(INIT_TORSO_JOINTS)
            + list(INIT_LEFT_ARM_JOINTS)
            + state_joint[idx, :7].tolist()
            + list(INIT_HEAD_JOINTS)
        )
        joint_names = _WHEEL_NAMES + _OBS_NAMES
        joints_arr = robot_mesh_gen.convert_to_sapien_joint_order(
            np.array(joint_vals), joint_names
        )
        for i, mesh in enumerate(robot_mesh_gen.compute_robot_meshes(joints_arr)):
            rr.log(
                f"world/robot/link_{i}",
                rr.Mesh3D(
                    vertex_positions=np.asarray(mesh.vertices, dtype=np.float32),
                    triangle_indices=np.asarray(mesh.faces, dtype=np.uint32),
                ),
            )

        # Gripper scalars: [action[9], state_eef[9]] → matches SeriesLines order.
        rr.log(
            "plots/gripper",
            rr.Scalars([float(action[idx, 9]), float(state_eef[idx, 9])]),
        )

    # Block on the matplotlib windows so the user can inspect the EEF curves
    # while the rerun viewer (separate process) continues running.
    plt.show()


if __name__ == "__main__":
    main()
