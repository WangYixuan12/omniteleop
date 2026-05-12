"""Compare `action` vs `action_cmd` from a processed LeRobotDataset variant in rerun.

Sibling to ``scripts/vis_episode_online.py`` — same rendering structure, but
sourced entirely from a single processed parquet variant produced by
``examples/port_datasets/port_dexmate_hdf5.py`` (no raw HDF5 input). The two
trails being compared are:

- ``action``     (blue)  — achieved EEF, FK on /obs/joint/{torso, right_arm}.
- ``action_cmd`` (green) — commanded EEF, FK on the IK target /action/eef/right.

A small per-frame divergence is expected; large divergence indicates a porter
bug. Both gripper dims (``action[9]``, ``action_cmd[9]``) are plotted in the
same gripper view, color-matched to the marker colors.

The variant must be ``dexmate_joint_eef`` (state = right-arm joints + gripper
so we can FK for the robot mesh; action = pos6d + binary gripper so ``action[9]``
exists). Other variants fail the shape asserts with a clear message.

Camera pose/intrinsics are NOT read from the parquet (the existing processed
data was generated before extrinsic/intrinsic columns were added). Same
fallback as the reference: hardcoded ZED intrinsics + FK on ``zed_depth_frame``
using ``INIT_TORSO_JOINTS / INIT_LEFT_ARM_JOINTS / INIT_HEAD_JOINTS``.

Depth: NOT in the parquet (would be auto-picked-up as a VISUAL policy input).
Read instead from the sidecar at
``<variant_root>/debug/depth/episode_<idx>.npz`` (key ``"depth"``, shape
``(T, H, W)`` uint16). The porter writes it in lockstep with the parquet.

Usage::

    python scripts/vis_episode_processed.py \\
        --process_data_subdir /home/yixuan/omniteleop/Dexmate/data/processed_data/test/dexmate_joint_eef \\
        --episode_index 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
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

_COLOR_ACTION_CMD = (0, 255, 0)   # green
_COLOR_ACTION     = (0, 80, 255)  # blue


def gram_schmidt_6d_to_R(r6: np.ndarray) -> np.ndarray:
    """Recover (3, 3) rotation from Zhou et al. 6-D representation."""
    a1, a2 = r6[:3], r6[3:6]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


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


def _local_extrema_indices(y: np.ndarray, min_prominence: float = 0.0) -> np.ndarray:
    """Indices of strict interior local minima/maxima of ``y``.

    Interior sign-changes of the first derivative; plateaus (zero derivative)
    are skipped so flat segments don't generate spurious extrema. If
    ``min_prominence > 0``, drop extrema whose y-value sits within that
    threshold of the previously kept extremum (cheap clutter filter for noisy
    trajectories — not the exact scipy ``peak_prominences`` definition).
    """
    if y.size < 3:
        return np.empty(0, dtype=int)
    dy = np.diff(y)
    sign = np.sign(dy)
    changes = np.where(np.diff(sign) != 0)[0] + 1
    valid = (sign[changes - 1] != 0) & (sign[changes] != 0) & (sign[changes - 1] != sign[changes])
    candidates = changes[valid]
    if min_prominence <= 0 or len(candidates) == 0:
        return candidates.astype(int)
    keep: list[int] = []
    last_y: float | None = None
    for c in candidates:
        cv = float(y[c])
        if last_y is None or abs(cv - last_y) >= min_prominence:
            keep.append(int(c))
            last_y = cv
    return np.array(keep, dtype=int)


def _build_eef_xyz_figs(
    t: np.ndarray,
    cmd_xyz: np.ndarray,
    action_xyz: np.ndarray,
    dt_s: float,
    *,
    min_prominence_m: float = 0.02,
) -> list[plt.Figure]:
    """Three figures (one per axis) comparing ``action_cmd`` vs ``action``.

    For each local extremum of ``action_cmd``, drop a dashed vertical from the
    cmd value to the achieved value and annotate the gap in mm. A single
    cross-correlation lag (action vs action_cmd, velocity-domain) is shown in
    the upper-left of each subplot — same convention as
    ``vis_teleop_curves.fig_eef_xyz``.
    """
    color_cmd    = (0.0, 1.0, 0.0)         # green — action_cmd
    color_action = (0.0, 80 / 255.0, 1.0)  # blue  — action
    figs: list[plt.Figure] = []
    for axis_idx, axis_name in enumerate("xyz"):
        cmd = cmd_xyz[:, axis_idx]
        act = action_xyz[:, axis_idx]
        lag_ms = estimate_lag_ms(cmd, act, dt_s)
        extrema = _local_extrema_indices(cmd, min_prominence=min_prominence_m)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(t, cmd, color=color_cmd,    lw=1.4, label="action_cmd")
        ax.plot(t, act, color=color_action, lw=1.2, label="action")

        for i in extrema:
            dv_mm = float(act[i] - cmd[i]) * 1000.0
            ax.plot([t[i], t[i]], [cmd[i], act[i]], color="0.55", lw=0.7, ls="--")
            ax.plot(t[i], cmd[i], "o", color=color_cmd,    ms=3)
            ax.plot(t[i], act[i], "o", color=color_action, ms=3)
            ymax = max(cmd[i], act[i])
            ax.annotate(
                f"Δ={dv_mm:+.1f} mm",
                xy=(t[i], ymax),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="0.25",
            )

        ax.text(
            0.01, 0.95,
            f"lag = {_format_lag_ms(lag_ms, include_unit=True)}   "
            f"(positive = action lags action_cmd)",
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
        "--process_data_subdir",
        type=str,
        default="/home/yixuan/omniteleop/Dexmate/data/processed_data/test/dexmate_joint_eef",
        help="Path to a processed LeRobotDataset variant root (must be dexmate_joint_eef).",
    )
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument(
        "--voxel", type=float, default=0.008, help="Voxel size for PCD downsample (m)."
    )
    args = parser.parse_args()

    # ── Load the requested episode ───────────────────────────────────────────
    # repo_id is required by the constructor but only used as a label when root
    # is provided; derive it from the variant dir name so it's informative.
    repo_id = Path(args.process_data_subdir).name
    # return_uint8=True: video frames default to float32 [0, 1]; we need raw
    # uint8 for direct rerun logging (and to avoid the all-zero astype trap).
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=args.process_data_subdir,
        episodes=[args.episode_index],
        return_uint8=True,
    )
    N = len(dataset)
    if N == 0:
        raise ValueError(f"Episode {args.episode_index} is empty under {args.process_data_subdir}.")

    # Pre-collect numeric trails so future-trail markers are O(1) per frame.
    state    = np.stack([dataset[i]["observation.state"].numpy()    for i in range(N)])  # (N, 8)
    action   = np.stack([dataset[i]["action"].numpy()               for i in range(N)])  # (N, 10)
    cmd      = np.stack([dataset[i]["action_cmd"].numpy()           for i in range(N)])  # (N, 10)

    # Depth sidecar (NOT in the parquet — see docstring).
    depth_sidecar = (
        Path(args.process_data_subdir)
        / "debug" / "depth" / f"episode_{args.episode_index:06d}.npz"
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

    if action.shape[1] != 10:
        raise ValueError(
            f"action shape: expected (N, 10) for eef-action variant, got {action.shape}. "
            "This script targets dexmate_joint_eef."
        )
    if cmd.shape[1] != 10:
        raise ValueError(f"action_cmd shape: expected (N, 10), got {cmd.shape}")
    if state.shape[1] != 8:
        raise ValueError(
            f"observation.state shape: expected (N, 8) for joint-state variant, got {state.shape}. "
            "This script targets dexmate_joint_eef."
        )

    action_pos = action[:, :3]
    cmd_pos    = cmd[:, :3]

    # ── EEF xyz figures: action_cmd vs action with extrema gaps + lag ────────
    # Built before the rerun loop so they pop up immediately. ``plt.show()`` at
    # the end of main blocks until the user closes the windows; the spawned
    # rerun viewer is a separate process and is unaffected.
    dt_s = 1.0 / float(dataset.fps) if dataset.fps else 1.0 / 30.0
    t_s = np.arange(N, dtype=np.float64) * dt_s
    eef_figs = _build_eef_xyz_figs(t_s, cmd_pos, action_pos, dt_s)
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
    sample_rgb = _rgb_to_hwc(dataset[0]["observation.images.left_rgb"])
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
        kin.cam_idx, torso_init, l_arm_init, state[0, :7].astype(np.float64), head_init
    )

    # ── rerun setup ──────────────────────────────────────────────────────────
    rr.init("vis_episode_processed", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Two named gripper series sharing one plot entity, color-keyed to markers.
    rr.log(
        "plots/gripper",
        rr.SeriesLines(
            colors=[_COLOR_ACTION_CMD, _COLOR_ACTION],
            names=["action_cmd", "action"],
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
    for idx in range(N + 1):
        rr.set_time("frame", sequence=idx)

        if idx == N:
            rr.log("world/action_cmd", rr.Clear(recursive=True))
            rr.log("world/action",     rr.Clear(recursive=True))
            continue

        frame = dataset[idx]

        # Future-trail markers + current pose frames.
        rr.log(
            "world/action_cmd/marker",
            rr.Points3D(cmd_pos[idx:], colors=[_COLOR_ACTION_CMD], radii=0.006),
        )
        rr.log(
            "world/action_cmd/frame",
            rr.Transform3D(
                translation=cmd_pos[idx], mat3x3=gram_schmidt_6d_to_R(cmd[idx, 3:9])
            ),
        )
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

        # RGB from parquet; depth from sidecar (camera pose set statically above).
        rgb = _rgb_to_hwc(frame["observation.images.left_rgb"])
        depth_mm = depth_stack[idx]
        rr.log("world/camera/rgb", rr.Image(rgb))
        rr.log("world/camera/depth", rr.DepthImage(depth_mm, meter=1000.0))

        # Point cloud (manual unproject + voxel downsample).
        depth_m = depth_mm.astype(np.float32) / 1000.0
        pts, mask = unproject_depth(depth_m, K, world_t_cam)
        cols = rgb[mask]
        pts, cols = voxel_downsample(pts, cols, args.voxel)
        rr.log("world/pcd", rr.Points3D(pts, colors=cols, radii=0.003))

        # Robot mesh: right arm from state[:7], rest pinned at INIT_*.
        joint_vals = (
            [0.0] * len(_WHEEL_NAMES)
            + list(INIT_TORSO_JOINTS)
            + list(INIT_LEFT_ARM_JOINTS)
            + state[idx, :7].tolist()
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

        # Gripper scalars: [action_cmd[9], action[9]] → matches SeriesLines order.
        rr.log(
            "plots/gripper",
            rr.Scalars([float(cmd[idx, 9]), float(action[idx, 9])]),
        )

    # Block on the matplotlib windows so the user can inspect the EEF curves
    # while the rerun viewer (separate process) continues running.
    plt.show()


if __name__ == "__main__":
    main()
