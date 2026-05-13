"""Compare a GT teleoperation episode against a policy rollout in rerun.

Renders the INFER world (RGB-D, point cloud, robot meshes — mirroring
``scripts/vis_episode.py``) and overlays the remaining GT / INFER EEF
trajectory markers. Per the user's spec, markers represent the *future*
trajectory from each frame onwards, so they "disappear one at a time" from
the start as the scrubber advances.

Computed / loaded:

- GT:    ``gt_eef_9d`` via FK on ``/obs/joint/{torso, right_arm}`` (mirrors
         ``port_dexmate_hdf5.py``).
- INFER: ``/action/eef_9d/right`` (green markers), ``/obs/eef_9d/right``
         (blue markers), ``/obs/gripper/right`` (blue scalar plot),
         ``/obs/joint/right_arm`` (used for FK to render the policy's actual
         robot pose), ``/obs/images/{left_rgb, depth}``.

INFER doesn't record torso / left_arm / head joints because
``policy_rollout.py`` holds them at ``INIT_*_JOINTS`` for the entire rollout,
so we plug those constants in for FK + robot-mesh rendering. ZED intrinsics
are hardcoded (same fallback as vis_episode.py).

Usage::

    python scripts/vis_episode_online.py \\
        --gt /home/yixuan/Dexmate/data/raw_data_renamed/test/episode_3.hdf5 \\
        --infer /home/yixuan/omniteleop/Dexmate/deploy/act_abs_eef_eef/4/episode_0.hdf5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import torch
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

# Same wheel / obs ordering as vis_episode.py.
_WHEEL_NAMES = ["B_wheel_j1", "B_wheel_j2", "R_wheel_j1", "R_wheel_j2", "L_wheel_j1", "L_wheel_j2"]
_OBS_NAMES = (
    list(_TORSO_JOINTS) + list(_LEFT_ARM_JOINTS) + list(_RIGHT_ARM_JOINTS) + list(_HEAD_JOINTS)
)


def mat_to_pos6d(M: np.ndarray) -> np.ndarray:
    """Convert (4, 4) SE(3) to 9D ``[tx, ty, tz, R[:,0], R[:,1]]`` (matches port_dexmate_hdf5)."""
    if M.shape != (4, 4):
        raise ValueError(f"expected (4, 4) SE(3) matrix, got shape {M.shape}")
    pos = M[:3, 3]
    R = M[:3, :3]
    r6 = np.concatenate([R[:, 0], R[:, 1]])
    return np.concatenate([pos, r6]).astype(np.float32)


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


def compute_gt_eef_9d(gt_path: str, kin: _Kin) -> np.ndarray:
    """Load GT joints and return ``(T, 9)`` eef_9d via right-arm FK."""
    with h5py.File(gt_path, "r") as f:
        torso = np.array(f["/obs/joint/torso"], dtype=np.float64)
        right_arm = np.array(f["/obs/joint/right_arm"], dtype=np.float64)
    if torso.shape[0] != right_arm.shape[0]:
        raise ValueError(
            f"GT frame counts disagree: torso={torso.shape[0]}, right_arm={right_arm.shape[0]}"
        )
    if torso.ndim != 2 or torso.shape[1] != 3:
        raise ValueError(f"GT torso shape: expected (T, 3), got {torso.shape}")
    if right_arm.ndim != 2 or right_arm.shape[1] != 7:
        raise ValueError(f"GT right_arm shape: expected (T, 7), got {right_arm.shape}")

    T = torso.shape[0]
    eef_9d = np.zeros((T, 9), dtype=np.float32)
    zero_l = np.zeros(7)
    zero_h = np.zeros(3)
    for t in range(T):
        M = kin.fk_link(kin.eef_idx, torso[t], zero_l, right_arm[t], zero_h)
        eef_9d[t] = mat_to_pos6d(M)
    return eef_9d


def report_pipeline_latencies(infer_path: str) -> None:
    """Save per-frame latency curves from ``time/*`` if recorded.

    Writes ``<dirname(infer_path)>/debug/latency.png`` with four overlaid curves
    (build_observation, inference, send, end_to_end) in milliseconds vs frame.
    Frame 0 is discarded because it captures cold-start costs (CUDA init, first
    model upload, autotune) that don't reflect steady-state latency. Frames
    where either side of a delta is 0 are masked to NaN (gaps in plot). The
    mean of each curve is appended to its legend label.
    """
    keys = (
        "begin_build_observation",
        "begin_inference",
        "finish_inference",
        "publish_command",
    )
    with h5py.File(infer_path, "r") as f:
        t_group = f.get("time")
        if t_group is None or not all(k in t_group for k in keys):
            return
        stamps = {k: np.array(t_group[k], dtype=np.int64) for k in keys}

    bbo = stamps["begin_build_observation"]
    bi = stamps["begin_inference"]
    fi = stamps["finish_inference"]
    pc = stamps["publish_command"]
    N = bbo.shape[0]
    if N < 2:
        return

    def delta_ms(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mask = (a > 0) & (b > 0)
        out = np.full(a.shape[0], np.nan, dtype=np.float64)
        out[mask] = (b[mask].astype(np.float64) - a[mask].astype(np.float64)) * 1e-6
        return out

    curves = {
        "build_observation (bbo → bi)": delta_ms(bbo, bi),
        "inference         (bi  → fi)": delta_ms(bi, fi),
        "send              (fi  → pc)": delta_ms(fi, pc),
        "end_to_end        (bbo → pc)": delta_ms(bbo, pc),
    }

    out_dir = Path(infer_path).parent / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "latency.png"

    # Drop the first frame (cold-start spike) before plotting / mean.
    frames = np.arange(1, N)
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, y in curves.items():
        y_plot = y[1:]
        m = float(np.nanmean(y_plot)) if np.isfinite(y_plot).any() else float("nan")
        ax.plot(frames, y_plot, label=f"{label}  mean={m:.2f} ms", linewidth=1.4)
    ax.set_xlabel("frame")
    ax.set_ylabel("latency (ms)")
    ax.set_title(f"Pipeline latencies — {Path(infer_path).name} (frame 0 dropped)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_infer(infer_path: str) -> dict:
    """Read everything the infer pass needs."""
    with h5py.File(infer_path, "r") as f:
        out = {
            "action_eef_9d": np.array(f["/action/eef_9d/right"], dtype=np.float32),
            "obs_eef_9d": np.array(f["/obs/eef_9d/right"], dtype=np.float32),
            "obs_gripper": np.array(f["/obs/gripper/right"], dtype=np.float32),
            "obs_right_arm": np.array(f["/obs/joint/right_arm"], dtype=np.float32),
            "left_rgb": np.array(f["/obs/images/left_rgb"], dtype=np.uint8),
            "depth_mm": np.array(f["/obs/images/depth"]),
        }
    a = out["action_eef_9d"]
    if a.ndim != 2 or a.shape[1] != 9:
        raise ValueError(f"/action/eef_9d/right shape: expected (T, 9), got {a.shape}")
    if out["obs_eef_9d"].ndim != 2 or out["obs_eef_9d"].shape[1] != 9:
        raise ValueError(f"/obs/eef_9d/right shape: expected (T, 9), got {out['obs_eef_9d'].shape}")
    T = a.shape[0]
    for k in ("obs_eef_9d", "obs_gripper", "obs_right_arm", "left_rgb", "depth_mm"):
        if out[k].shape[0] != T:
            raise ValueError(
                f"infer frame counts disagree: action={T}, {k}={out[k].shape[0]}"
            )
    if out["obs_right_arm"].shape[1] != 7:
        raise ValueError(f"obs/joint/right_arm shape: expected (T, 7), got {out['obs_right_arm'].shape}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt",
        type=str,
        default="/home/yixuan/Dexmate/data/raw_data_renamed/test/episode_3.hdf5",
        help="GT HDF5 (uses /obs/joint/{torso, right_arm} for FK).",
    )
    parser.add_argument(
        "--infer",
        type=str,
        default="/home/yixuan/Dexmate/deploy/act_abs_eef_eef/4/episode_0.hdf5",
        help="Inference HDF5 (uses /action/eef_9d/right, /obs/gripper/right, "
             "/obs/joint/right_arm, /obs/images/{left_rgb, depth}).",
    )
    parser.add_argument(
        "--voxel", type=float, default=0.008, help="Voxel size for PCD downsample (m)."
    )
    args = parser.parse_args()

    # Latency report from the policy-rollout's /time group (if present).
    report_pipeline_latencies(args.infer)

    kin = _Kin()

    # ── GT ───────────────────────────────────────────────────────────────────
    gt_eef_9d = compute_gt_eef_9d(args.gt, kin)
    gt_pos = gt_eef_9d[:, :3]
    N_gt = gt_pos.shape[0]

    # ── INFER ────────────────────────────────────────────────────────────────
    inf = load_infer(args.infer)
    infer_action_eef_9d = inf["action_eef_9d"]
    infer_pos = infer_action_eef_9d[:, :3]
    infer_obs_eef_9d = inf["obs_eef_9d"]
    infer_obs_pos = infer_obs_eef_9d[:, :3]
    infer_gripper = inf["obs_gripper"]
    infer_right_arm = inf["obs_right_arm"]
    left_rgb = inf["left_rgb"]
    depth_mm = inf["depth_mm"]
    N_infer = infer_pos.shape[0]
    H, W = left_rgb.shape[1], left_rgb.shape[2]
    print(f"GT: {N_gt} frames  |  INFER: {N_infer} frames  |  infer images: {H}x{W}")

    # Hardcoded ZED intrinsics (same fallback values as vis_episode.py).
    fx = fy = 770.1868 / 2.0
    cx = 990.2711 / 2.0
    cy = 637.7721 / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    K32 = K.astype(np.float32)

    # Fixed non-right-arm joints during the policy rollout.
    infer_torso = np.asarray(INIT_TORSO_JOINTS, dtype=np.float64)
    infer_l_arm = np.asarray(INIT_LEFT_ARM_JOINTS, dtype=np.float64)
    infer_head = np.asarray(INIT_HEAD_JOINTS, dtype=np.float64)

    # Camera (zed_depth_frame) sits on the torso/head chain, which is held at
    # INIT_* for the whole rollout — so extrinsics are constant.
    world_t_cam = kin.fk_link(
        kin.cam_idx, infer_torso, infer_l_arm, infer_right_arm[0].astype(np.float64), infer_head
    )

    robot_mesh_gen = RobotMeshGenerator(_ROBOT_NAME)

    # ── rerun ────────────────────────────────────────────────────────────────
    rr.init("vis_episode_online", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Static style for the gripper time series (blue).
    rr.log(
        "plots/gripper",
        rr.SeriesLines(colors=[(0, 80, 255)], names=["obs/gripper/right"], widths=[2.0]),
        static=True,
    )

    # Static camera pose + intrinsics (camera is on torso/head; INIT_* is constant).
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

    N_end = max(N_gt, N_infer)
    for idx in range(N_end + 1):
        rr.set_time("frame", sequence=idx)

        # ── GT future markers (red): gt_pos[idx:N_gt] ──────────────────────
        if idx < N_gt:
            tail = gt_pos[idx:]
            rr.log(
                "world/gt_eef/marker",
                rr.Points3D(tail, colors=[(255, 0, 0)], radii=0.006),
            )
            R = gram_schmidt_6d_to_R(gt_eef_9d[idx, 3:9])
            rr.log(
                "world/gt_eef/frame",
                rr.Transform3D(translation=gt_pos[idx], mat3x3=R),
            )
        elif idx == N_gt:
            rr.log("world/gt_eef", rr.Clear(recursive=True))

        # ── INFER future markers ───────────────────────────────────────────
        # green: /action/eef_9d/right (policy command tail)
        # blue : /obs/eef_9d/right    (achieved EEF tail)
        if idx < N_infer:
            tail_act = infer_pos[idx:]
            rr.log(
                "world/infer_eef/marker",
                rr.Points3D(tail_act, colors=[(0, 255, 0)], radii=0.006),
            )
            R = gram_schmidt_6d_to_R(infer_action_eef_9d[idx, 3:9])
            rr.log(
                "world/infer_eef/frame",
                rr.Transform3D(translation=infer_pos[idx], mat3x3=R),
            )

            tail_obs = infer_obs_pos[idx:]
            rr.log(
                "world/infer_obs_eef/marker",
                rr.Points3D(tail_obs, colors=[(0, 80, 255)], radii=0.006),
            )
            R_obs = gram_schmidt_6d_to_R(infer_obs_eef_9d[idx, 3:9])
            rr.log(
                "world/infer_obs_eef/frame",
                rr.Transform3D(translation=infer_obs_pos[idx], mat3x3=R_obs),
            )

            # ── RGB/D from infer (camera pose is static, set above) ────────
            rr.log("world/camera/rgb", rr.Image(left_rgb[idx]))
            rr.log("world/camera/depth", rr.DepthImage(depth_mm[idx], meter=1000.0))

            # ── Point cloud (manual unproject + voxel downsample) ──────────
            depth_m = depth_mm[idx] / 1000.0
            pts, mask = unproject_depth(depth_m, K, world_t_cam)
            cols = left_rgb[idx][mask]
            pts, cols = voxel_downsample(pts, cols, args.voxel)
            rr.log("world/pcd", rr.Points3D(pts, colors=cols, radii=0.003))

            # ── Robot meshes (right arm from infer, others from INIT_*) ────
            joint_vals = (
                [0.0] * len(_WHEEL_NAMES)
                + list(INIT_TORSO_JOINTS)
                + list(INIT_LEFT_ARM_JOINTS)
                + infer_right_arm[idx].tolist()
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

            # ── Gripper scalar ─────────────────────────────────────────────
            rr.log("plots/gripper", rr.Scalars(float(infer_gripper[idx])))
        elif idx == N_infer:
            # Final boundary: clear the last remaining INFER markers.
            rr.log("world/infer_eef", rr.Clear(recursive=True))
            rr.log("world/infer_obs_eef", rr.Clear(recursive=True))


if __name__ == "__main__":
    main()
