"""Visualise the right-arm EEF trajectory (GT vs. predicted) as marker points.

Builds on ``scripts/vis_episode.py``: same point-cloud / robot-mesh / camera
layout under a single ``frame`` timeline, plus per-frame logging of the
**future** right-EEF trajectory under ``world/eef_traj/{right,right_pred}``.
At frame ``t`` only frames ``t+1..N-1`` are logged, so markers naturally
disappear as the rerun scrubber passes them.

The left arm is fixed in the dataset and is not visualized. Predicted
right-EEF positions come from ``pred_vs_gt.npz`` produced by
``examples/dexmate/infer_dexmate.py``.

Usage::

    python scripts/vis_inference.py \\
        --hdf5 Dexmate/data/raw_data_renamed/test/episode_1.hdf5 \\
        --pred-npz Dexmate/infer/<run_id>/pred_vs_gt.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rerun as rr
import torch
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5
from yixuan_utilities.kinematics_helper import KinHelper
from yixuan_utilities.robot_mesh_generator import RobotMeshGenerator

_WHEEL_NAMES = ["B_wheel_j1", "B_wheel_j2", "R_wheel_j1", "R_wheel_j2", "L_wheel_j1", "L_wheel_j2"]
_OBS_NAMES = [
    "torso_j1",
    "torso_j2",
    "torso_j3",
    "L_arm_j1",
    "L_arm_j2",
    "L_arm_j3",
    "L_arm_j4",
    "L_arm_j5",
    "L_arm_j6",
    "L_arm_j7",
    "R_arm_j1",
    "R_arm_j2",
    "R_arm_j3",
    "R_arm_j4",
    "R_arm_j5",
    "R_arm_j6",
    "R_arm_j7",
    "head_j1",
    "head_j2",
    "head_j3",
]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdf5",
        type=str,
        default="/home/yixuan/omniteleop/Dexmate/data/raw_data/episode_1.hdf5",
        help="Path to episode HDF5 file",
    )
    parser.add_argument(
        "--voxel", type=float, default=0.008, help="Voxel size for PCD downsample (m)"
    )
    parser.add_argument(
        "--marker-radius",
        type=float,
        default=0.005,
        help="Radius (m) for trajectory marker points",
    )
    parser.add_argument(
        "--pred-npz",
        type=str,
        default=None,
        help="Path to pred_vs_gt.npz produced by examples/dexmate/infer_dexmate.py.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode index in the LeRobot dataset; if omitted, parsed from --hdf5 filename.",
    )
    args = parser.parse_args()

    data, _ = load_dict_from_hdf5(args.hdf5)

    # ── load arrays ──────────────────────────────────────────────────────────
    left_rgb = np.array(data["obs"]["images"]["left_rgb"]).astype(np.uint8)
    right_rgb = np.array(data["obs"]["images"]["right_rgb"]).astype(np.uint8)
    depth_mm = np.array(data["obs"]["images"]["depth"])

    obs_torso = np.array(data["obs"]["joint"]["torso"])
    obs_left_arm = np.array(data["obs"]["joint"]["left_arm"])
    obs_right_arm = np.array(data["obs"]["joint"]["right_arm"])
    obs_head = np.array(data["obs"]["joint"]["head"])

    eef_right = np.array(data["action"]["eef"]["right"])  # (N, 4, 4)

    # Translation-only marker positions; finite mask filters bypassed-IK frames.
    eef_right_xyz = eef_right[:, :3, 3].astype(np.float32)
    right_valid = np.isfinite(eef_right_xyz).all(axis=1)

    N = left_rgb.shape[0]
    H, W = left_rgb.shape[1], left_rgb.shape[2]
    print(f"Episode: {N} frames  |  images: {H}x{W}")

    # Optional: overlay predicted right-EEF trajectory from infer_dexmate.py.
    pred_xyz: np.ndarray | None = None
    pred_valid: np.ndarray | None = None
    if args.pred_npz:
        npz = np.load(args.pred_npz)
        ep = args.episode
        if ep is None:
            stem = Path(args.hdf5).stem  # e.g. "episode_3"
            ep = int(stem.split("_")[-1])
        ep_ids = npz["episode_ids"].tolist()
        if ep not in ep_ids:
            raise ValueError(f"Episode {ep} not in npz episode_ids {ep_ids}")
        idx = ep_ids.index(ep)
        lengths = npz["episode_lengths"]
        start = int(lengths[:idx].sum())
        end = start + int(lengths[idx])
        if end - start != N:
            raise ValueError(
                f"Frame count mismatch: HDF5 has {N}, npz episode {ep} has {end - start}"
            )
        pred_xyz = npz["pred"][start:end, :3].astype(np.float32)
        pred_valid = np.isfinite(pred_xyz).all(axis=1)
        print(f"Loaded predictions: ep={ep}, frames={end - start}")

    # ── intrinsics ───────────────────────────────────────────────────────────
    img_group = data["obs"]["images"]
    if "intrinsic" in img_group:
        K_arr = np.array(img_group["intrinsic"])
        K = K_arr[0] if K_arr.ndim == 3 else K_arr
        K = K.astype(np.float64)
    else:
        fx = fy = 770.1868 / 2.0
        cx = 990.2711 / 2.0
        cy = 637.7721 / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # ── extrinsics ───────────────────────────────────────────────────────────
    kin = KinHelper("vega_no_effector")
    joint_names_kin = [j.name for j in kin.sapien_robot.get_active_joints()]
    name_to_idx_kin = {n: i for i, n in enumerate(joint_names_kin)}
    cam_link_idx = kin.link_name_to_idx["zed_depth_frame"]

    if "extrinsic" in img_group:
        extrinsics = np.array(img_group["extrinsic"]).astype(np.float64)
        print("Using saved obs/images/extrinsic")
    else:
        print("No saved extrinsic — computing per-frame FK ...")
        extrinsics = np.zeros((N, 4, 4), dtype=np.float64)
        for idx in range(N):
            qpos = np.zeros(kin.sapien_robot.dof)
            joint_vals = (
                obs_torso[idx].tolist()
                + obs_left_arm[idx].tolist()
                + obs_right_arm[idx].tolist()
                + obs_head[idx].tolist()
            )
            for name, val in zip(_OBS_NAMES, joint_vals, strict=True):
                if name in name_to_idx_kin:
                    qpos[name_to_idx_kin[name]] = val
            extrinsics[idx] = kin.compute_fk_from_link_idx(qpos, [cam_link_idx])[0]

    robot_mesh_gen = RobotMeshGenerator("vega_no_effector")

    # ── rerun ────────────────────────────────────────────────────────────────
    rr.init("vis_inference", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    K32 = K.astype(np.float32)

    RIGHT_COLOR = np.array([255, 80, 80], dtype=np.uint8)  # red — right EEF future (GT)
    PRED_COLOR = np.array([60, 220, 80], dtype=np.uint8)  # green — predicted right EEF

    for idx in range(N):
        rr.set_time("frame", sequence=idx)

        world_t_cam = extrinsics[idx]

        # ── camera ──────────────────────────────────────────────────────────
        rr.log(
            "world/camera",
            rr.Transform3D(translation=world_t_cam[:3, 3], mat3x3=world_t_cam[:3, :3]),
        )
        rr.log("world/camera", rr.Pinhole(image_from_camera=K32, width=W, height=H))
        rr.log("world/camera/rgb", rr.Image(left_rgb[idx]))
        rr.log("world/camera/depth", rr.DepthImage(depth_mm[idx], meter=1000.0))
        rr.log("image/right_rgb", rr.Image(right_rgb[idx]))

        # ── colored point cloud ─────────────────────────────────────────────
        depth_m = depth_mm[idx] / 1000.0
        pts, mask = unproject_depth(depth_m, K, world_t_cam)
        cols = left_rgb[idx][mask]
        pts, cols = voxel_downsample(pts, cols, args.voxel)
        rr.log("world/pcd", rr.Points3D(pts, colors=cols, radii=0.003))

        # ── robot meshes ────────────────────────────────────────────────────
        joints_vals = (
            [0.0] * len(_WHEEL_NAMES)
            + obs_torso[idx].tolist()
            + obs_left_arm[idx].tolist()
            + obs_right_arm[idx].tolist()
            + obs_head[idx].tolist()
        )
        joint_names = _WHEEL_NAMES + _OBS_NAMES
        joints_arr = np.array(joints_vals)
        joints_arr = robot_mesh_gen.convert_to_sapien_joint_order(joints_arr, joint_names)
        meshes = robot_mesh_gen.compute_robot_meshes(joints_arr)
        for i, mesh in enumerate(meshes):
            rr.log(
                f"world/robot/link_{i}",
                rr.Mesh3D(
                    vertex_positions=np.asarray(mesh.vertices, dtype=np.float32),
                    triangle_indices=np.asarray(mesh.faces, dtype=np.uint32),
                ),
            )

        # ── current right-EEF coordinate frame (skip NaN frames) ────────────
        mat = eef_right[idx]
        if np.all(np.isfinite(mat)):
            rr.log(
                "world/eef/right",
                rr.Transform3D(translation=mat[:3, 3], mat3x3=mat[:3, :3]),
            )
        else:
            rr.log("world/eef/right", rr.Clear(recursive=True))

        # ── future right-EEF trajectory markers (frames idx+1..N-1) ─────────
        # Re-logging the same path each frame replaces the prior payload, so
        # markers behind the scrubber drop out without explicit cleanup.
        future_r = eef_right_xyz[idx + 1 :][right_valid[idx + 1 :]]
        if future_r.size:
            rr.log(
                "world/eef_traj/right",
                rr.Points3D(future_r, colors=RIGHT_COLOR, radii=args.marker_radius),
            )
        else:
            rr.log("world/eef_traj/right", rr.Clear(recursive=False))

        if pred_xyz is not None:
            future_p = pred_xyz[idx + 1 :][pred_valid[idx + 1 :]]
            if future_p.size:
                rr.log(
                    "world/eef_traj/right_pred",
                    rr.Points3D(future_p, colors=PRED_COLOR, radii=args.marker_radius),
                )
            else:
                rr.log("world/eef_traj/right_pred", rr.Clear(recursive=False))


if __name__ == "__main__":
    main()
