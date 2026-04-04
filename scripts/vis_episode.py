"""Visualise a recorded VR teleoperation episode.

Usage::

    python scripts/vis_episode.py data/episode_0.hdf5
    python scripts/vis_episode.py data/episode_0.hdf5 --fps 10 --voxel 0.01
    python scripts/vis_episode.py data/episode_0.hdf5 --fx 525 --fy 525 --cx 480 --cy 300
"""

from __future__ import annotations

import argparse
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import transforms3d as t3d
import viser
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5
from yixuan_utilities.kinematics_helper import KinHelper
from yixuan_utilities.robot_mesh_generator import RobotMeshGenerator

# ── joint name order expected by convert_to_sapien_joint_order ───────────────
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


def mat_to_wxyz_pos(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wxyz = t3d.quaternions.mat2quat(mat[:3, :3])
    return wxyz, mat[:3, 3]


def unproject_depth(
    depth_m: np.ndarray, K: np.ndarray, world_t_cam: np.ndarray, max_depth: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a depth image to a colored point cloud in world frame.

    Args:
        depth_m:     (H, W) float32, metres
        K:           (3, 3) camera intrinsics
        world_t_cam: (4, 4) camera pose in world frame
        max_depth:   discard points beyond this distance

    Returns:
        pts:    (M, 3) float32 world-frame XYZ
    """
    H, W = depth_m.shape
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")  # (H,W) each
    mask = (depth_m > 0.05) & (depth_m < max_depth)
    z = depth_m[mask]
    x = (u[mask] - K[0, 2]) / K[0, 0] * z
    y = (v[mask] - K[1, 2]) / K[1, 1] * z
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)  # (M, 4)
    pts_world = (world_t_cam @ pts_cam.T).T[:, :3]  # (M, 3)
    return pts_world.astype(np.float32), mask


def voxel_downsample(
    pts: np.ndarray, colors: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Simple voxel downsample using torch if CUDA available, else numpy."""
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
        # Fallback: random subsample
        idx = np.random.choice(len(pts), min(len(pts), 50000), replace=False)
        return pts[idx], colors[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdf5", type=str, default="data/episode_0.hdf5", help="Path to episode HDF5 file"
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Playback speed (Hz)")
    parser.add_argument(
        "--voxel", type=float, default=0.008, help="Voxel size for PCD downsample (m)"
    )
    args = parser.parse_args()

    data, _ = load_dict_from_hdf5(args.hdf5)
    fx = 770.1868 / 2.0
    fy = 770.1868 / 2.0
    cx = 990.2711 / 2.0
    cy = 637.7721 / 2.0

    # ── load arrays ──────────────────────────────────────────────────────────
    left_rgb = np.array(data["obs"]["images"]["left_rgb"]).astype(np.uint8)
    right_rgb = np.array(data["obs"]["images"]["right_rgb"]).astype(np.uint8)
    depth_mm = np.array(data["obs"]["images"]["depth"])

    obs_torso = np.array(data["obs"]["joint"]["torso"])  # (N,3)
    obs_left_arm = np.array(data["obs"]["joint"]["left_arm"])  # (N,7)
    obs_right_arm = np.array(data["obs"]["joint"]["right_arm"])  # (N,7)
    obs_head = np.array(data["obs"]["joint"]["head"])  # (N,3)

    eef_left = np.array(data["action"]["eef"]["left"])  # (N,4,4)
    eef_right = np.array(data["action"]["eef"]["right"])  # (N,4,4)

    N = left_rgb.shape[0]
    print(f"Episode: {N} frames  |  images: {left_rgb.shape[1]}x{left_rgb.shape[2]}")

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # ── KinHelper for camera FK ───────────────────────────────────────────────
    kin = KinHelper("vega_no_effector")
    joint_names_kin = [j.name for j in kin.sapien_robot.get_active_joints()]
    name_to_idx_kin = {n: i for i, n in enumerate(joint_names_kin)}
    cam_link_idx = kin.link_name_to_idx["zed_depth_frame"]

    # ── viser server ─────────────────────────────────────────────────────────
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    robot_mesh_gen = RobotMeshGenerator("vega_no_effector")

    cmap_depth = plt.colormaps.get_cmap("plasma")

    for idx in range(N):
        t0 = time.perf_counter()

        # ── build joint vector ────────────────────────────────────────────
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

        # ── camera pose via FK ────────────────────────────────────────────
        qpos = np.zeros(kin.sapien_robot.dof)
        for name, val in zip(joint_names, joints_vals, strict=False):
            if name in name_to_idx_kin:
                qpos[name_to_idx_kin[name]] = val
        world_t_cam = kin.compute_fk_from_link_idx(qpos, [cam_link_idx])[0]  # (4,4)

        # ── point cloud for this frame ────────────────────────────────────
        depth_m = depth_mm[idx] / 1000.0
        pts, mask = unproject_depth(depth_m, K, world_t_cam)
        cols = left_rgb[idx][mask].astype(np.uint8)  # (M,3) RGB

        pts, cols = voxel_downsample(pts, cols, args.voxel)

        # ── viser: point cloud ────────────────────────────────────────────
        server.scene.add_point_cloud(
            name="/pcd",
            points=pts,
            colors=cols,
            point_size=0.003,
        )

        # ── viser: robot meshes ───────────────────────────────────────────
        meshes = robot_mesh_gen.compute_robot_meshes(joints_arr)
        for i, mesh in enumerate(meshes):
            server.scene.add_mesh_trimesh(name=f"/robot/link_{i}", mesh=mesh)

        # ── viser: EEF frames ─────────────────────────────────────────────
        wxyz, pos = mat_to_wxyz_pos(eef_left[idx])
        server.scene.add_frame(
            "/eef/left", wxyz=wxyz, position=pos, axes_length=0.1, axes_radius=0.005
        )
        wxyz, pos = mat_to_wxyz_pos(eef_right[idx])
        server.scene.add_frame(
            "/eef/right", wxyz=wxyz, position=pos, axes_length=0.1, axes_radius=0.005
        )

        # ── viser: camera frame ───────────────────────────────────────────
        wxyz, pos = mat_to_wxyz_pos(world_t_cam)
        server.scene.add_frame(
            "/camera", wxyz=wxyz, position=pos, axes_length=0.05, axes_radius=0.003
        )

        # ── cv2: image panel ──────────────────────────────────────────────
        rgb_vis = np.concatenate(
            [
                cv2.resize(left_rgb[idx], (480, 300)),
                cv2.resize(right_rgb[idx], (480, 300)),
            ],
            axis=1,
        )
        rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)

        d_norm = np.clip(depth_mm[idx] / 3000.0, 0.0, 1.0)
        depth_vis = (cmap_depth(d_norm)[..., :3] * 255).astype(np.uint8)
        depth_vis = cv2.resize(cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR), (480, 300))

        frame_vis = np.concatenate([rgb_vis, depth_vis], axis=1)
        cv2.putText(
            frame_vis, f"{idx+1}/{N}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.imshow("episode", frame_vis)
        if cv2.waitKey(1) == ord("q"):
            break

        if idx == 0:
            time.sleep(2.0)  # wait for viser browser to connect

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, 1.0 / args.fps - elapsed))

    cv2.destroyAllWindows()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
