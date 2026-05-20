"""Visualise a recorded VR teleoperation episode in rerun.

Logs RGB-D, the colored point cloud, robot meshes, EEF and camera frames
under a single ``frame`` timeline so the rerun viewer's scrubber doubles as
prev/next/jump navigation.

Targets the ``raw_data_2`` HDF5 layout::

    obs/images/{head_left_rgb, head_depth, left_wrist_rgb, intrinsic, extrinsic}
    obs/joint/{torso, left_arm, right_arm, head}
    obs/gripper/{left, right}
    action/joint/{left_arm, right_arm, head, chassis_*}
    action/gripper/{left, right}
    stage, timestamp_ns

EEF frames are no longer stored in the episode, so they are recovered via
forward kinematics from the observed arm joints (``L_ee`` / ``R_ee`` links).

Usage::

    python scripts/vis_episode.py --hdf5 data/episode_0.hdf5
    python scripts/vis_episode.py --hdf5 data/episode_0.hdf5 --voxel 0.01
"""

from __future__ import annotations

import argparse

import numpy as np
import rerun as rr
import torch
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
        default="/home/yixuan/Dexmate/data/raw_data_2/episode_3.hdf5",
        help="Path to episode HDF5 file",
    )
    parser.add_argument(
        "--voxel", type=float, default=0.008, help="Voxel size for PCD downsample (m)"
    )
    args = parser.parse_args()

    data, _ = load_dict_from_hdf5(args.hdf5)

    # ── load image arrays (raw_data_2 layout) ────────────────────────────────
    img_group = data["obs"]["images"]
    head_rgb = np.array(img_group["head_left_rgb"]).astype(np.uint8)  # (N,H,W,3)
    wrist_rgb = np.array(img_group["left_wrist_rgb"]).astype(np.uint8)  # (N,H',W',3)
    depth_mm = np.array(img_group["head_depth"])  # (N,H,W) uint16, millimetres

    obs_torso = np.array(data["obs"]["joint"]["torso"])  # (N,3)
    obs_left_arm = np.array(data["obs"]["joint"]["left_arm"])  # (N,7)
    obs_right_arm = np.array(data["obs"]["joint"]["right_arm"])  # (N,7)
    obs_head = np.array(data["obs"]["joint"]["head"])  # (N,3)

    # ── scalar streams (gripper may be NaN, stage is an int label) ───────────
    obs_grip_left = np.array(data["obs"]["gripper"]["left"])  # (N,)
    obs_grip_right = np.array(data["obs"]["gripper"]["right"])  # (N,)
    act_grip_left = np.array(data["action"]["gripper"]["left"])  # (N,)
    act_grip_right = np.array(data["action"]["gripper"]["right"])  # (N,)
    stage = np.array(data["stage"])  # (N,)

    N = head_rgb.shape[0]
    H, W = head_rgb.shape[1], head_rgb.shape[2]
    print(f"Episode: {N} frames  |  head image: {H}x{W}  |  wrist: {wrist_rgb.shape[1:3]}")

    # ── intrinsics: prefer per-episode if recorded, else hardcoded ZED ───────
    if "intrinsic" in img_group:
        K_all = np.array(img_group["intrinsic"]).astype(np.float64)
        if K_all.ndim == 2:  # single shared matrix → broadcast over frames
            K_all = np.repeat(K_all[None], N, axis=0)
    else:
        fx = fy = 770.1868 / 2.0
        cx = 990.2711 / 2.0
        cy = 637.7721 / 2.0
        K_single = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        K_all = np.repeat(K_single[None], N, axis=0)

    # ── kinematics helper (camera extrinsic fallback + EEF via FK) ───────────
    kin = KinHelper("vega_no_effector")
    joint_names_kin = [j.name for j in kin.sapien_robot.get_active_joints()]
    name_to_idx_kin = {n: i for i, n in enumerate(joint_names_kin)}
    cam_link_idx = kin.link_name_to_idx["zed_depth_frame"]
    eef_link_idx = [kin.link_name_to_idx["L_ee"], kin.link_name_to_idx["R_ee"]]

    def build_qpos(idx: int) -> np.ndarray:
        """Map observed joints onto the sapien active-joint vector."""
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
        return qpos

    # ── extrinsics: prefer saved, else FK from observed joints ───────────────
    if "extrinsic" in img_group:
        extrinsics = np.array(img_group["extrinsic"]).astype(np.float64)
        print("Using saved obs/images/extrinsic")
    else:
        print("No saved extrinsic — computing per-frame FK ...")
        extrinsics = np.zeros((N, 4, 4), dtype=np.float64)
        for idx in range(N):
            extrinsics[idx] = kin.compute_fk_from_link_idx(build_qpos(idx), [cam_link_idx])[0]

    # ── EEF frames via FK from observed arm joints (action/eef no longer saved)
    print("Computing per-frame EEF frames (L_ee / R_ee) via FK ...")
    eef_left = np.zeros((N, 4, 4), dtype=np.float64)
    eef_right = np.zeros((N, 4, 4), dtype=np.float64)
    for idx in range(N):
        mats = kin.compute_fk_from_link_idx(build_qpos(idx), eef_link_idx)
        eef_left[idx], eef_right[idx] = mats[0], mats[1]

    robot_mesh_gen = RobotMeshGenerator("vega_no_effector")

    # ── rerun ────────────────────────────────────────────────────────────────
    rr.init("vis_episode", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    for idx in range(N):
        rr.set_time("frame", sequence=idx)

        world_t_cam = extrinsics[idx]
        K = K_all[idx]
        K32 = K.astype(np.float32)

        # ── camera: pose + intrinsics + 2D streams attached in image space ──
        rr.log(
            "world/camera",
            rr.Transform3D(translation=world_t_cam[:3, 3], mat3x3=world_t_cam[:3, :3]),
        )
        rr.log("world/camera", rr.Pinhole(image_from_camera=K32, width=W, height=H))
        rr.log("world/camera/rgb", rr.Image(head_rgb[idx]))
        rr.log("world/camera/depth", rr.DepthImage(depth_mm[idx], meter=1000.0))
        # Wrist RGB is monocular only — log under /image so it shows in 2D views.
        rr.log("image/left_wrist_rgb", rr.Image(wrist_rgb[idx]))

        # ── colored point cloud (manual unproject + voxel downsample) ───────
        depth_m = depth_mm[idx] / 1000.0
        pts, mask = unproject_depth(depth_m, K, world_t_cam)
        cols = head_rgb[idx][mask]
        pts, cols = voxel_downsample(pts, cols, args.voxel)
        rr.log("world/pcd", rr.Points3D(pts, colors=cols, radii=0.003))

        # ── robot meshes (already in world frame from compute_robot_meshes) ──
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

        # ── EEF frames (FK from observed arm joints) ────────────────────────
        for name, mat in (("world/eef/left", eef_left[idx]), ("world/eef/right", eef_right[idx])):
            if not np.all(np.isfinite(mat)):
                rr.log(name, rr.Clear(recursive=True))
                continue
            rr.log(name, rr.Transform3D(translation=mat[:3, 3], mat3x3=mat[:3, :3]))

        # ── scalar plots: gripper (obs vs action) and stage label ───────────
        rr.log("plot/gripper/obs_left", rr.Scalars(float(obs_grip_left[idx])))
        rr.log("plot/gripper/obs_right", rr.Scalars(float(obs_grip_right[idx])))
        rr.log("plot/gripper/act_left", rr.Scalars(float(act_grip_left[idx])))
        rr.log("plot/gripper/act_right", rr.Scalars(float(act_grip_right[idx])))
        rr.log("plot/stage", rr.Scalars(float(stage[idx])))


if __name__ == "__main__":
    main()
