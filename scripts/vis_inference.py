r"""Visualise bimanual EEF trajectories (GT vs. predicted) as marker points.

Builds on ``scripts/vis_episode.py``: same point-cloud / robot-mesh / camera
layout under a single ``frame`` timeline, with head RGB (+ depth) and wrist RGB
in dedicated 2D panels (rerun blueprint), plus per-frame logging of the
**future** EEF trajectory for BOTH arms under
``world/eef_traj/{left,right,left_pred,right_pred}``. At frame ``t`` only frames
``t+1..N-1`` are logged, so markers naturally disappear as the rerun scrubber
passes them.

Ground-truth trajectories come from FK on the recorded commanded arm joints
(``action/joint/{left,right}_arm``). Predicted trajectories come from
``pred_vs_gt.npz`` produced by ``examples/port_datasets/infer_dexmate.py`` for a
**bimanual** checkpoint: 20-D eef (``[L_pos,L_rot6d,L_grip, R_pos,R_rot6d,R_grip]``)
or 16-D joint (``[L_arm(7),L_grip, R_arm(7),R_grip]``), LEFT block then RIGHT.

Predicted markers are colored per-arm (blue=left, green=right) and recolored
yellow on frames the rollout safety guards would reject — the guards are a
faithful mirror of ``policy_rollout._action_to_joint_targets`` (coupled bimanual
Pink IK for eef, per-arm 10°/tick clamp, hardware joint limits, and per-arm
Cartesian workspace; a tick is rejected — and both chains frozen — if either arm
fails).

Targets the current raw teleop HDF5 layout (see ``vis_episode.py``)::

    obs/images/{head_left_rgb, head_depth, left_wrist_rgb, intrinsic, extrinsic}
    obs/joint/{torso, left_arm, right_arm, head}
    action/joint/{left_arm, right_arm}

Usage::

    python /home/yixuan/omniteleop/scripts/vis_inference.py \
        --hdf5 /home/yixuan/Dexmate/data/raw_data_renamed/test/episode_0.hdf5 \
        --pred-npz /home/yixuan/Dexmate/infer/act_dexmate_eef_eef_abs/pred_vs_gt.npz
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from dexmotion.motion_manager import MotionManager
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5
from yixuan_utilities.kinematics_helper import KinHelper
from yixuan_utilities.robot_mesh_generator import RobotMeshGenerator

from omniteleop.common.vr_mode_const import (
    INIT_HEAD_JOINTS,
    INIT_LEFT_ARM_JOINTS,
    INIT_RIGHT_ARM_JOINTS,
    INIT_TORSO_JOINTS,
)
from omniteleop.follower.workspace_check import (
    DEFAULT_LEFT_BOUNDS,
    DEFAULT_LEFT_LINK,
    DEFAULT_RIGHT_BOUNDS,
    DEFAULT_RIGHT_LINK,
    WorkspaceChecker,
)

_WHEEL_NAMES = ["B_wheel_j1", "B_wheel_j2", "R_wheel_j1", "R_wheel_j2", "L_wheel_j1", "L_wheel_j2"]
_ROBOT_NAME = "vega_no_effector"
_LEFT_ARM_EEF_LINK = "L_ee"
_RIGHT_ARM_EEF_LINK = "R_ee"
_LEFT_ARM_JOINTS = tuple(f"L_arm_j{i}" for i in range(1, 8))
_RIGHT_ARM_JOINTS = tuple(f"R_arm_j{i}" for i in range(1, 8))
_MAX_JOINT_ANGLE_STEP = np.deg2rad(10.0)
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


def gram_schmidt_6d_to_R(r6: np.ndarray) -> np.ndarray:
    """Recover 3x3 rotation from Zhou et al. 6-D representation [R[:,0], R[:,1]]."""
    a1, a2 = r6[:3], r6[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def action_joint_array(data: dict, side: str) -> tuple[np.ndarray, str]:
    """Return one arm's commanded joints, preferring the current short key."""
    action_joint = data["action"]["joint"]
    for key in (side, f"{side}_arm"):
        if key in action_joint:
            return np.array(action_joint[key], dtype=np.float64), f"action/joint/{key}"
    raise KeyError(f"Expected action/joint/{side} or action/joint/{side}_arm")


def qpos_from_arms(
    kin: KinHelper,
    name_to_idx: dict[str, int],
    torso: np.ndarray,
    left_arm: np.ndarray,
    right_arm: np.ndarray,
    head: np.ndarray,
) -> np.ndarray:
    """Map (torso, left_arm, right_arm, head) onto the sapien active-joint vector."""
    qpos = np.zeros(kin.sapien_robot.dof, dtype=np.float64)
    joint_vals = torso.tolist() + left_arm.tolist() + right_arm.tolist() + head.tolist()
    for name, val in zip(_OBS_NAMES, joint_vals, strict=True):
        if name in name_to_idx:
            qpos[name_to_idx[name]] = val
    return qpos


def compute_arm_eef_bimanual(
    kin: KinHelper,
    name_to_idx: dict[str, int],
    left_eef_link_idx: int,
    right_eef_link_idx: int,
    obs_torso: np.ndarray,
    left_arm: np.ndarray,
    right_arm: np.ndarray,
    obs_head: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """FK both arms to ``(N, 4, 4)`` EEF poses (NaN for non-finite joint frames).

    ``left_arm`` / ``right_arm`` are the per-arm 7-DOF joint streams (commanded
    GT or predicted). L_ee FK depends only on torso + left arm; R_ee only on
    torso + right arm, so passing both arms together is exact.
    """
    for nm, arr in (("left_arm", left_arm), ("right_arm", right_arm)):
        if arr.ndim != 2 or arr.shape[1] != 7:
            raise ValueError(f"{nm} shape: expected (N, 7), got {arr.shape}")
    n = left_arm.shape[0]
    if not all(arr.shape[0] == n for arr in (right_arm, obs_torso, obs_head)):
        raise ValueError(
            "Frame count mismatch in FK: "
            f"left={n}, right={right_arm.shape[0]}, torso={obs_torso.shape[0]}, "
            f"head={obs_head.shape[0]}"
        )

    eef_left = np.full((n, 4, 4), np.nan, dtype=np.float32)
    eef_right = np.full((n, 4, 4), np.nan, dtype=np.float32)
    for idx in range(n):
        lf = np.isfinite(left_arm[idx]).all()
        rf = np.isfinite(right_arm[idx]).all()
        if not (lf or rf):
            continue
        # Fill non-finite arm with INIT so FK of the finite arm stays valid.
        la = left_arm[idx] if lf else np.asarray(INIT_LEFT_ARM_JOINTS)
        ra = right_arm[idx] if rf else np.asarray(INIT_RIGHT_ARM_JOINTS)
        qpos = qpos_from_arms(kin, name_to_idx, obs_torso[idx], la, ra, obs_head[idx])
        mats = kin.compute_fk_from_link_idx(qpos, [left_eef_link_idx, right_eef_link_idx])
        if lf:
            eef_left[idx] = mats[0].astype(np.float32)
        if rf:
            eef_right[idx] = mats[1].astype(np.float32)
    return eef_left, eef_right


def arm_joint_limits_from_kin(
    kin: KinHelper, joint_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) limits for ``joint_names`` from the FK robot model."""
    limits_by_name = {}
    for joint in kin.sapien_robot.get_active_joints():
        if joint.name in joint_names:
            limits = np.asarray(joint.get_limits(), dtype=np.float64)
            if limits.shape[0] != 1 or limits.shape[1] != 2:
                raise ValueError(f"Unexpected limits for {joint.name}: {limits}")
            limits_by_name[joint.name] = limits[0]

    missing = [name for name in joint_names if name not in limits_by_name]
    if missing:
        raise ValueError(f"Robot model is missing arm joint limits: {missing}")

    stacked = np.stack([limits_by_name[name] for name in joint_names])
    return stacked[:, 0], stacked[:, 1]


def limit_joint_step(raw_target: np.ndarray, chained: np.ndarray) -> np.ndarray:
    """Mirror ArmProcessor.limit_joint_step: 10 deg/tick clamp against chain."""
    diff = raw_target.astype(np.float64) - chained.astype(np.float64)
    alphas = np.minimum(1.0, _MAX_JOINT_ANGLE_STEP / (np.abs(diff) + 1e-10))
    return chained + alphas * diff


def check_guarded_joint_target(
    side: str,
    joint_target: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    workspace_checker: WorkspaceChecker,
) -> tuple[bool, str]:
    """Mirror policy_rollout per-arm joint-target + workspace rejection checks."""
    if joint_target.shape != (7,):
        return False, f"{side}_arm_bad_shape:{joint_target.shape}"
    if not np.all(np.isfinite(joint_target)):
        return False, f"{side}_arm_non_finite"
    if np.any(joint_target < joint_lower) or np.any(joint_target > joint_upper):
        return False, f"{side}_arm_outside_limits"

    in_bounds, _ = workspace_checker.is_in_workspace(
        arm_joints=joint_target.tolist(),
        head=list(INIT_HEAD_JOINTS),
        torso=list(INIT_TORSO_JOINTS),
    )
    if not in_bounds:
        return False, f"{side}_workspace_out_of_bounds"
    return True, "ok"


def motion_manager_state_dict(
    left_arm: np.ndarray,
    right_arm: np.ndarray,
    motion_joint_limits: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Build the bimanual rollout context used to seed MotionManager.

    Both arms are commanded; torso/head are fixed (clipped into dexmotion's
    limits when provided), mirroring policy_rollout._motion_manager_state_dict.
    """

    def fixed_value(name: str, value: float) -> float:
        if motion_joint_limits is None or name not in motion_joint_limits:
            return float(value)
        lower, upper = motion_joint_limits[name]
        eps = 1e-4
        return float(np.clip(value, lower + eps, upper - eps))

    qd = {f"L_arm_j{i + 1}": float(q) for i, q in enumerate(left_arm)}
    qd.update({f"R_arm_j{i + 1}": float(q) for i, q in enumerate(right_arm)})
    qd.update(
        {
            f"torso_j{i + 1}": fixed_value(f"torso_j{i + 1}", float(q))
            for i, q in enumerate(INIT_TORSO_JOINTS)
        }
    )
    qd.update(
        {
            f"head_j{i + 1}": fixed_value(f"head_j{i + 1}", float(q))
            for i, q in enumerate(INIT_HEAD_JOINTS)
        }
    )
    return qd


def fixed_motion_joint_limits(motion_manager) -> dict[str, tuple[float, float]]:
    """Cache dexmotion limits for the fixed non-commanded context (torso + head)."""
    model = motion_manager.pin_robot.model
    name_to_idx = getattr(motion_manager, "_joint_name_to_q_idx_dict", {})
    fixed_names = {
        *(f"torso_j{i}" for i in range(1, 4)),
        *(f"head_j{i}" for i in range(1, 4)),
    }
    return {
        name: (
            float(model.lowerPositionLimit[idx]),
            float(model.upperPositionLimit[idx]),
        )
        for name, idx in name_to_idx.items()
        if name in fixed_names
    }


def evaluate_joint_prediction_safety_bimanual(
    pred_left_joints: np.ndarray,
    pred_right_joints: np.ndarray,
    left_lower: np.ndarray,
    left_upper: np.ndarray,
    right_lower: np.ndarray,
    right_upper: np.ndarray,
    left_workspace: WorkspaceChecker,
    right_workspace: WorkspaceChecker,
) -> tuple[np.ndarray, dict[str, int]]:
    """Run the rollout joint-action safety guards per frame (whole-tick pass).

    A frame passes only if BOTH arms pass (matching policy_rollout, which freezes
    both chains on any rejection). Each chain advances only on a passing frame.
    """
    n = pred_left_joints.shape[0]
    passes = np.zeros(n, dtype=bool)
    reasons: dict[str, int] = {}
    chained_left = np.array(INIT_LEFT_ARM_JOINTS, dtype=np.float64)
    chained_right = np.array(INIT_RIGHT_ARM_JOINTS, dtype=np.float64)

    for idx in range(n):
        guarded_left = limit_joint_step(pred_left_joints[idx], chained_left)
        guarded_right = limit_joint_step(pred_right_joints[idx], chained_right)
        ok_l, why_l = check_guarded_joint_target(
            "left", guarded_left.astype(np.float32), left_lower, left_upper, left_workspace
        )
        ok_r, why_r = check_guarded_joint_target(
            "right", guarded_right.astype(np.float32), right_lower, right_upper, right_workspace
        )
        if ok_l and ok_r:
            passes[idx] = True
            chained_left = guarded_left
            chained_right = guarded_right
            reasons["ok"] = reasons.get("ok", 0) + 1
        else:
            reason = why_l if not ok_l else why_r
            reasons[reason] = reasons.get(reason, 0) + 1
    return passes, reasons


def evaluate_eef_pose_prediction_safety_bimanual(
    pred_left_eef9: np.ndarray,
    pred_right_eef9: np.ndarray,
    left_lower: np.ndarray,
    left_upper: np.ndarray,
    right_lower: np.ndarray,
    right_upper: np.ndarray,
    left_workspace: WorkspaceChecker,
    right_workspace: WorkspaceChecker,
) -> tuple[np.ndarray, dict[str, int]]:
    """Run rollout EEF-action guards (coupled bimanual Pink IK) per frame.

    Mirrors policy_rollout._action_to_joint_targets: seed MM with both chained
    arms, solve one ``ik({L_ee, R_ee})`` (inter-arm self-collision aware), clamp
    each arm 10°/tick, then per-arm joint-limit + workspace checks. Whole-tick
    pass requires both arms; both chains advance only on a passing frame.
    """
    n = pred_left_eef9.shape[0]
    chained_left = np.array(INIT_LEFT_ARM_JOINTS, dtype=np.float64)
    chained_right = np.array(INIT_RIGHT_ARM_JOINTS, dtype=np.float64)
    mm = MotionManager(
        init_visualizer=False,
        joint_regions_to_lock=["BASE"],
        target_frames=[_LEFT_ARM_EEF_LINK, _RIGHT_ARM_EEF_LINK],
        initial_joint_configuration_dict=motion_manager_state_dict(
            chained_left, chained_right
        ),
    )
    if mm.pin_robot is None:
        raise ValueError("MotionManager failed to initialize")
    motion_limits = fixed_motion_joint_limits(mm)

    passes = np.zeros(n, dtype=bool)
    reasons: dict[str, int] = {}

    def bump(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    for idx in range(n):
        l9 = pred_left_eef9[idx].astype(np.float64)
        r9 = pred_right_eef9[idx].astype(np.float64)
        if l9.shape != (9,) or r9.shape != (9,) or not (
            np.all(np.isfinite(l9)) and np.all(np.isfinite(r9))
        ):
            bump("eef_non_finite")
            continue

        mm.set_joint_pos(
            motion_manager_state_dict(chained_left, chained_right, motion_limits)
        )
        tl = np.eye(4, dtype=np.float64)
        tl[:3, :3] = gram_schmidt_6d_to_R(l9[3:9])
        tl[:3, 3] = l9[:3]
        tr = np.eye(4, dtype=np.float64)
        tr[:3, :3] = gram_schmidt_6d_to_R(r9[3:9])
        tr[:3, 3] = r9[:3]

        try:
            arm_solution, in_collision, within_limits = mm.ik(
                target_pose={_LEFT_ARM_EEF_LINK: tl, _RIGHT_ARM_EEF_LINK: tr},
                type="pink",
            )
        except Exception as exc:
            bump(f"ik_exception:{type(exc).__name__}")
            continue
        if not arm_solution:
            bump("ik_no_solution")
            continue
        if in_collision:
            bump("ik_collision")
            continue
        if not within_limits:
            bump("ik_outside_limits")
            continue

        try:
            raw_left = np.array(
                [arm_solution[f"L_arm_j{i}"] for i in range(1, 8)], dtype=np.float64
            )
            raw_right = np.array(
                [arm_solution[f"R_arm_j{i}"] for i in range(1, 8)], dtype=np.float64
            )
        except KeyError as exc:
            bump(f"ik_missing_joint:{exc}")
            continue

        guarded_left = limit_joint_step(raw_left, chained_left)
        guarded_right = limit_joint_step(raw_right, chained_right)
        ok_l, why_l = check_guarded_joint_target(
            "left", guarded_left.astype(np.float32), left_lower, left_upper, left_workspace
        )
        ok_r, why_r = check_guarded_joint_target(
            "right", guarded_right.astype(np.float32), right_lower, right_upper, right_workspace
        )
        if ok_l and ok_r:
            passes[idx] = True
            chained_left = guarded_left
            chained_right = guarded_right
            bump("ok")
        else:
            bump(why_l if not ok_l else why_r)
    return passes, reasons


def npz_episode_lengths_text(ep_ids: list[int], lengths: np.ndarray) -> str:
    return ", ".join(f"{int(ep)}:{int(length)}" for ep, length in zip(ep_ids, lengths, strict=True))


def file_mtime_text(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ", timespec="minutes")
    except OSError:
        return "unknown"


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


def _resolve_pred_space(pred_space: str, n_cols: int) -> str:
    """Map a bimanual pred width to 'joint' (16) or 'eef' (20)."""
    if pred_space != "auto":
        return pred_space
    if n_cols == 16:
        return "joint"
    if n_cols == 20:
        return "eef"
    raise ValueError(
        f"Cannot auto-detect pred space for {n_cols}-D action. Expected bimanual "
        "16-D joint or 20-D eef; pass --pred-space explicitly for other layouts."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdf5",
        type=str,
        default="/home/yixuan/Dexmate/data/raw_data_renamed/test/episode_0.hdf5",
        help="Path to episode HDF5 file (current raw teleop layout).",
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
        help="Path to pred_vs_gt.npz produced by infer_dexmate.py (bimanual checkpoint).",
    )
    parser.add_argument(
        "--pred-space",
        choices=["auto", "joint", "eef"],
        default="auto",
        help=(
            "How to interpret npz['pred']: 'joint' = bimanual 16-D (7 joints + "
            "gripper per arm), FK'd to XYZ; 'eef' = bimanual 20-D (pos+rot6d+grip "
            "per arm), first 3 of each block used as world XYZ. 'auto' picks "
            "'joint' for 16 cols and 'eef' for 20."
        ),
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode index in the LeRobot dataset; if omitted, parsed from --hdf5 filename.",
    )
    args = parser.parse_args()

    data, _ = load_dict_from_hdf5(args.hdf5)

    # ── load arrays (current raw teleop layout) ──────────────────────────────
    img_group = data["obs"]["images"]
    head_rgb = np.array(img_group["head_left_rgb"]).astype(np.uint8)  # (N,H,W,3)
    wrist_rgb = np.array(img_group["left_wrist_rgb"]).astype(np.uint8)  # (N,H',W',3)
    depth_mm = np.array(img_group["head_depth"])  # (N,H,W) uint16, millimetres

    obs_torso = np.array(data["obs"]["joint"]["torso"])
    obs_left_arm = np.array(data["obs"]["joint"]["left_arm"])
    obs_right_arm = np.array(data["obs"]["joint"]["right_arm"])
    obs_head = np.array(data["obs"]["joint"]["head"])

    action_left_arm, left_key = action_joint_array(data, "left")
    action_right_arm, right_key = action_joint_array(data, "right")

    N = head_rgb.shape[0]
    H, W = head_rgb.shape[1], head_rgb.shape[2]
    for key, arr in ((left_key, action_left_arm), (right_key, action_right_arm)):
        if arr.shape[0] != N:
            raise ValueError(f"Frame count mismatch: images={N}, {key}={arr.shape[0]}")
    print(f"Episode: {N} frames  |  head image: {H}x{W}  |  wrist: {wrist_rgb.shape[1:3]}")

    kin = KinHelper(_ROBOT_NAME)
    joint_names_kin = [j.name for j in kin.sapien_robot.get_active_joints()]
    name_to_idx_kin = {n: i for i, n in enumerate(joint_names_kin)}
    cam_link_idx = kin.link_name_to_idx["zed_depth_frame"]
    left_eef_link_idx = kin.link_name_to_idx[_LEFT_ARM_EEF_LINK]
    right_eef_link_idx = kin.link_name_to_idx[_RIGHT_ARM_EEF_LINK]
    left_lower, left_upper = arm_joint_limits_from_kin(kin, _LEFT_ARM_JOINTS)
    right_lower, right_upper = arm_joint_limits_from_kin(kin, _RIGHT_ARM_JOINTS)
    left_workspace = WorkspaceChecker(
        robot_name=_ROBOT_NAME,
        arm_link=DEFAULT_LEFT_LINK,
        arm_joint_names=_LEFT_ARM_JOINTS,
        bounds=DEFAULT_LEFT_BOUNDS,
    )
    right_workspace = WorkspaceChecker(
        robot_name=_ROBOT_NAME,
        arm_link=DEFAULT_RIGHT_LINK,
        arm_joint_names=_RIGHT_ARM_JOINTS,
        bounds=DEFAULT_RIGHT_BOUNDS,
    )

    # ── GT future EEF (FK on commanded arm joints) ───────────────────────────
    print(f"Computing GT EEF via FK from {left_key} / {right_key} ...")
    gt_eef_left, gt_eef_right = compute_arm_eef_bimanual(
        kin, name_to_idx_kin, left_eef_link_idx, right_eef_link_idx,
        obs_torso, action_left_arm, action_right_arm, obs_head,
    )
    gt_left_xyz = gt_eef_left[:, :3, 3].astype(np.float32)
    gt_right_xyz = gt_eef_right[:, :3, 3].astype(np.float32)
    gt_left_valid = np.isfinite(gt_left_xyz).all(axis=1)
    gt_right_valid = np.isfinite(gt_right_xyz).all(axis=1)

    # ── optional predicted trajectory overlay ────────────────────────────────
    pred_left_xyz: np.ndarray | None = None
    pred_right_xyz: np.ndarray | None = None
    pred_left_valid: np.ndarray | None = None
    pred_right_valid: np.ndarray | None = None
    pred_safety_pass: np.ndarray | None = None
    if args.pred_npz:
        pred_npz_path = Path(args.pred_npz)
        npz = np.load(pred_npz_path)
        ep = args.episode
        if ep is None:
            ep = int(Path(args.hdf5).stem.split("_")[-1])  # "episode_3" → 3
        ep_ids = npz["episode_ids"].tolist()
        lengths = npz["episode_lengths"]
        if ep not in ep_ids:
            raise ValueError(
                f"Episode {ep} not in {pred_npz_path} episode_ids {ep_ids}. "
                f"Available episode lengths are {npz_episode_lengths_text(ep_ids, lengths)}."
            )
        idx = ep_ids.index(ep)
        start = int(lengths[:idx].sum())
        end = start + int(lengths[idx])
        if end - start != N:
            matching_eps = [
                int(candidate_ep)
                for candidate_ep, length in zip(ep_ids, lengths, strict=True)
                if int(length) == N
            ]
            matching_text = (
                f"Episode(s) with {N} frames in this NPZ: {matching_eps}."
                if matching_eps
                else f"No episode in this NPZ has {N} frames."
            )
            raise ValueError(
                f"Frame count mismatch: HDF5 {args.hdf5} has {N} frames, "
                f"but {pred_npz_path} episode {ep} has {end - start} frames. "
                f"NPZ modified: {file_mtime_text(pred_npz_path)}. "
                f"Available NPZ episode lengths are {npz_episode_lengths_text(ep_ids, lengths)}. "
                f"{matching_text} Regenerate pred_vs_gt.npz from the matching "
                "LeRobot dataset root/split, or pass the matching --pred-npz/--episode."
            )
        pred_arr = npz["pred"][start:end].astype(np.float32)
        if pred_arr.ndim != 2:
            raise ValueError(
                f"Expected npz['pred'] to be 2-D (frames, dims); got shape {npz['pred'].shape}"
            )
        n_cols = pred_arr.shape[1]
        pred_space = _resolve_pred_space(args.pred_space, n_cols)
        block = 8 if pred_space == "joint" else 10
        if n_cols < 2 * block:
            raise ValueError(
                f"{pred_space} bimanual pred needs ≥{2 * block} columns; got {n_cols}"
            )
        left_block = pred_arr[:, 0:block]
        right_block = pred_arr[:, block : 2 * block]

        if pred_space == "joint":
            pred_left_joints = left_block[:, :7]
            pred_right_joints = right_block[:, :7]
            pred_eef_left, pred_eef_right = compute_arm_eef_bimanual(
                kin, name_to_idx_kin, left_eef_link_idx, right_eef_link_idx,
                obs_torso, pred_left_joints, pred_right_joints, obs_head,
            )
            pred_left_xyz = pred_eef_left[:, :3, 3].astype(np.float32)
            pred_right_xyz = pred_eef_right[:, :3, 3].astype(np.float32)
            pred_safety_pass, safety_reasons = evaluate_joint_prediction_safety_bimanual(
                pred_left_joints, pred_right_joints,
                left_lower, left_upper, right_lower, right_upper,
                left_workspace, right_workspace,
            )
        else:
            pred_left_xyz = left_block[:, 0:3].astype(np.float32)
            pred_right_xyz = right_block[:, 0:3].astype(np.float32)
            pred_safety_pass, safety_reasons = evaluate_eef_pose_prediction_safety_bimanual(
                left_block[:, 0:9], right_block[:, 0:9],
                left_lower, left_upper, right_lower, right_upper,
                left_workspace, right_workspace,
            )

        pred_left_valid = np.isfinite(pred_left_xyz).all(axis=1)
        pred_right_valid = np.isfinite(pred_right_xyz).all(axis=1)
        if pred_safety_pass.shape != (N,):
            raise ValueError(
                f"Internal safety mask shape {pred_safety_pass.shape} != ({N},)"
            )
        print(
            f"Loaded predictions: ep={ep}, frames={end - start}, "
            f"pred_space={pred_space} (cols={n_cols})"
        )
        print(
            "Prediction safety guards (whole-tick, both arms): "
            f"pass={int(pred_safety_pass.sum())}, "
            f"fail={int((~pred_safety_pass).sum())}, reasons={safety_reasons}"
        )

    # ── intrinsics ───────────────────────────────────────────────────────────
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
    if "extrinsic" in img_group:
        extrinsics = np.array(img_group["extrinsic"]).astype(np.float64)
        print("Using saved obs/images/extrinsic")
    else:
        print("No saved extrinsic — computing per-frame FK ...")
        extrinsics = np.zeros((N, 4, 4), dtype=np.float64)
        for idx in range(N):
            qpos = qpos_from_arms(
                kin, name_to_idx_kin, obs_torso[idx], obs_left_arm[idx],
                obs_right_arm[idx], obs_head[idx],
            )
            extrinsics[idx] = kin.compute_fk_from_link_idx(qpos, [cam_link_idx])[0]

    robot_mesh_gen = RobotMeshGenerator("vega_no_effector")

    # ── rerun ────────────────────────────────────────────────────────────────
    rr.init("vis_inference", spawn=True)

    # Default layout: 3D world (PCD, robot, EEF traj) + head / wrist RGB panels.
    # Without an explicit blueprint the viewer often omits 2D image streams.
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/world", name="world (pcd + robot + EEF traj)"),
                rrb.Vertical(
                    rrb.Spatial2DView(origin="/world/camera", name="head_left_rgb"),
                    rrb.Spatial2DView(origin="/image/left_wrist_rgb", name="left_wrist_rgb"),
                ),
                column_shares=[2, 1],
            ),
            collapse_panels=True,
        ),
        make_active=True,
        make_default=True,
    )

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    K32 = K.astype(np.float32)

    GT_LEFT_COLOR = np.array([255, 160, 40], dtype=np.uint8)  # orange — left EEF future (GT)
    GT_RIGHT_COLOR = np.array([255, 80, 80], dtype=np.uint8)  # red — right EEF future (GT)
    PRED_LEFT_COLOR = np.array([60, 140, 240], dtype=np.uint8)  # blue — left predicted
    PRED_RIGHT_COLOR = np.array([60, 220, 80], dtype=np.uint8)  # green — right predicted
    PRED_FAIL_COLOR = np.array([255, 220, 40], dtype=np.uint8)  # yellow — failed guard

    def log_future_markers(path: str, xyz: np.ndarray, valid: np.ndarray,
                           idx: int, color: np.ndarray,
                           safety_pass: np.ndarray | None = None) -> None:
        """Log frames idx+1..N-1 of a trajectory; recolor yellow where guards fail."""
        future_mask = valid[idx + 1:]
        future_xyz = xyz[idx + 1:][future_mask]
        if not future_xyz.size:
            rr.log(path, rr.Clear(recursive=False))
            return
        if safety_pass is not None:
            future_pass = safety_pass[idx + 1:][future_mask]
            colors = np.where(
                future_pass[:, None], color[None, :], PRED_FAIL_COLOR[None, :]
            ).astype(np.uint8)
        else:
            colors = color
        rr.log(path, rr.Points3D(future_xyz, colors=colors, radii=args.marker_radius))

    for idx in range(N):
        rr.set_time("frame", sequence=idx)

        world_t_cam = extrinsics[idx]

        # ── camera ──────────────────────────────────────────────────────────
        rr.log(
            "world/camera",
            rr.Transform3D(translation=world_t_cam[:3, 3], mat3x3=world_t_cam[:3, :3]),
        )
        rr.log("world/camera", rr.Pinhole(image_from_camera=K32, width=W, height=H))
        rr.log("world/camera/rgb", rr.Image(head_rgb[idx]))
        rr.log("world/camera/depth", rr.DepthImage(depth_mm[idx], meter=1000.0))
        rr.log("image/left_wrist_rgb", rr.Image(wrist_rgb[idx]))

        # ── colored point cloud ─────────────────────────────────────────────
        depth_m = depth_mm[idx] / 1000.0
        pts, mask = unproject_depth(depth_m, K, world_t_cam)
        cols = head_rgb[idx][mask]
        pts, cols = voxel_downsample(pts, cols, args.voxel)
        rr.log("world/pcd", rr.Points3D(pts, colors=cols, radii=0.003))

        # ── robot meshes (observed pose) ────────────────────────────────────
        joints_vals = (
            [0.0] * len(_WHEEL_NAMES)
            + obs_torso[idx].tolist()
            + obs_left_arm[idx].tolist()
            + obs_right_arm[idx].tolist()
            + obs_head[idx].tolist()
        )
        joint_names = _WHEEL_NAMES + _OBS_NAMES
        joints_arr = robot_mesh_gen.convert_to_sapien_joint_order(
            np.array(joints_vals), joint_names
        )
        meshes = robot_mesh_gen.compute_robot_meshes(joints_arr)
        for i, mesh in enumerate(meshes):
            rr.log(
                f"world/robot/link_{i}",
                rr.Mesh3D(
                    vertex_positions=np.asarray(mesh.vertices, dtype=np.float32),
                    triangle_indices=np.asarray(mesh.faces, dtype=np.uint32),
                ),
            )

        # ── current EEF coordinate frames (skip NaN frames) ─────────────────
        for name, mat in (("world/eef/left", gt_eef_left[idx]), ("world/eef/right", gt_eef_right[idx])):
            if np.all(np.isfinite(mat)):
                rr.log(name, rr.Transform3D(translation=mat[:3, 3], mat3x3=mat[:3, :3]))
            else:
                rr.log(name, rr.Clear(recursive=True))

        # ── future GT EEF trajectory markers (both arms) ────────────────────
        log_future_markers("world/eef_traj/left", gt_left_xyz, gt_left_valid, idx, GT_LEFT_COLOR)
        log_future_markers("world/eef_traj/right", gt_right_xyz, gt_right_valid, idx, GT_RIGHT_COLOR)

        # ── future predicted EEF trajectory markers (both arms) ─────────────
        if pred_left_xyz is not None:
            assert pred_right_xyz is not None
            assert pred_left_valid is not None and pred_right_valid is not None
            log_future_markers(
                "world/eef_traj/left_pred", pred_left_xyz, pred_left_valid, idx,
                PRED_LEFT_COLOR, pred_safety_pass,
            )
            log_future_markers(
                "world/eef_traj/right_pred", pred_right_xyz, pred_right_valid, idx,
                PRED_RIGHT_COLOR, pred_safety_pass,
            )


if __name__ == "__main__":
    main()
