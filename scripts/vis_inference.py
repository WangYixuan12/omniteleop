r"""Visualise the right-arm EEF trajectory (GT vs. predicted) as marker points.

Builds on ``scripts/vis_episode.py``: same point-cloud / robot-mesh / camera
layout under a single ``frame`` timeline, plus per-frame logging of the
**future** right-EEF trajectory under ``world/eef_traj/{right,right_pred}``.
At frame ``t`` only frames ``t+1..N-1`` are logged, so markers naturally
disappear as the rerun scrubber passes them.

The left arm is fixed in the dataset and is not visualized. Predicted
right-EEF positions come from ``pred_vs_gt.npz`` produced by
``examples/dexmate/infer_dexmate.py``.

Usage::

    python /home/yixuan/omniteleop/scripts/vis_inference.py \
        --hdf5 /home/yixuan/Dexmate/data/raw_data_renamed/test/episode_0.hdf5 \
        --pred-npz /home/yixuan/Dexmate/infer/act_dexmate_joint_joint/pred_vs_gt.npz
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import rerun as rr
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
from omniteleop.follower.workspace_check import DEFAULT_RIGHT_BOUNDS, WorkspaceChecker

_WHEEL_NAMES = ["B_wheel_j1", "B_wheel_j2", "R_wheel_j1", "R_wheel_j2", "L_wheel_j1", "L_wheel_j2"]
_ROBOT_NAME = "vega_no_effector"
_RIGHT_ARM_EEF_LINK = "R_ee"
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


def action_right_joint_array(data: dict) -> tuple[np.ndarray, str]:
    """Return commanded right-arm joints, preferring the current short key."""
    action_joint = data["action"]["joint"]
    for key in ("right", "right_arm"):
        if key in action_joint:
            return np.array(action_joint[key], dtype=np.float64), f"action/joint/{key}"
    raise KeyError("Expected action/joint/right or action/joint/right_arm")


def qpos_from_observed_and_right_action(
    kin: KinHelper,
    name_to_idx: dict[str, int],
    torso: np.ndarray,
    left_arm: np.ndarray,
    right_action: np.ndarray,
    head: np.ndarray,
) -> np.ndarray:
    qpos = np.zeros(kin.sapien_robot.dof, dtype=np.float64)
    joint_vals = torso.tolist() + left_arm.tolist() + right_action.tolist() + head.tolist()
    for name, val in zip(_OBS_NAMES, joint_vals, strict=True):
        if name in name_to_idx:
            qpos[name_to_idx[name]] = val
    return qpos


def compute_right_action_eef(
    kin: KinHelper,
    name_to_idx: dict[str, int],
    eef_link_idx: int,
    obs_torso: np.ndarray,
    obs_left_arm: np.ndarray,
    action_right_arm: np.ndarray,
    obs_head: np.ndarray,
) -> np.ndarray:
    if action_right_arm.ndim != 2 or action_right_arm.shape[1] != len(_RIGHT_ARM_JOINTS):
        raise ValueError(
            "right action joint shape: expected "
            f"(N, {len(_RIGHT_ARM_JOINTS)}), got {action_right_arm.shape}"
        )
    n = action_right_arm.shape[0]
    if not all(arr.shape[0] == n for arr in (obs_torso, obs_left_arm, obs_head)):
        raise ValueError(
            "Frame count mismatch while computing right-action FK: "
            f"action={n}, torso={obs_torso.shape[0]}, left_arm={obs_left_arm.shape[0]}, "
            f"head={obs_head.shape[0]}"
        )

    out = np.full((n, 4, 4), np.nan, dtype=np.float32)
    for idx in range(n):
        if not np.isfinite(action_right_arm[idx]).all():
            continue
        qpos = qpos_from_observed_and_right_action(
            kin,
            name_to_idx,
            obs_torso[idx],
            obs_left_arm[idx],
            action_right_arm[idx],
            obs_head[idx],
        )
        out[idx] = kin.compute_fk_from_link_idx(qpos, [eef_link_idx])[0].astype(np.float32)
    return out


def right_arm_joint_limits_from_kin(kin: KinHelper) -> tuple[np.ndarray, np.ndarray]:
    """Return right-arm joint limits from the same robot model used for FK."""
    limits_by_name = {}
    for joint in kin.sapien_robot.get_active_joints():
        if joint.name in _RIGHT_ARM_JOINTS:
            limits = np.asarray(joint.get_limits(), dtype=np.float64)
            if limits.shape[0] != 1 or limits.shape[1] != 2:
                raise ValueError(f"Unexpected limits for {joint.name}: {limits}")
            limits_by_name[joint.name] = limits[0]

    missing = [name for name in _RIGHT_ARM_JOINTS if name not in limits_by_name]
    if missing:
        raise ValueError(f"Robot model is missing right-arm joint limits: {missing}")

    stacked = np.stack([limits_by_name[name] for name in _RIGHT_ARM_JOINTS])
    return stacked[:, 0], stacked[:, 1]


def limit_joint_step(raw_target: np.ndarray, chained_right: np.ndarray) -> np.ndarray:
    """Mirror ArmProcessor.limit_joint_step: 10 deg/tick clamp against chain."""
    diff = raw_target.astype(np.float64) - chained_right.astype(np.float64)
    alphas = np.minimum(1.0, _MAX_JOINT_ANGLE_STEP / (np.abs(diff) + 1e-10))
    return chained_right + alphas * diff


def check_safety_guarded_joint_target(
    joint_target: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    workspace_checker: WorkspaceChecker,
) -> tuple[bool, str]:
    """Mirror policy_rollout joint-target and workspace safety rejection checks."""
    if joint_target.shape != (len(_RIGHT_ARM_JOINTS),):
        return False, f"right_arm_bad_shape:{joint_target.shape}"
    if not np.all(np.isfinite(joint_target)):
        return False, "right_arm_non_finite"
    if np.any(joint_target < joint_lower) or np.any(joint_target > joint_upper):
        return False, "right_arm_outside_limits"

    in_bounds, _ = workspace_checker.is_in_workspace(
        right_arm=joint_target.tolist(),
        head=list(INIT_HEAD_JOINTS),
        torso=list(INIT_TORSO_JOINTS),
    )
    if not in_bounds:
        return False, "workspace_out_of_bounds"
    return True, "ok"


def motion_manager_state_dict(
    right_arm: np.ndarray,
    motion_joint_limits: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Build the fixed rollout context used to seed MotionManager."""

    def fixed_value(name: str, value: float) -> float:
        if motion_joint_limits is None or name not in motion_joint_limits:
            return float(value)
        lower, upper = motion_joint_limits[name]
        eps = 1e-4
        return float(np.clip(value, lower + eps, upper - eps))

    qd = {f"R_arm_j{i + 1}": float(q) for i, q in enumerate(right_arm)}
    qd.update(
        {
            f"L_arm_j{i + 1}": fixed_value(f"L_arm_j{i + 1}", float(q))
            for i, q in enumerate(INIT_LEFT_ARM_JOINTS)
        }
    )
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
    """Cache dexmotion limits for the fixed non-commanded rollout context."""
    model = motion_manager.pin_robot.model
    name_to_idx = getattr(motion_manager, "_joint_name_to_q_idx_dict", {})
    fixed_names = {
        *(f"L_arm_j{i}" for i in range(1, 8)),
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


def evaluate_joint_prediction_safety(
    pred_right_joints: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    workspace_checker: WorkspaceChecker,
) -> tuple[np.ndarray, dict[str, int]]:
    """Run the rollout joint-action safety guards on each predicted action."""
    passes = np.zeros(pred_right_joints.shape[0], dtype=bool)
    reasons: dict[str, int] = {}
    chained_right = np.array(INIT_RIGHT_ARM_JOINTS, dtype=np.float64)

    for idx, raw_right in enumerate(pred_right_joints.astype(np.float64)):
        guarded = limit_joint_step(raw_right, chained_right)
        ok, reason = check_safety_guarded_joint_target(
            guarded.astype(np.float32),
            joint_lower,
            joint_upper,
            workspace_checker,
        )
        passes[idx] = ok
        reasons[reason] = reasons.get(reason, 0) + 1
        if ok:
            chained_right = guarded

    return passes, reasons


def evaluate_eef_pose_prediction_safety(
    pred_eef_actions: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    workspace_checker: WorkspaceChecker,
) -> tuple[np.ndarray, dict[str, int]]:
    """Run rollout EEF-action safety guards, including IK, on 9D EEF predictions."""
    chained_right = np.array(INIT_RIGHT_ARM_JOINTS, dtype=np.float64)
    mm = MotionManager(
        init_visualizer=False,
        joint_regions_to_lock=["BASE"],
        target_frames=[_RIGHT_ARM_EEF_LINK],
        initial_joint_configuration_dict=motion_manager_state_dict(chained_right),
    )
    if mm.pin_robot is None:
        raise ValueError("MotionManager failed to initialize")
    motion_limits = fixed_motion_joint_limits(mm)

    passes = np.zeros(pred_eef_actions.shape[0], dtype=bool)
    reasons: dict[str, int] = {}

    for idx, action in enumerate(pred_eef_actions.astype(np.float64)):
        gripperless = action[:9]
        if gripperless.shape != (9,) or not np.all(np.isfinite(gripperless)):
            reason = "eef_non_finite"
            reasons[reason] = reasons.get(reason, 0) + 1
            continue

        mm.set_joint_pos(motion_manager_state_dict(chained_right, motion_limits))
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = gram_schmidt_6d_to_R(gripperless[3:9])
        T[:3, 3] = gripperless[:3]

        try:
            arm_solution, in_collision, within_limits = mm.ik(
                target_pose={_RIGHT_ARM_EEF_LINK: T},
                type="pink",
            )
        except Exception as exc:
            reason = f"ik_exception:{type(exc).__name__}"
            reasons[reason] = reasons.get(reason, 0) + 1
            continue

        if not arm_solution:
            reason = "ik_no_solution"
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        if in_collision:
            reason = "ik_collision"
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        if not within_limits:
            reason = "ik_outside_limits"
            reasons[reason] = reasons.get(reason, 0) + 1
            continue

        try:
            raw_right = np.array(
                [arm_solution[f"R_arm_j{i}"] for i in range(1, 8)],
                dtype=np.float64,
            )
        except KeyError as exc:
            reason = f"ik_missing_joint:{exc}"
            reasons[reason] = reasons.get(reason, 0) + 1
            continue

        guarded = limit_joint_step(raw_right, chained_right)
        ok, reason = check_safety_guarded_joint_target(
            guarded.astype(np.float32),
            joint_lower,
            joint_upper,
            workspace_checker,
        )
        passes[idx] = ok
        reasons[reason] = reasons.get(reason, 0) + 1
        if ok:
            chained_right = guarded

    return passes, reasons


def evaluate_xyz_prediction_safety(
    pred_xyz: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Fallback when EEF predictions provide XYZ only and cannot be IK-checked."""
    passes = np.zeros(pred_xyz.shape[0], dtype=bool)
    reasons: dict[str, int] = {}
    bounds = DEFAULT_RIGHT_BOUNDS
    for idx, xyz in enumerate(pred_xyz):
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            reason = "eef_xyz_non_finite"
        elif not (
            bounds["x"][0] <= xyz[0] <= bounds["x"][1]
            and bounds["y"][0] <= xyz[1] <= bounds["y"][1]
            and bounds["z"][0] <= xyz[2] <= bounds["z"][1]
        ):
            reason = "workspace_out_of_bounds_xyz_only"
        else:
            reason = "ok"
            passes[idx] = True
        reasons[reason] = reasons.get(reason, 0) + 1
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
        "--pred-space",
        choices=["auto", "joint", "eef"],
        default="auto",
        help=(
            "How to interpret npz['pred']: 'joint' = 7 right-arm joints (+optional gripper), "
            "FK'd to XYZ; 'eef' = first 3 columns used directly as world XYZ. "
            "'auto' picks 'joint' when pred has 7 or 8 columns, else 'eef'."
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

    # ── load arrays ──────────────────────────────────────────────────────────
    left_rgb = np.array(data["obs"]["images"]["left_rgb"]).astype(np.uint8)
    right_rgb = np.array(data["obs"]["images"]["right_rgb"]).astype(np.uint8)
    depth_mm = np.array(data["obs"]["images"]["depth"])

    obs_torso = np.array(data["obs"]["joint"]["torso"])
    obs_left_arm = np.array(data["obs"]["joint"]["left_arm"])
    obs_right_arm = np.array(data["obs"]["joint"]["right_arm"])
    obs_head = np.array(data["obs"]["joint"]["head"])

    action_right_arm, action_right_key = action_right_joint_array(data)

    N = left_rgb.shape[0]
    H, W = left_rgb.shape[1], left_rgb.shape[2]
    if action_right_arm.shape[0] != N:
        raise ValueError(
            f"Frame count mismatch: images={N}, {action_right_key}={action_right_arm.shape[0]}"
        )
    print(f"Episode: {N} frames  |  images: {H}x{W}")

    kin = KinHelper(_ROBOT_NAME)
    joint_names_kin = [j.name for j in kin.sapien_robot.get_active_joints()]
    name_to_idx_kin = {n: i for i, n in enumerate(joint_names_kin)}
    cam_link_idx = kin.link_name_to_idx["zed_depth_frame"]
    right_eef_link_idx = kin.link_name_to_idx[_RIGHT_ARM_EEF_LINK]
    right_joint_lower, right_joint_upper = right_arm_joint_limits_from_kin(kin)
    workspace_checker = WorkspaceChecker(robot_name=_ROBOT_NAME, arm_link=_RIGHT_ARM_EEF_LINK)

    print(f"Computing right EEF action with FK from {action_right_key} ...")
    eef_right = compute_right_action_eef(
        kin,
        name_to_idx_kin,
        right_eef_link_idx,
        obs_torso,
        obs_left_arm,
        action_right_arm,
        obs_head,
    )

    # Translation-only marker positions; finite mask filters invalid joint frames.
    eef_right_xyz = eef_right[:, :3, 3].astype(np.float32)
    right_valid = np.isfinite(eef_right_xyz).all(axis=1)

    # Optional: overlay predicted right-EEF trajectory from infer_dexmate.py.
    pred_xyz: np.ndarray | None = None
    pred_valid: np.ndarray | None = None
    pred_safety_pass: np.ndarray | None = None
    if args.pred_npz:
        pred_npz_path = Path(args.pred_npz)
        npz = np.load(pred_npz_path)
        ep = args.episode
        if ep is None:
            stem = Path(args.hdf5).stem  # e.g. "episode_3"
            ep = int(stem.split("_")[-1])
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
        n_pred_cols = pred_arr.shape[1]
        if args.pred_space == "auto":
            pred_space = "joint" if n_pred_cols in (7, 8) else "eef"
        else:
            pred_space = args.pred_space

        if pred_space == "joint":
            if n_pred_cols < len(_RIGHT_ARM_JOINTS):
                raise ValueError(
                    f"Joint-space pred needs ≥{len(_RIGHT_ARM_JOINTS)} columns "
                    f"(right-arm joints); got {n_pred_cols}"
                )
            pred_right_joints = pred_arr[:, : len(_RIGHT_ARM_JOINTS)]
            pred_eef = compute_right_action_eef(
                kin,
                name_to_idx_kin,
                right_eef_link_idx,
                obs_torso,
                obs_left_arm,
                pred_right_joints,
                obs_head,
            )
            pred_xyz = pred_eef[:, :3, 3].astype(np.float32)
            pred_safety_pass, safety_reasons = evaluate_joint_prediction_safety(
                pred_right_joints,
                right_joint_lower,
                right_joint_upper,
                workspace_checker,
            )
        else:
            if n_pred_cols < 3:
                raise ValueError(
                    f"EEF-space pred needs ≥3 columns (xyz); got {n_pred_cols}"
                )
            pred_xyz = pred_arr[:, :3]
            if n_pred_cols >= 9:
                pred_safety_pass, safety_reasons = evaluate_eef_pose_prediction_safety(
                    pred_arr[:, :9],
                    right_joint_lower,
                    right_joint_upper,
                    workspace_checker,
                )
            else:
                pred_safety_pass, safety_reasons = evaluate_xyz_prediction_safety(pred_xyz)
                print(
                    "Prediction safety: EEF-space pred has only XYZ columns; "
                    "using finite + direct XYZ workspace coloring only."
                )

        pred_valid = np.isfinite(pred_xyz).all(axis=1)
        if pred_safety_pass.shape != pred_valid.shape:
            raise ValueError(
                "Internal safety mask shape mismatch: "
                f"pred_valid={pred_valid.shape}, pred_safety_pass={pred_safety_pass.shape}"
            )
        fail_count = int((pred_valid & ~pred_safety_pass).sum())
        print(
            f"Loaded predictions: ep={ep}, frames={end - start}, "
            f"pred_space={pred_space} (cols={n_pred_cols})"
        )
        print(
            "Prediction safety guards: "
            f"pass={int((pred_valid & pred_safety_pass).sum())}, "
            f"fail={fail_count}, reasons={safety_reasons}"
        )

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
    PRED_FAIL_COLOR = np.array([255, 220, 40], dtype=np.uint8)  # yellow — failed guard

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
            assert pred_valid is not None
            assert pred_safety_pass is not None
            future_mask = pred_valid[idx + 1 :]
            future_p = pred_xyz[idx + 1 :][future_mask]
            if future_p.size:
                future_pass = pred_safety_pass[idx + 1 :][future_mask]
                future_colors = np.where(
                    future_pass[:, None],
                    PRED_COLOR[None, :],
                    PRED_FAIL_COLOR[None, :],
                ).astype(np.uint8)
                rr.log(
                    "world/eef_traj/right_pred",
                    rr.Points3D(
                        future_p,
                        colors=future_colors,
                        radii=args.marker_radius,
                    ),
                )
            else:
                rr.log("world/eef_traj/right_pred", rr.Clear(recursive=False))


if __name__ == "__main__":
    main()
