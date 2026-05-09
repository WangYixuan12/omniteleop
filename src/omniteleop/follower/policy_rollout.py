#!/usr/bin/env python3
"""Live policy rollout on a Dexmate right-arm robot.

Drives the right arm + Robotiq Hand-E from a trained LeRobot policy
(ACT or Diffusion). The policy variant (joint vs eef state/action,
absolute vs relative) is auto-detected from ``policy.config`` so a
single script handles every ``dexmate_<state>_<action>`` × {abs, delta}
combo.

Two-rate loop:

  outer ──── 100 Hz (control_rate from config) ─── safety + send
  inner ────  15 Hz  (matches training fps)      ─── policy.select_action

The inner tick is gated on ``round(control_rate / policy_fps)`` outer
ticks, so per training-time ``n_action_steps``/queue dynamics are preserved.

Safety guards (applied to the **command actually being sent**):

  1. Estop snapshot via robot.estop.get_status()
  2. NaN/inf rejection
  3. IK status check  (eef-action variants): solution / not is_collision,
     then validate the 7-DOF right-arm target that will actually be commanded
  4. WorkspaceChecker on the final commanded joint target
  
But no limit_joint_step() for 10-deg step limit here because the arm processor is not used.

Run::

    PYTHONPATH=/home/yixuan/lerobot_yifan/src \\
    python -m omniteleop.follower.policy_rollout \\
        --policy-path /home/yixuan/omniteleop/Dexmate/model/dp/dp_abs_eef_eef/checkpoints/025000/pretrained_model
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import torch
import tyro
from dexbot_utils import RobotInfo
from dexcomm import Node, RateLimiter
from dexcomm.codecs import DictDataCodec
from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot
from dexmotion.ik import LocalPinkIKSolver
from dexmotion.motion_manager import MotionManager
from loguru import logger

from omniteleop.common import get_config
from omniteleop.common.logging import setup_logging
from omniteleop.common.vr_mode_const import (
    INIT_HEAD_JOINTS,
    INIT_LEFT_ARM_JOINTS,
    INIT_RIGHT_ARM_JOINTS,
    INIT_TORSO_JOINTS,
    SAFE_RIGHT_ARM_JOINTS,
)
from omniteleop.follower.robotiq import build_hande_command, send_activate
from omniteleop.follower.workspace_check import WorkspaceChecker

# LeRobot policy stack
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors


# ── small math helpers (mirror port_dexmate_hdf5.py) ──────────────────────────


def _gram_schmidt_6d_to_R(r6: np.ndarray) -> np.ndarray:
    """Recover 3x3 rotation from Zhou et al. 6-D representation [R[:,0], R[:,1]]."""
    a1, a2 = r6[:3], r6[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def _mat_to_pos6d(M: np.ndarray) -> np.ndarray:
    """4x4 SE(3) → ``[tx, ty, tz, R[:,0], R[:,1]]`` (9-D)."""
    return np.concatenate([M[:3, 3], M[:3, 0], M[:3, 1]]).astype(np.float32)


# ── controller ───────────────────────────────────────────────────────────────


class _Mode(Enum):
    HOMING = "homing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    EXIT = "exit"


@dataclass
class _Variant:
    """Variant routing flags derived from policy.config."""

    state_dim: int
    action_dim: int
    state_is_eef: bool  # True if state_dim == 10
    action_is_eef: bool  # True if action_dim == 10
    uses_relative: bool
    relative_mode: Optional[str]


class PolicyRolloutController:
    def __init__(
        self,
        policy_path: str,
        namespace: str = "",
        debug: bool = False,
        config_name: Optional[str] = None,
        workspace_check: bool = True,
        policy_fps: int = 15,
        gripper_threshold: float = 0.5,
        camera_key: str = "right_rgb",
        image_h: int = 240,
        image_w: int = 320,
    ) -> None:
        self.node = Node(name="policy_rollout", namespace=namespace)

        robot_info = RobotInfo()
        self.has_torso = robot_info.has_torso
        self.has_chassis = robot_info.has_chassis

        from omniteleop import LIB_PATH

        config_path = (
            LIB_PATH / "configs" / f"{config_name}.yaml" if config_name else None
        )
        self.config = get_config(config_path)
        # Outer command-hold loop (must be high-frequency for set_joint_pos(wait_time=0)).
        self.control_rate = self.config.get_rate("control_rate", 100)
        self.feedback_rate = self.config.get_rate("feedback_rate", 50)
        # Inner policy/camera tick (matches training fps).
        self.policy_fps = int(policy_fps)
        self._policy_period_ticks = max(1, round(self.control_rate / self.policy_fps))

        self.gripper_threshold = float(gripper_threshold)
        self.camera_key = camera_key
        self.image_h = int(image_h)
        self.image_w = int(image_w)

        # Joint feedback publisher.
        joint_topic = self.config.get_topic("robot_joints")
        self.joint_pub = self.node.create_publisher(
            joint_topic, encoder=DictDataCodec.encode
        )

        # Workspace gate (right arm, in robot base frame).
        self._workspace_checker = WorkspaceChecker() if workspace_check else None
        self._last_workspace_warn_t = 0.0
        if self._workspace_checker is None:
            logger.warning("WorkspaceChecker disabled.")
        self._fixed_motion_joint_limits: dict[str, tuple[float, float]] = {}

        # Hardware + policy.
        self.initialize_robot()
        self.initialize_policy(policy_path)

        self._mode = _Mode.HOMING

    # ── initialisation ────────────────────────────────────────────────────────

    def initialize_robot(self) -> None:
        logger.info("Initialising robot hardware ...")
        robot_configs = get_robot_config()
        robot_configs.sensors["head_camera"].enabled = True
        self.robot = Robot(configs=robot_configs)
        if not self.robot.has_sensor("head_camera"):
            raise RuntimeError(
                "head_camera was enabled in robot config but is not available"
            )
        if self.robot.sensors.head_camera.wait_for_active(timeout=5.0):
            logger.info("Camera streaming started.")
        else:
            logger.warning(
                "head_camera did not report active within 5s; continuing because "
                "it may still become available before rollout starts."
            )
        self._set_home_position()
        # Activate Robotiq Hand-E (build_hande_command alone is movement-only —
        # send_activate first sends the rACT=1 latch).
        send_activate(self.robot.right_arm)
        # Send an explicit OPEN command before caching the gripper state, so the
        # first frame's commanded gripper is consistent with the robot's pose.
        self.robot.right_arm.send_ee_pass_through_message(
            build_hande_command(0.0)
        )
        self._last_gripper_cmd = 0.0
        self._last_cmd_right = np.array(self.robot.right_arm.get_joint_pos())
        logger.success("Robot ready (home, gripper open).")

    def initialize_policy(self, policy_path: str) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        pcfg = PreTrainedConfig.from_pretrained(policy_path)
        pcfg.pretrained_path = policy_path
        self._policy = (
            get_policy_class(pcfg.type).from_pretrained(policy_path).to(device).eval()
        )
        # ACT/Diffusion processors are saved with the checkpoint (no need to
        # supply dataset_stats — only Gr00t needs the override path).
        self._pre, self._post = make_pre_post_processors(
            policy_cfg=pcfg,
            pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        # Auto-detect variant routing.
        from lerobot.utils.constants import OBS_STATE, ACTION

        in_feats = self._policy.config.input_features
        out_feats = self._policy.config.output_features
        state_dim = int(in_feats[OBS_STATE].shape[0])
        action_dim = int(out_feats[ACTION].shape[0])
        self.variant = _Variant(
            state_dim=state_dim,
            action_dim=action_dim,
            state_is_eef=(state_dim == 10),
            action_is_eef=(action_dim == 10),
            uses_relative=bool(getattr(pcfg, "use_relative_actions", False)),
            relative_mode=getattr(pcfg, "relative_action_mode", None),
        )

        # Print chunking like infer_dexmate.py for sanity.
        chunk = getattr(pcfg, "chunk_size", None) or getattr(pcfg, "horizon", None)
        n_act = getattr(pcfg, "n_action_steps", None)
        n_obs = getattr(pcfg, "n_obs_steps", None)
        logger.info(
            f"Policy: type={pcfg.type} state_dim={state_dim} action_dim={action_dim} "
            f"chunk/horizon={chunk} n_action_steps={n_act} n_obs_steps={n_obs} "
            f"relative={self.variant.uses_relative} mode={self.variant.relative_mode}"
        )
        logger.info(
            f"Loop rates: control={self.control_rate}Hz, policy={self.policy_fps}Hz "
            f"(policy fires every {self._policy_period_ticks} control ticks)"
        )

        # MotionManager + IK: required if EITHER state OR action is in EEF space
        # (state_is_eef → FK in obs build; action_is_eef → IK in action decode).
        if self.variant.state_is_eef or self.variant.action_is_eef:
            self._init_motion_manager()
        else:
            self._motion_manager = None
            self._ik_solver = None

        # Reset action queue at start.
        self._policy.reset()

    def _init_motion_manager(self) -> None:
        qpos_dict = self._motion_manager_state_dict()
        self._motion_manager = MotionManager(
            init_visualizer=False,
            # Lock the floating base so dexmotion keeps torso/head in the
            # Pinocchio model. Without this, set_joint_pos() is arm-only and
            # rejects torso_j* entries from the robot state dict.
            joint_regions_to_lock=["BASE"],
            # Policy rollout observes/commands only the right EEF. If the
            # solver is left at its default ["L_ee", "R_ee"], right-only IK
            # targets are rejected as missing L_ee.
            target_frames=["R_ee"],
            initial_joint_configuration_dict=qpos_dict,
        )
        if self._motion_manager.pin_robot is None:
            raise ValueError("MotionManager failed to initialize")
        self._ik_solver = self._motion_manager.local_ik_solver
        if not isinstance(self._ik_solver, LocalPinkIKSolver):
            raise ValueError("LocalPinkIKSolver not available from MotionManager")
        self._cache_fixed_motion_joint_limits()
        self._motion_manager.set_joint_pos(self._motion_manager_state_dict())
        logger.info("MotionManager + LocalPinkIKSolver initialised.")

    def _cache_fixed_motion_joint_limits(self) -> None:
        """Cache dexmotion limits only for fixed non-commanded model context."""
        assert self._motion_manager is not None
        model = self._motion_manager.pin_robot.model
        name_to_idx = getattr(self._motion_manager, "_joint_name_to_q_idx_dict", {})
        fixed_names = {
            *(f"L_arm_j{i}" for i in range(1, 8)),
            *(f"torso_j{i}" for i in range(1, 4)),
            *(f"head_j{i}" for i in range(1, 4)),
        }
        self._fixed_motion_joint_limits = {
            name: (
                float(model.lowerPositionLimit[idx]),
                float(model.upperPositionLimit[idx]),
            )
            for name, idx in name_to_idx.items()
            if name in fixed_names
        }

    def _fixed_motion_joint_value(self, name: str, value: float) -> float:
        """Return a dexmotion-safe fixed context value without checking hardware."""
        limits = self._fixed_motion_joint_limits.get(name)
        if limits is None:
            return float(value)
        lower, upper = limits
        eps = 1e-4
        return float(np.clip(value, lower + eps, upper - eps))

    def _motion_manager_state_dict(self) -> dict[str, float]:
        """Right-arm live state plus fixed non-commanded joints for FK/IK."""
        qd = {
            f"R_arm_j{i + 1}": float(q)
            for i, q in enumerate(self.robot.right_arm.get_joint_pos())
        }
        qd.update(
            {
                f"L_arm_j{i + 1}": self._fixed_motion_joint_value(
                    f"L_arm_j{i + 1}", float(q)
                )
                for i, q in enumerate(INIT_LEFT_ARM_JOINTS)
            }
        )
        qd.update(
            {
                f"torso_j{i + 1}": self._fixed_motion_joint_value(
                    f"torso_j{i + 1}", float(q)
                )
                for i, q in enumerate(INIT_TORSO_JOINTS)
            }
        )
        qd.update(
            {
                f"head_j{i + 1}": self._fixed_motion_joint_value(
                    f"head_j{i + 1}", float(q)
                )
                for i, q in enumerate(INIT_HEAD_JOINTS)
            }
        )
        return qd

    def _set_home_position(self) -> None:
        """Interpolate torso/arms/head to INIT_*_JOINTS (right arm via SAFE_RIGHT_ARM_JOINTS)."""
        time.sleep(0.1)
        joint_delta = 0.01

        # Move torso directly to INIT_TORSO_JOINTS
        curr_torso = np.array(self.robot.torso.get_joint_pos())
        steps = max(1, int(np.max(np.abs(np.array(INIT_TORSO_JOINTS) - curr_torso)) / joint_delta))
        for i in range(steps):
            interp = curr_torso + (np.array(INIT_TORSO_JOINTS) - curr_torso) * (i + 1) / steps
            self.robot.torso.set_joint_pos(interp.tolist(), wait_time=0.1, exit_on_reach=True)

        # Left arm: go directly to INIT_LEFT_ARM_JOINTS
        curr = np.array(self.robot.left_arm.get_joint_pos())
        waypoint = np.array(INIT_LEFT_ARM_JOINTS)
        steps = max(1, int(np.max(np.abs(waypoint - curr)) / joint_delta))
        for i in range(steps):
            interp = curr + (waypoint - curr) * (i + 1) / steps
            self.robot.left_arm.set_joint_pos(interp.tolist(), wait_time=0.1, exit_on_reach=True)

        # Right arm: move to SAFE_RIGHT_ARM_JOINTS first, then INIT_RIGHT_ARM_JOINTS
        for waypoints in (SAFE_RIGHT_ARM_JOINTS, INIT_RIGHT_ARM_JOINTS):
            curr = np.array(self.robot.right_arm.get_joint_pos())
            waypoint = np.array(waypoints)
            steps = max(1, int(np.max(np.abs(waypoint - curr)) / joint_delta))
            for i in range(steps):
                interp = curr + (waypoint - curr) * (i + 1) / steps
                self.robot.right_arm.set_joint_pos(interp.tolist(), wait_time=0.1, exit_on_reach=True)

        # Move head directly to INIT_HEAD_JOINTS
        curr_head = np.array(self.robot.head.get_joint_pos())
        steps = max(1, int(np.max(np.abs(np.array(INIT_HEAD_JOINTS) - curr_head)) / joint_delta))
        for i in range(steps):
            interp = curr_head + (np.array(INIT_HEAD_JOINTS) - curr_head) * (i + 1) / steps
            self.robot.head.set_joint_pos(interp.tolist(), wait_time=0.1, exit_on_reach=True)
        logger.info("Robot at home position.")

    # ── observation ───────────────────────────────────────────────────────────

    def _read_camera_240x320_uint8(self) -> np.ndarray:
        img_dict = self.robot.sensors.head_camera.get_obs(obs_keys=[self.camera_key])
        # mdp_recorder.py:565 prefixes with "head_"; tolerate both shapes.
        if self.camera_key in img_dict:
            img = img_dict[self.camera_key]
        else:
            img = img_dict[f"head_{self.camera_key}"]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        # cv2.resize takes (W, H).
        return cv2.resize(img, (self.image_w, self.image_h), interpolation=cv2.INTER_AREA)

    def _current_eef_pose(self) -> np.ndarray:
        """4x4 SE(3) of right_ee in robot base frame, via MotionManager FK."""
        assert self._motion_manager is not None
        self._motion_manager.set_joint_pos(self._motion_manager_state_dict())
        fk = self._motion_manager.fk(
            frame_names=self._motion_manager.target_frames,
            qpos=self._motion_manager.get_joint_pos(),
        )
        return fk["R_ee"].np  # type: ignore[attr-defined]

    def _build_observation(self) -> dict:
        """Construct an unbatched dataset-style sample for ``self._pre()``."""
        right_arm = np.asarray(self.robot.right_arm.get_joint_pos(), dtype=np.float32)
        if self.variant.state_is_eef:
            T_ee = self._current_eef_pose()
            state = np.concatenate(
                [_mat_to_pos6d(T_ee), [self._last_gripper_cmd]]
            ).astype(np.float32)
        else:
            state = np.concatenate(
                [right_arm, [self._last_gripper_cmd]]
            ).astype(np.float32)

        img_hwc = self._read_camera_240x320_uint8()
        # Mirror what LeRobotDataset.__getitem__ returns: CHW float32 in [0, 1].
        img_chw = (
            torch.from_numpy(img_hwc).permute(2, 0, 1).float() / 255.0
        )

        return {
            "observation.state": torch.from_numpy(state),
            f"observation.images.{self.camera_key}": img_chw,
        }

    # ── policy step ──────────────────────────────────────────────────────────

    def _policy_step(self) -> np.ndarray:
        """Run one policy tick: returns (action_dim,) numpy in absolute space."""
        sample = self._build_observation()
        sample = self._pre(sample)
        with torch.no_grad():
            a = self._policy.select_action(sample)
        a_abs = self._post(a)
        return a_abs.squeeze(0).detach().cpu().numpy()

    def _action_to_joint_target(
        self, action: np.ndarray
    ) -> tuple[Optional[np.ndarray], float, str]:
        """Convert policy output to (joint_target_7, gripper, reason).

        Returns (None, _, reason) if IK fails — caller freezes on this tick.
        """
        gripper = float(action[-1])
        if not self.variant.action_is_eef:
            joint_target = action[:7].astype(np.float32)
            return joint_target, gripper, "ok"

        # EEF action: rebuild SE(3), run IK with status checks.
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _gram_schmidt_6d_to_R(action[3:9].astype(np.float64))
        T[:3, 3] = action[:3].astype(np.float64)
        assert self._ik_solver is not None
        assert self._motion_manager is not None
        self._motion_manager.set_joint_pos(self._motion_manager_state_dict())
        try:
            solution, is_collision, within_limits = self._ik_solver.solve_ik(
                target_pose_dict={"R_ee": T}
            )
        except Exception as e:  # solver internals can throw on degenerate inputs
            return None, gripper, f"ik_exception: {e}"

        if not solution:
            return None, gripper, "ik_no_solution"
        if is_collision:
            return None, gripper, "ik_collision"

        try:
            joint_target = np.array(
                [solution[f"R_arm_j{i + 1}"] for i in range(7)], dtype=np.float32
            )
        except KeyError as e:
            return None, gripper, f"ik_missing_joint:{e}"

        ok, why = self._check_right_arm_joint_target(joint_target)
        if not ok:
            return None, gripper, why
        if not within_limits:
            logger.debug(
                "Ignoring IK within_limits=False because the commanded "
                "7-DOF right arm target passed right-arm checks."
            )
        return joint_target, gripper, "ok"

    # ── safety ───────────────────────────────────────────────────────────────

    def _check_right_arm_joint_target(
        self, joint_target: np.ndarray
    ) -> tuple[bool, str]:
        """Validate only the 7-DOF right-arm command produced by IK."""
        if joint_target.shape != (7,):
            return False, f"right_arm_bad_shape:{joint_target.shape}"
        if not np.all(np.isfinite(joint_target)):
            return False, "right_arm_non_finite"

        jl = getattr(self.robot.right_arm, "_joint_limit", None)
        if jl is not None:
            lower, upper = jl[:, 0], jl[:, 1]
            if np.any(joint_target < lower) or np.any(joint_target > upper):
                return False, "right_arm_outside_limits"
        return True, "ok"

    def _check_safety(
        self,
        command_target: np.ndarray,
    ) -> tuple[bool, str]:
        """Run all guards on the actual command being sent."""
        # 1) NaN / inf
        if not np.all(np.isfinite(command_target)):
            return False, "non_finite"

        # 2) Workspace check (right arm only, in robot base frame). Head/torso
        # are held at their INIT poses for the duration of the rollout, so we
        # use those constants for FK rather than the current measurements
        # (matches vr_robot_controller.py).
        if self._workspace_checker is not None:
            in_bounds, eef_xyz = self._workspace_checker.is_in_workspace(
                right_arm=command_target.tolist(),
                head=list(INIT_HEAD_JOINTS),
                torso=list(INIT_TORSO_JOINTS),
            )
            if not in_bounds:
                now = time.monotonic()
                if now - self._last_workspace_warn_t > 0.5:
                    self._last_workspace_warn_t = now
                    bounds = self._workspace_checker.bounds
                    logger.warning(
                        f"Right EEF out of workspace: xyz={eef_xyz.round(3).tolist()} "
                        f"(bounds x{bounds['x']}, y{bounds['y']}, z{bounds['z']})"
                    )
                return False, "workspace_out_of_bounds"

        return True, "ok"

    # ── joint feedback thread (mirrors vr_robot_controller) ──────────────────

    def _publish_joint_feedback(self) -> None:
        rate = RateLimiter(self.feedback_rate)
        logger.info(f"Joint feedback publishing at {self.feedback_rate} Hz")
        while self._feedback_running:
            try:
                positions = {
                    "left_arm": self.robot.left_arm.get_joint_pos().tolist(),
                    "right_arm": self.robot.right_arm.get_joint_pos().tolist(),
                    "head": self.robot.head.get_joint_pos().tolist(),
                }
                if self.has_torso:
                    positions["torso"] = self.robot.torso.get_joint_pos().tolist()
                self.joint_pub.publish(
                    {"timestamp_ns": time.time_ns(), "joints": positions}
                )
            except Exception as e:
                logger.warning(f"Joint feedback error: {e}")
            rate.sleep()

    # ── control loop ──────────────────────────────────────────────────────────

    def run(self) -> None:
        import threading

        self._feedback_running = True
        feedback_thread = threading.Thread(
            target=self._publish_joint_feedback, daemon=True, name="JointFeedback"
        )
        feedback_thread.start()

        # Manual ENTER trigger so the operator can stand back.
        try:
            input(
                "\n>>> Robot at home, gripper open. Press ENTER to start policy "
                "rollout (Ctrl-C to abort) <<<\n"
            )
        except (EOFError, KeyboardInterrupt):
            logger.info("Aborted before starting.")
            self.cleanup()
            return

        self._mode = _Mode.RUNNING
        rate = RateLimiter(self.control_rate)
        logger.info(f"Control loop at {self.control_rate} Hz")

        # Targets that persist between policy ticks (held flat).
        joint_target = self._last_cmd_right.copy()
        gripper_target = self._last_gripper_cmd
        consecutive_rejects = 0
        REJECT_LIMIT = 30  # ~0.3 s at 100 Hz

        try:
            tick = 0
            while True:
                # Estop snapshot every tick (read first; bail before any send).
                est = self.robot.estop.get_status()
                if est.get("button_pressed") or est.get("software_estop_enabled"):
                    if self._mode != _Mode.PAUSED:
                        logger.warning(f"Estop active: {est}; freezing.")
                        self._mode = _Mode.PAUSED
                    rate.sleep()
                    tick += 1
                    continue
                if self._mode == _Mode.PAUSED:
                    logger.info("Estop released; resuming.")
                    self._policy.reset()  # fresh queue after pause
                    self._mode = _Mode.RUNNING

                # ─── inner: policy at policy_fps ──────────────────────────────
                if tick % self._policy_period_ticks == 0:
                    try:
                        action = self._policy_step()
                    except Exception as e:
                        logger.error(f"Policy step failed: {e}")
                        rate.sleep()
                        tick += 1
                        continue

                    cand_target, cand_gripper, reason = self._action_to_joint_target(action)
                    if cand_target is None:
                        consecutive_rejects += 1
                        logger.warning(f"Action decode rejected: {reason}")
                        if consecutive_rejects >= REJECT_LIMIT:
                            logger.error(
                                f"{consecutive_rejects} consecutive rejects — exiting."
                            )
                            break
                    else:
                        joint_target = cand_target
                        gripper_target = (
                            1.0 if cand_gripper >= self.gripper_threshold else 0.0
                        )

                # ─── outer: safety → send ────────────────────────────────────
                command_target = joint_target

                ok, why = self._check_safety(command_target)
                if ok:
                    self.robot.right_arm.set_joint_pos(command_target.tolist())
                    self.robot.right_arm.send_ee_pass_through_message(
                        build_hande_command(gripper_target)
                    )
                    self._last_cmd_right = command_target
                    self._last_gripper_cmd = gripper_target
                    consecutive_rejects = 0
                else:
                    consecutive_rejects += 1
                    if consecutive_rejects % 20 == 1:
                        logger.warning(f"Safety rejected: {why}; holding last command.")
                    # Re-send last good joint target to keep the motor live.
                    self.robot.right_arm.set_joint_pos(self._last_cmd_right.tolist())
                    if consecutive_rejects >= REJECT_LIMIT:
                        logger.error(
                            f"{consecutive_rejects} consecutive safety rejects — exiting."
                        )
                        break

                tick += 1
                rate.sleep()

        except KeyboardInterrupt:
            logger.info("PolicyRolloutController stopped by user.")
        finally:
            self._feedback_running = False
            feedback_thread.join(timeout=2.0)
            self.cleanup()

    def cleanup(self) -> None:
        if hasattr(self, "robot") and self.robot is not None:
            try:
                self.robot.shutdown()
            except Exception as e:
                logger.warning(f"robot.shutdown() raised: {e}")
        try:
            self.node.shutdown()
        except Exception as e:
            logger.warning(f"node.shutdown() raised: {e}")
        logger.info("PolicyRolloutController cleaned up.")


# ── CLI entry point ──────────────────────────────────────────────────────────


def main(
    policy_path: str,
    namespace: str = "",
    debug: bool = False,
    config_name: Optional[str] = None,
    workspace_check: bool = True,
    policy_fps: int = 15,
    gripper_threshold: float = 0.5,
) -> None:
    """Run live policy rollout on a Dexmate right arm.

    Args:
        policy_path: Path to a LeRobot ``pretrained_model`` directory
            (e.g. ``.../checkpoints/025000/pretrained_model``).
        namespace: Zenoh namespace.
        debug: Verbose logging.
        config_name: Optional omniteleop config file basename
            (under ``omniteleop/configs/``).
        workspace_check: Gate right-arm commands on the configured Cartesian
            workspace bounds. Default on.
        policy_fps: Policy/camera tick rate. Match the dataset fps used in
            training (15 for our Dexmate variants).
        gripper_threshold: Binarise the policy's gripper output at this value
            (matches the dataset's binarisation in port_dexmate_hdf5.py).
    """
    setup_logging(debug)
    ctrl = PolicyRolloutController(
        policy_path=policy_path,
        namespace=namespace,
        debug=debug,
        config_name=config_name,
        workspace_check=workspace_check,
        policy_fps=policy_fps,
        gripper_threshold=gripper_threshold,
    )
    ctrl.run()


if __name__ == "__main__":
    sys.exit(tyro.cli(main))
