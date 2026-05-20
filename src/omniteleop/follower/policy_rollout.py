#!/usr/bin/env python3
"""Live policy rollout on a Dexmate right-arm robot.

Drives the right arm + Robotiq Hand-E from a trained LeRobot policy
(ACT or Diffusion). The policy variant (joint vs eef state/action,
absolute vs relative) is auto-detected from ``policy.config`` so a
single script handles every ``dexmate_<state>_<action>`` × {abs, delta}
combo.

Two-rate loop:

  outer ──── 300 Hz (control_rate from config) ─── send held command
  inner ────  15 Hz  (matches training fps)      ─── policy + safety + decode

The inner tick is gated on ``round(control_rate / policy_fps)`` outer
ticks. By default ACT checkpoints keep their saved queue/ensemble dynamics;
pass ACT override options only when you want to change the checkpoint config
at deploy time.

Both joint- and EEF-action variants share the same decode/safety pipeline
in ``_action_to_joint_target`` — for EEF actions the raw 7-DOF target comes
from ``MotionManager.ik(type="pink")`` (seeded from the chained
last-commanded state, mirroring ``vr_reader._solve_and_apply_arm_ik``); for
joint actions it comes straight from ``action[:7]``. Both then go through
``ArmProcessor.limit_joint_step`` (10°/tick clamp against the chained
state) → ``_check_right_arm_joint_target`` → ``_check_safety`` before
``ArmProcessor.apply_positions`` commits the new chained pose.
``set_joint_pos`` to the arm matches ``vr_robot_controller`` (no
``wait_time``).

Safety guards (all run inside ``_action_to_joint_target``; the IK chain
``mm.right_arm`` / ``_chained_right_arm_qpos`` advances only after every
guard passes, so a rejected tick leaves the chain anchored to the last
safe pose for next tick's clamp):

  1. Estop snapshot via robot.estop.get_status() (outer loop); on release
     the IK chain re-syncs to observed.
  2. IK status check (EEF-action variants): solution / not is_collision /
     within_limits.
  3. ``ArmProcessor.limit_joint_step`` — 10°/tick clamp on the raw 7-DOF
     target (IK output for EEF actions, ``action[:7]`` for joint actions).
  4. ``_check_right_arm_joint_target`` — shape, NaN/inf, hardware joint
     limits on the post-clamp target.
  5. ``_check_safety`` — NaN/inf and ``WorkspaceChecker`` (Cartesian EEF
     bounds with head/torso held at ``INIT_*``) on the post-clamp target.

Any rejection bumps ``consecutive_rejects``; the loop exits when it hits
``REJECT_LIMIT``.

Run::

    # New repo (lerobot_original) + new model layout under ~/Dexmate/model/.
    PYTHONPATH=/home/yixuan/lerobot_original/src \\
    python -m omniteleop.follower.policy_rollout \\
        --policy-path /home/yixuan/Dexmate/model/act/dexmate_eef_eef/checkpoints/200000/pretrained_model

    # Diffusion variant:
    PYTHONPATH=/home/yixuan/lerobot_original/src \\
    python -m omniteleop.follower.policy_rollout \\
        --policy-path /home/yixuan/Dexmate/model/dp/dexmate_eef_eef/checkpoints/050000/pretrained_model

    # Relative-action ACT (chunk-anchor pinning engages automatically):
    PYTHONPATH=/home/yixuan/lerobot_original/src \\
    python -m omniteleop.follower.policy_rollout \\
        --policy-path /home/yixuan/Dexmate/model/act/dexmate_eef_eef_relative/checkpoints/050000/pretrained_model
"""

from __future__ import annotations

import pathlib
import signal
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
from omniteleop.common.recorder import EpisodeRecorder
from omniteleop.common.vr_mode_const import (
    INIT_HEAD_JOINTS,
    INIT_LEFT_ARM_JOINTS,
    INIT_RIGHT_ARM_JOINTS,
    INIT_TORSO_JOINTS,
    SAFE_RIGHT_ARM_JOINTS,
)
from omniteleop.follower.component_processors import ArmProcessor
from omniteleop.follower.robotiq import (
    RobotiqStatusMonitor,
    build_hande_command,
    poll_gripper_status,
    send_activate,
)
from omniteleop.follower.workspace_check import WorkspaceChecker

# LeRobot policy stack
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import RelativeActionsProcessorStep


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


def _next_record_subdir(record_root: str | pathlib.Path) -> pathlib.Path:
    """Reserve the next numeric rollout subdirectory under ``record_root``."""
    root = pathlib.Path(record_root)
    root.mkdir(parents=True, exist_ok=True)

    existing_indices = (
        int(path.name)
        for path in root.iterdir()
        if path.is_dir() and path.name.isdecimal()
    )
    next_idx = max(existing_indices, default=-1) + 1
    while True:
        candidate = root / f"{next_idx:01d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            next_idx += 1
            continue
        return candidate


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
    # Action-side last-dim indices that stay absolute when relative mode is on
    # (lerobot_original: PreTrainedConfig.relative_exclude_dims["action"]). None
    # when the policy was trained without relative actions.
    relative_exclude_action_dims: Optional[list[int]]


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
        camera_key: str = "left_rgb",
        image_h: int = 240,
        image_w: int = 320,
        act_n_action_steps: Optional[int] = None,
        act_temporal_ensemble_coeff: Optional[float] = None,
        record: bool = True,
        record_dir: Optional[str] = None,
    ) -> None:
        self.node = Node(name="policy_rollout", namespace=namespace)

        self._robot_info = RobotInfo()
        self.has_torso = self._robot_info.has_torso
        self.has_chassis = self._robot_info.has_chassis

        from omniteleop import LIB_PATH

        config_path = (
            LIB_PATH / "configs" / f"{config_name}.yaml" if config_name else None
        )
        self.config = get_config(config_path)
        # Outer command-hold loop (must be high-frequency for set_joint_pos(wait_time=0)).
        self.control_rate = self.config.get_rate("control_rate", 300)
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

        # IK-chain state. Holds the last safe_right we sent (= post-clamp IK
        # output). Seeded from observed joints after homing and on estop
        # release. Mirrors the leader's implicit chain through mm.right_arm.
        # initialize_robot() writes it; _init_motion_manager() writes
        # self._right_proc. Both run inside this __init__.
        self._chained_right_arm_qpos: Optional[np.ndarray] = None
        self._right_proc: Optional[ArmProcessor] = None

        # Hardware + policy.
        self.initialize_robot()
        self.initialize_policy(
            policy_path,
            act_n_action_steps=act_n_action_steps,
            act_temporal_ensemble_coeff=act_temporal_ensemble_coeff,
        )

        if record:
            if record_dir is None:
                tag = pathlib.Path(policy_path).parents[2].name
                record_dir = f"/home/yixuan/Dexmate/deploy/{tag}"
            rollout_record_dir = _next_record_subdir(record_dir)
            self._recorder: Optional[EpisodeRecorder] = EpisodeRecorder(
                str(rollout_record_dir)
            )
            logger.info(f"Recording rollouts to {rollout_record_dir}")
        else:
            self._recorder = None

        self._last_obs_left_rgb: Optional[np.ndarray] = None
        self._last_obs_depth_u16: Optional[np.ndarray] = None
        self._last_obs_right_arm_qpos: Optional[np.ndarray] = None
        self._last_obs_eef_9d: Optional[np.ndarray] = None
        self._last_obs_gripper = None  # np.float32; set in _build_observation

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
        # Initialise the IK chain at the observed home pose.
        self._chained_right_arm_qpos = self._last_cmd_right.astype(np.float32)

        # Robotiq Hand-E achieved-position sensor. Training-time
        # observation.state[9] is the raw float `gPO/255` (range ≈ [0.012, 0.655]
        # on this hardware), NOT the binary command. Feeding the binary command
        # at inference normalizes to ≈ +2.07 under MIN_MAX (way outside training
        # range) and destabilizes diffusion policies in particular. Mirror the
        # leader's pattern (vr_reader.py:1383, :1392): synchronous warmup poll
        # to seed the cache, then RobotiqStatusMonitor for async sampling. The
        # monitor's subscriber filters FC16 ACKs from FC03 status replies, so
        # it coexists with our outbound gripper writes on the same topic.
        warm = poll_gripper_status(
            self.robot.right_arm, function_code=0x03, timeout_s=0.5
        )
        if warm is None:
            logger.warning(
                "Gripper FC03 warmup failed — falling back to 0.0 until the "
                "monitor catches up. Verify enable_ee_pass_through=True."
            )
            self._last_obs_grip_right = 0.0
        else:
            self._last_obs_grip_right = float(warm["actual"])
        self._grip_monitor = RobotiqStatusMonitor(
            self.robot.right_arm, side="right", function_code=0x03
        )
        self._grip_monitor.send_status_request()
        logger.success(
            f"Robot ready (home, gripper open, achieved={self._last_obs_grip_right:.3f})."
        )

    def initialize_policy(
        self,
        policy_path: str,
        act_n_action_steps: Optional[int] = None,
        act_temporal_ensemble_coeff: Optional[float] = None,
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        pcfg = PreTrainedConfig.from_pretrained(policy_path)
        if pcfg.type == "act":
            cli_overrides: list[str] = []
            if act_n_action_steps is not None:
                cli_overrides.append(f"--n_action_steps={int(act_n_action_steps)}")
            if act_temporal_ensemble_coeff is not None:
                cli_overrides.append(
                    f"--temporal_ensemble_coeff={float(act_temporal_ensemble_coeff)}"
                )
            if cli_overrides:
                pcfg = PreTrainedConfig.from_pretrained(
                    policy_path,
                    cli_overrides=cli_overrides,
                )
                logger.info(
                    "Applied ACT checkpoint config overrides: "
                    + ", ".join(arg.removeprefix("--") for arg in cli_overrides)
                )
        pcfg.pretrained_path = policy_path
        self._policy = (
            get_policy_class(pcfg.type)
            .from_pretrained(policy_path, config=pcfg)
            .to(device)
            .eval()
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
        rel_exclude_map = getattr(pcfg, "relative_exclude_dims", None) or {}
        rel_exclude_action = (
            list(rel_exclude_map.get("action", [])) if isinstance(rel_exclude_map, dict) else None
        )
        self.variant = _Variant(
            state_dim=state_dim,
            action_dim=action_dim,
            state_is_eef=(state_dim == 10),
            action_is_eef=(action_dim == 10),
            uses_relative=bool(getattr(pcfg, "use_relative_actions", False)),
            relative_exclude_action_dims=rel_exclude_action,
        )

        # Locate the RelativeActionsProcessorStep so the control loop can pin its
        # cached state to the chunk anchor between in-chunk pops (see _policy_step).
        # Without this, post() would compute `a_t_abs = a_t_rel + s_t` instead of
        # `a_t_rel + s_replan`, causing action drift across the chunk.
        self._relative_step = next(
            (s for s in self._pre.steps if isinstance(s, RelativeActionsProcessorStep)),
            None,
        )
        self._chunk_anchor_active = (
            self._relative_step is not None and self._relative_step.enabled
        )
        self._chunk_ref_state: Optional[torch.Tensor] = None

        # Print chunking like infer_dexmate.py for sanity.
        chunk = getattr(pcfg, "chunk_size", None) or getattr(pcfg, "horizon", None)
        n_act = getattr(pcfg, "n_action_steps", None)
        n_obs = getattr(pcfg, "n_obs_steps", None)
        self._n_action_steps = max(1, int(n_act)) if n_act is not None else 1
        logger.info(
            f"Policy: type={pcfg.type} state_dim={state_dim} action_dim={action_dim} "
            f"chunk/horizon={chunk} n_action_steps={n_act} n_obs_steps={n_obs} "
            f"relative={self.variant.uses_relative} "
            f"relative_exclude_action_dims={self.variant.relative_exclude_action_dims} "
            f"chunk_anchor_active={self._chunk_anchor_active}"
        )
        logger.info(
            f"Loop rates: control={self.control_rate}Hz, policy={self.policy_fps}Hz "
            f"(policy fires every {self._policy_period_ticks} control ticks)"
        )

        # MotionManager + IK + ArmProcessor: always initialised because every
        # variant runs ArmProcessor.limit_joint_step (10°/tick clamp) and the
        # workspace + joint-limit safety gates before sending. EEF state/action
        # variants additionally use IK (action decode) and FK (obs build).
        self._init_motion_manager()

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
        # Right-arm processor: provides limit_joint_step (10°/tick clamp) and
        # apply_positions, exactly as on the leader (vr_reader.py:377).
        self._right_proc = ArmProcessor(
            "right", self.config, self._motion_manager, self._robot_info, "vr"
        )
        logger.info("MotionManager + LocalPinkIKSolver + ArmProcessor initialised.")

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

    def _motion_manager_state_dict(
        self, right_arm: Optional[np.ndarray] = None
    ) -> dict[str, float]:
        """Right-arm state plus fixed non-commanded joints for FK/IK.

        ``right_arm=None`` reads live observed joints (used for FK on the
        policy's observation). Pass an explicit array (e.g. the chained
        last-commanded joints) to seed MM before an IK call so
        ``limit_joint_step`` clamps against last-commanded, matching the
        leader's behaviour.
        """
        if right_arm is None:
            right_arm = self.robot.right_arm.get_joint_pos()
        qd = {
            f"R_arm_j{i + 1}": float(q)
            for i, q in enumerate(right_arm)
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

    def _read_head_camera(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read head camera streams for policy input + HDF5 recording.

        One ``get_obs()`` call covers all three streams: the ``camera_key``
        image feeds the policy after resize, while ``left_rgb`` and
        ``depth`` are kept at original resolution for the recorder
        (mirrors ``vr_reader._handle_recording``: depth in metres →
        uint16 millimetres).
        """
        keys = list({self.camera_key, "left_rgb", "depth"})
        obs = self.robot.sensors.head_camera.get_obs(obs_keys=keys)

        # mdp_recorder.py:565 prefixes with "head_"; tolerate both shapes.
        def pick(key: str) -> np.ndarray:
            return obs[key] if key in obs else obs[f"head_{key}"]

        policy_raw = pick(self.camera_key)
        if policy_raw.dtype != np.uint8:
            policy_raw = np.clip(policy_raw, 0, 255).astype(np.uint8)
        # cv2.resize takes (W, H).
        policy_img = cv2.resize(
            policy_raw, (self.image_w, self.image_h), interpolation=cv2.INTER_AREA
        )

        left_rgb = pick("left_rgb")
        if left_rgb.dtype != np.uint8:
            left_rgb = np.clip(left_rgb, 0, 255).astype(np.uint8)

        depth_u16 = np.clip(pick("depth") * 1000, 0, 65535).astype(np.uint16)
        return policy_img, left_rgb, depth_u16

    def _current_eef_pose(self) -> np.ndarray:
        """4x4 SE(3) of right_ee in robot base frame, via MotionManager FK."""
        assert self._motion_manager is not None
        self._motion_manager.set_joint_pos(self._motion_manager_state_dict())
        fk = self._motion_manager.fk(
            frame_names=self._motion_manager.target_frames,
            qpos=self._motion_manager.get_joint_pos(),
        )
        return fk["R_ee"].np  # type: ignore[attr-defined]

    def _poll_gripper_status_step(self) -> None:
        """Drain queued FC03 replies and send the next status request.

        Mirrors ``vr_reader._poll_gripper_status_step``. Called once per policy
        tick so ``self._last_obs_grip_right`` is refreshed at ~policy_fps,
        which matches the leader's 15 Hz cadence during recording.
        """
        events = self._grip_monitor.drain_status_events()
        if events:
            self._last_obs_grip_right = float(events[-1]["actual"])
        self._grip_monitor.expire_timeouts(0.5)
        self._grip_monitor.send_status_request()

    def _build_observation(self) -> dict:
        """Construct an unbatched dataset-style sample for ``self._pre()``."""
        # Refresh achieved gripper before reading state — training-time
        # observation.state[9] is the raw FC03 ``actual`` value, not the binary
        # command, so we must mirror what port_dexmate_hdf5.py recorded.
        self._poll_gripper_status_step()
        gripper_obs = self._last_obs_grip_right

        right_arm = np.asarray(self.robot.right_arm.get_joint_pos(), dtype=np.float32)
        if self.variant.state_is_eef:
            T_ee = self._current_eef_pose()
            obs_eef_9d = _mat_to_pos6d(T_ee)
            state = np.concatenate(
                [obs_eef_9d, [gripper_obs]]
            ).astype(np.float32)
        else:
            if self._motion_manager is not None:
                obs_eef_9d = _mat_to_pos6d(self._current_eef_pose())
            else:
                obs_eef_9d = np.full(9, np.nan, dtype=np.float32)
            state = np.concatenate(
                [right_arm, [gripper_obs]]
            ).astype(np.float32)

        img_hwc, left_rgb, depth_u16 = self._read_head_camera()
        self._last_obs_left_rgb = left_rgb
        self._last_obs_depth_u16 = depth_u16
        self._last_obs_right_arm_qpos = right_arm.copy()
        self._last_obs_eef_9d = obs_eef_9d.astype(np.float32)
        # Last element of observation.state (achieved gripper at obs build time).
        self._last_obs_gripper = np.float32(gripper_obs)
        # Mirror what LeRobotDataset.__getitem__ returns: CHW float32 in [0, 1].
        img_chw = (
            torch.from_numpy(img_hwc).permute(2, 0, 1).float() / 255.0
        )

        return {
            "observation.state": torch.from_numpy(state),
            f"observation.images.{self.camera_key}": img_chw,
        }

    # ── policy step ──────────────────────────────────────────────────────────

    def _policy_action_queue_len(self) -> int:
        """Peek at the policy's internal action queue (ACT or Diffusion)."""
        q = getattr(self._policy, "_action_queue", None)
        if q is not None:
            return len(q)
        queues = getattr(self._policy, "_queues", None)
        if queues is not None and "action" in queues:
            return len(queues["action"])
        return -1

    def _policy_step(self) -> tuple[np.ndarray, dict]:
        """Run one policy tick: returns (absolute action, timing info)."""
        qlen_before = self._policy_action_queue_len()
        begin_build_obs_ns = time.time_ns()
        t_obs_start = time.monotonic()
        sample = self._build_observation()
        t_obs_end = time.monotonic()

        sample = self._pre(sample)  # writes _last_state = current obs state

        # In-chunk pop: restore the chunk anchor so post() adds the same state
        # the model used during the relative subtraction at replan time.
        if self._chunk_anchor_active and qlen_before > 0:
            assert self._relative_step is not None  # implied by _chunk_anchor_active
            self._relative_step.set_cached_state(self._chunk_ref_state)

        begin_inference_ns = time.time_ns()
        t_inf_start = time.monotonic()
        with torch.no_grad():
            a = self._policy.select_action(sample)
        t_inf_end = time.monotonic()
        finish_inference_ns = time.time_ns()
        a_abs = self._post(a)

        # On replan, snapshot the (just-set) cache as the chunk anchor for
        # subsequent in-chunk pops in this chunk.
        if self._chunk_anchor_active and qlen_before == 0:
            assert self._relative_step is not None
            cached = self._relative_step.get_cached_state()
            self._chunk_ref_state = (
                cached.detach().clone() if cached is not None else None
            )

        info = {
            # Legacy alias kept so readers that grep for t_obs_ns_wallclock still work.
            "t_obs_ns_wallclock": begin_build_obs_ns,
            "begin_build_obs_ns": begin_build_obs_ns,
            "begin_inference_ns": begin_inference_ns,
            "finish_inference_ns": finish_inference_ns,
            "t_obs_end": t_obs_end,
            "obs_build_ms": (t_obs_end - t_obs_start) * 1000.0,
            "inference_ms": (t_inf_end - t_inf_start) * 1000.0,
            # qlen_before == 0 (or -1 on unknown impl) → this call ran the model.
            "replan": qlen_before <= 0,
        }
        return a_abs.squeeze(0).detach().cpu().numpy(), info

    def _action_to_joint_target(
        self, action: np.ndarray
    ) -> tuple[Optional[np.ndarray], float, str]:
        """Convert policy output to (joint_target_7, gripper, reason).

        Both branches (joint and EEF actions) go through the same safety
        pipeline: seed MM with the chained (last-commanded) state → produce a
        raw 7-DOF target (IK for EEF, action[:7] for joint) →
        ``limit_joint_step`` (10°/tick clamp against the chained state) →
        ``_check_right_arm_joint_target`` (hardware joint limits) →
        ``_check_safety`` (NaN + Cartesian workspace). The IK chain
        (``mm.right_arm`` / ``_chained_right_arm_qpos``) is committed only
        after every guard passes, so a rejected tick leaves the chain anchored
        to the last safe pose for next tick's clamp.

        Returns (None, _, reason) on any rejection — caller exits or freezes
        on this tick.
        """
        gripper = float(action[-1])
        assert self._motion_manager is not None
        assert self._right_proc is not None
        assert self._chained_right_arm_qpos is not None

        # Seed MM with the chained (last-commanded) right_arm so:
        #   - (EEF only) the Pink IK warm-starts from there (matches leader),
        #   - limit_joint_step clamps the raw target against it.
        # mm.ik() does NOT commit its solution back into mm.right_arm, so the
        # MM state stays at the chained value across the call.
        self._motion_manager.set_joint_pos(
            self._motion_manager_state_dict(right_arm=self._chained_right_arm_qpos)
        )

        if self.variant.action_is_eef:
            # EEF action — mirror vr_reader._solve_and_apply_arm_ik exactly.
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = _gram_schmidt_6d_to_R(action[3:9].astype(np.float64))
            T[:3, 3] = action[:3].astype(np.float64)
            try:
                arm_solution, in_collision, within_limits = self._motion_manager.ik(
                    target_pose={"R_ee": T},
                    type="pink",
                )
            except Exception as e:  # solver internals can throw on degenerate inputs
                return None, gripper, f"ik_exception: {e}"

            if not arm_solution:
                return None, gripper, "ik_no_solution"
            if in_collision:
                return None, gripper, "ik_collision"
            if not within_limits:
                return None, gripper, "ik_outside_limits"

            try:
                raw_right = [arm_solution[f"R_arm_j{i}"] for i in range(1, 8)]
            except KeyError as e:
                return None, gripper, f"ik_missing_joint:{e}"
        else:
            # Joint action — raw target straight from the policy.
            raw_right = action[:7].astype(np.float64).tolist()

        # 10°/tick clamp against the chained state.
        safe_right = self._right_proc.limit_joint_step(raw_right)
        joint_target = safe_right.astype(np.float32)

        # Validate the post-clamp target before committing the chain.
        ok, why = self._check_right_arm_joint_target(joint_target)
        if not ok:
            return None, gripper, why
        ok, why = self._check_safety(joint_target)
        if not ok:
            return None, gripper, why

        # Commit chain (mm.right_arm advances → next tick's limit_joint_step
        # clamps against this value; _chained_right_arm_qpos mirrors it so the
        # next IK call re-seeds MM from a known place).
        self._right_proc.apply_positions(safe_right.tolist())
        self._chained_right_arm_qpos = safe_right.astype(np.float32)

        return joint_target, gripper, "ok"

    def _action_eef_records(
        self,
        action: np.ndarray,
        sent_joint: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (SE(3) 4x4, pos6d 9-D) for the policy's proposed eef.

        - action_is_eef: rebuild from the policy's 10-D output.
        - else, when MotionManager is available and ``sent_joint`` is finite:
          FK on the proposed joint vector.
        - else: NaN-filled arrays (shape preserved so np.stack works).
        """
        if self.variant.action_is_eef:
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = _gram_schmidt_6d_to_R(action[3:9].astype(np.float64)).astype(np.float32)
            T[:3, 3] = action[:3].astype(np.float32)
            pos6d = action[:9].astype(np.float32)
            return T, pos6d

        if (
            self._motion_manager is not None
            and sent_joint is not None
            and np.all(np.isfinite(sent_joint))
        ):
            qd = self._motion_manager_state_dict()
            for i in range(7):
                qd[f"R_arm_j{i + 1}"] = float(sent_joint[i])
            self._motion_manager.set_joint_pos(qd)
            fk = self._motion_manager.fk(
                frame_names=self._motion_manager.target_frames,
                qpos=self._motion_manager.get_joint_pos(),
            )
            T = np.asarray(fk["R_ee"].np, dtype=np.float32)  # type: ignore[attr-defined]
            return T, _mat_to_pos6d(T)

        return (
            np.full((4, 4), np.nan, dtype=np.float32),
            np.full((9,), np.nan, dtype=np.float32),
        )

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
                arm_joints=command_target.tolist(),
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

        signal.signal(signal.SIGINT, signal.default_int_handler)

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
        REJECT_LIMIT = 1
        chunk_idx = -1
        last_replan_log_t = 0.0
        if self._recorder is not None:
            self._recorder.start()

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
                    # Drop the stale chunk anchor; the next tick will replan
                    # and capture a new one from the post-resume observation.
                    self._chunk_ref_state = None
                    # Re-sync the IK chain to the observed arm pose — the
                    # operator may have moved the arm while paused. Re-seed
                    # mm.right_arm via apply_positions so the next
                    # limit_joint_step clamps against the live state.
                    if self._right_proc is not None:
                        obs_right = np.array(
                            self.robot.right_arm.get_joint_pos(), dtype=np.float32
                        )
                        self._chained_right_arm_qpos = obs_right
                        self._right_proc.apply_positions(obs_right.tolist())
                    self._mode = _Mode.RUNNING

                # ─── inner: policy at policy_fps ──────────────────────────────
                step_info: Optional[dict] = None
                cand_target: Optional[np.ndarray] = None
                cand_gripper: float = 0.0
                action: Optional[np.ndarray] = None
                if tick % self._policy_period_ticks == 0:
                    try:
                        action, step_info = self._policy_step()
                    except Exception as e:
                        logger.error(f"Policy step failed: {e}")
                        rate.sleep()
                        tick += 1
                        continue

                    cand_target, cand_gripper, reason = self._action_to_joint_target(action)
                    if cand_target is None:
                        consecutive_rejects += 1
                        logger.warning(f"Action rejected: {reason}")
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
                        consecutive_rejects = 0

                # ─── outer: send (all safety guards ran in _action_to_joint_target) ─
                command_target = joint_target
                self.robot.right_arm.set_joint_pos(command_target.tolist())
                publish_command_ns = time.time_ns()
                self.robot.right_arm.send_ee_pass_through_message(
                    build_hande_command(gripper_target)
                )
                t_send = time.monotonic()
                if step_info is not None and step_info["replan"]:
                    if t_send - last_replan_log_t > 0.95:
                        last_replan_log_t = t_send
                        obs_to_send_ms = (t_send - step_info["t_obs_end"]) * 1000.0
                        logger.info(
                            f"chunk replan: obs_build={step_info['obs_build_ms']:.1f}ms "
                            f"inference={step_info['inference_ms']:.1f}ms "
                            f"obs→send={obs_to_send_ms:.1f}ms"
                        )
                self._last_cmd_right = command_target
                self._last_gripper_cmd = gripper_target

                # ─── record (per inner policy tick) ──────────────────────────
                if step_info is not None and action is not None:
                    if step_info["replan"]:
                        chunk_idx = 0
                    else:
                        chunk_idx = (chunk_idx + 1) % self._n_action_steps

                    if self._recorder is not None:
                        try:
                            if not self.variant.action_is_eef:
                                proposed_joint = action[:7].astype(np.float32)
                            elif cand_target is not None:
                                proposed_joint = cand_target.astype(np.float32)
                            else:
                                proposed_joint = np.full(7, np.nan, dtype=np.float32)

                            proposed_gripper = np.float32(
                                1.0 if cand_gripper >= self.gripper_threshold else 0.0
                            )
                            eef_4x4, eef_9d = self._action_eef_records(action, proposed_joint)
                            obs_left_rgb = self._last_obs_left_rgb
                            obs_depth_u16 = self._last_obs_depth_u16
                            obs_qpos = self._last_obs_right_arm_qpos
                            obs_eef_9d = self._last_obs_eef_9d
                            obs_gripper = self._last_obs_gripper
                            assert (
                                obs_left_rgb is not None
                                and obs_depth_u16 is not None
                                and obs_qpos is not None
                                and obs_eef_9d is not None
                                and obs_gripper is not None
                            )
                            self._recorder.record({
                                "time": {
                                    "timestamp_ns": np.int64(step_info["t_obs_ns_wallclock"]),
                                    "begin_build_observation": np.int64(
                                        step_info["begin_build_obs_ns"]
                                    ),
                                    "begin_inference": np.int64(
                                        step_info["begin_inference_ns"]
                                    ),
                                    "finish_inference": np.int64(
                                        step_info["finish_inference_ns"]
                                    ),
                                    "publish_command": np.int64(publish_command_ns),
                                    "obs_flag": np.bool_(step_info["replan"]),
                                    "action_number": np.int32(chunk_idx),
                                },
                                "obs": {
                                    "joint": {"right_arm": obs_qpos},
                                    "eef_9d": {"right": obs_eef_9d},
                                    "gripper": {"right": obs_gripper},
                                    "images": {
                                        "left_rgb": obs_left_rgb,
                                        "depth": obs_depth_u16,
                                    },
                                },
                                "action": {
                                    "eef": {"right": eef_4x4},
                                    "eef_9d": {"right": eef_9d},
                                    "joint": {"right_arm": proposed_joint},
                                    "gripper": {"right": proposed_gripper},
                                },
                            })
                        except Exception as e:
                            logger.warning(f"Recorder.record() failed: {e}")

                tick += 1
                rate.sleep()

        except KeyboardInterrupt:
            logger.info("PolicyRolloutController stopped by user.")
        finally:
            self._feedback_running = False
            feedback_thread.join(timeout=2.0)
            self.cleanup()

    def cleanup(self) -> None:
        recorder = getattr(self, "_recorder", None)
        if recorder is not None and recorder.recording:
            path = recorder.stop()
            if path is not None:
                deadline = time.monotonic() + 10.0
                while recorder.saving and time.monotonic() < deadline:
                    time.sleep(0.05)
                if recorder.saving:
                    logger.warning(f"Recorder still saving at exit: {path}")
                else:
                    logger.info(f"Recorded rollout → {path}")
        monitor = getattr(self, "_grip_monitor", None)
        if monitor is not None:
            try:
                monitor.close()
            except Exception as e:
                logger.warning(f"grip_monitor.close() raised: {e}")
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
    act_n_action_steps: Optional[int] = None,
    act_temporal_ensemble_coeff: Optional[float] = None,
    record: bool = True,
    record_dir: Optional[str] = None,
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
        act_n_action_steps: ACT-only checkpoint config override. Defaults to
            None, preserving the checkpoint value.
        act_temporal_ensemble_coeff: ACT-only checkpoint config override.
            Defaults to None, preserving the checkpoint value.
        record: Dump per-policy-tick frames to HDF5 for debug analysis.
        record_dir: Override save root. Each rollout records under the next
            numeric subdirectory (``0``, ``1``, ...). Defaults to
            ``/home/yixuan/Dexmate/deploy/<model-variant>`` derived from
            ``policy_path``.
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
        act_n_action_steps=act_n_action_steps,
        act_temporal_ensemble_coeff=act_temporal_ensemble_coeff,
        record=record,
        record_dir=record_dir,
    )
    ctrl.run()


if __name__ == "__main__":
    sys.exit(tyro.cli(main))
