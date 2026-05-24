#!/usr/bin/env python3
"""Live bimanual policy rollout on a Dexmate robot.

Drives BOTH arms + their Robotiq Hand-E grippers from a trained LeRobot
policy (ACT or Diffusion). The policy variant (joint vs eef state/action,
absolute vs relative) is auto-detected from ``policy.config`` so a single
script handles every ``dexmate_<state>_<action>`` × {abs, delta} combo.

Layout (matches ``port_dexmate_hdf5.py`` — LEFT block then RIGHT):

  eef   (20-D): [L_pos(3), L_rot6d(3..8), L_grip@9,
                 R_pos(10..12), R_rot6d(13..18), R_grip@19]
  joint (16-D): [L_arm(7), L_grip@7, R_arm(7), R_grip@15]

Cameras (driven by ``policy.config.input_features``):

  observation.images.head_rgb  — head ZED-X Mini left eye, cropped to the
      training region [220:600, 150:800] then resized to the model's HxW.
  observation.images.wrist_rgb — wrist ZED-M left eye (full frame, resized),
      requires a ZED-SDK publisher on ``sensors/wrist_zedm/*`` (optional;
      only read when the policy declares the feature).

Two-rate loop:

  outer ──── 100 Hz (control_rate from config) ─── send held commands
  inner ────  15 Hz  (matches training fps)      ─── policy + safety + decode

The inner tick is gated on ``round(control_rate / policy_fps)`` outer ticks.

EEF-action variants solve BOTH arms in a single
``MotionManager.ik(target_pose={"L_ee", "R_ee"}, type="pink")`` call (so the
solver accounts for inter-arm self-collision, mirroring
``vr_reader._solve_and_apply_arm_ik``); joint-action variants take the raw
7-DOF targets straight from the action blocks. Each arm's raw target then
goes through its own ``ArmProcessor.limit_joint_step`` (10°/tick clamp against
the chained state) → ``_check_arm_joint_target`` → ``_check_safety`` before the
IK chains commit.

Safety guards (all run inside ``_action_to_joint_targets``; the IK chains
``mm.{left,right}_arm`` / ``_chained_{left,right}_arm_qpos`` advance only after
EVERY guard passes for BOTH arms, so a rejected tick leaves both chains
anchored to the last safe pose for next tick's clamp):

  1. Estop snapshot via robot.estop.get_status() (outer loop); on release
     both IK chains re-sync to observed.
  2. IK status check (EEF-action variants): solution / not is_collision /
     within_limits — coupled across both arms.
  3. ``ArmProcessor.limit_joint_step`` — 10°/tick clamp on each arm's raw
     7-DOF target.
  4. ``_check_arm_joint_target`` — shape, NaN/inf, hardware joint limits on
     each post-clamp target.
  5. ``_check_safety`` — NaN/inf and per-arm ``WorkspaceChecker`` (Cartesian
     EEF bounds with head/torso held at ``INIT_*``).

Any rejection (either arm) bumps ``consecutive_rejects``; the loop exits when
it hits ``REJECT_LIMIT``.

Run::

    PYTHONPATH=/home/yixuan/lerobot_original/src \\
    python -m omniteleop.follower.policy_rollout \\
        --policy-path /home/yixuan/Dexmate/model/act/dexmate_eef_eef_abs/checkpoints/200000/pretrained_model
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
from dexbot_utils.configs.components.sensors.cameras import ZedXCameraConfig
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
    SAFE_LEFT_ARM_JOINTS,
    SAFE_RIGHT_ARM_JOINTS,
)
from omniteleop.follower.component_processors import ArmProcessor
from omniteleop.follower.robotiq import (
    RobotiqStatusMonitor,
    build_hande_command,
    poll_gripper_status,
    send_activate,
)
from omniteleop.follower.workspace_check import (
    DEFAULT_LEFT_BOUNDS,
    DEFAULT_LEFT_LINK,
    WorkspaceChecker,
)

# LeRobot policy stack
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import RelativeActionsProcessorStep

# Wrist ZED-M sensor id + obs key (mirrors vr_reader._WRIST_SENSOR_ID and the
# left-eye obs key it consumes from sensors/wrist_zedm/left_rgb).
_WRIST_SENSOR_ID = "wrist_zedm"
_WRIST_OBS_KEY = "left_rgb"

# HEAD-camera crop (raw pixel coords; rows [CROP_TOP:CROP_BOTTOM], cols
# [CROP_LEFT:CROP_RIGHT]) applied to the head RGB BEFORE resize. MUST match
# port_dexmate_hdf5.py — the model was trained on this cropped head view.
HEAD_CROP_TOP = 220
HEAD_CROP_BOTTOM = 600
HEAD_CROP_LEFT = 150
HEAD_CROP_RIGHT = 800


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


def _to_chw(img_hwc: np.ndarray) -> torch.Tensor:
    """HWC uint8 → CHW float32 in [0, 1] (mirrors LeRobotDataset.__getitem__)."""
    return (
        torch.from_numpy(np.ascontiguousarray(img_hwc)).permute(2, 0, 1).float()
        / 255.0
    )


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
    """Variant routing flags derived from policy.config (bimanual).

    Per-arm block is 10-D (eef: 9-D pose + gripper) or 8-D (joint: 7 joints +
    gripper); the full vector is LEFT block then RIGHT block.
    """

    state_dim: int
    action_dim: int
    state_is_eef: bool  # True if state_dim == 20 (10 per arm)
    action_is_eef: bool  # True if action_dim == 20 (10 per arm)
    arm_action_block: int  # 10 (eef) or 8 (joint) — per-arm slice width
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
        self.control_rate = self.config.get_rate("control_rate", 100)
        self.feedback_rate = self.config.get_rate("feedback_rate", 50)
        # Inner policy/camera tick (matches training fps).
        self.policy_fps = int(policy_fps)
        self._policy_period_ticks = max(1, round(self.control_rate / self.policy_fps))

        self.gripper_threshold = float(gripper_threshold)
        self.image_h = int(image_h)
        self.image_w = int(image_w)

        # Peek the policy config to learn which camera views it consumes BEFORE
        # building the robot (so we only enable/inject the wrist sensor when the
        # model actually needs it). The full policy is loaded in
        # initialize_policy(); image HxW is taken from the head feature shape.
        self._policy_path = policy_path
        self._detect_camera_inputs(policy_path)

        # Joint feedback publisher.
        joint_topic = self.config.get_topic("robot_joints")
        self.joint_pub = self.node.create_publisher(
            joint_topic, encoder=DictDataCodec.encode
        )

        # Per-arm workspace gates (in robot base frame). Left mirrors right
        # across the sagittal plane (see workspace_check.DEFAULT_LEFT_BOUNDS).
        if workspace_check:
            self._right_workspace_checker: Optional[WorkspaceChecker] = (
                WorkspaceChecker()
            )
            self._left_workspace_checker: Optional[WorkspaceChecker] = WorkspaceChecker(
                arm_link=DEFAULT_LEFT_LINK,
                arm_joint_names=tuple(f"L_arm_j{i}" for i in range(1, 8)),
                bounds=DEFAULT_LEFT_BOUNDS,
            )
        else:
            self._right_workspace_checker = None
            self._left_workspace_checker = None
            logger.warning("WorkspaceChecker disabled (both arms).")
        self._last_workspace_warn_t = {"left": 0.0, "right": 0.0}
        self._fixed_motion_joint_limits: dict[str, tuple[float, float]] = {}

        # IK-chain state (one per arm). Holds the last safe joints we sent
        # (= post-clamp IK output). Seeded from observed joints after homing and
        # on estop release. Mirrors the leader's implicit chain through
        # mm.{left,right}_arm. initialize_robot() seeds them; _init_motion_manager()
        # writes the ArmProcessors. Both run inside this __init__.
        self._chained_left_arm_qpos: Optional[np.ndarray] = None
        self._chained_right_arm_qpos: Optional[np.ndarray] = None
        self._left_proc: Optional[ArmProcessor] = None
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

        # Last-observation caches for the recorder (per arm + both cameras).
        self._last_obs_head_rgb: Optional[np.ndarray] = None
        self._last_obs_wrist_rgb: Optional[np.ndarray] = None
        self._last_obs_depth_u16: Optional[np.ndarray] = None
        self._last_obs_qpos_left: Optional[np.ndarray] = None
        self._last_obs_qpos_right: Optional[np.ndarray] = None
        self._last_obs_eef9_left: Optional[np.ndarray] = None
        self._last_obs_eef9_right: Optional[np.ndarray] = None
        self._last_obs_gripper_left = None  # np.float32; set in _build_observation
        self._last_obs_gripper_right = None

        # Wrist frame cache so a momentarily-missing publisher frame reuses the
        # last good one rather than crashing the policy step.
        self._last_wrist_full: Optional[np.ndarray] = None

        self._mode = _Mode.HOMING

    # ── initialisation ────────────────────────────────────────────────────────

    def _detect_camera_inputs(self, policy_path: str) -> None:
        """Read which observation.images.* views the policy expects.

        Sets ``self._use_head`` / ``self._use_wrist`` and overrides
        ``image_h``/``image_w`` from the head feature shape, so we enable only
        the cameras the model needs and resize to exactly its input size.
        """
        pcfg = PreTrainedConfig.from_pretrained(policy_path)
        in_feats = pcfg.input_features
        img_keys = [k for k in in_feats if k.startswith("observation.images.")]
        if not img_keys:
            raise ValueError(
                f"policy {policy_path} declares no observation.images.* inputs"
            )
        known = {"observation.images.head_rgb", "observation.images.wrist_rgb"}
        unknown = [k for k in img_keys if k not in known]
        if unknown:
            raise ValueError(
                f"unsupported image input feature(s): {unknown} "
                f"(only head_rgb / wrist_rgb are handled)"
            )
        self._use_head = "observation.images.head_rgb" in in_feats
        self._use_wrist = "observation.images.wrist_rgb" in in_feats
        if not self._use_head:
            raise ValueError(
                "policy must declare observation.images.head_rgb "
                f"(got image inputs: {img_keys})"
            )
        # Derive resize target from the head feature shape [C, H, W]; both views
        # share the same shape in our porter, so this also sizes the wrist view.
        shape = tuple(int(s) for s in in_feats["observation.images.head_rgb"].shape)
        if len(shape) != 3:
            raise ValueError(f"head_rgb feature shape must be [C, H, W], got {shape}")
        self.image_h, self.image_w = shape[1], shape[2]
        logger.info(
            f"Camera inputs: head={self._use_head} wrist={self._use_wrist} "
            f"resize=({self.image_h}x{self.image_w})"
        )

    def initialize_robot(self) -> None:
        logger.info("Initialising robot hardware ...")
        robot_configs = get_robot_config()
        robot_configs.sensors["head_camera"].enabled = True
        # Inject + enable the wrist ZED-M only when the policy uses it (mirrors
        # vr_reader): a multi-stream ZED config, RGB-only, published by a
        # standalone ZED-SDK process on sensors/wrist_zedm/{left,right}_rgb.
        if self._use_wrist:
            if _WRIST_SENSOR_ID not in robot_configs.sensors:
                robot_configs.sensors[_WRIST_SENSOR_ID] = ZedXCameraConfig(
                    name=_WRIST_SENSOR_ID,
                    enable_rgb=True,
                    enable_depth=False,
                )
            robot_configs.sensors[_WRIST_SENSOR_ID].enabled = True
        self.robot = Robot(configs=robot_configs)
        if not self.robot.has_sensor("head_camera"):
            raise RuntimeError(
                "head_camera was enabled in robot config but is not available"
            )
        if self.robot.sensors.head_camera.wait_for_active(timeout=5.0):
            logger.info("Head camera streaming started.")
        else:
            logger.warning(
                "head_camera did not report active within 5s; continuing because "
                "it may still become available before rollout starts."
            )
        if self._use_wrist:
            if not self.robot.has_sensor(_WRIST_SENSOR_ID):
                raise RuntimeError(
                    f"{_WRIST_SENSOR_ID} was enabled but is not available; the "
                    "policy needs observation.images.wrist_rgb"
                )
            if self.robot.sensors.wrist_zedm.wait_for_active(timeout=5.0):
                logger.info("Wrist camera streaming started.")
            else:
                logger.warning(
                    f"{_WRIST_SENSOR_ID} did not report active within 5s; the "
                    "rollout needs a ZED-SDK publisher on "
                    f"sensors/{_WRIST_SENSOR_ID}/*. Continuing — it may become "
                    "available before rollout starts."
                )
        self._set_home_position()

        # Activate + open both Robotiq Hand-E grippers. send_activate latches
        # rACT=1; build_hande_command alone is movement-only.
        send_activate(self.robot.left_arm)
        send_activate(self.robot.right_arm)
        self.robot.left_arm.send_ee_pass_through_message(build_hande_command(0.0))
        self.robot.right_arm.send_ee_pass_through_message(build_hande_command(0.0))
        self._last_gripper_cmd_left = 0.0
        self._last_gripper_cmd_right = 0.0

        self._last_cmd_left = np.array(self.robot.left_arm.get_joint_pos())
        self._last_cmd_right = np.array(self.robot.right_arm.get_joint_pos())
        # Initialise both IK chains at the observed home pose.
        self._chained_left_arm_qpos = self._last_cmd_left.astype(np.float32)
        self._chained_right_arm_qpos = self._last_cmd_right.astype(np.float32)

        # Robotiq Hand-E achieved-position sensors (one per arm). Training-time
        # observation.state[9]/[19] is the raw float `gPO/255` (the achieved
        # position the policy observed), NOT the binary command. Mirror the
        # leader: synchronous warmup poll to seed the cache, then
        # RobotiqStatusMonitor for async sampling. The monitor's subscriber
        # filters FC16 ACKs from FC03 status replies, so it coexists with our
        # outbound gripper writes on the same topic.
        self._last_obs_grip_left = self._warmup_gripper(self.robot.left_arm, "left")
        self._last_obs_grip_right = self._warmup_gripper(self.robot.right_arm, "right")
        self._grip_monitor_left = RobotiqStatusMonitor(
            self.robot.left_arm, side="left", function_code=0x03
        )
        self._grip_monitor_right = RobotiqStatusMonitor(
            self.robot.right_arm, side="right", function_code=0x03
        )
        self._grip_monitor_left.send_status_request()
        self._grip_monitor_right.send_status_request()
        logger.success(
            "Robot ready (home, grippers open, achieved "
            f"left={self._last_obs_grip_left:.3f} "
            f"right={self._last_obs_grip_right:.3f})."
        )

    @staticmethod
    def _warmup_gripper(arm, side: str) -> float:
        """Synchronous FC03 warmup poll to seed an arm's achieved-gripper cache."""
        warm = poll_gripper_status(arm, function_code=0x03, timeout_s=0.5)
        if warm is None:
            logger.warning(
                f"Gripper FC03 warmup failed ({side}) — falling back to 0.0 "
                "until the monitor catches up. Verify enable_ee_pass_through=True."
            )
            return 0.0
        return float(warm["actual"])

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

        # Auto-detect variant routing (bimanual: 20-D eef / 16-D joint).
        from lerobot.utils.constants import OBS_STATE, ACTION

        in_feats = self._policy.config.input_features
        out_feats = self._policy.config.output_features
        state_dim = int(in_feats[OBS_STATE].shape[0])
        action_dim = int(out_feats[ACTION].shape[0])
        if state_dim not in (16, 20):
            raise ValueError(
                f"unexpected bimanual state_dim={state_dim} (expected 16 joint / 20 eef)"
            )
        if action_dim not in (16, 20):
            raise ValueError(
                f"unexpected bimanual action_dim={action_dim} (expected 16 joint / 20 eef)"
            )
        rel_exclude_map = getattr(pcfg, "relative_exclude_dims", None) or {}
        rel_exclude_action = (
            list(rel_exclude_map.get("action", []))
            if isinstance(rel_exclude_map, dict)
            else None
        )
        self.variant = _Variant(
            state_dim=state_dim,
            action_dim=action_dim,
            state_is_eef=(state_dim == 20),
            action_is_eef=(action_dim == 20),
            arm_action_block=(10 if action_dim == 20 else 8),
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
            f"state_is_eef={self.variant.state_is_eef} "
            f"action_is_eef={self.variant.action_is_eef} "
            f"chunk/horizon={chunk} n_action_steps={n_act} n_obs_steps={n_obs} "
            f"relative={self.variant.uses_relative} "
            f"relative_exclude_action_dims={self.variant.relative_exclude_action_dims} "
            f"chunk_anchor_active={self._chunk_anchor_active}"
        )
        logger.info(
            f"Loop rates: control={self.control_rate}Hz, policy={self.policy_fps}Hz "
            f"(policy fires every {self._policy_period_ticks} control ticks)"
        )

        # MotionManager + IK + ArmProcessors: always initialised because every
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
            # Bimanual: solve/observe both end-effectors.
            target_frames=["L_ee", "R_ee"],
            initial_joint_configuration_dict=qpos_dict,
        )
        if self._motion_manager.pin_robot is None:
            raise ValueError("MotionManager failed to initialize")
        self._ik_solver = self._motion_manager.local_ik_solver
        if not isinstance(self._ik_solver, LocalPinkIKSolver):
            raise ValueError("LocalPinkIKSolver not available from MotionManager")
        self._cache_fixed_motion_joint_limits()
        self._motion_manager.set_joint_pos(self._motion_manager_state_dict())
        # Per-arm processors: provide limit_joint_step (10°/tick clamp) and
        # apply_positions, exactly as on the leader (vr_reader.py:475-476).
        self._left_proc = ArmProcessor(
            "left", self.config, self._motion_manager, self._robot_info, "vr"
        )
        self._right_proc = ArmProcessor(
            "right", self.config, self._motion_manager, self._robot_info, "vr"
        )
        logger.info("MotionManager + LocalPinkIKSolver + ArmProcessors initialised.")

    def _cache_fixed_motion_joint_limits(self) -> None:
        """Cache dexmotion limits only for fixed non-commanded model context.

        Both arms are commanded now, so only torso + head are fixed context.
        """
        assert self._motion_manager is not None
        model = self._motion_manager.pin_robot.model
        name_to_idx = getattr(self._motion_manager, "_joint_name_to_q_idx_dict", {})
        fixed_names = {
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
        self,
        left_arm: Optional[np.ndarray] = None,
        right_arm: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Both arms' state plus fixed torso/head joints for FK/IK.

        ``left_arm``/``right_arm`` default to live observed joints (used for FK
        on the policy's observation). Pass explicit arrays (e.g. the chained
        last-commanded joints) to seed MM before an IK call so
        ``limit_joint_step`` clamps against last-commanded, matching the
        leader's behaviour.
        """
        if left_arm is None:
            left_arm = self.robot.left_arm.get_joint_pos()
        if right_arm is None:
            right_arm = self.robot.right_arm.get_joint_pos()
        qd = {f"L_arm_j{i + 1}": float(q) for i, q in enumerate(left_arm)}
        qd.update({f"R_arm_j{i + 1}": float(q) for i, q in enumerate(right_arm)})
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
        """Interpolate torso/arms/head to INIT_*_JOINTS (arms via SAFE_* first)."""
        time.sleep(0.1)
        joint_delta = 0.01

        # Move torso directly to INIT_TORSO_JOINTS
        curr_torso = np.array(self.robot.torso.get_joint_pos())
        steps = max(
            1,
            int(np.max(np.abs(np.array(INIT_TORSO_JOINTS) - curr_torso)) / joint_delta),
        )
        for i in range(steps):
            interp = curr_torso + (np.array(INIT_TORSO_JOINTS) - curr_torso) * (
                i + 1
            ) / steps
            self.robot.torso.set_joint_pos(
                interp.tolist(), wait_time=0.1, exit_on_reach=True
            )

        # Left arm: move to SAFE_LEFT_ARM_JOINTS first, then INIT_LEFT_ARM_JOINTS
        for waypoints in (SAFE_LEFT_ARM_JOINTS, INIT_LEFT_ARM_JOINTS):
            curr = np.array(self.robot.left_arm.get_joint_pos())
            waypoint = np.array(waypoints)
            steps = max(1, int(np.max(np.abs(waypoint - curr)) / joint_delta))
            for i in range(steps):
                interp = curr + (waypoint - curr) * (i + 1) / steps
                self.robot.left_arm.set_joint_pos(
                    interp.tolist(), wait_time=0.1, exit_on_reach=True
                )

        # Right arm: move to SAFE_RIGHT_ARM_JOINTS first, then INIT_RIGHT_ARM_JOINTS
        for waypoints in (SAFE_RIGHT_ARM_JOINTS, INIT_RIGHT_ARM_JOINTS):
            curr = np.array(self.robot.right_arm.get_joint_pos())
            waypoint = np.array(waypoints)
            steps = max(1, int(np.max(np.abs(waypoint - curr)) / joint_delta))
            for i in range(steps):
                interp = curr + (waypoint - curr) * (i + 1) / steps
                self.robot.right_arm.set_joint_pos(
                    interp.tolist(), wait_time=0.1, exit_on_reach=True
                )

        # Move head directly to INIT_HEAD_JOINTS
        curr_head = np.array(self.robot.head.get_joint_pos())
        steps = max(
            1,
            int(np.max(np.abs(np.array(INIT_HEAD_JOINTS) - curr_head)) / joint_delta),
        )
        for i in range(steps):
            interp = curr_head + (np.array(INIT_HEAD_JOINTS) - curr_head) * (
                i + 1
            ) / steps
            self.robot.head.set_joint_pos(
                interp.tolist(), wait_time=0.1, exit_on_reach=True
            )
        logger.info("Robot at home position.")

    # ── observation ───────────────────────────────────────────────────────────

    @staticmethod
    def _pick_obs(obs: dict, key: str) -> np.ndarray:
        """Fetch ``key`` from a sensor obs dict, tolerating a sensor-name prefix.

        ``mdp_recorder.py`` prefixes head streams with ``head_``; the wrist ZED
        publisher uses bare keys. Try the bare key, then any ``*_<key>`` match.
        """
        if key in obs:
            return obs[key]
        for k in obs:
            if k.endswith("_" + key):
                return obs[k]
        raise KeyError(f"obs key {key!r} not found (available: {list(obs)})")

    def _read_head_camera(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read head camera streams for policy input + HDF5 recording.

        Returns ``(policy_head_hwc, full_left_rgb, depth_u16)``. The policy view
        is the head left eye cropped to the training region [220:600, 150:800]
        then resized to (image_w, image_h); the full-resolution left_rgb and
        depth (metres → uint16 millimetres) are kept for the recorder.
        """
        obs = self.robot.sensors.head_camera.get_obs(obs_keys=["left_rgb", "depth"])
        left_rgb = self._pick_obs(obs, "left_rgb")
        if left_rgb.dtype != np.uint8:
            left_rgb = np.clip(left_rgb, 0, 255).astype(np.uint8)

        h, w = left_rgb.shape[0], left_rgb.shape[1]
        if h < HEAD_CROP_BOTTOM or w < HEAD_CROP_RIGHT:
            raise ValueError(
                f"head frame {h}x{w} too small for training crop "
                f"rows [{HEAD_CROP_TOP}:{HEAD_CROP_BOTTOM}], "
                f"cols [{HEAD_CROP_LEFT}:{HEAD_CROP_RIGHT}]"
            )
        cropped = left_rgb[
            HEAD_CROP_TOP:HEAD_CROP_BOTTOM, HEAD_CROP_LEFT:HEAD_CROP_RIGHT
        ]
        # cv2.resize takes (W, H).
        policy_head = cv2.resize(
            cropped, (self.image_w, self.image_h), interpolation=cv2.INTER_AREA
        )

        depth_u16 = np.clip(self._pick_obs(obs, "depth") * 1000, 0, 65535).astype(
            np.uint16
        )
        return policy_head, left_rgb, depth_u16

    def _read_wrist_camera(self) -> tuple[np.ndarray, np.ndarray]:
        """Read the wrist ZED-M left eye for policy input + recording.

        Returns ``(policy_wrist_hwc, full_left_rgb)``. The wrist view is NOT
        cropped (matches the porter) — just resized to (image_w, image_h). A
        momentarily-missing publisher frame reuses the last good frame; a total
        absence raises so the operator fixes the wrist publisher before driving.
        """
        obs = self.robot.sensors.wrist_zedm.get_obs(obs_keys=[_WRIST_OBS_KEY])
        try:
            wrist = self._pick_obs(obs, _WRIST_OBS_KEY)
        except KeyError:
            wrist = None
        if wrist is None:
            if self._last_wrist_full is None:
                raise RuntimeError(
                    f"No wrist frame on sensors/{_WRIST_SENSOR_ID}/{_WRIST_OBS_KEY}; "
                    "the policy needs observation.images.wrist_rgb. Is the "
                    "ZED-SDK wrist publisher running?"
                )
            wrist = self._last_wrist_full
        if wrist.dtype != np.uint8:
            wrist = np.clip(wrist, 0, 255).astype(np.uint8)
        self._last_wrist_full = wrist
        policy_wrist = cv2.resize(
            wrist, (self.image_w, self.image_h), interpolation=cv2.INTER_AREA
        )
        return policy_wrist, wrist

    def _current_eef_poses(self) -> dict[str, np.ndarray]:
        """4x4 SE(3) of L_ee and R_ee in robot base frame, via MotionManager FK
        on the live observed joints."""
        assert self._motion_manager is not None
        self._motion_manager.set_joint_pos(self._motion_manager_state_dict())
        fk = self._motion_manager.fk(
            frame_names=self._motion_manager.target_frames,
            qpos=self._motion_manager.get_joint_pos(),
        )
        return {"L_ee": fk["L_ee"].np, "R_ee": fk["R_ee"].np}  # type: ignore[attr-defined]

    def _poll_gripper_status_step(self) -> None:
        """Drain queued FC03 replies and send the next status request (both arms).

        Mirrors ``vr_reader._poll_gripper_status_step``. Called once per policy
        tick so ``self._last_obs_grip_{left,right}`` refresh at ~policy_fps,
        matching the leader's 15 Hz cadence during recording.
        """
        for monitor, attr in (
            (self._grip_monitor_left, "_last_obs_grip_left"),
            (self._grip_monitor_right, "_last_obs_grip_right"),
        ):
            events = monitor.drain_status_events()
            if events:
                setattr(self, attr, float(events[-1]["actual"]))
            monitor.expire_timeouts(0.5)
            monitor.send_status_request()

    def _build_observation(self) -> dict:
        """Construct an unbatched dataset-style sample for ``self._pre()``.

        Builds the full bimanual ``observation.state`` (LEFT block then RIGHT,
        each ending in the achieved gripper) and the declared camera views.
        """
        # Refresh achieved grippers before reading state — training-time
        # observation.state[9]/[19] are the raw FC03 ``actual`` values.
        self._poll_gripper_status_step()
        grip_left = self._last_obs_grip_left
        grip_right = self._last_obs_grip_right

        left_arm = np.asarray(self.robot.left_arm.get_joint_pos(), dtype=np.float32)
        right_arm = np.asarray(self.robot.right_arm.get_joint_pos(), dtype=np.float32)

        # FK both EEFs (needed for eef state; also recorded for joint variants).
        poses = self._current_eef_poses()
        eef9_left = _mat_to_pos6d(poses["L_ee"])
        eef9_right = _mat_to_pos6d(poses["R_ee"])

        if self.variant.state_is_eef:
            state = np.concatenate(
                [eef9_left, [grip_left], eef9_right, [grip_right]]
            ).astype(np.float32)
        else:
            state = np.concatenate(
                [left_arm, [grip_left], right_arm, [grip_right]]
            ).astype(np.float32)

        sample: dict = {"observation.state": torch.from_numpy(state)}

        policy_head, head_full, depth_u16 = self._read_head_camera()
        if self._use_head:
            sample["observation.images.head_rgb"] = _to_chw(policy_head)
        if self._use_wrist:
            policy_wrist, wrist_full = self._read_wrist_camera()
            sample["observation.images.wrist_rgb"] = _to_chw(policy_wrist)
            self._last_obs_wrist_rgb = wrist_full

        # Cache for the recorder.
        self._last_obs_head_rgb = head_full
        self._last_obs_depth_u16 = depth_u16
        self._last_obs_qpos_left = left_arm.copy()
        self._last_obs_qpos_right = right_arm.copy()
        self._last_obs_eef9_left = eef9_left.astype(np.float32)
        self._last_obs_eef9_right = eef9_right.astype(np.float32)
        self._last_obs_gripper_left = np.float32(grip_left)
        self._last_obs_gripper_right = np.float32(grip_right)

        return sample

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

    # ── action decode + safety ─────────────────────────────────────────────────

    def _split_action_blocks(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (left_block, right_block) per the bimanual layout."""
        b = self.variant.arm_action_block
        return action[:b], action[b : 2 * b]

    @staticmethod
    def _block_to_se3(block: np.ndarray) -> np.ndarray:
        """EEF action block ``[pos(3), rot6d(6), grip]`` → 4x4 SE(3)."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _gram_schmidt_6d_to_R(block[3:9].astype(np.float64))
        T[:3, 3] = block[:3].astype(np.float64)
        return T

    def _action_to_joint_targets(
        self, action: np.ndarray
    ) -> tuple[Optional[dict[str, np.ndarray]], str, float, float]:
        """Convert bimanual policy output to per-arm joint targets + grippers.

        Both variants share the same safety pipeline: seed MM with the chained
        (last-commanded) state for both arms → produce raw 7-DOF targets (a
        single coupled Pink IK for EEF, the action blocks for joint) → per-arm
        ``limit_joint_step`` (10°/tick clamp) → ``_check_arm_joint_target``
        (hardware joint limits) → ``_check_safety`` (NaN + Cartesian workspace).
        The IK chains commit only after EVERY guard passes for BOTH arms, so a
        rejected tick leaves both chains anchored to the last safe pose.

        Returns ``(targets, reason, left_grip, right_grip)`` where ``targets``
        is ``{"left": joint7, "right": joint7}`` on success or ``None`` on any
        rejection (caller freezes/exits on this tick). Grippers are the raw
        (un-binarized) policy outputs and are returned in both cases.
        """
        assert self._motion_manager is not None
        assert self._left_proc is not None and self._right_proc is not None
        assert self._chained_left_arm_qpos is not None
        assert self._chained_right_arm_qpos is not None

        left_block, right_block = self._split_action_blocks(action)
        left_grip = float(left_block[-1])
        right_grip = float(right_block[-1])

        # Seed MM with both chained (last-commanded) arms so:
        #   - (EEF only) Pink IK warm-starts from there (matches leader),
        #   - limit_joint_step clamps each raw target against its chained arm.
        # mm.ik() does NOT commit its solution back into mm.{left,right}_arm, so
        # the MM state stays at the chained values across the call.
        self._motion_manager.set_joint_pos(
            self._motion_manager_state_dict(
                left_arm=self._chained_left_arm_qpos,
                right_arm=self._chained_right_arm_qpos,
            )
        )

        if self.variant.action_is_eef:
            # Coupled bimanual IK — both EEF targets in one solve so dexmotion
            # accounts for inter-arm self-collision (mirrors vr_reader).
            target_pose = {
                "L_ee": self._block_to_se3(left_block),
                "R_ee": self._block_to_se3(right_block),
            }
            try:
                arm_solution, in_collision, within_limits = self._motion_manager.ik(
                    target_pose=target_pose,
                    type="pink",
                )
            except Exception as e:  # solver internals can throw on degenerate inputs
                return None, f"ik_exception: {e}", left_grip, right_grip
            if not arm_solution:
                return None, "ik_no_solution", left_grip, right_grip
            if in_collision:
                return None, "ik_collision", left_grip, right_grip
            if not within_limits:
                return None, "ik_outside_limits", left_grip, right_grip
            try:
                raw_left = [arm_solution[f"L_arm_j{i}"] for i in range(1, 8)]
                raw_right = [arm_solution[f"R_arm_j{i}"] for i in range(1, 8)]
            except KeyError as e:
                return None, f"ik_missing_joint:{e}", left_grip, right_grip
        else:
            # Joint action — raw targets straight from the policy blocks.
            raw_left = left_block[:7].astype(np.float64).tolist()
            raw_right = right_block[:7].astype(np.float64).tolist()

        # 10°/tick clamp against each chained arm (clamps read MM arm state,
        # which was just seeded to the chained values and is unchanged by IK).
        safe_left = self._left_proc.limit_joint_step(raw_left)
        safe_right = self._right_proc.limit_joint_step(raw_right)
        target_left = safe_left.astype(np.float32)
        target_right = safe_right.astype(np.float32)

        # Validate BOTH post-clamp targets before committing either chain.
        for side, target in (("left", target_left), ("right", target_right)):
            ok, why = self._check_arm_joint_target(side, target)
            if not ok:
                return None, why, left_grip, right_grip
            ok, why = self._check_safety(side, target)
            if not ok:
                return None, why, left_grip, right_grip

        # Commit both chains (mm.{left,right}_arm advance → next tick's
        # limit_joint_step clamps against these; _chained_* mirror them so the
        # next IK call re-seeds MM from a known place).
        self._left_proc.apply_positions(safe_left.tolist())
        self._right_proc.apply_positions(safe_right.tolist())
        self._chained_left_arm_qpos = target_left
        self._chained_right_arm_qpos = target_right

        return (
            {"left": target_left, "right": target_right},
            "ok",
            left_grip,
            right_grip,
        )

    def _action_eef_record(
        self,
        side: str,
        action_block: np.ndarray,
        sent_joint: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (SE(3) 4x4, pos6d 9-D) for one arm's proposed eef.

        - action_is_eef: rebuild from the policy's action block.
        - else, when ``sent_joint`` is finite: FK on the proposed joint vector.
        - else: NaN-filled arrays (shape preserved so np.stack works).
        """
        if self.variant.action_is_eef:
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = _gram_schmidt_6d_to_R(
                action_block[3:9].astype(np.float64)
            ).astype(np.float32)
            T[:3, 3] = action_block[:3].astype(np.float32)
            pos6d = action_block[:9].astype(np.float32)
            return T, pos6d

        if (
            self._motion_manager is not None
            and sent_joint is not None
            and np.all(np.isfinite(sent_joint))
        ):
            prefix = "L" if side == "left" else "R"
            frame = "L_ee" if side == "left" else "R_ee"
            qd = self._motion_manager_state_dict()
            for i in range(7):
                qd[f"{prefix}_arm_j{i + 1}"] = float(sent_joint[i])
            self._motion_manager.set_joint_pos(qd)
            fk = self._motion_manager.fk(
                frame_names=self._motion_manager.target_frames,
                qpos=self._motion_manager.get_joint_pos(),
            )
            T = np.asarray(fk[frame].np, dtype=np.float32)  # type: ignore[attr-defined]
            return T, _mat_to_pos6d(T)

        return (
            np.full((4, 4), np.nan, dtype=np.float32),
            np.full((9,), np.nan, dtype=np.float32),
        )

    # ── safety ───────────────────────────────────────────────────────────────

    def _check_arm_joint_target(
        self, side: str, joint_target: np.ndarray
    ) -> tuple[bool, str]:
        """Validate one arm's 7-DOF command (shape, finiteness, hardware limits)."""
        if joint_target.shape != (7,):
            return False, f"{side}_arm_bad_shape:{joint_target.shape}"
        if not np.all(np.isfinite(joint_target)):
            return False, f"{side}_arm_non_finite"

        arm = self.robot.left_arm if side == "left" else self.robot.right_arm
        jl = getattr(arm, "_joint_limit", None)
        if jl is not None:
            lower, upper = jl[:, 0], jl[:, 1]
            if np.any(joint_target < lower) or np.any(joint_target > upper):
                return False, f"{side}_arm_outside_limits"
        return True, "ok"

    def _check_safety(
        self,
        side: str,
        command_target: np.ndarray,
    ) -> tuple[bool, str]:
        """Run all guards on one arm's command being sent."""
        # 1) NaN / inf
        if not np.all(np.isfinite(command_target)):
            return False, f"{side}_non_finite"

        # 2) Workspace check (this arm, in robot base frame). Head/torso are held
        # at their INIT poses for the duration of the rollout, so we use those
        # constants for FK rather than the current measurements.
        checker = (
            self._left_workspace_checker
            if side == "left"
            else self._right_workspace_checker
        )
        if checker is not None:
            in_bounds, eef_xyz = checker.is_in_workspace(
                arm_joints=command_target.tolist(),
                head=list(INIT_HEAD_JOINTS),
                torso=list(INIT_TORSO_JOINTS),
            )
            if not in_bounds:
                now = time.monotonic()
                if now - self._last_workspace_warn_t[side] > 0.5:
                    self._last_workspace_warn_t[side] = now
                    bounds = checker.bounds
                    logger.warning(
                        f"{side.capitalize()} EEF out of workspace: "
                        f"xyz={eef_xyz.round(3).tolist()} "
                        f"(bounds x{bounds['x']}, y{bounds['y']}, z{bounds['z']})"
                    )
                return False, f"{side}_workspace_out_of_bounds"

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
                "\n>>> Robot at home, grippers open. Press ENTER to start policy "
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
        joint_target_left = self._last_cmd_left.copy()
        joint_target_right = self._last_cmd_right.copy()
        gripper_target_left = self._last_gripper_cmd_left
        gripper_target_right = self._last_gripper_cmd_right
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
                    # Re-sync both IK chains to the observed arm poses — the
                    # operator may have moved an arm while paused. Re-seed
                    # mm.{left,right}_arm via apply_positions so the next
                    # limit_joint_step clamps against the live state.
                    if self._left_proc is not None and self._right_proc is not None:
                        obs_left = np.array(
                            self.robot.left_arm.get_joint_pos(), dtype=np.float32
                        )
                        obs_right = np.array(
                            self.robot.right_arm.get_joint_pos(), dtype=np.float32
                        )
                        self._chained_left_arm_qpos = obs_left
                        self._chained_right_arm_qpos = obs_right
                        self._left_proc.apply_positions(obs_left.tolist())
                        self._right_proc.apply_positions(obs_right.tolist())
                        # Align the held commands with the live pose so the
                        # first post-resume send doesn't snap either arm back to
                        # the pre-estop target before the next policy tick.
                        joint_target_left = obs_left.copy()
                        joint_target_right = obs_right.copy()
                    self._mode = _Mode.RUNNING

                # ─── inner: policy at policy_fps ──────────────────────────────
                step_info: Optional[dict] = None
                targets: Optional[dict[str, np.ndarray]] = None
                cand_left_grip: float = 0.0
                cand_right_grip: float = 0.0
                action: Optional[np.ndarray] = None
                if tick % self._policy_period_ticks == 0:
                    try:
                        action, step_info = self._policy_step()
                    except Exception as e:
                        logger.error(f"Policy step failed: {e}")
                        rate.sleep()
                        tick += 1
                        continue

                    targets, reason, cand_left_grip, cand_right_grip = (
                        self._action_to_joint_targets(action)
                    )
                    if targets is None:
                        consecutive_rejects += 1
                        logger.warning(f"Action rejected: {reason}")
                        if consecutive_rejects >= REJECT_LIMIT:
                            logger.error(
                                f"{consecutive_rejects} consecutive rejects — exiting."
                            )
                            break
                    else:
                        joint_target_left = targets["left"]
                        joint_target_right = targets["right"]
                        gripper_target_left = (
                            1.0 if cand_left_grip >= self.gripper_threshold else 0.0
                        )
                        gripper_target_right = (
                            1.0 if cand_right_grip >= self.gripper_threshold else 0.0
                        )
                        consecutive_rejects = 0

                # ─── outer: send (all safety guards ran in _action_to_joint_targets) ─
                self.robot.left_arm.set_joint_pos(joint_target_left.tolist())
                self.robot.right_arm.set_joint_pos(joint_target_right.tolist())
                publish_command_ns = time.time_ns()
                self.robot.left_arm.send_ee_pass_through_message(
                    build_hande_command(gripper_target_left)
                )
                self.robot.right_arm.send_ee_pass_through_message(
                    build_hande_command(gripper_target_right)
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
                self._last_cmd_left = joint_target_left
                self._last_cmd_right = joint_target_right
                self._last_gripper_cmd_left = gripper_target_left
                self._last_gripper_cmd_right = gripper_target_right

                # ─── record (per inner policy tick) ──────────────────────────
                if step_info is not None and action is not None:
                    if step_info["replan"]:
                        chunk_idx = 0
                    else:
                        chunk_idx = (chunk_idx + 1) % self._n_action_steps

                    if self._recorder is not None:
                        try:
                            self._record_frame(
                                action=action,
                                targets=targets,
                                cand_left_grip=cand_left_grip,
                                cand_right_grip=cand_right_grip,
                                step_info=step_info,
                                publish_command_ns=publish_command_ns,
                                chunk_idx=chunk_idx,
                            )
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

    def _record_frame(
        self,
        action: np.ndarray,
        targets: Optional[dict[str, np.ndarray]],
        cand_left_grip: float,
        cand_right_grip: float,
        step_info: dict,
        publish_command_ns: int,
        chunk_idx: int,
    ) -> None:
        """Assemble + push one bimanual debug frame to the recorder."""
        left_block, right_block = self._split_action_blocks(action)

        def proposed_joint(block: np.ndarray, committed: Optional[np.ndarray]):
            if not self.variant.action_is_eef:
                return block[:7].astype(np.float32)
            if committed is not None:
                return committed.astype(np.float32)
            return np.full(7, np.nan, dtype=np.float32)

        committed_left = targets["left"] if targets is not None else None
        committed_right = targets["right"] if targets is not None else None
        pj_left = proposed_joint(left_block, committed_left)
        pj_right = proposed_joint(right_block, committed_right)
        pg_left = np.float32(1.0 if cand_left_grip >= self.gripper_threshold else 0.0)
        pg_right = np.float32(1.0 if cand_right_grip >= self.gripper_threshold else 0.0)
        eef4_left, eef9_left = self._action_eef_record("left", left_block, pj_left)
        eef4_right, eef9_right = self._action_eef_record("right", right_block, pj_right)

        obs_head = self._last_obs_head_rgb
        obs_depth = self._last_obs_depth_u16
        obs_qpos_left = self._last_obs_qpos_left
        obs_qpos_right = self._last_obs_qpos_right
        obs_eef9_left = self._last_obs_eef9_left
        obs_eef9_right = self._last_obs_eef9_right
        obs_grip_left = self._last_obs_gripper_left
        obs_grip_right = self._last_obs_gripper_right
        assert (
            obs_head is not None
            and obs_depth is not None
            and obs_qpos_left is not None
            and obs_qpos_right is not None
            and obs_eef9_left is not None
            and obs_eef9_right is not None
            and obs_grip_left is not None
            and obs_grip_right is not None
        )

        # Image keys mirror the teleop layout written by leader/vr_reader.py
        # (obs/images/{head_left_rgb, head_depth, left_wrist_rgb}) so a single
        # viewer (scripts/vis_episode.py --deploy) reads both raw and rollout
        # episodes. These are the full-resolution head left eye / depth / wrist
        # left eye — the same streams those keys carry in teleop recordings.
        images = {"head_left_rgb": obs_head, "head_depth": obs_depth}
        if self._use_wrist:
            assert self._last_obs_wrist_rgb is not None
            images["left_wrist_rgb"] = self._last_obs_wrist_rgb

        self._recorder.record(  # type: ignore[union-attr]
            {
                "time": {
                    "timestamp_ns": np.int64(step_info["t_obs_ns_wallclock"]),
                    "begin_build_observation": np.int64(
                        step_info["begin_build_obs_ns"]
                    ),
                    "begin_inference": np.int64(step_info["begin_inference_ns"]),
                    "finish_inference": np.int64(step_info["finish_inference_ns"]),
                    "publish_command": np.int64(publish_command_ns),
                    "obs_flag": np.bool_(step_info["replan"]),
                    "action_number": np.int32(chunk_idx),
                },
                "obs": {
                    "joint": {
                        "left_arm": obs_qpos_left,
                        "right_arm": obs_qpos_right,
                    },
                    "eef_9d": {"left": obs_eef9_left, "right": obs_eef9_right},
                    "gripper": {"left": obs_grip_left, "right": obs_grip_right},
                    "images": images,
                },
                "action": {
                    "eef": {"left": eef4_left, "right": eef4_right},
                    "eef_9d": {"left": eef9_left, "right": eef9_right},
                    "joint": {"left_arm": pj_left, "right_arm": pj_right},
                    "gripper": {"left": pg_left, "right": pg_right},
                },
            }
        )

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
        for attr in ("_grip_monitor_left", "_grip_monitor_right"):
            monitor = getattr(self, attr, None)
            if monitor is not None:
                try:
                    monitor.close()
                except Exception as e:
                    logger.warning(f"{attr}.close() raised: {e}")
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
    """Run live bimanual policy rollout on a Dexmate robot.

    Args:
        policy_path: Path to a LeRobot ``pretrained_model`` directory
            (e.g. ``.../checkpoints/200000/pretrained_model``). Camera views
            (head_rgb / head_rgb+wrist_rgb) and state/action dims are read from
            its config.
        namespace: Zenoh namespace.
        debug: Verbose logging.
        config_name: Optional omniteleop config file basename
            (under ``omniteleop/configs/``).
        workspace_check: Gate each arm's commands on its configured Cartesian
            workspace bounds. Default on.
        policy_fps: Policy/camera tick rate. Match the dataset fps used in
            training (15 for our Dexmate variants).
        gripper_threshold: Binarise each gripper output at this value
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
