#!/usr/bin/env python3
"""Live policy rollout on a Dexmate robot (bimanual or single-arm).

Drives the policy's arm(s) + their Robotiq Hand-E grippers from a trained LeRobot
policy (ACT or Diffusion). The policy variant (joint vs eef state/action, absolute vs
relative, AND arm count) is auto-detected from ``policy.config`` so a single script
handles every ``dexmate_[<side>_]<state>_<action>`` × {abs, delta} combo.

A BIMANUAL policy (16-D joint / 20-D eef) drives both arms. A SINGLE-ARM policy
(8-D joint / 10-D eef) requires ``--arm-side left|right`` (the side is not stored in the
checkpoint); it drives only that arm and leaves the OTHER arm fully uncommanded (homed at
startup, then held by its controller — the coupled IK still receives the inactive arm's
observed pose so the solver holds it / collision-checks, but its solution is discarded).

Layout (matches ``port_dexmate_hdf5.py`` — bimanual is LEFT block then RIGHT; single-arm
is just the selected side's block):

  eef   (20-D): [L_pos(3), L_rot6d(3..8), L_grip@9,
                 R_pos(10..12), R_rot6d(13..18), R_grip@19]
  joint (16-D): [L_arm(7), L_grip@7, R_arm(7), R_grip@15]

Cameras (driven by ``policy.config.input_features``):

  observation.images.head_rgb  — head ZED-X Mini left eye at the model's HxW.
  observation.images.wrist_rgb — wrist ZED-M left eye at the model's HxW;
      requires a ZED-SDK publisher on ``sensors/wrist_zedm/*`` (optional;
      only read when the policy declares the feature).

Both publishers (tests/test_{head_zedx,wrist_zedm}_depth.py) crop+resize ROBOT-SIDE to
the model's HxW, so — exactly like the recorder leader/vr_reader.py — the frames are
consumed AS-IS (no workstation crop/resize) and the head is stamped with the shared
``omniteleop.common.head_camera.ZED_K`` intrinsic.

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

import numpy as np
import torch
import tyro
from dexbot_utils import RobotInfo
from dexbot_utils.configs.components.sensors.cameras import ZedXCameraConfig
from dexcomm import Node, RateLimiter
from dexcomm.codecs import DictDataCodec
from dexmotion.ik import LocalPinkIKSolver
from dexmotion.motion_manager import MotionManager

# LeRobot policy stack
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import RelativeActionsProcessorStep
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_POS_CONDITION_MASK
from loguru import logger

from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot
from omniteleop.common import get_config
from omniteleop.common.head_camera import ZED_K
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

# Wrist ZED-M sensor id + obs key (mirrors vr_reader._WRIST_SENSOR_ID and the
# left-eye obs key it consumes from sensors/wrist_zedm/left_rgb).
_WRIST_SENSOR_ID = "wrist_zedm"
_WRIST_OBS_KEY = "left_rgb"

# The head publisher (tests/test_head_zedx_depth.py) crops+resizes ROBOT-SIDE to the
# model's HxW before publishing, so — exactly like the recorder leader/vr_reader.py —
# the rollout consumes the frame AS-IS (no workstation crop/resize) and stamps the
# shared ZED_K (omniteleop.common.head_camera), the intrinsic of that cropped+resized
# view. Keep the publisher --crop/--resize in sync with head_camera.HEAD_CROP_TBLR /
# HEAD_RESIZE_HW (which define ZED_K).

# Canonical bimanual block order + per-side EEF frame / joint-name prefix. The rollout
# DRIVES only the policy's arm side(s) (``self._arm_sides``); for a single-arm policy the
# OTHER arm is left fully uncommanded (the coupled IK still receives its current observed
# pose so the solver holds it in place, but its solution is never sent).
_ARM_ORDER = ("left", "right")
_ARM_FRAME = {"left": "L_ee", "right": "R_ee"}
_ARM_PREFIX = {"left": "L", "right": "R"}


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
    return torch.from_numpy(np.ascontiguousarray(img_hwc)).permute(2, 0, 1).float() / 255.0


def _next_record_subdir(record_root: str | pathlib.Path) -> pathlib.Path:
    """Reserve the next numeric rollout subdirectory under ``record_root``."""
    root = pathlib.Path(record_root)
    root.mkdir(parents=True, exist_ok=True)

    existing_indices = (
        int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdecimal()
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
    """Variant routing flags derived from policy.config.

    Per-arm block is 10-D (eef: 9-D pose + gripper) or 8-D (joint: 7 joints +
    gripper). Bimanual is LEFT block then RIGHT block (16/20-D); single-arm is one
    block (8/10-D) for the side named by --arm-side (see controller._arm_sides).
    """

    state_dim: int
    action_dim: int
    state_is_eef: bool  # True if eef state (10-D single / 20-D bimanual)
    action_is_eef: bool  # True if eef action (10-D single / 20-D bimanual)
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
        positions_npz: Optional[str] = None,
        arm_side: Optional[str] = None,
    ) -> None:
        self.node = Node(name="policy_rollout", namespace=namespace)

        # Which arm a SINGLE-ARM policy drives. Required (and validated) only when the
        # policy turns out to be single-arm (8/10-D); ignored for bimanual policies.
        # Resolved into self._arm_sides in initialize_policy() once the dims are known.
        if arm_side is not None and arm_side not in _ARM_ORDER:
            raise ValueError(f"--arm-side must be 'left' or 'right', got {arm_side!r}")
        self._arm_side_flag = arm_side
        self._arm_sides: tuple[str, ...] = _ARM_ORDER

        self._robot_info = RobotInfo()
        self.has_torso = self._robot_info.has_torso
        self.has_chassis = self._robot_info.has_chassis

        from omniteleop import LIB_PATH

        config_path = LIB_PATH / "configs" / f"{config_name}.yaml" if config_name else None
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
        # Path to the precomputed changed-object positions npz (SceneDiff env-state
        # conditioning). Loaded in initialize_policy() only when the policy declares
        # observation.environment_state; ignored otherwise.
        self._positions_npz = positions_npz
        self._detect_camera_inputs(policy_path)

        # Resolve record_dir early — before initialize_robot() / _before_policy_init() —
        # so subclasses (e.g. live_scenediff_rollout) can read self._record_dir in their
        # hook to decide where to put SceneDiff outputs.
        # Mirror the model tree under deploy/ by replacing the model root prefix:
        #   .../Dexmate/model/dp/name/checkpoints/200000/pretrained_model
        #       → /home/yixuan/Dexmate/deploy/dp/name/checkpoints/200000/
        if record and record_dir is None:
            p = pathlib.Path(policy_path)
            _model_root = pathlib.Path("/home/yixuan/Dexmate/model")
            _deploy_root = pathlib.Path("/home/yixuan/Dexmate/deploy")
            try:
                record_dir = str(_deploy_root / p.parent.relative_to(_model_root))
            except ValueError:
                raise ValueError(
                    f"Cannot auto-derive record_dir: policy_path is not under "
                    f"{_model_root}. Pass --record-dir explicitly."
                ) from None
        self._record_dir: Optional[pathlib.Path] = (
            pathlib.Path(record_dir) if record_dir is not None else None
        )

        # Joint feedback publisher.
        joint_topic = self.config.get_topic("robot_joints")
        self.joint_pub = self.node.create_publisher(joint_topic, encoder=DictDataCodec.encode)

        # Per-arm workspace gates (in robot base frame). Left mirrors right
        # across the sagittal plane (see workspace_check.DEFAULT_LEFT_BOUNDS).
        if workspace_check:
            self._workspace_checker: dict[str, Optional[WorkspaceChecker]] = {
                "right": WorkspaceChecker(),
                "left": WorkspaceChecker(
                    arm_link=DEFAULT_LEFT_LINK,
                    arm_joint_names=tuple(f"L_arm_j{i}" for i in range(1, 8)),
                    bounds=DEFAULT_LEFT_BOUNDS,
                ),
            }
        else:
            self._workspace_checker = {"left": None, "right": None}
            logger.warning("WorkspaceChecker disabled (both arms).")
        self._last_workspace_warn_t = {"left": 0.0, "right": 0.0}
        self._fixed_motion_joint_limits: dict[str, tuple[float, float]] = {}

        # IK-chain state (one per arm). Holds the last safe joints we sent
        # (= post-clamp IK output). Seeded from observed joints after homing and
        # on estop release. Mirrors the leader's implicit chain through
        # mm.{left,right}_arm. initialize_robot() seeds them; _init_motion_manager()
        # writes the ArmProcessors. Both run inside this __init__.
        self._chained: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._proc: dict[str, Optional[ArmProcessor]] = {"left": None, "right": None}
        self._head_camera_intrinsic: Optional[np.ndarray] = None
        self._head_camera_extrinsic: Optional[np.ndarray] = None

        # Hardware + policy.
        self.initialize_robot()
        # Hook for subclasses to populate self._positions_npz AFTER the robot is homed
        # (head/torso at INIT, camera streaming) but BEFORE initialize_policy() loads
        # the policy and the env-state conditioning. Live SceneDiff conditioning
        # overrides this to capture a head frame at the exact rollout start viewpoint
        # and run detection while the GPU is still free of the policy. Default no-op,
        # so the standard rollout path is unchanged.
        self._before_policy_init()
        self.initialize_policy(
            policy_path,
            act_n_action_steps=act_n_action_steps,
            act_temporal_ensemble_coeff=act_temporal_ensemble_coeff,
        )

        if record:
            self._cache_head_camera_calibration()
            # self._record_dir was resolved above (before initialize_robot).
            rollout_record_dir = _next_record_subdir(self._record_dir)  # type: ignore[arg-type]
            self._recorder: Optional[EpisodeRecorder] = EpisodeRecorder(str(rollout_record_dir))
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
        # Observed EEF SE(3) per side (set each obs build); used to hold the inactive arm
        # in place during single-arm coupled IK.
        self._last_obs_eef_se3: Optional[dict[str, np.ndarray]] = None
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
            raise ValueError(f"policy {policy_path} declares no observation.images.* inputs")
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
                "policy must declare observation.images.head_rgb " f"(got image inputs: {img_keys})"
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
            raise RuntimeError("head_camera was enabled in robot config but is not available")
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
        self._chained["left"] = self._last_cmd_left.astype(np.float32)
        self._chained["right"] = self._last_cmd_right.astype(np.float32)

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

    def _before_policy_init(self) -> None:
        """Hook run after the robot is homed and before the policy/env-state load.

        Default no-op. Subclasses may override to capture a head frame and compute
        ``self._positions_npz`` (e.g. live SceneDiff conditioning) so that the
        subsequent ``initialize_policy`` -> ``_load_position_conditions`` picks it up
        through the normal path. Runs while the policy is not yet on the GPU.
        """

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
            get_policy_class(pcfg.type).from_pretrained(policy_path, config=pcfg).to(device).eval()
        )
        # ACT/Diffusion processors are saved with the checkpoint (no need to
        # supply dataset_stats — only Gr00t needs the override path).
        self._pre, self._post = make_pre_post_processors(
            policy_cfg=pcfg,
            pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        # Auto-detect variant routing: bimanual (16 joint / 20 eef = two arms) or
        # single-arm (8 joint / 10 eef = one arm). state/action share the per-arm block
        # size (8 joint, 10 eef) and the same arm COUNT.
        from lerobot.utils.constants import ACTION, OBS_STATE

        in_feats = self._policy.config.input_features
        out_feats = self._policy.config.output_features
        state_dim = int(in_feats[OBS_STATE].shape[0])
        action_dim = int(out_feats[ACTION].shape[0])
        _VALID_DIMS = (8, 10, 16, 20)
        if state_dim not in _VALID_DIMS:
            raise ValueError(
                f"unexpected state_dim={state_dim} (expected 8/10 single-arm, 16/20 bimanual)"
            )
        if action_dim not in _VALID_DIMS:
            raise ValueError(
                f"unexpected action_dim={action_dim} (expected 8/10 single-arm, 16/20 bimanual)"
            )
        n_arms_state = 2 if state_dim in (16, 20) else 1
        n_arms_action = 2 if action_dim in (16, 20) else 1
        if n_arms_state != n_arms_action:
            raise ValueError(
                f"state ({state_dim}-D) and action ({action_dim}-D) imply different arm "
                "counts; they must match (both single-arm or both bimanual)"
            )

        # Resolve which arm(s) to DRIVE. A single-arm policy REQUIRES --arm-side (the side
        # is not recoverable from the checkpoint); bimanual drives both and ignores it.
        if n_arms_state == 1:
            if self._arm_side_flag is None:
                raise ValueError(
                    f"single-arm policy (state_dim={state_dim}) requires --arm-side "
                    "left|right; the side is not stored in the checkpoint config"
                )
            self._arm_sides = (self._arm_side_flag,)
        else:
            self._arm_sides = _ARM_ORDER
            if self._arm_side_flag is not None:
                logger.warning(
                    f"--arm-side={self._arm_side_flag} ignored for a bimanual policy "
                    f"(state_dim={state_dim})."
                )

        rel_exclude_map = getattr(pcfg, "relative_exclude_dims", None) or {}
        rel_exclude_action = (
            list(rel_exclude_map.get("action", [])) if isinstance(rel_exclude_map, dict) else None
        )
        self.variant = _Variant(
            state_dim=state_dim,
            action_dim=action_dim,
            state_is_eef=(state_dim in (10, 20)),
            action_is_eef=(action_dim in (10, 20)),
            arm_action_block=(10 if action_dim in (10, 20) else 8),
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
        self._chunk_anchor_active = self._relative_step is not None and self._relative_step.enabled
        self._chunk_ref_state: Optional[torch.Tensor] = None

        # Print chunking like infer_dexmate.py for sanity.
        chunk = getattr(pcfg, "chunk_size", None) or getattr(pcfg, "horizon", None)
        n_act = getattr(pcfg, "n_action_steps", None)
        n_obs = getattr(pcfg, "n_obs_steps", None)
        self._n_action_steps = max(1, int(n_act)) if n_act is not None else 1
        logger.info(
            f"Policy: type={pcfg.type} state_dim={state_dim} action_dim={action_dim} "
            f"arm_sides={self._arm_sides} "
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

        # Position conditioning (SceneDiff env-state). Enabled only when the policy
        # declares observation.environment_state. We load the precomputed per-object
        # before/after positions here — BEFORE the control loop — and freeze (raise)
        # if they are missing/invalid, so the arms never move on bad conditioning.
        # The per-frame slot-selection mask is driven online by the model's auxiliary
        # stage head (see _policy_step).
        self._env_state_enabled = OBS_ENV_STATE in self._policy.config.input_features
        self._stage_num_classes = int(
            getattr(self._policy.config, "stage_prediction_num_classes", 2)
        )
        self._current_stage = 0
        self._env_state_vec: Optional[np.ndarray] = None
        self._active_position_condition: Optional[dict] = None
        if self._env_state_enabled:
            self._load_position_conditions()
            stage_head = getattr(self._policy.config, "stage_prediction_enabled", False)
            logger.info(
                "Position conditioning active: feeding "
                f"observation.environment_state ({self._env_state_vec.shape[0]}-D) + "
                "observation.pos_condition_mask."
            )
            if stage_head and (
                self._act_stage_prediction_supported()
                or callable(getattr(self._policy, "predict_stage", None))
            ):
                logger.info(
                    "Stage mask auto-switches via the model's stage head "
                    f"({self._stage_num_classes} classes)."
                )
            else:
                logger.warning(
                    "No usable stage head (stage_prediction_enabled=%s); "
                    "pos_condition_mask stays fixed on object 0.",
                    stage_head,
                )
        else:
            if self._positions_npz is not None:
                logger.warning(
                    "--positions_npz was given but the policy declares no "
                    "observation.environment_state; ignoring it."
                )
            logger.info("Position conditioning inactive (no env-state feature).")

        # Reset action queue at start.
        self._policy.reset()

    def _load_position_conditions(self) -> None:
        """Load precomputed changed-object positions into a constant env-state vector.

        Reads the ``episode_<N>.npz`` produced offline by
        ``scene_diff/scripts/extract_object_positions.py`` (driven by
        ``run_img_with_depth.sh``), validates it the same way the porter's
        ``load_episode_positions`` does — ``before_xyz``/``after_xyz`` each shape
        ``(M, 3)`` and all-finite, ``M`` == the policy's object count (stage classes) —
        and builds the flat slot-major ``(M*6,)`` vector ``[obj_i: before(3), after(3)]``.
        The saved diffusion preprocessor collapses that vector to the active object's
        6-D slot using ``observation.pos_condition_mask`` before the model.

        Raises (aborting before any arm motion) when the file is missing or malformed.
        That IS the quality gate: ``extract_object_positions.py`` only writes an npz for
        episodes that passed the object-count / valid-pixel checks, so a missing npz
        means the offline detection failed for this scene.
        """
        if self._positions_npz is None:
            raise ValueError(
                "policy declares observation.environment_state but --positions_npz was "
                "not provided; pass the episode_<N>.npz produced offline by "
                "scene_diff/scripts/extract_object_positions.py (run_img_with_depth.sh)"
            )
        path = pathlib.Path(self._positions_npz)
        if not path.exists():
            raise FileNotFoundError(
                f"--positions_npz not found: {path} (run run_img_with_depth.sh for this "
                "scene first; a missing file means the SceneDiff detection gate failed)"
            )
        m = self._stage_num_classes
        with np.load(path, allow_pickle=True) as data:
            missing = [k for k in ("before_xyz", "after_xyz") if k not in data.files]
            if missing:
                raise ValueError(f"{path}: missing keys {missing} (available: {list(data.files)})")
            before_xyz = np.asarray(data["before_xyz"], dtype=np.float32)
            after_xyz = np.asarray(data["after_xyz"], dtype=np.float32)

        for name, arr in (("before_xyz", before_xyz), ("after_xyz", after_xyz)):
            if arr.shape != (m, 3):
                raise ValueError(
                    f"{path}: {name} shape {tuple(arr.shape)} != ({m}, 3); M must match "
                    f"the policy's object count / stage classes ({m})"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{path}: {name} contains non-finite values")

        # Flat slot-major layout [obj_i: before(3), after(3)] — identical to the porter's
        # build_env_state_vector so the saved PositionConditionProcessorStep collapses it.
        self._env_state_vec = (
            np.concatenate([before_xyz, after_xyz], axis=1).reshape(-1).astype(np.float32)
        )
        logger.success(f"Loaded position conditioning ({m} object(s)) from {path}")
        for i in range(m):
            b, a = before_xyz[i], after_xyz[i]
            logger.info(
                f"  obj{i}: before=[{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}]  "
                f"after=[{a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}]"
            )

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
        self._proc["left"] = ArmProcessor(
            "left", self.config, self._motion_manager, self._robot_info, "vr"
        )
        self._proc["right"] = ArmProcessor(
            "right", self.config, self._motion_manager, self._robot_info, "vr"
        )
        logger.info("MotionManager + LocalPinkIKSolver + ArmProcessors initialised.")

    def _cache_head_camera_calibration(self) -> None:
        """Cache head-camera calibration once before rollout recording starts."""
        # ZED_K is the intrinsic of the publisher's cropped+resized frame — the SAME
        # value leader/vr_reader.py stamps and the reference scene stores.
        self._head_camera_intrinsic = ZED_K.copy()
        self._head_camera_extrinsic = self._compute_head_camera_extrinsic()
        xyz = self._head_camera_extrinsic[:3, 3]
        logger.info(
            "Cached rollout head-camera calibration: "
            f"xyz=[{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}]"
        )

    def _compute_head_camera_extrinsic(self) -> np.ndarray:
        """Compute ``world_T_head_camera`` from the fixed rollout home posture."""
        assert self._motion_manager is not None
        self._motion_manager.set_joint_pos(self._motion_manager_state_dict())
        fk = self._motion_manager.fk(
            frame_names=["zed_depth_frame"],
            qpos=self._motion_manager.get_joint_pos(),
        )
        return np.asarray(fk["zed_depth_frame"].np, dtype=np.float32)  # type: ignore[attr-defined]

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
                f"torso_j{i + 1}": self._fixed_motion_joint_value(f"torso_j{i + 1}", float(q))
                for i, q in enumerate(INIT_TORSO_JOINTS)
            }
        )
        qd.update(
            {
                f"head_j{i + 1}": self._fixed_motion_joint_value(f"head_j{i + 1}", float(q))
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
            interp = curr_torso + (np.array(INIT_TORSO_JOINTS) - curr_torso) * (i + 1) / steps
            self.robot.torso.set_joint_pos(interp.tolist(), wait_time=0.1, exit_on_reach=True)

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
            interp = curr_head + (np.array(INIT_HEAD_JOINTS) - curr_head) * (i + 1) / steps
            self.robot.head.set_joint_pos(interp.tolist(), wait_time=0.1, exit_on_reach=True)
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

    def _read_head_camera(self) -> tuple[np.ndarray, np.ndarray]:
        """Read the head camera streams for policy input + HDF5 recording.

        Consumes the publisher's frames AS-IS — identical to leader/vr_reader.py: the
        head publisher (tests/test_head_zedx_depth.py) already crops+resizes robot-side
        to the model's HxW, so there is NO workstation crop/resize. Returns
        ``(left_rgb uint8, depth_u16 mm)``; the matching intrinsic is the shared ZED_K.
        """
        obs = self.robot.sensors.head_camera.get_obs(obs_keys=["left_rgb", "depth"])
        left_rgb = self._pick_obs(obs, "left_rgb")
        if left_rgb.dtype != np.uint8:
            left_rgb = np.clip(left_rgb, 0, 255).astype(np.uint8)
        if left_rgb.shape[:2] != (self.image_h, self.image_w):
            raise ValueError(
                f"head frame {left_rgb.shape[:2]} != policy input "
                f"({self.image_h}, {self.image_w}); the publisher must crop+resize to the "
                "model HxW (tests/test_head_zedx_depth.py --crop/--resize, matching "
                "head_camera.HEAD_CROP_TBLR / HEAD_RESIZE_HW)"
            )
        # Depth metres → uint16 millimetres, as-is (matches vr_reader).
        depth_u16 = np.clip(self._pick_obs(obs, "depth") * 1000, 0, 65535).astype(np.uint16)
        return left_rgb, depth_u16

    def capture_head_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(left_rgb uint8 [H,W,3], depth_u16 [H,W] mm)`` from the live head
        camera at the current (homed) pose — the publisher's model-HxW frame, as-is.

        Used by live-conditioning subclasses (``_before_policy_init``) to build a
        SceneDiff "before" frame that matches the rollout start viewpoint AND the
        reference scene's shape (which make_deploy_before_hdf5.py validates).
        """
        return self._read_head_camera()

    def _read_wrist_camera(self) -> np.ndarray:
        """Read the wrist ZED-M left eye for policy input + recording.

        Consumes the publisher's frame AS-IS (no crop/resize), identical to vr_reader —
        the wrist publisher (tests/test_wrist_zedm_depth.py) already resized robot-side
        to the model's HxW. A momentarily-missing publisher frame reuses the last good
        frame; a total absence raises so the operator fixes the publisher before driving.
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
        if wrist.shape[:2] != (self.image_h, self.image_w):
            raise ValueError(
                f"wrist frame {wrist.shape[:2]} != policy input "
                f"({self.image_h}, {self.image_w}); the publisher must resize to the "
                "model HxW (tests/test_wrist_zedm_depth.py --resize)"
            )
        return wrist

    def _current_eef_poses(self) -> dict[str, np.ndarray]:
        """4x4 SE(3) of L_ee and R_ee in robot base frame, via MotionManager FK
        on the live observed joints.
        """
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

        Builds ``observation.state`` over the DRIVEN arm side(s) (bimanual: LEFT block
        then RIGHT; single-arm: just that side), each block ending in the achieved
        gripper, plus the declared camera views. Both arms' obs are still read/cached for
        the recorder, and both observed EEF poses are cached so a single-arm rollout can
        hold the inactive arm in place during the coupled IK.
        """
        # Refresh achieved grippers before reading state — training-time
        # observation.state grippers are the raw FC03 ``actual`` values.
        self._poll_gripper_status_step()
        grip_left = self._last_obs_grip_left
        grip_right = self._last_obs_grip_right

        left_arm = np.asarray(self.robot.left_arm.get_joint_pos(), dtype=np.float32)
        right_arm = np.asarray(self.robot.right_arm.get_joint_pos(), dtype=np.float32)

        # FK both EEFs (needed for eef state; also recorded for joint variants, and used
        # to hold the inactive arm in place during single-arm IK).
        poses = self._current_eef_poses()
        self._last_obs_eef_se3 = {"left": poses["L_ee"], "right": poses["R_ee"]}
        eef9_left = _mat_to_pos6d(poses["L_ee"])
        eef9_right = _mat_to_pos6d(poses["R_ee"])

        # Assemble the policy state from the DRIVEN arm side(s) only, in canonical order.
        _grip = {"left": grip_left, "right": grip_right}
        _arm = {"left": left_arm, "right": right_arm}
        _eef9 = {"left": eef9_left, "right": eef9_right}
        state = np.concatenate(
            [
                np.concatenate(
                    [_eef9[s] if self.variant.state_is_eef else _arm[s], [_grip[s]]]
                ).astype(np.float32)
                for s in self._arm_sides
            ]
        ).astype(np.float32)

        sample: dict = {"observation.state": torch.from_numpy(state)}

        policy_head, depth_u16 = self._read_head_camera()
        if self._use_head:
            sample["observation.images.head_rgb"] = _to_chw(policy_head)
        if self._use_wrist:
            policy_wrist = self._read_wrist_camera()
            sample["observation.images.wrist_rgb"] = _to_chw(policy_wrist)
            self._last_obs_wrist_rgb = policy_wrist

        # Cache the model-view frames for the recorder so deploy HDF5s match the
        # teleop recordings (model HxW + adjusted intrinsic), as vr_reader stores them.
        self._last_obs_head_rgb = policy_head
        self._last_obs_depth_u16 = depth_u16
        self._last_obs_qpos_left = left_arm.copy()
        self._last_obs_qpos_right = right_arm.copy()
        self._last_obs_eef9_left = eef9_left.astype(np.float32)
        self._last_obs_eef9_right = eef9_right.astype(np.float32)
        self._last_obs_gripper_left = np.float32(grip_left)
        self._last_obs_gripper_right = np.float32(grip_right)

        # Position conditioning: attach the constant per-episode env-state vector plus the
        # current one-hot stage mask. The saved preprocessor
        # (PositionConditionProcessorStep) collapses the (M*6,) vector to the active
        # object's 6-D [before, after] slot using this mask, then normalizes it.
        if self._env_state_enabled and self._env_state_vec is not None:
            sample[OBS_ENV_STATE] = torch.from_numpy(self._env_state_vec)
            self._write_stage_mask(sample)

        return sample

    def _write_stage_mask(self, sample: dict) -> None:
        """Write the current one-hot stage mask into an unprocessed observation."""
        mask = np.zeros((self._stage_num_classes,), dtype=np.float32)
        mask[self._current_stage] = 1.0
        sample[OBS_POS_CONDITION_MASK] = torch.from_numpy(mask)

    def _set_current_stage(self, stage: int) -> None:
        """Update the active position-condition stage after validation."""
        new_stage = int(stage)
        if not 0 <= new_stage < self._stage_num_classes:
            raise ValueError(f"predicted stage {new_stage} outside [0, {self._stage_num_classes})")
        if new_stage != self._current_stage:
            logger.info(f"Position-condition stage {self._current_stage} -> {new_stage}")
            self._current_stage = new_stage

    def _policy_is_act(self) -> bool:
        """Return whether the loaded policy is an ACT policy."""
        cfg_type = getattr(getattr(self._policy, "config", None), "type", None)
        return getattr(self._policy, "name", None) == "act" or cfg_type == "act"

    def _act_stage_prediction_supported(self) -> bool:
        """Return whether ACT stage logits can be read from the policy model."""
        if not self._policy_is_act():
            return False
        if not getattr(self._policy.config, "stage_prediction_enabled", False):
            return False
        return callable(getattr(getattr(self, "_policy", None), "model", None))

    def _predict_act_stage_from_batch(self, batch: dict) -> Optional[int]:
        """Predict current ACT stage from a preprocessed current observation.

        ACT trains ``stage_logits[:, i]`` against the mask at action offset ``i``.
        For selecting the env-state slot used by this replan's action chunk, the
        correct timestamp is offset 0: the current observation / first action.
        """
        if not self._act_stage_prediction_supported():
            return None

        model_batch = dict(batch)
        image_features = list(getattr(self._policy.config, "image_features", []) or [])
        if image_features:
            model_batch[OBS_IMAGES] = [model_batch[key] for key in image_features]

        eval_fn = getattr(self._policy, "eval", None)
        if callable(eval_fn):
            eval_fn()
        with torch.no_grad():
            output = self._policy.model(model_batch)

        if not isinstance(output, tuple) or len(output) < 3:
            return None
        stage_logits = output[2]
        if stage_logits is None:
            return None
        if stage_logits.ndim != 3:
            raise ValueError(
                "ACT stage logits must have shape (B, n_action_steps, classes), "
                f"got {tuple(stage_logits.shape)}"
            )
        if stage_logits.shape[-1] != self._stage_num_classes:
            raise ValueError(
                "ACT stage logits class dimension must match "
                f"stage_prediction_num_classes={self._stage_num_classes}, "
                f"got {tuple(stage_logits.shape)}"
            )
        return int(stage_logits[:, 0].argmax(dim=-1).reshape(-1)[0].item())

    def _position_condition_record(self, raw_sample: dict) -> Optional[dict]:
        """Return the position-condition stage/mask/6-D slot used by this sample."""
        if (
            not self._env_state_enabled
            or self._env_state_vec is None
            or OBS_POS_CONDITION_MASK not in raw_sample
        ):
            return None

        mask_value = raw_sample[OBS_POS_CONDITION_MASK]
        if isinstance(mask_value, torch.Tensor):
            mask = mask_value.detach().cpu().numpy()
        else:
            mask = np.asarray(mask_value)
        mask = np.asarray(mask, dtype=np.float32).reshape(-1)
        if mask.shape != (self._stage_num_classes,):
            raise ValueError(
                f"position-condition mask shape {tuple(mask.shape)} != "
                f"({self._stage_num_classes},)"
            )

        stage = int(mask.argmax())
        env_state = self._env_state_vec.reshape(self._stage_num_classes, 6)
        return {
            "stage": np.int32(stage),
            "mask": mask.copy(),
            "condition_6d": env_state[stage].astype(np.float32, copy=True),
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
        raw_sample = self._build_observation()
        t_obs_end = time.monotonic()

        sample = self._pre(dict(raw_sample))  # writes _last_state = current obs state

        # At replan time, ACT's auxiliary head predicts the stage for the current
        # observation at logits offset 0. Use that stage for THIS action chunk's
        # env-state slot before select_action() runs. The head does not consume
        # observation.environment_state, so the initial mask used for this probe
        # cannot leak the answer into the prediction.
        if self._env_state_enabled and qlen_before <= 0:
            stage = self._predict_act_stage_from_batch(sample)
            if stage is not None:
                self._set_current_stage(stage)
                self._write_stage_mask(raw_sample)
                sample = self._pre(dict(raw_sample))

        sample_position_condition = self._position_condition_record(raw_sample)
        if qlen_before <= 0:
            self._active_position_condition = sample_position_condition
        position_condition = getattr(self, "_active_position_condition", sample_position_condition)
        if position_condition is None:
            position_condition = sample_position_condition

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
            self._chunk_ref_state = cached.detach().clone() if cached is not None else None

        # Fallback for policy classes whose stage API depends on internal queues
        # populated by select_action() (e.g. DiffusionPolicy). ACT is handled above
        # so its current replan chunk is conditioned at the correct timestamp.
        predict_stage = getattr(self._policy, "predict_stage", None)
        if (
            self._env_state_enabled
            and callable(predict_stage)
            and not self._act_stage_prediction_supported()
            and qlen_before <= 0
        ):
            stage = predict_stage()
            if stage is not None:
                self._set_current_stage(int(stage.reshape(-1)[0].item()))

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
        if position_condition is not None:
            info["position_condition"] = position_condition
        return a_abs.squeeze(0).detach().cpu().numpy(), info

    # ── action decode + safety ─────────────────────────────────────────────────

    def _robot_arm(self, side: str):
        """The hardware arm handle for ``side`` ("left"/"right")."""
        return self.robot.left_arm if side == "left" else self.robot.right_arm

    def _split_action_blocks(self, action: np.ndarray) -> dict[str, np.ndarray]:
        """Split the policy action into the DRIVEN arm blocks, keyed by side.

        Bimanual → ``{"left": action[:b], "right": action[b:2b]}``; single-arm → one
        block ``{side: action[:b]}`` (``b`` = ``arm_action_block``).
        """
        b = self.variant.arm_action_block
        return {side: action[i * b : (i + 1) * b] for i, side in enumerate(self._arm_sides)}

    @staticmethod
    def _block_to_se3(block: np.ndarray) -> np.ndarray:
        """EEF action block ``[pos(3), rot6d(6), grip]`` → 4x4 SE(3)."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _gram_schmidt_6d_to_R(block[3:9].astype(np.float64))
        T[:3, 3] = block[:3].astype(np.float64)
        return T

    def _action_to_joint_targets(
        self, action: np.ndarray
    ) -> tuple[Optional[dict[str, np.ndarray]], str, dict[str, float]]:
        """Convert the policy output to per-arm joint targets + grippers for the DRIVEN
        arm side(s).

        Shared safety pipeline: seed MM with the chained (last-commanded) driven arms (and
        any inactive arm at its observed pose) → produce raw 7-DOF targets (a coupled Pink
        IK for EEF, the action blocks for joint) → per-arm ``limit_joint_step`` (10°/tick
        clamp) → ``_check_arm_joint_target`` (hardware joint limits) → ``_check_safety``
        (NaN + Cartesian workspace). The IK chains commit only after EVERY guard passes for
        ALL driven arms, so a rejected tick leaves them anchored to the last safe pose.

        For a single-arm policy the OTHER arm is never a free IK target: the coupled solve
        receives its current OBSERVED pose (so dexmotion holds it in place / collision-
        checks correctly) and its solution is discarded — it is left fully uncommanded.

        Returns ``(targets, reason, grips)`` where ``targets`` maps each driven side to its
        7-DOF target on success (``None`` on any rejection) and ``grips`` maps each driven
        side to its raw (un-binarized) policy gripper output.
        """
        assert self._motion_manager is not None
        for side in self._arm_sides:
            assert self._proc[side] is not None
            assert self._chained[side] is not None

        blocks = self._split_action_blocks(action)  # {side: block} over driven sides
        grips = {side: float(blocks[side][-1]) for side in self._arm_sides}

        # Seed MM: driven arms at their chained (last-commanded) pose so limit_joint_step
        # clamps against it and (EEF) Pink IK warm-starts there; any inactive arm at its
        # live observed pose. mm.ik() does NOT commit back into mm.{left,right}_arm.
        seed: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        for side in self._arm_sides:
            seed[side] = self._chained[side]
        self._motion_manager.set_joint_pos(
            self._motion_manager_state_dict(left_arm=seed["left"], right_arm=seed["right"])
        )

        raw: dict[str, list] = {}
        if self.variant.action_is_eef:
            # Coupled IK over BOTH frames so dexmotion accounts for inter-arm collision:
            # each driven arm targets the policy pose; an inactive arm targets its current
            # observed pose (held in place). Only driven-arm joints are extracted/sent.
            target_pose = {
                _ARM_FRAME[side]: self._block_to_se3(blocks[side]) for side in self._arm_sides
            }
            for side in _ARM_ORDER:
                if side not in self._arm_sides:
                    if self._last_obs_eef_se3 is None:
                        return None, "no_observed_eef_for_inactive_arm", grips
                    target_pose[_ARM_FRAME[side]] = self._last_obs_eef_se3[side].astype(np.float64)
            try:
                arm_solution, in_collision, within_limits = self._motion_manager.ik(
                    target_pose=target_pose,
                    type="pink",
                )
            except Exception as e:  # solver internals can throw on degenerate inputs
                return None, f"ik_exception: {e}", grips
            if not arm_solution:
                return None, "ik_no_solution", grips
            if in_collision:
                return None, "ik_collision", grips
            if not within_limits:
                return None, "ik_outside_limits", grips
            try:
                for side in self._arm_sides:
                    p = _ARM_PREFIX[side]
                    raw[side] = [arm_solution[f"{p}_arm_j{i}"] for i in range(1, 8)]
            except KeyError as e:
                return None, f"ik_missing_joint:{e}", grips
        else:
            # Joint action — raw targets straight from each driven block.
            for side in self._arm_sides:
                raw[side] = blocks[side][:7].astype(np.float64).tolist()

        # 10°/tick clamp against each chained driven arm (clamps read MM arm state,
        # which was just seeded to the chained values and is unchanged by IK).
        safe = {side: self._proc[side].limit_joint_step(raw[side]) for side in self._arm_sides}
        targets = {side: safe[side].astype(np.float32) for side in self._arm_sides}

        # Validate EVERY driven arm's post-clamp target before committing any chain.
        for side in self._arm_sides:
            ok, why = self._check_arm_joint_target(side, targets[side])
            if not ok:
                return None, why, grips
            ok, why = self._check_safety(side, targets[side])
            if not ok:
                return None, why, grips

        # Commit driven chains (mm.{side}_arm advance → next tick's limit_joint_step
        # clamps against these; _chained mirrors them so the next IK re-seeds MM cleanly).
        for side in self._arm_sides:
            self._proc[side].apply_positions(safe[side].tolist())
            self._chained[side] = targets[side]

        return targets, "ok", grips

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
            T[:3, :3] = _gram_schmidt_6d_to_R(action_block[3:9].astype(np.float64)).astype(
                np.float32
            )
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

    def _check_arm_joint_target(self, side: str, joint_target: np.ndarray) -> tuple[bool, str]:
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
        checker = self._workspace_checker[side]
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
                self.joint_pub.publish({"timestamp_ns": time.time_ns(), "joints": positions})
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
        self._current_stage = 0  # every rollout starts on object 0 (matches training)
        self._active_position_condition = None
        rate = RateLimiter(self.control_rate)
        logger.info(f"Control loop at {self.control_rate} Hz")

        # Targets that persist between policy ticks (held flat), one entry per DRIVEN arm.
        _last_cmd = {"left": self._last_cmd_left, "right": self._last_cmd_right}
        _last_gcmd = {
            "left": self._last_gripper_cmd_left,
            "right": self._last_gripper_cmd_right,
        }
        joint_targets: dict[str, np.ndarray] = {s: _last_cmd[s].copy() for s in self._arm_sides}
        gripper_targets: dict[str, float] = {s: _last_gcmd[s] for s in self._arm_sides}
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
                    self._active_position_condition = None
                    # Re-sync each DRIVEN IK chain to the observed arm pose — the
                    # operator may have moved an arm while paused. Re-seed
                    # mm.{side}_arm via apply_positions so the next limit_joint_step
                    # clamps against the live state. (An inactive arm is uncommanded,
                    # so it needs no resync.)
                    if all(self._proc[s] is not None for s in self._arm_sides):
                        for s in self._arm_sides:
                            obs = np.array(self._robot_arm(s).get_joint_pos(), dtype=np.float32)
                            self._chained[s] = obs
                            self._proc[s].apply_positions(obs.tolist())
                            # Align the held command with the live pose so the first
                            # post-resume send doesn't snap the arm back to the
                            # pre-estop target before the next policy tick.
                            joint_targets[s] = obs.copy()
                    self._mode = _Mode.RUNNING

                # ─── inner: policy at policy_fps ──────────────────────────────
                step_info: Optional[dict] = None
                targets: Optional[dict[str, np.ndarray]] = None
                cand_grips: dict[str, float] = {}
                action: Optional[np.ndarray] = None
                if tick % self._policy_period_ticks == 0:
                    try:
                        action, step_info = self._policy_step()
                    except Exception as e:
                        logger.error(f"Policy step failed: {e}")
                        rate.sleep()
                        tick += 1
                        continue

                    targets, reason, cand_grips = self._action_to_joint_targets(action)
                    if targets is None:
                        consecutive_rejects += 1
                        logger.warning(f"Action rejected: {reason}")
                        if consecutive_rejects >= REJECT_LIMIT:
                            logger.error(f"{consecutive_rejects} consecutive rejects — exiting.")
                            break
                    else:
                        for s in self._arm_sides:
                            joint_targets[s] = targets[s]
                            gripper_targets[s] = (
                                1.0 if cand_grips[s] >= self.gripper_threshold else 0.0
                            )
                        consecutive_rejects = 0

                # ─── outer: send DRIVEN arms only (an inactive arm stays uncommanded;
                #     all safety guards ran in _action_to_joint_targets) ─
                for s in self._arm_sides:
                    self._robot_arm(s).set_joint_pos(joint_targets[s].tolist())
                publish_command_ns = time.time_ns()
                for s in self._arm_sides:
                    self._robot_arm(s).send_ee_pass_through_message(
                        build_hande_command(gripper_targets[s])
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
                self._last_cmd_left = joint_targets.get("left", self._last_cmd_left)
                self._last_cmd_right = joint_targets.get("right", self._last_cmd_right)
                self._last_gripper_cmd_left = gripper_targets.get(
                    "left", self._last_gripper_cmd_left
                )
                self._last_gripper_cmd_right = gripper_targets.get(
                    "right", self._last_gripper_cmd_right
                )

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
                                cand_grips=cand_grips,
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
        cand_grips: dict[str, float],
        step_info: dict,
        publish_command_ns: int,
        chunk_idx: int,
    ) -> None:
        """Assemble + push one debug frame to the recorder.

        Each DRIVEN arm records the policy's proposed action; an inactive (uncommanded)
        arm records it HOLDING its last observed pose, so the bimanual recorder layout
        (both left + right blocks) stays intact for single-arm rollouts too.
        """
        blocks = self._split_action_blocks(action)  # {side: block} over driven sides
        obs_qpos = {"left": self._last_obs_qpos_left, "right": self._last_obs_qpos_right}
        obs_eef9 = {"left": self._last_obs_eef9_left, "right": self._last_obs_eef9_right}
        held_grip_cmd = {
            "left": self._last_gripper_cmd_left,
            "right": self._last_gripper_cmd_right,
        }

        eef4: dict[str, np.ndarray] = {}
        eef9: dict[str, np.ndarray] = {}
        pj: dict[str, np.ndarray] = {}
        pg: dict[str, np.float32] = {}
        for side in _ARM_ORDER:
            if side in self._arm_sides:
                block = blocks[side]
                committed = targets[side] if targets is not None else None
                if not self.variant.action_is_eef:
                    pj[side] = block[:7].astype(np.float32)
                elif committed is not None:
                    pj[side] = committed.astype(np.float32)
                else:
                    pj[side] = np.full(7, np.nan, dtype=np.float32)
                pg[side] = np.float32(
                    1.0 if cand_grips.get(side, 0.0) >= self.gripper_threshold else 0.0
                )
                eef4[side], eef9[side] = self._action_eef_record(side, block, pj[side])
            else:
                # Inactive arm: uncommanded → record it holding its last observed pose.
                assert obs_qpos[side] is not None and obs_eef9[side] is not None
                pj[side] = obs_qpos[side].astype(np.float32)
                pg[side] = np.float32(held_grip_cmd[side])
                eef9[side] = obs_eef9[side].astype(np.float32)
                eef4[side] = (
                    self._last_obs_eef_se3[side].astype(np.float32)
                    if self._last_obs_eef_se3 is not None
                    else np.full((4, 4), np.nan, dtype=np.float32)
                )

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
        # episodes. These are the model-view head left eye / depth / wrist left eye
        # (the same cropped+resized streams + intrinsic those keys carry in teleop
        # recordings, since the publisher crops+resizes robot-side).
        images = {"head_left_rgb": obs_head, "head_depth": obs_depth}
        if self._use_wrist:
            assert self._last_obs_wrist_rgb is not None
            images["left_wrist_rgb"] = self._last_obs_wrist_rgb
        if self._head_camera_intrinsic is not None and self._head_camera_extrinsic is not None:
            images["intrinsic"] = self._head_camera_intrinsic
            images["extrinsic"] = self._head_camera_extrinsic

        obs = {
            "joint": {
                "left_arm": obs_qpos_left,
                "right_arm": obs_qpos_right,
            },
            "eef_9d": {"left": obs_eef9_left, "right": obs_eef9_right},
            "gripper": {"left": obs_grip_left, "right": obs_grip_right},
            "images": images,
        }
        position_condition = step_info.get("position_condition")
        if position_condition is not None:
            obs["position_condition"] = {
                "stage": np.int32(position_condition["stage"]),
                "mask": np.asarray(position_condition["mask"], dtype=np.float32),
                "condition_6d": np.asarray(position_condition["condition_6d"], dtype=np.float32),
            }

        self._recorder.record(  # type: ignore[union-attr]
            {
                "time": {
                    "timestamp_ns": np.int64(step_info["t_obs_ns_wallclock"]),
                    "begin_build_observation": np.int64(step_info["begin_build_obs_ns"]),
                    "begin_inference": np.int64(step_info["begin_inference_ns"]),
                    "finish_inference": np.int64(step_info["finish_inference_ns"]),
                    "publish_command": np.int64(publish_command_ns),
                    "obs_flag": np.bool_(step_info["replan"]),
                    "action_number": np.int32(chunk_idx),
                },
                "obs": obs,
                "action": {
                    "eef": {"left": eef4["left"], "right": eef4["right"]},
                    "eef_9d": {"left": eef9["left"], "right": eef9["right"]},
                    "joint": {"left_arm": pj["left"], "right_arm": pj["right"]},
                    "gripper": {"left": pg["left"], "right": pg["right"]},
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
    positions_npz: Optional[str] = None,
    arm_side: Optional[str] = None,
) -> None:
    """Run a live policy rollout on a Dexmate robot (bimanual or single-arm).

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
            ``/home/yixuan/Dexmate/deploy/<model-tag>/<step>`` derived from
            ``policy_path`` (e.g. ``.../dexmate_eef_eef_abs_film/200000/``).
        positions_npz: Path to the precomputed changed-object positions
            ``episode_<N>.npz`` (from scene_diff/scripts/extract_object_positions.py,
            driven by run_img_with_depth.sh). REQUIRED when the policy declares
            ``observation.environment_state``: loaded before the control loop and fed as
            env-state conditioning. Ignored for non-position-conditioned policies.
        arm_side: REQUIRED for a single-arm policy (8-D joint / 10-D eef) — "left" or
            "right", the arm that policy drives. The side is not stored in the checkpoint,
            so it must be given explicitly. The OTHER arm is left fully uncommanded (homed
            at startup, then held by its controller). Ignored for bimanual policies.
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
        positions_npz=positions_npz,
        arm_side=arm_side,
    )
    ctrl.run()


if __name__ == "__main__":
    sys.exit(tyro.cli(main))
