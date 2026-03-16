"""Follower input handler for VR teleoperation (Oculus Quest 3).

Subscribes to the ``vr/joints`` Zenoh topic published by ``vr_reader.py``
(which already performed IK on the leader side) and converts the received
``VRJointData`` into a ``RobotCommand`` with direct ``JOINT`` arm commands.

This mirrors the ``ExoJoyconHandler`` interface so the downstream pipelines
(CommandProcessor, SafetyValidator, ArmProcessor, HeadProcessor, HandProcessor)
are entirely unchanged.

Gripper control
---------------
``VRJointData.left_gripper`` and ``right_gripper`` are floats in [0, 1] where
0 = open and 1 = fully closed.  The handler interpolates between the open and
close poses defined in ``input_handlers.vr.hands.<side>.poses``.

Head control
------------
The handler emits head joint positions with ``CommandMode.ABSOLUTE`` so the
HeadProcessor applies them directly (not as deltas).  The Quest3 config must
set ``head.mode: "manual"`` in the VR handler section so HeadProcessor
dispatches into the manual branch where absolute-vs-relative is respected.
"""

from __future__ import annotations

import time
import threading
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from dexcomm import Node
from dexcomm.codecs import DictDataCodec

from omniteleop.common import get_config
from omniteleop.follower.input_handlers.base_handler import (
    ArmCommandType,
    BaseInputHandler,
    CommandMode,
    RobotCommand,
)

# Stale-packet threshold: if no message received within this window, activate
# emergency stop to prevent the robot moving on outdated commands.
_STALE_PACKET_S = 0.2


class VRHandler(BaseInputHandler):
    """Input handler for Quest 3 controller-based VR teleoperation.

    Parameters
    ----------
    config:    ``input_handlers.vr`` dict from the YAML config.
    namespace: Zenoh topic namespace prefix.
    """

    def __init__(self, config: Dict[str, Any], namespace: str = "") -> None:
        super().__init__(config, namespace)

        self.node = Node(name="vr_handler", namespace=namespace)
        self.config = config

        # ── Latest received data (guarded by lock) ────────────────────────────
        self._latest_vr_data: Optional[Dict[str, Any]] = None
        self._last_recv_t: float = 0.0
        self._vr_lock = threading.RLock()

        # ── Subscribers / publishers ──────────────────────────────────────────
        self._vr_sub = None
        self._recorder_pub = None

        # ── Gripper hand poses loaded from config ─────────────────────────────
        # Shape: (2, n_dof)  — row 0 = open, row 1 = close
        self._left_hand_poses = self._load_hand_poses("left")
        self._right_hand_poses = self._load_hand_poses("right")

        # ── Control state ──────────────────────────────────────────────────────
        self._motion_control_started = False
        self.exit_after_publish = False

        logger.info("VRHandler initialised.")

    # ── Helper: load open/close pose arrays from config ───────────────────────

    def _load_hand_poses(self, side: str) -> Optional[np.ndarray]:
        """Return [open_pose, close_pose] as a (2, n_dof) array, or None."""
        try:
            poses = (
                self.config.get("hands", {}).get(side, {}).get("poses", {})
            )
            open_pose = np.array(poses.get("open", []), dtype=float)
            close_pose = np.array(poses.get("close", []), dtype=float)
            if open_pose.size == 0 or close_pose.size != open_pose.size:
                return None
            return np.stack([open_pose, close_pose])  # (2, n_dof)
        except Exception as exc:
            logger.warning(f"VRHandler: could not load {side} hand poses: {exc}")
            return None

    # ── BaseInputHandler interface ────────────────────────────────────────────

    def initialize(self) -> bool:
        if self.initialized:
            return True
        self.setup_subscribers()
        self.initialized = True
        self.running = True
        logger.info("VRHandler ready.")
        return True

    def setup_subscribers(self) -> None:
        vr_topic = self.config.get("topics", {}).get("vr_joints", "vr/joints")
        self._vr_sub = self.node.create_subscriber(
            vr_topic,
            self._on_vr_data,
            decoder=DictDataCodec.decode,
        )

        recorder_topic = get_config().get_topic("recorder_control")
        self._recorder_pub = self.node.create_publisher(
            recorder_topic, encoder=DictDataCodec.encode
        )

        resolved = self.node.resolve_topic(vr_topic)
        logger.info(f"VRHandler subscribed to {resolved}")

    def _on_vr_data(self, data: Dict[str, Any]) -> None:
        with self._vr_lock:
            self._latest_vr_data = data
            self._last_recv_t = time.monotonic()

    def process_inputs(self) -> Optional[RobotCommand]:
        with self._vr_lock:
            data = self._latest_vr_data.copy() if self._latest_vr_data else None
            age = time.monotonic() - self._last_recv_t if self._last_recv_t else 1.0

        if data is None:
            return None

        command = RobotCommand(timestamp_ns=time.time_ns())

        # ── Stale-packet safety guard ─────────────────────────────────────────
        if age > _STALE_PACKET_S:
            command.safety_flags.emergency_stop = True
            logger.warning(f"VR data stale ({age * 1000:.0f} ms) — estop active.")
            return command

        # ── Map flags ─────────────────────────────────────────────────────────
        command.safety_flags.emergency_stop = bool(data.get("estop", True))
        if data.get("exit_requested", False):
            command.safety_flags.exit_requested = True
            self.exit_after_publish = True
            logger.critical("Exit requested via VR.")

        # ── Track motion-control activation ──────────────────────────────────
        if not command.safety_flags.emergency_stop and not self._motion_control_started:
            self._motion_control_started = True
            logger.info("VR motion control started — e-stop released.")

        # ── Arm joint commands (JOINT / ABSOLUTE) ─────────────────────────────
        left_arm_pos: List[float] = data.get("left_arm_pos", [])
        right_arm_pos: List[float] = data.get("right_arm_pos", [])

        if left_arm_pos and self._has_valid_joints(left_arm_pos):
            command.input_components["left_arm"] = {
                "command_type": ArmCommandType.JOINT,
                "mode": CommandMode.ABSOLUTE,
                "pos": left_arm_pos,
            }

        if right_arm_pos and self._has_valid_joints(right_arm_pos):
            command.input_components["right_arm"] = {
                "command_type": ArmCommandType.JOINT,
                "mode": CommandMode.ABSOLUTE,
                "pos": right_arm_pos,
            }

        # ── Head command (ABSOLUTE) ────────────────────────────────────────────
        head_pos: List[float] = data.get("head_pos", [])
        if head_pos and self._has_valid_joints(head_pos):
            command.input_components["head"] = {
                "mode": CommandMode.ABSOLUTE,
                "pos": head_pos,
            }

        # ── Gripper / hand commands ────────────────────────────────────────────
        self._add_hand_command(
            command, "left", float(data.get("left_gripper", 0.0)), self._left_hand_poses
        )
        self._add_hand_command(
            command, "right", float(data.get("right_gripper", 0.0)), self._right_hand_poses
        )

        self.update_command(command)
        return command

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _has_valid_joints(positions: List[float]) -> bool:
        """Return False if any joint value is NaN or Inf (discard bad IK results)."""
        arr = np.asarray(positions, dtype=float)
        return bool(np.all(np.isfinite(arr)))

    def _add_hand_command(
        self,
        command: RobotCommand,
        side: str,
        gripper_t: float,
        hand_poses: Optional[np.ndarray],
    ) -> None:
        """Interpolate hand pose from gripper trigger [0=open, 1=close] and emit."""
        if hand_poses is None:
            return
        # Clamp + interpolate  open_pose + t*(close_pose - open_pose)
        t = float(np.clip(gripper_t, 0.0, 1.0))
        pos = (hand_poses[0] + t * (hand_poses[1] - hand_poses[0])).tolist()
        command.input_components[f"{side}_hand"] = {
            "mode": CommandMode.ABSOLUTE,
            "pos": pos,
        }

    def cleanup(self) -> None:
        self.running = False
        self.node.shutdown()
        logger.info("VRHandler cleaned up.")

    def get_exit_requested(self) -> bool:
        return self.exit_after_publish

    def needs_motion_manager(self) -> bool:
        return False
