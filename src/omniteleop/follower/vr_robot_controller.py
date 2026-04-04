#!/usr/bin/env python3
"""Follower robot controller for VR teleoperation.

Subscribes directly to the VRJointData stream published by vr_reader.py
and drives robot hardware — bypassing command_processor middleware.

Data flow::

    VRReader (leader)
        ↓  Zenoh "vr/joints"  (VRJointData)
    VRRobotController
        ├─ estop handling
        ├─ head joint positions
        ├─ arm joint positions  (whole_body stage only)
        └─ chassis velocity     (whole_body stage only, if robot has base)

Calibration stages (as published by vr_reader):
  static / head          — estop=True  → robot head moves, arms frozen
  whole_body_alignment   — estop=True  → same as above from controller perspective
  whole_body             — estop=False → all components active

Usage::

    omni-vr-robot
"""

from __future__ import annotations

import sys
import threading
import time
from enum import Enum
from typing import Optional

import numpy as np
import tyro
from dexbot_utils import RobotInfo
from dexcomm import Node, RateLimiter
from dexcomm.codecs import DictDataCodec
from dexcontrol.robot import Robot
from loguru import logger

from omniteleop import LIB_PATH
from omniteleop.common import get_config
from omniteleop.common.debug_display import get_debug_display
from omniteleop.common.logging import setup_logging
from omniteleop.common.schemas import VRJointData
from omniteleop.common.vr_mode_const import (
    INIT_HEAD_JOINTS,
    INIT_LEFT_ARM_JOINTS,
    INIT_RIGHT_ARM_JOINTS,
    INIT_TORSO_JOINTS,
)
from omniteleop.follower.robotiq import build_hande_command, send_activate


class _Mode(Enum):
    RUNNING = "running"
    STOP = "stop"
    EXIT = "exit"


class VRRobotController:
    """Drive robot hardware from VRJointData published by vr_reader.py."""

    def __init__(
        self,
        namespace: str = "",
        debug: bool = False,
        config_name: Optional[str] = None,
    ) -> None:
        self.node = Node(name="vr_robot_controller", namespace=namespace)

        robot_info = RobotInfo()
        self.has_torso = robot_info.has_torso
        self.has_chassis = robot_info.has_chassis

        config_path = None
        if config_name is not None:
            config_path = LIB_PATH / "configs" / f"{config_name}.yaml"
        self.config = get_config(config_path)

        self.control_rate = self.config.get_rate("control_rate", 100)
        self.feedback_rate = self.config.get_rate("feedback_rate", 50)

        # Zenoh subscriber for VR joint data
        vr_topic = self.config.get_topic("vr_joints", "vr/joints")
        self.vr_sub = self.node.create_subscriber(
            vr_topic, self._on_vr_joints, decoder=DictDataCodec.decode
        )

        # Joint feedback publisher
        joint_topic = self.config.get_topic("robot_joints")
        self.joint_pub = self.node.create_publisher(joint_topic, encoder=DictDataCodec.encode)

        self._latest: Optional[VRJointData] = None
        self._mode = _Mode.STOP
        self.initialize()

        self._debug_display = (
            get_debug_display("VRRobotController", self.control_rate, refresh_rate=10)
            if debug
            else None
        )

    # ── Zenoh callback ─────────────────────────────────────────────────────────

    def _on_vr_joints(self, data: dict) -> None:
        vr = VRJointData(**data)
        self._latest = vr

    # ── Initialisation ─────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Init robot hardware and move to home position."""
        logger.info("Initialising robot hardware ...")
        self.robot = Robot()
        self._set_home_position()
        send_activate(self.robot.left_arm)
        send_activate(self.robot.right_arm)
        self._mode = _Mode.STOP
        logger.success("VRRobotController ready.")

    def _set_home_position(self) -> None:
        # self.robot.estop.deactivate()
        # self.robot.head.set_mode("enable")
        time.sleep(0.1)

        joint_delta = 0.01

        # interpolate torso to target position
        curr_torso = self.robot.torso.get_joint_pos()
        steps = int(max(abs(np.array(INIT_TORSO_JOINTS) - np.array(curr_torso))) / joint_delta)
        for i in range(steps):
            interp_torso = (
                np.array(curr_torso)
                + (np.array(INIT_TORSO_JOINTS) - np.array(curr_torso)) * (i + 1) / steps
            )
            self.robot.torso.set_joint_pos(interp_torso.tolist(), wait_time=0.1, exit_on_reach=True)

        # interpolate left arm to target position
        curr_left = self.robot.left_arm.get_joint_pos()
        steps = int(max(abs(np.array(INIT_LEFT_ARM_JOINTS) - np.array(curr_left))) / joint_delta)
        for i in range(steps):
            interp_left = (
                np.array(curr_left)
                + (np.array(INIT_LEFT_ARM_JOINTS) - np.array(curr_left)) * (i + 1) / steps
            )
            self.robot.left_arm.set_joint_pos(
                interp_left.tolist(), wait_time=0.1, exit_on_reach=True
            )

        # interpolate right arm to target position
        curr_right = self.robot.right_arm.get_joint_pos()
        steps = int(max(abs(np.array(INIT_RIGHT_ARM_JOINTS) - np.array(curr_right))) / joint_delta)
        for i in range(steps):
            interp_right = (
                np.array(curr_right)
                + (np.array(INIT_RIGHT_ARM_JOINTS) - np.array(curr_right)) * (i + 1) / steps
            )
            self.robot.right_arm.set_joint_pos(
                interp_right.tolist(), wait_time=0.1, exit_on_reach=True
            )

        # interpolate head to target position
        curr_head = self.robot.head.get_joint_pos()
        steps = int(max(abs(np.array(INIT_HEAD_JOINTS) - np.array(curr_head))) / joint_delta)
        for i in range(steps):
            interp_head = (
                np.array(curr_head)
                + (np.array(INIT_HEAD_JOINTS) - np.array(curr_head)) * (i + 1) / steps
            )
            self.robot.head.set_joint_pos(interp_head.tolist(), wait_time=0.1, exit_on_reach=True)

        # self.robot.estop.activate()
        logger.info("Robot at home position.")

    # ── Joint feedback ─────────────────────────────────────────────────────────

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
                    {
                        "timestamp_ns": time.time_ns(),
                        "joints": positions,
                    }
                )
            except Exception as e:
                logger.warning(f"Joint feedback error: {e}")
            rate.sleep()

    # ── Control loop ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """Main control loop: read latest VRJointData and drive robot hardware."""
        self._feedback_running = True
        feedback_thread = threading.Thread(
            target=self._publish_joint_feedback, daemon=True, name="JointFeedback"
        )
        feedback_thread.start()

        rate = RateLimiter(self.control_rate)
        logger.info(f"Control loop at {self.control_rate} Hz")

        try:
            while True:
                vr = self._latest

                if vr is None:
                    rate.sleep()
                    continue

                # ── Estop ─────────────────────────────────────────────────────
                if vr.estop:
                    if self._mode != _Mode.STOP:
                        # self.robot.estop.activate()
                        self._mode = _Mode.STOP
                    # Head still moves during A/B stages (estop=True but head tracks)
                    if vr.head_pos:
                        self.robot.head.set_joint_pos(vr.head_pos, wait_time=0.0)
                    if self._debug_display is not None and vr.head_pos:
                        self._debug_display.print_robot_command(
                            {"head": {"pos": vr.head_pos}},
                            safety_flags={"estop": True},
                        )
                    rate.sleep()
                    continue

                # ── Estop released ────────────────────────────────────────────
                if self._mode == _Mode.STOP:
                    # self.robot.estop.deactivate()
                    # self.robot.head.set_mode("enable")
                    self._mode = _Mode.RUNNING

                # ── Head ──────────────────────────────────────────────────────
                if vr.head_pos:
                    self.robot.head.set_joint_pos(vr.head_pos, wait_time=0.0)

                # ── Arms (resetting + whole_body) ─────────────────────────────
                if vr.calib_stage in ("resetting", "whole_body", "whole_body_alignment"):
                    if vr.left_arm_pos:
                        self.robot.left_arm.set_joint_pos(vr.left_arm_pos)
                        left_arm_error = np.abs(np.array(vr.left_arm_pos) - self.robot.left_arm.get_joint_pos())
                        if np.any(left_arm_error > 0.3):
                            logger.info(f"Commanded left arm pos: {vr.left_arm_pos}")
                            logger.info(f"Current left arm pos: {self.robot.left_arm.get_joint_pos()}")
                            logger.warning(f"Warning: Large left arm error: {left_arm_error}")
                    if vr.right_arm_pos:
                        self.robot.right_arm.set_joint_pos(vr.right_arm_pos)
                        right_arm_error = np.abs(np.array(vr.right_arm_pos) - self.robot.right_arm.get_joint_pos())
                        if np.any(right_arm_error > 0.3):
                            logger.info(f"Commanded right arm pos: {vr.right_arm_pos}")
                            logger.info(f"Current right arm pos: {self.robot.right_arm.get_joint_pos()}")
                            logger.warning(f"Warning: Large right arm error: {right_arm_error}")

                # ── Grippers + chassis (whole_body only) ──────────────────────
                if vr.calib_stage == "whole_body":
                    self.robot.left_arm.send_ee_pass_through_message(
                        build_hande_command(vr.left_gripper)
                    )
                    self.robot.right_arm.send_ee_pass_through_message(
                        build_hande_command(vr.right_gripper)
                    )

                    if self.has_chassis and (vr.chassis_vx or vr.chassis_vy or vr.chassis_wz):
                        self.robot.chassis.set_velocity(
                            vx=vr.chassis_vx,
                            vy=vr.chassis_vy,
                            wz=vr.chassis_wz,
                            sequential_steering=abs(vr.chassis_vy) > 0.02,
                        )

                if self._debug_display is not None:
                    components: dict = {}
                    if vr.head_pos:
                        components["head"] = {"pos": vr.head_pos}
                    if vr.left_arm_pos:
                        components["left_arm"] = {"pos": vr.left_arm_pos}
                    if vr.right_arm_pos:
                        components["right_arm"] = {"pos": vr.right_arm_pos}
                    if self.has_chassis:
                        components["chassis"] = {
                            "vx": vr.chassis_vx,
                            "vy": vr.chassis_vy,
                            "wz": vr.chassis_wz,
                        }
                    self._debug_display.print_robot_command(
                        components,
                        safety_flags={"estop": vr.estop},
                    )

                rate.sleep()

        except KeyboardInterrupt:
            logger.info("VRRobotController stopped by user.")
        finally:
            self._feedback_running = False
            feedback_thread.join(timeout=2.0)
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources on shutdown."""
        if self.robot:
            # self.robot.estop.activate()
            self.robot.shutdown()
        self.node.shutdown()
        logger.info("VRRobotController cleaned up.")


# ── CLI entry point ────────────────────────────────────────────────────────────


def main(
    namespace: str = "",
    debug: bool = False,
    config_name: Optional[str] = None,
) -> None:
    """Follower robot controller for VR teleoperation.

    Subscribes to vr/joints (VRJointData) published by omni-vr and drives
    the robot hardware directly.
    """
    setup_logging(debug)
    ctrl = VRRobotController(namespace=namespace, debug=debug, config_name=config_name)
    ctrl.run()


if __name__ == "__main__":
    sys.exit(tyro.cli(main))
