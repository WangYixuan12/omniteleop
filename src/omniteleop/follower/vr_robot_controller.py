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
from dexcontrol.exceptions import ServiceUnavailableError
from dexcontrol.robot import Robot
from loguru import logger

from omniteleop import LIB_PATH
from omniteleop.common import get_config
from omniteleop.common.debug_display import get_debug_display
from omniteleop.common.logging import setup_logging
from omniteleop.common.recorder import FollowerEpisodeRecorder
from omniteleop.common.schemas import VRJointData
from omniteleop.common.vr_mode_const import (
    INIT_HEAD_JOINTS,
    INIT_LEFT_ARM_JOINTS,
    INIT_RIGHT_ARM_JOINTS,
    INIT_TORSO_JOINTS,
    SAFE_LEFT_ARM_JOINTS,
    SAFE_RIGHT_ARM_JOINTS,
)
from omniteleop.follower.robotiq import build_hande_command, send_activate

workspace_check = True


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
        workspace_check: bool = workspace_check,
        follower_save_dir: str = "/home/dexmate/yixuan/Dexmate/data/follower",
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

        # "track" → follow vr.head_pos; "fixed" → hold at INIT_HEAD_JOINTS
        self.head_mode = self.config.get("head_mode", "track")
        if self.head_mode not in ("track", "fixed"):
            raise ValueError(
                f"Invalid head_mode={self.head_mode!r}; expected 'track' or 'fixed'"
            )

        # "track" → follow vr.left_arm_pos; "fixed" → hold at INIT_LEFT_ARM_JOINTS
        self.left_arm_mode = self.config.get("left_arm_mode", "track")
        if self.left_arm_mode not in ("track", "fixed"):
            raise ValueError(
                f"Invalid left_arm_mode={self.left_arm_mode!r}; expected 'track' or 'fixed'"
            )

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

        if workspace_check:
            from omniteleop.follower.workspace_check import WorkspaceChecker
            self._workspace_checker = WorkspaceChecker()
        else:
            self._workspace_checker = None
            logger.warning("Right-arm workspace check disabled.")
        self._last_workspace_warn_t = 0.0

        # Per-100Hz-tick recorder. Episode_id is leader-driven via
        # VRJointData.episode_id (-1 idle, >=0 active). All rows include
        # vr_publish_ns = vr.timestamp_ns so the offline alignment script can
        # join with the leader-side HDF5 by exact match.
        self._follower_recorder = FollowerEpisodeRecorder(save_dir=follower_save_dir)
        self._prev_episode_id: int = -1

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

        # interpolate left arm: current → SAFE → INIT
        curr_left = self.robot.left_arm.get_joint_pos()
        for waypoint in (SAFE_LEFT_ARM_JOINTS, INIT_LEFT_ARM_JOINTS):
            steps = int(max(abs(np.array(waypoint) - np.array(curr_left))) / joint_delta)
            for i in range(steps):
                interp_left = (
                    np.array(curr_left)
                    + (np.array(waypoint) - np.array(curr_left)) * (i + 1) / steps
                )
                self.robot.left_arm.set_joint_pos(
                    interp_left.tolist(), wait_time=0.1, exit_on_reach=True
                )
            curr_left = waypoint

        # interpolate right arm: current → SAFE → INIT
        curr_right = self.robot.right_arm.get_joint_pos()
        for waypoint in (SAFE_RIGHT_ARM_JOINTS, INIT_RIGHT_ARM_JOINTS):
            steps = int(max(abs(np.array(waypoint) - np.array(curr_right))) / joint_delta)
            for i in range(steps):
                interp_right = (
                    np.array(curr_right)
                    + (np.array(waypoint) - np.array(curr_right)) * (i + 1) / steps
                )
                self.robot.right_arm.set_joint_pos(
                    interp_right.tolist(), wait_time=0.1, exit_on_reach=True
                )
            curr_right = waypoint

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
        # print("=" * 60)
        # print("Home position reached. Commanded vs actual joint values:")
        # print(f"  torso     cmd: {list(INIT_TORSO_JOINTS)}")
        # print(f"  torso     act: {list(self.robot.torso.get_joint_pos())}")
        # print(f"  left_arm  cmd: {list(INIT_LEFT_ARM_JOINTS)}")
        # print(f"  left_arm  act: {list(self.robot.left_arm.get_joint_pos())}")
        # print(f"  right_arm cmd: {list(INIT_RIGHT_ARM_JOINTS)}")
        # print(f"  right_arm act: {list(self.robot.right_arm.get_joint_pos())}")
        # print(f"  head      cmd: {list(INIT_HEAD_JOINTS)}")
        # print(f"  head      act: {list(self.robot.head.get_joint_pos())}")
        # print("=" * 60)
        logger.info("Robot at home position.")

    # ── Follower-side recorder helpers ─────────────────────────────────────────

    def _handle_episode_boundary(self, episode_id: int) -> None:
        """Watch `VRJointData.episode_id` and start/stop the follower recorder.

        Transitions:
            -1 → N        : start a fresh episode.
            N → -1        : stop current; nothing started.
            N → M (M != N): stop current, start M (handles missed -1 frames).
        """
        if episode_id == self._prev_episode_id:
            return
        if self._follower_recorder.recording:
            path = self._follower_recorder.stop()
            logger.info(f"FollowerEpisodeRecorder: stopped → {path}")
        if episode_id >= 0:
            self._follower_recorder.start(episode_id)
        self._prev_episode_id = episode_id

    @staticmethod
    def _sample_joint_state(component) -> tuple[np.ndarray, int]:
        """Atomic (pos, timestamp_ns) snapshot from a dexcontrol joint
        component; uses `_get_state` so pos and timestamp_ns come from the same
        state message. Returns (zeros, -1) if the state cache is still empty.
        """
        try:
            state = component._get_state()
        except ServiceUnavailableError:
            return np.zeros(0, dtype=np.float32), -1
        return np.asarray(state["pos"], dtype=np.float32), int(state["timestamp_ns"])

    def _record_follower_tick(
        self,
        vr: VRJointData,
        t_apply_ns: int,
        head_target: list[float],
        left_arm_target: list[float],
        right_arm_target: list[float],
        right_arm_applied: bool,
    ) -> None:
        """Build and push one per-tick row to the follower recorder.

        All `*_ns` fields are int64 in **dexmate** clock (the same clock as
        `time.time_ns()` on this machine); the only lambda-clock field is
        `vr_publish_ns`, copied verbatim from the inbound VRJointData. Offline
        alignment applies the dexmate→lambda offset (see `/meta/clock/...` in
        the leader-side HDF5) only to dexmate-tagged fields.
        """
        r_arm_pos, r_arm_ts = self._sample_joint_state(self.robot.right_arm)
        l_arm_pos, l_arm_ts = self._sample_joint_state(self.robot.left_arm)
        head_pos, head_ts = self._sample_joint_state(self.robot.head)
        if self.has_torso:
            torso_pos, torso_ts = self._sample_joint_state(self.robot.torso)
        else:
            torso_pos, torso_ts = np.zeros(3, dtype=np.float32), -1

        # Fixed-shape applied/* fields so HDF5 stacking is uniform across rows.
        left_applied = (
            np.asarray(left_arm_target, dtype=np.float32)
            if len(left_arm_target) == 7
            else np.zeros(7, dtype=np.float32)
        )
        right_target = (
            np.asarray(right_arm_target, dtype=np.float32)
            if len(right_arm_target) == 7
            else np.zeros(7, dtype=np.float32)
        )
        head_applied = (
            np.asarray(head_target, dtype=np.float32)
            if len(head_target) == 3
            else np.zeros(3, dtype=np.float32)
        )

        row = {
            "t_apply_ns": np.int64(t_apply_ns),
            "vr_publish_ns": np.int64(vr.timestamp_ns),
            "calib_stage": np.bytes_(vr.calib_stage),
            "applied": {
                "joint": {
                    "right_arm": right_target,
                    "left_arm": left_applied,
                    "head": head_applied,
                },
                "gripper": {
                    "right": np.float32(vr.right_gripper),
                    "left": np.float32(vr.left_gripper),
                },
                "right_arm_applied": np.bool_(right_arm_applied),
            },
            "obs": {
                "joint": {
                    "right_arm": r_arm_pos,
                    "left_arm": l_arm_pos,
                    "head": head_pos,
                    "torso": torso_pos,
                },
                "joint_ts": {
                    "right_arm": np.int64(r_arm_ts),
                    "left_arm": np.int64(l_arm_ts),
                    "head": np.int64(head_ts),
                    "torso": np.int64(torso_ts),
                },
            },
        }
        self._follower_recorder.record(row)

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

                # Leader-driven episode boundary. Start/stop the follower
                # recorder on -1 ↔ N transitions in VRJointData.episode_id.
                self._handle_episode_boundary(vr.episode_id)

                # ── Estop ─────────────────────────────────────────────────────
                if vr.estop:
                    if self._mode != _Mode.STOP:
                        # self.robot.estop.activate()
                        self._mode = _Mode.STOP
                    # Head still moves during A/B stages (estop=True but head tracks)
                    head_target = (
                        vr.head_pos if self.head_mode == "track" else INIT_HEAD_JOINTS
                    )
                    if head_target:
                        self.robot.head.set_joint_pos(head_target, wait_time=0.0)
                        # print(f"Commanded head pos: {head_target}")
                        # print(f"Current head pos: {self.robot.head.get_joint_pos()}")
                    if self._debug_display is not None and head_target:
                        self._debug_display.print_robot_command(
                            {"head": {"pos": head_target}},
                            safety_flags={"estop": True},
                        )
                    rate.sleep()
                    continue

                # ── Estop released ────────────────────────────────────────────
                if self._mode == _Mode.STOP:
                    # self.robot.estop.deactivate()
                    # self.robot.head.set_mode("enable")
                    self._mode = _Mode.RUNNING

                # Stamp once, immediately before the first set_joint_pos of
                # this iteration. All commands below execute synchronously
                # within a few hundred µs, so a single stamp captures the
                # apply moment with adequate precision for ms-level latency
                # analysis.
                t_apply_ns = time.time_ns()
                right_arm_applied = False

                # ── Head ──────────────────────────────────────────────────────
                head_target = (
                    vr.head_pos if self.head_mode == "track" else INIT_HEAD_JOINTS
                )
                if head_target:
                    self.robot.head.set_joint_pos(head_target, wait_time=0.0)

                # ── Arms (resetting + whole_body) ─────────────────────────────
                left_arm_target = (
                    vr.left_arm_pos
                    if self.left_arm_mode == "track"
                    else list(INIT_LEFT_ARM_JOINTS)
                )
                if vr.calib_stage in ("resetting", "whole_body", "whole_body_alignment"):
                    if left_arm_target:
                        self.robot.left_arm.set_joint_pos(left_arm_target)
                        left_arm_error = np.abs(
                            np.array(left_arm_target) - self.robot.left_arm.get_joint_pos()
                        )
                        if np.any(left_arm_error > 0.3):
                            logger.info(f"Commanded left arm pos: {left_arm_target}")
                            logger.info(
                                f"Current left arm pos: {self.robot.left_arm.get_joint_pos()}"
                            )
                            logger.warning(f"Warning: Large left arm error: {left_arm_error}")
                    if vr.right_arm_pos:
                        in_bounds = True
                        eef_xyz = None
                        if self._workspace_checker is not None:
                            head_for_fk = head_target if head_target else list(INIT_HEAD_JOINTS)
                            in_bounds, eef_xyz = self._workspace_checker.is_in_workspace(
                                right_arm=vr.right_arm_pos,
                                head=head_for_fk,
                                torso=list(INIT_TORSO_JOINTS),
                            )
                        if in_bounds:
                            self.robot.right_arm.set_joint_pos(vr.right_arm_pos)
                            right_arm_applied = True
                            right_arm_error = np.abs(
                                np.array(vr.right_arm_pos) - self.robot.right_arm.get_joint_pos()
                            )
                            if np.any(right_arm_error > 0.3):
                                logger.info(f"Commanded right arm pos: {vr.right_arm_pos}")
                                logger.info(
                                    f"Current right arm pos: {self.robot.right_arm.get_joint_pos()}"
                                )
                                logger.warning(
                                    f"Warning: Large right arm error: {right_arm_error}"
                                )
                        else:
                            now = time.monotonic()
                            if now - self._last_workspace_warn_t > 0.5:
                                self._last_workspace_warn_t = now
                                bounds = self._workspace_checker.bounds
                                logger.warning(
                                    f"Right EEF out of workspace: "
                                    f"xyz={eef_xyz.round(3).tolist()} "
                                    f"(bounds x{bounds['x']}, y{bounds['y']}, "
                                    f"z{bounds['z']}) — freezing right arm."
                                )

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
                    if head_target:
                        components["head"] = {"pos": head_target}
                    if left_arm_target:
                        components["left_arm"] = {"pos": left_arm_target}
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

                if self._follower_recorder.recording:
                    self._record_follower_tick(
                        vr=vr,
                        t_apply_ns=t_apply_ns,
                        head_target=list(head_target) if head_target else [],
                        left_arm_target=list(left_arm_target) if left_arm_target else [],
                        right_arm_target=list(vr.right_arm_pos) if vr.right_arm_pos else [],
                        right_arm_applied=right_arm_applied,
                    )

                rate.sleep()

        except KeyboardInterrupt:
            logger.info("VRRobotController stopped by user.")
        finally:
            self._feedback_running = False
            feedback_thread.join(timeout=2.0)
            if self._follower_recorder.recording:
                self._follower_recorder.stop()
            if (
                self._follower_recorder._save_thread is not None
                and self._follower_recorder._save_thread.is_alive()
            ):
                logger.info("Waiting for FollowerEpisodeRecorder save to finish ...")
                self._follower_recorder._save_thread.join()
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
    workspace_check: bool = workspace_check,
    follower_save_dir: str = "/home/dexmate/yixuan/Dexmate/data/follower",
) -> None:
    """Follower robot controller for VR teleoperation.

    Subscribes to vr/joints (VRJointData) published by omni-vr and drives
    the robot hardware directly.

    Args:
        workspace_check: Gate right-arm commands on the configured Cartesian
            workspace bounds. Pass --workspace-check to enable.
        follower_save_dir: Directory for per-100Hz follower recordings
            (``episode_<N>_follower.hdf5``). On Dexmate, default lives under
            the shared mount that maps to ``/home/yixuan/Dexmate/...`` on
            Lambda; the alignment script reads from the Lambda view directly.
    """
    setup_logging(debug)
    ctrl = VRRobotController(
        namespace=namespace,
        debug=debug,
        config_name=config_name,
        workspace_check=workspace_check,
        follower_save_dir=follower_save_dir,
    )
    ctrl.run()


if __name__ == "__main__":
    sys.exit(tyro.cli(main))
