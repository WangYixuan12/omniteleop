#!/usr/bin/env python3
"""Safely move both arms to a fixed shutdown pose.

Interpolates each arm from its current joint position to a hard-coded
shutdown target using small joint-space steps — mirroring the interpolation
in ``vr_robot_controller.VRRobotController._set_home_position`` — so the arms
ease into the pose instead of jumping.

Usage::

    omni-safearm-shutdown
"""

from __future__ import annotations

import sys
import time

import numpy as np
import tyro
from dexcontrol.robot import Robot
from loguru import logger

from omniteleop.common.logging import setup_logging

# Shutdown joint targets (rad), captured from robot.{side}_arm.get_joint_pos()
SHUTDOWN_RIGHT_ARM_JOINTS = [
    -0.9635317,
    0.05968153,
    -0.08829969,
    -1.7133373,
    -0.52900755,
    0.2920564,
    -0.18649715,
]
SHUTDOWN_LEFT_ARM_JOINTS = [
    1.0682026,
    0.2366649,
    -0.02464056,
    -1.6520728,
    0.42287582,
    -0.54276425,
    0.26863933,
]


def _interpolate_arm(arm, target: list[float], joint_delta: float = 0.01) -> None:
    """Step ``arm`` from its current pose to ``target`` in ``joint_delta`` increments."""
    curr = np.array(arm.get_joint_pos())
    target = np.array(target)
    steps = int(max(abs(target - curr)) / joint_delta)
    if steps <= 0:
        arm.set_joint_pos(target.tolist(), wait_time=0.1, exit_on_reach=True)
        return
    for i in range(steps):
        interp = curr + (target - curr) * (i + 1) / steps
        arm.set_joint_pos(interp.tolist(), wait_time=0.1, exit_on_reach=True)


def main() -> None:
    """Move both arms to the shutdown pose, then shut down the robot."""
    setup_logging(debug=False)
    logger.info("Connecting to robot ...")
    robot = Robot()
    try:
        time.sleep(0.1)

        logger.info(f"Moving left arm to shutdown pose: {SHUTDOWN_LEFT_ARM_JOINTS}")
        _interpolate_arm(robot.left_arm, SHUTDOWN_LEFT_ARM_JOINTS)

        logger.info(f"Moving right arm to shutdown pose: {SHUTDOWN_RIGHT_ARM_JOINTS}")
        _interpolate_arm(robot.right_arm, SHUTDOWN_RIGHT_ARM_JOINTS)

        logger.success("Arms at shutdown pose.")
        logger.info(f"  left_arm  act: {list(robot.left_arm.get_joint_pos())}")
        logger.info(f"  right_arm act: {list(robot.right_arm.get_joint_pos())}")
    finally:
        robot.shutdown()
        logger.info("Robot shut down.")


if __name__ == "__main__":
    sys.exit(tyro.cli(main))
