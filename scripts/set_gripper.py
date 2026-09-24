#!/usr/bin/env python3
"""Set the Robotiq Hand-E gripper(s) to a position and report the achieved value.

0.0 = fully OPEN, 1.0 = fully CLOSED (build_hande_command convention; obs/gripper
uses the same normalized scale). Run before record_arm_scan.py, which neither
commands nor records the gripper -- note the achieved value it prints.
"""
import argparse
import time

from dexcontrol.robot import Robot
from omniteleop.follower.robotiq import (
    build_hande_command,
    poll_gripper_status,
    send_ee_pass_through_with_timestamps,
)

ap = argparse.ArgumentParser()
ap.add_argument("--pos", type=float, required=True,
                help="0.0 = fully open .. 1.0 = fully closed")
ap.add_argument("--sides", default="left", choices=["left", "right", "both"])
ap.add_argument("--settle", type=float, default=2.5, help="seconds to wait before readback")
args = ap.parse_args()
if not 0.0 <= args.pos <= 1.0:
    raise ValueError(f"--pos must be in [0.0, 1.0], got {args.pos}")

sides = ["left", "right"] if args.sides == "both" else [args.sides]
robot = Robot()
try:
    for side in sides:
        arm = getattr(robot, f"{side}_arm")
        before = poll_gripper_status(arm, function_code=0x03, timeout_s=0.5)
        if before is None:
            raise SystemExit(
                f"{side}: FC03 warmup failed -- verify enable_ee_pass_through=True and "
                "that the detected EE type is UNKNOWN.")
        send_ee_pass_through_with_timestamps(arm, build_hande_command(args.pos))
        time.sleep(args.settle)
        after = poll_gripper_status(arm, function_code=0x03, timeout_s=0.5)
        if after is None:
            raise SystemExit(f"{side}: no FC03 reply after the move -- state unknown.")
        print(f"{side:5s} commanded {args.pos:.3f}   before {float(before['actual']):.4f}"
              f"   achieved {float(after['actual']):.4f}   <- record this")
finally:
    robot.shutdown()
