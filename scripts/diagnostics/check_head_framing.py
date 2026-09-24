#!/usr/bin/env python3
"""Command ONLY the head, then photograph what the head camera sees.

Moves the head and nothing else -- the arms and torso are never commanded.
Use it to confirm the left arm is in frame before committing to a scan run.
"""
import argparse
import time

import cv2
import numpy as np
from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot

ap = argparse.ArgumentParser()
ap.add_argument("--head", type=float, nargs=3, default=[-0.25, 0.0, 0.0],
                help="head_j1 head_j2 head_j3 target (rad)")
ap.add_argument("--move-time", type=float, default=4.0,
                help="unused since dexcontrol 0.5; the glide is paced by its step size")
ap.add_argument("--max-vel", type=float, default=0.35,
                help="unused; wait_kwargs max_vel is 'not used by Head' in dexcontrol")
ap.add_argument("--out", default="/home/yixuan/Dexmate/tmp/scratchpad/head_view.png")
args = ap.parse_args()

cfg = get_robot_config()
cfg.sensors["head_camera"].enabled = True
robot = Robot(configs=cfg)
# dexcontrol's idle monitor pauses every subscriber not read for 5 s (arms, chassis,
# hands...). The pause re-configures the process-wide Zenoh session and stalls camera
# delivery ~300 ms once, ~6 s after Robot(). Keep everything subscribed instead
# (measured 2026-09-02 in record_head_sweep_video.py: 340 ms -> 37 ms worst latency).
robot.set_subscription_policy("always_on")
try:
    head = robot.head
    cur = np.asarray(head.get_joint_pos(), np.float64).ravel()
    q = np.asarray(args.head, np.float64).ravel()
    if q.shape != cur.shape:
        raise ValueError(f"--head has {q.size} values, expected {cur.size}")
    lim = head.joint_pos_limit
    if lim is not None:
        lim = np.asarray(lim, np.float64)
        if lim.shape != (cur.size, 2):
            raise ValueError(f"joint_pos_limit is {lim.shape}, expected {(cur.size, 2)}")
        clipped = np.clip(q, lim[:, 0], lim[:, 1])
        for i in range(cur.size):
            if abs(clipped[i] - q[i]) > 1e-9:
                print(f"  head_j{i + 1} {q[i]:+.4f} clipped to limit {clipped[i]:+.4f}")
        q = clipped
    print("head now   ", np.round(cur, 4).tolist())
    print("head target", np.round(q, 4).tolist())
    # Glide in small joint-space steps via the COMPONENT-level set_joint_pos, which is
    # what omniteleop.wbc_robot_home._interp_joint does and what actually moves this
    # robot. robot.set_joint_pos (the robot-level aggregate) lost wait_time in
    # dexcontrol 0.5, and robot.move_to_joint_pos raises "settling stalled: head stopped
    # converging ... noprogress" while never moving the head at all.
    STEP, WAIT = 0.01, 0.1          # matches wbc_robot_home ARM_HOME_STEP / _interp_joint
    n = max(1, int(np.max(np.abs(q - cur)) / STEP))
    print(f"gliding in {n} steps of {STEP:g} rad")
    for i in range(n):
        head.set_joint_pos((cur + (q - cur) * (i + 1) / n).tolist(),
                           wait_time=WAIT, exit_on_reach=True)
    time.sleep(1.0)
    after = np.asarray(head.get_joint_pos(), np.float64).ravel()
    print("head after ", np.round(after, 4).tolist(),
          f" residual {np.abs(after - q).max():.4f} rad")
    if np.abs(after - q).max() > 0.05:
        print("  WARNING: head did not reach the target -- re-run; if it persists the "
              "head is not holding position")

    cam = robot.sensors.head_camera
    frame = None
    for _ in range(80):
        obs = cam.get_obs()
        if isinstance(obs, dict) and obs.get("left_rgb") is not None:
            frame = np.asarray(obs["left_rgb"])
            break
        time.sleep(0.1)
    if frame is None:
        raise SystemExit("no left_rgb frame within 8s -- is the head publisher running?")
    cv2.imwrite(args.out, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"saved {args.out}  {frame.shape} {frame.dtype}")
finally:
    robot.shutdown()
