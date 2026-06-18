"""Step 3 live odometry verification (REQUIRES A HUMAN AT THE ROBOT).

Run in the yixuan_yifan conda env while the bridge container is up:

    conda run -n yixuan_yifan python SLAM/scripts/odom_sanity.py --test units
    conda run -n yixuan_yifan python SLAM/scripts/odom_sanity.py --test watch

Tests:
  units : commands chassis.set_velocity(0.2, 0, 0) for 3 s and prints the raw
          state/chassis/drive `vel` values. If both read ~+0.2 the firmware
          reports signed m/s -> drive_state_mode: ms (current default).
          If they read ~+-2.3 rad/s -> set drive_state_mode: rad.
          !! Clears 0.6 m ahead of the robot before running. E-stop in reach.
  watch : passive — prints the bridge's integrated odometry pose (from zenoh
          slam/pose once SLAM runs, plus raw wheel state) at 2 Hz while YOU
          drive the robot with the usual teleop. Use it for:
            - 1 m tape-measure forward drive  -> |x error| < 5 %
            - 0.5 m strafe                    -> y sign and magnitude
            - 360 deg spin back to a tape mark -> yaw closure error
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_topics import ROBOT_PREFIX, find_zenoh_config  # noqa: E402


def test_units() -> None:
    from dexcontrol.core.config import get_robot_config
    from dexcontrol.robot import Robot

    print("!! The robot will drive FORWARD at 0.2 m/s for 3 s. Clear the area.")
    if input("Type 'go' to proceed: ").strip() != "go":
        print("aborted")
        return

    from dexcomm import Subscriber
    from dexcomm.codecs import JointStateCodec

    samples: list[np.ndarray] = []
    sub = Subscriber(f"{ROBOT_PREFIX}/state/chassis/drive",
                     callback=lambda m: samples.append(np.asarray(m["vel"])),
                     decoder=JointStateCodec.decode, config=find_zenoh_config())

    configs = get_robot_config()
    robot = Robot(configs=configs)
    try:
        t0 = time.time()
        while time.time() - t0 < 3.0:
            robot.chassis.set_velocity(0.2, 0.0, 0.0)
            time.sleep(0.05)
        robot.chassis.set_velocity(0.0, 0.0, 0.0)
    finally:
        robot.shutdown()
        sub.shutdown()

    if not samples:
        raise RuntimeError("no drive state received during the test")
    arr = np.stack(samples[len(samples) // 2:])  # steady-state half
    mean = arr.mean(axis=0)
    print(f"\nmean drive vel during 0.2 m/s forward: L={mean[0]:+.4f} R={mean[1]:+.4f}")
    if np.allclose(mean, 0.2, atol=0.05):
        print("=> values are signed m/s: keep drive_state_mode: ms")
    elif np.allclose(np.abs(mean), 0.2 / 0.0861, atol=0.6):
        print("=> values are joint rad/s: set drive_state_mode: rad")
    else:
        print("=> UNEXPECTED values; inspect manually before mapping")


def test_watch() -> None:
    from dexcomm import Subscriber
    from dexcomm.codecs import JointStateCodec, JsonDataCodec

    state: dict[str, object] = {}
    subs = [
        Subscriber(f"{ROBOT_PREFIX}/state/chassis/steer",
                   callback=lambda m: state.__setitem__("steer", m["pos"]),
                   decoder=JointStateCodec.decode, config=find_zenoh_config()),
        Subscriber(f"{ROBOT_PREFIX}/state/chassis/drive",
                   callback=lambda m: state.__setitem__("drive", m["vel"]),
                   decoder=JointStateCodec.decode, config=find_zenoh_config()),
        Subscriber(f"{ROBOT_PREFIX}/slam/pose",
                   callback=lambda m: state.__setitem__("pose", m),
                   decoder=JsonDataCodec.decode, config=find_zenoh_config()),
    ]
    print("watching (ctrl-c to stop); drive the robot via teleop now")
    try:
        while True:
            steer = state.get("steer")
            drive = state.get("drive")
            pose = state.get("pose")
            line = []
            if steer is not None and drive is not None:
                line.append(f"steer=[{steer[0]:+.3f},{steer[1]:+.3f}] "
                            f"drive=[{drive[0]:+.3f},{drive[1]:+.3f}]")
            if isinstance(pose, dict):
                x, y, _ = pose["pos"]
                line.append(f"slam x={x:+.3f} y={y:+.3f} "
                            f"yaw={np.degrees(pose['yaw']):+.1f}deg")
            print("  ".join(line) or "waiting for data...", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for s in subs:
            s.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", choices=["units", "watch"], required=True)
    args = parser.parse_args()
    {"units": test_units, "watch": test_watch}[args.test]()


if __name__ == "__main__":
    main()
