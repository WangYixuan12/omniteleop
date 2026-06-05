#!/usr/bin/env python3
"""Whole-body control for the Dexmate (Vega) robot via dexmate's high-level APIs, in SAPIEN.

Drives the whole-body IK solver
(:class:`omniteleop.follower.whole_body_ik.VegaWholeBodyIK`) and dispatches its output
through dexmate's **high-level motion-control API surface**:

- arms / torso / head -> ``set_joint_pos(q)``  (joint position control)
- mobile base -> velocity control, either:
    * high-level (default): ``chassis.move_straight`` / ``move_sideways`` / ``turn``
      (one dominant axis per tick), or
    * ``--direct_control``: ``chassis.set_velocity(vx, vy, wz)`` (coordinated,
      fine-grained; this is what ``set_motion_state`` / ``set_steering_angle`` /
      ``set_wheel_velocity`` build on).

By default the commands are applied to a SAPIEN simulation
(:class:`omniteleop.follower.wbc_sapien_sim.SapienSimRobot`). ``--real`` swaps in the
actual ``dexcontrol.Robot`` and issues identical calls to hardware (needs a live Vega
over Zenoh).

The targets are controlled from the terminal with ``--interactive`` (arrow keys +
type exact positions; see the on-screen help). Without ``--interactive`` the robot
holds its nominal posture.

Aggressive safety (for real-robot use): self-collision avoidance (a barrier over the
Dexmate collision spheres) plus a reactive hold gate that *stops* the robot and warns
when the CoM nears the wheel support-polygon edge (tip-over) or two links near
self-collision -- dangers the joint limits do not capture. See
:mod:`omniteleop.follower.wbc_safety`.

Examples::

    # interactive SAPIEN sim (needs a display):
    python scripts/wbc_sim_sapien.py --interactive

    # fine-grained base velocity control:
    python scripts/wbc_sim_sapien.py --interactive --direct_control

    # drive real hardware:
    python scripts/wbc_sim_sapien.py --interactive --real

Run with the dexmate conda env, e.g.
``/home/yixuan/miniforge3/envs/dexmate/bin/python scripts/wbc_sim_sapien.py``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pinocchio as pin

from omniteleop.follower.wbc_demo_utils import TargetController
from omniteleop.follower.wbc_sapien_sim import (
    DexcontrolRobotSink,
    SapienSimRobot,
    dispatch_base,
)
from omniteleop.follower.whole_body_ik import (
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    VegaWholeBodyIK,
    WBCConfig,
)

# Warn when the CoM ground-projection comes within this distance (m) of the
# support-polygon edge (or crosses it). The nominal Vega sits ~0.11 m inside the
# wheel triangle; it drops toward ~0.04 m only in genuinely dangerous poses (base
# unable to follow an extreme reach), so this default is quiet when stable.
SAFE_STABILITY_MARGIN = 0.06


def run_loop(ik, robot, args, controller: TargetController | None) -> None:
    """Main control loop: solve WBC, dispatch via the dexmate API, render."""
    left0 = ik.frame_pose(LEFT_EE_FRAME)
    right0 = ik.frame_pose(RIGHT_EE_FRAME)
    dt = 1.0 / args.rate

    t0 = time.perf_counter()
    last_print = 0.0
    last_warn = 0.0
    while True:
        now = time.perf_counter()
        t = now - t0
        if robot.viewer_closed:
            break

        if controller is not None:
            controller.poll()
            if controller.quit:
                break
            left_pos, right_pos = controller.left, controller.right
        else:
            left_pos, right_pos = left0.translation, right0.translation

        left_target = pin.SE3(left0.rotation, np.asarray(left_pos, dtype=float))
        right_target = pin.SE3(right0.rotation, np.asarray(right_pos, dtype=float))

        result = ik.solve(left_target, right_target, dt)

        # Dispatch through the dexmate high-level motion-control APIs.
        robot.left_arm.set_joint_pos(result.left_arm)
        robot.right_arm.set_joint_pos(result.right_arm)
        robot.torso.set_joint_pos(result.torso)
        robot.head.set_joint_pos(result.head)
        dispatch_base(robot, result.base_twist, args.direct_control)

        robot.set_targets(left_target, right_target)
        robot.step(dt)
        robot.render()

        # Safety: the solver already holds the robot (joints, and base on tip-over);
        # here we just surface the reason to the operator (throttled).
        if result.held and t - last_warn >= 0.3:
            last_warn = t
            print(f"\n[wbc_sapien] !! SAFETY HOLD: {result.safety_status} "
                  "-- reduce reach / move base.")
        elif (not result.held and result.safety_status != "ok"
              and t - last_warn >= 1.0):
            last_warn = t
            print(f"\n[wbc_sapien] ! {result.safety_status}")

        if t - last_print >= 0.5:
            last_print = t
            bx, by, byaw = result.base_pose
            tag = "  [HELD]" if result.held else ""
            print(
                f"t={t:6.1f}s  errL={result.left_ee_error * 1000:5.1f}mm  "
                f"errR={result.right_ee_error * 1000:5.1f}mm  "
                f"base x={bx:+.3f} y={by:+.3f} yaw={np.degrees(byaw):+5.1f}deg  "
                f"margin={result.stability_margin * 100:+4.1f}cm  "
                f"self={result.min_self_distance * 100:+4.1f}cm{tag}",
                end="\r",
            )

        sleep = dt - (time.perf_counter() - now)
        if sleep > 0:
            time.sleep(sleep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rate", type=float, default=100.0,
                        help="control/render rate in Hz (default: 100).")
    parser.add_argument("--interactive", action="store_true",
                        help="control the targets from the terminal (arrow keys / type xyz).")
    parser.add_argument("--direct_control", action="store_true",
                        help="base via fine-grained set_velocity instead of high-level "
                             "move_straight/move_sideways/turn.")
    parser.add_argument("--real", action="store_true",
                        help="drive the real dexcontrol.Robot (needs a live robot over Zenoh).")
    parser.add_argument("--no-viewer", action="store_true",
                        help="run headless (no SAPIEN window); for verification.")
    parser.add_argument("--safe-margin", type=float, default=SAFE_STABILITY_MARGIN,
                        help="tip-over warning threshold in meters (CoM-to-support-edge "
                             f"distance; default {SAFE_STABILITY_MARGIN}).")
    parser.add_argument("--urdf", default=None,
                        help="override the Vega URDF path (solver + sim must match).")
    args = parser.parse_args()

    config = WBCConfig()
    if args.urdf:
        config.urdf_path = args.urdf
    config.com_safety_margin = args.safe_margin
    ik = VegaWholeBodyIK(config)
    ik.reset()
    print(f"[wbc_sapien] model: nq={ik.model.nq} nv={ik.model.nv}  "
          f"collision_avoidance={ik.collision_enabled}  "
          f"com_safety={config.enable_com_safety} (margin {config.com_safety_margin * 100:.0f}cm)")

    if args.real:
        print("[wbc_sapien] connecting to real robot via dexcontrol ...")
        robot = DexcontrolRobotSink()
    else:
        robot = SapienSimRobot(urdf_path=ik.config.urdf_path, with_viewer=not args.no_viewer)

    base_mode = "set_velocity (fine-grained)" if args.direct_control \
        else "move_straight/sideways/turn (high-level)"
    print(f"[wbc_sapien] base control: {base_mode}  "
          f"{'(interactive)' if args.interactive else '(holding nominal pose)'} "
          "-- Ctrl-C to stop.")

    left0 = ik.frame_pose(LEFT_EE_FRAME)
    right0 = ik.frame_pose(RIGHT_EE_FRAME)
    try:
        if args.interactive:
            with TargetController(
                left0.translation, right0.translation, log_prefix="[wbc_sapien]"
            ) as controller:
                run_loop(ik, robot, args, controller)
        else:
            run_loop(ik, robot, args, None)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(robot.chassis, "stop"):
            robot.chassis.stop()
        robot.close()
        print("\n[wbc_sapien] stopped.")


if __name__ == "__main__":
    main()
