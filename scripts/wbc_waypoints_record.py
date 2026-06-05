#!/usr/bin/env python3
"""Drive the Vega whole-body IK through a fixed list of Cartesian waypoints, recorded to video.

Run in the dexmate conda env (has sapien + pinocchio + pink + dexcomm)::

    /home/yixuan/miniforge3/envs/dexmate/bin/python scripts/wbc_waypoints_record.py \
        --output /tmp/wbc_waypoints.mp4

Add ``--no-viewer`` to record headlessly and ``--repeat N`` to loop the sequence.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

import numpy as np
import pinocchio as pin

from omniteleop.follower.wbc_sapien_sim import SapienSimRobot
from omniteleop.follower.whole_body_ik import (
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    VegaWholeBodyIK,
    WBCConfig,
)

SAFE_STABILITY_MARGIN = 0.06


@dataclass
class Waypoint:
    """One dual-arm Cartesian target, as offsets from the home EE pose (world axes).

    Args:
        label: human-readable name shown in the log / on completion.
        left: ``(dx, dy, dz)`` offset (m) added to the home ``L_ee`` position.
        right: ``(dx, dy, dz)`` offset (m) added to the home ``R_ee`` position.
        yaw_deg: re-orient both targets by this yaw (deg) about the world vertical,
            in place (position unchanged). A target *orientation* change is what makes
            the base yaw to follow; a pure position offset only translates it.
        dwell: optional per-waypoint dwell override (s); ``None`` uses ``--dwell``.
    """

    label: str
    left: Sequence[float] = (0.0, 0.0, 0.0)
    right: Sequence[float] = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0
    dwell: Optional[float] = None


# A demo sweep that exercises every whole-body behavior in turn: drive the base
# forward to a far target, strafe it sideways, and yaw it to follow a re-orientation --
# returning home between moves. Offsets are from the home EE pose, in world axes.
DEFAULT_WAYPOINTS: List[Waypoint] = [
    Waypoint("home"),
    Waypoint("reach forward", left=(0.50, 0.0, 0.0), right=(0.50, 0.0, 0.0)),
    Waypoint("back home"),
    Waypoint("strafe left", left=(0.20, 0.40, 0.05), right=(0.20, 0.40, 0.05)),
    Waypoint("strafe right", left=(0.20, -0.40, 0.05), right=(0.20, -0.40, 0.05)),
    Waypoint("back home"),
    Waypoint("lift up", left=(0.15, 0.0, 0.25), right=(0.15, 0.0, 0.25)),
    Waypoint("back home"),
    Waypoint("turn left", yaw_deg=40.0),
    Waypoint("turn right", yaw_deg=-40.0),
    Waypoint("home"),
]


@dataclass
class ScheduleStep:
    """One control-tick's commanded targets, emitted by :func:`build_schedule`."""

    label: str
    phase: str  # "travel" (easing toward the waypoint) | "dwell" (holding it)
    left: pin.SE3
    right: pin.SE3
    waypoint_done: bool  # True on the final tick of this waypoint's dwell


def _rot_z(theta: float) -> np.ndarray:
    """Rotation matrix about +z by ``theta`` radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _smoothstep(s: float) -> float:
    """Smoothstep easing on ``[0, 1]`` (zero velocity at both endpoints)."""
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * (3.0 - 2.0 * s)


def _interp_pose(a: pin.SE3, b: pin.SE3, s: float) -> pin.SE3:
    """Interpolate ``a -> b`` at fraction ``s``: lerp translation, slerp rotation."""
    s = float(np.clip(s, 0.0, 1.0))
    trans = (1.0 - s) * a.translation + s * b.translation
    rel = pin.log3(a.rotation.T @ b.rotation)  # rotation vector from a to b
    rot = a.rotation @ pin.exp3(s * rel)
    return pin.SE3(rot, trans)


def _goal_pose(home: pin.SE3, offset: Sequence[float], yaw_deg: float) -> pin.SE3:
    """Absolute world target = home position + offset, re-oriented by ``yaw_deg``."""
    rot = _rot_z(np.radians(yaw_deg)) @ home.rotation
    trans = home.translation + np.asarray(offset, dtype=float)
    return pin.SE3(rot, trans)


def drive_base(robot: SapienSimRobot, twist: np.ndarray, eps: float = 1e-3) -> None:
    """Command the mobile base through the dexmate chassis velocity API.

    Routes the solver's base twist ``(vx, vy, wz)`` (m/s, m/s, rad/s; body frame,
    x-fwd / y-left, CCW+) to the matching ``dexcontrol.Chassis`` call: a single
    high-level helper (``move_straight`` / ``move_sideways`` / ``turn``) when the motion
    is essentially single-axis -- the other two components below ``eps``, so omitting
    them does not drift the base -- otherwise the coordinated ``set_velocity``, which
    preserves the full twist. With the sim's exact planar-joint integration this keeps
    the rendered base faithful to the solver whichever API is chosen.
    """
    vx, vy, wz = (float(v) for v in twist)
    ax, ay, aw = abs(vx), abs(vy), abs(wz)
    if max(ax, ay, aw) < eps:
        robot.chassis.stop()
    elif ay < eps and aw < eps:
        robot.chassis.move_straight(vx)
    elif ax < eps and aw < eps:
        robot.chassis.move_sideways(vy)
    elif ax < eps and ay < eps:
        robot.chassis.turn(wz)
    else:
        robot.chassis.set_velocity(vx, vy, wz)


def build_schedule(
    home_left: pin.SE3,
    home_right: pin.SE3,
    waypoints: Sequence[Waypoint],
    travel_time: float,
    dwell_time: float,
    dt: float,
) -> Iterator[ScheduleStep]:
    """Yield per-tick targets that ease through ``waypoints`` then dwell at each.

    The target travels from the previous waypoint to the next over ``travel_time``
    (smoothstep ease-in/out) and then holds for ``dwell_time`` (or the waypoint's own
    ``dwell``). Goals are computed from the *fixed* home pose, so each waypoint is an
    absolute world target independent of base drift accumulated so far.
    """
    prev_left, prev_right = home_left, home_right
    n_travel = max(1, round(travel_time / dt))
    for wp in waypoints:
        goal_left = _goal_pose(home_left, wp.left, wp.yaw_deg)
        goal_right = _goal_pose(home_right, wp.right, wp.yaw_deg)
        for i in range(1, n_travel + 1):
            s = _smoothstep(i / n_travel)
            yield ScheduleStep(
                wp.label, "travel",
                _interp_pose(prev_left, goal_left, s),
                _interp_pose(prev_right, goal_right, s),
                False,
            )
        dwell = dwell_time if wp.dwell is None else wp.dwell
        n_dwell = max(1, round(dwell / dt))
        for k in range(n_dwell):
            yield ScheduleStep(wp.label, "dwell", goal_left, goal_right, k == n_dwell - 1)
        prev_left, prev_right = goal_left, goal_right


def load_waypoints(path: str) -> List[Waypoint]:
    """Load a waypoint list from a JSON file (see the module docstring for the schema)."""
    with open(path) as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of waypoint objects")
    out: List[Waypoint] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{i}]: each waypoint must be a JSON object")
        out.append(Waypoint(
            label=str(item.get("label", f"wp{i}")),
            left=tuple(item.get("left", (0.0, 0.0, 0.0))),
            right=tuple(item.get("right", (0.0, 0.0, 0.0))),
            yaw_deg=float(item.get("yaw_deg", 0.0)),
            dwell=(None if item.get("dwell") is None else float(item["dwell"])),
        ))
    if not out:
        raise ValueError(f"{path}: waypoint list is empty")
    return out


def run(ik: VegaWholeBodyIK, robot: SapienSimRobot, args, waypoints: Sequence[Waypoint]) -> None:
    """Drive the robot through ``waypoints`` (optionally ``--repeat``-ed), recording video."""
    dt = 1.0 / args.rate
    home_left = ik.frame_pose(LEFT_EE_FRAME)
    home_right = ik.frame_pose(RIGHT_EE_FRAME)

    t0 = time.perf_counter()
    last_print = 0.0
    last_warn = 0.0
    for rep in range(args.repeat):
        for step in build_schedule(
            home_left, home_right, waypoints, args.travel, args.dwell, dt
        ):
            now = time.perf_counter()
            t = now - t0
            if robot.viewer_closed:
                return

            result = ik.solve(step.left, step.right, dt)

            robot.left_arm.set_joint_pos(result.left_arm)
            robot.right_arm.set_joint_pos(result.right_arm)
            robot.torso.set_joint_pos(result.torso)
            robot.head.set_joint_pos(result.head)
            # Drive the base through the dexmate chassis velocity API; the sim integrates
            # the twist with the same planar-joint exponential the solver uses, so the
            # rendered end-effectors stay on the target frames.
            drive_base(robot, result.base_twist)

            robot.set_targets(step.left, step.right)
            robot.step(dt)
            robot.render()

            # Safety: the solver already holds the robot on a violation; surface why.
            if result.held and t - last_warn >= 0.3:
                last_warn = t
                print(f"\n[wbc_waypts] !! SAFETY HOLD: {result.safety_status}")
            elif (not result.held and result.safety_status != "ok"
                  and t - last_warn >= 1.0):
                last_warn = t
                print(f"\n[wbc_waypts] ! {result.safety_status}")

            if t - last_print >= 0.25:
                last_print = t
                bx, by, byaw = result.base_pose
                tag = "  [HELD]" if result.held else ""
                rep_tag = f" rep {rep + 1}/{args.repeat}" if args.repeat > 1 else ""
                print(
                    f"t={t:6.1f}s{rep_tag}  {step.label:14s} {step.phase:6s}  "
                    f"errL={result.left_ee_error * 1000:5.1f}mm "
                    f"errR={result.right_ee_error * 1000:5.1f}mm  "
                    f"base x={bx:+.3f} y={by:+.3f} yaw={np.degrees(byaw):+5.1f}deg  "
                    f"margin={result.stability_margin * 100:+4.1f}cm{tag}",
                    end="\r",
                )

            if step.waypoint_done:
                bx, by, byaw = result.base_pose
                print(
                    f"\n[wbc_waypts] reached '{step.label}': "
                    f"errL={result.left_ee_error * 1000:.1f}mm "
                    f"errR={result.right_ee_error * 1000:.1f}mm  "
                    f"base x={bx:+.3f} y={by:+.3f} yaw={np.degrees(byaw):+.1f}deg"
                )

            sleep = dt - (time.perf_counter() - now)
            if sleep > 0:
                time.sleep(sleep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", default="/home/yixuan/Dexmate/wbc_waypoints.mp4",
                        help="output video path (mp4) (default: /home/yixuan/Dexmate/wbc_waypoints.mp4).")
    parser.add_argument("--points", default=None,
                        help="JSON file of waypoints (default: built-in DEFAULT_WAYPOINTS).")
    parser.add_argument("--travel", type=float, default=3.0,
                        help="seconds to ease the target from one waypoint to the next "
                             "(default: 3.0).")
    parser.add_argument("--dwell", type=float, default=2.0,
                        help="seconds to hold at each waypoint while the IK settles "
                             "(default: 2.0).")
    parser.add_argument("--repeat", type=int, default=1,
                        help="number of times to loop the whole sequence (default: 1).")
    parser.add_argument("--rate", type=float, default=100.0,
                        help="control/record rate in Hz (default: 100).")
    parser.add_argument("--width", type=int, default=960, help="video width (default: 960).")
    parser.add_argument("--height", type=int, default=540, help="video height (default: 540).")
    parser.add_argument("--video-fps", type=float, default=15.0,
                        help="recorded video fps (default: 15; frames are decimated from "
                             "the higher --rate control loop to real-time).")
    parser.add_argument("--no-viewer", action="store_true",
                        help="record headlessly (no SAPIEN window).")
    parser.add_argument("--safe-margin", type=float, default=SAFE_STABILITY_MARGIN,
                        help="tip-over warning threshold in meters "
                             f"(default {SAFE_STABILITY_MARGIN}).")
    parser.add_argument("--urdf", default=None, help="override the Vega URDF path.")
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be >= 1")

    waypoints = load_waypoints(args.points) if args.points else DEFAULT_WAYPOINTS

    cfg = WBCConfig()
    if args.urdf:
        cfg.urdf_path = args.urdf
    cfg.com_safety_margin = args.safe_margin
    ik = VegaWholeBodyIK(cfg)
    ik.reset()
    print(f"[wbc_waypts] model: nq={ik.model.nq} nv={ik.model.nv}  "
          f"collision_avoidance={ik.collision_enabled}  "
          f"com_safety={cfg.enable_com_safety} (margin {cfg.com_safety_margin * 100:.0f}cm)")
    print("[wbc_waypts] waypoint plan (offsets from home EE pose, world axes "
          "x-fwd/y-left/z-up):")
    for i, wp in enumerate(waypoints):
        print(f"   {i + 1:2d}. {wp.label:14s} "
              f"L=({wp.left[0]:+.2f},{wp.left[1]:+.2f},{wp.left[2]:+.2f}) "
              f"R=({wp.right[0]:+.2f},{wp.right[1]:+.2f},{wp.right[2]:+.2f}) "
              f"yaw={wp.yaw_deg:+.0f}deg")
    per_wp = args.travel + args.dwell
    print(f"[wbc_waypts] ~{per_wp:.1f}s/waypoint -> "
          f"~{per_wp * len(waypoints) * args.repeat:.0f}s total. Ctrl-C to stop and save.")

    robot = SapienSimRobot(
        urdf_path=ik.config.urdf_path,
        with_viewer=not args.no_viewer,
        record_video=args.output,
        video_fps=args.video_fps,
        video_size=(args.width, args.height),
    )
    try:
        run(ik, robot, args, waypoints)
    except KeyboardInterrupt:
        pass
    finally:
        robot.chassis.stop()  # zero the base velocity before tearing down
        robot.close()  # releases the video writer
        print("\n[wbc_waypts] stopped.")


if __name__ == "__main__":
    main()
