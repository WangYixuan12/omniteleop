#!/usr/bin/env python3
"""Drive the Vega whole-body IK through a fixed list of Cartesian waypoints, recorded to video.

The robot moves the EEF **target** smoothly (smoothstep ease-in/out) from the previous waypoint to
the next over ``--travel`` seconds, then dwells at it for ``--dwell`` seconds while the
IK settles. 

The mobile base is driven through the **official dexmate chassis velocity path**:
``chassis.set_velocity(vx, vy, wz, wait_time=0.0, sequential_steering=False)`` wrapped by
:class:`~omniteleop.follower.wbc_sapien_sim.BaseVelocityController` (deadband + slew +
e-stop). ``--base-loop`` selects how the commanded twist is formed: ``closed`` (default)
adds PD feedback (``base_tracking_twist``) that steers the rendered base onto the solver's
base pose -- so the rendered robot faithfully follows the IK and the end-effectors stay on
the drawn (raw-target) frames; ``open`` feeds only the IK base twist (feed-forward), so
command shaping (deadband/slew) and step inputs let the rendered base drift and the EEFs
visibly lag the markers. Both work in kinematic and ``--physics`` modes.

Edit :data:`DEFAULT_WAYPOINTS` below, or pass ``--points file.json`` (a list of
``{"label", "left":[x,y,z], "right":[x,y,z], "yaw_deg", "dwell"}`` objects; ``left`` /
``right`` are offsets from home in world axes).

After the scripted waypoint tour, the node traces a continuous **half-square +
half-circle** ground trajectory: a U of three ``--shape-side`` m sides (forward, left,
back) closed by a semicircle of radius ``--shape-side / 2`` back to home. Unlike the
waypoint tour -- which eases to a full stop at every waypoint -- this is one smooth sweep:
the EE target is swept along the path at ``--shape-speed`` m/s average (eased from rest
only at the two ends, see :func:`build_trajectory_schedule`) so the base drives the whole
shape without stop-and-go. Use ``--skip-tour`` to run only the shape, or ``--skip-shape``
to run only the original tour.

Run in the dexmate conda env (has sapien + pinocchio + pink + dexcomm)::

    /home/yixuan/miniforge3/envs/dexmate/bin/python scripts/wbc_waypoints_record.py \
        --output /tmp/wbc_waypoints.mp4

``--physics`` switches the base from kinematic gliding to **wheel actuation under SAPIEN
physics**: the solver's base twist is realized by swerve steer+drive on the actual wheel
joints (the base pose emerges from wheel-ground contact, as on hardware), and a PD base
controller (``base_tracking_twist``) makes the physical base track the solver's reference.
EE tracking is then *approximate* (wheel slip/lag) -- the realistic trade-off; the
kinematic default stays exact. See
``omniteleop.follower.wbc_sapien_physics.SapienPhysicsRobot``.

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

from omniteleop.follower.wbc_sapien_physics import SapienPhysicsRobot
from omniteleop.follower.wbc_sapien_sim import (
    BASE_VEL_LIMIT,
    BaseVelocityController,
    SapienSimRobot,
    base_tracking_twist,
)
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


# A wide-range demo tour: drive the base ~1 m forward to a far target, sweep it a full
# ~1.4 m side-to-side, reach high, and yaw it through large re-orientations -- returning
# home between moves. Offsets are from the home EE pose, in world axes. (For an even
# larger range, raise these and give the base time with a longer ``--travel``.)
DEFAULT_WAYPOINTS: List[Waypoint] = [
    Waypoint("home"),
    Waypoint("reach forward", left=(1.00, 0.00, 0.0), right=(1.00, 0.00, 0.0)),
    Waypoint("sweep left", left=(1.00, 0.70, 0.10), right=(1.00, 0.70, 0.10)),
    Waypoint("sweep right", left=(1.00, -0.70, 0.10), right=(1.00, -0.70, 0.10)),
    Waypoint("back home"),
    Waypoint("reach high", left=(0.30, 0.00, 0.30), right=(0.30, 0.00, 0.30)),
    Waypoint("back home"),
    Waypoint("turn left", yaw_deg=60.0),
    Waypoint("turn right", yaw_deg=-60.0),
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


def half_square_circle_path(side: float, arc_pts: int = 96) -> np.ndarray:
    """Closed half-square + half-circle ground path of ``(dx, dy, dz)`` offsets from home.

    In world axes (x forward, y left), starting at home ``(0, 0)``:

    * **half square** -- three sides of a square of length ``side``: forward to
      ``(side, 0)``, left to ``(side, side)``, then back to ``(0, side)``;
    * **half circle** -- a semicircle of radius ``side / 2`` (centered at ``(0, side/2)``,
      bulging to ``-x``) that closes the path from ``(0, side)`` back to home ``(0, 0)``.

    The arc is densified into ``arc_pts`` samples so the straight-line interpolation done
    by :func:`build_trajectory_schedule` approximates a true circle; the three straight
    sides need only their corners. ``z`` stays at the home EE height. Returns an
    ``(N, 3)`` array whose first and last rows are both home ``(0, 0, 0)``.
    """
    s = float(side)
    if s <= 0.0:
        raise ValueError(f"side must be > 0, got {side}")
    r = s / 2.0
    corners = [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)]  # forward, left, back
    # Semicircle from (0, s) [phi=pi/2] to (0, 0) [phi=3pi/2], bulging -x at phi=pi.
    phis = np.linspace(np.pi / 2.0, 3.0 * np.pi / 2.0, arc_pts)
    arc = np.column_stack([r * np.cos(phis), r + r * np.sin(phis)])
    xy = np.vstack([np.asarray(corners, dtype=float), arc[1:]])  # arc[0] == (0, s) corner
    return np.column_stack([xy, np.zeros(len(xy))])


def build_trajectory_schedule(
    home_left: pin.SE3,
    home_right: pin.SE3,
    path: np.ndarray,
    speed: float,
    dt: float,
    settle_time: float,
    label: str = "half-sq+circle",
) -> Iterator[ScheduleStep]:
    """Yield per-tick targets sweeping the EE smoothly along a continuous ``path``.

    ``path`` is an ``(N, 3)`` polyline of offsets from home (world axes). The target is
    swept along it by **arc length** at ``speed`` m/s average, with a single global
    smoothstep ease so it starts and ends at rest -- and, crucially, does *not* stop at
    the interior vertices (the square corners are rounded only by the base slew limit).
    This is what makes the shape one continuous trajectory rather than the stop-and-go of
    :func:`build_schedule`. Both arms get the *same* offset, so their relative transform is
    fixed and the base translates to follow. A final ``settle_time`` dwell lets the IK
    converge on the closing pose. Goals are absolute world targets from the fixed home pose.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        raise ValueError(f"path must be (N>=2, 3); got {path.shape}")
    deltas = np.diff(path, axis=0)
    seglen = np.linalg.norm(deltas, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(cum[-1])
    if total <= 0.0:
        raise ValueError("path has zero length")
    if speed <= 0.0:
        raise ValueError(f"speed must be > 0, got {speed}")

    def at(arc: float) -> np.ndarray:
        """Point at arc length ``arc`` (m) along the polyline."""
        arc = float(np.clip(arc, 0.0, total))
        i = int(np.clip(np.searchsorted(cum, arc, side="right") - 1, 0, len(seglen) - 1))
        frac = (arc - cum[i]) / seglen[i] if seglen[i] > 1e-9 else 0.0
        return path[i] + frac * deltas[i]

    n = max(1, round((total / speed) / dt))
    for k in range(1, n + 1):
        arc = _smoothstep(k / n) * total  # ease the whole sweep in/out, no interior stops
        off = at(arc)
        yield ScheduleStep(
            label, "travel",
            _goal_pose(home_left, off, 0.0),
            _goal_pose(home_right, off, 0.0),
            False,
        )
    goal_left = _goal_pose(home_left, path[-1], 0.0)
    goal_right = _goal_pose(home_right, path[-1], 0.0)
    n_dwell = max(1, round(settle_time / dt))
    for j in range(n_dwell):
        yield ScheduleStep(label, "dwell", goal_left, goal_right, j == n_dwell - 1)


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


def run(
    ik: VegaWholeBodyIK,
    robot: "SapienSimRobot | SapienPhysicsRobot",
    args,
    waypoints: Sequence[Waypoint],
    base_ctrl: BaseVelocityController,
) -> None:
    """Drive the robot through ``waypoints`` (optionally ``--repeat``-ed), recording video.

    All base motion goes through ``base_ctrl`` (deadband -> slew -> e-stop -> coordinated
    ``set_velocity(..., sequential_steering=False)``). With ``--base-loop closed`` (default)
    ``base_tracking_twist`` closes the loop so the rendered base tracks the solver's base
    (EEFs stay on the markers); with ``--base-loop open`` only the IK base twist is sent
    (the EEFs visibly lag under command shaping). The markers stay at the RAW waypoint pose.
    """
    dt = 1.0 / args.rate
    closed_loop = args.base_loop == "closed"
    home_left = ik.frame_pose(LEFT_EE_FRAME)
    home_right = ik.frame_pose(RIGHT_EE_FRAME)

    def make_schedule() -> Iterator[ScheduleStep]:
        """One full pass: the scripted waypoint tour, then the half-square+circle sweep."""
        if not args.skip_tour:
            yield from build_schedule(
                home_left, home_right, waypoints, args.travel, args.dwell, dt
            )
        if not args.skip_shape:
            yield from build_trajectory_schedule(
                home_left, home_right,
                half_square_circle_path(args.shape_side),
                args.shape_speed, dt, args.dwell,
            )

    t0 = time.perf_counter()
    last_print = 0.0
    last_warn = 0.0
    for rep in range(args.repeat):
        for step in make_schedule():
            now = time.perf_counter()
            t = now - t0
            if robot.viewer_closed:
                return

            result = ik.solve(step.left, step.right, dt)

            robot.left_arm.set_joint_pos(result.left_arm)
            robot.right_arm.set_joint_pos(result.right_arm)
            robot.torso.set_joint_pos(result.torso)
            robot.head.set_joint_pos(result.head)
            # Base command twist. CLOSED loop (--base-loop closed): base_tracking_twist is
            # the PD reference twist (feed-forward IK base twist + feedback on the base-pose
            # error) that steers the rendered base onto the solver's base, so the rendered
            # robot faithfully follows the whole-body IK and the EEFs stay on the raw-target
            # markers (whether wheel-driven physics or velocity-integrated kinematic). OPEN
            # loop (--base-loop open): just the IK base twist (feed-forward), so command
            # shaping (deadband/slew) and step inputs drift the rendered base off the
            # solver's base and the EEFs visibly lag the markers.
            if closed_loop:
                desired_twist = base_tracking_twist(
                    result.base_pose, result.base_twist, robot.base_pose)
            else:
                desired_twist = result.base_twist
            # Route EVERY base command through the controller: deadband -> slew -> e-stop
            # -> coordinated set_velocity(..., sequential_steering=False). hold momentarily
            # stops the base on a failed IK solve.
            base_ctrl.command(robot, desired_twist, hold=not result.success)

            robot.set_targets(step.left, step.right)  # markers at the RAW target
            robot.step(dt)
            robot.render()

            # Report the RENDERED EE error: the L_ee/R_ee links ride the base that was
            # actually driven, so this is the true (closed-loop) gap to the raw-target
            # markers -- not the solver's own result.left_ee_error, which is ~0 by design.
            eep = robot.ee_positions()
            err_l = float(np.linalg.norm(eep["L"] - step.left.translation))
            err_r = float(np.linalg.norm(eep["R"] - step.right.translation))
            base_xyz = robot.base_pose

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
                bx, by, byaw = base_xyz
                tag = "  [HELD]" if result.held else ""
                rep_tag = f" rep {rep + 1}/{args.repeat}" if args.repeat > 1 else ""
                print(
                    f"t={t:6.1f}s{rep_tag}  {step.label:14s} {step.phase:6s}  "
                    f"errL={err_l * 1000:5.1f}mm "
                    f"errR={err_r * 1000:5.1f}mm  "
                    f"base x={bx:+.3f} y={by:+.3f} yaw={np.degrees(byaw):+5.1f}deg  "
                    f"margin={result.stability_margin * 100:+4.1f}cm{tag}",
                    end="\r",
                )

            if step.waypoint_done:
                bx, by, byaw = base_xyz
                print(
                    f"\n[wbc_waypts] reached '{step.label}': "
                    f"errL={err_l * 1000:.1f}mm "
                    f"errR={err_r * 1000:.1f}mm  "
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
                        help="output video path (mp4).")
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
    parser.add_argument("--shape-side", type=float, default=0.8,
                        help="side length (m) of the half-square+circle trajectory traced "
                             "after the tour: a U of three sides closed by a semicircle of "
                             "radius shape-side/2 back to home (default: 0.8).")
    parser.add_argument("--shape-speed", type=float, default=0.25,
                        help="average speed (m/s) the EE marker is swept along the shape "
                             "trajectory; the base drives to follow it (default: 0.25).")
    parser.add_argument("--skip-tour", action="store_true",
                        help="skip the scripted waypoint tour; run only the half-square+"
                             "circle trajectory.")
    parser.add_argument("--skip-shape", action="store_true",
                        help="skip the half-square+circle trajectory; run only the tour.")
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
    parser.add_argument("--physics", action="store_true",
                        help="drive the base by ACTUATING THE WHEEL JOINTS under SAPIEN "
                             "physics (swerve steer+drive) instead of kinematically gliding "
                             "the root. EE tracking becomes approximate (wheel slip/lag).")
    parser.add_argument("--physics-hz", type=float, default=400.0,
                        help="physics substep rate in Hz for --physics (default: 400).")
    parser.add_argument("--wheel-friction", type=float, default=1.2,
                        help="wheel/ground friction coefficient for --physics (default: 1.2).")
    parser.add_argument("--base-deadband", type=float, default=0.01,
                        help="base-velocity deadband in m/s (ang = 2x); below this an axis "
                             "is zeroed (command jitter rejection; the closed loop tracks "
                             "through it).")
    parser.add_argument("--base-accel", type=float, default=2.0,
                        help="base-velocity slew limit in m/s^2 (ang = 2x): bounds the "
                             "base command's acceleration.")
    parser.add_argument("--base-loop", choices=("open", "closed"), default="closed",
                        help="base control loop: 'closed' (default) adds PD feedback so the "
                             "rendered base tracks the solver's base and the EEFs follow the "
                             "raw-target markers; 'open' feeds only the IK base twist, so "
                             "command shaping / step inputs let the rendered base drift and "
                             "the EEFs visibly lag the markers.")
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    if args.skip_tour and args.skip_shape:
        parser.error("--skip-tour and --skip-shape cannot both be set (nothing to run)")

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
    if not args.skip_tour:
        print("[wbc_waypts] waypoint plan (offsets from home EE pose, world axes "
              "x-fwd/y-left/z-up):")
        for i, wp in enumerate(waypoints):
            print(f"   {i + 1:2d}. {wp.label:14s} "
                  f"L=({wp.left[0]:+.2f},{wp.left[1]:+.2f},{wp.left[2]:+.2f}) "
                  f"R=({wp.right[0]:+.2f},{wp.right[1]:+.2f},{wp.right[2]:+.2f}) "
                  f"yaw={wp.yaw_deg:+.0f}deg")
    if not args.skip_shape:
        shape_path = half_square_circle_path(args.shape_side)
        shape_len = float(np.sum(np.linalg.norm(np.diff(shape_path, axis=0), axis=1)))
        print(f"[wbc_waypts] + half-square+circle sweep: U of 3x{args.shape_side:.2f}m "
              f"sides closed by a semicircle (r={args.shape_side / 2:.2f}m), "
              f"~{shape_len:.2f}m at {args.shape_speed:.2f}m/s avg "
              f"(~{shape_len / args.shape_speed:.0f}s).")
    per_wp = args.travel + args.dwell
    print(f"[wbc_waypts] ~{per_wp:.1f}s/waypoint -> "
          f"~{per_wp * len(waypoints) * args.repeat:.0f}s total tour. "
          f"Ctrl-C to stop and save.")

    base_mode = ("physics (wheels actuated: swerve steer+drive)" if args.physics
                 else "kinematic (root glide)")
    print(f"[wbc_waypts] base mode: {base_mode}")
    if args.physics:
        robot = SapienPhysicsRobot(
            urdf_path=ik.config.urdf_path,
            with_viewer=not args.no_viewer,
            record_video=args.output,
            video_fps=args.video_fps,
            video_size=(args.width, args.height),
            control_dt=1.0 / args.rate,
            physics_hz=args.physics_hz,
            wheel_friction=args.wheel_friction,
        )
    else:
        robot = SapienSimRobot(
            urdf_path=ik.config.urdf_path,
            with_viewer=not args.no_viewer,
            record_video=args.output,
            video_fps=args.video_fps,
            video_size=(args.width, args.height),
        )
    base_ctrl = BaseVelocityController(
        dt=1.0 / args.rate, max_lin_vel=BASE_VEL_LIMIT, max_ang_vel=BASE_VEL_LIMIT,
        deadband_lin=args.base_deadband, deadband_ang=2.0 * args.base_deadband,
        max_lin_accel=args.base_accel, max_ang_accel=2.0 * args.base_accel,
    )
    print(f"[wbc_waypts] base control: {args.base_loop}-loop, coordinated "
          f"set_velocity(sequential_steering=False) + "
          f"deadband({base_ctrl.deadband_lin}m/s,{base_ctrl.deadband_ang}rad/s) + slew("
          f"{base_ctrl.max_lin_accel}m/s^2,{base_ctrl.max_ang_accel}rad/s^2) + e-stop")
    try:
        run(ik, robot, args, waypoints, base_ctrl)
    except KeyboardInterrupt:
        pass
    finally:
        base_ctrl.estop()     # latch the e-stop, then zero the base before tearing down
        robot.chassis.stop()  # zero the base velocity before tearing down
        robot.close()  # releases the video writer
        print("\n[wbc_waypts] stopped.")


if __name__ == "__main__":
    main()
