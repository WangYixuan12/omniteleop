#!/usr/bin/env python3
"""Whole-body IK in SAPIEN driven by wbc_vr_leader's EEF targets, recorded to video.

Run the **whole-body VR leader** (``scripts/wbc_vr_leader.py``) and move the
controllers. This node:

1. subscribes to the published ``VRJointData`` (Zenoh topic ``vr/joints``);
2. reads the Cartesian ``L_ee`` / ``R_ee`` **target poses** the leader publishes
   (``VRJointData.left_ee_pose`` -- the exact targets it feeds to its arm IK, which
   track your controller's world motion: controller forward 10 cm -> target forward
   10 cm, independent of headset motion);
3. **interpolates** the low-rate command stream (the leader publishes at ~10 Hz)
   up to the ``--rate`` (100 Hz) IK ticks -- position lerped, orientation slerped
   over ``1/cmd-rate`` per command, like the reference's ``use_interpolation``
   (deps/rby1-wbc). The EE targets AND the head target ride the same segments, so
   every tracker sees a smooth glide, not 10 Hz staircase jumps;
4. runs **whole-body IK** to track the EE targets, coordinating the mobile base +
   torso + arms;
5. tracks the interpolated ``head_ee_pose`` (the calibrated headset pose) with the
   robot's own **stiff damped-LS head IK** -- ``VegaWholeBodyIK.solve_head``,
   vr_reader's ``head_mode: 'track'`` semantics (head_j2/j3 only) but solved
   against the **live whole-body configuration** with a full step per tick
   (``head_ik_gain`` >= ``--rate``), so the pan/tilt counter-rotates base yaw and
   torso lean within a tick and the camera stays on the operator's gaze (smoothing
   comes from the interpolation, like the arms). There is no WBC head task: the
   joint command is fed to ``ik.solve(head_joints=...)`` so the solver's model
   (collision spheres, CoM, FK) tracks the commanded head while its QP keeps the
   head DOFs pinned, and ``result.head`` echoes it to the robot;
6. shows it in a SAPIEN viewer and **records an mp4**.

IMPORTANT: this follower is driven by ``scripts/wbc_vr_leader.py``, **not** by
``omni-vr`` / ``src/omniteleop/leader/vr_reader.py``. Only ``wbc_vr_leader`` streams
the Cartesian ``left_ee_pose`` / ``right_ee_pose`` this node consumes; ``omni-vr``
publishes joint positions and never sets those fields, so running it here leaves the
robot stuck holding its nominal posture (status ``held(estop:...)``) forever.

While ``estop`` is set (the leader is in its ``static`` stage, before you grip-hold
to calibrate) or no EEF targets have been published yet, the robot holds its nominal
posture.

Run in the dexmate conda env (has sapien + pinocchio + pink + dexcomm)::

    /home/yixuan/miniforge3/envs/dexmate/bin/python scripts/wbc_vr_record.py \
        --output /tmp/wbc_vr.mp4

Add ``--no-viewer`` to record headlessly. The base is driven through
``BaseVelocityController`` (coordinated ``set_velocity(sequential_steering=False)`` +
deadband + slew + e-stop); the leader's e-stop (``static`` stage) zeros the base.
``--base-loop`` selects ``closed`` (default; PD feedback steers the rendered base onto the
solver's base so the EEFs follow the target frames) or ``open`` (only the IK base twist, so
the rendered base can drift off under command shaping / step inputs and the EEFs lag). In
open loop, the sparse VR target stream is low-passed before the slew limiter so staircase
updates do not lose most of their displacement to one-tick velocity bursts.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np
from dexcomm import Node
from dexcomm.codecs import DictDataCodec
from scipy.spatial.transform import Rotation

from omniteleop.common import get_config
from omniteleop.common.schemas import VRJointData
from omniteleop.follower.wbc_sapien_sim import (
    BASE_VEL_LIMIT,
    BaseVelocityController,
    SapienSimRobot,
    TwistLowPassFilter,
    base_tracking_twist,
)
from omniteleop.follower.whole_body_ik import (
    HEAD_FRAME,
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    VegaWholeBodyIK,
    WBCConfig,
)

SAFE_STABILITY_MARGIN = 0.05
DEFAULT_CLOSED_BASE_DEADBAND = 0
DEFAULT_CLOSED_BASE_ACCEL = 2.0
DEFAULT_OPEN_BASE_DEADBAND = 0
DEFAULT_OPEN_BASE_ACCEL = 1000.0
DEFAULT_OPEN_LOOP_TWIST_TAU = 0


class VRJointSubscriber:
    """Subscribes to the ``vr/joints`` stream and keeps the latest ``VRJointData``."""

    def __init__(self, namespace: str = "") -> None:
        self.node = Node(name="wbc_vr_record", namespace=namespace)
        self.topic = get_config().get_topic("vr_joints", "vr/joints")
        self._latest: Optional[VRJointData] = None
        self.sub = self.node.create_subscriber(
            self.topic, self._on_msg, decoder=DictDataCodec.decode
        )

    def _on_msg(self, data: dict) -> None:
        try:
            self._latest = VRJointData(**data)
        except (TypeError, ValueError):
            pass  # ignore malformed frames

    @property
    def latest(self) -> Optional[VRJointData]:
        """Most recently received ``VRJointData`` (or None before the first frame)."""
        return self._latest

    def close(self) -> None:
        """Close the Zenoh node (best-effort)."""
        try:
            self.node.close()
        except Exception:  # best-effort cleanup
            pass


def _pose_from_flat(flat: list[float]) -> np.ndarray:
    """Reconstruct a 4x4 pose matrix from a flattened (16-float) row-major matrix."""
    return np.asarray(flat, dtype=float).reshape(4, 4)


def _to_mat(pose) -> np.ndarray:
    """4x4 matrix from a pin.SE3 (its ``.homogeneous``) or a 4x4 array."""
    return np.asarray(pose.homogeneous if hasattr(pose, "homogeneous") else pose, dtype=float)


def vr_to_ee_targets(
    vr: Optional[VRJointData],
    left_prev: np.ndarray,
    right_prev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """L_ee/R_ee target poses (4x4) from the leader's published EEF poses.

    Uses the exact Cartesian targets the leader feeds to its arm IK (which track
    the controller's world motion), so there is no joint-space round-trip or FK
    convention mismatch. Holds the previous targets when the leader is not
    teleoperating (calibration stages publish empty EEF poses). The 4x4 matrices
    are accepted directly as targets by the whole-body IK solver.
    """
    if vr is None or not vr.left_ee_pose or not vr.right_ee_pose:
        return left_prev, right_prev
    return _pose_from_flat(vr.left_ee_pose), _pose_from_flat(vr.right_ee_pose)


def vr_to_head_target(
    vr: Optional[VRJointData], prev: np.ndarray
) -> np.ndarray:
    """Head-frame target pose (4x4) from the leader's ``head_ee_pose``.

    The calibrated headset pose, interpolated alongside the EE targets and tracked
    by ``ik.solve_head`` from the live whole-body configuration (head_mode 'track'
    semantics). A non-empty pose is adopted, an empty one holds the previous
    command (``prev``, seeded with the nominal head pose -- the head holds nominal
    until the leader first publishes). A non-empty but malformed pose is a
    protocol violation (wrong leader): raise.
    """
    if vr is None or not vr.head_ee_pose:
        return prev
    flat = np.asarray(vr.head_ee_pose, dtype=float)
    if flat.shape != (16,) or not np.all(np.isfinite(flat)):
        raise ValueError(
            f"VRJointData.head_ee_pose must be 16 finite floats (flattened 4x4), "
            f"got {vr.head_ee_pose!r}"
        )
    return flat.reshape(4, 4)


def _blend_pose(t0: np.ndarray, t1: np.ndarray, alpha: float) -> np.ndarray:
    """Geodesic pose blend: lerp the translation, slerp the rotation."""
    if alpha >= 1.0:
        return t1.copy()
    out = np.eye(4)
    out[:3, 3] = (1.0 - alpha) * t0[:3, 3] + alpha * t1[:3, 3]
    r0 = Rotation.from_matrix(t0[:3, :3])
    delta = (r0.inv() * Rotation.from_matrix(t1[:3, :3])).as_rotvec()
    out[:3, :3] = (r0 * Rotation.from_rotvec(alpha * delta)).as_matrix()
    return out


def _orientation_marker_pose(orientation_pose: np.ndarray, anchor_pose) -> np.ndarray:
    """Pose for visualizing only target orientation at an anchored world position."""
    out = np.eye(4)
    out[:3, :3] = np.asarray(orientation_pose, dtype=float)[:3, :3]
    out[:3, 3] = _to_mat(anchor_pose)[:3, 3]
    return out


class TargetInterpolator:
    """Per-tick lerp/slerp of the leader's low-rate pose commands.

    Generic over the number of pose streams -- here L_ee, R_ee, and the head frame
    all ride the same segments. The leader publishes commands at ``--cmd-rate``
    (default 10 Hz) while the IK runs at ``--rate`` (default 100 Hz). Each arriving
    command starts a new blend segment toward it, traversed in ``duration =
    1/cmd-rate``: ``alpha = min(1, elapsed/duration)``, position lerped,
    orientation slerped. This is the reference's interpolation (deps/rby1-wbc
    ``ee_targets.get_for_ik`` with ``use_interpolation: true``, ``duration =
    1/trajectory_frequency_hz``), with one refinement: the segment *start* is the
    pose currently being output (not the previous raw command), so the output stays
    continuous under command jitter. ``duration <= 0`` degrades to pass-through
    (zero-order hold, the old behavior).
    """

    def __init__(self, duration: float, *poses0: np.ndarray) -> None:
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError(f"duration must be finite and >= 0, got {duration}")
        if not poses0:
            raise ValueError("at least one initial pose is required")
        self.duration = float(duration)
        self._n = len(poses0)
        self.reset(*poses0)

    def reset(self, *poses: np.ndarray) -> None:
        """Snap to ``poses`` with no segment in flight."""
        if len(poses) != self._n:
            raise ValueError(f"expected {self._n} poses, got {len(poses)}")
        self._start = tuple(p.copy() for p in poses)
        self._end = tuple(p.copy() for p in poses)
        self._t0 = -np.inf  # alpha clamps to 1 -> output the end poses

    def push(self, *poses: np.ndarray, now: float) -> None:
        """Start a new segment from the current output toward a fresh command."""
        if len(poses) != self._n:
            raise ValueError(f"expected {self._n} poses, got {len(poses)}")
        self._start = self.at(now)
        self._end = tuple(p.copy() for p in poses)
        self._t0 = now

    def at(self, now: float) -> tuple[np.ndarray, ...]:
        """Blended target poses at time ``now``."""
        if self.duration <= 0.0:
            return tuple(p.copy() for p in self._end)
        alpha = min(1.0, max(0.0, (now - self._t0) / self.duration))
        return tuple(
            _blend_pose(s, e, alpha)
            for s, e in zip(self._start, self._end, strict=True)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", default="/tmp/wbc_vr.mp4",
                        help="output video path (mp4) (default: /tmp/wbc_vr.mp4).")
    parser.add_argument("--rate", type=float, default=100.0,
                        help="control/record rate in Hz (default: 100).")
    parser.add_argument("--cmd-rate", type=float, default=10.0,
                        help="leader command rate in Hz (default: 10, matching "
                             "wbc_vr_leader's default). Targets are lerp/slerp-"
                             "interpolated over 1/cmd-rate between commands so the "
                             "--rate IK sees a smooth glide (the reference's "
                             "use_interpolation / trajectory_frequency_hz). "
                             "0 disables interpolation (zero-order hold).")
    parser.add_argument("--width", type=int, default=960, help="video width (default: 960).")
    parser.add_argument("--height", type=int, default=540, help="video height (default: 540).")
    parser.add_argument("--video-fps", type=float, default=15.0,
                        help="recorded video fps (default: 15; frames are decimated "
                             "from the higher --rate control loop to real-time).")
    parser.add_argument("--no-viewer", action="store_true",
                        help="record headlessly (no SAPIEN window).")
    parser.add_argument("--namespace", default="",
                        help="Zenoh namespace (must match the leader; default empty).")
    parser.add_argument("--safe-margin", type=float, default=SAFE_STABILITY_MARGIN,
                        help="tip-over warning threshold in meters "
                             f"(default {SAFE_STABILITY_MARGIN}).")
    parser.add_argument("--urdf", default=None, help="override the Vega URDF path.")
    parser.add_argument("--base-loop", choices=("open", "closed"), default="closed",
                        help="base control loop: 'closed' (default) adds PD feedback so the "
                             "rendered base tracks the solver's base and the EEFs follow the "
                             "target frames; 'open' feeds only the IK base twist, so the "
                             "rendered base drifts under command shaping / step inputs and "
                             "the EEFs visibly lag.")
    parser.add_argument("--base-deadband", type=float, default=None,
                        help="base-velocity deadband in m/s (ang = 2x). Defaults to "
                             f"{DEFAULT_CLOSED_BASE_DEADBAND:g} in closed loop and "
                             f"{DEFAULT_OPEN_BASE_DEADBAND:g} in open loop.")
    parser.add_argument("--base-accel", type=float, default=None,
                        help="base-velocity slew limit in m/s^2 (ang = 2x). Defaults to "
                             f"{DEFAULT_CLOSED_BASE_ACCEL:g} in closed loop and "
                             f"{DEFAULT_OPEN_BASE_ACCEL:g} in open loop.")
    parser.add_argument("--open-loop-twist-tau", type=float,
                        default=DEFAULT_OPEN_LOOP_TWIST_TAU,
                        help="EMA time constant in seconds for open-loop IK base twist "
                             f"before deadband/slew (0 disables; default "
                             f"{DEFAULT_OPEN_LOOP_TWIST_TAU:g}). Ignored in closed loop.")
    args = parser.parse_args()

    if args.base_deadband is not None and args.base_deadband < 0.0:
        parser.error("--base-deadband must be >= 0")
    if args.base_accel is not None and args.base_accel <= 0.0:
        parser.error("--base-accel must be > 0")
    if args.open_loop_twist_tau < 0.0:
        parser.error("--open-loop-twist-tau must be >= 0")

    cfg = WBCConfig()
    if args.urdf:
        cfg.urdf_path = args.urdf
    cfg.com_safety_margin = args.safe_margin
    ik = VegaWholeBodyIK(cfg)
    left_frame, right_frame = LEFT_EE_FRAME, RIGHT_EE_FRAME
    print(f"[wbc_vr] safety: collision_avoidance={ik.collision_enabled}  "
          f"com_safety={cfg.enable_com_safety} (margin {cfg.com_safety_margin * 100:.0f}cm)")
    ik.reset()
    left0 = _to_mat(ik.frame_pose(left_frame))
    right0 = _to_mat(ik.frame_pose(right_frame))
    head0 = _to_mat(ik.frame_pose(HEAD_FRAME))
    print(f"[wbc_vr] model: nq={ik.model.nq} nv={ik.model.nv}")

    robot = SapienSimRobot(
        urdf_path=ik.config.urdf_path,
        with_viewer=not args.no_viewer,
        record_video=args.output,
        video_fps=args.video_fps,
        video_size=(args.width, args.height),
    )
    sub = VRJointSubscriber(args.namespace)
    print(f"[wbc_vr] subscribed to '{sub.topic}'. Run the whole-body VR leader "
          "(scripts/wbc_vr_leader.py -- NOT omni-vr) and grip-hold to calibrate. "
          "Ctrl-C to stop and save.")

    if args.rate <= 0:
        raise ValueError(f"--rate must be > 0, got {args.rate}")
    if args.cmd_rate < 0 or not np.isfinite(args.cmd_rate):
        raise ValueError(f"--cmd-rate must be finite and >= 0, got {args.cmd_rate}")
    dt = 1.0 / args.rate
    closed_loop = args.base_loop == "closed"
    base_deadband = args.base_deadband
    if base_deadband is None:
        base_deadband = (
            DEFAULT_CLOSED_BASE_DEADBAND if closed_loop else DEFAULT_OPEN_BASE_DEADBAND
        )
    base_accel = args.base_accel
    if base_accel is None:
        base_accel = DEFAULT_CLOSED_BASE_ACCEL if closed_loop else DEFAULT_OPEN_BASE_ACCEL
    base_ctrl = BaseVelocityController(
        dt=dt, max_lin_vel=BASE_VEL_LIMIT, max_ang_vel=BASE_VEL_LIMIT,
        deadband_lin=base_deadband, deadband_ang=2.0 * base_deadband,
        max_lin_accel=base_accel, max_ang_accel=2.0 * base_accel,
    )
    open_loop_filter = TwistLowPassFilter(dt=dt, tau=args.open_loop_twist_tau)
    open_loop_filter_msg = (
        f" + open-loop low-pass(tau={args.open_loop_twist_tau:g}s)"
        if not closed_loop and args.open_loop_twist_tau > 0.0 else ""
    )
    print(f"[wbc_vr] base control: {args.base_loop}-loop, coordinated "
          "set_velocity(sequential_steering=False) + "
          f"deadband({base_ctrl.deadband_lin}m/s,{base_ctrl.deadband_ang}rad/s) + "
          f"slew({base_ctrl.max_lin_accel}m/s^2,{base_ctrl.max_ang_accel}rad/s^2) + "
          f"e-stop{open_loop_filter_msg}")
    # Raw leader commands (4x4), held across frames and updated whenever the leader
    # publishes new ones (so the robot freezes, not snaps, when teleop pauses). The
    # IK does not consume them directly: each NEW command is pushed into the
    # interpolator, which the IK reads every tick.
    left_cmd = left0.copy()
    right_cmd = right0.copy()
    # Head-frame command (the leader's head_ee_pose, the calibrated headset pose),
    # seeded at the nominal head FK so the head holds nominal until first published.
    # It rides the same interpolation segments as the EE targets, and ik.solve_head
    # tracks the blended target STIFFLY each tick from the live configuration
    # (head_ik_gain=100 -> full step/tick): smoothing comes from the interpolation,
    # so the base-yaw/torso-lean compensation is not low-passed.
    head_cmd = head0.copy()
    last_cmd_ns = -1  # leader timestamp of the last command pushed into a segment
    interp = TargetInterpolator(
        1.0 / args.cmd_rate if args.cmd_rate > 0 else 0.0,
        left0, right0, head0,
    )
    left_target = left0.copy()
    right_target = right0.copy()
    t0 = time.perf_counter()
    last_print = 0.0
    last_warn = 0.0
    seen_data = False
    # estop is True while the leader is in its `static` stage and flips False the
    # instant it (re)calibrates on a grip-hold. We track its previous value so the
    # static->teleop edge can reset the robot to origin (see below). Start True: the
    # leader defaults to static, and the first calibration edge then resets harmlessly
    # (the robot already starts at origin/nominal).
    prev_estop = True
    ever_tracked = False        # have we ever received a non-empty left_ee_pose?
    warned_wrong_leader = False  # one-time "wrong leader" diagnostic latch
    try:
        while True:
            now = time.perf_counter()
            t = now - t0
            if robot.viewer_closed:
                break

            vr = sub.latest
            seen_data = seen_data or vr is not None
            tracking = vr is not None and bool(vr.left_ee_pose)
            ever_tracked = ever_tracked or tracking

            # Diagnose being driven by the wrong leader. This follower needs
            # wbc_vr_leader.py, whose stages are {static, teleop} and which streams
            # left_ee_pose/right_ee_pose. omni-vr/vr_reader.py instead uses stages
            # {head, whole_body_alignment, whole_body, resetting} and NEVER publishes
            # those fields, so the robot would sit in held(estop) forever. If we see a
            # stage outside wbc_vr_leader's set and have never gotten an EEF target,
            # say so loudly (once) instead of failing silently.
            if (vr is not None and not ever_tracked and not warned_wrong_leader
                    and vr.calib_stage not in ("static", "teleop")):
                warned_wrong_leader = True
                print(f"\n[wbc_vr] !! leader publishes no EEF targets (calib_stage="
                      f"'{vr.calib_stage}'). wbc_vr_record.py must be driven by "
                      "scripts/wbc_vr_leader.py, NOT omni-vr/vr_reader.py.")

            # On a fresh calibration (the leader's static->teleop edge), it re-anchors
            # its EEF targets to the nominal (origin) base frame. Reset the robot to
            # origin/nominal so the base drift accumulated during the previous teleop
            # session is cleared and the new session starts from home -- keeping the
            # follower's base frame consistent with the leader's re-anchoring.
            estop = bool(vr.estop) if vr is not None else True
            if prev_estop and not estop:
                ik.reset()
                robot.reset_base()
                open_loop_filter.reset()
                left_cmd, right_cmd = left0.copy(), right0.copy()
                head_cmd = head0.copy()  # ik.reset() put the head back at nominal
                interp.reset(left0, right0, head0)
                print("\n[wbc_vr] recalibrated -> reset robot to origin.")
            prev_estop = estop

            # Hold/refresh the raw leader commands; a NEW command (fresh leader
            # timestamp with EEF poses) starts an interpolation segment, so the IK
            # tracks a smooth 1/cmd-rate glide instead of low-rate staircase jumps.
            left_cmd, right_cmd = vr_to_ee_targets(vr, left_cmd, right_cmd)
            head_cmd = vr_to_head_target(vr, head_cmd)
            if (vr is not None and vr.timestamp_ns != last_cmd_ns
                    and vr.left_ee_pose and vr.right_ee_pose):
                last_cmd_ns = vr.timestamp_ns
                interp.push(left_cmd, right_cmd, head_cmd, now=now)
            left_target, right_target, head_target = interp.at(now)
            # Stiff head IK from the LIVE configuration (compensates base yaw /
            # torso lean within a tick), then the whole-body solve with the head
            # pinned at that command.
            head_joints = ik.solve_head(head_target, dt)

            result = ik.solve(left_target, right_target, dt, head_joints=head_joints)

            robot.left_arm.set_joint_pos(result.left_arm)
            robot.right_arm.set_joint_pos(result.right_arm)
            robot.torso.set_joint_pos(result.torso)
            robot.head.set_joint_pos(result.head)
            # Base command twist. CLOSED loop (--base-loop closed): base_tracking_twist
            # adds PD feedback that steers the rendered base onto the solver's base, so the
            # EEFs FOLLOW the target frames (the solver tracks the targets to ~0 in its own
            # base frame, but the deadband/slew-shaped command would otherwise drift the
            # rendered base off it). OPEN loop (--base-loop open): just the IK base twist,
            # so command shaping / step inputs let the rendered base drift and the EEFs lag.
            # The result is sent through the official set_velocity(..., sequential_steering=
            # False) path; hold zeros the base while the leader is e-stopped or solve fails.
            hold_base = estop or not result.success
            if closed_loop:
                desired_twist = base_tracking_twist(
                    result.base_pose, result.base_twist, robot.base_pose)
            else:
                desired_twist = result.base_twist
                if hold_base:
                    open_loop_filter.reset()
                else:
                    desired_twist = open_loop_filter.filter(desired_twist)
            base_ctrl.command(robot, desired_twist, hold=hold_base)

            head_target_marker = _orientation_marker_pose(
                head_target, ik.frame_pose(HEAD_FRAME)
            )
            robot.set_targets(left_target, right_target, head_target_marker)
            robot.step(dt)
            robot.render()

            # Safety: the solver already holds the robot (joints, and base on tip-over)
            # and zeroes the held DOFs; here we just surface the reason to the operator.
            if result.held and t - last_warn >= 0.3:
                last_warn = t
                print(f"\n[wbc_vr] !! SAFETY HOLD: {result.safety_status}")
            elif (not result.held and result.safety_status != "ok"
                  and t - last_warn >= 1.0):
                last_warn = t
                print(f"\n[wbc_vr] ! {result.safety_status}")

            if t - last_print >= 0.5:
                last_print = t
                stage = vr.calib_stage if vr is not None else "?"
                if result.held:
                    state = "HELD(safety)"
                elif tracking:
                    state = "tracking"
                elif not seen_data:
                    state = "waiting-for-VR"
                else:
                    # Show the leader's stage so a stuck robot is self-explanatory:
                    # "static" = grip-hold to calibrate; anything else = wrong leader.
                    state = f"held(estop:{stage})"
                # Report the TRUE rendered EE error: the L_ee/R_ee links ride the
                # velocity-driven base, so this is the real gap to the drawn target
                # frames (the solver's own result.left_ee_error is ~0 by construction).
                eep = robot.ee_positions()
                err_l = float(np.linalg.norm(eep["L"] - left_target[:3, 3]))
                err_r = float(np.linalg.norm(eep["R"] - right_target[:3, 3]))
                bx, by, byaw = robot.base_pose
                print(
                    f"t={t:6.1f}s  {state:18s}  errL={err_l * 1000:5.1f}mm  "
                    f"errR={err_r * 1000:5.1f}mm  "
                    f"base x={bx:+.3f} y={by:+.3f} yaw={np.degrees(byaw):+5.1f}deg  "
                    f"margin={result.stability_margin * 100:+4.1f}cm  "
                    f"self={result.min_self_distance * 100:+4.1f}cm",
                    end="\r",
                )

            sleep = dt - (time.perf_counter() - now)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        base_ctrl.estop()
        if hasattr(robot.chassis, "stop"):
            robot.chassis.stop()
        robot.close()   # releases the video writer
        sub.close()
        print("\n[wbc_vr] stopped.")


if __name__ == "__main__":
    main()
