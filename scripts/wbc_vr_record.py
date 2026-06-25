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
        --hdf5 /tmp/demo.hdf5

This writes one HDF5 file: the replayable ``VRJointData`` stream at the root and
simulation/debug datasets under ``/debug``. Replay that same stream through SAPIEN before
running it on the real robot::

    /home/yixuan/miniforge3/envs/dexmate/bin/python scripts/wbc_vr_record.py \
        --replay /tmp/demo.hdf5 --speed 0.25 --output /tmp/demo_replay.mp4

Add ``--no-viewer`` to record headlessly. ``--physics`` actuates the swerve wheel joints
under SAPIEN physics (``wbc_swerve`` steer+drive; the base pose *emerges* from wheel-ground
contact, as on hardware) instead of kinematically gliding the root -- EE tracking then
becomes approximate (wheel slip/lag), the realistic trade-off. See
``omniteleop.follower.wbc_sapien_physics.SapienPhysicsRobot``.

The base twist is shaped (deadband -> velocity clamp -> slew) and issued through the
official ``set_velocity(..., sequential_steering=False)`` path; the leader's e-stop
(``static`` stage) zeros the base. ``--base-loop`` selects how the commanded twist is
formed:

  * ``closed`` (default) -- the **same closed-loop base controller the real robot uses** in
    ``scripts/drive_box_record.py`` (``omniteleop.follower.base_closed_loop``): the solver's
    base twist as feed-forward plus PD feedback on the (solver base pose - measured base
    pose) error, so the wheel-driven/rendered base tracks the solver's reference and the
    EEFs follow the target frames. The reference here is the solver's per-tick
    ``result.base_pose`` (no leg/time-parameterized trajectory), so only ``pd_twist`` +
    ``shape_twist`` are used -- not the leg-completion / freshness machinery.
  * ``open`` -- only the IK base twist (feed-forward), so the measured base drifts off under
    command shaping / wheel dynamics and the EEFs visibly lag the target frames.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np

from omniteleop.common.schemas import VRJointData
from omniteleop.follower.wbc_sapien_sim import (
    BASE_VEL_LIMIT,
    SapienSimRobot,
)
from omniteleop.follower.whole_body_ik import (
    HEAD_FRAME,
    HEAD_JOINTS,
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    VegaWholeBodyIK,
    WBCConfig,
)
from omniteleop.wbc_record import (
    DEFAULT_REPLAY_GAP_LIMIT_MULTIPLE,
    DEFAULT_WBC_VR_OUTPUT,
    ReplaySource,
    StreamRecorder,
    TeleopStartGate,
    resolve_record_paths,
)
from omniteleop.wbc_stream import (
    HeadTargetLowPassFilter,
    HeadTargetPlanarDeadbandFilter,
    TargetInterpolator,
    _to_mat,
    deadband_planar_twist_by_reference,
    vr_to_ee_targets,
    vr_to_head_target,
)
from omniteleop.wbc_teleop import VRJointSubscriber, VRTeleopConfig

# Shared VR-follower control-loop tunables, loaded from the vr_teleop: block of
# follower/wbik.yaml so this SAPIEN sim follower and the real-robot follower
# (scripts/wbc_vr_robot.py) shape the head/base command stream identically -- a take
# recorded here must replay the same way on hardware. See omniteleop.wbc_teleop.
_VR_TELEOP = VRTeleopConfig.from_yaml()
DEFAULT_HEAD_LPF_TAU = _VR_TELEOP.head_lpf_tau
DEFAULT_HEAD_PLANAR_POS_DEADBAND = _VR_TELEOP.head_planar_pos_deadband
DEFAULT_HEAD_PLANAR_YAW_DEADBAND = _VR_TELEOP.head_planar_yaw_deadband
DEFAULT_BASE_POST_LINEAR_DEADBAND = _VR_TELEOP.base_post_linear_deadband
DEFAULT_BASE_POST_ANGULAR_DEADBAND = _VR_TELEOP.base_post_angular_deadband
# Closed-loop PD gains. Shared with base_closed_loop.BaseClosedLoopConfig (the same
# controller scripts/drive_box_record.py runs on the real robot); raise for snappier base
# tracking of fast VR motion. base_closed_loop itself is imported lazily inside main() so
# the module-load interpolation test can stub the heavy follower package.
DEFAULT_BASE_KP_XY = _VR_TELEOP.base_kp_xy
DEFAULT_BASE_KP_YAW = _VR_TELEOP.base_kp_yaw
DEFAULT_SPEED = _VR_TELEOP.replay_speed

# Base-twist shaping defaults for base_closed_loop.shape_twist (the angular limit is set to
# 2x the linear one below). The CLOSED-loop deadband + slew are the shared vr_teleop values
# (so the sim follower's closed loop matches the real robot, and a take replays the same);
# only the sim-follower-specific OPEN-loop variants (tighter quiet band, no slew limit) stay
# local -- the real robot has no open-loop mode.
DEFAULT_CLOSED_BASE_DEADBAND = _VR_TELEOP.base_deadband
DEFAULT_CLOSED_BASE_ACCEL = _VR_TELEOP.base_accel
DEFAULT_OPEN_BASE_DEADBAND = 0.01
DEFAULT_OPEN_BASE_ACCEL = 1000.0


def _orientation_marker_pose(orientation_pose: np.ndarray, anchor_pose) -> np.ndarray:
    """Pose for visualizing only target orientation at an anchored world position."""
    out = np.eye(4)
    out[:3, :3] = np.asarray(orientation_pose, dtype=float)[:3, :3]
    out[:3, 3] = _to_mat(anchor_pose)[:3, 3]
    return out


def _pin_head_mode_ik_joints(ik: VegaWholeBodyIK) -> None:
    """Pin head_j1/head_j2 in head_mode "ik", leaving head_j3 for camera tilt."""
    pin_names = ("head_j1", "head_j2")
    idx_by_name = dict(zip(HEAD_JOINTS, ik._head_idx_v, strict=True))  # noqa: SLF001
    rows = np.zeros((len(pin_names), ik.model.nv))
    for row, name in enumerate(pin_names):
        rows[row, idx_by_name[name]] = 1.0
    ik._head_j1_pin_A = rows  # noqa: SLF001


def _record_stream_frame_after_calibration(
    vr: VRJointData,
    recorder: StreamRecorder,
    gate: TeleopStartGate,
) -> bool:
    """Append ``vr`` once recording has started; return True on leader exit."""
    if gate.update(vr):
        recorder.append(vr)
    return bool(vr.exit_requested)


class DebugLogger:
    """Accumulates per-tick state and writes it under ``/debug`` in an HDF5 file.

    A no-op when ``path`` is None (``--no-hdf5``). Otherwise every :meth:`append`
    stores one control tick and :meth:`write` stacks the rows into fixed-shape
    arrays so the whole session replays offline. The captured columns reconstruct,
    per tick, what the leader commanded, the interpolated target the IK tracked, the
    whole-body solver result, the shaped base twist actually sent to the chassis, and
    the rendered (measured) base/EE state -- i.e. every place the EEF can diverge from
    the target (interpolation, solver hold, base shaping, wheel dynamics).
    """

    def __init__(self, path: Optional[str]) -> None:
        self.path = path
        self._rows: Optional[list[dict]] = [] if path else None

    @property
    def enabled(self) -> bool:
        """Whether this logger is collecting rows."""
        return self._rows is not None

    def __len__(self) -> int:
        return len(self._rows) if self._rows is not None else 0

    def append(self, **fields) -> None:
        """Record one control tick (ignored when disabled)."""
        if self._rows is not None:
            self._rows.append(fields)

    def write(self, meta: dict) -> None:
        """Stack the recorded ticks into the ``/debug`` group at ``self.path``."""
        if not self._rows:
            return
        import h5py  # noqa: PLC0415 -- optional dep, only needed at teardown

        rows = self._rows
        col = lambda k: np.asarray([r[k] for r in rows])

        def scol(k: str, width: int) -> np.ndarray:
            return np.array([str(r[k]).encode()[:width] for r in rows], dtype=f"S{width}")

        def ds(group, name: str, data, compress: bool = False) -> None:
            if compress:
                group.create_dataset(name, data=data, compression="gzip", compression_opts=4)
            else:
                group.create_dataset(name, data=data)

        with h5py.File(self.path, "a") as f:
            if "debug" in f:
                del f["debug"]
            root = f.create_group("debug")
            root.attrs.update(meta)
            root.attrs["n_frames"] = len(rows)

            tg = root.create_group("time")
            ds(tg, "t_wall_s", col("t"))
            ds(tg, "unix_ns", col("unix_ns"))
            ds(tg, "cmd_ns", col("cmd_ns"))
            tg.attrs["note"] = ("cmd_ns: leader VRJointData.timestamp_ns of the last "
                                "command pushed into the interpolator (-1 before first)")

            lg = root.create_group("leader")
            ds(lg, "calib_stage", scol("calib_stage", 16))
            ds(lg, "estop", col("estop"))
            ds(lg, "tracking", col("tracking"))
            ds(lg, "left_cmd_pose", col("left_cmd"), compress=True)
            ds(lg, "right_cmd_pose", col("right_cmd"), compress=True)
            ds(lg, "head_cmd_pose", col("head_cmd"), compress=True)
            lg.attrs["note"] = ("raw leader command poses (4x4 world-frame), held across "
                                "ticks between leader publishes")

            tt = root.create_group("target")
            ds(tt, "left_pose", col("left_target"), compress=True)
            ds(tt, "right_pose", col("right_target"), compress=True)
            ds(tt, "head_pose", col("head_target"), compress=True)
            ds(tt, "head_pose_pre_lpf", col("head_target_pre_lpf"), compress=True)
            tt.attrs["note"] = ("targets actually fed to the IK each tick; head_pose is "
                                "after --head-lpf-tau filtering and planar deadbanding, and "
                                "head_pose_pre_lpf is the interpolated head BEFORE the LPF; "
                                "L/R are interpolated")

            ig = root.create_group("ik")
            ds(ig, "q", col("q"), compress=True)
            ds(ig, "velocity", col("velocity"), compress=True)
            ds(ig, "base_pose", col("ik_base_pose"))
            ds(ig, "base_twist", col("ik_base_twist"))
            ds(ig, "torso", col("torso"))
            ds(ig, "left_arm", col("left_arm"))
            ds(ig, "right_arm", col("right_arm"))
            ds(ig, "head", col("head"))
            ds(ig, "left_ee_error", col("left_ee_error"))
            ds(ig, "right_ee_error", col("right_ee_error"))
            ds(ig, "success", col("success"))
            ds(ig, "held", col("held"))
            ds(ig, "stability_margin", col("stability_margin"))
            ds(ig, "min_self_distance", col("min_self_distance"))
            ds(ig, "safety_status", scol("safety_status", 96))
            ig.attrs["note"] = ("VegaWholeBodyIK.solve result: q/velocity are the full "
                                "configuration; *_ee_error are the solver's own residuals "
                                "(~0 by construction -- see rendered/* for the true gap)")

            bg = root.create_group("base_control")
            ds(bg, "raw_twist", col("raw_twist"))
            ds(bg, "pd_twist_out", col("pd_twist_out"))
            ds(bg, "pd_err", col("pd_err"))
            ds(bg, "shaped_twist", col("shaped_twist"))
            ds(bg, "hold_base", col("hold_base"))
            ds(bg, "base_settled", col("base_settled"))
            bg.attrs["note"] = ("pd_twist_out: PD+feed-forward twist before command shaping; "
                                "raw_twist: what feeds shape_twist; pd_err: "
                                "solver-minus-measured base-pose error (closed loop, else 0); "
                                "shaped_twist: post deadband->clamp->slew->post deadband, "
                                "exactly what set_velocity received; base_settled is retained "
                                "for schema compatibility and is always false")

            rg = root.create_group("rendered")
            ds(rg, "base_pose", col("meas_base_pose"))
            ds(rg, "ee_L_pos", col("ee_L_pos"))
            ds(rg, "ee_R_pos", col("ee_R_pos"))
            ds(rg, "ee_L_err", col("rendered_err_L"))
            ds(rg, "ee_R_err", col("rendered_err_R"))
            rg.attrs["note"] = ("measured/rendered sim state: integrated base pose and L/R "
                                "EE link world positions, with their distance to the drawn "
                                "target frames (the TRUE tracking error)")


def main() -> None:
    # Lazy import (not at module top level): importing this non-stubbed submodule forces
    # the real omniteleop.follower package __init__ to load, which the module-load
    # interpolation test deliberately avoids by stubbing only specific submodules.
    from omniteleop.follower import base_closed_loop as base_cl  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--replay", metavar="FILE",
                        help="replay a stream HDF5 through SAPIEN "
                             "instead of subscribing to the live leader.")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                        help=f"replay speed multiplier (default {DEFAULT_SPEED:g}; "
                             "<1 = slower).")
    parser.add_argument("--output", default=None,
                        help="output video path (mp4). Default: --hdf5 with .mp4 "
                             f"extension, or {DEFAULT_WBC_VR_OUTPUT} when --hdf5 is omitted.")
    parser.add_argument("--rate", type=float, default=100.0,
                        help="control/record rate in Hz (default: 100).")
    parser.add_argument("--cmd-rate", type=float, default=10.0,
                        help="leader command rate in Hz (default: 10, matching "
                             "wbc_vr_leader's default). Targets are lerp/slerp-"
                             "interpolated over 1/cmd-rate between commands so the "
                             "--rate IK sees a smooth glide (the reference's "
                             "use_interpolation / trajectory_frequency_hz). "
                             "0 disables interpolation (zero-order hold). Replay rejects "
                             f"recorded command gaps above "
                             f"{DEFAULT_REPLAY_GAP_LIMIT_MULTIPLE:g}x this nominal glide.")
    parser.add_argument("--width", type=int, default=960, help="video width (default: 960).")
    parser.add_argument("--height", type=int, default=540, help="video height (default: 540).")
    parser.add_argument("--video-fps", type=float, default=15.0,
                        help="recorded video fps (default: 15; frames are decimated "
                             "from the higher --rate control loop to real-time).")
    parser.add_argument("--no-viewer", action="store_true",
                        help="record headlessly (no SAPIEN window).")
    parser.add_argument("--namespace", default="",
                        help="Zenoh namespace (must match the leader; default empty).")
    parser.add_argument("--physics", action="store_true",
                        help="drive the base by ACTUATING THE WHEEL JOINTS under SAPIEN "
                             "physics (wbc_swerve steer+drive) instead of kinematically "
                             "gliding the root. EE tracking becomes approximate (wheel "
                             "slip/lag); pair with --base-loop closed to track the solver.")
    parser.add_argument("--physics-hz", type=float, default=400.0,
                        help="physics substep rate in Hz for --physics (default: 400).")
    parser.add_argument("--wheel-friction", type=float, default=1.2,
                        help="wheel/ground friction coefficient for --physics (default: 1.2).")
    parser.add_argument("--base-loop", choices=("open", "closed"), default="closed",
                        help="base control loop: 'closed' (default) is the SAME PD controller "
                             "the real robot uses (base_closed_loop): feed-forward IK base "
                             "twist + PD feedback on the (solver base - measured base) error, "
                             "so the base tracks the solver and the EEFs follow the target "
                             "frames; 'open' feeds only the IK base twist, so the measured "
                             "base drifts under shaping / wheel dynamics and the EEFs lag.")
    parser.add_argument("--base-deadband", type=float, default=None,
                        help="base-velocity deadband in m/s (ang = 2x). Defaults to "
                             f"{DEFAULT_CLOSED_BASE_DEADBAND:g} in closed loop and "
                             f"{DEFAULT_OPEN_BASE_DEADBAND:g} in open loop.")
    parser.add_argument("--base-accel", type=float, default=None,
                        help="base-velocity slew limit in m/s^2 (ang = 2x). Defaults to "
                             f"{DEFAULT_CLOSED_BASE_ACCEL:g} in closed loop and "
                             f"{DEFAULT_OPEN_BASE_ACCEL:g} in open loop.")
    parser.add_argument("--base-kp-xy", type=float, default=DEFAULT_BASE_KP_XY,
                        help="closed-loop proportional gain on planar base-pose error "
                             f"(1/s; default {DEFAULT_BASE_KP_XY:g}, the real-robot value). "
                             "Ignored in open loop.")
    parser.add_argument("--base-kp-yaw", type=float, default=DEFAULT_BASE_KP_YAW,
                        help="closed-loop proportional gain on base-yaw error "
                             f"(1/s; default {DEFAULT_BASE_KP_YAW:g}, the real-robot value). "
                             "Ignored in open loop.")
    parser.add_argument("--head-lpf-tau", type=float, default=DEFAULT_HEAD_LPF_TAU,
                        help="first-order low-pass time constant in seconds for the "
                             "interpolated head target before IK. 0 disables. "
                             f"Default {DEFAULT_HEAD_LPF_TAU:g}s suppresses headset "
                             "dither that otherwise resolves into base yaw in "
                             "head_mode='ik'.")
    parser.add_argument("--head-planar-pos-deadband", type=float,
                        default=DEFAULT_HEAD_PLANAR_POS_DEADBAND,
                        help="radial x/y deadband in meters for the planar head target "
                             "before IK. This preserves arbitrary movement direction and "
                             "only holds small headset translation at nominal. "
                             f"Default {DEFAULT_HEAD_PLANAR_POS_DEADBAND:g}.")
    parser.add_argument("--head-planar-yaw-deadband", type=float,
                        default=DEFAULT_HEAD_PLANAR_YAW_DEADBAND,
                        help="heading-yaw deadband in radians for the head target before "
                             "IK. Small headset yaw drift is held at nominal; larger turns "
                             "are preserved. "
                             f"Default {DEFAULT_HEAD_PLANAR_YAW_DEADBAND:g}.")
    parser.add_argument("--base-post-linear-deadband", type=float,
                        default=DEFAULT_BASE_POST_LINEAR_DEADBAND,
                        help="per-axis vx/vy deadband in m/s applied after base "
                             "deadband/clamp/slew using the same threshold for both "
                             "linear axes and the pre-slew raw command as the activity "
                             "reference. "
                             f"Default {DEFAULT_BASE_POST_LINEAR_DEADBAND:g}.")
    parser.add_argument("--base-post-angular-deadband", type=float,
                        default=DEFAULT_BASE_POST_ANGULAR_DEADBAND,
                        help="per-axis wz deadband in rad/s applied after base "
                             "deadband/clamp/slew using the pre-slew raw command as the "
                             "activity reference. "
                             f"Default {DEFAULT_BASE_POST_ANGULAR_DEADBAND:g}.")
    parser.add_argument("--hdf5", default=None,
                        help="path for the unified .hdf5 log. In live mode, the replayable "
                             "VRJointData stream is stored at the root and per-tick SAPIEN/"
                             "IK debug data is stored under /debug. Default: the --output "
                             "path with the extension replaced by .hdf5 (and --output "
                             "defaults to this path with .mp4 when --output is omitted).")
    parser.add_argument("--no-hdf5", action="store_true",
                        help="disable the post-run .hdf5 stream/debug log.")
    args = parser.parse_args()
    args.output, hdf5_path = resolve_record_paths(
        output=args.output, hdf5=args.hdf5, no_hdf5=args.no_hdf5,
    )

    if args.base_deadband is not None and args.base_deadband < 0.0:
        parser.error("--base-deadband must be >= 0")
    if args.base_accel is not None and args.base_accel <= 0.0:
        parser.error("--base-accel must be > 0")
    if args.base_kp_xy < 0.0 or args.base_kp_yaw < 0.0:
        parser.error("--base-kp-xy and --base-kp-yaw must be >= 0")
    if not np.isfinite(args.head_lpf_tau) or args.head_lpf_tau < 0.0:
        parser.error("--head-lpf-tau must be finite and >= 0")
    for flag, val in (
        ("--head-planar-pos-deadband", args.head_planar_pos_deadband),
        ("--head-planar-yaw-deadband", args.head_planar_yaw_deadband),
        ("--base-post-linear-deadband", args.base_post_linear_deadband),
        ("--base-post-angular-deadband", args.base_post_angular_deadband),
    ):
        if not np.isfinite(val) or val < 0.0:
            parser.error(f"{flag} must be finite and >= 0")
    if args.physics_hz <= 0.0:
        parser.error("--physics-hz must be > 0")
    if args.wheel_friction <= 0.0:
        parser.error("--wheel-friction must be > 0")

    cfg = WBCConfig()
    ik = VegaWholeBodyIK(cfg)
    if cfg.head_mode == "ik":
        _pin_head_mode_ik_joints(ik)
    left_frame, right_frame = LEFT_EE_FRAME, RIGHT_EE_FRAME
    print(f"[wbc_vr] safety: collision_avoidance={ik.collision_enabled}  "
          f"com_safety={cfg.enable_com_safety} (margin {cfg.com_safety_margin * 100:.0f}cm)")
    if cfg.head_mode == "ik":
        print("[wbc_vr] head IK pin: head_j1/head_j2 fixed; head_j3 tilt free.")
    ik.reset()
    left0 = _to_mat(ik.frame_pose(left_frame))
    right0 = _to_mat(ik.frame_pose(right_frame))
    head0 = _to_mat(ik.frame_pose(HEAD_FRAME))
    print(f"[wbc_vr] model: nq={ik.model.nq} nv={ik.model.nv}")

    if args.rate <= 0:
        raise ValueError(f"--rate must be > 0, got {args.rate}")
    if args.cmd_rate < 0 or not np.isfinite(args.cmd_rate):
        raise ValueError(f"--cmd-rate must be finite and >= 0, got {args.cmd_rate}")
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and > 0")
    dt = 1.0 / args.rate

    base_mode = ("physics (wheels actuated: swerve steer+drive)" if args.physics
                 else "kinematic (root glide)")
    print(f"[wbc_vr] base mode: {base_mode}")
    if args.physics:
        # Lazy import: pulls sapien + the whole_body_ik joint constants in only when the
        # physics backend is actually used, keeping the kinematic path (and the module-load
        # interpolation test, which stubs whole_body_ik) free of that dependency.
        from omniteleop.follower.wbc_sapien_physics import (  # noqa: PLC0415
            SapienPhysicsRobot,
        )
        robot = SapienPhysicsRobot(
            urdf_path=ik.config.urdf_path,
            with_viewer=not args.no_viewer,
            record_video=args.output,
            video_fps=args.video_fps,
            video_size=(args.width, args.height),
            control_dt=dt,             # SapienPhysicsRobot.step requires dt == control_dt
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
    replay = args.replay is not None
    record_gate = TeleopStartGate()
    stream_recorder = StreamRecorder() if hdf5_path is not None and not replay else None
    stop_requested = False

    def record_stream(vr: VRJointData) -> None:
        nonlocal stop_requested
        if stream_recorder is not None:
            stop_requested = (
                _record_stream_frame_after_calibration(vr, stream_recorder, record_gate)
                or stop_requested
            )
        elif vr.exit_requested:
            stop_requested = True

    if replay:
        source = ReplaySource.from_hdf5(args.replay, args.speed, clock=time.perf_counter)
        print(f"[wbc_vr] replay {args.replay}: {len(source._vr)} frames, "  # noqa: SLF001
              f"{source.total_duration:.1f}s at {args.speed:g}x. Ctrl-C to stop.")
    else:
        source = VRJointSubscriber(args.namespace, sink=record_stream, name="wbc_vr_record")
        print(f"[wbc_vr] subscribed to '{source.topic}'. Run the whole-body VR leader "
              "(scripts/wbc_vr_leader.py -- NOT omni-vr) and grip-hold to calibrate. "
              "Video/HDF5 recording starts automatically on the first teleop frame after "
              "calibration; left X or Ctrl-C stops and saves.")

    closed_loop = args.base_loop == "closed"
    base_deadband = args.base_deadband
    if base_deadband is None:
        base_deadband = (
            DEFAULT_CLOSED_BASE_DEADBAND if closed_loop else DEFAULT_OPEN_BASE_DEADBAND
        )
    base_accel = args.base_accel
    if base_accel is None:
        base_accel = DEFAULT_CLOSED_BASE_ACCEL if closed_loop else DEFAULT_OPEN_BASE_ACCEL
    # Per-axis base-twist shaping (deadband -> velocity clamp -> slew), applied by
    # base_closed_loop.shape_twist exactly as scripts/drive_box_record.py does on the real
    # robot. Angular limits are 2x the linear ones (matching the prior base controller).
    base_deadband_ang = 2.0 * base_deadband
    base_accel_ang = 2.0 * base_accel
    # Velocity envelope (the clamp the kinematic backend used): direction-preserving inside
    # pd_twist/shape_twist via base_closed_loop.limit_twist.
    base_max_lin = base_max_ang = BASE_VEL_LIMIT
    loop_msg = (f"closed-loop (base_closed_loop PD: kp_xy={args.base_kp_xy:g}, "
                f"kp_yaw={args.base_kp_yaw:g})" if closed_loop
                else "open-loop (feed-forward IK base twist only)")
    print(f"[wbc_vr] base control: {loop_msg} -> shape_twist("
          f"deadband {base_deadband:g}/{base_deadband_ang:g}, "
          f"slew {base_accel:g}/{base_accel_ang:g}, clamp {BASE_VEL_LIMIT:g}) -> "
          f"post_deadband {args.base_post_linear_deadband:g}/"
          f"{args.base_post_angular_deadband:g} -> "
          "set_velocity(sequential_steering=False); e-stop / failed solve zeros the base.")
    # Raw leader commands (4x4), held across frames and updated whenever the leader
    # publishes new ones (so the robot freezes, not snaps, when teleop pauses). The
    # IK does not consume them directly: each NEW command is pushed into the
    # interpolator, which the IK reads every tick.
    left_cmd = left0.copy()
    right_cmd = right0.copy()
    # Head-frame command (the leader's head_ee_pose, the calibrated headset pose),
    # seeded at the nominal head FK so the head holds nominal until first published.
    # It rides the same interpolation segments as the EE targets, then passes through
    # a small pose LPF before IK. In head_mode "ik" this keeps headset dither from
    # resolving directly into base yaw; set --head-lpf-tau 0 to disable.
    head_cmd = head0.copy()
    last_cmd_ns = -1  # leader timestamp of the last command pushed into a segment
    speed = args.speed if replay else 1.0
    base_dur = (1.0 / args.cmd_rate) / speed if args.cmd_rate > 0 else 0.0
    interp = TargetInterpolator(base_dur, left0, right0, head0)
    left_target = left0.copy()
    right_target = right0.copy()
    head_lpf = HeadTargetLowPassFilter(args.head_lpf_tau, head0)
    head_planar_deadband = HeadTargetPlanarDeadbandFilter(
        head0,
        position_deadband=args.head_planar_pos_deadband,
        yaw_deadband=args.head_planar_yaw_deadband,
    )
    print(f"[wbc_vr] head target LPF: tau={args.head_lpf_tau:g}s"
          f"{' (disabled)' if args.head_lpf_tau == 0.0 else ''}")
    print(f"[wbc_vr] head planar deadband: pos={args.head_planar_pos_deadband:g}m, "
          f"yaw={args.head_planar_yaw_deadband:g}rad")
    # Last shaped base twist, the slew anchor for base_closed_loop.shape_twist (replaces
    # BaseVelocityController's internal _prev). Reset to zero on hold/e-stop so the next
    # command ramps from rest.
    prev_cmd = np.zeros(3)
    if hasattr(source, "start"):
        source.start()
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
    recording_started = replay

    # Unified HDF5 (--hdf5 / --no-hdf5): live mode writes the replayable leader stream at
    # the root, then debug rows under /debug. Replay mode writes only /debug unless the
    # user points --hdf5 at an existing stream file.
    log = DebugLogger(hdf5_path)
    if log.enabled:
        print(f"[wbc_vr] HDF5 log -> {hdf5_path} (stream root, debug under /debug)")
    hdf5_meta = dict(
        rate=float(args.rate), cmd_rate=float(args.cmd_rate),
        base_loop=str(args.base_loop), physics=bool(args.physics),
        base_deadband=float(base_deadband), base_accel=float(base_accel),
        base_kp_xy=float(args.base_kp_xy), base_kp_yaw=float(args.base_kp_yaw),
        base_vel_limit=float(BASE_VEL_LIMIT),
        head_lpf_tau=float(args.head_lpf_tau),
        head_planar_pos_deadband=float(args.head_planar_pos_deadband),
        head_planar_yaw_deadband=float(args.head_planar_yaw_deadband),
        base_post_linear_deadband=float(args.base_post_linear_deadband),
        base_post_angular_deadband=float(args.base_post_angular_deadband),
        record_start="post_calibration",
        replay_file=str(args.replay or ""), replay_speed=float(args.speed),
        head_mode=str(cfg.head_mode), nq=int(ik.model.nq), nv=int(ik.model.nv),
        urdf_path=str(ik.config.urdf_path), safe_margin=float(cfg.com_safety_margin),
        namespace=str(args.namespace), video_output=str(args.output),
    )
    try:
        while True:
            now = time.perf_counter()
            t = now - t0
            if robot.viewer_closed:
                break
            if replay and source.done:
                break
            if stop_requested:
                print("\n[wbc_vr] leader requested stop (left X).")
                break

            vr = source.latest
            if vr is not None and vr.exit_requested:
                print("\n[wbc_vr] leader requested stop (left X).")
                break
            seen_data = seen_data or vr is not None
            if not replay and record_gate.update(vr) and not recording_started:
                recording_started = True
                print("\n[wbc_vr] recording started after calibration.")
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
                prev_cmd = np.zeros(3)  # restart the base-twist slew from rest
                left_cmd, right_cmd = left0.copy(), right0.copy()
                head_cmd = head0.copy()  # ik.reset() put the head back at nominal
                interp.reset(left0, right0, head0)
                head_lpf.reset(head0)
                head_planar_deadband.reset(head0)
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
                seg = source.segment_duration if replay else None
                duration = base_dur if seg is None else float(seg)
                if replay and base_dur > 0.0:
                    limit = DEFAULT_REPLAY_GAP_LIMIT_MULTIPLE * base_dur
                    if duration > limit:
                        raise RuntimeError(
                            f"replay command gap {duration:.6g}s exceeds {limit:.6g}s "
                            f"limit ({DEFAULT_REPLAY_GAP_LIMIT_MULTIPLE:g}x nominal "
                            f"{base_dur:.6g}s); inspect or re-record before replaying."
                        )
                interp.push(left_cmd, right_cmd, head_cmd, now=now, duration=duration)
            left_target, right_target, head_target_interp = interp.at(now)
            # Pre-LPF interpolated head target (logged for before/after-LPF diagnostics);
            # the LPF then smooths it into the value actually fed to the head IK.
            head_target = head_lpf.filter(head_target_interp, dt)
            head_target = head_planar_deadband.filter(head_target)
            if cfg.head_mode == "ik":
                # Head is a whole-body QP DOF: the head FrameTask tracks the headset
                # pose directly, so base + torso + head co-track it (no solve_head).
                result = ik.solve(
                    left_target, right_target, dt, head_target=head_target
                )
            else:
                # Stiff head IK from the LIVE configuration (compensates base yaw /
                # torso lean within a tick), then the whole-body solve with the head
                # pinned at that command.
                head_joints = ik.solve_head(head_target, dt)
                result = ik.solve(
                    left_target, right_target, dt, head_joints=head_joints
                )

            robot.left_arm.set_joint_pos(result.left_arm)
            robot.right_arm.set_joint_pos(result.right_arm)
            robot.torso.set_joint_pos(result.torso)
            robot.head.set_joint_pos(result.head)
            # Base command twist, via the SAME base_closed_loop helpers the real robot uses
            # (scripts/drive_box_record.py). CLOSED loop (--base-loop closed): pd_twist adds
            # PD feedback on the (solver base pose - measured base pose) error to the solver's
            # base twist (feed-forward), steering the wheel-driven/rendered base onto the
            # solver's reference so the EEFs FOLLOW the target frames. OPEN loop: just the IK
            # base twist (feed-forward), so command shaping / wheel dynamics drift the base
            # off and the EEFs lag. shape_twist then applies deadband -> velocity clamp ->
            # slew (replacing BaseVelocityController) before the coordinated send-now
            # set_velocity(..., sequential_steering=False). On hold (leader e-stop or failed
            # solve) the base is zeroed immediately and the slew anchor reset to rest.
            hold_base = estop or not result.success
            # Defaults so the debug log always has these even on a held tick.
            raw_twist = np.zeros(3)
            pd_err = np.zeros(3)
            base_settled = False
            pd_twist_out = np.zeros(3)
            if hold_base:
                shaped_twist = np.zeros(3)
                prev_cmd = np.zeros(3)
            else:
                if closed_loop:
                    raw_twist, pd_err = base_cl.pd_twist(
                        result.base_pose, result.base_twist, robot.base_pose,
                        kp_xy=args.base_kp_xy, kp_yaw=args.base_kp_yaw,
                        max_lin_speed=base_max_lin, max_ang_speed=base_max_ang,
                    )
                else:
                    raw_twist = np.asarray(result.base_twist, dtype=np.float64)
                # Snapshot the PD/FF twist before shaping and final post-deadbanding.
                pd_twist_out = np.asarray(raw_twist, dtype=np.float64).copy()
                shaped_twist = base_cl.shape_twist(
                    raw_twist, prev_cmd, dt,
                    deadband_lin=base_deadband, deadband_ang=base_deadband_ang,
                    max_lin_speed=base_max_lin, max_ang_speed=base_max_ang,
                    max_lin_accel=base_accel, max_ang_accel=base_accel_ang,
                )
                shaped_twist = deadband_planar_twist_by_reference(
                    shaped_twist,
                    reference_twist=raw_twist,
                    linear_deadband=args.base_post_linear_deadband,
                    angular_deadband=args.base_post_angular_deadband,
                )
                prev_cmd = shaped_twist
            robot.chassis.set_velocity(
                vx=float(shaped_twist[0]), vy=float(shaped_twist[1]),
                wz=float(shaped_twist[2]), wait_time=0.0, sequential_steering=False,
            )

            head_target_marker = _orientation_marker_pose(
                head_target, ik.frame_pose(HEAD_FRAME)
            )
            robot.set_targets(left_target, right_target, head_target_marker)
            robot.step(dt)
            robot.render(record_video=recording_started)

            # Rendered (measured) EE/base state AFTER this tick's base integration: the
            # L_ee/R_ee links ride the velocity-driven base, so this is the TRUE gap to
            # the drawn target frames (the solver's own result.*_ee_error is ~0). Computed
            # every tick (cheap link-pose reads) for both the debug log and the periodic
            # print below.
            eep = robot.ee_positions()
            err_l = float(np.linalg.norm(eep["L"] - left_target[:3, 3]))
            err_r = float(np.linalg.norm(eep["R"] - right_target[:3, 3]))
            bx, by, byaw = robot.base_pose

            if log.enabled and recording_started:
                log.append(
                    t=t, unix_ns=time.time_ns(), cmd_ns=int(last_cmd_ns),
                    calib_stage=(vr.calib_stage if vr is not None else "?"),
                    estop=bool(estop), tracking=bool(tracking), hold_base=bool(hold_base),
                    left_cmd=left_cmd.astype(np.float32),
                    right_cmd=right_cmd.astype(np.float32),
                    head_cmd=head_cmd.astype(np.float32),
                    left_target=left_target.astype(np.float32),
                    right_target=right_target.astype(np.float32),
                    head_target=head_target.astype(np.float32),
                    head_target_pre_lpf=head_target_interp.astype(np.float32),
                    q=np.asarray(result.q, dtype=np.float32),
                    velocity=np.asarray(result.velocity, dtype=np.float32),
                    ik_base_pose=np.asarray(result.base_pose, dtype=np.float32),
                    ik_base_twist=np.asarray(result.base_twist, dtype=np.float32),
                    torso=np.asarray(result.torso, dtype=np.float32),
                    left_arm=np.asarray(result.left_arm, dtype=np.float32),
                    right_arm=np.asarray(result.right_arm, dtype=np.float32),
                    head=np.asarray(result.head, dtype=np.float32),
                    left_ee_error=float(result.left_ee_error),
                    right_ee_error=float(result.right_ee_error),
                    success=bool(result.success), held=bool(result.held),
                    stability_margin=float(result.stability_margin),
                    min_self_distance=float(result.min_self_distance),
                    safety_status=str(result.safety_status),
                    raw_twist=np.asarray(raw_twist, dtype=np.float32),
                    pd_twist_out=np.asarray(pd_twist_out, dtype=np.float32),
                    pd_err=np.asarray(pd_err, dtype=np.float32),
                    shaped_twist=np.asarray(shaped_twist, dtype=np.float32),
                    base_settled=bool(base_settled),
                    meas_base_pose=np.array([bx, by, byaw], dtype=np.float32),
                    ee_L_pos=eep["L"].astype(np.float32),
                    ee_R_pos=eep["R"].astype(np.float32),
                    rendered_err_L=err_l, rendered_err_R=err_r,
                )

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
                elif not seen_data:
                    state = "waiting-for-VR"
                elif not recording_started:
                    state = "waiting-calib"
                elif tracking:
                    state = "tracking"
                else:
                    # Show the leader's stage so a stuck robot is self-explanatory:
                    # "static" = grip-hold to calibrate; anything else = wrong leader.
                    state = f"held(estop:{stage})"
                # err_l/err_r/bx/by/byaw are the per-tick rendered values computed above.
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
        if hasattr(robot.chassis, "stop"):
            robot.chassis.stop()  # zero the base velocity before tearing down
        robot.close()   # releases the video writer
        if hasattr(source, "close"):
            source.close()
        if stream_recorder is not None:
            if len(stream_recorder) > 0:
                try:
                    n_stream = stream_recorder.write(hdf5_path, meta=hdf5_meta)
                    print(f"\n[wbc_vr] stream log -> {hdf5_path} ({n_stream} frames)")
                except Exception as exc:
                    print(f"\n[wbc_vr] !! failed to write stream log {hdf5_path}: {exc}")
            elif hdf5_path is not None:
                print("\n[wbc_vr] no stream frames recorded; HDF5 stream not written "
                      "(did you hold right grip/squeeze to calibrate?).")
        if log.enabled:
            try:
                log.write(hdf5_meta)
                print(f"\n[wbc_vr] debug log -> {hdf5_path}:/debug ({len(log)} frames)")
            except Exception as exc:  # never let logging failure mask the shutdown
                print(f"\n[wbc_vr] !! failed to write debug log {hdf5_path}: {exc}")
        print("[wbc_vr] stopped.")


if __name__ == "__main__":
    main()
