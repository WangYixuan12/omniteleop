#!/usr/bin/env python3
"""Real-robot whole-body VR follower for ``wbc_vr_leader.py`` -- staged-safety bring-up.

Same control core as the SAPIEN sim follower (``scripts/wbc_vr_record.py``) -- subscribe
to the leader's Cartesian ``L_ee``/``R_ee``/head TARGET poses, interpolate the low-rate
command stream up to the IK rate, run ``VegaWholeBodyIK`` -- but driving the REAL Vega
hardware (``dexcontrol.robot.Robot``) instead of a simulator.

AVOID using low-level APIs like `set_motion_state`, `set_steering_angle`, or
`set_wheel_velocity` directly, which can command opposed steering/velocity and damage
the wheel motors. Route base motion through `set_velocity`.

Two real-robot modes:

  1. ``--replay FILE``   replay that SAPIEN-verified stream, TIME-SCALED (by ``replay_speed``
                         in ``follower/wbik.yaml``), through the same whole-body IK on the robot.
  2. ``--source live``   subscribe to the live leader and drive the robot in real time
                         (only after the replay path is verified).

All control-loop tunables -- replay speed, IK / command rates, head LPF + planar
deadbands, base PD gains, base slew / deadband / post-deadband -- come SOLELY from the
``vr_teleop:`` block of ``follower/wbik.yaml`` (shared with ``scripts/wbc_vr_record.py``),
so a take recorded/visualized in sim drives the robot identically. They are not CLI flags.

Pass ``--debug-dir DIR`` to write a per-tick ``/debug`` HDF5 (auto-named
``episode_<N>_debug.hdf5`` under ``DIR``, matching ``vr_reader`` / ``vis_teleop_curves``):
the base PD chain (odom pose +
raw wheels, IK base pose/twist, PD raw/err, shaped command), plus per-group joint
command/sent/measured -- enough to inspect commanded vs measured motion. OFF by default.

``--record`` starts automatically on the first engage after VR calibration, matching
``scripts/wbc_vr_record.py``'s post-calibration recording gate. It saves an episode in
the ``EpisodeRecorder`` schema of ``leader/vr_reader.py`` (mostly interchangeable):
``action``/``obs`` joints plus ``head_left_rgb``/``head_depth``/``left_wrist_rgb`` and the
``left``/``right`` gripper under ``action/gripper`` (leader trigger command) and
``obs/gripper`` (Robotiq FC03 achieved position). It adds ``torso`` to ``action/joint``
(the WBC commands the torso) and uses the shaped base twist for the ``chassis_*`` action;
``action/joint/*`` records the POST-CLAMP joint commands actually sent to hardware
(``_dbg['sent_joints']``), not the raw WBC result. Recording requires the FULL
``--enable torso,arms,head,base`` mask: it records the MEASURED wheel-odometry body twist
as the ``obs/joint`` ``chassis_*`` (achieved-velocity counterpart to the action) and the
measured base pose ``obs/base/pose`` ``(x, y, yaw)`` in the engage-origin world frame.

For the 10 Hz mobile policy (PLAN.md) each frame also records the WORLD-frame targets
given VERBATIM to ``ik.solve()`` this tick: ``action/eef/{left,right}`` and
``action/head`` (the head target post LPF/planar-deadband), each ``(4, 4)`` float32,
plus ``action/base/pose`` = the solver's commanded base pose ``result.base_pose``
(not a policy field; free to record and lets offline analysis derive base-frame
variants and IK-world vs odom-world tracking error).

This deviates from vr_reader's ``obs/images`` to support a MOVING base: the camera
``intrinsic`` is stored once (``(3, 3)``, constant), and the per-frame ``extrinsic`` is
NOT stored -- it was base-relative head FK (base-blind) and is recomputed offline as
``world_t_cam = world_t_base(obs/base/pose) . FK_cam(obs/joint)`` so the colored cloud
lands in the world frame (see ``scripts/vis_episode.py``). The wrist camera streams from
``sensors/wrist_zedm/*`` (needs an external ZED-SDK publisher; see vr_reader).

Run in the dexmate conda env (pinocchio + pink + dexcomm + dexcontrol)
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import asdict
from typing import Callable, Optional

import numpy as np
from dexcomm.codecs import DictDataCodec

from omniteleop.common import get_config
from omniteleop.common.head_camera import ZED_K
from omniteleop.common.recorder import EpisodeRecorder, peek_next_episode_id
from omniteleop.common.schemas import WBCFollowerStatus
from omniteleop.follower.whole_body_ik import (
    HEAD_FRAME,
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    VegaWholeBodyIK,
    WBCConfig,
)
from omniteleop.wbc_record import (
    DEFAULT_REPLAY_GAP_LIMIT_MULTIPLE,
    ReplaySource,
)
from omniteleop.wbc_robot_home import home_to_nominal
from omniteleop.wbc_robot_util import base_quiet_dispatch, parse_enable_mask
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
# follower/wbik.yaml so this real-robot follower and the SAPIEN sim follower
# (scripts/wbc_vr_record.py) shape the head/base command stream identically -- a take
# recorded in sim must replay the same way here. See omniteleop.wbc_teleop. The
# bring-up-specific knobs (homing, joint-step clamp, base max-speed, ...) stay local below.
_VR_TELEOP = VRTeleopConfig.from_yaml()
DEFAULT_HEAD_LPF_TAU = _VR_TELEOP.head_lpf_tau
DEFAULT_SPEED = _VR_TELEOP.replay_speed
DEFAULT_IK_RATE = _VR_TELEOP.ik_rate                # whole-body IK / control loop rate (Hz)
DEFAULT_CMD_RATE = _VR_TELEOP.cmd_rate              # leader command rate (Hz); 0 disables interp
DEFAULT_BASE_KP_XY = _VR_TELEOP.base_kp_xy          # closed-loop base PD gains (real-robot values)
DEFAULT_BASE_KP_YAW = _VR_TELEOP.base_kp_yaw
DEFAULT_BASE_ACCEL = _VR_TELEOP.base_accel          # m/s^2 base-twist slew (ang = 2x)
DEFAULT_BASE_DEADBAND = _VR_TELEOP.base_deadband    # m/s base-twist deadband (ang = 2x)
DEFAULT_BASE_MAX_SPEED = _VR_TELEOP.base_max_speed  # m/s base velocity clamp (ang = 2x)
DEFAULT_HEAD_PLANAR_POS_DEADBAND = _VR_TELEOP.head_planar_pos_deadband
DEFAULT_HEAD_PLANAR_YAW_DEADBAND = _VR_TELEOP.head_planar_yaw_deadband
DEFAULT_BASE_POST_LINEAR_DEADBAND = _VR_TELEOP.base_post_linear_deadband
DEFAULT_BASE_POST_ANGULAR_DEADBAND = _VR_TELEOP.base_post_angular_deadband
DEFAULT_STATUS_PUBLISH_RATE = 10.0
WBC_FOLLOWER_STATUS_TOPIC = "wbc/follower_status"

# Episode recording (--record): vr_reader EpisodeRecorder format so takes are
# interchangeable with VR-teleop recordings. Default cadence matches vr_reader's
# record_rate; the default dir is the same raw_data tree the leader writes to.
DEFAULT_RECORD_RATE = 10.0
DEFAULT_SAVE_DIR = "/home/yixuan/omniteleop/Dexmate/data/raw_data"

# Wrist camera (recorded under obs/images/left_wrist_rgb when --record). Like the head
# camera it is consumed through the dexcontrol Robot API, but as a multi-stream ZED-M
# published by a standalone ZED-SDK process on sensors/wrist_zedm/{left_rgb,right_rgb}
# (see leader/vr_reader.py + tests/test_wrist_zedm_depth.py). RGB-only; left eye only.
_WRIST_SENSOR_ID = "wrist_zedm"
_WRIST_OBS_KEY = "left_rgb"


def _wbc_follower_status_topic() -> str:
    return get_config().get_topic("wbc_follower_status", WBC_FOLLOWER_STATUS_TOPIC)


def _create_status_publisher(source):
    node = getattr(source, "node", None)
    if node is None:
        return None
    return node.create_publisher(_wbc_follower_status_topic(), encoder=DictDataCodec.encode)


def _build_follower_status(
    vr,
    result,
    *,
    estop: bool,
    hold: bool,
    hold_reason: str,
    timestamp_ns: Optional[int] = None,
) -> WBCFollowerStatus:
    """Build the lightweight status frame consumed by the WBC VR leader HUD."""
    return WBCFollowerStatus(
        timestamp_ns=int(time.time_ns() if timestamp_ns is None else timestamp_ns),
        stage=str(getattr(vr, "calib_stage", "static") if vr is not None else "static"),
        estop=bool(estop),
        success=bool(result.success),
        held=bool(result.held),
        hold=bool(hold),
        hold_reason=str(hold_reason or ""),
        safety_status=str(result.safety_status),
        left_ee_error_mm=float(result.left_ee_error) * 1000.0,
        right_ee_error_mm=float(result.right_ee_error) * 1000.0,
    )

# Hardware safety / bring-up defaults (robot-specific; conservative for a first slow
# bring-up). The shared closed-loop base-control tunables (PD gains, slew, deadband,
# base velocity clamp, post-deadband) come from the vr_teleop: block above; only the
# homing/joint-step/watchdog knobs live here.
DEFAULT_MAX_JOINT_STEP = 0.05   # rad/IK-tick clamp on arm/torso/head joint commands
DEFAULT_SOURCE_TIMEOUT = 0.5    # s without a fresh command / odom sample -> hold
# rad: measured-vs-nominal gate before the first engage. Sized to tolerate normal
# position-control steady-state droop (gravity/friction on a loaded arm joint is a few
# degrees) while still catching a GROSS mismatch (a joint-convention bug is >=0.5 rad,
# often pi). The per-tick --max-joint-step clamp closes whatever residual remains at
# engage in a couple of ticks (0.1 rad / 0.05 rad/tick @ 100Hz = ~20ms), so a few
# degrees here is smoothed away rather than commanded as a jump.
DEFAULT_HOME_TOL = 0.1
ARM_HOME_STEP = 0.005 # arm interpolation step when homing straight to nominal
# s: after the stepped ramp, hold each group at nominal and block until the MEASURED
# joints converge within --home-tol. The dexcontrol ramp returns after each substep's
# wait_time whether or not the joint physically caught up, so lag accumulates and an
# immediate readback catches the arm mid-flight (the variable residual). This drains it.
DEFAULT_HOME_SETTLE = 3.0
# Hold the current swerve steering if zero base command < quiet_hold_s seconds, then
# re-center the wheels to 0deg.
# (set_velocity(0,0,0) -> steering 0)
DEFAULT_BASE_QUIET_HOLD_S = 5
_JOINT_STEP_ABORT_TICKS = 25    # consecutive ticks demanding > 2x clamp -> abort


class _TrajLog:
    """Collect per-tick dry-run IK state; summarize and optionally write to HDF5."""

    def __init__(self, path: Optional[str], meta: Optional[dict] = None) -> None:
        self.path = path
        self.meta = meta or {}
        self._rows: list[dict] = []

    def append(self, **fields) -> None:
        """Record one IK tick (no-op when no output path is set)."""
        if self.path is not None:
            self._rows.append(fields)

    def __len__(self) -> int:
        return len(self._rows)

    def summary(self) -> str:
        """One-line headline over the run (max EE errors, holds, base travel)."""
        if not self._rows:
            return "no ticks"
        tele = [r for r in self._rows if not r["estop"]]
        if not tele:
            return f"{len(self._rows)} ticks, all e-stopped (no teleop frames)"
        el = np.array([r["left_ee_error"] for r in tele])
        er = np.array([r["right_ee_error"] for r in tele])
        held = sum(1 for r in tele if r["held"])
        ok = sum(1 for r in tele if r["success"])
        bp = np.array([r["base_pose"] for r in tele])
        travel = float(np.linalg.norm(bp[-1][:2] - bp[0][:2]))
        return (
            f"{len(tele)} teleop ticks | EEerr max L={el.max() * 1000:.1f}mm "
            f"R={er.max() * 1000:.1f}mm | solve_ok={ok}/{len(tele)} held={held} | "
            f"base travel={travel:.3f}m, final yaw={np.degrees(bp[-1][2]):+.1f}deg"
        )

    # Per-tick scalar columns, gzipped vector columns, and fixed-width string columns.
    _SCALAR_KEYS = (
        "t", "cmd_ns", "estop", "success", "held", "hold", "left_ee_error",
        "right_ee_error", "stability_margin", "clamp_max_over",
        "odom_age", "odom_drive_ts_ns",
    )
    _VECTOR_KEYS = (
        "q", "base_pose", "base_twist", "base_pd_raw", "base_pd_err",
        "base_cmd", "odom_pose", "odom_steer", "odom_wvel",
        "left_target", "right_target", "head_target",
        "cmd_torso", "cmd_left_arm", "cmd_right_arm", "cmd_head",
        "sent_torso", "sent_left_arm", "sent_right_arm", "sent_head",
        "meas_torso", "meas_left_arm", "meas_right_arm", "meas_head",
    )
    _STR_KEYS = (("safety_status", 96), ("hold_reason", 32), ("base_action", 12))

    def write(self) -> None:
        """Write the collected ticks under a ``/debug`` group (if a path is set).

        ``/debug`` holds every per-tick column (scalars, gzipped vectors, fixed-width
        strings) plus run metadata in its attrs. Columns absent this run (no driver) are
        skipped rather than erroring.
        """
        if not self.path or not self._rows:
            return
        import h5py  # noqa: PLC0415 -- optional dep, only needed at teardown

        rows = self._rows
        col = lambda k: np.asarray([r[k] for r in rows])
        with h5py.File(self.path, "w") as f:
            g = f.create_group("debug")
            g.attrs["schema"] = "wbc_vr_robot_debug/v1"
            g.attrs["n_frames"] = len(rows)
            for key, val in self.meta.items():
                g.attrs[key] = val
            for key in self._SCALAR_KEYS:
                if key in rows[0]:
                    g.create_dataset(key, data=col(key))
            for key, width in self._STR_KEYS:
                if key in rows[0]:
                    g.create_dataset(
                        key,
                        data=np.array([str(r[key]).encode()[:width] for r in rows],
                                      dtype=f"S{width}"),
                    )
            for key in self._VECTOR_KEYS:
                if key in rows[0]:
                    g.create_dataset(key, data=col(key),
                                     compression="gzip", compression_opts=4)


class OdometryThread:
    """Background swerve-odometry integrator -> latest measured base pose (x, y, yaw).

    Mirrors ``scripts/drive_box_record.py``: poll ``chassis.steering_angle`` /
    ``wheel_velocity`` with the firmware drive timestamp and integrate with
    ``SwerveOdometry.update_at`` (timestamp-driven, so dropped/stale samples don't
    dead-reckon). The closed-loop base controller reads :attr:`pose` as the *measured*
    base for ``base_closed_loop.pd_twist``; :attr:`age` lets the follower hold the base
    when odometry goes stale.
    """

    def __init__(self, robot, drive_state_mode: str = "ms", rate: float = 200.0) -> None:
        from omniteleop.follower.swerve_odometry import (  # noqa: PLC0415
            OdometryInputError,
            SwerveOdometry,
        )
        self._robot = robot
        self._odo_err = OdometryInputError
        chassis = robot.chassis
        self._odo = SwerveOdometry(
            drive_state_mode=drive_state_mode,
            max_steering_angle=float(getattr(chassis, "_max_steering_angle", 0.7)),
            max_linear_vel=float(getattr(chassis, "max_lin_vel", 1.5)),
        )
        self._period = 1.0 / rate
        self._lock = threading.Lock()
        self._pose = np.zeros(3)
        # Latest MEASURED body twist (vx, vy, w) from swerve wheel FK -- the instantaneous
        # velocity SwerveOdometry integrates into self._pose. Surfaced (drift-free, same
        # body frame/units as the commanded chassis_* action) for obs/joint recording.
        self._twist = np.zeros(3)
        # Raw wheel inputs of the last accepted sample, kept for /debug so base jitter
        # can be traced to the sensor (steering quantization / wheel-vel noise) vs the
        # integration. Snapshotted together with the pose under the same lock.
        self._steer = np.full(2, np.nan)
        self._wvel = np.full(2, np.nan)
        self._drive_ts = -1
        self._update_t = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="wbc_vr_odom")

    def start(self) -> None:
        """Begin the background integration thread."""
        self._thread.start()

    def _run(self) -> None:
        prev_ts: Optional[int] = None
        while not self._stop.is_set():
            now = time.perf_counter()
            try:
                steer = np.asarray(self._robot.chassis.steering_angle, dtype=np.float64)
                wvel = np.asarray(self._robot.chassis.wheel_velocity, dtype=np.float64)
                ts = int(self._robot.chassis.chassis_drive.get_timestamp_ns())
                if steer.shape == (2,) and wvel.shape == (2,) and (
                    prev_ts is None or ts > prev_ts
                ):
                    # Integrate AND store under the lock: update_at mutates self._odo.pose,
                    # so an engage-time reset_origin() (also locked) cannot be clobbered by
                    # an in-flight integration writing back a stale pre-reset pose.
                    with self._lock:
                        try:
                            pose = self._odo.update_at(steer, wvel, ts)
                        except self._odo_err:
                            pose = None  # invalid sample: hold last good pose, let age grow
                        if pose is not None:
                            prev_ts = ts
                            self._pose = pose.copy()
                            self._twist = self._odo.twist.copy()
                            self._steer = steer.copy()
                            self._wvel = wvel.copy()
                            self._drive_ts = ts
                            self._update_t = now
            except Exception:  # never let the odom thread die silently mid-run
                pass
            sleep = self._period - (time.perf_counter() - now)
            if sleep > 0:
                time.sleep(sleep)

    @property
    def pose(self) -> np.ndarray:
        """Latest integrated base pose ``(x, y, yaw)`` (copy)."""
        with self._lock:
            return self._pose.copy()

    @property
    def twist(self) -> np.ndarray:
        """Latest MEASURED body twist ``(vx, vy, w)`` from swerve wheel FK (copy).

        The instantaneous velocity ``SwerveOdometry`` integrates into :attr:`pose` --
        drift-free and in the same body frame / units as the commanded ``chassis_*``
        action, so it is the achieved-velocity obs counterpart. Holds the last integrated
        value when the wheel stream stalls (see :attr:`age` to detect staleness).
        """
        with self._lock:
            return self._twist.copy()

    def snapshot(self) -> dict:
        """Atomic copy of the latest odom state for /debug (pose + age + raw wheels).

        Taken under the lock so the pose the base PD uses this tick and the raw wheel
        inputs logged alongside it come from the SAME integrated sample.
        """
        with self._lock:
            age = (time.perf_counter() - self._update_t) if self._update_t else float("inf")
            return {
                "pose": self._pose.copy(),
                "age": age,
                "steer": self._steer.copy(),
                "wvel": self._wvel.copy(),
                "drive_ts_ns": int(self._drive_ts),
            }

    @property
    def age(self) -> float:
        """Seconds since the last accepted wheel sample (inf before the first)."""
        with self._lock:
            return time.perf_counter() - self._update_t if self._update_t else float("inf")

    def reset_origin(self) -> None:
        """Zero the integrated pose -- called on the engage edge with ``ik.reset()``."""
        with self._lock:
            self._odo.pose = np.zeros(3)
            self._pose = np.zeros(3)

    def stop(self) -> None:
        """Stop and join the background thread (best-effort)."""
        self._stop.set()
        self._thread.join(timeout=1.0)


class HardwareDriver:
    """Drive the real Vega (``dexcontrol.robot.Robot``) from the IK result.

    Homes to the WBC nominal posture before the first engage (so the IK's nominal
    re-anchoring matches the hardware), clamps per-tick joint steps, and runs the same
    ``base_closed_loop`` PD/shape path the real robot uses in ``drive_box_record.py``.
    Every joint/base actuator is gated by the ``--enable`` mask; the grippers are always
    activated and track the leader triggers. On any hold (e-stop / failed solve / safety
    hold / stale source or odom) the base is zeroed and joints are frozen (the grippers
    hold their last commanded position).
    """

    name = "hardware"

    def __init__(self, args: argparse.Namespace, ik: VegaWholeBodyIK, cfg: WBCConfig,
                 enable: dict) -> None:
        from dexbot_utils import RobotInfo  # noqa: PLC0415 -- hardware-only deps, lazy
        from dexcontrol.robot import Robot  # noqa: PLC0415

        from omniteleop.follower import base_closed_loop as base_cl  # noqa: PLC0415
        from omniteleop.follower.robotiq import (  # noqa: PLC0415
            build_hande_command,
            send_activate,
            send_ee_pass_through_with_timestamps,
        )
        from omniteleop.follower.whole_body_ik import (  # noqa: PLC0415
            HEAD_JOINTS,
            LEFT_ARM_JOINTS,
            RIGHT_ARM_JOINTS,
            TORSO_JOINTS,
        )

        self.args = args
        self.ik = ik
        self.cfg = cfg
        self.enable = enable
        # A policy take needs post-clamp sent joints for EVERY group (action/joint/*)
        # and odometry (obs/base/pose anchors the world-frame action targets), so
        # --record only makes sense with the full actuation mask. Fail fast here,
        # before any hardware connection, not on the first recorded frame.
        if args.record:
            missing = [g for g in ("torso", "arms", "head", "base") if not enable.get(g)]
            if missing:
                raise SystemExit(
                    "[wbc_vr_robot] --record requires --enable torso,arms,head,base; "
                    f"missing: {', '.join(missing)}"
                )
        self._base_cl = base_cl
        self._build_hande = build_hande_command
        self._send_ee_pass_through = send_ee_pass_through_with_timestamps
        # robot component name -> ordered IK joint names (for nominal extraction,
        # measured-q assembly, and command dispatch).
        self._joint_names = {
            "torso": list(TORSO_JOINTS), "left_arm": list(LEFT_ARM_JOINTS),
            "right_arm": list(RIGHT_ARM_JOINTS), "head": list(HEAD_JOINTS),
        }
        # Set everything close() touches BEFORE any hardware op, so a failure partway
        # through init (e.g. a failed homing gate) can still tear down cleanly.
        self.robot = None
        self._odom: Optional[OdometryThread] = None
        self._episode: Optional[EpisodeRecorder] = None
        self.has_torso = False
        self.has_chassis = False
        self._prev_base_cmd = np.zeros(3)        # post-projection: twist actually sent
        self._prev_base_shaped = np.zeros(3)     # pre-projection multi-axis slew anchor
        # Active base axis for the post-PD single-axis projection (parity with the WBC's
        # latch); reset to None on hold/engage so the wheel command re-picks from rest.
        self._base_axis: Optional[int] = None
        # Seconds the base command has been continuously quiet (zero): gates the
        # steering-hold-vs-recenter decision (base_quiet_dispatch). Reset on engage/hold.
        self._base_quiet_elapsed = 0.0
        self._prev_cmd: dict = {grp: None for grp in self._joint_names}
        self._overstep_ticks = 0
        # Per-tick /debug scratch: measured_q / actuate / _drive_base stash their
        # intermediates here and debug_row() assembles them after actuation. Touched
        # only by the (single-threaded) follower loop.
        self._dbg: dict = {}
        # Grippers (always active): cache the last commanded triggers for action/gripper
        # recording, the FC03 achieved-position obs (gPO/255), and the per-arm status
        # monitors (set up under --record). Initialized here so close() can tear down even
        # if init fails partway.
        self._last_gripper_cmd_left = 0.0
        self._last_gripper_cmd_right = 0.0
        self._last_obs_grip_left = float("nan")
        self._last_obs_grip_right = float("nan")
        self._grip_monitors: dict = {}
        # --record frame-freshness guard: the ZED-SDK publishers stream at ~15 fps while we
        # record at ~10 fps, so EVERY recorded frame must carry a camera frame strictly newer
        # than the previously recorded one (each head_left_rgb / left_wrist_rgb differs). When
        # WiFi clogging / an fps drop stalls a publisher below the record rate, get_obs returns
        # the SAME (or no) frame twice. These hold the publisher timestamp of the last RECORDED
        # frame per camera (-1 until the first is recorded); record_tick aborts the run the
        # moment a head/wrist frame is not strictly newer, so a take never records duplicates.
        self._last_rec_head_ns = -1
        self._last_rec_wrist_ns = -1

        # Joint convention: with the hardware-matching URDF (wbik.yaml urdf_path:
        # vega_with_robotiq, corrected torso_j2 range + grippers) the WBC joint output
        # maps 1:1 to the dexcontrol convention, so _to_hw() is identity. A mismatched
        # URDF would re-introduce the gap; _to_hw stays the single place to fix it.
        info = RobotInfo()
        self.has_torso = bool(info.has_torso)
        self.has_chassis = bool(info.has_chassis)
        try:
            print("[wbc_vr_robot] connecting to robot hardware ...")
            if args.record:
                # Recording needs the head camera AND the wrist ZED-M streaming, so build
                # the Robot with both sensors enabled (mirrors leader/vr_reader.py). The
                # wrist is a multi-stream ZED-M published by a standalone ZED-SDK process on
                # sensors/wrist_zedm/{left_rgb,right_rgb}; inject an RGB-only config if the
                # variant lacks it. A ZED-SDK publisher must run for frames to arrive.
                from dexbot_utils.configs.components.sensors.cameras import (  # noqa: PLC0415
                    ZedXCameraConfig,
                )
                from dexcontrol.core.config import get_robot_config  # noqa: PLC0415
                configs = get_robot_config()
                if "head_camera" not in configs.sensors:
                    raise SystemExit(
                        "[wbc_vr_robot] --record needs a head_camera sensor but the robot "
                        "config has none."
                    )
                configs.sensors["head_camera"].enabled = True
                if _WRIST_SENSOR_ID not in configs.sensors:
                    configs.sensors[_WRIST_SENSOR_ID] = ZedXCameraConfig(
                        name=_WRIST_SENSOR_ID, enable_rgb=True, enable_depth=False
                    )
                configs.sensors[_WRIST_SENSOR_ID].enabled = True
                self.robot = Robot(configs=configs)
            else:
                self.robot = Robot()
            self._check_startup_component_health()
            nq = ik.nominal_q()
            self._nominal = {
                grp: np.array([nq[ik._idx_q[n]] for n in names])  # noqa: SLF001
                for grp, names in self._joint_names.items()
            }
            self._init_recording()
            # Grippers are always active (the hardware is mounted): activate both Hand-E
            # grippers so the per-tick trigger commands move them. Under --record also set
            # up the FC03 achieved-position monitors for obs/gripper.
            print("[wbc_vr_robot] activating grippers ...")
            send_activate(self.robot.left_arm)
            send_activate(self.robot.right_arm)
            if args.record:
                self._init_gripper_monitors()
            if enable["base"]:
                if not self.has_chassis:
                    raise SystemExit("[wbc_vr_robot] --enable base but the robot has no chassis.")
                self._wait_chassis()
                self._odom = OdometryThread(self.robot, drive_state_mode=args.drive_state_mode)
                self._odom.start()
            home_to_nominal(self, arm_home_step=ARM_HOME_STEP)
        except BaseException:
            # Tear down partially-initialized hardware (odom thread, robot, base) before
            # propagating, so a homing-gate abort can't leak a running base/odom/robot.
            self.close()
            raise

    def _comp(self, grp: str):
        return getattr(self.robot, grp)

    def _to_hw(self, grp: str, joints: np.ndarray) -> np.ndarray:
        """Map WBC joints for ``grp`` to the dexcontrol convention.

        IDENTITY by construction: built on the hardware-matching URDF (``wbik.yaml``
        ``urdf_path`` -> ``vega_with_robotiq``, torso_j2 range corrected to [0,3.14]),
        the WBC joint output
        already matches dexcontrol 1:1. This stays the single chokepoint for every joint
        command (homing included) -- encode any per-joint sign/offset/scale here if a
        future URDF re-introduces a mismatch.
        """
        return np.asarray(joints, dtype=float)

    def stop_all_motion(self) -> None:
        """Immediately stop/hold every actuator this follower may have moved.

        This is the left-X / teardown stop path. It runs before recorder flushing or
        resource shutdown so a "stop the take" request cannot leave a previous 100 Hz
        position target in flight while HDF5 saving begins. The base/head/torso expose
        direct ``stop()`` methods; the arms do not, so cancel their previous position
        target by commanding the currently measured joint position.
        """
        robot = getattr(self, "robot", None)
        if robot is None:
            return

        errors: list[str] = []

        def _try(label: str, fn) -> None:
            try:
                fn()
            except Exception as exc:
                errors.append(f"{label}: {exc!r}")

        chassis = getattr(robot, "chassis", None)
        if getattr(self, "has_chassis", False) and chassis is not None and hasattr(chassis, "stop"):
            _try("chassis.stop", chassis.stop)

        for name in ("torso", "head"):
            comp = getattr(robot, name, None)
            if comp is not None and hasattr(comp, "stop"):
                _try(f"{name}.stop", comp.stop)

        for name in ("left_arm", "right_arm"):
            comp = getattr(robot, name, None)
            if comp is None:
                continue

            def _hold_current(comp=comp) -> None:
                q = np.asarray(comp.get_joint_pos(), dtype=float)
                if q.ndim != 1 or not np.all(np.isfinite(q)):
                    raise ValueError(f"bad measured joint position shape/value: {q}")
                comp.set_joint_pos(q.tolist(), wait_time=0.0)

            _try(f"{name}.hold_current", _hold_current)

        for name in ("left_hand", "right_hand", "left_gripper", "right_gripper"):
            comp = getattr(robot, name, None)
            if comp is not None and hasattr(comp, "stop"):
                _try(f"{name}.stop", comp.stop)

        if hasattr(self, "_prev_cmd"):
            for grp in self._prev_cmd:
                self._prev_cmd[grp] = None
        if hasattr(self, "_prev_base_cmd"):
            self._prev_base_cmd = np.zeros(3)
        if hasattr(self, "_prev_base_shaped"):
            self._prev_base_shaped = np.zeros(3)
        if hasattr(self, "_base_axis"):
            self._base_axis = None
        if hasattr(self, "_base_quiet_elapsed"):
            self._base_quiet_elapsed = 0.0

        if errors:
            print("\n[wbc_vr_robot] WARNING: stop_all_motion partial failures: "
                  + "; ".join(errors))

    def _check_startup_component_health(self) -> None:
        """Fail before motion if dexcontrol reports unhealthy components."""
        from dexcomm.codecs import ConnectionStatusEnum, OperationalStatusEnum  # noqa: PLC0415

        status = self.robot.get_component_status(show=False)
        states = status.get("states", {})
        if not states:
            raise SystemExit("[wbc_vr_robot] startup health check failed: no component status.")

        requested = set()
        if self.enable["arms"]:
            requested.update(("left_arm", "right_arm"))
        if self.enable["head"]:
            requested.add("head")
        if self.enable["torso"]:
            requested.add("torso")
        if self.enable["base"]:
            requested.update(("chassis_drive", "chassis_steer"))

        bad = []
        for name, state in sorted(states.items()):
            connected = state.get("connection") == ConnectionStatusEnum.CONNECTED
            operation = state.get("operation")
            error_info = state.get("error", {}) or {}
            error_msg = error_info.get("error_message", "")

            if not connected:
                bad.append(f"{name}: disconnected")
            if operation == OperationalStatusEnum.ERROR or error_msg:
                bad.append(f"{name}: {error_msg or 'operation ERROR'}")
            if name in requested and operation == OperationalStatusEnum.DISABLED:
                bad.append(f"{name}: disabled but requested by --enable")
            if name in requested and operation == OperationalStatusEnum.CALIBRATING:
                bad.append(f"{name}: calibrating but requested by --enable")

        if bad:
            raise SystemExit(
                "[wbc_vr_robot] startup health check FAILED; aborting before motion:\n  - "
                + "\n  - ".join(bad)
            )
        print("[wbc_vr_robot] startup health check OK: all reported components healthy.")

    def _wait_chassis(self, timeout: float = 10.0) -> None:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if np.asarray(self.robot.chassis.steering_angle).shape == (2,):
                return
            time.sleep(0.05)
        raise SystemExit(f"[wbc_vr_robot] no chassis state within {timeout:.0f}s.")

    def engage_reset(self, ik, left0, right0, head0) -> None:
        """Zero the odometry origin in lock-step with the IK reset at engage."""
        if self._odom is not None:
            self._odom.reset_origin()
        self._prev_base_cmd = np.zeros(3)
        self._prev_base_shaped = np.zeros(3)
        self._base_axis = None
        self._base_quiet_elapsed = 0.0
        for grp in self._prev_cmd:
            self._prev_cmd[grp] = None
        # If an episode is already active, re-anchor the cadence on re-engage. A fresh
        # episode starts only once the leader reaches calib_stage="teleop"; the optional
        # reference-alignment stage can actuate the robot without recording frames.
        if self._episode is not None and self._episode.recording:
            self._next_record_t = 0.0

    def start_recording_if_teleop(self, vr) -> None:
        """Start episode recording once the leader leaves any pre-record align stage."""
        if self._episode is None or self._episode.recording:
            return
        if vr is None or str(getattr(vr, "calib_stage", "")) != "teleop":
            return
        self._episode.start()
        self._next_record_t = 0.0

    def _read_measured_joints(self) -> dict:
        """Cached readback of every group's measured joints (``get_joint_pos``).

        Returns ``{group: np.ndarray | None}``; a group whose readback has the wrong
        shape or a non-finite value maps to ``None`` (so a bad sample is flagged in
        /debug and never used as a seed). Pure reads -- safe to call for instrumentation.
        """
        out: dict = {}
        for grp, names in self._joint_names.items():
            m = np.asarray(self._comp(grp).get_joint_pos(), dtype=float)
            out[grp] = m if (m.shape == (len(names),) and np.all(np.isfinite(m))) else None
        return out

    def extra_hold(self, now, last_cmd_wall, estop) -> Optional[str]:
        """Hold (beyond e-stop/solve/safety) until a fresh command, or on stale stream/odom."""
        from omniteleop.wbc_robot_util import compute_hold_reason  # noqa: PLC0415

        odom_age = self._odom.age if self._odom is not None else None
        return compute_hold_reason(
            estop, last_cmd_wall, now, self.args.source_timeout, odom_age
        )

    def actuate(self, result, left_gripper, right_gripper, enable, hold, dt) -> None:
        """Send the IK result to the enabled actuators (clamped); zero/freeze on hold."""
        from omniteleop.wbc_robot_util import clamp_joint_step  # noqa: PLC0415

        cmds = {"torso": result.torso, "left_arm": result.left_arm,
                "right_arm": result.right_arm, "head": result.head}
        grp_on = {"torso": enable["torso"], "head": enable["head"],
                  "left_arm": enable["arms"], "right_arm": enable["arms"]}
        self._dbg["cmd_joints"] = {g: np.asarray(c, dtype=float) for g, c in cmds.items()}
        if hold:
            # Freeze joints (send nothing) and re-seed the clamp anchor on resume.
            for grp in self._prev_cmd:
                self._prev_cmd[grp] = None
            self._dbg["sent_joints"] = None
            self._dbg["clamp_max_over"] = float("nan")
        else:
            max_over = 0.0
            sent: dict = {}
            for grp, cmd in cmds.items():
                if not grp_on[grp]:
                    continue
                prev = self._prev_cmd[grp]
                if prev is None:
                    prev = np.asarray(self._comp(grp).get_joint_pos(), dtype=float)
                clamped, requested = clamp_joint_step(prev, cmd, self.args.max_joint_step)
                max_over = max(max_over, requested)
                self._comp(grp).set_joint_pos(self._to_hw(grp, clamped).tolist(), wait_time=0.0)
                self._prev_cmd[grp] = clamped
                sent[grp] = clamped
            self._dbg["sent_joints"] = sent
            self._dbg["clamp_max_over"] = max_over
            # Grippers are always active: command both Hand-E grippers from the leader
            # triggers and stash the clipped values so record_tick logs them as
            # action/gripper (this non-hold branch is the only place that records).
            lg = float(np.clip(left_gripper, 0.0, 1.0))
            rg = float(np.clip(right_gripper, 0.0, 1.0))
            self._send_ee_pass_through(self.robot.left_arm, self._build_hande(lg))
            self._send_ee_pass_through(self.robot.right_arm, self._build_hande(rg))
            self._last_gripper_cmd_left = lg
            self._last_gripper_cmd_right = rg
            # Abort if the IK keeps demanding far more than the clamp (runaway / bad target).
            if self.args.max_joint_step > 0 and max_over > 2.0 * self.args.max_joint_step:
                self._overstep_ticks += 1
                if self._overstep_ticks > _JOINT_STEP_ABORT_TICKS:
                    raise RuntimeError(
                        f"IK joint step {max_over:.3f} rad exceeded 2x the "
                        f"{self.args.max_joint_step:g} rad clamp for "
                        f"{self._overstep_ticks} ticks -- aborting for safety."
                    )
            else:
                self._overstep_ticks = 0
        self._drive_base(result, hold, enable, dt)

    def _drive_base(self, result, hold, enable, dt) -> None:
        if self._odom is None or not enable["base"]:
            self._dbg["odom"] = None
            self._dbg["base_pd_raw"] = None
            self._dbg["base_pd_err"] = None
            self._dbg["base_cmd"] = None
            self._dbg["base_action"] = "off"
            return
        # Snapshot odom ONCE: the pose fed to the PD and the raw wheels logged for this
        # tick come from the same integrated sample.
        snap = self._odom.snapshot()
        self._dbg["odom"] = snap
        chassis = self.robot.chassis
        if hold:
            chassis.set_velocity(vx=0.0, vy=0.0, wz=0.0, wait_time=0.0,
                                 sequential_steering=True)
            self._prev_base_cmd = np.zeros(3)
            self._prev_base_shaped = np.zeros(3)
            self._base_axis = None
            self._base_quiet_elapsed = 0.0
            self._dbg["base_pd_raw"] = None
            self._dbg["base_pd_err"] = None
            self._dbg["base_cmd"] = np.zeros(3)
            self._dbg["base_action"] = "safety_hold"
            return
        base_cl = self._base_cl
        max_lin = self.args.base_max_speed
        max_ang = 2.0 * self.args.base_max_speed
        raw, err = base_cl.pd_twist(
            result.base_pose, result.base_twist, snap["pose"],
            kp_xy=self.args.base_kp_xy, kp_yaw=self.args.base_kp_yaw,
            max_lin_speed=max_lin, max_ang_speed=max_ang,
        )
        pd_raw = np.asarray(raw, dtype=float).copy()  # pre-shaping PD/FF output (debug)
        # Shape (deadband -> clamp -> slew) the FULL multi-axis command and keep that
        # multi-axis result as the slew anchor (_prev_base_shaped), so EVERY axis stays
        # "warm"; mask to the single dominant axis LAST. Order matters: if the single-axis
        # projection ran FIRST, it would feed shape_twist a signal that drops to zero on every
        # momentary axis switch (the QP/PD resolves leader noise into a brief off-axis win or a
        # quiet tick ~20% of the time), the post-deadband would then hard-zero the dominant
        # axis, and the slew limiter would re-ramp it from zero -- throttling the wheel command
        # ~36-43% below the commanded speed, so the base lags and the arms fall behind the
        # targets. Shaping first keeps the dominant axis at full slewed magnitude and low-passes
        # the axis selection (far fewer spurious switches), while the chassis still receives a
        # pure single-axis command. (See base_closed_loop.project_planar_twist_single_axis.)
        cmd = base_cl.shape_twist(
            raw, self._prev_base_shaped, dt,
            deadband_lin=self.args.base_deadband, deadband_ang=2.0 * self.args.base_deadband,
            max_lin_speed=max_lin, max_ang_speed=max_ang,
            max_lin_accel=self.args.base_accel, max_ang_accel=2.0 * self.args.base_accel,
        )
        cmd = deadband_planar_twist_by_reference(
            cmd,
            reference_twist=raw,
            linear_deadband=self.args.base_post_linear_deadband,
            angular_deadband=self.args.base_post_angular_deadband,
        )
        self._prev_base_shaped = cmd  # slew anchor stays multi-axis (every axis warm)
        if self.cfg.enable_base_single_axis:
            cmd, self._base_axis = base_cl.project_planar_twist_single_axis(
                cmd,
                xy_max_vel=self.cfg.base_xy_max_vel, yaw_max_vel=self.cfg.base_yaw_max_vel,
                deadband=self.cfg.base_single_axis_deadband,
                hysteresis_ratio=self.cfg.base_single_axis_hysteresis_ratio,
                prev_axis=self._base_axis,
            )
        self._prev_base_cmd = cmd  # post-projection: the twist actually sent to the chassis
        # On a quiet (zero) tick, hold the current steering with ZERO drive for a short window
        # instead of set_velocity(0,0,0): a swerve base re-centers its wheels to 0deg on a zero
        # twist (_compute_wheel_control maps 0 speed -> steering 0), which snaps the wheels
        # lateral->forward on brief intra-motion command dips; a sustained quiet re-centers
        # cleanly. Only the dispatch differs -- the PD/shape/single-axis pipeline above is
        # unchanged (see omniteleop.wbc_robot_util.base_quiet_dispatch).
        action, self._base_quiet_elapsed = base_quiet_dispatch(
            cmd, self._base_quiet_elapsed, dt, self.args.base_quiet_hold_s)
        if action == "hold":
            # Zero-drive hold of the CURRENT steering. Unlike set_velocity(0,0,0) -- which maps
            # zero speed to a FIXED steering 0deg -- set_wheel_velocity re-reads and re-commands
            # chassis.steering_angle, so a NaN/malformed read would publish a bad steer target.
            # Validate the live read first; on a bad sample fall back to the safe
            # set_velocity(0,0,0) re-center. The zero-drive hold itself is safe and exempt from
            # follower/todo.md's "Avoid" rule, which targets COMBINED motion (NONZERO velocity +
            # opposed steering -> wheels fight); at zero velocity there is no torque/scrub/fight.
            steer = np.asarray(chassis.steering_angle, dtype=float)
            if steer.shape == (2,) and np.all(np.isfinite(steer)):
                assert float(np.max(np.abs(cmd))) <= 1e-6  # base_quiet_dispatch invariant
                chassis.set_wheel_velocity(0.0)
            else:
                chassis.set_velocity(vx=0.0, vy=0.0, wz=0.0, wait_time=0.0,
                                     sequential_steering=True)
                action = "recenter"  # bad steering read -> safe re-center (logged in /debug)
        else:  # "drive" sends the active twist; "recenter" sends cmd (==0) -> steering 0deg
            chassis.set_velocity(vx=float(cmd[0]), vy=float(cmd[1]), wz=float(cmd[2]),
                                 wait_time=0.0, sequential_steering=True)
        self._dbg["base_pd_raw"] = pd_raw
        self._dbg["base_pd_err"] = np.asarray(err, dtype=float)
        self._dbg["base_cmd"] = np.asarray(cmd, dtype=float)
        self._dbg["base_action"] = action

    def debug_row(self, result) -> dict:
        """Assemble this tick's /debug record from the stashed intermediates.

        Called by run_loop AFTER actuate, so the open-loop measured-joint readback below
        runs post-command and never delays the chassis/joint write. Not-computed values
        (held ticks, base disabled) become NaN so every column keeps a consistent shape.
        """
        d = self._dbg
        widths = {"torso": 3, "left_arm": 7, "right_arm": 7, "head": 3}

        def vec(value, n):
            return (np.asarray(value, dtype=np.float32) if value is not None
                    else np.full(n, np.nan, dtype=np.float32))

        def grp(dct, g):
            return vec(None if dct is None else dct.get(g), widths[g])

        meas = d.get("meas_joints")
        if meas is None:                      # open-loop: read now (post-actuation)
            meas = self._read_measured_joints()
        odom = d.get("odom")
        row = {
            "base_twist": np.asarray(result.base_twist, dtype=np.float32),
            "base_pd_raw": vec(d.get("base_pd_raw"), 3),
            "base_pd_err": vec(d.get("base_pd_err"), 3),
            "base_cmd": vec(d.get("base_cmd"), 3),
            "base_action": str(d.get("base_action", "off")),
            "clamp_max_over": np.float32(d.get("clamp_max_over", np.nan)),
            "odom_pose": vec(odom["pose"] if odom else None, 3),
            "odom_steer": vec(odom["steer"] if odom else None, 2),
            "odom_wvel": vec(odom["wvel"] if odom else None, 2),
            "odom_age": np.float32(odom["age"] if odom else np.nan),
            "odom_drive_ts_ns": np.int64(odom["drive_ts_ns"] if odom else -1),
        }
        for g in self._joint_names:
            row[f"cmd_{g}"] = grp(d.get("cmd_joints"), g)
            row[f"sent_{g}"] = grp(d.get("sent_joints"), g)
            row[f"meas_{g}"] = grp(meas, g)
        return row

    # -- episode recording (vr_reader EpisodeRecorder format) -------------------

    def _init_gripper_monitors(self) -> None:
        """Set up the Robotiq FC03 achieved-position monitors for obs/gripper recording.

        Mirrors ``leader/vr_reader.py`` + ``follower/policy_rollout.py``: a synchronous
        warmup poll seeds ``self._last_obs_grip_{left,right}`` (gPO/255 in [0,1]), then a
        ``RobotiqStatusMonitor`` per arm queues FC03 status replies on the shared EE
        pass-through topic -- coexisting with this follower's FC16 gripper writes, which
        the monitor filters out. Only needed under ``--record`` (obs/gripper).
        """
        from omniteleop.follower.robotiq import (  # noqa: PLC0415
            RobotiqStatusMonitor,
            poll_gripper_status,
        )
        for side, arm in (("left", self.robot.left_arm), ("right", self.robot.right_arm)):
            warm = poll_gripper_status(arm, function_code=0x03, timeout_s=0.5)
            if warm is None:
                raise SystemExit(
                    f"[wbc_vr_robot] {side} gripper FC03 warmup failed -- "
                    f"obs/gripper/{side} would record NaN until the monitor catches up "
                    "(verify enable_ee_pass_through=True).")
            setattr(self, f"_last_obs_grip_{side}", float(warm["actual"]))
        self._grip_monitors = {
            "left": RobotiqStatusMonitor(self.robot.left_arm, side="left", function_code=0x03),
            "right": RobotiqStatusMonitor(self.robot.right_arm, side="right", function_code=0x03),
        }
        for monitor in self._grip_monitors.values():
            monitor.send_status_request()

    def _poll_gripper_status_step(self) -> None:
        """Drain queued FC03 replies into the obs cache, then re-request (both arms).

        Mirrors vr_reader/policy_rollout. Called at the record cadence from record_tick so
        ``self._last_obs_grip_{left,right}`` refresh at ~``--record-rate``.
        """
        for side, monitor in self._grip_monitors.items():
            events = monitor.drain_status_events()
            if events:
                setattr(self, f"_last_obs_grip_{side}", float(events[-1]["actual"]))
            monitor.expire_timeouts(0.5)
            monitor.send_status_request()

    def _await_record_cameras(self) -> None:
        """Warn (don't fail) if the --record cameras aren't streaming yet.

        head_camera + wrist_zedm are enabled in the --record Robot build. The wrist needs
        an external ZED-SDK publisher on ``sensors/wrist_zedm/*``; if it isn't up yet,
        record_tick skips frames until both stream, so warn but continue (it may start
        before the first engage).
        """
        for cam in ("head_camera", _WRIST_SENSOR_ID):
            if not self.robot.has_sensor(cam):
                raise SystemExit(
                    f"[wbc_vr_robot] --record enabled '{cam}' but it is not available on "
                    "this robot.")
            sensor = getattr(self.robot.sensors, cam)
            if hasattr(sensor, "wait_for_active") and not sensor.wait_for_active(timeout=5.0):
                extra = (f" -- needs a ZED-SDK publisher on sensors/{_WRIST_SENSOR_ID}/*"
                         if cam == _WRIST_SENSOR_ID else "")
                print(f"[wbc_vr_robot] WARNING: {cam} not active within 5s; record frames "
                      f"are skipped until it streams{extra}.")

    def _init_recording(self) -> None:
        """Set up the optional vr_reader-format episode recorder (``--record``).

        OFF unless ``--record``. When on, records (while engaged) the ``leader/vr_reader.py``
        main-HDF5 schema, EXCEPT:
          * ``action/joint`` adds ``torso`` (the WBC commands the torso here too);
          * the base action ``chassis_*`` is the shaped twist actually sent to the
            chassis (the WBC drives the base; the leader thumbstick is unused here);
          * with ``--enable base``, ``obs/joint`` adds the MEASURED base twist ``chassis_*``
            (wheel-odometry body twist, achieved-velocity counterpart to the action) and
            ``obs/base/pose`` adds the measured ``(x, y, yaw)`` base pose in the engage-origin
            world frame;
          * ``obs/images`` stores the ``intrinsic`` once (static, constant over a take) and
            does NOT store the per-frame ``extrinsic`` -- ``world_t_cam`` is recomputed offline
            from ``obs/base/pose`` + ``obs/joint`` (see ``scripts/vis_episode.py``).
        ``action/gripper`` is the leader trigger command, ``obs/gripper`` the Robotiq FC03
        achieved position, and ``obs/images`` carries head_left_rgb/head_depth/left_wrist_rgb.
        """
        self._record_period = 0.0
        self._next_record_t = 0.0
        if not self.args.record:
            return
        self._await_record_cameras()
        self._record_period = 1.0 / self.args.record_rate
        self._episode = EpisodeRecorder(self.args.save_dir)
        # Camera intrinsic is constant over a take -> store ONCE (obs/images/intrinsic is
        # (3,3), not (N,3,3)). The per-frame extrinsic is NOT stored: it was base-relative
        # head FK (base-blind, wrong once the base drives) and is fully recomputable, so it
        # is recomputed offline as world_t_cam = world_t_base(obs/base/pose) . FK_cam(obs/
        # joint) -- see scripts/vis_episode.py -- which lands the cloud in the world frame.
        self._episode.set_static({"obs": {"images": {"intrinsic": ZED_K.astype(np.float32)}}})
        print(f"[wbc_vr_robot] recording -> {self.args.save_dir} "
              f"(episode_{self._episode.episode_id}, {self.args.record_rate:g}Hz, "
              "head_left_rgb+head_depth+left_wrist_rgb, +torso action, +gripper obs/action, "
              "+base pose obs)")

    @staticmethod
    def _unwrap_frame(name: str, entry) -> tuple[np.ndarray, int]:
        """Split a ``get_obs(include_timestamp=True)`` stream entry into ``(array, frame_ns)``.

        The dexcontrol ZED sensor returns ``{"data": arr, "timestamp_ns": int, ...}`` per
        stream over Zenoh. ``frame_ns`` is the publisher's capture wall-clock, unique per
        camera frame -- record_tick compares it against the last recorded frame to catch a
        stalled/clogged publisher. Raises ValueError if the entry lacks a usable frame or a
        valid timestamp (e.g. an RTC stream, which cannot stamp frames), since the freshness
        guard must never be silently disabled.
        """
        if not isinstance(entry, dict) or "data" not in entry:
            raise ValueError(
                f"[wbc_vr_robot] {name} get_obs(include_timestamp=True) returned "
                f"{type(entry).__name__}; expected a dict with 'data' + 'timestamp_ns' "
                "(Zenoh transport). The frame-freshness guard needs the publisher timestamp."
            )
        ts = entry.get("timestamp_ns")
        if ts is None or int(ts) <= 0:
            raise ValueError(
                f"[wbc_vr_robot] {name} frame has no valid timestamp_ns ({ts!r}); cannot "
                "verify frame freshness. Ensure the ZED-SDK publisher stamps frames "
                "(tests/test_head_zedx_depth.py / test_wrist_zedm_depth.py, Zenoh transport)."
            )
        return entry["data"], int(ts)

    def _grab_head_images(self) -> Optional[tuple[np.ndarray, np.ndarray, int]]:
        """Poll the head camera -> ``(left_rgb uint8 HxWx3, depth uint16 HxW, frame_ns)`` or None.

        None until both streams have delivered a frame (parity with vr_reader's
        ``_all_selected_streams_ready`` -- so the saved HDF5 keeps consistent keys and shapes
        across frames). ``frame_ns`` is the publisher capture timestamp of the ``left_rgb``
        frame (``include_timestamp=True``); record_tick rejects a stale frame (camera stalled
        below the record rate). Raises ValueError on a malformed shape or a missing timestamp,
        so a miswired/legacy camera fails loudly instead of recording garbage or silently
        skipping the freshness check.
        """
        obs = self.robot.sensors.head_camera.get_obs(
            obs_keys=["left_rgb", "depth"], include_timestamp=True
        )
        left_entry = obs.get("left_rgb")
        depth_entry = obs.get("depth")
        if left_entry is None or depth_entry is None:
            return None
        left_rgb, frame_ns = self._unwrap_frame("head_camera left_rgb", left_entry)
        depth, _ = self._unwrap_frame("head_camera depth", depth_entry)
        left_rgb = np.asarray(left_rgb)
        depth = np.asarray(depth)
        if left_rgb.ndim != 3 or left_rgb.shape[2] != 3:
            raise ValueError(
                f"[wbc_vr_robot] head_camera left_rgb shape {left_rgb.shape} is not (H,W,3)."
            )
        if depth.ndim != 2:
            raise ValueError(
                f"[wbc_vr_robot] head_camera depth shape {depth.shape} is not (H,W)."
            )
        # meters -> millimeters, clipped to uint16 (matches vr_reader's head_depth).
        depth_u16 = np.clip(depth * 1000, 0, 65535).astype(np.uint16)
        return np.ascontiguousarray(left_rgb, dtype=np.uint8), depth_u16, frame_ns

    def _grab_wrist_image(self) -> Optional[tuple[np.ndarray, int]]:
        """Poll the wrist ZED-M left eye -> ``(left_wrist_rgb uint8 HxWx3, frame_ns)`` or None.

        Consumes the publisher's frame AS-IS (no crop/resize), like vr_reader: the wrist
        ZED-SDK publisher (tests/test_wrist_zedm_depth.py) already resized robot-side.
        Returns None until a frame arrives. ``frame_ns`` is the publisher capture timestamp
        (``include_timestamp=True``); record_tick rejects a stale frame (publisher stalled
        below the record rate). A momentary drop is deliberately NOT papered over with a
        cached frame -- a duplicate wrist image in the take is exactly what the freshness
        guard must catch. Raises ValueError on a malformed shape or a missing timestamp.
        """
        obs = self.robot.sensors.wrist_zedm.get_obs(
            obs_keys=[_WRIST_OBS_KEY], include_timestamp=True
        )
        entry = obs.get(_WRIST_OBS_KEY)
        if entry is None:  # tolerate a sensor-name-prefixed key (mirrors policy_rollout)
            for key, val in obs.items():
                if key.endswith("_" + _WRIST_OBS_KEY):
                    entry = val
                    break
        if entry is None:
            return None  # publisher delivered no frame this poll
        wrist, frame_ns = self._unwrap_frame(f"wrist_zedm {_WRIST_OBS_KEY}", entry)
        wrist = np.asarray(wrist)
        if wrist.ndim != 3 or wrist.shape[2] != 3:
            raise ValueError(
                f"[wbc_vr_robot] wrist_zedm {_WRIST_OBS_KEY} shape {wrist.shape} is not (H,W,3)."
            )
        return np.ascontiguousarray(wrist, dtype=np.uint8), frame_ns

    def record_tick(
        self,
        result,
        hold: bool,
        now: float,
        left_target: np.ndarray,
        right_target: np.ndarray,
        head_target: np.ndarray,
    ) -> None:
        """Append one vr_reader-format frame while engaged (no-op unless ``--record``).

        Throttled to ``--record-rate``; skips while ``hold`` (e-stop / failed solve /
        safety hold / stale source) so only live teleop is recorded. Must run AFTER
        :meth:`actuate` so ``self._prev_base_cmd`` holds this tick's chassis command and
        ``self._dbg['sent_joints']`` holds this tick's post-clamp joint commands.

        ``left_target`` / ``right_target`` / ``head_target`` are the WORLD-frame
        (engage-origin) 4x4 targets given to ``ik.solve()`` this tick -- the head
        target post LPF/planar-deadband -- recorded VERBATIM as ``action/eef/*`` and
        ``action/head``: the 10 Hz policy action (PLAN.md Conventions). Offline
        porting must never recompute or re-anchor them.
        """
        if self._episode is None or not self._episode.recording or hold:
            return
        # Throttle to --record-rate, locked to ideal timestamps (re-anchor if we fall
        # >1 period behind) so the saved HDF5 has a consistent FPS, like vr_reader.
        if self._next_record_t == 0.0:
            self._next_record_t = now
        if now < self._next_record_t:
            return
        self._next_record_t += self._record_period
        if now > self._next_record_t:
            self._next_record_t = now + self._record_period

        # Policy-action inputs: validate BEFORE touching cameras so a structurally
        # broken take (wrong enable mask / bad targets) aborts on the first frame.
        sent_joints = self._dbg.get("sent_joints") if hasattr(self, "_dbg") else None
        if not isinstance(sent_joints, dict):
            raise RuntimeError(
                "[wbc_vr_robot] record_tick requires post-clamp _dbg['sent_joints'] "
                "(actuate() must run before record_tick on a non-hold tick)"
            )
        left_target = np.asarray(left_target, dtype=np.float64)
        right_target = np.asarray(right_target, dtype=np.float64)
        head_target = np.asarray(head_target, dtype=np.float64)
        if (
            left_target.shape != (4, 4)
            or right_target.shape != (4, 4)
            or head_target.shape != (4, 4)
        ):
            raise RuntimeError(
                "[wbc_vr_robot] record_tick expected 4x4 policy targets, got "
                f"left={left_target.shape}, right={right_target.shape}, "
                f"head={head_target.shape}"
            )
        if (
            not np.all(np.isfinite(left_target))
            or not np.all(np.isfinite(right_target))
            or not np.all(np.isfinite(head_target))
        ):
            raise RuntimeError("[wbc_vr_robot] record_tick received a non-finite policy target")
        if self._odom is None:
            raise RuntimeError(
                "[wbc_vr_robot] record_tick requires odometry (--enable base): "
                "obs/base/pose anchors the world-frame policy action targets"
            )
        base_pose_cmd = np.asarray(result.base_pose, dtype=np.float32)
        if base_pose_cmd.shape != (3,) or not np.all(np.isfinite(base_pose_cmd)):
            raise RuntimeError(
                f"[wbc_vr_robot] record_tick result.base_pose {base_pose_cmd!r} is not a "
                "finite (3,) pose"
            )

        def action_joint(name: str, shape: tuple[int, ...]) -> np.ndarray:
            """Post-clamp sent command for group ``name`` this tick (action/joint)."""
            if name not in sent_joints:
                raise RuntimeError(
                    f"[wbc_vr_robot] record_tick missing post-clamp sent_joints[{name!r}] "
                    "-- recording requires the group in --enable"
                )
            arr = np.asarray(sent_joints[name], dtype=np.float32)
            if arr.shape != shape:
                raise RuntimeError(
                    f"[wbc_vr_robot] record_tick sent_joints[{name!r}] has shape "
                    f"{arr.shape}, expected {shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise RuntimeError(
                    f"[wbc_vr_robot] record_tick sent_joints[{name!r}] is non-finite"
                )
            return arr

        imgs = self._grab_head_images()
        wrist = self._grab_wrist_image()
        # Before the first recorded frame the cameras may still be warming up (the leader can
        # reach teleop before a publisher is fully up): tolerate a missing stream and skip
        # this tick, matching vr_reader. Once recording is underway a MISSING frame means a
        # publisher stalled below the record rate -- abort immediately rather than skip (a
        # skip would silently drop the take's cadence) or reuse a stale frame.
        recording_started = self._last_rec_head_ns >= 0
        if imgs is None or wrist is None:
            if recording_started:
                raise RuntimeError(
                    "[wbc_vr_robot] --record: a camera delivered no new frame "
                    f"(head={'ok' if imgs is not None else 'MISSING'}, "
                    f"wrist={'ok' if wrist is not None else 'MISSING'}) -- the publisher "
                    "stalled below the record rate (WiFi clog / camera fps drop). Aborting "
                    "so the take never records duplicate frames."
                )
            return  # camera still warming up -- skip rather than write inconsistent keys
        head_left_rgb, head_depth_u16, head_ns = imgs
        wrist_left_rgb, wrist_ns = wrist
        # Freshness guard: ~15 fps camera vs ~10 fps record means every recorded frame must
        # carry a strictly newer publisher timestamp than the last recorded one. A non-newer
        # frame is a stalled/clogged publisher repeating its last frame (get_obs re-returns
        # the cached latest) -> abort immediately so duplicate images never enter the take.
        if recording_started and (
            head_ns <= self._last_rec_head_ns or wrist_ns <= self._last_rec_wrist_ns
        ):
            raise RuntimeError(
                "[wbc_vr_robot] --record: stale camera frame (publisher not delivering new "
                f"frames at the record rate; WiFi clog / fps drop) -- head Δ="
                f"{head_ns - self._last_rec_head_ns} ns, wrist Δ="
                f"{wrist_ns - self._last_rec_wrist_ns} ns (need both > 0). Aborting so the "
                "take never records duplicate frames."
            )
        self._last_rec_head_ns = head_ns
        self._last_rec_wrist_ns = wrist_ns

        # Refresh achieved-gripper obs (FC03) at the record cadence (drains queued replies
        # into self._last_obs_grip_{left,right}, then re-requests).
        if self._grip_monitors:
            self._poll_gripper_status_step()

        # Measured joints -> obs/joint (also the offline FK source for world_t_cam).
        # Validate shapes aggressively -- a bad readback must abort, never be recorded as-is.
        obs: dict[str, np.ndarray] = {}
        for grp, names in self._joint_names.items():
            meas = np.asarray(self._comp(grp).get_joint_pos(), dtype=np.float32)
            if meas.shape != (len(names),) or not np.all(np.isfinite(meas)):
                raise ValueError(
                    f"[wbc_vr_robot] {grp} joint readback {meas.shape} is not "
                    f"({len(names)},) or non-finite -- cannot record."
                )
            obs[grp] = meas

        base_cmd = self._prev_base_cmd  # shaped twist actually sent to the chassis
        # obs/joint base = the MEASURED wheel-odometry body twist (achieved-velocity
        # counterpart to action/joint/chassis_*), recorded only when the base is actuated
        # (--enable base; otherwise there is no odometry thread). Read once here so all
        # frames in a take agree on keys -- the enable mask is fixed for the whole run.
        obs_joint: dict[str, np.ndarray] = {
            "left_arm": obs["left_arm"],
            "right_arm": obs["right_arm"],
            "head": obs["head"],
            "torso": obs["torso"],
        }
        if self._odom is not None:
            meas_twist = self._odom.twist
            obs_joint["chassis_vx"] = np.float32(meas_twist[0])
            obs_joint["chassis_vy"] = np.float32(meas_twist[1])
            obs_joint["chassis_wz"] = np.float32(meas_twist[2])
        images = {
            "head_left_rgb": head_left_rgb,
            "head_depth": head_depth_u16,
            "left_wrist_rgb": wrist_left_rgb,
            # intrinsic is recorded once via set_static (not per frame); extrinsic is NOT
            # recorded -- world_t_cam is recomputed offline from obs/base/pose + obs/joint.
        }
        obs_out: dict = {
            "joint": obs_joint,
            # Gripper obs = the Robotiq FC03 achieved position (gPO/255 in [0,1]), refreshed
            # above at the record cadence; NaN until the monitor first replies (matches
            # vr_reader's obs/gripper).
            "gripper": {
                "left": np.float32(self._last_obs_grip_left),
                "right": np.float32(self._last_obs_grip_right),
            },
            "images": images,
        }
        if self._odom is not None:
            # Measured base pose (x, y, yaw) in the engage-origin world frame (the odom is
            # zeroed on the engage edge), the pose counterpart to the obs/joint/chassis_*
            # body twist. Lets the cloud/camera/robot be placed in world frame offline:
            # world_t_cam = world_t_base(pose) . FK_cam(obs/joint); see scripts/vis_episode.py.
            # Only present with --enable base (no odom thread otherwise); the enable mask is
            # fixed for the whole run, so every frame in a take agrees on keys.
            obs_out["base"] = {"pose": np.asarray(self._odom.pose, dtype=np.float32)}
        frame = {
            "timestamp_ns": np.int64(time.time_ns()),
            "action": {
                "joint": {
                    # torso added vs vr_reader: the WBC commands the torso here. Every
                    # group records the POST-CLAMP command actually sent to hardware
                    # (actuate's _dbg['sent_joints']), not the raw WBC result.
                    "torso": action_joint("torso", (3,)),
                    "left_arm": action_joint("left_arm", (7,)),
                    "right_arm": action_joint("right_arm", (7,)),
                    "head": action_joint("head", (3,)),
                    # Base action = the shaped twist sent to the chassis (the WBC drives
                    # the base; the leader thumbstick chassis_* is unused here).
                    "chassis_vx": np.float32(base_cmd[0]),
                    "chassis_vy": np.float32(base_cmd[1]),
                    "chassis_wz": np.float32(base_cmd[2]),
                },
                # Gripper action = the clipped leader trigger commanded this tick (stashed
                # by actuate), matching vr_reader's action/gripper.
                "gripper": {
                    "left": np.float32(self._last_gripper_cmd_left),
                    "right": np.float32(self._last_gripper_cmd_right),
                },
                # 10 Hz policy action: the EXACT world-frame targets given to ik.solve()
                # this tick (head post LPF/planar-deadband), stored verbatim -- offline
                # porting must never recompute or re-anchor them (PLAN.md Conventions).
                "eef": {
                    "left": left_target.astype(np.float32),
                    "right": right_target.astype(np.float32),
                },
                "head": head_target.astype(np.float32),
                # Solver's commanded base pose in the same engage-origin world as
                # obs/base/pose. NOT a policy field: free to record, and lets offline
                # analysis derive base-frame variants and IK-vs-odom tracking error.
                "base": {"pose": base_pose_cmd},
            },
            "obs": obs_out,
        }
        self._episode.record(frame)

    def close(self) -> None:
        """Stop the base, the odom thread, and shut the robot down (best-effort).

        Safe to call on a partially-initialized driver (robot/odom may be None) so the
        engage-time teardown works even if __init__ failed mid-setup.
        """
        self.stop_all_motion()
        if self._odom is not None:
            self._odom.stop()
        # Flush the episode BEFORE shutting the robot down. EpisodeRecorder saves in a
        # daemon thread, so block here until it finishes -- otherwise process exit could
        # kill the save mid-write.
        if self._episode is not None and self._episode.recording:
            n = self._episode.num_frames()
            path = self._episode.stop()
            if path is None:
                print("\n[wbc_vr_robot] recording: 0 frames -- nothing saved.")
            else:
                print(f"\n[wbc_vr_robot] saving {n} frames -> {path} ...")
                while self._episode.saving:
                    time.sleep(0.05)
                print(f"[wbc_vr_robot] episode saved -> {path}")
        # Best-effort teardown of the FC03 gripper-status subscribers.
        for monitor in getattr(self, "_grip_monitors", {}).values():
            try:
                monitor.close()
            except Exception:
                pass
        try:
            if self.robot is not None:
                self.robot.shutdown()
        except Exception:
            pass


def _build_ik() -> tuple[VegaWholeBodyIK, WBCConfig]:
    cfg = WBCConfig()
    # head_mode "ik" pins cfg.head_ik_pinned_joints (wbik.yaml; default head_j1/head_j2)
    # out of the QP -- VegaWholeBodyIK builds the pin from the config, so this real-robot
    # follower and the sim follower (wbc_vr_record.py) fix the SAME head DOFs.
    ik = VegaWholeBodyIK(cfg)
    if cfg.head_mode == "ik":
        print(f"[wbc_vr_robot] head IK pin (wbik.yaml): {list(cfg.head_ik_pinned_joints)} "
              "fixed; other head DOFs tracked.")
    ik.reset()
    return ik, cfg


def run_loop(
    source,
    ik: VegaWholeBodyIK,
    cfg: WBCConfig,
    driver,
    args: argparse.Namespace,
    *,
    replay: bool,
    clock: Callable[[], float],
    realtime: bool,
    enable: dict,
    traj: Optional[_TrajLog] = None,
) -> None:
    """Drive the IK from ``source`` (live or replay), dispatching actuation to ``driver``.

    The clock is injectable so a dry run can advance a virtual clock as fast as the CPU
    allows while the replay source still releases frames on the (scaled) schedule.
    """
    dt = 1.0 / args.ik_rate
    left0 = _to_mat(ik.frame_pose(LEFT_EE_FRAME))
    right0 = _to_mat(ik.frame_pose(RIGHT_EE_FRAME))
    head0 = _to_mat(ik.frame_pose(HEAD_FRAME))

    speed = args.speed if replay else 1.0
    base_dur = (1.0 / args.cmd_rate) / speed if args.cmd_rate > 0 else 0.0
    interp = TargetInterpolator(base_dur, left0, right0, head0)
    head_lpf = HeadTargetLowPassFilter(args.head_lpf_tau, head0)
    head_planar_deadband = HeadTargetPlanarDeadbandFilter(
        head0,
        position_deadband=args.head_planar_pos_deadband,
        yaw_deadband=args.head_planar_yaw_deadband,
    )

    left_cmd, right_cmd, head_cmd = left0.copy(), right0.copy(), head0.copy()
    last_cmd_ns = -1
    last_cmd_wall: Optional[float] = None
    prev_estop = True
    source.start()
    status_pub = None if replay else _create_status_publisher(source)
    status_period = 1.0 / DEFAULT_STATUS_PUBLISH_RATE
    last_status_publish = -float("inf")
    t0 = clock()
    last_print = 0.0
    print(f"[wbc_vr_robot] {driver.name} loop @ {args.ik_rate:g}Hz IK "
          f"({'replay ' + format(args.speed, 'g') + 'x' if replay else 'live'}); "
          f"enable={[k for k, v in enable.items() if v]} grippers=on")

    while True:
        now = clock()
        t = now - t0
        if replay and source.done:
            break
        vr = source.latest
        if vr is not None and vr.exit_requested:
            # Left X is an immediate stop request, not a training frame: stop motion
            # before recorder shutdown, and do not append the exit/stop state.
            driver.stop_all_motion()
            print("\n[wbc_vr_robot] leader requested exit; motion stopped.")
            break

        estop = bool(vr.estop) if vr is not None else True
        if prev_estop and not estop:
            ik.reset()
            interp.reset(left0, right0, head0)
            head_lpf.reset(head0)
            head_planar_deadband.reset(head0)
            left_cmd, right_cmd, head_cmd = left0.copy(), right0.copy(), head0.copy()
            last_cmd_ns = -1
            driver.engage_reset(ik, left0, right0, head0)
            print("\n[wbc_vr_robot] engage: reset IK to nominal.")
        prev_estop = estop

        left_cmd, right_cmd = vr_to_ee_targets(vr, left_cmd, right_cmd)
        head_cmd = vr_to_head_target(vr, head_cmd)
        if (vr is not None and vr.timestamp_ns != last_cmd_ns
                and vr.left_ee_pose and vr.right_ee_pose):
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
            last_cmd_ns = vr.timestamp_ns
            last_cmd_wall = now

        left_target, right_target, head_target = interp.at(now)
        head_target = head_lpf.filter(head_target, dt)
        head_target = head_planar_deadband.filter(head_target)
        if cfg.head_mode == "ik":
            result = ik.solve(left_target, right_target, dt, head_target=head_target)
        else:
            head_joints = ik.solve_head(head_target, dt)
            result = ik.solve(left_target, right_target, dt, head_joints=head_joints)

        hold_reason = driver.extra_hold(now, last_cmd_wall, estop)
        hold = estop or (not result.success) or result.held or (hold_reason is not None)
        left_gripper = vr.left_gripper if vr is not None else 0.0
        right_gripper = vr.right_gripper if vr is not None else 0.0
        driver.actuate(result, left_gripper, right_gripper, enable, hold, dt)
        if hasattr(driver, "start_recording_if_teleop"):
            driver.start_recording_if_teleop(vr)
        if hasattr(driver, "record_tick"):
            driver.record_tick(
                result,
                hold,
                now,
                left_target=left_target,
                right_target=right_target,
                head_target=head_target,
            )
        if status_pub is not None and now - last_status_publish >= status_period:
            status = _build_follower_status(
                vr,
                result,
                estop=estop,
                hold=hold,
                hold_reason=hold_reason or "",
            )
            status_pub.publish(asdict(status))
            last_status_publish = now

        if traj is not None:
            dbg = driver.debug_row(result) if hasattr(driver, "debug_row") else {}
            traj.append(
                t=t, cmd_ns=int(last_cmd_ns), estop=bool(estop),
                success=bool(result.success), held=bool(result.held),
                hold=bool(hold), hold_reason=hold_reason or "",
                left_ee_error=float(result.left_ee_error),
                right_ee_error=float(result.right_ee_error),
                stability_margin=float(result.stability_margin),
                safety_status=str(result.safety_status),
                q=np.asarray(result.q, dtype=np.float32),
                base_pose=np.asarray(result.base_pose, dtype=np.float32),
                left_target=left_target.astype(np.float32),
                right_target=right_target.astype(np.float32),
                head_target=head_target.astype(np.float32),
                **dbg,
            )

        if t - last_print >= 0.5:
            last_print = t
            stage = vr.calib_stage if vr is not None else "?"
            if hold_reason is not None:
                state = f"HOLD:{hold_reason}"
            elif estop:
                state = f"hold({stage})"
            else:
                state = "teleop"
            print(f"t={t:6.1f}s  {state:14s}  errL={result.left_ee_error * 1000:5.1f}mm  "
                  f"errR={result.right_ee_error * 1000:5.1f}mm  "
                  f"margin={result.stability_margin * 100:+4.1f}cm  "
                  f"{result.safety_status}", end="\r")

        if realtime:
            sleep = dt - (clock() - now)
            if sleep > 0:
                time.sleep(sleep)
        else:
            clock.advance(dt)


def _run_ik_mode(args: argparse.Namespace, enable: dict) -> None:
    """Replay or live-drive the IK on the real robot."""
    ik, cfg = _build_ik()
    print(f"[wbc_vr_robot] model nq={ik.model.nq} nv={ik.model.nv} head_mode={cfg.head_mode} "
          f"urdf={cfg.urdf_path.rsplit('/', 1)[-1]}")
    replay = args.replay is not None
    clock: Callable[[], float] = time.perf_counter
    realtime = True

    if replay:
        source = ReplaySource.from_hdf5(args.replay, args.speed, clock=clock)
        print(f"[wbc_vr_robot] replay {args.replay}: {len(source._vr)} frames, "  # noqa: SLF001
              f"{source.total_duration:.1f}s at {args.speed:g}x")
    else:
        source = VRJointSubscriber(args.namespace, name="wbc_vr_robot")
        print(f"[wbc_vr_robot] live on '{source.topic}'. Ctrl-C to stop.")

    enabled = [k for k, v in enable.items() if v]
    debug_path: Optional[str] = None
    traj: Optional[_TrajLog] = None
    if args.debug_dir is not None:
        import os  # noqa: PLC0415

        os.makedirs(args.debug_dir, exist_ok=True)
        # Match vr_reader's DebugEpisodeRecorder naming. When --record is also on,
        # peek save_dir so episode_<N>_debug.hdf5 pairs with episode_<N>.hdf5.
        if args.record:
            episode_id = peek_next_episode_id(args.save_dir)
        else:
            episode_id = peek_next_episode_id(args.debug_dir, suffix="_debug")
        debug_path = os.path.join(args.debug_dir, f"episode_{episode_id}_debug.hdf5")
        meta = {
            "episode_id": int(episode_id),
            "mode": "replay" if replay else "live",
            "speed": float(args.speed) if replay else 1.0,
            "ik_rate": float(args.ik_rate),
            "cmd_rate": float(args.cmd_rate),
            "enable": ",".join(enabled),
            "urdf": cfg.urdf_path.rsplit("/", 1)[-1],
            "base_kp_xy": float(args.base_kp_xy),
            "base_kp_yaw": float(args.base_kp_yaw),
            "base_deadband": float(args.base_deadband),
            "base_accel": float(args.base_accel),
            "base_max_speed": float(args.base_max_speed),
            "base_quiet_hold_s": float(args.base_quiet_hold_s),
            "source_timeout": float(args.source_timeout),
            "max_joint_step": float(args.max_joint_step),
            "head_lpf_tau": float(args.head_lpf_tau),
            "head_planar_pos_deadband": float(args.head_planar_pos_deadband),
            "head_planar_yaw_deadband": float(args.head_planar_yaw_deadband),
            "base_post_linear_deadband": float(args.base_post_linear_deadband),
            "base_post_angular_deadband": float(args.base_post_angular_deadband),
            "replay_file": str(args.replay) if replay else "",
        }
        traj = _TrajLog(debug_path, meta=meta)
    print("=" * 72)
    print(f"[wbc_vr_robot] REAL ROBOT. enabled={enabled} grippers=on | "
          f"{'REPLAY ' + format(args.speed, 'g') + 'x' if replay else 'LIVE'} | "
          f"base_max={args.base_max_speed:g}m/s")
    print(f"  filters -> head_lpf={args.head_lpf_tau:g}s, "
          f"head_planar_deadband={args.head_planar_pos_deadband:g}m/"
          f"{args.head_planar_yaw_deadband:g}rad, "
          f"base_post_deadband={args.base_post_linear_deadband:g}m/s/"
          f"{args.base_post_angular_deadband:g}rad/s")
    print(f"  joints -> dexcontrol 1:1 (urdf {cfg.urdf_path.rsplit('/', 1)[-1]}). "
          "Robot homes to nominal, then moves. KEEP CLEAR. Ctrl-C aborts (zeros base).")
    if debug_path is not None:
        print(f"  /debug -> {debug_path}")
    if args.record:
        print(f"  recording -> {args.save_dir} @ {args.record_rate:g}Hz while engaged "
              "(head_left_rgb+head_depth+left_wrist_rgb, +torso action, +gripper obs/action).")
    print("=" * 72)

    driver = None
    try:
        # Construct INSIDE the try so a HardwareDriver init failure (e.g. failed homing
        # gate) still runs the finally cleanup. HardwareDriver.__init__ also self-cleans.
        driver = HardwareDriver(args, ik, cfg, enable)
        run_loop(source, ik, cfg, driver, args, replay=replay, clock=clock,
                 realtime=realtime, enable=enable, traj=traj)
    except KeyboardInterrupt:
        print("\n[wbc_vr_robot] interrupted -- stopping.")
    finally:
        if hasattr(source, "close"):
            source.close()
        if driver is not None:
            driver.close()
        if traj is not None:
            print(f"\n[wbc_vr_robot] summary: {traj.summary()}")
            traj.write()
            print(f"[wbc_vr_robot] /debug log -> {debug_path} ({len(traj)} ticks)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--replay", metavar="FILE",
                      help="drive the robot from a stream HDF5 written by wbc_vr_record.py.")
    mode.add_argument("--source", choices=("live",), default=None,
                      help="drive from the live leader (default when --replay is omitted).")
    parser.add_argument("--enable", default="arms,torso,head,base",
                        help="comma list of DOF groups to actuate: any of "
                             "arms,torso,head,base (or 'all'/'none'). Default "
                             "'arms,torso,head,base'. Unselected groups are still solved by IK"
                             "but not sent to that actuator. The grippers are always "
                             "active (tracking the leader triggers), independent of this mask.")
    parser.add_argument("--namespace", default="",
                        help="Zenoh namespace (must match the leader; default empty).")
    parser.add_argument("--debug-dir", dest="debug_dir", default=None,
                        help="write per-tick /debug HDF5 logs (base PD chain, odom + raw "
                             "wheels, joint cmd/sent/measured) under this "
                             "directory as episode_<N>_debug.hdf5 (same id as --record "
                             "episodes when both are set). OFF by default.")
    parser.add_argument("--record", action="store_true",
                        help="record an episode (vr_reader EpisodeRecorder format) while "
                             "engaged: action+obs joints (incl. torso), gripper obs/action, "
                             "head_left_rgb/head_depth/left_wrist_rgb, a static intrinsic, and "
                             "(with --enable base) the obs/base/pose used to rebuild "
                             "world_t_cam offline. The wrist camera needs an external ZED-SDK "
                             "publisher on sensors/wrist_zedm/*. OFF by default.")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR,
                        help=f"directory for recorded episodes (default {DEFAULT_SAVE_DIR}).")
    parser.add_argument("--record-rate", type=float, default=DEFAULT_RECORD_RATE,
                        help=f"episode record cadence in Hz (default {DEFAULT_RECORD_RATE:g}; "
                             "must be <= the IK rate (ik_rate in wbik.yaml)).")

    hw = parser.add_argument_group("hardware")
    hw.add_argument("--home-tol", type=float, default=DEFAULT_HOME_TOL,
                    help=f"max measured-vs-nominal joint error (rad) to pass the homing gate "
                         f"(default {DEFAULT_HOME_TOL:g}). Tolerates position-control "
                         "steady-state droop; the --max-joint-step clamp smooths the residual "
                         "at engage.")
    hw.add_argument("--home-settle", type=float, default=DEFAULT_HOME_SETTLE,
                    help=f"seconds to hold each group at nominal after the homing ramp, "
                         f"blocking until joints converge within --home-tol (default "
                         f"{DEFAULT_HOME_SETTLE:g}; 0 disables). Drains the ramp's tracking "
                         "lag so the gate snapshot is settled, not mid-flight.")
    hw.add_argument("--max-joint-step", type=float, default=DEFAULT_MAX_JOINT_STEP,
                    help=f"per-tick joint command clamp (rad), arms/torso/head (default "
                         f"{DEFAULT_MAX_JOINT_STEP:g}; 0 disables). Repeatedly exceeding 2x "
                         "this aborts the run.")
    hw.add_argument("--source-timeout", type=float, default=DEFAULT_SOURCE_TIMEOUT,
                    help=f"hold (zero base, freeze joints) if no fresh command/odom for this "
                         f"many seconds (default {DEFAULT_SOURCE_TIMEOUT:g}).")
    hw.add_argument("--drive-state-mode", choices=("ms", "rad"), default="ms",
                    help="swerve wheel_velocity units for odometry (default 'ms').")
    hw.add_argument("--base-quiet-hold-s", type=float, default=DEFAULT_BASE_QUIET_HOLD_S,
                    help="on a quiet (zero) base command, hold the current swerve steering "
                         "(zero drive) for this long before re-centering the wheels to 0deg "
                         f"(default {DEFAULT_BASE_QUIET_HOLD_S:g}; 0 = re-center on every quiet "
                         "tick, the original behavior).")
    args = parser.parse_args()
    # Control-loop tunables sourced SOLELY from wbik.yaml's vr_teleop: block
    # (VRTeleopConfig, loaded into the DEFAULT_* constants above); no longer
    # CLI-overridable. Bind them onto args -- the same post-parse mutation the script
    # already does for --debug-dir -- so run_loop / HardwareDriver / the /debug meta read
    # one namespace and YAML stays the single source. VRTeleopConfig.from_yaml already
    # validated each value (finite, >= 0, or > 0 for replay_speed/ik_rate).
    args.speed = DEFAULT_SPEED
    args.ik_rate = DEFAULT_IK_RATE
    args.cmd_rate = DEFAULT_CMD_RATE
    args.head_lpf_tau = DEFAULT_HEAD_LPF_TAU
    args.head_planar_pos_deadband = DEFAULT_HEAD_PLANAR_POS_DEADBAND
    args.head_planar_yaw_deadband = DEFAULT_HEAD_PLANAR_YAW_DEADBAND
    args.base_kp_xy = DEFAULT_BASE_KP_XY
    args.base_kp_yaw = DEFAULT_BASE_KP_YAW
    args.base_deadband = DEFAULT_BASE_DEADBAND
    args.base_accel = DEFAULT_BASE_ACCEL
    args.base_max_speed = DEFAULT_BASE_MAX_SPEED
    args.base_post_linear_deadband = DEFAULT_BASE_POST_LINEAR_DEADBAND
    args.base_post_angular_deadband = DEFAULT_BASE_POST_ANGULAR_DEADBAND

    for flag, val in (("--home-tol", args.home_tol), ("--max-joint-step", args.max_joint_step),
                      ("--home-settle", args.home_settle),
                      ("--base-quiet-hold-s", args.base_quiet_hold_s)):
        if not np.isfinite(val) or val < 0.0:
            parser.error(f"{flag} must be finite and >= 0")
    for flag, val in (("--source-timeout", args.source_timeout),
                      ("--record-rate", args.record_rate)):
        if not np.isfinite(val) or val <= 0.0:
            parser.error(f"{flag} must be finite and > 0")
    if args.record and args.record_rate > args.ik_rate:
        parser.error("--record-rate must be <= the IK rate (ik_rate in wbik.yaml; "
                     "cannot record faster than the loop)")
    try:
        enable = parse_enable_mask(args.enable)
    except ValueError as exc:
        parser.error(str(exc))

    _run_ik_mode(args, enable)


if __name__ == "__main__":
    main()
