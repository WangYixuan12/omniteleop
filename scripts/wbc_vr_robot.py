#!/usr/bin/env python3
"""Real-robot whole-body VR follower for ``wbc_vr_leader.py`` -- staged-safety bring-up.

Same control core as the SAPIEN sim follower (``scripts/wbc_vr_record.py``) -- subscribe
to the leader's Cartesian ``L_ee``/``R_ee``/head TARGET poses, interpolate the low-rate
command stream up to the IK rate, run ``VegaWholeBodyIK`` -- but driving the REAL Vega
hardware (``dexcontrol.robot.Robot``) instead of a simulator.

To bring this up safely we DON'T jump straight to live teleop. First use
``scripts/wbc_vr_record.py`` to visualize in SAPIEN and save one HDF5 file: the replayable
leader stream lives at the file root, and debug data lives under ``/debug``. This script
then has two real-robot modes:

  1. ``--replay FILE``   replay that SAPIEN-verified stream, TIME-SCALED (``--speed``,
                         default 0.25), through the same whole-body IK on the robot.
  2. ``--source live``   subscribe to the live leader and drive the robot in real time
                         (only after the replay path is verified).

``--enable`` selects which DOF groups actually actuate (default ``arms,torso,head,base``);
grippers stay OFF unless ``--grippers`` is given. Unselected groups are still solved by the
IK but not sent to that actuator.

Both real-robot modes always write a per-tick ``/debug`` HDF5 (auto-named
``wbc_debug_<mode>_<timestamp>.hdf5`` next to the replay file, or override with
``--traj-out``): the base PD chain (odom pose + raw wheels, IK base pose/twist, PD
raw/err, shaped command), the closed-loop seed, and per-group joint command/sent/measured
-- enough to diff a ``--closed-loop-q`` run against an open-loop one.

``--record`` saves an episode while engaged, in the same ``EpisodeRecorder`` schema as
``leader/vr_reader.py`` (so takes are interchangeable): ``action``/``obs`` joints plus
``head_left_rgb``/``head_depth`` (+ ``intrinsic``/``extrinsic``). It adds ``torso`` to
``action/joint`` (the WBC commands the torso) and uses the shaped base twist for the
``chassis_*`` action. Grippers and the wrist camera are NOT recorded (not installed) --
that code is kept in comments to restore once the hardware is mounted.

Run in the dexmate conda env (pinocchio + pink + dexcomm + dexcontrol)::

    # 1) record and visualize in SAPIEN (no real robot):
    /home/yixuan/miniforge3/envs/dexmate/bin/python scripts/wbc_vr_record.py \
        --hdf5 /tmp/demo.hdf5

    # 2) inspect the saved take in SAPIEN before hardware:
    /home/yixuan/miniforge3/envs/dexmate/bin/python scripts/wbc_vr_record.py \
        --replay /tmp/demo.hdf5 --speed 0.25 --output /tmp/demo_replay.mp4

    # 3) replay slowly ON THE ROBOT (after verifying step 2), recording a take:
    /home/yixuan/miniforge3/envs/dexmate/bin/python scripts/wbc_vr_robot.py \
        --replay /tmp/demo.hdf5 --speed 0.25 --record
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Callable, Optional

import numpy as np

from omniteleop.common.head_camera import ZED_K
from omniteleop.common.recorder import EpisodeRecorder
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
from omniteleop.wbc_robot_util import parse_enable_mask
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
DEFAULT_BASE_KP_XY = _VR_TELEOP.base_kp_xy          # closed-loop base PD gains (real-robot values)
DEFAULT_BASE_KP_YAW = _VR_TELEOP.base_kp_yaw
DEFAULT_BASE_ACCEL = _VR_TELEOP.base_accel          # m/s^2 base-twist slew (ang = 2x)
DEFAULT_BASE_DEADBAND = _VR_TELEOP.base_deadband    # m/s base-twist deadband (ang = 2x)
DEFAULT_HEAD_PLANAR_POS_DEADBAND = _VR_TELEOP.head_planar_pos_deadband
DEFAULT_HEAD_PLANAR_YAW_DEADBAND = _VR_TELEOP.head_planar_yaw_deadband
DEFAULT_BASE_POST_LINEAR_DEADBAND = _VR_TELEOP.base_post_linear_deadband
DEFAULT_BASE_POST_ANGULAR_DEADBAND = _VR_TELEOP.base_post_angular_deadband

# Episode recording (--record): vr_reader EpisodeRecorder format so takes are
# interchangeable with VR-teleop recordings. Default cadence matches vr_reader's
# record_rate; the default dir is the same raw_data tree the leader writes to.
DEFAULT_RECORD_RATE = 15.0
DEFAULT_SAVE_DIR = "/home/yixuan/omniteleop/Dexmate/data/raw_data"

# Hardware safety / bring-up defaults (robot-specific; conservative for a first slow
# bring-up). The shared closed-loop base-control tunables (PD gains, slew, deadband,
# post-deadband) come from the vr_teleop: block above; only the low first-bring-up base
# velocity clamp and the homing/joint-step/watchdog knobs live here.
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
DEFAULT_BASE_MAX_SPEED = 0.30   # m/s base velocity clamp (low for first bring-up; ang = 2x)
_JOINT_STEP_ABORT_TICKS = 25    # consecutive ticks demanding > 2x clamp -> abort


class _TrajLog:
    """Collect per-tick dry-run IK state; summarize and optionally write to HDF5."""

    def __init__(self, path: Optional[str], meta: Optional[dict] = None) -> None:
        self.path = path
        self.meta = meta or {}
        self._rows: list[dict] = []

    def append(self, **fields) -> None:
        """Record one IK tick."""
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
        "t", "cmd_ns", "estop", "success", "held", "hold", "closed_loop_active",
        "left_ee_error", "right_ee_error", "stability_margin", "clamp_max_over",
        "odom_age", "odom_drive_ts_ns",
    )
    _VECTOR_KEYS = (
        "q", "seed_q", "base_pose", "base_twist", "base_pd_raw", "base_pd_err",
        "base_cmd", "odom_pose", "odom_steer", "odom_wvel",
        "left_target", "right_target", "head_target",
        "cmd_torso", "cmd_left_arm", "cmd_right_arm", "cmd_head",
        "sent_torso", "sent_left_arm", "sent_right_arm", "sent_head",
        "meas_torso", "meas_left_arm", "meas_right_arm", "meas_head",
    )
    _STR_KEYS = (("safety_status", 96), ("hold_reason", 32))

    def write(self) -> None:
        """Write the collected ticks under a ``/debug`` group (if a path is set).

        ``/debug`` holds every per-tick column (scalars, gzipped vectors, fixed-width
        strings) plus run metadata in its attrs, so each file self-describes which run
        produced it (e.g. ``closed_loop_q``). Columns absent this run (no driver) are
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
    re-anchoring matches the hardware), optionally seeds the IK from the MEASURED state
    each tick (``--closed-loop-q``), clamps per-tick joint steps, and runs the same
    ``base_closed_loop`` PD/shape path the real robot uses in ``drive_box_record.py``.
    Every actuator is gated by the ``--enable`` mask; grippers stay off unless
    ``--grippers``. On any hold (e-stop / failed solve / safety hold / stale source or
    odom) the base is zeroed and joints are frozen.
    """

    name = "hardware"

    def __init__(self, args: argparse.Namespace, ik: VegaWholeBodyIK, cfg: WBCConfig,
                 enable: dict) -> None:
        from dexbot_utils import RobotInfo  # noqa: PLC0415 -- hardware-only deps, lazy
        from dexcontrol.robot import Robot  # noqa: PLC0415

        from omniteleop.follower import base_closed_loop as base_cl  # noqa: PLC0415
        from omniteleop.follower.robotiq import build_hande_command, send_activate  # noqa: PLC0415
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
        self._base_cl = base_cl
        self._build_hande = build_hande_command
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
        self._prev_base_cmd = np.zeros(3)
        self._prev_cmd: dict = {grp: None for grp in self._joint_names}
        self._overstep_ticks = 0
        # Per-tick /debug scratch: measured_q / actuate / _drive_base stash their
        # intermediates here and debug_row() assembles them after actuation. Touched
        # only by the (single-threaded) follower loop.
        self._dbg: dict = {}

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
                # Recording needs the head camera streaming, so build the Robot with
                # head_camera enabled (mirrors leader/vr_reader.py). The WRIST camera is
                # intentionally NOT enabled -- it is not installed; to record it later,
                # inject/enable a wrist_zedm ZedXCameraConfig here (see vr_reader) and
                # uncomment the wrist branches in _grab_head_images / record_tick below.
                from dexcontrol.core.config import get_robot_config  # noqa: PLC0415
                configs = get_robot_config()
                if "head_camera" not in configs.sensors:
                    raise SystemExit(
                        "[wbc_vr_robot] --record needs a head_camera sensor but the robot "
                        "config has none."
                    )
                configs.sensors["head_camera"].enabled = True
                self.robot = Robot(configs=configs)
            else:
                self.robot = Robot()
            nq = ik.nominal_q()
            self._nominal = {
                grp: np.array([nq[ik._idx_q[n]] for n in names])  # noqa: SLF001
                for grp, names in self._joint_names.items()
            }
            self._init_recording()
            if args.grippers:
                print("[wbc_vr_robot] activating grippers ...")
                send_activate(self.robot.left_arm)
                send_activate(self.robot.right_arm)
            if enable["base"]:
                if not self.has_chassis:
                    raise SystemExit("[wbc_vr_robot] --enable base but the robot has no chassis.")
                self._wait_chassis()
                self._odom = OdometryThread(self.robot, drive_state_mode=args.drive_state_mode)
                self._odom.start()
            self._home_to_nominal()
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

    def _wait_chassis(self, timeout: float = 10.0) -> None:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if np.asarray(self.robot.chassis.steering_angle).shape == (2,):
                return
            time.sleep(0.05)
        raise SystemExit(f"[wbc_vr_robot] no chassis state within {timeout:.0f}s.")

    def _interp_joint(self, grp: str, waypoints: list, step: float = 0.01) -> None:
        comp = self._comp(grp)
        cur = np.asarray(comp.get_joint_pos(), dtype=float)
        for wp in waypoints:
            wp = np.asarray(wp, dtype=float)
            n = max(1, int(np.max(np.abs(wp - cur)) / step))
            for i in range(n):
                q = cur + (wp - cur) * (i + 1) / n
                comp.set_joint_pos(q.tolist(), wait_time=0.1, exit_on_reach=True)
            cur = wp

    def _home_to_nominal(self) -> None:
        # Only home the ENABLED groups (a disabled group is not actuated, so moving it
        # would be surprising); the IK still models disabled groups at nominal, so they
        # are ASSUMED already there -- a limited-test caveat for partial --enable.
        on = {"torso": self.enable["torso"], "left_arm": self.enable["arms"],
              "right_arm": self.enable["arms"], "head": self.enable["head"]}
        nom = {grp: self._to_hw(grp, self._nominal[grp]) for grp in self._joint_names}
        homing = [g for g in self._joint_names if on[g]]
        skipped = [g for g in self._joint_names if not on[g]]
        print(f"[wbc_vr_robot] homing to WBC nominal (move clear; watch the robot). "
              f"homing={homing}" + (f"; assuming-at-nominal={skipped}" if skipped else ""))
        if on["torso"]:
            self._interp_joint("torso", [nom["torso"]])
        # Drive arms straight to nominal (no SAFE_* waypoint). Use a finer step so the
        # direct path is traversed slowly -- there is no intermediate safe pose, so the
        # slow glide lets the operator watch and Ctrl-C if it tracks toward a collision.
        if on["left_arm"]:
            self._interp_joint("left_arm", [nom["left_arm"]], step=ARM_HOME_STEP)
        if on["right_arm"]:
            self._interp_joint("right_arm", [nom["right_arm"]], step=ARM_HOME_STEP)
        if on["head"]:
            self._interp_joint("head", [nom["head"]])
        # Settle each enabled group at nominal: the stepped ramp can outrun the arm (each
        # dexcontrol substep returns after its wait_time whether or not the joint caught
        # up, so lag accumulates), so block here until the MEASURED joints are within
        # --home-tol (dexcontrol's reached-check uses the same max-abs metric as the gate
        # below) or --home-settle elapses -- draining the lag before the gate snapshot.
        if self.args.home_settle > 0:
            for grp in self._joint_names:
                if on[grp]:
                    self._comp(grp).set_joint_pos(
                        nom[grp].tolist(), wait_time=self.args.home_settle,
                        exit_on_reach=True,
                        exit_on_reach_kwargs={"tolerance": self.args.home_tol},
                    )
        bad = []
        for grp in self._joint_names:
            if not on[grp]:
                continue
            meas = np.asarray(self._comp(grp).get_joint_pos(), dtype=float)
            err = float(np.max(np.abs(meas - nom[grp])))
            if err > self.args.home_tol:
                bad.append(f"{grp} {err:.3f}rad")
        if bad:
            raise SystemExit(
                "[wbc_vr_robot] homing gate FAILED (measured vs WBC nominal): "
                + ", ".join(bad) + f" > --home-tol {self.args.home_tol}. Aborting before engage."
            )
        print("[wbc_vr_robot] homing gate OK: enabled groups are at WBC nominal.")
        # Center the swerve wheels to straight before engage (only if the base is
        # actuated). _compute_wheel_control maps zero velocity to steering 0, so the
        # run-loop's pre-engage holds already command steering->0 each tick, but with no
        # settle window; do it explicitly here so the steer joints are physically straight
        # (and given time to converge) before any base twist, instead of swinging from a
        # stale angle on the first engaged set_velocity. set_steering_angle commands the
        # steer joints to 0 with zero wheel velocity; wait_time holds the command so they
        # converge (--home-settle, repeated for ~max(wait_time-1, 0)s; 0 = single shot).
        if self.enable["base"] and self.has_chassis:
            print("[wbc_vr_robot] centering wheels to straight (steering -> 0) ...")
            self.robot.chassis.set_steering_angle(0.0, wait_time=self.args.home_settle)

    def engage_reset(self, ik, left0, right0, head0) -> None:
        """Zero the odometry origin in lock-step with the IK reset at engage."""
        if self._odom is not None:
            self._odom.reset_origin()
        self._prev_base_cmd = np.zeros(3)
        for grp in self._prev_cmd:
            self._prev_cmd[grp] = None
        # Begin the episode on the FIRST engage; a re-engage (live, after an e-stop)
        # continues the SAME take -- held frames are simply skipped -- so we only
        # re-anchor the record cadence rather than restarting (which would wipe frames).
        if self._episode is not None:
            if not self._episode.recording:
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

    def measured_q(self, ik) -> Optional[np.ndarray]:
        """Assemble the measured config for ``solve(current_q=)`` (``--closed-loop-q``).

        Returns None (open-loop) unless ``--closed-loop-q`` is set; also falls back to
        open-loop for any tick whose joint readback has an unexpected shape, so a bad
        sample can never inject a malformed seed. Open-loop does NOT read the joints here
        (that readback would sit in the pre-solve critical path and perturb the very
        timing the open-loop run is the baseline for); debug_row() reads them after
        actuation instead.
        """
        if not self.args.closed_loop_q:
            self._dbg["meas_joints"] = None        # debug_row reads post-actuation
            self._dbg["seed_q"] = None
            self._dbg["closed_loop_active"] = False
            return None
        from omniteleop.wbc_robot_util import set_base_in_q  # noqa: PLC0415

        meas = self._read_measured_joints()
        self._dbg["meas_joints"] = meas
        if any(meas[grp] is None for grp in self._joint_names):
            self._dbg["seed_q"] = None             # bad readback -> open-loop this tick
            self._dbg["closed_loop_active"] = False
            return None
        q = ik.q
        for grp, names in self._joint_names.items():
            for name, val in zip(names, meas[grp], strict=True):
                q[ik._idx_q[name]] = float(val)  # noqa: SLF001
        if self._odom is not None:
            q = set_base_in_q(q, self._odom.pose)
        self._dbg["seed_q"] = q.copy()
        self._dbg["closed_loop_active"] = True
        return q

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
            if self.args.grippers:
                self.robot.left_arm.send_ee_pass_through_message(
                    self._build_hande(float(np.clip(left_gripper, 0.0, 1.0)))
                )
                self.robot.right_arm.send_ee_pass_through_message(
                    self._build_hande(float(np.clip(right_gripper, 0.0, 1.0)))
                )
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
            return
        # Snapshot odom ONCE: the pose fed to the PD and the raw wheels logged for this
        # tick come from the same integrated sample.
        snap = self._odom.snapshot()
        self._dbg["odom"] = snap
        chassis = self.robot.chassis
        if hold:
            chassis.set_velocity(vx=0.0, vy=0.0, wz=0.0, wait_time=0.0,
                                 sequential_steering=False)
            self._prev_base_cmd = np.zeros(3)
            self._dbg["base_pd_raw"] = None
            self._dbg["base_pd_err"] = None
            self._dbg["base_cmd"] = np.zeros(3)
            return
        base_cl = self._base_cl
        max_lin = self.args.base_max_speed
        max_ang = 2.0 * self.args.base_max_speed
        raw, err = base_cl.pd_twist(
            result.base_pose, result.base_twist, snap["pose"],
            kp_xy=self.args.base_kp_xy, kp_yaw=self.args.base_kp_yaw,
            max_lin_speed=max_lin, max_ang_speed=max_ang,
        )
        cmd = base_cl.shape_twist(
            raw, self._prev_base_cmd, dt,
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
        self._prev_base_cmd = cmd
        chassis.set_velocity(vx=float(cmd[0]), vy=float(cmd[1]), wz=float(cmd[2]),
                             wait_time=0.0, sequential_steering=False)
        self._dbg["base_pd_raw"] = np.asarray(raw, dtype=float)
        self._dbg["base_pd_err"] = np.asarray(err, dtype=float)
        self._dbg["base_cmd"] = np.asarray(cmd, dtype=float)

    def debug_row(self, result) -> dict:
        """Assemble this tick's /debug record from the stashed intermediates.

        Called by run_loop AFTER actuate, so the open-loop measured-joint readback below
        runs post-command and never delays the chassis/joint write. Not-computed values
        (held ticks, base disabled) become NaN so every column keeps a consistent shape.
        """
        d = self._dbg
        nq = self.ik.model.nq
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
            "closed_loop_active": bool(d.get("closed_loop_active", False)),
            "seed_q": vec(d.get("seed_q"), nq),
            "base_twist": np.asarray(result.base_twist, dtype=np.float32),
            "base_pd_raw": vec(d.get("base_pd_raw"), 3),
            "base_pd_err": vec(d.get("base_pd_err"), 3),
            "base_cmd": vec(d.get("base_cmd"), 3),
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

    def _init_recording(self) -> None:
        """Set up the optional vr_reader-format episode recorder (``--record``).

        OFF unless ``--record``. When on, records (while engaged) the same main-HDF5
        schema as ``leader/vr_reader.py`` so takes are interchangeable, EXCEPT:
          * ``action/joint`` adds ``torso`` (the WBC commands the torso here too);
          * the base action ``chassis_*`` is the shaped twist actually sent to the
            chassis (the WBC drives the base; the leader thumbstick is unused here);
          * grippers + the wrist camera are omitted (not installed) -- their code is
            kept in comments so it can be restored once the hardware is mounted.
        """
        self._record_period = 0.0
        self._next_record_t = 0.0
        if not self.args.record:
            return
        self._record_period = 1.0 / self.args.record_rate
        self._episode = EpisodeRecorder(self.args.save_dir)
        print(f"[wbc_vr_robot] recording -> {self.args.save_dir} "
              f"(episode_{self._episode.episode_id}, {self.args.record_rate:g}Hz, "
              "head_left_rgb+head_depth +torso action; grippers/wrist OFF)")

    def _grab_head_images(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Poll the head camera -> ``(left_rgb uint8 HxWx3, depth uint16 HxW)`` or None.

        None until both streams have delivered a frame (parity with vr_reader's
        ``_all_selected_streams_ready`` -- so the saved HDF5 keeps consistent keys and
        shapes across frames). Raises ValueError on a malformed shape, so a miswired
        camera fails loudly instead of recording garbage.
        """
        obs = self.robot.sensors.head_camera.get_obs(obs_keys=["left_rgb", "depth"])
        left_rgb = obs.get("left_rgb")
        depth = obs.get("depth")
        if left_rgb is None or depth is None:
            return None
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
        return np.ascontiguousarray(left_rgb, dtype=np.uint8), depth_u16

    def _head_extrinsic(self, obs: dict) -> np.ndarray:
        """``base_t_zed_depth_frame`` (4x4) from the MEASURED joints (vr_reader parity).

        FK the observed torso/arm/head into the WBC model with the planar base left at
        the origin (``nominal_q``'s neutral base), so the head pose is base-RELATIVE
        and matches vr_reader's BASE-locked KinHelper extrinsic; recorded next to ZED_K.
        """
        q = self.ik.nominal_q()  # neutral planar root => world == base origin
        for grp, names in self._joint_names.items():
            for name, val in zip(names, obs[grp], strict=True):
                q[self.ik._idx_q[name]] = float(val)  # noqa: SLF001
        return np.asarray(self.ik.frame_pose(HEAD_FRAME, q).homogeneous, dtype=np.float32)

    def record_tick(self, result, hold: bool, now: float) -> None:
        """Append one vr_reader-format frame while engaged (no-op unless ``--record``).

        Throttled to ``--record-rate``; skips while ``hold`` (e-stop / failed solve /
        safety hold / stale source) so only live teleop is recorded. Must run AFTER
        :meth:`actuate` so ``self._prev_base_cmd`` holds this tick's chassis command.
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

        imgs = self._grab_head_images()
        if imgs is None:
            return  # camera still warming up -- skip rather than write inconsistent keys
        head_left_rgb, head_depth_u16 = imgs

        # Measured joints: drive both obs/joint and the extrinsic FK. Validate shapes
        # aggressively -- a bad readback must abort, never be recorded as-is.
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
        images = {
            "head_left_rgb": head_left_rgb,
            "head_depth": head_depth_u16,
            "intrinsic": ZED_K.astype(np.float32),
            "extrinsic": self._head_extrinsic(obs),
            # Wrist camera not installed -- restore when mounted (see vr_reader):
            # "left_wrist_rgb": self._last_wrist_left_rgb,
        }
        frame = {
            "timestamp_ns": np.int64(time.time_ns()),
            "action": {
                "joint": {
                    # torso added vs vr_reader: the WBC commands the torso here.
                    "torso": np.asarray(result.torso, dtype=np.float32),
                    "left_arm": np.asarray(result.left_arm, dtype=np.float32),
                    "right_arm": np.asarray(result.right_arm, dtype=np.float32),
                    "head": np.asarray(result.head, dtype=np.float32),
                    # Base action = the shaped twist sent to the chassis (the WBC drives
                    # the base; the leader thumbstick chassis_* is unused here).
                    "chassis_vx": np.float32(base_cmd[0]),
                    "chassis_vy": np.float32(base_cmd[1]),
                    "chassis_wz": np.float32(base_cmd[2]),
                },
                # Grippers not installed -- restore when mounted (vr uses the triggers):
                # "gripper": {
                #     "left": np.float32(np.clip(left_gripper, 0.0, 1.0)),
                #     "right": np.float32(np.clip(right_gripper, 0.0, 1.0)),
                # },
            },
            "obs": {
                "joint": {
                    "left_arm": obs["left_arm"],
                    "right_arm": obs["right_arm"],
                    "head": obs["head"],
                    "torso": obs["torso"],
                },
                # Grippers not installed -- restore when mounted (FC03 status poll):
                # "gripper": {
                #     "left": obs_grip_left,
                #     "right": obs_grip_right,
                # },
                "images": images,
            },
        }
        self._episode.record(frame)

    def close(self) -> None:
        """Stop the base, the odom thread, and shut the robot down (best-effort).

        Safe to call on a partially-initialized driver (robot/odom may be None) so the
        engage-time teardown works even if __init__ failed mid-setup.
        """
        try:
            if self.robot is not None and self.has_chassis and hasattr(
                self.robot.chassis, "stop"
            ):
                self.robot.chassis.stop()
        except Exception:
            pass
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
        try:
            if self.robot is not None:
                self.robot.shutdown()
        except Exception:
            pass


def _build_ik() -> tuple[VegaWholeBodyIK, WBCConfig]:
    cfg = WBCConfig()
    ik = VegaWholeBodyIK(cfg)
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
    traj: _TrajLog,
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
    t0 = clock()
    last_print = 0.0
    print(f"[wbc_vr_robot] {driver.name} loop @ {args.ik_rate:g}Hz IK "
          f"({'replay ' + format(args.speed, 'g') + 'x' if replay else 'live'}); "
          f"enable={[k for k, v in enable.items() if v]} grippers={args.grippers}")

    while True:
        now = clock()
        t = now - t0
        if replay and source.done:
            break
        vr = source.latest
        if vr is not None and vr.exit_requested:
            print("\n[wbc_vr_robot] leader requested exit.")
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
        measured_q = driver.measured_q(ik)
        if cfg.head_mode == "ik":
            result = ik.solve(left_target, right_target, dt,
                              head_target=head_target, current_q=measured_q)
        else:
            head_joints = ik.solve_head(head_target, dt)
            result = ik.solve(left_target, right_target, dt,
                              head_joints=head_joints, current_q=measured_q)

        hold_reason = driver.extra_hold(now, last_cmd_wall, estop)
        hold = estop or (not result.success) or result.held or (hold_reason is not None)
        left_gripper = vr.left_gripper if vr is not None else 0.0
        right_gripper = vr.right_gripper if vr is not None else 0.0
        driver.actuate(result, left_gripper, right_gripper, enable, hold, dt)
        if hasattr(driver, "record_tick"):
            driver.record_tick(result, hold, now)

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
    # Always capture a /debug HDF5 (auto-named per mode so the with/without
    # --closed-loop-q runs land in two distinct files next to the replay source).
    if args.traj_out is None:
        import datetime  # noqa: PLC0415
        import os  # noqa: PLC0415
        run_mode = "closedloopq" if args.closed_loop_q else "openloop"
        base_dir = os.path.dirname(os.path.abspath(args.replay)) if replay else os.getcwd()
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.traj_out = os.path.join(base_dir, f"wbc_debug_{run_mode}_{stamp}.hdf5")
    meta = {
        "mode": "replay" if replay else "live",
        "closed_loop_q": bool(args.closed_loop_q),
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
        "source_timeout": float(args.source_timeout),
        "max_joint_step": float(args.max_joint_step),
        "head_lpf_tau": float(args.head_lpf_tau),
        "head_planar_pos_deadband": float(args.head_planar_pos_deadband),
        "head_planar_yaw_deadband": float(args.head_planar_yaw_deadband),
        "base_post_linear_deadband": float(args.base_post_linear_deadband),
        "base_post_angular_deadband": float(args.base_post_angular_deadband),
        "replay_file": str(args.replay) if replay else "",
    }
    print("=" * 72)
    print(f"[wbc_vr_robot] REAL ROBOT. enabled={enabled} grippers={args.grippers} | "
          f"{'REPLAY ' + format(args.speed, 'g') + 'x' if replay else 'LIVE'} | "
          f"closed_loop_q={args.closed_loop_q} base_max={args.base_max_speed:g}m/s")
    print(f"  filters -> head_lpf={args.head_lpf_tau:g}s, "
          f"head_planar_deadband={args.head_planar_pos_deadband:g}m/"
          f"{args.head_planar_yaw_deadband:g}rad, "
          f"base_post_deadband={args.base_post_linear_deadband:g}m/s/"
          f"{args.base_post_angular_deadband:g}rad/s")
    print(f"  joints -> dexcontrol 1:1 (urdf {cfg.urdf_path.rsplit('/', 1)[-1]}). "
          "Robot homes to nominal, then moves. KEEP CLEAR. Ctrl-C aborts (zeros base).")
    print(f"  /debug -> {args.traj_out}")
    if args.record:
        print(f"  recording -> {args.save_dir} @ {args.record_rate:g}Hz while engaged "
              "(head_left_rgb+head_depth, +torso action; grippers/wrist OFF).")
    print("=" * 72)

    traj = _TrajLog(args.traj_out, meta=meta)
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
        print(f"\n[wbc_vr_robot] summary: {traj.summary()}")
        if args.traj_out:
            traj.write()
            print(f"[wbc_vr_robot] /debug log -> {args.traj_out} ({len(traj)} ticks)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--replay", metavar="FILE",
                      help="drive the robot from a stream HDF5 written by wbc_vr_record.py.")
    mode.add_argument("--source", choices=("live",), default=None,
                      help="drive from the live leader (default when --replay is omitted).")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                        help=f"replay speed multiplier (default {DEFAULT_SPEED:g}; <1 = slower).")
    parser.add_argument("--enable", default="arms,torso,head,base",
                        help="comma list of DOF groups to actuate: any of "
                             "arms,torso,head,base (or 'all'/'none'). Default "
                             "'arms,torso,head,base'. Unselected groups are still solved "
                             "but not sent to that actuator.")
    parser.add_argument("--grippers", action="store_true",
                        help="actuate the grippers from the trigger values (OFF by default).")
    parser.add_argument("--ik-rate", type=float, default=100.0,
                        help="whole-body IK solve rate in Hz (default 100).")
    parser.add_argument("--cmd-rate", type=float, default=10.0,
                        help="leader command rate in Hz used to size the interpolation glide "
                             "(default 10, matching wbc_vr_leader). 0 disables interpolation. "
                             f"Replay rejects recorded command gaps above "
                             f"{DEFAULT_REPLAY_GAP_LIMIT_MULTIPLE:g}x this nominal glide.")
    parser.add_argument("--head-lpf-tau", type=float, default=DEFAULT_HEAD_LPF_TAU,
                        help=f"head-target low-pass time constant s (default "
                             f"{DEFAULT_HEAD_LPF_TAU:g}; 0 disables).")
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
    parser.add_argument("--namespace", default="",
                        help="Zenoh namespace (must match the leader; default empty).")
    parser.add_argument("--traj-out", default=None,
                        help="write the per-tick /debug log (base PD chain, odom + raw "
                             "wheels, closed-loop seed, joint cmd/sent/measured) to this "
                             ".hdf5. Default: auto-named wbc_debug_<mode>_<timestamp>.hdf5 "
                             "next to the replay file.")
    parser.add_argument("--record", action="store_true",
                        help="record an episode (vr_reader EpisodeRecorder format) while "
                             "engaged: action+obs joints (incl. torso) and head_left_rgb/"
                             "head_depth (+intrinsic/extrinsic). Grippers and the wrist "
                             "camera are NOT recorded (not installed). OFF by default.")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR,
                        help=f"directory for recorded episodes (default {DEFAULT_SAVE_DIR}).")
    parser.add_argument("--record-rate", type=float, default=DEFAULT_RECORD_RATE,
                        help=f"episode record cadence in Hz (default {DEFAULT_RECORD_RATE:g}; "
                             "must be <= --ik-rate).")

    hw = parser.add_argument_group("hardware")
    hw.add_argument("--closed-loop-q", action="store_true",
                    help="seed the IK from the MEASURED robot config each tick "
                         "(solve(current_q=)). Default OFF (open-loop), matching the SAPIEN "
                         "replay path and avoiding joint-readback convention bugs; enable "
                         "once readback conventions are confirmed.")
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
    hw.add_argument("--base-deadband", type=float, default=DEFAULT_BASE_DEADBAND,
                    help=f"base-twist deadband m/s, ang=2x (default {DEFAULT_BASE_DEADBAND:g}).")
    hw.add_argument("--base-accel", type=float, default=DEFAULT_BASE_ACCEL,
                    help=f"base-twist slew m/s^2, ang=2x (default {DEFAULT_BASE_ACCEL:g}).")
    hw.add_argument("--base-kp-xy", type=float, default=DEFAULT_BASE_KP_XY,
                    help=f"closed-loop base PD gain, xy error (default {DEFAULT_BASE_KP_XY:g}).")
    hw.add_argument("--base-kp-yaw", type=float, default=DEFAULT_BASE_KP_YAW,
                    help=f"closed-loop base PD gain, yaw error (default {DEFAULT_BASE_KP_YAW:g}).")
    hw.add_argument("--base-max-speed", type=float, default=DEFAULT_BASE_MAX_SPEED,
                    help=f"base linear velocity clamp m/s, ang=2x (default "
                         f"{DEFAULT_BASE_MAX_SPEED:g}; conservative for first bring-up).")
    hw.add_argument("--base-post-linear-deadband", type=float,
                    default=DEFAULT_BASE_POST_LINEAR_DEADBAND,
                    help="per-axis vx/vy deadband in m/s applied after base deadband/clamp/"
                         "slew using the same threshold for both linear axes and the "
                         "pre-slew raw command as the activity reference. "
                         f"Default {DEFAULT_BASE_POST_LINEAR_DEADBAND:g}.")
    hw.add_argument("--base-post-angular-deadband", type=float,
                    default=DEFAULT_BASE_POST_ANGULAR_DEADBAND,
                    help="per-axis wz deadband in rad/s applied after base deadband/clamp/"
                         "slew using the pre-slew raw command as the activity reference. "
                         f"Default {DEFAULT_BASE_POST_ANGULAR_DEADBAND:g}.")
    args = parser.parse_args()

    if args.ik_rate <= 0:
        parser.error("--ik-rate must be > 0")
    if args.cmd_rate < 0 or not np.isfinite(args.cmd_rate):
        parser.error("--cmd-rate must be finite and >= 0")
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and > 0")
    for flag, val in (("--home-tol", args.home_tol), ("--max-joint-step", args.max_joint_step),
                      ("--home-settle", args.home_settle),
                      ("--base-deadband", args.base_deadband), ("--base-kp-xy", args.base_kp_xy),
                      ("--base-kp-yaw", args.base_kp_yaw),
                      ("--head-lpf-tau", args.head_lpf_tau),
                      ("--head-planar-pos-deadband", args.head_planar_pos_deadband),
                      ("--head-planar-yaw-deadband", args.head_planar_yaw_deadband),
                      ("--base-post-linear-deadband", args.base_post_linear_deadband),
                      ("--base-post-angular-deadband", args.base_post_angular_deadband)):
        if not np.isfinite(val) or val < 0.0:
            parser.error(f"{flag} must be finite and >= 0")
    for flag, val in (("--base-accel", args.base_accel),
                      ("--base-max-speed", args.base_max_speed),
                      ("--source-timeout", args.source_timeout),
                      ("--record-rate", args.record_rate)):
        if not np.isfinite(val) or val <= 0.0:
            parser.error(f"{flag} must be finite and > 0")
    if args.record and args.record_rate > args.ik_rate:
        parser.error("--record-rate must be <= --ik-rate (cannot record faster than the loop)")
    try:
        enable = parse_enable_mask(args.enable)
    except ValueError as exc:
        parser.error(str(exc))

    _run_ik_mode(args, enable)


if __name__ == "__main__":
    main()
