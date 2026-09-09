#!/usr/bin/env python3
"""Real-robot whole-body VR follower for ``wbc_vr_leader.py`` -- staged-safety bring-up.

HW-WORKAROUND(right-arm-frozen) -- TEMPORARY FORK of ``wbc_vr_robot.py``. Run THIS script
(not the canonical one) while the physical right arm must stay at nominal; delete it,
``whole_body_ik_tmp.py``, and ``wbik_tmp.yaml`` when the right arm can be used again.
``wbc_vr_leader.py`` needs NO change -- it continues publishing both Cartesian EE poses.
Differences from canonical, all tagged ``HW-WORKAROUND(right-arm-frozen)``:
  1. imports ``whole_body_ik_tmp`` (which reads ``wbik_tmp.yaml`` -> ``dead_joints:``);
  2. ``VRTeleopConfig.from_yaml(WBIK_TMP_CONFIG_PATH)`` so the tmp trio is self-contained;
  3. the QP pins all right-arm joints at config nominal, and drops the R_ee task entirely
     once the whole chain to R_ee is pinned (right arm + torso) -- it could then no longer
     pose the gripper, only anchor the base against head tracking;
  4. ``HardwareDriver.actuate`` independently replaces the real right-arm command with the
     same nominal vector, while the left arm remains fully actuated;
  5. startup and episode/debug metadata stamp the frozen-right-arm provenance.

CAVEAT while active: the right EE target is NOT tracked at all (see 3) -- ``action/eef/right``
records raw operator hand intent that the robot never served, and ``err R`` on the HUD is an
unserved residual, not a tracking error.
Record to a SEPARATE --save-dir: these episodes contain a nominal right-arm action and will
not transfer unchanged to a robot where the right arm is active.

Same control core as the SAPIEN sim follower (``scripts/wbc_vr_record.py``) -- subscribe
to the leader's Cartesian ``L_ee``/``R_ee``/head TARGET poses, interpolate the low-rate
command stream up to the IK rate, run ``VegaWholeBodyIK`` -- but driving the REAL Vega
hardware (``dexcontrol.robot.Robot``) instead of a simulator.

AVOID using low-level APIs like `set_motion_state`, `set_steering_angle`, or
`set_wheel_velocity` directly, which can command opposed steering/velocity and damage
the wheel motors. Route base motion through `set_velocity`.

Two real-robot modes:

  1. ``--replay FILE``   re-drive an episode THIS script recorded (``--record``).
  2. ``--source live``   subscribe to the live leader and drive the robot in real time.

``--replay`` is a PARAMETER-COMPARISON tool: it consumes an ``EpisodeRecorder`` take
(``episode_<N>.hdf5``) and re-issues only that take's ORIGINAL solver inputs -- the
world-frame ``action/eef/{left,right}`` + ``action/head`` 4x4 targets and the
``action/gripper/*`` trigger commands -- at their recorded cadence, time-scaled by
``replay_speed``. Everything the follower derived from them (joint commands, base twist)
is dropped and re-solved, so running the same file twice across an edit to
``follower/wbik_tmp.yaml`` isolates the effect of that edit on one fixed operator intent.
Because the recorded ``action/head`` was ALREADY written through the head low-pass +
planar deadband at record time, replay feeds it to the IK unfiltered (re-filtering would
double-apply the LPF lag): ``head_lpf_tau`` and ``head_planar_*_deadband`` are baked into
the file and are the one group of knobs this path cannot compare. NOTE: an episode
recorded by the CANONICAL follower replays fine here, but it was solved without
the nominal right-arm pins, so the comparison also carries that difference.

All control-loop tunables -- replay speed, IK / command rates, head LPF + planar
deadbands, base PD gains, base slew / deadband / post-deadband -- come SOLELY from the
``vr_teleop:`` block of ``follower/wbik_tmp.yaml`` (HW-WORKAROUND(right-arm-frozen): the tmp
fork reads the tmp YAML, not the canonical one), so editing that file is never silently
ignored. They are not CLI flags.

Pass ``--debug-dir DIR`` to write a per-tick ``/debug`` HDF5 (auto-named
``episode_<N>_debug.hdf5`` under ``DIR``, matching ``vr_reader`` / ``vis_teleop_curves``):
the base PD chain (odom pose +
raw wheels, IK base pose/twist, PD raw/err, shaped command), plus per-group joint
command/sent/measured -- enough to inspect commanded vs measured motion. OFF by default.

``--record`` starts automatically on the first engage after VR calibration, matching
``scripts/wbc_vr_record.py``'s post-calibration recording gate. It saves an episode in
the ``EpisodeRecorder`` schema of ``leader/vr_reader.py`` (mostly interchangeable):
``action``/``obs`` joints plus ``head_left_rgb``/``head_depth``/``{left,right}_wrist_rgb``
(``left``/``right`` = the ARM the wrist camera is on, not the stereo eye) and the
optional ``head_right_rgb`` (``--head-right-rgb``; here ``right`` DOES mean the stereo
eye, so FoundationStereo can produce head depth offline -- see
``calib/packing_data/scripts/fs_head_depth.py``) and the
``left``/``right`` gripper under ``action/gripper`` (leader trigger command) and
``obs/gripper`` (Robotiq FC03 achieved position). It adds ``torso`` to ``action/joint``
(the WBC commands the torso) and uses the shaped base twist for the ``chassis_*`` action;
``action/joint/*`` records the POST-CLAMP joint commands actually sent to hardware
(``_dbg['sent_joints']``), not the raw WBC result. Recording requires the FULL
``--enable torso,arms,head,base`` mask: it records the MEASURED wheel-odometry body twist
as the ``obs/joint`` ``chassis_*`` (achieved-velocity counterpart to the action) and the
measured base pose ``obs/base/pose`` ``(x, y, yaw)`` in the engage-origin world frame.

``--arkit-base {off,record,control}`` adds the drift-free base pose tracked by the
iPhone rigidly mounted to the E-stop tower (ARKit via Record3D over USB; the robot-side
publisher is ``record3d/tracking/base_pose_pub.py`` and the phone->base extrinsic comes
from ``record3d/calibration/calib_solve.py``). Swerve odometry integrates steering and
wheel velocity, so its yaw walks away over a take and drags every world-frame quantity
derived from it -- ``obs/base/pose`` and the offline ``world_t_cam`` -- with it; the
phone measures the base pose directly instead. ``record`` logs it alongside the wheel
odometry (``obs/base/pose_arkit`` engage-origin + ``obs/base/pose_arkit_world``, which is
unbroken across re-engages, plus the ``arkit_*`` /debug columns) while the wheel odometry
still closes the base PD loop -- so a take measures the drift before anything depends on
it. ``control`` additionally feeds the phone pose to ``base_closed_loop.pd_twist`` as the
MEASURED base, and holds the base on a stale (``stale-arkit``) or relocalized
(``arkit-jump``) tracker the same way it holds on stale odometry. Requires
``--enable base``; the phone pose is never a fallback for a missing odometry thread.

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
lands in the world frame (see ``scripts/vis_episode.py``). The wrist cameras stream from
``sensors/{left,right}_wrist_zedm/*`` (each needs its own external ZED-SDK publisher --
one process per camera, selected by ``--serial-number``; see vr_reader). A wrist whose
publisher never comes up (checked once at recording start, see ``_await_record_cameras``)
is recorded as UNAVAILABLE for the whole run rather than blocking the take: its
``{arm}_wrist_rgb``/``{arm}_wrist_frame_ns`` keys are simply absent from every frame.

Timing metadata for offline camera-latency audits (``scripts/audit_episode_latency.py``):
each frame stores the publisher capture stamps ``obs/images/{head_frame_ns,
head_depth_frame_ns,head_right_frame_ns,left_wrist_frame_ns,right_wrist_frame_ns}``
(camera-publisher-host clock) and ``obs/images/grab_wall_ns``
(workstation clock right after all grabs). ``meta/ntp`` preserves the robot
SoC-minus-workstation offset from ``Robot.query_ntp()``; ``meta/camera_ntp`` stores
camera-publisher-host-minus-workstation offsets from the ZED publishers' lightweight
clock services and is required for new recordings. Absolute camera staleness should use ``camera_ntp``;
the recorder itself keeps pairing the latest-ARRIVED image with fresh proprio (the
training-consistent convention; see plan.md).

Run in the dexmate conda env (pinocchio + pink + dexcomm + dexcontrol)
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from dataclasses import asdict
from typing import Callable, Optional

import numpy as np
from dexcomm.codecs import DictDataCodec, JsonDataCodec

from omniteleop.common import get_config
from omniteleop.common.head_camera import ZED_K
from omniteleop.common.recorder import EpisodeRecorder, peek_next_episode_id
from omniteleop.common.schemas import WBC_FOLLOWER_STAGE_ABORTED, WBCFollowerStatus
from omniteleop.common.streaming_recorder import StreamingEpisodeRecorder
from omniteleop.follower.arkit_base import ARKitBaseTracker
from omniteleop.follower.whole_body_ik_tmp import (  # HW-WORKAROUND(right-arm-frozen)
    DEFAULT_CONFIG_PATH as WBIK_TMP_CONFIG_PATH,
)
from omniteleop.follower.whole_body_ik_tmp import (
    HEAD_FRAME,
    LEFT_EE_FRAME,
    RIGHT_ARM_JOINTS,
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
    vr_to_ee_targets,
    vr_to_head_target,
)
from omniteleop.wbc_teleop import VRJointSubscriber, VRTeleopConfig, bind_vr_teleop_args

# Shared VR-follower control-loop tunables, loaded from the vr_teleop: block of
# follower/wbik.yaml so this real-robot follower and the SAPIEN sim follower
# (scripts/wbc_vr_record.py) shape the head/base command stream identically -- a take
# recorded in sim must replay the same way here. See omniteleop.wbc_teleop. The
# bring-up-specific knobs (homing, joint-step clamp, base max-speed, ...) stay local below.
# HW-WORKAROUND(right-arm-frozen): read the vr_teleop: block from wbik_tmp.yaml too, so the tmp
# trio (this script + whole_body_ik_tmp + wbik_tmp.yaml) is self-contained and editing the
# tmp YAML is never silently ignored. Values are currently identical to canonical wbik.yaml.
_VR_TELEOP = VRTeleopConfig.from_yaml(WBIK_TMP_CONFIG_PATH)
DEFAULT_HEAD_LPF_TAU = _VR_TELEOP.head_lpf_tau
DEFAULT_SPEED = _VR_TELEOP.replay_speed
DEFAULT_IK_RATE = _VR_TELEOP.ik_rate                # whole-body IK / control loop rate (Hz)
DEFAULT_CMD_RATE = _VR_TELEOP.cmd_rate              # leader command rate (Hz); 0 disables interp
DEFAULT_BASE_KP_XY = _VR_TELEOP.base_kp_xy          # closed-loop base PD gains (real-robot values)
DEFAULT_BASE_KP_YAW = _VR_TELEOP.base_kp_yaw
DEFAULT_BASE_YAW_HOLD_IN_XY = _VR_TELEOP.base_yaw_hold_in_xy
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
# --record stale-frame grace: when a record tick finds no strictly-newer frame from every
# camera, retry at the IK rate for up to this long before aborting the take. The publishers
# hold ~15 fps through these stalls (verified against their per-second pub-fps logs), so a
# non-newer frame means the DELIVERY path hiccupped: a WiFi burst, Zenoh backlog, or this
# process starving its Zenoh decode threads under solver load (worst while the torso moves).
# A frame is recorded only once BOTH publisher timestamps advance, so the no-duplicate
# guarantee is unchanged -- the grace converts a short delivery stall into one late frame
# instead of a dead take. 0 restores the old abort-on-first-stale-tick behavior.
DEFAULT_RECORD_STALE_GRACE = 0.20
# --head-right-rgb eye-pairing give-up: consecutive recorded frames whose right eye never
# matched the left before the retry budget expired, after which the retry is switched off
# for the rest of the episode. Guards the one regression the pairing gate could cause -- a
# publisher that never stamps the two eyes alike would otherwise pay the budget on EVERY
# frame, silently halving the record rate. 20 frames is 2 s at the default 10 Hz.
_RIGHT_PAIR_GIVE_UP = 20


# Wrist cameras (recorded under obs/images/{left,right}_wrist_rgb when --record). Like the
# head camera they are consumed through the dexcontrol Robot API, but as multi-stream
# ZED-Ms published by standalone ZED-SDK processes on sensors/<sensor_id>/{left_rgb,
# right_rgb} (see leader/vr_reader.py + tests/test_wrist_zedm_depth.py). RGB-only.
# "left"/"right" name the ARM the camera is mounted on (matching
# src/omniteleop/record/mdp_recorder.py), NOT the stereo eye -- only the left eye of each
# camera is ever published. Serials: left arm = 12417276, right arm = 12930616; the
# publishers select by --serial-number, so USB port order cannot swap the arms.
_WRIST_SENSOR_IDS: dict[str, str] = {
    "left": "left_wrist_zedm",
    "right": "right_wrist_zedm",
}
_WRIST_OBS_KEY = "left_rgb"  # stereo left eye -- the only stream the publishers send
_HEAD_SENSOR_ID = "head_camera"
_CAMERA_CLOCK_SAMPLE_COUNT = 15
_CAMERA_CLOCK_MIN_SAMPLES = 8
_CAMERA_CLOCK_TIMEOUT_S = 0.5
_CAMERA_CLOCK_MAX_CONSECUTIVE_FAILURES = 3


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
    episode_id: int = -1,
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
        episode_id=int(episode_id),
    )


def _recorder_episode_id(driver) -> int:
    """Episode index the driver's recorder will save next, or -1 without ``--record``."""
    recorder = getattr(driver, "_episode", None)
    return -1 if recorder is None else int(getattr(recorder, "episode_id", -1))


def _build_abort_status(reason: str, timestamp_ns: Optional[int] = None) -> WBCFollowerStatus:
    """Terminal status frame: the follower loop DIED (vs a recoverable hold)."""
    return WBCFollowerStatus(
        timestamp_ns=int(time.time_ns() if timestamp_ns is None else timestamp_ns),
        stage=WBC_FOLLOWER_STAGE_ABORTED,
        estop=True,
        success=False,
        held=True,
        hold=True,
        hold_reason=str(reason),
        safety_status="ABORTED",
    )


def _publish_abort_status(source, exc: BaseException) -> None:
    """Best-effort final ABORTED frame so the headset HUD banners instead of going stale.

    Runs on the exception path BEFORE ``source.close()`` tears the node down (the leader's
    subscriber is long-lived, so a fresh publisher on the same topic matches immediately).
    Must never raise: nothing here may mask the original traceback.
    """
    try:
        pub = _create_status_publisher(source)
        if pub is None:  # replay source: no node, no leader watching
            return
        text = str(exc).strip()
        reason = text.splitlines()[0][:160] if text else type(exc).__name__
        pub.publish(asdict(_build_abort_status(reason)))
        time.sleep(0.25)  # let zenoh flush before teardown closes the session
    except Exception:
        pass


def _camera_clock_topic(sensor_id: str) -> str:
    return f"sensors/{sensor_id}/clock"


def _ntp_offset_rtt_ns(t0: int, t1: int, t2: int, t3: int) -> tuple[int, int]:
    """NTP-style offset/rtt in ns for client/workstation t0,t3 and server t1,t2."""
    return round(((int(t1) - int(t0)) + (int(t2) - int(t3))) / 2.0), int(t3) - int(t0)


def _bytes_scalar(text: str) -> np.ndarray:
    return np.asarray(str(text).encode("utf-8"))


def _camera_clock_calibration_from_samples(
    samples: list[dict],
    *,
    sensor_id: str,
    source: str,
    queried_at_ns: int,
) -> dict:
    """Collapse NTP clock samples to the static ``meta/camera_ntp/*`` payload.

    Keep the lower-RTT half to avoid obvious delayed replies while staying close to
    dexcontrol's simple mean-offset style. Returns HDF5-safe scalar arrays.
    """
    valid = [
        s for s in samples
        if int(s.get("rtt_ns", 0)) > 0 and np.isfinite(float(s.get("offset_ns", 0)))
    ]
    if not valid:
        raise ValueError("no valid camera clock samples")
    valid.sort(key=lambda s: int(s["rtt_ns"]))
    keep = valid[:max(1, (len(valid) + 1) // 2)]
    return {
        "offset_ns": np.int64(round(float(np.mean([int(s["offset_ns"]) for s in keep])))),
        "rtt_ns": np.int64(round(float(np.mean([int(s["rtt_ns"]) for s in keep])))),
        "queried_at_ns": np.int64(int(queried_at_ns)),
        "sensor_id": _bytes_scalar(sensor_id),
        "source": _bytes_scalar(source),
    }


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
ARM_HOME_STEP = 0.01  # arm interpolation step when homing straight to nominal
# s: after the stepped ramp, hold each group at nominal and block until the MEASURED
# joints converge within --home-tol. The dexcontrol ramp returns after each substep's
# wait_time whether or not the joint physically caught up, so lag accumulates and an
# immediate readback catches the arm mid-flight (the variable residual). This drains it.
DEFAULT_HOME_SETTLE = 3.0
# Hold the current swerve steering on quiet teleop gaps. Negative disables automatic
# re-centering during the take; explicit safety/stop paths still stop the base.
DEFAULT_BASE_QUIET_HOLD_S = -1
_JOINT_STEP_ABORT_TICKS = 25    # consecutive ticks demanding > 2x clamp -> abort


class _TrajLog:
    """Collect per-tick IK /debug state and flush it to a per-episode HDF5.

    Under ``--record`` the follower calls :meth:`flush` each time an ``episode_<N>.hdf5``
    saves, writing a paired ``episode_<N>_debug.hdf5`` and releasing ``_rows`` -- so RAM
    stays bounded to one episode instead of growing across a multi-episode session.
    Without ``--record`` there are no episode boundaries: a single flush runs at teardown.
    ``debug_dir=None`` disables logging (``append`` is a no-op), so omitting ``--debug-dir``
    costs nothing per tick.
    """

    def __init__(self, debug_dir: Optional[str], meta: Optional[dict] = None) -> None:
        self.debug_dir = debug_dir
        self.meta = dict(meta or {})
        self._rows: list[dict] = []

    def append(self, **fields) -> None:
        """Record one IK tick (no-op when logging is disabled)."""
        if self.debug_dir is not None:
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
        "odom_age", "odom_drive_ts_ns", "joystick_axis",
        "arkit_age", "arkit_jumps",
    )
    _VECTOR_KEYS = (
        "q", "base_pose", "base_twist", "base_pd_raw", "base_pd_err",
        "base_cmd", "odom_pose", "odom_steer", "odom_wvel",
        "arkit_pose", "arkit_world", "arkit_rp",
        "rx_chassis", "projected_chassis", "joystick_cmd_pose",
        "left_target", "right_target", "head_target",
        "cmd_torso", "cmd_left_arm", "cmd_right_arm", "cmd_head",
        "sent_torso", "sent_left_arm", "sent_right_arm", "sent_head",
        "meas_torso", "meas_left_arm", "meas_right_arm", "meas_head",
    )
    _STR_KEYS = (("safety_status", 96), ("hold_reason", 32), ("base_action", 12))

    def flush(self, episode_id: int) -> Optional[str]:
        """Write buffered ticks to ``episode_<id>_debug.hdf5``, then release the buffer.

        Returns the written path, or ``None`` when logging is disabled or no ticks are
        buffered (a 0-frame stop, or the trailing idle after the last episode, writes
        nothing). Clearing ``_rows`` afterward is what bounds memory across a session.
        """
        if self.debug_dir is None or not self._rows:
            return None
        import os  # noqa: PLC0415

        path = os.path.join(self.debug_dir, f"episode_{int(episode_id)}_debug.hdf5")
        n = len(self._rows)
        summary = self.summary()
        self._write(path, dict(self.meta, episode_id=int(episode_id)))
        self._rows = []  # release the buffer -> /debug RAM stays bounded to one episode
        print(f"[wbc_vr_robot] /debug episode_{int(episode_id)} -> {path} "
              f"({n} ticks) | {summary}")
        return path

    def _write(self, path: str, meta: dict) -> None:
        """Serialize the buffered ticks under a ``/debug`` group.

        ``/debug`` holds every per-tick column (scalars, gzipped vectors, fixed-width
        strings) plus ``meta`` in its attrs. Columns absent this run (no driver) are
        skipped rather than erroring.
        """
        import h5py  # noqa: PLC0415 -- optional dep, only needed at flush

        rows = self._rows
        col = lambda k: np.asarray([r[k] for r in rows])
        with h5py.File(path, "w") as f:
            g = f.create_group("debug")
            g.attrs["schema"] = "wbc_vr_robot_debug/v1"
            g.attrs["n_frames"] = len(rows)
            for key, val in meta.items():
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
                 enable: dict, traj: Optional[_TrajLog] = None) -> None:
        from dexbot_utils import RobotInfo  # noqa: PLC0415 -- hardware-only deps, lazy
        from dexcontrol.robot import Robot  # noqa: PLC0415

        from omniteleop.follower import base_closed_loop as base_cl  # noqa: PLC0415
        from omniteleop.follower.robotiq import (  # noqa: PLC0415
            build_hande_command,
            send_activate,
            send_ee_pass_through_with_timestamps,
        )

        # HW-WORKAROUND(right-arm-frozen): take the model constants from the SAME module that
        # built self.ik, so _joint_names and the solver can never disagree (and this process
        # parses only wbik_tmp.yaml, not both configs).
        from omniteleop.follower.whole_body_ik_tmp import (  # noqa: PLC0415
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
        self._arkit: Optional[ARKitBaseTracker] = None
        self._world_frame_epoch = 0
        self._episode: Optional[EpisodeRecorder] = None
        # Optional per-tick /debug log (passed by _run_ik_mode when --debug-dir is set).
        # stop_recording_episode() flushes it once per saved episode so RAM stays bounded.
        self._traj: Optional[_TrajLog] = traj
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
        # than the previously recorded one (each head_left_rgb / {left,right}_wrist_rgb
        # differs). When the delivery path stalls below the record rate, get_obs returns the
        # SAME (or no) frame twice. These hold the publisher timestamp of the last RECORDED
        # frame per camera (-1 until the first is recorded); record_tick retries a non-newer
        # frame at the IK rate for up to --record-stale-grace, then aborts -- either way a
        # take never records duplicates.
        self._last_rec_head_ns = -1
        self._last_rec_wrist_ns: dict[str, int] = {arm: -1 for arm in _WRIST_SENSOR_IDS}
        # Per-run wrist availability (set by _await_record_cameras): False if that arm's
        # ZED-SDK publisher never came up, so recording proceeds head-only for that wrist
        # instead of blocking the take. See _await_record_cameras / record_tick.
        self._wrist_enabled: dict[str, bool] = {arm: True for arm in _WRIST_SENSOR_IDS}
        # Stale-grace state: loop time the current stale streak began (None while fresh),
        # plus a short per-recorded-frame arrival-age history (local wall clock minus the
        # publisher capture stamp; spans the robot<->workstation clock offset, so read the
        # head-vs-wrist split and the trend, not absolutes). Dumped on a stale abort to show
        # WHICH stream stalled and whether its delivery latency was climbing beforehand.
        # Entry = (loop_t, head_age_ms, left_wrist_age_ms, right_wrist_age_ms).
        self._stale_since: Optional[float] = None
        self._frame_age_log: deque[tuple[float, float, float, float]] = deque(maxlen=64)
        # --head-right-rgb eye-pairing state (see the gate in record_tick): loop time the
        # current unpaired streak began, consecutive frames recorded unpaired, the episode
        # total for the close-out report, and whether the retry is still armed.
        self._right_unpaired_since: Optional[float] = None
        self._right_unpaired_streak = 0
        self._rec_right_unpaired = 0
        self._right_pair_retry = True

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
                # Recording needs the head camera AND both wrist ZED-Ms streaming, so build
                # the Robot with all three sensors enabled (mirrors leader/vr_reader.py).
                # Each wrist is a multi-stream ZED-M published by its own standalone
                # ZED-SDK process; inject an RGB-only config if the variant lacks it. One
                # ZED-SDK publisher per camera must run for frames to arrive.
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
                for _wrist_id in _WRIST_SENSOR_IDS.values():
                    if _wrist_id not in configs.sensors:
                        configs.sensors[_wrist_id] = ZedXCameraConfig(
                            name=_wrist_id, enable_rgb=True, enable_depth=False
                        )
                    configs.sensors[_wrist_id].enabled = True
                self.robot = Robot(configs=configs)
            else:
                self.robot = Robot()
            # Component health: print the dexcontrol status table the same way the
            # official examples do (display_robot_info.py) -- report, don't gate.
            # The previous custom pass/fail check aborted startup on latched faults
            # that the official tooling only displays.
            self.robot.get_component_status(show=True)
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
            if args.record and not args.no_grippers:
                self._init_gripper_monitors()
            elif args.record:
                print("[wbc_vr_robot] --no-grippers: skipping the FC03 gripper-status "
                      "warmup; obs/gripper records NaN for this run.")
            if enable["base"]:
                if not self.has_chassis:
                    raise SystemExit("[wbc_vr_robot] --enable base but the robot has no chassis.")
                self._wait_chassis()
                self._odom = OdometryThread(self.robot, drive_state_mode=args.drive_state_mode)
                self._odom.start()
                if args.arkit_base != "off":
                    self._start_arkit_tracker()
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

        This is the left-Y teardown and left-X episode-rollover stop path. It runs before
        recorder flushing or resource shutdown so a stop request cannot leave a previous
        100 Hz position target in flight while HDF5 saving begins. The base/head/torso expose
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

    def _wait_chassis(self, timeout: float = 10.0) -> None:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if np.asarray(self.robot.chassis.steering_angle).shape == (2,):
                return
            time.sleep(0.05)
        raise SystemExit(f"[wbc_vr_robot] no chassis state within {timeout:.0f}s.")

    def _start_arkit_tracker(self, timeout: float = 5.0) -> None:
        """Subscribe to the iPhone base-pose publisher (--arkit-base), failing loudly.

        The publisher runs on the ROBOT (record3d/tracking/base_pose_pub.py), so its topic
        lives under the robot's Zenoh namespace, not the leader/follower ``--namespace``.
        A missing publisher aborts here rather than mid-take: under ``control`` the base PD
        would have no measured pose at all, and under ``record`` a take of NaN phone poses
        is worth nothing and would only be discovered offline.
        """
        import os  # noqa: PLC0415

        namespace = os.environ.get("ROBOT_NAME", "")
        if not namespace:
            raise SystemExit("[wbc_vr_robot] --arkit-base needs $ROBOT_NAME (the robot's "
                             "Zenoh namespace, e.g. dm/vg3cfe65689d-1).")
        tracker = ARKitBaseTracker(namespace, name="wbc_vr_robot_arkit")
        if not tracker.wait_first(timeout):
            tracker.close()
            raise SystemExit(
                f"[wbc_vr_robot] --arkit-base {self.args.arkit_base}: no pose on "
                f"'{tracker.topic}' within {timeout:.0f}s -- start the publisher on the "
                "robot (record3d/tracking/base_pose_pub.py) and check the phone is "
                "streaming (lsusb | grep 12a8)."
            )
        snap = tracker.snapshot()
        self._arkit = tracker
        print(f"[wbc_vr_robot] iPhone base tracking ({self.args.arkit_base}) on "
              f"'{tracker.topic}': base roll/pitch "
              f"{np.degrees(snap['rp'][0]):+.2f}/{np.degrees(snap['rp'][1]):+.2f}deg "
              f"(constant on a flat floor -- drift here means the mount moved)")

    def _arkit_drives_base(self) -> bool:
        """True when the phone pose -- not the wheel odometry -- closes the base PD loop."""
        return self._arkit is not None and self.args.arkit_base == "control"

    def engage_reset(self, ik, left0, right0, head0) -> None:
        """Zero the odometry origin in lock-step with the IK reset at engage."""
        if self._odom is not None:
            self._odom.reset_origin()
        if self._arkit is not None:
            # Re-zeros on the latest phone pose AND clears the relocalization latch, so a
            # re-engage is the documented recovery from an ARKit world jump. False means no
            # sample to zero on: obs/base/pose_arkit stays NaN and, under control, the
            # "arkit-no-origin" hold below keeps the base still until a pose arrives.
            if not self._arkit.reset_origin():
                print("[wbc_vr_robot] WARNING: no iPhone pose to zero the ARKit origin on "
                      "-- base tracking stays NaN until the publisher recovers.")
        self._world_frame_epoch += 1
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
        self._last_rec_head_ns = -1
        self._last_rec_wrist_ns = {arm: -1 for arm in _WRIST_SENSOR_IDS}
        self._next_record_t = 0.0
        self._right_unpaired_since = None
        self._right_unpaired_streak = 0
        self._rec_right_unpaired = 0
        self._right_pair_retry = True

    def stop_recording_episode(self) -> None:
        """Stop and save the active episode without shutting down hardware.

        When a /debug log is attached (--debug-dir), also flush this episode's paired
        ``episode_<N>_debug.hdf5`` and release its buffer -- so /debug RAM stays bounded to
        a single episode across a multi-episode collection session.
        """
        if self._episode is None or not self._episode.recording:
            return
        # Capture the id BEFORE stop() (it increments episode_id) so the /debug file pairs
        # with the episode_<N>.hdf5 written this call.
        saved_id = getattr(self._episode, "episode_id", None)
        n = self._episode.num_frames()
        path = self._episode.stop()
        if path is None:
            print("\n[wbc_vr_robot] recording: 0 frames -- nothing saved.")
            return
        print(f"\n[wbc_vr_robot] saving {n} frames -> {path} ...")
        while self._episode.saving:
            time.sleep(0.05)
        save_error = getattr(self._episode, "last_save_error", None)
        if save_error is not None:
            raise RuntimeError(
                f"[wbc_vr_robot] episode save failed for {path}: {save_error}"
            ) from save_error
        print(f"[wbc_vr_robot] episode saved -> {path}")
        if self.args.head_right_rgb and n:
            paired = n - self._rec_right_unpaired
            print(f"[wbc_vr_robot] head stereo pairing: {paired}/{n} frames "
                  f"({100.0 * paired / n:.1f}%) carry head_right_frame_ns == "
                  f"head_frame_ns. Only those yield FoundationStereo head depth.")
        traj = getattr(self, "_traj", None)
        if traj is not None and saved_id is not None:
            traj.flush(saved_id)

    def home_to_nominal(self) -> None:
        """Run the startup nominal homing routine while the process stays live."""
        self.stop_all_motion()
        home_to_nominal(self, arm_home_step=ARM_HOME_STEP)

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
        reason = compute_hold_reason(
            estop, last_cmd_wall, now, self.args.source_timeout, odom_age
        )
        if reason is not None or estop or not self._arkit_drives_base():
            return reason
        # Only when the phone closes the base loop: a stale or relocalized tracker makes
        # the measured base pose meaningless, so hold rather than drive on it. A jump
        # clears only on the next engage (which re-zeros the origin).
        if self._arkit.age > self.args.source_timeout:
            return "stale-arkit"
        if self._arkit.jumped:
            return "arkit-jump"
        if not self._arkit.has_origin:
            return "arkit-no-origin"
        return None

    def actuate(self, result, left_gripper, right_gripper, enable, hold, dt) -> None:
        """Send the IK result to the enabled actuators (clamped); zero/freeze on hold."""
        from omniteleop.wbc_robot_util import clamp_joint_step  # noqa: PLC0415

        cmds = {"torso": result.torso, "left_arm": result.left_arm,
                "right_arm": self._nominal["right_arm"].copy(), "head": result.head}
        # HW-WORKAROUND(right-arm-frozen): the solver already pins these joints at nominal,
        # but repeat the invariant at the last boundary before real hardware. Thus even a
        # future solver regression cannot send a moving right-arm command. Safety holds
        # still send no joint commands, matching the follower's existing stop semantics.
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
            self._dbg["arkit"] = None
            self._dbg["base_pd_raw"] = None
            self._dbg["base_pd_err"] = None
            self._dbg["base_cmd"] = None
            self._dbg["base_action"] = "off"
            return
        # Snapshot odom ONCE: the pose fed to the PD and the raw wheels logged for this
        # tick come from the same integrated sample.
        snap = self._odom.snapshot()
        self._dbg["odom"] = snap
        # Snapshot the phone tracker on every tick it runs -- /debug then carries both
        # measured poses for the SAME tick, which is how the odometry drift is measured
        # before --arkit-base control is trusted with the loop.
        arkit = self._arkit.snapshot() if self._arkit is not None else None
        self._dbg["arkit"] = arkit
        chassis = self.robot.chassis
        if hold:
            chassis.set_velocity(vx=0.0, vy=0.0, wz=0.0, wait_time=0.0)
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
        allow_yaw_hold = self._allow_base_yaw_hold_in_xy()
        # extra_hold() has already forced a hold (and returned above) if the phone pose
        # is stale, jumped, or has no origin, so this branch can never see a NaN pose.
        measured = arkit["pose"] if self._arkit_drives_base() else snap["pose"]
        raw, err = base_cl.pd_twist(
            result.base_pose, result.base_twist, measured,
            kp_xy=self.args.base_kp_xy, kp_yaw=self.args.base_kp_yaw,
            max_lin_speed=max_lin, max_ang_speed=max_ang,
        )
        raw = self._mask_base_twist(raw, allow_yaw_hold=allow_yaw_hold)
        pd_raw = np.asarray(raw, dtype=float).copy()  # pre-shaping PD/FF output (debug)
        # Shared post-PD shaping keeps the full multi-axis slew anchor warm, then projects
        # the dispatched command. No preferred axis here: WBC keeps its existing dominant
        # post-PD selection with configured hysteresis.
        cmd, self._prev_base_shaped, self._base_axis = base_cl.shape_project_twist(
            raw,
            self._prev_base_shaped,
            dt,
            base_dofs=self.cfg.base_dofs,
            allow_yaw_hold=allow_yaw_hold,
            deadband_lin=self.args.base_deadband, deadband_ang=2.0 * self.args.base_deadband,
            max_lin_speed=max_lin, max_ang_speed=max_ang,
            max_lin_accel=self.args.base_accel, max_ang_accel=2.0 * self.args.base_accel,
            post_linear_deadband=self.args.base_post_linear_deadband,
            post_angular_deadband=self.args.base_post_angular_deadband,
            enable_single_axis=self.cfg.enable_base_single_axis,
            xy_max_vel=self.cfg.base_xy_max_vel,
            yaw_max_vel=self.cfg.base_yaw_max_vel,
            single_axis_deadband=self.cfg.base_single_axis_deadband,
            dispatch_single_axis_deadband=self.cfg.base_dispatch_single_axis_deadband,
            single_axis_hysteresis_ratio=self.cfg.base_single_axis_hysteresis_ratio,
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
                chassis.set_velocity(vx=0.0, vy=0.0, wz=0.0, wait_time=0.0)
                action = "recenter"  # bad steering read -> safe re-center (logged in /debug)
        else:  # "drive" sends the active twist; "recenter" sends cmd (==0) -> steering 0deg
            chassis.set_velocity(vx=float(cmd[0]), vy=float(cmd[1]), wz=float(cmd[2]),
                                 wait_time=0.0)
        self._dbg["base_pd_raw"] = pd_raw
        self._dbg["base_pd_err"] = np.asarray(err, dtype=float)
        self._dbg["base_cmd"] = np.asarray(cmd, dtype=float)
        self._dbg["base_action"] = action

    def _allow_base_yaw_hold_in_xy(self) -> bool:
        return self.cfg.base_dofs == "xy" and bool(self.args.base_yaw_hold_in_xy)

    def _mask_base_twist(
        self,
        twist: np.ndarray,
        *,
        allow_yaw_hold: bool = False,
    ) -> np.ndarray:
        """Apply the WBC base-DOF policy to a downstream chassis twist."""
        base_cl = getattr(self, "_base_cl", None)
        if base_cl is None:
            from omniteleop.follower import base_closed_loop as base_cl  # noqa: PLC0415
        return base_cl.mask_planar_twist_for_base_dofs(
            twist,
            base_dofs=self.cfg.base_dofs,
            allow_yaw_hold=allow_yaw_hold,
        )

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
        arkit = d.get("arkit")
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
            # iPhone tracker (--arkit-base), NaN when it is not running: the engage-origin
            # pose to difference against odom_pose, the ARKit-world pose that survives
            # re-engages, and the base roll/pitch + jump count that say whether to believe
            # either of them.
            "arkit_pose": vec(arkit["pose"] if arkit else None, 3),
            "arkit_world": vec(arkit["world_pose"] if arkit else None, 3),
            "arkit_rp": vec(arkit["rp"] if arkit else None, 2),
            "arkit_age": np.float32(arkit["age"] if arkit else np.nan),
            "arkit_jumps": np.int64(arkit["jumps"] if arkit else -1),
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
        """Require the head camera before a take can start; wrist cameras are best-effort.

        head_camera is enabled in the --record Robot build and is mandatory: recording
        aborts early if it is unavailable. Each wrist ZED-M needs its own external
        ZED-SDK publisher on ``sensors/<sensor_id>/*``; if a wrist publisher never comes
        up (including both at once), that wrist is marked unavailable in
        ``self._wrist_enabled`` and the take proceeds without its ``{arm}_wrist_rgb``
        stream rather than blocking, so bring-up/testing without the wrist rigs still works.
        """
        if not self.robot.has_sensor(_HEAD_SENSOR_ID):
            raise SystemExit(
                f"[wbc_vr_robot] --record enabled '{_HEAD_SENSOR_ID}' but it is not "
                "available on this robot."
            )
        head_sensor = getattr(self.robot.sensors, _HEAD_SENSOR_ID)
        if (hasattr(head_sensor, "wait_for_active")
                and not head_sensor.wait_for_active(timeout=5.0)):
            raise RuntimeError(
                f"[wbc_vr_robot] --record requires {_HEAD_SENSOR_ID} active before "
                "recording; not active within 5s."
            )
        for arm, sensor_id in _WRIST_SENSOR_IDS.items():
            available = self.robot.has_sensor(sensor_id)
            if available:
                wrist_sensor = getattr(self.robot.sensors, sensor_id)
                if hasattr(wrist_sensor, "wait_for_active"):
                    available = wrist_sensor.wait_for_active(timeout=5.0)
            self._wrist_enabled[arm] = available
            if not available:
                print(
                    f"[wbc_vr_robot] --record: {sensor_id} not active within 5s -- "
                    f"recording WITHOUT {arm}_wrist_rgb this run (needs a ZED-SDK "
                    f"publisher on sensors/{sensor_id}/*)."
                )
        if not any(self._wrist_enabled.values()):
            print(
                "[wbc_vr_robot] --record: BOTH wrist cameras unavailable -- proceeding "
                "with head-only recording."
            )

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
        achieved position, and ``obs/images`` carries
        head_left_rgb/head_depth/{left,right}_wrist_rgb plus the per-frame capture stamps
        (head_frame_ns/head_depth_frame_ns/{left,right}_wrist_frame_ns/grab_wall_ns);
        ``meta/ntp`` stores the once-per-run robot-SoC-vs-local clock offset, while
        ``meta/camera_ntp`` stores camera-publisher-host-vs-local offsets
        (see :meth:`_query_ntp_calibration` and :meth:`_query_camera_clock_calibrations`).
        """
        self._record_period = 0.0
        self._next_record_t = 0.0
        self._stale_grace = float(self.args.record_stale_grace)
        if not np.isfinite(self._stale_grace) or self._stale_grace < 0.0:
            raise ValueError(
                f"--record-stale-grace must be finite and >= 0, got {self._stale_grace}"
            )
        if not self.args.record:
            return
        self._await_record_cameras()
        self._record_period = 1.0 / self.args.record_rate
        # EpisodeRecorder buffers the whole take in RAM (peak ~2x the take, because
        # _recursive_np_stack copies before writing); StreamingEpisodeRecorder appends to
        # resizable datasets through a bounded queue, so peak RAM is the queue (~200 MiB)
        # whatever the take length. Same file, same schema -- see
        # tests/test_streaming_recorder.py, which asserts the two are byte-identical.
        recorder_cls = (
            StreamingEpisodeRecorder if self.args.streaming_recorder else EpisodeRecorder
        )
        self._episode = recorder_cls(self.args.save_dir)
        # Camera intrinsic is constant over a take -> store ONCE (obs/images/intrinsic is
        # (3,3), not (N,3,3)). The per-frame extrinsic is NOT stored: it was base-relative
        # head FK (base-blind, wrong once the base drives) and is fully recomputable, so it
        # is recomputed offline as world_t_cam = world_t_base(obs/base/pose) . FK_cam(obs/
        # joint) -- see scripts/vis_episode.py -- which lands the cloud in the world frame.
        static: dict = {"obs": {"images": {"intrinsic": ZED_K.astype(np.float32)}}}
        static["meta"] = {
            "ntp": self._query_ntp_calibration(),
            "camera_ntp": self._query_camera_clock_calibrations(),
        }
        if self.args.head_right_rgb:
            static["meta"]["head_stereo"] = self._query_head_stereo_calibration()
        static["meta"].update(self._recording_control_metadata())
        self._episode.set_static(static)
        wrist_keys = "+".join(f"{arm}_wrist_rgb" for arm in _WRIST_SENSOR_IDS
                               if self._wrist_enabled[arm])
        head_keys = ("head_left_rgb" if self.args.no_head_depth
                     else "head_left_rgb+head_depth")
        if self.args.head_right_rgb:
            head_keys += "+head_right_rgb"
        print(f"[wbc_vr_robot] recording -> {self.args.save_dir} "
              f"(episode_{self._episode.episode_id}, {self.args.record_rate:g}Hz, "
              f"{'streaming' if self.args.streaming_recorder else 'buffered'}, "
              f"{head_keys}{'+' + wrist_keys if wrist_keys else ' (no wrist)'}"
              ", +torso action, +gripper obs/action, +base pose obs, +frame capture stamps)")

    def _recording_control_metadata(self) -> dict:
        """Stamp the temporary frozen-right-arm mode into every recorded episode."""
        return {
            "right_arm_mode": _bytes_scalar("frozen_nominal"),
            "dead_joints": _bytes_scalar(",".join(self.cfg.dead_joints)),
        }

    def _query_ntp_calibration(self) -> dict:
        """Robot->workstation clock offset, stored once per run as ``meta/ntp``.

        ``offset_ns`` follows dexcontrol's NTP convention (server minus client, i.e.
        robot SoC clock minus THIS machine's clock). This is preserved for diagnostics
        and as a legacy fallback in the audit tool. ZED image stamps are produced by the
        standalone camera publishers, so authoritative absolute camera staleness uses
        ``meta/camera_ntp`` from :meth:`_query_camera_clock_calibrations`.

        Raises before recording starts when unavailable: new takes must carry this
        calibration so timing audits can identify the SoC-vs-camera-host distinction.
        """
        query = getattr(self.robot, "query_ntp", None)
        if query is None:
            raise RuntimeError("[wbc_vr_robot] --record requires Robot.query_ntp().")
        try:
            result = query(sample_count=15)
        except Exception as exc:
            raise RuntimeError(
                f"[wbc_vr_robot] --record query_ntp failed: {type(exc).__name__}: {exc}"
            ) from exc
        offset = float(result.get("offset", np.nan))
        rtt = float(result.get("rtt", np.nan))
        if not result.get("success") or not np.isfinite(offset) or not np.isfinite(rtt):
            raise RuntimeError(f"[wbc_vr_robot] --record query_ntp unusable: {result}")
        print(f"[wbc_vr_robot] NTP calibration: robot(SoC) - local = {offset * 1e3:+.2f} ms "
              f"(rtt {rtt * 1e3:.2f} ms) -> meta/ntp")
        return {
            "offset_ns": np.int64(round(offset * 1e9)),
            "rtt_ns": np.int64(round(rtt * 1e9)),
            "queried_at_ns": np.int64(time.time_ns()),
        }

    def _query_camera_clock_calibrations(self) -> dict:
        """Camera-publisher-host clock offsets, stored under ``meta/camera_ntp``.

        The ZED image ``timestamp_ns`` fields are stamped by the standalone camera
        publishers, not by the robot SoC. Querying ``Robot.query_ntp()`` is still useful
        SoC metadata, but absolute camera staleness needs camera-host minus workstation
        offsets from ``sensors/<sensor_id>/clock``.

        The head clock service is required for new recordings. Each wrist clock service
        is queried only if that wrist camera came up (``self._wrist_enabled``); a
        disabled wrist simply has no ``head``/``left_wrist``/``right_wrist`` entry.
        """
        out: dict = {"head": self._query_camera_clock_calibration("head", _HEAD_SENSOR_ID)}
        for arm, sensor_id in _WRIST_SENSOR_IDS.items():
            if not self._wrist_enabled[arm]:
                continue
            out[f"{arm}_wrist"] = self._query_camera_clock_calibration(f"{arm}_wrist", sensor_id)
        return out

    def _query_camera_clock_calibration(self, label: str, sensor_id: str) -> dict:
        node = getattr(self.robot, "_node", None)
        if node is None:
            raise RuntimeError(
                f"[wbc_vr_robot] --record requires a DexComm node for {label} "
                "camera-host clock calibration."
            )
        topic = _camera_clock_topic(sensor_id)
        try:
            client = node.create_service_client(
                service_name=topic,
                request_encoder=JsonDataCodec.encode,
                response_decoder=JsonDataCodec.decode,
                timeout=_CAMERA_CLOCK_TIMEOUT_S,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[wbc_vr_robot] cannot create {label} camera clock client {topic!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        samples: list[dict] = []
        source = "zed_publisher_host"
        failures = 0
        last_error: Optional[BaseException] = None
        for i in range(_CAMERA_CLOCK_SAMPLE_COUNT):
            t0 = time.time_ns()
            request = {
                "client_send_time_ns": int(t0),
                "sample_count": int(_CAMERA_CLOCK_SAMPLE_COUNT),
                "sample_index": int(i),
            }
            try:
                response = client.call(request)
                t3 = time.time_ns()
                if not isinstance(response, dict):
                    raise ValueError(f"non-dict response {type(response).__name__}")
                t1 = int(response.get("server_receive_time_ns", 0))
                t2 = int(response.get("server_send_time_ns", 0))
                if t1 <= 0 or t2 <= 0 or t2 < t1:
                    raise ValueError(
                        f"invalid server timestamps receive={t1!r} send={t2!r}"
                )
                reply_sensor = response.get("sensor_id")
                if reply_sensor is not None and str(reply_sensor) != sensor_id:
                    raise ValueError(
                        f"sensor_id mismatch: expected {sensor_id!r}, got {reply_sensor!r}"
                    )
                offset_ns, rtt_ns = _ntp_offset_rtt_ns(t0, t1, t2, t3)
                if rtt_ns <= 0:
                    raise ValueError(f"invalid rtt_ns={rtt_ns}")
                samples.append({"offset_ns": offset_ns, "rtt_ns": rtt_ns})
                source = str(response.get("source", source))
                failures = 0
            except Exception as exc:
                last_error = exc
                failures += 1
                if failures >= _CAMERA_CLOCK_MAX_CONSECUTIVE_FAILURES:
                    break
            if i < _CAMERA_CLOCK_SAMPLE_COUNT - 1:
                time.sleep(0.01)

        if len(samples) < _CAMERA_CLOCK_MIN_SAMPLES:
            detail = (
                f"; last error: {type(last_error).__name__}: {last_error}"
                if last_error is not None else ""
            )
            raise RuntimeError(
                f"[wbc_vr_robot] {label} camera clock {topic!r} returned "
                f"{len(samples)}/{_CAMERA_CLOCK_SAMPLE_COUNT} usable samples; need "
                f">= {_CAMERA_CLOCK_MIN_SAMPLES}{detail}."
            )
        queried_at_ns = time.time_ns()
        cal = _camera_clock_calibration_from_samples(
            samples, sensor_id=sensor_id, source=source, queried_at_ns=queried_at_ns
        )
        print(
            f"[wbc_vr_robot] Camera clock calibration {label}/{sensor_id}: "
            f"camera_host - local = {int(cal['offset_ns']) / 1e6:+.2f} ms "
            f"(rtt {int(cal['rtt_ns']) / 1e6:.2f} ms, samples {len(samples)}) "
            f"-> meta/camera_ntp/{label}"
        )
        return cal

    def _query_head_stereo_calibration(self) -> dict:
        """Head stereo baseline + per-eye K, stored once as ``meta/head_stereo``.

        FoundationStereo turns disparity into metric depth with ``fx * baseline / disp``,
        so a stereo take is useless without these two numbers. They are read from the
        publisher's ``sensors/<id>/info`` service rather than hardcoded, because they are
        per-unit factory calibration and because ``left_K``/``right_K`` there already
        carry the publisher's ``--crop``/``--resize`` -- i.e. they describe the frames
        actually recorded, not the raw SVGA sensor.
        """
        node = getattr(self.robot, "_node", None)
        if node is None:
            raise RuntimeError(
                "[wbc_vr_robot] --head-right-rgb requires a DexComm node to query the "
                "head camera's stereo calibration."
            )
        topic = f"sensors/{_HEAD_SENSOR_ID}/info"
        try:
            client = node.create_service_client(
                service_name=topic,
                request_encoder=JsonDataCodec.encode,
                response_decoder=JsonDataCodec.decode,
                timeout=_CAMERA_CLOCK_TIMEOUT_S,
            )
            response = client.call({})
        except Exception as exc:
            raise RuntimeError(
                f"[wbc_vr_robot] --head-right-rgb: cannot query {topic!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(response, dict):
            raise RuntimeError(
                f"[wbc_vr_robot] {topic!r} returned {type(response).__name__}, expected dict."
            )
        stereo = response.get("stereo")
        if not isinstance(stereo, dict):
            raise RuntimeError(
                f"[wbc_vr_robot] {topic!r} has no 'stereo' block. Start the publisher from "
                f"a build that serves it (tests/test_head_zedx_depth.py) -- without the "
                f"baseline a head stereo take cannot be turned into metric depth."
            )
        if not stereo.get("rectified", False):
            raise RuntimeError(
                "[wbc_vr_robot] head stereo calibration is not rectified; "
                "FoundationStereo requires a rectified, row-aligned pair."
            )
        baseline_m = float(stereo["baseline_m"])
        if not 0.01 < baseline_m < 0.5:
            raise ValueError(
                f"[wbc_vr_robot] implausible head stereo baseline {baseline_m} m."
            )
        out = {"baseline_m": np.float32(baseline_m)}
        for eye in ("left_K", "right_K"):
            K = np.asarray(stereo[eye], dtype=np.float32)
            if K.shape != (3, 3):
                raise ValueError(
                    f"[wbc_vr_robot] head stereo {eye} is {K.shape}, expected (3, 3)."
                )
            out[eye] = K
        # The recorded frames are stamped with ZED_K; if the publisher's left_K disagrees,
        # the two are describing different geometry and the depth would be wrong.
        if not np.allclose(out["left_K"], ZED_K.astype(np.float32), rtol=1e-3, atol=1e-2):
            raise ValueError(
                f"[wbc_vr_robot] publisher left_K\n{out['left_K']}\ndisagrees with the "
                f"ZED_K stamped into obs/images/intrinsic\n{ZED_K}\n-- the publisher's "
                f"--crop/--resize no longer match omniteleop.common.head_camera."
            )
        print(
            f"[wbc_vr_robot] Head stereo calibration: baseline "
            f"{baseline_m * 1000:.2f} mm, left fx {float(out['left_K'][0, 0]):.2f} px "
            f"-> meta/head_stereo"
        )
        return out

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

    def _grab_head_images(
        self,
    ) -> Optional[
        tuple[np.ndarray, Optional[np.ndarray], int, Optional[int],
              Optional[np.ndarray], Optional[int]]
    ]:
        """Poll head camera -> ``(left_rgb, depth_u16, rgb_frame_ns, depth_frame_ns,
        right_rgb, right_frame_ns)`` or None.

        With ``--no-head-depth`` the depth stream is never polled and both depth elements
        are ``None`` (RGB-only take; see :meth:`record_tick`). Symmetrically, the right
        eye is polled only with ``--head-right-rgb`` and is ``None`` otherwise, so a
        mono take carries no right key at all rather than a fabricated frame.

        None until every selected stream has delivered a frame (parity with vr_reader's
        ``_all_selected_streams_ready`` -- so the saved HDF5 keeps consistent keys and shapes
        across frames). ``rgb_frame_ns`` is the publisher capture timestamp of the
        ``left_rgb`` frame (``include_timestamp=True``), and ``depth_frame_ns`` is recorded
        separately because the early-RGB publisher path can expose RGB one capture before
        the depth message is ready. record_tick rejects stale RGB frames (camera stalled
        below the record rate) but does not wait for same-frame RGBD. Raises
        ValueError on a malformed shape or a missing timestamp, so a miswired/legacy camera
        fails loudly instead of recording garbage or silently skipping the freshness check.
        """
        want_depth = not self.args.no_head_depth
        want_right = self.args.head_right_rgb
        obs_keys = ["left_rgb"]
        if want_right:
            obs_keys.append("right_rgb")
        if want_depth:
            obs_keys.append("depth")
        obs = self.robot.sensors.head_camera.get_obs(
            obs_keys=obs_keys,
            include_timestamp=True,
        )
        left_entry = obs.get("left_rgb")
        right_entry = obs.get("right_rgb") if want_right else None
        depth_entry = obs.get("depth") if want_depth else None
        if (
            left_entry is None
            or (want_right and right_entry is None)
            or (want_depth and depth_entry is None)
        ):
            return None
        left_rgb, frame_ns = self._unwrap_frame("head_camera left_rgb", left_entry)
        left_rgb = np.asarray(left_rgb)
        if left_rgb.ndim != 3 or left_rgb.shape[2] != 3:
            raise ValueError(
                f"[wbc_vr_robot] head_camera left_rgb shape {left_rgb.shape} is not (H,W,3)."
            )
        left_rgb = np.ascontiguousarray(left_rgb, dtype=np.uint8)
        right_rgb: Optional[np.ndarray] = None
        right_ns: Optional[int] = None
        if want_right:
            right_rgb, right_ns = self._unwrap_frame("head_camera right_rgb", right_entry)
            right_rgb = np.asarray(right_rgb)
            # FoundationStereo matches along image rows, so a right eye that is not the
            # SAME geometry as the left is unusable -- fail here rather than record a
            # pair that silently produces garbage disparity.
            if right_rgb.shape != left_rgb.shape:
                raise ValueError(
                    f"[wbc_vr_robot] head_camera right_rgb shape {right_rgb.shape} != "
                    f"left_rgb {left_rgb.shape}; the publisher must apply the same "
                    f"--crop/--resize to both eyes."
                )
            right_rgb = np.ascontiguousarray(right_rgb, dtype=np.uint8)
        if not want_depth:
            return left_rgb, None, frame_ns, None, right_rgb, right_ns
        depth, depth_ns = self._unwrap_frame("head_camera depth", depth_entry)
        # If a future pointcloud/depth policy requires same-frame RGBD again, restore:
        # if depth_ns != frame_ns:
        #     return None
        depth = np.asarray(depth)
        if depth.ndim != 2:
            raise ValueError(
                f"[wbc_vr_robot] head_camera depth shape {depth.shape} is not (H,W)."
            )
        # meters -> millimeters, clipped to uint16 (matches vr_reader's head_depth).
        depth_u16 = np.clip(depth * 1000, 0, 65535).astype(np.uint16)
        return left_rgb, depth_u16, frame_ns, depth_ns, right_rgb, right_ns

    def _grab_head_rgb(self) -> Optional[tuple[np.ndarray, int]]:
        """Poll only head RGB for a 2-D policy freshness check.

        Recording, SceneDiff, and ManiFlow still use :meth:`_grab_head_images`.
        LeRobot RGB policies call this lightweight path while waiting for a new
        frame, avoiding a cached depth fetch plus float32-metre to uint16-mm
        conversion on every 5 ms poll.  The topic, codec, and publisher capture
        timestamp contract are unchanged.
        """
        obs = self.robot.sensors.head_camera.get_obs(
            obs_keys=["left_rgb"], include_timestamp=True
        )
        entry = obs.get("left_rgb")
        if entry is None:
            return None
        left_rgb, frame_ns = self._unwrap_frame("head_camera left_rgb", entry)
        left_rgb = np.asarray(left_rgb)
        if left_rgb.ndim != 3 or left_rgb.shape[2] != 3:
            raise ValueError(
                f"[wbc_vr_robot] head_camera left_rgb shape {left_rgb.shape} is not (H,W,3)."
            )
        return np.ascontiguousarray(left_rgb, dtype=np.uint8), frame_ns

    def _grab_wrist_image(self, arm: str) -> Optional[tuple[np.ndarray, int]]:
        """Poll one arm's wrist ZED-M -> ``(<arm>_wrist_rgb uint8 HxWx3, frame_ns)`` or None.

        ``arm`` is "left" or "right" and names the ARM the camera is mounted on; the stream
        pulled is always that camera's stereo LEFT eye (``_WRIST_OBS_KEY``).

        Consumes the publisher's frame AS-IS (no crop/resize), like vr_reader: the wrist
        ZED-SDK publisher (tests/test_wrist_zedm_depth.py) already resized robot-side.
        Returns None until a frame arrives. ``frame_ns`` is the publisher capture timestamp
        (``include_timestamp=True``); record_tick rejects a stale frame (publisher stalled
        below the record rate). A momentary drop is deliberately NOT papered over with a
        cached frame -- a duplicate wrist image in the take is exactly what the freshness
        guard must catch. Raises ValueError on a malformed shape or a missing timestamp.
        """
        sensor_id = _WRIST_SENSOR_IDS[arm]
        obs = getattr(self.robot.sensors, sensor_id).get_obs(
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
        wrist, frame_ns = self._unwrap_frame(f"{sensor_id} {_WRIST_OBS_KEY}", entry)
        wrist = np.asarray(wrist)
        if wrist.ndim != 3 or wrist.shape[2] != 3:
            raise ValueError(
                f"[wbc_vr_robot] {sensor_id} {_WRIST_OBS_KEY} shape {wrist.shape} "
                "is not (H,W,3)."
            )
        return np.ascontiguousarray(wrist, dtype=np.uint8), frame_ns

    def _retry_or_abort_stale(
        self,
        now: float,
        *,
        head_ns: Optional[int],
        wrist_ns: dict[str, Optional[int]],
    ) -> None:
        """Retry-or-abort for a record tick whose camera frames did not all advance.

        Called mid-take when a stream is missing or its publisher timestamp is not
        strictly newer than the last recorded frame. Within ``--record-stale-grace`` of
        the streak's first stale tick the record schedule is pulled back to ``now``, so
        the IK loop re-polls the cameras on its very next tick (non-blocking: actuation
        cadence is untouched, and nothing is recorded until every timestamp advances).
        Past the grace it aborts with a per-camera Δ/age line plus the arrival ages of
        the recently recorded frames, so the console shows WHICH stream stalled and
        whether its delivery latency was already climbing.
        """
        wall_ns = time.time_ns()

        def diag(name: str, ts: Optional[int], last_ns: int) -> str:
            if ts is None:
                return f"{name}=MISSING"
            return f"{name} Δ={ts - last_ns} ns age={(wall_ns - ts) / 1e6:.0f} ms"

        detail = ", ".join((
            diag("head", head_ns, self._last_rec_head_ns),
            *(
                diag(f"{arm}_wrist", wrist_ns[arm], self._last_rec_wrist_ns[arm])
                for arm in _WRIST_SENSOR_IDS
                if self._wrist_enabled[arm]
            ),
        ))
        if self._stale_since is None:
            self._stale_since = now
            print(f"\n[wbc_vr_robot] --record: no fresh camera frame at the record tick "
                  f"({detail}); retrying at the IK rate for up to "
                  f"{self._stale_grace * 1000.0:.0f} ms.")
        if now - self._stale_since <= self._stale_grace:
            # Pull the schedule back so the next IK tick retries immediately; once frames
            # resume, the throttle re-anchors and the take continues with one late frame.
            self._next_record_t = now
            return
        history = "".join(
            f"\n    t-{now - t:5.2f}s  head_age={h:7.1f} ms  "
            f"left_wrist_age={lw:7.1f} ms  right_wrist_age={rw:7.1f} ms"
            for t, h, lw, rw in list(self._frame_age_log)[-20:]
        )
        raise RuntimeError(
            "[wbc_vr_robot] --record: stale camera frame for "
            f"{(now - self._stale_since) * 1000.0:.0f} ms (> --record-stale-grace "
            f"{self._stale_grace * 1000.0:.0f} ms) -- {detail} (need every Δ > 0). The "
            "publishers hold their fps through these stalls, so suspect the delivery "
            "path: WiFi burst, Zenoh backlog, or this process starving its Zenoh decode "
            "threads under solver load. Aborting so the take never records duplicate "
            "frames. Arrival ages of the last recorded frames (publisher stamp vs local "
            "clock; spans the robot NTP offset):"
            f"{history if history else ' <none recorded yet>'}"
        )

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

        active_wrist_arms = [arm for arm in _WRIST_SENSOR_IDS if self._wrist_enabled[arm]]
        imgs = self._grab_head_images()
        wrist = {
            arm: (self._grab_wrist_image(arm) if self._wrist_enabled[arm] else None)
            for arm in _WRIST_SENSOR_IDS
        }
        # Before the first recorded frame the cameras may still be warming up (the leader can
        # reach teleop before a publisher is fully up): tolerate a missing stream and skip
        # this tick, matching vr_reader. Only ACTIVE wrists (self._wrist_enabled) are
        # required here -- a wrist whose publisher never came up this run (see
        # _await_record_cameras) is permanently None and must not block every tick forever.
        recording_started = self._last_rec_head_ns >= 0
        if (
            imgs is None or any(wrist[arm] is None for arm in active_wrist_arms)
        ) and not recording_started:
            return  # camera still warming up -- skip rather than write inconsistent keys
        # Freshness guard: ~15 fps camera vs ~10 fps record means every recorded frame must
        # carry a strictly newer publisher timestamp than the last recorded one. A non-newer
        # (or missing) frame is the delivery path repeating the cached latest -- the
        # publishers hold their fps through these stalls, so suspect a WiFi burst, Zenoh
        # backlog, or this process starving its Zenoh decode threads under solver load.
        # Instead of dying on the first stale tick, retry at the IK rate for up to
        # --record-stale-grace: nothing is recorded until EVERY timestamp advances, so a take
        # still never contains duplicates; a short stall costs one late frame, not the take.
        # A disabled wrist has no timestamp at all and is excluded from this gate.
        head_ns = None if imgs is None else imgs[2]
        wrist_ns: dict[str, Optional[int]] = {
            arm: (None if w is None else w[1]) for arm, w in wrist.items()
        }
        if (
            head_ns is None
            or head_ns <= self._last_rec_head_ns
            or any(
                wrist_ns[arm] is None or wrist_ns[arm] <= self._last_rec_wrist_ns[arm]
                for arm in active_wrist_arms
            )
        ):
            self._retry_or_abort_stale(now, head_ns=head_ns, wrist_ns=wrist_ns)
            return
        if self._stale_since is not None:
            deltas = ", ".join(
                f"{arm} wrist Δ={wrist_ns[arm] - self._last_rec_wrist_ns[arm]} ns"
                for arm in active_wrist_arms
            )
            print(f"\n[wbc_vr_robot] --record: fresh frames resumed after "
                  f"{(now - self._stale_since) * 1000.0:.0f} ms (head Δ="
                  f"{head_ns - self._last_rec_head_ns} ns, {deltas}).")
            self._stale_since = None
        head_left_rgb, head_depth_u16, _, head_depth_ns, head_right_rgb, head_right_ns = imgs

        # Stereo eye pairing. FoundationStereo matches the two eyes of ONE capture, and
        # calib/packing_data/scripts/fs_head_depth.py drops any frame whose
        # head_right_frame_ns differs from head_frame_ns. The right eye arrives as its own
        # message, typically a beat after the left, so a single poll at the record tick
        # catches the PREVIOUS right frame about half the time -- measured 54.5% paired
        # over a 277-frame take, i.e. 45% of the head depth discarded offline. Retry at the
        # IK rate, exactly like the freshness guard above, to let the match land.
        #
        # Two deliberate differences from that guard:
        #   * it never aborts. An unpaired right eye costs one frame of head depth offline;
        #     it does not put a duplicate in the take. Past the budget the frame is recorded
        #     as-is, which is precisely the behaviour before this gate existed -- so the
        #     worst case here is the old behaviour, never a lost take.
        #   * the budget is ONE RECORD PERIOD, not --record-stale-grace, so a publisher that
        #     never pairs cannot cost more than one period per frame; _RIGHT_PAIR_GIVE_UP
        #     consecutive such frames then disarm the retry for the rest of the episode
        #     rather than halving the record rate in silence.
        if head_right_ns is not None and head_right_ns != head_ns:
            if self._right_pair_retry:
                if self._right_unpaired_since is None:
                    self._right_unpaired_since = now
                if now - self._right_unpaired_since <= min(self._stale_grace,
                                                           self._record_period):
                    self._next_record_t = now  # re-poll on the next IK tick
                    return
            self._rec_right_unpaired += 1
            self._right_unpaired_streak += 1
            if (self._right_pair_retry
                    and self._right_unpaired_streak >= _RIGHT_PAIR_GIVE_UP):
                self._right_pair_retry = False
                print(f"\n[wbc_vr_robot] --record: the head right eye did not pair with the "
                      f"left on {self._right_unpaired_streak} consecutive frames -- the "
                      f"publisher is not stamping the two eyes alike. Disarming the pairing "
                      f"retry for this episode; frames are still recorded, but "
                      f"fs_head_depth.py will skip them.")
        else:
            self._right_unpaired_streak = 0
        self._right_unpaired_since = None

        def _age_ms(ts: Optional[int], wall_ns: int) -> float:
            return float("nan") if ts is None else (wall_ns - ts) / 1e6

        # Arrival-age history for the stale-abort diagnostic: local wall clock minus the
        # publisher capture stamp (spans the camera-host->workstation clock offset, so
        # compare head vs wrist and the trend over time, not the absolute value). A
        # disabled wrist logs NaN rather than a fabricated age.
        wall_ns = time.time_ns()
        self._frame_age_log.append((
            now,
            _age_ms(head_ns, wall_ns),
            _age_ms(wrist_ns["left"], wall_ns),
            _age_ms(wrist_ns["right"], wall_ns),
        ))
        self._last_rec_head_ns = head_ns
        # Every ACTIVE arm's value is non-None past the freshness gate above; a disabled
        # arm's -1 sentinel is left untouched (never read once permanently disabled).
        for arm in active_wrist_arms:
            self._last_rec_wrist_ns[arm] = int(wrist_ns[arm])

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
            # Publisher capture stamps (camera-publisher-host clock) of THIS tick's frames plus the
            # local wall clock right after all grabs -- the raw material for offline
            # camera-latency audits/correction (scripts/audit_episode_latency.py, with
            # meta/camera_ntp when available). The freshness gate above requires head RGB and
            # every ACTIVE wrist stamp to advance; depth's stamp is recorded separately for
            # RGBD audits.
            "head_frame_ns": np.int64(head_ns),
            "grab_wall_ns": np.int64(wall_ns),
            # intrinsic is recorded once via set_static (not per frame); extrinsic is NOT
            # recorded -- world_t_cam is recomputed offline from obs/base/pose + obs/joint.
        }
        # --no-head-depth takes are RGB-only: no head_depth/head_depth_frame_ns keys at all
        # (like a wrist whose publisher never came up), never a faked or zeroed depth frame.
        if head_depth_u16 is not None:
            images["head_depth"] = head_depth_u16
            images["head_depth_frame_ns"] = np.int64(head_depth_ns)
        # --head-right-rgb adds the rectified right eye so FoundationStereo can run on the
        # head offline (processing.fs_stereo.run_fs). Same all-or-nothing rule as depth: a
        # mono take has no head_right_rgb/head_right_frame_ns keys at all. The stereo
        # baseline + per-eye K live once in meta/head_stereo, not per frame.
        if head_right_rgb is not None:
            images["head_right_rgb"] = head_right_rgb
            images["head_right_frame_ns"] = np.int64(head_right_ns)
        # left/right = the ARM each camera is mounted on (see _WRIST_SENSOR_IDS). A wrist
        # whose publisher never came up this run (self._wrist_enabled) simply has no
        # {arm}_wrist_rgb / {arm}_wrist_frame_ns key in this take, rather than a faked frame.
        for arm in active_wrist_arms:
            images[f"{arm}_wrist_rgb"] = wrist[arm][0]
            images[f"{arm}_wrist_frame_ns"] = np.int64(wrist_ns[arm])
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
            if self._arkit is not None:
                # Drift-free counterpart of obs/base/pose from the mount-rigid iPhone
                # (--arkit-base). ``pose_arkit`` shares obs/base/pose's engage-origin frame
                # so the two difference directly into the odometry drift; ``_world`` is the
                # ARKit session frame, which -- unlike the engage origin -- is continuous
                # across a re-engage inside one take. NaN while the tracker has no origin
                # or no sample yet (same convention as obs/gripper before its first reply).
                arkit = self._arkit.snapshot()
                obs_out["base"]["pose_arkit"] = np.asarray(arkit["pose"], dtype=np.float32)
                obs_out["base"]["pose_arkit_world"] = np.asarray(
                    arkit["world_pose"], dtype=np.float32)
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
        if self._arkit is not None:
            self._arkit.close()
        # Flush the episode BEFORE shutting the robot down. EpisodeRecorder saves in a
        # daemon thread, so block here until it finishes -- otherwise process exit could
        # kill the save mid-write.
        self.stop_recording_episode()
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
    # HW-WORKAROUND(right-arm-frozen): fail before hardware connection if the temporary
    # config no longer pins every right-arm DOF. Hardware dispatch is nominal regardless,
    # so allowing the solver to move an omitted joint would make its FK diverge from reality.
    missing_right_pins = [name for name in RIGHT_ARM_JOINTS if name not in cfg.dead_joints]
    if missing_right_pins:
        raise ValueError(
            "wbik_tmp.yaml must pin every right-arm joint at nominal; missing "
            f"{missing_right_pins}"
        )
    # head_mode "ik" pins cfg.head_ik_pinned_joints (wbik.yaml; default head_j1/head_j2)
    # out of the QP -- VegaWholeBodyIK builds the pin from the config, so this real-robot
    # follower and the sim follower (wbc_vr_record.py) fix the SAME head DOFs.
    ik = VegaWholeBodyIK(cfg)
    if cfg.head_mode == "ik":
        print(f"[wbc_vr_robot] head IK pin (wbik.yaml): {list(cfg.head_ik_pinned_joints)} "
              "fixed; other head DOFs tracked.")
    # Make this nonstandard actuation mode impossible for an operator to overlook.
    extra_pins = [name for name in cfg.dead_joints if name not in RIGHT_ARM_JOINTS]
    print("\n" + "!" * 78)
    print(f"[wbc_vr_robot] TEMPORARY RIGHT-ARM-FROZEN MODE -- "
          f"config {WBIK_TMP_CONFIG_PATH.name}")
    print(f"[wbc_vr_robot]   RIGHT ARM HELD AT NOMINAL: {list(RIGHT_ARM_JOINTS)}")
    print("[wbc_vr_robot]   LEFT ARM: all 7 joints active")
    if ik.right_ee_task_active:
        print("[wbc_vr_robot]   BOTH L_ee and R_ee tasks remain active in whole-body IK;")
        print("[wbc_vr_robot]   R_ee influences only the remaining movable whole-body DOFs.")
    else:
        print("[wbc_vr_robot]   R_ee task DROPPED from the QP: the whole chain to R_ee is")
        print("[wbc_vr_robot]   pinned, so it could only anchor the base against head")
        print("[wbc_vr_robot]   tracking. err R on the HUD is now an unserved residual.")
    if extra_pins:
        print(f"[wbc_vr_robot]   ADDITIONAL NOMINAL PINS: {extra_pins}")
    print("[wbc_vr_robot]   Use a separate --save-dir; recorded right-arm actions are nominal.")
    print("!" * 78 + "\n")
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
    last_home_request_ns = -1
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
          f"enable={[k for k, v in enable.items() if v]} "
          f"grippers={'on (detached: no obs)' if args.no_grippers else 'on'}")

    def resync_to_nominal(reason: str) -> None:
        """Re-anchor the solver and the target streams on the nominal posture.

        The engage transition and a home request both leave the HARDWARE at nominal, so
        the model, the interpolator/filters and the odometry origin have to be put back
        there in lock-step. Without this after a home request the solver kept tracking the
        pre-home targets from its stale configuration, so the leader HUD kept bannering
        that posture's self-collision warning while the robot stood at nominal.
        """
        nonlocal left_cmd, right_cmd, head_cmd, last_cmd_ns
        ik.reset()
        interp.reset(left0, right0, head0)
        head_lpf.reset(head0)
        head_planar_deadband.reset(head0)
        left_cmd, right_cmd, head_cmd = left0.copy(), right0.copy(), head0.copy()
        last_cmd_ns = -1
        driver.engage_reset(ik, left0, right0, head0)
        print(f"\n[wbc_vr_robot] {reason}: reset IK to nominal.")

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
        if estop and hasattr(driver, "stop_recording_episode"):
            driver.stop_recording_episode()
        if (
            vr is not None
            and bool(getattr(vr, "home_requested", False))
            and int(getattr(vr, "timestamp_ns", -1)) != last_home_request_ns
        ):
            last_home_request_ns = int(getattr(vr, "timestamp_ns", -1))
            if hasattr(driver, "home_to_nominal"):
                driver.home_to_nominal()
                # Homing moves the HARDWARE only. Re-anchor the solver on nominal too, or
                # it keeps solving the pre-home targets and the gate keeps reporting that
                # posture's self-collision distance until the next engage.
                resync_to_nominal("home")
        if prev_estop and not estop:
            resync_to_nominal("engage")
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
        if not replay:
            # Episode replay re-issues action/head, which THIS loop already wrote through
            # both filters at record time. Re-filtering would double-apply the LPF lag
            # (a 2nd-order head response the original take never had), so a replay with an
            # unedited wbik_tmp.yaml would not reproduce its own source. See the --replay
            # note in the module docstring: head_lpf_tau / head_planar_*_deadband are
            # baked into the file and are not comparable on this path.
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
                episode_id=_recorder_episode_id(driver),
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
          f"base_dofs={cfg.base_dofs} urdf={cfg.urdf_path.rsplit('/', 1)[-1]}")
    replay = args.replay is not None
    clock: Callable[[], float] = time.perf_counter
    realtime = True

    if replay:
        source = ReplaySource.from_episode_hdf5(args.replay, args.speed, clock=clock)
        print(f"[wbc_vr_robot] replay {args.replay}: {len(source._vr)} frames, "  # noqa: SLF001
              f"{source.total_duration:.1f}s at {args.speed:g}x (recorded action targets; "
              "head LPF/deadband bypassed -- already baked into the take)")
    else:
        source = VRJointSubscriber(args.namespace, name="wbc_vr_robot")
        print(f"[wbc_vr_robot] live on '{source.topic}'. Ctrl-C to stop.")

    enabled = [k for k, v in enable.items() if v]
    debug_path: Optional[str] = None
    debug_start_id = 0
    traj: Optional[_TrajLog] = None
    if args.debug_dir is not None:
        import os  # noqa: PLC0415

        os.makedirs(args.debug_dir, exist_ok=True)
        # Match vr_reader's DebugEpisodeRecorder naming. With --record, each saved
        # episode_<N>.hdf5 gets its episode_<N>_debug.hdf5 flushed (buffer freed) as it
        # saves, keyed by the recorder's ACTUAL id -- so this peek is only the STARTING id
        # (for the print, and for the single teardown file when --record is off).
        if args.record:
            debug_start_id = peek_next_episode_id(args.save_dir)
        else:
            debug_start_id = peek_next_episode_id(args.debug_dir, suffix="_debug")
        debug_path = os.path.join(args.debug_dir, f"episode_{debug_start_id}_debug.hdf5")
        meta = {
            "episode_id": int(debug_start_id),
            "mode": "replay" if replay else "live",
            "speed": float(args.speed) if replay else 1.0,
            "ik_rate": float(args.ik_rate),
            "cmd_rate": float(args.cmd_rate),
            "enable": ",".join(enabled),
            # HW-WORKAROUND(right-arm-frozen): stamp the nonstandard actuation mode into
            # every debug HDF5 so it cannot be mistaken for full dual-arm hardware data.
            "right_arm_mode": "frozen_nominal",
            "dead_joints": ",".join(cfg.dead_joints),
            "urdf": cfg.urdf_path.rsplit("/", 1)[-1],
            "base_dofs": cfg.base_dofs,
            "base_single_axis_deadband": float(cfg.base_single_axis_deadband),
            "base_single_axis_hysteresis_ratio": float(
                cfg.base_single_axis_hysteresis_ratio
            ),
            "base_kp_xy": float(args.base_kp_xy),
            "base_kp_yaw": float(args.base_kp_yaw),
            "base_yaw_hold_in_xy": bool(args.base_yaw_hold_in_xy),
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
        traj = _TrajLog(args.debug_dir, meta=meta)
    print("=" * 72)
    print(f"[wbc_vr_robot] REAL ROBOT. enabled={enabled} "
          f"grippers={'on (detached: no obs)' if args.no_grippers else 'on'} | "
          f"{'REPLAY ' + format(args.speed, 'g') + 'x' if replay else 'LIVE'} | "
          f"base_max={args.base_max_speed:g}m/s | base_dofs={cfg.base_dofs} | "
          f"yaw_hold_xy={bool(args.base_yaw_hold_in_xy)}")
    print(f"  filters -> head_lpf={args.head_lpf_tau:g}s, "
          f"head_planar_deadband={args.head_planar_pos_deadband:g}m/"
          f"{args.head_planar_yaw_deadband:g}rad, "
          f"base_post_deadband={args.base_post_linear_deadband:g}m/s/"
          f"{args.base_post_angular_deadband:g}rad/s")
    if args.arkit_base != "off":
        print(f"  base pose -> iPhone/ARKit '{args.arkit_base}' "
              f"({'closes the base PD loop' if args.arkit_base == 'control' else 'recorded only'}"
              "; wheel odometry always recorded).")
    print(f"  joints -> dexcontrol 1:1 (urdf {cfg.urdf_path.rsplit('/', 1)[-1]}). "
          "Robot homes to nominal, then moves. KEEP CLEAR. Ctrl-C aborts (zeros base).")
    if debug_path is not None:
        if args.record:
            print(f"  /debug -> {args.debug_dir}/episode_<N>_debug.hdf5 "
                  "(one per recorded episode; buffer freed as each saves)")
        else:
            print(f"  /debug -> {debug_path}")
    if args.record:
        head_streams = ("head_left_rgb" if args.no_head_depth
                        else "head_left_rgb+head_depth")
        if args.head_right_rgb:
            head_streams += "+head_right_rgb"
        print(f"  recording -> {args.save_dir} @ {args.record_rate:g}Hz while engaged "
              f"({head_streams}+left_wrist_rgb+right_wrist_rgb, +torso action, "
              "+gripper obs/action).")
    print("=" * 72)

    driver = None
    try:
        # Construct INSIDE the try so a HardwareDriver init failure (e.g. failed homing
        # gate) still runs the finally cleanup. HardwareDriver.__init__ also self-cleans.
        # Pass the /debug log so stop_recording_episode() flushes a paired
        # episode_<N>_debug.hdf5 per saved episode (and frees the buffer). None when
        # --debug-dir is omitted -> logging fully disabled.
        driver = HardwareDriver(args, ik, cfg, enable, traj=traj)
        run_loop(source, ik, cfg, driver, args, replay=replay, clock=clock,
                 realtime=realtime, enable=enable, traj=traj)
    except KeyboardInterrupt:
        print("\n[wbc_vr_robot] interrupted -- stopping.")
    except Exception as exc:
        # Tell the leader HUD the follower is DEAD (red banner) before teardown --
        # otherwise the operator only sees a stale "Stage: teleop" while the robot
        # freezes (e.g. the --record camera-freshness abort). Then re-raise.
        _publish_abort_status(source, exc)
        raise
    finally:
        if hasattr(source, "close"):
            source.close()
        if driver is not None:
            driver.close()
        if traj is not None:
            if args.record:
                # Per-episode logs were flushed as each episode saved (bounded RAM), and
                # driver.close() above flushed any episode still recording at exit. Only
                # trailing idle ticks (after the last save, tied to no saved episode)
                # remain -- dropped rather than written as a phantom episode_<N>_debug.hdf5.
                if len(traj):
                    print(f"[wbc_vr_robot] /debug: dropped {len(traj)} trailing idle "
                          "tick(s) not tied to any saved episode.")
            else:
                # No episode boundaries: one /debug log for the whole run.
                print(f"\n[wbc_vr_robot] summary: {traj.summary()}")
                traj.flush(debug_start_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--replay", metavar="FILE",
                      help="re-drive an episode_<N>.hdf5 recorded by THIS script's "
                           "--record, time-scaled by vr_teleop.replay_speed in "
                           "follower/wbik_tmp.yaml. Only the take's original solver inputs "
                           "are consumed (action/eef/{left,right}, action/head, "
                           "action/gripper/*); everything else is re-solved with the "
                           "current follower/wbik_tmp.yaml, so the same take can be "
                           "compared across parameter edits. The recorded head target is "
                           "fed to the IK unfiltered (record-time head LPF/deadband "
                           "already applied).")
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
                             "head_left_rgb/head_depth/{left,right}_wrist_rgb (left/right = "
                             "ARM), a static intrinsic, "
                             "per-frame capture stamps + a once-per-run meta/ntp clock offset "
                             "(latency audits), and (with --enable base) the obs/base/pose "
                             "used to rebuild world_t_cam offline. Each wrist camera needs "
                             "its own ZED-SDK publisher on sensors/{left,right}_wrist_zedm/*. "
                             "OFF by default.")
    parser.add_argument("--streaming-recorder", action="store_true",
                        help="stream each frame to HDF5 as the take runs instead of "
                             "buffering the whole episode in RAM. Peak RAM becomes the "
                             "writer queue (~200 MiB) instead of ~2x the take (a 5-minute "
                             "600x960 RGB+depth take buffers ~16 GiB), and a crash leaves "
                             "episode_<N>.hdf5.partial rather than losing the take. The "
                             "written file is identical either way. Requires the frame key "
                             "set to be fixed for the run, which it already is.")
    parser.add_argument("--no-head-depth", action="store_true",
                        help="record head_left_rgb ONLY: the head depth stream is never "
                             "polled and the take has no head_depth/head_depth_frame_ns "
                             "keys. Pair with the publisher's --no-enable-depth "
                             "(tests/test_head_zedx_depth.py) to also drop the SDK depth "
                             "compute, the PNG encode and ~50 Mbps from the head path. "
                             "RGB-only takes cannot feed the point-cloud porters "
                             "(port_wbc_mobile_zarr.py). Requires --record.")
    parser.add_argument("--head-right-rgb", action="store_true",
                        help="also record the head's RECTIFIED right eye as "
                             "head_right_rgb/head_right_frame_ns, so FoundationStereo can "
                             "compute head depth offline (processing.fs_stereo.run_fs) "
                             "instead of using the ZED SDK's on-device depth. Requires the "
                             "publisher to be started with --no-skip-right-rgb "
                             "(tests/test_head_zedx_depth.py), which also makes it serve "
                             "the stereo baseline + per-eye K this recorder stores in "
                             "meta/head_stereo. Roughly doubles head RGB bandwidth. "
                             "Requires --record.")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR,
                        help=f"directory for recorded episodes (default {DEFAULT_SAVE_DIR}).")
    parser.add_argument("--record-rate", type=float, default=DEFAULT_RECORD_RATE,
                        help=f"episode record cadence in Hz (default {DEFAULT_RECORD_RATE:g}; "
                             "must be <= the IK rate (ik_rate in wbik.yaml)).")
    parser.add_argument("--record-stale-grace", type=float,
                        default=DEFAULT_RECORD_STALE_GRACE,
                        help="seconds to keep retrying (at the IK rate) when a record tick "
                             "finds no strictly-newer frame from every camera, before "
                             f"aborting the take (default {DEFAULT_RECORD_STALE_GRACE:g}; "
                             "0 aborts on the first stale tick). Rides out short WiFi/Zenoh "
                             "delivery stalls without ever recording duplicate frames.")

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
    hw.add_argument("--no-grippers", action="store_true",
                    help="the Robotiq grippers are physically detached: skip the FC03 "
                         "achieved-position warmup/monitors that --record otherwise "
                         "requires (obs/gripper stays NaN for the whole take). The "
                         "activation and per-tick trigger writes still go out; they are "
                         "harmless with nothing on the EE bus. OFF by default.")
    hw.add_argument("--source-timeout", type=float, default=DEFAULT_SOURCE_TIMEOUT,
                    help=f"hold (zero base, freeze joints) if no fresh command/odom for this "
                         f"many seconds (default {DEFAULT_SOURCE_TIMEOUT:g}).")
    hw.add_argument("--drive-state-mode", choices=("ms", "rad"), default="ms",
                    help="swerve wheel_velocity units for odometry (default 'ms').")
    hw.add_argument("--arkit-base", dest="arkit_base",
                    choices=("off", "record", "control"), default="off",
                    help="use the mount-rigid iPhone (ARKit) as a drift-free base-pose "
                         "sensor. 'record': log it beside the wheel odometry "
                         "(obs/base/pose_arkit + pose_arkit_world, arkit_* /debug columns) "
                         "while odometry still closes the base PD loop -- measure the drift "
                         "before depending on it. 'control': also feed it to the base PD as "
                         "the MEASURED pose, holding the base on a stale ('stale-arkit') or "
                         "relocalized ('arkit-jump') tracker. Needs --enable base, "
                         "$ROBOT_NAME, and the robot-side publisher "
                         "(record3d/tracking/base_pose_pub.py). Default 'off'.")
    hw.add_argument("--base-quiet-hold-s", type=float, default=DEFAULT_BASE_QUIET_HOLD_S,
                    help="on a quiet (zero) base command, hold the current swerve steering "
                         "(zero drive) for this long before re-centering the wheels to 0deg; "
                         f"negative disables quiet-gap re-centering (default "
                         f"{DEFAULT_BASE_QUIET_HOLD_S:g}; 0 = re-center on every quiet tick, "
                         "the original behavior).")
    args = parser.parse_args()
    # Control-loop tunables sourced SOLELY from wbik.yaml's vr_teleop: block
    # (VRTeleopConfig, loaded into the DEFAULT_* constants above); no longer
    # CLI-overridable. Bind them onto args -- the same post-parse mutation the script
    # already does for --debug-dir -- so run_loop / HardwareDriver / the /debug meta read
    # one namespace and YAML stays the single source. VRTeleopConfig.from_yaml already
    # validated each value (finite, >= 0, or > 0 for replay_speed/ik_rate).
    bind_vr_teleop_args(args, _VR_TELEOP)

    for flag, val in (("--home-tol", args.home_tol), ("--max-joint-step", args.max_joint_step),
                      ("--home-settle", args.home_settle),
                      ("--record-stale-grace", args.record_stale_grace)):
        if not np.isfinite(val) or val < 0.0:
            parser.error(f"{flag} must be finite and >= 0")
    if not np.isfinite(args.base_quiet_hold_s):
        parser.error("--base-quiet-hold-s must be finite")
    for flag, val in (("--source-timeout", args.source_timeout),
                      ("--record-rate", args.record_rate)):
        if not np.isfinite(val) or val <= 0.0:
            parser.error(f"{flag} must be finite and > 0")
    if args.no_head_depth and not args.record:
        parser.error("--no-head-depth only affects recording; pass it with --record")
    if args.head_right_rgb and not args.record:
        parser.error("--head-right-rgb only affects recording; pass it with --record")
    if args.record and args.record_rate > args.ik_rate:
        parser.error("--record-rate must be <= the IK rate (ik_rate in wbik.yaml; "
                     "cannot record faster than the loop)")
    try:
        enable = parse_enable_mask(args.enable)
    except ValueError as exc:
        parser.error(str(exc))
    if args.arkit_base != "off" and not enable["base"]:
        parser.error("--arkit-base needs --enable base: the iPhone tracker rides alongside "
                     "the wheel odometry (which still supplies obs/joint/chassis_* and the "
                     "/debug wheel inputs), and it never drives a base that is not enabled.")
    _run_ik_mode(args, enable)


if __name__ == "__main__":
    main()
