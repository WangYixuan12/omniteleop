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

  1. ``--replay FILE``   re-drive an episode THIS script recorded (``--record``).
  2. ``--source live``   subscribe to the live leader and drive the robot in real time.

``--replay`` is a PARAMETER-COMPARISON tool: it consumes an ``EpisodeRecorder`` take
(``episode_<N>.hdf5``) and re-issues only that take's ORIGINAL solver inputs -- the
world-frame ``action/eef/{left,right}`` + ``action/head`` 4x4 targets and the
``action/gripper/*`` trigger commands -- at their recorded cadence, time-scaled by
``replay_speed``. Everything the follower derived from them (joint commands, base twist)
is dropped and re-solved, so running the same file twice across an edit to
``follower/wbik.yaml`` isolates the effect of that edit on one fixed operator intent.
Because the recorded ``action/head`` was ALREADY written through the head low-pass +
planar deadband at record time, replay feeds it to the IK unfiltered (re-filtering would
double-apply the LPF lag): ``head_lpf_tau`` and ``head_planar_*_deadband`` are baked into
the file and are the one group of knobs this path cannot compare.

All control-loop tunables -- replay speed, IK / command rates, head LPF + planar
deadbands, base PD gains, base slew / deadband / post-deadband -- come SOLELY from the
``vr_teleop:`` block of ``follower/wbik.yaml`` (shared with ``scripts/wbc_vr_record.py``),
so a take recorded/visualized in sim drives the robot identically. They are not CLI flags.

Pass ``--debug-dir DIR`` to write a per-tick ``/debug`` HDF5 (auto-named
``episode_<N>_debug.hdf5`` under ``DIR``, matching ``vr_reader`` / ``vis_teleop_curves``):
the base PD chain (odom pose +
raw wheels, IK base pose/twist, PD raw/err, shaped command), plus per-group joint
command/sent/measured -- enough to inspect commanded vs measured motion. OFF by default.

Set ``lock_torso_in_ik: true`` in ``follower/wbik.yaml`` (or pass
``--lock-torso-in-ik`` as a run-specific override) to hold all three torso joints at
the solver's reset posture and remove them as whole-body IK variables. The torso remains
in FK and safety calculations; the base, arms, and available head joints solve around
its fixed pose.

``--record`` starts automatically on the first engage after VR calibration, matching
``scripts/wbc_vr_record.py``'s post-calibration recording gate. It saves an episode in
the ``EpisodeRecorder`` schema of ``leader/vr_reader.py`` (mostly interchangeable):
``action``/``obs`` joints plus ``head_left_rgb``/``head_depth``/``{left,right}_wrist_rgb``
(``left``/``right`` = the ARM the wrist camera is on, not the stereo eye) and the
``left``/``right`` gripper under ``action/gripper`` (leader trigger command) and
``obs/gripper`` (Robotiq FC03 achieved position). It adds ``torso`` to ``action/joint``
(the WBC commands the torso) and uses the shaped base twist for the ``chassis_*`` action;
``action/joint/*`` records the POST-CLAMP joint commands actually sent to hardware
(``_dbg['sent_joints']``), not the raw WBC result. Recording requires the FULL
``--enable torso,arms,head,base`` mask: it records the MEASURED wheel-odometry body twist
as the ``obs/joint`` ``chassis_*`` (achieved-velocity counterpart to the action) and the
canonical control pose ``obs/base/pose`` ``(x, y, yaw)`` in the engage-origin world frame.
That pose is wheel odometry normally and ARKit with ``--arkit-base control``; the source
is explicit in ``meta/obs_base_pose_source``, and ``obs/base/pose_odom`` always retains
wheel odometry for diagnostics.

``--arkit-base {off,record,control}`` adds the drift-free base pose tracked by the
iPhone rigidly mounted to the E-stop tower (ARKit via Record3D over USB; the robot-side
publisher is ``record3d/tracking/base_pose_pub.py`` and the phone->base extrinsic comes
from ``record3d/calibration/calib_solve.py``). Swerve odometry integrates steering and
wheel velocity, so its yaw walks away over a take and drags every world-frame quantity
derived from it when it is the control source; the phone measures the base pose directly
instead. ``record`` logs it alongside the wheel
odometry (``obs/base/pose_arkit`` engage-origin + ``obs/base/pose_arkit_world``, which is
unbroken across re-engages, plus the ``arkit_*`` /debug columns) while the wheel odometry
still closes the base PD loop -- so a take measures the drift before anything depends on
it. ``control`` additionally makes the phone pose canonical for base PD,
``obs/base/pose``, live policy state, and offline ``world_t_cam`` composition, and holds
the base on a stale (``stale-arkit``) or relocalized
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
one process per camera, selected by ``--serial-number``; see vr_reader).

Timing metadata for offline camera-latency audits (``scripts/audit_episode_latency.py``):
each frame stores the publisher capture stamps ``obs/images/{head_frame_ns,
head_depth_frame_ns,left_wrist_frame_ns,right_wrist_frame_ns}`` (camera-publisher-host
clock) and ``obs/images/grab_wall_ns``
(workstation clock right after all grabs). ``meta/ntp`` preserves the robot
SoC-minus-workstation offset from ``Robot.query_ntp()``; ``meta/camera_ntp`` stores
camera-publisher-host-minus-workstation offsets from the ZED publishers' lightweight
clock services and is required for new recordings. Absolute camera staleness should use
``camera_ntp``. By default the recorder samples each wrist subscriber cache at the
100 Hz control cadence, keeps a short history of distinct frames, and pairs the
capture-nearest wrist frame to each 10 Hz head frame after clock correction. A row is
never written when either head-to-wrist skew exceeds ``--record-max-camera-skew-ms``.

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
from omniteleop.follower.whole_body_ik import (
    DEFAULT_CONFIG_PATH,
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
    vr_to_ee_targets,
    vr_to_head_target,
)
from omniteleop.wbc_teleop import VRJointSubscriber, VRTeleopConfig, bind_vr_teleop_args

# Shared VR-follower control-loop tunables, loaded from the vr_teleop: block of
# follower/wbik.yaml so this real-robot follower and the SAPIEN sim follower
# (scripts/wbc_vr_record.py) shape the head/base command stream identically -- a take
# recorded in sim must replay the same way here. See omniteleop.wbc_teleop. The
# bring-up-specific knobs (homing, joint-step clamp, base max-speed, ...) stay local below.
_VR_TELEOP = VRTeleopConfig.from_yaml()
DEFAULT_HEAD_LPF_TAU = _VR_TELEOP.head_lpf_tau
DEFAULT_SPEED = _VR_TELEOP.replay_speed             # --replay time scale (<1 = slower)
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
DEFAULT_SAVE_DIR = "/home/yixuan/Dexmate/data/raw_data"
# Saving is normally sub-second for the streaming recorder, but filesystem or writer
# failures must not hold robot teardown forever. This is deliberately a generous
# terminal-path deadline, not a performance target.
_RECORDER_SAVE_TIMEOUT_S = 120.0
# --record stale-frame grace: when a record tick finds no strictly-newer frame from every
# camera, retry at the IK rate for up to this long before aborting the take. The publishers
# hold at least 15 fps through these stalls (verified against their pub-fps logs), so a
# non-newer frame means the DELIVERY path hiccupped: a WiFi burst, Zenoh backlog, or this
# process starving its Zenoh decode threads under solver load (worst while the torso moves).
# A frame is recorded only once BOTH publisher timestamps advance, so the no-duplicate
# guarantee is unchanged -- the grace converts a short delivery stall into one late frame
# instead of a dead take. 0 restores the old abort-on-first-stale-tick behavior.
DEFAULT_RECORD_STALE_GRACE = 0.20
# An episode boundary clears the alignment histories. Do not commit frame zero from a
# single lucky cache snapshot: first observe multiple advancing frames from all three
# physical cameras and at least one coherent head/wrist set. This is a startup-only
# deadline; once armed, the stricter 200 ms mid-take stale/alignment grace is unchanged.
DEFAULT_RECORD_STARTUP_TIMEOUT = 3.0
_RECORD_STARTUP_MIN_HEAD_FRAMES = 4
_RECORD_STARTUP_MIN_WRIST_FRAMES = 3
# Maximum absolute head-to-each-wrist capture skew in a saved row. The recorder maps
# each publisher stamp into the local clock using meta/camera_ntp before comparing it,
# so this contract remains meaningful when the three publishers run on different hosts.
# With 15 Hz wrists, capture-nearest sampling has a theoretical ~33 ms half-period
# floor; 40 ms leaves a small clock-calibration/scheduling margin while still excluding
# the 70--90 ms same-row skew measured in episode_2.
DEFAULT_RECORD_MAX_CAMERA_SKEW_MS = 40.0
# Maximum capture-to-selection age after mapping each publisher timestamp into the
# recorder's local wall clock. A strictly advancing stream can still be several frames
# behind because of encode/network/decode backlog. The deployed 30 Hz head + 15 Hz
# wrists measured a 133 ms worst selected age at 10 Hz, so 150 ms keeps useful margin
# without admitting the ~180 ms selected ages of the old 15 Hz head profile.
DEFAULT_RECORD_MAX_CAMERA_AGE_MS = 150.0
_CAMERA_FUTURE_TOLERANCE_NS = 10_000_000
# The Hand-E pass-through is a request/response Modbus channel, not a 100 Hz command
# transport. Sending an FC16 write on every IK tick starves the FC03 achieved-position
# requests used by obs/gripper (episodes 5/6 accumulated thousands of write ACKs but no
# second status response). Changed commands are dispatched at <=20 Hz, unchanged commands
# get a low-rate keepalive, and an outstanding FC03 read receives a short write-free
# response window. The physical gripper is much slower than 20 Hz; this bounds added input
# latency to 50 ms while making the 10 Hz observation contract sustainable.
_GRIPPER_COMMAND_MAX_RATE_HZ = 20.0
_GRIPPER_COMMAND_MIN_PERIOD_S = 1.0 / _GRIPPER_COMMAND_MAX_RATE_HZ
_GRIPPER_COMMAND_KEEPALIVE_S = 1.0
# Episode 7 measured healthy FC03 replies at 5.5 ms median / 7.6 ms p95, with one
# 41.9 ms outlier. Give a read the full 75 ms transaction window before allowing an
# FC16 write, then expire it before the next 10 Hz status slot. The old 25 ms guard +
# 500 ms timeout let a keepalive/changed FC16 collide with a read and then blocked all
# replacement reads for half a second.
_GRIPPER_STATUS_REQUEST_TIMEOUT_S = 0.075
_GRIPPER_STATUS_WRITE_GUARD_S = _GRIPPER_STATUS_REQUEST_TIMEOUT_S
# Also keep FC16 out of the short interval immediately *before* the next scheduled
# FC03. Without this half of the time-division contract, a 100 Hz tick could still send
# a changed command 10 ms before the following read even though reads have same-tick
# priority. Twenty-five milliseconds is >3x episode 7's p95 healthy transaction time.
_GRIPPER_STATUS_PRE_READ_GUARD_S = 0.025
_GRIPPER_STATUS_MAX_AGE_PERIODS = 2.5
# Select against a slightly tighter limit than the persisted commit-time contract.
# Episode 8 measured sub-millisecond state assembly, so 0.1 period (10 ms at 10 Hz)
# comfortably covers selection -> final timestamp without weakening the offline gate.
_GRIPPER_STATUS_COMMIT_MARGIN_PERIODS = 0.1
# Enough history for the 200 ms stale grace plus scheduling/transport jitter. Head
# history is essential when the low-latency head publisher runs at 30 Hz while the
# wrists run at 15 Hz: always anchoring on the newest head frame can run ahead of the
# later-arriving wrist frames and starve an otherwise feasible 10 Hz coherent set.
_HEAD_ALIGNMENT_BUFFER_FRAMES = 16
_WRIST_ALIGNMENT_BUFFER_FRAMES = 16
# Wrist cameras (recorded under obs/images/{left,right}_wrist_rgb when --record). Like the
# head camera they are consumed through the dexcontrol Robot API, but as multi-stream
# ZED-Ms published by standalone ZED-SDK processes on sensors/<sensor_id>/{left_rgb,
# right_rgb} (see leader/vr_reader.py + tests/test_wrist_zedm_depth.py). RGB-only.
# "left"/"right" name the ARM the camera is mounted on (matching
# src/omniteleop/record/mdp_recorder.py), NOT the stereo eye. The established
# ``<arm>_wrist_rgb`` key remains the stereo LEFT eye for policy compatibility;
# ``<arm>_wrist_right_rgb`` is added only with --wrist-right-rgb.
_WRIST_SENSOR_IDS: dict[str, str] = {
    "left": "left_wrist_zedm",
    "right": "right_wrist_zedm",
}
_WRIST_OBS_KEY = "left_rgb"  # established policy/recording view: stereo left eye
_HEAD_SENSOR_ID = "head_camera"


class _HeadAlignmentSample:
    """One internally coherent head-cache sample with exact receive stamps."""

    __slots__ = (
        "left_rgb",
        "depth_u16",
        "left_capture_ns",
        "depth_capture_ns",
        "right_rgb",
        "right_capture_ns",
        "left_receive_ns",
        "depth_receive_ns",
        "right_receive_ns",
    )

    def __init__(
        self,
        *,
        left_rgb: np.ndarray,
        depth_u16: Optional[np.ndarray],
        left_capture_ns: int,
        depth_capture_ns: Optional[int],
        right_rgb: Optional[np.ndarray],
        right_capture_ns: Optional[int],
        left_receive_ns: int,
        depth_receive_ns: Optional[int],
        right_receive_ns: Optional[int],
    ) -> None:
        self.left_rgb = left_rgb
        self.depth_u16 = depth_u16
        self.left_capture_ns = left_capture_ns
        self.depth_capture_ns = depth_capture_ns
        self.right_rgb = right_rgb
        self.right_capture_ns = right_capture_ns
        self.left_receive_ns = left_receive_ns
        self.depth_receive_ns = depth_receive_ns
        self.right_receive_ns = right_receive_ns

    def images_tuple(self) -> tuple:
        """Legacy ``_grab_head_images`` six-field view used by ``record_tick``."""
        return (
            self.left_rgb,
            self.depth_u16,
            self.left_capture_ns,
            self.depth_capture_ns,
            self.right_rgb,
            self.right_capture_ns,
        )
_CAMERA_CLOCK_SAMPLE_COUNT = 15
_CAMERA_CLOCK_MIN_SAMPLES = 8
_CAMERA_CLOCK_TIMEOUT_S = 0.5
_CAMERA_CLOCK_MAX_CONSECUTIVE_FAILURES = 3
_RAW_EPISODE_SCHEMA = "omniteleop_wbc_mobile_raw/v5"


def _json_bytes(value) -> np.ndarray:
    """Stable UTF-8 JSON scalar accepted by both episode recorder backends."""
    import json  # noqa: PLC0415 -- provenance-only path
    from pathlib import Path  # noqa: PLC0415

    def default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (set, tuple)):
            return list(obj)
        raise TypeError(f"cannot JSON-encode {type(obj).__name__}")

    encoded = json.dumps(
        value,
        default=default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return np.asarray(encoded.encode("utf-8"))


def _sha256_file(path) -> str:
    """SHA-256 a small source/config file without depending on git cleanliness."""
    import hashlib  # noqa: PLC0415 -- provenance-only path

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
# s: let the arms' requested position mode reach the driver before the startup status
# table is queried, so it reports the settled state instead of a mid-enable snapshot.
_STATUS_SETTLE = 1.0


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
                "twist": self._twist.copy(),
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


def _wait_for_recorder_save(
    recorder,
    *,
    path: str,
    timeout_s: float = _RECORDER_SAVE_TIMEOUT_S,
) -> None:
    """Wait for async finalization without allowing teardown to spin forever."""
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while recorder.saving:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            message = (
                "[wbc_vr_robot] recorder did not finish within "
                f"{float(timeout_s):g}s for {path}; refusing to wait forever. "
                "Treat any .partial file as quarantined and inspect the save target."
            )
            invalidate = getattr(recorder, "invalidate_save", None)
            if callable(invalidate):
                invalidate(message)
            # The writer may have completed between the loop condition and the
            # invalidation lock. In that case there is no timeout left to report.
            if not recorder.saving:
                return
            raise TimeoutError(message)
        time.sleep(min(0.05, remaining))


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
        self._arkit: Optional[ARKitBaseTracker] = None
        self._world_frame_epoch = 0
        self._episode: Optional[EpisodeRecorder | StreamingEpisodeRecorder] = None
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
        # Wall-clock bracket around the most recent hardware command dispatch. A
        # scalar frame timestamp cannot say whether an action preceded or followed
        # the camera/state sample; record_tick persists this bracket per saved row.
        self._last_action_dispatch_start_wall_ns = -1
        self._last_action_dispatch_end_wall_ns = -1
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
        self._last_gripper_wire_message: dict[str, Optional[bytes]] = {
            "left": None,
            "right": None,
        }
        self._last_gripper_wire_send_monotonic = {
            "left": float("-inf"),
            "right": float("-inf"),
        }
        # FC03 reads and FC16 commands share one Modbus pass-through per arm. Status
        # reads are scheduled at the record cadence from the 100 Hz actuation loop and
        # always serviced before a possible command write on that tick.
        self._next_grip_status_request_monotonic = {
            "left": 0.0,
            "right": 0.0,
        }
        self._last_obs_grip_left = float("nan")
        self._last_obs_grip_right = float("nan")
        self._grip_monitors: dict = {}
        # A cached value alone is not evidence that gripper feedback is current. Keep
        # the complete latest FC03 event and a monotonically increasing event count per
        # arm. A saved row may zero-order-hold that event only inside the explicit
        # timestamp-age bound; this preserves the 10 Hz image/action grid across one
        # dropped Modbus reply without recreating episode_2's unbounded 481-row repeat.
        self._last_grip_status: dict[str, Optional[dict]] = {
            "left": None,
            "right": None,
        }
        self._grip_status_event_count = {"left": 0, "right": 0}
        self._recorded_grip_status_event_count = {"left": 0, "right": 0}
        self._grip_status_stale_since: Optional[float] = None
        self._grip_episode_start_wall_ns = -1
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
        # Dexcomm stamps each decoded Zenoh message on local receipt. _unwrap_frame
        # updates this cache without changing the long-standing grab return signatures;
        # record_tick persists it to separate network/decode latency from cache age.
        self._last_camera_receive_ns: dict[str, int] = {}
        # Head-anchored multi-camera alignment. All three subscriber caches are sampled
        # on every 100 Hz record_tick call (before the 10 Hz record throttle), but only
        # one copy of each distinct physical frame is retained. A due record row tries
        # newest-to-oldest unused head frames and keeps the freshest head anchor for
        # which both capture-nearest wrist frames satisfy the corrected skew limit.
        #
        # Tuple layout in each deque:
        #   (left_rgb, left_capture_ns, left_receive_ns,
        #    right_rgb_or_None, right_capture_ns_or_None, right_receive_ns_or_None)
        self._record_max_camera_skew_ns = 0
        self._record_max_camera_age_ns = 0
        self._camera_clock_offsets_ns: dict[str, int] = {}
        self._head_alignment_buffer: deque[_HeadAlignmentSample] = deque(
            maxlen=_HEAD_ALIGNMENT_BUFFER_FRAMES
        )
        self._wrist_alignment_buffers: dict[str, deque] = {
            arm: deque(maxlen=_WRIST_ALIGNMENT_BUFFER_FRAMES)
            for arm in _WRIST_SENSOR_IDS
        }
        self._wrist_alignment_lock = threading.Lock()
        # Recorder main loop and policy inference worker may both request a cache poll.
        # Serialize the two-arm read so an older call cannot finish after a newer call
        # and look like a publisher clock regression.
        self._wrist_alignment_poll_lock = threading.Lock()
        self._last_buffered_wrist_ns: dict[str, int] = {
            arm: -1 for arm in _WRIST_SENSOR_IDS
        }
        self._last_buffered_head_ns = -1
        self._camera_alignment_stale_since: Optional[float] = None
        self._camera_age_stale_since: Optional[float] = None
        self._record_camera_warmup_complete = True
        self._record_camera_warmup_since: Optional[float] = None
        # Publisher /info responses are queried for both stereo calibration and
        # reproducibility metadata. Cache one authoritative response per physical
        # camera so startup does not race two queries against a publisher restart.
        self._camera_info_responses: dict[str, dict] = {}
        # Stale-grace state: loop time the current stale streak began (None while fresh),
        # plus a short per-recorded-frame arrival-age history (local wall clock minus the
        # publisher capture stamp; spans the robot<->workstation clock offset, so read the
        # head-vs-wrist split and the trend, not absolutes). Dumped on a stale abort to show
        # WHICH stream stalled and whether its delivery latency was climbing beforehand.
        # Entry = (loop_t, head_age_ms, left_wrist_age_ms, right_wrist_age_ms).
        self._stale_since: Optional[float] = None
        self._frame_age_log: deque[tuple[float, float, float, float]] = deque(maxlen=64)
        # --head-right-rgb eye-pairing state (see record_tick): retry briefly when the
        # independently delivered right-eye message lags the left, then report how many
        # recorded frames actually carry matching capture timestamps.
        self._right_unpaired_since: Optional[float] = None
        self._right_unpaired_streak = 0
        self._rec_right_unpaired = 0
        self._right_pair_retry = True
        self._wrist_right_unpaired_since: dict[str, Optional[float]] = {
            arm: None for arm in _WRIST_SENSOR_IDS
        }
        self._wrist_right_unpaired_streak: dict[str, int] = {
            arm: 0 for arm in _WRIST_SENSOR_IDS
        }
        self._rec_wrist_right_unpaired: dict[str, int] = {
            arm: 0 for arm in _WRIST_SENSOR_IDS
        }
        self._wrist_right_pair_retry: dict[str, bool] = {
            arm: True for arm in _WRIST_SENSOR_IDS
        }

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
            # Settle first: Robot.__init__ ends with _set_default_state(), which only
            # REQUESTS position mode on both arms -- the driver's operational status
            # flips a beat later. Querying immediately prints a DISABLED arm that is
            # already on its way to ENABLED (seen on right_arm, which is requested last).
            time.sleep(_STATUS_SETTLE)
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
            if args.record:
                self._init_gripper_monitors()
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
        return (
            getattr(self, "_arkit", None) is not None
            and getattr(getattr(self, "args", None), "arkit_base", "off") == "control"
        )

    def configured_base_pose_source(self) -> str:
        """Pose source promised by the requested mode, even before sensors start.

        Recording static metadata is assembled before ``_start_arkit_tracker()`` during
        hardware initialization. It must describe the configured contract, not infer it
        from whether the tracker object happens to exist at that earlier instant.
        ``start_recording_episode`` later checks this promise against the live source.
        """
        mode = str(getattr(getattr(self, "args", None), "arkit_base", "off"))
        return "arkit" if mode == "control" else "wheel_odometry"

    def base_pose_source(self) -> str:
        """Canonical measured-pose source shared by control, recording, and policy state."""
        return "arkit" if self._arkit_drives_base() else "wheel_odometry"

    def policy_base_pose_snapshot(self) -> dict:
        """Snapshot the base pose in the same frame/source that closes the PD loop.

        Wheel odometry is always retained because it supplies achieved body twist and
        drift diagnostics. Under ``--arkit-base control`` the canonical policy pose is
        instead the engage-zeroed ARKit pose; using odometry there would train/deploy a
        world anchor different from the one controlling the physical base.
        """
        if self._odom is None:
            raise RuntimeError("base pose snapshot requires wheel odometry")
        odom = self._odom.snapshot()
        arkit = self._arkit.snapshot() if self._arkit is not None else None
        selected = arkit if self._arkit_drives_base() else odom
        if selected is None:
            raise RuntimeError("ARKit is the base pose source but has no snapshot")
        pose = np.asarray(selected["pose"], dtype=np.float64)
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            raise RuntimeError(
                f"{self.base_pose_source()} base pose {pose!r} is not finite (3,)"
            )
        return {
            "source": self.base_pose_source(),
            "pose": pose,
            "odom": odom,
            "arkit": arkit,
        }

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

    def start_recording_episode(self) -> bool:
        """Start one episode after resetting every episode-local freshness latch.

        Teleop and policy rollout deliberately share this boundary.  In particular,
        camera receive stamps, stereo-pair retry state, and consumed FC03 event counts
        must not leak from the preceding rollout into frame zero of the next one.
        Returns ``True`` only when a recorder was actually started.
        """
        if self._episode is None or self._episode.recording:
            return False
        configured_source = self.configured_base_pose_source()
        live_source = self.base_pose_source()
        if live_source != configured_source:
            raise RuntimeError(
                "[wbc_vr_robot] refusing to start recording: configured canonical "
                f"base source is {configured_source!r}, but the live controller would "
                f"record {live_source!r}"
            )
        self._episode.start()
        self._grip_episode_start_wall_ns = time.time_ns()
        self._last_rec_head_ns = -1
        self._last_rec_wrist_ns = {arm: -1 for arm in _WRIST_SENSOR_IDS}
        self._last_camera_receive_ns = {}
        alignment_lock = getattr(self, "_wrist_alignment_lock", None)
        if alignment_lock is None:
            self._wrist_alignment_lock = threading.Lock()
            alignment_lock = self._wrist_alignment_lock
        poll_lock = getattr(self, "_wrist_alignment_poll_lock", None)
        if poll_lock is None:
            self._wrist_alignment_poll_lock = threading.Lock()
            poll_lock = self._wrist_alignment_poll_lock
        # Lock in the same poll->history order as _poll_camera_alignment_buffers(). A
        # concurrent policy-camera grab that began before this episode must finish and
        # append before the reset; it can never append a pre-boundary sample afterward.
        with poll_lock:
            with alignment_lock:
                self._head_alignment_buffer = deque(
                    maxlen=_HEAD_ALIGNMENT_BUFFER_FRAMES
                )
                self._wrist_alignment_buffers = {
                    arm: deque(maxlen=_WRIST_ALIGNMENT_BUFFER_FRAMES)
                    for arm in _WRIST_SENSOR_IDS
                }
                self._last_buffered_head_ns = -1
                self._last_buffered_wrist_ns = {
                    arm: -1 for arm in _WRIST_SENSOR_IDS
                }
        self._camera_alignment_stale_since = None
        self._camera_age_stale_since = None
        self._record_camera_warmup_complete = False
        self._record_camera_warmup_since = None
        self._next_record_t = 0.0
        self._stale_since = None
        self._frame_age_log.clear()
        self._right_unpaired_since = None
        self._right_unpaired_streak = 0
        self._rec_right_unpaired = 0
        self._right_pair_retry = True
        self._wrist_right_unpaired_since = {arm: None for arm in _WRIST_SENSOR_IDS}
        self._wrist_right_unpaired_streak = {arm: 0 for arm in _WRIST_SENSOR_IDS}
        self._rec_wrist_right_unpaired = {arm: 0 for arm in _WRIST_SENSOR_IDS}
        self._wrist_right_pair_retry = {arm: True for arm in _WRIST_SENSOR_IDS}
        # Drain status replies queued before engage. Episodes 5/6 proved that merely
        # snapshotting the counters is not enough: their first rows consumed request-id
        # 0 replies that had sat queued for 107 s / 5 s. Folding those events into the
        # cache *before* the counter baseline makes them ineligible for frame zero.
        if getattr(self, "_grip_monitors", {}):
            self._poll_gripper_status_step(allow_request=False)
        # Status replies consumed before engage (including policy-rollout warm-up) are
        # not fresh observations for frame zero. The first record tick must see a newer
        # event from both arms whose callback time is after the episode boundary.
        self._recorded_grip_status_event_count = dict(self._grip_status_event_count)
        # Force the next actuation tick to issue a post-boundary FC03 before it is
        # allowed to send an FC16 command. Do not issue it here: this method runs after
        # actuation and could otherwise place FC03 immediately behind that tick's FC16.
        self._next_grip_status_request_monotonic = {
            side: 0.0 for side in getattr(self, "_grip_monitors", {})
        }
        self._grip_status_stale_since = None
        return True

    def start_recording_if_teleop(self, vr) -> None:
        """Start episode recording once the leader leaves any pre-record align stage."""
        if (
            vr is None
            or bool(getattr(vr, "estop", False))
            or str(getattr(vr, "calib_stage", "")) != "teleop"
        ):
            return
        self.start_recording_episode()

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
        _wait_for_recorder_save(self._episode, path=path)
        save_error = getattr(self._episode, "last_save_error", None)
        if save_error is not None:
            raise RuntimeError(
                f"[wbc_vr_robot] episode save failed for {path}: {save_error}"
            ) from save_error
        print(f"[wbc_vr_robot] episode saved -> {path}")
        if getattr(getattr(self, "args", None), "head_right_rgb", False) and n:
            paired = n - self._rec_right_unpaired
            print(f"[wbc_vr_robot] head stereo pairing: {paired}/{n} frames "
                  f"({100.0 * paired / n:.1f}%) carry head_right_frame_ns == "
                  f"head_frame_ns. Only those yield FoundationStereo head depth.")
        if getattr(getattr(self, "args", None), "wrist_right_rgb", False) and n:
            for arm in _WRIST_SENSOR_IDS:
                paired = n - self._rec_wrist_right_unpaired[arm]
                print(
                    f"[wbc_vr_robot] {arm} wrist stereo pairing: {paired}/{n} frames "
                    f"({100.0 * paired / n:.1f}%) carry {arm}_wrist_right_frame_ns == "
                    f"{arm}_wrist_frame_ns. Only those yield FoundationStereo depth."
                )
        traj = getattr(self, "_traj", None)
        if traj is not None and saved_id is not None:
            traj.flush(saved_id)

    def abort_recording_episode(self, reason: str) -> None:
        """Quarantine an exception-truncated episode instead of blessing it complete.

        Operator-requested boundaries (e-stop, Ctrl-C) still call
        :meth:`stop_recording_episode` and produce a normal complete take. Runtime data
        contract/safety exceptions call this path: already-captured frames remain
        inspectable under ``.hdf5.partial`` with ``complete=false`` and ``abort_reason``,
        but no porter glob can mistake them for training input.
        """
        if self._episode is None or not self._episode.recording:
            return
        n = self._episode.num_frames()
        abort = getattr(self._episode, "abort", None)
        if not callable(abort):
            self._episode.discard()
            print(
                "\n[wbc_vr_robot] recording aborted: recorder has no quarantine "
                f"support; discarded {n} frame(s). reason={reason}"
            )
            return
        path = abort(str(reason))
        if path is None:
            print(
                "\n[wbc_vr_robot] recording aborted before frame zero; no file "
                f"created. reason={reason}"
            )
            return
        print(
            f"\n[wbc_vr_robot] quarantining {n} aborted frame(s) -> {path} ..."
        )
        _wait_for_recorder_save(self._episode, path=path)
        save_error = getattr(self._episode, "last_save_error", None)
        if save_error is not None:
            raise RuntimeError(
                f"[wbc_vr_robot] failed to quarantine aborted take {path}: "
                f"{save_error}"
            ) from save_error
        print(
            f"[wbc_vr_robot] aborted take quarantined -> {path} "
            "(complete=false; excluded from training)"
        )

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

    def _command_grippers(
        self,
        left_gripper: float,
        right_gripper: float,
        *,
        now_monotonic: Optional[float] = None,
    ) -> None:
        """Dispatch bounded FC16 writes without colliding with FC03 status reads.

        ``action/gripper`` is the latest command that actually reached the shared
        pass-through, not an unsent 100 Hz desired value. Changed byte-level Hand-E
        commands are sent at no more than 20 Hz and unchanged commands once per second.
        When the status monitor has issued FC03, writes pause for its bounded 75 ms
        transaction window so its reply cannot be buried by the next FC16 ACK. Healthy
        replies arrive in roughly 5--8 ms, so normal changed-command latency is one
        100 Hz control tick; a missing reply cannot freeze manual control indefinitely.
        """
        now_mono = time.monotonic() if now_monotonic is None else float(now_monotonic)
        commands = {
            "left": float(np.clip(left_gripper, 0.0, 1.0)),
            "right": float(np.clip(right_gripper, 0.0, 1.0)),
        }
        arms = {"left": self.robot.left_arm, "right": self.robot.right_arm}

        # Keep object.__new__-constructed test drivers and older rollout wrappers
        # compatible while all normal HardwareDriver instances initialize these fields
        # in __init__.
        if not hasattr(self, "_last_gripper_wire_message"):
            self._last_gripper_wire_message = {"left": None, "right": None}
        if not hasattr(self, "_last_gripper_wire_send_monotonic"):
            self._last_gripper_wire_send_monotonic = {
                "left": float("-inf"),
                "right": float("-inf"),
            }

        monitors = getattr(self, "_grip_monitors", {})
        for side in ("left", "right"):
            monitor = monitors.get(side)
            if monitor is not None:
                next_status = float(
                    getattr(self, "_next_grip_status_request_monotonic", {}).get(
                        side, 0.0
                    )
                )
                if (
                    next_status > 0.0
                    and now_mono
                    >= next_status - _GRIPPER_STATUS_PRE_READ_GUARD_S
                ):
                    continue
                pending_age_fn = getattr(monitor, "pending_request_age_s", None)
                if not callable(pending_age_fn):
                    raise RuntimeError(
                        "[wbc_vr_robot] gripper status monitor must expose "
                        "pending_request_age_s(); cannot safely schedule FC16 writes "
                        "from a depth-only pending counter"
                    )
                pending_age = pending_age_fn()
                if (
                    pending_age is not None
                    and pending_age < _GRIPPER_STATUS_WRITE_GUARD_S
                ):
                    continue

            message = self._build_hande(commands[side])
            last_message = self._last_gripper_wire_message[side]
            elapsed = now_mono - float(
                self._last_gripper_wire_send_monotonic[side]
            )
            changed_due = (
                message != last_message
                and elapsed >= _GRIPPER_COMMAND_MIN_PERIOD_S
            )
            keepalive_due = elapsed >= _GRIPPER_COMMAND_KEEPALIVE_S
            if not (changed_due or keepalive_due):
                continue

            self._send_ee_pass_through(arms[side], message)
            self._last_gripper_wire_message[side] = message
            self._last_gripper_wire_send_monotonic[side] = now_mono
            setattr(self, f"_last_gripper_cmd_{side}", commands[side])

    def actuate(self, result, left_gripper, right_gripper, enable, hold, dt) -> None:
        """Send the IK result to the enabled actuators (clamped); zero/freeze on hold."""
        from omniteleop.wbc_robot_util import clamp_joint_step  # noqa: PLC0415

        self._last_action_dispatch_start_wall_ns = time.time_ns()
        self._last_action_dispatch_end_wall_ns = -1
        # Status traffic gets first use of each arm's shared Modbus pass-through. Doing
        # this before _command_grippers is what prevents a changed trigger (or FC16
        # keepalive) from being emitted immediately before the due FC03 read.
        if getattr(self, "_grip_monitors", {}):
            self._poll_gripper_status_step(now_monotonic=time.monotonic())
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
            # Grippers are always active. Their serial pass-through is scheduled below
            # the 100 Hz IK rate so FC03 achieved-state reads get a response window.
            # The cached action values update only when the corresponding FC16 command
            # really reaches the wire.
            self._command_grippers(left_gripper, right_gripper)
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
        self._last_action_dispatch_end_wall_ns = time.time_ns()

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
            self._last_grip_status[side] = dict(warm)
        self._grip_monitors = {
            "left": RobotiqStatusMonitor(self.robot.left_arm, side="left", function_code=0x03),
            "right": RobotiqStatusMonitor(self.robot.right_arm, side="right", function_code=0x03),
        }
        period = self._gripper_status_poll_period_s()
        for side, monitor in self._grip_monitors.items():
            event = monitor.send_status_request()
            sent_mono = float(event["send_monotonic_ns"]) / 1e9
            self._next_grip_status_request_monotonic[side] = sent_mono + period

    def _gripper_status_poll_period_s(self) -> float:
        """Status cadence, matched to the configured recording cadence."""
        period = float(getattr(self, "_record_period", 0.0))
        return period if np.isfinite(period) and period > 0.0 else 0.1

    def _poll_gripper_status_step(
        self,
        *,
        allow_request: bool = True,
        now_monotonic: Optional[float] = None,
    ) -> None:
        """Drain FC03 replies and, when due, issue one prioritized read per arm.

        Called from the 100 Hz actuation loop *before* FC16 command scheduling, but new
        requests are rate-limited to ``--record-rate``. ``record_tick`` uses
        ``allow_request=False`` so it may consume a callback that arrived during
        actuation/camera work without ever placing FC03 immediately behind an FC16.
        """
        now_mono = time.monotonic() if now_monotonic is None else float(now_monotonic)
        period = self._gripper_status_poll_period_s()
        if not hasattr(self, "_next_grip_status_request_monotonic"):
            self._next_grip_status_request_monotonic = {
                side: 0.0 for side in self._grip_monitors
            }
        for side, monitor in self._grip_monitors.items():
            events = monitor.drain_status_events()
            if events:
                latest = dict(events[-1])
                setattr(self, f"_last_obs_grip_{side}", float(latest["actual"]))
                self._last_grip_status[side] = latest
                self._grip_status_event_count[side] += len(events)
            monitor.expire_timeouts(_GRIPPER_STATUS_REQUEST_TIMEOUT_S)
            # Keep at most one read outstanding and one scheduled request per record
            # period. A timed-out slot advances normally rather than causing the old
            # 500 ms hole; late retries are re-anchored instead of sent in a burst.
            next_due = float(self._next_grip_status_request_monotonic.get(side, 0.0))
            if (
                allow_request
                and now_mono >= next_due
                and monitor.stats()["pending_status_requests"] == 0
            ):
                event = monitor.send_status_request()
                sent_mono = now_mono
                if isinstance(event, dict) and "send_monotonic_ns" in event:
                    sent_mono = float(event["send_monotonic_ns"]) / 1e9
                candidate = next_due + period if next_due > 0.0 else sent_mono + period
                self._next_grip_status_request_monotonic[side] = (
                    sent_mono + period if candidate <= now_mono else candidate
                )

    def _fresh_gripper_statuses(self) -> Optional[dict[str, dict]]:
        """Return bounded-age FC03 samples for one row, marking any zero-order hold.

        Requiring a distinct packet for every 10 Hz image row made one lost Modbus reply
        create a 500 ms timestamp hole in episode 7. The meaningful safety property is
        bounded sensor age, not packet uniqueness: reuse is selected with a 0.1-period
        commit margin inside the persisted 2.5-period limit and is written explicitly
        as ``sample_reused``. Both arms still pass atomically, and frame zero still
        requires callbacks after engage.
        """
        statuses = {
            side: self._last_grip_status[side]
            for side in ("left", "right")
        }
        if any(status is None for status in statuses.values()):
            return None
        episode_start_ns = int(getattr(self, "_grip_episode_start_wall_ns", -1))
        if episode_start_ns > 0:
            now_wall_ns = time.time_ns()
            record_period = float(getattr(self, "_record_period", 0.1))
            max_age_ns = round(
                (
                    _GRIPPER_STATUS_MAX_AGE_PERIODS
                    - _GRIPPER_STATUS_COMMIT_MARGIN_PERIODS
                )
                * record_period
                * 1e9
            )
            for side in ("left", "right"):
                status = statuses[side]
                read_ns = int(status.get("read_wall_ns", -1))
                request_ns = int(status.get("request_send_wall_ns", -1))
                # Provenance must describe a real, post-boundary transaction. Runtime
                # and offline checks use the same request and sample-age limits.
                if read_ns < episode_start_ns or request_ns <= 0:
                    return None
                request_latency_ns = read_ns - request_ns
                if (
                    request_latency_ns < 0
                    or request_latency_ns
                    > round(_GRIPPER_STATUS_REQUEST_TIMEOUT_S * 1e9)
                ):
                    return None
                age_ns = now_wall_ns - read_ns
                if age_ns < 0 or age_ns > max_age_ns:
                    return None
        if any(
            self._grip_status_event_count[side]
            < self._recorded_grip_status_event_count[side]
            for side in ("left", "right")
        ):
            raise RuntimeError("gripper status event counter regressed")
        out = {}
        for side in ("left", "right"):
            status = dict(statuses[side])
            status["sample_reused"] = (
                self._grip_status_event_count[side]
                == self._recorded_grip_status_event_count[side]
            )
            out[side] = status
        self._recorded_grip_status_event_count = dict(self._grip_status_event_count)
        return out

    def _validate_gripper_status_commit_age(
        self,
        statuses: dict[str, dict],
        commit_wall_ns: int,
    ) -> None:
        """Enforce the porter's exact 2.5-period age bound at final row commit."""
        max_age_ns = round(
            _GRIPPER_STATUS_MAX_AGE_PERIODS
            * float(getattr(self, "_record_period", 0.1))
            * 1e9
        )
        for side, status in statuses.items():
            read_wall_ns = int(status.get("read_wall_ns", -1))
            age_ns = int(commit_wall_ns) - read_wall_ns
            if age_ns < 0 or age_ns > max_age_ns:
                raise RuntimeError(
                    "[wbc_vr_robot] --record: gripper status crossed the exact "
                    f"commit-age contract while assembling the row: {side}={age_ns / 1e6:.3f} "
                    f"ms, maximum={max_age_ns / 1e6:.3f} ms. Aborting instead of "
                    "writing a row the offline policy porter would reject."
                )

    @staticmethod
    def _recordable_gripper_status(status: dict) -> dict:
        """Convert a parsed monitor event into a fixed, HDF5-safe telemetry tree."""

        def _int(name: str, default: int = -1) -> np.int64:
            value = status.get(name, default)
            return np.int64(default if value is None else value)

        return {
            "actual": np.float32(status["actual"]),
            "cmd_echo": np.float32(status["cmd_echo"]),
            "gPO": _int("gPO"),
            "gPR": _int("gPR"),
            "gCU": _int("gCU"),
            "gACT": _int("gACT"),
            "gGTO": _int("gGTO"),
            "gSTA": _int("gSTA_bits"),
            "gOBJ": _int("gOBJ"),
            "gFLT": _int("gFLT"),
            "kFLT": _int("kFLT"),
            "status_request_id": _int("status_request_id"),
            "request_send_wall_ns": _int("request_send_wall_ns"),
            "response_timestamp_ns": _int("response_timestamp_ns"),
            "response_sequence": _int("response_sequence"),
            "read_wall_ns": _int("read_wall_ns"),
            "sample_reused": np.bool_(bool(status.get("sample_reused", False))),
        }

    def _await_record_cameras(self) -> None:
        """Require the --record cameras to be streaming before a take can start.

        head_camera + both wrist ZED-Ms are enabled in the --record Robot build. Each wrist
        needs its own external ZED-SDK publisher on ``sensors/<sensor_id>/*``. New
        recordings fail early if any required camera is unavailable.
        """
        wrist_ids = set(_WRIST_SENSOR_IDS.values())
        for cam in ("head_camera", *_WRIST_SENSOR_IDS.values()):
            if not self.robot.has_sensor(cam):
                raise SystemExit(
                    f"[wbc_vr_robot] --record enabled '{cam}' but it is not available on "
                    "this robot.")
            sensor = getattr(self.robot.sensors, cam)
            if hasattr(sensor, "wait_for_active") and not sensor.wait_for_active(timeout=5.0):
                extra = (f" -- needs a ZED-SDK publisher on sensors/{cam}/*"
                         if cam in wrist_ids else "")
                raise RuntimeError(
                    f"[wbc_vr_robot] --record requires {cam} active before recording; "
                    f"not active within 5s{extra}."
                )

    def _init_recording(self) -> None:
        """Set up the optional vr_reader-format episode recorder (``--record``).

        OFF unless ``--record``. When on, records (while engaged) the ``leader/vr_reader.py``
        main-HDF5 schema, EXCEPT:
          * ``action/joint`` adds ``torso`` (the WBC commands the torso here too);
          * the base action ``chassis_*`` is the shaped twist actually sent to the
            chassis (the WBC drives the base; the leader thumbstick is unused here);
          * with ``--enable base``, ``obs/joint`` adds the MEASURED base twist ``chassis_*``
            (wheel-odometry body twist, achieved-velocity counterpart to the action),
            ``obs/base/pose`` adds the canonical control ``(x, y, yaw)`` pose in the
            engage-origin world frame, and ``obs/base/pose_odom`` always keeps wheel
            odometry. ``meta/obs_base_pose_source`` declares whether canonical means
            wheel odometry or ARKit;
          * ``obs/images`` stores the ``intrinsic`` once (static, constant over a take) and
            does NOT store the per-frame ``extrinsic`` -- ``world_t_cam`` is recomputed offline
            from ``obs/base/pose`` + ``obs/joint`` (see ``scripts/vis_episode.py``).
        ``action/gripper`` is the leader trigger command, ``obs/gripper`` the Robotiq FC03
        achieved position, and ``obs/images`` carries
        head_left_rgb/head_depth/{left,right}_wrist_rgb plus the per-frame capture stamps.
        With ``--head-right-rgb`` and/or ``--wrist-right-rgb``, it also carries the
        corresponding rectified right-eye images and stamps; the wrist names' first
        left/right component identifies the ARM, while ``_right_rgb`` identifies the eye.
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
        self._record_startup_timeout_s = float(getattr(
            self.args,
            "record_startup_timeout",
            DEFAULT_RECORD_STARTUP_TIMEOUT,
        ))
        if (
            not np.isfinite(self._record_startup_timeout_s)
            or self._record_startup_timeout_s <= 0.0
        ):
            raise ValueError(
                "--record-startup-timeout must be finite and > 0, got "
                f"{self._record_startup_timeout_s}"
            )
        max_camera_skew_ms = float(getattr(
            self.args,
            "record_max_camera_skew_ms",
            DEFAULT_RECORD_MAX_CAMERA_SKEW_MS,
        ))
        if not np.isfinite(max_camera_skew_ms) or max_camera_skew_ms < 0.0:
            raise ValueError(
                "--record-max-camera-skew-ms must be finite and >= 0, got "
                f"{max_camera_skew_ms}"
            )
        self._record_max_camera_skew_ns = int(round(max_camera_skew_ms * 1e6))
        max_camera_age_ms = float(getattr(
            self.args,
            "record_max_camera_age_ms",
            DEFAULT_RECORD_MAX_CAMERA_AGE_MS,
        ))
        if not np.isfinite(max_camera_age_ms) or max_camera_age_ms <= 0.0:
            raise ValueError(
                "--record-max-camera-age-ms must be finite and > 0, got "
                f"{max_camera_age_ms}"
            )
        self._record_max_camera_age_ns = int(round(max_camera_age_ms * 1e6))
        if not self.args.record:
            return
        self._await_record_cameras()
        self._record_period = 1.0 / self.args.record_rate
        recorder_cls = (
            StreamingEpisodeRecorder if self.args.streaming_recorder else EpisodeRecorder
        )
        self._episode = recorder_cls(self.args.save_dir)
        head_stereo = (
            self._query_head_stereo_calibration() if self.args.head_right_rgb else None
        )
        wrist_stereo = (
            self._query_wrist_stereo_calibrations()
            if getattr(self.args, "wrist_right_rgb", False)
            else None
        )
        # When recording the stereo pair, the publisher's rectified left_K describes the
        # exact VIEW.LEFT pixels in this run and is therefore authoritative. ZED_K remains
        # the compatibility fallback for the legacy left+SDK-depth path.
        head_intrinsic = (
            head_stereo["left_K"] if head_stereo is not None else ZED_K.astype(np.float32)
        )
        # Camera intrinsic is constant over a take -> store ONCE (obs/images/intrinsic is
        # (3,3), not (N,3,3)). The per-frame extrinsic is NOT stored: it was base-relative
        # head FK (base-blind, wrong once the base drives) and is fully recomputable, so it
        # is recomputed offline as world_t_cam = world_t_base(obs/base/pose) . FK_cam(obs/
        # joint) -- see scripts/vis_episode.py -- which lands the cloud in the world frame.
        camera_ntp = self._query_camera_clock_calibrations()
        required_clock_labels = {"head", "left_wrist", "right_wrist"}
        missing_clock_labels = sorted(required_clock_labels - set(camera_ntp))
        if missing_clock_labels:
            raise RuntimeError(
                "[wbc_vr_robot] camera clock calibration is missing "
                f"{missing_clock_labels}; capture-nearest camera alignment requires all "
                "three publisher clocks."
            )
        try:
            self._camera_clock_offsets_ns = {
                label: int(np.asarray(camera_ntp[label]["offset_ns"]))
                for label in required_clock_labels
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "[wbc_vr_robot] malformed camera clock calibration: every camera needs "
                "an integer offset_ns"
            ) from exc

        static: dict = {"obs": {"images": {"intrinsic": head_intrinsic}}}
        static["meta"] = {
            "ntp": self._query_ntp_calibration(),
            "camera_ntp": camera_ntp,
            "camera_alignment": {
                "mode": np.asarray(
                    (
                        b"head_capture_nearest"
                        if self._record_max_camera_skew_ns > 0
                        else b"latest_arrived_compatibility"
                    )
                ),
                "anchor": np.asarray(b"head_left_rgb_capture"),
                "clock_domain": np.asarray(b"local_via_camera_ntp"),
                "max_abs_skew_ns": np.int64(self._record_max_camera_skew_ns),
                "max_camera_age_ns": np.int64(self._record_max_camera_age_ns),
                "head_buffer_frames": np.int64(_HEAD_ALIGNMENT_BUFFER_FRAMES),
                "wrist_buffer_frames": np.int64(_WRIST_ALIGNMENT_BUFFER_FRAMES),
                "startup_timeout_ns": np.int64(round(
                    self._record_startup_timeout_s * 1e9
                )),
                "startup_min_head_frames": np.int64(
                    _RECORD_STARTUP_MIN_HEAD_FRAMES
                ),
                "startup_min_wrist_frames": np.int64(
                    _RECORD_STARTUP_MIN_WRIST_FRAMES
                ),
            },
            "cameras": self._recording_camera_metadata(),
            "provenance": self._recording_provenance_metadata(),
        }
        if head_stereo is not None:
            static["meta"]["head_stereo"] = head_stereo
        if wrist_stereo is not None:
            static["meta"]["wrist_stereo"] = wrist_stereo
        static["meta"].update(self._recording_control_metadata())
        self._episode.set_static(static)
        head_keys = (
            "head_left_rgb" if self.args.no_head_depth else "head_left_rgb+head_depth"
        )
        if self.args.head_right_rgb:
            head_keys += "+head_right_rgb"
        wrist_keys = "left_wrist_rgb+right_wrist_rgb"
        if getattr(self.args, "wrist_right_rgb", False):
            wrist_keys += "+{left,right}_wrist_right_rgb"
        print(f"[wbc_vr_robot] recording -> {self.args.save_dir} "
              f"(episode_{self._episode.episode_id}, {self.args.record_rate:g}Hz, "
              f"{'streaming' if self.args.streaming_recorder else 'buffered'}, "
              f"{head_keys}+{wrist_keys}, +torso action, "
              "+gripper obs/action, +base pose obs, +frame capture stamps, "
              f"camera alignment {'<=' + format(max_camera_skew_ms, 'g') + 'ms' if self._record_max_camera_skew_ns > 0 else 'OFF'})")

    def _recording_control_metadata(self) -> dict:
        """Static semantics needed to interpret actions, state, and clock fields."""
        if getattr(self.args, "policy_path", None):
            control_mode = "policy_rollout"
        elif getattr(self.args, "replay_episode", None):
            control_mode = "policy_episode_replay"
        elif getattr(self.args, "replay", None):
            control_mode = "wbc_episode_replay"
        else:
            control_mode = "wbc_vr_live"
        arkit_mode = str(getattr(self.args, "arkit_base", "off"))
        configured_base_source = self.configured_base_pose_source()
        return {
            "schema": np.asarray(_RAW_EPISODE_SCHEMA.encode()),
            "control_mode": np.asarray(control_mode.encode()),
            "command_source_clock": np.asarray(b"producer_wall_clock"),
            "local_timing_clock": np.asarray(b"time.time_ns"),
            "episode_timestamp_semantics": np.asarray(b"record_commit_wall_ns"),
            "action_target_frame": np.asarray(b"engage_origin_world"),
            "obs_base_pose_source": np.asarray(configured_base_source.encode()),
            "base_control_pose_source": np.asarray(configured_base_source.encode()),
            "arkit_mode": np.asarray(arkit_mode.encode()),
            "gripper_action_semantics": np.asarray(b"latest_fc16_command_sent"),
            "gripper_command_max_rate_hz": np.float64(
                _GRIPPER_COMMAND_MAX_RATE_HZ
            ),
            "gripper_command_keepalive_s": np.float64(
                _GRIPPER_COMMAND_KEEPALIVE_S
            ),
            "gripper_status_write_guard_s": np.float64(
                _GRIPPER_STATUS_WRITE_GUARD_S
            ),
            "gripper_status_request_timeout_s": np.float64(
                _GRIPPER_STATUS_REQUEST_TIMEOUT_S
            ),
            "gripper_status_pre_read_guard_s": np.float64(
                _GRIPPER_STATUS_PRE_READ_GUARD_S
            ),
            "gripper_status_poll_rate_hz": np.float64(
                1.0 / self._gripper_status_poll_period_s()
            ),
            "gripper_status_max_age_periods": np.float64(
                _GRIPPER_STATUS_MAX_AGE_PERIODS
            ),
            "gripper_status_selection_max_age_periods": np.float64(
                _GRIPPER_STATUS_MAX_AGE_PERIODS
                - _GRIPPER_STATUS_COMMIT_MARGIN_PERIODS
            ),
            "gripper_status_commit_margin_periods": np.float64(
                _GRIPPER_STATUS_COMMIT_MARGIN_PERIODS
            ),
        }

    def _recording_camera_metadata(self) -> dict:
        """Serial, firmware, geometry, and launch configuration for all cameras.

        The publisher is the only authority for the pixels that actually arrived. Its
        ``/info`` response captures resize/crop/depth/tracking flags and the opened SDK
        device, avoiding reconstruction from a launch command that may be stale.
        """
        out: dict = {}
        for label, sensor_id in (
            ("head", _HEAD_SENSOR_ID),
            *((f"{arm}_wrist", sid) for arm, sid in _WRIST_SENSOR_IDS.items()),
        ):
            info = self._query_camera_info_response(label, sensor_id)
            actual = info.get("actual")
            if not isinstance(actual, dict):
                raise RuntimeError(
                    f"[wbc_vr_robot] {label} camera /info has no 'actual' geometry"
                )
            try:
                serial = int(info["serial_number"])
                width = int(actual["width"])
                height = int(actual["height"])
                fps = float(actual["fps"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"[wbc_vr_robot] malformed {label} camera identity/geometry: {info}"
                ) from exc
            if serial <= 0 or width <= 0 or height <= 0 or not np.isfinite(fps) or fps <= 0:
                raise RuntimeError(
                    f"[wbc_vr_robot] invalid {label} camera identity/geometry: "
                    f"serial={serial}, {width}x{height}@{fps}"
                )
            status = str(info.get("status", ""))
            if status.lower() != "running":
                raise RuntimeError(
                    f"[wbc_vr_robot] {label} camera publisher status is {status!r}, "
                    "expected 'running'"
                )
            out[label] = {
                "sensor_id": np.asarray(sensor_id.encode()),
                "camera_id": np.asarray(str(info.get("camera_id", serial)).encode()),
                "serial_number": np.int64(serial),
                "model": np.asarray(str(info.get("model", "unknown")).encode()),
                "firmware_version": np.asarray(
                    str(info.get("firmware_version", "unknown")).encode()
                ),
                "status": np.asarray(status.encode()),
                "actual": {
                    "width": np.int64(width),
                    "height": np.int64(height),
                    "fps": np.float64(fps),
                },
                "configured_json": _json_bytes(info.get("configured", {})),
                "streams_json": _json_bytes(info.get("streams", {})),
            }
        return out

    def _recording_provenance_metadata(self) -> dict:
        """Self-contained executable/config snapshot for exact take reproduction."""
        import importlib.metadata  # noqa: PLC0415 -- provenance-only path
        import os  # noqa: PLC0415
        import platform  # noqa: PLC0415
        import socket  # noqa: PLC0415
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        repo = Path(__file__).resolve().parents[1]

        def git(*git_args: str) -> str:
            try:
                result = subprocess.run(
                    ["git", "-C", str(repo), *git_args],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
            except (OSError, subprocess.SubprocessError):
                return ""
            return result.stdout.strip() if result.returncode == 0 else ""

        source_paths = {
            "wbc_vr_robot": Path(__file__).resolve(),
            "wbc_vr_leader": repo / "scripts/wbc_vr_leader.py",
            "wbc_policy_rollout": repo / "scripts/wbc_policy_rollout.py",
            "port_wbc_mobile_hdf5": repo / "scripts/port_wbc_mobile_hdf5.py",
            "wbc_teleop": repo / "src/omniteleop/wbc_teleop.py",
            "base_closed_loop": repo / "src/omniteleop/follower/base_closed_loop.py",
            "whole_body_ik": repo / "src/omniteleop/follower/whole_body_ik.py",
            "wbik_yaml": Path(DEFAULT_CONFIG_PATH).resolve(),
            "head_zed_publisher": repo / "tests/test_head_zedx_depth.py",
            "wrist_zed_publisher": repo / "tests/test_wrist_zedm_depth.py",
        }
        source_snapshot: dict = {}
        for label, path in source_paths.items():
            if not path.is_file():
                raise RuntimeError(
                    f"[wbc_vr_robot] provenance source file is missing: {path}"
                )
            content = path.read_bytes()
            source_snapshot[label] = {
                "path": np.asarray(
                    str(path.relative_to(repo) if path.is_relative_to(repo) else path).encode()
                ),
                "sha256": np.asarray(_sha256_file(path).encode()),
                # Exact dirty-worktree content makes the hash actionable even when the
                # git commit alone cannot reproduce this run.
                "content": np.asarray(content),
            }

        versions: dict[str, str] = {}
        for package in ("omniteleop", "dexcomm", "dexcontrol", "numpy", "h5py", "pink"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "not-installed-as-distribution"

        args_dict = dict(vars(self.args))
        cfg_dict = asdict(self.cfg)
        tracked_status = git("status", "--porcelain=v1", "--untracked-files=no")
        return {
            "schema": np.asarray(_RAW_EPISODE_SCHEMA.encode()),
            "captured_at_ns": np.int64(time.time_ns()),
            "git_commit": np.asarray((git("rev-parse", "HEAD") or "unknown").encode()),
            "git_tracked_dirty": np.bool_(bool(tracked_status)),
            "git_tracked_status": np.asarray(tracked_status.encode()),
            "argv_json": _json_bytes(sys.argv),
            "parsed_args_json": _json_bytes(args_dict),
            "wbc_config_json": _json_bytes(cfg_dict),
            "software_versions_json": _json_bytes(versions),
            "host": {
                "hostname": np.asarray(socket.gethostname().encode()),
                "platform": np.asarray(platform.platform().encode()),
                "python_version": np.asarray(platform.python_version().encode()),
                "python_executable": np.asarray(sys.executable.encode()),
            },
            "robot": {
                "robot_name": np.asarray(os.getenv("ROBOT_NAME", "").encode()),
                "namespace": np.asarray(str(getattr(self.args, "namespace", "")).encode()),
            },
            "source_snapshot": source_snapshot,
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

        The head clock service and BOTH wrist clock services are required for new
        recordings; they land under the keys head/left_wrist/right_wrist.
        """
        out: dict = {}
        for label, sensor_id in (
            ("head", _HEAD_SENSOR_ID),
            *((f"{arm}_wrist", sid) for arm, sid in _WRIST_SENSOR_IDS.items()),
        ):
            out[label] = self._query_camera_clock_calibration(label, sensor_id)
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

    def _query_camera_info_response(self, label: str, sensor_id: str) -> dict:
        """Query and cache one camera publisher's authoritative ``/info`` response."""
        cache = getattr(self, "_camera_info_responses", None)
        if cache is None:
            cache = self._camera_info_responses = {}
        if sensor_id in cache:
            return cache[sensor_id]
        node = getattr(self.robot, "_node", None)
        if node is None:
            raise RuntimeError(
                f"[wbc_vr_robot] recording requires a DexComm node to query "
                f"the {label} camera's calibration."
            )
        topic = f"sensors/{sensor_id}/info"
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
                f"[wbc_vr_robot] cannot query {label} stereo info {topic!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(response, dict):
            raise RuntimeError(
                f"[wbc_vr_robot] {topic!r} returned {type(response).__name__}, expected dict."
            )
        cache[sensor_id] = response
        return response

    def _query_stereo_calibration(self, label: str, sensor_id: str) -> dict:
        """Return one publisher's live calibration for its rectified output frames."""
        response = self._query_camera_info_response(label, sensor_id)

        streams = response.get("streams")
        right_stream = streams.get("right_rgb") if isinstance(streams, dict) else None
        if not isinstance(right_stream, dict) or not right_stream.get("enabled", False):
            raise RuntimeError(
                f"[wbc_vr_robot] {label} publisher reports right_rgb disabled. Restart "
                "its ZED publisher with --no-skip-right-rgb."
            )

        stereo = response.get("stereo")
        if not isinstance(stereo, dict):
            topic = f"sensors/{sensor_id}/info"
            raise RuntimeError(
                f"[wbc_vr_robot] {topic!r} has no 'stereo' block. Start the publisher "
                "from a build that serves rectified baseline and per-eye intrinsics."
            )
        if not stereo.get("rectified", False):
            raise RuntimeError(
                f"[wbc_vr_robot] {label} stereo calibration is not rectified; "
                "FoundationStereo requires row-aligned rectified views."
            )
        baseline_m = float(stereo.get("baseline_m", np.nan))
        if not np.isfinite(baseline_m) or not 0.01 < baseline_m < 0.5:
            raise ValueError(
                f"[wbc_vr_robot] implausible {label} stereo baseline {baseline_m} m."
            )

        out: dict = {
            "baseline_m": np.float32(baseline_m),
            "rectified": np.bool_(True),
            "sensor_id": np.asarray(sensor_id.encode()),
        }
        for eye in ("left_K", "right_K"):
            K = np.asarray(stereo.get(eye), dtype=np.float32)
            if (
                K.shape != (3, 3)
                or not np.all(np.isfinite(K))
                or float(K[0, 0]) <= 0.0
                or float(K[1, 1]) <= 0.0
                or not np.allclose(K[2], (0.0, 0.0, 1.0), atol=1e-6)
            ):
                raise ValueError(
                    f"[wbc_vr_robot] {label} stereo {eye} is not a finite pinhole 3x3 "
                    f"matrix: {K!r}."
                )
            out[eye] = K

        return out

    def _query_head_stereo_calibration(self) -> dict:
        """Return live rectified head calibration for the exact published pixels."""
        out = self._query_stereo_calibration("head", _HEAD_SENSOR_ID)

        if not np.allclose(out["left_K"], ZED_K.astype(np.float32), rtol=1e-3, atol=1e-2):
            print(
                "[wbc_vr_robot] NOTE: live publisher left_K differs from the static "
                "ZED_K fallback; recording the live matrix that describes these frames."
            )
        print(
            f"[wbc_vr_robot] Head stereo calibration: baseline "
            f"{float(out['baseline_m']) * 1000:.2f} mm, "
            f"left fx {float(out['left_K'][0, 0]):.2f} px "
            "-> obs/images/intrinsic + meta/head_stereo"
        )
        return out

    def _query_wrist_stereo_calibrations(self) -> dict[str, dict]:
        """Return per-arm live calibration for both wrist ZED-M publishers."""
        out: dict[str, dict] = {}
        for arm, sensor_id in _WRIST_SENSOR_IDS.items():
            out[arm] = self._query_stereo_calibration(f"{arm}_wrist", sensor_id)
            print(
                f"[wbc_vr_robot] {arm.capitalize()} wrist stereo calibration: baseline "
                f"{float(out[arm]['baseline_m']) * 1000:.2f} mm, "
                f"left fx {float(out[arm]['left_K'][0, 0]):.2f} px "
                f"-> meta/wrist_stereo/{arm}"
            )
        return out

    def _unwrap_frame_sample(self, name: str, entry) -> tuple[np.ndarray, int, int]:
        """Return ``(array, capture_ns, local_receive_ns)`` for one stream entry.

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
        receive_ns = entry.get("receive_time_ns")
        if receive_ns is None or int(receive_ns) <= 0:
            if bool(getattr(getattr(self, "args", None), "record", False)):
                raise ValueError(
                    f"[wbc_vr_robot] {name} frame has no valid receive_time_ns "
                    f"({receive_ns!r}); cannot separate transport/decode latency from "
                    "subscriber-cache age. Install dexcomm >= 0.4.6."
                )
            receive_ns = -1
        else:
            receive_ns = int(receive_ns)
            if receive_ns > time.time_ns() + 1_000_000_000:
                raise ValueError(
                    f"[wbc_vr_robot] {name} receive_time_ns {receive_ns} is in the future"
                )
            if not hasattr(self, "_last_camera_receive_ns"):
                self._last_camera_receive_ns = {}
            self._last_camera_receive_ns[name] = receive_ns
        return entry["data"], int(ts), int(receive_ns)

    def _unwrap_frame(self, name: str, entry) -> tuple[np.ndarray, int]:
        """Backward-compatible ``(array, capture_ns)`` view of one camera sample."""
        data, capture_ns, _receive_ns = self._unwrap_frame_sample(name, entry)
        return data, capture_ns

    def _grab_head_images(
        self,
    ) -> Optional[
        tuple[
            np.ndarray,
            Optional[np.ndarray],
            int,
            Optional[int],
            Optional[np.ndarray],
            Optional[int],
        ]
    ]:
        """Poll the selected head streams and return images plus capture timestamps.

        The tuple is ``(left_rgb, depth_u16, left_ns, depth_ns, right_rgb, right_ns)``.
        Unselected depth/right streams are represented by ``None`` and are omitted from
        the recorded HDF5 entirely. Selected streams must all have delivered at least one
        frame before this method returns a tuple, keeping every episode schema fixed.
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
            right_rgb, right_ns = self._unwrap_frame(
                "head_camera right_rgb", right_entry
            )
            right_rgb = np.asarray(right_rgb)
            if right_rgb.shape != left_rgb.shape:
                raise ValueError(
                    f"[wbc_vr_robot] head_camera right_rgb shape {right_rgb.shape} != "
                    f"left_rgb {left_rgb.shape}; the publisher must apply the same "
                    "--crop/--resize to both eyes."
                )
            right_rgb = np.ascontiguousarray(right_rgb, dtype=np.uint8)

        if not want_depth:
            return left_rgb, None, frame_ns, None, right_rgb, right_ns

        depth, depth_ns = self._unwrap_frame("head_camera depth", depth_entry)
        depth = np.asarray(depth)
        if depth.ndim != 2:
            raise ValueError(
                f"[wbc_vr_robot] head_camera depth shape {depth.shape} is not (H,W)."
            )
        # meters -> millimeters, clipped to uint16 (matches vr_reader's head_depth).
        depth_u16 = np.clip(depth * 1000, 0, 65535).astype(np.uint16)
        return left_rgb, depth_u16, frame_ns, depth_ns, right_rgb, right_ns

    def _grab_head_alignment_sample(self) -> Optional[_HeadAlignmentSample]:
        """Return one new, stereo-coherent head sample for the alignment history.

        This path first inspects capture stamps and only copies image arrays when the
        left-eye capture is distinct. Polling it at the 100 Hz control rate therefore
        does not copy two 600x960 RGB arrays on every cache read. When stereo is
        requested, an independently delivered half-pair is held out until both eyes
        expose the exact same physical-capture stamp.
        """
        want_depth = not bool(getattr(self.args, "no_head_depth", False))
        want_right = bool(getattr(self.args, "head_right_rgb", False))
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

        left_data, left_ns, left_receive_ns = self._unwrap_frame_sample(
            "head_camera left_rgb", left_entry
        )
        last_ns = int(getattr(self, "_last_buffered_head_ns", -1))
        if left_ns < last_ns:
            raise RuntimeError(
                "[wbc_vr_robot] --record: BACKWARD CAMERA CLOCK STEP -- head moved "
                f"backward {(last_ns - left_ns) / 1e6:.3f} ms "
                f"(previous buffered={last_ns}, current={left_ns}). No regressed "
                "frame was recorded; verify chrony and restart the publisher before "
                "collecting again."
            )
        if left_ns == last_ns:
            return None

        right_data = None
        right_ns = None
        right_receive_ns = None
        if want_right:
            right_data, right_ns, right_receive_ns = self._unwrap_frame_sample(
                "head_camera right_rgb", right_entry
            )
            if right_ns != left_ns:
                # Left and right are independent Zenoh messages. Keep polling until the
                # cache contains a completed physical pair; never buffer a mixed pair.
                return None

        left_rgb = np.asarray(left_data)
        if left_rgb.ndim != 3 or left_rgb.shape[2] != 3:
            raise ValueError(
                f"[wbc_vr_robot] head_camera left_rgb shape {left_rgb.shape} is not "
                "(H,W,3)."
            )
        left_rgb = np.ascontiguousarray(left_rgb, dtype=np.uint8)
        right_rgb = None
        if right_data is not None:
            right_rgb = np.asarray(right_data)
            if right_rgb.shape != left_rgb.shape:
                raise ValueError(
                    f"[wbc_vr_robot] head_camera right_rgb shape {right_rgb.shape} != "
                    f"left_rgb {left_rgb.shape}; the publisher must apply the same "
                    "--crop/--resize to both eyes."
                )
            right_rgb = np.ascontiguousarray(right_rgb, dtype=np.uint8)

        depth_u16 = None
        depth_ns = None
        depth_receive_ns = None
        if want_depth:
            depth_data, depth_ns, depth_receive_ns = self._unwrap_frame_sample(
                "head_camera depth", depth_entry
            )
            depth = np.asarray(depth_data)
            if depth.ndim != 2:
                raise ValueError(
                    f"[wbc_vr_robot] head_camera depth shape {depth.shape} is not (H,W)."
                )
            depth_u16 = np.clip(depth * 1000, 0, 65535).astype(np.uint16)

        return _HeadAlignmentSample(
            left_rgb=left_rgb,
            depth_u16=depth_u16,
            left_capture_ns=int(left_ns),
            depth_capture_ns=None if depth_ns is None else int(depth_ns),
            right_rgb=right_rgb,
            right_capture_ns=None if right_ns is None else int(right_ns),
            left_receive_ns=int(left_receive_ns),
            depth_receive_ns=(
                None if depth_receive_ns is None else int(depth_receive_ns)
            ),
            right_receive_ns=(
                None if right_receive_ns is None else int(right_receive_ns)
            ),
        )

    def _grab_head_rgb_sample(self) -> Optional[tuple[np.ndarray, int, int]]:
        """Poll head RGB -> image, publisher capture, and exact local receive stamp.

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
        left_rgb, frame_ns, receive_ns = self._unwrap_frame_sample(
            "head_camera left_rgb", entry
        )
        left_rgb = np.asarray(left_rgb)
        if left_rgb.ndim != 3 or left_rgb.shape[2] != 3:
            raise ValueError(
                f"[wbc_vr_robot] head_camera left_rgb shape {left_rgb.shape} is not (H,W,3)."
            )
        return np.ascontiguousarray(left_rgb, dtype=np.uint8), frame_ns, receive_ns

    def _grab_head_rgb(self) -> Optional[tuple[np.ndarray, int]]:
        """Backward-compatible image/capture view used outside policy instrumentation."""
        sample = self._grab_head_rgb_sample()
        return None if sample is None else (sample[0], sample[1])

    def _grab_wrist_image_sample(
        self, arm: str
    ) -> Optional[tuple[np.ndarray, int, int]]:
        """Poll one wrist left eye -> image, publisher capture, local receive stamp.

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
        wrist, frame_ns, receive_ns = self._unwrap_frame_sample(
            f"{sensor_id} {_WRIST_OBS_KEY}", entry
        )
        wrist = np.asarray(wrist)
        if wrist.ndim != 3 or wrist.shape[2] != 3:
            raise ValueError(
                f"[wbc_vr_robot] {sensor_id} {_WRIST_OBS_KEY} shape {wrist.shape} "
                "is not (H,W,3)."
            )
        return np.ascontiguousarray(wrist, dtype=np.uint8), frame_ns, receive_ns

    def _grab_wrist_image(self, arm: str) -> Optional[tuple[np.ndarray, int]]:
        """Backward-compatible image/capture view used by recording and replay."""
        sample = self._grab_wrist_image_sample(arm)
        return None if sample is None else (sample[0], sample[1])

    def _grab_wrist_stereo_sample(
        self, arm: str
    ) -> Optional[
        tuple[np.ndarray, int, int, np.ndarray, int, int]
    ]:
        """Poll both wrist eyes with exact capture and local-receive stamps.

        The existing ``<arm>_wrist_rgb`` dataset remains the stereo LEFT eye for
        backward compatibility.  The returned right eye is recorded separately as
        ``<arm>_wrist_right_rgb`` when ``--wrist-right-rgb`` is selected.
        """
        sensor_id = _WRIST_SENSOR_IDS[arm]
        obs = getattr(self.robot.sensors, sensor_id).get_obs(
            obs_keys=["left_rgb", "right_rgb"], include_timestamp=True
        )

        def find_entry(stream: str):
            entry = obs.get(stream)
            if entry is None:  # tolerate sensor-name-prefixed transport keys
                for key, value in obs.items():
                    if key.endswith("_" + stream):
                        return value
            return entry

        left_entry = find_entry("left_rgb")
        right_entry = find_entry("right_rgb")
        if left_entry is None or right_entry is None:
            return None

        left_rgb, left_ns, left_receive_ns = self._unwrap_frame_sample(
            f"{sensor_id} left_rgb", left_entry
        )
        right_rgb, right_ns, right_receive_ns = self._unwrap_frame_sample(
            f"{sensor_id} right_rgb", right_entry
        )
        left_rgb = np.asarray(left_rgb)
        right_rgb = np.asarray(right_rgb)
        if left_rgb.ndim != 3 or left_rgb.shape[2] != 3:
            raise ValueError(
                f"[wbc_vr_robot] {sensor_id} left_rgb shape {left_rgb.shape} "
                "is not (H,W,3)."
            )
        if right_rgb.shape != left_rgb.shape:
            raise ValueError(
                f"[wbc_vr_robot] {sensor_id} right_rgb shape {right_rgb.shape} != "
                f"left_rgb {left_rgb.shape}; the publisher must apply the same "
                "--crop/--resize to both eyes."
            )
        return (
            np.ascontiguousarray(left_rgb, dtype=np.uint8),
            left_ns,
            left_receive_ns,
            np.ascontiguousarray(right_rgb, dtype=np.uint8),
            right_ns,
            right_receive_ns,
        )

    def _grab_wrist_stereo(
        self, arm: str
    ) -> Optional[tuple[np.ndarray, int, np.ndarray, int]]:
        """Backward-compatible stereo image/capture view."""
        sample = self._grab_wrist_stereo_sample(arm)
        if sample is None:
            return None
        return sample[0], sample[1], sample[3], sample[4]

    def _camera_alignment_enabled(self) -> bool:
        """Whether record rows use head-anchored capture-nearest wrist frames."""
        return int(getattr(self, "_record_max_camera_skew_ns", 0)) > 0

    def _capture_local_ns(self, label: str, capture_ns: int) -> int:
        """Map one camera-host capture stamp onto this process's wall clock."""
        offsets = getattr(self, "_camera_clock_offsets_ns", {})
        if label not in offsets:
            raise RuntimeError(
                f"[wbc_vr_robot] no camera clock offset for {label!r}; cannot align "
                "multi-camera captures"
            )
        # camera offset = camera_host - local, hence local = camera - offset.
        return int(capture_ns) - int(offsets[label])

    def _poll_camera_alignment_buffers(self) -> None:
        """Cache every distinct coherent camera frame at the control cadence.

        This reads only dexcontrol's decoded latest-frame subscriber caches. It never
        blocks for a frame and never asks the publishers for additional work. Stereo
        samples enter history only when both eyes share the exact physical-capture stamp
        required by FoundationStereo.
        """
        if not self._camera_alignment_enabled():
            return
        poll_lock = getattr(self, "_wrist_alignment_poll_lock", None)
        if poll_lock is None:
            self._wrist_alignment_poll_lock = threading.Lock()
            poll_lock = self._wrist_alignment_poll_lock
        with poll_lock:
            self._poll_camera_alignment_buffers_once()

    def _poll_wrist_alignment_buffers(self) -> None:
        """Compatibility alias for callers predating buffered head selection."""
        self._poll_camera_alignment_buffers()

    def _poll_camera_alignment_buffers_once(self) -> None:
        """One serialized three-camera cache poll; caller holds the poll lock."""
        head_sample = self._grab_head_alignment_sample()
        if head_sample is not None:
            capture_ns = int(head_sample.left_capture_ns)
            alignment_lock = getattr(self, "_wrist_alignment_lock", None)
            if alignment_lock is None:
                self._wrist_alignment_lock = threading.Lock()
                alignment_lock = self._wrist_alignment_lock
            with alignment_lock:
                last_seen_ns = int(getattr(self, "_last_buffered_head_ns", -1))
                if capture_ns < last_seen_ns:
                    raise RuntimeError(
                        "[wbc_vr_robot] --record: BACKWARD CAMERA CLOCK STEP -- head "
                        f"moved backward {(last_seen_ns - capture_ns) / 1e6:.3f} ms "
                        f"(previous buffered={last_seen_ns}, current={capture_ns}). "
                        "No regressed frame was recorded; verify chrony and restart "
                        "the publisher before collecting again."
                    )
                if capture_ns > last_seen_ns:
                    self._last_buffered_head_ns = capture_ns
                    self._head_alignment_buffer.append(head_sample)

        want_right = bool(getattr(self.args, "wrist_right_rgb", False))
        for arm in _WRIST_SENSOR_IDS:
            if want_right:
                stereo = self._grab_wrist_stereo_sample(arm)
                if stereo is None:
                    continue
                sample = stereo
                if int(sample[1]) != int(sample[4]):
                    # Independently delivered eyes can be momentarily one cache update
                    # apart. Do not poison the history; the next 100 Hz poll can observe
                    # the completed pair. Persistent mismatch becomes an alignment timeout.
                    continue
            else:
                mono = self._grab_wrist_image_sample(arm)
                if mono is None:
                    continue
                sample = (mono[0], int(mono[1]), int(mono[2]), None, None, None)

            capture_ns = int(sample[1])
            alignment_lock = getattr(self, "_wrist_alignment_lock", None)
            if alignment_lock is None:
                self._wrist_alignment_lock = threading.Lock()
                alignment_lock = self._wrist_alignment_lock
            with alignment_lock:
                last_seen_ns = int(self._last_buffered_wrist_ns[arm])
                if capture_ns < last_seen_ns:
                    raise RuntimeError(
                        "[wbc_vr_robot] --record: BACKWARD CAMERA CLOCK STEP -- "
                        f"{arm}_wrist moved backward "
                        f"{(last_seen_ns - capture_ns) / 1e6:.3f} ms "
                        f"(previous buffered={last_seen_ns}, current={capture_ns}). "
                        "No regressed frame was recorded; verify chrony and restart the "
                        "publisher before collecting again."
                    )
                if capture_ns == last_seen_ns:
                    continue
                self._last_buffered_wrist_ns[arm] = capture_ns
                self._wrist_alignment_buffers[arm].append(sample)

    def _poll_wrist_alignment_buffers_once(self) -> None:
        """Compatibility alias; caller must hold the shared poll lock."""
        self._poll_camera_alignment_buffers_once()

    def _select_aligned_camera_samples(
        self,
        *,
        arms=None,
        last_head_ns: Optional[int] = None,
        last_wrist_ns: Optional[dict[str, int]] = None,
        max_skew_ns: Optional[int] = None,
    ) -> tuple[
        Optional[_HeadAlignmentSample],
        Optional[dict[str, tuple]],
        dict[str, int],
    ]:
        """Choose the freshest unused head anchor with a coherent wrist set.

        The newest head cache entry is not always selectable: at 30 Hz it can arrive
        before the corresponding 15 Hz wrist frames and remain more than one skew limit
        ahead. Trying buffered head anchors newest-to-oldest avoids that phase-chasing
        failure while keeping every selected capture strictly newer than this consumer's
        own cursors. The returned head sample is the diagnostic newest candidate when no
        full set is currently valid; in that case wrist samples are ``None``.
        """
        selected_arms = tuple(_WRIST_SENSOR_IDS if arms is None else arms)
        unknown_arms = sorted(set(selected_arms) - set(_WRIST_SENSOR_IDS))
        if unknown_arms:
            raise ValueError(f"unknown wrist arm(s) for camera alignment: {unknown_arms}")
        head_used = (
            int(self._last_rec_head_ns)
            if last_head_ns is None
            else int(last_head_ns)
        )
        wrist_used = (
            self._last_rec_wrist_ns if last_wrist_ns is None else last_wrist_ns
        )
        limit_ns = (
            int(self._record_max_camera_skew_ns)
            if max_skew_ns is None
            else int(max_skew_ns)
        )
        alignment_lock = getattr(self, "_wrist_alignment_lock", None)
        if alignment_lock is None:
            head_history = list(self._head_alignment_buffer)
            wrist_histories = {
                arm: list(self._wrist_alignment_buffers[arm])
                for arm in selected_arms
            }
        else:
            with alignment_lock:
                head_history = list(self._head_alignment_buffer)
                wrist_histories = {
                    arm: list(self._wrist_alignment_buffers[arm])
                    for arm in selected_arms
                }
        head_candidates = [
            sample
            for sample in reversed(head_history)
            if int(sample.left_capture_ns) > head_used
        ]
        if not head_candidates:
            return None, None, {}

        newest_skew_ns: dict[str, int] = {}
        for candidate_index, head_sample in enumerate(head_candidates):
            head_local_ns = self._capture_local_ns(
                "head", int(head_sample.left_capture_ns)
            )
            selected: dict[str, tuple] = {}
            skew_ns: dict[str, int] = {}
            for arm in selected_arms:
                candidates = [
                    sample
                    for sample in wrist_histories[arm]
                    if int(sample[1]) > int(wrist_used.get(arm, -1))
                ]
                if not candidates:
                    continue
                label = f"{arm}_wrist"
                best = min(
                    candidates,
                    key=lambda sample: (
                        abs(
                            head_local_ns
                            - self._capture_local_ns(label, int(sample[1]))
                        ),
                        -int(sample[1]),
                    ),
                )
                skew = head_local_ns - self._capture_local_ns(label, int(best[1]))
                skew_ns[arm] = int(skew)
                if abs(skew) <= limit_ns:
                    selected[arm] = best
            if candidate_index == 0:
                newest_skew_ns = skew_ns
            if len(selected) == len(selected_arms):
                return head_sample, selected, skew_ns
        return head_candidates[0], None, newest_skew_ns

    def _select_aligned_wrist_samples(
        self,
        head_ns: int,
        *,
        arms=None,
        last_used_ns: Optional[dict[str, int]] = None,
        max_skew_ns: Optional[int] = None,
    ) -> tuple[Optional[dict[str, tuple]], dict[str, int]]:
        """Select each arm's closest unused capture to ``head_ns``.

        Returns ``(samples, skew_ns)`` where skew is corrected
        ``head_capture_local - wrist_capture_local``. ``samples`` is ``None`` if an arm
        has no unused history or its closest candidate exceeds the configured limit.
        Selection is non-consuming: recorder and policy-inference consumers keep
        independent freshness cursors while the bounded deque ages old frames out.
        """
        selected_arms = tuple(_WRIST_SENSOR_IDS if arms is None else arms)
        unknown_arms = sorted(set(selected_arms) - set(_WRIST_SENSOR_IDS))
        if unknown_arms:
            raise ValueError(f"unknown wrist arm(s) for camera alignment: {unknown_arms}")
        used = self._last_rec_wrist_ns if last_used_ns is None else last_used_ns
        limit_ns = (
            int(self._record_max_camera_skew_ns)
            if max_skew_ns is None
            else int(max_skew_ns)
        )
        alignment_lock = getattr(self, "_wrist_alignment_lock", None)
        if alignment_lock is None:
            histories = {
                arm: list(self._wrist_alignment_buffers[arm]) for arm in selected_arms
            }
        else:
            with alignment_lock:
                histories = {
                    arm: list(self._wrist_alignment_buffers[arm])
                    for arm in selected_arms
                }
        head_local_ns = self._capture_local_ns("head", int(head_ns))
        selected: dict[str, tuple] = {}
        skew_ns: dict[str, int] = {}
        for arm in selected_arms:
            candidates = [
                sample
                for sample in histories[arm]
                if int(sample[1]) > int(used.get(arm, -1))
            ]
            if not candidates:
                continue
            label = f"{arm}_wrist"
            best = min(
                candidates,
                key=lambda sample: (
                    abs(head_local_ns - self._capture_local_ns(label, int(sample[1]))),
                    -int(sample[1]),
                ),
            )
            skew = head_local_ns - self._capture_local_ns(label, int(best[1]))
            skew_ns[arm] = int(skew)
            if abs(skew) <= limit_ns:
                selected[arm] = best
        if len(selected) != len(selected_arms):
            return None, skew_ns
        return selected, skew_ns

    def _record_camera_startup_ready(self, now: float) -> bool:
        """Arm frame zero only after all three camera histories are advancing.

        Episode-local histories are intentionally cleared at engage. A single coherent
        cache snapshot does not prove that a publisher is still advancing: episodes 5/6
        each committed one row and then immediately failed. Requiring four 30 Hz head
        frames, three 15 Hz frames per wrist, and one selectable set gives roughly
        130--200 ms of startup evidence while preserving the configured 10 Hz cadence
        after the first commit. Unit drivers that predate this gate omit the state field
        and are treated as already warm for backward-compatible focused tests.
        """
        if bool(getattr(self, "_record_camera_warmup_complete", True)):
            return True
        if not self._camera_alignment_enabled():
            self._record_camera_warmup_complete = True
            return True

        alignment_lock = getattr(self, "_wrist_alignment_lock", None)
        if alignment_lock is None:
            head_history = list(self._head_alignment_buffer)
            wrist_histories = {
                arm: list(self._wrist_alignment_buffers[arm])
                for arm in _WRIST_SENSOR_IDS
            }
        else:
            with alignment_lock:
                head_history = list(self._head_alignment_buffer)
                wrist_histories = {
                    arm: list(self._wrist_alignment_buffers[arm])
                    for arm in _WRIST_SENSOR_IDS
                }
        counts = {
            "head": len(head_history),
            **{
                f"{arm}_wrist": len(wrist_histories[arm])
                for arm in _WRIST_SENSOR_IDS
            },
        }
        enough_history = (
            counts["head"] >= _RECORD_STARTUP_MIN_HEAD_FRAMES
            and all(
                counts[f"{arm}_wrist"] >= _RECORD_STARTUP_MIN_WRIST_FRAMES
                for arm in _WRIST_SENSOR_IDS
            )
        )
        aligned_head, aligned_wrists, skew_ns = self._select_aligned_camera_samples()
        if enough_history and aligned_head is not None and aligned_wrists is not None:
            elapsed_ms = (
                0.0
                if self._record_camera_warmup_since is None
                else (now - self._record_camera_warmup_since) * 1000.0
            )
            detail = ", ".join(
                f"{arm}={skew_ns[arm] / 1e6:+.1f}ms"
                for arm in _WRIST_SENSOR_IDS
            )
            print(
                "\n[wbc_vr_robot] --record: camera startup stable after "
                f"{elapsed_ms:.0f} ms ({counts}, {detail}); arming frame zero."
            )
            self._record_camera_warmup_complete = True
            self._record_camera_warmup_since = None
            self._camera_alignment_stale_since = None
            self._next_record_t = now
            return True

        if self._record_camera_warmup_since is None:
            self._record_camera_warmup_since = now
            print(
                "\n[wbc_vr_robot] --record: validating advancing camera startup "
                f"histories (need head>={_RECORD_STARTUP_MIN_HEAD_FRAMES}, each "
                f"wrist>={_RECORD_STARTUP_MIN_WRIST_FRAMES}, plus |skew|<="
                f"{self._record_max_camera_skew_ns / 1e6:g}ms); no frame committed yet."
            )
        elapsed = now - self._record_camera_warmup_since
        timeout_s = float(getattr(
            self,
            "_record_startup_timeout_s",
            DEFAULT_RECORD_STARTUP_TIMEOUT,
        ))
        if elapsed <= timeout_s:
            self._next_record_t = now
            return False

        def newest(values, index: int) -> Optional[int]:
            return None if not values else int(values[-1][index])

        newest_stamps = {
            "head": (
                None
                if not head_history
                else int(head_history[-1].left_capture_ns)
            ),
            **{
                f"{arm}_wrist": newest(wrist_histories[arm], 1)
                for arm in _WRIST_SENSOR_IDS
            },
        }
        skew_detail = {
            arm: (None if arm not in skew_ns else skew_ns[arm] / 1e6)
            for arm in _WRIST_SENSOR_IDS
        }
        raise RuntimeError(
            "[wbc_vr_robot] --record: camera startup never became stable within "
            f"--record-startup-timeout {timeout_s:g}s; counts={counts}, "
            f"newest_capture_ns={newest_stamps}, newest_head_skew_ms={skew_detail}. "
            "No frame was committed. Verify all three publisher FPS/clock diagnostics "
            "before re-engaging."
        )

    def _retry_or_abort_camera_alignment(
        self,
        now: float,
        *,
        head_ns: int,
        skew_ns: dict[str, int],
    ) -> None:
        """Retry a row until both head-to-wrist skews satisfy the hard contract."""
        limit_ms = int(self._record_max_camera_skew_ns) / 1e6

        def arm_detail(arm: str) -> str:
            if arm not in skew_ns:
                return f"{arm}=NO_UNUSED_PAIR"
            return f"{arm}={skew_ns[arm] / 1e6:+.1f}ms"

        detail = ", ".join(arm_detail(arm) for arm in _WRIST_SENSOR_IDS)
        if self._camera_alignment_stale_since is None:
            self._camera_alignment_stale_since = now
            print(
                "\n[wbc_vr_robot] --record: no capture-aligned wrist set for head "
                f"{head_ns} ({detail}; require |skew| <= {limit_ms:g}ms); retrying "
                f"at the IK rate for up to {self._stale_grace * 1000.0:.0f} ms."
            )
        if now - self._camera_alignment_stale_since <= self._stale_grace:
            self._next_record_t = now
            return
        alignment_lock = getattr(self, "_wrist_alignment_lock", None)
        if alignment_lock is None:
            buffered = {
                arm: [int(sample[1]) for sample in self._wrist_alignment_buffers[arm]]
                for arm in _WRIST_SENSOR_IDS
            }
        else:
            with alignment_lock:
                buffered = {
                    arm: [int(sample[1]) for sample in self._wrist_alignment_buffers[arm]]
                    for arm in _WRIST_SENSOR_IDS
                }
        raise RuntimeError(
            "[wbc_vr_robot] --record: camera alignment failed for "
            f"{(now - self._camera_alignment_stale_since) * 1000.0:.0f} ms "
            f"(> --record-stale-grace {self._stale_grace * 1000.0:.0f} ms): "
            f"head={head_ns}, {detail}, max_abs={limit_ms:g}ms, "
            f"buffered_wrist_capture_ns={buffered}. Aborting so training never sees "
            "a silently time-skewed multi-camera row."
        )

    def _camera_capture_ages_ns(
        self,
        wall_ns: int,
        *,
        head_ns: int,
        wrist_ns: dict[str, int],
    ) -> dict[str, int]:
        """Capture-to-local-selection ages after camera-host clock correction."""
        offsets = getattr(self, "_camera_clock_offsets_ns", {})
        required = {"head", *(f"{arm}_wrist" for arm in wrist_ns)}
        missing = sorted(required - set(offsets))
        if missing:
            raise RuntimeError(
                "[wbc_vr_robot] camera-age gate is missing clock offsets for "
                f"{missing}"
            )
        ages = {"head": int(wall_ns - (int(head_ns) - int(offsets["head"])))}
        ages.update(
            {
                f"{arm}_wrist": int(
                    wall_ns
                    - (int(stamp) - int(offsets[f"{arm}_wrist"]))
                )
                for arm, stamp in wrist_ns.items()
            }
        )
        return ages

    def _retry_or_abort_camera_age(
        self, now: float, *, ages_ns: dict[str, int]
    ) -> None:
        """Retry then abort when fresh frames are nevertheless transport-stale."""
        limit_ns = int(self._record_max_camera_age_ns)
        detail = ", ".join(
            f"{label}={age / 1e6:+.1f}ms" for label, age in ages_ns.items()
        )
        if self._camera_age_stale_since is None:
            self._camera_age_stale_since = now
            print(
                "\n[wbc_vr_robot] --record: fresh camera capture is outside the "
                f"age contract ({detail}; require -"
                f"{_CAMERA_FUTURE_TOLERANCE_NS / 1e6:g} <= age <= "
                f"{limit_ns / 1e6:g} ms); retrying at the IK rate for up to "
                f"{self._stale_grace * 1000.0:.0f} ms."
            )
        if now - self._camera_age_stale_since <= self._stale_grace:
            self._next_record_t = now
            return
        raise RuntimeError(
            "[wbc_vr_robot] --record: camera delivery/cache age stayed outside "
            f"--record-max-camera-age-ms={limit_ns / 1e6:g} for "
            f"{(now - self._camera_age_stale_since) * 1000.0:.0f} ms: {detail}. "
            "Aborting so image observations are not silently several frames older "
            "than robot state/action. Reduce camera bandwidth/encode latency or raise "
            "the limit only as an explicit dataset contract."
        )

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

        # A repeated stamp (delta == 0) is normally just a cached latest frame and gets
        # the short retry grace below. A stamp that moves BACKWARD is qualitatively
        # different: the publisher host's wall clock stepped, or the publisher restarted
        # with a broken clock epoch. Waiting cannot make the already-recorded timing
        # calibration valid again, so fail immediately. The caller's top-level exception
        # path publishes ABORTED to the leader HUD and HardwareDriver.close() stops every
        # actuator before flushing the partial episode.
        backward: list[str] = []

        def note_backward(name: str, ts: Optional[int], last_ns: int) -> None:
            if ts is not None and last_ns >= 0 and ts < last_ns:
                backward.append(
                    f"{name} moved backward {(last_ns - ts) / 1e6:.3f} ms "
                    f"(previous={last_ns}, current={ts})"
                )

        note_backward("head", head_ns, self._last_rec_head_ns)
        for arm in _WRIST_SENSOR_IDS:
            note_backward(
                f"{arm}_wrist", wrist_ns[arm], self._last_rec_wrist_ns[arm]
            )
        if backward:
            raise RuntimeError(
                "[wbc_vr_robot] --record: BACKWARD CAMERA CLOCK STEP -- "
                + "; ".join(backward)
                + ". Aborting immediately: no regressed frame was recorded, and the "
                  "once-per-run camera clock calibration is no longer valid. Park the "
                  "robot and verify chrony/publisher-host time before collecting again."
            )

        def diag(name: str, ts: Optional[int], last_ns: int) -> str:
            if ts is None:
                return f"{name}=MISSING"
            return f"{name} Δ={ts - last_ns} ns age={(wall_ns - ts) / 1e6:.0f} ms"

        detail = ", ".join((
            diag("head", head_ns, self._last_rec_head_ns),
            *(
                diag(f"{arm}_wrist", wrist_ns[arm], self._last_rec_wrist_ns[arm])
                for arm in _WRIST_SENSOR_IDS
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

    def _recording_extra_action(self) -> dict:
        """Subclass-specific raw action fields merged into each recorded frame."""
        return {}

    def record_tick(
        self,
        result,
        hold: bool,
        now: float,
        left_target: np.ndarray,
        right_target: np.ndarray,
        head_target: np.ndarray,
        source_timestamp_ns: int,
        source_receive_wall_ns: int,
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

        ``source_timestamp_ns`` is the command producer's wall timestamp (the VR
        leader in live teleop); ``source_receive_wall_ns`` is the follower subscriber
        callback's local wall timestamp for that exact command. Together with the
        dispatch/state brackets below they make every causal stage auditable.
        """
        if self._episode is None or not self._episode.recording or hold:
            return
        # Sample the already-decoded camera caches on EVERY 100 Hz control tick, not
        # only on due 10 Hz rows. This preserves the frames needed to choose the
        # freshest coherent head/wrist set without changing bandwidth or blocking
        # actuation.
        self._poll_camera_alignment_buffers()
        # At a fresh engage, prove that every publisher is advancing before committing
        # frame zero. This startup-only gate prevents a one-row file from being created
        # from stale subscriber caches; normal 10 Hz throttling and the stricter
        # mid-episode retry/abort contracts begin immediately after it passes.
        if not self._record_camera_startup_ready(now):
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
        source_timestamp_ns = int(source_timestamp_ns)
        source_receive_wall_ns = int(source_receive_wall_ns)
        dispatch_start_ns = int(getattr(
            self, "_last_action_dispatch_start_wall_ns", -1
        ))
        dispatch_end_ns = int(getattr(
            self, "_last_action_dispatch_end_wall_ns", -1
        ))
        if source_timestamp_ns <= 0 or source_receive_wall_ns <= 0:
            raise RuntimeError(
                "[wbc_vr_robot] record_tick requires positive source command publish "
                "and receive timestamps"
            )
        if dispatch_start_ns <= 0 or dispatch_end_ns < dispatch_start_ns:
            raise RuntimeError(
                "[wbc_vr_robot] record_tick requires a completed actuation timestamp "
                "bracket from actuate()"
            )
        if source_receive_wall_ns > dispatch_start_ns:
            raise RuntimeError(
                "[wbc_vr_robot] source command receive time is later than actuation "
                f"start ({source_receive_wall_ns} > {dispatch_start_ns})"
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

        want_wrist_right = bool(
            getattr(getattr(self, "args", None), "wrist_right_rgb", False)
        )
        aligned_head_sample: Optional[_HeadAlignmentSample] = None
        aligned_samples: Optional[dict[str, tuple]] = None
        alignment_skew_ns: dict[str, int] = {}
        if self._camera_alignment_enabled():
            (
                aligned_head_sample,
                aligned_samples,
                alignment_skew_ns,
            ) = self._select_aligned_camera_samples()
            imgs = (
                None
                if aligned_head_sample is None
                else aligned_head_sample.images_tuple()
            )
            wrist = (
                {arm: None for arm in _WRIST_SENSOR_IDS}
                if aligned_samples is None
                else {
                    arm: (sample[0], sample[1], sample[3], sample[4])
                    for arm, sample in aligned_samples.items()
                }
            )
        else:
            imgs = self._grab_head_images()
            if want_wrist_right:
                wrist = {
                    arm: self._grab_wrist_stereo(arm) for arm in _WRIST_SENSOR_IDS
                }
            else:
                wrist = {
                    arm: (
                        None
                        if (mono := self._grab_wrist_image(arm)) is None
                        else (mono[0], mono[1], None, None)
                    )
                    for arm in _WRIST_SENSOR_IDS
                }
        # Before the first recorded frame the cameras may still be warming up (the leader can
        # reach teleop before a publisher is fully up): tolerate a missing stream and skip
        # this tick, matching vr_reader.
        recording_started = self._last_rec_head_ns >= 0
        if imgs is None and not recording_started:
            return  # camera still warming up -- skip rather than write inconsistent keys
        if self._camera_alignment_enabled() and imgs is not None and aligned_samples is None:
            self._retry_or_abort_camera_alignment(
                now,
                head_ns=int(imgs[2]),
                skew_ns=alignment_skew_ns,
            )
            return
        if any(w is None for w in wrist.values()) and not recording_started:
            return  # compatibility mode: wrist camera still warming up
        if (
            getattr(self, "_camera_alignment_stale_since", None) is not None
            and aligned_samples is not None
        ):
            detail = ", ".join(
                f"{arm}={alignment_skew_ns[arm] / 1e6:+.1f}ms"
                for arm in _WRIST_SENSOR_IDS
            )
            print(
                "\n[wbc_vr_robot] --record: capture-aligned wrist frames resumed "
                f"after {(now - self._camera_alignment_stale_since) * 1000.0:.0f} ms "
                f"({detail})."
            )
            self._camera_alignment_stale_since = None
        # Freshness guard: ~15 fps camera vs ~10 fps record means every recorded frame must
        # carry a strictly newer publisher timestamp than the last recorded one. A non-newer
        # (or missing) frame is the delivery path repeating the cached latest -- the
        # publishers hold their fps through these stalls, so suspect a WiFi burst, Zenoh
        # backlog, or this process starving its Zenoh decode threads under solver load.
        # Instead of dying on the first stale tick, retry at the IK rate for up to
        # --record-stale-grace: nothing is recorded until EVERY timestamp advances, so a take
        # still never contains duplicates; a short stall costs one late frame, not the take.
        head_ns = None if imgs is None else imgs[2]
        wrist_ns: dict[str, Optional[int]] = {
            arm: (None if w is None else w[1]) for arm, w in wrist.items()
        }
        if (
            head_ns is None
            or head_ns <= self._last_rec_head_ns
            or any(
                ns is None or ns <= self._last_rec_wrist_ns[arm]
                for arm, ns in wrist_ns.items()
            )
        ):
            self._retry_or_abort_stale(now, head_ns=head_ns, wrist_ns=wrist_ns)
            return
        if self._stale_since is not None:
            deltas = ", ".join(
                f"{arm} wrist Δ={wrist_ns[arm] - self._last_rec_wrist_ns[arm]} ns"
                for arm in _WRIST_SENSOR_IDS
            )
            print(f"\n[wbc_vr_robot] --record: fresh frames resumed after "
                  f"{(now - self._stale_since) * 1000.0:.0f} ms (head Δ="
                  f"{head_ns - self._last_rec_head_ns} ns, {deltas}).")
            self._stale_since = None

        age_limit_ns = int(getattr(self, "_record_max_camera_age_ns", 0))
        if age_limit_ns > 0:
            capture_ages_ns = self._camera_capture_ages_ns(
                time.time_ns(),
                head_ns=int(head_ns),
                wrist_ns={arm: int(ns) for arm, ns in wrist_ns.items()},
            )
            bad_age = any(
                age < -_CAMERA_FUTURE_TOLERANCE_NS or age > age_limit_ns
                for age in capture_ages_ns.values()
            )
            if bad_age:
                self._retry_or_abort_camera_age(now, ages_ns=capture_ages_ns)
                return
            if getattr(self, "_camera_age_stale_since", None) is not None:
                detail = ", ".join(
                    f"{label}={age / 1e6:.1f}ms"
                    for label, age in capture_ages_ns.items()
                )
                print(
                    "\n[wbc_vr_robot] --record: camera capture age recovered "
                    f"({detail})."
                )
                self._camera_age_stale_since = None
        (
            head_left_rgb,
            head_depth_u16,
            _,
            head_depth_ns,
            head_right_rgb,
            head_right_ns,
        ) = imgs

        # The independently delivered right-eye message can lag the left by one frame.
        # Retry for at most one record period so FoundationStereo receives same-capture
        # pairs without blocking the IK loop. A persistent mismatch aborts the take:
        # different-capture eyes cannot be repaired offline and must never be written as
        # a nominal stereo pair.
        if head_right_ns is not None and head_right_ns != head_ns:
            if self._right_pair_retry:
                if self._right_unpaired_since is None:
                    self._right_unpaired_since = now
                if now - self._right_unpaired_since <= min(
                    self._stale_grace, self._record_period
                ):
                    self._next_record_t = now
                    return
            self._rec_right_unpaired += 1
            self._right_unpaired_streak += 1
            raise RuntimeError(
                "[wbc_vr_robot] --record: head stereo eyes stayed on different "
                f"captures for {(now - self._right_unpaired_since) * 1000.0:.1f} ms: "
                f"left={head_ns}, right={head_right_ns}. Aborting because "
                "FoundationStereo requires same-capture rectified eyes."
            )
        else:
            self._right_unpaired_streak = 0
        self._right_unpaired_since = None

        # Apply the same same-capture contract independently to each wrist stereo
        # pair. The two wrist cameras are not synchronized with each other (or the
        # head); only the left/right eyes within a physical ZED must share a stamp.
        if want_wrist_right:
            for arm, sample in wrist.items():
                assert sample is not None
                right_ns = sample[3]
                assert right_ns is not None
                if right_ns != wrist_ns[arm]:
                    if self._wrist_right_pair_retry[arm]:
                        if self._wrist_right_unpaired_since[arm] is None:
                            self._wrist_right_unpaired_since[arm] = now
                        if now - self._wrist_right_unpaired_since[arm] <= min(
                            self._stale_grace, self._record_period
                        ):
                            self._next_record_t = now
                            return
                    self._rec_wrist_right_unpaired[arm] += 1
                    self._wrist_right_unpaired_streak[arm] += 1
                    raise RuntimeError(
                        f"[wbc_vr_robot] --record: {arm} wrist stereo eyes stayed on "
                        "different captures for "
                        f"{(now - self._wrist_right_unpaired_since[arm]) * 1000.0:.1f} "
                        f"ms: left={wrist_ns[arm]}, right={right_ns}. Aborting because "
                        "FoundationStereo requires same-capture rectified eyes."
                    )
                else:
                    self._wrist_right_unpaired_streak[arm] = 0
                self._wrist_right_unpaired_since[arm] = None

        # Drain achieved-gripper callbacks at the record cadence. A frame is eligible
        # only while BOTH arms have post-engage FC03 samples inside the bounded age
        # window. A single missed reply is an explicit zero-order hold, not a dropped
        # image/action row; an unbounded cache still fails closed.
        gripper_statuses: Optional[dict[str, dict]] = None
        if self._grip_monitors:
            self._poll_gripper_status_step(allow_request=False)
            gripper_statuses = self._fresh_gripper_statuses()
            if gripper_statuses is None:
                if self._grip_status_stale_since is None:
                    self._grip_status_stale_since = now
                grace_s = max(0.5, self._stale_grace)
                if now - self._grip_status_stale_since <= grace_s:
                    # Do not advance the last-recorded camera stamps: the same candidate
                    # images remain valid while the missing FC03 reply catches up.
                    self._next_record_t = now
                    return
                stats = {
                    side: monitor.stats()
                    for side, monitor in self._grip_monitors.items()
                }
                raise RuntimeError(
                    "[wbc_vr_robot] --record: no valid bounded-age gripper status from "
                    f"both arms within {grace_s:.3f}s (selection maximum "
                    f"{(_GRIPPER_STATUS_MAX_AGE_PERIODS - _GRIPPER_STATUS_COMMIT_MARGIN_PERIODS) * self._record_period * 1000.0:.0f} "
                    "ms, reserving commit margin inside the exact offline limit). "
                    f"monitor_stats={stats}"
                )
            self._grip_status_stale_since = None

        # Arrival-age history for the stale-abort diagnostic: local wall clock minus the
        # publisher capture stamp (spans the camera-host->workstation clock offset, so
        # compare head vs wrist and the trend over time, not the absolute value).
        wall_ns = time.time_ns()
        self._frame_age_log.append((
            now,
            (wall_ns - head_ns) / 1e6,
            (wall_ns - wrist_ns["left"]) / 1e6,
            (wall_ns - wrist_ns["right"]) / 1e6,
        ))
        self._last_rec_head_ns = head_ns
        # Every value is non-None past the freshness gate above.
        self._last_rec_wrist_ns = {arm: int(ns) for arm, ns in wrist_ns.items()}

        # Measured joints -> obs/joint (also the offline FK source for world_t_cam).
        # Validate shapes aggressively -- a bad readback must abort, never be recorded as-is.
        state_read_start_ns = time.time_ns()
        obs: dict[str, np.ndarray] = {}
        for grp, names in self._joint_names.items():
            meas = np.asarray(self._comp(grp).get_joint_pos(), dtype=np.float32)
            if meas.shape != (len(names),) or not np.all(np.isfinite(meas)):
                raise ValueError(
                    f"[wbc_vr_robot] {grp} joint readback {meas.shape} is not "
                    f"({len(names)},) or non-finite -- cannot record."
                )
            obs[grp] = meas

        # One odometry snapshot supplies BOTH pose and twist below. Reading the two
        # properties separately can straddle a 200 Hz odometry update and create a
        # physically impossible state row. ARKit is snapshotted inside the same read
        # bracket, though it remains an independent sensor sample.
        odom_state = self._odom.snapshot()
        arkit_state = self._arkit.snapshot() if self._arkit is not None else None
        state_read_end_ns = time.time_ns()
        if state_read_start_ns < dispatch_end_ns or state_read_end_ns < state_read_start_ns:
            raise RuntimeError(
                "[wbc_vr_robot] invalid actuation/state timing order: "
                f"dispatch_end={dispatch_end_ns}, state_start={state_read_start_ns}, "
                f"state_end={state_read_end_ns}"
            )

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
            meas_twist = np.asarray(odom_state["twist"], dtype=np.float64)
            if meas_twist.shape != (3,) or not np.all(np.isfinite(meas_twist)):
                raise RuntimeError(
                    f"[wbc_vr_robot] odometry twist {meas_twist!r} is not finite (3,)"
                )
            obs_joint["chassis_vx"] = np.float32(meas_twist[0])
            obs_joint["chassis_vy"] = np.float32(meas_twist[1])
            obs_joint["chassis_wz"] = np.float32(meas_twist[2])
        images = {
            "head_left_rgb": head_left_rgb,
            # left/right = the ARM each camera is mounted on (see _WRIST_SENSOR_IDS).
            "left_wrist_rgb": wrist["left"][0],
            "right_wrist_rgb": wrist["right"][0],
            # Publisher capture stamps (camera-publisher-host clock) of THIS tick's frames plus the
            # local wall clock right after all grabs -- the raw material for offline
            # camera-latency audits/correction (scripts/audit_episode_latency.py, with
            # meta/camera_ntp when available). The freshness gate above requires head RGB and
            # both wrist stamps to advance; depth's stamp is recorded separately for RGBD audits.
            "head_frame_ns": np.int64(head_ns),
            "left_wrist_frame_ns": np.int64(wrist_ns["left"]),
            "right_wrist_frame_ns": np.int64(wrist_ns["right"]),
            "grab_wall_ns": np.int64(wall_ns),
            # intrinsic is recorded once via set_static (not per frame); extrinsic is NOT
            # recorded -- world_t_cam is recomputed offline from obs/base/pose + obs/joint.
        }
        if head_depth_u16 is not None:
            assert head_depth_ns is not None
            images["head_depth"] = head_depth_u16
            images["head_depth_frame_ns"] = np.int64(head_depth_ns)
        if head_right_rgb is not None:
            assert head_right_ns is not None
            images["head_right_rgb"] = head_right_rgb
            images["head_right_frame_ns"] = np.int64(head_right_ns)
        if want_wrist_right:
            for arm, sample in wrist.items():
                assert sample is not None and sample[2] is not None and sample[3] is not None
                images[f"{arm}_wrist_right_rgb"] = sample[2]
                images[f"{arm}_wrist_right_frame_ns"] = np.int64(sample[3])
        receive_keys = {
            "head_receive_ns": "head_camera left_rgb",
            **{
                f"{arm}_wrist_receive_ns": f"{sensor_id} left_rgb"
                for arm, sensor_id in _WRIST_SENSOR_IDS.items()
            },
        }
        if head_depth_u16 is not None:
            receive_keys["head_depth_receive_ns"] = "head_camera depth"
        if head_right_rgb is not None:
            receive_keys["head_right_receive_ns"] = "head_camera right_rgb"
        if want_wrist_right:
            receive_keys.update({
                f"{arm}_wrist_right_receive_ns": f"{sensor_id} right_rgb"
                for arm, sensor_id in _WRIST_SENSOR_IDS.items()
            })
        receive_cache = getattr(self, "_last_camera_receive_ns", {})
        receive_values = {
            output_key: receive_cache.get(cache_key)
            for output_key, cache_key in receive_keys.items()
        }
        if aligned_head_sample is not None:
            # As with wrists below, the global subscriber cache may already point at a
            # newer head frame than the buffered anchor selected for this row.
            receive_values["head_receive_ns"] = int(
                aligned_head_sample.left_receive_ns
            )
            if head_depth_u16 is not None:
                assert aligned_head_sample.depth_receive_ns is not None
                receive_values["head_depth_receive_ns"] = int(
                    aligned_head_sample.depth_receive_ns
                )
            if head_right_rgb is not None:
                assert aligned_head_sample.right_receive_ns is not None
                receive_values["head_right_receive_ns"] = int(
                    aligned_head_sample.right_receive_ns
                )
        if aligned_samples is not None:
            # The global subscriber cache now points at the newest polled wrist frame,
            # which may be newer than the history sample selected for this row. Preserve
            # the selected frame's own receive stamp so delivery/cache-age audits remain
            # exact.
            for arm, sample in aligned_samples.items():
                receive_values[f"{arm}_wrist_receive_ns"] = int(sample[2])
                if want_wrist_right:
                    assert sample[5] is not None
                    receive_values[f"{arm}_wrist_right_receive_ns"] = int(sample[5])
        if any(value is not None for value in receive_values.values()):
            missing = sorted(
                key for key, value in receive_values.items() if value is None
            )
            if missing:
                raise RuntimeError(
                    "[wbc_vr_robot] partial camera receive timestamps after a complete "
                    f"image grab; missing {missing}"
                )
            for key, value in receive_values.items():
                assert value is not None
                if value > wall_ns:
                    raise RuntimeError(
                        f"[wbc_vr_robot] obs/images/{key}={value} is later than "
                        f"grab_wall_ns={wall_ns}"
                    )
                images[key] = np.int64(value)
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
        if gripper_statuses is not None:
            # Full status provenance is tiny compared with RGB and turns a bad gPO trace
            # into a diagnosable failure (command echo, faults, object state, request /
            # response identity and wall-clock timing) instead of an unexplained scalar.
            obs_out["gripper_status"] = {
                side: self._recordable_gripper_status(status)
                for side, status in gripper_statuses.items()
            }
        if self._odom is not None:
            # Canonical measured pose in the SAME engage-origin frame/source that closes
            # the base PD loop: odometry normally, ARKit under --arkit-base control. This
            # keeps policy state and world-frame action targets physically consistent.
            # Wheel odometry remains explicit as pose_odom because it still supplies the
            # achieved chassis twist and is needed to audit drift.
            odom_pose = np.asarray(odom_state["pose"], dtype=np.float32)
            if odom_pose.shape != (3,) or not np.all(np.isfinite(odom_pose)):
                raise RuntimeError(
                    f"[wbc_vr_robot] odometry pose {odom_pose!r} is not finite (3,)"
                )
            canonical_pose = odom_pose
            if self._arkit_drives_base():
                assert arkit_state is not None
                canonical_pose = np.asarray(arkit_state["pose"], dtype=np.float32)
                if canonical_pose.shape != (3,) or not np.all(np.isfinite(canonical_pose)):
                    raise RuntimeError(
                        f"[wbc_vr_robot] ARKit control pose {canonical_pose!r} is not "
                        "finite (3,)"
                    )
            obs_out["base"] = {
                "pose": canonical_pose,
                "pose_odom": odom_pose,
            }
            if self._arkit is not None:
                # Explicit drift-free source. Under control this equals canonical ``pose``;
                # under record-only mode canonical remains the odometry control source.
                # ``_world`` is continuous across re-engages in one ARKit session.
                assert arkit_state is not None
                obs_out["base"]["pose_arkit"] = np.asarray(
                    arkit_state["pose"], dtype=np.float32
                )
                obs_out["base"]["pose_arkit_world"] = np.asarray(
                    arkit_state["world_pose"], dtype=np.float32)
        commit_wall_ns = time.time_ns()
        if gripper_statuses is not None:
            self._validate_gripper_status_commit_age(
                gripper_statuses,
                commit_wall_ns,
            )
        frame = {
            # Historical episode timestamp: now explicitly the final record-commit
            # wall time. The causal stage timestamps live in /timing below.
            "timestamp_ns": np.int64(commit_wall_ns),
            "timing": {
                "source_timestamp_ns": np.int64(source_timestamp_ns),
                "source_receive_wall_ns": np.int64(source_receive_wall_ns),
                "action_dispatch_start_wall_ns": np.int64(dispatch_start_ns),
                "action_dispatch_end_wall_ns": np.int64(dispatch_end_ns),
                "state_read_start_wall_ns": np.int64(state_read_start_ns),
                "state_read_end_wall_ns": np.int64(state_read_end_ns),
            },
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
        extra_action = self._recording_extra_action()
        if not isinstance(extra_action, dict):
            raise TypeError("_recording_extra_action() must return a dict")
        overlap = set(frame["action"]).intersection(extra_action)
        if overlap:
            raise RuntimeError(
                f"extra recording action collides with canonical keys {sorted(overlap)}"
            )
        frame["action"].update(extra_action)
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


def _build_ik(
    *, lock_torso_in_ik: Optional[bool] = None
) -> tuple[VegaWholeBodyIK, WBCConfig]:
    cfg = WBCConfig()
    if lock_torso_in_ik is not None:
        cfg.lock_torso_in_ik = bool(lock_torso_in_ik)
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
    last_cmd_receive_wall_ns = -1
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
          f"enable={[k for k, v in enable.items() if v]} grippers=on")

    def resync_to_nominal(reason: str) -> None:
        """Re-anchor the solver and the target streams on the nominal posture.

        The engage transition and a home request both leave the HARDWARE at nominal, so
        the model, the interpolator/filters and the odometry origin have to be put back
        there in lock-step. Without this after a home request the solver kept tracking the
        pre-home targets from its stale configuration, so the leader HUD kept bannering
        that posture's self-collision warning while the robot stood at nominal.
        """
        nonlocal left_cmd, right_cmd, head_cmd, last_cmd_ns, last_cmd_receive_wall_ns
        ik.reset()
        interp.reset(left0, right0, head0)
        head_lpf.reset(head0)
        head_planar_deadband.reset(head0)
        left_cmd, right_cmd, head_cmd = left0.copy(), right0.copy(), head0.copy()
        last_cmd_ns = -1
        last_cmd_receive_wall_ns = -1
        driver.engage_reset(ik, left0, right0, head0)
        print(f"\n[wbc_vr_robot] {reason}: reset IK to nominal.")

    while True:
        now = clock()
        t = now - t0
        if replay and source.done:
            break
        latest_sample = getattr(source, "latest_sample", None)
        if latest_sample is not None:
            vr, vr_receive_wall_ns = latest_sample
        else:
            vr = source.latest
            # Replay and test sources have no transport callback. Stamp when this loop
            # observes the released command, preserving the same causal schema.
            vr_receive_wall_ns = time.time_ns()
        # A specialized driver may consume fields from the same command object as the
        # Cartesian target path. Latch it once here so a subscriber callback cannot advance
        # ``source.latest`` between the arm/head solve and base dispatch/recording.
        if hasattr(driver, "latch_source_sample"):
            driver.latch_source_sample(vr)
        if vr is not None and vr.exit_requested:
            # Left X is an immediate stop request, not a training frame: stop motion
            # before recorder shutdown, and do not append the exit/stop state.
            driver.stop_all_motion()
            print("\n[wbc_vr_robot] leader requested exit; motion stopped.")
            break

        estop = bool(vr.estop) if vr is not None else True
        recorder = getattr(driver, "_episode", None)
        recording_active = recorder is not None and bool(
            getattr(recorder, "recording", False)
        )
        if estop and hasattr(driver, "stop_recording_episode"):
            if recording_active:
                # Streaming close can wait for queued HDF5 writes. Zero/freeze hardware
                # before that wait so an e-stop never leaves the preceding base command
                # latched while storage drains.
                if hasattr(driver, "stop_all_motion"):
                    driver.stop_all_motion()
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
            last_cmd_receive_wall_ns = int(vr_receive_wall_ns)
            last_cmd_wall = now

        left_target, right_target, head_target = interp.at(now)
        head_targets_pre_filtered = bool(
            getattr(source, "head_targets_pre_filtered", replay)
        )
        if not head_targets_pre_filtered:
            # Episode replay re-issues action/head after these filters, while raw leader
            # stream replay does not. Source provenance -- not the broad ``replay`` flag --
            # decides whether filtering here would be a duplicate.
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
                source_timestamp_ns=int(last_cmd_ns),
                source_receive_wall_ns=int(last_cmd_receive_wall_ns),
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
    ik, cfg = _build_ik(lock_torso_in_ik=args.lock_torso_in_ik)
    print(f"[wbc_vr_robot] model nq={ik.model.nq} nv={ik.model.nv} head_mode={cfg.head_mode} "
          f"base_dofs={cfg.base_dofs} lock_torso_in_ik={cfg.lock_torso_in_ik} "
          f"urdf={cfg.urdf_path.rsplit('/', 1)[-1]}")
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
            "urdf": cfg.urdf_path.rsplit("/", 1)[-1],
            "base_dofs": cfg.base_dofs,
            "lock_torso_in_ik": bool(cfg.lock_torso_in_ik),
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
    print(f"[wbc_vr_robot] REAL ROBOT. enabled={enabled} grippers=on | "
          f"{'REPLAY ' + format(args.speed, 'g') + 'x' if replay else 'LIVE'} | "
          f"base_max={args.base_max_speed:g}m/s | base_dofs={cfg.base_dofs} | "
          f"torso_ik={'fixed' if cfg.lock_torso_in_ik else 'free'} | "
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
        head_streams = (
            "head_left_rgb" if args.no_head_depth else "head_left_rgb+head_depth"
        )
        if args.head_right_rgb:
            head_streams += "+head_right_rgb"
        print(f"  recording -> {args.save_dir} @ {args.record_rate:g}Hz while engaged "
              f"({head_streams}+left_wrist_rgb+right_wrist_rgb, "
              f"{'streaming' if args.streaming_recorder else 'buffered'}, +torso action, "
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
        if driver is not None:
            try:
                driver.abort_recording_episode(str(exc))
            except Exception as quarantine_exc:
                print(
                    "[wbc_vr_robot] WARNING: failed to quarantine the aborted "
                    f"episode: {quarantine_exc}"
                )
                # Never let finally->driver.close() run the normal complete-save path
                # after quarantine itself failed. Losing forensic rows is preferable to
                # publishing a runtime-failed take as valid training data.
                episode = getattr(driver, "_episode", None)
                if episode is not None and getattr(episode, "recording", False):
                    try:
                        episode.discard()
                    except Exception as discard_exc:
                        print(
                            "[wbc_vr_robot] WARNING: failed to discard the "
                            f"unquarantined take: {discard_exc}"
                        )
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
                           "follower/wbik.yaml. Only the take's original solver inputs are "
                           "consumed (action/eef/{left,right}, action/head, "
                           "action/gripper/*); everything else is re-solved with the "
                           "current follower/wbik.yaml, so the same take can be compared "
                           "across parameter edits. The recorded head target is fed to the "
                           "IK unfiltered (record-time head LPF/deadband already applied).")
    mode.add_argument("--source", choices=("live",), default=None,
                      help="drive from the live leader (default when --replay is omitted).")
    parser.add_argument("--enable", default="arms,torso,head,base",
                        help="comma list of DOF groups to actuate: any of "
                             "arms,torso,head,base (or 'all'/'none'). Default "
                             "'arms,torso,head,base'. Unselected groups are still solved by IK "
                             "but not sent to that actuator. The grippers are always "
                             "active (tracking the leader triggers), independent of this mask.")
    parser.add_argument("--lock-torso-in-ik", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="override follower/wbik.yaml lock_torso_in_ik for this run. "
                             "When enabled, hard-pin torso_j1/j2/j3 inside the IK QP. The "
                             "torso stays at the solver reset posture (nominal after "
                             "engage/home) while the other DOFs solve the targets. This "
                             "differs from omitting torso from --enable, which only "
                             "suppresses hardware commands. Omit to use the YAML value.")
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
                             "selected head RGB/depth streams and {left,right}_wrist_rgb "
                             "(left/right = ARM), a static intrinsic, "
                             "per-frame capture stamps + a once-per-run meta/ntp clock offset "
                             "(latency audits), and (with --enable base) the obs/base/pose "
                             "used to rebuild world_t_cam offline. Each wrist camera needs "
                             "its own ZED-SDK publisher on sensors/{left,right}_wrist_zedm/*. "
                             "OFF by default.")
    parser.add_argument("--streaming-recorder", action="store_true",
                        help="append frames to HDF5 through a bounded writer queue instead "
                             "of buffering the whole take in RAM. Produces the same final "
                             "datasets; an interrupted hard crash leaves a .partial file. "
                             "Requires --record.")
    parser.add_argument("--no-head-depth", action="store_true",
                        help="do not poll or store ZED SDK head depth. Pair with the head "
                             "publisher's --no-enable-depth to avoid depth compute/traffic. "
                             "Use --head-right-rgb when depth will be reconstructed offline. "
                             "Requires --record.")
    parser.add_argument("--head-right-rgb", action="store_true",
                        help="also record the rectified head right eye and timestamp, plus "
                             "the live stereo baseline/intrinsics for offline FoundationStereo. "
                             "Requires a publisher started with --no-skip-right-rgb and "
                             "requires --record.")
    parser.add_argument("--wrist-right-rgb", action="store_true",
                        help="also record each wrist camera's rectified right eye and "
                             "timestamp, plus each live stereo baseline/intrinsics for "
                             "offline FoundationStereo. Existing {left,right}_wrist_rgb "
                             "remain the LEFT eyes. Requires both wrist publishers started "
                             "with --no-skip-right-rgb and requires --record.")
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
    parser.add_argument(
        "--record-startup-timeout",
        type=float,
        default=DEFAULT_RECORD_STARTUP_TIMEOUT,
        help="seconds allowed before frame zero for multiple advancing frames from "
             "the head and both wrists plus one capture-aligned set (default "
             f"{DEFAULT_RECORD_STARTUP_TIMEOUT:g}). This does not relax the shorter "
             "--record-stale-grace after recording begins.",
    )
    parser.add_argument(
        "--record-max-camera-skew-ms",
        type=float,
        default=DEFAULT_RECORD_MAX_CAMERA_SKEW_MS,
        help="maximum absolute capture-time skew between the head and EACH wrist in "
             "a saved row, after per-camera clock correction (default "
             f"{DEFAULT_RECORD_MAX_CAMERA_SKEW_MS:g} ms). Camera subscriber caches are "
             "sampled at the IK rate; the freshest buffered head anchor whose closest "
             "unused wrist frames pass is selected. A row is retried then aborted when "
             "no set satisfies the limit. 0 restores legacy latest-arrived pairing.",
    )
    parser.add_argument(
        "--record-max-camera-age-ms",
        type=float,
        default=DEFAULT_RECORD_MAX_CAMERA_AGE_MS,
        help="maximum camera capture-to-selection age after publisher-clock "
             f"correction (default {DEFAULT_RECORD_MAX_CAMERA_AGE_MS:g} ms). "
             "Fresh but backlogged streams are retried then abort the take instead "
             "of silently pairing old images with current robot state/action.",
    )

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
                      ("--record-stale-grace", args.record_stale_grace),
                      ("--record-max-camera-skew-ms", args.record_max_camera_skew_ms)):
        if not np.isfinite(val) or val < 0.0:
            parser.error(f"{flag} must be finite and >= 0")
    if (
        not np.isfinite(args.record_max_camera_age_ms)
        or args.record_max_camera_age_ms <= 0.0
    ):
        parser.error("--record-max-camera-age-ms must be finite and > 0")
    if (
        not np.isfinite(args.record_startup_timeout)
        or args.record_startup_timeout <= 0.0
    ):
        parser.error("--record-startup-timeout must be finite and > 0")
    if not np.isfinite(args.base_quiet_hold_s):
        parser.error("--base-quiet-hold-s must be finite")
    for flag, val in (("--source-timeout", args.source_timeout),
                      ("--record-rate", args.record_rate)):
        if not np.isfinite(val) or val <= 0.0:
            parser.error(f"{flag} must be finite and > 0")
    for flag, enabled in (
        ("--streaming-recorder", args.streaming_recorder),
        ("--no-head-depth", args.no_head_depth),
        ("--head-right-rgb", args.head_right_rgb),
        ("--wrist-right-rgb", args.wrist_right_rgb),
    ):
        if enabled and not args.record:
            parser.error(f"{flag} only affects recording; pass it with --record")
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
