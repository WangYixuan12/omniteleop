#!/usr/bin/env python3
"""Drive the Vega chassis through a closed box + full turn, record at 10 Hz, and
log chassis control accuracy + latency against swerve wheel odometry.

Trajectory (the user's script, wrapped with recording + metrics)::

    move_straight(+0.4, 2 s)   # +x  ~0.8 m forward
    move_sideways(-0.4, 4 s)   # -y  ~1.6 m right
    move_straight(-0.4, 2 s)   # -x  ~0.8 m back
    move_sideways(+0.4, 4 s)   # +y  ~1.6 m left      -> translation closes to origin
    turn(+0.6 rad/s, 360 deg)  # one full revolution  -> heading closes to origin
    stop()

Every leg goes through ``chassis.move_*/turn`` -> ``set_velocity(...,
sequential_steering=True)``, so each leg first re-steers the wheels in place
(~``steering_wait_time``) and only then drives for ``wait_time``. That built-in
re-steer is part of the command latency these helpers impose and is measured
explicitly below (command -> steer onset vs command -> drive onset).

The "actual base pose" reference logged against the commanded twist is swerve
wheel odometry (``omniteleop.follower.swerve_odometry``, drive_state_mode='ms',
the value verified for this robot), integrated from identity at the first
recorded sample.

Head left_rgb + depth from the head ZED publisher (``tests/test_head_zedx_depth.py``,
vision only) are recorded alongside for context.

Run, with the robot reachable and the dexmate env active::

    # measure idle ceiling + rates achieved under motion (this DRIVES the trajectory):
    python scripts/drive_box_record.py --probe-rate --no-record-images

    # head camera publisher (vision: left_rgb + depth), in a separate terminal
    python tests/test_head_zedx_depth.py

    python scripts/drive_box_record.py \
        --output ~/yixuan/Dexmate/SLAM/drive_box.hdf5

SAFETY: the base physically drives a ~0.8 m x ~1.6 m box then spins a full turn
in place — clear ~1.5 m radius and keep a hand on the e-stop. ``--dry-run`` records and
verifies all wiring WITHOUT moving the base (issues no chassis commands).
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import sys
import threading
import time
from typing import Optional

import numpy as np
import tyro
from loguru import logger

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, (_REPO / "src").as_posix())

from omniteleop.common.head_camera import ZED_K  # noqa: E402  (shared head intrinsic)
from omniteleop.follower import base_closed_loop as base_cl  # noqa: E402
from omniteleop.follower.swerve_odometry import (  # noqa: E402  (pure-numpy)
    VEGA1_MAX_LINEAR_VEL,
    VEGA1_MAX_STEERING_ANGLE,
    OdometryInputError,
    SwerveOdometry,
)


@dataclasses.dataclass
class Args:
    output: str = "/home/dexmate/yixuan/Dexmate/SLAM/box_test/drive_box.hdf5"
    """Output HDF5 (10 Hz recording). Metrics JSON is written next to it."""

    speed: float = 0.2
    """Linear speed (m/s) for every straight/sideways leg (the user's 0.4)."""

    leg_time: float = 2.0
    """Base drive time (s): x legs use this, y legs currently use 2x this."""

    turn_speed: float = 0.6
    """Final in-place turn rate (rad/s). Sign controls turn direction."""

    turn_revolutions: float = 1.0
    """Absolute revolutions for the final in-place turn. 1.0 returns heading to origin."""

    settle: float = 0.5
    """Stop + dwell (s) inserted between legs so each leg settles and the
    per-leg metrics separate cleanly. Set 0 for the back-to-back original."""

    record_rate: float = 10.0
    """HDF5 sampling rate (Hz). The user's 'record all data at 10 Hz'."""

    odom_rate: float = 100.0
    """Swerve-odometry integration rate (Hz). Higher than record_rate so the
    integrated pose does not under-sample the steering transients."""

    sensor_id: str = "head_camera"
    """Head ZED publisher id: sensors/<id>/{left_rgb,depth}."""

    record_images: bool = True
    """Record head left_rgb + depth at record_rate (the chosen scope)."""

    drive_state_mode: str = "ms"
    """'ms' (wheel_velocity is signed m/s, verified for this robot) or 'rad'."""

    startup_timeout_s: float = 15.0
    """Fail fast if a required stream is silent this long at startup."""

    dry_run: bool = False
    """Record + verify wiring but issue NO chassis motion (base stays parked)."""

    probe_rate: bool = False
    """Measure the idle chassis state publish-rate ceiling (and print recommended
    --odom-rate / --closed-loop-rate), THEN drive the normal trajectory and report
    the rates actually ACHIEVED under motion. This DRIVES the base and records like
    a normal run (pass --no-record-images to skip the head publisher). Note: every
    run already reports achieved rates; --probe-rate just adds the idle ceiling."""

    probe_duration_s: float = 3.0
    """Sampling window (s) for --probe-rate."""

    probe_poll_hz: float = 1000.0
    """Polling rate (Hz) for sampling firmware timestamps during --probe-rate."""

    closed_loop_source: str = "none"
    """Feedback source for optional pose-PD base control: 'none' or 'odom'.
    'none' preserves the original fixed-duration open-loop command sequence."""

    closed_loop_rate: float = 100.0
    """Closed-loop command rate (Hz) when --closed-loop-source is odom."""

    closed_loop_kp_xy: float = 1.0
    """Conservative proportional gain for planar pose error (1/s)."""

    closed_loop_kp_yaw: float = 1.5
    """Conservative proportional gain for yaw error (1/s)."""

    closed_loop_pos_tolerance: float = 0.02
    """Final planar pose tolerance (m) required before a closed-loop leg completes."""

    closed_loop_yaw_tolerance: float = 0.05
    """Final yaw tolerance (rad) required before a closed-loop leg completes."""

    closed_loop_stable_time: float = 0.25
    """The final pose error must remain within tolerance for this many seconds."""

    closed_loop_source_timeout_s: float = 0.5
    """Stop and fail if the selected feedback source is older than this."""

    closed_loop_timeout_scale: float = 3.0
    """Per-leg timeout multiplier applied to the nominal leg drive time."""

    closed_loop_timeout_margin_s: float = 1.0
    """Extra per-leg timeout margin (s) added after the scaled nominal time."""

    closed_loop_max_lin_speed: Optional[float] = None
    """Closed-loop translational speed clamp (m/s). None uses abs(--speed)."""

    closed_loop_max_ang_speed: Optional[float] = None
    """Closed-loop angular speed clamp (rad/s). None uses abs(--turn-speed)."""

    closed_loop_max_lin_accel: float = 0.4
    """Closed-loop translational acceleration/slew clamp (m/s^2)."""

    closed_loop_max_ang_accel: float = 0.8
    """Closed-loop angular acceleration/slew clamp (rad/s^2)."""

    closed_loop_deadband_lin: float = 0.01
    """Closed-loop linear command deadband (m/s) per translation axis."""

    closed_loop_deadband_ang: float = 0.02
    """Closed-loop angular command deadband (rad/s)."""


def _closed_loop_config(args: Args) -> base_cl.BaseClosedLoopConfig:
    return base_cl.BaseClosedLoopConfig(
        source=args.closed_loop_source,
        rate=args.closed_loop_rate,
        kp_xy=args.closed_loop_kp_xy,
        kp_yaw=args.closed_loop_kp_yaw,
        pos_tolerance=args.closed_loop_pos_tolerance,
        yaw_tolerance=args.closed_loop_yaw_tolerance,
        stable_time=args.closed_loop_stable_time,
        source_timeout_s=args.closed_loop_source_timeout_s,
        timeout_scale=args.closed_loop_timeout_scale,
        timeout_margin_s=args.closed_loop_timeout_margin_s,
        max_lin_speed=args.closed_loop_max_lin_speed,
        max_ang_speed=args.closed_loop_max_ang_speed,
        max_lin_accel=args.closed_loop_max_lin_accel,
        max_ang_accel=args.closed_loop_max_ang_accel,
        deadband_lin=args.closed_loop_deadband_lin,
        deadband_ang=args.closed_loop_deadband_ang,
    )


def _configure_zenoh_like_dexcontrol() -> Optional[pathlib.Path]:
    """Mirror dexcontrol's ZENOH_CONFIG discovery (see the head publisher)."""
    existing = os.getenv("ZENOH_CONFIG")
    if existing:
        return pathlib.Path(existing).expanduser()
    base_dir = pathlib.Path("~/.dexmate/comm/zenoh").expanduser()
    for pattern in ("**/*.dzcfg", "**/zenoh*config*.json5"):
        matches = sorted(p for p in base_dir.glob(pattern) if p.is_file())
        if matches:
            os.environ["ZENOH_CONFIG"] = matches[0].as_posix()
            return matches[0]
    return None


def _validate_args(args: Args) -> None:
    if args.drive_state_mode not in ("ms", "rad"):
        raise ValueError(f"--drive-state-mode must be 'ms' or 'rad', got {args.drive_state_mode!r}")
    if abs(args.turn_speed) <= 1e-9:
        raise ValueError("--turn-speed must be non-zero for the final turn")
    positive = {
        "record_rate": args.record_rate,
        "odom_rate": args.odom_rate,
        "probe_duration_s": args.probe_duration_s,
        "probe_poll_hz": args.probe_poll_hz,
    }
    for name, value in positive.items():
        if value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0, got {value}")
    base_cl.validate_config(_closed_loop_config(args), args.speed, args.turn_speed)


@dataclasses.dataclass
class Leg:
    """One commanded segment: a constant target twist held for ``drive_time``."""
    label: str
    vx: float
    vy: float
    wz: float
    drive_time: float
    # filled at execution time
    t_issue: float = float("nan")
    t_done: float = float("nan")
    completion_reason: str = "not_started"
    abort_reason: str = ""

    @property
    def target_speed(self) -> float:
        return float(np.hypot(np.hypot(self.vx, self.vy), 0.0)) or abs(self.wz)

    def expected_disp(self) -> tuple[float, float, float]:
        """Expected (dx, dy, dyaw) in the base frame over the drive window."""
        return self.vx * self.drive_time, self.vy * self.drive_time, self.wz * self.drive_time


def build_legs(args: Args) -> list[Leg]:
    """Box trajectory plus a final in-place revolution."""
    s, t = args.speed, args.leg_time
    turn_time = 2.0 * np.pi * abs(args.turn_revolutions) / abs(args.turn_speed)
    return [
       Leg(f"turn {360 * abs(args.turn_revolutions):.0f}deg", 0.0, 0.0, args.turn_speed, turn_time),
    ]
        #     Leg("forward +x", +s, 0.0, 0.0, t),
        # Leg("right -y", 0.0, -s, 0.0, 2*t),
        # Leg("back -x", -s, 0.0, 0.0, t),
        # Leg("left +y", 0.0, +s, 0.0, 2*t),
        # Leg(f"turn {360 * abs(args.turn_revolutions):.0f}deg", 0.0, 0.0, args.turn_speed, turn_time),

class SharedState:
    """Thread-shared command + latest-pose state, lock-guarded."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cmd = np.zeros(3)         # active commanded twist (vx, vy, wz)
        self.cmd_label = "idle"
        self.odom_pose = np.zeros(3)   # (x, y, yaw) integrated
        self.odom_twist = np.zeros(3)
        self.odom_update_t = float("nan")
        self.odom_updates = 0
        self.cl_ref_pose = np.full(3, np.nan)
        self.cl_feedback_pose = np.full(3, np.nan)
        self.cl_pose_error = np.full(3, np.nan)
        self.stop_flag = False


def odom_thread(robot, shared: SharedState, args: Args) -> None:
    """Integrate swerve odometry from live wheel state, gated on the firmware
    sample timestamp.

    The loop polls at ``odom_rate``, but integration is driven by the drive
    component's ``get_timestamp_ns()`` rather than wall-clock loop time: each
    distinct firmware sample is integrated once over its own measured ``dt``
    (see ``SwerveOdometry.update_at``). Polling faster than the chassis publishes
    therefore re-reads the same cached sample without double-integrating, and a
    stalled stream stops advancing the pose -- and stops refreshing
    ``odom_update_t``, so the closed-loop freshness check trips instead of
    dead-reckoning a stale velocity. The recorder samples the integrated pose at
    the (slower) record rate."""
    # Validity bounds track the robot's actual operating limits, read live from
    # the chassis config (SwerveOdometry scales them by its margin for headroom).
    chassis = robot.chassis
    max_steering_angle = float(getattr(chassis, "_max_steering_angle", VEGA1_MAX_STEERING_ANGLE))
    max_linear_vel = float(getattr(chassis, "max_lin_vel", VEGA1_MAX_LINEAR_VEL))
    odo = SwerveOdometry(
        drive_state_mode=args.drive_state_mode,
        max_steering_angle=max_steering_angle,
        max_linear_vel=max_linear_vel,
    )
    logger.info(
        f"odom validity bounds from robot limits "
        f"(steer {max_steering_angle:.2f} rad, lin {max_linear_vel:.2f} m/s): "
        f"|steer|<={odo.max_steer_abs:.2f} rad, |drive|<={odo.max_drive_abs:.2f} "
        f"{'m/s' if args.drive_state_mode == 'ms' else 'rad/s'}")
    period = 1.0 / args.odom_rate
    prev_ts: Optional[int] = None
    n_bad = 0
    last_bad_warn = float("-inf")
    while True:
        now = time.perf_counter()
        with shared.lock:
            stop = shared.stop_flag
        if stop:
            break
        try:
            steer = np.asarray(robot.chassis.steering_angle, dtype=np.float64)
            wvel = np.asarray(robot.chassis.wheel_velocity, dtype=np.float64)
            # Firmware timestamp of the drive sample drives integration; the steer
            # angle rides along (its transients are already down-weighted in the WLS).
            ts = int(robot.chassis.chassis_drive.get_timestamp_ns())
            if (
                steer.shape == (2,)
                and wvel.shape == (2,)
                and (prev_ts is None or ts > prev_ts)
            ):
                try:
                    pose = odo.update_at(steer, wvel, ts)
                except OdometryInputError as exc:
                    # Invalid wheel sample: skip it (hold the last good pose; let
                    # odom_update_t go stale so the closed-loop freshness check
                    # trips) -- but NEVER silently: warn on the first one, then at
                    # most ~1 Hz with a running count.
                    n_bad += 1
                    if now - last_bad_warn >= 1.0:
                        logger.warning(
                            f"odom: skipped {n_bad} invalid wheel sample(s); latest: {exc}")
                        last_bad_warn = now
                else:
                    prev_ts = ts
                    with shared.lock:
                        shared.odom_pose = pose.copy()
                        shared.odom_twist = odo.twist.copy()
                        shared.odom_update_t = now  # wall clock: consumed by fresh_age_s
                        shared.odom_updates += 1
        except Exception:
            logger.exception("odom thread stopped on error")
            break
        sleep = period - (time.perf_counter() - now)
        if sleep > 0:
            time.sleep(sleep)
    if n_bad:
        logger.warning(f"odom thread exit: {n_bad} invalid wheel sample(s) skipped total")


def _round_to(x: float, base: float = 10.0) -> float:
    """Round ``x`` to the nearest ``base``, floored at ``base``."""
    return float(max(base, round(x / base) * base))


def probe_state_rates(robot, duration_s: float, poll_hz: float) -> dict:
    """Sample chassis drive+steer firmware timestamps to estimate publish rates.

    Polls ``get_timestamp_ns()`` much faster than the producer publishes, collects
    the distinct firmware timestamps, and derives publish-interval statistics. The
    odometry-integration and PD-feedback paths cannot run faster than these, so
    they set the achievable ceiling for --odom-rate and --closed-loop-rate. The
    base is not commanded, so this is motion-free."""
    period = 1.0 / poll_hz
    drive_ts: list[int] = []
    steer_ts: list[int] = []
    drive_recv: list[float] = []  # wall-clock receipt of each NEW drive sample
    last_drive: Optional[int] = None
    t_end = time.perf_counter() + duration_s
    while time.perf_counter() < t_end:
        tick = time.perf_counter()
        try:
            d = int(robot.chassis.chassis_drive.get_timestamp_ns())
            s = int(robot.chassis.chassis_steer.get_timestamp_ns())
        except Exception:
            continue
        drive_ts.append(d)
        steer_ts.append(s)
        if d != last_drive:
            drive_recv.append(tick)
            last_drive = d
        sleep = period - (time.perf_counter() - tick)
        if sleep > 0:
            time.sleep(sleep)

    def _ts_stats(ts_list: list[int]) -> Optional[dict]:
        u = np.unique(np.asarray(ts_list, dtype=np.int64))
        if u.size < 3:
            return None
        dt = np.diff(u) * 1e-9  # firmware dt between distinct samples (s)
        dt = dt[dt > 0]
        if dt.size == 0:
            return None
        return dict(
            n_unique=int(u.size),
            hz_median=float(1.0 / np.median(dt)),
            dt_ms_median=float(np.median(dt) * 1e3),
            dt_ms_p95=float(np.percentile(dt, 95) * 1e3),
            dt_ms_max=float(np.max(dt) * 1e3),
        )

    recv = None
    if len(drive_recv) >= 3:
        rdt = np.diff(np.asarray(drive_recv, dtype=np.float64))
        recv = dict(
            hz_median=float(1.0 / np.median(rdt)),
            jitter_ms_p95=float((np.percentile(rdt, 95) - np.median(rdt)) * 1e3),
        )
    return dict(
        polls=len(drive_ts),
        drive=_ts_stats(drive_ts),
        steer=_ts_stats(steer_ts),
        drive_recv=recv,
    )


def _log_rate_probe(report: dict) -> None:
    """Print measured publish rates and recommended --odom-rate / --closed-loop-rate."""
    d, s = report["drive"], report["steer"]
    logger.info(f"[probe] polled {report['polls']} times")
    if d is None or s is None:
        logger.warning("[probe] too few distinct samples — is chassis state streaming?")
        return
    logger.info(
        f"[probe] drive state ~{d['hz_median']:5.0f} Hz  "
        f"(dt med {d['dt_ms_median']:.1f} / p95 {d['dt_ms_p95']:.1f} / max {d['dt_ms_max']:.1f} ms, "
        f"{d['n_unique']} uniq)")
    logger.info(
        f"[probe] steer state ~{s['hz_median']:5.0f} Hz  "
        f"(dt med {s['dt_ms_median']:.1f} / p95 {s['dt_ms_p95']:.1f} / max {s['dt_ms_max']:.1f} ms, "
        f"{s['n_unique']} uniq)")
    if report.get("drive_recv"):
        r = report["drive_recv"]
        logger.info(f"[probe] drive receive cadence ~{r['hz_median']:.0f} Hz "
                    f"(p95 jitter {r['jitter_ms_p95']:.1f} ms)")
    base = d["hz_median"]
    logger.info(
        f"[probe] RECOMMEND --odom-rate {_round_to(base * 1.5):.0f}  "
        f"(poll ~1.5x headroom; effective integration self-caps at the ~{base:.0f} Hz "
        f"drive publish rate)")
    logger.info(
        f"[probe] RECOMMEND --closed-loop-rate {_round_to(base):.0f}  "
        f"(= feedback refresh ceiling; a faster PD sees no new odom and only wastes CPU)")
    if d["dt_ms_max"] > 3.0 * d["dt_ms_median"]:
        logger.warning(
            f"[probe] drive stream has gaps (max dt {d['dt_ms_max']:.0f} ms ≈ "
            f"{d['dt_ms_max'] / d['dt_ms_median']:.0f}x median); keep SwerveOdometry "
            f"max_dt (default 100 ms) above this")


def main() -> None:
    args = tyro.cli(Args)
    _validate_args(args)
    cl_config = _closed_loop_config(args)
    closed_loop = cl_config.enabled
    out = pathlib.Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg = _configure_zenoh_like_dexcontrol()
    logger.info(f"Zenoh config: {cfg if cfg else '<dexcomm default>'}; "
                f"ROBOT_NAME={os.getenv('ROBOT_NAME')!r}")

    # ── connect to robot + subscribe to head frames ──────────────────────────
    from dexcomm import Node
    from dexcomm.codecs import DepthImageCodec, RGBImageCodec
    from dexcontrol.robot import Robot

    robot = Robot()
    node = Node(name="drive_box_record", namespace="")
    rgb_sub = depth_sub = None
    if args.record_images:
        rgb_sub = node.create_subscriber(
            f"sensors/{args.sensor_id}/left_rgb", decoder=RGBImageCodec.decode)
        depth_sub = node.create_subscriber(
            f"sensors/{args.sensor_id}/depth", decoder=DepthImageCodec.decode)

    # ── fail fast if any required stream is silent ────────────────────────────
    def _wait(getter, what: str):
        deadline = time.perf_counter() + args.startup_timeout_s
        while time.perf_counter() < deadline:
            v = getter()
            if v is not None:
                return v
            time.sleep(0.05)
        raise SystemExit(
            f"No {what} within {args.startup_timeout_s:.0f}s. Is the producer up? "
            "Need: chassis state (robot) and, with --record-images, the head "
            "publisher (tests/test_head_zedx_depth.py).")

    logger.info("Waiting for chassis state and head frames…")
    _ = _wait(lambda: robot.chassis.steering_angle if
              np.asarray(robot.chassis.steering_angle).shape == (2,) else None, "chassis state")

    probe_report: Optional[dict] = None
    if args.probe_rate:
        logger.info(f"Probing idle chassis state publish rates for {args.probe_duration_s:.0f}s "
                    f"(poll {args.probe_poll_hz:.0f} Hz; base parked)…")
        probe_report = probe_state_rates(robot, args.probe_duration_s, args.probe_poll_hz)
        _log_rate_probe(probe_report)
        logger.info("Now driving the trajectory to measure ACHIEVED rates under motion…")

    if args.record_images:
        _ = _wait(rgb_sub.get_latest, f"sensors/{args.sensor_id}/left_rgb")
        _ = _wait(depth_sub.get_latest, f"sensors/{args.sensor_id}/depth")
    head0 = np.asarray(robot.head.get_joint_pos(), dtype=np.float64)
    torso0 = np.asarray(robot.torso.get_joint_pos(), dtype=np.float64)
    if head0.shape != (3,) or torso0.shape != (3,):
        raise SystemExit(
            f"expected 3 head + 3 torso joints, got head {head0.shape}, torso {torso0.shape}")
    logger.info("All streams live.")

    legs = build_legs(args)
    total_drive = sum(l.drive_time for l in legs) + args.settle * len(legs)
    logger.info(f"Plan: {len(legs)} legs, ~{total_drive:.0f}s drive"
                + (f"  [closed-loop: {args.closed_loop_source}]" if closed_loop else "")
                + ("  [DRY RUN — base will NOT move]" if args.dry_run else ""))
    for l in legs:
        dx, dy, dyaw = l.expected_disp()
        logger.info(f"  {l.label:12s} cmd=({l.vx:+.2f},{l.vy:+.2f},{l.wz:+.2f}) "
                    f"{l.drive_time:4.1f}s  expect d=({dx:+.2f},{dy:+.2f}, {np.degrees(dyaw):+.0f}deg)")

    # ── start odometry + recorder threads ─────────────────────────────────────
    shared = SharedState()
    rec: dict[str, list] = {k: [] for k in (
        "t", "unix_ns", "cmd_vx", "cmd_vy", "cmd_wz", "cmd_label",
        "steer_l", "steer_r", "wvel_l", "wvel_r", "enc_l", "enc_r",
        "steer_ts_ns", "drive_ts_ns",
        "odom_x", "odom_y", "odom_yaw", "odom_vx", "odom_vy", "odom_w",
        "cl_ref_x", "cl_ref_y", "cl_ref_yaw",
        "cl_feedback_x", "cl_feedback_y", "cl_feedback_yaw",
        "cl_err_x", "cl_err_y", "cl_err_yaw",
        "head_j1", "head_j2", "head_j3", "torso_j1", "torso_j2", "torso_j3",
        "head_rgb", "head_depth")}

    t_epoch = time.perf_counter()  # single clock origin shared by rec["t"] AND leg windows
    odo_t = threading.Thread(target=odom_thread, args=(robot, shared, args), daemon=True)
    odo_t.start()

    rec_stop = threading.Event()

    def recorder() -> None:
        period = 1.0 / args.record_rate
        t0 = t_epoch
        while not rec_stop.is_set():
            tick = time.perf_counter()
            with shared.lock:
                cmd = shared.cmd.copy()
                cmd_label = shared.cmd_label
                odom_pose = shared.odom_pose.copy()
                odom_twist = shared.odom_twist.copy()
                cl_ref = shared.cl_ref_pose.copy()
                cl_feedback = shared.cl_feedback_pose.copy()
                cl_err = shared.cl_pose_error.copy()
            steer = np.asarray(robot.chassis.steering_angle, dtype=np.float64)
            wvel = np.asarray(robot.chassis.wheel_velocity, dtype=np.float64)
            enc = np.asarray(robot.chassis.wheel_encoder_pos, dtype=np.float64)
            head = np.asarray(robot.head.get_joint_pos(), dtype=np.float64)
            torso = np.asarray(robot.torso.get_joint_pos(), dtype=np.float64)

            rec["t"].append(tick - t0)
            rec["unix_ns"].append(time.time_ns())
            rec["cmd_vx"].append(cmd[0]); rec["cmd_vy"].append(cmd[1]); rec["cmd_wz"].append(cmd[2])
            rec["cmd_label"].append(cmd_label)
            rec["steer_l"].append(steer[0]); rec["steer_r"].append(steer[1])
            rec["wvel_l"].append(wvel[0]); rec["wvel_r"].append(wvel[1])
            rec["enc_l"].append(enc[0]); rec["enc_r"].append(enc[1])
            rec["steer_ts_ns"].append(int(robot.chassis.chassis_steer.get_timestamp_ns()))
            rec["drive_ts_ns"].append(int(robot.chassis.chassis_drive.get_timestamp_ns()))
            rec["odom_x"].append(odom_pose[0]); rec["odom_y"].append(odom_pose[1]); rec["odom_yaw"].append(odom_pose[2])
            rec["odom_vx"].append(odom_twist[0]); rec["odom_vy"].append(odom_twist[1]); rec["odom_w"].append(odom_twist[2])
            rec["cl_ref_x"].append(cl_ref[0]); rec["cl_ref_y"].append(cl_ref[1]); rec["cl_ref_yaw"].append(cl_ref[2])
            rec["cl_feedback_x"].append(cl_feedback[0]); rec["cl_feedback_y"].append(cl_feedback[1])
            rec["cl_feedback_yaw"].append(cl_feedback[2])
            rec["cl_err_x"].append(cl_err[0]); rec["cl_err_y"].append(cl_err[1]); rec["cl_err_yaw"].append(cl_err[2])
            for i, v in enumerate(head):
                rec[f"head_j{i+1}"].append(float(v))
            for i, v in enumerate(torso):
                rec[f"torso_j{i+1}"].append(float(v))
            if args.record_images:
                rmsg = rgb_sub.get_latest(); dmsg = depth_sub.get_latest()
                rec["head_rgb"].append(np.ascontiguousarray(rmsg["data"]) if rmsg else None)
                rec["head_depth"].append(np.ascontiguousarray(dmsg["data"]) if dmsg else None)
            sleep = period - (time.perf_counter() - tick)
            if sleep > 0:
                time.sleep(sleep)

    rec_t = threading.Thread(target=recorder, daemon=True)
    rec_t.start()
    time.sleep(0.5)  # a few parked samples before motion → clean latency baseline
    if closed_loop and args.closed_loop_source == "odom":
        _wait(
            lambda: True if np.isfinite(shared.odom_update_t) else None,
            "fresh wheel odometry for closed-loop source=odom",
        )

    # ── drive the sequence (main thread) ──────────────────────────────────────
    t_run0 = t_epoch  # leg windows share rec["t"]'s origin so compute_metrics aligns
    nan3 = np.full(3, np.nan)
    max_lin_speed, max_ang_speed = base_cl.closed_loop_limits(cl_config, args.speed, args.turn_speed)
    cl_stats = {"iters": 0, "secs": 0.0}  # achieved closed-loop PD command rate

    def set_cmd(
        vx: float,
        vy: float,
        wz: float,
        label: str,
        ref_pose: Optional[np.ndarray] = None,
        feedback_pose: Optional[np.ndarray] = None,
        pose_error: Optional[np.ndarray] = None,
    ) -> None:
        with shared.lock:
            shared.cmd = np.array([vx, vy, wz])
            shared.cmd_label = label
            shared.cl_ref_pose = nan3.copy() if ref_pose is None else np.asarray(ref_pose, dtype=np.float64).copy()
            shared.cl_feedback_pose = (
                nan3.copy() if feedback_pose is None else np.asarray(feedback_pose, dtype=np.float64).copy()
            )
            shared.cl_pose_error = (
                nan3.copy() if pose_error is None else np.asarray(pose_error, dtype=np.float64).copy()
            )

    def feedback_pose(now: float) -> np.ndarray:
        if args.closed_loop_source == "odom":
            with shared.lock:
                pose = shared.odom_pose.copy()
                update_t = shared.odom_update_t
            base_cl.fresh_age_s(update_t, now, cl_config.source_timeout_s, "odom feedback")
            return pose
        raise RuntimeError("feedback_pose called with closed_loop_source='none'")

    def run_open_loop_leg(leg: Leg) -> None:
        set_cmd(leg.vx, leg.vy, leg.wz, leg.label)
        if not args.dry_run:
            if leg.wz != 0.0:
                robot.chassis.turn(angular_speed=leg.wz, wait_time=leg.drive_time,
                                   center="base_center")
            elif leg.vx != 0.0:
                robot.chassis.move_straight(speed=leg.vx, wait_time=leg.drive_time)
            else:
                robot.chassis.move_sideways(speed=leg.vy, wait_time=leg.drive_time)
            leg.completion_reason = "open_loop_timed"
        else:
            time.sleep(leg.drive_time)
            leg.completion_reason = "dry_run_timed"

    def run_closed_loop_leg(leg: Leg) -> None:
        period = 1.0 / cl_config.rate
        leg_start = time.perf_counter()
        start_pose = feedback_pose(leg_start)
        reference_twist = np.array([leg.vx, leg.vy, leg.wz], dtype=np.float64)
        final_pose = base_cl.reference_pose(start_pose, reference_twist, leg.drive_time, leg.drive_time)
        timeout_s = cl_config.timeout_margin_s + cl_config.timeout_scale * leg.drive_time
        stable_since: Optional[float] = None
        prev_cmd = np.zeros(3, dtype=np.float64)
        prev_tick = leg_start
        logger.info(
            f"  closed-loop target={args.closed_loop_source} final="
            f"({final_pose[0]:+.3f},{final_pose[1]:+.3f},{np.degrees(final_pose[2]):+.1f}deg) "
            f"timeout={timeout_s:.1f}s"
        )
        while True:
            tick = time.perf_counter()
            cl_stats["iters"] += 1
            elapsed = tick - leg_start
            if elapsed > timeout_s:
                raise RuntimeError(
                    f"{leg.label}: closed-loop timeout after {elapsed:.2f}s "
                    f"(nominal {leg.drive_time:.2f}s)"
                )
            measured = feedback_pose(tick)
            reference = base_cl.reference_pose(start_pose, reference_twist, leg.drive_time, elapsed)
            feedforward = (
                reference_twist
                if elapsed < leg.drive_time
                else np.zeros(3, dtype=np.float64)
            )
            raw_cmd, err = base_cl.pd_twist(
                reference,
                feedforward,
                measured,
                kp_xy=cl_config.kp_xy,
                kp_yaw=cl_config.kp_yaw,
                max_lin_speed=max_lin_speed,
                max_ang_speed=max_ang_speed,
            )
            dt = max(1e-6, tick - prev_tick)
            cmd = base_cl.shape_twist(
                raw_cmd,
                prev_cmd,
                dt,
                deadband_lin=cl_config.deadband_lin,
                deadband_ang=cl_config.deadband_ang,
                max_lin_speed=max_lin_speed,
                max_ang_speed=max_ang_speed,
                max_lin_accel=cl_config.max_lin_accel,
                max_ang_accel=cl_config.max_ang_accel,
            )
            prev_cmd = cmd
            prev_tick = tick
            set_cmd(
                float(cmd[0]), float(cmd[1]), float(cmd[2]), leg.label,
                ref_pose=reference, feedback_pose=measured, pose_error=err,
            )
            robot.chassis.set_velocity(
                vx=float(cmd[0]),
                vy=float(cmd[1]),
                wz=float(cmd[2]),
                wait_time=0.0,
                sequential_steering=False,
            )
            if elapsed >= leg.drive_time and base_cl.within_tolerance(
                err, cl_config.pos_tolerance, cl_config.yaw_tolerance
            ):
                if stable_since is None:
                    stable_since = tick
                elif tick - stable_since >= cl_config.stable_time:
                    leg.completion_reason = "closed_loop_tolerance"
                    return
            else:
                stable_since = None
            sleep = period - (time.perf_counter() - tick)
            if sleep > 0:
                time.sleep(sleep)

    abort_reason: Optional[str] = None
    drive_t0 = time.perf_counter()
    with shared.lock:
        odom_updates0 = shared.odom_updates
    try:
        for leg in legs:
            leg.t_issue = time.perf_counter() - t_run0
            logger.info(f"→ {leg.label}")
            if closed_loop and not args.dry_run:
                cl_t0 = time.perf_counter()
                try:
                    run_closed_loop_leg(leg)
                except RuntimeError as exc:
                    leg.abort_reason = str(exc)
                    leg.completion_reason = "aborted"
                    abort_reason = str(exc)
                    logger.error(f"Closed-loop abort: {abort_reason}")
                finally:
                    cl_stats["secs"] += time.perf_counter() - cl_t0
            else:
                run_open_loop_leg(leg)
            leg.t_done = time.perf_counter() - t_run0
            set_cmd(0.0, 0.0, 0.0, "settle")
            if args.settle > 0:
                if not args.dry_run:
                    robot.chassis.stop()
                time.sleep(args.settle)
            if abort_reason is not None:
                break
    except KeyboardInterrupt:
        logger.warning("Interrupted — stopping base.")
    finally:
        if not args.dry_run:
            robot.chassis.stop()
        set_cmd(0.0, 0.0, 0.0, "done")
        time.sleep(0.5)  # capture the stop transient
        rec_stop.set(); rec_t.join(timeout=2.0)
        with shared.lock:
            shared.stop_flag = True
        odo_t.join(timeout=2.0)

    # ── rates actually achieved during the run ────────────────────────────────
    drive_secs = max(1e-9, time.perf_counter() - drive_t0)
    with shared.lock:
        odom_updates_total = shared.odom_updates - odom_updates0
    achieved_odom_hz = odom_updates_total / drive_secs
    achieved_cl_hz = (cl_stats["iters"] / cl_stats["secs"]) if cl_stats["secs"] > 0 else float("nan")
    if closed_loop:
        logger.info(
            f"[rates] ACHIEVED under motion: odom {achieved_odom_hz:.1f} Hz "
            f"(target {args.odom_rate:.0f}), closed-loop {achieved_cl_hz:.1f} Hz "
            f"(target {cl_config.rate:.0f})")
    else:
        logger.info(
            f"[rates] ACHIEVED under motion: odom {achieved_odom_hz:.1f} Hz "
            f"(target {args.odom_rate:.0f}); open-loop legs (no PD loop)")
    if probe_report and probe_report.get("drive"):
        pub = probe_report["drive"]["hz_median"]
        verdict = "at publish ceiling" if achieved_odom_hz >= 0.9 * pub else "below ceiling (poll/CPU-bound)"
        logger.info(f"[rates] idle publish ceiling ~{pub:.0f} Hz → achieved odom is {verdict}")

    # ── persist + metrics ─────────────────────────────────────────────────────
    _write_hdf5(out, rec, args, legs, extra_attrs=dict(
        achieved_odom_hz=float(achieved_odom_hz),
        achieved_closed_loop_hz=float(achieved_cl_hz),
        idle_publish_hz=(float(probe_report["drive"]["hz_median"])
                         if (probe_report and probe_report.get("drive")) else float("nan")),
    ))
    metrics = compute_metrics(rec, legs, args)
    metrics_path = out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    _log_metrics(metrics, legs)
    logger.info(f"HDF5: {out}  ({len(rec['t'])} frames)   metrics: {metrics_path}")

    if not args.dry_run:
        robot.chassis.stop()
    robot.shutdown()
    node.shutdown()
    if abort_reason is not None:
        raise SystemExit(f"closed-loop aborted: {abort_reason}")


def _stack(values: list) -> np.ndarray:
    """Stack a per-frame list into a fixed-shape array, carrying forward the last
    valid frame for any None (e.g. a camera that briefly had no fresh frame).
    Raises if every frame is None (stream never produced data)."""
    first = next((v for v in values if v is not None), None)
    if first is None:
        raise ValueError("stream produced no frames — cannot record (no dummy fill)")
    filled, last = [], first
    for v in values:
        last = v if v is not None else last
        filled.append(last)
    return np.asarray(filled)


def _write_hdf5(out: pathlib.Path, rec: dict, args: Args, legs: list[Leg],
                extra_attrs: Optional[dict] = None) -> None:
    import h5py

    n = len(rec["t"])
    arr = lambda k: np.asarray(rec[k])
    cl_config = _closed_loop_config(args)
    max_lin_speed, max_ang_speed = base_cl.closed_loop_limits(cl_config, args.speed, args.turn_speed)
    with h5py.File(out, "w") as f:
        f.attrs.update(dict(
            record_rate=args.record_rate, odom_rate=args.odom_rate, speed=args.speed,
            leg_time=args.leg_time, turn_speed=args.turn_speed,
            turn_revolutions=args.turn_revolutions, settle=args.settle,
            drive_state_mode=args.drive_state_mode, dry_run=args.dry_run,
            closed_loop_source=cl_config.source,
            closed_loop_rate=cl_config.rate,
            closed_loop_kp_xy=cl_config.kp_xy,
            closed_loop_kp_yaw=cl_config.kp_yaw,
            closed_loop_pos_tolerance=cl_config.pos_tolerance,
            closed_loop_yaw_tolerance=cl_config.yaw_tolerance,
            closed_loop_stable_time=cl_config.stable_time,
            closed_loop_source_timeout_s=cl_config.source_timeout_s,
            closed_loop_timeout_scale=cl_config.timeout_scale,
            closed_loop_timeout_margin_s=cl_config.timeout_margin_s,
            closed_loop_max_lin_speed=max_lin_speed,
            closed_loop_max_ang_speed=max_ang_speed,
            closed_loop_max_lin_accel=cl_config.max_lin_accel,
            closed_loop_max_ang_accel=cl_config.max_ang_accel,
            closed_loop_deadband_lin=cl_config.deadband_lin,
            closed_loop_deadband_ang=cl_config.deadband_ang,
            robot_name=str(os.getenv("ROBOT_NAME")), n_frames=n,
        ))
        if extra_attrs:
            f.attrs.update(extra_attrs)
        tg = f.create_group("time")
        tg.create_dataset("t_wall_s", data=arr("t"))
        tg.create_dataset("unix_ns", data=arr("unix_ns"))
        tg.create_dataset("steer_ts_ns", data=arr("steer_ts_ns"))
        tg.create_dataset("drive_ts_ns", data=arr("drive_ts_ns"))

        ag = f.create_group("action/chassis")
        ag.create_dataset("cmd_vel", data=np.column_stack([arr("cmd_vx"), arr("cmd_vy"), arr("cmd_wz")]))
        ag.create_dataset("label", data=np.array(rec["cmd_label"], dtype="S24"))

        cg = f.create_group("obs/chassis")
        cg.create_dataset("steering_angle", data=np.column_stack([arr("steer_l"), arr("steer_r")]))
        cg.create_dataset("wheel_velocity", data=np.column_stack([arr("wvel_l"), arr("wvel_r")]))
        cg.create_dataset("wheel_encoder_pos", data=np.column_stack([arr("enc_l"), arr("enc_r")]))

        og = f.create_group("obs/odom")
        og.create_dataset("pose", data=np.column_stack([arr("odom_x"), arr("odom_y"), arr("odom_yaw")]))
        og.create_dataset("twist", data=np.column_stack([arr("odom_vx"), arr("odom_vy"), arr("odom_w")]))

        clg = f.create_group("control/closed_loop")
        clg.create_dataset("reference_pose", data=np.column_stack([arr("cl_ref_x"), arr("cl_ref_y"), arr("cl_ref_yaw")]))
        clg.create_dataset(
            "feedback_pose",
            data=np.column_stack([arr("cl_feedback_x"), arr("cl_feedback_y"), arr("cl_feedback_yaw")]),
        )
        clg.create_dataset("pose_error_body", data=np.column_stack([arr("cl_err_x"), arr("cl_err_y"), arr("cl_err_yaw")]))
        clg.attrs["source"] = cl_config.source

        jg = f.create_group("obs/joint")
        jg.create_dataset("head", data=np.column_stack([arr("head_j1"), arr("head_j2"), arr("head_j3")]))
        jg.create_dataset("torso", data=np.column_stack([arr("torso_j1"), arr("torso_j2"), arr("torso_j3")]))

        if args.record_images:
            ig = f.create_group("obs/images")
            ig.create_dataset("head_left_rgb", data=_stack(rec["head_rgb"]).astype(np.uint8),
                              compression="gzip", compression_opts=4)
            ig.create_dataset("head_depth", data=_stack(rec["head_depth"]).astype(np.float32),
                              compression="gzip", compression_opts=4)
            # shared head intrinsic (omniteleop.common.head_camera.ZED_K) for the
            # publisher's crop/resize geometry — same value vr_reader stamps.
            ig.create_dataset("intrinsic", data=ZED_K.astype(np.float32))

        lg = f.create_group("legs")
        lg.create_dataset("label", data=np.array([l.label for l in legs], dtype="S24"))
        lg.create_dataset("cmd_vel", data=np.array([[l.vx, l.vy, l.wz] for l in legs]))
        lg.create_dataset("drive_time", data=np.array([l.drive_time for l in legs]))
        lg.create_dataset("t_issue", data=np.array([l.t_issue for l in legs]))
        lg.create_dataset("t_done", data=np.array([l.t_done for l in legs]))
        lg.create_dataset("completion_reason", data=np.array([l.completion_reason for l in legs], dtype="S64"))
        lg.create_dataset("abort_reason", data=np.array([l.abort_reason for l in legs], dtype="S256"))


def compute_metrics(rec: dict, legs: list[Leg], args: Args) -> dict:
    """Chassis control accuracy (commanded vs odom) and command latency.

    Pure function of the recorded arrays + the per-leg command windows, so it is
    unit-testable offline with synthetic logs (tests/test_drive_box_metrics.py).
    """
    t = np.asarray(rec["t"], dtype=np.float64)
    odom = np.column_stack([rec["odom_x"], rec["odom_y"], rec["odom_yaw"]]).astype(np.float64)
    steer = np.column_stack([rec["steer_l"], rec["steer_r"]]).astype(np.float64)
    meas_speed = np.hypot(np.asarray(rec["odom_vx"]), np.asarray(rec["odom_vy"]))
    meas_w = np.abs(np.asarray(rec["odom_w"]))

    def pose_at(series: np.ndarray, t_q: float) -> np.ndarray:
        i = int(np.clip(np.searchsorted(t, t_q) - 1, 0, len(t) - 1))
        return series[i]

    legs_out = []
    for leg in legs:
        if not np.isfinite(leg.t_issue) or not np.isfinite(leg.t_done):
            continue
        dx, dy, dyaw = leg.expected_disp()
        exp_lin = float(np.hypot(dx, dy))
        is_turn = leg.wz != 0.0
        win = (t >= leg.t_issue) & (t <= leg.t_done)
        tw = t[win]
        o0, o1 = pose_at(odom, leg.t_issue), pose_at(odom, leg.t_done)
        odom_d = o1 - o0
        # yaw via unwrap: differencing wrapped endpoint angles reads ~0 for a full
        # revolution; unwrapping the in-window series recovers the true ~2π.
        if int(win.sum()) >= 2:
            odom_d[2] = float(np.unwrap(odom[win, 2])[[0, -1]] @ [-1.0, 1.0])
        target = abs(leg.wz) if is_turn else leg.target_speed
        sig = meas_w[win] if is_turn else meas_speed[win]
        drive_onset = float(tw[sig > 0.1 * target][0] - leg.t_issue) if np.any(sig > 0.1 * target) else float("nan")
        s_ref = pose_at(steer, leg.t_issue)
        steer_dev = np.abs(steer[win] - s_ref).max(axis=1) if win.any() else np.array([])
        steer_onset = float(tw[steer_dev > 0.05][0] - leg.t_issue) if np.any(steer_dev > 0.05) else float("nan")
        legs_out.append(dict(
            label=leg.label,
            expected=dict(dx=dx, dy=dy, dyaw_deg=float(np.degrees(dyaw)), lin=exp_lin),
            odom=dict(dx=float(odom_d[0]), dy=float(odom_d[1]), dyaw_deg=float(np.degrees(odom_d[2])),
                      lin=float(np.hypot(odom_d[0], odom_d[1]))),
            lin_err_odom_m=float(np.hypot(odom_d[0], odom_d[1]) - exp_lin) if not is_turn else None,
            yaw_err_odom_deg=float(np.degrees(odom_d[2] - dyaw)) if is_turn else None,
            latency_steer_onset_s=steer_onset,
            latency_drive_onset_s=drive_onset,
        ))

    # return-to-origin closure (everything started at 0 at the first sample)
    closure = dict(
        odom_xy_m=float(np.hypot(odom[-1, 0], odom[-1, 1])),
        odom_yaw_deg=float(np.degrees(odom[-1, 2])),
    )
    # global command→response lag by cross-correlation of |commanded| vs |measured| speed
    cmd_speed = np.hypot(np.asarray(rec["cmd_vx"]), np.asarray(rec["cmd_vy"])) + np.abs(np.asarray(rec["cmd_wz"]))
    lag = _xcorr_lag(cmd_speed, meas_speed + meas_w, dt=float(np.median(np.diff(t))) if len(t) > 1 else 0.1)

    steer_lat = [l["latency_steer_onset_s"] for l in legs_out if np.isfinite(l["latency_steer_onset_s"])]
    drive_lat = [l["latency_drive_onset_s"] for l in legs_out if np.isfinite(l["latency_drive_onset_s"])]
    return dict(
        n_frames=len(t),
        effective_rate_hz=float((len(t) - 1) / (t[-1] - t[0])) if len(t) > 1 else 0.0,
        legs=legs_out,
        closure=closure,
        latency=dict(
            xcorr_lag_s=lag,
            mean_steer_onset_s=float(np.mean(steer_lat)) if steer_lat else None,
            mean_drive_onset_s=float(np.mean(drive_lat)) if drive_lat else None,
        ),
    )


def _xcorr_lag(cmd: np.ndarray, meas: np.ndarray, dt: float, max_lag_s: float = 2.0) -> float:
    """Lag (s, >0 = meas trails cmd) maximising normalised cross-correlation."""
    cmd = cmd - cmd.mean(); meas = meas - meas.mean()
    if cmd.std() < 1e-9 or meas.std() < 1e-9:
        return float("nan")
    max_lag = int(max_lag_s / dt)
    best_k, best_c = 0, -np.inf
    for k in range(0, max_lag + 1):  # only positive lags: response trails command
        a, b = cmd[: len(cmd) - k], meas[k:]
        if len(a) < 4:
            break
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        if c > best_c:
            best_c, best_k = c, k
    return best_k * dt


def _log_metrics(m: dict, legs: list[Leg]) -> None:
    logger.info(f"=== accuracy / latency  (rate {m['effective_rate_hz']:.1f} Hz, "
                f"{m['n_frames']} frames) ===")
    for L in m["legs"]:
        o = L["odom"]
        extra = (f"yawErr(odom)={L['yaw_err_odom_deg']:+.1f}deg"
                 if L["yaw_err_odom_deg"] is not None
                 else f"linErr(odom)={1000 * L['lin_err_odom_m']:+.0f}mm")
        logger.info(
            f"  {L['label']:12s} odom d=({o['dx']:+.3f},{o['dy']:+.3f}, "
            f"{o['dyaw_deg']:+.0f}d)  {extra}  "
            f"lat steer/drive={L['latency_steer_onset_s']:.2f}/{L['latency_drive_onset_s']:.2f}s")
    c = m["closure"]
    logger.info(f"  return-to-origin: odom |xy|={1000 * c['odom_xy_m']:.0f}mm "
                f"yaw={c['odom_yaw_deg']:+.1f}deg")
    lat = m["latency"]
    logger.info(f"  latency: xcorr={lat['xcorr_lag_s']:.2f}s  "
                f"mean steer-onset={lat['mean_steer_onset_s']}  "
                f"mean drive-onset={lat['mean_drive_onset_s']}  "
                f"(drive-onset includes the sequential re-steer)")


if __name__ == "__main__":
    main()
