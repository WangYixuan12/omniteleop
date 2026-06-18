#!/usr/bin/env python3
"""Drive the Vega chassis through a closed box, record at 10 Hz, and
log chassis control accuracy + latency against two independent base-pose sources.

Trajectory (the user's script, wrapped with recording + metrics)::

    move_straight(+0.4, 2 s)   # +x  ~0.8 m forward
    move_sideways(-0.4, 4 s)   # -y  ~1.6 m right
    move_straight(-0.4, 2 s)   # -x  ~0.8 m back
    move_sideways(+0.4, 4 s)   # +y  ~1.6 m left      -> translation closes to origin
    stop()

Every leg goes through ``chassis.move_*/turn`` -> ``set_velocity(...,
sequential_steering=True)``, so each leg first re-steers the wheels in place
(~``steering_wait_time``) and only then drives for ``wait_time``. That built-in
re-steer is part of the command latency these helpers impose and is measured
explicitly below (command -> steer onset vs command -> drive onset).

Two independent "actual base pose" references are logged against the commanded
twist, both starting at 0 at the first recorded sample:
  * swerve wheel odometry  (SLAM/bridge/swerve_odometry.py, drive_state_mode='ms',
    the value verified for this robot in SLAM/bridge/bridge_params.yaml)
  * ZED base pose          (state/base_pose_zed from scripts/zed_base_pose_node.py)

Run the tracking stack first, in separate terminals (head kept STILL — that is
where the head ZED is accurate)::

    python tests/test_head_zedx_depth.py --enable-tracking \
        --area-file /home/dexmate/yixuan/Dexmate/SLAM/zed/results/maps/lab_v1.area \
        --localization-only
    python scripts/zed_base_pose_node.py --duration 0 --save-csv ~/yixuan/Dexmate/SLAM/zed_drive.csv

Then, with the robot reachable and the dexmate env active::

    python scripts/drive_box_record.py \
        --zed-area-file /home/dexmate/yixuan/Dexmate/SLAM/zed/results/maps/lab_v1.area \
        --output ~/yixuan/Dexmate/SLAM/drive_box.hdf5

SAFETY: the base physically drives a ~0.8 m x ~1.6 m box — clear ~1.5 m radius
and keep a hand on the e-stop. ``--dry-run`` records and
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
sys.path.insert(0, (_REPO / "SLAM" / "bridge").as_posix())
sys.path.insert(0, (_REPO / "src").as_posix())

from swerve_odometry import SwerveOdometry  # noqa: E402  (pure-numpy, SLAM/bridge)
from omniteleop.common.head_camera import ZED_K  # noqa: E402  (shared head intrinsic)
from omniteleop.follower import base_closed_loop as base_cl  # noqa: E402


@dataclasses.dataclass
class Args:
    output: str = "/home/dexmate/yixuan/Dexmate/SLAM/box_test/drive_box.hdf5"
    """Output HDF5 (10 Hz recording). Metrics JSON is written next to it."""

    speed: float = 0.2
    """Linear speed (m/s) for every straight/sideways leg (the user's 0.4)."""

    leg_time: float = 2.0
    """Base drive time (s): x legs use this, y legs currently use 2x this."""

    turn_speed: float = 0.6
    """Angular speed reference (rad/s), used by closed-loop angular speed defaults
    and by future yaw legs if build_legs() is extended."""

    turn_revolutions: float = 1.0
    """Reserved yaw-leg setting; the active default trajectory is box-only."""

    settle: float = 0.5
    """Stop + dwell (s) inserted between legs so each leg settles and the
    per-leg metrics separate cleanly. Set 0 for the back-to-back original."""

    record_rate: float = 10.0
    """HDF5 sampling rate (Hz). The user's 'record all data at 10 Hz'."""

    odom_rate: float = 50.0
    """Swerve-odometry integration rate (Hz). Higher than record_rate so the
    integrated pose does not under-sample the steering transients."""

    sensor_id: str = "head_camera"
    """Head ZED publisher id: sensors/<id>/{left_rgb,depth} + state/base_pose_zed."""

    base_pose_topic: str = "state/base_pose_zed"
    """Topic published by scripts/zed_base_pose_node.py."""

    zed_area_file: Optional[str] = None
    """Optional .area map that the upstream ZED pose publisher must be using for
    relocalization. This script does not own the camera; it verifies the
    state/base_pose_zed metadata before driving."""

    require_localization_only: bool = True
    """When --zed-area-file is set, require the upstream ZED publisher to run
    with localization_only=True so the fixed map is not updated during the run."""

    record_images: bool = True
    """Record head left_rgb + depth at record_rate (the chosen scope)."""

    drive_state_mode: str = "ms"
    """'ms' (wheel_velocity is signed m/s, verified for this robot) or 'rad'."""

    startup_timeout_s: float = 15.0
    """Fail fast if a required stream is silent this long at startup."""

    dry_run: bool = False
    """Record + verify wiring but issue NO chassis motion (base stays parked)."""

    closed_loop_source: str = "none"
    """Feedback source for optional pose-PD base control: 'none', 'odom', or 'zed'.
    'none' preserves the original fixed-duration open-loop command sequence."""

    closed_loop_rate: float = 20.0
    """Closed-loop command rate (Hz) when --closed-loop-source is odom or zed."""

    closed_loop_kp_xy: float = 1.0
    """Conservative proportional gain for planar pose error (1/s)."""

    closed_loop_kp_yaw: float = 1.5
    """Conservative proportional gain for yaw error (1/s)."""

    closed_loop_pos_tolerance: float = 0.03
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


def _resolve_existing_area_file(path: Optional[str]) -> Optional[pathlib.Path]:
    if not path:
        return None
    area = pathlib.Path(path).expanduser().resolve()
    if not area.is_file():
        raise SystemExit(f"--zed-area-file does not exist: {area}")
    return area


def _same_file(a: pathlib.Path, b: pathlib.Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return a == b


def _check_zed_relocalization_contract(
    msg: dict,
    area_file: pathlib.Path,
    require_localization_only: bool,
) -> None:
    actual = msg.get("area_file_path")
    if not actual:
        raise SystemExit(
            "--zed-area-file was requested, but state/base_pose_zed does not report "
            "an area_file_path. Restart the ZED publisher with "
            "`tests/test_head_zedx_depth.py --enable-tracking --area-file "
            f"{area_file} --localization-only`, then restart zed_base_pose_node.py."
        )
    actual_path = pathlib.Path(str(actual)).expanduser().resolve()
    if not _same_file(actual_path, area_file):
        raise SystemExit(
            "ZED relocalization map mismatch: "
            f"requested {area_file}, but state/base_pose_zed reports {actual_path}"
        )
    if require_localization_only and not bool(msg.get("localization_only", False)):
        raise SystemExit(
            "--zed-area-file was requested with --require-localization-only, but "
            "state/base_pose_zed reports localization_only=False. Restart the ZED "
            "publisher with --localization-only for a fixed-map drive/policy run."
        )
    logger.info(
        "ZED relocalization verified: "
        f"area_file={area_file} "
        f"localization_only={bool(msg.get('localization_only', False))} "
        f"raw_coordinate_system={msg.get('raw_pose_coordinate_system')}"
    )


def _validate_args(args: Args) -> None:
    if args.drive_state_mode not in ("ms", "rad"):
        raise ValueError(f"--drive-state-mode must be 'ms' or 'rad', got {args.drive_state_mode!r}")
    positive = {"record_rate": args.record_rate, "odom_rate": args.odom_rate}
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
    """The active box trajectory; it closes in translation and keeps heading fixed."""
    s, t = args.speed, args.leg_time
    return [
        Leg("forward +x", +s, 0.0, 0.0, t),
        Leg("right -y", 0.0, -s, 0.0, 2*t),
        Leg("back -x", -s, 0.0, 0.0, t),
        Leg("left +y", 0.0, +s, 0.0, 2*t),
    ]

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
    """Integrate swerve odometry at ``odom_rate`` from live wheel state.

    The pose is integrated continuously here; the recorder samples it at the
    (slower) record rate, so the recorded pose is full-fidelity in value."""
    odo = SwerveOdometry(drive_state_mode=args.drive_state_mode)
    period = 1.0 / args.odom_rate
    prev = time.perf_counter()
    while True:
        now = time.perf_counter()
        with shared.lock:
            stop = shared.stop_flag
        if stop:
            break
        try:
            steer = np.asarray(robot.chassis.steering_angle, dtype=np.float64)
            wvel = np.asarray(robot.chassis.wheel_velocity, dtype=np.float64)
            dt = now - prev
            prev = now
            if steer.shape == (2,) and wvel.shape == (2,) and dt > 0:
                pose = odo.update(steer, wvel, dt)
                with shared.lock:
                    shared.odom_pose = pose.copy()
                    shared.odom_twist = odo.twist.copy()
                    shared.odom_update_t = now
                    shared.odom_updates += 1
        except Exception:
            logger.exception("odom thread stopped on error")
            break
        sleep = period - (time.perf_counter() - now)
        if sleep > 0:
            time.sleep(sleep)


def _apply_start_anchor(
    x: float, y: float, yaw: float, zed0: Optional[np.ndarray]
) -> tuple[float, float, float]:
    """Re-express a ZED-node odom-frame (x, y, yaw) in the record-start BASE frame.

    The ZED node reports poses in its own gravity-aligned odom frame whose yaw axis
    is wherever the base pointed when ZED first locked — not the chassis heading at
    record start (they differ by yaw0). We therefore BOTH translate to the first
    recorded sample AND rotate the displacement by -yaw0 into the base-forward frame.
    Subtracting yaw0 from the yaw column but NOT rotating xy leaves the trajectory
    correctly shaped yet rotated by yaw0 vs odom — odom needs no such step because it
    integrates from identity in this exact frame.
    """
    if zed0 is None:
        return x, y, yaw
    x0, y0, yaw0 = zed0
    c, s = np.cos(yaw0), np.sin(yaw0)
    dx, dy = x - x0, y - y0
    return c * dx + s * dy, -s * dx + c * dy, np.arctan2(np.sin(yaw - yaw0), np.cos(yaw - yaw0))


def _zed_relative(msg: Optional[dict], zed0: Optional[np.ndarray]) -> tuple[np.ndarray, bool, int]:
    """ZED BASE (x, y, yaw) in the record-start base frame (so it matches swerve odom)."""
    if msg is None:
        return np.full(3, np.nan), False, -1
    x, y, yaw = _apply_start_anchor(
        msg.get("x", np.nan), msg.get("y", np.nan), msg.get("yaw", np.nan), zed0
    )
    return np.array([x, y, yaw]), msg.get("state") == "OK", int(msg.get("sequence", -1))


def _zed_cam_relative(msg: Optional[dict], zed0: Optional[np.ndarray]) -> np.ndarray:
    """Raw ZED CAMERA (x, y, yaw) in the SAME record-start base frame as _zed_relative.

    Anchored with the base's ``zed0`` (NOT the camera's own start) so the camera curve
    overlays the base curve and their instantaneous gap is exactly the head->base FK
    lever arm. NaN when the ZED node predates the cam_* fields (graceful for old data).
    """
    if msg is None:
        return np.full(3, np.nan)
    x, y, yaw = _apply_start_anchor(
        msg.get("cam_x", np.nan), msg.get("cam_y", np.nan), msg.get("cam_yaw", np.nan), zed0
    )
    return np.array([x, y, yaw])


def main() -> None:
    args = tyro.cli(Args)
    _validate_args(args)
    cl_config = _closed_loop_config(args)
    closed_loop = cl_config.enabled
    out = pathlib.Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    zed_area_file = _resolve_existing_area_file(args.zed_area_file)

    cfg = _configure_zenoh_like_dexcontrol()
    logger.info(f"Zenoh config: {cfg if cfg else '<dexcomm default>'}; "
                f"ROBOT_NAME={os.getenv('ROBOT_NAME')!r}")

    # ── connect to robot + subscribe to ZED pose / head frames ────────────────
    from dexcomm import Node
    from dexcomm.codecs import DepthImageCodec, JsonDataCodec, RGBImageCodec
    from dexcontrol.robot import Robot

    robot = Robot()
    node = Node(name="drive_box_record", namespace="")
    zed_sub = node.create_subscriber(args.base_pose_topic, decoder=JsonDataCodec.decode)
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
            "Need: head publisher (test_head_zedx_depth.py --enable-tracking) and "
            "zed_base_pose_node.py for state/base_pose_zed.")

    logger.info("Waiting for chassis state, ZED base pose, and head frames…")
    _ = _wait(lambda: robot.chassis.steering_angle if
              np.asarray(robot.chassis.steering_angle).shape == (2,) else None, "chassis state")
    zed_start_msg = _wait(zed_sub.get_latest, "state/base_pose_zed (start zed_base_pose_node.py)")
    if zed_area_file is not None:
        _check_zed_relocalization_contract(
            zed_start_msg,
            zed_area_file,
            args.require_localization_only,
        )
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
        "zed_x", "zed_y", "zed_yaw", "zed_fresh", "zed_seq",
        "zed_cam_x", "zed_cam_y", "zed_cam_yaw",
        "cl_ref_x", "cl_ref_y", "cl_ref_yaw",
        "cl_feedback_x", "cl_feedback_y", "cl_feedback_yaw",
        "cl_err_x", "cl_err_y", "cl_err_yaw",
        "head_j1", "head_j2", "head_j3", "torso_j1", "torso_j2", "torso_j3",
        "head_rgb", "head_depth")}
    zed0 = {"v": None}  # ZED pose at first recorded sample (boxed for closure)

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
            zmsg = zed_sub.get_latest()
            if zed0["v"] is None and zmsg is not None:
                zed0["v"] = np.array([zmsg.get("x", 0.0), zmsg.get("y", 0.0), zmsg.get("yaw", 0.0)])
            zpose, zfresh, zseq = _zed_relative(zmsg, zed0["v"])
            zcam = _zed_cam_relative(zmsg, zed0["v"])

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
            rec["zed_x"].append(zpose[0]); rec["zed_y"].append(zpose[1]); rec["zed_yaw"].append(zpose[2])
            rec["zed_fresh"].append(bool(zfresh)); rec["zed_seq"].append(zseq)
            rec["zed_cam_x"].append(zcam[0]); rec["zed_cam_y"].append(zcam[1]); rec["zed_cam_yaw"].append(zcam[2])
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
    if closed_loop and args.closed_loop_source == "zed":
        _wait(
            lambda: True if zed0["v"] is not None else None,
            "record-start ZED anchor for closed-loop source=zed",
        )

    # ── drive the sequence (main thread) ──────────────────────────────────────
    t_run0 = t_epoch  # leg windows share rec["t"]'s origin so compute_metrics aligns
    nan3 = np.full(3, np.nan)
    max_lin_speed, max_ang_speed = base_cl.closed_loop_limits(cl_config, args.speed, args.turn_speed)
    zed_freshness = base_cl.FreshSequence()

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
        if args.closed_loop_source == "zed":
            msg = zed_sub.get_latest()
            if msg is None:
                raise RuntimeError("zed feedback has no message")
            pose, ok, seq = _zed_relative(msg, zed0["v"])
            if not ok:
                raise RuntimeError(f"zed feedback state is {msg.get('state')!r}, expected 'OK'")
            if not np.isfinite(pose).all():
                raise RuntimeError(f"zed feedback pose is not finite: {pose}")
            base_cl.fresh_sequence_age_s(
                seq,
                now,
                zed_freshness,
                cl_config.source_timeout_s,
                "zed feedback",
            )
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
                elif tick - stable_since >= args.closed_loop_stable_time:
                    leg.completion_reason = "closed_loop_tolerance"
                    return
            else:
                stable_since = None
            sleep = period - (time.perf_counter() - tick)
            if sleep > 0:
                time.sleep(sleep)

    abort_reason: Optional[str] = None
    try:
        for leg in legs:
            leg.t_issue = time.perf_counter() - t_run0
            logger.info(f"→ {leg.label}")
            if closed_loop and not args.dry_run:
                try:
                    run_closed_loop_leg(leg)
                except RuntimeError as exc:
                    leg.abort_reason = str(exc)
                    leg.completion_reason = "aborted"
                    abort_reason = str(exc)
                    logger.error(f"Closed-loop abort: {abort_reason}")
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

    # ── persist + metrics ─────────────────────────────────────────────────────
    _write_hdf5(out, rec, args, legs)
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


def _write_hdf5(out: pathlib.Path, rec: dict, args: Args, legs: list[Leg]) -> None:
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
            zed_area_file=str(args.zed_area_file or ""),
            require_localization_only=bool(args.require_localization_only),
            robot_name=str(os.getenv("ROBOT_NAME")), n_frames=n,
        ))
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

        zg = f.create_group("obs/zed")
        zg.create_dataset("pose", data=np.column_stack([arr("zed_x"), arr("zed_y"), arr("zed_yaw")]))
        zg.create_dataset("cam_pose", data=np.column_stack([arr("zed_cam_x"), arr("zed_cam_y"), arr("zed_cam_yaw")]))
        zg.create_dataset("fresh", data=arr("zed_fresh"))
        zg.create_dataset("sequence", data=arr("zed_seq"))
        zg.attrs["area_file"] = str(args.zed_area_file or "")
        zg.attrs["require_localization_only"] = bool(args.require_localization_only)

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
    """Chassis control accuracy (commanded vs odom vs ZED) and command latency.

    Pure function of the recorded arrays + the per-leg command windows, so it is
    unit-testable offline with synthetic logs (tests/test_drive_box_metrics.py).
    """
    t = np.asarray(rec["t"], dtype=np.float64)
    odom = np.column_stack([rec["odom_x"], rec["odom_y"], rec["odom_yaw"]]).astype(np.float64)
    zed = np.column_stack([rec["zed_x"], rec["zed_y"], rec["zed_yaw"]]).astype(np.float64)
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
        z0, z1 = pose_at(zed, leg.t_issue), pose_at(zed, leg.t_done)
        odom_d, zed_d = o1 - o0, z1 - z0
        # yaw via unwrap: differencing wrapped endpoint angles reads ~0 for a full
        # revolution; unwrapping the in-window series recovers the true ~2π.
        if int(win.sum()) >= 2:
            odom_d[2] = float(np.unwrap(odom[win, 2])[[0, -1]] @ [-1.0, 1.0])
            zed_d[2] = float(np.unwrap(zed[win, 2])[[0, -1]] @ [-1.0, 1.0])
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
            zed=dict(dx=float(zed_d[0]), dy=float(zed_d[1]), dyaw_deg=float(np.degrees(zed_d[2])),
                     lin=float(np.hypot(zed_d[0], zed_d[1]))),
            lin_err_odom_m=float(np.hypot(odom_d[0], odom_d[1]) - exp_lin) if not is_turn else None,
            yaw_err_odom_deg=float(np.degrees(odom_d[2] - dyaw)) if is_turn else None,
            latency_steer_onset_s=steer_onset,
            latency_drive_onset_s=drive_onset,
        ))

    # return-to-origin closure (everything started at 0 at the first sample)
    closure = dict(
        odom_xy_m=float(np.hypot(odom[-1, 0], odom[-1, 1])),
        odom_yaw_deg=float(np.degrees(odom[-1, 2])),
        zed_xy_m=float(np.hypot(zed[-1, 0], zed[-1, 1])) if np.isfinite(zed[-1, 0]) else None,
        zed_yaw_deg=float(np.degrees(zed[-1, 2])) if np.isfinite(zed[-1, 2]) else None,
    )
    # odom vs ZED agreement over the run (planar position)
    valid = np.isfinite(zed[:, 0])
    od_zed = (float(np.sqrt(np.mean(np.sum((odom[valid, :2] - zed[valid, :2]) ** 2, axis=1))))
              if valid.any() else None)
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
        odom_vs_zed_rms_m=od_zed,
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
        e, o, z = L["expected"], L["odom"], L["zed"]
        extra = (f"yawErr(odom)={L['yaw_err_odom_deg']:+.1f}deg"
                 if L["yaw_err_odom_deg"] is not None
                 else f"linErr(odom)={1000 * L['lin_err_odom_m']:+.0f}mm")
        logger.info(
            f"  {L['label']:12s} odom d=({o['dx']:+.3f},{o['dy']:+.3f}, "
            f"{o['dyaw_deg']:+.0f}d)  zed d=({z['dx']:+.3f},{z['dy']:+.3f}, "
            f"{z['dyaw_deg']:+.0f}d)  {extra}  "
            f"lat steer/drive={L['latency_steer_onset_s']:.2f}/{L['latency_drive_onset_s']:.2f}s")
    c = m["closure"]
    zed_xy = "n/a" if c["zed_xy_m"] is None else f"{1000 * c['zed_xy_m']:.0f}mm"
    zed_yaw = "n/a" if c["zed_yaw_deg"] is None else f"{c['zed_yaw_deg']:+.1f}deg"
    logger.info(f"  return-to-origin: odom |xy|={1000 * c['odom_xy_m']:.0f}mm "
                f"yaw={c['odom_yaw_deg']:+.1f}deg | zed |xy|={zed_xy} yaw={zed_yaw}")
    lat = m["latency"]
    logger.info(f"  latency: xcorr={lat['xcorr_lag_s']:.2f}s  "
                f"mean steer-onset={lat['mean_steer_onset_s']}  "
                f"mean drive-onset={lat['mean_drive_onset_s']}  "
                f"(drive-onset includes the sequential re-steer)")
    if m["odom_vs_zed_rms_m"] is not None:
        logger.info(f"  odom vs ZED planar RMS = {1000 * m['odom_vs_zed_rms_m']:.0f} mm")


if __name__ == "__main__":
    main()
