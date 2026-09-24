#!/usr/bin/env python3
"""Wrist ZED-M publisher: stereo RGB (optionally depth) via SDK → Zenoh.

Doubles as the publisher that ``vr_reader.py --cameras left_wrist_rgb ...``
subscribes to. ``dexsensor``'s ``usb_camera`` driver gives raw V4L2 only (and
tears frames); the ZED SDK's ``grab()`` is integrity-checked, so this script
bypasses dexsensor and publishes directly.

  pyzed (owns /dev/video10)
     │
     ├─ grab → LEFT view + RIGHT view  (+ depth only if --enable-depth)
     │
     ├─ publisher → sensors/<id>/left_rgb   (RGBImageCodec, RGB uint8)
     ├─ publisher → sensors/<id>/right_rgb  (RGBImageCodec, RGB uint8)
     ├─ publisher → sensors/<id>/depth      (DepthImageCodec)   [opt-in]
     ├─ service   → sensors/<id>/info       (JsonDataCodec)
     └─ service   → sensors/<id>/clock      (JsonDataCodec, publisher host clock)

Topic naming matches the head_camera convention (left_rgb / right_rgb /
depth) so vr_reader's ``ZedCameraSensor`` subscribes to wrist streams the
same way it does head streams.

RGB-only by default: vr_reader records the wrist RGB-only, and the NEURAL
stereo matcher + 1.8 MB depth frames throttle the capture loop hard. Run with
``--enable-depth`` only when you actually need depth.

This is a long-running dependency, *not* a one-shot test — start it and leave
it running for the whole vr_reader session (no ``--duration``). Confirm it is
reaching the workstation with ``python scripts/diagnostics/probe_wrist_stream.py`` there.

Important: only one process can hold the V4L2 device. Stop
``dexsensor launch ... --sensor wrist_zedm`` before running.

Usage::

    # production: one process per wrist camera, run until Ctrl-C, RGB-only.
    # The serial is pinned by sensor_id (_ARM_SERIALS), so no --serial-number needed
    # and a mismatched one is rejected rather than silently swapping the arms.
    python tests/test_wrist_zedm_depth.py --sensor-id left_wrist_zedm
    python tests/test_wrist_zedm_depth.py --sensor-id right_wrist_zedm
    # debug round-trip fps / depth
    python tests/test_wrist_zedm_depth.py --sensor-id left_wrist_zedm \
        --verify --enable-depth --save-dir ~/Dexmate
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import sys
import time
from typing import Literal, Optional

import cv2
import numpy as np
import pyzed.sl as sl
import tyro
from dexcomm import Node
from dexcomm.codecs import DepthImageCodec, JsonDataCodec, RGBImageCodec
from loguru import logger

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from omniteleop.common.zed_motion import ZedMotion, resolve_depth_mode
from omniteleop.common.zed_photometric import PhotometricTelemetry

# Native sizes: 2208x1242, 1920x1080, 1280x720, 672x376.
Resolution = Literal["HD720", "HD1080", "HD2K", "VGA"]
DepthMode = Literal["NEURAL", "NEURAL_LIGHT", "ULTRA", "QUALITY", "PERFORMANCE"]

# Which physical camera belongs on which arm. Publishing one arm's camera under the
# other arm's sensor_id mislabels every recorded frame downstream and is unrecoverable
# from the files, so the pairing is checked rather than trusted to the command line.
# Update this table (and only this) if a wrist camera is replaced.
_ARM_SERIALS: dict[str, int] = {
    "left_wrist_zedm": 12417276,
    "right_wrist_zedm": 12930616,
    # Aliases: NOT the ids anything consumes (wbc_vr_robot.py, the leader HUD and the
    # scan scripts all read sensors/<left|right>_wrist_zedm/*), but dexcontrol's built-in
    # sensors carry these names and a launch under them is an easy slip. Pinned so that
    # slip still cannot swap the arms on top of publishing to a topic nobody reads.
    "left_wrist_camera": 12417276,
    "right_wrist_camera": 12930616,
}

# Abort after this long with no successful grab. A camera that has dropped off USB
# (e.g. a SuperSpeed link stuck in a reset loop) fails EVERY subsequent grab, so
# retrying forever just spins the CPU while publishing nothing -- the process looks
# alive, the stream is dead, and the failure only surfaces much later as a
# record-side stale-frame abort. Die loudly instead so the operator sees which
# camera went and can restart it.
_GRAB_FAILURE_ABORT_S = 5.0

# Keep images below VR/actuator traffic under congestion while retaining the
# existing best-effort/latest-frame behavior.
_CAMERA_QOS = {
    "reliability": "best_effort",
    "congestion_control": "drop",
    "priority": "data_low",
}


@dataclasses.dataclass
class Args:
    """CLI options for the standalone wrist ZED-M publisher."""

    sensor_id: str = "wrist_zedm"
    """Sensor id used to derive topic names: sensors/<sensor_id>/{rgb,depth}.
    Must match what vr_reader.py / dexcontrol will subscribe to."""

    serial_number: int = 0
    """ZED serial to open. For a known --sensor-id (left_wrist_zedm /
    right_wrist_zedm) this is filled in from _ARM_SERIALS and a CONFLICTING value
    is rejected, so the arm assignment cannot flip between runs. Only an
    unrecognized sensor_id falls back to 0 = first available camera."""

    namespace: str = ""
    """Zenoh namespace passed to dexcomm.Node (matches vr_reader.py:257)."""

    resolution: Resolution = "HD720"
    """ZED capture resolution."""

    enable_imu: bool = False
    """Publish timestamped full-rate IMU batches without visual pose estimation."""

    enable_tracking: bool = False
    """Publish GEN_3 camera pose and full-rate IMU after stereo image delivery."""

    tracking_mode: Literal["GEN_1", "GEN_2", "GEN_3"] = "GEN_3"
    # Compute depth for tracking only; does not publish depth or change RGB size.
    tracking_depth_mode: Optional[DepthMode] = None
    """GEN_3 supports tracking with depth disabled."""

    area_memory: bool = False
    """Enable loop closure; off for continuous collection odometry."""

    enable_depth: bool = False
    """Compute + publish depth. Off by default — vr_reader uses the wrist
    RGB-only, and the stereo matcher throttles the capture loop. Turn on only
    to debug depth."""

    skip_right_rgb: bool = False
    """Skip the right_rgb stream. Off by default so recordings can retain the
    rectified pair for offline FoundationStereo. Turn on for a mono-only run."""

    depth_mode: DepthMode = "NEURAL_LIGHT"
    """Depth estimation algorithm (only used when --enable-depth). NEURAL is
    best quality but heaviest."""

    rate: int = 15
    """Camera fps. The ZED Mini at HD720 only supports a discrete set
    (15/30/60); any other value silently rounds DOWN to the nearest supported
    rate (e.g. 20 -> 15). Default 15 matches the downstream record cadence
    (wbc_vr_robot 10 Hz, vr_reader 15 Hz). Bump to 30 only if grab latency,
    not bandwidth, is the limiter."""

    enable_self_calib: bool = False
    """Allow the ZED SDK to refine stereo calibration during open. Off by default
    because these remounted wrist units have previously returned
    POTENTIAL_CALIBRATION_ISSUE. This switch exists for a controlled pixel-level
    rectification diagnostic; always re-run probe_camera_bandwidth.py --verify and
    inspect calibration repeatability before adopting it for data collection."""

    image_validity_check: int = 1
    """ZED SDK InitParameters.enable_image_validity_check level (0 = off). The ZED
    Mini occasionally delivers a TORN frame -- a byte slip in its side-by-side USB
    video frame, so the rows below one line belong to a neighbouring capture,
    wrapped horizontally, in both eyes at once (episode_9 left wrist frames 1300 and
    2058: 2 of 2339). Nothing downstream can tell: the SDK timestamp and sequence
    are regular. With the check on, grab() returns ERROR_CODE.CORRUPTED_FRAME for
    such a frame and it is SKIPPED (counted in the fps line and /info
    statistics), so a torn image never reaches a recording or a policy. A skipped
    15 fps frame leaves a 133 ms gap, inside wbc_vr_robot's 250 ms stale grace."""

    depth_min: float = 0.3
    """Minimum depth in metres."""

    depth_max: float = 6.0
    """Maximum depth in metres."""

    resize_h: int = 240
    """Output height: each stream is resized to (resize_h, resize_w) before
    publishing so only the small frame crosses WiFi. Set both resize_h and
    resize_w to 0 to publish the raw capture resolution. NOTE: wrist HD720 is
    16:9 (1280x720), so resizing to 4:3 stretches the wrist image — crop first if
    you need the original aspect ratio."""

    resize_w: int = 320
    """Output width (see resize_h)."""

    verify: bool = False
    """Subscribe to our own topics each frame to print round-trip fps. Off by
    default — decoding every frame in-process throttles a continuous publisher
    (and starves the info service)."""

    duration: float = 0.0
    """If > 0, stop after this many seconds. Leave at 0 for normal use: the
    publisher must outlive the vr_reader session."""

    save_dir: Optional[str] = None
    """If set (implies --verify), save the LAST decoded RGB(+depth) received by
    the subscriber once per second, proving the round-trip preserves data."""


def _configure_zenoh_like_dexcontrol() -> Optional[pathlib.Path]:
    """Mirror dexcontrol's default ZENOH_CONFIG discovery.

    vr_reader imports dexcontrol, whose package init sets ZENOH_CONFIG from
    ~/.dexmate/comm/zenoh before DexComm opens a session. This standalone
    publisher does not otherwise import dexcontrol, so do the same discovery
    here to keep publisher and vr_reader on the same Zenoh session.
    """
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


def _depth_to_color(depth_m: np.ndarray) -> np.ndarray:
    finite = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
    if len(finite) == 0:
        gray = np.zeros(depth_m.shape[:2], dtype=np.uint8)
    else:
        mn, mx = finite.min(), np.percentile(finite, 95)
        gray = np.clip((depth_m - mn) / (mx - mn + 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def _zed_image_timestamp_ns(zed: sl.Camera) -> int:
    """SDK IMAGE timestamp for the latest successful grab(), in UNIX ns."""
    timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
    ns = int(timestamp.get_nanoseconds())
    if ns <= 0:
        raise RuntimeError("ZED SDK returned no IMAGE timestamp after grab()")
    return ns


def _clock_response(sensor_id: str, sequence: int) -> dict:
    """NTP-style clock service response from this publisher host."""
    t1 = time.time_ns()
    return {
        "server_receive_time_ns": int(t1),
        "server_send_time_ns": int(time.time_ns()),
        "source": "zed_publisher_host",
        "sensor_id": str(sensor_id),
        "sequence": int(sequence),
    }


def _resolve_serial(sensor_id: str, serial_number: int) -> int:
    """Serial to open, cross-checked against the arm this sensor_id names.

    For a known wrist sensor_id the serial is pinned: omitting it fills in the
    right camera, and passing a different one is a swapped launch -- exactly the
    mistake that silently mislabels an entire recording session.
    """
    expected = _ARM_SERIALS.get(sensor_id)
    if expected is None:
        return serial_number  # unknown/ad-hoc sensor_id: caller is on their own
    if serial_number > 0 and serial_number != expected:
        other = next(
            (sid for sid, sn in _ARM_SERIALS.items() if sn == serial_number), None
        )
        hint = f" -- that serial is {other}" if other else ""
        raise SystemExit(
            f"sensor_id {sensor_id!r} is serial {expected}, but --serial-number "
            f"{serial_number} was given{hint}. Publishing one arm's camera under the "
            "other arm's topic mislabels every recorded frame. Fix the command, or "
            "update _ARM_SERIALS if a camera was replaced."
        )
    return expected


def main() -> None:
    args = tyro.cli(Args)
    verify = args.verify or args.save_dir is not None
    serial_number = _resolve_serial(args.sensor_id, args.serial_number)
    zenoh_config = _configure_zenoh_like_dexcontrol()

    # ── 1. Open ZED-M via SDK ─────────────────────────────────────────────────
    # DEPTH_MODE.NONE skips the stereo matcher entirely — that is what keeps the
    # RGB-only capture loop at full fps.
    depth_mode = getattr(sl.DEPTH_MODE, resolve_depth_mode(args))
    init = sl.InitParameters(
        camera_resolution=getattr(sl.RESOLUTION, args.resolution),
        camera_fps=args.rate,
        depth_mode=depth_mode,
        coordinate_system=sl.COORDINATE_SYSTEM.IMAGE,
        coordinate_units=sl.UNIT.METER,
        depth_minimum_distance=args.depth_min,
        depth_maximum_distance=args.depth_max,
    )
    if serial_number > 0:
        init.set_from_serial_number(serial_number)
    # The SDK re-runs self-calibration on open and rejects its own result once a
    # camera has been remounted (ERROR_CODE.POTENTIAL_CALIBRATION_ISSUE). Pin it
    # to the factory /usr/local/zed/settings/SN<serial>.conf instead.
    init.camera_disable_self_calib = not args.enable_self_calib
    if args.image_validity_check < 0:
        raise ValueError(f"--image-validity-check must be >= 0, got {args.image_validity_check}")
    init.enable_image_validity_check = int(args.image_validity_check)
    runtime = sl.RuntimeParameters()
    zed = sl.Camera()
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(
            f"ZED open failed for serial "
            f"{serial_number if serial_number > 0 else '<first available>'}: "
            f"{err}. Is dexsensor or another publisher holding the device?"
        )

    info = zed.get_camera_information()
    # Guard the whole point of --serial-number: never publish one arm's camera
    # under the other arm's sensor_id.
    if serial_number > 0 and int(info.serial_number) != serial_number:
        zed.close()
        raise ValueError(
            f"Requested serial {serial_number} but the SDK opened "
            f"{info.serial_number}."
        )
    logger.info(
        f"Camera: {info.camera_model} sn={info.serial_number} "
        f"fw={info.camera_configuration.firmware_version}"
    )

    if (args.resize_h > 0) != (args.resize_w > 0):
        raise ValueError(
            f"resize_h ({args.resize_h}) and resize_w ({args.resize_w}) must both "
            "be >0 (resize) or both 0 (skip resize)."
        )
    out_hw: Optional[tuple[int, int]] = (
        (args.resize_h, args.resize_w) if args.resize_h > 0 else None
    )
    cam_res = info.camera_configuration.resolution
    raw_w, raw_h = cam_res.width, cam_res.height
    # width/height are the *published* dims (after resize); all downstream
    # publish/info fields use them, so resizing is transparent to subscribers.
    width, height = (out_hw[1], out_hw[0]) if out_hw is not None else (raw_w, raw_h)
    logger.info(f"Capture {raw_w}x{raw_h} → publish {width}x{height} (WxH)")

    # FoundationStereo returns disparity; metric depth additionally needs the
    # rectified focal length and physical baseline. VIEW.LEFT/VIEW.RIGHT use the
    # SDK's rectified calibration, and both eyes undergo the same resize below.
    # The current 16:9 -> 4:3 default has different x/y scales, so adjust each
    # intrinsic axis independently rather than assuming one uniform scale.
    calib = info.camera_configuration.calibration_parameters
    baseline_m = float(calib.get_camera_baseline())
    if not 0.01 < baseline_m < 0.5:
        zed.close()
        raise SystemExit(
            f"implausible ZED stereo baseline {baseline_m} m; coordinate_units is "
            "METER, so this should be the physical inter-lens distance"
        )

    def _published_k(cam) -> list[list[float]]:
        sx, sy = width / raw_w, height / raw_h
        return [
            [float(cam.fx) * sx, 0.0, float(cam.cx) * sx],
            [0.0, float(cam.fy) * sy, float(cam.cy) * sy],
            [0.0, 0.0, 1.0],
        ]

    stereo = {
        "baseline_m": baseline_m,
        "rectified": True,
        "left_K": _published_k(calib.left_cam),
        "right_K": _published_k(calib.right_cam),
    }
    logger.info(
        f"Stereo calibration: baseline {baseline_m * 1000:.2f} mm, "
        f"left fx {stereo['left_K'][0][0]:.2f} px (published geometry)"
    )

    # ── 2. Set up Zenoh pub/sub via dexcomm ──────────────────────────────────
    left_topic = f"sensors/{args.sensor_id}/left_rgb"
    right_topic = f"sensors/{args.sensor_id}/right_rgb"
    depth_topic = f"sensors/{args.sensor_id}/depth"
    info_topic = f"sensors/{args.sensor_id}/info"
    clock_topic = f"sensors/{args.sensor_id}/clock"

    node = Node(name="wrist_zedm_test", namespace=args.namespace)
    logger.info(
        f"Zenoh config: {zenoh_config if zenoh_config is not None else '<dexcomm default>'}"
    )
    logger.info(
        f"ROBOT_NAME={os.getenv('ROBOT_NAME')!r}; node.namespace={node.namespace!r}"
    )
    left_pub = node.create_publisher(
        left_topic, encoder=RGBImageCodec.encode, qos=_CAMERA_QOS
    )
    right_pub = (
        node.create_publisher(
            right_topic, encoder=RGBImageCodec.encode, qos=_CAMERA_QOS
        )
        if not args.skip_right_rgb
        else None
    )
    depth_pub = (
        node.create_publisher(
            depth_topic, encoder=DepthImageCodec.encode, qos=_CAMERA_QOS
        )
        if args.enable_depth
        else None
    )
    motion = ZedMotion(zed, node, info, args, _CAMERA_QOS)
    photometric = PhotometricTelemetry(zed, sl)
    # Self-verify subscribers are opt-in: decoding every frame in-process
    # throttles the publisher and starves the info queryable.
    left_sub = right_sub = depth_sub = None
    if verify:
        left_sub = node.create_subscriber(left_topic, decoder=RGBImageCodec.decode)
        if right_pub is not None:
            right_sub = node.create_subscriber(right_topic, decoder=RGBImageCodec.decode)
        if args.enable_depth:
            depth_sub = node.create_subscriber(depth_topic, decoder=DepthImageCodec.decode)

    frame_stats = {
        "left_rgb": {"published": 0, "last_timestamp_ns": 0},
        "right_rgb": {"published": 0, "last_timestamp_ns": 0},
        "depth": {"published": 0, "last_timestamp_ns": 0},
    }
    # Torn frames rejected by the SDK validity check; kept OUT of frame_stats, whose
    # values are per-stream dicts updated on every publish.
    corrupted_stats = {"skipped": 0}
    sequence_state = {"latest": 0}
    info_stats = {"queries": 0, "last_log_s": 0.0}

    def _camera_info(_request: bytes | None = None) -> dict:
        info_stats["queries"] += 1
        now_s = time.perf_counter()
        if info_stats["queries"] == 1 or now_s - info_stats["last_log_s"] >= 10.0:
            logger.info(
                f"Camera info queried on {node.resolve_topic(info_topic)!r} "
                f"(count={info_stats['queries']})"
            )
            info_stats["last_log_s"] = now_s
        return {
            "type": "ZED_CAMERA",
            "camera_id": str(info.serial_number),
            "status": "running",
            "model": str(info.camera_model),
            "serial_number": int(info.serial_number),
            "firmware_version": str(info.camera_configuration.firmware_version),
            "actual": {
                "width": int(width),
                "height": int(height),
                "fps": int(args.rate),
            },
            # Queried once by wbc_vr_robot.py when wrist stereo recording is
            # enabled, then stored under meta/wrist_stereo/<arm> in the episode.
            "stereo": stereo,
            "configured": {
                "resolution": args.resolution,
                "self_calibration": bool(args.enable_self_calib),
                "image_validity_check": int(args.image_validity_check),
                "depth_mode": args.depth_mode,
                "sdk_depth_mode": resolve_depth_mode(args),
                "tracking_depth_mode": args.tracking_depth_mode,
                "depth_min": float(args.depth_min),
                "depth_max": float(args.depth_max),
                "resize_hw": [int(height), int(width)],
            },
            "streams": {
                "motion": {
                    "enabled": motion.enabled, "transport": "zenoh",
                    "topic": f"sensors/{args.sensor_id}/motion", "schema_version": 1,
                },
                "left_rgb": {"enabled": True, "transport": "zenoh", "topic": left_topic},
                "right_rgb": {
                    "enabled": not args.skip_right_rgb,
                    "transport": "zenoh",
                    "topic": right_topic,
                },
                "depth": {"enabled": args.enable_depth, "transport": "zenoh", "topic": depth_topic},
            },
            "motion": motion.info(),
            "photometric": photometric.snapshot,
            "statistics": {**frame_stats, "corrupted_skipped": corrupted_stats["skipped"]},
        }

    def _clock(_request: bytes | None = None) -> dict:
        return _clock_response(args.sensor_id, sequence_state["latest"])

    info_service = node.create_service(
        info_topic,
        _camera_info,
        response_encoder=JsonDataCodec.encode,
    )
    clock_service = node.create_service(
        clock_topic,
        _clock,
        response_encoder=JsonDataCodec.encode,
    )
    pub_topics = f"'{left_topic}'"
    if right_pub is not None:
        pub_topics += f", '{right_topic}'"
    if args.enable_depth:
        pub_topics += f", '{depth_topic}'"
    logger.info(f"Publishing on {pub_topics}  (depth={'on' if args.enable_depth else 'off'})")
    logger.info(f"Serving camera info on '{info_topic}'")
    logger.info(f"Serving publisher clock on '{clock_topic}'")
    resolved = f"left={node.resolve_topic(left_topic)!r}"
    if right_pub is not None:
        resolved += f", right={node.resolve_topic(right_topic)!r}"
    resolved += f", info={node.resolve_topic(info_topic)!r}"
    logger.info(f"Resolved wrist keys: {resolved}")
    if verify:
        logger.info("Subscribing to the same topics to verify round-trip")

    save_root: Optional[pathlib.Path] = None
    if args.save_dir:
        save_root = pathlib.Path(args.save_dir) / args.sensor_id
        save_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving 1 frame/s (post-subscriber decode) to {save_root}")

    # ── 3. Capture → publish → subscribe loop ────────────────────────────────
    left_mat = sl.Mat()
    right_mat = sl.Mat()
    depth_mat = sl.Mat()

    # seq is the monotonic frame identity carried in every payload; pub_count is only the
    # fps-log window counter and IS zeroed every second, so it cannot double as the
    # sequence source (that made seq cycle 1..15 and defeated downstream drop detection).
    seq = 0
    pub_count = 0
    corrupted_count = 0  # torn frames skipped in the current fps window
    sub_left_count = 0
    sub_right_count = 0
    sub_depth_count = 0
    t_window = time.perf_counter()
    t_start = t_window
    t_last_save = 0.0
    first_logged = False
    # perf_counter of the first failure in the current grab-failure streak (None while
    # grabs are succeeding); drives the _GRAB_FAILURE_ABORT_S bail-out below.
    grab_failed_since: Optional[float] = None

    try:
        while True:
            grab_err = motion.read(runtime)
            if grab_err == sl.ERROR_CODE.CORRUPTED_FRAME:
                # Torn USB frame (see Args.image_validity_check): drop it. Not a link
                # failure -- the next grab is normally fine -- so it must not feed the
                # _GRAB_FAILURE_ABORT_S timer.
                corrupted_count += 1
                corrupted_stats["skipped"] += 1
                continue
            if grab_err != sl.ERROR_CODE.SUCCESS:
                t_fail = time.perf_counter()
                if grab_failed_since is None:
                    grab_failed_since = t_fail
                    logger.error(
                        f"ZED grab failed ({grab_err}) on {args.sensor_id} "
                        f"sn={info.serial_number}; retrying up to "
                        f"{_GRAB_FAILURE_ABORT_S:g}s. If this persists, check "
                        "'journalctl -k | grep -i usb' for SuperSpeed resets."
                    )
                elif t_fail - grab_failed_since >= _GRAB_FAILURE_ABORT_S:
                    raise SystemExit(
                        f"ZED grab failed continuously for {_GRAB_FAILURE_ABORT_S:g}s on "
                        f"{args.sensor_id} sn={info.serial_number} (last error: "
                        f"{grab_err}). The camera has stopped delivering frames -- most "
                        "likely its USB SuperSpeed link dropped. Exiting rather than "
                        "publishing nothing while appearing healthy."
                    )
                time.sleep(0.01)  # never hot-spin while the link is down
                continue
            if grab_failed_since is not None:
                logger.info(
                    f"ZED grab recovered after "
                    f"{time.perf_counter() - grab_failed_since:.1f}s"
                )
                grab_failed_since = None
            # SDK timestamp of the grabbed image. This rides the codec-preserved
            # timestamp_ns field; extra payload keys are dropped by RGBImageCodec.
            ts = _zed_image_timestamp_ns(zed)

            zed.retrieve_image(left_mat, sl.VIEW.LEFT)
            # ZED returns BGRA — convert to RGB (the RGBImageCodec / vr_reader /
            # rerun all expect RGB; publishing BGR swaps the red/blue channels).
            left_rgb = cv2.cvtColor(left_mat.get_data(), cv2.COLOR_BGRA2RGB)
            if out_hw is not None:  # INTER_AREA: anti-aliased downscale
                left_rgb = cv2.resize(left_rgb, (width, height), interpolation=cv2.INTER_AREA)
            right_rgb = None
            if right_pub is not None:
                zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
                right_rgb = cv2.cvtColor(right_mat.get_data(), cv2.COLOR_BGRA2RGB)
                if out_hw is not None:
                    right_rgb = cv2.resize(right_rgb, (width, height), interpolation=cv2.INTER_AREA)

            depth_safe = None
            if args.enable_depth:
                zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                depth_m = np.ascontiguousarray(depth_mat.get_data())
                if out_hw is not None:  # NEAREST: never average across invalid pixels
                    depth_m = cv2.resize(depth_m, (width, height), interpolation=cv2.INTER_NEAREST)
                # Replace NaN/Inf so DepthImageCodec doesn't choke on them
                depth_safe = np.nan_to_num(
                    depth_m, nan=0.0, posinf=0.0, neginf=0.0
                ).astype(np.float32)

            seq += 1
            left_pub.publish(
                {
                    "data": left_rgb,
                    "timestamp_ns": ts,
                    "sequence": seq,
                    "width": width,
                    "height": height,
                }
            )
            if right_pub is not None:
                right_pub.publish(
                    {
                        "data": right_rgb,
                        "timestamp_ns": ts,
                        "sequence": seq,
                        "width": width,
                        "height": height,
                    }
                )
            if depth_pub is not None:
                depth_pub.publish(
                    {
                        "depth_values": depth_safe,
                        "timestamp_ns": ts,
                        "sequence": seq,
                        "width": width,
                        "height": height,
                    }
                )
            motion.after_images(ts, seq, runtime)
            photometric.update(ts)
            sequence_state["latest"] = seq
            for stream_stats in frame_stats.values():
                stream_stats["published"] = seq
                stream_stats["last_timestamp_ns"] = ts
            pub_count += 1

            now = time.perf_counter()

            # ── Optional self-verify: decode our own published frames ─────────
            left_msg = right_msg = depth_msg = None
            if verify:
                # get_latest is non-blocking; lags pub by ~1 frame.
                left_msg = left_sub.get_latest()
                right_msg = right_sub.get_latest() if right_sub is not None else None
                depth_msg = depth_sub.get_latest() if depth_sub is not None else None

                rgb_ready = left_msg is not None and (right_sub is None or right_msg is not None)
                depth_ready = depth_sub is None or depth_msg is not None
                if rgb_ready and depth_ready and not first_logged:
                    msg = (
                        f"First decoded frame — "
                        f"left {left_msg['data'].shape} {left_msg['data'].dtype}"
                    )
                    if right_msg is not None:
                        msg += f", right {right_msg['data'].shape} {right_msg['data'].dtype}"
                    assert left_msg["data"].shape == left_rgb.shape
                    if right_msg is not None:
                        assert right_msg["data"].shape == right_rgb.shape
                    if depth_msg is not None:
                        msg += (
                            f", depth {depth_msg['data'].shape} {depth_msg['data'].dtype}, "
                            f"finite={np.isfinite(depth_msg['data']).mean()*100:.1f}%"
                        )
                        assert depth_msg["data"].shape == depth_safe.shape
                        assert depth_msg["data"].dtype == np.float32
                    logger.info(msg)
                    logger.info("Round-trip shape/dtype check: OK")
                    first_logged = True

                if left_msg is not None:
                    sub_left_count += 1
                if right_msg is not None:
                    sub_right_count += 1
                if depth_msg is not None:
                    sub_depth_count += 1

            if now - t_window >= 1.0:
                dt = now - t_window
                line = f"pub: {pub_count / dt:5.1f} fps"
                if corrupted_count:
                    line += f"  | skipped {corrupted_count} CORRUPTED_FRAME"
                if verify:
                    line += (
                        f"  | sub L/R/D: {sub_left_count / dt:5.1f} / "
                        f"{sub_right_count / dt:5.1f} / {sub_depth_count / dt:5.1f} fps"
                    )
                logger.info(line)
                pub_count = sub_left_count = sub_right_count = sub_depth_count = 0
                corrupted_count = 0
                t_window = now

            if (
                save_root is not None
                and now - t_last_save >= 1.0
                and left_msg is not None
            ):
                idx = int(now - t_start)
                # decoded data is RGB; cv2.imwrite expects BGR
                cv2.imwrite(
                    str(save_root / f"left_rgb_{idx:04d}.jpg"),
                    cv2.cvtColor(left_msg["data"], cv2.COLOR_RGB2BGR),
                )
                if right_msg is not None:
                    cv2.imwrite(
                        str(save_root / f"right_rgb_{idx:04d}.jpg"),
                        cv2.cvtColor(right_msg["data"], cv2.COLOR_RGB2BGR),
                    )
                if depth_msg is not None:
                    depth_mm = np.clip(
                        depth_msg["data"] * 1000.0, 0, 65535
                    ).astype(np.uint16)
                    cv2.imwrite(str(save_root / f"depth_{idx:04d}.png"), depth_mm)
                    cv2.imwrite(
                        str(save_root / f"depth_vis_{idx:04d}.jpg"),
                        _depth_to_color(depth_msg["data"]),
                    )
                t_last_save = now

            if args.duration > 0 and now - t_start >= args.duration:
                break

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        motion.close()
        zed.close()
        clock_service.shutdown()
        info_service.shutdown()
        node.shutdown()


if __name__ == "__main__":
    main()
