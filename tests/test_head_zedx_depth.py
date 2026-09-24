#!/usr/bin/env python3
"""Head ZED-X-Mini publisher: stereo RGB (optionally depth) via SDK → Zenoh, cropped+resized.

Drop-in replacement for ``dexsensor launch --sensor head_camera`` that crops and
downsamples *before* the Zenoh encode, so only the small frames cross WiFi. The
head depth at full SVGA is ~58 Mbps and the dominant teleop-latency hog (it
saturates the robot→workstation WiFi link and queues the command path); cropping
to the manipulation ROI + downsampling to a tiny resolution cuts it to a couple
of Mbps. See ``tests/test_wrist_zedm_depth.py`` (the sibling wrist publisher) for
the same pattern; this one adds the ZED-X-Mini/GMSL open, depth, and crop+resize.

  pyzed (ZED X Mini, GMSL port 0)
     │
     ├─ grab → LEFT view + RIGHT view (+ DEPTH only if --enable-depth)
     │
     ├─ crop [top:bottom, left:right]  then resize → (resize_h, resize_w)
     │     RGB:   INTER_AREA   (anti-aliased downscale)
     │     DEPTH: INTER_NEAREST (never average across depth discontinuities /
     │            mix invalid 0/NaN pixels with valid ones)
     │
     ├─ publisher → sensors/<id>/left_rgb   (RGBImageCodec, RGB uint8)
     ├─ publisher → sensors/<id>/right_rgb  (RGBImageCodec, RGB uint8)
     ├─ publisher → sensors/<id>/depth      (DepthImageCodec, float32 m) [opt-in]
     ├─ publisher → sensors/<id>/pose       (JsonDataCodec, quat+trans) [--enable-tracking]
     ├─ service   → sensors/<id>/info       (JsonDataCodec)
     └─ service   → sensors/<id>/clock      (JsonDataCodec, publisher host clock)

``--enable-imu`` publishes full-rate inertial batches; ``--enable-tracking``
adds GEN_3 visual-inertial pose estimation and also enables IMU telemetry.
Both are opt-in. Pose estimation uses the full stereo input and can stall the
next capture even though the current RGB pair is published first. Live tests
failed the strict latency gate; use the diagnostics before enabling for recording.
Each tracker has an independent origin. Its pose is auxiliary data, not robot
base odometry or calibrated world-frame ground truth.

Topic naming matches the head_camera convention (left_rgb / right_rgb / depth)
so dexcontrol's ``ZedCameraSensor`` (what ``vr_reader.py`` polls via the Robot
API) subscribes to these exactly as it did to dexsensor's output — no vr_reader
change needed beyond the matching intrinsics (vr_reader.py ``ZED_K``).

IMPORTANT — only one process can hold the camera. Stop dexsensor's head_camera
before running this::

    dexsensor stop head_camera   # or just don't launch --sensor head_camera

This is a long-running dependency, *not* a one-shot test — start it and leave it
running for the whole vr_reader session (no ``--duration``).

Usage::

    # production: low-latency 30 Hz capture, stereo RGB, default crop/resize
    python tests/test_head_zedx_depth.py
    # add the SDK depth stream
    python tests/test_head_zedx_depth.py --enable-depth
    # debug round-trip fps / inspect cropped frames
    python tests/test_head_zedx_depth.py --verify --save-dir ~/Dexmate
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

sys.path.insert(0, (pathlib.Path(__file__).resolve().parent.parent / "src").as_posix())
from omniteleop.common.head_camera import (  # noqa: E402
    HEAD_CROP_TBLR,
    HEAD_RESIZE_HW,
    HEAD_RESOLUTION,
    crop_resize_intrinsics,
)

from omniteleop.common.zed_motion import ZedMotion

Resolution = Literal["SVGA", "HD1080", "HD1200"]
DepthMode = Literal["NEURAL", "NEURAL_LIGHT", "ULTRA", "QUALITY", "PERFORMANCE"]

# The head ZED X Mini, pinned by serial. Selecting by index (set_from_camera_id)
# resolves against the SDK's whole device list, so any other ZED on the machine --
# including a wrist ZED-M that has dropped off USB3 and lingers as a dead
# sn=0/NOT AVAILABLE entry -- can occupy slot 0 and get opened instead of the head.
# That surfaces as a bare CAMERA STREAM FAILED TO START with nothing wrong with the
# head at all. The serial is unambiguous. Update this if the head camera is replaced.
_HEAD_SERIAL: int = 50571637

# Camera payloads must yield to VR/actuator traffic if any link is congested.
# Best-effort + drop preserves the existing latest-frame semantics; data_low
# prevents large JPEG/PNG fragments from sharing control's interactive priority.
_CAMERA_QOS = {
    "reliability": "best_effort",
    "congestion_control": "drop",
    "priority": "data_low",
}


@dataclasses.dataclass
class Args:
    """CLI options for the standalone head ZED-X-Mini publisher."""

    sensor_id: str = "head_camera"
    """Sensor id used to derive topic names: sensors/<sensor_id>/{left_rgb,
    right_rgb,depth,info}. Must match what dexcontrol/vr_reader subscribes to
    — keep it 'head_camera'."""

    namespace: str = ""
    """Zenoh namespace passed to dexcomm.Node (matches vr_reader.py)."""

    resolution: Resolution = HEAD_RESOLUTION  # type: ignore[assignment]
    """ZED capture resolution. Default = omniteleop.common.head_camera.HEAD_RESOLUTION
    (HD1200); SVGA (960x600) is the SDK floor for ZED X Mini. The crop below is
    validated against the frame this resolution actually delivers."""

    rate: int = 30
    """Camera fps. The ZED X Mini at SVGA only supports a discrete set
    (15/30/60/120); any other value silently rounds DOWN to the nearest
    supported rate (e.g. 20 -> 15). Default 30 is intentional: live six-stream
    measurements at full 600x960 found 44--52 ms head arrival age versus
    144--150 ms at 15 Hz. The resulting aggregate camera uplink was ~29 Mbps,
    and bounded head history still supplies coherent 10 Hz rows with 15 Hz wrists."""

    depth_mode: DepthMode = "NEURAL"

    enable_depth: bool = False
    """Compute + publish depth. Off by default — the SDK depth pass is the
    dominant grab-loop cost and the frames are the bandwidth hog, and stereo
    consumers reconstruct depth themselves from left+right. Turn on for
    consumers that want the SDK's depth directly."""

    skip_right_rgb: bool = False
    """Skip the right_rgb stream. Off by default — the pair is published so
    downstream stereo (FoundationStereo, via the meta/head_stereo fx+baseline)
    has both views. Turn on to halve RGB bandwidth when only left is consumed."""

    depth_min: float = 0.1
    """Minimum depth in metres (matches the dexsensor head_camera config)."""

    depth_max: float = 8.0
    """Maximum depth in metres (matches the dexsensor head_camera config)."""

    crop: tuple[int, int, int, int] = HEAD_CROP_TBLR
    """Crop applied to every stream as (top, bottom, left, right) pixel indices on
    the raw capture frame (HEAD_RESOLUTION), i.e. img[top:bottom, left:right]. Default = the shared
    omniteleop.common.head_camera.HEAD_CROP_TBLR — the single source of truth for
    crop geometry + ZED_K (vr_reader / policy_rollout). Override on the CLI for a
    one-off without touching the consumers."""

    resize_h: int = HEAD_RESIZE_HW[0]
    """Output height after the crop (default = head_camera.HEAD_RESIZE_HW[0]). Set
    to 0 (with resize_w=0) to skip resize."""

    resize_w: int = HEAD_RESIZE_HW[1]
    """Output width after the crop (default = head_camera.HEAD_RESIZE_HW[1]). Set
    to 0 (with resize_h=0) to skip resize."""

    enable_imu: bool = False
    """Publish timestamped full-rate IMU batches without visual pose estimation."""

    enable_tracking: bool = False
    """Enable ZED positional tracking (VSLAM+IMU) and publish the camera pose
    on sensors/<id>/pose each grab as quaternion_xyzw + translation. Off by
    default — zero change to the image/depth path when off. The pose is the
    camera pose in the SDK tracking world, camera frame = optical convention
    (COORDINATE_SYSTEM.IMAGE); consume it with
    legacy/VSLAM/scripts/zed_base_pose_node.py to get the robot base pose."""

    tracking_mode: Literal["GEN_1", "GEN_2", "GEN_3"] = "GEN_3"
    """GEN_3 supports tracking without depth and image-first processing."""

    area_memory: bool = False
    """SDK enable_area_memory: drift correction via relocalization/loop closure
    (pose may jump on closure). Disable for smooth pure-odometry output."""

    area_file: Optional[str] = None
    """Optional ZED .area map to load before enabling positional tracking. Use
    this for relocalization against a prebuilt map."""

    localization_only: bool = False
    """require --area-file is set; 
    if not set, will update memory during running but will not overwrite original .area file on disk"""

    verify: bool = False
    """Subscribe to our own topics each frame to print round-trip fps. Off by
    default — decoding every frame in-process throttles the publisher."""

    duration: float = 0.0
    """If > 0, stop after this many seconds. Leave at 0 for normal use."""

    save_dir: Optional[str] = None
    """If set (implies --verify), save the LAST decoded frame received by the
    subscriber once per second, proving the round-trip preserves data."""


def _configure_zenoh_like_dexcontrol() -> Optional[pathlib.Path]:
    """Mirror dexcontrol's default ZENOH_CONFIG discovery (see wrist publisher)."""
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


def _validate_crop(crop: tuple[int, int, int, int], width: int, height: int) -> None:
    """Aggressively validate the crop window against the actual frame size."""
    top, bottom, left, right = crop
    if not (0 <= top < bottom <= height and 0 <= left < right <= width):
        raise ValueError(
            f"crop (top,bottom,left,right)={crop} is invalid for a {width}x{height} "
            f"(WxH) frame: need 0<=top<bottom<={height} and 0<=left<right<={width}."
        )


def _crop_resize(
    img: np.ndarray,
    crop: tuple[int, int, int, int],
    out_hw: Optional[tuple[int, int]],
    interpolation: int,
) -> np.ndarray:
    """Crop img[top:bottom, left:right], then resize to (h, w) if out_hw is set.

    out_hw is (height, width); cv2.resize wants (width, height). Works for both
    HxWx3 RGB and HxW float32 depth.
    """
    top, bottom, left, right = crop
    cropped = img[top:bottom, left:right]
    if out_hw is None:
        return np.ascontiguousarray(cropped)
    h, w = out_hw
    return cv2.resize(cropped, (w, h), interpolation=interpolation)


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


def main() -> None:
    args = tyro.cli(Args)
    verify = args.verify or args.save_dir is not None
    zenoh_config = _configure_zenoh_like_dexcontrol()
    area_path: Optional[pathlib.Path] = (
        pathlib.Path(args.area_file).expanduser().resolve() if args.area_file else None
    )

    if (args.resize_h > 0) != (args.resize_w > 0):
        raise ValueError(
            f"resize_h ({args.resize_h}) and resize_w ({args.resize_w}) must both "
            "be >0 (resize) or both 0 (skip resize)."
        )
    if area_path is not None:
        if not args.enable_tracking:
            raise ValueError("--area-file requires --enable-tracking")
        if not area_path.is_file():
            raise FileNotFoundError(f"--area-file does not exist: {area_path}")
        if not args.area_memory:
            raise ValueError("--area-file requires --area-memory; it is the SDK map/relocalization path")
    if args.localization_only and area_path is None:
        raise ValueError("--localization-only requires --area-file")
    out_hw: Optional[tuple[int, int]] = (
        (args.resize_h, args.resize_w) if args.resize_h > 0 else None
    )

    # ── 1. Open ZED X Mini via SDK on the requested GMSL port ─────────────────
    depth_mode = (
        getattr(sl.DEPTH_MODE, args.depth_mode) if args.enable_depth else sl.DEPTH_MODE.NONE
    )
    init = sl.InitParameters(
        camera_resolution=getattr(sl.RESOLUTION, args.resolution),
        camera_fps=args.rate,
        depth_mode=depth_mode,
        coordinate_system=sl.COORDINATE_SYSTEM.IMAGE,
        coordinate_units=sl.UNIT.METER,
        depth_minimum_distance=args.depth_min,
        depth_maximum_distance=args.depth_max,
    )
    init.set_from_serial_number(_HEAD_SERIAL)
    # Pin the FACTORY calibration. With self-calibration on, the SDK re-estimates fx at
    # every open() (measured 0.145% spread across three opens of this unit), so the
    # published left_K is not reproducible across publisher restarts and drifts past the
    # 0.107% tolerance of wbc_vr_robot_tmp's meta/head_stereo check. See
    # omniteleop.common.head_camera.HEAD_BASE_K, which stores the pinned values.
    init.camera_disable_self_calib = True
    runtime = sl.RuntimeParameters()
    zed = sl.Camera()
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(
            f"ZED open failed: {err}. Is dexsensor still holding the head_camera "
            f"(serial {_HEAD_SERIAL})? Stop it first."
        )

    info = zed.get_camera_information()
    logger.info(
        f"Camera: {info.camera_model} sn={info.serial_number} "
        f"fw={info.camera_configuration.firmware_version}"
    )

    cam_res = info.camera_configuration.resolution
    raw_w, raw_h = cam_res.width, cam_res.height
    logger.info(f"Raw capture resolution: {raw_w}x{raw_h}")
    _validate_crop(args.crop, raw_w, raw_h)

    top, bottom, left, right = args.crop
    crop_h, crop_w = bottom - top, right - left
    out_h, out_w = (out_hw if out_hw is not None else (crop_h, crop_w))
    logger.info(
        f"Pipeline: raw {raw_w}x{raw_h} → crop [{top}:{bottom},{left}:{right}] "
        f"({crop_w}x{crop_h}) → resize {out_w}x{out_h} (WxH)"
    )

    # ── 1c. Stereo calibration for the PUBLISHED frames ───────────────────────
    # FoundationStereo needs (left, right, fx, baseline) to turn disparity into
    # metric depth. `calibration_parameters` (not `_raw`) is the RECTIFIED pair,
    # which is what VIEW.LEFT/VIEW.RIGHT return, and left/right go through the
    # SAME _crop_resize below -- so the rows stay epipolar-aligned and only the
    # intrinsics need adjusting. The baseline is a physical distance and does
    # not scale with crop/resize.
    _calib = info.camera_configuration.calibration_parameters
    _baseline_m = float(_calib.get_camera_baseline())
    if not 0.01 < _baseline_m < 0.5:
        raise SystemExit(
            f"implausible ZED stereo baseline {_baseline_m} m; coordinate_units is "
            f"METER, so this should be the physical inter-lens distance"
        )

    def _sdk_k(cam) -> np.ndarray:
        return np.array([[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    _stereo = {
        "baseline_m": _baseline_m,
        "rectified": True,
        "left_K": crop_resize_intrinsics(
            _sdk_k(_calib.left_cam), args.crop, (out_h, out_w)).tolist(),
        "right_K": crop_resize_intrinsics(
            _sdk_k(_calib.right_cam), args.crop, (out_h, out_w)).tolist(),
    }
    logger.info(
        f"Stereo calibration: baseline {_baseline_m * 1000:.2f} mm, "
        f"left fx {_stereo['left_K'][0][0]:.2f} px (published geometry)"
    )

    # ── 2. Set up Zenoh pub/sub via dexcomm ───────────────────────────────────
    left_topic = f"sensors/{args.sensor_id}/left_rgb"
    right_topic = f"sensors/{args.sensor_id}/right_rgb"
    depth_topic = f"sensors/{args.sensor_id}/depth"
    pose_topic = f"sensors/{args.sensor_id}/pose"
    info_topic = f"sensors/{args.sensor_id}/info"
    clock_topic = f"sensors/{args.sensor_id}/clock"

    node = Node(name="head_zedx_pub", namespace=args.namespace)
    logger.info(
        f"Zenoh config: {zenoh_config if zenoh_config is not None else '<dexcomm default>'}"
    )
    logger.info(f"ROBOT_NAME={os.getenv('ROBOT_NAME')!r}; node.namespace={node.namespace!r}")
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
        "pose": {"published": 0, "last_timestamp_ns": 0},
    }
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
            "actual": {"width": int(out_w), "height": int(out_h), "fps": int(args.rate)},
            # Consumed by wbc_vr_robot_tmp.py --head-right-rgb so a stereo take
            # carries everything FoundationStereo needs (fx, baseline) without a
            # hardcoded constant anywhere downstream.
            "stereo": _stereo,
            "configured": {
                "resolution": args.resolution,
                "depth_mode": args.depth_mode,
                "depth_min": float(args.depth_min),
                "depth_max": float(args.depth_max),
                "crop": list(args.crop),
                "resize_hw": [int(out_h), int(out_w)],
                "tracking": {
                    "enabled": args.enable_tracking,
                    "mode": args.tracking_mode,
                    "area_memory": args.area_memory,
                    "area_file_path": area_path.as_posix() if area_path else None,
                    "localization_only": args.localization_only,
                    "pose_fields": [
                        "quaternion_xyzw",
                        "translation",
                        "state",
                        "confidence",
                        "timestamp_ns",
                        "zed_timestamp_ns",
                        "sequence",
                    ],
                    "translation_unit": "m",
                    "coordinate_system": "IMAGE_optical",
                },
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
                "pose": {
                    "enabled": args.enable_tracking,
                    "transport": "zenoh",
                    "topic": pose_topic,
                    "fields": ["quaternion_xyzw", "translation"],
                },
            },
            "statistics": frame_stats,
            "motion": motion.info(),
        }

    def _clock(_request: bytes | None = None) -> dict:
        return _clock_response(args.sensor_id, sequence_state["latest"])

    info_service = node.create_service(
        info_topic, _camera_info, response_encoder=JsonDataCodec.encode
    )
    clock_service = node.create_service(
        clock_topic, _clock, response_encoder=JsonDataCodec.encode
    )
    pub_topics = f"'{left_topic}'"
    if right_pub is not None:
        pub_topics += f", '{right_topic}'"
    if args.enable_depth:
        pub_topics += f", '{depth_topic}'"
    if motion.tracking_enabled:
        pub_topics += f", '{pose_topic}'"
    logger.info(f"Publishing on {pub_topics}  (depth={'on' if args.enable_depth else 'off'})")
    logger.info(f"Serving camera info on '{info_topic}'")
    logger.info(f"Serving publisher clock on '{clock_topic}'")
    resolved = f"left={node.resolve_topic(left_topic)!r}"
    if right_pub is not None:
        resolved += f", right={node.resolve_topic(right_topic)!r}"
    if args.enable_depth:
        resolved += f", depth={node.resolve_topic(depth_topic)!r}"
    if motion.tracking_enabled:
        resolved += f", pose={node.resolve_topic(pose_topic)!r}"
    resolved += f", info={node.resolve_topic(info_topic)!r}"
    logger.info(f"Resolved head keys: {resolved}")
    if verify:
        logger.info("Subscribing to the same topics to verify round-trip")

    save_root: Optional[pathlib.Path] = None
    if args.save_dir:
        save_root = pathlib.Path(args.save_dir) / args.sensor_id
        save_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving 1 frame/s (post-subscriber decode) to {save_root}")

    # ── 3. Capture → crop+resize → publish → (optional) subscribe loop ────────
    left_mat = sl.Mat()
    right_mat = sl.Mat()
    depth_mat = sl.Mat()

    seq = 0
    pub_count = 0
    sub_left_count = sub_right_count = sub_depth_count = 0
    t_window = time.perf_counter()
    t_start = t_window
    t_last_save = 0.0
    first_logged = False

    try:
        while True:
            if motion.read(runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            # SDK timestamp of the grabbed image. This rides the codec-preserved
            # timestamp_ns field; extra payload keys are dropped by RGBImageCodec.
            ts = _zed_image_timestamp_ns(zed)

            zed.retrieve_image(left_mat, sl.VIEW.LEFT)
            # ZED returns BGRA — convert to RGB (RGBImageCodec / vr_reader expect RGB).
            left_full = cv2.cvtColor(left_mat.get_data(), cv2.COLOR_BGRA2RGB)
            left_rgb = _crop_resize(left_full, args.crop, out_hw, cv2.INTER_AREA)

            seq += 1
            # Publish the policy-consumed RGB immediately. NEURAL depth retrieval can take
            # most of a camera frame; putting RGB behind it makes the workstation select
            # an older head image than the wrist image at record ticks.
            left_pub.publish(
                {"data": left_rgb, "timestamp_ns": ts, "sequence": seq,
                 "width": out_w, "height": out_h}
            )
            sequence_state["latest"] = seq
            frame_stats["left_rgb"]["published"] = seq
            frame_stats["left_rgb"]["last_timestamp_ns"] = ts

            right_rgb = None
            if right_pub is not None:
                zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
                right_full = cv2.cvtColor(right_mat.get_data(), cv2.COLOR_BGRA2RGB)
                right_rgb = _crop_resize(right_full, args.crop, out_hw, cv2.INTER_AREA)
                right_pub.publish(
                    {"data": right_rgb, "timestamp_ns": ts, "sequence": seq,
                     "width": out_w, "height": out_h}
                )

            motion.after_images(ts, seq, runtime)

            depth_safe = None
            if args.enable_depth:
                zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                depth_full = np.ascontiguousarray(depth_mat.get_data())  # float32 m, may hold NaN
                # NEAREST so we never blend valid metres with NaN/Inf invalid pixels.
                depth_crop = _crop_resize(depth_full, args.crop, out_hw, cv2.INTER_NEAREST)
                depth_safe = np.nan_to_num(
                    depth_crop, nan=0.0, posinf=0.0, neginf=0.0
                ).astype(np.float32)

            if depth_pub is not None:
                depth_pub.publish(
                    {"depth_values": depth_safe, "timestamp_ns": ts, "sequence": seq,
                     "width": out_w, "height": out_h}
                )
            if right_pub is not None:
                frame_stats["right_rgb"]["published"] = seq
                frame_stats["right_rgb"]["last_timestamp_ns"] = ts
            if depth_pub is not None:
                frame_stats["depth"]["published"] = seq
                frame_stats["depth"]["last_timestamp_ns"] = ts
            if motion.tracking_enabled:
                frame_stats["pose"]["published"] = motion.stats["published"]
                frame_stats["pose"]["last_timestamp_ns"] = motion.stats.get("last_published_pose_ns", 0)
            pub_count += 1

            now = time.perf_counter()

            left_msg = right_msg = depth_msg = None
            if verify:
                left_msg = left_sub.get_latest()
                right_msg = right_sub.get_latest() if right_sub is not None else None
                depth_msg = depth_sub.get_latest() if depth_sub is not None else None

                rgb_ready = left_msg is not None and (right_sub is None or right_msg is not None)
                depth_ready = depth_sub is None or depth_msg is not None
                if rgb_ready and depth_ready and not first_logged:
                    msg = f"First decoded frame — left {left_msg['data'].shape} {left_msg['data'].dtype}"
                    assert left_msg["data"].shape == left_rgb.shape
                    if right_msg is not None:
                        msg += f", right {right_msg['data'].shape}"
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
                if verify:
                    line += (
                        f"  | sub L/R/D: {sub_left_count / dt:5.1f} / "
                        f"{sub_right_count / dt:5.1f} / {sub_depth_count / dt:5.1f} fps"
                    )
                logger.info(line)
                pub_count = sub_left_count = sub_right_count = sub_depth_count = 0
                t_window = now

            if save_root is not None and now - t_last_save >= 1.0 and left_msg is not None:
                idx = int(now - t_start)
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
                    depth_mm = np.clip(depth_msg["data"] * 1000.0, 0, 65535).astype(np.uint16)
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
        if args.enable_tracking:
            zed.disable_positional_tracking()
        zed.close()
        clock_service.shutdown()
        info_service.shutdown()
        node.shutdown()


if __name__ == "__main__":
    main()
