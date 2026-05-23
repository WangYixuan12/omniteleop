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
     └─ service   → sensors/<id>/info       (JsonDataCodec)

Topic naming matches the head_camera convention (left_rgb / right_rgb /
depth) so vr_reader's ``ZedCameraSensor`` subscribes to wrist streams the
same way it does head streams.

RGB-only by default: vr_reader records the wrist RGB-only, and the NEURAL
stereo matcher + 1.8 MB depth frames throttle the capture loop hard. Run with
``--enable-depth`` only when you actually need depth.

This is a long-running dependency, *not* a one-shot test — start it and leave
it running for the whole vr_reader session (no ``--duration``). Confirm it is
reaching the workstation with ``python scripts/probe_wrist_stream.py`` there.

Important: only one process can hold the V4L2 device. Stop
``dexsensor launch ... --sensor wrist_zedm`` before running.

Usage::

    # production: run until Ctrl-C, RGB-only
    python tests/test_wrist_zedm_depth.py
    # debug round-trip fps / depth
    python tests/test_wrist_zedm_depth.py --verify --enable-depth --save-dir ~/Dexmate
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import time
from typing import Literal, Optional

import cv2
import numpy as np
import pyzed.sl as sl
import tyro
from dexcomm import Node
from dexcomm.codecs import DepthImageCodec, JsonDataCodec, RGBImageCodec
from loguru import logger

Resolution = Literal["HD720", "HD1080", "HD2K", "VGA"]
DepthMode = Literal["NEURAL", "NEURAL_LIGHT", "ULTRA", "QUALITY", "PERFORMANCE"]


@dataclasses.dataclass
class Args:
    """CLI options for the standalone wrist ZED-M publisher."""

    sensor_id: str = "wrist_zedm"
    """Sensor id used to derive topic names: sensors/<sensor_id>/{rgb,depth}.
    Must match what vr_reader.py / dexcontrol will subscribe to."""

    namespace: str = ""
    """Zenoh namespace passed to dexcomm.Node (matches vr_reader.py:257)."""

    resolution: Resolution = "HD720"
    """ZED capture resolution."""

    enable_depth: bool = False
    """Compute + publish depth. Off by default — vr_reader uses the wrist
    RGB-only, and the stereo matcher throttles the capture loop. Turn on only
    to debug depth."""

    depth_mode: DepthMode = "NEURAL"
    """Depth estimation algorithm (only used when --enable-depth). NEURAL is
    best quality but heaviest."""

    rate: int = 30
    """Camera fps."""

    depth_min: float = 0.3
    """Minimum depth in metres."""

    depth_max: float = 6.0
    """Maximum depth in metres."""

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


def main() -> None:
    args = tyro.cli(Args)
    verify = args.verify or args.save_dir is not None
    zenoh_config = _configure_zenoh_like_dexcontrol()

    # ── 1. Open ZED-M via SDK ─────────────────────────────────────────────────
    # DEPTH_MODE.NONE skips the stereo matcher entirely — that is what keeps the
    # RGB-only capture loop at full fps.
    depth_mode = (
        getattr(sl.DEPTH_MODE, args.depth_mode)
        if args.enable_depth
        else sl.DEPTH_MODE.NONE
    )
    init = sl.InitParameters(
        camera_resolution=getattr(sl.RESOLUTION, args.resolution),
        camera_fps=args.rate,
        depth_mode=depth_mode,
        coordinate_units=sl.UNIT.METER,
        depth_minimum_distance=args.depth_min,
        depth_maximum_distance=args.depth_max,
    )
    runtime = sl.RuntimeParameters()
    zed = sl.Camera()
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(
            f"ZED open failed: {err}. "
            f"Is dexsensor still holding /dev/video10?"
        )

    info = zed.get_camera_information()
    logger.info(
        f"Camera: {info.camera_model} sn={info.serial_number} "
        f"fw={info.camera_configuration.firmware_version}"
    )

    cam_res = info.camera_configuration.resolution
    width, height = cam_res.width, cam_res.height
    logger.info(f"Stream resolution: {width}x{height}")

    # ── 2. Set up Zenoh pub/sub via dexcomm ──────────────────────────────────
    left_topic = f"sensors/{args.sensor_id}/left_rgb"
    right_topic = f"sensors/{args.sensor_id}/right_rgb"
    depth_topic = f"sensors/{args.sensor_id}/depth"

    node = Node(name="wrist_zedm_test", namespace=args.namespace)
    logger.info(
        f"Zenoh config: {zenoh_config if zenoh_config is not None else '<dexcomm default>'}"
    )
    logger.info(
        f"ROBOT_NAME={os.getenv('ROBOT_NAME')!r}; node.namespace={node.namespace!r}"
    )
    left_pub = node.create_publisher(left_topic, encoder=RGBImageCodec.encode)
    right_pub = node.create_publisher(right_topic, encoder=RGBImageCodec.encode)
    depth_pub = (
        node.create_publisher(depth_topic, encoder=DepthImageCodec.encode)
        if args.enable_depth
        else None
    )
    # Self-verify subscribers are opt-in: decoding every frame in-process
    # throttles the publisher and starves the info queryable.
    left_sub = right_sub = depth_sub = None
    if verify:
        left_sub = node.create_subscriber(left_topic, decoder=RGBImageCodec.decode)
        right_sub = node.create_subscriber(right_topic, decoder=RGBImageCodec.decode)
        if args.enable_depth:
            depth_sub = node.create_subscriber(depth_topic, decoder=DepthImageCodec.decode)

    frame_stats = {
        "left_rgb": {"published": 0, "last_timestamp_ns": 0},
        "right_rgb": {"published": 0, "last_timestamp_ns": 0},
        "depth": {"published": 0, "last_timestamp_ns": 0},
    }
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
            "configured": {
                "resolution": args.resolution,
                "depth_mode": args.depth_mode,
                "depth_min": float(args.depth_min),
                "depth_max": float(args.depth_max),
            },
            "streams": {
                "left_rgb": {"enabled": True, "transport": "zenoh", "topic": left_topic},
                "right_rgb": {"enabled": True, "transport": "zenoh", "topic": right_topic},
                "depth": {"enabled": args.enable_depth, "transport": "zenoh", "topic": depth_topic},
            },
            "statistics": frame_stats,
        }

    info_topic = f"sensors/{args.sensor_id}/info"
    info_service = node.create_service(
        info_topic,
        _camera_info,
        response_encoder=JsonDataCodec.encode,
    )
    pub_topics = f"'{left_topic}', '{right_topic}'"
    if args.enable_depth:
        pub_topics += f", '{depth_topic}'"
    logger.info(f"Publishing on {pub_topics}  (depth={'on' if args.enable_depth else 'off'})")
    logger.info(f"Serving camera info on '{info_topic}'")
    logger.info(
        "Resolved wrist keys: "
        f"left={node.resolve_topic(left_topic)!r}, "
        f"right={node.resolve_topic(right_topic)!r}, "
        f"info={node.resolve_topic(info_topic)!r}"
    )
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

    pub_count = 0
    sub_left_count = 0
    sub_right_count = 0
    sub_depth_count = 0
    t_window = time.perf_counter()
    t_start = t_window
    t_last_save = 0.0
    first_logged = False

    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(left_mat, sl.VIEW.LEFT)
            zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
            # ZED returns BGRA — convert to RGB (the RGBImageCodec / vr_reader /
            # rerun all expect RGB; publishing BGR swaps the red/blue channels).
            left_rgb = cv2.cvtColor(left_mat.get_data(), cv2.COLOR_BGRA2RGB)
            right_rgb = cv2.cvtColor(right_mat.get_data(), cv2.COLOR_BGRA2RGB)

            depth_safe = None
            if args.enable_depth:
                zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                depth_m = np.ascontiguousarray(depth_mat.get_data())
                # Replace NaN/Inf so DepthImageCodec doesn't choke on them
                depth_safe = np.nan_to_num(
                    depth_m, nan=0.0, posinf=0.0, neginf=0.0
                ).astype(np.float32)

            ts = time.time_ns()
            seq = pub_count + 1
            left_pub.publish(
                {
                    "data": left_rgb,
                    "timestamp_ns": ts,
                    "sequence": seq,
                    "width": width,
                    "height": height,
                }
            )
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
                right_msg = right_sub.get_latest()
                depth_msg = depth_sub.get_latest() if depth_sub is not None else None

                rgb_ready = left_msg is not None and right_msg is not None
                depth_ready = depth_sub is None or depth_msg is not None
                if rgb_ready and depth_ready and not first_logged:
                    msg = (
                        f"First decoded frame — "
                        f"left {left_msg['data'].shape} {left_msg['data'].dtype}, "
                        f"right {right_msg['data'].shape} {right_msg['data'].dtype}"
                    )
                    assert left_msg["data"].shape == left_rgb.shape
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

            if (
                save_root is not None
                and now - t_last_save >= 1.0
                and left_msg is not None
                and right_msg is not None
            ):
                idx = int(now - t_start)
                # decoded data is RGB; cv2.imwrite expects BGR
                cv2.imwrite(
                    str(save_root / f"left_rgb_{idx:04d}.jpg"),
                    cv2.cvtColor(left_msg["data"], cv2.COLOR_RGB2BGR),
                )
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
        zed.close()
        info_service.shutdown()
        node.shutdown()


if __name__ == "__main__":
    main()
