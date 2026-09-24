#!/usr/bin/env python3
"""Handheld D455 walk-around scan of the room, for the background atlas.

This is the capture end of the background pipeline: FoundationStereo turns each
stored IR pair into metric depth (the ~95 mm factory baseline fixes the scale),
RGB-D odometry chains the frames into one trajectory, a TSDF fuses them, and the
fused atlas is registered ONCE into the camera_0 table world. Compared to a phone
splat the scale is guaranteed, the depth is the same 1.2 mm-rms FoundationStereo
depth as every other camera in the pipeline, and the coverage is whatever you walk.

Writes the record_pw_episode/record_static_cams ``camera_0`` layout so the FS
tooling reads it unchanged -- EXCEPT that ``camera_0_extrinsics`` is deliberately
absent. That dataset means "cam -> world, known per frame"; a handheld scan has no
known pose until the fusion solves one, and writing identity would claim a camera
that never moved. The scan fuser reads images + K + baseline directly.

Every delivered frameset is stored (30 fps, no schedule, no decimation): a scan
wants maximum overlap, and the fuser keyframes for itself. Budget ~4.9 GB/min.

The IR emitter stays ON at max power: the projected dots are the only stereo
texture the bare white walls have, and both IR imagers see the same dots per
exposure, so camera motion does not smear the pattern within a frame.

Colour exposure, gain and white balance are LOCKED to the explicit values in
--camera-profile (same bring-up as record_arm_scan.py's handheld), so a surface keeps
one appearance across the whole walk. They cannot be metered instead: with auto
exposure on, the D455 reports neither the exposure nor the gain it chose, and turning
auto off restores the stored manual values -- measured 2026-09-23, a "metered" lock
came out at the factory 166/64/4600 K and clipped half the frame.

The per-second status line prints a SHARPNESS number (mean IR gradient). It sits
at a steady value while you move gently and dips when you turn too fast -- watch
it, and slow down when it drops. CLIP is the share of colour pixels with a channel
at >= 250; the locked exposure cannot adapt, so a high value means lost highlights.

usage (dexmate2 env, camera on a USB3 cable):
    python scripts/record_scan.py --out-dir /data/scans/room_08_22 --serial 239222300740 \
        --camera-profile configs/camera_profiles/scan_mobile_20260923.json

Ctrl-C ends the take and finalises the file. Episodes auto-increment.
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_arm_scan import open_d455  # noqa: E402  (also puts src/ on sys.path)
from omniteleop.common import realsense_capture_meta as rsmeta  # noqa: E402

IR_W, IR_H = 1280, 720
COLOR_W, COLOR_H = 1280, 800
FPS = 30
SETTLE_S = 2.0
QUEUE_DEPTH = 64        # ~2 s of backlog before we call the disk too slow
MB_PER_FRAME = 2.7      # measured with lzf on this scene (IR pair + colour)


def pick_device(serial: str | None) -> str:
    devs = {d.get_info(rs.camera_info.serial_number): d for d in rs.context().query_devices()}
    if not devs:
        raise ValueError("no RealSense device connected")
    if serial is not None:
        if serial not in devs:
            raise ValueError(f"serial {serial} not connected; present: {sorted(devs)}")
        chosen = serial
    elif len(devs) == 1:
        chosen = next(iter(devs))
    else:
        raise ValueError(
            f"{len(devs)} devices connected ({sorted(devs)}); unplug the rig camera "
            f"you are NOT holding, or pass --serial")
    usb = devs[chosen].get_info(rs.camera_info.usb_type_descriptor)
    if not usb.startswith("3"):
        raise ValueError(
            f"{chosen} is on USB {usb}; the IR pair + colour at 30 fps needs USB 3 -- "
            f"use the blue-tab cable/port")
    return chosen


class ScanWriter(threading.Thread):
    """Appends framesets to the HDF5 off the capture thread's critical path."""

    def __init__(self, path: Path, q: queue.Queue, compression: str):
        super().__init__(daemon=True)
        self.path, self.q, self.error = path, q, None
        self.n = 0
        self.f = h5py.File(path, "w")
        g = self.f.create_group("observations/images")
        m = self.f.create_group("capture_meta")
        shapes = {
            "camera_0_left_ir": ((IR_H, IR_W), np.uint8, g),
            "camera_0_right_ir": ((IR_H, IR_W), np.uint8, g),
            "camera_0_color": ((COLOR_H, COLOR_W, 3), np.uint8, g),
            "camera_0_intrinsics": ((3, 3), np.float32, g),
            "camera_0_rgb_intrinsics": ((3, 3), np.float32, g),
            "camera_0_rgb_to_ir_extrinsic": ((4, 4), np.float32, g),
            "camera_0_baseline": ((), np.float32, g),
            "camera_0_host_ns": ((), np.int64, m),
            "camera_0_timestamp_ms": ((), np.float64, m),
            "camera_0_frame_numbers": ((3,), np.int64, m),
        }
        for suffix, shape, dtype in rsmeta.specs(3):
            shapes[f"camera_0_{suffix}"] = (shape, dtype, m)
        rsmeta.describe(m, "camera_0", ["left_ir", "right_ir", "color"])
        comp = None if compression == "none" else compression
        self.ds = {}
        for name, (shp, dt, grp) in shapes.items():
            self.ds[name] = grp.create_dataset(
                name, shape=(0,) + shp, maxshape=(None,) + shp, dtype=dt,
                chunks=(1,) + shp if shp else (4096,),
                compression=comp if shp and len(shp) >= 2 else None)

    def run(self):
        try:
            while True:
                rec = self.q.get()
                if rec is None:
                    return
                i = self.n
                for name, val in rec.items():
                    d = self.ds[name]
                    d.resize(i + 1, axis=0)
                    d[i] = val
                self.n += 1
        except Exception as exc:                      # surfaced by the main loop
            self.error = f"{type(exc).__name__}: {exc}"

    def close(self, attrs: dict):
        if self.is_alive():
            self.q.put(None, timeout=5.0)
            self.join(timeout=30.0)
        if self.is_alive() or self.error:
            raise RuntimeError(f"Room scan writer incomplete: {self.error}")
        self.f.attrs["num_frames"] = self.n
        for k, v in attrs.items():
            self.f.attrs[k] = v
        self.f.close()


class MjpegServer(threading.Thread):
    """Serves the latest JPEG as a multipart stream so a browser can show it live."""

    def __init__(self, port: int):
        super().__init__(daemon=True)
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.jpeg, self.cond = b"", threading.Condition()
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path != "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(b'<body style="margin:0;background:#000">'
                                     b'<img src="/stream" style="width:100%"></body>')
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                last = None
                try:
                    while True:
                        with srv.cond:
                            srv.cond.wait_for(lambda: srv.jpeg is not last, timeout=1.0)
                            last = srv.jpeg
                        if not last:
                            continue
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(last)).encode() + b"\r\n\r\n" + last + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), H)

    def run(self):
        self.httpd.serve_forever()

    def push(self, jpeg: bytes):
        with self.cond:
            self.jpeg = jpeg
            self.cond.notify_all()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--serial", default=None,
                    help="device to use; required when more than one is connected")
    ap.add_argument("--camera-profile", type=Path, required=True,
                    help="JSON keyed by D455 serial, SDK option units; must give this camera's "
                         "color_exposure_option, color_gain and white_balance")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds; default: record until Ctrl-C")
    ap.add_argument("--compression", choices=["lzf", "gzip", "none"], default="lzf")
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    ap.add_argument("--stream", action="store_true",
                    help="serve a live MJPEG preview (colour | left IR | ASIC depth) over HTTP")
    ap.add_argument("--stream-port", type=int, default=8765)
    args = ap.parse_args()
    if args.stream:
        import cv2
        preview = MjpegServer(args.stream_port)
        print(f"preview   http://localhost:{args.stream_port}/  "
              f"(ssh -L {args.stream_port}:localhost:{args.stream_port} if remote)")

    serial = pick_device(args.serial)
    settings = json.loads(args.camera_profile.read_text()).get(serial)
    if not settings or not {"color_exposure_option", "color_gain", "white_balance"} <= set(settings):
        raise ValueError(f"{args.camera_profile} gives no colour exposure/gain/white_balance for "
                         f"{serial}; the D455 cannot meter them, so they must be explicit")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    while (args.out_dir / f"episode_{n}.hdf5").exists():
        n += 1
    out_path = args.out_dir / f"episode_{n}.hdf5"

    free_gb = shutil.disk_usage(args.out_dir).free / 1e9
    mb_s = MB_PER_FRAME * FPS
    print(f"device    {serial} (USB 3)")
    print(f"output    {out_path}")
    print(f"rate      ~{mb_s:.0f} MB/s = ~{mb_s * 60 / 1000:.1f} GB/min "
          f"-> {free_gb:.0f} GB free = ~{free_gb * 1000 / mb_s / 60:.0f} min")
    if free_gb < args.min_free_gb:
        raise ValueError(f"only {free_gb:.1f} GB free (--min-free-gb {args.min_free_gb})")

    # Streams, emitter at max power, global time, colour locked to the profile. ASIC
    # depth is enabled but never stored: it feeds the live TOO CLOSE warning. Stereo
    # dies inside ~0.5 m, and a scan walked too close fuses into garbage.
    pipe, cam, depth_scale = open_d455(serial, profile_settings=settings)
    K_ir, K_rgb, T_rgb_to_ir, baseline = (cam["K_ir"], cam["K_rgb"], cam["T_rgb_to_ir"],
                                          cam["baseline"])

    print(f"baseline  {baseline * 1000:.1f} mm   settling IR auto-exposure ({SETTLE_S:g}s)...")
    deadline = time.perf_counter() + SETTLE_S
    while time.perf_counter() < deadline:
        pipe.wait_for_frames()

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    q: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
    writer = ScanWriter(out_path, q, args.compression)
    writer.start()
    if args.stream:
        preview.start()

    print(f"\nRECORDING -> {out_path}   (Ctrl-C to stop)")
    print("walk slowly; watch SHARPNESS -- it dips when you turn too fast\n")
    t0 = time.time()
    n_frames, last_report, last_n = 0, t0, 0
    status = ""
    capture_complete = False
    try:
        while not stop.is_set():
            if args.duration is not None and time.time() - t0 >= args.duration:
                break
            fs = pipe.wait_for_frames()
            f1, f2, fc = fs.get_infrared_frame(1), fs.get_infrared_frame(2), fs.get_color_frame()
            fd = fs.get_depth_frame()
            if not f1 or not f2 or not fc or not fd:
                raise ValueError(f"incomplete frameset at frame {n_frames}")
            ir = np.array(f1.get_data(), copy=True)
            rec = {
                "camera_0_left_ir": ir,
                "camera_0_right_ir": np.array(f2.get_data(), copy=True),
                "camera_0_color": np.array(fc.get_data(), copy=True),
                "camera_0_intrinsics": K_ir,
                "camera_0_rgb_intrinsics": K_rgb,
                "camera_0_rgb_to_ir_extrinsic": T_rgb_to_ir,
                "camera_0_baseline": baseline,
                "camera_0_host_ns": int(round(fc.get_timestamp() * 1e6)),
                "camera_0_timestamp_ms": float(fc.get_timestamp()),
                "camera_0_frame_numbers": np.array(
                    [f1.get_frame_number(), f2.get_frame_number(), fc.get_frame_number()],
                    np.int64),
            }
            rec.update({f"camera_0_{key}": value for key, value in
                        rsmeta.frame_metadata([f1, f2, fc], rs).items()})
            try:
                q.put(rec, timeout=2.0)
            except queue.Full:
                raise ValueError(
                    f"writer fell {QUEUE_DEPTH} frames behind -- disk cannot keep up. "
                    f"Write to the local NVMe, not a USB drive or network mount.")
            if writer.error:
                raise ValueError(f"writer failed: {writer.error}")
            n_frames += 1

            now = time.time()
            if now - last_report >= 1.0:
                sharp = float(np.abs(np.diff(
                    ir[180:540:4, 320:960:4].astype(np.int16), axis=1)).mean())
                z = np.asanyarray(fd.get_data())[::8, ::8]
                z = z[z > 0]
                zmed = float(np.median(z)) * depth_scale if z.size else float("nan")
                near = "  << TOO CLOSE, step back" if zmed < 0.5 else ""
                fps = (n_frames - last_n) / (now - last_report)
                clip = rsmeta.color_statistics(rec["camera_0_color"])[6] * 100
                status = (f"  {n_frames:6d} frames  {now - t0:6.1f}s  {fps:5.1f} fps  "
                          f"queue {q.qsize():2d}/{QUEUE_DEPTH}  SHARPNESS {sharp:5.2f}  "
                          f"CLIP {clip:4.1f}%  DEPTH {zmed:4.2f} m{near}")
                print(status, flush=True)
                last_report, last_n = now, n_frames
            if args.stream and n_frames % 3 == 0:          # ~10 Hz is plenty for a preview
                h = 360
                bgr = cv2.cvtColor(np.asanyarray(fc.get_data()), cv2.COLOR_RGB2BGR)
                col_v = cv2.resize(bgr, (COLOR_W * h // COLOR_H, h))
                ir_v = cv2.resize(cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR), (IR_W * h // IR_H, h))
                zmm = np.asanyarray(fd.get_data())
                z8 = np.clip(zmm.astype(np.float32) * depth_scale / 4.0 * 255, 0, 255).astype(np.uint8)
                dep_v = cv2.applyColorMap(z8, cv2.COLORMAP_JET)
                dep_v[zmm == 0] = 0
                dep_v = cv2.resize(dep_v, (IR_W * h // IR_H, h))
                view = np.hstack([col_v, ir_v, dep_v])
                cv2.putText(view, status.strip(), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 255) if "TOO CLOSE" in status else (0, 255, 0), 1, cv2.LINE_AA)
                ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise ValueError("jpeg encode failed")
                preview.push(buf.tobytes())
        capture_complete = True
    finally:
        pipe.stop()
        if args.stream:
            preview.httpd.shutdown()
        writer.close({
            "cam_ids": ["camera_0"],
            "serial": serial,
            "purpose": "handheld_room_scan",
            "capture_complete": capture_complete,
            **{k: cam[k] for k in ("sensor_settings_json", "ir_distortion_model",
                                   "rgb_distortion_model", "color_exposure_option",
                                   "color_gain", "color_white_balance",
                                   "color_auto_exposure", "color_auto_white_balance",
                                   "color_white_rb")},
            "scan_note": ("camera_0_extrinsics is absent ON PURPOSE: the per-frame pose "
                          "is unsolved. The scan fuser estimates the trajectory with "
                          "RGB-D odometry over FoundationStereo depth and registers the "
                          "fused atlas into the camera_0 table world once."),
            "fps": FPS,
            "emitter_enabled": 1,
            "streams": ["left_ir", "right_ir", "color"],
            "compression": args.compression,
            "timestamp_domain": "global_time (host epoch, ms)",
            "ir_coeffs": cam["ir_coeffs"],
            "rgb_coeffs": cam["rgb_coeffs"],
        })
        dur = time.time() - t0
        print(f"\nwrote {out_path}  ({writer.n} frames, "
              f"{out_path.stat().st_size / 1e9:.2f} GB, {dur:.1f}s, "
              f"{writer.n / max(dur, 1e-9):.1f} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
