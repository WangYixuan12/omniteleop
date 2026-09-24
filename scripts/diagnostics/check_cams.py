#!/usr/bin/env python3
"""Photograph every camera at once and measure what rate each one really delivers.

Use it to decide which D455 serial is the fixed overhead one before passing
``--static-serial`` to record_arm_scan.py: wave a hand in front of one camera, re-run,
and see which tile moved. Each tile is captioned with the identifier you pass on the
command line, so there is no guessing which serial is which physical camera.

Records nothing, commands no motion. Reads cameras only.

    check_cams.py --out /data/Dexmate/viz/cams.png
    check_cams.py --out /data/Dexmate/viz/cams.png --seconds 5 --no-robot

The rates it prints are what the SENSOR delivers, not what a scan stores -- the scan
subsamples the robot cameras to --cam-hz and the fixed D455 to --static-hz.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROBOT_CAMS = (("head", "head_camera"),
              ("left_wrist", "left_wrist_zedm"),
              ("right_wrist", "right_wrist_zedm"))
IR_W, IR_H, COLOR_W, COLOR_H, FPS = 1280, 720, 1280, 800, 30


def grab_d455(seconds: float):
    """{serial: (rgb, fps, note)} for every connected RealSense, all streaming at once."""
    import pyrealsense2 as rs
    ctx = rs.context()
    devs = list(ctx.query_devices())
    if not devs:
        return {}
    pipes, out = {}, {}
    for d in devs:
        sn = d.get_info(rs.camera_info.serial_number)
        usb = d.get_info(rs.camera_info.usb_type_descriptor)
        cfg = rs.config()
        cfg.enable_device(sn)
        cfg.enable_stream(rs.stream.infrared, 1, IR_W, IR_H, rs.format.y8, FPS)
        cfg.enable_stream(rs.stream.infrared, 2, IR_W, IR_H, rs.format.y8, FPS)
        cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.rgb8, FPS)
        p = rs.pipeline(ctx)
        try:
            prof = p.start(cfg)
        except Exception as exc:
            out[sn] = (None, 0.0, f"USB {usb} -- START FAILED: {exc}")
            continue
        ds = prof.get_device().first_depth_sensor()
        if ds.supports(rs.option.emitter_enabled):
            ds.set_option(rs.option.emitter_enabled, 1.0)
        pipes[sn] = (p, usb)
    for _ in range(15):                       # let auto-exposure settle
        for p, _u in pipes.values():
            p.poll_for_frames()
        time.sleep(0.05)
    n = {sn: 0 for sn in pipes}
    last = {sn: None for sn in pipes}
    t0 = time.time()
    while time.time() - t0 < seconds:
        for sn, (p, _u) in pipes.items():
            fs = p.poll_for_frames()
            if fs and fs.get_color_frame():
                n[sn] += 1
                last[sn] = np.asanyarray(fs.get_color_frame().get_data()).copy()
        time.sleep(0.001)
    el = time.time() - t0
    for sn, (p, usb) in pipes.items():
        out[sn] = (last[sn], n[sn] / el, f"USB {usb}")
        p.stop()
    return out


def grab_robot(seconds: float):
    """{name: (rgb, fps, note)} for the head and both wrists, straight off zenoh.

    Read from the topics rather than through dexcontrol, for two reasons: it needs no
    Robot connection (so this cannot disturb the head), and a stock Robot has no sensor
    for sensors/<side>_wrist_zedm/* -- its built-in *_wrist_camera is a MONOCULAR ZED X
    One model with one "rgb" stream, while the publisher emits the stereo pair as
    left_rgb/right_rgb the way the head does (wbc_vr_robot.py injects its own config).
    """
    import os

    os.environ.setdefault(
        "ZENOH_CONFIG",
        os.path.expanduser("~/.dexmate/comm/zenoh/dm_vg7ae4b55f3e-1.dzcfg"))
    from dexcomm import subscribe
    from dexcomm.codecs import RGBImageCodec

    ns = os.environ.get("ROBOT_NAME", "dm/vg7ae4b55f3e-1")
    latest, seen, subs = {}, {}, []
    for kind, attr in ROBOT_CAMS:
        seen[kind] = set()

        def _cb(msg, _k=kind):
            payload = getattr(msg, "data", msg)
            arr = payload.get("data") if isinstance(payload, dict) else payload
            ts = (payload.get("timestamp_ns") if isinstance(payload, dict)
                  else getattr(msg, "timestamp_ns", None))
            if arr is None:
                return
            latest[_k] = np.asarray(arr)
            if ts is not None:
                seen[_k].add(int(ts))

        subs.append(subscribe(f"{ns}/sensors/{attr}/left_rgb", _cb,
                              decoder=RGBImageCodec.decode))
    t0 = time.time()
    while time.time() - t0 < seconds:
        time.sleep(0.02)
    el = time.time() - t0
    for sub in subs:
        try:
            sub.undeclare()
        except Exception:
            pass
    return {k: (latest.get(k), len(seen[k]) / el,
                "no frames" if k not in latest else "zenoh")
            for k, _a in ROBOT_CAMS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/Dexmate/viz/cams.png")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--no-robot", action="store_true",
                    help="only look at the RealSense cameras")
    ap.add_argument("--tile-h", type=int, default=360)
    args = ap.parse_args()

    out = Path(args.out)
    if not out.parent.is_dir():
        raise SystemExit(f"--out directory {out.parent} does not exist")

    tiles = []
    print(f"sampling every camera for {args.seconds:g}s ...\n")
    print(f"{'camera':22s} {'rate':>9s}  {'resolution':>12s}  note")
    print("-" * 66)

    for sn, (img, fps, note) in sorted(grab_d455(args.seconds).items()):
        res = f"{img.shape[1]}x{img.shape[0]}" if img is not None else "-"
        print(f"{'D455 ' + sn:22s} {fps:6.1f} fps  {res:>12s}  {note}")
        tiles.append((f"D455 {sn}", img, fps, note))

    if not args.no_robot:
        for k, (img, fps, note) in grab_robot(args.seconds).items():
            res = f"{img.shape[1]}x{img.shape[0]}" if img is not None else "-"
            print(f"{k:22s} {fps:6.1f} fps  {res:>12s}  {note}")
            tiles.append((k, img, fps, note))

    live = [t for t in tiles if t[1] is not None]
    if not live:
        raise SystemExit("no camera produced a frame")
    H = args.tile_h
    imgs = []
    for name, img, fps, note in live:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (int(bgr.shape[1] * H / bgr.shape[0]), H))
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1] - 1, 46), (0, 0, 0), -1)
        cv2.putText(bgr, f"{name}  {fps:.1f} fps", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1] - 1, bgr.shape[0] - 1), (60, 60, 60), 3)
        imgs.append(bgr)
    rows, per = [], 3
    for i in range(0, len(imgs), per):
        row = imgs[i:i + per]
        w = sum(x.shape[1] for x in row)
        rows.append(np.hstack(row) if len(row) == per else
                    np.hstack(row + [np.zeros((H, max(1, max(x.shape[1] for x in imgs) * per - w), 3),
                                              np.uint8)]))
    W = max(r.shape[1] for r in rows)
    grid = np.vstack([np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0))) for r in rows])
    cv2.imwrite(str(out), grid)
    print(f"\nwrote {out}")
    print("Wave a hand in front of one camera and re-run: the tile that changes is that "
          "camera.\nThe D455 that does NOT move when you carry the other one is your "
          "--static-serial.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
