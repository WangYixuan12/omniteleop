#!/usr/bin/env python3
"""Measure on-wire bandwidth per camera stream — the WiFi load that drives lag.

Subscribes (raw zenoh, like probe_camera_timestamps.py) to several
``sensors/*`` keys at once and, over a window, sums ``len(sample.payload)`` per
key. The payload is the *encoded* (JPEG/compressed-depth) bytes that actually
cross the robot→workstation WiFi link, so bytes/s here == the uplink load that
queues the teleop command path. Decodes the first sample per key to confirm the
on-wire frame shape/dtype (e.g. that crop+resize really took effect).

Run on a host on the robot's Zenoh fabric, with the publishers running::

    python scripts/diagnostics/probe_camera_bandwidth.py --duration 6
    python scripts/diagnostics/probe_camera_bandwidth.py --duration 6 \
        --topics sensors/head_camera/left_rgb sensors/head_camera/depth \
                 sensors/wrist_zedm/left_rgb
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

# Importing dexcontrol sets $ZENOH_CONFIG to the robot's mTLS peer config.
import dexcontrol  # noqa: F401
import zenoh
from dexcomm.codecs import DepthImageCodec, RGBImageCodec

# Documented full-res @30fps baseline (memory: head-wrist-lag-wifi-bufferbloat),
# for a before/after reference. Mbps.
_BASELINE_MBPS = {
    "sensors/head_camera/left_rgb": 8.3,
    "sensors/head_camera/right_rgb": 8.3,
    "sensors/head_camera/depth": 57.9,
    "sensors/wrist_zedm/left_rgb": 13.7,
    "sensors/wrist_zedm/right_rgb": 13.0,
}


def _load_zenoh_config() -> zenoh.Config:
    cfg_path = os.getenv("ZENOH_CONFIG")
    if cfg_path and Path(cfg_path).is_file():
        print(f"using zenoh config: {cfg_path}")
        return zenoh.Config.from_file(cfg_path)
    print("WARNING: $ZENOH_CONFIG unset/missing — falling back to default peer config")
    return zenoh.Config()


def _resolve_topic(topic: str) -> str:
    namespace = os.getenv("ROBOT_NAME", "").strip("/")
    if namespace and not topic.startswith(f"{namespace}/"):
        return f"{namespace}/{topic}"
    return topic


def _decoder_for(topic: str):
    return DepthImageCodec.decode if "depth" in topic else RGBImageCodec.decode


class _Stat:
    __slots__ = ("count", "total_bytes", "shape", "dtype")

    def __init__(self) -> None:
        self.count = 0
        self.total_bytes = 0
        self.shape = None
        self.dtype = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument(
        "--topics",
        nargs="+",
        default=[
            "sensors/head_camera/left_rgb",
            "sensors/head_camera/depth",
            "sensors/wrist_zedm/left_rgb",
        ],
    )
    args = ap.parse_args()

    lock = threading.Lock()
    stats: dict[str, _Stat] = {t: _Stat() for t in args.topics}
    resolved = {t: _resolve_topic(t) for t in args.topics}

    session = zenoh.open(_load_zenoh_config())
    subs = []

    def make_cb(topic: str):
        decoder = _decoder_for(topic)
        st = stats[topic]

        def on_sample(sample: zenoh.Sample) -> None:
            payload = sample.payload.to_bytes()
            with lock:
                st.count += 1
                st.total_bytes += len(payload)
                if st.shape is None:  # decode just the first frame to confirm shape
                    try:
                        d = decoder(payload)
                        arr = d.get("data") if isinstance(d, dict) else None
                        if arr is not None:
                            st.shape, st.dtype = arr.shape, arr.dtype
                    except Exception:
                        pass

        return on_sample

    try:
        for t in args.topics:
            subs.append(session.declare_subscriber(resolved[t], make_cb(t)))
        print(f"measuring {len(args.topics)} keys for {args.duration:.1f}s ...\n")
        t0 = time.perf_counter()
        time.sleep(args.duration)
        elapsed = time.perf_counter() - t0
        for s in subs:
            s.undeclare()
    finally:
        session.close()

    print(f"{'topic':40s} {'fps':>6s} {'KB/frm':>8s} {'Mbps':>8s} "
          f"{'baseline':>9s} {'shape':>16s}")
    print("-" * 96)
    total = 0.0
    base_total = 0.0
    for t in args.topics:
        st = stats[t]
        fps = st.count / elapsed
        kb = (st.total_bytes / st.count / 1024.0) if st.count else 0.0
        mbps = st.total_bytes * 8 / elapsed / 1e6
        total += mbps
        base = _BASELINE_MBPS.get(t)
        base_total += base or 0.0
        base_s = f"{base:.1f}" if base is not None else "-"
        shape_s = f"{st.shape}" if st.shape is not None else "(no frames)"
        print(f"{t:40s} {fps:6.1f} {kb:8.1f} {mbps:8.2f} {base_s:>9s} {shape_s:>16s}")
    print("-" * 96)
    print(f"{'TOTAL':40s} {'':6s} {'':8s} {total:8.2f} {base_total:9.1f}  Mbps "
          f"(baseline = full-res @30fps)")
    if base_total > 0 and total > 0:
        print(f"\n→ {base_total / total:.1f}x less WiFi uplink than the full-res @30fps baseline.")


if __name__ == "__main__":
    main()
