#!/usr/bin/env python3
"""Probe whether the wrist ZED stream is reachable on THIS machine's namespace.

Reproduces exactly how ``vr_reader`` (via dexcontrol ``ZedCameraSensor``)
subscribes to the wrist camera: a default-namespace ``dexcomm.Node`` decoding
``sensors/wrist_zedm/left_rgb`` with ``RGBImageCodec``. A default-namespace Node
resolves its prefix from the ``ROBOT_NAME`` env var, so the *resolved key* must
match what the publisher (``tests/test_wrist_zedm_depth.py``) actually publishes.
It also mirrors dexcontrol's default ``ZENOH_CONFIG`` discovery; otherwise the
probe can see a different Zenoh session than ``vr_reader``.

Run it on BOTH machines and compare the printed namespace + frame count:

    # on the robot (same host as the publisher)
    python scripts/diagnostics/probe_wrist_stream.py
    # on the workstation that runs vr_reader (e.g. lambda)
    python scripts/diagnostics/probe_wrist_stream.py

Interpretation:
- Both print the SAME namespace and both receive frames → stream is fine.
- A host prints a namespace but receives 0 frames → the publisher resolved a
  DIFFERENT namespace or Zenoh config. Fix by launching the publisher with the
  same ROBOT_NAME/ZENOH_CONFIG, or ``--namespace "$ROBOT_NAME"``.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import threading
import time

from dexcomm import Node
from dexcomm.codecs import RGBImageCodec


def _configure_zenoh_like_dexcontrol() -> pathlib.Path | None:
    if os.getenv("ZENOH_CONFIG"):
        return pathlib.Path(os.environ["ZENOH_CONFIG"]).expanduser()

    base_dir = pathlib.Path("~/.dexmate/comm/zenoh").expanduser()
    for pattern in ("**/*.dzcfg", "**/zenoh*config*.json5"):
        matches = sorted(p for p in base_dir.glob(pattern) if p.is_file())
        if matches:
            os.environ["ZENOH_CONFIG"] = matches[0].as_posix()
            return matches[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="sensors/wrist_zedm/left_rgb")
    ap.add_argument("--duration", type=float, default=5.0)
    args = ap.parse_args()

    zenoh_config = _configure_zenoh_like_dexcontrol()
    node = Node(name="wrist_probe")  # no namespace → same as ZedCameraSensor's Node
    print(f"ROBOT_NAME env   = {os.environ.get('ROBOT_NAME')!r}")
    print(f"ZENOH_CONFIG env = {os.environ.get('ZENOH_CONFIG')!r}")
    print(f"zenoh config     = {zenoh_config if zenoh_config is not None else '<dexcomm default>'}")
    print(f"node.namespace   = {node.namespace!r}")
    print(f"subscribe topic  = {args.topic!r}")
    print(f"resolved key     = {node.resolve_topic(args.topic)!r}")

    lock = threading.Lock()
    callback_frames = 0
    last_shape = None
    last_sequence = None
    last_timestamp_ns = None

    def on_msg(msg: object) -> None:
        nonlocal callback_frames, last_shape, last_sequence, last_timestamp_ns
        if not isinstance(msg, dict) or msg.get("data") is None:
            return
        sequence = int(msg.get("sequence", 0) or 0)
        timestamp_ns = int(msg.get("timestamp_ns", msg.get("timestamp", 0)) or 0)
        with lock:
            callback_frames += 1
            last_shape = msg["data"].shape
            last_sequence = sequence
            last_timestamp_ns = timestamp_ns

    sub = node.create_subscriber(
        args.topic,
        callback=on_msg,
        decoder=RGBImageCodec.decode,
        buffer_size=1,
    )

    seen: set[tuple[str, int]] = set()
    latest_seen = False
    t0 = time.time()
    while time.time() - t0 < args.duration:
        msg = sub.get_latest()
        if isinstance(msg, dict) and msg.get("data") is not None:
            sequence = int(msg.get("sequence", 0) or 0)
            timestamp_ns = int(msg.get("timestamp_ns", msg.get("timestamp", 0)) or 0)
            with lock:
                latest_seen = True
                last_shape = msg["data"].shape
                last_sequence = sequence
                last_timestamp_ns = timestamp_ns
            # Codec defaults sequence to 0 when the publisher does not set it.
            # Treat only positive sequence numbers as frame identities; otherwise
            # fall back to timestamp_ns.
            if sequence > 0:
                seen.add(("sequence", sequence))
            elif timestamp_ns > 0:
                seen.add(("timestamp_ns", timestamp_ns))
        time.sleep(0.02)

    with lock:
        n_callback = callback_frames
        shape = last_shape
        sequence = last_sequence
        timestamp_ns = last_timestamp_ns
    n_metadata = len(seen)
    receiving = n_callback > 0 or latest_seen
    print(
        f"\ncallback frames in {args.duration:.0f}s = {n_callback}  "
        f"({n_callback / args.duration:.1f} fps)\n"
        f"metadata-unique frames = {n_metadata}  "
        f"last shape={shape}  last sequence={sequence}  "
        f"last timestamp_ns={timestamp_ns}"
    )
    print("RESULT:", "RECEIVING ✓" if receiving else "NO FRAMES ✗ (namespace/publisher mismatch)")
    node.shutdown()


if __name__ == "__main__":
    main()
