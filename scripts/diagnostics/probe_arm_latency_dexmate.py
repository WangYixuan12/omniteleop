#!/usr/bin/env python3
"""Dexmate-side dexcomm sniffer for the arm-command latency probe.

Run this on the dexmate machine (the same host that runs ``dextop node``) to
capture per-message receive timestamps for ``control/arm/right``. The companion
``probe_arm_latency_client.py`` (on lambda) publishes each rep's first command
with a non-zero ``sequence`` field; the 100 Hz hold-loop publishes that follow
use ``sequence=0`` so this sniffer can drop them.

Usage on dexmate::

    ssh dexmate
    conda activate dexmate
    python /path/to/omniteleop/scripts/diagnostics/probe_arm_latency_dexmate.py \
        --out /tmp/dexmate_arm_latency.csv

Wait for ``[ready]`` before starting the client. Stop with Ctrl-C; the CSV is
written on exit.

We use ``dexcomm.Node`` (the same transport ``dextop node`` uses) so peer
discovery is automatic.
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from dexcomm import Node
from dexcomm.codecs import JointCmdCodec


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        default="/tmp/dexmate_arm_latency.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--topic",
        default="control/arm/right",
        help="Topic to subscribe to (Node will prefix it with ROBOT_NAME).",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[int, int, int, list[float]]] = []
    lock = threading.Lock()

    node = Node(name="arm_latency_sniffer")

    def on_msg(msg: dict[str, Any]) -> None:
        # Callback runs on the dexcomm worker thread — keep it tiny.
        t_recv = time.time_ns()
        with lock:
            rows.append((
                t_recv,
                int(msg.get("sequence", 0)),
                int(msg.get("timestamp_ns", 0)),
                list(msg.get("pos", [])),
            ))

    sub = node.create_subscriber(
        topic=args.topic,
        callback=on_msg,
        decoder=JointCmdCodec.decode,
    )

    print(f"[ready] subscribed to {sub.topic}")
    print(f"        writing to {out_path} on Ctrl-C")
    sys.stdout.flush()

    stop = threading.Event()

    def shutdown(signum, frame):  # noqa: ANN001 - signal handler signature
        stop.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while not stop.is_set():
            time.sleep(0.2)
    finally:
        try:
            sub.shutdown()
        except Exception:
            pass
        try:
            node.shutdown()
        except Exception:
            pass

        with lock:
            snap = list(rows)
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_recv_ns_dexmate", "sequence", "client_timestamp_ns", "pos_csv"])
            for t, seq, client_ts, pos in snap:
                w.writerow([t, seq, client_ts, ",".join(f"{p:.6f}" for p in pos)])
        print(f"[done] wrote {len(snap)} rows to {out_path}")


if __name__ == "__main__":
    main()
