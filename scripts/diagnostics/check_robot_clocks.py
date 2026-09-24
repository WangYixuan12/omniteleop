#!/usr/bin/env python3
"""Report how far each robot clock is from this workstation's clock.

Three clocks stamp the data we record: this workstation (record/receive stamps), the
Jetson (camera frames, via the ZED publishers' ``sensors/<id>/clock`` service) and the
control SoC (joint states, via ``Robot.query_ntp()`` and the state-message stamps).
Run it before and after any time-sync fix. Healthy: the query_ntp and Jetson rows within
a few ms; the state-stamp rows are stamp AGE at read time and sit a few ms negative.
Fixed 2026-09-04 with Dexmate's SoC .dmfix + vega-1-orin-repair .run: SoC went from
+52.5 s to +1.3 ms; the Jetson now syncs chrony from the SoC (192.168.50.21).

    python scripts/diagnostics/check_robot_clocks.py
"""
from __future__ import annotations

import os
import time

import numpy as np

os.environ.setdefault("ROBOT_NAME", "dm/vg7ae4b55f3e-1")
os.environ.setdefault("ZENOH_CONFIG",
                      os.path.expanduser("~/.dexmate/comm/zenoh/dm_vg7ae4b55f3e-1.dzcfg"))

from dexcomm.codecs import JsonDataCodec  # noqa: E402
from dexcontrol.robot import Robot  # noqa: E402


def ntp_offset(client, n: int = 5) -> tuple[float, float]:
    """Best-of-n NTP-style (offset_ms, rtt_ms) of the publisher host minus this host."""
    best = None
    for _ in range(n):
        t0 = time.time_ns()
        r = client.call({})
        t3 = time.time_ns()
        t1, t2 = int(r["server_receive_time_ns"]), int(r["server_send_time_ns"])
        off, rtt = ((t1 - t0) + (t2 - t3)) / 2, (t3 - t0) - (t2 - t1)
        if best is None or rtt < best[1]:
            best = (off, rtt)
    return best[0] / 1e6, best[1] / 1e6


def main() -> None:
    robot = Robot()
    try:
        time.sleep(1.0)
        rows = []
        ntp = robot.query_ntp()
        rows.append(("SoC (Robot.query_ntp)", float(ntp["offset"]) * 1e3, float(ntp["rtt"]) * 1e3))
        for name in ("head", "left_arm", "torso"):
            comp = getattr(robot, name)
            d = [(int(comp.get_timestamp_ns()) - time.time_ns()) / 1e6 for _ in range(10)
                 if not time.sleep(0.02)]
            # Stamp minus read time = clock offset MINUS transport age, so these rows read a
            # few ms negative even when synced; they are informational, not part of the verdict.
            rows.append((f"SoC state stamp age ({name})", float(np.median(d)), float("nan")))
        for sid in ("head_camera", "left_wrist_zedm", "right_wrist_zedm"):
            try:
                c = robot._node.create_service_client(
                    service_name=f"sensors/{sid}/clock", request_encoder=JsonDataCodec.encode,
                    response_decoder=JsonDataCodec.decode, timeout=3.0)
                off, rtt = ntp_offset(c)
                rows.append((f"Jetson ({sid}/clock)", off, rtt))
            except Exception as exc:  # publisher not running
                rows.append((f"Jetson ({sid}/clock)", float("nan"), float("nan")))
                print(f"  {sid}: no clock service ({type(exc).__name__})")
        print(f"\n{'clock':34s} {'minus workstation':>18s} {'rtt':>8s}")
        for name, off, rtt in rows:
            unit = "s" if abs(off) >= 1000 else "ms"
            val = off / 1e3 if unit == "s" else off
            print(f"{name:34s} {val:+15.1f} {unit:2s} {rtt:8.1f}" if np.isfinite(rtt)
                  else f"{name:34s} {val:+15.1f} {unit:2s}")
        worst = max(abs(r[1]) for r in rows if np.isfinite(r[1]) and "stamp age" not in r[0])
        print("\nOK: all clocks within 10 ms" if worst < 10 else
              f"\nNOT SYNCED: worst offset {worst:.0f} ms -- see README section 0 / Dexmate clock fix")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
