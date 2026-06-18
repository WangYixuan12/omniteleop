"""Consume the SLAM pose from zenoh (what the omniteleop stack would use).

Run in the yixuan_yifan conda env while mapping or localization is up:

    conda run -n yixuan_yifan python SLAM/scripts/probe_slam_pose.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_topics import ROBOT_PREFIX, find_zenoh_config  # noqa: E402


def main() -> None:
    from dexcomm import Subscriber
    from dexcomm.codecs import JsonDataCodec

    topic = f"{ROBOT_PREFIX}/slam/pose"
    count = 0
    t_first = 0.0
    last: dict | None = None

    def cb(msg: dict) -> None:
        nonlocal count, t_first, last
        if count == 0:
            t_first = time.monotonic()
        count += 1
        last = msg

    sub = Subscriber(topic, callback=cb, decoder=JsonDataCodec.decode,
                     config=find_zenoh_config())
    print(f"listening on {topic} (ctrl-c to stop)")
    try:
        while True:
            time.sleep(1.0)
            if last is None:
                print("  no pose yet (slam_toolbox must have published map->odom)")
                continue
            rate = (count - 1) / max(time.monotonic() - t_first, 1e-9)
            x, y, z = last["pos"]
            age_ms = (time.time_ns() - last["timestamp_ns"]) / 1e6
            print(f"  {rate:5.1f} Hz  x={x:+.3f} y={y:+.3f} "
                  f"yaw={np.degrees(last['yaw']):+.1f}deg  age={age_ms:.0f}ms")
    except KeyboardInterrupt:
        pass
    finally:
        sub.shutdown()


if __name__ == "__main__":
    main()
