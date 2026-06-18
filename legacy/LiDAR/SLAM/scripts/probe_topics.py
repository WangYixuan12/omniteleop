"""Step 0 ground-truth probe for SLAM sensor inputs.

Run in the `yixuan_yifan` conda env on the robot Jetson (vega-1):

    conda run -n yixuan_yifan python SLAM/scripts/probe_topics.py

Two phases:
  A. Discovery: raw eclipse-zenoh wildcard subscriber on `dm/**` lists every
     live topic with message rate and payload size (finds the exact lidar key).
  B. Decode probe: dexcomm subscribers on the five SLAM-relevant topics report
     rate, decoded keys/shapes, and clock offset (receive time - msg timestamp).

The clock-offset numbers decide whether the ROS bridge can stamp messages with
the sensor timestamp (|offset| < 50 ms) or must fall back to receive time.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROBOT_PREFIX = "dm/vg3cfe65689d-1"


def find_zenoh_config() -> str | None:
    """Resolve the zenoh config exactly like dexcontrol/__init__.py does.

    The robot's services communicate over an mTLS peer overlay (scouting group
    224.0.0.237:50297) defined in ~/.dexmate/comm/zenoh/; default multicast
    scouting receives NOTHING (verified 2026-06-11).
    """
    env = os.getenv("ZENOH_CONFIG")
    if env:
        return env
    base_dir = Path("~/.dexmate/comm/zenoh/").expanduser()
    dzcfg = sorted(base_dir.glob("**/*.dzcfg"))
    if dzcfg:
        return dzcfg[0].as_posix()
    json5 = sorted(base_dir.glob("**/zenoh*config*.json5"))
    if json5:
        return json5[0].as_posix()
    return None

# topic suffix -> dexcomm codec class name
# NOTE (verified 2026-06-11): dexsensor publishes the 2D scan on the
# "/scan"-suffixed key, NOT the bare sensor name. chassis_imu is NOT
# published by this robot's firmware (passive subscriber, nothing to enable).
PROBE_TOPICS = {
    f"{ROBOT_PREFIX}/sensors/lidar_2d_front/scan": "LidarScan2DCodec",
    f"{ROBOT_PREFIX}/state/chassis/steer": "JointStateCodec",
    f"{ROBOT_PREFIX}/state/chassis/drive": "JointStateCodec",
    f"{ROBOT_PREFIX}/state/chassis_imu": "IMUDataCodec",
    f"{ROBOT_PREFIX}/state/ultrasonic": "UltrasonicStateCodec",
}


def discover(duration_s: float, config_path: str | None) -> None:
    """Phase A: list live zenoh keys under dm/** with rates and sizes."""
    import zenoh

    counts: dict[str, int] = defaultdict(int)
    nbytes: dict[str, int] = defaultdict(int)
    lock = threading.Lock()

    def on_sample(sample: "zenoh.Sample") -> None:
        key = str(sample.key_expr)
        with lock:
            counts[key] += 1
            nbytes[key] += len(sample.payload)

    print(f"=== Phase A: discovery on 'dm/**' for {duration_s:.0f} s ===")
    zcfg = zenoh.Config.from_file(config_path) if config_path else zenoh.Config()
    with zenoh.open(zcfg) as session:
        sub = session.declare_subscriber("dm/**", on_sample)
        time.sleep(duration_s)
        sub.undeclare()

    if not counts:
        print("  !! No samples received on dm/** — is dexsensor running and "
              "is this machine on the robot network?")
        return
    width = max(len(k) for k in counts)
    for key in sorted(counts):
        rate = counts[key] / duration_s
        avg_kb = nbytes[key] / counts[key] / 1024.0
        print(f"  {key:<{width}}  {rate:7.2f} Hz  {avg_kb:8.2f} KiB/msg")
    lidar_keys = [k for k in counts if fnmatch.fnmatch(k, "*lidar*")]
    print(f"  -> lidar-like keys: {lidar_keys or 'NONE'}")


class TopicStats:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.lock = threading.Lock()
        self.count = 0
        self.t_first_ns = 0
        self.t_last_ns = 0
        self.offsets_ms: list[float] = []
        self.last_msg: object = None
        self.errors: list[str] = []

    def on_msg(self, msg: object) -> None:
        now_ns = time.time_ns()
        with self.lock:
            self.count += 1
            if self.t_first_ns == 0:
                self.t_first_ns = now_ns
            self.t_last_ns = now_ns
            self.last_msg = msg
            ts = _extract_timestamp_ns(msg)
            if ts is not None:
                self.offsets_ms.append((now_ns - ts) / 1e6)


def _extract_timestamp_ns(msg: object) -> int | None:
    if not isinstance(msg, dict):
        return None
    for key in ("timestamp_ns", "timestamp"):
        if key in msg:
            val = int(msg[key])
            if val <= 0:
                return None
            # Heuristic: ns timestamps for 2025+ are > 1.7e18; treat smaller
            # magnitudes (us / ms / s) explicitly so offsets are comparable.
            if val > int(1e17):
                return val
            if val > int(1e14):
                return val * 1_000
            if val > int(1e11):
                return val * 1_000_000
            return val * 1_000_000_000
    return None


def _describe_value(val: object) -> str:
    if isinstance(val, np.ndarray):
        return f"ndarray{val.shape} {val.dtype}"
    if isinstance(val, (list, tuple)):
        return f"{type(val).__name__}[{len(val)}]"
    if isinstance(val, float):
        return f"float={val:.4g}"
    if isinstance(val, int):
        return f"int={val}"
    return type(val).__name__


def probe(duration_s: float, config_path: str | None) -> None:
    """Phase B: decoded subscriptions with rate + clock-offset stats."""
    from dexcomm import Subscriber
    from dexcomm import codecs as dexcodecs

    print(f"\n=== Phase B: decode probe for {duration_s:.0f} s ===")
    stats: list[TopicStats] = []
    subs: list[Subscriber] = []
    for topic, codec_name in PROBE_TOPICS.items():
        codec = getattr(dexcodecs, codec_name)
        st = TopicStats(topic)
        stats.append(st)
        subs.append(Subscriber(topic, callback=st.on_msg,
                               decoder=codec.decode, config=config_path))

    time.sleep(duration_s)
    for sub in subs:
        sub.shutdown()

    for st in stats:
        print(f"\n  {st.topic}")
        with st.lock:
            if st.count == 0:
                print("    !! NO MESSAGES")
                continue
            span_s = max((st.t_last_ns - st.t_first_ns) / 1e9, 1e-9)
            rate = (st.count - 1) / span_s if st.count > 1 else 0.0
            print(f"    count={st.count}  rate={rate:.2f} Hz")
            msg = st.last_msg
            if isinstance(msg, dict):
                for key, val in msg.items():
                    print(f"    .{key}: {_describe_value(val)}")
            else:
                print(f"    msg type: {type(msg).__name__}: {msg!r:.120}")
            if st.offsets_ms:
                off = np.asarray(st.offsets_ms)
                verdict = "OK: sensor-timestamp stamping usable" \
                    if abs(np.mean(off)) < 50.0 else \
                    "WARN: offset > 50 ms -> use receive-time stamping"
                print(f"    clock offset recv-msg [ms]: mean={np.mean(off):.2f} "
                      f"min={np.min(off):.2f} max={np.max(off):.2f}  ({verdict})")
            else:
                print("    no usable timestamp field -> receive-time stamping")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover-sec", type=float, default=5.0)
    parser.add_argument("--probe-sec", type=float, default=10.0)
    parser.add_argument("--skip-discovery", action="store_true",
                        help="Skip the raw-zenoh wildcard phase (e.g. inside "
                             "the container where eclipse-zenoh isn't installed)")
    args = parser.parse_args()
    config_path = find_zenoh_config()
    print(f"zenoh config: {config_path or 'dexcomm defaults (peer+multicast)'}")
    if not args.skip_discovery:
        discover(args.discover_sec, config_path)
    probe(args.probe_sec, config_path)


if __name__ == "__main__":
    main()
