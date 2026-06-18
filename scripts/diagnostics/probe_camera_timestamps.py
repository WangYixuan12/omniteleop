"""Probe camera-frame timestamps across the dexcontrol 0.4.6 API change.

Why this exists
---------------
`vr_reader.py` used to monkey-patch `dexcontrol.utils.subscribers.camera.RGBCameraSubscriber._data_handler`
to capture each Zenoh sample's HLC timestamp at arrival
(`sample.timestamp.get_time_as_ntp64().as_nanos()`) into `_latest_sample_ts_ns`,
and read it in `_handle_recording` as the per-frame capture time.

dexcontrol 0.4.6 removed those classes. The new `StreamSubscriber.get_latest()`
returns the dexcomm `Subscriber.get_latest()` value verbatim, which is exactly
`RGBImageCodec.decode(payload_bytes)` — a dict. The decoder only ever sees the
raw payload bytes, never the Zenoh `Sample`, so the only timestamp the new API
can expose is the payload field set by dexsensor at encode time.

What the two timestamps actually are
------------------------------------
    (A) Zenoh `sample.timestamp` HLC ns — what the old hook read.
        On this fabric it is absent: dexsensor does not stamp at publish, and a
        `peer` session has `timestamping` disabled by default. When the old hook
        saw a non-None value it was stamped by the *leader's own* session on
        receipt — i.e. ~= leader wall clock at arrival.
    (B) Payload `timestamp_ns` — what the new API exposes. Set by dexsensor on
        the robot at frame encode time; it lives in the *robot's* clock domain.

So the meaningful comparison is (B) payload_ts_ns vs `wall_ns` (leader wall
clock captured in this script's receive callback). Their gap is
transport-latency + leader<->robot clock offset; the clock offset itself is
queryable via `dexcontrol`'s `Robot.query_ntp()`.

Run on a host that already talks to the robot's Zenoh fabric (e.g. lambda with
the dexmate env), with the head camera publishing.

Usage:
    # 1. discover what keys are live on the fabric
    python scripts/diagnostics/probe_camera_timestamps.py --list --duration 5
    # 2. dump the exact structure of one decoded payload (real key names)
    python scripts/diagnostics/probe_camera_timestamps.py --inspect
    # 3. collect and compare payload_ts_ns vs leader wall clock
    python scripts/diagnostics/probe_camera_timestamps.py --duration 5
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections import deque
from pathlib import Path

# Importing dexcontrol sets $ZENOH_CONFIG to the robot's mTLS peer config under
# ~/.dexmate/comm/zenoh/<robot>/. A bare zenoh.Config() can't join that fabric.
import dexcontrol  # noqa: F401
import zenoh
from dexcomm.codecs import DepthImageCodec, RGBImageCodec


def _load_zenoh_config() -> zenoh.Config:
    cfg_path = os.getenv("ZENOH_CONFIG")
    if cfg_path and Path(cfg_path).is_file():
        print(f"using zenoh config: {cfg_path}")
        return zenoh.Config.from_file(cfg_path)
    print("WARNING: $ZENOH_CONFIG unset/missing — falling back to default peer config")
    return zenoh.Config()


def _resolve_topic(topic: str) -> str:
    """Prepend the dexcomm namespace so a raw zenoh sub matches dexsensor's keys.

    dexcomm `Node` prefixes every topic with its namespace; when constructed with
    no explicit namespace it uses the ROBOT_NAME env var. dexcontrol's camera
    sensors create their Node that way, so the on-wire key is
    `<ROBOT_NAME>/sensors/head_camera/left_rgb`, not the bare path.
    """
    namespace = os.getenv("ROBOT_NAME", "").strip("/")
    if namespace and not topic.startswith(f"{namespace}/"):
        return f"{namespace}/{topic}"
    return topic


def _decoder_for(topic: str):
    """Pick the codec dexcontrol would use for this stream (depth vs rgb)."""
    return DepthImageCodec.decode if "depth" in topic else RGBImageCodec.decode


def _discover(duration: float) -> None:
    """Subscribe to '**' and report every key expression seen, with sample counts."""
    counts: dict[str, int] = {}

    def on_sample(sample: zenoh.Sample) -> None:
        key = str(sample.key_expr)
        counts[key] = counts.get(key, 0) + 1

    session = zenoh.open(_load_zenoh_config())
    try:
        sub = session.declare_subscriber("**", on_sample)
        print(f"discovery: subscribed to '**' for {duration:.1f}s ...")
        time.sleep(duration)
        sub.undeclare()
    finally:
        session.close()

    if not counts:
        print(
            "no traffic at all on the fabric — the zenoh session connected but "
            "nothing is publishing (robot/dexsensor likely down or on another fabric)."
        )
        return
    print(f"\n=== {len(counts)} active key expressions ===")
    for key in sorted(counts):
        print(f"  {counts[key]:>6d}  {key}")


def _inspect(topic: str, duration: float) -> None:
    """Capture the first sample on `topic` and dump its full structure.

    Removes all guesswork about (a) whether the raw Zenoh sample carries an HLC
    timestamp and (b) what keys the decoded payload dict actually has.
    """
    decoder = _decoder_for(topic)
    grabbed: list[zenoh.Sample] = []

    def on_sample(sample: zenoh.Sample) -> None:
        if not grabbed:
            grabbed.append(sample)

    session = zenoh.open(_load_zenoh_config())
    try:
        sub = session.declare_subscriber(topic, on_sample)
        print(f"inspect: subscribed to {topic}; waiting for first sample (<= {duration:.1f}s) ...")
        deadline = time.time() + duration
        while not grabbed and time.time() < deadline:
            time.sleep(0.02)
        sub.undeclare()
    finally:
        session.close()

    if not grabbed:
        print("no sample arrived — run --list to confirm the topic is publishing.")
        return

    sample = grabbed[0]
    print("\n=== raw zenoh sample ===")
    print(f"  key_expr   : {sample.key_expr}")
    print(f"  kind       : {sample.kind}")
    print(f"  timestamp  : {sample.timestamp!r}  (type={type(sample.timestamp).__name__})")
    if sample.timestamp is not None:
        print(f"               HLC ns = {sample.timestamp.get_time_as_ntp64().as_nanos()}")
    print(f"  attachment : {sample.attachment!r}")
    payload_bytes = sample.payload.to_bytes()
    print(f"  payload    : {len(payload_bytes)} bytes")

    print("\n=== decoded payload ===")
    try:
        decoded = decoder(payload_bytes)
    except Exception as exc:
        print(f"  decode FAILED: {exc!r}")
        return
    print(f"  type: {type(decoded).__name__}")
    if isinstance(decoded, dict):
        for k, v in decoded.items():
            if hasattr(v, "shape"):
                print(f"  ['{k}'] : ndarray shape={v.shape} dtype={v.dtype}")
            elif isinstance(v, (bytes, bytearray)):
                print(f"  ['{k}'] : {type(v).__name__} len={len(v)}")
            else:
                print(f"  ['{k}'] : {type(v).__name__} = {v!r}")
        print(
            "\n  -> use the int-valued '*timestamp*' key above as the new-API "
            "per-frame capture time."
        )
    else:
        print(f"  (not a dict) repr: {decoded!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds to collect.")
    parser.add_argument(
        "--topic", type=str, default="sensors/head_camera/left_rgb",
        help="Zenoh topic of an RGB (or depth) stream.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Discovery mode: subscribe to '**' and print every key expr seen + counts.",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Grab one sample and dump raw-sample + decoded-payload structure.",
    )
    args = parser.parse_args()

    if args.list:
        _discover(args.duration)
        return

    topic = _resolve_topic(args.topic)

    if args.inspect:
        _inspect(topic, args.duration)
        return

    decoder = _decoder_for(topic)
    # (wall_ns_at_arrival, hlc_ns, payload_ts_ns)
    records: deque[tuple[int, int, int]] = deque()

    def on_sample(sample: zenoh.Sample) -> None:
        wall_ns = time.time_ns()
        hlc_ns = -1
        if sample.timestamp is not None:
            try:
                hlc_ns = int(sample.timestamp.get_time_as_ntp64().as_nanos())
            except Exception as exc:
                print(f"  [warn] HLC read failed: {exc}")

        payload_ts_ns = -1
        try:
            decoded = decoder(sample.payload.to_bytes())
            payload_ts_ns = int(decoded.get("timestamp_ns", -1))
        except Exception as exc:
            print(f"  [warn] payload decode failed: {exc}")

        records.append((wall_ns, hlc_ns, payload_ts_ns))

    session = zenoh.open(_load_zenoh_config())
    try:
        sub = session.declare_subscriber(topic, on_sample)
        print(f"subscribed to {topic}; collecting for {args.duration:.1f}s ...")
        time.sleep(args.duration)
        sub.undeclare()
    finally:
        session.close()

    if not records:
        print("no samples received — is the camera publishing?")
        return

    n = len(records)
    print(f"\n=== collected {n} samples ===")
    print(
        f"{'i':>4}  {'wall_ns (leader rx)':>22}  {'payload_ts_ns (robot)':>22}  "
        f"{'hlc_ns':>20}  {'payload-wall (ms)':>18}"
    )
    gaps_ms: list[float] = []
    hlc_seen = 0
    for i, (wall, hlc, pld) in enumerate(records):
        if hlc > 0:
            hlc_seen += 1
        if pld > 0:
            g_ms = (pld - wall) / 1e6
            gaps_ms.append(g_ms)
            print(f"{i:>4}  {wall:>22d}  {pld:>22d}  {hlc:>20d}  {g_ms:>18.2f}")
        else:
            print(f"{i:>4}  {wall:>22d}  {pld:>22d}  {hlc:>20d}  {'n/a':>18}")

    if gaps_ms:
        print(
            f"\npayload_ts − wall (ms):  n={len(gaps_ms)}  "
            f"mean={statistics.mean(gaps_ms):.2f}  "
            f"median={statistics.median(gaps_ms):.2f}  "
            f"stdev={statistics.pstdev(gaps_ms):.2f}  "
            f"min={min(gaps_ms):.2f}  max={max(gaps_ms):.2f}"
        )

    if hlc_seen == 0:
        print(
            "\nNOTE: every hlc_ns is -1 — the raw Zenoh samples carry no HLC timestamp."
            "\n  dexsensor does not attach a Zenoh timestamp at publish, and a `peer`"
            "\n  session has `timestamping` disabled by default. The old vr_reader hook's"
            "\n  HLC, when non-None, was therefore stamped by the *leader's own* session"
            "\n  on receipt — i.e. ~= wall_ns above (leader wall clock at arrival)."
        )

    print(
        "\nInterpretation:"
        "\n  • payload_ts_ns is dexsensor's frame-encode time, in the ROBOT's clock domain."
        "\n  • wall_ns is the leader's wall clock when this script received the sample;"
        "\n    it is the closest available proxy for what the old hook recorded."
        "\n  • payload_ts − wall therefore mixes transport latency with the leader<->robot"
        "\n    clock offset. Subtract the offset (dexcontrol Robot.query_ntp(), see"
        "\n    memory project_dexcontrol_ntp_query) to isolate true transport latency."
        "\n  • Bottom line for the vr_reader port: switching to payload_ts_ns changes the"
        "\n    recorded camera timestamp from leader-receipt-clock to robot-capture-clock."
        "\n    That is a deliberate semantic change — capture time is better for alignment,"
        "\n    but everything else in the HDF5 is stamped with leader time.time_ns()."
    )


if __name__ == "__main__":
    main()
