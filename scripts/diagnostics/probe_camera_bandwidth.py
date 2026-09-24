#!/usr/bin/env python3
"""Measure on-wire bandwidth per camera stream — the WiFi load that drives lag.

Subscribes (raw zenoh, like probe_camera_timestamps.py) to several
``sensors/*`` keys at once and, over a window, sums ``len(sample.payload)`` per
key. The payload is the *encoded* (JPEG/compressed-depth) bytes that actually
cross the robot→workstation WiFi link, so bytes/s here == the uplink load that
queues the teleop command path. It uses the same DexComm codecs as the recorder,
checks every publisher capture stamp, and queries the live clock + stereo-info
services to report corrected receive age, cross-camera skew, and calibration.

Run on a host on the robot's Zenoh fabric, with the publishers running::

    # Default: both eyes from all three physical cameras, with SDK depth off.
    python scripts/diagnostics/probe_camera_bandwidth.py --duration 6
"""

from __future__ import annotations

import argparse
import os
import statistics
import threading
import time
from collections import Counter, deque

import numpy as np
from dexcomm import Node
from dexcomm.codecs import DepthImageCodec, JsonDataCodec, RGBImageCodec
from omniteleop.common.stereo_quality import (
    aggregate_rectification_stats,
    feature_rectification_stats,
)

# Importing dexcontrol sets $ZENOH_CONFIG to the robot's mTLS peer config.
import dexcontrol  # noqa: F401

# Documented full-res @30fps baseline (memory: head-wrist-lag-wifi-bufferbloat),
# for a before/after reference. Mbps.
_BASELINE_MBPS = {
    "sensors/head_camera/left_rgb": 8.3,
    "sensors/head_camera/right_rgb": 8.3,
    "sensors/head_camera/depth": 57.9,
    "sensors/wrist_zedm/left_rgb": 13.7,
    "sensors/wrist_zedm/right_rgb": 13.0,
    "sensors/left_wrist_zedm/left_rgb": 13.7,
    "sensors/left_wrist_zedm/right_rgb": 13.0,
    "sensors/right_wrist_zedm/left_rgb": 13.7,
    "sensors/right_wrist_zedm/right_rgb": 13.0,
}


def _decoder_for(topic: str):
    return DepthImageCodec.decode if "depth" in topic else RGBImageCodec.decode


def _sensor_id(topic: str) -> str:
    parts = topic.strip("/").split("/")
    if len(parts) < 3 or parts[-3] != "sensors":
        raise ValueError(f"camera topic must look like sensors/<id>/<stream>, got {topic!r}")
    return parts[-2]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _query_clock(node: Node, sensor_id: str, sample_count: int = 9) -> dict:
    """NTP-style publisher-host minus local offset through the recorder service."""
    client = node.create_service_client(
        service_name=f"sensors/{sensor_id}/clock",
        request_encoder=JsonDataCodec.encode,
        response_decoder=JsonDataCodec.decode,
        timeout=1.0,
    )
    samples: list[tuple[int, int]] = []
    source = "unknown"
    try:
        for i in range(sample_count):
            t0 = time.time_ns()
            reply = client.call(
                {
                    "client_send_time_ns": t0,
                    "sample_count": sample_count,
                    "sample_index": i,
                }
            )
            t3 = time.time_ns()
            if not isinstance(reply, dict):
                continue
            t1 = int(reply.get("server_receive_time_ns", 0))
            t2 = int(reply.get("server_send_time_ns", 0))
            if t1 <= 0 or t2 < t1:
                continue
            offset = round(((t1 - t0) + (t2 - t3)) / 2.0)
            rtt = t3 - t0
            if rtt > 0:
                samples.append((int(rtt), int(offset)))
                source = str(reply.get("source", source))
            if i + 1 < sample_count:
                time.sleep(0.01)
    finally:
        shutdown = getattr(client, "shutdown", None)
        if shutdown is not None:
            shutdown()
    if len(samples) < max(3, sample_count // 3):
        raise RuntimeError(
            f"{sensor_id} clock returned only {len(samples)}/{sample_count} usable samples"
        )
    samples.sort()
    keep = samples[:max(1, (len(samples) + 1) // 2)]
    return {
        "offset_ns": round(statistics.mean(offset for _rtt, offset in keep)),
        "rtt_ns": round(statistics.mean(rtt for rtt, _offset in keep)),
        "source": source,
    }


def _query_info(node: Node, sensor_id: str) -> dict:
    client = node.create_service_client(
        service_name=f"sensors/{sensor_id}/info",
        request_encoder=JsonDataCodec.encode,
        response_decoder=JsonDataCodec.decode,
        timeout=2.0,
    )
    try:
        reply = client.call({})
    finally:
        shutdown = getattr(client, "shutdown", None)
        if shutdown is not None:
            shutdown()
    if not isinstance(reply, dict):
        raise RuntimeError(f"{sensor_id} /info returned {type(reply).__name__}")
    return reply


def _stereo_info_error(info: dict) -> str | None:
    streams = info.get("streams")
    right = streams.get("right_rgb") if isinstance(streams, dict) else None
    stereo = info.get("stereo")
    if not isinstance(right, dict) or right.get("enabled") is not True:
        return "right_rgb is not enabled"
    if not isinstance(stereo, dict):
        return "missing stereo block"
    if stereo.get("rectified") is not True:
        return "stereo.rectified is not true"
    baseline = stereo.get("baseline_m")
    if isinstance(baseline, bool) or not isinstance(baseline, (int, float)):
        return f"invalid baseline_m={baseline!r}"
    if not 0.01 < float(baseline) < 0.5:
        return f"implausible baseline_m={baseline!r}"
    for key in ("left_K", "right_K"):
        K = np.asarray(stereo.get(key))
        if K.shape != (3, 3) or not np.all(np.isfinite(K)):
            return f"invalid {key} shape/values"
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            return f"invalid {key} focal length"
    return None


def _resolution_description(info: dict) -> str:
    """Human-readable actual publisher geometry across old/new info schemas."""
    actual = info.get("actual")
    if isinstance(actual, dict):
        width, height, fps = (
            actual.get("width"),
            actual.get("height"),
            actual.get("fps"),
        )
        if all(isinstance(value, (int, float)) for value in (width, height, fps)):
            return f"{int(width)}x{int(height)}@{float(fps):g}Hz"
    fallback = info.get("actual_resolution", info.get("resolution"))
    return "?" if fallback is None else str(fallback)


class _Stat:
    __slots__ = (
        "arrival_ns",
        "capture_ns",
        "count",
        "decode_errors",
        "dtype",
        "sample_frames",
        "shape",
        "total_bytes",
    )

    def __init__(self) -> None:
        self.count = 0
        self.total_bytes = 0
        self.shape = None
        self.dtype = None
        self.capture_ns: list[int] = []
        self.arrival_ns: list[int] = []
        self.decode_errors = 0
        # One decoded RGB frame per publisher-clock second, capped for bounded RAM.
        # Same-capture eyes make the same deterministic bucket choice after the first
        # partial subscription second, giving samples distributed across the run.
        self.sample_frames: dict[int, tuple[int, np.ndarray]] = {}


def _corrected_ages_ms(stat: _Stat, offset_ns: int) -> list[float]:
    return [
        (arrival - (capture - offset_ns)) / 1e6
        for capture, arrival in zip(stat.capture_ns, stat.arrival_ns, strict=True)
    ]


def _live_rectification_stats(left: _Stat, right: _Stat, maximum: int = 5):
    """Empirical rectification from exact-capture live sample pairs."""
    left_by_capture = {
        int(capture_ns): frame for capture_ns, frame in left.sample_frames.values()
    }
    right_by_capture = {
        int(capture_ns): frame for capture_ns, frame in right.sample_frames.values()
    }
    common = sorted(set(left_by_capture) & set(right_by_capture))
    if not common:
        return None
    positions = np.unique(
        np.linspace(0, len(common) - 1, min(maximum, len(common)), dtype=np.int64)
    )
    rows = []
    for position in positions:
        capture_ns = common[int(position)]
        row = feature_rectification_stats(
            left_by_capture[capture_ns], right_by_capture[capture_ns]
        )
        if row is not None:
            rows.append(row)
    result = aggregate_rectification_stats(rows)
    if result is not None:
        result["sampled_pairs"] = int(len(positions))
        result["usable_pairs"] = int(len(rows))
    return result


def _nearest_skews_ms(
    anchor: _Stat,
    other: _Stat,
    *,
    anchor_offset_ns: int,
    other_offset_ns: int,
) -> list[float]:
    anchor_local = np.unique(
        np.asarray(anchor.capture_ns, dtype=np.int64) - anchor_offset_ns
    )
    other_local = np.sort(
        np.unique(np.asarray(other.capture_ns, dtype=np.int64) - other_offset_ns)
    )
    if anchor_local.size == 0 or other_local.size == 0:
        return []
    # Ignore subscription start/stop boundary anchors for which one publisher had
    # no overlapping history. The recorder likewise waits for a bounded wrist history.
    anchor_local = anchor_local[
        (anchor_local >= other_local[0]) & (anchor_local <= other_local[-1])
    ]
    indices = np.searchsorted(other_local, anchor_local)
    out: list[float] = []
    for stamp, index in zip(anchor_local, indices, strict=True):
        candidates = []
        if index < len(other_local):
            candidates.append(abs(int(stamp) - int(other_local[index])))
        if index:
            candidates.append(abs(int(stamp) - int(other_local[index - 1])))
        out.append(min(candidates) / 1e6)
    return out


def _simulate_recorder(
    stats: dict[str, _Stat],
    clocks: dict[str, dict],
    *,
    poll_hz: float,
    record_hz: float,
    max_age_ms: float,
    max_skew_ms: float,
    stale_grace_ms: float,
) -> dict:
    """Replay live callback events through the recorder's 100/10 Hz camera gates."""
    sensor_ids = ("head_camera", "left_wrist_zedm", "right_wrist_zedm")
    topics = {
        (sensor_id, eye): f"sensors/{sensor_id}/{eye}_rgb"
        for sensor_id in sensor_ids
        for eye in ("left", "right")
    }
    missing = [
        topic
        for topic in topics.values()
        if topic not in stats or not stats[topic].capture_ns
    ]
    missing_clocks = [sensor_id for sensor_id in sensor_ids if sensor_id not in clocks]
    if missing or missing_clocks:
        raise ValueError(
            f"cannot simulate recorder; missing streams={missing}, clocks={missing_clocks}"
        )
    events = {
        key: sorted(
            zip(stats[topic].arrival_ns, stats[topic].capture_ns, strict=True)
        )
        for key, topic in topics.items()
    }
    # Let every subscription acquire at least one complete pair, then keep clear of
    # teardown boundary effects at the end of the finite measurement window.
    start_ns = max(stream[0][0] for stream in events.values()) + 100_000_000
    end_ns = min(stream[-1][0] for stream in events.values()) - 50_000_000
    if end_ns <= start_ns:
        raise ValueError("measurement window is too short for recorder simulation")

    poll_period_ns = round(1e9 / poll_hz)
    record_period_ns = round(1e9 / record_hz)
    max_age_ns = round(max_age_ms * 1e6)
    max_skew_ns = round(max_skew_ms * 1e6)
    stale_grace_ns = round(stale_grace_ms * 1e6)
    pointers = {key: -1 for key in events}
    latest: dict[tuple[str, str], int] = {}
    head_history = deque(maxlen=16)
    wrist_history = {
        sensor_id: deque(maxlen=16) for sensor_id in sensor_ids[1:]
    }
    last_buffered_head = -1
    last_buffered = {sensor_id: -1 for sensor_id in sensor_ids[1:]}
    last_used = {sensor_id: -1 for sensor_id in sensor_ids}
    successes: list[int] = []
    accepted_ages_ms: list[float] = []
    accepted_skews_ms: list[float] = []
    rejects: Counter[str] = Counter()
    reject_since_ns: int | None = None
    longest_reject_ns = 0
    next_record_ns = start_ns

    # Poll during warmup too: a real recorder has already accumulated complete
    # stereo pairs when recording starts. Empty histories at start_ns fabricate
    # an initial rejection whenever the latest eyes straddle a capture boundary.
    warmup_ticks = (100_000_000 + poll_period_ns - 1) // poll_period_ns
    warmup_start_ns = start_ns - warmup_ticks * poll_period_ns
    for tick_ns in range(warmup_start_ns, end_ns + 1, poll_period_ns):
        for key, stream in events.items():
            pointer = pointers[key]
            while pointer + 1 < len(stream) and stream[pointer + 1][0] <= tick_ns:
                pointer += 1
            pointers[key] = pointer
            if pointer >= 0:
                latest[key] = int(stream[pointer][1])

        # Equivalent to _poll_wrist_alignment_buffers(): a history item exists only
        # once both eyes' latest cache entries describe the same physical capture.
        for sensor_id in sensor_ids[1:]:
            left = latest.get((sensor_id, "left"))
            right = latest.get((sensor_id, "right"))
            if left is not None and left == right and left > last_buffered[sensor_id]:
                wrist_history[sensor_id].append(left)
                last_buffered[sensor_id] = left
        head_left = latest.get(("head_camera", "left"))
        head_right = latest.get(("head_camera", "right"))
        if (
            head_left is not None
            and head_left == head_right
            and head_left > last_buffered_head
        ):
            head_history.append(head_left)
            last_buffered_head = head_left

        if tick_ns < next_record_ns:
            continue
        next_record_ns += record_period_ns
        if tick_ns > next_record_ns:
            next_record_ns = tick_ns + record_period_ns

        reason = None
        selected: dict[str, int] = {}
        skews_ns: dict[str, int] = {}
        selected_head = None
        if head_left is None or head_right is None:
            reason = "missing_head"
        else:
            head_candidates = [
                stamp
                for stamp in reversed(head_history)
                if stamp > last_used["head_camera"]
            ]
            for candidate_head in head_candidates:
                candidate_selected: dict[str, int] = {}
                candidate_skews: dict[str, int] = {}
                head_local = (
                    candidate_head - clocks["head_camera"]["offset_ns"]
                )
                for sensor_id in sensor_ids[1:]:
                    candidates = [
                        stamp
                        for stamp in wrist_history[sensor_id]
                        if stamp > last_used[sensor_id]
                    ]
                    if not candidates:
                        reason = f"missing_{sensor_id}"
                        break
                    chosen = min(
                        candidates,
                        key=lambda stamp: (
                            abs(
                                head_local
                                - (stamp - clocks[sensor_id]["offset_ns"])
                            ),
                            -stamp,
                        ),
                    )
                    skew = head_local - (
                        chosen - clocks[sensor_id]["offset_ns"]
                    )
                    candidate_selected[sensor_id] = chosen
                    candidate_skews[sensor_id] = skew
                    if abs(skew) > max_skew_ns:
                        reason = f"skew_{sensor_id}"
                        break
                else:
                    selected_head = candidate_head
                    selected = candidate_selected
                    skews_ns = candidate_skews
                    reason = None
                    break
            if selected_head is None and not head_candidates:
                reason = "head_pair" if head_left != head_right else "stale_head"
            if reason is None:
                assert selected_head is not None
                head_local = (
                    selected_head - clocks["head_camera"]["offset_ns"]
                )
                ages_ns = {
                    "head_camera": tick_ns - head_local,
                    **{
                        sensor_id: tick_ns
                        - (stamp - clocks[sensor_id]["offset_ns"])
                        for sensor_id, stamp in selected.items()
                    },
                }
                if any(age < -10_000_000 or age > max_age_ns for age in ages_ns.values()):
                    reason = "capture_age"

        if reason is not None:
            rejects[reason] += 1
            next_record_ns = tick_ns
            if reject_since_ns is None:
                reject_since_ns = tick_ns
            longest_reject_ns = max(longest_reject_ns, tick_ns - reject_since_ns)
            continue

        reject_since_ns = None
        assert selected_head is not None
        successes.append(tick_ns)
        last_used["head_camera"] = selected_head
        last_used.update(selected)
        accepted_ages_ms.extend(age / 1e6 for age in ages_ns.values())
        accepted_skews_ms.extend(abs(skew) / 1e6 for skew in skews_ns.values())

    gaps_ms = [
        (right - left) / 1e6
        for left, right in zip(successes, successes[1:])
    ]
    span_s = (successes[-1] - successes[0]) / 1e9 if len(successes) >= 2 else 0.0
    achieved_hz = (len(successes) - 1) / span_s if span_s > 0 else 0.0
    return {
        "rows": len(successes),
        "achieved_hz": achieved_hz,
        "gap_p50_ms": statistics.median(gaps_ms) if gaps_ms else float("nan"),
        "gap_max_ms": max(gaps_ms) if gaps_ms else float("nan"),
        "age_max_ms": max(accepted_ages_ms) if accepted_ages_ms else float("nan"),
        "skew_max_ms": max(accepted_skews_ms) if accepted_skews_ms else float("nan"),
        "longest_reject_ms": longest_reject_ns / 1e6,
        "would_abort": longest_reject_ns > stale_grace_ns,
        "rejects": dict(rejects),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument(
        "--topics",
        nargs="+",
        default=[
            "sensors/head_camera/left_rgb",
            "sensors/head_camera/right_rgb",
            "sensors/left_wrist_zedm/left_rgb",
            "sensors/left_wrist_zedm/right_rgb",
            "sensors/right_wrist_zedm/left_rgb",
            "sensors/right_wrist_zedm/right_rgb",
        ],
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="exit non-zero unless all streams sustain --min-fps, stereo timestamps "
             "pair exactly, and corrected age/skew stay within the recorder bounds",
    )
    ap.add_argument("--min-fps", type=float, default=12.0)
    ap.add_argument("--max-age-ms", type=float, default=150.0)
    ap.add_argument("--max-skew-ms", type=float, default=40.0)
    ap.add_argument("--poll-hz", type=float, default=100.0)
    ap.add_argument("--record-hz", type=float, default=10.0)
    ap.add_argument("--stale-grace-ms", type=float, default=250.0)
    ap.add_argument("--max-vertical-p90-px", type=float, default=5.0)
    ap.add_argument("--min-positive-disparity-fraction", type=float, default=0.7)
    ap.add_argument("--min-rectification-matches", type=int, default=15)
    args = ap.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        ap.error("--duration must be finite and > 0")
    if not np.isfinite(args.min_fps) or args.min_fps <= 0:
        ap.error("--min-fps must be finite and > 0")
    if not np.isfinite(args.poll_hz) or args.poll_hz <= 0:
        ap.error("--poll-hz must be finite and > 0")
    if not np.isfinite(args.record_hz) or args.record_hz <= 0:
        ap.error("--record-hz must be finite and > 0")
    if not np.isfinite(args.max_age_ms) or args.max_age_ms <= 0:
        ap.error("--max-age-ms must be finite and > 0")
    if not np.isfinite(args.max_skew_ms) or args.max_skew_ms < 0:
        ap.error("--max-skew-ms must be finite and >= 0")
    if not np.isfinite(args.stale_grace_ms) or args.stale_grace_ms < 0:
        ap.error("--stale-grace-ms must be finite and >= 0")
    if (
        not np.isfinite(args.max_vertical_p90_px)
        or args.max_vertical_p90_px <= 0
    ):
        ap.error("--max-vertical-p90-px must be finite and > 0")
    if (
        not np.isfinite(args.min_positive_disparity_fraction)
        or not 0 <= args.min_positive_disparity_fraction <= 1
    ):
        ap.error("--min-positive-disparity-fraction must be finite in [0,1]")
    if args.min_rectification_matches <= 0:
        ap.error("--min-rectification-matches must be > 0")

    lock = threading.Lock()
    stats: dict[str, _Stat] = {t: _Stat() for t in args.topics}
    node = Node(name="camera_bandwidth_probe")
    print(f"ROBOT_NAME={os.getenv('ROBOT_NAME')!r}")
    print(f"ZENOH_CONFIG={os.getenv('ZENOH_CONFIG')!r}")
    print(f"namespace={node.namespace!r}")
    sensor_ids = tuple(dict.fromkeys(_sensor_id(topic) for topic in args.topics))

    clocks: dict[str, dict] = {}
    infos: dict[str, dict] = {}
    service_errors: dict[str, str] = {}
    for sensor_id in sensor_ids:
        try:
            clocks[sensor_id] = _query_clock(node, sensor_id)
            infos[sensor_id] = _query_info(node, sensor_id)
        except Exception as exc:
            service_errors[sensor_id] = f"{type(exc).__name__}: {exc}"

    subs = []

    def make_cb(topic: str):
        decoder = _decoder_for(topic)
        st = stats[topic]

        def on_sample(message) -> None:
            arrival_ns = time.time_ns()
            payload = getattr(message, "data", message)
            try:
                payload = bytes(payload)
            except (TypeError, ValueError):
                with lock:
                    st.decode_errors += 1
                return
            try:
                decoded = decoder(payload)
                array = decoded.get("data") if isinstance(decoded, dict) else None
                capture_ns = int(
                    decoded.get("timestamp_ns", decoded.get("timestamp", 0))
                    if isinstance(decoded, dict)
                    else 0
                )
                if array is None or capture_ns <= 0:
                    raise ValueError("decoded payload lacks data/timestamp_ns")
            except Exception:
                with lock:
                    st.decode_errors += 1
                return
            with lock:
                st.count += 1
                st.total_bytes += len(payload)
                st.capture_ns.append(capture_ns)
                st.arrival_ns.append(arrival_ns)
                if st.shape is None:
                    st.shape, st.dtype = array.shape, array.dtype
                if "depth" not in topic:
                    bucket = capture_ns // 1_000_000_000
                    if bucket not in st.sample_frames:
                        if len(st.sample_frames) >= 20:
                            del st.sample_frames[min(st.sample_frames)]
                        st.sample_frames[bucket] = (capture_ns, array)

        return on_sample

    try:
        for t in args.topics:
            subs.append(
                node.create_subscriber(
                    topic=t,
                    callback=make_cb(t),
                    decoder=None,
                    buffer_size=1,
                )
            )
        print(f"measuring {len(args.topics)} keys for {args.duration:.1f}s ...\n")
        t0 = time.perf_counter()
        time.sleep(args.duration)
        elapsed = time.perf_counter() - t0
        for s in subs:
            s.shutdown()
    finally:
        node.shutdown()

    failures: list[str] = []
    print("publisher services:")
    for sensor_id in sensor_ids:
        if sensor_id in service_errors:
            print(f"  {sensor_id:20s} ERROR {service_errors[sensor_id]}")
            if args.verify:
                failures.append(f"{sensor_id}: {service_errors[sensor_id]}")
            continue
        clock = clocks[sensor_id]
        info = infos[sensor_id]
        stereo_error = _stereo_info_error(info)
        stereo = info.get("stereo", {})
        resolution = _resolution_description(info)
        print(
            f"  {sensor_id:20s} clock camera-local={clock['offset_ns'] / 1e6:+7.2f}ms "
            f"rtt={clock['rtt_ns'] / 1e6:5.2f}ms resolution={resolution!r} "
            f"stereo={'ERROR ' + stereo_error if stereo_error else 'rectified, baseline=' + format(float(stereo['baseline_m']) * 1000, '.2f') + 'mm'}"
        )
        if stereo_error is not None and args.verify:
            failures.append(f"{sensor_id}: {stereo_error}")

    print(f"\n{'topic':40s} {'fps':>6s} {'uniq':>6s} {'KB/frm':>8s} {'Mbps':>8s} "
          f"{'age p50/p95/max ms':>23s} {'shape':>16s}")
    print("-" * 122)
    total = 0.0
    base_total = 0.0
    for t in args.topics:
        st = stats[t]
        fps = st.count / elapsed
        unique = len(set(st.capture_ns))
        kb = (st.total_bytes / st.count / 1024.0) if st.count else 0.0
        mbps = st.total_bytes * 8 / elapsed / 1e6
        total += mbps
        base = _BASELINE_MBPS.get(t)
        base_total += base or 0.0
        shape_s = f"{st.shape}" if st.shape is not None else "(no frames)"
        sensor_id = _sensor_id(t)
        ages = (
            _corrected_ages_ms(st, clocks[sensor_id]["offset_ns"])
            if sensor_id in clocks
            else []
        )
        age_s = (
            f"{statistics.median(ages):5.1f}/{_percentile(ages, 95):5.1f}/{max(ages):5.1f}"
            if ages else "n/a"
        )
        print(
            f"{t:40s} {fps:6.1f} {unique:6d} {kb:8.1f} {mbps:8.2f} "
            f"{age_s:>23s} {shape_s:>16s}"
        )
        if args.verify:
            if fps < args.min_fps:
                failures.append(f"{t}: {fps:.2f} fps < {args.min_fps:g}")
            if unique != st.count:
                failures.append(f"{t}: {st.count - unique} duplicate capture stamp(s)")
            if st.decode_errors:
                failures.append(f"{t}: {st.decode_errors} decode error(s)")
            if ages and max(ages) > args.max_age_ms:
                failures.append(
                    f"{t}: max corrected arrival age {max(ages):.1f}ms > "
                    f"{args.max_age_ms:g}ms"
                )
    print("-" * 122)
    print(f"{'TOTAL':40s} {'':6s} {'':6s} {'':8s} {total:8.2f} "
          f"(full-res @30fps reference {base_total:.1f} Mbps)")
    if base_total > 0 and total > 0:
        print(f"\n→ {base_total / total:.1f}x less WiFi uplink than the full-res @30fps baseline.")

    print("\nstereo same-capture pairing:")
    by_topic = stats
    for sensor_id in sensor_ids:
        left_topic = f"sensors/{sensor_id}/left_rgb"
        right_topic = f"sensors/{sensor_id}/right_rgb"
        if left_topic not in by_topic or right_topic not in by_topic:
            continue
        left = by_topic[left_topic]
        right = by_topic[right_topic]
        left_set, right_set = set(left.capture_ns), set(right.capture_ns)
        common = left_set & right_set
        denominator = max(1, min(len(left_set), len(right_set)))
        ratio = len(common) / denominator
        shape_ok = left.shape == right.shape and left.shape is not None
        print(
            f"  {sensor_id:20s} exact common={len(common)}/{denominator} "
            f"({ratio * 100:5.1f}%) shapes_equal={shape_ok}"
        )
        if args.verify and (ratio < 0.95 or not shape_ok):
            failures.append(
                f"{sensor_id}: stereo pairing {ratio * 100:.1f}% / "
                f"shapes_equal={shape_ok}"
            )
        rectification = _live_rectification_stats(left, right)
        if rectification is None:
            print("    pixel rectification: INCONCLUSIVE (too few robust matches)")
            if args.verify:
                failures.append(
                    f"{sensor_id}: empirical rectification has too few robust matches"
                )
            continue
        print(
            "    pixel rectification: "
            f"{rectification['matches']} mutual matches across "
            f"{rectification['usable_pairs']}/{rectification['sampled_pairs']} pairs, "
            f"vertical p50/p90/worst-p90="
            f"{rectification['vertical_p50_px']:.2f}/"
            f"{rectification['vertical_p90_px']:.2f}/"
            f"{rectification['vertical_p90_worst_px']:.2f}px, positive disparity="
            f"{rectification['positive_disparity_fraction'] * 100:.1f}%"
        )
        if args.verify:
            if rectification["matches"] < args.min_rectification_matches:
                failures.append(
                    f"{sensor_id}: only {rectification['matches']} robust stereo "
                    f"matches < {args.min_rectification_matches}"
                )
            if rectification["vertical_p90_px"] > args.max_vertical_p90_px:
                failures.append(
                    f"{sensor_id}: median p90 vertical residual "
                    f"{rectification['vertical_p90_px']:.2f}px > "
                    f"{args.max_vertical_p90_px:g}px"
                )
            if (
                rectification["positive_disparity_fraction"]
                < args.min_positive_disparity_fraction
            ):
                failures.append(
                    f"{sensor_id}: positive-disparity fraction "
                    f"{rectification['positive_disparity_fraction']:.3f} < "
                    f"{args.min_positive_disparity_fraction:g}; eyes may be swapped"
                )

    head_topic = "sensors/head_camera/left_rgb"
    if head_topic in by_topic and "head_camera" in clocks:
        print("\nbest-available corrected head→wrist capture skew:")
        for sensor_id in ("left_wrist_zedm", "right_wrist_zedm"):
            topic = f"sensors/{sensor_id}/left_rgb"
            if topic not in by_topic or sensor_id not in clocks:
                continue
            skews = _nearest_skews_ms(
                by_topic[head_topic],
                by_topic[topic],
                anchor_offset_ns=clocks["head_camera"]["offset_ns"],
                other_offset_ns=clocks[sensor_id]["offset_ns"],
            )
            print(
                f"  {sensor_id:20s} p50={statistics.median(skews):5.1f}ms "
                f"p95={_percentile(skews, 95):5.1f}ms max={max(skews):5.1f}ms"
            )
            if args.verify and max(skews) > args.max_skew_ms:
                failures.append(
                    f"{sensor_id}: nearest capture max skew {max(skews):.1f}ms > "
                    f"{args.max_skew_ms:g}ms"
                )

    try:
        simulation = _simulate_recorder(
            stats,
            clocks,
            poll_hz=args.poll_hz,
            record_hz=args.record_hz,
            max_age_ms=args.max_age_ms,
            max_skew_ms=args.max_skew_ms,
            stale_grace_ms=args.stale_grace_ms,
        )
    except ValueError as exc:
        simulation = None
        print(f"\nrecorder cadence simulation: ERROR {exc}")
        if args.verify:
            failures.append(f"recorder simulation: {exc}")
    if simulation is not None:
        print(
            f"\nrecorder cadence simulation ({args.poll_hz:g}Hz poll -> "
            f"{args.record_hz:g}Hz rows): rows={simulation['rows']} "
            f"rate={simulation['achieved_hz']:.2f}Hz, "
            f"gap p50/max={simulation['gap_p50_ms']:.1f}/"
            f"{simulation['gap_max_ms']:.1f}ms, accepted max age="
            f"{simulation['age_max_ms']:.1f}ms/skew="
            f"{simulation['skew_max_ms']:.1f}ms, longest retry="
            f"{simulation['longest_reject_ms']:.1f}ms, "
            f"would_abort={simulation['would_abort']}, rejects="
            f"{simulation['rejects']}"
        )
        if args.verify:
            if simulation["would_abort"]:
                failures.append("recorder simulation exceeds stale grace")
            if simulation["achieved_hz"] < args.record_hz * 0.95:
                failures.append(
                    f"simulated recorder rate {simulation['achieved_hz']:.2f}Hz < "
                    f"95% of {args.record_hz:g}Hz"
                )

    if args.verify:
        if failures:
            print("\nVERIFY: FAIL")
            for failure in failures:
                print(f"  - {failure}")
            raise SystemExit(1)
        print("\nVERIFY: PASS — six-eye publishers can supply the 10 Hz recorder contract")


if __name__ == "__main__":
    main()
