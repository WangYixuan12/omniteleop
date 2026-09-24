#!/usr/bin/env python3
"""Per-take camera-latency audit for wbc_vr_robot --record episodes.

Reads the timing metadata written by the timing-aware recorder (2026-07) --
``obs/images/{head_frame_ns,left_wrist_frame_ns,grab_wall_ns}`` per frame plus optional
``obs/images/{head_depth_frame_ns,right_wrist_frame_ns}`` when depth carries its own
stamp / the second wrist camera is recorded, plus
the once-per-run ``meta/ntp`` clock calibration -- and reports, per episode:

  * record cadence (tick dt median/p99/max, hold gaps);
  * per-camera staleness D = grab_wall_ns - frame_ns in ms: the camera-publisher
    capture stamp subtracted from the local wall clock right after the grab. Raw
    D spans the unknown camera-host->local clock offset, so the JITTER (D - min D)
    is the primary readout: it is the part of the image age the policy cannot
    learn away. With ``meta/camera_ntp/{head,left_wrist,right_wrist}`` present
    (legacy takes: ``wrist``), ABSOLUTE staleness
    uses the matching camera-host offset. It falls back to legacy ``meta/ntp`` only
    with explicit wording: that offset is robot SoC - local, which may not equal the
    camera-publisher-host clock.
  * head-vs-wrist same-row capture skew. When both camera clock calibrations exist,
    each capture is first mapped into the workstation clock; otherwise the report is
    explicitly marked as an uncorrected raw-publisher-clock comparison;
  * the implied publisher frame period (min positive stamp step).

Old-format takes (no capture stamps) get the cadence block only.

Accepts raw episodes (``episode_*.hdf5``), the porter's timing sidecars
(``debug/timing/episode_*.npz``, same keys), or directories -- a directory is
scanned for ``episode_*.hdf5`` then ``episode_*.npz``.

Runs in the dexmate env (h5py + numpy only)::

    python scripts/audit_episode_latency.py ~/Dexmate/data/raw_data/episode_42.hdf5
    python scripts/audit_episode_latency.py ~/Dexmate/data/raw_data
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

import numpy as np

_PER_FRAME_KEYS = ("head_frame_ns", "left_wrist_frame_ns", "grab_wall_ns")
# right_wrist_frame_ns arrived with the second wrist camera; optional so single-wrist
# takes still audit.
_OPTIONAL_PER_FRAME_KEYS = (
    "head_depth_frame_ns",
    "right_wrist_frame_ns",
    "head_right_frame_ns",
    "left_wrist_right_frame_ns",
    "right_wrist_right_frame_ns",
    "head_receive_ns",
    "head_depth_receive_ns",
    "head_right_receive_ns",
    "left_wrist_receive_ns",
    "left_wrist_right_receive_ns",
    "right_wrist_receive_ns",
    "right_wrist_right_receive_ns",
)
_CAUSAL_TIMING_KEYS = (
    "source_timestamp_ns",
    "source_receive_wall_ns",
    "action_dispatch_start_wall_ns",
    "action_dispatch_end_wall_ns",
    "state_read_start_wall_ns",
    "state_read_end_wall_ns",
)
# left/right name the ARM. "wrist" is the legacy single-camera label; absent labels
# are skipped, so old and new takes both report whatever calibration they carry.
_CAMERA_NTP_LABELS = ("head", "left_wrist", "right_wrist", "wrist")
_CAMERA_NTP_INT_KEYS = ("offset_ns", "rtt_ns", "queried_at_ns")
_CAMERA_NTP_STR_KEYS = ("sensor_id", "source")
# 15 fps at HD720/SVGA is the deployed publisher config; used only for the
# "sampling-beat" reference line in the report, never for computation.
_UNIFORM_BEAT_STD_FRAC = 1.0 / np.sqrt(12.0)


def _episode_sort_key(path: pathlib.Path) -> tuple[int, str]:
    match = re.search(r"episode_(\d+)", path.name)
    return (int(match.group(1)) if match else sys.maxsize, path.name)


def _expand_paths(inputs: list[str]) -> list[pathlib.Path]:
    """Resolve CLI inputs to episode files; directories scan hdf5 then npz."""
    out: list[pathlib.Path] = []
    for raw in inputs:
        path = pathlib.Path(raw).expanduser()
        if path.is_dir():
            found = sorted(path.glob("episode_*.hdf5"), key=_episode_sort_key)
            if not found:
                found = sorted(path.glob("episode_*.npz"), key=_episode_sort_key)
            if not found:
                raise FileNotFoundError(f"{path}: no episode_*.hdf5 or episode_*.npz inside")
            out.extend(found)
        elif path.is_file():
            out.append(path)
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")
    return out


def _validate_stamps(name: str, arr: np.ndarray, frame_count: int, source: str,
                     *, strictly_increasing: bool = True) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape != (frame_count,) or arr.dtype != np.int64:
        raise ValueError(f"{source}: {name} must be ({frame_count},) int64, got "
                         f"{arr.shape} {arr.dtype}")
    if np.any(arr <= 0):
        raise ValueError(f"{source}: {name} has non-positive stamps")
    if frame_count > 1 and strictly_increasing and np.any(np.diff(arr) <= 0):
        raise ValueError(f"{source}: {name} is not strictly increasing -- duplicate or "
                         "reordered frames; the recorder freshness gate forbids this")
    if frame_count > 1 and not strictly_increasing and np.any(np.diff(arr) < 0):
        raise ValueError(f"{source}: {name} is not monotonic -- reordered frames")
    return arr


def _decode_scalar_text(value: np.ndarray) -> str:
    raw = value.item()
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def load_timing(path: pathlib.Path) -> dict:
    """Timing arrays from a raw episode HDF5 or a porter timing-sidecar npz.

    Returns ``timestamp_ns`` (T,) int64 always; ``head_frame_ns``/
    ``left_wrist_frame_ns``/``grab_wall_ns`` (T,) int64 and ``ntp_offset_ns``/
    ``ntp_rtt_ns`` (int) when the take carries them (all-or-none per group,
    validated). Raises ValueError on malformed data.
    """
    source = str(path)
    if path.suffix == ".npz":
        with np.load(path) as data:
            raw = {key: np.asarray(data[key]) for key in data.files}
    elif path.suffix == ".hdf5":
        import h5py  # noqa: PLC0415 -- lazy: keeps npz-only use h5py-free

        raw = {}
        with h5py.File(path, "r") as f:
            if "timestamp_ns" not in f:
                raise ValueError(f"{source}: missing timestamp_ns -- not a recorder episode")
            raw["timestamp_ns"] = np.asarray(f["timestamp_ns"][()])
            for key in (*_PER_FRAME_KEYS, *_OPTIONAL_PER_FRAME_KEYS):
                if f"obs/images/{key}" in f:
                    raw[key] = np.asarray(f[f"obs/images/{key}"][()])
            for key in _CAUSAL_TIMING_KEYS:
                if f"timing/{key}" in f:
                    raw[key] = np.asarray(f[f"timing/{key}"][()])
            for key in ("offset_ns", "rtt_ns"):
                if f"meta/ntp/{key}" in f:
                    raw[f"ntp_{key}"] = np.asarray(f[f"meta/ntp/{key}"][()])
            for label in _CAMERA_NTP_LABELS:
                group_path = f"meta/camera_ntp/{label}"
                if group_path not in f:
                    continue
                expected = set(_CAMERA_NTP_INT_KEYS) | set(_CAMERA_NTP_STR_KEYS)
                present = {key for key in expected if f"{group_path}/{key}" in f}
                if present != expected:
                    raise ValueError(
                        f"{source}: partial {group_path} -- has {sorted(present)}, "
                        f"missing {sorted(expected - present)}"
                    )
                for key in expected:
                    raw[f"camera_ntp_{label}_{key}"] = np.asarray(
                        f[f"{group_path}/{key}"][()]
                    )
    else:
        raise ValueError(f"{source}: expected .hdf5 or .npz, got {path.suffix!r}")

    ts = np.asarray(raw["timestamp_ns"])
    if ts.ndim != 1 or ts.size == 0 or ts.dtype != np.int64:
        raise ValueError(f"{source}: timestamp_ns must be (T>=1,) int64, got "
                         f"{ts.shape} {ts.dtype}")
    if ts.size > 1 and np.any(np.diff(ts) <= 0):
        raise ValueError(f"{source}: timestamp_ns is not strictly increasing")
    out: dict = {"timestamp_ns": ts}

    present = [key for key in _PER_FRAME_KEYS if key in raw]
    optional_present = [key for key in _OPTIONAL_PER_FRAME_KEYS if key in raw]
    if present and len(present) != len(_PER_FRAME_KEYS):
        raise ValueError(f"{source}: partial capture stamps -- has {present}, needs all of "
                         f"{list(_PER_FRAME_KEYS)}")
    if optional_present and not present:
        raise ValueError(
            f"{source}: optional capture stamps {optional_present} exist without "
            f"{list(_PER_FRAME_KEYS)}"
        )
    for key in present:
        out[key] = _validate_stamps(key, raw[key], ts.size, source)
    for key in optional_present:
        out[key] = _validate_stamps(
            key, raw[key], ts.size, source, strictly_increasing=False
        )
    causal_present = [key for key in _CAUSAL_TIMING_KEYS if key in raw]
    if causal_present and not present:
        raise ValueError(
            f"{source}: causal timing bracket exists without required capture stamps "
            f"{list(_PER_FRAME_KEYS)}"
        )
    if causal_present and len(causal_present) != len(_CAUSAL_TIMING_KEYS):
        missing = sorted(set(_CAUSAL_TIMING_KEYS) - set(causal_present))
        raise ValueError(
            f"{source}: partial causal timing bracket -- has {causal_present}, "
            f"missing {missing}"
        )
    for key in causal_present:
        out[key] = _validate_stamps(
            key,
            raw[key],
            ts.size,
            source,
            strictly_increasing=(
                key not in {"source_timestamp_ns", "source_receive_wall_ns"}
            ),
        )
    if causal_present:
        checks = (
            (out["source_receive_wall_ns"] <= out["action_dispatch_start_wall_ns"],
             "source_receive <= action_dispatch_start"),
            (out["action_dispatch_start_wall_ns"] <= out["action_dispatch_end_wall_ns"],
             "action_dispatch_start <= action_dispatch_end"),
            (out["grab_wall_ns"] <= out["state_read_start_wall_ns"],
             "grab_wall <= state_read_start"),
            (out["action_dispatch_end_wall_ns"] <= out["state_read_start_wall_ns"],
             "action_dispatch_end <= state_read_start"),
            (out["state_read_start_wall_ns"] <= out["state_read_end_wall_ns"],
             "state_read_start <= state_read_end"),
            (out["state_read_end_wall_ns"] <= ts,
             "state_read_end <= timestamp_ns(commit)"),
        )
        for valid, relation in checks:
            if not np.all(valid):
                frame = int(np.flatnonzero(~valid)[0])
                raise ValueError(
                    f"{source}: causal timing order violated at frame {frame}: {relation}"
                )
    if "ntp_offset_ns" in raw:
        offset = np.asarray(raw["ntp_offset_ns"])
        if offset.shape != () or offset.dtype != np.int64:
            raise ValueError(f"{source}: ntp offset_ns must be 0-d int64, got "
                             f"{offset.shape} {offset.dtype}")
        out["ntp_offset_ns"] = int(offset)
        if "ntp_rtt_ns" in raw:
            out["ntp_rtt_ns"] = int(np.asarray(raw["ntp_rtt_ns"]))
    for label in _CAMERA_NTP_LABELS:
        expected = {
            f"camera_ntp_{label}_{key}"
            for key in (*_CAMERA_NTP_INT_KEYS, *_CAMERA_NTP_STR_KEYS)
        }
        present = expected & set(raw)
        if not present:
            continue
        if present != expected:
            raise ValueError(
                f"{source}: partial camera_ntp {label} -- has {sorted(present)}, "
                f"missing {sorted(expected - present)}"
            )
        for key in _CAMERA_NTP_INT_KEYS:
            name = f"camera_ntp_{label}_{key}"
            val = np.asarray(raw[name])
            if val.shape != () or val.dtype != np.int64:
                raise ValueError(
                    f"{source}: {name} must be 0-d int64, got {val.shape} {val.dtype}"
                )
            if key != "offset_ns" and int(val) <= 0:
                raise ValueError(f"{source}: {name} must be positive, got {int(val)}")
            out[name] = int(val)
        for key in _CAMERA_NTP_STR_KEYS:
            name = f"camera_ntp_{label}_{key}"
            val = np.asarray(raw[name])
            if val.shape != () or val.dtype.kind not in "SUO":
                raise ValueError(
                    f"{source}: {name} must be 0-d string/bytes, got {val.shape} {val.dtype}"
                )
            out[name] = _decode_scalar_text(val)
    return out


def _stats_ms(values_ms: np.ndarray) -> dict:
    rel = values_ms - values_ms.min()
    return {
        "min": float(values_ms.min()), "max": float(values_ms.max()),
        "p50": float(np.percentile(values_ms, 50)),
        "p90": float(np.percentile(values_ms, 90)),
        "p99": float(np.percentile(values_ms, 99)),
        "std": float(values_ms.std()),
        "jitter_p90": float(np.percentile(rel, 90)),
        "jitter_max": float(rel.max()),
    }


def _camera_offset_ns(timing: dict, label: str) -> int | None:
    """Matching camera-host-minus-local offset for one image stream."""
    value = timing.get(f"camera_ntp_{label}_offset_ns")
    if value is None and label == "left_wrist":
        value = timing.get("camera_ntp_wrist_offset_ns")  # legacy single-wrist label
    return None if value is None else int(value)


def _capture_skew_stats(
    timing: dict,
    *,
    wrist_label: str,
    wrist_key: str,
) -> dict:
    """Corrected ``head_local - wrist_local`` capture skew statistics."""
    head = timing["head_frame_ns"]
    wrist = timing[wrist_key]
    head_offset = _camera_offset_ns(timing, "head")
    wrist_offset = _camera_offset_ns(timing, wrist_label)
    if head_offset is not None and wrist_offset is not None:
        # local capture = publisher stamp - (publisher_host - local).
        skew_ms = ((head - head_offset) - (wrist - wrist_offset)) / 1e6
        clock_source = "camera_ntp_corrected"
    else:
        skew_ms = (head - wrist) / 1e6
        clock_source = "raw_publisher_clock_uncorrected"
    return {
        "median": float(np.median(skew_ms)),
        "p90_abs": float(np.percentile(np.abs(skew_ms), 90)),
        "max_abs": float(np.abs(skew_ms).max()),
        "clock_source": clock_source,
    }


def audit_episode(path: pathlib.Path) -> dict:
    """Compute the audit stats for one episode; see the module docstring.

    Returns ``{"path", "frames", "cadence_ms", "has_capture_stamps"}`` plus --
    when the take has capture stamps -- ``"head"``/``"left_wrist"`` and optional
    ``"right_wrist"`` staleness-D
    stats dicts (``"absolute_*"`` keys added when a camera-host or fallback SoC
    offset is present), ``"skew_ms"`` head-vs-wrist stats, per-camera
    ``"*_frame_period_ms"``, and clock-offset metadata.
    """
    timing = load_timing(path)
    ts = timing["timestamp_ns"]
    result: dict = {
        "path": str(path),
        "frames": int(ts.size),
        "has_capture_stamps": "grab_wall_ns" in timing,
    }
    if ts.size > 1:
        dt_ms = np.diff(ts) / 1e6
        median = float(np.median(dt_ms))
        result["cadence_ms"] = {
            "median": median, "p99": float(np.percentile(dt_ms, 99)),
            "max": float(dt_ms.max()),
            "gaps": int(np.sum(dt_ms > 1.5 * median)),
        }
    if "source_timestamp_ns" in timing:
        result["causal_timing"] = {
            # Raw because the producer timestamp can belong to another host clock.
            "source_publish_to_receive_raw": _stats_ms(
                (timing["source_receive_wall_ns"] - timing["source_timestamp_ns"]) / 1e6
            ),
            "source_receive_to_dispatch": _stats_ms(
                (timing["action_dispatch_start_wall_ns"]
                 - timing["source_receive_wall_ns"]) / 1e6
            ),
            "action_dispatch_duration": _stats_ms(
                (timing["action_dispatch_end_wall_ns"]
                 - timing["action_dispatch_start_wall_ns"]) / 1e6
            ),
            "dispatch_to_state": _stats_ms(
                (timing["state_read_start_wall_ns"]
                 - timing["action_dispatch_end_wall_ns"]) / 1e6
            ),
            "grab_to_state": _stats_ms(
                (timing["state_read_start_wall_ns"] - timing["grab_wall_ns"]) / 1e6
            ),
            "state_read_duration": _stats_ms(
                (timing["state_read_end_wall_ns"]
                 - timing["state_read_start_wall_ns"]) / 1e6
            ),
            "state_to_commit": _stats_ms(
                (ts - timing["state_read_end_wall_ns"]) / 1e6
            ),
        }
    if not result["has_capture_stamps"]:
        return result

    grab = timing["grab_wall_ns"]
    offset_ns = timing.get("ntp_offset_ns")
    if offset_ns is not None:
        result["ntp_offset_ms"] = offset_ns / 1e6
        if "ntp_rtt_ns" in timing:
            result["ntp_rtt_ms"] = timing["ntp_rtt_ns"] / 1e6
    camera_ntp: dict[str, dict] = {}
    for label in _CAMERA_NTP_LABELS:
        prefix = f"camera_ntp_{label}_"
        if f"{prefix}offset_ns" in timing:
            camera_ntp[label] = {
                "offset_ms": timing[f"{prefix}offset_ns"] / 1e6,
                "rtt_ms": timing[f"{prefix}rtt_ns"] / 1e6,
                "queried_at_ns": timing[f"{prefix}queried_at_ns"],
                "sensor_id": timing[f"{prefix}sensor_id"],
                "source": timing[f"{prefix}source"],
            }
    if camera_ntp:
        result["camera_ntp"] = camera_ntp
    # One staleness block per camera stream present. The camera_ntp label matches the
    # stream ("left_wrist" for left_wrist_frame_ns); legacy takes carry their
    # calibration under the bare "wrist" label, so fall back to it for the left camera.
    streams = [("head", "head_frame_ns"), ("left_wrist", "left_wrist_frame_ns")]
    if "right_wrist_frame_ns" in timing:
        streams.append(("right_wrist", "right_wrist_frame_ns"))
    for label, key in streams:
        frame_ns = timing[key]
        d_ms = (grab - frame_ns) / 1e6
        stats = _stats_ms(d_ms)
        camera_offset_ns = _camera_offset_ns(timing, label)
        if camera_offset_ns is not None:
            # offset = camera_host - local  => local capture time = frame_ns - offset
            # => absolute staleness = grab - (frame_ns - offset) = D + offset.
            stats["absolute_min"] = stats["min"] + camera_offset_ns / 1e6
            stats["absolute_p50"] = stats["p50"] + camera_offset_ns / 1e6
            stats["absolute_max"] = stats["max"] + camera_offset_ns / 1e6
            stats["absolute_source"] = "camera_host"
        elif offset_ns is not None:
            # Fallback only: offset = robot SoC - local. The ZED publishers may run
            # on a different host clock, so this is not authoritative camera staleness.
            # =>  absolute staleness = grab - (frame_ns - offset) = D + offset.
            stats["absolute_min"] = stats["min"] + offset_ns / 1e6
            stats["absolute_p50"] = stats["p50"] + offset_ns / 1e6
            stats["absolute_max"] = stats["max"] + offset_ns / 1e6
            stats["absolute_source"] = "soc_fallback"
        receive_key = f"{label}_receive_ns"
        if receive_key in timing:
            receive_ns = timing[receive_key]
            if np.any(receive_ns > grab):
                frame = int(np.flatnonzero(receive_ns > grab)[0])
                raise ValueError(
                    f"{path}: {receive_key}[{frame}] is later than grab_wall_ns"
                )
            # Both are local wall-clock: cache age needs no NTP correction.
            stats["cache_age"] = _stats_ms((grab - receive_ns) / 1e6)
            # capture is on camera-host time. Add camera_host-local offset to map
            # capture onto the local receive clock; without it only jitter is valid.
            delivery = _stats_ms((receive_ns - frame_ns) / 1e6)
            if camera_offset_ns is not None:
                delivery["absolute_min"] = delivery["min"] + camera_offset_ns / 1e6
                delivery["absolute_p50"] = delivery["p50"] + camera_offset_ns / 1e6
                delivery["absolute_max"] = delivery["max"] + camera_offset_ns / 1e6
                delivery["absolute_source"] = "camera_host"
            elif offset_ns is not None:
                delivery["absolute_min"] = delivery["min"] + offset_ns / 1e6
                delivery["absolute_p50"] = delivery["p50"] + offset_ns / 1e6
                delivery["absolute_max"] = delivery["max"] + offset_ns / 1e6
                delivery["absolute_source"] = "soc_fallback"
            stats["delivery"] = delivery
        result[label] = stats
        if frame_ns.size > 1:
            period = float(np.diff(frame_ns).min() / 1e6)
            result[f"{label}_frame_period_ms"] = period
    if "head_depth_frame_ns" in timing:
        depth_ns = timing["head_depth_frame_ns"]
        stats = _stats_ms((grab - depth_ns) / 1e6)
        camera_offset_ns = timing.get("camera_ntp_head_offset_ns")
        if camera_offset_ns is not None:
            stats["absolute_min"] = stats["min"] + camera_offset_ns / 1e6
            stats["absolute_p50"] = stats["p50"] + camera_offset_ns / 1e6
            stats["absolute_max"] = stats["max"] + camera_offset_ns / 1e6
            stats["absolute_source"] = "camera_host"
        elif offset_ns is not None:
            stats["absolute_min"] = stats["min"] + offset_ns / 1e6
            stats["absolute_p50"] = stats["p50"] + offset_ns / 1e6
            stats["absolute_max"] = stats["max"] + offset_ns / 1e6
            stats["absolute_source"] = "soc_fallback"
        result["head_depth"] = stats
        rgb_depth_skew_ms = (timing["head_frame_ns"] - depth_ns) / 1e6
        result["head_rgb_depth_skew_ms"] = {
            "median": float(np.median(rgb_depth_skew_ms)),
            "p90_abs": float(np.percentile(np.abs(rgb_depth_skew_ms), 90)),
            "max_abs": float(np.abs(rgb_depth_skew_ms).max()),
        }
    # head-vs-wrist capture skew, one entry per wrist present. "skew_ms" stays the
    # head-vs-LEFT-wrist figure so existing reports/consumers keep their key.
    result["skew_ms"] = _capture_skew_stats(
        timing,
        wrist_label="left_wrist",
        wrist_key="left_wrist_frame_ns",
    )
    if "right_wrist_frame_ns" in timing:
        result["right_wrist_skew_ms"] = _capture_skew_stats(
            timing,
            wrist_label="right_wrist",
            wrist_key="right_wrist_frame_ns",
        )
    return result


def print_report(result: dict) -> None:
    """Print one :func:`audit_episode` result for CLI and dataset-port use."""
    print(f"\n=== {result['path']} ({result['frames']} frames) ===")
    cadence = result.get("cadence_ms")
    if cadence is not None:
        print(f"record cadence: median {cadence['median']:6.1f} ms  "
              f"p99 {cadence['p99']:6.1f}  max {cadence['max']:6.1f}  "
              f"hold gaps >1.5x: {cadence['gaps']}")
    causal = result.get("causal_timing")
    if causal is not None:
        def _p50_max(name: str) -> str:
            stats = causal[name]
            return f"{stats['p50']:.2f}/{stats['max']:.2f} ms"

        print(
            "causal timing p50/max: source receive->dispatch "
            f"{_p50_max('source_receive_to_dispatch')}; dispatch duration "
            f"{_p50_max('action_dispatch_duration')}; dispatch->state "
            f"{_p50_max('dispatch_to_state')}; grab->state "
            f"{_p50_max('grab_to_state')}; state read "
            f"{_p50_max('state_read_duration')}; state->commit "
            f"{_p50_max('state_to_commit')}"
        )
        raw = causal["source_publish_to_receive_raw"]
        print(
            "source publish->receive raw: "
            f"p50 {raw['p50']:+.2f} ms, max {raw['max']:+.2f} ms "
            "(producer/local clock offset not calibrated)"
        )
    if not result["has_capture_stamps"]:
        print("no capture stamps (old-format take) -- cadence audit only; re-record "
              "with the timing-aware recorder for the latency blocks")
        return
    if "camera_ntp" in result:
        parts = []
        for label in ("head", "left_wrist", "right_wrist", "wrist"):
            if label in result["camera_ntp"]:
                cal = result["camera_ntp"][label]
                parts.append(
                    f"{label}({cal['sensor_id']}): offset {cal['offset_ms']:+.2f} ms, "
                    f"rtt {cal['rtt_ms']:.2f} ms"
                )
        print("camera_ntp: " + "; ".join(parts))
    if "ntp_offset_ms" in result:
        rtt = result.get("ntp_rtt_ms")
        rtt_txt = f", rtt {rtt:.2f} ms" if rtt is not None else ""
        fallback_txt = (
            "; used only as fallback when camera_ntp is missing"
            if "camera_ntp" in result else
            "; fallback for absolute camera staleness, may differ from camera-host clock"
        )
        print(f"ntp: robot(SoC) - local = {result['ntp_offset_ms']:+.2f} ms{rtt_txt}"
              f"{fallback_txt}")
    elif "camera_ntp" in result:
        print("ntp: robot(SoC) NOT calibrated; camera_ntp is present for absolute camera "
              "staleness")
    else:
        print("ntp: NOT calibrated (query_ntp failed at record time) -- staleness "
              "numbers below span the unknown clock offset; trust the jitter only")
    for label in ("head", "left_wrist", "right_wrist"):
        if label not in result:
            continue
        stats = result[label]
        period = result.get(f"{label}_frame_period_ms")
        beat_txt = ""
        if period is not None:
            beat_std = period * _UNIFORM_BEAT_STD_FRAC
            beat_txt = (f"  | frame period ~{period:.1f} ms "
                        f"(pure sampling-beat std would be {beat_std:.1f} ms)")
        print(f"{label} staleness D: p50 {stats['p50']:6.1f} ms  p90 {stats['p90']:6.1f}  "
              f"max {stats['max']:6.1f}  | jitter p90 {stats['jitter_p90']:5.1f} ms  "
              f"max {stats['jitter_max']:5.1f}  std {stats['std']:5.1f}{beat_txt}")
        if "absolute_p50" in stats:
            source = stats.get("absolute_source", "unknown")
            label_txt = "absolute(camera)" if source == "camera_host" else "absolute(SoC fallback)"
            print(f"      {label_txt:17s} min {stats['absolute_min']:6.1f} ms  "
                  f"p50 {stats['absolute_p50']:6.1f}  max {stats['absolute_max']:6.1f}")
        if "delivery" in stats:
            delivery = stats["delivery"]
            cache = stats["cache_age"]
            if "absolute_p50" in delivery:
                delivery_txt = (
                    f"capture->receive p50 {delivery['absolute_p50']:.1f} ms "
                    f"(max {delivery['absolute_max']:.1f})"
                )
            else:
                delivery_txt = (
                    f"capture->receive raw p50 {delivery['p50']:.1f} ms "
                    "(clock offset unknown)"
                )
            print(
                f"      {delivery_txt}; receive->grab cache age p50 "
                f"{cache['p50']:.1f} ms (max {cache['max']:.1f})"
            )
    if "head_depth" in result:
        stats = result["head_depth"]
        print(f"headD staleness D: p50 {stats['p50']:6.1f} ms  p90 {stats['p90']:6.1f}  "
              f"max {stats['max']:6.1f}  | jitter p90 {stats['jitter_p90']:5.1f} ms  "
              f"max {stats['jitter_max']:5.1f}  std {stats['std']:5.1f}")
        if "absolute_p50" in stats:
            source = stats.get("absolute_source", "unknown")
            label_txt = "absolute(camera)" if source == "camera_host" else "absolute(SoC fallback)"
            print(f"      {label_txt:17s} min {stats['absolute_min']:6.1f} ms  "
                  f"p50 {stats['absolute_p50']:6.1f}  max {stats['absolute_max']:6.1f}")
        skew = result["head_rgb_depth_skew_ms"]
        print(f"head RGB-depth skew (RGB minus depth): median {skew['median']:+6.1f} ms  "
              f"p90|.| {skew['p90_abs']:6.1f}  max|.| {skew['max_abs']:6.1f}")
    skew = result["skew_ms"]
    print(f"head-vs-left_wrist capture skew: median {skew['median']:+6.1f} ms  "
          f"p90|.| {skew['p90_abs']:6.1f}  max|.| {skew['max_abs']:6.1f}  "
          f"[{skew['clock_source']}]")
    if "right_wrist_skew_ms" in result:
        skew = result["right_wrist_skew_ms"]
        print(f"head-vs-right_wrist capture skew: median {skew['median']:+6.1f} ms  "
              f"p90|.| {skew['p90_abs']:6.1f}  max|.| {skew['max_abs']:6.1f}  "
              f"[{skew['clock_source']}]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+",
                        help="raw episode .hdf5 files, porter timing-sidecar .npz files, "
                             "or directories containing them")
    args = parser.parse_args()
    for path in _expand_paths(args.paths):
        print_report(audit_episode(path))


if __name__ == "__main__":
    main()
