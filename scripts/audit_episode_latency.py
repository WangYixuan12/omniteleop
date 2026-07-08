#!/usr/bin/env python3
"""Per-take camera-latency audit for wbc_vr_robot --record episodes.

Reads the timing metadata written by the timing-aware recorder (2026-07) --
``obs/images/{head_frame_ns,left_wrist_frame_ns,grab_wall_ns}`` per frame plus optional
``obs/images/head_depth_frame_ns`` when depth carries its own stamp, plus
the once-per-run ``meta/ntp`` clock calibration -- and reports, per episode:

  * record cadence (tick dt median/p99/max, hold gaps);
  * per-camera staleness D = grab_wall_ns - frame_ns in ms: the camera-publisher
    capture stamp subtracted from the local wall clock right after the grab. Raw
    D spans the unknown camera-host->local clock offset, so the JITTER (D - min D)
    is the primary readout: it is the part of the image age the policy cannot
    learn away. With ``meta/camera_ntp/{head,wrist}`` present, ABSOLUTE staleness
    uses the matching camera-host offset. It falls back to legacy ``meta/ntp`` only
    with explicit wording: that offset is robot SoC - local, which may not equal the
    camera-publisher-host clock.
  * head-vs-wrist same-tick capture skew (both stamps are camera-publisher-host
    timestamps in the deployed same-host publisher setup, so this needs no
    workstation offset);
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
_OPTIONAL_PER_FRAME_KEYS = ("head_depth_frame_ns",)
_CAMERA_NTP_LABELS = ("head", "wrist")
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


def audit_episode(path: pathlib.Path) -> dict:
    """Compute the audit stats for one episode; see the module docstring.

    Returns ``{"path", "frames", "cadence_ms", "has_capture_stamps"}`` plus --
    when the take has capture stamps -- ``"head"``/``"wrist"`` staleness-D
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
    for label, key in (("head", "head_frame_ns"), ("wrist", "left_wrist_frame_ns")):
        frame_ns = timing[key]
        d_ms = (grab - frame_ns) / 1e6
        stats = _stats_ms(d_ms)
        camera_offset_ns = timing.get(f"camera_ntp_{label}_offset_ns")
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
    skew_ms = (timing["head_frame_ns"] - timing["left_wrist_frame_ns"]) / 1e6
    result["skew_ms"] = {
        "median": float(np.median(skew_ms)),
        "p90_abs": float(np.percentile(np.abs(skew_ms), 90)),
        "max_abs": float(np.abs(skew_ms).max()),
    }
    return result


def _print_report(result: dict) -> None:
    print(f"\n=== {result['path']} ({result['frames']} frames) ===")
    cadence = result.get("cadence_ms")
    if cadence is not None:
        print(f"record cadence: median {cadence['median']:6.1f} ms  "
              f"p99 {cadence['p99']:6.1f}  max {cadence['max']:6.1f}  "
              f"hold gaps >1.5x: {cadence['gaps']}")
    if not result["has_capture_stamps"]:
        print("no capture stamps (old-format take) -- cadence audit only; re-record "
              "with the timing-aware recorder for the latency blocks")
        return
    if "camera_ntp" in result:
        parts = []
        for label in ("head", "wrist"):
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
    for label in ("head", "wrist"):
        stats = result[label]
        period = result.get(f"{label}_frame_period_ms")
        beat_txt = ""
        if period is not None:
            beat_std = period * _UNIFORM_BEAT_STD_FRAC
            beat_txt = (f"  | frame period ~{period:.1f} ms "
                        f"(pure sampling-beat std would be {beat_std:.1f} ms)")
        print(f"{label:5s} staleness D: p50 {stats['p50']:6.1f} ms  p90 {stats['p90']:6.1f}  "
              f"max {stats['max']:6.1f}  | jitter p90 {stats['jitter_p90']:5.1f} ms  "
              f"max {stats['jitter_max']:5.1f}  std {stats['std']:5.1f}{beat_txt}")
        if "absolute_p50" in stats:
            source = stats.get("absolute_source", "unknown")
            label_txt = "absolute(camera)" if source == "camera_host" else "absolute(SoC fallback)"
            print(f"      {label_txt:17s} min {stats['absolute_min']:6.1f} ms  "
                  f"p50 {stats['absolute_p50']:6.1f}  max {stats['absolute_max']:6.1f}")
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
    print(f"head-vs-wrist capture skew: median {skew['median']:+6.1f} ms  "
          f"p90|.| {skew['p90_abs']:6.1f}  max|.| {skew['max_abs']:6.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+",
                        help="raw episode .hdf5 files, porter timing-sidecar .npz files, "
                             "or directories containing them")
    args = parser.parse_args()
    for path in _expand_paths(args.paths):
        _print_report(audit_episode(path))


if __name__ == "__main__":
    main()
