#!/usr/bin/env python3
"""Match static-camera frames to teleop frames across the two recording hosts.

``record_static_cams.py`` (xarm7) and ``wbc_vr_robot.py --record`` (lambda) write
independent files on independent clocks. This puts both on the TELEOP host's clock
and emits, for every teleop frame, the index of the nearest static frame::

    static_on_teleop = capture_meta/camera_<c>_host_ns + teleop_clock_offset
    head_on_teleop   = obs/images/head_frame_ns - meta/camera_ntp/head/offset_ns

``teleop_clock_offset`` is interpolated between the pre- and post-take probes
(ROOT attrs ``teleop_clock_{pre,post}_offset_ns``) rather than held constant, so
clock drift across a long static take is corrected instead of assumed away.

The output is an INDEX, not a copy: a static take is tens of GB and the teleop
frames are already on disk, so duplicating pixels to express a mapping is wasted
disk. Downstream gathers ``static[idx]`` itself.

Default is report-only. Pass ``--out`` to write the index.

Usage::

    python scripts/align_static_teleop.py \
        --static /data/static_takes/packing/episode_0.hdf5 \
        --teleop .../recorded/dexmate/lambda/episode_*.hdf5 \
        --out .../recorded/dexmate/alignment.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

# A 10 Hz recorder sampling a free-running 30 Hz camera can occasionally select
# source frames 133 ms apart as their phases cross, even though no scheduled tick
# was dropped. Its worst nearest-neighbour gap is then about 67 ms. Leave a small
# margin for scheduling jitter while still rejecting genuinely stale frames.
DEFAULT_MAX_GAP_MS = 70.0


def static_on_teleop_clock(path: Path, cam: int) -> np.ndarray:
    """Static capture stamps (ns) re-expressed on the teleop host's clock."""
    with h5py.File(path, "r") as f:
        key = f"capture_meta/camera_{cam}_host_ns"
        if key not in f:
            raise ValueError(f"{path}: missing {key}; not a record_static_cams episode")
        host_ns = np.asarray(f[key], np.int64)
        a = dict(f.attrs)

    missing = [k for k in ("teleop_clock_pre_offset_ns", "teleop_clock_pre_measured_at_ns")
               if k not in a]
    if missing:
        raise ValueError(
            f"{path}: missing ROOT attrs {missing}. The take was recorded without "
            f"--teleop-host, so its clock cannot be related to the teleop host and "
            f"these frames are unalignable."
        )
    if host_ns.ndim != 1 or host_ns.size == 0:
        raise ValueError(f"{path}: camera_{cam}_host_ns must be non-empty 1-D, got {host_ns.shape}")

    pre_o, pre_t = int(a["teleop_clock_pre_offset_ns"]), int(a["teleop_clock_pre_measured_at_ns"])
    if "teleop_clock_post_offset_ns" in a:
        post_o = int(a["teleop_clock_post_offset_ns"])
        post_t = int(a["teleop_clock_post_measured_at_ns"])
    else:
        # Post-probe failed; hold the pre-take offset flat and let --report show it.
        post_o, post_t = pre_o, pre_t + 1
    offset = np.interp(host_ns.astype(np.float64), [pre_t, post_t], [pre_o, post_o])
    return host_ns + offset.astype(np.int64)


def head_on_teleop_clock(path: Path) -> np.ndarray:
    """Head-camera capture stamps (ns) re-expressed on the teleop host's clock."""
    with h5py.File(path, "r") as f:
        if "obs/images/head_frame_ns" not in f:
            raise ValueError(f"{path}: missing obs/images/head_frame_ns; not a teleop episode")
        head_ns = np.asarray(f["obs/images/head_frame_ns"], np.int64)
        key = "meta/camera_ntp/head/offset_ns"
        if key not in f:
            raise ValueError(
                f"{path}: missing {key}. head_frame_ns is stamped on the camera "
                f"publisher's clock, so without this offset it cannot be moved onto "
                f"the teleop host's clock."
            )
        offset_ns = int(f[key][()])
    if head_ns.ndim != 1 or head_ns.size == 0:
        raise ValueError(f"{path}: head_frame_ns must be non-empty 1-D, got {head_ns.shape}")
    return head_ns - offset_ns


def match(static_ns: np.ndarray, head_ns: np.ndarray, max_gap_ns: int) -> dict:
    """Nearest static frame for each teleop frame; -1 where the gap is too large."""
    if not np.all(np.diff(static_ns) >= 0):
        raise ValueError("static timestamps are not monotonic; refusing to match")
    right = np.searchsorted(static_ns, head_ns)
    left = np.clip(right - 1, 0, static_ns.size - 1)
    right = np.clip(right, 0, static_ns.size - 1)
    pick_right = np.abs(static_ns[right] - head_ns) < np.abs(static_ns[left] - head_ns)
    idx = np.where(pick_right, right, left)
    gap = np.abs(static_ns[idx] - head_ns)
    idx = np.where(gap <= max_gap_ns, idx, -1)
    return {"static_idx": idx, "gap_ns": gap}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--static", type=Path, required=True,
                    help="episode_<N>.hdf5 from record_static_cams.py")
    ap.add_argument("--teleop", type=Path, nargs="+", required=True,
                    help="one or more episode_<N>.hdf5 from wbc_vr_robot.py --record")
    ap.add_argument("--cam", type=int, default=0,
                    help="static camera whose stamps define the match (default 0)")
    ap.add_argument("--max-gap-ms", type=float, default=DEFAULT_MAX_GAP_MS,
                    help=f"reject matches beyond this gap (default {DEFAULT_MAX_GAP_MS:g})")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the index here as JSON (default: report only)")
    args = ap.parse_args()

    max_gap_ns = int(args.max_gap_ms * 1e6)
    s_ns = static_on_teleop_clock(args.static, args.cam)
    span = (s_ns[-1] - s_ns[0]) / 1e9
    print(f"static {args.static.name}: {s_ns.size} frames, {span:.0f}s, "
          f"{s_ns.size / span:.2f} Hz (camera_{args.cam}, on teleop clock)")

    with h5py.File(args.static, "r") as f:
        a = dict(f.attrs)
    if "teleop_clock_post_offset_ns" in a:
        drift = (int(a["teleop_clock_post_offset_ns"])
                 - int(a["teleop_clock_pre_offset_ns"])) / 1e6
        print(f"  clock vs {a['teleop_clock_pre_host']}: "
              f"{int(a['teleop_clock_pre_offset_ns']) / 1e6:+.2f} -> "
              f"{int(a['teleop_clock_post_offset_ns']) / 1e6:+.2f} ms "
              f"(drift {drift:+.2f} ms, interpolated)")
    else:
        print("  !! no post-take probe; offset held flat, drift uncorrected")

    index, total, matched = {}, 0, 0
    print(f"\n{'episode':18s} {'frames':>6s} {'in-span':>8s} {'matched':>8s} "
          f"{'med':>7s} {'p95':>7s} {'max':>7s}")
    for p in args.teleop:
        h_ns = head_on_teleop_clock(p)
        m = match(s_ns, h_ns, max_gap_ns)
        ok = m["static_idx"] >= 0
        inside = int(((h_ns >= s_ns[0]) & (h_ns <= s_ns[-1])).sum())
        g = m["gap_ns"][ok] / 1e6
        stats = (f"{np.median(g):6.1f} {np.percentile(g, 95):6.1f} {g.max():6.1f}"
                 if ok.any() else f"{'-':>6s} {'-':>6s} {'-':>6s}")
        print(f"{p.name:18s} {h_ns.size:6d} {inside:8d} {int(ok.sum()):8d} {stats}")
        total += int(h_ns.size)
        matched += int(ok.sum())
        index[p.name] = {
            "teleop_path": str(p),
            "static_idx": m["static_idx"].tolist(),
            "gap_ms": (m["gap_ns"] / 1e6).round(3).tolist(),
        }

    print(f"\nmatched {matched}/{total} teleop frames "
          f"({100 * matched / total:.1f}%) within {args.max_gap_ms:g} ms")
    unused = s_ns.size - len({i for e in index.values()
                              for i in e["static_idx"] if i >= 0})
    print(f"{unused}/{s_ns.size} static frames ({100 * unused / s_ns.size:.0f}%) "
          f"pair with no teleop frame")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "static_path": str(args.static),
            "static_cam": args.cam,
            "max_gap_ms": args.max_gap_ms,
            "episodes": index,
        }, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
