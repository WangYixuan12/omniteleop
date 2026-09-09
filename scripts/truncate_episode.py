#!/usr/bin/env python3
r"""Cut a teleop episode to its first N frames, and its static-camera take to match.

Typical use: a grasp failed part-way through a take, ``vis_policy_episode.py`` showed
the data is good up to row N-1, and everything after it should be dropped from BOTH
the ``wbc_*_robot.py --record`` episode and the paired ``record_static_cams.py`` file.

The cut is done IN PLACE. Every per-frame dataset the recorders wrote is resizable, so
the teleop file is shrunk with ``Dataset.resize`` (no pixel copy). The static file is the
same except for its fixed-shape per-frame calibration arrays (``camera_<c>_intrinsics``,
``_extrinsics``, ``_rgb_intrinsics``, ``_rgb_to_ir_extrinsic``, ``_baseline``), which are
rewritten truncated. HDF5 does not return the freed space to the filesystem; run
``h5repack`` on the result if the disk matters.

The static cut point follows ``scripts/align_static_teleop.py``: static and teleop rows
are put on the teleop host clock, and every static row up to HALF A STATIC PERIOD after
the last kept teleop row is retained, so the nearest-neighbour match of every kept teleop
row is unchanged by the cut.

Default is a dry run that only reports the cut. Pass ``--apply`` to modify the files.
``--episode N`` loads ``episode_N.hdf5``, or the quarantined ``episode_N.hdf5.partial``
if that is all that exists. A cut does not rename a ``.partial`` to ``.hdf5``.

    python scripts/truncate_episode.py \
        --teleop_dir /data/Dexmate/data \
        --static_dir /data/Dexmate/static_09_05 \
        --episode 3 \
        --n-frames 311 --apply
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent


def _episode_path(directory: Path, episode: int) -> Path:
    """``episode_<N>.hdf5``, or the quarantined ``.hdf5.partial`` if that is all that exists."""
    complete = directory / f"episode_{episode}.hdf5"
    if complete.is_file():
        return complete
    partial = directory / f"episode_{episode}.hdf5.partial"
    if partial.is_file():
        return partial
    raise FileNotFoundError(f"episode_{episode}.hdf5[.partial] not found in {directory}")


def _load_align():
    spec = importlib.util.spec_from_file_location("align_static_teleop", _SCRIPTS / "align_static_teleop.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _per_frame_datasets(f: h5py.File, n_total: int, skip_prefix: str) -> list[h5py.Dataset]:
    """Every dataset whose leading dimension is the frame count (recorder layout)."""
    found: list[h5py.Dataset] = []

    def visit(name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim > 0 and not name.startswith(skip_prefix):
            if obj.shape[0] == n_total:
                found.append(obj)

    f.visititems(visit)
    return found


def _truncate(ds: h5py.Dataset, n: int) -> None:
    """Shrink ``ds`` to ``n`` leading rows: resize when possible, else rewrite."""
    if ds.maxshape[0] is None:
        ds.resize(n, axis=0)
        return
    parent, name = ds.parent, ds.name.rsplit("/", 1)[-1]
    data = ds[:n]
    attrs = dict(ds.attrs)
    del parent[name]
    new = parent.create_dataset(name, data=data)
    for k, v in attrs.items():
        new.attrs[k] = v


def static_keep_count(align, teleop: Path, static: Path, n: int) -> tuple[int, float]:
    """Static rows to keep so every kept teleop row still matches its nearest static row."""
    head_ns = align.head_on_teleop_clock(teleop)
    if head_ns.shape[0] < n:
        raise ValueError(f"teleop has {head_ns.shape[0]} rows, cannot keep {n}")
    with h5py.File(static, "r") as sf:
        rate = float(sf.attrs["record_rate_hz"])
        total = int(sf.attrs["num_frames"])
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"{static}: bad record_rate_hz {rate}")
    cutoff_ns = int(head_ns[n - 1]) + int(round(0.5e9 / rate))
    keep = 0
    for cam in (0, 1):
        s_ns = align.static_on_teleop_clock(static, cam)
        if s_ns.shape[0] != total:
            raise ValueError(f"{static}: camera_{cam} has {s_ns.shape[0]} stamps, attr says {total}")
        if np.any(np.diff(s_ns) < 0):
            raise ValueError(f"{static}: camera_{cam} stamps are not non-decreasing")
        keep = max(keep, int(np.searchsorted(s_ns, cutoff_ns, side="right")))
    if keep == 0:
        raise ValueError("no static frame precedes the last kept teleop row; clocks misaligned?")
    return keep, (int(head_ns[n - 1]) - int(head_ns[0])) / 1e9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--teleop_dir",
        type=Path,
        required=True,
        help="directory of recorded episode_<N>.hdf5[.partial] files",
    )
    ap.add_argument(
        "--static_dir",
        type=Path,
        default=None,
        help="directory of record_static_cams.py episode_<N>.hdf5[.partial] files paired with --teleop_dir",
    )
    ap.add_argument(
        "--episode",
        type=int,
        required=True,
        help="episode index N; loads episode_N.hdf5, or .hdf5.partial if that is all that exists",
    )
    ap.add_argument("--n-frames", type=int, required=True, help="teleop rows to KEEP (0..N-1)")
    ap.add_argument("--apply", action="store_true", help="modify the files (default: dry run)")
    args = ap.parse_args()
    if args.n_frames < 1:
        raise ValueError("--n-frames must be >= 1")
    if args.episode < 0:
        raise ValueError("--episode must be >= 0")

    teleop = _episode_path(args.teleop_dir, args.episode)
    static = None if args.static_dir is None else _episode_path(args.static_dir, args.episode)
    if teleop.name.endswith(".partial"):
        print(f"teleop {teleop} is a quarantined .partial (inspect/cut only; not renamed)")
    if static is not None and static.name.endswith(".partial"):
        print(f"static {static} is a quarantined .partial (inspect/cut only; not renamed)")

    with h5py.File(teleop, "r") as tf:
        n_total = int(tf.attrs["n_frames"])
        if int(tf["timestamp_ns"].shape[0]) != n_total:
            raise ValueError(f"{teleop}: n_frames attr {n_total} != timestamp rows")
        if args.n_frames > n_total:
            raise ValueError(f"--n-frames {args.n_frames} exceeds the {n_total} recorded rows")
        n_teleop_ds = len(_per_frame_datasets(tf, n_total, "meta/"))
    print(f"teleop {teleop}: {n_total} -> {args.n_frames} rows ({n_teleop_ds} per-frame datasets)")

    keep_static = None
    if static is not None:
        align = _load_align()
        keep_static, kept_s = static_keep_count(align, teleop, static, args.n_frames)
        with h5py.File(static, "r") as sf:
            s_total = int(sf.attrs["num_frames"])
            n_static_ds = len(_per_frame_datasets(sf, s_total, "\0"))
        print(f"static {static}: {s_total} -> {keep_static} rows "
              f"({n_static_ds} per-frame datasets; kept teleop span {kept_s:.2f}s)")

    if args.n_frames == n_total and (keep_static is None or keep_static == s_total):
        print("nothing to cut")
        return 0
    if not args.apply:
        print("dry run; re-run with --apply to cut in place")
        return 0

    with h5py.File(teleop, "r+") as tf:
        for ds in _per_frame_datasets(tf, n_total, "meta/"):
            _truncate(ds, args.n_frames)
        tf.attrs["n_frames"] = np.int64(args.n_frames)
    print(f"teleop cut to {args.n_frames} rows")

    if static is not None:
        assert keep_static is not None
        with h5py.File(static, "r+") as sf:
            for ds in _per_frame_datasets(sf, s_total, "\0"):
                _truncate(ds, keep_static)
            sf.attrs["num_frames"] = np.int64(keep_static)
        print(f"static cut to {keep_static} rows")
        head_ns = align.head_on_teleop_clock(teleop)
        for cam in (0, 1):
            m = align.match(align.static_on_teleop_clock(static, cam), head_ns, int(70e6))
            print(f"  verify camera_{cam}: {(m['static_idx'] >= 0).sum()}/{len(head_ns)} teleop rows "
                  f"matched within 70 ms, max gap {m['gap_ns'].max() / 1e6:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
