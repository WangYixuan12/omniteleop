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

``--cuts`` takes several episodes at once as ``EP:KEEP`` pairs. The count travels with its
own episode, so a transposed pair is visible at the call site rather than silently cutting
the wrong take to the wrong length. Every episode is resolved and checked BEFORE anything
is written, so a bad pair aborts the batch instead of leaving it half applied.

    python scripts/truncate_episode.py \
        --teleop_dir /data/Dexmate/data_mobile/basket \
        --static_dir /data/Dexmate/static_mobile/basket \
        --cuts 2:495 5:3347 --apply
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


def _parse_cuts(args) -> list[tuple[int, int]]:
    """The requested cuts as ``(episode, n_frames)``, from either argument form."""
    if args.cuts is None:
        if args.n_frames is None:
            raise ValueError("--episode requires --n-frames")
        return [(args.episode, args.n_frames)]
    if args.n_frames is not None:
        raise ValueError("--n-frames belongs to the --episode form; --cuts carries its own counts")
    pairs: list[tuple[int, int]] = []
    for item in args.cuts:
        episode, sep, keep = item.partition(":")
        if not sep:
            raise ValueError(f"--cuts takes EP:KEEP pairs, got {item!r}")
        try:
            pairs.append((int(episode), int(keep)))
        except ValueError:
            raise ValueError(f"--cuts takes EP:KEEP integers, got {item!r}") from None
    episodes = [e for e, _ in pairs]
    duplicated = sorted({e for e in episodes if episodes.count(e) > 1})
    if duplicated:
        raise ValueError(f"--cuts names an episode more than once: {duplicated}")
    return pairs


def plan_cut(align, teleop_dir: Path, static_dir: Path | None, episode: int, n_frames: int) -> dict:
    """Resolve one episode and work out its cut. Read-only: nothing is modified here."""
    if n_frames < 1:
        raise ValueError(f"episode {episode}: rows to keep must be >= 1, got {n_frames}")
    if episode < 0:
        raise ValueError(f"episode index must be >= 0, got {episode}")

    teleop = _episode_path(teleop_dir, episode)
    static = None if static_dir is None else _episode_path(static_dir, episode)

    with h5py.File(teleop, "r") as tf:
        n_total = int(tf.attrs["n_frames"])
        if int(tf["timestamp_ns"].shape[0]) != n_total:
            raise ValueError(f"{teleop}: n_frames attr {n_total} != timestamp rows")
        if n_frames > n_total:
            raise ValueError(f"{teleop}: asked to keep {n_frames} of {n_total} recorded rows")
        n_teleop_ds = len(_per_frame_datasets(tf, n_total, "meta/"))

    keep_static = s_total = n_static_ds = None
    kept_s = 0.0
    if static is not None:
        keep_static, kept_s = static_keep_count(align, teleop, static, n_frames)
        with h5py.File(static, "r") as sf:
            s_total = int(sf.attrs["num_frames"])
            n_static_ds = len(_per_frame_datasets(sf, s_total, "\0"))

    return dict(episode=episode, teleop=teleop, static=static, n_total=n_total,
                n_frames=n_frames, n_teleop_ds=n_teleop_ds, keep_static=keep_static,
                s_total=s_total, n_static_ds=n_static_ds, kept_s=kept_s)


def report(p: dict, prefix: str) -> bool:
    """Print what the cut would do; True when there is anything to cut."""
    if p["teleop"].name.endswith(".partial"):
        print(f"{prefix}teleop {p['teleop']} is a quarantined .partial (inspect/cut only; not renamed)")
    if p["static"] is not None and p["static"].name.endswith(".partial"):
        print(f"{prefix}static {p['static']} is a quarantined .partial (inspect/cut only; not renamed)")
    print(f"{prefix}teleop {p['teleop']}: {p['n_total']} -> {p['n_frames']} rows "
          f"({p['n_teleop_ds']} per-frame datasets)")
    if p["static"] is not None:
        print(f"{prefix}static {p['static']}: {p['s_total']} -> {p['keep_static']} rows "
              f"({p['n_static_ds']} per-frame datasets; kept teleop span {p['kept_s']:.2f}s)")
    return p["n_frames"] != p["n_total"] or (
        p["static"] is not None and p["keep_static"] != p["s_total"])


def apply_cut(align, p: dict, prefix: str) -> None:
    """Shrink the files in place."""
    with h5py.File(p["teleop"], "r+") as tf:
        for ds in _per_frame_datasets(tf, p["n_total"], "meta/"):
            _truncate(ds, p["n_frames"])
        tf.attrs["n_frames"] = np.int64(p["n_frames"])
    print(f"{prefix}teleop cut to {p['n_frames']} rows")

    if p["static"] is None:
        return
    with h5py.File(p["static"], "r+") as sf:
        for ds in _per_frame_datasets(sf, p["s_total"], "\0"):
            _truncate(ds, p["keep_static"])
        sf.attrs["num_frames"] = np.int64(p["keep_static"])
    print(f"{prefix}static cut to {p['keep_static']} rows")
    head_ns = align.head_on_teleop_clock(p["teleop"])
    for cam in (0, 1):
        m = align.match(align.static_on_teleop_clock(p["static"], cam), head_ns, int(70e6))
        print(f"{prefix}  verify camera_{cam}: {(m['static_idx'] >= 0).sum()}/{len(head_ns)} teleop rows "
              f"matched within 70 ms, max gap {m['gap_ns'].max() / 1e6:.0f} ms")


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
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument(
        "--episode",
        type=int,
        help="episode index N; loads episode_N.hdf5, or .hdf5.partial if that is all that exists",
    )
    which.add_argument(
        "--cuts",
        nargs="+",
        metavar="EP:KEEP",
        help="several episodes at once, each as episode:rows-to-keep (e.g. 2:495 5:3347)",
    )
    ap.add_argument("--n-frames", type=int, help="teleop rows to KEEP (0..N-1); use with --episode")
    ap.add_argument("--apply", action="store_true", help="modify the files (default: dry run)")
    args = ap.parse_args()

    cuts = _parse_cuts(args)
    align = None if args.static_dir is None else _load_align()

    # Resolve and check every episode before writing anything, so one bad pair aborts the
    # batch rather than leaving some episodes cut and the rest untouched.
    plans = [plan_cut(align, args.teleop_dir, args.static_dir, ep, n) for ep, n in cuts]

    multi = len(plans) > 1
    pending = []
    for p in plans:
        prefix = f"episode_{p['episode']}: " if multi else ""
        if report(p, prefix):
            pending.append(p)
        else:
            print(f"{prefix}nothing to cut")
    if not pending:
        return 0
    if not args.apply:
        print(f"dry run; re-run with --apply to cut {len(pending)} episode(s) in place")
        return 0

    for p in pending:
        apply_cut(align, p, f"episode_{p['episode']}: " if multi else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
