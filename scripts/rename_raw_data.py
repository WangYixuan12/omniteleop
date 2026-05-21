"""Split episode_*.hdf5 into train/val/test directories with consecutive renaming.

Source layout:
    /home/yixuan/omniteleop/Dexmate/data/raw_data/episode_{i}.hdf5

Destination layout:
    /home/yixuan/omniteleop/Dexmate/data/raw_data_renamed/{split}/episode_{j}.hdf5

Splits (source indices -> renamed consecutively starting at 0):
    train: 0..50            (those that exist; gaps are skipped)
    val:   52, 54, 57, 59, 61
    test:  53, 55, 58, 60, 62

By default files are COPIED so raw_data stays intact and HDF5 contents are not modified.
Use --mode hardlink for an instant zero-copy alternative on the same filesystem,
or --mode move to relocate the originals.

Run with --apply to perform the operation; otherwise prints a dry-run plan.
At the end, verifies that source-file count == total destination-file count.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

PATTERN = re.compile(r"^episode_(\d+)\.hdf5$")

DEFAULT_SRC = Path("/home/yixuan/omniteleop/Dexmate/data/raw_data")
DEFAULT_DST = Path("/home/yixuan/omniteleop/Dexmate/data/raw_data_renamed")

TRAIN_RANGE = [7] # list(range(12,81))
VAL_INDICES = None
TEST_INDICES = None


def collect_episodes(directory: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in directory.iterdir():
        if not p.is_file():
            continue
        m = PATTERN.match(p.name)
        if not m:
            continue
        out[int(m.group(1))] = p
    return out


def build_split_plan(source_files: dict[int, Path], split_name: str,
                     src_indices: list[int] | None, require_all: bool,
                     dst_dir: Path) -> list[tuple[Path, Path, int]]:
    """Return list of (src_path, dst_path, old_idx) sorted by source index."""
    if src_indices is None:
        print(f"  [{split_name}] no source indices configured (skipped)")
        return []

    available = sorted(i for i in src_indices if i in source_files)
    missing = [i for i in src_indices if i not in source_files]
    if missing:
        msg = f"missing source indices for {split_name}: {missing}"
        if require_all:
            raise SystemExit(f"ERROR: {msg}")
        print(f"  [{split_name}] {msg} (skipped)")
    plan = []
    for new_idx, old_idx in enumerate(available):
        plan.append((source_files[old_idx], dst_dir / f"episode_{new_idx}.hdf5", old_idx))
    return plan


def transfer(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        dst.hardlink_to(src)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"unknown mode: {mode}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help=f"source directory (default: {DEFAULT_SRC})")
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST,
                    help=f"destination root directory (default: {DEFAULT_DST})")
    ap.add_argument("--train", action="store_true", help="produce the train split")
    ap.add_argument("--val", action="store_true", help="produce the val split")
    ap.add_argument("--test", action="store_true", help="produce the test split")
    ap.add_argument("--mode", choices=("copy", "hardlink", "move"), default="copy",
                    help="how to materialize files in dst (default: copy)")
    ap.add_argument("--apply", action="store_true",
                    help="actually perform operations; otherwise dry-run")
    args = ap.parse_args()

    if not args.src.is_dir():
        print(f"ERROR: source directory not found: {args.src}", file=sys.stderr)
        sys.exit(1)

    # If no split flag is given, do all three.
    if not (args.train or args.val or args.test):
        args.train = args.val = args.test = True

    source_files = collect_episodes(args.src)
    print(f"Source: {args.src}")
    print(f"Found {len(source_files)} episode files in source.")

    splits: list[tuple[str, list[int] | None, bool]] = []
    if args.train:
        splits.append(("train", TRAIN_RANGE, False))   # gaps allowed
    if args.val:
        splits.append(("val", VAL_INDICES, True))      # all must exist
    if args.test:
        splits.append(("test", TEST_INDICES, True))    # all must exist

    overall_plan: dict[str, list[tuple[Path, Path, int]]] = {}
    used_src_indices: set[int] = set()
    for name, indices, require_all in splits:
        print(f"\n[{name}]")
        split_dir = args.dst / name
        plan = build_split_plan(source_files, name, indices, require_all, split_dir)
        overall_plan[name] = plan
        for src, dst, old_idx in plan:
            print(f"  episode_{old_idx}.hdf5  ->  {name}/{dst.name}")
            if old_idx in used_src_indices:
                raise SystemExit(f"ERROR: source index {old_idx} assigned to multiple splits")
            used_src_indices.add(old_idx)
        print(f"  ({len(plan)} files)")

    total_planned = sum(len(p) for p in overall_plan.values())
    print(f"\nTotal files planned: {total_planned}")
    print(f"Source files matched by any split: {len(used_src_indices)}")
    unused = sorted(set(source_files.keys()) - used_src_indices)
    if unused:
        print(f"Source episodes NOT used by any split: {unused}")

    if not args.apply:
        print("\nDry run. Re-run with --apply to perform the operation.")
        return

    # Pre-flight: refuse to overwrite anything.
    for name, plan in overall_plan.items():
        for _src, dst, _ in plan:
            if dst.exists():
                raise SystemExit(f"ERROR: destination already exists: {dst}")

    for name, plan in overall_plan.items():
        split_dir = args.dst / name
        split_dir.mkdir(parents=True, exist_ok=True)
        for src, dst, _ in plan:
            transfer(src, dst, args.mode)
        print(f"[{name}] wrote {len(plan)} files to {split_dir}")

    # Verification.
    print("\nVerification:")
    ok = True
    total_actual = 0
    for name, plan in overall_plan.items():
        split_dir = args.dst / name
        actual_files = sorted(p for p in split_dir.iterdir() if PATTERN.match(p.name))
        actual = len(actual_files)
        expected = len(plan)
        total_actual += actual
        # Also confirm the renamed indices form 0..N-1.
        idx_set = {int(PATTERN.match(p.name).group(1)) for p in actual_files}
        contiguous = idx_set == set(range(expected))
        status = "OK" if (actual == expected and contiguous) else "MISMATCH"
        if status != "OK":
            ok = False
        print(f"  {name}: expected {expected}, found {actual}, "
              f"contiguous=[0..{expected - 1}]={contiguous} -> {status}")
    print(f"  total destination files: {total_actual}; total planned: {total_planned}")
    if total_actual != total_planned:
        ok = False

    if not ok:
        sys.exit(1)
    print("All splits verified.")


if __name__ == "__main__":
    main()
