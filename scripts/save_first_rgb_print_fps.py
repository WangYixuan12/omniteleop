"""Save the first left_rgb frame of every HDF5 episode as a PNG.

Usage::

    python scripts/save_first_rgb.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/home/yixuan/omniteleop/Dexmate/data/raw_data"),
        help="Directory containing episode HDF5 files",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/home/yixuan/omniteleop/Dexmate/debug/first_rgb"),
        help="Output directory for first-frame PNGs",
    )
    args = parser.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    hdf5_files = sorted(args.src.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in {args.src}")

    skipped: list[tuple[str, str]] = []
    for path in hdf5_files:
        try:
            data, _ = load_dict_from_hdf5(str(path))
        except OSError as e:
            print(f"[SKIP] {path.name}: {e}")
            skipped.append((path.name, str(e)))
            continue
        left_rgb = np.array(data["obs"]["images"]["left_rgb"])
        if left_rgb.ndim != 4 or left_rgb.shape[0] == 0:
            raise ValueError(
                f"{path.name}: expected left_rgb of shape (N,H,W,3), got {left_rgb.shape}"
            )
        first = left_rgb[0].astype(np.uint8)
        out_path = args.dst / f"{path.stem}.png"
        Image.fromarray(first).save(out_path)

        ts_ns = np.asarray(data["timestamp_ns"]).astype(np.int64)
        n = ts_ns.shape[0]
        duration_s = (ts_ns[-1] - ts_ns[0]) / 1e9 if n >= 2 else 0.0
        fps = (n - 1) / duration_s if duration_s > 0 else float("nan")
        print(f"{path.name}: fps = {fps:.2f}  ({n} frames, {duration_s:.2f}s)")

    if skipped:
        print(f"\nSkipped {len(skipped)} unreadable file(s):")
        for name, err in skipped:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
