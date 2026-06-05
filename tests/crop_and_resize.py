"""Crop and resize the first frame of the head-left and left-wrist RGB streams.

Reads the first frame of ``obs/images/head_left_rgb`` and
``obs/images/left_wrist_rgb`` from an episode HDF5, crops each with the
per-camera ``top, bottom, left, right`` bounds below, resizes the crop to
``(--resize_h, --resize_w)``, and writes the crops and resized crops as PNGs
under ``--output_dir/crop/`` and ``--output_dir/resize/``.

Usage::

    python tests/crop_and_resize.py \
        --hdf5 /home/yixuan/Dexmate/data/raw_data/episode_0.hdf5 \
        --output_dir /home/yixuan/Dexmate/debug/crop_and_resize \
        --resize_h 120 --resize_w 160
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

# ── per-camera crop bounds (top, bottom, left, right), in pixels ─────────────
# Edit these to match the desired region for each stream.
CROP_BOUNDS: dict[str, tuple[int, int, int, int]] = {
    "head_left_rgb": (300, 600, 375, 775),
    "left_wrist_rgb": (0, 720, 0, 1280),
}

_KEYS = {
    "head_left_rgb": "obs/images/head_left_rgb",
    "left_wrist_rgb": "obs/images/left_wrist_rgb",
}


def crop_image(img: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = bounds
    h, w = img.shape[:2]
    if not (0 <= top < bottom <= h):
        raise ValueError(f"top/bottom {top},{bottom} out of range for height {h}")
    if not (0 <= left < right <= w):
        raise ValueError(f"left/right {left},{right} out of range for width {w}")
    return img[top:bottom, left:right]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", default="/home/yixuan/Dexmate/data/raw_data/episode_0.hdf5")
    parser.add_argument("--output_dir", default="/home/yixuan/Dexmate/debug/crop_and_resize")
    parser.add_argument("--resize_h", type=int, default=120)
    parser.add_argument("--resize_w", type=int, default=160)
    args = parser.parse_args()

    crop_dir = Path(args.output_dir) / "crop"
    resize_dir = Path(args.output_dir) / "resize"
    crop_dir.mkdir(parents=True, exist_ok=True)
    resize_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.hdf5, "r") as f:
        for name, key in _KEYS.items():
            if key not in f:
                raise ValueError(f"{key} not found in {args.hdf5}")
            frame = np.asarray(f[key][0])  # first frame, RGB uint8
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"{key} first frame has shape {frame.shape}, expected (H, W, 3)")

            cropped = crop_image(frame, CROP_BOUNDS[name])
            resized = cv2.resize(
                cropped, (args.resize_w, args.resize_h), interpolation=cv2.INTER_AREA
            )

            # cv2 writes BGR; convert from RGB so colors are correct on disk.
            cv2.imwrite(str(crop_dir / f"{name}.png"), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(resize_dir / f"{name}.png"), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
            print(f"{name}: {frame.shape[:2]} -> crop {cropped.shape[:2]} -> resize {resized.shape[:2]}")

    print(f"saved crops to {crop_dir}")
    print(f"saved resized to {resize_dir}")


if __name__ == "__main__":
    main()
