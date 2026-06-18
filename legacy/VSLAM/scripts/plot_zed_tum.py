#!/usr/bin/env python3
"""Plot a ZED positional-tracking trajectory (TUM format) for a headless box.

This visualizes the *camera path* written by ``zed_area_map_headless.py`` — the
only spatial artifact you can inspect before relocalization. A ZED ``.area`` file
holds sparse relocalization features/keyframes, not dense geometry, so it cannot
be rendered as a map; the trajectory is what tells you whether the run actually
covered the space (loop closures, drift, gaps).

The headless sample writes TUM in ``COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP``:
columns ``ts x y z qx qy qz qw`` with **+Y up**, so the ground plane is X–Z.

Example:
  python scripts/plot_zed_tum.py \
      ~/yixuan/Dexmate/SLAM/zed/logs/lab_v1_online_.tum \
      --output ~/yixuan/Dexmate/SLAM/zed/logs/lab_v1_online_traj.png
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")  # headless Jetson / SSH: no display, write a PNG
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_tum(path: pathlib.Path) -> np.ndarray:
    """Load a TUM file as an (N, 8) float array, validating the column count."""
    rows = []
    for ln, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            raise ValueError(
                f"{path}:{ln}: expected 8 TUM columns (ts x y z qx qy qz qw), "
                f"got {len(parts)}: {line!r}"
            )
        rows.append([float(p) for p in parts])
    if not rows:
        raise ValueError(f"{path}: no pose rows found")
    return np.asarray(rows, dtype=np.float64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tum_file", type=pathlib.Path)
    p.add_argument("--output", type=pathlib.Path, help="PNG path; default: <tum>.png")
    p.add_argument(
        "--up",
        choices=["y", "z"],
        default="y",
        help="Up axis of the TUM data. The headless sample uses RIGHT_HANDED_Y_UP "
        "=> 'y'. Pass 'z' only if you changed the coordinate system.",
    )
    args = p.parse_args()

    data = load_tum(args.tum_file)
    t = data[:, 0] - data[0, 0]
    x, y, z = data[:, 1], data[:, 2], data[:, 3]

    # Ground plane = the two non-up axes; height = the up axis.
    if args.up == "y":
        gx, gy, gx_lbl, gy_lbl, height, h_lbl = x, z, "x (m)", "z (m)", y, "y (up, m)"
    else:
        gx, gy, gx_lbl, gy_lbl, height, h_lbl = x, y, "x (m)", "y (m)", z, "z (up, m)"

    seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    path_len = float(seg.sum())
    end_gap = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2 + (z[-1] - z[0]) ** 2))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.5))
    sc = ax0.scatter(gx, gy, c=t, cmap="viridis", s=4)
    ax0.plot(gx, gy, "-", lw=0.4, color="gray", alpha=0.5)
    ax0.scatter([gx[0]], [gy[0]], c="lime", s=90, edgecolor="k", zorder=5, label="start")
    ax0.scatter([gx[-1]], [gy[-1]], c="red", s=90, edgecolor="k", zorder=5, label="end")
    ax0.set_aspect("equal", adjustable="datalim")
    ax0.set_xlabel(gx_lbl)
    ax0.set_ylabel(gy_lbl)
    ax0.set_title(
        f"top-down path  |  {len(t)} poses, {t[-1]:.0f}s\n"
        f"path length {path_len:.1f} m, start->end gap {end_gap*100:.0f} cm"
    )
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax0, label="time (s)")

    ax1.plot(t, height, lw=0.8)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel(h_lbl)
    ax1.set_title(f"height over time  (range {height.max()-height.min():.2f} m)")
    ax1.grid(True, alpha=0.3)

    fig.suptitle(str(args.tum_file.name))
    fig.tight_layout()
    out = args.output or args.tum_file.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"[plot] wrote {out}")
    print(
        f"[plot] {len(t)} poses, {t[-1]:.1f}s, path {path_len:.2f} m, "
        f"start->end gap {end_gap*100:.1f} cm"
    )
    print(
        f"[plot] extent  {gx_lbl[:1]}:[{gx.min():+.2f},{gx.max():+.2f}]  "
        f"{gy_lbl[:1]}:[{gy.min():+.2f},{gy.max():+.2f}]  "
        f"{h_lbl[:1]}(up):[{height.min():+.2f},{height.max():+.2f}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
