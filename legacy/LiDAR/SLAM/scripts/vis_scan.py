"""Step 0 lidar scan visualization + angle-convention analysis.

Run in the `yixuan_yifan` conda env on the robot Jetson:

    conda run -n yixuan_yifan python SLAM/scripts/vis_scan.py

Saves a polar + cartesian PNG to $SLAM_RESULTS_DIR/debug/ (default
/home/dexmate/yixuan/Dexmate/SLAM) and prints angle statistics.

Empirical calibration procedure (decides bridge params `angle_sign`, `laser_yaw`):
  1. Place a box ~1 m in front-LEFT of the robot.
  2. Run this script. In the cartesian plot (x-forward, y-left), the box must
     appear in the +x/+y quadrant. If it appears at +x/-y the angles are
     mirrored -> angle_sign = -1 in the bridge params.
  3. Face the robot square to a flat wall; the wall should be parallel to the
     y axis. A consistent tilt is the laser_yaw offset.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probe_topics import find_zenoh_config

LIDAR_TOPIC = "dm/vg3cfe65689d-1/sensors/lidar_2d_front/scan"
# All host-side results go to the (sshfs) results tree; the docker daemon
# cannot mount it, so only conda/host stages write here.
RESULTS_DIR = Path(os.getenv("SLAM_RESULTS_DIR",
                             "/home/dexmate/yixuan/Dexmate/SLAM"))
OUT_DIR = RESULTS_DIR / "debug"


def grab_scans(n_scans: int, timeout_s: float) -> list[dict]:
    from dexcomm import Subscriber
    from dexcomm.codecs import LidarScan2DCodec

    scans: list[dict] = []
    sub = Subscriber(LIDAR_TOPIC, decoder=LidarScan2DCodec.decode,
                     config=find_zenoh_config())
    try:
        deadline = time.monotonic() + timeout_s
        last_ts = None
        while len(scans) < n_scans and time.monotonic() < deadline:
            msg = sub.wait_for_message(timeout=max(deadline - time.monotonic(), 0.1))
            if msg is None:
                continue
            ts = msg.get("timestamp", msg.get("timestamp_ns"))
            if ts == last_ts:
                time.sleep(0.02)
                continue
            last_ts = ts
            scans.append(msg)
    finally:
        sub.shutdown()
    if not scans:
        raise RuntimeError(
            f"No lidar scan received on {LIDAR_TOPIC} within {timeout_s}s. "
            "Is `dexsensor launch --sensor lidar_2d` running?")
    return scans


def analyze(scan: dict) -> None:
    for key in ("ranges", "angles"):
        if key not in scan:
            raise ValueError(f"Scan missing '{key}'; got keys {sorted(scan)}")
    ranges = np.asarray(scan["ranges"], dtype=np.float64)
    angles = np.asarray(scan["angles"], dtype=np.float64)
    if ranges.shape != angles.shape or ranges.ndim != 1:
        raise ValueError(f"Shape mismatch: ranges{ranges.shape} angles{angles.shape}")

    n = ranges.size
    print(f"points/scan        : {n}")
    print(f"angle range [rad]  : [{angles.min():.4f}, {angles.max():.4f}] "
          f"(span {np.degrees(angles.max() - angles.min()):.1f} deg)")
    diffs = np.diff(angles)
    if diffs.size:
        frac_inc = float(np.mean(diffs > 0))
        print(f"angle monotonicity : {frac_inc * 100:.1f}% increasing steps")
        print(f"angle increment    : median={np.degrees(np.median(diffs)):.4f} deg, "
              f"std={np.degrees(np.std(diffs)):.4f} deg "
              f"(uniform={'yes' if np.std(diffs) < 1e-4 else 'NO -> binning required'})")
    finite = np.isfinite(ranges)
    valid = finite & (ranges > 0.05)
    print(f"invalid returns    : zeros/sub-5cm={int(np.sum(finite & ~valid))}, "
          f"non-finite={int(np.sum(~finite))}, valid={int(np.sum(valid))}")
    if np.any(valid):
        print(f"range stats [m]    : min={ranges[valid].min():.3f} "
              f"max={ranges[valid].max():.3f} mean={ranges[valid].mean():.3f}")
    if "intensities" in scan:
        inten = np.asarray(scan["intensities"], dtype=np.float64)
        print(f"intensities        : min={inten.min():.0f} max={inten.max():.0f}")
    ts = scan.get("timestamp", scan.get("timestamp_ns"))
    print(f"timestamp field    : {ts} (now={time.time_ns()})")


def plot(scans: list[dict], out_path: Path) -> None:
    fig, (ax_polar, ax_cart) = plt.subplots(
        1, 2, figsize=(14, 7), subplot_kw=None)
    ax_polar.remove()
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")

    cmap = plt.get_cmap("viridis")
    for i, scan in enumerate(scans):
        ranges = np.asarray(scan["ranges"], dtype=np.float64)
        angles = np.asarray(scan["angles"], dtype=np.float64)
        valid = np.isfinite(ranges) & (ranges > 0.05)
        color = cmap(i / max(len(scans) - 1, 1))
        ax_polar.scatter(angles[valid], ranges[valid], s=2, color=color)
        x = ranges[valid] * np.cos(angles[valid])
        y = ranges[valid] * np.sin(angles[valid])
        ax_cart.scatter(x, y, s=2, color=color, label=f"scan {i}")

    ax_polar.set_title("raw (angle, range)")
    ax_cart.set_title("cartesian: x-forward, y-left (REP-103 if box-front-left at +x/+y)")
    ax_cart.set_xlabel("x [m] (forward?)")
    ax_cart.set_ylabel("y [m] (left?)")
    ax_cart.axhline(0, color="gray", lw=0.5)
    ax_cart.axvline(0, color="gray", lw=0.5)
    ax_cart.annotate("", xy=(0.5, 0), xytext=(0, 0),
                     arrowprops=dict(color="red", arrowstyle="->"))
    ax_cart.text(0.52, 0.02, "+x", color="red")
    ax_cart.set_aspect("equal")
    ax_cart.legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"\nsaved plot -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-scans", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--out", type=Path,
                        default=OUT_DIR / "lidar_scan.png")
    args = parser.parse_args()

    scans = grab_scans(args.n_scans, args.timeout)
    print(f"=== analysis of last scan (of {len(scans)}) ===")
    analyze(scans[-1])
    plot(scans, args.out)


if __name__ == "__main__":
    main()
