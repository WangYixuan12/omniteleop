"""Headless snapshot of the live SLAM state (map + scan + robot pose) as PNG.

No robot motion and no GUI needed. Two stages because matplotlib lives in the
conda env while /map (ROS) lives in the container:

  # 1. dump current /map, /scan and TF poses to /slam/debug/map_snapshot.npz
  docker exec yixuan-slam-mapping bash -c \
      'source /opt/ros/humble/setup.bash && python3 /slam/scripts/vis_map_snapshot.py --dump'

  # 2. render the PNG into $SLAM_RESULTS_DIR/debug (conda env, host; also
  #    copies the npz next to it)
  /home/dexmate/miniconda3/envs/yixuan_yifan/bin/python \
      /home/dexmate/yixuan_slam_runtime/scripts/vis_map_snapshot.py --render

Works identically with the localization container (SLAM_CONTAINER name differs).
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
from pathlib import Path

import numpy as np

# Written by the dump stage inside the container (= /slam/debug, the local
# runtime dir docker can mount); read back by the host render stage.
SNAPSHOT = Path(__file__).resolve().parent.parent / "debug" / "map_snapshot.npz"
RESULTS_DIR = Path(os.getenv("SLAM_RESULTS_DIR",
                             "/home/dexmate/yixuan/Dexmate/SLAM"))


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def dump() -> None:
    import rclpy
    from nav_msgs.msg import OccupancyGrid
    from rclpy.node import Node
    from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy)
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = Node("map_snapshot")
    got: dict[str, object] = {}

    map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(OccupancyGrid, "/map",
                             lambda m: got.__setitem__("map", m), map_qos)
    node.create_subscription(LaserScan, "/scan",
                             lambda m: got.__setitem__("scan", m), 5)
    buf = Buffer()
    TransformListener(buf, node)

    deadline = node.get_clock().now().nanoseconds / 1e9 + 20.0
    base_tf = laser_tf = None
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.2)
        if "map" in got and "scan" in got:
            try:
                base_tf = buf.lookup_transform("map", "base_link", rclpy.time.Time())
                laser_tf = buf.lookup_transform("map", "laser", rclpy.time.Time())
                break
            except Exception:
                pass
        if node.get_clock().now().nanoseconds / 1e9 > deadline:
            raise RuntimeError(
                f"timed out: have {sorted(got)}, TF map->base_link "
                f"{'ok' if base_tf else 'missing'} — is slam_toolbox processing scans?")

    m = got["map"]
    s = got["scan"]
    oq = m.info.origin.orientation
    origin_yaw = quat_to_yaw(oq.x, oq.y, oq.z, oq.w)
    if abs(origin_yaw) > 1e-6:
        raise ValueError(f"map origin has yaw {origin_yaw}; renderer assumes 0")

    def tf_to_xyyaw(tf) -> list[float]:
        t, q = tf.transform.translation, tf.transform.rotation
        return [t.x, t.y, quat_to_yaw(q.x, q.y, q.z, q.w)]

    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        SNAPSHOT,
        grid=np.asarray(m.data, dtype=np.int8).reshape(m.info.height, m.info.width),
        resolution=m.info.resolution,
        origin=[m.info.origin.position.x, m.info.origin.position.y],
        scan_ranges=np.asarray(s.ranges, dtype=np.float32),
        scan_angle_min=s.angle_min,
        scan_angle_inc=s.angle_increment,
        base_pose=tf_to_xyyaw(base_tf),
        laser_pose=tf_to_xyyaw(laser_tf),
        stamp_ns=int(s.header.stamp.sec) * 1_000_000_000 + int(s.header.stamp.nanosec),
    )
    print(f"wrote {SNAPSHOT} (map {m.info.width}x{m.info.height} @ {m.info.resolution} m)")
    rclpy.try_shutdown()


def render(out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not SNAPSHOT.exists():
        raise FileNotFoundError(f"{SNAPSHOT} missing — run the --dump stage first")
    d = np.load(SNAPSHOT)
    grid = d["grid"]
    res = float(d["resolution"])
    ox, oy = d["origin"]
    h, w = grid.shape

    # occupancy: -1 unknown, 0 free, 100 occupied
    img = np.full(grid.shape, 0.6)          # unknown -> gray
    img[grid == 0] = 1.0                     # free -> white
    img[grid > 50] = 0.0                     # occupied -> black

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower",
              extent=[ox, ox + w * res, oy, oy + h * res])

    lx, ly, lyaw = d["laser_pose"]
    r = d["scan_ranges"]
    th = d["scan_angle_min"] + np.arange(r.size) * d["scan_angle_inc"] + lyaw
    fin = np.isfinite(r)
    ax.scatter(lx + r[fin] * np.cos(th[fin]), ly + r[fin] * np.sin(th[fin]),
               s=2, c="red", label="live scan")

    bx, by, byaw = d["base_pose"]
    ax.plot(bx, by, "bo", ms=8)
    ax.annotate("", xy=(bx + 0.4 * math.cos(byaw), by + 0.4 * math.sin(byaw)),
                xytext=(bx, by), arrowprops=dict(color="blue", arrowstyle="-|>"))
    ax.set_title(f"SLAM map + live scan  (robot at x={bx:+.2f} y={by:+.2f} "
                 f"yaw={math.degrees(byaw):+.1f}°)")
    ax.set_xlabel("map x [m]")
    ax.set_ylabel("map y [m]")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")
    npz_copy = out_path.parent / SNAPSHOT.name
    if npz_copy.resolve() != SNAPSHOT.resolve():
        shutil.copy2(SNAPSHOT, npz_copy)
        print(f"copied {npz_copy}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dump", action="store_true")
    mode.add_argument("--render", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=RESULTS_DIR / "debug" / "map_snapshot.png")
    args = parser.parse_args()
    if args.dump:
        dump()
    else:
        render(args.out)


if __name__ == "__main__":
    main()
