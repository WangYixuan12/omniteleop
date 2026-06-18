"""Export the live SLAM state as a point cloud (.ply + top-down preview PNG).

slam_toolbox is a 2D SLAM: it has NO native point-cloud output (only the /map
OccupancyGrid, the map->base_link TF and the serialized posegraph). What we
CAN derive from it is a planar cloud, from two sources:

  - "map"  points: centers of occupied /map cells (z = 0, floor-projected)
  - "scan" points: live /scan returns projected into the map frame through
    the SLAM TF and accumulated over --secs (z = lidar mount height; drive
    the robot during the window to densify coverage)

A true 3D cloud is impossible from this stack — the RPLidar senses a single
horizontal plane (see memory: vega-1 has no 3D lidar); 3D would have to come
from the head ZED depth, which is a separate pipeline.

Two stages because rclpy lives in the container and the results tree lives on
sshfs that docker cannot mount:

  # 1. accumulate scans + grab the map (writes /slam/debug/pointcloud_dump.npz)
  docker exec yixuan-slam-mapping bash -c \
      'source /opt/ros/humble/setup.bash && \
       python3 /slam/scripts/export_pointcloud.py --dump --secs 10'

  # 2. convert to $SLAM_RESULTS_DIR/pointcloud/<name>.ply (+ preview PNG)
  conda run -n yixuan_yifan python \
      /home/dexmate/yixuan_slam_runtime/scripts/export_pointcloud.py \
      --convert --name lab_v1
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

# Dump stage runs in the container: __file__ is /slam/scripts/... so this is
# /slam/debug (the docker-mountable local runtime dir).
DUMP_DEFAULT = Path(__file__).resolve().parent.parent / "debug" / "pointcloud_dump.npz"
# Convert stage runs on the host and must find the same file through the
# runtime dir, regardless of whether the repo or the runtime copy is invoked.
RUNTIME_DIR = Path(os.getenv("SLAM_RUNTIME_DIR", "/home/dexmate/yixuan_slam_runtime"))
RESULTS_DIR = Path(os.getenv("SLAM_RESULTS_DIR", "/home/dexmate/yixuan/Dexmate/SLAM"))

VOXEL_M = 0.01  # dedup quantization for accumulated scan points

MAP_RGB = (60, 60, 60)
SCAN_RGB = (220, 40, 40)


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    import math
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def dump(secs: float) -> None:
    import rclpy
    from nav_msgs.msg import OccupancyGrid
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = Node("pointcloud_dump")
    got: dict[str, object] = {}

    map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(OccupancyGrid, "/map",
                             lambda m: got.__setitem__("map", m), map_qos)
    node.create_subscription(LaserScan, "/scan",
                             lambda m: got.__setitem__("scan", m), 5)
    buf = Buffer()
    TransformListener(buf, node)

    chunks: list[np.ndarray] = []
    n_scans = 0
    laser_z = None
    last_stamp = None
    t_end = time.monotonic() + secs
    while rclpy.ok() and time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.2)
        s = got.get("scan")
        if s is None:
            continue
        stamp = (s.header.stamp.sec, s.header.stamp.nanosec)
        if stamp == last_stamp:
            continue
        try:
            tf = buf.lookup_transform("map", s.header.frame_id, rclpy.time.Time())
        except Exception:
            continue  # TF not up yet; keep spinning
        last_stamp = stamp
        t = tf.transform.translation
        q = tf.transform.rotation
        lyaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        laser_z = t.z

        r = np.asarray(s.ranges, dtype=np.float32)
        th = s.angle_min + np.arange(r.size, dtype=np.float32) * s.angle_increment
        if r.shape != th.shape:
            raise ValueError(f"ranges{r.shape} vs angles{th.shape} mismatch")
        valid = np.isfinite(r) & (r > max(s.range_min, 0.05)) & (r < s.range_max - 1e-3)
        if not np.any(valid):
            continue
        a = th[valid] + lyaw
        pts = np.stack([t.x + r[valid] * np.cos(a),
                        t.y + r[valid] * np.sin(a)], axis=1)
        chunks.append(pts.astype(np.float32))
        n_scans += 1

    if n_scans == 0:
        raise RuntimeError(
            f"no scan projected within {secs}s — are /scan and the SLAM TF "
            "alive? (docker logs the bridge / slam_toolbox)")
    if "map" not in got:
        raise RuntimeError("/map never arrived — is slam_toolbox running?")

    pts = np.concatenate(chunks, axis=0)
    # dedup: quantize to VOXEL_M so a stationary robot doesn't blow up the file
    keys = np.round(pts / VOXEL_M).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    pts = pts[np.sort(idx)]

    m = got["map"]
    oq = m.info.origin.orientation
    if abs(quat_to_yaw(oq.x, oq.y, oq.z, oq.w)) > 1e-6:
        raise ValueError("map origin has nonzero yaw; converter assumes 0")

    DUMP_DEFAULT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        DUMP_DEFAULT,
        scan_points_xy=pts,
        laser_z=float(laser_z),
        n_scans=n_scans,
        span_secs=secs,
        grid=np.asarray(m.data, dtype=np.int8).reshape(m.info.height, m.info.width),
        resolution=m.info.resolution,
        origin=[m.info.origin.position.x, m.info.origin.position.y],
        stamp_ns=time.time_ns(),
    )
    print(f"wrote {DUMP_DEFAULT}: {pts.shape[0]} scan pts from {n_scans} scans, "
          f"map {m.info.width}x{m.info.height} @ {m.info.resolution} m")
    rclpy.try_shutdown()


def occupied_cell_centers(grid: np.ndarray, res: float,
                          origin: np.ndarray) -> np.ndarray:
    if grid.ndim != 2:
        raise ValueError(f"grid must be 2D, got {grid.shape}")
    iy, ix = np.nonzero(grid > 50)
    return np.stack([origin[0] + (ix + 0.5) * res,
                     origin[1] + (iy + 0.5) * res], axis=1).astype(np.float32)


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    if xyz.shape[0] != rgb.shape[0] or xyz.shape[1] != 3 or rgb.shape[1] != 3:
        raise ValueError(f"xyz{xyz.shape} vs rgb{rgb.shape} mismatch")
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n")
    with open(path, "w") as f:
        f.write(header)
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")


def convert(name: str, dump_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not dump_path.exists():
        raise FileNotFoundError(f"{dump_path} missing — run the --dump stage first")
    d = np.load(dump_path)
    scan_xy = d["scan_points_xy"]
    laser_z = float(d["laser_z"])
    map_xy = occupied_cell_centers(d["grid"], float(d["resolution"]), d["origin"])
    if map_xy.shape[0] == 0:
        print("WARN: map has no occupied cells yet (early mapping?)")

    map_xyz = np.concatenate([map_xy, np.zeros((map_xy.shape[0], 1),
                                               dtype=np.float32)], axis=1)
    scan_xyz = np.concatenate([scan_xy, np.full((scan_xy.shape[0], 1), laser_z,
                                                dtype=np.float32)], axis=1)
    xyz = np.concatenate([map_xyz, scan_xyz], axis=0)
    rgb = np.concatenate([
        np.tile(np.array(MAP_RGB, dtype=np.uint8), (map_xyz.shape[0], 1)),
        np.tile(np.array(SCAN_RGB, dtype=np.uint8), (scan_xyz.shape[0], 1)),
    ], axis=0)

    out_dir = RESULTS_DIR / "pointcloud"
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / f"{name}.ply"
    write_ply(ply_path, xyz, rgb)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(map_xy[:, 0], map_xy[:, 1], s=2, c="dimgray",
               label=f"map cells ({map_xy.shape[0]})")
    ax.scatter(scan_xy[:, 0], scan_xy[:, 1], s=2, c="red",
               label=f"accumulated scans ({scan_xy.shape[0]}, "
                     f"{int(d['n_scans'])} scans/{float(d['span_secs']):.0f}s)")
    ax.set_title(f"SLAM point cloud '{name}' (planar: scans at z={laser_z:.3f} m)")
    ax.set_xlabel("map x [m]")
    ax.set_ylabel("map y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", markerscale=4)
    fig.tight_layout()
    png_path = out_dir / f"{name}_preview.png"
    fig.savefig(png_path, dpi=130)

    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    print(f"saved {ply_path} ({xyz.shape[0]} pts) and {png_path}")
    print(f"bounds [m]: x[{lo[0]:+.2f},{hi[0]:+.2f}] y[{lo[1]:+.2f},{hi[1]:+.2f}] "
          f"z[{lo[2]:+.2f},{hi[2]:+.2f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dump", action="store_true")
    mode.add_argument("--convert", action="store_true")
    parser.add_argument("--secs", type=float, default=10.0,
                        help="scan accumulation window for --dump")
    parser.add_argument("--name", default="slam_cloud",
                        help="output basename for --convert")
    parser.add_argument("--dump-file", type=Path,
                        default=RUNTIME_DIR / "debug" / "pointcloud_dump.npz",
                        help="npz produced by --dump (host-side path)")
    args = parser.parse_args()
    if args.dump:
        dump(args.secs)
    else:
        convert(args.name, args.dump_file)


if __name__ == "__main__":
    main()
