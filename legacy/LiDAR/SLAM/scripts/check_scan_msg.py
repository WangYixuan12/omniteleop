"""Print stats of one /scan message (run inside the SLAM container)."""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


def main() -> None:
    rclpy.init()
    node = Node("scan_check")
    done = rclpy.task.Future()

    def cb(m: LaserScan) -> None:
        r = np.array(m.ranges)
        fin = np.isfinite(r)
        print(f"bins={len(r)} finite={int(fin.sum())} "
              f"rmin={r[fin].min():.3f} rmax={r[fin].max():.3f} "
              f"angle_min={m.angle_min:.4f} inc={m.angle_increment:.6f} "
              f"frame={m.header.frame_id} stamp={m.header.stamp.sec}.{m.header.stamp.nanosec:09d}",
              flush=True)
        done.set_result(True)

    node.create_subscription(LaserScan, "/scan", cb, 5)
    rclpy.spin_until_future_complete(node, done, timeout_sec=8.0)
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
