"""Check whether /scan messages are transformable at their exact stamp.

Reproduces the renderer's lookup: for each incoming scan, try
lookup_transform(map, laser, scan.stamp) with zero timeout and report
extrapolation failures (run inside the SLAM container).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener


def main() -> None:
    rclpy.init()
    node = Node("scan_tf_check")
    buf = Buffer()
    TransformListener(buf, node)
    results = {"ok": 0, "fail": 0}
    errors: list[str] = []

    def cb(m: LaserScan) -> None:
        try:
            buf.lookup_transform("map", m.header.frame_id, m.header.stamp)
            results["ok"] += 1
        except Exception as e:  # noqa: BLE001
            results["fail"] += 1
            if len(errors) < 3:
                errors.append(str(e)[:200])

    node.create_subscription(LaserScan, "/scan", cb, 5)
    end = node.get_clock().now().nanoseconds + int(10e9)
    while rclpy.ok() and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    print(f"lookups at scan stamp: ok={results['ok']} fail={results['fail']}", flush=True)
    for e in errors:
        print(f"  e.g.: {e}", flush=True)
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
