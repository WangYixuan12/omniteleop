"""Zenoh <-> ROS2 bridge for Vega-1 2D SLAM.

Runs inside the SLAM container (ros:humble + dexcomm==0.4.19).

Forward path (zenoh -> ROS):
  sensors/lidar_2d_front/scan  -> /scan (sensor_msgs/LaserScan, binned uniform)
  state/chassis/{steer,drive}  -> swerve FK odometry -> /odom + TF odom->base_link
  static TF base_link->laser from the URDF lidar_mount (+ calibrated yaw)

Reverse path (ROS -> zenoh), active once slam_toolbox publishes map->odom:
  tf2 lookup map->base_link -> <prefix>/slam/pose (JSON via JsonDataCodec)

Threading: dexcomm callbacks run on Rust threads and only store the latest
message under a lock; all ROS publishing happens on rclpy timers.

Usage:
  python3 /slam/bridge/zenoh_ros_bridge.py --params /slam/bridge/bridge_params.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformBroadcaster, TransformListener

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swerve_odometry import SwerveOdometry  # noqa: E402

from dexcomm import Publisher as ZenohPublisher  # noqa: E402
from dexcomm import Subscriber as ZenohSubscriber  # noqa: E402
from dexcomm.codecs import JointStateCodec, JsonDataCodec, LidarScan2DCodec  # noqa: E402


def find_zenoh_config() -> str | None:
    """Same resolution order as dexcontrol: ZENOH_CONFIG env, then ~/.dexmate."""
    env = os.getenv("ZENOH_CONFIG")
    if env:
        return env
    base_dir = Path("~/.dexmate/comm/zenoh/").expanduser()
    for pattern in ("**/*.dzcfg", "**/zenoh*config*.json5"):
        found = sorted(base_dir.glob(pattern))
        if found:
            return found[0].as_posix()
    return None


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) quaternion for a pure z rotation."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def ns_to_ros_time(ns: int):
    from builtin_interfaces.msg import Time

    return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))


class Latest:
    """Thread-safe latest-message slot written by dexcomm callbacks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._msg: dict | None = None
        self._recv_ns: int = 0
        self._seq: int = 0

    def store(self, msg: dict) -> None:
        with self._lock:
            self._msg = msg
            self._recv_ns = time.time_ns()
            self._seq += 1

    def read(self) -> tuple[dict | None, int, int]:
        with self._lock:
            return self._msg, self._recv_ns, self._seq


class ZenohRosBridge(Node):
    def __init__(self, params: dict) -> None:
        super().__init__("zenoh_ros_bridge")
        self.p = params
        prefix = params["robot_prefix"]
        self.frames = params["frames"]
        zcfg = find_zenoh_config()
        self.get_logger().info(f"zenoh config: {zcfg or 'dexcomm defaults'}")

        # --- zenoh side ---
        self.lidar = Latest()
        self.steer = Latest()
        self.drive = Latest()
        self._zsubs = [
            ZenohSubscriber(f"{prefix}/{params['topics']['lidar']}",
                            callback=self.lidar.store,
                            decoder=LidarScan2DCodec.decode, config=zcfg),
            ZenohSubscriber(f"{prefix}/{params['topics']['steer']}",
                            callback=self.steer.store,
                            decoder=JointStateCodec.decode, config=zcfg),
            ZenohSubscriber(f"{prefix}/{params['topics']['drive']}",
                            callback=self.drive.store,
                            decoder=JointStateCodec.decode, config=zcfg),
        ]
        self._zpub_pose = ZenohPublisher(f"{prefix}/{params['topics']['pose_out']}",
                                         encoder=JsonDataCodec.encode, config=zcfg)

        # --- ROS side ---
        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", sensor_qos)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf_pub = TransformBroadcaster(self)
        self.tf_static = StaticTransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._publish_static_tf()

        # --- scan binning precomputation ---
        sp = params["scan"]
        self.num_bins = int(sp["num_bins"])
        if self.num_bins < 90:
            raise ValueError(f"num_bins={self.num_bins} unreasonably small")
        self.angle_min = -math.pi
        self.angle_inc = 2.0 * math.pi / self.num_bins
        self.angle_sign = float(sp["angle_sign"])
        if self.angle_sign not in (1.0, -1.0):
            raise ValueError(f"angle_sign must be +-1, got {self.angle_sign}")
        self.range_min = float(sp["range_min"])
        self.range_max = float(sp["range_max"])
        self.scan_stamp_sensor = sp["stamp_mode"] == "sensor"

        # --- odometry ---
        op = params["odom"]
        self.odo = SwerveOdometry(drive_state_mode=op["drive_state_mode"],
                                  lateral_weight=float(op["lateral_weight"]))
        self.odom_stamp_sensor = op["stamp_mode"] == "sensor"
        self._last_odo_seq = 0
        self._last_odo_recv_ns = 0

        # --- timers ---
        self._last_scan_seq = 0
        self._last_scan_warn = 0.0
        self.create_timer(0.01, self._scan_tick)
        self.create_timer(1.0 / float(op["rate_hz"]), self._odom_tick)
        self.create_timer(1.0 / float(params["pose_out_rate_hz"]), self._pose_tick)
        self.create_timer(2.0, self._health_tick)
        self.get_logger().info("bridge up")

    # ------------------------------------------------------------------ scan
    def _publish_static_tf(self) -> None:
        xyz = self.p["laser_mount_xyz"]
        if len(xyz) != 3:
            raise ValueError(f"laser_mount_xyz must have 3 entries, got {xyz}")
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.frames["base"]
        t.child_frame_id = self.frames["laser"]
        t.transform.translation.x = float(xyz[0])
        t.transform.translation.y = float(xyz[1])
        t.transform.translation.z = float(xyz[2])
        qx, qy, qz, qw = yaw_to_quat(float(self.p["scan"]["laser_yaw"]))
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_static.sendTransform(t)

    def _scan_tick(self) -> None:
        msg, recv_ns, seq = self.lidar.read()
        if msg is None or seq == self._last_scan_seq:
            return
        self._last_scan_seq = seq

        ranges = np.asarray(msg["ranges"], dtype=np.float64)
        angles = np.asarray(msg["angles"], dtype=np.float64)
        if ranges.shape != angles.shape or ranges.ndim != 1 or ranges.size == 0:
            raise ValueError(
                f"bad scan: ranges{ranges.shape} angles{angles.shape}")

        theta = self.angle_sign * angles
        theta = np.arctan2(np.sin(theta), np.cos(theta))  # wrap to [-pi, pi)
        valid = np.isfinite(ranges) & (ranges >= self.range_min) & (ranges <= self.range_max)

        bins = np.full(self.num_bins, np.inf, dtype=np.float64)
        idx = np.round((theta[valid] - self.angle_min) / self.angle_inc).astype(int) % self.num_bins
        np.minimum.at(bins, idx, ranges[valid])

        out = LaserScan()
        ts_ns = int(msg.get("timestamp_ns", msg.get("timestamp", 0)))
        out.header.stamp = ns_to_ros_time(ts_ns if (self.scan_stamp_sensor and ts_ns > 0)
                                          else recv_ns)
        out.header.frame_id = self.frames["laser"]
        out.angle_min = self.angle_min
        out.angle_max = self.angle_min + (self.num_bins - 1) * self.angle_inc
        out.angle_increment = self.angle_inc
        out.time_increment = 0.0
        out.scan_time = float(msg.get("scan_time") or 0.1)
        out.range_min = self.range_min
        out.range_max = self.range_max
        out.ranges = bins.astype(np.float32).tolist()
        # Re-stamp odom->base_link at the exact scan time: scan stamps lead the
        # newest drive-state TF sample by up to ~20 ms, and exact-stamp
        # consumers (e.g. Foxglove) refuse to extrapolate into the future.
        self._send_odom_tf(out.header.stamp)
        self.scan_pub.publish(out)

    # ------------------------------------------------------------------ odom
    def _send_odom_tf(self, stamp) -> None:
        """Broadcast odom->base_link from the current integrated pose."""
        pose = self.odo.pose
        qx, qy, qz, qw = yaw_to_quat(float(pose[2]))
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.frames["odom"]
        t.child_frame_id = self.frames["base"]
        t.transform.translation.x = float(pose[0])
        t.transform.translation.y = float(pose[1])
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_pub.sendTransform(t)

    def _odom_tick(self) -> None:
        steer_msg, _, steer_seq = self.steer.read()
        drive_msg, drive_recv_ns, drive_seq = self.drive.read()
        if steer_msg is None or drive_msg is None:
            return
        if drive_seq == self._last_odo_seq:
            return  # no new drive sample; don't integrate stale velocities

        steer_pos = np.asarray(steer_msg["pos"], dtype=np.float64)
        drive_vel = np.asarray(drive_msg["vel"], dtype=np.float64)
        if steer_pos.shape != (2,) or drive_vel.shape != (2,):
            raise ValueError(
                f"chassis state shapes: steer{steer_pos.shape} drive{drive_vel.shape}")

        if self._last_odo_recv_ns == 0:
            self._last_odo_recv_ns = drive_recv_ns
            self._last_odo_seq = drive_seq
            return
        dt = (drive_recv_ns - self._last_odo_recv_ns) / 1e9
        self._last_odo_recv_ns = drive_recv_ns
        self._last_odo_seq = drive_seq
        if not (0.0 < dt < 1.0):
            self.get_logger().warning(f"odom dt={dt:.3f}s out of range, skipping")
            return

        pose = self.odo.update(steer_pos, drive_vel, dt)
        vx, vy, wz = self.odo.twist

        stamp_ns = int(drive_msg.get("timestamp_ns", 0))
        stamp = ns_to_ros_time(stamp_ns if (self.odom_stamp_sensor and stamp_ns > 0)
                               else drive_recv_ns)
        qx, qy, qz, qw = yaw_to_quat(float(pose[2]))
        self._send_odom_tf(stamp)

        od = Odometry()
        od.header.stamp = stamp
        od.header.frame_id = self.frames["odom"]
        od.child_frame_id = self.frames["base"]
        od.pose.pose.position.x = float(pose[0])
        od.pose.pose.position.y = float(pose[1])
        od.pose.pose.orientation.x = qx
        od.pose.pose.orientation.y = qy
        od.pose.pose.orientation.z = qz
        od.pose.pose.orientation.w = qw
        od.twist.twist.linear.x = float(vx)
        od.twist.twist.linear.y = float(vy)
        od.twist.twist.angular.z = float(wz)
        self.odom_pub.publish(od)

    # --------------------------------------------------------------- pose out
    def _pose_tick(self) -> None:
        try:
            tf = self.tf_buffer.lookup_transform(self.frames["map"], self.frames["base"],
                                                 rclpy.time.Time())
        except Exception:
            return  # slam_toolbox not up yet; silent until map frame exists
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._zpub_pose.publish({
            "timestamp_ns": time.time_ns(),
            "frame": self.frames["map"],
            "child_frame": self.frames["base"],
            "pos": [tf.transform.translation.x, tf.transform.translation.y,
                    tf.transform.translation.z],
            "quat_wxyz": [q.w, q.x, q.y, q.z],
            "yaw": yaw,
        })

    # ----------------------------------------------------------------- health
    def _health_tick(self) -> None:
        _, recv_ns, seq = self.lidar.read()
        now = time.time()
        if seq == 0:
            if now - self._last_scan_warn > 10.0:
                self.get_logger().warning(
                    "no lidar scans received yet — is `dexsensor launch --sensor "
                    "lidar_2d` running? (zenoh peer discovery can take ~10 s)")
                self._last_scan_warn = now
        elif time.time_ns() - recv_ns > 2_000_000_000:
            if now - self._last_scan_warn > 10.0:
                self.get_logger().warning(
                    f"lidar silent for {(time.time_ns() - recv_ns) / 1e9:.1f} s — "
                    "dexsensor may have died; SLAM is coasting on odometry")
                self._last_scan_warn = now

    def shutdown(self) -> None:
        for sub in self._zsubs:
            sub.shutdown()
        self._zpub_pose.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path,
                        default=Path(__file__).resolve().parent / "bridge_params.yaml")
    args = parser.parse_args()
    params = yaml.safe_load(args.params.read_text())

    rclpy.init()
    node = ZenohRosBridge(params)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
