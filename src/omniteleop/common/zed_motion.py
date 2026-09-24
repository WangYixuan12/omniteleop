"""Camera-owned ZED tracking, with image publication before GEN_3 processing.

Only the capture thread calls the SDK. The worker receives owned Python values
and performs telemetry publication; transport can never block image capture.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import deque

import numpy as np
from loguru import logger


def _finite(value, shape):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Invalid motion field: expected finite {shape}, got {array.shape}")
    return array


def resolve_depth_mode(args):
    """SDK depth can support tracking without adding a depth publication stream."""
    if args.enable_depth:
        return args.depth_mode
    if args.enable_tracking and getattr(args, "tracking_depth_mode", None):
        return args.tracking_depth_mode
    return "NONE"


class ZedMotion:
    """Optional pose + full-rate IMU telemetry for an already-open camera."""

    def __init__(self, zed, node, camera_info, args, qos):
        import pyzed.sl as sl
        from dexcomm.codecs import JsonDataCodec

        self.sl, self.zed = sl, zed
        self.tracking_enabled = bool(args.enable_tracking)
        self.enabled = self.tracking_enabled or bool(getattr(args, "enable_imu", False))
        self.split = self.tracking_enabled and resolve_depth_mode(args) == "NONE"
        self.sensor_id = args.sensor_id
        self.session_id = uuid.uuid4().hex
        self.queue = queue.Queue(maxsize=128)
        self.stop = threading.Event()
        self.worker = None
        self.image_ages = deque(maxlen=3600)
        self.processing_ms = deque(maxlen=3600)
        self.stats = dict(enqueued=0, published=0, queue_overflow=0,
                          publish_errors=0, sdk_errors=0, timestamp_mismatch=0,
                          imu_samples=0, imu_duplicates=0, last_image_ns=0)
        self.last_imu_ns = 0
        self.metadata = dict(enabled=self.enabled, tracking_enabled=self.tracking_enabled,
                             imu_enabled=self.enabled, session_id=self.session_id,
                             image_before_tracking=self.split)
        if not self.enabled:
            return
        if self.split and args.tracking_mode != "GEN_3":
            raise ValueError("Tracking without SDK depth requires GEN_3; use tracking_depth_mode for GEN_1")
        if self.tracking_enabled:
            params = sl.PositionalTrackingParameters()
            params.mode = getattr(sl.POSITIONAL_TRACKING_MODE, args.tracking_mode)
            params.enable_area_memory = args.area_memory
            params.enable_pose_smoothing = False
            params.enable_imu_fusion = True
            params.set_gravity_as_origin = False
            if getattr(args, "area_file", None):
                params.area_file_path = args.area_file
                params.enable_localization_only = args.localization_only
            err = zed.enable_positional_tracking(params)
            if err != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Cannot enable {self.sensor_id} tracking: {err}")
            self.pose = sl.Pose()
            self.orientation, self.translation = sl.Orientation(), sl.Translation()
        self.metadata.update(
            mode=args.tracking_mode if self.tracking_enabled else "IMU_ONLY",
            sdk_depth_mode=resolve_depth_mode(args),
            area_memory=args.area_memory if self.tracking_enabled else False,
            pose_smoothing=False, imu_fusion=self.tracking_enabled, gravity_as_origin=False,
            coordinate_system="IMAGE", translation_units="m",
            quaternion_order="xyzw", gyro_units="rad/s", acceleration_units="m/s^2",
            acceleration_includes_gravity=True, serial_number=int(camera_info.serial_number),
            sdk_version=sl.Camera.get_sdk_version(),
            world_frame=f"zed_start_{self.sensor_id}_{self.session_id}",
            pose_frame="left_camera_optical", imu_frame="sdk_imu",
            pose_validation_policy="visual_odometry_required_v1",
            camera_imu_transform_sdk=_finite(
                camera_info.sensors_configuration.camera_imu_transform.m, (4, 4)
            ).tolist(),
        )
        telemetry_qos = {**qos, "priority": "background"}
        self.motion_pub = node.create_publisher(
            f"sensors/{self.sensor_id}/motion", encoder=JsonDataCodec.encode, qos=telemetry_qos)
        # Preserve the pre-existing head pose topic; wrists use the same schema.
        self.pose_pub = (node.create_publisher(
            f"sensors/{self.sensor_id}/pose", encoder=JsonDataCodec.encode, qos=telemetry_qos)
            if self.tracking_enabled else None)
        self.worker = threading.Thread(target=self._publish, daemon=True,
                                       name=f"{self.sensor_id}_motion")
        self.worker.start()
        logger.info(f"{self.sensor_id} {self.metadata['mode']} motion telemetry ON; "
                    f"images before processing={self.split}, area_memory={args.area_memory}")

    def _publish(self):
        while not self.stop.is_set() or not self.queue.empty():
            try:
                packet = self.queue.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                packet["telemetry_send_ns"] = time.time_ns()
                self.motion_pub.publish(packet)
                if self.pose_pub is not None:
                    self.pose_pub.publish(packet["pose"])
                    self.stats["last_published_pose_ns"] = packet["pose"]["timestamp_ns"]
                self.stats["published"] += 1
            except Exception as exc:
                self.stats["publish_errors"] += 1
                logger.error(f"{self.sensor_id} motion publication failed: {exc}")
            finally:
                self.queue.task_done()

    def read(self, runtime):
        return self.zed.read() if self.split else self.zed.grab(runtime)

    def after_images(self, image_ns, sequence, runtime):
        """Called AFTER both eyes were published; never waits for transport."""
        image_sent_ns = time.time_ns()
        self.image_ages.append((image_sent_ns - image_ns) / 1e6)
        self.stats["last_image_ns"] = image_ns
        if not self.enabled:
            return
        start = time.perf_counter_ns()
        try:
            self._collect(image_ns, image_sent_ns, sequence, runtime)
        except Exception as exc:
            self.stats["sdk_errors"] += 1
            if self.stats["sdk_errors"] <= 3 or self.stats["sdk_errors"] % 300 == 0:
                logger.error(f"{self.sensor_id} motion sample rejected: {exc}")
        finally:
            self.processing_ms.append((time.perf_counter_ns() - start) / 1e6)

    def _collect(self, image_ns, image_sent_ns, sequence, runtime):
        sl = self.sl
        if self.split:
            err = self.zed.grab(runtime)
            if err != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"post-image grab: {err}")
        grab_ns = int(self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
        if grab_ns != image_ns:
            self.stats["timestamp_mismatch"] += 1
            raise ValueError(f"read/grab changed capture: {image_ns} -> {grab_ns}")
        pose = dict(timestamp_ns=image_ns, image_timestamp_ns=image_ns,
                    sequence=sequence, session_id=self.session_id, valid=False,
                    state="DISABLED", available=False)
        if self.tracking_enabled:
            state = self.zed.get_position(self.pose, sl.REFERENCE_FRAME.WORLD)
            status = self.zed.get_positional_tracking_status()
            health = {key: str(getattr(status, key)) for key in
                      ("odometry_status", "spatial_memory_status", "tracking_fusion_status")}
            self.metadata["tracking_health"] = health
            visual_ok = (health["odometry_status"] == "OK" and
                         health["tracking_fusion_status"].replace(" ", "_") in ("VISUAL", "VISUAL_INERTIAL"))
            pose_ns = int(self.pose.timestamp.get_nanoseconds())
            valid = bool(self.pose.valid) and state == sl.POSITIONAL_TRACKING_STATE.OK
            valid = valid and pose_ns == image_ns
            pose = dict(timestamp_ns=pose_ns, zed_timestamp_ns=pose_ns,
                        image_timestamp_ns=image_ns, sequence=sequence,
                        session_id=self.session_id, state=str(state), valid=valid and visual_ok,
                        sdk_valid=bool(self.pose.valid), tracking_health=health,
                        invalid_reason=(None if valid and visual_ok else
                                        "sdk_pose_or_timestamp_invalid" if not valid else
                                        "visual_odometry_unavailable"),
                        confidence=int(self.pose.pose_confidence),
                        coordinate_system="IMAGE_optical", area_memory=self.metadata["area_memory"],
                        world_frame=self.metadata["world_frame"])
            if valid:
                q = _finite(self.pose.get_orientation(self.orientation).get(), (4,))
                norm = np.linalg.norm(q)
                if not .99 < norm < 1.01:
                    raise ValueError(f"Invalid pose quaternion norm {norm}")
                pose.update(quaternion_xyzw=(q / norm).tolist(),
                            translation=_finite(self.pose.get_translation(self.translation).get(), (3,)).tolist(),
                            covariance=_finite(np.asarray(self.pose.pose_covariance).reshape(6, 6), (6, 6)).tolist())
            pose["available"] = True
        batch = []
        err = self.zed.get_sensors_data_batch(batch)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"IMU batch retrieval: {err}")
        stamps, gyro, acc = [], [], []
        for sensors in batch:
            imu = sensors.get_imu_data()
            if not imu.is_available:
                continue
            stamp = int(imu.timestamp.get_nanoseconds())
            if stamp <= self.last_imu_ns:
                self.stats["imu_duplicates"] += 1
                continue
            stamps.append(stamp)
            gyro.append(list(imu.get_angular_velocity()))
            acc.append(list(imu.get_linear_acceleration()))
            self.last_imu_ns = stamp
        if stamps:
            gyro = np.deg2rad(_finite(gyro, (len(stamps), 3))).tolist()
            acc = _finite(acc, (len(stamps), 3)).tolist()
        self.stats["imu_samples"] += len(stamps)
        packet = dict(schema_version=1, sensor_id=self.sensor_id,
                      session_id=self.session_id, sequence=sequence,
                      image_timestamp_ns=image_ns, image_sent_ns=image_sent_ns,
                      pose=pose, imu=dict(timestamp_ns=stamps, gyro_rad_s=gyro, acc_m_s2=acc),
                      producer_queue_overflow=self.stats["queue_overflow"],
                      producer_sdk_errors=self.stats["sdk_errors"])
        try:
            self.queue.put_nowait(packet)
            self.stats["enqueued"] += 1
        except queue.Full:
            self.stats["queue_overflow"] += 1

    def info(self):
        def quantiles(values):
            snapshot = list(values)
            return dict(zip(("p50", "p95", "p99", "max"),
                            np.percentile(snapshot, (50, 95, 99, 100)).tolist())) if snapshot else {}
        return dict(**self.metadata, statistics=dict(self.stats),
                    queue_size=self.queue.qsize(),
                    image_publish_age_ms=quantiles(self.image_ages),
                    after_image_processing_ms=quantiles(self.processing_ms))

    def close(self):
        self.stop.set()
        if self.worker is not None:
            self.worker.join(timeout=.5)
