"""Versioned, synchronized stop-and-capture head stereo sweep artifact."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

HEAD_STEREO_SWEEP_SCHEMA = "dexmate_head_stereo_sweep_v1"


def _array(value, dtype, *, name: str) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    if result.dtype.kind == "f" and not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    result.setflags(write=False)
    return result


def _integer_array(value, dtype, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")
    limits = np.iinfo(dtype)
    too_low = raw.dtype.kind == "i" and raw.size and np.any(raw < limits.min)
    too_high = raw.size and np.any(raw > limits.max)
    if too_low or too_high:
        raise ValueError(f"{name} contains values outside {np.dtype(dtype)} range")
    return _array(raw, dtype, name=name)


def _float_array(value, dtype, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind != "f":
        raise ValueError(f"{name} must be floating point")
    return _array(raw, dtype, name=name)


def _camera_matrix(value, *, name: str) -> np.ndarray:
    result = _float_array(value, np.float32, name=name)
    if (
        result.shape != (3, 3)
        or result[0, 0] <= 0
        or result[1, 1] <= 0
        or not np.allclose(result[2], (0, 0, 1), atol=1e-6)
    ):
        raise ValueError(f"{name} must be a finite pinhole (3,3) matrix")
    return result


@dataclass(frozen=True)
class HeadStereoSweep:
    """Seven (or more) settled, exact-timestamp rectified stereo observations.

    ``base_t_cam`` is the optical left-camera pose from full measured-joint FK.  The
    base does not move during capture, so this is also the sweep's fusion frame.
    """

    left_rgb: np.ndarray
    right_rgb: np.ndarray
    capture_timestamp_ns: np.ndarray
    left_receive_ns: np.ndarray
    right_receive_ns: np.ndarray
    head_joints: np.ndarray
    torso_joints: np.ndarray
    left_arm_joints: np.ndarray
    right_arm_joints: np.ndarray
    base_t_cam: np.ndarray
    target_pan_deg: np.ndarray
    measured_pan_deg: np.ndarray
    capture_joint_motion_deg: np.ndarray
    left_k: np.ndarray
    right_k: np.ndarray
    baseline_m: float
    sensor_id: str = "head_camera"
    coordinate_frame: str = "base"

    def __post_init__(self) -> None:
        left = _integer_array(self.left_rgb, np.uint8, name="left_rgb")
        right = _integer_array(self.right_rgb, np.uint8, name="right_rgb")
        if left.ndim != 4 or left.shape[-1] != 3 or right.shape != left.shape:
            raise ValueError("left/right RGB must share non-empty (N,H,W,3) shape")
        views = left.shape[0]
        if views < 1:
            raise ValueError("stereo sweep must contain at least one view")
        object.__setattr__(self, "left_rgb", left)
        object.__setattr__(self, "right_rgb", right)

        timestamps = _integer_array(
            self.capture_timestamp_ns, np.int64, name="capture_timestamp_ns"
        )
        left_receive = _integer_array(self.left_receive_ns, np.int64, name="left_receive_ns")
        right_receive = _integer_array(self.right_receive_ns, np.int64, name="right_receive_ns")
        for name, value in (
            ("capture_timestamp_ns", timestamps),
            ("left_receive_ns", left_receive),
            ("right_receive_ns", right_receive),
        ):
            if value.shape != (views,) or np.any(value <= 0):
                raise ValueError(f"{name} must be positive with one value per view")
        if views > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError("capture timestamps must be strictly increasing")
        object.__setattr__(self, "capture_timestamp_ns", timestamps)
        object.__setattr__(self, "left_receive_ns", left_receive)
        object.__setattr__(self, "right_receive_ns", right_receive)

        joint_shapes = {
            "head_joints": 3,
            "torso_joints": 3,
            "left_arm_joints": 7,
            "right_arm_joints": 7,
        }
        for name, width in joint_shapes.items():
            value = _float_array(getattr(self, name), np.float32, name=name)
            if value.shape != (views, width):
                raise ValueError(f"{name} must have shape ({views},{width})")
            object.__setattr__(self, name, value)

        transforms = _float_array(self.base_t_cam, np.float32, name="base_t_cam")
        if transforms.shape != (views, 4, 4):
            raise ValueError(f"base_t_cam must have shape ({views},4,4)")
        expected_last = np.broadcast_to(np.array([0, 0, 0, 1]), (views, 4))
        if not np.allclose(transforms[:, 3], expected_last, atol=1e-5):
            raise ValueError("base_t_cam has an invalid homogeneous last row")
        rotations = transforms[:, :3, :3]
        eye = np.broadcast_to(np.eye(3), rotations.shape)
        if not np.allclose(
            rotations @ rotations.transpose(0, 2, 1), eye, atol=2e-3
        ) or not np.allclose(np.linalg.det(rotations), 1.0, atol=2e-3):
            raise ValueError("base_t_cam rotations must be proper and orthonormal")
        object.__setattr__(self, "base_t_cam", transforms)

        target = _float_array(self.target_pan_deg, np.float32, name="target_pan_deg")
        measured = _float_array(self.measured_pan_deg, np.float32, name="measured_pan_deg")
        motion = _float_array(
            self.capture_joint_motion_deg,
            np.float32,
            name="capture_joint_motion_deg",
        )
        if (
            target.shape != (views,)
            or measured.shape != (views,)
            or motion.shape != (views,)
            or np.any(motion < 0)
        ):
            raise ValueError(
                "target/measured pan and non-negative joint motion need one value per view"
            )
        delta = np.diff(target)
        if views > 1 and not (np.all(delta > 0) or np.all(delta < 0)):
            raise ValueError("target pans must be strictly monotonic and unique")
        object.__setattr__(self, "target_pan_deg", target)
        object.__setattr__(self, "measured_pan_deg", measured)
        object.__setattr__(self, "capture_joint_motion_deg", motion)

        object.__setattr__(self, "left_k", _camera_matrix(self.left_k, name="left_k"))
        object.__setattr__(self, "right_k", _camera_matrix(self.right_k, name="right_k"))
        baseline = float(self.baseline_m)
        if not np.isfinite(baseline) or not 0.01 < baseline < 0.5:
            raise ValueError(f"baseline_m is implausible: {baseline}")
        object.__setattr__(self, "baseline_m", baseline)
        if not str(self.sensor_id).strip() or not str(self.coordinate_frame).strip():
            raise ValueError("sensor_id and coordinate_frame must be non-empty")
        if self.coordinate_frame != "base":
            raise ValueError("head stereo sweep v1 requires coordinate_frame='base'")

    @property
    def intrinsics(self) -> np.ndarray:
        """Per-view left camera matrices for downstream cache construction."""
        result = np.repeat(self.left_k[None], len(self.left_rgb), axis=0)
        result.setflags(write=False)
        return result

    def save(self, path: Path | str) -> None:
        """Atomically write a new artifact; never silently overwrite a capture."""
        path = Path(path)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"stereo sweep exists: {path}; delete it explicitly")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.building-{os.getpid()}")
        try:
            with h5py.File(temporary, "x") as out:
                out.attrs["schema"] = HEAD_STEREO_SWEEP_SCHEMA
                out.attrs["coordinate_frame"] = self.coordinate_frame
                images = out.require_group("obs/images")
                images.create_dataset("head_left_rgb", data=self.left_rgb, compression="lzf")
                images.create_dataset("head_right_rgb", data=self.right_rgb, compression="lzf")
                images.create_dataset("head_frame_ns", data=self.capture_timestamp_ns)
                # Both eyes are independently transported, but this artifact admits only
                # physical pairs whose publisher capture stamps are exactly equal.
                images.create_dataset("head_right_frame_ns", data=self.capture_timestamp_ns)
                images.create_dataset("head_receive_ns", data=self.left_receive_ns)
                images.create_dataset("head_right_receive_ns", data=self.right_receive_ns)
                images.create_dataset("intrinsic", data=self.intrinsics)
                images.create_dataset("extrinsic", data=self.base_t_cam)
                joints = out.require_group("obs/joint")
                joints.create_dataset("head", data=self.head_joints)
                joints.create_dataset("torso", data=self.torso_joints)
                joints.create_dataset("left_arm", data=self.left_arm_joints)
                joints.create_dataset("right_arm", data=self.right_arm_joints)
                meta = out.require_group("meta")
                meta.create_dataset("target_pan_deg", data=self.target_pan_deg)
                meta.create_dataset("measured_pan_deg", data=self.measured_pan_deg)
                meta.create_dataset("capture_joint_motion_deg", data=self.capture_joint_motion_deg)
                stereo = meta.require_group("head_stereo")
                stereo.create_dataset("baseline_m", data=np.float32(self.baseline_m))
                stereo.create_dataset("left_K", data=self.left_k)
                stereo.create_dataset("right_K", data=self.right_k)
                stereo.create_dataset("rectified", data=np.bool_(True))
                stereo.create_dataset("sensor_id", data=np.bytes_(self.sensor_id))
                out.flush()
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: Path | str) -> "HeadStereoSweep":
        """Load and revalidate every field in a v1 sweep."""
        path = Path(path)
        with h5py.File(path, "r") as data:
            if str(data.attrs.get("schema", "")) != HEAD_STEREO_SWEEP_SCHEMA:
                raise ValueError(f"{path}: unsupported stereo-sweep schema")
            left_ns = data["obs/images/head_frame_ns"][:]
            right_ns = data["obs/images/head_right_frame_ns"][:]
            if not np.array_equal(left_ns, right_ns):
                raise ValueError(f"{path}: left/right capture timestamps are not exact pairs")
            rectified = np.asarray(data["meta/head_stereo/rectified"])
            if rectified.shape != () or rectified.dtype != np.bool_ or not bool(rectified):
                raise ValueError(f"{path}: stereo images are not rectified")
            sensor = np.asarray(data["meta/head_stereo/sensor_id"]).item()
            if isinstance(sensor, bytes):
                sensor = sensor.decode("utf-8")
            return cls(
                left_rgb=data["obs/images/head_left_rgb"][:],
                right_rgb=data["obs/images/head_right_rgb"][:],
                capture_timestamp_ns=left_ns,
                left_receive_ns=data["obs/images/head_receive_ns"][:],
                right_receive_ns=data["obs/images/head_right_receive_ns"][:],
                head_joints=data["obs/joint/head"][:],
                torso_joints=data["obs/joint/torso"][:],
                left_arm_joints=data["obs/joint/left_arm"][:],
                right_arm_joints=data["obs/joint/right_arm"][:],
                base_t_cam=data["obs/images/extrinsic"][:],
                target_pan_deg=data["meta/target_pan_deg"][:],
                measured_pan_deg=data["meta/measured_pan_deg"][:],
                capture_joint_motion_deg=data["meta/capture_joint_motion_deg"][:],
                left_k=data["meta/head_stereo/left_K"][:],
                right_k=data["meta/head_stereo/right_K"][:],
                baseline_m=float(np.asarray(data["meta/head_stereo/baseline_m"])),
                sensor_id=str(sensor),
                coordinate_frame=str(data.attrs["coordinate_frame"]),
            )
