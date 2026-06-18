"""Robot base positioning from head-ZED positional tracking + forward kinematics.

The head ZED X Mini supports the ZED SDK positional-tracking module (VSLAM +
IMU fusion; https://docs.stereolabs.com/docs/development/zed-sdk/modules/positional-tracking).
The SDK reports the pose of the tracking camera in a "tracking world" frame
fixed at ``enable_positional_tracking()`` time. The camera rides on the
torso+head chain, so the rigid base pose follows from forward kinematics:

    T_odom_base(t) = T_odom_track · T_track_cam(t) · ( T_base_cam(q_t) )⁻¹

where

  * ``T_track_cam`` — camera pose from ``zed.get_position(..., WORLD)``. The ZED
    SDK tracks the camera's LEFT EYE — ``getPosition()`` returns "the pose of the
    Camera Frame located on the left eye" (https://docs.stereolabs.com/docs/
    development/zed-sdk/modules/positional-tracking/coordinate-frames), so the
    matching URDF frame is ``zed_left_camera``, NOT ``zed_depth_frame`` (the URDF
    places the latter 11.5 mm behind the left eye along the optical z; using it
    couples a yaw-dependent ≲23 mm error into the base position). With the
    publisher's explicit ``COORDINATE_SYSTEM.IMAGE`` the camera frame is the
    optical convention (x right, y down, z forward) — exactly
    ``zed_left_camera``'s orientation (mounted rpy=(-π/2, 0, -π/2) off
    ``head_l3``), so no axis remap is needed anywhere.
  * ``T_base_cam(q)`` — Pinocchio FK on the Vega URDF, base → ``zed_left_camera``,
    a function of the 3 torso + 3 head joints only (name-matched to
    ``state/torso`` / ``state/head``, which use URDF joint names and convention —
    same parity verified by scripts/analysis/verify_fk_and_residual.py for the arm chain).
  * ``T_odom_track`` — fixed anchor chosen at the first tracked sample so that
    the odom frame coincides with the robot base at t₀ (gravity-aligned iff the
    base started on flat ground). Robust to whatever world convention the SDK
    picked (set_gravity_as_origin etc.).

Pure math + Pinocchio only — no camera, no Zenoh. The camera-owning publisher
(tests/test_head_zedx_depth.py --enable-tracking) publishes ``T_track_cam``;
scripts/zed_base_pose_node.py joins it with live joint states through this
module. Offline unit test: tests/test_zed_base_pose.py.
"""

from __future__ import annotations

import importlib.util
import pathlib
from typing import Optional, Sequence

import numpy as np
import pinocchio as pin

# vega_1.urdf from the dexmate_urdf wheel (same model family the rest of the
# stack uses; resolved from the installed package so the path works on both
# the Jetson and the workstation).
_TORSO_JOINTS = ("torso_j1", "torso_j2", "torso_j3")
_HEAD_JOINTS = ("head_j1", "head_j2", "head_j3")
# The ZED SDK's positional tracking reports the pose of the LEFT EYE, so FK must
# target the left-camera optical frame. zed_depth_frame is a different URDF link
# (11.5 mm behind the left eye) used elsewhere for depth/teleop, NOT for tracking.
DEFAULT_CAMERA_FRAME = "zed_left_camera"


def default_vega_urdf() -> str:
    """Path of vega_1.urdf inside the installed dexmate_urdf package."""
    spec = importlib.util.find_spec("dexmate_urdf")
    if spec is None or spec.origin is None:
        raise ImportError(
            "dexmate_urdf is not installed in this environment; pass urdf_path "
            "explicitly to VegaHeadCamFK."
        )
    root = pathlib.Path(spec.origin).parent
    urdf = root / "robots" / "humanoid" / "vega_1" / "vega_1.urdf"
    if not urdf.is_file():
        raise FileNotFoundError(f"vega_1.urdf not found under {root}")
    return urdf.as_posix()


def _validate_se3(T: np.ndarray, what: str) -> np.ndarray:
    """Validate a 4x4 homogeneous transform (shape, finiteness, SO(3) block)."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{what} must be a 4x4 matrix, got shape {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError(f"{what} contains non-finite values:\n{T}")
    row_dev = np.abs(T[3] - (0.0, 0.0, 0.0, 1.0)).max()
    if row_dev > 1e-6:
        raise ValueError(
            f"{what} last row must be [0,0,0,1], got {T[3]!r} (max dev {row_dev:.3e})"
        )
    R = T[:3, :3]
    orth_dev = np.abs(R @ R.T - np.eye(3)).max()
    if orth_dev > 1e-5:
        raise ValueError(
            f"{what} rotation block is not orthonormal (max |RRᵀ-I| = {orth_dev:.3e})"
        )
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-5):
        raise ValueError(f"{what} rotation block has det {np.linalg.det(R):.6f} ≠ +1")
    return T


def _validate_joints(pos: Sequence[float], what: str) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape != (3,):
        raise ValueError(f"{what} must have exactly 3 joint positions, got shape {pos.shape}")
    if not np.all(np.isfinite(pos)):
        raise ValueError(f"{what} contains non-finite values: {pos}")
    return pos


def pose_from_quat_trans(
    quat_xyzw: Sequence[float],
    translation: Sequence[float],
    what: str = "zed pose",
    max_norm_dev: float = 1e-3,
) -> np.ndarray:
    """Build an exact SE(3) matrix from the SDK's quaternion + translation.

    Preferred over the SDK's 4x4 matrix: that matrix is denormalized float32
    (det(R)≈1.002, T[3,3]≈0.99992 observed live), whereas a unit quaternion
    re-normalized in
    float64 gives an exactly orthonormal rotation — the only loss left is the
    float32 quantization of the quaternion itself (~1e-7, far below tracker
    noise). Quaternion order is (x, y, z, w) as returned by
    ``sl.Pose.get_orientation().get()``.
    """
    q = np.asarray(quat_xyzw, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"{what} quaternion must have shape (4,), got {q.shape}")
    if t.shape != (3,):
        raise ValueError(f"{what} translation must have shape (3,), got {t.shape}")
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(t))):
        raise ValueError(f"{what} quaternion/translation contain non-finite values")
    norm = np.linalg.norm(q)
    if abs(norm - 1.0) > max_norm_dev:
        raise ValueError(
            f"{what} quaternion norm {norm!r} deviates from 1 by more than "
            f"{max_norm_dev} — not a rotation"
        )
    x, y, z, w = q / norm
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    T[:3, 3] = t
    return T


class VegaHeadCamFK:
    """FK base → head-camera optical frame on the Vega URDF (fixed-base model).

    Only torso_j1..3 and head_j1..3 move this chain; all other joints stay at
    ``pin.neutral`` (wheels are continuous joints — neutral keeps their cos/sin
    representation valid).
    """

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        camera_frame: str = DEFAULT_CAMERA_FRAME,
    ) -> None:
        self.urdf_path = urdf_path or default_vega_urdf()
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        if not self.model.existFrame(camera_frame):
            raise ValueError(
                f"Frame {camera_frame!r} not found in {self.urdf_path}; "
                f"available camera-ish frames: "
                f"{[f.name for f in self.model.frames if 'cam' in f.name or 'zed' in f.name]}"
            )
        self.camera_frame = camera_frame
        self._frame_id = self.model.getFrameId(camera_frame)

        self._q_neutral = pin.neutral(self.model)
        self._idx_q: dict[str, int] = {}
        for name in _TORSO_JOINTS + _HEAD_JOINTS:
            if not self.model.existJointName(name):
                raise ValueError(f"Joint {name!r} not found in {self.urdf_path}")
            joint_id = self.model.getJointId(name)
            joint = self.model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError(
                    f"Joint {name!r} has nq={joint.nq}, expected 1 (bounded revolute)"
                )
            self._idx_q[name] = joint.idx_q

    def base_T_cam(self, torso_pos: Sequence[float], head_pos: Sequence[float]) -> np.ndarray:
        """T_base_cam(q): pose of the camera optical frame in the base frame.

        Args:
            torso_pos: (3,) torso_j1..3 from ``state/torso`` ``pos`` (URDF order).
            head_pos:  (3,) head_j1..3  from ``state/head``  ``pos`` (URDF order).
        """
        torso_pos = _validate_joints(torso_pos, "torso_pos")
        head_pos = _validate_joints(head_pos, "head_pos")
        q = self._q_neutral.copy()
        for name, value in zip(_TORSO_JOINTS, torso_pos):
            q[self._idx_q[name]] = value
        for name, value in zip(_HEAD_JOINTS, head_pos):
            q[self._idx_q[name]] = value
        pin.framesForwardKinematics(self.model, self.data, q)
        return np.array(self.data.oMf[self._frame_id].homogeneous, dtype=np.float64)


class ZedBasePoseEstimator:
    """Fuses ZED camera world poses with FK into robot base poses in an odom frame.

    The odom frame is the robot base frame at the first ``update()`` call
    (anchor). All inputs/outputs are 4x4 row-major homogeneous matrices.
    """

    def __init__(self, fk: VegaHeadCamFK) -> None:
        self.fk = fk
        self.odom_T_track: Optional[np.ndarray] = None

    @property
    def anchored(self) -> bool:
        return self.odom_T_track is not None

    def reset(self) -> None:
        """Drop the anchor; the next update re-anchors odom = base@now."""
        self.odom_T_track = None

    def update(
        self,
        track_T_cam: np.ndarray,
        torso_pos: Sequence[float],
        head_pos: Sequence[float],
    ) -> np.ndarray:
        """Return T_odom_base for one tracked camera pose + matching joint state.

        Only call with poses the tracker reported as OK — feeding SEARCHING /
        diverged poses will silently corrupt the anchor.

        ``track_T_cam`` must be a strict SE(3) matrix. Build it from the SDK's
        quaternion+translation via ``pose_from_quat_trans`` (exactly orthonormal
        in float64) — do not pass the SDK's raw 4x4 matrix, which is denormalized
        float32 and will fail validation.
        """
        track_T_cam = _validate_se3(track_T_cam, "track_T_cam")
        base_T_cam = _validate_se3(
            self.fk.base_T_cam(torso_pos, head_pos), "base_T_cam (FK output)"
        )
        cam_T_base = np.linalg.inv(base_T_cam)
        if self.odom_T_track is None:
            # anchor: odom := base@t0  ⇒  odom_T_track = base_T_cam(q0) · cam_T_track
            self.odom_T_track = base_T_cam @ np.linalg.inv(track_T_cam)
        odom_T_base = self.odom_T_track @ track_T_cam @ cam_T_base
        return _validate_se3(odom_T_base, "odom_T_base")


def se3_to_xy_yaw(T: np.ndarray) -> tuple[float, float, float]:
    """Planar (x, y, yaw) of a base pose — yaw from the x-axis projection."""
    T = _validate_se3(T, "T")
    return (
        float(T[0, 3]),
        float(T[1, 3]),
        float(np.arctan2(T[1, 0], T[0, 0])),
    )
