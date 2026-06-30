"""Reference-pose alignment helpers for WBC VR leader sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from omniteleop.follower.whole_body_ik import (
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    LEFT_EE_FRAME,
    RIGHT_ARM_JOINTS,
    RIGHT_EE_FRAME,
    TORSO_JOINTS,
    VegaWholeBodyIK,
    WBCConfig,
)
from omniteleop.leader.wbc_headset_hud import HandAlignmentStatus

REFERENCE_ALIGN_POS_TOL_MM = 30.0
REFERENCE_ALIGN_ROT_TOL_DEG = 10.0
REFERENCE_ALIGN_STABLE_S = 1.0


def _to_mat(pose) -> np.ndarray:
    """4x4 matrix from a pin.SE3 (``.homogeneous``) or a 4x4 array."""
    return np.asarray(pose.homogeneous if hasattr(pose, "homogeneous") else pose, dtype=float)


def optional_reference_path(path: Optional[str]) -> Optional[str]:
    """Normalize optional reference path CLI input."""
    if path is None:
        return None
    text = str(path).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _normalize_frame_idx(length: int, frame_idx: int, key: str) -> int:
    idx = frame_idx + length if frame_idx < 0 else frame_idx
    if length <= 0 or idx < 0 or idx >= length:
        raise ValueError(
            f"reference episode {key!r} does not contain frame {frame_idx}; "
            f"dataset length is {length}"
        )
    return idx


def _episode_joint_frame(f, key: str, names: list[str], frame_idx: int) -> np.ndarray:
    if key not in f:
        raise ValueError(f"reference episode missing {key!r}")
    arr = np.asarray(f[key], dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(names):
        raise ValueError(
            f"reference episode {key!r} must have shape (N,{len(names)}) "
            f"got {arr.shape}"
        )
    idx = _normalize_frame_idx(arr.shape[0], frame_idx, key)
    out = arr[idx]
    if not np.all(np.isfinite(out)):
        raise ValueError(f"reference episode {key!r}[{idx}] contains non-finite values")
    return out


def _set_joint_group(
    q: np.ndarray, ik: VegaWholeBodyIK, names: list[str], values: np.ndarray
) -> None:
    for name, value in zip(names, values, strict=True):
        q[ik._idx_q[name]] = float(value)  # noqa: SLF001 -- FK reconstruction


def load_reference_ee_poses(
    path: Optional[str],
    ik: Optional[VegaWholeBodyIK] = None,
    *,
    frame_idx: int = -1,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Load reference L/R EEF poses from a raw-data episode's observed frame.

    By default this reconstructs the final recorded frame.
    """
    ref_path = optional_reference_path(path)
    if ref_path is None:
        return None
    p = Path(ref_path)
    if not p.exists():
        raise FileNotFoundError(f"alignment reference episode not found: {p}")

    own_ik = ik is None
    if ik is None:
        ik = VegaWholeBodyIK(WBCConfig())
    q = ik.nominal_q()

    import h5py  # noqa: PLC0415 -- optional dep, only needed when alignment is enabled

    with h5py.File(p, "r") as f:
        _set_joint_group(
            q,
            ik,
            TORSO_JOINTS,
            _episode_joint_frame(f, "obs/joint/torso", TORSO_JOINTS, frame_idx),
        )
        _set_joint_group(
            q,
            ik,
            LEFT_ARM_JOINTS,
            _episode_joint_frame(f, "obs/joint/left_arm", LEFT_ARM_JOINTS, frame_idx),
        )
        _set_joint_group(
            q,
            ik,
            RIGHT_ARM_JOINTS,
            _episode_joint_frame(f, "obs/joint/right_arm", RIGHT_ARM_JOINTS, frame_idx),
        )
        _set_joint_group(
            q,
            ik,
            HEAD_JOINTS,
            _episode_joint_frame(f, "obs/joint/head", HEAD_JOINTS, frame_idx),
        )
        if "obs/base/pose" in f:
            base = np.asarray(f["obs/base/pose"], dtype=float)
            if base.ndim != 2 or base.shape[1] != 3:
                raise ValueError(
                    "reference episode 'obs/base/pose' must have shape (N,3) "
                    f"got {base.shape}"
                )
            idx = _normalize_frame_idx(base.shape[0], frame_idx, "obs/base/pose")
            x, y, yaw = base[idx]
            if not np.all(np.isfinite([x, y, yaw])):
                raise ValueError(f"reference episode 'obs/base/pose'[{idx}] is non-finite")
            q[0] = float(x)
            q[1] = float(y)
            q[2] = float(np.cos(yaw))
            q[3] = float(np.sin(yaw))

    left = _to_mat(ik.frame_pose(LEFT_EE_FRAME, q))
    right = _to_mat(ik.frame_pose(RIGHT_EE_FRAME, q))
    if own_ik:
        del ik
    return left, right


def pose_rotation_error_deg(reference: np.ndarray, current: np.ndarray) -> float:
    """Geodesic orientation error from ``reference`` to ``current`` in degrees."""
    delta = Rotation.from_matrix(reference[:3, :3]).inv() * Rotation.from_matrix(
        current[:3, :3]
    )
    return float(np.rad2deg(delta.magnitude()))


class ReferenceAlignmentGate:
    """Stable-window gate comparing live hand targets against reference EEF poses."""

    def __init__(self, left_reference: np.ndarray, right_reference: np.ndarray) -> None:
        self.left_reference = np.asarray(left_reference, dtype=float)
        self.right_reference = np.asarray(right_reference, dtype=float)
        self._stable_since: Optional[float] = None

    def reset(self) -> None:
        """Forget any accumulated stable-window time."""
        self._stable_since = None

    def status(
        self,
        left_target: Optional[np.ndarray],
        right_target: Optional[np.ndarray],
        *,
        now: float,
    ) -> Optional[HandAlignmentStatus]:
        """Current hand target error against the loaded reference episode pose."""
        if left_target is None or right_target is None:
            self._stable_since = None
            return None

        left_delta_mm = (left_target[:3, 3] - self.left_reference[:3, 3]) * 1000.0
        right_delta_mm = (right_target[:3, 3] - self.right_reference[:3, 3]) * 1000.0
        left_rot_deg = pose_rotation_error_deg(self.left_reference, left_target)
        right_rot_deg = pose_rotation_error_deg(self.right_reference, right_target)

        probe = HandAlignmentStatus(
            left_delta_mm=left_delta_mm,
            right_delta_mm=right_delta_mm,
            left_rot_deg=left_rot_deg,
            right_rot_deg=right_rot_deg,
            pos_tolerance_mm=REFERENCE_ALIGN_POS_TOL_MM,
            rot_tolerance_deg=REFERENCE_ALIGN_ROT_TOL_DEG,
            stable_s=0.0,
            stable_required_s=REFERENCE_ALIGN_STABLE_S,
        )
        if probe.left_ok and probe.right_ok:
            if self._stable_since is None:
                self._stable_since = now
            stable_s = max(0.0, now - self._stable_since)
        else:
            self._stable_since = None
            stable_s = 0.0
        return HandAlignmentStatus(
            left_delta_mm=left_delta_mm,
            right_delta_mm=right_delta_mm,
            left_rot_deg=left_rot_deg,
            right_rot_deg=right_rot_deg,
            pos_tolerance_mm=REFERENCE_ALIGN_POS_TOL_MM,
            rot_tolerance_deg=REFERENCE_ALIGN_ROT_TOL_DEG,
            stable_s=stable_s,
            stable_required_s=REFERENCE_ALIGN_STABLE_S,
        )
