#!/usr/bin/env python3
"""Pure helpers shared by the WBC VR teleop followers (sim + real robot).

These are the command-stream primitives that MUST behave identically in the
SAPIEN sim follower (``scripts/wbc_vr_record.py``) and the real-robot follower
(``scripts/wbc_vr_robot.py``), so a recorded leader stream replays through the
whole-body IK the same way it was visualized in sim: per-tick lerp/slerp
interpolation of the leader's low-rate pose commands, the head-target low-pass
filter, base command postprocessing, and the flat<->4x4 pose decode.

Deliberately depends on ONLY numpy + scipy -- no dexcontrol / sapien / pinocchio
and, crucially, no ``omniteleop.follower`` package ``__init__`` (which is heavy and
pulls hardware deps). That keeps this module safe to import at top level from both
followers and from the stub-loaded interpolation test.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


def _to_mat(pose) -> np.ndarray:
    """4x4 matrix from a pin.SE3 (its ``.homogeneous``) or a 4x4 array."""
    return np.asarray(pose.homogeneous if hasattr(pose, "homogeneous") else pose, dtype=float)


def _pose_from_flat(flat) -> np.ndarray:
    """Reconstruct a 4x4 pose matrix from a flattened (16-float) row-major matrix."""
    return np.asarray(flat, dtype=float).reshape(4, 4)


def vr_to_ee_targets(vr, left_prev: np.ndarray, right_prev: np.ndarray):
    """L_ee/R_ee target poses (4x4) from the leader's published EEF poses.

    Uses the exact Cartesian targets the leader feeds to its arm IK (which track
    the controller's world motion), so there is no joint-space round-trip or FK
    convention mismatch. Holds the previous targets when the leader is not
    teleoperating (calibration stages publish empty EEF poses). The 4x4 matrices
    are accepted directly as targets by the whole-body IK solver.
    """
    if vr is None or not vr.left_ee_pose or not vr.right_ee_pose:
        return left_prev, right_prev
    return _pose_from_flat(vr.left_ee_pose), _pose_from_flat(vr.right_ee_pose)


def vr_to_head_target(vr, prev: np.ndarray) -> np.ndarray:
    """Head-frame target pose (4x4) from the leader's ``head_ee_pose``.

    The calibrated headset pose, interpolated alongside the EE targets and tracked
    by ``ik.solve_head`` from the live whole-body configuration (head_mode 'track'
    semantics). A non-empty pose is adopted, an empty one holds the previous
    command (``prev``, seeded with the nominal head pose -- the head holds nominal
    until the leader first publishes). A non-empty but malformed pose is a
    protocol violation (wrong leader): raise.
    """
    if vr is None or not vr.head_ee_pose:
        return prev
    flat = np.asarray(vr.head_ee_pose, dtype=float)
    if flat.shape != (16,) or not np.all(np.isfinite(flat)):
        raise ValueError(
            f"VRJointData.head_ee_pose must be 16 finite floats (flattened 4x4), "
            f"got {vr.head_ee_pose!r}"
        )
    return flat.reshape(4, 4)


def _blend_pose(t0: np.ndarray, t1: np.ndarray, alpha: float) -> np.ndarray:
    """Geodesic pose blend: lerp the translation, slerp the rotation."""
    if alpha >= 1.0:
        return t1.copy()
    out = np.eye(4)
    out[:3, 3] = (1.0 - alpha) * t0[:3, 3] + alpha * t1[:3, 3]
    r0 = Rotation.from_matrix(t0[:3, :3])
    delta = (r0.inv() * Rotation.from_matrix(t1[:3, :3])).as_rotvec()
    out[:3, :3] = (r0 * Rotation.from_rotvec(alpha * delta)).as_matrix()
    return out


def _wrap_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def _rotz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _heading_yaw(pose: np.ndarray) -> float:
    """Yaw of a head/optical frame's viewing direction projected onto the ground."""
    rot = np.asarray(pose, dtype=float)[:3, :3]
    fwd_xy = rot[:2, 2]
    norm = float(np.linalg.norm(fwd_xy))
    if norm > 1e-8:
        return float(np.arctan2(fwd_xy[1], fwd_xy[0]))

    right_xy = rot[:2, 0]
    norm = float(np.linalg.norm(right_xy))
    if norm <= 1e-8:
        return 0.0
    return float(np.arctan2(right_xy[1], right_xy[0]) + np.pi / 2.0)


def _set_heading_yaw(pose: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate ``pose`` about world z so its projected heading yaw becomes ``yaw``."""
    out = np.asarray(pose, dtype=float).copy()
    delta = float(_wrap_pi(float(yaw) - _heading_yaw(out)))
    out[:3, :3] = _rotz(delta) @ out[:3, :3]
    return out


def _validate_nonnegative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value}")


def deadband_planar_twist_by_reference(
    twist: np.ndarray,
    *,
    reference_twist: np.ndarray | None = None,
    linear_deadband: float = 0.0,
    angular_deadband: float = 0.0,
) -> np.ndarray:
    """Zero quiet base residue per component.

    ``vx`` and ``vy`` use the same ``linear_deadband`` independently; ``wz`` uses
    ``angular_deadband``. The optional reference lets a slewed command keep ramping
    while its raw target for that same component is active.
    """
    _validate_nonnegative("linear_deadband", linear_deadband)
    _validate_nonnegative("angular_deadband", angular_deadband)
    out = np.asarray(twist, dtype=np.float64).copy()
    if out.shape != (3,):
        raise ValueError(f"expected planar twist shape (3,), got {out.shape}")
    ref = out if reference_twist is None else np.asarray(reference_twist, dtype=np.float64)
    if ref.shape != (3,):
        raise ValueError(f"expected reference twist shape (3,), got {ref.shape}")

    if abs(float(ref[0])) < linear_deadband:
        out[0] = 0.0
    if abs(float(ref[1])) < linear_deadband:
        out[1] = 0.0
    if abs(float(ref[2])) < angular_deadband:
        out[2] = 0.0
    return out


class TargetInterpolator:
    """Per-tick lerp/slerp of the leader's low-rate pose commands.

    Generic over the number of pose streams -- here L_ee, R_ee, and the head frame
    all ride the same segments. The leader publishes commands at ``--cmd-rate``
    (default 10 Hz) while the IK runs at ``--rate`` (default 100 Hz). Each arriving
    command starts a new blend segment toward it, traversed in ``duration``:
    ``alpha = min(1, elapsed/duration)``, position lerped, orientation slerped. This
    matches the reference's interpolation (deps/rby1-wbc ``ee_targets.get_for_ik``
    with ``use_interpolation: true``, ``duration = 1/trajectory_frequency_hz``): the
    segment start is the previous raw command target, not the current interpolated
    output. ``duration <= 0`` degrades to pass-through (zero-order hold).

    The default segment duration is fixed at construction (``1/cmd-rate``). A single
    :meth:`push` may override it via ``duration=`` -- needed for slow/uneven REPLAY,
    where each segment must span the *stretched* inter-command interval
    (recorded inter-arrival dt / speed); otherwise the glide finishes in
    ``1/cmd-rate`` and then holds, turning slow replay into a staircase.
    """

    def __init__(self, duration: float, *poses0: np.ndarray) -> None:
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError(f"duration must be finite and >= 0, got {duration}")
        if not poses0:
            raise ValueError("at least one initial pose is required")
        self.duration = float(duration)
        self._n = len(poses0)
        self.reset(*poses0)

    def reset(self, *poses: np.ndarray) -> None:
        """Snap to ``poses`` with no segment in flight."""
        if len(poses) != self._n:
            raise ValueError(f"expected {self._n} poses, got {len(poses)}")
        self._start = tuple(p.copy() for p in poses)
        self._end = tuple(p.copy() for p in poses)
        self._t0 = -np.inf  # alpha clamps to 1 -> output the end poses
        self._seg_duration = self.duration

    def push(self, *poses: np.ndarray, now: float, duration: Optional[float] = None) -> None:
        """Start a new segment from the previous raw command toward a fresh command.

        ``duration`` (optional) overrides this segment's blend time; ``None`` uses the
        constructor default. Used by slow replay to stretch the glide over the scaled
        inter-command interval.
        """
        if len(poses) != self._n:
            raise ValueError(f"expected {self._n} poses, got {len(poses)}")
        if duration is not None and (not np.isfinite(duration) or duration < 0.0):
            raise ValueError(f"segment duration must be finite and >= 0, got {duration}")
        self._start = tuple(p.copy() for p in self._end)
        self._end = tuple(p.copy() for p in poses)
        self._t0 = now
        self._seg_duration = self.duration if duration is None else float(duration)

    def at(self, now: float) -> tuple[np.ndarray, ...]:
        """Blended target poses at time ``now``."""
        if self._seg_duration <= 0.0:
            return tuple(p.copy() for p in self._end)
        alpha = min(1.0, max(0.0, (now - self._t0) / self._seg_duration))
        return tuple(
            _blend_pose(s, e, alpha)
            for s, e in zip(self._start, self._end, strict=True)
        )


class HeadTargetLowPassFilter:
    """First-order low-pass filter for a pose-valued head target."""

    def __init__(self, tau: float, initial_pose: np.ndarray) -> None:
        if not np.isfinite(tau) or tau < 0.0:
            raise ValueError(f"tau must be finite and >= 0, got {tau}")
        self.tau = float(tau)
        self._state = np.asarray(initial_pose, dtype=float).copy()

    def reset(self, pose: np.ndarray) -> None:
        """Snap the filter state to ``pose`` with no transient."""
        self._state = np.asarray(pose, dtype=float).copy()

    def filter(self, pose: np.ndarray, dt: float) -> np.ndarray:
        """Return the low-passed pose for this tick."""
        target = np.asarray(pose, dtype=float)
        if self.tau == 0.0:
            self._state = target.copy()
            return target.copy()
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError(f"dt must be finite and >= 0, got {dt}")
        alpha = 1.0 - float(np.exp(-dt / self.tau)) if dt > 0.0 else 0.0
        self._state = _blend_pose(self._state, target, alpha)
        return self._state.copy()


class HeadTargetPlanarDeadbandFilter:
    """Suppress small planar head-target noise without snapping direction."""

    def __init__(
        self,
        initial_pose: np.ndarray,
        *,
        position_deadband: float,
        yaw_deadband: float,
    ) -> None:
        _validate_nonnegative("position_deadband", position_deadband)
        _validate_nonnegative("yaw_deadband", yaw_deadband)
        self.position_deadband = float(position_deadband)
        self.yaw_deadband = float(yaw_deadband)
        self.reset(initial_pose)

    def reset(self, pose: np.ndarray) -> None:
        """Use ``pose`` as the planar x/y/yaw anchor."""
        self._anchor = np.asarray(pose, dtype=float).copy()
        self._anchor_yaw = _heading_yaw(self._anchor)

    def filter(self, pose: np.ndarray) -> np.ndarray:
        """Return ``pose`` with quiet planar x/y/yaw motion held at the anchor."""
        target = np.asarray(pose, dtype=float)
        out = target.copy()
        planar_xy = target[:2, 3] - self._anchor[:2, 3]
        if float(np.linalg.norm(planar_xy)) < self.position_deadband:
            out[0, 3] = self._anchor[0, 3]
            out[1, 3] = self._anchor[1, 3]
        yaw_delta = float(_wrap_pi(_heading_yaw(target) - self._anchor_yaw))
        if abs(yaw_delta) < self.yaw_deadband:
            out = _set_heading_yaw(out, self._anchor_yaw)
        return out
