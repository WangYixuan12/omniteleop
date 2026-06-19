"""Swerve-drive forward-kinematics odometry for the Vega-1 chassis.

Pure numpy, no ROS imports. The sign convention is pinned by
tests/test_drive_box_metrics.py::test_swerve_odometry_sign_convention.

Geometry (from dexmate_urdf vega_1.urdf, matches omniteleop wbc_swerve.py):
  - two steered wheel modules at L=(0.185, 0.225), R=(0.185, -0.225) [m, base frame]
  - steer joints rotate about -z  =>  wheel heading in base frame = -steer_angle
  - drive joints: L about +y, R about -y  =>  positive joint speed drives the
    robot toward +x with sign DRIVE_SIGN = [+1, -1]
  - wheel radius 0.0861 m (calibrated value from wbc_swerve.py)

Twist estimation: weighted least squares over 4 constraint rows (2 per wheel).
For wheel i at p=(px,py) with heading theta, the ground-contact velocity of a
rigid body with twist (vx, vy, w) is v_i = (vx - w*py, vy + w*px).
  rolling row : v_i . (cos t, sin t)  = s_i   (measured contact speed)
  lateral row : v_i . (-sin t, cos t) = 0     (no-slip, down-weighted: violated
                                               during steering transients)
"""

from __future__ import annotations

import numpy as np

WHEEL_POS = np.array([[0.185, 0.225], [0.185, -0.225]])  # [L, R]
WHEEL_RADIUS = 0.0861
DRIVE_SIGN = np.array([1.0, -1.0])  # [L, R]

# Robot operating limits (dexbot_utils Vega1ChassisConfig)
VEGA1_MAX_STEERING_ANGLE = 2.35  # rad
VEGA1_MAX_LINEAR_VEL = 0.8       # m/s (per-wheel contact speed limit)


class OdometryInputError(ValueError):
    """Raised when a wheel sample is non-finite or out of sane range.

    Odometry is stateful, so integrating a NaN/inf (or absurd) sample would
    permanently poison the pose. The integrator refuses such samples instead of
    silently corrupting the estimate; callers decide policy (e.g. log + skip,
    holding the last good pose)."""


def _check_sample(name: str, arr: np.ndarray, bound: float) -> None:
    """Raise ``OdometryInputError`` if ``arr`` is non-finite or exceeds ``bound``."""
    if not np.all(np.isfinite(arr)):
        raise OdometryInputError(f"{name} is not finite: {arr.tolist()}")
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > bound:
        raise OdometryInputError(
            f"{name} exceeds bound |{bound:g}|: {arr.tolist()}")


def contact_speeds(drive_vel: np.ndarray, mode: str) -> np.ndarray:
    """Signed ground-contact speeds [L, R] in m/s from state/chassis/drive vel.

    mode 'ms' : vel is already a signed contact speed in m/s.
    mode 'rad': vel is the raw joint rate in rad/s (needs sign + radius).
    """
    drive_vel = np.asarray(drive_vel, dtype=np.float64)
    if drive_vel.shape != (2,):
        raise ValueError(f"drive_vel must have shape (2,), got {drive_vel.shape}")
    if mode == "ms":
        return drive_vel
    if mode == "rad":
        return DRIVE_SIGN * drive_vel * WHEEL_RADIUS
    raise ValueError(f"unknown drive_state_mode {mode!r} (expected 'ms' or 'rad')")


def twist_from_wheels(
    steer_pos: np.ndarray,
    speeds: np.ndarray,
    lateral_weight: float = 0.5,
) -> np.ndarray:
    """Estimate body twist (vx, vy, w) from steer angles and contact speeds."""
    steer_pos = np.asarray(steer_pos, dtype=np.float64)
    speeds = np.asarray(speeds, dtype=np.float64)
    if steer_pos.shape != (2,) or speeds.shape != (2,):
        raise ValueError(
            f"expected shapes (2,), got steer {steer_pos.shape}, speeds {speeds.shape}")

    theta = -steer_pos  # steer axis is -z
    c, s = np.cos(theta), np.sin(theta)
    px, py = WHEEL_POS[:, 0], WHEEL_POS[:, 1]

    rows = []
    rhs = []
    weights = []
    for i in range(2):
        rows.append([c[i], s[i], px[i] * s[i] - py[i] * c[i]])
        rhs.append(speeds[i])
        weights.append(1.0)
        rows.append([-s[i], c[i], px[i] * c[i] + py[i] * s[i]])
        rhs.append(0.0)
        weights.append(lateral_weight)

    A = np.asarray(rows) * np.asarray(weights)[:, None]
    b = np.asarray(rhs) * np.asarray(weights)
    twist, *_ = np.linalg.lstsq(A, b, rcond=None)
    return twist


def integrate_se2(pose: np.ndarray, twist: np.ndarray, dt: float) -> np.ndarray:
    """Exact constant-twist SE(2) integration of pose (x, y, yaw)."""
    pose = np.asarray(pose, dtype=np.float64)
    twist = np.asarray(twist, dtype=np.float64)
    if pose.shape != (3,) or twist.shape != (3,):
        raise ValueError(f"expected shapes (3,), got pose {pose.shape}, twist {twist.shape}")
    if dt < 0:
        raise ValueError(f"dt must be >= 0, got {dt}")

    vx, vy, w = twist
    x, y, yaw = pose
    if abs(w) < 1e-9:
        dx_l = vx * dt
        dy_l = vy * dt
        dyaw = 0.0
    else:
        wt = w * dt
        sin_wt, cos_wt = np.sin(wt), np.cos(wt)
        dx_l = (vx * sin_wt + vy * (cos_wt - 1.0)) / w
        dy_l = (vx * (1.0 - cos_wt) + vy * sin_wt) / w
        dyaw = wt
    cy, sy = np.cos(yaw), np.sin(yaw)
    new = np.array([
        x + cy * dx_l - sy * dy_l,
        y + sy * dx_l + cy * dy_l,
        np.arctan2(np.sin(yaw + dyaw), np.cos(yaw + dyaw)),
    ])
    return new


class SwerveOdometry:
    """Stateful odometry integrator: feed wheel states, read pose + twist."""

    def __init__(
        self,
        drive_state_mode: str = "ms",
        lateral_weight: float = 0.5,
        max_dt: float = 0.1,
        max_steering_angle: float = VEGA1_MAX_STEERING_ANGLE,
        max_linear_vel: float = VEGA1_MAX_LINEAR_VEL,
        sanity_margin: float = 2.0,
    ):
        if drive_state_mode not in ("ms", "rad"):
            raise ValueError(f"bad drive_state_mode {drive_state_mode!r}")
        if max_dt <= 0.0:
            raise ValueError(f"max_dt must be > 0, got {max_dt}")
        if sanity_margin < 1.0:
            raise ValueError(f"sanity_margin must be >= 1, got {sanity_margin}")
        self.mode = drive_state_mode
        self.lateral_weight = float(lateral_weight)
        self.max_dt = float(max_dt)
        # Validity bounds = robot operating limit x margin, in each input's units.
        # Drive is a contact speed (m/s) in 'ms' mode but a wheel rate (rad/s) in
        # 'rad' mode, so convert the speed bound by the wheel radius.
        self.max_steer_abs = float(sanity_margin) * float(max_steering_angle)
        contact_bound = float(sanity_margin) * float(max_linear_vel)
        self.max_drive_abs = (
            contact_bound if self.mode == "ms" else contact_bound / WHEEL_RADIUS
        )
        self.pose = np.zeros(3)
        self.twist = np.zeros(3)
        self._prev_ts_ns: int | None = None  # timestamp of last integrated sample

    def update(self, steer_pos, drive_vel, dt: float) -> np.ndarray:
        """Integrate one step over an explicit ``dt`` (seconds). Low-level: the
        caller owns ``dt``. Prefer ``update_at`` when a measured sample timestamp
        is available.

        Raises ``OdometryInputError`` if ``steer_pos``/``drive_vel``/``dt`` are
        non-finite or out of sane range; state is left untouched so a rejected
        sample cannot poison the pose."""
        steer_pos = np.asarray(steer_pos, dtype=np.float64)
        drive_vel = np.asarray(drive_vel, dtype=np.float64)
        _check_sample("steer_pos", steer_pos, self.max_steer_abs)
        _check_sample("drive_vel", drive_vel, self.max_drive_abs)
        if not np.isfinite(dt):
            raise OdometryInputError(f"dt is not finite: {dt}")
        speeds = contact_speeds(drive_vel, self.mode)
        self.twist = twist_from_wheels(steer_pos, speeds, self.lateral_weight)
        self.pose = integrate_se2(self.pose, self.twist, dt)
        return self.pose

    def update_at(self, steer_pos, drive_vel, t_ns: int) -> np.ndarray:
        """Timestamp-driven integration from a measured sample timestamp.

        ``t_ns`` is the firmware/state timestamp (nanoseconds) of this wheel
        sample. ``dt`` is derived from the previous integrated sample's timestamp
        -- not the caller's loop period -- so the integral is independent of
        polling jitter and robust to dropped/stale samples:

          * first sample (or just after construction): the timestamp is latched
            and nothing is integrated (``dt`` is unknown); pose is returned as-is.
          * non-advancing sample (``t_ns <=`` last integrated -- e.g. a cached
            ``get_latest()`` value polled faster than it refreshes): skipped, so a
            stalled stream does NOT dead-reckon a stale velocity forward.
          * advancing sample: ``dt = (t_ns - prev) * 1e-9``, clamped to ``max_dt``
            to bound the jump across a dropout. Non-finite/out-of-range data
            raises ``OdometryInputError`` (via ``update``) without advancing the
            timestamp, so a later good sample integrates across the gap.

        Returns the current pose for the latched/stale cases.
        """
        t_ns = int(t_ns)
        if self._prev_ts_ns is not None and t_ns > self._prev_ts_ns:
            dt = min((t_ns - self._prev_ts_ns) * 1e-9, self.max_dt)
            # update() validates and raises before mutating any state, so on an
            # invalid sample _prev_ts_ns stays put and the next good sample
            # integrates across the gap (clamped to max_dt).
            pose = self.update(steer_pos, drive_vel, dt)
            self._prev_ts_ns = t_ns
            return pose
        if self._prev_ts_ns is None:
            self._prev_ts_ns = t_ns
        return self.pose
