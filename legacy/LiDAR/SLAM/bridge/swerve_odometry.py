"""Swerve-drive forward-kinematics odometry for the Vega-1 chassis.

Pure numpy, no ROS imports (unit-tested by SLAM/tests/test_swerve_odometry.py).

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

    def __init__(self, drive_state_mode: str = "ms", lateral_weight: float = 0.5):
        if drive_state_mode not in ("ms", "rad"):
            raise ValueError(f"bad drive_state_mode {drive_state_mode!r}")
        self.mode = drive_state_mode
        self.lateral_weight = float(lateral_weight)
        self.pose = np.zeros(3)
        self.twist = np.zeros(3)

    def update(self, steer_pos, drive_vel, dt: float) -> np.ndarray:
        speeds = contact_speeds(drive_vel, self.mode)
        self.twist = twist_from_wheels(steer_pos, speeds, self.lateral_weight)
        self.pose = integrate_se2(self.pose, self.twist, dt)
        return self.pose
