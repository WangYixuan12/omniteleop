"""Synthetic checks for the swerve FK odometry (no hardware needed).

Run:  python3 -m pytest SLAM/tests/test_swerve_odometry.py -q
(or inside the SLAM container: pytest /slam/tests/ -q)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bridge"))

from swerve_odometry import (  # noqa: E402
    DRIVE_SIGN,
    WHEEL_POS,
    WHEEL_RADIUS,
    SwerveOdometry,
    contact_speeds,
    integrate_se2,
    twist_from_wheels,
)


def wheel_speeds_for_twist(twist, steer_pos):
    """Inverse model: contact speeds consistent with a body twist (for tests)."""
    vx, vy, w = twist
    theta = -np.asarray(steer_pos)
    speeds = []
    for i in range(2):
        px, py = WHEEL_POS[i]
        v = np.array([vx - w * py, vy + w * px])
        d = np.array([np.cos(theta[i]), np.sin(theta[i])])
        speeds.append(v @ d)
    return np.array(speeds)


def test_pure_forward():
    twist = twist_from_wheels(steer_pos=[0.0, 0.0], speeds=[0.3, 0.3])
    np.testing.assert_allclose(twist, [0.3, 0.0, 0.0], atol=1e-9)


def test_pure_strafe():
    # heading +pi/2 (wheels pointing +y) requires steer = -pi/2 (axis -z)
    twist = twist_from_wheels(steer_pos=[-np.pi / 2, -np.pi / 2], speeds=[0.2, 0.2])
    np.testing.assert_allclose(twist, [0.0, 0.2, 0.0], atol=1e-9)


def test_pure_spin():
    # consistent wheel states for twist (0, 0, 0.5) via the inverse model
    w = 0.5
    for i in range(2):
        px, py = WHEEL_POS[i]
        v = np.array([-w * py, w * px])
        heading = np.arctan2(v[1], v[0])
        speed = np.linalg.norm(v)
        if i == 0:
            steer = [-heading]
            speeds = [speed]
        else:
            steer.append(-heading)
            speeds.append(speed)
    twist = twist_from_wheels(steer, speeds)
    np.testing.assert_allclose(twist, [0.0, 0.0, w], atol=1e-9)


@pytest.mark.parametrize("true_twist", [
    (0.4, 0.0, 0.0),
    (0.0, -0.25, 0.0),
    (0.3, 0.1, 0.4),
    (-0.2, 0.0, -0.6),
])
def test_roundtrip_inverse_forward(true_twist):
    # arbitrary steer config consistent with the twist: align wheels with the
    # local contact velocity direction so no-slip rows hold exactly
    vx, vy, w = true_twist
    steer, speeds = [], []
    for i in range(2):
        px, py = WHEEL_POS[i]
        v = np.array([vx - w * py, vy + w * px])
        steer.append(-np.arctan2(v[1], v[0]))
        speeds.append(np.linalg.norm(v))
    twist = twist_from_wheels(steer, speeds)
    np.testing.assert_allclose(twist, true_twist, atol=1e-9)


def test_contact_speeds_modes():
    np.testing.assert_allclose(contact_speeds([0.2, 0.2], "ms"), [0.2, 0.2])
    rad = np.array([1.0, -1.0])
    np.testing.assert_allclose(
        contact_speeds(rad, "rad"), DRIVE_SIGN * rad * WHEEL_RADIUS)
    with pytest.raises(ValueError):
        contact_speeds([0.1, 0.2, 0.3], "ms")
    with pytest.raises(ValueError):
        contact_speeds([0.1, 0.2], "bogus")


def test_integrate_straight_and_arc():
    pose = integrate_se2(np.zeros(3), [1.0, 0.0, 0.0], 2.0)
    np.testing.assert_allclose(pose, [2.0, 0.0, 0.0], atol=1e-12)
    # quarter circle: vx=v, w=v/r -> ends at (r, r) heading +90deg
    v, r = 0.5, 2.0
    pose = integrate_se2(np.zeros(3), [v, 0.0, v / r], (np.pi / 2) * r / v)
    np.testing.assert_allclose(pose, [r, r, np.pi / 2], atol=1e-9)


def test_full_circle_closure():
    odo = SwerveOdometry(drive_state_mode="ms")
    w = 0.8
    steer, speeds = [], []
    for i in range(2):
        px, py = WHEEL_POS[i]
        v = np.array([-w * py, w * px])
        steer.append(-np.arctan2(v[1], v[0]))
        speeds.append(np.linalg.norm(v))
    n, total_t = 2000, 2 * np.pi / w
    for _ in range(n):
        odo.update(steer, speeds, total_t / n)
    np.testing.assert_allclose(odo.pose[:2], [0.0, 0.0], atol=1e-6)
    assert abs(np.arctan2(np.sin(odo.pose[2]), np.cos(odo.pose[2]))) < 1e-6
