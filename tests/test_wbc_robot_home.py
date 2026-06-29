# ruff: noqa: SLF001

from __future__ import annotations

import types

import numpy as np
import pytest

from omniteleop import wbc_robot_home


class _FakeJointComponent:
    def __init__(self, q):
        self.q = np.asarray(q, dtype=float)
        self.commands = []

    def get_joint_pos(self):
        return self.q.copy()

    def set_joint_pos(self, q, **kwargs):
        self.q = np.asarray(q, dtype=float)
        self.commands.append((self.q.copy(), kwargs))


class _FakeChassis:
    def __init__(self):
        self.steer_commands = []

    def set_steering_angle(self, angle, wait_time=0.0):
        self.steer_commands.append((angle, wait_time))


def _fake_home_driver(*, collision_enabled: bool = False):
    comps = {
        "torso": _FakeJointComponent([0.0]),
        "left_arm": _FakeJointComponent([0.0]),
        "right_arm": _FakeJointComponent([0.0]),
        "head": _FakeJointComponent([0.0]),
    }
    driver = types.SimpleNamespace()
    driver._joint_names = {
        "torso": ["torso_j1"],
        "left_arm": ["L_arm_j1"],
        "right_arm": ["R_arm_j1"],
        "head": ["head_j1"],
    }
    driver._nominal = {
        "torso": np.array([0.11]),
        "left_arm": np.array([0.22]),
        "right_arm": np.array([-0.33]),
        "head": np.array([0.44]),
    }
    driver.enable = {"arms": True, "torso": False, "head": False, "base": True}
    driver.args = types.SimpleNamespace(home_settle=0.0, home_tol=0.1)
    driver.cfg = types.SimpleNamespace(self_collision_floor=0.01, self_collision_safe_dist=0.02)
    driver.has_chassis = True
    driver.robot = types.SimpleNamespace(chassis=_FakeChassis())
    driver.ik = types.SimpleNamespace(collision_enabled=collision_enabled)
    driver._comp = lambda grp: comps[grp]
    driver._to_hw = lambda _grp, joints: np.asarray(joints, dtype=float)
    driver._components = comps
    return driver


def test_home_to_nominal_homes_enabled_groups_and_centers_base():
    driver = _fake_home_driver(collision_enabled=False)

    wbc_robot_home.home_to_nominal(driver, arm_home_step=1.0)

    assert not driver._components["torso"].commands
    assert not driver._components["head"].commands
    np.testing.assert_allclose(driver._components["left_arm"].q, [0.22])
    np.testing.assert_allclose(driver._components["right_arm"].q, [-0.33])
    assert driver.robot.chassis.steer_commands == [(0.0, 0.0)]


def test_home_guard_aborts_and_stops_on_worsening_self_collision():
    driver = _fake_home_driver(collision_enabled=True)
    driver._components["left_arm"].q = np.array([0.0])
    driver.ik = types.SimpleNamespace(
        collision_enabled=True,
        _idx_q={"L_arm_j1": 0, "R_arm_j1": 1, "torso_j1": 2, "head_j1": 3},
        nominal_q=lambda: np.zeros(4),
        self_collision_distance=lambda q: 0.005 if q[0] > 0.5 else 0.02,
    )
    driver._read_measured_joints = lambda: {
        "torso": np.array([0.0]),
        "left_arm": np.array([0.0]),
        "right_arm": np.array([0.0]),
        "head": np.array([0.0]),
    }
    driver.stop_calls = 0
    driver.stop_all_motion = lambda: setattr(driver, "stop_calls", driver.stop_calls + 1)

    guard = wbc_robot_home._make_home_guard(driver, "left_arm")

    with pytest.raises(SystemExit, match="homing ABORTED: left_arm self-collision"):
        guard(np.array([1.0]))
    assert driver.stop_calls == 1


def test_home_guard_aborts_immediately_below_safe_distance():
    driver = _fake_home_driver(collision_enabled=True)
    driver._components["left_arm"].q = np.array([0.0])
    driver.ik = types.SimpleNamespace(
        collision_enabled=True,
        _idx_q={"L_arm_j1": 0, "R_arm_j1": 1, "torso_j1": 2, "head_j1": 3},
        nominal_q=lambda: np.zeros(4),
        self_collision_distance=lambda q: 0.015 if q[0] > 0.5 else 0.03,
    )
    driver._read_measured_joints = lambda: {
        "torso": np.array([0.0]),
        "left_arm": np.array([0.0]),
        "right_arm": np.array([0.0]),
        "head": np.array([0.0]),
    }
    driver.stop_calls = 0
    driver.stop_all_motion = lambda: setattr(driver, "stop_calls", driver.stop_calls + 1)

    guard = wbc_robot_home._make_home_guard(driver, "left_arm")

    with pytest.raises(SystemExit, match="homing ABORTED: left_arm self-collision"):
        guard(np.array([1.0]))
    assert driver.stop_calls == 1
