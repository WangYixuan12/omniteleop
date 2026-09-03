from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from omniteleop.wbc_teleop import (
    JoystickTeleopConfig,
    VRJointSubscriber,
    VRTeleopConfig,
    bind_joystick_teleop_args,
    bind_vr_teleop_args,
)


def test_vr_joint_subscriber_keeps_command_and_receive_stamp_atomic(monkeypatch):
    import omniteleop.wbc_teleop as wbc_teleop

    sub = object.__new__(VRJointSubscriber)
    sub._latest = None
    sub._latest_sample = None
    sub._sink = None
    monkeypatch.setattr(wbc_teleop.time, "time_ns", lambda: 987_654_321)

    sub._on_msg({"timestamp_ns": 123, "estop": False})

    command, receive_ns = sub.latest_sample
    assert command is sub.latest
    assert command.timestamp_ns == 123
    assert receive_ns == 987_654_321


def _write_vr_teleop_yaml(path: Path, **overrides) -> None:
    values = {
        "replay_speed": 1.0,
        "ik_rate": 100.0,
        "cmd_rate": 10.0,
        "stick_max_vx": 0.3,
        "stick_max_vy": 0.3,
        "stick_max_wz": 0.5,
        "stick_deadzone": 0.1,
        "head_lpf_tau": 0.35,
        "head_planar_pos_deadband": 0.04,
        "head_planar_yaw_deadband": 0.30,
        "base_kp_xy": 1.0,
        "base_kp_yaw": 1.5,
        "base_yaw_hold_in_xy": True,
        "base_deadband": 0.02,
        "base_accel": 0.4,
        "base_max_speed": 0.45,
        "base_post_linear_deadband": 0.02,
        "base_post_angular_deadband": 0.12,
    }
    values.update(overrides)
    path.write_text(yaml.safe_dump({"vr_teleop": values}), encoding="utf-8")


def test_vr_teleop_config_loads_xy_yaw_hold_flag(tmp_path):
    path = tmp_path / "wbik.yaml"
    _write_vr_teleop_yaml(path, base_yaw_hold_in_xy=True)

    cfg = VRTeleopConfig.from_yaml(path)

    assert cfg.base_yaw_hold_in_xy is True


def test_vr_teleop_config_rejects_non_bool_xy_yaw_hold_flag(tmp_path):
    path = tmp_path / "wbik.yaml"
    _write_vr_teleop_yaml(path, base_yaw_hold_in_xy=1.0)

    with pytest.raises(ValueError, match=r"vr_teleop\.base_yaw_hold_in_xy"):
        VRTeleopConfig.from_yaml(path)


def _write_joystick_teleop_yaml(path: Path, **overrides) -> None:
    values = {
        "translation_speed": 0.15,
        "rotation_speed": 0.25,
        "single_axis_hysteresis_ratio": 0.0,
    }
    values.update(overrides)
    path.write_text(yaml.safe_dump({"joystick_teleop": values}), encoding="utf-8")


def test_joystick_teleop_config_loads_shared_axis_policy(tmp_path):
    path = tmp_path / "wbik.yaml"
    _write_joystick_teleop_yaml(path)

    cfg = JoystickTeleopConfig.from_yaml(path)

    assert cfg.translation_speed == 0.15
    assert cfg.rotation_speed == 0.25
    assert cfg.single_axis_hysteresis_ratio == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("translation_speed", 0.0),
        ("rotation_speed", -0.1),
        ("single_axis_hysteresis_ratio", -0.1),
    ),
)
def test_joystick_teleop_config_rejects_invalid_values(tmp_path, field, value):
    path = tmp_path / "wbik.yaml"
    _write_joystick_teleop_yaml(path, **{field: value})

    with pytest.raises(ValueError, match=rf"joystick_teleop\.{field}"):
        JoystickTeleopConfig.from_yaml(path)


@pytest.mark.parametrize(
    ("field", "value"),
    (("stick_max_vx", 0.0), ("stick_deadzone", -0.1)),
)
def test_vr_teleop_config_rejects_invalid_stick_values(tmp_path, field, value):
    path = tmp_path / "wbik.yaml"
    _write_vr_teleop_yaml(path, **{field: value})

    with pytest.raises(ValueError, match=rf"vr_teleop\.{field}"):
        VRTeleopConfig.from_yaml(path)


def test_shared_arg_binding_keeps_wbc_values_and_applies_joystick_head_override():
    vt = VRTeleopConfig.from_yaml()
    joystick = JoystickTeleopConfig.from_yaml()
    wbc_args = SimpleNamespace()
    joystick_args = SimpleNamespace()

    bind_vr_teleop_args(wbc_args, vt)
    bind_vr_teleop_args(joystick_args, vt, head_track=True)
    bind_joystick_teleop_args(joystick_args, vt, joystick)

    assert wbc_args.head_planar_pos_deadband == vt.head_planar_pos_deadband
    assert wbc_args.head_planar_yaw_deadband == vt.head_planar_yaw_deadband
    assert joystick_args.head_planar_pos_deadband == 0.0
    assert joystick_args.head_planar_yaw_deadband == 0.0
    assert vt.stick_max_vx == vt.stick_max_vy == 0.3
    assert vt.stick_max_wz == 0.5
    assert vt.stick_deadzone == 0.1
    assert joystick_args.joystick_stick_max_vx == vt.stick_max_vx
    assert joystick_args.joystick_stick_max_vy == vt.stick_max_vy
    assert joystick_args.joystick_translation_speed == 0.15
    assert joystick_args.joystick_rotation_speed == 0.25
