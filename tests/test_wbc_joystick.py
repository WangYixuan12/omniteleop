# ruff: noqa: SLF001
"""Tests for the joystick mobile-manipulation teleop scripts + their shared primitives.

Covers the three new pieces: the ``lock_base_in_ik`` IK mode, the shared
:class:`JoystickBaseShaper`, and the new leader/follower scripts' geometry + config
overrides. Hardware/SAPIEN-free (the SAPIEN integration is exercised separately); run
under the dexmate env with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from omniteleop.follower.base_closed_loop import integrate_se2
from omniteleop.follower.joystick_base import JoystickBaseShaper
from omniteleop.follower.whole_body_ik import (
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    VegaWholeBodyIK,
    WBCConfig,
)
from omniteleop.wbc_teleop import JoystickTeleopConfig, VRTeleopConfig


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_joytest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _shaper_args(vt: VRTeleopConfig) -> SimpleNamespace:
    joystick = JoystickTeleopConfig.from_yaml()
    return SimpleNamespace(
        base_kp_xy=vt.base_kp_xy, base_kp_yaw=vt.base_kp_yaw, base_max_speed=vt.base_max_speed,
        base_deadband=vt.base_deadband, base_accel=vt.base_accel,
        base_post_linear_deadband=vt.base_post_linear_deadband,
        base_post_angular_deadband=vt.base_post_angular_deadband,
        base_yaw_hold_in_xy=vt.base_yaw_hold_in_xy,
        joystick_stick_max_vx=vt.stick_max_vx,
        joystick_stick_max_wz=vt.stick_max_wz,
        joystick_single_axis_hysteresis_ratio=joystick.single_axis_hysteresis_ratio,
    )


# --- lock_base_in_ik ----------------------------------------------------------------------

def test_lock_base_in_ik_freezes_base_but_arm_tracks():
    cfg = WBCConfig(lock_base_in_ik=True, head_mode="track")
    ik = VegaWholeBodyIK(cfg)
    ik.reset()
    left = np.asarray(ik.frame_pose(LEFT_EE_FRAME).homogeneous)
    right = np.asarray(ik.frame_pose(RIGHT_EE_FRAME).homogeneous)
    left = left.copy()
    left[0, 3] += 0.04  # reachable 4 cm forward
    res = None
    max_base_twist = 0.0
    for _ in range(200):
        res = ik.solve(left, right, 0.01)
        max_base_twist = max(max_base_twist, float(np.max(np.abs(res.base_twist))))
    assert max_base_twist == 0.0                      # base never moves in the QP
    assert np.allclose(res.base_pose, 0.0)            # base pose pinned at origin
    assert res.left_ee_error < 0.004                  # torso+arms still reach the target
    assert res.success and not res.held


def test_lock_base_in_ik_does_not_change_default_or_xy_modes():
    # Default (base free) still lets the base move to help reach a far target.
    cfg = WBCConfig(head_mode="track")
    assert not cfg.lock_base_in_ik
    ik = VegaWholeBodyIK(cfg)
    ik.reset()
    left = np.asarray(ik.frame_pose(LEFT_EE_FRAME).homogeneous).copy()
    right = np.asarray(ik.frame_pose(RIGHT_EE_FRAME).homogeneous)
    left[0, 3] += 0.30  # far target -> base assists
    for _ in range(200):
        res = ik.solve(left, right, 0.01)
    assert abs(res.base_pose[0]) > 0.01
    # "xy" still pins only yaw when the base is not locked.
    ik_xy = VegaWholeBodyIK(WBCConfig(head_mode="track", base_dofs="xy"))
    assert ik_xy._base_dof_mask.tolist() == [True, True, False]
    # lock overrides base_dofs entirely.
    ik_lock = VegaWholeBodyIK(WBCConfig(head_mode="track", base_dofs="xy_yaw", lock_base_in_ik=True))
    assert ik_lock._base_dof_mask.tolist() == [False, False, False]


# --- JoystickBaseShaper -------------------------------------------------------------------

def _drive(cfg, joy, ticks=300, blocked=False):
    vt = VRTeleopConfig.from_yaml()
    sh = JoystickBaseShaper.from_configs(cfg, _shaper_args(vt))
    sh.reset(np.zeros(3))
    meas = np.zeros(3)
    cmds = []
    for _ in range(ticks):
        cmd, _, _ = sh.step(np.asarray(joy, dtype=float), meas, 0.01)
        cmds.append(cmd.copy())
        if not blocked:
            meas = integrate_se2(meas, cmd, 0.01)
    return np.array(cmds), sh


def test_shaper_drives_forward():
    cmds, _ = _drive(WBCConfig(), [0.3, 0.0, 0.0])
    assert cmds[-1][0] > 0.2 and abs(cmds[-1][1]) < 1e-9 and abs(cmds[-1][2]) < 1e-9


def test_shaper_command_speed_capped_when_blocked():
    # The reference itself is unbounded when blocked (no lead clamp); the emitted wheel
    # command must still respect base_max_speed via pd_twist/shape_twist's limit_twist.
    vt = VRTeleopConfig.from_yaml()
    cmds, _ = _drive(WBCConfig(), [0.3, 0.0, 0.0], ticks=400, blocked=True)
    assert float(np.max(np.abs(cmds))) <= vt.base_max_speed + 1e-6


def test_shaper_single_axis_projection():
    cmds, _ = _drive(WBCConfig(), [0.3, 0.0, 0.4])  # drive + turn -> one axis
    assert int(np.sum(np.abs(cmds[-1]) > 1e-6)) == 1


def test_shaper_strafe_intent_wins_over_stale_forward_reference():
    cfg = WBCConfig()
    vt = VRTeleopConfig.from_yaml()
    sh = JoystickBaseShaper.from_configs(cfg, _shaper_args(vt))
    sh.reset(np.zeros(3))
    sh.cmd_pose = np.array([0.20, 0.0, 0.0])  # residual lead from a previous forward segment

    cmd = np.zeros(3)
    for _ in range(20):
        cmd, _, _ = sh.step(np.array([0.12, 0.18, 0.0]), np.zeros(3), 0.01)

    assert abs(cmd[0]) < 1e-9
    assert cmd[1] > 0.0
    assert sh.prev_axis == 1


def test_shaper_strafe_intent_does_not_retain_forward_hysteresis():
    cfg = WBCConfig()
    vt = VRTeleopConfig.from_yaml()
    sh = JoystickBaseShaper.from_configs(cfg, _shaper_args(vt))
    sh.reset(np.zeros(3))
    sh.prev_axis = 0
    sh.prev_shaped = np.array([0.08, 0.08, 0.0])

    cmd, _, _ = sh.step(np.array([0.16, 0.1688, 0.0]), np.zeros(3), 0.01)

    assert abs(cmd[0]) < 1e-9
    assert cmd[1] > 0.0
    assert sh.prev_axis == 1


def test_shaper_does_not_accumulate_off_axis_reference_debt():
    cfg = WBCConfig()
    vt = VRTeleopConfig.from_yaml()
    sh = JoystickBaseShaper.from_configs(cfg, _shaper_args(vt))
    sh.reset(np.zeros(3))
    measured = np.zeros(3)

    # Matches the stable diagonal command recorded during the intended strafes in
    # wbc_joystick_dbg_20260707_090344.hdf5. The dominant joystick intent is vy.
    for _ in range(200):
        cmd, _, _ = sh.step(np.array([0.13, 0.17, 0.0]), measured, 0.01)
        measured = integrate_se2(measured, cmd, 0.01)

    assert abs(sh.cmd_pose[0]) < 1e-9
    assert sh.cmd_pose[1] > 0.0

    release_cmds = []
    for _ in range(200):
        cmd, _, _ = sh.step(np.zeros(3), measured, 0.01)
        release_cmds.append(cmd.copy())
        measured = integrate_se2(measured, cmd, 0.01)

    assert float(np.max(np.abs(np.asarray(release_cmds)[:, 0]))) < 1e-9


def test_shaper_xy_drops_deliberate_yaw_keeps_translation():
    cfg = replace(WBCConfig(), base_dofs="xy")
    cmds_yaw, _ = _drive(cfg, [0.0, 0.0, 0.5])
    assert abs(cmds_yaw[-1][2]) < 1e-6                 # deliberate yaw dropped
    cmds_strafe, _ = _drive(cfg, [0.3, 0.2, 0.0])
    assert abs(cmds_strafe[-1][2]) < 1e-6              # strafe fine, no yaw
    assert float(np.hypot(cmds_strafe[-1][0], cmds_strafe[-1][1])) > 0.05


def test_shaper_xy_allows_heading_hold_correction():
    # In "xy" with yaw-hold, a base yawed off its reference gets a corrective wz even
    # though the deliberate yaw stick is zero.
    cfg = replace(WBCConfig(), base_dofs="xy")
    vt = VRTeleopConfig.from_yaml()
    assert vt.base_yaw_hold_in_xy  # precondition from wbik.yaml
    sh = JoystickBaseShaper.from_configs(cfg, _shaper_args(vt))
    sh.reset(np.zeros(3))
    measured = np.array([0.0, 0.0, 0.3])  # base physically rotated +0.3 rad off reference
    # Hold the drift and let the slew-limited correction ramp past the single-axis deadband.
    cmd = np.zeros(3)
    for _ in range(100):
        cmd, _, _ = sh.step(np.array([0.0, 0.0, 0.0]), measured, 0.01)
    assert cmd[2] < -1e-3  # sustained corrective yaw back toward heading 0
    assert abs(cmd[0]) < 1e-9 and abs(cmd[1]) < 1e-9  # translation axes untouched


# --- leader geometry + stick mapping ------------------------------------------------------

def _leader_stub(calib=None):
    m = _load_script("wbc_joystick_leader")
    L = object.__new__(m.JoystickVRLeader)
    L.robot_base_t_vr_base_eef = np.eye(4) if calib is None else calib
    L._hf_offset_left = np.zeros(3)
    L._hf_offset_right = np.zeros(3)
    L._hf_engage_t = None
    L._hf_blend_dur = 0.5
    L._ee_offset_left = np.eye(3)
    L._ee_offset_right = np.eye(3)
    L.last_left_target = None
    L.last_right_target = None
    vt = VRTeleopConfig.from_yaml()
    joystick_cfg = JoystickTeleopConfig.from_yaml()
    L.stick_max_vx = vt.stick_max_vx
    L.stick_max_vy = vt.stick_max_vy
    L.stick_max_wz = vt.stick_max_wz
    L.stick_deadzone = vt.stick_deadzone
    cfg = WBCConfig()
    L._joystick_base_dofs = cfg.base_dofs
    L._joystick_single_axis = cfg.enable_base_single_axis
    L._joystick_xy_max_vel = vt.stick_max_vx
    L._joystick_yaw_max_vel = vt.stick_max_wz
    L._joystick_axis_deadband = cfg.base_single_axis_deadband
    L._joystick_axis_hysteresis = joystick_cfg.single_axis_hysteresis_ratio
    L._joystick_axis = None
    return m, L


def test_leader_hand_position_is_head_independent():
    _m, L = _leader_stub()
    vr = np.eye(4)
    vr[:3, 3] = [0.4, 0.2, 0.1]
    vr[:3, :3] = Rotation.from_euler("z", 0.3).as_matrix()
    head_a = np.eye(4)
    head_b = np.eye(4)
    head_b[:3, :3] = Rotation.from_euler("z", 1.2).as_matrix()
    pa = L._bprime_pos("left", vr, head_a, head_a)
    pb = L._bprime_pos("left", vr, head_b, head_b)
    assert np.allclose(pa, pb)               # a head turn does NOT move the hand
    assert np.allclose(pa, [0.4, 0.2, 0.1])


def test_leader_hand_rides_base_calibration_yaw():
    calib = np.eye(4)
    calib[:3, :3] = Rotation.from_euler("z", np.pi / 2).as_matrix()
    _m, L = _leader_stub(calib)
    vr = np.eye(4)
    vr[:3, 3] = [0.4, 0.2, 0.1]
    assert np.allclose(L._bprime_pos("left", vr), [-0.2, 0.4, 0.1], atol=1e-9)


def test_leader_eef_orientation_is_absolute_controller_frame():
    _m, L = _leader_stub()
    vr = np.eye(4)
    vr[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.3, 0.5]).as_matrix()
    T = L._eef_target("left", vr, now=0.0)
    assert np.allclose(T[:3, :3], vr[:3, :3])  # identity calib + identity C


def test_leader_thumbstick_right_drives_left_turns():
    _m, L = _leader_stub()

    def stick(left, right):
        return {"left_thumbstick": np.array(left), "right_thumbstick": np.array(right)}

    vx, vy, wz = L._thumbstick_to_chassis(stick([0, 0], [0, -1]))  # right up -> forward
    assert vx > 0 and abs(vy) < 1e-9 and abs(wz) < 1e-9
    vx, vy, wz = L._thumbstick_to_chassis(stick([0, 0], [1, 0]))   # right right -> strafe
    assert abs(vx) < 1e-9 and vy < 0 and abs(wz) < 1e-9
    vx, vy, wz = L._thumbstick_to_chassis(stick([1, 0], [0, 0]))   # left right -> yaw
    assert abs(vx) < 1e-9 and abs(vy) < 1e-9 and wz < 0
    vx, vy, wz = L._thumbstick_to_chassis(stick([0, 1], [0, 0]))   # left Y unused
    assert abs(vx) < 1e-9 and abs(vy) < 1e-9 and abs(wz) < 1e-9


def test_leader_publishes_diagonal_strafe_as_single_axis():
    _m, L = _leader_stub()
    transforms = {
        "left_thumbstick": np.zeros(2),
        "right_thumbstick": np.array([0.85, -0.48]),
    }

    vx, vy, wz = L._thumbstick_to_chassis(transforms)

    assert abs(vx) < 1e-9
    assert vy < 0.0
    assert abs(wz) < 1e-9
    assert L._joystick_axis == 1


# --- real follower config + base-drive plumbing -------------------------------------------

def test_real_follower_config_overrides():
    m = _load_script("wbc_joystick_robot")
    ik, cfg = m._build_joystick_ik(None)
    assert cfg.lock_base_in_ik and cfg.head_mode == "track"
    assert cfg.base_dofs in ("xy", "xy_yaw")  # read from wbik.yaml
    del ik

    ik, cfg = m._build_joystick_ik(None, lock_torso_in_ik=False)
    assert not cfg.lock_torso_in_ik
    del ik


def test_real_follower_drive_base_and_hold():
    m = _load_script("wbc_joystick_robot")
    vt = VRTeleopConfig.from_yaml()

    class Chassis:
        def __init__(self):
            self.calls = []
            self.steering_angle = np.zeros(2)

        def set_velocity(self, **k):
            self.calls.append(("vel", k))

        def set_wheel_velocity(self, v):
            self.calls.append(("wheel", v))

    class Odom:
        def snapshot(self):
            return {"pose": np.zeros(3), "age": 0.0, "steer": np.zeros(2),
                    "wvel": np.zeros(2), "drive_ts_ns": 0}

    d = object.__new__(m.JoystickHardwareDriver)
    d._odom = Odom()
    d._arkit = None
    d.robot = SimpleNamespace(chassis=Chassis())
    d.cfg = WBCConfig()
    d.args = SimpleNamespace(base_quiet_hold_s=-1.0)
    d._dbg = {}
    d._base_quiet_elapsed = 0.0
    d._prev_base_cmd = np.zeros(3)
    d._shaper = JoystickBaseShaper.from_configs(WBCConfig(), _shaper_args(vt))
    d._shaper.reset(np.zeros(3))
    d._source = SimpleNamespace(
        latest=SimpleNamespace(chassis_vx=0.3, chassis_vy=0.0, chassis_wz=0.0)
    )
    d._latched_source_sample = d._source.latest
    for _ in range(50):
        d._drive_base(None, False, {"base": True}, 0.01)
    kind, kw = d.robot.chassis.calls[-1]
    assert kind == "vel" and kw["vx"] > 0.1 and abs(kw["vy"]) < 1e-6
    assert d._dbg["base_action"] == "drive"
    d._drive_base(None, True, {"base": True}, 0.01)  # hold -> zero + park
    assert d.robot.chassis.calls[-1] == (
        "vel", {"vx": 0.0, "vy": 0.0, "wz": 0.0, "wait_time": 0.0}
    )
    assert np.allclose(d._shaper.cmd_pose, 0.0)

    # The exact diagonal strafe shape observed in the July 7 logs must reach the real
    # chassis driver as pure vy, with no hidden vx component.
    d._source.latest = SimpleNamespace(chassis_vx=0.13, chassis_vy=0.17, chassis_wz=0.0)
    d.latch_source_sample(d._source.latest)
    for _ in range(50):
        d._drive_base(None, False, {"base": True}, 0.01)
    kind, kw = d.robot.chassis.calls[-1]
    assert kind == "vel"
    assert abs(kw["vx"]) < 1e-9 and kw["vy"] > 0.0 and abs(kw["wz"]) < 1e-9
    np.testing.assert_allclose(d._dbg["rx_chassis"], [0.13, 0.17, 0.0])
    np.testing.assert_allclose(d._dbg["projected_chassis"], [0.0, 0.17, 0.0])
    assert d._dbg["joystick_axis"] == 1

    metadata = d._recording_control_metadata()
    assert metadata["control_mode"] == np.bytes_("joystick")
    assert metadata["demonstration_source"] == np.bytes_("human_teleoperation")
    assert metadata["eef_target_frame"] == np.bytes_("current_base")
    assert metadata["action_target_frame"] == np.bytes_("current_base")
    assert metadata["head_target_frame"] == np.bytes_("current_base")
    assert metadata["policy_action_schema"] == np.bytes_(
        "omniteleop_wbc_joystick_action/v1"
    )
    assert metadata["base_reference_source"] == np.bytes_("joystick_integrator")
    assert metadata["applied_chassis_group"] == np.bytes_("action/joint")
    assert metadata["measured_chassis_group"] == np.bytes_("obs/joint")
    np.testing.assert_array_equal(
        metadata["chassis_dataset_axes"],
        np.asarray([b"chassis_vx", b"chassis_vy", b"chassis_wz"]),
    )
    d.args.policy_path = "checkpoint"
    assert d._recording_control_metadata()["demonstration_source"] == np.bytes_(
        "policy_rollout"
    )


def test_real_follower_debug_hdf5_keeps_joystick_reference_chain(tmp_path):
    m = _load_script("wbc_joystick_robot")
    traj = m._TrajLog(
        str(tmp_path),
        meta={
            "control": "joystick",
            "eef_target_frame": "current_base",
            "base_reference_source": "joystick_integrator",
        },
    )
    traj.append(
        estop=False,
        success=True,
        held=False,
        left_ee_error=0.0,
        right_ee_error=0.0,
        base_pose=np.zeros(3),
        rx_chassis=np.array([0.13, 0.17, 0.0]),
        projected_chassis=np.array([0.0, 0.17, 0.0]),
        joystick_cmd_pose=np.array([0.0, 0.01, 0.0]),
        joystick_axis=1,
    )

    path = traj.flush(3)

    with h5py.File(path, "r") as f:
        g = f["debug"]
        np.testing.assert_allclose(g["rx_chassis"][0], [0.13, 0.17, 0.0])
        np.testing.assert_allclose(g["projected_chassis"][0], [0.0, 0.17, 0.0])
        np.testing.assert_allclose(g["joystick_cmd_pose"][0], [0.0, 0.01, 0.0])
        assert int(g["joystick_axis"][0]) == 1
        assert g.attrs["eef_target_frame"] == "current_base"
        assert g.attrs["base_reference_source"] == "joystick_integrator"


def test_real_follower_records_joystick_reference_as_base_pose(monkeypatch):
    from dataclasses import dataclass

    @dataclass
    class _Result:
        base_pose: np.ndarray

    m = _load_script("wbc_joystick_robot")
    captured = {}

    def spy_super(self, result, *a, **k):  # what HardwareDriver.record_tick would receive
        captured["base_pose"] = np.asarray(result.base_pose).copy()

    monkeypatch.setattr(m.HardwareDriver, "record_tick", spy_super)
    d = object.__new__(m.JoystickHardwareDriver)
    d._shaper = SimpleNamespace(cmd_pose=np.array([1.5, -0.5, 0.2]))

    # recording + not hold -> substitute the joystick reference.
    d._episode = SimpleNamespace(recording=True)
    m.JoystickHardwareDriver.record_tick(d, _Result(base_pose=np.zeros(3)), False, 0.0)
    assert np.allclose(captured["base_pose"], [1.5, -0.5, 0.2])
    # not recording -> no substitution (and no wasted per-tick allocation).
    d._episode = SimpleNamespace(recording=False)
    m.JoystickHardwareDriver.record_tick(d, _Result(base_pose=np.zeros(3)), False, 0.0)
    assert np.allclose(captured["base_pose"], [0.0, 0.0, 0.0])
    # hold -> no substitution even while recording (base recorder drops the frame).
    d._episode = SimpleNamespace(recording=True)
    m.JoystickHardwareDriver.record_tick(d, _Result(base_pose=np.zeros(3)), True, 0.0)
    assert np.allclose(captured["base_pose"], [0.0, 0.0, 0.0])


def test_real_follower_latches_one_source_sample_for_arm_and_base():
    m = _load_script("wbc_joystick_robot")
    d = object.__new__(m.JoystickHardwareDriver)
    first = SimpleNamespace(chassis_vx=0.2, chassis_vy=-0.1, chassis_wz=0.3)
    second = SimpleNamespace(chassis_vx=-0.2, chassis_vy=0.1, chassis_wz=-0.3)
    d._source = SimpleNamespace(latest=second)
    d._latched_source_sample = None

    d.latch_source_sample(first)

    np.testing.assert_allclose(d._read_joystick(), [0.2, -0.1, 0.3])


def test_real_follower_records_joystick_intent_chain_as_auxiliary_action():
    m = _load_script("wbc_joystick_robot")
    d = object.__new__(m.JoystickHardwareDriver)
    d._dbg = {
        "rx_chassis": np.array([0.2, 0.0, 0.0]),
        "projected_chassis": np.array([0.2, 0.0, 0.0]),
        "base_pd_raw": np.array([0.18, 0.0, 0.0]),
        "base_pd_err": np.array([-0.01, 0.0, 0.0]),
    }
    d._shaper = SimpleNamespace(
        cmd_pose=np.array([0.4, -0.1, 0.2]),
        prev_shaped=np.array([0.15, 0.0, 0.0]),
    )

    extra = d._recording_extra_action()

    np.testing.assert_allclose(extra["chassis"]["source_body"], [0.2, 0.0, 0.0])
    np.testing.assert_allclose(extra["chassis"]["intent_body"], [0.2, 0.0, 0.0])
    np.testing.assert_allclose(extra["chassis"]["command_pose"], [0.4, -0.1, 0.2])


def test_joystick_parser_supplies_every_inherited_driver_argument(monkeypatch):
    import sys

    m = _load_script("wbc_joystick_robot")
    captured = {}
    monkeypatch.setattr(sys, "argv", ["wbc_joystick_robot.py", "--enable", "none"])
    monkeypatch.setattr(
        m,
        "_run_joystick_ik_mode",
        lambda args, enable: captured.update(args=args, enable=enable),
    )

    m.main()

    args = captured["args"]
    assert args.arkit_base == "off"
    assert not args.streaming_recorder
    assert not args.no_head_depth
    assert not args.head_right_rgb
    assert not args.wrist_right_rgb
    assert args.record_startup_timeout > 0.0
    assert args.record_max_camera_skew_ms >= 0.0
    assert args.record_max_camera_age_ms > 0.0
    assert args.lock_torso_in_ik is None


def test_joystick_exception_path_quarantines_recording():
    m = _load_script("wbc_joystick_robot")
    calls = []
    driver = SimpleNamespace(
        abort_recording_episode=lambda reason: calls.append(reason),
    )

    m._quarantine_failed_recording(driver, RuntimeError("camera stalled"))

    assert calls == ["camera stalled"]
