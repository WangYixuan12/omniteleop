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
import pytest
from scipy.spatial.transform import Rotation

from omniteleop.common.schemas import VRJointData
from omniteleop.follower.base_closed_loop import integrate_se2, wrap_pi
from omniteleop.follower.joystick_base import (
    JoystickBaseShaper,
    fixed_speed_planar_twist,
)
from omniteleop.follower.whole_body_ik import (
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    VegaWholeBodyIK,
    WBCConfig,
)
from omniteleop.wbc_record import ReplaySource, StreamRecorder
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
        joystick_translation_speed=joystick.translation_speed,
        joystick_rotation_speed=joystick.rotation_speed,
        joystick_single_axis_hysteresis_ratio=joystick.single_axis_hysteresis_ratio,
        joystick_reference_lead_xy=joystick.reference_lead_xy,
        joystick_reference_lead_yaw=joystick.reference_lead_yaw,
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
    assert cmds[-1][0] == pytest.approx(0.18)
    assert abs(cmds[-1][1]) < 1e-9 and abs(cmds[-1][2]) < 1e-9


@pytest.mark.parametrize("sign,held,response", [(1, False, 1.0), (-1, True, 0.7)])
def test_quarter_turn_uses_feedback_and_does_not_repeat(sign, held, response):
    args = _shaper_args(VRTeleopConfig.from_yaml())
    args.turn_90 = True
    sh = JoystickBaseShaper.from_configs(WBCConfig(), args)
    measured = np.array([0.2, -0.3, sign * np.deg2rad(170)])
    start = measured.copy()
    sh.reset(measured)
    replay_shaper = JoystickBaseShaper.from_configs(
        WBCConfig(), _shaper_args(VRTeleopConfig.from_yaml())
    )
    replay_shaper.reset(measured)
    yaw = np.array([0.0, 0.0, sign * sh.rotation_speed])
    # Engaging with an already-held stick cannot launch a macro.
    assert np.allclose(sh.step(yaw, measured, 0.01)[0], 0)
    sh.step(np.zeros(3), measured, 0.01)
    commands = []
    for i in range(1400):
        stick = yaw if held or i == 0 else np.zeros(3)
        cmd, _, _ = sh.step(stick, measured, 0.01)
        replay_cmd, _, _ = replay_shaper.step(sh.last_projected_joystick, measured, 0.01)
        if sh.turn_target_yaw is not None:
            np.testing.assert_allclose(replay_cmd, cmd, atol=1e-12)
        commands.append(cmd.copy())
        if i == 30 and not held:
            assert sh.last_projected_joystick[2] == yaw[2]  # dense policy label after release
        measured = integrate_se2(measured, response * cmd, 0.01)
    angle = float(wrap_pi(measured[2] - start[2]))
    assert abs(angle - sign * np.pi / 2) <= sh.TURN_TOLERANCE_RAD
    assert sh.turn_target_yaw is None
    np.testing.assert_allclose(measured[:2], start[:2])
    np.testing.assert_allclose(commands[-1], 0)
    assert np.max(np.abs(np.asarray(commands)[:, 2])) <= sh.rotation_speed
    assert np.any((np.abs(np.asarray(commands)[:, 2]) > 0.06)
                  & (np.abs(np.asarray(commands)[:, 2]) < 0.20))
    # Releasing AFTER completion rearms the next relative turn.
    sh.step(np.zeros(3), measured, 0.01)
    sh.step(yaw, measured, 0.01)
    assert float(wrap_pi(sh.turn_target_yaw - measured[2])) == pytest.approx(sign * np.pi / 2)


@pytest.mark.parametrize("cancel", ["hold", "translation", "opposite", "reset"])
def test_quarter_turn_cancellation_discards_target(cancel):
    args = _shaper_args(VRTeleopConfig.from_yaml())
    args.turn_90 = True
    sh = JoystickBaseShaper.from_configs(WBCConfig(), args)
    measured = np.zeros(3)
    sh.reset(measured)
    sh.step(np.zeros(3), measured, 0.01)
    sh.step(np.array([0., 0., 0.3]), measured, 0.01)
    assert sh.turn_target_yaw is not None
    if cancel == "hold":
        sh.hold(measured)
    elif cancel == "reset":
        sh.reset(measured)
    else:
        stick = [0.18, 0., 0.] if cancel == "translation" else [0., 0., -0.3]
        sh.step(np.array(stick), measured, 0.01)
    assert sh.turn_target_yaw is None
    # A held or opposite yaw cannot restart until neutral is observed.
    for _ in range(10):
        cmd, _, _ = sh.step(np.array([0., 0., 0.3]), measured, 0.01)
        assert cmd[2] == 0
        assert sh.turn_target_yaw is None


def test_quarter_turn_parks_despite_xy_drift_and_heading_noise():
    args = _shaper_args(VRTeleopConfig.from_yaml())
    args.turn_90 = True
    sh = JoystickBaseShaper.from_configs(WBCConfig(), args)
    sh.reset(np.zeros(3))
    sh.step(np.zeros(3), np.zeros(3), 0.01)
    sh.step(np.array([0., 0., 0.3]), np.zeros(3), 0.01)
    # Episode 2: translation drift during yaw must never steer the wheels to XY.
    measured = np.array([0.13, 0.07, 0.0])
    for heading in np.linspace(0, np.pi / 2 - 0.02, 650):
        measured[2] = heading
        cmd, _, _ = sh.step(np.zeros(3), measured, 0.01)
        np.testing.assert_array_equal(cmd[:2], 0)
    assert sh.turn_target_yaw is None
    previous = cmd[2]
    for i in range(2000):
        measured = np.array([0.13 + 0.004 * np.sin(i), 0.07,
                             np.pi / 2 + 0.04 * np.sin(i)])
        cmd, _, _ = sh.step(np.zeros(3), measured, 0.01)
        np.testing.assert_array_equal(cmd[:2], 0)
        assert abs(cmd[2]) <= abs(previous) + 1e-12
        assert abs(cmd[2] - previous) <= 2 * sh.accel * 0.01 + 1e-12
        previous = cmd[2]
        if i > 40:
            np.testing.assert_array_equal(cmd, 0)
        np.testing.assert_allclose(sh.cmd_pose, measured)
    cmd, _, _ = sh.step(np.array([0.18, 0., 0.]), measured, 0.01)
    assert cmd[0] > 0
    assert cmd[1] == cmd[2] == 0


def test_quarter_turn_blocked_timeout_and_configuration():
    args = _shaper_args(VRTeleopConfig.from_yaml())
    args.turn_90 = True
    with pytest.raises(ValueError, match="xy_yaw"):
        JoystickBaseShaper.from_configs(WBCConfig(base_dofs="xy"), args)
    sh = JoystickBaseShaper.from_configs(WBCConfig(), args)
    sh.step(np.zeros(3), np.zeros(3), 0.01)
    sh.step(np.array([0., 0., 0.3]), np.zeros(3), 0.01)
    with pytest.raises(RuntimeError, match="timed out"):
        for _ in range(2000):
            sh.step(np.zeros(3), np.zeros(3), 0.01)
    assert sh.turn_target_yaw is None


def test_shaper_command_speed_capped_when_blocked():
    # Pose-error feedback must not exceed the lower joystick fixed-speed envelope even
    # while the reference sits at its full lead over a blocked base.
    joystick = JoystickTeleopConfig.from_yaml()
    cmds, _ = _drive(WBCConfig(), [0.3, 0.0, 0.0], ticks=400, blocked=True)
    assert float(np.max(np.linalg.norm(cmds[:, :2], axis=1))) <= (
        joystick.translation_speed + 1e-6
    )
    yaw_cmds, _ = _drive(WBCConfig(), [0.0, 0.0, 0.5], ticks=400, blocked=True)
    assert float(np.max(np.abs(yaw_cmds[:, 2]))) <= joystick.rotation_speed + 1e-6


def test_shaper_reference_lead_is_bounded_when_base_lags():
    # Episode 12 regression: the chassis achieved ~0.11 m/s against the 0.15 m/s
    # reference and the open-loop integrator wound 0.9 m ahead, so the base kept driving
    # for seconds after every stick release. The lead over the measured pose must stay
    # bounded while the stick is held, and the post-release catch-up must be at most
    # that lead.
    joystick = JoystickTeleopConfig.from_yaml()
    vt = VRTeleopConfig.from_yaml()
    sh = JoystickBaseShaper.from_configs(WBCConfig(), _shaper_args(vt))
    sh.reset(np.zeros(3))
    meas = np.zeros(3)
    leads = []
    for _ in range(2000):  # 20 s of forward stick with the base at 70% of the command
        cmd, _, err = sh.step(np.array([0.3, 0.0, 0.0]), meas, 0.01)
        leads.append(float(np.hypot(*(sh.cmd_pose[:2] - meas[:2]))))  # lead this tick
        meas = integrate_se2(meas, 0.7 * cmd, 0.01)
    assert max(leads) <= joystick.reference_lead_xy + 1e-9
    assert leads[-1] == pytest.approx(joystick.reference_lead_xy)   # saturated, not decaying
    assert err[0] == pytest.approx(joystick.reference_lead_xy)      # body-frame error = lead
    assert cmd[0] > 0.0                                             # still driving forward

    released_from = meas.copy()
    for _ in range(600):  # release: the base may repay the lead and nothing more
        cmd, _, _ = sh.step(np.zeros(3), meas, 0.01)
        meas = integrate_se2(meas, cmd, 0.01)
    coast = float(np.hypot(*(meas[:2] - released_from[:2])))
    assert coast <= joystick.reference_lead_xy + 1e-3
    assert float(np.max(np.abs(cmd))) < 1e-6                        # and it has stopped


def test_shaper_reference_yaw_lead_is_bounded_when_base_lags():
    joystick = JoystickTeleopConfig.from_yaml()
    vt = VRTeleopConfig.from_yaml()
    sh = JoystickBaseShaper.from_configs(WBCConfig(), _shaper_args(vt))
    sh.reset(np.zeros(3))
    meas = np.zeros(3)
    for _ in range(2000):
        cmd, _, err = sh.step(np.array([0.0, 0.0, 0.5]), meas, 0.01)
        meas = integrate_se2(meas, 0.7 * cmd, 0.01)
        assert abs(float(wrap_pi(sh.cmd_pose[2] - meas[2]))) <= joystick.reference_lead_yaw + 1e-9
    assert err[2] == pytest.approx(joystick.reference_lead_yaw)
    assert cmd[2] > 0.0


def test_shaper_zero_lead_is_a_pure_velocity_command():
    vt = VRTeleopConfig.from_yaml()
    args = _shaper_args(vt)
    args.joystick_reference_lead_xy = 0.0
    args.joystick_reference_lead_yaw = 0.0
    sh = JoystickBaseShaper.from_configs(WBCConfig(), args)
    sh.reset(np.zeros(3))
    meas = np.zeros(3)
    for _ in range(300):
        cmd, pd_raw, err = sh.step(np.array([0.3, 0.0, 0.0]), meas, 0.01)
        np.testing.assert_allclose(sh.cmd_pose, meas, atol=1e-12)  # pinned to measured
        np.testing.assert_allclose(err, 0.0, atol=1e-12)
        np.testing.assert_allclose(pd_raw, [0.18, 0.0, 0.0], atol=1e-9)  # feed-forward only
        meas = integrate_se2(meas, 0.5 * cmd, 0.01)  # base far slower than commanded
    assert meas[0] > 0.1  # the base did move, so the pin was exercised


def test_shaper_rejects_negative_or_nonfinite_lead():
    vt = VRTeleopConfig.from_yaml()
    for bad in (-0.01, float("nan"), float("inf")):
        args = _shaper_args(vt)
        args.joystick_reference_lead_xy = bad
        with pytest.raises(ValueError, match="reference_lead"):
            JoystickBaseShaper.from_configs(WBCConfig(), args)


def test_shaper_rejects_nonfinite_measured_pose():
    vt = VRTeleopConfig.from_yaml()
    sh = JoystickBaseShaper.from_configs(WBCConfig(), _shaper_args(vt))
    sh.reset(np.zeros(3))
    with pytest.raises(ValueError, match="measured_pose"):
        sh.step(np.array([0.3, 0.0, 0.0]), np.array([np.nan, 0.0, 0.0]), 0.01)


def test_fixed_speed_mapping_preserves_direction_not_stick_magnitude():
    slow = fixed_speed_planar_twist(
        np.array([0.03, 0.04, -0.01]),
        translation_speed=0.15,
        rotation_speed=0.25,
    )
    fast = fixed_speed_planar_twist(
        np.array([0.3, 0.4, -0.8]),
        translation_speed=0.15,
        rotation_speed=0.25,
    )
    np.testing.assert_allclose(slow, [0.09, 0.12, -0.25])
    np.testing.assert_allclose(fast, slow)


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
    L._joystick_translation_speed = joystick_cfg.translation_speed
    L._joystick_rotation_speed = joystick_cfg.rotation_speed
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
    assert vx == pytest.approx(0.18) and abs(vy) < 1e-9 and abs(wz) < 1e-9
    vx, vy, wz = L._thumbstick_to_chassis(stick([0, 0], [1, 0]))   # right right -> strafe
    assert abs(vx) < 1e-9 and vy == pytest.approx(-0.18) and abs(wz) < 1e-9
    vx, vy, wz = L._thumbstick_to_chassis(stick([1, 0], [0, 0]))   # left right -> yaw
    assert abs(vx) < 1e-9 and abs(vy) < 1e-9 and wz == pytest.approx(-0.30)
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
    assert vy == pytest.approx(-0.18)
    assert abs(wz) < 1e-9
    assert L._joystick_axis == 1


# --- real follower config + base-drive plumbing -------------------------------------------

def test_real_follower_config_overrides():
    m = _load_script("wbc_joystick_robot")
    ik, cfg = m._build_joystick_ik(None)
    assert cfg.lock_base_in_ik
    assert cfg.head_mode == JoystickTeleopConfig.from_yaml().head_mode  # joystick block
    assert cfg.head_mode in ("track", "fixed")
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
    # Collection runs joystick_teleop.head_mode "fixed" (head pinned at nominal), and
    # the recorded head contract reads the --enable mask, so both must be present here
    # exactly as the real driver carries them.
    d.cfg = replace(WBCConfig(), head_mode="fixed")
    d.enable = {"arms": True, "torso": True, "head": True, "base": True}
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
    np.testing.assert_allclose(d._dbg["projected_chassis"], [0.0, 0.18, 0.0])
    assert d._dbg["joystick_axis"] == 1

    metadata = d._recording_control_metadata()
    assert metadata["schema"] == np.bytes_("omniteleop_joystick_mobile_raw/v2")
    assert metadata["control_mode"] == np.bytes_("joystick")
    assert metadata["demonstration_source"] == np.bytes_("human_teleoperation")
    assert metadata["eef_target_frame"] == np.bytes_("current_base")
    assert metadata["action_target_frame"] == np.bytes_("current_base")
    assert metadata["head_target_frame"] == np.bytes_("current_base")
    # head_mode "fixed" pins the head at nominal, so action/head (policy dims 20-28) is
    # a leader-side label this take never executed. Both facts are recorded because
    # head_actuated also depends on the --enable mask, which nothing else writes out.
    assert metadata["head_mode"] == np.bytes_("fixed")
    assert bool(metadata["head_actuated"]) is False
    assert metadata["policy_action_schema"] == np.bytes_(
        "omniteleop_wbc_joystick_action/v1"
    )
    assert metadata["base_reference_source"] == np.bytes_("joystick_integrator")
    assert metadata["chassis_turn_step_deg"] == 0
    assert metadata["chassis_intent_mapping"] == np.bytes_("fixed_direction/v1")
    assert float(metadata["chassis_translation_speed_mps"]) == pytest.approx(0.18)
    assert float(metadata["chassis_rotation_speed_radps"]) == pytest.approx(0.30)
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
    # head_mode "track" drives the neck from action/head, so the same dims ARE executed.
    d.cfg = replace(d.cfg, head_mode="track")
    assert bool(d._recording_control_metadata()["head_actuated"]) is True
    # ...unless the head group was never enabled for actuation on this take.
    d.enable = {**d.enable, "head": False}
    assert bool(d._recording_control_metadata()["head_actuated"]) is False
    d.enable = {**d.enable, "head": True}
    d.cfg = replace(d.cfg, head_mode="fixed")
    d._shaper.turn_90 = True
    metadata = d._recording_control_metadata()
    assert metadata["chassis_turn_step_deg"] == 90
    assert metadata["base_reference_source"] == np.bytes_("joystick_integrator_with_quarter_turn_goal")


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
    assert not extra["chassis"]["turn_active"]
    d._shaper.turn_target_yaw = 1.2
    extra = d._recording_extra_action()
    assert extra["chassis"]["turn_active"]
    assert extra["chassis"]["turn_target_yaw"] == pytest.approx(1.2)


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
    assert not args.turn_90
    assert not args.streaming_recorder
    assert not args.no_head_depth
    assert not args.head_right_rgb
    assert not args.wrist_right_rgb
    assert args.record_startup_timeout > 0.0
    assert args.record_max_camera_skew_ms >= 0.0
    assert args.record_max_camera_age_ms > 0.0
    assert args.lock_torso_in_ik is None
    assert args.joystick_translation_speed == pytest.approx(0.18)
    assert args.joystick_rotation_speed == pytest.approx(0.30)
    monkeypatch.setattr(sys, "argv", ["wbc_joystick_robot.py", "--turn-90"])
    m.main()
    assert captured["args"].turn_90
    for flags in (["--enable", "none"], ["--replay", "episode.hdf5"]):
        monkeypatch.setattr(sys, "argv", ["wbc_joystick_robot.py", "--turn-90", *flags])
        with pytest.raises(SystemExit):
            m.main()


def test_joystick_exception_path_quarantines_recording():
    m = _load_script("wbc_joystick_robot")
    calls = []
    driver = SimpleNamespace(
        abort_recording_episode=lambda reason: calls.append(reason),
    )

    m._quarantine_failed_recording(driver, RuntimeError("camera stalled"))

    assert calls == ["camera stalled"]


def _replay_vr(timestamp_ns: int, chassis) -> VRJointData:
    pose = np.eye(4).reshape(-1).tolist()
    return VRJointData(
        timestamp_ns=timestamp_ns,
        calib_stage="teleop",
        estop=False,
        left_ee_pose=pose,
        right_ee_pose=pose,
        head_ee_pose=pose,
        chassis_vx=float(chassis[0]),
        chassis_vy=float(chassis[1]),
        chassis_wz=float(chassis[2]),
    )


def test_joystick_raw_stream_replay_marks_head_targets_unfiltered(tmp_path):
    times = iter([1.0, 1.1])
    recorder = StreamRecorder(clock=lambda: next(times))
    recorder.append(_replay_vr(10, [0.2, 0.0, 0.0]))
    recorder.append(_replay_vr(20, [0.0, -0.1, 0.0]))
    path = tmp_path / "leader_stream.hdf5"
    recorder.write(str(path))

    source = ReplaySource.from_joystick_hdf5(str(path))

    assert not source.head_targets_pre_filtered
    assert source.replay_kind == "leader_stream"
    assert source._vr[0].chassis_vx == pytest.approx(0.2)


def _write_joystick_replay_episode(
    path,
    *,
    include_intent: bool = True,
    schema: str = "omniteleop_joystick_mobile_raw/v1",
):
    n = 2
    poses = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    with h5py.File(path, "w") as f:
        f["meta/schema"] = np.asarray(schema.encode())
        f["meta/control_mode"] = np.asarray(b"joystick")
        f["meta/policy_action_schema"] = np.asarray(
            b"omniteleop_wbc_joystick_action/v1"
        )
        for key in ("action_target_frame", "eef_target_frame", "head_target_frame"):
            f[f"meta/{key}"] = np.asarray(b"current_base")
        if schema.endswith("/v2"):
            f["meta/chassis_intent_mapping"] = np.asarray(b"fixed_direction/v1")
            f["meta/chassis_translation_speed_mps"] = np.float64(0.15)
            f["meta/chassis_rotation_speed_radps"] = np.float64(0.25)
        f["timestamp_ns"] = np.array([1_000_000_000, 1_100_000_000], np.int64)
        f["action/eef/left"] = poses
        f["action/eef/right"] = poses
        f["action/head"] = poses
        f["action/gripper/left"] = np.zeros(n, np.float32)
        f["action/gripper/right"] = np.ones(n, np.float32)
        if include_intent:
            intent = (
                [[0.15, 0.0, 0.0], [0.0, -0.15, 0.0]]
                if schema.endswith("/v2")
                else [[0.2, 0.0, 0.0], [0.0, -0.1, 0.3]]
            )
            f["action/chassis/intent_body"] = np.asarray(intent, np.float32)


def test_joystick_episode_replay_restores_intent_and_skips_second_head_filter(
    tmp_path,
):
    path = tmp_path / "episode_0.hdf5"
    _write_joystick_replay_episode(path)

    source = ReplaySource.from_joystick_hdf5(str(path))

    assert source.head_targets_pre_filtered
    assert source.replay_kind == "joystick_episode"
    np.testing.assert_allclose(
        [
            source._vr[1].chassis_vx,
            source._vr[1].chassis_vy,
            source._vr[1].chassis_wz,
        ],
        [0.0, -0.1, 0.3],
    )


def test_joystick_episode_replay_accepts_fixed_speed_raw_v2(tmp_path):
    path = tmp_path / "episode_0.hdf5"
    _write_joystick_replay_episode(
        path, schema="omniteleop_joystick_mobile_raw/v2"
    )

    source = ReplaySource.from_joystick_hdf5(str(path))

    assert source.replay_kind == "joystick_episode"


def test_generated_fixed_speed_episode_rejects_clock_stretch(tmp_path):
    path = tmp_path / 'trajectory.hdf5'
    _write_joystick_replay_episode(path, schema='omniteleop_joystick_mobile_raw/v2')
    with h5py.File(path, 'a') as f:
        f['meta/required_replay_speed'] = 1.
    with pytest.raises(ValueError, match='fixed-speed.*--speed'):
        ReplaySource.from_joystick_hdf5(str(path), speed=.5)
    assert ReplaySource.from_joystick_hdf5(str(path), speed=1.).speed == 1.


def test_joystick_episode_replay_refuses_to_guess_intent_from_applied_twist(
    tmp_path,
):
    path = tmp_path / "episode_0.hdf5"
    _write_joystick_replay_episode(path, include_intent=False)

    with pytest.raises(ValueError, match="action/chassis/intent_body"):
        ReplaySource.from_joystick_hdf5(str(path))


def _yaml_with_joystick_head_mode(tmp_path, head_mode: str) -> str:
    """The canonical wbik.yaml with joystick_teleop.head_mode swapped (top-level untouched)."""
    import re

    from omniteleop.follower.whole_body_ik import DEFAULT_CONFIG_PATH

    src = Path(DEFAULT_CONFIG_PATH).read_text()
    out, n = re.subn(r'^  head_mode:\s*"[a-z]+"', f'  head_mode: "{head_mode}"', src, flags=re.M)
    assert n == 1
    path = tmp_path / "wbik.yaml"
    path.write_text(out)
    return str(path)


def test_joystick_ik_takes_head_mode_from_the_joystick_block(tmp_path):
    m = _load_script("wbc_joystick_robot")
    _ik, cfg = m._build_joystick_ik(_yaml_with_joystick_head_mode(tmp_path, "fixed"))
    assert cfg.head_mode == "fixed" and cfg.lock_base_in_ik
    _ik, cfg = m._build_joystick_ik(_yaml_with_joystick_head_mode(tmp_path, "track"))
    assert cfg.head_mode == "track" and cfg.lock_base_in_ik   # top-level "ik" never leaks in
    assert JoystickTeleopConfig.from_yaml().head_mode == "fixed"  # the collection default


# --- recording disposition on operator stop vs fault --------------------------------------

@pytest.mark.parametrize("failure,quarantined", [(KeyboardInterrupt, False), (RuntimeError, True)])
def test_ctrl_c_publishes_the_take_while_a_fault_quarantines_it(monkeypatch, failure, quarantined):
    """Ctrl-C is an operator boundary: frames recorded so far publish as a normal take.

    ``driver.close()`` runs ``stop_recording_episode()``, so reaching close() without
    ``abort_recording_episode()`` is what saves the partial recording as ``episode_<N>.hdf5``.
    A runtime fault instead quarantines it as ``.hdf5.partial`` with ``complete=false``.
    """
    mod = _load_script("wbc_joystick_robot")
    events = []

    class _Driver:
        def __init__(self, *args, **kwargs):
            pass

        def abort_recording_episode(self, reason):
            events.append("quarantine")

        def close(self):
            events.append("close")   # -> HardwareDriver.stop_recording_episode()

    ik = SimpleNamespace(model=SimpleNamespace(nq=24, nv=23))
    cfg = SimpleNamespace(head_mode="fixed", base_dofs="xy_yaw", lock_base_in_ik=True,
                          lock_torso_in_ik=True, urdf_path="a/b.urdf",
                          enable_base_single_axis=True)
    monkeypatch.setattr(mod, "_build_joystick_ik", lambda *a, **kw: (ik, cfg))
    monkeypatch.setattr(mod, "ReplaySource", SimpleNamespace(
        from_joystick_hdf5=lambda *a, **kw: SimpleNamespace(
            _vr=[0], total_duration=1.0, replay_kind="joystick_episode",
            head_targets_pre_filtered=True, close=lambda: None)))
    monkeypatch.setattr(mod, "JoystickHardwareDriver", _Driver)
    monkeypatch.setattr(mod, "_publish_abort_status", lambda *a, **kw: None)

    def _fail(*a, **kw):
        raise failure("stopped")

    monkeypatch.setattr(mod, "run_loop", _fail)
    args = SimpleNamespace(replay="take.hdf5", speed=1.0, config=None, lock_torso_in_ik=None,
                           debug_dir=None, record=False, arkit_base="off", turn_90=False,
                           joystick_translation_speed=0.18, joystick_rotation_speed=0.30)
    enable = dict.fromkeys(["arms", "torso", "head", "base"], True)

    if quarantined:
        with pytest.raises(RuntimeError):
            mod._run_joystick_ik_mode(args, enable)
    else:
        mod._run_joystick_ik_mode(args, enable)   # Ctrl-C is swallowed, not re-raised

    assert ("quarantine" in events) is quarantined
    assert "close" in events
