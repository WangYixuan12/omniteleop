# ruff: noqa: SLF001

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_wbc_vr_leader():
    path = Path(__file__).resolve().parents[1] / "scripts" / "wbc_vr_leader.py"
    spec = importlib.util.spec_from_file_location("wbc_vr_leader", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def test_walking_forward_after_calibration_keeps_head_target_height_level():
    leader_mod = _load_wbc_vr_leader()
    leader = leader_mod.WBCVRLeader.__new__(leader_mod.WBCVRLeader)

    c = np.sqrt(0.5)
    # zed_depth_frame nominal: optical z points forward and down. In WBC head_mode
    # "ik", head_ee_pose translation is a real whole-body target, so walking
    # horizontally must not inherit this optical pitch.
    leader.T_base_head = _pose(
        np.array(
            [
                [0.0, -c, c],
                [-1.0, 0.0, 0.0],
                [0.0, -c, -c],
            ]
        ),
        np.array([0.09, 0.025, 1.41]),
    )

    # Level headset looking along robot +x: local z is horizontal forward.
    level_head = _pose(
        np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        ),
        np.array([0.0, 0.0, 1.6]),
    )
    leader._calibrate(level_head)

    head1 = level_head.copy()
    head1[0, 3] += 0.8

    h_start = leader._map_head_target(level_head)
    h_end = leader._map_head_target(head1)
    delta = h_end[:3, 3] - h_start[:3, 3]

    assert delta[0] > 0.55
    assert abs(delta[2]) < 1e-9


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about ``axis`` by ``angle`` (radians)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _calib_leader():
    """A WBCVRLeader with a yawed room calibration and non-trivial nominal EE poses.

    The headset is yawed (room->base rotation != identity) so the position tests also
    confirm the gravity-aligned yaw is applied.
    """
    leader_mod = _load_wbc_vr_leader()
    leader = leader_mod.WBCVRLeader.__new__(leader_mod.WBCVRLeader)
    # Nominal head: level optical frame (z forward) so _heading_yaw is well defined.
    leader.T_base_head = _pose(
        np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        np.array([0.09, 0.0, 1.45]),
    )
    leader._nominal_left = _pose(_rot([0.0, 0.0, 1.0], 0.3), np.array([0.40, 0.22, 1.05]))
    leader._nominal_right = _pose(_rot([0.0, 0.0, 1.0], -0.3), np.array([0.40, -0.22, 1.05]))
    leader.last_left_target = None
    leader.last_right_target = None
    return leader_mod, leader


def _level_headset(yaw: float, pos: np.ndarray) -> np.ndarray:
    """Headset pose looking horizontally, rotated by ``yaw`` about world z."""
    base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    out = np.eye(4)
    out[:3, :3] = _rot([0.0, 0.0, 1.0], yaw) @ base
    out[:3, 3] = pos
    return out


def _abs_ori(leader, vr):
    """The C=identity EEF orientation for a controller pose (R_yaw @ vr_now_rot)."""
    return (leader.robot_base_t_vr_base_eef @ vr)[:3, :3]


def test_is_tracked_rejects_sentinel_and_nonfinite_poses():
    """A NaN/malformed pose must read as untracked (else it poisons last-good targets)."""
    leader_mod = _load_wbc_vr_leader()
    good = _pose(np.eye(3), np.array([0.2, 0.1, 1.0]))
    assert leader_mod._is_tracked(good, leader_mod.INVALID_LEFT_POSE)
    # Exact untracked sentinel -> not tracked.
    assert not leader_mod._is_tracked(leader_mod.INVALID_LEFT_POSE, leader_mod.INVALID_LEFT_POSE)
    # NaN -> not tracked / not valid (previously np.allclose(.,sentinel) was False -> "tracked").
    nan_pose = good.copy()
    nan_pose[0, 3] = np.nan
    assert not leader_mod._is_valid_pose(nan_pose)
    assert not leader_mod._is_tracked(nan_pose, leader_mod.INVALID_LEFT_POSE)
    # Malformed bottom row -> not valid.
    bad_row = good.copy()
    bad_row[3, :] = np.array([1.0, 0.0, 0.0, 1.0])
    assert not leader_mod._is_valid_pose(bad_row)


def _rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle (rad) between two rotation matrices."""
    r = Ra.T @ Rb
    return float(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))


# -- EEF orientation offset C (controller->gripper "tool-mount" re-labeling) -----------


def _eo_leader():
    """A calibrated head-mode leader (identity offset) for exercising the EEF offset C.

    head mode orientation is R_yaw @ vr_now_rot @ C -- the same formula the removed
    mappings used -- so these C tests still pin the live orientation behavior. _hf_engage_t
    is None so the engage blend is complete (offset 0) and _eef_target returns the steady map.
    """
    leader_mod, leader = _calib_leader()
    leader._ee_offset_left = np.eye(3)
    leader._ee_offset_right = np.eye(3)
    leader._hf_offset_left = np.zeros(3)
    leader._hf_offset_right = np.zeros(3)
    leader._hf_engage_t = None
    leader._hf_blend_dur = 0.5
    leader._calibrate(_level_headset(0.0, np.array([1.0, 0.0, 1.6])))
    return leader_mod, leader


def _eo_ori(leader, side, vr_now):
    """EEF target orientation for a controller pose in head mode (any valid head works --
    orientation is R_yaw @ vr_now_rot @ C, independent of head_target)."""
    head = _level_headset(0.0, np.array([1.0, 0.0, 1.6]))
    return leader._eef_target(
        side, vr_now, vr_head=head, head_target=head, now=1.0
    )[:3, :3]


def test_ee_offset_identity_matches_uncalibrated_orientation():
    """C = identity reproduces the raw R_yaw @ vr_now orientation (no behavior change)."""
    _, leader = _eo_leader()
    vr1 = _pose(_rot([0.4, 0.1, 0.9], 1.1), np.array([1.2, 0.2, 1.4]))
    np.testing.assert_allclose(_eo_ori(leader, "left", vr1), _abs_ori(leader, vr1), atol=1e-12)


def test_ee_offset_maps_neutral_grip_to_nominal():
    """With C = (R_yaw @ vr_ref)^T @ nom_R, the neutral grip reads as the nominal frame."""
    _, leader = _eo_leader()
    vr_ref_l = _pose(_rot([0.3, 0.5, 0.8], 0.7), np.array([1.2, 0.2, 1.4]))
    vr_ref_r = _pose(_rot([0.1, -0.7, 0.4], -0.5), np.array([1.2, -0.2, 1.4]))
    r_yaw = leader.robot_base_t_vr_base_eef[:3, :3]
    leader._ee_offset_left = (r_yaw @ vr_ref_l[:3, :3]).T @ leader._nominal_left[:3, :3]
    leader._ee_offset_right = (r_yaw @ vr_ref_r[:3, :3]).T @ leader._nominal_right[:3, :3]

    np.testing.assert_allclose(
        _eo_ori(leader, "left", vr_ref_l), leader._nominal_left[:3, :3], atol=1e-9)
    np.testing.assert_allclose(
        _eo_ori(leader, "right", vr_ref_r), leader._nominal_right[:3, :3], atol=1e-9)


def test_ee_offset_preserves_absolute_no_contortion_property():
    """A frozen C keeps gripper deviation-from-nominal equal to the controller's angle from
    the fixed NEUTRAL reference -- NOT from the engage pose. So no relative-style contortion:
    angle(R_target, nom) == angle(vr_now, vr_ref), independent of any engage anchor."""
    _, leader = _eo_leader()
    vr_ref = _pose(_rot([0.3, 0.5, 0.8], 0.7), np.zeros(3))
    r_yaw = leader.robot_base_t_vr_base_eef[:3, :3]
    leader._ee_offset_left = (r_yaw @ vr_ref[:3, :3]).T @ leader._nominal_left[:3, :3]
    vr_now = _pose(_rot([0.5, 0.1, 0.3], 1.2), np.array([1.3, 0.25, 1.45]))

    dev = _rot_angle(leader._nominal_left[:3, :3], _eo_ori(leader, "left", vr_now))
    np.testing.assert_allclose(dev, _rot_angle(vr_ref[:3, :3], vr_now[:3, :3]), atol=1e-9)


def test_ee_offset_does_not_change_translation():
    """C is a pure rotation: it must not move the EEF target position."""
    _, leader = _eo_leader()
    head = _level_headset(0.0, np.array([1.0, 0.0, 1.6]))
    vr1 = _pose(_rot([0.2, 1.0, 0.3], 0.9), np.array([1.45, 0.05, 1.5]))

    leader._ee_offset_left = _rot([0.4, 0.2, 0.7], 1.3)  # arbitrary non-identity
    pos_with = leader._eef_target("left", vr1, vr_head=head, head_target=head, now=1.0)[:3, 3]
    leader._ee_offset_left = np.eye(3)
    pos_without = leader._eef_target("left", vr1, vr_head=head, head_target=head, now=1.0)[:3, 3]
    np.testing.assert_allclose(pos_with, pos_without, atol=1e-12)


def test_recomputing_offset_each_engage_reproduces_relative_invariant():
    """The knife-edge (codex claim D): if C were recomputed from the ENGAGE pose vr0
    instead of frozen at a fixed neutral reference, gripper deviation collapses to
    angle(vr_now, vr0) -- exactly the rejected relative-orientation invariant that
    contorts the wrist on big rolls."""
    _, leader = _eo_leader()
    r_yaw = leader.robot_base_t_vr_base_eef[:3, :3]
    vr0 = _pose(_rot([0.1, 0.2, 0.9], 0.6), np.array([1.2, 0.2, 1.4]))
    vr_now = _pose(_rot([0.5, 0.1, 0.3], 2.0), np.array([1.3, 0.25, 1.45]))  # big roll
    # C recomputed from the engage pose -- the thing we must NOT do.
    leader._ee_offset_left = (r_yaw @ vr0[:3, :3]).T @ leader._nominal_left[:3, :3]

    dev = _rot_angle(leader._nominal_left[:3, :3], _eo_ori(leader, "left", vr_now))
    np.testing.assert_allclose(dev, _rot_angle(vr0[:3, :3], vr_now[:3, :3]), atol=1e-9)


def test_load_ee_offsets_missing_file_is_identity(tmp_path):
    """A missing offset file (or None path) loads identity (raw controller convention)."""
    leader_mod = _load_wbc_vr_leader()
    for path in (None, str(tmp_path / "nope.yaml")):
        cl, cr = leader_mod._load_ee_offsets(path)
        np.testing.assert_allclose(cl, np.eye(3))
        np.testing.assert_allclose(cr, np.eye(3))


def test_save_load_ee_offsets_round_trip(tmp_path):
    """Saved offsets reload to the same rotation (quaternion round-trip)."""
    leader_mod = _load_wbc_vr_leader()
    cl = _rot([0.2, 0.5, 0.8], 1.1)
    cr = _rot([0.9, 0.1, 0.3], -0.7)
    path = str(tmp_path / "ee_offset.yaml")
    leader_mod._save_ee_offsets(path, cl, cr)
    rl, rr = leader_mod._load_ee_offsets(path)
    np.testing.assert_allclose(rl, cl, atol=1e-9)
    np.testing.assert_allclose(rr, cr, atol=1e-9)


def test_follower_status_overlay_lines_show_hold_and_safety_warning():
    """The headset HUD should surface the real follower hold/safety state."""
    leader_mod = _load_wbc_vr_leader()
    status = leader_mod.WBCFollowerStatus(
        timestamp_ns=123,
        stage="teleop",
        estop=False,
        success=False,
        held=True,
        hold=True,
        hold_reason="source_timeout",
        safety_status="HELD: self-collision -1.2cm",
        left_ee_error_mm=12.3,
        right_ee_error_mm=45.6,
    )

    lines = leader_mod._follower_status_overlay_lines(status, status_age_s=0.2)

    assert "Follower: HOLD:source_timeout" in lines
    assert "WARNING: IK solve failed" in lines
    # The gate-prefixed safety string (carrying the CoM margin) is surfaced verbatim
    # only when non-ok; the always-on margin line is gone.
    assert "HELD: self-collision -1.2cm" in lines
    assert "err L/R=12/46mm" in lines


def test_follower_status_overlay_lines_warn_when_status_stale():
    leader_mod = _load_wbc_vr_leader()
    status = leader_mod.WBCFollowerStatus(timestamp_ns=123, stage="teleop")

    lines = leader_mod._follower_status_overlay_lines(status, status_age_s=1.6)

    assert "WARNING: follower status stale 1.6s" in lines


def test_follower_status_overlay_color_marks_alerts_red():
    leader_mod = _load_wbc_vr_leader()
    red = (0, 0, 255)
    white = (255, 255, 255)

    assert leader_mod._follower_status_overlay_color("WARNING: IK solve failed") == red
    assert leader_mod._follower_status_overlay_color("WARN: CoM margin 1.2cm") == red
    assert leader_mod._follower_status_overlay_color("HELD: self-collision -1.2cm") == red
    assert leader_mod._follower_status_overlay_color("err L/R=12/46mm") == white
    assert leader_mod._follower_status_overlay_color("Follower: HOLD:source_timeout") == white


def test_follower_status_overlay_lines_hide_normal_teleop_state():
    """The HUD stage line already says teleop; do not repeat it as follower state."""
    leader_mod = _load_wbc_vr_leader()
    status = leader_mod.WBCFollowerStatus(
        timestamp_ns=123,
        stage="teleop",
        estop=False,
        hold=False,
        success=True,
        safety_status="ok",
        left_ee_error_mm=1.2,
        right_ee_error_mm=3.4,
    )

    lines = leader_mod._follower_status_overlay_lines(status, status_age_s=0.2)

    assert "Follower: teleop" not in lines
    assert lines == ["err L/R=1/3mm"]


def test_stop_hud_forces_stage_stop_on_left_x():
    leader_mod = _load_wbc_vr_leader()
    leader = leader_mod.WBCVRLeader.__new__(leader_mod.WBCVRLeader)
    calls = []

    class FakeHUD:
        def poll_and_send(self, **kwargs):
            calls.append(kwargs)

    status = leader_mod.WBCFollowerStatus(timestamp_ns=123, stage="teleop")
    leader._hud = FakeHUD()
    leader._follower_status_snapshot = lambda: (status, 0.1)

    leader_mod.WBCVRLeader._send_stop_hud(leader)

    assert calls == [{"stage": "stop", "status": status, "status_age_s": 0.1}]


# -- head-target orientation: gravity-aligned, no yaw -> pitch/roll cone ---------------


def _head_leader():
    """A leader whose nominal head (zed_depth_frame) is pitched 45 deg DOWN.

    The downward pitch is exactly what made the old full-frame orientation calibration
    sweep a cone on a yaw-only head turn (operator yaw about gravity became a rotation
    about the 45-deg-tilted optical axis). A level nominal head would hide the bug.
    """
    leader_mod = _load_wbc_vr_leader()
    leader = leader_mod.WBCVRLeader.__new__(leader_mod.WBCVRLeader)
    c = np.sqrt(0.5)
    leader.T_base_head = _pose(
        np.array([[0.0, -c, c], [-1.0, 0.0, 0.0], [0.0, -c, -c]]),
        np.array([0.09, 0.025, 1.41]),
    )
    return leader_mod, leader


def _forward(R: np.ndarray) -> np.ndarray:
    """Optical viewing axis (local z) of a head/optical rotation."""
    return R[:, 2]


def test_head_target_orientation_starts_at_nominal():
    """At the calibration instant the head target orientation IS the nominal head FK.

    C_head is captured so R_yaw @ vr_calib_R @ C_head == T_base_head: the camera engages
    at the -45 deg down-look with no jump, regardless of where the operator's head is.
    """
    _, leader = _head_leader()
    cal = _level_headset(0.5, np.array([0.2, -0.1, 1.6]))
    leader._calibrate(cal)
    out = leader._map_head_target(cal)
    np.testing.assert_allclose(out[:3, :3], leader.T_base_head[:3, :3], atol=1e-9)


def test_head_target_yaw_only_is_pure_camera_yaw_no_pitch_roll():
    """A yaw-only head turn rotates the camera purely about base z (no cone).

    out_R == Rz(dtheta) @ nominal, where dtheta is the operator's turn since
    calibration -- so the forward-axis elevation (pitch) is invariant and roll stays 0.
    The old full-frame map produced ~100 deg of pitch/roll on this same input.
    """
    _, leader = _head_leader()
    cal_yaw = 0.7  # operator not facing robot-forward at calibration (exercises R_yaw)
    cal = _level_headset(cal_yaw, np.array([0.2, -0.1, 1.6]))
    leader._calibrate(cal)
    nominal_pitch = float(np.arcsin(_forward(leader.T_base_head[:3, :3])[2]))
    for dtheta in (-2.0, -1.0, -0.3, 0.3, 1.0, 2.0):
        vr = _level_headset(cal_yaw + dtheta, np.array([0.2, -0.1, 1.6]))
        out = leader._map_head_target(vr)
        expected = _rot([0.0, 0.0, 1.0], dtheta) @ leader.T_base_head[:3, :3]
        np.testing.assert_allclose(out[:3, :3], expected, atol=1e-9)
        # Forward-axis elevation (pitch) unchanged -> no yaw -> pitch coupling.
        assert abs(float(np.arcsin(_forward(out[:3, :3])[2])) - nominal_pitch) < 1e-9


def test_head_target_orientation_is_gravity_yaw_conjugated_head_motion():
    """General invariant: head_R == R_yaw @ dR_world @ R_yaw.T @ nominal.

    The operator's head rotation since calibration (dR_world, a world-frame premultiply)
    is conjugated by the PURE gravity yaw R_yaw, so gravity maps to base z; a real
    pitch/roll then maps to camera pitch/roll while yaw stays yaw.
    """
    _, leader = _head_leader()
    cal = _level_headset(0.3, np.array([0.1, -0.2, 1.6]))
    leader._calibrate(cal)
    r_yaw = leader.robot_base_t_vr_base_eef[:3, :3]
    d_world = _rot([0.2, 1.0, -0.3], 0.7)  # arbitrary head rotation since calibration
    vr = _pose(d_world @ cal[:3, :3], cal[:3, 3])
    out = leader._map_head_target(vr)
    expected = r_yaw @ d_world @ r_yaw.T @ leader.T_base_head[:3, :3]
    np.testing.assert_allclose(out[:3, :3], expected, atol=1e-9)


def test_head_target_pitch_tracks_operator_head_pitch():
    """An operator head pitch (no yaw) moves camera pitch but not heading."""
    _, leader = _head_leader()
    cal = _level_headset(0.0, np.array([0.0, 0.0, 1.6]))  # facing robot-forward
    leader._calibrate(cal)
    f0 = _forward(leader.T_base_head[:3, :3])
    base_yaw = np.degrees(np.arctan2(f0[1], f0[0]))
    prev = float(np.arcsin(f0[2]))
    for phi in (0.1, 0.3, 0.6):  # pitch the head DOWN about world-left (+y)
        vr = _pose(_rot([0.0, 1.0, 0.0], phi) @ cal[:3, :3], cal[:3, 3])
        f = _forward(leader._map_head_target(vr)[:3, :3])
        # Heading yaw unchanged (a pure pitch stays in the x-z plane).
        assert abs(np.degrees(np.arctan2(f[1], f[0])) - base_yaw) < 1e-6
        # Camera pitches monotonically further down as the head pitches down.
        pitch = float(np.arcsin(np.clip(f[2], -1, 1)))
        assert pitch < prev + 1e-9
        prev = pitch


# -- head-relative hand position (yaw-only B', _bprime_pos) -----------------------

def _head_pose(yaw: float, pos: np.ndarray) -> np.ndarray:
    """Headset/optical pose with the viewing axis (z) horizontal-forward, yawed about
    gravity by ``yaw`` so ``_heading_yaw`` returns ``yaw`` (away from the vertical-axis
    degeneracy a pure Rz would hit)."""
    r0 = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    return _pose(_rot([0.0, 0.0, 1.0], yaw) @ r0, np.asarray(pos, dtype=float))


def _headmode_leader(leader_mod):
    leader = leader_mod.WBCVRLeader.__new__(leader_mod.WBCVRLeader)
    leader._nominal_left = _pose(np.eye(3), np.array([0.5, 0.3, 1.0]))
    leader._nominal_right = _pose(np.eye(3), np.array([0.5, -0.3, 1.0]))
    leader.robot_base_t_vr_base_eef = np.eye(4)  # identity room->base: head_target == raw head
    leader._ee_offset_left = np.eye(3)
    leader._ee_offset_right = np.eye(3)
    leader._hf_offset_left = np.zeros(3)
    leader._hf_offset_right = np.zeros(3)
    leader._hf_engage_t = None
    leader._hf_blend_dur = 0.5
    return leader


def test_head_relative_engage_blend_starts_at_nominal_then_reaches_bprime():
    """The startup blend ramps from nominal (no engage lunge) to B' over _hf_blend_dur."""
    leader_mod = _load_wbc_vr_leader()
    leader = _headmode_leader(leader_mod)
    head0 = _head_pose(0.0, np.array([0.0, 0.0, 1.4]))
    vr_l0 = _pose(np.eye(3), np.array([0.6, 0.2, 1.3]))  # hands NOT at nominal -> B' != nominal
    leader._capture_head_frame_anchors(vr_l0, vr_l0, head0, head0, now=0.0)

    out0 = leader._eef_target("left", vr_l0, vr_head=head0, head_target=head0, now=0.0)
    np.testing.assert_allclose(out0[:3, 3], leader._nominal_left[:3, 3], atol=1e-9)  # no lunge

    out1 = leader._eef_target("left", vr_l0, vr_head=head0, head_target=head0, now=0.5)
    np.testing.assert_allclose(
        out1[:3, 3], leader._bprime_pos("left", vr_l0, head0, head0), atol=1e-9
    )  # blend complete -> exactly B'


def test_head_relative_centers_rigid_turn_and_plants_look_around():
    """A rigid head+hands turn keeps the hand fixed in the camera (centered),
    while a head-only look-around (hand fixed in the room) keeps it world-planted."""
    leader_mod = _load_wbc_vr_leader()
    hyaw, rotz = leader_mod._heading_yaw, leader_mod._rotz
    leader = _headmode_leader(leader_mod)
    head_pos = np.array([0.0, 0.0, 1.4])
    x_off = np.array([0.5, 0.1, -0.2])  # hand offset in the head frame
    yaws = np.radians(np.linspace(0.0, 90.0, 10))
    h0 = _head_pose(0.0, head_pos)
    hand0 = _pose(np.eye(3), head_pos + h0[:3, :3] @ x_off)
    leader._capture_head_frame_anchors(hand0, hand0, h0, h0, now=0.0)

    # Rigid together-turn: hand fixed relative to head -> output fixed in the head-yaw frame.
    cam = []
    for yaw in yaws:
        h = _head_pose(yaw, head_pos)
        hand = _pose(np.eye(3), head_pos + h[:3, :3] @ x_off)
        out = leader._eef_target("left", hand, vr_head=h, head_target=h, now=1.0)
        cam.append(rotz(-hyaw(h)) @ (out[:3, 3] - h[:3, 3]))
    cam = np.array(cam)
    assert np.max(np.linalg.norm(cam - cam[0], axis=1)) < 1e-9  # centered

    # Head-only look-around: hand fixed in the world -> output stays planted in the world.
    hand_fixed = _pose(np.eye(3), np.array([0.7, 0.2, 1.3]))
    world = np.array([
        leader._eef_target(
            "left", hand_fixed, vr_head=_head_pose(yaw, head_pos),
            head_target=_head_pose(yaw, head_pos), now=1.0,
        )[:3, 3]
        for yaw in yaws
    ])
    assert np.max(np.linalg.norm(world - world[0], axis=1)) < 1e-9  # planted
