# ruff: noqa: SLF001
"""Nominal-head camera-view knob (wbik.yaml nominal_posture.head_j1).

The zed_depth_frame at the WBC nominal posture inherits the torso pitch and looks
~45.3 deg DOWN with the head joints at zero, keeping the robot base in the camera
view. ``nominal_posture.head_j1`` (the NECK PITCH joint, FK-verified) is the single
knob that raises the default view; everything else derives from it:

  * ``VegaWholeBodyIK.nominal_q()`` -> follower homing + engage reset + head-target
    LPF/deadband/interpolator seeds (``frame_pose(HEAD_FRAME)``);
  * head_mode "ik" PINS head_j1 (dq = 0), so the raise is structural;
  * the leader anchors its calibration (``C_head``) on the nominal head FK, so the
    operator's straight-ahead gaze starts at exactly the configured view.

Cross-validated with Codex (gpt-5.5): C_head anchoring holds for any nominal head
pose; raising the view moves _heading_yaw AWAY from its near-vertical degeneracy and
improves the pan-axis projected-yaw geometry (~1.38x cone amplification at 0 ->
~1.12x at head_j1 = -0.3).

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yixuan/miniforge3/envs/dexmate/bin/python \
     -m pytest tests/test_wbc_nominal_head.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from omniteleop.follower.whole_body_ik import (
    HEAD_FRAME,
    HEAD_JOINTS,
    DEFAULT_NOMINAL_POSTURE,
    VegaWholeBodyIK,
    WBCConfig,
)

RAISED_HEAD_J1 = -0.3  # rad; ~45.3 deg -> ~28.1 deg down-look (1:1 rad -> view pitch)


def _load_wbc_vr_leader():
    path = Path(__file__).resolve().parents[1] / "scripts" / "wbc_vr_leader.py"
    spec = importlib.util.spec_from_file_location("wbc_vr_leader", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fast_config(**overrides) -> WBCConfig:
    """WBCConfig without the collision-sphere model (URDF load dominates test time)."""
    return WBCConfig(enable_collision_avoidance=False, **overrides)


def _raised_config() -> WBCConfig:
    return _fast_config(
        nominal_posture={**DEFAULT_NOMINAL_POSTURE, "head_j1": RAISED_HEAD_J1}
    )


def _zero_head_config() -> WBCConfig:
    return _fast_config(
        nominal_posture={**DEFAULT_NOMINAL_POSTURE, "head_j1": 0.0}
    )


def _to_mat(pose) -> np.ndarray:
    return np.asarray(pose.homogeneous if hasattr(pose, "homogeneous") else pose, dtype=float)


def _view_pitch_deg(head_pose: np.ndarray) -> float:
    """Downward pitch (deg, + = looking down) of the optical z viewing direction."""
    fwd = np.asarray(head_pose, dtype=float)[:3, 2]
    return float(np.degrees(np.arcsin(-fwd[2] / np.linalg.norm(fwd))))


def _level_headset(pos: np.ndarray) -> np.ndarray:
    """Headset pose looking horizontally along +x (optical z forward)."""
    out = np.eye(4)
    out[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    out[:3, 3] = pos
    return out


# -- nominal_q / FK ---------------------------------------------------------------


def test_nominal_head_j1_flows_into_nominal_q_and_raises_view():
    """The yaml knob must land in nominal_q and pitch the camera view 1:1 (rad->rad)."""
    ik0 = VegaWholeBodyIK(_zero_head_config())
    ik1 = VegaWholeBodyIK(_raised_config())

    assert ik1.nominal_q()[ik1._idx_q["head_j1"]] == pytest.approx(RAISED_HEAD_J1)
    # Untouched head joints stay zero (head_j2 pan / head_j3 tilt are not the knob).
    for name in ("head_j2", "head_j3"):
        assert ik1.nominal_q()[ik1._idx_q[name]] == pytest.approx(0.0)

    ik0.reset()
    ik1.reset()
    pitch0 = _view_pitch_deg(_to_mat(ik0.frame_pose(HEAD_FRAME)))
    pitch1 = _view_pitch_deg(_to_mat(ik1.frame_pose(HEAD_FRAME)))
    assert pitch0 == pytest.approx(45.3, abs=0.5)  # base in view at all-zero head
    # head_j1 maps ~1:1 onto view pitch: -0.3 rad = -17.19 deg of down-look.
    assert pitch0 - pitch1 == pytest.approx(np.degrees(0.3), abs=0.5)


def test_homing_nominal_extraction_includes_configured_head():
    """The HardwareDriver homing extraction pattern must pick up the head knob."""
    ik = VegaWholeBodyIK(_raised_config())
    nq = ik.nominal_q()
    head = np.array([nq[ik._idx_q[n]] for n in HEAD_JOINTS])
    np.testing.assert_allclose(head, [RAISED_HEAD_J1, 0.0, 0.0])


def test_unknown_nominal_posture_key_raises():
    """A typo'd joint must fail fast, not silently home to an unintended posture."""
    cfg = _fast_config(
        nominal_posture={**DEFAULT_NOMINAL_POSTURE, "head_i1": RAISED_HEAD_J1}
    )
    with pytest.raises(ValueError, match="head_i1"):
        VegaWholeBodyIK(cfg)


# -- leader calibration anchoring ---------------------------------------------------


def test_leader_calibration_anchors_camera_at_configured_nominal_view():
    """At calibration the mapped head target must equal the RAISED nominal head FK.

    C_head is captured as (R_yaw @ vr_calib_R)^T @ T_base_head, so this must hold for
    ANY configured nominal head pose -- the operator's straight-ahead gaze then starts
    at exactly the configured camera view, and operator pitch rides on top of it.
    """
    ik = VegaWholeBodyIK(_raised_config())
    ik.reset()
    t_base_head = _to_mat(ik.frame_pose(HEAD_FRAME))

    leader_mod = _load_wbc_vr_leader()
    leader = leader_mod.WBCVRLeader.__new__(leader_mod.WBCVRLeader)
    leader.T_base_head = t_base_head

    headset = _level_headset(np.array([0.3, -0.2, 1.62]))
    leader._calibrate(headset)
    mapped = leader._map_head_target(headset)
    np.testing.assert_allclose(mapped, t_base_head, atol=1e-9)
    assert _view_pitch_deg(mapped) == pytest.approx(
        45.3 + np.degrees(RAISED_HEAD_J1), abs=0.5
    )

    # An operator pitch-down of 10 deg adds onto the configured view, not onto -45:
    # rotate the headset about its camera-right axis (local x = world -y here, so the
    # DOWN direction is a NEGATIVE rotation about that axis).
    pitched = headset.copy()
    x_axis = headset[:3, 0]
    c, s = np.cos(np.radians(-10)), np.sin(np.radians(-10))
    k = np.array(
        [
            [0.0, -x_axis[2], x_axis[1]],
            [x_axis[2], 0.0, -x_axis[0]],
            [-x_axis[1], x_axis[0], 0.0],
        ]
    )
    pitched[:3, :3] = (np.eye(3) + s * k + (1.0 - c) * (k @ k)) @ headset[:3, :3]
    mapped_pitched = leader._map_head_target(pitched)
    assert _view_pitch_deg(mapped_pitched) == pytest.approx(
        _view_pitch_deg(mapped) + 10.0, abs=0.2
    )


def _view_yaw_deg(head_pose: np.ndarray) -> float:
    """Projected heading yaw (deg) -- both nominal views must face +x exactly."""
    fwd = np.asarray(head_pose, dtype=float)[:3, 2]
    return float(np.degrees(np.arctan2(fwd[1], fwd[0])))


def test_bprime_hand_targets_shift_only_by_nominal_head_position_delta():
    """B' hand mapping is yaw+position-of-head only: raising the view must shift the
    hand targets by EXACTLY the nominal head-position delta (a ~2-3 cm second-order
    effect), with no orientation/pitch coupling."""
    ik0 = VegaWholeBodyIK(_fast_config())
    ik1 = VegaWholeBodyIK(_raised_config())
    ik0.reset()
    ik1.reset()
    head0 = _to_mat(ik0.frame_pose(HEAD_FRAME))
    head1 = _to_mat(ik1.frame_pose(HEAD_FRAME))
    # The URDF's 2.5 cm lateral head offset leaves a ~5e-4 deg residual nominal yaw;
    # both views must face +x up to that numerical residual.
    assert abs(_view_yaw_deg(head0)) < 0.01 and abs(_view_yaw_deg(head1)) < 0.01

    leader_mod = _load_wbc_vr_leader()
    leader = leader_mod.WBCVRLeader.__new__(leader_mod.WBCVRLeader)

    vr_head = _level_headset(np.array([0.0, 0.0, 1.6]))
    vr_hand = np.eye(4)
    vr_hand[:3, 3] = np.array([0.35, 0.20, 1.15])  # hand fwd/left/below the headset

    p0 = leader._bprime_pos("left", vr_hand, vr_head, head0)
    p1 = leader._bprime_pos("left", vr_hand, vr_head, head1)
    # atol: the residual nominal-yaw difference above rotates the ~0.5 m hand-rel-head
    # lever by <~1e-5 rad -> micron-scale; anything beyond that is a real coupling.
    np.testing.assert_allclose(p1 - p0, head1[:3, 3] - head0[:3, 3], atol=1e-5)


# -- pinned-joint pipeline invariant -------------------------------------------------


def test_head_pin_holds_configured_nominal_under_solve():
    """head_mode "ik" pins head_j1: 50 solve ticks at the nominal targets must keep
    head_j1 at the configured value and the camera at the raised view (the follower's
    steady-state invariant)."""
    cfg = _raised_config()
    assert "head_j1" in tuple(cfg.head_ik_pinned_joints)
    ik = VegaWholeBodyIK(cfg)
    ik.reset()
    left = _to_mat(ik.frame_pose("L_ee"))
    right = _to_mat(ik.frame_pose("R_ee"))
    head = _to_mat(ik.frame_pose(HEAD_FRAME))

    for _ in range(50):
        result = ik.solve(left, right, 0.01, head_target=head)
        assert result.success and not result.held, f"solve failed: {result.safety_status}"

    q = ik.configuration.q
    assert q[ik._idx_q["head_j1"]] == pytest.approx(RAISED_HEAD_J1, abs=1e-6)
    assert _view_pitch_deg(_to_mat(ik.frame_pose(HEAD_FRAME))) == pytest.approx(
        45.3 + np.degrees(RAISED_HEAD_J1), abs=0.6
    )
