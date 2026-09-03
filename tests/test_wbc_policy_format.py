"""Unit tests for omniteleop.wbc_policy_format (policy axes + pose codecs).

numpy-only; run via the dexmate env with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
"""

from __future__ import annotations

import numpy as np
import pytest

from omniteleop.wbc_policy_format import (
    ACTION_AXES,
    GRIPPER_BINARY_THRESHOLD,
    GRIPPER_DIMS,
    JOYSTICK_ACTION_AXES,
    JOYSTICK_POLICY_ACTION_SCHEMA,
    POSITION_DIMS,
    ROTATION_DIMS,
    SKIP_NORMALIZATION_DIMS,
    STATE_AXES,
    WBC_POLICY_ACTION_SCHEMA,
    action_axes_for_policy_schema,
    base_pose_to_mat,
    build_state_vector,
    mat_to_pos6d,
    pos6d_to_mat,
    transform_pose,
)


def test_gripper_binary_threshold_matches_dataset_contract():
    assert GRIPPER_BINARY_THRESHOLD == 0.5


def test_axes_layout():
    assert len(STATE_AXES) == 32
    assert len(ACTION_AXES) == 29
    assert STATE_AXES[:29] == ACTION_AXES
    assert STATE_AXES[-3:] == ["base_x", "base_y", "base_yaw"]
    assert ACTION_AXES[-9:] == [
        "head_tx", "head_ty", "head_tz",
        "head_r00", "head_r10", "head_r20",
        "head_r01", "head_r11", "head_r21",
    ]
    assert ACTION_AXES[9] == "left_gripper"
    assert ACTION_AXES[19] == "right_gripper"
    # No duplicate axis names anywhere (indices must be unambiguous downstream).
    assert len(set(STATE_AXES)) == len(STATE_AXES)


def test_joystick_action_schema_extends_wbc_with_body_twist():
    assert len(JOYSTICK_ACTION_AXES) == 32
    assert JOYSTICK_ACTION_AXES[:29] == ACTION_AXES
    assert JOYSTICK_ACTION_AXES[-3:] == [
        "chassis_intent_vx_body",
        "chassis_intent_vy_body",
        "chassis_intent_wz_body",
    ]
    assert action_axes_for_policy_schema(WBC_POLICY_ACTION_SCHEMA) == ACTION_AXES
    assert action_axes_for_policy_schema(JOYSTICK_POLICY_ACTION_SCHEMA) == JOYSTICK_ACTION_AXES
    with pytest.raises(ValueError, match="policy action schema"):
        action_axes_for_policy_schema("unknown/v99")


def test_pos6d_roundtrip_preserves_pose_columns():
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [0.2, -0.1, 0.4]
    T[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    v = mat_to_pos6d(T)
    assert v.shape == (9,)
    assert v.dtype == np.float32
    out = pos6d_to_mat(v)
    np.testing.assert_allclose(out[:3, 3], T[:3, 3], atol=1e-7)
    np.testing.assert_allclose(out[:3, 0], T[:3, 0], atol=1e-7)
    np.testing.assert_allclose(out[:3, 1], T[:3, 1], atol=1e-7)
    # Gram-Schmidt must rebuild a right-handed orthonormal rotation.
    np.testing.assert_allclose(out[:3, :3] @ out[:3, :3].T, np.eye(3), atol=1e-7)
    assert np.linalg.det(out[:3, :3]) > 0.0


def test_pos6d_to_mat_renormalizes_noisy_columns():
    T = np.eye(4)
    v = mat_to_pos6d(T).astype(np.float64)
    v[3:6] *= 1.05  # de-normalize the first rotation column
    out = pos6d_to_mat(v)
    np.testing.assert_allclose(out[:3, :3] @ out[:3, :3].T, np.eye(3), atol=1e-7)


def test_mat_to_pos6d_rejects_bad_input():
    with pytest.raises(ValueError, match="expected"):
        mat_to_pos6d(np.eye(3))
    bad = np.eye(4)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        mat_to_pos6d(bad)


def test_pos6d_to_mat_rejects_bad_input():
    with pytest.raises(ValueError, match="expected 9"):
        pos6d_to_mat(np.zeros(8))
    bad = np.zeros(9)
    bad[3] = 1.0
    bad[7] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        pos6d_to_mat(bad)


def test_base_pose_transform_between_world_and_base():
    world_T_base = base_pose_to_mat(np.array([1.0, 2.0, np.pi / 2]))
    base_T_target = np.eye(4)
    base_T_target[:3, 3] = [0.3, 0.0, 0.5]
    world_T_target = transform_pose(world_T_base, base_T_target)
    np.testing.assert_allclose(world_T_target[:3, 3], [1.0, 2.3, 0.5], atol=1e-7)
    recovered = transform_pose(np.linalg.inv(world_T_base), world_T_target)
    np.testing.assert_allclose(recovered, base_T_target, atol=1e-7)


def test_base_pose_to_mat_rejects_bad_shape():
    with pytest.raises(ValueError, match="base pose"):
        base_pose_to_mat(np.zeros(2))


def test_transform_pose_rejects_bad_shapes():
    with pytest.raises(ValueError, match="4,4"):
        transform_pose(np.eye(3), np.eye(4))


def _base_T(tx, ty, tz):
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    return T


def test_relative_and_normalization_dim_groups():
    # Positions, 6-D rotations, and grippers partition the 29-D action exactly.
    assert POSITION_DIMS == [0, 1, 2, 10, 11, 12, 20, 21, 22]
    assert ROTATION_DIMS == [
        3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28
    ]
    assert GRIPPER_DIMS == [9, 19]
    assert sorted(POSITION_DIMS + ROTATION_DIMS + GRIPPER_DIMS) == list(range(len(ACTION_AXES)))
    assert SKIP_NORMALIZATION_DIMS == ROTATION_DIMS
    assert all(ACTION_AXES[d].rsplit("_", 1)[-1].startswith("r") for d in SKIP_NORMALIZATION_DIMS)


def test_build_state_vector_base_frame_is_verbatim_fk():
    poses = {"left": _base_T(0.5, 0.3, 0.9), "right": _base_T(0.5, -0.3, 0.9),
             "head": _base_T(0.1, 0.0, 1.4)}
    base_pose = np.array([1.0, 2.0, np.pi / 2])
    s = build_state_vector(poses, base_pose, 0.25, 0.75, state_frame="base")
    assert s.shape == (len(STATE_AXES),) == (32,)
    # Base frame: EEF/head blocks are the FK poses untouched; base dims appended.
    np.testing.assert_allclose(s[0:3], [0.5, 0.3, 0.9], atol=1e-6)
    np.testing.assert_allclose(s[20:23], [0.1, 0.0, 1.4], atol=1e-6)
    assert s[9] == np.float32(0.25) and s[19] == np.float32(0.75)
    np.testing.assert_allclose(s[29:32], base_pose, atol=1e-6)


def test_build_state_vector_world_frame_composes_base_pose():
    poses = {"left": _base_T(0.3, 0.0, 0.5), "right": _base_T(0.4, 0.0, 0.5),
             "head": _base_T(0.2, 0.1, 1.1)}
    base_pose = np.array([1.0, 2.0, np.pi / 2])  # 90deg yaw: (x,y)->(-y,x)+t
    w = build_state_vector(poses, base_pose, 0.25, 0.75, state_frame="world")
    b = build_state_vector(poses, base_pose, 0.25, 0.75, state_frame="base")
    world_T_base = base_pose_to_mat(base_pose)
    # World frame: each EEF/head block == world_T_base @ base_T_block.
    for name, sl in (("left", slice(0, 9)), ("right", slice(10, 19)), ("head", slice(20, 29))):
        expected = mat_to_pos6d(world_T_base @ poses[name])
        np.testing.assert_allclose(w[sl], expected, atol=1e-6)
    # Left EEF at base (0.3,0,0.5) -> world (1-0, 2+0.3, 0.5) = (1.0, 2.3, 0.5).
    np.testing.assert_allclose(w[0:3], [1.0, 2.3, 0.5], atol=1e-6)
    # Grippers + base pose dims are identical in both frames (the world anchor).
    assert w[9] == b[9] and w[19] == b[19]
    np.testing.assert_allclose(w[29:32], b[29:32], atol=1e-7)
    np.testing.assert_allclose(w[29:32], base_pose, atol=1e-6)


def test_build_state_vector_rejects_bad_frame_and_base():
    poses = {"left": np.eye(4), "right": np.eye(4), "head": np.eye(4)}
    with pytest.raises(ValueError, match="state_frame"):
        build_state_vector(poses, np.zeros(3), 0.0, 0.0, state_frame="odom")
    with pytest.raises(ValueError, match="base_pose"):
        build_state_vector(poses, np.zeros(2), 0.0, 0.0, state_frame="base")
