"""Unit tests for scripts/wbc_policy_rollout.py's pure action splitting.

The WBC policy action is WORLD-frame (verbatim ik.solve targets, PLAN.md), so
``split_policy_action`` must return the encoded poses unchanged -- no base
composition of any kind -- plus thresholded binary gripper commands.

numpy-only; run via the dexmate env with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from omniteleop.wbc_policy_format import JOYSTICK_ACTION_AXES, mat_to_pos6d


def _load_rollout():
    path = Path(__file__).resolve().parents[1] / "scripts" / "wbc_policy_rollout.py"
    name = "wbc_policy_rollout_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module  # dataclass string-annotation resolution needs this
    spec.loader.exec_module(module)
    return module


def test_split_policy_action_returns_world_targets_verbatim():
    mod = _load_rollout()
    world_T_left = np.eye(4)
    world_T_left[:3, 3] = [1.0, 2.3, 0.5]
    world_T_right = np.eye(4)
    world_T_right[:3, 3] = [1.0, 2.4, 0.5]
    world_T_head = np.eye(4)
    c, s = np.cos(0.4), np.sin(0.4)
    world_T_head[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    world_T_head[:3, 3] = [0.9, 2.2, 1.1]
    action = np.concatenate([
        mat_to_pos6d(world_T_left), [0.0],
        mat_to_pos6d(world_T_right), [1.0],
        mat_to_pos6d(world_T_head),
    ]).astype(np.float32)

    targets = mod.split_policy_action(action)

    np.testing.assert_allclose(targets["left"][:3, 3], [1.0, 2.3, 0.5], atol=1e-6)
    np.testing.assert_allclose(targets["right"][:3, 3], [1.0, 2.4, 0.5], atol=1e-6)
    np.testing.assert_allclose(targets["head"][:3, 3], [0.9, 2.2, 1.1], atol=1e-6)
    # Poses come back VERBATIM (float32 round-trip) -- world frame in, world frame out.
    np.testing.assert_allclose(targets["left"], world_T_left, atol=1e-6)
    np.testing.assert_allclose(targets["head"], world_T_head, atol=1e-6)
    assert targets["left_gripper"] == np.float32(0.0)
    assert targets["right_gripper"] == np.float32(1.0)


def test_split_policy_action_binarizes_gripper_predictions_at_threshold():
    mod = _load_rollout()
    action = np.concatenate([
        mat_to_pos6d(np.eye(4)), [0.499],
        mat_to_pos6d(np.eye(4)), [0.5],
        mat_to_pos6d(np.eye(4)),
    ]).astype(np.float32)

    targets = mod.split_policy_action(action)

    assert targets["left_gripper"] == np.float32(0.0)
    assert targets["right_gripper"] == np.float32(1.0)


def test_split_policy_action_reorthonormalizes_noisy_rotation():
    mod = _load_rollout()
    action = np.concatenate([
        mat_to_pos6d(np.eye(4)), [0.0],
        mat_to_pos6d(np.eye(4)), [0.0],
        mat_to_pos6d(np.eye(4)),
    ]).astype(np.float32)
    action[3:9] *= 1.07  # a raw policy output is never exactly orthonormal
    targets = mod.split_policy_action(action)
    R = targets["left"][:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-7)
    assert np.linalg.det(R) > 0.0


def test_split_policy_action_rejects_bad_input():
    mod = _load_rollout()
    with pytest.raises(ValueError, match="29-D or 32-D"):
        mod.split_policy_action(np.zeros(31, dtype=np.float32))
    bad = np.zeros(29, dtype=np.float32)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        mod.split_policy_action(bad)


def test_split_joystick_policy_action_keeps_current_base_chassis_command():
    mod = _load_rollout()
    action = np.concatenate([
        mat_to_pos6d(np.eye(4)), [0.0],
        mat_to_pos6d(np.eye(4)), [1.0],
        mat_to_pos6d(np.eye(4)),
        [0.2, -0.1, 0.35],
    ]).astype(np.float32)
    assert action.shape == (len(JOYSTICK_ACTION_AXES),)

    targets = mod.split_policy_action(action)

    np.testing.assert_allclose(targets["chassis"], [0.2, -0.1, 0.35])


def _chunk_row(x: float, grip: float) -> np.ndarray:
    pose = np.eye(4)
    pose[0, 3] = x
    return np.concatenate([
        mat_to_pos6d(pose), [grip], mat_to_pos6d(pose), [grip], mat_to_pos6d(pose)
    ]).astype(np.float32)


def _joystick_chunk_row(x: float, grip: float, chassis) -> np.ndarray:
    return np.concatenate([_chunk_row(x, grip), np.asarray(chassis, np.float32)])


def test_scheduled_actions_from_chunk_decodes_rows_with_timestamps():
    mod = _load_rollout()
    chunk = np.stack([_chunk_row(0.1, 0.0), _chunk_row(0.2, 1.0)])
    ts = np.array([5.0, 5.1])
    actions = mod.scheduled_actions_from_chunk(chunk, ts)
    assert len(actions) == 2
    assert actions[0].timestamp == 5.0 and actions[1].timestamp == 5.1
    np.testing.assert_allclose(actions[0].left[0, 3], 0.1, atol=1e-6)
    np.testing.assert_allclose(actions[1].head[0, 3], 0.2, atol=1e-6)
    assert float(actions[0].grip_left) == 0.0
    assert float(actions[1].grip_right) == 1.0
    # Rows go through split_policy_action: rotations come back orthonormal.
    R = actions[0].left[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-7)


def test_scheduled_joystick_actions_carry_chassis_command():
    mod = _load_rollout()
    chunk = np.stack([
        _joystick_chunk_row(0.1, 0.0, [0.2, 0.0, 0.0]),
        _joystick_chunk_row(0.2, 1.0, [0.0, -0.1, 0.0]),
    ])
    actions = mod.scheduled_actions_from_chunk(chunk, np.array([5.0, 5.1]))
    np.testing.assert_allclose(actions[0].chassis, [0.2, 0.0, 0.0])
    np.testing.assert_allclose(actions[1].chassis, [0.0, -0.1, 0.0])

    halfway = mod._interpolate_scheduled(actions[0], actions[1], 5.05)
    # Poses interpolate, but joystick intent is a sampled command and stays ZOH.
    assert halfway.left[0, 3] == pytest.approx(0.15)
    np.testing.assert_allclose(halfway.chassis, [0.2, 0.0, 0.0])


def test_scheduled_actions_from_chunk_rejects_bad_input():
    mod = _load_rollout()
    row = _chunk_row(0.0, 0.0)
    with pytest.raises(ValueError, match="one timestamp per frame"):
        mod.scheduled_actions_from_chunk(np.stack([row, row]), np.array([1.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        mod.scheduled_actions_from_chunk(np.stack([row, row]), np.array([1.1, 1.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        mod.scheduled_actions_from_chunk(np.stack([row, row]), np.array([1.0, np.nan]))
    bad = np.stack([row, row])
    bad[1, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        mod.scheduled_actions_from_chunk(bad, np.array([1.0, 1.1]))


def test_chunk_schedule_preserves_training_action_offset():
    mod = _load_rollout()

    timestamps = mod.chunk_schedule_timestamps(
        observation_time=5.0,
        frame_count=3,
        dataset_dt=0.1,
        action_offset_frames=2,
    )

    np.testing.assert_allclose(timestamps, [5.2, 5.3, 5.4], atol=1e-12)
    np.testing.assert_allclose(
        mod.chunk_schedule_timestamps(
            observation_time=5.0,
            frame_count=3,
            dataset_dt=0.1,
            action_offset_frames=0,
        ),
        [5.0, 5.1, 5.2],
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frame_count": 0}, "frame_count"),
        ({"dataset_dt": 0.0}, "dataset_dt"),
        ({"action_offset_frames": -1}, "action_offset_frames"),
    ],
)
def test_chunk_schedule_rejects_bad_contract(kwargs, message):
    mod = _load_rollout()
    values = {
        "observation_time": 5.0,
        "frame_count": 2,
        "dataset_dt": 0.1,
        "action_offset_frames": 0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        mod.chunk_schedule_timestamps(**values)
