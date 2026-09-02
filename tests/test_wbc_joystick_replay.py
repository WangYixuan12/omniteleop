"""Joystick replay distinguishes raw leader streams from filtered episode actions."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from omniteleop.common.schemas import VRJointData
from omniteleop.wbc_record import ReplaySource, StreamRecorder


def _vr(timestamp_ns: int, chassis) -> VRJointData:
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
    recorder.append(_vr(10, [0.2, 0.0, 0.0]))
    recorder.append(_vr(20, [0.0, -0.1, 0.0]))
    path = tmp_path / "leader_stream.hdf5"
    recorder.write(str(path))

    source = ReplaySource.from_joystick_hdf5(str(path))

    assert not source.head_targets_pre_filtered
    assert source.replay_kind == "leader_stream"
    assert source._vr[0].chassis_vx == pytest.approx(0.2)  # noqa: SLF001


def _write_joystick_episode(path, *, include_intent: bool = True):
    n = 2
    poses = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    with h5py.File(path, "w") as f:
        f["meta/schema"] = np.asarray(b"omniteleop_joystick_mobile_raw/v1")
        f["meta/control_mode"] = np.asarray(b"joystick")
        f["meta/policy_action_schema"] = np.asarray(
            b"omniteleop_wbc_joystick_action/v1"
        )
        for key in ("action_target_frame", "eef_target_frame", "head_target_frame"):
            f[f"meta/{key}"] = np.asarray(b"current_base")
        f["timestamp_ns"] = np.array([1_000_000_000, 1_100_000_000], np.int64)
        f["action/eef/left"] = poses
        f["action/eef/right"] = poses
        f["action/head"] = poses
        f["action/gripper/left"] = np.zeros(n, np.float32)
        f["action/gripper/right"] = np.ones(n, np.float32)
        if include_intent:
            f["action/chassis/intent_body"] = np.array(
                [[0.2, 0.0, 0.0], [0.0, -0.1, 0.3]], np.float32
            )


def test_joystick_episode_replay_restores_intent_and_skips_second_head_filter(tmp_path):
    path = tmp_path / "episode_0.hdf5"
    _write_joystick_episode(path)

    source = ReplaySource.from_joystick_hdf5(str(path))

    assert source.head_targets_pre_filtered
    assert source.replay_kind == "joystick_episode"
    np.testing.assert_allclose(
        [source._vr[1].chassis_vx, source._vr[1].chassis_vy, source._vr[1].chassis_wz],  # noqa: SLF001
        [0.0, -0.1, 0.3],
    )


def test_joystick_episode_replay_refuses_to_guess_intent_from_applied_twist(tmp_path):
    path = tmp_path / "episode_0.hdf5"
    _write_joystick_episode(path, include_intent=False)

    with pytest.raises(ValueError, match="action/chassis/intent_body"):
        ReplaySource.from_joystick_hdf5(str(path))
