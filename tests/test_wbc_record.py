"""Unit tests for omniteleop.wbc_record (faithful record + time-scaled replay).

Light: imports only omniteleop.wbc_record (-> VRJointData + numpy), no robot/IK/sim.
Run via the dexmate env with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from omniteleop.common.schemas import VRJointData
from omniteleop.wbc_record import (
    ReplaySource,
    StreamRecorder,
    TeleopStartGate,
    _decode_frames,
    _encode_frames,
    resolve_record_paths,
)


class _FakeClock:
    """Manually advanced monotonic clock for deterministic replay timing tests."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t


def _teleop_frame(
    ts: int,
    x: float,
    *,
    stage: str = "teleop",
    right_gripper: float = 0.7,
) -> VRJointData:
    left = (np.arange(16, dtype=float) * 0.1 + x).tolist()
    right = (np.arange(16, dtype=float) * 0.2 - x).tolist()
    head = (np.arange(16, dtype=float) * 0.05).tolist()
    return VRJointData(
        timestamp_ns=ts,
        left_gripper=0.3, right_gripper=right_gripper,
        chassis_vx=0.1, chassis_vy=-0.05, chassis_wz=0.2,
        estop=False, exit_requested=False,
        calib_stage=stage,
        left_ee_pose=left, right_ee_pose=right, head_ee_pose=head,
    )


def _static_frame(ts: int, *, home_requested: bool = False) -> VRJointData:
    # The leader's static stage: estop, empty EE/head poses, zero grippers/chassis.
    return VRJointData(
        timestamp_ns=ts,
        estop=True,
        calib_stage="static",
        home_requested=home_requested,
    )


def test_teleop_start_gate_latches_on_first_post_calibration_frame():
    gate = TeleopStartGate()

    assert not gate.started
    assert not gate.update(None)
    assert not gate.update(_static_frame(1))

    assert gate.update(_teleop_frame(2, 0.0, right_gripper=0.0))
    assert gate.started
    assert gate.update(_static_frame(3))


def test_encode_decode_round_trips_every_field():
    frames = [
        (0.00, _static_frame(1000, home_requested=True)),
        (0.10, _teleop_frame(1100, 0.5)),
        (0.20, _teleop_frame(1200, -0.25)),
    ]
    decoded = _decode_frames(_encode_frames(frames))
    assert len(decoded) == len(frames)
    for (m0, a), (m1, b) in zip(frames, decoded, strict=True):
        assert m0 == pytest.approx(m1)
        assert a.timestamp_ns == b.timestamp_ns
        assert a.calib_stage == b.calib_stage
        assert a.estop == b.estop
        assert a.exit_requested == b.exit_requested
        assert a.home_requested == b.home_requested
        assert (a.left_gripper, a.right_gripper) == pytest.approx(
            (b.left_gripper, b.right_gripper)
        )
        assert (a.chassis_vx, a.chassis_vy, a.chassis_wz) == pytest.approx(
            (b.chassis_vx, b.chassis_vy, b.chassis_wz)
        )
        # Empty stays empty; present poses survive exactly.
        assert bool(a.left_ee_pose) == bool(b.left_ee_pose)
        np.testing.assert_allclose(a.left_ee_pose, b.left_ee_pose)
        np.testing.assert_allclose(a.right_ee_pose, b.right_ee_pose)
        np.testing.assert_allclose(a.head_ee_pose, b.head_ee_pose)

def test_encode_rejects_non_wbc_joint_stream():
    # A non-empty joint field means a non-wbc leader (omni-vr) whose stream this path
    # does not model -- guard loudly rather than silently dropping the joints.
    bad = VRJointData(timestamp_ns=1, left_arm_pos=[0.0] * 7, calib_stage="whole_body")
    with pytest.raises(ValueError, match="wbc_vr_leader"):
        _encode_frames([(0.0, bad)])


def test_replay_releases_on_a_clock_stretched_by_one_over_speed():
    # recv_mono 0.0/0.1/0.2 at speed 0.25 -> scaled release offsets 0.0/0.4/0.8.
    frames = [
        (0.0, _teleop_frame(10, 0.0)),
        (0.1, _teleop_frame(20, 1.0)),
        (0.2, _teleop_frame(30, 2.0)),
    ]
    clk = _FakeClock(100.0)  # nonzero start to catch t0-anchoring bugs
    src = ReplaySource(frames, speed=0.25, clock=clk)
    src.start()
    assert src.total_duration == pytest.approx(0.8)

    clk.t = 100.0
    assert src.latest.timestamp_ns == 10
    assert src.segment_duration is None  # first frame: no prior segment

    clk.t = 100.0 + 0.39
    assert src.latest.timestamp_ns == 10  # frame 1 not due until +0.4
    assert not src.done

    clk.t = 100.0 + 0.41
    assert src.latest.timestamp_ns == 20
    assert src.segment_duration == pytest.approx(0.4)  # stretched 0.1s gap

    clk.t = 100.0 + 0.81
    assert src.latest.timestamp_ns == 30
    assert src.segment_duration == pytest.approx(0.4)
    assert src.done


def test_replay_speed_one_is_real_time():
    frames = [(0.0, _teleop_frame(1, 0.0)), (0.5, _teleop_frame(2, 1.0))]
    clk = _FakeClock()
    src = ReplaySource(frames, speed=1.0, clock=clk)
    src.start()
    clk.t = 0.5
    assert src.latest.timestamp_ns == 2
    assert src.segment_duration == pytest.approx(0.5)


def test_replay_rejects_bad_inputs():
    frames = [(0.0, _teleop_frame(1, 0.0))]
    for bad_speed in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="speed"):
            ReplaySource(frames, speed=bad_speed)
    with pytest.raises(ValueError, match="empty"):
        ReplaySource([], speed=0.25)


def test_recorder_write_and_from_hdf5_round_trip(tmp_path):
    rec = StreamRecorder(clock=_FakeClock(5.0))
    # Stamp two frames; the fake clock returns 5.0 for both (duration 0 is fine here).
    rec.append(_static_frame(1))
    rec.append(_teleop_frame(2, 0.5))
    assert len(rec) == 2
    path = str(tmp_path / "stream.hdf5")
    rec.write(path, meta={"speed_hint": 0.25})

    src = ReplaySource.from_hdf5(path, speed=0.25, clock=_FakeClock())
    vr = [src._vr[i] for i in range(len(src._vr))]  # noqa: SLF001 -- inspect decoded frames
    assert [f.timestamp_ns for f in vr] == [1, 2]
    assert vr[0].estop and not vr[1].estop
    np.testing.assert_allclose(vr[1].left_ee_pose, _teleop_frame(2, 0.5).left_ee_pose)


def test_resolve_record_paths_hdf5_only():
    output, hdf5 = resolve_record_paths(output=None, hdf5="/tmp/demo.hdf5", no_hdf5=False)
    assert output == "/tmp/demo.mp4"
    assert hdf5 == "/tmp/demo.hdf5"


def test_resolve_record_paths_output_only():
    output, hdf5 = resolve_record_paths(
        output="/tmp/demo.mp4", hdf5=None, no_hdf5=False,
    )
    assert output == "/tmp/demo.mp4"
    assert hdf5 == "/tmp/demo.hdf5"


def test_resolve_record_paths_defaults():
    output, hdf5 = resolve_record_paths(output=None, hdf5=None, no_hdf5=False)
    assert output == "/tmp/wbc_vr.mp4"
    assert hdf5 == "/tmp/wbc_vr.hdf5"


def test_resolve_record_paths_no_hdf5():
    output, hdf5 = resolve_record_paths(
        output=None, hdf5="/tmp/demo.hdf5", no_hdf5=True,
    )
    assert output == "/tmp/demo.mp4"
    assert hdf5 is None


# --- episode-take replay (ReplaySource.from_episode_hdf5) --------------------------


def _episode_pose(i: int, offset: float) -> np.ndarray:
    """A distinct homogeneous 4x4 per frame (rotation about z + a moving origin)."""
    theta = 0.05 * i + offset
    pose = np.eye(4)
    pose[:3, :3] = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    pose[:3, 3] = [0.1 * i + offset, 0.2 - 0.01 * i, 0.9]
    return pose


def _write_episode(path: str, n: int = 4, *, dt_ns: int = 100_000_000) -> dict:
    """Write a minimal wbc_vr_robot --record take; returns the arrays written."""
    arrays = {
        "timestamp_ns": np.arange(n, dtype=np.int64) * dt_ns + 1_700_000_000_000_000_000,
        "action/eef/left": np.stack([_episode_pose(i, 0.0) for i in range(n)]).astype(
            np.float32
        ),
        "action/eef/right": np.stack([_episode_pose(i, 0.5) for i in range(n)]).astype(
            np.float32
        ),
        "action/head": np.stack([_episode_pose(i, 1.0) for i in range(n)]).astype(np.float32),
        "action/gripper/left": np.linspace(0.0, 1.0, n).astype(np.float32),
        "action/gripper/right": np.linspace(1.0, 0.0, n).astype(np.float32),
        # Derived columns the replay must IGNORE (re-solved from the targets instead).
        "action/joint/torso": np.zeros((n, 3), dtype=np.float32),
        "obs/joint/torso": np.zeros((n, 3), dtype=np.float32),
    }
    with h5py.File(path, "w") as f:
        for key, val in arrays.items():
            f.create_dataset(key, data=val)
    return arrays


def test_from_episode_hdf5_replays_the_takes_own_solver_targets(tmp_path):
    path = str(tmp_path / "episode_8.hdf5")
    arrays = _write_episode(path, n=4)

    src = ReplaySource.from_episode_hdf5(path, clock=_FakeClock())
    vr = list(src._vr)  # noqa: SLF001 -- inspect decoded frames

    assert len(vr) == 4
    assert [f.timestamp_ns for f in vr] == arrays["timestamp_ns"].tolist()
    # Every frame is engaged teleop: an episode only contains engaged, post-calib frames.
    assert all(not f.estop and f.calib_stage == "teleop" for f in vr)
    assert all(not f.exit_requested and not f.home_requested for f in vr)
    # Targets round-trip verbatim as flattened row-major 4x4.
    for i, frame in enumerate(vr):
        np.testing.assert_allclose(
            frame.left_ee_pose, arrays["action/eef/left"][i].reshape(16), rtol=1e-6
        )
        np.testing.assert_allclose(
            frame.right_ee_pose, arrays["action/eef/right"][i].reshape(16), rtol=1e-6
        )
        np.testing.assert_allclose(
            frame.head_ee_pose, arrays["action/head"][i].reshape(16), rtol=1e-6
        )
    np.testing.assert_allclose(
        [f.left_gripper for f in vr], arrays["action/gripper/left"], rtol=1e-6
    )
    np.testing.assert_allclose(
        [f.right_gripper for f in vr], arrays["action/gripper/right"], rtol=1e-6
    )


def test_from_episode_hdf5_releases_at_the_recorded_cadence(tmp_path):
    path = str(tmp_path / "episode_8.hdf5")
    _write_episode(path, n=3, dt_ns=100_000_000)  # 10 Hz
    clk = _FakeClock()

    src = ReplaySource.from_episode_hdf5(path, clock=clk)
    src.start()
    assert src.latest is not None and src.segment_duration is None  # frame 0 at t=0
    clk.t = 0.05
    assert src.latest.timestamp_ns == 1_700_000_000_000_000_000  # still frame 0
    clk.t = 0.10
    assert src.latest.timestamp_ns == 1_700_000_000_100_000_000
    assert src.segment_duration == pytest.approx(0.1)
    assert not src.done
    clk.t = 0.20
    assert src.latest.timestamp_ns == 1_700_000_000_200_000_000
    assert src.done
    assert src.speed == 1.0  # episode replay is always real time
    assert src.total_duration == pytest.approx(0.2)


def test_from_episode_hdf5_time_scales_by_replay_speed(tmp_path):
    # The followers pass vr_teleop.replay_speed from their own config file, so a slow
    # replay must stretch the recorded cadence exactly as from_hdf5 does.
    path = str(tmp_path / "episode_8.hdf5")
    _write_episode(path, n=3, dt_ns=100_000_000)  # 10 Hz
    clk = _FakeClock()

    src = ReplaySource.from_episode_hdf5(path, 0.25, clock=clk)
    src.start()
    assert src.speed == 0.25
    assert src.total_duration == pytest.approx(0.8)  # 0.2 s take stretched 4x

    clk.t = 0.39
    assert src.latest.timestamp_ns == 1_700_000_000_000_000_000  # frame 1 not due yet
    clk.t = 0.41
    assert src.latest.timestamp_ns == 1_700_000_000_100_000_000
    assert src.segment_duration == pytest.approx(0.4)  # stretched 0.1 s gap
    assert not src.done


def test_from_episode_hdf5_rejects_a_leader_stream_recording(tmp_path):
    rec = StreamRecorder(clock=_FakeClock(5.0))
    rec.append(_teleop_frame(1, 0.0))
    rec.append(_teleop_frame(2, 0.5))
    path = str(tmp_path / "stream.hdf5")
    rec.write(path)

    with pytest.raises(ValueError, match="wbc_vr_stream/v1 leader recording"):
        ReplaySource.from_episode_hdf5(path)


def test_from_episode_hdf5_rejects_a_malformed_take(tmp_path):
    # Non-increasing timestamps: the release schedule would go backwards.
    path = str(tmp_path / "episode_bad_ts.hdf5")
    _write_episode(path, n=4)
    with h5py.File(path, "r+") as f:
        f["timestamp_ns"][2] = f["timestamp_ns"][1]
    with pytest.raises(ValueError, match="strictly increasing"):
        ReplaySource.from_episode_hdf5(path)

    # A pose whose bottom row is not [0, 0, 0, 1] is not a transform.
    path = str(tmp_path / "episode_bad_pose.hdf5")
    _write_episode(path, n=4)
    with h5py.File(path, "r+") as f:
        f["action/head"][1, 3, :] = [0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="homogeneous"):
        ReplaySource.from_episode_hdf5(path)

    # A NaN target must never reach the IK.
    path = str(tmp_path / "episode_nan.hdf5")
    _write_episode(path, n=4)
    with h5py.File(path, "r+") as f:
        f["action/eef/left"][2, 0, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        ReplaySource.from_episode_hdf5(path)


def test_from_episode_hdf5_rejects_a_file_missing_action_targets(tmp_path):
    path = str(tmp_path / "not_an_episode.hdf5")
    with h5py.File(path, "w") as f:
        f.create_dataset("timestamp_ns", data=np.arange(3, dtype=np.int64))
    with pytest.raises(ValueError, match="action/eef/left"):
        ReplaySource.from_episode_hdf5(path)
