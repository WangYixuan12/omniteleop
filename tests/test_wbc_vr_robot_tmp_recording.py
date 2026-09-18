# ruff: noqa: SLF001
"""Regression tests for temporary follower recording metadata and save errors."""

from __future__ import annotations

import importlib.util
import time
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

from omniteleop.common.streaming_recorder import StreamingEpisodeRecorder


def _load_wbc_vr_robot_tmp():
    path = Path(__file__).resolve().parents[1] / "scripts" / "wbc_vr_robot_tmp.py"
    spec = importlib.util.spec_from_file_location("wbc_vr_robot_tmp_recording_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _wait(recorder, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while recorder.saving and time.time() < deadline:
        time.sleep(0.01)
    assert not recorder.saving, "recorder still saving after timeout"


def test_frozen_right_arm_metadata_saves_with_streaming_recorder(tmp_path):
    """The exact temporary-follower metadata must be HDF5-compatible."""
    mod = _load_wbc_vr_robot_tmp()
    driver = object.__new__(mod.HardwareDriver)
    driver.cfg = types.SimpleNamespace(dead_joints=("R_arm_j1", "R_arm_j2"))
    metadata = mod.HardwareDriver._recording_control_metadata(driver)

    recorder = StreamingEpisodeRecorder(str(tmp_path))
    recorder.set_static({"meta": metadata})
    recorder.start()
    recorder.record({"timestamp_ns": np.int64(1)})
    path = recorder.stop()
    _wait(recorder)

    assert recorder.last_save_error is None, recorder.last_save_error
    assert path is not None
    with h5py.File(path, "r") as h5file:
        assert h5file["meta/right_arm_mode"][()].item() == b"frozen_nominal"
        assert h5file["meta/dead_joints"][()].item() == b"R_arm_j1,R_arm_j2"


def test_stop_recording_episode_raises_async_save_error(capsys):
    """A failed writer must not be reported or debug-flushed as a saved episode."""
    mod = _load_wbc_vr_robot_tmp()
    driver = object.__new__(mod.HardwareDriver)
    save_error = TypeError("No conversion path for dtype")
    episode = types.SimpleNamespace(
        recording=True,
        saving=False,
        episode_id=1,
        last_save_error=save_error,
        num_frames=lambda: 3,
    )

    def stop():
        episode.recording = False
        episode.episode_id += 1
        return "/tmp/episode_1.hdf5"

    episode.stop = stop
    driver._episode = episode
    driver.args = types.SimpleNamespace(head_right_rgb=False)
    flushed = []
    driver._traj = types.SimpleNamespace(flush=flushed.append)

    with pytest.raises(RuntimeError, match="episode save failed") as exc_info:
        mod.HardwareDriver.stop_recording_episode(driver)

    assert exc_info.value.__cause__ is save_error
    assert "episode saved" not in capsys.readouterr().out
    assert flushed == []
