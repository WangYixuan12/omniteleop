"""Review decisions override machine flags without modifying the recording."""

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from omniteleop.wrist_corruption import annotate_episode, make_annotations, source_identity
from omniteleop.wrist_review import (
    candidate_indices,
    load_review,
    recover_review,
    save_decision,
)


def fixture_episode(tmp_path):
    path = tmp_path / "episode_3.hdf5"
    with h5py.File(path, "w") as f:
        f.attrs.update(complete=True, n_frames=5)
        f["timestamp_ns"] = np.arange(5, dtype=np.int64)
        f["action/test"] = np.arange(5)
        for key in ("left_wrist_rgb", "left_wrist_right_rgb"):
            f[f"obs/images/{key}"] = np.zeros((5, 36, 64, 3), dtype=np.uint8)
    arrays = make_annotations(
        np.array([0, 0.8, 0.2, 0, 0.1]), [4], {"threshold": 0.35, "review_threshold": 0.1}
    )
    annotate_episode(path, source_identity(path), arrays, {})
    # Simulate a legacy recording: future annotations do not create this field.
    with h5py.File(path, "r+") as f:
        f["quality/left_wrist/manual_corrupted"] = np.array([False, False, False, False, True])
    return path


def test_human_clean_overrides_flags_and_reset_restores_original(tmp_path):
    path = fixture_episode(tmp_path)
    state = load_review(path)
    assert candidate_indices(state, "review", True).tolist() == [2]
    revision = save_decision(path, 1, 0, state["revision"])
    state = load_review(path)
    assert not state["corrupted"][1]
    assert state["auto_corrupted"][1]
    assert state["review_decision"][1] == 0
    assert candidate_indices(state, "auto", True).tolist() == []
    with pytest.raises(ValueError, match="another reviewer"):
        save_decision(path, 2, 1, 0)
    revision = save_decision(path, 2, 1, revision)
    state = load_review(path)
    assert state["corrupted"][2] and state["review_candidate"][2]
    assert candidate_indices(state, "review", True).tolist() == []
    revision = save_decision(path, 1, -1, revision)
    assert load_review(path)["corrupted"][1]
    revision = save_decision(path, 4, 0, revision)
    assert not load_review(path)["corrupted"][4]
    assert load_review(path)["manual_corrupted"][4]
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["action/test"][:], np.arange(5))
        assert not f["obs/images/left_wrist_rgb"][:].any()
        assert len(f["quality/left_wrist/review_history"]) == 4


def test_pending_write_is_recoverable(tmp_path):
    path = fixture_episode(tmp_path)
    event = {
        "frame": 2,
        "decision": 1,
        "previous": -1,
        "revision": 1,
        "time_ns": 123,
        "history_index": 0,
    }
    with h5py.File(path, "r+") as f:
        f["quality/left_wrist"].attrs["pending_review_json"] = json.dumps(event)
        # Simulate interruption after extending the history but before writing its row.
        f["quality/left_wrist"].create_dataset(
            "review_history", shape=(1,), maxshape=(None,), dtype=h5py.string_dtype("utf-8")
        )
    with pytest.raises(ValueError, match="Interrupted"):
        load_review(path)
    assert recover_review(path)
    assert not recover_review(path)
    state = load_review(path)
    assert state["corrupted"][2] and state["revision"] == 1


def test_batch_rerun_preserves_human_decisions(tmp_path):
    path = fixture_episode(tmp_path)
    save_decision(path, 1, 0, 0)
    before = path.stat().st_mtime_ns
    repo = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/diagnostics/label_wrist_corruption.py"),
            "--data-dir",
            str(tmp_path),
            "--work-dir",
            str(tmp_path / "work"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert path.stat().st_mtime_ns == before
    assert not load_review(path)["corrupted"][1]


def test_batch_labels_generic_filename_and_skips_incomplete(tmp_path):
    source = fixture_episode(tmp_path)
    path = source.with_name("take_a.hdf5")
    source.rename(path)
    with h5py.File(path, "r+") as f:
        del f["quality"]
    with h5py.File(tmp_path / "active.hdf5", "w") as f:
        f.attrs["complete"] = False
    repo = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/diagnostics/label_wrist_corruption.py"),
            "--data-dir",
            str(tmp_path),
            "--work-dir",
            str(tmp_path / "work"),
            "--workers",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert load_review(path)["score"].shape == (5,)
    with h5py.File(path, "r") as f:
        assert "manual_corrupted" not in f["quality/left_wrist"]
    with h5py.File(tmp_path / "active.hdf5", "r") as f:
        assert "quality" not in f
