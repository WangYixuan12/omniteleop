"""Behavioral tests for detection and adding quality masks without altering recordings."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from omniteleop.wrist_corruption import (
    ANNOTATION_KEY,
    FEATURE_VERSION,
    IMAGE_KEYS,
    annotate_episode,
    forest_scores,
    frame_features,
    make_annotations,
    read_labels,
    source_identity,
)


def episode(tmp_path):
    path = tmp_path / "episode_3.hdf5"
    with h5py.File(path, "w") as f:
        f.attrs.update(complete=True, n_frames=5, user_note="preserve me")
        f["timestamp_ns"] = np.arange(5, dtype=np.int64) * 100_000_000
        f["action/joint/left_arm"] = np.arange(35).reshape(5, 7)
        f["obs/images/intrinsic"] = np.eye(3)
        f["meta/static_same_length_as_episode"] = np.arange(5)
        f["meta/string"] = b"original"
        for key in IMAGE_KEYS:
            f[key] = np.arange(5 * 36 * 64 * 3, dtype=np.uint8).reshape(5, 36, 64, 3)
            f[key].attrs["color"] = "RGB"
    return path


def snapshot(path):
    data, attrs = {}, {}
    with h5py.File(path, "r") as f:
        attrs["/"] = dict(f.attrs)

        def visit(name, obj):
            if not name.startswith("quality"):
                attrs[name] = dict(obj.attrs)
                if isinstance(obj, h5py.Dataset):
                    data[name] = obj[()]

        f.visititems(visit)
    return data, attrs


def test_annotation_preserves_all_original_pixels_actions_timestamps_and_metadata(tmp_path):
    path = episode(tmp_path)
    identity = source_identity(path)
    before, attributes = snapshot(path)
    model = {"threshold": 0.5, "review_threshold": 0.15}
    arrays = make_annotations(np.array([0, 0.8, 0.2, 0, 0.1]), [4], model)
    provenance = {"model": "test", "source": identity}
    assert annotate_episode(path, identity, arrays, provenance) == "written"
    after, new_attributes = snapshot(path)
    assert attributes == new_attributes
    assert before.keys() == after.keys()
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    with h5py.File(path, "r") as f:
        g = f[ANNOTATION_KEY]
        np.testing.assert_array_equal(g["corrupted"], [False, True, False, False, True])
        np.testing.assert_array_equal(g["review_candidate"], [False, False, True, False, False])
        assert g["corrupted"].dtype == np.bool_
        assert json.loads(g.attrs["provenance_json"]) == provenance
    assert annotate_episode(path, identity, arrays, provenance) == "already_present"
    arrays = make_annotations(np.array([0, 0.8, 0.9, 0, 0.1]), [4], model)
    with pytest.raises(ValueError, match="different annotations"):
        annotate_episode(path, identity, arrays, provenance)


def test_changed_or_partial_sources_are_rejected(tmp_path):
    path = episode(tmp_path)
    identity = source_identity(path)
    arrays = make_annotations(np.zeros(5), [], {"threshold": 0.5, "review_threshold": 0.15})
    with h5py.File(path, "r+") as f:
        f["timestamp_ns"][4] += 1
    with pytest.raises(ValueError, match="changed since scan"):
        annotate_episode(path, identity, arrays, {})
    with h5py.File(path, "r") as f:
        assert "quality" not in f
    with h5py.File(path, "r+") as f:
        f.attrs["complete"] = False
    with pytest.raises(ValueError, match="not marked complete"):
        source_identity(path)
    partial = path.with_suffix(".hdf5.partial")
    path.rename(partial)
    with pytest.raises(ValueError, match="recently modified partial"):
        source_identity(partial)
    old = time.time() - 20
    os.utime(partial, (old, old))
    identity = source_identity(partial)
    assert annotate_episode(partial, identity, arrays, {}) == "written"
    with h5py.File(partial, "r") as f:
        assert not f.attrs["complete"]
        assert len(f["timestamp_ns"]) == 5
    with h5py.File(partial, "r+") as f:
        del f["action/joint/left_arm"]
        f["action/joint/left_arm"] = np.zeros((4, 7))
    with pytest.raises(ValueError, match="inconsistent partial"):
        source_identity(partial)


def test_parse_manual_labels_and_validate_bounds(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text("ep3:\n210\n210\n416\n# comment\nepisode_5:\n2\n")
    assert read_labels(path) == {"episode_3": [210, 416], "episode_5": [2]}
    assert read_labels(path, 1)["episode_5"] == [1]
    model = {"threshold": 0.5, "review_threshold": 0.15}
    with pytest.raises(ValueError, match="out of range"):
        make_annotations(np.zeros(5), [5], model)
    with pytest.raises(ValueError, match="finite"):
        make_annotations(np.array([np.nan]), [], model)
    path.write_text("210\n")
    with pytest.raises(ValueError, match="expected episode header"):
        read_labels(path)


def test_json_forest_routes_samples_and_rejects_wrong_feature_schema():
    model = {
        "feature_version": FEATURE_VERSION,
        "feature_count": 1,
        "trees": [
            {
                "left": [1, -1, -1],
                "right": [2, -1, -1],
                "feature": [0, -2, -2],
                "threshold": [2, -2, -2],
                "positive": [0.5, 0.1, 0.9],
            }
        ],
    }
    np.testing.assert_allclose(forest_scores(np.array([[1], [2], [3]]), model), [0.1, 0.1, 0.9])
    with pytest.raises(ValueError, match="schema"):
        forest_scores(np.zeros((3, 2)), model)


def test_features_are_finite_at_episode_boundaries_and_model_detects_synthetic_strip():
    rng = np.random.default_rng(4)
    texture = rng.integers(0, 256, (180, 320), dtype=np.uint8)
    import cv2

    texture = cv2.GaussianBlur(texture, (9, 9), 2)
    frames = np.repeat(np.stack([texture, texture])[None], 7, axis=0)
    clean = np.array([frame_features(frames, i) for i in (0, 3, 6)])
    assert np.isfinite(clean).all()
    assert np.isfinite(frame_features(frames[:3], 1)).all()
    frames[3, :, 95:] = np.roll(frames[3, :, 95:], 130, axis=-1)
    damaged = np.array(frame_features(frames, 3))
    assert np.isfinite(damaged).all()
    # The full-width patch matcher should find a large new displacement in both
    # eyes, even though the reference images themselves contain normal texture.
    assert np.linalg.norm(damaged - clean[1]) > 100
    model_path = Path(__file__).resolve().parents[1] / "configs/camera_quality/left_wrist_v1.json"
    if model_path.exists():
        model = json.loads(model_path.read_text())
        scores = forest_scores(np.stack([clean[1], damaged]), model)
        assert scores[1] > scores[0]


def test_cli_scan_is_read_only_and_apply_adds_a_matching_mask(tmp_path):
    path = episode(tmp_path)
    identity = source_identity(path)
    model = {
        "feature_version": FEATURE_VERSION,
        "feature_count": 356,
        "threshold": 0.35,
        "review_threshold": 0.1,
        "trees": [
            {"left": [-1], "right": [-1], "feature": [-2], "threshold": [-2], "positive": [0.8]}
        ],
    }
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model))
    report = tmp_path / "report"
    script = (
        Path(__file__).resolve().parents[1] / "scripts/diagnostics/annotate_wrist_corruption.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "scan",
            "--data-dir",
            str(tmp_path),
            "--model",
            str(model_path),
            "--report-dir",
            str(report),
            "--workers",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert source_identity(path) == identity
    manifest = json.loads((report / "manifest.json").read_text())
    assert manifest["scan_complete"] is True
    subprocess.run(
        [
            sys.executable,
            str(script),
            "apply",
            "--data-dir",
            str(tmp_path),
            "--report-dir",
            str(report),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with h5py.File(path, "r") as f:
        assert f[f"{ANNOTATION_KEY}/corrupted"][:].all()
        assert len(f["timestamp_ns"]) == 5
