"""Contract tests for the WBC point-cloud zarr porter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from omniteleop.wbc_artifacts import load_dino_artifact, validate_mask_artifact
from omniteleop.wbc_policy_format import (
    JOYSTICK_ACTION_AXES,
    JOYSTICK_POLICY_ACTION_SCHEMA,
)


def _load_porter():
    path = Path(__file__).resolve().parents[1] / "scripts" / "port_wbc_mobile_zarr.py"
    spec = importlib.util.spec_from_file_location("port_wbc_mobile_zarr_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def porter():
    return _load_porter()


def _write_positions(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        positions=np.array([[0.5, 0.1, 0.8], [0.7, -0.2, 0.6]], np.float32),
        order=np.array([1, 0], np.int64),
        valid=np.ones(2, np.float32),
        scene_diff_obj_ids=np.array([11, 23], np.int64),
        frame=np.asarray("world"),
    )
    return path


def _write_dino(path: Path, base: Path, *, feats=None, valid=None,
                input_scale="[0,1]") -> Path:
    if feats is None:
        feats = np.ones((2, 1280), np.float32)
    if valid is None:
        valid = np.ones(2, np.float32)
    with np.load(base, allow_pickle=False) as z:
        order = z["order"]
        obj_ids = z["scene_diff_obj_ids"]
    np.savez(
        path,
        dino_region_feats=feats,
        dino_valid=valid,
        base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        scene_diff_obj_ids=obj_ids,
        order=order,
        dino_model="dinov3_vith16plus",
        dino_source="original/rgb_scenediff.png [0,1] + video_1 RLEs",
        dino_input_scale=input_scale,
        created="2026-07-19T00:00:00+00:00",
    )
    return path


@pytest.mark.parametrize(
    ("feats", "valid", "match"),
    [
        (np.ones((2, 4), np.float32), np.ones(2, np.float32), "1280"),
        (np.ones((2, 1280), np.float64), np.ones(2, np.float32), "float32"),
        (np.vstack([np.zeros((1, 1280)), np.ones((1, 1280))]).astype(np.float32),
         np.ones(2, np.float32), "zero"),
        (np.ones((2, 1280), np.float32), np.asarray(1.0, np.float32), "dino_valid"),
    ],
)
def test_dino_loader_rejects_malformed_sidecars(tmp_path, feats, valid, match):
    base = _write_positions(tmp_path / "raw" / "episode_3.npz")
    sidecar = _write_dino(tmp_path / "raw" / "episode_3_dino.npz", base,
                          feats=feats, valid=valid)
    with pytest.raises(ValueError, match=match):
        load_dino_artifact(base, sidecar, object_nums=2)


def test_dino_loader_rejects_unstamped_input_scale(tmp_path):
    base = _write_positions(tmp_path / "raw" / "episode_3.npz")
    sidecar = _write_dino(
        tmp_path / "raw" / "episode_3_dino.npz", base, input_scale="[0,255]"
    )
    with pytest.raises(ValueError, match="dino_input_scale"):
        load_dino_artifact(base, sidecar, object_nums=2)


def _mask_arrays(frames: int, height: int, width: int):
    masks = np.zeros((frames, height, width), np.uint8)
    masks[:, 0:4, 0:4] = 1
    masks[:, 5:9, 5:9] = 2
    areas = np.full((frames, 2), 16, np.int32)
    largest = np.ones((frames, 2), np.float32)
    components = np.ones((frames, 2), np.int16)
    return masks, areas, largest, components


def _write_mask(path: Path, raw_path: Path, positions_path: Path, *, tracker=None,
                rgb_sha256=None, bad_areas=False, include_schema=True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw_path, "r") as raw:
        rgb = np.asarray(raw["obs/images/head_left_rgb"])
    masks, areas, largest, components = _mask_arrays(*rgb.shape[:3])
    if bad_areas:
        areas[0, 0] += 1
    with np.load(positions_path, allow_pickle=False) as z:
        order = z["order"]
        task_to_obj = z["scene_diff_obj_ids"][order]
    fields = dict(
        masks=masks,
        qc_areas=areas,
        qc_largest_frac=largest,
        qc_n_components=components,
        split="raw",
        episode_number=np.int64(3),
        raw_hdf5=str(raw_path.resolve()),
        tracked_rgb_sha256=(rgb_sha256 or hashlib.sha256(rgb.tobytes()).hexdigest()),
        rgb_key="obs/images/head_left_rgb",
        n_frames=np.int64(rgb.shape[0]),
        tracker=tracker or "sam3.1_multiplex",
        image_size=np.int64(1008),
        legend="0=background 1=box(src) 2=cloth(dst)",
        task_to_obj=task_to_obj.astype(np.int64),
        order=order.astype(np.int64),
        seed_areas=np.array([16, 16], np.int64),
        positions_npz=str(positions_path.resolve()),
        positions_sha256=hashlib.sha256(positions_path.read_bytes()).hexdigest(),
        seed_frame0="tracker_render (bilinear+antialias+0.5, IoU-gated vs native seed)",
        frame0_iou=np.ones(2, np.float32),
        created="2026-07-19T00:00:00+00:00",
    )
    if include_schema:
        fields["artifact_schema"] = "dexmate_wbc_masks_v1"
    np.savez_compressed(path, **fields)
    return path


def _raw_rgb(path: Path, frames=6, height=10, width=10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as raw:
        raw["obs/images/head_left_rgb"] = np.zeros(
            (frames, height, width, 3), dtype=np.uint8)
    return path


def test_mask_loader_enforces_tracker_qc_source_and_slot_contract(tmp_path):
    raw = _raw_rgb(tmp_path / "raw_data" / "episode_3.hdf5")
    positions = _write_positions(tmp_path / "positions" / "raw" / "episode_3.npz")
    masks_dir = tmp_path / "masks"
    mask_path = _write_mask(masks_dir / "raw" / "episode_3.npz", raw, positions)

    artifact = validate_mask_artifact(
        mask_path, source="raw", episode_number=3,
        raw_hdf5=raw, positions_npz=positions)
    assert artifact.load_masks().shape == (6, 10, 10)

    _write_mask(mask_path, raw, positions, tracker="xmem")
    with pytest.raises(RuntimeError, match="changed after validation"):
        artifact.load_masks()
    with pytest.raises(ValueError, match="tracker"):
        validate_mask_artifact(
            mask_path, source="raw", episode_number=3,
            raw_hdf5=raw, positions_npz=positions)

    _write_mask(mask_path, raw, positions, rgb_sha256="0" * 64)
    with pytest.raises(ValueError, match="tracked_rgb_sha256"):
        validate_mask_artifact(
            mask_path, source="raw", episode_number=3,
            raw_hdf5=raw, positions_npz=positions)

    _write_mask(mask_path, raw, positions, include_schema=False)
    with pytest.raises(ValueError, match="artifact_schema"):
        validate_mask_artifact(
            mask_path, source="raw", episode_number=3,
            raw_hdf5=raw, positions_npz=positions)

    _write_mask(mask_path, raw, positions, bad_areas=True)
    with pytest.raises(ValueError, match="qc_areas"):
        validate_mask_artifact(
            mask_path, source="raw", episode_number=3,
            raw_hdf5=raw, positions_npz=positions)


def _convert_args(raw_dir: Path, out_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        raw_dir=raw_dir, out_root=out_root, name="txn", fps=10,
        num_points=32, pool_size=32, chunk_frames=2, state_frame="world",
        crop_min=(-10.0, -10.0, -10.0), crop_max=(10.0, 10.0, 10.0),
        min_depth=0.05, max_depth=8.0, seed=0, include_recovery_data=None,
        split_csv=None, positions_dir=None, object_nums=2, masks_dir=None,
        dino=False,
    )


@pytest.mark.parametrize("existing_name", ["txn_train.zarr", "txn_train.meta.json"])
def test_existing_output_aborts_before_conversion_and_is_preserved(
        porter, tmp_path, existing_name):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _raw_rgb(raw_dir / "episode_0.hdf5")  # intentionally not a complete recorder take
    out_root = tmp_path / "out"
    existing = out_root / existing_name
    if existing.suffix == ".zarr":
        existing.mkdir(parents=True)
        sentinel = existing / "sentinel"
    else:
        existing.parent.mkdir(parents=True)
        sentinel = existing
    sentinel.write_text("old generation")

    with pytest.raises(RuntimeError, match=r"Delete .* manually"):
        porter.convert(_convert_args(raw_dir, out_root))

    assert sentinel.read_text() == "old generation"


def test_failed_conversion_removes_new_output_root(porter, tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _raw_rgb(raw_dir / "episode_0.hdf5")  # intentionally incomplete
    out_root = tmp_path / "new_output"

    with pytest.raises(RuntimeError):
        porter.convert(_convert_args(raw_dir, out_root))

    assert not out_root.exists()


def _camera_take(path, intrinsic=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as raw:
        raw["obs/images/head_left_rgb"] = np.zeros((2, 4, 5, 3), np.uint8)
        raw["obs/images/head_depth"] = np.zeros((2, 4, 5), np.uint16)
        raw["obs/images/intrinsic"] = (
            np.eye(3, dtype=np.float32) if intrinsic is None else intrinsic
        )
    return {"path": path}


def test_camera_contract_records_exact_calibration(porter, tmp_path):
    intrinsic = np.array([[100.0, 0, 2.0], [0, 101.0, 1.5], [0, 0, 1]], np.float32)
    work = [
        _camera_take(tmp_path / "episode_0.hdf5", intrinsic),
        _camera_take(tmp_path / "episode_1.hdf5", intrinsic),
    ]

    contract = porter.validate_camera_preprocessing_contract(work)

    assert contract["head_image_hw"] == [4, 5]
    assert contract["rgb_dtype"] == "uint8"
    assert contract["depth_dtype"] == "uint16"
    assert contract["intrinsic"] == intrinsic.astype(np.float64).tolist()


def test_camera_contract_rejects_episode_calibration_drift(porter, tmp_path):
    first = _camera_take(tmp_path / "episode_0.hdf5")
    changed = np.eye(3, dtype=np.float32)
    changed[0, 0] = 2.0
    second = _camera_take(tmp_path / "episode_1.hdf5", changed)

    with pytest.raises(ValueError, match="differs"):
        porter.validate_camera_preprocessing_contract([first, second])


def test_structured_zarr_writer_preserves_axes_and_target_only_labels(
    porter, tmp_path, monkeypatch
):
    frames = 3
    args = argparse.Namespace(
        chunk_frames=2,
        num_points=4,
        object_nums=2,
        fps=10,
        pool_size=4,
        state_frame="world",
        crop_min=(-1.0, -1.0, -1.0),
        crop_max=(1.0, 1.0, 1.0),
        crop_frame="world",
        min_depth=0.05,
        max_depth=8.0,
    )

    def fake_episode_arrays(*_args, **_kwargs):
        return (
            np.zeros((frames, 4, 6), np.float32),
            np.zeros((frames, 32), np.float32),
            np.zeros((frames, 29), np.float32),
            5,
            1,
            4,
            None,
        )

    monkeypatch.setattr(porter, "episode_arrays", fake_episode_arrays)
    condition = np.arange(18, dtype=np.float32).reshape(3, 2, 3)
    sequence = np.array([0, 2, 1], np.int64)
    progress = np.array([0, 0, 1, 1, 2], np.int64)
    active = sequence[progress]
    key = ("raw", 7)
    zarr_path = tmp_path / "structured.zarr"

    porter.write_split_zarr(
        zarr_path,
        [{"source": "raw", "raw_index": 7, "path": tmp_path / "episode_7.hdf5", "trim": None}],
        None,
        args,
        np.random.default_rng(0),
        None,
        {key: (condition, sequence, progress, active)},
        None,
        None,
    )

    import zarr  # noqa: PLC0415

    root = zarr.open(str(zarr_path), mode="r")
    assert root["data/env_state"].shape == (frames, 3, 2, 3)
    assert root["data/task_sequence"].dtype == np.int64
    np.testing.assert_array_equal(root["data/task_sequence"][:], np.tile(sequence, (frames, 1)))
    np.testing.assert_array_equal(root["data/progress_index"][:, 0], progress[1:4])
    np.testing.assert_array_equal(root["data/active_task_id"][:, 0], active[1:4])


def test_joystick_zarr_writer_preserves_chassis_policy_action(
    porter, tmp_path, monkeypatch
):
    frames = 2
    args = argparse.Namespace(
        chunk_frames=2,
        num_points=4,
        object_nums=2,
        fps=10,
        pool_size=4,
        state_frame="base",
        crop_min=(-1.0, -1.0, -1.0),
        crop_max=(1.0, 1.0, 1.0),
        crop_frame="base",
        min_depth=0.05,
        max_depth=8.0,
        policy_action_schema=JOYSTICK_POLICY_ACTION_SCHEMA,
        action_axes=JOYSTICK_ACTION_AXES,
    )
    expected_actions = np.zeros((frames, len(JOYSTICK_ACTION_AXES)), np.float32)
    expected_actions[:, -3:] = [[0.1, -0.2, 0.3], [0.4, 0.0, -0.5]]

    def fake_episode_arrays(*_args, **kwargs):
        assert kwargs["policy_action_schema"] == JOYSTICK_POLICY_ACTION_SCHEMA
        return (
            np.zeros((frames, 4, 6), np.float32),
            np.zeros((frames, 32), np.float32),
            expected_actions,
            frames,
            0,
            frames,
            None,
        )

    class FakeArray:
        def __init__(self, *, shape=None, data=None, dtype=None, **_kwargs):
            self.attrs = {}
            self.values = (
                np.asarray(data)
                if data is not None
                else np.empty(shape, dtype=np.dtype(dtype))
            )

        def append(self, values):
            self.values = np.concatenate((self.values, np.asarray(values)), axis=0)

    class FakeGroup:
        def __init__(self):
            self.groups = {}
            self.arrays = {}

        def create_group(self, name):
            self.groups[name] = FakeGroup()
            return self.groups[name]

        def create_dataset(self, name, **kwargs):
            self.arrays[name] = FakeArray(**kwargs)
            return self.arrays[name]

    root = FakeGroup()

    monkeypatch.setattr(porter, "episode_arrays", fake_episode_arrays)
    monkeypatch.setattr(porter, "_zarr_writable_root", lambda _path: (root, 2))
    monkeypatch.setattr(porter, "_zarr_blosc", lambda: None)
    zarr_path = tmp_path / "joystick.zarr"
    porter.write_split_zarr(
        zarr_path,
        [{"source": "raw", "raw_index": 0,
          "path": tmp_path / "episode_0.hdf5", "trim": None}],
        None,
        args,
        np.random.default_rng(0),
        None,
        None,
        None,
        None,
    )

    stored = root.groups["data"].arrays["action"].values
    assert stored.shape == (frames, len(JOYSTICK_ACTION_AXES))
    np.testing.assert_array_equal(stored, expected_actions)
