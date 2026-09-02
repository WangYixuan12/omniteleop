from dataclasses import replace

import numpy as np

from omniteleop.follower.live_object_condition import LiveObjectArtifacts
from omniteleop.follower.scenediff_live import (
    load_live_object_artifacts,
    write_live_capture_hdf5,
)


def test_nonidentity_order_rearranges_positions_dino_ids_and_seed_together(tmp_path):
    artifacts = LiveObjectArtifacts(
        positions_size=np.array([[1, 2, 3], [4, 5, 6]], np.float32),
        valid_size=np.ones(2, np.float32),
        scene_diff_obj_ids_size=np.array([11, 23], np.int64),
        seed_size_slots=np.array([[1, 0], [0, 2]], np.uint8),
        dino_region_feats_size=np.array([[10, 11], [20, 21]], np.float32),
        bootstrap_rgb=np.zeros((2, 2, 3), np.uint8),
        bootstrap_depth_mm=np.ones((2, 2), np.uint16),
        intrinsic=np.eye(3, dtype=np.float32),
        world_t_cam=np.eye(4, dtype=np.float32),
        camera_timestamp_ns=123,
        world_frame_epoch=7,
        source_artifact=tmp_path / "bootstrap.npz",
        provenance_hashes=("a" * 64, "b" * 64, "c" * 64),
        dino_model="model",
        dino_source="source",
        dino_input_scale="[0,1]",
    )

    condition = artifacts.arrange([1, 0])

    np.testing.assert_array_equal(condition.positions, [[4, 5, 6], [1, 2, 3]])
    np.testing.assert_array_equal(condition.dino_region_feats, [[20, 21], [10, 11]])
    np.testing.assert_array_equal(condition.scene_diff_obj_ids, [23, 11])
    np.testing.assert_array_equal(condition.seed_task_slots, [[2, 0], [0, 1]])
    assert not condition.positions.flags.writeable
    assert not condition.seed_task_slots.flags.writeable


def test_condition_rejects_invalid_order_or_object_ids(tmp_path):
    artifacts = LiveObjectArtifacts(
        positions_size=np.zeros((2, 3), np.float32),
        valid_size=np.ones(2, np.float32),
        scene_diff_obj_ids_size=np.array([11, 23], np.int64),
        seed_size_slots=np.array([[1, 2]], np.uint8),
        dino_region_feats_size=None,
        bootstrap_rgb=np.zeros((1, 2, 3), np.uint8),
        bootstrap_depth_mm=np.ones((1, 2), np.uint16),
        intrinsic=np.eye(3, dtype=np.float32),
        world_t_cam=np.eye(4, dtype=np.float32),
        camera_timestamp_ns=1,
        world_frame_epoch=0,
        source_artifact=tmp_path / "bootstrap.npz",
    )
    condition = artifacts.arrange([0, 1])

    with np.testing.assert_raises_regex(ValueError, "permutation"):
        replace(condition, order=np.array([0, 0], np.int64))
    with np.testing.assert_raises_regex(ValueError, "unique and non-negative"):
        replace(condition, scene_diff_obj_ids=np.array([11, 11], np.int64))


def test_artifacts_own_readonly_copies_without_freezing_caller_arrays(tmp_path):
    positions = np.zeros((2, 3), np.float32)
    rgb = np.zeros((2, 2, 3), np.uint8)
    artifacts = LiveObjectArtifacts(
        positions_size=positions,
        valid_size=np.ones(2, np.float32),
        scene_diff_obj_ids_size=np.array([1, 2], np.int64),
        seed_size_slots=np.array([[1, 2]], np.uint8),
        dino_region_feats_size=None,
        bootstrap_rgb=rgb,
        bootstrap_depth_mm=np.ones((2, 2), np.uint16),
        intrinsic=np.eye(3, dtype=np.float32),
        world_t_cam=np.eye(4, dtype=np.float32),
        camera_timestamp_ns=1,
        world_frame_epoch=0,
        source_artifact=tmp_path / "bootstrap.npz",
    )

    assert positions.flags.writeable
    assert rgb.flags.writeable
    assert not np.shares_memory(artifacts.positions_size, positions)
    assert not np.shares_memory(artifacts.bootstrap_rgb, rgb)


def test_bootstrap_loader_joins_native_capture_and_provenance(tmp_path):
    bootstrap = tmp_path / "bootstrap.npz"
    np.savez_compressed(
        bootstrap,
        schema="scenediff_live_object_bootstrap_v1",
        positions=np.zeros((2, 3), np.float32),
        valid=np.ones(2, np.float32),
        scene_diff_obj_ids=np.array([5, 7], np.int64),
        seed_size_slots=np.array([[1, 2]], np.uint8),
        frame="world",
        positions_sha256="a" * 64,
        object_masks_sha256="b" * 64,
        rgb_scenediff_sha256="c" * 64,
    )
    capture = tmp_path / "capture.hdf5"
    write_live_capture_hdf5(
        capture,
        np.zeros((2, 3, 3), np.uint8),
        np.ones((2, 3), np.uint16),
        np.eye(4, dtype=np.float32),
        np.eye(3, dtype=np.float32),
        camera_timestamp_ns=99,
        world_frame_epoch=4,
    )

    artifacts = load_live_object_artifacts(bootstrap, capture)

    assert artifacts.camera_timestamp_ns == 99
    assert artifacts.world_frame_epoch == 4
    assert artifacts.bootstrap_rgb.shape == (2, 3, 3)
    assert artifacts.provenance_hashes == ("a" * 64, "b" * 64, "c" * 64)


def test_bootstrap_loader_rejects_invalid_dino_valid(tmp_path):
    bootstrap = tmp_path / "bootstrap.npz"
    np.savez_compressed(
        bootstrap,
        schema="scenediff_live_object_bootstrap_v1",
        positions=np.zeros((2, 3), np.float32),
        valid=np.ones(2, np.float32),
        scene_diff_obj_ids=np.array([5, 7], np.int64),
        seed_size_slots=np.array([[1, 2]], np.uint8),
        frame="world",
        positions_sha256="a" * 64,
        object_masks_sha256="b" * 64,
        rgb_scenediff_sha256="c" * 64,
        dino_region_feats=np.ones((2, 4), np.float32),
        dino_valid=np.array([1, 0], np.float32),
        dino_model="model",
        dino_source="source",
        dino_input_scale="[0,1]",
    )
    capture = tmp_path / "capture.hdf5"
    write_live_capture_hdf5(
        capture,
        np.zeros((2, 3, 3), np.uint8),
        np.ones((2, 3), np.uint16),
        np.eye(4, dtype=np.float32),
        np.eye(3, dtype=np.float32),
        camera_timestamp_ns=99,
        world_frame_epoch=4,
    )

    with np.testing.assert_raises_regex(ValueError, "dino_valid"):
        load_live_object_artifacts(bootstrap, capture)
