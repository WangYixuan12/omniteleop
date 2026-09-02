# ruff: noqa: SLF001

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from omniteleop.follower.live_object_condition import (
    LiveObjectArtifacts,
    SceneRequirements,
)


def _load_rollout():
    path = Path(__file__).resolve().parents[1] / "scripts" / "wbc_policy_rollout.py"
    name = "wbc_live_object_bootstrap_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _artifacts(rgb: np.ndarray, timestamp_ns: int, source) -> LiveObjectArtifacts:
    seed = np.array([[1, 1, 0], [0, 2, 2]], dtype=np.uint8)
    return LiveObjectArtifacts(
        positions_size=np.array([[1, 0, 0], [2, 0, 0]], np.float32),
        valid_size=np.ones(2, np.float32),
        scene_diff_obj_ids_size=np.array([7, 3], np.int64),
        seed_size_slots=seed,
        dino_region_feats_size=np.stack((np.full(4, 1), np.full(4, 2))).astype(np.float32),
        bootstrap_rgb=rgb,
        bootstrap_depth_mm=np.full(rgb.shape[:2], 1000, np.uint16),
        intrinsic=np.eye(3, dtype=np.float32),
        world_t_cam=np.eye(4, dtype=np.float32),
        camera_timestamp_ns=timestamp_ns,
        world_frame_epoch=4,
        source_artifact=source,
        provenance_hashes=("a" * 64, "b" * 64, "c" * 64),
        dino_model="model",
        dino_source="source",
        dino_input_scale="[0,1]",
    )


def _args(tmp_path):
    return SimpleNamespace(save_dir=str(tmp_path), prompt_before_capture=False)


def test_live_bootstrap_publishes_only_after_atomic_arrangement_and_validation(
    tmp_path, monkeypatch
) -> None:
    mod = _load_rollout()
    frames = iter(((11, 100), (22, 200)))
    def grab_head():
        value, timestamp = next(frames)
        return (
            np.full((2, 3, 3), value, np.uint8),
            np.full((2, 3), 1000, np.uint16),
            timestamp,
        )

    driver = SimpleNamespace(
        _episode=SimpleNamespace(episode_id=5),
        _world_frame_epoch=4,
        _grab_head_images=grab_head,
    )
    captured: dict = {}

    def write_capture(path, rgb, *_args, **_kwargs):
        captured["rgb"] = rgb.copy()
        captured["timestamp"] = 100
        path.write_bytes(b"capture")

    def invoke(_args, _live_hdf5, out_dir, requirements):
        assert requirements.needs_env_dino
        (out_dir / "episode_0.npz").write_bytes(b"positions")
        (out_dir / "bootstrap_size_slots.npz").write_bytes(b"bootstrap")
        (out_dir / "deploy_slot_overlay.png").write_bytes(b"overlay")

    def load(bootstrap_path, _capture_path):
        return _artifacts(captured["rgb"], captured["timestamp"], bootstrap_path)

    class Bundle:
        scene_requirements = SceneRequirements(
            object_nums=2, needs_env_state=True, needs_env_dino=True,
        )
        intrinsic = np.eye(3, dtype=np.float32)

        def validate_live_object_condition(self, condition, *, capture_validation_frame,
                                           stage_dir):
            assert not (tmp_path / "scene_diff" / "episode_5").exists()
            fresh = capture_validation_frame()
            assert fresh.head_timestamp_ns == 200
            assert fresh.world_frame_epoch == condition.world_frame_epoch
            (stage_dir / "validation_report.json").write_text("{}")

        def install_live_object_condition(self, condition):
            self.condition = condition

    bundle = Bundle()
    monkeypatch.setattr(mod, "_write_live_capture_hdf5", write_capture)
    monkeypatch.setattr(mod, "_invoke_live_scenediff", invoke)
    monkeypatch.setattr(mod, "load_live_object_artifacts", load)
    monkeypatch.setattr(mod, "_prompt_for_task_order", lambda *_args: [1, 0])
    monkeypatch.setattr(
        mod, "_live_world_T_zed", lambda *_args: np.eye(4, dtype=np.float32)
    )

    condition = mod._prepare_live_object_condition(
        _args(tmp_path), driver, object(), bundle
    )
    final = tmp_path / "scene_diff" / "episode_5"
    assert final.is_dir()
    assert not list(final.parent.glob(".episode_5.building-*"))
    np.testing.assert_array_equal(condition.order, [1, 0])
    np.testing.assert_array_equal(condition.positions[:, 0], [2, 1])
    np.testing.assert_array_equal(
        condition.seed_task_slots, np.array([[2, 2, 0], [0, 1, 1]], np.uint8)
    )
    assert condition.source_artifact == final / "bootstrap_size_slots.npz"
    assert bundle.condition is condition


def test_existing_live_bootstrap_aborts_before_capture(tmp_path) -> None:
    mod = _load_rollout()
    final = tmp_path / "scene_diff" / "episode_2"
    final.mkdir(parents=True)
    driver = SimpleNamespace(
        _episode=SimpleNamespace(episode_id=2),
        _grab_head_images=lambda: pytest.fail("capture must not run"),
    )
    bundle = SimpleNamespace(
        scene_requirements=SceneRequirements(object_nums=2, needs_env_state=True)
    )
    with pytest.raises(FileExistsError, match="delete it manually"):
        mod._prepare_live_object_condition(_args(tmp_path), driver, object(), bundle)


def test_failed_live_bootstrap_leaves_no_final_directory(tmp_path, monkeypatch) -> None:
    mod = _load_rollout()
    driver = SimpleNamespace(
        _episode=SimpleNamespace(episode_id=3),
        _world_frame_epoch=1,
        _grab_head_images=lambda: (
            np.zeros((2, 3, 3), np.uint8),
            np.ones((2, 3), np.uint16),
            10,
        ),
    )
    bundle = SimpleNamespace(
        scene_requirements=SceneRequirements(object_nums=2, needs_env_state=True),
        intrinsic=np.eye(3, dtype=np.float32),
    )
    monkeypatch.setattr(
        mod, "_live_world_T_zed", lambda *_args: np.eye(4, dtype=np.float32)
    )
    monkeypatch.setattr(
        mod,
        "_write_live_capture_hdf5",
        lambda path, *_args, **_kwargs: path.write_bytes(b"capture"),
    )
    monkeypatch.setattr(
        mod, "_invoke_live_scenediff", lambda *_args: (_ for _ in ()).throw(
            RuntimeError("detection failed")
        )
    )

    with pytest.raises(RuntimeError, match="detection failed"):
        mod._prepare_live_object_condition(_args(tmp_path), driver, object(), bundle)
    parent = tmp_path / "scene_diff"
    assert not (parent / "episode_3").exists()
    assert not list(parent.glob(".episode_3.building-*"))


def test_live_bootstrap_rejects_missing_slot_overlay(tmp_path, monkeypatch) -> None:
    mod = _load_rollout()
    driver = SimpleNamespace(
        _episode=SimpleNamespace(episode_id=4),
        _world_frame_epoch=1,
        _grab_head_images=lambda: (
            np.zeros((2, 3, 3), np.uint8),
            np.ones((2, 3), np.uint16),
            10,
        ),
    )
    bundle = SimpleNamespace(
        scene_requirements=SceneRequirements(object_nums=2, needs_env_state=True),
        intrinsic=np.eye(3, dtype=np.float32),
    )
    monkeypatch.setattr(
        mod, "_live_world_T_zed", lambda *_args: np.eye(4, dtype=np.float32)
    )
    monkeypatch.setattr(
        mod,
        "_write_live_capture_hdf5",
        lambda path, *_args, **_kwargs: path.write_bytes(b"capture"),
    )

    def invoke(_args, _live_hdf5, out_dir, _requirements):
        (out_dir / "episode_0.npz").write_bytes(b"positions")
        (out_dir / "bootstrap_size_slots.npz").write_bytes(b"bootstrap")

    monkeypatch.setattr(mod, "_invoke_live_scenediff", invoke)
    monkeypatch.setattr(
        mod,
        "load_live_object_artifacts",
        lambda *_args: pytest.fail("artifact loading must wait for the overlay gate"),
    )

    with pytest.raises(FileNotFoundError, match="size-slot overlay"):
        mod._prepare_live_object_condition(_args(tmp_path), driver, object(), bundle)
    parent = tmp_path / "scene_diff"
    assert not (parent / "episode_4").exists()
    assert not list(parent.glob(".episode_4.building-*"))
