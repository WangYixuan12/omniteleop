# ruff: noqa: SLF001

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from omniteleop.follower.live_object_condition import SceneRequirements
from omniteleop.wbc_policy_format import (
    JOYSTICK_ACTION_AXES,
    JOYSTICK_POLICY_ACTION_SCHEMA,
    STATE_AXES,
)


def _load_module():
    maniflow = Path("/home/yixuan/ManiFlow_Policy/ManiFlow")
    if str(maniflow) not in sys.path:
        sys.path.insert(0, str(maniflow))
    path = Path(__file__).resolve().parents[1] / "scripts" / "wbc_maniflow_rollout.py"
    name = "wbc_maniflow_rollout_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FakePolicy:
    def __init__(self, action_steps: int, action_dim: int = 29):
        self.action_steps = action_steps
        self.action_dim = action_dim
        self.seen = None

    def predict_action(self, obs):
        assert not torch.is_grad_enabled()
        self.seen = obs
        values = torch.arange(
            self.action_steps * self.action_dim, dtype=torch.float32
        ).reshape(1, self.action_steps, self.action_dim)
        return {"action": values}


@pytest.mark.parametrize("action_frame_mode", ["world", "mof"])
@pytest.mark.parametrize("action_dim", [29, 32])
@pytest.mark.parametrize(
    "position_mode,sampling,mask_channels,expected_extra",
    [
        ("none", "fps", False, set()),
        ("none", "mask_stratified", False, {"point_mask"}),
        ("none", "mask_stratified", True, {"point_mask"}),
        ("concat", "fps", False, {"env_state"}),
        ("grounding_tokens", "fps", False, {"env_state", "env_dino"}),
        (
            "grounding_tokens",
            "mask_stratified",
            True,
            {"env_state", "env_dino", "point_mask"},
        ),
    ],
)
def test_predict_chunk_supplies_only_checkpoint_required_raw_keys(
    position_mode, sampling, mask_channels, expected_extra, action_frame_mode,
    action_dim
) -> None:
    mod = _load_module()
    bundle = mod.ManiFlowBundle.__new__(mod.ManiFlowBundle)
    needs_state = position_mode != "none"
    needs_dino = position_mode == "grounding_tokens"
    needs_mask = sampling == "mask_stratified" or mask_channels
    bundle.scene_requirements = SceneRequirements(
        object_nums=2 if (needs_state or needs_dino or needs_mask) else 0,
        needs_env_state=needs_state,
        needs_env_dino=needs_dino,
        needs_point_mask=needs_mask,
    )
    bundle._torch = torch
    bundle.device = "cpu"
    bundle.n_obs_steps = 2
    bundle.n_action_steps = 3
    bundle.action_dim = action_dim
    bundle.policy = _FakePolicy(bundle.n_action_steps, action_dim)
    bundle._env_state_tensor = torch.arange(6, dtype=torch.float32)[None] if needs_state else None
    bundle._env_dino_tensor = torch.arange(8, dtype=torch.float32)[None] if needs_dino else None
    bundle._last_inference_trace = None
    bundle._post_capture_t0 = 0.0
    bundle._episode_min_free_bytes = None

    history = [
        mod.ManiFlowObservation(
            state=np.full(32, step, np.float32),
            point_cloud=np.full((16, 6), step, np.float32),
            point_mask=np.full(16, step + 1, np.int8) if needs_mask else None,
        )
        for step in range(2)
    ]
    result = bundle.predict_chunk(history)

    assert set(bundle.policy.seen) == {"point_cloud", "agent_pos"} | expected_extra
    assert bundle.policy.seen["point_cloud"].shape == (1, 2, 16, 6)
    assert bundle.policy.seen["agent_pos"].shape == (1, 2, 32)
    if needs_mask:
        assert bundle.policy.seen["point_mask"].shape == (1, 2, 16)
        assert bundle.policy.seen["point_mask"].dtype == torch.int8
    assert result.shape == (3, action_dim)
    np.testing.assert_array_equal(
        result,
        np.arange(3 * action_dim, dtype=np.float32).reshape(3, action_dim),
    )


def test_pointcloud_meta_requires_explicit_joystick_action_contract(tmp_path) -> None:
    mod = _load_module()
    bundle = mod.ManiFlowBundle.__new__(mod.ManiFlowBundle)
    bundle.policy_action_schema = JOYSTICK_POLICY_ACTION_SCHEMA
    bundle.num_points = 8
    bundle.conditioning_spec = SimpleNamespace(
        needs_point_mask=False,
        needs_env_dino=False,
    )
    meta = {
        "schema": "wbc_maniflow_pointcloud_v3",
        "state_frame": "base",
        "state_axes": list(STATE_AXES),
        "crop_min": [-1.0, -1.0, 0.0],
        "crop_max": [1.0, 1.0, 2.0],
        "crop_frame": "base",
        "min_depth_m": 0.1,
        "max_depth_m": 2.0,
        "num_points": 8,
        "point_frame": "world (engage-origin)",
        "point_channels": ["x", "y", "z", "r", "g", "b"],
        "pool_size": 16,
        "seed": 0,
        "rng_scheme": "numpy.default_rng(seed); one choice draw per frame",
        "head_image_hw": [4, 5],
        "intrinsic": [[100.0, 0.0, 2.0], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]],
        "rgb_dtype": "uint8",
        "depth_dtype": "uint16",
        "depth_scale_m": 0.001,
        "fps": 10,
        "policy_action_schema": JOYSTICK_POLICY_ACTION_SCHEMA,
        "action_axes": list(JOYSTICK_ACTION_AXES),
        "action_frame": "current_base",
        "chassis_action": {
            "axes": list(JOYSTICK_ACTION_AXES[-3:]),
            "frame": "current_base_body",
            "source_dataset": "action/chassis/intent_body",
            "temporal_hold": "zero_order_hold_between_record_ticks",
        },
        **mod.sampler_signature(),
    }
    path = tmp_path / "joystick.meta.json"
    path.write_text(json.dumps(meta))

    assert bundle._load_meta(path)["action_frame"] == "current_base"

    meta["action_frame"] = "world"
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="current_base"):
        bundle._load_meta(path)


class _TrackerOwner:
    def __init__(self):
        self.prepared = 0

    def prepare_many(self, rgb):
        self.prepared += 1
        return rgb


class _TrackerSession:
    def __init__(self, masks):
        self.masks = masks
        self.steps = 0

    def step_many(self, prepared):
        self.steps += 1
        assert len(prepared) == len(self.masks)
        return self.masks.copy()


def _bare_mask_bundle(mod):
    bundle = mod.ManiFlowBundle.__new__(mod.ManiFlowBundle)
    bundle.scene_requirements = SceneRequirements(object_nums=2, needs_point_mask=True)
    bundle.conditioning_spec = SimpleNamespace(object_nums=2)
    bundle.head_hw = (16, 16)
    bundle._condition = SimpleNamespace(world_frame_epoch=9, positions=np.zeros((2, 3)))
    masks = np.zeros((2, 16, 16), np.uint8)
    masks[:, :, :8] = 1
    masks[:, :, 8:] = 2
    bundle._tracker_owner = _TrackerOwner()
    bundle._tracker_session = _TrackerSession(masks)
    bundle._blowup_runs = np.zeros(2, np.int64)
    bundle._first_window_gate_pending = False
    bundle._last_tracker_timestamp_ns = 5
    bundle._observation_timestamp_floor_ns = 9
    bundle._last_inference_trace = None
    bundle._post_capture_t0 = 0.0
    bundle.intrinsic = np.array([[100, 0, 7.5], [0, 100, 7.5], [0, 0, 1]], np.float64)
    bundle.num_points = 8
    bundle.pool_size = 32
    bundle._rng = np.random.default_rng(7)
    bundle.crop_frame = "world"
    bundle.crop_min = (-10.0, -10.0, 0.0)
    bundle.crop_max = (10.0, 10.0, 2.0)
    bundle.min_depth = 0.1
    bundle.max_depth = 2.0
    return bundle


def _raw(mod, timestamp, epoch=9):
    return mod.ManiFlowRawFrame(
        rgb=np.zeros((16, 16, 3), np.uint8),
        depth_mm=np.full((16, 16), 1000, np.uint16),
        state=np.zeros(32, np.float32),
        head_timestamp_ns=timestamp,
        world_t_cam=np.eye(4, dtype=np.float32),
        world_frame_epoch=epoch,
    )


def test_observation_window_advances_tracker_once_after_all_frames_validate() -> None:
    mod = _load_module()
    bundle = _bare_mask_bundle(mod)
    observations = bundle.build_observations([_raw(mod, 10), _raw(mod, 20)])
    assert bundle._tracker_owner.prepared == 1
    assert bundle._tracker_session.steps == 1
    assert [obs.head_timestamp_ns for obs in observations] == [10, 20]
    assert all(obs.point_mask is not None for obs in observations)
    assert bundle._last_tracker_timestamp_ns == 20


def test_invalid_complete_window_does_not_advance_tracker_or_rng() -> None:
    mod = _load_module()
    bundle = _bare_mask_bundle(mod)
    before = bundle._rng.bit_generator.state
    with pytest.raises(RuntimeError, match="world-frame epoch"):
        bundle.build_observations([_raw(mod, 10), _raw(mod, 20, epoch=10)])
    assert bundle._tracker_owner.prepared == 0
    assert bundle._tracker_session.steps == 0
    assert bundle._rng.bit_generator.state == before


def test_window_older_than_validation_frame_does_not_advance_tracker() -> None:
    mod = _load_module()
    bundle = _bare_mask_bundle(mod)

    with pytest.raises(RuntimeError, match="strictly increasing"):
        bundle.build_observations([_raw(mod, 6), _raw(mod, 8)])

    assert bundle._tracker_owner.prepared == 0
    assert bundle._tracker_session.steps == 0


def test_live_condition_device_copy_failure_is_atomic() -> None:
    mod = _load_module()
    bundle = mod.ManiFlowBundle.__new__(mod.ManiFlowBundle)
    bundle.scene_requirements = SceneRequirements(
        object_nums=2, needs_env_state=True, needs_env_dino=True
    )
    bundle._check_live_object_condition = lambda condition: None
    bundle.device = "cuda"
    bundle._validation_report = {"head_timestamp_ns": 20}
    old_condition = object()
    old_state = object()
    old_dino = object()
    bundle._condition = old_condition
    bundle._env_state_tensor = old_state
    bundle._env_dino_tensor = old_dino
    bundle._last_tracker_timestamp_ns = 11
    bundle._observation_timestamp_floor_ns = 12

    class FakeTensor:
        def __init__(self, owner):
            self.owner = owner

        def to(self, device):
            self.owner.copies += 1
            if self.owner.copies == 2:
                raise RuntimeError("synthetic DINO copy failure")
            return self

    class FakeTorch:
        copies = 0

        def from_numpy(self, value):
            return FakeTensor(self)

    bundle._torch = FakeTorch()
    condition = SimpleNamespace(
        env_state=np.zeros(6, np.float32),
        env_dino=np.zeros(8, np.float32),
        camera_timestamp_ns=15,
    )

    with pytest.raises(RuntimeError, match="synthetic DINO copy failure"):
        bundle.install_live_object_condition(condition)

    assert bundle._condition is old_condition
    assert bundle._env_state_tensor is old_state
    assert bundle._env_dino_tensor is old_dino
    assert bundle._last_tracker_timestamp_ns == 11
    assert bundle._observation_timestamp_floor_ns == 12
