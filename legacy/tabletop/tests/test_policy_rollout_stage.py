from __future__ import annotations

# ruff: noqa: SLF001
from types import SimpleNamespace

import numpy as np
import torch
from lerobot.utils.constants import OBS_ENV_STATE, OBS_POS_CONDITION_MASK

from omniteleop.follower.policy_rollout import PolicyRolloutController


def _mask(stage: int, classes: int = 2) -> torch.Tensor:
    out = np.zeros((classes,), dtype=np.float32)
    out[stage] = 1.0
    return torch.from_numpy(out)


class _FakeACTModel:
    def __call__(self, batch: dict[str, torch.Tensor]):
        del batch
        # Offset 0 predicts object/stage 1 for the current observation timestamp.
        stage_logits = torch.tensor([[[0.0, 10.0], [10.0, 0.0]]], dtype=torch.float32)
        actions = torch.zeros((1, 2, 1), dtype=torch.float32)
        return actions, (None, None), stage_logits


class _FakeACTPolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            type="act",
            stage_prediction_enabled=True,
            stage_prediction_num_classes=2,
            n_action_steps=2,
            image_features=[],
        )
        self.model = _FakeACTModel()
        self._action_queue: list[torch.Tensor] = []
        self.action_stage_ids: list[int] = []

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = torch.as_tensor(batch[OBS_POS_CONDITION_MASK])
        self.action_stage_ids.append(int(mask.argmax(dim=-1).reshape(-1)[0].item()))
        return torch.tensor([[0.0]], dtype=torch.float32)


def test_replan_action_uses_stage_predicted_from_current_observation() -> None:
    ctrl = PolicyRolloutController.__new__(PolicyRolloutController)
    ctrl._policy = _FakeACTPolicy()
    ctrl._env_state_enabled = True
    ctrl._stage_num_classes = 2
    ctrl._current_stage = 0
    ctrl._env_state_vec = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        dtype=np.float32,
    )
    ctrl._chunk_anchor_active = False
    ctrl._relative_step = None
    ctrl._chunk_ref_state = None

    def build_observation() -> dict[str, torch.Tensor]:
        return {
            "observation.state": torch.zeros((1,), dtype=torch.float32),
            OBS_ENV_STATE: torch.zeros((12,), dtype=torch.float32),
            OBS_POS_CONDITION_MASK: _mask(ctrl._current_stage),
        }

    def preprocess(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return dict(sample)

    def postprocess(action: torch.Tensor) -> torch.Tensor:
        return action

    ctrl._build_observation = build_observation
    ctrl._pre = preprocess
    ctrl._post = postprocess

    _, info = ctrl._policy_step()

    assert info["replan"] is True
    assert ctrl._policy.action_stage_ids == [1]
    assert ctrl._current_stage == 1
    assert info["position_condition"]["stage"] == 1
    np.testing.assert_array_equal(
        info["position_condition"]["mask"], np.array([0.0, 1.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        info["position_condition"]["condition_6d"],
        np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float32),
    )


class _FakeRecorder:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    def record(self, frame: dict) -> None:
        self.frames.append(frame)


def test_record_frame_saves_position_condition_when_present() -> None:
    ctrl = PolicyRolloutController.__new__(PolicyRolloutController)
    ctrl.variant = SimpleNamespace(action_is_eef=True, arm_action_block=10)
    ctrl.gripper_threshold = 0.5
    ctrl._use_wrist = False
    ctrl._recorder = _FakeRecorder()
    ctrl._last_obs_head_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    ctrl._last_obs_depth_u16 = np.zeros((2, 2), dtype=np.uint16)
    ctrl._last_obs_qpos_left = np.zeros((7,), dtype=np.float32)
    ctrl._last_obs_qpos_right = np.ones((7,), dtype=np.float32)
    ctrl._last_obs_eef9_left = np.arange(9, dtype=np.float32)
    ctrl._last_obs_eef9_right = np.arange(9, dtype=np.float32) + 10
    ctrl._last_obs_gripper_left = np.float32(0.25)
    ctrl._last_obs_gripper_right = np.float32(0.75)
    ctrl._head_camera_intrinsic = np.eye(3, dtype=np.float32)
    ctrl._head_camera_extrinsic = np.eye(4, dtype=np.float32) * 2.0

    step_info = {
        "t_obs_ns_wallclock": 1,
        "begin_build_obs_ns": 2,
        "begin_inference_ns": 3,
        "finish_inference_ns": 4,
        "replan": True,
        "position_condition": {
            "stage": 1,
            "mask": np.array([0.0, 1.0], dtype=np.float32),
            "condition_6d": np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float32),
        },
    }

    ctrl._record_frame(
        action=np.zeros((20,), dtype=np.float32),
        targets={
            "left": np.zeros((7,), dtype=np.float32),
            "right": np.zeros((7,), dtype=np.float32),
        },
        cand_left_grip=0.0,
        cand_right_grip=1.0,
        step_info=step_info,
        publish_command_ns=5,
        chunk_idx=0,
    )

    position_condition = ctrl._recorder.frames[0]["obs"]["position_condition"]
    assert position_condition["stage"] == np.int32(1)
    np.testing.assert_array_equal(
        position_condition["mask"], step_info["position_condition"]["mask"]
    )
    np.testing.assert_array_equal(
        position_condition["condition_6d"], step_info["position_condition"]["condition_6d"]
    )
    images = ctrl._recorder.frames[0]["obs"]["images"]
    np.testing.assert_array_equal(images["intrinsic"], ctrl._head_camera_intrinsic)
    np.testing.assert_array_equal(images["extrinsic"], ctrl._head_camera_extrinsic)


def test_in_chunk_record_keeps_replan_position_condition() -> None:
    ctrl = PolicyRolloutController.__new__(PolicyRolloutController)
    ctrl._policy = _FakeACTPolicy()
    ctrl._policy._action_queue = [torch.zeros((1,), dtype=torch.float32)]
    ctrl._env_state_enabled = True
    ctrl._stage_num_classes = 2
    ctrl._current_stage = 1
    ctrl._env_state_vec = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        dtype=np.float32,
    )
    ctrl._active_position_condition = {
        "stage": np.int32(0),
        "mask": np.array([1.0, 0.0], dtype=np.float32),
        "condition_6d": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
    }
    ctrl._chunk_anchor_active = False
    ctrl._relative_step = None
    ctrl._chunk_ref_state = None

    def build_observation() -> dict[str, torch.Tensor]:
        return {
            "observation.state": torch.zeros((1,), dtype=torch.float32),
            OBS_ENV_STATE: torch.zeros((12,), dtype=torch.float32),
            OBS_POS_CONDITION_MASK: _mask(ctrl._current_stage),
        }

    def preprocess(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return dict(sample)

    def postprocess(action: torch.Tensor) -> torch.Tensor:
        return action

    ctrl._build_observation = build_observation
    ctrl._pre = preprocess
    ctrl._post = postprocess

    _, info = ctrl._policy_step()

    assert info["replan"] is False
    assert info["position_condition"]["stage"] == 0
    np.testing.assert_array_equal(
        info["position_condition"]["condition_6d"],
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
    )
