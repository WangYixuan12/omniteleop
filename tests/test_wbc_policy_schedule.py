"""Unit tests for wbc_policy_rollout's scheduled-action buffer + sampler.

Covers the rby1-wbc port (plan.md Task 2): stale-frame dropping, chunk
overwrite/trim semantics, previous/future bracketing with lerp/slerp,
last-executed anchoring, and the terminal-knot-once-then-None exhaustion
semantics that keeps the stale-source watchdog armed.

Buffer tests are numpy-only (dexmate env, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1);
the predict_chunk/select_action tests at the bottom need lerobot + torch and
skip elsewhere (run them via the dexmate_lerobot env).
"""

from __future__ import annotations

import importlib.util
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_rollout():
    path = Path(__file__).resolve().parents[1] / "scripts" / "wbc_policy_rollout.py"
    name = "wbc_policy_rollout_schedule_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module  # dataclass string-annotation resolution needs this
    spec.loader.exec_module(module)
    return module


def _pose(x: float, yaw: float = 0.0) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4)
    pose[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    pose[0, 3] = x
    return pose


def _act(mod, t: float, x: float, yaw: float = 0.0, grip: float = 0.0):
    return mod.ScheduledPolicyAction(
        timestamp=float(t), left=_pose(x, yaw), right=_pose(x), head=_pose(x),
        grip_left=np.float32(grip), grip_right=np.float32(grip),
    )


def _yaw_of(pose: np.ndarray) -> float:
    return float(np.arctan2(pose[1, 0], pose[0, 0]))


# ── drop_stale_actions ──────────────────────────────────────────────────────
def test_drop_stale_actions_drops_leading_frames_only():
    mod = _load_rollout()
    actions = [_act(mod, 0.1 * k, x=float(k)) for k in range(1, 5)]
    kept = mod.drop_stale_actions(actions, cutoff=0.25)
    assert [a.timestamp for a in kept] == pytest.approx([0.3, 0.4])
    # boundary is strict: a frame AT the cutoff is stale (cannot execute "now")
    assert [a.timestamp for a in
            mod.drop_stale_actions(actions, cutoff=actions[2].timestamp)] == (
        pytest.approx([0.4])
    )
    assert mod.drop_stale_actions(actions, cutoff=1.0) == []
    assert mod.drop_stale_actions(actions, cutoff=0.0) == actions
    with pytest.raises(ValueError, match="finite"):
        mod.drop_stale_actions(actions, cutoff=float("nan"))


# ── ActionScheduleBuffer.queue ──────────────────────────────────────────────
def test_queue_overwrites_overlapping_future_and_trims_past():
    mod = _load_rollout()
    buf = mod.ActionScheduleBuffer()
    buf.queue([_act(mod, 0.1 * k, x=0.1 * k) for k in range(1, 9)], now=0.0)
    assert len(buf) == 8
    # Newest plan wins for t >= 0.35; entries already past (t < now=0.25) trim.
    buf.queue([_act(mod, 0.35, x=10.35), _act(mod, 0.45, x=10.45)], now=0.25)
    assert len(buf) == 3  # kept 0.3 (past entries 0.1, 0.2 trimmed) + 2 new
    # The 0.3 knot survived and the new plan starts at 0.35: sampling at 0.30
    # brackets [0.3 old, 0.35 new].
    out = buf.sample(0.30)
    assert out is not None
    assert out.left[0, 3] == pytest.approx(0.3)


def test_queue_validates_input():
    mod = _load_rollout()
    buf = mod.ActionScheduleBuffer()
    with pytest.raises(ValueError, match="at least one"):
        buf.queue([], now=0.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        buf.queue([_act(mod, 0.2, x=0.0), _act(mod, 0.1, x=0.0)], now=0.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        buf.queue([_act(mod, float("inf"), x=0.0)], now=0.0)


# ── ActionScheduleBuffer.sample ─────────────────────────────────────────────
def test_sample_empty_buffer_returns_none():
    mod = _load_rollout()
    assert mod.ActionScheduleBuffer().sample(1.0) is None


def test_sample_before_first_knot_returns_it_verbatim():
    mod = _load_rollout()
    buf = mod.ActionScheduleBuffer()
    buf.queue([_act(mod, 1.0, x=0.5, grip=1.0)], now=0.0)
    out = buf.sample(0.4)
    assert out is not None
    assert out.left[0, 3] == pytest.approx(0.5)
    assert float(out.grip_left) == pytest.approx(1.0)
    assert out.timestamp == pytest.approx(0.4)  # executed-at time, not the knot


def test_sample_interpolates_poses_but_holds_grippers_until_next_knot():
    mod = _load_rollout()
    buf = mod.ActionScheduleBuffer()
    buf.queue([
        _act(mod, 1.0, x=0.0, yaw=0.0, grip=0.0),
        _act(mod, 1.1, x=0.1, yaw=np.pi / 2, grip=1.0),
    ], now=0.0)
    between = buf.sample(1.05)
    assert between is not None
    assert between.left[0, 3] == pytest.approx(0.05)          # lerped position
    assert _yaw_of(between.left) == pytest.approx(np.pi / 4)  # slerped 90 deg -> 45
    assert float(between.grip_left) == 0.0                    # categorical hold
    assert float(between.grip_right) == 0.0
    # right/head ride the same knots (x only here)
    assert between.right[0, 3] == pytest.approx(0.05)
    assert between.head[0, 3] == pytest.approx(0.05)

    at_knot = buf.sample(1.1)
    assert at_knot is not None
    assert float(at_knot.grip_left) == 1.0
    assert float(at_knot.grip_right) == 1.0


def test_sample_prefers_last_executed_anchor_over_passed_knot():
    mod = _load_rollout()
    buf = mod.ActionScheduleBuffer()
    buf.queue([
        _act(mod, 1.0, x=0.0),
        _act(mod, 1.1, x=1.0),
        _act(mod, 1.3, x=2.0),
    ], now=0.0)
    first = buf.sample(1.05)  # midway 1.0 -> 1.1: x=0.5, remembered as last-executed
    assert first is not None and first.left[0, 3] == pytest.approx(0.5)
    # q=1.15 passes the 1.1 knot (x=1.0), but the anchor is the value ACTUALLY
    # sent at 1.05 (rby1 continuity): alpha=(1.15-1.05)/(1.3-1.05)=0.4 ->
    # x = 0.5 + 0.4*(2.0-0.5) = 1.1 (vs 1.25 if the passed knot anchored).
    out = buf.sample(1.15)
    assert out is not None
    assert out.left[0, 3] == pytest.approx(1.1)


def test_sample_interpolates_from_last_executed_to_next_chunk():
    mod = _load_rollout()
    buf = mod.ActionScheduleBuffer()
    buf.queue([_act(mod, 1.0, x=0.0)], now=0.0)
    consumed = buf.sample(1.0)  # consumes the only knot (terminal, verbatim)
    assert consumed is not None and consumed.left[0, 3] == pytest.approx(0.0)
    buf.queue([_act(mod, 1.2, x=0.2, grip=1.0)], now=1.05)
    # Interpolate FROM the last-executed action TO the newly queued one.
    out = buf.sample(1.1)
    assert out is not None
    assert out.left[0, 3] == pytest.approx(0.1)
    assert float(out.grip_left) == 0.0
    at_knot = buf.sample(1.2)
    assert at_knot is not None
    assert float(at_knot.grip_left) == 1.0


def test_sample_exhaustion_returns_terminal_once_then_aborts():
    mod = _load_rollout()
    buf = mod.ActionScheduleBuffer()
    buf.queue([_act(mod, 1.0, x=0.7)], now=0.0)
    out = buf.sample(1.05)  # trajectory tail crossed: terminal knot, exactly once
    assert out is not None and out.left[0, 3] == pytest.approx(0.7)
    with pytest.raises(RuntimeError, match="underflow"):
        buf.sample(1.15)
    with pytest.raises(RuntimeError, match="late chunks"):
        buf.queue([_act(mod, 1.3, x=0.8)], now=1.2)


# ── predict_chunk / select_action shim (needs lerobot + torch) ──────────────
class _FakeDiffusionPolicy:
    """Minimal modeling_diffusion contract: obs queues + predict_action_chunk.

    The chunk is a deterministic function of BOTH stacked history steps
    (``s[0] + 2*s[-1]`` over the 29 action dims, + the frame index), so the
    tests below prove predict_chunk feeds the queues with the right content,
    padding, and order.
    """

    def __init__(self, torch, n_obs: int = 2, n_action: int = 3) -> None:
        self._torch = torch
        self.config = SimpleNamespace(
            image_features=["observation.images.head_rgb"],
            n_obs_steps=n_obs, n_action_steps=n_action,
        )
        self._n_obs = n_obs
        self._n_action = n_action
        self.reset()

    def reset(self) -> None:
        self._queues = {
            "observation.state": deque(maxlen=self._n_obs),
            "observation.images": deque(maxlen=self._n_obs),
            "action": deque(maxlen=self._n_action),
        }

    def predict_action_chunk(self, batch):
        states = self._torch.stack(list(self._queues["observation.state"]), dim=1)
        assert states.shape[1] == self._n_obs
        base = states[:, 0, :29] + 2.0 * states[:, -1, :29]
        return self._torch.stack(
            [base + float(k) for k in range(self._n_action)], dim=1
        )


def _fake_bundle(mod, torch):
    # White-box fixture: poking bundle privates is the point (hence the noqas).
    bundle = mod._PolicyBundle.__new__(mod._PolicyBundle)  # noqa: SLF001
    bundle._torch = torch  # noqa: SLF001
    bundle.use_head = True
    bundle.wrist_arms = ()
    bundle.image_hw = (4, 6)
    bundle.n_obs_steps = 2
    bundle.n_action_steps = 3
    bundle._uses_obs_queues = True  # noqa: SLF001
    bundle.state_frame = "base"
    # Non-conditioned bundle: _sample_dict skips the env-state branch.
    bundle.use_env_state = False
    bundle.env_state_dim = 0
    bundle._env_state = None  # noqa: SLF001
    bundle.policy = _FakeDiffusionPolicy(torch)
    bundle.pre = lambda sample: {k: v.unsqueeze(0).float() for k, v in sample.items()}
    bundle.post = lambda chunk: chunk
    bundle._obs_history = deque(maxlen=2)  # noqa: SLF001
    bundle._chunk_tail = deque()  # noqa: SLF001
    return bundle


def _state(v: float) -> np.ndarray:
    return np.full(32, v, dtype=np.float32)


_IMG = np.zeros((4, 6, 3), dtype=np.uint8)


def test_predict_chunk_feeds_history_and_pads_like_populate_queues():
    torch = pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    mod = _load_rollout()
    bundle = _fake_bundle(mod, torch)
    # Full history [A=1, B=2]: base = 1 + 2*2 = 5, rows 5, 6, 7.
    chunk = bundle.predict_chunk(
        [mod.PolicyObservation(_state(1.0), _IMG, None),
         mod.PolicyObservation(_state(2.0), _IMG, None)]
    )
    assert chunk.shape == (3, 29)
    np.testing.assert_allclose(chunk[:, 0], [5.0, 6.0, 7.0], atol=1e-6)
    # Short history [C=3] left-pads by repeating: base = 3 + 2*3 = 9.
    bundle.policy.reset()
    chunk = bundle.predict_chunk([mod.PolicyObservation(_state(3.0), _IMG, None)])
    np.testing.assert_allclose(chunk[:, 0], [9.0, 10.0, 11.0], atol=1e-6)
    with pytest.raises(ValueError, match="obs_history"):
        bundle.predict_chunk([])


def test_select_action_shim_matches_chunked_replan_cadence():
    torch = pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    mod = _load_rollout()
    bundle = _fake_bundle(mod, torch)
    # Replan on call 1 with padded history [A, A]: base = 3*A = 30.
    a1 = bundle.select_action(_state(10.0), _IMG, None)
    a2 = bundle.select_action(_state(20.0), _IMG, None)  # in-chunk pop
    a3 = bundle.select_action(_state(30.0), _IMG, None)  # in-chunk pop
    # Replan on call 4 with rolling history [C=30, D=40]: base = 30 + 80 = 110.
    a4 = bundle.select_action(_state(40.0), _IMG, None)
    assert a1[0] == pytest.approx(30.0)
    assert a2[0] == pytest.approx(31.0)
    assert a3[0] == pytest.approx(32.0)
    assert a4[0] == pytest.approx(110.0)


def test_sample_dict_injects_env_state_when_conditioned():
    """_sample_dict rides env-state only from the atomically installed condition."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    mod = _load_rollout()
    bundle = _fake_bundle(mod, torch)
    bundle.use_env_state = True
    bundle.env_state_dim = 6
    bundle.scene_requirements = mod.SceneRequirements(
        object_nums=2, needs_env_state=True
    )
    obs = mod.PolicyObservation(_state(1.0), _IMG, None)
    # Conditioned but not set yet -> must raise (never silently drop the env-state key).
    with pytest.raises(RuntimeError, match="live condition was not installed"):
        bundle._sample_dict(obs)  # noqa: SLF001
    # Wrong length rejected.
    with pytest.raises(ValueError, match=r"\(6,\)"):
        bundle.install_live_object_condition(SimpleNamespace(
            positions=np.zeros((2, 3), np.float32),
            env_state=np.arange(5, dtype=np.float32),
        ))
    # Set + inject: the sample carries the exact vector.
    env = np.array([1.5, 0.7, 1.1, 1.0, -0.2, 0.7], dtype=np.float32)
    bundle.install_live_object_condition(SimpleNamespace(
        positions=env.reshape(2, 3), env_state=env,
    ))
    sd = bundle._sample_dict(obs)  # noqa: SLF001
    assert "observation.environment_state" in sd
    np.testing.assert_allclose(sd["observation.environment_state"].cpu().numpy(), env)


def test_live_condition_rejected_on_unconditioned_policy():
    torch = pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    mod = _load_rollout()
    bundle = _fake_bundle(mod, torch)  # use_env_state=False
    bundle.scene_requirements = mod.SceneRequirements()
    with pytest.raises(RuntimeError, match="unconditioned checkpoint"):
        bundle.install_live_object_condition(SimpleNamespace(
            positions=np.zeros((2, 3), np.float32),
            env_state=np.zeros(6, dtype=np.float32),
        ))
