from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "diagnostics"
        / "replay_wbc_maniflow_deployment.py"
    )
    name = "replay_wbc_maniflow_deployment_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_recovery_windows_start_from_porter_kept_interval():
    mod = _load_module()

    windows = mod.rollout_windows(
        669,
        dataset_fps=10.0,
        policy_interval=0.4,
        n_obs_steps=2,
        max_seconds=0,
        frame_start=409,
        frame_end=669,
    )

    assert windows[0].tolist() == [413, 414]
    assert windows[-1].tolist() == [665, 666]


def test_replay_duration_limit_is_relative_to_kept_start():
    mod = _load_module()

    windows = mod.rollout_windows(
        669,
        dataset_fps=10.0,
        policy_interval=0.4,
        n_obs_steps=2,
        max_seconds=0.8,
        frame_start=409,
        frame_end=669,
    )

    assert [window.tolist() for window in windows] == [[413, 414]]


def test_replay_windows_reject_invalid_kept_interval():
    mod = _load_module()

    with pytest.raises(ValueError, match="invalid kept frame interval"):
        mod.rollout_windows(
            20,
            dataset_fps=10.0,
            policy_interval=0.4,
            n_obs_steps=2,
            max_seconds=0,
            frame_start=20,
            frame_end=20,
        )
