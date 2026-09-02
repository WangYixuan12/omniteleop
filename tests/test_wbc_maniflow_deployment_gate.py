from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "diagnostics"
        / "check_wbc_maniflow_deployment_gate.py"
    )
    name = "check_wbc_maniflow_deployment_gate_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_scheduler_simulation_allows_initial_wait_and_overlapping_replans():
    mod = _load_module()
    underflows = mod.simulated_underflow_count(
        queue_times=np.array([0.20, 0.50]),
        action_offsets=np.array([
            [0.30, 0.40, 0.50, 0.60, 0.70],
            [0.60, 0.70, 0.80, 0.90, 1.00],
        ]),
        dropped=np.array([0, 0]),
        command_dt=0.10,
    )
    assert underflows == 0


def test_scheduler_simulation_detects_armed_gap_before_late_replan():
    mod = _load_module()
    underflows = mod.simulated_underflow_count(
        queue_times=np.array([0.20, 0.80]),
        action_offsets=np.array([
            [0.30, 0.40],
            [0.90, 1.00],
        ]),
        dropped=np.array([0, 0]),
        command_dt=0.10,
    )
    assert underflows == 1


def test_policy_episode_key_preserves_source_and_natural_episode_order(tmp_path):
    mod = _load_module()
    root = tmp_path / "policy_io"
    paths = [
        root / "raw" / "episode_10.hdf5",
        root / "raw" / "episode_2.hdf5",
        root / "recovery" / "episode_1.hdf5",
    ]

    ordered = sorted(paths, key=lambda path: mod.policy_episode_key(path, root))

    assert ordered == [paths[1], paths[0], paths[2]]


def test_scalar_text_decodes_hdf5_byte_scalars():
    mod = _load_module()

    assert mod.scalar_text(np.asarray(b"grounding_tokens")) == "grounding_tokens"
    assert mod.scalar_text(np.asarray("world")) == "world"


def _write_unconditioned_policy_io(path: Path) -> None:
    path.parent.mkdir(parents=True)
    with h5py.File(path, "x") as data:
        data.create_dataset("timing/post_capture_s", data=np.array([0.05]))
        data.create_dataset(
            "timing/episode_min_free_bytes", data=np.array([3 * 1024**3], np.int64)
        )
        data.create_dataset("scheduler_s", data=np.array([0.001]))
        data.create_dataset("queue_offset_s", data=np.array([0.2]))
        data.create_dataset(
            "action_offsets_s",
            data=np.arange(0.3, 1.1, 0.1, dtype=np.float64)[None],
        )
        data.create_dataset("n_dropped", data=np.array([0], np.int64))
        schedule = data.create_group("schedule")
        schedule.create_dataset("dataset_fps", data=10.0)
        schedule.create_dataset("policy_interval_s", data=0.4)
        schedule.create_dataset("execution_latency_s", data=0.0)
        schedule.create_dataset("command_fps", data=10.0)
        schedule.create_dataset("n_action_steps", data=8)
        schedule.create_dataset("n_obs_steps", data=2)
        policy = data.create_group("policy")
        policy.create_dataset("checkpoint_sha256", data=np.bytes_("a" * 64))
        policy.create_dataset("weights", data=np.bytes_("ema_model"))
        policy.create_dataset("position_condition_mode", data=np.bytes_("none"))
        policy.create_dataset("point_sampling_mode", data=np.bytes_("fps"))
        policy.create_dataset("mask_channels", data=False)
        policy.create_dataset("action_frame_mode", data=np.bytes_("world"))
        policy.create_dataset("num_inference_steps", data=2)
        policy.create_dataset("pointcloud_meta_sha256", data=np.bytes_("b" * 64))
        policy.create_dataset("n_obs_steps", data=2)
        policy.create_dataset("n_action_steps", data=8)
        policy.create_dataset("dataset_fps", data=10.0)
        policy.create_dataset("policy_warmed_before_tracker", data=True)


def test_gate_main_accepts_complete_single_contract_evidence(tmp_path, monkeypatch):
    mod = _load_module()
    root = tmp_path / "policy_io"
    _write_unconditioned_policy_io(root / "episode_0.hdf5")
    report = tmp_path / "gate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_wbc_maniflow_deployment_gate.py",
            "--policy-io-root",
            str(root),
            "--report",
            str(report),
            "--min-episodes",
            "1",
            "--min-windows",
            "1",
        ],
    )

    mod.main()

    assert report.is_file()
    assert '"passed": true' in report.read_text()
