#!/usr/bin/env python3
"""Aggregate full-path policy-IO timing and scheduler evidence against ADR 0001."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def policy_episode_key(path: Path, root: Path) -> tuple[tuple[str, ...], int]:
    """Natural source/episode identity shared by evidence and cleanup artifacts."""
    relative = path.relative_to(root)
    try:
        episode_number = int(path.stem.split("_")[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"policy-IO filename is not episode_<N>: {path}") from exc
    return relative.parts[:-1], episode_number


def scalar_text(value) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode("utf-8") if isinstance(scalar, bytes) else str(scalar)


def timing_summary(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        raise ValueError("timing evidence is empty")
    return {
        "mean_s": float(values.mean()),
        "p50_s": float(np.percentile(values, 50)),
        "p95_s": float(np.percentile(values, 95)),
        "p99_s": float(np.percentile(values, 99)),
        "max_s": float(values.max()),
    }


def simulated_underflow_count(
    queue_times: np.ndarray,
    action_offsets: np.ndarray,
    dropped: np.ndarray,
    *,
    command_dt: float,
) -> int:
    """Replay queue/tick ordering through the rollout buffer's terminal semantics."""
    buffer: list[float] = []
    armed = False
    poisoned = False
    underflows = 0
    next_tick = 0.0

    def sample(tick: float) -> None:
        nonlocal poisoned, underflows
        passed = False
        while buffer and buffer[0] <= tick:
            buffer.pop(0)
            passed = True
        if not buffer and not passed and armed and not poisoned:
            poisoned = True
            underflows += 1

    for queue_time, offsets, count in zip(
        queue_times, action_offsets, dropped, strict=True
    ):
        while next_tick < queue_time:
            sample(next_tick)
            next_tick += command_dt
        kept = [float(value) for value in offsets[int(count):]]
        if kept and not poisoned:
            while buffer and buffer[-1] >= kept[0]:
                buffer.pop()
            buffer = [value for value in buffer if value >= queue_time]
            buffer.extend(kept)
            armed = True
        while next_tick <= queue_time:
            sample(next_tick)
            next_tick += command_dt
    return underflows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-io-root", type=Path, required=True)
    parser.add_argument("--cleanup-root", type=Path,
                        help="tracker cleanup reports (default: sibling artifacts/)")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--glob", default="episode_*.hdf5")
    parser.add_argument("--min-episodes", type=int, default=48)
    parser.add_argument("--min-windows", type=int, default=1000)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--dataset-fps", type=float, default=10.0)
    parser.add_argument("--policy-interval", type=float, default=0.4)
    parser.add_argument("--execution-latency", type=float, default=0.0)
    parser.add_argument("--command-fps", type=float, default=10.0)
    args = parser.parse_args()
    for name, value in (
        ("--dataset-fps", args.dataset_fps),
        ("--policy-interval", args.policy_interval),
        ("--command-fps", args.command_fps),
    ):
        if not np.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if not np.isfinite(args.execution_latency) or args.execution_latency < 0:
        parser.error("--execution-latency must be finite and non-negative")
    if args.n_action_steps < 1 or args.min_episodes < 1 or args.min_windows < 1:
        parser.error("step/episode/window minima must be positive")
    if args.report.exists() or args.report.is_symlink():
        raise FileExistsError(f"report exists: {args.report}; delete it manually")
    paths = list(args.policy_io_root.rglob(args.glob))
    if not paths:
        raise FileNotFoundError(f"no {args.glob} under {args.policy_io_root}")
    paths.sort(key=lambda path: policy_episode_key(path, args.policy_io_root))

    totals = []
    free_bytes = []
    underflows = 0
    windows = 0
    per_episode = []
    episode_keys: list[tuple[tuple[str, ...], int]] = []
    scene_modes = set()
    policy_contracts: set[tuple] = set()
    expected_schedule = {
        "dataset_fps": float(args.dataset_fps),
        "policy_interval_s": float(args.policy_interval),
        "execution_latency_s": float(args.execution_latency),
        "command_fps": float(args.command_fps),
        "n_action_steps": int(args.n_action_steps),
    }
    for path in paths:
        with h5py.File(path, "r") as data:
            post = np.asarray(data["timing/post_capture_s"], dtype=np.float64)
            scheduler = np.asarray(data["scheduler_s"], dtype=np.float64)
            queue_times = np.asarray(data["queue_offset_s"], dtype=np.float64)
            offsets = np.asarray(data["action_offsets_s"], dtype=np.float64)
            dropped = np.asarray(data["n_dropped"], dtype=np.int64)
            free = np.asarray(
                data["timing/episode_min_free_bytes"], dtype=np.int64
            )
            schedule = {
                key: np.asarray(data[f"schedule/{key}"]).item()
                for key in expected_schedule
            }
            scheduled_n_obs_steps = int(
                np.asarray(data["schedule/n_obs_steps"]).item()
            )
            position_mode = scalar_text(data["policy/position_condition_mode"])
            sampling_mode = scalar_text(data["policy/point_sampling_mode"])
            mask_channels = bool(np.asarray(data["policy/mask_channels"]).item())
            action_frame_mode = scalar_text(data["policy/action_frame_mode"])
            policy_n_obs_steps = int(
                np.asarray(data["policy/n_obs_steps"]).item()
            )
            policy_n_action_steps = int(
                np.asarray(data["policy/n_action_steps"]).item()
            )
            policy_dataset_fps = float(
                np.asarray(data["policy/dataset_fps"]).item()
            )
            policy_warmed = bool(
                np.asarray(data["policy/policy_warmed_before_tracker"]).item()
            )
            num_inference_steps = int(
                np.asarray(data["policy/num_inference_steps"]).item()
            )
            checkpoint_sha256 = scalar_text(data["policy/checkpoint_sha256"])
            pointcloud_meta_sha256 = scalar_text(
                data["policy/pointcloud_meta_sha256"]
            )
            weights = scalar_text(data["policy/weights"])
            for name, digest in (
                ("checkpoint_sha256", checkpoint_sha256),
                ("pointcloud_meta_sha256", pointcloud_meta_sha256),
            ):
                if len(digest) != 64 or any(
                    char not in "0123456789abcdef" for char in digest
                ):
                    raise ValueError(f"{path}: policy/{name} is not a SHA-256 digest")
            if weights not in {"model", "ema_model"}:
                raise ValueError(f"{path}: unsupported policy weights {weights!r}")
            if position_mode not in {"none", "concat", "grounding_tokens"}:
                raise ValueError(f"{path}: invalid position mode {position_mode!r}")
            if sampling_mode not in {"fps", "mask_stratified"}:
                raise ValueError(f"{path}: invalid sampling mode {sampling_mode!r}")
            if action_frame_mode not in {"world", "mof"}:
                raise ValueError(
                    f"{path}: invalid action frame mode {action_frame_mode!r}"
                )
            mof_contract = ()
            if action_frame_mode == "mof":
                mof_contract = (
                    tuple(
                        item.decode("utf-8") if isinstance(item, bytes) else str(item)
                        for item in np.asarray(
                            data["policy/mof_enabled_experts"]
                        ).tolist()
                    ),
                    scalar_text(data["policy/mof_canonical"]),
                    scalar_text(data["policy/mof_router_mode"]),
                )
            scene_modes.add((position_mode, sampling_mode, mask_channels))
            policy_contracts.add((
                checkpoint_sha256,
                weights,
                position_mode,
                sampling_mode,
                mask_channels,
                action_frame_mode,
                num_inference_steps,
                pointcloud_meta_sha256,
                policy_n_obs_steps,
                policy_n_action_steps,
                policy_dataset_fps,
                mof_contract,
            ))
        mismatched_schedule = {
            key: (schedule[key], expected)
            for key, expected in expected_schedule.items()
            if schedule[key] != expected
        }
        if mismatched_schedule:
            raise ValueError(
                f"{path}: recorded schedule differs from gate arguments: "
                f"{mismatched_schedule}"
            )
        if scheduled_n_obs_steps != policy_n_obs_steps:
            raise ValueError(
                f"{path}: schedule n_obs_steps={scheduled_n_obs_steps} != policy "
                f"{policy_n_obs_steps}"
            )
        if policy_n_action_steps != args.n_action_steps:
            raise ValueError(
                f"{path}: policy n_action_steps={policy_n_action_steps} != gate "
                f"{args.n_action_steps}"
            )
        if not np.isclose(
            policy_dataset_fps, args.dataset_fps, atol=1e-9, rtol=0.0
        ):
            raise ValueError(
                f"{path}: policy dataset_fps={policy_dataset_fps} != gate "
                f"{args.dataset_fps}"
            )
        if not policy_warmed:
            raise ValueError(f"{path}: policy was not warmed before tracker loading")
        if num_inference_steps < 1:
            raise ValueError(f"{path}: num_inference_steps must be positive")
        if scheduled_n_obs_steps < 1:
            raise ValueError(f"{path}: n_obs_steps must be positive")
        row_count = len(post)
        if row_count == 0:
            raise ValueError(f"{path}: policy-IO evidence has no windows")
        if any(len(value) != row_count for value in (
            scheduler, queue_times, offsets, dropped, free
        )):
            raise ValueError(f"{path}: policy-IO telemetry lengths disagree")
        if (
            not np.all(np.isfinite(post))
            or not np.all(np.isfinite(scheduler))
            or not np.all(np.isfinite(queue_times))
            or np.any(post < 0)
            or np.any(scheduler < 0)
            or np.any(queue_times < 0)
            or np.any(np.diff(queue_times) <= 0)
        ):
            raise ValueError(f"{path}: timing/queue telemetry is invalid")
        if offsets.shape != (row_count, args.n_action_steps):
            raise ValueError(
                f"{path}: action_offsets_s {offsets.shape} != "
                f"({row_count}, {args.n_action_steps})"
            )
        if np.any(dropped < 0) or np.any(dropped > args.n_action_steps):
            raise ValueError(f"{path}: n_dropped is outside the action chunk")
        if (
            not np.all(np.isfinite(offsets))
            or np.any(np.diff(offsets, axis=1) <= 0)
        ):
            raise ValueError(f"{path}: action_offsets_s is not finite/increasing")
        if np.any(free < 0) or np.any(np.diff(free) > 0):
            raise ValueError(
                f"{path}: episode_min_free_bytes is missing or not a running minimum"
            )
        episode_keys.append(policy_episode_key(path, args.policy_io_root))
        episode_underflows = simulated_underflow_count(
            queue_times,
            offsets,
            dropped,
            command_dt=1.0 / args.command_fps,
        )
        underflows += episode_underflows
        windows += len(post)
        totals.append(post + scheduler)
        free_bytes.append(free[free >= 0])
        per_episode.append({
            "path": str(path),
            "windows": len(post),
            "simulated_underflows": int(episode_underflows),
            "timing": timing_summary(post + scheduler),
        })

    total = np.concatenate(totals)
    free_chunks = [value for value in free_bytes if len(value)]
    valid_free = (
        np.concatenate(free_chunks) if free_chunks else np.empty(0, dtype=np.int64)
    )
    dataset_dt = 1.0 / args.dataset_fps
    deadline = (
        (args.n_action_steps - 1) * dataset_dt
        - args.policy_interval
        - args.execution_latency
    )
    if deadline <= 0:
        raise ValueError(f"configured deadline is non-positive: {deadline:.6f}s")
    p99_limit = deadline - 0.050
    failures = []
    if len(paths) < args.min_episodes:
        failures.append(f"episodes {len(paths)} < {args.min_episodes}")
    if windows < args.min_windows:
        failures.append(f"windows {windows} < {args.min_windows}")
    if float(total.max()) >= deadline:
        failures.append(f"maximum {total.max():.6f}s is not below {deadline:.6f}s")
    if float(np.percentile(total, 99)) > p99_limit:
        failures.append(
            f"p99 {np.percentile(total, 99):.6f}s exceeds {p99_limit:.6f}s"
        )
    if underflows:
        failures.append(f"scheduler simulation found {underflows} underflow(s)")
    if len(valid_free) == 0:
        failures.append("no device-free-memory telemetry")
        minimum_free = None
    else:
        minimum_free = int(valid_free.min())
        if minimum_free < 2 * 1024**3:
            failures.append(f"minimum free device memory {minimum_free / 2**30:.2f} GiB < 2")

    if len(scene_modes) != 1:
        failures.append(f"policy-IO files contain inconsistent modes: {sorted(scene_modes)}")
    if len(policy_contracts) != 1:
        failures.append(
            f"policy-IO files contain {len(policy_contracts)} distinct checkpoint contracts"
        )
    position_mode, sampling_mode, mask_channels = next(iter(scene_modes))
    tracker_applicable = (
        position_mode != "none"
        or sampling_mode == "mask_stratified"
        or mask_channels
    )
    cleanup_summary: dict = {"applicable": tracker_applicable}
    if tracker_applicable:
        cleanup_root = (
            args.cleanup_root
            if args.cleanup_root is not None
            else args.policy_io_root.parent / "artifacts"
        )
        cleanup_paths = [
            cleanup_root.joinpath(*parent_parts, f"episode_{episode}",
                                  "tracker_cleanup_report.json")
            for parent_parts, episode in episode_keys
        ]
        cleanup_reports = []
        for cleanup_path in cleanup_paths:
            if not cleanup_path.is_file():
                failures.append(
                    f"{cleanup_path}: cleanup report for policy-IO episode is missing"
                )
                continue
            cleanup = json.loads(cleanup_path.read_text())
            if cleanup.get("schema") != "sam31_tracker_cleanup_v1":
                failures.append(f"{cleanup_path}: invalid cleanup schema")
                continue
            cleanup_reports.append((cleanup_path, cleanup))
            if not cleanup.get("minimum_free_passed", False):
                failures.append(f"{cleanup_path}: tracker peak-memory gate failed")
            if not cleanup.get("recovery_passed", False):
                failures.append(f"{cleanup_path}: tracker recovery gate failed")
            if cleanup.get("cleanup_errors"):
                failures.append(f"{cleanup_path}: tracker cleanup errors were recorded")
        if len(cleanup_reports) < 3:
            failures.append(
                f"tracker cleanup reports {len(cleanup_reports)} < 3 consecutive cycles"
            )
        recovered = [
            int(report["recovered_used_bytes"])
            for _path, report in cleanup_reports
            if report.get("recovered_used_bytes") is not None
        ]
        monotonic_triples = [
            index
            for index in range(max(0, len(recovered) - 2))
            if recovered[index] < recovered[index + 1] < recovered[index + 2]
        ]
        if monotonic_triples:
            failures.append(
                "tracker recovered device use grows over consecutive three-cycle "
                f"windows starting at {monotonic_triples}"
            )
        cleanup_summary.update({
            "root": str(cleanup_root),
            "reports": len(cleanup_reports),
            "recovered_used_bytes": recovered,
            "monotonic_growth_triple_starts": monotonic_triples,
        })

    policy_contract_value = (
        next(iter(policy_contracts)) if len(policy_contracts) == 1 else None
    )
    policy_contract_report = None
    if policy_contract_value is not None:
        policy_contract_report = {
            "checkpoint_sha256": policy_contract_value[0],
            "weights": policy_contract_value[1],
            "position_condition_mode": policy_contract_value[2],
            "point_sampling_mode": policy_contract_value[3],
            "mask_channels": policy_contract_value[4],
            "action_frame_mode": policy_contract_value[5],
            "num_inference_steps": policy_contract_value[6],
            "pointcloud_meta_sha256": policy_contract_value[7],
            "n_obs_steps": policy_contract_value[8],
            "n_action_steps": policy_contract_value[9],
            "dataset_fps": policy_contract_value[10],
            "mof": policy_contract_value[11],
        }
    report = {
        "schema": "wbc_maniflow_deployment_gate_v1",
        "passed": not failures,
        "failures": failures,
        "episodes": len(paths),
        "windows": windows,
        "deadline_s": deadline,
        "p99_limit_s": p99_limit,
        "timing": timing_summary(total),
        "simulated_underflows": underflows,
        "minimum_device_free_bytes": minimum_free,
        "tracker_cleanup": cleanup_summary,
        "schedule": expected_schedule,
        "policy_contract": policy_contract_report,
        "per_episode": per_episode,
        "note": (
            "This aggregates synchronized full-path rollout telemetry and the exact "
            "episode-matched tracker cleanup reports. Cadence-drift review and fixed/"
            "stream parity remain separate gates."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError(f"deployment gate failed; see {args.report}")


if __name__ == "__main__":
    main()
