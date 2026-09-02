#!/usr/bin/env python3
"""Run the exact live ManiFlow/SAM/cloud path over accepted raw episodes, without a robot."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from omniteleop.common.recorder import EpisodeRecorder
from omniteleop.follower.live_object_condition import (
    LiveObjectCondition,
    LiveValidationFrame,
)
from omniteleop.follower.scenediff_live import load_live_tracker_api
from omniteleop.wbc_artifacts import (
    DINO_INPUT_SCALE,
    DINO_MODEL,
    DINO_SOURCE,
    load_dino_artifact,
    validate_mask_artifact,
)
from omniteleop.wbc_pointcloud import world_t_head_from_state
from omniteleop.wbc_policy_format import WBCPolicyFK


def _load_script(name: str, filename: str):
    path = Path(__file__).resolve().parents[1] / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


maniflow_rollout = _load_script(
    "wbc_maniflow_deployment_replay_bundle", "wbc_maniflow_rollout.py"
)
porter = _load_script(
    "wbc_maniflow_deployment_replay_porter", "port_wbc_mobile_hdf5.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _episode_number(path: Path) -> int:
    return int(path.stem.split("_")[1])


def rollout_windows(
    frame_count: int,
    *,
    dataset_fps: float,
    policy_interval: float,
    n_obs_steps: int,
    max_seconds: float,
    frame_start: int = 0,
    frame_end: int | None = None,
) -> list[np.ndarray]:
    """Seed is raw frame 0; policy bursts use the porter's exact kept interval."""
    stride_float = dataset_fps * policy_interval
    stride = round(stride_float)
    if not np.isclose(stride_float, stride, atol=1e-9, rtol=0.0):
        raise ValueError(
            "policy_interval * dataset_fps must be an integer frame stride; "
            f"got {stride_float}"
        )
    if stride < n_obs_steps:
        raise ValueError("policy interval is shorter than the observation burst")
    frame_start = int(frame_start)
    limit = frame_count if frame_end is None else int(frame_end)
    if not 0 <= frame_start < limit <= frame_count:
        raise ValueError(
            f"invalid kept frame interval [{frame_start}, {limit}) for T={frame_count}"
        )
    if max_seconds > 0:
        limit = min(
            limit,
            frame_start + int(np.floor(max_seconds * dataset_fps)) + 1,
        )
    windows = []
    for first in range(frame_start + stride, limit, stride):
        indices = np.arange(first, first + n_obs_steps, dtype=np.int64)
        if int(indices[-1]) >= limit:
            break
        windows.append(indices)
    return windows


def _wait_for_save(recorder: EpisodeRecorder) -> None:
    while recorder.saving:
        time.sleep(0.05)
    if recorder.last_save_error is not None:
        raise RuntimeError(
            f"policy-IO evidence save failed: {recorder.last_save_error}"
        ) from recorder.last_save_error


def _state_at(raw, frame_index: int, fk: WBCPolicyFK, state_frame: str) -> np.ndarray:
    action_eef, action_head = porter.load_action_targets_world(raw, frame_index)
    state, _action = porter.compute_frame_state_action(
        raw,
        frame_index,
        fk,
        action_eef,
        action_head,
        state_frame=state_frame,
    )
    return np.asarray(state, dtype=np.float32)


def _condition_for_episode(
    *,
    bundle,
    tracker_api,
    mask_path: Path,
    source: str,
    episode_number: int,
    scene_diff_root: Path,
    evidence_dir: Path,
    raw_hdf5: Path,
    raw,
    fk: WBCPolicyFK,
) -> LiveObjectCondition:
    with np.load(mask_path, allow_pickle=False) as mask_artifact:
        positions_npz = Path(str(mask_artifact["positions_npz"]))
    with np.load(positions_npz, allow_pickle=False) as positions_artifact:
        positions_size = np.asarray(positions_artifact["positions"], np.float32)
        order = np.asarray(positions_artifact["order"], np.int64)
        obj_ids_size = np.asarray(
            positions_artifact["scene_diff_obj_ids"], np.int64
        )
        valid = np.asarray(positions_artifact["valid"], np.float32)
    object_nums = bundle.scene_requirements.object_nums
    if (
        positions_size.shape != (object_nums, 3)
        or order.shape != (object_nums,)
        or not np.array_equal(np.sort(order), np.arange(object_nums))
        or not np.array_equal(valid, np.ones(object_nums, np.float32))
    ):
        raise ValueError(f"{positions_npz}: invalid live-condition fields")
    seed, _seed_info = tracker_api.build_frame0_seed(
        scene_diff_root / source / f"episode_{episode_number}",
        positions_npz,
    )
    dino = None
    if bundle.scene_requirements.needs_env_dino:
        sidecar = positions_npz.with_name(positions_npz.stem + "_dino.npz")
        dino = load_dino_artifact(
            positions_npz, sidecar, object_nums=object_nums
        ).reshape(object_nums, -1)
    state0 = _state_at(raw, 0, fk, bundle.state_frame)
    timestamp0 = int(raw["timestamp_ns"][0])
    source_artifact = evidence_dir / "bootstrap_size_slots.npz"
    with source_artifact.open("xb") as stream:
        np.savez_compressed(
            stream,
            positions=positions_size,
            order=order,
            scene_diff_obj_ids=obj_ids_size,
            seed_task_slots=seed,
            source_positions_npz=np.asarray(str(positions_npz)),
            source_mask_npz=np.asarray(str(mask_path)),
        )
    return LiveObjectCondition(
        order=order,
        positions=positions_size[order],
        scene_diff_obj_ids=obj_ids_size[order],
        seed_task_slots=seed,
        dino_region_feats=dino,
        bootstrap_rgb=np.asarray(raw["obs/images/head_left_rgb"][0]),
        bootstrap_depth_mm=np.asarray(raw["obs/images/head_depth"][0]),
        intrinsic=np.asarray(raw["obs/images/intrinsic"], np.float32),
        world_t_cam=world_t_head_from_state(
            state0, bundle.state_frame
        ).astype(np.float32),
        camera_timestamp_ns=timestamp0,
        world_frame_epoch=0,
        source_artifact=source_artifact,
        provenance_hashes=(
            _sha256(positions_npz),
            _sha256(mask_path),
            _sha256(raw_hdf5),
        ),
        dino_model=DINO_MODEL if dino is not None else None,
        dino_source=DINO_SOURCE if dino is not None else None,
        dino_input_scale=DINO_INPUT_SCALE if dino is not None else None,
    )


def _raw_frame(bundle, raw, frame_index: int, fk: WBCPolicyFK):
    state = _state_at(raw, frame_index, fk, bundle.state_frame)
    return maniflow_rollout.ManiFlowRawFrame(
        rgb=np.asarray(raw["obs/images/head_left_rgb"][frame_index]),
        depth_mm=np.asarray(raw["obs/images/head_depth"][frame_index]),
        state=state,
        head_timestamp_ns=int(raw["timestamp_ns"][frame_index]),
        world_t_cam=world_t_head_from_state(
            state, bundle.state_frame
        ).astype(np.float32),
        world_frame_epoch=0,
    )


def _record_episode(
    *,
    bundle,
    raw_hdf5: Path,
    mask_path: Path,
    source: str,
    episode_number: int,
    args,
    tracker_api,
    fk: WBCPolicyFK,
    trim: tuple[int, int] | None,
) -> Path:
    evidence_dir = args.out_root / "artifacts" / source / f"episode_{episode_number}"
    evidence_dir.mkdir(parents=True)
    recorder = EpisodeRecorder(str(args.out_root / "policy_io" / source))
    recorder.episode_id = episode_number
    recorder.start()
    runtime_started = False
    body_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    output: str | None = None
    try:
        with h5py.File(raw_hdf5, "r") as raw:
            frame_count = len(raw["obs/images/head_left_rgb"])
            t0 = porter.first_valid_frame(
                np.asarray(raw["obs/gripper/left"]),
                np.asarray(raw["obs/gripper/right"]),
                raw_hdf5,
            )
            kept_start, kept_end = porter.resolve_episode_window(
                frame_count, t0, trim, raw_hdf5
            )
            windows = rollout_windows(
                frame_count,
                dataset_fps=args.dataset_fps,
                policy_interval=args.policy_interval,
                n_obs_steps=bundle.n_obs_steps,
                max_seconds=args.max_seconds,
                frame_start=kept_start,
                frame_end=kept_end,
            )
            if not windows:
                raise ValueError(f"{raw_hdf5}: no complete replay windows")

            bundle.reset()
            if bundle.scene_requirements.needs_bootstrap:
                condition = _condition_for_episode(
                    bundle=bundle,
                    tracker_api=tracker_api,
                    mask_path=mask_path,
                    source=source,
                    episode_number=episode_number,
                    scene_diff_root=args.scene_diff_root,
                    evidence_dir=evidence_dir,
                    raw_hdf5=raw_hdf5,
                    raw=raw,
                    fk=fk,
                )
                validation_index = max(1, int(windows[0][0]) - 1)
                validation_raw = _raw_frame(bundle, raw, validation_index, fk)
                bundle.validate_live_object_condition(
                    condition,
                    capture_validation_frame=lambda: LiveValidationFrame(
                        rgb=validation_raw.rgb,
                        depth_mm=validation_raw.depth_mm,
                        world_t_cam=validation_raw.world_t_cam,
                        head_timestamp_ns=validation_raw.head_timestamp_ns,
                        world_frame_epoch=validation_raw.world_frame_epoch,
                    ),
                    stage_dir=evidence_dir,
                )
                bundle.install_live_object_condition(condition)

            bundle.start_episode_runtime(
                max_seconds=args.max_seconds,
                policy_interval=args.policy_interval,
                dataset_dt=1.0 / args.dataset_fps,
            )
            runtime_started = True
            static = {"policy": bundle.inference_metadata()}
            condition_metadata = bundle.live_condition_metadata()
            if condition_metadata is not None:
                static["live_object_condition"] = condition_metadata
            static["replay"] = {
                "source": np.asarray(source.encode()),
                "raw_hdf5": np.asarray(str(raw_hdf5).encode()),
                "mask_artifact": np.asarray(str(mask_path).encode()),
                "kept_raw_frame_start": np.asarray(kept_start, np.int64),
                "kept_raw_frame_end_exclusive": np.asarray(kept_end, np.int64),
            }
            static["schedule"] = {
                "dataset_fps": np.asarray(args.dataset_fps, np.float64),
                "policy_interval_s": np.asarray(args.policy_interval, np.float64),
                "execution_latency_s": np.asarray(
                    args.execution_latency, np.float64
                ),
                "command_fps": np.asarray(args.command_fps, np.float64),
                "n_action_steps": np.asarray(bundle.n_action_steps, np.int64),
                "n_obs_steps": np.asarray(bundle.n_obs_steps, np.int64),
            }
            recorder.set_static(static)

            dataset_dt = 1.0 / args.dataset_fps
            for window_index, indices in enumerate(windows):
                raws = [
                    _raw_frame(bundle, raw, int(index), fk) for index in indices
                ]
                observations = bundle.build_observations(raws)
                chunk = bundle.predict_chunk(observations)
                policy_end = time.perf_counter()
                logical_t_obs = float(indices[-1] - kept_start) * dataset_dt
                action_offsets = logical_t_obs + dataset_dt * np.arange(
                    len(chunk), dtype=np.float64
                )
                actions = maniflow_rollout.rollout.scheduled_actions_from_chunk(
                    chunk, action_offsets
                )
                trace = bundle.last_inference_trace()
                post_capture_s = float(trace["timing"]["post_capture_s"])
                stale_cutoff_offset = (
                    logical_t_obs
                    + post_capture_s
                    + (time.perf_counter() - policy_end)
                )
                kept = maniflow_rollout.rollout.drop_stale_actions(
                    actions, stale_cutoff_offset + args.execution_latency
                )
                scheduler_s = time.perf_counter() - policy_end
                queue_offset = logical_t_obs + post_capture_s + scheduler_s
                prospective_coverage = (
                    kept[-1].timestamp - stale_cutoff_offset if kept else 0.0
                )
                row = {
                    "t": np.float64(logical_t_obs),
                    "timestamp_ns": np.int64(raws[-1].head_timestamp_ns),
                    "state": observations[-1].state,
                    "action": chunk,
                    "action_offsets_s": action_offsets,
                    "n_dropped": np.int64(len(chunk) - len(kept)),
                    "scheduler_s": np.float32(scheduler_s),
                    "queue_offset_s": np.float64(queue_offset),
                    "prospective_coverage_s": np.float32(prospective_coverage),
                    "window_index": np.int64(window_index),
                }
                row.update(trace)
                recorder.record(row)
    except BaseException as exc:
        body_error = exc
    finally:
        if runtime_started or bundle.scene_requirements.needs_bootstrap:
            try:
                bundle.finish_episode_runtime()
            except BaseException as exc:
                cleanup_error = exc
        try:
            output = recorder.stop()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
            else:
                cleanup_error.add_note(
                    f"recorder stop also failed: {type(exc).__name__}: {exc}"
                )
        try:
            _wait_for_save(recorder)
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
            else:
                cleanup_error.add_note(
                    f"evidence save also failed: {type(exc).__name__}: {exc}"
                )
    if body_error is not None:
        if cleanup_error is not None:
            body_error.add_note(
                f"cleanup also failed: {type(cleanup_error).__name__}: "
                f"{cleanup_error}"
            )
        raise body_error
    if cleanup_error is not None:
        raise cleanup_error
    if output is None:
        raise RuntimeError(f"{raw_hdf5}: no policy-IO evidence was recorded")
    return Path(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pointcloud-meta", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--scene-diff-root", type=Path, required=True)
    parser.add_argument("--scene-diff-repo", type=Path,
                        default=Path("/home/yixuan/scene_diff"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", default=["raw", "recovery"])
    parser.add_argument("--expected-episodes", type=int, default=48)
    parser.add_argument("--dataset-fps", type=float, default=10.0)
    parser.add_argument("--policy-interval", type=float, default=0.4)
    parser.add_argument("--execution-latency", type=float, default=0.0)
    parser.add_argument("--command-fps", type=float, default=10.0)
    parser.add_argument("--max-seconds", type=float, default=85.0)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()
    if args.out_root.exists() or args.out_root.is_symlink():
        raise FileExistsError(f"output exists: {args.out_root}; delete it manually")
    if (
        not np.isfinite(args.dataset_fps)
        or args.dataset_fps <= 0
        or not np.isfinite(args.policy_interval)
        or args.policy_interval <= 0
        or not np.isfinite(args.execution_latency)
        or args.execution_latency < 0
        or not np.isfinite(args.command_fps)
        or args.command_fps <= 0
        or not np.isfinite(args.max_seconds)
        or args.max_seconds < 0
        or args.expected_episodes < 1
    ):
        parser.error("cadence/count arguments are invalid")

    artifacts = []
    for source in args.sources:
        artifacts.extend(
            (source, path)
            for path in (args.masks_dir / source).glob("episode_*.npz")
        )
    artifacts.sort(key=lambda item: (item[0], _episode_number(item[1])))
    if len(artifacts) != args.expected_episodes:
        raise ValueError(
            f"found {len(artifacts)} mask artifacts, expected {args.expected_episodes}"
        )
    episode_inputs = []
    trim_specs: dict[Path, dict[int, tuple[int, int]]] = {}
    for source, mask_path in artifacts:
        with np.load(mask_path, allow_pickle=False) as data:
            raw_hdf5 = Path(str(data["raw_hdf5"]))
            positions_npz = Path(str(data["positions_npz"]))
            if str(data["split"]) != source:
                raise ValueError(f"{mask_path}: source provenance mismatch")
        if not raw_hdf5.is_file():
            raise FileNotFoundError(raw_hdf5)
        episode_number = _episode_number(mask_path)
        validate_mask_artifact(
            mask_path,
            source=source,
            episode_number=episode_number,
            raw_hdf5=raw_hdf5,
            positions_npz=positions_npz,
        )
        trim = None
        if source == "recovery":
            recovery_dir = raw_hdf5.parent.resolve()
            if recovery_dir not in trim_specs:
                trim_specs[recovery_dir] = porter.load_trim_spec(recovery_dir)
            if episode_number not in trim_specs[recovery_dir]:
                raise ValueError(
                    f"{raw_hdf5}: recovery episode has no trim.csv row"
                )
            trim = trim_specs[recovery_dir][episode_number]
        episode_inputs.append(
            (source, episode_number, mask_path, raw_hdf5, trim)
        )

    bundle = maniflow_rollout.ManiFlowBundle(
        str(args.checkpoint),
        num_inference_steps=args.num_inference_steps,
        use_ema=not args.no_ema,
        meta_path=str(args.pointcloud_meta),
        scene_diff_repo=str(args.scene_diff_repo),
    )
    if not np.isclose(bundle.dataset_fps, args.dataset_fps, atol=1e-9, rtol=0.0):
        raise ValueError(
            f"checkpoint metadata fps {bundle.dataset_fps} != {args.dataset_fps}"
        )
    tracker_api = (
        load_live_tracker_api(args.scene_diff_repo)
        if bundle.scene_requirements.needs_bootstrap else None
    )
    fk = WBCPolicyFK()
    args.out_root.mkdir(parents=True)
    outputs = []
    try:
        for index, (source, episode_number, mask_path, raw_hdf5, trim) in enumerate(
            episode_inputs, start=1
        ):
            print(
                f"[{index}/{len(episode_inputs)}] full-stack replay "
                f"{source}/episode_{episode_number}",
                flush=True,
            )
            outputs.append(_record_episode(
                bundle=bundle,
                raw_hdf5=raw_hdf5,
                mask_path=mask_path,
                source=source,
                episode_number=episode_number,
                args=args,
                tracker_api=tracker_api,
                fk=fk,
                trim=trim,
            ))
    finally:
        try:
            bundle.finish_episode_runtime()
        except BaseException:
            pass
        del bundle
        gc.collect()

    manifest = args.out_root / "manifest.txt"
    with manifest.open("x") as stream:
        stream.write("\n".join(str(path) for path in outputs) + "\n")
    print(
        f"full-stack policy-IO evidence -> {args.out_root / 'policy_io'}\n"
        "Run check_wbc_maniflow_deployment_gate.py on that directory next."
    )


if __name__ == "__main__":
    main()
