#!/usr/bin/env python3
"""WBC mobile-policy rollout: 10 Hz policy ticks over the 100 Hz whole-body IK loop.

Drives the real robot from a LeRobot checkpoint trained on the
``port_wbc_mobile_hdf5.py`` schema (PLAN.md Conventions):

  observation.state (32,) = base-frame ACHIEVED L/R EEF pos3+rot6+gripper and
      zed_depth_frame head pos3+rot6 (WBC FK on measured joints, base zero) +
      the canonical measured base pose (x, y, yaw) in the engage-origin world
      (wheel odometry or ARKit, as declared by the training dataset).
  action (29,) = WORLD-frame L/R EEF + head targets (pos3+rot6) + binary grippers,
      for the original whole-body-base controller; or
  action (32,) = current-base versions of those 29 values + body-frame joystick
      chassis intent (vx, vy, wz), for joystick-collected policies.

The action is decoded with NO base composition: ``split_policy_action`` rebuilds
the three 4x4 poses (Gram-Schmidt on the 6-D rotation). WBC world targets go to
the whole-body solver; joystick current-base targets go to base-locked arm IK,
while the appended chassis intent is held ZOH at 10 Hz and processed by the same
causal 100 Hz joystick integrator/PD/slew/dispatch path used for collection. The
learned/replayed head
target is already the post-LPF/post-deadband solver input recorded in
``action/head``, so rollout does not filter it a second time. Engage mirrors
teleop: ``ik.reset()`` +
``OdometryThread.reset_origin()`` re-anchor the world frame at the rollout
start pose.

The base-pose source is a deployment contract, not a tuning choice. New LeRobot
checkpoints auto-resolve it from ``train_config.json -> dataset.root ->
dexmate_meta.json``; moved datasets use ``--policy-dataset-meta`` and legacy
checkpoints require ``--policy-base-pose-source wheel_odometry``. Rollout refuses
to construct the hardware driver when this source disagrees with ``--arkit-base``.
For a checkpoint with wrist-image inputs, the ported dataset also declares the exact
camera selection contract (clock domain, head-to-wrist skew ceiling, and capture-age
ceiling). Rollout requires verified head-capture-nearest training data and refuses to
construct hardware unless the live ``--record-max-camera-{skew,age}-ms`` values match.

Hardware access reuses ``wbc_vr_robot.HardwareDriver`` (same homing gate,
per-tick joint clamp, base PD/shaping, gripper pass-through, Robotiq FC03
obs monitors, cameras) so a rollout obeys the identical safety behavior as
data collection. Every rollout records:

  <save-dir>/episode_N.hdf5           the standard recorder schema (action/eef,
                                      action/head = the POLICY world targets)
  <save-dir>/policy_io/episode_N.hdf5 live rollout: ONE record per predicted
                                      chunk (state, full (n, 29) chunk, schedule
                                      offsets, timing, and live-condition
                                      instrumentation); replay: one record per
                                      released command (state/action vectors)

After timeout, replay completion, or Ctrl-C during motion, the current episode
is stopped and fully saved while the policy and robot connection stay live.
Press Enter at the idle prompt to home and start another rollout; press Ctrl-C
at that prompt to shut down. Position-conditioned policies rerun SceneDiff for
every new rollout before recording and inference begin.

Inference is asynchronous (ported from deps/rby1-wbc, see plan.md): an
inference worker thread gathers ``n_obs_steps`` fresh observations at the
dataset cadence, predicts a WHOLE action chunk every ``--policy-interval``
seconds, stamps frame k with the wall-clock time
``t_obs + (training_action_offset + k) / dataset_fps``,
drops frames already unreachable by ``inference_end + --execution-latency``,
and queues the rest into a ``ScheduledPolicyAction`` buffer. The main 100 Hz
loop runs a 10 Hz sampler that lerps/slerps the scheduled trajectory at "now"
and pushes one ``TargetInterpolator`` segment -- so a slow inference never
stalls a WBC tick. If inference dies or the buffer runs dry, ``last_cmd_wall``
stops advancing and the existing ``--source-timeout`` watchdog holds the robot.
Scheduling math lives entirely on the workstation ``perf_counter`` timeline;
the camera ``timestamp_ns`` (SDK capture time on the camera-publisher-host
clock) is used for same-publisher freshness gating and, for multi-camera
policies, mapped through the recording-time camera clock calibrations so each
live head observation receives the same capture-nearest wrist pairing used by
the training recorder.

Run in the dexmate_lerobot conda env ON the robot (needs lerobot + the hardware
SDK). Checkpoints emit absolute actions: observation.state EEF/head are
BASE-frame and the policy emits world-frame targets directly. MoF checkpoints
instead declare the frame they want via ``mof_state_frame``.

Two sources can drive the loop (exactly one of ``--policy-path`` /
``--replay-episode``): a live policy, or a recorded ``wbc_vr_robot.py --record``
episode replayed VERBATIM at real time (1x) through the very same
``TargetInterpolator`` -> ``ik.solve()`` -> ``actuate()`` path. The one replay
difference: the stateful head LPF/planar-deadband filters are SKIPPED, because
``action/head`` was recorded post-filter (the exact ``ik.solve()`` input) and
re-filtering would double-process it. Replay still assembles the full
observation (cameras + 32-D state) every command tick so its ``policy_io`` log
diffs directly against a live rollout.

``--align-reference FILE`` (required; pass ``none`` to deliberately skip) makes
both modes start from the same physical configuration: after engage the robot
autonomously glides its arms to the reference EEF poses (the leader's
``load_reference_ee_poses``, head held at nominal), waits until the MEASURED
poses hold within the leader's alignment tolerances, then blocks on Enter
before the take actually starts. Recording begins only after that gate.

Live object conditioning is activated only by checkpoint-declared scene requirements.
After engage and reference alignment, the rollout captures a head RGB-D frame with its
engage-origin ``world_T_zed``, runs the short-lived SceneDiff bootstrap, and requires the
operator to map the fresh size-slot overlay to ``[src box, dst cloth]``. Positions, optional
DINO descriptors, and seed labels are reordered together, checked against a separately
captured validation frame, then installed atomically before recording or policy motion.

Usage:
  python scripts/wbc_policy_rollout.py --policy-path /path/to/checkpoint \
      --align-reference ~/Dexmate/data/raw_data/reference.hdf5 \
      [--auto-start] [--max-seconds 120] [--save-dir ~/Dexmate/data/raw_data_rollout]
  # A conditioned checkpoint activates SceneDiff automatically:
  python scripts/wbc_policy_rollout.py --policy-path /path/to/conditioned_checkpoint \
      --align-reference ~/Dexmate/data/raw_data/reference.hdf5
  python scripts/wbc_policy_rollout.py \
      --replay-episode ~/Dexmate/data/raw_data/episode_0.hdf5 \
      --align-reference ~/Dexmate/data/raw_data/reference.hdf5
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import itertools
import json
import shutil
import signal
import tempfile
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from omniteleop.follower.live_object_condition import (
    LiveObjectCondition,
    LiveValidationFrame,
    SceneRequirements,
)
from omniteleop.follower.scenediff_live import (
    invoke_scenediff as _invoke_scenediff,
)
from omniteleop.follower.scenediff_live import (
    load_live_object_artifacts,
    load_live_tracker_api,
    prompt_for_slot_order,
)
from omniteleop.follower.scenediff_live import (
    write_live_capture_hdf5 as _write_live_capture_hdf5,
)
from omniteleop.wbc_artifacts import validate_live_condition_with_tracker
from omniteleop.wbc_pointcloud import MAX_DEPTH_M, MIN_DEPTH_M
from omniteleop.wbc_policy_format import (
    ACTION_AXES,
    GRIPPER_BINARY_THRESHOLD,
    JOYSTICK_ACTION_AXES,
    JOYSTICK_POLICY_ACTION_SCHEMA,
    STATE_AXES,
    WBC_POLICY_ACTION_SCHEMA,
    WBCPolicyFK,
    action_axes_for_policy_schema,
    base_pose_to_mat,
    build_state_vector,
    mat_to_pos6d,
    pos6d_to_mat,
)
from omniteleop.wbc_stream import _blend_pose  # numpy+scipy only, light import

DEFAULT_ROLLOUT_SAVE_DIR = str(Path("~/Dexmate/data/raw_data_rollout").expanduser())
DEFAULT_REPLAY_SAVE_DIR = str(Path("~/Dexmate/data/replay_raw_data").expanduser())
DEFAULT_ALIGN_SECONDS = 3.0
DEFAULT_ALIGN_TIMEOUT_S = 20.0
# Terminal save deadline. Healthy streaming finalization is normally sub-second; this
# generous bound exists only so a dead writer/filesystem cannot wedge robot teardown.
_RECORDER_SAVE_TIMEOUT_S = 120.0
# Recorded inter-frame gaps beyond this get a load-time warning: record_tick skips
# during holds, so a legit take can gap; replay then holds (stale-source) and glides
# across the gap -- safe, but the operator should know the take pauses.
REPLAY_GAP_WARN_S = 0.5
# Replay frame 0 must start near the robot's current (post-alignment) pose: a
# distant first target would be commanded as a snap (clamp-limited, but a large
# IK/base transient). Fail closed instead.
REPLAY_START_POS_TOL_M = 0.15
REPLAY_START_ROT_TOL_DEG = 30.0
_BASE_POSE_SOURCES = frozenset({"wheel_odometry", "arkit"})
_CAMERA_FUTURE_TOLERANCE_NS = 10_000_000
_CAMERA_ALIGNMENT_MODES = frozenset(
    {
        "head_capture_nearest",
        "latest_arrived_compatibility",
        "latest_arrived_legacy_unverified",
    }
)
_CAMERA_ALIGNMENT_FIELDS = (
    "camera_alignment_mode",
    "camera_alignment_clock_domain",
    "camera_alignment_max_abs_skew_ns",
    "camera_alignment_max_camera_age_ns",
    "camera_alignment_verified",
)
_ACTION_OFFSET_SEMANTICS = "all action components shifted together"
_ACTION_TIMING_CONFIG_FIELDS = (
    "dataset_fps",
    "action_offset_frames",
    "action_offset_semantics",
)

# --- Checkpoint-driven live SceneDiff object bootstrap -------------------------------
# Mobile analog of live_scenediff_rollout.py: after reference alignment we capture ONE
# head frame, stamp it with the FK world_T_zed extrinsic (engage-origin world), diff it
# against the fixed reference scene, and feed the arranged object positions to the policy
# as a constant observation.environment_state. SceneDiff runs in its own interpreter
# (scene_diff/.venv-merged) so its VRAM (SAM3 + DINOv3) is a subprocess, never shared
# with the loaded policy.
DEFAULT_SCENE_DIFF_REPO = "/home/yixuan/scene_diff"
DEFAULT_SCENE_DIFF_PYTHON = "/home/yixuan/scene_diff/.venv-merged/bin/python"
# The mobile "after"/moved reference = the LAST frame of the fixed reference episode, the
# SAME scene training diffed against (run_wbc_pos_condition.sh REFERENCE_LAST). Positions
# land in the live engage-origin WORLD frame because make_deploy_before_hdf5.py uses the
# live capture's OWN embedded extrinsic (world_T_zed), not the reference's calibration.
DEFAULT_SCENE_REFERENCE = "/home/yixuan/Dexmate/data/scene_diff/_before/reference_last.hdf5"
# The live-capture HDF5 writer, the SceneDiff subprocess invoker and the operator-order
# prompt are shared with the tabletop path in omniteleop.follower.scenediff_live (imported
# above): the single-frame HDF5 layout and the run_live_pos_condition.sh env interface are
# cross-repo CONTRACTS, kept in ONE place. Only the mobile-specific pieces live below --
# the FK world_T_zed (canonical base-pose composition) and the flat single-stage env-state.


def _checked_base_pose_source(value, context: str) -> str:
    """Return a valid canonical pose-source name or fail with provenance."""
    source = str(value)
    if source not in _BASE_POSE_SOURCES:
        raise ValueError(
            f"{context}: base_pose_source={source!r}; expected one of "
            f"{sorted(_BASE_POSE_SOURCES)}"
        )
    return source


def _dataset_meta_file(path: str | Path) -> Path:
    """Accept either ``dexmate_meta.json`` itself or its dataset directory."""
    candidate = Path(path).expanduser()
    return candidate / "dexmate_meta.json" if candidate.is_dir() else candidate


def _checked_policy_action_schema(value, context: str) -> str:
    schema = str(value)
    try:
        action_axes_for_policy_schema(schema)
    except ValueError as exc:
        raise ValueError(f"{context}: {exc}") from exc
    return schema


def _read_dataset_policy_action_schema(
    path: str | Path, *, required: bool
) -> tuple[str, Path] | None:
    """Read and validate the porter's action layout/frame declaration."""
    meta_path = _dataset_meta_file(path)
    if not meta_path.is_file():
        if required:
            raise FileNotFoundError(f"policy training dataset metadata not found: {meta_path}")
        return None
    try:
        payload = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read policy training metadata {meta_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{meta_path}: expected a JSON object")
    value = payload.get("policy_action_schema", payload.get("schema"))
    if value is None:
        if required:
            raise ValueError(f"{meta_path}: missing policy_action_schema")
        return None
    schema = _checked_policy_action_schema(value, str(meta_path))
    expected_axes = action_axes_for_policy_schema(schema)
    if "action_axes" in payload and list(payload["action_axes"]) != expected_axes:
        raise ValueError(
            f"{meta_path}: action_axes disagree with {schema!r}"
        )
    action_frame = str(payload.get("action_frame", ""))
    expected_prefix = (
        "current_base" if schema == JOYSTICK_POLICY_ACTION_SCHEMA else "world"
    )
    if action_frame and not action_frame.startswith(expected_prefix):
        raise ValueError(
            f"{meta_path}: action_frame={action_frame!r} disagrees with {schema!r}"
        )
    return schema, meta_path.resolve()


def _discover_checkpoint_policy_action_schema(
    policy_path: str | Path,
    policy_config,
    *,
    explicit_dataset_meta: str | Path | None = None,
) -> tuple[str | None, str]:
    """Resolve the versioned 29-D/32-D action meaning from checkpoint provenance."""
    declarations: list[tuple[str, str]] = []
    config_value = getattr(policy_config, "policy_action_schema", None)
    if config_value not in (None, ""):
        declarations.append(
            (_checked_policy_action_schema(config_value, "checkpoint config"), "config.json")
        )
    checkpoint = Path(policy_path).expanduser()
    candidates: list[tuple[Path, bool]] = []
    if explicit_dataset_meta is not None:
        candidates.append((_dataset_meta_file(explicit_dataset_meta), True))
    elif checkpoint.is_dir():
        candidates.append((checkpoint / "dexmate_meta.json", False))
        train_config_path = checkpoint / "train_config.json"
        if train_config_path.is_file():
            try:
                train_config = json.loads(train_config_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot read {train_config_path}: {exc}") from exc
            dataset = train_config.get("dataset") if isinstance(train_config, dict) else None
            dataset_root = dataset.get("root") if isinstance(dataset, dict) else None
            if dataset_root:
                candidates.append((_dataset_meta_file(str(dataset_root)), False))
    seen: set[Path] = set()
    for candidate, required in candidates:
        key = candidate.expanduser().absolute()
        if key in seen:
            continue
        seen.add(key)
        loaded = _read_dataset_policy_action_schema(candidate, required=required)
        if loaded is not None:
            schema, resolved = loaded
            declarations.append((schema, str(resolved)))
    schemas = {schema for schema, _origin in declarations}
    if len(schemas) > 1:
        detail = ", ".join(f"{origin} -> {schema}" for schema, origin in declarations)
        raise ValueError(f"checkpoint policy-action provenance disagrees: {detail}")
    if not declarations:
        return None, "undeclared"
    return declarations[0][0], "; ".join(origin for _schema, origin in declarations)


def _read_dataset_base_pose_source(path: str | Path, *, required: bool) -> tuple[str, Path] | None:
    """Read the canonical source stamped by ``port_wbc_mobile_hdf5.py``."""
    meta_path = _dataset_meta_file(path)
    if not meta_path.is_file():
        if required:
            raise FileNotFoundError(f"policy training dataset metadata not found: {meta_path}")
        return None
    try:
        payload = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read policy training metadata {meta_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{meta_path}: expected a JSON object")
    value = payload.get("base_pose_source")
    if value is None:
        if required:
            raise ValueError(
                f"{meta_path}: missing base_pose_source; pass the original v3 porter "
                "metadata or use --policy-base-pose-source for a legacy dataset"
            )
        return None
    return _checked_base_pose_source(value, str(meta_path)), meta_path.resolve()


def _discover_checkpoint_base_pose_source(
    policy_path: str | Path,
    policy_config,
    *,
    explicit_dataset_meta: str | Path | None = None,
) -> tuple[str | None, str]:
    """Discover the training pose source from config/ported-dataset provenance.

    LeRobot checkpoints save ``train_config.json`` beside ``config.json``. Its
    ``dataset.root`` points back to the porter's ``dexmate_meta.json``. A copied
    ``dexmate_meta.json`` in the checkpoint is also accepted so moved checkpoints
    remain self-contained. All available declarations must agree.
    """
    declarations: list[tuple[str, str]] = []
    config_value = getattr(policy_config, "base_pose_source", None)
    if config_value not in (None, ""):
        declarations.append(
            (_checked_base_pose_source(config_value, "checkpoint config"), "config.json")
        )

    checkpoint = Path(policy_path).expanduser()
    meta_candidates: list[tuple[Path, bool]] = []
    if explicit_dataset_meta is not None:
        meta_candidates.append((_dataset_meta_file(explicit_dataset_meta), True))
    elif checkpoint.is_dir():
        meta_candidates.append((checkpoint / "dexmate_meta.json", False))
        train_config_path = checkpoint / "train_config.json"
        if train_config_path.is_file():
            try:
                train_config = json.loads(train_config_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot read {train_config_path}: {exc}") from exc
            dataset = train_config.get("dataset") if isinstance(train_config, dict) else None
            dataset_root = dataset.get("root") if isinstance(dataset, dict) else None
            if dataset_root:
                meta_candidates.append(
                    (_dataset_meta_file(str(dataset_root)), False)
                )

    seen: set[Path] = set()
    for candidate, required in meta_candidates:
        key = candidate.expanduser().absolute()
        if key in seen:
            continue
        seen.add(key)
        loaded = _read_dataset_base_pose_source(candidate, required=required)
        if loaded is not None:
            source, resolved = loaded
            declarations.append((source, str(resolved)))

    sources = {source for source, _origin in declarations}
    if len(sources) > 1:
        detail = ", ".join(f"{origin} -> {source}" for source, origin in declarations)
        raise ValueError(f"checkpoint base-pose provenance disagrees: {detail}")
    if not declarations:
        return None, "undeclared"
    return declarations[0][0], "; ".join(origin for _source, origin in declarations)


def _checked_camera_alignment_contract(values: dict, context: str) -> dict:
    """Validate one ported training/live multi-camera timing declaration."""
    mode = str(values["camera_alignment_mode"])
    clock_domain = str(values["camera_alignment_clock_domain"])
    max_skew_ns = values["camera_alignment_max_abs_skew_ns"]
    max_camera_age_ns = values["camera_alignment_max_camera_age_ns"]
    verified = values["camera_alignment_verified"]
    if mode not in _CAMERA_ALIGNMENT_MODES:
        raise ValueError(
            f"{context}: camera_alignment_mode={mode!r}; expected one of "
            f"{sorted(_CAMERA_ALIGNMENT_MODES)}"
        )
    if not isinstance(verified, (bool, np.bool_)):
        raise ValueError(f"{context}: camera_alignment_verified must be boolean")
    verified = bool(verified)
    if max_skew_ns is not None:
        if isinstance(max_skew_ns, (bool, np.bool_)) or not isinstance(
            max_skew_ns, (int, np.integer)
        ):
            raise ValueError(
                f"{context}: camera_alignment_max_abs_skew_ns must be an integer or null"
            )
        max_skew_ns = int(max_skew_ns)
    if max_camera_age_ns is not None:
        if isinstance(max_camera_age_ns, (bool, np.bool_)) or not isinstance(
            max_camera_age_ns, (int, np.integer)
        ):
            raise ValueError(
                f"{context}: camera_alignment_max_camera_age_ns must be an integer or null"
            )
        max_camera_age_ns = int(max_camera_age_ns)

    if mode == "head_capture_nearest":
        if clock_domain != "local_via_camera_ntp":
            raise ValueError(
                f"{context}: head_capture_nearest requires local_via_camera_ntp"
            )
        if (
            max_skew_ns is None
            or max_skew_ns <= 0
            or max_camera_age_ns is None
            or max_camera_age_ns <= 0
            or not verified
        ):
            raise ValueError(
                f"{context}: head_capture_nearest requires positive skew/age bounds "
                "and camera_alignment_verified=true"
            )
    elif mode == "latest_arrived_compatibility":
        if (
            clock_domain != "local_via_camera_ntp"
            or max_skew_ns != 0
            or max_camera_age_ns is None
            or max_camera_age_ns <= 0
            or verified
        ):
            raise ValueError(
                f"{context}: latest_arrived_compatibility requires local clock, "
                "zero skew bound, positive age bound, and verified=false"
            )
    elif (
        clock_domain != "raw_publisher_clock_unverified"
        or max_skew_ns is not None
        or max_camera_age_ns is not None
        or verified
    ):
        raise ValueError(
            f"{context}: latest_arrived_legacy_unverified requires raw unverified "
            "clock, null skew bound, and verified=false"
        )
    return {
        "mode": mode,
        "clock_domain": clock_domain,
        "max_abs_skew_ns": max_skew_ns,
        "max_camera_age_ns": max_camera_age_ns,
        "verified": verified,
    }


def _read_dataset_camera_alignment_contract(
    path: str | Path, *, required: bool
) -> tuple[dict, Path] | None:
    meta_path = _dataset_meta_file(path)
    if not meta_path.is_file():
        if required:
            raise FileNotFoundError(f"policy training dataset metadata not found: {meta_path}")
        return None
    try:
        payload = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read policy training metadata {meta_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{meta_path}: expected a JSON object")
    present = {key for key in _CAMERA_ALIGNMENT_FIELDS if key in payload}
    if not present:
        if required:
            raise ValueError(
                f"{meta_path}: missing camera-alignment metadata; re-port the dataset "
                "or explicitly declare the legacy checkpoint contract"
            )
        return None
    if present != set(_CAMERA_ALIGNMENT_FIELDS):
        raise ValueError(
            f"{meta_path}: partial camera-alignment metadata; has {sorted(present)}, "
            f"missing {sorted(set(_CAMERA_ALIGNMENT_FIELDS) - present)}"
        )
    contract = _checked_camera_alignment_contract(payload, str(meta_path))
    return contract, meta_path.resolve()


def _discover_checkpoint_camera_alignment_contract(
    policy_path: str | Path,
    policy_config,
    *,
    explicit_dataset_meta: str | Path | None = None,
) -> tuple[dict | None, str]:
    """Discover and cross-check the training image-pairing contract."""
    declarations: list[tuple[dict, str]] = []
    sentinel = object()
    config_values = {
        key: getattr(policy_config, key, sentinel) for key in _CAMERA_ALIGNMENT_FIELDS
    }
    config_present = {key for key, value in config_values.items() if value is not sentinel}
    if config_present:
        if config_present != set(_CAMERA_ALIGNMENT_FIELDS):
            raise ValueError(
                "checkpoint config has partial camera-alignment metadata; has "
                f"{sorted(config_present)}, missing "
                f"{sorted(set(_CAMERA_ALIGNMENT_FIELDS) - config_present)}"
            )
        declarations.append(
            (
                _checked_camera_alignment_contract(config_values, "checkpoint config"),
                "config.json",
            )
        )

    checkpoint = Path(policy_path).expanduser()
    meta_candidates: list[tuple[Path, bool]] = []
    if explicit_dataset_meta is not None:
        meta_candidates.append((_dataset_meta_file(explicit_dataset_meta), True))
    elif checkpoint.is_dir():
        meta_candidates.append((checkpoint / "dexmate_meta.json", False))
        train_config_path = checkpoint / "train_config.json"
        if train_config_path.is_file():
            try:
                train_config = json.loads(train_config_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot read {train_config_path}: {exc}") from exc
            dataset = train_config.get("dataset") if isinstance(train_config, dict) else None
            dataset_root = dataset.get("root") if isinstance(dataset, dict) else None
            if dataset_root:
                meta_candidates.append((_dataset_meta_file(str(dataset_root)), False))

    seen: set[Path] = set()
    for candidate, required in meta_candidates:
        key = candidate.expanduser().absolute()
        if key in seen:
            continue
        seen.add(key)
        loaded = _read_dataset_camera_alignment_contract(candidate, required=required)
        if loaded is not None:
            contract, resolved = loaded
            declarations.append((contract, str(resolved)))

    signatures = {
        (
            contract["mode"],
            contract["clock_domain"],
            contract["max_abs_skew_ns"],
            contract["max_camera_age_ns"],
            contract["verified"],
        )
        for contract, _origin in declarations
    }
    if len(signatures) > 1:
        detail = ", ".join(
            f"{origin} -> {contract}" for contract, origin in declarations
        )
        raise ValueError(f"checkpoint camera-alignment provenance disagrees: {detail}")
    if not declarations:
        return None, "undeclared"
    return declarations[0][0], "; ".join(origin for _contract, origin in declarations)


def _checked_action_timing_contract(values: dict, context: str) -> dict:
    """Validate the cadence and coherent future-label offset used for training."""
    fps = values["dataset_fps"]
    offset = values["action_offset_frames"]
    semantics = str(values["action_offset_semantics"])
    if isinstance(fps, (bool, np.bool_)) or not isinstance(
        fps, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{context}: dataset_fps must be numeric")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{context}: dataset_fps must be finite and > 0, got {fps!r}")
    if isinstance(offset, (bool, np.bool_)) or not isinstance(offset, (int, np.integer)):
        raise ValueError(f"{context}: action_offset_frames must be an integer")
    offset = int(offset)
    if offset < 0:
        raise ValueError(
            f"{context}: action_offset_frames must be >= 0, got {offset}"
        )
    if semantics != _ACTION_OFFSET_SEMANTICS:
        raise ValueError(
            f"{context}: action_offset_semantics={semantics!r}; expected "
            f"{_ACTION_OFFSET_SEMANTICS!r}"
        )
    return {
        "dataset_fps": fps,
        "action_offset_frames": offset,
        "action_offset_semantics": semantics,
    }


def _read_dataset_action_timing_contract(
    path: str | Path, *, required: bool
) -> tuple[dict, Path] | None:
    """Read the porter's action-label timing declaration from one dataset sidecar."""
    meta_path = _dataset_meta_file(path)
    if not meta_path.is_file():
        if required:
            raise FileNotFoundError(f"policy training dataset metadata not found: {meta_path}")
        return None
    try:
        payload = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read policy training metadata {meta_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{meta_path}: expected a JSON object")

    # ``fps`` exists in older sidecars, so it alone does not imply that the newer
    # action-offset contract is present. Any offset field does, and then the complete
    # declaration is mandatory.
    offset_fields = {
        "action_offset_frames",
        "pose_action_offset_frames",
        "action_offset_semantics",
    }
    present = offset_fields.intersection(payload)
    if not present:
        if required:
            raise ValueError(
                f"{meta_path}: missing action-timing metadata; re-port the dataset or "
                "explicitly declare the legacy checkpoint offset"
            )
        return None
    required_fields = {"fps", "action_offset_frames", "action_offset_semantics"}
    missing = sorted(required_fields - set(payload))
    if missing:
        raise ValueError(
            f"{meta_path}: partial action-timing metadata; missing {missing}"
        )
    if (
        "pose_action_offset_frames" in payload
        and payload["pose_action_offset_frames"] != payload["action_offset_frames"]
    ):
        raise ValueError(
            f"{meta_path}: pose_action_offset_frames disagrees with "
            "action_offset_frames"
        )
    contract = _checked_action_timing_contract(
        {
            "dataset_fps": payload["fps"],
            "action_offset_frames": payload["action_offset_frames"],
            "action_offset_semantics": payload["action_offset_semantics"],
        },
        str(meta_path),
    )
    return contract, meta_path.resolve()


def _discover_checkpoint_action_timing_contract(
    policy_path: str | Path,
    policy_config,
    *,
    explicit_dataset_meta: str | Path | None = None,
) -> tuple[dict | None, str]:
    """Discover and cross-check cadence/label-offset provenance for a checkpoint."""
    declarations: list[tuple[dict, str]] = []
    sentinel = object()
    config_values = {
        key: getattr(policy_config, key, sentinel)
        for key in _ACTION_TIMING_CONFIG_FIELDS
    }
    config_present = {
        key for key, value in config_values.items() if value is not sentinel
    }
    if config_present:
        if config_present != set(_ACTION_TIMING_CONFIG_FIELDS):
            raise ValueError(
                "checkpoint config has partial action-timing metadata; has "
                f"{sorted(config_present)}, missing "
                f"{sorted(set(_ACTION_TIMING_CONFIG_FIELDS) - config_present)}"
            )
        declarations.append(
            (
                _checked_action_timing_contract(config_values, "checkpoint config"),
                "config.json",
            )
        )

    checkpoint = Path(policy_path).expanduser()
    meta_candidates: list[tuple[Path, bool]] = []
    if explicit_dataset_meta is not None:
        meta_candidates.append((_dataset_meta_file(explicit_dataset_meta), True))
    elif checkpoint.is_dir():
        meta_candidates.append((checkpoint / "dexmate_meta.json", False))
        train_config_path = checkpoint / "train_config.json"
        if train_config_path.is_file():
            try:
                train_config = json.loads(train_config_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot read {train_config_path}: {exc}") from exc
            dataset = train_config.get("dataset") if isinstance(train_config, dict) else None
            dataset_root = dataset.get("root") if isinstance(dataset, dict) else None
            if dataset_root:
                meta_candidates.append((_dataset_meta_file(str(dataset_root)), False))

    seen: set[Path] = set()
    for candidate, required in meta_candidates:
        key = candidate.expanduser().absolute()
        if key in seen:
            continue
        seen.add(key)
        loaded = _read_dataset_action_timing_contract(candidate, required=required)
        if loaded is not None:
            contract, resolved = loaded
            declarations.append((contract, str(resolved)))

    signatures = {
        (
            contract["dataset_fps"],
            contract["action_offset_frames"],
            contract["action_offset_semantics"],
        )
        for contract, _origin in declarations
    }
    if len(signatures) > 1:
        detail = ", ".join(
            f"{origin} -> {contract}" for contract, origin in declarations
        )
        raise ValueError(f"checkpoint action-timing provenance disagrees: {detail}")
    if not declarations:
        return None, "undeclared"
    return declarations[0][0], "; ".join(origin for _contract, origin in declarations)


def _runtime_base_pose_source(arkit_mode: str) -> str:
    """Canonical source selected by the shared hardware driver's control mode."""
    return "arkit" if str(arkit_mode) == "control" else "wheel_odometry"


def _resolve_rollout_base_pose_source(
    *,
    arkit_mode: str,
    policy=None,
    replay_source=None,
    cli_source: str | None = None,
) -> tuple[str, str]:
    """Bind training/replay semantics to the live control source, fail closed."""
    runtime_source = _runtime_base_pose_source(arkit_mode)
    if policy is not None:
        declared = getattr(policy, "base_pose_source", None)
        if declared is not None:
            declared = _checked_base_pose_source(declared, "checkpoint metadata")
        if cli_source is not None:
            cli_source = _checked_base_pose_source(cli_source, "command line")
        if declared is not None and cli_source is not None and declared != cli_source:
            raise ValueError(
                f"--policy-base-pose-source={cli_source} contradicts checkpoint "
                f"metadata {declared}"
            )
        expected = declared or cli_source
        if expected is None:
            raise ValueError(
                "checkpoint does not declare base_pose_source. Pass "
                "--policy-dataset-meta <ported_dataset>/dexmate_meta.json, or for a "
                "legacy checkpoint explicitly pass --policy-base-pose-source "
                "wheel_odometry"
            )
        provenance = str(
            getattr(policy, "base_pose_source_provenance", "--policy-base-pose-source")
            if declared is not None
            else "--policy-base-pose-source"
        )
    elif replay_source is not None:
        expected = _checked_base_pose_source(
            getattr(replay_source, "base_pose_source", None), "replay episode"
        )
        provenance = str(getattr(replay_source, "path", "replay episode"))
    else:
        raise ValueError("base-pose source resolution needs a policy or replay episode")

    if expected != runtime_source:
        required_mode = "control" if expected == "arkit" else "off or record"
        raise ValueError(
            f"training/replay base_pose_source={expected} ({provenance}), but "
            f"--arkit-base {arkit_mode} makes live policy/control state use "
            f"{runtime_source}; use --arkit-base {required_mode}"
        )
    if policy is not None:
        policy.base_pose_source = expected
        policy.base_pose_source_provenance = provenance
    return expected, runtime_source


def _runtime_camera_alignment_contract(
    max_skew_ms: float, max_camera_age_ms: float
) -> dict:
    if not np.isfinite(max_skew_ms) or max_skew_ms < 0:
        raise ValueError("camera skew bound must be finite and >= 0")
    if not np.isfinite(max_camera_age_ms) or max_camera_age_ms <= 0:
        raise ValueError("camera age bound must be finite and > 0")
    max_skew_ns = int(round(float(max_skew_ms) * 1e6))
    return {
        "mode": (
            "head_capture_nearest"
            if max_skew_ns > 0
            else "latest_arrived_compatibility"
        ),
        "clock_domain": "local_via_camera_ntp",
        "max_abs_skew_ns": max_skew_ns,
        "max_camera_age_ns": int(round(float(max_camera_age_ms) * 1e6)),
        "verified": max_skew_ns > 0,
    }


def _resolve_rollout_camera_alignment(
    *,
    policy,
    runtime_max_skew_ms: float,
    runtime_max_camera_age_ms: float,
    cli_mode: str | None = None,
    cli_max_skew_ms: float | None = None,
    cli_max_camera_age_ms: float | None = None,
) -> tuple[dict | None, dict]:
    """Bind a multi-view checkpoint to the exact live frame-selection contract."""
    runtime = _runtime_camera_alignment_contract(
        runtime_max_skew_ms, runtime_max_camera_age_ms
    )
    wrist_arms = tuple(getattr(policy, "wrist_arms", ()))
    if not wrist_arms:
        return None, runtime

    declared = getattr(policy, "camera_alignment_contract", None)
    if declared is not None:
        declared = _checked_camera_alignment_contract(
            {
                "camera_alignment_mode": declared["mode"],
                "camera_alignment_clock_domain": declared["clock_domain"],
                "camera_alignment_max_abs_skew_ns": declared["max_abs_skew_ns"],
                "camera_alignment_max_camera_age_ns": declared["max_camera_age_ns"],
                "camera_alignment_verified": declared["verified"],
            },
            "checkpoint metadata",
        )

    cli_contract = None
    if cli_mode is not None:
        if cli_max_skew_ms is None:
            cli_max_skew_ms = 0.0 if cli_mode == "latest_arrived_compatibility" else None
        if cli_max_skew_ms is None:
            raise ValueError(
                "--policy-camera-max-skew-ms is required with "
                "--policy-camera-alignment-mode head_capture_nearest"
            )
        if cli_max_camera_age_ms is None:
            raise ValueError(
                "--policy-camera-max-age-ms is required with a legacy checkpoint "
                "camera-alignment declaration"
            )
        cli_contract = _runtime_camera_alignment_contract(
            cli_max_skew_ms, cli_max_camera_age_ms
        )
        if cli_contract["mode"] != cli_mode:
            raise ValueError(
                f"--policy-camera-alignment-mode {cli_mode} contradicts "
                f"--policy-camera-max-skew-ms {cli_max_skew_ms:g}"
            )

    if declared is not None and cli_contract is not None and declared != cli_contract:
        raise ValueError(
            f"command-line camera alignment {cli_contract} contradicts checkpoint "
            f"metadata {declared}"
        )
    expected = declared or cli_contract
    if expected is None:
        raise ValueError(
            "multi-camera checkpoint does not declare its training camera-alignment "
            "contract. Pass --policy-dataset-meta <ported_dataset>/dexmate_meta.json, "
            "or explicitly pass --policy-camera-alignment-mode and "
            "--policy-camera-max-skew-ms/--policy-camera-max-age-ms for a legacy "
            "checkpoint"
        )
    if not expected["verified"]:
        raise ValueError(
            f"multi-camera checkpoint was trained with unverified alignment mode "
            f"{expected['mode']!r}; re-record/retrain from verified "
            "head-capture-nearest data"
        )
    if expected != runtime:
        provenance = getattr(
            policy, "camera_alignment_provenance", "command-line declaration"
        )
        raise ValueError(
            f"training camera alignment {expected} ({provenance}) does not exactly "
            f"match live recorder/inference alignment {runtime}; set "
            "--record-max-camera-skew-ms to the training bound"
        )
    policy.camera_alignment_contract = expected
    if declared is None:
        policy.camera_alignment_provenance = "command-line declaration"
    return expected, runtime


def _resolve_rollout_action_timing(
    *,
    policy,
    runtime_dataset_fps: float,
    cli_action_offset_frames: int | None = None,
) -> dict:
    """Bind training label timing to the live chunk scheduler, fail closed."""
    if not np.isfinite(runtime_dataset_fps) or runtime_dataset_fps <= 0.0:
        raise ValueError("--dataset-fps must be finite and > 0")
    runtime_dataset_fps = float(runtime_dataset_fps)

    declared_fps = getattr(policy, "dataset_fps", None)
    declared_offset = getattr(policy, "action_offset_frames", None)
    if (declared_fps is None) != (declared_offset is None):
        raise ValueError(
            "checkpoint has a partial action-timing declaration; dataset_fps and "
            "action_offset_frames must be declared together"
        )
    declared = None
    if declared_fps is not None:
        declared = _checked_action_timing_contract(
            {
                "dataset_fps": declared_fps,
                "action_offset_frames": declared_offset,
                "action_offset_semantics": getattr(
                    policy, "action_offset_semantics", _ACTION_OFFSET_SEMANTICS
                ),
            },
            "checkpoint metadata",
        )

    cli_offset = None
    if cli_action_offset_frames is not None:
        cli_offset = _checked_action_timing_contract(
            {
                "dataset_fps": runtime_dataset_fps,
                "action_offset_frames": cli_action_offset_frames,
                "action_offset_semantics": _ACTION_OFFSET_SEMANTICS,
            },
            "command line",
        )

    if declared is not None and not np.isclose(
        declared["dataset_fps"], runtime_dataset_fps, rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            f"--dataset-fps {runtime_dataset_fps:g} != training metadata "
            f"{declared['dataset_fps']:g}"
        )
    if (
        declared is not None
        and cli_offset is not None
        and declared["action_offset_frames"] != cli_offset["action_offset_frames"]
    ):
        raise ValueError(
            f"--policy-action-offset-frames={cli_offset['action_offset_frames']} "
            f"contradicts checkpoint metadata {declared['action_offset_frames']}"
        )

    resolved = declared or cli_offset
    if resolved is None:
        raise ValueError(
            "checkpoint does not declare action_offset_frames. Pass "
            "--policy-dataset-meta <ported_dataset>/dexmate_meta.json, or for a "
            "legacy zero-offset checkpoint explicitly pass "
            "--policy-action-offset-frames 0"
        )
    # The command-line cadence is authoritative at runtime after the exact-match check.
    resolved = dict(resolved)
    resolved["dataset_fps"] = runtime_dataset_fps
    policy.dataset_fps = runtime_dataset_fps
    policy.action_offset_frames = resolved["action_offset_frames"]
    policy.action_offset_semantics = resolved["action_offset_semantics"]
    if declared is None:
        policy.action_timing_provenance = "command-line declaration"
    return resolved


def split_policy_action(
    action: np.ndarray,
) -> dict[str, np.ndarray | np.float32 | None]:
    """29-D WBC or 32-D joystick action -> pose, gripper, and chassis targets.

    Pure reshaping: positions pass through, 6-D rotations are re-orthonormalized
    by ``pos6d_to_mat``. The outputs are already in the action schema's declared
    frame (engage-world for 29-D WBC, current-base for 32-D joystick) -- never
    compose them with a base pose at the 10 Hz policy tick or 100 Hz IK tick.
    Postprocessed gripper predictions are thresholded at the same value used to
    binarize the training actions.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape not in ((len(ACTION_AXES),), (len(JOYSTICK_ACTION_AXES),)):
        raise ValueError(
            f"expected 29-D or 32-D WBC/joystick action, got {action.shape}"
        )
    if not np.all(np.isfinite(action)):
        raise ValueError("policy action contains non-finite values")
    return {
        "left": pos6d_to_mat(action[0:9]),
        "right": pos6d_to_mat(action[10:19]),
        "head": pos6d_to_mat(action[20:29]),
        "left_gripper": np.float32(action[9] >= GRIPPER_BINARY_THRESHOLD),
        "right_gripper": np.float32(action[19] >= GRIPPER_BINARY_THRESHOLD),
        "chassis": (
            None
            if action.shape == (len(ACTION_AXES),)
            else action[29:32].copy()
        ),
    }


def encode_replay_action(
    left: np.ndarray,
    right: np.ndarray,
    head: np.ndarray,
    grip_left,
    grip_right,
    chassis: np.ndarray | None = None,
) -> np.ndarray:
    """Replayed targets -> the matching 29-D WBC or 32-D joystick vector.

    Only for the ``policy_io`` log, so replay and live rollouts are directly
    diffable; the robot itself is driven from the 4x4 matrices, never from this
    encoding.
    """
    grips = np.asarray([grip_left, grip_right], dtype=np.float32)
    if not np.all(np.isfinite(grips)):
        raise ValueError(f"replay gripper commands must be finite, got {grips}")
    parts = [
        mat_to_pos6d(left), grips[:1], mat_to_pos6d(right), grips[1:2],
        mat_to_pos6d(head),
    ]
    if chassis is not None:
        chassis_arr = np.asarray(chassis, dtype=np.float32).reshape(-1)
        if chassis_arr.shape != (3,) or not np.all(np.isfinite(chassis_arr)):
            raise ValueError(f"replay chassis intent must be finite (3,), got {chassis_arr}")
        parts.append(chassis_arr)
    return np.concatenate(parts).astype(np.float32)


@dataclass(frozen=True)
class ScheduledPolicyAction:
    """One chunk frame scheduled for execution at a wall-clock time.

    ``timestamp`` is an interpolation KNOT on the workstation ``perf_counter``
    timeline (clock domain (a), plan.md Conventions) -- the moment the 10 Hz
    sampler should command these targets -- not a hard release barrier: the
    sampler blends between bracketing knots.
    """

    timestamp: float
    left: np.ndarray            # (4, 4) target in the policy action frame
    right: np.ndarray           # (4, 4)
    head: np.ndarray            # (4, 4)
    grip_left: np.float32
    grip_right: np.float32
    # Joystick policy: body-frame post-projection intent, held ZOH between knots.
    # None preserves the deployed 29-D WBC action contract.
    chassis: np.ndarray | None = None


def chunk_schedule_timestamps(
    *,
    observation_time: float,
    frame_count: int,
    dataset_dt: float,
    action_offset_frames: int,
) -> np.ndarray:
    """Wall-clock schedule for labels sourced from ``obs_index + offset``.

    A porter offset changes the physical time represented by every action label; it
    is not merely dataset provenance. Row ``k`` therefore belongs at
    ``observation_time + (action_offset_frames + k) * dataset_dt``. This keeps
    inference latency visible to the stale-frame drop instead of executing a future
    target early.
    """
    if not np.isfinite(observation_time):
        raise ValueError("observation_time must be finite")
    if isinstance(frame_count, bool) or not isinstance(frame_count, (int, np.integer)):
        raise ValueError("frame_count must be an integer")
    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    if not np.isfinite(dataset_dt) or dataset_dt <= 0.0:
        raise ValueError("dataset_dt must be finite and > 0")
    if isinstance(action_offset_frames, (bool, np.bool_)) or not isinstance(
        action_offset_frames, (int, np.integer)
    ):
        raise ValueError("action_offset_frames must be an integer")
    if action_offset_frames < 0:
        raise ValueError("action_offset_frames must be >= 0")
    return float(observation_time) + float(dataset_dt) * (
        int(action_offset_frames) + np.arange(int(frame_count), dtype=np.float64)
    )


def scheduled_actions_from_chunk(
    chunk: np.ndarray, timestamps: np.ndarray
) -> list[ScheduledPolicyAction]:
    """Decode an ``(n, 29|32)`` action chunk + per-frame wall-clock timestamps.

    Row ``k`` goes through ``split_policy_action`` (the same decode as a single
    policy action: Gram-Schmidt re-orthonormalization, NO base composition) and
    is paired with ``timestamps[k]``. Timestamps must be finite and strictly
    increasing -- they are the synthesized ``t_obs + k * dataset_dt`` schedule.
    """
    chunk = np.asarray(chunk, dtype=np.float32)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    valid_dim = chunk.ndim == 2 and chunk.shape[1] in (
        len(ACTION_AXES), len(JOYSTICK_ACTION_AXES)
    )
    if not valid_dim or chunk.shape[0] < 1 or chunk.shape[0] != ts.shape[0]:
        raise ValueError(
            f"chunk {chunk.shape} / timestamps {ts.shape}: expected (n>=1, "
            "29|32) with one timestamp per frame"
        )
    if not np.all(np.isfinite(ts)) or (ts.size > 1 and not np.all(np.diff(ts) > 0)):
        raise ValueError("chunk timestamps must be finite and strictly increasing")
    out: list[ScheduledPolicyAction] = []
    for k in range(chunk.shape[0]):
        targets = split_policy_action(chunk[k])
        out.append(ScheduledPolicyAction(
            timestamp=float(ts[k]),
            left=targets["left"], right=targets["right"], head=targets["head"],
            grip_left=targets["left_gripper"], grip_right=targets["right_gripper"],
            chassis=targets["chassis"],
        ))
    return out


def drop_stale_actions(
    actions: Sequence[ScheduledPolicyAction], cutoff: float
) -> list[ScheduledPolicyAction]:
    """Drop chunk frames not strictly after ``cutoff`` (inference end + latency).

    Port of rby1_wbc_policy.py's PD1.2 stale-action handling: a frame whose
    desired time precedes the soonest achievable execution time would ask the
    robot to time-travel, so it is DISCARDED -- the chunk is never time-shifted
    (that would hide latency as a systematic lag). Sorted input means only
    leading frames drop; an all-stale chunk returns ``[]`` (caller logs it).
    """
    if not np.isfinite(cutoff):
        raise ValueError(f"stale cutoff must be finite, got {cutoff}")
    return [a for a in actions if a.timestamp > cutoff]


def _interpolate_scheduled(
    prev: ScheduledPolicyAction, future: ScheduledPolicyAction, query_time: float
) -> ScheduledPolicyAction:
    """Blend poses while holding discrete grippers and chassis intent ZOH."""
    if (prev.chassis is None) != (future.chassis is None):
        raise ValueError("cannot interpolate across WBC/joystick action schemas")
    if future.timestamp <= prev.timestamp:
        return future
    alpha = float(np.clip(
        (query_time - prev.timestamp) / (future.timestamp - prev.timestamp), 0.0, 1.0
    ))
    return ScheduledPolicyAction(
        timestamp=float(query_time),
        left=_blend_pose(prev.left, future.left, alpha),
        right=_blend_pose(prev.right, future.right, alpha),
        head=_blend_pose(prev.head, future.head, alpha),
        grip_left=prev.grip_left,
        grip_right=prev.grip_right,
        chassis=(None if prev.chassis is None else prev.chassis.copy()),
    )


class ActionScheduleBuffer:
    """Thread-safe scheduled-action buffer + sampler (port of rby1_policy.py).

    ``queue()`` (inference thread) lets the newest chunk overwrite overlapping
    FUTURE entries and trims entries already in the past (``queue_actions``,
    rby1_policy.py 717-731). ``sample()`` (main thread, 10 Hz) evaluates the
    scheduled trajectory at a query time: previous/future bracketing knots,
    positions lerped and rotations slerped, anchored on the LAST EXECUTED sample
    for continuity across replans (``_last_executed_action``). Binary grippers
    hold the latest due knot instead of becoming interpolated widths; before the
    first knot the first knot passes through verbatim.

    Deliberate deviation from rby1 (plan.md hold semantics): rby1 returns the
    final scheduled action verbatim FOREVER once the trajectory is exhausted,
    which keeps ``last_cmd_wall`` fresh and defeats the stale-source watchdog.
    Here the terminal knot is returned exactly once (so the final waypoint is
    actually commanded); the next sample is a terminal underflow that aborts the
    episode immediately and rejects late chunks. Behaviorally identical otherwise: the
    ``TargetInterpolator`` holds its segment end with or without a verbatim
    re-push.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buffer: deque[ScheduledPolicyAction] = deque()
        self._last_executed: ScheduledPolicyAction | None = None
        self._armed = False
        self._terminal_underflow = False

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def queue(self, actions: Sequence[ScheduledPolicyAction], now: float) -> None:
        """Queue a chunk: the newest plan wins for the overlapping future."""
        actions = list(actions)
        if not actions:
            raise ValueError("queue() requires at least one scheduled action")
        ts = [a.timestamp for a in actions]
        if not all(np.isfinite(t) for t in ts) or any(
            b <= a for a, b in itertools.pairwise(ts)
        ):
            raise ValueError(
                "scheduled actions must carry finite, strictly increasing timestamps"
            )
        has_chassis = [action.chassis is not None for action in actions]
        if any(has_chassis) and not all(has_chassis):
            raise ValueError("one scheduled chunk cannot mix WBC and joystick actions")
        with self._lock:
            if self._terminal_underflow:
                raise RuntimeError(
                    "action buffer already underflowed; late chunks cannot resume motion"
                )
            while self._buffer and self._buffer[-1].timestamp >= actions[0].timestamp:
                self._buffer.pop()
            while self._buffer and self._buffer[0].timestamp < now:
                self._buffer.popleft()
            self._buffer.extend(actions)
            self._armed = True

    def sample(self, query_time: float) -> ScheduledPolicyAction | None:
        """Scheduled-trajectory value at ``query_time``; ``None`` when idle.

        ``None`` before the first chunk arrives. Once armed, exhaustion after the
        terminal knot raises and permanently poisons the buffer.
        The returned action carries ``timestamp=query_time`` and is remembered
        as the last-executed anchor for the next call.
        """
        with self._lock:
            passed: ScheduledPolicyAction | None = None
            while self._buffer and self._buffer[0].timestamp <= query_time:
                passed = self._buffer.popleft()
            future = self._buffer[0] if self._buffer else None
            prev = self._last_executed
            if prev is None or prev.timestamp > query_time:
                prev = passed
            if future is None:
                out = passed  # terminal knot: commanded once, then None -> watchdog
            elif prev is None:
                out = future  # before the first knot: command it verbatim (early glide)
            else:
                out = _interpolate_scheduled(prev, future, query_time)
            discrete_source = passed if passed is not None else (
                prev if prev is not None else future
            )
            if out is None or discrete_source is None:
                if self._armed:
                    self._terminal_underflow = True
                    raise RuntimeError(
                        "terminal action-buffer underflow after the first chunk was armed"
                    )
                return None
            executed = replace(
                out,
                timestamp=float(query_time),
                grip_left=discrete_source.grip_left,
                grip_right=discrete_source.grip_right,
                chassis=(
                    None
                    if discrete_source.chassis is None
                    else discrete_source.chassis.copy()
                ),
            )
            self._last_executed = executed
            return executed


@dataclass(frozen=True)
class ReplayFrame:
    """One released replay command: verbatim recorded targets + raw segment gap."""

    index: int
    left: np.ndarray
    right: np.ndarray
    head: np.ndarray
    grip_left: np.floating
    grip_right: np.floating
    segment_duration: float
    chassis: np.ndarray | None = None


class RecordedEpisodeSource:
    """Real-time (1x) scheduler over a ``wbc_vr_robot.py --record`` episode.

    Replays the REAL per-episode recorder schema (``action/eef/{left,right}``,
    ``action/head``, ``action/gripper/{left,right}``, ``timestamp_ns``) --
    deliberately NOT ``omniteleop.wbc_record.ReplaySource``, which replays the
    pre-WBC leader-joint stream at scalable speed. Frame ``t`` is released once
    ``timestamp_ns[t] - timestamp_ns[0]`` REAL seconds have elapsed since
    :meth:`start`; there is no speed scaling anywhere in this class.
    ``segment_duration`` is the raw inter-frame timestamp gap, meant for
    ``TargetInterpolator.push(duration=...)`` so each glide spans the actual
    recorded inter-command interval (frame 0 has no previous frame: 0.0, an
    immediate snap -- the per-tick joint clamp bounds any residual step).

    Known, deliberate property (plan.md Task 1): pushing frame ``t`` at its
    release time and gliding over its trailing gap means ``ik.solve()`` reaches
    frame ``t`` one segment (~one record period) after its recorded timestamp.
    The trajectory SHAPE and total pacing are preserved; the whole replay is
    uniformly late by ~1/record-rate, the same causal lag the interpolator adds
    to live commands.
    """

    def __init__(self, path: str | Path) -> None:
        import h5py  # noqa: PLC0415 -- only the replay path needs it

        self.path = str(path)
        with h5py.File(self.path, "r") as f:
            left = self._required(f, "action/eef/left")
            right = self._required(f, "action/eef/right")
            head = self._required(f, "action/head")
            grip_left = self._required(f, "action/gripper/left")
            grip_right = self._required(f, "action/gripper/right")
            ts = self._required(f, "timestamp_ns")
            schema = self._optional_text(f, "meta/schema")
            policy_schema = self._optional_text(f, "meta/policy_action_schema")
            control_mode = self._optional_text(f, "meta/control_mode")
            action_target_frame = self._optional_text(f, "meta/action_target_frame")
            eef_target_frame = self._optional_text(f, "meta/eef_target_frame")
            head_target_frame = self._optional_text(f, "meta/head_target_frame")
            declared_source = self._optional_text(f, "meta/obs_base_pose_source")
            control_source = self._optional_text(f, "meta/base_control_pose_source")
            chassis_intent = (
                self._required(f, "action/chassis/intent_body")
                if (
                    schema == "omniteleop_joystick_mobile_raw/v1"
                    or policy_schema == JOYSTICK_POLICY_ACTION_SCHEMA
                    or control_mode == "joystick"
                )
                else None
            )

        joystick = chassis_intent is not None
        if joystick:
            declarations = {
                "meta/schema": (schema, "omniteleop_joystick_mobile_raw/v1"),
                "meta/policy_action_schema": (
                    policy_schema, JOYSTICK_POLICY_ACTION_SCHEMA
                ),
                "meta/control_mode": (control_mode, "joystick"),
                "meta/action_target_frame": (action_target_frame, "current_base"),
                "meta/eef_target_frame": (eef_target_frame, "current_base"),
                "meta/head_target_frame": (head_target_frame, "current_base"),
            }
            bad = [
                f"{key}={actual!r} (expected {expected!r})"
                for key, (actual, expected) in declarations.items()
                if actual != expected
            ]
            if bad:
                raise RuntimeError(
                    f"{self.path}: conflicting joystick replay contract: "
                    + "; ".join(bad)
                )
            self.policy_action_schema = JOYSTICK_POLICY_ACTION_SCHEMA
            self.action_target_frame = "current_base"
        else:
            if policy_schema not in (None, WBC_POLICY_ACTION_SCHEMA):
                raise RuntimeError(
                    f"{self.path}: unsupported policy action schema {policy_schema!r}"
                )
            if action_target_frame not in (None, "engage_origin_world", "world"):
                raise RuntimeError(
                    f"{self.path}: WBC replay action frame must be world, got "
                    f"{action_target_frame!r}"
                )
            self.policy_action_schema = WBC_POLICY_ACTION_SCHEMA
            self.action_target_frame = "world"

        if schema in {
            "omniteleop_wbc_mobile_raw/v3",
            "omniteleop_wbc_mobile_raw/v4",
            "omniteleop_wbc_mobile_raw/v5",
            "omniteleop_joystick_mobile_raw/v1",
        } and declared_source is None:
            raise RuntimeError(
                f"{self.path}: explicit-base raw replay is missing "
                "meta/obs_base_pose_source"
            )
        self.base_pose_source = _checked_base_pose_source(
            declared_source or "wheel_odometry", self.path
        )
        if control_source is not None:
            control_source = _checked_base_pose_source(control_source, self.path)
            if control_source != self.base_pose_source:
                raise RuntimeError(
                    f"{self.path}: replay canonical base source {self.base_pose_source} "
                    f"!= recorded control source {control_source}"
                )

        if ts.ndim != 1 or ts.shape[0] < 1:
            raise RuntimeError(f"{self.path}: timestamp_ns must be (T>=1,), got {ts.shape}")
        if not np.issubdtype(ts.dtype, np.integer):
            raise RuntimeError(f"{self.path}: timestamp_ns must be integer ns, got {ts.dtype}")
        n = int(ts.shape[0])
        if n > 1 and not np.all(np.diff(ts) > 0):
            raise RuntimeError(f"{self.path}: timestamp_ns must be strictly increasing")
        for name, mats in (
            ("action/eef/left", left),
            ("action/eef/right", right),
            ("action/head", head),
        ):
            if mats.shape != (n, 4, 4):
                raise RuntimeError(f"{self.path}: {name} must be ({n},4,4), got {mats.shape}")
            if not np.all(np.isfinite(mats)):
                raise RuntimeError(f"{self.path}: {name} contains non-finite values")
            for col in (0, 1):
                norms = np.linalg.norm(mats[:, :3, col], axis=1)
                bad = np.where(np.abs(norms - 1.0) > 1e-2)[0]
                if bad.size:
                    raise RuntimeError(
                        f"{self.path}: {name} frame {int(bad[0])} rotation column {col} "
                        f"has norm {norms[bad[0]]:.4f} (not unit); corrupt recording?"
                    )
        for name, grips in (
            ("action/gripper/left", grip_left),
            ("action/gripper/right", grip_right),
        ):
            if grips.shape != (n,):
                raise RuntimeError(f"{self.path}: {name} must be ({n},), got {grips.shape}")
            if not np.all(np.isfinite(grips)):
                raise RuntimeError(f"{self.path}: {name} contains non-finite values")
        if chassis_intent is not None:
            if chassis_intent.shape != (n, 3):
                raise RuntimeError(
                    f"{self.path}: action/chassis/intent_body must be ({n},3), "
                    f"got {chassis_intent.shape}"
                )
            if not np.all(np.isfinite(chassis_intent)):
                raise RuntimeError(
                    f"{self.path}: action/chassis/intent_body contains non-finite values"
                )

        self._left = np.asarray(left, dtype=np.float64)
        self._right = np.asarray(right, dtype=np.float64)
        self._head = np.asarray(head, dtype=np.float64)
        # Raw dtype on purpose: replay must send bit-identical gripper commands.
        self._grip_left = grip_left
        self._grip_right = grip_right
        self._chassis_intent = (
            None
            if chassis_intent is None
            else np.asarray(chassis_intent, dtype=np.float32)
        )
        self._rel_s = (ts - ts[0]).astype(np.float64) / 1e9
        self._n = n
        self._wall0: float | None = None
        self._next = 0
        if n > 1:
            gaps = np.diff(self._rel_s)
            big = gaps > REPLAY_GAP_WARN_S
            if np.any(big):
                print(f"[wbc_policy_rollout] WARNING: {self.path} has {int(big.sum())} "
                      f"recorded gap(s) up to {float(gaps.max()):.2f}s (mid-take holds?); "
                      "replay will hold, then glide across each gap at its recorded pace")

    def _required(self, f, key: str) -> np.ndarray:
        if key not in f:
            raise RuntimeError(
                f"{self.path}: missing required HDF5 dataset {key} -- replay needs an "
                "episode recorded by the updated wbc_vr_robot.py --record"
            )
        return np.asarray(f[key])

    def _optional_text(self, f, key: str) -> str | None:
        if key not in f:
            return None
        value = np.asarray(f[key][()])
        if value.shape != ():
            raise RuntimeError(f"{self.path}: {key} must be scalar text")
        scalar = value.item()
        if isinstance(scalar, (bytes, np.bytes_)):
            try:
                return bytes(scalar).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RuntimeError(f"{self.path}: {key} is not valid UTF-8") from exc
        if isinstance(scalar, str):
            return scalar
        raise RuntimeError(f"{self.path}: {key} must be scalar text")

    @property
    def n_frames(self) -> int:
        """Number of recorded commands in the replay."""
        return self._n

    @property
    def duration_s(self) -> float:
        """Recorded wall-clock span in seconds."""
        return float(self._rel_s[-1])

    @property
    def released(self) -> int:
        """Number of replay commands released so far."""
        return self._next

    @property
    def done(self) -> bool:
        """Whether every recorded command has been released."""
        return self._next >= self._n

    def start(self, now: float) -> None:
        """Anchor the wall clock; frame 0 is due immediately."""
        self._wall0 = float(now)
        self._next = 0

    def advance(self, now: float) -> ReplayFrame | None:
        """Release the next frame once its recorded time offset has elapsed.

        At most one frame per call (a stalled loop catches up over the next few
        100 Hz ticks, preserving every recorded waypoint); ``None`` while no
        frame is due.
        """
        if self._wall0 is None:
            raise RuntimeError("RecordedEpisodeSource.advance() called before start()")
        if self.done:
            return None
        i = self._next
        if now - self._wall0 < self._rel_s[i]:
            return None
        self._next = i + 1
        return ReplayFrame(
            index=i,
            left=self._left[i],
            right=self._right[i],
            head=self._head[i],
            grip_left=self._grip_left[i],
            grip_right=self._grip_right[i],
            segment_duration=0.0 if i == 0 else float(self._rel_s[i] - self._rel_s[i - 1]),
            chassis=(
                None
                if self._chassis_intent is None
                else self._chassis_intent[i].copy()
            ),
        )


def wbc_tick(
    *,
    ik,
    driver,
    enable: dict,
    interp,
    head_lpf,
    head_deadband,
    now: float,
    dt: float,
    grip_left,
    grip_right,
    last_cmd_wall: float | None,
    live_head_filters: bool,
    estop: bool = False,
    source_timestamp_ns: int = -1,
    source_receive_wall_ns: int = -1,
):
    """One 100 Hz WBC tick: interpolate -> head shaping -> solve -> actuate -> record.

    ``live_head_filters=False`` is the policy/replay path: ``action/head`` was
    trained/recorded as the POST-LPF/POST-deadband pose (the exact ``ik.solve()``
    input), so re-running the stateful filters would double-process it -- extra
    first-order lag plus a deadband on an already-shaped signal. Alignment still
    passes ``True`` because it creates a fresh live glide target. Everything else
    is identical in both modes: the per-tick joint-step clamp, hold logic, and
    the closed-loop base PD inside ``driver.actuate()`` are hardware safety
    layers, never target postprocessing, and must not be bypassed.
    """
    left_target, right_target, head_target = interp.at(now)
    if live_head_filters:
        head_target = head_lpf.filter(head_target, dt)
        head_target = head_deadband.filter(head_target)
    head_mode = str(getattr(getattr(driver, "cfg", None), "head_mode", "ik"))
    if head_mode == "track":
        head_joints = ik.solve_head(head_target, dt)
        result = ik.solve(left_target, right_target, dt, head_joints=head_joints)
    else:
        result = ik.solve(left_target, right_target, dt, head_target=head_target)
    hold_reason = driver.extra_hold(now, last_cmd_wall, estop=estop)
    # compute_hold_reason deliberately returns None on estop (the caller owns that
    # layer), so estop must be OR'd in here, exactly like the teleop follower.
    hold = estop or (not result.success) or result.held or (hold_reason is not None)
    driver.actuate(result, float(grip_left), float(grip_right), enable, hold, dt)
    if driver._episode is not None:  # noqa: SLF001 -- same recording path as teleop
        driver.record_tick(result, hold, now, left_target=left_target,
                           right_target=right_target, head_target=head_target,
                           source_timestamp_ns=source_timestamp_ns,
                           source_receive_wall_ns=source_receive_wall_ns)
    return result, hold, hold_reason


def load_alignment_references(
    path: str, ik, *, target_frame: str = "world"
) -> tuple[np.ndarray, np.ndarray]:
    """Reference L/R EEF poses via the leader's loader (no reimplemented parsing)."""
    from omniteleop.leader.wbc_reference_alignment import (  # noqa: PLC0415 -- heavy
        load_reference_ee_poses,
    )

    ref = load_reference_ee_poses(path, ik, target_frame=target_frame)
    if ref is None:
        raise RuntimeError(f"alignment reference episode yielded no poses: {path!r}")
    return ref


def _achieved_world_ee_poses(driver, fk: WBCPolicyFK) -> tuple[np.ndarray, np.ndarray]:
    """Where the arms ACTUALLY are in the canonical control-feedback world."""
    measured = driver._read_measured_joints()  # noqa: SLF001 -- deliberate reuse
    for grp in ("torso", "left_arm", "right_arm", "head"):
        if measured.get(grp) is None:
            raise RuntimeError(f"measured {grp} joints unavailable; cannot verify alignment")
    if driver._odom is None:  # noqa: SLF001
        raise RuntimeError("odometry required to verify alignment in the world frame")
    q = fk.q_from_raw(
        measured["torso"], measured["left_arm"], measured["right_arm"], measured["head"],
        base_xyyaw=_policy_base_pose(driver),
    )
    return fk.frame_pose(fk.left_ee_frame, q), fk.frame_pose(fk.right_ee_frame, q)


def _achieved_ee_poses(
    driver, fk: WBCPolicyFK, *, target_frame: str
) -> tuple[np.ndarray, np.ndarray]:
    """Measured arm poses in the action frame selected by the policy contract."""
    if target_frame == "world":
        return _achieved_world_ee_poses(driver, fk)
    if target_frame != "current_base":
        raise ValueError(f"unsupported action target frame {target_frame!r}")
    measured = driver._read_measured_joints()  # noqa: SLF001 -- deliberate reuse
    for grp in ("torso", "left_arm", "right_arm", "head"):
        if measured.get(grp) is None:
            raise RuntimeError(f"measured {grp} joints unavailable; cannot verify alignment")
    q = fk.q_from_raw(
        measured["torso"], measured["left_arm"], measured["right_arm"],
        measured["head"], base_xyyaw=None,
    )
    return fk.frame_pose(fk.left_ee_frame, q), fk.frame_pose(fk.right_ee_frame, q)


def run_reference_alignment(
    *,
    ik,
    driver,
    enable: dict,
    interp,
    head_lpf,
    head_deadband,
    left_ref: np.ndarray,
    right_ref: np.ndarray,
    head_current: np.ndarray,
    dt: float,
    read_achieved,
    align_seconds: float = DEFAULT_ALIGN_SECONDS,
    timeout_s: float = DEFAULT_ALIGN_TIMEOUT_S,
    now_fn=time.perf_counter,
    sleep_fn=time.sleep,
) -> dict:
    """Actively glide the arms to the reference EEF poses; return once settled.

    Unlike the leader's passive ``ReferenceAlignmentGate`` (a human drives the
    targets until they match), the robot drives ITSELF here: one interpolator
    segment toward ``(left_ref, right_ref)`` over ``align_seconds`` while the
    head HOLDS ``head_current`` -- the nominal pose captured right after
    ``ik.reset()``, bit-for-bit the same nominal-head computation the leader
    gates on, so holding it IS the head alignment. The loop is the standard
    LIVE 100 Hz tick (head LPF/deadband active; clamp/hold/base-PD safety all
    inherited). ``last_cmd_wall=now`` because the glide target is internally
    generated -- there is no external source to go stale; odometry staleness
    still holds through ``extra_hold``. Converged = both arms' MEASURED world
    poses within the leader's tolerances for ``REFERENCE_ALIGN_STABLE_S``;
    raises RuntimeError (hard abort -- the caller's ``finally`` stops motion)
    if not converged within ``timeout_s``.
    """
    from omniteleop.leader.wbc_reference_alignment import (  # noqa: PLC0415 -- heavy
        REFERENCE_ALIGN_POS_TOL_MM,
        REFERENCE_ALIGN_ROT_TOL_DEG,
        REFERENCE_ALIGN_STABLE_S,
        pose_rotation_error_deg,
    )

    t_start = now_fn()
    interp.push(left_ref, right_ref, head_current, now=t_start, duration=align_seconds)
    stable_since: float | None = None
    ticks = 0
    last_print = t_start
    pos_mm = (float("inf"), float("inf"))
    rot_deg = (float("inf"), float("inf"))
    last_tick_status = "not-run"
    while True:
        now = now_fn()
        if now - t_start > timeout_s:
            raise RuntimeError(
                f"[wbc_policy_rollout] reference alignment did not converge within "
                f"{timeout_s:g}s (L {pos_mm[0]:.0f}mm/{rot_deg[0]:.1f}deg, "
                f"R {pos_mm[1]:.0f}mm/{rot_deg[1]:.1f}deg vs tol "
                f"{REFERENCE_ALIGN_POS_TOL_MM:g}mm/{REFERENCE_ALIGN_ROT_TOL_DEG:g}deg; "
                f"last WBC tick: {last_tick_status})"
            )
        result, hold, hold_reason = wbc_tick(
            ik=ik, driver=driver, enable=enable, interp=interp,
            head_lpf=head_lpf, head_deadband=head_deadband, now=now, dt=dt,
            grip_left=0.0, grip_right=0.0, last_cmd_wall=now,
            live_head_filters=True,
        )
        if not hold:
            last_tick_status = "ok"
        elif hold_reason is not None:
            last_tick_status = f"HOLD:{hold_reason}"
        elif not result.success:
            last_tick_status = "HOLD:ik-failed"
        elif result.held:
            last_tick_status = "HOLD:safety-gate"
        else:
            last_tick_status = "HOLD"
        ticks += 1
        left_now, right_now = read_achieved()
        pos_mm = (
            float(np.linalg.norm(left_now[:3, 3] - left_ref[:3, 3]) * 1000.0),
            float(np.linalg.norm(right_now[:3, 3] - right_ref[:3, 3]) * 1000.0),
        )
        rot_deg = (
            pose_rotation_error_deg(left_ref, left_now),
            pose_rotation_error_deg(right_ref, right_now),
        )
        ok = (
            pos_mm[0] <= REFERENCE_ALIGN_POS_TOL_MM
            and pos_mm[1] <= REFERENCE_ALIGN_POS_TOL_MM
            and rot_deg[0] <= REFERENCE_ALIGN_ROT_TOL_DEG
            and rot_deg[1] <= REFERENCE_ALIGN_ROT_TOL_DEG
        )
        if ok:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= REFERENCE_ALIGN_STABLE_S:
                return {"elapsed": now - t_start, "ticks": ticks,
                        "pos_err_mm": pos_mm, "rot_err_deg": rot_deg}
        else:
            stable_since = None
        if now - last_print >= 0.5:
            last_print = now
            stable_s = 0.0 if stable_since is None else now - stable_since
            print(f"[wbc_policy_rollout] aligning: L {pos_mm[0]:5.0f}mm/{rot_deg[0]:4.1f}deg "
                  f"R {pos_mm[1]:5.0f}mm/{rot_deg[1]:4.1f}deg  "
                  f"stable {stable_s:.1f}/{REFERENCE_ALIGN_STABLE_S:g}s", end="\r")
        sleep = dt - (now_fn() - now)
        if sleep > 0:
            sleep_fn(sleep)


def _load_wbc_vr_robot():
    """Load the teleop follower script as a module (HardwareDriver, _build_ik, ...)."""
    path = Path(__file__).resolve().parent / "wbc_vr_robot.py"
    spec = importlib.util.spec_from_file_location("wbc_vr_robot_for_rollout", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_wbc_joystick_robot():
    """Load the joystick follower's IK/driver without making scripts a package."""
    path = Path(__file__).resolve().parent / "wbc_joystick_robot.py"
    spec = importlib.util.spec_from_file_location(
        "wbc_joystick_robot_for_rollout", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class PolicyObservation:
    """One raw policy observation plus the exact sensor/state timing it used."""

    state: np.ndarray               # (32,) float32, frame per the checkpoint
    head_rgb: np.ndarray            # (H, W, 3) uint8
    # arm ("left"/"right") -> (H, W, 3) uint8. Empty when the policy has no wrist
    # input; carries exactly the arms the checkpoint declares.
    wrist_rgb: dict[str, np.ndarray]
    base_pose: np.ndarray | None = None
    head_frame_ns: int = -1
    head_receive_wall_ns: int = -1
    wrist_frame_ns: dict[str, int] | None = None
    wrist_receive_wall_ns: dict[str, int] | None = None
    # Corrected head-local minus wrist-local capture skew for exactly the images above.
    camera_skew_ns: dict[str, int] | None = None
    # Capture-to-grab-end age in the local clock after camera-host correction.
    camera_age_ns: dict[str, int] | None = None
    grab_start_wall_ns: int = -1
    grab_end_wall_ns: int = -1
    state_sensor_read_start_wall_ns: int = -1
    state_sensor_read_end_wall_ns: int = -1
    state_ready_wall_ns: int = -1
    ready_wall_ns: int = -1
    ready_perf_s: float = float("nan")
    gripper_event_count: dict[str, int] | None = None
    gripper_read_wall_ns: dict[str, int] | None = None


@dataclass(frozen=True)
class _PolicyStateReading:
    """One internally consistent state vector and its local read bracket."""

    state: np.ndarray
    base_pose: np.ndarray
    sensor_read_start_wall_ns: int
    sensor_read_end_wall_ns: int
    state_ready_wall_ns: int
    gripper_event_count: dict[str, int]
    gripper_read_wall_ns: dict[str, int]


class _PolicyBundle:
    """Checkpoint + pre/post processors + chunk-level inference for the WBC schema."""

    def __init__(self, policy_path: str, device: str | None = None,
                 scene_diff_repo: str = DEFAULT_SCENE_DIFF_REPO,
                 dataset_meta_path: str | None = None) -> None:
        import torch  # noqa: PLC0415 -- heavy, hardware/GPU path only
        from lerobot.configs import PreTrainedConfig  # noqa: PLC0415
        from lerobot.policies import (  # noqa: PLC0415
            get_policy_class,
            make_pre_post_processors,
        )
        from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE  # noqa: PLC0415

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        pcfg = PreTrainedConfig.from_pretrained(policy_path)
        pcfg.pretrained_path = policy_path
        (
            declared_policy_action_schema,
            self.policy_action_schema_provenance,
        ) = _discover_checkpoint_policy_action_schema(
            policy_path,
            pcfg,
            explicit_dataset_meta=dataset_meta_path,
        )
        self.checkpoint_path = str(Path(policy_path).expanduser().resolve())
        (
            self.base_pose_source,
            self.base_pose_source_provenance,
        ) = _discover_checkpoint_base_pose_source(
            policy_path,
            pcfg,
            explicit_dataset_meta=dataset_meta_path,
        )
        (
            self.camera_alignment_contract,
            self.camera_alignment_provenance,
        ) = _discover_checkpoint_camera_alignment_contract(
            policy_path,
            pcfg,
            explicit_dataset_meta=dataset_meta_path,
        )
        action_timing, self.action_timing_provenance = (
            _discover_checkpoint_action_timing_contract(
                policy_path,
                pcfg,
                explicit_dataset_meta=dataset_meta_path,
            )
        )
        self.dataset_fps = (
            None if action_timing is None else action_timing["dataset_fps"]
        )
        self.action_offset_frames = (
            None if action_timing is None else action_timing["action_offset_frames"]
        )
        self.action_offset_semantics = (
            None if action_timing is None else action_timing["action_offset_semantics"]
        )
        self.position_condition_mode = str(
            getattr(pcfg, "position_condition_mode", "not_applicable")
        )
        self.action_frame_mode = str(getattr(pcfg, "action_frame_mode", "world"))
        if self.action_frame_mode == "mof":
            # MoF derives every reference frame itself and lifts base -> world in-graph, so
            # the checkpoint states which frame it wants.
            self.state_frame = str(getattr(pcfg, "mof_state_frame", "base"))
        else:
            self.state_frame = "base"
        self.policy = (
            get_policy_class(pcfg.type).from_pretrained(policy_path, config=pcfg)
            .to(self.device).eval()
        )
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=pcfg, pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        in_feats = self.policy.config.input_features
        out_feats = self.policy.config.output_features
        state_dim = int(in_feats[OBS_STATE].shape[0])
        action_dim = int(out_feats[ACTION].shape[0])
        if state_dim != len(STATE_AXES) or action_dim not in (
            len(ACTION_AXES), len(JOYSTICK_ACTION_AXES)
        ):
            raise ValueError(
                f"checkpoint dims state={state_dim}/action={action_dim} do not match "
                f"the supported WBC schemas ({len(STATE_AXES)}/29 or "
                "32); wrong checkpoint?"
            )
        self.action_dim = action_dim
        inferred_policy_action_schema = (
            JOYSTICK_POLICY_ACTION_SCHEMA
            if action_dim == len(JOYSTICK_ACTION_AXES)
            else WBC_POLICY_ACTION_SCHEMA
        )
        if (
            declared_policy_action_schema is not None
            and declared_policy_action_schema != inferred_policy_action_schema
        ):
            raise ValueError(
                "checkpoint output dimension contradicts policy-action provenance: "
                f"{action_dim}-D implies {inferred_policy_action_schema!r}, but "
                f"metadata declares {declared_policy_action_schema!r}"
            )
        if (
            inferred_policy_action_schema == JOYSTICK_POLICY_ACTION_SCHEMA
            and declared_policy_action_schema is None
        ):
            raise ValueError(
                "32-D checkpoint has no versioned joystick action declaration. Pass "
                "--policy-dataset-meta for the original ported dataset; refusing to "
                "guess the chassis/frame semantics from dimension alone."
            )
        self.policy_action_schema = inferred_policy_action_schema
        self.action_target_frame = (
            "current_base"
            if self.policy_action_schema == JOYSTICK_POLICY_ACTION_SCHEMA
            else "world"
        )
        if self.policy_action_schema == JOYSTICK_POLICY_ACTION_SCHEMA:
            if self.action_frame_mode == "mof":
                raise ValueError("32-D joystick actions are incompatible with MoF world lifting")
            self.action_frame_mode = "current_base"
            self.state_frame = "base"
        img_keys = [k for k in in_feats if k.startswith("observation.images.")]
        # left/right name the ARM the wrist camera is mounted on, matching the porter's
        # observation.images.{left,right}_wrist_rgb.
        known = {
            "observation.images.head_rgb",
            "observation.images.left_wrist_rgb",
            "observation.images.right_wrist_rgb",
        }
        if not img_keys or set(img_keys) - known:
            raise ValueError(f"unsupported image inputs {img_keys}; expected subset of {known}")
        self.use_head = "observation.images.head_rgb" in in_feats
        self.wrist_arms: list[str] = [
            arm
            for arm in ("left", "right")
            if f"observation.images.{arm}_wrist_rgb" in in_feats
        ]
        if not self.use_head:
            raise ValueError("policy must consume observation.images.head_rgb")
        chw = in_feats["observation.images.head_rgb"].shape  # (C, H, W)
        self.image_hw = (int(chw[1]), int(chw[2]))

        # Optional SceneDiff env-state: the checkpoint declares a constant object_nums*3
        # world-frame vector. It is installed only as part of one ordered live condition,
        # before the observation queues can fill.
        self.use_env_state = OBS_ENV_STATE in in_feats
        self.env_state_dim = int(in_feats[OBS_ENV_STATE].shape[0]) if self.use_env_state else 0
        if self.use_env_state and (self.env_state_dim < 3 or self.env_state_dim % 3 != 0):
            raise ValueError(
                f"observation.environment_state dim {self.env_state_dim} must be a positive "
                "multiple of 3 (object_nums*3); wrong checkpoint?"
            )
        self._env_state: np.ndarray | None = None
        self.scene_requirements = SceneRequirements(
            object_nums=self.env_state_dim // 3 if self.use_env_state else 0,
            needs_env_state=self.use_env_state,
        )
        self.scene_diff_repo = Path(scene_diff_repo).expanduser().resolve()
        self._condition: LiveObjectCondition | None = None
        self._validation_report: dict | None = None

        self.n_obs_steps = int(getattr(pcfg, "n_obs_steps", 1))
        n_action_steps = getattr(pcfg, "n_action_steps", None)
        if n_action_steps is None:
            raise ValueError("checkpoint config lacks n_action_steps; cannot chunk")
        self.n_action_steps = int(n_action_steps)
        if self.n_obs_steps < 1 or self.n_action_steps < 1:
            raise ValueError(
                f"bad checkpoint config: n_obs_steps={self.n_obs_steps} "
                f"n_action_steps={self.n_action_steps}"
            )
        # Diffusion-family policies read observations from internal deques that
        # predict_action_chunk stacks over n_obs_steps; ACT-family read the batch
        # directly (single obs step). predict_chunk feeds the queues explicitly.
        self._uses_obs_queues = hasattr(self.policy, "_queues")
        if self.n_obs_steps > 1 and not self._uses_obs_queues:
            raise ValueError(
                f"policy type {pcfg.type!r} with n_obs_steps={self.n_obs_steps} but no "
                "observation queues -- chunk-level inference unsupported"
            )

        # select_action() compat state (vis_wbc_policy_prediction.py): rolling raw
        # observation history + the unconsumed tail of the last predicted chunk.
        self._obs_history: deque[PolicyObservation] = deque(maxlen=self.n_obs_steps)
        self._chunk_tail: deque[np.ndarray] = deque()
        self._last_inference_trace: dict | None = None
        self._model_environment_trace: np.ndarray | None = None

    def reset(self) -> None:
        self.policy.reset()
        self._obs_history.clear()
        self._chunk_tail.clear()
        self._last_inference_trace = None
        self._model_environment_trace = None
        self._env_state = None
        self._condition = None
        self._validation_report = None

    def inference_metadata(self) -> dict[str, np.ndarray]:
        """Constant checkpoint metadata written once in the policy-IO episode."""
        policy_action_schema = str(
            getattr(self, "policy_action_schema", WBC_POLICY_ACTION_SCHEMA)
        )
        action_dim = int(getattr(self, "action_dim", len(ACTION_AXES)))
        action_target_frame = str(
            getattr(
                self,
                "action_target_frame",
                "current_base"
                if policy_action_schema == JOYSTICK_POLICY_ACTION_SCHEMA
                else "world",
            )
        )
        metadata = {
            "checkpoint_path": np.asarray(self.checkpoint_path.encode("utf-8")),
            "position_condition_mode": np.asarray(
                self.position_condition_mode.encode("utf-8")
            ),
            "action_frame_mode": np.asarray(self.action_frame_mode.encode("utf-8")),
            "policy_action_schema": np.asarray(
                policy_action_schema.encode("utf-8")
            ),
            "policy_action_schema_provenance": np.asarray(
                str(getattr(self, "policy_action_schema_provenance", "legacy inferred")).encode(
                    "utf-8"
                )
            ),
            "action_target_frame": np.asarray(
                action_target_frame.encode("utf-8")
            ),
            "action_dim": np.asarray(action_dim, np.int64),
        }
        source = getattr(self, "base_pose_source", None)
        if source is not None:
            metadata["base_pose_source"] = np.asarray(str(source).encode("utf-8"))
            metadata["base_pose_source_provenance"] = np.asarray(
                str(getattr(self, "base_pose_source_provenance", "unknown")).encode(
                    "utf-8"
                )
            )
        alignment = getattr(self, "camera_alignment_contract", None)
        if alignment is not None:
            metadata["camera_alignment_mode"] = np.asarray(
                str(alignment["mode"]).encode("utf-8")
            )
            metadata["camera_alignment_clock_domain"] = np.asarray(
                str(alignment["clock_domain"]).encode("utf-8")
            )
            if alignment["max_abs_skew_ns"] is not None:
                metadata["camera_alignment_max_abs_skew_ns"] = np.asarray(
                    alignment["max_abs_skew_ns"], np.int64
                )
            if alignment["max_camera_age_ns"] is not None:
                metadata["camera_alignment_max_camera_age_ns"] = np.asarray(
                    alignment["max_camera_age_ns"], np.int64
                )
            metadata["camera_alignment_verified"] = np.asarray(
                bool(alignment["verified"]), np.bool_
            )
            metadata["camera_alignment_provenance"] = np.asarray(
                str(getattr(self, "camera_alignment_provenance", "unknown")).encode(
                    "utf-8"
                )
            )
        dataset_fps = getattr(self, "dataset_fps", None)
        action_offset_frames = getattr(self, "action_offset_frames", None)
        if dataset_fps is not None and action_offset_frames is not None:
            metadata["dataset_fps"] = np.asarray(dataset_fps, np.float64)
            metadata["action_offset_frames"] = np.asarray(
                action_offset_frames, np.int64
            )
            metadata["action_offset_semantics"] = np.asarray(
                str(
                    getattr(
                        self, "action_offset_semantics", _ACTION_OFFSET_SEMANTICS
                    )
                ).encode("utf-8")
            )
            metadata["action_timing_provenance"] = np.asarray(
                str(getattr(self, "action_timing_provenance", "unknown")).encode(
                    "utf-8"
                )
            )
        return metadata

    def last_inference_trace(self) -> dict | None:
        """Small CPU snapshot from the most recent completed inference."""
        return self._last_inference_trace

    def describe(self) -> str:
        """One-line banner for _run_rollout. Other policy families override this."""
        return (f"LeRobot policy on {self.device}; head+"
                f"{'+'.join(f'{a}_wrist' for a in self.wrist_arms) or 'no-wrist'} "
                f"@ {self.image_hw}; "
                f"state_frame={self.state_frame}; "
                f"base_pose_source={self.base_pose_source or 'undeclared'}; "
                f"camera_alignment="
                f"{(self.camera_alignment_contract or {}).get('mode', 'undeclared')}; "
                f"timing={self.dataset_fps or 'undeclared'}fps/"
                f"offset={self.action_offset_frames if self.action_offset_frames is not None else 'undeclared'}; "  # noqa: E501
                f"action={self.policy_action_schema}/{self.action_dim}D "
                f"frame={self.action_target_frame}; "
                f"n_obs_steps={self.n_obs_steps} n_action_steps={self.n_action_steps}")

    def install_live_object_condition(self, condition: LiveObjectCondition) -> None:
        """Install the ordered condition through the same family-neutral seam."""
        if not self.scene_requirements.needs_bootstrap:
            raise RuntimeError("unconditioned checkpoint cannot install a live condition")
        if len(condition.positions) != self.scene_requirements.object_nums:
            raise ValueError("live condition object count does not match checkpoint")
        env_state = np.asarray(condition.env_state, dtype=np.float32).reshape(-1)
        if env_state.shape != (self.env_state_dim,) or not np.all(np.isfinite(env_state)):
            raise ValueError(
                f"env-state must be a finite ({self.env_state_dim},) vector, "
                f"got {env_state.shape}"
            )
        self._env_state = env_state.copy()
        self._model_environment_trace = None
        self._condition = condition

    def validate_live_object_condition(
        self,
        condition: LiveObjectCondition,
        *,
        capture_validation_frame,
        stage_dir: Path,
    ) -> dict:
        """Validate position-only LeRobot conditions with a disposable eager tracker."""
        api = load_live_tracker_api(self.scene_diff_repo)
        owner = api.Sam31VosTracker(do_compile=False)
        validation_error: BaseException | None = None
        try:
            fresh = capture_validation_frame()
            report = validate_live_condition_with_tracker(
                owner,
                condition,
                fresh,
                min_depth_m=MIN_DEPTH_M,
                max_depth_m=MAX_DEPTH_M,
                stage_dir=stage_dir,
                tracker_compile=False,
            )
            self._validation_report = report
            return report
        except BaseException as exc:
            validation_error = exc
            raise
        finally:
            cleanup_errors: list[BaseException] = []
            try:
                owner.close()
            except BaseException as close_error:
                cleanup_errors.append(close_error)
            del owner
            gc.collect()
            if self._torch.cuda.is_available():
                try:
                    self._torch.cuda.synchronize()
                    self._torch.cuda.empty_cache()
                    self._torch.cuda.synchronize()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                if validation_error is not None:
                    for cleanup_error in cleanup_errors:
                        validation_error.add_note(
                            "tracker cleanup also failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                else:
                    primary = cleanup_errors[0]
                    for cleanup_error in cleanup_errors[1:]:
                        primary.add_note(
                            "additional tracker cleanup failure: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                    raise primary

    def start_episode_runtime(self, **_kwargs) -> None:
        """LeRobot env-state conditioning has no steady-state tracker."""

    def finish_episode_runtime(self) -> None:
        """LeRobot env-state conditioning owns no episode-scoped runtime."""

    @property
    def initial_head_timestamp_ns(self) -> int:
        """Freshness floor that keeps the validation frame out of policy history."""
        if self._validation_report is not None:
            return int(self._validation_report["head_timestamp_ns"])
        if self._condition is not None:
            return int(self._condition.camera_timestamp_ns)
        return -1

    def live_condition_metadata(self) -> dict[str, np.ndarray] | None:
        if self._condition is None:
            return None
        condition = self._condition
        result: dict = {
            "order": condition.order,
            "positions": condition.positions,
            "scene_diff_obj_ids": condition.scene_diff_obj_ids,
            "camera_timestamp_ns": np.asarray(condition.camera_timestamp_ns, np.int64),
            "world_frame_epoch": np.asarray(condition.world_frame_epoch, np.int64),
            "source_artifact": np.asarray(str(condition.source_artifact).encode()),
            "provenance_hashes": np.asarray(
                [value.encode() for value in condition.provenance_hashes]
            ),
        }
        if self._validation_report is not None:
            result["validation"] = {
                key: np.asarray(value) for key, value in self._validation_report.items()
            }
        return result

    def _chw(self, img_hwc: np.ndarray):
        import cv2  # noqa: PLC0415

        h, w = self.image_hw
        if img_hwc.shape[:2] != (h, w):
            img_hwc = cv2.resize(img_hwc, (w, h), interpolation=cv2.INTER_AREA)
        t = self._torch.from_numpy(np.ascontiguousarray(img_hwc))
        return t.permute(2, 0, 1).float() / 255.0

    def _sample_dict(self, obs: PolicyObservation) -> dict:
        """Validate one raw observation and build the (unbatched) sample dict."""
        state = np.asarray(obs.state, dtype=np.float32).reshape(-1)
        if state.shape != (len(STATE_AXES),) or not np.all(np.isfinite(state)):
            raise ValueError(f"bad observation.state (shape {state.shape} or non-finite)")
        sample: dict = {
            "observation.state": self._torch.from_numpy(state),
            "observation.images.head_rgb": self._chw(obs.head_rgb),
        }
        for arm in self.wrist_arms:
            frame = obs.wrist_rgb.get(arm)
            if frame is None:
                raise ValueError(
                    f"policy consumes {arm}_wrist_rgb but no {arm} wrist frame is available"
                )
            sample[f"observation.images.{arm}_wrist_rgb"] = self._chw(frame)
        if self.use_env_state:
            if self._env_state is None:
                raise RuntimeError(
                    "policy consumes observation.environment_state but the "
                    "checkpoint-derived live condition was not installed"
                )
            sample["observation.environment_state"] = self._torch.from_numpy(self._env_state)
        return sample

    def predict_chunk(self, obs_history: Sequence[PolicyObservation]) -> np.ndarray:
        """One full forward pass -> ``(n_action_steps, action_dim)`` policy actions.

        ``obs_history`` is oldest-first at the DATASET cadence (1/fps apart, the
        spacing the n_obs_steps>1 policy was trained on -- NOT the replan
        interval), at most ``n_obs_steps`` long; shorter histories are left-padded
        by repeating the oldest sample, exactly like LeRobot's ``populate_queues``
        bootstraps an episode start. Frame 0 of the returned chunk is the
        CURRENT-step command (modeling_diffusion.py extracts actions from
        ``start = n_obs_steps - 1``).

        Bypasses the policy's internal ACTION queue: each observation runs through
        ``self.pre``, the observation queues are populated the way
        ``modeling_diffusion.select_action`` does it, and ``predict_action_chunk``
        runs once. ``self.post`` de-normalizes the whole ``(1, n, 29)`` chunk.
        """
        from lerobot.policies.utils import populate_queues  # noqa: PLC0415
        from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES  # noqa: PLC0415

        history = list(obs_history)
        if not 1 <= len(history) <= self.n_obs_steps:
            raise ValueError(
                f"obs_history must hold 1..{self.n_obs_steps} samples, got {len(history)}"
            )
        history = [history[0]] * (self.n_obs_steps - len(history)) + history
        batch = None
        for obs in history:
            batch = dict(self.pre(self._sample_dict(obs)))
            # The pipeline emits a placeholder ACTION key at inference; it must not
            # reach the queues (modeling_diffusion.select_action pops it likewise).
            batch.pop(ACTION, None)
            if self._uses_obs_queues:
                if self.policy.config.image_features:
                    batch[OBS_IMAGES] = self._torch.stack(
                        [batch[key] for key in self.policy.config.image_features], dim=-4
                    )
                self.policy._queues = populate_queues(  # noqa: SLF001
                    self.policy._queues, batch  # noqa: SLF001
                )
        with self._torch.no_grad():
            chunk = self.policy.predict_action_chunk(batch)
        model_env = None
        if self.use_env_state and self._model_environment_trace is None:
            if self._uses_obs_queues:
                # This is the same queue and stack order consumed inside DiffusionPolicy's
                # predict_action_chunk. It has already passed the selector and normalizer.
                model_env = self._torch.stack(
                    list(self.policy._queues[OBS_ENV_STATE]), dim=1  # noqa: SLF001
                )
            else:
                # Direct-batch policies consume the last preprocessed observation.
                model_env = batch[OBS_ENV_STATE]
                if model_env.ndim == 2:
                    model_env = model_env.unsqueeze(1)
        action_dim = int(getattr(self, "action_dim", len(ACTION_AXES)))
        if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[1] < self.n_action_steps:
            raise ValueError(
                f"predict_action_chunk returned {tuple(chunk.shape)}; expected "
                f"(1, >={self.n_action_steps}, {action_dim})"
            )
        chunk = self.post(chunk[:, : self.n_action_steps])
        out = np.asarray(chunk.detach().cpu().numpy(), dtype=np.float32)
        out = out.reshape(-1, out.shape[-1])
        if out.shape != (self.n_action_steps, action_dim) or not np.all(
            np.isfinite(out)
        ):
            raise ValueError(f"bad action chunk {out.shape} (or non-finite)")
        if model_env is not None:
            model_env_np = np.asarray(model_env.detach().cpu().numpy(), dtype=np.float32)
            expected = (1, self.n_obs_steps, self.env_state_dim)
            if model_env_np.shape != expected or not np.all(np.isfinite(model_env_np)):
                raise ValueError(
                    f"bad post-normalizer environment tensor {model_env_np.shape}; "
                    f"expected finite {expected}"
                )
            # The raw environment state is pinned for the episode, so selector +
            # normalizer output is invariant. Snapshot the exact model queue once and
            # reuse it in subsequent in-memory records (no repeated GPU copy).
            self._model_environment_trace = model_env_np[0].copy()
        self._last_inference_trace = None
        if self.use_env_state:
            if self._env_state is None:
                raise RuntimeError("env-state-conditioned inference has no raw env-state")
            if self._model_environment_trace is None:
                raise RuntimeError("env-state-conditioned inference has no normalized env trace")
            self._last_inference_trace = {
                "position_condition": {
                    "raw_source_destination_xyz": self._env_state.copy(),
                    "model_input_normalized": self._model_environment_trace.copy(),
                },
                "prediction": {
                    # First postprocessed action: left XYZ then right XYZ.
                    "first_arm_xyz": np.stack((out[0, 0:3], out[0, 10:13])).astype(
                        np.float32, copy=True
                    ),
                },
            }
        return out

    def select_action(self, state: np.ndarray, head_rgb: np.ndarray,
                      wrist_rgb: dict[str, np.ndarray]) -> np.ndarray:
        """One action per call at the dataset cadence (offline-eval compat shim).

        Used by ``vis_wbc_policy_prediction.py``, NOT by the live rollout (its
        inference worker calls :meth:`predict_chunk`). Keeps a rolling
        ``n_obs_steps`` observation history, replans via :meth:`predict_chunk`
        when the previous chunk is exhausted, and pops one frame per call --
        the same replan cadence, observation history content, and chunk
        anchoring as the old lerobot-internal-queue path.
        """
        self._obs_history.append(PolicyObservation(
            state=np.asarray(state, dtype=np.float32).reshape(-1),
            head_rgb=head_rgb,
            wrist_rgb=wrist_rgb,
        ))
        if not self._chunk_tail:
            self._chunk_tail.extend(self.predict_chunk(tuple(self._obs_history)))
        return np.asarray(self._chunk_tail.popleft(), dtype=np.float32).reshape(-1)


def _policy_base_pose(driver) -> np.ndarray:
    """Measured engage-origin pose from the source that closes the base PD loop."""
    snapshot = getattr(driver, "policy_base_pose_snapshot", None)
    if callable(snapshot):
        pose = np.asarray(snapshot()["pose"], dtype=np.float64)
    else:
        pose = np.asarray(driver._odom.pose, dtype=np.float64)  # noqa: SLF001
    if pose.shape != (3,) or not np.all(np.isfinite(pose)):
        raise RuntimeError(f"policy base pose {pose!r} is not finite (3,)")
    return pose


def _build_state_reading(
    driver, fk: WBCPolicyFK, state_frame: str = "base"
) -> _PolicyStateReading:
    """Build state from one odom snapshot and a counter-consistent gripper read.

    ``state_frame`` MUST match the checkpoint: ``"base"`` for ordinary policies,
    or whatever ``mof_state_frame`` declares for MoF ones. See ``build_state_vector``.
    """
    read_start_ns = time.time_ns()
    measured = driver._read_measured_joints()  # noqa: SLF001 -- deliberate reuse
    for grp in ("torso", "left_arm", "right_arm", "head"):
        if measured.get(grp) is None:
            raise RuntimeError(f"measured {grp} joints unavailable/malformed; cannot build state")
    if driver._odom is None:  # noqa: SLF001
        raise RuntimeError("odometry required (--enable base) for the state base dims")

    policy_base_snapshot = getattr(driver, "policy_base_pose_snapshot", None)
    if callable(policy_base_snapshot):
        base_state = policy_base_snapshot()
        base_pose = np.asarray(base_state["pose"], dtype=np.float32)
    else:  # lightweight compatibility for non-hardware diagnostics/tests
        odom_snapshot = getattr(driver._odom, "snapshot", None)  # noqa: SLF001
        if callable(odom_snapshot):
            odom_state = odom_snapshot()
            base_pose = np.asarray(odom_state["pose"], dtype=np.float32)
        else:
            base_pose = np.asarray(driver._odom.pose, dtype=np.float32)  # noqa: SLF001

    # _poll_gripper_status_step writes value/status before incrementing the event
    # counter. Retry the tiny read if the main thread crossed that commit edge so the
    # value, status timestamp, and counter in this policy observation agree.
    for _attempt in range(3):
        counts_before = dict(getattr(driver, "_grip_status_event_count", {}))
        grip_l = float(driver._last_obs_grip_left)  # noqa: SLF001
        grip_r = float(driver._last_obs_grip_right)  # noqa: SLF001
        statuses = dict(getattr(driver, "_last_grip_status", {}))
        counts_after = dict(getattr(driver, "_grip_status_event_count", {}))
        if counts_before == counts_after:
            break
    else:
        raise RuntimeError("gripper status changed repeatedly while building policy state")
    if not (np.isfinite(grip_l) and np.isfinite(grip_r)):
        raise RuntimeError("gripper obs still NaN (no FC03 reply yet); cannot build state")
    read_end_ns = time.time_ns()

    poses = fk.base_frame_poses(
        measured["torso"], measured["left_arm"], measured["right_arm"], measured["head"]
    )
    state = build_state_vector(poses, base_pose, grip_l, grip_r, state_frame=state_frame)
    ready_ns = time.time_ns()

    def _status_read_ns(side: str) -> int:
        status = statuses.get(side)
        return int(status.get("read_wall_ns", -1)) if isinstance(status, dict) else -1

    return _PolicyStateReading(
        state=state,
        base_pose=base_pose.copy(),
        sensor_read_start_wall_ns=read_start_ns,
        sensor_read_end_wall_ns=read_end_ns,
        state_ready_wall_ns=ready_ns,
        gripper_event_count={
            side: int(counts_after.get(side, -1)) for side in ("left", "right")
        },
        gripper_read_wall_ns={side: _status_read_ns(side) for side in ("left", "right")},
    )


def _build_state(driver, fk: WBCPolicyFK, state_frame: str = "base") -> np.ndarray:
    """Compatibility view returning only the 32-D vector."""
    return _build_state_reading(driver, fk, state_frame).state


def _grab_policy_head_rgb(driver) -> tuple[np.ndarray, int, int] | None:
    """Read head RGB without repeatedly converting depth when the driver supports it.

    Returns ``(image, publisher_capture_ns, local_receive_wall_ns)``. Custom/test
    drivers with only the historical helpers keep working with receive ``-1``.
    """
    sample_fn = getattr(driver, "_grab_head_rgb_sample", None)
    if callable(sample_fn):
        return sample_fn()
    rgb_only = getattr(driver, "_grab_head_rgb", None)
    if callable(rgb_only):
        sample = rgb_only()
        return None if sample is None else (sample[0], int(sample[1]), -1)
    rgbd = driver._grab_head_images()  # noqa: SLF001 -- backward-compatible audited read
    return None if rgbd is None else (rgbd[0], int(rgbd[2]), -1)


def _grab_policy_wrist_rgb(driver, arm: str) -> tuple[np.ndarray, int, int] | None:
    """Exact wrist sample, with a receive-less fallback for legacy test drivers."""
    sample_fn = getattr(driver, "_grab_wrist_image_sample", None)
    if callable(sample_fn):
        return sample_fn(arm)
    sample = driver._grab_wrist_image(arm)  # noqa: SLF001 -- compatibility cache read
    return None if sample is None else (sample[0], int(sample[1]), -1)


def _policy_observation_history_log(
    observations: Sequence[PolicyObservation], t0: float
) -> dict:
    """Small, fixed-shape trace of every observation consumed by one inference."""
    obs = list(observations)
    if not obs:
        raise ValueError("policy observation history cannot be empty")

    def _mapping_value(sample: PolicyObservation, field_name: str, key: str) -> int:
        mapping = getattr(sample, field_name)
        return -1 if mapping is None else int(mapping.get(key, -1))

    def _base_pose(sample: PolicyObservation) -> np.ndarray:
        value = sample.base_pose
        if value is None:
            value = np.asarray(sample.state, dtype=np.float32)[29:32]
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise RuntimeError(f"bad policy observation base pose {value!r}")
        return value

    return {
        "state": np.stack([np.asarray(sample.state, np.float32) for sample in obs]),
        "base_pose": np.stack([_base_pose(sample) for sample in obs]),
        "timing": {
            "head_frame_ns": np.asarray(
                [sample.head_frame_ns for sample in obs], dtype=np.int64
            ),
            "head_receive_wall_ns": np.asarray(
                [sample.head_receive_wall_ns for sample in obs], dtype=np.int64
            ),
            "left_wrist_frame_ns": np.asarray(
                [_mapping_value(sample, "wrist_frame_ns", "left") for sample in obs],
                dtype=np.int64,
            ),
            "right_wrist_frame_ns": np.asarray(
                [_mapping_value(sample, "wrist_frame_ns", "right") for sample in obs],
                dtype=np.int64,
            ),
            "left_wrist_receive_wall_ns": np.asarray(
                [
                    _mapping_value(sample, "wrist_receive_wall_ns", "left")
                    for sample in obs
                ],
                dtype=np.int64,
            ),
            "right_wrist_receive_wall_ns": np.asarray(
                [
                    _mapping_value(sample, "wrist_receive_wall_ns", "right")
                    for sample in obs
                ],
                dtype=np.int64,
            ),
            "head_minus_left_wrist_capture_ns": np.asarray(
                [_mapping_value(sample, "camera_skew_ns", "left") for sample in obs],
                dtype=np.int64,
            ),
            "head_minus_right_wrist_capture_ns": np.asarray(
                [_mapping_value(sample, "camera_skew_ns", "right") for sample in obs],
                dtype=np.int64,
            ),
            "head_camera_age_ns": np.asarray(
                [_mapping_value(sample, "camera_age_ns", "head") for sample in obs],
                dtype=np.int64,
            ),
            "left_wrist_camera_age_ns": np.asarray(
                [_mapping_value(sample, "camera_age_ns", "left_wrist") for sample in obs],
                dtype=np.int64,
            ),
            "right_wrist_camera_age_ns": np.asarray(
                [_mapping_value(sample, "camera_age_ns", "right_wrist") for sample in obs],
                dtype=np.int64,
            ),
            "grab_start_wall_ns": np.asarray(
                [sample.grab_start_wall_ns for sample in obs], dtype=np.int64
            ),
            "grab_end_wall_ns": np.asarray(
                [sample.grab_end_wall_ns for sample in obs], dtype=np.int64
            ),
            "state_sensor_read_start_wall_ns": np.asarray(
                [sample.state_sensor_read_start_wall_ns for sample in obs], dtype=np.int64
            ),
            "state_sensor_read_end_wall_ns": np.asarray(
                [sample.state_sensor_read_end_wall_ns for sample in obs], dtype=np.int64
            ),
            "state_ready_wall_ns": np.asarray(
                [sample.state_ready_wall_ns for sample in obs], dtype=np.int64
            ),
            "ready_wall_ns": np.asarray(
                [sample.ready_wall_ns for sample in obs], dtype=np.int64
            ),
            "ready_offset_s": np.asarray(
                [sample.ready_perf_s - t0 for sample in obs], dtype=np.float64
            ),
            "left_gripper_event_count": np.asarray(
                [
                    _mapping_value(sample, "gripper_event_count", "left")
                    for sample in obs
                ],
                dtype=np.int64,
            ),
            "right_gripper_event_count": np.asarray(
                [
                    _mapping_value(sample, "gripper_event_count", "right")
                    for sample in obs
                ],
                dtype=np.int64,
            ),
            "left_gripper_read_wall_ns": np.asarray(
                [
                    _mapping_value(sample, "gripper_read_wall_ns", "left")
                    for sample in obs
                ],
                dtype=np.int64,
            ),
            "right_gripper_read_wall_ns": np.asarray(
                [
                    _mapping_value(sample, "gripper_read_wall_ns", "right")
                    for sample in obs
                ],
                dtype=np.int64,
            ),
        },
    }


class _InferenceWorker:
    """Async chunk inference: observation gathering + GPU forward OFF the 100 Hz thread.

    Each cycle: gather ``n_obs_steps`` freshness-gated observations spaced
    ``dataset_dt`` apart (the training cadence -- an n_obs_steps=2 policy must see
    0.1 s obs spacing, not the replan interval), run ``predict_chunk``, stamp
    frame ``k`` with ``t_obs + (action_offset_frames + k) * dataset_dt``
    (``t_obs`` = the LAST observation's local grab time), drop
    frames not strictly after ``inference_end + execution_latency``, and queue
    the survivors. All schedule math lives on the workstation ``perf_counter``
    timeline. Camera ``frame_ns`` stamps are used for the same-publisher
    strictly-increasing freshness gate and, for wrist policies, mapped through the
    HardwareDriver's ``meta/camera_ntp`` offsets to choose the same head-nearest wrist
    captures as training. Raw stamps, receive times, and corrected skews are logged.

    THREADING AUDIT (plan.md Task 4 Step 3) -- every cross-thread call is a
    read-only, thread-safe cache read:

    * ``driver._grab_head_rgb`` / ``_grab_wrist_image``: dexcontrol camera
      ``get_obs`` on Zenoh subscriber caches (documented "Thread Safety: This
      method is thread-safe", dexcontrol zed_camera.py). A custom driver's
      historical ``_grab_head_images`` remains a compatibility fallback.
    * ``driver._read_measured_joints`` (via ``_build_state``): dexcontrol
      ``get_joint_pos`` subscriber-cache reads.
    * ``driver.policy_base_pose_snapshot``: lock-protected odom/ARKit cache snapshots.
    * ``driver._last_obs_grip_left/right``: plain float reads (GIL-atomic),
      written only by the main thread's gripper polling.

    All hardware WRITES (actuate, gripper commands, record_tick) stay on the
    main thread. ``io_log.record`` is called only from THIS thread in live mode
    (bare list append; the main thread joins the worker before ``io_log.stop``).
    A crash never dies silently: the traceback is printed and ``failed`` is set;
    the main loop converts it into a clean shutdown through its ``finally``.
    """

    def __init__(self, *, policy: _PolicyBundle, driver, fk: WBCPolicyFK,
                 buffer: ActionScheduleBuffer, io_log, t0: float, dataset_dt: float,
                 policy_interval: float, execution_latency: float,
                 stop_event: threading.Event) -> None:
        if not np.isfinite(dataset_dt) or dataset_dt <= 0.0:
            raise ValueError(f"dataset_dt must be finite and > 0, got {dataset_dt}")
        if not np.isfinite(policy_interval) or policy_interval <= 0.0:
            raise ValueError(f"policy_interval must be finite and > 0, got {policy_interval}")
        if not np.isfinite(execution_latency) or execution_latency < 0.0:
            raise ValueError(f"execution_latency must be finite and >= 0, got {execution_latency}")
        self._policy = policy
        self._driver = driver
        self._fk = fk
        self._buffer = buffer
        self._io_log = io_log
        self._t0 = float(t0)
        self._dataset_dt = float(dataset_dt)
        action_offset_frames = getattr(policy, "action_offset_frames", 0)
        if isinstance(action_offset_frames, (bool, np.bool_)) or not isinstance(
            action_offset_frames, (int, np.integer)
        ) or action_offset_frames < 0:
            raise ValueError(
                "policy.action_offset_frames must be a non-negative integer, got "
                f"{action_offset_frames!r}"
            )
        self._action_offset_frames = int(action_offset_frames)
        self._policy_interval = float(policy_interval)
        self._execution_latency = float(execution_latency)
        self._stop = stop_event
        self.failed = threading.Event()
        self.fail_reason = ""
        self._last_head_ns = int(getattr(policy, "initial_head_timestamp_ns", -1))
        # Per-arm last-used capture stamp; empty when the policy has no wrist input.
        self._last_wrist_ns: dict[str, int] = {
            arm: -1 for arm in getattr(policy, "wrist_arms", ())
        }
        self._max_camera_skew_ns = int(
            getattr(driver, "_record_max_camera_skew_ns", 0)
        )
        self._max_camera_age_ns = int(
            getattr(driver, "_record_max_camera_age_ns", 0)
        )
        self._align_policy_cameras = bool(
            self._last_wrist_ns and self._max_camera_skew_ns > 0
        )
        if self._align_policy_cameras:
            offsets = getattr(driver, "_camera_clock_offsets_ns", {})
            required = {"head", *(f"{arm}_wrist" for arm in self._last_wrist_ns)}
            missing = sorted(required - set(offsets))
            if missing:
                raise RuntimeError(
                    "live policy camera alignment requires camera clock offsets for "
                    f"{missing}; the HardwareDriver recording initialization must query "
                    "all publisher clocks"
                )
            if not callable(getattr(driver, "_poll_camera_alignment_buffers", None)):
                raise RuntimeError(
                    "live policy camera alignment requires HardwareDriver's wrist "
                    "and head history buffers"
                )
            if not callable(getattr(driver, "_select_aligned_camera_samples", None)):
                raise RuntimeError(
                    "live policy camera alignment requires HardwareDriver's "
                    "freshest coherent capture-set selector"
                )
        if self._max_camera_age_ns > 0:
            offsets = getattr(driver, "_camera_clock_offsets_ns", {})
            required = {"head", *(f"{arm}_wrist" for arm in self._last_wrist_ns)}
            missing = sorted(required - set(offsets))
            if missing:
                raise RuntimeError(
                    "live policy camera-age gate requires camera clock offsets for "
                    f"{missing}"
                )
            if not callable(getattr(driver, "_camera_capture_ages_ns", None)):
                raise RuntimeError(
                    "live policy camera-age gate requires HardwareDriver's corrected "
                    "capture-age helper"
                )
        self._last_seen_head_ns = -1
        grip_counts = getattr(driver, "_grip_status_event_count", None)
        # HardwareDriver always exposes the counters. Compatibility drivers that do
        # not are not freshness-gated on gripper telemetry.
        self._last_grip_event_count: dict[str, int] = (
            {side: int(grip_counts.get(side, 0)) for side in ("left", "right")}
            if isinstance(grip_counts, dict)
            else {}
        )
        # Per-grab budget for a strictly-fresh frame: the cameras run ~15 fps, so a
        # new frame normally lands well inside 2 dataset periods; beyond that the
        # cycle is skipped and retried (record_tick's stale-grace abort backstops
        # a truly stalled publisher from the main thread).
        self._grab_timeout = max(2.0 * self._dataset_dt, 0.2)
        self._skipped = 0
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="wbc-policy-inference"
        )

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float) -> bool:
        """Join the worker; True when it exited within ``timeout``."""
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _grab_fresh_observation(self) -> PolicyObservation | None:
        """One freshness-gated observation, or None on timeout/stop.

        Every publisher stamp and, on HardwareDriver, both FC03 event counters must
        strictly advance past the previous sample. The policy therefore never sees a
        duplicate image or silently repeated achieved-gripper observation.
        """
        deadline = time.perf_counter() + self._grab_timeout
        while not self._stop.is_set():
            grab_start_wall_ns = time.time_ns()
            selected_head_sample = None
            selected_samples = None
            camera_skew_ns: dict[str, int] = {}
            if self._align_policy_cameras:
                # Shared with the main recorder: bounded, lock-protected histories are
                # continuously filled even while this thread runs GPU inference. Each
                # consumer supplies independent cursors, so neither can steal frames.
                self._driver._poll_camera_alignment_buffers()  # noqa: SLF001
                (
                    selected_head_sample,
                    selected_samples,
                    camera_skew_ns,
                ) = self._driver._select_aligned_camera_samples(  # noqa: SLF001
                    arms=tuple(self._last_wrist_ns),
                    last_head_ns=self._last_head_ns,
                    last_wrist_ns=self._last_wrist_ns,
                    max_skew_ns=self._max_camera_skew_ns,
                )
                head = (
                    None
                    if selected_head_sample is None or selected_samples is None
                    else (
                        selected_head_sample.left_rgb,
                        int(selected_head_sample.left_capture_ns),
                        int(selected_head_sample.left_receive_ns),
                    )
                )
                wrists = (
                    {arm: None for arm in self._last_wrist_ns}
                    if selected_samples is None
                    else {
                        arm: (sample[0], int(sample[1]), int(sample[2]))
                        for arm, sample in selected_samples.items()
                    }
                )
            else:
                head = _grab_policy_head_rgb(self._driver)
                wrists = {
                    arm: _grab_policy_wrist_rgb(self._driver, arm)
                    for arm in self._policy.wrist_arms
                }
            grab_end_wall_ns = time.time_ns()
            head_ns = None if head is None else int(head[1])
            if head_ns is not None:
                if head_ns < self._last_seen_head_ns:
                    raise RuntimeError(
                        "head camera capture timestamp moved backward by "
                        f"{(self._last_seen_head_ns - head_ns) / 1e6:.3f} ms during "
                        "policy observation gathering"
                    )
                self._last_seen_head_ns = head_ns
            wrist_ns = {
                arm: (None if w is None else int(w[1])) for arm, w in wrists.items()
            }
            camera_age_ns: dict[str, int] = {}
            age_ok = True
            if (
                self._max_camera_age_ns > 0
                and head_ns is not None
                and all(ns is not None for ns in wrist_ns.values())
            ):
                camera_age_ns = self._driver._camera_capture_ages_ns(  # noqa: SLF001
                    grab_end_wall_ns,
                    head_ns=head_ns,
                    wrist_ns={arm: int(ns) for arm, ns in wrist_ns.items()},
                )
                age_ok = all(
                    -_CAMERA_FUTURE_TOLERANCE_NS <= age <= self._max_camera_age_ns
                    for age in camera_age_ns.values()
                )
            grip_counts_raw = getattr(self._driver, "_grip_status_event_count", None)
            grip_counts = (
                {
                    side: int(grip_counts_raw.get(side, 0))
                    for side in ("left", "right")
                }
                if isinstance(grip_counts_raw, dict)
                else {}
            )
            grippers_fresh = not self._last_grip_event_count or all(
                grip_counts[side] > self._last_grip_event_count[side]
                for side in ("left", "right")
            )
            # Every declared stream must advance past what this worker last used, so the
            # policy never sees any camera frame twice.
            fresh = head_ns is not None and head_ns > self._last_head_ns and all(
                ns is not None and ns > self._last_wrist_ns[arm]
                for arm, ns in wrist_ns.items()
            ) and grippers_fresh and age_ok
            if fresh:
                reading = _build_state_reading(
                    self._driver, self._fk, self._policy.state_frame
                )
                if self._last_grip_event_count and any(
                    reading.gripper_event_count[side]
                    <= self._last_grip_event_count[side]
                    for side in ("left", "right")
                ):
                    # A compatibility/custom driver changed its counters during the
                    # candidate read. Retry without committing any camera stamp.
                    continue
                self._last_head_ns = head_ns
                for arm, ns in wrist_ns.items():
                    self._last_wrist_ns[arm] = ns
                if self._last_grip_event_count:
                    self._last_grip_event_count = dict(reading.gripper_event_count)
                ready_wall_ns = time.time_ns()
                ready_perf_s = time.perf_counter()
                return PolicyObservation(
                    state=reading.state, head_rgb=head[0],
                    wrist_rgb={arm: w[0] for arm, w in wrists.items()},
                    base_pose=reading.base_pose,
                    head_frame_ns=head_ns,
                    head_receive_wall_ns=int(head[2]),
                    wrist_frame_ns={arm: int(w[1]) for arm, w in wrists.items()},
                    wrist_receive_wall_ns={
                        arm: int(w[2]) for arm, w in wrists.items()
                    },
                    camera_skew_ns=dict(camera_skew_ns),
                    camera_age_ns=dict(camera_age_ns),
                    grab_start_wall_ns=grab_start_wall_ns,
                    grab_end_wall_ns=grab_end_wall_ns,
                    state_sensor_read_start_wall_ns=reading.sensor_read_start_wall_ns,
                    state_sensor_read_end_wall_ns=reading.sensor_read_end_wall_ns,
                    state_ready_wall_ns=reading.state_ready_wall_ns,
                    ready_wall_ns=ready_wall_ns,
                    ready_perf_s=ready_perf_s,
                    gripper_event_count=reading.gripper_event_count,
                    gripper_read_wall_ns=reading.gripper_read_wall_ns,
                )
            if time.perf_counter() >= deadline:
                return None
            self._stop.wait(0.005)
        return None

    def _gather(self) -> tuple[list[PolicyObservation], float] | None:
        """``n_obs_steps`` fresh observations at the dataset cadence + t_obs."""
        obs_list: list[PolicyObservation] = []
        prev_grab_t = 0.0
        for i in range(self._policy.n_obs_steps):
            if i:
                self._stop.wait(
                    max(0.0, prev_grab_t + self._dataset_dt - time.perf_counter())
                )
                if self._stop.is_set():
                    return None
            obs = self._grab_fresh_observation()
            if obs is None:
                return None
            prev_grab_t = (
                obs.ready_perf_s
                if np.isfinite(obs.ready_perf_s)
                else time.perf_counter()
            )
            obs_list.append(obs)
        return obs_list, prev_grab_t

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                cycle_t0 = time.perf_counter()
                gathered = self._gather()
                if gathered is None:
                    if self._stop.is_set():
                        break
                    self._skipped += 1
                    if self._skipped <= 3 or self._skipped % 20 == 0:
                        print(f"\n[wbc_policy_rollout] inference worker: no fresh "
                              f"camera frame within {self._grab_timeout:.2f}s "
                              f"(skip #{self._skipped}); retrying")
                    continue  # the grab loop already waited out its budget
                if self._stop.is_set():
                    break
                obs_list, t_obs = gathered
                latest_obs = obs_list[-1]
                obs_ready_wall_ns = int(latest_obs.ready_wall_ns)
                observation_history = _policy_observation_history_log(
                    obs_list, self._t0
                )
                t_pred0 = time.perf_counter()
                chunk = self._policy.predict_chunk(obs_list)
                if self._stop.is_set():
                    break
                policy_end = time.perf_counter()
                n = chunk.shape[0]
                timestamps = chunk_schedule_timestamps(
                    observation_time=t_obs,
                    frame_count=n,
                    dataset_dt=self._dataset_dt,
                    action_offset_frames=self._action_offset_frames,
                )
                actions = scheduled_actions_from_chunk(chunk, timestamps)
                queue_time = time.perf_counter()
                kept = drop_stale_actions(actions, queue_time + self._execution_latency)
                prospective_coverage = (
                    kept[-1].timestamp - queue_time if kept else 0.0
                )
                if kept:
                    if self._stop.is_set():
                        break
                    self._buffer.queue(kept, now=queue_time)
                    if kept[-1].timestamp < queue_time + self._policy_interval:
                        print(f"\n[wbc_policy_rollout] WARNING: chunk covers only "
                              f"{prospective_coverage:.2f}s past queue time "
                              f"but the next replan is ~{self._policy_interval:g}s away "
                              f"(inference {policy_end - t_pred0:.2f}s): the buffer may "
                              "underflow terminally. Lower --policy-interval or "
                              "speed up inference.")
                else:
                    print(f"\n[wbc_policy_rollout] WARNING: entire chunk stale "
                          f"(inference {policy_end - t_pred0:.2f}s + latency "
                          f"{self._execution_latency:g}s passed the last frame at "
                          f"t_obs+{(self._action_offset_frames + n - 1) * self._dataset_dt:.2f}s); "
                          "nothing queued")
                schedule_end = time.perf_counter()
                io_frame = {
                    "t": np.float64(t_obs - self._t0),
                    "timestamp_ns": np.int64(time.time_ns()),
                    "state": latest_obs.state,
                    "observation_history": observation_history,
                    # FULL pre-drop chunk: constant (n_action_steps, action_dim) shape so
                    # EpisodeRecorder can np.stack; n_dropped says what was queued.
                    "action": chunk,
                    "action_offsets_s": (timestamps - self._t0).astype(np.float64),
                    # Exact base pose embedded in the final model observation. The old
                    # post-inference odom re-read could be hundreds of milliseconds newer.
                    "base_pose": observation_history["base_pose"][-1],
                    "inference_s": np.float32(policy_end - t_pred0),
                    "scheduler_s": np.float32(schedule_end - policy_end),
                    # Actual completion of scheduler preparation, on the same
                    # perf_counter epoch as action_offsets_s. This is deliberately
                    # recorded instead of reconstructed from several rounded durations.
                    "queue_offset_s": np.float64(schedule_end - self._t0),
                    "prospective_coverage_s": np.float32(prospective_coverage),
                    "obs_gather_s": np.float32(t_pred0 - cycle_t0),
                    # Local wall/perf timestamps at the exact policy-observation handoff.
                    # Join head_frame_ns to main episode meta/camera_ntp/head for
                    # capture->policy-ready latency; the frame timestamp_ns is post-inference.
                    "obs_ready_wall_ns": np.int64(obs_ready_wall_ns),
                    "obs_ready_offset_s": np.float64(t_obs - self._t0),
                    "n_dropped": np.int64(n - len(kept)),
                    "head_frame_ns": np.int64(latest_obs.head_frame_ns),
                    "head_receive_wall_ns": np.int64(
                        latest_obs.head_receive_wall_ns
                    ),
                    # -1 when the policy declares no such wrist input.
                    "left_wrist_frame_ns": np.int64(
                        -1
                        if latest_obs.wrist_frame_ns is None
                        else latest_obs.wrist_frame_ns.get("left", -1)
                    ),
                    "right_wrist_frame_ns": np.int64(
                        -1
                        if latest_obs.wrist_frame_ns is None
                        else latest_obs.wrist_frame_ns.get("right", -1)
                    ),
                    "left_wrist_receive_wall_ns": np.int64(
                        -1
                        if latest_obs.wrist_receive_wall_ns is None
                        else latest_obs.wrist_receive_wall_ns.get("left", -1)
                    ),
                    "right_wrist_receive_wall_ns": np.int64(
                        -1
                        if latest_obs.wrist_receive_wall_ns is None
                        else latest_obs.wrist_receive_wall_ns.get("right", -1)
                    ),
                    "head_minus_left_wrist_capture_ns": np.int64(
                        -1
                        if latest_obs.camera_skew_ns is None
                        else latest_obs.camera_skew_ns.get("left", -1)
                    ),
                    "head_minus_right_wrist_capture_ns": np.int64(
                        -1
                        if latest_obs.camera_skew_ns is None
                        else latest_obs.camera_skew_ns.get("right", -1)
                    ),
                    "head_camera_age_ns": np.int64(
                        -1
                        if latest_obs.camera_age_ns is None
                        else latest_obs.camera_age_ns.get("head", -1)
                    ),
                    "left_wrist_camera_age_ns": np.int64(
                        -1
                        if latest_obs.camera_age_ns is None
                        else latest_obs.camera_age_ns.get("left_wrist", -1)
                    ),
                    "right_wrist_camera_age_ns": np.int64(
                        -1
                        if latest_obs.camera_age_ns is None
                        else latest_obs.camera_age_ns.get("right_wrist", -1)
                    ),
                    "state_sensor_read_start_wall_ns": np.int64(
                        latest_obs.state_sensor_read_start_wall_ns
                    ),
                    "state_sensor_read_end_wall_ns": np.int64(
                        latest_obs.state_sensor_read_end_wall_ns
                    ),
                    "state_ready_wall_ns": np.int64(latest_obs.state_ready_wall_ns),
                }
                trace_fn = getattr(self._policy, "last_inference_trace", None)
                if trace_fn is not None:
                    trace = trace_fn()
                    if trace is not None:
                        io_frame.update(trace)
                # EpisodeRecorder.record is a bare in-memory list append. HDF5 is written
                # only after the worker is joined at episode shutdown.
                if self._stop.is_set():
                    break
                self._io_log.record(io_frame)
                self._stop.wait(
                    max(0.0, self._policy_interval - (time.perf_counter() - cycle_t0))
                )
        except Exception as exc:  # surfaced via self.failed -- never a silent death
            traceback.print_exc()
            self.fail_reason = f"{type(exc).__name__}: {exc}"
            self.failed.set()


def _live_world_T_zed(driver, fk: WBCPolicyFK) -> np.ndarray:
    """FK ``world_T_zed`` for the CURRENT head pose (engage-origin world frame).

    Identical construction to ``make_wbc_before_hdf5.world_T_zed_at`` (training): ``world_T_zed
    = base_pose_to_mat(odom base pose) @ base_T_zed``, where ``base_T_zed =
    fk.base_frame_poses(...)["head"]`` is the ``zed_depth_frame`` OPTICAL pose (z-forward,
    x-right, y-down) -- the exact convention ``extract_first_hdf5_rgb_depth`` back-projects
    depth through (finding 6). So the detected centroids land in the SAME engage-origin world
    as the policy's ``action`` targets and ``observation.state[29:32]`` base anchor.
    """
    measured = driver._read_measured_joints()  # noqa: SLF001 -- same read as _build_state
    for grp in ("torso", "left_arm", "right_arm", "head"):
        if measured.get(grp) is None:
            raise RuntimeError(f"measured {grp} joints unavailable; cannot compute head extrinsic")
    base_T_zed = np.asarray(
        fk.base_frame_poses(measured["torso"], measured["left_arm"],
                            measured["right_arm"], measured["head"])["head"],
        dtype=np.float64,
    )
    if driver._odom is None:  # noqa: SLF001
        raise RuntimeError("base feedback required (--enable base) to anchor world_T_zed")
    base_pose = _policy_base_pose(driver)
    world_T_zed = base_pose_to_mat(base_pose) @ base_T_zed
    if world_T_zed.shape != (4, 4) or not np.all(np.isfinite(world_T_zed)):
        raise RuntimeError(f"bad world_T_zed {world_T_zed.shape} (or non-finite)")
    return world_T_zed.astype(np.float32)


def _scene_log(msg: str) -> None:
    print(f"[wbc_policy_rollout] {msg}", flush=True)


def _prompt_for_task_order(object_nums: int, overlay_path: Path) -> list[int]:
    """Operator picks the task order from the size-slot overlay (shared prompt + task hint).

    Returns ``order`` (length ``object_nums``): ``order[k]`` is the size-slot index assigned
    to task slot k of ``[s1_src, s1_dst, ...]``. For the box->cloth task (object_nums=2):
    s1_src = the BOX, s1_dst = the CLOTH.
    """
    return prompt_for_slot_order(
        object_nums, overlay_path, log=_scene_log,
        hint="for the box->cloth task, s1_src = the BOX, s1_dst = the CLOTH.",
    )


def _invoke_live_scenediff(args: argparse.Namespace, live_hdf5: Path, out_dir: Path,
                           requirements: SceneRequirements) -> Path:
    """Adapt the rollout CLI args to the shared SceneDiff invoker; return episode_0.npz."""
    return _invoke_scenediff(
        repo=args.scene_diff_repo, python=args.scene_diff_python,
        live_hdf5=live_hdf5, out_dir=out_dir, reference=args.reference_hdf5,
        sam=args.sam, obj_num=requirements.object_nums, matching="prompt",
        dino=requirements.needs_env_dino,
        config=args.scenediff_config or None, timeout=args.scenediff_timeout,
        # live_capture.hdf5 embeds the FK world_T_zed extrinsic, so the backprojected
        # positions are engage-origin WORLD -- stamp the npz accordingly (checked below).
        frame_label="world",
        log=_scene_log,
    )


def _live_object_output_dir(save_dir: str | Path, episode_id: int) -> Path:
    if not isinstance(episode_id, (int, np.integer)) or int(episode_id) < 0:
        raise RuntimeError(
            f"live object condition needs a pending episode_<N>.hdf5 id, got {episode_id!r}"
        )
    return Path(save_dir) / "scene_diff" / f"episode_{int(episode_id)}"


def _capture_live_validation_frame(
    driver,
    fk: WBCPolicyFK,
    *,
    newer_than_ns: int,
    timeout_s: float = 2.0,
) -> LiveValidationFrame:
    """Capture one strictly newer RGB-D frame with its current world transform."""
    deadline = time.perf_counter() + timeout_s
    while True:
        head = driver._grab_head_images()  # noqa: SLF001 -- audited cache read
        timestamp_ns = -1 if head is None else int(head[2])
        if timestamp_ns > int(newer_than_ns):
            epoch_before = int(driver._world_frame_epoch)  # noqa: SLF001
            world_t_cam = _live_world_T_zed(driver, fk)
            epoch_after = int(driver._world_frame_epoch)  # noqa: SLF001
            if epoch_before != epoch_after:
                raise RuntimeError("world-frame epoch changed during RGB-D capture")
            return LiveValidationFrame(
                rgb=head[0],
                depth_mm=head[1],
                world_t_cam=world_t_cam,
                head_timestamp_ns=timestamp_ns,
                world_frame_epoch=epoch_after,
            )
        if time.perf_counter() >= deadline:
            raise RuntimeError(
                f"head camera delivered no frame newer than {newer_than_ns} "
                f"within {timeout_s:.1f}s"
            )
        time.sleep(0.005)


def _prepare_live_object_condition(args: argparse.Namespace, driver, fk: WBCPolicyFK,
                                   bundle) -> LiveObjectCondition:
    """Build, order, validate, install, and publish one complete episode condition."""
    from omniteleop.common.head_camera import ZED_K  # noqa: PLC0415

    requirements = bundle.scene_requirements
    if not requirements.needs_bootstrap:
        raise RuntimeError("checkpoint has no live scene requirements")
    episode = getattr(driver, "_episode", None)
    if episode is None:
        raise RuntimeError(
            "live object conditioning requires recording to pair its output with episode_N"
        )
    final_dir = _live_object_output_dir(
        args.save_dir, getattr(episode, "episode_id", None)
    )
    if final_dir.exists() or final_dir.is_symlink():
        raise FileExistsError(
            f"live SceneDiff output exists: {final_dir}; delete it manually before rerunning"
        )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(
        prefix=f".{final_dir.name}.building-", dir=final_dir.parent
    ))

    try:
        if args.prompt_before_capture:
            input("\n>>> Robot is at the rollout start pose. Stage the scene, then press "
                  "ENTER to capture the head frame and run SceneDiff (Ctrl-C to abort) <<<\n")

        intrinsic = np.asarray(getattr(bundle, "intrinsic", ZED_K), dtype=np.float32)
        bootstrap = _capture_live_validation_frame(
            driver, fk, newer_than_ns=-1
        )
        live_hdf5 = stage_dir / "live_capture.hdf5"
        _write_live_capture_hdf5(
            live_hdf5,
            bootstrap.rgb,
            bootstrap.depth_mm,
            bootstrap.world_t_cam,
            intrinsic,
            camera_timestamp_ns=bootstrap.head_timestamp_ns,
            world_frame_epoch=bootstrap.world_frame_epoch,
        )
        print(
            f"[wbc_policy_rollout] live bootstrap frame -> {live_hdf5}  "
            f"rgb={bootstrap.rgb.shape} depth={bootstrap.depth_mm.shape}",
            flush=True,
        )

        _invoke_live_scenediff(args, live_hdf5, stage_dir, requirements)
        overlay_path = stage_dir / "deploy_slot_overlay.png"
        if not overlay_path.is_file():
            raise FileNotFoundError(
                "SceneDiff produced no size-slot overlay; refusing to ask the operator "
                f"to order unseen objects: {overlay_path}"
            )
        artifacts = load_live_object_artifacts(
            stage_dir / "bootstrap_size_slots.npz", live_hdf5
        )
        if len(artifacts.positions_size) != requirements.object_nums:
            raise ValueError("SceneDiff bootstrap object count does not match checkpoint")
        if requirements.needs_env_dino and artifacts.dino_region_feats_size is None:
            raise ValueError("grounding_tokens checkpoint requires live DINO features")

        order = _prompt_for_task_order(
            requirements.object_nums, overlay_path
        )
        condition = artifacts.arrange(order)
        validate = getattr(bundle, "validate_live_object_condition", None)
        if validate is None:
            raise RuntimeError("policy bundle lacks the live-condition validation hook")
        validate(
            condition,
            capture_validation_frame=lambda: _capture_live_validation_frame(
                driver,
                fk,
                newer_than_ns=condition.camera_timestamp_ns,
            ),
            stage_dir=stage_dir,
        )
        published = replace(
            condition, source_artifact=final_dir / "bootstrap_size_slots.npz"
        )
        bundle.install_live_object_condition(published)
        if final_dir.exists() or final_dir.is_symlink():
            raise FileExistsError(
                f"live SceneDiff output appeared during bootstrap: {final_dir}; "
                "delete it manually"
            )
        stage_dir.rename(final_dir)
        print(
            f"[wbc_policy_rollout] live object condition ready: order={order} "
            f"positions={np.round(published.positions, 3).tolist()} -> {final_dir}",
            flush=True,
        )
        return published
    except BaseException as bootstrap_error:
        try:
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
        except BaseException as cleanup_error:
            raise bootstrap_error from cleanup_error
        raise


def _run_persistent_session(
    *,
    run_episode: Callable[[bool], None],
    home_to_nominal: Callable[[], None],
    stop_all_motion: Callable[[], None],
    auto_start: bool,
    input_fn: Callable[[str], str] = input,
) -> None:
    """Run episodes until Ctrl-C is pressed at the between-episode prompt."""
    first_episode = True
    while True:
        if not first_episode:
            try:
                input_fn(
                    "\n[wbc_policy_rollout] Press Enter to home and start the next "
                    "rollout (Ctrl-C exits) ... "
                )
            except KeyboardInterrupt:
                print("\n[wbc_policy_rollout] idle Ctrl-C: exiting session.")
                return
        try:
            if not first_episode:
                home_to_nominal()
            run_episode(first_episode and not auto_start)
        except KeyboardInterrupt:
            stop_all_motion()
            print("\n[wbc_policy_rollout] rollout stopped; process remains ready.")
        first_episode = False


def _start_rollout_recorders(driver, io_log) -> None:
    """Start the paired recorders through the driver's canonical episode boundary."""
    driver.start_recording_episode()
    io_log.start()


def _set_policy_io_static(io_log, policy, *, args=None) -> None:
    """Refresh episode-specific policy and live-condition metadata before recording."""
    static: dict = {}
    if policy is not None:
        metadata_fn = getattr(policy, "inference_metadata", None)
        if metadata_fn is not None:
            static["policy"] = metadata_fn()
        condition_fn = getattr(policy, "live_condition_metadata", None)
        if condition_fn is not None:
            condition = condition_fn()
            if condition is not None:
                static["live_object_condition"] = condition
    if args is not None:
        static["runtime"] = {
            "policy_action_schema": np.asarray(
                str(
                    getattr(args, "policy_action_schema", WBC_POLICY_ACTION_SCHEMA)
                ).encode()
            ),
            "action_target_frame": np.asarray(
                str(getattr(args, "action_target_frame", "world")).encode()
            ),
            "base_pose_source": np.asarray(
                _runtime_base_pose_source(getattr(args, "arkit_base", "off")).encode()
            ),
            "arkit_mode": np.asarray(
                str(getattr(args, "arkit_base", "off")).encode()
            ),
            "camera_alignment_mode": np.asarray(
                (
                    b"head_capture_nearest"
                    if float(getattr(args, "record_max_camera_skew_ms", 0.0)) > 0
                    else b"latest_arrived_compatibility"
                )
            ),
            "camera_alignment_clock_domain": np.asarray(
                b"local_via_camera_ntp"
            ),
            "camera_alignment_max_abs_skew_ns": np.asarray(
                round(float(getattr(args, "record_max_camera_skew_ms", 0.0)) * 1e6),
                np.int64,
            ),
            "camera_alignment_max_camera_age_ns": np.asarray(
                round(float(getattr(args, "record_max_camera_age_ms", 0.0)) * 1e6),
                np.int64,
            ),
            "camera_startup_timeout_ns": np.asarray(
                round(float(getattr(args, "record_startup_timeout", 3.0)) * 1e9),
                np.int64,
            ),
        }
        static["schedule"] = {
            "dataset_fps": np.asarray(args.dataset_fps, np.float64),
            "action_offset_frames": np.asarray(
                0 if policy is None else policy.action_offset_frames, np.int64
            ),
            "action_offset_s": np.asarray(
                0.0
                if policy is None
                else policy.action_offset_frames / args.dataset_fps,
                np.float64,
            ),
            "policy_interval_s": np.asarray(args.policy_interval, np.float64),
            "execution_latency_s": np.asarray(args.execution_latency, np.float64),
            "command_fps": np.asarray(args.cmd_rate, np.float64),
            "n_action_steps": np.asarray(
                0 if policy is None else policy.n_action_steps, np.int64
            ),
            "n_obs_steps": np.asarray(
                0 if policy is None else policy.n_obs_steps, np.int64
            ),
        }
    io_log.set_static(static)


def _synchronize_rollout_episode_id(driver, io_log) -> int:
    """Assign one unused episode id to the main and policy-IO recorders."""
    main_log = getattr(driver, "_episode", None)
    if main_log is None:
        raise RuntimeError("[wbc_policy_rollout] main episode recorder is unavailable")
    if any(
        getattr(recorder, state, False)
        for recorder in (main_log, io_log)
        for state in ("recording", "saving")
    ):
        raise RuntimeError(
            "[wbc_policy_rollout] cannot assign an episode id while a recorder is active"
        )
    episode_id = max(int(main_log.episode_id), int(io_log.episode_id))
    main_log.episode_id = episode_id
    io_log.episode_id = episode_id
    return episode_id


def _prepare_rollout_episode_condition(args, driver, fk, policy, io_log) -> int:
    """Assign the paired id and satisfy checkpoint-declared scene requirements."""
    episode_id = _synchronize_rollout_episode_id(driver, io_log)
    requirements = (
        SceneRequirements() if policy is None else policy.scene_requirements
    )
    if requirements.needs_bootstrap:
        driver.stop_all_motion()
        _prepare_live_object_condition(args, driver, fk, policy)
    return episode_id


def _wait_for_recorder_save(
    recorder,
    *,
    timeout_s: float = _RECORDER_SAVE_TIMEOUT_S,
) -> bool:
    """Wait boundedly for one async save, deferring Ctrl-C until disposition."""
    interrupted = False
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while recorder.saving:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            message = (
                "[wbc_policy_rollout] recorder did not finish within "
                f"{float(timeout_s):g}s ({type(recorder).__name__}); refusing to "
                "wait forever"
            )
            invalidate = getattr(recorder, "invalidate_save", None)
            if callable(invalidate):
                invalidate(message)
            if not recorder.saving:
                return interrupted
            raise TimeoutError(message)
        try:
            time.sleep(min(0.05, remaining))
        except KeyboardInterrupt:
            interrupted = True
    return interrupted


def _call_recorder_terminal(action: Callable[[], str | None]) -> tuple[str | None, bool]:
    """Run stop/abort as one SIGINT-deferred transition on the main thread.

    A SIGINT between ``recording=False`` and the recorder's save/path setup used to make
    the paired cleanup lose that path. Linux delivers a blocked pending SIGINT when the
    old mask is restored; catch it *after* retaining the result and report it to the
    caller for deferred re-raise once both files have a disposition.
    """
    can_mask = (
        threading.current_thread() is threading.main_thread()
        and hasattr(signal, "pthread_sigmask")
        and hasattr(signal, "SIG_BLOCK")
        and hasattr(signal, "SIG_SETMASK")
    )
    if not can_mask:
        return action(), False

    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    result: str | None = None
    action_error: BaseException | None = None
    try:
        result = action()
    except BaseException as exc:
        action_error = exc

    interrupted = False
    try:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    except KeyboardInterrupt:
        interrupted = True

    if action_error is not None:
        raise action_error
    return result, interrupted


def _stop_policy_io_log(io_log) -> None:
    """Stop and verify a standalone policy-IO recorder save."""
    if not io_log.recording:
        return
    path, interrupted = _call_recorder_terminal(io_log.stop)
    interrupted = _wait_for_recorder_save(io_log) or interrupted
    if io_log.last_save_error is not None:
        raise RuntimeError(
            f"[wbc_policy_rollout] policy IO save failed: {io_log.last_save_error}"
        ) from io_log.last_save_error
    if path is not None:
        print(f"[wbc_policy_rollout] policy IO log -> {path}")
    if interrupted:
        raise KeyboardInterrupt


def _finish_rollout_episode(
    *,
    driver,
    io_log,
    stop_event,
    worker,
    policy=None,
    abort_reason: str | None = None,
) -> None:
    """Stop motion and fully finalize one episode without closing hardware.

    A normal completion or operator Ctrl-C uses ``stop()`` for both paired logs. A
    runtime failure uses ``abort()`` for both, preserving synchronized forensic
    ``.partial`` files with ``complete=false`` while keeping them out of every training
    glob.
    """
    main_log = getattr(driver, "_episode", None)
    cleanup_errors: list[BaseException] = []
    recorder_errors: list[BaseException] = []
    interrupted = False

    def run_cleanup_step(action: Callable[[], None]) -> None:
        nonlocal interrupted
        while True:
            try:
                action()
            except KeyboardInterrupt:
                interrupted = True
                continue
            except BaseException as exc:
                cleanup_errors.append(exc)
            return

    run_cleanup_step(stop_event.set)
    run_cleanup_step(driver.stop_all_motion)

    worker_joined = worker is None
    if worker is not None:
        while True:
            try:
                worker_joined = worker.join(timeout=10.0)
            except KeyboardInterrupt:
                interrupted = True
                continue
            except BaseException as exc:
                cleanup_errors.append(
                    RuntimeError(
                        "[wbc_policy_rollout] inference worker join failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                )
            break

    if worker_joined and policy is not None:
        finish_runtime = getattr(policy, "finish_episode_runtime", None)
        if finish_runtime is not None:
            run_cleanup_step(finish_runtime)

    if not worker_joined:
        cleanup_errors.append(
            RuntimeError(
                "[wbc_policy_rollout] inference worker still running after 10s; "
                "refusing to start another rollout"
            )
        )
    if worker is not None and worker.failed.is_set():
        cleanup_errors.append(
            RuntimeError(
                f"[wbc_policy_rollout] inference worker died: {worker.fail_reason}"
            )
        )
    effective_abort_reason = abort_reason
    if effective_abort_reason is None and cleanup_errors:
        first = cleanup_errors[0]
        effective_abort_reason = f"{type(first).__name__}: {first}"

    if main_log is None:
        recorder_errors.append(
            RuntimeError("[wbc_policy_rollout] main episode recorder is unavailable")
        )
    else:
        main_active = bool(main_log.recording)
        io_active = bool(io_log.recording)
        if main_active != io_active:
            for recorder in (main_log, io_log):
                if recorder.recording:
                    run_cleanup_step(recorder.discard)
            recorder_errors.append(
                RuntimeError(
                    "[wbc_policy_rollout] recorder activity is asymmetric; discarded pair"
                )
            )
        elif main_active:
            main_nonempty = main_log.num_frames() > 0
            io_nonempty = io_log.num_frames() > 0
            if not main_nonempty or not io_nonempty:
                run_cleanup_step(main_log.discard)
                run_cleanup_step(io_log.discard)
                if main_nonempty != io_nonempty:
                    recorder_errors.append(
                        RuntimeError(
                            "[wbc_policy_rollout] asymmetric recorder frames; discarded pair"
                        )
                    )
            elif main_log.episode_id != io_log.episode_id:
                run_cleanup_step(main_log.discard)
                run_cleanup_step(io_log.discard)
                recorder_errors.append(
                    RuntimeError(
                        "[wbc_policy_rollout] recorder episode ids diverged before save: "
                        f"main={main_log.episode_id}, policy_io={io_log.episode_id}"
                    )
                )
            else:
                saved_id = int(main_log.episode_id)
                paths: list[str] = []
                for name, recorder in (("main", main_log), ("policy IO", io_log)):
                    path: str | None = None
                    while recorder.recording:
                        try:
                            if effective_abort_reason is None:
                                action = recorder.stop
                            else:
                                abort = getattr(recorder, "abort", None)
                                if not callable(abort):
                                    raise RuntimeError(
                                        f"{type(recorder).__name__} has no abort() "
                                        "quarantine API"
                                    )
                                action = lambda abort=abort: abort(effective_abort_reason)
                            path, terminal_interrupted = _call_recorder_terminal(action)
                            interrupted = terminal_interrupted or interrupted
                        except KeyboardInterrupt:
                            interrupted = True
                            continue
                        except BaseException as exc:
                            error = RuntimeError(
                                f"[wbc_policy_rollout] {name} recorder "
                                f"{'stop' if effective_abort_reason is None else 'abort'} "
                                "failed: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            cleanup_errors.append(error)
                            recorder_errors.append(error)
                            if (
                                effective_abort_reason is not None
                                and getattr(recorder, "recording", False)
                            ):
                                # Prevent the outer HardwareDriver.close() from falling
                                # through to a normal complete save after quarantine
                                # itself failed.
                                try:
                                    recorder.discard()
                                except BaseException as discard_exc:
                                    discard_error = RuntimeError(
                                        f"[wbc_policy_rollout] {name} fallback discard "
                                        f"failed: {type(discard_exc).__name__}: "
                                        f"{discard_exc}"
                                    )
                                    cleanup_errors.append(discard_error)
                                    recorder_errors.append(discard_error)
                        break
                    if path is not None:
                        paths.append(path)

                for name, recorder in (("main", main_log), ("policy IO", io_log)):
                    try:
                        interrupted = _wait_for_recorder_save(recorder) or interrupted
                    except BaseException as exc:
                        error = RuntimeError(
                            f"[wbc_policy_rollout] {name} recorder save wait failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        cleanup_errors.append(error)
                        recorder_errors.append(error)

                for name, recorder in (("main", main_log), ("policy IO", io_log)):
                    save_error = getattr(recorder, "last_save_error", None)
                    if save_error is not None:
                        error = RuntimeError(
                            f"[wbc_policy_rollout] {name} save failed: {save_error}"
                        )
                        cleanup_errors.append(error)
                        recorder_errors.append(error)

                if recorder_errors:
                    for path in paths:
                        Path(path).unlink(missing_ok=True)
                else:
                    if paths:
                        disposition = (
                            "saved"
                            if effective_abort_reason is None
                            else "quarantined incomplete"
                        )
                        print(
                            f"[wbc_policy_rollout] episode pair {disposition} -> "
                            f"{paths}"
                        )
                    traj = getattr(driver, "_traj", None)
                    if traj is not None and effective_abort_reason is None:
                        run_cleanup_step(lambda: traj.flush(saved_id))

    cleanup_errors.extend(
        error for error in recorder_errors if error not in cleanup_errors
    )
    if cleanup_errors:
        primary = cleanup_errors[0]
        for extra in cleanup_errors[1:]:
            primary.add_note(f"additional cleanup failure: {type(extra).__name__}: {extra}")
        raise primary
    if interrupted:
        raise KeyboardInterrupt


def _run_rollout_episode(
    args: argparse.Namespace,
    *,
    policy,
    source: RecordedEpisodeSource | None,
    state_frame: str,
    action_target_frame: str = "world",
    mode: str,
    ik,
    fk: WBCPolicyFK,
    driver,
    enable: dict,
    io_log,
    nominal_poses: tuple[np.ndarray, np.ndarray, np.ndarray],
    worker_factory=None,
    prompt_for_engage: bool,
) -> None:
    """Prepare, execute, and fully save exactly one rollout episode."""
    from omniteleop.wbc_stream import (  # noqa: PLC0415
        HeadTargetLowPassFilter,
        HeadTargetPlanarDeadbandFilter,
        TargetInterpolator,
    )

    schedule_buffer = ActionScheduleBuffer()
    stop_event = threading.Event()
    worker: _InferenceWorker | None = None
    left0, right0, head0 = (pose.copy() for pose in nominal_poses)
    try:
        dt = 1.0 / args.ik_rate
        cmd_period = 1.0 / args.cmd_rate
        interp = TargetInterpolator(cmd_period, left0, right0, head0)
        head_lpf = HeadTargetLowPassFilter(args.head_lpf_tau, head0)
        head_deadband = HeadTargetPlanarDeadbandFilter(
            head0,
            position_deadband=args.head_planar_pos_deadband,
            yaw_deadband=args.head_planar_yaw_deadband,
        )
        if prompt_for_engage:
            input(
                f"[wbc_policy_rollout] robot homed. Press Enter to ENGAGE the {mode} "
                "(Ctrl-C ends this rollout) ... "
            )

        # Engage: identical world-frame re-anchoring to the teleop follower. The
        # odometry origin is zeroed HERE, before any alignment glide -- matching
        # how a recorded episode's world frame was anchored at ITS engage edge
        # (the leader's align stage also runs after engage, before recording).
        ik.reset()
        interp.reset(left0, right0, head0)
        head_lpf.reset(head0)
        head_deadband.reset(head0)
        driver.engage_reset(ik, left0, right0, head0)
        latch_chassis = getattr(driver, "latch_chassis_intent", None)
        if callable(latch_chassis):
            latch_chassis(np.zeros(3, dtype=np.float32))
        driver._overstep_ticks = 0  # noqa: SLF001 -- per-episode safety history
        if policy is not None:
            policy.reset()

        # Wait for the first FC03 gripper replies so the state is finite.
        wait_t0 = time.perf_counter()
        while True:
            driver._poll_gripper_status_step()  # noqa: SLF001
            if np.isfinite(driver._last_obs_grip_left) and np.isfinite(  # noqa: SLF001
                    driver._last_obs_grip_right):  # noqa: SLF001
                break
            if time.perf_counter() - wait_t0 > 5.0:
                raise SystemExit("[wbc_policy_rollout] no FC03 gripper status within 5 s")
            time.sleep(0.05)

        # Reference alignment: glide the arms to the shared start pose, then hand
        # control to a human Enter-press. Runs BEFORE recording starts, so the
        # glide is never part of the take (mirrors the leader's align stage).
        start_left, start_right, start_head = left0, right0, head0
        did_align = False
        if args.align_reference:
            reference_frame = (
                "base" if action_target_frame == "current_base" else "world"
            )
            left_ref, right_ref = load_alignment_references(
                args.align_reference, ik, target_frame=reference_frame
            )
            print(f"[wbc_policy_rollout] gliding to the reference pose from "
                  f"{args.align_reference} ...")
            run_reference_alignment(
                ik=ik, driver=driver, enable=enable, interp=interp,
                head_lpf=head_lpf, head_deadband=head_deadband,
                left_ref=left_ref, right_ref=right_ref, head_current=head0, dt=dt,
                read_achieved=lambda: _achieved_ee_poses(
                    driver, fk, target_frame=action_target_frame
                ),
            )
            start_left, start_right = left_ref, right_ref
            # No 100 Hz ticks run while blocked on Enter: stop/hold everything so
            # no chassis twist or in-flight position target survives the prompt.
            driver.stop_all_motion()
            did_align = True

        # Checkpoint-driven live object condition: bootstrap after the robot reaches the
        # rollout start pose, arrange all representations once, and validate before recording.
        # Held safe during the minutes-long SceneDiff subprocess (no 100 Hz ticks run while it
        # blocks). The subprocess owns the GPU; the loaded diffusion policy sits idle.
        # The single arranged condition is installed after the earlier policy.reset().
        _prepare_rollout_episode_condition(args, driver, fk, policy, io_log)

        if did_align:
            driver.stop_all_motion()
            input(f"\n[wbc_policy_rollout] at reference pose. Press Enter to actually "
                  f"start the {mode} online ... ")

        if policy is not None:
            policy.start_episode_runtime(
                max_seconds=args.max_seconds,
                policy_interval=args.policy_interval,
                dataset_dt=1.0 / args.dataset_fps,
            )
        _set_policy_io_static(io_log, policy, args=args)

        _start_rollout_recorders(driver, io_log)

        print(f"[wbc_policy_rollout] engaged: {mode} over "
              f"WBC {args.ik_rate:g} Hz; recording -> {args.save_dir}")
        t0 = time.perf_counter()
        now = t0
        next_cmd_t = now
        if source is not None:
            source.start(t0)
        else:
            # Start inference only now: observations must be post-engage/post-align
            # (the odometry origin and ik.reset() world anchor are already set).
            worker = (worker_factory or _InferenceWorker)(
                policy=policy, driver=driver, fk=fk, buffer=schedule_buffer,
                io_log=io_log, t0=t0, dataset_dt=1.0 / args.dataset_fps,
                policy_interval=args.policy_interval,
                execution_latency=args.execution_latency, stop_event=stop_event,
            )
            worker.start()
        replay_end_wall: float | None = None
        last_cmd_wall: float | None = None
        last_source_timestamp_ns = -1
        last_source_receive_wall_ns = -1
        grip_l = np.float32(0.0)
        grip_r = np.float32(0.0)
        ticks = 0
        while True:
            now = time.perf_counter()
            if worker is not None and worker.failed.is_set():
                raise RuntimeError(
                    f"[wbc_policy_rollout] inference worker died: {worker.fail_reason}"
                )
            if args.max_seconds > 0 and now - t0 >= args.max_seconds:
                print(f"\n[wbc_policy_rollout] --max-seconds {args.max_seconds:g} reached.")
                break

            if source is not None:
                frame = source.advance(now)
                if frame is not None:
                    if callable(latch_chassis):
                        if frame.chassis is None:
                            raise RuntimeError(
                                "joystick replay frame is missing chassis intent"
                            )
                        latch_chassis(frame.chassis)
                    # Same observation assembly as the live branch: exercises the
                    # full camera/state pipeline during replay and yields a
                    # policy_io log directly diffable against a live rollout.
                    head_imgs = driver._grab_head_images()  # noqa: SLF001
                    # Drain every wrist stream the driver tracks (keys are the arms).
                    for _arm in driver._last_rec_wrist_ns:  # noqa: SLF001
                        driver._grab_wrist_image(_arm)  # noqa: SLF001
                    if head_imgs is None:
                        raise RuntimeError("head camera delivered no frame at the replay tick")
                    if frame.index == 0:
                        # Frame 0 is a snap (duration 0): require it to start near
                        # the robot's current pose (the aligned reference) so the
                        # snap is a residual, never a jump. Fail closed otherwise.
                        from omniteleop.leader.wbc_reference_alignment import (  # noqa: PLC0415
                            pose_rotation_error_deg,
                        )

                        for name, tgt, cur in (("left", frame.left, start_left),
                                               ("right", frame.right, start_right),
                                               ("head", frame.head, start_head)):
                            dist = float(np.linalg.norm(tgt[:3, 3] - cur[:3, 3]))
                            rot = pose_rotation_error_deg(cur, tgt)
                            if dist > REPLAY_START_POS_TOL_M or rot > REPLAY_START_ROT_TOL_DEG:
                                raise RuntimeError(
                                    f"replay frame 0 {name} target is {dist * 1000:.0f} mm / "
                                    f"{rot:.0f} deg from the current pose (tol "
                                    f"{REPLAY_START_POS_TOL_M * 1000:.0f} mm / "
                                    f"{REPLAY_START_ROT_TOL_DEG:g} deg) -- align to "
                                    "THIS episode's reference before replaying"
                                )
                    state = _build_state(driver, fk, state_frame)
                    interp.push(frame.left, frame.right, frame.head,
                                now=now, duration=frame.segment_duration)
                    grip_l, grip_r = frame.grip_left, frame.grip_right
                    last_cmd_wall = now
                    last_source_timestamp_ns = time.time_ns()
                    last_source_receive_wall_ns = last_source_timestamp_ns
                    io_log.record({
                        "t": np.float64(now - t0),
                        "timestamp_ns": np.int64(time.time_ns()),
                        "state": state,
                        "action": encode_replay_action(
                            frame.left, frame.right, frame.head,
                            frame.grip_left, frame.grip_right,
                            chassis=frame.chassis),
                        "target": {
                            "left": frame.left.astype(np.float32),
                            "right": frame.right.astype(np.float32),
                            "head": frame.head.astype(np.float32),
                        },
                        "base_pose": np.asarray(state[29:32], np.float32),
                        "inference_s": np.float32(0.0),  # no model ran
                    })
                    if source.done:
                        # Let the last glide finish before stopping; at least one
                        # tick must still run (a 1-frame episode has duration 0).
                        replay_end_wall = now + max(frame.segment_duration, dt)
                if replay_end_wall is not None and now >= replay_end_wall:
                    print(f"\n[wbc_policy_rollout] replay finished: {source.n_frames} "
                          f"frames in {now - t0:.1f}s.")
                    break
            elif now >= next_cmd_t:
                # 10 Hz command sampler: NO camera grab, state build, or GPU work
                # on this thread -- the inference worker owns those. last_cmd_wall
                # advances ONLY when the buffer yields a payload, so the untouched
                # compute_hold_reason watchdog covers a dead/stalled worker
                # (awaiting-command before the first chunk, stale-source once the
                # trajectory is exhausted past --source-timeout). `now` is fresh at
                # push time by construction (the old inline path backdated the
                # interpolator segment by the inference duration).
                scheduled = schedule_buffer.sample(now)
                if scheduled is not None:
                    if callable(latch_chassis):
                        if scheduled.chassis is None:
                            raise RuntimeError(
                                "32-D joystick rollout received a 29-D scheduled action"
                            )
                        # The policy emits a 10 Hz intent. Keep it piecewise constant;
                        # the driver's causal 100 Hz shaper supplies slew/PD/dispatch.
                        latch_chassis(scheduled.chassis)
                    interp.push(scheduled.left, scheduled.right, scheduled.head,
                                now=now, duration=cmd_period)
                    grip_l = scheduled.grip_left
                    grip_r = scheduled.grip_right
                    last_cmd_wall = now
                    last_source_timestamp_ns = time.time_ns()
                    last_source_receive_wall_ns = last_source_timestamp_ns
                next_cmd_t += cmd_period
                if now > next_cmd_t:
                    next_cmd_t = now + cmd_period

            # Policy/replay head targets skip the stateful filters: action/head is
            # already post-LPF/post-deadband (see wbc_tick's docstring).
            result, hold, hold_reason = wbc_tick(
                ik=ik, driver=driver, enable=enable, interp=interp,
                head_lpf=head_lpf, head_deadband=head_deadband, now=now, dt=dt,
                grip_left=grip_l, grip_right=grip_r, last_cmd_wall=last_cmd_wall,
                live_head_filters=False,
                source_timestamp_ns=last_source_timestamp_ns,
                source_receive_wall_ns=last_source_receive_wall_ns,
            )

            ticks += 1
            if ticks % 50 == 0:
                state_str = f"HOLD:{hold_reason}" if hold_reason else ("hold" if hold else "run")
                progress = (f"frame {source.released}/{source.n_frames}  "
                            if source is not None else "")
                print(f"t={now - t0:6.1f}s  {state_str:12s} {progress}"
                      f"errL={result.left_ee_error * 1000:5.1f}mm "
                      f"errR={result.right_ee_error * 1000:5.1f}mm  "
                      f"{result.safety_status}", end="\r")
            sleep = dt - (time.perf_counter() - now)
            if sleep > 0:
                time.sleep(sleep)
    except BaseException as body_error:
        try:
            _finish_rollout_episode(
                driver=driver,
                io_log=io_log,
                stop_event=stop_event,
                worker=worker,
                policy=policy,
                abort_reason=(
                    None
                    if isinstance(body_error, KeyboardInterrupt)
                    else f"{type(body_error).__name__}: {body_error}"
                ),
            )
        except BaseException as cleanup_error:
            if isinstance(body_error, KeyboardInterrupt):
                raise
            raise body_error from cleanup_error
        raise
    else:
        _finish_rollout_episode(
            driver=driver,
            io_log=io_log,
            stop_event=stop_event,
            worker=worker,
            policy=policy,
        )


def _run_session_with_close(
    *,
    run_session: Callable[[], None],
    close: Callable[[], None],
) -> None:
    """Run one persistent session without letting close replace its primary error."""
    try:
        run_session()
    except BaseException as session_error:
        try:
            close()
        except BaseException as close_error:
            raise session_error from close_error
        raise
    else:
        close()


def _run_rollout(args: argparse.Namespace, *, policy_factory=None,
                 worker_factory=None) -> None:
    """Drive the 100 Hz WBC loop from a policy (or a recorded episode).

    ``policy_factory(policy_path)`` and ``worker_factory(**kwargs)`` default to
    :class:`_PolicyBundle` / :class:`_InferenceWorker` (LeRobot image policies).
    ``scripts/wbc_maniflow_rollout.py`` swaps in a ManiFlow point-cloud pair so both
    policy families share ONE hardware loop, alignment, scheduler and watchdog -- a
    second copy of this loop is exactly the kind of divergence that gets a robot hurt.
    A replacement bundle must expose ``describe``/``reset``/``predict_chunk``, the
    scheduling dimensions, ``scene_requirements``, and the live-condition/runtime hooks.
    """
    from omniteleop.common.recorder import EpisodeRecorder  # noqa: PLC0415
    from omniteleop.wbc_robot_util import parse_enable_mask  # noqa: PLC0415

    mod = _load_wbc_vr_robot()

    enable = parse_enable_mask(args.enable)
    missing = [g for g in ("torso", "arms", "head", "base") if not enable.get(g)]
    if missing:
        raise SystemExit(
            f"[wbc_policy_rollout] rollout/replay requires --enable torso,arms,head,base "
            f"(missing: {', '.join(missing)}): the action commands the whole body and "
            "the state needs odometry"
        )

    source: RecordedEpisodeSource | None = None
    policy: _PolicyBundle | None = None
    if args.replay_episode:
        source = RecordedEpisodeSource(args.replay_episode)
        state_frame = "base"
        mode = "replay"
        print(f"[wbc_policy_rollout] replaying {args.replay_episode}: "
              f"{source.n_frames} frames over {source.duration_s:.1f}s (real-time)")
    else:
        mode = "rollout"
        print(f"[wbc_policy_rollout] loading policy {args.policy_path} ...")
        policy = (
            policy_factory(args.policy_path)
            if policy_factory is not None
            else _PolicyBundle(
                args.policy_path,
                scene_diff_repo=args.scene_diff_repo,
                dataset_meta_path=getattr(args, "policy_dataset_meta", None),
            )
        )
        state_frame = policy.state_frame
        print(f"[wbc_policy_rollout] {policy.describe()}")
        try:
            action_timing = _resolve_rollout_action_timing(
                policy=policy,
                runtime_dataset_fps=float(args.dataset_fps),
                cli_action_offset_frames=getattr(
                    args, "policy_action_offset_frames", None
                ),
            )
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"[wbc_policy_rollout] {exc}; refusing to engage") from exc
        print(
            "[wbc_policy_rollout] action timing contract: "
            f"{action_timing['dataset_fps']:g} fps, labels/schedule +"
            f"{action_timing['action_offset_frames']} frame(s) "
            f"({action_timing['action_offset_frames'] / action_timing['dataset_fps']:.3f}s)"
        )
        chunk_coverage = policy.n_action_steps / args.dataset_fps
        if args.policy_interval > chunk_coverage:
            print(f"[wbc_policy_rollout] WARNING: --policy-interval "
                  f"{args.policy_interval:g}s exceeds the chunk coverage "
                  f"{chunk_coverage:g}s ({policy.n_action_steps} steps @ "
                  f"{args.dataset_fps:g} fps): the scheduled buffer will run dry "
                  "between replans -> periodic stale-source holds")

    declared_policy_schema = str(
        getattr(
            policy if policy is not None else source,
            "policy_action_schema",
            WBC_POLICY_ACTION_SCHEMA,
        )
    )
    try:
        policy_action_axes = action_axes_for_policy_schema(declared_policy_schema)
    except ValueError as exc:
        raise SystemExit(f"[wbc_policy_rollout] {exc}; refusing to engage") from exc
    expected_action_dim = len(policy_action_axes)
    declared_action_dim = getattr(policy, "action_dim", expected_action_dim)
    if int(declared_action_dim) != expected_action_dim:
        raise SystemExit(
            "[wbc_policy_rollout] checkpoint action dimension contradicts its schema: "
            f"{declared_action_dim} vs {declared_policy_schema} ({expected_action_dim})"
        )
    expected_target_frame = (
        "current_base"
        if declared_policy_schema == JOYSTICK_POLICY_ACTION_SCHEMA
        else "world"
    )
    declared_target_frame = str(
        getattr(
            policy if policy is not None else source,
            "action_target_frame",
            expected_target_frame,
        )
    )
    if declared_target_frame != expected_target_frame:
        raise SystemExit(
            "[wbc_policy_rollout] policy action frame contradicts its schema: "
            f"{declared_target_frame!r} vs {declared_policy_schema} "
            f"({expected_target_frame!r})"
        )
    joystick_policy = declared_policy_schema == JOYSTICK_POLICY_ACTION_SCHEMA
    args.policy_action_schema = declared_policy_schema
    args.action_target_frame = expected_target_frame

    lock_torso = getattr(args, "lock_torso_in_ik", None)
    if joystick_policy:
        joystick_mod = _load_wbc_joystick_robot()
        ik, cfg = joystick_mod._build_joystick_ik(  # noqa: SLF001
            getattr(args, "config", None), lock_torso_in_ik=lock_torso
        )
        driver_factory = lambda: joystick_mod.JoystickHardwareDriver(
            args, ik, cfg, enable, source=None
        )
        if cfg.head_mode != "track" or not cfg.lock_base_in_ik:
            raise SystemExit(
                "[wbc_policy_rollout] joystick policy requires head_mode=track and "
                "lock_base_in_ik=true"
            )
    else:
        ik, cfg = (
            mod._build_ik()  # noqa: SLF001
            if lock_torso is None
            else mod._build_ik(lock_torso_in_ik=lock_torso)  # noqa: SLF001
        )
        driver_factory = lambda: mod.HardwareDriver(args, ik, cfg, enable)
        if cfg.head_mode != "ik":
            raise SystemExit(
                "[wbc_policy_rollout] 29-D WBC policy requires wbik.yaml "
                "head_mode='ik'"
            )
    print(
        f"[wbc_policy_rollout] model nq={ik.model.nq} head_mode={cfg.head_mode} "
        f"action={declared_policy_schema} ({expected_action_dim}-D, "
        f"{expected_target_frame})"
    )

    try:
        base_pose_source, runtime_base_pose_source = _resolve_rollout_base_pose_source(
            arkit_mode=getattr(args, "arkit_base", "off"),
            policy=policy,
            replay_source=source,
            cli_source=getattr(args, "policy_base_pose_source", None),
        )
    except ValueError as exc:
        raise SystemExit(f"[wbc_policy_rollout] {exc}") from exc
    print(
        f"[wbc_policy_rollout] base pose contract: training/replay={base_pose_source}, "
        f"live control={runtime_base_pose_source}"
    )
    if policy is not None:
        try:
            camera_alignment, runtime_camera_alignment = (
                _resolve_rollout_camera_alignment(
                    policy=policy,
                    runtime_max_skew_ms=float(
                        getattr(
                            args,
                            "record_max_camera_skew_ms",
                            getattr(mod, "DEFAULT_RECORD_MAX_CAMERA_SKEW_MS", 40.0),
                        )
                    ),
                    runtime_max_camera_age_ms=float(
                        getattr(
                            args,
                            "record_max_camera_age_ms",
                            getattr(mod, "DEFAULT_RECORD_MAX_CAMERA_AGE_MS", 150.0),
                        )
                    ),
                    cli_mode=getattr(args, "policy_camera_alignment_mode", None),
                    cli_max_skew_ms=getattr(args, "policy_camera_max_skew_ms", None),
                    cli_max_camera_age_ms=getattr(
                        args, "policy_camera_max_age_ms", None
                    ),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"[wbc_policy_rollout] {exc}") from exc
        if camera_alignment is None:
            print(
                "[wbc_policy_rollout] camera alignment contract: head-only policy; "
                "multi-camera parity not applicable"
            )
        else:
            print(
                "[wbc_policy_rollout] camera alignment contract: "
                f"training/live={camera_alignment['mode']} <= "
                f"{camera_alignment['max_abs_skew_ns'] / 1e6:g} ms skew, "
                f"{camera_alignment['max_camera_age_ns'] / 1e6:g} ms age "
                f"[{runtime_camera_alignment['clock_domain']}]"
            )

    fk = WBCPolicyFK(ik=ik)  # FK on the live solver's own model
    driver = driver_factory()
    signal.signal(signal.SIGINT, signal.default_int_handler)

    def run_initialized_session() -> None:
        actual_source_fn = getattr(driver, "base_pose_source", None)
        if callable(actual_source_fn):
            actual_source = str(actual_source_fn())
            if actual_source != base_pose_source:
                raise RuntimeError(
                    "hardware driver base-pose source changed after validation: "
                    f"expected {base_pose_source}, got {actual_source}"
                )
        io_log = EpisodeRecorder(str(Path(args.save_dir) / "policy_io"))
        if policy is not None:
            # Fail metadata/provenance construction before any episode prompt. The
            # episode-specific condition tree is added again immediately before start().
            metadata_fn = getattr(policy, "inference_metadata", None)
            if metadata_fn is not None:
                metadata_fn()
        nominal_poses = (
            np.asarray(ik.frame_pose(fk.left_ee_frame).homogeneous, dtype=float),
            np.asarray(ik.frame_pose(fk.right_ee_frame).homogeneous, dtype=float),
            np.asarray(ik.frame_pose(fk.head_frame).homogeneous, dtype=float),
        )

        def run_episode(prompt_for_engage: bool) -> None:
            _run_rollout_episode(
                args,
                policy=policy,
                source=source,
                state_frame=state_frame,
                action_target_frame=expected_target_frame,
                mode=mode,
                ik=ik,
                fk=fk,
                driver=driver,
                enable=enable,
                io_log=io_log,
                nominal_poses=nominal_poses,
                worker_factory=worker_factory,
                prompt_for_engage=prompt_for_engage,
            )

        _run_persistent_session(
            run_episode=run_episode,
            home_to_nominal=driver.home_to_nominal,
            stop_all_motion=driver.stop_all_motion,
            auto_start=args.auto_start,
        )

    _run_session_with_close(run_session=run_initialized_session, close=driver.close)


def build_parser() -> argparse.ArgumentParser:
    """Every CLI flag of the shared rollout. ``wbc_maniflow_rollout.py`` extends this."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy-path", default=None,
                        help="LeRobot checkpoint dir trained on the WBC 32/29 schema "
                             "(exactly one of --policy-path / --replay-episode).")
    parser.add_argument(
        "--policy-dataset-meta",
        default=None,
        help="ported training dataset's dexmate_meta.json (or its directory). "
             "Normally auto-discovered through the checkpoint train_config.json; "
             "use this if the dataset moved.",
    )
    parser.add_argument(
        "--policy-action-offset-frames",
        type=int,
        default=None,
        help="legacy-checkpoint fallback declaring the coherent future action-label "
             "offset used by the porter. New checkpoints auto-read it from "
             "--policy-dataset-meta; use 0 only for a confirmed zero-offset dataset.",
    )
    parser.add_argument(
        "--policy-base-pose-source",
        choices=tuple(sorted(_BASE_POSE_SOURCES)),
        default=None,
        help="legacy-checkpoint fallback declaring whether state[29:32] was trained "
             "from wheel odometry or ARKit. New checkpoints auto-read this from "
             "--policy-dataset-meta and reject a contradictory value.",
    )
    parser.add_argument(
        "--policy-camera-alignment-mode",
        choices=("head_capture_nearest", "latest_arrived_compatibility"),
        default=None,
        help="legacy-checkpoint fallback declaring how training selected wrist frames. "
             "New checkpoints auto-read this from --policy-dataset-meta.",
    )
    parser.add_argument(
        "--policy-camera-max-skew-ms",
        type=float,
        default=None,
        help="legacy-checkpoint training skew bound. Required with "
             "--policy-camera-alignment-mode head_capture_nearest; must match the "
             "live --record-max-camera-skew-ms exactly.",
    )
    parser.add_argument(
        "--policy-camera-max-age-ms",
        type=float,
        default=None,
        help="legacy-checkpoint training capture-age ceiling. Required with "
             "--policy-camera-alignment-mode and must match the live "
             "--record-max-camera-age-ms exactly.",
    )
    parser.add_argument("--replay-episode", default=None,
                        help="recorded episode_*.hdf5 (wbc_vr_robot.py --record schema) "
                             "to replay verbatim at real time through the same "
                             "actuation path (exactly one of --policy-path / "
                             "--replay-episode).")
    parser.add_argument("--align-reference", default=None,
                        help="reference episode HDF5: after engage, glide the arms to "
                             "its final-frame EEF poses and wait for Enter before the "
                             "take starts. Required in both modes; pass 'none' to "
                             "deliberately start from the current pose instead.")
    parser.add_argument("--enable", default="arms,torso,head,base",
                        help="DOF groups to actuate (rollout requires all four; the "
                             "grippers are always active).")
    parser.add_argument(
        "--lock-torso-in-ik",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override follower/wbik.yaml lock_torso_in_ik for either WBC or "
             "joystick policy execution; omit to use the YAML value.",
    )
    parser.add_argument("--namespace", default="",
                        help="Zenoh namespace (matches wbc_vr_robot; default empty).")
    parser.add_argument("--save-dir", default=None,
                        help=f"episode + policy_io logs (default "
                             f"{DEFAULT_ROLLOUT_SAVE_DIR} with --policy-path, "
                             f"{DEFAULT_REPLAY_SAVE_DIR} with --replay-episode).")
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="skip the Enter-to-engage prompt for the first rollout only",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="end each rollout after this many seconds (0 = run until Ctrl-C)",
    )
    parser.add_argument("--policy-interval", type=float, default=0.4,
                        help="seconds between inference-thread replans (default 0.4: "
                             "replans halfway through an 8-step 10 Hz chunk; a chunk "
                             "covers n_action_steps/--dataset-fps seconds).")
    parser.add_argument("--execution-latency", type=float, default=0.0,
                        help="stale-action cutoff offset: chunk frames scheduled at or "
                             "before inference-end + this many seconds are dropped, "
                             "never time-shifted (default 0.0).")
    parser.add_argument("--dataset-fps", type=float, default=10.0,
                        help="fps the TRAINING dataset was ported at (sets the "
                             "synthesized chunk timestep and the observation-history "
                             "spacing; default 10).")
    mod = _load_wbc_vr_robot()
    hw = parser.add_argument_group("hardware (defaults match wbc_vr_robot.py)")
    hw.add_argument("--home-tol", type=float, default=mod.DEFAULT_HOME_TOL)
    hw.add_argument("--home-settle", type=float, default=mod.DEFAULT_HOME_SETTLE)
    hw.add_argument("--max-joint-step", type=float, default=mod.DEFAULT_MAX_JOINT_STEP)
    hw.add_argument("--source-timeout", type=float, default=mod.DEFAULT_SOURCE_TIMEOUT)
    hw.add_argument("--drive-state-mode", choices=("ms", "rad"), default="ms")
    hw.add_argument("--base-quiet-hold-s", type=float,
                    default=mod.DEFAULT_BASE_QUIET_HOLD_S)
    hw.add_argument("--record-rate", type=float, default=mod.DEFAULT_RECORD_RATE)
    hw.add_argument("--record-stale-grace", type=float,
                    default=mod.DEFAULT_RECORD_STALE_GRACE)
    hw.add_argument(
        "--record-startup-timeout",
        type=float,
        default=mod.DEFAULT_RECORD_STARTUP_TIMEOUT,
        help="startup-only camera stability deadline before frame zero (default "
             "matches wbc_vr_robot.py)",
    )
    hw.add_argument(
        "--record-max-camera-skew-ms",
        type=float,
        default=mod.DEFAULT_RECORD_MAX_CAMERA_SKEW_MS,
        help="maximum corrected head-to-each-wrist capture skew in recorded rollout "
             "rows (default matches wbc_vr_robot.py)",
    )
    hw.add_argument(
        "--record-max-camera-age-ms",
        type=float,
        default=mod.DEFAULT_RECORD_MAX_CAMERA_AGE_MS,
        help="maximum corrected capture-to-selection age for recording and live "
             "policy observations (default matches wbc_vr_robot.py)",
    )
    hw.add_argument(
        "--streaming-recorder",
        dest="streaming_recorder",
        action="store_true",
        default=True,
        help="stream rollout episodes to HDF5 with bounded RAM (default on)",
    )
    hw.add_argument(
        "--buffered-recorder",
        dest="streaming_recorder",
        action="store_false",
        help="use the legacy in-RAM episode recorder",
    )
    hw.add_argument(
        "--no-head-depth",
        dest="no_head_depth",
        action="store_true",
        default=True,
        help="do not subscribe to ZED SDK depth while rolling out (default on)",
    )
    hw.add_argument(
        "--head-depth",
        dest="no_head_depth",
        action="store_false",
        help="record legacy ZED SDK head depth",
    )
    hw.add_argument(
        "--head-right-rgb",
        action="store_true",
        help="also record the rectified head right eye for offline stereo",
    )
    hw.add_argument(
        "--wrist-right-rgb",
        action="store_true",
        help="also record the rectified right eye of both wrist cameras",
    )
    hw.add_argument(
        "--arkit-base",
        choices=("off", "record", "control"),
        default="off",
        help="iPhone base tracking mode passed through to the shared hardware driver",
    )

    pc = parser.add_argument_group(
        "checkpoint-driven live object bootstrap (--policy-path only)"
    )
    pc.add_argument("--reference-hdf5", default=DEFAULT_SCENE_REFERENCE,
                    help="fixed 'after'/moved reference scene the live frame is diffed against "
                         "(the SAME scene training used; default the Part-A reference_last.hdf5).")
    pc.add_argument("--prompt-before-capture", action="store_true",
                    help="pause for ENTER at the rollout start pose and before the head capture "
                         "(post-reference-alignment when --align-reference is supplied).")
    pc.add_argument(
        "--sam", default="sam3",
        help="SAM version for change detection (default sam3).",
    )
    pc.add_argument("--scenediff-config", default=None,
                    help="override the SceneDiff config path (relative to the scene_diff repo).")
    pc.add_argument("--scenediff-timeout", type=float, default=1200.0,
                    help="wall-clock timeout (s) for the SceneDiff subprocess (default 1200).")
    pc.add_argument("--scene-diff-repo", default=DEFAULT_SCENE_DIFF_REPO,
                    help="scene_diff repo holding run_live_pos_condition.sh.")
    pc.add_argument("--scene-diff-python", default=DEFAULT_SCENE_DIFF_PYTHON,
                    help="interpreter for SceneDiff (scene_diff/.venv-merged/bin/python).")
    return parser


def finalize_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate the parsed args and fill the wbik.yaml-sourced hardware defaults in place."""
    mod = _load_wbc_vr_robot()

    if bool(args.policy_path) == bool(args.replay_episode):
        parser.error("exactly one of --policy-path / --replay-episode is required")
    if args.replay_episode and (
        args.policy_dataset_meta is not None
        or args.policy_base_pose_source is not None
        or args.policy_camera_alignment_mode is not None
        or args.policy_camera_max_skew_ms is not None
        or args.policy_camera_max_age_ms is not None
    ):
        parser.error(
            "--policy-dataset-meta and policy contract fallbacks apply only with "
            "--policy-path; replay reads its source from the episode"
        )
    if args.policy_path and args.policy_base_pose_source is not None:
        runtime_source = _runtime_base_pose_source(args.arkit_base)
        if args.policy_base_pose_source != runtime_source:
            parser.error(
                f"--policy-base-pose-source {args.policy_base_pose_source} conflicts "
                f"with --arkit-base {args.arkit_base} (live source {runtime_source})"
            )
    if args.policy_camera_alignment_mode is None:
        if (
            args.policy_camera_max_skew_ms is not None
            or args.policy_camera_max_age_ms is not None
        ):
            parser.error(
                "--policy-camera-max-skew-ms/--policy-camera-max-age-ms require "
                "--policy-camera-alignment-mode"
            )
    else:
        if (
            args.policy_camera_max_age_ms is None
            or not np.isfinite(args.policy_camera_max_age_ms)
            or args.policy_camera_max_age_ms <= 0
        ):
            parser.error(
                "--policy-camera-alignment-mode requires finite "
                "--policy-camera-max-age-ms > 0"
            )
    if args.policy_camera_alignment_mode == "head_capture_nearest":
        if (
            args.policy_camera_max_skew_ms is None
            or not np.isfinite(args.policy_camera_max_skew_ms)
            or args.policy_camera_max_skew_ms <= 0
        ):
            parser.error(
                "head_capture_nearest requires finite "
                "--policy-camera-max-skew-ms > 0"
            )
    elif (
        args.policy_camera_alignment_mode == "latest_arrived_compatibility"
        and args.policy_camera_max_skew_ms not in (None, 0.0)
    ):
        parser.error(
            "latest_arrived_compatibility requires "
            "--policy-camera-max-skew-ms 0 (or omit it)"
        )
    if args.policy_path and not np.isfinite(args.scenediff_timeout):
        parser.error("--scenediff-timeout must be finite")
    if args.align_reference is None:
        parser.error("--align-reference is required (pass '--align-reference none' to "
                     "deliberately start from the current pose)")
    from omniteleop.leader.wbc_reference_alignment import (  # noqa: PLC0415 -- heavy
        optional_reference_path,
    )
    # 'none'/'null'/'' -> None: alignment explicitly skipped.
    args.align_reference = optional_reference_path(args.align_reference)
    if args.save_dir is None:
        args.save_dir = (DEFAULT_REPLAY_SAVE_DIR if args.replay_episode
                         else DEFAULT_ROLLOUT_SAVE_DIR)

    # Both modes always record (episode + policy IO); the HardwareDriver builds the
    # cameras only under record=True and the observation pipeline needs them anyway.
    args.record = True
    args.replay = None
    args.debug_dir = None
    # Control-loop tunables sourced from wbik.yaml's vr_teleop: block, exactly like
    # wbc_vr_robot.main(): one namespace, YAML stays the single source of truth.
    args.ik_rate = mod.DEFAULT_IK_RATE
    args.cmd_rate = mod.DEFAULT_CMD_RATE
    args.head_lpf_tau = mod.DEFAULT_HEAD_LPF_TAU
    args.head_planar_pos_deadband = mod.DEFAULT_HEAD_PLANAR_POS_DEADBAND
    args.head_planar_yaw_deadband = mod.DEFAULT_HEAD_PLANAR_YAW_DEADBAND
    args.base_kp_xy = mod.DEFAULT_BASE_KP_XY
    args.base_kp_yaw = mod.DEFAULT_BASE_KP_YAW
    args.base_yaw_hold_in_xy = mod.DEFAULT_BASE_YAW_HOLD_IN_XY
    args.base_deadband = mod.DEFAULT_BASE_DEADBAND
    args.base_accel = mod.DEFAULT_BASE_ACCEL
    args.base_max_speed = mod.DEFAULT_BASE_MAX_SPEED
    args.base_post_linear_deadband = mod.DEFAULT_BASE_POST_LINEAR_DEADBAND
    args.base_post_angular_deadband = mod.DEFAULT_BASE_POST_ANGULAR_DEADBAND
    # Harmless for 29-D WBC rollouts and required by JoystickBaseShaper for 32-D
    # checkpoints. Read the same canonical YAML sections as joystick collection.
    from omniteleop.wbc_teleop import (  # noqa: PLC0415
        JoystickTeleopConfig,
        VRTeleopConfig,
        bind_joystick_teleop_args,
    )

    joystick_vr_cfg = VRTeleopConfig.from_yaml()
    joystick_cfg = JoystickTeleopConfig.from_yaml()
    bind_joystick_teleop_args(args, joystick_vr_cfg, joystick_cfg)

    if args.max_seconds < 0 or not np.isfinite(args.max_seconds):
        parser.error("--max-seconds must be finite and >= 0")
    if not np.isfinite(args.policy_interval) or args.policy_interval <= 0:
        parser.error("--policy-interval must be finite and > 0")
    if not np.isfinite(args.execution_latency) or args.execution_latency < 0:
        parser.error("--execution-latency must be finite and >= 0")
    if not np.isfinite(args.dataset_fps) or args.dataset_fps <= 0:
        parser.error("--dataset-fps must be finite and > 0")
    if (
        args.policy_action_offset_frames is not None
        and args.policy_action_offset_frames < 0
    ):
        parser.error("--policy-action-offset-frames must be >= 0")
    if (
        not np.isfinite(args.record_max_camera_skew_ms)
        or args.record_max_camera_skew_ms < 0
    ):
        parser.error("--record-max-camera-skew-ms must be finite and >= 0")
    if (
        not np.isfinite(args.record_max_camera_age_ms)
        or args.record_max_camera_age_ms <= 0
    ):
        parser.error("--record-max-camera-age-ms must be finite and > 0")
    if (
        not np.isfinite(args.record_startup_timeout)
        or args.record_startup_timeout <= 0
    ):
        parser.error("--record-startup-timeout must be finite and > 0")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    finalize_args(parser, args)
    _run_rollout(args)


if __name__ == "__main__":
    main()
