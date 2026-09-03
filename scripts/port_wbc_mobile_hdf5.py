#!/usr/bin/env python3
"""Port WBC mobile-manipulation raw HDF5 episodes to a LeRobotDataset.

Consumes the raw takes written by ``scripts/wbc_vr_robot.py --record`` (10 Hz,
EpisodeRecorder schema, flat ``<raw-dir>/episode_*.hdf5``) and emits the WBC
policy schema of ``omniteleop.wbc_policy_format`` (see PLAN.md Conventions):

  observation.state (32,) float32
      base-frame ACHIEVED left/right EEF pos3+rot6 (+ raw FC03 gripper) and
      zed_depth_frame head pos3+rot6, from WBC FK on the MEASURED ``obs/joint``
      with the base at zero -- computed on the SAME model the follower solves
      with (``WBCPolicyFK``; never a different URDF / KinHelper) -- then the
      canonical measured base pose ``obs/base/pose`` (x, y, yaw, engage-origin
      world): the policy's only absolute world anchor. Its source is declared by
      ``meta/obs_base_pose_source`` (wheel odometry for legacy takes, or ARKit).
  action (29,) or (32,) float32
      WBC takes retain the 29-D WORLD-frame pose/gripper contract. Joystick takes
      use current-base pose/head targets and append body-frame ``[vx,vy,wz]`` from
      ``action/chassis/intent_body``: the follower-effective intent after DOF masking
      and single-axis projection, before integration/PD/slew/dispatch. In both modes
      the recorded values are copied verbatim; no FK or base composition is applied.

Required raw datasets (RuntimeError if absent -- re-record with the updated
recorder; the porter never substitutes ``episode_*_debug.hdf5`` data):
``action/eef/{left,right}`` (T,4,4), ``action/head`` (T,4,4),
``obs/base/pose`` (T,3), plus the usual ``obs/joint``, ``obs/gripper``,
``action/gripper``, and ``obs/images/{head_left_rgb,left_wrist_rgb,
right_wrist_rgb,intrinsic}`` (intrinsic is STATIC (3,3) in this recorder).
``obs/images/head_depth`` is OPTIONAL: image policies never consume it, and current
stereo takes intentionally use ``--no-head-depth`` so FoundationStereo can run offline.

Wrist cameras: ``left``/``right`` name the ARM. Both are ported by default as
``observation.images.{left,right}_wrist_rgb``. ``right_wrist_rgb`` is OPTIONAL so
single-wrist takes still port -- but all takes in one run must agree, since a LeRobot
feature schema is fixed at dataset creation (see ``detect_right_wrist_presence``).
In pre-two-wrist takes the key ``left_wrist_rgb`` physically holds the RIGHT arm's
camera; nothing in the file records which, so it ports as left.

Gripper policy: ``obs/gripper`` is NaN until the first FC03 reply, so leading
frames with a NaN gripper are DROPPED; a NaN after the first finite value is
data corruption and raises. Action grippers are binarized at 0.5 like the
tabletop porter.

Sidecars per episode (not in the parquet):
``debug/depth/episode_XXXXXX.npz``  key ``depth``   (T', resize_h, resize_w) uint16,
    written only when the raw take contains SDK ``head_depth``
``debug/calib/episode_XXXXXX.npz``  keys ``extrinsic`` (world_T_zed, from
    world_T_base(obs/base/pose) @ base_T_zed), ``base_extrinsic`` (base_T_zed
    from the state FK), ``intrinsic`` (static intrinsic rescaled to the output
    size, broadcast (T',3,3)).
``debug/timing/episode_XXXXXX.npz`` key ``timestamp_ns`` (T',) always; plus, for
    takes recorded with the timing-aware recorder, the per-frame capture stamps
    ``head_frame_ns``/``left_wrist_frame_ns``/``grab_wall_ns`` (T'), optional
    ``head_depth_frame_ns``/``right_wrist_frame_ns`` plus the three stereo right-eye
    stamps (T') when present, and the 0-d
    ``ntp_offset_ns``/``ntp_rtt_ns``/``ntp_queried_at_ns`` SoC clock calibration,
    plus optional 0-d ``camera_ntp_{head,left_wrist,right_wrist}_*`` (legacy
    ``camera_ntp_wrist_*``) camera-publisher-host clock
    calibration -- everything scripts/audit_episode_latency.py needs, preserved
    through the port so latency analysis / camera-latency-correction ablations
    never require the raw takes.

Episode ordering & splits: raw ``episode_<N>.hdf5`` are ported in NUMERIC index
order (the recorder writes non-zero-padded names, so a lexicographic sort is
wrong). ``--split-csv`` (default ``<raw-dir>/split.csv``) assigns episodes to a
declared benchmark split; each split is written to its OWN
``<root>/<split>/<repo_id>/`` dataset. Rows are ``episode, split`` where
``episode`` is ``<N>`` (raw) or ``recovery/<N>``; every episode NOT listed
defaults to ``--default-split`` (``train`` by default). This lets one collection
directory such as ``raw_data/offline_val`` be ported directly without copying or
renaming episodes. If the DEFAULT
split.csv is absent every episode goes to ``train``; an explicitly-passed missing
``--split-csv`` errors. Each split dataset is re-opened and cross-checked by
``verify()`` right after it is written.

Recovery episodes (``--include_recovery_data <dir>``, default off): extra takes
in ``<dir>/episode_<N>.hdf5`` are APPENDED after the raw episodes (within their
split), each trimmed to a segment named in the REQUIRED ``<dir>/trim.csv`` (one
header row, then ``episode, frame_start, frame_end`` per row; ABSOLUTE INCLUSIVE
raw-frame indices, ``frame_end == -1`` -> last frame). Every recovery file must
have exactly one trim row and vice-versa (else RuntimeError). The recovery dir is
only READ, never modified.

Index log (``--log_index <csv>``, default off): one row per emitted episode --
``split``, ``source`` (raw/recovery), ``raw_data_index`` (the N in
``episode_<N>.hdf5``), ``processed_data_index`` (0-based LeRobot episode index
WITHIN its split), ``num_frames_before`` (raw take length), ``num_frames_after``
(frames actually written after the leading-NaN drop or the trim).

Position conditioning (``--positions-dir <dir>``, default off): attach a CONSTANT
``observation.environment_state`` (``--object_nums`` * 3, default 6) to every frame of
every ported episode -- the SceneDiff changed-object 3D positions in the engage-origin
WORLD frame (the SAME frame as ``action`` / ``observation.state[29:32]``), ARRANGED
``[s1_src, s1_dst, ...]`` by the task order. Read per episode from
``<dir>/<source>/episode_<N>.npz`` (``source`` in {raw, recovery}, so a raw
``episode_<N>`` and a ``recovery/<N>`` sharing an index never collide), written by
scene_diff Part A (``run_wbc_pos_condition.sh``). Every ported episode MUST have its
npz -- validated UP FRONT, before any dataset is written or overwritten. This is a
single-stage task, so NO per-frame ``observation.pos_condition_mask`` is emitted (the
one constant vector holds throughout, forward-compatible with N grasp cycles).

Structured simulator conditioning (``--structured-env-state-from-raw``): read
the authoritative ``task/env_state`` ``(3,2,3)``, categorical
``task/task_sequence`` ``(3,)``, target-only ``progress_index`` and checked
``active_task_id`` from each raw HDF5. This mode is mutually exclusive with
SceneDiff ``--positions-dir`` and never flattens the task axes. For collection
roots whose episodes already live under split directories, port each directory
with its matching default split, for example::

  python scripts/port_wbc_mobile_hdf5.py \
    --raw-dir /home/yixuan/Dexmate/bigym/raw_data/train \
    --root /home/yixuan/Dexmate/bigym/processed_data \
    --repo-id dexmate_wbc_drawer_dish2rack_close_jar \
    --default-split train --structured-env-state-from-raw

  python scripts/port_wbc_mobile_hdf5.py \
    --raw-dir /home/yixuan/Dexmate/bigym/raw_data/offline_val \
    --root /home/yixuan/Dexmate/bigym/processed_data \
    --repo-id dexmate_wbc_drawer_dish2rack_close_jar \
    --default-split offline_val --structured-env-state-from-raw

Environments: the unit tests need only the dexmate env (pinocchio/pink); the
actual conversion + ``verify()`` additionally need ``lerobot`` and ``tqdm``
(``dexmate_lerobot`` env) -- both imports are lazy.

Usage:
  python scripts/port_wbc_mobile_hdf5.py \
    --raw-dir ~/Dexmate/data/raw_data --root ~/Dexmate/data/processed_wbc \
    --fps 10 --resize-h 120 --resize-w 160 --repo-id dexmate_wbc_eef_head \
    --split-csv ~/Dexmate/data/raw_data/split.csv \
    --include_recovery_data ~/Dexmate/data/raw_data/recovery \
    --log_index ~/Dexmate/data/raw_data/log_index.csv \
    --positions-dir ~/Dexmate/data/scene_diff/positions --object_nums 2
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

import cv2
import h5py
import numpy as np

from omniteleop.wbc_policy_format import (
    ACTION_AXES,
    GRIPPER_BINARY_THRESHOLD,
    GRIPPER_DIMS,
    JOYSTICK_POLICY_ACTION_SCHEMA,
    SKIP_NORMALIZATION_DIMS,
    STATE_AXES,
    STATE_FRAMES,
    WBC_POLICY_ACTION_SCHEMA,
    WBCPolicyFK,
    action_axes_for_policy_schema,
    base_pose_to_mat,
    build_state_vector,
    mat_to_pos6d,
    pos6d_to_mat,
)

# Silence the per-encode libsvtav1 banner (matches the tabletop porter).
os.environ.setdefault("SVT_LOG", "1")

DEFAULT_RAW_DIR = Path("~/Dexmate/data/raw_data").expanduser()
DEFAULT_ROOT = Path("~/Dexmate/data/processed_wbc").expanduser()
DEFAULT_FPS = 10
DEFAULT_RESIZE_H = 120
DEFAULT_RESIZE_W = 160
DEFAULT_TASK = "dexmate_wbc"
DEFAULT_REPO_ID = "dexmate_wbc_eef_head"
ROBOT_TYPE = "dexmate_vega_wbc"
# Splits are written to <root>/<split>/<repo_id>/ (tabletop porter layout).
ALL_SPLITS = (
    "train",
    "val",
    "test",
    "offline_val",
    "rollout_val",
    "iid_test",
    "spatial_ood_test",
)
DEFAULT_SPLIT = "train"  # episodes not named in split.csv land here
# Rotation-column sanity for RECORDED action targets (float32 leader poses).
_ROT_COL_NORM_TOL = 1e-2
# A Hand-E encoder count is 1/255 of normalized achieved position. Requiring two
# counts avoids treating float noise as physical motion. A command state must persist
# for half a second at 10 Hz before this gate calls it intentional.
_GRIPPER_MIN_RESPONSE_RANGE = 2.0 / 255.0
_GRIPPER_MIN_COMMAND_RUN = 5
# Raw v4 required one distinct FC03 event per row. Raw v5 preserves the 10 Hz
# image/action grid across an isolated dropped reply by explicitly zero-order-holding a
# status sample, but only inside this age bound. The bad first rows in episodes 5/6 were
# 107 s / 5 s old and remain unconditionally invalid.
_GRIPPER_STATUS_MAX_AGE_PERIODS = 2.5
_GRIPPER_STATUS_REQUEST_TIMEOUT_V4_S = 0.5
_GRIPPER_STATUS_REQUEST_TIMEOUT_V5_S = 0.075

_REQUIRED_ACTION_TARGETS = ("action/eef/left", "action/eef/right", "action/head")
_BASE_POSE_SOURCES = frozenset({"wheel_odometry", "arkit"})
_RAW_SCHEMA_V3 = "omniteleop_wbc_mobile_raw/v3"
_RAW_SCHEMA_V4 = "omniteleop_wbc_mobile_raw/v4"
_RAW_SCHEMA_V5 = "omniteleop_wbc_mobile_raw/v5"
_RAW_JOYSTICK_SCHEMA_V1 = "omniteleop_joystick_mobile_raw/v1"
_RAW_SCHEMAS_WITH_EXPLICIT_BASE = frozenset({
    _RAW_SCHEMA_V3,
    _RAW_SCHEMA_V4,
    _RAW_SCHEMA_V5,
    _RAW_JOYSTICK_SCHEMA_V1,
})
_RAW_SCHEMAS_WITH_CAMERA_ALIGNMENT = frozenset({
    _RAW_SCHEMA_V4, _RAW_SCHEMA_V5, _RAW_JOYSTICK_SCHEMA_V1,
})
_RAW_SCHEMAS_WITH_GRIPPER_PROVENANCE = frozenset({
    _RAW_SCHEMA_V4, _RAW_SCHEMA_V5, _RAW_JOYSTICK_SCHEMA_V1,
})
_RAW_SCHEMAS_WITH_GRIPPER_REUSE = frozenset({
    _RAW_SCHEMA_V5, _RAW_JOYSTICK_SCHEMA_V1,
})

# Strict ``episode_<int>.hdf5`` name; excludes any ``episode_*_debug.hdf5`` sidecar.
_EPISODE_RE = re.compile(r"^episode_(\d+)\.hdf5$")

_LATENCY_AUDITOR = None


def _load_latency_auditor():
    """Load the sibling audit script without requiring ``scripts`` to be a package."""
    global _LATENCY_AUDITOR
    if _LATENCY_AUDITOR is not None:
        return _LATENCY_AUDITOR
    path = Path(__file__).resolve().with_name("audit_episode_latency.py")
    spec = importlib.util.spec_from_file_location("wbc_audit_episode_latency", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import the required latency auditor at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LATENCY_AUDITOR = module
    return module


def audit_work_list(work: list[dict]) -> dict[Path, dict]:
    """Run and print the mandatory timing audit for every raw episode.

    Dataset processors call this before creating, deleting, or overwriting outputs.
    Malformed/non-monotonic timing aborts the whole port. Latency magnitudes remain
    reported measurements: no pass/fail threshold is imposed without an explicit
    experiment contract.
    """
    auditor = _load_latency_auditor()
    results: dict[Path, dict] = {}
    failures: list[str] = []
    print(f"\nRunning required episode-latency audit ({len(work)} episode(s)) ...")
    for item in work:
        path = Path(item["path"])
        try:
            result = auditor.audit_episode(path)
            auditor.print_report(result)
            results[path] = result
        except Exception as exc:
            failures.append(f"- {path}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError(
            "Required episode-latency audit failed; no dataset was written:\n"
            + "\n".join(failures)
        )
    logging.info("Required episode-latency audit passed for %d episode(s)", len(results))
    return results


def episode_index_from_path(path: Path) -> int:
    """Integer N parsed from an ``episode_<N>.hdf5`` filename (for numeric sort)."""
    match = _EPISODE_RE.match(Path(path).name)
    if match is None:
        raise RuntimeError(f"episode filename does not match episode_<int>.hdf5: {Path(path).name}")
    return int(match.group(1))


def find_episode_files(raw_dir: Path) -> list[Path]:
    """``episode_<N>.hdf5`` in ``raw_dir``, sorted by the NUMERIC index N.

    The recorder writes non-zero-padded names (``episode_1``..``episode_61``), so a
    plain lexicographic sort is wrong (``episode_11`` before ``episode_2``); we sort by
    the parsed integer. Names that are not ``episode_<int>.hdf5`` (e.g. any
    ``episode_*_debug.hdf5``) are ignored, never ported.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise RuntimeError(f"raw dir not found: {raw_dir}")
    episode_files = sorted(
        (p for p in raw_dir.glob("episode_*.hdf5") if _EPISODE_RE.match(p.name)),
        key=episode_index_from_path,
    )
    if not episode_files:
        raise RuntimeError(f"no episode_*.hdf5 found in {raw_dir}")
    return episode_files


def load_trim_spec(recovery_dir: Path) -> dict[int, tuple[int, int]]:
    """Parse ``<recovery_dir>/trim.csv`` into ``{episode_index: (frame_start, frame_end)}``.

    Format (one header row, then one row per episode)::

        episode, frame_start, frame_end
        25, 409, -1

    ``frame_start`` / ``frame_end`` are ABSOLUTE, INCLUSIVE raw-frame indices into
    ``episode_<idx>.hdf5``; ``frame_end == -1`` means "through the last frame". Raises
    on a missing file, a malformed row, a non-integer field, a negative index, an
    inverted range, or a duplicate episode. The file is only READ, never modified.
    """
    trim_path = Path(recovery_dir) / "trim.csv"
    if not trim_path.exists():
        raise RuntimeError(
            f"--include_recovery_data requires the trim file {trim_path} (not found)"
        )
    spec: dict[int, tuple[int, int]] = {}
    first_row = True
    with open(trim_path, newline="") as handle:
        for lineno, row in enumerate(csv.reader(handle), start=1):
            cells = [c.strip() for c in row]
            if not cells or all(c == "" for c in cells):
                continue  # blank line
            if len(cells) != 3:
                raise RuntimeError(
                    f"{trim_path}:{lineno}: expected 3 columns "
                    f"'episode, frame_start, frame_end', got {len(cells)}: {row}"
                )
            if first_row and not cells[0].lstrip("-").isdigit():
                first_row = False
                continue  # header row (e.g. 'episode, frame_start, frame_end')
            first_row = False
            try:
                episode, frame_start, frame_end = (int(c) for c in cells)
            except ValueError as exc:
                raise RuntimeError(f"{trim_path}:{lineno}: non-integer field in {row}") from exc
            if episode < 0:
                raise RuntimeError(f"{trim_path}:{lineno}: negative episode index {episode}")
            if frame_start < 0:
                raise RuntimeError(
                    f"{trim_path}:{lineno}: frame_start must be >= 0, got {frame_start}"
                )
            if frame_end < -1:
                raise RuntimeError(
                    f"{trim_path}:{lineno}: frame_end must be >= 0 or -1, got {frame_end}"
                )
            if frame_end != -1 and frame_end < frame_start:
                raise RuntimeError(
                    f"{trim_path}:{lineno}: frame_end {frame_end} < frame_start {frame_start}"
                )
            if episode in spec:
                raise RuntimeError(f"{trim_path}:{lineno}: duplicate episode {episode}")
            spec[episode] = (frame_start, frame_end)
    if not spec:
        raise RuntimeError(f"{trim_path}: no trim rows found")
    return spec


def resolve_episode_window(
    frame_count: int, t0: int, trim: tuple[int, int] | None, source: str | Path
) -> tuple[int, int]:
    """The half-open ``[start, end)`` frame window an episode contributes.

    ``trim is None`` (raw episode): ``[t0, frame_count)`` -- the porter's usual
    leading-NaN-gripper drop. ``trim=(frame_start, frame_end)`` (recovery episode):
    the ABSOLUTE, INCLUSIVE ``[frame_start, frame_end]`` (``frame_end == -1`` -> last
    frame), independent of ``t0``; a NaN obs gripper inside the window is caught
    per-frame downstream (``compute_frame_state_action``) and raises there.
    """
    if trim is None:
        return t0, frame_count
    frame_start, frame_end = trim
    end_incl = frame_count - 1 if frame_end == -1 else frame_end
    if not 0 <= frame_start <= end_incl <= frame_count - 1:
        raise RuntimeError(
            f"{source}: trim [{frame_start}, {frame_end}] out of range for T={frame_count} "
            f"(resolved last frame {end_incl}; need 0 <= start <= end <= T-1)"
        )
    return frame_start, end_incl + 1


def _validate_finite(
    name: str, arr: np.ndarray, source: str | Path, *, check_finite: bool = True
) -> None:
    if arr.size == 0:
        raise RuntimeError(f"{source}: {name} is empty (shape={arr.shape})")
    if arr.dtype.kind not in "biufc":
        raise RuntimeError(f"{source}: {name} has non-numeric dtype {arr.dtype}")
    if check_finite and arr.dtype.kind in "fc" and not np.all(np.isfinite(arr)):
        idx = np.argwhere(~np.isfinite(arr))[0]
        raise RuntimeError(
            f"{source}: {name} is non-finite at index {idx.tolist()} "
            f"(shape={arr.shape}, dtype={arr.dtype})"
        )


def read_required_array(
    f: h5py.File, key: str, source: str | Path, *, check_finite: bool = True
) -> np.ndarray:
    """Load a required dataset; RuntimeError names the missing path (PLAN.md)."""
    if key not in f:
        raise RuntimeError(
            f"{source}: missing required HDF5 dataset {key} -- re-record with the "
            "updated wbc_vr_robot.py recorder (debug files are never a substitute)"
        )
    node = f[key]
    if not isinstance(node, h5py.Dataset):
        raise RuntimeError(f"{source}: {key} is not an HDF5 dataset")
    arr = np.asarray(node[()])
    _validate_finite(key, arr, source, check_finite=check_finite)
    return arr


def _read_optional_hdf5_text(
    f: h5py.File, key: str, source: str | Path
) -> str | None:
    """Read one optional scalar UTF-8/ASCII HDF5 dataset."""
    if key not in f:
        return None
    node = f[key]
    if not isinstance(node, h5py.Dataset):
        raise RuntimeError(f"{source}: {key} is not an HDF5 dataset")
    value = np.asarray(node[()])
    if value.shape != ():
        raise RuntimeError(f"{source}: {key} must be scalar text, got {value.shape}")
    scalar = value.item()
    if isinstance(scalar, (bytes, np.bytes_)):
        try:
            return bytes(scalar).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"{source}: {key} is not valid UTF-8") from exc
    if isinstance(scalar, str):
        return scalar
    raise RuntimeError(
        f"{source}: {key} must be scalar text, got {type(scalar).__name__}"
    )


def _inspect_open_base_pose_contract(
    f: h5py.File, source: str | Path
) -> dict[str, str | bool | None]:
    """Validate which sensor owns canonical ``obs/base/pose`` in one raw take.

    Raw-v3 always retains wheel odometry for diagnostics while allowing ARKit to be
    the canonical control/policy pose. Older takes predate this declaration and are
    interpreted as wheel odometry, matching their recorder semantics.
    """
    canonical = read_required_array(f, "obs/base/pose", source)
    if canonical.ndim != 2 or canonical.shape[1] != 3:
        raise RuntimeError(
            f"{source}: obs/base/pose must be (T,3), got {canonical.shape}"
        )

    schema = _read_optional_hdf5_text(f, "meta/schema", source)
    provenance_schema = _read_optional_hdf5_text(
        f, "meta/provenance/schema", source
    )
    if schema is not None and provenance_schema is not None and schema != provenance_schema:
        raise RuntimeError(
            f"{source}: meta/schema={schema!r} disagrees with "
            f"meta/provenance/schema={provenance_schema!r}"
        )
    schema = schema or provenance_schema

    declared_source = _read_optional_hdf5_text(
        f, "meta/obs_base_pose_source", source
    )
    legacy_inferred = declared_source is None
    base_pose_source = declared_source or "wheel_odometry"
    if base_pose_source not in _BASE_POSE_SOURCES:
        raise RuntimeError(
            f"{source}: meta/obs_base_pose_source={base_pose_source!r}; expected one "
            f"of {sorted(_BASE_POSE_SOURCES)}"
        )

    control_source = _read_optional_hdf5_text(
        f, "meta/base_control_pose_source", source
    )
    if control_source is not None and control_source not in _BASE_POSE_SOURCES:
        raise RuntimeError(
            f"{source}: meta/base_control_pose_source={control_source!r}; expected one "
            f"of {sorted(_BASE_POSE_SOURCES)}"
        )
    if control_source is not None and control_source != base_pose_source:
        raise RuntimeError(
            f"{source}: canonical obs/base/pose source {base_pose_source!r} disagrees "
            f"with base control source {control_source!r}"
        )

    if schema in _RAW_SCHEMAS_WITH_EXPLICIT_BASE:
        if legacy_inferred:
            raise RuntimeError(
                f"{source}: {schema} requires meta/obs_base_pose_source"
            )
        if control_source is None:
            raise RuntimeError(
                f"{source}: {schema} requires meta/base_control_pose_source"
            )
        if "obs/base/pose_odom" not in f:
            raise RuntimeError(
                f"{source}: {schema} requires diagnostic obs/base/pose_odom"
            )

    reference_key = (
        "obs/base/pose_arkit" if base_pose_source == "arkit" else "obs/base/pose_odom"
    )
    if base_pose_source == "arkit" and reference_key not in f:
        raise RuntimeError(
            f"{source}: ARKit is the declared canonical base pose but {reference_key} "
            "is missing"
        )
    if reference_key in f:
        reference = read_required_array(f, reference_key, source)
        if reference.shape != canonical.shape:
            raise RuntimeError(
                f"{source}: {reference_key} shape {reference.shape} != canonical "
                f"obs/base/pose shape {canonical.shape}"
            )
        if not np.array_equal(canonical, reference):
            mismatch = np.argwhere(canonical != reference)
            frame = int(mismatch[0, 0]) if mismatch.size else 0
            raise RuntimeError(
                f"{source}: obs/base/pose does not exactly match declared "
                f"{base_pose_source} stream {reference_key} (first mismatch frame {frame})"
            )

    return {
        "base_pose_source": base_pose_source,
        "base_control_pose_source": control_source,
        "raw_schema": schema,
        "legacy_inferred": legacy_inferred,
    }


def inspect_episode_base_pose_contract(hdf5_path: Path) -> dict[str, str | bool | None]:
    """Open and validate one episode's canonical base-pose source contract."""
    path = Path(hdf5_path)
    with h5py.File(path, "r") as f:
        return _inspect_open_base_pose_contract(f, path)


def validate_work_base_pose_contract(work: list[dict]) -> str:
    """Require every episode in one processed dataset to use one pose source."""
    by_source: dict[str, list[Path]] = {}
    failures: list[str] = []
    for item in work:
        path = Path(item["path"])
        try:
            contract = inspect_episode_base_pose_contract(path)
            source = str(contract["base_pose_source"])
            by_source.setdefault(source, []).append(path)
        except Exception as exc:
            failures.append(f"- {path}: {exc}")
    if failures:
        raise RuntimeError(
            "Raw base-pose contract validation failed; no dataset was written:\n"
            + "\n".join(failures)
        )
    if len(by_source) != 1:
        detail = ", ".join(
            f"{source}={len(paths)} (e.g. {paths[0]})"
            for source, paths in sorted(by_source.items())
        )
        raise RuntimeError(
            "source episodes mix canonical base-pose sources; train/port wheel "
            f"odometry and ARKit separately: {detail}"
        )
    if not by_source:
        raise RuntimeError("cannot validate base-pose source for an empty work list")
    return next(iter(by_source))


def _inspect_open_policy_action_contract(
    f: h5py.File, source: str | Path
) -> dict[str, object]:
    """Validate the action frame/layout declared by one raw episode.

    Historical WBC takes predate these metadata fields and retain their established
    29-D engage-world contract. Joystick takes are intentionally strict: silently
    inferring the human chassis intent from an integrated pose or an applied twist
    would create a different learning target, so all declarations and the exact
    post-projection intent dataset are required.
    """
    raw_schema = (
        _read_optional_hdf5_text(f, "meta/schema", source)
        or _read_optional_hdf5_text(f, "meta/provenance/schema", source)
    )
    control_mode = _read_optional_hdf5_text(f, "meta/control_mode", source)
    policy_schema = _read_optional_hdf5_text(
        f, "meta/policy_action_schema", source
    )
    frame_values = {
        name: _read_optional_hdf5_text(f, f"meta/{name}", source)
        for name in ("action_target_frame", "eef_target_frame", "head_target_frame")
    }

    joystick_markers = (
        raw_schema == _RAW_JOYSTICK_SCHEMA_V1,
        policy_schema == JOYSTICK_POLICY_ACTION_SCHEMA,
        control_mode == "joystick",
    )
    is_joystick = any(joystick_markers)
    if is_joystick:
        expected = {
            "raw schema": (raw_schema, _RAW_JOYSTICK_SCHEMA_V1),
            "meta/control_mode": (control_mode, "joystick"),
            "meta/policy_action_schema": (
                policy_schema,
                JOYSTICK_POLICY_ACTION_SCHEMA,
            ),
            "meta/action_target_frame": (
                frame_values["action_target_frame"],
                "current_base",
            ),
            "meta/eef_target_frame": (
                frame_values["eef_target_frame"],
                "current_base",
            ),
            "meta/head_target_frame": (
                frame_values["head_target_frame"],
                "current_base",
            ),
        }
        bad = [
            f"{name}={actual!r} (expected {wanted!r})"
            for name, (actual, wanted) in expected.items()
            if actual != wanted
        ]
        if bad:
            raise RuntimeError(
                f"{source}: incomplete/conflicting joystick policy contract: "
                + "; ".join(bad)
            )
        intent = read_required_array(f, "action/chassis/intent_body", source)
        if intent.ndim != 2 or intent.shape[1] != 3:
            raise RuntimeError(
                f"{source}: action/chassis/intent_body must be (T,3), got "
                f"{intent.shape}"
            )
        if "action/eef/left" in f and intent.shape[0] != f["action/eef/left"].shape[0]:
            raise RuntimeError(
                f"{source}: action/chassis/intent_body frame count {intent.shape[0]} "
                f"!= action/eef/left {f['action/eef/left'].shape[0]}"
            )
        resolved_schema = JOYSTICK_POLICY_ACTION_SCHEMA
        action_frame = "current_base"
    else:
        if raw_schema == _RAW_JOYSTICK_SCHEMA_V1 or control_mode == "joystick":
            raise RuntimeError(f"{source}: malformed joystick policy contract")
        if policy_schema not in (None, WBC_POLICY_ACTION_SCHEMA):
            raise RuntimeError(
                f"{source}: unknown meta/policy_action_schema={policy_schema!r}"
            )
        for name, value in frame_values.items():
            if value not in (None, "engage_origin_world", "world"):
                raise RuntimeError(
                    f"{source}: WBC {name}={value!r}; expected engage-origin world"
                )
        resolved_schema = WBC_POLICY_ACTION_SCHEMA
        action_frame = "world"

    return {
        "policy_action_schema": resolved_schema,
        "action_axes": action_axes_for_policy_schema(resolved_schema),
        "action_target_frame": action_frame,
        "control_mode": control_mode or ("joystick" if is_joystick else "wbc"),
        "raw_schema": raw_schema,
    }


def inspect_episode_policy_action_contract(hdf5_path: Path) -> dict[str, object]:
    """Open and validate one episode's policy-action schema/frame contract."""
    path = Path(hdf5_path)
    with h5py.File(path, "r") as f:
        return _inspect_open_policy_action_contract(f, path)


def validate_work_policy_action_contract(work: list[dict]) -> dict[str, object]:
    """Require every take in one processed dataset to share one action meaning."""
    contracts: dict[tuple[str, str], list[Path]] = {}
    failures: list[str] = []
    first_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for item in work:
        path = Path(item["path"])
        try:
            contract = inspect_episode_policy_action_contract(path)
            key = (
                str(contract["policy_action_schema"]),
                str(contract["action_target_frame"]),
            )
            contracts.setdefault(key, []).append(path)
            first_by_key.setdefault(key, contract)
        except Exception as exc:
            failures.append(f"- {path}: {exc}")
    if failures:
        raise RuntimeError(
            "Raw policy-action contract validation failed; no dataset was written:\n"
            + "\n".join(failures)
        )
    if not contracts:
        raise RuntimeError("cannot validate policy action contract for an empty work list")
    if len(contracts) != 1:
        detail = ", ".join(
            f"{schema}/{frame}={len(paths)} (e.g. {paths[0]})"
            for (schema, frame), paths in sorted(contracts.items())
        )
        raise RuntimeError(
            "source episodes mix policy action contracts; keep joystick/current-base "
            f"and WBC/world datasets separate: {detail}"
        )
    key = next(iter(contracts))
    return first_by_key[key]


_CAMERA_ALIGNMENT_MODES = frozenset({
    "head_capture_nearest",
    "latest_arrived_compatibility",
})


def _read_required_int64_scalar(
    f: h5py.File, key: str, source: str | Path
) -> int:
    if key not in f or not isinstance(f[key], h5py.Dataset):
        raise RuntimeError(f"{source}: missing required scalar dataset {key}")
    value = np.asarray(f[key][()])
    if value.shape != () or value.dtype != np.int64:
        raise RuntimeError(
            f"{source}: {key} must be 0-d int64, got {value.shape} {value.dtype}"
        )
    return int(value)


def _timing_camera_offset_ns(timing: dict, label: str) -> int | None:
    value = timing.get(f"camera_ntp_{label}_offset_ns")
    if value is None and label == "left_wrist":
        value = timing.get("camera_ntp_wrist_offset_ns")
    return None if value is None else int(np.asarray(value))


def _inspect_open_camera_alignment_contract(
    f: h5py.File,
    source: str | Path,
    timing: dict | None,
    *,
    raw_schema: str | None = None,
) -> dict:
    """Validate the recorder's head-anchored multi-camera timing contract."""
    if raw_schema is None:
        raw_schema = (
            _read_optional_hdf5_text(f, "meta/schema", source)
            or _read_optional_hdf5_text(f, "meta/provenance/schema", source)
        )
    group = "meta/camera_alignment"
    if group not in f:
        if raw_schema in _RAW_SCHEMAS_WITH_CAMERA_ALIGNMENT:
            raise RuntimeError(
                f"{source}: {raw_schema} requires {group}; re-record with the "
                "capture-nearest recorder"
            )
        return {
            "mode": "latest_arrived_legacy_unverified",
            "clock_domain": "raw_publisher_clock_unverified",
            "max_abs_skew_ns": None,
            "max_camera_age_ns": None,
            "wrist_buffer_frames": None,
            "verified": False,
            "measured_max_abs_skew_ns": {},
            "measured_max_camera_age_ns": {},
            "raw_schema": raw_schema,
        }
    if not isinstance(f[group], h5py.Group):
        raise RuntimeError(f"{source}: {group} must be an HDF5 group")

    required_text = {
        "mode": "head_capture_nearest or latest_arrived_compatibility",
        "anchor": "head_left_rgb_capture",
        "clock_domain": "local_via_camera_ntp",
    }
    text_values: dict[str, str] = {}
    for key, expected in required_text.items():
        value = _read_optional_hdf5_text(f, f"{group}/{key}", source)
        if value is None:
            raise RuntimeError(f"{source}: missing required {group}/{key} ({expected})")
        text_values[key] = value
    mode = text_values["mode"]
    if mode not in _CAMERA_ALIGNMENT_MODES:
        raise RuntimeError(
            f"{source}: {group}/mode={mode!r}; expected one of "
            f"{sorted(_CAMERA_ALIGNMENT_MODES)}"
        )
    if text_values["anchor"] != "head_left_rgb_capture":
        raise RuntimeError(
            f"{source}: {group}/anchor={text_values['anchor']!r}, expected "
            "'head_left_rgb_capture'"
        )
    if text_values["clock_domain"] != "local_via_camera_ntp":
        raise RuntimeError(
            f"{source}: {group}/clock_domain={text_values['clock_domain']!r}, expected "
            "'local_via_camera_ntp'"
        )
    max_skew_ns = _read_required_int64_scalar(
        f, f"{group}/max_abs_skew_ns", source
    )
    max_camera_age_ns = _read_required_int64_scalar(
        f, f"{group}/max_camera_age_ns", source
    )
    buffer_frames = _read_required_int64_scalar(
        f, f"{group}/wrist_buffer_frames", source
    )
    if buffer_frames <= 0:
        raise RuntimeError(
            f"{source}: {group}/wrist_buffer_frames must be > 0, got {buffer_frames}"
        )
    if max_camera_age_ns <= 0:
        raise RuntimeError(
            f"{source}: {group}/max_camera_age_ns must be > 0, got "
            f"{max_camera_age_ns}"
        )
    if mode == "head_capture_nearest" and max_skew_ns <= 0:
        raise RuntimeError(
            f"{source}: head_capture_nearest requires max_abs_skew_ns > 0, got "
            f"{max_skew_ns}"
        )
    if mode == "latest_arrived_compatibility" and max_skew_ns != 0:
        raise RuntimeError(
            f"{source}: latest_arrived_compatibility requires max_abs_skew_ns == 0, "
            f"got {max_skew_ns}"
        )
    if timing is None:
        raise RuntimeError(
            f"{source}: {group} exists without per-frame camera timing; cannot verify it"
        )

    measured: dict[str, int] = {}
    measured_age: dict[str, int] = {}
    head_offset = _timing_camera_offset_ns(timing, "head")
    if head_offset is None:
        raise RuntimeError(f"{source}: {mode} requires meta/camera_ntp/head")
    head_local = timing["head_frame_ns"] - head_offset
    head_age = timing["grab_wall_ns"] - head_local
    bad_head_age = (head_age < -10_000_000) | (head_age > max_camera_age_ns)
    if np.any(bad_head_age):
        frame = int(np.flatnonzero(bad_head_age)[0])
        raise RuntimeError(
            f"{source}: corrected head camera age at frame {frame} is "
            f"{int(head_age[frame]) / 1e6:+.3f} ms, outside the declared "
            f"[-10, {max_camera_age_ns / 1e6:g}] ms contract"
        )
    measured_age["head"] = int(head_age.max())
    wrist_streams = [("left_wrist", "left_wrist_frame_ns")]
    if "right_wrist_frame_ns" in timing:
        wrist_streams.append(("right_wrist", "right_wrist_frame_ns"))
    for label, key in wrist_streams:
        wrist_offset = _timing_camera_offset_ns(timing, label)
        if wrist_offset is None:
            raise RuntimeError(f"{source}: {mode} requires meta/camera_ntp/{label}")
        wrist_local = timing[key] - wrist_offset
        if mode == "head_capture_nearest":
            skew_ns = head_local - wrist_local
            abs_skew = np.abs(skew_ns)
            measured[label] = int(abs_skew.max())
            bad = abs_skew > max_skew_ns
            if np.any(bad):
                frame = int(np.flatnonzero(bad)[0])
                raise RuntimeError(
                    f"{source}: corrected head-minus-{label} capture skew at frame "
                    f"{frame} is {int(skew_ns[frame]) / 1e6:+.3f} ms, exceeding the "
                    f"declared {max_skew_ns / 1e6:g} ms contract"
                )
        wrist_age = timing["grab_wall_ns"] - wrist_local
        bad_age = (wrist_age < -10_000_000) | (wrist_age > max_camera_age_ns)
        if np.any(bad_age):
            frame = int(np.flatnonzero(bad_age)[0])
            raise RuntimeError(
                f"{source}: corrected {label} camera age at frame {frame} is "
                f"{int(wrist_age[frame]) / 1e6:+.3f} ms, outside the declared "
                f"[-10, {max_camera_age_ns / 1e6:g}] ms contract"
            )
        measured_age[label] = int(wrist_age.max())

    return {
        "mode": mode,
        "clock_domain": text_values["clock_domain"],
        "max_abs_skew_ns": max_skew_ns,
        "max_camera_age_ns": max_camera_age_ns,
        "wrist_buffer_frames": buffer_frames,
        "verified": mode == "head_capture_nearest",
        "measured_max_abs_skew_ns": measured,
        "measured_max_camera_age_ns": measured_age,
        "raw_schema": raw_schema,
    }


def inspect_episode_camera_alignment_contract(hdf5_path: Path) -> dict:
    path = Path(hdf5_path)
    with h5py.File(path, "r") as f:
        if "timestamp_ns" not in f:
            raise RuntimeError(f"{path}: missing timestamp_ns")
        frame_count = int(f["timestamp_ns"].shape[0])
        timing = load_episode_timing(f, path, frame_count)
        return _inspect_open_camera_alignment_contract(f, path, timing)


def validate_work_camera_alignment_contract(work: list[dict]) -> dict:
    """Require one camera-pairing semantic/limit throughout a training dataset."""
    by_signature: dict[tuple, list[Path]] = {}
    contracts: dict[tuple, dict] = {}
    failures: list[str] = []
    for item in work:
        path = Path(item["path"])
        try:
            contract = inspect_episode_camera_alignment_contract(path)
            signature = (
                contract["mode"],
                contract["clock_domain"],
                contract["max_abs_skew_ns"],
                contract["max_camera_age_ns"],
            )
            by_signature.setdefault(signature, []).append(path)
            contracts[signature] = contract
        except Exception as exc:
            failures.append(f"- {path}: {exc}")
    if failures:
        raise RuntimeError(
            "Raw camera-alignment contract validation failed; no dataset was written:\n"
            + "\n".join(failures)
        )
    if not by_signature:
        raise RuntimeError("cannot validate camera alignment for an empty work list")
    if len(by_signature) != 1:
        detail = ", ".join(
            f"{signature}={len(paths)} (e.g. {paths[0]})"
            for signature, paths in by_signature.items()
        )
        raise RuntimeError(
            "source episodes mix camera-alignment semantics/limits; port them "
            f"separately: {detail}"
        )
    return contracts[next(iter(by_signature))]


_TIMING_KEYS = ("head_frame_ns", "left_wrist_frame_ns", "grab_wall_ns")
# right_wrist_frame_ns rides with the second wrist camera; optional so single-wrist
# takes recorded before it existed still port.
_OPTIONAL_TIMING_KEYS = (
    "head_depth_frame_ns",
    "right_wrist_frame_ns",
    "head_right_frame_ns",
    "left_wrist_right_frame_ns",
    "right_wrist_right_frame_ns",
    "head_receive_ns",
    "head_depth_receive_ns",
    "head_right_receive_ns",
    "left_wrist_receive_ns",
    "left_wrist_right_receive_ns",
    "right_wrist_receive_ns",
    "right_wrist_right_receive_ns",
)
# Per-frame causal timing added after capture/receive timestamps. All six fields are
# required together: a partial bracket cannot answer whether the action, image, and
# measured robot state belong to the same point on the timeline.
_CAUSAL_TIMING_KEYS = (
    "source_timestamp_ns",
    "source_receive_wall_ns",
    "action_dispatch_start_wall_ns",
    "action_dispatch_end_wall_ns",
    "state_read_start_wall_ns",
    "state_read_end_wall_ns",
)
_NTP_KEYS = ("offset_ns", "rtt_ns", "queried_at_ns")
# "wrist" is the legacy single-camera label (wbc_vr_robot before the two-wrist split);
# absent labels are skipped, so old and new episodes both load whatever they carry.
_CAMERA_NTP_LABELS = ("head", "left_wrist", "right_wrist", "wrist")
_CAMERA_NTP_INT_KEYS = ("offset_ns", "rtt_ns", "queried_at_ns")
_CAMERA_NTP_STR_KEYS = ("sensor_id", "source")


def load_episode_timing(f: h5py.File, source: str | Path, frame_count: int) -> dict | None:
    """Optional per-frame capture-stamp block written by the timing-aware recorder.

    Returns None for takes predating the timing fields (pre-2026-07 recorder).
    When present the fields are validated aggressively: ALL of
    ``obs/images/{head_frame_ns,left_wrist_frame_ns,grab_wall_ns}`` must exist
    together (a partial set is a recorder bug, not an old take), be
    ``(frame_count,)`` int64, positive, and strictly increasing (the recorder's
    freshness gate forbids duplicate frames). ``meta/ntp`` is optional on top --
    the recorder omits it when ``query_ntp`` fails -- but must be complete 0-d
    int64 when present (``offset_ns`` may be negative; the others must not be).
    Returned dict maps the per-frame keys to (T,) arrays and ``ntp_<key>`` to 0-d
    arrays, matching the ``debug/timing`` sidecar layout. Optional
    ``obs/images/head_depth_frame_ns`` is optional for compatibility with takes recorded
    before depth carried its own stamp; when present it is validated and preserved.
    ``obs/images/right_wrist_frame_ns`` is optional the same way (absent on
    single-wrist takes). ``meta/camera_ntp/{head,left_wrist,right_wrist}`` groups --
    plus the legacy single-camera ``wrist`` label -- are flattened to
    ``camera_ntp_<label>_<key>`` scalar sidecar entries; absent labels are skipped.
    """
    present = [key for key in _TIMING_KEYS if f"obs/images/{key}" in f]
    optional_present = [key for key in _OPTIONAL_TIMING_KEYS if f"obs/images/{key}" in f]
    causal_present = [key for key in _CAUSAL_TIMING_KEYS if f"timing/{key}" in f]
    if not present:
        if optional_present or causal_present:
            raise RuntimeError(
                f"{source}: optional timing fields {optional_present + causal_present} "
                "exist without "
                f"required {list(_TIMING_KEYS)}; corrupt timing metadata"
            )
        return None
    if len(present) != len(_TIMING_KEYS):
        missing = sorted(set(_TIMING_KEYS) - set(present))
        raise RuntimeError(
            f"{source}: partial timing fields -- obs/images has {present} but is "
            f"missing {missing}; a timing-aware take must carry all of them"
        )
    timing: dict[str, np.ndarray] = {}
    for key in _TIMING_KEYS:
        arr = np.asarray(f[f"obs/images/{key}"][()])
        if arr.shape != (frame_count,) or arr.dtype != np.int64:
            raise RuntimeError(
                f"{source}: obs/images/{key} must be ({frame_count},) int64, got "
                f"{arr.shape} {arr.dtype}"
            )
        if np.any(arr <= 0):
            raise RuntimeError(f"{source}: obs/images/{key} has non-positive stamps")
        if arr.shape[0] > 1 and np.any(np.diff(arr) <= 0):
            raise RuntimeError(
                f"{source}: obs/images/{key} is not strictly increasing -- the "
                "recorder freshness gate forbids duplicate frames; corrupt take"
            )
        timing[key] = arr
    if causal_present and len(causal_present) != len(_CAUSAL_TIMING_KEYS):
        missing = sorted(set(_CAUSAL_TIMING_KEYS) - set(causal_present))
        raise RuntimeError(
            f"{source}: partial timing causal bracket -- timing has "
            f"{causal_present} but is missing {missing}"
        )
    for key in causal_present:
        arr = np.asarray(f[f"timing/{key}"][()])
        if arr.shape != (frame_count,) or arr.dtype != np.int64:
            raise RuntimeError(
                f"{source}: timing/{key} must be ({frame_count},) int64, got "
                f"{arr.shape} {arr.dtype}"
            )
        if np.any(arr <= 0):
            raise RuntimeError(f"{source}: timing/{key} has non-positive stamps")
        # One source command may legitimately feed two adjacent 10 Hz record rows;
        # all locally generated action/state stages, however, occur once per row.
        strict = key not in {"source_timestamp_ns", "source_receive_wall_ns"}
        delta = np.diff(arr)
        if arr.shape[0] > 1 and np.any(delta <= 0 if strict else delta < 0):
            order = "strictly increasing" if strict else "monotonic"
            raise RuntimeError(f"{source}: timing/{key} is not {order}")
        timing[key] = arr
    if causal_present:
        source_receive = timing["source_receive_wall_ns"]
        dispatch_start = timing["action_dispatch_start_wall_ns"]
        dispatch_end = timing["action_dispatch_end_wall_ns"]
        state_start = timing["state_read_start_wall_ns"]
        state_end = timing["state_read_end_wall_ns"]
        commit = np.asarray(f["timestamp_ns"][()])
        causal_checks = (
            (source_receive <= dispatch_start, "source_receive <= action_dispatch_start"),
            (dispatch_start <= dispatch_end, "action_dispatch_start <= action_dispatch_end"),
            (timing["grab_wall_ns"] <= state_start, "grab_wall <= state_read_start"),
            (dispatch_end <= state_start, "action_dispatch_end <= state_read_start"),
            (state_start <= state_end, "state_read_start <= state_read_end"),
            (state_end <= commit, "state_read_end <= timestamp_ns(commit)"),
        )
        for valid, relation in causal_checks:
            if not np.all(valid):
                frame = int(np.flatnonzero(~valid)[0])
                raise RuntimeError(
                    f"{source}: causal timing order violated at frame {frame}: {relation}"
                )
    for key in optional_present:
        arr = np.asarray(f[f"obs/images/{key}"][()])
        if arr.shape != (frame_count,) or arr.dtype != np.int64:
            raise RuntimeError(
                f"{source}: obs/images/{key} must be ({frame_count},) int64, got "
                f"{arr.shape} {arr.dtype}"
            )
        if np.any(arr <= 0):
            raise RuntimeError(f"{source}: obs/images/{key} has non-positive stamps")
        # Every RGB eye is a camera stream behind the recorder's freshness gate and
        # must be strictly increasing. head_depth_frame_ns alone may repeat when SDK
        # depth lags RGB by a frame.
        if key != "head_depth_frame_ns":
            if arr.shape[0] > 1 and np.any(np.diff(arr) <= 0):
                raise RuntimeError(
                    f"{source}: obs/images/{key} is not strictly increasing -- the "
                    "recorder freshness gate forbids duplicate frames; corrupt take"
                )
        elif arr.shape[0] > 1 and np.any(np.diff(arr) < 0):
            raise RuntimeError(f"{source}: obs/images/{key} is not monotonic -- corrupt take")
        timing[key] = arr
    ntp_present = [key for key in _NTP_KEYS if f"meta/ntp/{key}" in f]
    if ntp_present and len(ntp_present) != len(_NTP_KEYS):
        missing = sorted(set(_NTP_KEYS) - set(ntp_present))
        raise RuntimeError(f"{source}: partial meta/ntp -- has {ntp_present}, missing {missing}")
    for key in ntp_present:
        val = np.asarray(f[f"meta/ntp/{key}"][()])
        if val.shape != () or val.dtype != np.int64:
            raise RuntimeError(
                f"{source}: meta/ntp/{key} must be 0-d int64, got {val.shape} {val.dtype}"
            )
        if key != "offset_ns" and int(val) <= 0:
            raise RuntimeError(f"{source}: meta/ntp/{key} must be positive, got {int(val)}")
        timing[f"ntp_{key}"] = val
    for label in _CAMERA_NTP_LABELS:
        group_path = f"meta/camera_ntp/{label}"
        if group_path not in f:
            continue
        node = f[group_path]
        if not isinstance(node, h5py.Group):
            raise RuntimeError(f"{source}: {group_path} must be an HDF5 group")
        expected = set(_CAMERA_NTP_INT_KEYS) | set(_CAMERA_NTP_STR_KEYS)
        present = {key for key in expected if f"{group_path}/{key}" in f}
        if present != expected:
            raise RuntimeError(
                f"{source}: partial {group_path} -- has {sorted(present)}, "
                f"missing {sorted(expected - present)}"
            )
        for key in _CAMERA_NTP_INT_KEYS:
            val = np.asarray(f[f"{group_path}/{key}"][()])
            if val.shape != () or val.dtype != np.int64:
                raise RuntimeError(
                    f"{source}: {group_path}/{key} must be 0-d int64, got "
                    f"{val.shape} {val.dtype}"
                )
            if key != "offset_ns" and int(val) <= 0:
                raise RuntimeError(f"{source}: {group_path}/{key} must be positive, got {int(val)}")
            timing[f"camera_ntp_{label}_{key}"] = val
        for key in _CAMERA_NTP_STR_KEYS:
            val = np.asarray(f[f"{group_path}/{key}"][()])
            if val.shape != () or val.dtype.kind not in "SUO":
                raise RuntimeError(
                    f"{source}: {group_path}/{key} must be a 0-d string/bytes scalar, "
                    f"got {val.shape} {val.dtype}"
                )
            timing[f"camera_ntp_{label}_{key}"] = val
    return timing


def _validate_target_batch(name: str, mats: np.ndarray, source: str | Path) -> None:
    """(T,4,4) finite pose batch whose rotation columns are unit within tolerance."""
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise RuntimeError(f"{source}: {name} must be (T,4,4), got {mats.shape}")
    _validate_finite(name, mats, source)
    for col in (0, 1):
        norms = np.linalg.norm(mats[:, :3, col], axis=1)
        bad = np.abs(norms - 1.0) > _ROT_COL_NORM_TOL
        if bad.any():
            t = int(np.argwhere(bad)[0][0])
            raise RuntimeError(
                f"{source}: {name} rotation column {col} is not unit at frame {t} "
                f"(norm={norms[t]:.6f}) -- corrupt target"
            )


def first_valid_frame(
    gripper_left: np.ndarray, gripper_right: np.ndarray, source: str | Path
) -> int:
    """First index where BOTH obs grippers are finite; leading NaNs are dropped.

    ``obs/gripper`` is NaN until the FC03 monitor's first reply. A NaN AFTER the
    first finite value cannot happen in a healthy take (the recorder holds the
    last reply), so it raises instead of being repaired.
    """
    finite = np.isfinite(gripper_left) & np.isfinite(gripper_right)
    if not finite.any():
        raise RuntimeError(f"{source}: obs/gripper never becomes finite (no FC03 reply)")
    t0 = int(np.argmax(finite))
    for name, g in (("obs/gripper/left", gripper_left), ("obs/gripper/right", gripper_right)):
        later = ~np.isfinite(g[t0:])
        if later.any():
            t_bad = t0 + int(np.argmax(later))
            raise RuntimeError(
                f"{source}: {name} is NaN at frame {t_bad}, after the first finite "
                f"frame {t0} -- corrupt take (mid-take NaN is never interpolated)"
            )
    return t0


def _longest_true_run(mask: np.ndarray) -> int:
    """Length of the longest contiguous True run in a one-dimensional mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or mask.size == 0:
        return 0
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    lengths = edges[1::2] - edges[::2]
    return int(lengths.max()) if lengths.size else 0


def validate_gripper_feedback(
    gripper_obs: Mapping[str, np.ndarray],
    gripper_act: Mapping[str, np.ndarray],
    t0: int,
    source: str | Path,
) -> None:
    """Reject commanded gripper motion with an effectively constant encoder.

    A stationary gripper is legitimate when the operator never changes its command.
    It is not legitimate when both open and close are each commanded for a sustained
    interval but gPO never moves by even two encoder counts. That pattern means the
    achieved-state channel is stale/broken and would corrupt observation.state.
    """
    for side in ("left", "right"):
        obs = np.asarray(gripper_obs[side][t0:], dtype=np.float64)
        act = np.asarray(gripper_act[side][t0:], dtype=np.float64)
        out_of_range = (obs < -1e-6) | (obs > 1.0 + 1e-6)
        if np.any(out_of_range):
            local_bad = int(np.flatnonzero(out_of_range)[0])
            raise RuntimeError(
                f"{source}: obs/gripper/{side}[{t0 + local_bad}]="
                f"{obs[local_bad]:.6f} is outside [0,1]"
            )
        closed = act >= GRIPPER_BINARY_THRESHOLD
        open_run = _longest_true_run(~closed)
        close_run = _longest_true_run(closed)
        response_range = float(np.ptp(obs))
        if (
            open_run >= _GRIPPER_MIN_COMMAND_RUN
            and close_run >= _GRIPPER_MIN_COMMAND_RUN
            and response_range < _GRIPPER_MIN_RESPONSE_RANGE
        ):
            raise RuntimeError(
                f"{source}: obs/gripper/{side} is stale or stuck: action commands "
                f"open for {open_run} consecutive frame(s) and close for "
                f"{close_run}, but achieved gPO range is only "
                f"{response_range:.6f} (< {_GRIPPER_MIN_RESPONSE_RANGE:.6f}, two "
                "encoder counts). Do not train on this take. Re-record with the "
                "fresh-status recorder and inspect obs/gripper_status provenance."
            )


def validate_gripper_status_provenance(
    raw: h5py.File,
    *,
    frame_count: int,
    fps: int,
    timestamp_ns: np.ndarray,
    gripper_obs: Mapping[str, np.ndarray],
    timing: dict | None,
    raw_schema: str,
    source: str | Path,
) -> dict[str, np.ndarray]:
    """Validate timestamped FC03 provenance for raw v4/v5 takes.

    V4 requires one distinct transaction per row. V5 permits an explicitly marked
    zero-order hold while the callback remains at most 2.5 frame periods old. In both
    schemas both arms are atomic, callback time precedes row commit/state read, and the
    achieved scalar exactly matches ``obs/gripper``. Returned arrays are copied into
    the processed timing sidecar for post-hoc latency audits.
    """
    groups = {
        side: f"obs/gripper_status/{side}" for side in ("left", "right")
    }
    present = {side: group in raw for side, group in groups.items()}
    if not any(present.values()):
        if raw_schema in _RAW_SCHEMAS_WITH_GRIPPER_PROVENANCE:
            raise RuntimeError(
                f"{source}: {raw_schema} take is missing obs/gripper_status for both "
                "arms; achieved-state provenance cannot be verified"
            )
        return {}
    if not all(present.values()):
        missing = [side for side, exists in present.items() if not exists]
        raise RuntimeError(
            f"{source}: partial obs/gripper_status; missing arm(s) {missing}"
        )

    max_age_ns = round(_GRIPPER_STATUS_MAX_AGE_PERIODS / float(fps) * 1e9)
    request_timeout_s = (
        _GRIPPER_STATUS_REQUEST_TIMEOUT_V5_S
        if raw_schema in _RAW_SCHEMAS_WITH_GRIPPER_REUSE
        else _GRIPPER_STATUS_REQUEST_TIMEOUT_V4_S
    )
    max_request_ns = round(request_timeout_s * 1e9)
    commit = np.asarray(timestamp_ns, dtype=np.int64)
    state_start = None
    if timing is not None and "state_read_start_wall_ns" in timing:
        state_start = np.asarray(timing["state_read_start_wall_ns"], dtype=np.int64)
    sidecar: dict[str, np.ndarray] = {}

    for side, group in groups.items():
        arrays: dict[str, np.ndarray] = {}
        for key in (
            "status_request_id",
            "request_send_wall_ns",
            "read_wall_ns",
            "actual",
        ):
            path = f"{group}/{key}"
            if path not in raw:
                raise RuntimeError(f"{source}: missing required {path}")
            arrays[key] = np.asarray(raw[path][()])

        for key in ("status_request_id", "request_send_wall_ns", "read_wall_ns"):
            arr = arrays[key]
            if arr.shape != (frame_count,) or arr.dtype != np.int64:
                raise RuntimeError(
                    f"{source}: {group}/{key} must be ({frame_count},) int64, got "
                    f"{arr.shape} {arr.dtype}"
                )
            if np.any(arr < 0 if key == "status_request_id" else arr <= 0):
                raise RuntimeError(f"{source}: {group}/{key} has invalid values")
            if frame_count > 1:
                delta = np.diff(arr)
                if raw_schema in _RAW_SCHEMAS_WITH_GRIPPER_REUSE:
                    if np.any(delta < 0):
                        raise RuntimeError(
                            f"{source}: {group}/{key} regresses in raw v5"
                        )
                elif np.any(delta <= 0):
                    raise RuntimeError(
                        f"{source}: {group}/{key} is not strictly increasing; cached "
                        "FC03 feedback was reused"
                    )

        if raw_schema in _RAW_SCHEMAS_WITH_GRIPPER_REUSE:
            reused_path = f"{group}/sample_reused"
            if reused_path not in raw:
                raise RuntimeError(f"{source}: raw v5 is missing required {reused_path}")
            reused = np.asarray(raw[reused_path][()])
            if reused.shape != (frame_count,) or reused.dtype.kind != "b":
                raise RuntimeError(
                    f"{source}: {reused_path} must be ({frame_count},) bool, got "
                    f"{reused.shape} {reused.dtype}"
                )
            if frame_count and bool(reused[0]):
                raise RuntimeError(
                    f"{source}: {reused_path}[0] is true; frame zero must use a "
                    "post-engage FC03 callback"
                )
            request_ids = arrays["status_request_id"]
            repeated = np.zeros(frame_count, dtype=bool)
            if frame_count > 1:
                repeated[1:] = np.diff(request_ids) == 0
            if not np.array_equal(reused, repeated):
                frame = int(np.flatnonzero(reused != repeated)[0])
                raise RuntimeError(
                    f"{source}: {reused_path}[{frame}]={bool(reused[frame])} does "
                    f"not match status_request_id reuse={bool(repeated[frame])}"
                )
            # A repeated id denotes the exact same sample, not merely another packet
            # with a coincidentally duplicated identifier. Its timing and value must be
            # bit-identical; advancing ids must advance both timestamps.
            if frame_count > 1:
                for key in ("request_send_wall_ns", "read_wall_ns"):
                    delta = np.diff(arrays[key])
                    if np.any(delta[repeated[1:]] != 0):
                        frame = int(np.flatnonzero(
                            repeated[1:] & (delta != 0)
                        )[0]) + 1
                        raise RuntimeError(
                            f"{source}: {group}/{key} changes at reused frame {frame}"
                        )
                    if np.any(delta[~repeated[1:]] <= 0):
                        frame = int(np.flatnonzero(
                            (~repeated[1:]) & (delta <= 0)
                        )[0]) + 1
                        raise RuntimeError(
                            f"{source}: {group}/{key} does not advance with the new "
                            f"status at frame {frame}"
                        )
        else:
            reused = np.zeros(frame_count, dtype=bool)

        request = arrays["request_send_wall_ns"]
        read = arrays["read_wall_ns"]
        request_latency = read - request
        if np.any(request_latency < 0) or np.any(request_latency > max_request_ns):
            frame = int(np.flatnonzero(
                (request_latency < 0) | (request_latency > max_request_ns)
            )[0])
            raise RuntimeError(
                f"{source}: {side} FC03 request latency at frame {frame} is "
                f"{request_latency[frame] / 1e6:.1f} ms (expected 0.."
                f"{max_request_ns / 1e6:.0f} ms)"
            )
        status_age = commit - read
        if np.any(status_age < 0) or np.any(status_age > max_age_ns):
            frame = int(np.flatnonzero(
                (status_age < 0) | (status_age > max_age_ns)
            )[0])
            raise RuntimeError(
                f"{source}: obs/gripper_status/{side} at frame {frame} is "
                f"{status_age[frame] / 1e6:.1f} ms old at commit (maximum "
                f"{max_age_ns / 1e6:.0f} ms); stale pre-episode FC03 reply"
            )
        if state_start is not None and np.any(read > state_start):
            frame = int(np.flatnonzero(read > state_start)[0])
            raise RuntimeError(
                f"{source}: {side} gripper status callback is later than state read "
                f"at frame {frame}"
            )

        actual = np.asarray(arrays["actual"], dtype=np.float32)
        if actual.shape != (frame_count,) or not np.all(np.isfinite(actual)):
            raise RuntimeError(
                f"{source}: {group}/actual must be finite ({frame_count},), got "
                f"{actual.shape}"
            )
        if raw_schema in _RAW_SCHEMAS_WITH_GRIPPER_REUSE and frame_count > 1:
            repeated_rows = reused[1:]
            if np.any(actual[1:][repeated_rows] != actual[:-1][repeated_rows]):
                local = int(np.flatnonzero(
                    repeated_rows & (actual[1:] != actual[:-1])
                )[0])
                frame = local + 1
                raise RuntimeError(
                    f"{source}: {group}/actual changes at reused frame {frame}"
                )
        obs = np.asarray(gripper_obs[side], dtype=np.float32)
        if not np.allclose(actual, obs, rtol=0.0, atol=1e-6):
            frame = int(np.flatnonzero(np.abs(actual - obs) > 1e-6)[0])
            raise RuntimeError(
                f"{source}: {group}/actual[{frame}]={actual[frame]:.6f} does not "
                f"match obs/gripper/{side}={obs[frame]:.6f}"
            )

        prefix = f"gripper_{side}_"
        sidecar[prefix + "status_request_id"] = arrays["status_request_id"]
        sidecar[prefix + "request_send_wall_ns"] = request
        sidecar[prefix + "read_wall_ns"] = read
        sidecar[prefix + "sample_reused"] = reused
        sidecar[prefix + "request_latency_ns"] = request_latency
        sidecar[prefix + "status_age_ns"] = status_age

    return sidecar


def load_action_targets(
    raw: h5py.File,
    t: int,
    *,
    pose_action_offset: int = 0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Recorded solver targets from ``t + pose_action_offset`` in their declared frame.

    Returns ``({"left": (4,4), "right": (4,4)}, head (4,4))``. RuntimeError on a
    missing dataset (named), a bad frame index, a wrong shape, or a non-finite /
    non-rotation transform.  A positive offset terminally repeats the final pose.
    The caller must use the same resolved action frame for the gripper commands;
    :func:`compute_frame_state_action` exposes ``action_t`` for that purpose.
    """
    if not isinstance(pose_action_offset, int) or pose_action_offset < 0:
        raise ValueError(
            f"pose_action_offset must be a non-negative integer, got {pose_action_offset!r}"
        )
    source = raw.filename
    for key in _REQUIRED_ACTION_TARGETS:
        if key not in raw:
            raise RuntimeError(
                f"{source}: missing required HDF5 dataset {key} -- re-record with the "
                "updated wbc_vr_robot.py recorder (debug files are never a substitute)"
            )
    out: dict[str, np.ndarray] = {}
    for name, key in (
        ("left", "action/eef/left"),
        ("right", "action/eef/right"),
        ("head", "action/head"),
    ):
        ds = raw[key]
        if ds.ndim != 3 or ds.shape[1:] != (4, 4):
            raise RuntimeError(f"{source}: {key} must be (T,4,4), got {ds.shape}")
        if not 0 <= t < ds.shape[0]:
            raise RuntimeError(
                f"{source}: frame index {t} out of range for {key} " f"(T={ds.shape[0]})"
            )
        action_t = min(t + pose_action_offset, ds.shape[0] - 1)
        mat = np.asarray(ds[action_t], dtype=np.float64)
        _validate_target_batch(key, mat[None], source)
        out[name] = mat
    return {"left": out["left"], "right": out["right"]}, out["head"]


def load_action_targets_world(
    raw: h5py.File,
    t: int,
    *,
    pose_action_offset: int = 0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Backward-compatible name for legacy callers of :func:`load_action_targets`."""
    return load_action_targets(raw, t, pose_action_offset=pose_action_offset)


def compute_frame_state_action(
    raw: h5py.File,
    t: int,
    fk: WBCPolicyFK,
    action_eef: Mapping[str, np.ndarray],
    action_head: np.ndarray,
    *,
    state_frame: str = "base",
    action_t: int | None = None,
    policy_action_schema: str = WBC_POLICY_ACTION_SCHEMA,
) -> tuple[np.ndarray, np.ndarray]:
    """One frame's state and schema-selected 29-D or 32-D action vectors.

    State: achieved FK poses of the measured ``obs/joint`` (base zero in q), in the
    ``state_frame`` frame (``"base"`` = egocentric default; ``"world"`` = composed
    to the engage-origin world via ``obs/base/pose``, for MoF checkpoints that
    declare ``mof_state_frame="world"``), + raw obs
    grippers + ``obs/base/pose`` (dims 29-31, world anchor, same in both frames).
    Action: ``mat_to_pos6d`` of the given targets in the episode's declared action
    frame (engage-world for WBC, current-base for joystick), verbatim from raw, plus
    binarized action grippers from the same raw action frame. ``action_t`` defaults
    to ``t``; a porter applying a future action offset must pass the resolved future
    index here so one 29-D label never mixes two command times.
    """
    source = raw.filename

    def at(key: str) -> np.ndarray:
        arr = np.asarray(raw[key][t], dtype=np.float64) if key in raw else None
        if arr is None:
            raise RuntimeError(f"{source}: missing required HDF5 dataset {key}")
        _validate_finite(f"{key}[{t}]", arr, source)
        return arr

    if action_t is None:
        action_t = t
    if isinstance(action_t, bool) or not isinstance(action_t, (int, np.integer)):
        raise ValueError(f"action_t must be an integer frame index, got {action_t!r}")
    action_t = int(action_t)

    def action_at(key: str) -> np.ndarray:
        if key not in raw:
            raise RuntimeError(f"{source}: missing required HDF5 dataset {key}")
        ds = raw[key]
        if not 0 <= action_t < ds.shape[0]:
            raise RuntimeError(
                f"{source}: action frame index {action_t} out of range for {key} "
                f"(T={ds.shape[0]})"
            )
        arr = np.asarray(ds[action_t], dtype=np.float64)
        _validate_finite(f"{key}[{action_t}]", arr, source)
        return arr

    poses = fk.base_frame_poses(
        at("obs/joint/torso"),
        at("obs/joint/left_arm"),
        at("obs/joint/right_arm"),
        at("obs/joint/head"),
    )
    base_pose = at("obs/base/pose")
    if base_pose.shape != (3,):
        raise RuntimeError(f"{source}: obs/base/pose[{t}] must be (3,), got {base_pose.shape}")
    grip_obs_l, grip_obs_r = float(at("obs/gripper/left")), float(at("obs/gripper/right"))
    grip_act_l = float(
        float(action_at("action/gripper/left")) >= GRIPPER_BINARY_THRESHOLD
    )
    grip_act_r = float(
        float(action_at("action/gripper/right")) >= GRIPPER_BINARY_THRESHOLD
    )

    state = build_state_vector(poses, base_pose, grip_obs_l, grip_obs_r, state_frame=state_frame)
    components: list[np.ndarray | list[float]] = [
        mat_to_pos6d(action_eef["left"]),
        [grip_act_l],
        mat_to_pos6d(action_eef["right"]),
        [grip_act_r],
        mat_to_pos6d(action_head),
    ]
    if policy_action_schema == JOYSTICK_POLICY_ACTION_SCHEMA:
        chassis_intent = action_at("action/chassis/intent_body").reshape(-1)
        if chassis_intent.shape != (3,):
            raise RuntimeError(
                f"{source}: action/chassis/intent_body[{action_t}] must be (3,), "
                f"got {chassis_intent.shape}"
            )
        components.append(chassis_intent)
    try:
        action_axes = action_axes_for_policy_schema(policy_action_schema)
    except ValueError as exc:
        raise RuntimeError(f"{source}: {exc}") from exc
    action = np.concatenate(components).astype(np.float32)
    if action.shape != (len(action_axes),):
        raise RuntimeError(
            f"{source}: frame {t} built action {action.shape}, "
            f"expected ({len(action_axes)},) for {policy_action_schema}"
        )
    return state, action


def load_and_validate_episode(hdf5_path: Path, fps: int) -> dict:
    """Read + cross-validate every array the port needs; shared with verify().

    Returns arrays plus ``t0`` (first frame with finite obs grippers). All frame
    counts must agree; the required policy datasets must exist; the record
    cadence must match ``fps`` at the median.
    """
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        raise RuntimeError(f"episode file not found: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        if not bool(f.attrs.get("complete", False)):
            raise RuntimeError(
                f"{hdf5_path}: HDF5 complete flag is missing/false; this is an "
                "interrupted or legacy-unverified take, not policy training input"
            )
        if "n_frames" not in f.attrs:
            raise RuntimeError(
                f"{hdf5_path}: complete take is missing HDF5 n_frames attribute"
            )
        try:
            declared_frames = int(f.attrs["n_frames"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"{hdf5_path}: invalid HDF5 n_frames={f.attrs['n_frames']!r}"
            ) from exc
        rgb = read_required_array(f, "obs/images/head_left_rgb", hdf5_path)
        wrist_rgb = read_required_array(f, "obs/images/left_wrist_rgb", hdf5_path)
        # Right wrist is OPTIONAL: takes recorded before the second camera only carry
        # left_wrist_rgb. NOTE in those legacy files that key is physically the RIGHT
        # arm's camera -- nothing in the file distinguishes them, so it ports as left.
        right_wrist_rgb = (
            read_required_array(f, "obs/images/right_wrist_rgb", hdf5_path)
            if "obs/images/right_wrist_rgb" in f
            else None
        )
        # Image-policy training does not consume depth. Current six-eye recordings
        # intentionally omit the bandwidth-heavy SDK depth and reconstruct it offline
        # with FoundationStereo, so preserve legacy depth when present but never invent
        # or require it here.
        depth = (
            read_required_array(f, "obs/images/head_depth", hdf5_path)
            if "obs/images/head_depth" in f
            else None
        )
        intrinsic = read_required_array(f, "obs/images/intrinsic", hdf5_path)
        joints_obs = {
            grp: read_required_array(f, f"obs/joint/{grp}", hdf5_path)
            for grp in ("torso", "left_arm", "right_arm", "head")
        }
        base_pose = read_required_array(f, "obs/base/pose", hdf5_path)
        base_pose_contract = _inspect_open_base_pose_contract(f, hdf5_path)
        policy_action_contract = _inspect_open_policy_action_contract(f, hdf5_path)
        gripper_obs = {
            side: read_required_array(f, f"obs/gripper/{side}", hdf5_path, check_finite=False)
            for side in ("left", "right")
        }
        gripper_act = {
            side: read_required_array(f, f"action/gripper/{side}", hdf5_path)
            for side in ("left", "right")
        }
        eef_left = read_required_array(f, "action/eef/left", hdf5_path)
        eef_right = read_required_array(f, "action/eef/right", hdf5_path)
        head_target = read_required_array(f, "action/head", hdf5_path)
        chassis_intent = (
            read_required_array(f, "action/chassis/intent_body", hdf5_path)
            if policy_action_contract["policy_action_schema"]
            == JOYSTICK_POLICY_ACTION_SCHEMA
            else None
        )
        timestamp_ns = read_required_array(f, "timestamp_ns", hdf5_path)
        timing = load_episode_timing(f, hdf5_path, rgb.shape[0])
        camera_alignment_contract = _inspect_open_camera_alignment_contract(
            f,
            hdf5_path,
            timing,
            raw_schema=base_pose_contract["raw_schema"],
        )
        gripper_status_timing = validate_gripper_status_provenance(
            f,
            frame_count=rgb.shape[0],
            fps=fps,
            timestamp_ns=timestamp_ns,
            gripper_obs=gripper_obs,
            timing=timing,
            raw_schema=base_pose_contract["raw_schema"],
            source=hdf5_path,
        )
        if gripper_status_timing:
            timing = dict(timing or {})
            timing.update(gripper_status_timing)

    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise RuntimeError(
            f"{hdf5_path}: expected (T,H,W,3) uint8 head RGB, got " f"{rgb.shape} {rgb.dtype}"
        )
    for _name, _arr in (
        ("left_wrist_rgb", wrist_rgb),
        *((("right_wrist_rgb", right_wrist_rgb),) if right_wrist_rgb is not None else ()),
    ):
        if _arr.ndim != 4 or _arr.shape[-1] != 3 or _arr.dtype != np.uint8:
            raise RuntimeError(
                f"{hdf5_path}: expected (T,H,W,3) uint8 {_name}, got "
                f"{_arr.shape} {_arr.dtype}"
            )
    if depth is not None:
        if depth.ndim != 3 or depth.dtype != np.uint16:
            raise RuntimeError(
                f"{hdf5_path}: expected (T,H,W) uint16 depth, got "
                f"{depth.shape} {depth.dtype}"
            )
        if depth.shape[1:] != rgb.shape[1:3]:
            raise RuntimeError(
                f"{hdf5_path}: depth/rgb spatial mismatch: "
                f"{depth.shape[1:]} vs {rgb.shape[1:3]}"
            )
    # This recorder stores the head intrinsic ONCE (static (3,3)), not per frame.
    if intrinsic.shape != (3, 3):
        raise RuntimeError(
            f"{hdf5_path}: expected STATIC (3,3) obs/images/intrinsic (wbc_vr_robot "
            f"recorder), got {intrinsic.shape}"
        )

    frame_count = rgb.shape[0]
    if declared_frames != frame_count:
        raise RuntimeError(
            f"{hdf5_path}: HDF5 n_frames={declared_frames} disagrees with "
            f"head_left_rgb={frame_count}"
        )
    if frame_count < 2:
        raise RuntimeError(
            f"{hdf5_path}: policy episode has only {frame_count} frame(s); at least 2 "
            "are required to verify cadence and form a state/action sequence"
        )
    checks: list[tuple[str, np.ndarray]] = [
        ("obs/images/left_wrist_rgb", wrist_rgb),
        *(
            (("obs/images/right_wrist_rgb", right_wrist_rgb),)
            if right_wrist_rgb is not None
            else ()
        ),
        *(
            (("obs/images/head_depth", depth),)
            if depth is not None
            else ()
        ),
        ("obs/base/pose", base_pose),
        ("timestamp_ns", timestamp_ns),
        ("action/eef/left", eef_left),
        ("action/eef/right", eef_right),
        ("action/head", head_target),
        *(
            (("action/chassis/intent_body", chassis_intent),)
            if chassis_intent is not None
            else ()
        ),
    ]
    checks += [(f"obs/joint/{g}", a) for g, a in joints_obs.items()]
    checks += [(f"obs/gripper/{s}", a) for s, a in gripper_obs.items()]
    checks += [(f"action/gripper/{s}", a) for s, a in gripper_act.items()]
    for name, arr in checks:
        if arr.shape[0] != frame_count:
            raise RuntimeError(
                f"{hdf5_path}: frame counts disagree: head_left_rgb={frame_count}, "
                f"{name}={arr.shape[0]}"
            )
    expected_joint_dims = {"torso": 3, "left_arm": 7, "right_arm": 7, "head": 3}
    for grp, dim in expected_joint_dims.items():
        if joints_obs[grp].ndim != 2 or joints_obs[grp].shape[1] != dim:
            raise RuntimeError(
                f"{hdf5_path}: obs/joint/{grp} must be (T,{dim}), got " f"{joints_obs[grp].shape}"
            )
    if base_pose.ndim != 2 or base_pose.shape[1] != 3:
        raise RuntimeError(f"{hdf5_path}: obs/base/pose must be (T,3), got {base_pose.shape}")
    for name, arr in (
        ("action/eef/left", eef_left),
        ("action/eef/right", eef_right),
        ("action/head", head_target),
    ):
        _validate_target_batch(name, arr, hdf5_path)
    if chassis_intent is not None and (
        chassis_intent.ndim != 2 or chassis_intent.shape[1] != 3
    ):
        raise RuntimeError(
            f"{hdf5_path}: action/chassis/intent_body must be (T,3), got "
            f"{chassis_intent.shape}"
        )

    # Record cadence: the recorder throttles to --record-rate (= dataset fps), but
    # hold windows leave gaps. The median tick must match fps; gaps are warned.
    ts = timestamp_ns.astype(np.int64)
    if frame_count > 1:
        dt = np.diff(ts) / 1e9
        if np.any(dt <= 0):
            raise RuntimeError(f"{hdf5_path}: timestamp_ns is not strictly increasing")
        median_dt = float(np.median(dt))
        if abs(median_dt - 1.0 / fps) > 0.15 / fps:
            raise RuntimeError(
                f"{hdf5_path}: median frame dt {median_dt * 1000:.1f} ms does not match "
                f"--fps {fps} ({1000.0 / fps:.1f} ms); re-record or fix --fps"
            )
        gaps = int(np.sum(dt > 1.5 / fps))
        if gaps:
            logging.warning(
                "%s: %d hold gap(s) > %.0f ms in the 10 Hz stream",
                hdf5_path.name,
                gaps,
                1500.0 / fps,
            )

    t0 = first_valid_frame(gripper_obs["left"], gripper_obs["right"], hdf5_path)
    if t0 > 0:
        logging.info("%s: dropping %d leading frame(s) with NaN obs/gripper", hdf5_path.name, t0)
    validate_gripper_feedback(gripper_obs, gripper_act, t0, hdf5_path)

    return {
        "source_path": hdf5_path,
        "rgb": rgb,
        "wrist_rgb": wrist_rgb,          # LEFT arm
        "right_wrist_rgb": right_wrist_rgb,  # RIGHT arm, None on single-wrist takes
        "depth": depth,
        "intrinsic": intrinsic,
        "base_pose": base_pose,
        "base_pose_source": base_pose_contract["base_pose_source"],
        "base_pose_contract": base_pose_contract,
        "policy_action_contract": policy_action_contract,
        "policy_action_schema": policy_action_contract["policy_action_schema"],
        "action_axes": policy_action_contract["action_axes"],
        "camera_alignment_contract": camera_alignment_contract,
        "frame_count": frame_count,
        "t0": t0,
        "timestamp_ns": ts,
        "timing": timing,
    }


# ---------------------------------------------------------------------------
# Optional ENV-state conditioning: SceneDiff changed-object positions
# ---------------------------------------------------------------------------
# With --positions-dir, the porter attaches a CONSTANT observation.environment_state
# (object_nums*3,) to every frame of each episode: the SceneDiff-detected object 3D
# positions in the engage-origin WORLD frame -- the SAME frame as the action targets and
# observation.state[29:32] base anchor -- ARRANGED by the episode's task order. Part A
# (scene_diff/scripts/extract_object_positions.py, via run_wbc_pos_condition.sh) writes,
# per episode, ``positions`` (object_nums,3) in a fixed SIZE order plus the resolved task
# ``order`` (grasp->box=src, release->cloth=dst), under
# ``<positions-dir>/<source>/episode_<N>.npz`` (source in {raw, recovery}; the subdir keys
# the file so raw episode_<N> and recovery/<N> never collide). This porter reorders by that
# ``order`` into [s1_src, s1_dst, ...] and broadcasts the ONE vector to every frame (the
# single-stage "same condition throughout" requirement, forward-compatible with N grasp
# cycles). Position-extraction quality (object count == object_nums; per-object n_valid)
# is gated UPSTREAM by Part A, which only emits an npz for passing episodes; here we
# defensively check shape/finiteness/permutation/frame and fail fast on a missing file.
OBS_ENV_STATE_KEY = "observation.environment_state"
OBS_TASK_SEQUENCE_KEY = "observation.task_sequence"
PROGRESS_INDEX_KEY = "progress_index"
ACTIVE_TASK_ID_KEY = "active_task_id"
STRUCTURED_ENV_STATE_SHAPE = (3, 2, 3)
STRUCTURED_STAGE_NAMES = ("drawer", "plate", "jar")
SUPPORTED_TASK_SEQUENCES = ((0, 1, 2), (0, 2, 1))
DEFAULT_OBJECT_NUMS = 2
# Stage roles in the arranged env-state: [s1_src, s1_dst, s2_src, s2_dst, ...].
STAGE_ROLES = ("src", "dst")


def env_state_axes(object_nums: int) -> list[str]:
    """Axis names for the (object_nums*3,) ARRANGED env-state.

    For the standard task (object_nums a multiple of 2 = n_stages grasp cycles) the slots
    are named by stage and role (s1_src_x .. sN_dst_z); otherwise a generic obj{i}_{x,y,z}
    naming is used. Mirrors port_dexmate_hdf5.env_state_axes.
    """
    n_stages, rem = divmod(object_nums, len(STAGE_ROLES))
    if rem == 0:
        return [
            f"s{s + 1}_{role}_{a}" for s in range(n_stages) for role in STAGE_ROLES for a in "xyz"
        ]
    return [f"obj{i}_{a}" for i in range(object_nums) for a in "xyz"]


def build_env_state_vector(positions: np.ndarray, order: list[int]) -> np.ndarray:
    """Reorder the size-ordered ``positions`` (M,3) by ``order`` and flatten to (M*3,)."""
    return positions[order].reshape(-1).astype(np.float32)


def load_episode_positions(
    positions_dir: Path, source: str, raw_index: int, object_nums: int
) -> np.ndarray:
    """``<positions-dir>/<source>/episode_<N>.npz`` -> ARRANGED env-state (object_nums*3,).

    Keyed by ``(source, raw_index)`` -- the source subdir ({raw, recovery}) disambiguates a
    raw ``episode_<N>`` from a ``recovery/<N>`` that share an index. Reads the size-ordered
    ``positions`` (object_nums,3) and the resolved ``order`` written by Part A and returns
    ``positions[order]`` flattened. Fails fast if the file is missing, a key is absent,
    ``positions`` is not (object_nums,3), any value is non-finite, ``order`` is not a
    permutation of range(object_nums), or the npz frame is not the WORLD frame this port
    conditions in (guards against pointing --positions-dir at the tabletop positions).
    """
    path = Path(positions_dir) / source / f"episode_{raw_index}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"positions file for {source}/episode_{raw_index} not found: {path} "
            "(run run_wbc_pos_condition.sh / scene_diff Part A first, or drop --positions-dir)"
        )
    with np.load(path, allow_pickle=True) as data:
        for key in ("positions", "order"):
            if key not in data:
                raise ValueError(f"{path}: missing '{key}' array (regenerate with Part A)")
        positions = np.asarray(data["positions"], dtype=np.float32)
        order = [int(x) for x in np.asarray(data["order"]).reshape(-1)]
        frame = str(data["frame"]) if "frame" in data else "world"
    if frame != "world":
        raise ValueError(
            f"{path}: env-state npz frame is {frame!r}, expected 'world' -- the WBC port "
            "conditions in the engage-origin world frame (is --positions-dir the tabletop set?)"
        )
    if positions.shape != (object_nums, 3):
        raise ValueError(
            f"{path}: positions shape {positions.shape} != ({object_nums}, 3) "
            "(object count must match --object_nums; gated in extract_object_positions.py)"
        )
    _validate_finite("positions", positions, path)
    if sorted(order) != list(range(object_nums)):
        raise ValueError(f"{path}: order {order} is not a permutation of range({object_nums})")
    vector = build_env_state_vector(positions, order)
    if vector.shape != (object_nums * 3,):
        raise ValueError(f"{path}: assembled env_state {vector.shape} != ({object_nums * 3},)")
    return vector


def validate_all_episode_positions(
    positions_dir: Path, work: list[dict], object_nums: int
) -> dict[tuple[str, int], np.ndarray]:
    """Load + validate EVERY work item's env-state up front; fail before any dataset write.

    Returns ``{(source, raw_index): env_state_vector}``. Aggregates ALL per-episode failures
    into one error so a single run surfaces every missing/bad npz at once.
    """
    env_state_by_key: dict[tuple[str, int], np.ndarray] = {}
    failures: list[str] = []
    for item in work:
        key = (item["source"], item["raw_index"])
        try:
            env_state_by_key[key] = load_episode_positions(
                positions_dir, item["source"], item["raw_index"], object_nums
            )
        except Exception as exc:
            failures.append(f"- {item['source']}/episode_{item['raw_index']}: {exc}")
    if failures:
        raise RuntimeError(
            "Changed-object position validation failed; no dataset was written:\n"
            + "\n".join(failures)
        )
    return env_state_by_key


def load_structured_env_state_from_raw(
    hdf5_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read semantic conditions, task plan, and ordinal progress labels."""
    with h5py.File(hdf5_path, "r") as raw:
        condition = read_required_array(
            raw, "task/env_state", hdf5_path
        ).astype(np.float32)
        task_sequence = read_required_array(
            raw, "task/task_sequence", hdf5_path
        )
        progress_index = read_required_array(
            raw, PROGRESS_INDEX_KEY, hdf5_path
        )
        active_task_id = read_required_array(
            raw, ACTIVE_TASK_ID_KEY, hdf5_path
        )
        frame_count = int(raw["obs/images/head_left_rgb"].shape[0])
    if condition.shape != STRUCTURED_ENV_STATE_SHAPE:
        raise RuntimeError(
            f"{hdf5_path}: task/env_state must be "
            f"{STRUCTURED_ENV_STATE_SHAPE}, got {condition.shape}"
        )
    if task_sequence.shape != (3,) or task_sequence.dtype != np.int64:
        raise RuntimeError(
            f"{hdf5_path}: task/task_sequence must be (3,) int64, "
            f"got {task_sequence.shape} {task_sequence.dtype}"
        )
    sequence = tuple(int(value) for value in task_sequence)
    if sequence not in SUPPORTED_TASK_SEQUENCES:
        raise RuntimeError(
            f"{hdf5_path}: unsupported task/task_sequence {sequence}"
        )
    if (
        progress_index.shape != (frame_count,)
        or progress_index.dtype != np.int64
    ):
        raise RuntimeError(
            f"{hdf5_path}: progress_index must be ({frame_count},) int64, "
            f"got {progress_index.shape} {progress_index.dtype}"
        )
    if progress_index[0] != 0 or progress_index[-1] != 2:
        raise RuntimeError(
            f"{hdf5_path}: accepted progress_index must begin at 0 and end at 2"
        )
    differences = np.diff(progress_index)
    if np.any((differences < 0) | (differences > 1)):
        raise RuntimeError(
            f"{hdf5_path}: progress_index must be monotonic 0->1->2"
        )
    if (
        active_task_id.shape != (frame_count,)
        or active_task_id.dtype != np.int64
    ):
        raise RuntimeError(
            f"{hdf5_path}: active_task_id must be ({frame_count},) int64"
        )
    if not np.array_equal(active_task_id, task_sequence[progress_index]):
        raise RuntimeError(
            f"{hdf5_path}: active_task_id != task_sequence[progress_index]"
        )
    return condition, task_sequence, progress_index, active_task_id


def validate_all_structured_env_state(
    work: list[dict],
) -> dict[
    tuple[str, int],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Fail before writing if any raw episode violates the structured contract."""
    result = {}
    failures = []
    for item in work:
        key = (item["source"], item["raw_index"])
        try:
            result[key] = load_structured_env_state_from_raw(item["path"])
        except Exception as exc:
            failures.append(f"- {item['path']}: {exc}")
    if failures:
        raise RuntimeError(
            "Structured raw env-state validation failed; no dataset was written:\n"
            + "\n".join(failures)
        )
    return result


def build_features(
    resize_h: int,
    resize_w: int,
    object_nums: int | None = None,
    *,
    has_right_wrist: bool = True,
    structured_env_state: bool = False,
    action_axes: list[str] | tuple[str, ...] = ACTION_AXES,
) -> dict:
    """LeRobot feature schema.

    Wrist cameras are keyed by ARM (``left``/``right``), matching the raw
    ``obs/images/*`` schema. ``has_right_wrist=False`` drops the right camera for
    single-wrist source takes -- declaring a feature the frames never fill would
    make LeRobot reject every add_frame.
    """
    features = {
        "observation.images.head_rgb": {
            "dtype": "video",
            "shape": (resize_h, resize_w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.left_wrist_rgb": {
            "dtype": "video",
            "shape": (resize_h, resize_w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_AXES),),
            "names": {"axes": list(STATE_AXES)},
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_axes),),
            "names": {"axes": list(action_axes)},
        },
    }
    if has_right_wrist:
        features["observation.images.right_wrist_rgb"] = {
            "dtype": "video",
            "shape": (resize_h, resize_w, 3),
            "names": ["height", "width", "channels"],
        }
    # Optional constant global conditioning: SceneDiff object positions (ENV state). This
    # is float32 (NOT an image/video dtype), so no policy auto-loads it as a VISUAL input;
    # the train command wires it into input_features as type ENV (diffusion consumes
    # observation.environment_state natively). Single-stage -> no pos_condition_mask.
    if structured_env_state:
        features[OBS_ENV_STATE_KEY] = {
            "dtype": "float32",
            "shape": STRUCTURED_ENV_STATE_SHAPE,
            "names": None,
        }
        features[OBS_TASK_SEQUENCE_KEY] = {
            "dtype": "int64",
            "shape": (3,),
            "names": {"axes": ["progress0", "progress1", "progress2"]},
        }
        features[PROGRESS_INDEX_KEY] = {
            "dtype": "int64",
            "shape": (1,),
            "names": {"axes": ["progress"]},
        }
        features[ACTIVE_TASK_ID_KEY] = {
            "dtype": "int64",
            "shape": (1,),
            "names": {"axes": ["task_id"]},
        }
    elif object_nums is not None:
        axes = env_state_axes(object_nums)
        features[OBS_ENV_STATE_KEY] = {
            "dtype": "float32",
            "shape": (len(axes),),
            "names": {"axes": axes},
        }
    return features


def detect_right_wrist_presence(work: list[dict]) -> bool:
    """Whether the source takes carry the right wrist camera; all must agree.

    A LeRobot dataset's feature schema is fixed at creation, so a mixed batch would
    either declare ``observation.images.right_wrist_rgb`` that some frames cannot fill
    (``add_frame`` rejects the frame) or silently drop the second camera from the takes
    that do have it. Refuse instead and make the caller port the two sets separately.
    """
    with_right: list[str] = []
    without_right: list[str] = []
    for item in work:
        with h5py.File(item["path"], "r") as f:
            bucket = with_right if "obs/images/right_wrist_rgb" in f else without_right
            bucket.append(str(item["path"]))
    if with_right and without_right:
        raise RuntimeError(
            "source episodes disagree on the right wrist camera, so one LeRobot feature "
            "schema cannot describe them; port the two sets separately.\n"
            f"  {len(with_right)} with right_wrist_rgb, e.g. {with_right[0]}\n"
            f"  {len(without_right)} without, e.g. {without_right[0]}"
        )
    return bool(with_right)


def _sidecar_path(variant_root: Path, kind: str, episode_idx: int) -> Path:
    return Path(variant_root) / "debug" / kind / f"episode_{episode_idx:06d}.npz"


def _metadata_sidecar_path(variant_root: Path, episode_idx: int) -> Path:
    return (
        Path(variant_root)
        / "debug"
        / "provenance"
        / f"episode_{episode_idx:06d}.hdf5"
    )


def write_metadata_sidecar(raw_path: Path, output_path: Path) -> None:
    """Copy raw static ``/meta`` exactly, without duplicating per-frame payloads."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw_path, "r") as source, h5py.File(output_path, "w") as target:
        target.attrs["schema"] = "omniteleop_raw_metadata_sidecar/v1"
        target.attrs["raw_filename"] = Path(raw_path).name
        if "meta" in source:
            source.copy("meta", target)
        else:
            target.create_group("meta")


def _hdf5_leaf_arrays(group: h5py.Group) -> dict[str, np.ndarray]:
    """Flatten HDF5 datasets below ``group`` for exact sidecar verification."""
    out: dict[str, np.ndarray] = {}

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = np.asarray(obj[()])

    group.visititems(visitor)
    return out


def timing_sidecar_arrays(ep: dict, start: int, end: int) -> dict[str, np.ndarray]:
    """``debug/timing`` payload for the kept window ``[start, end)``.

    The record-tick wall clock always; the capture stamps + once-per-run NTP and
    camera-NTP calibrations ride along when the take has them (timing-aware recorder).
    Per-frame (ndim 1) arrays are trimmed to the kept window like every other
    stream; 0-d calibration scalars are copied verbatim.
    """
    arrays: dict[str, np.ndarray] = {"timestamp_ns": ep["timestamp_ns"][start:end]}
    if ep["timing"] is not None:
        for key, arr in ep["timing"].items():
            arrays[key] = arr[start:end] if arr.ndim == 1 else arr
    return arrays


def _scaled_intrinsic(
    intrinsic: np.ndarray, raw_hw: tuple[int, int], out_hw: tuple[int, int]
) -> np.ndarray:
    """Static (3,3) intrinsic rescaled from the raw frame size to the output size."""
    k = np.asarray(intrinsic, dtype=np.float32).copy()
    k[0, :] *= out_hw[1] / raw_hw[1]  # fx, 0, cx scale with width
    k[1, :] *= out_hw[0] / raw_hw[0]  # 0, fy, cy scale with height
    return k


def add_episode(
    dataset,
    hdf5_path: Path,
    fk: WBCPolicyFK,
    resize_h: int,
    resize_w: int,
    task: str,
    variant_root: Path,
    episode_idx: int,
    fps: int,
    state_frame: str = "base",
    *,
    policy_action_schema: str = WBC_POLICY_ACTION_SCHEMA,
    pose_action_offset: int = 0,
    trim: tuple[int, int] | None = None,
    env_state_vec: np.ndarray | None = None,
    task_sequence: np.ndarray | None = None,
    progress_index: np.ndarray | None = None,
    active_task_id: np.ndarray | None = None,
) -> tuple[int, int]:
    """Port one raw episode; returns ``(num_frames_before, num_frames_after)``.

    ``num_frames_before`` is the raw take length; ``num_frames_after`` is the number
    of frames written -- after the leading-NaN-gripper drop (``trim is None``) or after
    the ABSOLUTE INCLUSIVE ``trim=(frame_start, frame_end)`` recovery segment.
    ``env_state_vec`` is constant per episode and may be the legacy flat
    SceneDiff vector or the simulator-native semantic (3,2,3) condition.
    ``pose_action_offset`` is the legacy name for a coherent future ACTION offset:
    EEF, head, and gripper labels all come from the same future raw frame while the
    observation remains at the current frame.
    ``task_sequence`` is categorical policy input; ``progress_index`` is stage-
    head supervision; ``active_task_id`` is a redundant checked target.
    """
    if not task:
        raise ValueError("task must be a non-empty string")
    ep = load_and_validate_episode(hdf5_path, fps)
    if ep["policy_action_schema"] != policy_action_schema:
        raise RuntimeError(
            f"{hdf5_path}: policy action schema changed after preflight: "
            f"{ep['policy_action_schema']!r} != {policy_action_schema!r}"
        )
    rgb, wrist_rgb, depth = ep["rgb"], ep["wrist_rgb"], ep["depth"]
    right_wrist_rgb = ep["right_wrist_rgb"]
    frame_count = ep["frame_count"]
    start, end = resolve_episode_window(frame_count, ep["t0"], trim, hdf5_path)
    kept = end - start
    structured_fields = (task_sequence, progress_index, active_task_id)
    if any(value is not None for value in structured_fields):
        if any(value is None for value in structured_fields):
            raise RuntimeError(
                f"{hdf5_path}: structured condition requires task_sequence, "
                "progress_index, and active_task_id together"
            )
        if env_state_vec is None or env_state_vec.shape != STRUCTURED_ENV_STATE_SHAPE:
            raise RuntimeError(
                f"{hdf5_path}: structured labels require env_state "
                f"{STRUCTURED_ENV_STATE_SHAPE}"
            )
        task_sequence = np.asarray(task_sequence)
        progress_index = np.asarray(progress_index)
        active_task_id = np.asarray(active_task_id)
        if task_sequence.shape != (3,) or task_sequence.dtype != np.int64:
            raise RuntimeError(f"{hdf5_path}: task_sequence must be (3,) int64")
        if (
            progress_index.shape != (frame_count,)
            or progress_index.dtype != np.int64
        ):
            raise RuntimeError(
                f"{hdf5_path}: progress_index must be ({frame_count},) int64"
            )
        if (
            active_task_id.shape != (frame_count,)
            or active_task_id.dtype != np.int64
        ):
            raise RuntimeError(
                f"{hdf5_path}: active_task_id must be ({frame_count},) int64"
            )

    depth_resized = (
        np.empty((kept, resize_h, resize_w), dtype=np.uint16)
        if depth is not None
        else None
    )
    extrinsic = np.empty((kept, 4, 4), dtype=np.float32)
    base_extrinsic = np.empty((kept, 4, 4), dtype=np.float32)

    with h5py.File(hdf5_path, "r") as raw:
        for i, t in enumerate(range(start, end)):
            action_t = min(t + pose_action_offset, frame_count - 1)
            action_eef, action_head = load_action_targets(
                raw, t, pose_action_offset=pose_action_offset
            )
            state, action = compute_frame_state_action(
                raw,
                t,
                fk,
                action_eef,
                action_head,
                state_frame=state_frame,
                action_t=action_t,
                policy_action_schema=policy_action_schema,
            )

            # Calib sidecar: recover base_T_zed and world_T_zed from the state head
            # block, which is base_T_zed in "base" frame and world_T_zed in "world"
            # frame -- so derive both via the odometry base pose regardless of frame.
            head_mat = pos6d_to_mat(state[20:29])
            world_T_base = base_pose_to_mat(state[29:32])
            if state_frame == "world":
                world_t_zed = head_mat
                base_t_zed = np.linalg.inv(world_T_base) @ head_mat
            else:
                base_t_zed = head_mat
                world_t_zed = world_T_base @ head_mat
            base_extrinsic[i] = base_t_zed.astype(np.float32)
            extrinsic[i] = world_t_zed.astype(np.float32)

            rgb_resized = cv2.resize(rgb[t], (resize_w, resize_h), interpolation=cv2.INTER_AREA)
            wrist_resized = cv2.resize(
                wrist_rgb[t], (resize_w, resize_h), interpolation=cv2.INTER_AREA
            )
            right_wrist_resized = (
                None
                if right_wrist_rgb is None
                else cv2.resize(
                    right_wrist_rgb[t], (resize_w, resize_h), interpolation=cv2.INTER_AREA
                )
            )
            if depth_resized is not None:
                # INTER_NEAREST keeps uint16 millimeter depth exact across edges.
                depth_resized[i] = cv2.resize(
                    depth[t], (resize_w, resize_h), interpolation=cv2.INTER_NEAREST
                )

            frame = {
                "observation.images.head_rgb": rgb_resized,
                "observation.images.left_wrist_rgb": wrist_resized,
                "observation.state": state,
                "action": action,
                "task": task,
            }
            if right_wrist_resized is not None:
                frame["observation.images.right_wrist_rgb"] = right_wrist_resized
            if env_state_vec is not None:
                frame[OBS_ENV_STATE_KEY] = env_state_vec
            if progress_index is not None:
                frame[OBS_TASK_SEQUENCE_KEY] = task_sequence
                frame[PROGRESS_INDEX_KEY] = np.asarray(
                    [progress_index[t]], dtype=np.int64
                )
                frame[ACTIVE_TASK_ID_KEY] = np.asarray(
                    [active_task_id[t]], dtype=np.int64
                )
            dataset.add_frame(frame)

    dataset.save_episode()

    depth_path: Path | None = None
    if depth_resized is not None:
        depth_path = _sidecar_path(variant_root, "depth", episode_idx)
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(depth_path, depth=depth_resized)

    intrinsic_resized = _scaled_intrinsic(ep["intrinsic"], rgb.shape[1:3], (resize_h, resize_w))
    calib_path = _sidecar_path(variant_root, "calib", episode_idx)
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        calib_path,
        extrinsic=extrinsic,
        base_extrinsic=base_extrinsic,
        intrinsic=np.broadcast_to(intrinsic_resized, (kept, 3, 3)).copy(),
    )

    timing_path = _sidecar_path(variant_root, "timing", episode_idx)
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(timing_path, **timing_sidecar_arrays(ep, start, end))

    metadata_path = _metadata_sidecar_path(variant_root, episode_idx)
    write_metadata_sidecar(ep["source_path"], metadata_path)

    logging.info(
        "Ported %s: %d frames [%d,%d) of %d, sidecars %s / %s / %s / %s",
        hdf5_path.name,
        kept,
        start,
        end,
        frame_count,
        depth_path.name if depth_path is not None else "depth=absent (stereo-only raw)",
        calib_path.name,
        timing_path.name,
        metadata_path.name,
    )
    return frame_count, kept


def build_work_list(raw_dir: Path, include_recovery_data: Path | None) -> list[dict]:
    """Ordered port plan: raw episodes (numeric-sorted), then -- if requested -- the

    trimmed recovery episodes (numeric-sorted, APPENDED last).

    Each item is ``{"source": "raw"|"recovery", "path": Path, "raw_index": int,
    "trim": (frame_start, frame_end) | None}``; ``processed_data_index`` is the item's
    position in the returned list. Enforces a strict 1:1 match between the recovery
    ``episode_<N>.hdf5`` files and the ``trim.csv`` rows.
    """
    work = [
        {"source": "raw", "path": p, "raw_index": episode_index_from_path(p), "trim": None}
        for p in find_episode_files(raw_dir)
    ]
    if include_recovery_data is None:
        return work

    recovery_dir = Path(include_recovery_data)
    if not recovery_dir.is_dir():
        raise RuntimeError(f"--include_recovery_data dir not found: {recovery_dir}")
    if recovery_dir.resolve() == Path(raw_dir).resolve():
        raise RuntimeError(
            f"--include_recovery_data must differ from --raw-dir (both {recovery_dir})"
        )
    trim_spec = load_trim_spec(recovery_dir)
    recovery_files = find_episode_files(recovery_dir)
    file_indices = {episode_index_from_path(p) for p in recovery_files}
    missing_rows = sorted(file_indices - set(trim_spec))
    missing_files = sorted(set(trim_spec) - file_indices)
    if missing_rows or missing_files:
        raise RuntimeError(
            f"{recovery_dir / 'trim.csv'} does not match the recovery episodes 1:1: "
            f"episode files without a trim row={missing_rows}, "
            f"trim rows without a file={missing_files}"
        )
    for p in recovery_files:
        idx = episode_index_from_path(p)
        work.append({"source": "recovery", "path": p, "raw_index": idx, "trim": trim_spec[idx]})
    return work


def _parse_episode_ref(ref: str, source: Path, lineno: int) -> tuple[str, int]:
    """``'61'`` -> ``('raw', 61)``; ``'recovery/56'`` -> ``('recovery', 56)``.

    The key matches the ``(source, raw_index)`` of a :func:`build_work_list` item.
    """
    if "/" in ref:
        prefix, _, rest = ref.partition("/")
        if prefix not in ("raw", "recovery") or not rest.isdigit():
            raise RuntimeError(
                f"{source}:{lineno}: bad episode ref {ref!r}; expected '<int>' or 'recovery/<int>'"
            )
        return prefix, int(rest)
    if not ref.isdigit():
        raise RuntimeError(
            f"{source}:{lineno}: bad episode ref {ref!r}; expected '<int>' or 'recovery/<int>'"
        )
    return "raw", int(ref)


def parse_split_csv(split_csv: Path) -> dict[tuple[str, int], str]:
    """Parse ``split.csv`` into ``{(source, raw_index): split}``.

    Format (one header row, then one row per NON-``train`` episode)::

        episode, split
        61, test
        recovery/56, test

    ``episode`` is a bare raw index ``<N>`` (source ``"raw"``) or ``recovery/<N>``
    (source ``"recovery"``); ``split`` is one of :data:`ALL_SPLITS`. Episodes NOT
    listed default to :data:`DEFAULT_SPLIT` (see :func:`assign_splits`). Raises on a
    missing file, a malformed row, an unknown split, or a duplicate episode. Only
    READ, never modified.
    """
    split_csv = Path(split_csv)
    if not split_csv.exists():
        raise RuntimeError(f"split csv not found: {split_csv}")
    spec: dict[tuple[str, int], str] = {}
    with open(split_csv, newline="") as handle:
        for lineno, row in enumerate(csv.reader(handle), start=1):
            cells = [c.strip() for c in row]
            if not cells or all(c == "" for c in cells):
                continue  # blank line
            if cells[0].lower() == "episode":
                continue  # header row (e.g. 'episode, split')
            if len(cells) != 2:
                raise RuntimeError(
                    f"{split_csv}:{lineno}: expected 2 columns 'episode, split', "
                    f"got {len(cells)}: {row}"
                )
            episode_ref, split = cells
            key = _parse_episode_ref(episode_ref, split_csv, lineno)
            if split not in ALL_SPLITS:
                raise RuntimeError(
                    f"{split_csv}:{lineno}: unknown split {split!r}; expected one of {ALL_SPLITS}"
                )
            if key in spec:
                raise RuntimeError(f"{split_csv}:{lineno}: duplicate episode {episode_ref!r}")
            spec[key] = split
    return spec


def assign_splits(
    work: list[dict],
    split_map: Mapping[tuple[str, int], str],
    include_recovery_data: Path | None = None,
    default_split: str = DEFAULT_SPLIT,
) -> dict[str, list[dict]]:
    """Partition the ordered work list into ``{split: [items...]}`` per ``split_map``.

    Items not named in ``split_map`` go to ``default_split``. Within
    each split the raw-then-recovery numeric order of ``work`` is preserved, and empty
    splits are dropped. Raises if ``split_map`` names an episode absent from ``work``
    (a stale split.csv -- e.g. a ``recovery/<N>`` row without ``--include_recovery_data``).
    """
    if default_split not in ALL_SPLITS:
        raise ValueError(
            f"default_split must be one of {ALL_SPLITS}, got {default_split!r}"
        )
    keys = {(w["source"], w["raw_index"]) for w in work}
    unknown = sorted(set(split_map) - keys)
    if unknown:
        refs = ", ".join(f"recovery/{i}" if s == "recovery" else str(i) for s, i in unknown)
        hint = (
            ""
            if include_recovery_data is not None
            else " (recovery episodes require --include_recovery_data)"
        )
        raise RuntimeError(f"split csv references episodes not in the port plan: {refs}{hint}")
    buckets: dict[str, list[dict]] = {s: [] for s in ALL_SPLITS}
    for w in work:
        buckets[split_map.get((w["source"], w["raw_index"]), default_split)].append(w)
    return {s: buckets[s] for s in ALL_SPLITS if buckets[s]}


def write_log_index(log_index_path: Path, rows: list[dict]) -> None:
    """Write the raw->processed episode index map as CSV (with a header).

    Columns: ``split``, ``source`` (raw/recovery), ``raw_data_index`` (N in
    ``episode_<N>.hdf5``), ``processed_data_index`` (0-based LeRobot episode index
    WITHIN its split), ``num_frames_before`` (raw take length), ``num_frames_after``
    (frames written after the drop/trim).
    """
    log_index_path = Path(log_index_path)
    log_index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_index_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "source",
                "raw_data_index",
                "processed_data_index",
                "num_frames_before",
                "num_frames_after",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["split"],
                    r["source"],
                    r["raw_index"],
                    r["processed_index"],
                    r["num_before"],
                    r["num_after"],
                ]
            )
    logging.info("Wrote index log %s (%d episode(s))", log_index_path, len(rows))


def _write_meta(
    variant_root: Path,
    split: str,
    state_frame: str,
    fps: int,
    resize_h: int,
    resize_w: int,
    task: str,
    base_pose_source: str,
    camera_alignment_contract: dict,
    policy_action_schema: str = WBC_POLICY_ACTION_SCHEMA,
    pose_action_offset: int = 0,
    object_nums: int | None = None,
    structured_env_state: bool = False,
) -> None:
    """Write the ``dexmate_meta.json`` sidecar describing this split's schema."""
    action_axes = action_axes_for_policy_schema(policy_action_schema)
    joystick = policy_action_schema == JOYSTICK_POLICY_ACTION_SCHEMA
    meta = {
        "schema": policy_action_schema,
        "policy_action_schema": policy_action_schema,
        "split": split,
        "state_axes": list(STATE_AXES),
        "action_axes": action_axes,
        # "base": egocentric achieved FK (the default). "world": composed to
        # engage-origin world so state shares the action frame, for MoF checkpoints
        # declaring mof_state_frame="world".
        "state_frame": state_frame,
        "action_frame": (
            "current_base (verbatim joystick IK targets; chassis intent in body frame)"
            if joystick
            else "world (verbatim ik.solve targets, engage-origin)"
        ),
        "base_pose_source": base_pose_source,
        "base_pose_semantics": (
            "observation.state[29:32] is raw obs/base/pose and anchors all "
            "world-frame state/calibration composition"
        ),
        "camera_alignment_mode": camera_alignment_contract["mode"],
        "camera_alignment_clock_domain": camera_alignment_contract["clock_domain"],
        "camera_alignment_max_abs_skew_ns": camera_alignment_contract[
            "max_abs_skew_ns"
        ],
        "camera_alignment_max_camera_age_ns": camera_alignment_contract[
            "max_camera_age_ns"
        ],
        "camera_alignment_verified": bool(camera_alignment_contract["verified"]),
        "action_offset_frames": pose_action_offset,
        "pose_action_offset_frames": pose_action_offset,
        "action_offset_semantics": "all action components shifted together",
        "skip_normalization_dims": SKIP_NORMALIZATION_DIMS,
        "fps": fps,
        "resize_h": resize_h,
        "resize_w": resize_w,
        "task": task,
        "robot_type": ROBOT_TYPE,
        "gripper_binary_threshold": GRIPPER_BINARY_THRESHOLD,
    }
    if joystick:
        meta.update({
            "chassis_intent_dims": [29, 30, 31],
            "chassis_intent_frame": "current_base_body",
            "chassis_intent_units": ["m/s", "m/s", "rad/s"],
            "chassis_intent_stage": "post_mask_projection_pre_controller",
            "chassis_intent_temporal_semantics": "zero_order_hold",
        })
    # Constant ENV-state conditioning (only when --positions-dir was used).
    # Env-state is world-frame and per-episode CONSTANT.
    if structured_env_state:
        meta.update(
            {
                "env_state_key": OBS_ENV_STATE_KEY,
                "env_state_shape": list(STRUCTURED_ENV_STATE_SHAPE),
                "env_state_task_names": list(STRUCTURED_STAGE_NAMES),
                "env_state_role_names": ["src", "dst"],
                "env_state_coordinate_names": ["x", "y", "z"],
                "env_state_frame": "engage-origin world",
                "env_state_normalization": (
                    "shared_xyz_across_stage_and_role"
                ),
                "task_sequence_key": OBS_TASK_SEQUENCE_KEY,
                "task_sequence_semantics": (
                    "categorical_progress_to_task_input_no_normalization"
                ),
                "progress_index_key": PROGRESS_INDEX_KEY,
                "progress_index_semantics": "stage_head_target_only",
                "active_task_id_key": ACTIVE_TASK_ID_KEY,
                "active_task_id_semantics": "derived_target_only",
            }
        )
    elif object_nums is not None:
        meta["env_state_key"] = OBS_ENV_STATE_KEY
        meta["object_nums"] = object_nums
        meta["env_state_axes"] = env_state_axes(object_nums)
        meta["env_state_frame"] = "world (engage-origin, same as action targets)"
    (variant_root / "dexmate_meta.json").write_text(json.dumps(meta, indent=2))


def verify(
    dataset,
    first_item: dict,
    fk: WBCPolicyFK,
    resize_h: int,
    resize_w: int,
    fps: int,
    state_frame: str,
    variant_root: Path,
    split: str,
    env_state_vec: np.ndarray | None = None,
    task_sequence: np.ndarray | None = None,
    progress_index: np.ndarray | None = None,
    active_task_id: np.ndarray | None = None,
    pose_action_offset: int = 0,
    policy_action_schema: str = WBC_POLICY_ACTION_SCHEMA,
) -> None:
    """Re-open a written split dataset and cross-check frame 0 against a fresh build.

    Confirms ``observation.state``/``action`` for the split's first episode first kept
    frame match an independent recomputation (SAME FK model); the action EEF/head blocks
    are the VERBATIM world targets (no composition); grippers are binary; depth/calib are
    NOT parquet features; and the depth/calib/timing sidecars round-trip. Needs lerobot --
    only the real conversion calls it (the dexmate-only unit tests never do).
    """
    meta = dataset.meta
    logging.info(
        "verify[%s]: fps=%s episodes=%s frames=%s robot=%s",
        split,
        meta.fps,
        meta.total_episodes,
        meta.total_frames,
        meta.robot_type,
    )

    for forbidden in (
        "observation.images.head_depth",
        "observation.images.depth",
        "observation.extrinsic",
        "observation.intrinsic",
        "observation.base_extrinsic",
    ):
        if forbidden in meta.features:
            raise RuntimeError(
                f"verify[{split}]: {forbidden} must NOT be a dataset feature -- it belongs "
                "in the debug sidecar under <variant_root>/debug/"
            )

    item = dataset[0]
    head_rgb = item["observation.images.head_rgb"]
    wrist_rgb = item["observation.images.left_wrist_rgb"]
    state = item["observation.state"].cpu().numpy()
    action = item["action"].cpu().numpy()
    cams = [("head_rgb", head_rgb), ("left_wrist_rgb", wrist_rgb)]
    # Right wrist is present iff the schema declared it (single-wrist sources omit it).
    if "observation.images.right_wrist_rgb" in meta.features:
        cams.append(
            ("right_wrist_rgb", item["observation.images.right_wrist_rgb"])
        )
    for cam_key, cam in cams:
        if tuple(cam.shape) != (3, resize_h, resize_w):
            raise RuntimeError(
                f"verify[{split}]: {cam_key} shape {tuple(cam.shape)} != "
                f"(3, {resize_h}, {resize_w})"
            )
    if state.shape != (len(STATE_AXES),):
        raise RuntimeError(f"verify[{split}]: state shape {state.shape} != ({len(STATE_AXES)},)")
    action_axes = action_axes_for_policy_schema(policy_action_schema)
    if action.shape != (len(action_axes),):
        raise RuntimeError(
            f"verify[{split}]: action shape {action.shape} != ({len(action_axes)},)"
        )

    # ENV-state conditioning: the declared feature and the first frame's stored vector must
    # match the loaded positions exactly iff positions were requested (feature must be ABSENT
    # otherwise). Constancy across frames follows from broadcasting the one vector in add_episode.
    if env_state_vec is not None:
        if OBS_ENV_STATE_KEY not in meta.features:
            raise RuntimeError(
                f"verify[{split}]: {OBS_ENV_STATE_KEY} missing though positions were requested"
            )
        env_item = item[OBS_ENV_STATE_KEY].cpu().numpy()
        if env_item.shape != env_state_vec.shape:
            raise RuntimeError(
                f"verify[{split}]: env_state shape {env_item.shape} != {env_state_vec.shape}"
            )
        if not np.allclose(env_item, env_state_vec, atol=1e-5):
            raise RuntimeError(f"verify[{split}]: stored env_state[0] != loaded positions vector")
    elif OBS_ENV_STATE_KEY in meta.features:
        raise RuntimeError(
            f"verify[{split}]: {OBS_ENV_STATE_KEY} present but positions were not requested"
        )
    if progress_index is not None:
        for key in (
            OBS_TASK_SEQUENCE_KEY,
            PROGRESS_INDEX_KEY,
            ACTIVE_TASK_ID_KEY,
        ):
            if key not in meta.features:
                raise RuntimeError(
                    f"verify[{split}]: structured feature {key} is missing"
                )
        stored_sequence = np.asarray(
            item[OBS_TASK_SEQUENCE_KEY].cpu().numpy()
        ).reshape(-1)
        if stored_sequence.shape != (3,) or not np.array_equal(
            stored_sequence, task_sequence
        ):
            raise RuntimeError(
                f"verify[{split}]: stored task_sequence differs from raw"
            )
        stored_progress = np.asarray(
            item[PROGRESS_INDEX_KEY].cpu().numpy()
        ).reshape(-1)
        if stored_progress.shape != (1,) or int(stored_progress[0]) != int(
            progress_index[0]
        ):
            raise RuntimeError(
                f"verify[{split}]: stored progress_index[0] differs from raw"
            )
        stored_task = np.asarray(
            item[ACTIVE_TASK_ID_KEY].cpu().numpy()
        ).reshape(-1)
        if stored_task.shape != (1,) or int(stored_task[0]) != int(
            active_task_id[0]
        ):
            raise RuntimeError(
                f"verify[{split}]: stored active_task_id[0] differs from raw"
            )
    else:
        forbidden = (
            OBS_TASK_SEQUENCE_KEY,
            PROGRESS_INDEX_KEY,
            ACTIVE_TASK_ID_KEY,
        )
        if any(key in meta.features for key in forbidden):
            raise RuntimeError(
                f"verify[{split}]: structured labels present outside raw mode"
            )

    # Recompute the first stored frame independently (same FK) and pull the raw verbatim
    # targets to confirm the action is a straight copy (no base composition).
    ep = load_and_validate_episode(first_item["path"], fps)
    start, end = resolve_episode_window(
        ep["frame_count"], ep["t0"], first_item["trim"], first_item["path"]
    )
    with h5py.File(first_item["path"], "r") as raw:
        action_start = min(start + pose_action_offset, ep["frame_count"] - 1)
        action_eef, action_head = load_action_targets(
            raw, start, pose_action_offset=pose_action_offset
        )
        exp_state, exp_action = compute_frame_state_action(
            raw,
            start,
            fk,
            action_eef,
            action_head,
            state_frame=state_frame,
            action_t=action_start,
            policy_action_schema=policy_action_schema,
        )
        raw_left = mat_to_pos6d(
            np.asarray(raw["action/eef/left"][action_start], dtype=np.float64)
        )
        raw_head = mat_to_pos6d(
            np.asarray(raw["action/head"][action_start], dtype=np.float64)
        )
        raw_grip_left = float(
            float(raw["action/gripper/left"][action_start])
            >= GRIPPER_BINARY_THRESHOLD
        )
        raw_grip_right = float(
            float(raw["action/gripper/right"][action_start])
            >= GRIPPER_BINARY_THRESHOLD
        )
        raw_base = np.asarray(raw["obs/base/pose"][start], dtype=np.float64)
        raw_chassis_intent = (
            np.asarray(
                raw["action/chassis/intent_body"][action_start],
                dtype=np.float64,
            )
            if policy_action_schema == JOYSTICK_POLICY_ACTION_SCHEMA
            else None
        )
        depth_start = (
            np.asarray(raw["obs/images/head_depth"][start])
            if "obs/images/head_depth" in raw
            else None
        )
    if not np.allclose(state, exp_state, atol=1e-4):
        raise RuntimeError(f"verify[{split}]: stored state[0] != recompute at frame {start}")
    if not np.allclose(action, exp_action, atol=1e-4):
        raise RuntimeError(f"verify[{split}]: stored action[0] != recompute at frame {start}")
    if not np.allclose(action[:9], raw_left, atol=1e-4):
        raise RuntimeError(f"verify[{split}]: action left block is not the verbatim world target")
    if not np.allclose(action[20:29], raw_head, atol=1e-4):
        raise RuntimeError(f"verify[{split}]: action head block is not the verbatim world target")
    if float(action[9]) != raw_grip_left or float(action[19]) != raw_grip_right:
        raise RuntimeError(
            f"verify[{split}]: action grippers do not come from raw action frame "
            f"{action_start}"
        )
    if raw_chassis_intent is not None and not np.allclose(
        action[29:32], raw_chassis_intent, atol=1e-7
    ):
        raise RuntimeError(
            f"verify[{split}]: joystick chassis intent is not the exact raw "
            f"action frame {action_start}"
        )
    if not np.allclose(state[29:32], raw_base, atol=1e-5):
        raise RuntimeError(f"verify[{split}]: state base dims != obs/base/pose[{start}]")
    for idx in GRIPPER_DIMS:
        if float(action[idx]) not in (0.0, 1.0):
            raise RuntimeError(
                f"verify[{split}]: action gripper dim {idx} is not binary "
                f"({float(action[idx])})"
            )

    # Optional SDK-depth sidecar roundtrip (episode 0, frame 0 == window start).
    # Stereo-only raw takes deliberately have neither raw SDK depth nor this sidecar.
    depth_path = _sidecar_path(variant_root, "depth", 0)
    depth_stack: np.ndarray | None = None
    if depth_start is None:
        if depth_path.exists():
            raise RuntimeError(
                f"verify[{split}]: stereo-only raw unexpectedly produced depth sidecar "
                f"{depth_path}"
            )
    else:
        if not depth_path.exists():
            raise RuntimeError(f"verify[{split}]: depth sidecar missing: {depth_path}")
        with np.load(depth_path) as data:
            depth_stack = data["depth"]
        if depth_stack.dtype != np.uint16 or depth_stack.shape[1:] != (resize_h, resize_w):
            raise RuntimeError(
                f"verify[{split}]: depth sidecar {depth_stack.dtype} "
                f"{depth_stack.shape[1:]} != uint16 ({resize_h}, {resize_w})"
            )
        expected_depth = cv2.resize(
            depth_start, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST
        )
        if not np.array_equal(depth_stack[0], expected_depth):
            raise RuntimeError(f"verify[{split}]: depth sidecar[0] roundtrip mismatch")

    # Calib sidecar: keys/shapes + the static intrinsic rescale.
    calib_path = _sidecar_path(variant_root, "calib", 0)
    if not calib_path.exists():
        raise RuntimeError(f"verify[{split}]: calib sidecar missing: {calib_path}")
    with np.load(calib_path) as data:
        extrinsic_sc, base_extrinsic_sc, intrinsic_sc = (
            data["extrinsic"],
            data["base_extrinsic"],
            data["intrinsic"],
        )
    if extrinsic_sc.shape[1:] != (4, 4) or base_extrinsic_sc.shape[1:] != (4, 4):
        raise RuntimeError(
            f"verify[{split}]: calib extrinsics not (T,4,4): "
            f"{extrinsic_sc.shape}, {base_extrinsic_sc.shape}"
        )
    if intrinsic_sc.shape[1:] != (3, 3):
        raise RuntimeError(f"verify[{split}]: calib intrinsic not (T,3,3): {intrinsic_sc.shape}")
    expected_intrinsic = _scaled_intrinsic(
        ep["intrinsic"], ep["rgb"].shape[1:3], (resize_h, resize_w)
    )
    if not np.allclose(intrinsic_sc[0], expected_intrinsic, atol=1e-4):
        raise RuntimeError(f"verify[{split}]: calib intrinsic[0] != rescaled static intrinsic")

    # Timing sidecar: the record-tick wall clock must round-trip exactly; the
    # capture-stamp/NTP keys must be present iff the raw take carries them.
    timing_path = _sidecar_path(variant_root, "timing", 0)
    if not timing_path.exists():
        raise RuntimeError(f"verify[{split}]: timing sidecar missing: {timing_path}")
    with np.load(timing_path) as data:
        timing_keys = set(data.keys())
        ts_sc = data["timestamp_ns"]
    if not np.array_equal(ts_sc, ep["timestamp_ns"][start:end]):
        raise RuntimeError(f"verify[{split}]: timing sidecar timestamp_ns roundtrip mismatch")
    expected_timing = set() if ep["timing"] is None else set(ep["timing"])
    if timing_keys != expected_timing | {"timestamp_ns"}:
        raise RuntimeError(
            f"verify[{split}]: timing sidecar keys {sorted(timing_keys)} != raw take's "
            f"{sorted(expected_timing | {'timestamp_ns'})}"
        )

    metadata_path = _metadata_sidecar_path(variant_root, 0)
    if not metadata_path.exists():
        raise RuntimeError(f"verify[{split}]: provenance sidecar missing: {metadata_path}")
    with h5py.File(first_item["path"], "r") as raw, h5py.File(metadata_path, "r") as sidecar:
        raw_meta = _hdf5_leaf_arrays(raw["meta"]) if "meta" in raw else {}
        sidecar_meta = _hdf5_leaf_arrays(sidecar["meta"])
        if set(raw_meta) != set(sidecar_meta):
            raise RuntimeError(
                f"verify[{split}]: provenance sidecar meta keys differ from raw: "
                f"{sorted(sidecar_meta)} != {sorted(raw_meta)}"
            )
        for key in raw_meta:
            if not np.array_equal(raw_meta[key], sidecar_meta[key]):
                raise RuntimeError(
                    f"verify[{split}]: provenance sidecar meta/{key} mismatch"
                )

    logging.info(
        "verify[%s] OK: head_rgb=%s wrist_rgb=%s (%s) state=%s action=%s depth=%s "
        "calib=extrinsic%s+base%s+intrinsic%s timing=%s provenance=%s",
        split,
        tuple(head_rgb.shape),
        tuple(wrist_rgb.shape),
        "left+right"
        if "observation.images.right_wrist_rgb" in meta.features
        else "left only",
        tuple(state.shape),
        tuple(action.shape),
        tuple(depth_stack.shape) if depth_stack is not None else "absent",
        tuple(extrinsic_sc.shape),
        tuple(base_extrinsic_sc.shape),
        tuple(intrinsic_sc.shape),
        sorted(timing_keys),
        metadata_path.name,
    )


def apply_shared_xyz_env_state_stats(dataset) -> None:
    """Replace element-wise (3,2,3) stats with one shared statistic per XYZ."""
    from lerobot.datasets.compute_stats import get_feature_stats  # noqa: PLC0415
    from lerobot.datasets.io_utils import write_stats  # noqa: PLC0415

    values = np.asarray(dataset.hf_dataset[OBS_ENV_STATE_KEY], dtype=np.float32)
    if values.ndim != 4 or tuple(values.shape[1:]) != STRUCTURED_ENV_STATE_SHAPE:
        raise RuntimeError(
            f"processed {OBS_ENV_STATE_KEY} must be (T,3,2,3), got "
            f"{values.shape}"
        )
    shared = get_feature_stats(
        values.reshape(-1, 3),
        axis=0,
        keepdims=False,
    )
    dataset.meta.stats[OBS_ENV_STATE_KEY] = {
        name: (
            value
            if name == "count"
            else np.broadcast_to(
                np.asarray(value).reshape(1, 1, 3),
                STRUCTURED_ENV_STATE_SHAPE,
            ).copy()
        )
        for name, value in shared.items()
    }
    write_stats(dataset.meta.stats, dataset.root)


def convert_dataset(
    raw_dir: Path,
    repo_id: str,
    fps: int,
    resize_h: int,
    resize_w: int,
    task: str,
    root: Path,
    overwrite: bool,
    state_frame: str = "base",
    include_recovery_data: Path | None = None,
    log_index: Path | None = None,
    split_map: Mapping[tuple[str, int], str] | None = None,
    positions_dir: Path | None = None,
    object_nums: int = DEFAULT_OBJECT_NUMS,
    structured_env_state_from_raw: bool = False,
    default_split: str = DEFAULT_SPLIT,
    pose_action_offset: int = 0,
) -> dict:
    """Full conversion (needs lerobot + tqdm -- run in the dexmate_lerobot env).

    Splits the episodes per ``split_map`` (from split.csv) into
    ``<root>/<split>/<repo_id>/`` datasets (unlisted -> ``default_split``); one tqdm
    bar spans every episode across splits, and each split is ``verify()``-ed after it is
    written. Returns ``{split: LeRobotDataset}``.

    When ``positions_dir`` is set, every ported episode also carries a constant
    ``observation.environment_state`` (object_nums*3,) from
    ``<positions_dir>/<source>/episode_<N>.npz`` -- pre-validated for ALL episodes before
    any dataset is created/overwritten (fail fast, keyed by ``(source, raw_index)``).
    """
    from lerobot.datasets import LeRobotDataset, recompute_stats  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415

    if not isinstance(pose_action_offset, int) or pose_action_offset < 0:
        raise ValueError(
            f"pose_action_offset must be a non-negative integer, got {pose_action_offset!r}"
        )
    if structured_env_state_from_raw and positions_dir is not None:
        raise ValueError(
            "--structured-env-state-from-raw and --positions-dir are mutually "
            "exclusive; simulation ground truth must remain authoritative"
        )
    work = build_work_list(raw_dir, include_recovery_data)
    # State dims 29:32 and every world-frame composition depend on this sensor. A
    # mixed-source dataset would present one feature with two physical meanings.
    # Validate before latency audits and, critically, before any output is removed.
    base_pose_source = validate_work_base_pose_contract(work)
    policy_action_contract = validate_work_policy_action_contract(work)
    policy_action_schema = str(policy_action_contract["policy_action_schema"])
    action_axes = list(policy_action_contract["action_axes"])
    if (
        policy_action_schema == JOYSTICK_POLICY_ACTION_SCHEMA
        and state_frame != "base"
    ):
        raise ValueError(
            "joystick/current-base policy actions require --state-frame base; a "
            "world-frame achieved-pose state would silently mix coordinate systems"
        )
    camera_alignment_contract = validate_work_camera_alignment_contract(work)
    splits = assign_splits(
        work,
        split_map or {},
        include_recovery_data,
        default_split=default_split,
    )
    n_recovery = sum(item["source"] == "recovery" for item in work)
    logging.info(
        "Port plan: %d episode(s) = %d raw + %d recovery -> splits {%s} "
        "(state_frame=%s, action_schema=%s, base_pose_source=%s, camera_alignment=%s)",
        len(work),
        len(work) - n_recovery,
        n_recovery,
        ", ".join(f"{s}:{len(v)}" for s, v in splits.items()),
        state_frame,
        policy_action_schema,
        base_pose_source,
        camera_alignment_contract["mode"],
    )

    # Mandatory and deliberately before any output path is removed or created. The
    # zarr and MoF sibling porters call this same shared gate.
    audit_work_list(work)

    # Fixed per dataset, so decide once across ALL takes before any split is created.
    has_right_wrist = detect_right_wrist_presence(work)
    if has_right_wrist:
        logging.info("Wrist cameras: left + right (both arms)")
    else:
        # Loud on purpose: pre-two-wrist takes stored the RIGHT arm's camera under
        # left_wrist_rgb, and nothing in the file distinguishes them, so this port
        # necessarily labels that footage as the left arm.
        logging.warning(
            "Wrist cameras: left only -- these are single-wrist takes, so no "
            "observation.images.right_wrist_rgb is written. If they predate the "
            "second camera, their left_wrist_rgb is PHYSICALLY THE RIGHT ARM and is "
            "being labelled left; do not mix this dataset with two-wrist takes."
        )

    # ENV-state conditioning: validate EVERY episode's positions npz UP FRONT, before the
    # pre-existing-dir check below removes anything -- a missing/bad npz must abort with no
    # data written or deleted. Keyed by (source, raw_index) via the positions subdir.
    env_state_by_key: dict[tuple[str, int], np.ndarray] = {}
    structured_by_key: dict[
        tuple[str, int],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    if positions_dir is not None:
        env_state_by_key = validate_all_episode_positions(positions_dir, work, object_nums)
        logging.info(
            "ENV-state conditioning ON: object_nums=%d, loaded positions for %d "
            "episode(s) from %s",
            object_nums,
            len(env_state_by_key),
            positions_dir,
        )
    elif structured_env_state_from_raw:
        structured_by_key = validate_all_structured_env_state(work)
        logging.info(
            "Structured simulator ENV-state ON: loaded semantic (3,2,3) "
            "condition, task plan, and progress targets for %d episode(s)",
            len(structured_by_key),
        )

    # Fail fast on pre-existing split dirs BEFORE any heavy work / partial writes.
    variant_roots = {s: Path(root) / s / repo_id for s in splits}
    for variant_root in variant_roots.values():
        if variant_root.exists():
            if not overwrite:
                raise RuntimeError(f"{variant_root} already exists. Use --overwrite to replace it.")
            logging.info("Removing existing %s (--overwrite)", variant_root)
            shutil.rmtree(variant_root)

    fk = WBCPolicyFK()
    datasets: dict[str, object] = {}
    log_rows: list[dict] = []
    grand_total = 0
    with tqdm(total=len(work), desc="Porting episodes", unit="ep") as pbar:
        for split, items in splits.items():
            variant_root = variant_roots[split]
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                features=build_features(
                    resize_h,
                    resize_w,
                    object_nums if positions_dir is not None else None,
                    has_right_wrist=has_right_wrist,
                    structured_env_state=structured_env_state_from_raw,
                    action_axes=action_axes,
                ),
                root=variant_root,
                robot_type=ROBOT_TYPE,
                use_videos=True,
            )
            split_total = 0
            for processed_idx, item in enumerate(items):
                key = (item["source"], item["raw_index"])
                if structured_env_state_from_raw:
                    (
                        env_vec,
                        task_sequence,
                        progress_index,
                        active_task_id,
                    ) = structured_by_key[key]
                else:
                    env_vec = (
                        env_state_by_key[key]
                        if positions_dir is not None
                        else None
                    )
                    task_sequence = None
                    progress_index = None
                    active_task_id = None
                num_before, num_after = add_episode(
                    dataset,
                    item["path"],
                    fk,
                    resize_h,
                    resize_w,
                    task,
                    variant_root=variant_root,
                    episode_idx=processed_idx,
                    fps=fps,
                    state_frame=state_frame,
                    policy_action_schema=policy_action_schema,
                    pose_action_offset=pose_action_offset,
                    trim=item["trim"],
                    env_state_vec=env_vec,
                    task_sequence=task_sequence,
                    progress_index=progress_index,
                    active_task_id=active_task_id,
                )
                split_total += num_after
                log_rows.append(
                    {
                        "split": split,
                        "source": item["source"],
                        "raw_index": item["raw_index"],
                        "processed_index": processed_idx,
                        "num_before": num_before,
                        "num_after": num_after,
                    }
                )
                pbar.update(1)
            dataset.finalize()
            dataset = recompute_stats(dataset, skip_image_video=True)
            if structured_env_state_from_raw:
                apply_shared_xyz_env_state_stats(dataset)
            _write_meta(
                variant_root,
                split,
                state_frame,
                fps,
                resize_h,
                resize_w,
                task,
                base_pose_source,
                camera_alignment_contract,
                policy_action_schema=policy_action_schema,
                pose_action_offset=pose_action_offset,
                object_nums=object_nums if positions_dir is not None else None,
                structured_env_state=structured_env_state_from_raw,
            )
            first = items[0]
            first_key = (first["source"], first["raw_index"])
            if structured_env_state_from_raw:
                (
                    first_env,
                    first_sequence,
                    first_progress,
                    first_task_id,
                ) = structured_by_key[first_key]
            else:
                first_env = (
                    env_state_by_key[first_key]
                    if positions_dir is not None
                    else None
                )
                first_sequence = None
                first_progress = None
                first_task_id = None
            verify(
                dataset,
                first,
                fk,
                resize_h,
                resize_w,
                fps,
                state_frame,
                variant_root,
                split,
                env_state_vec=first_env,
                task_sequence=first_sequence,
                progress_index=first_progress,
                active_task_id=first_task_id,
                pose_action_offset=pose_action_offset,
                policy_action_schema=policy_action_schema,
            )
            datasets[split] = dataset
            grand_total += split_total
            logging.info(
                "Split %s finalized: %d episode(s), %d frames", split, len(items), split_total
            )

    logging.info(
        "Done: %d split(s), %d episode(s), %d frames total", len(splits), len(work), grand_total
    )
    if log_index is not None:
        write_log_index(log_index, log_rows)
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--resize-h", type=int, default=DEFAULT_RESIZE_H)
    parser.add_argument("--resize-w", type=int, default=DEFAULT_RESIZE_W)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=f"dataset repo-id (default {DEFAULT_REPO_ID}, or "
        f"{DEFAULT_REPO_ID}_world with --world-state).",
    )
    parser.add_argument(
        "--world-state",
        action="store_true",
        help="emit observation.state EEF/head in the WORLD frame "
        "(composed via obs/base/pose) instead of the base-frame default. "
        "Needed by MoF checkpoints trained with mof_state_frame=\"world\"; "
        "ordinary base-frame checkpoints (including MoF's default) do not want it.",
    )
    parser.add_argument(
        "--include-recovery-data",
        "--include_recovery_data",
        dest="include_recovery_data",
        type=Path,
        default=None,
        help="directory of extra recovery episode_<N>.hdf5 takes to "
        "APPEND after the raw episodes. Requires <dir>/trim.csv "
        "('episode, frame_start, frame_end' rows; ABSOLUTE INCLUSIVE "
        "raw-frame indices, frame_end=-1 -> last frame). Every "
        "recovery file must have exactly one trim row and vice-versa. "
        "The recovery dir is only read, never modified.",
    )
    parser.add_argument(
        "--log-index",
        "--log_index",
        dest="log_index",
        type=Path,
        default=None,
        help="write a CSV mapping raw episode index -> processed dataset "
        "index (columns: split, source, raw_data_index, "
        "processed_data_index, num_frames_before, num_frames_after).",
    )
    parser.add_argument(
        "--split-csv",
        "--split_csv",
        dest="split_csv",
        type=Path,
        default=None,
        help="CSV assigning episodes to a declared benchmark split (default "
        "<raw-dir>/split.csv). Rows 'episode, split' where episode is "
        "'<N>' (raw) or 'recovery/<N>'; unlisted episodes use "
        "--default-split. Each "
        "split is written to <root>/<split>/<repo_id>/. If the DEFAULT "
        "file is absent all episodes use --default-split; an explicitly-passed "
        "missing file errors.",
    )
    parser.add_argument(
        "--default-split",
        choices=ALL_SPLITS,
        default=DEFAULT_SPLIT,
        help="split assigned to episodes absent from split.csv; use this when "
        "porting one collector split directory directly",
    )
    parser.add_argument(
        "--positions-dir",
        "--positions_dir",
        dest="positions_dir",
        type=Path,
        default=None,
        help="enable ENV-state conditioning: root of the SceneDiff positions "
        "with <source>/episode_<N>.npz (source in {raw, recovery}), e.g. "
        "~/Dexmate/data/scene_diff/positions. Each ported episode gets a "
        "CONSTANT observation.environment_state (object_nums*3,) of the "
        "arranged object WORLD positions. Every ported episode must have "
        "its npz (validated up front); off by default.",
    )
    parser.add_argument(
        "--structured-env-state-from-raw",
        action="store_true",
        help="use simulator ground truth task/env_state (3,2,3) and the "
        "categorical task_sequence, progress_index target, and checked "
        "active_task_id directly from raw HDF5. This is "
        "mutually exclusive with --positions-dir and never flattens the "
        "condition.",
    )
    parser.add_argument(
        "--action-offset",
        "--pose-action-offset",
        dest="pose_action_offset",
        type=int,
        default=0,
        help="read the complete action label (EEF/head targets and both gripper "
        "commands) this many raw frames in the future, terminally repeating the "
        "last complete action; observations stay at the current frame (default 0). "
        "--pose-action-offset is retained as a legacy alias.",
    )
    parser.add_argument(
        "--object_nums",
        "--object-nums",
        dest="object_nums",
        type=int,
        default=DEFAULT_OBJECT_NUMS,
        help=f"number of conditioned objects; env-state dim = object_nums*3 "
        f"(default {DEFAULT_OBJECT_NUMS}). Only used with --positions-dir.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.positions_dir is not None and args.object_nums < 1:
        raise SystemExit(f"--object_nums must be >= 1, got {args.object_nums}")
    if args.positions_dir is not None and args.structured_env_state_from_raw:
        raise SystemExit(
            "--positions-dir and --structured-env-state-from-raw are mutually "
            "exclusive"
        )

    state_frame = "world" if args.world_state else "base"
    if state_frame not in STATE_FRAMES:  # defensive; STATE_FRAMES is the source of truth
        raise SystemExit(f"invalid state_frame {state_frame!r}")
    repo_id = args.repo_id or (
        f"{DEFAULT_REPO_ID}_world" if args.world_state else DEFAULT_REPO_ID
    )

    # Resolve the split map. An explicit --split-csv must exist; the default
    # <raw-dir>/split.csv is optional (absent -> every episode goes to DEFAULT_SPLIT).
    explicit_split = args.split_csv is not None
    split_csv = args.split_csv if explicit_split else args.raw_dir / "split.csv"

    try:
        if split_csv.exists():
            split_map = parse_split_csv(split_csv)
            logging.info("Using split csv %s (%d listed episode(s))", split_csv, len(split_map))
        elif explicit_split:
            raise RuntimeError(f"--split-csv not found: {split_csv}")
        else:
            logging.info(
                "No split csv at %s; all episodes -> %s",
                split_csv,
                args.default_split,
            )
            split_map = {}
        convert_dataset(
            args.raw_dir,
            repo_id,
            args.fps,
            args.resize_h,
            args.resize_w,
            args.task,
            args.root,
            args.overwrite,
            state_frame=state_frame,
            include_recovery_data=args.include_recovery_data,
            log_index=args.log_index,
            split_map=split_map,
            positions_dir=args.positions_dir,
            object_nums=args.object_nums,
            structured_env_state_from_raw=args.structured_env_state_from_raw,
            default_split=args.default_split,
            pose_action_offset=args.pose_action_offset,
        )
    except Exception as exc:  # CLI boundary: name the failure, exit 1
        logging.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
