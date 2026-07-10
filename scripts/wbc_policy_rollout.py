#!/usr/bin/env python3
"""WBC mobile-policy rollout: 10 Hz policy ticks over the 100 Hz whole-body IK loop.

Drives the real robot from a LeRobot checkpoint trained on the
``port_wbc_mobile_hdf5.py`` schema (PLAN.md Conventions):

  observation.state (32,) = base-frame ACHIEVED L/R EEF pos3+rot6+gripper and
      zed_depth_frame head pos3+rot6 (WBC FK on measured joints, base zero) +
      the measured odometry base pose (x, y, yaw) in the engage-origin world.
  action (29,) = WORLD-frame L/R EEF + head targets (pos3+rot6) + binary grippers.

The action is decoded with NO base composition, ever: ``split_policy_action``
only rebuilds the three 4x4 poses (Gram-Schmidt on the 6-D rotation) and the
policy's world targets go straight into the ``TargetInterpolator`` ->
``ik.solve(left, right, dt, head_target=...)`` path. The learned/replayed head
target is already the post-LPF/post-deadband solver input recorded in
``action/head``, so rollout does not filter it a second time. Engage mirrors
teleop: ``ik.reset()`` +
``OdometryThread.reset_origin()`` re-anchor the world frame at the rollout
start pose.

Hardware access reuses ``wbc_vr_robot.HardwareDriver`` (same homing gate,
per-tick joint clamp, base PD/shaping, gripper pass-through, Robotiq FC03
obs monitors, cameras) so a rollout obeys the identical safety behavior as
data collection. Every rollout records:

  <save-dir>/episode_N.hdf5           the standard recorder schema (action/eef,
                                      action/head = the POLICY world targets)
  <save-dir>/policy_io/episode_N.hdf5 live rollout: ONE record per predicted
                                      chunk (state, full (n, 29) chunk, schedule
                                      offsets, timing); replay: one record per
                                      released command (state/action vectors)

Inference is asynchronous (ported from deps/rby1-wbc, see plan.md): an
inference worker thread gathers ``n_obs_steps`` fresh observations at the
dataset cadence, predicts a WHOLE action chunk every ``--policy-interval``
seconds, stamps frame k with the wall-clock time ``t_obs + k / dataset_fps``,
drops frames already unreachable by ``inference_end + --execution-latency``,
and queues the rest into a ``ScheduledPolicyAction`` buffer. The main 100 Hz
loop runs a 10 Hz sampler that lerps/slerps the scheduled trajectory at "now"
and pushes one ``TargetInterpolator`` segment -- so a slow inference never
stalls a WBC tick. If inference dies or the buffer runs dry, ``last_cmd_wall``
stops advancing and the existing ``--source-timeout`` watchdog holds the robot.
Scheduling math lives entirely on the workstation ``perf_counter`` timeline;
the camera ``timestamp_ns`` (SDK capture time on the camera-publisher-host
clock) is used only for same-publisher freshness gating (never converted
across clocks).

Run in the dexmate_lerobot conda env ON the robot (needs lerobot + the hardware
SDK). Absolute and relative checkpoints are both supported:

- ``use_relative_actions=false``: observation.state EEF/head are BASE-frame; the
  policy emits world-frame targets directly.
- ``use_relative_actions=true``: observation.state EEF/head must be WORLD-frame
  (port with ``--relative-actions``) because LeRobot's relative step subtracts
  observation.state[:29] from the world action. Chunk-level inference makes the
  anchoring trivial: ``predict_chunk`` runs the preprocessor on the replan
  observation (caching its state) and the postprocessor de-relativizes the WHOLE
  chunk against that cached anchor in one call -- no queue peeking.

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

Position conditioning (``--position-condition``): a checkpoint trained with a SceneDiff
``observation.environment_state`` (README step 3) is REQUIRED to pass ``--position-condition``.
After engage and after ``--align-reference`` has brought the arms to the shared start pose
(before recording / inference), the rollout captures one head frame, stamps it with the FK
``world_T_zed`` extrinsic (engage-origin world, the same frame as ``action`` /
``observation.state[29:32]``), and shells out to ``scene_diff/run_live_pos_condition.sh`` (in
its own ``.venv-merged`` interpreter) to diff it against ``--reference-hdf5`` and reduce the
change to the object world positions. The operator resolves the ``[box, cloth]`` order from the
size-slot overlay (``--pos-cond-matching prompt --prompt-after-capture``), and the flat
``positions[order]`` is pinned as a constant env-state for the whole episode (single stage --
no per-frame stage mask). A missing npz means the detection gate failed; the rollout raises
before policy motion starts.

Usage:
  python scripts/wbc_policy_rollout.py --policy-path /path/to/checkpoint \
      --align-reference ~/Dexmate/data/raw_data/reference.hdf5 \
      [--auto-start] [--max-seconds 120] [--save-dir ~/Dexmate/data/raw_data_rollout]
  # position-conditioned checkpoint (live SceneDiff after reference alignment):
  python scripts/wbc_policy_rollout.py --policy-path /path/to/conditioned_checkpoint \
      --align-reference ~/Dexmate/data/raw_data/reference.hdf5 \
      --position-condition --prompt-after-capture
  python scripts/wbc_policy_rollout.py \
      --replay-episode ~/Dexmate/data/raw_data/episode_0.hdf5 \
      --align-reference ~/Dexmate/data/raw_data/reference.hdf5
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import threading
import time
import traceback
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from omniteleop.follower.scenediff_live import (  # dependency-light shared live plumbing
    invoke_scenediff as _invoke_scenediff,
)
from omniteleop.follower.scenediff_live import (
    prompt_for_slot_order,
)
from omniteleop.follower.scenediff_live import (
    write_live_capture_hdf5 as _write_live_capture_hdf5,
)
from omniteleop.wbc_policy_format import (
    ACTION_AXES,
    GRIPPER_BINARY_THRESHOLD,
    STATE_AXES,
    WBCPolicyFK,
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
# Recorded inter-frame gaps beyond this get a load-time warning: record_tick skips
# during holds, so a legit take can gap; replay then holds (stale-source) and glides
# across the gap -- safe, but the operator should know the take pauses.
REPLAY_GAP_WARN_S = 0.5
# Replay frame 0 must start near the robot's current (post-alignment) pose: a
# distant first target would be commanded as a snap (clamp-limited, but a large
# IK/base transient). Fail closed instead.
REPLAY_START_POS_TOL_M = 0.15
REPLAY_START_ROT_TOL_DEG = 30.0

# --- Live SceneDiff position conditioning (--position-condition) --------------------
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
DEFAULT_POS_COND_REFERENCE = "/home/yixuan/Dexmate/data/scene_diff/_before/reference_last.hdf5"
# The live-capture HDF5 writer, the SceneDiff subprocess invoker and the operator-order
# prompt are shared with the tabletop path in omniteleop.follower.scenediff_live (imported
# above): the single-frame HDF5 layout and the run_live_pos_condition.sh env interface are
# cross-repo CONTRACTS, kept in ONE place. Only the mobile-specific pieces live below --
# the FK world_T_zed (base-odometry composition) and the flat single-stage env-state.


def split_policy_action(action: np.ndarray) -> dict[str, np.ndarray | np.float32]:
    """29-D policy action -> world-frame 4x4 targets + binary gripper commands.

    Pure reshaping: positions pass through, 6-D rotations are re-orthonormalized
    by ``pos6d_to_mat``. The outputs are ALREADY world-frame ``ik.solve()``
    targets -- never compose them with any base pose, at the 10 Hz policy tick
    or the 100 Hz WBC tick. Postprocessed gripper predictions are thresholded at
    the same value used to binarize the training actions.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (len(ACTION_AXES),):
        raise ValueError(f"expected {len(ACTION_AXES)}-D WBC action, got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError("policy action contains non-finite values")
    return {
        "left": pos6d_to_mat(action[0:9]),
        "right": pos6d_to_mat(action[10:19]),
        "head": pos6d_to_mat(action[20:29]),
        "left_gripper": np.float32(action[9] >= GRIPPER_BINARY_THRESHOLD),
        "right_gripper": np.float32(action[19] >= GRIPPER_BINARY_THRESHOLD),
    }


def encode_replay_action(
    left: np.ndarray,
    right: np.ndarray,
    head: np.ndarray,
    grip_left,
    grip_right,
) -> np.ndarray:
    """Replayed targets -> the 29-D ``ACTION_AXES`` vector a policy would emit.

    Only for the ``policy_io`` log, so replay and live rollouts are directly
    diffable; the robot itself is driven from the 4x4 matrices, never from this
    encoding.
    """
    grips = np.asarray([grip_left, grip_right], dtype=np.float32)
    if not np.all(np.isfinite(grips)):
        raise ValueError(f"replay gripper commands must be finite, got {grips}")
    return np.concatenate(
        [mat_to_pos6d(left), grips[:1], mat_to_pos6d(right), grips[1:2], mat_to_pos6d(head)]
    ).astype(np.float32)


@dataclass(frozen=True)
class ScheduledPolicyAction:
    """One chunk frame scheduled for execution at a wall-clock time.

    ``timestamp`` is an interpolation KNOT on the workstation ``perf_counter``
    timeline (clock domain (a), plan.md Conventions) -- the moment the 10 Hz
    sampler should command these targets -- not a hard release barrier: the
    sampler blends between bracketing knots.
    """

    timestamp: float
    left: np.ndarray            # (4, 4) world-frame ik.solve target
    right: np.ndarray           # (4, 4)
    head: np.ndarray            # (4, 4)
    grip_left: np.float32
    grip_right: np.float32


def scheduled_actions_from_chunk(
    chunk: np.ndarray, timestamps: np.ndarray
) -> list[ScheduledPolicyAction]:
    """Decode an ``(n, 29)`` action chunk + per-frame wall-clock timestamps.

    Row ``k`` goes through ``split_policy_action`` (the same decode as a single
    policy action: Gram-Schmidt re-orthonormalization, NO base composition) and
    is paired with ``timestamps[k]``. Timestamps must be finite and strictly
    increasing -- they are the synthesized ``t_obs + k * dataset_dt`` schedule.
    """
    chunk = np.asarray(chunk, dtype=np.float32)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[0] != ts.shape[0]:
        raise ValueError(
            f"chunk {chunk.shape} / timestamps {ts.shape}: expected (n>=1, "
            f"{len(ACTION_AXES)}) with one timestamp per frame"
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
    """Blend poses at ``query_time`` while holding the previous binary grippers."""
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
    actually commanded); afterwards ``sample()`` returns ``None``, the caller
    stops advancing ``last_cmd_wall``, and the untouched ``compute_hold_reason``
    watchdog holds the robot after ``--source-timeout`` -- covering a dead or
    stalled inference thread. Behaviorally identical otherwise: the
    ``TargetInterpolator`` holds its segment end with or without a verbatim
    re-push.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buffer: deque[ScheduledPolicyAction] = deque()
        self._last_executed: ScheduledPolicyAction | None = None

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
        with self._lock:
            while self._buffer and self._buffer[-1].timestamp >= actions[0].timestamp:
                self._buffer.pop()
            while self._buffer and self._buffer[0].timestamp < now:
                self._buffer.popleft()
            self._buffer.extend(actions)

    def sample(self, query_time: float) -> ScheduledPolicyAction | None:
        """Scheduled-trajectory value at ``query_time``; ``None`` when idle.

        ``None`` before the first chunk arrives and once the trajectory is
        exhausted (after the terminal knot has been returned once) -- the caller
        must then NOT advance ``last_cmd_wall``, arming the source watchdog.
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
            gripper_source = passed if passed is not None else (
                prev if prev is not None else future
            )
            if out is None or gripper_source is None:
                return None
            executed = replace(
                out,
                timestamp=float(query_time),
                grip_left=gripper_source.grip_left,
                grip_right=gripper_source.grip_right,
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

        self._left = np.asarray(left, dtype=np.float64)
        self._right = np.asarray(right, dtype=np.float64)
        self._head = np.asarray(head, dtype=np.float64)
        # Raw dtype on purpose: replay must send bit-identical gripper commands.
        self._grip_left = grip_left
        self._grip_right = grip_right
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

    @property
    def n_frames(self) -> int:
        return self._n

    @property
    def duration_s(self) -> float:
        return float(self._rel_s[-1])

    @property
    def released(self) -> int:
        return self._next

    @property
    def done(self) -> bool:
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
    result = ik.solve(left_target, right_target, dt, head_target=head_target)
    hold_reason = driver.extra_hold(now, last_cmd_wall, estop=estop)
    # compute_hold_reason deliberately returns None on estop (the caller owns that
    # layer), so estop must be OR'd in here, exactly like the teleop follower.
    hold = estop or (not result.success) or result.held or (hold_reason is not None)
    driver.actuate(result, float(grip_left), float(grip_right), enable, hold, dt)
    if driver._episode is not None:  # noqa: SLF001 -- same recording path as teleop
        driver.record_tick(result, hold, now, left_target=left_target,
                           right_target=right_target, head_target=head_target)
    return result, hold, hold_reason


def load_alignment_references(path: str, ik) -> tuple[np.ndarray, np.ndarray]:
    """Reference L/R EEF poses via the leader's loader (no reimplemented parsing)."""
    from omniteleop.leader.wbc_reference_alignment import (  # noqa: PLC0415 -- heavy
        load_reference_ee_poses,
    )

    ref = load_reference_ee_poses(path, ik)
    if ref is None:
        raise RuntimeError(f"alignment reference episode yielded no poses: {path!r}")
    return ref


def _achieved_world_ee_poses(driver, fk: WBCPolicyFK) -> tuple[np.ndarray, np.ndarray]:
    """Where the arms ACTUALLY are: measured-joint FK composed with odometry."""
    measured = driver._read_measured_joints()  # noqa: SLF001 -- deliberate reuse
    for grp in ("torso", "left_arm", "right_arm", "head"):
        if measured.get(grp) is None:
            raise RuntimeError(f"measured {grp} joints unavailable; cannot verify alignment")
    if driver._odom is None:  # noqa: SLF001
        raise RuntimeError("odometry required to verify alignment in the world frame")
    q = fk.q_from_raw(
        measured["torso"], measured["left_arm"], measured["right_arm"], measured["head"],
        base_xyyaw=np.asarray(driver._odom.pose, dtype=float),  # noqa: SLF001
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
    while True:
        now = now_fn()
        if now - t_start > timeout_s:
            raise RuntimeError(
                f"[wbc_policy_rollout] reference alignment did not converge within "
                f"{timeout_s:g}s (L {pos_mm[0]:.0f}mm/{rot_deg[0]:.1f}deg, "
                f"R {pos_mm[1]:.0f}mm/{rot_deg[1]:.1f}deg vs tol "
                f"{REFERENCE_ALIGN_POS_TOL_MM:g}mm/{REFERENCE_ALIGN_ROT_TOL_DEG:g}deg)"
            )
        wbc_tick(ik=ik, driver=driver, enable=enable, interp=interp,
                 head_lpf=head_lpf, head_deadband=head_deadband, now=now, dt=dt,
                 grip_left=0.0, grip_right=0.0, last_cmd_wall=now,
                 live_head_filters=True)
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


@dataclass(frozen=True)
class PolicyObservation:
    """One raw (unbatched) policy observation sample."""

    state: np.ndarray               # (32,) float32, frame per the checkpoint
    head_rgb: np.ndarray            # (H, W, 3) uint8
    wrist_rgb: np.ndarray | None    # (H, W, 3) uint8; None if the policy has no wrist


class _PolicyBundle:
    """Checkpoint + pre/post processors + chunk-level inference for the WBC schema."""

    def __init__(self, policy_path: str, device: str | None = None) -> None:
        import torch  # noqa: PLC0415 -- heavy, hardware/GPU path only
        from lerobot.configs import PreTrainedConfig  # noqa: PLC0415
        from lerobot.policies import (  # noqa: PLC0415
            get_policy_class,
            make_pre_post_processors,
        )
        from lerobot.processor import RelativeActionsProcessorStep  # noqa: PLC0415
        from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE  # noqa: PLC0415

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        pcfg = PreTrainedConfig.from_pretrained(policy_path)
        pcfg.pretrained_path = policy_path
        # Relative checkpoints are supported: state must be WORLD-frame (see
        # _build_state / --relative-actions). state_frame follows this flag.
        self.use_relative_actions = bool(getattr(pcfg, "use_relative_actions", False))
        self.state_frame = "world" if self.use_relative_actions else "base"
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
        if state_dim != len(STATE_AXES) or action_dim != len(ACTION_AXES):
            raise ValueError(
                f"checkpoint dims state={state_dim}/action={action_dim} do not match "
                f"the WBC schema ({len(STATE_AXES)}/{len(ACTION_AXES)}); wrong checkpoint?"
            )
        img_keys = [k for k in in_feats if k.startswith("observation.images.")]
        known = {"observation.images.head_rgb", "observation.images.wrist_rgb"}
        if not img_keys or set(img_keys) - known:
            raise ValueError(f"unsupported image inputs {img_keys}; expected subset of {known}")
        self.use_head = "observation.images.head_rgb" in in_feats
        self.use_wrist = "observation.images.wrist_rgb" in in_feats
        if not self.use_head:
            raise ValueError("policy must consume observation.images.head_rgb")
        chw = in_feats["observation.images.head_rgb"].shape  # (C, H, W)
        self.image_hw = (int(chw[1]), int(chw[2]))

        # Optional SceneDiff position conditioning: the policy declares
        # observation.environment_state (a constant object_nums*3 world-frame vector). The
        # live wrapper computes it once at the rollout start pose and calls set_env_state()
        # BEFORE inference; _sample_dict then rides it on every observation (finding 2:
        # inject before the obs queues fill, else a conditioned checkpoint KeyErrors on the
        # OBS_ENV_STATE queue).
        self.use_env_state = OBS_ENV_STATE in in_feats
        self.env_state_dim = int(in_feats[OBS_ENV_STATE].shape[0]) if self.use_env_state else 0
        if self.use_env_state and (self.env_state_dim < 3 or self.env_state_dim % 3 != 0):
            raise ValueError(
                f"observation.environment_state dim {self.env_state_dim} must be a positive "
                "multiple of 3 (object_nums*3); wrong checkpoint?"
            )
        self._env_state: np.ndarray | None = None

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

        # Relative checkpoints need the enabled RelativeActionsProcessorStep: the
        # preprocessor caches the replan observation's state and the postprocessor
        # de-relativizes the WHOLE (1, n, 29) chunk against it (broadcast add-back,
        # empirically exact). No per-pop cache pinning / queue peeking remains.
        relative_step = next(
            (s for s in self.pre.steps if isinstance(s, RelativeActionsProcessorStep)),
            None,
        )
        if self.use_relative_actions and (relative_step is None or not relative_step.enabled):
            raise ValueError(
                "relative checkpoint but no enabled RelativeActionsProcessorStep in the "
                "preprocessor -- cannot chunk-anchor; refusing to roll out"
            )

        # select_action() compat state (vis_wbc_policy_prediction.py): rolling raw
        # observation history + the unconsumed tail of the last predicted chunk.
        self._obs_history: deque[PolicyObservation] = deque(maxlen=self.n_obs_steps)
        self._chunk_tail: deque[np.ndarray] = deque()

    def reset(self) -> None:
        self.policy.reset()
        self._obs_history.clear()
        self._chunk_tail.clear()

    def describe(self) -> str:
        """One-line banner for _run_rollout. Other policy families override this."""
        return (f"LeRobot policy on {self.device}; head+"
                f"{'wrist' if self.use_wrist else 'no-wrist'} @ {self.image_hw}; "
                f"relative={self.use_relative_actions} state_frame={self.state_frame}; "
                f"n_obs_steps={self.n_obs_steps} n_action_steps={self.n_action_steps}")

    def set_env_state(self, vec: np.ndarray) -> None:
        """Pin the constant SceneDiff position condition fed to every observation.

        ``vec`` is the arranged ``positions[order]`` world-frame vector (length
        ``env_state_dim`` = object_nums*3). Held for the whole episode -- identical to
        training, where the env-state is a per-episode constant. Must be called before the
        inference worker starts.
        """
        if not self.use_env_state:
            raise RuntimeError(
                "checkpoint has no observation.environment_state input; cannot set env-state")
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.shape != (self.env_state_dim,) or not np.all(np.isfinite(v)):
            raise ValueError(
                f"env-state must be a finite ({self.env_state_dim},) vector, got {v.shape}")
        self._env_state = v

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
        if self.use_wrist:
            if obs.wrist_rgb is None:
                raise ValueError("policy consumes wrist_rgb but no wrist frame is available")
            sample["observation.images.wrist_rgb"] = self._chw(obs.wrist_rgb)
        if self.use_env_state:
            if self._env_state is None:
                raise RuntimeError(
                    "policy consumes observation.environment_state but set_env_state() was "
                    "never called -- run with --position-condition")
            sample["observation.environment_state"] = self._torch.from_numpy(self._env_state)
        return sample

    def predict_chunk(self, obs_history: Sequence[PolicyObservation]) -> np.ndarray:
        """One full forward pass -> ``(n_action_steps, 29)`` float32 ABSOLUTE actions.

        ``obs_history`` is oldest-first at the DATASET cadence (1/fps apart, the
        spacing the n_obs_steps>1 policy was trained on -- NOT the replan
        interval), at most ``n_obs_steps`` long; shorter histories are left-padded
        by repeating the oldest sample, exactly like LeRobot's ``populate_queues``
        bootstraps an episode start. Frame 0 of the returned chunk is the
        CURRENT-step command (modeling_diffusion.py extracts actions from
        ``start = n_obs_steps - 1``).

        Bypasses the policy's internal ACTION queue: each observation runs through
        ``self.pre`` (so the RelativeActionsProcessorStep cache ends on the LAST =
        replan observation), the observation queues are populated the way
        ``modeling_diffusion.select_action`` does it, and ``predict_action_chunk``
        runs once. ``self.post`` de-normalizes the whole ``(1, n, 29)`` chunk; for
        relative checkpoints AbsoluteActionsProcessorStep broadcasts the cached
        anchor over every frame, de-relativizing the entire chunk against the
        replan state (empirically exact round trip).
        """
        from lerobot.policies.utils import populate_queues  # noqa: PLC0415
        from lerobot.utils.constants import ACTION, OBS_IMAGES  # noqa: PLC0415

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
        if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[1] < self.n_action_steps:
            raise ValueError(
                f"predict_action_chunk returned {tuple(chunk.shape)}; expected "
                f"(1, >={self.n_action_steps}, {len(ACTION_AXES)})"
            )
        chunk = self.post(chunk[:, : self.n_action_steps])
        out = np.asarray(chunk.detach().cpu().numpy(), dtype=np.float32)
        out = out.reshape(-1, out.shape[-1])
        if out.shape != (self.n_action_steps, len(ACTION_AXES)) or not np.all(
            np.isfinite(out)
        ):
            raise ValueError(f"bad action chunk {out.shape} (or non-finite)")
        return out

    def select_action(self, state: np.ndarray, head_rgb: np.ndarray,
                      wrist_rgb: np.ndarray | None) -> np.ndarray:
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


def _build_state(driver, fk: WBCPolicyFK, state_frame: str = "base") -> np.ndarray:
    """32-D observation.state from measured joints (FK base zero) + odom pose.

    ``state_frame`` MUST match the checkpoint: ``"base"`` for absolute policies,
    ``"world"`` for relative ones (LeRobot subtracts observation.state[:29] from the
    world action, so the state must be world-frame). See ``build_state_vector``.
    """
    measured = driver._read_measured_joints()  # noqa: SLF001 -- deliberate reuse
    for grp in ("torso", "left_arm", "right_arm", "head"):
        if measured.get(grp) is None:
            raise RuntimeError(f"measured {grp} joints unavailable/malformed; cannot build state")
    poses = fk.base_frame_poses(
        measured["torso"], measured["left_arm"], measured["right_arm"], measured["head"]
    )
    if driver._odom is None:  # noqa: SLF001
        raise RuntimeError("odometry required (--enable base) for the state base dims")
    base_pose = np.asarray(driver._odom.pose, dtype=np.float32)  # noqa: SLF001
    grip_l = float(driver._last_obs_grip_left)  # noqa: SLF001
    grip_r = float(driver._last_obs_grip_right)  # noqa: SLF001
    if not (np.isfinite(grip_l) and np.isfinite(grip_r)):
        raise RuntimeError("gripper obs still NaN (no FC03 reply yet); cannot build state")
    return build_state_vector(poses, base_pose, grip_l, grip_r, state_frame=state_frame)


class _InferenceWorker:
    """Async chunk inference: observation gathering + GPU forward OFF the 100 Hz thread.

    Each cycle: gather ``n_obs_steps`` freshness-gated observations spaced
    ``dataset_dt`` apart (the training cadence -- an n_obs_steps=2 policy must see
    0.1 s obs spacing, not the replan interval), run ``predict_chunk``, stamp
    frame ``k`` with ``t_obs + k * dataset_dt`` (``t_obs`` = the LAST
    observation's local grab time; frame 0 is the current-step command), drop
    frames not strictly after ``inference_end + execution_latency``, and queue
    the survivors. All schedule math lives on the workstation ``perf_counter``
    timeline; the camera ``frame_ns`` stamps (SDK capture time, camera-publisher-
    host clock) are used ONLY for the same-publisher strictly-increasing freshness
    gate and logged raw for offline latency analysis -- join them against the
    episode file's ``meta/camera_ntp`` from the same run for absolute staleness
    (plan.md clock-domain rules).

    THREADING AUDIT (plan.md Task 4 Step 3) -- every cross-thread call is a
    read-only, thread-safe cache read:

    * ``driver._grab_head_images`` / ``_grab_wrist_image``: dexcontrol camera
      ``get_obs`` on zenoh subscriber caches (documented "Thread Safety: This
      method is thread-safe", dexcontrol zed_camera.py).
    * ``driver._read_measured_joints`` (via ``_build_state``): dexcontrol
      ``get_joint_pos`` subscriber-cache reads.
    * ``driver._odom.pose``: OdometryThread property, lock-protected copy.
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
        self._policy_interval = float(policy_interval)
        self._execution_latency = float(execution_latency)
        self._stop = stop_event
        self.failed = threading.Event()
        self.fail_reason = ""
        self._last_head_ns = -1
        self._last_wrist_ns = -1  # stays -1 when the policy has no wrist input
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

        Both publisher stamps must strictly advance past the previous sample this
        worker used (same-clock comparison per publisher, mirroring record_tick's
        guard): the policy must never see a camera frame twice.
        """
        deadline = time.perf_counter() + self._grab_timeout
        while not self._stop.is_set():
            head = self._driver._grab_head_images()  # noqa: SLF001 -- audited read
            wrist = (
                self._driver._grab_wrist_image()  # noqa: SLF001 -- audited read
                if self._policy.use_wrist else None
            )
            head_ns = None if head is None else int(head[2])
            wrist_ns = None if wrist is None else int(wrist[1])
            fresh = head_ns is not None and head_ns > self._last_head_ns and (
                not self._policy.use_wrist
                or (wrist_ns is not None and wrist_ns > self._last_wrist_ns)
            )
            if fresh:
                state = _build_state(self._driver, self._fk, self._policy.state_frame)
                self._last_head_ns = head_ns
                if wrist_ns is not None:
                    self._last_wrist_ns = wrist_ns
                return PolicyObservation(
                    state=state, head_rgb=head[0],
                    wrist_rgb=None if wrist is None else wrist[0],
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
            prev_grab_t = time.perf_counter()
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
                obs_list, t_obs = gathered
                t_pred0 = time.perf_counter()
                chunk = self._policy.predict_chunk(obs_list)
                t_end = time.perf_counter()
                n = chunk.shape[0]
                timestamps = t_obs + self._dataset_dt * np.arange(n, dtype=np.float64)
                actions = scheduled_actions_from_chunk(chunk, timestamps)
                kept = drop_stale_actions(actions, t_end + self._execution_latency)
                if kept:
                    self._buffer.queue(kept, now=t_end)
                    if kept[-1].timestamp < t_end + self._policy_interval:
                        print(f"\n[wbc_policy_rollout] WARNING: chunk covers only "
                              f"{kept[-1].timestamp - t_end:.2f}s past inference end "
                              f"but the next replan is ~{self._policy_interval:g}s away "
                              f"(inference {t_end - t_pred0:.2f}s): the buffer will run "
                              "dry -> stale-source holds. Lower --policy-interval or "
                              "speed up inference.")
                else:
                    print(f"\n[wbc_policy_rollout] WARNING: entire chunk stale "
                          f"(inference {t_end - t_pred0:.2f}s + latency "
                          f"{self._execution_latency:g}s passed the last frame at "
                          f"t_obs+{(n - 1) * self._dataset_dt:.2f}s); nothing queued")
                self._io_log.record({
                    "t": np.float64(t_obs - self._t0),
                    "timestamp_ns": np.int64(time.time_ns()),
                    "state": obs_list[-1].state,
                    # FULL pre-drop chunk: constant (n_action_steps, 29) shape so
                    # EpisodeRecorder can np.stack; n_dropped says what was queued.
                    "action": chunk,
                    "action_offsets_s": (timestamps - self._t0).astype(np.float64),
                    "base_pose": np.asarray(self._driver._odom.pose, np.float32),  # noqa: SLF001
                    "inference_s": np.float32(t_end - t_pred0),
                    "obs_gather_s": np.float32(t_pred0 - cycle_t0),
                    "n_dropped": np.int64(n - len(kept)),
                    "head_frame_ns": np.int64(self._last_head_ns),
                    "wrist_frame_ns": np.int64(self._last_wrist_ns),
                })
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
        raise RuntimeError("odometry required (--enable base) to anchor world_T_zed")
    base_pose = np.asarray(driver._odom.pose, dtype=np.float64)  # noqa: SLF001
    world_T_zed = base_pose_to_mat(base_pose) @ base_T_zed
    if world_T_zed.shape != (4, 4) or not np.all(np.isfinite(world_T_zed)):
        raise RuntimeError(f"bad world_T_zed {world_T_zed.shape} (or non-finite)")
    return world_T_zed.astype(np.float32)


def _pos_cond_log(msg: str) -> None:
    print(f"[wbc_policy_rollout] {msg}", flush=True)


def _prompt_for_pos_cond_order(object_nums: int, overlay_path: Path) -> list[int]:
    """Operator picks the task order from the size-slot overlay (shared prompt + task hint).

    Returns ``order`` (length ``object_nums``): ``order[k]`` is the size-slot index assigned
    to task slot k of ``[s1_src, s1_dst, ...]``. For the box->cloth task (object_nums=2):
    s1_src = the BOX, s1_dst = the CLOTH.
    """
    return prompt_for_slot_order(
        object_nums, overlay_path, log=_pos_cond_log,
        hint="for the box->cloth task, s1_src = the BOX, s1_dst = the CLOTH.",
    )


def _invoke_live_scenediff(args: argparse.Namespace, live_hdf5: Path, out_dir: Path,
                           obj_num: int) -> Path:
    """Adapt the rollout CLI args to the shared SceneDiff invoker; return episode_0.npz."""
    return _invoke_scenediff(
        repo=args.scene_diff_repo, python=args.scene_diff_python,
        live_hdf5=live_hdf5, out_dir=out_dir, reference=args.reference_hdf5,
        sam=args.sam, obj_num=obj_num, matching=args.pos_cond_matching,
        config=args.scenediff_config or None, timeout=args.scenediff_timeout,
        log=_pos_cond_log,
    )


def _position_condition_output_dir(save_dir: str | Path, episode_id: int) -> Path:
    if not isinstance(episode_id, (int, np.integer)) or int(episode_id) < 0:
        raise RuntimeError(
            f"position-condition needs a pending episode_<N>.hdf5 id, got {episode_id!r}"
        )
    return Path(save_dir) / "scene_diff" / f"episode_{int(episode_id)}"


def _prepare_live_position_condition(args: argparse.Namespace, driver, fk: WBCPolicyFK,
                                     bundle: "_PolicyBundle") -> np.ndarray:
    """Capture -> SceneDiff -> arranged env-state ``(bundle.env_state_dim,)`` in engage-origin world.

    ``object_nums`` is DERIVED from the policy's env-state dim (``env_state_dim // 3``), never
    hardcoded. Returns ``positions[order].reshape(-1)`` -- the FLAT vector the mobile policy
    consumes directly (single stage; no before/after split, unlike the tabletop 2-stage
    collapse). SceneDiff outputs land under ``<save_dir>/scene_diff/episode_N/``, where
    ``N`` is the pending main recorder id for the paired ``episode_N.hdf5``.
    """
    from omniteleop.common.head_camera import ZED_K  # noqa: PLC0415

    object_nums = bundle.env_state_dim // 3
    episode = getattr(driver, "_episode", None)
    if episode is None:
        raise RuntimeError("--position-condition requires rollout recording to pair SceneDiff "
                           "outputs with episode_<N>.hdf5")
    out_dir = _position_condition_output_dir(args.save_dir, getattr(episode, "episode_id", None))
    out_dir.mkdir(parents=True, exist_ok=True)
    live_hdf5 = out_dir / "live_capture.hdf5"

    if args.prompt_before_capture:
        input("\n>>> Robot is at the rollout start pose. Stage the scene, then press ENTER "
              "to capture the head frame and run SceneDiff (Ctrl-C to abort) <<<\n")

    # 1) One head frame at the rollout start viewpoint + its FK world_T_zed extrinsic.
    head = driver._grab_head_images()  # noqa: SLF001 -- audited camera-cache read
    if head is None:
        raise RuntimeError("head camera delivered no frame for the position-condition capture")
    rgb, depth_u16 = head[0], head[1]
    extrinsic = _live_world_T_zed(driver, fk)
    intrinsic = np.asarray(ZED_K, dtype=np.float32)
    _write_live_capture_hdf5(live_hdf5, rgb, depth_u16, extrinsic, intrinsic)
    print(f"[wbc_policy_rollout] live 'before' frame -> {live_hdf5}  rgb={rgb.shape} "
          f"depth={depth_u16.shape}  cam_xyz=[{extrinsic[0,3]:.3f},{extrinsic[1,3]:.3f},"
          f"{extrinsic[2,3]:.3f}]", flush=True)

    # 2) SceneDiff (own interpreter). 3) resolve the [box,cloth] order (operator prompt live).
    npz = _invoke_live_scenediff(args, live_hdf5, out_dir, object_nums)
    with np.load(npz, allow_pickle=True) as data:
        if "positions" not in data or "order" not in data:
            raise ValueError(f"{npz}: expected positions+order, got {list(data.files)}")
        positions = np.asarray(data["positions"], dtype=np.float32)
        order = [int(x) for x in np.asarray(data["order"]).reshape(-1)]
    if positions.shape != (object_nums, 3) or not np.all(np.isfinite(positions)):
        raise ValueError(
            f"{npz}: positions {tuple(positions.shape)} != ({object_nums}, 3) or non-finite")

    if args.pos_cond_matching == "prompt" and args.prompt_after_capture:
        order = _prompt_for_pos_cond_order(object_nums, out_dir / "deploy_slot_overlay.png")
    elif args.pos_cond_matching == "prompt":
        print("[wbc_policy_rollout] WARNING: pos_cond_matching=prompt but --prompt-after-capture "
              "is off; using the npz's identity (size) order with no operator input.", flush=True)
    if sorted(order) != list(range(object_nums)):
        raise ValueError(f"{npz}: order {order} is not a permutation of range({object_nums})")

    env_vec = positions[order].reshape(-1).astype(np.float32)
    print(f"[wbc_policy_rollout] position condition ready: order={order} "
          f"env_state={np.round(env_vec, 3).tolist()}", flush=True)
    return env_vec


def _run_rollout(args: argparse.Namespace, *, policy_factory=None,
                 worker_factory=None) -> None:
    """Drive the 100 Hz WBC loop from a policy (or a recorded episode).

    ``policy_factory(policy_path)`` and ``worker_factory(**kwargs)`` default to
    :class:`_PolicyBundle` / :class:`_InferenceWorker` (LeRobot image policies).
    ``scripts/wbc_maniflow_rollout.py`` swaps in a ManiFlow point-cloud pair so both
    policy families share ONE hardware loop, alignment, scheduler and watchdog -- a
    second copy of this loop is exactly the kind of divergence that gets a robot hurt.
    A replacement bundle must expose ``describe``/``reset``/``predict_chunk``/
    ``state_frame``/``n_obs_steps``/``n_action_steps``/``use_env_state``.
    """
    from omniteleop.common.recorder import EpisodeRecorder  # noqa: PLC0415
    from omniteleop.wbc_robot_util import parse_enable_mask  # noqa: PLC0415
    from omniteleop.wbc_stream import (  # noqa: PLC0415
        HeadTargetLowPassFilter,
        HeadTargetPlanarDeadbandFilter,
        TargetInterpolator,
    )

    mod = _load_wbc_vr_robot()

    enable = parse_enable_mask(args.enable)
    missing = [g for g in ("torso", "arms", "head", "base") if not enable.get(g)]
    if missing:
        raise SystemExit(
            f"[wbc_policy_rollout] rollout/replay requires --enable torso,arms,head,base "
            f"(missing: {', '.join(missing)}): the action commands the whole body and "
            "the state needs odometry"
        )

    ik, cfg = mod._build_ik()  # noqa: SLF001 -- same solver construction as teleop
    if cfg.head_mode != "ik":
        raise SystemExit(
            "[wbc_policy_rollout] wbik.yaml head_mode must be 'ik': the policy head "
            "action is a zed_depth_frame pose target consumed by the head FrameTask"
        )
    print(f"[wbc_policy_rollout] model nq={ik.model.nq} head_mode={cfg.head_mode}")

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
        policy = (policy_factory or _PolicyBundle)(args.policy_path)
        state_frame = policy.state_frame
        print(f"[wbc_policy_rollout] {policy.describe()}")
        chunk_coverage = policy.n_action_steps / args.dataset_fps
        if args.policy_interval > chunk_coverage:
            print(f"[wbc_policy_rollout] WARNING: --policy-interval "
                  f"{args.policy_interval:g}s exceeds the chunk coverage "
                  f"{chunk_coverage:g}s ({policy.n_action_steps} steps @ "
                  f"{args.dataset_fps:g} fps): the scheduled buffer will run dry "
                  "between replans -> periodic stale-source holds")
        # A conditioned checkpoint MUST get its env-state (else _sample_dict raises); an
        # unconditioned one must NOT be asked to (finding 2). Enforce the match up front.
        if args.position_condition and not policy.use_env_state:
            raise SystemExit(
                "[wbc_policy_rollout] --position-condition given but the checkpoint declares "
                "no observation.environment_state input")
        if policy.use_env_state and not args.position_condition:
            raise SystemExit(
                "[wbc_policy_rollout] checkpoint requires observation.environment_state "
                f"(dim {policy.env_state_dim}); pass --position-condition (+ --reference-hdf5) "
                "so it is computed live at the rollout start pose")

    fk = WBCPolicyFK(ik=ik)  # FK on the live solver's own model
    driver = mod.HardwareDriver(args, ik, cfg, enable)
    io_log = EpisodeRecorder(str(Path(args.save_dir) / "policy_io"))
    schedule_buffer = ActionScheduleBuffer()
    stop_event = threading.Event()
    worker: _InferenceWorker | None = None
    try:
        dt = 1.0 / args.ik_rate
        cmd_period = 1.0 / args.cmd_rate
        left0 = np.asarray(ik.frame_pose(fk.left_ee_frame).homogeneous, dtype=float)
        right0 = np.asarray(ik.frame_pose(fk.right_ee_frame).homogeneous, dtype=float)
        head0 = np.asarray(ik.frame_pose(fk.head_frame).homogeneous, dtype=float)
        interp = TargetInterpolator(cmd_period, left0, right0, head0)
        head_lpf = HeadTargetLowPassFilter(args.head_lpf_tau, head0)
        head_deadband = HeadTargetPlanarDeadbandFilter(
            head0,
            position_deadband=args.head_planar_pos_deadband,
            yaw_deadband=args.head_planar_yaw_deadband,
        )

        if not args.auto_start:
            input(f"[wbc_policy_rollout] robot homed. Press Enter to ENGAGE the {mode} "
                  "(Ctrl-C stops motion at any time) ... ")

        # Engage: identical world-frame re-anchoring to the teleop follower. The
        # odometry origin is zeroed HERE, before any alignment glide -- matching
        # how a recorded episode's world frame was anchored at ITS engage edge
        # (the leader's align stage also runs after engage, before recording).
        ik.reset()
        interp.reset(left0, right0, head0)
        head_lpf.reset(head0)
        head_deadband.reset(head0)
        driver.engage_reset(ik, left0, right0, head0)
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
            left_ref, right_ref = load_alignment_references(args.align_reference, ik)
            print(f"[wbc_policy_rollout] gliding to the reference pose from "
                  f"{args.align_reference} ...")
            run_reference_alignment(
                ik=ik, driver=driver, enable=enable, interp=interp,
                head_lpf=head_lpf, head_deadband=head_deadband,
                left_ref=left_ref, right_ref=right_ref, head_current=head0, dt=dt,
                read_achieved=lambda: _achieved_world_ee_poses(driver, fk),
            )
            start_left, start_right = left_ref, right_ref
            # No 100 Hz ticks run while blocked on Enter: stop/hold everything so
            # no chassis twist or in-flight position target survives the prompt.
            driver.stop_all_motion()
            did_align = True

        # Live SceneDiff position condition: capture ONE head frame after the robot is at
        # the rollout start pose (post --align-reference when supplied), diff it against the
        # fixed reference, and pin the constant env-state BEFORE recording/inference starts.
        # Held safe during the minutes-long SceneDiff subprocess (no 100 Hz ticks run while it
        # blocks). The subprocess owns the GPU; the loaded diffusion policy sits idle.
        # set_env_state survives the earlier policy.reset().
        if args.position_condition and policy is not None:
            driver.stop_all_motion()
            env_vec = _prepare_live_position_condition(args, driver, fk, policy)
            policy.set_env_state(env_vec)

        if did_align:
            driver.stop_all_motion()
            input(f"\n[wbc_policy_rollout] at reference pose. Press Enter to actually "
                  f"start the {mode} online ... ")

        if driver._episode is not None:  # noqa: SLF001 -- record like a teleop take
            driver._episode.start()  # noqa: SLF001
            driver._next_record_t = 0.0  # noqa: SLF001
        io_log.start()

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
        grip_l = np.float32(0.0)
        grip_r = np.float32(0.0)
        ticks = 0
        while True:
            now = time.perf_counter()
            if args.max_seconds > 0 and now - t0 >= args.max_seconds:
                print(f"\n[wbc_policy_rollout] --max-seconds {args.max_seconds:g} reached.")
                break
            if worker is not None and worker.failed.is_set():
                raise RuntimeError(
                    f"[wbc_policy_rollout] inference worker died: {worker.fail_reason}"
                )

            if source is not None:
                frame = source.advance(now)
                if frame is not None:
                    # Same observation assembly as the live branch: exercises the
                    # full camera/state pipeline during replay and yields a
                    # policy_io log directly diffable against a live rollout.
                    head_imgs = driver._grab_head_images()  # noqa: SLF001
                    driver._grab_wrist_image()  # noqa: SLF001
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
                    io_log.record({
                        "t": np.float64(now - t0),
                        "timestamp_ns": np.int64(time.time_ns()),
                        "state": state,
                        "action": encode_replay_action(
                            frame.left, frame.right, frame.head,
                            frame.grip_left, frame.grip_right),
                        "target": {
                            "left": frame.left.astype(np.float32),
                            "right": frame.right.astype(np.float32),
                            "head": frame.head.astype(np.float32),
                        },
                        "base_pose": np.asarray(driver._odom.pose, np.float32),  # noqa: SLF001
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
                    interp.push(scheduled.left, scheduled.right, scheduled.head,
                                now=now, duration=cmd_period)
                    grip_l = scheduled.grip_left
                    grip_r = scheduled.grip_right
                    last_cmd_wall = now
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
    except KeyboardInterrupt:
        print("\n[wbc_policy_rollout] Ctrl-C: stopping motion.")
    finally:
        # Order matters: signal the worker (instant), halt the robot immediately
        # (never wait behind a GPU forward), then join the worker BEFORE stopping
        # the io_log it records into (bare list append, single-writer contract).
        stop_event.set()
        driver.stop_all_motion()
        if worker is not None and not worker.join(timeout=10.0):
            print("[wbc_policy_rollout] WARNING: inference worker still running after "
                  "10s (GPU forward in flight?); the policy IO log may lose its last "
                  "chunk record")
        if io_log.recording:
            path = io_log.stop()
            while io_log.saving:
                time.sleep(0.05)
            if path is not None:
                print(f"[wbc_policy_rollout] policy IO log -> {path}")
        driver.close()


def build_parser() -> argparse.ArgumentParser:
    """Every CLI flag of the shared rollout. ``wbc_maniflow_rollout.py`` extends this."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy-path", default=None,
                        help="LeRobot checkpoint dir trained on the WBC 32/29 schema "
                             "(exactly one of --policy-path / --replay-episode).")
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
    parser.add_argument("--namespace", default="",
                        help="Zenoh namespace (matches wbc_vr_robot; default empty).")
    parser.add_argument("--save-dir", default=None,
                        help=f"episode + policy_io logs (default "
                             f"{DEFAULT_ROLLOUT_SAVE_DIR} with --policy-path, "
                             f"{DEFAULT_REPLAY_SAVE_DIR} with --replay-episode).")
    parser.add_argument("--auto-start", action="store_true",
                        help="skip the Enter-to-engage prompt.")
    parser.add_argument("--max-seconds", type=float, default=0.0,
                        help="stop after this many seconds (0 = run until Ctrl-C).")
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

    pc = parser.add_argument_group("position conditioning (live SceneDiff, --policy-path only)")
    pc.add_argument("--position-condition", action="store_true",
                    help="compute a live SceneDiff observation.environment_state after engage "
                         "and post-reference-alignment: capture one head frame, diff it against "
                         "--reference-hdf5, and pin the arranged object world positions for the "
                         "whole episode. REQUIRED for a checkpoint that declares "
                         "observation.environment_state.")
    pc.add_argument("--reference-hdf5", default=DEFAULT_POS_COND_REFERENCE,
                    help="fixed 'after'/moved reference scene the live frame is diffed against "
                         "(the SAME scene training used; default the Part-A reference_last.hdf5).")
    pc.add_argument("--pos-cond-matching", default="prompt",
                    choices=("prompt", "size", "hungarian"),
                    help="ordering forwarded to extract_object_positions.py. Live scenes have no "
                         "demo trajectory, so use 'prompt' (default): the operator reads the "
                         "size-slot overlay and types [box, cloth].")
    pc.add_argument("--prompt-before-capture", action="store_true",
                    help="pause for ENTER at the rollout start pose and before the head capture "
                         "(post-reference-alignment when --align-reference is supplied).")
    pc.add_argument("--prompt-after-capture", action="store_true",
                    help="in prompt mode, preview the size-slot overlay after capture (background "
                         "viewer if a display exists, else the path is printed) and read the task "
                         "order from the terminal. Without it, prompt mode uses the identity order.")
    pc.add_argument("--sam", default="sam3", help="SAM version for change detection (default sam3).")
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
    if args.position_condition and not args.policy_path:
        parser.error("--position-condition requires --policy-path (replay has no policy to "
                     "condition)")
    if args.position_condition and not np.isfinite(args.scenediff_timeout):
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

    if args.max_seconds < 0 or not np.isfinite(args.max_seconds):
        parser.error("--max-seconds must be finite and >= 0")
    if not np.isfinite(args.policy_interval) or args.policy_interval <= 0:
        parser.error("--policy-interval must be finite and > 0")
    if not np.isfinite(args.execution_latency) or args.execution_latency < 0:
        parser.error("--execution-latency must be finite and >= 0")
    if not np.isfinite(args.dataset_fps) or args.dataset_fps <= 0:
        parser.error("--dataset-fps must be finite and > 0")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    finalize_args(parser, args)
    _run_rollout(args)


if __name__ == "__main__":
    main()
