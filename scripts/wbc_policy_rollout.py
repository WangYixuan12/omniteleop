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
            gripper_source = passed if passed is not None else (
                prev if prev is not None else future
            )
            if out is None or gripper_source is None:
                if self._armed:
                    self._terminal_underflow = True
                    raise RuntimeError(
                        "terminal action-buffer underflow after the first chunk was armed"
                    )
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


@dataclass(frozen=True)
class PolicyObservation:
    """One raw (unbatched) policy observation sample."""

    state: np.ndarray               # (32,) float32, frame per the checkpoint
    head_rgb: np.ndarray            # (H, W, 3) uint8
    # arm ("left"/"right") -> (H, W, 3) uint8. Empty when the policy has no wrist
    # input; carries exactly the arms the checkpoint declares.
    wrist_rgb: dict[str, np.ndarray]


class _PolicyBundle:
    """Checkpoint + pre/post processors + chunk-level inference for the WBC schema."""

    def __init__(self, policy_path: str, device: str | None = None,
                 scene_diff_repo: str = DEFAULT_SCENE_DIFF_REPO) -> None:
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
        self.checkpoint_path = str(Path(policy_path).expanduser().resolve())
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
        if state_dim != len(STATE_AXES) or action_dim != len(ACTION_AXES):
            raise ValueError(
                f"checkpoint dims state={state_dim}/action={action_dim} do not match "
                f"the WBC schema ({len(STATE_AXES)}/{len(ACTION_AXES)}); wrong checkpoint?"
            )
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
        return {
            "checkpoint_path": np.asarray(self.checkpoint_path.encode("utf-8")),
            "position_condition_mode": np.asarray(
                self.position_condition_mode.encode("utf-8")
            ),
            "action_frame_mode": np.asarray(self.action_frame_mode.encode("utf-8")),
        }

    def last_inference_trace(self) -> dict | None:
        """Small CPU snapshot from the most recent completed inference."""
        return self._last_inference_trace

    def describe(self) -> str:
        """One-line banner for _run_rollout. Other policy families override this."""
        return (f"LeRobot policy on {self.device}; head+"
                f"{'+'.join(f'{a}_wrist' for a in self.wrist_arms) or 'no-wrist'} "
                f"@ {self.image_hw}; "
                f"state_frame={self.state_frame}; "
                f"action_frame={self.action_frame_mode}; "
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
        """One full forward pass -> ``(n_action_steps, 29)`` float32 ABSOLUTE actions.

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


def _build_state(driver, fk: WBCPolicyFK, state_frame: str = "base") -> np.ndarray:
    """32-D observation.state from measured joints (FK base zero) + odom pose.

    ``state_frame`` MUST match the checkpoint: ``"base"`` for ordinary policies,
    or whatever ``mof_state_frame`` declares for MoF ones. See ``build_state_vector``.
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


def _grab_policy_head_rgb(driver) -> tuple[np.ndarray, int] | None:
    """Read head RGB without repeatedly converting depth when the driver supports it.

    Custom/test drivers that only implement the historical RGB-D helper keep
    working through the fallback.  Both paths return the same publisher capture
    timestamp used by the freshness gate.
    """
    rgb_only = getattr(driver, "_grab_head_rgb", None)
    if callable(rgb_only):
        return rgb_only()
    rgbd = driver._grab_head_images()  # noqa: SLF001 -- backward-compatible audited read
    return None if rgbd is None else (rgbd[0], int(rgbd[2]))


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

    * ``driver._grab_head_rgb`` / ``_grab_wrist_image``: dexcontrol camera
      ``get_obs`` on Zenoh subscriber caches (documented "Thread Safety: This
      method is thread-safe", dexcontrol zed_camera.py). A custom driver's
      historical ``_grab_head_images`` remains a compatibility fallback.
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
        self._last_head_ns = int(getattr(policy, "initial_head_timestamp_ns", -1))
        # Per-arm last-used capture stamp; empty when the policy has no wrist input.
        self._last_wrist_ns: dict[str, int] = {
            arm: -1 for arm in getattr(policy, "wrist_arms", ())
        }
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
            head = _grab_policy_head_rgb(self._driver)
            wrists = {
                arm: self._driver._grab_wrist_image(arm)  # noqa: SLF001 -- audited read
                for arm in self._policy.wrist_arms
            }
            head_ns = None if head is None else int(head[1])
            wrist_ns = {
                arm: (None if w is None else int(w[1])) for arm, w in wrists.items()
            }
            # Every declared stream must advance past what this worker last used, so the
            # policy never sees any camera frame twice.
            fresh = head_ns is not None and head_ns > self._last_head_ns and all(
                ns is not None and ns > self._last_wrist_ns[arm]
                for arm, ns in wrist_ns.items()
            )
            if fresh:
                state = _build_state(self._driver, self._fk, self._policy.state_frame)
                self._last_head_ns = head_ns
                for arm, ns in wrist_ns.items():
                    self._last_wrist_ns[arm] = ns
                return PolicyObservation(
                    state=state, head_rgb=head[0],
                    wrist_rgb={arm: w[0] for arm, w in wrists.items()},
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
                if self._stop.is_set():
                    break
                obs_list, t_obs = gathered
                obs_ready_wall_ns = time.time_ns()
                t_pred0 = time.perf_counter()
                chunk = self._policy.predict_chunk(obs_list)
                if self._stop.is_set():
                    break
                policy_end = time.perf_counter()
                n = chunk.shape[0]
                timestamps = t_obs + self._dataset_dt * np.arange(n, dtype=np.float64)
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
                          f"t_obs+{(n - 1) * self._dataset_dt:.2f}s); nothing queued")
                schedule_end = time.perf_counter()
                io_frame = {
                    "t": np.float64(t_obs - self._t0),
                    "timestamp_ns": np.int64(time.time_ns()),
                    "state": obs_list[-1].state,
                    # FULL pre-drop chunk: constant (n_action_steps, 29) shape so
                    # EpisodeRecorder can np.stack; n_dropped says what was queued.
                    "action": chunk,
                    "action_offsets_s": (timestamps - self._t0).astype(np.float64),
                    "base_pose": np.asarray(self._driver._odom.pose, np.float32),  # noqa: SLF001
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
                    "head_frame_ns": np.int64(self._last_head_ns),
                    # -1 when the policy declares no such wrist input.
                    "left_wrist_frame_ns": np.int64(self._last_wrist_ns.get("left", -1)),
                    "right_wrist_frame_ns": np.int64(self._last_wrist_ns.get("right", -1)),
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
        raise RuntimeError("odometry required (--enable base) to anchor world_T_zed")
    base_pose = np.asarray(driver._odom.pose, dtype=np.float64)  # noqa: SLF001
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
    """Start reusable recorders with fresh camera/cadence state."""
    if driver._episode is not None:  # noqa: SLF001 -- shared rollout recorder
        driver._episode.start()  # noqa: SLF001
        driver._last_rec_head_ns = -1  # noqa: SLF001
        driver._last_rec_wrist_ns = dict.fromkeys(  # noqa: SLF001
            driver._last_rec_wrist_ns, -1  # noqa: SLF001
        )
        driver._next_record_t = 0.0  # noqa: SLF001
        driver._stale_since = None  # noqa: SLF001
        driver._frame_age_log.clear()  # noqa: SLF001
    io_log.start()


def _set_policy_io_static(io_log, policy, *, args=None) -> None:
    """Refresh episode-specific policy and live-condition metadata before recording."""
    if policy is None:
        io_log.set_static({})
        return
    static: dict = {}
    metadata_fn = getattr(policy, "inference_metadata", None)
    if metadata_fn is not None:
        static["policy"] = metadata_fn()
    condition_fn = getattr(policy, "live_condition_metadata", None)
    if condition_fn is not None:
        condition = condition_fn()
        if condition is not None:
            static["live_object_condition"] = condition
    if args is not None:
        static["schedule"] = {
            "dataset_fps": np.asarray(args.dataset_fps, np.float64),
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


def _wait_for_recorder_save(recorder) -> bool:
    """Wait for one async save, returning whether Ctrl-C was deferred."""
    interrupted = False
    while recorder.saving:
        try:
            time.sleep(0.05)
        except KeyboardInterrupt:
            interrupted = True
    return interrupted


def _stop_policy_io_log(io_log) -> None:
    """Stop and verify a standalone policy-IO recorder save."""
    if not io_log.recording:
        return
    path = io_log.stop()
    interrupted = _wait_for_recorder_save(io_log)
    if io_log.last_save_error is not None:
        raise RuntimeError(
            f"[wbc_policy_rollout] policy IO save failed: {io_log.last_save_error}"
        ) from io_log.last_save_error
    if path is not None:
        print(f"[wbc_policy_rollout] policy IO log -> {path}")
    if interrupted:
        raise KeyboardInterrupt


def _finish_rollout_episode(*, driver, io_log, stop_event, worker, policy=None) -> None:
    """Stop motion and fully flush one episode without closing hardware."""
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
                            path = recorder.stop()
                        except KeyboardInterrupt:
                            interrupted = True
                            continue
                        except BaseException as exc:
                            error = RuntimeError(
                                f"[wbc_policy_rollout] {name} recorder stop failed: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            cleanup_errors.append(error)
                            recorder_errors.append(error)
                        break
                    if path is not None:
                        paths.append(path)

                for recorder in (main_log, io_log):
                    interrupted = _wait_for_recorder_save(recorder) or interrupted

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
                        print(f"[wbc_policy_rollout] episode pair saved -> {paths}")
                    traj = getattr(driver, "_traj", None)
                    if traj is not None:
                        run_cleanup_step(lambda: traj.flush(saved_id))

    cleanup_errors.extend(
        error for error in recorder_errors if error not in cleanup_errors
    )
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
    except BaseException as body_error:
        try:
            _finish_rollout_episode(
                driver=driver,
                io_log=io_log,
                stop_event=stop_event,
                worker=worker,
                policy=policy,
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
        policy = (
            policy_factory(args.policy_path)
            if policy_factory is not None
            else _PolicyBundle(
                args.policy_path, scene_diff_repo=args.scene_diff_repo
            )
        )
        state_frame = policy.state_frame
        print(f"[wbc_policy_rollout] {policy.describe()}")
        if hasattr(policy, "dataset_fps") and not np.isclose(
            float(policy.dataset_fps), float(args.dataset_fps), rtol=0.0, atol=1e-9
        ):
            raise SystemExit(
                f"[wbc_policy_rollout] --dataset-fps {args.dataset_fps:g} != "
                f"training metadata {float(policy.dataset_fps):g}; refusing to engage"
            )
        chunk_coverage = policy.n_action_steps / args.dataset_fps
        if args.policy_interval > chunk_coverage:
            print(f"[wbc_policy_rollout] WARNING: --policy-interval "
                  f"{args.policy_interval:g}s exceeds the chunk coverage "
                  f"{chunk_coverage:g}s ({policy.n_action_steps} steps @ "
                  f"{args.dataset_fps:g} fps): the scheduled buffer will run dry "
                  "between replans -> periodic stale-source holds")

    fk = WBCPolicyFK(ik=ik)  # FK on the live solver's own model
    driver = mod.HardwareDriver(args, ik, cfg, enable)
    signal.signal(signal.SIGINT, signal.default_int_handler)

    def run_initialized_session() -> None:
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
