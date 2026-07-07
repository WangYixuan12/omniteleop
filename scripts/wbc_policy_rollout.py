#!/usr/bin/env python3
"""WBC mobile-policy rollout: 10 Hz policy ticks over the 100 Hz whole-body IK loop.

Drives the real robot from a LeRobot checkpoint trained on the
``port_wbc_mobile_hdf5.py`` schema (PLAN.md Conventions):

  observation.state (32,) = base-frame ACHIEVED L/R EEF pos3+rot6+gripper and
      zed_depth_frame head pos3+rot6 (WBC FK on measured joints, base zero) +
      the measured odometry base pose (x, y, yaw) in the engage-origin world.
  action (29,) = WORLD-frame L/R EEF + head targets (pos3+rot6) + grippers.

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
  <save-dir>/policy_io/episode_N.hdf5 per policy tick: state/action vectors,
                                      decoded world targets, base pose, timing

The loop is synchronous: policy inference runs inline at the 10 Hz tick, so a
slow inference stalls WBC ticks (the joint clamp + interpolator glide absorb
it; ``--source-timeout`` holds the base if inference stalls badly).

Run in the dexmate_lerobot conda env ON the robot (needs lerobot + the hardware
SDK). Absolute and relative checkpoints are both supported:

- ``use_relative_actions=false``: observation.state EEF/head are BASE-frame; the
  policy emits world-frame targets directly.
- ``use_relative_actions=true``: observation.state EEF/head must be WORLD-frame
  (port with ``--relative-actions``) because LeRobot's relative step subtracts
  observation.state[:29] from the world action. The rollout builds world-frame
  state and pins the RelativeActionsProcessorStep cache to the chunk-anchor state
  between in-chunk pops (like ``follower/policy_rollout.py``) so every action
  popped from a diffusion chunk is de-relativized against the state at replan.

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

Usage:
  python scripts/wbc_policy_rollout.py --policy-path /path/to/checkpoint \
      --align-reference ~/Dexmate/data/raw_data/reference.hdf5 \
      [--auto-start] [--max-seconds 120] [--save-dir ~/Dexmate/data/raw_data_rollout]
  python scripts/wbc_policy_rollout.py \
      --replay-episode ~/Dexmate/data/raw_data/episode_0.hdf5 \
      --align-reference ~/Dexmate/data/raw_data/reference.hdf5
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from omniteleop.wbc_policy_format import (
    ACTION_AXES,
    STATE_AXES,
    WBCPolicyFK,
    build_state_vector,
    mat_to_pos6d,
    pos6d_to_mat,
)

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


def split_policy_action(action: np.ndarray) -> dict[str, np.ndarray | np.float32]:
    """29-D policy action -> world-frame 4x4 targets + gripper commands.

    Pure reshaping: positions pass through, 6-D rotations are re-orthonormalized
    by ``pos6d_to_mat``. The outputs are ALREADY world-frame ``ik.solve()``
    targets -- never compose them with any base pose, at the 10 Hz policy tick
    or the 100 Hz WBC tick.
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
        "left_gripper": np.float32(action[9]),
        "right_gripper": np.float32(action[19]),
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


class _PolicyBundle:
    """Checkpoint + pre/post processors + observation assembly for the WBC schema."""

    def __init__(self, policy_path: str, device: str | None = None) -> None:
        import torch  # noqa: PLC0415 -- heavy, hardware/GPU path only
        from lerobot.configs import PreTrainedConfig  # noqa: PLC0415
        from lerobot.policies import (  # noqa: PLC0415
            get_policy_class,
            make_pre_post_processors,
        )
        from lerobot.processor import RelativeActionsProcessorStep  # noqa: PLC0415
        from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: PLC0415

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

        # Relative-action chunk anchoring: locate the RelativeActionsProcessorStep so
        # every action popped from a diffusion chunk is de-relativized against the state
        # at REPLAN (not the drifting current obs). Mirrors follower/policy_rollout.py.
        self._relative_step = next(
            (s for s in self.pre.steps if isinstance(s, RelativeActionsProcessorStep)),
            None,
        )
        self._chunk_anchor_active = (
            self._relative_step is not None and self._relative_step.enabled
        )
        self._chunk_ref_state = None
        if self.use_relative_actions and not self._chunk_anchor_active:
            raise ValueError(
                "relative checkpoint but no enabled RelativeActionsProcessorStep in the "
                "preprocessor -- cannot chunk-anchor; refusing to roll out"
            )

    def reset(self) -> None:
        self.policy.reset()
        self._chunk_ref_state = None

    def _action_queue_len(self) -> int:
        """Peek at the policy's internal action queue length (ACT or Diffusion)."""
        q = getattr(self.policy, "_action_queue", None)
        if q is not None:
            return len(q)
        queues = getattr(self.policy, "_queues", None)
        if queues is not None and "action" in queues:
            return len(queues["action"])
        return -1

    def _chw(self, img_hwc: np.ndarray):
        import cv2  # noqa: PLC0415

        h, w = self.image_hw
        if img_hwc.shape[:2] != (h, w):
            img_hwc = cv2.resize(img_hwc, (w, h), interpolation=cv2.INTER_AREA)
        t = self._torch.from_numpy(np.ascontiguousarray(img_hwc))
        return t.permute(2, 0, 1).float() / 255.0

    def select_action(self, state: np.ndarray, head_rgb: np.ndarray,
                      wrist_rgb: np.ndarray | None) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (len(STATE_AXES),) or not np.all(np.isfinite(state)):
            raise ValueError(f"bad observation.state (shape {state.shape} or non-finite)")
        sample: dict = {
            "observation.state": self._torch.from_numpy(state),
            "observation.images.head_rgb": self._chw(head_rgb),
        }
        if self.use_wrist:
            if wrist_rgb is None:
                raise ValueError("policy consumes wrist_rgb but no wrist frame is available")
            sample["observation.images.wrist_rgb"] = self._chw(wrist_rgb)

        # Chunk anchoring (relative only): at replan (empty queue) the pre-processor
        # caches the fresh state -> snapshot it as the chunk anchor after post(). For
        # in-chunk pops (non-empty queue) restore that anchor before post() so
        # AbsoluteActionsProcessorStep adds back the replan state, not the drifting
        # current obs. No-op for absolute checkpoints.
        qlen_before = self._action_queue_len() if self._chunk_anchor_active else -1
        processed = self.pre(sample)
        if self._chunk_anchor_active and qlen_before > 0:
            self._relative_step.set_cached_state(self._chunk_ref_state)
        with self._torch.no_grad():
            a = self.policy.select_action(processed)
        a = self.post(a)
        if self._chunk_anchor_active and qlen_before == 0:
            cached = self._relative_step.get_cached_state()
            self._chunk_ref_state = None if cached is None else cached.detach().clone()
        return np.asarray(a.detach().cpu().numpy(), dtype=np.float32).reshape(-1)


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


def _run_rollout(args: argparse.Namespace) -> None:
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
        policy = _PolicyBundle(args.policy_path)
        state_frame = policy.state_frame
        print(f"[wbc_policy_rollout] policy on {policy.device}; head+"
              f"{'wrist' if policy.use_wrist else 'no-wrist'} @ {policy.image_hw}; "
              f"relative={policy.use_relative_actions} state_frame={policy.state_frame}")

    fk = WBCPolicyFK(ik=ik)  # FK on the live solver's own model
    driver = mod.HardwareDriver(args, ik, cfg, enable)
    io_log = EpisodeRecorder(str(Path(args.save_dir) / "policy_io"))
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
        next_policy_t = now
        if source is not None:
            source.start(t0)
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
            elif now >= next_policy_t:
                t_inf0 = time.perf_counter()
                head_imgs = driver._grab_head_images()  # noqa: SLF001
                wrist = driver._grab_wrist_image()  # noqa: SLF001
                if head_imgs is None:
                    raise RuntimeError("head camera delivered no frame at the policy tick")
                if policy.use_wrist and wrist is None:
                    raise RuntimeError("wrist camera delivered no frame at the policy tick")
                state = _build_state(driver, fk, state_frame)
                action = policy.select_action(
                    state, head_imgs[0], None if wrist is None else wrist[0]
                )
                targets = split_policy_action(action)
                interp.push(targets["left"], targets["right"], targets["head"],
                            now=now, duration=cmd_period)
                grip_l, grip_r = targets["left_gripper"], targets["right_gripper"]
                last_cmd_wall = now
                io_log.record({
                    "t": np.float64(now - t0),
                    "timestamp_ns": np.int64(time.time_ns()),
                    "state": state,
                    "action": action,
                    "target": {
                        "left": targets["left"].astype(np.float32),
                        "right": targets["right"].astype(np.float32),
                        "head": targets["head"].astype(np.float32),
                    },
                    "base_pose": np.asarray(driver._odom.pose, np.float32),  # noqa: SLF001
                    "inference_s": np.float32(time.perf_counter() - t_inf0),
                })
                # Re-anchor if inference overran whole policy periods.
                next_policy_t += cmd_period
                if now > next_policy_t:
                    next_policy_t = now + cmd_period

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
        driver.stop_all_motion()
        if io_log.recording:
            path = io_log.stop()
            while io_log.saving:
                time.sleep(0.05)
            if path is not None:
                print(f"[wbc_policy_rollout] policy IO log -> {path}")
        driver.close()


def main() -> None:
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
    args = parser.parse_args()

    if bool(args.policy_path) == bool(args.replay_episode):
        parser.error("exactly one of --policy-path / --replay-episode is required")
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
    _run_rollout(args)


if __name__ == "__main__":
    main()
