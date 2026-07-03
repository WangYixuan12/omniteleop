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
policy's world targets go straight into the SAME ``TargetInterpolator`` ->
head LPF/planar deadband -> ``ik.solve(left, right, dt, head_target=...)``
path the teleop follower runs (``wbc_vr_robot.py``), with
``WBCConfig(head_mode="ik")`` so camera pan keeps resolving through base yaw
exactly as in the training data. Engage mirrors teleop: ``ik.reset()`` +
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

Usage:
  python scripts/wbc_policy_rollout.py --policy-path /path/to/checkpoint \
      [--auto-start] [--max-seconds 120] [--save-dir ~/Dexmate/data/raw_data_rollout]
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np

from omniteleop.wbc_policy_format import (
    ACTION_AXES,
    STATE_AXES,
    WBCPolicyFK,
    build_state_vector,
    pos6d_to_mat,
)

DEFAULT_ROLLOUT_SAVE_DIR = str(Path("~/Dexmate/data/raw_data_rollout").expanduser())


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
            f"[wbc_policy_rollout] rollout requires --enable torso,arms,head,base "
            f"(missing: {', '.join(missing)}): the policy commands the whole body and "
            "the state needs odometry"
        )

    ik, cfg = mod._build_ik()  # noqa: SLF001 -- same solver construction as teleop
    if cfg.head_mode != "ik":
        raise SystemExit(
            "[wbc_policy_rollout] wbik.yaml head_mode must be 'ik': the policy head "
            "action is a zed_depth_frame pose target consumed by the head FrameTask"
        )
    print(f"[wbc_policy_rollout] model nq={ik.model.nq} head_mode={cfg.head_mode}")

    print(f"[wbc_policy_rollout] loading policy {args.policy_path} ...")
    policy = _PolicyBundle(args.policy_path)
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
            input("[wbc_policy_rollout] robot homed. Press Enter to ENGAGE the policy "
                  "(Ctrl-C stops motion at any time) ... ")

        # Engage: identical world-frame re-anchoring to the teleop follower.
        ik.reset()
        interp.reset(left0, right0, head0)
        head_lpf.reset(head0)
        head_deadband.reset(head0)
        driver.engage_reset(ik, left0, right0, head0)
        policy.reset()
        if driver._episode is not None:  # noqa: SLF001 -- record like a teleop take
            driver._episode.start()  # noqa: SLF001
            driver._next_record_t = 0.0  # noqa: SLF001
        io_log.start()

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

        print(f"[wbc_policy_rollout] engaged: policy {args.cmd_rate:g} Hz over "
              f"WBC {args.ik_rate:g} Hz; recording -> {args.save_dir}")
        t0 = time.perf_counter()
        now = t0
        next_policy_t = now
        last_cmd_wall: float | None = None
        grip_l = np.float32(0.0)
        grip_r = np.float32(0.0)
        left_target, right_target, head_target = left0, right0, head0
        ticks = 0
        while True:
            now = time.perf_counter()
            if args.max_seconds > 0 and now - t0 >= args.max_seconds:
                print(f"\n[wbc_policy_rollout] --max-seconds {args.max_seconds:g} reached.")
                break

            if now >= next_policy_t:
                t_inf0 = time.perf_counter()
                head_imgs = driver._grab_head_images()  # noqa: SLF001
                wrist = driver._grab_wrist_image()  # noqa: SLF001
                if head_imgs is None:
                    raise RuntimeError("head camera delivered no frame at the policy tick")
                if policy.use_wrist and wrist is None:
                    raise RuntimeError("wrist camera delivered no frame at the policy tick")
                state = _build_state(driver, fk, policy.state_frame)
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

            left_target, right_target, head_target = interp.at(now)
            head_target = head_lpf.filter(head_target, dt)
            head_target = head_deadband.filter(head_target)
            result = ik.solve(left_target, right_target, dt, head_target=head_target)
            hold_reason = driver.extra_hold(now, last_cmd_wall, estop=False)
            hold = (not result.success) or result.held or (hold_reason is not None)
            driver.actuate(result, float(grip_l), float(grip_r), enable, hold, dt)
            if driver._episode is not None:  # noqa: SLF001
                driver.record_tick(result, hold, now, left_target=left_target,
                                   right_target=right_target, head_target=head_target)

            ticks += 1
            if ticks % 50 == 0:
                state_str = f"HOLD:{hold_reason}" if hold_reason else ("hold" if hold else "run")
                print(f"t={now - t0:6.1f}s  {state_str:12s} "
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
    parser.add_argument("--policy-path", required=True,
                        help="LeRobot checkpoint dir trained on the WBC 32/29 schema.")
    parser.add_argument("--enable", default="arms,torso,head,base",
                        help="DOF groups to actuate (rollout requires all four; the "
                             "grippers are always active).")
    parser.add_argument("--namespace", default="",
                        help="Zenoh namespace (matches wbc_vr_robot; default empty).")
    parser.add_argument("--save-dir", default=DEFAULT_ROLLOUT_SAVE_DIR,
                        help=f"rollout episode + policy_io logs (default "
                             f"{DEFAULT_ROLLOUT_SAVE_DIR}).")
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
    args = parser.parse_args()

    # Rollouts always record (episode + policy IO); the HardwareDriver builds the
    # cameras only under record=True and the policy needs them anyway.
    args.record = True
    args.replay = None
    args.debug_dir = None
    args.speed = mod.DEFAULT_SPEED
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
    args.base_yaw_hold_deadband = mod.DEFAULT_BASE_YAW_HOLD_DEADBAND
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
