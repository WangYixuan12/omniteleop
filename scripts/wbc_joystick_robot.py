#!/usr/bin/env python3
"""Real-robot joystick follower: arm/optional-torso IK, JOYSTICK base, head-track.

The joystick-teleop sibling of ``scripts/wbc_vr_robot.py``. It reuses that follower's
``HardwareDriver`` (hardware bring-up, homing, per-joint clamp/dispatch, grippers,
odometry, episode recording) and its shared ``run_loop``. Its effective control contract
also differs in Cartesian target frame and head mode, not only in how the base moves:

  * the base is EXCLUDED from the whole-body IK (``WBCConfig.lock_base_in_ik`` -> the QP
    coordinates torso + arms only, against a base held at the origin), and
  * the chassis is driven DIRECTLY from the leader's joystick ``chassis_vx/vy/wz`` through
    the shared :class:`omniteleop.follower.joystick_base.JoystickBaseShaper` -- the SAME
    ``pd_twist`` -> ``shape_twist`` -> single-axis path the WBC follower applies to its QP
    base twist, so a take shapes the base identically in sim and on hardware. The head runs
    ``head_mode: track`` (neck pan/tilt via ``solve_head``; the base does NOT follow it).

Config comes from the canonical ``follower/wbik.yaml`` (``--config`` to override) with three
joystick-mode overrides applied in code: ``lock_base_in_ik=True``, ``head_mode="track"``,
and the head planar deadbands forced to 0 (that deadband suppresses head-driven base motion
in ``head_mode: ik`` -- meaningless here, and it otherwise gives head-track a ~17deg pan
dead zone). ``base_dofs`` is read FROM the config (``xy`` restricts the joystick to
translation, ``xy_yaw`` allows yaw); the arm IK sees a locked base either way.

Run in the dexmate conda env (pinocchio + pink + dexcomm + dexcontrol). SAME safety rules
as wbc_vr_robot.py -- route base motion only through ``set_velocity``; the robot homes to
nominal, then moves. KEEP CLEAR.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from omniteleop.follower.joystick_base import JoystickBaseShaper
from omniteleop.follower.whole_body_ik import VegaWholeBodyIK, WBCConfig
from omniteleop.wbc_policy_format import JOYSTICK_POLICY_ACTION_SCHEMA
from omniteleop.wbc_robot_util import base_quiet_dispatch, parse_enable_mask
from omniteleop.wbc_teleop import (
    JoystickTeleopConfig,
    VRTeleopConfig,
    bind_joystick_teleop_args,
    bind_vr_teleop_args,
)

# Load the base real follower as a module (scripts/ is not a package; tests do this too).
# Importing binds module-level constants; main() is __main__-guarded, so nothing runs.
_BASE_PATH = Path(__file__).resolve().parent / "wbc_vr_robot.py"
_spec = importlib.util.spec_from_file_location("wbc_vr_robot", _BASE_PATH)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

HardwareDriver = _base.HardwareDriver
run_loop = _base.run_loop
_TrajLog = _base._TrajLog
VRJointSubscriber = _base.VRJointSubscriber
ReplaySource = _base.ReplaySource
_publish_abort_status = _base._publish_abort_status
peek_next_episode_id = _base.peek_next_episode_id

_JOYSTICK_RAW_EPISODE_SCHEMA = "omniteleop_joystick_mobile_raw/v1"


class JoystickHardwareDriver(HardwareDriver):
    """``HardwareDriver`` whose base is joystick-driven (not from the whole-body QP).

    Everything except the base -- homing, the per-joint clamp/dispatch, grippers, odometry,
    recording -- is inherited unchanged. ``_drive_base`` is overridden to shape the leader's
    ``chassis_*`` joystick twist through :class:`JoystickBaseShaper` (whose integrated pose
    reference is odometry-PD tracked and single-axis projected exactly like
    the WBC base path) and dispatch it with the SAME swerve quiet-hold logic as the parent.
    """

    def __init__(self, args, ik, cfg, enable, source, traj=None) -> None:
        # The joystick twist is read from the live command source each base tick; the shaper
        # carries the integrator / slew / single-axis state. Set BEFORE super().__init__ so a
        # partial-init teardown (close()) never sees them unset.
        self._source = source
        self._latched_source_sample = None
        self._shaper = JoystickBaseShaper.from_configs(cfg, args)
        super().__init__(args, ik, cfg, enable, traj=traj)

    def engage_reset(self, ik, left0, right0, head0) -> None:
        """Zero the joystick reference in lock-step with the odom origin at engage."""
        super().engage_reset(ik, left0, right0, head0)  # resets odom origin to (0,0,0)
        self._shaper.reset(np.zeros(3))

    def latch_source_sample(self, vr) -> None:
        """Latch the command object already selected by the shared arm/head loop."""
        self._latched_source_sample = vr

    def latch_chassis_intent(self, chassis) -> None:
        """Install one policy-produced chassis intent without fabricating a VR frame."""
        value = np.asarray(chassis, dtype=float).reshape(-1)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"chassis intent must be finite (3,), got {value!r}")
        self._latched_source_sample = argparse.Namespace(
            chassis_vx=float(value[0]),
            chassis_vy=float(value[1]),
            chassis_wz=float(value[2]),
        )

    def _read_joystick(self) -> np.ndarray:
        """Latched leader/policy body twist ``(vx, vy, wz)`` (zero before the first)."""
        vr = self._latched_source_sample
        if vr is None:
            return np.zeros(3)
        return np.array(
            [float(vr.chassis_vx), float(vr.chassis_vy), float(vr.chassis_wz)], dtype=float
        )

    def _drive_base(self, result, hold, enable, dt) -> None:
        # No chassis / base disabled: defer to the parent (sets the off-state /debug fields).
        if self._odom is None or not enable["base"]:
            self._dbg["rx_chassis"] = None
            self._dbg["projected_chassis"] = None
            self._dbg["joystick_axis"] = -1
            super()._drive_base(result, hold, enable, dt)
            return
        base_state = self.policy_base_pose_snapshot()
        snap = base_state["odom"]
        measured_pose = base_state["pose"]
        self._dbg["odom"] = snap
        self._dbg["arkit"] = base_state["arkit"]
        chassis = self.robot.chassis
        if hold:
            # Park the joystick reference on the measured base so there is no lunge on
            # resume, and zero the wheels (the parent's hold behavior).
            self._shaper.hold(measured_pose)
            chassis.set_velocity(vx=0.0, vy=0.0, wz=0.0, wait_time=0.0)
            self._prev_base_cmd = np.zeros(3)
            self._base_quiet_elapsed = 0.0
            self._dbg["base_pd_raw"] = None
            self._dbg["base_pd_err"] = None
            self._dbg["base_cmd"] = np.zeros(3)
            self._dbg["base_action"] = "safety_hold"
            self._dbg["rx_chassis"] = np.zeros(3)
            self._dbg["projected_chassis"] = np.zeros(3)
            self._dbg["joystick_axis"] = -1
            return
        rx_chassis = self._read_joystick()
        cmd, pd_raw, pd_err = self._shaper.step(rx_chassis, measured_pose, dt)
        if self.cfg.enable_base_single_axis and np.count_nonzero(np.abs(cmd) > 1e-9) > 1:
            raise RuntimeError(
                f"single-axis joystick shaper emitted multi-axis hardware command {cmd}"
            )
        self._prev_base_cmd = cmd
        # Dispatch via the SAME swerve quiet-hold path as HardwareDriver._drive_base: on a
        # quiet command hold the current steering with zero drive (avoid the lateral->forward
        # wheel snap set_velocity(0,0,0) causes on brief intra-motion dips); otherwise send
        # the twist. Only the shaping source differs (joystick, above) -- the dispatch and its
        # safety envelope (set_velocity / a zero-drive wheel hold) are unchanged.
        action, self._base_quiet_elapsed = base_quiet_dispatch(
            cmd, self._base_quiet_elapsed, dt, self.args.base_quiet_hold_s)
        if action == "hold":
            steer = np.asarray(chassis.steering_angle, dtype=float)
            if steer.shape == (2,) and np.all(np.isfinite(steer)):
                assert float(np.max(np.abs(cmd))) <= 1e-6  # base_quiet_dispatch invariant
                chassis.set_wheel_velocity(0.0)
            else:
                chassis.set_velocity(vx=0.0, vy=0.0, wz=0.0, wait_time=0.0)
                action = "recenter"
        else:
            chassis.set_velocity(vx=float(cmd[0]), vy=float(cmd[1]), wz=float(cmd[2]),
                                 wait_time=0.0)
        self._dbg["base_pd_raw"] = pd_raw
        self._dbg["base_pd_err"] = pd_err
        self._dbg["base_cmd"] = np.asarray(cmd, dtype=float)
        self._dbg["base_action"] = action
        self._dbg["rx_chassis"] = rx_chassis
        self._dbg["projected_chassis"] = self._shaper.last_projected_joystick.copy()
        self._dbg["joystick_axis"] = (
            -1 if self._shaper.last_intent_axis is None else self._shaper.last_intent_axis
        )

    def debug_row(self, result) -> dict:
        """Extend the inherited WBC row with the direct joystick reference chain."""
        row = super().debug_row(result)

        def vec(value):
            return (np.asarray(value, dtype=np.float32) if value is not None
                    else np.full(3, np.nan, dtype=np.float32))

        row["rx_chassis"] = vec(self._dbg.get("rx_chassis"))
        row["projected_chassis"] = vec(self._dbg.get("projected_chassis"))
        row["joystick_cmd_pose"] = np.asarray(self._shaper.cmd_pose, dtype=np.float32)
        row["joystick_axis"] = np.int64(self._dbg.get("joystick_axis", -1))
        return row

    def _recording_extra_action(self) -> dict:
        """Keep policy intent and controller state beside the applied base command.

        ``action/chassis/intent_body`` is the 32-D joystick policy label: the
        follower-effective intent after DOF masking/single-axis projection and before
        integration, odometry feedback, slew limiting, or hardware dispatch. The parent
        already records the shaped command actually sent as ``action/joint/chassis_*``.
        """
        source = np.asarray(self._dbg.get("rx_chassis"), dtype=np.float32)
        intent = np.asarray(self._dbg.get("projected_chassis"), dtype=np.float32)
        pd_raw = np.asarray(self._dbg.get("base_pd_raw"), dtype=np.float32)
        pd_error = np.asarray(self._dbg.get("base_pd_err"), dtype=np.float32)
        values = {
            "source_body": source,
            "intent_body": intent,
            "command_pose": np.asarray(self._shaper.cmd_pose, dtype=np.float32),
            "pd_raw_body": pd_raw,
            "pd_error_body": pd_error,
            "prev_shaped_body": np.asarray(self._shaper.prev_shaped, dtype=np.float32),
        }
        bad = {name: value.shape for name, value in values.items()
               if value.shape != (3,) or not np.all(np.isfinite(value))}
        if bad:
            raise RuntimeError(f"cannot record malformed joystick action fields: {bad}")
        values["active_axis"] = np.int8(self._dbg.get("joystick_axis", -1))
        return {"chassis": values}

    def _recording_control_metadata(self) -> dict:
        out = super()._recording_control_metadata()
        if getattr(self.args, "policy_path", None):
            demonstration_source = "policy_rollout"
        elif getattr(self.args, "replay_episode", None):
            demonstration_source = "policy_episode_replay"
        elif getattr(self.args, "replay", None):
            demonstration_source = "joystick_episode_replay"
        else:
            demonstration_source = "human_teleoperation"
        out.update({
            "schema": np.bytes_(_JOYSTICK_RAW_EPISODE_SCHEMA),
            "control_mode": np.bytes_("joystick"),
            "demonstration_source": np.bytes_(demonstration_source),
            "policy_action_schema": np.bytes_(JOYSTICK_POLICY_ACTION_SCHEMA),
            "action_target_frame": np.bytes_("current_base"),
            "eef_target_frame": np.bytes_("current_base"),
            "head_target_frame": np.bytes_("current_base"),
            "head_target_semantics": np.bytes_("view_direction_pan_tilt_position_roll_ignored"),
            "base_reference_source": np.bytes_("joystick_integrator"),
            "action_base_pose_semantics": np.bytes_("joystick_command_reference"),
            "chassis_intent_path": np.bytes_("action/chassis/intent_body"),
            "chassis_intent_frame": np.bytes_("current_base_body"),
            "chassis_intent_units": np.bytes_("m_per_s,m_per_s,rad_per_s"),
            "chassis_intent_stage": np.bytes_("post_mask_projection_pre_controller"),
            "chassis_intent_temporal_semantics": np.bytes_("zero_order_hold"),
            "applied_chassis_group": np.bytes_("action/joint"),
            "measured_chassis_group": np.bytes_("obs/joint"),
            "chassis_dataset_axes": np.asarray(
                [b"chassis_vx", b"chassis_vy", b"chassis_wz"]
            ),
        })
        return out

    def _recording_provenance_metadata(self) -> dict:
        """Extend inherited provenance with the actual joystick entry points/config."""
        out = super()._recording_provenance_metadata()
        out["schema"] = np.asarray(_JOYSTICK_RAW_EPISODE_SCHEMA.encode())
        repo = _BASE_PATH.parents[1]
        configured = getattr(self.args, "config", None)
        selected_config = (
            Path(configured).expanduser().resolve()
            if configured is not None
            else Path(_base.DEFAULT_CONFIG_PATH).resolve()
        )
        paths = {
            "wbc_joystick_robot": Path(__file__).resolve(),
            "wbc_joystick_leader": repo / "scripts/wbc_joystick_leader.py",
            "joystick_base": repo / "src/omniteleop/follower/joystick_base.py",
            "wbc_record": repo / "src/omniteleop/wbc_record.py",
            "selected_wbik_yaml": selected_config,
        }
        snapshots = out["source_snapshot"]
        for label, path in paths.items():
            if not path.is_file():
                raise RuntimeError(
                    f"[wbc_joystick_robot] provenance source file is missing: {path}"
                )
            snapshots[label] = {
                "path": np.asarray(
                    str(path.relative_to(repo) if path.is_relative_to(repo) else path).encode()
                ),
                "sha256": np.asarray(_base._sha256_file(path).encode()),
                "content": np.asarray(path.read_bytes()),
            }
        return out

    def record_tick(self, result, hold, *args, **kwargs):
        """Record the JOYSTICK reference pose as action/base/pose (not the locked origin).

        With the base excluded from the IK, ``result.base_pose`` is a constant origin, so it
        would make ``action/base/pose`` useless. Substitute the shaper's integrated command
        pose -- the base trajectory the operator commanded -- which is what downstream tools
        want for controller diagnostics (the 32-D policy label itself comes from
        ``action/chassis/intent_body``; ``vis_episode`` uses obs/base/pose from odometry).
        The semantics of this field therefore differ from the WBC follower's
        solver-commanded ``result.base_pose``.

        The substitution (and its allocation) is skipped whenever the base recorder would
        drop the frame anyway -- on ``hold`` or when not recording -- so this stays free on
        the 100 Hz loop in the common no-record case.
        """
        if not hold and self._episode is not None and self._episode.recording:
            result = replace(result, base_pose=np.asarray(self._shaper.cmd_pose, dtype=float))
        return super().record_tick(result, hold, *args, **kwargs)


def _build_joystick_ik(
    config: Optional[str], *, lock_torso_in_ik: Optional[bool] = None
) -> tuple[VegaWholeBodyIK, WBCConfig]:
    """Build the IK with joystick-mode overrides on top of the canonical wbik.yaml."""
    cfg = WBCConfig.from_yaml(config)
    if lock_torso_in_ik is not None:
        cfg.lock_torso_in_ik = bool(lock_torso_in_ik)
    cfg = replace(
        cfg,
        lock_base_in_ik=True,   # base excluded from the whole-body IK (joystick-driven)
        head_mode="track",      # neck pan/tilt tracks the headset; base does not follow it
    )
    ik = VegaWholeBodyIK(cfg)
    ik.reset()
    return ik, cfg


def _quarantine_failed_recording(driver, exc: BaseException) -> None:
    """Keep an exception-truncated joystick take out of the training-data glob."""
    if driver is None:
        return
    try:
        driver.abort_recording_episode(str(exc))
    except Exception as quarantine_exc:
        print(
            "[wbc_joystick_robot] WARNING: failed to quarantine the aborted "
            f"episode: {quarantine_exc}"
        )
        # Match the whole-body follower's last-resort behavior: if quarantine itself
        # fails, discard instead of letting close() publish the take as complete.
        episode = getattr(driver, "_episode", None)
        if episode is not None and getattr(episode, "recording", False):
            try:
                episode.discard()
            except Exception as discard_exc:
                print(
                    "[wbc_joystick_robot] WARNING: failed to discard the "
                    f"unquarantined take: {discard_exc}"
                )


def _run_joystick_ik_mode(args: argparse.Namespace, enable: dict) -> None:
    """Replay or live-drive the joystick follower (mirrors wbc_vr_robot._run_ik_mode)."""
    ik, cfg = _build_joystick_ik(
        args.config, lock_torso_in_ik=args.lock_torso_in_ik
    )
    print(f"[wbc_joystick_robot] model nq={ik.model.nq} nv={ik.model.nv} "
          f"head_mode={cfg.head_mode} base_dofs={cfg.base_dofs} "
          f"lock_base_in_ik={cfg.lock_base_in_ik} "
          f"lock_torso_in_ik={cfg.lock_torso_in_ik} "
          f"urdf={cfg.urdf_path.rsplit('/', 1)[-1]}")
    replay = args.replay is not None
    clock: Callable[[], float] = time.perf_counter

    if replay:
        source = ReplaySource.from_joystick_hdf5(args.replay, args.speed, clock=clock)
        print(f"[wbc_joystick_robot] replay {args.replay}: {len(source._vr)} frames, "  # noqa: SLF001
              f"{source.total_duration:.1f}s at {args.speed:g}x "
              f"({source.replay_kind}; head targets "
              f"{'already filtered' if source.head_targets_pre_filtered else 'filtered live'})")
    else:
        source = VRJointSubscriber(args.namespace, name="wbc_joystick_robot")
        print(f"[wbc_joystick_robot] live on '{source.topic}'. Ctrl-C to stop.")

    enabled = [k for k, v in enable.items() if v]
    traj: Optional[_TrajLog] = None
    debug_start_id = 0
    if args.debug_dir is not None:
        import os  # noqa: PLC0415

        os.makedirs(args.debug_dir, exist_ok=True)
        debug_start_id = (
            peek_next_episode_id(args.save_dir) if args.record
            else peek_next_episode_id(args.debug_dir, suffix="_debug")
        )
        meta = {
            "episode_id": int(debug_start_id),
            "mode": "replay" if replay else "live", "control": "joystick",
            "speed": float(args.speed) if replay else 1.0,
            "ik_rate": float(args.ik_rate), "cmd_rate": float(args.cmd_rate),
            "enable": ",".join(enabled), "urdf": cfg.urdf_path.rsplit("/", 1)[-1],
            "base_dofs": cfg.base_dofs, "lock_base_in_ik": True,
            "lock_torso_in_ik": bool(cfg.lock_torso_in_ik),
            "eef_target_frame": "current_base",
            "head_target_frame": "current_base",
            "policy_action_schema": JOYSTICK_POLICY_ACTION_SCHEMA,
            "base_reference_source": "joystick_integrator",
            "base_kp_xy": float(args.base_kp_xy), "base_kp_yaw": float(args.base_kp_yaw),
            "base_yaw_hold_in_xy": bool(args.base_yaw_hold_in_xy),
            "base_deadband": float(args.base_deadband), "base_accel": float(args.base_accel),
            "base_max_speed": float(args.base_max_speed),
            "base_quiet_hold_s": float(args.base_quiet_hold_s),
            "source_timeout": float(args.source_timeout),
            "max_joint_step": float(args.max_joint_step),
            "head_lpf_tau": float(args.head_lpf_tau),
            "joystick_stick_max_vx": float(args.joystick_stick_max_vx),
            "joystick_stick_max_vy": float(args.joystick_stick_max_vy),
            "joystick_stick_max_wz": float(args.joystick_stick_max_wz),
            "joystick_single_axis_hysteresis_ratio": float(
                args.joystick_single_axis_hysteresis_ratio
            ),
            "replay_file": str(args.replay) if replay else "",
        }
        traj = _TrajLog(args.debug_dir, meta=meta)

    print("=" * 72)
    print(f"[wbc_joystick_robot] REAL ROBOT (JOYSTICK base). enabled={enabled} grippers=on | "
          f"{'REPLAY ' + format(args.speed, 'g') + 'x' if replay else 'LIVE'} | "
          f"base_max={args.base_max_speed:g}m/s | base_dofs={cfg.base_dofs} | "
          f"torso_ik={'fixed' if cfg.lock_torso_in_ik else 'free'} | "
          f"single_axis={cfg.enable_base_single_axis}")
    if args.arkit_base != "off":
        arkit_role = (
            "closes the joystick base PD loop"
            if args.arkit_base == "control"
            else "recorded only"
        )
        print(f"  base pose -> iPhone/ARKit '{args.arkit_base}' "
              f"({arkit_role}"
              "; wheel odometry always recorded).")
    if args.record:
        head_streams = "head_left_rgb" if args.no_head_depth else "head_left_rgb+head_depth"
        if args.head_right_rgb:
            head_streams += "+head_right_rgb"
        print(f"  recording -> {args.save_dir} @ {args.record_rate:g}Hz while engaged "
              f"({head_streams}+left_wrist_rgb+right_wrist_rgb, "
              f"{'streaming' if args.streaming_recorder else 'buffered'}, "
              "32-D policy action incl. chassis intent).")
    print("  Robot homes to nominal, then moves. KEEP CLEAR. Ctrl-C aborts (zeros base).")
    print("=" * 72)

    driver = None
    try:
        driver = JoystickHardwareDriver(args, ik, cfg, enable, source=source, traj=traj)
        run_loop(source, ik, cfg, driver, args, replay=replay, clock=clock,
                 realtime=True, enable=enable, traj=traj)
    except KeyboardInterrupt:
        print("\n[wbc_joystick_robot] interrupted -- stopping.")
    except Exception as exc:
        _publish_abort_status(source, exc)
        _quarantine_failed_recording(driver, exc)
        raise
    finally:
        if hasattr(source, "close"):
            source.close()
        if driver is not None:
            driver.close()
        if traj is not None and not args.record and len(traj):
            print(f"\n[wbc_joystick_robot] summary: {traj.summary()}")
            traj.flush(debug_start_id)


def _bind_vr_teleop(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Bind the vr_teleop: tunables from wbik.yaml onto args (joystick-mode overrides applied).

    Mirrors wbc_vr_robot.main()'s post-parse binding, reading from ``--config`` (default
    canonical wbik.yaml). The head planar deadbands are forced to 0 for head-track (see the
    module docstring); everything else -- rates, base PD/slew/deadband/clamp -- is verbatim.
    """
    vt = VRTeleopConfig.from_yaml(args.config)
    joystick_cfg = JoystickTeleopConfig.from_yaml(args.config)
    bind_vr_teleop_args(args, vt, head_track=True)
    bind_joystick_teleop_args(args, vt, joystick_cfg)
    if args.record and args.record_rate > args.ik_rate:
        parser.error("--record-rate must be <= the IK rate (ik_rate in wbik.yaml)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--replay", metavar="FILE",
                      help="re-drive either a raw joystick leader stream or an "
                           "episode_<N>.hdf5 recorded by this script. Episode replay "
                           "uses action/eef, action/head, action/gripper, and the exact "
                           "action/chassis/intent_body labels; legacy joystick episodes "
                           "without that intent field are rejected.")
    mode.add_argument("--source", choices=("live",), default=None,
                      help="drive from the live leader (default when --replay is omitted).")
    parser.add_argument("--config", default=None,
                        help="wbik.yaml path (default: canonical follower/wbik.yaml). "
                             "base_dofs is read from it; lock_base_in_ik / head_mode:track / "
                             "head deadbands are forced for joystick mode.")
    parser.add_argument("--enable", default="arms,torso,head,base",
                        help="comma list of DOF groups to actuate (arms,torso,head,base or "
                             "'all'/'none'). Default all. The base needs the chassis to move.")
    parser.add_argument("--lock-torso-in-ik", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="override follower/wbik.yaml lock_torso_in_ik for this run. "
                             "Omit to use the YAML value; this is independent of the "
                             "torso entry in --enable.")
    parser.add_argument("--namespace", default="",
                        help="Zenoh namespace (must match the leader; default empty).")
    parser.add_argument("--debug-dir", dest="debug_dir", default=None,
                        help="write per-tick /debug HDF5 logs under this directory.")
    parser.add_argument("--record", action="store_true",
                        help="record an episode while engaged (vr_reader EpisodeRecorder "
                             "format). Needs --enable arms,torso,head,base.")
    parser.add_argument("--streaming-recorder", action="store_true",
                        help="write through the bounded streaming HDF5 recorder. "
                             "Requires --record.")
    parser.add_argument("--no-head-depth", action="store_true",
                        help="omit live head depth while recording. Requires --record.")
    parser.add_argument("--head-right-rgb", action="store_true",
                        help="record the rectified head right eye for offline stereo. "
                             "Requires --record.")
    parser.add_argument("--wrist-right-rgb", action="store_true",
                        help="record both wrist right eyes for offline stereo. "
                             "Requires --record.")
    parser.add_argument("--save-dir", default=_base.DEFAULT_SAVE_DIR,
                        help=f"directory for recorded episodes (default {_base.DEFAULT_SAVE_DIR}).")
    parser.add_argument("--record-rate", type=float, default=_base.DEFAULT_RECORD_RATE,
                        help="episode record cadence in Hz "
                             f"(default {_base.DEFAULT_RECORD_RATE:g}).")
    parser.add_argument("--record-stale-grace", type=float,
                        default=_base.DEFAULT_RECORD_STALE_GRACE,
                        help="seconds to retry a stale camera frame before aborting a take.")
    parser.add_argument("--record-startup-timeout", type=float,
                        default=_base.DEFAULT_RECORD_STARTUP_TIMEOUT,
                        help="seconds allowed for advancing, capture-aligned camera "
                             "frames before recording frame zero.")
    parser.add_argument("--record-max-camera-skew-ms", type=float,
                        default=_base.DEFAULT_RECORD_MAX_CAMERA_SKEW_MS,
                        help="maximum corrected head-to-wrist capture skew per row.")
    parser.add_argument("--record-max-camera-age-ms", type=float,
                        default=_base.DEFAULT_RECORD_MAX_CAMERA_AGE_MS,
                        help="maximum corrected camera capture-to-selection age per row.")
    hw = parser.add_argument_group("hardware")
    hw.add_argument("--home-tol", type=float, default=_base.DEFAULT_HOME_TOL)
    hw.add_argument("--home-settle", type=float, default=_base.DEFAULT_HOME_SETTLE)
    hw.add_argument("--max-joint-step", type=float, default=_base.DEFAULT_MAX_JOINT_STEP)
    hw.add_argument("--source-timeout", type=float, default=_base.DEFAULT_SOURCE_TIMEOUT)
    hw.add_argument("--drive-state-mode", choices=("ms", "rad"), default="ms")
    hw.add_argument("--arkit-base", dest="arkit_base",
                    choices=("off", "record", "control"), default="off",
                    help="use iPhone/ARKit base pose: record it, or also close the "
                         "joystick command-pose PD loop with it. Needs --enable base.")
    hw.add_argument("--base-quiet-hold-s", type=float, default=_base.DEFAULT_BASE_QUIET_HOLD_S)
    args = parser.parse_args()

    _bind_vr_teleop(args, parser)
    for flag, val in (("--home-tol", args.home_tol), ("--max-joint-step", args.max_joint_step),
                      ("--home-settle", args.home_settle),
                      ("--record-stale-grace", args.record_stale_grace),
                      ("--record-max-camera-skew-ms", args.record_max_camera_skew_ms)):
        if not np.isfinite(val) or val < 0.0:
            parser.error(f"{flag} must be finite and >= 0")
    for flag, val in (
        ("--record-startup-timeout", args.record_startup_timeout),
        ("--record-max-camera-age-ms", args.record_max_camera_age_ms),
    ):
        if not np.isfinite(val) or val <= 0.0:
            parser.error(f"{flag} must be finite and > 0")
    if not np.isfinite(args.base_quiet_hold_s):
        parser.error("--base-quiet-hold-s must be finite")
    for flag, val in (
        ("--source-timeout", args.source_timeout),
        ("--record-rate", args.record_rate),
    ):
        if not np.isfinite(val) or val <= 0.0:
            parser.error(f"{flag} must be finite and > 0")
    for flag, enabled in (
        ("--streaming-recorder", args.streaming_recorder),
        ("--no-head-depth", args.no_head_depth),
        ("--head-right-rgb", args.head_right_rgb),
        ("--wrist-right-rgb", args.wrist_right_rgb),
    ):
        if enabled and not args.record:
            parser.error(f"{flag} only affects recording; pass it with --record")
    try:
        enable = parse_enable_mask(args.enable)
    except ValueError as exc:
        parser.error(str(exc))
    if args.arkit_base != "off" and not enable["base"]:
        parser.error("--arkit-base needs --enable base")

    _run_joystick_ik_mode(args, enable)


if __name__ == "__main__":
    main()
