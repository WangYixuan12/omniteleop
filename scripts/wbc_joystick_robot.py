#!/usr/bin/env python3
"""Real-robot joystick follower: torso+arm whole-body IK, JOYSTICK base, head-track.

The joystick-teleop sibling of ``scripts/wbc_vr_robot.py``. It reuses that follower's
``HardwareDriver`` (hardware bring-up, homing, per-joint clamp/dispatch, grippers,
odometry, episode recording) and its ``run_loop`` unchanged, and changes only how the
BASE moves:

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
        self._shaper = JoystickBaseShaper.from_configs(cfg, args)
        super().__init__(args, ik, cfg, enable, traj=traj)

    def engage_reset(self, ik, left0, right0, head0) -> None:
        """Zero the joystick reference in lock-step with the odom origin at engage."""
        super().engage_reset(ik, left0, right0, head0)  # resets odom origin to (0,0,0)
        self._shaper.reset(np.zeros(3))

    def _read_joystick(self) -> np.ndarray:
        """Latest leader joystick body twist ``(vx, vy, wz)`` (zeros if no command yet)."""
        vr = getattr(self._source, "latest", None)
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
        snap = self._odom.snapshot()
        self._dbg["odom"] = snap
        chassis = self.robot.chassis
        if hold:
            # Park the joystick reference on the measured base so there is no lunge on
            # resume, and zero the wheels (the parent's hold behavior).
            self._shaper.hold(snap["pose"])
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
        cmd, pd_raw, pd_err = self._shaper.step(rx_chassis, snap["pose"], dt)
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

    def _recording_control_metadata(self) -> dict:
        return {
            "control_mode": np.bytes_("joystick"),
            "eef_target_frame": np.bytes_("current_base"),
            "base_reference_source": np.bytes_("joystick_integrator"),
            "action_base_pose_semantics": np.bytes_("joystick_command_reference"),
        }

    def record_tick(self, result, hold, *args, **kwargs):
        """Record the JOYSTICK reference pose as action/base/pose (not the locked origin).

        With the base excluded from the IK, ``result.base_pose`` is a constant origin, so it
        would make ``action/base/pose`` useless. Substitute the shaper's integrated command
        pose -- the base trajectory the operator commanded -- which is what downstream tools
        want (the policy schema ignores this field; ``vis_episode`` uses obs/base/pose from
        odometry). The semantics of this field therefore differ from the WBC follower's
        solver-commanded ``result.base_pose``.

        The substitution (and its allocation) is skipped whenever the base recorder would
        drop the frame anyway -- on ``hold`` or when not recording -- so this stays free on
        the 100 Hz loop in the common no-record case.
        """
        if not hold and self._episode is not None and self._episode.recording:
            result = replace(result, base_pose=np.asarray(self._shaper.cmd_pose, dtype=float))
        return super().record_tick(result, hold, *args, **kwargs)


def _build_joystick_ik(config: Optional[str]) -> tuple[VegaWholeBodyIK, WBCConfig]:
    """Build the IK with joystick-mode overrides on top of the canonical wbik.yaml."""
    cfg = replace(
        WBCConfig.from_yaml(config),
        lock_base_in_ik=True,   # base excluded from the whole-body IK (joystick-driven)
        head_mode="track",      # neck pan/tilt tracks the headset; base does not follow it
    )
    return VegaWholeBodyIK(cfg), cfg


def _run_joystick_ik_mode(args: argparse.Namespace, enable: dict) -> None:
    """Replay or live-drive the joystick follower (mirrors wbc_vr_robot._run_ik_mode)."""
    ik, cfg = _build_joystick_ik(args.config)
    print(f"[wbc_joystick_robot] model nq={ik.model.nq} nv={ik.model.nv} "
          f"head_mode={cfg.head_mode} base_dofs={cfg.base_dofs} lock_base_in_ik="
          f"{cfg.lock_base_in_ik} urdf={cfg.urdf_path.rsplit('/', 1)[-1]}")
    replay = args.replay is not None
    clock: Callable[[], float] = time.perf_counter

    if replay:
        source = ReplaySource.from_hdf5(args.replay, args.speed, clock=clock)
        print(f"[wbc_joystick_robot] replay {args.replay}: {len(source._vr)} frames, "  # noqa: SLF001
              f"{source.total_duration:.1f}s at {args.speed:g}x")
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
            "eef_target_frame": "current_base",
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
          f"single_axis={cfg.enable_base_single_axis}")
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
                      help="drive from a stream HDF5 (replays the recorded joystick + targets).")
    mode.add_argument("--source", choices=("live",), default=None,
                      help="drive from the live leader (default when --replay is omitted).")
    parser.add_argument("--config", default=None,
                        help="wbik.yaml path (default: canonical follower/wbik.yaml). "
                             "base_dofs is read from it; lock_base_in_ik / head_mode:track / "
                             "head deadbands are forced for joystick mode.")
    parser.add_argument("--enable", default="arms,torso,head,base",
                        help="comma list of DOF groups to actuate (arms,torso,head,base or "
                             "'all'/'none'). Default all. The base needs the chassis to move.")
    parser.add_argument("--namespace", default="",
                        help="Zenoh namespace (must match the leader; default empty).")
    parser.add_argument("--debug-dir", dest="debug_dir", default=None,
                        help="write per-tick /debug HDF5 logs under this directory.")
    parser.add_argument("--record", action="store_true",
                        help="record an episode while engaged (vr_reader EpisodeRecorder "
                             "format). Needs --enable arms,torso,head,base.")
    parser.add_argument("--save-dir", default=_base.DEFAULT_SAVE_DIR,
                        help=f"directory for recorded episodes (default {_base.DEFAULT_SAVE_DIR}).")
    parser.add_argument("--record-rate", type=float, default=_base.DEFAULT_RECORD_RATE,
                        help=f"episode record cadence in Hz (default {_base.DEFAULT_RECORD_RATE:g}).")
    parser.add_argument("--record-stale-grace", type=float, default=_base.DEFAULT_RECORD_STALE_GRACE,
                        help="seconds to retry a stale camera frame before aborting a take.")
    hw = parser.add_argument_group("hardware")
    hw.add_argument("--home-tol", type=float, default=_base.DEFAULT_HOME_TOL)
    hw.add_argument("--home-settle", type=float, default=_base.DEFAULT_HOME_SETTLE)
    hw.add_argument("--max-joint-step", type=float, default=_base.DEFAULT_MAX_JOINT_STEP)
    hw.add_argument("--source-timeout", type=float, default=_base.DEFAULT_SOURCE_TIMEOUT)
    hw.add_argument("--drive-state-mode", choices=("ms", "rad"), default="ms")
    hw.add_argument("--base-quiet-hold-s", type=float, default=_base.DEFAULT_BASE_QUIET_HOLD_S)
    args = parser.parse_args()

    _bind_vr_teleop(args, parser)
    for flag, val in (("--home-tol", args.home_tol), ("--max-joint-step", args.max_joint_step),
                      ("--home-settle", args.home_settle),
                      ("--record-stale-grace", args.record_stale_grace)):
        if not np.isfinite(val) or val < 0.0:
            parser.error(f"{flag} must be finite and >= 0")
    if not np.isfinite(args.base_quiet_hold_s):
        parser.error("--base-quiet-hold-s must be finite")
    for flag, val in (("--source-timeout", args.source_timeout), ("--record-rate", args.record_rate)):
        if not np.isfinite(val) or val <= 0.0:
            parser.error(f"{flag} must be finite and > 0")
    try:
        enable = parse_enable_mask(args.enable)
    except ValueError as exc:
        parser.error(str(exc))

    _run_joystick_ik_mode(args, enable)


if __name__ == "__main__":
    main()
