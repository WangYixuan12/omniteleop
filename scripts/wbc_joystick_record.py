#!/usr/bin/env python3
"""SAPIEN joystick follower: torso+arm whole-body IK, JOYSTICK base, head-track (sim).

The sim sibling of ``scripts/wbc_joystick_robot.py`` and the safe way to verify the
joystick paradigm before touching hardware. It renders the Vega in SAPIEN and drives it
exactly as the real joystick follower does:

  * base EXCLUDED from the whole-body IK (``WBCConfig.lock_base_in_ik``) -- the QP
    coordinates torso + arms only; the arm EEF targets are base-frame, so they ride the
    robot as the base moves;
  * the chassis is driven from the leader's joystick ``chassis_vx/vy/wz`` through the SAME
    :class:`omniteleop.follower.joystick_base.JoystickBaseShaper` the real follower uses, so
    the base is shaped identically in sim and on hardware;
  * head runs ``head_mode: track`` (neck pan/tilt via ``solve_head``; base does not follow).

Pair it with ``scripts/wbc_joystick_leader.py``. Drive with the thumbsticks (right stick =
forward/strafe, left stick X = turn) and watch the arms hold their base-frame targets while
the base moves and the neck tracks your gaze. ``--replay FILE`` replays a recorded stream.

Run in the dexmate conda env (pinocchio + pink + dexcomm + sapien).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Optional

import numpy as np

from omniteleop.follower.joystick_base import JoystickBaseShaper
from omniteleop.follower.wbc_sapien_sim import SapienSimRobot
from omniteleop.follower.whole_body_ik import (
    HEAD_FRAME,
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    VegaWholeBodyIK,
    WBCConfig,
)
from omniteleop.wbc_record import DEFAULT_REPLAY_GAP_LIMIT_MULTIPLE, ReplaySource
from omniteleop.wbc_stream import (
    HeadTargetLowPassFilter,
    HeadTargetPlanarDeadbandFilter,
    TargetInterpolator,
    _to_mat,
    vr_to_ee_targets,
    vr_to_head_target,
)
from omniteleop.wbc_teleop import JoystickTeleopConfig, VRJointSubscriber, VRTeleopConfig


def _build_joystick_ik(config):
    """VegaWholeBodyIK with the joystick-mode overrides on the canonical wbik.yaml."""
    cfg = replace(
        WBCConfig.from_yaml(config), lock_base_in_ik=True, head_mode="track",
    )
    return VegaWholeBodyIK(cfg), cfg


def _shaper_args(
    vt: VRTeleopConfig, joystick: JoystickTeleopConfig
) -> SimpleNamespace:
    """The base-shaping tunables JoystickBaseShaper.from_configs expects, from wbik.yaml."""
    return SimpleNamespace(
        base_kp_xy=vt.base_kp_xy, base_kp_yaw=vt.base_kp_yaw,
        base_max_speed=vt.base_max_speed, base_deadband=vt.base_deadband,
        base_accel=vt.base_accel, base_post_linear_deadband=vt.base_post_linear_deadband,
        base_post_angular_deadband=vt.base_post_angular_deadband,
        base_yaw_hold_in_xy=vt.base_yaw_hold_in_xy,
        joystick_stick_max_vx=vt.stick_max_vx,
        joystick_stick_max_wz=vt.stick_max_wz,
        joystick_single_axis_hysteresis_ratio=joystick.single_axis_hysteresis_ratio,
    )


class _DbgLog:
    """Per-tick joystick-base debug log, flushed to an HDF5 on teardown for offline analysis.

    Captures the full base pipeline every IK tick so a symptom like "strafe moves the base
    forward" can be traced end-to-end: ``rx_chassis`` (twist received from the leader) ->
    ``pd_raw``/``pd_err`` (shaper PD output + pose error) -> ``base_cmd`` (shaped twist sent
    to set_velocity) -> ``base_pose`` (the integrated sim base), plus ``cmd_pose`` (the
    shaper reference), the single-axis latch ``prev_axis``, and the ``estop``/``hold_base``/
    ``success`` gating. Read it back with h5py under the ``/debug`` group.
    """

    _VEC = ("rx_chassis", "base_cmd", "pd_raw", "pd_err", "cmd_pose", "base_pose",
            "left_tgt", "right_tgt")
    _INT = ("estop", "hold_base", "success", "prev_axis")
    _F32 = ("left_ee_err", "right_ee_err")

    def __init__(self, path: str) -> None:
        self.path = path
        self._rows: list[dict] = []

    def append(self, **fields) -> None:
        self._rows.append(fields)

    def __len__(self) -> int:
        return len(self._rows)

    def write(self) -> Optional[str]:
        if not self._rows:
            return None
        import h5py  # noqa: PLC0415 -- optional dep, only needed at teardown

        rows = self._rows
        col = lambda k: np.asarray([r[k] for r in rows])
        with h5py.File(self.path, "w") as fh:
            g = fh.create_group("debug")
            g.attrs["schema"] = "wbc_joystick_record_debug/v1"
            g.attrs["n_frames"] = len(rows)
            g.create_dataset("t", data=col("t").astype(np.float64))
            g.create_dataset(
                "stage", data=np.array([str(r["stage"]).encode()[:16] for r in rows], dtype="S16")
            )
            for k in self._INT:
                g.create_dataset(k, data=col(k).astype(np.int64))
            for k in self._F32:
                g.create_dataset(k, data=col(k).astype(np.float32))
            for k in self._VEC:
                g.create_dataset(k, data=col(k).astype(np.float32), compression="gzip",
                                 compression_opts=4)
        return self.path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--replay", metavar="FILE",
                        help="replay a stream HDF5 through SAPIEN instead of the live leader.")
    parser.add_argument("--namespace", default="",
                        help="Zenoh namespace (must match the leader; default empty).")
    parser.add_argument("--config", default=None,
                        help="wbik.yaml path (default canonical). base_dofs is read from it; "
                             "lock_base_in_ik / head_mode:track / head deadbands are forced.")
    parser.add_argument("--no-viewer", action="store_true", help="render headlessly.")
    parser.add_argument("--output", default=None, help="optional mp4 video path.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--debug-hdf5", default=None,
                        help="path for the per-tick base-debug HDF5 (rx_chassis, pd_raw/err, "
                             "base_cmd, cmd_pose, base_pose, single-axis latch, gating). "
                             "Default: auto-named wbc_joystick_dbg_<timestamp>.hdf5 in the cwd.")
    parser.add_argument("--no-debug-hdf5", action="store_true",
                        help="disable the per-tick base-debug HDF5 log.")
    args = parser.parse_args()

    dbg_log: Optional[_DbgLog] = None
    if not args.no_debug_hdf5:
        dbg_path = args.debug_hdf5 or f"wbc_joystick_dbg_{time.strftime('%Y%m%d_%H%M%S')}.hdf5"
        dbg_log = _DbgLog(dbg_path)
        print(f"[wbc_joystick] base-debug HDF5 -> {dbg_path} (per-tick; written on exit)")

    ik, cfg = _build_joystick_ik(args.config)
    vt = VRTeleopConfig.from_yaml(args.config)
    joystick_cfg = JoystickTeleopConfig.from_yaml(args.config)
    dt = 1.0 / vt.ik_rate
    ik.reset()
    left0 = _to_mat(ik.frame_pose(LEFT_EE_FRAME))
    right0 = _to_mat(ik.frame_pose(RIGHT_EE_FRAME))
    head0 = _to_mat(ik.frame_pose(HEAD_FRAME))
    print(f"[wbc_joystick] model nq={ik.model.nq} nv={ik.model.nv} head_mode={cfg.head_mode} "
          f"base_dofs={cfg.base_dofs} lock_base_in_ik={cfg.lock_base_in_ik} "
          f"single_axis={cfg.enable_base_single_axis}")

    robot = SapienSimRobot(
        urdf_path=cfg.urdf_path, with_viewer=not args.no_viewer,
        record_video=args.output, video_fps=args.video_fps,
        video_size=(args.width, args.height),
    )

    replay = args.replay is not None
    if replay:
        source = ReplaySource.from_hdf5(args.replay, vt.replay_speed, clock=time.perf_counter)
        print(f"[wbc_joystick] replay {args.replay}: {len(source._vr)} frames, "  # noqa: SLF001
              f"{source.total_duration:.1f}s. Ctrl-C to stop.")
    else:
        source = VRJointSubscriber(args.namespace, name="wbc_joystick_record")
        print(f"[wbc_joystick] subscribed to '{source.topic}'. Run scripts/wbc_joystick_leader.py "
              "and grip-hold to calibrate; drive with the thumbsticks. Ctrl-C stops.")

    shaper = JoystickBaseShaper.from_configs(cfg, _shaper_args(vt, joystick_cfg))
    speed = vt.replay_speed if replay else 1.0
    base_dur = (1.0 / vt.cmd_rate) / speed if vt.cmd_rate > 0 else 0.0
    interp = TargetInterpolator(base_dur, left0, right0, head0)
    head_lpf = HeadTargetLowPassFilter(vt.head_lpf_tau, head0)
    # head-track: zero the planar deadband (it exists only to stop head->base motion in
    # head_mode:ik and otherwise creates a ~17deg pan dead zone here).
    head_planar_deadband = HeadTargetPlanarDeadbandFilter(
        head0, position_deadband=0.0, yaw_deadband=0.0,
    )
    left_cmd, right_cmd, head_cmd = left0.copy(), right0.copy(), head0.copy()
    last_cmd_ns = -1
    prev_estop = True
    if hasattr(source, "start"):
        source.start()
    t0 = time.perf_counter()
    last_print = 0.0
    print("[wbc_joystick] base control: JOYSTICK -> shared JoystickBaseShaper (odom-PD + "
          "shape_twist + single-axis) -> set_velocity; e-stop / failed solve zeros the base.")

    try:
        while True:
            now = time.perf_counter()
            t = now - t0
            if robot.viewer_closed:
                break
            if replay and source.done:
                break
            vr = source.latest
            if vr is not None and vr.exit_requested:
                print("\n[wbc_joystick] leader requested stop (left Y).")
                break

            estop = bool(vr.estop) if vr is not None else True
            if prev_estop and not estop:
                ik.reset()
                robot.reset_base()
                shaper.reset(np.zeros(3))
                left_cmd, right_cmd, head_cmd = left0.copy(), right0.copy(), head0.copy()
                interp.reset(left0, right0, head0)
                head_lpf.reset(head0)
                head_planar_deadband.reset(head0)
                last_cmd_ns = -1
                print("\n[wbc_joystick] engage -> reset robot to origin/nominal.")
            prev_estop = estop

            left_cmd, right_cmd = vr_to_ee_targets(vr, left_cmd, right_cmd)
            head_cmd = vr_to_head_target(vr, head_cmd)
            if (vr is not None and vr.timestamp_ns != last_cmd_ns
                    and vr.left_ee_pose and vr.right_ee_pose):
                last_cmd_ns = vr.timestamp_ns
                seg = source.segment_duration if replay else None
                duration = base_dur if seg is None else float(seg)
                if replay and base_dur > 0.0 and duration > DEFAULT_REPLAY_GAP_LIMIT_MULTIPLE * base_dur:
                    raise RuntimeError(
                        f"replay command gap {duration:.6g}s exceeds "
                        f"{DEFAULT_REPLAY_GAP_LIMIT_MULTIPLE:g}x nominal {base_dur:.6g}s")
                interp.push(left_cmd, right_cmd, head_cmd, now=now, duration=duration)
            left_target, right_target, head_target = interp.at(now)
            head_target = head_lpf.filter(head_target, dt)
            head_target = head_planar_deadband.filter(head_target)

            # Torso + arms (base locked out of the IK); head via the stiff live-config tracker.
            head_joints = ik.solve_head(head_target, dt)
            result = ik.solve(left_target, right_target, dt, head_joints=head_joints)
            robot.left_arm.set_joint_pos(result.left_arm)
            robot.right_arm.set_joint_pos(result.right_arm)
            robot.torso.set_joint_pos(result.torso)
            robot.head.set_joint_pos(result.head)

            # Base: joystick -> shared shaper (measured = rendered SAPIEN base pose).
            hold_base = estop or not result.success
            pd_raw = np.zeros(3)
            pd_err = np.zeros(3)
            if hold_base:
                shaper.hold(np.asarray(robot.base_pose, dtype=float))
                base_cmd = np.zeros(3)
            else:
                joy = np.array(
                    [float(vr.chassis_vx), float(vr.chassis_vy), float(vr.chassis_wz)],
                    dtype=float,
                ) if vr is not None else np.zeros(3)
                base_cmd, pd_raw, pd_err = shaper.step(
                    joy, np.asarray(robot.base_pose, dtype=float), dt)
            robot.chassis.set_velocity(
                vx=float(base_cmd[0]), vy=float(base_cmd[1]), wz=float(base_cmd[2]),
                wait_time=0.0, sequential_steering=False,
            )
            robot.set_targets(left_target, right_target, head_target)
            robot.step(dt)
            robot.render(record_video=args.output is not None)

            # rx_chassis = joystick twist RECEIVED from the leader; base_cmd = shaped twist
            # sent to set_velocity; base_pose = integrated sim base. Logged EVERY tick to the
            # debug HDF5 (analyse the base direction offline), printed at ~2 Hz.
            stage = vr.calib_stage if vr is not None else "?"
            rxc = ((float(vr.chassis_vx), float(vr.chassis_vy), float(vr.chassis_wz))
                   if vr is not None else (0.0, 0.0, 0.0))
            if dbg_log is not None:
                dbg_log.append(
                    t=t, stage=stage, estop=int(estop), hold_base=int(hold_base),
                    success=int(result.success),
                    rx_chassis=np.asarray(rxc, dtype=np.float32),
                    base_cmd=np.asarray(base_cmd, dtype=np.float32),
                    pd_raw=np.asarray(pd_raw, dtype=np.float32),
                    pd_err=np.asarray(pd_err, dtype=np.float32),
                    cmd_pose=np.asarray(shaper.cmd_pose, dtype=np.float32),
                    prev_axis=(-1 if shaper.prev_axis is None else int(shaper.prev_axis)),
                    base_pose=np.asarray(robot.base_pose, dtype=np.float32),
                    left_ee_err=float(result.left_ee_error),
                    right_ee_err=float(result.right_ee_error),
                    left_tgt=np.asarray(left_target[:3, 3], dtype=np.float32),
                    right_tgt=np.asarray(right_target[:3, 3], dtype=np.float32),
                )
            if t - last_print >= 0.5:
                last_print = t
                bx, by, byaw = robot.base_pose
                print(f"[follower dbg] t={t:5.1f}s {stage:7s} estop={estop} "
                      f"hold_base={hold_base} rx_chassis=[{rxc[0]:+.2f} {rxc[1]:+.2f} "
                      f"{rxc[2]:+.2f}] cmd=[{base_cmd[0]:+.2f} {base_cmd[1]:+.2f} "
                      f"{base_cmd[2]:+.2f}] base=[{bx:+.2f} {by:+.2f} "
                      f"{np.degrees(byaw):+4.0f}deg] axis={shaper.prev_axis}")

            sleep = dt - (time.perf_counter() - now)
            if sleep > 0:
                # Pace the loop at ik_rate. ReplaySource is wall-clock paced, so replay plays
                # at real time either way -- sleeping just avoids busy-spinning a core.
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n[wbc_joystick] interrupted.")
    finally:
        if hasattr(robot.chassis, "stop"):
            robot.chassis.stop()
        if hasattr(source, "close"):
            source.close()
        robot.close()
        if dbg_log is not None and len(dbg_log):
            written = dbg_log.write()
            print(f"\n[wbc_joystick] base-debug HDF5 -> {written} ({len(dbg_log)} ticks)")
        print("\n[wbc_joystick] stopped.")


if __name__ == "__main__":
    main()
