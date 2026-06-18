#!/usr/bin/env python3
"""Step the head through discrete poses, HOLDING the camera still at each one, to
check the head forward kinematics in ``zed_base_pose`` on hardware. The base is
parked, so for ANY head pose a correct FK must invert the head motion and report
the SAME (still) base in ``scripts/zed_base_pose_node.py``; a wrong
frame/axis/sign/geometry shows up as the estimated base shifting between poses.

Motion = step-and-dwell: move smoothly to the next position (``--move-s``), then
HOLD it fixed for ``--dwell-s`` (default 4 s). The dwell is the point — while the
camera is static the ZED estimate settles, so each hold gives a clean base
reading; compare the readings across holds (they should all coincide).

Two complementary phases, both anchored at whatever pose the head starts in (read
live), so they work when the head is pre-pitched to stay upright under a crouched
torso (e.g. torso_j2≈π/2 → head_j1≈-0.79):

* ``pitch`` — step δ on the ANTIPARALLEL pair head_j1 & head_j3 together
  (head_j2 forced to 0). Net rotation is ``Ry(δ)·Ry(-δ)=const`` → the camera
  keeps a FIXED orientation ("looks the same direction") while the lever arm
  TRANSLATES it (tens of mm) between holds. Clean, rotation-free; tests the
  translational FK geometry. NOTE: INSENSITIVE to the zed_depth_frame↔
  zed_left_camera 11.5 mm offset (a constant translation offset cancels with no
  camera rotation).

* ``yaw`` — step δ on head_j2 (the camera pans/rotates to each hold). This DOES
  exercise the rotational FK and the 11.5 mm frame choice: with the correct
  (left-eye) frame the parked base reads the same at every hold; the wrong
  (depth) frame would offset it a few mm per hold (reported in the plan).

How to run (three terminals; torso and base stay still)::

    # A — head ZED publisher with positional tracking
    python tests/test_head_zedx_depth.py --enable-tracking \
        [--area-file /path/lab.area --localization-only]
    # B — base-pose fusion + trajectory video (watch this!)
    python scripts/zed_base_pose_node.py \
        --save_video /home/dexmate/yixuan/Dexmate/SLAM/test/tmp.mp4
    # C — THIS script (dexmate env, robot reachable)
    python scripts/sweep_head_fixed_view.py            # both phases

Expected: in B, each 4 s hold is a flat segment; all the flat segments sit at the
same base XY/z (within ~mm ZED parked noise) while this script reports the camera
sitting at a different position/orientation each hold.

``--dry-run`` prints the planned hold poses + FK predictions with NO robot
connection and NO motion (pass --head / --torso to match your posture).

SAFETY: only the 3 head joints move; torso/arms/base are never commanded. Keep
the head clear. Ctrl-C / completion returns the head to its starting pose.
"""

from __future__ import annotations

import dataclasses
import math
import os
import pathlib
import sys
import time
from typing import Callable, Optional

import numpy as np
import tyro
from loguru import logger

sys.path.insert(0, (pathlib.Path(__file__).resolve().parent.parent / "src").as_posix())

from omniteleop.follower.zed_base_pose import (  # noqa: E402
    VegaHeadCamFK,
    ZedBasePoseEstimator,
)

# URDF head revolute velocity limit (vega_1.urdf head_j* <limit velocity="3.2"/>).
_HEAD_VEL_LIMIT = 3.2


@dataclasses.dataclass
class Args:
    mode: str = "both"
    """Which phase(s) to run: 'pitch' (fixed-view translation), 'yaw' (rotation),
    or 'both' (pitch then yaw, sequentially)."""

    pitch_amplitude: float = 0.6
    """Peak |δ| (rad) added to head_j1 & head_j3 in the pitch phase. Clamped to
    the room left around the live head pose (head_j1 near its limit shrinks it)."""

    yaw_amplitude: float = 0.6
    """Peak |δ| (rad) added to head_j2 in the yaw phase. Clamped to joint limits."""

    steps: int = 5
    """Number of discrete hold positions per phase, evenly spaced over [-amp, +amp]
    (odd values include the δ=0 baseline)."""

    dwell_s: float = 4.0
    """Seconds to hold the head STILL at each position (the clean-reading window)."""

    move_s: float = 1.5
    """Seconds for each smooth move between hold positions (and into/out of a phase)."""

    hold_rate_hz: float = 5.0
    """Rate (Hz) at which the hold target is re-sent during a dwell to keep it held."""

    return_to: str = "initial"
    """Where to leave the head when done: 'initial' (live start pose) or 'zero'."""

    margin_rad: float = 0.05
    """Safety margin kept away from each head joint limit when clamping amplitude."""

    max_speed_frac: float = 0.5
    """Reject a phase whose biggest move's mean speed exceeds this fraction of the
    3.2 rad/s head velocity limit."""

    dry_run: bool = False
    """Print planned hold poses + FK predictions and exit; never move the robot."""

    head: str = "0,0,0"
    """Head joints (rad, 'j1,j2,j3') assumed for FK in --dry-run (ignored live)."""

    torso: str = "0,0,0"
    """Torso joints (rad, 'j1,j2,j3') assumed for FK in --dry-run (ignored live)."""

    urdf_path: Optional[str] = None
    """Vega URDF override; defaults to vega_1.urdf from dexmate_urdf."""


@dataclasses.dataclass
class Phase:
    name: str  # "pitch" | "yaw"
    amplitude: float
    target_fn: Callable[[np.ndarray, float], list]
    fixed_orientation: bool


def _parse_vec3(s: str, what: str) -> np.ndarray:
    parts = [p for p in s.replace(",", " ").split() if p]
    if len(parts) != 3:
        raise ValueError(f"{what} must be 3 comma/space-separated numbers, got {s!r}")
    return np.asarray([float(p) for p in parts], dtype=np.float64)


def _joint_limits(fk: VegaHeadCamFK, name: str) -> tuple[float, float]:
    jid = fk.model.getJointId(name)
    iq = fk.model.joints[jid].idx_q
    return float(fk.model.lowerPositionLimit[iq]), float(fk.model.upperPositionLimit[iq])


def _phase_deltas(amplitude: float, steps: int) -> np.ndarray:
    """Discrete hold offsets, evenly spaced over [-amplitude, +amplitude]."""
    return np.linspace(-amplitude, amplitude, steps)


def _pitch_target(head0: np.ndarray, delta: float) -> list[float]:
    """Antiparallel j1/j3 step, j2 forced to 0 → camera keeps a fixed orientation."""
    return [float(head0[0] + delta), 0.0, float(head0[2] + delta)]


def _yaw_target(head0: np.ndarray, delta: float) -> list[float]:
    """Pan the camera: only head_j2 moves; j1/j3 held at the live start pose."""
    return [float(head0[0]), float(head0[1] + delta), float(head0[2])]


def _rot_dev_deg(R_ref: np.ndarray, R: np.ndarray) -> float:
    cos = (np.trace(R_ref.T @ R) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(cos)))))


def _safe_amp_pitch(fk: VegaHeadCamFK, head0: np.ndarray, margin: float) -> float:
    lo1, hi1 = _joint_limits(fk, "head_j1")
    lo3, hi3 = _joint_limits(fk, "head_j3")
    return min(hi1 - head0[0], head0[0] - lo1, hi3 - head0[2], head0[2] - lo3) - margin


def _safe_amp_yaw(fk: VegaHeadCamFK, head0: np.ndarray, margin: float) -> float:
    lo2, hi2 = _joint_limits(fk, "head_j2")
    return min(hi2 - head0[1], head0[1] - lo2) - margin


def _clamp_amplitude(requested: float, amp_max: float, label: str) -> float:
    if amp_max <= 0.0:
        raise ValueError(
            f"{label}: no safe travel around the start pose (max {amp_max:.3f} rad); "
            "the head is at a joint limit — move it before sweeping, or lower --margin-rad."
        )
    if requested <= 0.0:
        raise ValueError(f"{label}: amplitude must be > 0, got {requested}")
    if requested > amp_max:
        logger.warning(
            f"{label}: amplitude {requested:.3f} exceeds the safe limit {amp_max:.3f} rad "
            f"around the start pose; clamping to {amp_max:.3f}."
        )
        return amp_max
    return requested


def _max_move_speed(amplitude: float, args: Args) -> float:
    """Mean speed (rad/s) of the largest single move: δ=0→first hold, between
    holds, and the return to 0, all over ``move_s``."""
    path = np.concatenate([[0.0], _phase_deltas(amplitude, args.steps), [0.0]])
    max_jump = float(np.max(np.abs(np.diff(path)))) if path.size > 1 else 0.0
    return max_jump / args.move_s


def _check_speed(amplitude: float, args: Args, label: str) -> float:
    peak = _max_move_speed(amplitude, args)
    cap = args.max_speed_frac * _HEAD_VEL_LIMIT
    if peak > cap:
        raise ValueError(
            f"{label}: biggest move's speed {peak:.2f} rad/s exceeds "
            f"{args.max_speed_frac:.0%} of the {_HEAD_VEL_LIMIT} rad/s head limit "
            f"({cap:.2f} rad/s). Lower the amplitude/steps or raise --move-s."
        )
    return peak


def _build_phases(args: Args, fk: VegaHeadCamFK, head0: np.ndarray) -> list[Phase]:
    if args.mode not in ("pitch", "yaw", "both"):
        raise ValueError(f"--mode must be 'pitch', 'yaw', or 'both', got {args.mode!r}")
    phases: list[Phase] = []
    if args.mode in ("pitch", "both"):
        amp = _clamp_amplitude(
            args.pitch_amplitude, _safe_amp_pitch(fk, head0, args.margin_rad), "pitch"
        )
        _check_speed(amp, args, "pitch")
        phases.append(Phase("pitch", amp, _pitch_target, fixed_orientation=True))
    if args.mode in ("yaw", "both"):
        amp = _clamp_amplitude(
            args.yaw_amplitude, _safe_amp_yaw(fk, head0, args.margin_rad), "yaw"
        )
        _check_speed(amp, args, "yaw")
        phases.append(Phase("yaw", amp, _yaw_target, fixed_orientation=False))
    return phases


def _depth_frame_offsets(
    fk: VegaHeadCamFK, torso: np.ndarray, head0: np.ndarray, phase: Phase, deltas: np.ndarray
) -> np.ndarray:
    """Base offset (mm) the WRONG (zed_depth_frame) FK would report at each hold of
    a PARKED base, anchored at δ=0. Correct (left-eye) FK gives 0 at every hold;
    this is the discriminating signal for the 11.5 mm frame fix."""
    fk_depth = VegaHeadCamFK(urdf_path=fk.urdf_path, camera_frame="zed_depth_frame")
    est = ZedBasePoseEstimator(fk_depth)
    q0 = phase.target_fn(head0, 0.0)
    est.update(fk.base_T_cam(torso, q0), torso, q0)  # anchor at the δ=0 baseline
    out = []
    for d in deltas:
        q = phase.target_fn(head0, float(d))
        out.append(1000.0 * float(np.linalg.norm(est.update(fk.base_T_cam(torso, q), torso, q)[:3, 3])))
    return np.asarray(out)


def _log_phase_plan(
    fk: VegaHeadCamFK, torso: np.ndarray, head0: np.ndarray, phase: Phase, args: Args
) -> None:
    deltas = _phase_deltas(phase.amplitude, args.steps)
    cam0 = fk.base_T_cam(torso, phase.target_fn(head0, 0.0))
    Ts = [fk.base_T_cam(torso, phase.target_fn(head0, float(d))) for d in deltas]
    pos = np.array([T[:3, 3] for T in Ts])
    rot = max(_rot_dev_deg(cam0[:3, :3], T[:3, :3]) for T in Ts)
    span = 1000.0 * (pos.max(0) - pos.min(0))
    logger.info(
        f"[{phase.name}] {args.steps} holds over ±{math.degrees(phase.amplitude):.0f}°, "
        f"{args.dwell_s:g}s dwell + {args.move_s:g}s moves "
        f"(~{args.steps * (args.dwell_s + args.move_s):.0f}s), "
        f"max move speed {_max_move_speed(phase.amplitude, args):.2f} rad/s"
    )
    if phase.fixed_orientation:
        if rot > 1e-3:
            raise RuntimeError(
                f"[{phase.name}] orientation NOT constant (max dev {rot:.4f}° at "
                f"torso={np.round(torso,3).tolist()}, head0={np.round(head0,3).tolist()}). "
                "The head_j1/head_j3 antiparallel cancellation is broken — check the URDF."
            )
        logger.info(
            f"[{phase.name}] camera FIXED orientation ({rot:.1e}°), translates "
            f"span(x,y,z)=({span[0]:.0f},{span[1]:.0f},{span[2]:.0f}) mm across holds → "
            "every hold's base must coincide. (insensitive to the 11.5 mm frame offset)"
        )
    else:
        off = _depth_frame_offsets(fk, torso, head0, phase, deltas)
        logger.info(
            f"[{phase.name}] camera ROTATES across ±{math.degrees(phase.amplitude):.0f}°, "
            f"translates span(x,y,z)=({span[0]:.0f},{span[1]:.0f},{span[2]:.0f}) mm → every "
            f"hold's base must coincide. Frame check: WRONG(depth) frame would offset the "
            f"base up to {off.max():.1f} mm per hold; correct(left) frame gives 0."
        )


def _configure_zenoh_like_dexcontrol() -> Optional[pathlib.Path]:
    """Mirror dexcontrol's ZENOH_CONFIG discovery (see the head publisher)."""
    existing = os.getenv("ZENOH_CONFIG")
    if existing:
        return pathlib.Path(existing).expanduser()
    base_dir = pathlib.Path("~/.dexmate/comm/zenoh").expanduser()
    for pattern in ("**/*.dzcfg", "**/zenoh*config*.json5"):
        matches = sorted(p for p in base_dir.glob(pattern) if p.is_file())
        if matches:
            os.environ["ZENOH_CONFIG"] = matches[0].as_posix()
            return matches[0]
    return None


def _hold_still(robot, target: list[float], dwell_s: float, hold_rate_hz: float) -> None:
    """Keep the head commanded at ``target`` for ``dwell_s`` (re-sent to stay held)."""
    period = 1.0 / hold_rate_hz
    t_end = time.perf_counter() + dwell_s
    while True:
        now = time.perf_counter()
        if now >= t_end:
            break
        robot.head.set_joint_pos(target, wait_time=0.0)
        time.sleep(min(period, t_end - now))


def _execute_phase(robot, fk: VegaHeadCamFK, torso: np.ndarray, head0: np.ndarray,
                   phase: Phase, args: Args) -> None:
    cam0 = fk.base_T_cam(torso, phase.target_fn(head0, 0.0))
    deltas = _phase_deltas(phase.amplitude, args.steps)
    n = len(deltas)
    logger.info(f"[{phase.name}] stepping through {n} holds — read the base during each "
                f"{args.dwell_s:g}s hold (it should be the same at every hold).")
    for i, d in enumerate(deltas):
        target = phase.target_fn(head0, float(d))
        logger.info(f"[{phase.name}] → pos {i+1}/{n}: δ={math.degrees(d):+6.1f}° "
                    f"head={[round(x,3) for x in target]} (move {args.move_s:g}s)")
        robot.head.set_joint_pos(target, wait_time=args.move_s, exit_on_reach=True)
        act = np.asarray(robot.head.get_joint_pos(), dtype=np.float64)
        Tc = fk.base_T_cam(torso, act)
        cam_d = 1000.0 * float(np.linalg.norm(Tc[:3, 3] - cam0[:3, 3]))
        rot = _rot_dev_deg(cam0[:3, :3], Tc[:3, :3])
        logger.info(f"[{phase.name}]   HOLD {args.dwell_s:g}s — cam Δpos={cam_d:5.1f}mm "
                    f"Δrot={rot:5.1f}° (base should read steady, same as other holds)")
        _hold_still(robot, target, args.dwell_s, args.hold_rate_hz)


def _run_robot(args: Args, fk: VegaHeadCamFK) -> None:
    cfg = _configure_zenoh_like_dexcontrol()
    logger.info(f"Zenoh config: {cfg if cfg else '<dexcomm default>'}; "
                f"ROBOT_NAME={os.getenv('ROBOT_NAME')!r}")
    from dexcontrol.robot import Robot

    robot = Robot()
    try:
        try:
            robot.head.set_mode("enable")
        except Exception as exc:  # noqa: BLE001 — firmware/mode quirks shouldn't abort
            logger.warning(f"head.set_mode('enable') failed ({exc}); continuing.")

        head0 = np.asarray(robot.head.get_joint_pos(), dtype=np.float64)
        torso = np.asarray(robot.torso.get_joint_pos(), dtype=np.float64)
        if head0.shape != (3,) or torso.shape != (3,):
            raise SystemExit(
                f"expected 3 head + 3 torso joints, got head {head0.shape}, torso {torso.shape}"
            )
        logger.info(f"head start = {np.round(head0,4).tolist()}, "
                    f"torso (held) = {np.round(torso,4).tolist()}")

        phases = _build_phases(args, fk, head0)
        for ph in phases:
            _log_phase_plan(fk, torso, head0, ph, args)
        end_target = head0.tolist() if args.return_to == "initial" else [0.0, 0.0, 0.0]

        try:
            for ph in phases:
                _execute_phase(robot, fk, torso, head0, ph, args)
        except KeyboardInterrupt:
            logger.info("Stopping early — returning head.")

        logger.info(f"Returning head to {args.return_to} {np.round(end_target,4).tolist()} "
                    f"over {args.move_s:g}s…")
        robot.head.set_joint_pos(end_target, wait_time=args.move_s, exit_on_reach=True)
    finally:
        robot.shutdown()
    logger.info("Done.")


def _run_dry(args: Args, fk: VegaHeadCamFK) -> None:
    head0 = _parse_vec3(args.head, "--head")
    torso = _parse_vec3(args.torso, "--torso")
    logger.info(f"[dry-run] head start = {head0.tolist()}, torso = {torso.tolist()}")
    phases = _build_phases(args, fk, head0)
    for ph in phases:
        _log_phase_plan(fk, torso, head0, ph, args)
        deltas = _phase_deltas(ph.amplitude, args.steps)
        cam0 = fk.base_T_cam(torso, ph.target_fn(head0, 0.0))
        off = (_depth_frame_offsets(fk, torso, head0, ph, deltas)
               if not ph.fixed_orientation else np.zeros_like(deltas))
        logger.info(f"[{ph.name}] hold positions:")
        logger.info(f"  {'#':>2} {'δ[deg]':>8} {'head_cmd':>24} {'camΔpos[mm]':>11} "
                    f"{'camΔrot[deg]':>12} {'depthErr[mm]':>12}")
        for i, d in enumerate(deltas):
            tgt = ph.target_fn(head0, float(d))
            T = fk.base_T_cam(torso, tgt)
            dp = 1000.0 * float(np.linalg.norm(T[:3, 3] - cam0[:3, 3]))
            dr = _rot_dev_deg(cam0[:3, :3], T[:3, :3])
            logger.info(f"  {i+1:>2} {math.degrees(d):+8.1f} "
                        f"[{tgt[0]:+.3f},{tgt[1]:+.3f},{tgt[2]:+.3f}] {dp:11.1f} {dr:12.2f} "
                        f"{off[i]:12.1f}")
    logger.info("Dry-run only — no robot connection, no motion.")


def main() -> None:
    args = tyro.cli(Args)
    for name, val in {"dwell_s": args.dwell_s, "move_s": args.move_s,
                      "hold_rate_hz": args.hold_rate_hz}.items():
        if val <= 0.0:
            raise ValueError(f"--{name.replace('_','-')} must be > 0, got {val}")
    if args.steps < 2:
        raise ValueError(f"--steps must be >= 2, got {args.steps}")
    if args.return_to not in ("initial", "zero"):
        raise ValueError(f"--return-to must be 'initial' or 'zero', got {args.return_to!r}")

    fk = VegaHeadCamFK(urdf_path=args.urdf_path)
    logger.info(f"FK model: {fk.urdf_path} → camera frame {fk.camera_frame!r}")
    if args.dry_run:
        _run_dry(args, fk)
    else:
        _run_robot(args, fk)


if __name__ == "__main__":
    main()
