#!/usr/bin/env python3
"""Stream a sinusoidal position command on a single right-arm joint.

The other six joints are held at ``INIT_RIGHT_ARM_JOINTS`` and the target
joint tracks ``init_j + amplitude * sin(2*pi*f*t)``. A half-cosine envelope
ramps the amplitude up and back down so the cycle never starts/ends with a
step input.

Per iteration we log:
* ``t_rel_s``      — seconds since the sinusoid started (monotonic)
* ``t_send_ns``    — ``time.time_ns()`` immediately before publishing
* ``robot_ts_ns``  — ``right_arm.get_timestamp_ns()`` (robot-clock, raw)
* ``cmd_j_rad``    — commanded position for the swept joint
* ``obs_j_rad``    — observed position for the swept joint
* ``obs_vel_j_rad_s`` — observed velocity for the swept joint

After the run we print rms / max tracking error and a rough phase-lag estimate
(cross-correlation peak between zero-mean cmd and obs traces).

Heads-up: ``Robot()`` instantiation will move the head to its home pose as
part of ``_set_default_state``. Stop ``vr_robot_controller*`` first.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
from dexcomm import RateLimiter
from dexcontrol.robot import Robot
from loguru import logger

from omniteleop.common.vr_mode_const import INIT_RIGHT_ARM_JOINTS


def _envelope(t: float, total: float, ramp: float) -> float:
    """Half-cosine ramp at both ends so the sinusoid starts/ends smoothly."""
    if ramp <= 0.0:
        return 1.0
    if t < ramp:
        return 0.5 * (1.0 - math.cos(math.pi * t / ramp))
    if t > total - ramp:
        s = (total - t) / ramp
        return 0.5 * (1.0 - math.cos(math.pi * max(s, 0.0)))
    return 1.0


def _phase_lag_ms(cmd: np.ndarray, obs: np.ndarray, dt_s: float) -> float:
    """Estimate obs-vs-cmd lag (positive = obs trails cmd) via cross-correlation."""
    c = cmd - np.mean(cmd)
    o = obs - np.mean(obs)
    if np.linalg.norm(c) == 0 or np.linalg.norm(o) == 0:
        return float("nan")
    corr = np.correlate(o, c, mode="full")
    lag_samples = int(np.argmax(corr) - (len(c) - 1))
    return lag_samples * dt_s * 1e3


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--joint-idx", type=int, default=3,
                   help="Right-arm joint index (0..6). Default 3 = R_arm_j4.")
    p.add_argument("--amplitude", type=float, default=0.2,
                   help="Sinusoid amplitude (rad).")
    p.add_argument("--freq-hz", type=float, default=0.5,
                   help="Sinusoid frequency (Hz).")
    p.add_argument("--duration-s", type=float, default=10.0,
                   help="Total sweep duration (seconds), including ramps.")
    p.add_argument("--ramp-s", type=float, default=1.0,
                   help="Half-cosine envelope length at each end (seconds).")
    p.add_argument("--hold-rate-hz", type=float, default=100.0,
                   help="Command-publish rate (matches dexmate-onboard watchdog).")
    p.add_argument("--out", default="/home/yixuan/Dexmate/debug/arm_sinusoid/sinusoid.csv",
                   help="Output CSV path. Parent dir is created if missing.")
    args = p.parse_args()

    if not (0 <= args.joint_idx < 7):
        raise SystemExit(f"--joint-idx must be in [0, 7), got {args.joint_idx}")
    if args.ramp_s * 2.0 > args.duration_s:
        raise SystemExit("2 * --ramp-s must be <= --duration-s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    init = np.array(INIT_RIGHT_ARM_JOINTS, dtype=np.float64)
    j = args.joint_idx
    omega = 2.0 * math.pi * args.freq_hz

    logger.info("Connecting to robot (heads-up: head will move to home) ...")
    robot = Robot()

    try:
        if not robot.right_arm.wait_for_active(5.0):
            raise SystemExit("right_arm not active within 5 s")

        logger.info("Moving right arm to INIT_RIGHT_ARM_JOINTS ...")
        robot.right_arm.set_joint_pos(
            init, wait_time=2.0, exit_on_reach=True,
            exit_on_reach_kwargs={"tolerance": 0.02},
        )
        time.sleep(0.5)

        # Reach into the dexcomm publisher so we can stamp `timestamp_ns` on
        # each publish (mirrors probe_arm_latency_client.py).
        publisher = robot.right_arm._publisher  # noqa: SLF001
        cmd = init.copy()
        limiter = RateLimiter(args.hold_rate_hz)
        rows: list[dict] = []

        logger.info(
            f"Streaming sinusoid: j={j} (R_arm_j{j+1})  A={args.amplitude} rad  "
            f"f={args.freq_hz} Hz  T={args.duration_s} s @ {args.hold_rate_hz:g} Hz"
        )
        t_start = time.monotonic()
        while True:
            t_rel = time.monotonic() - t_start
            if t_rel >= args.duration_s:
                break

            env = _envelope(t_rel, args.duration_s, args.ramp_s)
            cmd_j = float(init[j] + env * args.amplitude * math.sin(omega * t_rel))
            cmd[j] = cmd_j

            t_send = time.time_ns()
            publisher.publish({"pos": cmd.tolist(), "timestamp_ns": t_send})

            try:
                obs_pos = robot.right_arm.get_joint_pos()
                obs_vel = robot.right_arm.get_joint_vel()
                robot_ts = int(robot.right_arm.get_timestamp_ns())
            except Exception:
                obs_pos = np.full(7, np.nan)
                obs_vel = np.full(7, np.nan)
                robot_ts = -1

            rows.append({
                "t_rel_s": t_rel,
                "t_send_ns": t_send,
                "robot_ts_ns": robot_ts,
                "cmd_j_rad": cmd_j,
                "obs_j_rad": float(obs_pos[j]),
                "obs_vel_j_rad_s": float(obs_vel[j]),
            })
            limiter.sleep()

        logger.info("Returning right arm to INIT ...")
        robot.right_arm.set_joint_pos(
            init, wait_time=2.0, exit_on_reach=True,
            exit_on_reach_kwargs={"tolerance": 0.02},
        )

        # Write CSV.
        if not rows:
            raise SystemExit("No samples logged")
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        logger.info(f"Wrote {len(rows)} samples to {out_path}")

        # Stats — exclude the ramp regions so phase-lag isn't biased by the
        # smooth-start envelope.
        cmd_arr = np.array([r["cmd_j_rad"] for r in rows])
        obs_arr = np.array([r["obs_j_rad"] for r in rows])
        t_arr = np.array([r["t_rel_s"] for r in rows])
        mid = (t_arr >= args.ramp_s) & (t_arr <= args.duration_s - args.ramp_s)
        c_mid = cmd_arr[mid]
        o_mid = obs_arr[mid]

        if c_mid.size < 4:
            logger.warning("Too few samples after ramp exclusion; skipping stats.")
            return

        err = o_mid - c_mid
        rms = float(np.sqrt(np.mean(err ** 2)))
        max_err = float(np.max(np.abs(err)))
        dt_med = float(np.median(np.diff(t_arr[mid])))
        lag_ms = _phase_lag_ms(c_mid, o_mid, dt_med)
        cmd_amp = (c_mid.max() - c_mid.min()) / 2.0
        obs_amp = (o_mid.max() - o_mid.min()) / 2.0
        atten_db = 20.0 * math.log10(obs_amp / cmd_amp) if cmd_amp > 0 else float("nan")

        logger.info(
            "Tracking (steady-state window):\n"
            f"  rms error   = {rms * 1000:7.2f} mrad\n"
            f"  max error   = {max_err * 1000:7.2f} mrad\n"
            f"  cmd amp     = {cmd_amp * 1000:7.2f} mrad\n"
            f"  obs amp     = {obs_amp * 1000:7.2f} mrad  ({atten_db:+.2f} dB)\n"
            f"  est. lag    = {lag_ms:7.2f} ms (obs trails cmd, "
            f"xcorr peak @ dt_med={dt_med * 1e3:.2f} ms)"
        )

    finally:
        try:
            if not robot.is_shutdown():
                robot.shutdown()
        except Exception as exc:
            logger.warning(f"Shutdown error: {exc}")


if __name__ == "__main__":
    main()
