#!/usr/bin/env python3
"""Probe base-yaw drift vs turn-follow for the Vega whole-body IK.

Reproduces kinematically the live ``wbc_vr_leader`` symptom -- operator standing
still, only a handheld controller moving, chassis keeps rotating -- and sweeps the
base yaw damping (``base_orientation_cost[2]``) measuring, per scenario:

  * chassis stillness for a stationary operator (no yaw drift), and
  * how much turn-follow remains for intentional operator turns, and
  * EE tracking accuracy (the priority) throughout.

Sweep conclusion (now baked into wbik.yaml): steady-state EE tracking is <= 2 mm /
~0 deg at EVERY yaw cost -- the 7-DOF arms absorb even 90 deg turns -- so yaw is
damped at 50000 like x/y, where the chassis is dead still for a stationary
operator. Only fast >= 180 deg turns resolve slowly (base creeps ~3.6 deg/s on a
persistent residual). Intermediate costs (500..10000) are the worst of both:
visible wander (3..16 deg) plus a parasitic post-walk skew (~11 deg at 10000).

Mechanism under test: the base FrameTask is re-targeted to the *current* pose every
solve (pure velocity damping, no restoring force), so a *persistent* EE residual --
e.g. a controller orientation the wrist cannot fully reach, or posture/collision
terms pushing the arms -- drives a steady yaw rate forever. Pink squares task costs
in the QP Hessian, so yaw 50 vs EE 10000 resists with weight (50/10000)^2 = 4e-5:
essentially free. With x/y damped at 50000 (the arm-priority tuning), yaw is the
only cheap base DOF left and collects every residual.

Run in the dexmate conda env::

    /home/yixuan/miniforge3/envs/dexmate/bin/python scripts/diagnostics/probe_wbc_yaw_drift.py \
        --yaw-costs 50 1000 5000 50000

Scenarios (all at the follower's 100 Hz solve rate, head IK in the loop like
``wbc_vr_record``):

  wiggle_pos   right hand wanders +/-6..8 cm (Lissajous), orientation fixed;
               left hand and operator stationary.        -> chassis must stay still
  wiggle_full  same wander plus +/-20 deg wrist wobble.  -> chassis must stay still
  ori_hold     both controllers held yawed +25 deg at the nominal positions
               (persistent orientation residual).        -> chassis must stay still
  turn45       operator turns 45 deg in place (targets rotate about the base
               vertical, positions swing along).         -> base SHOULD follow
  turn90       operator turns 90 deg in place -- beyond what the arms can absorb,
               so EE tracking REQUIRES the base to come around.
  walk50       operator walks 0.5 m forward (beyond arm reach).
                                                         -> base SHOULD translate
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pinocchio as pin

from omniteleop.follower.whole_body_ik import (
    HEAD_FRAME,
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    VegaWholeBodyIK,
    WBCConfig,
)

DT = 0.01  # the follower's 100 Hz IK tick

SCENARIOS = (
    "wiggle_pos", "wiggle_full", "ori_hold", "turn45", "turn90", "turn180", "walk50"
)
# Stationary-operator scenarios: the chassis must not rotate.
STATIONARY = ("wiggle_pos", "wiggle_full", "ori_hold")
DURATIONS = {
    "wiggle_pos": 20.0,
    "wiggle_full": 20.0,
    "ori_hold": 20.0,
    "turn45": 15.0,
    "turn90": 30.0,
    "turn180": 30.0,
    "walk50": 10.0,
}


def _rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _roty(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _ori_err_deg(r_target: np.ndarray, r_actual: np.ndarray) -> float:
    """Geodesic angle (deg) between two rotation matrices."""
    cosang = (np.trace(r_target.T @ r_actual) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))


def _targets(
    name: str, t: float, left0: pin.SE3, right0: pin.SE3
) -> tuple[pin.SE3, pin.SE3]:
    """EE targets at time ``t`` for a scenario, from the nominal EE poses."""
    if name == "wiggle_pos" or name == "wiggle_full":
        # Right hand wanders (Lissajous, +/-6..8 cm); left hand stays.
        dp = np.array([
            0.06 * np.sin(2 * np.pi * 0.35 * t),
            0.08 * np.sin(2 * np.pi * 0.25 * t + 1.0),
            0.05 * np.sin(2 * np.pi * 0.45 * t),
        ])
        rot = right0.rotation
        if name == "wiggle_full":  # +/-20 deg wrist wobble on top (world frame)
            rot = (
                _rotz(0.35 * np.sin(2 * np.pi * 0.30 * t))
                @ _roty(0.25 * np.sin(2 * np.pi * 0.20 * t + 0.7))
                @ rot
            )
        return left0, pin.SE3(rot, right0.translation + dp)
    if name == "ori_hold":
        # Controllers held turned +25 deg about world z, positions nominal: a
        # persistent orientation residual if the wrists cannot fully reach it.
        rot = _rotz(np.radians(25.0))
        return (
            pin.SE3(rot @ left0.rotation, left0.translation.copy()),
            pin.SE3(rot @ right0.rotation, right0.translation.copy()),
        )
    if name in ("turn45", "turn90", "turn180"):
        # Operator turns in place: poses rotate about the base origin. 45/90 deg
        # turn out to be arm-absorbable (the 7-DOF arms cover them with the base
        # barely moving); 180 deg is not, so the base MUST come around to track.
        rot = _rotz(np.radians(float(name.removeprefix("turn"))))
        return (
            pin.SE3(rot @ left0.rotation, rot @ left0.translation),
            pin.SE3(rot @ right0.rotation, rot @ right0.translation),
        )
    if name == "walk50":
        dp = np.array([0.5, 0.0, 0.0])
        return (
            pin.SE3(left0.rotation, left0.translation + dp),
            pin.SE3(right0.rotation, right0.translation + dp),
        )
    raise ValueError(f"unknown scenario {name!r}")


def run_scenario(name: str, yaw_cost: float, duration: float) -> dict:
    cfg = WBCConfig()
    cfg.base_orientation_cost = (
        cfg.base_orientation_cost[0],
        cfg.base_orientation_cost[1],
        float(yaw_cost),
    )
    ik = VegaWholeBodyIK(cfg)
    ik.reset()
    left0 = ik.frame_pose(LEFT_EE_FRAME).copy()
    right0 = ik.frame_pose(RIGHT_EE_FRAME).copy()
    head_target = ik.frame_pose(HEAD_FRAME).copy()  # operator gaze fixed in world

    steps = int(round(duration / DT))
    yaw = np.empty(steps)
    xy = np.empty((steps, 2))
    pos_err = np.empty(steps)   # max of L/R position error (m)
    ori_err = np.empty(steps)   # max of L/R orientation error (deg)
    n_fail = 0
    n_held = 0
    t_solve = 0.0

    for k in range(steps):
        t = k * DT
        lt, rt = _targets(name, t, left0, right0)
        tic = time.perf_counter()
        head_cmd = ik.solve_head(head_target, DT)
        res = ik.solve(lt, rt, DT, head_joints=head_cmd)
        t_solve += time.perf_counter() - tic
        n_fail += 0 if res.success else 1
        n_held += 1 if res.held else 0
        bx, by, byaw = res.base_pose
        yaw[k] = byaw
        xy[k] = (bx, by)
        pos_err[k] = max(res.left_ee_error, res.right_ee_error)
        ori_err[k] = max(
            _ori_err_deg(lt.rotation, ik.frame_pose(LEFT_EE_FRAME).rotation),
            _ori_err_deg(rt.rotation, ik.frame_pose(RIGHT_EE_FRAME).rotation),
        )

    yaw = np.degrees(np.unwrap(yaw))
    # Drift rate: linear fit of yaw over the last 40% of the run. A converged solve
    # has ~0 slope; "chassis keeps rotating" shows up as a sustained slope.
    tail = slice(int(steps * 0.6), steps)
    tt = np.arange(steps)[tail] * DT
    drift = float(np.polyfit(tt, yaw[tail], 1)[0]) if steps > 10 else float("nan")
    settle = slice(int(steps * 0.6), steps)  # steady-state tracking accuracy
    return {
        "scenario": name,
        "yaw_cost": yaw_cost,
        "yaw_final_deg": float(yaw[-1]),
        "yaw_max_deg": float(np.max(np.abs(yaw))),
        "yaw_drift_deg_s": drift,
        "xy_final_m": float(np.linalg.norm(xy[-1])),
        "pos_err_rms_mm": float(np.sqrt(np.mean(pos_err[settle] ** 2)) * 1e3),
        "pos_err_max_mm": float(np.max(pos_err) * 1e3),
        "ori_err_rms_deg": float(np.sqrt(np.mean(ori_err[settle] ** 2))),
        "ori_err_max_deg": float(np.max(ori_err)),
        "n_fail": n_fail,
        "n_held": n_held,
        "solve_ms": t_solve / steps * 1e3,
    }


HEADER = (
    f"{'scenario':<12} {'yaw_cost':>8} {'yaw_end':>8} {'yaw_max':>8} {'drift/s':>8} "
    f"{'|xy|':>6} {'pRMS':>7} {'pMAX':>7} {'oRMS':>6} {'oMAX':>6} "
    f"{'fail':>4} {'held':>4} {'ms':>5}"
)


def _fmt(r: dict) -> str:
    return (
        f"{r['scenario']:<12} {r['yaw_cost']:>8.0f} {r['yaw_final_deg']:>7.2f}° "
        f"{r['yaw_max_deg']:>7.2f}° {r['yaw_drift_deg_s']:>7.3f}° "
        f"{r['xy_final_m']:>5.3f}m {r['pos_err_rms_mm']:>6.1f}mm "
        f"{r['pos_err_max_mm']:>6.1f}mm {r['ori_err_rms_deg']:>5.2f}° "
        f"{r['ori_err_max_deg']:>5.2f}° {r['n_fail']:>4d} {r['n_held']:>4d} "
        f"{r['solve_ms']:>5.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--yaw-costs", type=float, nargs="+", default=[50.0],
                        help="base_orientation_cost[2] values to sweep (default: 50, "
                             "the current config -- reproduces the drift).")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                        choices=SCENARIOS, help="subset of scenarios to run.")
    parser.add_argument("--duration-scale", type=float, default=1.0,
                        help="scale every scenario duration (default 1.0).")
    args = parser.parse_args()
    if args.duration_scale <= 0:
        raise ValueError(f"--duration-scale must be > 0, got {args.duration_scale}")

    print(HEADER)
    print("-" * len(HEADER))
    for yaw_cost in args.yaw_costs:
        for name in args.scenarios:
            r = run_scenario(name, yaw_cost, DURATIONS[name] * args.duration_scale)
            print(_fmt(r), flush=True)
        print("-" * len(HEADER))
    print(
        "stationary scenarios (wiggle_*, ori_hold): want drift/s ~ 0 and small yaw;\n"
        "turn45/90: want pRMS/oRMS ~ 0 (arms may absorb the turn; base motion is\n"
        "optional); walk50: want tracking with bounded arm stretch; always: small\n"
        "pRMS/oRMS (tracking accuracy is the priority)."
    )


if __name__ == "__main__":
    main()
