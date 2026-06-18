"""Head-to-head: pink CBF barrier vs the reference's mink velocity-damper rule.

Both collision-avoidance rules run inside the SAME VegaWholeBodyIK solver (same
URDF, same 90-sphere collision model, same kept cross-group pairs, same tasks,
limits, QP solver, and reactive SafetyGate). Only the proactive in-QP rule is
swapped:

  cbf   : pink SelfCollisionBarrier (Vega current config)
          per-step allowed closure  = gain * (d - 0.02) * dt   [gain 1.0 /s]
  mink  : the reference deps/rby1-wbc rule (mink CollisionAvoidanceLimit),
          ported verbatim to the pink QP (variable = dq displacement):
          per-step allowed closure  = 0.85 * (d - 0.01)  for pairs with d < 0.02
          (no closure at all once d <= 0.01)
  none  : no proactive rule -- reactive SafetyGate only (ablation)

Scenarios (kinematic rollouts, default 100 Hz like the deployed IK loop):
  clap  : both EE targets at the midpoint between the hands -> head-on approach
  cross : left/right EE targets swapped -> arms must pass each other
  torso : right EE commanded into the chest
  walk  : both targets 0.5 m forward -> no conflict; tracking must be unaffected

Metrics: min cross-group sphere distance over the run (floor breaches), final
standoff, final EE errors, gate holds, QP failures, late-phase distance chatter,
mean solve time.

Run:
    /home/yixuan/miniforge3/envs/dexmate/bin/python scripts/diagnostics/probe_collision_rule_compare.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from omniteleop.follower.whole_body_ik import (
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    VegaWholeBodyIK,
)


class MinkStyleDamper:
    """The reference's mink CollisionAvoidanceLimit rule as a pink barrier.

    mink (deps/rby1-wbc values: gain=0.85/step, min=0.01 m, detect=0.02 m)
    constrains, for every collision pair currently closer than ``detect``::

        d_dot >= -gain * (d - d_min) / dt      (no constraint when d >= detect)
        d_dot >= 0                              (when d <= d_min)

    In pink's QP the variable is the displacement dq, so the same rule reads
    ``-J_d dq <= gain * (d - d_min)`` with ``J_d = dd/dq`` (the contact-normal
    Jacobian, built exactly like pink's SelfCollisionBarrier rows).
    """

    def __init__(self, gain: float = 0.85, d_min: float = 0.01, detect: float = 0.02):
        if not (0.0 < gain <= 1.0):
            raise ValueError(f"gain must be in (0, 1], got {gain}")
        if d_min < 0.0 or detect <= d_min:
            raise ValueError(f"need 0 <= d_min < detect, got {d_min}, {detect}")
        self.gain = gain
        self.d_min = d_min
        self.detect = detect

    def compute_qp_objective(self, configuration):
        nv = configuration.model.nv
        return np.zeros((nv, nv)), np.zeros(nv)

    def compute_qp_inequalities(self, configuration, dt: float):
        model = configuration.model
        data = configuration.data
        cm = configuration.collision_model
        cd = configuration.collision_data
        nv = model.nv

        g_rows: list[np.ndarray] = []
        h_rows: list[float] = []
        for k in range(len(cm.collisionPairs)):
            dr = cd.distanceResults[k]
            d = float(dr.min_distance)
            if d >= self.detect:
                continue
            w1 = np.asarray(dr.getNearestPoint1(), dtype=float)
            w2 = np.asarray(dr.getNearestPoint2(), dtype=float)
            if np.allclose(w1, w2):
                continue
            n = (w1 - w2) / np.linalg.norm(w1 - w2)
            if d < 0.0:  # witness points swap sides under penetration
                n = -n
            cp = cm.collisionPairs[k]
            j1 = cm.geometryObjects[cp.first].parentJoint
            j2 = cm.geometryObjects[cp.second].parentJoint
            r1 = w1 - data.oMi[j1].translation
            r2 = w2 - data.oMi[j2].translation
            J1 = pin.getJointJacobian(model, data, j1, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J2 = pin.getJointJacobian(model, data, j2, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            # dd/dq: separation speed of the witness points along the normal.
            jrow = n @ J1[:3] + (np.cross(r1, n)) @ J1[3:]
            jrow -= n @ J2[:3] + (np.cross(r2, n)) @ J2[3:]
            g_rows.append(-jrow)
            h_rows.append(self.gain * (d - self.d_min) if d > self.d_min else 0.0)

        if not g_rows:
            # pink vstacks barrier outputs unconditionally; emit a vacuous row.
            return np.zeros((1, nv)), np.array([1.0])
        return np.vstack(g_rows), np.asarray(h_rows, dtype=float)


@dataclass
class RunStats:
    min_dist: float        # executed (post-gate) closest distance over the run
    cand_min: float        # closest distance any QP *candidate* step reached
    cand_floor: int        # candidate steps below the 1 cm reactive floor
    cand_pen: int          # candidate steps in actual penetration (d < 0)
    final_dist: float
    final_err_l: float
    final_err_r: float
    holds: int
    qp_fails: int
    solve_ms: float
    base_dist: float


def make_solver(variant: str):
    """Build a solver for the variant; returns (ik, candidate-distance log)."""
    if variant == "cbf-all":
        from omniteleop.follower.whole_body_ik import WBCConfig

        ik = VegaWholeBodyIK(WBCConfig(n_collision_pairs=10**6))
    else:
        ik = VegaWholeBodyIK()
    if not ik.collision_enabled:
        raise RuntimeError("collision sphere model failed to load; probe is meaningless")
    if variant in ("cbf", "cbf-all"):
        pass  # SelfCollisionBarrier (wbik.yaml config)
    elif variant == "mink":
        ik.barriers = [MinkStyleDamper()]  # reference horizon: detect 2 cm
    elif variant == "minkW":
        # The mink rule with the 10 cm horizon the (removed) Vega mink backend
        # used (wbc_safety.DEFAULT_COLLISION_DETECT_DIST), sized for low-rate steps.
        ik.barriers = [MinkStyleDamper(detect=0.10)]
    elif variant == "none":
        ik.barriers = None
    else:
        raise ValueError(f"unknown variant {variant!r}")
    ik.reset()

    # Log the *candidate* self-distance each step (what the QP proposed, before
    # the reactive gate may revert it) by wrapping the gate check.
    cand_log: list[float] = []
    orig_check = ik._gate.check

    def check_logged(margin, self_dist):
        cand_log.append(self_dist)
        return orig_check(margin, self_dist)

    ik._gate.check = check_logged
    return ik, cand_log


def targets_for(ik: VegaWholeBodyIK, scenario: str):
    left = ik.frame_pose(LEFT_EE_FRAME)
    right = ik.frame_pose(RIGHT_EE_FRAME)
    if scenario == "clap":
        mid = 0.5 * (left.translation + right.translation)
        return pin.SE3(left.rotation, mid), pin.SE3(right.rotation, mid)
    if scenario == "cross":
        return (
            pin.SE3(left.rotation, right.translation.copy()),
            pin.SE3(right.rotation, left.translation.copy()),
        )
    if scenario == "torso":
        chest = ik.frame_pose("torso_l3").translation.copy()
        return left, pin.SE3(right.rotation, chest)
    if scenario == "walk":
        d = np.array([0.5, 0.0, 0.0])
        return (
            pin.SE3(left.rotation, left.translation + d),
            pin.SE3(right.rotation, right.translation + d),
        )
    raise ValueError(f"unknown scenario {scenario!r}")


def rollout(variant: str, scenario: str, dt: float, n_steps: int) -> RunStats:
    ik, cand_log = make_solver(variant)
    left_t, right_t = targets_for(ik, scenario)
    dists, holds, fails, times = [], 0, 0, []
    res = None
    for _ in range(n_steps):
        t0 = time.perf_counter()
        res = ik.solve(left_t, right_t, dt)
        times.append(time.perf_counter() - t0)
        dists.append(res.min_self_distance)
        holds += int(res.held)
        fails += int(not res.success)
    dists = np.asarray(dists)
    cand = np.asarray(cand_log)
    return RunStats(
        min_dist=float(dists.min()),
        cand_min=float(cand.min()),
        cand_floor=int((cand < 0.01).sum()),
        cand_pen=int((cand < 0.0).sum()),
        final_dist=float(dists[-1]),
        final_err_l=res.left_ee_error,
        final_err_r=res.right_ee_error,
        holds=holds,
        qp_fails=fails,
        solve_ms=float(np.mean(times) * 1e3),
        base_dist=float(np.hypot(res.base_pose[0], res.base_pose[1])),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hz", type=float, default=100.0, help="control rate (Hz)")
    parser.add_argument("--seconds", type=float, default=10.0, help="rollout length")
    parser.add_argument(
        "--scenarios", nargs="+", default=["clap", "cross", "torso", "walk"]
    )
    parser.add_argument(
        "--variants", nargs="+", default=["cbf", "cbf-all", "mink", "minkW", "none"]
    )
    parser.add_argument(
        "--dt-sweep", action="store_true",
        help="additionally run 'clap' at 100/30/10 Hz for each variant",
    )
    args = parser.parse_args()

    dt = 1.0 / args.hz
    n_steps = round(args.seconds * args.hz)

    hdr = (
        f"{'scenario':<8} {'variant':<7} {'exec_d(cm)':>10} {'cand_d(cm)':>10} "
        f"{'<floor':>6} {'pen':>4} {'errL(cm)':>8} {'errR(cm)':>8} "
        f"{'holds':>5} {'fails':>5} {'ms':>5} {'base(m)':>7}"
    )

    def show(tag: str, variant: str, s: RunStats) -> None:
        print(
            f"{tag:<8} {variant:<7} {s.min_dist * 100:>10.2f} "
            f"{s.cand_min * 100:>10.2f} {s.cand_floor:>6d} {s.cand_pen:>4d} "
            f"{s.final_err_l * 100:>8.2f} {s.final_err_r * 100:>8.2f} "
            f"{s.holds:>5d} {s.qp_fails:>5d} {s.solve_ms:>5.1f} {s.base_dist:>7.2f}"
        )

    print(f"control rate {args.hz:g} Hz, {n_steps} steps ({args.seconds:g} s)")
    print(hdr)
    print("-" * len(hdr))
    for scenario in args.scenarios:
        for variant in args.variants:
            show(scenario, variant, rollout(variant, scenario, dt, n_steps))
        print()

    if args.dt_sweep:
        print("dt sweep on 'clap' (same wall-clock duration):")
        print(hdr)
        print("-" * len(hdr))
        for hz in (100.0, 30.0, 10.0):
            for variant in args.variants:
                s = rollout(variant, "clap", 1.0 / hz, round(args.seconds * hz))
                show(f"{hz:>5.0f}Hz", variant, s)
            print()


if __name__ == "__main__":
    main()
