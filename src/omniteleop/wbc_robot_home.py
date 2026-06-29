#!/usr/bin/env python3
# ruff: noqa: SLF001
"""Homing helpers for the real-robot WBC follower.

These functions are intentionally driver-bound rather than hardware-free: they call the
``HardwareDriver`` component accessors and stop path, but keep the long homing/guard logic
out of ``scripts/wbc_vr_robot.py``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def _interp_joint(
    driver,
    grp: str,
    waypoints: list,
    step: float = 0.01,
    guard: Optional[Callable[[np.ndarray], None]] = None,
) -> None:
    comp = driver._comp(grp)
    cur = np.asarray(comp.get_joint_pos(), dtype=float)
    for wp in waypoints:
        wp = np.asarray(wp, dtype=float)
        n = max(1, int(np.max(np.abs(wp - cur)) / step))
        for i in range(n):
            q = cur + (wp - cur) * (i + 1) / n
            # Vet the candidate posture for self-collision BEFORE commanding it; the
            # guard raises SystemExit (after stopping motion) on a worsening violation.
            if guard is not None:
                guard(q)
            comp.set_joint_pos(q.tolist(), wait_time=0.1, exit_on_reach=True)
        cur = wp


def _make_home_guard(driver, grp: str) -> Optional[Callable[[np.ndarray], None]]:
    """Build a per-substep self-collision guard for homing ``grp`` to nominal.

    Returns ``None`` when self-collision avoidance is unavailable (sphere model not
    loaded), so homing keeps its prior behavior. Otherwise returns a callable
    ``guard(cmd)`` that, for each homing waypoint ``cmd`` of ``grp``, assembles the
    full-body config -- ``grp`` at ``cmd`` and every OTHER group at its MEASURED
    position (snapshotted once here, since the sequential homing leaves them static
    during this group's sweep) -- and checks its closest cross-group self-collision
    distance against the WBC ``self_collision_safe_dist``.

    Homing is conservative: any candidate below the 2 cm safe distance is vetoed before
    it is commanded. A veto halts motion and raises ``SystemExit``.
    """
    if not driver.ik.collision_enabled:
        return None

    # Snapshot the OTHER groups' measured joints once; only `grp` moves this sweep.
    meas = driver._read_measured_joints()
    base_q = driver.ik.nominal_q()
    for g, names in driver._joint_names.items():
        if g == grp:
            continue
        vals = meas.get(g)
        if vals is None:
            continue  # bad readback -> leave that group at nominal (best effort)
        for name, val in zip(names, vals, strict=True):
            base_q[driver.ik._idx_q[name]] = float(val)
    idx = [driver.ik._idx_q[n] for n in driver._joint_names[grp]]

    def candidate_q(cmd: np.ndarray) -> np.ndarray:
        q = base_q.copy()
        for k, val in zip(idx, np.asarray(cmd, dtype=float), strict=True):
            q[k] = float(val)
        return q

    def guard(cmd: np.ndarray) -> None:
        dist = driver.ik.self_collision_distance(candidate_q(cmd))
        if dist < driver.cfg.self_collision_safe_dist:
            driver.stop_all_motion()
            raise SystemExit(
                f"[wbc_vr_robot] homing ABORTED: {grp} self-collision "
                f"{dist * 100:.1f}cm < safe distance "
                f"{driver.cfg.self_collision_safe_dist * 100:.0f}cm -- "
                f"the direct path to WBC nominal would self-collide. "
                f"Reposition the arms clear and retry."
            )

    return guard


def home_to_nominal(driver, *, arm_home_step: float) -> None:
    """Home the enabled real-robot groups to the WBC nominal posture."""
    # Only home the ENABLED groups (a disabled group is not actuated, so moving it
    # would be surprising); the IK still models disabled groups at nominal, so they
    # are ASSUMED already there -- a limited-test caveat for partial --enable.
    on = {
        "torso": driver.enable["torso"],
        "left_arm": driver.enable["arms"],
        "right_arm": driver.enable["arms"],
        "head": driver.enable["head"],
    }
    nom = {grp: driver._to_hw(grp, driver._nominal[grp]) for grp in driver._joint_names}
    homing = [g for g in driver._joint_names if on[g]]
    skipped = [g for g in driver._joint_names if not on[g]]
    guarded = driver.ik.collision_enabled
    print(
        f"[wbc_vr_robot] homing to WBC nominal (move clear; watch the robot). "
        f"homing={homing}"
        + (f"; assuming-at-nominal={skipped}" if skipped else "")
        + (
            "; self-collision guard ON"
            if guarded
            else "; self-collision guard OFF (no sphere model)"
        )
    )
    # Each enabled group is driven straight to nominal (no SAFE_* waypoint), so the
    # direct path is traversed slowly -- the slow glide lets the operator watch, and
    # the per-substep guard (when a collision-sphere model is loaded) stops + aborts
    # before a waypoint would drive the body into a worsening self-collision.
    if on["torso"]:
        _interp_joint(
            driver,
            "torso",
            [nom["torso"]],
            step=arm_home_step,
            guard=_make_home_guard(driver, "torso"),
        )
    if on["left_arm"]:
        _interp_joint(
            driver,
            "left_arm",
            [nom["left_arm"]],
            step=arm_home_step,
            guard=_make_home_guard(driver, "left_arm"),
        )
    if on["right_arm"]:
        _interp_joint(
            driver,
            "right_arm",
            [nom["right_arm"]],
            step=arm_home_step,
            guard=_make_home_guard(driver, "right_arm"),
        )
    if on["head"]:
        _interp_joint(driver, "head", [nom["head"]], guard=_make_home_guard(driver, "head"))
    # Settle each enabled group at nominal: the stepped ramp can outrun the arm (each
    # dexcontrol substep returns after its wait_time whether or not the joint caught
    # up, so lag accumulates), so block here until the MEASURED joints are within
    # --home-tol (dexcontrol's reached-check uses the same max-abs metric as the gate
    # below) or --home-settle elapses -- draining the lag before the gate snapshot.
    if driver.args.home_settle > 0:
        for grp in driver._joint_names:
            if on[grp]:
                driver._comp(grp).set_joint_pos(
                    nom[grp].tolist(),
                    wait_time=driver.args.home_settle,
                    exit_on_reach=True,
                    exit_on_reach_kwargs={"tolerance": driver.args.home_tol},
                )
    bad = []
    for grp in driver._joint_names:
        if not on[grp]:
            continue
        meas = np.asarray(driver._comp(grp).get_joint_pos(), dtype=float)
        err = float(np.max(np.abs(meas - nom[grp])))
        if err > driver.args.home_tol:
            bad.append(f"{grp} {err:.3f}rad")
    if bad:
        raise SystemExit(
            "[wbc_vr_robot] homing gate FAILED (measured vs WBC nominal): "
            + ", ".join(bad)
            + f" > --home-tol {driver.args.home_tol}. Aborting before engage."
        )
    print("[wbc_vr_robot] homing gate OK: enabled groups are at WBC nominal.")
    # Center the swerve wheels to straight before engage (only if the base is
    # actuated). _compute_wheel_control maps zero velocity to steering 0, so the
    # run-loop's pre-engage holds already command steering->0 each tick, but with no
    # settle window; do it explicitly here so the steer joints are physically straight
    # (and given time to converge) before any base twist, instead of swinging from a
    # stale angle on the first engaged set_velocity. set_steering_angle commands the
    # steer joints to 0 with zero wheel velocity; wait_time holds the command so they
    # converge (--home-settle, repeated for ~max(wait_time-1, 0)s; 0 = single shot).
    if driver.enable["base"] and driver.has_chassis:
        print("[wbc_vr_robot] centering wheels to straight (steering -> 0) ...")
        driver.robot.chassis.set_steering_angle(0.0, wait_time=driver.args.home_settle)
