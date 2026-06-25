#!/usr/bin/env python3
"""Pure, hardware-free helpers for the real-robot WBC follower (wbc_vr_robot.py).

Kept out of the (heavy, dexcontrol-importing) follower script so the safety-critical
bits are unit-testable on their own: the per-DOF enable mask, the per-tick joint-step
clamp, and -- the one Codex flagged as must-not-get-wrong -- the planar-root base
pose encoding used to feed a MEASURED base pose into ``VegaWholeBodyIK.solve(current_q=)``.

numpy only.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# DOF groups the follower can independently enable/actuate. Grippers are handled
# separately (a distinct --grippers interlock), not part of this motion mask.
ENABLE_GROUPS = ("arms", "torso", "head", "base")


def parse_enable_mask(spec: str) -> dict[str, bool]:
    """Parse an ``--enable`` spec like ``"arms,torso,head"`` into a per-group dict.

    ``"all"`` -> every group on; ``""``/``"none"`` -> every group off. Unknown groups
    raise (fail loud rather than silently leaving a DOF unexpectedly frozen/moving).
    """
    text = (spec or "").strip().lower()
    if text in ("", "none"):
        return {g: False for g in ENABLE_GROUPS}
    if text == "all":
        return {g: True for g in ENABLE_GROUPS}
    requested = [s.strip() for s in text.split(",") if s.strip()]
    unknown = [r for r in requested if r not in ENABLE_GROUPS]
    if unknown:
        raise ValueError(
            f"unknown --enable group(s) {unknown}; valid: {list(ENABLE_GROUPS)} "
            "(or 'all' / 'none')"
        )
    return {g: (g in requested) for g in ENABLE_GROUPS}


def clamp_joint_step(
    prev: np.ndarray, target: np.ndarray, max_step: float
) -> tuple[np.ndarray, float]:
    """Clamp the per-joint change ``|target - prev|`` to ``max_step`` radians.

    Returns ``(clamped_target, max_abs_requested_step)``. ``max_step <= 0`` (or
    non-finite) disables clamping and returns ``target`` unchanged. The follower uses
    the returned ``max_abs_requested_step`` to detect a large IK jump (and abort if it
    keeps exceeding the clamp), while the clamped target keeps a single bad tick from
    snapping the arm.
    """
    prev = np.asarray(prev, dtype=float)
    target = np.asarray(target, dtype=float)
    if prev.shape != target.shape:
        raise ValueError(f"prev/target shape mismatch: {prev.shape} vs {target.shape}")
    delta = target - prev
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    if not np.isfinite(max_step) or max_step <= 0.0:
        return target.copy(), max_abs
    return prev + np.clip(delta, -max_step, max_step), max_abs


def compute_hold_reason(
    estop: bool,
    last_cmd_wall: Optional[float],
    now: float,
    source_timeout: float,
    odom_age: Optional[float] = None,
) -> Optional[str]:
    """Decide whether the real-robot follower should HOLD this tick (and why).

    Returns a short reason string (hold) or None (run). This is the watchdog layered on
    top of the e-stop / failed-solve / safety holds the caller already handles:

    * ``"awaiting-command"`` -- engaged (not e-stopped) but no valid EE command has
      arrived yet: freeze rather than actuate toward the nominal seed targets.
    * ``"stale-source"`` -- no fresh leader command for longer than ``source_timeout``.
    * ``"stale-odom"`` -- the latest odometry sample is older than ``source_timeout``
      (only checked when ``odom_age`` is provided, i.e. the base is enabled).
    """
    if estop:
        return None
    if last_cmd_wall is None:
        return "awaiting-command"
    if (now - last_cmd_wall) > source_timeout:
        return "stale-source"
    if odom_age is not None and odom_age > source_timeout:
        return "stale-odom"
    return None


def set_base_in_q(q: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    """Write a planar base pose ``(x, y, yaw)`` into a pinocchio config's root joint.

    The Vega whole-body model uses a planar root joint: ``q[0]=x``, ``q[1]=y``, and the
    rotation is stored as ``(cos(yaw), sin(yaw))`` at ``q[2:4]`` (unit-norm -- the solver
    validates it). This is the encoding required to feed a MEASURED base pose into
    ``VegaWholeBodyIK.solve(current_q=...)``; getting it wrong (e.g. storing yaw at
    ``q[2]``) silently corrupts the closed-loop seed. Returns a copy; ``q`` is not mutated.
    """
    q = np.asarray(q, dtype=float).copy()
    if q.shape[0] < 4:
        raise ValueError(f"q must have >= 4 entries for the planar root, got {q.shape}")
    base_pose = np.asarray(base_pose, dtype=float)
    if base_pose.shape != (3,):
        raise ValueError(f"base_pose must be (x, y, yaw) shape (3,), got {base_pose.shape}")
    x, y, yaw = float(base_pose[0]), float(base_pose[1]), float(base_pose[2])
    q[0], q[1] = x, y
    q[2], q[3] = np.cos(yaw), np.sin(yaw)
    return q
