"""Pure helpers for closed-loop planar base pose tracking.

This module intentionally has no robot, Zenoh, or recorder dependencies. Callers own
their feedback source and command transport; these helpers provide the shared SE(2)
reference math, pose-PD command, command shaping, and freshness checks.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from omniteleop.wbc_stream import deadband_planar_twist_by_reference

BASE_DOF_MODES = ("xy_yaw", "xy")


@dataclasses.dataclass(frozen=True)
class BaseClosedLoopConfig:
    """Tuning and safety knobs for planar base pose-PD control."""

    source: str = "none"
    rate: float = 100.0
    kp_xy: float = 1.0
    kp_yaw: float = 1.5
    pos_tolerance: float = 0.03
    yaw_tolerance: float = 0.05
    stable_time: float = 0.25
    source_timeout_s: float = 0.5
    timeout_scale: float = 3.0
    timeout_margin_s: float = 1.0
    max_lin_speed: Optional[float] = None
    max_ang_speed: Optional[float] = None
    max_lin_accel: float = 0.4
    max_ang_accel: float = 0.8
    deadband_lin: float = 0.01
    deadband_ang: float = 0.02

    @property
    def enabled(self) -> bool:
        """Whether closed-loop feedback is configured."""
        return self.source != "none"


@dataclasses.dataclass
class FreshSequence:
    """Track freshness for sequence-numbered pose streams."""

    last_seq: Optional[int] = None
    last_update_t: float = float("nan")


def validate_config(
    config: BaseClosedLoopConfig,
    default_lin_speed: float,
    default_ang_speed: float,
) -> None:
    """Validate closed-loop config and its fallback speed limits."""
    if config.source not in ("none", "odom"):
        raise ValueError(
            "--closed-loop-source must be one of 'none' or 'odom', "
            f"got {config.source!r}"
        )
    positive = {
        "closed_loop_rate": config.rate,
        "closed_loop_pos_tolerance": config.pos_tolerance,
        "closed_loop_yaw_tolerance": config.yaw_tolerance,
        "closed_loop_stable_time": config.stable_time,
        "closed_loop_source_timeout_s": config.source_timeout_s,
        "closed_loop_timeout_scale": config.timeout_scale,
        "closed_loop_timeout_margin_s": config.timeout_margin_s,
    }
    for name, value in positive.items():
        if value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0, got {value}")
    optional_positive = {
        "closed_loop_max_lin_speed": config.max_lin_speed,
        "closed_loop_max_ang_speed": config.max_ang_speed,
    }
    for name, value in optional_positive.items():
        if value is not None and value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0 when set, got {value}")
    nonnegative = {
        "closed_loop_max_lin_accel": config.max_lin_accel,
        "closed_loop_max_ang_accel": config.max_ang_accel,
        "closed_loop_deadband_lin": config.deadband_lin,
        "closed_loop_deadband_ang": config.deadband_ang,
    }
    for name, value in nonnegative.items():
        if value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0, got {value}")
    closed_loop_limits(config, default_lin_speed, default_ang_speed)


def closed_loop_limits(
    config: BaseClosedLoopConfig,
    default_lin_speed: float,
    default_ang_speed: float,
) -> tuple[float, float]:
    """Resolve speed clamps from config and open-loop defaults."""
    max_lin = abs(default_lin_speed) if config.max_lin_speed is None else config.max_lin_speed
    max_ang = abs(default_ang_speed) if config.max_ang_speed is None else config.max_ang_speed
    return max(1e-6, float(max_lin)), max(1e-6, float(max_ang))


def wrap_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def integrate_se2(pose: np.ndarray, twist: np.ndarray, dt: float) -> np.ndarray:
    """Exact constant-twist SE(2) integration of pose (x, y, yaw)."""
    pose = np.asarray(pose, dtype=np.float64)
    twist = np.asarray(twist, dtype=np.float64)
    if pose.shape != (3,) or twist.shape != (3,):
        raise ValueError(f"expected shapes (3,), got pose {pose.shape}, twist {twist.shape}")
    if dt < 0:
        raise ValueError(f"dt must be >= 0, got {dt}")

    vx, vy, w = twist
    x, y, yaw = pose
    if abs(w) < 1e-9:
        dx_l = vx * dt
        dy_l = vy * dt
        dyaw = 0.0
    else:
        wt = w * dt
        sin_wt, cos_wt = np.sin(wt), np.cos(wt)
        dx_l = (vx * sin_wt + vy * (cos_wt - 1.0)) / w
        dy_l = (vx * (1.0 - cos_wt) + vy * sin_wt) / w
        dyaw = wt
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        x + cy * dx_l - sy * dy_l,
        y + sy * dx_l + cy * dy_l,
        wrap_pi(yaw + dyaw),
    ], dtype=np.float64)


def limit_twist(twist: np.ndarray, max_lin_speed: float, max_ang_speed: float) -> np.ndarray:
    """Clamp planar speed by vector norm and yaw speed by magnitude."""
    out = np.asarray(twist, dtype=np.float64).copy()
    lin = float(np.hypot(out[0], out[1]))
    if max_lin_speed >= 0.0 and lin > max_lin_speed > 0.0:
        out[:2] *= max_lin_speed / lin
    if max_ang_speed >= 0.0:
        out[2] = float(np.clip(out[2], -max_ang_speed, max_ang_speed))
    return out


def project_planar_twist_single_axis(
    twist: np.ndarray,
    *,
    xy_max_vel: float,
    yaw_max_vel: float,
    deadband: float,
    hysteresis_ratio: float = 0.0,
    prev_axis: Optional[int] = None,
) -> tuple[np.ndarray, Optional[int]]:
    """Keep only the dominant planar-twist axis (vx XOR vy XOR wz); zero the others.

    Shared by the WBC solver (``VegaWholeBodyIK``, on its QP base twist) and the followers
    (on the post-PD wheel command) so both pick the active axis with IDENTICAL semantics: the
    dominant axis is the largest ``|component|`` normalized by its velocity cap (``xy_max_vel``
    for vx/vy, ``yaw_max_vel`` for wz) -- so m/s and rad/s compare as a fraction of max speed.
    Below ``deadband`` (on that normalized scale) the twist is quiet and ALL axes are zeroed
    (active axis -> ``None``). A relative ``hysteresis_ratio`` retains ``prev_axis`` unless a
    different axis beats it by that fraction (and only while ``prev_axis`` itself stays at or
    above ``deadband``), so two near-equal axes do not chatter the chassis at the loop rate.
    Returns ``(projected_twist, active_axis)``; thread ``active_axis`` back in as ``prev_axis``
    next tick (reset to ``None`` on hold / standstill).
    """
    out = np.asarray(twist, dtype=np.float64).copy()
    if out.shape != (3,):
        raise ValueError(f"expected planar twist shape (3,), got {out.shape}")
    caps = np.array([xy_max_vel, xy_max_vel, yaw_max_vel], dtype=np.float64)
    if not np.all(np.isfinite(caps)) or np.any(caps <= 0.0):
        raise ValueError(
            f"caps must be finite and > 0, got xy_max_vel={xy_max_vel}, yaw_max_vel={yaw_max_vel}"
        )
    if not np.isfinite(deadband) or deadband < 0.0:
        raise ValueError(f"deadband must be finite and >= 0, got {deadband}")
    if not np.isfinite(hysteresis_ratio) or hysteresis_ratio < 0.0:
        raise ValueError(f"hysteresis_ratio must be finite and >= 0, got {hysteresis_ratio}")
    norm = np.abs(out) / caps
    k = int(np.argmax(norm))
    if norm[k] < deadband:
        return np.zeros(3), None
    if (
        prev_axis is not None
        and prev_axis != k
        and norm[prev_axis] >= deadband
        and norm[k] < (1.0 + hysteresis_ratio) * norm[prev_axis]
    ):
        k = prev_axis  # the new leader isn't decisively larger: keep the active axis
    keep = out[k]
    out[:] = 0.0
    out[k] = keep
    return out, k


def mask_planar_twist_for_base_dofs(
    twist: np.ndarray,
    *,
    base_dofs: str,
    allow_yaw_hold: bool = False,
) -> np.ndarray:
    """Apply the WBC base-DOF policy to a downstream planar chassis twist."""
    if base_dofs not in BASE_DOF_MODES:
        raise ValueError(f"base_dofs must be one of {BASE_DOF_MODES}, got {base_dofs!r}")
    out = np.asarray(twist, dtype=np.float64).copy()
    if out.shape != (3,):
        raise ValueError(f"expected planar twist shape (3,), got {out.shape}")
    if base_dofs == "xy" and not allow_yaw_hold:
        out[2] = 0.0
    return out


def project_planar_twist_single_axis_for_base_dofs(
    twist: np.ndarray,
    *,
    base_dofs: str,
    allow_yaw_hold: bool = False,
    xy_max_vel: float,
    yaw_max_vel: float,
    deadband: float,
    hysteresis_ratio: float = 0.0,
    prev_axis: Optional[int] = None,
) -> tuple[np.ndarray, Optional[int]]:
    """Single-axis project a command after applying the WBC base-DOF policy."""
    selector_twist = mask_planar_twist_for_base_dofs(
        twist,
        base_dofs=base_dofs,
        allow_yaw_hold=allow_yaw_hold,
    )
    if base_dofs == "xy" and not allow_yaw_hold and prev_axis == 2:
        prev_axis = None
    return project_planar_twist_single_axis(
        selector_twist,
        xy_max_vel=xy_max_vel,
        yaw_max_vel=yaw_max_vel,
        deadband=deadband,
        hysteresis_ratio=hysteresis_ratio,
        prev_axis=prev_axis,
    )


def reference_pose(
    start_pose: np.ndarray,
    reference_twist: np.ndarray,
    drive_time: float,
    elapsed_s: float,
) -> np.ndarray:
    """Time-parameterized SE(2) reference, clamped at its final target."""
    dt = float(np.clip(elapsed_s, 0.0, drive_time))
    return integrate_se2(start_pose, reference_twist, dt)


def pose_error_body(target_pose: np.ndarray, measured_pose: np.ndarray) -> np.ndarray:
    """Pose error target-measured, with xy expressed in the measured base frame."""
    target = np.asarray(target_pose, dtype=np.float64)
    measured = np.asarray(measured_pose, dtype=np.float64)
    dx, dy = target[0] - measured[0], target[1] - measured[1]
    c, s = np.cos(-measured[2]), np.sin(-measured[2])
    return np.array([
        c * dx - s * dy,
        s * dx + c * dy,
        wrap_pi(target[2] - measured[2]),
    ], dtype=np.float64)


def pd_twist(
    reference_pose: np.ndarray,
    feedforward_twist: np.ndarray,
    measured_pose: np.ndarray,
    kp_xy: float,
    kp_yaw: float,
    max_lin_speed: float,
    max_ang_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Feed-forward + proportional pose feedback command in measured body frame."""
    err = pose_error_body(reference_pose, measured_pose)
    # Feed-forward twist is expressed in the reference base frame; rotate xy into
    # the measured base frame before adding body-frame feedback.
    cf, sf = np.cos(err[2]), np.sin(err[2])
    ff = np.asarray(feedforward_twist, dtype=np.float64)
    ff_body = np.array([
        cf * ff[0] - sf * ff[1],
        sf * ff[0] + cf * ff[1],
        ff[2],
    ], dtype=np.float64)
    raw = ff_body + np.array([kp_xy * err[0], kp_xy * err[1], kp_yaw * err[2]])
    return limit_twist(raw, max_lin_speed, max_ang_speed), err


def shape_twist(
    twist: np.ndarray,
    previous: np.ndarray,
    dt: float,
    deadband_lin: float,
    deadband_ang: float,
    max_lin_speed: float,
    max_ang_speed: float,
    max_lin_accel: float,
    max_ang_accel: float,
) -> np.ndarray:
    """Apply deadband, velocity clamps, and per-axis acceleration clamps."""
    target = np.asarray(twist, dtype=np.float64).copy()
    prev = np.asarray(previous, dtype=np.float64)
    if abs(target[0]) < deadband_lin:
        target[0] = 0.0
    if abs(target[1]) < deadband_lin:
        target[1] = 0.0
    if abs(target[2]) < deadband_ang:
        target[2] = 0.0
    target = limit_twist(target, max_lin_speed, max_ang_speed)
    dmax = np.array([
        max(0.0, max_lin_accel) * max(0.0, dt),
        max(0.0, max_lin_accel) * max(0.0, dt),
        max(0.0, max_ang_accel) * max(0.0, dt),
    ])
    shaped = prev + np.clip(target - prev, -dmax, dmax)
    return limit_twist(shaped, max_lin_speed, max_ang_speed)


def shape_project_twist(
    raw: np.ndarray,
    previous_shaped: np.ndarray,
    dt: float,
    *,
    base_dofs: str,
    allow_yaw_hold: bool,
    deadband_lin: float,
    deadband_ang: float,
    max_lin_speed: float,
    max_ang_speed: float,
    max_lin_accel: float,
    max_ang_accel: float,
    post_linear_deadband: float,
    post_angular_deadband: float,
    enable_single_axis: bool,
    xy_max_vel: float,
    yaw_max_vel: float,
    single_axis_deadband: float,
    single_axis_hysteresis_ratio: float,
    prev_axis: Optional[int],
    preferred_axis: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[int]]:
    """Shared post-PD base shaping and final single-axis projection.

    Returns ``(command, shaped_anchor, active_axis)``. ``shaped_anchor`` is the full
    multi-axis shaped signal retained for the next slew-limiter tick; ``command`` is the
    masked/projected twist safe to dispatch. A non-``None`` ``preferred_axis`` lets a
    direct operator command retain its selected intent axis while active. The WBC path
    omits it and uses the dominant post-PD command with configured hysteresis.
    """
    shaped = shape_twist(
        raw,
        previous_shaped,
        dt,
        deadband_lin=deadband_lin,
        deadband_ang=deadband_ang,
        max_lin_speed=max_lin_speed,
        max_ang_speed=max_ang_speed,
        max_lin_accel=max_lin_accel,
        max_ang_accel=max_ang_accel,
    )
    shaped = deadband_planar_twist_by_reference(
        shaped,
        reference_twist=raw,
        linear_deadband=post_linear_deadband,
        angular_deadband=post_angular_deadband,
    )
    shaped = mask_planar_twist_for_base_dofs(
        shaped,
        base_dofs=base_dofs,
        allow_yaw_hold=allow_yaw_hold,
    )
    if not enable_single_axis:
        return shaped.copy(), shaped, None

    if preferred_axis is None:
        command, axis = project_planar_twist_single_axis_for_base_dofs(
            shaped,
            base_dofs=base_dofs,
            allow_yaw_hold=allow_yaw_hold,
            xy_max_vel=xy_max_vel,
            yaw_max_vel=yaw_max_vel,
            deadband=single_axis_deadband,
            hysteresis_ratio=single_axis_hysteresis_ratio,
            prev_axis=prev_axis,
        )
    else:
        if preferred_axis not in (0, 1, 2):
            raise ValueError(f"preferred_axis must be 0, 1, 2, or None, got {preferred_axis}")
        selector = mask_planar_twist_for_base_dofs(
            shaped,
            base_dofs=base_dofs,
            allow_yaw_hold=allow_yaw_hold,
        )
        cap = xy_max_vel if preferred_axis < 2 else yaw_max_vel
        command = np.zeros(3, dtype=np.float64)
        if abs(float(selector[preferred_axis])) / cap < single_axis_deadband:
            axis = None
        else:
            command[preferred_axis] = float(selector[preferred_axis])
            axis = preferred_axis
    command = mask_planar_twist_for_base_dofs(
        command,
        base_dofs=base_dofs,
        allow_yaw_hold=allow_yaw_hold,
    )
    return command, shaped, axis


def within_tolerance(err_body: np.ndarray, pos_tolerance: float, yaw_tolerance: float) -> bool:
    """Whether a body-frame pose error satisfies final leg tolerances."""
    return (
        float(np.hypot(err_body[0], err_body[1])) <= pos_tolerance
        and abs(float(err_body[2])) <= yaw_tolerance
    )


def fresh_age_s(update_t: float, now: float, timeout_s: float, label: str) -> float:
    """Return source age or raise if it is unavailable/stale."""
    if not np.isfinite(update_t):
        raise RuntimeError(f"{label} has not produced a pose yet")
    age = now - update_t
    if age > timeout_s:
        raise RuntimeError(f"{label} stale for {age:.3f}s > {timeout_s:.3f}s")
    return age


def fresh_sequence_age_s(
    seq: int,
    now: float,
    tracker: FreshSequence,
    timeout_s: float,
    label: str,
) -> float:
    """Return sequence stream age, updating freshness when seq changes."""
    if seq < 0:
        raise RuntimeError(f"{label} has no valid sequence number")
    if tracker.last_seq != seq:
        tracker.last_seq = seq
        tracker.last_update_t = now
    return fresh_age_s(tracker.last_update_t, now, timeout_s, label)
