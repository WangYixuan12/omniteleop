"""Swerve-drive inverse kinematics for the Vega mobile base.

Maps a body-frame base twist ``(vx, vy, wz)`` (x-fwd, y-left, CCW+) to per-wheel
**steering angle** and **drive angular velocity** for the two actively steered+driven
wheels (left, right). Pure geometry -- no SAPIEN/pinocchio dependency -- so it is unit
testable on its own.

This ports the math of ``dexcontrol.core.chassis.Chassis`` (verified against it):

- For a wheel at base-frame position ``p=(px, py)`` the ground-contact velocity of a
  rigid body moving with twist ``(vx, vy, wz)`` is ``v = (vx - wz*py, vy + wz*px)``.
- The steering **joint** value that points the wheel along ``v`` is ``q = -atan2(vy, vx)``
  -- the minus sign is because the URDF steer axis is ``-z`` (a positive joint value yaws
  the wheel clockwise about +z). dexcontrol uses the same ``-atan2``.
- There are two joint solutions: ``(q, +speed)`` and ``(wrap(q±pi), -speed)`` (drive
  backward instead of steering past 90°). We pick the one whose **measured** steer angle
  change is smallest and that lies within the steer limit -- avoids large steer slews and
  respects the real chassis' ±2.35 rad operating clamp.
- The drive joint is a wheel of radius ``R``: angular velocity ``omega = sign * speed / R``.
  The per-wheel ``sign`` (``L=+1``, ``R=-1``) accounts for the opposite drive axes
  (``L_wheel_j2`` axis ``+y``, ``R_wheel_j2`` axis ``-y``); it was verified empirically
  (a forward twist must roll the base in +x). It is constant across steer angles because
  the steer joint rotates the wheel heading and its drive axis together.

Geometry constants come from ``vega_no_effector.urdf`` (steer pivots at
``(0.185, ±0.225, 0.133)``); the rolling radius defaults to the value measured from the
settled wheel-axle height in SAPIEN (~0.086 m) and can be overridden / calibrated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

# Steer-pivot xy in the base frame (URDF: L/R at (0.185, ±0.225)).
WHEEL_XY: Dict[str, Tuple[float, float]] = {
    "L": (0.185, 0.225),
    "R": (0.185, -0.225),
}
# Drive-joint angular-velocity sign per wheel (opposite URDF drive axes +y / -y),
# verified empirically: forward twist => base advances in +x.
DRIVE_SIGN: Dict[str, float] = {"L": 1.0, "R": -1.0}

# Steering limits. The URDF revolute stop is wider, but dexcontrol/Vega1ChassisConfig
# clips steering commands at the real chassis operating limit.
URDF_STEER_LIMIT: float = 2.722
STEER_LIMIT: float = 2.35
DEFAULT_WHEEL_RADIUS: float = 0.0861

# The two steer solutions are 180 deg apart, so a "nearest to current" pick is
# decided by sub-degree noise when the wheel sits near the 90 deg ambiguity --
# and the two wheels can then split onto opposite branches and fight. Within this
# tolerance of an equal-distance tie we break deterministically (see
# ``resolve_steer_drive``) so both wheels always agree.
STEER_TIE_TOL: float = 0.35

# A steer solution whose desired angle is past the +/-STEER_LIMIT stop gets clamped, which
# points the wheel off the commanded heading. When that clamp distortion is LARGE -- the
# base drives ~45 deg diagonal on a "go straight" issued after the steering wound up near
# +/-pi (the antipodal branch clamps ~0.79 rad off) -- the OTHER branch is reachable and
# realizes the heading exactly, so we switch to it. A MARGINAL clamp (a hair past the stop)
# is left alone: switching would force a ~pi steer slew to shave a fraction of a degree.
# This cutoff (rad) is the large/marginal boundary; it sits in the empty gap between the
# ~0.10 rad yaw-wheel clamp and the ~0.79 rad diagonal (verified across single-axis twists).
CLAMP_DISTORTION_TOL: float = 0.35


@dataclass(frozen=True)
class WheelCommand:
    """One driven wheel's command: steer joint angle (rad) and drive rate (rad/s)."""

    steer: float        # target steering joint position (rad), within ±STEER_LIMIT
    drive_omega: float  # target drive joint angular velocity (rad/s)
    speed: float        # signed linear contact speed (m/s) along the wheel heading
    flipped: bool       # True if the "steer ±pi, drive backward" solution was chosen


def wheel_contact_velocity(
    twist: Tuple[float, float, float], wheel_xy: Tuple[float, float]
) -> Tuple[float, float]:
    """Ground-contact velocity ``(vx, vy)`` of a wheel under a body twist.

    ``twist`` is ``(vx, vy, wz)`` in the base frame; ``wheel_xy`` is the wheel's
    ``(px, py)`` base-frame position. Returns the wheel's planar velocity.
    """
    if len(twist) != 3:
        raise ValueError(f"twist must be (vx, vy, wz); got {twist!r}")
    if len(wheel_xy) != 2:
        raise ValueError(f"wheel_xy must be (px, py); got {wheel_xy!r}")
    vx, vy, wz = (float(c) for c in twist)
    px, py = (float(c) for c in wheel_xy)
    return (vx - wz * py, vy + wz * px)


def resolve_steer_drive(
    v_wheel: Tuple[float, float],
    current_steer: float,
    steer_limit: float = STEER_LIMIT,
    min_speed: float = 1e-6,
    tie_tol: float = STEER_TIE_TOL,
    clamp_distortion_tol: float = CLAMP_DISTORTION_TOL,
) -> Tuple[float, float, bool]:
    """Pick the steer angle + signed speed for a wheel velocity (dexcontrol convention).

    Returns ``(steer_angle, signed_speed, flipped)``. Of the two 180-deg-apart solutions
    ``(base_angle, +speed)`` and ``(base_angle±pi, -speed)`` -- the same physical wheel
    velocity -- we pick a *reachable* one when it matters, else the one nearer
    ``current_steer`` (smallest steer slew). A stationary wheel holds its current steer.

    **Reachable branch (the diagonal fix).** The two branches are pi apart and the reachable
    steer arc (``2*steer_limit``) is wider than pi, so at least one branch is always within
    the stop. The plain "nearest then clamp" rule (old dexcontrol) picks the near-but-out-of
    range branch when the steering has wound up near ``±pi`` and CLAMPS it -- pointing the
    wheel up to ~45 deg off the commanded velocity, so a "drive straight" drives the base
    diagonally. We measure each branch's *clamp distortion* (how far the stop pulls it off
    the commanded heading) and, when one branch clamps far off while the other is reachable
    (distortion gap ``> clamp_distortion_tol``), take the reachable branch -- which realizes
    the heading exactly. A MARGINAL clamp (a hair past the stop, e.g. the yaw wheel ~0.1 rad
    or a near-limit strafe) is left to the slew rule below, so we never do a ~pi steer slew
    to shave a fraction of a degree. This matches the fixed
    ``dexcontrol.core.chassis.Chassis._compute_wheel_control``.

    **Tie robustness.** Near the 90 deg ambiguity the two distances are nearly equal, so
    dexcontrol's strict ``<`` would resolve on sub-degree noise -- and the two wheels
    could split onto opposite branches and drive against each other on straight motion.
    Within ``tie_tol`` of an equal-distance tie we therefore force dexcontrol's tie choice
    (the flipped solution) deterministically, so both wheels always agree.
    """
    speed = math.hypot(v_wheel[0], v_wheel[1])
    if speed < min_speed:
        # No meaningful motion: hold the current steering, no drive.
        return current_steer, 0.0, False

    base_angle = -math.atan2(v_wheel[1], v_wheel[0])  # -z steer axis
    flipped_angle = base_angle + math.pi if base_angle < 0 else base_angle - math.pi

    def _clamp(a: float) -> float:
        return max(-steer_limit, min(steer_limit, a))

    # How far the physical steer stop pulls each branch off the commanded heading.
    base_distortion = abs(base_angle - _clamp(base_angle))
    flip_distortion = abs(flipped_angle - _clamp(flipped_angle))

    if base_distortion + clamp_distortion_tol < flip_distortion:
        flipped = False             # flip clamps far off-heading; base is reachable
    elif flip_distortion + clamp_distortion_tol < base_distortion:
        flipped = True              # base clamps far off-heading; flip is reachable
    else:
        # Comparable clamp distortion: nearest-branch pick on the UNCLAMPED angles (the
        # clamp is applied to the winner only), tie -> flipped so both wheels agree.
        d_base = abs(base_angle - current_steer)
        d_flip = abs(flipped_angle - current_steer)
        if abs(d_base - d_flip) <= tie_tol:
            flipped = True          # dexcontrol takes the flipped solution on a tie
        else:
            flipped = d_flip < d_base   # otherwise the nearer solution (smallest slew)
    steer, signed = (flipped_angle, -speed) if flipped else (base_angle, speed)
    steer = _clamp(steer)           # clamp the winner (dexcontrol)
    return steer, signed, flipped


def swerve_command(
    twist: Tuple[float, float, float],
    current_steer: Dict[str, float],
    wheel_radius: float = DEFAULT_WHEEL_RADIUS,
    steer_limit: float = STEER_LIMIT,
    tie_tol: float = STEER_TIE_TOL,
) -> Dict[str, WheelCommand]:
    """Full swerve IK: body twist -> per-wheel steer angle + drive angular velocity.

    Args:
        twist: base-frame ``(vx, vy, wz)`` (m/s, m/s, rad/s).
        current_steer: measured steer joint angle (rad) per wheel, keys ``"L"``/``"R"``.
        wheel_radius: rolling radius (m) used for ``omega = speed / R``.
        steer_limit: steering command limit (±rad).

    Returns:
        ``{"L": WheelCommand, "R": WheelCommand}``.
    """
    if wheel_radius <= 1e-6:
        raise ValueError(f"wheel_radius must be positive; got {wheel_radius}")
    for key in ("L", "R"):
        if key not in current_steer:
            raise ValueError(f"current_steer missing wheel {key!r}: {current_steer!r}")
    out: Dict[str, WheelCommand] = {}
    for key in ("L", "R"):
        v_wheel = wheel_contact_velocity(twist, WHEEL_XY[key])
        steer, signed_speed, flipped = resolve_steer_drive(
            v_wheel, float(current_steer[key]), steer_limit, tie_tol=tie_tol
        )
        drive_omega = DRIVE_SIGN[key] * signed_speed / wheel_radius
        out[key] = WheelCommand(
            steer=steer, drive_omega=drive_omega, speed=signed_speed, flipped=flipped
        )
    return out
