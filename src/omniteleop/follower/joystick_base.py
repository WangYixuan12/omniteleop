"""Joystick base-twist shaping shared by both joystick-teleop followers.

The joystick-teleop followers (``scripts/wbc_joystick_robot.py`` on hardware,
``scripts/wbc_joystick_record.py`` in SAPIEN) drive the chassis DIRECTLY from the
controller thumbsticks instead of from the whole-body QP: the base is excluded from the
IK (``WBCConfig.lock_base_in_ik``) and the operator selects a chassis direction by hand.
After deadband/masking/axis selection, this module maps every active translation to one
fixed linear speed and every active turn to one fixed angular speed. It then turns that
intent into a wheel command through the SAME closed-loop
path the WBC followers use for their QP base twist -- ``pd_twist`` feedback on a base pose
reference, then ``shape_twist`` (deadband -> clamp -> slew), post-deadband, the
``base_dofs`` policy mask, and the single-axis projection -- so a take recorded in sim
shapes the base identically on hardware. The difference from the WBC path is the pose
reference: a joystick INTEGRATOR (``integrate_se2`` of the stick twist) rather than the
solver's ``result.base_pose``. When single-axis mode is enabled, the stick twist is
projected before integration, matching the solver's base-pose update ordering.

Unlike the WBC path, the reference is NOT integrated open-loop: after each integration
step it is clamped to at most ``reference_lead_xy`` / ``reference_lead_yaw`` ahead of the
measured pose (``joystick_teleop:`` in ``wbik.yaml``). A base that cannot keep up (blocked
/ slipping / speed-clamped) therefore accrues at most that bounded lead, and ``pd_twist``
repays at most that much as a catch-up once the base comes free -- instead of the whole
distance the stick was held, which on a real take grew to 0.9 m and kept the chassis
driving for seconds after every stick release. :meth:`JoystickBaseShaper.hold` still parks
the reference exactly on the measured pose.

:class:`JoystickBaseShaper` owns the small amount of per-tick carry state (the integrated
reference pose, the slew anchor, the single-axis latch). The caller does the actual
dispatch (``chassis.set_velocity`` / the swerve quiet-hold), which differs between the
SAPIEN sim and the real swerve base -- exactly as it already differs between the two WBC
followers.

With ``turn_90`` enabled, one live yaw gesture expands into a finite stream of these
same fixed-speed intents. Integration stops just before the measured-relative heading
goal, then the existing PD controller settles the remaining heading error. At completion,
yaw slews to zero and the reference parks on feedback until new intent arrives. XY drift
is discarded during turning so it cannot trigger steering corrections at the endpoint.
Expanded intents remain recorded, but replay through the ordinary velocity controller
does not reproduce this turn-specific stop state exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from omniteleop.follower.base_closed_loop import (
    integrate_se2,
    mask_planar_twist_for_base_dofs,
    pd_twist,
    project_planar_twist_single_axis_for_base_dofs,
    shape_project_twist,
    wrap_pi,
)


def fixed_speed_planar_twist(
    twist: np.ndarray,
    *,
    translation_speed: float,
    rotation_speed: float,
) -> np.ndarray:
    """Map nonzero planar intent to fixed translation/yaw speeds.

    The x/y direction is preserved and normalized to ``translation_speed``; yaw keeps
    only its sign and uses ``rotation_speed``. With the default single-axis policy this
    produces exactly one of ``+/-translation_speed`` or ``+/-rotation_speed``. Keeping
    the vector form also gives direction-only behavior if single-axis mode is disabled.
    """
    value = np.asarray(twist, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"planar twist must be finite (3,), got {value!r}")
    speeds = np.asarray([translation_speed, rotation_speed], dtype=float)
    if not np.all(np.isfinite(speeds)) or np.any(speeds <= 0.0):
        raise ValueError(
            "fixed joystick translation/rotation speeds must be finite and > 0, "
            f"got {speeds.tolist()}"
        )

    out = np.zeros(3, dtype=float)
    linear_norm = float(np.linalg.norm(value[:2]))
    if linear_norm > 0.0:
        out[:2] = value[:2] * (float(translation_speed) / linear_norm)
    if value[2] != 0.0:
        out[2] = np.copysign(float(rotation_speed), value[2])
    return out


@dataclass
class JoystickBaseShaper:
    """Shape a raw joystick body twist ``(vx, vy, wz)`` into a chassis command.

    Tunables mirror the ``vr_teleop:`` block of ``wbik.yaml`` (angular limits are ``2x``
    the linear ones, matching both WBC followers). ``base_dofs`` /
    ``enable_base_single_axis`` and the single-axis parameters come from the same
    ``WBCConfig`` the IK uses, so the chassis obeys the configured base policy (``"xy"``
    drops joystick yaw; ``"xy_yaw"`` keeps it) even though the base is locked out of the IK.
    """

    base_dofs: str
    kp_xy: float
    kp_yaw: float
    max_speed: float                  # base_max_speed (m/s); angular cap = 2x
    deadband: float                   # base_deadband (m/s); angular = 2x
    accel: float                      # base_accel (m/s^2); angular = 2x
    post_linear_deadband: float
    post_angular_deadband: float
    yaw_hold_in_xy: bool
    enable_single_axis: bool
    xy_max_vel: float                 # WBC base_xy_max_vel (single-axis normalization)
    yaw_max_vel: float                # WBC base_yaw_max_vel
    single_axis_deadband: float
    dispatch_single_axis_deadband: float
    single_axis_hysteresis_ratio: float
    intent_xy_max_vel: float           # joystick mapping cap used to normalize stick intent
    intent_yaw_max_vel: float
    intent_hysteresis_ratio: float
    translation_speed: float           # fixed active translation magnitude (m/s)
    rotation_speed: float              # fixed active yaw magnitude (rad/s)
    reference_lead_xy: float           # max reference lead over the measured pose (m)
    reference_lead_yaw: float          # max reference heading lead (rad); 0 = no lead
    turn_90: bool = False              # live collection: one measured quarter-turn per gesture

    # -- per-tick carry state (reset on engage; parked on hold) --------------------
    cmd_pose: np.ndarray = field(default_factory=lambda: np.zeros(3))
    prev_shaped: np.ndarray = field(default_factory=lambda: np.zeros(3))
    prev_axis: Optional[int] = None
    last_projected_joystick: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_intent_axis: Optional[int] = None
    turn_target_yaw: Optional[float] = field(default=None, init=False)
    _turn_armed: bool = field(default=False, init=False)
    _turn_elapsed: float = field(default=0.0, init=False)
    _turn_parked: bool = field(default=False, init=False)

    TURN_TOLERANCE_RAD = float(np.deg2rad(2.0))

    def __post_init__(self) -> None:
        leads = np.asarray([self.reference_lead_xy, self.reference_lead_yaw], dtype=float)
        if not np.all(np.isfinite(leads)) or np.any(leads < 0.0):
            raise ValueError(
                "reference_lead_xy / reference_lead_yaw must be finite and >= 0, got "
                f"{leads.tolist()}"
            )
        if self.turn_90:
            if self.base_dofs != "xy_yaw":
                raise ValueError("--turn-90 requires base_dofs: xy_yaw")
            if self.accel <= 0 or self.kp_yaw <= 0:
                raise ValueError("--turn-90 requires positive base_accel and base_kp_yaw")
            angular_floor = max(
                2 * self.deadband, self.post_angular_deadband,
                self.dispatch_single_axis_deadband * self.yaw_max_vel,
            )
            if min(self.kp_yaw * self.TURN_TOLERANCE_RAD,
                   self.rotation_speed, 2 * self.max_speed) <= angular_floor:
                raise ValueError("--turn-90 yaw speed/gain is too low for the configured deadbands")

    def cancel_turn(self) -> None:
        """Discard a pending turn; require a fresh neutral input before another."""
        self.turn_target_yaw = None
        self._turn_armed = False
        self._turn_elapsed = 0.0
        self._turn_parked = True

    @property
    def _allow_yaw_hold(self) -> bool:
        # Matches the WBC followers: in "xy" the QP yaw is hard-locked, but closed-loop
        # odometry may still command a small wz to hold measured heading against drift.
        return self.base_dofs == "xy" and bool(self.yaw_hold_in_xy)

    def reset(self, base_pose: Optional[np.ndarray] = None) -> None:
        """Engage: seed the reference at ``base_pose`` (default origin) and clear state."""
        self.cancel_turn()
        self.cmd_pose = (
            np.zeros(3) if base_pose is None else np.asarray(base_pose, dtype=float).copy()
        )
        self.prev_shaped = np.zeros(3)
        self.prev_axis = None
        self.last_projected_joystick = np.zeros(3)
        self.last_intent_axis = None

    def hold(self, measured_pose: np.ndarray) -> None:
        """Hold (e-stop / failed solve / stale): park the reference at the measured pose.

        Parking the integrated reference on the measured base means the PD error is zero
        on resume, so the base does not lunge to a stale reference after the hold clears.
        """
        self.cancel_turn()
        self.cmd_pose = np.asarray(measured_pose, dtype=float).copy()
        self.prev_shaped = np.zeros(3)
        self.prev_axis = None
        self.last_projected_joystick = np.zeros(3)
        self.last_intent_axis = None

    def _quarter_turn_intent(
        self, joy: np.ndarray, measured: np.ndarray, dt: float
    ) -> np.ndarray:
        """Latch relative yaw from feedback, with no queued/repeated turns.

        Neutral releases the gesture but lets an active turn finish. Translation or
        opposite yaw cancels; safety holds use cancel_turn(). A timeout bounds motion
        if valid odometry reports a blocked base. The returned intent remains a dense
        fixed-speed policy label; endpoint deceleration belongs to the pose controller.
        """
        translating = bool(np.any(joy[:2]))
        if self.turn_target_yaw is not None:
            error = float(wrap_pi(self.turn_target_yaw - measured[2]))
            self._turn_elapsed += dt
            if translating or (joy[2] * error < 0):
                self.cancel_turn()
                self.cmd_pose = measured.copy()
            elif abs(error) <= self.TURN_TOLERANCE_RAD:
                self.cancel_turn()
                return np.zeros(3)
            elif self._turn_elapsed > 2 * (np.pi / 2) / min(
                self.rotation_speed, 2 * self.max_speed
            ) + 5.0:
                self.cancel_turn()
                raise RuntimeError("90-degree joystick turn timed out before reaching measured heading")
        elif joy[2] == 0:
            self._turn_armed = True
        elif self._turn_armed and not translating:
            self._turn_parked = False
            self.turn_target_yaw = float(wrap_pi(
                measured[2] + np.copysign(np.pi / 2, joy[2])
            ))
            self._turn_armed = False
            self._turn_elapsed = 0.0

        if self.turn_target_yaw is not None:
            error = float(wrap_pi(self.turn_target_yaw - measured[2]))
            reference_remaining = float(wrap_pi(self.turn_target_yaw - self.cmd_pose[2]))
            # End the velocity intent before its next integration step crosses the
            # goal. The existing bounded reference/PD/slew chain then brakes and
            # settles, just as it does for a recorded or policy-produced release.
            # No special turn-only controller is hidden behind the policy labels.
            if np.sign(error) * reference_remaining <= self.rotation_speed * dt:
                return np.zeros(3)
            return np.array([0.0, 0.0, np.copysign(self.rotation_speed, error)])
        joy = joy.copy()
        joy[2] = 0.0
        return joy

    def step(
        self, joystick_twist: np.ndarray, measured_pose: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One tick: raw stick twist -> shaped single-axis chassis command.

        Returns ``(cmd, pd_raw, pd_err)`` -- the wheel command to dispatch, the pre-shaping
        PD/feed-forward twist, and the body-frame pose error -- so the caller can log the
        same ``/debug`` fields the WBC followers do. Advances the internal reference,
        slew anchor, and single-axis latch.
        """
        allow_yaw_hold = self._allow_yaw_hold
        # The shared base_max_speed remains a hard safety ceiling, but joystick motion
        # has its own lower fixed-speed contract. Capping the PD correction here is
        # essential: otherwise accumulated pose error could still accelerate the
        # chassis above the operator-selected speed.
        max_lin = min(self.max_speed, self.translation_speed)
        max_ang = min(2.0 * self.max_speed, self.rotation_speed)
        # Feed-forward = the stick twist, STRICTLY masked to the base policy (no yaw-hold
        # exemption): in "xy" the operator's deliberate yaw is dropped so the reference
        # never rotates, the analog of the WBC follower's QP-pinned result.base_twist. The
        # yaw-hold exemption below applies only to the PD OUTPUT, so a small corrective wz
        # can still hold measured heading against wheel/ground drift during a strafe.
        joy = mask_planar_twist_for_base_dofs(
            joystick_twist, base_dofs=self.base_dofs, allow_yaw_hold=False
        )
        intent_axis: Optional[int] = None
        if self.enable_single_axis:
            # Match VegaWholeBodyIK: project BEFORE integrating the reference so an
            # off-axis stick component cannot accumulate pose debt that is repaid later.
            joy, intent_axis = project_planar_twist_single_axis_for_base_dofs(
                joy,
                base_dofs=self.base_dofs,
                allow_yaw_hold=False,
                xy_max_vel=self.intent_xy_max_vel,
                yaw_max_vel=self.intent_yaw_max_vel,
                deadband=self.single_axis_deadband,
                hysteresis_ratio=self.intent_hysteresis_ratio,
                prev_axis=self.prev_axis,
            )
        joy = fixed_speed_planar_twist(
            joy,
            translation_speed=self.translation_speed,
            rotation_speed=self.rotation_speed,
        )
        measured = np.asarray(measured_pose, dtype=float)
        if measured.shape != (3,) or not np.all(np.isfinite(measured)):
            raise ValueError(f"measured_pose must be finite (3,), got {measured_pose!r}")
        if self.turn_90:
            joy = self._quarter_turn_intent(joy, measured, dt)
            intent_axis = 2 if joy[2] != 0.0 else (
                intent_axis if np.any(joy) else None
            )
        self.last_projected_joystick = np.asarray(joy, dtype=float).copy()
        self.last_intent_axis = intent_axis
        if self.turn_90 and self._turn_parked:
            if np.any(joy):
                self._turn_parked = False
            else:
                # A completed/cancelled gesture is a stop, not a request to recover
                # accumulated XY drift or chase tracking noise. Brake only the yaw
                # command actually dispatched, then remain parked until fresh intent.
                self.cmd_pose = measured.copy()
                previous_yaw = self.prev_shaped[2] if self.prev_axis == 2 else 0.0
                yaw = np.sign(previous_yaw) * max(
                    0.0, abs(previous_yaw) - 2.0 * self.accel * max(dt, 0.0)
                )
                cmd = np.array([0.0, 0.0, yaw])
                self.prev_shaped = cmd.copy()
                self.prev_axis = 2 if yaw else None
                return cmd, np.zeros(3), np.zeros(3)
        if self.turn_90 and self.turn_target_yaw is not None:
            # Pure rotation must not accumulate a translation correction that can
            # win the single-axis selector during endpoint braking.
            self.cmd_pose[:2] = measured[:2]
            self.prev_shaped[:2] = 0.0
            intent_axis = 2
        # Reference integrator, clamped to a bounded lead over the measured pose (see the
        # module docstring): the carrot stays at most reference_lead_* ahead of the chassis.
        self.cmd_pose = integrate_se2(self.cmd_pose, joy, dt)
        lead_xy = self.cmd_pose[:2] - measured[:2]
        lead_dist = float(np.hypot(lead_xy[0], lead_xy[1]))
        if lead_dist > self.reference_lead_xy:
            self.cmd_pose[:2] = measured[:2] + lead_xy * (self.reference_lead_xy / lead_dist)
        lead_yaw = float(wrap_pi(self.cmd_pose[2] - measured[2]))
        if abs(lead_yaw) > self.reference_lead_yaw:
            self.cmd_pose[2] = float(
                wrap_pi(measured[2] + np.copysign(self.reference_lead_yaw, lead_yaw))
            )
        raw, err = pd_twist(
            self.cmd_pose, joy, measured_pose,
            kp_xy=self.kp_xy, kp_yaw=self.kp_yaw,
            max_lin_speed=max_lin, max_ang_speed=max_ang,
        )
        raw = mask_planar_twist_for_base_dofs(
            raw, base_dofs=self.base_dofs, allow_yaw_hold=allow_yaw_hold
        )
        pd_raw = np.asarray(raw, dtype=float).copy()
        # Shared with HardwareDriver._drive_base. Joystick intent supplies the preferred
        # axis while active; quiet ticks fall back to dominant post-PD correction.
        cmd, self.prev_shaped, self.prev_axis = shape_project_twist(
            raw,
            self.prev_shaped,
            dt,
            base_dofs=self.base_dofs,
            allow_yaw_hold=allow_yaw_hold,
            deadband_lin=self.deadband, deadband_ang=2.0 * self.deadband,
            max_lin_speed=max_lin, max_ang_speed=max_ang,
            max_lin_accel=self.accel, max_ang_accel=2.0 * self.accel,
            post_linear_deadband=self.post_linear_deadband,
            post_angular_deadband=self.post_angular_deadband,
            enable_single_axis=self.enable_single_axis,
            xy_max_vel=self.xy_max_vel,
            yaw_max_vel=self.yaw_max_vel,
            single_axis_deadband=self.single_axis_deadband,
            dispatch_single_axis_deadband=self.dispatch_single_axis_deadband,
            single_axis_hysteresis_ratio=self.single_axis_hysteresis_ratio,
            prev_axis=self.prev_axis,
            preferred_axis=intent_axis,
        )
        return cmd, pd_raw, np.asarray(err, dtype=float)

    @classmethod
    def from_configs(cls, cfg, args) -> "JoystickBaseShaper":
        """Build from a ``WBCConfig`` and a follower ``args`` namespace (vr_teleop values).

        ``args`` carries the ``vr_teleop:`` tunables the followers already bind onto their
        argparse namespace (``base_kp_xy``, ``base_max_speed``, ``base_deadband``, ...), so
        both followers construct the shaper the same way from the shared ``wbik.yaml``.
        """
        return cls(
            base_dofs=cfg.base_dofs,
            kp_xy=float(args.base_kp_xy),
            kp_yaw=float(args.base_kp_yaw),
            max_speed=float(args.base_max_speed),
            deadband=float(args.base_deadband),
            accel=float(args.base_accel),
            post_linear_deadband=float(args.base_post_linear_deadband),
            post_angular_deadband=float(args.base_post_angular_deadband),
            yaw_hold_in_xy=bool(args.base_yaw_hold_in_xy),
            enable_single_axis=bool(cfg.enable_base_single_axis),
            xy_max_vel=float(cfg.base_xy_max_vel),
            yaw_max_vel=float(cfg.base_yaw_max_vel),
            single_axis_deadband=float(cfg.base_single_axis_deadband),
            dispatch_single_axis_deadband=float(cfg.base_dispatch_single_axis_deadband),
            single_axis_hysteresis_ratio=float(cfg.base_single_axis_hysteresis_ratio),
            intent_xy_max_vel=float(args.joystick_stick_max_vx),
            intent_yaw_max_vel=float(args.joystick_stick_max_wz),
            intent_hysteresis_ratio=float(args.joystick_single_axis_hysteresis_ratio),
            translation_speed=float(args.joystick_translation_speed),
            rotation_speed=float(args.joystick_rotation_speed),
            reference_lead_xy=float(args.joystick_reference_lead_xy),
            reference_lead_yaw=float(args.joystick_reference_lead_yaw),
            turn_90=bool(getattr(args, "turn_90", False)),
        )
