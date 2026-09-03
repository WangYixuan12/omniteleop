"""Joystick base-twist shaping shared by both joystick-teleop followers.

The joystick-teleop followers (``scripts/wbc_joystick_robot.py`` on hardware,
``scripts/wbc_joystick_record.py`` in SAPIEN) drive the chassis DIRECTLY from the
controller thumbsticks instead of from the whole-body QP: the base is excluded from the
IK (``WBCConfig.lock_base_in_ik``) and the operator commands ``chassis_vx/vy/wz`` by hand.
This module turns that raw stick twist into a wheel command through the SAME closed-loop
path the WBC followers use for their QP base twist -- ``pd_twist`` feedback on a base pose
reference, then ``shape_twist`` (deadband -> clamp -> slew), post-deadband, the
``base_dofs`` policy mask, and the single-axis projection -- so a take recorded in sim
shapes the base identically on hardware. The difference from the WBC path is the pose
reference: a joystick INTEGRATOR (``integrate_se2`` of the stick twist) rather than the
solver's ``result.base_pose``. When single-axis mode is enabled, the stick twist is
projected before integration, matching the solver's base-pose update ordering.

Like the WBC path, the reference is integrated open-loop: a base that cannot keep up
(blocked / slipping / speed-clamped) winds it arbitrarily far ahead of the measured pose,
and ``pd_twist`` then repays that whole lead as a catch-up once the base comes free. Only
:meth:`JoystickBaseShaper.hold` bounds it, by parking the reference on the measured pose.

:class:`JoystickBaseShaper` owns the small amount of per-tick carry state (the integrated
reference pose, the slew anchor, the single-axis latch). The caller does the actual
dispatch (``chassis.set_velocity`` / the swerve quiet-hold), which differs between the
SAPIEN sim and the real swerve base -- exactly as it already differs between the two WBC
followers.
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
)


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

    # -- per-tick carry state (reset on engage; parked on hold) --------------------
    cmd_pose: np.ndarray = field(default_factory=lambda: np.zeros(3))
    prev_shaped: np.ndarray = field(default_factory=lambda: np.zeros(3))
    prev_axis: Optional[int] = None
    last_projected_joystick: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_intent_axis: Optional[int] = None

    @property
    def _allow_yaw_hold(self) -> bool:
        # Matches the WBC followers: in "xy" the QP yaw is hard-locked, but closed-loop
        # odometry may still command a small wz to hold measured heading against drift.
        return self.base_dofs == "xy" and bool(self.yaw_hold_in_xy)

    def reset(self, base_pose: Optional[np.ndarray] = None) -> None:
        """Engage: seed the reference at ``base_pose`` (default origin) and clear state."""
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
        self.cmd_pose = np.asarray(measured_pose, dtype=float).copy()
        self.prev_shaped = np.zeros(3)
        self.prev_axis = None
        self.last_projected_joystick = np.zeros(3)
        self.last_intent_axis = None

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
        max_lin = self.max_speed
        max_ang = 2.0 * self.max_speed
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
        self.last_projected_joystick = np.asarray(joy, dtype=float).copy()
        self.last_intent_axis = intent_axis
        # Open-loop reference integrator (see module docstring).
        self.cmd_pose = integrate_se2(self.cmd_pose, joy, dt)
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
        )
