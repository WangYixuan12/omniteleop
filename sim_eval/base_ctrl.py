"""Planar base-twist shaping for driving the OmniGibson holonomic base from the WBC output.

Pure-numpy port of the relevant bits of omniteleop/follower/base_closed_loop.py (that module
lives in the omniteleop venv and pulls in omniteleop.wbc_stream, neither importable from the
behavior conda env). The WBC returns base_pose (world x,y,yaw) + base_twist (feed-forward, in
the reference base frame); the follower turns that into a chassis velocity command via
pd_twist (rotate feed-forward into the measured base frame + PD on the SE(2) pose error) then
shape_twist (deadband + speed clamp + acceleration/slew clamp). We reproduce that here.
"""
from __future__ import annotations
import numpy as np


def wrap_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


def limit_twist(twist, max_lin_speed, max_ang_speed):
    out = np.asarray(twist, dtype=np.float64).copy()
    lin = float(np.hypot(out[0], out[1]))
    if max_lin_speed >= 0.0 and lin > max_lin_speed > 0.0:
        out[:2] *= max_lin_speed / lin
    if max_ang_speed >= 0.0:
        out[2] = float(np.clip(out[2], -max_ang_speed, max_ang_speed))
    return out


def pose_error_body(target_pose, measured_pose):
    """SE(2) error (target - measured), with xy expressed in the measured base frame."""
    t = np.asarray(target_pose, dtype=np.float64)
    m = np.asarray(measured_pose, dtype=np.float64)
    dx, dy = t[0] - m[0], t[1] - m[1]
    c, s = np.cos(m[2]), np.sin(m[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap_pi(t[2] - m[2])])


def pd_twist(reference_pose, feedforward_twist, measured_pose, kp_xy, kp_yaw,
             max_lin_speed, max_ang_speed):
    """Feed-forward + proportional pose feedback command, in the measured base frame."""
    err = pose_error_body(reference_pose, measured_pose)
    cf, sf = np.cos(err[2]), np.sin(err[2])
    ff = np.asarray(feedforward_twist, dtype=np.float64)
    ff_body = np.array([cf * ff[0] - sf * ff[1], sf * ff[0] + cf * ff[1], ff[2]])
    raw = ff_body + np.array([kp_xy * err[0], kp_xy * err[1], kp_yaw * err[2]])
    return limit_twist(raw, max_lin_speed, max_ang_speed)


def shape_twist(twist, previous, dt, deadband_lin, deadband_ang,
                max_lin_speed, max_ang_speed, max_lin_accel, max_ang_accel):
    """Deadband, speed clamp, and per-axis acceleration (slew) clamp."""
    target = np.asarray(twist, dtype=np.float64).copy()
    if abs(target[0]) < deadband_lin:
        target[0] = 0.0
    if abs(target[1]) < deadband_lin:
        target[1] = 0.0
    if abs(target[2]) < deadband_ang:
        target[2] = 0.0
    target = limit_twist(target, max_lin_speed, max_ang_speed)
    prev = np.asarray(previous, dtype=np.float64)
    dmax = np.array([max(0.0, max_lin_accel) * max(0.0, dt),
                     max(0.0, max_lin_accel) * max(0.0, dt),
                     max(0.0, max_ang_accel) * max(0.0, dt)])
    shaped = prev + np.clip(target - prev, -dmax, dmax)
    return limit_twist(shaped, max_lin_speed, max_ang_speed)
