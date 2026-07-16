#!/usr/bin/env python3
"""Analyze a wbc_joystick_record base-debug HDF5 (schema wbc_joystick_record_debug/v1).

Answers "does a strafe stick actually strafe?" by decomposing the per-tick base motion into
the ROBOT BODY frame (forward dx / left dy) and correlating it with the received joystick
twist ``rx_chassis``. A clean setup gives corr(rx_vy, body_dy) ~= +/-1 and corr(rx_vy,
body_dx) ~= 0; if instead rx_vy correlates with body_dx (forward), a strafe command is
producing forward motion. It also reports the base YAW during strafes -- if the base is
yawed, a body-frame strafe looks like forward/back in a world/third-person view even though
it is correct in the robot's own frame.

Usage:  python scripts/analyze_joystick_dbg.py wbc_joystick_dbg_<stamp>.hdf5
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hdf5", help="base-debug HDF5 written by wbc_joystick_record.py")
    ap.add_argument("--min-mag", type=float, default=0.05,
                    help="minimum |rx| (normalized) to count a tick as commanded (default 0.05).")
    args = ap.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        g = f["debug"]
        rx = g["rx_chassis"][:]        # (N, 3) received joystick twist
        cmd = g["base_cmd"][:]         # (N, 3) shaped twist sent to set_velocity
        bp = g["base_pose"][:]         # (N, 3) integrated sim base (x, y, yaw)
        stage = [s.decode() for s in g["stage"][:]]
        hold = g["hold_base"][:]
        n = g.attrs["n_frames"]
    print(f"{args.hdf5}: {n} ticks, stages={sorted(set(stage))}, "
          f"held {int(np.sum(hold))} ticks")

    # Per-tick world base delta rotated into the body frame at the tick's start yaw.
    dbp = np.diff(bp, axis=0)
    yaw = bp[:-1, 2]
    c, s = np.cos(-yaw), np.sin(-yaw)
    body_dx = c * dbp[:, 0] - s * dbp[:, 1]      # forward (+x body)
    body_dy = s * dbp[:, 0] + c * dbp[:, 1]      # left (+y body)
    rxv = rx[:-1]

    active = (np.abs(rxv[:, 0]) + np.abs(rxv[:, 1]) + np.abs(rxv[:, 2])) > args.min_mag
    strafe = active & (np.abs(rxv[:, 1]) > np.abs(rxv[:, 0]) + 1e-6) & (np.abs(rxv[:, 1]) > np.abs(rxv[:, 2]))
    fwd = active & (np.abs(rxv[:, 0]) > np.abs(rxv[:, 1]) + 1e-6) & (np.abs(rxv[:, 0]) > np.abs(rxv[:, 2]))
    turn = active & (np.abs(rxv[:, 2]) > np.abs(rxv[:, 0])) & (np.abs(rxv[:, 2]) > np.abs(rxv[:, 1]))

    def report(mask: np.ndarray, label: str) -> None:
        if int(mask.sum()) < 3:
            print(f"  {label}: <3 ticks, skipped")
            return
        print(f"  {label} ({int(mask.sum())} ticks): "
              f"body dx(forward)={body_dx[mask].sum():+.3f} m  "
              f"body dy(left)={body_dy[mask].sum():+.3f} m  "
              f"dyaw={np.degrees(np.diff(bp[:, 2])[mask].sum()):+.0f} deg  "
              f"base_yaw~{np.degrees(np.mean(yaw[mask])):+.0f} deg")

    print("\nmotion by dominant received-stick axis (body frame):")
    report(strafe, "STRAFE (rx vy) -> want dy")
    report(fwd, "FORWARD (rx vx) -> want dx")
    report(turn, "TURN (rx wz) -> want dyaw")

    if int(active.sum()) >= 5:
        def corr(a, b):
            return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-9 and np.std(b) > 1e-9 else float("nan")
        print("\ncorrelations over commanded ticks (want vy<->dy, vx<->dx):")
        print(f"  corr(rx_vy, body_dx)={corr(rxv[active, 1], body_dx[active]):+.2f}  (should be ~0)")
        print(f"  corr(rx_vy, body_dy)={corr(rxv[active, 1], body_dy[active]):+.2f}  (should be ~+/-1)")
        print(f"  corr(rx_vx, body_dx)={corr(rxv[active, 0], body_dx[active]):+.2f}  (should be ~+/-1)")

    # Did the shaper preserve the axis, or swap/mix it? Compare rx vs the shaped command.
    both = active & (np.abs(cmd[:-1, :2]).sum(axis=1) > 1e-4)
    if int(both.sum()) >= 5:
        print("\nshaper axis check (rx vs base_cmd, commanded ticks):")
        print(f"  corr(rx_vy, cmd_vy)={float(np.corrcoef(rxv[both,1], cmd[:-1][both,1])[0,1]):+.2f}  "
              f"corr(rx_vy, cmd_vx)={float(np.corrcoef(rxv[both,1], cmd[:-1][both,0])[0,1]):+.2f}")


if __name__ == "__main__":
    main()
