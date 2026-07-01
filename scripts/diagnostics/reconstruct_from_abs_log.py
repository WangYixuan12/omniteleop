#!/usr/bin/env python3
"""Replay the REAL controller motion (from the good absolute log) through 4 mappings.

Because target_abs(t) = G @ vr_now(t), the controller's motion relative to engage,
inv(target_abs[t0]) @ target_abs(t), is independent of the unknown calibration G. So
from the *good* absolute log we can reconstruct what each candidate EEF mapping would
have commanded for the SAME hand motion, feed it to the real whole-body IK, and see
which mappings keep the arm sane:

  abs : target_abs(t)                              (the log; known-good reference)
  A   : current relative  (pos rel + body-ori rel) nom_p+(pabs-pabs0); nom_R@Rabs0.T@Rabs
  B   : rigid relative                             nom @ inv(target_abs[t0]) @ target_abs
  C   : relative pos + absolute ori                nom_p+(pabs-pabs0); Rabs(t)

Each mapping is settled to its own engage pose, then the trajectory is replayed and the
arm-joint excursion measured (so abs's engage lunge is excluded -- we isolate the rotate).
"""
from __future__ import annotations

import h5py
import numpy as np

from omniteleop.follower.whole_body_ik import (
    HEAD_FRAME,
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    VegaWholeBodyIK,
    WBCConfig,
)

ABS = "/home/yixuan/Dexmate/wbc/wbc_teleop_06_20_abs_rotate.hdf5"


def to_mat(p):
    return np.asarray(p.homogeneous if hasattr(p, "homogeneous") else p, dtype=float)


def homog(R, p):
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = p
    return out


def main():
    with h5py.File(ABS, "r") as f:
        track = f["leader/tracking"][:].astype(bool)
        TL = f["target/left_pose"][:][track].astype(float)
        TR = f["target/right_pose"][:][track].astype(float)
    print(f"abs log tracking frames: {len(TL)}")

    ik = VegaWholeBodyIK(WBCConfig())
    ik.reset()
    nomL = to_mat(ik.frame_pose(LEFT_EE_FRAME))
    nomR = to_mat(ik.frame_pose(RIGHT_EE_FRAME))
    head0 = to_mat(ik.frame_pose(HEAD_FRAME))

    L0, R0 = TL[0], TR[0]  # engage poses

    def build(name, T, nom, T0):
        Rt, pt = T[:3, :3], T[:3, 3]
        R0_, p0_ = T0[:3, :3], T0[:3, 3]
        if name == "abs":
            return T
        if name == "A(current)":
            return homog(nom[:3, :3] @ R0_.T @ Rt, nom[:3, 3] + (pt - p0_))
        if name == "B(rigid)":
            return nom @ np.linalg.inv(T0) @ T
        if name == "C(relpos+absori)":
            return homog(Rt, nom[:3, 3] + (pt - p0_))
        raise ValueError(name)

    dt = 0.01
    print(f"\n{'mapping':18s} {'rangeL':>7s} {'rangeR':>7s} {'max|qL|':>8s} {'max|qR|':>8s} "
          f"{'held':>5s} {'fail':>5s} {'minSelf':>8s} {'eeLerr':>7s}")
    for name in ("abs", "A(current)", "B(rigid)", "C(relpos+absori)"):
        ik.reset()
        # Settle to the engage pose so abs's initial lunge is excluded.
        for _ in range(200):
            ik.solve(build(name, L0, nomL, L0), build(name, R0, nomR, R0), dt, head_target=head0)
        qL, qR, held, fail, min_self, ee = [], [], 0, 0, 1e9, 0.0
        for i in range(len(TL)):
            tL = build(name, TL[i], nomL, L0)
            tR = build(name, TR[i], nomR, R0)
            res = ik.solve(tL, tR, dt, head_target=head0)
            qL.append(np.asarray(res.left_arm))
            qR.append(np.asarray(res.right_arm))
            held += int(res.held)
            fail += int(not res.success)
            min_self = min(min_self, float(res.min_self_distance))
            ee = max(ee, float(res.left_ee_error), float(res.right_ee_error))
        qL, qR = np.array(qL), np.array(qR)
        print(f"{name:18s} {(qL.max(0)-qL.min(0)).max():7.2f} {(qR.max(0)-qR.min(0)).max():7.2f} "
              f"{np.abs(qL).max():8.2f} {np.abs(qR).max():8.2f} {held:5d} {fail:5d} "
              f"{min_self*100:7.1f}cm {ee*1000:6.0f}mm")


if __name__ == "__main__":
    main()
