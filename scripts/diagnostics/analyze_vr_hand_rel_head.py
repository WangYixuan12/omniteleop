#!/usr/bin/env python3
"""Decide whether the operator's HANDS stay fixed relative to the HEADSET during a turn.

Reads a RAW-VR log written by ``wbc_vr_leader.py --debug-vr`` (per-tick raw headset +
both controller poses, untouched Quest room-frame). For each frame it expresses each
hand in the HEAD frame, ``head_T_hand = inv(raw_head) @ raw_hand``:

  * If the operator turns head + hands TOGETHER rigidly, ``head_T_hand`` position is
    CONSTANT through the turn  ->  the "arms sit left after a turn" symptom is a pure
    follower/base geometry artifact, fixable by HEAD-RELATIVE hand mapping (transfer the
    operator hand-relative-to-head pose onto the robot's head frame).
  * If the hand POSITION drifts in the head frame while the head sweeps  ->  the hands
    really move relative to the head, so head-relative mapping would just reproduce the
    drift; the fix must instead FRAME THE HANDS with the base/camera.

This is the clean rigid-turn test: hold both controllers fixed relative to your head and
rotate in place ~80-90 deg, no reaching.

Run:  python scripts/diagnostics/analyze_vr_hand_rel_head.py /path/to/vr_raw_*.hdf5
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np


def _geodesic_deg(r0: np.ndarray, r: np.ndarray) -> float:
    """Angle (deg) between two rotation matrices."""
    c = (np.trace(r0.T @ r) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hdf5", help="vr_raw_<timestamp>.hdf5 from wbc_vr_leader.py --debug-vr")
    args = ap.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        g = f["vr"]
        t = g["t"][()]
        stage = g["stage"][()]
        H, L, R = g["raw_head"][()], g["raw_left"][()], g["raw_right"][()]

    fin = lambda A: np.all(np.isfinite(A.reshape(len(A), -1)), axis=1)
    m = (stage == b"teleop") & fin(H) & fin(L) & fin(R)
    idx = np.where(m)[0]
    if len(idx) < 20:
        raise SystemExit(f"only {len(idx)} valid teleop frames -- need a longer take")
    print(f"{len(idx)} valid teleop frames over {t[idx][-1] - t[idx][0]:.1f}s")

    i0 = idx[0]
    Hinv0 = np.linalg.inv(H[i0])
    head_sweep = np.array([_geodesic_deg(H[i0][:3, :3], H[i][:3, :3]) for i in idx])

    # hand position in the head frame; drift = distance from the first-frame value
    def hand_in_head(A):
        return np.array([(np.linalg.inv(H[i]) @ A[i])[:3, 3] for i in idx])

    Lh, Rh = hand_in_head(L), hand_in_head(R)
    Ld = np.linalg.norm(Lh - Lh[0], axis=1) * 1000.0
    Rd = np.linalg.norm(Rh - Rh[0], axis=1) * 1000.0

    print(f"head rotation swept:  max {head_sweep.max():.0f} deg "
          f"(end {head_sweep[-1]:.0f} deg)")
    print(f"L hand-in-head POSITION drift (mm): mean {Ld.mean():.0f}  p95 "
          f"{np.percentile(Ld, 95):.0f}  max {Ld.max():.0f}")
    print(f"R hand-in-head POSITION drift (mm): mean {Rd.mean():.0f}  p95 "
          f"{np.percentile(Rd, 95):.0f}  max {Rd.max():.0f}")

    # correlate drift with how far the head has turned -- a turn-COUPLED drift is the
    # tell for "hands move relative to head as I rotate" (vs incidental hand motion).
    if head_sweep.max() > 15:
        cL = np.corrcoef(head_sweep, Ld)[0, 1]
        cR = np.corrcoef(head_sweep, Rd)[0, 1]
        print(f"corr(head sweep, hand drift): L {cL:+.2f}  R {cR:+.2f}  "
              "(+ => drift grows with the turn)")

    drift = max(np.percentile(Ld, 95), np.percentile(Rd, 95))
    print("\nVERDICT:")
    if head_sweep.max() < 30:
        print(f"  Inconclusive -- head only swept {head_sweep.max():.0f} deg; "
              "redo with a deliberate ~80-90 deg in-place turn.")
    elif drift < 50:
        print(f"  RIGID: hands stay fixed in the head frame (p95 drift {drift:.0f} mm "
              f"< 50 mm) while the head swept {head_sweep.max():.0f} deg.")
        print("  => FIX = head-relative hand POSITION (transfer hand-rel-head onto the "
              "robot head frame); keeps planted-on-look-around + centered-on-together-turn.")
    elif drift > 150:
        print(f"  DRIFT: hands move in the head frame (p95 drift {drift:.0f} mm "
              f"> 150 mm) as the head sweeps {head_sweep.max():.0f} deg.")
        print("  => FIX = frame the hands with the base/camera (head<->hand blend); "
              "head-relative mapping would only reproduce the drift.")
    else:
        print(f"  PARTIAL: p95 drift {drift:.0f} mm over a {head_sweep.max():.0f} deg "
              "turn -- some hand drift. A head<->hand blend is the safer fix; quantify "
              "before committing to pure head-relative.")


if __name__ == "__main__":
    main()
