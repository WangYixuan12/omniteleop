#!/usr/bin/env python3
"""Attribute the parked-base wobble to the FK chain vs the ZED pose tracking.

T_odom_base = odom_T_track · track_T_cam(ZED) · base_T_cam(q)^{-1}, with
odom_T_track constant. base_T_cam(q) is an EXACT, noise-free function of the
joints (Pinocchio). So the parked-base orientation wobble can be split by
*repeatability*:

  * A wobble that is a repeatable function of head pose q  ->  FK chain
    (wrong camera extrinsic / joint axis / encoder mapping). Visiting the same
    q always gives the same error; no hysteresis with velocity.
  * A wobble that is NOT repeatable -- different each time the same q is
    revisited, or split by motion direction (hysteresis), or slowly drifting --
    can only come from track_T_cam(ZED), the only other time-varying input.

Two model-free tests:
  1. Revisit test: for sample pairs at the SAME head pose (|Δq|<q_tol) but far
     apart in time (|Δt|>t_gap, independent visits), how different is the base
     orientation error? RMS(Δerror) near sqrt(2)*RMS(error) => non-repeatable
     (ZED). Near 0 => repeatable (FK). A same-pose & same-time control bounds
     the joint-quantisation floor.
  2. Hysteresis plot: base orientation error vs head pitch, coloured by the sign
     of the pitch velocity. A single shared curve => FK; split loops => ZED.

Run:
    conda run -n yixuan_yifan python scripts/probe_zed_fk_vs_tracking.py
"""

from __future__ import annotations

import csv
import dataclasses
import pathlib
import sys

import numpy as np
import tyro

sys.path.insert(0, (pathlib.Path(__file__).resolve().parent.parent / "src").as_posix())


@dataclasses.dataclass
class Args:
    csv: str = "/home/dexmate/yixuan/Dexmate/SLAM/drive.csv"
    out_png: str = "/home/dexmate/yixuan/Dexmate/SLAM/fk_vs_tracking.png"
    q_tol_deg: float = 2.0
    """Two samples count as the 'same head pose' within this max per-joint angle."""
    t_gap_s: float = 1.0
    """Pairs must be at least this far apart in time to be independent revisits."""


def _load(path: str):
    rows = list(csv.DictReader(open(path)))
    col = lambda k: np.array([float(r[k]) for r in rows], dtype=np.float64)
    t = col("pose_match_time_ns") / 1e9
    t -= t[0]
    head = np.column_stack([col("head_j1"), col("head_j2"), col("head_j3")])
    # base orientation error as a small-angle vector (deg), about its own median
    rpy = np.column_stack([col("roll"), col("pitch"), col("yaw")])
    e = np.degrees(rpy - np.median(rpy, axis=0))
    return t, head, e, col("x"), col("y")


def _revisit_test(t, head_deg, e, q_tol, t_gap):
    """RMS base-orientation-error difference for same-pose pairs, by time gap."""
    n = len(t)
    emag = np.linalg.norm(e, axis=1)
    base_rms = emag.std()  # spread of the error itself
    same_far, same_near = [], []
    for i in range(n):
        dq = np.max(np.abs(head_deg - head_deg[i]), axis=1)
        dt = np.abs(t - t[i])
        de = np.linalg.norm(e - e[i], axis=1)
        same = dq < q_tol
        far = same & (dt > t_gap)
        near = same & (dt < 0.12) & (dt > 0)  # adjacent frames, same pose: floor
        same_far.extend(de[far].tolist())
        same_near.extend(de[near].tolist())
    same_far = np.asarray(same_far)
    same_near = np.asarray(same_near)
    return base_rms, same_far, same_near


def main() -> None:
    args = tyro.cli(Args)
    t, head, e, x, y = _load(args.csv)
    head_deg = np.degrees(head)
    emag = np.linalg.norm(e, axis=1)

    print(f"loaded {len(t)} rows; base orientation error rms {emag.std():.3f} deg, "
          f"max {emag.max():.3f} deg")

    base_rms, far, near = _revisit_test(t, head_deg, e, args.q_tol_deg, args.t_gap_s)
    # if the error were a pure function of pose, revisiting the same pose at a
    # different time would reproduce it: Δerror -> 0. If independent of pose,
    # Δerror -> sqrt(2)*rms. Report the fraction of variance that is NON-repeatable.
    nonrep = (np.mean(far ** 2) / (2 * base_rms ** 2)) if base_rms > 0 else float("nan")
    print(
        f"\nREVISIT TEST  (same head pose within {args.q_tol_deg:.0f}°):\n"
        f"  same pose & adjacent frame  (floor) : RMS Δerr "
        f"{np.sqrt(np.mean(near ** 2)):.3f} deg  (n={len(near)})\n"
        f"  same pose & >{args.t_gap_s:.0f}s apart (revisit): RMS Δerr "
        f"{np.sqrt(np.mean(far ** 2)):.3f} deg  (n={len(far)})\n"
        f"  error spread (rms)                  : {base_rms:.3f} deg\n"
        f"  => non-repeatable fraction of variance: {100 * nonrep:.0f}%  "
        f"(100% = all ZED tracking, 0% = all repeatable/FK)"
    )

    # velocity / hysteresis quantification
    dt = np.gradient(t)
    v3 = np.gradient(head_deg[:, 2]) / dt  # head pitch velocity [deg/s]
    # signed correlation of each base-error component with signed pitch velocity
    sgn = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    print(
        "\nVELOCITY (hysteresis) — signed corr(base err comp, head pitch vel):\n"
        f"  roll {sgn(e[:,0], v3):+.2f}   pitch {sgn(e[:,1], v3):+.2f}   "
        f"yaw {sgn(e[:,2], v3):+.2f}"
    )

    # variance explained by ANGLE (FK candidate) vs by VELOCITY (ZED-dynamic)
    def r2(target, feats):
        A = np.column_stack([np.ones_like(target)] + feats)
        coef, *_ = np.linalg.lstsq(A, target, rcond=None)
        resid = target - A @ coef
        return 1 - np.var(resid) / np.var(target)
    j1, j2, j3 = head_deg.T
    ang = [j1, j3, j1 ** 2, j3 ** 2, j1 * j3, np.sin(np.radians(j3)), np.cos(np.radians(j3))]
    vel = [v3, np.abs(v3), np.gradient(j1) / dt]
    print(
        "\nVARIANCE of |base orientation error| explained by:\n"
        f"  head ANGLE (static FK)      R² = {r2(emag, ang):.2f}\n"
        f"  head VELOCITY (ZED-dynamic) R² = {r2(emag, vel):.2f}\n"
        f"  both                        R² = {r2(emag, ang + vel):.2f}"
    )

    # plot: error vs pitch coloured by velocity sign (hysteresis)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    up = v3 > 5
    dn = v3 < -5
    ax[0].scatter(head_deg[up, 2], e[up, 1], s=10, c="tab:red", alpha=.5, label="pitch ↑")
    ax[0].scatter(head_deg[dn, 2], e[dn, 1], s=10, c="tab:blue", alpha=.5, label="pitch ↓")
    ax[0].set_xlabel("head_j3 (pitch) [deg]")
    ax[0].set_ylabel("base PITCH error [deg]")
    ax[0].set_title("base error vs head pose, by motion direction\n"
                    "(one shared curve = FK; split loops / cloud = ZED)")
    ax[0].legend(); ax[0].grid(alpha=.3)

    sc = ax[1].scatter(head_deg[:, 2], emag, s=10, c=t, cmap="viridis")
    ax[1].set_xlabel("head_j3 (pitch) [deg]")
    ax[1].set_ylabel("|base orientation error| [deg]")
    ax[1].set_title("|base error| vs head pose, coloured by time\n"
                    "(repeatable single curve = FK; scatter = ZED/drift)")
    ax[1].grid(alpha=.3)
    fig.colorbar(sc, ax=ax[1], label="t [s]")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=90)
    print(f"\nsaved {args.out_png}")


if __name__ == "__main__":
    main()
