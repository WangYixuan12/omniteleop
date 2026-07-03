#!/usr/bin/env python3
"""Empirically test how tracking (vs freeing) head-z trades camera height for reach.

In ``head_mode: "ik"`` the head is tracked by a whole-body objective. The gravity-aligned
``head_world_position_cost [x, y, z]`` (world axes) decides what the camera does when the
arms reach high: the arms and head both hang off the torso top, so a HIGH EE target needs
torso extension, which also raises the head. If world-z is tracked (z > 0) the camera is
pinned to the operator's fixed standing height and the torso will not over-extend -- steady
camera, capped reach. If world-z is free (z = 0) the torso straightens and the camera rides
up -- more reach, moving camera.

This harness grounds the comparison in a real arms-lifted take (``life_arm.hdf5``):

  1. reconstruct, at the highest-reach frame, the base-relative L/R EE poses and the
     head (``zed_depth_frame``) pose via FK from the recorded joints;
  2. sweep the two EE targets UPWARD in world-z by ``dz`` while holding the head target
     FIXED (the operator cannot command head height);
  3. for each candidate config, seed the recorded arms-up state, run the whole-body IK to
     convergence, and record EE-position error, head-aim error, camera world-z, torso,
     base drift, and the CoM stability margin.

Run in the dexmate conda env::

    /home/yixuan/miniforge3/envs/dexmate/bin/python \
        scripts/diagnostics/verify_head_z_reach.py \
        --hdf5 /home/yixuan/Dexmate/data/raw_data/life_arm.hdf5
"""

from __future__ import annotations

import argparse
from typing import Callable

import h5py
import numpy as np
import pinocchio as pin

from omniteleop.follower.whole_body_ik import (
    HEAD_FRAME,
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    LEFT_EE_FRAME,
    RIGHT_ARM_JOINTS,
    RIGHT_EE_FRAME,
    TORSO_JOINTS,
    VegaWholeBodyIK,
    WBCConfig,
)

DT = 0.01
N_ITERS = 600
_GROUP_JOINTS = {
    "torso": list(TORSO_JOINTS),
    "left_arm": list(LEFT_ARM_JOINTS),
    "right_arm": list(RIGHT_ARM_JOINTS),
    "head": list(HEAD_JOINTS),
}


def _load_recorded_joints(path: str, source: str) -> dict[str, np.ndarray]:
    """Per-group joint trajectories ``(T, n)`` from an EpisodeRecorder HDF5."""
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for grp, names in _GROUP_JOINTS.items():
            key = f"{source}/joint/{grp}"
            if key not in f:
                raise KeyError(f"{path} has no dataset {key!r}")
            arr = np.asarray(f[key], dtype=float)
            if arr.ndim != 2 or arr.shape[1] != len(names):
                raise ValueError(
                    f"{key} has shape {arr.shape}, expected (T, {len(names)})"
                )
            out[grp] = arr
    lengths = {grp: a.shape[0] for grp, a in out.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"group trajectories have mismatched lengths: {lengths}")
    return out


def _config_from_joints(ik: VegaWholeBodyIK, frame_joints: dict[str, np.ndarray]) -> np.ndarray:
    """Full model ``q`` (planar base at origin) from one frame's per-group joints."""
    q = ik.nominal_q()
    for grp, names in _GROUP_JOINTS.items():
        vals = np.asarray(frame_joints[grp], dtype=float)
        if vals.shape != (len(names),):
            raise ValueError(f"{grp} frame joints shape {vals.shape} != ({len(names)},)")
        for name, val in zip(names, vals, strict=True):
            q[ik._idx_q[name]] = float(val)  # noqa: SLF001
    return q


def _pick_highest_reach_frame(ik: VegaWholeBodyIK, joints: dict[str, np.ndarray]) -> int:
    """Index of the frame whose mean L/R EE world-z (base frame) is greatest."""
    n = joints["torso"].shape[0]
    mean_ee_z = np.full(n, -np.inf)
    for t in range(n):
        q = _config_from_joints(ik, {g: joints[g][t] for g in _GROUP_JOINTS})
        lz = ik.frame_pose(LEFT_EE_FRAME, q).translation[2]
        rz = ik.frame_pose(RIGHT_EE_FRAME, q).translation[2]
        mean_ee_z[t] = 0.5 * (lz + rz)
    return int(np.argmax(mean_ee_z))


def _converge(
    ik: VegaWholeBodyIK, left: pin.SE3, right: pin.SE3, head: pin.SE3, n: int
):
    result = None
    for _ in range(n):
        result = ik.solve(left, right, DT, head_target=head)
    return result


def _make_configs() -> dict[str, Callable[[], WBCConfig]]:
    """Named WBC config variants to A/B compare (all head_mode "ik").

    ``z-tracked``: the current wbik.yaml default (``WBCConfig()``) --
        ``head_world_position_cost [10000, 10000, 10000]``, so the camera HEIGHT tracks the
        operator head target (world-frame). Pins camera height; caps vertical reach.
    ``z-free``: world x/y tracked, world-z freed ([10000, 10000, 0]) -- the torso may
        straighten to a high EE target, camera rides up with it.
    ``z200``: world x/y tracked, world-z a weak comfort prior ([10000, 10000, 200]) --
        gently anchors camera height but still yields to a genuine high reach.
    """
    return {
        "z-tracked world (default)": lambda: WBCConfig(),
        "z-free (world xy only)": lambda: WBCConfig(
            head_position_cost=0.0, head_world_position_cost=(10000.0, 10000.0, 0.0)
        ),
        "z200 comfort prior": lambda: WBCConfig(
            head_position_cost=0.0, head_world_position_cost=(10000.0, 10000.0, 200.0)
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--hdf5", default="/home/yixuan/Dexmate/data/raw_data/lift_arm.hdf5")
    ap.add_argument("--source", choices=("action", "obs"), default="action",
                    help="which recorded joint stream to reconstruct targets from "
                         "(action=WBC command, obs=measured). Default action.")
    ap.add_argument("--frame", type=int, default=-1,
                    help="frame index to anchor targets on; -1 = auto (highest reach).")
    ap.add_argument("--dz-max", type=float, default=0.40,
                    help="max upward EE-target sweep (m) above the anchor pose.")
    ap.add_argument("--dz-step", type=float, default=0.05)
    ap.add_argument("--iters", type=int, default=N_ITERS)
    ap.add_argument("--out", default=None, help="optional PNG path for the summary plot.")
    args = ap.parse_args()

    joints = _load_recorded_joints(args.hdf5, args.source)
    n_frames = joints["torso"].shape[0]
    probe = VegaWholeBodyIK(WBCConfig())
    frame = args.frame if args.frame >= 0 else _pick_highest_reach_frame(probe, joints)
    if not 0 <= frame < n_frames:
        raise ValueError(f"--frame {frame} out of range [0, {n_frames})")
    frame_joints = {g: joints[g][frame] for g in _GROUP_JOINTS}

    # Base-relative anchor targets via FK on the recorded config.
    q_anchor = _config_from_joints(probe, frame_joints)
    left0 = probe.frame_pose(LEFT_EE_FRAME, q_anchor)
    right0 = probe.frame_pose(RIGHT_EE_FRAME, q_anchor)
    head0 = probe.frame_pose(HEAD_FRAME, q_anchor)
    head0_rot = head0.rotation
    print(f"[verify] {args.hdf5}")
    print(f"[verify] source={args.source}  frames={n_frames}  anchor frame={frame}")
    print(f"[verify] anchor EE z: L={left0.translation[2]:.3f} R={right0.translation[2]:.3f} m"
          f"   head z={head0.translation[2]:.3f} m  (head target HELD fixed)")
    print(f"[verify] torso(anchor)={np.round(frame_joints['torso'], 3)}\n")

    dzs = np.arange(0.0, args.dz_max + 1e-9, args.dz_step)
    configs = _make_configs()
    results: dict[str, list[dict]] = {name: [] for name in configs}

    for name, make in configs.items():
        ik = VegaWholeBodyIK(make())
        print(f"=== config: {name} ===")
        print(f"{'dz(m)':>6} {'L_err':>7} {'R_err':>7} {'hOri':>6} {'headZ':>7} {'hZerr':>7} "
              f"{'torso(j1,j2,j3)':>22} {'baseXY':>7} {'stabM':>6} {'ok/held':>8}")
        for dz in dzs:
            up = np.array([0.0, 0.0, float(dz)])
            left = pin.SE3(left0.rotation, left0.translation + up)
            right = pin.SE3(right0.rotation, right0.translation + up)
            # Seed from the RECORDED arms-up state (not nominal): tests "continue reaching
            # upward from this take frame" rather than steady-state convergence from home.
            # _update_from_measured re-seeds q without touching the nominal posture target.
            ik.reset()
            ik._update_from_measured(q_anchor)  # noqa: SLF001
            res = _converge(ik, left, right, head0, args.iters)
            head_fk = ik.frame_pose(HEAD_FRAME)
            head_z = float(head_fk.translation[2])
            head_z_err = head_z - float(head0.translation[2])
            # Camera-aim error: geodesic angle current-vs-target head rotation (deg). Confirms
            # orientation still tracks when head_position_cost=0 (FrameTask is orient-only).
            head_ori_err = float(np.degrees(
                np.linalg.norm(pin.log3(head_fk.rotation.T @ head0_rot))
            ))
            base_xy = float(np.linalg.norm(res.base_pose[:2]))
            ok = bool(res.success and not res.held)
            row = {
                "dz": float(dz),
                "left_err": float(res.left_ee_error),
                "right_err": float(res.right_ee_error),
                "head_ori_err": head_ori_err,
                "head_z": head_z,
                "head_z_err": head_z_err,
                "torso": np.asarray(res.torso, dtype=float),
                "base_xy": base_xy,
                "stability_margin": float(res.stability_margin),
                "safety_status": str(res.safety_status),
                "success": bool(res.success),
                "held": bool(res.held),
                "ok": ok,
            }
            results[name].append(row)
            flag = "" if row["safety_status"].lower() == "ok" else f" {row['safety_status']}"
            print(f"{dz:6.2f} {res.left_ee_error*1000:5.1f}mm {res.right_ee_error*1000:5.1f}mm "
                  f"{head_ori_err:5.1f} {head_z:7.3f} {head_z_err*1000:+5.0f}mm "
                  f"{str(np.round(row['torso'],3)):>22} {base_xy:7.3f} {row['stability_margin']:6.3f} "
                  f"{str(res.success)[0]}/{str(res.held)[0]}{flag}")
        print()

    # Headline: reach gain = highest dz each config holds within an EE-error budget AND a
    # VALID solve (converged, not safety-held) -- a held/gated step is not a reachable point.
    budget = 0.02
    print(f"=== reach within {budget*1000:.0f} mm EE-error budget AND success&!held (max dz) ===")
    for name in configs:
        good = [r["dz"] for r in results[name]
                if r["ok"] and max(r["left_err"], r["right_err"]) <= budget]
        best = max(good) if good else float("nan")
        print(f"  {name:42s}  dz_max = {best:.2f} m")

    if args.out:
        _plot(results, dzs, args.out)
        print(f"\n[verify] wrote {args.out}")


def _plot(results: dict[str, list[dict]], dzs: np.ndarray, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, rows in results.items():
        ee = [max(r["left_err"], r["right_err"]) * 1000 for r in rows]
        hz = [r["head_z"] for r in rows]
        tj = [r["torso"][1] for r in rows]  # torso_j2 (main extension DOF)
        axes[0].plot(dzs, ee, "-o", label=name)
        axes[1].plot(dzs, hz, "-o", label=name)
        axes[2].plot(dzs, tj, "-o", label=name)
    axes[0].axhline(20.0, ls="--", c="k", lw=0.8, label="20 mm budget")
    axes[0].set(xlabel="EE target raised dz (m)", ylabel="max EE pos error (mm)",
                title="Reach error vs. commanded height")
    axes[1].set(xlabel="EE target raised dz (m)", ylabel="achieved camera world-z (m)",
                title="Camera height")
    axes[2].set(xlabel="EE target raised dz (m)", ylabel="torso_j2 (rad)",
                title="Torso extension DOF")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    main()
