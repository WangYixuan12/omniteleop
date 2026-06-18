#!/usr/bin/env python3
"""Offline lag probe: is the ZED camera pose aligned in time with the joints?

The base was parked while only the head pitched, yet zed_base_pose_node.py's
fused base x wandered ~0.1 m (see drive.csv). This script tests the user's
hypothesis — a timestamp misalignment between the camera pose and the joint
state used in FK — without any hardware, by re-fusing the *recorded* run.

Idea: the recorded T_odom_base together with the recorded joints uniquely
recovers the true camera trajectory in odom,

    odom_T_cam(t) = T_odom_base(t) · base_T_cam(q_t),

which is exactly what the ZED measured (a fixed rigid transform off the SDK
world frame). That trajectory is independent of any joint/pose lag. We can then
RE-FUSE it against joints shifted in time by tau,

    T_odom_base(t; tau) = odom_T_cam(t) · base_T_cam(q_{t+tau})^{-1},

and find the tau that minimises the parked base's residual motion. tau* != 0
(with a large error reduction) is direct evidence of pose/joint misalignment;
tau* ~ 0 means the wander comes from something else (VSLAM drift, FK/extrinsic
error, encoder noise) and the timestamps are fine.

Run:
    conda run -n yixuan_yifan python scripts/probe_zed_joint_pose_lag.py \
        --csv /home/dexmate/yixuan/Dexmate/SLAM/drive.csv
"""

from __future__ import annotations

import csv
import dataclasses
import pathlib
import sys

import numpy as np
import tyro

sys.path.insert(0, (pathlib.Path(__file__).resolve().parent.parent / "src").as_posix())

from omniteleop.follower.zed_base_pose import VegaHeadCamFK  # noqa: E402


@dataclasses.dataclass
class Args:
    csv: str = "/home/dexmate/yixuan/Dexmate/SLAM/drive.csv"
    """drive.csv written by zed_base_pose_node.py --save-csv."""

    tau_max_ms: float = 120.0
    """Scan joint-shift tau over [-tau_max, +tau_max]."""

    tau_step_ms: float = 2.0
    """Resolution of the tau scan."""

    joint_time: str = "pose"
    """Time axis the joints are placed on for interpolation:
    'pose'  -> pose_match_time_ns (joint paired to its pose's capture time), or
    'recv'  -> head_receive_time_ns (when this node received the joint)."""


def _load(path: str) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise ValueError(f"{path} has no rows")
    col = lambda k: np.array([float(r[k]) for r in rows], dtype=np.float64)
    T = np.empty((len(rows), 4, 4))
    for i, r in enumerate(rows):
        for a in range(4):
            for b in range(4):
                T[i, a, b] = float(r[f"T{a}{b}"])
    return {
        "pose_t": col("pose_match_time_ns") / 1e9,
        "head_recv_t": col("head_receive_time_ns") / 1e9,
        "torso_recv_t": col("torso_receive_time_ns") / 1e9,
        "head": np.column_stack([col("head_j1"), col("head_j2"), col("head_j3")]),
        "torso": np.column_stack([col("torso_j1"), col("torso_j2"), col("torso_j3")]),
        "x": col("x"),
        "y": col("y"),
        "z": col("z"),
        "T": T,
    }


def _interp_joints(t_query: np.ndarray, t_src: np.ndarray, q_src: np.ndarray) -> np.ndarray:
    """Per-DOF linear interpolation of joints sampled at (possibly unsorted) t_src."""
    order = np.argsort(t_src)
    ts, qs = t_src[order], q_src[order]
    return np.column_stack([np.interp(t_query, ts, qs[:, k]) for k in range(qs.shape[1])])


def main() -> None:
    args = tyro.cli(Args)
    d = _load(args.csv)
    fk = VegaHeadCamFK()
    n = len(d["pose_t"])
    print(f"loaded {n} rows from {args.csv}")
    print(f"camera frame: {fk.camera_frame}")

    # joint time axis for interpolation
    jt = d["pose_t"] if args.joint_time == "pose" else d["head_recv_t"]

    # true camera trajectory in odom (lag-independent): odom_T_cam = T_odom_base @ base_T_cam(q)
    base_T_cam = np.array([fk.base_T_cam(d["torso"][i], d["head"][i]) for i in range(n)])
    odom_T_cam = d["T"] @ base_T_cam
    cam_p = odom_T_cam[:, :3, 3]
    step = np.linalg.norm(np.diff(cam_p, axis=0), axis=1)
    print(
        f"camera path in odom: span {np.round(np.ptp(cam_p, 0), 4)} m, "
        f"length {step.sum():.3f} m, step median/max "
        f"{np.median(step) * 1e3:.1f}/{step.max() * 1e3:.1f} mm"
    )

    # baseline (recorded) parked-base residual
    base_p = d["T"][:, :3, 3]
    base_xy0 = base_p[:, :2]
    rms0 = np.sqrt(np.mean(np.sum((base_xy0 - base_xy0.mean(0)) ** 2, axis=1)))
    print(
        f"\nrecorded base residual (parked): "
        f"x[{d['x'].min():+.3f},{d['x'].max():+.3f}] "
        f"y[{d['y'].min():+.3f},{d['y'].max():+.3f}] m, "
        f"xy-RMS {rms0 * 1e3:.1f} mm"
    )

    # re-fuse with joints shifted by tau; minimise parked-base spread
    taus = np.arange(-args.tau_max_ms, args.tau_max_ms + 1e-9, args.tau_step_ms) / 1e3
    rms = np.empty_like(taus)
    spanx = np.empty_like(taus)
    for j, tau in enumerate(taus):
        q_sh = _interp_joints(d["pose_t"] + tau, jt, d["head"])
        bTc = np.array([fk.base_T_cam(d["torso"][i], q_sh[i]) for i in range(n)])
        odom_T_base = odom_T_cam @ np.linalg.inv(bTc)
        # re-anchor so odom = base @ sample 0 (matches the node's convention)
        anchor = odom_T_base[0]
        rel = np.linalg.inv(anchor)[None] @ odom_T_base
        xy = rel[:, :2, 3]
        rms[j] = np.sqrt(np.mean(np.sum((xy - xy.mean(0)) ** 2, axis=1)))
        spanx[j] = np.ptp(xy[:, 0])

    best = int(np.argmin(rms))
    print(f"\njoint interp axis: {args.joint_time}  (tau>0 = use LATER joints)")
    print(f"{'tau_ms':>8} {'xy_RMS_mm':>10} {'x_span_mm':>10}")
    # print a coarse slice of the curve plus the minimum
    show = sorted(set(list(range(0, len(taus), max(1, len(taus) // 24))) + [best]))
    for j in show:
        mark = "  <== min" if j == best else ""
        print(f"{taus[j] * 1e3:8.0f} {rms[j] * 1e3:10.1f} {spanx[j] * 1e3:10.1f}{mark}")

    print(
        f"\nBEST tau = {taus[best] * 1e3:+.0f} ms: xy-RMS "
        f"{rms[best] * 1e3:.1f} mm  (recorded {rms0 * 1e3:.1f} mm, "
        f"reduction {100 * (1 - rms[best] / rms0):.0f}%)"
    )
    if abs(taus[best] * 1e3) < args.tau_step_ms * 1.5:
        print(
            "=> tau* ~ 0: a constant pose/joint time offset does NOT explain the "
            "wander. Look elsewhere (VSLAM drift, FK/extrinsic, encoder)."
        )
    else:
        print(
            f"=> tau* != 0 and error drops {100 * (1 - rms[best] / rms0):.0f}%: the "
            "joints used in FK are misaligned with the camera pose by ~"
            f"{taus[best] * 1e3:+.0f} ms."
        )


if __name__ == "__main__":
    main()
