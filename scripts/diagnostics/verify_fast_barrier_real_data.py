"""Verify the Tier 2 self-collision barrier on recorded real-robot trajectories.

Replays the joint trajectories in ``/home/yixuan/Dexmate/data/raw_data/*.hdf5`` through
both pink's stock :class:`~pink.barriers.SelfCollisionBarrier` and the vectorized
:class:`~omniteleop.follower.fast_self_collision_barrier.FastSelfCollisionBarrier`, and
reports the per-take max residual in the barrier value ``h``, the Jacobian ``J``, and
the assembled QP rows ``(G, h)`` -- plus the closest clearance each take reaches (so we
know the contact/penetration regime was actually exercised, not just easy poses).

Run::

    PYTHONPATH=src /home/yixuan/miniforge3/envs/dexmate/bin/python \
        scripts/diagnostics/verify_fast_barrier_real_data.py
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from pink.barriers import SelfCollisionBarrier

from omniteleop.follower.fast_self_collision_barrier import FastSelfCollisionBarrier
from omniteleop.follower.whole_body_ik import (
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    TORSO_JOINTS,
    VegaWholeBodyIK,
    WBCConfig,
)

RAW_DATA_DIR = Path("/home/yixuan/Dexmate/data/raw_data")
DT = 0.01
MAX_FRAMES_PER_TAKE = 400  # cap runtime on long takes (episode_0)


def make_solver():
    """Solver + matching fast and stock barriers.

    ``ik.collision_barrier`` is itself the fast barrier after integration, so the stock
    reference is built explicitly here -- comparing fast against ``ik.collision_barrier``
    would be a no-op.
    """
    ik = VegaWholeBodyIK(WBCConfig())
    ik.reset()
    cfg = ik.config
    kw = dict(
        n_collision_pairs=ik.collision_barrier.dim,
        gain=cfg.collision_barrier_gain,
        safe_displacement_gain=cfg.collision_safe_displacement_gain,
        d_min=cfg.self_collision_safe_dist,
    )
    fast = FastSelfCollisionBarrier(ik.collision_sphere_model, **kw)
    stock = SelfCollisionBarrier(**kw)
    return ik, fast, stock


def assemble_q(ik, torso, left_arm, right_arm, head, base_pose):
    q = ik.nominal_q().copy()
    x, y, yaw = (float(v) for v in base_pose)
    q[0], q[1], q[2], q[3] = x, y, np.cos(yaw), np.sin(yaw)
    for names, vals in (
        (TORSO_JOINTS, torso), (LEFT_ARM_JOINTS, left_arm),
        (RIGHT_ARM_JOINTS, right_arm), (HEAD_JOINTS, head),
    ):
        for name, val in zip(names, vals, strict=True):
            q[ik._idx_q[name]] = float(val)  # noqa: SLF001
    return q


def main() -> int:
    files = sorted(RAW_DATA_DIR.glob("*.hdf5"))
    if not files:
        print(f"no HDF5 takes under {RAW_DATA_DIR}")
        return 1
    ik, fast, stock = make_solver()
    d_min = ik.config.self_collision_safe_dist
    print(f"d_min={d_min*100:.1f}cm  dim={ik.collision_barrier.dim}  "
          f"#pairs={len(ik.configuration.collision_model.collisionPairs)}\n")
    hdr = f"{'take':32s} {'frames':>6s} {'closest':>9s} {'max|dh|':>10s} " \
          f"{'max|dJ|':>10s} {'max|dG|':>10s} {'max|dh_qp|':>11s}"
    print(hdr)
    print("-" * len(hdr))

    g_h = g_j = g_g = g_hqp = 0.0
    g_closest = np.inf
    g_frames = 0
    for path in files:
        with h5py.File(path, "r") as f:
            if "obs/joint/torso" not in f:
                print(f"{path.name:32s}  (no obs/joint/torso, skipped)")
                continue
            torso = f["obs/joint/torso"][:]
            larm = f["obs/joint/left_arm"][:]
            rarm = f["obs/joint/right_arm"][:]
            head = f["obs/joint/head"][:]
            base = f["obs/base/pose"][:] if "obs/base/pose" in f else None
        n = torso.shape[0]
        step = max(1, n // MAX_FRAMES_PER_TAKE)
        m_h = m_j = m_g = m_hqp = 0.0
        closest = np.inf
        frames = 0
        for t in range(0, n, step):
            bp = (0.0, 0.0, 0.0) if base is None else base[t]
            q = assemble_q(ik, torso[t], larm[t], rarm[t], head[t], bp)
            ik.configuration.update(q)
            h_s = stock.compute_barrier(ik.configuration)
            h_f = fast.compute_barrier(ik.configuration)
            J_s = stock.compute_jacobian(ik.configuration)
            J_f = fast.compute_jacobian(ik.configuration)
            Gs, hs = stock.compute_qp_inequalities(ik.configuration, dt=DT)
            Gf, hf = fast.compute_qp_inequalities(ik.configuration, dt=DT)
            m_h = max(m_h, float(np.max(np.abs(h_s - h_f))))
            m_j = max(m_j, float(np.max(np.abs(J_s - J_f))))
            m_g = max(m_g, float(np.max(np.abs(Gs - Gf))))
            m_hqp = max(m_hqp, float(np.max(np.abs(hs - hf))))
            closest = min(closest, float(h_s.min()) + d_min)  # signed clearance (m)
            frames += 1
        print(f"{path.name:32s} {frames:6d} {closest*100:+8.2f}c {m_h:10.2e} "
              f"{m_j:10.2e} {m_g:10.2e} {m_hqp:11.2e}")
        g_h, g_j, g_g, g_hqp = max(g_h, m_h), max(g_j, m_j), max(g_g, m_g), max(g_hqp, m_hqp)
        g_closest = min(g_closest, closest)
        g_frames += frames

    print("-" * len(hdr))
    print(f"{'OVERALL':32s} {g_frames:6d} {g_closest*100:+8.2f}c {g_h:10.2e} "
          f"{g_j:10.2e} {g_g:10.2e} {g_hqp:11.2e}")
    ok = g_h < 1e-12 and g_j < 1e-10 and g_g < 1e-9 and g_hqp < 1e-12
    print(f"\n{'EQUIVALENT' if ok else 'MISMATCH'} "
          f"(bars: dh<1e-12, dJ<1e-10, dG<1e-9, dh_qp<1e-12)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
