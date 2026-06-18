"""Two-part verification:

1. FK PARITY — compare FK computed by ``verify_action_obs_eef_lag.FKHelper``
   (the same code path as ``port_dexmate_hdf5.FKHelper``) against the 9-D
   pos6d state stored under ``/home/yixuan/Dexmate/data/processed_data``.
   Confirms the verify-script's FK is bit-equivalent to the port script's.

2. RESIDUAL ANALYSIS — after shifting by the per-episode best lag k, check
   whether ``obs_eef[t+k] == action_eef[t]`` exactly. Quantify the residual
   and break down its likely causes (constant offset, per-frame jitter,
   correlation with EEF speed → would indicate residual phase error).

Episode mapping: the port script enumerates ``sorted(raw_dir.glob('*.hdf5'))``
which is **lexicographic** ordering, so processed episode_index 0/1/2/... maps
to raw ``episode_0, episode_1, episode_10, episode_11, ..., episode_19,
episode_2, episode_20, ...``. We sort the same way here.

Usage:
  conda run -n dexmate_lerobot python scripts/analysis/verify_fk_and_residual.py \
      --raw-split-dir /home/yixuan/Dexmate/data/raw_data_renamed/train \
      --processed-variant /home/yixuan/Dexmate/data/processed_data/train/dexmate_eef_eef \
      --max-episodes 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from verify_action_obs_eef_lag import FKHelper, se3_errors, sweep_lag


def mat_to_pos6d(T: np.ndarray) -> np.ndarray:
    """(N, 4, 4) -> (N, 9): [tx, ty, tz, R[:,0], R[:,1]]."""
    pos = T[:, :3, 3]
    c0 = T[:, :3, 0]
    c1 = T[:, :3, 1]
    return np.concatenate([pos, c0, c1], axis=1)


def pos6d_to_mat(p: np.ndarray) -> np.ndarray:
    """(N, 9) -> (N, 4, 4) via Gram-Schmidt on the two columns."""
    pos = p[:, :3]
    c0 = p[:, 3:6]
    c1 = p[:, 6:9]
    c0 = c0 / np.linalg.norm(c0, axis=1, keepdims=True)
    c1 = c1 - (c1 * c0).sum(1, keepdims=True) * c0
    c1 = c1 / np.linalg.norm(c1, axis=1, keepdims=True)
    c2 = np.cross(c0, c1)
    R = np.stack([c0, c1, c2], axis=2)
    out = np.zeros((p.shape[0], 4, 4), dtype=p.dtype)
    out[:, :3, :3] = R
    out[:, :3, 3] = pos
    out[:, 3, 3] = 1.0
    return out


def load_processed_episode_state(parquet_path: Path, ep_idx: int) -> np.ndarray:
    df = pd.read_parquet(parquet_path)
    sub = df[df["episode_index"] == ep_idx].sort_values("frame_index")
    return np.stack([np.asarray(x, dtype=np.float64) for x in sub["observation.state"].to_numpy()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-split-dir", type=Path,
                    default=Path("/home/yixuan/Dexmate/data/raw_data_renamed/train"),
                    help="Raw HDF5 dir for ONE split (matches the processed variant's split).")
    ap.add_argument("--processed-variant", type=Path,
                    default=Path("/home/yixuan/Dexmate/data/processed_data/train/dexmate_eef_eef"))
    ap.add_argument("--max-episodes", type=int, default=5)
    ap.add_argument("--max-lag", type=int, default=15)
    args = ap.parse_args()

    parquet = args.processed_variant / "data" / "chunk-000" / "file-000.parquet"
    if not parquet.exists():
        raise SystemExit(f"Missing: {parquet}")

    # Match port_dexmate_hdf5.py: lexicographic sort by stem.
    raw_files = sorted(args.raw_split_dir.glob("episode_*.hdf5"))[: args.max_episodes]
    if not raw_files:
        raise SystemExit(f"No episode_*.hdf5 under {args.raw_split_dir}")

    fk = FKHelper()

    # ---------------- Part 1: FK parity ----------------
    print("=" * 78)
    print("PART 1: FK parity (this script's FK  vs  processed state)")
    print("=" * 78)
    print(f"{'episode':<12}{'T':>5}{'pos_max(mm)':>14}{'pos_mean(mm)':>14}{'rot_max(deg)':>14}{'rot_mean(deg)':>15}")
    print("-" * 74)
    parity_pos_max = []
    parity_rot_max = []
    for ep_idx, f in enumerate(raw_files):
        with h5py.File(f, "r") as h:
            torso = h["obs/joint/torso"][:].astype(np.float64)
            right_arm = h["obs/joint/right_arm"][:].astype(np.float64)
        my_eef = fk.fk_batch(torso, right_arm)             # (T, 4, 4)
        proc_state = load_processed_episode_state(parquet, ep_idx)  # (T, 10)
        if proc_state.shape[0] != my_eef.shape[0]:
            print(f"{f.stem:<12}LEN MISMATCH raw={my_eef.shape[0]} proc={proc_state.shape[0]} — episode mapping wrong")
            continue
        proc_eef = pos6d_to_mat(proc_state[:, :9])
        te, re = se3_errors(my_eef, proc_eef)
        parity_pos_max.append(te.max())
        parity_rot_max.append(re.max())
        print(f"{f.stem:<12}{my_eef.shape[0]:>5}"
              f"{te.max()*1000:>14.6f}{te.mean()*1000:>14.6f}"
              f"{re.max():>14.6f}{re.mean():>15.6f}")
    if not parity_pos_max:
        print("(no episodes matched in length — check raw split / sort order)")
    else:
        print(f"\nOverall max:  pos={max(parity_pos_max)*1000:.6f} mm   rot={max(parity_rot_max):.6f} deg")
        if max(parity_pos_max) < 1e-4 and max(parity_rot_max) < 0.05:
            print("=> FK is numerically equivalent (residual is float32 round-off "
                  "from parquet storage + Gram-Schmidt re-orthonormalization).")
        else:
            print("=> FK MISMATCH — investigate URDF / link / joint mapping.")

    # ---------------- Part 2: Residual after best lag ----------------
    print("\n" + "=" * 78)
    print("PART 2: Residual obs_eef[t+k] vs action_eef[t] after best lag")
    print("=" * 78)
    for ep_idx, f in enumerate(raw_files):
        with h5py.File(f, "r") as h:
            action_eef = h["action/eef/right"][:].astype(np.float64)
            torso = h["obs/joint/torso"][:].astype(np.float64)
            right_arm = h["obs/joint/right_arm"][:].astype(np.float64)
            ts = h["timestamp_ns"][:]
        T = action_eef.shape[0]
        if T < 2 * args.max_lag + 4:
            continue
        obs_eef = fk.fk_batch(torso, right_arm)
        lags, te_sweep, re_sweep = sweep_lag(action_eef, obs_eef, args.max_lag)
        cost = te_sweep * 1000.0 + re_sweep
        i_best = int(np.argmin(cost))
        k = int(lags[i_best])
        # Aligned slices
        lo = max(0, -k)
        hi = min(T, T - k)
        a = action_eef[lo:hi]
        b = obs_eef[lo + k : hi + k]
        te, re = se3_errors(a, b)

        # Speed of action EEF (m/s and deg/s) at the aligned frames using raw timestamps.
        dt = np.diff(ts.astype(np.float64)) / 1e9          # (T-1,)
        # mid-point per-frame speed for action
        vel_pos = np.linalg.norm(np.diff(a[:, :3, 3], axis=0), axis=1) / dt[lo : hi - 1]
        # rot speed via geodesic
        Rrel = np.einsum("nij,njk->nik",
                         np.transpose(a[:-1, :3, :3], (0, 2, 1)), a[1:, :3, :3])
        cos = np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1) / 2, -1, 1)
        vel_rot = np.degrees(np.arccos(cos)) / dt[lo : hi - 1]
        # average residual at frames i and i+1 to align dims
        te_mid = 0.5 * (te[:-1] + te[1:])
        re_mid = 0.5 * (re[:-1] + re[1:])
        # Pearson correlation of residual vs action speed
        def corr(x, y):
            x = (x - x.mean()) / (x.std() + 1e-12)
            y = (y - y.mean()) / (y.std() + 1e-12)
            return float((x * y).mean())
        rho_t = corr(te_mid, vel_pos)
        rho_r = corr(re_mid, vel_rot)
        # Constant bias (mean offset) vs random jitter
        bias_pos = a[:, :3, 3].mean(0) - b[:, :3, 3].mean(0)
        bias_pos_norm = float(np.linalg.norm(bias_pos))

        print(
            f"\n{f.stem}  T={T}  best_lag={k:+d}\n"
            f"  trans residual: mean={te.mean()*1000:7.3f} mm  "
            f"median={np.median(te)*1000:7.3f}  std={te.std()*1000:7.3f}  "
            f"p95={np.percentile(te,95)*1000:7.3f}  max={te.max()*1000:7.3f}\n"
            f"  rot   residual: mean={re.mean():7.4f} deg  "
            f"median={np.median(re):7.4f}  std={re.std():7.4f}  "
            f"p95={np.percentile(re,95):7.4f}  max={re.max():7.4f}\n"
            f"  fraction of frames with trans err < 1 mm : "
            f"{(te < 1e-3).mean()*100:5.2f}%\n"
            f"  fraction of frames with rot   err < 0.1 deg: "
            f"{(re < 0.1).mean()*100:5.2f}%\n"
            f"  constant translational bias |mean(a-b)| = {bias_pos_norm*1000:.3f} mm\n"
            f"  corr(trans residual, action lin speed) = {rho_t:+.3f}   "
            f"(action speed mean {vel_pos.mean():.3f} m/s)\n"
            f"  corr(rot   residual, action rot speed) = {rho_r:+.3f}   "
            f"(action speed mean {vel_rot.mean():.2f} deg/s)"
        )


if __name__ == "__main__":
    main()
