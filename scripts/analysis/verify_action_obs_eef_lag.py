"""Verify whether /action/eef/right (leader IK target) matches FK-derived
right-arm EEF (FK on /obs/joint/{torso, right_arm}) up to a consistent
temporal offset.

For each episode we:
  1. Load right-arm action eef (T, 4, 4) and obs joints (torso, right_arm).
  2. Compute FK(torso[t], right_arm[t]) -> obs_eef[t] in (T, 4, 4).
  3. For lag k in [-K, K]:  compare action_eef[t] vs obs_eef[t + k]
     over the valid overlap, reporting mean translation error (m) and
     mean rotation error (deg).
  4. Pick the k with minimum (translation + rotation) cost as the per-
     episode best lag. Aggregate across episodes to check consistency.

Usage:
  conda run -n dexmate_lerobot python scripts/analysis/verify_action_obs_eef_lag.py \
      --raw-dir /home/yixuan/Dexmate/data/raw_data \
      --max-episodes 20 --max-lag 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


_ROBOT_NAME = "vega_no_effector"
_RIGHT_ARM_EEF_LINK = "R_ee"
_TORSO_JOINTS = ("torso_j1", "torso_j2", "torso_j3")
_RIGHT_ARM_JOINTS = tuple(f"R_arm_j{i}" for i in range(1, 8))


class FKHelper:
    def __init__(self) -> None:
        from yixuan_utilities.kinematics_helper import KinHelper

        self.kin = KinHelper(_ROBOT_NAME)
        self._eef_idx = self.kin.link_name_to_idx[_RIGHT_ARM_EEF_LINK]
        active = self.kin.sapien_robot.get_active_joints()
        self._joint_idx = {j.name: i for i, j in enumerate(active)}
        self._dof = self.kin.sapien_robot.dof

    def fk_batch(self, torso: np.ndarray, right_arm: np.ndarray) -> np.ndarray:
        T = torso.shape[0]
        out = np.zeros((T, 4, 4), dtype=np.float64)
        qpos = np.zeros(self._dof, dtype=np.float64)
        for t in range(T):
            for i, n in enumerate(_TORSO_JOINTS):
                qpos[self._joint_idx[n]] = torso[t, i]
            for i, n in enumerate(_RIGHT_ARM_JOINTS):
                qpos[self._joint_idx[n]] = right_arm[t, i]
            out[t] = self.kin.compute_fk_from_link_idx(qpos, [self._eef_idx])[0]
        return out


def se3_errors(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame translation (m) and rotation (deg) error between two (N,4,4) arrays."""
    t_err = np.linalg.norm(A[:, :3, 3] - B[:, :3, 3], axis=1)
    R_rel = np.einsum("nij,njk->nik", np.transpose(A[:, :3, :3], (0, 2, 1)), B[:, :3, :3])
    cos = (np.trace(R_rel, axis1=1, axis2=2) - 1.0) / 2.0
    cos = np.clip(cos, -1.0, 1.0)
    rot_err = np.degrees(np.arccos(cos))
    return t_err, rot_err


def sweep_lag(
    action_eef: np.ndarray,
    obs_eef: np.ndarray,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (lags, mean_trans_err_m, mean_rot_err_deg)."""
    T = action_eef.shape[0]
    lags = np.arange(-max_lag, max_lag + 1)
    trans = np.empty(len(lags))
    rot = np.empty(len(lags))
    for i, k in enumerate(lags):
        # action[t] vs obs[t + k]; valid t in [max(0,-k), min(T, T-k))
        lo = max(0, -k)
        hi = min(T, T - k)
        a = action_eef[lo:hi]
        b = obs_eef[lo + k : hi + k]
        te, re = se3_errors(a, b)
        trans[i] = te.mean()
        rot[i] = re.mean()
    return lags, trans, rot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("/home/yixuan/Dexmate/data/raw_data"))
    ap.add_argument("--max-episodes", type=int, default=20)
    ap.add_argument("--max-lag", type=int, default=15, help="lag sweep range in frames")
    args = ap.parse_args()

    files = sorted(
        args.raw_dir.glob("episode_*.hdf5"),
        key=lambda p: int(p.stem.split("_")[1]),
    )[: args.max_episodes]
    if not files:
        raise SystemExit(f"No episode_*.hdf5 found under {args.raw_dir}")

    print(f"FK on {len(files)} episodes from {args.raw_dir}\n")
    fk = FKHelper()

    per_ep_best_lag: list[int] = []
    per_ep_best_trans: list[float] = []
    per_ep_best_rot: list[float] = []
    per_ep_zero_trans: list[float] = []
    per_ep_zero_rot: list[float] = []
    per_ep_dt_ms: list[float] = []

    print(f"{'episode':<12}{'T':>5}{'dt_ms':>8}{'best_lag':>10}{'t_err_mm@best':>16}{'r_err_deg@best':>16}{'t_err_mm@0':>14}{'r_err_deg@0':>14}")
    print("-" * 95)

    for f in files:
        with h5py.File(f, "r") as h:
            action_eef = h["action/eef/right"][:].astype(np.float64)
            torso = h["obs/joint/torso"][:].astype(np.float64)
            right_arm = h["obs/joint/right_arm"][:].astype(np.float64)
            ts = h["timestamp_ns"][:]
        T = action_eef.shape[0]
        if T < 2 * args.max_lag + 4:
            continue
        dt_ms = float(np.diff(ts).mean()) / 1e6
        obs_eef = fk.fk_batch(torso, right_arm)
        lags, te, re = sweep_lag(action_eef, obs_eef, args.max_lag)
        # Cost combining translation (mm) and rotation (deg) on roughly comparable scales.
        cost = te * 1000.0 + re
        i_best = int(np.argmin(cost))
        i_zero = int(np.where(lags == 0)[0][0])
        per_ep_best_lag.append(int(lags[i_best]))
        per_ep_best_trans.append(float(te[i_best]))
        per_ep_best_rot.append(float(re[i_best]))
        per_ep_zero_trans.append(float(te[i_zero]))
        per_ep_zero_rot.append(float(re[i_zero]))
        per_ep_dt_ms.append(dt_ms)
        print(
            f"{f.stem:<12}{T:>5}{dt_ms:>8.2f}{lags[i_best]:>10}"
            f"{te[i_best]*1000:>16.2f}{re[i_best]:>16.3f}"
            f"{te[i_zero]*1000:>14.2f}{re[i_zero]:>14.3f}"
        )

    if not per_ep_best_lag:
        raise SystemExit("No episodes long enough to sweep.")

    lags_arr = np.asarray(per_ep_best_lag)
    print("\n=== Aggregate ===")
    print(f"episodes processed: {len(lags_arr)}")
    print(f"mean dt: {np.mean(per_ep_dt_ms):.2f} ms (=> ~{1000.0/np.mean(per_ep_dt_ms):.2f} fps)")
    print(f"best lag (frames): mean={lags_arr.mean():.2f}  median={np.median(lags_arr):.1f}  std={lags_arr.std():.2f}  min={lags_arr.min()} max={lags_arr.max()}")
    best_lag_ms = lags_arr.mean() * np.mean(per_ep_dt_ms)
    print(f"best lag (ms approx, mean*dt): {best_lag_ms:.1f} ms")
    print(f"trans err @ best lag: mean={np.mean(per_ep_best_trans)*1000:.2f} mm  median={np.median(per_ep_best_trans)*1000:.2f} mm")
    print(f"rot   err @ best lag: mean={np.mean(per_ep_best_rot):.3f} deg  median={np.median(per_ep_best_rot):.3f} deg")
    print(f"trans err @ lag=0   : mean={np.mean(per_ep_zero_trans)*1000:.2f} mm  median={np.median(per_ep_zero_trans)*1000:.2f} mm")
    print(f"rot   err @ lag=0   : mean={np.mean(per_ep_zero_rot):.3f} deg  median={np.median(per_ep_zero_rot):.3f} deg")

    # Histogram of best lags
    print("\nbest-lag histogram:")
    for lag in range(int(lags_arr.min()), int(lags_arr.max()) + 1):
        c = int((lags_arr == lag).sum())
        print(f"  {lag:>+3d}: {'#'*c} ({c})")


if __name__ == "__main__":
    main()
