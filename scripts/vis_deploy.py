"""Compare GT vs policy action vs achieved EEF positions as matplotlib plots.

Saves three PNGs (``eef_x.png``, ``eef_y.png``, ``eef_z.png``) to ``--out``.
Each plot overlays four lines:

- red    = ``gt_obs``     (FK from GT ``/obs/joint/{torso, right_arm}``)
- orange = ``gt_action``  (FK from GT ``/obs/joint/torso`` + ``/action/joint/right_arm``)
- green  = ``infer_action`` (``/action/eef_9d/right`` in --infer)
- blue   = ``infer_obs``    (``/obs/eef_9d/right`` in --infer)

Frame ranges where ``/action/gripper/right`` equals 1 are shaded gold so
grasp/release phases are obvious on each curve.

GT and INFER lengths can differ; each line just ends at its own length.

Usage::

    python scripts/vis_eef_curves.py \
        --gt /home/yixuan/Dexmate/data/raw_data_renamed/test/episode_0.hdf5 \
        --infer /home/yixuan/Dexmate/deploy/dp/dexmate_eef_eef/last/0/episode_0.hdf5

``--out`` defaults to the parent directory of ``--infer`` when omitted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from yixuan_utilities.kinematics_helper import KinHelper

_ROBOT_NAME = "vega_no_effector"
_RIGHT_ARM_EEF_LINK = "R_ee"
_TORSO_JOINTS = ("torso_j1", "torso_j2", "torso_j3")
_HEAD_JOINTS = ("head_j1", "head_j2", "head_j3")
_LEFT_ARM_JOINTS = tuple(f"L_arm_j{i}" for i in range(1, 8))
_RIGHT_ARM_JOINTS = tuple(f"R_arm_j{i}" for i in range(1, 8))

_AXES = ("x", "y", "z")
_GT_COLOR = "tab:red"
_GT_ACTION_JOINT_COLOR = "tab:orange"
_ACTION_COLOR = "tab:green"
_OBS_COLOR = "tab:blue"


class _Kin:
    """Shared KinHelper with a right-arm-EEF FK convenience method."""

    def __init__(self) -> None:
        self.kin = KinHelper(_ROBOT_NAME)
        self.eef_idx = self.kin.link_name_to_idx[_RIGHT_ARM_EEF_LINK]
        active = self.kin.sapien_robot.get_active_joints()
        self.name_to_idx = {j.name: i for i, j in enumerate(active)}
        self.dof = self.kin.sapien_robot.dof

    def fk_eef_pos(self, torso, l_arm, r_arm, head) -> np.ndarray:
        qpos = np.zeros(self.dof, dtype=np.float64)
        for vals, names in (
            (torso, _TORSO_JOINTS),
            (l_arm, _LEFT_ARM_JOINTS),
            (r_arm, _RIGHT_ARM_JOINTS),
            (head, _HEAD_JOINTS),
        ):
            for v, n in zip(vals, names, strict=True):
                if n in self.name_to_idx:
                    qpos[self.name_to_idx[n]] = v
        M = self.kin.compute_fk_from_link_idx(qpos, [self.eef_idx])[0]
        if M.shape != (4, 4):
            raise ValueError(f"expected (4, 4) SE(3), got {M.shape}")
        return M[:3, 3].astype(np.float32)


def compute_gt_pos(gt_path: str, kin: _Kin) -> np.ndarray:
    """Return ``(T, 3)`` GT right-EEF positions via FK (mirrors port_dexmate_hdf5)."""
    with h5py.File(gt_path, "r") as f:
        torso = np.array(f["/obs/joint/torso"], dtype=np.float64)
        right_arm = np.array(f["/obs/joint/right_arm"], dtype=np.float64)
    if torso.shape[0] != right_arm.shape[0]:
        raise ValueError(
            f"GT frame counts disagree: torso={torso.shape[0]}, right_arm={right_arm.shape[0]}"
        )
    if torso.ndim != 2 or torso.shape[1] != 3:
        raise ValueError(f"GT torso shape: expected (T, 3), got {torso.shape}")
    if right_arm.ndim != 2 or right_arm.shape[1] != 7:
        raise ValueError(f"GT right_arm shape: expected (T, 7), got {right_arm.shape}")
    T = torso.shape[0]
    pos = np.zeros((T, 3), dtype=np.float32)
    zero_l = np.zeros(7)
    zero_h = np.zeros(3)
    for t in range(T):
        pos[t] = kin.fk_eef_pos(torso[t], zero_l, right_arm[t], zero_h)
    return pos


def compute_gt_action_joint_pos(gt_path: str, kin: _Kin) -> np.ndarray:
    """Return ``(T, 3)`` right-EEF positions via FK on commanded right-arm joints.

    Uses observed torso (``/obs/joint/torso``) with commanded right arm
    (``/action/joint/right_arm``), matching ``port_dexmate_hdf5`` action EEF.
    """
    with h5py.File(gt_path, "r") as f:
        torso = np.array(f["/obs/joint/torso"], dtype=np.float64)
        right_arm = np.array(f["/action/joint/right_arm"], dtype=np.float64)
    if torso.shape[0] != right_arm.shape[0]:
        raise ValueError(
            f"GT frame counts disagree: torso={torso.shape[0]}, "
            f"action/right_arm={right_arm.shape[0]}"
        )
    if torso.ndim != 2 or torso.shape[1] != 3:
        raise ValueError(f"GT torso shape: expected (T, 3), got {torso.shape}")
    if right_arm.ndim != 2 or right_arm.shape[1] != 7:
        raise ValueError(f"GT action/right_arm shape: expected (T, 7), got {right_arm.shape}")
    T = torso.shape[0]
    pos = np.zeros((T, 3), dtype=np.float32)
    zero_l = np.zeros(7)
    zero_h = np.zeros(3)
    for t in range(T):
        pos[t] = kin.fk_eef_pos(torso[t], zero_l, right_arm[t], zero_h)
    return pos


def load_infer_pos(infer_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(action_pos, obs_pos, action_gripper)`` from the INFER file.

    Shapes: ``action_pos`` and ``obs_pos`` are ``(T, 3)``; ``action_gripper`` is
    ``(T,)`` with values in ``{0.0, 1.0}`` (binarised in ``policy_rollout``).
    """
    with h5py.File(infer_path, "r") as f:
        act = np.array(f["/action/eef_9d/right"], dtype=np.float32)
        obs = np.array(f["/obs/eef_9d/right"], dtype=np.float32)
        grip = np.array(f["/action/gripper/right"], dtype=np.float32)
    if act.ndim != 2 or act.shape[1] != 9:
        raise ValueError(f"/action/eef_9d/right shape: expected (T, 9), got {act.shape}")
    if obs.ndim != 2 or obs.shape[1] != 9:
        raise ValueError(f"/obs/eef_9d/right shape: expected (T, 9), got {obs.shape}")
    if act.shape[0] != obs.shape[0]:
        raise ValueError(
            f"INFER frame counts disagree: action={act.shape[0]}, obs={obs.shape[0]}"
        )
    if grip.ndim != 1 or grip.shape[0] != act.shape[0]:
        raise ValueError(
            f"/action/gripper/right shape: expected ({act.shape[0]},), got {grip.shape}"
        )
    return act[:, :3], obs[:, :3], grip


def export_infer_csv(infer_path: str, out_dir: Path) -> None:
    """Write ``/action`` and ``/obs`` EEF arrays (9D + gripper) to two CSVs in ``out_dir``.

    Each CSV has 10 columns: the 9D EEF pose followed by the matching gripper
    scalar (``/action/gripper/right`` for the action file, ``/obs/gripper/right``
    for the obs file). File names are ``{stem}_eef_action.csv`` and
    ``{stem}_eef_obs.csv`` where ``stem`` is the INFER file stem (e.g.
    ``episode_0``).
    """
    with h5py.File(infer_path, "r") as f:
        act = np.array(f["/action/eef_9d/right"], dtype=np.float32)
        obs = np.array(f["/obs/eef_9d/right"], dtype=np.float32)
        act_grip = np.array(f["/action/gripper/right"], dtype=np.float32)
        obs_grip = np.array(f["/obs/gripper/right"], dtype=np.float32)
    stem = Path(infer_path).stem
    header = ",".join([*(f"d{i}" for i in range(9)), "gripper"])
    for arr, grip, suffix, eef_key, grip_key in (
        (act, act_grip, "eef_action", "/action/eef_9d/right", "/action/gripper/right"),
        (obs, obs_grip, "eef_obs", "/obs/eef_9d/right", "/obs/gripper/right"),
    ):
        if arr.ndim != 2 or arr.shape[1] != 9:
            raise ValueError(f"{eef_key} shape: expected (T, 9), got {arr.shape}")
        if grip.ndim != 1 or grip.shape[0] != arr.shape[0]:
            raise ValueError(
                f"{grip_key} shape: expected ({arr.shape[0]},), got {grip.shape}"
            )
        out = np.concatenate([arr, grip[:, None]], axis=1)
        out_path = out_dir / f"{stem}_{suffix}.csv"
        np.savetxt(out_path, out, delimiter=",", header=header, comments="")
        print(f"saved {out_path}")


def contig_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return ``[(start, end_exclusive), ...]`` for contiguous True runs in ``mask``."""
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt",
        type=str,
        default="/home/yixuan/Dexmate/data/raw_data_renamed/test/episode_3.hdf5",
        help="GT HDF5 (uses /obs/joint/{torso, right_arm} for FK).",
    )
    parser.add_argument(
        "--infer",
        type=str,
        default="/home/yixuan/omniteleop/Dexmate/deploy/act_abs_eef_eef/4/episode_0.hdf5",
        help="Inference HDF5 (uses /action/eef_9d/right and /obs/eef_9d/right).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for the three PNGs (default: parent directory of --infer).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out is not None else Path(args.infer).resolve().parent
    if not out_dir.is_dir():
        raise ValueError(f"--out is not an existing directory: {out_dir}")

    export_infer_csv(args.infer, out_dir)

    kin = _Kin()
    gt_obs_pos = compute_gt_pos(args.gt, kin)
    gt_action_pos = compute_gt_action_joint_pos(args.gt, kin)
    infer_action_pos, infer_obs_pos, action_gripper = load_infer_pos(args.infer)

    n_gt_obs = gt_obs_pos.shape[0]
    n_gt_action = gt_action_pos.shape[0]
    n_infer_action = infer_action_pos.shape[0]
    n_infer_obs = infer_obs_pos.shape[0]
    grip_runs = contig_runs(action_gripper >= 0.5)
    print(
        f"gt_obs: {n_gt_obs} frames  |  gt_action: {n_gt_action} frames  "
        f"|  infer_action: {n_infer_action} frames  |  infer_obs: {n_infer_obs} frames  "
        f"|  gripper==1 runs: {grip_runs}"
    )

    gt_obs_frames = np.arange(n_gt_obs)
    gt_action_frames = np.arange(n_gt_action)
    infer_action_frames = np.arange(n_infer_action)
    infer_obs_frames = np.arange(n_infer_obs)

    for i, ax_name in enumerate(_AXES):
        fig, ax = plt.subplots(figsize=(10, 4))
        for j, (s, e) in enumerate(grip_runs):
            # axvspan takes inclusive xmin / exclusive-style xmax; use end-1 so
            # the shaded band covers frames [s, e-1] without overshooting the line.
            ax.axvspan(
                s, e - 1, color="gold", alpha=0.25, zorder=0,
                label="gripper=1" if j == 0 else None,
            )
        ax.plot(gt_obs_frames, gt_obs_pos[:, i], color=_GT_COLOR, label="gt_obs", linewidth=2.0)
        ax.plot(
            gt_action_frames, gt_action_pos[:, i],
            color=_GT_ACTION_JOINT_COLOR, label="gt_action", linewidth=2.0,
        )
        ax.plot(
            infer_action_frames, infer_action_pos[:, i],
            color=_ACTION_COLOR, label="infer_action", linewidth=2.0,
        )
        ax.plot(
            infer_obs_frames, infer_obs_pos[:, i],
            color=_OBS_COLOR, label="infer_obs", linewidth=2.0,
        )
        ax.set_xlabel("frame")
        ax.set_ylabel(f"EEF {ax_name} (m)")
        ax.set_title(f"Right-EEF {ax_name}: gt_obs / gt_action / infer_action / infer_obs")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()

        out_path = out_dir / f"eef_{ax_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
