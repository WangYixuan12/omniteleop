"""Plot every intermediate stage of a recorded VR teleop episode.

Reads a main episode HDF5 (`episode_<N>.hdf5`) and an optional debug HDF5
(`episode_<N>_debug.hdf5`, produced by `DebugEpisodeRecorder` in vr_reader.py)
and produces ~12 matplotlib figures plus a printed per-joint summary.

Stages visualised
-----------------
- Per-joint commanded vs actual (arms, head)
- Raw IK output before rate-limiting (debug file)
- Commanded EEF target — main `action/eef` plus debug `eef_used_by_ik`.
  These must match: every EEF dataset is the robot-frame target used by IK,
  independent of start mode.
- Rate-limit clipping size and joint-tracking-error histograms
- Loop period and timing breakdown (IK solve, publish)
- Stage-equivalence: published payload vs rate-limited cmd (must be 0)
- Gripper command (no actual feedback recorded)
- Chassis velocity command, with raw thumbstick overlay

Usage::
    python scripts/vis_teleop_curves.py --hdf5 /path/episode_5.hdf5
    python scripts/vis_teleop_curves.py --hdf5 ... --save-dir out/ --no-fk
"""

from __future__ import annotations

import argparse
import pathlib
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5


def _arr(x) -> np.ndarray:
    """Materialise an h5py dataset to a numpy array."""
    return np.asarray(x)


# ── HDF5 loaders ─────────────────────────────────────────────────────────────


def _load_main(path: str) -> dict:
    data, h5 = load_dict_from_hdf5(path)
    out = {
        "ts_ns": _arr(data["timestamp_ns"]),
        "cmd_left": _arr(data["action"]["joint"]["left_arm"]),
        "cmd_right": _arr(data["action"]["joint"]["right_arm"]),
        "cmd_head": _arr(data["action"]["joint"]["head"]),
        "cmd_eef_left": _arr(data["action"]["eef"]["left"]),
        "cmd_eef_right": _arr(data["action"]["eef"]["right"]),
        "grip_left": _arr(data["action"]["gripper"]["left"]),
        "grip_right": _arr(data["action"]["gripper"]["right"]),
        "vx": _arr(data["action"]["joint"]["chassis_vx"]),
        "vy": _arr(data["action"]["joint"]["chassis_vy"]),
        "wz": _arr(data["action"]["joint"]["chassis_wz"]),
        "act_left": _arr(data["obs"]["joint"]["left_arm"]),
        "act_right": _arr(data["obs"]["joint"]["right_arm"]),
        "act_head": _arr(data["obs"]["joint"]["head"]),
        "act_torso": _arr(data["obs"]["joint"]["torso"]),
    }
    h5.close()
    return out


def _load_debug(path: str) -> dict:
    data, h5 = load_dict_from_hdf5(path)
    out = {
        "ts_ns": _arr(data["timestamp_ns"]),
        "monotonic_ns": _arr(data["timing"]["monotonic_ns"]),
        "ik_solve_ms": _arr(data["timing"]["ik_solve_ms"]),
        "publish_ms": _arr(data["timing"]["publish_ms"]),
        "calib_stage": _arr(data["calib_stage"]),
        "vr_head": _arr(data["vr_raw"]["head"]),
        "vr_left_wrist": _arr(data["vr_raw"]["left_wrist"]),
        "vr_right_wrist": _arr(data["vr_raw"]["right_wrist"]),
        "left_thumbstick": _arr(data["vr_raw"]["left_thumbstick"]),
        "right_thumbstick": _arr(data["vr_raw"]["right_thumbstick"]),
        "left_trigger": _arr(data["vr_raw"]["left_trigger"]),
        "right_trigger": _arr(data["vr_raw"]["right_trigger"]),
        "robot_base_t_vr_base": _arr(data["calib"]["robot_base_t_vr_base"]),
        "vr_to_robot_left": _arr(data["calib"]["vr_to_robot_left"]),
        "vr_to_robot_right": _arr(data["calib"]["vr_to_robot_right"]),
        "eef_used_by_ik_left": _arr(data["target"]["eef_used_by_ik"]["left"]),
        "eef_used_by_ik_right": _arr(data["target"]["eef_used_by_ik"]["right"]),
        "eef_recorded_main_left": _arr(data["target"]["eef_recorded_main"]["left"]),
        "eef_recorded_main_right": _arr(data["target"]["eef_recorded_main"]["right"]),
        "ik_raw_left": _arr(data["ik"]["left_arm_raw"]),
        "ik_raw_right": _arr(data["ik"]["right_arm_raw"]),
        "ik_success": _arr(data["ik"]["status"]["success"]),
        "ik_in_collision": _arr(data["ik"]["status"]["in_collision"]),
        "ik_within_limits": _arr(data["ik"]["status"]["within_limits"]),
        "ik_failure_reason": _arr(data["ik"]["status"]["failure_reason"]),
        "publish_head_pos": _arr(data["publish"]["payload"]["head_pos"]),
        "publish_left_arm_pos": _arr(data["publish"]["payload"]["left_arm_pos"]),
        "publish_right_arm_pos": _arr(data["publish"]["payload"]["right_arm_pos"]),
    }
    h5.close()
    return out


# ── Plotting helpers ─────────────────────────────────────────────────────────


def fig_arm_joints(
    t: np.ndarray,
    cmd: np.ndarray,
    actual: np.ndarray,
    ik_raw: Optional[np.ndarray],
    side: str,
    fig_id: int,
) -> plt.Figure:
    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(10, 14))
    for i in range(7):
        ax = axes[i]
        if ik_raw is not None:
            ax.plot(t, ik_raw[:, i], color="lightgray", linestyle="--", lw=1.0, label="ik_raw")
        ax.plot(t, cmd[:, i], color="C0", lw=1.4, label="cmd")
        ax.plot(t, actual[:, i], color="C3", lw=1.0, label="actual")
        ax.set_ylabel(f"{side}_arm_j{i+1}\n(rad)", fontsize=9)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle(f"Fig {fig_id}: {side} arm joint cmd vs actual", fontsize=11)
    fig.tight_layout()
    return fig


def fig_head_joints(
    t: np.ndarray, cmd: np.ndarray, actual: np.ndarray, fig_id: int
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i in range(3):
        axes[i].plot(t, cmd[:, i], color="C0", lw=1.4, label="cmd")
        axes[i].plot(t, actual[:, i], color="C3", lw=1.0, label="actual")
        axes[i].set_ylabel(f"head_j{i+1} (rad)", fontsize=9)
        axes[i].grid(alpha=0.3)
        if i == 0:
            axes[i].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle(f"Fig {fig_id}: head joint cmd vs actual", fontsize=11)
    fig.tight_layout()
    return fig


def fig_eef_xyz(
    t: np.ndarray,
    target_used: np.ndarray,
    target_main: np.ndarray,
    actual_eef: Optional[np.ndarray],
    side: str,
    fig_id: int,
) -> plt.Figure:
    """target_used / target_main: (N,4,4); actual_eef: (N,3) FK-derived."""
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    labels = "xyz"
    for i in range(3):
        ax = axes[i]
        ax.plot(t, target_used[:, i, 3], color="C0", lw=1.4, label="target (used_by_ik)")
        ax.plot(
            t,
            target_main[:, i, 3],
            color="gray",
            lw=0.9,
            ls=":",
            label="target (main action/eef)",
        )
        if actual_eef is not None:
            ax.plot(t, actual_eef[:, i], color="C3", lw=1.0, label="actual (FK)")
        ax.set_ylabel(f"{labels[i]} (m)", fontsize=9)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle(f"Fig {fig_id}: {side} arm EEF xyz", fontsize=11)
    fig.tight_layout()
    return fig


def fig_grippers(t: np.ndarray, gl: np.ndarray, gr: np.ndarray, fig_id: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, gl, color="C0", lw=1.2, label="left trigger (cmd)")
    ax.plot(t, gr, color="C3", lw=1.2, label="right trigger (cmd)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("trigger value [0..1]")
    ax.set_title(f"Fig {fig_id}: gripper command (no actual feedback recorded)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_chassis(
    t: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    wz: np.ndarray,
    dbg: Optional[dict],
    fig_id: int,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    cmds = [(vx, "vx (m/s)"), (vy, "vy (m/s)"), (wz, "wz (rad/s)")]
    for ax, (cmd, lbl) in zip(axes, cmds):
        ax.plot(t, cmd, color="C0", lw=1.2, label="cmd")
        ax.set_ylabel(lbl)
        ax.grid(alpha=0.3)
    if dbg is not None:
        # _thumbstick_to_chassis maps:
        #   vx ← -left_thumbstick[1], vy ← -left_thumbstick[0],
        #   wz ← -right_thumbstick[0]
        ls = dbg["left_thumbstick"]
        rs = dbg["right_thumbstick"]
        raws = [-ls[:, 1], -ls[:, 0], -rs[:, 0]]
        for ax, raw in zip(axes, raws):
            ax2 = ax.twinx()
            ax2.plot(t, raw, color="lightgray", lw=0.8, label="raw stick")
            ax2.set_ylabel("stick", color="gray", fontsize=8)
            ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle(f"Fig {fig_id}: chassis velocity (cmd) + raw thumbstick", fontsize=11)
    fig.tight_layout()
    return fig


def fig_loop_and_timing(
    t_ns: np.ndarray, dbg: Optional[dict], fig_id: int
) -> plt.Figure:
    """Combined: loop period + per-frame ik_solve / publish timings."""
    dt_ms = np.diff(t_ns).astype(np.float64) / 1e6
    fig, ax = plt.subplots(figsize=(10, 5))
    loop_x = np.arange(1, len(dt_ms) + 1)
    ax.plot(loop_x, dt_ms, lw=0.8, color="C0", label="loop period")
    if dbg is not None:
        x = np.arange(len(dbg["ik_solve_ms"]))
        ax.plot(x, dbg["ik_solve_ms"], lw=0.8, color="C1", label="ik_solve_ms")
        ax.plot(x, dbg["publish_ms"], lw=0.8, color="C2", label="publish_ms")
    mean = float(np.mean(dt_ms)) if len(dt_ms) else float("nan")
    std = float(np.std(dt_ms)) if len(dt_ms) else float("nan")
    p95 = float(np.percentile(dt_ms, 95)) if len(dt_ms) else float("nan")
    ax.axhline(
        mean, color="C3", ls="--", lw=0.9, alpha=0.6, label=f"loop mean = {mean:.2f} ms"
    )
    ax.axhline(
        p95, color="C3", ls=":", lw=0.9, alpha=0.6, label=f"loop p95 = {p95:.2f} ms"
    )
    ax.set_xlabel("frame index")
    ax.set_ylabel("ms")
    ax.set_title(
        f"Fig {fig_id}: loop period + timing breakdown "
        f"(loop mean={mean:.2f}ms std={std:.2f}ms p95={p95:.2f}ms)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_error_hist(
    cmd: np.ndarray,
    actual: np.ndarray,
    ik_raw: Optional[np.ndarray],
    side: str,
    fig_id: int,
) -> plt.Figure:
    fig, axes = plt.subplots(7, 2, figsize=(11, 14))
    for i in range(7):
        if ik_raw is not None:
            valid = ~np.isnan(ik_raw[:, i])
            clip = cmd[valid, i] - ik_raw[valid, i]
            axes[i, 0].hist(clip, bins=60, color="C0")
            axes[i, 0].set_title(
                f"{side}_j{i+1}  cmd − ik_raw (rate-limit clip)", fontsize=8
            )
        else:
            axes[i, 0].axis("off")
        err = actual[:, i] - cmd[:, i]
        axes[i, 1].hist(err, bins=60, color="C3")
        rms = float(np.sqrt(np.mean(err**2)))
        peak = float(np.max(np.abs(err)))
        axes[i, 1].set_title(
            f"{side}_j{i+1}  actual − cmd  RMS={rms:.4f}  peak={peak:.4f}",
            fontsize=8,
        )
    fig.suptitle(f"Fig {fig_id}: {side} arm error histograms", fontsize=11)
    fig.tight_layout()
    return fig


def fig_stage_eq(
    cmd_left: np.ndarray,
    cmd_right: np.ndarray,
    cmd_head: np.ndarray,
    dbg: dict,
    fig_id: int,
) -> plt.Figure:
    """Stage 5 == Stage 4: published payload − rate-limited cmd should be 0."""
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    axes[0].plot(np.max(np.abs(dbg["publish_left_arm_pos"] - cmd_left), axis=1), color="C0")
    axes[0].set_ylabel("max |Δ| left_arm")
    axes[1].plot(
        np.max(np.abs(dbg["publish_right_arm_pos"] - cmd_right), axis=1), color="C0"
    )
    axes[1].set_ylabel("max |Δ| right_arm")
    axes[2].plot(np.max(np.abs(dbg["publish_head_pos"] - cmd_head), axis=1), color="C0")
    axes[2].set_ylabel("max |Δ| head")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("frame index")
    fig.suptitle(f"Fig {fig_id}: published payload − rate-limited cmd  (should be 0)")
    fig.tight_layout()
    return fig


# ── FK helper ────────────────────────────────────────────────────────────────


def compute_actual_eef(
    main: dict, robot_name: str = "vega_no_effector"
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame actual EEF positions (N,3) for left and right arm."""
    from yixuan_utilities.kinematics_helper import KinHelper

    kin = KinHelper(robot_name)
    name_to_idx = {j.name: i for i, j in enumerate(kin.sapien_robot.get_active_joints())}
    L_idx = kin.link_name_to_idx["L_ee"]
    R_idx = kin.link_name_to_idx["R_ee"]

    n = main["act_left"].shape[0]
    out_l = np.full((n, 3), np.nan, dtype=np.float32)
    out_r = np.full((n, 3), np.nan, dtype=np.float32)
    qpos_template = np.zeros(kin.sapien_robot.dof)

    sources: list[tuple[str, np.ndarray, int]] = []
    for j in range(min(3, main["act_torso"].shape[1])):
        sources.append((f"torso_j{j+1}", main["act_torso"], j))
    for j in range(7):
        sources.append((f"L_arm_j{j+1}", main["act_left"], j))
    for j in range(7):
        sources.append((f"R_arm_j{j+1}", main["act_right"], j))
    for j in range(min(3, main["act_head"].shape[1])):
        sources.append((f"head_j{j+1}", main["act_head"], j))

    for i in range(n):
        qpos = qpos_template.copy()
        for jn, src, col in sources:
            if jn in name_to_idx:
                qpos[name_to_idx[jn]] = src[i, col]
        l_mat = kin.compute_fk_from_link_idx(qpos, [L_idx])[0]
        r_mat = kin.compute_fk_from_link_idx(qpos, [R_idx])[0]
        out_l[i] = l_mat[:3, 3]
        out_r[i] = r_mat[:3, 3]
    return out_l, out_r


def estimate_lag_ms(cmd: np.ndarray, actual: np.ndarray, dt_s: float) -> float:
    """Cross-correlation peak lag in ms (positive = actual lags cmd)."""
    a = cmd - cmd.mean()
    b = actual - actual.mean()
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    xc = np.correlate(b, a, mode="full")
    lag_idx = int(np.argmax(xc) - (len(a) - 1))
    return lag_idx * dt_s * 1000.0


def assert_eef_targets_match(main: dict, dbg: dict, atol: float = 1e-5) -> None:
    """Require all EEF datasets to be the same robot-frame IK target."""
    checks = [
        ("left", main["cmd_eef_left"], dbg["eef_used_by_ik_left"]),
        ("left", main["cmd_eef_left"], dbg["eef_recorded_main_left"]),
        ("right", main["cmd_eef_right"], dbg["eef_used_by_ik_right"]),
        ("right", main["cmd_eef_right"], dbg["eef_recorded_main_right"]),
    ]
    for side, main_eef, debug_eef in checks:
        diff = np.abs(main_eef - debug_eef)
        max_diff = float(np.nanmax(diff))
        if max_diff > atol:
            raise ValueError(
                f"{side} EEF mismatch: main action/eef and debug EEF differ by "
                f"max {max_diff:.6g}. Regenerate the episode with vr_reader.py so "
                "all EEF datasets are the robot-frame targets used by IK."
            )


def print_summary(main: dict, dbg: Optional[dict]) -> None:
    ts_ns = main["ts_ns"]
    dt_ns = np.diff(ts_ns).astype(np.float64)
    dt_mean_s = float(np.mean(dt_ns)) / 1e9 if len(dt_ns) else 0.0

    print()
    print("══════ Per-joint tracking error and estimated lag ══════")
    print(f"{'joint':<14} {'rms (rad)':>10} {'peak (rad)':>12} {'lag (ms)':>10}")
    for side, cmd, actual in [
        ("L", main["cmd_left"], main["act_left"]),
        ("R", main["cmd_right"], main["act_right"]),
    ]:
        for j in range(7):
            err = actual[:, j] - cmd[:, j]
            rms = float(np.sqrt(np.mean(err**2)))
            peak = float(np.max(np.abs(err)))
            lag = estimate_lag_ms(cmd[:, j], actual[:, j], dt_mean_s)
            print(f"{side}_arm_j{j+1:<7} {rms:>10.4f} {peak:>12.4f} {lag:>10.1f}")
    for j in range(min(3, main["act_head"].shape[1])):
        err = main["act_head"][:, j] - main["cmd_head"][:, j]
        rms = float(np.sqrt(np.mean(err**2)))
        peak = float(np.max(np.abs(err)))
        lag = estimate_lag_ms(main["cmd_head"][:, j], main["act_head"][:, j], dt_mean_s)
        print(f"head_j{j+1:<10} {rms:>10.4f} {peak:>12.4f} {lag:>10.1f}")

    print()
    print("══════ Loop period ══════")
    if len(dt_ns):
        dt_ms = dt_ns / 1e6
        print(
            f"mean = {dt_ms.mean():.2f} ms, "
            f"std = {dt_ms.std():.2f} ms, "
            f"p95 = {np.percentile(dt_ms, 95):.2f} ms"
        )

    if dbg is not None:
        print()
        print("══════ IK status breakdown ══════")
        reasons, counts = np.unique(dbg["ik_failure_reason"], return_counts=True)
        for r, c in zip(reasons, counts):
            r_str = r.decode() if isinstance(r, (bytes, np.bytes_)) else str(r)
            print(f"  {r_str:<30} {int(c)}")

        print()
        print("══════ Timing ══════")
        for name, vals in [
            ("ik_solve_ms", dbg["ik_solve_ms"]),
            ("publish_ms", dbg["publish_ms"]),
        ]:
            v = vals[~np.isnan(vals)]
            if len(v) == 0:
                print(f"  {name}: all NaN")
            else:
                print(
                    f"  {name}: mean={v.mean():.3f}, "
                    f"p95={np.percentile(v, 95):.3f}, max={v.max():.3f}"
                )


# ── main ─────────────────────────────────────────────────────────────────────


def _maybe_show_or_save(figs: dict, save_dir: Optional[str]) -> None:
    if save_dir:
        out = pathlib.Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            path = out / f"{name}.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            print(f"saved {path}")
    plt.show()


def main_fn() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", default="/home/yixuan/omniteleop/Dexmate/raw_data/episode_0.hdf5", help="Path to main episode HDF5")
    parser.add_argument(
        "--debug", default="/home/yixuan/omniteleop/Dexmate/raw_data/episode_0_debug.hdf5", help="Path to debug HDF5 (auto-derived if omitted)"
    )
    parser.add_argument(
        "--save-dir",
        default="/home/yixuan/omniteleop/Dexmate/debug/plots",
        help="If set, save PNGs instead of plt.show()",
    )
    parser.add_argument(
        "--no-fk",
        action="store_true",
        help="Skip KinHelper FK (faster, but no actual EEF curves)",
    )
    parser.add_argument("--robot-name", default="vega_no_effector")
    args = parser.parse_args()

    main_path = pathlib.Path(args.hdf5)
    if args.debug:
        debug_path = pathlib.Path(args.debug)
    else:
        if main_path.stem.endswith("_debug"):
            warnings.warn("--hdf5 looks like a debug file; pass the main file instead.")
        debug_path = main_path.parent / f"{main_path.stem}_debug.hdf5"

    main = _load_main(str(main_path))
    if debug_path.exists():
        dbg = _load_debug(str(debug_path))
        print(
            f"loaded main ({main['ts_ns'].shape[0]} frames) + "
            f"debug ({dbg['ts_ns'].shape[0]} frames)"
        )
        if dbg["ts_ns"].shape[0] != main["ts_ns"].shape[0]:
            warnings.warn(
                f"frame count mismatch: main={main['ts_ns'].shape[0]} "
                f"debug={dbg['ts_ns'].shape[0]}"
            )
        assert_eef_targets_match(main, dbg)
    else:
        dbg = None
        print(
            f"loaded main ({main['ts_ns'].shape[0]} frames); "
            f"debug file not found at {debug_path}"
        )

    ts0 = main["ts_ns"][0]
    t = (main["ts_ns"] - ts0).astype(np.float64) / 1e9

    figs: dict[str, plt.Figure] = {}
    figs["fig01_left_arm"] = fig_arm_joints(
        t,
        main["cmd_left"],
        main["act_left"],
        dbg["ik_raw_left"] if dbg else None,
        "L",
        1,
    )
    figs["fig02_right_arm"] = fig_arm_joints(
        t,
        main["cmd_right"],
        main["act_right"],
        dbg["ik_raw_right"] if dbg else None,
        "R",
        2,
    )
    figs["fig03_head"] = fig_head_joints(t, main["cmd_head"], main["act_head"], 3)

    if not args.no_fk:
        print("computing FK for actual EEF (this can take a few seconds) …")
        actual_eef_l, actual_eef_r = compute_actual_eef(main, args.robot_name)
    else:
        actual_eef_l = actual_eef_r = None

    target_used_l = dbg["eef_used_by_ik_left"] if dbg else main["cmd_eef_left"]
    target_used_r = dbg["eef_used_by_ik_right"] if dbg else main["cmd_eef_right"]
    target_main_l = main["cmd_eef_left"]
    target_main_r = main["cmd_eef_right"]

    figs["fig04_left_eef"] = fig_eef_xyz(
        t, target_used_l, target_main_l, actual_eef_l, "L", 4
    )
    figs["fig05_right_eef"] = fig_eef_xyz(
        t, target_used_r, target_main_r, actual_eef_r, "R", 5
    )

    figs["fig06_grippers"] = fig_grippers(t, main["grip_left"], main["grip_right"], 6)
    figs["fig07_chassis"] = fig_chassis(
        t, main["vx"], main["vy"], main["wz"], dbg, 7
    )
    figs["fig08_loop_and_timing"] = fig_loop_and_timing(main["ts_ns"], dbg, 8)

    if dbg is not None:
        figs["fig12_stage_eq"] = fig_stage_eq(
            main["cmd_left"], main["cmd_right"], main["cmd_head"], dbg, 12
        )

    print_summary(main, dbg)
    _maybe_show_or_save(figs, args.save_dir)


if __name__ == "__main__":
    main_fn()
