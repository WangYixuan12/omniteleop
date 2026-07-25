"""Collect scripted-expert episodes as RAW HDF5 takes, in the real recorder's schema.

Output is drop-in for the existing pipeline -- the sim takes go through the SAME porters and
the SAME visualizer as hardware takes:

    behavior         python collect_demos.py --task mobilepickplace --episodes 20
    dexmate_lerobot  python scripts/port_wbc_mobile_hdf5.py --raw-dir <out>/raw_data ...
    dexmate_maniflow python scripts/port_wbc_mobile_zarr.py --raw-dir <out>/raw_data ...
    <either>         python scripts/vis_episode_processed_wbc.py --dataset_dir ...

Writes, under `--out` (default ~/Dexmate/data/sim_<task>):

    raw_data/episode_<N>.hdf5              the take (schema: see obs_pipeline)
    raw_data/split.csv                     episode -> train/test (last --n-test are test)
    scene_diff/positions/raw/episode_<N>.npz
        the ENV-state condition. On hardware SceneDiff infers the two objects' world positions
        from a FRAME-0 image diff; in sim they are ground truth snapshotted at episode START
        (after reset, before any expert motion) via `objects_of_interest()`, then mapped into
        the engage-origin world -- same file format (`positions` (2,3) [src, dst], `order`
        identity, `frame` "world"), so `--positions-dir` works unchanged. Do NOT snapshot
        after success: the source object then coincides with the destination.

Episodes are recorded at `--fps` (default 10, the dataset rate) by subsampling the 100 Hz WBC
control loop, and FAILED episodes are dropped by default (`--keep-failures` to keep them).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
import argparse
import csv
import dataclasses
import json
import numpy as np

import omnigibson as og

from vega_og_env import VegaOGEnv
from obs_pipeline import SimObsRecorder
import zed_sim
from tasks import TASKS

P = lambda *a: print(*a, flush=True)


def save_hdf5(path, frames, static):
    """Stack per-frame dicts and write the nested HDF5 (mirrors common/recorder.py)."""
    import h5py

    def stack(dicts):
        out = {}
        for key, val in dicts[0].items():
            out[key] = stack([d[key] for d in dicts]) if isinstance(val, dict) \
                else np.stack([np.asarray(d[key]) for d in dicts])
        return out

    def merge(dst, src):
        for key, val in src.items():
            if isinstance(val, dict):
                merge(dst.setdefault(key, {}), val)
            else:
                dst[key] = np.asarray(val)

    data = stack(frames)
    merge(data, static)

    def write(group, prefix, sub):
        for key, item in sub.items():
            if isinstance(item, dict):
                write(group, prefix + key + "/", item)
            else:
                group.create_dataset(prefix + key, data=item)

    with h5py.File(path, "w") as f:
        write(f, "/", data)


def run_episode(env, task, recorder, seed, fps, max_ticks, action_hz):
    """One scripted episode -> (frames, success, src_og, dst_og).

    ``src_og`` / ``dst_og`` are the FRAME-0 objects-of-interest in OmniGibson world,
    snapshotted after reset and before any expert motion (matches hardware SceneDiff).
    """
    env.reset(seed=seed)
    recorder.setup()
    src_dst = np.asarray(task.objects_of_interest(env), dtype=np.float64)
    if src_dst.shape != (2, 3) or not np.all(np.isfinite(src_dst)):
        raise ValueError(f"objects_of_interest must be finite (2,3), got {src_dst.shape}")
    every = max(1, int(round(action_hz / fps)))
    dt_ns = int(round(1e9 / fps))
    frames, t_ns = [], 0
    for i in range(max_ticks):
        cmd = task.expert_step(env, {})
        resp = env.wbc_tick(cmd.left_target, cmd.right_target, head_og=cmd.head_target,
                            grip_l=cmd.gripper_left, grip_r=cmd.gripper_right)
        if i % every == 0:
            frames.append(recorder.frame(cmd, t_ns, wbc_resp=resp))
            t_ns += dt_ns
        if cmd.done:
            break
    return frames, bool(task.success(env)), src_dst[0].copy(), src_dst[1].copy()


def write_positions(path, src_wbc, dst_wbc):
    """SceneDiff-format env-state npz; sim GT replaces the image-diff estimate.

    ``src``/``dst`` must be the pre-manipulation (frame-0) poses. A near-coincident pair
    usually means the caller snapshotted after success (source already on destination).
    """
    src = np.asarray(src_wbc, dtype=np.float32).reshape(3)
    dst = np.asarray(dst_wbc, dtype=np.float32).reshape(3)
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        raise ValueError(f"{path}: non-finite src/dst positions")
    sep = float(np.linalg.norm(src - dst))
    if sep < 0.05:
        raise ValueError(
            f"{path}: src/dst only {sep * 100:.1f} cm apart — likely snapshotted after "
            "success (source already on destination). Snapshot at episode start."
        )
    np.savez(path, positions=np.stack([src, dst]),
             order=np.array([0, 1], dtype=np.int64), frame="world")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mobilepickplace", choices=list(TASKS))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default=None, help="dataset root (default ~/Dexmate/data/sim_<task>)")
    ap.add_argument("--fps", type=int, default=10, help="dataset rate (porter --fps must match)")
    ap.add_argument("--max-ticks", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=2, help="last N episodes -> test split")
    ap.add_argument("--port", type=int, default=5640)
    ap.add_argument("--obs-hw", type=int, nargs=2, default=None,
                    help="raw head render size (default: whatever the zed realism switches "
                         "imply -- zed_sim.head_render_hw)")
    ap.add_argument("--keep-failures", action="store_true")
    zed_sim.ZedSimOptions.add_cli(ap)
    args = ap.parse_args()
    zed_opts = zed_sim.ZedSimOptions.from_args(args)
    obs_hw = tuple(args.obs_hw) if args.obs_hw else zed_sim.head_render_hw(zed_opts)

    root = args.out or os.path.expanduser(f"~/Dexmate/data/sim_{args.task}")
    raw_dir = os.path.join(root, "raw_data")
    pos_dir = os.path.join(root, "scene_diff", "positions", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(pos_dir, exist_ok=True)

    task = TASKS[args.task](rng=np.random.default_rng(args.seed0))
    mobile = getattr(task, "MOBILE", False)
    action_hz = 100
    env = VegaOGEnv(task=task, lock_base=not mobile, mobile=mobile, wbc_port=args.port,
                    pos_kp=4000, action_hz=action_hz, obs_hw=obs_hw,
                    grasping_mode=task.GRASPING_MODE,
                    robot_pos=task.ROBOT_POS, robot_yaw=task.ROBOT_YAW)
    if mobile:
        env.base_x_max = getattr(task, "BASE_X_MAX", None)
    recorder = SimObsRecorder(env, seed=args.seed0, zed=zed_opts)

    kept, xyz_lo, xyz_hi = [], None, None
    for k in range(args.episodes):
        seed = args.seed0 + k
        task.rng = np.random.default_rng(seed)
        recorder._rng = np.random.default_rng(seed)   # depth dropouts reproducible per episode
        frames, success, src, dst = run_episode(env, task, recorder, seed, args.fps,
                                                 args.max_ticks, action_hz)
        if not success and not args.keep_failures:
            P(f"[collect] seed {seed}: FAILED -- dropped ({len(frames)} frames)")
            continue
        idx = len(kept)
        to_wbc = lambda p: (env.T_align_inv @ np.append(p, 1.0))[:3]
        src_wbc, dst_wbc = to_wbc(src), to_wbc(dst)
        save_hdf5(os.path.join(raw_dir, f"episode_{idx}.hdf5"), frames,
                  {"obs": {"images": {"intrinsic": recorder.intrinsic()}}})
        write_positions(os.path.join(pos_dir, f"episode_{idx}.npz"), src_wbc, dst_wbc)
        kept.append(idx)
        # track the object envelope so the ManiFlow --crop-min/--crop-max can be set from data
        for p in (src_wbc, dst_wbc):
            xyz_lo = p if xyz_lo is None else np.minimum(xyz_lo, p)
            xyz_hi = p if xyz_hi is None else np.maximum(xyz_hi, p)
        P(f"[collect] seed {seed}: SUCCESS -> episode_{idx}.hdf5  ({len(frames)} frames) "
          f"src@{np.round(src_wbc,3)} dst@{np.round(dst_wbc,3)} "
          f"|src-dst|={np.linalg.norm(src_wbc - dst_wbc)*100:.1f}cm")

    with open(os.path.join(raw_dir, "split.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "split"])
        for i in kept:
            w.writerow([i, "test" if i >= len(kept) - args.n_test else "train"])

    meta = {"task": args.task, "episodes": len(kept), "fps": args.fps,
            "obs_hw": list(obs_hw), "seed0": args.seed0,
            "zed_realism": dataclasses.asdict(zed_opts),
            "object_world_min": None if xyz_lo is None else xyz_lo.round(3).tolist(),
            "object_world_max": None if xyz_hi is None else xyz_hi.round(3).tolist()}
    with open(os.path.join(root, "sim_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    P(f"[collect] kept {len(kept)}/{args.episodes} episodes -> {raw_dir}")
    P(f"[collect] objects (engage-origin world) span {meta['object_world_min']} .. "
      f"{meta['object_world_max']}  (set the zarr --crop-min/--crop-max around this)")
    if zed_opts.match_zed_fov:
        # The real ZED's wider field puts fewer pixels on the workspace, so a crop fitted to the
        # objects alone starves the cloud below the 1024-point FPS budget; it then has to span the
        # EEF targets too (the grasped object is carried back toward the robot).
        P("[collect] NOTE --match-zed-fov: widen the crop to cover the EEF targets, e.g. "
          "--crop-min 0.35 -0.52 0.55 --crop-max 1.65 0.46 1.04 for mobilepickplace")
    env.close()


if __name__ == "__main__":
    main()
