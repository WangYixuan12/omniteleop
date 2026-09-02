#!/usr/bin/env python3
"""Render the ManiFlow WBC world-frame point cloud with SceneDiff + EEF overlays.

Durable form of the scratchpad harness that produced ``pc_final.png``. Builds the
SAME cropped + FPS-resampled cloud that ``omniteleop.wbc_pointcloud`` /
``port_wbc_mobile_zarr.py`` write into the zarr, then overlays SceneDiff object
positions and world-frame EEF targets so you can confirm the cloud, the action
frame, and the object positions share one engage-origin world.

Two inputs are independent:

* ``--hdf5`` / ``--raw-dir``  — which episode(s) supply the RGB-D cloud + EEF
* ``--positions-dir``         — SceneDiff ``*.npz`` overlay(s)

By default each plot shows ONLY that episode's SceneDiff positions (matched by
stem, e.g. ``episode_1.hdf5`` <-> ``episode_1.npz``). Pass ``--overlay-all-
positions`` to scatter every episode's objects on every subplot (the original
scratchpad behaviour — useful for inspecting the crop envelope, misleading for
per-take alignment).

Outputs default to ``~/Dexmate/data/visualization/``.

Run in the ``dexmate`` conda env (pinocchio for ``WBCPolicyFK``, torch for FPS)::

    # one episode (matched SceneDiff objects only)
    python scripts/diagnostics/render_wbc_pointcloud.py \
        --hdf5 ~/Dexmate/data/raw_data/episode_1.hdf5

    # every raw take -> one PNG each
    python scripts/diagnostics/render_wbc_pointcloud.py \
        --raw-dir ~/Dexmate/data/raw_data

    # crop-envelope check: scatter ALL episodes' SceneDiff objects on every subplot
    python scripts/diagnostics/render_wbc_pointcloud.py \
        --hdf5 ~/Dexmate/data/raw_data/episode_1.hdf5 \
        --overlay-all-positions
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from omniteleop.wbc_pointcloud import (
    CROP_MAX,
    CROP_MIN,
    build_world_cloud,
    resample_clouds,
    world_t_head_from_state,
)
from omniteleop.wbc_policy_format import WBCPolicyFK

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = Path("~/Dexmate/data/raw_data").expanduser()
DEFAULT_POSITIONS_DIR = Path("~/Dexmate/data/scene_diff/positions/raw").expanduser()
DEFAULT_OUT_DIR = Path("~/Dexmate/data/visualization").expanduser()
DEFAULT_NUM_POINTS = 4096
DEFAULT_POOL_SIZE = 16384
_EPISODE_RE = re.compile(r"^episode_(\d+)$")


def _load_hdf5_porter():
    path = _SCRIPTS_DIR / "port_wbc_mobile_hdf5.py"
    spec = importlib.util.spec_from_file_location("port_wbc_mobile_hdf5", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import the HDF5 porter at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_frames(spec: str, n_frames: int) -> list[int]:
    """``'0,mid,last'`` / ``'0,292,584'`` -> frame indices in ``[0, n_frames)``."""
    out: list[int] = []
    for token in spec.split(","):
        token = token.strip().lower()
        if token in ("mid", "middle"):
            out.append(n_frames // 2)
        elif token in ("last", "end"):
            out.append(n_frames - 1)
        else:
            out.append(int(token))
    for t in out:
        if not 0 <= t < n_frames:
            raise ValueError(f"frame {t} out of range for episode with {n_frames} frames")
    return out


def _episode_stem(hdf5_path: Path) -> str:
    stem = hdf5_path.stem
    if not _EPISODE_RE.match(stem):
        raise ValueError(f"expected episode_<N>.hdf5, got {hdf5_path.name}")
    return stem


def _discover_episodes(raw_dir: Path) -> list[Path]:
    paths = sorted(
        raw_dir.glob("episode_*.hdf5"),
        key=lambda p: int(_EPISODE_RE.match(p.stem).group(1)),  # type: ignore[union-attr]
    )
    return [p for p in paths if _EPISODE_RE.match(p.stem)]


def _load_scenediff_positions_file(path: Path) -> np.ndarray:
    """One SceneDiff npz -> ``(N_obj, 3)`` world points after ``positions[order]``."""
    with np.load(path) as data:
        if "positions" not in data or "order" not in data:
            raise KeyError(f"{path} missing 'positions'/'order'")
        pos = np.asarray(data["positions"])[np.asarray(data["order"])]
    if pos.ndim != 2 or pos.shape[-1] != 3:
        raise ValueError(f"{path}: expected (N, 3) after order, got {pos.shape}")
    return pos


def _load_all_scenediff_positions(positions_dir: Path) -> np.ndarray:
    """Stack every ``*.npz`` -> ``(E, N_obj, 3)``. Empty dir -> ``(0, 0, 3)``."""
    files = sorted(positions_dir.glob("*.npz"))
    if not files:
        return np.zeros((0, 0, 3), dtype=np.float64)
    rows = [_load_scenediff_positions_file(p) for p in files]
    n_obj = {r.shape[0] for r in rows}
    if len(n_obj) != 1:
        raise ValueError(f"inconsistent object counts across SceneDiff files: {n_obj}")
    return np.stack(rows)


def _resolve_overlay_positions(
    positions_dir: Path,
    episode_stem: str,
    *,
    overlay_all: bool,
) -> np.ndarray:
    """Return ``(E, N_obj, 3)`` to scatter: matched episode only, or every take."""
    if overlay_all:
        return _load_all_scenediff_positions(positions_dir)
    matched = positions_dir / f"{episode_stem}.npz"
    if not matched.is_file():
        print(f"  warning: no SceneDiff file {matched.name}; plotting without objects")
        return np.zeros((0, 0, 3), dtype=np.float64)
    return _load_scenediff_positions_file(matched)[None, ...]


def _build_frame_clouds(
    hdf5_path: Path,
    frames: list[int],
    fk: WBCPolicyFK,
    porter,
) -> tuple[list[np.ndarray], list[tuple[int, np.ndarray, np.ndarray, float, int]]]:
    """Return cropped clouds + per-frame ``(t, world_T_zed, action, recon_err, n_pts)``."""
    raw_clouds: list[np.ndarray] = []
    metas: list[tuple[int, np.ndarray, np.ndarray, float, int]] = []
    with h5py.File(hdf5_path, "r") as f:
        intrinsic = np.asarray(f["obs/images/intrinsic"][()], dtype=np.float64)
        for t in frames:
            action_eef, action_head = porter.load_action_targets_world(f, t)
            state, action = porter.compute_frame_state_action(
                f, t, fk, action_eef, action_head, state_frame="base"
            )
            world_t_zed = world_t_head_from_state(state, "base")
            state_w, _ = porter.compute_frame_state_action(
                f, t, fk, action_eef, action_head, state_frame="world"
            )
            recon_err = float(
                np.abs(world_t_zed - world_t_head_from_state(state_w, "world")).max()
            )
            cloud = build_world_cloud(
                np.asarray(f["obs/images/head_depth"][t]),
                np.asarray(f["obs/images/head_left_rgb"][t]),
                intrinsic,
                world_t_zed,
            )
            raw_clouds.append(cloud)
            metas.append((t, world_t_zed, action, recon_err, int(cloud.shape[0])))
    return raw_clouds, metas


def _render(
    final: np.ndarray,
    metas: list[tuple[int, np.ndarray, np.ndarray, float, int]],
    allpos: np.ndarray,
    num_points: int,
    out_path: Path,
    *,
    title_prefix: str = "",
) -> None:
    n = len(metas)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9), squeeze=False)
    obj_labels = ("box", "cloth")
    obj_colors = ("orange", "lime")
    for j, (t, world_t_zed, action, _, _) in enumerate(metas):
        pts = final[j, :, :3]
        colors = np.clip(final[j, :, 3:], 0.0, 1.0)
        cam = world_t_zed[:3, 3]
        eef_l, eef_r = action[0:3], action[10:13]
        for row, (i0, i1, label) in enumerate(
            [(0, 1, "top-down (X-Y)"), (0, 2, "side (X-Z)")]
        ):
            ax = axes[row, j]
            ax.scatter(pts[:, i0], pts[:, i1], c=colors, s=3, linewidths=0)
            ax.scatter(cam[i0], cam[i1], c="k", marker="^", s=90, label="camera", zorder=5)
            ax.scatter(eef_l[i0], eef_l[i1], c="r", marker="o", s=70, label="L eef", zorder=5)
            ax.scatter(eef_r[i0], eef_r[i1], c="b", marker="o", s=70, label="R eef", zorder=5)
            for obj_i in range(allpos.shape[1]):
                name = obj_labels[obj_i] if obj_i < len(obj_labels) else f"obj{obj_i}"
                color = obj_colors[obj_i] if obj_i < len(obj_colors) else "magenta"
                ax.scatter(
                    allpos[:, obj_i, i0],
                    allpos[:, obj_i, i1],
                    c=color,
                    marker="*",
                    s=45,
                    label=name,
                    zorder=4,
                )
            prefix = f"{title_prefix} " if title_prefix else ""
            ax.set_title(f"{prefix}frame {t}  {label}  [{num_points} pts, FPS]")
            ax.set_aspect("equal")
            if row == 0 and j == 0:
                ax.legend(fontsize=6, loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=95)
    plt.close(fig)


def _render_episode(
    hdf5_path: Path,
    *,
    positions_dir: Path,
    overlay_all: bool,
    frames_spec: str,
    num_points: int,
    pool_size: int,
    device: str,
    out_path: Path,
    fk: WBCPolicyFK,
    porter,
) -> None:
    stem = _episode_stem(hdf5_path)
    with h5py.File(hdf5_path, "r") as f:
        n_frames = int(f["obs/images/head_depth"].shape[0])
    frames = _parse_frames(frames_spec, n_frames)
    print(f"\n=== {stem}  T={n_frames}  frames={frames}")

    allpos = _resolve_overlay_positions(
        positions_dir, stem, overlay_all=overlay_all
    )
    if allpos.size:
        print(
            f"  SceneDiff overlay: {allpos.shape[0]} take(s) x {allpos.shape[1]} objects"
            + (" (ALL episodes)" if overlay_all else f" (matched {stem}.npz)")
        )

    raw_clouds, metas = _build_frame_clouds(hdf5_path, frames, fk, porter)
    t0 = time.time()
    final = resample_clouds(
        raw_clouds,
        num_points,
        pool_size=pool_size,
        rng=np.random.default_rng(0),
        device=device,
    )
    print(
        f"  resample_clouds -> {final.shape}  ({time.time() - t0:.2f}s)"
    )
    inside = np.all(
        (final[..., :3] > np.array(CROP_MIN)) & (final[..., :3] < np.array(CROP_MAX)),
        axis=-1,
    )
    print(f"  all points inside crop: {bool(inside.all())}")
    for i, (t, _, _, recon_err, n_pts) in enumerate(metas):
        print(
            f"  frame {t:3d}: cropped {n_pts:6d} -> {num_points}, "
            f"world_T_zed recon err {recon_err:.1e}"
        )

    _render(
        final,
        metas,
        allpos,
        num_points,
        out_path,
        title_prefix=stem,
    )
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--hdf5",
        type=Path,
        default=None,
        help="single raw episode HDF5 (default if neither --hdf5 nor --raw-dir)",
    )
    src.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="process every episode_*.hdf5 in this directory",
    )
    parser.add_argument(
        "--positions-dir",
        type=Path,
        default=DEFAULT_POSITIONS_DIR,
        help="SceneDiff positions/raw directory (*.npz with positions+order)",
    )
    parser.add_argument(
        "--overlay-all-positions",
        action="store_true",
        help="scatter EVERY episode's SceneDiff objects on every subplot "
        "(crop-envelope check). Default: only the matched episode's objects.",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default="0,mid,last",
        help="comma-separated frame indices, or mid/last tokens (default: 0,mid,last)",
    )
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="torch device for farthest-point sampling (cuda|cpu)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output PNG for --hdf5 mode "
        f"(default: {DEFAULT_OUT_DIR}/<stem>_pc_final.png)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"output directory for --raw-dir mode (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    if args.num_points > args.pool_size:
        raise ValueError(f"num_points {args.num_points} > pool_size {args.pool_size}")

    default_out_dir = DEFAULT_OUT_DIR
    default_out_dir.mkdir(parents=True, exist_ok=True)

    if args.raw_dir is not None:
        raw_dir = args.raw_dir.expanduser()
        if not raw_dir.is_dir():
            raise FileNotFoundError(raw_dir)
        episodes = _discover_episodes(raw_dir)
        if not episodes:
            raise FileNotFoundError(f"no episode_*.hdf5 under {raw_dir}")
        out_dir = (args.out_dir or default_out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"rendering {len(episodes)} episodes from {raw_dir} -> {out_dir}")
    else:
        hdf5 = (args.hdf5 or (DEFAULT_RAW_DIR / "episode_1.hdf5")).expanduser()
        if not hdf5.is_file():
            raise FileNotFoundError(hdf5)
        episodes = [hdf5]
        out_dir = None

    porter = _load_hdf5_porter()
    fk = WBCPolicyFK()

    for hdf5_path in episodes:
        stem = _episode_stem(hdf5_path)
        if out_dir is not None:
            out_path = out_dir / f"{stem}_pc_final.png"
        else:
            out_path = (
                args.out or (default_out_dir / f"{stem}_pc_final.png")
            ).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
        _render_episode(
            hdf5_path,
            positions_dir=args.positions_dir.expanduser(),
            overlay_all=args.overlay_all_positions,
            frames_spec=args.frames,
            num_points=args.num_points,
            pool_size=args.pool_size,
            device=args.device,
            out_path=out_path,
            fk=fk,
            porter=porter,
        )


if __name__ == "__main__":
    main()
