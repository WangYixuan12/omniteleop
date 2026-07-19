#!/usr/bin/env python3
"""Offline GT-vs-predicted validation of a WBC MoF checkpoint.

The MoF sibling of ``scripts/vis_wbc_policy_prediction.py``: runs the deploy
bundle's EXACT inference path (``MoFBundle.predict_chunk`` on the stored port
observations) over a ported safetensors episode and compares the predicted
29-D world actions against the recorded ground truth. This is the gate before
any hardware rollout.

Outputs, per entity (left / right / head):
- translation MAE (m) and geodesic rotation error (deg) over all predicted
  steps, printed to stdout;
- an overlay figure: GT (solid) vs 1-step-ahead prediction (dashed) position
  traces per axis, rotation-error series, and the gripper command series.
- a Rerun recording with the world-frame action horizons, the same final mixed
  action expressed in every MoF representation, and the learned router weights
  across diffusion denoising steps. Use ``--save FILE.rrd`` over SSH.

Environment: ``dexmate_mof``.

Usage::

    python scripts/vis_wbc_mof_prediction.py \
      --policy-path ~/mofpo/data/outputs/wbc_v1_mof_moe_500ep/checkpoints/latest.ckpt \
      --episode .../processed_wbc/mof/dexmate_wbc_mof/test/episode_0.safetensors \
      --output ~/Dexmate/data/visualization/mof_pred_test0.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from mof.common.mof_transform_util import (
    convert_base_pose_action_to_rel_traj_generic,
    convert_base_pose_action_to_rel_trans_generic,
    convert_world_action_to_frame,
    transform_obs_pose_dict_to_frame,
    world_frame_to_transform,
)
from safetensors.numpy import load_file

from omniteleop.wbc_mof_bundle import MoFBundle
from omniteleop.wbc_policy_format import GRIPPER_BINARY_THRESHOLD

ENTITIES = {
    "left": (slice(0, 3), slice(3, 9), 9),
    "right": (slice(10, 13), slice(13, 19), 19),
    "head": (slice(20, 23), slice(23, 29), None),
}
REPRESENTATIONS = ("base", "base_rel_trans", "left", "right", "rel_traj")

_COLOR_GT = (255, 0, 0)
_COLOR_PRED = (230, 60, 230)
_COLOR_ERR = (235, 235, 235)
_COLOR_BASE = (230, 190, 0)
_ROUTER_COLORS = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
)
_PLAYBACK_FPS = 10.0
DEFAULT_EPISODE = Path(
    "~/Dexmate/data/box2cloth/processed_wbc/mof/dexmate_wbc_mof/test/episode_0.safetensors"
).expanduser()


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) column rot6d -> (..., 3, 3); columns are stored orthonormal."""
    c1, c2 = rot6d[..., 0:3], rot6d[..., 3:6]
    c1 = c1 / np.linalg.norm(c1, axis=-1, keepdims=True)
    c2 = c2 - np.sum(c1 * c2, axis=-1, keepdims=True) * c1
    c2 = c2 / np.linalg.norm(c2, axis=-1, keepdims=True)
    return np.stack([c1, c2, np.cross(c1, c2)], axis=-1)


def _geodesic_deg(rot6d_a: np.ndarray, rot6d_b: np.ndarray) -> np.ndarray:
    Ra, Rb = _rot6d_to_matrix(rot6d_a), _rot6d_to_matrix(rot6d_b)
    tr = np.einsum("...ij,...ij->...", Ra, Rb)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def evaluate(
    bundle: MoFBundle, arrays: dict[str, np.ndarray], stride: int
) -> tuple[
    list[tuple[int, np.ndarray, np.ndarray]],
    list[np.ndarray],
    np.ndarray,
]:
    T = arrays["action"].shape[0]
    n_obs, n_act = bundle.n_obs_steps, bundle.n_action_steps
    records = []  # (t, gt (n_act, 29), pred (n_act, 29))
    router_traces: list[np.ndarray] = []
    captured_router_probs = []

    def _capture_router(_module, _inputs, output) -> None:
        if not isinstance(output, dict) or "router_probs" not in output:
            return
        probs = output["router_probs"].detach().cpu().numpy()
        if probs.shape[0] != 1 or probs.ndim != 2:
            raise RuntimeError(f"unexpected inference router shape {probs.shape}")
        captured_router_probs.append(np.asarray(probs[0], dtype=np.float32))

    hook = bundle.policy.model.register_forward_hook(_capture_router)
    try:
        for t in range(n_obs - 1, T - n_act, stride):
            history = [bundle.obs_from_port_arrays(arrays, i) for i in range(t - n_obs + 1, t + 1)]
            trace_start = len(captured_router_probs)
            pred = bundle.predict_chunk(history)
            trace = np.stack(captured_router_probs[trace_start:])
            if trace.shape[0] != bundle.num_inference_steps:
                raise RuntimeError(
                    f"captured {trace.shape[0]} router steps, expected {bundle.num_inference_steps}"
                )
            router_traces.append(trace)

            # Match the shared rollout: grippers become {0,1} commands at
            # GRIPPER_BINARY_THRESHOLD before actuation (split_policy_action).
            pred = np.asarray(pred, dtype=np.float32).copy()
            for _, _, grip_ch in ENTITIES.values():
                if grip_ch is not None:
                    pred[..., grip_ch] = (pred[..., grip_ch] >= GRIPPER_BINARY_THRESHOLD).astype(
                        np.float32
                    )
            gt = arrays["action"][t : t + n_act]
            records.append((t, gt, pred))
    finally:
        hook.remove()
    if not records:
        raise SystemExit(f"episode too short ({T} frames) for a single chunk")
    timesteps = bundle.policy.noise_scheduler.timesteps.detach().cpu().numpy().copy()
    return records, router_traces, timesteps


def report(records: list[tuple[int, np.ndarray, np.ndarray]]) -> dict:
    gt = np.stack([r[1] for r in records])  # (K, n_act, 29)
    pred = np.stack([r[2] for r in records])
    metrics: dict[str, float] = {}
    print(f"chunks evaluated: {len(records)} x {gt.shape[1]} steps")
    for name, (pos_sl, rot_sl, grip_ch) in ENTITIES.items():
        t_mae = float(np.abs(pred[..., pos_sl] - gt[..., pos_sl]).mean())
        r_err = float(_geodesic_deg(pred[..., rot_sl], gt[..., rot_sl]).mean())
        metrics[f"{name}/translation_mae_m"] = t_mae
        metrics[f"{name}/rotation_err_deg"] = r_err
        line = f"  {name:>5}: translation MAE {t_mae * 100:6.2f} cm | rotation err {r_err:6.2f} deg"
        if grip_ch is not None:
            # pred grippers are already the rollout's {0,1} commands (see evaluate).
            acc = float((pred[..., grip_ch] == gt[..., grip_ch]).mean())
            metrics[f"{name}/gripper_acc"] = acc
            line += f" | gripper acc {acc * 100:5.1f}%"
        print(line)
    return metrics


def render(records: list[tuple[int, np.ndarray, np.ndarray]], out_path: Path) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ts = np.array([r[0] for r in records])
    gt1 = np.stack([r[1][0] for r in records])  # first predicted step vs GT
    pred1 = np.stack([r[2][0] for r in records])

    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    for row, (name, (pos_sl, rot_sl, _grip_ch)) in enumerate(ENTITIES.items()):
        for col, axis_name in enumerate("xyz"):
            ax = axes[row][col]
            ch = pos_sl.start + col
            ax.plot(ts, gt1[:, ch], "-", color="tab:red", label="GT")
            ax.plot(ts, pred1[:, ch], "--", color="tab:purple", label="pred")
            ax.set_ylabel(f"{name} {axis_name} [m]")
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=8)
        rot_err = _geodesic_deg(pred1[:, rot_sl], gt1[:, rot_sl])
        axes[3][list(ENTITIES).index(name)].plot(ts, rot_err, color="tab:orange")
        axes[3][list(ENTITIES).index(name)].set_ylabel(f"{name} rot err [deg]")
        axes[3][list(ENTITIES).index(name)].set_xlabel("frame")
    grip_ax = axes[0][2].twinx()
    for name, (_, _, grip_ch) in ENTITIES.items():
        if grip_ch is None:
            continue
        grip_ax.step(ts, gt1[:, grip_ch], where="post", alpha=0.35, label=f"{name} grip GT")
        grip_ax.step(
            ts, pred1[:, grip_ch], where="post", alpha=0.35, ls="--", label=f"{name} grip pred"
        )
    grip_ax.set_ylim(-0.1, 1.1)
    grip_ax.legend(loc="lower right", fontsize=6)
    fig.suptitle("WBC MoF: GT vs 1-step-ahead predicted world action")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"overlay figure -> {out_path}")


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert one wxyz quaternion to a rotation matrix."""
    q = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"invalid quaternion {q}")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_matrix(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _quat_wxyz_to_matrix(quat)
    out[:3, 3] = np.asarray(pos, dtype=np.float64)
    return out


def _project_world_to_pixel(
    pts_world: np.ndarray,
    intrinsic: np.ndarray,
    world_t_cam: np.ndarray,
    z_min: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
    cam_t_world = np.linalg.inv(world_t_cam)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    pts_cam = (cam_t_world @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    valid = z > z_min
    z_safe = np.where(valid, z, 1.0)
    u = intrinsic[0, 0] * pts_cam[:, 0] / z_safe + intrinsic[0, 2]
    v = intrinsic[1, 1] * pts_cam[:, 1] / z_safe + intrinsic[1, 2]
    return np.stack([u, v], axis=-1).astype(np.float32), valid


def _scale_intrinsic(
    intrinsic: np.ndarray,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Scale a pinhole intrinsic across an anisotropic image resize."""
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    out = np.asarray(intrinsic, dtype=np.float64).copy()
    out[0, :] *= target_w / source_w
    out[1, :] *= target_h / source_h
    return out


def _unproject_depth(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    world_t_cam: np.ndarray,
    max_depth: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth_m.shape
    v, u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    mask = (depth_m > 0.05) & (depth_m < max_depth)
    z = depth_m[mask]
    x = (u[mask] - intrinsic[0, 2]) / intrinsic[0, 0] * z
    y = (v[mask] - intrinsic[1, 2]) / intrinsic[1, 1] * z
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)
    return (world_t_cam @ pts_cam.T).T[:, :3].astype(np.float32), mask


def _voxel_downsample(
    pts: np.ndarray, colors: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-average on CUDA; deterministic bounded subsample otherwise."""
    if pts.shape[0] == 0:
        return pts, colors
    try:
        pts_t = torch.from_numpy(pts).cuda()
        colors_t = torch.from_numpy(colors.astype(np.float32)).cuda()
        voxels = torch.floor(pts_t / voxel_size).int()
        keys = voxels[:, 0] * (1 << 20) + voxels[:, 1] * (1 << 10) + voxels[:, 2]
        _, inv = torch.unique(keys, return_inverse=True)
        out_pts = torch.zeros(inv.max() + 1, 3, device="cuda")
        out_colors = torch.zeros(inv.max() + 1, 3, device="cuda")
        counts = torch.zeros(inv.max() + 1, device="cuda")
        out_pts.scatter_add_(0, inv.unsqueeze(1).expand(-1, 3), pts_t)
        out_colors.scatter_add_(0, inv.unsqueeze(1).expand(-1, 3), colors_t)
        counts.scatter_add_(0, inv, torch.ones(len(inv), device="cuda"))
        return (out_pts / counts.unsqueeze(1)).cpu().numpy(), (
            out_colors / counts.unsqueeze(1)
        ).cpu().numpy().astype(np.uint8)
    except Exception:
        count = min(len(pts), 50_000)
        indices = np.linspace(0, len(pts) - 1, count, dtype=np.int64)
        return pts[indices], colors[indices]


def _episode_meta_entry(meta_path: Path, episode: Path) -> dict | None:
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[rerun] cannot read {meta_path}; trying sidecar filename fallback: {exc}")
        return None
    try:
        relative = episode.resolve().relative_to(meta_path.parent.resolve()).as_posix()
    except ValueError:
        relative = f"{episode.parent.name}/{episode.name}"
    entry = next((item for item in meta.get("episodes", []) if item.get("file") == relative), None)
    if entry is None:
        print(f"[rerun] {relative!r} is not listed in {meta_path}; trying filename fallback")
    return entry


def _align_sidecar_frames(
    array: np.ndarray,
    frame_count: int,
    entry: dict | None,
    label: str,
) -> np.ndarray:
    if array.shape[0] == frame_count:
        return array
    if entry is not None:
        start, end = map(int, entry.get("window", (0, 0)))
        if end - start == frame_count and array.shape[0] >= end:
            return array[start:end]
    raise ValueError(f"{label} has {array.shape[0]} frames, expected {frame_count}")


def _load_sidecars(
    episode: Path,
    meta_path: Path,
    sidecar_root: Path | None,
    frame_count: int,
) -> dict[str, np.ndarray]:
    entry = _episode_meta_entry(meta_path, episode)
    split = str(entry["split"]) if entry is not None else episode.parent.name
    if entry is not None:
        episode_index = int(entry["processed_index"])
    else:
        try:
            episode_index = int(episode.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return {}

    candidates = []
    if sidecar_root is not None:
        candidates.append(sidecar_root)
    dataset_root = meta_path.parent
    if dataset_root.parent.name == "mof":
        processed_root = dataset_root.parent.parent
        candidates.extend(
            [
                processed_root / split / "dexmate_wbc_eef_head",
                processed_root / split / "dexmate_wbc_eef_head_rel",
            ]
        )
    root = next((path for path in candidates if (path / "debug").is_dir()), None)
    if root is None:
        checked = ", ".join(str(path) for path in candidates) or "no candidate roots"
        print(f"[rerun] no calibration/depth sidecars ({checked}); using RGB + poses only")
        return {}

    suffix = f"episode_{episode_index:06d}.npz"
    result: dict[str, np.ndarray] = {}
    calib_path = root / "debug" / "calib" / suffix
    try:
        with np.load(calib_path) as calib:
            result["extrinsic"] = _align_sidecar_frames(
                calib["extrinsic"].astype(np.float64), frame_count, entry, "extrinsic"
            )
            result["intrinsic"] = _align_sidecar_frames(
                calib["intrinsic"].astype(np.float64), frame_count, entry, "intrinsic"
            )
    except (OSError, KeyError, ValueError) as exc:
        print(f"[rerun] calibration unavailable at {calib_path}: {exc}")

    depth_path = root / "debug" / "depth" / suffix
    try:
        with np.load(depth_path) as depth_file:
            depth = _align_sidecar_frames(depth_file["depth"], frame_count, entry, "depth")
        result["depth"] = depth
    except (OSError, KeyError, ValueError) as exc:
        print(f"[rerun] depth unavailable at {depth_path}: {exc}")

    if "intrinsic" in result and "depth" not in result:
        print("[rerun] calibrated 2D overlay disabled: native intrinsic image size is unknown")
        result.pop("intrinsic")

    if result:
        print(f"[rerun] sidecars -> {root}")
    return result


def _mof_representations(
    bundle: MoFBundle,
    arrays: dict[str, np.ndarray],
    t: int,
    actions_world: np.ndarray,
) -> dict[str, np.ndarray]:
    """Express final world actions exactly like dataset precompute, without normalization."""
    policy = bundle.policy
    layout = policy.action_layout
    device = bundle.device

    def _at(key: str):
        value = np.ascontiguousarray(arrays[key][t : t + 1][None]).astype(np.float32)
        return torch.from_numpy(value).to(device)

    world_obs = {}
    for entity in layout.entities:
        world_obs[entity.obs_pos_key] = _at(entity.obs_pos_key)
        world_obs[entity.obs_quat_key] = _at(entity.obs_quat_key)

    world_t_base = world_frame_to_transform(_at("base_pos"), _at("base_quat"), policy.quat_to_mat)
    pos_keys = tuple(entity.obs_pos_key for entity in layout.entities)
    quat_keys = tuple(entity.obs_quat_key for entity in layout.entities)
    base_obs = transform_obs_pose_dict_to_frame(
        world_obs,
        world_t_base,
        policy.quat_to_mat,
        policy.mat_to_quat,
        pos_keys=pos_keys,
        quat_keys=quat_keys,
    )

    actions = torch.from_numpy(np.ascontiguousarray(actions_world).astype(np.float32)).to(device)
    with torch.no_grad():
        base = convert_world_action_to_frame(actions, world_t_base, layout=layout)
        converted = {"base": base}
        for frame_name in ("left", "right"):
            entity = layout.entity(frame_name)
            world_t_frame = world_frame_to_transform(
                _at(entity.obs_pos_key), _at(entity.obs_quat_key), policy.quat_to_mat
            )
            converted[frame_name] = convert_world_action_to_frame(
                actions, world_t_frame, layout=layout
            )

        pos_refs = {entity.name: base_obs[entity.obs_pos_key][:, -1:] for entity in layout.entities}
        pose_refs = {
            entity.name: (
                base_obs[entity.obs_pos_key][:, -1:],
                base_obs[entity.obs_quat_key][:, -1:],
            )
            for entity in layout.entities
        }
        converted["base_rel_trans"] = convert_base_pose_action_to_rel_trans_generic(
            base, pos_refs, layout=layout
        )
        converted["rel_traj"] = convert_base_pose_action_to_rel_traj_generic(
            base, pose_refs, policy.quat_to_mat, layout=layout
        )
    return {
        name: value.detach().cpu().numpy().astype(np.float32) for name, value in converted.items()
    }


def _log_action_horizon(rr, root: str, action: np.ndarray, color) -> None:
    for name, (pos_sl, rot_sl, _grip_ch) in ENTITIES.items():
        pos = action[:, pos_sl]
        rotations = _rot6d_to_matrix(action[:, rot_sl])
        rr.log(
            f"{root}/{name}/horizon",
            rr.LineStrips3D([pos], colors=[color], radii=0.003),
        )
        rr.log(
            f"{root}/{name}/horizon_points",
            rr.Points3D(pos, colors=[color], radii=0.006),
        )
        rr.log(
            f"{root}/{name}/horizon_z_axes",
            rr.Arrows3D(
                origins=pos,
                vectors=rotations[..., :, 2] * 0.04,
                colors=[color],
                radii=0.001,
            ),
        )
        rr.log(
            f"{root}/{name}/current",
            rr.Transform3D(translation=pos[0], mat3x3=rotations[0]),
            rr.TransformAxes3D(axis_length=0.07),
        )


def _log_first_step_errors(rr, root: str, gt: np.ndarray, pred: np.ndarray) -> None:
    for name, (pos_sl, _rot_sl, _grip_ch) in ENTITIES.items():
        rr.log(
            f"{root}/{name}",
            rr.LineStrips3D(
                [np.stack([gt[0, pos_sl], pred[0, pos_sl]])],
                colors=[_COLOR_ERR],
                radii=0.0015,
            ),
        )


def render_rerun(
    bundle: MoFBundle,
    arrays: dict[str, np.ndarray],
    records: list[tuple[int, np.ndarray, np.ndarray]],
    router_traces: list[np.ndarray],
    diffusion_timesteps: np.ndarray,
    episode: Path,
    sidecar_root: Path | None,
    save_path: Path | None,
    stride: int,
    voxel_size: float,
) -> None:
    frame_count = int(arrays["action"].shape[0])
    sidecars = _load_sidecars(
        episode=episode,
        meta_path=Path(bundle.port_meta_path),
        sidecar_root=sidecar_root,
        frame_count=frame_count,
    )

    rr.init("vis_wbc_mof_prediction", spawn=save_path is None)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(save_path))
        print(f"[rerun] recording -> {save_path}")

    gripper_views = [
        rrb.TimeSeriesView(
            origin=f"scene/gripper/{side}",
            name=f"{side.title()} gripper",
            axis_y=rrb.ScalarAxis(range=(-0.2, 1.1), zoom_lock=True),
        )
        for side in ("left", "right")
    ]
    scene_tab = rrb.Vertical(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="scene/world", name="World: GT (red) vs pred (magenta)"),
            rrb.Spatial2DView(origin="scene/world/camera", name="Head RGB / depth"),
            rrb.Spatial2DView(origin="scene/wrist", name="Wrist RGB"),
            column_shares=[2.0, 1.3, 1.3],
        ),
        rrb.Horizontal(*gripper_views),
        row_shares=[2.2, 1.0],
        name="Scene",
    )
    frame_tab = rrb.Grid(
        *[
            rrb.Spatial3DView(
                origin=f"mof_frames/{name}",
                name=(
                    name if name in ("base", "left", "right") else f"{name} (representation space)"
                ),
            )
            for name in REPRESENTATIONS
        ],
        rrb.TimeSeriesView(
            origin="router/mean",
            name="Mean expert weights by chunk",
            axis_y=rrb.ScalarAxis(range=(0.0, 1.0), zoom_lock=True),
        ),
        rrb.TensorView(
            origin="router/heatmap",
            name="Weights by denoising step",
            visible=False,
        ),
        rrb.TextDocumentView(
            origin="router/legend",
            name="Router interpretation",
            visible=False,
        ),
        grid_columns=3,
        name="MoF Frames",
    )
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Tabs(scene_tab, frame_tab),
            rrb.TimePanel(timeline="chunk", fps=_PLAYBACK_FPS / stride),
            auto_views=False,
        )
    )

    rr.log("scene/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for representation in REPRESENTATIONS:
        rr.log(
            f"mof_frames/{representation}",
            rr.ViewCoordinates.RIGHT_HAND_Z_UP,
            static=True,
        )
        rr.log(
            f"mof_frames/{representation}/reference_axes",
            rr.Transform3D(translation=[0.0, 0.0, 0.0], mat3x3=np.eye(3)),
            rr.TransformAxes3D(axis_length=0.12),
            static=True,
        )

    for side in ("left", "right"):
        rr.log(
            f"scene/gripper/{side}",
            rr.SeriesLines(
                colors=[_COLOR_GT, _COLOR_PRED],
                names=[f"{side} GT", f"{side} pred"],
                widths=[2.0, 2.0],
            ),
            static=True,
        )

    expert_names = tuple(bundle.policy.model.enabled_experts)
    expert_colors = [
        _ROUTER_COLORS[index % len(_ROUTER_COLORS)] for index in range(len(expert_names))
    ]
    rr.log(
        "router/mean",
        rr.SeriesLines(
            colors=expert_colors,
            names=list(expert_names),
            widths=[2.0] * len(expert_names),
        ),
        static=True,
    )
    timestep_text = ", ".join(str(int(value)) for value in diffusion_timesteps)
    rr.log(
        "router/legend",
        rr.TextDocument(
            "**Heatmap layout**\n\n"
            f"Rows (expert order): `{', '.join(expert_names)}`  \n"
            "Columns: diffusion inference order, high noise to low noise.  \n"
            f"Scheduler timesteps: `{timestep_text}`",
            media_type="text/markdown",
        ),
        static=True,
    )

    rr.log(
        "scene/world/base_path",
        rr.LineStrips3D([arrays["base_pos"]], colors=[_COLOR_BASE], radii=0.008),
        static=True,
    )

    for chunk_index, ((t, gt, pred), router_trace) in enumerate(
        zip(records, router_traces, strict=True)
    ):
        rr.set_time("chunk", sequence=chunk_index)

        _log_action_horizon(rr, "scene/world/chunk/gt", gt, _COLOR_GT)
        _log_action_horizon(rr, "scene/world/chunk/pred", pred, _COLOR_PRED)
        _log_first_step_errors(rr, "scene/world/chunk/error", gt, pred)

        world_t_base = _pose_matrix(arrays["base_pos"][t], arrays["base_quat"][t])
        rr.log(
            "scene/world/base",
            rr.Transform3D(translation=world_t_base[:3, 3], mat3x3=world_t_base[:3, :3]),
            rr.TransformAxes3D(axis_length=0.25),
        )

        rgb = arrays["head_image"][t]
        wrist = arrays["wrist_image"][t]
        world_t_cam = _pose_matrix(arrays["head_pos"][t], arrays["head_quat"][t])
        if "extrinsic" in sidecars:
            world_t_cam = sidecars["extrinsic"][t]
        rr.log(
            "scene/world/camera",
            rr.Transform3D(translation=world_t_cam[:3, 3], mat3x3=world_t_cam[:3, :3]),
        )
        rr.log("scene/world/camera/rgb", rr.Image(rgb))
        rr.log("scene/wrist/rgb", rr.Image(wrist))

        if "intrinsic" in sidecars:
            native_intrinsic = sidecars["intrinsic"][t]
            height, width = rgb.shape[:2]
            native_hw = (
                tuple(int(v) for v in sidecars["depth"].shape[1:])
                if "depth" in sidecars
                else (height, width)
            )
            display_intrinsic = _scale_intrinsic(native_intrinsic, native_hw, (height, width))
            rr.log(
                "scene/world/camera",
                rr.Pinhole(
                    image_from_camera=display_intrinsic.astype(np.float32),
                    width=width,
                    height=height,
                ),
            )
            for tag, action, color in (
                ("gt_horizon_2d", gt, _COLOR_GT),
                ("pred_horizon_2d", pred, _COLOR_PRED),
            ):
                arm_points = np.concatenate(
                    [action[:, ENTITIES[side][0]] for side in ("left", "right")], axis=0
                )
                uv, valid = _project_world_to_pixel(arm_points, display_intrinsic, world_t_cam)
                if valid.any():
                    rr.log(
                        f"scene/world/camera/{tag}",
                        rr.Points2D(uv[valid], colors=[color], radii=2.5),
                    )

            if "depth" in sidecars:
                depth_mm = sidecars["depth"][t]
                display_depth_mm = cv2.resize(
                    depth_mm,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                rr.log(
                    "scene/world/camera/depth",
                    rr.DepthImage(display_depth_mm, meter=1000.0),
                )
                pts, mask = _unproject_depth(
                    depth_mm.astype(np.float32) / 1000.0,
                    native_intrinsic,
                    world_t_cam,
                )
                depth_h, depth_w = depth_mm.shape
                depth_rgb = cv2.resize(rgb, (depth_w, depth_h), interpolation=cv2.INTER_AREA)
                pts, colors = _voxel_downsample(pts, depth_rgb[mask], voxel_size)
                rr.log("scene/world/point_cloud", rr.Points3D(pts, colors=colors, radii=0.003))

        combined = np.stack([gt, pred])
        representations = _mof_representations(bundle, arrays, t, combined)
        for representation in REPRESENTATIONS:
            gt_rep, pred_rep = representations[representation]
            root = f"mof_frames/{representation}"
            _log_action_horizon(rr, f"{root}/gt", gt_rep, _COLOR_GT)
            _log_action_horizon(rr, f"{root}/pred", pred_rep, _COLOR_PRED)
            _log_first_step_errors(rr, f"{root}/error", gt_rep, pred_rep)

        for side in ("left", "right"):
            grip_ch = ENTITIES[side][2]
            rr.log(
                f"scene/gripper/{side}",
                rr.Scalars([float(gt[0, grip_ch]), float(pred[0, grip_ch])]),
            )

        rr.log(
            "router/heatmap",
            rr.Tensor(
                router_trace.T,
                dim_names=["expert", "denoising_step"],
                value_range=(0.0, 1.0),
            ),
        )
        rr.log("router/mean", rr.Scalars(router_trace.mean(axis=0)))

    print(f"[rerun] logged {len(records)} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--policy-path",
        required=True,
        help="mofpo workspace .ckpt (e.g. .../checkpoints/latest.ckpt)",
    )
    parser.add_argument(
        "--episode",
        type=Path,
        default=DEFAULT_EPISODE,
        help=f"ported safetensors episode (default {DEFAULT_EPISODE})",
    )
    parser.add_argument(
        "--port-meta", default=None, help="override the training port's meta.json path"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="frames between evaluated chunks (default n_action_steps)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="stream Rerun headlessly to this .rrd instead of spawning a viewer",
    )
    parser.add_argument(
        "--sidecar-root",
        type=Path,
        default=None,
        help="LeRobot dataset root containing debug/calib + debug/depth (auto-detected)",
    )
    parser.add_argument("--voxel", type=float, default=0.008, help="point-cloud voxel size (m)")
    parser.add_argument(
        "--seed", type=int, default=0, help="diffusion seed for reproducible predictions"
    )
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output", type=Path, default=None, help="overlay PNG path (default alongside the episode)"
    )
    args = parser.parse_args()

    bundle = MoFBundle(
        args.policy_path,
        device=args.device,
        use_ema=not args.no_ema,
        num_inference_steps=args.num_inference_steps,
        port_meta_path=args.port_meta,
    )
    print(bundle.describe())
    arrays = load_file(str(args.episode))
    stride = args.stride if args.stride is not None else bundle.n_action_steps
    if stride < 1:
        raise ValueError(f"--stride must be >= 1, got {stride}")
    if args.voxel <= 0:
        raise ValueError(f"--voxel must be > 0, got {args.voxel}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    records, router_traces, diffusion_timesteps = evaluate(bundle, arrays, stride)
    report(records)
    out = args.output or args.episode.with_suffix(".mof_pred.png")
    render(records, Path(out))
    render_rerun(
        bundle=bundle,
        arrays=arrays,
        records=records,
        router_traces=router_traces,
        diffusion_timesteps=diffusion_timesteps,
        episode=args.episode,
        sidecar_root=args.sidecar_root,
        save_path=args.save,
        stride=stride,
        voxel_size=args.voxel,
    )


if __name__ == "__main__":
    main()
