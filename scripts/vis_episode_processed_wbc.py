"""Visualize a processed WBC mobile-manipulation LeRobotDataset in rerun.

Sibling to ``scripts/vis_episode_processed.py`` (the arm-only tabletop viewer);
same rendering structure, but sourced from the WBC ``eef_head`` variant produced
by ``scripts/port_wbc_mobile_hdf5.py`` (schema ``omniteleop.wbc_policy_format``).

Unlike the tabletop viewer there is ONE variant (no joint sibling): the processed
WBC dataset stores EEF/head poses, NOT joint angles, so this viewer renders no URDF
mesh. Instead the mobile base is drawn explicitly from the policy's own odometry.

Everything policy-facing is shown:

``observation.state`` (32-D, ``STATE_AXES``) -- the base-egocentric observation:
  * ``[0:9]``  left EEF  pos3 + rot6 in the BASE frame (achieved FK)
  * ``[9]``    left gripper (raw continuous FC03 reading)
  * ``[10:19]`` right EEF pos3 + rot6 in the BASE frame (achieved FK)
  * ``[19]``   right gripper (raw)
  * ``[20:29]`` head (``zed_depth_frame``) pos3 + rot6 in the BASE frame (achieved FK)
  * ``[29:32]`` base pose ``(x, y, yaw)`` -- measured odometry in the engage-origin
    WORLD frame; the policy's only absolute world anchor.

``action`` (29-D, ``ACTION_AXES``) -- the WORLD-frame targets fed VERBATIM to
``ik.solve()`` at the record tick:
  * ``[0:9]`` / ``[10:19]`` left / right EEF targets (pos3 + rot6, WORLD frame)
  * ``[9]`` / ``[19]`` left / right gripper commands (binarized 0/1)
  * ``[20:29]`` head target (pos3 + rot6, WORLD frame)

``observation.images.head_rgb`` / ``observation.images.wrist_rgb`` -- the two
egocentric camera streams.

Frame reconciliation (the crux of this viewer): ``action`` is WORLD-frame while
``observation.state`` EEF/head poses are BASE-frame. To render both in one 3D
scene the achieved (state) poses are composed to world with the measured base
pose, EXACTLY as the porter builds its calib sidecar
(``world_T_eef = base_pose_to_mat(state[29:32]) @ pos6d_to_mat(state_block)``);
this reconstruction reproduces the sidecar ``extrinsic`` to ~1e-8. Consequently
the achieved (blue) EEF markers are spatially consistent with the depth point
cloud, and the whole arm+cloud scene sweeps through the fixed world frame as the
base drives -- a faithful depiction of the base motion the policy conditions on.

Panels:
  * 3D world -- depth point cloud (world frame), per-arm EEF markers/frames
    (red = ``action`` world target, blue = ``observation.state`` achieved->world),
    head frames (red target / blue achieved, the latter coincident with the camera),
    the mobile base triad + its full odometry path (gold), optional
    ``observation.environment_state`` position-condition markers, and the head
    camera pinhole/frustum.
  * Head RGB / depth -- ``observation.images.head_rgb`` + depth sidecar, overlaid
    with the projected achieved (blue) and commanded (red) EEF pixels.
  * Wrist RGB -- ``observation.images.wrist_rgb`` (plain 2D; no calibration/depth).
  * Gripper time series (left, right) -- red = ``action`` (binary command),
    blue = ``observation.state`` (raw reading).
  * Base pose time series -- ``base_x`` / ``base_y`` (m) and ``base_yaw`` (rad).

Matplotlib (shown after the viewer is populated):
  * Per-arm EEF xyz -- red = ``action`` (world), blue = ``observation.state``
    composed to world; per-axis velocity cross-correlation lag annotated
    (positive = achieved lags command -> controller tracking lag).
  * Base odometry top-down -- the ``(x, y)`` drive path with heading arrows.

Sidecars (NOT in the parquet; written by the porter in lockstep):
  ``debug/depth/episode_XXXXXX.npz``  key ``depth``     (T, H, W) uint16 (mm)
  ``debug/calib/episode_XXXXXX.npz``  keys ``extrinsic`` (T,4,4 world_T_zed),
      ``base_extrinsic`` (T,4,4 base_T_zed), ``intrinsic`` (T,3,3, already
      rescaled to the stored resize_h x resize_w frame).

Usage::

    python scripts/vis_episode_processed_wbc.py \\
        --dataset_dir /home/yixuan/Dexmate/data/processed_wbc/dexmate_wbc_eef_head \\
        --episode_index 0

    # headless over SSH -> open later with `rerun FILE.rrd`:
    python scripts/vis_episode_processed_wbc.py --save /tmp/wbc_ep0.rrd

Run in the ``dexmate_lerobot`` env (needs lerobot + rerun).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from omniteleop.wbc_policy_format import (
    ACTION_AXES,
    STATE_AXES,
    base_pose_to_mat,
    pos6d_to_mat,
)

_DEFAULT_DATASET_DIR = "/home/yixuan/Dexmate/data/processed_wbc/dexmate_wbc_eef_head"

# Default gRPC endpoint of a running rerun Viewer (matches a bare `rerun
# --headless`, i.e. no --port). Used by --connect for the MCP debug workflow.
_DEFAULT_VIEWER_URL = "rerun+http://127.0.0.1:9876/proxy"

# ── Policy vector layout (omniteleop.wbc_policy_format; validated against the
#    dataset's stored axis names below). EEF/head offsets are shared by state and
#    action; the base block is state-only. ────────────────────────────────────
_ARM_SIDES = ("left", "right")
_EEF_BLOCK = {"left": slice(0, 9), "right": slice(10, 19)}   # pos3 + rot6
_GRIP_IDX = {"left": 9, "right": 19}
_HEAD_BLOCK = slice(20, 29)                                   # pos3 + rot6
_BASE_BLOCK = slice(29, 32)                                   # (x, y, yaw), state only

_COLOR_ACTION = (255, 0, 0)    # red  — action (world-frame command)
_COLOR_STATE  = (0, 80, 255)   # blue — observation.state (achieved, composed to world)
_COLOR_BASE   = (230, 190, 0)  # gold — mobile base odometry
_COLOR_BASE_X = (230, 60, 60)
_COLOR_BASE_Y = (60, 200, 60)
_COLOR_BASE_YAW = (80, 120, 255)
_COLOR_POS_COND_SRC = (0, 200, 255)
_COLOR_POS_COND_DST = (255, 225, 25)
_COLOR_POS_COND_OTHER = (
    (180, 220, 80),
    (255, 140, 40),
    (180, 120, 255),
    (255, 105, 180),
)

OBS_ENV_STATE_KEY = "observation.environment_state"

_GRIPPER_Y_MIN = -0.2
_GRIPPER_Y_MAX = 1.1
_POS_COND_RADIUS_3D = 0.018
_HEAD_DEPTH_ENTITY = "/world/camera/depth"


# ── Generic helpers copied verbatim from vis_episode_processed.py so the two
#    processed-dataset viewers share identical projection / lag / plot logic. ──
def project_world_to_pixel(
    pts_world: np.ndarray, K: np.ndarray, world_t_cam: np.ndarray, z_min: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Project (N, 3) world points into pixel coords (camera frame: +x right,
    +y down, +z forward). Returns ``(uv, valid)`` with ``uv`` (N, 2) float32 and
    ``valid`` a boolean mask of points in front of the camera."""
    pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
    cam_t_world = np.linalg.inv(world_t_cam)
    pts_homog = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    pts_cam = (cam_t_world @ pts_homog.T).T[:, :3]
    z = pts_cam[:, 2]
    valid = z > z_min
    z_safe = np.where(valid, z, 1.0)
    u = K[0, 0] * pts_cam[:, 0] / z_safe + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / z_safe + K[1, 2]
    return np.stack([u, v], axis=-1).astype(np.float32), valid


def unproject_depth(
    depth_m: np.ndarray, K: np.ndarray, world_t_cam: np.ndarray, max_depth: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a depth image to world-frame XYZ. Returns (pts, mask)."""
    H, W = depth_m.shape
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    mask = (depth_m > 0.05) & (depth_m < max_depth)
    z = depth_m[mask]
    x = (u[mask] - K[0, 2]) / K[0, 0] * z
    y = (v[mask] - K[1, 2]) / K[1, 1] * z
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)
    pts_world = (world_t_cam @ pts_cam.T).T[:, :3]
    return pts_world.astype(np.float32), mask


def voxel_downsample(
    pts: np.ndarray, colors: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-average downsample on CUDA when available; random subsample otherwise."""
    if pts.shape[0] == 0:
        return pts, colors
    try:
        pts_t = torch.from_numpy(pts).cuda()
        cols_t = torch.from_numpy(colors.astype(np.float32)).cuda()
        voxels = torch.floor(pts_t / voxel_size).int()
        keys = voxels[:, 0] * (1 << 20) + voxels[:, 1] * (1 << 10) + voxels[:, 2]
        _, inv = torch.unique(keys, return_inverse=True)
        out_pts = torch.zeros(inv.max() + 1, 3, device="cuda")
        out_cols = torch.zeros(inv.max() + 1, 3, device="cuda")
        cnt = torch.zeros(inv.max() + 1, device="cuda")
        out_pts.scatter_add_(0, inv.unsqueeze(1).expand(-1, 3), pts_t)
        out_cols.scatter_add_(0, inv.unsqueeze(1).expand(-1, 3), cols_t)
        cnt.scatter_add_(0, inv, torch.ones(len(inv), device="cuda"))
        out_pts = (out_pts / cnt.unsqueeze(1)).cpu().numpy()
        out_cols = (out_cols / cnt.unsqueeze(1)).cpu().numpy().astype(np.uint8)
        return out_pts, out_cols
    except Exception:
        idx = np.random.choice(len(pts), min(len(pts), 50000), replace=False)
        return pts[idx], colors[idx]


def _rgb_to_hwc(t) -> np.ndarray:
    """Normalize an RGB observation to (H, W, 3) uint8."""
    a = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    if a.ndim == 3 and a.shape[0] == 3:
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    return a


def head_depth_hidden_overrides() -> dict[str, rrb.EntityBehavior]:
    """Hide head depth by default while keeping it toggleable in the sidebar."""
    return {_HEAD_DEPTH_ENTITY: rrb.EntityBehavior(visible=False)}


def _position_condition_labels(axis_names: list[str] | None, point_count: int) -> list[str]:
    """Convert environment-state axis names into one label per xyz point."""
    if axis_names is None or len(axis_names) != point_count * 3:
        return [f"obj{i}" for i in range(point_count)]

    labels: list[str] = []
    for point_idx in range(point_count):
        names = axis_names[point_idx * 3 : point_idx * 3 + 3]
        suffixes = ("_x", "_y", "_z")
        prefixes: list[str] = []
        for name, suffix in zip(names, suffixes, strict=True):
            if not name.endswith(suffix):
                prefixes = []
                break
            prefixes.append(name[: -len(suffix)])
        if prefixes and len(set(prefixes)) == 1 and prefixes[0]:
            labels.append(prefixes[0])
        else:
            labels.append(f"obj{point_idx}")
    return labels


def _position_condition_colors(labels: list[str]) -> np.ndarray:
    """Stable marker colors, highlighting src/dst pairs when names expose them."""
    colors: list[tuple[int, int, int]] = []
    for idx, label in enumerate(labels):
        if label.endswith(("_src", "_before")):
            colors.append(_COLOR_POS_COND_SRC)
        elif label.endswith(("_dst", "_after")):
            colors.append(_COLOR_POS_COND_DST)
        else:
            colors.append(_COLOR_POS_COND_OTHER[idx % len(_COLOR_POS_COND_OTHER)])
    return np.asarray(colors, dtype=np.uint8)


def _position_condition_link_strips(
    points: np.ndarray, finite: np.ndarray, labels: list[str]
) -> tuple[list[np.ndarray], list[tuple[int, int, int]], list[str]]:
    """Build src->dst line strips for paired SceneDiff condition points."""
    strips: list[np.ndarray] = []
    colors: list[tuple[int, int, int]] = []
    line_labels: list[str] = []
    for start in range(0, len(labels) - 1, 2):
        if not (finite[start] and finite[start + 1]):
            continue
        left, right = labels[start], labels[start + 1]
        if not (left.endswith("_src") and right.endswith("_dst")):
            continue
        strips.append(np.stack([points[start], points[start + 1]]).astype(np.float32))
        colors.append(_COLOR_POS_COND_DST)
        stage = left[: -len("_src")]
        line_labels.append(f"{stage}: src->dst")
    return strips, colors, line_labels


def load_processed_wbc_position_condition(dataset) -> dict[str, np.ndarray | list[str]] | None:
    """Read optional processed-WBC position condition from ``observation.environment_state``.

    The porter writes this as a flat ``object_nums * 3`` world-frame vector on every
    frame. This helper reshapes it to ``(N, object_nums, 3)`` and derives point labels
    from the feature axis names when available.
    """
    features = getattr(getattr(dataset, "meta", None), "features", {})
    if OBS_ENV_STATE_KEY not in features:
        return None

    rows: list[np.ndarray] = []
    for idx in range(len(dataset)):
        frame = dataset[idx]
        if OBS_ENV_STATE_KEY not in frame:
            raise KeyError(
                f"{OBS_ENV_STATE_KEY} is declared in dataset features but missing from frame {idx}"
            )
        value = frame[OBS_ENV_STATE_KEY]
        arr = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
        rows.append(np.asarray(arr, dtype=np.float32).reshape(-1))

    env_state = np.stack(rows).astype(np.float32)
    if env_state.shape[1] == 0 or env_state.shape[1] % 3 != 0:
        raise ValueError(
            f"{OBS_ENV_STATE_KEY} must have a non-empty multiple-of-3 width, "
            f"got {env_state.shape}"
        )
    if not np.all(np.isfinite(env_state)):
        bad = np.argwhere(~np.isfinite(env_state))[0]
        raise ValueError(f"{OBS_ENV_STATE_KEY} has a non-finite value at {bad.tolist()}")

    feature = features[OBS_ENV_STATE_KEY]
    axes = None
    if isinstance(feature, Mapping):
        names = feature.get("names")
        if isinstance(names, Mapping) and "axes" in names:
            axes = [str(axis) for axis in names["axes"]]
    point_count = env_state.shape[1] // 3
    return {
        "points": env_state.reshape(len(dataset), point_count, 3),
        "labels": _position_condition_labels(axes, point_count),
    }


def _format_lag_ms(lag_ms: float, *, include_unit: bool = False) -> str:
    if not np.isfinite(lag_ms):
        return "n/a"
    text = f"{lag_ms:.1f}"
    return f"{text} ms" if include_unit else text


def estimate_lag_ms(
    cmd: np.ndarray,
    actual: np.ndarray,
    dt_s: float,
    *,
    max_lag_ms: float = 1000.0,
    min_corr: float = 0.25,
    min_overlap_frac: float = 0.5,
) -> float:
    """Velocity cross-correlation peak lag in ms (positive = ``actual`` lags ``cmd``).

    Verbatim copy of ``vis_teleop_curves.estimate_lag_ms`` so plots share the
    identical lag definition with the raw-HDF5 viewers.
    """
    if dt_s <= 0 or not np.isfinite(dt_s):
        return float("nan")
    cmd = np.asarray(cmd, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if cmd.ndim != 1 or actual.ndim != 1 or len(cmd) != len(actual) or len(cmd) < 4:
        return float("nan")
    if not (np.all(np.isfinite(cmd)) and np.all(np.isfinite(actual))):
        return float("nan")
    a = np.diff(cmd)
    b = np.diff(actual)
    n = len(a)
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    min_overlap = max(3, int(np.ceil(n * min_overlap_frac)))
    max_lag_samples = round(max_lag_ms / (dt_s * 1000.0))
    max_lag_samples = min(max_lag_samples, n - min_overlap)
    if max_lag_samples < 1:
        return float("nan")
    lags = np.arange(-max_lag_samples, max_lag_samples + 1, dtype=np.float64)
    scores = np.full(lags.shape, np.nan, dtype=np.float64)
    for i, lag in enumerate(lags.astype(int)):
        if lag >= 0:
            aw = a[: n - lag]
            bw = b[lag:]
        else:
            aw = a[-lag:]
            bw = b[: n + lag]
        aw = aw - aw.mean()
        bw = bw - bw.mean()
        denom = np.linalg.norm(aw) * np.linalg.norm(bw)
        if denom > 1e-12:
            scores[i] = float(np.dot(bw, aw) / denom)
    if np.all(np.isnan(scores)):
        return float("nan")
    best_i = int(np.nanargmax(scores))
    best_corr = float(scores[best_i])
    if best_corr < min_corr:
        return float("nan")
    lag_samples = float(lags[best_i])
    if 0 < best_i < len(scores) - 1 and np.all(np.isfinite(scores[best_i - 1 : best_i + 2])):
        y0, y1, y2 = scores[best_i - 1 : best_i + 2]
        denom = y0 - (2.0 * y1) + y2
        if abs(denom) > 1e-12:
            offset = 0.5 * (y0 - y2) / denom
            lag_samples += float(np.clip(offset, -1.0, 1.0))
    return lag_samples * dt_s * 1000.0


def _build_eef_xyz_figs(
    t: np.ndarray,
    action_xyz: np.ndarray,
    state_xyz: np.ndarray,
    dt_s: float,
    prefix: str = "",
) -> list[plt.Figure]:
    """Three figures (one per axis) comparing ``action`` (world) vs
    ``observation.state`` (composed to world). Per-axis velocity cross-correlation
    lag (positive = state lags action) annotated upper-left."""
    color_action = (1.0, 0.0, 0.0)         # red  — action (world target)
    color_state  = (0.0, 80 / 255.0, 1.0)  # blue — observation.state (achieved->world)
    figs: list[plt.Figure] = []
    for axis_idx, axis_name in enumerate("xyz"):
        act = action_xyz[:, axis_idx]
        st  = state_xyz[:, axis_idx]
        lag_ms = estimate_lag_ms(act, st, dt_s)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(t, act, color=color_action, lw=1.4, label="action (world)")
        ax.plot(t, st,  color=color_state,  lw=1.2, label="observation.state (→world)")

        ax.text(
            0.01, 0.95,
            f"lag = {_format_lag_ms(lag_ms, include_unit=True)}   "
            f"(positive = observation.state lags action)",
            transform=ax.transAxes, fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
        )
        ax.set_xlabel("t (s)")
        ax.set_ylabel(f"eef_{axis_name} (m)")
        ax.set_title(f"{prefix}EEF {axis_name}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        figs.append(fig)
    return figs


def _build_base_path_fig(base_xy: np.ndarray, base_yaw: np.ndarray) -> plt.Figure:
    """Top-down ``(x, y)`` odometry path with sampled heading arrows.

    The base pose ``state[29:32]`` is the policy's only absolute world anchor, so a
    bird's-eye of the drive complements the (moving) 3D base triad.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(base_xy[:, 0], base_xy[:, 1], color=(0.9, 0.75, 0.0), lw=1.6, label="base path")
    n = len(base_xy)
    step = max(1, n // 25)
    idx = np.arange(0, n, step)
    ax.quiver(
        base_xy[idx, 0], base_xy[idx, 1],
        np.cos(base_yaw[idx]), np.sin(base_yaw[idx]),
        color=(0.55, 0.45, 0.0), angles="xy", scale=25, width=0.004, alpha=0.8,
    )
    ax.scatter(base_xy[0, 0], base_xy[0, 1], c="green", s=70, zorder=5, label="start")
    ax.scatter(base_xy[-1, 0], base_xy[-1, 1], c="red", s=70, zorder=5, label="end")
    ax.set_xlabel("base_x (m)")
    ax.set_ylabel("base_y (m)")
    ax.set_title("Base odometry (engage-origin world, top-down)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=_DEFAULT_DATASET_DIR,
        help="Processed WBC variant root (the dexmate_wbc_eef_head dataset produced "
             "by scripts/port_wbc_mobile_hdf5.py).",
    )
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument(
        "--voxel", type=float, default=0.008, help="Voxel size for PCD downsample (m)."
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Log every Nth frame (>=1). Trims the per-frame image/depth/pointcloud "
        "cost, which dominates memory.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="If set, stream to this .rrd file headlessly instead of spawning a "
        "viewer (useful over SSH: generate on the server, open later with "
        "`rerun FILE.rrd`). Default: spawn the viewer.",
    )
    parser.add_argument(
        "--connect",
        type=str,
        nargs="?",
        const=_DEFAULT_VIEWER_URL,
        default=None,
        help="Log to an ALREADY-RUNNING rerun Viewer over gRPC instead of "
        "spawning one or writing an .rrd. Bare --connect uses "
        f"{_DEFAULT_VIEWER_URL} (a `rerun --headless` with no --port); pass a URL "
        "to override. This is the display-less / MCP debug workflow: start "
        "`rerun --headless` (optionally `--port N`), then --connect to it so the "
        "rerun MCP can inspect the same viewer. Mutually exclusive with --save.",
    )
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError(f"--stride must be >= 1, got {args.stride}")
    if args.connect is not None and args.save is not None:
        parser.error("--connect and --save are mutually exclusive")

    dataset_root = Path(args.dataset_dir)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"--dataset_dir not a directory: {dataset_root}")

    # return_uint8=True: video frames default to float32 [0, 1]; we need raw uint8
    # for direct rerun logging (and to avoid the all-zero astype trap).
    dataset = LeRobotDataset(
        repo_id=dataset_root.name,
        root=dataset_root,
        episodes=[args.episode_index],
        return_uint8=True,
    )
    N = len(dataset)
    if N == 0:
        raise ValueError(f"Episode {args.episode_index} is empty under {dataset_root}.")

    # Self-describing schema check: fail loudly if pointed at a non-WBC variant.
    state_axes = list(dataset.meta.features["observation.state"]["names"]["axes"])
    action_axes = list(dataset.meta.features["action"]["names"]["axes"])
    if state_axes != list(STATE_AXES):
        raise ValueError(
            "observation.state axes do not match the WBC schema "
            f"(omniteleop.wbc_policy_format.STATE_AXES).\n got:      {state_axes}\n "
            f"expected: {list(STATE_AXES)}"
        )
    if action_axes != list(ACTION_AXES):
        raise ValueError(
            "action axes do not match the WBC schema "
            f"(omniteleop.wbc_policy_format.ACTION_AXES).\n got:      {action_axes}\n "
            f"expected: {list(ACTION_AXES)}"
        )
    for key in ("observation.images.head_rgb", "observation.images.wrist_rgb"):
        if key not in dataset.meta.features:
            raise ValueError(f"{dataset_root.name} is missing {key}.")
    position_condition = load_processed_wbc_position_condition(dataset)
    position_condition_points: np.ndarray | None = None
    position_condition_labels: list[str] = []
    position_condition_colors: np.ndarray | None = None
    if position_condition is not None:
        position_condition_points = np.asarray(position_condition["points"], dtype=np.float32)
        position_condition_labels = list(position_condition["labels"])
        position_condition_colors = _position_condition_colors(position_condition_labels)

    # observation.state EEF/head frame (porter dexmate_meta.json). "base" (default,
    # absolute-action dataset) must be composed to world for display; "world"
    # (--relative-actions dataset) is already world-frame -- do NOT re-compose.
    state_frame = "base"
    meta_json = dataset_root / "dexmate_meta.json"
    if meta_json.exists():
        state_frame = json.loads(meta_json.read_text()).get("state_frame", "base")
    if state_frame not in ("base", "world"):
        raise ValueError(f"unexpected state_frame {state_frame!r} in {meta_json}")
    print(f"observation.state EEF/head frame: {state_frame}")

    state = np.stack([dataset[i]["observation.state"].numpy() for i in range(N)]).astype(np.float64)   # (N, 32)
    action = np.stack([dataset[i]["action"].numpy() for i in range(N)]).astype(np.float64)             # (N, 29)
    if state.shape[1] != len(STATE_AXES):
        raise ValueError(f"observation.state must be (N, {len(STATE_AXES)}), got {state.shape}")
    if action.shape[1] != len(ACTION_AXES):
        raise ValueError(f"action must be (N, {len(ACTION_AXES)}), got {action.shape}")

    # ── Sidecars (NOT in the parquet; see docstring). ────────────────────────
    depth_sidecar = dataset_root / "debug" / "depth" / f"episode_{args.episode_index:06d}.npz"
    calib_sidecar = dataset_root / "debug" / "calib" / f"episode_{args.episode_index:06d}.npz"
    for p in (depth_sidecar, calib_sidecar):
        if not p.exists():
            raise FileNotFoundError(
                f"sidecar missing: {p}. Re-run port_wbc_mobile_hdf5.py --overwrite."
            )
    with np.load(depth_sidecar) as data:
        depth_stack = data["depth"]  # (T, H, W) uint16
    with np.load(calib_sidecar) as data:
        extrinsic_stack = data["extrinsic"].astype(np.float64)   # (T, 4, 4) world_T_zed
        intrinsic_stack = data["intrinsic"].astype(np.float64)   # (T, 3, 3)
    for name, arr in (("depth", depth_stack), ("extrinsic", extrinsic_stack),
                      ("intrinsic", intrinsic_stack)):
        if arr.shape[0] != N:
            raise ValueError(f"{name} sidecar frame count {arr.shape[0]} != dataset {N}")
    if depth_stack.dtype != np.uint16:
        raise ValueError(f"depth sidecar dtype {depth_stack.dtype}, expected uint16")

    # ── Achieved (state) EEF/head poses in WORLD frame. "base" datasets store them
    #    base-frame, so compose via the measured base pose (matches the porter's calib
    #    sidecar to ~1e-8); "world" datasets already store world_T_eef -- use verbatim
    #    (composing again would double-apply the base transform). The base triad still
    #    needs world_t_base either way. ────────────────────────────────────────────
    world_t_base = np.stack([base_pose_to_mat(state[i, _BASE_BLOCK]) for i in range(N)])  # (N, 4, 4)

    def _to_world(block: slice, i: int) -> np.ndarray:
        T = pos6d_to_mat(state[i, block])
        return T if state_frame == "world" else world_t_base[i] @ T

    state_world = {
        side: np.stack([_to_world(_EEF_BLOCK[side], i) for i in range(N)])
        for side in _ARM_SIDES
    }                                                                                     # (N, 4, 4)
    state_world_head = np.stack([_to_world(_HEAD_BLOCK, i) for i in range(N)])             # == extrinsic
    action_mat = {
        side: np.stack([pos6d_to_mat(action[i, _EEF_BLOCK[side]]) for i in range(N)])
        for side in _ARM_SIDES
    }                                                                                     # (N, 4, 4)
    action_head = np.stack([pos6d_to_mat(action[i, _HEAD_BLOCK]) for i in range(N)])       # (N, 4, 4)
    state_world_pos = {side: state_world[side][:, :3, 3] for side in _ARM_SIDES}
    action_pos = {side: action_mat[side][:, :3, 3] for side in _ARM_SIDES}
    base_xy = state[:, 29:31]
    base_yaw = state[:, 31]

    # Sanity: reconstructed world_T_zed must equal the porter's calib extrinsic.
    head_err = float(np.abs(state_world_head - extrinsic_stack).max())
    if head_err > 1e-4:
        raise ValueError(
            f"state head→world disagrees with calib extrinsic by {head_err:.2e} m "
            "(base composition / schema mismatch)"
        )

    sample_rgb = _rgb_to_hwc(dataset[0]["observation.images.head_rgb"])
    H, W = sample_rgb.shape[:2]
    dt_s = 1.0 / float(dataset.fps) if dataset.fps else 0.1
    print(
        f"episode {args.episode_index}: {N} frames | rgb {H}x{W} | fps {dataset.fps} | "
        f"head→world vs calib max err {head_err:.2e} m"
    )
    for side in _ARM_SIDES:
        d = np.linalg.norm(action_pos[side] - state_world_pos[side], axis=1)
        print(f"  {side} EEF |action - state→world| mean {d.mean()*1000:.1f} mm, "
              f"max {d.max()*1000:.1f} mm")
    print(f"  base drove x∈[{base_xy[:,0].min():.2f},{base_xy[:,0].max():.2f}] "
          f"y∈[{base_xy[:,1].min():.2f},{base_xy[:,1].max():.2f}] m, "
          f"yaw∈[{base_yaw.min():.3f},{base_yaw.max():.3f}] rad")
    if position_condition_points is not None:
        print(
            f"  position condition: {len(position_condition_labels)} world points "
            f"({', '.join(position_condition_labels)})"
        )

    # ── Matplotlib figures (EEF xyz per arm + base top-down). ────────────────
    t_s = np.arange(N, dtype=np.float64) * dt_s
    figs: list[plt.Figure] = []
    for side in _ARM_SIDES:
        figs += _build_eef_xyz_figs(t_s, action_pos[side], state_world_pos[side], dt_s,
                                    prefix=f"{side} ")
    figs.append(_build_base_path_fig(base_xy, base_yaw))
    print(f"built {len(figs)} matplotlib figures (6 EEF xyz + 1 base path)")

    # ── rerun setup ──────────────────────────────────────────────────────────
    rr.init("vis_episode_processed_wbc", spawn=args.save is None and args.connect is None)
    if args.connect is not None:
        rr.connect_grpc(args.connect)
        print(f"Connecting to running viewer at {args.connect} (no viewer spawned)")
    elif args.save is not None:
        rr.save(args.save)
        print(f"Streaming to {args.save} (headless; no viewer spawned)")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    playback_fps = float(dataset.fps) if dataset.fps else 10.0
    lookback = max(400, min(N, 5000))
    lookahead = min(80, max(1, N // 50 + 5))
    depth_override = head_depth_hidden_overrides()

    def _cursor_range(origin: str, name: str, y_range=None) -> rrb.TimeSeriesView:
        axis_y = (
            rrb.ScalarAxis(range=y_range, zoom_lock=True) if y_range is not None else None
        )
        return rrb.TimeSeriesView(
            origin=origin, name=name, axis_y=axis_y,
            time_ranges=[
                rrb.VisibleTimeRange(
                    "frame",
                    start=rrb.TimeRangeBoundary.cursor_relative(seq=-lookback),
                    end=rrb.TimeRangeBoundary.cursor_relative(seq=lookahead),
                )
            ],
        )

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        origin="world",
                        name="World (pcd + EEF + base odom)",
                        overrides=depth_override,
                    ),
                    rrb.Spatial2DView(
                        origin="world/camera",
                        name="Head RGB / depth",
                        overrides=depth_override,
                    ),
                    rrb.Spatial2DView(origin="wrist", name="Wrist RGB"),
                    column_shares=[2.0, 1.3, 1.3],
                ),
                rrb.Horizontal(
                    _cursor_range("plots/gripper_left", "Left gripper",
                                  (_GRIPPER_Y_MIN, _GRIPPER_Y_MAX)),
                    _cursor_range("plots/gripper_right", "Right gripper",
                                  (_GRIPPER_Y_MIN, _GRIPPER_Y_MAX)),
                    _cursor_range("plots/base_pose", "Base pose (x,y,yaw)"),
                    column_shares=[1.0, 1.0, 1.4],
                ),
                row_shares=[2.2, 1.0],
            ),
            rrb.TimePanel(timeline="frame", fps=playback_fps),
        )
    )

    # Named series: gripper (action red / state blue); base pose (x, y, yaw).
    for side in _ARM_SIDES:
        rr.log(
            f"plots/gripper_{side}",
            rr.SeriesLines(
                colors=[_COLOR_ACTION, _COLOR_STATE],
                names=[f"{side} action", f"{side} observation.state"],
                widths=[2.0, 2.0],
            ),
            static=True,
        )
    rr.log(
        "plots/base_pose",
        rr.SeriesLines(
            colors=[_COLOR_BASE_X, _COLOR_BASE_Y, _COLOR_BASE_YAW],
            names=["base_x (m)", "base_y (m)", "base_yaw (rad)"],
            widths=[2.0, 2.0, 2.0],
        ),
        static=True,
    )

    # Full base odometry path (static gold line + points) so the whole drive is
    # visible while the per-frame base triad moves along it.
    base_path_xyz = np.column_stack([base_xy, np.zeros(N)]).astype(np.float32)
    rr.log("world/base_path", rr.LineStrips3D([base_path_xyz], colors=[_COLOR_BASE], radii=0.008),
           static=True)
    rr.log("world/base_path/pts", rr.Points3D(base_path_xyz, colors=[_COLOR_BASE], radii=0.006),
           static=True)

    # ── Per-frame loop ───────────────────────────────────────────────────────
    for idx in list(range(0, N, args.stride)) + [N]:
        rr.set_time("frame", sequence=idx)

        if idx == N:
            rr.log("world/action", rr.Clear(recursive=True))
            rr.log("world/state", rr.Clear(recursive=True))
            rr.log("world/base", rr.Clear(recursive=True))
            rr.log("world/position_condition", rr.Clear(recursive=True))
            rr.log("world/camera/eef_state_2d", rr.Clear(recursive=False))
            rr.log("world/camera/eef_action_2d", rr.Clear(recursive=False))
            continue

        frame = dataset[idx]
        world_t_cam = extrinsic_stack[idx]
        K = intrinsic_stack[idx]
        K32 = K.astype(np.float32)
        rr.log("world/camera",
               rr.Transform3D(translation=world_t_cam[:3, 3], mat3x3=world_t_cam[:3, :3]))
        rr.log("world/camera", rr.Pinhole(image_from_camera=K32, width=W, height=H))

        # Mobile base triad (odometry pose).
        base_tf = world_t_base[idx]
        rr.log("world/base",
               rr.Transform3D(translation=base_tf[:3, 3], mat3x3=base_tf[:3, :3]),
               rr.TransformAxes3D(axis_length=0.25))

        # Per-arm EEF: red = action world target (trail + triad), blue = achieved
        # observation.state composed to world (trail + triad).
        for side in _ARM_SIDES:
            am = action_mat[side][idx]
            rr.log(f"world/action/{side}/marker",
                   rr.Points3D(action_pos[side][idx:], colors=[_COLOR_ACTION], radii=0.006))
            rr.log(f"world/action/{side}/frame",
                   rr.Transform3D(translation=am[:3, 3], mat3x3=am[:3, :3]),
                   rr.TransformAxes3D(axis_length=0.12))
            sm = state_world[side][idx]
            rr.log(f"world/state/{side}/marker",
                   rr.Points3D(state_world_pos[side][idx:], colors=[_COLOR_STATE], radii=0.006))
            rr.log(f"world/state/{side}/frame",
                   rr.Transform3D(translation=sm[:3, 3], mat3x3=sm[:3, :3]),
                   rr.TransformAxes3D(axis_length=0.12))

        # Head: red = action world target; blue = achieved (coincident with camera).
        ah = action_head[idx]
        rr.log("world/action/head/frame",
               rr.Transform3D(translation=ah[:3, 3], mat3x3=ah[:3, :3]),
               rr.TransformAxes3D(axis_length=0.2))
        rr.log("world/action/head/pt",
               rr.Points3D(ah[None, :3, 3], colors=[_COLOR_ACTION], radii=0.02))
        sh = state_world_head[idx]
        rr.log("world/state/head/frame",
               rr.Transform3D(translation=sh[:3, 3], mat3x3=sh[:3, :3]),
               rr.TransformAxes3D(axis_length=0.2))
        rr.log("world/state/head/pt",
               rr.Points3D(sh[None, :3, 3], colors=[_COLOR_STATE], radii=0.02))

        # Head RGB (parquet) + depth (sidecar), both under world/camera.
        rgb = _rgb_to_hwc(frame["observation.images.head_rgb"])
        depth_mm = depth_stack[idx]
        rr.log("world/camera/rgb", rr.Image(rgb))
        rr.log("world/camera/depth", rr.DepthImage(depth_mm, meter=1000.0))

        # Wrist RGB: plain 2D panel (no calibration/depth for the wrist camera).
        rr.log("wrist/rgb", rr.Image(_rgb_to_hwc(frame["observation.images.wrist_rgb"])))

        # 2D overlays: project achieved (blue) + commanded (red) EEF into the head image.
        for tag, pos_map, color in (
            ("eef_state_2d", state_world_pos, _COLOR_STATE),
            ("eef_action_2d", action_pos, _COLOR_ACTION),
        ):
            pts = np.stack([pos_map[s][idx] for s in _ARM_SIDES])
            uv, valid = project_world_to_pixel(pts, K, world_t_cam)
            if valid.any():
                rr.log(f"world/camera/{tag}",
                       rr.Points2D(uv[valid], colors=[color], radii=2.5))
            else:
                rr.log(f"world/camera/{tag}", rr.Clear(recursive=False))

        # Point cloud (manual unproject + voxel downsample), world frame.
        depth_m = depth_mm.astype(np.float32) / 1000.0
        pts, mask = unproject_depth(depth_m, K, world_t_cam)
        cols = rgb[mask]
        pts, cols = voxel_downsample(pts, cols, args.voxel)
        rr.log("world/pcd", rr.Points3D(pts, colors=cols, radii=0.003))

        # Optional processed SceneDiff position condition, already in the same
        # engage-origin world frame as action and base odometry.
        if position_condition_points is not None and position_condition_colors is not None:
            pc_points = position_condition_points[idx]
            finite = np.all(np.isfinite(pc_points), axis=1)
            label_arr = np.asarray(position_condition_labels, dtype=object)
            if finite.any():
                rr.log(
                    "world/position_condition/points",
                    rr.Points3D(
                        pc_points[finite],
                        colors=position_condition_colors[finite],
                        radii=np.full(int(finite.sum()), _POS_COND_RADIUS_3D, dtype=np.float32),
                        labels=label_arr[finite].tolist(),
                        show_labels=True,
                    ),
                )
            else:
                rr.log("world/position_condition/points", rr.Clear(recursive=False))
            strips, strip_colors, strip_labels = _position_condition_link_strips(
                pc_points, finite, position_condition_labels
            )
            if strips:
                rr.log(
                    "world/position_condition/links",
                    rr.LineStrips3D(
                        strips,
                        colors=strip_colors,
                        radii=np.full(len(strips), 0.006, dtype=np.float32),
                        labels=strip_labels,
                        show_labels=True,
                    ),
                )
            else:
                rr.log("world/position_condition/links", rr.Clear(recursive=False))

        # Gripper scalars per arm: [action (binary), observation.state (raw)].
        for side in _ARM_SIDES:
            g = _GRIP_IDX[side]
            rr.log(f"plots/gripper_{side}",
                   rr.Scalars([float(action[idx, g]), float(state[idx, g])]))
        # Base pose scalars: x, y, yaw.
        rr.log("plots/base_pose",
               rr.Scalars([float(state[idx, 29]), float(state[idx, 30]), float(state[idx, 31])]))

    if figs and args.save is None and args.connect is None:
        plt.show()


if __name__ == "__main__":
    main()
