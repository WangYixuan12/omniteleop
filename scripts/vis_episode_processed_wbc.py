"""Visualize processed WBC mobile-manipulation data in rerun.

Two processed formats of the SAME raw takes, selected by which flag you pass:

``--dataset_dir``  the LeRobotDataset written by ``scripts/port_wbc_mobile_hdf5.py``
    (head/wrist RGB video + depth/calib sidecars). The default when ``--zarr`` is
    not given.
``--zarr``         the ManiFlow point-cloud zarr written by
    ``scripts/port_wbc_mobile_zarr.py``. Requires ``--maniflow-run-dir`` and renders
    both the stored 1024-point cloud and the exact policy selection configured in the
    run's ``.hydra/config.yaml``. Head/Wrist RGB, depth and the full SAM3.1 mask come
    directly from the raw HDF5/mask NPZ through the zarr episode manifest; this mode
    never opens a sibling LeRobot dataset. Runs trained with ``action_frame_mode:
    mof`` additionally get a MoF-frames tab (see Panels).

Sibling to ``scripts/vis_episode_processed.py`` (the arm-only tabletop viewer);
same rendering structure, schema ``omniteleop.wbc_policy_format``.

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

``observation.images.head_rgb`` / ``observation.images.{left,right}_wrist_rgb``
(left/right = the ARM the wrist camera is on; right only for two-wrist datasets) -- the
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
  * Zarr policy comparison -- Stored RGB 3D, Policy RGB 3D and (only when consumed)
    Policy masks 3D, with identical initial cameras and shared world context.
  * MoF frames tab (zarr mode, runs trained with ``action_frame_mode: mof``) -- the
    GT action window (``horizon`` steps, refs at the current frame = last obs step,
    edge-replicated at episode bounds like the training sampler's padding)
    re-expressed in every enabled expert representation through the run checkout's
    own ``maniflow.model.common.frame_transforms`` (world hub), one 3D view per
    expert, round-trip-checked back to world per episode. Each view also carries the
    point cloud that expert's tower actually reads: the SAME policy selection shown
    in "Policy RGB 3D", placed by the run's ``mof.cloud_frame`` (``world`` -> the
    shared world cloud in every view; ``family`` -> re-expressed into the expert's
    obs-family frame), coloured by the SAM3.1 point labels when the run consumes
    them. The sibling of ``vis_wbc_mof_prediction.py``'s frames tab, minus
    router/predictions (this viewer has no checkpoint).
  * Head RGB + full SAM3.1 mask. Depth and projected EEF pixels are logged but hidden
    from the automatic blueprint, so they can be enabled from the Rerun sidebar.
  * Wrist RGB -- ``observation.images.{left,right}_wrist_rgb`` (one plain 2D panel
    per arm present; no calibration/depth for the wrists).
  * Gripper time series (left, right) -- red = ``action`` (binary command),
    blue = ``observation.state`` (raw reading).
  * Base pose time series -- ``base_x`` / ``base_y`` (m) and ``base_yaw`` (rad).

Matplotlib (shown after the viewer is populated):
  * Per-arm EEF xyz -- red = ``action`` (world), blue = ``observation.state``
    composed to world; per-axis velocity cross-correlation lag annotated
    (positive = achieved lags command -> controller tracking lag).
  * Base odometry top-down -- the ``(x, y)`` drive path with heading arrows.

Sidecars (``--dataset_dir`` only; NOT in the parquet; written by the porter in lockstep):
  ``debug/depth/episode_XXXXXX.npz``  key ``depth``     (T, H, W) uint16 (mm)
  ``debug/calib/episode_XXXXXX.npz``  keys ``extrinsic`` (T,4,4 world_T_zed),
      ``base_extrinsic`` (T,4,4 base_T_zed), ``intrinsic`` (T,3,3, already
      rescaled to the stored resize_h x resize_w frame).

Usage::

    python scripts/vis_episode_processed_wbc.py \\
        --dataset_dir /home/yixuan/Dexmate/data/processed_wbc/train/dexmate_wbc_eef_head \\
        --episode_index 0

    python scripts/vis_episode_processed_wbc.py \\
        --zarr /home/yixuan/Dexmate/data/processed_wbc/maniflow/dexmate_wbc_test.zarr \\
        --maniflow-run-dir /home/yixuan/ManiFlow_Policy/ManiFlow/data/outputs/RUN \\
        --episode_index 0

    # headless over SSH -> open later with `rerun FILE.rrd`:
    python scripts/vis_episode_processed_wbc.py --save /tmp/wbc_ep0.rrd

``--dataset_dir`` needs ``lerobot`` (``dexmate_lerobot``). Config-faithful zarr mode
needs ``dexmate_maniflow`` (PyTorch3D + Rerun + OmegaConf + h5py).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import zarr
from omegaconf import OmegaConf

from omniteleop.wbc_policy_format import (
    ACTION_AXES,
    STATE_AXES,
    base_pose_to_mat,
    pos6d_to_mat,
)

if __package__:
    from scripts import vis_episode as _raw_vis_episode
else:
    import vis_episode as _raw_vis_episode

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
_COLOR_MASK_SRC = (255, 140, 0)   # orange
_COLOR_MASK_DST = (0, 220, 255)   # cyan
_COLOR_MASK_BG = (120, 120, 120)  # gray
OBS_ENV_STATE_KEY = "observation.environment_state"

_GRIPPER_Y_MIN = -0.2
_GRIPPER_Y_MAX = 1.1
_POS_COND_RADIUS_3D = 0.018
_MOF_CLOUD_RADIUS = 0.004   # < the 0.006 horizon points, so the GT chunk stays legible
_HEAD_DEPTH_ENTITY = "/world/camera/depth"
_HEAD_EEF_STATE_ENTITY = "/world/camera/eef_state_2d"
_HEAD_EEF_ACTION_ENTITY = "/world/camera/eef_action_2d"
_MANIFEST_SCHEMAS = ("wbc_maniflow_pointcloud_v2", "wbc_maniflow_pointcloud_v3")
_MASK_LEGEND = "0=background 1=box(src) 2=cloth(dst)"
_NOMINAL_MASK_QUOTAS = {1: 64, 2: 64, 0: 128}


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


def _mask_label_colors(labels: np.ndarray) -> np.ndarray:
    """SAM3.1 point labels (any shape, ``{0,1,2}``) -> ``(..., 3)`` uint8 colors."""
    colors = np.empty((*labels.shape, 3), dtype=np.uint8)
    colors[labels == 0] = _COLOR_MASK_BG
    colors[labels == 1] = _COLOR_MASK_SRC
    colors[labels == 2] = _COLOR_MASK_DST
    return colors


def head_optional_hidden_overrides() -> dict[str, rrb.EntityBehavior]:
    """Keep depth/projected EEF entities sidebar-toggleable but hidden by default."""
    return {
        path: rrb.EntityBehavior(visible=False)
        for path in (
            _HEAD_DEPTH_ENTITY,
            _HEAD_EEF_STATE_ENTITY,
            _HEAD_EEF_ACTION_ENTITY,
        )
    }


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
        "labels": _raw_vis_episode.position_condition_labels(axes, point_count),
    }


# ── Episode sources. Both formats carry the SAME 32-D state / 29-D action; they
#    differ only in the observation (RGB+depth sidecars vs a stored point cloud).
#    LeRobot is imported lazily so --zarr still runs in envs without lerobot. ──
def load_lerobot_episode(dataset_root: Path, episode_index: int) -> dict:
    """The LeRobotDataset variant written by ``scripts/port_wbc_mobile_hdf5.py``."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    # return_uint8=True: video frames default to float32 [0, 1]; we need raw uint8
    # for direct rerun logging (and to avoid the all-zero astype trap).
    dataset = LeRobotDataset(
        repo_id=dataset_root.name, root=dataset_root,
        episodes=[episode_index], return_uint8=True,
    )
    n = len(dataset)
    if n == 0:
        raise ValueError(f"Episode {episode_index} is empty under {dataset_root}.")

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
    # left/right wrist name the ARM. Left is required; right is present only for
    # datasets ported from two-wrist takes (see port_wbc_mobile_hdf5.build_features).
    for key in ("observation.images.head_rgb", "observation.images.left_wrist_rgb"):
        if key not in dataset.meta.features:
            raise ValueError(f"{dataset_root.name} is missing {key}.")

    # observation.state EEF/head frame (porter dexmate_meta.json). "base" (the
    # default) must be composed to world for display; "world" (--world-state
    # dataset) is already world-frame -- do NOT re-compose.
    state_frame = "base"
    meta_json = dataset_root / "dexmate_meta.json"
    if meta_json.exists():
        state_frame = json.loads(meta_json.read_text()).get("state_frame", "base")
    if state_frame not in ("base", "world"):
        raise ValueError(f"unexpected state_frame {state_frame!r} in {meta_json}")

    state = np.stack([dataset[i]["observation.state"].numpy() for i in range(n)]).astype(np.float64)
    action = np.stack([dataset[i]["action"].numpy() for i in range(n)]).astype(np.float64)

    # ── Sidecars (NOT in the parquet; see docstring). ────────────────────────
    depth_sidecar = dataset_root / "debug" / "depth" / f"episode_{episode_index:06d}.npz"
    calib_sidecar = dataset_root / "debug" / "calib" / f"episode_{episode_index:06d}.npz"
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
        if arr.shape[0] != n:
            raise ValueError(f"{name} sidecar frame count {arr.shape[0]} != dataset {n}")
    if depth_stack.dtype != np.uint16:
        raise ValueError(f"depth sidecar dtype {depth_stack.dtype}, expected uint16")

    sample_rgb = _rgb_to_hwc(dataset[0]["observation.images.head_rgb"])
    return {
        "kind": "lerobot", "label": str(dataset_root), "N": n,
        "fps": float(dataset.fps) if dataset.fps else 10.0,
        "state": state, "action": action, "state_frame": state_frame,
        "dataset": dataset, "depth": depth_stack, "extrinsic": extrinsic_stack,
        "intrinsic": intrinsic_stack, "hw": sample_rgb.shape[:2],
        "position_condition": load_processed_wbc_position_condition(dataset),
    }


def load_maniflow_run(run_dir: Path) -> dict:
    """Resolve the policy switches that define the input shown by the zarr viewer."""
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"--maniflow-run-dir has no .hydra/config.yaml: {run_dir}")
    cfg = OmegaConf.load(config_path)
    required = (
        "point_sampling_mode", "mask_channels", "position_condition_mode", "object_nums"
    )
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(
            f"{config_path}: missing explicit policy switch(es) {missing}; legacy run "
            "configs are rejected rather than assigned inferred defaults"
        )
    sampling = str(cfg.point_sampling_mode)
    position = str(cfg.position_condition_mode)
    if sampling not in ("fps", "mask_stratified"):
        raise ValueError(f"{config_path}: unsupported point_sampling_mode={sampling!r}")
    if position not in ("none", "concat", "grounding_tokens"):
        raise ValueError(f"{config_path}: unsupported position_condition_mode={position!r}")
    mask_channels = bool(cfg.mask_channels)
    object_nums = int(cfg.object_nums)
    needs_point_mask = sampling == "mask_stratified" or mask_channels
    if needs_point_mask and object_nums != 2:
        raise ValueError(f"mask modes require object_nums=2, got {object_nums}")
    if not bool(cfg.policy.downsample_points):
        raise ValueError(f"{config_path}: viewer requires policy.downsample_points=true")
    # action_frame_mode postdates the conditioning switches. Unlike them, a config
    # WITHOUT the key is not ambiguous: the key did not exist before MoF landed, and
    # the incumbent path is byte-identical to today's world mode -- so "world" is a
    # fact about such runs, not an inferred default.
    frame_mode = str(cfg.get("action_frame_mode", "world"))
    if frame_mode not in ("world", "mof"):
        raise ValueError(f"{config_path}: unsupported action_frame_mode={frame_mode!r}")
    mof = None
    if frame_mode == "mof":
        if "mof" not in cfg or cfg.mof is None:
            raise ValueError(f"{config_path}: action_frame_mode=mof without a mof block")
        experts = tuple(str(name) for name in cfg.mof.enabled_experts)
        if not experts or len(set(experts)) != len(experts):
            raise ValueError(
                f"{config_path}: mof.enabled_experts must be non-empty and unique, "
                f"got {list(experts)}"
            )
        mof = {
            "enabled_experts": experts,
            "canonical_space": str(cfg.mof.canonical_space),
            "router_mode": str(cfg.mof.router_mode),
            # Required, exactly as the policy reads it: which frame each tower's point
            # cloud lives in. Validated against the checkout's ALLOWED_CLOUD_FRAMES in
            # compute_mof_family_clouds.
            "cloud_frame": str(cfg.mof.cloud_frame),
            "horizon": int(cfg.horizon),
            "n_obs_steps": int(cfg.n_obs_steps),
        }
    return {
        "run_dir": run_dir,
        "config_path": config_path,
        "point_sampling_mode": sampling,
        "mask_channels": mask_channels,
        "needs_point_mask": needs_point_mask,
        "position_condition_mode": position,
        "object_nums": object_nums,
        "visual_cond_len": int(cfg.policy.visual_cond_len),
        "use_pc_color": bool(cfg.policy.use_pc_color),
        "device": str(cfg.training.device),
        "train_zarr": Path(str(cfg.robotwin_task.dataset.zarr_path)).expanduser(),
        "position_dino_dim": int(cfg.policy.get("position_dino_dim", 1280)),
        "action_frame_mode": frame_mode,
        "mof": mof,
    }


def import_run_sampler(run_dir: Path):
    """Import the active checkout's point_process.py, never a viewer-side copy."""
    checkout = next(
        (
            parent for parent in (run_dir, *run_dir.parents)
            if (parent / "maniflow/model/vision_3d/point_process.py").is_file()
        ),
        None,
    )
    if checkout is not None:
        sys.path.insert(0, str(checkout))
    module = importlib.import_module("maniflow.model.vision_3d.point_process")
    module_path = Path(module.__file__).resolve()
    if checkout is None:
        checkout = module_path.parents[3]
    if not module_path.is_relative_to(checkout.resolve()):
        raise RuntimeError(f"imported sampler {module_path} instead of checkout {checkout}")
    return module, checkout


def import_run_frame_transforms(checkout: Path):
    """Import the run checkout's vendored MoF transforms, never a viewer-side copy.

    ``import_run_sampler`` already put the checkout on ``sys.path``; this pulls the
    world-hub transform module plus the WBC action layout and pins their provenance
    the same way the sampler import does.
    """
    transforms = importlib.import_module("maniflow.model.common.frame_transforms")
    layouts = importlib.import_module("maniflow.model.common.action_layout")
    for module in (transforms, layouts):
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(checkout.resolve()):
            raise RuntimeError(f"imported {module_path} instead of checkout {checkout}")
    return transforms, layouts.wbc_layout()


def compute_mof_gt_representations(
    transforms, layout, state: np.ndarray, action: np.ndarray, mof: dict
) -> dict[str, np.ndarray]:
    """GT action windows re-expressed in every enabled expert representation.

    Mirrors the training-time transform exactly: frame ``idx`` is the LAST obs step
    (window index To-1), the action chunk is the ``horizon`` window starting at
    ``idx - (To - 1)`` (edge-replicated at episode bounds, like SequenceSampler
    padding), and frames/refs come from raw agent_pos at ``idx`` via the checkout's
    ``action_frames_from_agent_pos`` / ``entity_refs_in_base``. Every representation
    is round-tripped back to world as a per-episode transform check.
    """
    experts = mof["enabled_experts"]
    for name in (*experts, mof["canonical_space"]):
        if name not in transforms.ALLOWED_EXPERTS:
            raise ValueError(
                f"unknown MoF expert {name!r} (allowed: {transforms.ALLOWED_EXPERTS})"
            )
    n = action.shape[0]
    starts = np.arange(n) - (mof["n_obs_steps"] - 1)
    window = np.clip(starts[:, None] + np.arange(mof["horizon"])[None, :], 0, n - 1)
    chunks = torch.from_numpy(action[window].astype(np.float32))      # (N, H, 29)
    agent_pos = torch.from_numpy(state.astype(np.float32))[:, None]   # (N, 1, 32)
    frames = transforms.action_frames_from_agent_pos(agent_pos, layout)
    refs = transforms.entity_refs_in_base(agent_pos, layout)

    representations: dict[str, np.ndarray] = {}
    worst = 0.0
    for name in experts:
        rep = transforms.world_to_expert_pose(chunks, name, frames, refs, layout)
        back = transforms.expert_pose_to_world(rep, name, frames, refs, layout)
        worst = max(worst, float((back - chunks).abs().max()))
        representations[name] = rep.numpy()
    if worst > 1e-3:
        raise ValueError(
            f"MoF expert round trip diverges by {worst:.2e} on this episode "
            "(checkout transforms disagree with the stored world actions)"
        )
    print(
        f"  mof representations: {', '.join(experts)} (round-trip max err {worst:.1e})"
    )
    return representations


def compute_mof_family_clouds(
    transforms, layout, state: np.ndarray, points_world: np.ndarray, mof: dict
) -> dict[str, np.ndarray]:
    """The selected policy cloud as every enabled expert's tower actually reads it.

    Mirrors ``ManiFlowTransformerPointcloudPolicy._mof_build_conds``: the points are
    selected ONCE in the world frame (FPS/mask-stratified is rigid-equivariant, so the
    selection is frame-independent), then placed by ``mof.cloud_frame`` --

    * ``world`` (v1-faithful): the pointnet runs once and EVERY tower reads the same
      world-frame points, including the base-anchored ones whose state has its base
      block zeroed.
    * ``family``: re-expressed into each obs FAMILY's per-step frame
      (``frame_from_world_per_step``), the world family excepted -- it always keeps
      world points.

    Experts sharing a family (e.g. ``base_rel_trans`` and ``rel_traj``, both ``base``)
    therefore share one cloud, and it is the FAMILY frame, not the expert's own
    representation space: a rel_traj chunk is ref-shifted/rotated per entity, its cloud
    is not, exactly as the tower sees them. Returned in real metres like the GT chunks.
    """
    cloud_frame = mof["cloud_frame"]
    if cloud_frame not in transforms.ALLOWED_CLOUD_FRAMES:
        raise ValueError(
            f"unknown mof.cloud_frame {cloud_frame!r} "
            f"(allowed: {transforms.ALLOWED_CLOUD_FRAMES})"
        )
    if points_world.ndim != 3 or points_world.shape[0] != state.shape[0]:
        raise ValueError(
            f"policy cloud must be (N, P, 3) with N={state.shape[0]}, "
            f"got {points_world.shape}"
        )
    agent_pos = torch.from_numpy(state.astype(np.float32))[:, None]         # (N, 1, 32)
    pts_world = torch.from_numpy(points_world.astype(np.float32))[:, None]  # (N, 1, P, 3)
    by_family: dict[str, np.ndarray] = {}
    clouds: dict[str, np.ndarray] = {}
    for expert in mof["enabled_experts"]:
        family = transforms.FAMILY_OF_EXPERT[expert]
        if family not in by_family:
            if cloud_frame == "world" or family == "world":
                by_family[family] = points_world
            else:
                frame_t_world = transforms.frame_from_world_per_step(
                    agent_pos, family, layout
                )                                                           # (N, 1, 4, 4)
                by_family[family] = transforms.reexpress_points(
                    pts_world, frame_t_world
                )[:, 0].numpy()
        clouds[expert] = by_family[family]
    mapping = ", ".join(
        f"{expert}→{transforms.FAMILY_OF_EXPERT[expert]}"
        for expert in mof["enabled_experts"]
    )
    print(f"  mof cloud frames: {mapping} ({points_world.shape[1]} pts each)")
    return clouds


def _mof_rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) column rot6d -> (..., 3, 3). GT rot6d is clean FK output in every
    representation (rel_traj stores R_ref^T R), so the Gram-Schmidt here is exact --
    this is display-only and never touches noisy interpolants."""
    c1, c2 = rot6d[..., 0:3], rot6d[..., 3:6]
    c1 = c1 / np.linalg.norm(c1, axis=-1, keepdims=True)
    c2 = c2 - np.sum(c1 * c2, axis=-1, keepdims=True) * c1
    c2 = c2 / np.linalg.norm(c2, axis=-1, keepdims=True)
    return np.stack([c1, c2, np.cross(c1, c2)], axis=-1)


def _log_mof_horizon(root: str, chunk: np.ndarray, layout, current_step: int) -> None:
    """One expert view: per-entity horizon trace + triad at the current (To-1) step."""
    for entity in layout.entities:
        pos = chunk[:, entity.pos]
        rotations = _mof_rot6d_to_matrix(chunk[:, entity.rot])
        rr.log(
            f"{root}/{entity.name}/horizon",
            rr.LineStrips3D([pos], colors=[_COLOR_ACTION], radii=0.003),
        )
        rr.log(
            f"{root}/{entity.name}/horizon_points",
            rr.Points3D(pos, colors=[_COLOR_ACTION], radii=0.006),
        )
        rr.log(
            f"{root}/{entity.name}/horizon_z_axes",
            rr.Arrows3D(
                origins=pos,
                vectors=rotations[..., :, 2] * 0.04,
                colors=[_COLOR_ACTION],
                radii=0.001,
            ),
        )
        rr.log(
            f"{root}/{entity.name}/current",
            rr.Transform3D(
                translation=pos[current_step], mat3x3=rotations[current_step]
            ),
            rr.TransformAxes3D(axis_length=0.07),
        )


def require_policy_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"run was configured for {device}, but CUDA is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"run was configured for {device}, but only {torch.cuda.device_count()} "
                "CUDA device(s) are visible"
            )
    return device


def load_zarr_meta(zarr_path: Path) -> dict:
    meta_path = zarr_path.with_suffix(".meta.json")
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"{meta_path} missing; rebuild with scripts/port_wbc_mobile_zarr.py --overwrite"
        )
    meta = json.loads(meta_path.read_text())
    # Both schemas carry the raw-frame episode manifest this viewer needs; v3 only adds
    # the live-preprocessing contract (camera, RNG, crop frame, artifact provenance),
    # which nothing here reads. Pinning v2 alone rejected the zarrs the current porter
    # writes.
    if meta.get("schema") not in _MANIFEST_SCHEMAS:
        raise ValueError(
            f"{meta_path}: schema {meta.get('schema')!r} lacks the raw-frame episode "
            f"manifest (need one of {', '.join(_MANIFEST_SCHEMAS)}); rebuild the zarr "
            "with the current porter"
        )
    return meta


def validate_preprocessing_match(view_meta: dict, train_meta: dict, *,
                                 needs_point_mask: bool,
                                 position_condition_mode: str) -> None:
    """Reject a train/view pair that was not built by the same preprocessing path."""
    keys = (
        "schema", "num_points", "point_channels", "point_frame", "crop_min", "crop_max",
        "min_depth_m", "max_depth_m", "pool_size", "downsample", "sampler", "kdline_h",
        "start_idx", "fpsample_version", "state_axes", "state_frame", "action_axes",
        "action_frame", "fps", "seed",
    )
    for key in keys:
        if view_meta.get(key) != train_meta.get(key):
            raise ValueError(
                f"view/train zarr preprocessing mismatch at {key}: "
                f"{view_meta.get(key)!r} != {train_meta.get(key)!r}"
            )
    if needs_point_mask and view_meta.get("point_mask") != train_meta.get("point_mask"):
        raise ValueError("view/train zarr point_mask provenance differs")
    if (position_condition_mode != "none" and
            view_meta.get("env_state") != train_meta.get("env_state")):
        raise ValueError("view/train zarr env_state provenance differs")
    if (position_condition_mode == "grounding_tokens" and
            view_meta.get("env_dino") != train_meta.get("env_dino")):
        raise ValueError("view/train zarr env_dino provenance differs")


def _episode_manifest_entry(meta: dict, episode_index: int, start: int, end: int,
                            needs_point_mask: bool) -> dict:
    manifest = meta.get("episode_manifest")
    if not isinstance(manifest, list) or len(manifest) != int(meta.get("episodes", -1)):
        raise ValueError("zarr metadata has an invalid episode_manifest")
    if not 0 <= episode_index < len(manifest):
        raise ValueError(f"episode manifest has no index {episode_index}")
    entry = manifest[episode_index]
    expected = {
        "zarr_episode_index": episode_index,
        "zarr_frame_start": start,
        "zarr_frame_end_exclusive": end,
    }
    for key, value in expected.items():
        if entry.get(key) != value:
            raise ValueError(f"episode manifest {key}={entry.get(key)!r}, expected {value}")
    raw_start = int(entry.get("raw_frame_start", -1))
    raw_end = int(entry.get("raw_frame_end_exclusive", -1))
    if raw_start < 0 or raw_end - raw_start != end - start:
        raise ValueError(
            f"episode manifest raw range [{raw_start},{raw_end}) does not match "
            f"zarr range [{start},{end})"
        )
    hdf5_path = Path(str(entry.get("hdf5_path", "")))
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"manifest HDF5 not found: {hdf5_path}")
    if needs_point_mask:
        mask_path = Path(str(entry.get("mask_npz_path", "")))
        if not mask_path.is_file():
            raise FileNotFoundError(f"manifest SAM3.1 mask NPZ not found: {mask_path}")
    return entry


def load_maniflow_zarr_episode(zarr_path: Path, episode_index: int, policy: dict) -> dict:
    """One episode of the ManiFlow point-cloud zarr (``scripts/port_wbc_mobile_zarr.py``).

    Returns the STORED ``(N, num_points, 6)`` world XYZ+RGB cloud verbatim. Policy
    downsampling is reproduced separately after applying the training-zarr normalizer.
    """
    root = zarr.open(str(zarr_path), mode="r")
    for key in ("data/point_cloud", "data/state", "data/action", "meta/episode_ends"):
        if key not in root:
            raise ValueError(f"{zarr_path}: missing {key} (not a ManiFlow WBC zarr)")
    ends = np.asarray(root["meta/episode_ends"]).astype(np.int64)
    if not 0 <= episode_index < len(ends):
        raise ValueError(
            f"--episode_index {episode_index} out of range [0, {len(ends)}) for {zarr_path}")
    start = 0 if episode_index == 0 else int(ends[episode_index - 1])
    end = int(ends[episode_index])
    n = end - start
    if n <= 0:
        raise ValueError(f"{zarr_path}: episode {episode_index} is empty ([{start}, {end}))")

    meta = load_zarr_meta(zarr_path)
    manifest_entry = _episode_manifest_entry(
        meta, episode_index, start, end, policy["needs_point_mask"]
    )
    point_cloud = np.asarray(root["data/point_cloud"][start:end], dtype=np.float32)
    state = np.asarray(root["data/state"][start:end], dtype=np.float64)
    action = np.asarray(root["data/action"][start:end], dtype=np.float64)
    if point_cloud.ndim != 3 or point_cloud.shape[2] != 6:
        raise ValueError(f"{zarr_path}: point_cloud must be (T, P, 6), got {point_cloud.shape}")

    state_frame = meta.get("state_frame", "world")
    if state_frame not in ("base", "world"):
        raise ValueError(f"unexpected state_frame {state_frame!r} in zarr metadata")
    point_mask = None
    if policy["needs_point_mask"]:
        if "data/point_mask" not in root:
            raise ValueError(f"{zarr_path}: configured mask mode needs data/point_mask")
        point_mask = np.asarray(root["data/point_mask"][start:end])
        if point_mask.dtype != np.int8 or point_mask.shape != point_cloud.shape[:2]:
            raise ValueError(
                f"{zarr_path}: point_mask {point_mask.shape} {point_mask.dtype} does not "
                f"match {point_cloud.shape[:2]} int8"
            )
        if point_mask.min() < 0 or point_mask.max() > 2:
            raise ValueError(f"{zarr_path}: point_mask labels outside {{0,1,2}}")

    position_condition = None
    if policy["position_condition_mode"] != "none":
        if "data/env_state" not in root:
            raise ValueError(f"{zarr_path}: configured position mode needs data/env_state")
        env_state = np.asarray(root["data/env_state"][start:end], dtype=np.float32)
        expected = policy["object_nums"] * 3
        if env_state.shape != (n, expected) or not np.all(np.isfinite(env_state)):
            raise ValueError(f"{zarr_path}: env_state must be finite ({n}, {expected})")
        if not np.all(np.abs(env_state - env_state[0]) <= 1e-6):
            raise ValueError(f"{zarr_path}: env_state varies within episode {episode_index}")
        axes = meta.get("env_state", {}).get("axes")
        labels = _raw_vis_episode.position_condition_labels(axes, policy["object_nums"])
        position_condition = {
            "points": env_state.reshape(n, policy["object_nums"], 3),
            "labels": labels,
        }
    if policy["position_condition_mode"] == "grounding_tokens":
        if "data/env_dino" not in root:
            raise ValueError(f"{zarr_path}: grounding_tokens needs data/env_dino")
        env_dino = np.asarray(root["data/env_dino"][start:end], dtype=np.float32)
        expected = policy["object_nums"] * policy["position_dino_dim"]
        if env_dino.shape != (n, expected) or not np.all(np.isfinite(env_dino)):
            raise ValueError(f"{zarr_path}: env_dino must be finite ({n}, {expected})")
        if not np.all(env_dino == env_dino[0]):
            raise ValueError(f"{zarr_path}: env_dino varies within episode {episode_index}")
    return {
        "kind": "zarr", "label": f"{zarr_path} [episode {episode_index}: frames {start}..{end})",
        "N": n, "fps": float(meta.get("fps", 10)),
        "state": state, "action": action, "state_frame": state_frame,
        "point_cloud": point_cloud, "point_mask": point_mask,
        "meta": meta, "manifest_entry": manifest_entry,
        "position_condition": position_condition,
    }


def attach_raw_camera_streams(source: dict, *, needs_point_mask: bool) -> dict:
    """Load only the requested raw camera streams using the exact manifest window."""
    entry = source["manifest_entry"]
    start = int(entry["raw_frame_start"])
    end = int(entry["raw_frame_end_exclusive"])
    hdf5_path = Path(entry["hdf5_path"])
    keys = {
        "head_rgb": "obs/images/head_left_rgb",
        "depth": "obs/images/head_depth",
        "wrist_rgb": "obs/images/left_wrist_rgb",
        "intrinsic": "obs/images/intrinsic",
    }
    with h5py.File(hdf5_path, "r") as raw:
        for key in keys.values():
            if key not in raw:
                raise ValueError(f"{hdf5_path}: missing {key}")
        raw_frames = int(raw[keys["head_rgb"]].shape[0])
        if not 0 <= start < end <= raw_frames:
            raise ValueError(
                f"{hdf5_path}: manifest raw range [{start},{end}) outside {raw_frames} frames"
            )
        head_rgb = np.asarray(raw[keys["head_rgb"]][start:end])
        depth = np.asarray(raw[keys["depth"]][start:end])
        wrist_rgb = np.asarray(raw[keys["wrist_rgb"]][start:end])
        # Right wrist is optional: single-wrist takes only carry left_wrist_rgb.
        right_wrist_rgb = (
            np.asarray(raw["obs/images/right_wrist_rgb"][start:end])
            if "obs/images/right_wrist_rgb" in raw
            else None
        )
        intrinsic_raw = np.asarray(raw[keys["intrinsic"]])
    n = source["N"]
    if head_rgb.shape[0] != n or depth.shape[0] != n or wrist_rgb.shape[0] != n:
        raise ValueError(f"{hdf5_path}: manifest camera slice does not have {n} frames")
    if head_rgb.dtype != np.uint8 or wrist_rgb.dtype != np.uint8 or depth.dtype != np.uint16:
        raise ValueError(f"{hdf5_path}: expected uint8 RGB and uint16 depth")
    if right_wrist_rgb is not None and (
        right_wrist_rgb.shape[0] != n or right_wrist_rgb.dtype != np.uint8
    ):
        raise ValueError(
            f"{hdf5_path}: right_wrist_rgb slice {right_wrist_rgb.shape} "
            f"{right_wrist_rgb.dtype} is not {n} frames of uint8"
        )
    if intrinsic_raw.shape == (3, 3):
        intrinsic = np.repeat(intrinsic_raw[None], n, axis=0)
    elif intrinsic_raw.shape[0] >= end and intrinsic_raw.shape[1:] == (3, 3):
        intrinsic = intrinsic_raw[start:end]
    else:
        raise ValueError(f"{hdf5_path}: unsupported intrinsic shape {intrinsic_raw.shape}")

    mask_image = None
    if needs_point_mask:
        mask_path = Path(entry["mask_npz_path"])
        with np.load(mask_path, allow_pickle=False) as data:
            for key in ("masks", "legend", "n_frames"):
                if key not in data.files:
                    raise ValueError(f"{mask_path}: missing {key!r}")
            if str(data["legend"]) != _MASK_LEGEND:
                raise ValueError(f"{mask_path}: unexpected legend {str(data['legend'])!r}")
            masks = np.asarray(data["masks"])
            n_frames = int(data["n_frames"])
            raw_hdf5 = str(data["raw_hdf5"]) if "raw_hdf5" in data.files else None
        if (masks.dtype != np.uint8 or masks.ndim != 3 or
                n_frames != masks.shape[0] or n_frames != raw_frames):
            raise ValueError(f"{mask_path}: invalid full-frame SAM3.1 mask tensor")
        if raw_hdf5 is not None and Path(raw_hdf5).resolve() != hdf5_path.resolve():
            raise ValueError(f"{mask_path}: raw_hdf5={raw_hdf5!r} != {hdf5_path}")
        mask_image = masks[start:end]
        if mask_image.shape != depth.shape:
            raise ValueError(
                f"{mask_path}: mask slice {mask_image.shape} != depth {depth.shape}"
            )
    source = dict(source)
    source.update({
        "head_rgb": head_rgb,
        "depth": depth,
        "wrist_rgb": wrist_rgb,               # LEFT arm
        "right_wrist_rgb": right_wrist_rgb,   # RIGHT arm, None on single-wrist takes
        "intrinsic": intrinsic.astype(np.float64),
        "hw": head_rgb.shape[1:3],
        "mask_image": mask_image,
    })
    return source


def point_cloud_limits(train_zarr: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the limits normalizer's float32 scale/offset without loading all data."""
    root = zarr.open(str(train_zarr), mode="r")
    if "data/point_cloud" not in root:
        raise ValueError(f"{train_zarr}: missing data/point_cloud")
    points = root["data/point_cloud"]
    if points.ndim != 3 or points.shape[-1] != 6 or points.dtype != np.float32:
        raise ValueError(f"{train_zarr}: expected float32 point_cloud (T,N,6), got {points}")
    chunk_frames = points.chunks[0] if points.chunks else 32
    input_min = np.full(6, np.inf, dtype=np.float32)
    input_max = np.full(6, -np.inf, dtype=np.float32)
    for start in range(0, points.shape[0], chunk_frames):
        block = np.asarray(points[start:start + chunk_frames], dtype=np.float32)
        input_min = np.minimum(input_min, block.min(axis=(0, 1)))
        input_max = np.maximum(input_max, block.max(axis=(0, 1)))
    minimum = torch.from_numpy(input_min)
    maximum = torch.from_numpy(input_max)
    input_range = maximum - minimum
    ignore = input_range < 1e-4
    input_range[ignore] = 2.0
    scale = 2.0 / input_range
    offset = -1.0 - scale * minimum
    offset[ignore] = -minimum[ignore]
    return scale, offset


def select_policy_points(source: dict, policy: dict, sampler, device: torch.device,
                         scale: torch.Tensor, offset: torch.Tensor) -> dict:
    """Run the configured deterministic sampler in batches and return raw-world picks."""
    stored = source["point_cloud"]
    labels = source["point_mask"]
    target = policy["visual_cond_len"]
    if stored.shape[1] < target:
        raise ValueError(f"stored point count {stored.shape[1]} < visual_cond_len {target}")
    picked_points: list[np.ndarray] = []
    picked_indices: list[np.ndarray] = []
    picked_labels: list[np.ndarray] = []
    xyz_scale = scale[:3].to(device)
    xyz_offset = offset[:3].to(device)
    for start in range(0, source["N"], 128):
        raw = torch.from_numpy(stored[start:start + 128]).to(device)
        xyz = raw[..., :3] * xyz_scale + xyz_offset
        batch_labels = None
        if labels is not None:
            batch_labels = torch.from_numpy(labels[start:start + 128]).to(device)
        if policy["point_sampling_mode"] == "mask_stratified":
            _, selected_labels, indices, _ = sampler.mask_stratified_sample(
                xyz, batch_labels, num_points=target
            )
        elif stored.shape[1] > target:
            _, indices = sampler.fps_torch(xyz, num_points=target)
            selected_labels = None if batch_labels is None else torch.gather(
                batch_labels, 1, indices.long()
            )
        else:
            indices = torch.arange(stored.shape[1], device=device, dtype=torch.long)
            indices = indices.unsqueeze(0).expand(raw.shape[0], -1)
            selected_labels = batch_labels
        if indices.dtype != torch.long:
            raise RuntimeError(f"sampler returned index dtype {indices.dtype}, expected long")
        indices_np = indices.cpu().numpy()
        raw_np = stored[start:start + len(indices_np)]
        picked_points.append(raw_np[np.arange(len(indices_np))[:, None], indices_np])
        picked_indices.append(indices_np)
        if selected_labels is not None:
            picked_labels.append(selected_labels.cpu().numpy().astype(np.int8))
    return {
        "points": np.concatenate(picked_points),
        "indices": np.concatenate(picked_indices),
        "labels": np.concatenate(picked_labels) if picked_labels else None,
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
        default=None,
        help="Processed WBC LeRobot variant root (dexmate_wbc_eef_head from "
             "scripts/port_wbc_mobile_hdf5.py). Default when --zarr is omitted: "
             f"{_DEFAULT_DATASET_DIR}. Mutually exclusive with --zarr.",
    )
    parser.add_argument(
        "--zarr",
        type=str,
        default=None,
        help="ManiFlow point-cloud zarr (scripts/port_wbc_mobile_zarr.py), e.g. "
             "~/Dexmate/data/processed_wbc/maniflow/dexmate_wbc_test.zarr. "
             "Requires --maniflow-run-dir and v2 episode-manifest metadata.",
    )
    parser.add_argument(
        "--maniflow-run-dir",
        type=str,
        default=None,
        help="Training run containing .hydra/config.yaml. Zarr mode reads the active "
             "sampler/mask/grounding/action-frame switches and training-zarr path "
             "only from here.",
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
    if args.zarr is not None and args.maniflow_run_dir is None:
        parser.error("--zarr requires --maniflow-run-dir")
    if args.zarr is None and args.maniflow_run_dir is not None:
        parser.error("--maniflow-run-dir is only valid with --zarr")
    if args.zarr is not None and args.dataset_dir is not None:
        parser.error("--dataset_dir is LeRobot-only and cannot be combined with --zarr")

    policy = None
    policy_selection = None
    mof_transforms = None
    mof_action_layout = None
    if args.zarr is not None:
        zarr_path = Path(args.zarr).expanduser()
        run_dir = Path(args.maniflow_run_dir).expanduser().resolve()
        policy = load_maniflow_run(run_dir)
        sampler, checkout = import_run_sampler(run_dir)
        if policy["mof"] is not None:
            mof_transforms, mof_action_layout = import_run_frame_transforms(checkout)
        policy_device = require_policy_device(policy["device"])
        source = load_maniflow_zarr_episode(zarr_path, args.episode_index, policy)
        train_meta = load_zarr_meta(policy["train_zarr"])
        validate_preprocessing_match(
            source["meta"], train_meta,
            needs_point_mask=policy["needs_point_mask"],
            position_condition_mode=policy["position_condition_mode"],
        )
        source = attach_raw_camera_streams(
            source, needs_point_mask=policy["needs_point_mask"]
        )
        scale, offset = point_cloud_limits(policy["train_zarr"])
        policy_selection = select_policy_points(
            source, policy, sampler, policy_device, scale, offset
        )
        print("ManiFlow policy input configuration:")
        print(f"  run: {run_dir}")
        print(f"  sampler implementation: {Path(sampler.__file__).resolve()}")
        print(f"  point_sampling_mode: {policy['point_sampling_mode']}")
        print(f"  mask_channels: {policy['mask_channels']}")
        print(f"  needs_point_mask: {policy['needs_point_mask']}")
        print(f"  position_condition_mode: {policy['position_condition_mode']}")
        print(f"  action_frame_mode: {policy['action_frame_mode']}")
        if policy["mof"] is not None:
            print(
                f"  mof experts: {', '.join(policy['mof']['enabled_experts'])} "
                f"(canonical {policy['mof']['canonical_space']}, "
                f"router {policy['mof']['router_mode']}, "
                f"cloud_frame {policy['mof']['cloud_frame']})"
            )
        print(f"  visual_cond_len: {policy['visual_cond_len']}")
        print(f"  use_pc_color: {policy['use_pc_color']} (viewer uses unaugmented RGB)")
        print(f"  device: {policy_device}")
        print(f"  training zarr normalizer: {policy['train_zarr']}")
        print(f"  ManiFlow checkout: {checkout}")
    else:
        dataset_root = Path(args.dataset_dir or _DEFAULT_DATASET_DIR).expanduser()
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"--dataset_dir not a directory: {dataset_root}")
        source = load_lerobot_episode(dataset_root, args.episode_index)

    is_zarr = source["kind"] == "zarr"
    has_camera = is_zarr or "dataset" in source
    # Which wrist arms this source actually carries; drives both the blueprint panels
    # and the per-frame logging so a missing right camera never leaves an empty view.
    if not has_camera:
        wrist_arms: list[str] = []
    elif is_zarr:
        wrist_arms = ["left"] + (["right"] if source["right_wrist_rgb"] is not None else [])
    else:
        wrist_arms = [
            arm
            for arm in ("left", "right")
            if f"observation.images.{arm}_wrist_rgb" in source["dataset"].meta.features
        ]
    N = source["N"]
    state = source["state"]
    action = source["action"]
    state_frame = source["state_frame"]
    if state.shape[1] != len(STATE_AXES):
        raise ValueError(f"observation.state must be (N, {len(STATE_AXES)}), got {state.shape}")
    if action.shape[1] != len(ACTION_AXES):
        raise ValueError(f"action must be (N, {len(ACTION_AXES)}), got {action.shape}")
    print(f"source: {source['label']}")
    print(f"observation.state EEF/head frame: {state_frame}")

    position_condition = source["position_condition"]
    position_condition_points: np.ndarray | None = None
    position_condition_labels: list[str] = []
    position_condition_colors: np.ndarray | None = None
    if position_condition is not None:
        position_condition_points = np.asarray(position_condition["points"], dtype=np.float32)
        position_condition_labels = list(position_condition["labels"])
        if is_zarr:
            position_condition_colors = np.asarray([
                _COLOR_MASK_SRC if "src" in label.lower() else _COLOR_MASK_DST
                for label in position_condition_labels
            ], dtype=np.uint8)
        else:
            position_condition_colors = _raw_vis_episode.position_condition_colors(
                position_condition_labels
            )

    if has_camera:
        depth_stack = source["depth"]
        intrinsic_stack = source["intrinsic"]
        H, W = source["hw"]
        if not is_zarr:
            dataset = source["dataset"]
            extrinsic_stack = source["extrinsic"]
    if is_zarr:
        point_cloud_stack = source["point_cloud"]   # (N, num_points, 6) world XYZ + RGB[0,1]

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
    if is_zarr:
        # The zarr's world-frame head state is the camera extrinsic used by its porter.
        extrinsic_stack = state_world_head
    action_mat = {
        side: np.stack([pos6d_to_mat(action[i, _EEF_BLOCK[side]]) for i in range(N)])
        for side in _ARM_SIDES
    }                                                                                     # (N, 4, 4)
    action_head = np.stack([pos6d_to_mat(action[i, _HEAD_BLOCK]) for i in range(N)])       # (N, 4, 4)
    state_world_pos = {side: state_world[side][:, :3, 3] for side in _ARM_SIDES}
    action_pos = {side: action_mat[side][:, :3, 3] for side in _ARM_SIDES}
    base_xy = state[:, 29:31]
    base_yaw = state[:, 31]

    dt_s = 1.0 / source["fps"]
    if has_camera and not is_zarr:
        # Sanity: reconstructed world_T_zed must equal the porter's calib extrinsic.
        head_err = float(np.abs(state_world_head - extrinsic_stack).max())
        if head_err > 1e-4:
            raise ValueError(
                f"state head→world disagrees with calib extrinsic by {head_err:.2e} m "
                "(base composition / schema mismatch)"
            )
        print(f"episode {args.episode_index}: {N} frames | rgb {H}x{W} | "
              f"fps {source['fps']:g} | head→world vs calib max err {head_err:.2e} m")
    elif is_zarr:
        entry = source["manifest_entry"]
        print(
            f"episode {args.episode_index}: {N} frames | raw RGB {H}x{W} | "
            f"fps {source['fps']:g} | {entry['source']}/episode_{entry['raw_index']} "
            f"raw[{entry['raw_frame_start']}:{entry['raw_frame_end_exclusive']}]"
        )
    if is_zarr:
        # The stored cloud is world-frame, so the camera never moves it: the only cross
        # check available here is that every point sits inside the recorded crop box.
        meta = source["meta"]
        lo = np.asarray(meta["crop_min"])
        hi = np.asarray(meta["crop_max"])
        xyz = point_cloud_stack[..., :3]
        if not np.all((xyz > lo) & (xyz < hi)):
            raise ValueError(f"point_cloud escapes the crop {meta['crop_min']}..{meta['crop_max']} "
                             "recorded in the .meta.json")
        rgb_min, rgb_max = float(point_cloud_stack[..., 3:].min()), float(point_cloud_stack[..., 3:].max())
        if not (0.0 <= rgb_min and rgb_max <= 1.0):
            raise ValueError(f"point_cloud rgb outside [0,1]: [{rgb_min}, {rgb_max}]")
        print(f"  point_cloud {point_cloud_stack.shape[1]} pts x 6 (world XYZ + RGB) | "
              f"crop {meta['crop_min']}..{meta['crop_max']}")
    if not has_camera and not is_zarr:
        raise ValueError(f"unsupported source kind {source['kind']!r}")
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

    # ── MoF: GT action window in every enabled expert representation (zarr mode,
    #    action_frame_mode=mof runs only; transforms from the run checkout). ──
    mof_reps: dict[str, np.ndarray] | None = None
    mof_clouds: dict[str, np.ndarray] = {}
    mof_cloud_colors: np.ndarray | None = None
    mof_current_step = 0
    if mof_transforms is not None:
        mof_current_step = policy["mof"]["n_obs_steps"] - 1
        mof_reps = compute_mof_gt_representations(
            mof_transforms, mof_action_layout, state, action, policy["mof"]
        )
        # Same selection the "Policy RGB 3D" view shows -- the tower's actual input --
        # coloured by its SAM3.1 labels when the run consumes them, RGB otherwise.
        mof_clouds = compute_mof_family_clouds(
            mof_transforms, mof_action_layout, state,
            policy_selection["points"][..., :3], policy["mof"],
        )
        mof_cloud_colors = (
            _mask_label_colors(policy_selection["labels"])
            if policy_selection["labels"] is not None
            else (np.clip(policy_selection["points"][..., 3:], 0.0, 1.0) * 255.0
                  ).astype(np.uint8)
        )
    mof_experts = tuple(mof_reps) if mof_reps is not None else ()

    # ── Matplotlib figures (EEF xyz per arm + base top-down). ────────────────
    t_s = np.arange(N, dtype=np.float64) * dt_s
    figs: list[plt.Figure] = []
    if not is_zarr:
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

    playback_fps = source["fps"]
    lookback = max(400, min(N, 5000))
    lookahead = min(80, max(1, N // 50 + 5))
    hidden_head_entities = head_optional_hidden_overrides()
    hidden_depth_only = {_HEAD_DEPTH_ENTITY: rrb.EntityBehavior(visible=False)}

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

    bottom_row = rrb.Horizontal(
        _cursor_range("plots/gripper_left", "Left gripper",
                      (_GRIPPER_Y_MIN, _GRIPPER_Y_MAX)),
        _cursor_range("plots/gripper_right", "Right gripper",
                      (_GRIPPER_Y_MIN, _GRIPPER_Y_MAX)),
        _cursor_range("plots/base_pose", "Base pose"),
        column_shares=[1.0, 1.0, 1.0],
    )
    if is_zarr:
        common_3d = [
            "/world/action/**", "/world/state/**", "/world/base/**",
            "/world/base_path/**", "/world/position_condition/**", "/world/camera",
        ]
        center = (lo + hi) / 2.0
        eye_controls = rrb.EyeControls3D(
            position=(center + np.array([-2.0, -2.0, 1.5])).tolist(),
            look_target=center.tolist(),
            eye_up=(0.0, 0.0, 1.0),
        )
        views_3d = [
            rrb.Spatial3DView(
                origin="world", name="Stored RGB 3D",
                contents=[*common_3d, "/world/stored/**"],
                eye_controls=eye_controls,
            ),
            rrb.Spatial3DView(
                origin="world", name="Policy RGB 3D",
                contents=[*common_3d, "/world/policy_rgb/**"],
                eye_controls=eye_controls,
            ),
        ]
        if policy["needs_point_mask"]:
            views_3d.append(rrb.Spatial3DView(
                origin="world", name="Policy masks 3D",
                contents=[*common_3d, "/world/policy_masks/**"],
                eye_controls=eye_controls,
            ))
        head_contents = [
            "/world/camera/rgb", _HEAD_DEPTH_ENTITY,
            _HEAD_EEF_STATE_ENTITY, _HEAD_EEF_ACTION_ENTITY,
        ]
        if policy["needs_point_mask"]:
            head_contents.append("/world/camera/mask")
        views_2d = rrb.Grid(
            contents=[
                rrb.Spatial2DView(
                    origin="world/camera", name="Head RGB+mask",
                    contents=head_contents,
                    overrides=hidden_head_entities,
                ),
                *[rrb.Spatial2DView(origin=f"wrist/{_a}", name=f"{_a.capitalize()} wrist RGB")
                  for _a in wrist_arms],
            ],
            # head + one panel per wrist present.
            grid_columns=1 + len(wrist_arms),
            column_shares=[1.0] * (1 + len(wrist_arms)),
        )
        layout = rrb.Vertical(
            rrb.Grid(
                contents=views_3d,
                grid_columns=3,
                column_shares=[1.0, 1.0, 1.0],
            ),
            views_2d,
            bottom_row,
            row_shares=[2.2, 1.2, 1.0],
            name="Scene",
        )
        if mof_experts:
            # Same tab structure as vis_wbc_mof_prediction.py: base/left/right/world
            # are metric frames, rel_trans/rel_traj are representation spaces.
            layout = rrb.Tabs(
                layout,
                rrb.Grid(
                    *[
                        rrb.Spatial3DView(
                            origin=f"mof_frames/{name}",
                            name=(
                                name
                                if name in ("world", "base", "left", "right")
                                else f"{name} (representation space)"
                            ),
                        )
                        for name in mof_experts
                    ],
                    grid_columns=3,
                    name="MoF frames (GT action)",
                ),
            )
    else:
        world_view = rrb.Spatial3DView(
            origin="world", name="World (pcd + EEF + base odom)",
            overrides=hidden_depth_only,
        )
        top_row = (
            rrb.Horizontal(
                world_view,
                rrb.Spatial2DView(
                    origin="world/camera", name="Head RGB / depth",
                    overrides=hidden_depth_only,
                ),
                *[rrb.Spatial2DView(origin=f"wrist/{_a}", name=f"{_a.capitalize()} wrist RGB")
                  for _a in wrist_arms],
                # world + head + one column per wrist present.
                column_shares=[2.0, 1.3] + [1.3] * len(wrist_arms),
            )
            if has_camera else world_view
        )
        layout = rrb.Vertical(top_row, bottom_row, row_shares=[2.2, 1.0])
    rr.send_blueprint(
        rrb.Blueprint(
            layout,
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
    for name in mof_experts:
        rr.log(f"mof_frames/{name}", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(
            f"mof_frames/{name}/reference_axes",
            rr.Transform3D(translation=[0.0, 0.0, 0.0], mat3x3=np.eye(3)),
            rr.TransformAxes3D(axis_length=0.12),
            static=True,
        )
    if is_zarr and policy["needs_point_mask"]:
        rr.log(
            "world/camera",
            rr.AnnotationContext([
                (0, "background", (0, 0, 0, 0)),
                (1, "source", (*_COLOR_MASK_SRC, 255)),
                (2, "destination", (*_COLOR_MASK_DST, 255)),
            ]),
            static=True,
        )

    # ── Per-frame loop ───────────────────────────────────────────────────────
    for idx in list(range(0, N, args.stride)) + [N]:
        rr.set_time("frame", sequence=idx)

        if idx == N:
            rr.log("world/action", rr.Clear(recursive=True))
            rr.log("world/state", rr.Clear(recursive=True))
            rr.log("world/base", rr.Clear(recursive=True))
            rr.log("world/position_condition", rr.Clear(recursive=True))
            if is_zarr:
                rr.log("world/stored", rr.Clear(recursive=True))
                rr.log("world/policy_rgb", rr.Clear(recursive=True))
                if policy["needs_point_mask"]:
                    rr.log("world/policy_masks", rr.Clear(recursive=True))
            # /gt only: the static reference_axes sibling must survive the clear.
            for name in mof_experts:
                rr.log(f"mof_frames/{name}/gt", rr.Clear(recursive=True))
            if has_camera:
                rr.log("world/camera/eef_state_2d", rr.Clear(recursive=False))
                rr.log("world/camera/eef_action_2d", rr.Clear(recursive=False))
            continue

        if has_camera:
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

        if has_camera:
            if is_zarr:
                rgb = source["head_rgb"][idx]
                wrist_frames = {
                    arm: source["wrist_rgb" if arm == "left" else "right_wrist_rgb"][idx]
                    for arm in wrist_arms
                }
            else:
                frame = dataset[idx]
                rgb = _rgb_to_hwc(frame["observation.images.head_rgb"])
                wrist_frames = {
                    arm: _rgb_to_hwc(frame[f"observation.images.{arm}_wrist_rgb"])
                    for arm in wrist_arms
                }
            depth_mm = depth_stack[idx]
            rr.log("world/camera/rgb", rr.Image(rgb))
            rr.log("world/camera/depth", rr.DepthImage(depth_mm, meter=1000.0))
            if is_zarr and policy["needs_point_mask"]:
                rr.log(
                    "world/camera/mask",
                    rr.SegmentationImage(source["mask_image"][idx], opacity=0.5),
                )

            # Wrist RGB: plain 2D panel per arm (no calibration/depth for the wrists).
            for _arm, _wrist_frame in wrist_frames.items():
                rr.log(f"wrist/{_arm}/rgb", rr.Image(_wrist_frame))

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

        if is_zarr:
            # Stored is the porter's 1024-point tensor; policy_rgb is the exact configured
            # normalized-XYZ sampler selection, rendered back at its raw world XYZ/RGB.
            cloud = point_cloud_stack[idx]
            rr.log("world/stored/rgb", rr.Points3D(
                cloud[:, :3],
                colors=(np.clip(cloud[:, 3:], 0.0, 1.0) * 255.0).astype(np.uint8),
                radii=0.006,
            ))
            selected = policy_selection["points"][idx]
            rr.log("world/policy_rgb/rgb", rr.Points3D(
                selected[:, :3],
                colors=(np.clip(selected[:, 3:], 0.0, 1.0) * 255.0).astype(np.uint8),
                radii=0.006,
            ))
            selected_labels = policy_selection["labels"]
            if selected_labels is not None:
                frame_labels = selected_labels[idx]
                mask_colors = _mask_label_colors(frame_labels)
                rr.log("world/policy_masks/labels", rr.Points3D(
                    selected[:, :3], colors=mask_colors, radii=0.006,
                ))
                if policy["point_sampling_mode"] == "mask_stratified":
                    counts = {
                        label: int((frame_labels == label).sum())
                        for label in (1, 2, 0)
                    }
                    quota_text = (
                        f"q_src={counts[1]} | q_dst={counts[2]} | "
                        f"q_bg={counts[0]} | total={len(frame_labels)}"
                    )
                    names = {1: "src", 2: "dst", 0: "bg"}
                    shortfalls = [
                        f"{names[label]}={counts[label]}/{nominal}"
                        for label, nominal in _NOMINAL_MASK_QUOTAS.items()
                        if counts[label] < nominal
                    ]
                    warning = bool(shortfalls)
                    if warning:
                        quota_text += "\n⚠ QUOTA SHORTFALL: " + ", ".join(shortfalls)
                    anchor = np.asarray(
                        [[lo[0], hi[1], hi[2]]], dtype=np.float32
                    )
                    rr.log("world/policy_masks/quota", rr.Points3D(
                        anchor,
                        colors=[_COLOR_ACTION if warning else (255, 255, 255)],
                        radii=0.004,
                        labels=[quota_text],
                        show_labels=True,
                    ))
        elif has_camera:
            # Point cloud (manual unproject + voxel downsample), world frame.
            depth_m = depth_mm.astype(np.float32) / 1000.0
            pts, mask = unproject_depth(depth_m, K, world_t_cam)
            cols = rgb[mask]
            pts, cols = voxel_downsample(pts, cols, args.voxel)
            rr.log("world/pcd", rr.Points3D(pts, colors=cols, radii=0.003))
        else:
            raise ValueError(f"unsupported source kind {source['kind']!r}")

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

        if mof_reps is not None:
            for name, reps in mof_reps.items():
                root = f"mof_frames/{name}/gt"
                _log_mof_horizon(root, reps[idx], mof_action_layout, mof_current_step)
                rr.log(f"{root}/cloud", rr.Points3D(
                    mof_clouds[name][idx],
                    colors=mof_cloud_colors[idx],
                    radii=_MOF_CLOUD_RADIUS,
                ))

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
