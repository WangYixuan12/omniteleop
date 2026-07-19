"""Offline WBC-policy inference vs ground truth, visualized in rerun + matplotlib.

Sibling to ``scripts/vis_episode_processed_wbc.py`` (the GT-only WBC dataset
viewer); shares its layout, world-frame reconciliation, and helpers. The
difference: this viewer runs ``scripts/wbc_policy_rollout.py``'s EXACT inference
path (``_PolicyBundle.select_action`` + ``split_policy_action``) offline over a
processed ``dexmate_wbc_eef_head`` episode, so the rollout's policy stack is
verified without the robot, and it overlays the policy's PREDICTED action against
the dataset's GROUND-TRUTH action.

For each dataset frame the policy is fed the real ``observation.state`` (32-D) +
head/wrist frames and emits its 29-D world-frame action. This is open-loop w.r.t.
actions (teacher-forced observations): every step sees the real recorded
observation, exactly like a behavioral-cloning validation pass. Because the
policy is driven through ``select_action`` frame by frame at the dataset's 10 Hz
cadence, the emitted actions are the SAME ones the rollout would command at each
10 Hz policy tick (diffusion re-plans every ``n_action_steps`` and holds in
between, so the prediction is a staircase).

Frames (PLAN.md Conventions):
- ``action`` (29-D) is WORLD-frame (engage-origin): left/right EEF pos3+rot6+grip,
  head pos3+rot6. Both GT and predicted actions plot directly in the world.
- ``observation.state`` (32-D) is base-frame achieved FK + odometry base pose
  (dims 29-31). Its base pose places the base triad + odometry path (context) and
  its calib sidecar gives world_T_cam for the depth point cloud.

Colors match the sibling dataset viewer where they overlap:
  red = ground-truth recorded ``action``  (``vis_episode_processed_wbc`` action),
  magenta = policy prediction,  gold = mobile-base odometry,
  white = per-entity GT<->prediction error line.

Panels:
  * 3D world -- depth point cloud, per-entity (left, right, head) GT pose (red)
    vs predicted pose (magenta) as spheres + coordinate frames, a white error
    line between them, faint full-trajectory trails, base triad + gold odometry
    path, and the head-camera pinhole.
  * Head RGB / depth -- overlaid with the projected GT (red) and predicted
    (magenta) EEF pixels.
  * Wrist RGB -- plain 2D (no calibration/depth).
  * Gripper time series (left, right) -- GT (red) vs prediction (magenta),
    both as the {0,1} commands after ``split_policy_action`` binarization
    (same threshold as the live rollout), not the soft policy logits.

Matplotlib: per entity, x/y/z position + geodesic rotation-error curves, GT vs
prediction, annotated with translation MAE (mm) and rotation MAE (deg).

Sidecars (written by the porter): ``debug/depth`` and ``debug/calib`` npz, same
as ``vis_episode_processed_wbc.py``.

Run in the ``dexmate_lerobot`` env (needs lerobot + rerun + the checkpoint deps).

Usage::

    python scripts/vis_wbc_policy_prediction.py \
        --policy-path /home/yixuan/Dexmate/model/dp/dexmate_wbc_eef_head_toy/checkpoints/last/pretrained_model \
        --dataset_dir /home/yixuan/Dexmate/data/processed_wbc/dexmate_wbc_eef_head \
        --episode_index 0
    # headless over SSH: add --save /tmp/wbc_pred.rrd  (open later: rerun FILE.rrd)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from omniteleop.wbc_policy_format import ACTION_AXES, STATE_AXES, base_pose_to_mat

# Per-entity column layout in the 29-D action / 32-D state (shared prefix). Each
# EEF entity: pos [b:b+3], rot6 [b+3:b+9], gripper at b+9. Head: no gripper.
_ENTITIES = {
    "left": {"pos": 0, "rot": 3, "grip": 9},
    "right": {"pos": 10, "rot": 13, "grip": 19},
    "head": {"pos": 20, "rot": 23, "grip": None},
}
_ARM_SIDES = ("left", "right")
_BASE_BLOCK = slice(29, 32)  # observation.state base pose (x, y, yaw)

_COLOR_GT = (255, 0, 0)       # red     — ground-truth recorded action (sibling convention)
_COLOR_PRED = (230, 60, 230)  # magenta — policy prediction
_COLOR_ERR = (235, 235, 235)  # white   — GT<->prediction error line
_COLOR_BASE = (230, 190, 0)   # gold    — mobile-base odometry
_GT_MPL = (0.85, 0.0, 0.0)
_PRED_MPL = (0.85, 0.2, 0.85)

_GRIPPER_Y_MIN = -0.2
_GRIPPER_Y_MAX = 1.1
_PLAYBACK_FPS = 10.0


def gram_schmidt_6d_to_R(r6: np.ndarray) -> np.ndarray:
    """Recover (3, 3) rotation from Zhou et al. 6-D representation [R[:,0], R[:,1]]."""
    a1, a2 = np.asarray(r6[:3], float), np.asarray(r6[3:6], float)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def geodesic_deg(r6_a: np.ndarray, r6_b: np.ndarray) -> float:
    """Geodesic angle (deg) between two 6-D rotations."""
    Ra, Rb = gram_schmidt_6d_to_R(r6_a), gram_schmidt_6d_to_R(r6_b)
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# ── Generic helpers copied verbatim from vis_episode_processed_wbc.py so the WBC
#    viewers share identical projection / unproject / downsample logic. ─────────
def project_world_to_pixel(pts_world, K, world_t_cam, z_min=0.01):
    """Project (N, 3) world points into pixel coords (camera frame: +x right,
    +y down, +z forward). Returns ``(uv, valid)``."""
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


def unproject_depth(depth_m, K, world_t_cam, max_depth=3.0):
    """Unproject a depth image (m) to world XYZ. Returns (pts, mask)."""
    H, W = depth_m.shape
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    mask = (depth_m > 0.05) & (depth_m < max_depth)
    z = depth_m[mask]
    x = (u[mask] - K[0, 2]) / K[0, 0] * z
    y = (v[mask] - K[1, 2]) / K[1, 1] * z
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)
    return (world_t_cam @ pts_cam.T).T[:, :3].astype(np.float32), mask


def voxel_downsample(pts, colors, voxel_size):
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
        return (out_pts / cnt.unsqueeze(1)).cpu().numpy(), (
            out_cols / cnt.unsqueeze(1)).cpu().numpy().astype(np.uint8)
    except Exception:
        idx = np.random.choice(len(pts), min(len(pts), 50000), replace=False)
        return pts[idx], colors[idx]


def _rgb_to_hwc(t) -> np.ndarray:
    """Normalize an RGB observation to (H, W, 3) uint8."""
    a = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    if a.ndim == 3 and a.shape[0] == 3:
        a = np.transpose(a, (1, 2, 0))
    return a.astype(np.uint8) if a.dtype != np.uint8 else a


def _load_rollout_module():
    """Load wbc_policy_rollout.py as a module (reuse its exact inference path)."""
    path = Path(__file__).resolve().parent / "wbc_policy_rollout.py"
    spec = importlib.util.spec_from_file_location("wbc_policy_rollout_for_vis", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclass string-annotation resolution needs this
    spec.loader.exec_module(module)
    return module


def run_inference(bundle, split_policy_action, dataset, N: int,
                  use_wrist: bool) -> tuple[np.ndarray, np.ndarray]:
    """Drive the policy over the episode; return (pred_actions (N,29), infer_ms (N,)).

    Each prediction is pushed through the rollout's OWN ``split_policy_action`` so
    this offline gate exercises the EXACT decode the robot runs at
    ``wbc_policy_rollout.py`` (finiteness validation + ``pos6d_to_mat`` rotation
    decode + gripper binarization at ``GRIPPER_BINARY_THRESHOLD``). Gripper dims
    9/19 in the returned array are the commanded {0,1} values, not the soft
    policy output. If the policy emits an action the hardware rollout would
    reject, this validation aborts here instead of on the robot.
    """
    bundle.reset()
    preds = np.empty((N, len(ACTION_AXES)), dtype=np.float32)
    infer_ms = np.empty(N, dtype=np.float64)
    for i in range(N):
        fr = dataset[i]
        state = fr["observation.state"].numpy().astype(np.float32)
        head = _rgb_to_hwc(fr["observation.images.head_rgb"])
        wrist = _rgb_to_hwc(fr["observation.images.wrist_rgb"]) if use_wrist else None
        t0 = time.perf_counter()
        action = bundle.select_action(state, head, wrist)
        infer_ms[i] = (time.perf_counter() - t0) * 1000.0
        # Same decode path the rollout commands at each 10 Hz tick; raises on a
        # non-finite / malformed action exactly as the robot would. Overwrite
        # gripper dims with the binarized commands the robot actually receives.
        decoded = split_policy_action(action)
        action = np.asarray(action, dtype=np.float32).copy()
        action[_ENTITIES["left"]["grip"]] = decoded["left_gripper"]
        action[_ENTITIES["right"]["grip"]] = decoded["right_gripper"]
        preds[i] = action
        if (i + 1) % 50 == 0:
            print(f"  inference {i + 1}/{N}", end="\r")
    print()
    return preds, infer_ms


def print_metrics(gt: np.ndarray, pred: np.ndarray) -> None:
    """Per-entity translation/rotation error + binary gripper accuracy."""
    print("\n=== prediction vs ground-truth action (open-loop, teacher-forced obs) ===")
    for name, cols in _ENTITIES.items():
        p, r = cols["pos"], cols["rot"]
        d = np.linalg.norm(gt[:, p:p + 3] - pred[:, p:p + 3], axis=1) * 1000.0  # mm
        rot = np.array([geodesic_deg(gt[t, r:r + 6], pred[t, r:r + 6]) for t in range(len(gt))])
        line = (f"  {name:5s}  pos MAE {d.mean():6.1f} mm  p95 {np.percentile(d, 95):6.1f} mm"
                f"  |  rot MAE {rot.mean():5.1f} deg")
        if cols["grip"] is not None:
            g = cols["grip"]
            # pred grippers are already the rollout's {0,1} commands (see run_inference).
            gmae = float(np.abs(gt[:, g] - pred[:, g]).mean())
            gacc = float((gt[:, g] == pred[:, g]).mean())
            line += f"  |  grip MAE {gmae:.3f} acc {gacc * 100:4.0f}%"
        print(line)


def build_mpl_figs(t_s, gt, pred) -> list[plt.Figure]:
    """One figure per entity: x/y/z position + geodesic rotation-error, GT vs pred."""
    figs = []
    for name, cols in _ENTITIES.items():
        p, r = cols["pos"], cols["rot"]
        rot = np.array([geodesic_deg(gt[k, r:r + 6], pred[k, r:r + 6]) for k in range(len(gt))])
        fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
        for ax_i, axis_name in enumerate("xyz"):
            g, pr = gt[:, p + ax_i], pred[:, p + ax_i]
            mae_mm = np.abs(g - pr).mean() * 1000.0
            axes[ax_i].plot(t_s, g, color=_GT_MPL, lw=1.6, label="ground truth")
            axes[ax_i].plot(t_s, pr, color=_PRED_MPL, lw=1.3, label="prediction")
            axes[ax_i].text(0.01, 0.95, f"MAE = {mae_mm:.1f} mm", transform=axes[ax_i].transAxes,
                            fontsize=9, va="top",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))
            axes[ax_i].set_ylabel(f"{name}_{axis_name} (m)")
            axes[ax_i].grid(alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8)
        axes[3].plot(t_s, rot, color=(0.4, 0.2, 0.7), lw=1.3)
        axes[3].text(0.01, 0.95, f"rot MAE = {rot.mean():.1f} deg", transform=axes[3].transAxes,
                     fontsize=9, va="top",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))
        axes[3].set_ylabel(f"{name} rot err (deg)")
        axes[3].set_xlabel("t (s)")
        axes[3].grid(alpha=0.3)
        fig.suptitle(f"{name} EEF/head — action prediction vs ground truth")
        fig.tight_layout()
        figs.append(fig)
    return figs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-path", required=True,
                        help="LeRobot checkpoint (WBC 32/29 schema).")
    parser.add_argument("--dataset_dir",
                        default="/home/yixuan/Dexmate/data/processed_wbc/dexmate_wbc_eef_head",
                        help="processed dexmate_wbc_eef_head dataset root.")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--voxel", type=float, default=0.008, help="PCD voxel size (m).")
    parser.add_argument("--stride", type=int, default=1, help="log every Nth frame (>=1).")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="cap frames (0 = whole episode).")
    parser.add_argument("--save", type=str, default=None,
                        help="stream to this .rrd headlessly instead of spawning a viewer.")
    parser.add_argument("--no_mpl", action="store_true", help="skip the matplotlib figures.")
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError(f"--stride must be >= 1, got {args.stride}")

    root = Path(args.dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"--dataset_dir not a directory: {root}")

    dataset = LeRobotDataset(repo_id=root.name, root=root,
                             episodes=[args.episode_index], return_uint8=True)
    N = len(dataset)
    if N == 0:
        raise ValueError(f"episode {args.episode_index} is empty under {root}")
    if args.max_frames > 0:
        N = min(N, args.max_frames)

    # Self-describing schema guard: this viewer is specific to the WBC eef+head layout.
    state_axes = list(dataset.meta.features["observation.state"]["names"]["axes"])
    action_axes = list(dataset.meta.features["action"]["names"]["axes"])
    if state_axes != list(STATE_AXES) or action_axes != list(ACTION_AXES):
        raise ValueError(
            f"dataset axes do not match the WBC schema; got state={len(state_axes)}D "
            f"action={len(action_axes)}D. This viewer is for dexmate_wbc_eef_head only."
        )
    for key in ("observation.images.head_rgb", "observation.images.wrist_rgb"):
        if key not in dataset.meta.features:
            raise ValueError(f"{root.name} is missing {key}.")

    gt_action = np.stack([dataset[i]["action"].numpy() for i in range(N)]).astype(np.float32)
    gt_state = np.stack([dataset[i]["observation.state"].numpy() for i in range(N)]).astype(np.float64)

    # Camera sidecars (world_T_cam extrinsic + intrinsic + depth), same as the sibling viewer.
    calib = np.load(root / "debug" / "calib" / f"episode_{args.episode_index:06d}.npz")
    extrinsic = calib["extrinsic"].astype(np.float64)   # (T, 4, 4) world_T_cam
    intrinsic = calib["intrinsic"].astype(np.float64)   # (T, 3, 3)
    depth_stack = np.load(root / "debug" / "depth" / f"episode_{args.episode_index:06d}.npz")["depth"]
    for nm, arr in (("extrinsic", extrinsic), ("intrinsic", intrinsic), ("depth", depth_stack)):
        if arr.shape[0] < N:
            raise ValueError(f"{nm} sidecar has {arr.shape[0]} frames < {N}")

    # Base triad / odometry path from the state base pose (context for the world scene).
    world_t_base = np.stack([base_pose_to_mat(gt_state[i, _BASE_BLOCK]) for i in range(N)])
    base_xy = gt_state[:, 29:31]

    # --- Load the rollout's OWN policy bundle and run its inference path offline. ---
    roll = _load_rollout_module()
    print(f"[vis] loading policy via wbc_policy_rollout._PolicyBundle: {args.policy_path}")
    bundle = roll._PolicyBundle(args.policy_path)  # noqa: SLF001 -- deliberate reuse
    print(f"[vis] policy on {bundle.device}; head+"
          f"{'wrist' if bundle.use_wrist else 'no-wrist'} @ {bundle.image_hw}")
    pred_action, infer_ms = run_inference(
        bundle, roll.split_policy_action, dataset, N, bundle.use_wrist
    )
    print(f"[vis] inference: {infer_ms.mean():.1f} ms/frame mean, "
          f"{np.percentile(infer_ms, 95):.1f} ms p95")
    print_metrics(gt_action, pred_action)

    sample_rgb = _rgb_to_hwc(dataset[0]["observation.images.head_rgb"])
    H, W = sample_rgb.shape[:2]
    print(f"[vis] episode {args.episode_index}: {N} frames  |  rgb {H}x{W}")

    # ── rerun setup ──────────────────────────────────────────────────────────
    rr.init("vis_wbc_policy_prediction", spawn=args.save is None)
    if args.save is not None:
        rr.save(args.save)
        print(f"[vis] streaming to {args.save} (headless)")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    lookback = max(400, min(N, 5000))
    lookahead = min(80, max(1, N // 50 + 5))

    def _gripper_view(origin, name):
        return rrb.TimeSeriesView(
            origin=origin, name=name,
            axis_y=rrb.ScalarAxis(range=(_GRIPPER_Y_MIN, _GRIPPER_Y_MAX), zoom_lock=True),
            time_ranges=[rrb.VisibleTimeRange(
                "frame",
                start=rrb.TimeRangeBoundary.cursor_relative(seq=-lookback),
                end=rrb.TimeRangeBoundary.cursor_relative(seq=lookahead))],
        )

    rr.send_blueprint(rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", name="World: GT (red) vs pred (magenta)"),
                rrb.Spatial2DView(origin="world/camera", name="Head RGB / depth"),
                rrb.Spatial2DView(origin="wrist", name="Wrist RGB"),
                column_shares=[2.0, 1.3, 1.3],
            ),
            rrb.Horizontal(
                _gripper_view("plots/gripper_left", "Left gripper"),
                _gripper_view("plots/gripper_right", "Right gripper"),
            ),
            row_shares=[2.2, 1.0],
        ),
        rrb.TimePanel(timeline="frame", fps=_PLAYBACK_FPS),
    ))

    for side in _ARM_SIDES:
        rr.log(f"plots/gripper_{side}",
               rr.SeriesLines(colors=[_COLOR_GT, _COLOR_PRED],
                              names=[f"{side} GT", f"{side} pred"], widths=[2.0, 2.0]),
               static=True)

    # Static context: full GT/pred trajectory trails + the gold base odometry path.
    for name, cols in _ENTITIES.items():
        p = cols["pos"]
        rr.log(f"world/gt/{name}/trail",
               rr.Points3D(gt_action[:, p:p + 3], colors=[_COLOR_GT], radii=0.002), static=True)
        rr.log(f"world/pred/{name}/trail",
               rr.Points3D(pred_action[:, p:p + 3], colors=[_COLOR_PRED], radii=0.002), static=True)
    base_path_xyz = np.column_stack([base_xy, np.zeros(N)]).astype(np.float32)
    rr.log("world/base_path", rr.LineStrips3D([base_path_xyz], colors=[_COLOR_BASE], radii=0.008),
           static=True)

    # ── per-frame loop ─────────────────────────────────────────────────────────
    for idx in list(range(0, N, args.stride)) + [N]:
        rr.set_time("frame", sequence=idx)
        if idx == N:
            for name in _ENTITIES:
                rr.log(f"world/gt/{name}/cur", rr.Clear(recursive=True))
                rr.log(f"world/pred/{name}/cur", rr.Clear(recursive=True))
                rr.log(f"world/err/{name}", rr.Clear(recursive=False))
            rr.log("world/base", rr.Clear(recursive=True))
            rr.log("world/camera/eef_gt_2d", rr.Clear(recursive=False))
            rr.log("world/camera/eef_pred_2d", rr.Clear(recursive=False))
            continue

        world_t_cam = extrinsic[idx]
        K = intrinsic[idx]
        rr.log("world/camera", rr.Transform3D(translation=world_t_cam[:3, 3],
                                              mat3x3=world_t_cam[:3, :3]))
        rr.log("world/camera", rr.Pinhole(image_from_camera=K.astype(np.float32), width=W, height=H))

        # Mobile base triad (odometry pose) — context for the moving world scene.
        # Rerun >=0.34: axes are a sibling TransformAxes3D, not Transform3D(axis_length=...).
        base_tf = world_t_base[idx]
        rr.log("world/base",
               rr.Transform3D(translation=base_tf[:3, 3], mat3x3=base_tf[:3, :3]),
               rr.TransformAxes3D(axis_length=0.25))

        # Per-entity current GT (red) vs predicted (magenta) pose + error line.
        for name, cols in _ENTITIES.items():
            p, r = cols["pos"], cols["rot"]
            g_pos, p_pos = gt_action[idx, p:p + 3], pred_action[idx, p:p + 3]
            rr.log(f"world/gt/{name}/cur", rr.Points3D(g_pos[None], colors=[_COLOR_GT], radii=0.012))
            rr.log(f"world/gt/{name}/cur/frame",
                   rr.Transform3D(translation=g_pos,
                                  mat3x3=gram_schmidt_6d_to_R(gt_action[idx, r:r + 6])),
                   rr.TransformAxes3D(axis_length=0.07))
            rr.log(f"world/pred/{name}/cur", rr.Points3D(p_pos[None], colors=[_COLOR_PRED], radii=0.012))
            rr.log(f"world/pred/{name}/cur/frame",
                   rr.Transform3D(translation=p_pos,
                                  mat3x3=gram_schmidt_6d_to_R(pred_action[idx, r:r + 6])),
                   rr.TransformAxes3D(axis_length=0.07))
            rr.log(f"world/err/{name}",
                   rr.LineStrips3D([np.stack([g_pos, p_pos])], colors=[_COLOR_ERR], radii=0.0015))

        # Head RGB + depth + wrist RGB.
        frame = dataset[idx]
        rgb = _rgb_to_hwc(frame["observation.images.head_rgb"])
        depth_mm = depth_stack[idx]
        rr.log("world/camera/rgb", rr.Image(rgb))
        rr.log("world/camera/depth", rr.DepthImage(depth_mm, meter=1000.0))
        rr.log("wrist/rgb", rr.Image(_rgb_to_hwc(frame["observation.images.wrist_rgb"])))

        # 2D overlay: project GT (red) + predicted (magenta) left/right EEF into the head image.
        for tag, src, color in (("eef_gt_2d", gt_action, _COLOR_GT),
                                ("eef_pred_2d", pred_action, _COLOR_PRED)):
            pts = np.stack([src[idx, _ENTITIES[s]["pos"]:_ENTITIES[s]["pos"] + 3] for s in _ARM_SIDES])
            uv, valid = project_world_to_pixel(pts, K, world_t_cam)
            if valid.any():
                rr.log(f"world/camera/{tag}", rr.Points2D(uv[valid], colors=[color], radii=2.5))
            else:
                rr.log(f"world/camera/{tag}", rr.Clear(recursive=False))

        # Colored pointcloud from head depth (world frame via calib extrinsic).
        pts, mask = unproject_depth(depth_mm.astype(np.float32) / 1000.0, K, world_t_cam)
        pts, cols_pc = voxel_downsample(pts, rgb[mask], args.voxel)
        rr.log("world/pcd", rr.Points3D(pts, colors=cols_pc, radii=0.003))

        # Gripper scalars: GT vs predicted (left, right).
        for side in _ARM_SIDES:
            g = _ENTITIES[side]["grip"]
            rr.log(f"plots/gripper_{side}",
                   rr.Scalars([float(gt_action[idx, g]), float(pred_action[idx, g])]))

    if not args.no_mpl and args.save is None:
        dt_s = 1.0 / float(dataset.fps) if dataset.fps else 0.1
        build_mpl_figs(np.arange(N) * dt_s, gt_action, pred_action)
        print("[vis] showing matplotlib figures (close them to exit) ...")
        plt.show()


if __name__ == "__main__":
    main()
