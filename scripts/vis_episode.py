"""Visualise a recorded VR teleoperation episode (or a policy rollout) in rerun.

Logs RGB-D, the colored point cloud, robot meshes, and EEF obs/action markers
under a single ``frame`` timeline so the rerun viewer's scrubber doubles as
prev/next/jump navigation.

Two episode layouts are supported via ``--deploy``:

* default (teleop / ``raw_data`` layout, written by ``leader/vr_reader.py`` or
  ``scripts/wbc_vr_robot.py``)::

      obs/images/{head_left_rgb, head_depth, left_wrist_rgb}
      obs/images/intrinsic   (3, 3) static (wbc_vr_robot) or (N, 3, 3)
      obs/images/extrinsic   optional base_t_cam; recomputed by FK when absent
      obs/joint/{torso, left_arm, right_arm, head[, chassis_*]}
      obs/base/pose          optional (N, 3) = (x, y, yaw), present with a moving base
      obs/gripper/{left, right}
      action/joint/{left_arm, right_arm, head, chassis_*}
      action/gripper/{left, right}
      timestamp_ns  (legacy episodes may also carry a top-level ``stage``)

  EEF obs/action markers come from forward kinematics on the *both-arm* joint
  streams (``L_ee`` / ``R_ee``): obs = ``obs/joint/*_arm``, action =
  ``action/joint/*_arm`` (with torso/head taken from the obs streams).

* ``--deploy`` (policy rollout, written by ``follower/policy_rollout.py``)::

      time/{begin_build_observation, begin_inference, finish_inference, publish_command, ...}
      obs/images/{head_left_rgb, head_depth, left_wrist_rgb, intrinsic, extrinsic}
      obs/joint/{left_arm, right_arm}
      obs/eef_9d/{left, right}
      obs/gripper/{left, right}
      obs/position_condition/{stage, mask, condition_6d}  (optional)
      action/eef_9d/{left, right}
      action/gripper/{left, right}

  EEF markers are read directly for both arms: obs = ``obs/eef_9d/{left,right}``
  (blue), action = ``action/eef_9d/{left,right}`` (red).
  Position-conditioned rollouts also show the selected stage/mask and the
  active 6-D [before_xyz, after_xyz] condition as text plus 3D before/after
  markers and a 2D overlay projected onto ``head_left_rgb``.
  Torso/head joints aren't recorded, so they're held at ``INIT_*`` for
  robot-mesh FK. The saved ``obs/images/{intrinsic, extrinsic}`` calibration is
  required in this mode -- the script errors out rather than falling back to
  hardcoded ZED intrinsics / FK camera extrinsics (a single shared matrix is
  broadcast over frames). A ``debug/latency.png`` is written from the ``time``
  group.

When ``obs/base/pose`` is present (``wbc_vr_robot.py --enable base``), the head-camera
extrinsic (``world_t_cam = world_t_base · base_t_cam``), robot meshes, colored point
cloud, and EEF markers are all composed with the per-frame ``world_t_base``, so a *driven*
episode renders in a fixed world frame (the base pose at engage): static scene geometry
stays put while the robot drives through it. Each frame is still rendered on its own (the
cloud is replaced, not accumulated). Episodes without ``obs/base/pose`` fall back to
identity -> base-relative rendering (unchanged).

EEF markers represent the *future* trajectory from each frame onwards (obs blue,
action red), so the point cloud shrinks one point at a time from the start as
the scrubber advances; a coordinate frame marks the current-frame pose.

Usage::

    python scripts/vis_episode.py --hdf5 data/episode_0.hdf5
    python scripts/vis_episode.py --hdf5 deploy/act/0/episode_0.hdf5 --deploy
    python scripts/vis_episode.py --hdf5 data/episode_0.hdf5 --save out.rrd

Robot link geometry is logged once as static and animated per-frame with a
Transform3D, rather than re-sending full meshes each frame -- this keeps the
recording (and viewer memory) ~20x smaller with no visual change.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5
from yixuan_utilities.kinematics_helper import KinHelper
from yixuan_utilities.robot_mesh_generator import RobotMeshGenerator

from omniteleop.common.vr_mode_const import INIT_HEAD_JOINTS, INIT_TORSO_JOINTS

# ── joint name order expected by convert_to_sapien_joint_order ───────────────
_WHEEL_NAMES = ["B_wheel_j1", "B_wheel_j2", "R_wheel_j1", "R_wheel_j2", "L_wheel_j1", "L_wheel_j2"]
_OBS_NAMES = [
    "torso_j1",
    "torso_j2",
    "torso_j3",
    "L_arm_j1",
    "L_arm_j2",
    "L_arm_j3",
    "L_arm_j4",
    "L_arm_j5",
    "L_arm_j6",
    "L_arm_j7",
    "R_arm_j1",
    "R_arm_j2",
    "R_arm_j3",
    "R_arm_j4",
    "R_arm_j5",
    "R_arm_j6",
    "R_arm_j7",
    "head_j1",
    "head_j2",
    "head_j3",
]

# Marker colors: obs = blue, action = red.
_OBS_COLOR = (0, 80, 255)
_ACTION_COLOR = (255, 0, 0)

# Deploy position-condition markers (3D world + 2D head RGB overlay).
_POS_COND_COLOR_BEFORE = (0, 200, 255)
_POS_COND_COLOR_AFTER = (255, 225, 25)
_POS_COND_RADIUS_3D = 0.015
_POS_COND_RADIUS_2D = 2.5


def gram_schmidt_6d_to_R(r6: np.ndarray) -> np.ndarray:
    """Recover (3, 3) rotation from Zhou et al. 6-D representation."""
    a1, a2 = r6[:3], r6[3:6]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


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


def project_world_to_pixel(
    pts_world: np.ndarray, K: np.ndarray, world_t_cam: np.ndarray, z_min: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Project (N, 3) world points into pixel coords using the same convention
    as ``unproject_depth`` (camera frame: +x right, +y down, +z forward).

    Returns ``(uv, valid)`` where ``uv`` is (N, 2) float32 pixel coords and
    ``valid`` is a boolean mask of points in front of the camera.
    """
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


def report_pipeline_latencies(time_group: dict, hdf5_path: str) -> None:
    """Save per-frame latency curves from a rollout's ``time`` group.

    Writes ``<dirname(hdf5_path)>/debug/latency.png`` with four overlaid curves
    (build_observation, inference, send, end_to_end) in milliseconds vs frame.
    Frame 0 is discarded because it captures cold-start costs (CUDA init, first
    model upload, autotune) that don't reflect steady-state latency. Frames
    where either side of a delta is 0 are masked to NaN (gaps in plot). The
    mean of each curve is appended to its legend label. No-op if the expected
    keys are absent (e.g. a teleop episode).
    """
    keys = (
        "begin_build_observation",
        "begin_inference",
        "finish_inference",
        "publish_command",
    )
    if not all(k in time_group for k in keys):
        return
    stamps = {k: np.asarray(time_group[k], dtype=np.int64) for k in keys}

    bbo = stamps["begin_build_observation"]
    bi = stamps["begin_inference"]
    fi = stamps["finish_inference"]
    pc = stamps["publish_command"]
    N = bbo.shape[0]
    if N < 2:
        return

    def delta_ms(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mask = (a > 0) & (b > 0)
        out = np.full(a.shape[0], np.nan, dtype=np.float64)
        out[mask] = (b[mask].astype(np.float64) - a[mask].astype(np.float64)) * 1e-6
        return out

    curves = {
        "build_observation (bbo → bi)": delta_ms(bbo, bi),
        "inference         (bi  → fi)": delta_ms(bi, fi),
        "send              (fi  → pc)": delta_ms(fi, pc),
        "end_to_end        (bbo → pc)": delta_ms(bbo, pc),
    }

    out_dir = Path(hdf5_path).parent / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "latency.png"

    # Drop the first frame (cold-start spike) before plotting / mean.
    frames = np.arange(1, N)
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, y in curves.items():
        y_plot = y[1:]
        m = float(np.nanmean(y_plot)) if np.isfinite(y_plot).any() else float("nan")
        ax.plot(frames, y_plot, label=f"{label}  mean={m:.2f} ms", linewidth=1.4)
    ax.set_xlabel("frame")
    ax.set_ylabel("latency (ms)")
    ax.set_title(f"Pipeline latencies — {Path(hdf5_path).name} (frame 0 dropped)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote latency report to {out_path}")


def eef9_to_pos_R(eef9: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an ``(N, 9)`` eef stream into positions and rotations.

    Non-finite rows yield NaN pose so the renderer can skip them.
    """
    if eef9.ndim != 2 or eef9.shape[1] != 9:
        raise ValueError(f"expected (N, 9) eef_9d stream, got shape {eef9.shape}")
    N = eef9.shape[0]
    pos = eef9[:, :3].astype(np.float64)
    Rm = np.full((N, 3, 3), np.nan, dtype=np.float64)
    for i in range(N):
        if np.all(np.isfinite(eef9[i])):
            Rm[i] = gram_schmidt_6d_to_R(eef9[i, 3:9])
    return pos, Rm


def lift_base_pose_se3(pose_xyyaw: np.ndarray) -> np.ndarray:
    """Lift planar base poses ``(N, 3) = (x, y, yaw)`` to ``(N, 4, 4)`` ``world_t_base``.

    The mobile base is planar (z = 0); yaw is a right-handed rotation about world +z,
    matching the body convention of the recorded ``obs/joint/chassis_*`` twist
    (+x forward, +y left). Camera height enters later via ``base_t_cam`` FK, not here.
    """
    pose = np.asarray(pose_xyyaw, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 3:
        raise ValueError(f"expected (N, 3) base pose (x, y, yaw), got shape {pose.shape}")
    n = pose.shape[0]
    T = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    c, s = np.cos(pose[:, 2]), np.sin(pose[:, 2])
    T[:, 0, 0], T[:, 0, 1] = c, -s
    T[:, 1, 0], T[:, 1, 1] = s, c
    T[:, 0, 3] = pose[:, 0]
    T[:, 1, 3] = pose[:, 1]
    return T


def load_world_t_base(obs_group: dict, num_frames: int) -> np.ndarray:
    """Per-frame ``world_t_base`` ``(N, 4, 4)`` from ``obs/base/pose``; identity if absent.

    Episodes recorded with a moving base (``scripts/wbc_vr_robot.py --enable base``) carry
    ``obs/base/pose`` ``(N, 3) = (x, y, yaw)`` in the engage-origin world frame. Without it
    (deploy rollouts, legacy teleop) the base is treated as fixed at the world origin
    (identity) -> base-relative rendering, i.e. unchanged behavior.
    """
    base_group = obs_group.get("base") if isinstance(obs_group, dict) else None
    if not isinstance(base_group, dict) or "pose" not in base_group:
        return np.tile(np.eye(4, dtype=np.float64), (num_frames, 1, 1))
    pose = np.asarray(base_group["pose"], dtype=np.float64)
    if pose.shape != (num_frames, 3):
        raise ValueError(
            f"obs/base/pose shape {pose.shape} != ({num_frames}, 3) = (x, y, yaw)"
        )
    return lift_base_pose_se3(pose)


def transform_points_se3(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply per-frame SE(3) ``T`` ``(N, 4, 4)`` to points ``pts`` ``(N, 3)``.

    Non-finite point rows propagate to NaN (the renderer skips them).
    """
    pts = np.asarray(pts, dtype=np.float64)
    return np.einsum("nij,nj->ni", T[:, :3, :3], pts) + T[:, :3, 3]


def transform_rots_se3(T: np.ndarray, Rm: np.ndarray) -> np.ndarray:
    """Apply the per-frame rotation of ``T`` ``(N, 4, 4)`` to rotations ``Rm`` ``(N, 3, 3)``."""
    Rm = np.asarray(Rm, dtype=np.float64)
    return np.einsum("nij,njk->nik", T[:, :3, :3], Rm)


def load_deploy_position_condition(
    obs_group: dict, num_frames: int
) -> dict[str, np.ndarray] | None:
    """Read optional deploy position-condition arrays from an episode obs group."""
    if "position_condition" not in obs_group:
        return None
    group = obs_group["position_condition"]
    missing = [key for key in ("stage", "mask", "condition_6d") if key not in group]
    if missing:
        raise KeyError(
            f"obs/position_condition missing {missing}; have {list(group)}"
        )

    stage = np.asarray(group["stage"], dtype=np.int32).reshape(-1)
    mask = np.asarray(group["mask"], dtype=np.float32)
    condition_6d = np.asarray(group["condition_6d"], dtype=np.float32)
    if stage.shape != (num_frames,):
        raise ValueError(
            f"obs/position_condition/stage shape {stage.shape} != ({num_frames},)"
        )
    if mask.ndim != 2 or mask.shape[0] != num_frames:
        raise ValueError(
            "obs/position_condition/mask must have shape (N, M), "
            f"got {mask.shape} for N={num_frames}"
        )
    if condition_6d.shape != (num_frames, 6):
        raise ValueError(
            "obs/position_condition/condition_6d must have shape (N, 6), "
            f"got {condition_6d.shape}"
        )
    return {"stage": stage, "mask": mask, "condition_6d": condition_6d}


def format_position_condition_frame(
    stage: np.integer | int, mask: np.ndarray, condition_6d: np.ndarray
) -> str:
    """Format one frame's selected position condition for a rerun text view."""
    cond = np.asarray(condition_6d, dtype=np.float32).reshape(6)
    mask_arr = np.asarray(mask, dtype=np.float32).reshape(-1)
    before = ", ".join(f"{v:.3f}" for v in cond[:3])
    after = ", ".join(f"{v:.3f}" for v in cond[3:])
    mask_text = ", ".join(f"{v:.3f}" for v in mask_arr)
    return (
        f"stage: {int(stage)}\n"
        f"mask: [{mask_text}]\n"
        f"before: [{before}]\n"
        f"after: [{after}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdf5",
        type=str,
        default="/home/yixuan/Dexmate/data/raw_data/episode_7.hdf5",
        help="Path to episode HDF5 file",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Read a policy-rollout episode (follower/policy_rollout.py layout) "
        "instead of a teleop recording: EEF markers come from "
        "obs/action eef_9d/{left,right}, torso/head joints are held at INIT_*, "
        "saved obs/images/{intrinsic,extrinsic} are required (no hardcoded/FK "
        "fallback), and a debug/latency.png is written.",
    )
    parser.add_argument(
        "--voxel", type=float, default=0.012, help="Voxel size for PCD downsample (m)"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="If set, stream to this .rrd file headlessly instead of spawning a "
        "viewer (useful over SSH: generate on the server, open later with "
        "`rerun FILE.rrd`). Default: spawn the viewer (single-script workflow).",
    )
    args = parser.parse_args()

    data, _ = load_dict_from_hdf5(args.hdf5)

    if args.deploy and "time" in data:
        report_pipeline_latencies(data["time"], args.hdf5)

    # ── load image arrays (shared key names across both layouts) ─────────────
    # The unified key names match leader/vr_reader.py; rollout files recorded
    # before policy_rollout.py was renamed used {left_rgb, depth, wrist_rgb},
    # so fall back to those legacy names to keep viewing old episodes.
    img_group = data["obs"]["images"]

    def _img_key(primary: str, legacy: str) -> str | None:
        if primary in img_group:
            return primary
        if legacy in img_group:
            return legacy
        return None

    head_key = _img_key("head_left_rgb", "left_rgb")
    depth_key = _img_key("head_depth", "depth")
    wrist_key = _img_key("left_wrist_rgb", "wrist_rgb")
    if head_key is None or depth_key is None:
        raise KeyError(
            f"obs/images missing head RGB/depth (have: {list(img_group)}); "
            "expected head_left_rgb/head_depth (or legacy left_rgb/depth)"
        )
    head_rgb = np.array(img_group[head_key]).astype(np.uint8)  # (N,H,W,3)
    depth_mm = np.array(img_group[depth_key])  # (N,H,W) uint16, millimetres
    wrist_rgb = (
        np.array(img_group[wrist_key]).astype(np.uint8) if wrist_key is not None else None
    )

    N = head_rgb.shape[0]
    H, W = head_rgb.shape[1], head_rgb.shape[2]

    # ── arm joints (both layouts record obs left/right arm) ──────────────────
    obs_left_arm = np.array(data["obs"]["joint"]["left_arm"])  # (N,7)
    obs_right_arm = np.array(data["obs"]["joint"]["right_arm"])  # (N,7)

    # Torso/head: teleop records them; rollouts hold them at INIT_* (used here
    # for robot-mesh + camera-extrinsic FK only).
    if args.deploy:
        obs_torso = np.tile(np.asarray(INIT_TORSO_JOINTS, dtype=np.float64), (N, 1))
        obs_head = np.tile(np.asarray(INIT_HEAD_JOINTS, dtype=np.float64), (N, 1))
    else:
        obs_torso = np.array(data["obs"]["joint"]["torso"])  # (N,3)
        obs_head = np.array(data["obs"]["joint"]["head"])  # (N,3)

    # ── scalar streams (gripper may be NaN) ──────────────────────────────────
    obs_grip_left = np.array(data["obs"]["gripper"]["left"])  # (N,)
    obs_grip_right = np.array(data["obs"]["gripper"]["right"])  # (N,)
    act_grip_left = np.array(data["action"]["gripper"]["left"])  # (N,)
    act_grip_right = np.array(data["action"]["gripper"]["right"])  # (N,)
    position_condition = (
        load_deploy_position_condition(data["obs"], N) if args.deploy else None
    )

    wrist_note = f"  |  wrist: {wrist_rgb.shape[1:3]}" if wrist_rgb is not None else "  |  no wrist"
    print(
        f"Episode ({'deploy' if args.deploy else 'teleop'}): {N} frames  |  "
        f"head image: {H}x{W}{wrist_note}"
    )
    if position_condition is not None:
        print("Position condition: plotting obs/position_condition stage/mask/condition_6d")

    # ── intrinsics: deploy requires the saved matrix (no fallback); teleop
    #    falls back to hardcoded ZED when absent. ──────────────────────────────
    if "intrinsic" in img_group:
        K_all = np.array(img_group["intrinsic"], dtype=np.float64)
        if K_all.shape == (3, 3):  # single shared matrix → broadcast over frames
            K_all = np.repeat(K_all[None], N, axis=0)
        if K_all.shape != (N, 3, 3):
            raise ValueError(
                f"obs/images/intrinsic shape {K_all.shape} is not (3, 3) or ({N}, 3, 3)"
            )
        print(f"Using saved obs/images/intrinsic {K_all.shape}")
    elif args.deploy:
        raise KeyError(
            f"obs/images/intrinsic missing in --deploy episode {args.hdf5!r}; "
            "refusing to fall back to hardcoded ZED intrinsics "
            f"(obs/images has: {list(img_group)})"
        )
    else:
        fx = fy = 770.1868 / 2.0
        cx = 990.2711 / 2.0
        cy = 637.7721 / 2.0
        K_single = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        K_all = np.repeat(K_single[None], N, axis=0)

    # ── kinematics helper (camera extrinsic fallback + EEF via FK) ───────────
    kin = KinHelper("vega_no_effector")
    joint_names_kin = [j.name for j in kin.sapien_robot.get_active_joints()]
    name_to_idx_kin = {n: i for i, n in enumerate(joint_names_kin)}
    cam_link_idx = kin.link_name_to_idx["zed_depth_frame"]
    left_eef_idx = kin.link_name_to_idx["L_ee"]
    right_eef_idx = kin.link_name_to_idx["R_ee"]

    def build_qpos(torso_v, left_v, right_v, head_v) -> np.ndarray:
        """Map per-arm/torso/head joint values onto the sapien active-joint vector."""
        qpos = np.zeros(kin.sapien_robot.dof)
        joint_vals = list(torso_v) + list(left_v) + list(right_v) + list(head_v)
        for name, val in zip(_OBS_NAMES, joint_vals, strict=True):
            if name in name_to_idx_kin:
                qpos[name_to_idx_kin[name]] = val
        return qpos

    def fk_eef_series(torso, left, right, head, link_idxs) -> tuple[np.ndarray, np.ndarray]:
        """Per-frame FK of ``link_idxs`` → positions (L,N,3) and rotations (L,N,3,3)."""
        pos = np.full((len(link_idxs), N, 3), np.nan, dtype=np.float64)
        Rm = np.full((len(link_idxs), N, 3, 3), np.nan, dtype=np.float64)
        for idx in range(N):
            mats = kin.compute_fk_from_link_idx(
                build_qpos(torso[idx], left[idx], right[idx], head[idx]), link_idxs
            )
            for k, M in enumerate(mats):
                if np.all(np.isfinite(M)):
                    pos[k, idx] = M[:3, 3]
                    Rm[k, idx] = M[:3, :3]
        return pos, Rm

    # ── extrinsics: deploy requires the saved matrix (no FK fallback); teleop
    #    falls back to per-frame FK from the (obs / INIT) joints when absent. ──
    if "extrinsic" in img_group:
        extrinsics = np.array(img_group["extrinsic"], dtype=np.float64)
        if extrinsics.shape == (4, 4):  # single shared pose → broadcast over frames
            extrinsics = np.repeat(extrinsics[None], N, axis=0)
        if extrinsics.shape != (N, 4, 4):
            raise ValueError(
                f"obs/images/extrinsic shape {extrinsics.shape} is not (4, 4) or ({N}, 4, 4)"
            )
        print(f"Using saved obs/images/extrinsic {extrinsics.shape}")
    elif args.deploy:
        raise KeyError(
            f"obs/images/extrinsic missing in --deploy episode {args.hdf5!r}; "
            "refusing to fall back to FK camera extrinsics "
            f"(obs/images has: {list(img_group)})"
        )
    else:
        print("No saved extrinsic — computing per-frame FK ...")
        extrinsics = np.zeros((N, 4, 4), dtype=np.float64)
        for idx in range(N):
            extrinsics[idx] = kin.compute_fk_from_link_idx(
                build_qpos(obs_torso[idx], obs_left_arm[idx], obs_right_arm[idx], obs_head[idx]),
                [cam_link_idx],
            )[0]

    # ── world frame: compose obs/base/pose so the cloud, robot, and EEF markers
    #    render in a fixed WORLD frame as the base drives. The extrinsic above is
    #    base-RELATIVE (base_t_cam, saved or FK); world_t_cam = world_t_base · base_t_cam.
    #    No obs/base/pose (deploy rollouts, legacy teleop) -> identity, i.e. unchanged
    #    base-relative rendering. ────────────────────────────────────────────────
    world_t_base = load_world_t_base(data["obs"], N)
    extrinsics = world_t_base @ extrinsics  # base_t_cam -> world_t_cam

    # ── EEF obs (blue) / action (red) markers ────────────────────────────────
    # markers: list of (entity_path, color, pos (N,3), R (N,3,3)). The future
    # trajectory tail is logged per frame as Points3D(pos[idx:]); the current
    # pose is a Transform3D under "{path}/frame".
    print("Building EEF obs/action marker streams ...")
    markers: list[tuple[str, tuple[int, int, int], np.ndarray, np.ndarray]] = []
    if args.deploy:
        # Both arms, read directly from the recorded eef_9d streams.
        for side in ("left", "right"):
            obs_pos, obs_R = eef9_to_pos_R(
                np.array(data["obs"]["eef_9d"][side], dtype=np.float64)
            )
            act_pos, act_R = eef9_to_pos_R(
                np.array(data["action"]["eef_9d"][side], dtype=np.float64)
            )
            markers.append((f"world/eef_obs/{side}", _OBS_COLOR, obs_pos, obs_R))
            markers.append((f"world/eef_action/{side}", _ACTION_COLOR, act_pos, act_R))
    else:
        # Both arms via FK. obs uses obs joints; action uses action arm joints
        # with torso/head from obs (action streams don't carry torso/head).
        act_left_arm = np.array(data["action"]["joint"]["left_arm"])  # (N,7)
        act_right_arm = np.array(data["action"]["joint"]["right_arm"])  # (N,7)
        obs_pos, obs_R = fk_eef_series(
            obs_torso, obs_left_arm, obs_right_arm, obs_head, [left_eef_idx, right_eef_idx]
        )
        act_pos, act_R = fk_eef_series(
            obs_torso, act_left_arm, act_right_arm, obs_head, [left_eef_idx, right_eef_idx]
        )
        markers.append(("world/eef_obs/left", _OBS_COLOR, obs_pos[0], obs_R[0]))
        markers.append(("world/eef_obs/right", _OBS_COLOR, obs_pos[1], obs_R[1]))
        markers.append(("world/eef_action/left", _ACTION_COLOR, act_pos[0], act_R[0]))
        markers.append(("world/eef_action/right", _ACTION_COLOR, act_pos[1], act_R[1]))

    # Lift EEF markers (built base-relative for teleop / in the recorded eef frame for
    # deploy) into the world frame with the same per-frame world_t_base (identity when
    # obs/base/pose is absent, so deploy/legacy episodes are unchanged).
    markers = [
        (
            path,
            color,
            transform_points_se3(world_t_base, pos),
            transform_rots_se3(world_t_base, Rm),
        )
        for path, color, pos, Rm in markers
    ]

    robot_mesh_gen = RobotMeshGenerator("vega_no_effector")

    # ── rerun ────────────────────────────────────────────────────────────────
    rr.init("vis_episode", spawn=args.save is None)
    if args.save is not None:
        rr.save(args.save)
        print(f"Streaming to {args.save} (headless; no viewer spawned)")

    # ── default layout: wrist RGB (if present), world PCD, head RGB, grippers ─
    # Entity paths below mirror where the streams are logged further down; an
    # explicit blueprint also stops the viewer falling back to a stale auto
    # layout that reported "Entity not found in view" for the wrist image.
    # In --deploy mode the depth stream clutters the views (a translucent layer
    # over head_left_rgb and a backprojected point cloud competing with the
    # colored /world/pcd in 3D), so hide /world/camera/depth by default in both.
    # It stays in the blueprint and can be toggled back on from the viewer.
    depth_override = (
        {"/world/camera/depth": rrb.EntityBehavior(visible=False)} if args.deploy else None
    )

    side_panels: list = []
    if wrist_rgb is not None:
        side_panels.append(
            rrb.Spatial2DView(origin="/image/left_wrist_rgb", name="left_wrist_rgb")
        )
    # Head RGB 2D view must root at the pinhole entity (/world/camera), not the
    # Image child (/world/camera/rgb), so Points2D overlays logged as siblings
    # of rgb (e.g. position_condition_2d) appear on top of the image.
    side_panels.append(
        rrb.Spatial2DView(
            origin="/world/camera", name="head_left_rgb", overrides=depth_override
        )
    )
    side_panels.append(
        rrb.TimeSeriesView(
            name="left gripper (obs + action)",
            contents=[
                "/plot/gripper/obs_left",
                "/plot/gripper/act_left",
            ],
        )
    )
    side_panels.append(
        rrb.TimeSeriesView(
            name="right gripper (obs + action)",
            contents=[
                "/plot/gripper/obs_right",
                "/plot/gripper/act_right",
            ],
        )
    )
    if position_condition is not None:
        side_panels.append(
            rrb.TextLogView(
                origin="/text/position_condition", name="position_condition"
            )
        )
        side_panels.append(
            rrb.TimeSeriesView(
                name="position condition stage",
                contents=["/plot/position_condition/stage"],
                visible=False,
            )
        )
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world", name="world pointcloud", overrides=depth_override
            ),
            rrb.Vertical(*side_panels),
            column_shares=[2, 1],
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint, make_active=True, make_default=True)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # ── distinct colors per gripper series (otherwise rerun auto-picks green for both)
    rr.log(
        "plot/gripper/obs_left",
        rr.SeriesLines(colors=[230, 25, 75], names="obs_left"),
        static=True,
    )
    rr.log(
        "plot/gripper/obs_right",
        rr.SeriesLines(colors=[0, 130, 200], names="obs_right"),
        static=True,
    )
    rr.log(
        "plot/gripper/act_left",
        rr.SeriesLines(colors=[245, 130, 48], names="act_left"),
        static=True,
    )
    rr.log(
        "plot/gripper/act_right",
        rr.SeriesLines(colors=[60, 180, 75], names="act_right"),
        static=True,
    )
    if position_condition is not None:
        rr.log(
            "plot/position_condition/stage",
            rr.SeriesLines(colors=[255, 225, 25], names="stage"),
            static=True,
        )

    # ── robot link geometry: log ONCE as static, then animate per-frame via
    #    Transform3D in the loop below. Re-logging full Mesh3D every frame costs
    #    ~42 MB/frame here (39 links, 1.85M verts ≈ 28 GB over a 664-frame
    #    episode); static geometry collapses that to a one-time cost. Pixel
    #    identical — verified 0 m vertex diff vs compute_robot_meshes.
    robot_link_names = list(robot_mesh_gen.meshes.keys())
    for name in robot_link_names:
        m = robot_mesh_gen.meshes[name]
        rr.log(
            f"world/robot/{name}",
            rr.Mesh3D(
                vertex_positions=np.asarray(m.vertices, dtype=np.float32),
                triangle_indices=np.asarray(m.faces, dtype=np.uint32),
            ),
            static=True,
        )

    for idx in range(N):
        rr.set_time("frame", sequence=idx)

        world_t_cam = extrinsics[idx]
        K = K_all[idx]
        K32 = K.astype(np.float32)

        # ── camera: pose + intrinsics + 2D streams attached in image space ──
        rr.log(
            "world/camera",
            rr.Transform3D(translation=world_t_cam[:3, 3], mat3x3=world_t_cam[:3, :3]),
        )
        rr.log("world/camera", rr.Pinhole(image_from_camera=K32, width=W, height=H))
        rr.log("world/camera/rgb", rr.Image(head_rgb[idx]))
        rr.log("world/camera/depth", rr.DepthImage(depth_mm[idx], meter=1000.0))
        # Wrist RGB is monocular only — log under /image so it shows in 2D views.
        if wrist_rgb is not None:
            rr.log("image/left_wrist_rgb", rr.Image(wrist_rgb[idx]))

        # ── colored point cloud (manual unproject + voxel downsample) ───────
        depth_m = depth_mm[idx] / 1000.0
        pts, mask = unproject_depth(depth_m, K, world_t_cam)
        cols = head_rgb[idx][mask]
        pts, cols = voxel_downsample(pts, cols, args.voxel)
        rr.log("world/pcd", rr.Points3D(pts, colors=cols, radii=0.003))

        # ── robot pose: per-frame transforms only (geometry is static above) ──
        joints_vals = (
            [0.0] * len(_WHEEL_NAMES)
            + obs_torso[idx].tolist()
            + obs_left_arm[idx].tolist()
            + obs_right_arm[idx].tolist()
            + obs_head[idx].tolist()
        )
        joint_names = _WHEEL_NAMES + _OBS_NAMES
        qpos = robot_mesh_gen.convert_to_sapien_joint_order(
            np.array(joints_vals), joint_names
        )
        link_tf = robot_mesh_gen.compute_fk_from_link_names(
            qpos, robot_link_names, in_obj_frame=True
        )
        wtb = world_t_base[idx]  # base-relative link poses -> world frame (identity if no base)
        for name in robot_link_names:
            tf = wtb @ link_tf[name]
            rr.log(
                f"world/robot/{name}",
                rr.Transform3D(translation=tf[:3, 3], mat3x3=tf[:3, :3]),
            )

        # ── EEF obs (blue) / action (red): future-trajectory tail + frame ───
        for path, color, pos, Rm in markers:
            tail = pos[idx:]
            finite = np.all(np.isfinite(tail), axis=1)
            rr.log(f"{path}/marker", rr.Points3D(tail[finite], colors=[color], radii=0.006))
            if np.all(np.isfinite(pos[idx])) and np.all(np.isfinite(Rm[idx])):
                rr.log(f"{path}/frame", rr.Transform3D(translation=pos[idx], mat3x3=Rm[idx]))
            else:
                rr.log(f"{path}/frame", rr.Clear(recursive=True))

        # ── scalar plots: gripper (obs vs action) ───────────────────────────
        rr.log("plot/gripper/obs_left", rr.Scalars(float(obs_grip_left[idx])))
        rr.log("plot/gripper/obs_right", rr.Scalars(float(obs_grip_right[idx])))
        rr.log("plot/gripper/act_left", rr.Scalars(float(act_grip_left[idx])))
        rr.log("plot/gripper/act_right", rr.Scalars(float(act_grip_right[idx])))

        # ── optional deploy position condition: selected [before, after] ─────
        if position_condition is not None:
            pc_stage = position_condition["stage"][idx]
            pc_mask = position_condition["mask"][idx]
            pc6 = position_condition["condition_6d"][idx]
            before = pc6[:3].astype(np.float32)
            after = pc6[3:].astype(np.float32)
            if np.all(np.isfinite(pc6)):
                rr.log(
                    "world/position_condition/before",
                    rr.Points3D(
                        before[None], colors=[_POS_COND_COLOR_BEFORE], radii=_POS_COND_RADIUS_3D
                    ),
                )
                rr.log(
                    "world/position_condition/after",
                    rr.Points3D(
                        after[None], colors=[_POS_COND_COLOR_AFTER], radii=_POS_COND_RADIUS_3D
                    ),
                )
                rr.log(
                    "world/position_condition/delta",
                    rr.LineStrips3D(
                        [np.stack([before, after])], colors=[_POS_COND_COLOR_AFTER]
                    ),
                )

                condition_uv, condition_valid = project_world_to_pixel(
                    np.stack([before, after]), K, world_t_cam
                )
                valid_idx = np.flatnonzero(condition_valid)
                if len(valid_idx) > 0:
                    condition_colors = np.asarray(
                        [_POS_COND_COLOR_BEFORE, _POS_COND_COLOR_AFTER], dtype=np.uint8
                    )[valid_idx]
                    rr.log(
                        "world/camera/position_condition_2d",
                        rr.Points2D(
                            condition_uv[valid_idx],
                            colors=condition_colors,
                            radii=np.full(len(valid_idx), _POS_COND_RADIUS_2D, dtype=np.float32),
                        ),
                    )
                else:
                    rr.log("world/camera/position_condition_2d", rr.Clear(recursive=False))
                if np.all(condition_valid):
                    rr.log(
                        "world/camera/position_condition_vector_2d",
                        rr.LineStrips2D(
                            [condition_uv.astype(np.float32)],
                            colors=[_POS_COND_COLOR_AFTER],
                            radii=1.5,
                        ),
                    )
                else:
                    rr.log(
                        "world/camera/position_condition_vector_2d",
                        rr.Clear(recursive=False),
                    )
            else:
                rr.log("world/position_condition", rr.Clear(recursive=True))
                rr.log("world/camera/position_condition_2d", rr.Clear(recursive=False))
                rr.log(
                    "world/camera/position_condition_vector_2d", rr.Clear(recursive=False)
                )
            rr.log("plot/position_condition/stage", rr.Scalars(float(pc_stage)))
            rr.log(
                "text/position_condition",
                rr.TextLog(format_position_condition_frame(pc_stage, pc_mask, pc6)),
            )


if __name__ == "__main__":
    main()
