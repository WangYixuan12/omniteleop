"""Visualise a recorded VR teleoperation episode (or a policy rollout) in rerun.

Logs RGB-D, the colored point cloud, robot meshes, and EEF obs/action markers
under a single ``frame`` timeline so the rerun viewer's scrubber doubles as
prev/next/jump navigation.

Two episode layouts are supported via ``--deploy``:

* default (teleop / ``raw_data`` layout, written by ``leader/vr_reader.py``)::

      obs/images/{head_left_rgb, head_depth, left_wrist_rgb, intrinsic, extrinsic}
      obs/joint/{torso, left_arm, right_arm, head}
      obs/gripper/{left, right}
      action/joint/{left_arm, right_arm, head, chassis_*}
      action/gripper/{left, right}
      timestamp_ns  (legacy episodes may also carry a top-level ``stage``)

  EEF obs/action markers come from forward kinematics on the *both-arm* joint
  streams (``L_ee`` / ``R_ee``): obs = ``obs/joint/*_arm``, action =
  ``action/joint/*_arm`` (with torso/head taken from the obs streams).

* ``--deploy`` (policy rollout, written by ``follower/policy_rollout.py``)::

      time/{begin_build_observation, begin_inference, finish_inference, publish_command, ...}
      obs/images/{head_left_rgb, head_depth, left_wrist_rgb}   (no intrinsic/extrinsic)
      obs/joint/{left_arm, right_arm}
      obs/eef_9d/{left, right}
      obs/gripper/{left, right}
      action/eef_9d/{left, right}
      action/gripper/{left, right}

  EEF markers are read directly for both arms: obs = ``obs/eef_9d/{left,right}``
  (blue), action = ``action/eef_9d/{left,right}`` (red).
  Torso/head joints aren't recorded, so they're held at ``INIT_*``
  for robot-mesh / camera-extrinsic FK, and ZED intrinsics fall back to the
  hardcoded values. A ``debug/latency.png`` is written from the ``time`` group.

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
    """Split an ``(N, 9)`` eef stream into positions ``(N, 3)`` and rotations
    ``(N, 3, 3)``; non-finite rows yield NaN pose so the renderer can skip them."""
    if eef9.ndim != 2 or eef9.shape[1] != 9:
        raise ValueError(f"expected (N, 9) eef_9d stream, got shape {eef9.shape}")
    N = eef9.shape[0]
    pos = eef9[:, :3].astype(np.float64)
    Rm = np.full((N, 3, 3), np.nan, dtype=np.float64)
    for i in range(N):
        if np.all(np.isfinite(eef9[i])):
            Rm[i] = gram_schmidt_6d_to_R(eef9[i, 3:9])
    return pos, Rm


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
        "ZED intrinsics fall back to hardcoded, and a debug/latency.png is written.",
    )
    parser.add_argument(
        "--voxel", type=float, default=0.008, help="Voxel size for PCD downsample (m)"
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
    # "stage" was dropped from recordings; keep showing it for legacy episodes.
    stage = np.array(data["stage"]) if "stage" in data else None  # (N,) or None

    wrist_note = f"  |  wrist: {wrist_rgb.shape[1:3]}" if wrist_rgb is not None else "  |  no wrist"
    print(
        f"Episode ({'deploy' if args.deploy else 'teleop'}): {N} frames  |  "
        f"head image: {H}x{W}{wrist_note}"
    )

    # ── intrinsics: prefer per-episode if recorded, else hardcoded ZED ───────
    if "intrinsic" in img_group:
        K_all = np.array(img_group["intrinsic"]).astype(np.float64)
        if K_all.ndim == 2:  # single shared matrix → broadcast over frames
            K_all = np.repeat(K_all[None], N, axis=0)
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

    # ── extrinsics: prefer saved, else FK from the (obs / INIT) joints ───────
    if "extrinsic" in img_group:
        extrinsics = np.array(img_group["extrinsic"]).astype(np.float64)
        print("Using saved obs/images/extrinsic")
    else:
        print("No saved extrinsic — computing per-frame FK ...")
        extrinsics = np.zeros((N, 4, 4), dtype=np.float64)
        for idx in range(N):
            extrinsics[idx] = kin.compute_fk_from_link_idx(
                build_qpos(obs_torso[idx], obs_left_arm[idx], obs_right_arm[idx], obs_head[idx]),
                [cam_link_idx],
            )[0]

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
    side_panels: list = []
    if wrist_rgb is not None:
        side_panels.append(
            rrb.Spatial2DView(origin="/image/left_wrist_rgb", name="left_wrist_rgb")
        )
    side_panels.append(rrb.Spatial2DView(origin="/world/camera/rgb", name="head_left_rgb"))
    side_panels.append(
        rrb.TimeSeriesView(
            name="gripper (obs + action)",
            contents=[
                "/plot/gripper/obs_left",
                "/plot/gripper/obs_right",
                "/plot/gripper/act_left",
                "/plot/gripper/act_right",
            ],
        )
    )
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world", name="world pointcloud"),
            rrb.Vertical(*side_panels),
            column_shares=[2, 1],
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint, make_active=True, make_default=True)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # ── distinct colors per gripper series (otherwise rerun auto-picks green for both)
    rr.log("plot/gripper/obs_left", rr.SeriesLines(colors=[230, 25, 75], names="obs_left"), static=True)
    rr.log("plot/gripper/obs_right", rr.SeriesLines(colors=[0, 130, 200], names="obs_right"), static=True)
    rr.log("plot/gripper/act_left", rr.SeriesLines(colors=[245, 130, 48], names="act_left"), static=True)
    rr.log("plot/gripper/act_right", rr.SeriesLines(colors=[60, 180, 75], names="act_right"), static=True)

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
        for name in robot_link_names:
            tf = link_tf[name]
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


if __name__ == "__main__":
    main()
