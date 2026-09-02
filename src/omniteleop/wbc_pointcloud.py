"""World-frame head point clouds for the 3D (ManiFlow) WBC policy.

Turns one head RGB-D frame into the ``(num_points, 6)`` XYZ+RGB cloud the policy
consumes. Lives here rather than in the converter script because the LIVE rollout must
reproduce this preprocessing BIT-FOR-BIT -- a train/deploy mismatch in the frame, the
crop, or the depth scale silently poisons the policy (same reason ``wbc_policy_format``
is shared by the porter and the rollout).

Frame convention: points are expressed in the ENGAGE-ORIGIN WORLD frame, matching the
29-D WBC action targets, ``observation.state[29:32]`` odometry anchor, and SceneDiff
object positions. A 32-D joystick policy deliberately keeps its action targets in the
current-base frame while observing this world cloud plus a base-frame proprioceptive
state; that mixed observation/action frame contract is explicit in the dataset sidecar.
The base MOVES, so the head pose must be composed through odometry
(``world_T_zed = world_T_base @ base_T_zed``). The camera is the ZED
``zed_depth_frame`` (optical: z forward, x right, y down), matching the pinhole
backprojection below.

Verified end-to-end on the raw takes: the SceneDiff object positions -- produced by a
completely independent pipeline -- land exactly on the reconstructed surfaces (box on
the shelf, cloth on the tabletop), and the world EEF targets land on the grasped box.
"""

from __future__ import annotations

import numpy as np

# Recorder stores head depth as uint16 millimetres.
DEPTH_SCALE_M = 1e-3

# Depth validity window. The ZED returns 0 for invalid pixels and garbage past a few
# metres; 3.0 m comfortably covers the room the takes were recorded in.
MIN_DEPTH_M = 0.05
MAX_DEPTH_M = 3.0

# Workspace crop in the WORLD frame, MEASURED from all 41 raw takes. The farthest-point
# sample below spreads a FIXED budget ~uniformly in space, so every cubic metre of room the
# crop keeps is budget stolen from the two small objects. Shrinking the box from the room
# (2.0 x 2.6 x 1.3 m) to the task volume (1.5 x 1.95 x 0.72 m) nearly doubles the points
# that land on them: box 48 -> 83, cloth 40 -> 74 out of 1024.
#
# Three envelopes constrain it; the tightest bound per axis wins, with >= 10 cm margin:
#   objects at rest  x[0.97, 1.54]  y[-0.63, 0.77]  z[0.66, 1.13]  (SceneDiff centroids)
#   box IN HAND      x[0.33, 1.38]  y[-0.80, 0.75]  z[0.83, 1.17]  (bimanual EEF midpoint
#                                                                   while BOTH grippers are
#                                                                   commanded closed)
#   all EEF targets  x[0.17, 1.40]  y[-0.98, 1.19]  z[0.73, 1.21]
#
# The box-in-hand envelope is what forbids a tight x_min: the box is carried BACK toward the
# robot (to x = 0.33) before being placed, so an x_min above ~0.30 crops the grasped box out
# of the observation on up to 42% of the transport frames. z_min is bounded from above by the
# CLOTH (z ~ 0.66-0.70), not by the box -- z_min = 0.70 halves the cloth's points. The volume
# saved comes from the floor, ceiling, far wall and lower shelves. World-frame, hence
# independent of where the base has driven; every frame still keeps >= 13.9k points.
#
# Deliberately NOT done: DBSCAN / radius-outlier removal (the ``pcd_cluster`` step of the
# DemoGen-style pipelines). Measured on these takes it deletes 3-4% of points overall but
# ~10% of the BOX's -- whose far, thin surface is genuinely sparse -- so it LOWERS box
# coverage (37 -> 30 points at 512) while costing ~6 ms/frame. FPS does over-select sparse
# regions (~15% of its picks have <6 neighbours within 3 cm, ~4x their share of the cloud),
# but on real ZED depth those picks are mostly true object edges, not noise.
CROP_MIN = (0.20, -0.90, 0.60)
CROP_MAX = (1.70, 1.05, 1.32)

# XYZ + RGB. ManiFlow slices [..., :3] when use_pc_color=False
# (maniflow_pointcloud_policy.py:184), so colours ride along at zero cost and
# `use_pc_color` becomes a train-time flag rather than a reconversion.
POINT_CHANNELS = 6

# Downsampler. fpsample's bucket ("QuickFPS") variant, measured against an exact CUDA FPS
# on these takes: same object coverage (box 86 vs 83, cloth 77 vs 74 of 1024) and the same
# 0.044 m mean nearest-neighbour spread, at 2.0 ms/frame instead of 16.2 ms/frame -- the
# live rollout samples one frame at a time, where the exact CUDA implementation is
# kernel-launch bound and cannot amortise over a batch.
#
# `start_idx` MUST stay pinned: fpsample's default picks a RANDOM seed point, which would
# make the training data irreproducible AND make the live cloud differ from the one the
# policy trained on -- a silent, unfalsifiable train/deploy skew. Pinned, the output is
# bit-identical across processes, environments and machines (verified).
FPS_SAMPLER = "fpsample.bucket_fps_kdline_sampling"
FPS_KDLINE_H = 5
FPS_START_IDX = 0


def world_t_head_from_state(state: np.ndarray, state_frame: str = "base") -> np.ndarray:
    """``world_T_zed`` (4,4) from a 32-D WBC state vector.

    ``state[20:29]`` is the head pos6d in ``state_frame``; ``state[29:32]`` is the
    odometry base pose (identical in both frames). Mirrors the calib-sidecar derivation
    in ``port_wbc_mobile_hdf5.add_episode``.
    """
    from omniteleop.wbc_policy_format import base_pose_to_mat, pos6d_to_mat  # noqa: PLC0415

    if state_frame not in ("base", "world"):
        raise ValueError(f"state_frame must be 'base' or 'world', got {state_frame!r}")
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (32,):
        raise ValueError(f"state must be (32,), got {state.shape}")
    head_mat = pos6d_to_mat(state[20:29])
    if state_frame == "world":
        return head_mat
    return base_pose_to_mat(state[29:32]) @ head_mat


def unproject_head_frame(
    depth_mm: np.ndarray,
    rgb: np.ndarray,
    intrinsic: np.ndarray,
    world_t_cam: np.ndarray,
    *,
    min_depth: float = MIN_DEPTH_M,
    max_depth: float = MAX_DEPTH_M,
    return_pixel_index: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """One head frame -> ``(M, 6)`` float32 world XYZ + RGB[0,1], invalid depth dropped.

    ``depth_mm`` (H,W) uint16 millimetres, ``rgb`` (H,W,3) uint8, ``intrinsic`` (3,3) at
    the SAME resolution as depth, ``world_t_cam`` (4,4) camera-to-world.

    ``return_pixel_index=True`` additionally returns the flat row-major pixel index
    (M,) int64 of each point's source pixel — the lineage that lets per-pixel labels
    (e.g. tracked object masks) ride the SAME selection as XYZRGB instead of being
    re-projected. The cloud values are identical either way.
    """
    if depth_mm.ndim != 2:
        raise ValueError(f"depth must be (H, W), got {depth_mm.shape}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be (H, W, 3), got {rgb.shape}")
    if rgb.shape[:2] != depth_mm.shape:
        raise ValueError(f"rgb {rgb.shape[:2]} != depth {depth_mm.shape} (must match)")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"intrinsic must be (3, 3), got {intrinsic.shape}")
    if world_t_cam.shape != (4, 4):
        raise ValueError(f"world_t_cam must be (4, 4), got {world_t_cam.shape}")
    if not 0.0 <= min_depth < max_depth:
        raise ValueError(f"need 0 <= min_depth < max_depth, got {min_depth}, {max_depth}")

    k = np.asarray(intrinsic, dtype=np.float64)
    depth_m = depth_mm.astype(np.float32) * DEPTH_SCALE_M
    height, width = depth_m.shape
    v, u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    valid = (depth_m > min_depth) & (depth_m < max_depth)

    z = depth_m[valid].astype(np.float64)
    x = (u[valid] - k[0, 2]) / k[0, 0] * z
    y = (v[valid] - k[1, 2]) / k[1, 1] * z
    cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)
    world = (np.asarray(world_t_cam, dtype=np.float64) @ cam.T).T[:, :3]

    colors = rgb[valid].astype(np.float32) / 255.0
    cloud = np.concatenate([world.astype(np.float32), colors], axis=1)
    if return_pixel_index:
        # depth_m[valid] walks the (H, W) grid row-major, exactly ravel order.
        return cloud, np.flatnonzero(valid.ravel())
    return cloud


def crop_workspace(
    cloud: np.ndarray,
    crop_min: tuple[float, float, float] = CROP_MIN,
    crop_max: tuple[float, float, float] = CROP_MAX,
    *,
    return_mask: bool = False,
    frame_t_world: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Keep only the points strictly inside the workspace box.

    ``return_mask=True`` additionally returns the (M,) keep mask over the INPUT rows,
    so per-point side data can follow the same selection.

    ``frame_t_world`` (4,4) makes the box FOLLOW A MOVING FRAME -- pass ``base_T_world`` to crop
    in the robot's base frame. Points are tested after transforming by it and returned unchanged
    in world coordinates, so only the selection changes. Default None keeps the historical
    fixed-world-box behaviour bit-for-bit.

    A fixed world box cannot describe a long-horizon take at all: on the sim dish2rack episodes
    the two objects span y in [-3.52, +0.33] against a box covering [-0.90, +1.05], and z in
    [+0.22, +0.75] against a box starting at +0.60, so the destination falls entirely outside it.
    Widening the box to the whole route (1.05 x 3.85 m of floor) instead spreads the 1024-point
    budget over mostly-empty room.
    """
    if cloud.ndim != 2 or cloud.shape[1] != POINT_CHANNELS:
        raise ValueError(f"cloud must be (M, {POINT_CHANNELS}), got {cloud.shape}")
    lo = np.asarray(crop_min, dtype=np.float32)
    hi = np.asarray(crop_max, dtype=np.float32)
    if lo.shape != (3,) or hi.shape != (3,):
        raise ValueError(f"crop bounds must be 3-vectors, got {lo.shape}, {hi.shape}")
    if not np.all(lo < hi):
        raise ValueError(f"crop_min {crop_min} must be strictly below crop_max {crop_max}")
    xyz = cloud[:, :3]
    if frame_t_world is not None:
        frame_t_world = np.asarray(frame_t_world, dtype=np.float64)
        if frame_t_world.shape != (4, 4) or not np.all(np.isfinite(frame_t_world)):
            raise ValueError(f"frame_t_world must be a finite (4,4), got {frame_t_world.shape}")
        xyz = (frame_t_world[:3, :3] @ xyz.T.astype(np.float64)).T + frame_t_world[:3, 3]
        xyz = xyz.astype(np.float32)
    keep = np.all((xyz > lo) & (xyz < hi), axis=1)
    if return_mask:
        return cloud[keep], keep
    return cloud[keep]


def build_world_cloud(
    depth_mm: np.ndarray,
    rgb: np.ndarray,
    intrinsic: np.ndarray,
    world_t_cam: np.ndarray,
    *,
    crop_min: tuple[float, float, float] = CROP_MIN,
    crop_max: tuple[float, float, float] = CROP_MAX,
    min_depth: float = MIN_DEPTH_M,
    max_depth: float = MAX_DEPTH_M,
    return_pixel_index: bool = False,
    frame_t_world: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Unproject + crop one head frame into a variable-length ``(M, 6)`` world cloud.

    ``return_pixel_index=True`` additionally returns each kept point's flat row-major
    source-pixel index (M,) int64 (unproject validity and crop composed).
    """
    if not return_pixel_index:
        cloud = unproject_head_frame(depth_mm, rgb, intrinsic, world_t_cam,
                                     min_depth=min_depth, max_depth=max_depth)
        return crop_workspace(cloud, crop_min, crop_max, frame_t_world=frame_t_world)
    cloud, pix_idx = unproject_head_frame(depth_mm, rgb, intrinsic, world_t_cam,
                                          min_depth=min_depth, max_depth=max_depth,
                                          return_pixel_index=True)
    cropped, keep = crop_workspace(cloud, crop_min, crop_max, return_mask=True,
                                   frame_t_world=frame_t_world)
    return cropped, pix_idx[keep]


def sampler_signature() -> dict:
    """The exact downsampler identity, recorded in the zarr's ``.meta.json``.

    The live rollout cross-checks this against its own module. Any drift -- a different
    ``h``, an unpinned ``start_idx``, a different fpsample build -- silently yields a
    different point set at deploy time than the policy trained on, which no downstream
    check would catch.
    """
    import fpsample  # noqa: PLC0415

    return {
        "sampler": FPS_SAMPLER,
        "kdline_h": FPS_KDLINE_H,
        "start_idx": FPS_START_IDX,
        "fpsample_version": getattr(fpsample, "__version__", "unknown"),
    }


def farthest_point_indices(points_xyz: np.ndarray, num_points: int) -> np.ndarray:
    """Farthest-point sampling: ``(N, 3)`` -> ``(num_points,)`` indices into it.

    ``start_idx`` is pinned: fpsample's default is a RANDOM seed point, which would make
    the training data non-reproducible and, worse, make the live cloud differ from the one
    the policy trained on. With it pinned the result is bit-identical across processes,
    environments and machines (verified).
    """
    import fpsample  # noqa: PLC0415 -- native ext; only the converter/rollout need it

    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points_xyz.shape}")
    if not 0 < num_points <= points_xyz.shape[0]:
        raise ValueError(
            f"need 0 < num_points <= N, got num_points={num_points}, N={points_xyz.shape[0]}")
    # The kd-line bucketing needs at least one point per leaf of a height-h tree; fpsample
    # otherwise fails on a bare `assert`.
    if points_xyz.shape[0] < 2 ** FPS_KDLINE_H:
        raise ValueError(
            f"need N >= 2**{FPS_KDLINE_H} = {2 ** FPS_KDLINE_H} points for "
            f"{FPS_SAMPLER} (h={FPS_KDLINE_H}), got N={points_xyz.shape[0]}")
    return fpsample.bucket_fps_kdline_sampling(
        np.ascontiguousarray(points_xyz), num_points,
        h=FPS_KDLINE_H, start_idx=FPS_START_IDX)


def resample_clouds(
    clouds: list[np.ndarray],
    num_points: int,
    *,
    pool_size: int,
    rng: np.random.Generator,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """``[(M_i, 6)] -> (B, num_points, 6)`` float32 via uniform pooling + FPS.

    Each cloud is first uniformly subsampled to a common ``pool_size`` (bounding the FPS
    cost), then farthest-point-sampled to ``num_points`` so the retained points cover the
    workspace evenly rather than clustering wherever the depth sensor happened to hit
    densely. FPS runs on the XYZ channels; the RGB channels follow the chosen indices.

    ``return_indices=True`` additionally returns ``(B, num_points)`` int64 — each output
    row's index into its INPUT cloud (pool draw and FPS pick composed), so per-point side
    data follows the exact same selection. The cloud values and the rng draw sequence are
    identical either way (one ``rng.choice`` per cloud).

    Raises if any cloud has fewer than ``num_points`` points -- with the measured crop
    every frame keeps >13k, so a short frame means corrupt depth, not a normal case, and
    zero-padding it would inject phantom geometry at the world origin.
    """
    if not clouds:
        raise ValueError("clouds is empty")
    if num_points > pool_size:
        raise ValueError(f"num_points {num_points} > pool_size {pool_size}")
    out = np.empty((len(clouds), num_points, POINT_CHANNELS), dtype=np.float32)
    indices = np.empty((len(clouds), num_points), dtype=np.int64)
    for i, cloud in enumerate(clouds):
        if cloud.ndim != 2 or cloud.shape[1] != POINT_CHANNELS:
            raise ValueError(f"cloud {i} must be (M, {POINT_CHANNELS}), got {cloud.shape}")
        if cloud.shape[0] < num_points:
            raise ValueError(
                f"cloud {i} has only {cloud.shape[0]} points after the workspace crop, "
                f"fewer than num_points={num_points} -- corrupt depth or a crop that "
                "misses the scene; refusing to zero-pad phantom geometry"
            )
        # replace=True only when the crop kept fewer points than the pool, which merely
        # duplicates rows that FPS then dedups by construction (distinct farthest picks).
        take = rng.choice(cloud.shape[0], size=pool_size,
                          replace=cloud.shape[0] < pool_size)
        pooled = cloud[take]
        fps_picks = farthest_point_indices(pooled[:, :3], num_points)
        out[i] = pooled[fps_picks]
        indices[i] = take[fps_picks]
    if return_indices:
        return out, indices
    return out


def build_resampled_world_clouds(
    depth_mm: np.ndarray,
    rgb: np.ndarray,
    intrinsic: np.ndarray,
    world_t_cam: np.ndarray,
    *,
    num_points: int,
    pool_size: int,
    rng: np.random.Generator,
    crop_min: tuple[float, float, float] = CROP_MIN,
    crop_max: tuple[float, float, float] = CROP_MAX,
    min_depth: float = MIN_DEPTH_M,
    max_depth: float = MAX_DEPTH_M,
    masks: np.ndarray | None = None,
    object_nums: int = 2,
    crop_frames: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
    """Build a window of sampled clouds and optional labels through one row lineage.

    Supplying ``masks`` changes neither point values nor RNG draws. Pixel labels follow
    valid-depth selection, workspace crop, uniform pooling, and XYZ-only FPS indices.
    """
    depth_mm = np.asarray(depth_mm)
    rgb = np.asarray(rgb)
    world_t_cam = np.asarray(world_t_cam)
    if depth_mm.ndim != 3:
        raise ValueError(f"depth window must be (T,H,W), got {depth_mm.shape}")
    frames, height, width = depth_mm.shape
    if depth_mm.dtype != np.uint16:
        raise ValueError(f"depth window must be uint16 millimetres, got {depth_mm.dtype}")
    if rgb.shape != (frames, height, width, 3):
        raise ValueError(
            f"rgb window {rgb.shape} != {(frames, height, width, 3)}"
        )
    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb window must be uint8, got {rgb.dtype}")
    intrinsic = np.asarray(intrinsic)
    if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
        raise ValueError(f"intrinsic must be a finite (3,3) matrix, got {intrinsic.shape}")
    if world_t_cam.shape != (frames, 4, 4):
        raise ValueError(
            f"world_t_cam window {world_t_cam.shape} != {(frames, 4, 4)}"
        )
    if not np.all(np.isfinite(world_t_cam)):
        raise ValueError("world_t_cam window contains non-finite values")
    if frames < 1:
        raise ValueError("observation window is empty")
    if masks is not None:
        masks = np.asarray(masks)
        if masks.shape != (frames, height, width) or masks.dtype not in (
            np.dtype(np.uint8), np.dtype(np.int8)
        ):
            raise ValueError(
                f"masks must be {(frames, height, width)} uint8/int8, got "
                f"{masks.shape} {masks.dtype}"
            )
        if masks.min() < 0 or masks.max() > object_nums:
            raise ValueError(f"mask labels must be in [0,{object_nums}]")

    clouds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for frame_index in range(frames):
        kwargs = dict(
            crop_min=crop_min,
            crop_max=crop_max,
            min_depth=min_depth,
            max_depth=max_depth,
            frame_t_world=None if crop_frames is None else crop_frames[frame_index],
        )
        if masks is None:
            cloud = build_world_cloud(
                depth_mm[frame_index], rgb[frame_index], intrinsic,
                world_t_cam[frame_index], **kwargs
            )
        else:
            cloud, pixel_indices = build_world_cloud(
                depth_mm[frame_index], rgb[frame_index], intrinsic,
                world_t_cam[frame_index], return_pixel_index=True, **kwargs
            )
            labels.append(masks[frame_index].ravel()[pixel_indices])
        clouds.append(cloud)

    point_mask = None
    if masks is None:
        point_cloud = resample_clouds(
            clouds, num_points, pool_size=pool_size, rng=rng
        )
    else:
        point_cloud, source_indices = resample_clouds(
            clouds, num_points, pool_size=pool_size, rng=rng, return_indices=True
        )
        point_mask = np.stack([
            frame_labels[source_indices[frame_index]]
            for frame_index, frame_labels in enumerate(labels)
        ]).astype(np.int8, copy=False)

    diagnostics = {
        "cropped_point_counts": np.asarray(
            [len(cloud) for cloud in clouds], dtype=np.int64
        ),
    }
    if point_mask is not None:
        diagnostics["selected_label_counts"] = np.stack([
            np.bincount(frame, minlength=object_nums + 1)
            for frame in point_mask
        ]).astype(np.int64, copy=False)
    return point_cloud, point_mask, diagnostics
