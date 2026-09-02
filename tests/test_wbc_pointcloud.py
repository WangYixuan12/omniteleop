"""Unit tests for omniteleop.wbc_pointcloud (world-frame head point clouds).

numpy + fpsample; run via the dexmate env with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
"""

from __future__ import annotations

import numpy as np
import pytest

from omniteleop.wbc_pointcloud import (
    POINT_CHANNELS,
    build_resampled_world_clouds,
    build_world_cloud,
    crop_workspace,
    farthest_point_indices,
    resample_clouds,
    sampler_signature,
    unproject_head_frame,
    world_t_head_from_state,
)
from omniteleop.wbc_policy_format import base_pose_to_mat, mat_to_pos6d

FX, FY, CX, CY = 100.0, 120.0, 15.0, 10.0
K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])


def _blank(h: int = 20, w: int = 30):
    return np.zeros((h, w), np.uint16), np.zeros((h, w, 3), np.uint8)


def _yaw_pose(yaw: float, tx: float, ty: float, tz: float) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                             [np.sin(yaw), np.cos(yaw), 0.0],
                             [0.0, 0.0, 1.0]])
    pose[:3, 3] = (tx, ty, tz)
    return pose


def test_unproject_matches_analytic_pinhole_under_a_nontrivial_extrinsic():
    depth, rgb = _blank()
    u, v, z_m = 22, 7, 1.25
    depth[v, u] = int(z_m * 1000)
    rgb[v, u] = (255, 128, 0)
    world_t_cam = _yaw_pose(np.pi / 2, 0.4, -0.3, 1.4)

    cloud = unproject_head_frame(depth, rgb, K, world_t_cam)

    assert cloud.shape == (1, POINT_CHANNELS)
    cam = np.array([(u - CX) / FX * z_m, (v - CY) / FY * z_m, z_m, 1.0])
    expected = (world_t_cam @ cam)[:3]
    np.testing.assert_allclose(cloud[0, :3], expected, atol=1e-5)
    np.testing.assert_allclose(cloud[0, 3:], np.array([255, 128, 0]) / 255.0, atol=1e-6)


def test_unproject_drops_depth_outside_the_validity_window():
    depth, rgb = _blank()
    depth[1, 1] = 0            # invalid (sensor hole)
    depth[2, 2] = 30           # 0.03 m, below min_depth
    depth[3, 3] = 9000         # 9.0 m, above max_depth
    depth[4, 4] = 1500         # 1.5 m, the only keeper
    cloud = unproject_head_frame(depth, rgb, K, np.eye(4))
    assert cloud.shape == (1, POINT_CHANNELS)
    np.testing.assert_allclose(cloud[0, 2], 1.5, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"depth_mm": np.zeros(5, np.uint16)},
        {"rgb": np.zeros((20, 30), np.uint8)},
        {"rgb": np.zeros((4, 4, 3), np.uint8)},
        {"intrinsic": np.eye(4)},
        {"world_t_cam": np.eye(3)},
        {"min_depth": 4.0},  # >= max_depth
    ],
)
def test_unproject_rejects_bad_inputs(kwargs):
    depth, rgb = _blank()
    args = {"depth_mm": depth, "rgb": rgb, "intrinsic": K, "world_t_cam": np.eye(4)}
    args.update(kwargs)
    with pytest.raises(ValueError):
        unproject_head_frame(**args)


def test_crop_workspace_keeps_only_the_strict_interior():
    cloud = np.array([
        [0.5, 0.0, 1.0, 0, 0, 0],    # inside
        [0.0, 0.0, 1.0, 0, 0, 0],    # on the lower x bound -> dropped (strict)
        [1.5, 0.0, 1.0, 0, 0, 0],    # above the upper x bound
        [0.5, -2.0, 1.0, 0, 0, 0],   # below the lower y bound
    ], dtype=np.float32)
    kept = crop_workspace(cloud, (0.0, -1.0, 0.0), (1.0, 1.0, 2.0))
    assert kept.shape == (1, POINT_CHANNELS)
    np.testing.assert_allclose(kept[0, :3], [0.5, 0.0, 1.0])


def test_crop_workspace_rejects_bad_shape_and_inverted_bounds():
    with pytest.raises(ValueError):
        crop_workspace(np.zeros((4, 3), np.float32))
    with pytest.raises(ValueError):
        crop_workspace(np.zeros((4, POINT_CHANNELS), np.float32), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))


def test_world_t_head_from_state_composes_the_odometry_in_base_frame():
    base_pose = np.array([0.7, -0.2, 0.3])          # x, y, yaw
    base_t_zed = _yaw_pose(0.15, 0.05, 0.0, 1.4)
    state = np.zeros(32)
    state[20:29] = mat_to_pos6d(base_t_zed)
    state[29:32] = base_pose

    got = world_t_head_from_state(state, "base")
    np.testing.assert_allclose(got, base_pose_to_mat(base_pose) @ base_t_zed, atol=1e-9)

    # In "world" frame the head block IS world_T_zed; the base pose must not be applied.
    state_world = np.zeros(32)
    state_world[20:29] = mat_to_pos6d(base_t_zed)
    state_world[29:32] = base_pose
    np.testing.assert_allclose(world_t_head_from_state(state_world, "world"), base_t_zed, atol=1e-9)


def test_world_t_head_from_state_rejects_bad_frame_before_decoding_the_pose():
    # A bad frame name must not surface as a confusing "degenerate rotation" error.
    with pytest.raises(ValueError, match="state_frame"):
        world_t_head_from_state(np.zeros(32), "camera")
    with pytest.raises(ValueError, match=r"\(32,\)"):
        world_t_head_from_state(np.zeros(29), "base")


def test_build_world_cloud_unprojects_then_crops():
    depth, rgb = _blank()
    depth[10, 15] = 1000       # ~1 m straight ahead of the principal point
    depth[11, 16] = 2500       # 2.5 m -> outside a 1.5 m crop
    cloud = build_world_cloud(depth, rgb, K, np.eye(4),
                              crop_min=(-1.0, -1.0, 0.0), crop_max=(1.0, 1.0, 1.5))
    assert cloud.shape == (1, POINT_CHANNELS)
    np.testing.assert_allclose(cloud[0, 2], 1.0, atol=1e-6)


def test_farthest_point_indices_picks_distinct_spread_points():
    # Two far-apart clusters of near-duplicates: FPS must take one from each, not two
    # neighbours. Uses >= 2**FPS_KDLINE_H points, the kd-line sampler's minimum.
    rng = np.random.default_rng(0)
    near = rng.normal(0.0, 1e-3, size=(32, 3)).astype(np.float32)
    far = np.array([10.0, 0.0, 0.0], np.float32) + rng.normal(0.0, 1e-3, size=(32, 3))
    pts = np.concatenate([near, far.astype(np.float32)])
    idx = farthest_point_indices(pts, 2)
    assert idx.shape == (2,)
    assert (idx < 32).sum() == 1 and (idx >= 32).sum() == 1


def test_farthest_point_indices_rejects_a_cloud_too_small_for_the_bucket_sampler():
    with pytest.raises(ValueError, match=r"need N >= 2\*\*5"):
        farthest_point_indices(np.zeros((8, 3), np.float32), 4)


def test_farthest_point_indices_is_deterministic():
    """fpsample's default start_idx is RANDOM; we pin it, else train/deploy clouds diverge."""
    pts = np.random.default_rng(0).random((4096, 3)).astype(np.float32)
    first = farthest_point_indices(pts, 256)
    for _ in range(4):
        np.testing.assert_array_equal(farthest_point_indices(pts, 256), first)


def test_sampler_signature_pins_the_downsampler():
    sig = sampler_signature()
    assert sig["sampler"] == "fpsample.bucket_fps_kdline_sampling"
    assert sig["kdline_h"] == 5
    assert sig["start_idx"] == 0          # never None -- that would be a random seed point
    assert sig["fpsample_version"]


def test_farthest_point_indices_rejects_bad_shape_and_oversized_k():
    with pytest.raises(ValueError):
        farthest_point_indices(np.zeros((8, 2), np.float32), 4)
    with pytest.raises(ValueError):
        farthest_point_indices(np.zeros((4, 3), np.float32), 5)


def test_resample_clouds_returns_a_fixed_size_distinct_subset_of_the_input():
    rng = np.random.default_rng(0)
    clouds = [rng.random((500, POINT_CHANNELS)).astype(np.float32) for _ in range(3)]
    out = resample_clouds(clouds, 64, pool_size=128, rng=rng)

    assert out.shape == (3, 64, POINT_CHANNELS)
    assert out.dtype == np.float32
    for i, cloud in enumerate(clouds):
        rows = {tuple(p) for p in cloud}
        assert all(tuple(p) in rows for p in out[i]), "resampled points must come from the input"
        assert len({tuple(p[:3]) for p in out[i]}) == 64, "FPS must return distinct points"


def test_resample_return_indices_preserves_cloud_and_rng_sequence():
    cloud = np.arange(500 * POINT_CHANNELS, dtype=np.float32).reshape(500, POINT_CHANNELS)
    rng_plain = np.random.default_rng(17)
    rng_indexed = np.random.default_rng(17)

    plain = resample_clouds([cloud], 64, pool_size=128, rng=rng_plain)
    indexed, source_idx = resample_clouds(
        [cloud], 64, pool_size=128, rng=rng_indexed, return_indices=True)

    np.testing.assert_array_equal(indexed, plain)
    np.testing.assert_array_equal(indexed[0], cloud[source_idx[0]])
    np.testing.assert_array_equal(
        rng_indexed.integers(0, 2**31, size=16),
        rng_plain.integers(0, 2**31, size=16),
    )


def test_build_world_cloud_pixel_indices_compose_validity_and_crop():
    depth, rgb = _blank(h=8, w=8)
    depth[1, 2] = 1000   # valid and inside crop
    depth[3, 4] = 2000   # valid depth but outside z crop
    depth[6, 7] = 0      # invalid depth
    cloud, pixel_idx = build_world_cloud(
        depth, rgb, np.array([[100.0, 0, 0], [0, 100.0, 0], [0, 0, 1.0]]),
        np.eye(4), crop_min=(-1, -1, 0.5), crop_max=(1, 1, 1.5),
        return_pixel_index=True)

    assert cloud.shape == (1, POINT_CHANNELS)
    np.testing.assert_array_equal(pixel_idx, [1 * 8 + 2])


def test_resample_clouds_refuses_to_pad_a_short_cloud():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="fewer than num_points"):
        resample_clouds([np.zeros((8, POINT_CHANNELS), np.float32)], 16,
                        pool_size=32, rng=rng)


def test_resample_clouds_rejects_pool_smaller_than_num_points_and_empty_input():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        resample_clouds([np.zeros((64, POINT_CHANNELS), np.float32)], 32, pool_size=16,
                        rng=rng)
    with pytest.raises(ValueError):
        resample_clouds([], 4, pool_size=8, rng=rng)


def test_resampled_window_labels_preserve_cloud_and_rng_sequence():
    frames, height, width = 2, 20, 30
    depth = np.full((frames, height, width), 1000, dtype=np.uint16)
    masks = np.zeros((frames, height, width), dtype=np.uint8)
    masks[:, :, width // 3:2 * width // 3] = 1
    masks[:, :, 2 * width // 3:] = 2
    rgb = np.zeros((frames, height, width, 3), dtype=np.uint8)
    rgb[..., 0] = masks * 50
    poses = np.repeat(np.eye(4)[None], frames, axis=0)
    rng_plain = np.random.default_rng(41)
    rng_masked = np.random.default_rng(41)
    kwargs = dict(
        intrinsic=K,
        world_t_cam=poses,
        num_points=64,
        pool_size=128,
        crop_min=(-10.0, -10.0, 0.0),
        crop_max=(10.0, 10.0, 2.0),
    )

    plain, no_mask, _ = build_resampled_world_clouds(
        depth, rgb, rng=rng_plain, **kwargs
    )
    labeled, point_mask, diagnostics = build_resampled_world_clouds(
        depth, rgb, masks=masks, rng=rng_masked, **kwargs
    )

    np.testing.assert_array_equal(labeled, plain)
    assert no_mask is None
    np.testing.assert_array_equal(
        point_mask, np.rint(labeled[..., 3] * 255.0 / 50.0).astype(np.int8)
    )
    assert diagnostics["selected_label_counts"].shape == (frames, 3)
    np.testing.assert_array_equal(
        rng_masked.integers(0, 2**31, size=16),
        rng_plain.integers(0, 2**31, size=16),
    )


@pytest.mark.parametrize(
    ("field", "dtype", "message"),
    [("depth", np.float32, "uint16"), ("rgb", np.float32, "uint8")],
)
def test_resampled_window_rejects_non_native_rgbd_dtypes(field, dtype, message):
    depth = np.full((1, 20, 30), 1000, dtype=np.uint16)
    rgb = np.zeros((1, 20, 30, 3), dtype=np.uint8)
    if field == "depth":
        depth = depth.astype(dtype)
    else:
        rgb = rgb.astype(dtype)
    with pytest.raises(ValueError, match=message):
        build_resampled_world_clouds(
            depth,
            rgb,
            K,
            np.eye(4)[None],
            num_points=64,
            pool_size=128,
            rng=np.random.default_rng(1),
            crop_min=(-10.0, -10.0, 0.0),
            crop_max=(10.0, 10.0, 2.0),
        )
