"""Offline unit tests for omniteleop.follower.zed_base_pose (no camera, no Zenoh).

Synthetic round-trip: pick a ground-truth base trajectory T_odom_base(t) and
joint trajectories q(t); generate the camera poses a ZED tracker would report,
T_track_cam(t) = T_track_odom · T_odom_base(t) · T_base_cam(q_t), with an
arbitrary track-world offset; feed them to ZedBasePoseEstimator and require the
ground-truth base trajectory back to numerical precision. This validates the
anchoring + composition math independent of hardware.

Run:
    conda run -n yixuan_yifan python -m pytest tests/test_zed_base_pose.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, (Path(__file__).resolve().parent.parent / "src").as_posix())

from omniteleop.follower.zed_base_pose import (  # noqa: E402
    DEFAULT_CAMERA_FRAME,
    VegaHeadCamFK,
    ZedBasePoseEstimator,
    pose_from_quat_trans,
    se3_to_xy_yaw,
)


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    return T


def _trans(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


@pytest.fixture(scope="module")
def fk() -> VegaHeadCamFK:
    return VegaHeadCamFK()


def test_fk_uses_real_tracker_left_eye_frame(fk: VegaHeadCamFK) -> None:
    """The default FK frame is the LEFT EYE — the point the ZED SDK actually
    tracks (getPosition returns "the Camera Frame located on the left eye"), not
    zed_depth_frame. At q=0 the optical axis is base +x and the eye sits 25 mm
    left."""
    assert DEFAULT_CAMERA_FRAME == "zed_left_camera"
    assert fk.camera_frame == "zed_left_camera"
    T = fk.base_T_cam([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    optical_axis = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(optical_axis, [1.0, 0.0, 0.0], atol=1e-5)
    assert abs(T[1, 3] - 0.025) < 1e-6, "ZED frame should sit +25 mm left"

    # zed_depth_frame shares the orientation but is 11.5 mm behind the left eye
    # along the optical z (forward) axis — feeding the SDK's left-eye pose to a
    # depth-frame FK is the bug this default avoids.
    depth = VegaHeadCamFK(camera_frame="zed_depth_frame").base_T_cam(
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(depth[:3, :3], T[:3, :3], atol=1e-10)
    off_optical = T[:3, :3].T @ (depth[:3, 3] - T[:3, 3])
    np.testing.assert_allclose(off_optical, [0.0, 0.0, -0.0115], atol=1e-6)


def test_head_yaw_swings_optical_axis(fk: VegaHeadCamFK) -> None:
    """head_j2 (axis +z) by +0.5 rad must yaw the optical axis by +0.5 rad."""
    T = fk.base_T_cam([0.0, 0.0, 0.0], [0.0, 0.5, 0.0])
    optical_axis = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
    expected = np.array([np.cos(0.5), np.sin(0.5), 0.0])
    np.testing.assert_allclose(optical_axis, expected, atol=1e-5)


def test_estimator_recovers_base_trajectory(fk: VegaHeadCamFK) -> None:
    """Round-trip: synthetic camera poses → estimator → ground-truth base poses."""
    rng = np.random.default_rng(0)
    # arbitrary track-world: tracker booted in some unknown gravity-aligned frame
    track_T_odom = _trans(0.7, -1.3, 0.4) @ _rot_z(1.1) @ _rot_x(0.3)

    est = ZedBasePoseEstimator(fk)
    assert not est.anchored

    n = 50
    errs_t, errs_r = [], []
    for i in range(n):
        s = i / (n - 1)
        # ground truth base: drive an arc while joints sweep
        odom_T_base = _trans(1.5 * s, 0.8 * np.sin(2 * s), 0.0) @ _rot_z(0.9 * s)
        torso = np.array([0.3 + 0.4 * s, 1.2 - 0.3 * s, 0.1 * np.sin(3 * s)])
        head = np.array([-0.5 * s, 0.6 * np.cos(2 * s) - 0.3, 0.2 * s])
        torso += rng.normal(0, 1e-12, 3)  # exercise float path, not behavior

        track_T_cam = track_T_odom @ odom_T_base @ fk.base_T_cam(torso, head)
        out = est.update(track_T_cam, torso, head)

        errs_t.append(np.linalg.norm(out[:3, 3] - odom_T_base[:3, 3]))
        R_rel = out[:3, :3].T @ odom_T_base[:3, :3]
        errs_r.append(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))

    assert est.anchored
    assert max(errs_t) < 1e-9, f"translation error {max(errs_t):.3e} m"
    # arccos of a trace ≈ 1 bottoms out at ~sqrt(eps) ≈ 1.5e-8; 1e-6 rad is still
    # far below any physical effect and catches real frame-convention bugs.
    assert max(errs_r) < 1e-6, f"rotation error {max(errs_r):.3e} rad"

    # the very first sample defines odom: T_odom_base(0) = identity by anchoring
    est2 = ZedBasePoseEstimator(fk)
    torso0, head0 = [0.5, 1.0, 0.0], [0.1, -0.2, 0.3]
    first = est2.update(
        track_T_odom @ fk.base_T_cam(torso0, head0), torso0, head0
    )
    np.testing.assert_allclose(first, np.eye(4), atol=1e-10)


def test_camera_only_motion_keeps_base_fixed(fk: VegaHeadCamFK) -> None:
    """Pure head/torso motion (base parked) must leave the estimate at identity.

    This is the discriminating property of the FK fusion — a wrong frame
    convention shows up here as a moving base.
    """
    track_T_odom = _rot_z(-0.7) @ _trans(0.2, 0.1, 1.0)
    est = ZedBasePoseEstimator(fk)
    for s in np.linspace(0.0, 1.0, 25):
        torso = [0.2 + 0.5 * s, 1.4 - 0.6 * s, 0.2 * s]
        head = [0.8 * s - 0.4, 1.0 * np.sin(s), 0.3 * np.cos(2 * s)]
        track_T_cam = track_T_odom @ fk.base_T_cam(torso, head)  # odom_T_base = I
        out = est.update(track_T_cam, torso, head)
        np.testing.assert_allclose(out, np.eye(4), atol=1e-9)
    x, y, yaw = se3_to_xy_yaw(out)
    assert (abs(x) + abs(y) + abs(yaw)) < 1e-9


def test_default_fk_matches_real_tracker_frame_when_head_moves() -> None:
    """The default estimator must use the frame the ZED SDK actually tracks — the
    LEFT EYE (``zed_left_camera``).

    The SDK reports the left-eye pose; feeding those poses to the default FK keeps
    a parked base at identity under pure head motion. An estimator that FKs to
    ``zed_depth_frame`` instead (11.5 mm behind the left eye) turns the *same*
    poses into a moving-base artifact — the bug fixed by the default frame. This
    test would fail if the default regressed to zed_depth_frame.
    """
    left_fk = VegaHeadCamFK(camera_frame="zed_left_camera")  # what the SDK tracks
    est_default = ZedBasePoseEstimator(VegaHeadCamFK())       # default == left eye
    est_depth = ZedBasePoseEstimator(VegaHeadCamFK(camera_frame="zed_depth_frame"))
    track_T_odom = _rot_z(0.4) @ _trans(-0.3, 0.2, 0.5)

    max_depth_drift = 0.0
    for s in np.linspace(0.0, 1.0, 11):
        torso = [0.15 * s, 0.4 + 0.2 * s, -0.2 * s]
        head = [0.4 * np.sin(s), 1.1 * s - 0.3, -0.5 * np.cos(s)]
        track_T_cam = track_T_odom @ left_fk.base_T_cam(torso, head)  # SDK = left eye
        np.testing.assert_allclose(
            est_default.update(track_T_cam, torso, head), np.eye(4), atol=1e-9
        )
        out_depth = est_depth.update(track_T_cam, torso, head)
        max_depth_drift = max(max_depth_drift, float(np.linalg.norm(out_depth[:3, 3])))
    assert max_depth_drift > 1e-3, (
        "FK to zed_depth_frame should move a parked base under head motion "
        f"(got {max_depth_drift:.2e} m) — the tracked frame must matter"
    )


def test_estimator_requires_strict_se3(fk: VegaHeadCamFK) -> None:
    """Post-cleanup contract: the estimator strictly validates its camera-pose
    input. Callers build it from the SDK quaternion (pose_from_quat_trans, exact
    SE(3)); the SDK's raw denormalized 4x4 is no longer accepted."""
    torso, head = [0.5, 1.2, 0.0], [-0.3, 0.1, 0.2]
    track_T_cam = _rot_z(0.4) @ _trans(1.0, -0.5, 0.9) @ fk.base_T_cam(torso, head)

    # exact SE(3) (as pose_from_quat_trans produces) → anchors to identity
    est = ZedBasePoseEstimator(fk)
    first = est.update(track_T_cam, torso, head)
    np.testing.assert_allclose(first, np.eye(4), atol=1e-9)

    # the old SDK-style denormalized matrix is now rejected, not silently fixed
    dirty = track_T_cam.copy()
    dirty[:3, :3] *= 1.0017  # det(R) ≈ 1.005
    dirty[3, 3] = 0.9999
    with pytest.raises(ValueError, match="orthonormal|det|last row"):
        ZedBasePoseEstimator(fk).update(dirty, torso, head)


def test_pose_from_quat_trans_matches_scipy() -> None:
    """xyzw quaternion → rotation must agree with scipy for random rotations,
    and survive float32 quantization (the SDK's native precision) to ~1e-7."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(7)
    for _ in range(50):
        rot = Rotation.random(rng=rng)
        t = rng.normal(0, 5, 3)
        T = pose_from_quat_trans(rot.as_quat(), t)  # scipy as_quat is xyzw too
        np.testing.assert_allclose(T[:3, :3], rot.as_matrix(), atol=1e-12)
        np.testing.assert_allclose(T[:3, 3], t, atol=0)
        np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=0)

        # float32 round-trip, as the SDK stores it
        q32 = rot.as_quat().astype(np.float32).astype(np.float64)
        T32 = pose_from_quat_trans(q32, t.astype(np.float32))
        R = T32[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)  # exact SO(3)
        np.testing.assert_allclose(R, rot.as_matrix(), atol=5e-7)


def test_pose_from_quat_trans_validation() -> None:
    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        pose_from_quat_trans([0, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        pose_from_quat_trans([0, 0, 0, 1], [0, 0])
    with pytest.raises(ValueError, match="non-finite"):
        pose_from_quat_trans([np.nan, 0, 0, 1], [0, 0, 0])
    with pytest.raises(ValueError, match="norm"):
        pose_from_quat_trans([0.5, 0.5, 0.5, 0.6], [0, 0, 0])  # norm ≈ 1.05


def test_input_validation(fk: VegaHeadCamFK) -> None:
    est = ZedBasePoseEstimator(fk)

    with pytest.raises(ValueError, match="must have exactly 3"):
        fk.base_T_cam([0.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        fk.base_T_cam([np.nan, 0.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="4x4"):
        est.update(np.eye(3), [0, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError, match="non-finite"):
        est.update(np.full((4, 4), np.nan), [0, 0, 0], [0, 0, 0])
    bad = np.eye(4)
    bad[:3, :3] *= 2.0  # not orthonormal
    with pytest.raises(ValueError, match="orthonormal"):
        est.update(bad, [0, 0, 0], [0, 0, 0])
    refl = np.eye(4)
    refl[0, 0] = -1.0  # orthonormal but det = -1
    with pytest.raises(ValueError, match="det"):
        est.update(refl, [0, 0, 0], [0, 0, 0])
    skewed = np.eye(4)
    skewed[3, :] = [0.1, 0, 0, 1]
    with pytest.raises(ValueError, match="last row"):
        est.update(skewed, [0, 0, 0], [0, 0, 0])


def test_unknown_frame_or_joint_raises() -> None:
    with pytest.raises(ValueError, match="not found"):
        VegaHeadCamFK(camera_frame="no_such_frame")
