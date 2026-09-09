#!/usr/bin/env python3
"""Head-camera (ZED X Mini) crop/resize geometry + intrinsics.

Shared by the teleop recorder (``leader/vr_reader.py``) and the policy-rollout
consumer (``follower/policy_rollout.py``) so the recorded training data and the
inference-time observation are processed identically and stamped with the SAME
intrinsic.

The head publisher (``tests/test_head_zedx_depth.py``) crops the raw SVGA frame to
``HEAD_CROP_TBLR`` then resizes to ``HEAD_RESIZE_HW`` ROBOT-SIDE before publishing.
So both the recorder and the rollout consume the model-ready frame AS-IS and stamp
``ZED_K`` — the intrinsic of that cropped+resized view (NOT the raw SVGA frame).
Keep these constants in sync with the publisher's ``--crop`` / ``--resize``.
"""

from __future__ import annotations

import numpy as np

# ZED capture resolution, then the crop applied to that raw frame as rows [top:bottom],
# cols [left:right], then a resize to (height, width). SINGLE SOURCE OF TRUTH for the
# head-camera geometry, consumed by every stage so they stay in lock-step:
#   * tests/test_head_zedx_depth.py — --resolution/--crop/--resize default to these
#     (robot-side).
#   * leader/vr_reader.py, follower/policy_rollout.py — stamp ZED_K (derived below).
#   * scripts/drive_box_record.py — records ZED_K alongside the head frames.
#   * scripts/record_arm_scan.py — requires the head frames to be HEAD_RESIZE_HW.
# Change them HERE and every consumer follows.
#
# Currently: HD1200 capture (1920x1200, the sensor's full 16:10 field of view; SVGA is its
# 2x2-binned 960x600) resized to 1200x750 (WxH) with no ROI crop -- 1.56x the pixels of the
# SVGA profile at the SAME field of view. Chosen 2026-09-03 for the world model after a
# six-eye probe: 72 Mbps, max recorder age 139 ms, zero rejects (raw HD1080 fails the 10 Hz
# recorder contract). The earlier SVGA profile is
#     HEAD_RESOLUTION = "SVGA"; HEAD_CROP_TBLR = (0, 600, 0, 960); HEAD_RESIZE_HW = (600, 960)
# with HEAD_BASE_K fx 377.13342, cx 489.7027, cy 319.49548 (see git history); the
# manipulation-ROI crop used before that was (300, 600, 375, 775) -> (240, 320).
# Rerun the test_head_zedx_depth.py publisher after changing these constants.
HEAD_RESOLUTION: str = "HD1200"
HEAD_CROP_TBLR: tuple[int, int, int, int] = (0, 1200, 0, 1920)
HEAD_RESIZE_HW: tuple[int, int] = (750, 1200)
# (H, W); does not affect pose tracking, which runs on the raw capture.

# Intrinsics of the raw HD1200 head frame (1920x1200), pre-crop: the ZED SDK's RECTIFIED
# left-cam calibration for this unit at HD1200, i.e. exactly what
# ``tests/test_head_zedx_depth.py`` reports as ``stereo.left_K`` before its crop/resize
# (SDK ``camera_configuration.calibration_parameters.left_cam``; rectified, so fx == fy).
# Taking the SDK value verbatim keeps ZED_K and the publisher's left_K algebraically
# identical for ANY --crop/--resize, since both apply crop_resize_intrinsics() to this
# same base -- which is what wbc_vr_robot's meta/head_stereo check compares.
#
# These are the repeatable RECTIFIED factory-derived values reported by the publisher with
# ``init.camera_disable_self_calib = True`` (self-calibration re-estimates fx at every open,
# measured 0.145% spread on this unit, which breaks the 0.107% meta/head_stereo tolerance).
# Read 2026-09-03 from sensors/head_camera/info at HD1200 -> 1200x750 (fx 471.34171,
# cx 612.01920, cy 399.30573) and divided by the exact 0.625 resize; NOT 2x the SVGA
# constant, which the SDK's HD1200 calibration differs from by ~0.1 px.
#
# CONSEQUENCE: obs/images/intrinsic differs from every SVGA-era take. port_wbc_mobile_zarr.py
# requires ONE exact intrinsic across all source takes, so SVGA takes and HD1200 takes cannot
# be ported together, and checkpoints trained on one expect it at rollout. Stereo recordings
# stamp the live publisher left_K directly; this constant remains the compatibility fallback
# for legacy left+SDK-depth paths and other consumers. These numbers are per-unit and
# resolution-specific: re-read them from the publisher's info service if the camera or
# HEAD_RESOLUTION changes.
HEAD_BASE_K = np.array(
    [
        [754.146728515625, 0.0, 979.230712890625],
        [0.0, 754.146728515625, 638.88916015625],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def crop_resize_intrinsics(
    K: np.ndarray, crop_tblr: tuple[int, int, int, int], out_hw: tuple[int, int]
) -> np.ndarray:
    """Adjust pinhole K for ``img[top:bottom, left:right]`` then a resize to (h, w).

    Cropping shifts the principal point by the crop top-left; resizing scales the
    focal length + principal point by the per-axis resize ratio.
    """
    top, bottom, left, right = crop_tblr
    out_h, out_w = out_hw
    sx = out_w / (right - left)
    sy = out_h / (bottom - top)
    K2 = K.copy()
    K2[0, 2] -= left  # crop shifts the principal point
    K2[1, 2] -= top
    K2[0, 0] *= sx  # resize scales focal length + principal point
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


# Intrinsics of the model-view (cropped to HEAD_CROP_TBLR + resized to HEAD_RESIZE_HW)
# head frame — the one recorded into obs/images/intrinsic and fed to SceneDiff.
ZED_K = crop_resize_intrinsics(HEAD_BASE_K, HEAD_CROP_TBLR, HEAD_RESIZE_HW).astype(
    np.float32
)
