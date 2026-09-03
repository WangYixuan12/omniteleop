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

# Crop applied to the raw SVGA frame as rows [top:bottom], cols [left:right], then a
# resize to (height, width). SINGLE SOURCE OF TRUTH for the head-camera geometry,
# consumed by every stage so they stay in lock-step:
#   * tests/test_head_zedx_depth.py — --crop/--resize default to these (robot-side).
#   * leader/vr_reader.py, follower/policy_rollout.py — stamp ZED_K (derived below).
#   * scripts/drive_box_record.py — records ZED_K alongside the head frames.
# Change them HERE and every consumer follows.
#
# Currently set to the FULL SVGA frame (no ROI crop and no resize): 960x600 (WxH).
# To re-enable the manipulation-ROI crop, restore:
#     HEAD_CROP_TBLR = (300, 600, 375, 775)   # 300x400 ROI
#     HEAD_RESIZE_HW = (240, 320)
# rerun the test_head_zedx_depth.py publisher after changing these constants.
HEAD_CROP_TBLR: tuple[int, int, int, int] = (0, 600, 0, 960)  # (300, 600, 375, 775)
HEAD_RESIZE_HW: tuple[int, int] = (600, 960)  # (240, 320)
# (H, W); does not affect pose tracking on the raw SVGA (600, 960).

# Intrinsics of the raw SVGA head frame (960x600), pre-crop: the ZED SDK's RECTIFIED
# left-cam calibration for this unit at SVGA, i.e. exactly what
# ``tests/test_head_zedx_depth.py`` reports as ``stereo.left_K`` before its crop/resize
# (SDK ``camera_configuration.calibration_parameters.left_cam``; rectified, so fx == fy).
# Taking the SDK value verbatim keeps ZED_K and the publisher's left_K algebraically
# identical for ANY --crop/--resize, since both apply crop_resize_intrinsics() to this
# same base -- which is what wbc_vr_robot_tmp's meta/head_stereo check compares.
#
# These are the repeatable RECTIFIED factory-derived values reported by the publisher with
# ``init.camera_disable_self_calib = True``. The raw/unrectified values in
# /usr/local/zed/settings/SN50571637.conf intentionally differ because VIEW.LEFT is
# rectified by the SDK. With self-calibration enabled, three opens instead measured fx
# 384.9714, 385.0734, and 385.2446 px; an earlier edit mistakenly promoted the middle
# self-calibrated draw to the factory constant. Restarting the publisher with self-calib
# disabled exposed the stable value below (fx 377.13342).
#
# CONSEQUENCE: obs/images/intrinsic changes by ~2% for NEW takes. port_wbc_mobile_zarr.py
# requires ONE exact intrinsic across all source takes, so takes recorded before and after
# this change cannot be ported together, and checkpoints trained on the old value expect it
# at rollout. Stereo recordings now stamp the live publisher left_K directly; this constant
# remains the compatibility fallback for legacy left+SDK-depth paths and other consumers.
# These numbers are per-unit and resolution-specific (SVGA): re-read them from the
# publisher's info service if the camera or --resolution changes.
HEAD_BASE_K = np.array(
    [
        [377.13342, 0.0, 489.7027],
        [0.0, 377.13342, 319.49548],
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
