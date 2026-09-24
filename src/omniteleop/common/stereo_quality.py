"""Pixel-level diagnostics for nominally rectified stereo RGB pairs."""

from __future__ import annotations

import numpy as np


def feature_rectification_stats(
    left: np.ndarray,
    right: np.ndarray,
    *,
    nfeatures: int = 800,
    ratio: float = 0.72,
    min_matches: int = 5,
) -> dict | None:
    """Measure row residuals using mutual Lowe-ratio SIFT correspondences.

    Publisher metadata proves which calibration was advertised; this function checks
    the delivered pixels themselves. Mutual matching rejects many repeated-texture
    outliers that made the earlier one-way matcher report unstable p90 residuals.
    ``None`` means the scene did not contain enough trustworthy correspondences.
    """
    left = np.asarray(left)
    right = np.asarray(right)
    if (
        left.shape != right.shape
        or left.ndim != 3
        or left.shape[2] != 3
        or left.dtype != np.uint8
        or right.dtype != np.uint8
    ):
        raise ValueError(
            "stereo feature check requires equal (H,W,3) uint8 eyes, got "
            f"{left.shape} {left.dtype} and {right.shape} {right.dtype}"
        )
    if not isinstance(nfeatures, int) or nfeatures <= 0:
        raise ValueError(f"nfeatures must be a positive integer, got {nfeatures!r}")
    if not np.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError(f"ratio must be finite in (0,1), got {ratio!r}")
    if not isinstance(min_matches, int) or min_matches <= 0:
        raise ValueError(
            f"min_matches must be a positive integer, got {min_matches!r}"
        )
    try:
        import cv2  # noqa: PLC0415 -- optional diagnostic dependency
    except ImportError:
        return None

    gray_left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    detector = cv2.SIFT_create(nfeatures=nfeatures)
    key_left, desc_left = detector.detectAndCompute(gray_left, None)
    key_right, desc_right = detector.detectAndCompute(gray_right, None)
    if (
        desc_left is None
        or desc_right is None
        or len(desc_left) < 2
        or len(desc_right) < 2
    ):
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def ratio_pairs(query, train) -> dict[tuple[int, int], object]:
        accepted: dict[tuple[int, int], object] = {}
        for neighbors in matcher.knnMatch(query, train, k=2):
            if len(neighbors) != 2:
                continue
            best, second = neighbors
            if best.distance < ratio * second.distance:
                accepted[(best.queryIdx, best.trainIdx)] = best
        return accepted

    left_to_right = ratio_pairs(desc_left, desc_right)
    right_to_left = ratio_pairs(desc_right, desc_left)
    mutual = [
        match
        for (left_index, right_index), match in left_to_right.items()
        if (right_index, left_index) in right_to_left
    ]
    if len(mutual) < min_matches:
        return None

    vertical = np.asarray(
        [
            key_left[match.queryIdx].pt[1] - key_right[match.trainIdx].pt[1]
            for match in mutual
        ],
        dtype=np.float64,
    )
    disparity = np.asarray(
        [
            key_left[match.queryIdx].pt[0] - key_right[match.trainIdx].pt[0]
            for match in mutual
        ],
        dtype=np.float64,
    )
    return {
        "method": "sift_mutual_lowe_ratio_0.72",
        "matches": int(len(mutual)),
        "vertical_p50_px": float(np.median(np.abs(vertical))),
        "vertical_p90_px": float(np.percentile(np.abs(vertical), 90)),
        "disparity_p50_px": float(np.median(disparity)),
        "positive_disparity_fraction": float(np.mean(disparity >= -0.5)),
    }


def aggregate_rectification_stats(rows: list[dict]) -> dict | None:
    """Aggregate several frame diagnostics without letting one rich frame dominate."""
    if not rows:
        return None
    matches = np.asarray([int(row["matches"]) for row in rows], dtype=np.int64)
    if np.any(matches <= 0):
        raise ValueError("rectification rows must each contain positive matches")
    weights = matches.astype(np.float64)
    positive = np.asarray(
        [float(row["positive_disparity_fraction"]) for row in rows],
        dtype=np.float64,
    )
    return {
        "method": str(rows[0].get("method", "unknown")),
        "frames": int(len(rows)),
        "matches": int(matches.sum()),
        "vertical_p50_px": float(
            np.median([float(row["vertical_p50_px"]) for row in rows])
        ),
        "vertical_p90_px": float(
            np.median([float(row["vertical_p90_px"]) for row in rows])
        ),
        "vertical_p90_worst_px": float(
            max(float(row["vertical_p90_px"]) for row in rows)
        ),
        "disparity_p50_px": float(
            np.median([float(row["disparity_p50_px"]) for row in rows])
        ),
        "positive_disparity_fraction": float(np.average(positive, weights=weights)),
    }
