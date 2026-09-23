"""Offline features and safe annotations for the ZED Mini horizontal-strip fault.

The detector consumes both eyes of the LEFT wrist and a +/-2-row temporal window.
It does not infer that an unflagged image is clean. No pixels or timesteps are removed.
Inference uses a JSON forest (no pickle, sklearn, robot, or GPU dependency).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

FEATURE_VERSION = "zedm_strip_v1_320x180_stereo_context2"
ANNOTATION_KEY = "quality/left_wrist"
IMAGE_KEYS = ("obs/images/left_wrist_rgb", "obs/images/left_wrist_right_rgb")


def episode_paths(directory: Path) -> list[Path]:
    """Include finalized and quarantined recordings without treating one as the other."""
    return sorted([*directory.glob("*.hdf5"), *directory.glob("*.hdf5.partial")])


def validate_recording(f: h5py.File, path: Path) -> None:
    """Require consistent stored rows; preserve incomplete status for partial files."""
    partial = path.name.endswith(".hdf5.partial")
    if not bool(f.attrs.get("complete", False)) and not partial:
        raise ValueError(f"{path}: recording is not marked complete")
    n = len(f["timestamp_ns"])
    if int(f.attrs.get("n_frames", -1)) != n:
        raise ValueError(f"{path}: n_frames disagrees with timestamp_ns")
    if partial:

        def check(name, obj):
            if (
                isinstance(obj, h5py.Dataset)
                and obj.ndim
                and name.startswith(("obs/", "action/", "timing/"))
                and name != "obs/images/intrinsic"
                and obj.shape[0] != n
            ):
                raise ValueError(f"{path}:{name}: inconsistent partial frame count")

        f.visititems(check)


def read_labels(path: Path, index_base: int = 0) -> dict[str, list[int]]:
    """Parse ep3:/episode_3: sections; missing entries are unlabeled, not verified clean."""
    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 or 1")
    result: dict[str, list[int]] = {}
    current = None
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"(?:ep|episode_)(\d+):", line)
        if match:
            current = f"episode_{int(match[1])}"
            result.setdefault(current, [])
        elif line.isdigit() and current is not None:
            idx = int(line) - index_base
            if idx < 0:
                raise ValueError(f"{path}:{number}: negative index after index-base conversion")
            result[current].append(idx)
        else:
            raise ValueError(f"{path}:{number}: expected episode header or frame number: {line!r}")
    if not result or not any(result.values()):
        raise ValueError("labels contain no positive frames")
    return {key: sorted(set(indices)) for key, indices in result.items()}


def source_identity(path: Path) -> dict:
    """Identify a closed, consistent recording, including quarantined partial files."""
    path = path.resolve()
    before = path.stat()
    if not path.name.endswith((".hdf5", ".hdf5.partial")):
        raise ValueError(f"not an episode filename: {path}")
    with h5py.File(path, "r") as f:
        validate_recording(f, path)
        if (
            path.name.endswith(".hdf5.partial")
            and ANNOTATION_KEY not in f
            and time.time_ns() - before.st_mtime_ns < 10_000_000_000
        ):
            raise ValueError(f"{path}: recently modified partial; retry after 10 idle seconds")
        timestamps = f["timestamp_ns"][:]
        n = len(timestamps)
        if n < 3 or timestamps.shape != (n,) or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"{path}: need >=3 monotonically increasing timestamps")
        if int(f.attrs.get("n_frames", -1)) != n:
            raise ValueError(f"{path}: n_frames disagrees with timestamp_ns")
        shape = None
        for key in IMAGE_KEYS:
            ds = f[key]
            if ds.ndim != 4 or ds.shape[0] != n or ds.shape[-1] != 3 or ds.dtype != np.uint8:
                raise ValueError(f"{path}:{key}: expected uint8 (N,H,W,3)")
            if shape is not None and ds.shape != shape:
                raise ValueError(f"{path}: stereo eyes have different shapes")
            shape = ds.shape
        identity = {
            "path": str(path),
            "size": before.st_size,
            "mtime_ns": before.st_mtime_ns,
            "frames": n,
            "image_shape": list(shape),
            "timestamps_sha256": hashlib.sha256(timestamps.tobytes()).hexdigest(),
        }
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"{path}: changed while inspecting")
    return identity


def read_gray(path: Path, destination: Path) -> np.ndarray:
    """Cache only small grayscale stereo frames; HDF5 files stay read-only."""
    identity = source_identity(path)
    images = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.uint8,
        shape=(identity["frames"], 2, 180, 320),
    )
    with h5py.File(path, "r") as f:
        for i in range(len(images)):
            for eye, key in enumerate(IMAGE_KEYS):
                resized = cv2.resize(f[key][i], (320, 180), interpolation=cv2.INTER_AREA)
                images[i, eye] = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    images.flush()
    if source_identity(path) != identity:
        raise ValueError(f"{path}: changed while reading images")
    return images


def _summary(a: np.ndarray) -> list[float]:
    return [float(np.mean(a)), *np.percentile(a, [50, 75, 90, 95, 99]).tolist(), float(np.max(a))]


def appearance_features(images: np.ndarray, i: int) -> list[float]:
    """Spatial row discontinuities and temporal novelty; no labels enter features."""
    cur = images[i].astype(np.float32)
    refs = [
        images[j].astype(np.float32)
        for j in range(max(0, i - 2), min(len(images), i + 3))
        if j != i
    ]
    out = []
    for eye in range(2):
        im = cur[eye]
        for step in (1, 3, 6):
            diff = np.abs(im[step:] - im[:-step])
            row = diff.mean(axis=1)
            dx = np.abs(im[:, step:] - im[:, :-step]).mean()
            out += _summary(row) + _summary(diff.max(axis=1))
            out += [float(row.max() / (dx + 1)), float(row.max() / (np.median(row) + 1))]
        differences = np.stack([abs(im - ref[eye]) for ref in refs])
        for residual in (differences.min(axis=0), np.median(differences, axis=0)):
            row = residual.mean(axis=1)
            out += _summary(row) + _summary(abs(np.diff(row))) + _summary(residual)
        row = np.abs(im[3:] - im[:-3]).mean(axis=1)
        reference_rows = np.stack([np.abs(r[eye][3:] - r[eye][:-3]).mean(axis=1) for r in refs])
        out += _summary(np.maximum(0, row - np.median(reference_rows, axis=0)))
        out += _summary(differences.mean(axis=(1, 2)))
    return out


def displacement_features(images: np.ndarray, i: int) -> list[float]:
    """Match textured patches over the full width to detect large displaced strips.

    Two independent temporal references avoid mistaking the clean neighbor of a tear
    for another tear. Looking across the width recovers jumps too large for local flow.
    """
    out = []
    for eye in range(2):
        im = images[i, eye]
        candidates = []
        references = [j for j in (i - 2, i + 2) if 0 <= j < len(images)]
        if not references:
            references = [j for j in (0, len(images) - 1) if j != i]
        for j in references:
            ref = images[j, eye]
            rows = []
            for y in range(0, 169, 8):
                scores = []
                for x in (32, 128, 224):
                    template = im[y : y + 12, x : x + 64]
                    lo, hi = max(0, y - 16), min(180, y + 28)
                    corr = cv2.matchTemplate(ref[lo:hi], template, cv2.TM_CCOEFF_NORMED)
                    _, value, _, location = cv2.minMaxLoc(corr)
                    local = corr[:, max(0, x - 16) : min(corr.shape[1], x + 17)].max()
                    scores.append(
                        [
                            location[0] - x,
                            location[1] + lo - y,
                            value,
                            float(value - local),
                            float(template.std()),
                        ]
                    )
                rows.append(scores)
            rows = np.asarray(rows)
            dx, _, corr, gain, std = [rows[:, :, k] for k in range(5)]
            large = (abs(dx) > 40) & (corr > 0.75) & (std > 7)
            row_dx = np.median(dx, axis=1)
            candidates.append(
                _summary(np.where(large, corr, 0))
                + _summary(np.where(large, gain, 0))
                + _summary(large.sum(axis=1))
                + _summary(abs(np.diff(row_dx)))
                + _summary(np.where(large, abs(dx), 0))
                + [
                    float((large & (gain > 0.15)).sum()),
                    float(np.max(abs(row_dx - np.median(row_dx)))),
                ]
            )
        out += np.max(candidates, axis=0).tolist() + np.min(candidates, axis=0).tolist()
    return out


def frame_features(images: np.ndarray, i: int) -> list[float]:
    return appearance_features(images, i) + displacement_features(images, i)


def forest_scores(features: np.ndarray, model: dict) -> np.ndarray:
    """Evaluate exported sklearn tree structure using only NumPy, with no pickle load."""
    x = np.asarray(features, dtype=np.float32)
    if model["feature_version"] != FEATURE_VERSION or x.shape[1] != model["feature_count"]:
        raise ValueError("model and feature schema do not match")
    if not np.isfinite(x).all():
        raise ValueError("non-finite detector features")
    scores = np.zeros(len(x), dtype=np.float64)
    for tree in model["trees"]:
        left, right, feature = (
            np.asarray(tree[key], dtype=np.int32) for key in ("left", "right", "feature")
        )
        threshold = np.asarray(tree["threshold"])
        nodes = np.zeros(len(x), dtype=np.int32)
        while True:
            active = np.flatnonzero(left[nodes] != -1)
            if not len(active):
                break
            node = nodes[active]
            nodes[active] = np.where(
                x[active, feature[node]] <= threshold[node], left[node], right[node]
            )
        scores += np.asarray(tree["positive"])[nodes]
    return (scores / len(model["trees"])).astype(np.float32)


def make_annotations(scores: np.ndarray, manual_indices: list[int], model: dict) -> dict:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1 or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("scores must be a finite one-dimensional array in [0,1]")
    manual = np.zeros(len(scores), dtype=bool)
    if any(i < 0 or i >= len(scores) for i in manual_indices):
        raise ValueError("manual frame index out of range")
    manual[manual_indices] = True
    automatic = scores >= model["threshold"]
    review = (scores >= model["review_threshold"]) & ~automatic & ~manual
    return {
        "corrupted": automatic | manual,
        "auto_corrupted": automatic,
        "review_candidate": review,
        "score": scores,
    }


def annotation_digest(arrays: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode())
        digest.update(np.asarray(arrays[name]).tobytes())
    return digest.hexdigest()


def annotate_episode(path: Path, identity: dict, arrays: dict, provenance: dict) -> str:
    """Add one new quality group. Existing annotations are never silently overwritten.

    Reapplying identical arrays is a no-op. All mutations are restricted to quality/;
    the caller must finish scanning before opening the file for writing. HDF5 is not
    transactional: a pending group is retained after an interrupted write for inspection.
    """
    n = identity["frames"]
    if set(arrays) != {
        "corrupted",
        "auto_corrupted",
        "review_candidate",
        "score",
    }:
        raise ValueError("unexpected annotation fields")
    if any(np.asarray(a).shape != (n,) for a in arrays.values()):
        raise ValueError("annotation length differs from episode length")
    scores = np.asarray(arrays["score"])
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("scores must be finite and within [0,1]")
    for key in ("corrupted", "auto_corrupted", "review_candidate"):
        if np.asarray(arrays[key]).dtype != np.bool_:
            raise ValueError(f"{key} must have bool dtype")
    if np.any(arrays["auto_corrupted"] & ~arrays["corrupted"]):
        raise ValueError("corrupted must include automatic flags")
    digest = annotation_digest(arrays)
    with h5py.File(path, "r") as f:
        if ANNOTATION_KEY in f:
            existing = f[ANNOTATION_KEY]
            if (
                existing.attrs.get("arrays_sha256") == digest
                and existing.attrs.get("provenance_json") == json.dumps(provenance, sort_keys=True)
                and all(
                    k in existing and np.array_equal(existing[k][:], v) for k, v in arrays.items()
                )
            ):
                return "already_present"
            raise ValueError(f"{path}: {ANNOTATION_KEY} already exists with different annotations")
    if source_identity(path) != identity:
        raise ValueError(f"{path}: source changed since scan; rescan before annotating")
    pending = "quality/_left_wrist_pending"
    with h5py.File(path, "r+") as f:
        if ANNOTATION_KEY in f or pending in f:
            raise ValueError(f"{path}: existing or interrupted annotation; inspect before retrying")
        validate_recording(f, path)
        if len(f["timestamp_ns"]) != n:
            raise ValueError(f"{path}: source frame count changed")
        group = f.create_group(pending)
        group.attrs["schema"] = "left_wrist_corruption_v1"
        group.attrs["frame_index_base"] = 0
        group.attrs["semantics"] = (
            "corrupted includes automatic flags and optional imported positives; "
            "false is not verified clean"
        )
        group.attrs["score_semantics"] = "forest vote, not a calibrated probability"
        group.attrs["arrays_sha256"] = digest
        group.attrs["provenance_json"] = json.dumps(provenance, sort_keys=True)
        for key, values in arrays.items():
            group.create_dataset(key, data=values, compression="gzip")
        f.flush()
        f.move(pending, ANNOTATION_KEY)
        f.flush()
    with h5py.File(path, "r") as f:
        for key, values in arrays.items():
            if not np.array_equal(f[f"{ANNOTATION_KEY}/{key}"][:], values):
                raise RuntimeError(f"{path}: annotation verification failed for {key}")
    return "written"
