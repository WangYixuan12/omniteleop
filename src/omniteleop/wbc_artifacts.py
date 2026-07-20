"""Validate SceneDiff artifacts consumed by the WBC point-cloud pipeline.

This module is deliberately independent of the conversion script.  It owns the
on-disk contracts for tracked masks and DINO region features, while the porter owns
episode discovery, point-cloud construction, and zarr publication.

Validation is strict: arrays are never silently cast or reshaped, mask QC is
recomputed from the pixels, and every artifact is tied to the current raw take and
positions NPZ.  A :class:`ValidatedMaskArtifact` also pins the file stat between the
up-front validation pass and the later episode load, closing the otherwise easy
validate-then-swap hole without retaining all masks in memory.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np

MASK_ARTIFACT_SCHEMA = "dexmate_wbc_masks_v1"
MASK_LEGEND = "0=background 1=box(src) 2=cloth(dst)"
MASK_TRACKER = "sam3.1_multiplex"
MASK_RGB_KEY = "obs/images/head_left_rgb"
MASK_IMAGE_SIZE = 1008
MASK_SEED_FRAME0 = (
    "tracker_render (bilinear+antialias+0.5, IoU-gated vs native seed)"
)
MASK_FRAME0_IOU_MIN = 0.8

DINO_DIM = 1280
DINO_MODEL = "dinov3_vith16plus"
DINO_SOURCE = "original/rgb_scenediff.png [0,1] + video_1 RLEs"
DINO_INPUT_SCALE = "[0,1]"

_TASK_LABELS = (1, 2)
_SHA256_HEX_LEN = 64


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _required_scalar(npz, key: str, path: Path, *, dtype=None):
    if key not in npz.files:
        raise ValueError(f"{path}: missing '{key}'")
    value = np.asarray(npz[key])
    if value.shape != ():
        raise ValueError(f"{path}: {key} shape {value.shape} != scalar")
    if dtype is not None and value.dtype != np.dtype(dtype):
        raise ValueError(f"{path}: {key} dtype {value.dtype} != {np.dtype(dtype)}")
    return value.item()


def _required_array(npz, key: str, path: Path, *, shape: tuple[int, ...], dtype) -> np.ndarray:
    if key not in npz.files:
        raise ValueError(f"{path}: missing '{key}'")
    value = np.asarray(npz[key])
    expected_dtype = np.dtype(dtype)
    if value.shape != shape or value.dtype != expected_dtype:
        raise ValueError(
            f"{path}: {key} {value.shape} {value.dtype} != {shape} {expected_dtype}"
        )
    return value


def _validate_permutation(order: np.ndarray, path: Path) -> None:
    if not np.array_equal(np.sort(order), np.arange(len(order), dtype=order.dtype)):
        raise ValueError(f"{path}: order {order.tolist()} is not a permutation")


def _canonical_stored_path(value: object, artifact_path: Path, field: str) -> Path:
    text = str(value)
    if not text:
        raise ValueError(f"{artifact_path}: {field} is empty")
    return Path(text).expanduser().resolve()


def _raw_rgb_contract(raw_hdf5: Path, *, compute_digest: bool) -> tuple[int, int, int, str | None]:
    digest = hashlib.sha256() if compute_digest else None
    with h5py.File(raw_hdf5, "r") as raw:
        if MASK_RGB_KEY not in raw:
            raise ValueError(f"{raw_hdf5}: missing {MASK_RGB_KEY}")
        rgb = raw[MASK_RGB_KEY]
        if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError(
                f"{raw_hdf5}: {MASK_RGB_KEY} {rgb.shape} {rgb.dtype} != "
                "(T,H,W,3) uint8"
            )
        if rgb.shape[0] < 1 or rgb.shape[1] < 1 or rgb.shape[2] < 1:
            raise ValueError(f"{raw_hdf5}: {MASK_RGB_KEY} has an empty dimension")
        if digest is not None:
            # Hash the logical C-order RGB bytes, independent of HDF5 chunk layout.
            for start in range(0, rgb.shape[0], 32):
                block = np.ascontiguousarray(rgb[start:start + 32])
                digest.update(block.tobytes())
        frames, height, width = map(int, rgb.shape[:3])
    return frames, height, width, None if digest is None else digest.hexdigest()


def _recompute_mask_qc(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = masks.shape[0]
    areas = np.zeros((frames, len(_TASK_LABELS)), dtype=np.int32)
    largest = np.ones((frames, len(_TASK_LABELS)), dtype=np.float32)
    components = np.zeros((frames, len(_TASK_LABELS)), dtype=np.int16)
    for frame_index in range(frames):
        for class_index, label in enumerate(_TASK_LABELS):
            selected = (masks[frame_index] == label).astype(np.uint8)
            area = int(selected.sum())
            areas[frame_index, class_index] = area
            if area == 0:
                continue
            count, _, stats, _ = cv2.connectedComponentsWithStats(
                selected, connectivity=8
            )
            component_areas = stats[1:, cv2.CC_STAT_AREA]
            components[frame_index, class_index] = count - 1
            largest[frame_index, class_index] = (
                float(component_areas.max()) / float(area)
            )
    return areas, largest, components


def _mask_gate_failure(
    masks: np.ndarray,
    areas: np.ndarray,
    seed_areas: np.ndarray,
    frame0_iou: np.ndarray,
) -> str | None:
    if float(frame0_iou.min()) < MASK_FRAME0_IOU_MIN:
        return (
            f"frame0_iou {frame0_iou.tolist()} has a value below "
            f"{MASK_FRAME0_IOU_MIN}"
        )
    height, width = masks.shape[1:]
    for class_index, label in enumerate(_TASK_LABELS):
        floor = max(16.0, 0.01 * int(seed_areas[class_index]))
        early = areas[1:6, class_index]
        if len(early) == 5 and bool((early < floor).all()):
            return (
                f"label {label} is lost throughout frames 1-5 "
                f"({early.tolist()} < {floor:.1f})"
            )
        blown_up = areas[:, class_index] >= 0.8 * height * width
        if len(blown_up) >= 3 and np.convolve(
            blown_up, np.ones(3, dtype=np.int8), mode="valid"
        ).max() >= 3:
            return f"label {label} covers >=80% of three consecutive frames"
    return None


@dataclass(frozen=True)
class ValidatedMaskArtifact:
    """A validated mask NPZ whose identity is pinned until its masks are loaded."""

    path: Path
    frame_count: int
    image_hw: tuple[int, int]
    file_size: int
    mtime_ns: int

    def load_masks(self) -> np.ndarray:
        """Load the mask pixels, rejecting replacement after validation."""
        before = self.path.stat()
        if before.st_size != self.file_size or before.st_mtime_ns != self.mtime_ns:
            raise RuntimeError(
                f"{self.path}: mask artifact changed after validation; restart the port"
            )
        with np.load(self.path, allow_pickle=False) as npz:
            masks = _required_array(
                npz,
                "masks",
                self.path,
                shape=(self.frame_count, *self.image_hw),
                dtype=np.uint8,
            ).copy()
        after = self.path.stat()
        if (
            after.st_size != self.file_size
            or after.st_mtime_ns != self.mtime_ns
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RuntimeError(
                f"{self.path}: mask artifact changed while it was being loaded"
            )
        if masks.min() < 0 or masks.max() > len(_TASK_LABELS):
            raise ValueError(f"{self.path}: masks contain a label outside 0..2")
        return masks


def validate_mask_artifact(
    path: Path | str,
    *,
    source: str,
    episode_number: int,
    raw_hdf5: Path | str,
    positions_npz: Path | str,
) -> ValidatedMaskArtifact:
    """Validate a tracked-mask artifact and return a stat-pinned lazy loader.

    Args:
        path: Mask NPZ produced by ``scene_diff/scripts/build_wbc_masks.py``.
        source: Expected source namespace (``raw`` or ``recovery``).
        episode_number: Episode index inside that namespace.
        raw_hdf5: Current recorder take the masks must describe.
        positions_npz: Current positions artifact that defines task-slot identity.

    Returns:
        A handle that can load mask pixels later without retaining the whole batch.
    """
    path = Path(path).expanduser().resolve()
    raw_hdf5 = Path(raw_hdf5).expanduser().resolve()
    positions_npz = Path(positions_npz).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"mask artifact not found: {path}")
    if not raw_hdf5.exists():
        raise FileNotFoundError(f"raw HDF5 not found: {raw_hdf5}")
    if not positions_npz.exists():
        raise FileNotFoundError(f"positions NPZ not found: {positions_npz}")

    initial_stat = path.stat()
    with np.load(path, allow_pickle=False) as npz:
        has_schema = "artifact_schema" in npz.files
        if has_schema:
            schema = str(_required_scalar(npz, "artifact_schema", path))
            if schema != MASK_ARTIFACT_SCHEMA:
                raise ValueError(
                    f"{path}: artifact_schema {schema!r} != {MASK_ARTIFACT_SCHEMA!r}"
                )
            if "tracked_rgb_sha256" not in npz.files:
                raise ValueError(f"{path}: schema artifact is missing tracked_rgb_sha256")
        elif "tracked_rgb_sha256" in npz.files:
            raise ValueError(
                f"{path}: tracked_rgb_sha256 exists without artifact_schema"
            )

        frame_count, height, width, rgb_digest = _raw_rgb_contract(
            raw_hdf5, compute_digest=has_schema
        )
        masks = _required_array(
            npz,
            "masks",
            path,
            shape=(frame_count, height, width),
            dtype=np.uint8,
        )
        qc_shape = (frame_count, len(_TASK_LABELS))
        stored_areas = _required_array(
            npz, "qc_areas", path, shape=qc_shape, dtype=np.int32
        )
        stored_largest = _required_array(
            npz, "qc_largest_frac", path, shape=qc_shape, dtype=np.float32
        )
        stored_components = _required_array(
            npz, "qc_n_components", path, shape=qc_shape, dtype=np.int16
        )
        order = _required_array(
            npz, "order", path, shape=(len(_TASK_LABELS),), dtype=np.int64
        )
        task_to_obj = _required_array(
            npz, "task_to_obj", path, shape=(len(_TASK_LABELS),), dtype=np.int64
        )
        seed_areas = _required_array(
            npz, "seed_areas", path, shape=(len(_TASK_LABELS),), dtype=np.int64
        )
        frame0_iou = _required_array(
            npz, "frame0_iou", path, shape=(len(_TASK_LABELS),), dtype=np.float32
        )

        expected_scalars = {
            "split": source,
            "episode_number": int(episode_number),
            "rgb_key": MASK_RGB_KEY,
            "n_frames": frame_count,
            "tracker": MASK_TRACKER,
            "image_size": MASK_IMAGE_SIZE,
            "legend": MASK_LEGEND,
            "seed_frame0": MASK_SEED_FRAME0,
        }
        scalar_dtypes = {
            "episode_number": np.int64,
            "n_frames": np.int64,
            "image_size": np.int64,
        }
        for key, expected in expected_scalars.items():
            actual = _required_scalar(npz, key, path, dtype=scalar_dtypes.get(key))
            if actual != expected:
                raise ValueError(f"{path}: {key} {actual!r} != {expected!r}")

        stored_raw = _canonical_stored_path(
            _required_scalar(npz, "raw_hdf5", path), path, "raw_hdf5"
        )
        if stored_raw != raw_hdf5:
            raise ValueError(f"{path}: raw_hdf5 {stored_raw} != {raw_hdf5}")
        stored_positions = _canonical_stored_path(
            _required_scalar(npz, "positions_npz", path), path, "positions_npz"
        )
        if stored_positions != positions_npz:
            raise ValueError(
                f"{path}: positions_npz {stored_positions} != {positions_npz}"
            )
        stored_positions_sha = str(
            _required_scalar(npz, "positions_sha256", path)
        )
        current_positions_sha = _sha256_file(positions_npz)
        if stored_positions_sha != current_positions_sha:
            raise ValueError(
                f"{path}: positions_sha256 does not match {positions_npz}"
            )
        _required_scalar(npz, "created", path)

        if has_schema:
            stored_rgb_sha = str(
                _required_scalar(npz, "tracked_rgb_sha256", path)
            )
            if len(stored_rgb_sha) != _SHA256_HEX_LEN or stored_rgb_sha != rgb_digest:
                raise ValueError(
                    f"{path}: tracked_rgb_sha256 does not match {raw_hdf5}"
                )

        if masks.min() < 0 or masks.max() > len(_TASK_LABELS):
            raise ValueError(f"{path}: masks contain a label outside 0..2")
        _validate_permutation(order, path)
        if (task_to_obj < 0).any() or len(np.unique(task_to_obj)) != len(task_to_obj):
            raise ValueError(f"{path}: task_to_obj {task_to_obj.tolist()} is invalid")
        if (seed_areas <= 0).any():
            raise ValueError(f"{path}: seed_areas {seed_areas.tolist()} must be positive")
        if not np.all(np.isfinite(frame0_iou)) or np.any(
            (frame0_iou < 0.0) | (frame0_iou > 1.0)
        ):
            raise ValueError(f"{path}: frame0_iou {frame0_iou.tolist()} is invalid")
        if not np.all(np.isfinite(stored_largest)) or np.any(
            (stored_largest < 0.0) | (stored_largest > 1.0)
        ):
            raise ValueError(f"{path}: qc_largest_frac is outside [0,1]")

        with np.load(positions_npz, allow_pickle=False) as positions:
            base_order = _required_array(
                positions,
                "order",
                positions_npz,
                shape=(len(_TASK_LABELS),),
                dtype=np.int64,
            )
            base_ids = _required_array(
                positions,
                "scene_diff_obj_ids",
                positions_npz,
                shape=(len(_TASK_LABELS),),
                dtype=np.int64,
            )
        _validate_permutation(base_order, positions_npz)
        if (base_ids < 0).any() or len(np.unique(base_ids)) != len(base_ids):
            raise ValueError(
                f"{positions_npz}: scene_diff_obj_ids {base_ids.tolist()} is invalid"
            )
        if not np.array_equal(order, base_order):
            raise ValueError(f"{path}: order disagrees with {positions_npz}")
        expected_task_to_obj = base_ids[base_order]
        if not np.array_equal(task_to_obj, expected_task_to_obj):
            raise ValueError(
                f"{path}: task_to_obj {task_to_obj.tolist()} != positions mapping "
                f"{expected_task_to_obj.tolist()}"
            )

        areas, largest, components = _recompute_mask_qc(masks)
        if not np.array_equal(stored_areas, areas):
            raise ValueError(f"{path}: qc_areas does not match masks")
        if not np.allclose(stored_largest, largest, rtol=0.0, atol=1e-7):
            raise ValueError(f"{path}: qc_largest_frac does not match masks")
        if not np.array_equal(stored_components, components):
            raise ValueError(f"{path}: qc_n_components does not match masks")
        gate_failure = _mask_gate_failure(masks, areas, seed_areas, frame0_iou)
        if gate_failure is not None:
            raise ValueError(f"{path}: tracker QC gate failed: {gate_failure}")

    validated_stat = path.stat()
    if (
        initial_stat.st_size != validated_stat.st_size
        or initial_stat.st_mtime_ns != validated_stat.st_mtime_ns
    ):
        raise RuntimeError(f"{path}: mask artifact changed during validation")
    if not has_schema:
        logging.warning(
            "%s: legacy mask artifact has no RGB content digest; exact path, slot, "
            "positions, pixels, QC, and tracker contracts were verified. Rebuild or "
            "restamp it with the current builder for full content provenance.",
            path,
        )
    return ValidatedMaskArtifact(
        path=path,
        frame_count=frame_count,
        image_hw=(height, width),
        file_size=validated_stat.st_size,
        mtime_ns=validated_stat.st_mtime_ns,
    )


def load_dino_artifact(
    positions_npz: Path | str,
    sidecar_npz: Path | str,
    *,
    object_nums: int,
) -> np.ndarray:
    """Validate a DINO sidecar and return its task-slot-ordered feature vector."""
    positions_npz = Path(positions_npz).expanduser().resolve()
    sidecar_npz = Path(sidecar_npz).expanduser().resolve()
    if object_nums < 1:
        raise ValueError(f"object_nums must be positive, got {object_nums}")
    if not positions_npz.exists():
        raise FileNotFoundError(f"positions NPZ not found: {positions_npz}")
    if not sidecar_npz.exists():
        raise FileNotFoundError(f"DINO sidecar not found: {sidecar_npz}")

    shape = (object_nums,)
    with np.load(positions_npz, allow_pickle=False) as base:
        base_order = _required_array(
            base, "order", positions_npz, shape=shape, dtype=np.int64
        )
        base_ids = _required_array(
            base, "scene_diff_obj_ids", positions_npz, shape=shape, dtype=np.int64
        )
    _validate_permutation(base_order, positions_npz)
    if (base_ids < 0).any() or len(np.unique(base_ids)) != object_nums:
        raise ValueError(
            f"{positions_npz}: scene_diff_obj_ids {base_ids.tolist()} is invalid"
        )

    with np.load(sidecar_npz, allow_pickle=False) as sidecar:
        feats = _required_array(
            sidecar,
            "dino_region_feats",
            sidecar_npz,
            shape=(object_nums, DINO_DIM),
            dtype=np.float32,
        )
        valid = _required_array(
            sidecar, "dino_valid", sidecar_npz, shape=shape, dtype=np.float32
        )
        stored_order = _required_array(
            sidecar, "order", sidecar_npz, shape=shape, dtype=np.int64
        )
        stored_ids = _required_array(
            sidecar,
            "scene_diff_obj_ids",
            sidecar_npz,
            shape=shape,
            dtype=np.int64,
        )
        base_sha = str(_required_scalar(sidecar, "base_sha256", sidecar_npz))
        model = str(_required_scalar(sidecar, "dino_model", sidecar_npz))
        source = str(_required_scalar(sidecar, "dino_source", sidecar_npz))
        _required_scalar(sidecar, "created", sidecar_npz)

    if base_sha != _sha256_file(positions_npz):
        raise ValueError(
            f"{sidecar_npz}: base_sha256 does not match {positions_npz}"
        )
    if model != DINO_MODEL:
        raise ValueError(f"{sidecar_npz}: dino_model {model!r} != {DINO_MODEL!r}")
    if source != DINO_SOURCE:
        raise ValueError(f"{sidecar_npz}: dino_source {source!r} != {DINO_SOURCE!r}")
    if not np.array_equal(stored_order, base_order):
        raise ValueError(f"{sidecar_npz}: order disagrees with {positions_npz}")
    if not np.array_equal(stored_ids, base_ids):
        raise ValueError(
            f"{sidecar_npz}: scene_diff_obj_ids disagrees with {positions_npz}"
        )
    if not np.array_equal(valid, np.ones(object_nums, dtype=np.float32)):
        raise ValueError(f"{sidecar_npz}: dino_valid {valid.tolist()} is not all ones")
    if not np.all(np.isfinite(feats)):
        raise ValueError(f"{sidecar_npz}: dino_region_feats contains non-finite values")
    zero_rows = np.flatnonzero(np.all(feats == 0.0, axis=1))
    if zero_rows.size:
        raise ValueError(
            f"{sidecar_npz}: dino_region_feats has zero slot(s) {zero_rows.tolist()}"
        )
    return feats[base_order].reshape(-1).copy()
