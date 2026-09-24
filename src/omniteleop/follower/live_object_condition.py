"""Immutable live SceneDiff artifacts and the single task-ordering operation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _readonly(value, dtype, *, shape=None, name="array") -> np.ndarray:
    # Own the backing storage: setting the write flag on a contiguous caller-owned
    # camera array must not freeze the producer's buffer as a side effect.
    array = np.array(value, dtype=dtype, order="C", copy=True)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} {array.shape} != {shape}")
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SceneRequirements:
    """Checkpoint-derived fields required from the live object bootstrap."""

    object_nums: int = 0
    needs_env_state: bool = False
    needs_env_dino: bool = False
    needs_point_mask: bool = False
    needs_bootstrap: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_nums", int(self.object_nums))
        needs = bool(
            self.needs_env_state or self.needs_env_dino or self.needs_point_mask
        )
        object.__setattr__(self, "needs_bootstrap", needs)
        if needs and self.object_nums != 2:
            raise ValueError(
                f"live object conditioning requires exactly 2 task slots, got "
                f"{self.object_nums}"
            )
        if not needs and self.object_nums not in (0, 2):
            raise ValueError("empty scene requirements use object_nums 0 or 2")


@dataclass(frozen=True)
class LiveValidationFrame:
    """One fresh RGB-D frame captured after tracker warm-up for the scene gate."""

    rgb: np.ndarray
    depth_mm: np.ndarray
    world_t_cam: np.ndarray
    head_timestamp_ns: int
    world_frame_epoch: int

    def __post_init__(self) -> None:
        rgb = _readonly(self.rgb, np.uint8, name="validation rgb")
        depth = _readonly(self.depth_mm, np.uint16, name="validation depth")
        if rgb.shape != (*depth.shape, 3):
            raise ValueError("validation RGB/depth shapes disagree")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_mm", depth)
        object.__setattr__(self, "world_t_cam", _readonly(
            self.world_t_cam, np.float32, shape=(4, 4), name="validation world_t_cam"
        ))
        if int(self.head_timestamp_ns) < 0:
            raise ValueError("validation head_timestamp_ns must be non-negative")
        object.__setattr__(self, "head_timestamp_ns", int(self.head_timestamp_ns))
        object.__setattr__(self, "world_frame_epoch", int(self.world_frame_epoch))


@dataclass(frozen=True)
class LiveObjectCondition:
    """Task-slot ordered, episode-scoped condition installed atomically in a bundle."""

    order: np.ndarray
    positions: np.ndarray
    scene_diff_obj_ids: np.ndarray
    seed_task_slots: np.ndarray
    dino_region_feats: np.ndarray | None
    bootstrap_rgb: np.ndarray
    bootstrap_depth_mm: np.ndarray
    intrinsic: np.ndarray
    world_t_cam: np.ndarray
    camera_timestamp_ns: int
    world_frame_epoch: int
    source_artifact: Path
    provenance_hashes: tuple[str, ...] = ()
    dino_model: str | None = None
    dino_source: str | None = None
    dino_input_scale: str | None = None

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions)
        if positions.ndim != 2 or positions.shape[0] < 1:
            raise ValueError(f"positions must be nonempty (K,3), got {positions.shape}")
        count = int(positions.shape[0])
        order = _readonly(self.order, np.int64, shape=(count,), name="order")
        if not np.array_equal(np.sort(order), np.arange(count, dtype=np.int64)):
            raise ValueError(f"order {order.tolist()} is not a {count}-slot permutation")
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "positions", _readonly(
            self.positions, np.float32, shape=(count, 3), name="positions"))
        obj_ids = _readonly(
            self.scene_diff_obj_ids, np.int64, shape=(count,),
            name="scene_diff_obj_ids")
        if (obj_ids < 0).any() or len(np.unique(obj_ids)) != count:
            raise ValueError("SceneDiff object ids must be unique and non-negative")
        object.__setattr__(self, "scene_diff_obj_ids", obj_ids)
        seed = _readonly(self.seed_task_slots, np.uint8, name="seed_task_slots")
        if seed.ndim != 2 or not set(np.unique(seed)).issubset(set(range(count + 1))):
            raise ValueError("seed_task_slots has invalid shape or labels")
        if any(not np.any(seed == label) for label in range(1, count + 1)):
            raise ValueError("seed_task_slots must contain every task label")
        object.__setattr__(self, "seed_task_slots", seed)
        if self.dino_region_feats is not None:
            dino = _readonly(self.dino_region_feats, np.float32,
                             name="dino_region_feats")
            if dino.ndim != 2 or dino.shape[0] != count:
                raise ValueError(f"dino_region_feats has invalid shape {dino.shape}")
            object.__setattr__(self, "dino_region_feats", dino)
            if any(
                getattr(self, name) is None
                for name in ("dino_model", "dino_source", "dino_input_scale")
            ):
                raise ValueError("DINO features require complete model/source/scale metadata")
        elif any(
            getattr(self, name) is not None
            for name in ("dino_model", "dino_source", "dino_input_scale")
        ):
            raise ValueError("DINO metadata is present without DINO features")
        object.__setattr__(self, "bootstrap_rgb", _readonly(
            self.bootstrap_rgb, np.uint8, name="bootstrap_rgb"))
        object.__setattr__(self, "bootstrap_depth_mm", _readonly(
            self.bootstrap_depth_mm, np.uint16, name="bootstrap_depth_mm"))
        if self.bootstrap_rgb.shape != (*self.bootstrap_depth_mm.shape, 3):
            raise ValueError("bootstrap RGB/depth shapes disagree")
        object.__setattr__(self, "intrinsic", _readonly(
            self.intrinsic, np.float32, shape=(3, 3), name="intrinsic"))
        object.__setattr__(self, "world_t_cam", _readonly(
            self.world_t_cam, np.float32, shape=(4, 4), name="world_t_cam"))
        if int(self.camera_timestamp_ns) < 0:
            raise ValueError("camera_timestamp_ns must be non-negative")
        object.__setattr__(self, "camera_timestamp_ns", int(self.camera_timestamp_ns))
        object.__setattr__(self, "world_frame_epoch", int(self.world_frame_epoch))
        object.__setattr__(self, "source_artifact", Path(self.source_artifact))
        for value in self.provenance_hashes:
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"invalid SHA-256 provenance value {value!r}")
        for field_name in ("dino_model", "dino_source", "dino_input_scale"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, str(value))

    @property
    def env_state(self) -> np.ndarray:
        """Flatten task-slot positions for the policy's constant env-state."""
        return self.positions.reshape(-1)

    @property
    def env_dino(self) -> np.ndarray | None:
        """Flatten task-slot DINO descriptors for grounding-token checkpoints."""
        return None if self.dino_region_feats is None else self.dino_region_feats.reshape(-1)


@dataclass(frozen=True)
class LiveObjectArtifacts:
    """Fresh size-slot artifact before the operator confirms task identity."""

    positions_size: np.ndarray
    valid_size: np.ndarray
    scene_diff_obj_ids_size: np.ndarray
    seed_size_slots: np.ndarray
    dino_region_feats_size: np.ndarray | None
    bootstrap_rgb: np.ndarray
    bootstrap_depth_mm: np.ndarray
    intrinsic: np.ndarray
    world_t_cam: np.ndarray
    camera_timestamp_ns: int
    world_frame_epoch: int
    source_artifact: Path
    provenance_hashes: tuple[str, ...] = ()
    dino_model: str | None = None
    dino_source: str | None = None
    dino_input_scale: str | None = None

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_size)
        if positions.ndim != 2 or positions.shape[0] < 1:
            raise ValueError(
                f"positions_size must be nonempty (K,3), got {positions.shape}"
            )
        count = int(positions.shape[0])
        object.__setattr__(self, "positions_size", _readonly(
            self.positions_size, np.float32, shape=(count, 3), name="positions_size"))
        valid = _readonly(self.valid_size, np.float32, shape=(count,), name="valid_size")
        if not np.array_equal(valid, np.ones(count, dtype=np.float32)):
            raise ValueError("every size slot must be valid")
        object.__setattr__(self, "valid_size", valid)
        ids = _readonly(self.scene_diff_obj_ids_size, np.int64, shape=(count,),
                        name="scene_diff_obj_ids_size")
        if (ids < 0).any() or len(np.unique(ids)) != count:
            raise ValueError("SceneDiff object ids must be unique and non-negative")
        object.__setattr__(self, "scene_diff_obj_ids_size", ids)
        seed = _readonly(self.seed_size_slots, np.uint8, name="seed_size_slots")
        if seed.ndim != 2 or not set(np.unique(seed)).issubset(set(range(count + 1))):
            raise ValueError("seed_size_slots has invalid shape or labels")
        if any(not np.any(seed == label) for label in range(1, count + 1)):
            raise ValueError("seed_size_slots must contain every size-slot label")
        object.__setattr__(self, "seed_size_slots", seed)
        if self.dino_region_feats_size is not None:
            dino = _readonly(self.dino_region_feats_size, np.float32,
                             name="dino_region_feats_size")
            if dino.ndim != 2 or dino.shape[0] != count:
                raise ValueError(f"dino_region_feats_size has invalid shape {dino.shape}")
            object.__setattr__(self, "dino_region_feats_size", dino)
            if any(
                getattr(self, name) is None
                for name in ("dino_model", "dino_source", "dino_input_scale")
            ):
                raise ValueError("DINO features require complete model/source/scale metadata")
        elif any(
            getattr(self, name) is not None
            for name in ("dino_model", "dino_source", "dino_input_scale")
        ):
            raise ValueError("DINO metadata is present without DINO features")
        object.__setattr__(self, "bootstrap_rgb", _readonly(
            self.bootstrap_rgb, np.uint8, name="bootstrap_rgb"))
        object.__setattr__(self, "bootstrap_depth_mm", _readonly(
            self.bootstrap_depth_mm, np.uint16, name="bootstrap_depth_mm"))
        if self.bootstrap_rgb.shape != (*self.bootstrap_depth_mm.shape, 3):
            raise ValueError("bootstrap RGB/depth shapes disagree")
        object.__setattr__(self, "intrinsic", _readonly(
            self.intrinsic, np.float32, shape=(3, 3), name="intrinsic"))
        object.__setattr__(self, "world_t_cam", _readonly(
            self.world_t_cam, np.float32, shape=(4, 4), name="world_t_cam"))
        object.__setattr__(self, "camera_timestamp_ns", int(self.camera_timestamp_ns))
        if int(self.camera_timestamp_ns) < 0:
            raise ValueError("camera_timestamp_ns must be non-negative")
        object.__setattr__(self, "world_frame_epoch", int(self.world_frame_epoch))
        object.__setattr__(self, "source_artifact", Path(self.source_artifact))
        for value in self.provenance_hashes:
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"invalid SHA-256 provenance value {value!r}")
        for field_name in ("dino_model", "dino_source", "dino_input_scale"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, str(value))

    def arrange(self, order) -> LiveObjectCondition:
        """Apply the sole size-slot -> task-slot permutation to every representation."""
        order = np.asarray(order, dtype=np.int64)
        count = len(self.positions_size)
        if order.shape != (count,) or not np.array_equal(
            np.sort(order), np.arange(count, dtype=np.int64)
        ):
            raise ValueError(f"order {order.tolist()} is not a {count}-slot permutation")
        seed_task = np.zeros_like(self.seed_size_slots)
        for task_slot, size_slot in enumerate(order):
            seed_task[self.seed_size_slots == int(size_slot) + 1] = task_slot + 1
        dino = (
            None if self.dino_region_feats_size is None
            else self.dino_region_feats_size[order]
        )
        return LiveObjectCondition(
            order=order,
            positions=self.positions_size[order],
            scene_diff_obj_ids=self.scene_diff_obj_ids_size[order],
            seed_task_slots=seed_task,
            dino_region_feats=dino,
            bootstrap_rgb=self.bootstrap_rgb,
            bootstrap_depth_mm=self.bootstrap_depth_mm,
            intrinsic=self.intrinsic,
            world_t_cam=self.world_t_cam,
            camera_timestamp_ns=self.camera_timestamp_ns,
            world_frame_epoch=self.world_frame_epoch,
            source_artifact=self.source_artifact,
            provenance_hashes=self.provenance_hashes,
            dino_model=self.dino_model,
            dino_source=self.dino_source,
            dino_input_scale=self.dino_input_scale,
        )
