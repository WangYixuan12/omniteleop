#!/usr/bin/env python3
"""Live rollout of a ManiFlow 3D (point-cloud) policy on the Vega WBC robot.

The 3D sibling of ``scripts/wbc_policy_rollout.py``. It does NOT fork the hardware loop:
it imports that script and swaps in a ManiFlow policy bundle + observation worker via
``_run_rollout(..., policy_factory=, worker_factory=)``. Engage, reference alignment, the
asynchronous chunk scheduler, the stale-action drop, the 100 Hz WBC tick, the hold
watchdog, recording and shutdown are all THE SAME CODE. A second copy of that loop is
exactly the divergence that gets a robot hurt.

What differs is only the observation:

  LeRobot   head_rgb (+ wrist_rgb) uint8 images  + observation.state (32,)
  ManiFlow  point_cloud (num_points, 6)          + agent_pos         (32,)

The checkpoint's conditioning modes are the only deployment switches. The shared
rollout installs one ordered live object condition when those modes require positions,
DINO descriptors, or tracked masks; no rollout flag can select a different model path.

The cloud is rebuilt live by ``omniteleop.wbc_pointcloud`` -- the SAME module the zarr
converter used -- from one head RGB-D frame, unprojected through ``ZED_K`` and the FK
``world_T_zed`` (engage-origin WORLD frame), cropped to the workspace box, uniformly
pooled and farthest-point-sampled. Every one of those parameters is read from the
``.meta.json`` the training zarr was written with, NOT re-specified here: a train/deploy
mismatch in the frame, the crop or the depth scale silently poisons the policy, so the
numbers have exactly one source of truth. ``--pointcloud-meta`` overrides the path when
the zarr has moved.

Observation cadence: the two ``n_obs_steps`` head frames are grabbed at the dataset
period FIRST, and the cloud construction runs afterwards over both. Building the cloud
inline would stretch the observation spacing past the 0.1 s the policy was trained on.

ManiFlow's consistency-flow training means 1-2 denoising steps are usually enough:
``--num-inference-steps 2`` cuts inference far below the diffusion baseline's ~0.4 s and
lets you shrink ``--policy-interval``.

Environment: the ``dexmate_maniflow`` conda env -- a clone of ``dexmate_lerobot`` (which keeps
the editable dexcontrol/omniteleop installs and pinocchio) plus ``timm`` and a source-built
``pytorch3d``. Neither ``dexmate`` nor ``maniflow`` has both halves. ``fpsample`` (the pinned
downsampler) comes with the clone. See the README "Maniflow" section for the recipe.

Usage::

    python scripts/wbc_maniflow_rollout.py \
      --policy-path ~/ManiFlow_Policy/ManiFlow/data/outputs/<exp>_seed0/checkpoints/latest.ckpt \
      --save-dir ~/Dexmate/data/rollout/maniflow \
      --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5 \
      --num-inference-steps 2 --policy-interval 0.4 --max-seconds 85

    # A conditioned checkpoint activates SceneDiff/tracking automatically:
    python scripts/wbc_maniflow_rollout.py --policy-path <conditioned ckpt> \
      --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5 \
      --num-inference-steps 2

    # replay a recorded take through the identical actuation path (no policy):
    python scripts/wbc_maniflow_rollout.py --replay-episode ~/Dexmate/data/raw_data/episode_1.hdf5 \
      --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5
"""

from __future__ import annotations

import functools
import gc
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from omniteleop.follower.live_object_condition import (
    LiveObjectCondition,
    SceneRequirements,
)
from omniteleop.follower.scenediff_live import load_live_tracker_api
from omniteleop.wbc_artifacts import (
    DINO_INPUT_SCALE,
    DINO_MODEL,
    DINO_SOURCE,
    MASK_ARTIFACT_SCHEMA,
    MASK_FRAME0_IOU_MIN,
    MASK_IMAGE_SIZE,
    MASK_LEGEND,
    MASK_RGB_KEY,
    MASK_SEED_FRAME0,
    MASK_TRACKER,
    POINT_MASK_SELECTION,
    require_scene_consistency,
    validate_live_condition_with_tracker,
)
from omniteleop.wbc_pointcloud import (
    POINT_CHANNELS,
    build_resampled_world_clouds,
    sampler_signature,
    world_t_head_from_state,
)
from omniteleop.wbc_policy_format import ACTION_AXES, STATE_AXES, base_pose_to_mat


def _load_rollout():
    """Import the shared rollout by path (``scripts/`` is not a package)."""
    path = Path(__file__).resolve().parent / "wbc_policy_rollout.py"
    spec = importlib.util.spec_from_file_location("wbc_policy_rollout", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import the shared rollout at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rollout = _load_rollout()

DEFAULT_SAVE_DIR = str(Path("~/Dexmate/data/rollout/maniflow").expanduser())
MIN_TRACKER_FREE_BYTES = 2 * 1024**3
TRACKER_RECOVERY_TOLERANCE_BYTES = 256 * 1024**2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ManiFlowObservation:
    """One raw (unbatched) ManiFlow observation. ``.state`` is what the io_log records."""

    state: np.ndarray         # (32,) float32, frame per the training meta
    point_cloud: np.ndarray   # (num_points, 6) float32, world XYZ + RGB[0,1]
    point_mask: np.ndarray | None = None
    head_timestamp_ns: int = -1
    native_mask: np.ndarray | None = None
    selected_label_counts: np.ndarray | None = None


@dataclass(frozen=True)
class ManiFlowRawFrame:
    """One freshness-gated native RGB-D/state sample before tracker mutation."""

    rgb: np.ndarray
    depth_mm: np.ndarray
    state: np.ndarray
    head_timestamp_ns: int
    world_t_cam: np.ndarray
    world_frame_epoch: int


class ManiFlowBundle:
    """ManiFlow checkpoint + live cloud preprocessing, in ``_PolicyBundle``'s shape."""

    # Compatibility attributes consumed by the family-neutral hardware loop.
    # ManiFlow's observation is the point cloud, so it declares no wrist streams;
    # the loop keys its per-arm freshness gate off this (see _PolicyBundle.wrist_arms).
    wrist_arms: tuple[str, ...] = ()
    use_relative_actions = False

    def __init__(self, policy_path: str, *, device: str | None = None,
                 num_inference_steps: int | None = None, use_ema: bool = True,
                 meta_path: str | None = None,
                 scene_diff_repo: str = "/home/yixuan/scene_diff") -> None:
        import dill  # noqa: PLC0415 -- heavy, GPU path only
        import hydra  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from maniflow.model.common.conditioning import ConditioningSpec  # noqa: PLC0415
        from omegaconf import OmegaConf  # noqa: PLC0415

        OmegaConf.register_new_resolver("eval", eval, replace=True)
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = Path(policy_path)
        if not ckpt.is_file():
            raise ValueError(
                f"--policy-path must be a ManiFlow .ckpt FILE, got {ckpt} "
                "(e.g. .../checkpoints/latest.ckpt)")
        self.checkpoint_path = str(ckpt.expanduser().resolve())
        self.checkpoint_sha256 = _sha256_file(Path(self.checkpoint_path))
        payload = torch.load(ckpt, map_location="cpu", pickle_module=dill, weights_only=False)
        for key in ("cfg", "state_dicts"):
            if key not in payload:
                raise ValueError(f"{ckpt}: not a ManiFlow checkpoint (missing '{key}')")
        cfg = payload["cfg"]

        # Schema gate: the checkpoint must speak the WBC 32/29 point-cloud schema.
        shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
        pc_shape = tuple(shape_meta["obs"]["point_cloud"]["shape"])
        state_dim = int(shape_meta["obs"]["agent_pos"]["shape"][0])
        action_dim = int(shape_meta["action"]["shape"][0])
        if len(pc_shape) != 2 or pc_shape[1] != POINT_CHANNELS:
            raise ValueError(
                f"{ckpt}: point_cloud shape {pc_shape}, expected (N, {POINT_CHANNELS})"
            )
        if state_dim != len(STATE_AXES) or action_dim != len(ACTION_AXES):
            raise ValueError(
                f"{ckpt}: agent_pos={state_dim}/action={action_dim} do not match the WBC "
                f"schema ({len(STATE_AXES)}/{len(ACTION_AXES)}); wrong checkpoint?")
        self.num_points = int(pc_shape[0])

        self.conditioning_spec = ConditioningSpec(
            position_condition_mode=str(
                cfg.policy.get("position_condition_mode", "none") or "none"
            ),
            point_sampling_mode=str(
                cfg.policy.get("point_sampling_mode", "fps") or "fps"
            ),
            mask_channels=bool(cfg.policy.get("mask_channels", False)),
            action_frame_mode=str(
                cfg.policy.get("action_frame_mode", "world") or "world"
            ),
            object_nums=int(cfg.policy.get("object_nums", 2)),
            dino_dim=int(cfg.policy.get("position_dino_dim", 1280)),
        )
        self.position_condition_mode = self.conditioning_spec.position_condition_mode
        self.point_sampling_mode = self.conditioning_spec.point_sampling_mode
        self.mask_channels = self.conditioning_spec.mask_channels
        self.action_frame_mode = self.conditioning_spec.action_frame_mode
        self.use_env_state = self.conditioning_spec.needs_env_state
        self.env_state_dim = (
            self.conditioning_spec.object_nums * 3 if self.use_env_state else 0
        )
        self.scene_requirements = SceneRequirements(
            object_nums=(self.conditioning_spec.object_nums
                         if any((self.conditioning_spec.needs_env_state,
                                 self.conditioning_spec.needs_env_dino,
                                 self.conditioning_spec.needs_point_mask)) else 0),
            needs_env_state=self.conditioning_spec.needs_env_state,
            needs_env_dino=self.conditioning_spec.needs_env_dino,
            needs_point_mask=self.conditioning_spec.needs_point_mask,
        )
        self._condition: LiveObjectCondition | None = None
        self._env_state_tensor = None
        self._env_dino_tensor = None
        self._tracker_owner = None
        self._tracker_session = None
        self._first_window_gate_pending = False
        self._blowup_runs = np.zeros(self.conditioning_spec.object_nums, dtype=np.int64)
        self._last_tracker_timestamp_ns = -1
        self._observation_timestamp_floor_ns = -1
        self._last_inference_trace: dict | None = None
        self._validation_report: dict | None = None
        self._post_capture_t0 = 0.0
        self._tracker_baseline_used_bytes: int | None = None
        self._episode_min_free_bytes: int | None = None
        self._tracker_environment: dict[str, str] | None = None
        self._rollout_frame_capacity: int | None = None
        self._policy_warmed = False
        self.scene_diff_repo = Path(scene_diff_repo).expanduser().resolve()

        # Live preprocessing MUST match training exactly -> read it from the zarr's sidecar.
        self.meta_path = Path(meta_path) if meta_path else self._meta_path_from_cfg(cfg, ckpt)
        self.meta_path = self.meta_path.expanduser().resolve()
        self.pointcloud_meta_sha256 = _sha256_file(self.meta_path)
        meta = self._load_meta(self.meta_path)
        self.env_dino_meta = meta.get("env_dino")
        self.state_frame = meta["state_frame"]
        self.crop_min = tuple(float(v) for v in meta["crop_min"])
        self.crop_max = tuple(float(v) for v in meta["crop_max"])
        # The frame the crop box is expressed in MUST come from the dataset, not be assumed. A
        # checkpoint trained with `crop_frame: "base"` carries its box with the robot; applying
        # the same numbers in world coordinates at rollout crops the scene away entirely once the
        # robot leaves its start pose -- recomputed on the long-horizon takes, mid-episode frames
        # keep 0 points as world against 54k-77k as base, and the resampler then refuses to run.
        self.crop_frame = str(meta.get("crop_frame", "world"))
        if self.crop_frame not in ("world", "base"):
            raise ValueError(f"{self.meta_path}: invalid crop_frame {self.crop_frame!r}")
        self.min_depth = float(meta["min_depth_m"])
        self.max_depth = float(meta["max_depth_m"])
        if (
            len(self.crop_min) != 3
            or len(self.crop_max) != 3
            or not np.all(np.isfinite((*self.crop_min, *self.crop_max)))
            or not np.all(np.asarray(self.crop_min) < np.asarray(self.crop_max))
        ):
            raise ValueError(f"{self.meta_path}: invalid workspace crop")
        if not 0.0 <= self.min_depth < self.max_depth:
            raise ValueError(f"{self.meta_path}: invalid depth bounds")
        self.dataset_fps = float(meta.get("fps", 10))
        # The pre-FPS uniform pool is part of the trained point distribution too: default
        # to the porter's recorded value, not a CLI constant.
        self.pool_size = int(meta["pool_size"])
        if self.pool_size < self.num_points:
            raise ValueError(
                f"metadata pool_size {self.pool_size} < num_points {self.num_points}"
            )
        self._rng_seed = int(meta["seed"])
        self._rng = np.random.default_rng(self._rng_seed)

        self.intrinsic = np.asarray(meta["intrinsic"], dtype=np.float64)
        self.head_hw = tuple(int(v) for v in meta["head_image_hw"])
        if self.intrinsic.shape != (3, 3):
            raise ValueError(f"ZED_K must be (3,3), got {self.intrinsic.shape}")
        if self.intrinsic[0, 0] <= 0 or self.intrinsic[1, 1] <= 0:
            raise ValueError(f"{self.meta_path}: intrinsic focal lengths must be positive")

        self.policy = hydra.utils.instantiate(cfg.policy)
        state_dicts = payload["state_dicts"]
        which = "ema_model" if (use_ema and "ema_model" in state_dicts) else "model"
        if which not in state_dicts:
            raise ValueError(f"{ckpt}: no '{which}' in state_dicts {sorted(state_dicts)}")
        self.policy.load_state_dict(state_dicts[which])
        if self.policy.conditioning_spec != self.conditioning_spec:
            raise ValueError("checkpoint cfg and instantiated policy conditioning specs differ")
        self.policy.validate_inference_contract()
        self.policy.to(self.device).eval()
        self.weights = which

        if num_inference_steps is not None:
            if num_inference_steps < 1:
                raise ValueError(f"--num-inference-steps must be >= 1, got {num_inference_steps}")
            self.policy.num_inference_steps = int(num_inference_steps)
        self.num_inference_steps = int(self.policy.num_inference_steps)

        self.n_obs_steps = int(self.policy.n_obs_steps)
        self.n_action_steps = int(self.policy.n_action_steps)
        if self.n_obs_steps < 1 or self.n_action_steps < 1:
            raise ValueError(f"bad checkpoint config: n_obs_steps={self.n_obs_steps} "
                             f"n_action_steps={self.n_action_steps}")
        if not self.scene_requirements.needs_bootstrap:
            self._warm_policy_for_memory_baseline(None)

    @staticmethod
    def _meta_path_from_cfg(cfg, ckpt: Path) -> Path:
        try:
            zarr_path = Path(str(cfg.robotwin_task.dataset.zarr_path))
        except Exception as exc:  # a non-WBC ManiFlow cfg
            raise ValueError(
                f"{ckpt}: cfg has no robotwin_task.dataset.zarr_path; pass --pointcloud-meta"
            ) from exc
        return zarr_path.with_suffix(".meta.json")

    def _load_meta(self, path: Path) -> dict:
        """The training zarr's ``.meta.json``: the ONE source of the live preprocessing."""
        if not path.exists():
            raise FileNotFoundError(
                f"point-cloud meta not found: {path}. It records the crop/depth/frame the "
                "checkpoint was trained with; deploying without it would guess. Pass "
                "--pointcloud-meta if the zarr moved.")
        meta = json.loads(path.read_text())
        for key in (
            "schema", "state_frame", "crop_min", "crop_max", "crop_frame", "min_depth_m",
            "max_depth_m", "num_points", "point_frame", "sampler", "pool_size",
            "seed", "rng_scheme", "head_image_hw", "intrinsic", "rgb_dtype",
            "depth_dtype", "depth_scale_m", "fps",
        ):
            if key not in meta:
                raise ValueError(
                    f"{path}: missing '{key}' (regenerate with port_wbc_mobile_zarr.py)"
                )
        if meta["schema"] != "wbc_maniflow_pointcloud_v3":
            raise ValueError(
                f"{path}: schema {meta['schema']!r} is not "
                "'wbc_maniflow_pointcloud_v3'; rebuild the zarr to record native "
                "camera calibration and RNG contracts"
            )
        if meta["rgb_dtype"] != "uint8" or meta["depth_dtype"] != "uint16":
            raise ValueError(f"{path}: unsupported RGB-D dtypes")
        if float(meta["depth_scale_m"]) != 0.001:
            raise ValueError(f"{path}: depth_scale_m must be 0.001")
        if meta["rng_scheme"] != "numpy.default_rng(seed); one choice draw per frame":
            raise ValueError(f"{path}: unsupported RNG scheme {meta['rng_scheme']!r}")
        if meta.get("point_channels") != ["x", "y", "z", "r", "g", "b"]:
            raise ValueError(f"{path}: invalid point channel order")
        if not str(meta.get("action_frame", "")).startswith("world"):
            raise ValueError(f"{path}: action_frame is not engage-origin world")
        if meta.get("state_axes") != list(STATE_AXES):
            raise ValueError(f"{path}: state axis order differs from the WBC contract")
        if meta.get("action_axes") != list(ACTION_AXES):
            raise ValueError(f"{path}: action axis order differs from the WBC contract")
        if not np.isfinite(float(meta["fps"])) or float(meta["fps"]) <= 0:
            raise ValueError(f"{path}: fps must be finite and positive")
        intrinsic = np.asarray(meta["intrinsic"])
        if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
            raise ValueError(f"{path}: invalid intrinsic {intrinsic.shape}")
        head_hw = tuple(meta["head_image_hw"])
        if len(head_hw) != 2 or any(int(v) < 1 for v in head_hw):
            raise ValueError(f"{path}: invalid head_image_hw {head_hw}")
        # The downsampler must be the SAME one, configured the SAME way, or the live cloud
        # is a different point set than the policy trained on -- and nothing downstream
        # would notice. fpsample's start_idx defaults to a random seed point, so an
        # unpinned build silently skews every observation.
        ours = sampler_signature()
        for key in ("sampler", "kdline_h", "start_idx"):
            if meta.get(key) != ours[key]:
                raise ValueError(
                    f"{path}: downsampler {key}={meta.get(key)!r} but this build uses "
                    f"{ours[key]!r} -- the live cloud would not match the training cloud")
        if meta.get("fpsample_version") != ours["fpsample_version"]:
            raise ValueError(
                f"{path}: zarr used fpsample {meta.get('fpsample_version')}, this env "
                f"has {ours['fpsample_version']}; rebuild or use the pinned environment"
            )
        if int(meta["num_points"]) != self.num_points:
            raise ValueError(
                f"{path}: num_points {meta['num_points']} != the checkpoint's {self.num_points} "
                "-- this meta belongs to a different zarr")
        if not str(meta["point_frame"]).startswith("world"):
            raise ValueError(f"{path}: point_frame {meta['point_frame']!r} is not world-frame")
        if meta["state_frame"] not in ("base", "world"):
            raise ValueError(f"{path}: unexpected state_frame {meta['state_frame']!r}")
        if self.conditioning_spec.needs_point_mask:
            point_mask = meta.get("point_mask")
            expected = {
                "artifact_schema": MASK_ARTIFACT_SCHEMA,
                "legend": MASK_LEGEND,
                "tracker": MASK_TRACKER,
                "builder": "scene_diff/scripts/build_wbc_masks.py",
                "source_rgb_key": MASK_RGB_KEY,
                "tracker_image_size": MASK_IMAGE_SIZE,
                "seed_frame0": MASK_SEED_FRAME0,
                "frame0_iou_min": MASK_FRAME0_IOU_MIN,
                "selection": POINT_MASK_SELECTION,
            }
            if not isinstance(point_mask, dict):
                raise ValueError(
                    f"{path}: checkpoint needs point masks but metadata has none"
                )
            mismatched = {
                key: (point_mask.get(key), value)
                for key, value in expected.items()
                if point_mask.get(key) != value
            }
            if mismatched:
                raise ValueError(
                    f"{path}: point-mask tracker/source contract mismatches {mismatched}"
                )
        if self.conditioning_spec.needs_env_dino:
            dino = meta.get("env_dino", {})
            expected = {
                "dino_input_scale": DINO_INPUT_SCALE,
                "dino_model": DINO_MODEL,
                "dino_source": DINO_SOURCE,
            }
            mismatched = {
                key: (dino.get(key), value)
                for key, value in expected.items()
                if dino.get(key) != value
            }
            if mismatched:
                raise ValueError(
                    f"{path}: checkpoint needs exact DINO metadata; mismatches {mismatched}"
                )
        return meta

    def describe(self) -> str:
        """Return a one-line deployment contract summary."""
        return (f"ManiFlow {type(self.policy).__name__} ({self.weights}) on {self.device}; "
                f"point_cloud {self.num_points}x{POINT_CHANNELS} world-frame, crop "
                f"{list(self.crop_min)}..{list(self.crop_max)}; state_frame={self.state_frame}; "
                f"position_condition={self.position_condition_mode}, "
                f"sampling={self.point_sampling_mode}, mask_channels={self.mask_channels}, "
                f"action_frame={self.action_frame_mode}; "
                f"n_obs_steps={self.n_obs_steps} n_action_steps={self.n_action_steps} "
                f"num_inference_steps={self.num_inference_steps}; meta={self.meta_path}")

    def reset(self) -> None:
        """Reset policy and episode-scoped condition state after complete teardown."""
        if self._tracker_session is not None or self._tracker_owner is not None:
            raise RuntimeError("previous episode tracker runtime was not finished")
        self.policy.reset()
        self._condition = None
        self._env_state_tensor = None
        self._env_dino_tensor = None
        self._first_window_gate_pending = False
        self._blowup_runs.fill(0)
        self._last_tracker_timestamp_ns = -1
        self._observation_timestamp_floor_ns = -1
        self._last_inference_trace = None
        self._validation_report = None
        self._tracker_baseline_used_bytes = None
        self._episode_min_free_bytes = None
        self._tracker_environment = None
        self._rollout_frame_capacity = None

    def _check_live_object_condition(self, condition: LiveObjectCondition) -> None:
        requirements = self.scene_requirements
        if not requirements.needs_bootstrap:
            raise RuntimeError("unconditioned checkpoint cannot install a live condition")
        if len(condition.positions) != requirements.object_nums:
            raise ValueError("live condition object count does not match checkpoint")
        if condition.bootstrap_rgb.shape[:2] != self.head_hw:
            raise ValueError(
                f"bootstrap image {condition.bootstrap_rgb.shape[:2]} != "
                f"training metadata {self.head_hw}"
            )
        if not np.array_equal(condition.intrinsic, self.intrinsic.astype(np.float32)):
            raise ValueError("bootstrap intrinsic differs from training metadata")
        if requirements.needs_env_dino:
            if condition.dino_region_feats is None:
                raise ValueError("grounding_tokens requires DINO region features")
            expected = (requirements.object_nums, self.conditioning_spec.dino_dim)
            if condition.dino_region_feats.shape != expected:
                raise ValueError(
                    f"DINO features {condition.dino_region_feats.shape} != {expected}"
                )
            expected_metadata = {
                "dino_model": DINO_MODEL,
                "dino_source": DINO_SOURCE,
                "dino_input_scale": DINO_INPUT_SCALE,
            }
            actual_metadata = {
                key: getattr(condition, key) for key in expected_metadata
            }
            if actual_metadata != expected_metadata:
                raise ValueError(
                    f"live DINO provenance {actual_metadata} != {expected_metadata}"
                )

    def _warm_policy_for_memory_baseline(
        self, condition: LiveObjectCondition | None
    ) -> None:
        """Run one shape-faithful inference without consuming deployment RNG state."""
        if not self._torch.cuda.is_available():
            return
        if self._torch.is_autocast_enabled("cuda"):
            raise RuntimeError("CUDA autocast leaked into ManiFlow policy warm-up")
        requirements = self.scene_requirements
        if requirements.needs_bootstrap:
            if condition is None:
                raise RuntimeError("conditioned policy warm-up requires a live condition")
            self._check_live_object_condition(condition)

        state = np.zeros((1, self.n_obs_steps, len(STATE_AXES)), np.float32)
        # Valid identity 6-D rotations for left/right/head keep MoF frame transforms
        # nonsingular while exercising the exact deployed tensor shapes.
        for first_column, second_column in ((3, 7), (13, 17), (23, 27)):
            state[..., first_column] = 1.0
            state[..., second_column] = 1.0
        observation = {
            "point_cloud": self._torch.zeros(
                (1, self.n_obs_steps, self.num_points, POINT_CHANNELS),
                dtype=self._torch.float32,
                device=self.device,
            ),
            "agent_pos": self._torch.from_numpy(state).to(self.device),
        }
        if requirements.needs_point_mask:
            observation["point_mask"] = self._torch.zeros(
                (1, self.n_obs_steps, self.num_points),
                dtype=self._torch.int8,
                device=self.device,
            )
        if requirements.needs_env_state:
            observation["env_state"] = self._torch.from_numpy(
                condition.env_state.copy()[None]
            ).to(self.device)
        if requirements.needs_env_dino:
            observation["env_dino"] = self._torch.from_numpy(
                condition.env_dino.copy()[None]
            ).to(self.device)

        device = self._torch.device(self.device)
        cuda_index = (
            self._torch.cuda.current_device()
            if device.index is None else device.index
        )
        try:
            with self._torch.random.fork_rng(devices=[cuda_index]):
                self._torch.cuda.synchronize()
                with self._torch.inference_mode():
                    result = self.policy.predict_action(observation)
                if self._torch.is_autocast_enabled("cuda"):
                    raise RuntimeError("ManiFlow policy warm-up leaked CUDA autocast")
                action = result.get("action")
                expected = (1, self.n_action_steps, len(ACTION_AXES))
                if action is None or tuple(action.shape) != expected:
                    shape = None if action is None else tuple(action.shape)
                    raise ValueError(
                        f"policy warm-up returned action {shape}, expected {expected}"
                    )
                if not bool(self._torch.isfinite(action).all().item()):
                    raise ValueError("policy warm-up returned non-finite actions")
                self._torch.cuda.synchronize()
        finally:
            self.policy.reset()
            observation.clear()
            if "result" in locals():
                del result
            gc.collect()
            self._torch.cuda.empty_cache()
            self._torch.cuda.synchronize()
        self._policy_warmed = True

    def install_live_object_condition(self, condition: LiveObjectCondition) -> None:
        """Validate and install all checkpoint-required object fields atomically."""
        self._check_live_object_condition(condition)
        requirements = self.scene_requirements
        # Materialize every device tensor before publishing any field on the bundle.
        # An allocation/copy failure must leave the previous episode state untouched.
        env_state_tensor = (
            self._torch.from_numpy(condition.env_state.copy()[None]).to(self.device)
            if requirements.needs_env_state else None
        )
        env_dino_tensor = (
            self._torch.from_numpy(condition.env_dino.copy()[None]).to(self.device)
            if requirements.needs_env_dino else None
        )
        tracker_timestamp_ns = condition.camera_timestamp_ns
        observation_timestamp_floor_ns = max(
            condition.camera_timestamp_ns,
            -1 if self._validation_report is None else int(
                self._validation_report["head_timestamp_ns"]
            ),
        )
        self._condition = condition
        self._env_state_tensor = env_state_tensor
        self._env_dino_tensor = env_dino_tensor
        self._last_tracker_timestamp_ns = tracker_timestamp_ns
        self._observation_timestamp_floor_ns = observation_timestamp_floor_ns

    def _tracker_api(self):
        return load_live_tracker_api(self.scene_diff_repo)

    def _record_device_memory(self, *, synchronize: bool = False) -> tuple[int, int] | None:
        if not self._torch.cuda.is_available():
            return None
        if synchronize:
            self._torch.cuda.synchronize()
        free_bytes, total_bytes = self._torch.cuda.mem_get_info()
        reserved_bytes = int(self._torch.cuda.memory_reserved())
        peak_reserved_bytes = int(self._torch.cuda.max_memory_reserved())
        # mem_get_info observes the post-kernel instant. Combine its estimate of
        # non-PyTorch usage with PyTorch's peak-reserved counter so transient SAM/MoF
        # peaks are not missed after their kernels have completed.
        external_used_bytes = max(
            0, int(total_bytes) - int(free_bytes) - reserved_bytes
        )
        estimated_peak_free_bytes = max(
            0, int(total_bytes) - external_used_bytes - peak_reserved_bytes
        )
        observed_or_peak_free = min(int(free_bytes), estimated_peak_free_bytes)
        self._episode_min_free_bytes = (
            observed_or_peak_free if self._episode_min_free_bytes is None
            else min(self._episode_min_free_bytes, observed_or_peak_free)
        )
        return int(free_bytes), int(total_bytes)

    def validate_live_object_condition(
        self,
        condition: LiveObjectCondition,
        *,
        capture_validation_frame,
        stage_dir: Path,
    ) -> dict:
        """Warm SAM if needed, then validate on a separately captured fresh RGB-D frame."""
        self._check_live_object_condition(condition)
        api = self._tracker_api()
        self._tracker_environment = api.probe_live_dependencies()
        self._warm_policy_for_memory_baseline(condition)
        baseline = self._record_device_memory(synchronize=True)
        if baseline is not None:
            self._tracker_baseline_used_bytes = baseline[1] - baseline[0]
            self._torch.cuda.reset_peak_memory_stats()
        needs_masks = self.scene_requirements.needs_point_mask
        owner = api.Sam31VosTracker(do_compile=needs_masks)
        validation_error: BaseException | None = None
        try:
            minimum_capacity = max(16, owner.max_obj_ptrs_in_encoder)
            if needs_masks:
                # Compilation is non-evidentiary. Its repeated frame never enters either
                # the validation session or the later rollout session.
                with owner.start(
                    condition.bootstrap_rgb,
                    condition.seed_task_slots,
                    frame_capacity=minimum_capacity,
                ) as compile_session:
                    compile_session.step_many(condition.bootstrap_rgb[None])

            fresh = capture_validation_frame()
            serializable = validate_live_condition_with_tracker(
                owner,
                condition,
                fresh,
                min_depth_m=self.min_depth,
                max_depth_m=self.max_depth,
                stage_dir=stage_dir,
                tracker_compile=needs_masks,
            )
            self._validation_report = serializable
            self._record_device_memory(synchronize=True)
            if (
                self._episode_min_free_bytes is not None
                and self._episode_min_free_bytes < MIN_TRACKER_FREE_BYTES
            ):
                raise RuntimeError(
                    "policy/tracker peak leaves only "
                    f"{self._episode_min_free_bytes / 2**30:.2f} GiB free; "
                    "deployment requires at least 2 GiB"
                )
            if needs_masks:
                self._tracker_owner = owner
                owner = None
            return serializable
        except BaseException as exc:
            validation_error = exc
            raise
        finally:
            if owner is not None:
                cleanup_errors: list[BaseException] = []
                try:
                    owner.close()
                except BaseException as close_error:
                    cleanup_errors.append(close_error)
                del owner
                gc.collect()
                if self._torch.cuda.is_available():
                    try:
                        self._torch.cuda.synchronize()
                        self._torch.cuda.empty_cache()
                        self._torch.cuda.synchronize()
                        if (
                            not needs_masks
                            and self._tracker_baseline_used_bytes is not None
                        ):
                            free_bytes, total_bytes = self._torch.cuda.mem_get_info()
                            recovered_used = int(total_bytes - free_bytes)
                            excess = recovered_used - self._tracker_baseline_used_bytes
                            if excess > TRACKER_RECOVERY_TOLERANCE_BYTES:
                                cleanup_errors.append(RuntimeError(
                                    "post-tracker device use remained "
                                    f"{excess / 2**20:.0f} MiB above the policy-only "
                                    "baseline (limit 256 MiB)"
                                ))
                            report = {
                                "schema": "sam31_tracker_cleanup_v1",
                                "policy_only_baseline_used_bytes": (
                                    self._tracker_baseline_used_bytes
                                ),
                                "minimum_free_bytes": self._episode_min_free_bytes,
                                "recovered_used_bytes": recovered_used,
                                "recovery_excess_bytes": excess,
                                "minimum_free_passed": (
                                    self._episode_min_free_bytes is not None
                                    and self._episode_min_free_bytes
                                    >= MIN_TRACKER_FREE_BYTES
                                ),
                                "recovery_passed": (
                                    excess <= TRACKER_RECOVERY_TOLERANCE_BYTES
                                ),
                                "cleanup_errors": [
                                    f"{type(error).__name__}: {error}"
                                    for error in cleanup_errors
                                ],
                            }
                            with (
                                Path(stage_dir) / "tracker_cleanup_report.json"
                            ).open("x") as stream:
                                json.dump(report, stream, indent=2)
                            self._tracker_baseline_used_bytes = None
                            self._episode_min_free_bytes = None
                    except BaseException as cleanup_error:
                        cleanup_errors.append(cleanup_error)
                if not needs_masks:
                    self._tracker_baseline_used_bytes = None
                    self._episode_min_free_bytes = None
                if cleanup_errors:
                    if validation_error is not None:
                        for cleanup_error in cleanup_errors:
                            validation_error.add_note(
                                "tracker cleanup also failed: "
                                f"{type(cleanup_error).__name__}: {cleanup_error}"
                            )
                    else:
                        primary = cleanup_errors[0]
                        for cleanup_error in cleanup_errors[1:]:
                            primary.add_note(
                                "additional tracker cleanup failure: "
                                f"{type(cleanup_error).__name__}: {cleanup_error}"
                            )
                        raise primary

    @property
    def initial_head_timestamp_ns(self) -> int:
        """Freshness floor excluding bootstrap/validation frames from live history."""
        return int(self._observation_timestamp_floor_ns)

    def start_episode_runtime(
        self,
        *,
        max_seconds: float,
        policy_interval: float,
        dataset_dt: float,
    ) -> None:
        """Seed a clean rollout tracker session after bootstrap validation."""
        if self.scene_requirements.needs_bootstrap and self._condition is None:
            raise RuntimeError("live object condition was not installed")
        if not self.scene_requirements.needs_point_mask:
            return
        if self._tracker_owner is None:
            raise RuntimeError("compiled tracker owner was not retained after validation")
        if max_seconds > 0:
            windows = int(np.ceil(max_seconds / policy_interval)) + 2
            capacity = 1 + windows * self.n_obs_steps
        else:
            capacity = 10_000
        capacity = max(
            capacity, self._tracker_owner.max_obj_ptrs_in_encoder
        )
        self._rollout_frame_capacity = capacity
        self._tracker_session = self._tracker_owner.start(
            self._condition.bootstrap_rgb,
            self._condition.seed_task_slots,
            frame_capacity=capacity,
        )
        self._first_window_gate_pending = True

    def finish_episode_runtime(self) -> None:
        """Release tracker state/model and enforce the device-memory recovery gate."""
        errors: list[BaseException] = []
        baseline_used = self._tracker_baseline_used_bytes
        minimum_free = self._episode_min_free_bytes
        recovered_used: int | None = None
        recovery_excess: int | None = None
        if self._tracker_session is not None:
            try:
                self._tracker_session.close()
            except BaseException as exc:  # cleanup must continue to model release
                errors.append(exc)
            self._tracker_session = None
        if self._tracker_owner is not None:
            try:
                self._tracker_owner.close()
            except BaseException as exc:
                errors.append(exc)
            self._tracker_owner = None
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
            self._torch.cuda.empty_cache()
            self._torch.cuda.synchronize()
            free_bytes, total_bytes = self._torch.cuda.mem_get_info()
            if (
                self._episode_min_free_bytes is not None
                and self._episode_min_free_bytes < MIN_TRACKER_FREE_BYTES
            ):
                errors.append(RuntimeError(
                    f"episode tracker free-memory minimum was "
                    f"{self._episode_min_free_bytes / 2**30:.2f} GiB (<2 GiB)"
                ))
            if self._tracker_baseline_used_bytes is not None:
                recovered_used = int(total_bytes - free_bytes)
                excess = recovered_used - self._tracker_baseline_used_bytes
                recovery_excess = excess
                if excess > TRACKER_RECOVERY_TOLERANCE_BYTES:
                    errors.append(RuntimeError(
                        f"post-tracker device use remained {excess / 2**20:.0f} MiB "
                        "above the episode's policy-only baseline (limit 256 MiB)"
                    ))
        if self._condition is not None and baseline_used is not None:
            report = {
                "schema": "sam31_tracker_cleanup_v1",
                "policy_only_baseline_used_bytes": baseline_used,
                "minimum_free_bytes": minimum_free,
                "recovered_used_bytes": recovered_used,
                "recovery_excess_bytes": recovery_excess,
                "minimum_free_passed": (
                    minimum_free is not None and minimum_free >= MIN_TRACKER_FREE_BYTES
                ),
                "recovery_passed": (
                    recovery_excess is not None
                    and recovery_excess <= TRACKER_RECOVERY_TOLERANCE_BYTES
                ),
                "cleanup_errors": [
                    f"{type(error).__name__}: {error}" for error in errors
                ],
            }
            report_path = (
                self._condition.source_artifact.parent / "tracker_cleanup_report.json"
            )
            try:
                with report_path.open("x") as stream:
                    json.dump(report, stream, indent=2)
            except BaseException as exc:
                errors.append(exc)
        self._tracker_baseline_used_bytes = None
        self._episode_min_free_bytes = None
        if errors:
            primary = errors[0]
            for extra in errors[1:]:
                primary.add_note(
                    f"additional tracker cleanup failure: {type(extra).__name__}: {extra}"
                )
            raise primary

    def inference_metadata(self) -> dict[str, np.ndarray]:
        """Return checkpoint/preprocessing/MoF metadata stored once per episode."""
        spec = self.conditioning_spec
        metadata = {
            "checkpoint_path": np.asarray(str(self.checkpoint_path).encode("utf-8")),
            "checkpoint_sha256": np.asarray(
                self.checkpoint_sha256.encode()
            ),
            "weights": np.asarray(self.weights.encode()),
            "position_condition_mode": np.asarray(spec.position_condition_mode.encode()),
            "point_sampling_mode": np.asarray(spec.point_sampling_mode.encode()),
            "mask_channels": np.asarray(spec.mask_channels),
            "action_frame_mode": np.asarray(spec.action_frame_mode.encode()),
            "pointcloud_meta": np.asarray(str(self.meta_path).encode("utf-8")),
            "pointcloud_meta_sha256": np.asarray(
                self.pointcloud_meta_sha256.encode()
            ),
            "num_inference_steps": np.asarray(self.num_inference_steps, np.int64),
            "policy_warmed_before_tracker": np.asarray(self._policy_warmed),
            "n_obs_steps": np.asarray(self.n_obs_steps, np.int64),
            "n_action_steps": np.asarray(self.n_action_steps, np.int64),
            "dataset_fps": np.asarray(self.dataset_fps, np.float64),
        }
        if spec.action_frame_mode == "mof":
            metadata.update({
                "mof_enabled_experts": np.asarray(
                    [name.encode() for name in self.policy.mof_enabled_experts]
                ),
                "mof_canonical": np.asarray(self.policy.mof_canonical.encode()),
                "mof_router_mode": np.asarray(self.policy.mof_router_mode.encode()),
            })
        return metadata

    def live_condition_metadata(self) -> dict[str, np.ndarray] | None:
        """Episode-specific immutable condition and validation provenance."""
        if self._condition is None:
            return None
        condition = self._condition
        metadata: dict[str, np.ndarray] = {
            "order": condition.order,
            "positions": condition.positions,
            "scene_diff_obj_ids": condition.scene_diff_obj_ids,
            "camera_timestamp_ns": np.asarray(condition.camera_timestamp_ns, np.int64),
            "world_frame_epoch": np.asarray(condition.world_frame_epoch, np.int64),
            "source_artifact": np.asarray(str(condition.source_artifact).encode()),
            "provenance_hashes": np.asarray(
                [value.encode() for value in condition.provenance_hashes]
            ),
            "seed_task_slots_sha256": np.asarray(
                hashlib.sha256(
                    condition.seed_task_slots.tobytes(order="C")
                ).hexdigest().encode()
            ),
        }
        if condition.dino_region_feats is not None:
            metadata.update({
                "dino_model": np.asarray(str(condition.dino_model).encode()),
                "dino_source": np.asarray(str(condition.dino_source).encode()),
                "dino_input_scale": np.asarray(str(condition.dino_input_scale).encode()),
                "dino_feature_sha256": np.asarray(
                    hashlib.sha256(
                        condition.dino_region_feats.tobytes(order="C")
                    ).hexdigest().encode()
                ),
            })
        if self._validation_report is not None:
            metadata["validation"] = {
                key: np.asarray(value) for key, value in self._validation_report.items()
            }
        if self._tracker_environment is not None:
            metadata["tracker_environment"] = {
                key: np.asarray(value.encode())
                for key, value in self._tracker_environment.items()
            }
            metadata["tracker_compile"] = np.asarray(
                self.scene_requirements.needs_point_mask
            )
        if self._rollout_frame_capacity is not None:
            metadata["tracker_frame_capacity"] = np.asarray(
                self._rollout_frame_capacity, np.int64
            )
        return metadata

    def build_observations(
        self, raws: list[ManiFlowRawFrame]
    ) -> list[ManiFlowObservation]:
        """Advance masks transactionally, then build clouds through the porter helper."""
        if not raws:
            raise ValueError("no raw head frames to build observations from")
        self._post_capture_t0 = time.perf_counter()
        timestamps = np.asarray([raw.head_timestamp_ns for raw in raws], dtype=np.int64)
        freshness_floor = max(
            self._last_tracker_timestamp_ns, self._observation_timestamp_floor_ns
        )
        if np.any(np.diff(timestamps) <= 0) or timestamps[0] <= freshness_floor:
            raise RuntimeError("policy-consumed head timestamps are not strictly increasing")
        if self._condition is not None and any(
            raw.world_frame_epoch != self._condition.world_frame_epoch for raw in raws
        ):
            raise RuntimeError("world-frame epoch changed after object bootstrap")
        for raw in raws:
            if raw.rgb.shape != (*self.head_hw, 3) or raw.depth_mm.shape != self.head_hw:
                raise ValueError(
                    f"head RGB-D {raw.rgb.shape}/{raw.depth_mm.shape} != "
                    f"training metadata {(*self.head_hw, 3)}/{self.head_hw}"
                )
            if raw.rgb.dtype != np.uint8 or raw.depth_mm.dtype != np.uint16:
                raise ValueError(
                    f"head RGB-D must be uint8/uint16, got "
                    f"{raw.rgb.dtype}/{raw.depth_mm.dtype}"
                )
            if (
                raw.state.shape != (len(STATE_AXES),)
                or raw.state.dtype != np.float32
                or not np.all(np.isfinite(raw.state))
            ):
                raise ValueError("raw policy state has the wrong shape/dtype or is non-finite")
            if (
                raw.world_t_cam.shape != (4, 4)
                or raw.world_t_cam.dtype != np.float32
                or not np.all(np.isfinite(raw.world_t_cam))
            ):
                raise ValueError("raw world_t_cam is invalid")
        rgb = np.stack([raw.rgb for raw in raws])
        depth = np.stack([raw.depth_mm for raw in raws])
        states = np.stack([raw.state for raw in raws]).astype(np.float32, copy=False)
        world_t_cams = np.stack([raw.world_t_cam for raw in raws])

        tracker_prepare_s = 0.0
        tracker_propagate_s = 0.0
        native_masks = None
        if self.scene_requirements.needs_point_mask:
            if self._tracker_session is None:
                raise RuntimeError("mask-consuming checkpoint has no rollout tracker session")
            tracker_t0 = time.perf_counter()
            prepared = self._tracker_owner.prepare_many(rgb)
            tracker_prepare_s = time.perf_counter() - tracker_t0
            tracker_t0 = time.perf_counter()
            native_masks = self._tracker_session.step_many(prepared)
            tracker_propagate_s = time.perf_counter() - tracker_t0
            image_area = self.head_hw[0] * self.head_hw[1]
            for frame_mask in native_masks:
                for slot in range(self.conditioning_spec.object_nums):
                    area = int((frame_mask == slot + 1).sum())
                    self._blowup_runs[slot] = (
                        self._blowup_runs[slot] + 1
                        if area >= 0.8 * image_area else 0
                    )
                    if self._blowup_runs[slot] >= 3:
                        raise RuntimeError(
                            f"task slot {slot} mask covered >=80% of three consumed frames"
                        )
        cloud_t0 = time.perf_counter()
        crop_frames = None
        if self.crop_frame == "base":
            crop_frames = np.stack([
                np.linalg.inv(base_pose_to_mat(state[29:32])) for state in states
            ])
        point_cloud, point_mask, diagnostics = build_resampled_world_clouds(
            depth,
            rgb,
            self.intrinsic,
            world_t_cams,
            num_points=self.num_points,
            pool_size=self.pool_size,
            rng=self._rng,
            crop_min=self.crop_min,
            crop_max=self.crop_max,
            crop_frames=crop_frames,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            masks=native_masks,
            object_nums=self.conditioning_spec.object_nums,
        )
        cloud_s = time.perf_counter() - cloud_t0

        first_gate = None
        first_iou = np.full(self.conditioning_spec.object_nums, np.nan, np.float32)
        first_area_ratio = np.full(
            self.conditioning_spec.object_nums, np.nan, np.float32
        )
        if self._first_window_gate_pending:
            first_gate = require_scene_consistency(
                native_masks[-1],
                depth[-1],
                self.intrinsic,
                world_t_cams[-1],
                self._condition.positions,
                min_depth_m=self.min_depth,
                max_depth_m=self.max_depth,
            )
            import cv2  # noqa: PLC0415 -- only on the one-time pre-action gate

            seed_native = cv2.resize(
                self._condition.seed_task_slots,
                (self.head_hw[1], self.head_hw[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            for slot in range(self.conditioning_spec.object_nums):
                seed_slot = seed_native == slot + 1
                live_slot = native_masks[-1] == slot + 1
                union = int(np.logical_or(seed_slot, live_slot).sum())
                first_iou[slot] = (
                    float(np.logical_and(seed_slot, live_slot).sum()) / union
                    if union else np.nan
                )
                first_area_ratio[slot] = float(live_slot.sum()) / float(seed_slot.sum())
            self._first_window_gate_pending = False

        gaps = np.diff(np.concatenate((
            np.asarray([self._last_tracker_timestamp_ns], dtype=np.int64), timestamps
        )))
        self._last_tracker_timestamp_ns = int(timestamps[-1])
        selected_counts = diagnostics.get("selected_label_counts")
        trace = {
            "consumed_head_timestamp_ns": timestamps,
            "tracker_gap_ns": gaps,
            "tracker_preprocess_s": np.float32(tracker_prepare_s),
            "tracker_propagate_s": np.float32(tracker_propagate_s),
            "cloud_s": np.float32(cloud_s),
            "first_window_scene_gate_valid": np.asarray(first_gate is not None),
            "first_window_seed_iou": first_iou,
            "first_window_area_ratio": first_area_ratio,
            "first_window_centroid_drift_m": (
                np.full(self.conditioning_spec.object_nums, np.nan, np.float32)
                if first_gate is None else first_gate["position_errors_m"]
            ),
            "first_window_valid_depth_fraction": (
                np.full(self.conditioning_spec.object_nums, np.nan, np.float32)
                if first_gate is None else first_gate["valid_depth_fractions"]
            ),
            "first_window_largest_component_area": (
                np.full(self.conditioning_spec.object_nums, -1, np.int64)
                if first_gate is None else first_gate["largest_component_areas"]
            ),
        }
        if native_masks is not None:
            trace.update({
                "native_point_mask": native_masks,
                "native_mask_areas": np.stack([
                    [(frame == label).sum()
                     for label in range(1, self.conditioning_spec.object_nums + 1)]
                    for frame in native_masks
                ]).astype(np.int64),
                "selected_point_mask_counts": selected_counts[:, 1:],
            })
        self._last_inference_trace = {"mask_runtime": trace}

        return [ManiFlowObservation(
            state=states[index],
            point_cloud=point_cloud[index],
            point_mask=None if point_mask is None else point_mask[index],
            head_timestamp_ns=timestamps[index],
            native_mask=None if native_masks is None else native_masks[index],
            selected_label_counts=(
                None if selected_counts is None else selected_counts[index, 1:]
            ),
        ) for index in range(len(raws))]

    def predict_chunk(self, obs_history) -> np.ndarray:
        """One forward pass -> ``(n_action_steps, 29)`` float32 ABSOLUTE world actions.

        ``obs_history`` is oldest-first at the dataset cadence, at most ``n_obs_steps``
        long; a shorter history is left-padded by repeating the oldest sample, matching
        how the LeRobot bundle bootstraps an episode start. ManiFlow's ``predict_action``
        slices its own action window at ``start = n_obs_steps - 1``, so frame 0 of the
        returned chunk is the CURRENT-step command -- the convention the scheduler stamps.
        """
        history = list(obs_history)
        if not 1 <= len(history) <= self.n_obs_steps:
            raise ValueError(
                f"obs_history must hold 1..{self.n_obs_steps} samples, got {len(history)}")
        history = [history[0]] * (self.n_obs_steps - len(history)) + history

        point_cloud = np.stack([o.point_cloud for o in history])[None]   # (1, To, N, 6)
        agent_pos = np.stack([o.state for o in history])[None]           # (1, To, 32)
        obs_dict = {
            "point_cloud": self._torch.from_numpy(np.ascontiguousarray(point_cloud, np.float32)
                                                  ).to(self.device),
            "agent_pos": self._torch.from_numpy(np.ascontiguousarray(agent_pos, np.float32)
                                                ).to(self.device),
        }
        if self.scene_requirements.needs_point_mask:
            masks = [obs.point_mask for obs in history]
            if any(mask is None for mask in masks):
                raise RuntimeError("point_mask is missing from a mask-conditioned observation")
            obs_dict["point_mask"] = self._torch.from_numpy(
                np.ascontiguousarray(np.stack(masks)[None], dtype=np.int8)
            ).to(self.device)
        if self.scene_requirements.needs_env_state:
            if self._env_state_tensor is None:
                raise RuntimeError("live env_state was not atomically installed")
            obs_dict["env_state"] = self._env_state_tensor
        if self.scene_requirements.needs_env_dino:
            if self._env_dino_tensor is None:
                raise RuntimeError("live env_dino was not atomically installed")
            obs_dict["env_dino"] = self._env_dino_tensor
        if self._torch.is_autocast_enabled("cuda"):
            raise RuntimeError("CUDA autocast leaked into ManiFlow inference")
        if self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
        policy_t0 = time.perf_counter()
        with self._torch.inference_mode():
            result = self.policy.predict_action(obs_dict)
        if self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
        if self._torch.is_autocast_enabled("cuda"):
            raise RuntimeError("ManiFlow inference leaked CUDA autocast")
        policy_s = time.perf_counter() - policy_t0
        memory = self._record_device_memory()
        chunk = result["action"]
        if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[1] != self.n_action_steps:
            raise ValueError(
                f"predict_action returned {tuple(chunk.shape)}; expected "
                f"(1, {self.n_action_steps}, {len(ACTION_AXES)})")
        out = np.asarray(
            chunk.detach().cpu().numpy(), dtype=np.float32
        ).reshape(-1, chunk.shape[-1])
        if out.shape != (self.n_action_steps, len(ACTION_AXES)) or not np.all(np.isfinite(out)):
            raise ValueError(f"bad action chunk {out.shape} (or non-finite)")
        if self._last_inference_trace is None:
            self._last_inference_trace = {}
        self._last_inference_trace["timing"] = {
            "policy_s": np.float32(policy_s),
            "post_capture_s": np.float32(time.perf_counter() - self._post_capture_t0),
            "device_free_bytes": np.int64(-1 if memory is None else memory[0]),
            "device_total_bytes": np.int64(-1 if memory is None else memory[1]),
            "episode_min_free_bytes": np.int64(
                -1 if self._episode_min_free_bytes is None
                else self._episode_min_free_bytes
            ),
            "device_max_allocated_bytes": np.int64(
                -1 if memory is None else self._torch.cuda.max_memory_allocated()
            ),
            "device_max_reserved_bytes": np.int64(
                -1 if memory is None else self._torch.cuda.max_memory_reserved()
            ),
        }
        return out

    def last_inference_trace(self) -> dict | None:
        """Return the fixed-shape telemetry tree from the latest inference."""
        return self._last_inference_trace


class ManiFlowInferenceWorker(rollout._InferenceWorker):  # noqa: SLF001 -- deliberate reuse
    """The shared async worker, with a point-cloud observation instead of images.

    Only ``_gather`` is replaced: the raw head frames + states are captured at the dataset
    cadence, THEN the clouds are built in one batch. The base class's freshness gate,
    schedule stamping, stale drop, io_log record and failure surfacing are untouched.
    """

    def _grab_fresh_raw(self) -> ManiFlowRawFrame | None:
        """One freshness-gated raw RGB-D/state sample, or None on timeout/stop.

        The head stamp must strictly advance past the previous sample this worker used
        (same-clock, same-publisher comparison): the policy must never see a frame twice.
        No wrist camera: the 3D policy has no image input at all.
        """
        deadline = time.perf_counter() + self._grab_timeout
        while not self._stop.is_set():
            head = self._driver._grab_head_images()  # noqa: SLF001 -- audited cache read
            head_ns = None if head is None else int(head[2])
            if head_ns is not None and head_ns > self._last_head_ns:
                state = rollout._build_state(  # noqa: SLF001
                    self._driver, self._fk, self._policy.state_frame)
                self._last_head_ns = head_ns
                return ManiFlowRawFrame(
                    rgb=head[0],
                    depth_mm=head[1],
                    state=state,
                    head_timestamp_ns=head_ns,
                    world_t_cam=world_t_head_from_state(
                        state, self._policy.state_frame
                    ).astype(np.float32),
                    world_frame_epoch=int(self._driver._world_frame_epoch),  # noqa: SLF001
                )
            if time.perf_counter() >= deadline:
                return None
            self._stop.wait(0.005)
        return None

    def _gather(self) -> tuple[list[ManiFlowObservation], float] | None:
        """``n_obs_steps`` fresh raw frames at the dataset cadence, then batched clouds."""
        raws: list[ManiFlowRawFrame] = []
        prev_grab_t = 0.0
        for i in range(self._policy.n_obs_steps):
            if i:
                self._stop.wait(
                    max(0.0, prev_grab_t + self._dataset_dt - time.perf_counter()))
                if self._stop.is_set():
                    return None
            raw = self._grab_fresh_raw()
            if raw is None:
                return None
            prev_grab_t = time.perf_counter()
            raws.append(raw)
        return self._policy.build_observations(raws), prev_grab_t


def main() -> None:
    parser = rollout.build_parser()
    parser.description = __doc__
    mf = parser.add_argument_group("maniflow (3D point-cloud policy)")
    mf.add_argument("--num-inference-steps", type=int, default=None,
                    help="override the checkpoint's flow-ODE step count. ManiFlow is trained "
                         "with a consistency objective, so 1-2 is usually enough and cuts "
                         "inference far below the diffusion baseline (default: checkpoint's).")
    mf.add_argument("--no-ema", action="store_true",
                    help="load the raw 'model' weights instead of the EMA copy (default: EMA "
                         "when the checkpoint has one).")
    mf.add_argument("--pointcloud-meta", default=None,
                    help="override the training zarr's <name>_<split>.meta.json, which supplies "
                         "the crop/depth/frame the live cloud MUST be built with. Default: the "
                         "sidecar of the zarr_path recorded in the checkpoint's cfg.")
    args = parser.parse_args()

    if args.save_dir is None and not args.replay_episode:
        args.save_dir = DEFAULT_SAVE_DIR
    rollout.finalize_args(parser, args)

    policy_factory = functools.partial(
        ManiFlowBundle, num_inference_steps=args.num_inference_steps,
        use_ema=not args.no_ema, meta_path=args.pointcloud_meta,
        scene_diff_repo=args.scene_diff_repo,
    )
    rollout._run_rollout(  # noqa: SLF001 -- the shared hardware loop, by design
        args, policy_factory=policy_factory, worker_factory=ManiFlowInferenceWorker)


if __name__ == "__main__":
    main()
