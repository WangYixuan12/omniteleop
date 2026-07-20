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

``--position-condition`` is shared too: a checkpoint trained with
``position_condition_mode`` = ``concat`` or ``scene_query_tokens`` declares it, the
shared rollout captures ONE head frame at the start pose, runs SceneDiff against the
fixed reference and pins the arranged [box, cloth] world vector via ``set_env_state``;
this bundle feeds it to the policy as ``obs['env_state']`` on every chunk.

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

    # position-conditioned checkpoint (live SceneDiff after reference alignment):
    python scripts/wbc_maniflow_rollout.py --policy-path <conditioned ckpt> \
      --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5 \
      --position-condition --prompt-after-capture --num-inference-steps 2

    # replay a recorded take through the identical actuation path (no policy):
    python scripts/wbc_maniflow_rollout.py --replay-episode ~/Dexmate/data/raw_data/episode_1.hdf5 \
      --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5
"""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from omniteleop.wbc_pointcloud import (
    POINT_CHANNELS,
    build_world_cloud,
    resample_clouds,
    sampler_signature,
    world_t_head_from_state,
)
from omniteleop.wbc_policy_format import ACTION_AXES, STATE_AXES


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


@dataclass(frozen=True)
class ManiFlowObservation:
    """One raw (unbatched) ManiFlow observation. ``.state`` is what the io_log records."""

    state: np.ndarray         # (32,) float32, frame per the training meta
    point_cloud: np.ndarray   # (num_points, 6) float32, world XYZ + RGB[0,1]


class ManiFlowBundle:
    """ManiFlow checkpoint + live cloud preprocessing, in ``_PolicyBundle``'s shape."""

    # The shared loop never asks a ManiFlow policy for these two, but reads them for the
    # --position-condition cross-check; the real values are set from the checkpoint cfg.
    use_wrist = False
    use_relative_actions = False

    def __init__(self, policy_path: str, *, device: str | None = None,
                 num_inference_steps: int | None = None, use_ema: bool = True,
                 meta_path: str | None = None, pool_size: int | None = None,
                 seed: int = 0, allow_sampler_mismatch: bool = False) -> None:
        import dill  # noqa: PLC0415 -- heavy, GPU path only
        import hydra  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from omegaconf import OmegaConf  # noqa: PLC0415

        from omniteleop.common.head_camera import HEAD_RESIZE_HW, ZED_K  # noqa: PLC0415

        OmegaConf.register_new_resolver("eval", eval, replace=True)
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = Path(policy_path)
        if not ckpt.is_file():
            raise ValueError(
                f"--policy-path must be a ManiFlow .ckpt FILE, got {ckpt} "
                "(e.g. .../checkpoints/latest.ckpt)")
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
            raise ValueError(f"{ckpt}: point_cloud shape {pc_shape}, expected (N, {POINT_CHANNELS})")
        if state_dim != len(STATE_AXES) or action_dim != len(ACTION_AXES):
            raise ValueError(
                f"{ckpt}: agent_pos={state_dim}/action={action_dim} do not match the WBC "
                f"schema ({len(STATE_AXES)}/{len(ACTION_AXES)}); wrong checkpoint?")
        self.num_points = int(pc_shape[0])

        # SceneDiff position conditioning: declared by the checkpoint's policy cfg
        # (concat | scene_query_tokens). The shared _run_rollout cross-checks these
        # against --position-condition and computes the live env-state vector; we only
        # have to consume it in predict_chunk.
        self.position_condition_mode = str(
            cfg.policy.get("position_condition_mode", "none") or "none")
        if self.position_condition_mode not in ("none", "concat", "scene_query_tokens"):
            raise ValueError(
                f"{ckpt}: unknown position_condition_mode "
                f"{self.position_condition_mode!r}")
        self.use_env_state = self.position_condition_mode != "none"
        self.env_state_dim = (int(cfg.policy.get("object_nums", 2)) * 3
                              if self.use_env_state else 0)
        self._env_state: np.ndarray | None = None

        # Live preprocessing MUST match training exactly -> read it from the zarr's sidecar.
        self.meta_path = Path(meta_path) if meta_path else self._meta_path_from_cfg(cfg, ckpt)
        self._allow_sampler_mismatch = bool(allow_sampler_mismatch)
        meta = self._load_meta(self.meta_path)
        self.state_frame = meta["state_frame"]
        self.crop_min = tuple(float(v) for v in meta["crop_min"])
        self.crop_max = tuple(float(v) for v in meta["crop_max"])
        self.min_depth = float(meta["min_depth_m"])
        self.max_depth = float(meta["max_depth_m"])
        self.dataset_fps = float(meta.get("fps", 10))
        # The pre-FPS uniform pool is part of the trained point distribution too: default
        # to the porter's recorded value, not a CLI constant.
        self.pool_size = int(meta["pool_size"]) if pool_size is None else int(pool_size)
        if pool_size is not None and self.pool_size != int(meta["pool_size"]):
            print(f"[wbc_maniflow_rollout] WARNING: --pool-size {self.pool_size} != the "
                  f"training zarr's {int(meta['pool_size'])}; the live pre-FPS pool "
                  "distribution will differ from training.", flush=True)
        if self.pool_size < self.num_points:
            raise ValueError(f"--pool-size {self.pool_size} < num_points {self.num_points}")
        self._rng = np.random.default_rng(seed)

        self.intrinsic = np.asarray(ZED_K, dtype=np.float64)
        self.head_hw = tuple(HEAD_RESIZE_HW)
        if self.intrinsic.shape != (3, 3):
            raise ValueError(f"ZED_K must be (3,3), got {self.intrinsic.shape}")

        self.policy = hydra.utils.instantiate(cfg.policy)
        state_dicts = payload["state_dicts"]
        which = "ema_model" if (use_ema and "ema_model" in state_dicts) else "model"
        if which not in state_dicts:
            raise ValueError(f"{ckpt}: no '{which}' in state_dicts {sorted(state_dicts)}")
        self.policy.load_state_dict(state_dicts[which])
        # A checkpoint saved before set_normalizer() ran loads without complaint and then
        # dies inside predict_action's unnormalize -- i.e. on the robot, after engage.
        # Fail here instead, where it costs nothing.
        required_stats = ["action", "point_cloud", "agent_pos"]
        if self.use_env_state:
            required_stats.append("env_state")
        missing = [k for k in required_stats
                   if k not in self.policy.normalizer.params_dict]
        if missing:
            raise ValueError(
                f"{ckpt}: the '{which}' normalizer has no {missing} stats -- it was saved "
                "before set_normalizer(); the policy cannot un-normalize its actions")
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

    @staticmethod
    def _meta_path_from_cfg(cfg, ckpt: Path) -> Path:
        try:
            zarr_path = Path(str(cfg.robotwin_task.dataset.zarr_path))
        except Exception as exc:  # noqa: BLE001 -- a non-WBC ManiFlow cfg
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
        for key in ("state_frame", "crop_min", "crop_max", "min_depth_m", "max_depth_m",
                    "num_points", "point_frame", "sampler", "pool_size"):
            if key not in meta:
                raise ValueError(f"{path}: missing '{key}' (regenerate with port_wbc_mobile_zarr.py)")
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
            msg = (f"zarr built with fpsample {meta.get('fpsample_version')}, this env has "
                   f"{ours['fpsample_version']}: the sampled point set may silently differ "
                   "from training")
            if not self._allow_sampler_mismatch:
                raise ValueError(
                    f"{path}: {msg}. Rebuild the zarr in this env, or pass "
                    "--unsafe-sampler-mismatch to proceed anyway.")
            print(f"[wbc_maniflow_rollout] WARNING (--unsafe-sampler-mismatch): {msg}.",
                  flush=True)
        if int(meta["num_points"]) != self.num_points:
            raise ValueError(
                f"{path}: num_points {meta['num_points']} != the checkpoint's {self.num_points} "
                "-- this meta belongs to a different zarr")
        if not str(meta["point_frame"]).startswith("world"):
            raise ValueError(f"{path}: point_frame {meta['point_frame']!r} is not world-frame")
        if meta["state_frame"] not in ("base", "world"):
            raise ValueError(f"{path}: unexpected state_frame {meta['state_frame']!r}")
        return meta

    def describe(self) -> str:
        return (f"ManiFlow {type(self.policy).__name__} ({self.weights}) on {self.device}; "
                f"point_cloud {self.num_points}x{POINT_CHANNELS} world-frame, crop "
                f"{list(self.crop_min)}..{list(self.crop_max)}; state_frame={self.state_frame}; "
                f"position_condition={self.position_condition_mode}"
                f"{f' ({self.env_state_dim}-D)' if self.use_env_state else ''}; "
                f"n_obs_steps={self.n_obs_steps} n_action_steps={self.n_action_steps} "
                f"num_inference_steps={self.num_inference_steps}; meta={self.meta_path}")

    def reset(self) -> None:
        self.policy.reset()

    def set_env_state(self, vec: np.ndarray) -> None:
        """Pin the constant SceneDiff position condition (arranged, engage-origin world).

        Same contract as ``_PolicyBundle.set_env_state``: called once by the shared
        rollout after the live SceneDiff run, before the inference worker starts, and
        held for the whole episode -- exactly the per-episode-constant vector training
        saw. The policy normalizes it with the point-cloud XYZ affine internally.
        """
        if not self.use_env_state:
            raise RuntimeError(
                f"checkpoint's position_condition_mode is 'none'; drop --position-condition")
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.shape != (self.env_state_dim,) or not np.all(np.isfinite(v)):
            raise ValueError(
                f"env-state must be a finite ({self.env_state_dim},) vector, got {v.shape}")
        self._env_state = v

    def build_observations(
        self, raws: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> list[ManiFlowObservation]:
        """``[(head_rgb, head_depth_mm, state)] -> [ManiFlowObservation]``.

        Runs AFTER the cadence-critical grabs, so the observation spacing stays at the
        dataset period rather than being stretched by the cloud construction. Identical
        construction to ``port_wbc_mobile_zarr.episode_arrays`` -- same module, same
        pinned sampler (cross-checked against the training ``.meta.json``).
        """
        if not raws:
            raise ValueError("no raw head frames to build observations from")
        clouds = []
        for rgb, depth_mm, state in raws:
            if rgb.shape[:2] != self.head_hw:
                raise ValueError(
                    f"head frame {rgb.shape[:2]} != the {self.head_hw} ZED_K was derived for; "
                    "the publisher's HEAD_RESIZE_HW changed since training")
            world_t_zed = world_t_head_from_state(state, self.state_frame)
            clouds.append(build_world_cloud(
                depth_mm, rgb, self.intrinsic, world_t_zed,
                crop_min=self.crop_min, crop_max=self.crop_max,
                min_depth=self.min_depth, max_depth=self.max_depth))
        sampled = resample_clouds(clouds, self.num_points, pool_size=self.pool_size,
                                  rng=self._rng)
        return [ManiFlowObservation(state=raw[2], point_cloud=pc)
                for raw, pc in zip(raws, sampled)]

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
        if self.use_env_state:
            if self._env_state is None:
                raise RuntimeError(
                    f"checkpoint's position_condition_mode is "
                    f"{self.position_condition_mode!r} but set_env_state() was never "
                    "called -- run with --position-condition")
            # (1, 3K): the policy broadcasts the constant vector across n_obs_steps.
            obs_dict["env_state"] = self._torch.from_numpy(
                self._env_state[None]).to(self.device)
        with self._torch.no_grad():
            result = self.policy.predict_action(obs_dict)
        chunk = result["action"]
        if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[1] != self.n_action_steps:
            raise ValueError(
                f"predict_action returned {tuple(chunk.shape)}; expected "
                f"(1, {self.n_action_steps}, {len(ACTION_AXES)})")
        out = np.asarray(chunk.detach().cpu().numpy(), dtype=np.float32).reshape(-1, chunk.shape[-1])
        if out.shape != (self.n_action_steps, len(ACTION_AXES)) or not np.all(np.isfinite(out)):
            raise ValueError(f"bad action chunk {out.shape} (or non-finite)")
        return out


class ManiFlowInferenceWorker(rollout._InferenceWorker):  # noqa: SLF001 -- deliberate reuse
    """The shared async worker, with a point-cloud observation instead of images.

    Only ``_gather`` is replaced: the raw head frames + states are captured at the dataset
    cadence, THEN the clouds are built in one batch. The base class's freshness gate,
    schedule stamping, stale drop, io_log record and failure surfacing are untouched.
    """

    def _grab_fresh_raw(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """One freshness-gated ``(head_rgb, head_depth_mm, state)``, or None on timeout/stop.

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
                return head[0], head[1], state
            if time.perf_counter() >= deadline:
                return None
            self._stop.wait(0.005)
        return None

    def _gather(self) -> tuple[list[ManiFlowObservation], float] | None:
        """``n_obs_steps`` fresh raw frames at the dataset cadence, then batched clouds."""
        raws: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
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
    mf.add_argument("--pool-size", type=int, default=None,
                    help="uniform pool before the farthest-point sample. Default: the "
                         "training zarr's recorded pool_size (a live mismatch skews the "
                         "point distribution; overriding prints a loud warning).")
    mf.add_argument("--pc-seed", type=int, default=0,
                    help="seed for the uniform pool subsample (FPS itself is deterministic).")
    mf.add_argument("--unsafe-sampler-mismatch", action="store_true",
                    help="proceed even if this env's fpsample version differs from the one "
                         "the training zarr was built with (default: refuse).")
    args = parser.parse_args()

    if args.save_dir is None and not args.replay_episode:
        args.save_dir = DEFAULT_SAVE_DIR
    rollout.finalize_args(parser, args)

    if args.pool_size is not None and args.pool_size < 1:
        parser.error(f"--pool-size must be >= 1, got {args.pool_size}")

    policy_factory = functools.partial(
        ManiFlowBundle, num_inference_steps=args.num_inference_steps,
        use_ema=not args.no_ema, meta_path=args.pointcloud_meta,
        pool_size=args.pool_size, seed=args.pc_seed,
        allow_sampler_mismatch=args.unsafe_sampler_mismatch,
    )
    rollout._run_rollout(  # noqa: SLF001 -- the shared hardware loop, by design
        args, policy_factory=policy_factory, worker_factory=ManiFlowInferenceWorker)


if __name__ == "__main__":
    main()
