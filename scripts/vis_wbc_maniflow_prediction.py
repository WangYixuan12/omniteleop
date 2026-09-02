#!/usr/bin/env python3
"""Offline GT-vs-predicted validation of a WBC ManiFlow point-cloud checkpoint.

The ManiFlow sibling of ``scripts/vis_wbc_mof_prediction.py``: replays ONE ported
zarr episode through the checkpoint's own ``policy.predict_action`` -- the exact
call ``scripts/wbc_maniflow_rollout.py`` makes on hardware -- and compares the
predicted 29-D WORLD actions against the recorded ground truth. This is the gate
before spending robot time on a rollout.

What "offline rollout" means here: the observations always come from the recorded
episode, never from the policy's own consequences. At every ``--stride``-th frame
the stored ``n_obs_steps`` observations go in and the returned ``n_action_steps``
chunk is scored against ``action[t : t + n_action_steps]``. Nothing is fed back --
an offline replay cannot move the robot, so the trajectory stays on the human take.

Observations are read verbatim from the zarr written by
``scripts/port_wbc_mobile_zarr.py`` (``data/point_cloud``, ``data/state`` ->
``agent_pos``, plus ``data/point_mask`` / ``data/env_state`` / ``data/env_dino``
when the checkpoint's conditioning modes require them). Those are the same tensors
training consumed and the same ones the live cloud builder reproduces on hardware,
so the rollout's live-preprocessing gates (crop/depth/sampler provenance) have
nothing to check here: no cloud is rebuilt from RGB-D. The train/eval zarr pair is
still checked for identical preprocessing, because the checkpoint's normalizer was
fit on the training zarr.

Weights come from the ``.ckpt`` payload; the run's ``.hydra/config.yaml`` is read
too (``--maniflow-run-dir``, default ``<ckpt>/../..``) exactly as
``scripts/vis_episode_processed_wbc.py --zarr`` reads it. It is the second,
independent record of the same switches -- the two are cross-checked and a
disagreement is fatal -- and it alone names the sampler the encoder runs, the
``visual_cond_len`` budget it keeps, and the MoF expert set. Both the sampler and
the MoF transforms are imported from the run's own ManiFlow checkout, never from a
viewer-side copy.

Outputs, per entity (left / right / head):
- translation MAE (m) and geodesic rotation error (deg) over all predicted steps,
  printed to stdout, plus gripper command accuracy for the two arms;
- an overlay figure: GT (solid) vs 1-step-ahead prediction (dashed) position traces
  per axis, rotation-error series, and the gripper command series;
- a Rerun recording, ``--save FILE.rrd`` over SSH:
  * Scene tab -- the stored world cloud AND the config's own sampler selection (the
    ``visual_cond_len`` points the encoder actually keeps, plus their mask labels
    when the run is mask-conditioned), each with the GT and predicted world action
    horizons and their first-step error, the base odometry path, head/wrist RGB
    (with the action horizons reprojected into the head image) and gripper series;
  * MoF frames tab (``action_frame_mode: mof``) -- the same GT vs predicted chunk
    re-expressed in every enabled expert representation through the run checkout's
    own ``frame_transforms``, round-trip-checked back to world;
  * MoF router tab -- learned expert weights across the flow-integration steps.

Environment: ``dexmate_maniflow``.

Usage::

    RUN=~/Dexmate/model/maniflow/maniflow_mof_pos_200_seed0
    python scripts/vis_wbc_maniflow_prediction.py \
      --policy-path $RUN/checkpoints/epoch=090-val_loss=0.174436.ckpt \
      --zarr ~/Dexmate/data/processed_wbc/maniflow/dexmate_wbc_test.zarr \
      --episode_index 1

    # headless over SSH -> open later with `rerun FILE.rrd`:
    python scripts/vis_wbc_maniflow_prediction.py --policy-path <ckpt> \
      --save ~/Dexmate/tmp/scratchpad/maniflow_pred_ep0.rrd
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import dill
import hydra
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import zarr
from maniflow.model.common.conditioning import ConditioningSpec
from omegaconf import OmegaConf

from omniteleop.wbc_pointcloud import POINT_CHANNELS
from omniteleop.wbc_policy_format import (
    ACTION_AXES,
    GRIPPER_BINARY_THRESHOLD,
    STATE_AXES,
    base_pose_to_mat,
    pos6d_to_mat,
)

# (position slice, rot6d slice, gripper channel) inside the 29-D world action.
ENTITIES = {
    "left": (slice(0, 3), slice(3, 9), 9),
    "right": (slice(10, 13), slice(13, 19), 19),
    "head": (slice(20, 23), slice(23, 29), None),
}
_STATE_HEAD_BLOCK = slice(20, 29)
_STATE_BASE_BLOCK = slice(29, 32)

_COLOR_GT = (255, 0, 0)        # red     — recorded action
_COLOR_PRED = (230, 60, 230)   # magenta — policy prediction
_COLOR_ERR = (235, 235, 235)   # white   — first-step GT↔pred error segment
_COLOR_BASE = (230, 190, 0)    # gold    — base odometry
_ROUTER_COLORS = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
)
_DEFAULT_ZARR = Path(
    "~/Dexmate/data/processed_wbc/maniflow/dexmate_wbc_test.zarr"
).expanduser()
_DEFAULT_OUTPUT_DIR = Path("~/Dexmate/tmp/scratchpad").expanduser()
_HEAD_DEPTH_ENTITY = "/world/camera/depth"


def _import_zarr_viewer():
    """Import the sibling zarr viewer for its loaders (``scripts`` is not a package)."""
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    path = scripts_dir / "vis_episode_processed_wbc.py"
    spec = importlib.util.spec_from_file_location("vis_episode_processed_wbc", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import the zarr viewer at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


viewer = _import_zarr_viewer()

# Mask legend colors stay identical to the input viewer's, one source of truth.
_COLOR_MASK_BG = viewer._COLOR_MASK_BG      # noqa: SLF001
_COLOR_MASK_SRC = viewer._COLOR_MASK_SRC    # noqa: SLF001
_COLOR_MASK_DST = viewer._COLOR_MASK_DST    # noqa: SLF001


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) column rot6d -> (..., 3, 3); columns are stored orthonormal."""
    c1, c2 = rot6d[..., 0:3], rot6d[..., 3:6]
    c1 = c1 / np.linalg.norm(c1, axis=-1, keepdims=True)
    c2 = c2 - np.sum(c1 * c2, axis=-1, keepdims=True) * c1
    c2 = c2 / np.linalg.norm(c2, axis=-1, keepdims=True)
    return np.stack([c1, c2, np.cross(c1, c2)], axis=-1)


def _geodesic_deg(rot6d_a: np.ndarray, rot6d_b: np.ndarray) -> np.ndarray:
    Ra, Rb = _rot6d_to_matrix(rot6d_a), _rot6d_to_matrix(rot6d_b)
    tr = np.einsum("...ij,...ij->...", Ra, Rb)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


class OfflineManiFlowPolicy:
    """A WBC ManiFlow checkpoint driven from stored zarr observations.

    The checkpoint half of ``wbc_maniflow_rollout.ManiFlowBundle`` (schema gate,
    conditioning spec, weights, inference contract) without its live cloud
    preprocessing: offline, the observation IS the porter's stored tensor.
    """

    def __init__(self, policy_path: str, *, device: str | None = None,
                 use_ema: bool = True, num_inference_steps: int | None = None) -> None:
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = Path(policy_path).expanduser()
        if not ckpt.is_file():
            raise ValueError(
                f"--policy-path must be a ManiFlow .ckpt FILE, got {ckpt} "
                "(e.g. .../checkpoints/epoch=030-val_loss=0.130616.ckpt)")
        self.checkpoint_path = ckpt.resolve()
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
                f"{ckpt}: point_cloud shape {pc_shape}, expected (N, {POINT_CHANNELS})")
        if state_dim != len(STATE_AXES) or action_dim != len(ACTION_AXES):
            raise ValueError(
                f"{ckpt}: agent_pos={state_dim}/action={action_dim} do not match the WBC "
                f"schema ({len(STATE_AXES)}/{len(ACTION_AXES)}); wrong checkpoint?")
        self.num_points = int(pc_shape[0])

        self.conditioning_spec = ConditioningSpec(
            position_condition_mode=str(
                cfg.policy.get("position_condition_mode", "none") or "none"),
            point_sampling_mode=str(cfg.policy.get("point_sampling_mode", "fps") or "fps"),
            mask_channels=bool(cfg.policy.get("mask_channels", False)),
            action_frame_mode=str(cfg.policy.get("action_frame_mode", "world") or "world"),
            object_nums=int(cfg.policy.get("object_nums", 2)),
            dino_dim=int(cfg.policy.get("position_dino_dim", 1280)),
        )
        self.action_frame_mode = self.conditioning_spec.action_frame_mode

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
        self.expert_names = (
            tuple(self.policy.mof_enabled_experts) if self.action_frame_mode == "mof" else ()
        )
        try:
            self.train_zarr = Path(str(cfg.robotwin_task.dataset.zarr_path))
        except Exception as exc:  # a non-WBC ManiFlow cfg
            raise ValueError(f"{ckpt}: cfg has no robotwin_task.dataset.zarr_path") from exc

    def describe(self) -> str:
        """Return a one-line summary of what this checkpoint consumes and emits."""
        spec = self.conditioning_spec
        return (
            f"ManiFlow {type(self.policy).__name__} ({self.weights}) on {self.device}; "
            f"point_cloud {self.num_points}x{POINT_CHANNELS}; "
            f"position_condition={spec.position_condition_mode}, "
            f"sampling={spec.point_sampling_mode}, mask_channels={spec.mask_channels}, "
            f"action_frame={self.action_frame_mode}"
            + (f" [{', '.join(self.expert_names)}]" if self.expert_names else "")
            + f"; n_obs_steps={self.n_obs_steps} n_action_steps={self.n_action_steps} "
            f"num_inference_steps={self.num_inference_steps}; "
            f"train zarr (normalizer) {self.train_zarr}"
        )

    def policy_switches(self) -> dict:
        """The switch dict the zarr episode loader validates its arrays against."""
        spec = self.conditioning_spec
        return {
            "needs_point_mask": spec.needs_point_mask,
            "position_condition_mode": spec.position_condition_mode,
            "object_nums": spec.object_nums,
            "position_dino_dim": spec.dino_dim,
        }

    def point_cloud_normalization(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The encoder's own point-cloud ``(scale, offset)``.

        The sibling viewer refits these from the training zarr; the checkpoint already
        carries the fitted values, so read them here -- exact by construction and
        without a pass over 18k training frames.
        """
        params = self.policy.normalizer.params_dict["point_cloud"]
        scale = params["scale"].detach().cpu()
        offset = params["offset"].detach().cpu()
        if scale.shape != (POINT_CHANNELS,) or offset.shape != (POINT_CHANNELS,):
            raise ValueError(
                f"point_cloud normalizer is {tuple(scale.shape)}/{tuple(offset.shape)}, "
                f"expected ({POINT_CHANNELS},)")
        return scale, offset

    def _window(self, array: np.ndarray, t: int, dtype) -> torch.Tensor:
        """(1, n_obs_steps, ...) device tensor of the observations ending at frame ``t``."""
        window = array[t - self.n_obs_steps + 1 : t + 1]
        if window.shape[0] != self.n_obs_steps:
            raise ValueError(
                f"frame {t} has {window.shape[0]} observation steps, "
                f"needs {self.n_obs_steps}")
        return torch.from_numpy(np.ascontiguousarray(window[None], dtype=dtype)).to(self.device)

    def predict_chunk(self, source: dict, t: int) -> np.ndarray:
        """``(n_action_steps, 29)`` float32 ABSOLUTE world actions for frame ``t``.

        ManiFlow slices its own action window at ``start = n_obs_steps - 1``, so row 0
        of the chunk is the CURRENT-step command -- the convention the live scheduler
        stamps and the one scored against ``action[t : t + n_action_steps]`` here.
        """
        spec = self.conditioning_spec
        obs = {
            "point_cloud": self._window(source["point_cloud"], t, np.float32),
            "agent_pos": self._window(source["state"], t, np.float32),
        }
        if spec.needs_point_mask:
            obs["point_mask"] = self._window(source["point_mask"], t, np.int8)
        if spec.needs_env_state:
            env_state = source["position_condition"]["points"][t].reshape(-1)
            obs["env_state"] = torch.from_numpy(
                np.ascontiguousarray(env_state[None], dtype=np.float32)).to(self.device)
        if spec.needs_env_dino:
            obs["env_dino"] = torch.from_numpy(
                np.ascontiguousarray(source["env_dino"][t][None], dtype=np.float32)
            ).to(self.device)
        with torch.inference_mode():
            result = self.policy.predict_action(obs)
        chunk = np.asarray(result["action"].detach().cpu().numpy(), dtype=np.float32)
        if chunk.shape != (1, self.n_action_steps, len(ACTION_AXES)):
            raise ValueError(
                f"predict_action returned {chunk.shape}, expected "
                f"(1, {self.n_action_steps}, {len(ACTION_AXES)})")
        chunk = chunk[0]
        if not np.all(np.isfinite(chunk)):
            raise ValueError(f"predicted chunk at frame {t} is not finite")
        return chunk


def check_run_config(bundle: OfflineManiFlowPolicy, run: dict) -> None:
    """Reject a ``--maniflow-run-dir`` whose config is not this checkpoint's.

    The run's ``.hydra/config.yaml`` and the checkpoint payload record the same
    switches twice; a disagreement means the two belong to different runs, and every
    config-faithful view below (sampler selection, MoF experts) would then describe a
    model that did not produce these predictions.
    """
    spec = bundle.conditioning_spec
    mismatches = {
        key: (mine, theirs)
        for key, mine, theirs in (
            ("point_sampling_mode", spec.point_sampling_mode, run["point_sampling_mode"]),
            ("mask_channels", spec.mask_channels, run["mask_channels"]),
            ("position_condition_mode", spec.position_condition_mode,
             run["position_condition_mode"]),
            ("object_nums", spec.object_nums, run["object_nums"]),
            ("action_frame_mode", bundle.action_frame_mode, run["action_frame_mode"]),
            ("position_dino_dim", spec.dino_dim, run["position_dino_dim"]),
        )
        if mine != theirs
    }
    if run["mof"] is not None:
        for key, mine, theirs in (
            ("mof.enabled_experts", bundle.expert_names, run["mof"]["enabled_experts"]),
            ("n_obs_steps", bundle.n_obs_steps, run["mof"]["n_obs_steps"]),
        ):
            if mine != theirs:
                mismatches[key] = (mine, theirs)
    if mismatches:
        detail = ", ".join(f"{key}: ckpt {mine!r} != run {theirs!r}"
                           for key, (mine, theirs) in mismatches.items())
        raise ValueError(
            f"--maniflow-run-dir {run['run_dir']} does not describe "
            f"{bundle.checkpoint_path}: {detail}")


def mof_representations(transforms, layout, state_window: np.ndarray,
                        chunks: np.ndarray, experts: tuple[str, ...]) -> dict[str, np.ndarray]:
    """One chunk stack re-expressed in every enabled expert representation.

    ``state_window`` is the raw ``(n_obs_steps, 32)`` agent_pos the policy conditioned
    on; frames/refs come from it exactly as ``ManiFlowTransformerPointcloudPolicy.
    _mof_frames_refs`` builds them, so these are the spaces the experts predicted in.
    Each representation is round-tripped back to world as a transform check.
    """
    agent_pos = torch.from_numpy(np.ascontiguousarray(state_window[None], dtype=np.float32))
    batch = torch.from_numpy(np.ascontiguousarray(chunks, dtype=np.float32))
    agent_pos = agent_pos.expand(batch.shape[0], -1, -1)
    frames = transforms.action_frames_from_agent_pos(agent_pos, layout)
    refs = transforms.entity_refs_in_base(agent_pos, layout)
    out: dict[str, np.ndarray] = {}
    for name in experts:
        rep = transforms.world_to_expert_pose(batch, name, frames, refs, layout)
        back = transforms.expert_pose_to_world(rep, name, frames, refs, layout)
        error = float((back - batch).abs().max())
        if error > 1e-3:
            raise ValueError(f"MoF expert {name!r} round trip diverges by {error:.2e}")
        out[name] = rep.numpy()
    return out


def resolve_train_zarr(bundle: OfflineManiFlowPolicy, eval_zarr: Path,
                       override: Path | None) -> Path:
    """The training zarr whose stats the checkpoint's normalizer carries.

    The cfg records the TRAINING machine's path; when that is not mounted here, fall
    back to the same filename beside the evaluation zarr (how these datasets are
    mirrored). Guessing a different dataset would silently compare against foreign
    preprocessing, so an unresolvable path is an error, not a skipped check.
    """
    if override is not None:
        path = override.expanduser()
        if not path.exists():
            raise FileNotFoundError(f"--train-zarr not found: {path}")
        return path
    candidates = [bundle.train_zarr.expanduser(), eval_zarr.parent / bundle.train_zarr.name]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"the checkpoint's training zarr is not available here (tried "
        f"{', '.join(str(p) for p in candidates)}); pass --train-zarr")


def attach_env_dino(source: dict, zarr_path: Path, bundle: OfflineManiFlowPolicy) -> dict:
    """Slice ``data/env_dino`` for this episode (the loader validates but drops it)."""
    entry = source["manifest_entry"]
    start = int(entry["zarr_frame_start"])
    end = int(entry["zarr_frame_end_exclusive"])
    root = zarr.open(str(zarr_path), mode="r")
    env_dino = np.asarray(root["data/env_dino"][start:end], dtype=np.float32)
    expected = bundle.conditioning_spec.object_nums * bundle.conditioning_spec.dino_dim
    if env_dino.shape != (source["N"], expected):
        raise ValueError(
            f"{zarr_path}: env_dino {env_dino.shape} != ({source['N']}, {expected})")
    source = dict(source)
    source["env_dino"] = env_dino
    return source


def evaluate(bundle: OfflineManiFlowPolicy, source: dict, stride: int) -> tuple[
    list[tuple[int, np.ndarray, np.ndarray]], list[np.ndarray]
]:
    """Chunk predictions over the episode + (mof only) per-flow-step router weights."""
    T = source["N"]
    n_obs, n_act = bundle.n_obs_steps, bundle.n_action_steps
    action = source["action"].astype(np.float32)
    records: list[tuple[int, np.ndarray, np.ndarray]] = []
    router_traces: list[np.ndarray] = []
    captured: list[np.ndarray] = []

    def _capture_router(_module, _inputs, output) -> None:
        # The MoF mixture model returns (native expert outputs, router probs).
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(f"unexpected MoF model output {type(output)}")
        probs = output[1].detach().cpu().numpy()
        if probs.ndim != 2 or probs.shape[0] != 1:
            raise RuntimeError(f"unexpected inference router shape {probs.shape}")
        captured.append(np.asarray(probs[0], dtype=np.float32))

    hook = (
        bundle.policy.model.register_forward_hook(_capture_router)
        if bundle.action_frame_mode == "mof" else None
    )
    try:
        for t in range(n_obs - 1, T - n_act, stride):
            trace_start = len(captured)
            pred = bundle.predict_chunk(source, t).copy()
            if hook is not None:
                trace = np.stack(captured[trace_start:])
                if trace.shape[0] != bundle.num_inference_steps:
                    raise RuntimeError(
                        f"captured {trace.shape[0]} router steps, expected "
                        f"{bundle.num_inference_steps}")
                router_traces.append(trace)
            # Match the shared rollout: grippers become {0,1} commands at
            # GRIPPER_BINARY_THRESHOLD before actuation (split_policy_action).
            for _, _, grip_ch in ENTITIES.values():
                if grip_ch is not None:
                    pred[:, grip_ch] = (
                        pred[:, grip_ch] >= GRIPPER_BINARY_THRESHOLD).astype(np.float32)
            records.append((t, action[t : t + n_act], pred))
    finally:
        if hook is not None:
            hook.remove()
    if not records:
        raise SystemExit(
            f"episode too short ({T} frames) for one {n_obs}+{n_act}-frame window")
    return records, router_traces


def report(records: list[tuple[int, np.ndarray, np.ndarray]]) -> dict[str, float]:
    gt = np.stack([r[1] for r in records])      # (K, n_act, 29)
    pred = np.stack([r[2] for r in records])
    metrics: dict[str, float] = {}
    print(f"chunks evaluated: {len(records)} x {gt.shape[1]} steps")
    for name, (pos_sl, rot_sl, grip_ch) in ENTITIES.items():
        t_mae = float(np.abs(pred[..., pos_sl] - gt[..., pos_sl]).mean())
        r_err = float(_geodesic_deg(pred[..., rot_sl], gt[..., rot_sl]).mean())
        metrics[f"{name}/translation_mae_m"] = t_mae
        metrics[f"{name}/rotation_err_deg"] = r_err
        line = f"  {name:>5}: translation MAE {t_mae * 100:6.2f} cm | rotation err {r_err:6.2f} deg"
        if grip_ch is not None:
            # pred grippers are already the rollout's {0,1} commands (see evaluate).
            acc = float((pred[..., grip_ch] == gt[..., grip_ch]).mean())
            metrics[f"{name}/gripper_acc"] = acc
            line += f" | gripper acc {acc * 100:5.1f}%"
        print(line)
    return metrics


def render(records: list[tuple[int, np.ndarray, np.ndarray]], out_path: Path,
           title: str) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ts = np.array([r[0] for r in records])
    gt1 = np.stack([r[1][0] for r in records])    # first predicted step vs GT
    pred1 = np.stack([r[2][0] for r in records])

    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    for row, (name, (pos_sl, rot_sl, _grip_ch)) in enumerate(ENTITIES.items()):
        for col, axis_name in enumerate("xyz"):
            ax = axes[row][col]
            ch = pos_sl.start + col
            ax.plot(ts, gt1[:, ch], "-", color="tab:red", label="GT")
            ax.plot(ts, pred1[:, ch], "--", color="tab:purple", label="pred")
            ax.set_ylabel(f"{name} {axis_name} [m]")
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=8)
        col = list(ENTITIES).index(name)
        axes[3][col].plot(ts, _geodesic_deg(pred1[:, rot_sl], gt1[:, rot_sl]), color="tab:orange")
        axes[3][col].set_ylabel(f"{name} rot err [deg]")
        axes[3][col].set_xlabel("frame")
    grip_ax = axes[0][2].twinx()
    for name, (_, _, grip_ch) in ENTITIES.items():
        if grip_ch is None:
            continue
        grip_ax.step(ts, gt1[:, grip_ch], where="post", alpha=0.35, label=f"{name} grip GT")
        grip_ax.step(ts, pred1[:, grip_ch], where="post", alpha=0.35, ls="--",
                     label=f"{name} grip pred")
    grip_ax.set_ylim(-0.1, 1.1)
    grip_ax.legend(loc="lower right", fontsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"overlay figure -> {out_path}")


def _log_action_horizon(root: str, action: np.ndarray, color) -> None:
    for name, (pos_sl, rot_sl, _grip_ch) in ENTITIES.items():
        pos = action[:, pos_sl]
        rotations = _rot6d_to_matrix(action[:, rot_sl])
        rr.log(f"{root}/{name}/horizon", rr.LineStrips3D([pos], colors=[color], radii=0.003))
        rr.log(f"{root}/{name}/horizon_points", rr.Points3D(pos, colors=[color], radii=0.006))
        rr.log(
            f"{root}/{name}/horizon_z_axes",
            rr.Arrows3D(origins=pos, vectors=rotations[..., :, 2] * 0.04,
                        colors=[color], radii=0.001),
        )
        rr.log(
            f"{root}/{name}/current",
            rr.Transform3D(translation=pos[0], mat3x3=rotations[0]),
            rr.TransformAxes3D(axis_length=0.07),
        )


def _log_first_step_errors(root: str, gt: np.ndarray, pred: np.ndarray) -> None:
    for name, (pos_sl, _rot_sl, _grip_ch) in ENTITIES.items():
        rr.log(
            f"{root}/{name}",
            rr.LineStrips3D([np.stack([gt[0, pos_sl], pred[0, pos_sl]])],
                            colors=[_COLOR_ERR], radii=0.0015),
        )


def render_rerun(bundle: OfflineManiFlowPolicy, source: dict, run: dict,
                 policy_selection: dict,
                 records: list[tuple[int, np.ndarray, np.ndarray]],
                 router_traces: list[np.ndarray], mof: dict | None,
                 save_path: Path | None, connect_url: str | None, stride: int) -> None:
    state = source["state"]
    state_frame = source["state_frame"]
    world_t_base = np.stack([base_pose_to_mat(state[i, _STATE_BASE_BLOCK])
                             for i in range(source["N"])])
    head = np.stack([pos6d_to_mat(state[i, _STATE_HEAD_BLOCK]) for i in range(source["N"])])
    if state_frame == "base":
        head = world_t_base @ head
    fps = float(source["fps"])

    rr.init("vis_wbc_maniflow_prediction",
            spawn=save_path is None and connect_url is None)
    if connect_url is not None:
        rr.connect_grpc(connect_url)
        print(f"[rerun] connected to {connect_url}")
    elif save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(save_path))
        print(f"[rerun] recording -> {save_path}")

    # The depth image back-projects into the 3D view (and tints the head RGB), burying
    # the stored cloud and the action horizons: log it, but leave it sidebar-toggleable.
    hidden_depth = {_HEAD_DEPTH_ENTITY: rrb.EntityBehavior(visible=False)}
    crop_min = np.asarray(source["meta"]["crop_min"], dtype=np.float64)
    crop_max = np.asarray(source["meta"]["crop_max"], dtype=np.float64)
    center = (crop_min + crop_max) / 2.0
    eye_controls = rrb.EyeControls3D(
        position=(center + np.array([-2.0, -2.0, 1.5])).tolist(),
        look_target=center.tolist(),
        eye_up=(0.0, 0.0, 1.0),
    )
    # One 3D view per cloud rendering, all sharing the prediction/scene context: the
    # stored 1024-point tensor vs the run config's own sampler selection (what the
    # encoder actually consumes) vs its mask labels.
    common_3d = ["/world/gt/**", "/world/pred/**", "/world/error/**", "/world/base/**",
                 "/world/base_path/**", "/world/camera"]
    views_3d = [
        rrb.Spatial3DView(origin="world", name="Stored cloud + GT (red) vs pred (magenta)",
                          contents=[*common_3d, "/world/stored/**"],
                          overrides=hidden_depth, eye_controls=eye_controls),
        rrb.Spatial3DView(origin="world",
                          name=f"Policy cloud ({run['visual_cond_len']} pts, "
                               f"{run['point_sampling_mode']})",
                          contents=[*common_3d, "/world/policy_rgb/**"],
                          overrides=hidden_depth, eye_controls=eye_controls),
    ]
    if policy_selection["labels"] is not None:
        views_3d.append(
            rrb.Spatial3DView(origin="world", name="Policy cloud: mask labels",
                              contents=[*common_3d, "/world/policy_masks/**"],
                              overrides=hidden_depth, eye_controls=eye_controls))
    scene = rrb.Vertical(
        rrb.Horizontal(*views_3d),
        rrb.Horizontal(
            rrb.Spatial2DView(origin="world/camera", name="Head RGB / depth",
                              overrides=hidden_depth),
            *[
                rrb.Spatial2DView(origin=f"wrist/{_a}", name=f"{_a.capitalize()} wrist RGB")
                for _a in (
                    ("left", "right") if source["right_wrist_rgb"] is not None else ("left",)
                )
            ],
            *[
                rrb.TimeSeriesView(origin=f"plots/gripper_{side}",
                                   name=f"{side.title()} gripper",
                                   axis_y=rrb.ScalarAxis(range=(-0.2, 1.1), zoom_lock=True))
                for side in ("left", "right")
            ],
        ),
        row_shares=[2.2, 1.4],
        name="Scene",
    )
    tabs = [scene]
    if mof is not None:
        # Same structure as vis_wbc_mof_prediction.py's frames tab: base/left/right are
        # metric frames, *_rel_* are representation spaces.
        tabs.append(rrb.Grid(
            *[
                rrb.Spatial3DView(
                    origin=f"mof_frames/{name}",
                    name=(name if name in ("world", "base", "left", "right")
                          else f"{name} (representation space)"),
                )
                for name in mof["experts"]
            ],
            grid_columns=3,
            name="MoF frames (GT vs pred)",
        ))
    if router_traces:
        tabs.append(rrb.Vertical(
            rrb.TimeSeriesView(origin="router/mean", name="Mean expert weights by chunk",
                               axis_y=rrb.ScalarAxis(range=(0.0, 1.0), zoom_lock=True)),
            rrb.TensorView(origin="router/heatmap", name="Weights by flow step"),
            name="MoF router",
        ))
    layout = rrb.Tabs(*tabs) if len(tabs) > 1 else scene
    rr.send_blueprint(
        rrb.Blueprint(layout, rrb.TimePanel(timeline="chunk", fps=fps / stride),
                      auto_views=False)
    )

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    base_path_xyz = np.column_stack(
        [state[:, 29:31], np.zeros(source["N"])]).astype(np.float32)
    rr.log("world/base_path",
           rr.LineStrips3D([base_path_xyz], colors=[_COLOR_BASE], radii=0.008), static=True)
    for side in ("left", "right"):
        rr.log(
            f"plots/gripper_{side}",
            rr.SeriesLines(colors=[_COLOR_GT, _COLOR_PRED],
                           names=[f"{side} GT", f"{side} pred"], widths=[2.0, 2.0]),
            static=True,
        )
    if router_traces:
        colors = [_ROUTER_COLORS[i % len(_ROUTER_COLORS)]
                  for i in range(len(bundle.expert_names))]
        rr.log("router/mean",
               rr.SeriesLines(colors=colors, names=list(bundle.expert_names),
                              widths=[2.0] * len(bundle.expert_names)),
               static=True)
    if mof is not None:
        for name in mof["experts"]:
            rr.log(f"mof_frames/{name}", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            rr.log(f"mof_frames/{name}/reference_axes",
                   rr.Transform3D(translation=[0.0, 0.0, 0.0], mat3x3=np.eye(3)),
                   rr.TransformAxes3D(axis_length=0.12), static=True)

    for chunk_index, (t, gt, pred) in enumerate(records):
        rr.set_time("chunk", sequence=chunk_index)
        rr.set_time("frame", sequence=t)

        _log_action_horizon("world/gt", gt, _COLOR_GT)
        _log_action_horizon("world/pred", pred, _COLOR_PRED)
        _log_first_step_errors("world/error", gt, pred)

        base_tf = world_t_base[t]
        rr.log("world/base", rr.Transform3D(translation=base_tf[:3, 3], mat3x3=base_tf[:3, :3]),
               rr.TransformAxes3D(axis_length=0.25))

        for entity, cloud in (("world/stored/rgb", source["point_cloud"][t]),
                              ("world/policy_rgb/rgb", policy_selection["points"][t])):
            rr.log(entity, rr.Points3D(
                cloud[:, :3],
                colors=(np.clip(cloud[:, 3:], 0.0, 1.0) * 255.0).astype(np.uint8),
                radii=0.006,
            ))
        if policy_selection["labels"] is not None:
            labels = policy_selection["labels"][t]
            mask_colors = np.empty((len(labels), 3), dtype=np.uint8)
            mask_colors[labels == 0] = _COLOR_MASK_BG
            mask_colors[labels == 1] = _COLOR_MASK_SRC
            mask_colors[labels == 2] = _COLOR_MASK_DST
            rr.log("world/policy_masks/labels", rr.Points3D(
                policy_selection["points"][t][:, :3], colors=mask_colors, radii=0.006))

        world_t_cam = head[t]
        K = source["intrinsic"][t]
        rgb = source["head_rgb"][t]
        height, width = rgb.shape[:2]
        rr.log("world/camera",
               rr.Transform3D(translation=world_t_cam[:3, 3], mat3x3=world_t_cam[:3, :3]))
        rr.log("world/camera",
               rr.Pinhole(image_from_camera=K.astype(np.float32), width=width, height=height))
        rr.log("world/camera/rgb", rr.Image(rgb))
        depth_mm = source["depth"][t]
        if depth_mm.shape != (height, width):
            depth_mm = cv2.resize(depth_mm, (width, height), interpolation=cv2.INTER_NEAREST)
        rr.log("world/camera/depth", rr.DepthImage(depth_mm, meter=1000.0))
        # left/right = the ARM; right is absent on single-wrist takes.
        rr.log("wrist/left/rgb", rr.Image(source["wrist_rgb"][t]))
        if source["right_wrist_rgb"] is not None:
            rr.log("wrist/right/rgb", rr.Image(source["right_wrist_rgb"][t]))

        for tag, chunk, color in (("gt_2d", gt, _COLOR_GT), ("pred_2d", pred, _COLOR_PRED)):
            arm_points = np.concatenate(
                [chunk[:, ENTITIES[side][0]] for side in ("left", "right")], axis=0)
            uv, valid = viewer.project_world_to_pixel(arm_points, K, world_t_cam)
            if valid.any():
                rr.log(f"world/camera/{tag}", rr.Points2D(uv[valid], colors=[color], radii=2.5))
            else:
                rr.log(f"world/camera/{tag}", rr.Clear(recursive=False))

        for side in ("left", "right"):
            grip_ch = ENTITIES[side][2]
            rr.log(f"plots/gripper_{side}",
                   rr.Scalars([float(gt[0, grip_ch]), float(pred[0, grip_ch])]))

        if mof is not None:
            window = state[t - bundle.n_obs_steps + 1 : t + 1]
            representations = mof_representations(
                mof["transforms"], mof["layout"], window,
                np.stack([gt, pred]), mof["experts"])
            for name, (gt_rep, pred_rep) in representations.items():
                _log_action_horizon(f"mof_frames/{name}/gt", gt_rep, _COLOR_GT)
                _log_action_horizon(f"mof_frames/{name}/pred", pred_rep, _COLOR_PRED)
                _log_first_step_errors(f"mof_frames/{name}/error", gt_rep, pred_rep)

        if router_traces:
            trace = router_traces[chunk_index]
            rr.log("router/heatmap", rr.Tensor(trace.T, dim_names=["expert", "flow_step"],
                                               value_range=(0.0, 1.0)))
            rr.log("router/mean", rr.Scalars(trace.mean(axis=0)))

    print(f"[rerun] logged {len(records)} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy-path", required=True,
                        help="ManiFlow workspace .ckpt (e.g. .../checkpoints/epoch=030-*.ckpt)")
    parser.add_argument("--zarr", type=Path, default=_DEFAULT_ZARR,
                        help=f"ported ManiFlow zarr (default {_DEFAULT_ZARR})")
    parser.add_argument("--maniflow-run-dir", type=Path, default=None,
                        help="training run holding .hydra/config.yaml; the sampler / "
                             "visual_cond_len / MoF switches shown come from there "
                             "(default: the checkpoint's own run dir, <ckpt>/../..)")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--train-zarr", type=Path, default=None,
                        help="training zarr the checkpoint's normalizer was fit on "
                             "(default: the cfg path, else the same filename beside --zarr)")
    parser.add_argument("--stride", type=int, default=None,
                        help="frames between evaluated chunks (default n_action_steps)")
    parser.add_argument("--save", type=Path, default=None,
                        help="stream Rerun headlessly to this .rrd instead of spawning a viewer")
    parser.add_argument("--connect", type=str, default=None,
                        help="log to an already-running Rerun viewer over gRPC "
                             "(e.g. rerun+http://127.0.0.1:9876/proxy)")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"overlay PNG path (default under {_DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--seed", type=int, default=0,
                        help="flow-sampling seed for reproducible predictions")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.connect is not None and args.save is not None:
        parser.error("--connect and --save are mutually exclusive")

    bundle = OfflineManiFlowPolicy(
        args.policy_path, device=args.device, use_ema=not args.no_ema,
        num_inference_steps=args.num_inference_steps,
    )
    print(bundle.describe())

    # The run config is the second, independent record of the same switches: it names
    # the sampler the encoder runs, the point budget it keeps, and the MoF experts.
    run_dir = (args.maniflow_run_dir.expanduser().resolve() if args.maniflow_run_dir
               else bundle.checkpoint_path.parent.parent)
    run = viewer.load_maniflow_run(run_dir)
    check_run_config(bundle, run)
    sampler, checkout = viewer.import_run_sampler(run_dir)
    mof = None
    if run["mof"] is not None:
        transforms, layout = viewer.import_run_frame_transforms(checkout)
        mof = {"transforms": transforms, "layout": layout,
               "experts": run["mof"]["enabled_experts"]}
    print(f"run config: {run['config_path']}")
    print(f"  sampler implementation: {Path(sampler.__file__).resolve()}")
    print(f"  point_sampling_mode: {run['point_sampling_mode']} | "
          f"mask_channels: {run['mask_channels']} | "
          f"position_condition_mode: {run['position_condition_mode']}")
    print(f"  visual_cond_len: {run['visual_cond_len']} (of {bundle.num_points} stored) | "
          f"use_pc_color: {run['use_pc_color']}")
    if run["mof"] is not None:
        print(f"  mof experts: {', '.join(run['mof']['enabled_experts'])} "
              f"(canonical {run['mof']['canonical_space']}, "
              f"router {run['mof']['router_mode']})")
    print(f"  ManiFlow checkout: {checkout}")

    zarr_path = args.zarr.expanduser()
    switches = bundle.policy_switches()
    source = viewer.load_maniflow_zarr_episode(zarr_path, args.episode_index, switches)
    if source["point_cloud"].shape[1] != bundle.num_points:
        raise ValueError(
            f"{zarr_path}: {source['point_cloud'].shape[1]} points per frame, but the "
            f"checkpoint expects {bundle.num_points}")
    train_zarr = resolve_train_zarr(bundle, zarr_path, args.train_zarr)
    viewer.validate_preprocessing_match(
        source["meta"], viewer.load_zarr_meta(train_zarr),
        needs_point_mask=switches["needs_point_mask"],
        position_condition_mode=switches["position_condition_mode"],
    )
    print(f"preprocessing matches the training zarr {train_zarr}")
    if bundle.conditioning_spec.needs_env_dino:
        source = attach_env_dino(source, zarr_path, bundle)
    source = viewer.attach_raw_camera_streams(
        source, needs_point_mask=switches["needs_point_mask"])
    print(f"source: {source['label']}")

    # The run's own sampler, fed the checkpoint's own normalization: the exact points
    # the encoder keeps out of the stored cloud.
    scale, offset = bundle.point_cloud_normalization()
    policy_selection = viewer.select_policy_points(
        source, run, sampler, torch.device(bundle.device), scale, offset)
    print(f"  policy point selection: {policy_selection['points'].shape[1]} pts/frame")

    stride = args.stride if args.stride is not None else bundle.n_action_steps
    if stride < 1:
        raise ValueError(f"--stride must be >= 1, got {stride}")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records, router_traces = evaluate(bundle, source, stride)
    report(records)
    if router_traces:
        weights = np.stack(router_traces).reshape(-1, len(bundle.expert_names)).mean(axis=0)
        print("  router mean weights: " + ", ".join(
            f"{name} {weight:.3f}" for name, weight in zip(bundle.expert_names, weights,
                                                           strict=True)))

    run_name = bundle.checkpoint_path.parent.parent.name
    title = (f"WBC ManiFlow [{run_name} / {bundle.checkpoint_path.stem}] "
             f"episode {args.episode_index}: GT vs 1-step-ahead predicted world action")
    out = args.output or (
        _DEFAULT_OUTPUT_DIR
        / f"maniflow_pred_{run_name}_ep{args.episode_index}.png"
    )
    render(records, Path(out).expanduser(), title)
    render_rerun(bundle, source, run, policy_selection, records, router_traces, mof,
                 args.save, args.connect, stride)


if __name__ == "__main__":
    main()
