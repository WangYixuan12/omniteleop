"""MoF (Mixture of Frames) checkpoint bundle for the WBC deploy pipeline.

Loads a mofpo ``train_mof_moe_wbc`` checkpoint and exposes the same bundle
surface as ``wbc_policy_rollout._PolicyBundle`` / ``wbc_maniflow_rollout
.ManiFlowBundle``: ``reset()``, ``predict_chunk(obs_history)`` returning an
``(n_action_steps, 29)`` ABSOLUTE world-frame action chunk, plus observation
builders for the offline (ported safetensors) and live (state + camera) paths.

Deploy-safety gates, all enforced at load time (before any robot motion):

- the checkpoint must be a WBC MoF checkpoint: 29-D action, ``action_layout:
  wbc``, and ``external_rot6d_convention: column`` — a row-convention
  checkpoint would emit transposed rotations that the 29-D world decode
  silently accepts;
- the training port's ``meta.json`` (source of truth) must match this
  pipeline: schema, column rot6d, wxyz quats, fps, 224x224 anisotropic squash;
- the checkpoint's normalizer must carry fitted statistics.

Environment: ``dexmate_mof`` (a dexmate_lerobot clone + editable mofpo +
robomimic/threadpoolctl + the ManiFlow env's pytorch3d build).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from omniteleop.wbc_policy_format import ACTION_AXES, STATE_AXES

EXPECTED_SCHEMA = "wbc_mof_safetensors_v1"
IMAGE_HW = (224, 224)
# World-frame 32-D state blocks (see wbc_policy_format.STATE_AXES).
_ENTITY_STATE_BLOCKS = {
    "left_ee": (slice(0, 3), slice(3, 9)),
    "right_ee": (slice(10, 13), slice(13, 19)),
    "head": (slice(20, 23), slice(23, 29)),
}
_GRIPPER_STATE_DIMS = (9, 19)
# Obs keys the policy's runtime path consumes, in the porter's naming.
OBS_KEYS = (
    "head_image",
    "wrist_image",
    "left_ee_pos",
    "left_ee_quat",
    "right_ee_pos",
    "right_ee_quat",
    "head_pos",
    "head_quat",
    "base_pos",
    "base_quat",
    "proprioception_grippers",
)


def _rot6d_columns_to_quat_wxyz(rot6d: np.ndarray) -> np.ndarray:
    """(N, 6) column-major rot6d -> (N, 4) wxyz quats (same math as the porter)."""
    from scipy.spatial.transform import Rotation

    c1, c2 = rot6d[:, 0:3], rot6d[:, 3:6]
    R = np.stack([c1, c2, np.cross(c1, c2)], axis=-1)
    xyzw = Rotation.from_matrix(R).as_quat()
    return np.concatenate([xyzw[:, 3:4], xyzw[:, 0:3]], axis=1).astype(np.float32)


class MoFObservation:
    """One raw (unbatched) MoF observation in the porter's key naming."""

    __slots__ = ("arrays", "state")

    def __init__(self, arrays: dict[str, np.ndarray], state: np.ndarray | None = None):
        missing = [k for k in OBS_KEYS if k not in arrays]
        if missing:
            raise ValueError(f"MoF observation missing keys: {missing}")
        self.arrays = arrays
        self.state = state  # (32,) world state for the io_log, when available


class MoFBundle:
    """mofpo WBC checkpoint in the shared rollout bundle shape."""

    # MoF's checkpoint schema has a single ``wrist_image`` (see OBS_KEYS), which is the
    # LEFT arm's camera. Declared per-arm so the shared inference worker's freshness
    # gate grabs exactly that stream; widening MoF to both wrists would change OBS_KEYS
    # and invalidate existing checkpoints.
    wrist_arms: tuple[str, ...] = ("left",)
    # No SceneDiff position conditioning in v1 checkpoints; the shared rollout
    # cross-checks this against --position-condition.
    position_condition_mode = "none"
    use_env_state = False
    env_state_dim = 0
    # The MoF policy consumes WORLD-frame entity poses (its runtime path
    # re-expresses them per frame internally).
    state_frame = "world"

    def __init__(
        self,
        policy_path: str,
        *,
        device: str | None = None,
        use_ema: bool = True,
        num_inference_steps: int | None = None,
        port_meta_path: str | None = None,
    ) -> None:
        import dill  # -- heavy, GPU path only
        import hydra
        import torch
        from omegaconf import OmegaConf

        OmegaConf.register_new_resolver("eval", eval, replace=True)
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = Path(policy_path)
        if not ckpt.is_file():
            raise ValueError(
                f"--policy-path must be a mofpo .ckpt FILE, got {ckpt} "
                "(e.g. .../checkpoints/latest.ckpt)"
            )
        payload = torch.load(ckpt, map_location="cpu", pickle_module=dill, weights_only=False)
        for key in ("cfg", "state_dicts"):
            if key not in payload:
                raise ValueError(f"{ckpt}: not a mofpo workspace checkpoint (missing '{key}')")
        cfg = payload["cfg"]

        # ---- checkpoint schema gates -------------------------------------
        action_dim = int(cfg.shape_meta["action"]["shape"][0])
        if action_dim != len(ACTION_AXES):
            raise ValueError(f"{ckpt}: action dim {action_dim} != WBC {len(ACTION_AXES)}")
        layout = str(cfg.policy.get("action_layout", "bigym"))
        if layout != "wbc":
            raise ValueError(
                f"{ckpt}: policy.action_layout {layout!r} != 'wbc' -- not a WBC checkpoint"
            )
        convention = str(cfg.policy.get("external_rot6d_convention", "row"))
        if convention != "column":
            raise ValueError(
                f"{ckpt}: external_rot6d_convention {convention!r}; a WBC deploy needs "
                "'column' -- a row checkpoint would emit transposed rotations"
            )

        # ---- training-port meta guard ------------------------------------
        self.port_meta_path = (
            Path(port_meta_path) if port_meta_path else Path(str(cfg.demos_dir)) / "meta.json"
        )
        self._validate_port_meta(self.port_meta_path)

        # ---- rebuild the policy through the workspace payload ------------
        cls = hydra.utils.get_class(cfg._target_)
        import tempfile

        self._workspace_dir = tempfile.mkdtemp(prefix="wbc_mof_bundle_")
        workspace = cls(cfg, output_dir=self._workspace_dir)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        use_ema = bool(use_ema and cfg.training.use_ema)
        self.policy = workspace.ema_model if use_ema else workspace.model
        self.weights = "ema_model" if use_ema else "model"

        if "action" not in self.policy.normalizer.params_dict:
            raise ValueError(
                f"{ckpt}: the {self.weights} normalizer has no fitted stats -- the "
                "checkpoint was saved before set_normalizer(); it cannot un-normalize"
            )
        self.policy.to(self.device).eval()

        if num_inference_steps is not None:
            if num_inference_steps < 1:
                raise ValueError(f"--num-inference-steps must be >= 1, got {num_inference_steps}")
            self.policy.num_inference_steps = int(num_inference_steps)
        self.num_inference_steps = int(self.policy.num_inference_steps)
        self.n_obs_steps = int(self.policy.n_obs_steps)
        self.n_action_steps = int(self.policy.n_action_steps)
        self.dataset_fps = float(self._port_meta.get("fps", 10))
        self.epoch = payload.get("epoch")

    def _validate_port_meta(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"training port meta not found: {path}. It records the preprocessing the "
                "checkpoint was trained with; deploying without it would guess. Pass "
                "--port-meta if the ported dataset moved."
            )
        meta = json.loads(path.read_text())
        checks = {
            "schema": EXPECTED_SCHEMA,
            "rotation_convention": "column",
            "quaternion_convention": "wxyz",
            "image_resize": "anisotropic",
        }
        for key, expected in checks.items():
            if meta.get(key) != expected:
                raise ValueError(f"{path}: {key} = {meta.get(key)!r}, expected {expected!r}")
        if tuple(meta.get("image_resolution", ())) != IMAGE_HW:
            raise ValueError(
                f"{path}: image_resolution {meta.get('image_resolution')} != {list(IMAGE_HW)}"
            )
        self._port_meta = meta

    # ------------------------------------------------------------------ #
    # observation builders
    # ------------------------------------------------------------------ #

    @staticmethod
    def obs_from_port_arrays(arrays: dict[str, np.ndarray], t: int) -> MoFObservation:
        """Frame ``t`` of a ported safetensors episode -> one observation.

        The stored tensors ARE the training observation — no reprocessing.
        """
        return MoFObservation({key: np.asarray(arrays[key][t]) for key in OBS_KEYS})

    @staticmethod
    def obs_from_live(
        state: np.ndarray, head_rgb: np.ndarray, wrist_rgb: np.ndarray
    ) -> MoFObservation:
        """Live ``(world state (32,), head_rgb, wrist_rgb)`` -> one observation.

        Mirrors the porter exactly: per-entity world pos + wxyz quat from the
        state blocks, planar base lifted to a 3-D pose, raw grippers, and the
        224x224 anisotropic INTER_AREA squash.
        """
        import cv2

        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (len(STATE_AXES),):
            raise ValueError(f"state must be ({len(STATE_AXES)},), got {state.shape}")
        arrays: dict[str, np.ndarray] = {}
        for cam_key, frame in (("head_image", head_rgb), ("wrist_image", wrist_rgb)):
            frame = np.asarray(frame)
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                raise ValueError(
                    f"{cam_key} must be (H,W,3) uint8, got {frame.shape} {frame.dtype}"
                )
            arrays[cam_key] = cv2.resize(
                frame, (IMAGE_HW[1], IMAGE_HW[0]), interpolation=cv2.INTER_AREA
            )
        for name, (pos_sl, rot_sl) in _ENTITY_STATE_BLOCKS.items():
            arrays[f"{name}_pos"] = state[pos_sl]
            arrays[f"{name}_quat"] = _rot6d_columns_to_quat_wxyz(state[rot_sl][None])[0]
        x, y, yaw = float(state[29]), float(state[30]), float(state[31])
        arrays["base_pos"] = np.array([x, y, 0.0], dtype=np.float32)
        arrays["base_quat"] = np.array(
            [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32
        )
        arrays["proprioception_grippers"] = state[list(_GRIPPER_STATE_DIMS)]
        return MoFObservation(arrays, state=state)

    # ------------------------------------------------------------------ #
    # bundle surface
    # ------------------------------------------------------------------ #

    def describe(self) -> str:
        """One-line summary printed at load time."""
        canonical = self.policy._canonical_space  # noqa: SLF001
        return (
            f"MoF {type(self.policy).__name__} ({self.weights}, epoch={self.epoch}) on "
            f"{self.device}; experts={tuple(self.policy.model.enabled_experts)} "
            f"canonical={canonical} layout=wbc column-rot6d; "
            f"n_obs_steps={self.n_obs_steps} n_action_steps={self.n_action_steps} "
            f"num_inference_steps={self.num_inference_steps}; port_meta={self.port_meta_path}"
        )

    def reset(self) -> None:
        """No-op: the MoF policy is stateless between chunks."""

    def set_env_state(self, vec: np.ndarray) -> None:
        """v1 checkpoints are unconditioned; the shared rollout must not call this."""
        raise RuntimeError("v1 MoF checkpoints are unconditioned; drop --position-condition")

    def predict_chunk(self, obs_history: "list") -> np.ndarray:
        """Oldest-first observation history -> ``(n_action_steps, 29)`` world chunk.

        Accepts either prebuilt :class:`MoFObservation` items (the offline vis
        path) or the shared inference worker's raw observations (anything with
        ``.state`` / ``.head_rgb`` / ``.wrist_rgb``), which are converted with
        :meth:`obs_from_live`. A shorter history is left-padded by repeating the
        oldest sample (the same episode-start bootstrap as the sibling
        bundles). The output is COLUMN-convention rot6d, exactly the recorded
        action format ``split_policy_action`` decodes.
        """
        history = [
            o
            if isinstance(o, MoFObservation)
            # o.wrist_rgb is keyed by arm; MoF's single wrist_image is the left one.
            else self.obs_from_live(o.state, o.head_rgb, o.wrist_rgb["left"])
            for o in obs_history
        ]
        if not 1 <= len(history) <= self.n_obs_steps:
            raise ValueError(
                f"obs_history must hold 1..{self.n_obs_steps} samples, got {len(history)}"
            )
        history = [history[0]] * (self.n_obs_steps - len(history)) + history

        torch = self._torch
        obs_dict = {}
        for key in OBS_KEYS:
            stacked = np.stack([o.arrays[key] for o in history])  # (To, ...)
            if key.endswith("_image"):
                stacked = np.moveaxis(stacked, -1, 1).astype(np.float32) / 255.0
            else:
                stacked = stacked.astype(np.float32)
            obs_dict[key] = torch.from_numpy(np.ascontiguousarray(stacked)[None]).to(self.device)
        with torch.no_grad():
            result = self.policy.predict_action(obs_dict)
        chunk = result["action"]
        if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[1] != self.n_action_steps:
            raise ValueError(
                f"predict_action returned {tuple(chunk.shape)}; expected "
                f"(1, {self.n_action_steps}, {len(ACTION_AXES)})"
            )
        out = np.asarray(chunk.detach().cpu().numpy(), dtype=np.float32)[0]
        if out.shape != (self.n_action_steps, len(ACTION_AXES)) or not np.all(np.isfinite(out)):
            raise ValueError(f"bad action chunk {out.shape} (or non-finite)")
        return out
