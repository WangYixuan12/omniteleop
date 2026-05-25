#!/usr/bin/env python3
"""Live bimanual policy rollout with on-the-fly SceneDiff position conditioning.

Thin wrapper around :class:`omniteleop.follower.policy_rollout.PolicyRolloutController`
for position-conditioned policies (those declaring ``observation.environment_state``).
Instead of pointing the rollout at a precomputed ``episode_<N>.npz`` produced offline
from precollected deploy data (``scene_diff/run_deploy_pos_condition.sh``), this script
computes the position condition LIVE, from the actual head observation at the rollout's
start:

  1. Build the controller, which homes the robot (head/torso at the fixed ``INIT_*``
     pose) and starts the head camera.
  2. ``_before_policy_init`` hook (runs AFTER homing, BEFORE the policy loads — so the
     GPU is still free): grab one head frame at that exact start viewpoint, write it as
     a single-frame "live capture" HDF5, then shell out to SceneDiff.
  3. SceneDiff runs in its own interpreter (``scene_diff/.venv-merged``) via
     ``scene_diff/run_live_pos_condition.sh``: it attaches the fixed reference scene's
     extrinsic/intrinsic to the live frame, runs SAM3 change detection against the
     reference, and reduces the change masks to ``episode_0.npz`` (``before_xyz`` =
     detected start positions in the robot base frame, ``after_xyz`` = hardcoded goals).
  4. ``self._positions_npz`` is set to that npz; ``initialize_policy`` then loads it
     through the normal ``_load_position_conditions`` path and the rollout proceeds.

The position condition is computed ONCE at startup and held fixed for the whole episode
(only the per-frame stage mask switches online via the model's stage head) — identical
to training, where the env-state is a per-episode constant.

Two Python environments are involved and cannot share a process: the robot + policy run
in conda ``dexmate_lerobot`` (this script); SceneDiff (SAM3, torch, pycocotools) runs in
``scene_diff/.venv-merged`` (the subprocess). They run sequentially, so the GPU is never
shared.

Run::

    PYTHONPATH=/home/yixuan/lerobot_original/src \\
    python -m omniteleop.follower.live_scenediff_rollout \\
        --policy-path /home/yixuan/Dexmate/model/act/dexmate_eef_eef_abs_2cam_pos_20_10/checkpoints/200000/pretrained_model

A missing npz after SceneDiff means the detection gate failed (wrong object count / too
few valid pixels); the script raises before any arm motion, just like the offline path.
Inspect ``<record_dir>/scene_diff/deploy_before_overlay.png`` and ``.../_work/skip.json``.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from typing import Optional

import h5py
import numpy as np
import tyro
from loguru import logger

from omniteleop.common.logging import setup_logging
from omniteleop.follower.policy_rollout import PolicyRolloutController

# Defaults match scene_diff/run_live_pos_condition.sh + run_deploy_pos_condition.sh.
_DEFAULT_SCENE_DIFF_REPO = "/home/yixuan/scene_diff"
_DEFAULT_SCENE_DIFF_PYTHON = "/home/yixuan/scene_diff/.venv-merged/bin/python"
_DEFAULT_REFERENCE_HDF5 = "/home/yixuan/Dexmate/data/reference_scene/episode_0.hdf5"
_RUN_SCRIPT_NAME = "run_live_pos_condition.sh"

# HDF5 keys read by scene_diff/scripts/make_deploy_before_hdf5.py.
_RGB_KEY = "obs/images/head_left_rgb"
_DEPTH_KEY = "obs/images/head_depth"
_EXTRINSIC_KEY = "obs/images/extrinsic"
_INTRINSIC_KEY = "obs/images/intrinsic"


class LiveSceneDiffRolloutController(PolicyRolloutController):
    """PolicyRolloutController that computes its position condition from the live
    head observation at rollout start (see module docstring)."""

    def __init__(
        self,
        *,
        scene_diff_repo: str = _DEFAULT_SCENE_DIFF_REPO,
        scene_diff_python: str = _DEFAULT_SCENE_DIFF_PYTHON,
        reference_hdf5: str = _DEFAULT_REFERENCE_HDF5,
        sam: str = "sam3",
        obj_num: int = 2,
        crop: tuple[int, int, int, int] = (220, 600, 150, 800),
        scenediff_config: Optional[str] = None,
        scenediff_timeout: float = 1200.0,
        prompt_before_capture: bool = False,
        **rollout_kwargs,
    ) -> None:
        # Stored BEFORE super().__init__() because the base __init__ calls our
        # _before_policy_init() hook, which reads these.
        self._sd_repo = pathlib.Path(scene_diff_repo)
        self._sd_python = scene_diff_python
        self._sd_reference = reference_hdf5
        self._sd_sam = sam
        self._sd_obj_num = int(obj_num)
        self._sd_crop = tuple(int(c) for c in crop)
        self._sd_config = scenediff_config
        self._sd_timeout = float(scenediff_timeout)
        self._sd_prompt_before_capture = bool(prompt_before_capture)
        # live_out_dir is derived in _run_live_scenediff from self._record_dir
        # (resolved by the base class before _before_policy_init is called).

        # env-state conditioning is REQUIRED here — this script exists to produce it.
        # Pass positions_npz=None; the hook fills _positions_npz before the load.
        rollout_kwargs.setdefault("positions_npz", None)
        super().__init__(**rollout_kwargs)

    # ── live SceneDiff hook (runs after home, before policy/env-state load) ──────

    def _before_policy_init(self) -> None:
        if not self._env_state_feature_present():
            raise ValueError(
                "live_scenediff_rollout requires a position-conditioned policy "
                "(declaring observation.environment_state); the given policy has none. "
                "Use policy_rollout.py directly for non-conditioned policies."
            )
        try:
            self._run_live_scenediff()
        except (KeyboardInterrupt, EOFError):
            logger.warning("Live SceneDiff aborted by operator; cleaning up.")
            self.cleanup()
            raise SystemExit(130)

    def _env_state_feature_present(self) -> bool:
        """Peek the policy config for observation.environment_state without loading it."""
        from lerobot.configs import PreTrainedConfig
        from lerobot.utils.constants import OBS_ENV_STATE

        pcfg = PreTrainedConfig.from_pretrained(self._policy_path)
        return OBS_ENV_STATE in pcfg.input_features

    def _run_live_scenediff(self) -> None:
        if self._record_dir is None:
            raise ValueError(
                "live_scenediff_rollout requires record=True (the default) so that "
                "SceneDiff outputs can be placed at <record_dir>/scene_diff/. "
                "Pass --record or remove --no-record."
            )
        out_dir = self._record_dir / "scene_diff"
        out_dir.mkdir(parents=True, exist_ok=True)
        live_hdf5 = out_dir / "live_capture.hdf5"

        if self._sd_prompt_before_capture:
            input(
                "\n>>> Robot homed (head at INIT). Stage the scene, then press ENTER to "
                "capture the head frame and run SceneDiff (Ctrl-C to abort) <<<\n"
            )

        # 1) Capture one head frame at the exact rollout start viewpoint.
        left_rgb, depth_u16 = self.capture_head_frame()

        # Compute the actual FK extrinsic for this specific head pose.
        # initialize_policy() hasn't run yet, so we initialise the MotionManager
        # here to do the FK; initialize_policy() will reinitialise it with the
        # same result (head/torso are at fixed INIT constants throughout rollout).
        from omniteleop.follower.policy_rollout import ZED_K
        self._init_motion_manager()
        live_extrinsic = self._compute_head_camera_extrinsic()
        live_intrinsic = ZED_K.copy()
        logger.info(
            f"Live head-camera FK extrinsic xyz="
            f"[{live_extrinsic[0,3]:.3f}, {live_extrinsic[1,3]:.3f}, {live_extrinsic[2,3]:.3f}]"
        )

        self._write_live_capture(live_hdf5, left_rgb, depth_u16, live_extrinsic, live_intrinsic)
        logger.info(
            f"Live SceneDiff 'before' frame -> {live_hdf5} "
            f"rgb={left_rgb.shape}{left_rgb.dtype} depth={depth_u16.shape}{depth_u16.dtype}"
        )

        # 2) Run SceneDiff in scene_diff/.venv-merged.
        npz = self._invoke_scenediff(live_hdf5, out_dir)

        # 3) Hand the npz to the normal env-state load path.
        self._positions_npz = str(npz)
        logger.success(f"Live SceneDiff position condition ready -> {npz}")

    def _write_live_capture(
        self,
        path: pathlib.Path,
        left_rgb: np.ndarray,
        depth_u16: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
    ) -> None:
        """Write the single live head frame as a deploy-like HDF5.

        Includes the FK-computed extrinsic (world_T_head_camera, (4,4) float32,
        pinhole convention) and intrinsic ((3,3) float32) so
        make_deploy_before_hdf5.py uses the actual camera pose rather than
        copying the reference scene's calibration. A leading frame axis of 1 is
        added to RGB/depth so the reader's ``rgb_ds[frame]`` indexing works;
        extrinsic/intrinsic are stored without a frame axis (they are constant
        for the whole capture and extract_first_hdf5_rgb_depth handles both).
        """
        if left_rgb.ndim != 3 or left_rgb.shape[2] != 3:
            raise ValueError(f"head rgb must be (H, W, 3), got {left_rgb.shape}")
        if depth_u16.ndim != 2:
            raise ValueError(f"head depth must be (H, W), got {depth_u16.shape}")
        if left_rgb.shape[:2] != depth_u16.shape:
            raise ValueError(
                f"rgb {left_rgb.shape[:2]} != depth {depth_u16.shape} (must match)"
            )
        if extrinsic.shape != (4, 4):
            raise ValueError(f"extrinsic must be (4, 4), got {extrinsic.shape}")
        if intrinsic.shape != (3, 3):
            raise ValueError(f"intrinsic must be (3, 3), got {intrinsic.shape}")
        rgb = np.ascontiguousarray(left_rgb, dtype=np.uint8)[None, ...]
        depth = np.ascontiguousarray(depth_u16, dtype=np.uint16)[None, ...]
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.create_dataset(_RGB_KEY, data=rgb)
            f.create_dataset(_DEPTH_KEY, data=depth)
            f.create_dataset(_EXTRINSIC_KEY, data=extrinsic.astype(np.float32))
            f.create_dataset(_INTRINSIC_KEY, data=intrinsic.astype(np.float32))

    def _invoke_scenediff(
        self, live_hdf5: pathlib.Path, out_dir: pathlib.Path
    ) -> pathlib.Path:
        """Run run_live_pos_condition.sh in .venv-merged and return the npz path."""
        script = self._sd_repo / _RUN_SCRIPT_NAME
        if not script.exists():
            raise FileNotFoundError(f"SceneDiff driver not found: {script}")
        if not pathlib.Path(self._sd_python).exists():
            raise FileNotFoundError(
                f"SceneDiff interpreter not found: {self._sd_python} "
                "(expected scene_diff/.venv-merged)"
            )

        # Clean env: drop the rollout's PYTHONPATH (e.g. lerobot_original/src) and
        # PYTHONHOME so they cannot shadow the .venv-merged interpreter's modules.
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["PYTHON"] = self._sd_python
        env["REFERENCE"] = self._sd_reference
        env["SAM"] = self._sd_sam
        env["OBJ_NUM"] = str(self._sd_obj_num)
        ct, cb, cl, cr = self._sd_crop
        env["CROP_TOP"], env["CROP_BOTTOM"] = str(ct), str(cb)
        env["CROP_LEFT"], env["CROP_RIGHT"] = str(cl), str(cr)
        if self._sd_config:
            env["CONFIG"] = self._sd_config

        cmd = ["bash", str(script), str(live_hdf5), str(out_dir)]
        logger.info(f"Running SceneDiff: PYTHON={self._sd_python} {' '.join(cmd)}")
        # Inherit stdio so SAM3 progress is visible; enforce a wall-clock timeout.
        result = subprocess.run(
            cmd, env=env, cwd=str(self._sd_repo), timeout=self._sd_timeout
        )
        npz = out_dir / "episode_0.npz"
        if result.returncode != 0:
            raise RuntimeError(
                f"SceneDiff failed (exit {result.returncode}); see output above and "
                f"scratch under {out_dir / '_work'}"
            )
        if not npz.exists():
            raise FileNotFoundError(
                "SceneDiff produced no position condition — the detection gate failed "
                f"(wrong object count / too few valid pixels). Inspect "
                f"{out_dir / 'deploy_before_overlay.png'} and "
                f"{out_dir / '_work' / 'skip.json'}, fix the scene/params, and rerun."
            )
        return npz


# ── CLI entry point ──────────────────────────────────────────────────────────


def main(
    policy_path: str,
    namespace: str = "",
    debug: bool = False,
    config_name: Optional[str] = None,
    workspace_check: bool = True,
    policy_fps: int = 15,
    gripper_threshold: float = 0.5,
    act_n_action_steps: Optional[int] = None,
    act_temporal_ensemble_coeff: Optional[float] = None,
    record: bool = True,
    record_dir: Optional[str] = None,
    scene_diff_repo: str = _DEFAULT_SCENE_DIFF_REPO,
    scene_diff_python: str = _DEFAULT_SCENE_DIFF_PYTHON,
    reference_hdf5: str = _DEFAULT_REFERENCE_HDF5,
    sam: str = "sam3",
    obj_num: int = 2,
    crop: tuple[int, int, int, int] = (220, 600, 150, 800),
    scenediff_config: Optional[str] = None,
    scenediff_timeout: float = 1200.0,
    prompt_before_capture: bool = False,
) -> None:
    """Run live bimanual rollout, computing the SceneDiff position condition at startup.

    Forwards the rollout args to :class:`PolicyRolloutController`; the rest configure
    the live SceneDiff step.

    Args:
        policy_path: LeRobot ``pretrained_model`` dir. MUST declare
            ``observation.environment_state`` (a position-conditioned policy).
        namespace, debug, config_name, workspace_check, policy_fps, gripper_threshold,
        act_n_action_steps, act_temporal_ensemble_coeff, record, record_dir:
            Same as ``policy_rollout.main`` (see that module). SceneDiff outputs
            (live_capture.hdf5, episode_0.npz, overlays, _work/) are written to
            ``<record_dir>/scene_diff/`` automatically; ``record`` must be True
            (the default).
        scene_diff_repo: Path to the scene_diff repo holding run_live_pos_condition.sh.
        scene_diff_python: Interpreter for SceneDiff (scene_diff/.venv-merged/bin/python).
        reference_hdf5: Fixed reference ("after"/moved) scene episode HDF5.
        sam: SAM version for change detection ({sam1, sam3, sam3.1}). Default sam3.
        obj_num: Required number of changed objects (== policy stage classes). Default 2.
        crop: (top, bottom, left, right) head-pixel crop before SceneDiff. Must match the
            training crop used to build the reference positions. Default (220,600,150,800).
        scenediff_config: Override the SceneDiff config path (relative to scene_diff
            repo). Default: chosen by ``sam`` inside the shell driver.
        scenediff_timeout: Wall-clock timeout (s) for the SceneDiff subprocess.
        prompt_before_capture: If set, pause for ENTER after homing and before capturing
            the head frame (lets the operator stage the scene with the arms at INIT).
            Default off — the operator stages before launch and inspects the detection
            overlay at the rollout's own ENTER gate.
    """
    setup_logging(debug)
    ctrl = LiveSceneDiffRolloutController(
        policy_path=policy_path,
        namespace=namespace,
        debug=debug,
        config_name=config_name,
        workspace_check=workspace_check,
        policy_fps=policy_fps,
        gripper_threshold=gripper_threshold,
        act_n_action_steps=act_n_action_steps,
        act_temporal_ensemble_coeff=act_temporal_ensemble_coeff,
        record=record,
        record_dir=record_dir,
        scene_diff_repo=scene_diff_repo,
        scene_diff_python=scene_diff_python,
        reference_hdf5=reference_hdf5,
        sam=sam,
        obj_num=obj_num,
        crop=crop,
        scenediff_config=scenediff_config,
        scenediff_timeout=scenediff_timeout,
        prompt_before_capture=prompt_before_capture,
    )
    ctrl.run()


if __name__ == "__main__":
    sys.exit(tyro.cli(main))
