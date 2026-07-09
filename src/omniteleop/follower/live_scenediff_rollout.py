#!/usr/bin/env python3
"""Live policy rollout (bimanual or single-arm) with on-the-fly SceneDiff position conditioning.

Thin wrapper around :class:`omniteleop.follower.policy_rollout.PolicyRolloutController`
for position-conditioned policies (those declaring ``observation.environment_state``).
Bimanual (16/20-D) and single-arm (8/10-D) policies are both supported; a single-arm
policy requires ``--arm-side left|right`` (forwarded to the base controller), which is
validated up front — before the SceneDiff subprocess runs.
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
     reference, and reduces the change masks to ``episode_0.npz`` holding the size-ordered
     changed-object ``positions`` (robot base frame) plus a task ``order`` (the
     ``--pos-cond-matching`` permutation: ``size`` from conditions.json, or ``prompt`` =
     identity + a size-slot LABEL overlay for the operator).
  4. ``_resolve_condition_npz`` converts ``positions[order]`` into the ``before_xyz``
     (sources) / ``after_xyz`` (goals) layout the base ``_load_position_conditions`` reads
     and writes ``episode_0_condition.npz``; in ``prompt`` mode with ``--prompt-after-capture``
     the operator first picks ``order`` by inspecting the overlay (opened in a background OS
     image viewer when a display is available, else printed for manual opening). The object
     count is ``2 x stage_prediction_num_classes`` (the env-state feature is the collapsed
     6-D slot, so it does NOT encode the object count), overriding ``--obj-num``.
     ``self._positions_npz`` is set to the converted npz and the rollout proceeds.

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

import pathlib
import sys
from typing import Optional

import numpy as np
import tyro
from loguru import logger

from omniteleop.common.head_camera import ZED_K
from omniteleop.common.logging import setup_logging
from omniteleop.follower.policy_rollout import PolicyRolloutController
from omniteleop.follower.scenediff_live import (
    invoke_scenediff,
    prompt_for_slot_order,
    write_live_capture_hdf5,
)

# Defaults match scene_diff/run_live_pos_condition.sh + run_deploy_pos_condition.sh.
_DEFAULT_SCENE_DIFF_REPO = "/home/yixuan/scene_diff"
_DEFAULT_SCENE_DIFF_PYTHON = "/home/yixuan/scene_diff/.venv-merged/bin/python"
_DEFAULT_REFERENCE_HDF5 = "/home/yixuan/Dexmate/data/reference_scene/episode_0.hdf5"


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
        scenediff_config: Optional[str] = None,
        scenediff_timeout: float = 1200.0,
        prompt_before_capture: bool = False,
        pos_cond_matching: str = "size",
        prompt_after_capture: bool = False,
        **rollout_kwargs,
    ) -> None:
        # Stored BEFORE super().__init__() because the base __init__ calls our
        # _before_policy_init() hook, which reads these.
        self._sd_repo = pathlib.Path(scene_diff_repo)
        self._sd_python = scene_diff_python
        self._sd_reference = reference_hdf5
        self._sd_sam = sam
        self._sd_obj_num = int(obj_num)
        self._sd_config = scenediff_config
        self._sd_timeout = float(scenediff_timeout)
        self._sd_prompt_before_capture = bool(prompt_before_capture)
        self._sd_pos_cond_matching = str(pos_cond_matching)
        self._sd_prompt_after_capture = bool(prompt_after_capture)
        # Derived in _before_policy_init from the policy's env-state dim (overrides obj_num).
        self._sd_object_nums: Optional[int] = None
        self._sd_stage_classes: Optional[int] = None
        # live_out_dir is derived in _run_live_scenediff from self._record_dir
        # (resolved by the base class before _before_policy_init is called).

        # env-state conditioning is REQUIRED here — this script exists to produce it.
        # Pass positions_npz=None; the hook fills _positions_npz before the load.
        rollout_kwargs.setdefault("positions_npz", None)
        super().__init__(**rollout_kwargs)

    # ── live SceneDiff hook (runs after home, before policy/env-state load) ──────

    def _before_policy_init(self) -> None:
        from lerobot.configs import PreTrainedConfig
        from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE

        # Peek the policy config (cheap; no model load) to validate prerequisites BEFORE
        # the minutes-long SceneDiff subprocess runs.
        pcfg = PreTrainedConfig.from_pretrained(self._policy_path)
        if OBS_ENV_STATE not in pcfg.input_features:
            raise ValueError(
                "live_scenediff_rollout requires a position-conditioned policy "
                "(declaring observation.environment_state); the given policy has none. "
                "Use policy_rollout.py directly for non-conditioned policies."
            )
        # A single-arm policy (8/10-D) needs --arm-side. Fail now rather than after the
        # expensive SceneDiff run (the base initialize_policy() also enforces this).
        # Bimanual policies (16/20-D) ignore --arm-side.
        state_dim = int(pcfg.input_features[OBS_STATE].shape[0])
        if state_dim in (8, 10) and self._arm_side_flag is None:
            raise ValueError(
                f"single-arm policy (state_dim={state_dim}) requires --arm-side "
                "left|right; the side is not stored in the checkpoint config"
            )

        # Derive the SceneDiff object count from the policy's STAGE count, NOT the env-state
        # feature dim. The stored env_state (object_nums*3 = stage_classes*6) is COLLAPSED to
        # the active stage's 6-D [source xyz, goal xyz] before the model, so the env-state
        # feature is a fixed 6-D that does NOT encode the object count (env_state_dim//3 would
        # wrongly give 2). Each stage pairs one source object with one goal object, so
        # object_nums = 2 x stage_prediction_num_classes. Override the CLI obj_num to match.
        stage_classes = int(getattr(pcfg, "stage_prediction_num_classes", 2))
        if stage_classes < 1:
            raise ValueError(
                f"policy stage_prediction_num_classes={stage_classes} must be >= 1"
            )
        self._sd_stage_classes = stage_classes
        self._sd_object_nums = 2 * stage_classes
        env_state_dim = int(pcfg.input_features[OBS_ENV_STATE].shape[0])
        if env_state_dim not in (6, self._sd_object_nums * 3):
            logger.warning(
                f"env-state feature dim {env_state_dim} is neither the collapsed 6 nor the "
                f"stored {self._sd_object_nums * 3} (= object_nums*3); proceeding with "
                f"object_nums={self._sd_object_nums} from the stage count."
            )
        if self._sd_object_nums != self._sd_obj_num:
            logger.warning(
                f"overriding obj_num={self._sd_obj_num} with {self._sd_object_nums} = "
                f"2 x stage_prediction_num_classes ({stage_classes}); SceneDiff detects all "
                "source+goal objects."
            )
            self._sd_obj_num = self._sd_object_nums

        try:
            self._run_live_scenediff()
        except (KeyboardInterrupt, EOFError):
            logger.warning("Live SceneDiff aborted by operator; cleaning up.")
            self.cleanup()
            raise SystemExit(130)

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
        self._init_motion_manager()
        live_extrinsic = self._compute_head_camera_extrinsic()
        # Intrinsic of the publisher's cropped+resized frame (capture_head_frame returns
        # it as-is) — the shared ZED_K, identical to the reference scene's stored
        # intrinsic, which SceneDiff needs to align the live "before" frame.
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

        # 3) Convert SceneDiff's size-ordered positions+order npz into the before_xyz/
        #    after_xyz npz the base _load_position_conditions reads (prompting the operator
        #    for the order first, in prompt mode), then hand THAT to the env-state load path.
        condition_npz = self._resolve_condition_npz(npz, out_dir)
        self._positions_npz = str(condition_npz)
        logger.success(f"Live SceneDiff position condition ready -> {condition_npz}")

    # ── positions+order npz -> before_xyz/after_xyz (+ optional operator prompt) ──

    def _resolve_condition_npz(
        self, episode_npz: pathlib.Path, out_dir: pathlib.Path
    ) -> pathlib.Path:
        """Convert SceneDiff's size-ordered ``positions`` + task ``order`` npz into the
        ``before_xyz``/``after_xyz`` npz that the base ``_load_position_conditions`` reads.

        The policy's env-state is ``positions[order]`` arranged as per-stage [source, goal]
        pairs, so ``before_xyz = arranged[0::2]`` (sources) and ``after_xyz = arranged[1::2]``
        (goals), each ``(stage_classes, 3)`` — exactly what the loader validates against the
        policy's ``stage_num_classes``. In ``prompt`` mode with ``--prompt-after-capture`` the
        operator overrides ``order`` by inspecting the size-slot overlay first.
        """
        object_nums = int(self._sd_object_nums)
        with np.load(episode_npz, allow_pickle=True) as data:
            if "positions" not in data.files or "order" not in data.files:
                raise ValueError(
                    f"{episode_npz}: expected `positions`+`order` (got {list(data.files)}); "
                    "regenerate with the current extract_object_positions.py"
                )
            positions = np.asarray(data["positions"], dtype=np.float32)
            order = [int(x) for x in np.asarray(data["order"]).reshape(-1)]

        if positions.shape != (object_nums, 3):
            raise ValueError(
                f"{episode_npz}: positions {tuple(positions.shape)} != ({object_nums}, 3) "
                "(object count derived from the policy's env-state dim)"
            )
        if not np.all(np.isfinite(positions)):
            raise ValueError(f"{episode_npz}: positions contains non-finite values")

        if self._sd_pos_cond_matching == "prompt" and self._sd_prompt_after_capture:
            order = self._prompt_for_order(object_nums, out_dir / "deploy_slot_overlay.png")
        elif self._sd_pos_cond_matching == "prompt":
            logger.warning(
                "pos_cond_matching=prompt but --prompt-after-capture is off; using the npz's "
                "identity (size) order with no operator input."
            )

        if sorted(order) != list(range(object_nums)):
            raise ValueError(
                f"{episode_npz}: order {order} is not a permutation of range({object_nums})"
            )

        arranged = positions[order]            # [s1_src, s1_dst, s2_src, s2_dst, ...]
        before_xyz = arranged[0::2]            # sources -> per-stage "before"
        after_xyz = arranged[1::2]             # goals   -> per-stage "after"
        stage_classes = object_nums // 2
        if before_xyz.shape != (stage_classes, 3) or after_xyz.shape != (stage_classes, 3):
            raise ValueError(
                f"{episode_npz}: split before={tuple(before_xyz.shape)} "
                f"after={tuple(after_xyz.shape)} != ({stage_classes}, 3)"
            )

        out = out_dir / "episode_0_condition.npz"
        np.savez(
            out,
            before_xyz=before_xyz.astype(np.float32),
            after_xyz=after_xyz.astype(np.float32),
            order=np.asarray(order, dtype=np.int64),
            positions=positions,
        )
        logger.success(
            f"Converted env-state -> {out}  order={order}  "
            f"sources={before_xyz.tolist()}  goals={after_xyz.tolist()}"
        )
        return out

    def _prompt_for_order(self, object_nums: int, overlay_path: pathlib.Path) -> list[int]:
        """Show the size-slot overlay and read the task order from the operator.

        Returns ``order`` (length ``object_nums``): ``order[k]`` is the size-slot index
        assigned to task slot k of ``[s1_src, s1_dst, s2_src, s2_dst, ...]``. Re-asks until
        the answers form a permutation of ``range(object_nums)``.
        """
        return prompt_for_slot_order(object_nums, overlay_path, log=logger.info)

    def _write_live_capture(
        self,
        path: pathlib.Path,
        left_rgb: np.ndarray,
        depth_u16: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
    ) -> None:
        """Write the single live head frame as a deploy-like HDF5.

        Includes the FK-computed extrinsic (world_T_head_camera, (4,4) float32, pinhole
        convention) and intrinsic ((3,3) float32) so make_deploy_before_hdf5.py uses the
        actual camera pose rather than copying the reference scene's calibration. The
        layout is owned by :mod:`omniteleop.follower.scenediff_live`.
        """
        write_live_capture_hdf5(path, left_rgb, depth_u16, extrinsic, intrinsic)

    def _invoke_scenediff(
        self, live_hdf5: pathlib.Path, out_dir: pathlib.Path
    ) -> pathlib.Path:
        """Run run_live_pos_condition.sh in .venv-merged and return the npz path."""
        return invoke_scenediff(
            repo=self._sd_repo,
            python=self._sd_python,
            live_hdf5=live_hdf5,
            out_dir=out_dir,
            reference=self._sd_reference,
            sam=self._sd_sam,
            obj_num=self._sd_obj_num,
            matching=self._sd_pos_cond_matching,
            config=self._sd_config or None,
            timeout=self._sd_timeout,
            log=logger.info,
        )


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
    scenediff_config: Optional[str] = None,
    scenediff_timeout: float = 1200.0,
    prompt_before_capture: bool = False,
    pos_cond_matching: str = "size",
    prompt_after_capture: bool = False,
    arm_side: Optional[str] = None,
) -> None:
    """Run a live rollout (bimanual or single-arm), computing the SceneDiff condition at startup.

    Forwards the rollout args to :class:`PolicyRolloutController`; the rest configure
    the live SceneDiff step.

    Args:
        policy_path: LeRobot ``pretrained_model`` dir. MUST declare
            ``observation.environment_state`` (a position-conditioned policy).
        namespace, debug, config_name, workspace_check, policy_fps, gripper_threshold,
        act_n_action_steps, act_temporal_ensemble_coeff, record, record_dir, arm_side:
            Same as ``policy_rollout.main`` (see that module). ``arm_side`` is REQUIRED
            for a single-arm policy (8/10-D) and ignored for bimanual; it is validated up
            front, before the SceneDiff subprocess runs. The publisher crops+resizes
            robot-side, so the captured "before" frame is the model-view (120x160) frame
            the reference scene stores, which SceneDiff requires. SceneDiff outputs
            (live_capture.hdf5, episode_0.npz, overlays, _work/) are written to
            ``<record_dir>/scene_diff/`` automatically; ``record`` must be True
            (the default).
        scene_diff_repo: Path to the scene_diff repo holding run_live_pos_condition.sh.
        scene_diff_python: Interpreter for SceneDiff (scene_diff/.venv-merged/bin/python).
        reference_hdf5: Fixed reference ("after"/moved) scene episode HDF5.
        sam: SAM version for change detection ({sam1, sam3, sam3.1}). Default sam3.
        obj_num: Hint for the number of changed objects SceneDiff should detect. OVERRIDDEN
            at startup by object_nums = 2 x stage_prediction_num_classes from the checkpoint
            (the env-state feature is the collapsed 6-D slot and does NOT encode the object
            count), so it rarely needs setting.
        scenediff_config: Override the SceneDiff config path (relative to scene_diff
            repo). Default: chosen by ``sam`` inside the shell driver.
        scenediff_timeout: Wall-clock timeout (s) for the SceneDiff subprocess.
        prompt_before_capture: If set, pause for ENTER after homing and before capturing
            the head frame (lets the operator stage the scene with the arms at INIT).
            Default off — the operator stages before launch and inspects the detection
            overlay at the rollout's own ENTER gate.
        pos_cond_matching: Order source forwarded to extract_object_positions.py
            (size|hungarian|prompt). For a LIVE scene use ``prompt`` (``size`` needs a
            ``<raw_root>/deploy/conditions.json`` that does not exist live). Default ``size``.
        prompt_after_capture: In ``prompt`` mode, preview the size-slot overlay after capture
            (a background OS image viewer if a display exists, else the path is printed) and
            read the task order from the terminal — NON-blocking, so it never hangs headless.
            Without it, ``prompt`` mode falls back to the identity (size) order. Default off.
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
        scenediff_config=scenediff_config,
        scenediff_timeout=scenediff_timeout,
        prompt_before_capture=prompt_before_capture,
        pos_cond_matching=pos_cond_matching,
        prompt_after_capture=prompt_after_capture,
        arm_side=arm_side,
    )
    ctrl.run()


if __name__ == "__main__":
    sys.exit(tyro.cli(main))
