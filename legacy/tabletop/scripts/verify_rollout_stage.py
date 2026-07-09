"""Re-run policy_rollout.py's ACT stage head on a saved live-rollout episode.

Loads the same checkpoint + pre/post processors that ``PolicyRolloutController``
uses, rebuilds each frame's model input from the recorded raw observations
(exactly mirroring ``_build_observation``: 20-D eef state, cropped+resized head,
resized wrist), and replays ``_predict_act_stage_from_batch`` (stage logits at
action offset 0, argmax). It then compares the offline prediction against the
``obs/position_condition/stage`` that was actually recorded during the rollout.

Run in the ``dexmate_lerobot`` conda env (the one that has lerobot):

    conda run -n dexmate_lerobot python scripts/verify_rollout_stage.py \
        --episode /home/yixuan/Dexmate/deploy/act/dexmate_eef_eef_abs_2cam_pos_16_8/last/0/episode_0.hdf5 \
        --policy-path /home/yixuan/Dexmate/model/act/dexmate_eef_eef_abs_2cam_pos_16_8/checkpoints/200000/pretrained_model
"""

from __future__ import annotations

import argparse

import cv2
import h5py
import numpy as np
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_POS_CONDITION_MASK

# Mirror the constants in policy_rollout.py.
HEAD_CROP_TOP, HEAD_CROP_BOTTOM = 220, 600
HEAD_CROP_LEFT, HEAD_CROP_RIGHT = 150, 800


def _to_chw(img_hwc: np.ndarray) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(img_hwc)).permute(2, 0, 1).float() / 255.0
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--policy-path", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── load policy + processors exactly like PolicyRolloutController._load_policy ──
    pcfg = PreTrainedConfig.from_pretrained(args.policy_path)
    pcfg.pretrained_path = args.policy_path
    policy = (
        get_policy_class(pcfg.type)
        .from_pretrained(args.policy_path, config=pcfg)
        .to(device)
        .eval()
    )
    pre, _post = make_pre_post_processors(
        policy_cfg=pcfg,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    n_classes = int(getattr(pcfg, "stage_prediction_num_classes", 2))
    n_action_steps = int(getattr(pcfg, "n_action_steps"))
    image_h, image_w = (
        int(pcfg.input_features["observation.images.head_rgb"].shape[1]),
        int(pcfg.input_features["observation.images.head_rgb"].shape[2]),
    )
    use_wrist = "observation.images.wrist_rgb" in pcfg.input_features
    image_features = list(getattr(pcfg, "image_features", []) or [])
    print(
        f"policy type={pcfg.type} stage_enabled="
        f"{getattr(pcfg, 'stage_prediction_enabled', False)} n_classes={n_classes} "
        f"n_action_steps={n_action_steps} resize=({image_h}x{image_w}) "
        f"use_wrist={use_wrist} image_features={image_features}"
    )

    # ── load recorded episode ──
    with h5py.File(args.episode, "r") as f:
        eef9_l = f["obs/eef_9d/left"][:]
        eef9_r = f["obs/eef_9d/right"][:]
        grip_l = f["obs/gripper/left"][:]
        grip_r = f["obs/gripper/right"][:]
        head = f["obs/images/head_left_rgb"][:]
        wrist = f["obs/images/left_wrist_rgb"][:] if use_wrist else None
        rec_stage = f["obs/position_condition/stage"][:]
        rec_cond6d = f["obs/position_condition/condition_6d"][:]
        action_number = f["time/action_number"][:]
        obs_flag = f["time/obs_flag"][:]

    n = len(eef9_l)
    print(f"episode frames={n}  recorded stage unique={np.unique(rec_stage)}")

    # env_state: the live code feeds the flat (n_classes*6,) [before,after per object]
    # vector and the preprocessor collapses it to the active 6-D slot via the mask.
    # The stage head does NOT consume environment_state (it reads robot-state + image
    # features only), so the offline prediction is invariant to the inactive slot.
    # We fill the active slot (object 0) with the recorded condition_6d and replicate
    # it into the other slot; an independence check below confirms it does not matter.
    cond6d_active = rec_cond6d[0].astype(np.float32)
    env_state_vec = np.tile(cond6d_active, n_classes).astype(np.float32)

    def build_raw(i: int, stage_for_mask: int) -> dict:
        state = np.concatenate(
            [eef9_l[i], [grip_l[i]], eef9_r[i], [grip_r[i]]]
        ).astype(np.float32)
        cropped = head[i][
            HEAD_CROP_TOP:HEAD_CROP_BOTTOM, HEAD_CROP_LEFT:HEAD_CROP_RIGHT
        ]
        policy_head = cv2.resize(
            cropped, (image_w, image_h), interpolation=cv2.INTER_AREA
        )
        mask = np.zeros((n_classes,), dtype=np.float32)
        mask[stage_for_mask] = 1.0
        sample = {
            "observation.state": torch.from_numpy(state),
            "observation.images.head_rgb": _to_chw(policy_head),
            OBS_ENV_STATE: torch.from_numpy(env_state_vec.copy()),
            OBS_POS_CONDITION_MASK: torch.from_numpy(mask),
        }
        if use_wrist:
            policy_wrist = cv2.resize(
                wrist[i], (image_w, image_h), interpolation=cv2.INTER_AREA
            )
            sample["observation.images.wrist_rgb"] = _to_chw(policy_wrist)
        return sample

    @torch.no_grad()
    def predict(sample_raw: dict) -> np.ndarray:
        """Return stage_logits (n_action_steps, n_classes) for one frame."""
        batch = pre(dict(sample_raw))
        model_batch = dict(batch)
        if image_features:
            model_batch[OBS_IMAGES] = [model_batch[k] for k in image_features]
        out = policy.model(model_batch)
        stage_logits = out[2]
        return stage_logits[0].detach().cpu().numpy()  # (n_action_steps, n_classes)

    # ── replay prediction on every frame (mask = recorded stage, irrelevant to head) ──
    pred_off0 = np.empty(n, dtype=np.int32)
    prob1_off0 = np.empty(n, dtype=np.float32)
    pred_anyoff = np.empty(n, dtype=np.int32)  # any offset in the chunk predicts 1?
    for i in range(n):
        logits = predict(build_raw(i, int(rec_stage[i])))  # (n_action_steps, n_classes)
        l0 = logits[0]
        pred_off0[i] = int(np.argmax(l0))
        prob1_off0[i] = float(
            np.exp(l0[1]) / np.exp(l0).sum()
        ) if n_classes == 2 else np.nan
        pred_anyoff[i] = int((logits.argmax(axis=-1) == 1).any())

    # Replan frames = where the live loop actually ran the head (queue empty).
    replan = obs_flag.astype(bool)

    print("\n=== offset-0 predicted stage vs recorded stage ===")
    match_all = np.array_equal(pred_off0, rec_stage)
    print(f"all-frames exact match: {match_all}")
    mism = np.where(pred_off0 != rec_stage)[0]
    print(f"mismatching frames: {len(mism)}  indices={mism.tolist()}")

    print("\n=== restricted to replan frames (obs_flag=True, where live actually predicts) ===")
    rp_idx = np.where(replan)[0]
    rp_match = np.array_equal(pred_off0[replan], rec_stage[replan])
    print(f"replan frames: {rp_idx.tolist()}")
    print(f"replan exact match: {rp_match}")
    print(f"replan recorded stage: {rec_stage[replan].tolist()}")
    print(f"replan predicted stage (off0): {pred_off0[replan].tolist()}")
    print(f"replan P(class1) off0: {np.round(prob1_off0[replan], 4).tolist()}")

    print("\n=== does ANY chunk offset ever predict stage 1? ===")
    print(f"frames where some offset argmax==1: {int(pred_anyoff.sum())} / {n}")
    print(f"max P(class1) over all frames @off0: {float(prob1_off0.max()):.4f} "
          f"(min {float(prob1_off0.min()):.4f})")

    # ── env_state independence check: perturb the inactive slot, confirm stage logits ──
    # are unchanged. This validates that omitting the true npz slot is sound.
    i0 = int(rp_idx[0]) if len(rp_idx) else 0
    base = predict(build_raw(i0, 0))
    saved = env_state_vec.copy()
    env_state_vec[:] = saved + np.random.RandomState(0).randn(*saved.shape) * 5.0
    pert = predict(build_raw(i0, 0))
    env_state_vec[:] = saved
    max_abs = float(np.abs(base - pert).max())
    print(
        f"\n=== env_state independence (frame {i0}) ===\n"
        f"max |Δ stage_logits| after perturbing env_state by ~N(0,5): {max_abs:.2e} "
        f"(should be ~0 → stage head ignores env_state)"
    )


if __name__ == "__main__":
    main()
