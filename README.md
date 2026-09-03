# 🎮 Omniteleop - Teleoperation Stack for Dexmate Robots

Python

## 📦 Installation

```shell
pip install omniteleop
```

## Official docs

- [Dexmate](https://docs.dexmate.ai/dSBwCBpol8PGkSXTS9bJ)
- [ZED camera](https://www.stereolabs.com/docs)
  - [Depth sensing](https://www.stereolabs.com/docs/depth-sensing)
  - Update 'dexsensor gen-cfg' if reinstall

## Latency

- Dedicated robot/workstation Ethernet setup, preflight, process placement, and
before/after measurement: [WBC_ETHERNET.md](WBC_ETHERNET.md)
- WLAN

```bash
sudo iw dev wlP1p1s0 set power_save off
sudo nmcli connection modify Dexmate_5G 802-11-wireless.powersave 2
```

- Head camera
  - Something worthy of trying if latency in dexmate@vega-1:~/.dexmate/sensors/default.toml
    - Under [[sensors]] id = "head_camera"
      - [sensors.params] rate = 15
      - [sensors.streams] right_rgb = false
      - [sensors.params] depth_mode = "NEURAL_LIGHT"

```python
python scripts/robotiq_gripper_cmd_actual_omni.py # gripper latency by sending cmd in different curves
```



# WBC



## Reference codebase

- [HoMMi](/home/yixuan/hommi)



## Robot preparation

1. sshfs [yixuan@192.168.0.166](mailto:yixuan@192.168.0.166):/home/yixuan/omniteleop /home/dexmate/yixuan/omniteleop_yifan
  sshfs [yixuan@192.168.0.166](mailto:yixuan@192.168.0.166):/media/yixuan/portable_ssd/Dexmate /home/dexmate/yixuan/Dexmate
2. run '(dexmate) dextop node start' '(yixuan_yifan) python /home/dexmate/yixuan/omniteleop_yifan/tests/test_wrist_zedm_depth.py' and then '(yixuan_yifan) python /home/dexmate/yixuan/omniteleop_yifan/tests/test_head_zedx_depth.py' in tmux
3. robot=Robot() to prevent arm falling

> Run head before wrist may cause resolution error

> Head and wrist: 240x320. Increasing to 640x480 under wbc_vr_robot's --record-rate (10 Hz) leads to stale frame due to bandwidth-throttled. Bump only after confirming sustained fps with --verify.

> If change resolution in the future, change HEAD_RESIZE_HW in [head_camera.py](../../src/omniteleop/common/head_camera.py) (head) and resize_h/resize_w in [test_wrist_zedm_depth.py](../../tests/test_wrist_zedm_depth.py) (wrist), then **restart both publishers**. `ZED_K` and the recorded `obs/images/intrinsic` scale automatically, and `vis_episode.py` adapts (reads the saved intrinsic + frame shape). Two things do NOT auto-follow: (1) keep the aspect ratio **4:3**, or also update `_TILE_W`/`_TILE_H` in [wbc_headset_hud.py](../../src/omniteleop/leader/wbc_headset_hud.py) (+ the `8/3` fallback in `web/vr_client.html`) to avoid HUD stretch; (2) the **training-dataset** resolution is set separately by `port_wbc_mobile_hdf5.py --resize-h/--resize-w` (default 240x320), independent of the recording resolution.

> Decreased for latency: `_HUD_SCALE`, `--hud-rate` default 10 Hz, JPEG quality 60 -> 40 in `poll_and_send`. Revert if the HUD gets too blurry/choppy to be useful.

See [PIPELINE_WBC](./PIPELINE_WBC.md).

## Mobile conditioned policy pipeline

**1.** Record data:

```bash
# on lambda/leader machine
# Terminal 1
(dexmate) python scripts/wbc_vr_leader.py --calibrate-ee-offset
# one-time calibration, save src/omniteleop/leader/ee_offset.yaml for future auto load
(dexmate) python scripts/wbc_vr_leader.py
# generate reference episode_0.hdf5 for future --align-reference
(dexmate) python /home/yixuan/omniteleop/scripts/wbc_vr_leader.py --align-reference /home/yixuan/Dexmate/data/raw_data_reference/reference.hdf5

# Terminal 2
(dexmate) python scripts/wbc_vr_robot.py \
  --record \
  --save-dir /home/yixuan/Dexmate/data/box2cloth/raw_data
  # --debug-dir /home/yixuan/Dexmate/data/raw_data_debug

# for teleoperator: twist hand when pick, forward when place

(dexmate) python /home/yixuan/omniteleop/scripts/vis_episode.py --hdf5 /home/yixuan/Dexmate/data/box2cloth/raw_data/episode_0.hdf5

# jitter should stay ~1 camera frame; 
(dexmate) python scripts/audit_episode_latency.py /home/yixuan/Dexmate/data/box2cloth/raw_data/episode_0.hdf5
```

**1.5** Generate the SceneDiff position condition (per-episode `[box, cloth]` 3D positions).

Diffs FRAME 0 of every raw + recovery episode (discovered by **globbing**
`~/Dexmate/data/box2cloth/raw_data/` + `.../recovery/` — **no** `log_index.csv`, which is produced later
by the porter) against the LAST frame of the fixed reference scene, and reduces the change
masks to the two objects' 3D positions in the **engage-origin WORLD frame** (same frame as
`action` and `observation.state[29:32]`). Order `[box=source, cloth=target]` is the world-frame
midpoint of `action/eef/{left,right}` at the successful grasp/release keyframe, Hungarian-matched
to the centroids with a distance + ambiguity guard — a bad episode is SKIPPED (see
`scene_diff/skip.json`) rather than silently mislabeled.

```bash
conda activate lerobot   # SAM3 runs here; make_wbc_before_hdf5 + extract auto-use dexmate_lerobot
bash /home/yixuan/scene_diff/run_wbc_pos_condition.sh
#   reference : last frame of ~/Dexmate/data/box2cloth/scene_diff/reference/episode_0.hdf5
#   positions : ~/Dexmate/data/box2cloth/scene_diff/positions/<source>/episode_<N>.npz   (porter input, step 2)
#   VERIFY    : ~/Dexmate/data/box2cloth/scene_diff/visualization/hungarian/<source>/episode_<N>.png

# subset e.g.: bash /home/yixuan/scene_diff/run_wbc_pos_condition.sh 1 25
```

> Detection knobs in `scene_diff/configs/scenediff_config.yml`:
> `models.sam.mask_assignment_order: smallest_first` (keeps the box from being absorbed by the
> shelf/closet mask) and `detection.min_detection_pixel: 800` (drops small false-change regions —
> a remote e-stop, box sub-fragments — that were outranking the cloth; real box/cloth masks are
> ≥2680px). Recovery HDF5s keep their untrimmed frame 0, so the condition is read there even
> though the porter later trims their training window.

> **Position conditioning (Part B, implemented):** step 2's porter attaches these as a
> constant 6-D `observation.environment_state` via `--positions-dir ~/Dexmate/data/box2cloth/scene_diff/positions --object_nums 2`. Train wiring is step 3.

240×320 → use full-frame crop `0 240 0 320` (SceneDiff resize 392×518).

```bash
# symlink scene_diff episodes into train/ / test/ by log_index.csv, then plot
LAYOUT=/tmp/wbc_scene_diff_vis_layout
rm -rf "$LAYOUT" && mkdir -p "$LAYOUT/train" "$LAYOUT/test"
(dexmate_lerobot) python - <<'PY'
import pandas as pd
from pathlib import Path
log = pd.read_csv("/home/yixuan/Dexmate/data/box2cloth/raw_data/log_index.csv")
sd, layout = Path("/home/yixuan/Dexmate/data/box2cloth/scene_diff"), Path("/tmp/wbc_scene_diff_vis_layout")
for _, r in log.iterrows():
    src, idx = r["source"], int(r["raw_data_index"])
    (layout / r["split"] / f"episode_{idx}").symlink_to(sd / src / f"episode_{idx}")
PY
(lerobot) python /home/yixuan/scene_diff/scripts/visualize_train_distribution.py \
  --scene-diff-root /tmp/wbc_scene_diff_vis_layout \
  --split train --valid-split test \
  --rollout-dir /home/yixuan/Dexmate/data/box2cloth/rollout/concat_abs_minmax \
  --reference-hdf5 /home/yixuan/Dexmate/data/box2cloth/scene_diff/_before/raw/episode_1.hdf5 \
  --crop 0 240 0 320 \
  --output /home/yixuan/Dexmate/data/visualization/wbc_train_test_distribution.png
#   -> ~/Dexmate/data/visualization/wbc_train_test_distribution.png
#      train (n=46) | valid=processed_wbc/test (n=2) | test=concat_abs_minmax rollouts (n=6)
```

**2.** Port raw HDF5 to a LeRobot dataset (add `--positions-dir` for position conditioning):

```bash
(dexmate_lerobot) python scripts/port_wbc_mobile_hdf5.py \
    --raw-dir ~/Dexmate/data/box2cloth/raw_data \
    --root ~/Dexmate/data/box2cloth/processed_wbc \
    --repo-id dexmate_wbc_eef_head \
    --include_recovery_data ~/Dexmate/data/box2cloth/raw_data/recovery \
    --log_index ~/Dexmate/data/box2cloth/raw_data/log_index.csv \
    --split-csv /home/yixuan/Dexmate/data/box2cloth/raw_data/split.csv \
    --positions-dir ~/Dexmate/data/box2cloth/scene_diff/positions --object_nums 2
```

`--positions-dir` (optional; omit for an unconditioned dataset) adds a **constant** 6-D  
`observation.environment_state` = the two objects' world positions arranged `[box(src), cloth(dst)]`,  
read from `<positions-dir>/<source>/episode_<N>.npz` (step 1.5 output; `source ∈ {raw, recovery}`  
keys the file so a raw and a recovery episode sharing an index never collide). Every ported episode  
**must** have its npz — the porter validates them all up front and aborts before writing/overwriting  
anything if any is missing. Single-stage, so no per-frame `pos_condition_mask` is emitted. The  
`meta/stats.json` `min`/`max` feed the `ENV: MIN_MAX` normalization at train time (step 3).

```bash
(dexmate_lerobot) python scripts/vis_episode_processed_wbc.py \
  --dataset_dir /home/yixuan/Dexmate/data/box2cloth/processed_wbc/train/dexmate_wbc_eef_head \
  --episode_index 0
```

**3.** Train in LeRobot

The checkpoint must report `observation.state` dim `32`, `action` dim `29`, image keys `observation.images.head_rgb` and optionally `observation.images.wrist_rgb`; with position conditioning it also carries `observation.environment_state` dim `6` (the SceneDiff `[box, cloth]` world positions from steps 1.5/2). Continuous state/action dims and the constant env-state are all normalized with dataset `min`/`max` (`MIN_MAX` mode; the porter already writes these into `meta/stats.json`).

```markdown
Current implemented schema:

observation.state (32) — do not compose world_T_base into observation.state[:29]
  0-28   base-frame achieved left/right EEF + grippers, base-frame achieved zed_depth_frame head pose
         (WBC FK on measured joints with base zeroed; grippers = raw FC03 reading)
  29-31  obs/base/pose (engage-origin world)

action (29) — WORLD-frame targets passed to VegaWholeBodyIK.solve(..., head_target=...)
  0-2    left EEF position (tx, ty, tz)
  3-8    left EEF orientation (r00, r10, r20, r01, r11, r21)
  9      left gripper command (clipped/binarized 0/1)
  10-12  right EEF position
  13-18  right EEF orientation
  19     right gripper command
  20-22  head position (post-LPF/post-deadband):
  23-28  head orientation

wbik.yaml must keep head_mode: "ik" for rollout.
```

**Vanilla DP — full command.** Copy-pasteable as-is; only `--job_name` / `--output_dir` need to
change per run. Position conditioning is toggled by the `observation.environment_state` input
feature + the `"ENV"` mapping entry (see the notes below the block).

```bash
# default config at [configuration_diffusion.py](../lerobot_original/src/lerobot/policies/diffusion/configuration_diffusion.py)
# --policy.position_condition_mode=concat/film
(dexmate_lerobot) lerobot-train \
  --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
  --policy.horizon=16 --policy.n_action_steps=8 \
  '--policy.down_dims=[256,512,1024]' \
  --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=16 \
  '--policy.input_features={"observation.images.head_rgb": {"type": "VISUAL", "shape": [3, 120, 160]}, "observation.images.wrist_rgb": {"type": "VISUAL", "shape": [3, 120, 160]}, "observation.state": {"type": "STATE", "shape": [32]}, "observation.environment_state": {"type": "ENV", "shape": [6]}}' \
  '--policy.normalization_mapping={"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX", "ENV": "MIN_MAX"}' \
  '--policy.skip_normalization_dims={"observation.state": [3,4,5,6,7,8,13,14,15,16,17,18,23,24,25,26,27,28], "action": [3,4,5,6,7,8,13,14,15,16,17,18,23,24,25,26,27,28]}' \
  --policy.position_condition_mode=concat \
  --dataset.repo_id=dexmate_wbc_eef_head \
  --dataset.root=/home/yixuan/Dexmate/data/box2cloth/processed_wbc/train/dexmate_wbc_eef_head \
  --dataset.image_transforms.enable=true \
  --batch_size=32 --steps=100000 --save_freq=50000 --save_best=true \
  --output_dir=/home/yixuan/Dexmate/model/dp/world_concat \
  --wandb.enable=true \
  --wandb.project=dexmate_wbc_mobile \
  --job_name=world_concat \
  --dataset.val_root=/home/yixuan/Dexmate/data/box2cloth/processed_wbc/test/dexmate_wbc_eef_head --val_freq=5000
# measured (4090, batch 32, horizon 16, this dataset):
#   down_dims=[256,512,1024]  -> 90.2M params, 36 ms/step, 2.52 GiB peak (~1.0 h / 100k)
#   down_dims=[512,1024,2048] -> 280M params, 62 ms/step, 5.44 GiB peak (~1.7 h / 100k)  <- LeRobot default
```



**ACT (alternative to diffusion).** Same dataset, same 32/29 schema, same skip-normalization dims —
only the policy flags change. Position conditioning is turned on/off by adding/dropping the SAME two
ENV bits (`observation.environment_state` input feature + `"ENV": "MIN_MAX"`), so this is one command
with a two-line diff, not two pipelines.

```bash
# WITH position condition (dataset ported with step 2 --positions-dir)
(dexmate_lerobot) lerobot-train \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=16 --policy.n_action_steps=8 \
  '--policy.input_features={"observation.images.head_rgb": {"type": "VISUAL", "shape": [3, 120, 160]}, "observation.images.wrist_rgb": {"type": "VISUAL", "shape": [3, 120, 160]}, "observation.state": {"type": "STATE", "shape": [32]}, "observation.environment_state": {"type": "ENV", "shape": [6]}}' \
  '--policy.normalization_mapping={"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX", "ENV": "MIN_MAX"}' \
  '--policy.skip_normalization_dims={"observation.state": [3,4,5,6,7,8,13,14,15,16,17,18,23,24,25,26,27,28], "action": [3,4,5,6,7,8,13,14,15,16,17,18,23,24,25,26,27,28]}' \
  --dataset.repo_id=dexmate_wbc_eef_head \
  --dataset.root=/home/yixuan/Dexmate/data/box2cloth/processed_wbc/train/dexmate_wbc_eef_head \
  --dataset.image_transforms.enable=true \
  --batch_size=32 --policy.optimizer_lr=2e-4 --policy.optimizer_lr_backbone=2e-5 --steps=200000 --save_freq=50000 --save_best=true --log_freq=2000 \
  --output_dir=/home/yixuan/Dexmate/model/act/dexmate_wbc_eef_head \
  --wandb.enable=true --wandb.project=dexmate_wbc_mobile --job_name=act_pos_abs \
  --dataset.val_root=/home/yixuan/Dexmate/data/box2cloth/processed_wbc/test/dexmate_wbc_eef_head --val_freq=5000

# WITHOUT position condition: delete "observation.environment_state" from --policy.input_features
# and the "ENV" entry from --policy.normalization_mapping (change --job_name/--output_dir).
# Works on a conditioned dataset too -- the extra feature is simply not declared as an input.
# Optional: --policy.use_separate_rgb_encoder_per_camera=true gives head/wrist their own ResNet
# backbone instead of sharing one (only meaningful with 2 cameras).
```

Differences from the tabletop [train_dexmate_act.sh](../lerobot_original/examples/port_datasets/train_dexmate_act.sh) — do not copy it verbatim:

- `--policy.chunk_size` replaces diffusion's `--policy.horizon`; ACT pins `n_obs_steps=1`.
- **No** `--policy.position_condition_mode`**.** `concat`/`film` is diffusion-only. ACT ingests the
env-state as one extra transformer-encoder token (`encoder_env_state_input_proj`, +4096 params),
so conditioning is always concat-like.
- **Leave** `--policy.stage_prediction_enabled` **at its default** `false`**.** The tabletop script sets it
`true`; it requires a per-frame `observation.pos_condition_mask`, which this single-stage porter
does not emit — ACT raises on the first backward pass. Same reason `stage_prediction_*` is absent
above. The env shape `[6]` equals the processor's `condition_dim`, so `PositionConditionProcessorStep`
passes the vector through unmasked; the N-cycle caveat from the diffusion block applies identically.
- Tabletop normalizes STATE/ACTION with `MEAN_STD`; WBC uses `MIN_MAX` (+ rotation-dim skips) to
stay consistent with the diffusion run on this dataset. ENV stays `MIN_MAX` in both.
- `--policy.optimizer_lr=2e-4` overrides ACT's `1e-5` default (matches the tabletop script).
`--policy.optimizer_lr_backbone=2e-5` keeps the ResNet at 1/10 of that (ACT already exposes this
flag; diffusion does not — DP uses one LR for the whole model).
- **Do not set** `--policy.temporal_ensemble_coeff`: ACT then requires `n_action_steps=1`, and
`wbc_policy_rollout.py` schedules whole chunks — a 1-step chunk covers 0.1 s and the buffer runs dry.

Deploy is unchanged: `scripts/wbc_policy_rollout.py` is policy-type agnostic (it reads `n_action_steps`
off the checkpoint and feeds ACT the batch directly instead of the diffusion obs queues). Point
`--policy-path` at the ACT checkpoint. The checkpoint's declared input features are the sole authority:
an `observation.environment_state` input triggers the live SceneDiff bootstrap automatically, while an
unconditioned checkpoint skips it. There is no rollout-side conditioning-mode flag.

**Mixture of Frames (optional).** `--policy.action_frame_mode=mof` swaps the single
world-frame U-Net for **one full U-Net per reference frame** plus a learned router, the same
Mixture-of-Frames recipe as the
[legacy standalone mofpo run](#legacy-standalone-mof-reproduction). The integrated LeRobot
policy defaults to four experts, dropping `base` as a reparameterization of
`base_rel_trans`; `base` stays selectable for the five-tower set. `world` is the default
`action_frame_mode` and is byte-identical to before the feature.
Implementation:
`../lerobot_original/src/lerobot/policies/diffusion/mof/` (`action_layout.py` +
`frame_transforms.py` are mirrored copies of the mofpo modules; `stats.py` and
`mixture.py` are LeRobot-native).

**Same dataset as the world run** — no re-port. The porter stores `observation.state[:29]`
base-frame, and MoF lifts it to world in-graph from the planar base pose at `29:32`
(`lift_base_state_to_world`, verified equal to the independently ported world-state dataset to
`1.2e-7`). So DP-world vs DP-mof differ in the action-frame machinery
and **nothing else** — provided you width-match them (see `down_dims` below).

**DP+MoF — full command.** Same dataset, same batch/steps/val wiring as the vanilla command
above; the differing flags are itemized under the block.

```bash
(dexmate_lerobot) lerobot-train \
  --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
  --policy.action_frame_mode=mof \
  --policy.horizon=16 --policy.n_action_steps=8 \
  '--policy.down_dims=[256,512,1024]' \
  --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=16 \
  '--policy.input_features={"observation.images.head_rgb": {"type": "VISUAL", "shape": [3, 120, 160]}, "observation.images.wrist_rgb": {"type": "VISUAL", "shape": [3, 120, 160]}, "observation.state": {"type": "STATE", "shape": [32]}, "observation.environment_state": {"type": "ENV", "shape": [6]}}' \
  '--policy.normalization_mapping={"VISUAL": "MEAN_STD", "STATE": "IDENTITY", "ACTION": "IDENTITY", "ENV": "IDENTITY"}' \
  --policy.position_condition_mode=concat --policy.mof_env_frame=family \
  --policy.mof_stats_root=/home/yixuan/Dexmate/data/box2cloth/processed_wbc/train/dexmate_wbc_eef_head \
  --dataset.repo_id=dexmate_wbc_eef_head \
  --dataset.root=/home/yixuan/Dexmate/data/box2cloth/processed_wbc/train/dexmate_wbc_eef_head \
  --dataset.image_transforms.enable=true \
  --batch_size=32 --steps=100000 --save_freq=50000 --save_best=true \
  --output_dir=/home/yixuan/Dexmate/model/dp/mof_concat --job_name=mof_concat \
  --wandb.enable=true --wandb.project=dexmate_wbc_mobile \
  --dataset.val_root=/home/yixuan/Dexmate/data/box2cloth/processed_wbc/test/dexmate_wbc_eef_head --val_freq=5000
# skip_normalization_dims in modeling_diffusion.py

# measured at the FOUR-expert default (4090, batch 32, horizon 16, this dataset):
#   down_dims=[256,512,1024]  -> 294M params, 85 ms/step, 5.71 GiB peak (~2.4 h / 100k)
#   down_dims=[512,1024,2048] -> 1.05B params; weights+grad+Adam alone are 15.7 GiB, so it does NOT fit a 24 GB card once activations are added. 
# LeRobot's DEFAULT width (280M) happens to be
# capacity-matched to MoF-narrow (294M)
# defaults: mof_enabled_experts=[base_rel_trans,left,right,rel_traj]
#           mof_canonical_space=base_rel_trans  mof_router_mode=learned
#           mof_expert_loss_coef=1.0            mof_state_frame=base
#           mof_env_frame=world
```



Exactly five flags differ from the vanilla command above (everything else — dataset, batch,
steps, val wiring, `horizon`/`n_action_steps`, `position_condition_mode`, `down_dims`,
`noise_scheduler_type`/`num_inference_steps` — is identical, which is the point: both commands
now sample DDIM/16, so a rollout/latency comparison isn't confounded by sampler choice):


|             | flag                                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **adds**    | `--policy.action_frame_mode=mof`, `--policy.mof_env_frame=family`, `--policy.mof_stats_root=...`          |
| **drops**   | `--policy.skip_normalization_dims`                                                                        |
| **changes** | `--policy.normalization_mapping` (STATE/ACTION/ENV → `IDENTITY`)                                          |


All of these are enforced by `_validate_mof()`, which raises with the fix in the
message. Why each one:

- `--policy.mof_stats_root` must repeat `--dataset.root`. MoF's affines are fitted from the
parquet, not from `meta/stats.json`, and they are **not** derivable from the batch. Left unset the
buffers stay NaN and the first forward pass raises `non-finite stats` rather than training on
garbage — which is also what makes loading a checkpoint work, since its fitted buffers overwrite
them.
- `STATE`/`ACTION` must be `IDENTITY`. MoF needs raw world poses to build frames and returns raw
world actions, so it owns normalization itself: it fits per-expert action and per-family state
min/max over the **exact training windows** (one parquet-only pass at construction, seconds) and
carries them in the checkpoint. **Drop** `--policy.skip_normalization_dims` — the rotation-dim
skip moves inside those fitted affines.
- `ENV` must be `IDENTITY` **when** `--policy.mof_env_frame=family`, for the same reason one dim
up: MoF transforms raw world XYZ, so it also fits `{family}__env_state` itself. Under
`mof_env_frame=world` leave `ENV` at `MIN_MAX` — the processor-side collapse in
`processor_diffusion.py` is correct there and stays in charge.
- `DDIM` + `--policy.num_inference_steps=16` (now pinned in **both** commands, not a MoF-only
flag): `_validate_mof()` rejects a `None` `num_inference_steps` outright, because the
`num_train_timesteps=100` DDPM default would cost MoF 4x100 U-Net forwards per chunk — four
towers per denoise step blows the 0.4 s chunk budget. The vanilla command pins the same
scheduler/step-count so the two runs aren't also confounded by sampler choice; check
`--policy-interval` against a real measurement before hardware either way.
- `stage_prediction_enabled` must stay `false` (the stage head needs the single `global_cond` that
MoF replaces with one per family). It is already the default, so it is not in either command.

**Width and the fair-comparison caveat.** `--policy.down_dims=[256,512,1024]` (mofpo's paper
recipe) is set in **both** commands. It is not enforced by the config, but the default width does
not fit: four towers at LeRobot's default `[512,1024,2048]` measure **1.05B params**, whose
weights+grad+Adam alone are **15.7 GiB**, so a 24 GB card OOMs once activations land. Measured on
a 4090 at batch 32 / horizon 16 on this dataset:


| run                    | `down_dims`                         | params | step  | peak                   | 100k steps         |
| ---------------------- | ----------------------------------- | ------ | ----- | ---------------------- | ------------------ |
| DP world               | `[256,512,1024]`                    | 90.2M  | 36 ms | 2.52 GiB               | ~1.0 h             |
| DP world               | `[512,1024,2048]` (LeRobot default) | 280M   | 62 ms | 5.44 GiB               | ~1.7 h             |
| DP **mof** (4 experts) | `[256,512,1024]`                    | 294M   | 85 ms | 5.71 GiB               | ~2.4 h             |
| DP **mof** (4 experts) | `[512,1024,2048]`                   | 1.05B  | —     | 15.7 GiB weights alone | does not fit 24 GB |


The two commands above are **width-matched** (both `[256,512,1024]`), so MoF carries 3.3x the
parameters — that is just the four towers, and it is the honest way to isolate the action-frame
machinery. If you would rather control for *capacity*, note that DP-world at LeRobot's default
width (280M) is near-identical in size to MoF-narrow (294M). Either control is defensible; state
which one you used when reporting a delta.

**Position-condition frame.** `--policy.mof_env_frame` decides what each obs family sees of the
task. `world` (default) is the incumbent behaviour — one world-frame env-state rides every
family's conditioning vector. `family` re-expresses it into that family's per-step frame, which
matters because `reexpress_agent_pos` zeroes the base block of a robot-anchored family's state:
without it, a `base`/`left`/`right` tower holds a world-frame object position it cannot relate to
its own state. This keeps the position condition consistent with each expert; an RGB image itself
carries no frame. There is no separate ablation flag; the ablation is this flag's two values.
See `docs/adr/0002-dp-mof-reexpresses-the-position-condition.md`.

**Object-mask channels are not supported here.** A ResNet has no point-budget sampling stage,
and `mask_channels=true` is designed but deferred; see
`docs/adr/0003-dp-mask-channels-are-a-separate-image-feature.md`.

**Parity pre-flight** before trusting a MoF run — reduces the mixture to the incumbent model:
`--policy.mof_enabled_experts=[world] --policy.mof_canonical_space=world --policy.mof_router_mode=fixed_uniform`.

Deploy and offline validation are **unchanged commands**: `wbc_policy_rollout.py` reads
`action_frame_mode` off the checkpoint and takes `observation.state`'s frame from
`mof_state_frame`; the base-frame state it
already builds is what a default MoF checkpoint wants. `vis_wbc_policy_prediction.py` feeds the
dataset state verbatim and needs nothing. Tests:
`../lerobot_original/tests/policies/test_diffusion_mof.py` (42).

**4.**

Validate the checkpoint OFFLINE before any hardware rollout — run the rollout's exact inference path (_PolicyBundle.select_action + split_policy_action) over a processed episode and overlay the policy's predicted 29-D world-frame action against the recorded ground-truth action. Rerun shows per-entity (left/right/head) GT (red) vs predicted (magenta) pose + error line, GT/pred EEF reprojected into the head image, gripper series, the base odometry path, and the depth point cloud; matplotlib + stdout report per-entity translation/rotation MAE. Run in dexmate_lerobot:

```bash
(dexmate_lerobot) python scripts/vis_wbc_policy_prediction.py \
  --policy-path ~/Dexmate/model/dp/nopos_abs_minmax/checkpoints/last/pretrained_model \
  --dataset_dir ~/Dexmate/data/box2cloth/processed_wbc/test/dexmate_wbc_eef_head \
  --episode_index 0
```

Roll out the trained policy on the real robot. Inference is asynchronous (rby1-wbc
style scheduled chunks, see `plan.md`): a worker thread replans a whole action chunk
every `--policy-interval` (default 0.4 s), stale frames older than
`inference_end + --execution-latency` are dropped, and a 10 Hz sampler feeds the
100 Hz WBC tick. Waiting for the first chunk is allowed; after the first chunk has armed
the scheduler, exhausting its scheduled actions is a terminal episode failure that enters
safe hold and rejects any later result. `--dataset-fps` (default 10) must match the porter fps.

```bash
# replay
python scripts/wbc_policy_rollout.py --replay-episode ~/Dexmate/data/box2cloth/raw_data/episode_1.hdf5 \
    --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5

(dexmate_lerobot) python scripts/wbc_policy_rollout.py \
  --policy-path /home/yixuan/Dexmate/model/dp/nopos_abs_minmax/checkpoints/last/pretrained_model \
  --save-dir ~/Dexmate/data/box2cloth/rollout/nopos_abs \
  --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5 \
  --max-seconds 100
# Ctrl+C once to save
# if the console warns "buffer will run dry" (100-step DDPM is ~0.4 s/chunk),
# lower --policy-interval (e.g. 0.3) or reduce the checkpoint's num_inference_steps

# A checkpoint trained with observation.environment_state automatically bootstraps the scene.
# Launch conditioned LeRobot policies from an environment containing SAM3.1/SceneDiff.
# After engage and --align-reference, rollout captures one head frame, runs SceneDiff against
# the same reference used for training, displays a fresh size-slot overlay, and requires an
# explicit [source box, destination cloth] permutation before motion. Positions are then pinned
# for the episode. An unconditioned checkpoint skips SceneDiff and the operator prompt.
(policy_env) python scripts/wbc_policy_rollout.py \
  --policy-path /home/yixuan/Dexmate/model/dp/concat_abs_minmax/checkpoints/last/pretrained_model \
  --save-dir ~/Dexmate/data/box2cloth/rollout/concat_abs_minmax \
  --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5 \
  --max-seconds 100
#   --reference-hdf5 ~/Dexmate/data/box2cloth/scene_diff/_before/reference_last.hdf5   (default)
#   --prompt-before-capture   pause to stage the scene after alignment, before the head grab
#   SceneDiff outputs -> <save-dir>/scene_diff/episode_N/{live_capture.hdf5,episode_0.npz,
#                        bootstrap_size_slots.npz,deploy_slot_overlay.png,validation_*.{npz,json}}
#   If that final episode directory already exists, rollout exits before capture and asks you
#   to delete it manually. A failed bootstrap never publishes a partial final directory.

python scripts/vis_episode.py --hdf5 /home/yixuan/Dexmate/data/box2cloth/rollout/concat_abs_minmax/episode_0.hdf5
```



## Legacy standalone MoF reproduction

The active MoF path is now `action_frame_mode=mof` inside the LeRobot diffusion policy
(step 3). This section preserves the
older standalone [MoF](https://mofpo.github.io/) WBC v1 pipeline only, for reproducing its safetensor
dataset, mofpo checkpoint, offline visualizer, and hardware bundle. It uses the `wbc` branch of
[Crdr2/mofpo](https://github.com/Crdr2/mofpo) at `/home/yixuan/mofpo` (fork of the paper code;
upstream `pointW/mofpo`) and its separate legacy environments.

The standalone model denoises the 29-D world action in **five reference frames at once** (`base`,
`base_rel_trans`, `left`, `right`, `rel_traj` — the head is a third pose entity in every frame but
not a frame itself) and fuses the predictions with a learned per-timestep router, canonical space
`base_rel_trans`. Decisions/ADR are in `.scratch/mof-wbc-integration/` and
`docs/adr/0001-mof-sim-stack-for-wbc.md` (both gitignored, local).

**Frame conventions:** actions verbatim world-frame column rot6d (the checkpoint
sets `external_rot6d_convention: column`, so predictions come out in exactly the
recorded action format — no row-convention pass); obs are per-entity world
pos + wxyz quats derived from the world-frame 32-D state blocks; images are the
**224×224 anisotropic squash** of the 240×320 frames (full FOV kept).

**1.** Port raw takes to MoF safetensors (env `dexmate_lerobot`; SceneDiff
positions are ALWAYS attached — a conditioned run later is a train-config flip):

```bash
(dexmate_lerobot) python scripts/port_wbc_mobile_mof.py \
  --raw-dir ~/Dexmate/data/box2cloth/raw_data \
  --out-root ~/Dexmate/data/box2cloth/processed_wbc/mof \
  --include-recovery-data ~/Dexmate/data/box2cloth/raw_data/recovery \
  --positions-dir ~/Dexmate/data/box2cloth/scene_diff/positions
# -> dexmate_wbc_mof/{train,test}/episode_<i>.safetensors + meta.json
#    (46 train / 2 test episodes, 18,555 frames == the LeRobot dataset;
#     meta.json enumerates the tensor schema and is the deploy source of truth)
```

**2.** Train in mofpo (env `mof`; no simulator — `task.env_runner: null` skips
rollout eval and top-k checkpoints on `val_loss`):

```bash
(dexmate_mof) cd ~/mofpo && python train.py --config-name=train_mof_moe_wbc \
  logging.mode=offline hydra.run.dir=data/outputs/wbc_v1_mof_moe_500ep
# paper recipe: batch 128, DDIM 50/16, EMA, 500 epochs (~30 s/epoch + val on the
# 4090, ~14 GB), horizon 16 / 8 action steps @ 10 Hz. training.resume=True:
# relaunching the same run dir resumes from latest.ckpt.
```

**3.** Validate OFFLINE before any hardware run (env `dexmate_mof` — a
`dexmate_lerobot` clone + `mofpo_root.pth` + `pip install --no-deps threadpoolctl robomimic==<install.sh pin>` + a compatible pytorch3d build;
mofpo has no `__init__.py`, so the `.pth` does what `pip install -e` cannot):

```bash
(dexmate_mof) python scripts/vis_wbc_mof_prediction.py \
  --policy-path ~/mofpo/data/outputs/wbc_v1_mof_moe_500ep/checkpoints/latest.ckpt \
  --episode ~/Dexmate/data/box2cloth/processed_wbc/mof/dexmate_wbc_mof/test/episode_0.safetensors \
  --output ~/Dexmate/data/visualization/mof_pred_test0.png
# per-entity translation/rotation MAE + gripper accuracy + GT-vs-pred overlays,
# through the deploy bundle's exact inference path (load-time gates included)
```

**4.** Roll out on the robot — the same shared hardware loop as every policy
(`wbc_policy_rollout.py` via `policy_factory`; not even the inference worker is
forked):

```bash
(dexmate_mof) python scripts/wbc_mof_rollout.py \
  --policy-path ~/mofpo/data/outputs/wbc_v1_mof_moe_500ep/checkpoints/latest.ckpt \
  --save-dir ~/Dexmate/data/box2cloth/rollout/mof \
  --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5 \
  --policy-interval 0.6 --max-seconds 100
# 16-step DDIM x 5 experts measured ~0.6 s/chunk with the GPU shared by a
# training job (chunk covers 0.8 s): keep --policy-interval >= inference time,
# or pass --num-inference-steps 8. --replay-episode works identically.
```

The bundle refuses to load (before engage): a non-WBC or row-convention
checkpoint, a missing/mismatched port `meta.json`, or a normalizer without
fitted stats. v1 checkpoints are unconditioned, so their checkpoint-derived scene requirements are
empty and the shared rollout performs no SceneDiff bootstrap.

# Debug



### Arm joint out of limit

```bash
# on dexmate machine
python dexcontrol/examples/advanced_examples/disable_arm_motors.py disable --side right --joint-idx 0 --release-brake
# lift eef up
python dexcontrol/examples/advanced_examples/disable_arm_motors.py brake --side right --joints 0 --no-enable
python dexcontrol/examples/troubleshooting/clear_error.py
```



### Arm dead

```python
# ipython
import numpy as np
q = robot.right_arm.get_joint_pos().copy(); q[5] = 0.3   # j6 -> +0.3
robot.right_arm.set_joint_pos(q.tolist(), wait_time=2.0, exit_on_reach=True)
```

```bash
# if dead
python dexcontrol/examples/troubleshooting/display_robot_info.py
python dexcontrol/examples/advanced_examples/config_force_torque_sensor.py get --side both
python dexcontrol/examples/advanced_examples/config_force_torque_sensor.py set --side right --enable
```



### Arm joint to eef

```bash
python /home/yixuan/omniteleop/scripts/misc/check_eef_pos.py # robot.right_arm.get_joint_pos() first, and the run this to calculate eef from joint
```



### Collision

```bash
python /home/yixuan/omniteleop/scripts/diagnostics/browse_collision_pairs_sapien.py # --interactive , run with wbc_vr_leader
```



### Wheel steering

```python
robot.chassis.set_steering_angle(0.0, wait_time=1.5)
```

> Check revised _compute_wheel_control in `[dexcontrol/src/dexcontrol/core/chassis.py](dexcontrol/src/dexcontrol/core/chassis.py)`: take the physically reachable branch when the other's clamp distortion is large, a *marginal clamp is left to the nearest-branch pick.



### Replay on real

```bash
python scripts/wbc_vr_record.py --hdf5 /home/yixuan/Dexmate/tmp/test.hdf5 --physics --base-loop closed
python scripts/wbc_vr_robot.py --replay /home/yixuan/Dexmate/wbc/real/move_sideway.hdf5 --record
```



### Wheel odom

```bash
(yixuan_yifan) python /home/dexmate/yixuan/omniteleop_yifan/tests/test_head_zedx_depth.py

(yixuan_yifan) python scripts/drive_box_record.py --output /home/dexmate/yixuan/Dexmate/SLAM/test/box_closed.hdf5 --closed-loop-source odom

(yixuan_yifan) python scripts/misc/plot_drive_box.py --input /home/dexmate/yixuan/Dexmate/SLAM/test/box_closed.hdf5
```
