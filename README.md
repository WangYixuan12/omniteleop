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
  --save-dir /home/yixuan/Dexmate/data/raw_data
  # --debug-dir /home/yixuan/Dexmate/data/raw_data_debug

# for teleoperator: twist hand when pick, forward when place

(dexmate) python /home/yixuan/omniteleop/scripts/vis_episode.py --hdf5 /home/yixuan/Dexmate/data/raw_data/episode_0.hdf5

# per-take camera-latency audit (uses the recorded SDK capture stamps +
# meta/camera_ntp camera-host clock offsets, falling back to meta/ntp SoC NTP;
# jitter should stay ~1 camera frame; a WiFi-degraded take shows up here
# BEFORE it reaches training). Also accepts a directory or the porter's
# debug/timing sidecars. The publishers must run the current code -- they
# serve the sensors/<id>/clock service the recorder calibrates against.
(dexmate) python scripts/audit_episode_latency.py /home/yixuan/Dexmate/data/raw_data/episode_0.hdf5
```

**1.5** Generate the SceneDiff position condition (per-episode `[box, cloth]` 3D positions).

Diffs FRAME 0 of every raw + recovery episode (discovered by **globbing**
`~/Dexmate/data/raw_data/` + `.../recovery/` — **no** `log_index.csv`, which is produced later
by the porter) against the LAST frame of the fixed reference scene, and reduces the change
masks to the two objects' 3D positions in the **engage-origin WORLD frame** (same frame as
`action` and `observation.state[29:32]`). Order `[box=source, cloth=target]` is the world-frame
midpoint of `action/eef/{left,right}` at the successful grasp/release keyframe, Hungarian-matched
to the centroids with a distance + ambiguity guard — a bad episode is SKIPPED (see
`scene_diff/skip.json`) rather than silently mislabeled.

```bash
conda activate lerobot   # SAM3 runs here; make_wbc_before_hdf5 + extract auto-use dexmate_lerobot
bash /home/yixuan/scene_diff/run_wbc_pos_condition.sh
#   reference : last frame of ~/Dexmate/data/scene_diff/reference/episode_0.hdf5
#   positions : ~/Dexmate/data/scene_diff/positions/<source>/episode_<N>.npz   (porter input, step 2)
#   VERIFY    : ~/Dexmate/data/scene_diff/visualization/hungarian/<source>/episode_<N>.png
#               (green dot on the box, red arrow to the yellow cloth)
# subset e.g.: bash /home/yixuan/scene_diff/run_wbc_pos_condition.sh 1 25
```

> Both the box and cloth are randomized per episode so they differ from the fixed reference and
> SceneDiff detects both. Detection knobs in `scene_diff/configs/scenediff_config.yml`:
> `models.sam.mask_assignment_order: smallest_first` (keeps the box from being absorbed by the
> shelf/closet mask) and `detection.min_detection_pixel: 800` (drops small false-change regions —
> a remote e-stop, box sub-fragments — that were outranking the cloth; real box/cloth masks are
> ≥2680px). Recovery HDF5s keep their untrimmed frame 0, so the condition is read there even
> though the porter later trims their training window.

> **Position conditioning (Part B, implemented):** step 2's porter attaches these as a
> constant 6-D `observation.environment_state` via `--positions-dir ~/Dexmate/data/scene_diff/positions --object_nums 2`. Train wiring is step 3.

Verify train / valid / test **object-mask** coverage (image-space density on a shared head
backdrop). Masks for train+valid live under `scene_diff/{raw,recovery}/` (splits from the
porter's `log_index.csv`: train ↔ `processed_wbc/train`, valid ↔ `processed_wbc/test`).
Test = live rollout masks under `rollout/<run>/scene_diff/episode_*/`. WBC head frames are
already 240×320 → use full-frame crop `0 240 0 320` (SceneDiff resize 392×518).

```bash
# symlink scene_diff episodes into train/ / test/ by log_index.csv, then plot
LAYOUT=/tmp/wbc_scene_diff_vis_layout
rm -rf "$LAYOUT" && mkdir -p "$LAYOUT/train" "$LAYOUT/test"
(dexmate_lerobot) python - <<'PY'
import pandas as pd
from pathlib import Path
log = pd.read_csv("/home/yixuan/Dexmate/data/raw_data/log_index.csv")
sd, layout = Path("/home/yixuan/Dexmate/data/scene_diff"), Path("/tmp/wbc_scene_diff_vis_layout")
for _, r in log.iterrows():
    src, idx = r["source"], int(r["raw_data_index"])
    (layout / r["split"] / f"episode_{idx}").symlink_to(sd / src / f"episode_{idx}")
PY
(lerobot) python /home/yixuan/scene_diff/scripts/visualize_train_distribution.py \
  --scene-diff-root /tmp/wbc_scene_diff_vis_layout \
  --split train --valid-split test \
  --rollout-dir /home/yixuan/Dexmate/data/rollout/concat_abs_minmax \
  --reference-hdf5 /home/yixuan/Dexmate/data/scene_diff/_before/raw/episode_1.hdf5 \
  --crop 0 240 0 320 \
  --output /home/yixuan/Dexmate/data/visualization/wbc_train_test_distribution.png
#   -> ~/Dexmate/data/visualization/wbc_train_test_distribution.png
#      train (n=46) | valid=processed_wbc/test (n=2) | test=concat_abs_minmax rollouts (n=6)
```

**2.** Port raw HDF5 to a LeRobot dataset (add `--positions-dir` for position conditioning):

```bash
(dexmate_lerobot) python scripts/port_wbc_mobile_hdf5.py \
    --raw-dir ~/Dexmate/data/raw_data \
    --root ~/Dexmate/data/processed_wbc \
    --repo-id dexmate_wbc_eef_head \
    --include_recovery_data ~/Dexmate/data/raw_data/recovery \
    --log_index ~/Dexmate/data/raw_data/log_index.csv \
    --split-csv /home/yixuan/Dexmate/data/raw_data/split.csv \
    --positions-dir ~/Dexmate/data/scene_diff/positions --object_nums 2
```

`--positions-dir` (optional; omit for an unconditioned dataset) adds a **constant** 6-D
`observation.environment_state` = the two objects' world positions arranged `[box(src), cloth(dst)]`,
read from `<positions-dir>/<source>/episode_<N>.npz` (step 1.5 output; `source ∈ {raw, recovery}`
keys the file so a raw and a recovery episode sharing an index never collide). Every ported episode
**must** have its npz — the porter validates them all up front and aborts before writing/overwriting
anything if any is missing. Single-stage, so no per-frame `pos_condition_mask` is emitted. The
`meta/stats.json` `min`/`max` feed the `ENV: MIN_MAX` normalization at train time (step 3).

Visualize the ported dataset: `observation.state` (base-frame EEF/head FK composed to world via the odometry base pose, matching the calib sidecar), world-frame `action` targets.

```bash
(dexmate_lerobot) python scripts/vis_episode_processed_wbc.py \
  --dataset_dir /home/yixuan/Dexmate/data/processed_wbc/train/dexmate_wbc_eef_head \
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



Frame summary vs ManiFlow (world-frame state/cloud by default): see the comparison table under [Maniflow](#maniflow).

```bash
# change --job_name and --output_dir
# Position condition:
# update --policy.input_features and --policy.normalization_mapping and --policy.position_condition_mode=concat/film
# default config at [configuration_diffusion.py](../lerobot_original/src/lerobot/policies/diffusion/configuration_diffusion.py)
(dexmate_lerobot) lerobot-train \
  --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
  --policy.horizon=16 --policy.n_action_steps=8 --policy.use_relative_actions=false \
  '--policy.input_features={"observation.images.head_rgb": {"type": "VISUAL", "shape": [3, 120, 160]}, "observation.images.wrist_rgb": {"type": "VISUAL", "shape": [3, 120, 160]}, "observation.state": {"type": "STATE", "shape": [32]}, "observation.environment_state": {"type": "ENV", "shape": [6]}}' \
  '--policy.normalization_mapping={"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX", "ENV": "MIN_MAX"}' \
  '--policy.skip_normalization_dims={"observation.state": [3,4,5,6,7,8,13,14,15,16,17,18,23,24,25,26,27,28], "action": [3,4,5,6,7,8,13,14,15,16,17,18,23,24,25,26,27,28]}' \
  --policy.position_condition_mode=concat \
  --dataset.repo_id=dexmate_wbc_eef_head \
  --dataset.root=/home/yixuan/Dexmate/data/processed_wbc/train/dexmate_wbc_eef_head \
  --dataset.image_transforms.enable=true \
  --batch_size=32 --steps=100000 --save_freq=50000 --save_best=true \
  --output_dir=/home/yixuan/Dexmate/model/dp/dexmate_wbc_eef_head \
  --wandb.enable=true \
  --wandb.project=dexmate_wbc_mobile \
  --job_name=concat_abs \
  --dataset.val_root=/home/yixuan/Dexmate/data/processed_wbc/test/dexmate_wbc_eef_head --val_freq=5000
# --save_best=true overwrites output_dir/checkpoints/best whenever val loss improves
# (requires val_root + val_freq). Periodic save_freq checkpoints and checkpoints/last are unchanged.
# Prefer checkpoints/best/pretrained_model for rollout.
# The 6-D rotation dims (state/action 3-8, 13-18, 23-28) skip normalization; grippers and base x,y,yaw (state 29-31) stay min-max normalized.

# Position conditioning (needs a dataset ported with step 2 --positions-dir): the ENV input
# feature "observation.environment_state" [6] and the "ENV": "MIN_MAX" mapping above enable it.
# STATE/ACTION/ENV all use MIN_MAX (dataset min/max). Add --policy.position_condition_mode=film
# to route env through a separate identity-init per-block FiLM (default "concat" appends it to the
# shared global conditioning). Single stage -> stage_prediction_enabled stays false and NO
# observation.pos_condition_mask. Drop both ENV bits for an unconditioned run. For N>1 grasp cycles
# set the ENV shape to [object_nums*3] (condition_dim follows the feature shape, so the whole vector
# conditions the model); per-stage SELECTION would additionally need observation.pos_condition_mask
# + --policy.stage_prediction_enabled=true (out of scope here).
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
  --dataset.root=/home/yixuan/Dexmate/data/processed_wbc/train/dexmate_wbc_eef_head \
  --dataset.image_transforms.enable=true \
  --batch_size=32 --policy.optimizer_lr=2e-4 --policy.optimizer_lr_backbone=2e-5 --steps=200000 --save_freq=50000 --save_best=true --log_freq=2000 \
  --output_dir=/home/yixuan/Dexmate/model/act/dexmate_wbc_eef_head \
  --wandb.enable=true --wandb.project=dexmate_wbc_mobile --job_name=act_pos_abs \
  --dataset.val_root=/home/yixuan/Dexmate/data/processed_wbc/test/dexmate_wbc_eef_head --val_freq=5000

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
Mixture-of-Frames recipe as [ManiFlow's](#maniflow) `action_frame_mode=mof` and the
[legacy standalone mofpo run](#legacy-standalone-mof-reproduction). Both integrated policies
default to the **same four** experts, dropping `base` as a reparameterization of
`base_rel_trans` (ManiFlow ADR 0002 amendment); `base` stays selectable for the five-tower
set. `world` is the default `action_frame_mode` and is byte-identical to before the feature.
Implementation:
`../lerobot_original/src/lerobot/policies/diffusion/mof/` (`action_layout.py` +
`frame_transforms.py` are mirrored copies of the ManiFlow/mofpo modules; `stats.py` and
`mixture.py` are LeRobot-native).

**Same dataset as the world run** — no re-port. The porter stores `observation.state[:29]`
base-frame, and MoF lifts it to world in-graph from the planar base pose at `29:32`
(`lift_base_state_to_world`, verified equal to the independently ported world state in both
`dexmate_wbc_eef_head_rel` and the ManiFlow zarr to `1.2e-7`). So DP-world vs DP-mof differ in
the action-frame machinery and **nothing else**.

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
# measured at FIVE experts: 361M params, 6.88 GB at batch 32, 0.11 s/step on the 4090
#   (~3 h for 100k steps). Re-measure at the four-expert default before quoting.
# defaults: mof_enabled_experts=[base_rel_trans,left,right,rel_traj]
#           mof_canonical_space=base_rel_trans  mof_router_mode=learned
#           mof_expert_loss_coef=1.0            mof_state_frame=base
#           mof_env_frame=world
# extra logged metrics: diffusion_loss, expert_loss_native/<expert>, router_weight/<expert>,
#                       router_entropy (ln 4 = 1.386 at init = uniform routing)
```

Six flags differ from the world command above, and five of them are **required** — the config
raises with the fix in the message if you forget:

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
- `--policy.down_dims=[256,512,1024]` (mofpo's paper recipe). Five copies of LeRobot's default
`[512,1024,2048]` measured **1.31B params ≈ 19.5 GiB** of weights+grad+Adam and would not fit a
24 GB card; four copies has not been re-measured, and the paper recipe is the reason to use this
width regardless. The incumbent world baseline trains at LeRobot's default `[512,1024,2048]`, so
**re-run it at** `[256,512,1024]` **(90M)** before reading any DP-world vs DP-mof delta — otherwise
the two differ by 5.7x in width as well as in action-frame machinery.
- `DDIM` + `--policy.num_inference_steps=16`: four towers per denoise step, so the DDPM-100
default would blow the 0.4 s chunk budget. mofpo measured ~0.6 s/chunk for 16-step DDIM × 5
experts on a shared GPU — check `--policy-interval` against the real number before hardware.
- `use_relative_actions` and `stage_prediction_enabled` are rejected (both are answered by
`mof_canonical_space`, resp. need the single `global_cond` MoF replaces with one per family).

**Position-condition frame.** `--policy.mof_env_frame` decides what each obs family sees of the
task. `world` (default) is the incumbent behaviour — one world-frame env-state rides every
family's conditioning vector. `family` re-expresses it into that family's per-step frame, which
matters because `reexpress_agent_pos` zeroes the base block of a robot-anchored family's state:
without it, a `base`/`left`/`right` tower holds a world-frame object position it cannot relate to
its own state. This is the transferable half of ManiFlow's `[mof.cloud_frame](#maniflow)` — the
scene half does not cross over, since an RGB image carries no frame. There is no separate
ablation flag; the ablation is this flag's two values. See `docs/adr/0002-dp-mof-reexpresses-the-position-condition.md`.

**Object masks are not supported here.** ManiFlow's `point_sampling_mode` has no counterpart (a
ResNet has no point-budget sampling stage), and `mask_channels=true` is designed but deferred —
see `docs/adr/0003-dp-mask-channels-are-a-separate-image-feature.md`. A DP-vs-ManiFlow pair must
therefore be matched with `point_sampling_mode=fps mask_channels=false` on the ManiFlow side.

**Parity pre-flight** before trusting a MoF run — reduces the mixture to the incumbent model:
`--policy.mof_enabled_experts=[world] --policy.mof_canonical_space=world --policy.mof_router_mode=fixed_uniform`.

Deploy and offline validation are **unchanged commands**: `wbc_policy_rollout.py` reads
`action_frame_mode` off the checkpoint and takes `observation.state`'s frame from
`mof_state_frame` instead of inferring it from `use_relative_actions`; the base-frame state it
already builds is what a default MoF checkpoint wants. `vis_wbc_policy_prediction.py` feeds the
dataset state verbatim and needs nothing. Tests:
`../lerobot_original/tests/policies/test_diffusion_mof.py` (42).

**Relative actions (optional).** To train with `use_relative_actions=true` (each chunk action relative to the chunk's current observation, mirroring `train_dexmate_diffusion.sh`), `observation.state` EEF/head must share the world action frame — LeRobot's relative step subtracts `observation.state[:29]` from the world action, so a base-frame state would be cross-frame. Re-port with `--relative-actions` (world-frame state; default repo-id `dexmate_wbc_eef_head_rel`):

```bash
(dexmate_lerobot) python scripts/port_wbc_mobile_hdf5.py \
    --raw-dir ~/Dexmate/data/raw_data \
    --root ~/Dexmate/data/processed_wbc \
    --repo-id dexmate_wbc_eef_head_rel \
    --relative-actions \
    --include_recovery_data ~/Dexmate/data/raw_data/recovery \
    --log_index ~/Dexmate/data/raw_data/log_index_rel.csv
```

Then recompute action stats over the relative distribution and train with the relative flags. Translations AND 6-D rotations are made relative; only the binary grippers stay absolute (`relative_exclude_dims` `[9, 19]` — the raw-FC03 state gripper is on a different scale than the 0/1 command). The column-wise rotation delta is not itself a rotation, but the post-processor adds the same chunk-anchor state back before the Gram-Schmidt decode, so it is exactly invertible; rotation dims still skip normalization (relative deltas concentrate near 0, where a min/max rescale would amplify noise). `chunk_size` = horizon; `reference_offset` = `n_obs_steps − 1` = 1 for diffusion. For **ACT** the anchor is the chunk start (`action_delta_indices = range(0, chunk_size)`), so use `--operation.reference_offset=0` and `--operation.chunk_size=<--policy.chunk_size>`.

```bash
REL_ROOT=/home/yixuan/Dexmate/data/processed_wbc/dexmate_wbc_eef_head_rel
REL_EXCLUDE='[9,19]'
(dexmate_lerobot) lerobot-edit-dataset \
  --repo_id=dexmate_wbc_eef_head_rel --root=$REL_ROOT --new_root=$REL_ROOT \
  --operation.type=recompute_stats --operation.overwrite=true \
  --operation.relative_action=true --operation.chunk_size=16 --operation.reference_offset=1 \
  --operation.relative_exclude_dims="$REL_EXCLUDE" --operation.num_workers=4

(dexmate_lerobot) lerobot-train \
  --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
  --policy.horizon=16 --policy.n_action_steps=8 --policy.use_relative_actions=true \
  '--policy.relative_exclude_dims={"action": [9, 19]}' \
  '--policy.input_features={"observation.images.head_rgb": {"type": "VISUAL", "shape": [3, 120, 160]}, "observation.images.wrist_rgb": {"type": "VISUAL", "shape": [3, 120, 160]}, "observation.state": {"type": "STATE", "shape": [32]}, "observation.environment_state": {"type": "ENV", "shape": [6]}}' \
  '--policy.normalization_mapping={"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX", "ENV": "MIN_MAX"}' \
  '--policy.skip_normalization_dims={"observation.state": [3,4,5,6,7,8,13,14,15,16,17,18,23,24,25,26,27,28], "action": [3,4,5,6,7,8,13,14,15,16,17,18,23,24,25,26,27,28]}' \
  --dataset.repo_id=dexmate_wbc_eef_head_rel --dataset.root=$REL_ROOT \
  --batch_size=32 --steps=100000 --save_freq=50000 \
  --output_dir=/home/yixuan/Dexmate/model/dp/dexmate_wbc_eef_head_rel
```



`scripts/wbc_policy_rollout.py` and `scripts/vis_wbc_policy_prediction.py` auto-detect `use_relative_actions` from the checkpoint and build world-frame state + chunk-anchor de-relativization accordingly — same commands as below (point `--policy-path` at the relative checkpoint).

**4.**

Validate the checkpoint OFFLINE before any hardware rollout — run the rollout's exact inference path (_PolicyBundle.select_action + split_policy_action) over a processed episode and overlay the policy's predicted 29-D world-frame action against the recorded ground-truth action. Rerun shows per-entity (left/right/head) GT (red) vs predicted (magenta) pose + error line, GT/pred EEF reprojected into the head image, gripper series, the base odometry path, and the depth point cloud; matplotlib + stdout report per-entity translation/rotation MAE. Run in dexmate_lerobot:

```bash
(dexmate_lerobot) python scripts/vis_wbc_policy_prediction.py \
  --policy-path ~/Dexmate/model/dp/nopos_abs_minmax/checkpoints/last/pretrained_model \
  --dataset_dir ~/Dexmate/data/processed_wbc/test/dexmate_wbc_eef_head \
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
python scripts/wbc_policy_rollout.py --replay-episode ~/Dexmate/data/raw_data/episode_1.hdf5 \
    --align-reference ~/Dexmate/data/raw_data/reference.hdf5

(dexmate_lerobot) python scripts/wbc_policy_rollout.py \
  --policy-path /home/yixuan/Dexmate/model/dp/nopos_abs_minmax/checkpoints/last/pretrained_model \
  --save-dir ~/Dexmate/data/rollout/nopos_abs \
  --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5 \
  --max-seconds 100
# Ctrl+C once to save
# if the console warns "buffer will run dry" (100-step DDPM is ~0.4 s/chunk),
# lower --policy-interval (e.g. 0.3) or reduce the checkpoint's num_inference_steps

# A checkpoint trained with observation.environment_state automatically bootstraps the scene.
# Launch conditioned LeRobot policies from dexmate_maniflow: validation uses in-process SAM3.1.
# After engage and --align-reference, rollout captures one head frame, runs SceneDiff against
# the same reference used for training, displays a fresh size-slot overlay, and requires an
# explicit [source box, destination cloth] permutation before motion. Positions are then pinned
# for the episode. An unconditioned checkpoint skips SceneDiff and the operator prompt.
(dexmate_maniflow) python scripts/wbc_policy_rollout.py \
  --policy-path /home/yixuan/Dexmate/model/dp/concat_abs_minmax/checkpoints/last/pretrained_model \
  --save-dir ~/Dexmate/data/rollout/concat_abs_minmax \
  --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5 \
  --max-seconds 100
#   --reference-hdf5 ~/Dexmate/data/scene_diff/_before/reference_last.hdf5   (default)
#   --prompt-before-capture   pause to stage the scene after alignment, before the head grab
#   SceneDiff outputs -> <save-dir>/scene_diff/episode_N/{live_capture.hdf5,episode_0.npz,
#                        bootstrap_size_slots.npz,deploy_slot_overlay.png,validation_*.{npz,json}}
#   If that final episode directory already exists, rollout exits before capture and asks you
#   to delete it manually. A failed bootstrap never publishes a partial final directory.

python scripts/vis_episode.py --hdf5 /home/yixuan/Dexmate/data/rollout/concat_abs_minmax/episode_0.hdf5
```



## ManiFlow

[ManiFlow](https://maniflow-policy.github.io/) (CoRL 2025) trained on the **same raw takes** as the
diffusion/ACT pipeline above, as a **3D point-cloud** policy. Repo: `/home/yixuan/ManiFlow_Policy`.
The zarr's `action` is **bit-identical** to the LeRobot dataset; `data/state` is the same 32-D layout
but defaulted to world frame (see table). Visual obs is a world-frame head point cloud instead of RGB.

Action-frame modeling is a config choice inside this policy family: `action_frame_mode=world` is the
incumbent world-absolute ManiFlow path and remains the default; `action_frame_mode=mof` enables the
integrated Mixture-of-Frames path, four experts by default
(`base_rel_trans, left, right, rel_traj`; add the fifth back with
`mof.enabled_experts=[base,base_rel_trans,left,right,rel_traj]`). Both modes use the same zarr,
trainer, checkpoint format, deployment bundle, and `dexmate_maniflow` environment.

> The LeRobot diffusion policy carries the **same switch** with the same four experts, canonical
> space, router and native-frame aux loss — see [Mixture of Frames](#mobile-conditioned-policy-pipeline)
> in step 3 above. The two implementations are independent (separate envs, mirrored frame-transform
> modules), so a DP-mof vs ManiFlow-mof comparison isolates the visual representation (RGB vs point
> cloud) with the action-frame machinery held fixed.
>
> Two knobs here have no DP counterpart, so a matched pair must pin them: run the ManiFlow side at
> `point_sampling_mode=fps mask_channels=false` (DP mask channels are deferred) and at
> `position_condition_mode=concat` (`grounding_tokens` needs `env_dino`, which the LeRobot porter
> does not write). `mof.cloud_frame=family` **does** have a counterpart —
> `--policy.mof_env_frame=family` — but it moves only the position condition, not the scene. The
> matched cell must also run on the legacy `processed_wbc/maniflow` zarr: `maniflow_v3` carries no
> `env_state`.

```
data/point_cloud  (T, 1024, 6) float32   engage-origin WORLD XYZ + RGB[0,1]
data/state        (T, 32)      float32   agent_pos  (WORLD frame by default)
data/action       (T, 29)      float32   verbatim ik.solve targets (WORLD)
data/env_state    (T, 6)       float32   optional constant [box(src), cloth(dst)] XYZ
data/env_dino     (T, 2560)    float32   optional frame-0 DINOv3 features, task-slot order
data/point_mask   (T, 1024)    int8      optional 0=background, 1=box/src, 2=cloth/dst
meta/episode_ends (E,)         int64
```

**Frame conventions (defaults)** — engage-origin WORLD throughout except DP's egocentric EEF/head:


|                                                | DP / `port_wbc_mobile_hdf5.py`                                           | ManiFlow / `port_wbc_mobile_zarr.py`                    |
| ---------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------- |
| Obs EEF/head (`state[:29]` / `agent_pos[:29]`) | **base** (egocentric FK; do not compose `world_T_base` into `[:29]`)     | **world** (`compose(base, odom)`)                       |
| Obs base pose (`state[29:32]`)                 | world (`obs/base/pose`)                                                  | world (same)                                            |
| Action (29)                                    | **world** (verbatim `ik.solve` targets)                                  | **world** (bit-identical)                               |
| Visual obs                                     | RGB `head_rgb` (+ optional `wrist_rgb`)                                  | world-frame head point cloud                            |
| Env state (optional `--positions-dir`)         | world `[box, cloth]`                                                     | world (same)                                            |
| Frame override                                 | `--relative-actions` → state in **world** (needed for relative training) | `--state-frame base` → egocentric state (LeRobot-style) |




### Model pipeline

Shared dimensions: batch `B=64`, observation steps `To=2`, stored points `N=1024`, policy points `P=256`, task slots `K=2`, XYZRGB channels `C=6` (or 8 with mask painting), point feature width `Fp=128`, state feature width `Fs=64`, visual width `Do=192`, DiTX width `D=768`.

```diff
point_cloud (B*To, 1024, 6) + optional point_mask (B*To, 1024)
 --normalize XYZ (dataset min/max, mode=limits; sampling runs here)-->
+ --point_sampling_mode-->
+   ├─ fps: pytorch3d scene-wide FPS -> 256
+   └─ mask_stratified (requires point_mask):
+        1. split 1024 into src / dst / background by point_mask
+        2. target quotas 64 / 64 / 128
+        3. unused object quota -> other object, then background;
+           if background is short, give leftover slots back to src then dst
+        4. FPS inside each group
+        5. merge picks and sort indices back to stored-point order
+ -> selected points (B*To, 256, 6) + selected mask labels when point_mask is present
+ --mask_channels-->
+   ├─ false: keep XYZRGB -> (B*To, 256, 6)
+   └─ true (requires point_mask): append [is_src, is_dst] -> (B*To, 256, 8)
  --PointNet pointwise-->
point_feat (B*To, 256, 128)
```

Mask sampling / `mask_channels` are orthogonal to the mode below and may be on in any of them.

`point_mask` stores **one of three mutually exclusive classes** per point as int8 —
`0` background, `1` box/src, `2` cloth/dst — because the sampler needs to tell the two objects
apart to fill its `64/64/128` quotas. The network never sees that integer: `mask_channels=true`
expands it to **two binary channels** `[label == 1, label == 2]`, so src is `[1,0]`, dst is `[0,1]`
and background is `[0,0]` (no third channel). The same 0/1/2 legend is what the full-frame SAM3.1
mask NPZs carry.

#### `position_condition_mode=none`

No additional task specification.

```text
point_feat (B*To, 256, 128)
agent_pos (B*To, 32)
  --state MLP + repeat over points--> state_feat (B*To, 256, 64)

cat(point_feat, state_feat)
  -> visual (B*To, 256, 192)
  -> vis_cond (B, To*P=512, 192)
  -> DiTX visual context (B, 512, 768)
```



#### `position_condition_mode=concat`

Cheap position-only comparison.
Normalized `env_state=[src_xyz,dst_xyz]` widens `agent_pos` 32→38 before the state MLP.

```diff
point_feat (B*To, 256, 128)
agent_pos (B*To, 32)
+ --concat normalized env_state=[src_xyz,dst_xyz]--> (B*To, 38)
  --state MLP + repeat over points--> state_feat (B*To, 256, 64)

cat(point_feat, state_feat)
  -> visual (B*To, 256, 192)
  -> vis_cond (B, To*P=512, 192)
  -> DiTX visual context (B, 512, 768)
```



#### `position_condition_mode=grounding_tokens`

Pinned to `object_nums=2`. Validates per-episode constancy, takes frame 0, and appends two
role-tagged tokens after the visual context (no observation-step embedding — they name the
episode's source/destination). In the full ladder cell, current object evidence still comes from
the tracked mask on the point cloud; this mode can also be ablated with plain FPS and no mask
channels.

```diff
point_feat (B*To, 256, 128)
agent_pos (B*To, 32)
  --state MLP + repeat over points--> state_feat (B*To, 256, 64)

cat(point_feat, state_feat)
  -> visual (B*To, 256, 192)
  -> vis_cond (B, To*P=512, 192)
  -> DiTX visual context (B, 512, 768)

+ env_dino+env_state frame-0 slots --> raw [DINOv3_1280 | xyz_3] × K=2
+   --LayerNorm(1280)→Linear(1280,64), cat XYZ, project D=768-->
+   --+ src/dst role embeddings + normalize-->
+   --append after visual tokens--> DiTX context (B, 514, 768)
```

The wide `env_state`/`env_dino` constants are removed before encoder flattening in both training and
inference. Mask consumers accept only labels `{0,1,2}`. `position_condition_dropout` stays zero: the
pipeline has no classifier-free-guidance inference path, and nonzero dropout would also mismatch the
training student with the eval-mode EMA teacher.

Other deliberate deviations from the LeRobot pipeline:

- **Workspace crop, measured not guessed:** `x∈[0.20,1.70] y∈[-0.90,1.05] z∈[0.60,1.32]` (world,
`src/omniteleop/wbc_pointcloud.py`). FPS spreads a fixed budget ~uniformly in space, so every cubic
metre of room the crop keeps is budget stolen from the two small objects. **Task-specific** — a new
object layout needs new `--crop-min/--crop-max`.
- **Normalization is ManiFlow's** `mode='limits'` (per-dim min–max) for point cloud, state and
action, including the 6-D rotations. `env_state` deliberately reuses the point-cloud XYZ affine
(tiled once per task slot; under `mof.cloud_frame=family`, the matching family's
`{family}__point_cloud` XYZ affine); `env_dino` and integer mask labels use identity normalizers.
DiTX applies LayerNorm to the DINO block in-graph.
- **Online augmentation is RGB-only:** when `use_pc_color=true`, 20% of training windows receive one
temporally consistent brightness/contrast/saturation/hue transform (`[0.3, 0.4, 0.5, 0.08]`). It is
applied before normalization and disabled for validation. XYZ, state, action, masks, `env_state` and
`env_dino` are not augmented; there is no online crop, rotation, translation or point jitter.
- **Pointcloud frame**

Two crop bounds are **not** free to tighten:

- `x_min` ≤ ~0.30 — the box is carried **back toward the robot** to `x = 0.334` before being placed
(bimanual EEF midpoint while both grippers are closed). `x_min = 0.45` crops the grasped box out of
the observation on 17% of transport frames; `x_min = 0.60` on 42%.
- `z_min` ≤ ~0.62 — bounded by the **cloth** (`z ≈ 0.66–0.70`), not the box. `z_min = 0.70` halves the  
cloth's points (40 → 22) while helping the box.



### Environment

Use `dexmate_maniflow` **for every integrated ManiFlow stage**: zarr build/inspection, world or MoF
training, checkpoint inspection, SceneDiff DINO extraction, offline SAM3.1 mask generation, offline
inference, and live rollout. The live SceneDiff change-detection bootstrap remains a short-lived
subprocess, but causal SAM3.1 tracking and ManiFlow inference share the `dexmate_maniflow` process.

The old generic `maniflow` environment has been retired. `dexmate_mof` is retained only for the
legacy standalone mofpo reproduction documented at the end of this README. See
[MANIFLOW_ENVIRONMENT.md](MANIFLOW_ENVIRONMENT.md) for the full bootstrap recipe.

### Run it from scratch

Everything runs in `dexmate_maniflow` except where a line says otherwise.

**0.** Record raw takes — "Mobile conditioned policy pipeline" step 1:

```bash
(dexmate) python scripts/wbc_vr_leader.py --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5
(dexmate) python scripts/wbc_vr_robot.py --record --save-dir ~/Dexmate/data/raw_data
# -> raw_data/{,recovery/}episode_<N>.hdf5
```

**1.** Build the SceneDiff artifacts. Positions first; masks and DINO both read them.

```bash
cd ~/scene_diff
# 1a. positions: [box, cloth] world XYZ per episode.
# run_wbc_pos_condition.sh defaults: RAW_ROOT=~/Dexmate/data/raw_data,
# REFERENCE_SRC + OUTPUT_ROOT under ~/Dexmate/data/scene_diff.
(lerobot) bash run_wbc_pos_condition.sh
# -> scene_diff/positions/<source>/episode_<N>.npz
# VERIFY scene_diff/visualization/hungarian/<source>/episode_<N>.png
# (green dot on the box, red arrow to the yellow cloth); skipped episodes land in skip.json

# 1b. SAM3.1 masks, ~30 s/episode on the 4090. Skip 1b (and --masks-dir below) for
# fps / mask_channels=false checkpoints.
(dexmate_maniflow) python scripts/build_wbc_masks.py \
  --scenediff-root ~/Dexmate/data/scene_diff \
  --masks-dir ~/Dexmate/data/scene_diff/masks
# check masks/qc_summary.json and masks/viz/ before porting

# 1c. DINOv3 sidecars, written beside each positions NPZ. 
(lerobot) python scripts/extract_object_dino_feats.py \
  --scenediff-root ~/Dexmate/data/scene_diff \
  --config configs/scenediff_config.yml
```

**2.** Port to the ManiFlow zarr:

```bash
(dexmate_maniflow) cd ~/omniteleop && python scripts/port_wbc_mobile_zarr.py \
  --raw-dir ~/Dexmate/data/raw_data \
  --include-recovery-data ~/Dexmate/data/raw_data/recovery \
  --out-root ~/Dexmate/data/processed_wbc/maniflow \
  --positions-dir ~/Dexmate/data/scene_diff/positions \
  --masks-dir ~/Dexmate/data/scene_diff/masks \
  --dino
# ~3 min, CPU. -> dexmate_wbc_{train,test}.zarr + .meta.json, split by raw_data/split.csv
# --positions-dir -> data/env_state (T, 6)     --dino -> data/env_dino (T, 2560)
# --masks-dir     -> data/point_mask (T, 1024), exact stored-point lineage
# drop --masks-dir for an fps-only zarr; env_state/env_dino do not depend on masks
```

- `--out-root` must not already exist. The porter validates every source before creating it and
removes the root it created if conversion fails; it never reuses or overwrites one.
- The stored 1024 points are downsampled by the encoder to `visual_cond_len` (256), so raising
`--num-points` only helps if you raise `visual_cond_len` too.
- The optional arrays are additive — each policy mode loads only the keys it consumes, so one zarr  
serves the whole ladder including the unconditioned control. The sibling `.meta.json` (crop/depth/  
frame/RNG/sampler parameters, camera contract, mask/DINO provenance, ordered raw-frame manifest) is  
the source of truth for live preprocessing and visualization.

**3.** Inspect what the policy will actually consume. The viewer reads the sampler, `mask_channels`,
`position_condition_mode`, `visual_cond_len`, CUDA device and training-zarr normalizer strictly from
`<run>/.hydra/config.yaml` — no manual sampler overrides — so a one-epoch run is enough to look:

```bash
(dexmate_maniflow) bash ~/ManiFlow_Policy/scripts/train_dexmate_wbc.sh smoke-mask_mof  \
  mof.cloud_frame=family action_frame_mode=mof point_sampling_mode=mask_stratified mask_channels=true \
  position_condition_mode=grounding_tokens training.num_epochs=1

(dexmate_maniflow) python scripts/vis_episode_processed_wbc.py \
  --zarr ~/Dexmate/data/processed_wbc/maniflow/dexmate_wbc_test.zarr \
  --maniflow-run-dir ~/Dexmate/model/maniflow/maniflow_smoke-mask_mof_seed0 \
  --episode_index 0
```

```text
Scene tab:
  Stored RGB 3D | Policy RGB 3D | Policy masks 3D (only when point_mask is consumed)
  Head RGB+mask | Wrist RGB     | empty
  Left gripper  | Right gripper | Base pose (x, y, yaw)

MoF frames (GT action) tab — only when action_frame_mode=mof:
  one 3D view per enabled expert (base / base_rel_trans / left / right / rel_traj / …)
```

- `Stored RGB 3D` is all 1024 porter points; `Policy RGB 3D` is only the run sampler's deterministic
selection on training-normalized XYZ, drawn back at raw world XYZ/RGB; `Policy masks 3D` colors the
selected labels source=orange, destination=cyan, background=gray and warns whenever a mask-stratified
frame falls below the nominal `64/64/128` quota.
- Both tabs share the `frame=t` scrub. Scene shows that one frame; MoF treats `t` as the last obs step
and draws the training `horizon` GT chunk (`t−(To−1)` … `+horizon`, refs from `agent_pos[t]`,
edge-replicated at episode bounds) — GT only, this viewer has no checkpoint.
- Head RGB/depth, Wrist RGB and the full SAM3.1 mask come from the raw HDF5/mask NPZ through the
manifest. Depth and projected EEF pixels are logged but hidden; enable them from the Rerun sidebar.

**4.** Train. `train_maniflow_robotwin_workspace.py` is ManiFlow's only env-runner-free trainer, which
is why the task config lives under `config/robotwin_task/` (a Hydra group name *is* the config key) —
nothing about it is RoboTwin:

```bash
# mof.expert_arch = full_ditx | compact_ditx
(dexmate_maniflow) bash ~/ManiFlow_Policy/scripts/train_dexmate_wbc.sh baseline \
  mof.cloud_frame=world action_frame_mode=world point_sampling_mode=fps mask_channels=false position_condition_mode=none \
  training.num_epochs=200 training.checkpoint_every=10 training.val_every=5

(dexmate_maniflow) bash ~/ManiFlow_Policy/scripts/train_dexmate_wbc.sh mof_pos \
  mof.cloud_frame=family action_frame_mode=mof point_sampling_mode=mask_stratified mask_channels=true \
  position_condition_mode=grounding_tokens \
  training.num_epochs=200 training.checkpoint_every=10 training.val_every=5

# args: <addition_info> [seed] [gpu_id] [hydra overrides...]
# checkpoints -> ~/Dexmate/model/maniflow/<exp>_seed<seed>/checkpoints/ (+ .hydra/config.yaml)
# default zarr: .../processed_wbc/maniflow/dexmate_wbc_train.zarr, override with ZARR_PATH=
# training.resume: relaunching the same tag+seed continues from its latest.ckpt
```

**How many epochs?** The diffusion policy that rolled out successfully did 100k steps × batch 32 =
**3.2M samples** over 17,951 frames — 178 epochs-equivalent. ManiFlow has 16,659 train windows at
batch 64 → **261 steps/epoch**:


| `num_epochs`       | steps      | samples   | vs DP     |
| ------------------ | ---------- | --------- | --------- |
| 192                | 50,112     | 3.20M     | 1.00x     |
| **300**            | **78,300** | **5.00M** | **1.56x** |
| 1010 (cfg default) | 263,610    | 16.8M     | 5.26x     |


Use **300**, not the sample-matched 192: every batch splits 75% flow / 25% consistency, so at 300
epochs the flow branch alone sees 3.75M samples ≈ 1.17x DP's whole budget. Watch `val_loss` (every 5
epochs) and `train_action_mse_error`; the `topk` manager keeps the best-val checkpoint beside
`latest.ckpt`.

> ⚠️ `num_epochs` sets the **cosine LR horizon**. Choose it at launch: stopping a longer-horizon run
> early does not produce the same schedule as training to that epoch count. Also do not use
> `training.debug=true` as a smoke test — the workspace rewrites it to 100 epochs × 10 steps, and
> `position_condition_dropout` must stay `0.0` (any nonzero value fails at policy construction).

Implementation lives in `point_process.py` (mask-stratified sampler), `pointnet_extractor.py`
(mask-aware sampling/channels), `ditx.py` (grounding tokens), `maniflow_pointcloud_policy.py` (mode
plumbing), `dexmate_wbc_dataset.py` (schema/normalizers), `action_layout.py` + `frame_transforms.py`
(MoF representations), `mof_mixture.py` (expert towers and router), and `train_dexmate_wbc.sh`.

**5.** Validate OFFLINE before any hardware run. `scripts/vis_wbc_maniflow_prediction.py` replays one
held-out zarr episode through the checkpoint's own `policy.predict_action` — the exact call the rollout
makes — and scores the predicted world actions against the recording:

```bash
(dexmate_maniflow) python scripts/vis_wbc_maniflow_prediction.py \
  --policy-path ~/Dexmate/model/maniflow/<exp>_seed0/checkpoints/<epoch>.ckpt \
  --zarr ~/Dexmate/data/processed_wbc/maniflow/dexmate_wbc_test.zarr \
  --episode_index 0 --save ~/Dexmate/tmp/scratchpad/maniflow_pred_ep0.rrd
# per-entity translation MAE / rotation error / gripper accuracy on stdout, a GT-vs-pred
# overlay PNG, and a Rerun recording (--save for SSH, --connect for a running viewer)
# --stride (default n_action_steps), --num-inference-steps, --no-ema, --seed, --train-zarr
```

```text
Scene tab:
  Stored cloud (1024) | Policy cloud (visual_cond_len, run's sampler)
                      | Policy cloud: mask labels (only when point_mask is consumed)
  Head RGB/depth | Wrist RGB | Left gripper | Right gripper
  every 3D view carries the GT (red) and predicted (magenta) action horizons, their
  first-step error segment, the base odometry path and the head camera frustum

MoF frames (GT vs pred) tab — only when action_frame_mode=mof:
  the same chunk pair in every enabled expert, round-trip-checked back to world
MoF router tab: learned expert weights per flow step (heatmap + per-chunk means)
```

- Observations come verbatim from the zarr, so no SceneDiff tracker or live cloud builder is involved.
It is chunk-wise open loop: nothing is fed back, since an offline replay cannot move the robot.
- The rendered policy cloud uses the **checkpoint's own** cloud normalizer statistics
(`point_cloud`, or `{family}__point_cloud` when `mof.cloud_frame=family`), and predicted grippers
are binarized at `GRIPPER_BINARY_THRESHOLD` the way `split_policy_action` does before actuation —
so both the points and the gripper accuracy are what deployment would produce.
- The training zarr comes from the run's cfg (falling back to the same filename beside `--zarr`),
and its preprocessing is checked against the evaluation zarr's, because the normalizer was fit there.

**6.** Live rollout. `scripts/wbc_maniflow_rollout.py` does **not** fork the hardware loop: it imports
`wbc_policy_rollout.py` and swaps in a ManiFlow bundle + observation worker, so engage,
`--align-reference`, `--replay-episode`, the async chunk scheduler, stale-action drop, the 100 Hz WBC
tick, the hold watchdog, recording and shutdown are the same code. The bundle instantiates the
checkpoint's own configuration, so this one command loads `world` or `mof` checkpoints alike; do not
use the legacy `wbc_mof_rollout.py` for an integrated ManiFlow checkpoint.

> **Deployment status:** the live code path is implemented and covered by CPU-side tests, but it is
> **not robot-ready** until every pinned-GPU gate below has passed.
> Unit tests alone are not permission to move the robot.

```bash
# Run this only after step 5's offline gate passes.
(dexmate_maniflow) python scripts/wbc_maniflow_rollout.py \
  --policy-path ~/Dexmate/model/maniflow/<exp>_seed0/checkpoints/latest.ckpt \
  --pointcloud-meta \
    ~/Dexmate/data/processed_wbc/maniflow/dexmate_wbc_train.meta.json \
  --save-dir ~/Dexmate/data/rollout/maniflow \
  --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5 \
  --num-inference-steps 2 --policy-interval 0.4 --max-seconds 85
```

- `--policy-path` is a ManiFlow `.ckpt` **file** (`torch.load` + `dill`), not a LeRobot dir; EMA
weights by default (`--no-ema` for raw). `--num-inference-steps 2` is enough for the consistency-flow
objective, but the co-resident replay below — not a standalone timing estimate — decides whether the
step count and interval are safe.
- `--pointcloud-meta` is a relocation override only. The v3 sidecar supplies image size, intrinsic,
dtypes, depth scale, crop, frame, RNG seed/scheme, pool size and pinned sampler signature, and any
mismatch (including `--dataset-fps`) fails before engage. There are no live pool-size, seed, sampler
or conditioning overrides: the checkpoint is the sole mode authority.


| Checkpoint configuration                      | Raw observation keys supplied by rollout         |
| --------------------------------------------- | ------------------------------------------------ |
| `none + fps + mask_channels=false`            | `point_cloud`, `agent_pos`                       |
| `position_condition_mode=concat`              | above + episode-constant `env_state`             |
| `position_condition_mode=grounding_tokens`    | above + episode-constant `env_state`, `env_dino` |
| `mask_stratified` and/or `mask_channels=true` | above as applicable + causal `point_mask`        |
| any row with `action_frame_mode=mof`          | identical keys; no rollout-side frame transform  |


`policy.predict_action` owns normalization, mask-stratified sampling, mask painting, grounding-token
construction, MoF expert transforms and world-action reconstruction; deployment supplies raw
observations only.

For any nonempty scene requirement, rollout runs one ordered, fail-closed bootstrap before motion:
SceneDiff exports size-slot positions, the native SAM3.1 seed and (when required) DINO features in a
temporary directory; the operator inspects a fresh overlay and enters the permutation to
`[source box, destination cloth]`, which reorders positions, DINO and seed labels together (identity is
never assumed); a disposable tracker session validates both slots against valid depth and their stored
world positions (≥95% valid-depth support, ≤20 mm error) and aborts before motion on failure; only then
is the directory published as `<save-dir>/scene_diff/episode_N` (an existing path aborts before
capture).

Mask-free conditioned modes unload SAM3.1 after validation; mask-consuming modes keep the compiled
model but seed a **fresh** rollout session, gathering and validating all `n_obs_steps` samples before
advancing the tracker — a partial window advances neither
tracker state nor sampling RNG — and label points through the porter's exact pixel → valid depth → crop
→ pool → FPS lineage. Object absence is valid; stale-mask reuse and fallback samplers are not. Invalid
mask shapes/dtypes/labels, timestamp regression, world-epoch change, tracker failure, or a slot
covering ≥80% of the image for three consecutive consumed frames aborts through safe hold. Run
`scripts/diagnostics/qc_wbc_live_masks.py` after each take to review the recorded masks and telemetry.

Before robot use, produce and review all of the following on the exact deployment RTX 4090 / torch /
CUDA stack and selected checkpoint:

- compiled fixed-video versus append-only SAM3.1 masks must be bit-exact;
- rollout-cadence burst/gap replay over all 48 accepted episodes must be reviewed and a numeric drift
threshold recorded in ADR 0001;
- a co-resident replay of at least 1,000 windows across all 48 episodes must report post-capture maximum
`<300 ms`, p99 `<=250 ms`, and zero simulated action-buffer underflows for the default eight-step,
10 Hz, 0.4 s schedule;
- peak device memory must leave at least 2 GiB free; teardown must return within 256 MiB of the warmed
policy-only baseline; and three load/warm/run/unload cycles must show no monotonic growth.

The executable evidence commands and complete acceptance matrix are in the
[live rollout implementation plan](../ManiFlow_Policy/docs/wbc_maniflow_live_rollout_plan.md), with the
authoritative tracker and ordering decisions in
[ADR 0001](../ManiFlow_Policy/docs/adr/0001-track-live-masks-at-policy-observation-cadence.md) and
[ADR 0003](../ManiFlow_Policy/docs/adr/0003-install-one-ordered-live-object-condition.md).

## Legacy standalone MoF reproduction

The active MoF paths are now `action_frame_mode=mof` inside ManiFlow above and inside the LeRobot
diffusion policy (step 3). This section preserves the
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
  --raw-dir ~/Dexmate/data/raw_data \
  --out-root ~/Dexmate/data/processed_wbc/mof \
  --include-recovery-data ~/Dexmate/data/raw_data/recovery \
  --positions-dir ~/Dexmate/data/scene_diff/positions
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
`dexmate_lerobot` clone + `mofpo_root.pth` + `pip install --no-deps threadpoolctl robomimic==<install.sh pin>` + the ManiFlow env's pytorch3d;
mofpo has no `__init__.py`, so the `.pth` does what `pip install -e` cannot):

```bash
(dexmate_mof) python scripts/vis_wbc_mof_prediction.py \
  --policy-path ~/mofpo/data/outputs/wbc_v1_mof_moe_500ep/checkpoints/latest.ckpt \
  --episode ~/Dexmate/data/processed_wbc/mof/dexmate_wbc_mof/test/episode_0.safetensors \
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
  --save-dir ~/Dexmate/data/rollout/mof \
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

