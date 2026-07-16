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
# Terminal 1, on lambda/leader machine.
(dexmate) python scripts/wbc_vr_leader.py --calibrate-ee-offset
# one-time calibration, save src/omniteleop/leader/ee_offset.yaml for future auto load
(dexmate) python scripts/wbc_vr_leader.py
# generate reference episode_0.hdf5 for future --align-reference
(dexmate) python /home/yixuan/omniteleop/scripts/wbc_vr_leader.py --align-reference /home/yixuan/Dexmate/data/raw_data_reference/reference.hdf5

# Terminal 2, on robot machine.
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
  --batch_size=32 --policy.optimizer_lr=2e-4 --steps=200000 --save_freq=50000 --save_best=true --log_freq=2000 \
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
- **Do not set** `--policy.temporal_ensemble_coeff`: ACT then requires `n_action_steps=1`, and
`wbc_policy_rollout.py` schedules whole chunks — a 1-step chunk covers 0.1 s and the buffer runs dry.

Deploy is unchanged: `scripts/wbc_policy_rollout.py` is policy-type agnostic (it reads `n_action_steps`
off the checkpoint and feeds ACT the batch directly instead of the diffusion obs queues). Point
`--policy-path` at the ACT checkpoint and add `--position-condition` iff it was trained with the ENV
feature — the rollout cross-checks the two and aborts on a mismatch.

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
  --policy-path ~/Dexmate/model/dp/dexmate_wbc_eef_head/checkpoints/last/pretrained_model \
  --dataset_dir ~/Dexmate/data/processed_wbc/test/dexmate_wbc_eef_head \
  --episode_index 0
```

Roll out the trained policy on the real robot. Inference is asynchronous (rby1-wbc
style scheduled chunks, see `plan.md`): a worker thread replans a whole action chunk
every `--policy-interval` (default 0.4 s), stale frames older than
`inference_end + --execution-latency` are dropped, and a 10 Hz sampler feeds the
100 Hz WBC tick — a slow inference never freezes the control loop (the robot holds
via the `--source-timeout` watchdog instead). `--dataset-fps` (default 10) must match
the porter fps.

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

# Position-conditioned checkpoint (trained with observation.environment_state, step 3):
# add --position-condition. After engage and --align-reference, the rollout captures ONE head
# frame with the arms already at the reference pose, stamps it with the FK world_T_zed
# extrinsic (engage-origin world), diffs it against the SAME reference the training positions
# used (default reference_last.hdf5), and pins the [box, cloth] world positions as a constant
# env-state for the whole episode. SceneDiff runs in scene_diff/.venv-merged (its SAM3/DINOv3
# VRAM is a subprocess) and takes ~minutes; a missing npz (detection gate failed) aborts before
# recording/policy motion. Live scenes have no demo trajectory, so the operator types the order
# from the size-slot overlay (--prompt-after-capture).
(dexmate_lerobot) python scripts/wbc_policy_rollout.py \
  --policy-path /home/yixuan/Dexmate/model/dp/concat_abs_minmax/checkpoints/last/pretrained_model \
  --save-dir ~/Dexmate/data/rollout/concat_abs_minmax \
  --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5 \
  --position-condition --prompt-after-capture --max-seconds 100
#   --reference-hdf5 ~/Dexmate/data/scene_diff/_before/reference_last.hdf5   (default)
#   --prompt-before-capture   pause to stage the scene after alignment, before the head grab
#   SceneDiff outputs -> <save-dir>/scene_diff/{live_capture.hdf5,episode_0.npz,
#                        deploy_slot_overlay.png (VERIFY: type s1_src=BOX, s1_dst=CLOTH)}

python scripts/vis_episode.py --hdf5 /home/yixuan/Dexmate/data/rollout/concat_abs_minmax/episode_0.hdf5
```



## Maniflow

[ManiFlow](https://maniflow-policy.github.io/) (CoRL 2025) trained on the **same raw takes** as the
diffusion/ACT pipeline above, as a **3D point-cloud** policy. Repo: `/home/yixuan/ManiFlow_Policy`.
The zarr's `action` is **bit-identical** to the LeRobot dataset; `data/state` is the same 32-D layout
but defaulted to world frame (see table). Visual obs is a world-frame head point cloud instead of RGB.

```
data/point_cloud  (T, 1024, 6) float32   engage-origin WORLD XYZ + RGB[0,1]
data/state        (T, 32)      float32   agent_pos  (WORLD frame by default)
data/action       (T, 29)      float32   verbatim ik.solve targets (WORLD)
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


Other deliberate deviations from the LeRobot pipeline:

- **Workspace crop, measured not guessed:** `x∈[0.20,1.70] y∈[-0.90,1.05] z∈[0.60,1.32]` (world,
`src/omniteleop/wbc_pointcloud.py`). FPS spreads a fixed budget ~uniformly in space, so every cubic
metre of room the crop keeps is budget stolen from the two small objects. **Task-specific** — a new
object layout needs new `--crop-min/--crop-max`.
- **Normalization is ManiFlow's** `mode='limits'` (per-dim min–max) on *all* dims incl. the 6-D
rotations (same scale family as DP/ACT `MIN_MAX`, but ManiFlow also normalizes the rotation dims).



### Point-cloud density: what the network actually sees

Measured with the real SceneDiff per-object masks, so "points on the box" is exact. Two knobs
dominate, and they compound:


|                                                         | box pts | cloth pts |
| ------------------------------------------------------- | ------- | --------- |
| old crop, `visual_cond_len=128`                         | 5       | 4         |
| **new crop**, stored 1024                               | 83      | 74        |
| **new crop**, `visual_cond_len=512` (what the net sees) | ~40     | ~30       |


`DP3Encoder` farthest-point-samples the stored 1024 down to `visual_cond_len` before the transformer
sees anything — so *that* number, not the zarr's `num_points`, is the real budget. RoboTwin's 128 is
tuned for a tabletop crop; here it left ~5 points on the box. 512 costs ~3x the step time and forces
`batch_size 64` (10.6 GB; 128 needs 19.8 GB). 1024 (`downsample_points: false`) would give the full
83/74 but OOMs above batch 64.

Two crop bounds are **not** free to tighten:

- `x_min` ≤ ~0.30 — the box is carried **back toward the robot** to `x = 0.334` before being placed
(bimanual EEF midpoint while both grippers are closed). `x_min = 0.45` crops the grasped box out of
the observation on 17% of transport frames; `x_min = 0.60` on 42%.
- `z_min` ≤ ~0.62 — bounded by the **cloth** (`z ≈ 0.66–0.70`), not the box. `z_min = 0.70` halves the
cloth's points (40 → 22) while helping the box.



### Ported from the DemoGen-style `pcd_tmp.py`, and what was rejected

- **Tighter workspace crop** — adopted. This is the real lesson of those pipelines (their workspace is
0.65 × 0.90 × 0.44 m); ours was the whole room. Box 48 → 83 points, cloth 40 → 74, zero latency cost.
- `fpsample` **bucket-FPS** — adopted (`h=5`, `start_idx=0`). Same object coverage as an exact CUDA FPS
(box 86 vs 83) and the same 0.044 m mean nearest-neighbour spread, but **2.0 ms/frame instead of
16.2**: the live rollout samples one frame at a time, where a CUDA FPS is kernel-launch bound and
cannot amortise over a batch. Live preprocessing 97 ms → **19 ms** for both obs steps; conversion
runs in 2m55s and no longer needs a GPU.
  > ⚠️ `start_idx` **must** stay pinned. fpsample's default seeds from a *random* point, which would
  > make the training data irreproducible and silently skew the live cloud away from the trained one.
  > `meta.json` records `sampler`/`kdline_h`/`start_idx`/`fpsample_version`; `wbc_maniflow_rollout.py`
  > refuses to load on a mismatch and warns on a version drift.
- **DBSCAN / radius-outlier removal (**`pcd_cluster`**)** — *rejected, measured*. It deletes 3–4% of points
overall but ~~10% of the **box's** — whose far, thin surface is genuinely sparse — and so *lowers* box
coverage (37 → 30 at 512 tokens) while costing ~6 ms/frame. FPS does over-select sparse regions (~~15%
of its picks have <6 neighbours within 3 cm, ~4x their share of the cloud), but on real ZED depth
those picks are mostly true object edges, not sensor noise. Their sim depth is clean; ours is not.



### Environments

**One env for the whole pipeline:** `dexmate_maniflow`**.** Build the zarr, visualize it, train, and roll
out on hardware — all in the same interpreter, so train/deploy can never drift on a library version.

It is a clone of `dexmate_lerobot` (which keeps the *editable* dexcontrol/omniteleop installs pointing
at the same sources, plus pinocchio/pink, lerobot, rerun, `fpsample`), with a small delta:

```bash
conda create -n dexmate_maniflow --clone dexmate_lerobot -y
P=~/miniforge3/envs/dexmate_maniflow/bin/python

# 1. the rollout's policy import chain needs only timm on top of the clone.
#    --no-deps: do NOT let it bump the pinned numpy 2.2.6 (lerobot caps <2.3.0).
$P -m pip install --no-deps timm

# 2. TRAINING additionally needs numba (maniflow.common.sampler) and **zarr 2**.
#    ManiFlow's ReplayBuffer is bound to the zarr-v2 API in several places -- it calls
#    zarr.open(path, mode) POSITIONALLY and zarr.group(read_only_store), both of which
#    zarr 3 rejects -- so zarr 3 would need upstream ManiFlow edits. Nothing in the env
#    depends on zarr 3 (it has no reverse dependencies), so downgrading is free.
#    port_wbc_mobile_zarr.py itself is version-agnostic and always writes format-2 stores.
$P -m pip install "zarr<3" numba

# 3. `pip install -e ManiFlow` installs NOTHING importable (maniflow/ has no __init__.py;
#    upstream only works because its trainers chdir into ManiFlow/). A .pth fixes that.
echo /home/yixuan/ManiFlow_Policy/ManiFlow > ~/miniforge3/envs/dexmate_maniflow/lib/python3.12/site-packages/maniflow_root.pth

# 4. pytorch3d is a hard import in pointnet_extractor and must be built from source.
#    Build from `main`, NOT @stable: release 0.7.9 only claims torch <=2.4, but main
#    compiles clean against torch 2.7.1+cu126 / py3.12. Its CUDA sample_farthest_points
#    is what DP3Encoder calls on every forward.
export TORCH_CUDA_ARCH_LIST=8.9 FORCE_CUDA=1 MAX_JOBS=$(nproc) CUDA_HOME=/usr   # RTX 4090 = sm_89
$P -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@main"
```

Verified in this one env: the converter writes clouds **bit-identical** to the reference zarr; both
`vis_episode_processed_wbc.py` modes render; training reaches the **same** `val_loss 2.519692897796631`
as a run under upstream's pinned torch 2.4.1 / numpy 1.26.4; and the rollout loads the resulting
`latest.ckpt` and rebuilds a live cloud bit-identical to the training one.

Upstream's own `requirements.txt` is unsatisfiable as written (`numpy==1.23.5` vs its own
`numba==0.61.2`, which needs numpy ≥1.24), and its sim/cloud deps (`sapien`, `mplib`, `dm_control`,
`open3d`, `azure`, `deepspeed`) are not needed here. `transformers`/`yacs`/`diffusers` are pulled in by
the clone already or unused.

**1.** Build the zarr. Splits reuse `raw_data/split.csv`; each split gets its own zarr. The sibling `<name>_<split>.meta.json`
records the crop/depth/frame and is the **single source of truth** the live rollout re-reads.

```bash
conda activate dexmate_maniflow && cd ~/omniteleop
python scripts/port_wbc_mobile_zarr.py \
  --raw-dir ~/Dexmate/data/raw_data \
  --out-root ~/Dexmate/data/processed_wbc/maniflow \
  --name dexmate_wbc --num-points 1024 \
  --include-recovery-data ~/Dexmate/data/raw_data/recovery --overwrite \
  --positions-dir ~/Dexmate/data/scene_diff/positions
# ~3 min, CPU only. -> dexmate_wbc_{train,test}.zarr (46 / 2 episodes; 17,951 / 604 frames)
# The encoder farthest-point-samples the stored 1024 down to policy.visual_cond_len (512),
# so raising --num-points only helps if you also raise visual_cond_len.
# --positions-dir adds data/env_state (T, 6): the SceneDiff [box, cloth] world positions
# (positions/<raw|recovery>/episode_<N>.npz, same loading + (source, raw_index) keying as
# the LeRobot porter), validated up front and broadcast per frame. Purely additive -- a
# position_condition_mode=none run on this zarr trains bit-identically to one without it.
```

**2.** Look at the actual network input before training — the STORED cloud, post-crop and post-FPS,
with the EEF/base overlays (no RGB/depth/wrist panels; that format has none):

```bash
conda activate dexmate_maniflow && cd ~/omniteleop
python scripts/vis_episode_processed_wbc.py \
  --zarr ~/Dexmate/data/processed_wbc/maniflow/dexmate_wbc_train.zarr --episode_index 0
# the same script renders the LeRobot dataset via --dataset_dir; both modes work here
```

**3.** Train. `train_maniflow_robotwin_workspace.py` is ManiFlow's only **env-runner-free** trainer
(`env_runner = None`, `RUN_ROLLOUT = False`, `RUN_VALIDATION = True` are hardcoded) — which is why our
task config lives under `config/robotwin_task/`: a Hydra group name *is* the config key. Nothing about
it is RoboTwin.

```bash
(dexmate_maniflow) bash ~/ManiFlow_Policy/scripts/train_dexmate_wbc.sh pc1024 0 0 \
  training.num_epochs=200 training.checkpoint_every=10 training.val_every=5
# args: <addition_info> <seed> <gpu_id> [hydra overrides...]
# checkpoints → ~/Dexmate/model/maniflow/<exp>_seed<seed>/checkpoints/

# position-conditioned variants (zarr must have data/env_state; one switch drives
# dataset + policy — see "Position conditioning" below):
(dexmate_maniflow) bash ~/ManiFlow_Policy/scripts/train_dexmate_wbc.sh concat 0 0 \
  position_condition_mode=concat training.num_epochs=200 training.checkpoint_every=10 training.val_every=5
(dexmate_maniflow) bash ~/ManiFlow_Policy/scripts/train_dexmate_wbc.sh sqrt 0 0 \
  position_condition_mode=scene_query_tokens training.num_epochs=200 training.checkpoint_every=10 training.val_every=5
```

**How many epochs?** The diffusion policy that rolled out successfully did 100k steps × batch 32 =
**3.2M samples** over 17,951 frames — 178 epochs-equivalent. ManiFlow has 16,659 train windows at
batch 64 → **261 steps/epoch**:


| `num_epochs`       | steps      | samples   | vs DP     | wall time |
| ------------------ | ---------- | --------- | --------- | --------- |
| 192                | 50,112     | 3.20M     | 1.00x     | 4.7 h     |
| **300**            | **78,300** | **5.00M** | **1.56x** | **7.3 h** |
| 1010 (cfg default) | 263,610    | 16.8M     | 5.26x     | 24.7 h    |


Use **300**, not the sample-matched 192: ManiFlow splits every batch 75% flow / 25% consistency
(`flow_batch_ratio` / `consistency_batch_ratio`), so at 300 epochs the *flow* branch alone sees 3.75M
samples ≈ 1.17x DP's whole budget. Watch `val_loss` (every 5 epochs) and `train_action_mse_error`; the
`topk` manager keeps the best-val checkpoint alongside `latest.ckpt`.

> ⚠️ `num_epochs` sets the **cosine LR horizon** (`num_training_steps = steps_per_epoch * num_epochs`,
> workspace line 150). Killing a 1010-epoch run at epoch 300 is **not** the same as training for 300 —
> the LR would still sit at ~79% of peak, never annealed. Choose it at launch.

Measured on the RTX 4090 (`batch_size: 64`, `visual_cond_len: 512`): 337 ms/step (2.97 it/s), ~88 s/epoch,
~12.6 GB peak. Each checkpoint is **2.7 GB** (model + EMA + optimizer); `checkpoint_every: 50` would put
the first `latest.ckpt` 73 min in, hence `=20` above. `training.resume: True`, so relaunching with the
same `<addition_info>` and `<seed>` resumes from that run dir's `latest.ckpt` — use a new tag for a fresh
run. `ZARR_PATH=...` overrides the dataset. Do **not** use `training.debug=True` as a smoke test: the
workspace rewrites it to 100 epochs × 10 steps, which is slower than the real short run above.

Files added: `maniflow/dataset/dexmate_wbc_dataset.py` (subclasses `RoboTwinDataset`, pins the 32/29
schema, handles `env_state`), `maniflow/config/robotwin_task/dexmate_wbc_pointcloud.yaml`,
`maniflow/config/maniflow_pointcloud_policy_dexmate.yaml`, `scripts/train_dexmate_wbc.sh`.
Upstream ManiFlow edits (all additive and gated behind `position_condition_mode`; with the default
`none`, model behavior and state-dict keys are unchanged — verified by a `strict=True` load of the
pre-edit `pc1024` checkpoint): `robotwin_dataset.py` (`extra_buffer_keys` arg),
`pointnet_extractor.py` (DP3Encoder scene-query branch), `ditx.py` (SQRT context tokens),
`maniflow_pointcloud_policy.py` (mode plumbing + env-state normalization).

**4.** Live rollout. `scripts/wbc_maniflow_rollout.py` does **not** fork the hardware loop: it imports
`wbc_policy_rollout.py` and swaps in a ManiFlow bundle + observation worker via
`_run_rollout(..., policy_factory=, worker_factory=)`. Engage, `--align-reference`, the async chunk
scheduler, stale-action drop, the 100 Hz WBC tick, the hold watchdog, recording and shutdown are all
the same code. `--replay-episode` works identically.

```bash
(dexmate_maniflow) python scripts/wbc_maniflow_rollout.py \
  --policy-path /home/yixuan/Dexmate/model/maniflow/dexmate_wbc-maniflow_pointcloud_policy_dexmate-pc1024_seed0/checkpoints/latest.ckpt \
  --save-dir ~/Dexmate/data/rollout/maniflow \
  --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5 \
  --num-inference-steps 2 --policy-interval 0.3 --max-seconds 85
```

- `--policy-path` is a ManiFlow `.ckpt` **file** (`torch.load` + `dill`), not a LeRobot dir. EMA
weights by default (`--no-ema` for raw).
- `--num-inference-steps 2`: ManiFlow's consistency-flow objective makes 1–2 flow steps enough.
Measured ~139 ms/chunk at 2 steps (vs ~0.4 s for the 100-step DDPM baseline), so `--policy-interval`
can drop to ~0.3 s.
- The live cloud is rebuilt by `omniteleop.wbc_pointcloud` — **the same module the converter used** —
with the crop/depth/frame read from the training `.meta.json` (`--pointcloud-meta` if the zarr moved).
Verified **bit-identical** to the stored training cloud on raw `episode_1` frame 0.
- The `n_obs_steps` head frames are grabbed at the dataset period **first**, then both clouds are built
in one batched farthest-point call (~97 ms for 2 frames). Building inline would stretch the
observation spacing past the 0.1 s the policy was trained on.
- `--position-condition` works exactly like the LeRobot rollout (same shared code path): a checkpoint
trained with `position_condition_mode` ≠ `none` declares `env_state`, the rollout captures one head
frame at the start pose, runs SceneDiff against the fixed reference, and pins the arranged
`[box, cloth]` world vector for the whole episode. Both directions are enforced up front — a
conditioned checkpoint without the flag and vice versa exit before engage.



### Position conditioning (implemented)

**Current tensor pipeline.** Dimensions: batch `B=64`, observation steps `To=2`, stored points
`N=1024`, visual/FPS points per observation `P=512`, object slots `K=2`, input channels `C=6`
(XYZRGB), point features `Fp=128`, state features `Fs=64`, visual features `Do=Fp+Fs=192`,
SQRT anchor width `A=135`, and DiTX width `D=768`.

```text
none / concat:
  point_cloud (B*To=128, N=1024, C=6) XYZRGB
    --FPS on XYZ--> sampled_cloud (B*To=128, P=512, C=6)
    --PointNetEncoderXYZRGB, pointwise--> point_feat (B*To=128, P=512, Fp=128)

scene_query_tokens (SQRT):
  point_cloud (B*To=128, N=1024, C=6) XYZRGB
    --PointNetEncoderXYZRGB on ALL points-->
       full_feat (B*To=128, N=1024, Fp=128)
      |--3-NN queries interpolate local_feat from the full pre-FPS field
      `--FPS on XYZ + gather features-->
         point_feat (B*To=128, P=512, Fp=128)

agent_pos:
  none / SQRT: agent_pos (B*To=128, state_dim=32)
  concat: cat(agent_pos (B*To=128, state_dim=32),
              env_state (B*To=128, 3*K=6))
          -> conditioned_state (B*To=128, state_dim+3*K=38)
  --state_mlp--> state_feat (B*To=128, Fs=64)

cat(point_feat, state_feat repeated over P=512 points)
  -> visual (B*To=128, P=512, Do=192)
  --reshape observations--> vis_cond (B=64, To*P=1024, Do=192)
  --Linear(Do=192,D=768) + visual position embedding-->
     visual_context (B=64, To*P=1024, D=768)

SQRT only:
  anchor_raw = cat feature axis [local_feat(Fp=128), query_xyz(xyz=3),
                                 pair_delta(xyz=3), nearest_distance_m(1)]
             -> (B*To=128, K=2, A=135)
             -> reshape (B=64, To=2, K=2, A=135)
  --Linear(A=135,D=768) + role/obs-step embeddings + LayerNorm-->
     position_tokens (B=64, To*K=4, D=768)
  cat([visual_context, position_tokens], dim=1)
    -> context_c (B=64, To*P+To*K=1028, D=768)
```

`query_xyz` and `pair_delta = dst - src` use the same normalized XYZ coordinates as the cloud;
`nearest_distance_m` is in metres. The 3-NN local feature is blanked when the nearest observed point
is farther than `position_condition_max_query_dist` (currently 0.1 m), while the other anchor fields
remain available.

Motivation: the unconditioned `pc1024` rollout grasped the box but placed it **left of the cloth** —
the policy needed to be told *where* to place. The condition is the per-episode-constant 6-D
`env_state` = SceneDiff `[box_xyz, cloth_xyz]` in the engage-origin world frame (the same frame as
the cloud, state and action). One config switch, `position_condition_mode`, drives dataset + policy:

- `none` (default) — baseline; checkpoints byte-compatible with pre-feature code.
- `concat` — `env_state` appended to `agent_pos` (32→38) before the encoder. The trivial
baseline: 6 numbers pushed through the shared 64-D state MLP and smeared over every visual token.
- `scene_query_tokens` (SQRT, designed by Codex from the `o2oafford/` reference —
`~/ManiFlow_Policy/docs/sqrt_position_conditioning.md` is the full design doc) — each object
position becomes a **role token**: the DP3 per-point feature field (all 1024 pre-FPS points, so the
condition does not inherit FPS's small-object coverage loss) is interpolated at the box/cloth
positions O2O-Afford-style (3-NN inverse-distance), concatenated with the query xyz, the signed
src→dst displacement and the nearest-point distance, projected to 768-D, tagged
with learned source/target role + obs-step embeddings, then appended to DiTX's cross-attention
context (4 extra tokens on top of 1024). Every action token can attend
directly to "box here", "cloth here", "move this way". ~109k extra params, <0.3 GB extra train memory.

Normalization (both modes): `env_state` reuses the **point-cloud XYZ affine** (tiled per slot), not
independently fitted stats, so queries and cloud share one normalized coordinate system. Ablate
`none` vs `concat` vs `scene_query_tokens` on the signed **lateral placement error** of the
placing-hand actions in the post-grasp segment (val_loss barely separates these; see the design doc's
ablation section for the counterfactual target-shift probe).

Before trusting a conditioned run, sanity-check the SceneDiff positions: the `obj_num` forced
bijection can promote a runner-up region when the real object fails the visibility gate. All 48
current npz files (41 raw + 7 recovery) pass `valid == 1`; median per-episode worst `matched_dist`
is 0.098 m. The one outlier, `raw/episode_61` at **0.382 m**, is already the held-out *test*
episode (split.csv), so it never trains — the training set's worst match is 0.168 m.

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

