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
  --policy-path ~/Dexmate/model/dp/nopos_abs_minmax/checkpoints/last/pretrained_model \
  --dataset_dir ~/Dexmate/data/box2cloth/processed_wbc/test/dexmate_wbc_eef_head \
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



## ManiFlow

[ManiFlow](https://maniflow-policy.github.io/) (CoRL 2025) trained on the **same raw takes** as the
diffusion/ACT pipeline above, as a **3D point-cloud** policy. Repo: `/home/yixuan/ManiFlow_Policy`.
The zarr's `action` is **bit-identical** to the LeRobot dataset; `data/state` is the same 32-D layout
but defaulted to world frame (see table). Visual obs is a world-frame head point cloud instead of RGB.

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

Cheap position-only comparison; **only currently deployable** conditioned mode.
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
(tiled once per task slot); `env_dino` and integer mask labels use identity normalizers. DiTX applies
LayerNorm to the DINO block in-graph.
- **Online augmentation is RGB-only:** when `use_pc_color=true`, 20% of training windows receive one
temporally consistent brightness/contrast/saturation/hue transform (`[0.3, 0.4, 0.5, 0.08]`). It is
applied before normalization and disabled for validation. XYZ, state, action, masks, `env_state` and
`env_dino` are not augmented; there is no online crop, rotation, translation or point jitter.

Two crop bounds are **not** free to tighten:

- `x_min` ≤ ~0.30 — the box is carried **back toward the robot** to `x = 0.334` before being placed
(bimanual EEF midpoint while both grippers are closed). `x_min = 0.45` crops the grasped box out of
the observation on 17% of transport frames; `x_min = 0.60` on 42%.
- `z_min` ≤ ~0.62 — bounded by the **cloth** (`z ≈ 0.66–0.70`), not the box. `z_min = 0.70` halves the  
cloth's points (40 → 22) while helping the box.



### Environments

**One env for the ManiFlow stages:** `dexmate_maniflow`**.** Build/inspect the zarr, train, and run the
currently supported live modes in the same interpreter, so those stages cannot drift on a library
version. SceneDiff DINO extraction and SAM3.1 mask tracking run separately in `lerobot` and
`dexmate_lerobot`, respectively, before the zarr build.

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

Verified in this stack: enriching a zarr with masks and DINO leaves `point_cloud`, `state`, `action`
and `env_state` bit-identical to the pre-mask build; the porter also recomputes frame 0 and checks the
cloud plus mask labels through the same row lineage. The current rollout rebuilds the maskless live
cloud bit-identically to training for the modes it supports (see the rollout limitation below).

Upstream's own `requirements.txt` is unsatisfiable as written (`numpy==1.23.5` vs its own
`numba==0.61.2`, which needs numpy ≥1.24), and its sim/cloud deps (`sapien`, `mplib`, `dm_control`,
`open3d`, `azure`, `deepspeed`) are not needed here. `transformers`/`yacs`/`diffusers` are pulled in by
the clone already or unused.

**1.** Build the mask-grounded artifacts and zarr. Run these after the position extractor has written
`positions/{raw,recovery}/episode_<N>.npz`. DINO is a sibling sidecar tied to the positions NPZ by
SHA-256; masks cover the full untrimmed episode and carry the same provenance hash. A stale sidecar or
mask is rejected before the porter writes anything.

```bash
(lerobot) cd ~/scene_diff
python scripts/extract_object_dino_feats.py \
  --scenediff-root ~/Dexmate/data/box2cloth/scene_diff

(dexmate_lerobot) cd ~/scene_diff
python scripts/build_wbc_masks.py \
  --scenediff-root ~/Dexmate/data/box2cloth/scene_diff
# inspect masks/qc_summary.json and masks/viz/ before porting

(dexmate_maniflow) cd ~/omniteleop
python scripts/port_wbc_mobile_zarr.py \
  --raw-dir ~/Dexmate/data/box2cloth/raw_data \
  --out-root ~/Dexmate/data/box2cloth/processed_wbc/maniflow \
  --name dexmate_wbc --num-points 1024 \
  --include-recovery-data ~/Dexmate/data/box2cloth/raw_data/recovery \
  --positions-dir ~/Dexmate/data/box2cloth/scene_diff/positions \
  --masks-dir ~/Dexmate/data/box2cloth/scene_diff/masks \
  --dino --overwrite
# ~3 min, CPU only. -> dexmate_wbc_{train,test}.zarr (46 / 2 episodes; 17,951 / 604 frames)
# The encoder downsamples stored 1024 to policy.visual_cond_len (active baseline: 256),
# so raising --num-points only helps if you also raise visual_cond_len.
# --positions-dir -> data/env_state (T, 6) float32, constant per episode
# --dino          -> data/env_dino (T, 2560) float32, constant per episode
# --masks-dir     -> data/point_mask (T, 1024) int8, exact stored-point lineage
```

The optional arrays are additive: each policy mode loads only the keys it consumes. Thus the same
enriched zarr supports the complete experiment ladder, including the unconditioned control. The
sibling `.meta.json` records crop/depth/frame/sampler plus mask/DINO provenance; it remains the source
of truth for live preprocessing.

**2.** Look at the actual network input before training — the STORED cloud, post-crop and post-FPS,
with the EEF/base overlays (no RGB/depth/wrist panels; that format has none):

```bash
conda activate dexmate_maniflow && cd ~/omniteleop
python scripts/vis_episode_processed_wbc.py \
  --zarr ~/Dexmate/data/box2cloth/processed_wbc/maniflow/dexmate_wbc_train.zarr \
  --episode_index 0
# the same script renders the LeRobot dataset via --dataset_dir; both modes work here
```

**3.** Train. `train_maniflow_robotwin_workspace.py` is ManiFlow's only **env-runner-free** trainer
(`env_runner = None`, `RUN_ROLLOUT = False`, `RUN_VALIDATION = True` are hardcoded) — which is why our
task config lives under `config/robotwin_task/`: a Hydra group name *is* the config key. Nothing about
it is RoboTwin.

```bash
(dexmate_maniflow) bash ~/ManiFlow_Policy/scripts/train_dexmate_wbc.sh control_256 \
  point_sampling_mode=fps mask_channels=false position_condition_mode=none \
  training.num_epochs=300 training.checkpoint_every=10 training.val_every=5

(dexmate_maniflow) bash ~/ManiFlow_Policy/scripts/train_dexmate_wbc.sh stratified_256 \
  point_sampling_mode=mask_stratified mask_channels=false position_condition_mode=none \
  training.num_epochs=300 training.checkpoint_every=10 training.val_every=5

(dexmate_maniflow) bash ~/ManiFlow_Policy/scripts/train_dexmate_wbc.sh channels_256 \
  point_sampling_mode=mask_stratified mask_channels=true position_condition_mode=none \
  training.num_epochs=300 training.checkpoint_every=10 training.val_every=5

(dexmate_maniflow) bash ~/ManiFlow_Policy/scripts/train_dexmate_wbc.sh grounding_256 \
  point_sampling_mode=mask_stratified mask_channels=true \
  position_condition_mode=grounding_tokens \
  training.num_epochs=300 training.checkpoint_every=10 training.val_every=5

# Cheap position-only comparison (no mask consumer):
(dexmate_maniflow) bash ~/ManiFlow_Policy/scripts/train_dexmate_wbc.sh concat_256 \
  point_sampling_mode=fps mask_channels=false position_condition_mode=concat \
  training.num_epochs=300 training.checkpoint_every=10 training.val_every=5

# args: <addition_info> [seed] [gpu_id] [hydra overrides...]
# checkpoints → ~/Dexmate/model/maniflow/<exp>_seed<seed>/checkpoints/
# default zarr: ~/Dexmate/data/box2cloth/processed_wbc/maniflow/dexmate_wbc_train.zarr
# override it with ZARR_PATH=/other/train.zarr
```

**How many epochs?** The diffusion policy that rolled out successfully did 100k steps × batch 32 =
**3.2M samples** over 17,951 frames — 178 epochs-equivalent. ManiFlow has 16,659 train windows at
batch 64 → **261 steps/epoch**:


| `num_epochs`       | steps      | samples   | vs DP     |
| ------------------ | ---------- | --------- | --------- |
| 192                | 50,112     | 3.20M     | 1.00x     |
| **300**            | **78,300** | **5.00M** | **1.56x** |
| 1010 (cfg default) | 263,610    | 16.8M     | 5.26x     |


Use **300**, not the sample-matched 192: ManiFlow splits every batch 75% flow / 25% consistency
(`flow_batch_ratio` / `consistency_batch_ratio`), so at 300 epochs the *flow* branch alone sees 3.75M
samples ≈ 1.17x DP's whole budget. Watch `val_loss` (every 5 epochs) and `train_action_mse_error`; the
`topk` manager keeps the best-val checkpoint alongside `latest.ckpt`.

> ⚠️ `num_epochs` sets the **cosine LR horizon** (`num_training_steps = steps_per_epoch * num_epochs`,
> workspace line 150). Killing a 1010-epoch run at epoch 300 is **not** the same as training for 300 —
> the LR would still sit at ~79% of peak, never annealed. Choose it at launch.



The old 512-token timing is not the estimate for this ladder; every current cell runs at 256. Two
full-size one-epoch runs (`none+fps` and full grounding) have completed training and checkpointing.
The sampler tests cover quota/spill/shortage/over-segmentation/empty/determinism, and every ladder
cell has run `compute_loss + backward + predict_action` on the real zarr.

`training.resume: true`, so relaunching the same `<addition_info>` and seed resumes from its
`latest.ckpt`; use a new tag for a fresh run. The epoch-end workspace now explicitly returns the
student to train mode while keeping the EMA teacher in eval mode. `position_condition_dropout` is
retired and must remain `0.0`; any nonzero value fails at policy construction. Do not use
`training.debug=true` as a smoke test—the workspace rewrites it to 100 epochs × 10 steps.

Active implementation files: `point_process.py` (batched mask-stratified sampler),
`pointnet_extractor.py` (mask-aware sampling/channels), `ditx.py` (grounding tokens),
`maniflow_pointcloud_policy.py` (mode plumbing), `dexmate_wbc_dataset.py` (schema and normalizers),
the Dexmate task/policy YAMLs, the Robotwin workspace EMA-mode fix, and
`scripts/train_dexmate_wbc.sh`.

**4.** Live rollout. `scripts/wbc_maniflow_rollout.py` does **not** fork the hardware loop: it imports
`wbc_policy_rollout.py` and swaps in a ManiFlow bundle + observation worker via
`_run_rollout(..., policy_factory=, worker_factory=)`. Engage, `--align-reference`, the async chunk
scheduler, stale-action drop, the 100 Hz WBC tick, the hold watchdog, recording and shutdown are all
the same code. `--replay-episode` works identically.

```bash
(dexmate_maniflow) python scripts/wbc_maniflow_rollout.py \
  --policy-path /home/yixuan/Dexmate/model/maniflow/<exp>_seed0/checkpoints/latest.ckpt \
  --save-dir ~/Dexmate/data/box2cloth/rollout/maniflow \
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
- `--position-condition` is currently valid for the active `concat` mode: the shared rollout runs
SceneDiff once and pins `[box, cloth]` `env_state` for the episode. Both directions are checked before
engage—a concat checkpoint without the flag, or the flag with an unconditioned checkpoint, exits.

> **Live limitation:** `wbc_maniflow_rollout.py` does not yet supply `point_mask` or `env_dino`, and
> its mode gate predates `grounding_tokens`. Therefore only `none` and `concat` checkpoints are
> deployable through this script today. Do not deploy `mask_stratified`, `mask_channels=true`, or
> `grounding_tokens` checkpoints until the planned causal SAM3.1 tracking integration is complete.
> The future worker must advance masks only on policy-consumed RGB frames and expose an explicit
> `--mask-condition` contract.



## MoF (Mixture of Frames)

[MoF](https://mofpo.github.io/) trained on the **same box2cloth raw takes** as the
diffusion/ACT/ManiFlow pipelines above, as a fourth policy family. Repo: the
`wbc` branch of `[Crdr2/mofpo](https://github.com/Crdr2/mofpo)` at `/home/yixuan/mofpo`
(fork of the paper code; upstream `pointW/mofpo`). MoF denoises the 29-D world
action in **five reference frames at once** (`base`, `base_rel_trans`, `left`,
`right`, `rel_traj` — the head is a third pose entity in every frame but not a
frame itself) and fuses the predictions with a learned per-timestep router,
canonical space `base_rel_trans`. Decisions/ADR in `.scratch/mof-wbc-integration/`
and `docs/adr/0001-mof-sim-stack-for-wbc.md` (both gitignored, local).

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
(mof) cd ~/mofpo && python train.py --config-name=train_mof_moe_wbc \
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
  --save-dir ~/Dexmate/data/rollout/mof \
  --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5 \
  --policy-interval 0.6 --max-seconds 100
# 16-step DDIM x 5 experts measured ~0.6 s/chunk with the GPU shared by a
# training job (chunk covers 0.8 s): keep --policy-interval >= inference time,
# or pass --num-inference-steps 8. --replay-episode works identically.
```

The bundle refuses to load (before engage): a non-WBC or row-convention
checkpoint, a missing/mismatched port `meta.json`, or a normalizer without
fitted stats. v1 checkpoints are unconditioned — do not pass
`--position-condition`.

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

