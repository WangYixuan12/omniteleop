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

## Robot preparation
1. sshfs [yixuan@128.59.19.217](mailto:yixuan@128.59.19.217):/home/yixuan/omniteleop /home/dexmate/yixuan/omniteleop_yifan
2. run '(dexmate) dextop node start' '(yixuan_yifan) python /home/dexmate/yixuan/omniteleop_yifan/tests/test_wrist_zedm_depth.py
  ' and then '(yixuan_yifan) python /home/dexmate/yixuan/omniteleop_yifan/tests/test_head_zedx_depth.py'  in tmux
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

The checkpoint must report `observation.state` dim `32`, `action` dim `29`, image keys `observation.images.head_rgb` and optionally `observation.images.wrist_rgb`; with position conditioning it also carries `observation.environment_state` dim `6` (the SceneDiff `[box, cloth]` world positions from steps 1.5/2). Continuous state/action dims are normalized with 1–99 percentile stats (`QUANTILES` mode; the porter already writes `q01`/`q99` into `meta/stats.json`), while the constant env-state uses `MIN_MAX`.

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

```bash
# change --job_name and --output_dir
# Position condition:
# update --policy.input_features and --policy.normalization_mapping and --policy.position_condition_mode=concat/film
(dexmate_lerobot) lerobot-train \
  --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
  --policy.horizon=16 --policy.n_action_steps=8 --policy.use_relative_actions=false \
  '--policy.input_features={"observation.images.head_rgb": {"type": "VISUAL", "shape": [3, 120, 160]}, "observation.images.wrist_rgb": {"type": "VISUAL", "shape": [3, 120, 160]}, "observation.state": {"type": "STATE", "shape": [32]}, "observation.environment_state": {"type": "ENV", "shape": [6]}}' \
  '--policy.normalization_mapping={"VISUAL": "MEAN_STD", "STATE": "QUANTILES", "ACTION": "QUANTILES", "ENV": "MIN_MAX"}' \
  '--policy.skip_normalization_dims={"observation.state": [3,4,5,6,7,8,13,14,15,16,17,18,23,24,25,26,27,28], "action": [3,4,5,6,7,8,13,14,15,16,17,18,23,24,25,26,27,28]}' \
  --policy.position_condition_mode=concat \
  --dataset.repo_id=dexmate_wbc_eef_head \
  --dataset.root=/home/yixuan/Dexmate/data/processed_wbc/train/dexmate_wbc_eef_head \
  --dataset.image_transforms.enable=true \
  --batch_size=32 --steps=100000 --save_freq=50000 \
  --output_dir=/home/yixuan/Dexmate/model/dp/dexmate_wbc_eef_head \
  --wandb.enable=true \
  --wandb.project=dexmate_wbc_mobile \
  --job_name=concat_abs \
  --dataset.val_root=/home/yixuan/Dexmate/data/processed_wbc/test/dexmate_wbc_eef_head --val_freq=5000
# The 6-D rotation dims (state/action 3-8, 13-18, 23-28) skip normalization; grippers and base x,y,yaw (state 29-31) stay quantile-normalized.

# Position conditioning (needs a dataset ported with step 2 --positions-dir): the ENV input
# feature "observation.environment_state" [6] and the "ENV": "MIN_MAX" mapping above enable it.
# ENV is MIN_MAX, NOT quantiles -- the env-state is CONSTANT per episode, so q01/q99 are
# meaningless; the dataset min/max is the correct scale. Add --policy.position_condition_mode=film
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
  '--policy.normalization_mapping={"VISUAL": "MEAN_STD", "STATE": "QUANTILES", "ACTION": "QUANTILES", "ENV": "MIN_MAX"}' \
  '--policy.skip_normalization_dims={"observation.state": [3,4,5,6,7,8,13,14,15,16,17,18,23,24,25,26,27,28], "action": [3,4,5,6,7,8,13,14,15,16,17,18,23,24,25,26,27,28]}' \
  --dataset.repo_id=dexmate_wbc_eef_head \
  --dataset.root=/home/yixuan/Dexmate/data/processed_wbc/train/dexmate_wbc_eef_head \
  --dataset.image_transforms.enable=true \
  --batch_size=32 --policy.optimizer_lr=2e-4 --steps=200000 --save_freq=50000 --log_freq=2000 \
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
- **No `--policy.position_condition_mode`.** `concat`/`film` is diffusion-only. ACT ingests the
  env-state as one extra transformer-encoder token (`encoder_env_state_input_proj`, +4096 params),
  so conditioning is always concat-like.
- **Leave `--policy.stage_prediction_enabled` at its default `false`.** The tabletop script sets it
  `true`; it requires a per-frame `observation.pos_condition_mask`, which this single-stage porter
  does not emit — ACT raises on the first backward pass. Same reason `stage_prediction_*` is absent
  above. The env shape `[6]` equals the processor's `condition_dim`, so `PositionConditionProcessorStep`
  passes the vector through unmasked; the N-cycle caveat from the diffusion block applies identically.
- Tabletop normalizes STATE/ACTION with `MEAN_STD`; WBC uses `QUANTILES` (+ rotation-dim skips) to
  stay consistent with the diffusion run on this dataset. ENV stays `MIN_MAX` in both.
- `--policy.optimizer_lr=2e-4` overrides ACT's `1e-5` default (matches the tabletop script).
- **Do not set `--policy.temporal_ensemble_coeff`**: ACT then requires `n_action_steps=1`, and
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
    --repo-id dexmate_wbc_eef_head \
    --include_recovery_data ~/Dexmate/data/raw_data/recovery \
    --log_index ~/Dexmate/data/raw_data/log_index.csv
```

Then recompute action stats over the relative distribution and train with the relative flags. Translations AND 6-D rotations are made relative; only the binary grippers stay absolute (`relative_exclude_dims` `[9, 19]` — the raw-FC03 state gripper is on a different scale than the 0/1 command). The column-wise rotation delta is not itself a rotation, but the post-processor adds the same chunk-anchor state back before the Gram-Schmidt decode, so it is exactly invertible; rotation dims still skip normalization (relative deltas concentrate near 0, where a q01/q99 rescale would amplify noise). `chunk_size` = horizon; `reference_offset` = `n_obs_steps − 1` = 1 for diffusion. For **ACT** the anchor is the chunk start (`action_delta_indices = range(0, chunk_size)`), so use `--operation.reference_offset=0` and `--operation.chunk_size=<--policy.chunk_size>`.

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
  '--policy.normalization_mapping={"VISUAL": "MEAN_STD", "STATE": "QUANTILES", "ACTION": "QUANTILES", "ENV": "MIN_MAX"}' \
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
  --policy-path ~/Dexmate/model/dp/nopos_abs/checkpoints/last/pretrained_model \
  --save-dir ~/Dexmate/data/rollout/nopos_abs \
  --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5 \
  --max-seconds 85
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
  --policy-path /home/yixuan/Dexmate/model/robo02/concat_abs_minmax/checkpoints/last/pretrained_model \
  --save-dir ~/Dexmate/data/rollout/concat_abs_minmax \
  --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5 \
  --position-condition --prompt-after-capture --max-seconds 85
#   --reference-hdf5 ~/Dexmate/data/scene_diff/_before/reference_last.hdf5   (default)
#   --prompt-before-capture   pause to stage the scene after alignment, before the head grab
#   SceneDiff outputs -> <save-dir>/scene_diff/{live_capture.hdf5,episode_0.npz,
#                        deploy_slot_overlay.png (VERIFY: type s1_src=BOX, s1_dst=CLOTH)}

python scripts/vis_episode.py --hdf5 /home/yixuan/Dexmate/data/rollout/concat_abs_minmax/episode_0.hdf5
```



# Debug



### Arm joint out of limit

```bash
# on dexmate machine
python dexcontrol/examples/advanced_examples/disable_arm_motors.py disable --side right --joint-idx * --release-brake
# lift eef up
python dexcontrol/examples/advanced_examples/disable_arm_motors.py brake --side right --joints 5 --no-enable
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
