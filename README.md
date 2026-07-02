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

- Head camera
  - Something worthy of trying if latency in dexmate@vega-1:~/.dexmate/sensors/default.toml
    - Under [[sensors]] id = "head_camera" 
      - [sensors.params] rate = 15
      - [sensors.streams] right_rgb = false
      - [sensors.params] depth_mode = "NEURAL_LIGHT"

```python
python scripts/robotiq_gripper_cmd_actual_omni.py # gripper latency by sending cmd in different curves
```

## Tabletop Manip

1. sshfs [yixuan@128.59.19.217](mailto:yixuan@128.59.19.217):/home/yixuan/omniteleop /home/dexmate/yixuan/omniteleop_yifan
2. ssh dexmate, conda activate dexmate
3. run '(dexmate) dextop node start' '(yixuan_yifan) python /home/dexmate/yixuan/omniteleop_yifan/tests/test_wrist_zedm_depth.py
  ' and then '(yixuan_yifan) python /home/dexmate/yixuan/omniteleop_yifan/tests/test_head_zedx_depth.py'  in tmux

> Run head before wrist may cause resolution error

> Head and wrist: 240x320. Increasing to 640x480 under wbc_vr_robot's --record-rate (10 Hz) leads to stale frame due to bandwidth-throttled. Bump only after confirming sustained fps with --verify. 

> If change resolution in the future, change HEAD_RESIZE_HW in [head_camera.py](src/omniteleop/common/head_camera.py) (head) and resize_h/resize_w in [test_wrist_zedm_depth.py](tests/test_wrist_zedm_depth.py) (wrist), then **restart both publishers**. `ZED_K` and the recorded `obs/images/intrinsic` scale automatically, and `vis_episode.py` adapts (reads the saved intrinsic + frame shape). Two things do NOT auto-follow: (1) keep the aspect ratio **4:3**, or also update `_TILE_W`/`_TILE_H` in [wbc_headset_hud.py](src/omniteleop/leader/wbc_headset_hud.py) (+ the `8/3` fallback in `web/vr_client.html`) to avoid HUD stretch; (2) the **training-dataset** resolution is set separately by `port_wbc_mobile_hdf5.py --resize-h/--resize-w` (default 240x320), independent of the recording resolution.

> Try decreasing _HUD_SCALE, HUD rate (--hud-rate, 10 Hz) and JPEG quality (60 in poll_and_send) in future.

3.1 test wrist camera via  ZED SDK (do not run dexsensor): 

```python
python /home/dexmate/yixuan/omniteleop/tests/test_wrist_zedm_depth.py --duration 10 --save-dir /home/dexmate/yixuan/Dexmate
```

3.2 check RoI:

```python
python /home/yixuan/omniteleop/tests/crop_and_resize.py
```

1. ssh lambda, conda activate dexmate
2. run
  (dexmate) python src/omniteleop/follower/vr_robot_controller.py # workspace_check = True, joint positions in vr_mode_const, head_mode and left_arm_mode in src/omniteleop/configs/vega_1_f5d6.yaml
3. (dexmate) python src/omniteleop/leader/vr_reader.py --manual_reposition /home/yixuan/Dexmate/data/raw_data/episode_0.hdf5
4. (dexmate) python src/omniteleop/leader/vr_reader.py # start-mode=fixed_pose

> debug /target/eef_used_by_ik/right: IK input
> /action/joint/right_arm: IK output + step clamping

```bash
Follow exactly:
# start_mode in main()
follow_hand mode:
1. Trigger (track head)
2. Trigger (track hand)
3. X
4. Move to new position
5. Y (track head, initialize hand, track hand)
6. A-Manip-B-X
7. Repeat 3-6

fixed_pose mode: 
1. Move to new position
2. Y (initialize hand, track head)
3. A-Trigger (track hand)-Manip-B-X
4. Reset
5. Repeat 2-3

python /home/yixuan/omniteleop/scripts/save_rgb_print_fps.py # if you forgot position last time

python /home/dexmate/yixuan/omniteleop_yifan/scripts/admittance_control.py run #calibrate first

before shutdown:
python -m omniteleop.follower.safearm_shutdown # reposition arm so that they fall onto table

visualize:
python /home/yixuan/omniteleop/scripts/vis_episode.py --hdf5 /home/yixuan/Dexmate/data/raw_data/episode_100.hdf5
```

```bash
python /home/yixuan/omniteleop/scripts/rename_raw_data.py # revise train/val/test range
```

```bash
python /home/yixuan/omniteleop/scripts/vis_teleop_curves.py --episode-id 0 # (Optional) (Need to save_debug in vr_reader)
```

## Policy

See [Lerobot README](../lerobot_yifan/README.md).
Run infer_dexmate.py before deploy.

## Deploy

```bash
python -m omniteleop.follower.policy_rollout \
      --policy-path /home/yixuan/Dexmate/model/act/dexmate_eef_eef_abs_2cam_pos_20_10/checkpoints/last/pretrained_model \
      --record_dir /home/yixuan/Dexmate/deploy/act/dexmate_eef_eef_abs_2cam_pos_20_10/last
(--arm-side right) # only for single-arm
# ACT deploy uses checkpoint n_action_steps and temporal_ensemble_coeff by default.
# use the training stats baked into the checkpoint

(dexmate_lerobot) python -m omniteleop.follower.live_scenediff_rollout \
    --policy-path /home/yixuan/Dexmate/model/dp/dexmate_right_eef_eef_abs_film/checkpoints/last/pretrained_model \
    --prompt-before-capture \
 --pos-cond-matching prompt --prompt-after-capture --arm-side right # only for single-arm
# [--record-dir]
```

```bash
python /home/yixuan/omniteleop/scripts/vis_episode.py --deploy --hdf5 /home/yixuan/Dexmate/deploy/dp/dexmate_right_eef_eef_abs_film/checkpoints/last/0/episode_0.hdf5
# rerun, transmission latency plot under debug/
```

# WBC

See [PIPELINE_WBC](./PIPELINE_WBC.md).

## WBC mobile policy pipeline

Current implemented schema:

- `observation.state`: 32-D, base-frame achieved left/right EEF + grippers, base-frame achieved `zed_depth_frame` head pose, then measured `obs/base/pose`.
- `action`: 29-D, world-frame left/right EEF targets + grippers + world-frame `head_target`, matching the targets passed to `VegaWholeBodyIK.solve(..., head_target=...)`.
- `wbik.yaml` must keep `head_mode: "ik"` for rollout. The policy must be trained with absolute actions (`use_relative_actions=false`).

**1.** Record data:

```bash
# Terminal 1, on lambda/leader machine.
(dexmate) python scripts/wbc_vr_leader.py --calibrate-ee-offset
# one-time calibration, save src/omniteleop/leader/ee_offset.yaml for future auto load
(dexmate) python scripts/wbc_vr_leader.py
# generate reference episode_0.hdf5 for future --align-reference
(dexmate) python /home/yixuan/omniteleop/scripts/wbc_vr_leader.py --align-reference /home/yixuan/Dexmate/data/raw_data/episode_0.hdf5

# Terminal 2, on robot machine.
(dexmate) python scripts/wbc_vr_robot.py \
  --record \
  --save-dir /home/yixuan/Dexmate/data/raw_data \
  --debug-dir /home/yixuan/Dexmate/data/raw_data_debug

(dexmate) python /home/yixuan/omniteleop/scripts/vis_episode.py --hdf5 /home/yixuan/Dexmate/data/raw_data/episode_0.hdf5
```

**2.** Port raw HDF5 to a LeRobot dataset:

```bash
(dexmate_lerobot) python scripts/port_wbc_mobile_hdf5.py \
  --raw-dir /home/yixuan/Dexmate/data/raw_data \
  --root /home/yixuan/Dexmate/data/processed_wbc \
  --repo-id dexmate_wbc_eef_head \
  --resize-h 240 \
  --resize-w 320
```

Visualize the ported dataset: `observation.state` (base-frame EEF/head FK composed to world via the odometry base pose, matching the calib sidecar), world-frame `action` targets.

```bash
(dexmate_lerobot) python scripts/vis_episode_processed_wbc.py \
  --dataset_dir /home/yixuan/Dexmate/data/processed_wbc/dexmate_wbc_eef_head \
  --episode_index 0
```

**3.** Train in LeRobot

The checkpoint must report `observation.state` dim `32`, `action` dim `29`, image keys `observation.images.head_rgb` and optionally `observation.images.wrist_rgb`, and `use_relative_actions=false`. Skip MIN_MAX normalization on the 6-D rotation dims (state/action dims `3-8, 13-18, 23-28`); base `x,y,yaw` (state `29-31`) stays normalized.

```bash
(dexmate_lerobot) lerobot-train \
  --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
  --policy.horizon=16 --policy.n_action_steps=8 --policy.use_relative_actions=false \
  '--policy.input_features={"observation.images.head_rgb": {"type": "VISUAL", "shape": [3, 240, 320]}, "observation.images.wrist_rgb": {"type": "VISUAL", "shape": [3, 240, 320]}, "observation.state": {"type": "STATE", "shape": [32]}}' \
  '--policy.normalization_mapping={"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"}' \
  '--policy.skip_normalization_dims={"observation.state": [3,4,5,6,7,8,13,14,15,16,17,18,23,24,25,26,27,28], "action": [3,4,5,6,7,8,13,14,15,16,17,18,23,24,25,26,27,28]}' \
  --dataset.repo_id=dexmate_wbc_eef_head \
  --dataset.root=/home/yixuan/Dexmate/data/processed_wbc/dexmate_wbc_eef_head \
  --batch_size=32 --steps=200000 --save_freq=50000 \
  --output_dir=/home/yixuan/Dexmate/model/dp/dexmate_wbc_eef_head
```

Validate the checkpoint OFFLINE before any hardware rollout — run the rollout's exact inference path (`_PolicyBundle.select_action` + `split_policy_action`) over a processed episode and overlay the policy's predicted 29-D world-frame action against the recorded ground-truth action. Rerun shows per-entity (left/right/head) GT (red) vs predicted (magenta) pose + error line, GT/pred EEF reprojected into the head image, gripper series, the base odometry path, and the depth point cloud; matplotlib + stdout report per-entity translation/rotation MAE. Run in `dexmate_lerobot`:

```bash
(dexmate_lerobot) python scripts/vis_wbc_policy_prediction.py \
  --policy-path /home/yixuan/Dexmate/model/dp/dexmate_wbc_eef_head/checkpoints/last/pretrained_model \
  --dataset_dir /home/yixuan/Dexmate/data/processed_wbc/dexmate_wbc_eef_head \
  --episode_index 0
# test data == train data for now; headless over SSH: add --save pred_ep0.rrd (open later with `rerun pred_ep0.rrd`)
```

**4.** Roll out the trained policy on the real robot. NOTE: run in `dexmate_lerobot` (needs both lerobot and the hardware SDK; the plain `dexmate` env has no lerobot):

```bash
(dexmate_lerobot) python scripts/wbc_policy_rollout.py \
  --policy-path /home/yixuan/Dexmate/model/dp/dexmate_wbc_eef_head/checkpoints/last/pretrained_model \
  --save-dir /home/yixuan/Dexmate/data/raw_data_rollout \
  --max-seconds 120
```

# Debug

### Arm joint out of limit

```bash
# on dexmate machine
python dexcontrol/examples/advanced_examples/disable_arm_motors.py disable --side right --joint-idx 6 --release-brake
# lift eef up
python dexcontrol/examples/advanced_examples/disable_arm_motors.py brake --side right --joints 6 --no-enable
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

