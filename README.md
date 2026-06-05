# 🎮 Omniteleop - Teleoperation Stack for Dexmate Robots

Python

## 📦 Installation

```shell
pip install omniteleop
```

## ✨ Features

- 🕹️ **JoyCon Controller Support** - Use Nintendo JoyCon for robot control
- 📊 **Telemetry Viewer** - Real-time visualization of joint data

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

## Data collection

1. sshfs [yixuan@128.59.19.217](mailto:yixuan@128.59.19.217):/home/yixuan/omniteleop /home/dexmate/yixuan/omniteleop_yifan
2. ssh dexmate, conda activate dexmate
3. run '(dexmate) dextop node start' '(yixuan_yifan) python /home/dexmate/yixuan/omniteleop_yifan/tests/test_wrist_zedm_depth.py
  ' and then '(yixuan_yifan) python /home/dexmate/yixuan/omniteleop_yifan/tests/test_head_zedx_depth.py'  in tmux

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

python /home/dexmate/yixuan/omniteleop_yifan/scripts/admittance_control.py run

before shutdown:
python -m omniteleop.follower.safearm_shutdown # reposition arm so that they fall onto table

visualize:
python /home/yixuan/omniteleop/scripts/vis_episode.py --hdf5 /home/yixuan/Dexmate/data/raw_data/episode_100.hdf5
```

- scripts/vis_episode.py (Optional)

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
python scripts/vis_deploy.py \
        --gt /home/yixuan/Dexmate/data/raw_data_renamed/test/episode_0.hdf5 \
        --infer /home/yixuan/Dexmate/deploy/dp/dexmate_eef_eef/last/2/episode_0.hdf5
```

```bash
python /home/yixuan/omniteleop/scripts/vis_episode.py --deploy --hdf5 /home/yixuan/Dexmate/deploy/dp/dexmate_right_eef_eef_abs_film/checkpoints/last/0/episode_0.hdf5
# rerun, transmission latency plot under debug/
```

## Misc

```bash
python /home/yixuan/omniteleop/src/omniteleop/leader/check_eef_pos.py # robot.right_arm.get_joint_pos() first, and the run this to calculate eef from joint 
```

