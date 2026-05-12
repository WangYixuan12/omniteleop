# 🎮 Omniteleop - Teleoperation Stack for Dexmate Robots

Python

## 📦 Installation

```shell
pip install omniteleop
```

## ✨ Features

- 🕹️ **JoyCon Controller Support** - Use Nintendo JoyCon for robot control
- 💪 **Exoskeleton Arm Control** - Intuitive arm teleoperation via Dynamixel exoskeleton
- 🛡️ **Safety System** - Built-in emergency stop and joint limits enforcement
- 📹 **Data Collection** - Record teleoperation data for policy learning
- 🔄 **Trajectory Replay** - Replay recorded robot trajectories
- 📊 **Telemetry Viewer** - Real-time visualization of joint data

## Official docs

- [Dexmate](https://docs.dexmate.ai/dSBwCBpol8PGkSXTS9bJ)
- [ZED camera](https://www.stereolabs.com/docs) 
  - [Depth sensing](https://www.stereolabs.com/docs/depth-sensing)

## Data collection
0. check network on lambda, dexmate
1. ssh dexmate, conda activate dexmate
2. run '(dexmate) dextop node start' and 'dexsensor launch --config ~/.dexmate/sensors/default.toml --sensor head_camera' in tmux
3. ssh lambda
4. run
  (dexmate) python src/omniteleop/follower/vr_robot_controller.py # workspace_check = True, joint positions in vr_mode_const, head_mode and left_arm_mode in src/omniteleop/configs/vega_1_f5d6.yaml  
    (dexmate) python src/omniteleop/leader/vr_reader.py # start-mode=fixed_pose

> /action/eef/right: IK input
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
4. Repeat 2-3
# if you forgot position last time, run scripts/save_first_rgb_print_fps.py
```

- scripts/vis_episode.py (Optional)

```bash
1. scripts/rename_raw_data.py  # revise train/val/test range
```

- scripts/vis_teleop_curves.py (Optional) (Need to save_debug in vr_reader)

## Policy

See [Lerobot README](../lerobot_yifan/README.md).
Run infer_dexmate.py before deploy.

## Deploy

```bash
python -m omniteleop.follower.policy_rollout --policy-path /home/yixuan/omniteleop/Dexmate/model/act/act_abs_joint_eef/checkpoints/400000/pretrained_model
# act_n_action_steps=1, act_temporal_ensemble_coeff=0.01 for ACT eval
```

```bash
python /home/yixuan/Dexmate/deploy/dump_subset.py --input_path /home/yixuan/Dexmate/deploy/act_abs_eef_eef/episode_0.hdf5
# visualize in csv
```

```bash
python /home/yixuan/omniteleop/scripts/vis_episode_online.py # rerun, transmission latency plot under debug/
python /home/yixuan/omniteleop/scripts/vis_eef_curves.py
```

