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

python /home/dexmate/yixuan/omniteleop_yifan/scripts/admittance_control.py run #calibrate first

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
python /home/yixuan/omniteleop/scripts/vis_episode.py --deploy --hdf5 /home/yixuan/Dexmate/deploy/dp/dexmate_right_eef_eef_abs_film/checkpoints/last/0/episode_0.hdf5
# rerun, transmission latency plot under debug/
```



## Misc (/home/yixuan/omniteleop/scripts/misc)

```bash
python /home/yixuan/omniteleop/scripts/misc/check_eef_pos.py # robot.right_arm.get_joint_pos() first, and the run this to calculate eef from joint 
```



# WBC

See [PIPELINE_WBC](./PIPELINE_WBC.md).

```bash
python /home/yixuan/omniteleop/scripts/inspect_joints_sapien.py

python /home/yixuan/omniteleop/scripts/wbc_waypoints_record.py --output /home/yixuan/Dexmate/wbc/wbc_waypoint_wheel_open.mp4 --physics --base-loop open

(dexmate) python /home/yixuan/omniteleop/scripts/wbc_vr_leader.py
(dexmate) python /home/yixuan/omniteleop/scripts/wbc_vr_record.py --output /home/yixuan/Dexmate/wbc/wbc_teleop_06_19_wheelclosed.mp4 --physics --base-loop closed
# raise --base-kp-xy/--base-kp-yaw for snappier physics tracking


python scripts/play_side_by_side_videos.py --video_1 /home/yixuan/Dexmate/wbc/wbc_waypoint_kinematics.mp4 --video_2 /home/yixuan/Dexmate/wbc/wbc_waypoint_wheel.mp4 --video_result /home/yixuan/Dexmate/wbc/wbc_waypoint.mp4
```

```bash
(yixuan_yifan) python /home/dexmate/yixuan/omniteleop_yifan/tests/test_head_zedx_depth.py

(yixuan_yifan) python scripts/drive_box_record.py --output /home/dexmate/yixuan/Dexmate/SLAM/test/box_closed.hdf5 --closed-loop-source odom

(yixuan_yifan) python scripts/misc/plot_drive_box.py --input /home/dexmate/yixuan/Dexmate/SLAM/test/box_closed.hdf5
```

```bash
python /home/yixuan/omniteleop/scripts/wbc_vr_leader.py --calibrate-ee-offset

(dexmate) python /home/yixuan/omniteleop/scripts/wbc_vr_leader.py

# python scripts/wbc_vr_record.py --hdf5 /home/yixuan/Dexmate/wbc/real/test.hdf5 --physics --base-loop closed
# (dexmate) python scripts/wbc_vr_robot.py --replay /home/yixuan/Dexmate/wbc/real/move_sideway.hdf5 --record
# on lambda machine
# --closed-loop-q once joint readback is confirmed

robot.chassis.set_steering_angle(0.0, wait_time=1.5)

(dexmate) python scripts/wbc_vr_robot.py --record
```



# Debug

```bash
ipython
  import numpy as np
  q = robot.right_arm.get_joint_pos().copy(); q[5] = 0.3   # j6 -> +0.3
  robot.right_arm.set_joint_pos(q.tolist(), wait_time=2.0, exit_on_reach=True)
# if dead
python dexcontrol/examples/advanced_examples/config_force_torque_sensor.py get --side both
python dexcontrol/examples/advanced_examples/config_force_torque_sensor.py set --side right --enable
```

