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

## 🚀 Quick Start

```shell
omni-arm       # Exoskeleton arm reader
omni-joycon    # JoyCon controller reader
omni-cmd       # Command processor with safety
omni-robot     # Robot controller
omni-recorder  # MDP recorder for policy learning
omni-telemetry # Telemetry viewer
```

## Data collection

1. ssh dexmate, conda activate dexmate
2. run '(dexmate) dextop node start' and 'dexsensor launch --config ~/.dexmate/sensors/default.toml --sensor head_camera' in tmux
3. ssh lambda
4. run '(dexmate) vr_robot_controller.py' and '(dexmate) vr_reader.py'
5. revise head_mode and left_arm_mode in src/omniteleop/configs/vega_1_f5d6.yaml

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
```

## Train
VR controller → fixed_pose calibration → robot-base EEF target → IK → robot motion

## Infer
camera/proprio obs → robot-base EEF target → IK → check speed/workspace/collision → robot motion