import numpy as np

INIT_HEAD_JOINTS = [-np.pi / 8.0, 0.0, 0.0]
INIT_TORSO_JOINTS = [np.pi / 2.0, np.pi, np.pi / 8.0]
INIT_LEFT_ARM_JOINTS = [np.pi / 3.0, 0.0, 0.0, -np.pi * 2.0 / 3.0, 0.0, 0.0, 0.0]
INIT_RIGHT_ARM_JOINTS = [-np.pi / 3.0, 0.0, 0.0, -np.pi * 2.0 / 3.0, 0.0, 0.0, 0.0]

# Fixed start-pose for data collection (start_mode="fixed_pose"): every episode
# begins with arms at these joint configurations regardless of human hand pose.
FIXED_LEFT_ARM_JOINTS = [
    1.3744241,
    -0.19773883,
    -0.8283245,
    -1.5078056,
    1.6454824,
    -1.2567854,
    1.0643193,
]
FIXED_RIGHT_ARM_JOINTS = [
    -1.6086193,
    0.04208861,
    0.41367093,
    -1.5254492,
    -0.30602953,
    1.2391926,
    -0.28625494,
]
INIT_JOINTS_DICT = {
    "head_j1": -np.pi / 8,
    "torso_j1": np.pi / 2,
    "torso_j2": np.pi,
    "torso_j3": np.pi / 8,
    "L_arm_j1": np.pi / 3,
    "L_arm_j4": -np.pi * 2 / 3,
    "R_arm_j1": -np.pi / 3,
    "R_arm_j4": -np.pi * 2 / 3,
}
