import numpy as np

INIT_HEAD_JOINTS = [-np.pi / 8.0, 0.0, 0.0]
INIT_TORSO_JOINTS = [np.pi / 2.0, np.pi, np.pi / 8.0]
SAFE_LEFT_ARM_JOINTS = [np.pi / 3.0, 0.0, 0.0, -np.pi * 2.0 / 3.0, 0.0, 0.0, 0.0]
SAFE_RIGHT_ARM_JOINTS = [-np.pi / 3.0, 0.0, 0.0, -np.pi * 2.0 / 3.0, 0.0, 0.0, 0.0]
INIT_LEFT_ARM_JOINTS = [
    1.4998975,
    0.19498293,
    -0.21315183,
    -1.3266282,
    0.49401367,
    -1.3039546,
    0.67662925
]
INIT_RIGHT_ARM_JOINTS = [
    -1.6348639,
    0.05716477,
    0.38512436,
    -1.6580907,
    -0.5880904,
    1.1136702,
    -0.42996886
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
