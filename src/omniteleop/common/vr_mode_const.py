import numpy as np

INIT_HEAD_JOINTS = [0.0, 0.0, 0.0]
INIT_TORSO_JOINTS = [np.pi / 6.0, np.pi / 3.0, np.pi / 6.0]
INIT_LEFT_ARM_JOINTS = [np.pi / 2.0, 0.0, 0.0, -np.pi / 2.0, 0.0, 0.0, 0.0]
INIT_RIGHT_ARM_JOINTS = [-np.pi / 2.0, 0.0, 0.0, -np.pi / 2.0, 0.0, 0.0, 0.0]
INIT_JOINTS_DICT = {
    "torso_j1": np.pi / 6,
    "torso_j2": np.pi / 3,
    "torso_j3": np.pi / 6,
    "L_arm_j1": np.pi / 2,
    "L_arm_j4": -np.pi / 2,
    "R_arm_j1": -np.pi / 2,
    "R_arm_j4": -np.pi / 2,
}
