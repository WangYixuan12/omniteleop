#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import math

import numpy as np
from yixuan_utilities.kinematics_helper import KinHelper

##########################
RIGHT_ARM_JOINTS=[-0.7908,  0.0321, -0.3563, -1.2637, -2.0933,  0.4582, -0.6889]
##########################

DEFAULT_TORSO = [math.pi / 2.0, math.pi, math.pi / 8.0]
DEFAULT_HEAD = [-math.pi / 8.0, 0.0, 0.0]


def parse_vec(text: str, expected_len: int, name: str) -> list[float]:
    value = ast.literal_eval(text)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list/tuple, got {type(value).__name__}")
    if len(value) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {len(value)}")
    return [float(x) for x in value]


def put_joints(qpos: np.ndarray, name_to_idx: dict[str, int], names: list[str], values: list[float]) -> None:
    if len(names) != len(values):
        raise ValueError(f"joint names/values length mismatch: {len(names)} vs {len(values)}")
    missing = [name for name in names if name not in name_to_idx]
    if missing:
        raise ValueError(f"joints not found in robot model: {missing}")
    for name, value in zip(names, values, strict=True):
        qpos[name_to_idx[name]] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-name", default="vega_no_effector")
    parser.add_argument("--right-link", default="R_ee")
    parser.add_argument("--right-arm", default=str(RIGHT_ARM_JOINTS), help="7 right arm joints, Python-list syntax")
    parser.add_argument("--torso", default=str(DEFAULT_TORSO), help="3 torso joints")
    parser.add_argument("--head", default=str(DEFAULT_HEAD), help="3 head joints")
    args = parser.parse_args()

    right_arm = parse_vec(args.right_arm, 7, "--right-arm")
    torso = parse_vec(args.torso, 3, "--torso")
    head = parse_vec(args.head, 3, "--head")

    kin = KinHelper(args.robot_name)
    name_to_idx = {joint.name: i for i, joint in enumerate(kin.sapien_robot.get_active_joints())}

    if args.right_link not in kin.link_name_to_idx:
        raise ValueError(f"link {args.right_link!r} not found")

    qpos = np.zeros(kin.sapien_robot.dof, dtype=np.float64)
    put_joints(qpos, name_to_idx, ["torso_j1", "torso_j2", "torso_j3"], torso)
    put_joints(qpos, name_to_idx, [f"R_arm_j{i}" for i in range(1, 8)], right_arm)
    put_joints(qpos, name_to_idx, ["head_j1", "head_j2", "head_j3"], head)

    right_ee = kin.compute_fk_from_link_idx(qpos, [kin.link_name_to_idx[args.right_link]])[0]
    xyz = right_ee[:3, 3]

    print("robot/world_T_right_ee:")
    print(np.array2string(right_ee, precision=6, suppress_small=False))
    print("xyz:", np.array2string(xyz, precision=6))
    print("z:", float(xyz[2]))


if __name__ == "__main__":
    main()