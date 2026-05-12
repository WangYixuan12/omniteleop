#!/usr/bin/env python3
"""Right-arm Cartesian workspace check for VR teleoperation.

The user mapped a safe Cartesian region for the right end-effector (in robot
base frame) by manually driving the EEF to each corner and reading off xyz via
``src/omniteleop/leader/check_eef_pos.py``. This module wraps that FK pipeline
into a reusable checker so the follower can reject unsafe joint commands at
runtime.

Usage as a CLI (mirrors ``check_eef_pos.py``)::

    python -m omniteleop.follower.workspace_check \\
        --right-arm "[-1.047, 0, 0, -2.094, 0, 0, 0]"
"""

from __future__ import annotations

import argparse
import ast
import math
from typing import Sequence

import numpy as np
from yixuan_utilities.kinematics_helper import KinHelper

DEFAULT_RIGHT_BOUNDS: dict[str, tuple[float, float]] = {
    "x": (0.0, 0.75),
    "y": (-0.70, 0.20),
    "z": (0.72, 1.10), # 0.72 is hitting the table
}
DEFAULT_RIGHT_LINK = "R_ee"
DEFAULT_ROBOT_NAME = "vega_no_effector"

_DEFAULT_TORSO = [math.pi / 2.0, math.pi, math.pi / 8.0]
_DEFAULT_HEAD = [-math.pi / 8.0, 0.0, 0.0]
_DEFAULT_RIGHT_ARM = [-math.pi / 3.0, 0.0, 0.0, -math.pi * 2.0 / 3.0, 0.0, 0.0, 0.0]


class WorkspaceChecker:
    """FK + axis-aligned bounds check for a single arm end-effector."""

    def __init__(
        self,
        robot_name: str = DEFAULT_ROBOT_NAME,
        arm_link: str = DEFAULT_RIGHT_LINK,
        bounds: dict[str, tuple[float, float]] | None = None,
        torso_joint_names: Sequence[str] = ("torso_j1", "torso_j2", "torso_j3"),
        arm_joint_names: Sequence[str] = tuple(f"R_arm_j{i}" for i in range(1, 8)),
        head_joint_names: Sequence[str] = ("head_j1", "head_j2", "head_j3"),
    ) -> None:
        self.bounds = dict(bounds) if bounds is not None else dict(DEFAULT_RIGHT_BOUNDS)
        for axis in ("x", "y", "z"):
            if axis not in self.bounds:
                raise ValueError(f"bounds is missing axis {axis!r}")
            lo, hi = self.bounds[axis]
            if lo > hi:
                raise ValueError(f"bounds[{axis!r}] has lo > hi: ({lo}, {hi})")

        self.kin = KinHelper(robot_name)
        if arm_link not in self.kin.link_name_to_idx:
            raise ValueError(f"link {arm_link!r} not found in {robot_name!r}")
        self.arm_link = arm_link
        self.arm_link_idx = self.kin.link_name_to_idx[arm_link]

        name_to_idx = {
            j.name: i for i, j in enumerate(self.kin.sapien_robot.get_active_joints())
        }
        self._torso_qidx = self._resolve(name_to_idx, torso_joint_names)
        self._arm_qidx = self._resolve(name_to_idx, arm_joint_names)
        self._head_qidx = self._resolve(name_to_idx, head_joint_names)

        self._qpos = np.zeros(self.kin.sapien_robot.dof, dtype=np.float64)

    @staticmethod
    def _resolve(name_to_idx: dict[str, int], names: Sequence[str]) -> list[int]:
        missing = [n for n in names if n not in name_to_idx]
        if missing:
            raise ValueError(f"joints not found in robot model: {missing}")
        return [name_to_idx[n] for n in names]

    def compute_eef_xyz(
        self,
        right_arm: Sequence[float],
        head: Sequence[float],
        torso: Sequence[float],
    ) -> np.ndarray:
        """Run FK on (torso, right_arm, head) and return EEF xyz in robot base frame."""
        if len(right_arm) != len(self._arm_qidx):
            raise ValueError(
                f"right_arm has {len(right_arm)} joints, expected {len(self._arm_qidx)}"
            )
        if len(head) != len(self._head_qidx):
            raise ValueError(
                f"head has {len(head)} joints, expected {len(self._head_qidx)}"
            )
        if len(torso) != len(self._torso_qidx):
            raise ValueError(
                f"torso has {len(torso)} joints, expected {len(self._torso_qidx)}"
            )

        for q, v in zip(self._torso_qidx, torso, strict=True):
            self._qpos[q] = v
        for q, v in zip(self._arm_qidx, right_arm, strict=True):
            self._qpos[q] = v
        for q, v in zip(self._head_qidx, head, strict=True):
            self._qpos[q] = v

        pose = self.kin.compute_fk_from_link_idx(self._qpos, [self.arm_link_idx])[0]
        return np.asarray(pose[:3, 3], dtype=np.float64)

    def is_in_workspace(
        self,
        right_arm: Sequence[float],
        head: Sequence[float],
        torso: Sequence[float],
    ) -> tuple[bool, np.ndarray]:
        """Return (in_bounds, xyz). Bounds are inclusive on both sides."""
        xyz = self.compute_eef_xyz(right_arm, head, torso)
        in_bounds = (
            self.bounds["x"][0] <= xyz[0] <= self.bounds["x"][1]
            and self.bounds["y"][0] <= xyz[1] <= self.bounds["y"][1]
            and self.bounds["z"][0] <= xyz[2] <= self.bounds["z"][1]
        )
        return in_bounds, xyz


# ── CLI entry point ───────────────────────────────────────────────────────────


def _parse_vec(text: str, expected_len: int, name: str) -> list[float]:
    value = ast.literal_eval(text)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list/tuple, got {type(value).__name__}")
    if len(value) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {len(value)}")
    return [float(x) for x in value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--robot-name", default=DEFAULT_ROBOT_NAME)
    parser.add_argument("--right-link", default=DEFAULT_RIGHT_LINK)
    parser.add_argument(
        "--right-arm", default=str(_DEFAULT_RIGHT_ARM), help="7 right arm joints"
    )
    parser.add_argument("--torso", default=str(_DEFAULT_TORSO), help="3 torso joints")
    parser.add_argument("--head", default=str(_DEFAULT_HEAD), help="3 head joints")
    args = parser.parse_args()

    right_arm = _parse_vec(args.right_arm, 7, "--right-arm")
    torso = _parse_vec(args.torso, 3, "--torso")
    head = _parse_vec(args.head, 3, "--head")

    checker = WorkspaceChecker(robot_name=args.robot_name, arm_link=args.right_link)
    in_bounds, xyz = checker.is_in_workspace(
        right_arm=right_arm, head=head, torso=torso
    )

    print(f"xyz: [{xyz[0]:+.6f}, {xyz[1]:+.6f}, {xyz[2]:+.6f}]")
    print("axis  lo         val        hi         status")
    for axis, val in zip("xyz", xyz, strict=True):
        lo, hi = checker.bounds[axis]
        ok = "OK" if lo <= val <= hi else "OUT"
        print(f"  {axis}   {lo:+.4f}   {val:+.4f}   {hi:+.4f}   {ok}")
    print(f"IN_WORKSPACE: {in_bounds}")


if __name__ == "__main__":
    main()
