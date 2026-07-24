"""Fast regression check for the scripted Robotiq pregrasp path."""

import numpy as np

from sim_eval.tasks.pick_place_task import PickPlaceTask


class _Obj:
    def __init__(self, pos, extent=(0.06, 0.06, 0.06), top=0.84):
        self.pos = np.asarray(pos, dtype=float)
        self.aabb_extent = np.asarray(extent, dtype=float)
        self.aabb = (np.zeros(3), np.array([0.0, 0.0, top]))


class _FakeEnv:
    grip_open = 1.0
    grip_close = -1.0

    def __init__(self, apple):
        self._apple = apple
        self._left = np.eye(4)
        self._left[:3, 3] = [1.02, 0.72, 0.88]
        self._right = np.eye(4)
        self._right[:3, 3] = [1.02, 0.25, 0.88]
        self._finger_offset = np.array([0.15, 0.0, 0.0])

    def link_pose(self, name):
        return (self._left if name == "L_ee" else self._right).copy()

    def obj_pos(self, obj):
        return obj.pos.copy()

    def finger_grasp_point(self, arm):
        return self._left[:3, :3] @ self._finger_offset + self._left[:3, 3]


def test_precontact_grasp_point_approaches_apple_monotonically():
    apple = _Obj([1.25, 0.72, 0.79])
    bowl = _Obj([1.22, 0.55, 0.79])
    env = _FakeEnv(apple)
    task = PickPlaceTask()
    task.apple = apple
    task.bowl = bowl
    task.expert_reset(env)

    finger_points = [env.finger_grasp_point("left")]
    while True:
        cmd = task.expert_step(env, None)
        if cmd.phase not in {"pregrasp", "approach"}:
            break
        finger_points.append(
            cmd.left_target[:3, :3] @ task._grasp_off_local + cmd.left_target[:3, 3]
        )

    distances = np.linalg.norm(np.asarray(finger_points) - apple.pos, axis=1)
    assert np.all(np.diff(distances) <= 1e-8)
