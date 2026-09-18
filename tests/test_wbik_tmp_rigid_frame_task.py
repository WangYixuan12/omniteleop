# ruff: noqa: SLF001
"""An EE task whose whole chain is pinned must leave the QP, not anchor the base.

With ``dead_joints`` covering the right arm AND the torso, R_ee is rigid to the chassis:
the task can no longer pose the gripper, only fight the head world-position term for the
one free fore/aft DOF. Left in, the QP parked the base at the least-squares tie and gave
the operator half of every forward step (books_08_24/episode_0).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from omniteleop.follower.whole_body_ik_tmp import (
    HEAD_FRAME,
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    VegaWholeBodyIK,
    WBCConfig,
)

RIGHT_ARM = tuple(f"R_arm_j{i}" for i in range(1, 8))
TORSO = ("torso_j1", "torso_j2", "torso_j3")


def _ik(dead_joints):
    cfg = dataclasses.replace(
        WBCConfig.from_yaml(), dead_joints=tuple(dead_joints), enable_collision_avoidance=False
    )
    return VegaWholeBodyIK(cfg)


def test_right_ee_task_dropped_when_its_whole_chain_is_pinned():
    ik = _ik(RIGHT_ARM + TORSO)
    assert not ik.right_ee_task_active
    assert ik.right_ee_task not in ik.tasks
    assert ik.left_ee_task in ik.tasks


@pytest.mark.parametrize("dead", [(), RIGHT_ARM, TORSO])
def test_right_ee_task_kept_while_any_chain_joint_is_free(dead):
    """A live torso still gives R_ee real authority -- the task must stay."""
    ik = _ik(dead)
    assert ik.right_ee_task_active
    assert ik.right_ee_task in ik.tasks


def test_frame_rigid_to_base_matches_the_kinematic_chain():
    ik = _ik(RIGHT_ARM + TORSO)
    assert ik._frame_rigid_to_base(RIGHT_EE_FRAME)
    assert not ik._frame_rigid_to_base(LEFT_EE_FRAME)  # seven live joints of its own
    # The head chain is rigid to the base too, but through head_ik_pinned_joints rather
    # than dead_joints -- the helper deliberately reads only the latter.
    assert not ik._frame_rigid_to_base(HEAD_FRAME)


def test_base_follows_the_head_forward_with_the_right_hand_parked():
    """The regression itself: a parked right-hand target must not halve base advance."""
    ik = _ik(RIGHT_ARM + TORSO)
    q0 = ik.reset()
    left = ik.frame_pose(LEFT_EE_FRAME, q0).homogeneous.copy()
    right = ik.frame_pose(RIGHT_EE_FRAME, q0).homogeneous.copy()
    head = ik.frame_pose(HEAD_FRAME, q0).homogeneous.copy()

    step = 0.30  # operator walks forward; head and left hand ride along, right hand stays
    head[0, 3] += step
    left[0, 3] += step
    for _ in range(400):
        result = ik.solve(left, right, 0.01, head_target=head)
        assert result.success

    advance = float(result.base_pose[0] - ik.base_pose_from_q(q0)[0])
    residual = float(head[0, 3] - ik.frame_pose(HEAD_FRAME, result.q).translation[0])
    assert advance == pytest.approx(step, abs=0.01), f"base advanced {advance:.4f} m"
    assert abs(residual) < 0.01, f"head fore/aft residual {residual:.4f} m"


def test_dropping_the_task_does_not_move_the_frozen_right_arm():
    ik = _ik(RIGHT_ARM + TORSO)
    q0 = ik.reset()
    left = ik.frame_pose(LEFT_EE_FRAME, q0).homogeneous.copy()
    right = ik.frame_pose(RIGHT_EE_FRAME, q0).homogeneous.copy()
    head = ik.frame_pose(HEAD_FRAME, q0).homogeneous.copy()
    head[0, 3] += 0.30
    for _ in range(100):
        result = ik.solve(left, right, 0.01, head_target=head)
    nominal = ik.nominal_q()
    idx = {ik.model.names[j]: ik.model.idx_qs[j] for j in range(1, ik.model.njoints)}
    for name in RIGHT_ARM + TORSO:
        assert result.q[idx[name]] == pytest.approx(nominal[idx[name]], abs=1e-12), name
    assert np.all(np.isfinite(result.base_twist))
