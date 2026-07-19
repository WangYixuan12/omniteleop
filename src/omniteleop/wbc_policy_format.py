"""WBC mobile-policy vector layout and pose codecs (see PLAN.md Conventions).

Single source of truth for the 10 Hz policy-facing schema shared by the dataset
porter (``scripts/port_wbc_mobile_hdf5.py``) and the rollout script
(``scripts/wbc_policy_rollout.py``):

``observation.state`` (32,): base-frame ACHIEVED left/right EEF (pos3 + rot6) +
gripper, base-frame achieved ``zed_depth_frame`` head pose (pos3 + rot6), then the
measured odometry base pose ``(x, y, yaw)`` in the engage-origin world frame — the
policy's only absolute world anchor (images are egocentric).

``action`` (29,): the WORLD-frame targets given VERBATIM to ``ik.solve()`` at the
record tick — left/right EEF targets + binary gripper commands + the
post-LPF/post-deadband head target. Rollout feeds these straight back into the
``TargetInterpolator`` -> ``ik.solve()`` path with no base composition.

numpy-only to IMPORT (like ``wbc_stream``): the porter, the rollout script, and
stub-loaded recorder tests can import it without pulling pinocchio or hardware
deps. :class:`WBCPolicyFK` defers its heavy imports (pinocchio/pink through
``VegaWholeBodyIK``) to instantiation.
"""

from __future__ import annotations

import numpy as np

LEFT_EEF_AXES = [
    "left_tx",
    "left_ty",
    "left_tz",
    "left_r00",
    "left_r10",
    "left_r20",
    "left_r01",
    "left_r11",
    "left_r21",
    "left_gripper",
]
RIGHT_EEF_AXES = [
    "right_tx",
    "right_ty",
    "right_tz",
    "right_r00",
    "right_r10",
    "right_r20",
    "right_r01",
    "right_r11",
    "right_r21",
    "right_gripper",
]
HEAD_AXES = [
    "head_tx",
    "head_ty",
    "head_tz",
    "head_r00",
    "head_r10",
    "head_r20",
    "head_r01",
    "head_r11",
    "head_r21",
]
BASE_AXES = ["base_x", "base_y", "base_yaw"]
ACTION_AXES = LEFT_EEF_AXES + RIGHT_EEF_AXES + HEAD_AXES
STATE_AXES = ACTION_AXES + BASE_AXES

# observation.state EEF/head frame. "base": achieved FK in the base frame (the
# egocentric default, absolute-action training). "world": each EEF/head block
# composed to the engage-origin world via the odometry base pose, so the state
# shares the ACTION's world frame -- REQUIRED for use_relative_actions, whose
# relative step subtracts observation.state[:29] from the world action element-wise
# (cross-frame subtraction is meaningless otherwise). See PLAN.md / port_wbc_mobile_hdf5.
STATE_FRAMES = ("base", "world")

# Action grippers are categorical open/closed commands. Dataset conversion and
# live policy rollout share this threshold so their command semantics cannot drift.
GRIPPER_BINARY_THRESHOLD = 0.5

# Dim groups of the 29-D action layout (observation.state[:29] mirrors it),
# derived from the axis names so they can never drift from ACTION_AXES.
POSITION_DIMS = [i for i, a in enumerate(ACTION_AXES) if a.rsplit("_", 1)[-1] in ("tx", "ty", "tz")]
ROTATION_DIMS = [i for i, a in enumerate(ACTION_AXES) if a.rsplit("_", 1)[-1].startswith("r")]
GRIPPER_DIMS = [i for i, a in enumerate(ACTION_AXES) if a.endswith("_gripper")]

# Under use_relative_actions, translations AND 6-D rotation columns are made
# relative (LeRobot subtracts observation.state[:29] element-wise, broadcast over
# the chunk). The column-wise rotation delta is NOT itself a rotation, but it is
# exactly inverted by AbsoluteActionsProcessorStep -- the same chunk-anchor state
# is added back BEFORE pos6d_to_mat's Gram-Schmidt -- so decoded rotations are
# exact. Only the binary grippers stay absolute: the state gripper is a raw FC03
# reading on a different scale, so subtracting it would corrupt the 0/1 command.
# Base pose dims 29-31 live in the STATE only (never in the 29-D action).
RELATIVE_EXCLUDE_DIMS = GRIPPER_DIMS

# 6-D rotation dims ride raw via skip_normalization_dims (absolute columns are
# already unit-bounded; relative rotation deltas concentrate near 0).
SKIP_NORMALIZATION_DIMS = ROTATION_DIMS


def mat_to_pos6d(matrix: np.ndarray) -> np.ndarray:
    """(4,4) pose -> 9-vector [t, R[:,0], R[:,1]] (float32), the pos3+rot6 encoding."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"expected (4,4) pose, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("pose contains non-finite values")
    return np.concatenate([arr[:3, 3], arr[:3, 0], arr[:3, 1]]).astype(np.float32)


def pos6d_to_mat(pos6d: np.ndarray) -> np.ndarray:
    """9-vector [t, R[:,0], R[:,1]] -> (4,4) pose, Gram-Schmidt re-orthonormalized.

    The inverse of :func:`mat_to_pos6d` for valid rotations; for a noisy policy
    output the two columns are projected back onto SO(3) (Zhou et al. 6D rotation).
    """
    v = np.asarray(pos6d, dtype=np.float64).reshape(-1)
    if v.shape != (9,):
        raise ValueError(f"expected 9 values, got {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError("pos6d contains non-finite values")
    a1, a2 = v[3:6], v[6:9]
    n1 = np.linalg.norm(a1)
    if n1 < 1e-8:
        raise ValueError("pos6d first rotation column is degenerate (norm ~ 0)")
    b1 = a1 / n1
    b2 = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(b2)
    if n2 < 1e-8:
        raise ValueError("pos6d rotation columns are colinear")
    b2 = b2 / n2
    b3 = np.cross(b1, b2)
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = v[:3]
    out[:3, :3] = np.stack([b1, b2, b3], axis=1)
    return out


def base_pose_to_mat(base_xyyaw: np.ndarray) -> np.ndarray:
    """Planar base pose (x, y, yaw) -> (4,4) world_T_base (z = 0, yaw about world z)."""
    b = np.asarray(base_xyyaw, dtype=np.float64).reshape(-1)
    if b.shape != (3,):
        raise ValueError(f"expected base pose (3,), got {b.shape}")
    if not np.all(np.isfinite(b)):
        raise ValueError("base pose contains non-finite values")
    c, s = np.cos(b[2]), np.sin(b[2])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:2, 3] = b[:2]
    return T


def transform_pose(left_T_right: np.ndarray, right_T_obj: np.ndarray) -> np.ndarray:
    """Compose two (4,4) poses: left_T_obj = left_T_right @ right_T_obj."""
    a = np.asarray(left_T_right, dtype=np.float64)
    b = np.asarray(right_T_obj, dtype=np.float64)
    if a.shape != (4, 4) or b.shape != (4, 4):
        raise ValueError(f"expected two (4,4) poses, got {a.shape} and {b.shape}")
    return a @ b


class WBCPolicyFK:
    """FK on the SAME WBC model the follower solves with (``wbik.yaml`` URDF).

    The offline==online guarantee for ``observation.state``: achieved base-frame
    EEF/head poses come from ``VegaWholeBodyIK``'s own model (``vega_with_robotiq``),
    never a different URDF or a SAPIEN ``KinHelper``. Shared by the dataset porter
    (``scripts/port_wbc_mobile_hdf5.py``) and the rollout state builder
    (``scripts/wbc_policy_rollout.py``). Heavy deps import here, not at module load.
    """

    def __init__(self, ik=None) -> None:
        """Reuse an existing ``VegaWholeBodyIK`` or build a collision-free one.

        ``ik``: the rollout's live solver (FKs on its own model); ``None`` builds a
        collision-free instance (porter).
        """
        from omniteleop.follower.whole_body_ik import (  # -- heavy, lazy
            HEAD_FRAME,
            HEAD_JOINTS,
            LEFT_ARM_JOINTS,
            LEFT_EE_FRAME,
            RIGHT_ARM_JOINTS,
            RIGHT_EE_FRAME,
            TORSO_JOINTS,
            VegaWholeBodyIK,
            WBCConfig,
        )
        from omniteleop.wbc_robot_util import set_base_in_q

        self.ik = VegaWholeBodyIK(WBCConfig(enable_collision_avoidance=False)) if ik is None else ik
        self.left_ee_frame = LEFT_EE_FRAME
        self.right_ee_frame = RIGHT_EE_FRAME
        self.head_frame = HEAD_FRAME
        self._groups = (
            ("torso", list(TORSO_JOINTS)),
            ("left_arm", list(LEFT_ARM_JOINTS)),
            ("right_arm", list(RIGHT_ARM_JOINTS)),
            ("head", list(HEAD_JOINTS)),
        )
        self._set_base_in_q = set_base_in_q

    def q_from_raw(self, torso, left_arm, right_arm, head, base_xyyaw=None) -> np.ndarray:
        """Full pinocchio q from recorder-convention joint groups (+ optional base).

        ``base_xyyaw=None`` puts the planar root at the origin, so downstream FK
        poses are base-frame -- the ``observation.state`` convention.
        """
        base = (
            np.zeros(3, dtype=float)
            if base_xyyaw is None
            else np.asarray(base_xyyaw, dtype=float).reshape(-1)
        )
        if base.shape != (3,) or not np.all(np.isfinite(base)):
            raise ValueError(f"base_xyyaw must be finite (3,), got {base!r}")
        q = self._set_base_in_q(self.ik.nominal_q(), base)
        values = {"torso": torso, "left_arm": left_arm, "right_arm": right_arm, "head": head}
        for group, names in self._groups:
            arr = np.asarray(values[group], dtype=float).reshape(-1)
            if arr.shape != (len(names),):
                raise ValueError(f"{group} joints must have shape ({len(names)},), got {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{group} joints contain non-finite values")
            for name, value in zip(names, arr, strict=True):
                q[self.ik._idx_q[name]] = float(value)  # noqa: SLF001
        return q

    def frame_pose(self, frame_name: str, q: np.ndarray) -> np.ndarray:
        """World pose (4,4) of ``frame_name`` at ``q`` (base-frame when base is zero)."""
        return np.asarray(self.ik.frame_pose(frame_name, q).homogeneous, dtype=np.float64)

    def base_frame_poses(self, torso, left_arm, right_arm, head) -> dict[str, np.ndarray]:
        """base_T_{L_ee, R_ee, zed_depth_frame} at the given joints (base zero in q)."""
        q = self.q_from_raw(torso, left_arm, right_arm, head, base_xyyaw=None)
        return {
            "left": self.frame_pose(self.left_ee_frame, q),
            "right": self.frame_pose(self.right_ee_frame, q),
            "head": self.frame_pose(self.head_frame, q),
        }


def build_state_vector(
    base_poses: dict,
    base_pose: np.ndarray,
    grip_left: float,
    grip_right: float,
    *,
    state_frame: str = "base",
) -> np.ndarray:
    """Assemble the 32-D ``observation.state`` from base-frame EEF/head poses.

    Single source of truth for the state schema, shared by the porter and the
    rollout so they never disagree on the frame.

    Args:
        base_poses: ``{"left","right","head"}`` -> ``base_T_frame`` (4,4), the
            achieved FK poses (base at zero), e.g. from ``WBCPolicyFK.base_frame_poses``.
        base_pose: measured odometry ``(x, y, yaw)`` in the engage-origin world.
        grip_left, grip_right: raw achieved gripper readings (state grippers).
        state_frame: ``"base"`` keeps the EEF/head blocks base-frame (absolute-action
            default); ``"world"`` composes each block to world via ``base_pose``
            (``world_T_frame = base_pose_to_mat(base_pose) @ base_T_frame``) so the
            state shares the world ACTION frame -- required for use_relative_actions.

    The base pose occupies dims 29-31 in BOTH frames (the world anchor).
    """
    if state_frame not in STATE_FRAMES:
        raise ValueError(f"state_frame must be one of {STATE_FRAMES}, got {state_frame!r}")
    bp = np.asarray(base_pose, dtype=np.float64).reshape(-1)
    if bp.shape != (3,) or not np.all(np.isfinite(bp)):
        raise ValueError(f"base_pose must be finite (3,), got {base_pose!r}")
    world_T_base = base_pose_to_mat(bp) if state_frame == "world" else None

    def block(name: str) -> np.ndarray:
        T = np.asarray(base_poses[name], dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"base_poses[{name!r}] must be (4,4), got {T.shape}")
        if world_T_base is not None:
            T = world_T_base @ T
        return mat_to_pos6d(T)

    state = np.concatenate(
        [
            block("left"),
            [grip_left],
            block("right"),
            [grip_right],
            block("head"),
            bp,
        ]
    ).astype(np.float32)
    if state.shape != (len(STATE_AXES),):
        raise ValueError(f"built state {state.shape}, expected ({len(STATE_AXES)},)")
    return state
