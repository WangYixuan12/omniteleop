"""Whole-body inverse kinematics for the Dexmate (Vega) mobile manipulator.

Task-based QP differential IK that coordinates the **mobile base, torso, and both
arms in a single solve** to track dual-arm end-effector targets. This is the key
capability beyond omniteleop's existing arm-only IK (``dexmotion`` is configured
with ``joint_regions_to_lock=["BASE"]``, so the base is never in the kinematic
chain and the chassis is a separate velocity pass-through).

The design mirrors the reference whole-body controller in ``deps/rby1-wbc``
(``rby1/whole_body_ik.py`` + ``config/wbik.yaml``), which uses the ``mink``
framework on a MuJoCo model. Here we use ``pink`` — the same task-based QP-IK
framework (same ``FrameTask``/``PostureTask``/``VelocityLimit``/``solve_ik`` API)
— on a Pinocchio model loaded directly from the existing Vega URDF. ``pink`` is
already a dependency of omniteleop (via ``dexmotion``); no MuJoCo model is needed.

The mobile base enters the kinematic chain as a **planar root joint** (x, y, yaw),
which structurally fixes z/roll/pitch to the ground plane. A low-cost base
``FrameTask`` keeps the base near home unless the arms cannot otherwise reach the
targets — the analog of the reference's ``base_ground_task``.
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pinocchio as pin
from pink import Configuration, solve_ik
from pink.limits import ConfigurationLimit, VelocityLimit
from pink.tasks import FrameTask, PostureTask
from scipy.spatial import ConvexHull

try:  # SelfCollisionBarrier needs pink>=3 + hpp-fcl; degrade gracefully without it.
    from pink.barriers import SelfCollisionBarrier
except Exception:  # pragma: no cover - older pink
    SelfCollisionBarrier = None

from omniteleop.follower import wbc_safety
from omniteleop.follower.wbc_safety import SafetyGate, collision_group

# --- Vega model constants (vega_no_effector.urdf convention) -------------------

DEFAULT_URDF = (
    "/home/yixuan/yixuan_utilities/src/yixuan_utilities/assets/robot/"
    "vega-urdf/vega_no_effector.urdf"
)

WHEEL_JOINTS = [
    "B_wheel_j1", "B_wheel_j2",
    "R_wheel_j1", "R_wheel_j2",
    "L_wheel_j1", "L_wheel_j2",
]
TORSO_JOINTS = ["torso_j1", "torso_j2", "torso_j3"]
LEFT_ARM_JOINTS = [f"L_arm_j{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]
HEAD_JOINTS = ["head_j1", "head_j2", "head_j3"]

LEFT_EE_FRAME = "L_ee"
RIGHT_EE_FRAME = "R_ee"
HEAD_FRAME = "zed_depth_frame"
BASE_FRAME = "base"

# Nominal "natural" posture, in the *URDF* joint convention. This matches the
# known-good init used by tests/test_vr_ik_sapien.py for this exact URDF.
#
# IMPORTANT: this is NOT the same convention as
# ``omniteleop.common.vr_mode_const.INIT_*`` — those constants are in the
# real-robot/dexcontrol convention (e.g. torso ~= pi there, which exceeds this
# URDF's torso limit of 1.57 rad). Do not reuse them here.
DEFAULT_NOMINAL_POSTURE: Dict[str, float] = {
    "torso_j1": np.pi / 6,
    "torso_j2": np.pi / 3,
    "torso_j3": np.pi / 6,
    "L_arm_j1": np.pi / 2,
    "L_arm_j4": -np.pi / 2,
    "R_arm_j1": -np.pi / 2,
    "R_arm_j4": -np.pi / 2,
    "head_j1": -np.pi / 8,
}

Pose = Union[np.ndarray, pin.SE3]


def _as_se3(pose: Pose) -> pin.SE3:
    """Accept a 4x4 homogeneous matrix or a ``pin.SE3`` and return ``pin.SE3``."""
    if isinstance(pose, pin.SE3):
        return pose
    arr = np.asarray(pose, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix or pin.SE3, got shape {arr.shape}")
    return pin.SE3(arr[:3, :3], arr[:3, 3])


def _support_polygon_from_model(model: pin.Model) -> np.ndarray:
    """Wheel ground-contact footprint (CCW polygon, base-frame xy) for tip-over checks.

    Computed from the full (un-reduced) model at the neutral configuration, where the
    planar base sits at the origin so the world frame coincides with the base frame.
    """
    data = model.createData()
    pin.framesForwardKinematics(model, data, pin.neutral(model))
    pts = np.array([
        data.oMf[fid].translation[:2]
        for fid, frame in enumerate(model.frames)
        if "wheel" in frame.name.lower()
    ])
    return pts[ConvexHull(pts).vertices]  # ConvexHull orders 2D vertices CCW


def _signed_dist_to_convex_polygon(point: np.ndarray, polygon: np.ndarray) -> float:
    """Signed distance (m) from ``point`` to a CCW convex ``polygon`` boundary.

    Positive when the point is inside, negative when outside (the minimum over the
    edges of the inward-normal distance).
    """
    margin = np.inf
    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        edge = polygon[(i + 1) % n] - a
        length = float(np.hypot(edge[0], edge[1]))
        if length < 1e-9:
            continue
        # Inward normal for a CCW polygon is (-edge_y, edge_x) / length.
        dist = (-edge[1] * (point[0] - a[0]) + edge[0] * (point[1] - a[1])) / length
        margin = min(margin, dist)
    return float(margin)


@dataclass
class WBCConfig:
    """Tunable parameters for :class:`VegaWholeBodyIK`.

    Task-cost magnitudes preserve the priority hierarchy of the reference
    (end-effector >> posture >> base regularization). Absolute values follow
    Pink conventions rather than Mink's, so they differ numerically from
    ``deps/rby1-wbc/config/wbik.yaml`` while keeping the same ordering.
    """

    urdf_path: str = DEFAULT_URDF

    # End-effector tracking (highest priority). Position and orientation are weighted
    # equally, matching the reference (deps/rby1-wbc: ee_pos_cost == ee_ori_cost).
    ee_position_cost: float = 1.0
    ee_orientation_cost: float = 1.0
    lm_damping_ee: float = 1e-5

    # Base regularization (keep base home unless arms can't reach). Per-axis:
    #   position  -> (x, y, z): low x/y so the base may translate; high z (grounded)
    #   orientation -> (roll, pitch, yaw): high roll/pitch (stay level); LOW yaw (turn)
    # The yaw cost must be small relative to the EE cost or the base won't rotate to
    # follow the operator -- the arms just contort instead. 0.005 = the reference's
    # yaw/EE ratio (deps/rby1-wbc: base yaw 50 vs EE 10000); at 0.05 a 45 deg operator
    # turn only rotated the base ~4 deg, at 0.005 it rotates ~42 deg. z and roll/pitch
    # are structurally fixed by the planar joint, so their costs only document intent.
    base_position_cost: Sequence[float] = (0.05, 0.05, 1.0)
    base_orientation_cost: Sequence[float] = (1.0, 1.0, 0.005)
    lm_damping_base: float = 1e-6  # matches the reference's base_ground lm_damping

    # Posture regularization toward the nominal arm/torso/head configuration.
    posture_cost: float = 1e-2

    # Optional head-frame tracking (off by default; head holds nominal posture).
    enable_head_task: bool = False
    head_position_cost: float = 0.5
    head_orientation_cost: float = 0.2

    # Velocity limits. Arm/torso/head limits come from the URDF; the planar base
    # is unbounded in the URDF, so we cap it here (matches reference base limits).
    base_xy_max_vel: float = 1.0   # m/s
    base_yaw_max_vel: float = 1.0  # rad/s

    # QP solver. ``quadprog`` is the backend available via qpsolvers in this env
    # (the reference used ``daqp``).
    solver: str = "quadprog"
    damping: float = 1e-6

    # --- Safety (for real-robot use); see omniteleop.follower.wbc_safety ---------
    # Proactive self-collision avoidance (a SelfCollisionBarrier over the Dexmate
    # collision-sphere model) plus a reactive hold gate (stop before tipping or
    # self-colliding). Both are on by default; the barrier degrades gracefully to
    # off (with a warning) if pink lacks barriers or the sphere URDF is unavailable.
    enable_collision_avoidance: bool = True
    enable_com_safety: bool = True
    collision_spheres_urdf: str = wbc_safety.DEFAULT_COLLISION_SPHERES_URDF
    self_collision_safe_dist: float = wbc_safety.DEFAULT_SELF_COLLISION_SAFE_DIST
    self_collision_floor: float = wbc_safety.DEFAULT_SELF_COLLISION_FLOOR
    nominal_pair_keep_dist: float = wbc_safety.DEFAULT_NOMINAL_PAIR_KEEP_DIST
    n_collision_pairs: int = wbc_safety.DEFAULT_N_COLLISION_PAIRS
    collision_barrier_gain: float = 1.0
    com_safety_margin: float = wbc_safety.DEFAULT_COM_SAFETY_MARGIN

    nominal_posture: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_NOMINAL_POSTURE)
    )


@dataclass
class WBCResult:
    """Output of one :meth:`VegaWholeBodyIK.solve` call."""

    q: np.ndarray            # full configuration (nq,)
    velocity: np.ndarray     # joint velocity from the QP (nv,)
    base_pose: np.ndarray    # (x, y, yaw) of the mobile base, in world frame
    base_twist: np.ndarray   # (vx, vy, wz) -> maps to chassis velocity command
    torso: np.ndarray        # (3,)
    left_arm: np.ndarray     # (7,)
    right_arm: np.ndarray    # (7,)
    head: np.ndarray         # (3,)
    left_ee_error: float     # meters, target vs. achieved L_ee position
    right_ee_error: float    # meters
    success: bool            # False if the QP solve failed (configuration held)
    stability_margin: float  # signed dist (m) of CoM ground-projection to the wheel
                             # support polygon; <= 0 => tip-over risk (see solver)
    # --- safety (see omniteleop.follower.wbc_safety) ---------------------------
    held: bool = False                     # True => step vetoed; robot holds this pose
    min_self_distance: float = float("inf")  # closest cross-group self-collision dist (m)
    safety_status: str = "ok"              # "ok", or a "WARN:"/"HELD:" reason string


class VegaWholeBodyIK:
    """Whole-body differential IK solver for the Vega mobile manipulator.

    Typical use (kinematic loop)::

        ik = VegaWholeBodyIK()
        ik.reset()                                   # start from nominal posture
        left0 = ik.frame_pose(LEFT_EE_FRAME)         # current EE poses as targets
        right0 = ik.frame_pose(RIGHT_EE_FRAME)
        while True:
            result = ik.solve(left_target, right_target, dt)
            display(result.q)                        # e.g. Meshcat / SAPIEN
    """

    def __init__(self, config: Optional[WBCConfig] = None) -> None:
        self.config = config or WBCConfig()
        self._build_model()
        self._build_tasks_and_limits()
        self.reset()

    # -- construction -----------------------------------------------------------

    def _build_model(self) -> None:
        cfg = self.config
        urdf = cfg.urdf_path
        package_dir = str(Path(urdf).parent)

        # Planar root joint => mobile base (x, y, yaw) is part of the chain.
        model_full = pin.buildModelFromUrdf(urdf, pin.JointModelPlanar())
        visual_full = pin.buildGeomFromUrdf(
            model_full, urdf, pin.GeometryType.VISUAL, package_dirs=package_dir
        )
        collision_full = pin.buildGeomFromUrdf(
            model_full, urdf, pin.GeometryType.COLLISION, package_dirs=package_dir
        )

        # Self-collision geometry: the Dexmate collision-sphere model (90 spheres;
        # link names match this URDF). Loaded as a separate geometry model so the
        # SelfCollisionBarrier uses clean sphere-sphere distances. Optional/graceful.
        sphere_full = None
        if cfg.enable_collision_avoidance and SelfCollisionBarrier is not None:
            try:
                sphere_full = pin.buildGeomFromUrdf(
                    model_full, cfg.collision_spheres_urdf, pin.GeometryType.COLLISION
                )
            except Exception as exc:  # missing URDF / no hpp-fcl: disable, warn
                _warnings.warn(
                    f"self-collision avoidance disabled (could not load "
                    f"{cfg.collision_spheres_urdf}: {exc})",
                    stacklevel=2,
                )

        # Lock the passive wheel joints; they are not whole-body DOFs.
        lock_ids = [
            model_full.getJointId(j)
            for j in WHEEL_JOINTS
            if model_full.existJointName(j)
        ]
        q_ref = pin.neutral(model_full)
        geom_models = [visual_full, collision_full]
        if sphere_full is not None:
            geom_models.append(sphere_full)
        self.model, reduced_geoms = pin.buildReducedModel(
            model_full, geom_models, lock_ids, q_ref
        )
        self.visual_model, self.collision_model = reduced_geoms[0], reduced_geoms[1]
        self.collision_sphere_model = reduced_geoms[2] if sphere_full is not None else None

        # Cap mobile-base velocity (planar nv layout: 0=vx, 1=vy, 2=yaw-rate).
        self.model.velocityLimit[0] = cfg.base_xy_max_vel
        self.model.velocityLimit[1] = cfg.base_xy_max_vel
        self.model.velocityLimit[2] = cfg.base_yaw_max_vel

        self.data = self.model.createData()

        # Tip-over stability check: a CoM data buffer (separate from the Pink
        # configuration's data so it isn't disturbed) and the wheel support polygon
        # (from the full model, since the reduced model merges the locked wheels).
        self._com_data = self.model.createData()
        self._support_polygon = _support_polygon_from_model(model_full)

        # joint name -> first index in q (used to set posture and read out groups)
        self._idx_q: Dict[str, int] = {
            self.model.names[j]: self.model.idx_qs[j]
            for j in range(1, self.model.njoints)
        }

        # Self-collision pairs (after _idx_q, since the nominal filter needs it).
        self._collision_enabled = False
        self.collision_sphere_data = None
        if self.collision_sphere_model is not None:
            self._setup_collision_pairs()

    def _setup_collision_pairs(self) -> None:
        """Add cross-group sphere collision pairs, dropping nominal overlaps.

        Pairs only the body / left-arm / right-arm groups against each other (never
        within a chain). Pairs that are closer than ``nominal_pair_keep_dist`` at the
        nominal posture -- the by-design shoulder-vs-torso overlaps -- are removed, so
        the barrier starts feasible and the reactive monitor doesn't false-trip.
        """
        cfg = self.config
        sm = self.collision_sphere_model
        group_of = [collision_group(sm.geometryObjects[i].name) for i in range(sm.ngeoms)]
        candidates = wbc_safety.cross_group_index_pairs(group_of)

        # Compute nominal-pose distances for every candidate pair in one batch.
        for i, j in candidates:
            sm.addCollisionPair(pin.CollisionPair(i, j))
        tmp_data = sm.createData()
        q_nom = self.nominal_q()
        Configuration(self.model, self.model.createData(), q_nom,
                      collision_model=sm, collision_data=tmp_data)
        dist_map = {
            candidates[k]: tmp_data.distanceResults[k].min_distance
            for k in range(len(candidates))
        }
        kept = wbc_safety.filter_nominal_overlaps(
            candidates, lambda i, j: dist_map[(i, j)], cfg.nominal_pair_keep_dist
        )

        sm.removeAllCollisionPairs()
        for i, j in kept:
            sm.addCollisionPair(pin.CollisionPair(i, j))
        self.collision_sphere_data = sm.createData()
        self._collision_enabled = len(sm.collisionPairs) > 0

    def _build_tasks_and_limits(self) -> None:
        cfg = self.config
        self.left_ee_task = FrameTask(
            LEFT_EE_FRAME, cfg.ee_position_cost, cfg.ee_orientation_cost,
            lm_damping=cfg.lm_damping_ee,
        )
        self.right_ee_task = FrameTask(
            RIGHT_EE_FRAME, cfg.ee_position_cost, cfg.ee_orientation_cost,
            lm_damping=cfg.lm_damping_ee,
        )
        self.base_task = FrameTask(
            BASE_FRAME, list(cfg.base_position_cost), list(cfg.base_orientation_cost),
            lm_damping=cfg.lm_damping_base,
        )
        # PostureTask covers actuated joints only (the planar root is excluded by
        # Pink), so base regularization lives in base_task above.
        self.posture_task = PostureTask(cost=cfg.posture_cost)

        self.tasks = [
            self.left_ee_task,
            self.right_ee_task,
            self.base_task,
            self.posture_task,
        ]

        self.head_task: Optional[FrameTask] = None
        if cfg.enable_head_task:
            self.head_task = FrameTask(
                HEAD_FRAME, cfg.head_position_cost, cfg.head_orientation_cost,
                lm_damping=cfg.lm_damping_ee,
            )
            self.tasks.append(self.head_task)

        self.config_limit = ConfigurationLimit(self.model)
        self.velocity_limit = VelocityLimit(self.model)
        self.limits = [self.config_limit, self.velocity_limit]

        # Proactive self-collision barrier (closest n pairs become QP rows).
        self.collision_barrier = None
        self.barriers = None
        if self._collision_enabled and SelfCollisionBarrier is not None:
            n_pairs = len(self.collision_sphere_model.collisionPairs)
            dim = max(1, min(cfg.n_collision_pairs, n_pairs))
            self.collision_barrier = SelfCollisionBarrier(
                n_collision_pairs=dim,
                gain=cfg.collision_barrier_gain,
                d_min=cfg.self_collision_safe_dist,
            )
            self.barriers = [self.collision_barrier]

        # Reactive hold gate (tip-over + self-collision); shared with the Mink backend.
        self._gate = SafetyGate(
            com_safety_margin=cfg.com_safety_margin,
            self_collision_floor=cfg.self_collision_floor,
            self_collision_warn=cfg.self_collision_safe_dist,
            enable_com=cfg.enable_com_safety,
            enable_collision=self._collision_enabled,
        )

    # -- posture / state --------------------------------------------------------

    def nominal_q(self) -> np.ndarray:
        """Build the nominal configuration, clamped to the URDF joint limits."""
        q = pin.neutral(self.model)
        for name, value in self.config.nominal_posture.items():
            if name in self._idx_q:
                q[self._idx_q[name]] = value
        lo, hi = self.model.lowerPositionLimit, self.model.upperPositionLimit
        finite = np.isfinite(lo) & np.isfinite(hi)
        q[finite] = np.clip(q[finite], lo[finite] + 1e-3, hi[finite] - 1e-3)
        return q

    def reset(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset the solver to ``q`` (defaults to the nominal posture).

        The posture and base regularization targets are anchored here, so the
        arms return toward this posture and the base toward this location when
        the end-effector targets do not require otherwise.
        """
        q0 = self.nominal_q() if q is None else np.asarray(q, dtype=float).copy()
        if self._collision_enabled:
            self.configuration = Configuration(
                self.model, self.data, q0,
                collision_model=self.collision_sphere_model,
                collision_data=self.collision_sphere_data,
            )
        else:
            self.configuration = Configuration(self.model, self.data, q0)
        self.posture_task.set_target(q0)
        self.base_task.set_target(
            self.configuration.get_transform_frame_to_world(BASE_FRAME)
        )
        self._nominal_q = q0.copy()
        self._gate.reset(self.stability_margin(q0), self._min_self_distance())
        return q0

    def _min_self_distance(self) -> float:
        """Closest distance (m) over the kept cross-group collision pairs.

        Reads the live ``Configuration`` collision data (refreshed on every
        ``integrate``/``update``); ``inf`` when collision avoidance is disabled.
        """
        if not self._collision_enabled:
            return float("inf")
        cd = self.configuration.collision_data
        n = len(self.configuration.collision_model.collisionPairs)
        if n == 0:
            return float("inf")
        return float(min(cd.distanceResults[k].min_distance for k in range(n)))

    @property
    def q(self) -> np.ndarray:
        """Current full configuration vector (copy)."""
        return self.configuration.q.copy()

    @property
    def collision_enabled(self) -> bool:
        """True if self-collision avoidance is active (sphere model loaded)."""
        return self._collision_enabled

    def frame_pose(self, frame: str, q: Optional[np.ndarray] = None) -> pin.SE3:
        """Forward kinematics: world pose of ``frame`` at ``q`` (or current state)."""
        if q is None:
            return self.configuration.get_transform_frame_to_world(frame)
        config = Configuration(self.model, self.data, np.asarray(q, dtype=float))
        return config.get_transform_frame_to_world(frame)

    # -- solve ------------------------------------------------------------------

    def solve(
        self,
        left_target: Pose,
        right_target: Pose,
        dt: float,
        head_target: Optional[Pose] = None,
    ) -> WBCResult:
        """Run one differential-IK step toward the given end-effector targets.

        Args:
            left_target: desired ``L_ee`` pose (4x4 matrix or ``pin.SE3``).
            right_target: desired ``R_ee`` pose.
            dt: integration timestep (s).
            head_target: optional ``zed_depth_frame`` pose (only used when the
                config enables the head task).

        Returns:
            A :class:`WBCResult` with the updated configuration and a per-component
            decomposition (base pose/twist, torso, arms, head) plus EE errors.
        """
        self.left_ee_task.set_target(_as_se3(left_target))
        self.right_ee_task.set_target(_as_se3(right_target))
        if self.head_task is not None and head_target is not None:
            self.head_task.set_target(_as_se3(head_target))

        q_before = self.configuration.q.copy()
        success = True
        try:
            velocity = solve_ik(
                self.configuration,
                self.tasks,
                dt,
                solver=self.config.solver,
                limits=self.limits,
                barriers=self.barriers,
                damping=self.config.damping,
            )
        except Exception:  # infeasible QP / barrier / limit hit: hold the current pose
            velocity = np.zeros(self.model.nv)
            success = False

        self.configuration.integrate_inplace(velocity, dt)

        # Reactive safety gate: veto a step that tips or self-collides (and worsens).
        # Self-collision/CoM metrics are base-frame (base-pose invariant), so a
        # collision hold freezes only the joints (planar base nq=4, nv=3) and lets the
        # base keep tracking; a tip-over hold (or QP failure) freezes everything.
        margin = self.stability_margin(self.configuration.q)
        self_dist = self._min_self_distance()
        status = self._gate.check(margin, self_dist)
        held = status.held or not success
        if held:
            if status.hold_base or not success:
                self.configuration.update(q_before)         # freeze everything
                velocity = np.zeros(self.model.nv)
            else:
                q_hold = self.configuration.q.copy()
                q_hold[4:] = q_before[4:]                    # freeze joints, keep base
                self.configuration.update(q_hold)
                velocity = velocity.copy()
                velocity[3:] = 0.0                           # zero joint vel, keep base twist
            self_dist = self._min_self_distance()

        return self._decompose(
            velocity, left_target, right_target, success,
            held=held, min_self_distance=self_dist, safety_status=status.message,
        )

    # -- readout ----------------------------------------------------------------

    def _group(self, q: np.ndarray, names: Sequence[str]) -> np.ndarray:
        return np.array([q[self._idx_q[n]] for n in names])

    def base_pose_from_q(self, q: np.ndarray) -> np.ndarray:
        """Extract (x, y, yaw) of the mobile base from a configuration vector.

        The planar root joint stores its rotation as (cos, sin) at q[2:4].
        """
        return np.array([q[0], q[1], float(np.arctan2(q[3], q[2]))])

    def ee_targets_from_arm_joints(
        self,
        left_arm: Sequence[float],
        right_arm: Sequence[float],
        head: Optional[Sequence[float]] = None,
    ) -> tuple[pin.SE3, pin.SE3]:
        """Recover Cartesian L_ee/R_ee target poses from arm joint angles.

        ``vr_reader`` publishes IK-solved *joint* positions, not Cartesian EEF
        poses. This builds a configuration from the nominal posture with the given
        arm (and optional head) joints substituted in, then returns the forward
        kinematics of the two end-effector frames -- the targets the whole-body
        solver then tracks while coordinating the base and torso.

        Arm joint angles use the same convention as this URDF (verified against
        ``tests/test_vr_ik_sapien.py``, which applies dexmotion's arm solutions
        directly to this model). Out-of-range values are clamped to the limits.
        """
        q = self.nominal_q()
        for name, val in zip(LEFT_ARM_JOINTS, left_arm, strict=True):
            q[self._idx_q[name]] = val
        for name, val in zip(RIGHT_ARM_JOINTS, right_arm, strict=True):
            q[self._idx_q[name]] = val
        if head is not None:
            for name, val in zip(HEAD_JOINTS, head, strict=True):
                if name in self._idx_q:
                    q[self._idx_q[name]] = val
        lo, hi = self.model.lowerPositionLimit, self.model.upperPositionLimit
        finite = np.isfinite(lo) & np.isfinite(hi)
        q[finite] = np.clip(q[finite], lo[finite] + 1e-3, hi[finite] - 1e-3)
        return self.frame_pose(LEFT_EE_FRAME, q), self.frame_pose(RIGHT_EE_FRAME, q)

    def center_of_mass(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """World-frame center of mass at ``q`` (defaults to the current state)."""
        qq = self.configuration.q if q is None else np.asarray(q, dtype=float)
        return pin.centerOfMass(self.model, self._com_data, qq).copy()

    def stability_margin(self, q: Optional[np.ndarray] = None) -> float:
        """Signed distance (m) of the CoM ground projection to the support polygon.

        Positive = the CoM is inside the wheel footprint (stable); <= 0 means the
        CoM has left the support polygon and the robot is tipping. Joint limits do
        not capture this failure mode, so it is checked separately.
        """
        qq = self.configuration.q if q is None else np.asarray(q, dtype=float)
        com = self.center_of_mass(qq)
        x, y, yaw = self.base_pose_from_q(qq)
        # CoM in the base frame (rotate world -> base by -yaw, drop z).
        cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
        dx, dy = com[0] - x, com[1] - y
        com_base = np.array([cos_y * dx - sin_y * dy, sin_y * dx + cos_y * dy])
        return _signed_dist_to_convex_polygon(com_base, self._support_polygon)

    def _decompose(
        self,
        velocity: np.ndarray,
        left_target: Pose,
        right_target: Pose,
        success: bool,
        held: bool = False,
        min_self_distance: float = float("inf"),
        safety_status: str = "ok",
    ) -> WBCResult:
        q = self.configuration.q.copy()
        left_err = float(np.linalg.norm(
            self.configuration.get_transform_frame_to_world(LEFT_EE_FRAME).translation
            - _as_se3(left_target).translation
        ))
        right_err = float(np.linalg.norm(
            self.configuration.get_transform_frame_to_world(RIGHT_EE_FRAME).translation
            - _as_se3(right_target).translation
        ))
        return WBCResult(
            q=q,
            velocity=velocity.copy(),
            base_pose=self.base_pose_from_q(q),
            base_twist=velocity[:3].copy(),  # (vx, vy, wz) -> chassis command
            torso=self._group(q, TORSO_JOINTS),
            left_arm=self._group(q, LEFT_ARM_JOINTS),
            right_arm=self._group(q, RIGHT_ARM_JOINTS),
            head=self._group(q, HEAD_JOINTS),
            left_ee_error=left_err,
            right_ee_error=right_err,
            success=success,
            stability_margin=self.stability_margin(q),
            held=held,
            min_self_distance=min_self_distance,
            safety_status=safety_status,
        )
