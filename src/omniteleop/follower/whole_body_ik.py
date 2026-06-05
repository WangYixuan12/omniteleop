"""Whole-body inverse kinematics for the Dexmate (Vega) mobile manipulator.

Following ``deps/rby1-wbc``, the QP is built explicitly (:func:`pink.build_ik`),
augmented with a hard **CoM-over-base** inequality that bounds a centroid proxy
(by default the top torso link, like the reference; optionally the true CoM) near
its nominal lean to prevent tip-over, and solved with ``daqp`` -- the reference's
solver -- rather than calling :func:`pink.solve_ik` directly.
"""

# paras in WBCConfig, consider posture_cost: 100, current_posture_cost: 100

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pinocchio as pin
import qpsolvers
from pink import Configuration, build_ik
from pink.barriers import SelfCollisionBarrier
from pink.limits import ConfigurationLimit, VelocityLimit
from pink.tasks import FrameTask, PostureTask
from scipy.spatial import ConvexHull

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
    "torso_j3": np.pi / 8,
    "L_arm_j1": np.pi / 2,
    "L_arm_j4": -np.pi / 2,
    "R_arm_j1": -np.pi / 2,
    "R_arm_j4": -np.pi / 2,
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

    # End-effector tracking (highest priority). Position and orientation weighted equally
    # (deps/rby1-wbc: ee_pos_cost == ee_ori_cost). The absolute magnitude (10000) is the
    # reference's literal value; it is only a *gauge* -- Pink's QP depends solely on the
    # ratios of the other costs to this one, so scaling every cost together is a no-op.
    # What matters is that base/posture/current below are set as ratios of it.
    ee_position_cost: float = 10000.0
    ee_orientation_cost: float = 10000.0
    lm_damping_ee: float = 1e-5

    # Base regularization. The base task is re-targeted to the *current* base pose every
    # solve (see solve()), so these costs act as velocity damping -- how hard the base
    # resists translating/rotating to help the arms reach, not a restoring force toward
    # home. These are the reference's literal values (deps/rby1-wbc:
    # base_ground_position_cost [50,50,1e5], base_ground_orientation_cost [1e5,1e5,50]);
    # only their ratio to ee_position_cost (10000) matters. Per-axis:
    #   position    -> (x, y, z): x/y = 50 (ratio 0.005 to EE) so the base may translate;
    #                  z = 1e5 is a no-op (the planar joint fixes z).
    #   orientation -> (roll, pitch, yaw): roll/pitch = 1e5 are no-ops (planar-fixed);
    #                  yaw = 50 (ratio 0.005). The yaw ratio must stay small or the base
    #                  won't rotate to follow the operator -- the arms just contort
    #                  instead (at ratio 0.05 a 45 deg turn rotated the base only ~4 deg;
    #                  at 0.005 it tracks fully). A stiffer base xy (ratio 0.05) kept the
    #                  base ~2x steadier under sensor jitter but is otherwise equivalent
    #                  for tracking now that the collision barrier no longer biases base
    #                  motion (see collision_safe_displacement_gain).
    base_position_cost: Sequence[float] = (50.0, 50.0, 100000.0)
    base_orientation_cost: Sequence[float] = (100000.0, 100000.0, 50.0)
    lm_damping_base: float = 1e-6  # matches the reference's base_ground lm_damping

    # Posture regularization toward the nominal arm/torso/head configuration.
    posture_cost: float = 50.0

    # Joint-space velocity damping -- the analog of the reference's *second* posture
    # task (deps/rby1-wbc: current_posture_cost_main). A PostureTask re-targeted to the
    # *current* configuration every solve has zero error, so it contributes only a
    # per-DOF term cost**2 * ||dq_joints||**2 to the QP Hessian: velocity damping on the
    # actuated joints. (Pink's PostureTask excludes the planar root, which base_task
    # already damps -- so this targets only joint-space jitter / null-space drift.) It is
    # markedly stronger than the global ``damping`` below (1e-6). Default matches
    # ``posture_cost`` (both 50 = the reference's literal current==nominal weighting,
    # deps/rby1-wbc: current_posture_cost_main 50; ratio 0.005 to EE) -- a mild velocity
    # damping. (Raising the ratio to ~0.01, i.e. 100 here, trims joint accel/jerk a
    # further ~10% under sensor jitter at no measurable EE cost.) 0 disables it.
    current_posture_cost: float = 50.0

    # Optional head-frame tracking (off by default; head holds nominal posture).
    enable_head_task: bool = False
    head_position_cost: float = 0.5
    head_orientation_cost: float = 0.2

    # Velocity limits. Arm/torso/head limits come from the URDF; the planar base
    # is unbounded in the URDF, so we cap it here (matches reference base limits).
    base_xy_max_vel: float = 1.0   # m/s
    base_yaw_max_vel: float = 1.0  # rad/s
    # Global headroom factor applied to every velocity limit (base + joints), matching
    # the reference's ``velocity_limit_scale`` -- keeps a margin below the hard limits.
    velocity_limit_scale: float = 0.9

    # QP solver. ``daqp`` matches the reference (deps/rby1-wbc); the solve path mirrors
    # it too -- build_ik -> augment with the CoM-over-base inequality -> solve_problem.
    solver: str = "daqp"
    damping: float = 1e-6

    # --- Safety (for real-robot use); see omniteleop.follower.wbc_safety ---------
    # Self-collision: a proactive SelfCollisionBarrier over the Dexmate collision-sphere
    # model plus a reactive hold gate (the hard guarantee). Tip-over: enforced
    # *proactively* as a hard QP inequality (the reference's
    # _add_com_over_base_xy_inequalities approach) that bounds a centroid proxy's
    # horizontal over-base offset within +/-com_safety_margin of its nominal value, so
    # the robot keeps moving while staying upright instead of stopping after the fact.
    # The barrier degrades gracefully to off (with a warning) if pink lacks barriers or
    # the sphere URDF is unavailable.
    enable_collision_avoidance: bool = True
    enable_com_safety: bool = True  # add the hard CoM-over-base QP inequality
    collision_spheres_urdf: str = wbc_safety.DEFAULT_COLLISION_SPHERES_URDF
    self_collision_safe_dist: float = wbc_safety.DEFAULT_SELF_COLLISION_SAFE_DIST
    self_collision_floor: float = wbc_safety.DEFAULT_SELF_COLLISION_FLOOR
    nominal_pair_keep_dist: float = wbc_safety.DEFAULT_NOMINAL_PAIR_KEEP_DIST
    n_collision_pairs: int = wbc_safety.DEFAULT_N_COLLISION_PAIRS
    collision_barrier_gain: float = 1.0
    # pink's SelfCollisionBarrier adds, *besides* its hard CBF distance constraint, a
    # "safe backup velocity" regularization term to the QP *objective* (weighted by this
    # gain). At Pink's O(1) task-cost scale that term is strong enough to bias the
    # redundancy resolution and stall the mobile base mid-maneuver: a 90 deg operator turn
    # rotated the base only ~53 deg and a 60 cm walk advanced it ~33 cm, because the
    # barrier's backup policy fought the base motion. 0 drops the objective term entirely
    # (gain 1.0->0 sweeps the base 53->90 deg / 33->60 cm) while the hard 2 cm collision
    # constraint is unaffected (verified still held), and matches the reference's
    # constraint-only collision avoidance (mink CollisionAvoidanceLimit has no such term).
    collision_safe_displacement_gain: float = 0.0
    # Centroid proxy whose horizontal over-base offset the tip-over inequality bounds:
    # top link "torso_l3", or "com" for the true whole-body centre of
    # mass. The torso-link proxy is a jacobian-linearized lean indicator, exactly like
    # the reference; "com" swaps in the real CoM (jacobianCenterOfMass).
    com_centroid: str = "torso_l3"
    # Half-width (m) of the box the centroid proxy's over-base offset may deviate from
    # its nominal value before the QP inequality clamps it (reference uses ~0.08 m).
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
    stability_margin: float  # signed dist (m) of true CoM ground-projection to the
                             # wheel support polygon (diagnostic); the tip-over QP
                             # bound keeps the robot upright so this stays positive
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
        self._warn_if_barrier_infeasible()

    def _warn_if_barrier_infeasible(self) -> None:
        """Warn if the self-collision barrier is already infeasible at the start posture.

        The ``SelfCollisionBarrier`` enforces a *hard* QP constraint that the closest
        cross-group self-collision distance stay >= ``self_collision_safe_dist``. If the
        nominal posture already sits below that threshold, the QP is infeasible from the
        very first solve, so every :meth:`solve` fails (``success=False``) and the robot
        holds in place while the targets move -- while the reactive gate still reports
        "ok" (it is a *solver* infeasibility, not a gate trip). The default safe distance
        clears the nominal posture by only ~2 mm, so raising it, or editing the nominal
        posture, easily tips into this silent freeze. Surface it loudly instead.
        """
        if not self._collision_enabled:
            return
        d = self._min_self_distance()
        d_min = self.config.self_collision_safe_dist
        if d < d_min:
            _warnings.warn(
                f"self-collision barrier is infeasible at the start posture: closest "
                f"self-distance {d * 100:.2f} cm < self_collision_safe_dist "
                f"{d_min * 100:.2f} cm. The whole-body QP will be infeasible and the "
                f"robot will hold in place every step (solve() success=False, gate "
                f"'ok'). Lower self_collision_safe_dist below {d * 100:.2f} cm, fix the "
                f"nominal posture, or set enable_collision_avoidance=False.",
                stacklevel=2,
            )

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
        # Global velocity headroom (reference's velocity_limit_scale): scale every
        # limit -- base and URDF joint limits alike -- before VelocityLimit reads them.
        self.model.velocityLimit *= cfg.velocity_limit_scale

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

        # Resolve the tip-over centroid proxy (a torso frame, or the true CoM).
        self._resolve_com_centroid()

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

    def _resolve_com_centroid(self) -> None:
        """Resolve the tip-over centroid proxy from ``cfg.com_centroid``.

        Sets ``_com_use_com`` (true whole-body CoM) or ``_com_frame_id`` (a torso link,
        the reference's torso-link lean proxy). For a link the tracked point is the
        link's **centre of mass** (mid-link), not the BODY-frame origin: that origin
        sits at the proximal joint (the *bottom* of the link, ~14% up for ``torso_l3``),
        whereas the reference rides mid-link (its torso-5 frame origin coincides with
        the link CoM, ~50-56% up). Tracking the CoM -- a material point on the same body
        -- restores that mid-link placement (``torso_l3`` CoM is ~56% up). The link CoM
        lever is constant in the joint frame, so it is cached here; the nominal over-base
        offset the box bound is centred on is computed later in :meth:`reset` (needs
        ``q0``).
        """
        cfg = self.config
        self._com_use_com = False
        self._com_frame_id = -1
        self._com_frame_joint = -1
        self._com_link_lever = np.zeros(3)
        self._com_over_base_target: Optional[np.ndarray] = None
        if not cfg.enable_com_safety:
            return
        name = cfg.com_centroid.strip()
        if name.lower() in ("com", "centerofmass", "center_of_mass"):
            self._com_use_com = True
        elif self.model.existFrame(name):
            self._com_frame_id = self.model.getFrameId(name)
            jid = self.model.frames[self._com_frame_id].parentJoint
            self._com_frame_joint = jid
            inertia = self.model.inertias[jid]
            # Cache the link CoM lever (mid-link point). For a (near-)massless frame,
            # fall back to a zero lever -> the frame origin, so the proxy stays defined.
            self._com_link_lever = (
                inertia.lever.copy() if inertia.mass > 1e-6 else np.zeros(3)
            )
        else:
            raise ValueError(
                f"com_centroid {name!r} is not a model frame; use a torso link name "
                f"(e.g. 'torso_l3') or 'com' for the whole-body centre of mass"
            )

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

        # Optional second posture task, re-targeted to the *current* configuration each
        # solve (zero error => pure joint-space velocity damping; see WBCConfig). The
        # analog of the reference's current_posture_task. Off by default (cost 0); the
        # base already has its own velocity damping via base_task.
        self.current_posture_task: Optional[PostureTask] = None
        if cfg.current_posture_cost > 0.0:
            self.current_posture_task = PostureTask(cost=cfg.current_posture_cost)
            self.tasks.append(self.current_posture_task)

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
                safe_displacement_gain=cfg.collision_safe_displacement_gain,
                d_min=cfg.self_collision_safe_dist,
            )
            self.barriers = [self.collision_barrier]

        # Reactive hold gate for self-collision only. Tip-over (CoM-over-base) is now
        # enforced proactively inside the QP (_add_com_over_base_inequality), so the
        # gate's CoM branch is disabled here -- unlike the Mink backend, which still
        # relies on the gate for both.
        self._gate = SafetyGate(
            self_collision_floor=cfg.self_collision_floor,
            self_collision_warn=cfg.self_collision_safe_dist,
            enable_com=False,
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

        The posture target is anchored here, so the arms return toward this posture
        when the end-effector targets do not require otherwise. The base target is
        *not* anchored: :meth:`solve` re-sets it to the current base pose each step
        (velocity damping), so the base stays where it is pushed rather than springing
        back here. The set below is just an initial seed.
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
        if self.current_posture_task is not None:
            self.current_posture_task.set_target(q0)  # seed; solve() re-targets each step
        self.base_task.set_target(  # initial seed; solve() re-targets to current pose
            self.configuration.get_transform_frame_to_world(BASE_FRAME)
        )
        self._nominal_q = q0.copy()
        # Anchor the tip-over box bound on the centroid proxy's nominal over-base offset.
        if self.config.enable_com_safety:
            self._com_over_base_target = self._centroid_over_base(q0)[0]
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
        # Damp the base at its current pose (re-targeted every solve), so it stays
        # where pushed instead of springing toward home (matches deps/rby1-wbc).
        self.base_task.set_target(
            self.configuration.get_transform_frame_to_world(BASE_FRAME)
        )
        if self.current_posture_task is not None:
            # Zero-error target at the current q => per-joint velocity damping this step.
            self.current_posture_task.set_target(self.configuration.q)

        q_before = self.configuration.q.copy()
        success = True
        qp_error = ""
        try:
            # Build the IK QP explicitly, augment it with the hard CoM-over-base
            # inequality, then solve -- the reference's build_ik -> add inequalities
            # -> solve_problem path (rather than calling pink.solve_ik directly).
            self.configuration.check_limits(safety_break=False)
            problem = build_ik(
                self.configuration, self.tasks, dt,
                self.config.damping, limits=self.limits, barriers=self.barriers,
            )
            # The reference also injects torso *velocity equalities* here (problem.A) to
            # enforce its 6-joint torso linkage (torso_0/4/5 == 0, torso_3 ==
            # -(torso_1+torso_2); 6 joints, 2 independent DOFs). Vega's torso is three
            # independent revolute joints with no mimic/coupling, so there is nothing to
            # constrain and that step is intentionally omitted (problem.A stays None).
            self._add_com_over_base_inequality(problem)
            result = qpsolvers.solve_problem(problem, solver=self.config.solver)
            if not result.found or result.x is None:
                raise RuntimeError("no QP solution (infeasible constraints?)")
            velocity = result.x / dt
        except Exception as exc:  # infeasible QP / barrier / limit hit: hold the pose
            # Don't swallow the reason: a persistent failure here (e.g. a self-collision
            # barrier that is infeasible already at the start posture, see
            # _warn_if_barrier_infeasible) otherwise manifests as the robot silently
            # freezing while the targets move, with the gate still reporting "ok".
            velocity = np.zeros(self.model.nv)
            success = False
            qp_error = f"{type(exc).__name__}: {exc}"

        self.configuration.integrate_inplace(velocity, dt)

        # Reactive self-collision gate (tip-over is enforced by the QP inequality
        # above, so it is not re-checked here). The self-collision metric is base-frame
        # (base-pose invariant), so a collision hold freezes only the joints (planar
        # base nq=4, nv=3) and lets the base keep tracking; a QP failure freezes all.
        margin = self.stability_margin(self.configuration.q)
        self_dist = self._min_self_distance()
        status = self._gate.check(margin, self_dist)
        held = status.held or not success
        if held:
            if not success:
                self.configuration.update(q_before)          # QP failed: freeze all
                velocity = np.zeros(self.model.nv)
            else:
                q_hold = self.configuration.q.copy()
                q_hold[4:] = q_before[4:]                     # freeze joints, keep base
                self.configuration.update(q_hold)
                velocity = velocity.copy()
                velocity[3:] = 0.0                           # zero joint vel, keep base twist
            self_dist = self._min_self_distance()

        safety_status = status.message if success else f"QP solve failed -- {qp_error}"
        return self._decompose(
            velocity, left_target, right_target, success,
            held=held, min_self_distance=self_dist, safety_status=safety_status,
        )

    def _centroid_over_base(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Horizontal offset of the tip-over centroid from the base, and its Jacobian.

        Returns ``(d_base_xy, jac_base_xy)`` where ``d_base_xy`` (2,) is the centroid's
        position relative to the base, expressed in the base frame and projected to the
        ground plane, and ``jac_base_xy`` (2 x nv) is ``d(d_base_xy)/dΔq``. The centroid
        is the true whole-body CoM (``com_centroid == "com"``) or a torso link frame
        (the reference's lean proxy).

        The offset is invariant to the planar base DOFs -- translating or yawing the
        base moves the centroid and the base together -- so those Jacobian columns are
        zeroed; only joint motion can change it.
        """
        data = self._com_data
        if self._com_use_com:
            jac = pin.jacobianCenterOfMass(self.model, data, q)  # 3 x nv; sets data.com[0]
            point = data.com[0]
        else:
            pin.forwardKinematics(self.model, data, q)
            pin.computeJointJacobians(self.model, data, q)
            pin.updateFramePlacements(self.model, data)
            jac = pin.getFrameJacobian(
                self.model, data, self._com_frame_id, pin.LOCAL_WORLD_ALIGNED
            )[:3]
            point = data.oMf[self._com_frame_id].translation
        jac = np.asarray(jac).copy()
        jac[:, :3] = 0.0  # invariant to the planar base DOFs (x, y, yaw)
        x, y, yaw = self.base_pose_from_q(q)
        cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
        rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])  # world -> base (xy block)
        d_base = rot @ np.array([point[0] - x, point[1] - y])
        jac_base = rot @ jac[:2, :]
        return d_base, jac_base

    def _add_com_over_base_inequality(self, problem: qpsolvers.Problem) -> None:
        """Augment the IK QP with the hard tip-over (CoM-over-base) inequality.

        Ports the reference's ``_add_com_over_base_xy_inequalities``: rather than a
        reactive stop, the QP itself bounds the centroid proxy's horizontal over-base
        offset within ``+/-com_safety_margin`` of its nominal value, so the robot keeps
        moving while staying upright. With ``com_centroid`` a torso frame this is the
        reference's jacobian-linearized lean proxy; with ``"com"`` it is the true CoM.

        Per axis the offset ``d(q ⊕ Δq) ≈ d(q) + J Δq`` is held within
        ``[target - margin, target + margin]``. When already outside the box the row
        relaxes to "move back only" (``h = 0``), matching the reference's recovery
        branch, so the solver is never wedged.
        """
        cfg = self.config
        if not cfg.enable_com_safety or self._com_over_base_target is None:
            return

        d_base, jac_base = self._centroid_over_base(self.configuration.q)
        err = d_base - self._com_over_base_target
        bound = cfg.com_safety_margin

        g_rows: list[np.ndarray] = []
        h_rows: list[float] = []
        for k in range(2):  # x, y axes of the over-base offset
            row = jac_base[k]
            e = float(err[k])
            if abs(e) <= bound:  # inside the box: keep both sides within +/-bound
                g_rows += [row, -row]
                h_rows += [bound - e, bound + e]
            elif e > bound:  # past the +bound: allow moving back only
                g_rows += [row]
                h_rows += [0.0]
            else:  # past the -bound: allow moving back only
                g_rows += [-row]
                h_rows += [0.0]

        g_add = np.vstack(g_rows)
        h_add = np.asarray(h_rows, dtype=float)
        if problem.G is None:
            problem.G, problem.h = g_add, h_add
        else:
            problem.G = np.vstack([problem.G, g_add])
            problem.h = np.hstack([problem.h, h_add])

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
