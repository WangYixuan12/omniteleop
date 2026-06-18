"""Whole-body IK for the Dexmate (Vega) robot using the Mink framework (MuJoCo).

A faithful port of the reference solver ``deps/rby1-wbc/rby1/whole_body_ik.py``
(Mink on MuJoCo), adapted to Vega. It is an alternative to the Pinocchio/Pink
:class:`omniteleop.follower.whole_body_ik.VegaWholeBodyIK`, and produces the same
:class:`WBCResult` so the demos can use either.

MuJoCo loads the Vega URDF (and its meshes) directly; ``mujoco.MjSpec`` is used to add
a **free base joint** and **end-effector sites** programmatically (no hand-authored
MJCF). The base is a 6-DOF free joint constrained to the ground plane by a base task +
a free-joint velocity limit (ported from the reference), mirroring rby1-wbc exactly.

Task costs use Mink's magnitude (EE ~1e4) but the **same ratios** as the tuned Pink
version: EE >> posture, base xy moderate, base yaw small (so the base rotates to follow
operator turns -- see ``whole_body_ik.py`` notes).
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import mink
import mujoco
import numpy as np
from mink import Constraint, Limit
from scipy.spatial import ConvexHull

from omniteleop.follower import wbc_safety
from omniteleop.follower.wbc_safety import SafetyGate, collision_group
from omniteleop.follower.whole_body_ik import (
    DEFAULT_NOMINAL_POSTURE,
    DEFAULT_URDF,
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    TORSO_JOINTS,
    WBCResult,
    _signed_dist_to_convex_polygon,
)

WHEEL_BODIES = ["B_wheel_l1", "B_wheel_l2", "R_wheel_l1", "R_wheel_l2",
                "L_wheel_l1", "L_wheel_l2"]
LEFT_EE_SITE = "L_ee_site"
RIGHT_EE_SITE = "R_ee_site"
HEAD_SITE = "zed_depth_frame_site"
BASE_BODY = "base"

Pose = Union[np.ndarray, "mink.SE3"]


class FreeJointVelocityLimit(Limit):
    """Per-axis velocity cap on a MuJoCo free joint (ported from deps/rby1-wbc).

    Used to constrain the base to planar motion: linear x/y and yaw allowed, z and
    roll/pitch blocked (max velocity 0).
    """

    def __init__(self, model, joint_id, ang_max, lin_max):
        self.model = model
        assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE, "joint must be free"
        dof0 = model.jnt_dofadr[joint_id]
        # MuJoCo free-joint qvel order is [vx, vy, vz, wx, wy, wz].
        self.indices = np.arange(dof0, dof0 + 6, dtype=int)
        self.limit = np.concatenate([np.asarray(lin_max, float), np.asarray(ang_max, float)])
        self.proj = np.zeros((6, model.nv))
        self.proj[np.arange(6), self.indices] = 1.0

    def compute_qp_inequalities(self, configuration, dt: float) -> Constraint:
        """Box velocity constraint on the free joint's 6 DOFs for this step."""
        vmax_dt = self.limit * float(dt)
        G = np.vstack((self.proj, -self.proj))
        h = np.hstack((vmax_dt, vmax_dt))
        return Constraint(G, h)


@dataclass
class MinkWBCConfig:
    """Tunable parameters for :class:`VegaWholeBodyIKMink` (Mink magnitude, ~1e4)."""

    urdf_path: str = DEFAULT_URDF
    # End-effector tracking (highest priority); position == orientation (ref ratio).
    ee_pos_cost: float = 10000.0
    ee_ori_cost: float = 10000.0
    lm_damping_ee: float = 1e-5
    # Base-ground task on the free joint. position (x, y, z): low xy, high z (grounded);
    # orientation (roll, pitch, yaw): high roll/pitch (upright), LOW yaw (turn to follow).
    base_position_cost: tuple = (500.0, 500.0, 100000.0)      # ratios 0.05, 0.05, 10
    base_orientation_cost: tuple = (100000.0, 100000.0, 50.0)  # ratios 10, 10, 0.005
    lm_damping_base: float = 1e-6
    posture_cost: float = 100.0   # ratio 0.01
    # Velocity limits.
    base_xy_max_vel: float = 1.0   # m/s
    base_yaw_max_vel: float = 1.0  # rad/s
    arm_max_vel: float = 2.7       # rad/s (per arm joint)
    torso_max_vel: float = 2.0
    head_max_vel: float = 3.0
    # Solver (daqp matches the reference; quadprog also available).
    solver: str = "daqp"
    damping: float = 1e-6

    # --- Safety (for real-robot use); see omniteleop.follower.wbc_safety ---------
    # Proactive self-collision avoidance (a CollisionAvoidanceLimit over the Dexmate
    # collision spheres, injected as massless MuJoCo geoms) plus a reactive hold gate
    # (stop before tipping or self-colliding). On by default; degrades gracefully to
    # off (with a warning) if the sphere URDF cannot be loaded.
    enable_collision_avoidance: bool = True
    enable_com_safety: bool = True
    collision_spheres_urdf: str = wbc_safety.DEFAULT_COLLISION_SPHERES_URDF
    self_collision_safe_dist: float = wbc_safety.DEFAULT_SELF_COLLISION_SAFE_DIST
    self_collision_floor: float = wbc_safety.DEFAULT_SELF_COLLISION_FLOOR
    collision_detect_dist: float = wbc_safety.DEFAULT_COLLISION_DETECT_DIST
    nominal_pair_keep_dist: float = wbc_safety.DEFAULT_NOMINAL_PAIR_KEEP_DIST
    collision_limit_gain: float = 0.3
    com_safety_margin: float = wbc_safety.DEFAULT_COM_SAFETY_MARGIN

    nominal_posture: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_NOMINAL_POSTURE)
    )


def _as_matrix(pose: Pose) -> np.ndarray:
    if isinstance(pose, mink.SE3):
        return pose.as_matrix()
    arr = np.asarray(pose, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"expected a 4x4 matrix or mink.SE3, got {arr.shape}")
    return arr


class VegaWholeBodyIKMink:
    """Mink/MuJoCo whole-body IK for Vega (alternative to the Pink solver)."""

    def __init__(self, config: Optional[MinkWBCConfig] = None) -> None:
        self.config = config or MinkWBCConfig()
        self._build_model()
        self._build_tasks_and_limits()
        self.reset()

    # -- construction ----------------------------------------------------------

    def _build_model(self) -> None:
        spec = mujoco.MjSpec.from_file(self.config.urdf_path)
        spec.body(BASE_BODY).add_freejoint()  # mobile base (constrained to plane below)
        for body, site in [("L_ee", LEFT_EE_SITE), ("R_ee", RIGHT_EE_SITE),
                            ("zed_depth_frame", HEAD_SITE)]:
            spec.body(body).add_site(name=site)
        # Inject the collision spheres as MuJoCo geoms BEFORE compiling.
        self._collision_geom_names: Dict[str, list] = {"body": [], "left": [], "right": []}
        if self.config.enable_collision_avoidance:
            self._add_collision_spheres(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self._ft = np.zeros(6)  # scratch fromto buffer for mj_geomDistance

        self.base_joint_id = 0  # the free joint added first
        assert self.model.jnt_type[self.base_joint_id] == mujoco.mjtJoint.mjJNT_FREE
        self._base_dof = self.model.jnt_dofadr[self.base_joint_id]  # 0

        # joint name -> qpos / dof address (for nominal posture and read-out)
        self._qadr = {}
        self._dofadr = {}
        for name in TORSO_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + HEAD_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._qadr[name] = self.model.jnt_qposadr[jid]
            self._dofadr[name] = self.model.jnt_dofadr[jid]

        self._support_polygon = self._compute_support_polygon()

        # Self-collision pairs (needs _qadr for the nominal-pose filter).
        self._collision_enabled = False
        self._collision_pair_ids: list = []
        self._collision_geom_pairs: list = []
        if any(self._collision_geom_names.values()):
            self._setup_collision_pairs()

    def _add_collision_spheres(self, spec: "mujoco.MjSpec") -> None:
        """Add the Dexmate collision spheres to ``spec`` as massless query geoms.

        Each sphere becomes a geom on its link, grouped (body/left/right) for the
        cross-group collision pairs. ``density=0`` keeps them out of the inertia/CoM;
        ``contype``/``conaffinity`` are nonzero so Mink's CollisionAvoidanceLimit
        considers them; ``group=4`` keeps them on a toggleable visual layer.
        """
        try:
            spheres = wbc_safety.parse_collision_spheres(self.config.collision_spheres_urdf)
        except Exception as exc:  # missing/unreadable URDF: disable, warn
            _warnings.warn(
                f"self-collision avoidance disabled (could not load "
                f"{self.config.collision_spheres_urdf}: {exc})",
                stacklevel=2,
            )
            return
        for link, items in spheres.items():
            grp = collision_group(link)
            if grp is None or spec.body(link) is None:
                continue
            body = spec.body(link)
            for i, (radius, center) in enumerate(items):
                geom = body.add_geom()
                geom.name = f"col_{link}_{i}"
                geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
                geom.size = [radius, 0.0, 0.0]
                geom.pos = center
                geom.contype = 1
                geom.conaffinity = 1
                geom.group = 4
                geom.density = 0.0
                geom.rgba = [1.0, 0.4, 0.0, 0.3]
                self._collision_geom_names[grp].append(geom.name)

    def _setup_collision_pairs(self) -> None:
        """Build the kept cross-group geom pairs, dropping nominal-pose overlaps.

        Filters out structurally-overlapping pairs (shoulder spheres inside the torso)
        once at the nominal posture so neither the proactive limit nor the reactive
        monitor trips on them. Produces both the geom-id pairs (reactive min-distance)
        and the singleton name pairs (proactive CollisionAvoidanceLimit).
        """
        names = self._collision_geom_names
        ids = {
            g: [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names[g]]
            for g in ("body", "left", "right")
        }
        d = mujoco.MjData(self.model)
        d.qpos[:] = self._nominal_qpos()
        mujoco.mj_forward(self.model, d)
        keep = self.config.nominal_pair_keep_dist
        for a, b in wbc_safety.CROSS_GROUPS:
            for ia, na in zip(ids[a], names[a], strict=True):
                for ib, nb in zip(ids[b], names[b], strict=True):
                    if mujoco.mj_geomDistance(self.model, d, ia, ib, 5.0, self._ft) >= keep:
                        self._collision_pair_ids.append((ia, ib))
                        self._collision_geom_pairs.append(([na], [nb]))
        self._collision_enabled = len(self._collision_pair_ids) > 0

    def _nominal_qpos(self) -> np.ndarray:
        q = self.model.qpos0.copy()
        q[0:7] = [0, 0, 0, 1, 0, 0, 0]  # base at origin, upright
        for name, val in self.config.nominal_posture.items():
            if name in self._qadr:
                q[self._qadr[name]] = val
        # clamp limited joints to their range
        for jid in range(self.model.njnt):
            if self.model.jnt_limited[jid]:
                adr = self.model.jnt_qposadr[jid]
                lo, hi = self.model.jnt_range[jid]
                q[adr] = np.clip(q[adr], lo + 1e-3, hi - 1e-3)
        return q

    def _compute_support_polygon(self) -> np.ndarray:
        """Wheel ground-contact footprint (base-frame xy) at the nominal pose."""
        d = mujoco.MjData(self.model)
        d.qpos[:] = self._nominal_qpos() if hasattr(self, "_qadr") else self.model.qpos0
        mujoco.mj_forward(self.model, d)
        pts = []
        for name in WHEEL_BODIES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                pts.append(d.xpos[bid][:2])
        pts = np.array(pts)
        return pts[ConvexHull(pts).vertices]

    def _build_tasks_and_limits(self) -> None:
        cfg = self.config
        self.left_ee_task = mink.FrameTask(LEFT_EE_SITE, "site", cfg.ee_pos_cost,
                                           cfg.ee_ori_cost, lm_damping=cfg.lm_damping_ee)
        self.right_ee_task = mink.FrameTask(RIGHT_EE_SITE, "site", cfg.ee_pos_cost,
                                            cfg.ee_ori_cost, lm_damping=cfg.lm_damping_ee)
        self.base_task = mink.FrameTask(BASE_BODY, "body", list(cfg.base_position_cost),
                                        list(cfg.base_orientation_cost),
                                        lm_damping=cfg.lm_damping_base)
        self.posture_task = mink.PostureTask(self.model, cost=cfg.posture_cost)
        self.tasks = [self.left_ee_task, self.right_ee_task, self.base_task, self.posture_task]

        vels: Dict[str, float] = {}
        for n in TORSO_JOINTS:
            vels[n] = cfg.torso_max_vel
        for n in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS:
            vels[n] = cfg.arm_max_vel
        for n in HEAD_JOINTS:
            vels[n] = cfg.head_max_vel
        self.limits = [
            mink.ConfigurationLimit(self.model),
            mink.VelocityLimit(self.model, vels),
            FreeJointVelocityLimit(
                self.model, self.base_joint_id,
                lin_max=[cfg.base_xy_max_vel, cfg.base_xy_max_vel, 0.0],
                ang_max=[0.0, 0.0, cfg.base_yaw_max_vel],
            ),
        ]

        # Proactive self-collision avoidance (decelerates the approach near contact).
        if self._collision_enabled:
            self.limits.append(mink.CollisionAvoidanceLimit(
                model=self.model,
                geom_pairs=self._collision_geom_pairs,
                gain=cfg.collision_limit_gain,
                minimum_distance_from_collisions=cfg.self_collision_safe_dist,
                collision_detection_distance=cfg.collision_detect_dist,
            ))

        # Reactive hold gate (tip-over + self-collision); shared with the Pink backend.
        self._gate = SafetyGate(
            com_safety_margin=cfg.com_safety_margin,
            self_collision_floor=cfg.self_collision_floor,
            self_collision_warn=cfg.self_collision_safe_dist,
            enable_com=cfg.enable_com_safety,
            enable_collision=self._collision_enabled,
        )

    # -- state -----------------------------------------------------------------

    def reset(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset to ``q`` (default nominal posture) and re-anchor the posture target."""
        q0 = self._nominal_qpos() if q is None else np.asarray(q, dtype=float).copy()
        self.configuration = mink.Configuration(self.model, q0)
        self.posture_task.set_target(q0)
        self._nominal_q = q0.copy()
        self._gate.reset(self.stability_margin(q0), self._min_self_distance(q0))
        return q0

    def _min_self_distance(self, q: Optional[np.ndarray] = None) -> float:
        """Closest distance (m) over the kept cross-group sphere pairs (``inf`` if off)."""
        if not self._collision_enabled:
            return float("inf")
        self.data.qpos[:] = self.configuration.q if q is None else q
        mujoco.mj_forward(self.model, self.data)
        mn = np.inf
        for ia, ib in self._collision_pair_ids:
            d = mujoco.mj_geomDistance(self.model, self.data, ia, ib, 5.0, self._ft)
            if d < mn:
                mn = d
        return float(mn)

    @property
    def q(self) -> np.ndarray:
        """Current full MuJoCo configuration vector (copy)."""
        return self.configuration.q.copy()

    @property
    def collision_enabled(self) -> bool:
        """True if self-collision avoidance is active (spheres injected)."""
        return self._collision_enabled

    def frame_pose(self, site_name: str, q: Optional[np.ndarray] = None) -> np.ndarray:
        """World pose (4x4) of a site at ``q`` (or the current configuration)."""
        if q is None:
            self.data.qpos[:] = self.configuration.q
        else:
            self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[sid].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[sid]
        return T

    # -- solve -----------------------------------------------------------------

    def solve(self, left_target: Pose, right_target: Pose, dt: float,
              head_target: Optional[Pose] = None) -> WBCResult:
        """One differential-IK step toward the EE targets; returns a WBCResult.

        ``head_target`` is accepted for API parity with the Pink solver but unused
        (no head task; the head holds its nominal posture).
        """
        del head_target
        cfg = self.config
        self.left_ee_task.set_target(mink.SE3.from_matrix(_as_matrix(left_target)))
        self.right_ee_task.set_target(mink.SE3.from_matrix(_as_matrix(right_target)))

        # Base-ground task: hold current x, y and yaw, z = 0 (damping, like the
        # reference -- not a spring back to the origin).
        q = self.configuration.q
        yaw = float(np.arctan2(2 * (q[3] * q[6] + q[4] * q[5]),
                               1 - 2 * (q[5] ** 2 + q[6] ** 2)))
        base_target = np.eye(4)
        base_target[0, 3], base_target[1, 3] = q[0], q[1]
        c, s = np.cos(yaw), np.sin(yaw)
        base_target[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        self.base_task.set_target(mink.SE3.from_matrix(base_target))

        q_before = self.configuration.q.copy()
        success = True
        try:
            vel = mink.solve_ik(self.configuration, self.tasks, dt, cfg.solver,
                                cfg.damping, limits=self.limits)
        except Exception:  # infeasible QP / collision limit: hold pose
            vel = np.zeros(self.model.nv)
            success = False
        self.configuration.integrate_inplace(vel, dt)

        # Reactive safety gate: veto a step that tips or self-collides (and worsens).
        # Metrics are base-frame (base-pose invariant), so a collision hold freezes
        # only the joints (free-joint qpos 0:7, nv 0:6) and lets the base keep
        # tracking; a tip-over hold (or QP failure) freezes everything.
        margin = self.stability_margin(self.configuration.q)
        self_dist = self._min_self_distance()
        status = self._gate.check(margin, self_dist)
        held = status.held or not success
        if held:
            if status.hold_base or not success:
                self.configuration.update(q_before)         # freeze everything
                vel = np.zeros(self.model.nv)
            else:
                q_hold = self.configuration.q.copy()
                q_hold[7:] = q_before[7:]                    # freeze joints, keep base
                self.configuration.update(q_hold)
                vel = vel.copy()
                vel[6:] = 0.0                                # zero joint vel, keep base twist
            self_dist = self._min_self_distance()

        return self._decompose(
            vel, left_target, right_target, success,
            held=held, min_self_distance=self_dist, safety_status=status.message,
        )

    # -- read-out --------------------------------------------------------------

    def _group(self, q: np.ndarray, names) -> np.ndarray:
        return np.array([q[self._qadr[n]] for n in names])

    def base_pose_from_q(self, q: np.ndarray) -> np.ndarray:
        """Extract (x, y, yaw) of the free base from a configuration vector."""
        yaw = float(np.arctan2(2 * (q[3] * q[6] + q[4] * q[5]),
                               1 - 2 * (q[5] ** 2 + q[6] ** 2)))
        return np.array([q[0], q[1], yaw])

    def stability_margin(self, q: np.ndarray) -> float:
        """Signed distance (m) of the CoM ground projection to the support polygon."""
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        com = self.data.subtree_com[0].copy()  # whole-body CoM (world)
        x, y, yaw = self.base_pose_from_q(q)
        cy, sy = np.cos(-yaw), np.sin(-yaw)
        dx, dy = com[0] - x, com[1] - y
        com_base = np.array([cy * dx - sy * dy, sy * dx + cy * dy])
        return _signed_dist_to_convex_polygon(com_base, self._support_polygon)

    def _ee_pos_err(self, site_name: str, target: Pose) -> float:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.data.qpos[:] = self.configuration.q
        mujoco.mj_forward(self.model, self.data)
        return float(np.linalg.norm(self.data.site_xpos[sid] - _as_matrix(target)[:3, 3]))

    def _decompose(self, vel, left_target, right_target, success,
                   held: bool = False, min_self_distance: float = float("inf"),
                   safety_status: str = "ok") -> WBCResult:
        q = self.configuration.q.copy()
        base_twist = np.array([vel[self._base_dof], vel[self._base_dof + 1],
                               vel[self._base_dof + 5]])  # vx, vy, wz
        return WBCResult(
            q=q,
            velocity=vel.copy(),
            base_pose=self.base_pose_from_q(q),
            base_twist=base_twist,
            torso=self._group(q, TORSO_JOINTS),
            left_arm=self._group(q, LEFT_ARM_JOINTS),
            right_arm=self._group(q, RIGHT_ARM_JOINTS),
            head=self._group(q, HEAD_JOINTS),
            left_ee_error=self._ee_pos_err(LEFT_EE_SITE, left_target),
            right_ee_error=self._ee_pos_err(RIGHT_EE_SITE, right_target),
            success=success,
            stability_margin=self.stability_margin(q),
            held=held,
            min_self_distance=min_self_distance,
            safety_status=safety_status,
        )
