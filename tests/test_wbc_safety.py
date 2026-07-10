"""Safety tests for the whole-body IK solver (self-collision + tip-over hold).

Covers the Pink whole-body IK backend, plus a unit test of the backend-agnostic
:class:`~omniteleop.follower.wbc_safety.SafetyGate`.

Needs mujoco, so run with pytest in the dexmate env::

    /home/yixuan/miniforge3/envs/dexmate/bin/python -m pytest tests/test_wbc_safety.py
"""

from __future__ import annotations

import numpy as np

from omniteleop.follower.wbc_safety import SafetyGate, parse_collision_spheres
from omniteleop.follower.whole_body_ik import (
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    LEFT_EE_FRAME,
    RIGHT_ARM_JOINTS,
    RIGHT_EE_FRAME,
    ROBOTIQ_PROXY_SPHERES,
    TORSO_JOINTS,
    VegaWholeBodyIK,
    WBCConfig,
)

DT = 0.01  # 100 Hz control rate


def _steps(seconds: float) -> int:
    """Iteration count for ``seconds`` of simulated time (keeps tests dt-invariant)."""
    return round(seconds / DT)


# --- solver helpers ------------------------------------------------------------

def _to_mat(pose) -> np.ndarray:
    return np.asarray(pose.homogeneous if hasattr(pose, "homogeneous") else pose, dtype=float)


def _make(**cfg_overrides):
    """Build a solver; return ``(ik, left0_mat, right0_mat)``."""
    ik = VegaWholeBodyIK(WBCConfig(**cfg_overrides))
    ik.reset()
    return ik, _to_mat(ik.frame_pose(LEFT_EE_FRAME)), _to_mat(ik.frame_pose(RIGHT_EE_FRAME))


def _shift(mat: np.ndarray, d) -> np.ndarray:
    out = mat.copy()
    out[:3, 3] = out[:3, 3] + np.asarray(d, dtype=float)
    return out


# --- SafetyGate unit logic (no solver / no heavy deps) -------------------------

def test_safety_gate_allows_when_safe():
    gate = SafetyGate(com_safety_margin=0.05, self_collision_floor=0.01)
    gate.reset(margin=0.10, self_dist=0.10)
    st = gate.check(margin=0.09, self_dist=0.09)  # both well above thresholds
    assert not st.held and st.message == "ok"


def test_safety_gate_holds_when_worsening_below_floor():
    gate = SafetyGate(com_safety_margin=0.05, self_collision_floor=0.01)
    gate.reset(margin=0.10, self_dist=0.02)
    st = gate.check(margin=0.10, self_dist=0.005)  # self-dist below floor and dropping
    assert st.held and not st.hold_base  # collision hold => base may keep moving
    assert "self-collision" in st.message


def test_safety_gate_tipover_holds_base():
    gate = SafetyGate(com_safety_margin=0.05, self_collision_floor=0.01)
    gate.reset(margin=0.06, self_dist=0.10)
    st = gate.check(margin=0.03, self_dist=0.10)  # CoM margin below threshold and dropping
    assert st.held and st.hold_base  # tip-over => freeze everything


def test_safety_gate_allows_recovery():
    gate = SafetyGate(com_safety_margin=0.05, self_collision_floor=0.01)
    gate.reset(margin=0.10, self_dist=0.005)  # start below the collision floor
    st = gate.check(margin=0.10, self_dist=0.008)  # moving apart (improving) => allowed
    assert not st.held


# --- collision avoidance -------------------------------------------------------

def _run_cross(ik, left0, right0, seconds: float = 12.0):
    """Command the two hands deep across each other; track the closest self-distance."""
    worst = np.inf
    res = None
    for _ in range(_steps(seconds)):
        res = ik.solve(_shift(left0, [0, -0.40, 0]), _shift(right0, [0, 0.40, 0]), DT)
        worst = min(worst, res.min_self_distance)
    return res, worst


def test_collision_prevents_penetration_pink():
    ik, left0, right0 = _make()
    assert ik.collision_enabled
    _, worst = _run_cross(ik, left0, right0)
    assert worst > -0.005  # never meaningfully penetrates (mm-level numerical only)


def test_collision_barrier_includes_robotiq_proxy_pairs():
    ik, _, _ = _make()
    sm = ik.collision_sphere_model
    names = [obj.name for obj in sm.geometryObjects]
    left = {i for i, name in enumerate(names) if name.startswith("L_robotiq")}
    right = {i for i, name in enumerate(names) if name.startswith("R_robotiq")}

    assert left and right
    assert any(
        (pair.first in left and pair.second in right)
        or (pair.first in right and pair.second in left)
        for pair in sm.collisionPairs
    )


def test_collision_sphere_model_uses_robot_urdf_torso_mesh_origin():
    cfg = WBCConfig()
    raw_spheres = parse_collision_spheres(cfg.collision_spheres_urdf)
    ik = VegaWholeBodyIK(cfg)
    geoms = {obj.name: obj for obj in ik.collision_sphere_model.geometryObjects}

    # The sphere URDF comes from dexmate_urdf, while the WBC robot URDF is the
    # hardware-matching vega_with_robotiq. Their torso mesh origins differ in y,
    # so torso sphere placements must be translated into the robot URDF convention.
    expected_offsets = {
        "torso_l1_0": np.array([0.0, -0.0800, 0.0]),
        "torso_l2_0": np.array([0.0, -0.0200, 0.0]),
        "torso_l3_6": np.array([0.0, -0.0775, 0.0]),
    }
    for geom_name, offset in expected_offsets.items():
        link_name, sphere_idx = geom_name.rsplit("_", 1)
        expected = raw_spheres[link_name][int(sphere_idx)][1] + offset
        np.testing.assert_allclose(geoms[geom_name].placement.translation, expected)

    # Head mesh origins match between the two URDFs, so head spheres stay unchanged.
    np.testing.assert_allclose(
        geoms["head_l1_1"].placement.translation,
        raw_spheres["head_l1"][1][1],
    )


def test_robotiq_proxy_spheres_follow_robotiq_frames():
    ik, _, _ = _make()
    sm = ik.collision_sphere_model
    names = [obj.name for obj in sm.geometryObjects]

    for frame_name in ("L_robotiq", "R_robotiq"):
        frame_pose = ik.configuration.get_transform_frame_to_world(frame_name)
        for label, xyz, _ in ROBOTIQ_PROXY_SPHERES:
            idx = names.index(f"{frame_name}_{label}")
            expected = frame_pose.act(np.asarray(xyz, dtype=float))
            actual = ik.collision_sphere_data.oMg[idx].translation
            np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_normal_reach_not_held_pink():
    ik, left0, right0 = _make()
    res = None
    for _ in range(_steps(5.0)):
        res = ik.solve(_shift(left0, [0.05, 0, 0.05]), _shift(right0, [0.05, 0, 0.05]), DT)
    assert not res.held and res.left_ee_error < 5e-3


# --- vectorized self-collision distance equivalence (Tier 1 latency change) -----

def _coal_min_self_distance(ik) -> float:
    """Reference: closest cross-group clearance read straight from coal DistanceResults."""
    cd = ik.configuration.collision_data
    pairs = ik.configuration.collision_model.collisionPairs
    return float(min(cd.distanceResults[k].min_distance for k in range(len(pairs))))


def test_min_self_distance_matches_coal():
    """Vectorized ``_min_self_distance`` == coal's per-pair min across random configs.

    Tier 1 latency change: the reactive gate's closest-distance metric is now a numpy
    sphere reduction instead of a Python loop over coal ``DistanceResults``. For the
    all-sphere model the two are exactly equal, so the gate sees an identical value and
    control behaviour is unchanged.
    """
    ik, _, _ = _make()
    assert ik._sc_radius is not None  # noqa: SLF001 -- vectorized path active (all-sphere)
    lo, hi = ik.model.lowerPositionLimit, ik.model.upperPositionLimit
    idx = [ik._idx_q[n] for n in TORSO_JOINTS + LEFT_ARM_JOINTS  # noqa: SLF001
           + RIGHT_ARM_JOINTS + HEAD_JOINTS]
    rng = np.random.default_rng(0)
    for _ in range(40):
        q = ik.nominal_q().copy()
        for i in idx:
            q[i] = float(rng.uniform(lo[i] + 1e-3, hi[i] - 1e-3))
        ik.configuration.update(q)
        np.testing.assert_allclose(
            ik._min_self_distance(), _coal_min_self_distance(ik), atol=1e-9  # noqa: SLF001
        )

# --- tip-over ------------------------------------------------------------------

# The solver enforces tip-over *inside* the QP (the RBY1-style hard inequality on a centroid
# proxy -- by default the top torso link "torso_l3") rather than with a reactive hold.
# With a near-locked base the arms alone must perform an extreme reach that would tip
# the robot: the bound keeps the proxy near its nominal lean -- so the robot stays
# upright while moving -- where disabling it lets the CoM drop toward the support edge.
# (A free base just slides under the CoM, so the bound only bites when the base sticks.)
_LOCKED_BASE = (2000.0, 2000.0, 2000.0)
_TIP_REACH = [1.1, 0.0, -0.6]


def test_com_inequality_true_com_mode_pink():
    """The 'com' centroid (true whole-body CoM) bounds tip-over the same way."""
    on, left0, right0 = _make(enable_collision_avoidance=False,
                              base_position_cost=_LOCKED_BASE, com_centroid="com")
    worst = np.inf
    held_any = False
    for _ in range(_steps(8.5)):
        res = on.solve(_shift(left0, _TIP_REACH), _shift(right0, _TIP_REACH), DT)
        worst = min(worst, res.stability_margin)
        held_any = held_any or res.held
    assert not held_any
    assert worst > 0.05                   # CoM kept well inside the support polygon


# --- proxy (torso_l3) box vs the true-CoM gate floor ---------------------------
#
# com_centroid: "torso_l3" bounds only the torso_l3 *frame origin* over the base. In the
# kinematic chain torso_l3's own lower joint (torso_j3) and the arms sit ABOVE that
# origin, so they swing the upper-body mass -- moving the TRUE whole-body CoM that
# stability_margin scores -- without moving the proxy at all. These tests answer "can
# torso_l3 satisfy the QP box while the true CoM fails the safety gate?": yes. A config
# can sit dead centre of the box (proxy err == 0, full QP margin) yet drive the true CoM
# below the SafetyGate's com_safety_margin floor.

# Joints that leave the torso_l3 frame ORIGIN fixed (proxy err stays exactly 0, so the
# hard QP inequality is satisfied with full margin) while still moving the whole-body CoM.
_PROXY_INVARIANT_JOINTS = ["torso_j3", *LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS]


def _proxy_err(ik, q):
    """torso_l3 over-base offset minus its nominal (box-centre) value, base-frame xy."""
    return ik._centroid_over_base(q)[0] - ik._com_over_base_target  # noqa: SLF001


def test_torso_l3_proxy_blind_to_upper_body_swing():
    """torso_j3/arm motion moves the TRUE CoM but not the torso_l3 box proxy.

    Root cause of the gap: torso_l3's origin is upstream of torso_j3 and the arms, so
    their motion is invisible to ``_centroid_over_base`` (the QP box quantity) yet clearly
    shifts ``stability_margin`` (the true whole-body CoM).
    """
    ik, _, _ = _make()
    q0 = ik.reset()
    lo, hi = ik.model.lowerPositionLimit, ik.model.upperPositionLimit

    q = q0.copy()
    q[ik._idx_q["torso_j3"]] = 1.5  # noqa: SLF001 -- swing torso_l3's contents forward
    for name in ("L_arm_j2", "R_arm_j2"):
        i = ik._idx_q[name]  # noqa: SLF001
        q[i] = float(np.clip(q0[i] - 1.0, lo[i] + 1e-3, hi[i] - 1e-3))

    # The torso_l3 proxy does not move at all ...
    np.testing.assert_allclose(_proxy_err(ik, q), _proxy_err(ik, q0), atol=1e-9)
    # ... but the true whole-body CoM (what stability_margin scores) clearly does.
    assert abs(ik.stability_margin(q) - ik.stability_margin(q0)) > 0.02


def test_in_box_config_violates_true_com_gate():
    """A config dead-centre in the torso_l3 QP box can still trip the true-CoM gate.

    Existence proof. Searching only the proxy-invariant joints keeps the torso_l3
    over-base offset exactly at its nominal box centre (``proxy_err == 0`` -- the hard QP
    inequality is satisfied with full margin), yet at least one such config drives the
    TRUE whole-body CoM closer to the support edge than ``com_safety_margin``, i.e. below
    the SafetyGate floor. So "torso_l3 inside the box" does NOT imply "true CoM safe by
    the gate's own threshold", and the ``SafetyGate`` (CoM branch enabled) would hold-base
    on the worsening step from nominal into that pose while the QP box raises no objection.
    """
    ik, _, _ = _make()
    q0 = ik.reset()
    floor = ik.config.com_safety_margin
    lo, hi = ik.model.lowerPositionLimit, ik.model.upperPositionLimit
    idx = [ik._idx_q[n] for n in _PROXY_INVARIANT_JOINTS]  # noqa: SLF001

    nominal_margin = ik.stability_margin(q0)
    assert nominal_margin > floor  # the design point is safe by the gate floor

    rng = np.random.default_rng(0)
    worst_margin = np.inf
    for _ in range(600):
        q = q0.copy()
        for i in idx:
            q[i] = float(rng.uniform(lo[i] + 1e-3, hi[i] - 1e-3))
        # These joints cannot move the torso_l3 origin: the box is satisfied exactly.
        assert np.max(np.abs(_proxy_err(ik, q))) < 1e-9
        worst_margin = min(worst_margin, ik.stability_margin(q))

    # An exactly-in-box config whose true CoM is below the gate floor exists.
    assert worst_margin < floor, (
        f"no in-box config breached the {floor * 100:.0f}cm true-CoM floor "
        f"(worst {worst_margin * 100:.2f}cm)"
    )

    # The true-CoM SafetyGate would HOLD (and hold the base) on the worsening step from
    # nominal into that pose; the torso_l3 QP box, by contrast, stays fully satisfied.
    gate = SafetyGate(com_safety_margin=floor, enable_com=True, enable_collision=False)
    gate.reset(margin=nominal_margin, self_dist=np.inf)
    status = gate.check(margin=worst_margin, self_dist=np.inf)
    assert status.held and status.hold_base


def test_pink_solver_gate_guards_true_com():
    """The Pink solver arms the reactive gate's true-CoM branch (tip-over backstop).

    The proactive QP inequality bounds only the torso_l3 *proxy*; the reactive SafetyGate
    is additionally armed on the true whole-body CoM (``enable_com=True``), so a step that
    drives the real CoM below ``com_safety_margin`` while worsening is reverted. This is
    why the proxy-passes-but-true-CoM-unsafe gap (test_in_box_config_violates_true_com_gate)
    is caught at runtime rather than left unguarded.
    """
    ik, _, _ = _make()
    assert ik._gate.enable_com is True  # noqa: SLF001
