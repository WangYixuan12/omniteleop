"""Shared safety machinery for the Vega whole-body IK solvers (real-robot use).

Both the Pink (:mod:`omniteleop.follower.whole_body_ik`) and Mink
(:mod:`omniteleop.follower.whole_body_ik_mink`) solvers use this module to add two
aggressive safety layers, so the behavior is identical across backends:

The nominal collision-sphere model has some spheres overlapping *by design* (e.g. the
shoulder spheres sit inside the torso spheres). Those structurally-overlapping pairs
are filtered out once at build time (the standard SRDF "disable always-colliding
pairs" step) via :func:`filter_nominal_overlaps`, so neither the proactive constraint
nor the reactive monitor trips on them.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Dexmate collision-sphere URDF (link names match vega_no_effector.urdf; primitives
# only, so it needs no mesh package dirs). Absolute path => readable from any env.
DEFAULT_COLLISION_SPHERES_URDF = (
    "/home/yixuan/miniforge3/envs/dexmate/lib/python3.12/site-packages/"
    "dexmate_urdf/robots/humanoid/vega_1/vega_1_collision_spheres.collision.urdf"
)

# --- safety defaults (meters) --------------------------------------------------
# Proactive separation the in-solve constraint tries to maintain between cross-group
# spheres. Reactive floor below which a *worsening* step is vetoed (robot holds).
DEFAULT_SELF_COLLISION_SAFE_DIST = 0.02
DEFAULT_SELF_COLLISION_FLOOR = 0.01
# Mink-only: distance at which the velocity limit starts decelerating the approach.
DEFAULT_COLLISION_DETECT_DIST = 0.10
# SRDF filter: drop cross-group pairs closer than this at the nominal posture.
DEFAULT_NOMINAL_PAIR_KEEP_DIST = 0.02
# Pink-only: number of closest pairs the barrier constrains (QP rows).
DEFAULT_N_COLLISION_PAIRS = 40
# CoM ground-projection must stay at least this far inside the wheel support polygon.
DEFAULT_COM_SAFETY_MARGIN = 0.08

# Cross-group collision pairs to check (never intra-group: adjacent links in a chain
# are always in contact). Mirrors deps/rby1-wbc's (base_torso, arm) and (arm, arm).
CROSS_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("body", "left"),
    ("body", "right"),
    ("left", "right"),
)


def collision_group(name: str) -> Optional[str]:
    """Map a link / collision-geom name to its group, or ``None`` to ignore it.

    Works for both Pinocchio geometry-object names and MuJoCo body/link names, which
    both start with the link name (e.g. ``"L_arm_l3_0"`` -> ``"left"``, ``"torso_l2"``
    -> ``"body"``). Wheels and the ``arm_center``/camera frames have no spheres.
    """
    if name.startswith("L_arm"):
        return "left"
    if name.startswith("R_arm"):
        return "right"
    if name.startswith(("base", "torso", "head")):
        return "body"
    return None


def parse_collision_spheres(urdf_path: str) -> Dict[str, List[Tuple[float, np.ndarray]]]:
    """Parse a collision-sphere URDF into ``{link_name: [(radius, center_xyz), ...]}``.

    Only ``<collision><geometry><sphere>`` entries are read (the Dexmate sphere model
    contains nothing else). Used by the Mink backend to inject the spheres as MuJoCo
    geoms; the Pink backend loads the same URDF directly via Pinocchio.
    """
    root = ET.parse(urdf_path).getroot()
    out: Dict[str, List[Tuple[float, np.ndarray]]] = {}
    for link in root.findall("link"):
        spheres: List[Tuple[float, np.ndarray]] = []
        for col in link.findall("collision"):
            sph = col.find("geometry/sphere")
            if sph is None:
                continue
            radius = float(sph.get("radius"))
            origin = col.find("origin")
            xyz = origin.get("xyz") if origin is not None else "0 0 0"
            center = np.array([float(v) for v in xyz.split()], dtype=float)
            spheres.append((radius, center))
        if spheres:
            out[link.get("name")] = spheres
    return out


def cross_group_index_pairs(group_of: List[Optional[str]]) -> List[Tuple[int, int]]:
    """All cross-group ``(i, j)`` index pairs given each item's group label.

    ``group_of[k]`` is the group ("body"/"left"/"right"/None) of geometry index ``k``.
    Returns the candidate self-collision pairs before the nominal-overlap filter.
    """
    members: Dict[str, List[int]] = {"body": [], "left": [], "right": []}
    for idx, grp in enumerate(group_of):
        if grp in members:
            members[grp].append(idx)
    pairs: List[Tuple[int, int]] = []
    for a, b in CROSS_GROUPS:
        for i in members[a]:
            for j in members[b]:
                pairs.append((i, j))
    return pairs


def filter_nominal_overlaps(
    pairs: List[Tuple[int, int]],
    nominal_distance: "callable",
    keep_dist: float,
) -> List[Tuple[int, int]]:
    """Drop pairs closer than ``keep_dist`` at the nominal posture (SRDF-style).

    ``nominal_distance(i, j)`` returns the current (nominal-pose) distance between the
    two geometries in meters. Structurally-overlapping pairs (e.g. shoulder spheres
    inside the torso) would otherwise make the proactive constraint infeasible and the
    reactive monitor trip immediately, so they are removed once at build time.
    """
    return [(i, j) for (i, j) in pairs if nominal_distance(i, j) >= keep_dist]


@dataclass
class SafetyStatus:
    """Outcome of one :meth:`SafetyGate.check` call.

    Both safety metrics are evaluated in the **base frame**, so they are invariant to
    the mobile base pose -- only joint motion can change them. A collision hold
    therefore freezes the arm/torso/head joints but lets the base keep tracking (base
    motion cannot self-collide and often relieves the situation, avoiding deadlocks).
    A tip-over hold freezes everything (``hold_base``), since stopping is the safe
    response and retracting the arms still recovers the margin.
    """

    held: bool          # True => reject the step, hold the previous joint configuration
    message: str        # "ok", or a "WARN:"/"HELD:" human-readable reason
    hold_base: bool = False  # True => also freeze the mobile base (tip-over)


class SafetyGate:
    """Reactive hold gate shared by both whole-body IK backends.

    Tracks the previous step's CoM stability margin and closest self-collision
    distance. A candidate step is *held* (rejected) when a metric is below its safety
    threshold **and** worse than the previous accepted step -- so the robot stops
    before tipping or self-colliding, yet motion that recovers safety is allowed.
    """

    def __init__(
        self,
        com_safety_margin: float = DEFAULT_COM_SAFETY_MARGIN,
        self_collision_floor: float = DEFAULT_SELF_COLLISION_FLOOR,
        self_collision_warn: float = DEFAULT_SELF_COLLISION_SAFE_DIST,
        enable_com: bool = True,
        enable_collision: bool = True,
    ) -> None:
        self.com_safety_margin = com_safety_margin
        self.self_collision_floor = self_collision_floor
        self.self_collision_warn = self_collision_warn
        self.enable_com = enable_com
        self.enable_collision = enable_collision
        self._prev_margin = np.inf
        self._prev_dist = np.inf

    def reset(self, margin: float, self_dist: float) -> None:
        """Anchor the previous-step baselines (call from the solver's ``reset``)."""
        self._prev_margin = float(margin)
        self._prev_dist = float(self_dist)

    def check(self, margin: float, self_dist: float) -> SafetyStatus:
        """Decide whether the candidate step (margin, self_dist) is allowed.

        Args:
            margin: CoM-over-base stability margin of the candidate config (m); the
                signed distance of the CoM ground projection to the support polygon.
            self_dist: closest cross-group self-collision distance of the candidate
                config (m); ``inf`` when collision avoidance is disabled.

        Returns:
            A :class:`SafetyStatus`; ``held`` true means the solver must revert to the
            previous configuration and zero its output velocity.
        """
        warnings: List[str] = []
        held = False
        hold_base = False

        if self.enable_com and margin < self.com_safety_margin:
            warnings.append(
                f"CoM margin {margin * 100:+.1f}cm < {self.com_safety_margin * 100:.0f}cm"
            )
            if margin < self._prev_margin:  # tipping further => hold everything
                held = True
                hold_base = True

        if self.enable_collision and self_dist < self.self_collision_warn:
            warnings.append(f"self-collision {self_dist * 100:+.1f}cm")
            if self_dist < self.self_collision_floor and self_dist < self._prev_dist:
                held = True  # freeze joints; base may keep moving (base-invariant)

        if not held:
            self._prev_margin = float(margin)
            self._prev_dist = float(self_dist)

        if not warnings:
            return SafetyStatus(False, "ok")
        prefix = "HELD" if held else "WARN"
        return SafetyStatus(held, f"{prefix}: " + "; ".join(warnings), hold_base)
