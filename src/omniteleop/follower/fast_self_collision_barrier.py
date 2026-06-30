"""Tier 2 latency prototype: a vectorized all-sphere self-collision barrier.

`pink.barriers.SelfCollisionBarrier` dominates the WBC tick (~5.5 ms of 10 ms): each
solve it reads coal's per-pair ``DistanceResult.min_distance`` over **all ~2900**
cross-group pairs twice (in ``compute_barrier`` and ``compute_jacobian``, two Python
loops over pybind objects), ranks them, then builds Jacobian rows for the closest
``dim`` pairs with two ``getJointJacobian`` calls *per pair* (~240 calls).

The Dexmate collision model is **all spheres**, so every quantity pink derives from
coal has an exact closed form from the sphere centres ``c`` (= ``oMg[i].translation``)
and radii ``r``:

* signed clearance of pair ``(a, b)``:  ``d = ‖c_a - c_b‖ - r_a - r_b``  (== coal
  ``min_distance``, validated bit-identically across separation/contact/penetration);
* nearest (witness) points:  ``w1 = c_a - r_a u``, ``w2 = c_b + r_b u`` with
  ``u = (c_a - c_b)/‖c_a - c_b‖``;
* contact normal:  ``n = (w1 - w2)/‖w1 - w2‖ = sign(d) · u``.

:class:`FastSelfCollisionBarrier` overrides ``compute_barrier`` / ``compute_jacobian``
with this math: one vectorized numpy pass over the centres to rank pairs, then the
Jacobian rows assembled with one ``getJointJacobian`` per **unique** joint (≤ ~20, not
~240) and a couple of ``einsum`` contractions. It reads only ``collision_data.oMg``
(geometry placements) and the robot Jacobians ``pin.computeJointJacobians`` already
populated for the task rows -- **never coal's narrowphase results** -- so it reproduces
pink's exact ``(h, J)`` and therefore the exact QP, while skipping the all-pair scan.

The base :class:`~pink.barriers.barrier.Barrier` turns ``(h, J)`` into the QP rows
identically (``G = -J/dt``, ``h = gain · h``); with ``safe_displacement_gain = 0`` (the
WBC config) the barrier's QP **objective** is exactly zero, so reproducing ``compute_
barrier`` and ``compute_jacobian`` is sufficient for full QP equivalence.

This is a PROTOTYPE (Tier 2 in ``TODO.md``). It is a behavior-preserving drop-in for the
barrier object, but the *full* latency win also needs the live ``Configuration`` to stop
running coal ``computeDistances`` every tick (it does so unconditionally when a
``collision_model`` is attached) -- that integration step touches ``whole_body_ik.py``
and is deliberately deferred. See ``build_centers_from_placements`` for the
narrowphase-free center source used by the standalone profiling/equivalence harness.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
from pink.barriers import SelfCollisionBarrier
from pink.configuration import Configuration

_LWA = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED


class FastSelfCollisionBarrier(SelfCollisionBarrier):
    """All-sphere vectorized drop-in for :class:`pink.barriers.SelfCollisionBarrier`.

    Reproduces the stock barrier's ``compute_barrier`` (h) and ``compute_jacobian`` (J)
    exactly for a sphere-only collision model, without reading coal's per-pair
    ``DistanceResult``. Construction caches, per kept collision pair, the two geometry
    indices, their parent-joint ids, and the per-geometry radii.

    Args:
        collision_model: the (reduced) sphere ``pin.GeometryModel`` whose
            ``collisionPairs`` the live ``Configuration`` evaluates -- the same object
            passed to ``Configuration(collision_model=...)``. Every geometry must be a
            ``coal.Sphere``; otherwise a ``TypeError`` is raised (use the stock barrier).
        n_collision_pairs, gain, safe_displacement_gain, d_min: forwarded to
            :class:`SelfCollisionBarrier` unchanged.
    """

    def __init__(
        self,
        collision_model: pin.GeometryModel,
        n_collision_pairs: int,
        gain: float = 1.0,
        safe_displacement_gain: float = 0.0,
        d_min: float = 0.02,
    ) -> None:
        super().__init__(
            n_collision_pairs=n_collision_pairs,
            gain=gain,
            safe_displacement_gain=safe_displacement_gain,
            d_min=d_min,
        )
        geoms = collision_model.geometryObjects
        non_sphere = [
            g.name for g in geoms if type(g.geometry).__name__ != "Sphere"
        ]
        if non_sphere:
            raise TypeError(
                "FastSelfCollisionBarrier requires an all-sphere collision model; "
                f"non-sphere geometries: {non_sphere[:5]}"
                f"{'...' if len(non_sphere) > 5 else ''}. Use SelfCollisionBarrier."
            )
        pairs = collision_model.collisionPairs
        if len(pairs) < self.dim:
            raise ValueError(
                f"n_collision_pairs ({self.dim}) exceeds the number of collision "
                f"pairs in the model ({len(pairs)})."
            )
        # Per-pair geometry indices and parent joints (constant for a fixed model);
        # per-geometry radii. Index k aligns with collision_data.distanceResults[k],
        # so the selection matches the stock barrier's argpartition over the same k.
        self._n_pairs = len(pairs)
        self._gi = np.fromiter((p.first for p in pairs), dtype=np.intp, count=len(pairs))
        self._gj = np.fromiter((p.second for p in pairs), dtype=np.intp, count=len(pairs))
        self._ji = np.fromiter(
            (geoms[p.first].parentJoint for p in pairs), dtype=np.intp, count=len(pairs)
        )
        self._jj = np.fromiter(
            (geoms[p.second].parentJoint for p in pairs), dtype=np.intp, count=len(pairs)
        )
        self._radius = np.array([float(g.geometry.radius) for g in geoms], dtype=float)

    # -- shared broadphase ------------------------------------------------------

    def _validate(self, configuration: Configuration) -> None:
        """Reject a configuration whose collision pairs no longer match the cache.

        The cached pair/joint/radius arrays are built once from the model passed at
        construction. If the live ``Configuration`` mutates ``collisionPairs`` (add /
        remove), the cached indices would silently address the wrong pairs, so fail loud
        (mirrors the pair-count guard the reactive gate's ``_min_self_distance`` applies).
        """
        n = len(configuration.collision_model.collisionPairs)
        if n != self._n_pairs:
            raise ValueError(
                f"collision-pair count changed ({n} != cached {self._n_pairs}); "
                "rebuild FastSelfCollisionBarrier for the current model."
            )

    def _centers(self, configuration: Configuration) -> np.ndarray:
        """World-frame sphere centres ``(ngeom, 3)`` from the geometry placements."""
        oMg = configuration.collision_data.oMg
        return np.array([oMg[i].translation for i in range(len(oMg))], dtype=float)

    def _select(self, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Signed clearances over all pairs and the ``dim`` closest pair indices.

        ``clearance[k] = ‖c_a - c_b‖ - r_a - r_b`` for pair ``k = (a, b)`` -- bit-equal
        to coal's ``distanceResults[k].min_distance`` for spheres. The closest-``dim``
        selection uses the identical ``argpartition(-clearance, -dim)[-dim:]`` the stock
        barrier applies (a constant ``d_min`` shift leaves the partition unchanged), so
        the chosen pair *set* matches.
        """
        diff = centers[self._gi] - centers[self._gj]
        cdist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        clearance = cdist - self._radius[self._gi] - self._radius[self._gj]
        idx = np.argpartition(-clearance, -self.dim)[-self.dim :]
        return clearance, idx

    # -- pink Barrier overrides -------------------------------------------------

    def compute_barrier(self, configuration: Configuration) -> np.ndarray:
        """Barrier vector ``h = clearance - d_min`` over the ``dim`` closest pairs."""
        self._validate(configuration)
        clearance, idx = self._select(self._centers(configuration))
        return clearance[idx] - self.d_min

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """Stacked gradients ``∂h/∂q`` of the ``dim`` closest pairs (== pink's).

        Row ``i`` of pair ``(a, b)`` with normal ``n`` and witness points ``w1, w2``:

            J_i = nᵀ J¹_p + (r1 x n)ᵀ J¹_w - nᵀ J²_p - (r2 x n)ᵀ J²_w,

        with ``r1 = w1 - o_{j1}``, ``r2 = w2 - o_{j2}`` (joint origins) and ``J^{1,2}``
        the LOCAL_WORLD_ALIGNED joint Jacobians. Identical to
        :meth:`SelfCollisionBarrier.compute_jacobian` but the witness points / normal
        come from the sphere closed form and each unique joint's Jacobian is fetched
        once and reused across pairs.
        """
        self._validate(configuration)
        model, data = configuration.model, configuration.data
        centers = self._centers(configuration)
        _, idx = self._select(centers)

        a, b = self._gi[idx], self._gj[idx]
        j1, j2 = self._ji[idx], self._jj[idx]

        c1, c2 = centers[a], centers[b]
        sep = c1 - c2
        sep_norm = np.linalg.norm(sep, axis=1, keepdims=True)
        u = sep / np.where(sep_norm == 0.0, 1.0, sep_norm)
        w1 = c1 - self._radius[a, None] * u
        w2 = c2 + self._radius[b, None] * u
        dvec = w1 - w2
        nrm = np.linalg.norm(dvec, axis=1, keepdims=True)
        # Coincident witness points (||w1-w2|| = |clearance| -> 0, i.e. just touching):
        # normal undefined, pink leaves that row zero (np.allclose(w1, w2)). Replicate
        # per-pair, dividing safely so masked rows don't leak nan/inf into the others.
        skip = np.isclose(w1, w2).all(axis=1)
        n = dvec / np.where(nrm == 0.0, 1.0, nrm)

        # One getJointJacobian per UNIQUE joint (≤ ~20), reused across pairs.
        uniq, inv = np.unique(np.concatenate([j1, j2]), return_inverse=True)
        jac = np.empty((len(uniq), 6, model.nv))
        origin = np.empty((len(uniq), 3))
        for u_idx, jid in enumerate(uniq):
            jac[u_idx] = pin.getJointJacobian(model, data, int(jid), _LWA)
            origin[u_idx] = data.oMi[int(jid)].translation
        inv1, inv2 = inv[: self.dim], inv[self.dim :]
        J1, J2 = jac[inv1], jac[inv2]
        r1 = w1 - origin[inv1]
        r2 = w2 - origin[inv2]

        # Linear (n·J_p) and angular ((rxn)·J_w) contributions, both arms, vectorized.
        rows = np.einsum("id,idv->iv", n, J1[:, :3, :] - J2[:, :3, :])
        rows += np.einsum("id,idv->iv", np.cross(r1, n), J1[:, 3:, :])
        rows -= np.einsum("id,idv->iv", np.cross(r2, n), J2[:, 3:, :])
        rows[skip] = 0.0

        # Coincident sphere CENTRES (||c_a - c_b|| -> 0, deep mutual penetration): the
        # closed-form normal u is undefined, but coal still returns finite witness points
        # along an arbitrary axis and pink builds a (generally nonzero) row -- so the
        # zeroed closed-form row above would diverge. Reproduce pink exactly for those
        # pairs from coal's witness points. Measure-zero and unreachable in closed loop
        # (the reactive gate holds at self_collision_floor, ~r_a+r_b before centres can
        # meet), but this keeps the drop-in behavior-preserving even from a pathological
        # seed/measured q. Needs the live narrowphase (always present in the solver).
        degenerate = np.flatnonzero(sep_norm[:, 0] < 1e-9)
        for i in degenerate:
            rows[i] = self._coal_jacobian_row(configuration, int(idx[i]))
        return np.nan_to_num(rows)

    def _coal_jacobian_row(self, configuration: Configuration, k: int) -> np.ndarray:
        """Pink's exact Jacobian row for pair ``k`` from coal's witness points.

        Used only for the degenerate coincident-centre fallback in
        :meth:`compute_jacobian`. Mirrors :meth:`SelfCollisionBarrier.compute_jacobian`
        verbatim for a single pair, reading ``collision_data.distanceResults[k]`` (the
        live narrowphase). Raises if that result is unavailable, rather than guessing.
        """
        model, data = configuration.model, configuration.data
        cd = configuration.collision_data
        if k >= len(cd.distanceResults):
            raise ValueError(
                f"coincident-centre fallback needs coal distanceResults[{k}], but only "
                f"{len(cd.distanceResults)} are populated (run the live narrowphase)."
            )
        dr = cd.distanceResults[k]
        w1 = np.asarray(dr.getNearestPoint1())
        w2 = np.asarray(dr.getNearestPoint2())
        if np.allclose(w1, w2):
            return np.zeros(model.nv)
        j1, j2 = int(self._ji[k]), int(self._jj[k])
        n = (w1 - w2) / np.linalg.norm(w1 - w2)
        r1 = w1 - data.oMi[j1].translation
        r2 = w2 - data.oMi[j2].translation
        J1 = pin.getJointJacobian(model, data, j1, _LWA)
        J2 = pin.getJointJacobian(model, data, j2, _LWA)
        row = n @ J1[:3] + np.cross(r1, n) @ J1[3:]
        row -= n @ J2[:3] + np.cross(r2, n) @ J2[3:]
        return row


def build_centers_from_placements(
    model: pin.Model,
    data: pin.Data,
    collision_model: pin.GeometryModel,
    collision_data: pin.GeometryData,
    q: np.ndarray,
) -> np.ndarray:
    """Sphere centres from a placement-only update -- the narrowphase-free fast path.

    ``Configuration.update`` runs coal ``computeCollisions`` + ``computeDistances``
    (the all-pair narrowphase Tier 2 removes). The barrier itself only needs the
    geometry placements, so for the standalone profiling/equivalence harness this
    populates ``collision_data.oMg`` with ``pin.updateGeometryPlacements`` alone (after
    the forward kinematics the joint Jacobians also require). Returns the ``(ngeom, 3)``
    centre array.
    """
    pin.forwardKinematics(model, data, q)
    pin.computeJointJacobians(model, data, q)
    pin.updateGeometryPlacements(model, data, collision_model, collision_data)
    oMg = collision_data.oMg
    return np.array([oMg[i].translation for i in range(len(oMg))], dtype=float)
