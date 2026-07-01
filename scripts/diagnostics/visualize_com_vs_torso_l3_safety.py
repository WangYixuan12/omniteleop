#!/usr/bin/env python3
"""Visualize postures where the torso_l3 QP proxy is SAFE but the true CoM is UNSAFE.

Background (see also ``tests/test_wbc_safety.py`` and
``scripts/diagnostics/probe_com_centroid_safety.py``):

  * The whole-body IK enforces tip-over with a *hard QP inequality* on the
    ``com_centroid`` proxy. With the shipped ``com_centroid: "torso_l3"`` that bounds only
    the **torso_l3 frame origin** over the base, within ``+/-com_safety_margin`` of its
    nominal offset (per axis).
  * ``stability_margin`` / the ``SafetyGate`` instead score the **true whole-body CoM**'s
    signed distance to the wheel support polygon, and call it unsafe below
    ``com_safety_margin``.

These are different quantities. ``torso_j3`` (torso_l3's own lower joint) and the arms sit
ABOVE the torso_l3 origin in the kinematic chain, so they swing the upper-body mass --
moving the true CoM -- WITHOUT moving the proxy at all. This script searches those
proxy-invariant joints for postures that sit *exactly* at the centre of the torso_l3 box
(``proxy_err == 0``, so the QP inequality is satisfied with full margin) yet drive the true
CoM below the gate floor, and renders them in SAPIEN.

Each posture is drawn with floor overlays so the gap is visible at a glance:

  * GRAY  triangle  -- the wheel support polygon (tip-over boundary).
  * YELLOW triangle -- the support polygon eroded by ``com_safety_margin`` (the gate's
                       "true CoM must stay inside this" boundary).
  * GREEN  dot+box  -- the torso_l3 proxy ground point and its ``+/-com_safety_margin`` QP
                       box. The dot stays pinned at the box centre for every posture.
  * RED    dot      -- the true whole-body CoM ground projection (with a drop line from the
                       3D CoM). It pokes OUTSIDE the yellow boundary: gate-unsafe, while the
                       green proxy never leaves its box.

Run in the dexmate conda env (has sapien + pinocchio + pink)::

    # interactive viewer, auto-cycling the postures (needs a display):
    /home/yixuan/miniforge3/envs/dexmate/bin/python \
        scripts/diagnostics/visualize_com_vs_torso_l3_safety.py

    # headless: write one annotated PNG per posture:
    /home/yixuan/miniforge3/envs/dexmate/bin/python \
        scripts/diagnostics/visualize_com_vs_torso_l3_safety.py --no-viewer --out /tmp/com_gap
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from omniteleop.follower.sapien_env import prepare_sapien_render_env
from omniteleop.follower.whole_body_ik import (
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    VegaWholeBodyIK,
    WBCConfig,
)

# Joints that leave the torso_l3 frame ORIGIN fixed (proxy_err stays exactly 0, so the QP
# box is satisfied with full margin) while still swinging the whole-body CoM: torso_l3's
# own lower joint plus both arms. Restricting the search to these makes the rendered claim
# unambiguous -- the proxy sits dead centre in its box in every posture shown.
_PROXY_INVARIANT_JOINTS = ["torso_j3", *LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS]

_BLUE = [0.10, 0.35, 1.0]   # wheel support polygon (tip-over boundary)
_YELLOW = [1.0, 0.85, 0.05]  # support polygon eroded by com_safety_margin (gate boundary)
_GREEN = [0.1, 0.85, 0.1]    # torso_l3 proxy point + its QP box
_RED = [0.95, 0.08, 0.08]    # true whole-body CoM (3D, ground projection, drop line)


class Posture:
    """One found counterexample: a config in-box yet true-CoM unsafe."""

    def __init__(self, q: np.ndarray, margin: float, com: np.ndarray,
                 proxy_err: np.ndarray) -> None:
        self.q = q
        self.margin = float(margin)       # true-CoM signed dist to support polygon (m)
        self.com = np.asarray(com, float)  # world (== base) CoM, 3-vector
        self.proxy_err = np.asarray(proxy_err, float)  # torso_l3 over-base offset error (xy)


# --- posture search (no SAPIEN needed) ----------------------------------------

def find_postures(
    ik: VegaWholeBodyIK, n: int, seed: int, samples: int
) -> List[Posture]:
    """Random-search proxy-invariant joints for in-box, true-CoM-unsafe postures.

    Returns up to ``n`` postures, greedily chosen to spread their CoM ground projection
    (so the rendered leans are visually distinct), worst (lowest margin) first.
    """
    q0 = ik.reset()
    floor = float(ik.config.com_safety_margin)
    target = ik._com_over_base_target  # noqa: SLF001 -- box centre (nominal proxy offset)
    lo = ik.model.lowerPositionLimit
    hi = ik.model.upperPositionLimit
    idx = [ik._idx_q[name] for name in _PROXY_INVARIANT_JOINTS]  # noqa: SLF001

    nominal_margin = ik.stability_margin(q0)
    if not nominal_margin > floor:
        raise RuntimeError(
            f"nominal margin {nominal_margin:.3f}m is not above the floor {floor:.3f}m; "
            "the design point is already unsafe -- aborting."
        )

    rng = np.random.default_rng(seed)
    cands: List[Posture] = []
    for _ in range(samples):
        q = q0.copy()
        for i in idx:
            q[i] = float(rng.uniform(lo[i] + 1e-3, hi[i] - 1e-3))
        err = ik._centroid_over_base(q)[0] - target  # noqa: SLF001
        # These joints cannot move the torso_l3 origin; assert the invariant holds so a
        # model change that breaks it is caught loudly rather than silently mis-claimed.
        if np.max(np.abs(err)) > 1e-7:
            raise RuntimeError(
                f"proxy_err {err} is non-zero for proxy-invariant joints -- the torso_l3 "
                "origin moved; the search assumption is violated."
            )
        margin = ik.stability_margin(q)
        if margin < floor:
            cands.append(Posture(q, margin, ik.center_of_mass(q), err))

    if not cands:
        raise RuntimeError(
            f"no in-box posture breached the {floor * 100:.0f}cm true-CoM floor in "
            f"{samples} samples (worst stayed safe). Increase --samples."
        )
    cands.sort(key=lambda p: p.margin)  # worst (lowest margin) first
    return _select_diverse(cands, n)


def _select_diverse(cands: List[Posture], n: int, min_sep: float = 0.015) -> List[Posture]:
    """Pick the ``n`` deepest (lowest-margin) postures that are visually distinct.

    ``cands`` is sorted worst-first; take them in that order but skip any whose CoM ground
    projection is within ``min_sep`` of one already chosen, so the rendered set is both
    clearly unsafe and not near-duplicates. Backfill with the next-worst if too few remain.
    """
    chosen: List[Posture] = []
    for p in cands:
        if all(np.linalg.norm(p.com[:2] - c.com[:2]) >= min_sep for c in chosen):
            chosen.append(p)
            if len(chosen) == n:
                return chosen
    for p in cands:  # backfill (relax the separation) if fewer than n distinct exist
        if p not in chosen:
            chosen.append(p)
            if len(chosen) == n:
                break
    return chosen


# --- geometry helpers ---------------------------------------------------------

def erode_convex_ccw(poly: np.ndarray, margin: float) -> Optional[np.ndarray]:
    """Shrink a CCW convex polygon inward by ``margin`` (intersection of inward half-planes).

    Returns the eroded vertices (CCW), or None if the margin exceeds the inradius (the
    eroded polygon would be empty / degenerate).
    """
    poly = np.asarray(poly, dtype=float)
    n = len(poly)
    lines: List[Tuple[np.ndarray, float]] = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        edge = b - a
        length = float(np.hypot(edge[0], edge[1]))
        if length < 1e-9:
            return None
        inward = np.array([-edge[1], edge[0]]) / length  # CCW inward normal
        lines.append((inward, float(inward @ a) + margin))
    verts = []
    for i in range(n):
        (n1, c1), (n2, c2) = lines[(i - 1) % n], lines[i]
        a_mat = np.array([n1, n2])
        det = float(np.linalg.det(a_mat))
        if abs(det) < 1e-9:
            return None
        verts.append(np.linalg.solve(a_mat, np.array([c1, c2])))
    verts = np.array(verts)
    # Empty/inverted erosion check: the eroded centroid must still be inside the original.
    if not np.all(np.isfinite(verts)):
        return None
    return verts


# --- SAPIEN scene -------------------------------------------------------------

class SafetyScene:
    """Loads the Vega robot and draws the support polygon, safety boundary, CoM and proxy."""

    def __init__(self, ik: VegaWholeBodyIK, *, with_viewer: bool,
                 image_size: Optional[Tuple[int, int]], ground_z: float = -0.05) -> None:
        prepare_sapien_render_env()
        import sapien  # noqa: PLC0415 -- heavy, env-specific dep; imported on use

        self._sapien = sapien
        self._ground_z = ground_z
        # The wheel frames (hence the support polygon) sit at z ~= 0, so draw the floor
        # overlays at the robot's footprint plane rather than at the lowered ground plane.
        self._marker_z = 0.012
        self.floor = float(ik.config.com_safety_margin)
        self._support = np.asarray(ik._support_polygon, dtype=float)  # noqa: SLF001
        self._proxy_xy = np.asarray(ik._com_over_base_target, dtype=float)  # noqa: SLF001

        scene = sapien.Scene()
        scene.add_ground(ground_z)
        scene.set_ambient_light([0.55, 0.55, 0.55])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
        scene.add_directional_light([1, -1, -1], [0.3, 0.3, 0.3])
        self._scene = scene

        self._viewer = None
        if with_viewer:
            self._viewer = scene.create_viewer()
            self._viewer.set_camera_xyz(x=1.7, y=1.4, z=1.7)
            self._viewer.set_camera_rpy(r=0.0, p=-0.7, y=-np.pi / 4.0)

        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        self._robot = loader.load(ik.config.urdf_path)
        self._dof = self._robot.dof
        active = self._robot.get_active_joints()
        self._sapien_idx = {j.get_name(): i for i, j in enumerate(active)}
        self._limits = np.array([j.get_limits()[0] for j in active], dtype=float)
        # IK q-index for each SAPIEN active joint (None -> grippers/wheels, left at 0).
        self._ik_qidx = {
            j.get_name(): ik._idx_q.get(j.get_name())  # noqa: SLF001
            for j in active
        }

        self._draw_static_overlays()
        self._com_dot = self._sphere(_RED, 0.035)
        self._com_dot_3d = self._sphere(_RED, 0.028)
        self._proxy_dot = self._sphere(_GREEN, 0.028)
        self._drop_line: Optional[object] = None  # rebuilt per posture (length varies)
        self._last_posture: Optional[Posture] = None

        # Offscreen camera for headless PNG dumps: a 3/4 perspective of the posture. The
        # footprint geometry (support polygon, safety boundary, CoM projection -- which sits
        # UNDER the base and self-occludes in any 3D view) is drawn as a matplotlib top-down
        # schematic instead; save_png composites the two side by side.
        self._cam_persp = None
        self._image_size = image_size
        if image_size is not None:
            w, h = image_size
            self._cam_persp = scene.add_camera("persp", w // 2, h, 1.0, 0.05, 100.0)
            self._cam_persp.set_local_pose(
                self._look_at(eye=[1.9, -1.5, 1.7], target=[0.0, 0.0, 0.35])
            )

    # -- marker builders -------------------------------------------------------

    def _sphere(self, color, radius: float):
        builder = self._scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius, material=list(color))
        return builder.build_kinematic()

    def _segment(self, p0, p1, color, radius: float, z: float):
        """A kinematic capsule lying flat from ``p0`` to ``p1`` (xy) at height ``z``."""
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        mid = 0.5 * (p0 + p1)
        d = p1 - p0
        length = float(np.hypot(d[0], d[1]))
        yaw = float(np.arctan2(d[1], d[0]))
        builder = self._scene.create_actor_builder()
        builder.add_capsule_visual(
            self._sapien.Pose(), radius=radius, half_length=max(length / 2.0, 1e-4),
            material=list(color),
        )
        actor = builder.build_kinematic()
        actor.set_pose(self._sapien.Pose(
            p=[float(mid[0]), float(mid[1]), z],
            q=[float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))],
        ))
        return actor

    def _polygon(self, verts: np.ndarray, color, radius: float) -> None:
        verts = np.asarray(verts, float)
        n = len(verts)
        for i in range(n):
            self._segment(verts[i], verts[(i + 1) % n], color, radius, self._marker_z)

    def _draw_static_overlays(self) -> None:
        # Wheel support polygon (tip-over boundary).
        self._polygon(self._support, _BLUE, 0.014)
        # Support polygon eroded by the safety margin: the gate's true-CoM boundary.
        eroded = erode_convex_ccw(self._support, self.floor)
        if eroded is not None:
            self._polygon(eroded, _YELLOW, 0.011)
        # torso_l3 proxy +/-margin QP box, centred on the nominal proxy offset.
        cx, cy = self._proxy_xy
        m = self.floor
        box = np.array([[cx + m, cy + m], [cx - m, cy + m],
                        [cx - m, cy - m], [cx + m, cy - m]])
        self._polygon(box, _GREEN, 0.008)

    def _look_at(self, eye, target, up=(0.0, 0.0, 1.0)):
        from scipy.spatial.transform import Rotation  # noqa: PLC0415

        eye = np.asarray(eye, float)
        fwd = np.asarray(target, float) - eye
        fwd /= np.linalg.norm(fwd)
        left = np.cross(np.asarray(up, float), fwd)
        left /= np.linalg.norm(left)
        up2 = np.cross(fwd, left)
        rot = np.column_stack([fwd, left, up2])
        qx, qy, qz, qw = Rotation.from_matrix(rot).as_quat()
        return self._sapien.Pose(p=eye, q=[qw, qx, qy, qz])

    # -- per-posture update ----------------------------------------------------

    def show(self, posture: Posture) -> None:
        """Set the robot to ``posture`` and move the CoM/proxy markers."""
        self._last_posture = posture
        qpos = np.zeros(self._dof)
        for name, sidx in self._sapien_idx.items():
            qidx = self._ik_qidx[name]
            if qidx is None:
                continue
            lo, hi = self._limits[sidx]
            val = float(posture.q[qidx])
            qpos[sidx] = float(np.clip(val, lo, hi)) if np.all(np.isfinite([lo, hi])) else val
        self._robot.set_qpos(qpos)

        com = posture.com
        self._com_dot_3d.set_pose(self._sapien.Pose(p=[float(com[0]), float(com[1]),
                                                       float(com[2])]))
        self._com_dot.set_pose(self._sapien.Pose(p=[float(com[0]), float(com[1]),
                                                    self._marker_z]))
        # torso_l3 proxy ground point = nominal box centre + its (≈0) error.
        px, py = self._proxy_xy + posture.proxy_err
        self._proxy_dot.set_pose(self._sapien.Pose(p=[float(px), float(py), self._marker_z]))
        # Drop line from the 3D CoM down to its ground projection.
        if self._drop_line is not None:
            self._drop_line.remove_from_scene()
        self._drop_line = self._segment_vertical(
            [com[0], com[1], self._marker_z], [com[0], com[1], float(com[2])], _RED, 0.006)

    def _segment_vertical(self, p0, p1, color, radius: float):
        """A kinematic capsule between two points that differ in z (the CoM drop line)."""
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        mid = 0.5 * (p0 + p1)
        length = float(abs(p1[2] - p0[2]))
        builder = self._scene.create_actor_builder()
        # Capsule long axis is local +x; rotate it to +z (-90 deg about y).
        builder.add_capsule_visual(
            self._sapien.Pose(q=[0.70710678, 0.0, -0.70710678, 0.0]),
            radius=radius, half_length=max(length / 2.0, 1e-4), material=list(color),
        )
        actor = builder.build_kinematic()
        actor.set_pose(self._sapien.Pose(p=[float(mid[0]), float(mid[1]), float(mid[2])]))
        return actor

    # -- output ----------------------------------------------------------------

    def render_once(self) -> None:
        """Update the scene and draw one viewer frame (no-op without a viewer)."""
        self._scene.update_render()
        if self._viewer is not None:
            self._viewer.render()

    def _grab(self, camera) -> np.ndarray:
        camera.take_picture()
        rgba = np.asarray(camera.get_picture("Color"))
        return (np.clip(rgba[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)[..., ::-1]

    def _schematic_bgr(self, posture: Posture, width_px: int, height_px: int) -> np.ndarray:
        """Top-down footprint schematic (BGR) -- the unoccluded safety picture.

        Shows the wheel support polygon, the safety boundary (eroded by com_safety_margin),
        the torso_l3 +/-margin QP box with its proxy point, and the true-CoM ground
        projection. The proxy stays in its box while the CoM sits outside the boundary.
        """
        import matplotlib  # noqa: PLC0415 -- optional dep, only for the schematic panel
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib.patches import Polygon as MplPolygon  # noqa: PLC0415

        dpi = 100
        fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
        ax = fig.add_subplot(111)
        support = np.vstack([self._support, self._support[0]])
        ax.plot(support[:, 0], support[:, 1], color="#1a5cff", lw=2.5,
                label="wheel support (tip-over)")
        eroded = erode_convex_ccw(self._support, self.floor)
        if eroded is not None:
            poly = np.vstack([eroded, eroded[0]])
            ax.add_patch(MplPolygon(eroded, closed=True, facecolor="#fff3c4",
                                    edgecolor="#e6b800", lw=2.0, alpha=0.6))
            ax.plot(poly[:, 0], poly[:, 1], color="#e6b800", lw=2.0,
                    label=f"safe (eroded {self.floor * 100:.0f}cm)")
        cx, cy = self._proxy_xy
        m = self.floor
        ax.add_patch(plt.Rectangle((cx - m, cy - m), 2 * m, 2 * m, fill=False,
                                   edgecolor="#15b015", lw=2.0, ls="--",
                                   label="torso_l3 QP box"))
        px, py = self._proxy_xy + posture.proxy_err
        ax.plot([px], [py], "o", color="#15b015", ms=9, label="torso_l3 proxy (in box)")
        ax.plot([posture.com[0]], [posture.com[1]], "X", color="#e00", ms=13,
                label=f"true CoM ({posture.margin * 100:+.1f}cm)")
        ax.plot([0], [0], "+", color="0.4", ms=10, mew=1.5)

        ax.set_aspect("equal")
        ax.set_xlabel("base x — forward (m)")
        ax.set_ylabel("base y — left (m)")
        ax.set_title(f"top-down footprint\nCoM margin {posture.margin * 100:.1f}cm "
                     f"< floor {self.floor * 100:.0f}cm  (proxy err≈0)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
        ax.set_xlim(-0.45, 0.35)
        ax.set_ylim(-0.40, 0.40)
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgb = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        plt.close(fig)
        return np.ascontiguousarray(rgb[..., ::-1])  # RGB -> BGR

    def save_png(self, path: str, *, label: str = "") -> None:
        """Composite the SAPIEN perspective (left) and the footprint schematic (right)."""
        if self._cam_persp is None or self._image_size is None:
            raise RuntimeError("no offscreen camera was created")
        import cv2  # noqa: PLC0415 -- optional dep, only needed for PNG output

        self._scene.update_render()
        left = self._grab(self._cam_persp)
        right = self._schematic_bgr(posture=self._last_posture,
                                    width_px=self._image_size[0] - left.shape[1],
                                    height_px=left.shape[0])
        frame = np.ascontiguousarray(np.concatenate([left, right], axis=1))
        if label:
            cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "SAPIEN posture", (12, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.imwrite(path, frame)

    @property
    def viewer_closed(self) -> bool:
        """True once the user has closed the interactive viewer window."""
        return self._viewer is not None and self._viewer.closed

    def close(self) -> None:
        """Close the interactive viewer window, if one is open."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


# --- driver -------------------------------------------------------------------

def _print_table(postures: List[Posture], floor: float) -> None:
    print(f"\nfound {len(postures)} postures (torso_l3 proxy_err == 0, in box; "
          f"true-CoM floor = {floor * 100:.0f}cm):")
    print(f"  {'#':>2}  {'true-CoM margin':>16}  {'CoM base xy (m)':>22}  proxy_err(mm)")
    for i, p in enumerate(postures):
        print(f"  {i:>2}  {p.margin * 100:>13.2f}cm  "
              f"[{p.com[0]:+.3f}, {p.com[1]:+.3f}]{'':>6}  "
              f"[{p.proxy_err[0] * 1e3:+.2e}, {p.proxy_err[1] * 1e3:+.2e}]")
    print("  (every posture: torso_l3 inside its QP box, true CoM below the gate floor.)\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--num", type=int, default=5, help="postures to find/show (default 5).")
    parser.add_argument("--samples", type=int, default=6000,
                        help="random-search budget for the posture search (default 6000).")
    parser.add_argument("--seed", type=int, default=0, help="search RNG seed (default 0).")
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=True,
                        help="open an interactive SAPIEN viewer (default on; needs a display). "
                             "Use --no-viewer for headless PNG output.")
    parser.add_argument("--out", default=None,
                        help="directory to write one annotated PNG per posture (offscreen).")
    parser.add_argument("--dwell", type=float, default=3.0,
                        help="seconds each posture is shown in the viewer cycle (default 3).")
    parser.add_argument("--width", type=int, default=1280, help="PNG width (default 1280).")
    parser.add_argument("--height", type=int, default=720, help="PNG height (default 720).")
    args = parser.parse_args()

    if not args.viewer and not args.out:
        parser.error("--no-viewer needs --out DIR (nothing would be shown otherwise).")

    ik = VegaWholeBodyIK(WBCConfig())  # com_centroid: torso_l3 (the shipped proxy)
    print(f"[viz] com_centroid={ik.config.com_centroid!r}  "
          f"com_safety_margin={ik.config.com_safety_margin}  urdf={Path(ik.config.urdf_path).name}")
    postures = find_postures(ik, args.num, args.seed, args.samples)
    _print_table(postures, float(ik.config.com_safety_margin))

    image_size = (args.width, args.height) if args.out else None
    scene = SafetyScene(ik, with_viewer=args.viewer, image_size=image_size)

    try:
        if args.out:
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            floor_cm = ik.config.com_safety_margin * 100
            for i, p in enumerate(postures):
                scene.show(p)
                path = str(out_dir / f"posture_{i:02d}_margin_{p.margin * 100:.1f}cm.png")
                label = f"posture {i}: CoM {p.margin * 100:.1f}cm < {floor_cm:.0f}cm floor"
                scene.save_png(path, label=label)
                print(f"[viz] wrote {path}")

        if args.viewer:
            import time  # noqa: PLC0415

            print("[viz] viewer: auto-cycling postures (RED CoM dot leaves the YELLOW safety "
                  "boundary; GREEN proxy stays in its box). Close the window to exit.")
            idx = 0
            scene.show(postures[idx])
            last_switch = time.perf_counter()
            while not scene.viewer_closed:
                now = time.perf_counter()
                if now - last_switch >= args.dwell:
                    idx = (idx + 1) % len(postures)
                    scene.show(postures[idx])
                    p = postures[idx]
                    print(f"[viz] posture {idx}: true-CoM margin {p.margin * 100:.2f}cm "
                          f"(floor {ik.config.com_safety_margin * 100:.0f}cm) -- proxy in box",
                          end="\r")
                    last_switch = now
                scene.render_once()
    finally:
        scene.close()
    print("\n[viz] done.")


if __name__ == "__main__":
    main()
