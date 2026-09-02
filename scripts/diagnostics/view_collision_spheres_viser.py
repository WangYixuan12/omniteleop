#!/usr/bin/env python3
"""Inspect the WBC self-collision sphere model against the robot's visual meshes.

The barrier never sees the robot: it sees ``collision_sphere_model``, the Dexmate
sphere URDF plus the fixed proxy spheres :mod:`omniteleop.follower.whole_body_ik`
injects for the Robotiq grippers and the wrist cameras. Whether that stand-in actually
wraps the hardware is a *geometric* question the QP cannot answer, so this script
answers it two ways:

``--fit``
    Headless report. For each distal mesh, the per-vertex signed distance outside the
    union of every sphere -- ``% of vertices outside`` and the worst protrusion, in mm
    -- plus a re-fit of the proxy cover from the mesh itself (evenly spaced centers
    along the principal axis, radius from the assigned vertices). This is where the
    ``WRIST_CAM_PROXY_SPHERES`` radii came from; re-run it after any URDF change.

default
    A viser scene: visual meshes plus every sphere, tinted by the ``wbc_safety``
    collision group that decides which pairs the barrier even considers (body / left /
    right), with the proxy spheres called out. Joint sliders drive Pinocchio FK, and a
    readout tracks the closest *cross-group* pair live, so you can drive the arms into
    a cross-body pose and watch which pair actually goes critical.

Run in the dexmate2 conda env::

    /home/yifan/miniconda3/envs/dexmate2/bin/python \
        scripts/diagnostics/view_collision_spheres_viser.py --fit
    /home/yifan/miniconda3/envs/dexmate2/bin/python \
        scripts/diagnostics/view_collision_spheres_viser.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pinocchio as pin
import trimesh

from omniteleop.follower.wbc_safety import collision_group
from omniteleop.follower.whole_body_ik import (
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    TORSO_JOINTS,
    VegaWholeBodyIK,
)

# Distal meshes that the sphere model has to stand in for, as
# (link frame, mesh path relative to the URDF, mesh scale). The URDF gives all three
# an identity visual origin except the gripper, whose mesh is offset on L/R_robotiq.
PROBE_MESHES = (
    ("L_robotiq", "meshes/collision/robotiq_2f85.obj", 1.0,
     pin.SE3(pin.rpy.rpyToMatrix(1.5707963, 0.0, 0.0), np.array([0.0, 0.0934, -0.2005]))),
    ("L_wrist_zed_mini", "meshes/visual/ZEDM.stl", 0.001, pin.SE3.Identity()),
    ("L_wrist_cam_bracket",
     "meshes/visual/2f85_zed_mini_mount_revised_v3_16.15_fixed.stl", 0.001,
     pin.SE3.Identity()),
)

GROUP_COLORS = {"body": (110, 130, 160), "left": (70, 170, 110), "right": (200, 130, 60)}
PROXY_COLORS = {"body": (110, 130, 160), "left": (40, 220, 120), "right": (255, 150, 40)}


def sphere_world(ik: VegaWholeBodyIK, q: np.ndarray):
    """Sphere names, world centers and radii at configuration ``q``."""
    sm = ik.collision_sphere_model
    sd = ik.collision_sphere_data
    pin.updateGeometryPlacements(ik.model, ik.data, sm, sd, q)
    names = [g.name for g in sm.geometryObjects]
    centers = np.array([sd.oMg[i].translation for i in range(len(names))])
    radii = np.array([g.geometry.radius for g in sm.geometryObjects])
    return names, centers, radii


def probe_vertices(ik: VegaWholeBodyIK, frame: str, rel: str, scale: float,
                   mesh_offset: pin.SE3) -> np.ndarray:
    """Mesh vertices in world coordinates at the current FK state."""
    mesh = trimesh.load(Path(ik.config.urdf_path).parent / rel, force="mesh")
    verts = np.asarray(mesh.vertices, dtype=float) * scale
    oMf = ik.data.oMf[ik.model.getFrameId(frame)] * mesh_offset
    return (oMf.rotation @ verts.T).T + oMf.translation


def fit_cover(verts: np.ndarray, n: int):
    """``n`` centers evenly spaced along the principal axis; radius per assigned vertex."""
    origin = verts.mean(axis=0)
    axis = np.linalg.svd(verts - origin, full_matrices=False)[2][0]
    t = (verts - origin) @ axis
    edges = np.linspace(t.min(), t.max(), n + 1)
    centers = origin + (0.5 * (edges[:-1] + edges[1:]))[:, None] * axis
    dist = np.linalg.norm(verts[:, None, :] - centers[None], axis=2)
    assign = np.argmin(dist, axis=1)
    radii = np.array([dist[assign == k, k].max() if (assign == k).any() else 0.0
                      for k in range(n)])
    return centers, radii


def report_fit(ik: VegaWholeBodyIK, max_spheres: int) -> None:
    """Print sphere coverage of each distal mesh, then a re-fit of its proxy cover."""
    q = ik.nominal_q()
    pin.forwardKinematics(ik.model, ik.data, q)
    pin.updateFramePlacements(ik.model, ik.data)
    names, centers, radii = sphere_world(ik, q)
    print(f"sphere model: {len(names)} spheres "
          f"({sum(1 for n in names if collision_group(n) is None)} ungrouped/ignored)")

    for frame, rel, scale, offset in PROBE_MESHES:
        if not ik.model.existFrame(frame):
            print(f"\n=== {frame}: absent from {Path(ik.config.urdf_path).name}")
            continue
        world = probe_vertices(ik, frame, rel, scale, offset)
        # Signed distance outside the union of every sphere (> 0 == outside all of them).
        outside = np.min(
            np.linalg.norm(world[:, None, :] - centers[None], axis=2) - radii[None],
            axis=1,
        )
        print(f"\n=== {frame} ({len(world)} verts, {Path(rel).name})")
        print(f"  vertices outside every sphere : {100 * np.mean(outside > 0):5.1f}%")
        print(f"  max protrusion                : {outside.max() * 1000:5.1f} mm "
              f"(safe_dist {ik.config.self_collision_safe_dist * 1000:.0f} mm, "
              f"floor {ik.config.self_collision_floor * 1000:.0f} mm)")
        if outside.max() <= ik.config.self_collision_floor:
            print("  -> within the reactive floor; no proxy spheres of its own needed")
            continue
        # Re-fit in the LINK frame, so the numbers drop straight into the proxy tuples.
        oMf = ik.data.oMf[ik.model.getFrameId(frame)]
        local = np.array([oMf.actInv(p) for p in world])
        print(f"  re-fit cover in the {frame} frame:")
        for n in range(1, max_spheres + 1):
            c, r = fit_cover(local, n)
            entries = ", ".join(
                "(({:.3f}, {:.3f}, {:.3f}), {:.3f})".format(*c[k], r[k] + 5e-4)
                for k in range(n)
            )
            print(f"    n={n}: max r={r.max() * 1000:5.1f} mm  {entries}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fit", action="store_true",
                    help="print the headless coverage / cover-fit report and exit")
    ap.add_argument("--max-spheres", type=int, default=3,
                    help="largest cover size to fit per mesh in --fit (default: 3)")
    ap.add_argument("--port", type=int, default=8080, help="viser port (default: 8080)")
    args = ap.parse_args()

    ik = VegaWholeBodyIK()
    if ik.collision_sphere_model is None:
        raise RuntimeError(
            "no collision sphere model was built -- check enable_collision_avoidance "
            "and collision_spheres_urdf in wbik.yaml"
        )
    if args.fit:
        report_fit(ik, args.max_spheres)
        return

    import viser  # noqa: PLC0415 -- only the GUI path needs it

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")

    # VegaWholeBodyIK keeps only the visual *model*; the barrier owns the sphere data.
    visual_data = ik.visual_model.createData()
    joints = TORSO_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + HEAD_JOINTS
    q = ik.nominal_q()
    pin.forwardKinematics(ik.model, ik.data, q)
    pin.updateFramePlacements(ik.model, ik.data)
    pin.updateGeometryPlacements(ik.model, ik.data, ik.visual_model, visual_data, q)

    # Visual meshes: kept translucent so the spheres underneath stay readable.
    mesh_handles = []
    for i, geom in enumerate(ik.visual_model.geometryObjects):
        try:
            mesh = trimesh.load(geom.meshPath, force="mesh")
        except Exception:  # a link with a primitive or an unreadable mesh
            continue
        mesh.apply_scale(np.asarray(geom.meshScale).reshape(3))
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh, vertex_colors=np.tile([210, 210, 215, 90], (len(mesh.vertices), 1))
        )
        mesh_handles.append((server.scene.add_mesh_trimesh(f"/robot/{geom.name}", mesh), i))

    names, centers, radii = sphere_world(ik, q)
    groups = [collision_group(n) for n in names]
    proxy = [n.startswith(("L_robotiq", "R_robotiq", "L_wrist", "R_wrist")) for n in names]
    sphere_handles = [
        server.scene.add_icosphere(
            f"/spheres/{n}",
            radius=float(radii[i]),
            color=(PROXY_COLORS if proxy[i] else GROUP_COLORS)[groups[i] or "body"],
            opacity=0.85 if proxy[i] else 0.45,
            position=centers[i],
            visible=groups[i] is not None,
        )
        for i, n in enumerate(names)
    ]

    # Cross-group pairs are the only ones the barrier constrains, so they are the only
    # ones worth reading out. Taken straight from the model the solver built.
    pairs = [(cp.first, cp.second) for cp in ik.collision_sphere_model.collisionPairs]
    server.gui.add_text("pairs", f"{len(pairs)} cross-group pairs", disabled=True)
    closest = server.gui.add_text("closest pair", "", disabled=True)
    show_meshes = server.gui.add_checkbox("show meshes", True)
    show_body = server.gui.add_checkbox("show body spheres", True)

    sliders = {}
    for label, group in (("torso", TORSO_JOINTS), ("left arm", LEFT_ARM_JOINTS),
                         ("right arm", RIGHT_ARM_JOINTS), ("head", HEAD_JOINTS)):
        with server.gui.add_folder(label, expand_by_default=(label != "head")):
            for name in group:
                idx = ik.model.idx_qs[ik.model.getJointId(name)]
                jid = ik.model.getJointId(name)
                sliders[name] = server.gui.add_slider(
                    name,
                    min=float(ik.model.lowerPositionLimit[idx]),
                    max=float(ik.model.upperPositionLimit[idx]),
                    step=0.01,
                    initial_value=float(q[idx]),
                    hint=f"joint {jid}, q[{idx}]",
                )
    reset = server.gui.add_button("reset to nominal")

    def refresh() -> None:
        cur = ik.nominal_q().copy()
        for name in joints:
            cur[ik.model.idx_qs[ik.model.getJointId(name)]] = sliders[name].value
        pin.forwardKinematics(ik.model, ik.data, cur)
        pin.updateFramePlacements(ik.model, ik.data)
        pin.updateGeometryPlacements(
            ik.model, ik.data, ik.visual_model, visual_data, cur
        )
        for handle, i in mesh_handles:
            placement = visual_data.oMg[i]
            handle.position = placement.translation
            handle.wxyz = pin.Quaternion(placement.rotation).coeffs()[[3, 0, 1, 2]]
        _, c, r = sphere_world(ik, cur)
        for i, handle in enumerate(sphere_handles):
            handle.position = c[i]
        if pairs:
            gaps = np.array([
                np.linalg.norm(c[i] - c[j]) - r[i] - r[j] for i, j in pairs
            ])
            k = int(np.argmin(gaps))
            closest.value = (f"{gaps[k] * 1000:.0f} mm  "
                             f"{names[pairs[k][0]]} <-> {names[pairs[k][1]]}")

    for slider in sliders.values():
        slider.on_update(lambda _: refresh())

    @reset.on_click
    def _(_) -> None:
        nominal = ik.nominal_q()
        for name, slider in sliders.items():
            slider.value = float(nominal[ik.model.idx_qs[ik.model.getJointId(name)]])
        refresh()

    @show_meshes.on_update
    def _(_) -> None:
        for handle, _i in mesh_handles:
            handle.visible = show_meshes.value

    @show_body.on_update
    def _(_) -> None:
        for i, handle in enumerate(sphere_handles):
            if groups[i] == "body":
                handle.visible = show_body.value

    refresh()
    print(f"viser serving on http://localhost:{args.port} -- Ctrl-C to stop")
    server.sleep_forever()


if __name__ == "__main__":
    main()
