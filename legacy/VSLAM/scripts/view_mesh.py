#!/usr/bin/env python3
"""Visualize ZED SDK OBJ meshes (per-vertex RGB) with Open3D.

Run inside the open3d-env conda env:
  conda run -n open3d-env python3 view_mesh.py <path-or-glob> [opts]

Examples:
  # final mesh, interactive rotate/zoom window
  python3 view_mesh.py lab_v1_visual_mesh.obj

  # overlay the camera trajectory (TUM) written alongside the mesh
  python3 view_mesh.py lab_v1_visual_mesh.obj --traj lab_v1.tum

  # also draw a camera coordinate frame every 50th pose
  python3 view_mesh.py lab_v1_visual_mesh.obj --traj lab_v1.tum --traj-frames 50

  # step through the snapshot sequence in one window
  #   N / Right = next, P / Left = previous, Q / Esc = quit
  python3 view_mesh.py 'lab_v1_spatial_snapshots/spatial_map_*.obj'

  # overlay all snapshots in a single view instead of stepping
  python3 view_mesh.py 'lab_v1_spatial_snapshots/spatial_map_*.obj' --overlay
"""
import argparse, glob, os, sys
import numpy as np
import open3d as o3d


def load(path):
    """Load an OBJ, returns a mesh with normals, or None if empty."""
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles) == 0:
        return None
    mesh.compute_vertex_normals()
    return mesh


def _tube_path(pts, radius, resolution=8):
    """A merged cylinder tube following ``pts``, colored green (start) -> red (end).

    Open3D's legacy viewer ignores ``LineSet`` line width on most backends, so a
    1px polyline is the only thing it will draw. Rendering the path as actual
    cylinder geometry is the reliable way to get a thick, depth-shaded trajectory.
    """
    # Drop near-duplicate consecutive points so cylinders aren't degenerate (the
    # camera sits nearly still for many frames); always keep the true endpoints.
    keep = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - keep[-1]) >= max(radius * 0.5, 1e-3):
            keep.append(p)
    if len(keep) < 2:
        keep = [pts[0], pts[-1]]
    keep = np.asarray(keep)
    nseg = len(keep) - 1

    z = np.array([0.0, 0.0, 1.0])
    green, red = np.array([0.1, 0.9, 0.1]), np.array([0.9, 0.1, 0.1])
    tube = o3d.geometry.TriangleMesh()
    for i in range(nseg):
        p0, p1 = keep[i], keep[i + 1]
        v = p1 - p0
        h = float(np.linalg.norm(v))
        if h < 1e-9:
            continue
        cyl = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=h, resolution=resolution)
        # create_cylinder is centered at the origin along +Z; align +Z to v.
        vn = v / h
        axis = np.cross(z, vn)
        s = float(np.linalg.norm(axis))
        c = float(np.dot(z, vn))
        if s < 1e-8:  # parallel or anti-parallel to +Z
            if c < 0:
                cyl.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(
                    [np.pi, 0.0, 0.0]), center=(0, 0, 0))
        else:
            cyl.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(
                axis / s * np.arctan2(s, c)), center=(0, 0, 0))
        cyl.translate((p0 + p1) / 2)
        frac = i / max(nseg - 1, 1)
        cyl.paint_uniform_color((1 - frac) * green + frac * red)
        tube += cyl
    tube.compute_vertex_normals()
    return tube


def load_traj(path, frame_stride=0, frame_size=0.15, radius=0.02):
    """Build Open3D geometries for a ZED TUM trajectory (ts x y z qx qy qz qw).

    The mesh and the TUM file are both written by ``zed_area_map_headless.py`` in
    the same ``RIGHT_HANDED_Y_UP`` frame, so the path overlays the mesh with no
    transform. Returns a list: the path as a ``radius``-thick tube (colored
    start->end by time), start (green) and end (red) markers, and -- when
    ``frame_stride > 0`` -- a camera coordinate frame at every ``frame_stride``-th
    pose to show heading.
    """
    rows = []
    for ln, line in enumerate(open(path).read().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            raise ValueError(
                f"{path}:{ln}: expected 8 TUM columns (ts x y z qx qy qz qw), "
                f"got {len(parts)}: {line!r}")
        rows.append([float(p) for p in parts])
    if not rows:
        raise ValueError(f"{path}: no pose rows found")
    data = np.asarray(rows, dtype=np.float64)
    pts = data[:, 1:4]
    n = len(pts)

    geoms = []
    if n > 1:
        geoms.append(_tube_path(pts, radius))

    for p, color in ((pts[0], [0.1, 0.9, 0.1]), (pts[-1], [0.9, 0.1, 0.1])):
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(radius * 2.5, 0.04))
        marker.translate(p)
        marker.paint_uniform_color(color)
        marker.compute_vertex_normals()
        geoms.append(marker)

    if frame_stride > 0:
        for i in range(0, n, frame_stride):
            qx, qy, qz, qw = data[i, 4:8]
            # Open3D wants quaternion as [w, x, y, z]
            rot = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
            cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
            cf.rotate(rot, center=(0, 0, 0))
            cf.translate(pts[i])
            geoms.append(cf)

    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if n > 1 else 0.0
    print(f"[traj] {path}: {n} poses, path length {length:.2f} m")
    return geoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="path or glob to .obj file(s)")
    ap.add_argument("--overlay", action="store_true",
                    help="draw all matched meshes at once instead of stepping")
    ap.add_argument("--traj", help="overlay a ZED camera trajectory (TUM .tum file)")
    ap.add_argument("--traj-frames", type=int, default=0, metavar="STRIDE",
                    help="draw a camera coordinate frame every STRIDE-th pose "
                         "(0 = path only, no orientation frames)")
    ap.add_argument("--traj-frame-size", type=float, default=0.15,
                    help="size (m) of the per-pose camera coordinate frames")
    ap.add_argument("--traj-radius", type=float, default=0.02,
                    help="trajectory tube radius in meters (thickness of the path)")
    args = ap.parse_args()

    # Resolve and validate paths up front. Exit cleanly here *before* creating
    # any Open3D objects -- on aarch64/Tegra an early exit with live Open3D
    # geometry segfaults during interpreter teardown.
    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(
            f"No .obj matched '{args.pattern}'\n"
            f"  cwd: {os.getcwd()}\n"
            f"  Pass an absolute path or run from the maps dir, e.g.:\n"
            f"    python3 {os.path.basename(sys.argv[0])} "
            f"$LOCAL/maps/lab_v1_visual_mesh.obj")
    extras = []
    extras.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
    if args.traj:
        extras += load_traj(args.traj, args.traj_frames, args.traj_frame_size,
                            args.traj_radius)

    # ---- single mesh or overlay: one static window ----
    if len(paths) == 1 or args.overlay:
        geoms = []
        for p in paths:
            m = load(p)
            if m is None:
                print(f"skip (empty): {p}")
                continue
            geoms.append(m)
        if not geoms:
            sys.exit("nothing to show (all meshes empty)")
        o3d.visualization.draw_geometries(geoms + extras,
                                          window_name=args.pattern,
                                          mesh_show_back_face=True)
        return

    # ---- multiple meshes: step through with keys ----
    meshes = [(p, load(p)) for p in paths]
    meshes = [(p, m) for p, m in meshes if m is not None]
    if not meshes:
        sys.exit("nothing to show (all meshes empty)")

    state = {"i": 0}
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="snapshots: N/Right next, P/Left prev, Q quit")
    for g in extras:
        vis.add_geometry(g)
    vis.add_geometry(meshes[0][1])

    def render():
        i = state["i"]
        p, _ = meshes[i]
        print(f"[{i + 1}/{len(meshes)}] {p}")

    def step(d):
        def cb(v):
            v.remove_geometry(meshes[state["i"]][1], reset_bounding_box=False)
            state["i"] = (state["i"] + d) % len(meshes)
            v.add_geometry(meshes[state["i"]][1], reset_bounding_box=False)
            render()
            return False
        return cb

    # N / Right -> next ; P / Left -> previous
    for key in (ord("N"), 262):
        vis.register_key_callback(key, step(+1))
    for key in (ord("P"), 263):
        vis.register_key_callback(key, step(-1))

    render()
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
