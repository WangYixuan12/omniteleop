#!/usr/bin/env python3
"""Visualize the Robotiq meshes exactly as referenced by the Vega URDF.

The Vega URDF uses a millimeter-scale GLB for the Robotiq visual geometry and an OBJ for
collision geometry. This script parses those URDF entries, applies each entry's
``origin`` and ``scale`` without modifying any source asset, and saves an overlay image.

Run with the requested conda env:

    /home/yixuan/miniforge3/envs/dexmate/bin/python \
        scripts/misc/vis_robotiq_collision.py --out /tmp/robotiq_collision.png
"""

from __future__ import annotations

import argparse
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

DEFAULT_URDF = (
    Path("~/yixuan_utilities/src/yixuan_utilities/assets/robot/vega-urdf/"
         "vega_with_robotiq.urdf").expanduser()
)


class MeshEntry:
    """One visual or collision mesh entry from a URDF link."""

    def __init__(
        self,
        *,
        kind: Literal["visual", "collision"],
        link_name: str,
        mesh_path: Path,
        xyz: np.ndarray,
        rpy: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        self.kind = kind
        self.link_name = link_name
        self.mesh_path = mesh_path
        self.xyz = xyz
        self.rpy = rpy
        self.scale = scale


def _parse_vector(text: str | None, *, default: tuple[float, float, float]) -> np.ndarray:
    if text is None:
        return np.array(default, dtype=float)
    values = np.fromstring(text, sep=" ", dtype=float)
    if values.shape != (3,):
        raise ValueError(f"expected 3 vector values, got {text!r}")
    return values


def _resolve_mesh_path(urdf_path: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        raise ValueError(
            "package:// mesh paths are not supported by this standalone script: "
            f"{filename}"
        )
    if filename.startswith("file://"):
        return Path(filename.removeprefix("file://")).expanduser().resolve()
    path = Path(filename).expanduser()
    if not path.is_absolute():
        path = urdf_path.parent / path
    return path.resolve()


def extract_mesh_entries(urdf_path: str | Path, link_name: str) -> dict[str, MeshEntry]:
    """Extract the visual and collision mesh entries for a URDF link."""
    urdf_path = Path(urdf_path).expanduser().resolve()
    root = ET.parse(urdf_path).getroot()
    link = root.find(f"./link[@name='{link_name}']")
    if link is None:
        available = [node.attrib.get("name", "") for node in root.findall("./link")]
        raise ValueError(
            f"link {link_name!r} not found in {urdf_path}; available links: {available}"
        )

    entries: dict[str, MeshEntry] = {}
    for kind in ("visual", "collision"):
        node = link.find(kind)
        if node is None:
            raise ValueError(f"link {link_name!r} has no <{kind}> entry")
        origin = node.find("origin")
        geometry = node.find("geometry")
        mesh = geometry.find("mesh") if geometry is not None else None
        if mesh is None or "filename" not in mesh.attrib:
            raise ValueError(f"link {link_name!r} <{kind}> has no mesh filename")
        entries[kind] = MeshEntry(
            kind=kind,  # type: ignore[arg-type]
            link_name=link_name,
            mesh_path=_resolve_mesh_path(urdf_path, mesh.attrib["filename"]),
            xyz=_parse_vector(
                None if origin is None else origin.attrib.get("xyz"),
                default=(0, 0, 0),
            ),
            rpy=_parse_vector(
                None if origin is None else origin.attrib.get("rpy"),
                default=(0, 0, 0),
            ),
            scale=_parse_vector(mesh.attrib.get("scale"), default=(1, 1, 1)),
        )
    return entries


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """Return URDF fixed-axis RPY rotation matrix."""
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def origin_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    """Build a homogeneous transform from a URDF origin."""
    transform = np.eye(4)
    transform[:3, :3] = rpy_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def _load_as_mesh(mesh_path: Path):
    import trimesh  # noqa: PLC0415

    loaded = trimesh.load(mesh_path, force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif hasattr(loaded, "to_geometry"):
        mesh = loaded.to_geometry()
    else:
        mesh = loaded.dump(concatenate=True)
    if mesh.vertices.size == 0:
        raise ValueError(f"mesh has no vertices: {mesh_path}")
    return mesh.copy()


def load_urdf_mesh(entry: MeshEntry):
    """Load a mesh and apply its URDF mesh scale and origin transform."""
    mesh = _load_as_mesh(entry.mesh_path)
    mesh.apply_scale(entry.scale)
    mesh.apply_transform(origin_transform(entry.xyz, entry.rpy))
    return mesh


def _set_equal_axes(ax, points: np.ndarray) -> None:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    mid = (lo + hi) / 2.0
    span = max(float((hi - lo).max()), 1e-6) * 1.08
    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
    ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)
    ax.set_box_aspect((1, 1, 1))


def _scatter_mesh(ax, mesh, *, color: str, label: str, stride: int, alpha: float) -> None:
    vertices = np.asarray(mesh.vertices)
    if len(vertices) > stride:
        vertices = vertices[:: max(1, len(vertices) // stride)]
    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        s=2.0,
        c=color,
        alpha=alpha,
        label=label,
    )


def save_overlay(
    *,
    visual_mesh,
    collision_mesh,
    entries: dict[str, MeshEntry],
    out: str | Path,
    point_budget: int = 6000,
    interactive: bool = False,
) -> None:
    """Save a multi-view overlay of visual GLB and collision OBJ meshes."""
    import matplotlib  # noqa: PLC0415

    if not interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    points = np.vstack([visual_mesh.vertices, collision_mesh.vertices])
    views = [(20, -60, "iso"), (90, -90, "top"), (0, -90, "front"), (0, 0, "side")]
    fig = plt.figure(figsize=(14, 7))
    for idx, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 4, idx, projection="3d")
        _scatter_mesh(
            ax,
            visual_mesh,
            color="tab:blue",
            label="visual GLB",
            stride=point_budget,
            alpha=0.5,
        )
        _scatter_mesh(
            ax,
            collision_mesh,
            color="tab:orange",
            label="collision OBJ",
            stride=point_budget,
            alpha=0.45,
        )
        _set_equal_axes(ax, points)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.tick_params(labelsize=7)
        if idx == 1:
            ax.legend(loc="upper left", fontsize=8)

    v_entry = entries["visual"]
    c_entry = entries["collision"]
    fig.suptitle(
        "\n".join(
            [
                f"{v_entry.link_name}: URDF-applied Robotiq visual/collision mesh overlay",
                f"visual: {v_entry.mesh_path.name} scale={v_entry.scale.tolist()} | "
                f"collision: {c_entry.mesh_path.name} scale={c_entry.scale.tolist()}",
            ]
        ),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=150)
    if interactive:
        plt.show()


def _bounds_report(name: str, mesh) -> str:
    lo, hi = mesh.bounds
    extent = hi - lo
    return (
        f"{name}: vertices={len(mesh.vertices)} faces={len(mesh.faces)} "
        f"bounds_min={np.array2string(lo, precision=4)} "
        f"bounds_max={np.array2string(hi, precision=4)} "
        f"extent={np.array2string(extent, precision=4)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help="URDF containing the Robotiq mesh refs.",
    )
    parser.add_argument("--link", default="L_robotiq", choices=("L_robotiq", "R_robotiq"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/robotiq_urdf_mesh_overlay.png"))
    parser.add_argument(
        "--point-budget",
        type=int,
        default=6000,
        help="max plotted points per mesh per view.",
    )
    parser.add_argument("--interactive", action="store_true", help="also show a matplotlib window.")
    args = parser.parse_args()

    entries = extract_mesh_entries(args.urdf, args.link)
    visual_mesh = load_urdf_mesh(entries["visual"])
    collision_mesh = load_urdf_mesh(entries["collision"])

    print(f"URDF: {Path(args.urdf).resolve()}")
    for kind, entry in entries.items():
        print(
            f"{kind}: path={entry.mesh_path} xyz={entry.xyz.tolist()} "
            f"rpy={entry.rpy.tolist()} scale={entry.scale.tolist()}"
        )
    print(_bounds_report("visual", visual_mesh))
    print(_bounds_report("collision", collision_mesh))
    print(f"centroid_delta_m={np.linalg.norm(visual_mesh.centroid - collision_mesh.centroid):.6f}")

    save_overlay(
        visual_mesh=visual_mesh,
        collision_mesh=collision_mesh,
        entries=entries,
        out=args.out,
        point_budget=args.point_budget,
        interactive=args.interactive,
    )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
