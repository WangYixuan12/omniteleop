from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/misc/vis_robotiq_collision.py"
URDF_PATH = Path(
    "~/yixuan_utilities/src/yixuan_utilities/assets/robot/vega-urdf/"
    "vega_with_robotiq.urdf"
).expanduser()


def _load_module():
    spec = importlib.util.spec_from_file_location("vis_robotiq_collision", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extracts_urdf_visual_and_collision_mesh_entries() -> None:
    module = _load_module()

    entries = module.extract_mesh_entries(URDF_PATH, "L_robotiq")

    assert entries["visual"].mesh_path == URDF_PATH.parent / "meshes/visual/robotiq_2f85.glb"
    assert entries["collision"].mesh_path == URDF_PATH.parent / "meshes/collision/robotiq_2f85.obj"
    assert np.allclose(entries["visual"].scale, [0.001, 0.001, 0.001])
    assert np.allclose(entries["collision"].scale, [1.0, 1.0, 1.0])
    assert np.allclose(entries["visual"].xyz, entries["collision"].xyz)
    assert np.allclose(entries["visual"].rpy, entries["collision"].rpy)


def test_loads_urdf_transformed_meshes_with_comparable_bounds() -> None:
    module = _load_module()
    entries = module.extract_mesh_entries(URDF_PATH, "R_robotiq")

    visual = module.load_urdf_mesh(entries["visual"])
    collision = module.load_urdf_mesh(entries["collision"])

    visual_extent = visual.bounds[1] - visual.bounds[0]
    collision_extent = collision.bounds[1] - collision.bounds[0]

    assert visual.vertices.shape[0] > 0
    assert collision.vertices.shape[0] > 0
    assert visual_extent.max() < 0.3
    assert collision_extent.max() < 0.3
    assert np.linalg.norm(visual.centroid - collision.centroid) < 0.03
