"""Contract tests for scripts/port_wbc_mobile_mof.py (MoF safetensors porter).

Builds tiny synthetic raw takes with the shared `_write_raw` fixture from
test_port_wbc_mobile_hdf5.py, ports them, and asserts the safetensors +
meta.json contract the mofpo WBC dataset consumes: per-entity world-frame
pos/quat obs derived from the SAME state/action assembly as the LeRobot and
ManiFlow porters (cross-checked bit-exactly), 224x224 squashed images, the
29-D world action passed through verbatim (column rot6d), a constant
per-episode env_state, split.csv routing, and the abort-before-write guard
for missing SceneDiff positions.
"""

import importlib.util
import json
import types
from collections.abc import Callable
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pytest
from safetensors.numpy import load_file

from omniteleop.wbc_policy_format import ACTION_AXES, WBCPolicyFK


def _load_by_path(module_name: str, relative: str) -> types.ModuleType:
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mof_porter() -> types.ModuleType:
    return _load_by_path("port_wbc_mobile_mof_test", "scripts/port_wbc_mobile_mof.py")


@pytest.fixture(scope="module")
def hdf5_porter() -> types.ModuleType:
    return _load_by_path("port_wbc_mobile_hdf5_for_mof_test", "scripts/port_wbc_mobile_hdf5.py")


@pytest.fixture(scope="module")
def write_raw() -> Callable[..., Path]:
    module = _load_by_path(
        "test_port_wbc_mobile_hdf5_fixture", "tests/test_port_wbc_mobile_hdf5.py"
    )
    return cast("Callable[..., Path]", module._write_raw)  # noqa: SLF001


@pytest.fixture(scope="module")
def fk() -> WBCPolicyFK:
    return WBCPolicyFK()


def _write_positions(positions_dir: Path, source: str, index: int, seed: int) -> np.ndarray:
    d = positions_dir / source
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.2, 1.2, size=(2, 3)).astype(np.float32)
    np.savez(d / f"episode_{index}.npz", positions=positions, order=np.array([1, 0]), frame="world")
    return positions[[1, 0]].reshape(-1)


@pytest.fixture()
def ported(tmp_path: Path, mof_porter: types.ModuleType, write_raw: Callable[..., Path]) -> dict:
    """Two raw takes (episode_1 -> test split), positions for both, ported."""
    raw_dir = tmp_path / "raw_data"
    raw_dir.mkdir()
    write_raw(raw_dir / "episode_0.hdf5", T=4)
    write_raw(raw_dir / "episode_1.hdf5", T=3)
    (raw_dir / "split.csv").write_text("episode, split\n1, test\n")
    env_states = {
        ("raw", 0): _write_positions(tmp_path / "positions", "raw", 0, seed=10),
        ("raw", 1): _write_positions(tmp_path / "positions", "raw", 1, seed=11),
    }
    out_root = tmp_path / "out"
    args = mof_porter.build_args(
        raw_dir=raw_dir,
        out_root=out_root,
        name="dexmate_wbc_mof",
        positions_dir=tmp_path / "positions",
    )
    mof_porter.convert(args)
    return {"root": out_root / "dexmate_wbc_mof", "raw_dir": raw_dir, "env_states": env_states}


EXPECTED_KEYS = {
    "head_image": ("uint8", (224, 224, 3)),
    "wrist_image": ("uint8", (224, 224, 3)),
    "left_ee_pos": ("float32", (3,)),
    "left_ee_quat": ("float32", (4,)),
    "right_ee_pos": ("float32", (3,)),
    "right_ee_quat": ("float32", (4,)),
    "head_pos": ("float32", (3,)),
    "head_quat": ("float32", (4,)),
    "base_pos": ("float32", (3,)),
    "base_quat": ("float32", (4,)),
    "proprioception_grippers": ("float32", (2,)),
    "action": ("float32", (29,)),
    "env_state": ("float32", (6,)),
}


def test_schema_layout_and_meta(ported: dict) -> None:
    root = ported["root"]
    train = sorted((root / "train").glob("episode_*.safetensors"))
    test = sorted((root / "test").glob("episode_*.safetensors"))
    assert len(train) == 1 and len(test) == 1

    data = load_file(train[0])
    assert set(data) == set(EXPECTED_KEYS)
    T = data["action"].shape[0]
    assert T == 4  # episode_0 kept in full (no leading-NaN grippers)
    for key, (dtype, shape) in EXPECTED_KEYS.items():
        assert data[key].dtype == np.dtype(dtype), key
        assert data[key].shape == (T, *shape), key

    meta = json.loads((root / "meta.json").read_text())
    # meta.json is the authoritative sidecar: its tensor schema must describe
    # the actual files exactly.
    assert meta["tensors"] == {
        k: {"dtype": str(data[k].dtype), "frame_shape": list(data[k].shape[1:])}
        for k in sorted(data)
    }
    assert meta["fps"] == 10
    assert meta["image_resolution"] == [224, 224]
    assert meta["image_resize"] == "anisotropic"
    assert meta["rotation_convention"] == "column"
    assert meta["quaternion_convention"] == "wxyz"
    assert meta["state_frame"] == "world"
    assert meta["action_axes"] == list(ACTION_AXES)
    assert meta["env_state"]["constant_per_episode"] is True
    episodes = meta["episodes"]
    assert {e["split"] for e in episodes} == {"train", "test"}
    src = next(e for e in episodes if e["split"] == "train")
    assert src["source"] == "raw" and src["raw_index"] == 0
    # Source resolution recorded from the data, not assumed.
    assert src["source_image_hw"] == [8, 6]


def test_world_pose_cross_check_bit_exact(
    ported: dict, hdf5_porter: types.ModuleType, fk: WBCPolicyFK
) -> None:
    """Obs entities and action must match the shared state/action assembly."""
    root = ported["root"]
    data = load_file(next((root / "train").glob("*.safetensors")))
    raw_path = ported["raw_dir"] / "episode_0.hdf5"

    with h5py.File(raw_path, "r") as raw:
        for t in range(data["action"].shape[0]):
            eef, head = hdf5_porter.load_action_targets_world(raw, t)
            state, action = hdf5_porter.compute_frame_state_action(
                raw, t, fk, eef, head, state_frame="world"
            )
            assert np.array_equal(data["action"][t], action)
            # positions come verbatim from the world state blocks
            assert np.array_equal(data["left_ee_pos"][t], state[0:3])
            assert np.array_equal(data["right_ee_pos"][t], state[10:13])
            assert np.array_equal(data["head_pos"][t], state[20:23])
            assert np.array_equal(data["proprioception_grippers"][t], state[[9, 19]])
            # quats (wxyz) must reproduce the state's rot6d columns exactly
            for quat_key, rot_sl in (
                ("left_ee_quat", slice(3, 9)),
                ("right_ee_quat", slice(13, 19)),
                ("head_quat", slice(23, 29)),
            ):
                w, x, y, z = data[quat_key][t].astype(np.float64)
                R = np.array(
                    [
                        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    ]
                )
                cols = np.concatenate([R[:, 0], R[:, 1]])
                np.testing.assert_allclose(cols, state[rot_sl], atol=1e-6)
            # planar base lifted to 3-D: [x, y, 0] + yaw-about-z quat (wxyz)
            bx, by, byaw = state[29:32].astype(np.float64)
            np.testing.assert_allclose(data["base_pos"][t], [bx, by, 0.0], atol=1e-7)
            expected_bq = [np.cos(byaw / 2), 0.0, 0.0, np.sin(byaw / 2)]
            np.testing.assert_allclose(data["base_quat"][t], expected_bq, atol=1e-6)


def test_env_state_constant_and_arranged(ported: dict) -> None:
    root = ported["root"]
    for split, key in (("train", ("raw", 0)), ("test", ("raw", 1))):
        data = load_file(next((root / split).glob("*.safetensors")))
        expected = ported["env_states"][key]
        assert np.allclose(
            data["env_state"], expected[None].repeat(data["env_state"].shape[0], axis=0)
        )


def test_missing_positions_aborts_before_write(
    tmp_path: Path, mof_porter: types.ModuleType, write_raw: Callable[..., Path]
) -> None:
    raw_dir = tmp_path / "raw_data"
    raw_dir.mkdir()
    write_raw(raw_dir / "episode_0.hdf5", T=3)
    write_raw(raw_dir / "episode_1.hdf5", T=3)
    _write_positions(tmp_path / "positions", "raw", 0, seed=1)  # episode_1 missing
    out_root = tmp_path / "out"
    args = mof_porter.build_args(
        raw_dir=raw_dir,
        out_root=out_root,
        name="x",
        positions_dir=tmp_path / "positions",
    )
    with pytest.raises(RuntimeError, match="no dataset was written"):
        mof_porter.convert(args)
    assert not (out_root / "x").exists()


def test_image_squash_preserves_content(ported: dict) -> None:
    root = ported["root"]
    data = load_file(next((root / "train").glob("*.safetensors")))
    raw_path = ported["raw_dir"] / "episode_0.hdf5"
    with h5py.File(raw_path, "r") as raw:
        src = raw["obs/images/head_left_rgb"][0].astype(np.float64)
    squashed = data["head_image"][0].astype(np.float64)
    # Anisotropic resize keeps the global appearance: per-channel means match.
    np.testing.assert_allclose(squashed.mean(axis=(0, 1)), src.mean(axis=(0, 1)), atol=6.0)
