# ruff: noqa: SLF001
"""Unit tests for scripts/port_wbc_mobile_hdf5.py (pure conversion helpers).

Builds tiny synthetic raw HDF5 takes in the wbc_vr_robot.py --record schema and
checks the 32-D state / 29-D action assembly, the verbatim world-frame action
policy, the required-dataset errors, and the NaN-gripper policy. Debug HDF5 is
deliberately NOT an input to the porter, so none is synthesized here.

Needs the dexmate env (pinocchio/pink for WBCPolicyFK); lerobot is NOT required
(the porter imports it lazily, only for the actual dataset write).
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

from omniteleop.wbc_policy_format import (
    ACTION_AXES,
    JOYSTICK_ACTION_AXES,
    JOYSTICK_POLICY_ACTION_SCHEMA,
    STATE_AXES,
    WBCPolicyFK,
    base_pose_to_mat,
    mat_to_pos6d,
)


def _load_porter():
    path = Path(__file__).resolve().parents[1] / "scripts" / "port_wbc_mobile_hdf5.py"
    spec = importlib.util.spec_from_file_location("port_wbc_mobile_hdf5_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def porter():
    return _load_porter()


@pytest.fixture(scope="module")
def fk():
    return WBCPolicyFK()  # loads the WBC model once for the whole module


def _pose(tx, ty, tz, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    m[:3, 3] = [tx, ty, tz]
    return m


def _write_raw(path: Path, T: int = 3, *, gripper_left=None, timing: bool = False,
               depth_timing: bool = False, camera_ntp: bool = False,
               camera_alignment: bool = False,
               causal_timing: bool = False,
               depth: bool = True,
               base_pose_source: str | None = None,
               drop_keys: tuple[str, ...] = ()) -> Path:
    """A minimal wbc_vr_robot-schema raw take (images tiny, 10 Hz timestamps).

    ``timing=True`` adds the timing-aware recorder fields: per-frame capture
    stamps under ``obs/images`` and the once-per-run ``meta/ntp`` calibration
    (negative offset on purpose -- the robot clock may trail the local one).
    ``depth_timing=True`` adds the optional head-depth capture stamp introduced
    after RGB and depth were allowed to differ by one camera frame.
    ``causal_timing=True`` adds the action/state causal bracket under ``timing``.
    """
    rng = np.random.default_rng(0)
    grip_l = np.full(T, 0.25, np.float32) if gripper_left is None else np.asarray(
        gripper_left, np.float32)
    with h5py.File(path, "w") as f:
        f.attrs["complete"] = True
        f.attrs["n_frames"] = T
        f["obs/joint/torso"] = np.tile([0.78, 1.5708, 0.0], (T, 1)).astype(np.float32)
        f["obs/joint/left_arm"] = np.tile(
            [0.844, 0.3, 0.0, -1.556, 0.0, 1.271, 0.0], (T, 1)).astype(np.float32)
        f["obs/joint/right_arm"] = np.tile(
            [-0.844, -0.3, 0.0, -1.556, 0.0, -1.271, 0.0], (T, 1)).astype(np.float32)
        f["obs/joint/head"] = np.zeros((T, 3), np.float32)
        for ax in ("vx", "vy", "wz"):
            f[f"obs/joint/chassis_{ax}"] = np.zeros(T, np.float32)
            f[f"action/joint/chassis_{ax}"] = np.zeros(T, np.float32)
        base_pose = np.stack(
            [np.linspace(0, 0.2, T), np.zeros(T), np.full(T, 0.1)], axis=1
        ).astype(np.float32)
        f["obs/base/pose"] = base_pose
        if base_pose_source is not None:
            if base_pose_source not in {"wheel_odometry", "arkit"}:
                raise ValueError(base_pose_source)
            f["meta/schema"] = np.asarray(b"omniteleop_wbc_mobile_raw/v3")
            f["meta/provenance/schema"] = np.asarray(
                b"omniteleop_wbc_mobile_raw/v3"
            )
            f["meta/obs_base_pose_source"] = np.asarray(base_pose_source.encode())
            f["meta/base_control_pose_source"] = np.asarray(base_pose_source.encode())
            f["obs/base/pose_odom"] = base_pose
            if base_pose_source == "arkit":
                arkit_pose = base_pose.copy()
                arkit_pose[:, 0] += np.float32(0.5)
                f["obs/base/pose_arkit"] = arkit_pose
                f["obs/base/pose"][:] = arkit_pose
        f["obs/gripper/left"] = grip_l
        f["obs/gripper/right"] = np.full(T, 0.75, np.float32)
        f["action/gripper/left"] = np.full(T, 0.9, np.float32)   # binarizes to 1.0
        f["action/gripper/right"] = np.full(T, 0.2, np.float32)  # binarizes to 0.0
        f["action/joint/torso"] = np.tile([0.78, 1.5708, 0.0], (T, 1)).astype(np.float32)
        f["action/joint/left_arm"] = np.zeros((T, 7), np.float32)
        f["action/joint/right_arm"] = np.zeros((T, 7), np.float32)
        f["action/joint/head"] = np.zeros((T, 3), np.float32)
        f["action/eef/left"] = np.stack(
            [_pose(0.5, 0.3, 0.9, yaw=0.2 + 0.01 * t) for t in range(T)])
        f["action/eef/right"] = np.stack(
            [_pose(0.5, -0.3, 0.9, yaw=-0.2 - 0.01 * t) for t in range(T)])
        f["action/head"] = np.stack(
            [_pose(0.1, 0.0, 1.4, yaw=0.05 * t) for t in range(T)])
        f["action/base/pose"] = np.zeros((T, 3), np.float32)
        f["obs/images/head_left_rgb"] = rng.integers(
            0, 255, (T, 8, 6, 3), dtype=np.uint8)
        if depth:
            f["obs/images/head_depth"] = rng.integers(
                0, 4000, (T, 8, 6), dtype=np.uint16)
        f["obs/images/left_wrist_rgb"] = rng.integers(
            0, 255, (T, 8, 6, 3), dtype=np.uint8)
        f["obs/images/intrinsic"] = np.array(
            [[100.0, 0, 3.0], [0, 100.0, 4.0], [0, 0, 1]], np.float32)
        f["timestamp_ns"] = (1_000_000_000 + np.arange(T) * 100_000_000).astype(np.int64)
        if timing:
            steps = np.arange(T, dtype=np.int64) * 100_000_000
            f["obs/images/head_frame_ns"] = 900_000_000 + steps
            if depth_timing:
                if not depth:
                    raise ValueError("depth_timing synthetic data requires depth=True")
                f["obs/images/head_depth_frame_ns"] = 895_000_000 + steps
            f["obs/images/left_wrist_frame_ns"] = 890_000_000 + steps
            f["obs/images/grab_wall_ns"] = 998_000_000 + steps
            f["meta/ntp/offset_ns"] = np.int64(-12_000_000)
            f["meta/ntp/rtt_ns"] = np.int64(3_000_000)
            f["meta/ntp/queried_at_ns"] = np.int64(950_000_000)
        if camera_ntp:
            if not timing:
                raise ValueError("camera_ntp synthetic data requires timing=True")
            f["meta/camera_ntp/head/offset_ns"] = np.int64(5_000_000)
            f["meta/camera_ntp/head/rtt_ns"] = np.int64(2_000_000)
            f["meta/camera_ntp/head/queried_at_ns"] = np.int64(960_000_000)
            f["meta/camera_ntp/head/sensor_id"] = np.asarray(b"head_camera")
            f["meta/camera_ntp/head/source"] = np.asarray(b"zed_publisher_host")
            f["meta/camera_ntp/wrist/offset_ns"] = np.int64(7_000_000)
            f["meta/camera_ntp/wrist/rtt_ns"] = np.int64(2_500_000)
            f["meta/camera_ntp/wrist/queried_at_ns"] = np.int64(961_000_000)
            f["meta/camera_ntp/wrist/sensor_id"] = np.asarray(b"wrist_zedm")
            f["meta/camera_ntp/wrist/source"] = np.asarray(b"zed_publisher_host")
        if camera_alignment:
            if not (timing and camera_ntp):
                raise ValueError(
                    "camera_alignment synthetic data requires timing=True, camera_ntp=True"
                )
            # Alignment is the raw-v4 contract. Supply the explicit wheel source when
            # this synthetic caller did not already request an ARKit/wheel declaration.
            for schema_key in ("meta/schema", "meta/provenance/schema"):
                if schema_key in f:
                    f[schema_key][...] = np.asarray(b"omniteleop_wbc_mobile_raw/v4")
                else:
                    f[schema_key] = np.asarray(b"omniteleop_wbc_mobile_raw/v4")
            if "meta/obs_base_pose_source" not in f:
                f["meta/obs_base_pose_source"] = np.asarray(b"wheel_odometry")
                f["meta/base_control_pose_source"] = np.asarray(b"wheel_odometry")
                f["obs/base/pose_odom"] = base_pose
            f["meta/camera_alignment/mode"] = np.asarray(b"head_capture_nearest")
            f["meta/camera_alignment/anchor"] = np.asarray(
                b"head_left_rgb_capture"
            )
            f["meta/camera_alignment/clock_domain"] = np.asarray(
                b"local_via_camera_ntp"
            )
            f["meta/camera_alignment/max_abs_skew_ns"] = np.int64(40_000_000)
            f["meta/camera_alignment/max_camera_age_ns"] = np.int64(150_000_000)
            f["meta/camera_alignment/wrist_buffer_frames"] = np.int64(16)
            steps = np.arange(T, dtype=np.int64) * 100_000_000
            for side in ("left", "right"):
                group = f"obs/gripper_status/{side}"
                f[f"{group}/status_request_id"] = np.arange(T, dtype=np.int64)
                f[f"{group}/request_send_wall_ns"] = 990_000_000 + steps
                f[f"{group}/read_wall_ns"] = 995_000_000 + steps
                f[f"{group}/actual"] = f[f"obs/gripper/{side}"][...]
        if causal_timing:
            if not timing:
                raise ValueError("causal_timing synthetic data requires timing=True")
            steps = np.arange(T, dtype=np.int64) * 100_000_000
            f["timing/source_timestamp_ns"] = 970_000_000 + steps
            f["timing/source_receive_wall_ns"] = 980_000_000 + steps
            f["timing/action_dispatch_start_wall_ns"] = 985_000_000 + steps
            f["timing/action_dispatch_end_wall_ns"] = 990_000_000 + steps
            f["timing/state_read_start_wall_ns"] = 999_000_000 + steps
            f["timing/state_read_end_wall_ns"] = 999_500_000 + steps
        for key in drop_keys:
            del f[key]
    return path


def test_compute_frame_state_action_layout_and_values(porter, fk, tmp_path):
    path = _write_raw(tmp_path / "episode_0.hdf5")
    with h5py.File(path, "r") as raw:
        t = 1
        action_eef, action_head = porter.load_action_targets_world(raw, t)
        state, action = porter.compute_frame_state_action(raw, t, fk, action_eef, action_head)

        assert state.shape == (len(STATE_AXES),) == (32,)
        assert action.shape == (len(ACTION_AXES),) == (29,)
        assert state.dtype == np.float32 and action.dtype == np.float32

        # Action blocks are the raw world-frame matrices VERBATIM (no composition).
        np.testing.assert_allclose(
            action[:9], mat_to_pos6d(raw["action/eef/left"][t]), atol=1e-7)
        np.testing.assert_allclose(
            action[10:19], mat_to_pos6d(raw["action/eef/right"][t]), atol=1e-7)
        np.testing.assert_allclose(
            action[20:29], mat_to_pos6d(raw["action/head"][t]), atol=1e-7)
        # Grippers: action binarized at 0.5, state raw achieved.
        assert action[9] == 1.0 and action[19] == 0.0
        assert state[9] == np.float32(0.25) and state[19] == np.float32(0.75)

        # State head block == FK of the measured obs joints (base zero) on the
        # WBC's own model; base dims == obs/base/pose.
        poses = fk.base_frame_poses(
            raw["obs/joint/torso"][t], raw["obs/joint/left_arm"][t],
            raw["obs/joint/right_arm"][t], raw["obs/joint/head"][t])
        np.testing.assert_allclose(state[20:29], mat_to_pos6d(poses["head"]), atol=1e-6)
        np.testing.assert_allclose(state[:9], mat_to_pos6d(poses["left"]), atol=1e-6)
        np.testing.assert_allclose(state[29:32], raw["obs/base/pose"][t], atol=1e-7)


def test_joystick_policy_contract_appends_latched_chassis_intent(porter, fk, tmp_path):
    path = _write_raw(
        tmp_path / "episode_0.hdf5", base_pose_source="wheel_odometry"
    )
    intent = np.array(
        [[0.2, 0.0, 0.0], [0.0, -0.15, 0.0], [0.0, 0.0, 0.3]],
        dtype=np.float32,
    )
    with h5py.File(path, "a") as raw:
        for key in ("meta/schema", "meta/provenance/schema"):
            del raw[key]
            raw[key] = np.asarray(b"omniteleop_joystick_mobile_raw/v1")
        raw["meta/control_mode"] = np.asarray(b"joystick")
        raw["meta/action_target_frame"] = np.asarray(b"current_base")
        raw["meta/eef_target_frame"] = np.asarray(b"current_base")
        raw["meta/head_target_frame"] = np.asarray(b"current_base")
        raw["meta/policy_action_schema"] = np.asarray(
            JOYSTICK_POLICY_ACTION_SCHEMA.encode()
        )
        raw["action/chassis/intent_body"] = intent

    contract = porter.validate_work_policy_action_contract(
        [{"path": path}]
    )
    assert contract["policy_action_schema"] == JOYSTICK_POLICY_ACTION_SCHEMA
    assert contract["action_axes"] == JOYSTICK_ACTION_AXES
    with h5py.File(path, "r") as raw:
        t = 1
        action_eef, action_head = porter.load_action_targets(raw, t)
        state, action = porter.compute_frame_state_action(
            raw,
            t,
            fk,
            action_eef,
            action_head,
            policy_action_schema=JOYSTICK_POLICY_ACTION_SCHEMA,
        )
    assert state.shape == (32,)
    assert action.shape == (32,)
    np.testing.assert_allclose(action[-3:], intent[t])


def test_policy_contract_rejects_mixed_wbc_and_joystick_takes(porter, tmp_path):
    wbc = _write_raw(tmp_path / "episode_0.hdf5")
    joystick = _write_raw(
        tmp_path / "episode_1.hdf5", base_pose_source="wheel_odometry"
    )
    with h5py.File(joystick, "a") as raw:
        for key in ("meta/schema", "meta/provenance/schema"):
            del raw[key]
            raw[key] = np.asarray(b"omniteleop_joystick_mobile_raw/v1")
        raw["meta/control_mode"] = np.asarray(b"joystick")
        raw["meta/action_target_frame"] = np.asarray(b"current_base")
        raw["meta/eef_target_frame"] = np.asarray(b"current_base")
        raw["meta/head_target_frame"] = np.asarray(b"current_base")
        raw["meta/policy_action_schema"] = np.asarray(
            JOYSTICK_POLICY_ACTION_SCHEMA.encode()
        )
        raw["action/chassis/intent_body"] = np.zeros((3, 3), np.float32)

    with pytest.raises(RuntimeError, match="mix policy action contracts"):
        porter.validate_work_policy_action_contract(
            [{"path": wbc}, {"path": joystick}]
        )


def test_policy_loader_rejects_incomplete_and_one_frame_takes(porter, tmp_path):
    incomplete = _write_raw(tmp_path / "episode_0.hdf5")
    with h5py.File(incomplete, "r+") as raw:
        raw.attrs["complete"] = False
    with pytest.raises(RuntimeError, match="complete flag is missing/false"):
        porter.load_and_validate_episode(incomplete, fps=10)

    one_frame = _write_raw(tmp_path / "episode_1.hdf5", T=1)
    with pytest.raises(RuntimeError, match="only 1 frame"):
        porter.load_and_validate_episode(one_frame, fps=10)


def test_policy_loader_rejects_declared_frame_count_mismatch(porter, tmp_path):
    path = _write_raw(tmp_path / "episode_0.hdf5", T=3)
    with h5py.File(path, "r+") as raw:
        raw.attrs["n_frames"] = 99

    with pytest.raises(RuntimeError, match="n_frames=99 disagrees"):
        porter.load_and_validate_episode(path, fps=10)


def test_required_latency_audit_runs_for_every_work_item(porter, tmp_path, capsys):
    paths = [
        _write_raw(tmp_path / f"episode_{index}.hdf5", timing=True)
        for index in (0, 1)
    ]
    work = [
        {"source": "raw", "path": path, "raw_index": index, "trim": None}
        for index, path in enumerate(paths)
    ]

    results = porter.audit_work_list(work)

    assert set(results) == set(paths)
    output = capsys.readouterr().out
    assert "Running required episode-latency audit (2 episode(s))" in output
    assert all(str(path) in output for path in paths)


def test_required_latency_audit_fails_on_backward_stamp(porter, tmp_path):
    path = _write_raw(tmp_path / "episode_0.hdf5", timing=True)
    with h5py.File(path, "r+") as raw:
        stamps = np.asarray(raw["obs/images/head_frame_ns"])
        stamps[2] = stamps[1] - 1
        raw["obs/images/head_frame_ns"][:] = stamps
    work = [{"source": "raw", "path": path, "raw_index": 0, "trim": None}]

    with pytest.raises(RuntimeError, match="Required episode-latency audit failed"):
        porter.audit_work_list(work)


def test_compute_frame_state_action_world_frame_composes_base_pose(porter, fk, tmp_path):
    path = _write_raw(tmp_path / "episode_0.hdf5")
    with h5py.File(path, "r") as raw:
        t = 1
        base_pose = np.asarray(raw["obs/base/pose"][t], dtype=np.float64)
        assert not np.allclose(base_pose, 0.0)  # non-trivial base so world != base
        action_eef, action_head = porter.load_action_targets_world(raw, t)
        s_base, a_base = porter.compute_frame_state_action(
            raw, t, fk, action_eef, action_head, state_frame="base")
        s_world, a_world = porter.compute_frame_state_action(
            raw, t, fk, action_eef, action_head, state_frame="world")

        # Action is world-verbatim regardless of state_frame.
        np.testing.assert_array_equal(a_base, a_world)
        # World state EEF/head == world_T_base @ base_T_eef (from the base-frame FK).
        poses = fk.base_frame_poses(
            raw["obs/joint/torso"][t], raw["obs/joint/left_arm"][t],
            raw["obs/joint/right_arm"][t], raw["obs/joint/head"][t])
        world_T_base = base_pose_to_mat(base_pose)
        for name, sl in (("left", slice(0, 9)), ("right", slice(10, 19)),
                         ("head", slice(20, 29))):
            np.testing.assert_allclose(
                s_world[sl], mat_to_pos6d(world_T_base @ poses[name]), atol=1e-6)
        # Grippers + base pose dims unchanged between frames.
        assert s_world[9] == s_base[9] and s_world[19] == s_base[19]
        np.testing.assert_allclose(s_world[29:32], s_base[29:32], atol=1e-7)
        # The world head block IS the camera extrinsic world_T_zed.
        np.testing.assert_allclose(
            s_world[20:29], mat_to_pos6d(world_T_base @ poses["head"]), atol=1e-6)


@pytest.mark.parametrize("missing", ["action/eef/left", "action/eef/right", "action/head"])
def test_load_action_targets_world_requires_datasets(porter, tmp_path, missing):
    path = _write_raw(tmp_path / "episode_0.hdf5", drop_keys=(missing,))
    with h5py.File(path, "r") as raw, pytest.raises(RuntimeError, match=missing):
        porter.load_action_targets_world(raw, 0)


def test_load_action_targets_world_rejects_bad_index_and_nonfinite(porter, tmp_path):
    path = _write_raw(tmp_path / "episode_0.hdf5")
    with h5py.File(path, "r") as raw, pytest.raises(RuntimeError, match="out of range"):
        porter.load_action_targets_world(raw, 99)
    with h5py.File(path, "r+") as raw:
        bad = np.asarray(raw["action/head"][0])
        bad[0, 3] = np.nan
        raw["action/head"][0] = bad
    with h5py.File(path, "r") as raw, pytest.raises(RuntimeError, match="non-finite"):
        porter.load_action_targets_world(raw, 0)


def test_action_offset_shifts_pose_and_gripper_as_one_label(porter, fk, tmp_path):
    path = _write_raw(tmp_path / "episode_0.hdf5", T=3)
    with h5py.File(path, "r+") as raw:
        raw["action/gripper/left"][:] = [0.1, 0.9, 0.1]
        raw["action/gripper/right"][:] = [0.9, 0.1, 0.9]
    with h5py.File(path, "r") as raw:
        eef, head = porter.load_action_targets_world(
            raw, 0, pose_action_offset=1
        )
        np.testing.assert_array_equal(eef["left"], raw["action/eef/left"][1])
        np.testing.assert_array_equal(eef["right"], raw["action/eef/right"][1])
        np.testing.assert_array_equal(head, raw["action/head"][1])
        _state, action = porter.compute_frame_state_action(
            raw, 0, fk, eef, head, action_t=1
        )
        assert action[9] == 1.0
        assert action[19] == 0.0

        terminal_eef, terminal_head = porter.load_action_targets_world(
            raw, 2, pose_action_offset=1
        )
        np.testing.assert_array_equal(
            terminal_eef["left"], raw["action/eef/left"][2]
        )
        np.testing.assert_array_equal(terminal_head, raw["action/head"][2])
        _state, terminal_action = porter.compute_frame_state_action(
            raw, 2, fk, terminal_eef, terminal_head, action_t=2
        )
        assert terminal_action[9] == 0.0
        assert terminal_action[19] == 1.0

    with h5py.File(path, "r") as raw, pytest.raises(
        ValueError, match="pose_action_offset"
    ):
        porter.load_action_targets_world(raw, 0, pose_action_offset=-1)


def test_episode_requires_obs_base_pose(porter, tmp_path):
    path = _write_raw(tmp_path / "episode_0.hdf5", drop_keys=("obs/base/pose",))
    with pytest.raises(RuntimeError, match="obs/base/pose"):
        porter.load_and_validate_episode(path, fps=10)


def test_base_pose_contract_accepts_legacy_wheel_and_v3_sources(porter, tmp_path):
    legacy = _write_raw(tmp_path / "episode_0.hdf5")
    legacy_contract = porter.inspect_episode_base_pose_contract(legacy)
    assert legacy_contract["base_pose_source"] == "wheel_odometry"
    assert legacy_contract["legacy_inferred"] is True
    assert porter.load_and_validate_episode(legacy, fps=10)["base_pose_source"] == (
        "wheel_odometry"
    )

    wheel = _write_raw(
        tmp_path / "episode_1.hdf5", base_pose_source="wheel_odometry"
    )
    wheel_contract = porter.inspect_episode_base_pose_contract(wheel)
    assert wheel_contract["raw_schema"] == "omniteleop_wbc_mobile_raw/v3"
    assert wheel_contract["base_pose_source"] == "wheel_odometry"
    assert wheel_contract["legacy_inferred"] is False

    arkit = _write_raw(tmp_path / "episode_2.hdf5", base_pose_source="arkit")
    arkit_contract = porter.inspect_episode_base_pose_contract(arkit)
    assert arkit_contract["base_pose_source"] == "arkit"
    with h5py.File(arkit, "r") as raw:
        np.testing.assert_array_equal(raw["obs/base/pose"], raw["obs/base/pose_arkit"])
        assert not np.array_equal(raw["obs/base/pose"], raw["obs/base/pose_odom"])


def test_base_pose_contract_rejects_mislabeled_v3_episode(porter, tmp_path):
    mismatch = _write_raw(
        tmp_path / "episode_0.hdf5", base_pose_source="arkit"
    )
    with h5py.File(mismatch, "r+") as raw:
        raw["obs/base/pose"][1, 0] += np.float32(0.01)
    with pytest.raises(RuntimeError, match="does not exactly match declared arkit"):
        porter.inspect_episode_base_pose_contract(mismatch)

    missing_odom = _write_raw(
        tmp_path / "episode_1.hdf5", base_pose_source="wheel_odometry"
    )
    with h5py.File(missing_odom, "r+") as raw:
        del raw["obs/base/pose_odom"]
    with pytest.raises(RuntimeError, match="requires diagnostic obs/base/pose_odom"):
        porter.inspect_episode_base_pose_contract(missing_odom)


def test_work_base_pose_contract_rejects_mixed_sources(porter, tmp_path):
    wheel = _write_raw(
        tmp_path / "episode_0.hdf5", base_pose_source="wheel_odometry"
    )
    arkit = _write_raw(tmp_path / "episode_1.hdf5", base_pose_source="arkit")
    work = [
        {"source": "raw", "path": wheel, "raw_index": 0, "trim": None},
        {"source": "raw", "path": arkit, "raw_index": 1, "trim": None},
    ]
    with pytest.raises(RuntimeError, match="mix canonical base-pose sources"):
        porter.validate_work_base_pose_contract(work)


def test_processed_meta_records_base_pose_source(porter, tmp_path):
    camera_alignment_contract = {
        "mode": "head_capture_nearest",
        "clock_domain": "local_via_camera_ntp",
        "max_abs_skew_ns": 40_000_000,
        "max_camera_age_ns": 150_000_000,
        "verified": True,
    }
    porter._write_meta(
        tmp_path,
        "train",
        "base",
        10,
        120,
        160,
        "test task",
        "arkit",
        camera_alignment_contract,
    )
    import json

    meta = json.loads((tmp_path / "dexmate_meta.json").read_text())
    assert meta["base_pose_source"] == "arkit"
    assert "observation.state[29:32]" in meta["base_pose_semantics"]
    assert meta["camera_alignment_mode"] == "head_capture_nearest"
    assert meta["camera_alignment_clock_domain"] == "local_via_camera_ntp"
    assert meta["camera_alignment_max_abs_skew_ns"] == 40_000_000
    assert meta["camera_alignment_max_camera_age_ns"] == 150_000_000
    assert meta["camera_alignment_verified"] is True


def test_camera_alignment_contract_verifies_corrected_capture_skew(
    porter, tmp_path
):
    path = _write_raw(
        tmp_path / "episode_0.hdf5",
        timing=True,
        camera_ntp=True,
        camera_alignment=True,
    )

    contract = porter.inspect_episode_camera_alignment_contract(path)
    assert contract["raw_schema"] == "omniteleop_wbc_mobile_raw/v4"
    assert contract["mode"] == "head_capture_nearest"
    assert contract["clock_domain"] == "local_via_camera_ntp"
    assert contract["max_abs_skew_ns"] == 40_000_000
    assert contract["max_camera_age_ns"] == 150_000_000
    assert contract["measured_max_abs_skew_ns"] == {"left_wrist": 12_000_000}
    assert contract["measured_max_camera_age_ns"] == {
        "head": 103_000_000,
        "left_wrist": 115_000_000,
    }
    assert contract["verified"] is True

    loaded = porter.load_and_validate_episode(path, fps=10)
    assert loaded["camera_alignment_contract"]["verified"] is True


def test_camera_alignment_contract_rejects_excess_corrected_skew(porter, tmp_path):
    path = _write_raw(
        tmp_path / "episode_0.hdf5",
        timing=True,
        camera_ntp=True,
        camera_alignment=True,
    )
    with h5py.File(path, "r+") as raw:
        raw["obs/images/left_wrist_frame_ns"][:] -= np.int64(100_000_000)

    with pytest.raises(RuntimeError, match=r"frame 0.*exceeding the declared 40 ms"):
        porter.load_and_validate_episode(path, fps=10)


def test_camera_alignment_contract_rejects_excess_capture_age(porter, tmp_path):
    path = _write_raw(
        tmp_path / "episode_0.hdf5",
        timing=True,
        camera_ntp=True,
        camera_alignment=True,
    )
    with h5py.File(path, "r+") as raw:
        raw["obs/images/grab_wall_ns"][:] += np.int64(100_000_000)

    with pytest.raises(RuntimeError, match=r"head camera age at frame 0.*declared"):
        porter.load_and_validate_episode(path, fps=10)


def test_raw_v4_requires_camera_alignment_contract(porter, tmp_path):
    path = _write_raw(
        tmp_path / "episode_0.hdf5",
        timing=True,
        camera_ntp=True,
        camera_alignment=True,
    )
    with h5py.File(path, "r+") as raw:
        del raw["meta/camera_alignment"]

    with pytest.raises(RuntimeError, match=r"raw/v4 requires meta/camera_alignment"):
        porter.inspect_episode_camera_alignment_contract(path)


def test_work_camera_alignment_contract_rejects_mixed_semantics(porter, tmp_path):
    legacy = _write_raw(
        tmp_path / "episode_0.hdf5", timing=True, camera_ntp=True
    )
    aligned = _write_raw(
        tmp_path / "episode_1.hdf5",
        timing=True,
        camera_ntp=True,
        camera_alignment=True,
    )
    work = [
        {"source": "raw", "path": legacy, "raw_index": 0, "trim": None},
        {"source": "raw", "path": aligned, "raw_index": 1, "trim": None},
    ]

    with pytest.raises(RuntimeError, match="mix camera-alignment semantics/limits"):
        porter.validate_work_camera_alignment_contract(work)


def test_leading_nan_grippers_dropped_and_midtake_nan_raises(porter, tmp_path):
    lead = np.array([np.nan, 0.3, 0.3], np.float32)
    ep = porter.load_and_validate_episode(
        _write_raw(tmp_path / "episode_0.hdf5", gripper_left=lead), fps=10)
    assert ep["t0"] == 1  # leading NaN frame dropped

    mid = np.array([0.3, np.nan, 0.3], np.float32)
    with pytest.raises(RuntimeError, match="NaN at frame 1"):
        porter.load_and_validate_episode(
            _write_raw(tmp_path / "episode_1.hdf5", gripper_left=mid), fps=10)


def test_cadence_mismatch_raises(porter, tmp_path):
    path = _write_raw(tmp_path / "episode_0.hdf5")
    with h5py.File(path, "r+") as f:
        ts = np.asarray(f["timestamp_ns"])
        del f["timestamp_ns"]
        f["timestamp_ns"] = (ts - ts[0]) * 5 + ts[0]  # 2 Hz take vs --fps 10
    with pytest.raises(RuntimeError, match="does not match"):
        porter.load_and_validate_episode(path, fps=10)


def test_timing_fields_roundtrip_and_old_takes_get_none(porter, tmp_path):
    path = _write_raw(tmp_path / "episode_0.hdf5", T=4, timing=True)
    ep = porter.load_and_validate_episode(path, fps=10)
    timing = ep["timing"]
    assert timing is not None
    with h5py.File(path, "r") as f:
        for key in ("head_frame_ns", "left_wrist_frame_ns", "grab_wall_ns"):
            np.testing.assert_array_equal(timing[key], np.asarray(f[f"obs/images/{key}"]))
            assert timing[key].dtype == np.int64
        for key in ("offset_ns", "rtt_ns", "queried_at_ns"):
            assert int(timing[f"ntp_{key}"]) == int(np.asarray(f[f"meta/ntp/{key}"]))
    assert int(timing["ntp_offset_ns"]) == -12_000_000  # negative offset is legal
    np.testing.assert_array_equal(
        ep["timestamp_ns"], 1_000_000_000 + np.arange(4) * 100_000_000)

    old = porter.load_and_validate_episode(_write_raw(tmp_path / "episode_1.hdf5"), fps=10)
    assert old["timing"] is None


def test_causal_timing_roundtrips_and_rejects_partial_or_misordered(porter, tmp_path):
    path = _write_raw(
        tmp_path / "episode_0.hdf5", T=4, timing=True, causal_timing=True
    )
    ep = porter.load_and_validate_episode(path, fps=10)
    for key in porter._CAUSAL_TIMING_KEYS:
        with h5py.File(path, "r") as raw:
            np.testing.assert_array_equal(ep["timing"][key], raw[f"timing/{key}"])

    partial = _write_raw(
        tmp_path / "episode_1.hdf5", timing=True, causal_timing=True,
        drop_keys=("timing/state_read_end_wall_ns",),
    )
    with pytest.raises(RuntimeError, match="partial timing causal bracket"):
        porter.load_and_validate_episode(partial, fps=10)

    misordered = _write_raw(
        tmp_path / "episode_2.hdf5", timing=True, causal_timing=True
    )
    with h5py.File(misordered, "r+") as raw:
        values = np.asarray(raw["timing/state_read_start_wall_ns"])
        values[1] = np.asarray(raw["obs/images/grab_wall_ns"])[1] - 1
        del raw["timing/state_read_start_wall_ns"]
        raw["timing/state_read_start_wall_ns"] = values
    with pytest.raises(RuntimeError, match="causal timing order violated"):
        porter.load_and_validate_episode(misordered, fps=10)


@pytest.mark.parametrize(("drop", "match"), [
    (("obs/images/grab_wall_ns",), "partial timing"),
    (("obs/images/head_frame_ns", "obs/images/left_wrist_frame_ns"), "partial timing"),
    (("meta/ntp/rtt_ns",), "partial meta/ntp"),
])
def test_partial_timing_or_ntp_raises(porter, tmp_path, drop, match):
    path = _write_raw(tmp_path / "episode_0.hdf5", timing=True, drop_keys=drop)
    with pytest.raises(RuntimeError, match=match):
        porter.load_and_validate_episode(path, fps=10)


def test_partial_camera_ntp_raises(porter, tmp_path):
    path = _write_raw(
        tmp_path / "episode_0.hdf5",
        timing=True,
        camera_ntp=True,
        drop_keys=("meta/camera_ntp/head/rtt_ns",),
    )
    with pytest.raises(RuntimeError, match="partial meta/camera_ntp/head"):
        porter.load_and_validate_episode(path, fps=10)


def test_duplicate_capture_stamp_and_bad_dtype_raise(porter, tmp_path):
    path = _write_raw(tmp_path / "episode_0.hdf5", timing=True)
    with h5py.File(path, "r+") as f:
        stamps = np.asarray(f["obs/images/head_frame_ns"])
        stamps[2] = stamps[1]  # duplicate frame -- the freshness gate forbids this
        del f["obs/images/head_frame_ns"]
        f["obs/images/head_frame_ns"] = stamps
    with pytest.raises(RuntimeError, match="strictly increasing"):
        porter.load_and_validate_episode(path, fps=10)

    path2 = _write_raw(tmp_path / "episode_1.hdf5", timing=True)
    with h5py.File(path2, "r+") as f:
        stamps = np.asarray(f["obs/images/grab_wall_ns"]).astype(np.int32)
        del f["obs/images/grab_wall_ns"]
        f["obs/images/grab_wall_ns"] = stamps
    with pytest.raises(RuntimeError, match="int64"):
        porter.load_and_validate_episode(path2, fps=10)


def test_timing_sidecar_arrays_trims_per_frame_keeps_scalars(porter, tmp_path):
    path = _write_raw(tmp_path / "episode_0.hdf5", T=5, timing=True, depth_timing=True)
    ep = porter.load_and_validate_episode(path, fps=10)
    arrays = porter.timing_sidecar_arrays(ep, 1, 4)
    assert set(arrays) == {
        "timestamp_ns", "head_frame_ns", "head_depth_frame_ns",
        "left_wrist_frame_ns", "grab_wall_ns",
        "ntp_offset_ns", "ntp_rtt_ns", "ntp_queried_at_ns",
    }
    for key in (
        "timestamp_ns", "head_frame_ns", "head_depth_frame_ns",
        "left_wrist_frame_ns", "grab_wall_ns",
    ):
        assert arrays[key].shape == (3,)
    np.testing.assert_array_equal(arrays["timestamp_ns"], ep["timestamp_ns"][1:4])
    np.testing.assert_array_equal(arrays["head_frame_ns"], ep["timing"]["head_frame_ns"][1:4])
    np.testing.assert_array_equal(
        arrays["head_depth_frame_ns"], ep["timing"]["head_depth_frame_ns"][1:4]
    )
    assert arrays["ntp_offset_ns"].shape == ()

    old = porter.load_and_validate_episode(_write_raw(tmp_path / "episode_1.hdf5"), fps=10)
    assert set(porter.timing_sidecar_arrays(old, 0, 3)) == {"timestamp_ns"}


def test_metadata_sidecar_copies_static_meta_exactly(porter, tmp_path):
    raw_path = _write_raw(tmp_path / "episode_0.hdf5", timing=True)
    with h5py.File(raw_path, "a") as raw:
        raw["meta/provenance/schema"] = np.asarray(b"omniteleop_wbc_mobile_raw/v2")
        raw["meta/provenance/source_snapshot/wbc_vr_robot/sha256"] = np.asarray(
            b"a" * 64
        )
        raw["meta/cameras/head/serial_number"] = np.int64(12930616)
    output = porter._metadata_sidecar_path(tmp_path / "processed", 0)

    porter.write_metadata_sidecar(raw_path, output)

    with h5py.File(raw_path, "r") as raw, h5py.File(output, "r") as copied:
        expected = porter._hdf5_leaf_arrays(raw["meta"])
        actual = porter._hdf5_leaf_arrays(copied["meta"])
        assert set(actual) == set(expected)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        assert copied.attrs["schema"] == "omniteleop_raw_metadata_sidecar/v1"


def test_stereo_only_take_ports_without_fabricating_depth(porter, fk, tmp_path):
    path = _write_raw(
        tmp_path / "episode_0.hdf5", T=3, timing=True, depth=False
    )
    with h5py.File(path, "a") as raw:
        steps = np.arange(3, dtype=np.int64) * 100_000_000
        raw["obs/images/head_right_frame_ns"] = 900_000_000 + steps
        raw["obs/images/left_wrist_right_frame_ns"] = 890_000_000 + steps

    ep = porter.load_and_validate_episode(path, fps=10)
    assert ep["depth"] is None
    timing = porter.timing_sidecar_arrays(ep, 0, 3)
    assert "head_right_frame_ns" in timing
    assert "left_wrist_right_frame_ns" in timing

    class _Dataset:
        def __init__(self):
            self.frames = []
            self.saved = False

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            self.saved = True

    dataset = _Dataset()
    root = tmp_path / "processed"
    before, after = porter.add_episode(
        dataset,
        path,
        fk,
        resize_h=4,
        resize_w=3,
        task="test",
        variant_root=root,
        episode_idx=0,
        fps=10,
    )

    assert (before, after) == (3, 3)
    assert dataset.saved and len(dataset.frames) == 3
    assert not porter._sidecar_path(root, "depth", 0).exists()
    assert porter._sidecar_path(root, "calib", 0).exists()
    timing_path = porter._sidecar_path(root, "timing", 0)
    assert timing_path.exists()
    assert porter._metadata_sidecar_path(root, 0).exists()
    with np.load(timing_path) as data:
        assert "head_right_frame_ns" in data
        assert "left_wrist_right_frame_ns" in data


def test_sustained_gripper_command_with_constant_feedback_is_rejected(
    porter, tmp_path
):
    path = _write_raw(tmp_path / "episode_0.hdf5", T=20, depth=False)
    with h5py.File(path, "r+") as raw:
        raw["action/gripper/left"][:10] = 0.0
        raw["action/gripper/left"][10:] = 1.0
        raw["obs/gripper/left"][:] = 3.0 / 255.0

    with pytest.raises(RuntimeError, match=r"obs/gripper/left is stale or stuck"):
        porter.load_and_validate_episode(path, fps=10)


def test_gripper_quality_gate_allows_stationary_command_or_encoder_motion(
    porter, tmp_path
):
    stationary = _write_raw(tmp_path / "episode_0.hdf5", T=20, depth=False)
    assert porter.load_and_validate_episode(stationary, fps=10)["frame_count"] == 20

    moving = _write_raw(tmp_path / "episode_1.hdf5", T=20, depth=False)
    with h5py.File(moving, "r+") as raw:
        raw["action/gripper/left"][:10] = 0.0
        raw["action/gripper/left"][10:] = 1.0
        raw["obs/gripper/left"][:] = np.linspace(3 / 255.0, 0.6, 20)
    assert porter.load_and_validate_episode(moving, fps=10)["frame_count"] == 20


def test_raw_v4_rejects_stale_gripper_status_and_preserves_valid_timing(
    porter, tmp_path
):
    path = _write_raw(
        tmp_path / "episode_0.hdf5",
        T=5,
        timing=True,
        camera_ntp=True,
        camera_alignment=True,
        causal_timing=True,
    )
    episode = porter.load_and_validate_episode(path, fps=10)
    assert "gripper_left_read_wall_ns" in episode["timing"]
    assert "gripper_right_status_request_id" in episode["timing"]

    with h5py.File(path, "r+") as raw:
        for key in ("request_send_wall_ns", "read_wall_ns"):
            raw[f"obs/gripper_status/left/{key}"][:] -= np.int64(500_000_000)

    with pytest.raises(RuntimeError, match=r"505\.0 ms old at commit"):
        porter.load_and_validate_episode(path, fps=10)


def test_raw_v4_rejects_reused_gripper_status_event(porter, tmp_path):
    path = _write_raw(
        tmp_path / "episode_0.hdf5",
        T=3,
        timing=True,
        camera_ntp=True,
        camera_alignment=True,
    )
    with h5py.File(path, "r+") as raw:
        raw["obs/gripper_status/right/status_request_id"][1] = np.int64(0)

    with pytest.raises(RuntimeError, match="status_request_id is not strictly increasing"):
        porter.load_and_validate_episode(path, fps=10)


def test_raw_v5_accepts_explicit_bounded_gripper_status_reuse(porter, tmp_path):
    path = _write_raw(
        tmp_path / "episode_0.hdf5",
        T=3,
        timing=True,
        camera_ntp=True,
        camera_alignment=True,
        causal_timing=True,
    )
    with h5py.File(path, "r+") as raw:
        for schema_key in ("meta/schema", "meta/provenance/schema"):
            raw[schema_key][...] = np.asarray(b"omniteleop_wbc_mobile_raw/v5")
        for side in ("left", "right"):
            group = raw[f"obs/gripper_status/{side}"]
            group["status_request_id"][:] = np.asarray([0, 0, 2], np.int64)
            group["request_send_wall_ns"][:] = np.asarray(
                [990_000_000, 990_000_000, 1_190_000_000], np.int64
            )
            group["read_wall_ns"][:] = np.asarray(
                [995_000_000, 995_000_000, 1_195_000_000], np.int64
            )
            group["sample_reused"] = np.asarray([False, True, False], np.bool_)

    episode = porter.load_and_validate_episode(path, fps=10)
    for side in ("left", "right"):
        np.testing.assert_array_equal(
            episode["timing"][f"gripper_{side}_sample_reused"],
            [False, True, False],
        )
        np.testing.assert_array_equal(
            episode["timing"][f"gripper_{side}_status_age_ns"],
            [5_000_000, 105_000_000, 5_000_000],
        )


def test_raw_v5_rejects_unmarked_or_mutated_gripper_status_reuse(porter, tmp_path):
    path = _write_raw(
        tmp_path / "episode_0.hdf5",
        T=3,
        timing=True,
        camera_ntp=True,
        camera_alignment=True,
    )
    with h5py.File(path, "r+") as raw:
        for schema_key in ("meta/schema", "meta/provenance/schema"):
            raw[schema_key][...] = np.asarray(b"omniteleop_wbc_mobile_raw/v5")
        for side in ("left", "right"):
            group = raw[f"obs/gripper_status/{side}"]
            group["sample_reused"] = np.zeros(3, np.bool_)
        right = raw["obs/gripper_status/right"]
        right["status_request_id"][1] = np.int64(0)
        right["request_send_wall_ns"][1] = right["request_send_wall_ns"][0]
        right["read_wall_ns"][1] = right["read_wall_ns"][0]

    with pytest.raises(RuntimeError, match="does not match status_request_id reuse"):
        porter.load_and_validate_episode(path, fps=10)


def test_camera_ntp_metadata_preserved_in_timing_sidecar(porter, tmp_path):
    path = _write_raw(tmp_path / "episode_0.hdf5", T=5, timing=True, camera_ntp=True)
    ep = porter.load_and_validate_episode(path, fps=10)
    arrays = porter.timing_sidecar_arrays(ep, 1, 4)

    assert int(arrays["camera_ntp_head_offset_ns"]) == 5_000_000
    assert int(arrays["camera_ntp_wrist_offset_ns"]) == 7_000_000
    assert arrays["camera_ntp_head_sensor_id"].shape == ()
    assert arrays["camera_ntp_head_sensor_id"].item() == b"head_camera"
    assert arrays["camera_ntp_wrist_source"].item() == b"zed_publisher_host"
    np.testing.assert_array_equal(arrays["head_frame_ns"], ep["timing"]["head_frame_ns"][1:4])


def test_corrupt_rotation_column_raises(porter, tmp_path):
    path = _write_raw(tmp_path / "episode_0.hdf5")
    with h5py.File(path, "r+") as f:
        bad = np.asarray(f["action/eef/left"][1])
        bad[:3, 0] *= 1.5  # de-normalized rotation column
        f["action/eef/left"][1] = bad
    with pytest.raises(RuntimeError, match="not unit"):
        porter.load_and_validate_episode(path, fps=10)


def test_scaled_intrinsic(porter):
    k = np.array([[100.0, 0, 3.0], [0, 200.0, 4.0], [0, 0, 1]], np.float32)
    out = porter._scaled_intrinsic(k, raw_hw=(8, 6), out_hw=(4, 3))
    np.testing.assert_allclose(out[0], [50.0, 0.0, 1.5])   # width halved
    np.testing.assert_allclose(out[1], [0.0, 100.0, 2.0])  # height halved
    np.testing.assert_allclose(out[2], [0.0, 0.0, 1.0])


# --- Numeric episode ordering -------------------------------------------------

def test_episode_index_from_path(porter):
    assert porter.episode_index_from_path(Path("episode_5.hdf5")) == 5
    assert porter.episode_index_from_path(Path("/a/b/episode_61.hdf5")) == 61
    for bad in ("episode_5_debug.hdf5", "episode_.hdf5", "episode_5.h5", "foo.hdf5"):
        with pytest.raises(RuntimeError, match=r"episode_<int>\.hdf5"):
            porter.episode_index_from_path(Path(bad))


def test_find_episode_files_numeric_sort_ignores_nonmatching(porter, tmp_path):
    # Non-zero-padded names: a lexicographic sort would give 1, 11, 2 -- wrong.
    for name in ("episode_2.hdf5", "episode_11.hdf5", "episode_1.hdf5",
                 "episode_3_debug.hdf5", "notes.txt"):
        (tmp_path / name).touch()
    found = porter.find_episode_files(tmp_path)
    assert [p.name for p in found] == ["episode_1.hdf5", "episode_2.hdf5", "episode_11.hdf5"]


def test_find_episode_files_empty_raises(porter, tmp_path):
    with pytest.raises(RuntimeError, match="no episode_"):
        porter.find_episode_files(tmp_path)


# --- Recovery trim.csv --------------------------------------------------------

def _write_trim(recovery_dir: Path, text: str) -> Path:
    recovery_dir.mkdir(parents=True, exist_ok=True)
    (recovery_dir / "trim.csv").write_text(text)
    return recovery_dir


def test_load_trim_spec_parses_header_spaces_and_sentinel(porter, tmp_path):
    rec = _write_trim(tmp_path / "recovery",
                      "episode, frame_start, frame_end\n25, 409, -1\n27, 245, 300\n")
    assert porter.load_trim_spec(rec) == {25: (409, -1), 27: (245, 300)}


def test_load_trim_spec_missing_file_raises(porter, tmp_path):
    (tmp_path / "recovery").mkdir()
    with pytest.raises(RuntimeError, match="requires the trim file"):
        porter.load_trim_spec(tmp_path / "recovery")


@pytest.mark.parametrize("text, match", [
    ("episode, frame_start, frame_end\n25, 409, -1\n25, 5, 9\n", "duplicate episode 25"),
    ("episode, frame_start, frame_end\nfoo, 1, 2\n", "non-integer"),
    ("episode, frame_start, frame_end\n25, 10, 3\n", "frame_end 3 < frame_start 10"),
    ("episode, frame_start, frame_end\n25, -1, 5\n", "frame_start must be >= 0"),
    ("episode, frame_start, frame_end\n25, 0, -2\n", "frame_end must be >= 0 or -1"),
    ("episode, frame_start\n25, 409\n", "expected 3 columns"),
    ("episode, frame_start, frame_end\n", "no trim rows"),
])
def test_load_trim_spec_rejects_bad_rows(porter, tmp_path, text, match):
    rec = _write_trim(tmp_path / "recovery", text)
    with pytest.raises(RuntimeError, match=match):
        porter.load_trim_spec(rec)


def test_resolve_window(porter):
    # Raw episode: leading-NaN drop window [t0, T).
    assert porter.resolve_episode_window(10, 2, None, "src") == (2, 10)
    # Recovery: ABSOLUTE INCLUSIVE [start, end]; -1 -> last frame.
    assert porter.resolve_episode_window(10, 0, (3, 7), "src") == (3, 8)
    assert porter.resolve_episode_window(10, 0, (3, -1), "src") == (3, 10)
    assert porter.resolve_episode_window(10, 5, (0, 0), "src") == (0, 1)
    for bad in ((3, 20), (20, -1), (5, 4)):
        with pytest.raises(RuntimeError, match="out of range"):
            porter.resolve_episode_window(10, 0, bad, "src")


# --- Work list (raw then recovery) + log index --------------------------------

def _make_dirs(porter, tmp_path, raw_idx, rec_idx, trim_text):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for i in raw_idx:
        (raw_dir / f"episode_{i}.hdf5").touch()
    rec_dir = raw_dir / "recovery"
    rec_dir.mkdir()
    for i in rec_idx:
        (rec_dir / f"episode_{i}.hdf5").touch()
    (rec_dir / "trim.csv").write_text(trim_text)
    return raw_dir, rec_dir


def test_build_work_list_raw_only(porter, tmp_path):
    raw_dir, _ = _make_dirs(porter, tmp_path, [2, 11, 1], [], "episode\n")
    work = porter.build_work_list(raw_dir, None)  # recovery subdir ignored (one level down)
    assert [(w["source"], w["raw_index"], w["trim"]) for w in work] == [
        ("raw", 1, None), ("raw", 2, None), ("raw", 11, None)]


def test_build_work_list_appends_recovery_sorted(porter, tmp_path):
    raw_dir, rec_dir = _make_dirs(
        porter, tmp_path, [1, 2], [27, 25],
        "episode, frame_start, frame_end\n25, 409, -1\n27, 245, 300\n")
    work = porter.build_work_list(raw_dir, rec_dir)
    assert [(w["source"], w["raw_index"], w["trim"]) for w in work] == [
        ("raw", 1, None), ("raw", 2, None),
        ("recovery", 25, (409, -1)), ("recovery", 27, (245, 300))]


def test_build_work_list_strict_mismatch_raises(porter, tmp_path):
    # episode_99 file has no trim row.
    raw_dir, rec_dir = _make_dirs(
        porter, tmp_path, [1], [25, 99],
        "episode, frame_start, frame_end\n25, 409, -1\n")
    with pytest.raises(RuntimeError, match=r"without a trim row=\[99\]"):
        porter.build_work_list(raw_dir, rec_dir)
    # trim row 50 has no file.
    (rec_dir / "trim.csv").write_text(
        "episode, frame_start, frame_end\n25, 409, -1\n99, 1, -1\n50, 1, -1\n")
    with pytest.raises(RuntimeError, match=r"without a file=\[50\]"):
        porter.build_work_list(raw_dir, rec_dir)


def test_build_work_list_recovery_equals_raw_raises(porter, tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "episode_1.hdf5").touch()
    (raw_dir / "trim.csv").write_text("episode, frame_start, frame_end\n1, 0, -1\n")
    with pytest.raises(RuntimeError, match="must differ from --raw-dir"):
        porter.build_work_list(raw_dir, raw_dir)


def test_write_log_index_roundtrip(porter, tmp_path):
    rows = [
        {"split": "train", "source": "raw", "raw_index": 1, "processed_index": 0,
         "num_before": 100, "num_after": 100},
        {"split": "test", "source": "recovery", "raw_index": 25, "processed_index": 0,
         "num_before": 500, "num_after": 91},
    ]
    out = tmp_path / "logs" / "log_index.csv"  # parent auto-created
    porter.write_log_index(out, rows)
    with open(out, newline="") as fh:
        got = list(csv.reader(fh))
    assert got[0] == ["split", "source", "raw_data_index", "processed_data_index",
                      "num_frames_before", "num_frames_after"]
    assert got[1] == ["train", "raw", "1", "0", "100", "100"]
    assert got[2] == ["test", "recovery", "25", "0", "500", "91"]


def test_work_plan_frame_counts(porter, tmp_path):
    """Numeric order, recovery appended, trimmed counts -- via public helpers, no lerobot."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_raw(raw_dir / "episode_2.hdf5", T=5)
    _write_raw(raw_dir / "episode_1.hdf5", T=5)
    rec_dir = raw_dir / "recovery"
    rec_dir.mkdir()
    _write_raw(rec_dir / "episode_9.hdf5", T=8)
    (rec_dir / "trim.csv").write_text("episode, frame_start, frame_end\n9, 3, -1\n")

    work = porter.build_work_list(raw_dir, rec_dir)
    got = []
    for item in work:
        ep = porter.load_and_validate_episode(item["path"], fps=10)
        start, end = porter.resolve_episode_window(
            ep["frame_count"], ep["t0"], item["trim"], item["path"])
        got.append((item["source"], item["raw_index"], ep["frame_count"], end - start))
    # Raw sorted numerically (1 before 2), recovery appended; recovery trimmed [3,7] = 5.
    assert got == [("raw", 1, 5, 5), ("raw", 2, 5, 5), ("recovery", 9, 8, 5)]


# --- split.csv ----------------------------------------------------------------

def test_parse_split_csv_parses_raw_and_recovery_refs(porter, tmp_path):
    p = tmp_path / "split.csv"
    p.write_text(
        "episode, split\n61, test\nrecovery/56, test\n2, val\n3, offline_val\n"
    )
    assert porter.parse_split_csv(p) == {
        ("raw", 61): "test",
        ("recovery", 56): "test",
        ("raw", 2): "val",
        ("raw", 3): "offline_val",
    }


def test_parse_split_csv_missing_file_raises(porter, tmp_path):
    with pytest.raises(RuntimeError, match="split csv not found"):
        porter.parse_split_csv(tmp_path / "nope.csv")


@pytest.mark.parametrize("text, match", [
    ("episode, split\n61, prod\n", "unknown split"),
    ("episode, split\n61, test\n61, val\n", "duplicate episode"),
    ("episode, split\nfoo, test\n", "bad episode ref"),
    ("episode, split\nrecovery/x, test\n", "bad episode ref"),
    ("episode, split\n61\n", "expected 2 columns"),
])
def test_parse_split_csv_rejects_bad_rows(porter, tmp_path, text, match):
    p = tmp_path / "split.csv"
    p.write_text(text)
    with pytest.raises(RuntimeError, match=match):
        porter.parse_split_csv(p)


def test_assign_splits_partitions_and_defaults_train(porter, tmp_path):
    raw_dir, rec_dir = _make_dirs(
        porter, tmp_path, [1, 2], [56], "episode, frame_start, frame_end\n56, 0, -1\n")
    work = porter.build_work_list(raw_dir, rec_dir)
    split_map = {("raw", 2): "test", ("recovery", 56): "test"}
    splits = porter.assign_splits(work, split_map, rec_dir)
    # ALL_SPLITS order, empty splits dropped; unlisted episode_1 -> train.
    assert list(splits) == ["train", "test"]
    assert [(w["source"], w["raw_index"]) for w in splits["train"]] == [("raw", 1)]
    assert [(w["source"], w["raw_index"]) for w in splits["test"]] == [
        ("raw", 2), ("recovery", 56)]


def test_assign_splits_can_default_one_collection_directory(porter, tmp_path):
    raw_dir, _ = _make_dirs(porter, tmp_path, [1, 2], [], "episode\n")
    work = porter.build_work_list(raw_dir, None)
    splits = porter.assign_splits(
        work,
        {},
        default_split="spatial_ood_test",
    )
    assert list(splits) == ["spatial_ood_test"]
    assert [item["raw_index"] for item in splits["spatial_ood_test"]] == [1, 2]

    with pytest.raises(ValueError, match="default_split"):
        porter.assign_splits(work, {}, default_split="not_a_split")


def test_assign_splits_rejects_stale_reference(porter, tmp_path):
    raw_dir, _ = _make_dirs(porter, tmp_path, [1, 2], [], "episode\n")
    work = porter.build_work_list(raw_dir, None)
    with pytest.raises(RuntimeError, match=r"not in the port plan: recovery/56"):
        porter.assign_splits(work, {("recovery", 56): "test"}, None)


# --- ENV-state conditioning (--positions-dir) ---------------------------------

def _write_positions_npz(path: Path, positions, order, frame="world") -> Path:
    """Write a minimal Part-A positions npz (positions + order + frame)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, positions=np.asarray(positions, np.float32),
             order=np.asarray(order, np.int64), frame=np.asarray(frame))
    return path


def test_env_state_axes_stage_named_and_generic(porter):
    assert porter.env_state_axes(2) == [
        "s1_src_x", "s1_src_y", "s1_src_z", "s1_dst_x", "s1_dst_y", "s1_dst_z"]
    # 4 objects = 2 grasp cycles -> s1_*, then s2_*.
    assert porter.env_state_axes(4) == [
        "s1_src_x", "s1_src_y", "s1_src_z", "s1_dst_x", "s1_dst_y", "s1_dst_z",
        "s2_src_x", "s2_src_y", "s2_src_z", "s2_dst_x", "s2_dst_y", "s2_dst_z"]
    # Odd count falls back to generic obj{i} naming.
    assert porter.env_state_axes(3) == [
        "obj0_x", "obj0_y", "obj0_z", "obj1_x", "obj1_y", "obj1_z",
        "obj2_x", "obj2_y", "obj2_z"]


def test_build_env_state_vector_reorders_and_flattens(porter):
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32)
    vec = porter.build_env_state_vector(positions, [1, 0])  # swap the two size slots
    assert vec.dtype == np.float32
    np.testing.assert_array_equal(vec, [4.0, 5.0, 6.0, 1.0, 2.0, 3.0])


def test_load_episode_positions_reorders_by_npz_order(porter, tmp_path):
    # size-order slot0=box, slot1=cloth; order=[0,1] -> [s1_src=box, s1_dst=cloth].
    _write_positions_npz(tmp_path / "raw" / "episode_7.npz",
                         [[1.5, 0.7, 1.1], [1.0, -0.2, 0.7]], [0, 1])
    vec = porter.load_episode_positions(tmp_path, "raw", 7, object_nums=2)
    np.testing.assert_allclose(vec, [1.5, 0.7, 1.1, 1.0, -0.2, 0.7], atol=1e-6)
    # A non-identity order actually permutes the flattened vector.
    _write_positions_npz(tmp_path / "raw" / "episode_8.npz",
                         [[1.5, 0.7, 1.1], [1.0, -0.2, 0.7]], [1, 0])
    vec2 = porter.load_episode_positions(tmp_path, "raw", 8, object_nums=2)
    np.testing.assert_allclose(vec2, [1.0, -0.2, 0.7, 1.5, 0.7, 1.1], atol=1e-6)


def test_load_episode_positions_source_subdir_disambiguates_collision(porter, tmp_path):
    # raw/episode_25 and recovery/episode_25 share an index but are different files.
    _write_positions_npz(tmp_path / "raw" / "episode_25.npz",
                         [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [0, 1])
    _write_positions_npz(tmp_path / "recovery" / "episode_25.npz",
                         [[9.0, 9.0, 9.0], [8.0, 8.0, 8.0]], [0, 1])
    raw_vec = porter.load_episode_positions(tmp_path, "raw", 25, object_nums=2)
    rec_vec = porter.load_episode_positions(tmp_path, "recovery", 25, object_nums=2)
    np.testing.assert_allclose(raw_vec, [1, 1, 1, 2, 2, 2])
    np.testing.assert_allclose(rec_vec, [9, 9, 9, 8, 8, 8])


@pytest.mark.parametrize("mutate, match", [
    (lambda p: None, "not found"),  # never write the file
    (lambda p: _write_positions_npz(p, [[1.0, 1.0, 1.0]], [0]), r"!= \(2, 3\)"),  # wrong count
    (lambda p: _write_positions_npz(p, [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [0, 0]),
     "not a permutation"),
    (lambda p: _write_positions_npz(p, [[1.0, 1.0, 1.0], [np.nan, 2.0, 2.0]], [0, 1]),
     "non-finite"),
    (lambda p: _write_positions_npz(p, [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [0, 1],
                                    frame="robot_base"), "expected 'world'"),
])
def test_load_episode_positions_fails_fast(porter, tmp_path, mutate, match):
    mutate(tmp_path / "raw" / "episode_3.npz")
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError), match=match):
        porter.load_episode_positions(tmp_path, "raw", 3, object_nums=2)


def test_validate_all_episode_positions_aggregates_failures(porter, tmp_path):
    _write_positions_npz(tmp_path / "raw" / "episode_1.npz",
                         [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [0, 1])
    # episode_2 missing entirely; recovery/5 has a bad permutation.
    _write_positions_npz(tmp_path / "recovery" / "episode_5.npz",
                         [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [1, 1])
    work = [
        {"source": "raw", "raw_index": 1, "path": Path("x"), "trim": None},
        {"source": "raw", "raw_index": 2, "path": Path("x"), "trim": None},
        {"source": "recovery", "raw_index": 5, "path": Path("x"), "trim": (0, -1)},
    ]
    with pytest.raises(RuntimeError, match="position validation failed") as exc:
        porter.validate_all_episode_positions(tmp_path, work, object_nums=2)
    msg = str(exc.value)
    assert "raw/episode_2" in msg and "recovery/episode_5" in msg
    assert "raw/episode_1" not in msg  # the good one is not reported


def test_validate_all_episode_positions_returns_keyed_vectors(porter, tmp_path):
    _write_positions_npz(tmp_path / "raw" / "episode_1.npz",
                         [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [0, 1])
    _write_positions_npz(tmp_path / "recovery" / "episode_1.npz",
                         [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]], [0, 1])
    work = [
        {"source": "raw", "raw_index": 1, "path": Path("x"), "trim": None},
        {"source": "recovery", "raw_index": 1, "path": Path("x"), "trim": (0, -1)},
    ]
    got = porter.validate_all_episode_positions(tmp_path, work, object_nums=2)
    assert set(got) == {("raw", 1), ("recovery", 1)}
    np.testing.assert_allclose(got[("raw", 1)], [1, 1, 1, 2, 2, 2])
    np.testing.assert_allclose(got[("recovery", 1)], [3, 3, 3, 4, 4, 4])


def test_build_features_env_state_optional(porter):
    base = porter.build_features(120, 160)
    assert porter.OBS_ENV_STATE_KEY not in base
    conditioned = porter.build_features(120, 160, object_nums=2)
    feat = conditioned[porter.OBS_ENV_STATE_KEY]
    assert feat["dtype"] == "float32"
    assert feat["shape"] == (6,)
    assert feat["names"]["axes"] == porter.env_state_axes(2)
    # Single-stage task -> NO per-frame pos_condition_mask feature.
    assert "observation.pos_condition_mask" not in conditioned


def test_structured_raw_condition_plan_and_progress_roundtrip(porter, tmp_path):
    path = _write_raw(tmp_path / "episode_7.hdf5", T=6)
    condition = np.arange(18, dtype=np.float32).reshape(3, 2, 3)
    stages = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    sequence = np.array([0, 2, 1], dtype=np.int64)
    task_ids = sequence[stages]
    with h5py.File(path, "a") as raw:
        raw["task/env_state"] = condition
        raw["task/task_sequence"] = sequence
        raw["progress_index"] = stages
        raw["active_task_id"] = task_ids
    loaded_condition, loaded_sequence, loaded_stages, loaded_tasks = (
        porter.load_structured_env_state_from_raw(path)
    )
    np.testing.assert_array_equal(loaded_condition, condition)
    np.testing.assert_array_equal(loaded_sequence, sequence)
    np.testing.assert_array_equal(loaded_stages, stages)
    np.testing.assert_array_equal(loaded_tasks, task_ids)


def test_structured_raw_rejects_flat_or_nonmonotonic_labels(porter, tmp_path):
    path = _write_raw(tmp_path / "episode_8.hdf5", T=4)
    with h5py.File(path, "a") as raw:
        raw["task/env_state"] = np.zeros((18,), dtype=np.float32)
        raw["task/task_sequence"] = np.array([0, 2, 1], dtype=np.int64)
        raw["progress_index"] = np.array([0, 2, 1, 2], dtype=np.int64)
        raw["active_task_id"] = np.array([0, 1, 2, 1], dtype=np.int64)
    with pytest.raises(RuntimeError, match="task/env_state"):
        porter.load_structured_env_state_from_raw(path)
    with h5py.File(path, "a") as raw:
        del raw["task/env_state"]
        raw["task/env_state"] = np.zeros((3, 2, 3), dtype=np.float32)
    with pytest.raises(RuntimeError, match="monotonic"):
        porter.load_structured_env_state_from_raw(path)


def test_structured_features_preserve_rank_and_categorical_plan(porter):
    features = porter.build_features(
        120,
        160,
        structured_env_state=True,
    )
    assert features[porter.OBS_ENV_STATE_KEY]["shape"] == (3, 2, 3)
    assert features[porter.OBS_TASK_SEQUENCE_KEY]["dtype"] == "int64"
    assert features[porter.OBS_TASK_SEQUENCE_KEY]["shape"] == (3,)
    assert features[porter.PROGRESS_INDEX_KEY] == {
        "dtype": "int64",
        "shape": (1,),
        "names": {"axes": ["progress"]},
    }
    assert features[porter.ACTIVE_TASK_ID_KEY]["dtype"] == "int64"
