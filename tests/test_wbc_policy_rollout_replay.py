"""Replay-mode units for scripts/wbc_policy_rollout.py (plan.md Tasks 1-3).

Covers ``RecordedEpisodeSource`` (real-time release against a fake clock, raw
segment durations, bit-identical grippers, missing-dataset errors) and the
Task 2 regression: the stateful head filters must NEVER run in replay or
live-policy ticks because ``action/head`` is already post-LPF/post-deadband.

Run via the dexmate env with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

from omniteleop.wbc_policy_format import JOYSTICK_POLICY_ACTION_SCHEMA, mat_to_pos6d


def _load_rollout():
    path = Path(__file__).resolve().parents[1] / "scripts" / "wbc_policy_rollout.py"
    name = "wbc_policy_rollout_replay_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module  # dataclass string-annotation resolution needs this
    spec.loader.exec_module(module)
    return module


def _rollout_source_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "wbc_policy_rollout.py"


def _pose(tx: float, ty: float = 0.0, tz: float = 0.5, yaw: float = 0.0) -> np.ndarray:
    out = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    out[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    out[:3, 3] = [tx, ty, tz]
    return out


# Deliberately uneven gaps (0.1 s then 0.15 s) so the raw inter-frame durations
# are distinguishable from the nominal record period.
_TS_NS = np.array(
    [1_000_000_000_000, 1_000_100_000_000, 1_000_250_000_000], dtype=np.int64
)


def _write_episode(
    path: Path,
    drop: tuple[str, ...] = (),
    mutate=None,
    base_pose_source: str | None = None,
) -> Path:
    T = len(_TS_NS)
    data = {
        "action/eef/left": np.stack(
            [_pose(0.4 + 0.01 * t) for t in range(T)]
        ).astype(np.float32),
        "action/eef/right": np.stack(
            [_pose(0.4, -0.3 + 0.01 * t) for t in range(T)]
        ).astype(np.float32),
        "action/head": np.stack(
            [_pose(0.1, 0.0, 1.2, 0.1 * t) for t in range(T)]
        ).astype(np.float32),
        "action/gripper/left": np.array([0.12345678, 0.99999988, 0.0], dtype=np.float32),
        "action/gripper/right": np.array([1.0, 0.33000001, 0.5], dtype=np.float32),
        "timestamp_ns": _TS_NS.copy(),
    }
    if mutate is not None:
        mutate(data)
    with h5py.File(path, "w") as f:
        for key, arr in data.items():
            if key in drop:
                continue
            f.create_dataset(key, data=arr)
        if base_pose_source is not None:
            f["meta/schema"] = np.asarray(b"omniteleop_wbc_mobile_raw/v3")
            f["meta/obs_base_pose_source"] = np.asarray(base_pose_source.encode())
            f["meta/base_control_pose_source"] = np.asarray(base_pose_source.encode())
    return path


@pytest.mark.parametrize(
    "missing",
    [
        "action/eef/left",
        "action/eef/right",
        "action/head",
        "action/gripper/left",
        "action/gripper/right",
        "timestamp_ns",
    ],
)
def test_missing_dataset_raises_runtimeerror_naming_key(tmp_path, missing):
    mod = _load_rollout()
    ep = _write_episode(tmp_path / "episode_0.hdf5", drop=(missing,))
    with pytest.raises(RuntimeError, match=missing.replace("/", "/")):
        mod.RecordedEpisodeSource(ep)


def test_realtime_release_and_raw_segment_durations(tmp_path):
    """Frame t is held until timestamp_ns[t]-timestamp_ns[0] REAL seconds elapse."""
    mod = _load_rollout()
    src = mod.RecordedEpisodeSource(_write_episode(tmp_path / "episode_0.hdf5"))
    src.start(now=50.0)

    frame0 = src.advance(50.0)
    assert frame0 is not None and frame0.index == 0
    assert frame0.segment_duration == 0.0  # no previous frame: snap
    assert not src.done

    # 0.099 s elapsed < the 0.1 s recorded gap: frame 1 must NOT be released.
    assert src.advance(50.099) is None
    frame1 = src.advance(50.1)
    assert frame1 is not None and frame1.index == 1
    assert frame1.segment_duration == float((_TS_NS[1] - _TS_NS[0]) / 1e9)

    # Frame 2 sits 0.25 s after frame 0 with a raw 0.15 s gap -- no speed scaling.
    assert src.advance(50.2) is None
    frame2 = src.advance(50.25)
    assert frame2 is not None and frame2.index == 2
    assert frame2.segment_duration == float((_TS_NS[2] - _TS_NS[1]) / 1e9)
    assert src.done
    assert src.advance(51.0) is None
    assert src.base_pose_source == "wheel_odometry"  # legacy episode inference

    # Targets come back verbatim (float32 storage round-trip only).
    np.testing.assert_allclose(frame1.left, _pose(0.41), atol=1e-6)
    np.testing.assert_allclose(frame2.head[:3, 3], [0.1, 0.0, 1.2], atol=1e-6)


def test_joystick_episode_replay_requires_and_releases_exact_intent(tmp_path):
    mod = _load_rollout()
    path = _write_episode(
        tmp_path / "episode_0.hdf5", base_pose_source="wheel_odometry"
    )
    intent = np.array(
        [[0.2, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.3]],
        dtype=np.float32,
    )
    with h5py.File(path, "a") as raw:
        del raw["meta/schema"]
        raw["meta/schema"] = np.asarray(b"omniteleop_joystick_mobile_raw/v1")
        raw["meta/policy_action_schema"] = np.asarray(
            JOYSTICK_POLICY_ACTION_SCHEMA.encode()
        )
        raw["meta/control_mode"] = np.asarray(b"joystick")
        for key in ("action_target_frame", "eef_target_frame", "head_target_frame"):
            raw[f"meta/{key}"] = np.asarray(b"current_base")
        raw["action/chassis/intent_body"] = intent

    src = mod.RecordedEpisodeSource(path)
    assert src.policy_action_schema == JOYSTICK_POLICY_ACTION_SCHEMA
    assert src.action_target_frame == "current_base"
    src.start(1.0)
    frame = src.advance(1.0)
    assert frame is not None
    np.testing.assert_array_equal(frame.chassis, intent[0])


def test_replay_reads_explicit_base_pose_source(tmp_path):
    mod = _load_rollout()
    src = mod.RecordedEpisodeSource(
        _write_episode(
            tmp_path / "episode_0.hdf5", base_pose_source="arkit"
        )
    )
    assert src.base_pose_source == "arkit"


def test_gripper_values_bit_identical_to_raw(tmp_path):
    """Replay must send EXACTLY the recorded gripper commands: no binarize/clip."""
    mod = _load_rollout()
    ep = tmp_path / "episode_0.hdf5"
    _write_episode(ep)
    with h5py.File(ep, "r") as f:
        raw_l = np.asarray(f["action/gripper/left"])
        raw_r = np.asarray(f["action/gripper/right"])
    src = mod.RecordedEpisodeSource(ep)
    src.start(now=0.0)
    for t in range(3):
        frame = src.advance(1e9)  # far past every release time
        assert frame is not None
        assert frame.grip_left == raw_l[t]
        assert frame.grip_right == raw_r[t]


def test_advance_before_start_raises(tmp_path):
    mod = _load_rollout()
    src = mod.RecordedEpisodeSource(_write_episode(tmp_path / "episode_0.hdf5"))
    with pytest.raises(RuntimeError, match="start"):
        src.advance(0.0)


def test_rejects_corrupt_rotation_and_non_finite(tmp_path):
    mod = _load_rollout()

    def bad_rot(data):
        data["action/eef/left"][1, :3, 0] *= 2.0

    ep = _write_episode(tmp_path / "bad_rot.hdf5", mutate=bad_rot)
    with pytest.raises(RuntimeError, match="rotation"):
        mod.RecordedEpisodeSource(ep)

    def bad_nan(data):
        data["action/head"][0, 0, 3] = np.nan

    ep = _write_episode(tmp_path / "bad_nan.hdf5", mutate=bad_nan)
    with pytest.raises(RuntimeError, match="finite"):
        mod.RecordedEpisodeSource(ep)


def test_rejects_non_increasing_timestamps(tmp_path):
    mod = _load_rollout()

    def bad_ts(data):
        data["timestamp_ns"][2] = data["timestamp_ns"][1]

    ep = _write_episode(tmp_path / "bad_ts.hdf5", mutate=bad_ts)
    with pytest.raises(RuntimeError, match="increasing"):
        mod.RecordedEpisodeSource(ep)


def test_encode_replay_action_roundtrips_through_split(tmp_path):
    """Replay io_log actions use the same 29-D layout a live policy would emit."""
    mod = _load_rollout()
    left, right, head = _pose(0.4), _pose(0.4, -0.3), _pose(0.1, 0.0, 1.2, 0.3)
    action = mod.encode_replay_action(
        left, right, head, np.float32(0.25), np.float32(1.0)
    )
    assert action.shape == (29,) and action.dtype == np.float32
    np.testing.assert_array_equal(action[0:9], mat_to_pos6d(left))
    np.testing.assert_array_equal(action[10:19], mat_to_pos6d(right))
    np.testing.assert_array_equal(action[20:29], mat_to_pos6d(head))
    assert action[9] == np.float32(0.25) and action[19] == np.float32(1.0)
    targets = mod.split_policy_action(action)
    np.testing.assert_allclose(targets["head"], head, atol=1e-6)


class _SpyFilter:
    """Stands in for HeadTargetLowPassFilter / HeadTargetPlanarDeadbandFilter."""

    def __init__(self, tag: float) -> None:
        self.calls = 0
        self._tag = tag

    def filter(self, pose, *args):
        self.calls += 1
        out = np.asarray(pose, dtype=float).copy()
        out[2, 3] += self._tag  # watermark so downstream consumption is provable
        return out


class _FakeInterp:
    def __init__(self, left, right, head) -> None:
        self._poses = (left, right, head)

    def at(self, now):
        return tuple(p.copy() for p in self._poses)


class _FakeIK:
    def __init__(self) -> None:
        self.head_targets: list[np.ndarray] = []

    def solve(self, left, right, dt, head_target=None):
        self.head_targets.append(np.asarray(head_target, dtype=float))
        return types.SimpleNamespace(success=True, held=False)


class _FakeDriver:
    def __init__(self) -> None:
        self._episode = None
        self.actuated: list[tuple] = []

    def extra_hold(self, now, last_cmd_wall, estop):
        return None

    def actuate(self, result, grip_l, grip_r, enable, hold, dt):
        self.actuated.append((grip_l, grip_r, hold))


def test_joystick_tick_uses_head_tracking_solver_path():
    mod = _load_rollout()
    left, right, head = _pose(0.4), _pose(0.4, -0.3), _pose(0.1, 0.0, 1.2)

    class TrackIK:
        def __init__(self):
            self.head_target = None
            self.head_joints = None

        def solve_head(self, target, dt):
            self.head_target = np.asarray(target)
            return np.array([0.1, 0.2, 0.3])

        def solve(self, left_target, right_target, dt, *, head_joints):
            self.head_joints = np.asarray(head_joints)
            return types.SimpleNamespace(success=True, held=False)

    ik = TrackIK()
    driver = _FakeDriver()
    driver.cfg = types.SimpleNamespace(head_mode="track")
    mod.wbc_tick(
        ik=ik, driver=driver, enable={"arms": True},
        interp=_FakeInterp(left, right, head),
        head_lpf=_SpyFilter(0.0), head_deadband=_SpyFilter(0.0),
        now=1.0, dt=0.01, grip_left=0.0, grip_right=0.0,
        last_cmd_wall=1.0, live_head_filters=False,
    )
    np.testing.assert_array_equal(ik.head_target, head)
    np.testing.assert_allclose(ik.head_joints, [0.1, 0.2, 0.3])


def _tick(mod, *, live_head_filters):
    left, right, head = _pose(0.4), _pose(0.4, -0.3), _pose(0.1, 0.0, 1.2)
    ik = _FakeIK()
    driver = _FakeDriver()
    lpf = _SpyFilter(0.010)
    deadband = _SpyFilter(0.100)
    mod.wbc_tick(
        ik=ik, driver=driver, enable={"arms": True}, interp=_FakeInterp(left, right, head),
        head_lpf=lpf, head_deadband=deadband, now=1.0, dt=0.01,
        grip_left=0.0, grip_right=0.0, last_cmd_wall=1.0,
        live_head_filters=live_head_filters,
    )
    return ik, driver, lpf, deadband, head


def test_replay_tick_never_runs_head_filters():
    """action/head is already post-LPF/post-deadband: re-filtering = double lag."""
    mod = _load_rollout()
    ik, driver, lpf, deadband, head = _tick(mod, live_head_filters=False)
    assert lpf.calls == 0 and deadband.calls == 0
    np.testing.assert_array_equal(ik.head_targets[0], head)  # verbatim into ik.solve
    assert len(driver.actuated) == 1  # clamp/hold/safety path still fully active


def test_wbc_tick_estop_forces_hold():
    """compute_hold_reason returns None on estop: the tick must OR it in itself."""
    mod = _load_rollout()
    left, right, head = _pose(0.4), _pose(0.4, -0.3), _pose(0.1, 0.0, 1.2)
    driver = _FakeDriver()
    mod.wbc_tick(
        ik=_FakeIK(), driver=driver, enable={"arms": True},
        interp=_FakeInterp(left, right, head),
        head_lpf=_SpyFilter(0.0), head_deadband=_SpyFilter(0.0), now=1.0, dt=0.01,
        grip_left=0.0, grip_right=0.0, last_cmd_wall=1.0,
        live_head_filters=True, estop=True,
    )
    assert driver.actuated[0][2] is True  # hold


def test_run_rollout_policy_ticks_skip_head_filters_like_replay():
    """The hardware loop must not double-filter solver-ready policy/replay targets."""
    tree = ast.parse(_rollout_source_path().read_text())
    rollout_episode = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_run_rollout_episode"
    )
    calls = [
        n
        for n in ast.walk(rollout_episode)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "wbc_tick"
    ]
    # run_reference_alignment is tested separately; this is the main online loop call.
    assert len(calls) == 1
    kwargs = {kw.arg: kw.value for kw in calls[0].keywords}
    assert isinstance(kwargs["live_head_filters"], ast.Constant)
    assert kwargs["live_head_filters"].value is False


def test_reference_alignment_tick_keeps_head_filters_active():
    mod = _load_rollout()
    ik, driver, lpf, deadband, head = _tick(mod, live_head_filters=True)
    assert lpf.calls == 1 and deadband.calls == 1
    # LPF then deadband both ran, and their (watermarked) output reached ik.solve.
    expected = head.copy()
    expected[2, 3] += 0.010 + 0.100
    np.testing.assert_allclose(ik.head_targets[0], expected)
    assert len(driver.actuated) == 1
