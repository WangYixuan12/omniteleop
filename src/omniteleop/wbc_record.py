#!/usr/bin/env python3
"""Faithful record + time-scaled replay of the WBC VR leader's command stream.

The staged-safety bring-up records the leader's published ``VRJointData`` stream
while the operator teleops in VR (visual feedback from the SAPIEN sim follower),
then replays it -- time-scaled, e.g. 0.25x -- through the SAME whole-body IK on the
real robot (``scripts/wbc_vr_robot.py``). Recording captures EVERY ``VRJointData``
field plus the local monotonic arrival time, so replay is faithful to what the
follower would have seen live; only the source (a file) and the clock (stretched)
differ.

``ReplaySource`` has a second entry point, :meth:`ReplaySource.from_episode_hdf5`:
it replays the REAL-ROBOT follower's own ``EpisodeRecorder`` take by re-issuing the
world-frame ``action/eef``/``action/head`` targets it handed to ``ik.solve()``, so one
recorded take can be re-driven through a re-tuned ``follower/wbik.yaml``. Both entry
points take the same ``speed`` scale.

Light by design (``VRJointData`` + numpy; ``h5py`` only at file I/O), so the
recorder/replayer is unit-testable without any robot, sim, or solver imports.
"""

from __future__ import annotations

import pathlib
import time
from typing import Callable, Optional

import numpy as np

from omniteleop.common.schemas import VRJointData
from omniteleop.wbc_policy_format import JOYSTICK_POLICY_ACTION_SCHEMA

# Flattened-4x4 pose fields, recorded as (N, 16) float64 + a per-frame presence flag
# (the leader publishes empty lists for these outside teleop).
_POSE_FIELDS = ("left_ee_pose", "right_ee_pose", "head_ee_pose")
# Joint-list fields the WBC leader always leaves empty (it streams Cartesian EE
# targets, not joints). Recorded as a presence flag only; a non-empty value means a
# non-WBC leader (e.g. omni-vr) whose stream this replay path does not model -> error.
_JOINT_FIELDS = ("left_arm_pos", "right_arm_pos", "head_pos")
_FLOAT_FIELDS = (
    "left_gripper",
    "right_gripper",
    "chassis_vx",
    "chassis_vy",
    "chassis_wz",
)
_BOOL_FIELDS = ("estop", "exit_requested", "home_requested")
_STAGE_WIDTH = 16
DEFAULT_REPLAY_GAP_LIMIT_MULTIPLE = 5.0
DEFAULT_WBC_VR_OUTPUT = "/tmp/wbc_vr.mp4"

# --- Episode-HDF5 replay (ReplaySource.from_episode_hdf5) ---------------------------
# The real-robot follower's OWN take (``EpisodeRecorder`` schema, written by
# ``wbc_vr_robot.py --record``) stores, per frame, the WORLD-frame 4x4 targets it handed
# VERBATIM to ``ik.solve()``. Replaying THOSE -- instead of the leader's VRJointData
# stream -- re-drives the identical operator intent through a RE-TUNED
# ``follower/wbik.yaml``, so two parameter sets can be compared on one take.
# VRJointData field -> episode dataset path.
_EPISODE_POSE_DATASETS = {
    "left_ee_pose": "action/eef/left",
    "right_ee_pose": "action/eef/right",
    "head_ee_pose": "action/head",
}
_EPISODE_SCALAR_DATASETS = {
    "left_gripper": "action/gripper/left",
    "right_gripper": "action/gripper/right",
}
_EPISODE_TIME_DATASET = "timestamp_ns"
_EPISODE_JOYSTICK_INTENT_DATASET = "action/chassis/intent_body"


def _decode_episode_frames(
    arrays: dict[str, np.ndarray],
    *,
    chassis_dataset: str | None = None,
) -> list[tuple[float, VRJointData]]:
    """Rebuild ``(recv_mono, VRJointData)`` rows from an episode take's action targets.

    Only the solver INPUTS are consumed (the ``action/eef/*`` + ``action/head`` 4x4
    targets and the ``action/gripper/*`` trigger commands): everything the follower
    computed FROM them -- joint commands, base twist, obs -- is deliberately dropped so
    the replay re-solves with the current ``wbik.yaml``. Arrival times come from the
    frame ``timestamp_ns`` (the recorder's wall clock), which is what makes the replay
    honour the take's real cadence including any recording stall.

    Every frame is synthesized as engaged teleop (``estop=False``,
    ``calib_stage="teleop"``): an episode only ever contains frames recorded while the
    follower was engaged and past leader calibration.
    """
    ts = np.asarray(arrays[_EPISODE_TIME_DATASET], dtype=np.int64)
    if ts.ndim != 1 or ts.size < 2:
        raise ValueError(
            f"{_EPISODE_TIME_DATASET} must be a 1-D array of >= 2 frames, got shape {ts.shape}"
        )
    n = int(ts.size)
    if not np.all(np.diff(ts) > 0):
        bad = int(np.argmin(np.diff(ts)))
        raise ValueError(
            f"{_EPISODE_TIME_DATASET} must be strictly increasing; frames {bad} -> {bad + 1} "
            f"go {int(ts[bad])} -> {int(ts[bad + 1])} ns"
        )
    poses: dict[str, np.ndarray] = {}
    for fld, path in _EPISODE_POSE_DATASETS.items():
        mat = np.asarray(arrays[path], dtype=np.float64)
        if mat.shape != (n, 4, 4):
            raise ValueError(f"{path} must have shape {(n, 4, 4)}, got {mat.shape}")
        if not np.all(np.isfinite(mat)):
            raise ValueError(f"{path} contains non-finite values")
        # Homogeneous bottom row: catches a transposed/garbage pose before it is handed
        # to the IK as a Cartesian target.
        if not np.allclose(mat[:, 3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
            raise ValueError(f"{path} rows are not homogeneous 4x4 transforms")
        poses[fld] = mat
    scalars: dict[str, np.ndarray] = {}
    for fld, path in _EPISODE_SCALAR_DATASETS.items():
        val = np.asarray(arrays[path], dtype=np.float64)
        if val.shape != (n,):
            raise ValueError(f"{path} must have shape {(n,)}, got {val.shape}")
        if not np.all(np.isfinite(val)):
            raise ValueError(f"{path} contains non-finite values")
        scalars[fld] = val
    chassis: np.ndarray | None = None
    if chassis_dataset is not None:
        if chassis_dataset not in arrays:
            raise ValueError(
                f"joystick episode is missing required {chassis_dataset}; refusing to "
                "guess human intent from the feedback-shaped hardware twist"
            )
        chassis = np.asarray(arrays[chassis_dataset], dtype=np.float64)
        if chassis.shape != (n, 3):
            raise ValueError(f"{chassis_dataset} must have shape {(n, 3)}, got {chassis.shape}")
        if not np.all(np.isfinite(chassis)):
            raise ValueError(f"{chassis_dataset} contains non-finite values")
    recv = (ts - ts[0]) * 1e-9
    frames: list[tuple[float, VRJointData]] = []
    for i in range(n):
        kwargs: dict = {
            "timestamp_ns": int(ts[i]),
            "calib_stage": "teleop",
            "estop": False,
        }
        for fld, mat in poses.items():
            kwargs[fld] = mat[i].reshape(16).tolist()
        for fld, val in scalars.items():
            kwargs[fld] = float(val[i])
        if chassis is not None:
            kwargs.update(
                chassis_vx=float(chassis[i, 0]),
                chassis_vy=float(chassis[i, 1]),
                chassis_wz=float(chassis[i, 2]),
            )
        frames.append((float(recv[i]), VRJointData(**kwargs)))
    return frames


def resolve_record_paths(
    *,
    output: str | None,
    hdf5: str | None,
    no_hdf5: bool,
    default_output: str = DEFAULT_WBC_VR_OUTPUT,
) -> tuple[str, str | None]:
    """Pair ``--output`` (mp4) with ``--hdf5`` when only one is given."""
    if output is None:
        output = str(pathlib.Path(hdf5).with_suffix(".mp4")) if hdf5 else default_output
    hdf5_path = None if no_hdf5 else (
        hdf5 or str(pathlib.Path(output).with_suffix(".hdf5"))
    )
    return output, hdf5_path


class TeleopStartGate:
    """Latch open once the WBC leader leaves calibration/static mode."""

    def __init__(self) -> None:
        self._started = False

    @property
    def started(self) -> bool:
        """Whether the gate has latched open."""
        return self._started

    def update(self, vr: Optional[VRJointData]) -> bool:
        """Return True once a post-calibration teleop frame has arrived."""
        if self._started:
            return True
        if _is_post_calibration_teleop(vr):
            self._started = True
        return self._started


def _is_post_calibration_teleop(vr: Optional[VRJointData]) -> bool:
    """True for WBC leader frames after grip/squeeze calibration completes."""
    return (
        vr is not None
        and not bool(getattr(vr, "estop", True))
        and str(getattr(vr, "calib_stage", "")) == "teleop"
    )


def _finite_float(value: object, default: float = 0.0) -> float:
    """Best-effort finite float conversion for optional stream fields."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _encode_frames(frames: list[tuple[float, VRJointData]]) -> dict[str, np.ndarray]:
    """Pack ``(recv_mono, VRJointData)`` rows into fixed-shape arrays for storage."""
    if not frames:
        raise ValueError("cannot encode an empty frame list")
    n = len(frames)
    out: dict[str, np.ndarray] = {
        "recv_mono": np.asarray([m for m, _ in frames], dtype=np.float64),
        "timestamp_ns": np.asarray([int(vr.timestamp_ns) for _, vr in frames], dtype=np.int64),
        "calib_stage": np.asarray(
            [str(vr.calib_stage).encode()[:_STAGE_WIDTH] for _, vr in frames],
            dtype=f"S{_STAGE_WIDTH}",
        ),
    }
    for fld in _POSE_FIELDS:
        present = np.zeros(n, dtype=bool)
        data = np.zeros((n, 16), dtype=np.float64)
        for i, (_, vr) in enumerate(frames):
            val = getattr(vr, fld)
            if val:
                arr = np.asarray(val, dtype=np.float64)
                if arr.shape != (16,):
                    raise ValueError(f"{fld}[{i}] must be 16 floats, got shape {arr.shape}")
                present[i] = True
                data[i] = arr
        out[f"{fld}__present"] = present
        out[fld] = data
    for fld in _JOINT_FIELDS:
        present = np.asarray([bool(getattr(vr, fld)) for _, vr in frames], dtype=bool)
        if present.any():
            raise ValueError(
                f"{fld} is non-empty in the recorded stream: this replay path models the "
                "wbc_vr_leader (Cartesian EE targets, empty joint fields) only, not omni-vr."
            )
        out[f"{fld}__present"] = present
    for fld in _FLOAT_FIELDS:
        out[fld] = np.asarray(
            [_finite_float(getattr(vr, fld, 0.0)) for _, vr in frames],
            dtype=np.float64,
        )
    for fld in _BOOL_FIELDS:
        out[fld] = np.asarray([bool(getattr(vr, fld)) for _, vr in frames], dtype=bool)
    return out


def _decode_frames(arrays: dict[str, np.ndarray]) -> list[tuple[float, VRJointData]]:
    """Inverse of :func:`_encode_frames`: rebuild ``(recv_mono, VRJointData)`` rows."""
    recv = np.asarray(arrays["recv_mono"], dtype=np.float64)
    ts = np.asarray(arrays["timestamp_ns"], dtype=np.int64)
    stage = arrays["calib_stage"]
    n = len(recv)
    frames: list[tuple[float, VRJointData]] = []
    for i in range(n):
        kwargs: dict = {"timestamp_ns": int(ts[i])}
        stage_i = stage[i]
        kwargs["calib_stage"] = stage_i.decode() if isinstance(stage_i, bytes) else str(stage_i)
        for fld in _POSE_FIELDS:
            kwargs[fld] = (
                np.asarray(arrays[fld][i], dtype=float).tolist()
                if bool(arrays[f"{fld}__present"][i])
                else []
            )
        for fld in _JOINT_FIELDS:
            kwargs[fld] = []
        for fld in _FLOAT_FIELDS:
            kwargs[fld] = float(arrays[fld][i])
        for fld in _BOOL_FIELDS:
            kwargs[fld] = bool(arrays[fld][i])
        frames.append((float(recv[i]), VRJointData(**kwargs)))
    return frames


class StreamRecorder:
    """Accumulate received ``VRJointData`` (+ monotonic arrival time), write a replay file.

    Used by ``wbc_vr_record.py`` live mode: subscribe to ``vr/joints`` while the
    operator teleops in SAPIEN, append every frame after grip/squeeze calibration
    starts teleop, and persist on shutdown. The resulting file is replayed by
    ``wbc_vr_robot.py --replay``.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._frames: list[tuple[float, VRJointData]] = []

    def append(self, vr: VRJointData) -> None:
        """Record one received frame, stamped with the current monotonic time."""
        self._frames.append((self._clock(), vr))

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def duration(self) -> float:
        """Wall-clock span of the recording in seconds (0 if < 2 frames)."""
        if len(self._frames) < 2:
            return 0.0
        return self._frames[-1][0] - self._frames[0][0]

    def write(self, path: str, meta: Optional[dict] = None) -> int:
        """Write the recorded stream to an HDF5 file; returns the frame count."""
        if not self._frames:
            raise RuntimeError("no frames recorded; nothing to write")
        import h5py  # noqa: PLC0415 -- optional dep, only needed at write time

        arrays = _encode_frames(self._frames)
        with h5py.File(path, "w") as f:
            if meta:
                f.attrs.update(meta)
            f.attrs["n_frames"] = len(self._frames)
            f.attrs["schema"] = "wbc_vr_stream/v1"
            grp = f.create_group("frames")
            for key, val in arrays.items():
                grp.create_dataset(key, data=val, compression="gzip", compression_opts=4)
        return len(self._frames)


class ReplaySource:
    """Replay recorded ``VRJointData`` on a clock stretched by ``1/speed``.

    Mirrors the live ``VRJointSubscriber`` ``.latest`` so the follower loop is
    source-agnostic. ``.latest`` returns the most recent frame whose *scaled* release
    time has elapsed since :meth:`start`, advancing an internal cursor (None before
    the first frame is due). On each advance to a fresh frame, :attr:`segment_duration`
    is the scaled gap from the previous frame -- the interpolation-segment length the
    follower must use so the glide spans the stretched interval (not ``1/cmd-rate``),
    which is what makes slow replay actually slow instead of a staircase.
    """

    def __init__(
        self,
        frames: list[tuple[float, VRJointData]],
        speed: float,
        clock: Callable[[], float] = time.perf_counter,
        *,
        head_targets_pre_filtered: bool = False,
        replay_kind: str = "leader_stream",
    ) -> None:
        if not np.isfinite(speed) or speed <= 0.0:
            raise ValueError(f"speed must be finite and > 0, got {speed}")
        if not frames:
            raise ValueError("cannot replay an empty frame list")
        mono0 = frames[0][0]
        self.speed = float(speed)
        self._clock = clock
        self.head_targets_pre_filtered = bool(head_targets_pre_filtered)
        self.replay_kind = str(replay_kind)
        self._vr = [vr for _, vr in frames]
        # Scaled release offsets from start (monotonically non-decreasing).
        self._release = [max(0.0, (m - mono0) / self.speed) for m, _ in frames]
        self._t0: Optional[float] = None
        self._cursor = -1
        self.segment_duration: Optional[float] = None

    def start(self) -> None:
        """Anchor the replay clock at the current time."""
        self._t0 = self._clock()
        self._cursor = -1
        self.segment_duration = None

    @property
    def started(self) -> bool:
        """Whether :meth:`start` has been called (clock anchored)."""
        return self._t0 is not None

    @property
    def latest(self) -> Optional[VRJointData]:
        """Most recent frame whose scaled release time has elapsed (None if not yet)."""
        if self._t0 is None:
            self.start()
        elapsed = self._clock() - self._t0
        while self._cursor + 1 < len(self._vr) and self._release[self._cursor + 1] <= elapsed:
            prev = self._cursor
            self._cursor += 1
            self.segment_duration = (
                self._release[self._cursor] - self._release[prev] if prev >= 0 else None
            )
        return self._vr[self._cursor] if self._cursor >= 0 else None

    @property
    def done(self) -> bool:
        """True once the final frame has been released."""
        return (
            self._t0 is not None
            and self._cursor >= len(self._vr) - 1
            and (self._clock() - self._t0) >= self._release[-1]
        )

    @property
    def total_duration(self) -> float:
        """Total scaled replay length in seconds."""
        return self._release[-1]

    @classmethod
    def from_hdf5(
        cls, path: str, speed: float, clock: Callable[[], float] = time.perf_counter
    ) -> ReplaySource:
        """Load a recording written by :meth:`StreamRecorder.write` and build a source."""
        import h5py  # noqa: PLC0415 -- optional dep, only needed when replaying from a file

        with h5py.File(path, "r") as f:
            schema = f.attrs.get("schema", "")
            if schema != "wbc_vr_stream/v1":
                raise ValueError(
                    f"{path}: not a wbc_vr_stream/v1 recording (schema={schema!r})"
                )
            grp = f["frames"]
            arrays = {key: grp[key][()] for key in grp}
        return cls(
            _decode_frames(arrays),
            speed,
            clock,
            head_targets_pre_filtered=False,
            replay_kind="leader_stream",
        )

    @classmethod
    def from_episode_hdf5(
        cls,
        path: str,
        speed: float = 1.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> ReplaySource:
        """Build a source from an ``EpisodeRecorder`` take (``wbc_vr_robot.py --record``).

        Replays the take's own ``action/eef/{left,right}`` + ``action/head`` solver
        targets (see :func:`_decode_episode_frames`) at the recorded cadence, so the same
        operator intent can be re-driven through a re-tuned ``follower/wbik.yaml``.
        ``speed`` stretches that cadence exactly as it does for :meth:`from_hdf5` (the
        followers pass ``vr_teleop.replay_speed`` from their own config file; 1.0 = the
        take's real time).
        """
        import h5py  # noqa: PLC0415 -- optional dep, only needed when replaying from a file

        wanted = (
            _EPISODE_TIME_DATASET,
            *_EPISODE_POSE_DATASETS.values(),
            *_EPISODE_SCALAR_DATASETS.values(),
        )
        with h5py.File(path, "r") as f:
            missing = [key for key in wanted if key not in f]
            if missing:
                schema = f.attrs.get("schema", "")
                hint = (
                    " -- this is a wbc_vr_stream/v1 leader recording (wbc_vr_record.py); "
                    "the real-robot follower now replays its OWN episode takes"
                    if schema == "wbc_vr_stream/v1" else ""
                )
                raise ValueError(
                    f"{path}: not a wbc_vr_robot episode take, missing {missing}{hint}"
                )
            arrays = {key: f[key][()] for key in wanted}
        return cls(
            _decode_episode_frames(arrays),
            speed,
            clock,
            head_targets_pre_filtered=True,
            replay_kind="wbc_episode",
        )

    @classmethod
    def from_joystick_hdf5(
        cls,
        path: str,
        speed: float = 1.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> ReplaySource:
        """Replay either a raw joystick leader stream or a joystick episode take.

        Leader streams carry pre-filter headset targets and their published chassis
        command directly. Episode takes carry already-filtered Cartesian targets plus
        the explicitly recorded follower-effective joystick intent. The distinction is
        retained on ``head_targets_pre_filtered`` so the shared follower loop applies
        the head filter exactly once.
        """
        import h5py  # noqa: PLC0415 -- optional dep, only needed for replay

        with h5py.File(path, "r") as f:
            root_schema = f.attrs.get("schema", "")
            if isinstance(root_schema, bytes):
                root_schema = root_schema.decode("utf-8", errors="replace")
            if root_schema == "wbc_vr_stream/v1":
                is_stream = True
            else:
                is_stream = False
                def text(key: str) -> str | None:
                    if key not in f:
                        return None
                    value = np.asarray(f[key][()])
                    if value.shape != ():
                        raise ValueError(f"{path}: {key} must be scalar text")
                    scalar = value.item()
                    return (
                        bytes(scalar).decode("utf-8")
                        if isinstance(scalar, (bytes, np.bytes_))
                        else str(scalar)
                    )

                declarations = {
                    "meta/schema": "omniteleop_joystick_mobile_raw/v1",
                    "meta/control_mode": "joystick",
                    "meta/policy_action_schema": JOYSTICK_POLICY_ACTION_SCHEMA,
                    "meta/action_target_frame": "current_base",
                    "meta/eef_target_frame": "current_base",
                    "meta/head_target_frame": "current_base",
                }
                actual = {key: text(key) for key in declarations}
                if not any(
                    actual[key] == expected for key, expected in declarations.items()
                ):
                    raise ValueError(
                        f"{path}: expected a wbc_vr_stream/v1 file or joystick episode, "
                        f"got metadata {actual}"
                    )
                bad = [
                    f"{key}={actual[key]!r} (expected {expected!r})"
                    for key, expected in declarations.items()
                    if actual[key] != expected
                ]
                if bad:
                    raise ValueError(
                        f"{path}: incomplete/conflicting joystick replay contract: "
                        + "; ".join(bad)
                    )
                wanted = (
                    _EPISODE_TIME_DATASET,
                    *_EPISODE_POSE_DATASETS.values(),
                    *_EPISODE_SCALAR_DATASETS.values(),
                    _EPISODE_JOYSTICK_INTENT_DATASET,
                )
                missing = [key for key in wanted if key not in f]
                if missing:
                    raise ValueError(
                        f"{path}: joystick episode is missing required datasets {missing}"
                    )
                arrays = {key: f[key][()] for key in wanted}
        if is_stream:
            return cls.from_hdf5(path, speed, clock)
        return cls(
            _decode_episode_frames(
                arrays,
                chassis_dataset=_EPISODE_JOYSTICK_INTENT_DATASET,
            ),
            speed,
            clock,
            head_targets_pre_filtered=True,
            replay_kind="joystick_episode",
        )
