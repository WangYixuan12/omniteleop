#!/usr/bin/env python3
"""Whole-body VR leader: publish live EEF target poses for ``wbc_vr_record.py``.

1. reads Quest 3 controller + headset poses (WebXR, via ``WebXRVRReader``);
2. on a trigger-hold, captures the headset pose and computes two calibration
   transforms: a full head-pose transform for gaze tracking, and a gravity-aligned
   yaw+translation transform for wrist/EEF target positions;
3. each frame publishes the **Cartesian L_ee / R_ee target poses** on the
   ``vr/joints`` topic (``VRJointData``), exactly the field ``wbc_vr_record.py``
   consumes to drive its whole-body IK (base + torso + arms). The EEF target is:

       POSITION     yaw-only HEAD-RELATIVE (see ``_bprime_pos``): the hand rides the
                    robot's live head-yaw frame, so a head+hands turn keeps the hands
                    CENTERED in the camera (the base/camera yaw follows the head) while a
                    head-only look-around leaves them PLANTED in the room. A startup blend
                    ramps from the nominal EE at engage so the arm does not lunge.
       ORIENTATION  R = R_yaw @ vr_now_rot @ C: tracks the controller absolutely (so big
                    wrist rotations stay out of the arm's joint limits), with a constant
                    per-side tool-mount offset ``C`` (``--calibrate-ee-offset``). A fully
                    relative orientation was tried but contorts the wrist on large rolls
                    (scripts/diagnostics/decompose_rel_contortion.py).
4. each ``teleop`` frame also publishes the **calibrated headset pose**

       head_ee_pose[:3, :3] = R_yaw @ vr_headset[:3, :3] @ C_head
       head_ee_pose[:3,  3] = (robot_base_t_vr_base_eef @ vr_headset)[:3, 3]

   -- the head-frame (zed_depth_frame) teleop target. BOTH orientation and position
   use the gravity-aligned room calibration (``R_yaw = robot_base_t_vr_base_eef``):
   position so horizontal headset motion stays horizontal even though the nominal
   optical frame is pitched ~45 deg down, and orientation so an operator yaw about
   gravity stays a pure camera yaw instead of sweeping a tilted-axis cone (a yaw-only
   head turn otherwise leaks ~100 deg of pitch/roll -- the exact symptom the full-frame
   calibration produced). ``C_head`` is a constant body offset captured at calibration
   so the camera still starts at the nominal -45 deg down-look (head_ee_pose == nominal
   head FK at the calibration instant); this mirrors the EEF ``R_yaw @ vr @ C`` map.
   The follower either tracks this pose as a WBC head task (``head_mode: 'ik'``) or
   uses its orientation for dedicated pan/tilt head IK (``head_mode: 'track'``).

Commands are published at a low rate (``cmd_rate`` in ``follower/wbik.yaml``, default
**10 Hz**); the follower lerp/slerp-interpolates each command over ``1/cmd_rate`` up to
its 100 Hz IK ticks. The leader publishes at exactly this ``cmd_rate`` (loaded from the
same file), so leader and follower cannot drift. This mirrors the reference deps/rby1-wbc
split: 10 Hz ``trajectory_frequency_hz`` commands, 100 Hz ``ik_frequency_hz`` with
``use_interpolation: true``.

This preserves the ``vr_reader`` controller convention (controller forward 10 cm ->
target forward 10 cm in the calibrated base frame), with two deliberate changes:

  **Gravity-aligned calibration.** The nominal ``zed_depth_frame`` is pitched
  downward, so using the full head frame to calibrate targets would make a horizontal
  walk create a fake target-height change AND make an operator yaw sweep a tilted-axis
  cone (leaking pitch/roll into a yaw-only head turn). EEF and head targets therefore
  use headset yaw + translation only; the head orientation adds a constant body offset
  so it still starts at the nominal down-look (see step 4).

  **No legacy ``INIT_JOINT`` constants.** ``vr_reader`` anchors the calibration on
  ``FK(head_link)`` evaluated at ``omniteleop.common.vr_mode_const.INIT_*`` (the
  real-robot/dexcontrol-convention init posture). Here the calibration reference
  head pose comes from the **whole-body IK model's own nominal posture**
  (``VegaWholeBodyIK`` / ``WBCConfig.nominal_posture``), so the leader and the
  ``wbc_vr_record`` follower agree on the head/base frames and nothing depends on
  the legacy init constants.

Because the follower derives base motion from the EE targets, ``chassis_*`` is
published for schema completeness / other followers but is **ignored** by
``wbc_vr_record.py``. ``head_pos`` / ``*_arm_pos`` are left empty (the follower
computes all joints itself); ``head_ee_pose`` carries the Cartesian head target
from step 4 (empty while ``static``).

Run in the dexmate conda env (has pinocchio + pink + dexcomm + aiohttp/socketio)::

    # USB: adb reverse tcp:5067 tcp:5067, then open http://localhost:5067 in the
    #      Quest browser and tap "Start XR".
    /home/yixuan/miniforge3/envs/dexmate/bin/python scripts/wbc_vr_leader.py

    # ... and in another shell, the follower:
    /home/yixuan/miniforge3/envs/dexmate/bin/python scripts/wbc_vr_record.py

Controls (mirrors ``vr_reader``):
  * **hold right grip trigger >= 1 s** in ``static`` -> capture calibration and
    begin streaming targets (``teleop``);
  * index triggers -> ``left_gripper`` / ``right_gripper`` [0, 1];
  * **left X button** -> publish an exit request and stop the take (the headset HUD
    updates to ``Stage: stopped`` after the stop command is sent);
  * left thumbstick -> ``chassis_vx`` / ``chassis_vy``; right thumbstick x ->
    ``chassis_wz`` (published, but unused by ``wbc_vr_record``).
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from dexcomm import Node
from dexcomm.codecs import DictDataCodec
from scipy.spatial.transform import Rotation

from omniteleop.common import get_config
from omniteleop.common.schemas import VRJointData, WBCFollowerStatus
from omniteleop.follower.whole_body_ik import (
    HEAD_FRAME,
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    VegaWholeBodyIK,
    WBCConfig,
)
from omniteleop.leader import wbc_headset_hud as _wbc_headset_hud
from omniteleop.leader.communication.webxr_vr_reader import WebXRVRReader
from omniteleop.leader.wbc_headset_hud import (
    _DEFAULT_HUD_CAMERAS,
    HandAlignmentStatus,
    WBCHeadsetHUD,
)
from omniteleop.leader.wbc_reference_alignment import (
    REFERENCE_ALIGN_POS_TOL_MM,
    REFERENCE_ALIGN_ROT_TOL_DEG,
    REFERENCE_ALIGN_STABLE_S,
    ReferenceAlignmentGate,
    load_reference_ee_poses,
    optional_reference_path,
)
from omniteleop.wbc_teleop import VRTeleopConfig

# Command publish rate comes SOLELY from follower/wbik.yaml's vr_teleop.cmd_rate (no CLI
# flag), so the leader publishes at exactly the rate the followers interpolate up from --
# they cannot drift. The Vega URDF likewise comes from WBCConfig (wbik.yaml urdf_path).
_VR_TELEOP = VRTeleopConfig.from_yaml()
DEFAULT_RATE = _VR_TELEOP.cmd_rate  # leader publish rate = follower cmd_rate (Hz)

# Sentinel poses the WebXR reader emits (post-transform, in the robot frame) when a
# controller is not tracked. Copied from ``vr_reader.INVALID_{LEFT,RIGHT}_POSE`` so
# we can hold the last good target instead of snapping to garbage.
INVALID_LEFT_POSE = np.array(
    [[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)
INVALID_RIGHT_POSE = np.array(
    [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)

DEFAULT_CERT = "/home/yixuan/omniteleop/tests/cert.pem"
DEFAULT_KEY = "/home/yixuan/omniteleop/tests/key.pem"
_TRIGGER_PRESS = 0.7  # right grip value counted as a press
_TRACK_ATOL = 1e-6  # tolerance for matching the untracked sentinel poses
DEFAULT_HEADSET_HUD_RATE = 10.0
WBC_FOLLOWER_STATUS_TOPIC = "wbc/follower_status"
_follower_status_overlay_lines = _wbc_headset_hud.follower_status_overlay_lines
_follower_status_overlay_color = _wbc_headset_hud.follower_status_overlay_color

# Default file storing the per-side controller->gripper orientation offset (see
# WBCVRLeader._eef_target / --calibrate-ee-offset). Version-controlled with the repo so a
# dataset's mapping convention is reproducible; absent -> identity (raw controller frame).
DEFAULT_EE_OFFSET_FILE = str(
    Path(__file__).resolve().parents[1] / "src" / "omniteleop" / "leader" / "ee_offset.yaml"
)


def _to_mat(pose) -> np.ndarray:
    """4x4 matrix from a pin.SE3 (``.homogeneous``) or a 4x4 array."""
    return np.asarray(pose.homogeneous if hasattr(pose, "homogeneous") else pose, dtype=float)


def _rotz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _heading_yaw(pose: np.ndarray) -> float:
    """Yaw of a head/optical frame's viewing direction projected onto the ground."""
    rot = np.asarray(pose, dtype=float)[:3, :3]
    fwd_xy = rot[:2, 2]
    norm = float(np.linalg.norm(fwd_xy))
    if norm > 1e-8:
        return float(np.arctan2(fwd_xy[1], fwd_xy[0]))

    # Degenerate only when the viewing axis is almost vertical. The optical x-axis
    # is camera-right, so heading is 90 deg counter-clockwise from that projection.
    right_xy = rot[:2, 0]
    norm = float(np.linalg.norm(right_xy))
    if norm <= 1e-8:
        return 0.0
    return float(np.arctan2(right_xy[1], right_xy[0]) + np.pi / 2.0)


def _gravity_aligned_calibration(
    base_t_head_nominal: np.ndarray, vr_t_head_calib: np.ndarray
) -> np.ndarray:
    """Room-to-base transform for EEF targets, using yaw + translation only.

    The robot's nominal ``zed_depth_frame`` is pitched downward. Using the full
    head frame as the wrist calibration therefore pitches the entire VR room, so a
    horizontal operator walk creates a fake EEF height change. This calibration
    keeps gravity aligned: headset position maps to the nominal head position and
    headset yaw maps to nominal head yaw, while pitch/roll are left out of the
    room-to-base transform.
    """
    base_t_head_nominal = np.asarray(base_t_head_nominal, dtype=float)
    vr_t_head_calib = np.asarray(vr_t_head_calib, dtype=float)
    yaw = _heading_yaw(base_t_head_nominal) - _heading_yaw(vr_t_head_calib)
    rot = _rotz(yaw)
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = base_t_head_nominal[:3, 3] - rot @ vr_t_head_calib[:3, 3]
    return out


def _is_valid_pose(pose: np.ndarray) -> bool:
    """True if ``pose`` is a finite 4x4 with a valid ``[0, 0, 0, 1]`` bottom row.

    Guards against malformed/NaN frames: ``np.allclose(x, sentinel)`` is False for
    NaN, so without this a NaN pose would read as "tracked" and poison the last-good
    target (or, at calibration, the captured anchors -- and thus every later target).
    """
    pose = np.asarray(pose, dtype=float)
    return (
        pose.shape == (4, 4)
        and bool(np.all(np.isfinite(pose)))
        and bool(np.allclose(pose[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6))
    )


def _is_tracked(pose: np.ndarray, sentinel: np.ndarray) -> bool:
    """True only if ``pose`` is a valid 4x4 that isn't the untracked sentinel."""
    return _is_valid_pose(pose) and not np.allclose(pose, sentinel, atol=_TRACK_ATOL)


def _left_x_stop_requested(*, x_now: bool, prev_x: bool) -> bool:
    """True on the rising edge of the left X button."""
    return bool(x_now and not prev_x)


def _project_so3(rot: np.ndarray) -> np.ndarray:
    """Nearest proper rotation (det = +1) to ``rot`` via SVD orthonormalization."""
    u, _, vt = np.linalg.svd(np.asarray(rot, dtype=float))
    out = u @ vt
    if np.linalg.det(out) < 0.0:
        u = u.copy()
        u[:, -1] *= -1.0
        out = u @ vt
    return out


def _mean_rotation(mats: list[np.ndarray]) -> np.ndarray:
    """Chordal-L2 mean of a list of 3x3 SO(3) matrices (proper rotation average)."""
    if not mats:
        raise ValueError("cannot average an empty list of rotations")
    return Rotation.from_matrix(np.stack(mats)).mean().as_matrix()


def _load_ee_offsets(path: Optional[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-side controller->gripper offset rotations ``(C_left, C_right)`` from YAML.

    The file stores a scipy ``[x, y, z, w]`` quaternion per side under keys ``left`` and
    ``right``. A missing/None path -- or a side left unspecified -- yields identity: the
    gripper then inherits the controller's raw axis convention (the pre-calibration
    behavior). Each quaternion is normalized to a valid SO(3) matrix on load.
    """
    ident = (np.eye(3), np.eye(3))
    if not path or not Path(path).exists():
        return ident
    data = yaml.safe_load(Path(path).read_text()) or {}

    def _side(key: str) -> np.ndarray:
        q = data.get(key)
        if q is None:
            return np.eye(3)
        q = np.asarray(q, dtype=float)
        if q.shape != (4,):
            raise ValueError(
                f"ee_offset '{key}' must be a 4-vector quaternion [x, y, z, w]; got {q.shape}"
            )
        return Rotation.from_quat(q).as_matrix()

    return _side("left"), _side("right")


def _save_ee_offsets(path: str, c_left: np.ndarray, c_right: np.ndarray) -> None:
    """Write per-side offset rotations as ``[x, y, z, w]`` quaternions to a YAML file."""
    ql = Rotation.from_matrix(_project_so3(c_left)).as_quat().tolist()
    qr = Rotation.from_matrix(_project_so3(c_right)).as_quat().tolist()
    header = (
        "# Controller -> gripper orientation offset: a fixed body-frame 'tool-mount'\n"
        "# re-labeling captured by `wbc_vr_leader.py --calibrate-ee-offset`. Applied as\n"
        "#   R_target = R_yaw @ vr_controller_R @ C   (per side)\n"
        "# so the operator's NEUTRAL grip maps to the nominal L_ee/R_ee frame\n"
        "# (x:down, y:left, z:forward). Quaternions are scipy [x, y, z, w]. Delete this\n"
        "# file (or set a side to [0, 0, 0, 1]) to fall back to the raw controller frame.\n"
    )
    body = yaml.safe_dump({"left": ql, "right": qr}, default_flow_style=True, sort_keys=False)
    Path(path).write_text(header + body)


def _describe_axes(rot: np.ndarray) -> str:
    """Where each local axis of ``rot`` points in the base frame (x-fwd, y-left, z-up)."""
    names = {
        (1, 0, 0): "forward", (-1, 0, 0): "back",
        (0, 1, 0): "left", (0, -1, 0): "right",
        (0, 0, 1): "up", (0, 0, -1): "down",
    }
    out = []
    for ax, col in zip("xyz", np.asarray(rot, dtype=float).T, strict=True):
        key = tuple(int(c) for c in np.rint(col))
        out.append(f"{ax}:{names.get(key, 'oblique')}")
    return ", ".join(out)


class _VRLog:
    """Collect per-tick RAW VR poses and mapped targets.

    Diagnostic only: lets us measure the operator's hand pose RELATIVE TO the headset
    directly from the raw Quest stream -- the one thing the follower-side targets cannot
    show cleanly, because the leader maps the head (absolute) and the hands (relative to
    nominal) differently. ``raw_*`` are the untouched Quest poses (room frame); ``map_*``
    are the published targets, kept alongside so the mapping itself can be reproduced.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._rows: list[dict] = []

    def append(self, **fields) -> None:
        self._rows.append(fields)

    def __len__(self) -> int:
        return len(self._rows)

    def write(self) -> None:
        if not self._rows:
            return
        import h5py  # noqa: PLC0415 -- optional dep, only needed at teardown

        rows = self._rows
        col = lambda k: np.asarray([r[k] for r in rows])
        with h5py.File(self.path, "w") as f:
            g = f.create_group("vr")
            g.attrs["schema"] = "wbc_vr_leader_raw/v1"
            g.attrs["n_frames"] = len(rows)
            g.create_dataset("t", data=col("t"))
            g.create_dataset(
                "stage",
                data=np.array([str(r["stage"]).encode()[:16] for r in rows], dtype="S16"),
            )
            for key in ("raw_head", "raw_left", "raw_right",
                        "map_head", "map_left", "map_right"):
                g.create_dataset(key, data=col(key).astype(np.float32),
                                 compression="gzip", compression_opts=4)


class WBCVRLeader:
    """Reads Quest poses, calibrates once, and streams Cartesian EEF targets."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.rate = args.rate
        self.hold_seconds = args.hold_seconds
        self.stick_max_vx = args.stick_max_vx
        self.stick_max_vy = args.stick_max_vy
        self.stick_max_wz = args.stick_max_wz
        self.stick_deadzone = args.stick_deadzone
        # EEF target POSITION is yaw-only HEAD-RELATIVE (see _bprime_pos): the hand rides
        # the robot's live head-yaw frame, so a head+hands turn keeps the hands centered in
        # the camera (the base/camera yaw follows the head) while a head-only look-around
        # leaves them planted in the room. A startup blend (_hf_blend_dur) ramps from the
        # nominal EE at engage so the arm does not lunge. Verified offline + on hardware
        # (scripts/diagnostics/analyze_vr_hand_rel_head.py).
        self._hf_offset_left = np.zeros(3)
        self._hf_offset_right = np.zeros(3)
        self._hf_engage_t: Optional[float] = None
        self._hf_blend_dur = 0.5  # s, nominal -> B' engage ramp (transient, decays to zero)
        # Per-side constant controller->gripper orientation offset C, right-multiplied in
        # _eef_target (R = R_yaw @ vr_controller_R @ C). Identity unless --calibrate-ee-offset
        # wrote a file: identity makes the gripper inherit the controller's raw axis
        # convention; a calibrated C maps the neutral grip to the nominal L_ee/R_ee frame
        # (x:down, y:left, z:forward). Frozen on purpose (see _eef_target docstring).
        self._ee_offset_file = getattr(args, "ee_offset_file", None)
        self._ee_offset_left, self._ee_offset_right = _load_ee_offsets(self._ee_offset_file)
        if np.allclose(self._ee_offset_left, np.eye(3)) and np.allclose(
            self._ee_offset_right, np.eye(3)
        ):
            print("[wbc_vr_leader] EEF orientation offset: identity (gripper uses the raw "
                  "controller convention). Run --calibrate-ee-offset to map your neutral grip "
                  "to the nominal x:down,y:left,z:forward gripper frame.")
        else:
            ol = np.rad2deg(Rotation.from_matrix(self._ee_offset_left).magnitude())
            orr = np.rad2deg(Rotation.from_matrix(self._ee_offset_right).magnitude())
            print(f"[wbc_vr_leader] EEF orientation offset loaded from {self._ee_offset_file} "
                  f"(L {ol:.1f} deg, R {orr:.1f} deg from identity).")

        # Zenoh publisher on the same topic the follower subscribes to.
        self.node = Node(name="wbc_vr_leader", namespace=args.namespace)
        config = get_config()
        self.topic = config.get_topic("vr_joints", "vr/joints")
        self.pub = self.node.create_publisher(self.topic, encoder=DictDataCodec.encode)
        self._follower_status: Optional[WBCFollowerStatus] = None
        self._follower_status_t: Optional[float] = None
        self._status_lock = threading.Lock()
        self.status_topic = config.get_topic("wbc_follower_status", WBC_FOLLOWER_STATUS_TOPIC)
        self.status_sub = self.node.create_subscriber(
            self.status_topic, self._on_follower_status, decoder=DictDataCodec.decode
        )

        # Calibration reference head pose, from the whole-body IK model's *nominal*
        # posture (NOT the legacy INIT_JOINT constants). At nominal the planar base
        # is at the world origin, so frame_pose() values are already in the base
        # frame, and they match exactly what wbc_vr_record's VegaWholeBodyIK uses.
        cfg = WBCConfig()  # urdf_path comes from wbik.yaml (no --urdf override)
        ik = VegaWholeBodyIK(cfg)
        ik.reset()
        self.T_base_head = _to_mat(ik.frame_pose(HEAD_FRAME))
        if self.T_base_head.shape != (4, 4):
            raise ValueError(
                f"nominal head FK has shape {self.T_base_head.shape}, expected (4, 4)"
            )
        # Nominal EE poses (base frame) — only used to report the calibration delta.
        self._nominal_left = _to_mat(ik.frame_pose(LEFT_EE_FRAME))
        self._nominal_right = _to_mat(ik.frame_pose(RIGHT_EE_FRAME))
        self._align_reference_path = optional_reference_path(
            getattr(args, "align_reference", None)
        )
        self._align_gate: Optional[ReferenceAlignmentGate] = None
        self._last_alignment_status: Optional[HandAlignmentStatus] = None
        if self._align_reference_path is not None:
            ref = load_reference_ee_poses(self._align_reference_path, ik)
            assert ref is not None
            self._align_gate = ReferenceAlignmentGate(*ref)
            print(
                "[wbc_vr_leader] alignment reference loaded from "
                f"{self._align_reference_path} "
                f"(pos_tol={REFERENCE_ALIGN_POS_TOL_MM:g}mm, "
                f"rot_tol={REFERENCE_ALIGN_ROT_TOL_DEG:g}deg, "
                f"stable={REFERENCE_ALIGN_STABLE_S:g}s)."
            )
        del ik  # solver no longer needed; we stream Cartesian EE + head targets
        # (The head IK itself runs follower-side -- VegaWholeBodyIK.solve_head against
        # the live whole-body configuration; this leader maps the headset through
        # the full calibration while EEF targets use gravity-aligned yaw.)

        self.quest = WebXRVRReader(
            host=args.host, port=args.port,
            ssl_certfile=args.cert, ssl_keyfile=args.key,
        )
        self._hud: Optional[WBCHeadsetHUD] = (
            WBCHeadsetHUD(self.quest, args.hud_cameras) if args.headset_hud else None
        )
        self._hud_period = 1.0 / float(args.hud_rate) if args.hud_rate > 0.0 else float("inf")

        # State
        self.stage = "static"  # "static" (estop) or "teleop"
        # Constant headset->camera body offset C_head (3x3), captured at calibration so
        # the gravity-aligned head orientation map starts at the nominal zed_depth_frame
        # (-45 deg down-look). None until _calibrate. See _map_head_target.
        self._head_ori_offset: Optional[np.ndarray] = None
        self.robot_base_t_vr_base_eef: Optional[np.ndarray] = None
        self.last_left_target: Optional[np.ndarray] = None
        self.last_right_target: Optional[np.ndarray] = None
        self._trigger_start: Optional[float] = None
        self._prev_x = False
        self.running = False

        # Optional RAW-VR diagnostic log (--debug-vr DIR): per-tick raw headset +
        # controller poses (and the mapped targets) -> HDF5 on teardown. OFF by default.
        self._vrlog: Optional[_VRLog] = None
        debug_vr = getattr(args, "debug_vr", None)
        if debug_vr:
            import datetime  # noqa: PLC0415
            import os  # noqa: PLC0415

            os.makedirs(debug_vr, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._vrlog = _VRLog(os.path.join(debug_vr, f"vr_raw_{stamp}.hdf5"))
            print(f"[wbc_vr_leader] RAW-VR debug log -> {self._vrlog.path}")

    # -- helpers ----------------------------------------------------------------

    def _on_follower_status(self, data: dict) -> None:
        try:
            status = WBCFollowerStatus(**data)
        except (TypeError, ValueError):
            return
        with self._status_lock:
            self._follower_status = status
            self._follower_status_t = time.perf_counter()

    def _follower_status_snapshot(self) -> tuple[Optional[WBCFollowerStatus], Optional[float]]:
        with self._status_lock:
            status = self._follower_status
            received_t = self._follower_status_t
        age = None if received_t is None else time.perf_counter() - received_t
        return status, age

    def _send_stop_hud(self) -> None:
        """Show a final stop state in the headset HUD after the stop frame is published."""
        if self._hud is None:
            return
        status, status_age_s = self._follower_status_snapshot()
        self._hud.poll_and_send(stage="stop", status=status, status_age_s=status_age_s)

    def _trigger_held(self, transforms) -> bool:
        """True once the right grip trigger is continuously held for hold_seconds."""
        val = float(transforms["right_hand_trigger"])
        now = time.perf_counter()
        if val > _TRIGGER_PRESS:
            if self._trigger_start is None:
                self._trigger_start = now
            elif now - self._trigger_start >= self.hold_seconds:
                self._trigger_start = None
                return True
        else:
            self._trigger_start = None
        return False

    def _thumbstick_to_chassis(self, transforms) -> tuple[float, float, float]:
        def dz(v: float) -> float:
            return v if abs(v) > self.stick_deadzone else 0.0

        left = transforms["left_thumbstick"]
        right = transforms["right_thumbstick"]
        # Quest stick y is negative when pushed up -> negate so forward is positive.
        vx = -dz(left[1]) * self.stick_max_vx
        vy = -dz(left[0]) * self.stick_max_vy
        wz = -dz(right[0]) * self.stick_max_wz
        return vx, vy, wz

    def _calibrate(self, headset_pose: np.ndarray) -> None:
        """Anchor room-to-base transforms so the headset maps onto the nominal head."""
        headset_pose = np.asarray(headset_pose, dtype=float)
        if headset_pose.shape != (4, 4):
            raise ValueError(f"headset pose has shape {headset_pose.shape}, expected (4, 4)")
        self.robot_base_t_vr_base_eef = _gravity_aligned_calibration(
            self.T_base_head, headset_pose
        )
        # Head orientation rides the SAME gravity-aligned yaw R_yaw as the position/EEF
        # map (so operator yaw about gravity stays a pure camera yaw -- no tilted-axis
        # cone), plus a constant body offset C_head that re-labels the calibration
        # headset orientation onto the nominal zed_depth_frame. Then
        #   head_R = R_yaw @ vr_head_R @ C_head,
        # and at calibration R_yaw @ vr_calib_R @ C_head == T_base_head (the camera
        # starts at the nominal -45 deg down-look). Mirrors the EEF C-offset structure.
        r_yaw = self.robot_base_t_vr_base_eef[:3, :3]
        self._head_ori_offset = (r_yaw @ headset_pose[:3, :3]).T @ self.T_base_head[:3, :3]

    def _map_head_target(self, vr_head: np.ndarray) -> np.ndarray:
        """Calibrated head target: the headset pose mapped into the robot base frame.

        Both orientation and position ride the gravity-aligned room calibration
        ``robot_base_t_vr_base_eef`` (yaw + translation), so neither couples gravity
        into the wrong axis:

          * POSITION: ``head_mode: "ik"`` makes this translation a real whole-body
            target; mapping it through the downward-pitched optical frame would turn a
            horizontal headset move into a height command.
          * ORIENTATION: ``head_R = R_yaw @ vr_head_R @ C_head``. Because ``R_yaw`` is a
            pure gravity yaw, an operator yaw about gravity stays a pure CAMERA yaw. The
            earlier full-frame map (``robot_base_t_vr_base @ vr_head``) instead rotated
            the camera about the optical frame's ~45-deg-down axis, so a yaw-only head
            turn swept a cone and leaked ~100 deg of pitch/roll. ``C_head`` (captured in
            _calibrate) is constant, so it preserves that no-coupling property while
            re-anchoring the start at the nominal down-look.

        At the calibration instant the pose equals the nominal head FK by construction.
        """
        assert self._head_ori_offset is not None
        assert self.robot_base_t_vr_base_eef is not None
        vr_head = np.asarray(vr_head, dtype=float)
        if vr_head.shape != (4, 4):
            raise ValueError(f"headset pose has shape {vr_head.shape}, expected (4, 4)")
        r_yaw = self.robot_base_t_vr_base_eef
        out = np.eye(4)
        out[:3, :3] = r_yaw[:3, :3] @ vr_head[:3, :3] @ self._head_ori_offset
        out[:3, 3] = (r_yaw @ vr_head)[:3, 3]
        return out

    def _bprime_pos(
        self, side: str, vr_now: np.ndarray, vr_head: np.ndarray, head_target: np.ndarray
    ) -> np.ndarray:
        """Yaw-only head-relative hand POSITION (base frame) -- the EEF target position map.

        Transfers the operator's hand-position-relative-to-head onto the robot's LIVE
        head-yaw frame, using only the head GRAVITY YAW (``_heading_yaw``), not its
        downward-pitched optical pose::

            x_yaw     = Rz(-yaw_room) @ (vr_now_pos - vr_head_pos)   # hand rel head, yaw frame
            pos_base  = head_target_pos + Rz(yaw_base) @ x_yaw

        For a head+hands turn (hand rigid relative to head) ``x_yaw`` is constant, so the
        hand stays fixed in the head-yaw frame and rides the base yaw -> CENTERED in the
        camera. For a head-only look-around (hand fixed in the room) ``x_yaw`` counter-
        rotates with the head, so ``pos_base`` stays world-fixed -> PLANTED. Using the
        FULL head pose instead (``head_target @ inv(vr_head) @ vr_now``) re-introduces the
        camera pitch/relabel and swings the arm with the gaze, which is why only the yaw is
        transferred (verified offline, scripts/diagnostics/analyze_vr_hand_rel_head.py).
        """
        yaw_room = _heading_yaw(vr_head)
        yaw_base = _heading_yaw(head_target)
        x_yaw = _rotz(-yaw_room) @ (vr_now[:3, 3] - vr_head[:3, 3])
        return head_target[:3, 3] + _rotz(yaw_base) @ x_yaw

    def _capture_head_frame_anchors(
        self, vr_left: np.ndarray, vr_right: np.ndarray, vr_head: np.ndarray,
        head_target: np.ndarray, now: float,
    ) -> None:
        """Capture the engage state for the startup blend.

        ``_bprime_pos`` is absolute-style (it places the hand at the operator's actual
        hand-rel-head pose), so at engage it sits ~10-20 cm off ``nominal_ee``. A CONSTANT
        anchor to nominal would re-break one of the two behaviors (a world-frame constant
        sweeps in the rotating camera; a head-frame constant swings on look-around -- both
        verified offline), so instead the offset is ramped to zero over ``_hf_blend_dur``
        in ``_eef_target``: a transient that decays, leaving the steady map exactly B'.
        """
        self._hf_offset_left = (
            self._bprime_pos("left", vr_left, vr_head, head_target) - self._nominal_left[:3, 3]
        )
        self._hf_offset_right = (
            self._bprime_pos("right", vr_right, vr_head, head_target) - self._nominal_right[:3, 3]
        )
        self._hf_engage_t = now

    def _eef_target(self, side: str, vr_now: np.ndarray, *,
                    vr_head: Optional[np.ndarray] = None,
                    head_target: Optional[np.ndarray] = None,
                    now: float = 0.0) -> np.ndarray:
        """Map one controller pose to its base-frame EEF target.

        ``side`` (``"left"``/``"right"``) selects the per-arm nominal EE pose and engage
        offset. Requires valid head geometry (``vr_head`` + ``head_target``).

        POSITION is yaw-only head-relative (``_bprime_pos``): the hand rides the robot's
        live head-yaw frame, so a head+hands turn keeps the hands centered in the camera
        while a head-only look-around leaves them planted. ``_bprime_pos`` is absolute-style
        (it places the hand at the operator's actual hand-rel-head pose), so at engage it
        sits ~10-20 cm off ``nominal_ee``; the per-side engage offset ``_hf_offset_*`` is
        ramped to zero over ``_hf_blend_dur`` so the arm does not lunge (a transient that
        decays, leaving the steady map exactly B').

        ORIENTATION is ``R_yaw @ vr_now_rot @ C`` (``R_yaw = robot_base_t_vr_base_eef``):
        it tracks the operator's actual hand orientation -- which they servo by sight -- so
        a large wrist rotation does NOT drive the arm into joint limits. A purely relative
        orientation (``nominal_ee_rot @ vr0_rot.T @ vr_now_rot``) was tried first but
        contorts the wrist on big rolls (scripts/diagnostics/decompose_rel_contortion.py).

        The fixed per-side offset ``C`` (``self._ee_offset_left``/``_right``; identity unless
        ``--calibrate-ee-offset`` wrote a file) is right-multiplied onto the orientation. It
        is a constant body-frame re-labeling of the controller axes onto the gripper, so the
        operator's neutral grip reads as the nominal ``L_ee``/``R_ee`` frame (x:down, y:left,
        z:forward). Being CONSTANT it preserves the non-contorting property; recomputing
        ``C`` from the engage pose each frame would instead reproduce the rejected relative
        orientation (deviation tied to rotation-since-engage).
        """
        if vr_head is None or head_target is None:
            raise ValueError("_eef_target requires vr_head and head_target")
        # Fixed controller->gripper "tool-mount" offset (identity unless calibrated).
        ee_offset = getattr(self, f"_ee_offset_{side}", None)
        if ee_offset is None:
            ee_offset = np.eye(3)
        # POSITION: yaw-only head-relative (_bprime_pos), with the per-side engage offset
        # ramped to zero over _hf_blend_dur so the arm does not lunge at engage.
        off = self._hf_offset_left if side == "left" else self._hf_offset_right
        blend = 1.0 if self._hf_engage_t is None else float(
            np.clip((now - self._hf_engage_t) / self._hf_blend_dur, 0.0, 1.0)
        )
        out = np.eye(4)
        out[:3, 3] = self._bprime_pos(side, vr_now, vr_head, head_target) - (1.0 - blend) * off
        out[:3, :3] = self.robot_base_t_vr_base_eef[:3, :3] @ vr_now[:3, :3] @ ee_offset
        return out

    def _map_targets(self, vr_l: np.ndarray, vr_r: np.ndarray, *,
                     vr_head: Optional[np.ndarray] = None,
                     head_target: Optional[np.ndarray] = None,
                     now: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Map controller poses to base-frame EEF targets, holding last-good if untracked.

        The hand POSITION rides the live head-yaw frame, so on a head-tracking loss
        (``head_target`` None) hold the last-good hand targets rather than updating them
        from a stale/invalid head (Codex review).
        """
        assert self.robot_base_t_vr_base_eef is not None
        if head_target is None:
            return self.last_left_target, self.last_right_target
        if _is_tracked(vr_l, INVALID_LEFT_POSE):
            self.last_left_target = self._eef_target(
                "left", vr_l, vr_head=vr_head, head_target=head_target, now=now)
        if _is_tracked(vr_r, INVALID_RIGHT_POSE):
            self.last_right_target = self._eef_target(
                "right", vr_r, vr_head=vr_head, head_target=head_target, now=now)
        return self.last_left_target, self.last_right_target

    def _publish(
        self,
        left_target: Optional[np.ndarray],
        right_target: Optional[np.ndarray],
        head_target: Optional[np.ndarray],
        left_gripper: float,
        right_gripper: float,
        chassis: tuple[float, float, float],
        *,
        exit_requested: bool = False,
    ) -> None:
        left_flat = left_target.reshape(-1).tolist() if left_target is not None else []
        right_flat = right_target.reshape(-1).tolist() if right_target is not None else []
        head_flat = head_target.reshape(-1).tolist() if head_target is not None else []
        data = VRJointData(
            timestamp_ns=time.time_ns(),
            head_pos=[],
            left_arm_pos=[],
            right_arm_pos=[],
            left_gripper=float(left_gripper),
            right_gripper=float(right_gripper),
            chassis_vx=float(chassis[0]),
            chassis_vy=float(chassis[1]),
            chassis_wz=float(chassis[2]),
            estop=(self.stage == "static"),
            exit_requested=bool(exit_requested),
            calib_stage=self.stage,
            left_ee_pose=left_flat,
            right_ee_pose=right_flat,
            head_ee_pose=head_flat,
        )
        # Publisher was created with encoder=DictDataCodec.encode, so pass the
        # plain dict and let it encode (mirrors vr_reader._publish).
        self.pub.publish(asdict(data))

    # -- main loop --------------------------------------------------------------

    def run(self) -> None:
        """Main loop: poll the Quest, calibrate on a grip-hold, then stream EEF targets."""
        self.running = True
        self.quest.start()
        print("[wbc_vr_leader] waiting for Quest data ...")
        self.quest.wait_for_data()
        print(f"[wbc_vr_leader] Quest connected. Publishing on '{self.topic}'.")
        print("[wbc_vr_leader] Stage static — hold right grip trigger "
              f">= {self.hold_seconds:.0f}s to calibrate and start streaming targets.")

        dt = 1.0 / self.rate
        last_print = 0.0
        last_hud = -float("inf")
        t0 = time.perf_counter()
        try:
            while self.running:
                now = time.perf_counter()
                transforms = self.quest.get_latest_transformation()
                if transforms is None:
                    time.sleep(dt)
                    continue

                # left X -> publish one final exit frame so recorders can stop/save.
                x_now = bool(transforms["left_x_button"])
                if _left_x_stop_requested(x_now=x_now, prev_x=self._prev_x):
                    self._prev_x = x_now
                    # STOP FIRST: the exit frame is what halts the robot (the follower's
                    # run_loop stops motion the instant it sees vr.exit_requested). Publish
                    # it before any HUD work so nothing below can delay the robot stop.
                    self._publish(
                        self.last_left_target, self.last_right_target, None,
                        transforms["left_index_trigger"], transforms["right_index_trigger"],
                        self._thumbstick_to_chassis(transforms),
                        exit_requested=True,
                    )
                    print("\n[wbc_vr_leader] stop requested (left X).")
                    # THEN reflect the stop in the headset HUD. This runs only after the
                    # stop command is already out, so it adds NO latency to stopping the
                    # robot. The brief sleep lets the async camera_frame emit flush to the
                    # Quest before close() stops the server loop -- otherwise the queued
                    # emit is dropped and the operator never sees the updated stage.
                    self.stage = "stopped"
                    if self._hud is not None:
                        self._send_stop_hud()
                        time.sleep(0.2)
                    break
                self._prev_x = x_now

                vr_l = transforms["left_wrist"]
                vr_r = transforms["right_wrist"]

                if self.stage == "static":
                    if self._trigger_held(transforms):
                        if not (_is_valid_pose(transforms["head"])
                                and _is_tracked(vr_l, INVALID_LEFT_POSE)
                                and _is_tracked(vr_r, INVALID_RIGHT_POSE)):
                            print("\n[wbc_vr_leader] headset + both controllers must be "
                                  "tracked to calibrate — hold them in view and retry.")
                        else:
                            self._calibrate(transforms["head"])
                            # head_target must be ready BEFORE mapping hands: the hand
                            # position rides the live head-yaw frame.
                            head_target = self._map_head_target(transforms["head"])
                            self._capture_head_frame_anchors(
                                vr_l, vr_r, transforms["head"], head_target, now)
                            self._map_targets(  # seed last-good targets (starts at nominal)
                                vr_l, vr_r, vr_head=transforms["head"],
                                head_target=head_target, now=now)
                            d = self.last_left_target[:3, 3] - self._nominal_left[:3, 3]
                            if self._align_gate is not None:
                                self.stage = "align"
                                self._align_gate.reset()
                                self._last_alignment_status = (
                                    self._align_gate.status(
                                        self.last_left_target, self.last_right_target, now=now
                                    )
                                )
                                print(f"\n[wbc_vr_leader] calibrated -> align. "
                                      f"L target pos offset from nominal = "
                                      f"[{d[0]:+.3f} {d[1]:+.3f} {d[2]:+.3f}] m")
                            else:
                                self.stage = "teleop"
                                print(f"\n[wbc_vr_leader] calibrated -> teleop. "
                                      f"L target pos offset from nominal = "
                                      f"[{d[0]:+.3f} {d[1]:+.3f} {d[2]:+.3f}] m")
                    if self.stage == "static":
                        self._publish(None, None, None, 0.0, 0.0, (0.0, 0.0, 0.0))
                elif self.stage == "align":
                    head_target = (
                        self._map_head_target(transforms["head"])
                        if _is_valid_pose(transforms["head"]) else None
                    )
                    left_target, right_target = self._map_targets(
                        vr_l, vr_r, vr_head=transforms["head"],
                        head_target=head_target, now=now)
                    assert self._align_gate is not None
                    alignment = self._align_gate.status(
                        left_target, right_target, now=now
                    )
                    self._last_alignment_status = alignment
                    if alignment is not None and alignment.ready:
                        self.stage = "teleop"
                        print("\n[wbc_vr_leader] reference aligned -> teleop.")
                    self._publish(
                        left_target, right_target, head_target,
                        transforms["left_index_trigger"], transforms["right_index_trigger"],
                        self._thumbstick_to_chassis(transforms),
                    )
                elif self.stage == "teleop":
                    # Head target FIRST: the hands are mapped relative to the live head-yaw
                    # frame, so they need head_target this tick. Hold-last-good on a
                    # malformed/NaN headset frame -> empty head target (the follower keeps
                    # its previous head command rather than rejecting a NaN pose), and
                    # _map_targets then also holds the last-good hands.
                    head_target = (
                        self._map_head_target(transforms["head"])
                        if _is_valid_pose(transforms["head"]) else None
                    )
                    left_target, right_target = self._map_targets(
                        vr_l, vr_r, vr_head=transforms["head"],
                        head_target=head_target, now=now)
                    chassis = self._thumbstick_to_chassis(transforms)
                    self._publish(
                        left_target, right_target, head_target,
                        transforms["left_index_trigger"], transforms["right_index_trigger"],
                        chassis,
                    )
                    if self._vrlog is not None:
                        nanmat = np.full((4, 4), np.nan)
                        self._vrlog.append(
                            t=now - t0, stage=self.stage,
                            raw_head=np.asarray(transforms["head"], dtype=float),
                            raw_left=np.asarray(vr_l, dtype=float),
                            raw_right=np.asarray(vr_r, dtype=float),
                            map_head=head_target if head_target is not None else nanmat,
                            map_left=left_target if left_target is not None else nanmat,
                            map_right=right_target if right_target is not None else nanmat,
                        )

                if self._hud is not None and now - last_hud >= self._hud_period:
                    status, status_age_s = self._follower_status_snapshot()
                    self._hud.poll_and_send(
                        stage=self.stage,
                        status=status,
                        status_age_s=status_age_s,
                        alignment=(
                            self._last_alignment_status if self.stage == "align" else None
                        ),
                    )
                    last_hud = now

                t = now - t0
                if t - last_print >= 0.5:
                    last_print = t
                    if self.stage in ("align", "teleop") and self.last_left_target is not None:
                        lp = self.last_left_target[:3, 3]
                        rp = self.last_right_target[:3, 3]
                        print(f"t={t:6.1f}s  {self.stage:6s}  "
                              f"L=[{lp[0]:+.3f} {lp[1]:+.3f} {lp[2]:+.3f}]  "
                              f"R=[{rp[0]:+.3f} {rp[1]:+.3f} {rp[2]:+.3f}] (base frame)",
                              end="\r")
                    else:
                        print(f"t={t:6.1f}s  static (hold right grip to calibrate)    ",
                              end="\r")

                sleep = dt - (time.perf_counter() - now)
                if sleep > 0:
                    time.sleep(sleep)
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.quest.close()
            self.node.shutdown()
            if self._vrlog is not None and len(self._vrlog) > 0:
                self._vrlog.write()
                print(f"[wbc_vr_leader] RAW-VR log -> {self._vrlog.path} "
                      f"({len(self._vrlog)} frames)")
            print("\n[wbc_vr_leader] stopped.")

    # -- calibration helper -----------------------------------------------------

    def calibrate_ee_offset(
        self, capture_seconds: float, out_file: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Capture the neutral-grip controller orientation and save the offset ``C``.

        Hold BOTH controllers in the grip you want to read as the gripper's nominal frame
        (x:down, y:left, z:forward) -- typically pointing forward, fingers down -- while
        facing the robot's forward, then hold the right grip trigger for ``hold_seconds``.
        We average the controller orientations over ``capture_seconds``, compute
        ``C_side = (R_yaw @ vr_ref_R)^T @ nom_R`` per side, project to SO(3), write them to
        ``out_file``, and print residuals so you can confirm the result before trusting it.
        Returns ``(C_left, C_right)``.
        """
        if not np.isfinite(capture_seconds) or capture_seconds <= 0:
            raise ValueError(f"capture_seconds must be finite and > 0, got {capture_seconds}")
        self.quest.start()
        print("[calibrate-ee-offset] waiting for Quest data ...")
        self.quest.wait_for_data()
        print("[calibrate-ee-offset] Hold BOTH controllers in your NEUTRAL grip (the pose you\n"
              "  want to read as gripper x:down,y:left,z:forward), face the robot's forward,\n"
              f"  and hold the right grip trigger >= {self.hold_seconds:g}s to capture.")
        try:
            # 1) wait for a trigger-hold with the headset + both controllers tracked.
            while True:
                transforms = self.quest.get_latest_transformation()
                if transforms is None:
                    time.sleep(1.0 / self.rate)
                    continue
                if self._trigger_held(transforms):
                    if not (_is_valid_pose(transforms["head"])
                            and _is_tracked(transforms["left_wrist"], INVALID_LEFT_POSE)
                            and _is_tracked(transforms["right_wrist"], INVALID_RIGHT_POSE)):
                        print("[calibrate-ee-offset] headset + both controllers must be tracked "
                              "-- hold them in view and retry.")
                        continue
                    break
            # 2) average the orientations over the capture window.
            heads, lefts, rights = [], [], []
            t_end = time.perf_counter() + capture_seconds
            while time.perf_counter() < t_end:
                transforms = self.quest.get_latest_transformation()
                if transforms is None:
                    time.sleep(1.0 / self.rate)
                    continue
                head = transforms["head"]
                vr_l, vr_r = transforms["left_wrist"], transforms["right_wrist"]
                if (_is_valid_pose(head)
                        and _is_tracked(vr_l, INVALID_LEFT_POSE)
                        and _is_tracked(vr_r, INVALID_RIGHT_POSE)):
                    heads.append(np.asarray(head, dtype=float))
                    lefts.append(np.asarray(vr_l, dtype=float)[:3, :3])
                    rights.append(np.asarray(vr_r, dtype=float)[:3, :3])
                time.sleep(1.0 / self.rate)
            if len(lefts) < 3:
                raise RuntimeError(
                    f"captured only {len(lefts)} tracked frames; hold steadier/longer and retry."
                )
        finally:
            self.quest.close()

        # 3) gravity-aligned yaw from the averaged headset; mean controller orientations.
        head_avg = np.eye(4)
        head_avg[:3, :3] = _mean_rotation([h[:3, :3] for h in heads])
        head_avg[:3, 3] = np.mean([h[:3, 3] for h in heads], axis=0)
        rot_yaw = _gravity_aligned_calibration(self.T_base_head, head_avg)[:3, :3]
        vr_ref_l, vr_ref_r = _mean_rotation(lefts), _mean_rotation(rights)

        # 4) C maps the neutral grip to nominal: R_yaw @ vr_ref @ C == nom_R.
        c_left = _project_so3((rot_yaw @ vr_ref_l).T @ self._nominal_left[:3, :3])
        c_right = _project_so3((rot_yaw @ vr_ref_r).T @ self._nominal_right[:3, :3])

        # 5) report jitter (grip steadiness), offset magnitude, and the residual + resulting
        #    gripper axes for the neutral grip (should land exactly on the nominal frame).
        def _jitter(ref, mats):
            return np.rad2deg(
                (Rotation.from_matrix(ref).inv() * Rotation.from_matrix(np.stack(mats))).magnitude()
            )

        def _resid(ref, offset, nom):
            got = rot_yaw @ ref @ offset
            ang = np.rad2deg(
                (Rotation.from_matrix(nom).inv() * Rotation.from_matrix(got)).magnitude()
            )
            return got, float(ang)

        jit_l, jit_r = _jitter(vr_ref_l, lefts), _jitter(vr_ref_r, rights)
        got_l, res_l = _resid(vr_ref_l, c_left, self._nominal_left[:3, :3])
        got_r, res_r = _resid(vr_ref_r, c_right, self._nominal_right[:3, :3])
        off_l = np.rad2deg(Rotation.from_matrix(c_left).magnitude())
        off_r = np.rad2deg(Rotation.from_matrix(c_right).magnitude())
        print(f"\n[calibrate-ee-offset] captured {len(lefts)} frames over {capture_seconds:g}s")
        print(f"  grip jitter (max):   L={jit_l.max():5.2f} deg  R={jit_r.max():5.2f} deg "
              "(re-do if large -- hold steadier)")
        print(f"  offset magnitude:    L={off_l:5.1f} deg  R={off_r:5.1f} deg "
              "(controller<->gripper convention gap)")
        print(f"  neutral grip -> L gripper frame: {_describe_axes(got_l)} "
              f"(residual {res_l:.2f} deg)")
        print(f"  neutral grip -> R gripper frame: {_describe_axes(got_r)} "
              f"(residual {res_r:.2f} deg)")
        if max(res_l, res_r) > 1.0:
            print("  WARNING: residual > 1 deg -- C did not land on nominal; check inputs.")

        # 6) persist (version-controlled) so every later run auto-loads the same convention.
        _save_ee_offsets(out_file, c_left, c_right)
        print(f"[calibrate-ee-offset] wrote {out_file}. Re-run the leader (it auto-loads this "
              "file) to teleop with the gripper in the x:down,y:left,z:forward convention.")
        return c_left, c_right


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--namespace", default="",
                        help="Zenoh namespace (must match the follower; default empty).")
    parser.add_argument("--host", default="0.0.0.0", help="WebXR bind host (default 0.0.0.0).")
    parser.add_argument("--port", type=int, default=5067, help="WebXR port (default 5067).")
    parser.add_argument("--cert", default=DEFAULT_CERT, help="TLS cert for the WebXR server.")
    parser.add_argument("--key", default=DEFAULT_KEY, help="TLS key for the WebXR server.")
    parser.add_argument("--hold-seconds", type=float, default=1.0,
                        help="right-grip hold time to calibrate/start (default 1.0).")
    parser.add_argument("--stick-max-vx", type=float, default=0.3)
    parser.add_argument("--stick-max-vy", type=float, default=0.2)
    parser.add_argument("--stick-max-wz", type=float, default=0.5)
    parser.add_argument("--stick-deadzone", type=float, default=0.1,
                        help="per-axis thumbstick deadzone applied independently before "
                             "publishing chassis_vx, chassis_vy, and chassis_wz "
                             "(default 0.1).")
    parser.add_argument("--headset-hud", action=argparse.BooleanOptionalAction, default=True,
                        help="show the robot camera/status HUD in the Quest headset "
                             "(default: enabled). Use --no-headset-hud to disable.")
    parser.add_argument("--hud-rate", type=float, default=DEFAULT_HEADSET_HUD_RATE,
                        help=f"headset HUD refresh rate in Hz (default "
                             f"{DEFAULT_HEADSET_HUD_RATE:g}).")
    parser.add_argument("--hud-cameras", nargs="+", choices=(
        "head_left_rgb", "head_right_rgb", "head_depth", "left_wrist_rgb", "right_wrist_rgb"
    ), default=list(_DEFAULT_HUD_CAMERAS),
                        help="camera streams to read for the headset HUD. RGB streams are "
                             "shown side-by-side; head_depth is ignored in the preview to "
                             "match vr_reader.py.")
    parser.add_argument("--align-reference", default=None,
                        help="optional raw-data episode HDF5 whose last obs/joint frame "
                             "defines the reference L/R EEF pose. When set, the leader "
                             "enters an align stage after calibration and only switches to "
                             "teleop after the current hand targets match that reference "
                             "within the fixed reference-alignment tolerances. Omit or "
                             "pass 'None' to disable.")
    parser.add_argument("--ee-offset-file", default=DEFAULT_EE_OFFSET_FILE,
                        help="YAML of per-side controller->gripper orientation offsets (scipy "
                             "[x,y,z,w] quats). Auto-loaded at startup; missing -> identity (raw "
                             "controller convention). Written by --calibrate-ee-offset. "
                             f"Default: {DEFAULT_EE_OFFSET_FILE}.")
    parser.add_argument("--calibrate-ee-offset", action="store_true",
                        help="One-time helper: hold both controllers in your neutral grip and "
                             "hold the right grip trigger to capture the offset C that maps the "
                             "neutral grip to the gripper's nominal x:down,y:left,z:forward frame, "
                             "write it to --ee-offset-file, and exit (does not stream targets).")
    parser.add_argument("--ee-capture-seconds", type=float, default=1.5,
                        help="averaging window for --calibrate-ee-offset (default 1.5).")
    parser.add_argument("--debug-vr", dest="debug_vr", default=None,
                        help="write a per-tick RAW-VR diagnostic HDF5 (raw headset + both "
                             "controller poses, plus the mapped targets) under this directory "
                             "as vr_raw_<timestamp>.hdf5. Lets us measure hand-relative-to-head "
                             "motion directly from the Quest stream. OFF by default.")
    args = parser.parse_args()
    # Publish rate comes SOLELY from wbik.yaml's vr_teleop.cmd_rate (the rate the follower
    # interpolates up from), so leader and follower cannot drift; the URDF comes from
    # WBCConfig's urdf_path. Neither is a CLI flag. Bind cmd_rate onto args (the established
    # post-parse pattern).
    args.rate = DEFAULT_RATE
    if args.rate <= 0:
        raise ValueError(
            f"wbik.yaml vr_teleop.cmd_rate must be > 0 for the leader, got {args.rate}"
        )
    if args.headset_hud and (not np.isfinite(args.hud_rate) or args.hud_rate <= 0.0):
        parser.error("--hud-rate must be finite and > 0 when --headset-hud is enabled")

    leader = WBCVRLeader(args)
    if args.calibrate_ee_offset:
        leader.calibrate_ee_offset(args.ee_capture_seconds, args.ee_offset_file)
    else:
        leader.run()


if __name__ == "__main__":
    main()
