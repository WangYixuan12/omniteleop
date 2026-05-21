#!/usr/bin/env python3
"""VR leader node for teleoperation via Quest 3 (WebXR).

Architecture
------------
IK is solved here on the leader side:
- Arm IK: MotionManager.ik(type="pink") with self-collision avoidance
- Head IK: KinHelper.compute_ik_from_mat()

Data flow::

    Quest 3 (WebXR browser)
        ↓  Socket.IO  (head + wrist poses + triggers/sticks)
    WebXRVRReader
        ↓  latest VRFrame:
           head, left_wrist, right_wrist 4x4 poses in VR frame
           triggers, buttons, left/right thumbsticks
    VRReader.run()
        ├─ camera poll/cache
        │     head_camera RGB/depth cached for episode recording and browser preview
        ├─ calibration/state machine
        │     static → head → whole_body_alignment → whole_body
        │     left X resets to static; left Y interpolates joints through resetting
        ├─ head path
        │     if stage != static:
        │       robot_base_t_vr_base @ vr_head
        │       → KinHelper.compute_ik_from_mat()
        │       → head_pos command
        ├─ arm target path
        │     skipped unless stage is whole_body_alignment/whole_body
        │     invalid controller pose → skip with NaN debug telemetry
        │     follow_hand:
        │       target = robot_base_t_vr_base @ vr_wrist
        │       alignment stage interpolates target before live tracking
        │     fixed_pose:
        │       alignment stage holds INIT arm joints and computes vr_to_robot_*
        │       whole_body target = vr_to_robot_* @ vr_wrist
        ├─ arm IK path
        │     target EEF poses
        │       → MotionManager.ik(type="pink")
        │       → "ik_raw"
        │       → ArmProcessor.limit_joint_step()  (10 deg max per joint/frame)
        │       → "cmd"
        │       → MotionManager/current_qpos warm start for next frame
        ├─ chassis/gripper path
        │     thumbsticks → chassis_vx/chassis_vy/chassis_wz
        │     index triggers → left_gripper/right_gripper
        ├─ recording path  (only while recording in whole_body)
        │     main HDF5:
        │       action/joint/* = safe published commands
        │       obs/joint/* = robot joint feedback sampled through Robot API
        │     debug HDF5:
        │       raw VR wrist poses, calibration matrices, EEF target used by IK,
        │       raw IK solution/status, and exact publish payload.
        │       Every dataset named eef/* is in the robot/world frame used by IK.
        └─ publish path
              VRJointData(head_pos, arm_pos, grippers, chassis, estop, calib_stage)
              → Zenoh "vr/joints"
              → VRRobotController (follower)
                    estop=True: head only
                    estop=False: head, arms, grippers, chassis are applied

Calibration stages
------------------
static              Initial — hold right index trigger ≥ 1 s → head.
head                Head tracks VR headset.
                    Hold right index trigger ≥ 1 s → whole_body_alignment.
whole_body_alignment  Arms interpolate toward current controller positions.
                    Auto-advances to whole_body when both distances < 0.02 m.
whole_body          Live tracking with self-collision avoidance.

Published calib_stage (VRJointData):
  The literal internal stage string is published: "static", "head",
  "resetting", "whole_body_alignment", or "whole_body".
  estop=True only for "static" and "head"; the follower applies arm commands
  during "resetting", "whole_body_alignment", and "whole_body".

Usage::

    omni-vr
    # or with SSL for WiFi:
    python -m omniteleop.leader.vr_reader --ssl-certfile cert.pem --ssl-keyfile key.pem
"""

from __future__ import annotations

import base64
import dataclasses
import pathlib
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, Literal, Optional

import cv2
import numpy as np
import sapien
import tyro
from dexbot_utils import RobotInfo
from dexbot_utils.configs.components.sensors.cameras import CameraConfig
from dexcomm import Node, RateLimiter
from dexcomm.codecs import DictDataCodec
from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot as _Robot
from dexmotion.motion_manager import MotionManager
from loguru import logger
from rich.console import Console
from scipy.spatial.transform import Rotation, Slerp
from yixuan_utilities.kinematics_helper import KinHelper

from omniteleop.common import get_config
from omniteleop.common.debug_display import get_debug_display
from omniteleop.common.log_utils import suppress_loguru_module
from omniteleop.common.logging import setup_logging
from omniteleop.common.recorder import (
    EpisodeRecorder,
    _progress_bar,
    _recursive_np_stack,
    _save_dict_with_progress,
)
from omniteleop.common.schemas import VRJointData
from omniteleop.common.vis_utils import concat_img_h
from omniteleop.common.vr_mode_const import (
    INIT_HEAD_JOINTS,
    INIT_JOINTS_DICT,
    INIT_LEFT_ARM_JOINTS,
    INIT_RIGHT_ARM_JOINTS,
    INIT_TORSO_JOINTS,
    # SAFE_LEFT_ARM_JOINTS,  # unused: arms → SAFE skipped on Y-press
    # SAFE_RIGHT_ARM_JOINTS,
)
from omniteleop.follower.component_processors import ArmProcessor
from omniteleop.follower.robotiq import (
    RobotiqStatusMonitor,
    poll_gripper_status,
)
from omniteleop.follower.workspace_check import DEFAULT_LEFT_BOUNDS, DEFAULT_RIGHT_BOUNDS
from omniteleop.leader.communication.webxr_vr_reader import VRFrame, WebXRVRReader

StartMode = Literal["follow_hand", "fixed_pose"]
workspace_check = True

CameraStream = Literal[
    "head_left_rgb",
    "head_right_rgb",
    "head_depth",
    "left_wrist_rgb",
    "right_wrist_rgb",
    "wrist_depth",
]
_DEFAULT_CAMERAS: list[CameraStream] = ["head_left_rgb", "head_depth", "left_wrist_rgb"]
# Stream id → (camera namespace prefix, head_camera obs_key for that stream)
_HEAD_STREAM_TO_OBS_KEY: dict[CameraStream, str] = {
    "head_left_rgb": "left_rgb",
    "head_right_rgb": "right_rgb",
    "head_depth": "depth",
}
# Stream id → which half of the side-by-side stereo frame to use.
# dexsensor's usb_camera driver publishes the raw 2*W x H ZED-M output as one
# topic; we slice it here. wrist_depth is intentionally absent — depth would
# require running the ZED SDK, which dexsensor's usb_camera driver does not do.
_WRIST_STREAM_TO_HALF: dict[CameraStream, str] = {
    "left_wrist_rgb": "left_rgb",
    "right_wrist_rgb": "right_rgb",
}
_WRIST_SENSOR_ID = "wrist_zedm"
_WRIST_TOPIC = f"sensors/{_WRIST_SENSOR_ID}/rgb"

console = Console()

# Sentinel poses emitted by WebXRVRReader when a controller is not tracked
INVALID_LEFT_POSE = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
INVALID_RIGHT_POSE = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# HARDCODED right now
fx = 770.1868 / 2.0
fy = 770.1868 / 2.0
cx = 990.2711 / 2.0
cy = 637.7721 / 2.0
ZED_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

_HEAD_IK_JOINTS = {"head_j2", "head_j3"}
_HEAD_MOTOR_JOINTS = ["head_j1", "head_j2", "head_j3"]
_TORSO_JOINTS = ["torso_j1", "torso_j2", "torso_j3"]

_INTERP_ALPHA = 0.1  # approach interpolation step per frame
_RESET_SPEED = 1.0  # rad/s — joint speed during Y-press reset

# ── Sapien visualisation helpers ───────────────────────────────────────────────

_AXIS_LEN = 0.15
_AXIS_RADIUS = 0.007
_ROT_Z_P90 = Rotation.from_euler("z", 90, degrees=True)
_ROT_Y_N90 = Rotation.from_euler("y", -90, degrees=True)


def _make_kinematic(scene: sapien.Scene, build_fn: Callable[[Any], None]) -> sapien.Entity:
    b = scene.create_actor_builder()
    build_fn(b)
    return b.build_kinematic()


def make_frame_markers(scene: sapien.ActorBuilder, color: list[float]) -> list[sapien.Entity]:
    """Return (origin, X, Y, Z) kinematic capsule actors."""

    def sphere(b: sapien.ActorBuilder) -> None:
        b.add_sphere_visual(radius=_AXIS_RADIUS * 2, material=color)

    def axis(c: list[float]) -> Callable[[Any], None]:
        def build(b: sapien.ActorBuilder) -> None:
            b.add_capsule_visual(
                sapien.Pose(), radius=_AXIS_RADIUS, half_length=_AXIS_LEN / 2, material=c
            )

        return build

    return [
        _make_kinematic(scene, sphere),
        _make_kinematic(scene, axis([1.0, 0.15, 0.15])),  # X — red
        _make_kinematic(scene, axis([0.15, 1.0, 0.15])),  # Y — green
        _make_kinematic(scene, axis([0.15, 0.15, 1.0])),  # Z — blue
    ]


def update_frame_markers(markers: list[sapien.Entity], mat4: np.ndarray) -> None:
    p, R = mat4[:3, 3], mat4[:3, :3]
    rot = Rotation.from_matrix(R)
    half = _AXIS_LEN / 2

    def pose(center: np.ndarray, r: np.ndarray) -> sapien.Pose:
        q = r.as_quat()
        return sapien.Pose(p=center, q=[q[3], q[0], q[1], q[2]])

    markers[0].set_pose(sapien.Pose(p=p))
    markers[1].set_pose(pose(p + R[:, 0] * half, rot))
    markers[2].set_pose(pose(p + R[:, 1] * half, rot * _ROT_Z_P90))
    markers[3].set_pose(pose(p + R[:, 2] * half, rot * _ROT_Y_N90))

class DebugEpisodeRecorder:
    """Parallel recorder for the debug HDF5 (`episode_<N>_debug.hdf5`).

    Holds every intermediate stage of the teleop pipeline (raw VR pose, IK
    target, IK status, raw IK output, calibration matrices, exact published
    payload, timing). Frame layout is fixed-shape: failures use NaN, never `[]`.
    """

    def __init__(self, save_dir: str) -> None:
        self._save_dir = pathlib.Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._frames: list[dict] = []
        self.recording = False
        self.saving = False
        self.save_progress: float = 0.0
        self._save_thread: Optional[threading.Thread] = None
        self._episode_id: int = 0

    def start(self, episode_id: int) -> None:
        """Start a new debug episode."""
        self._frames = []
        self._episode_id = episode_id
        self.recording = True
        logger.info(f"DebugEpisodeRecorder: recording started (episode_{episode_id})")

    def record(self, frame: dict) -> None:
        """Append one debug frame."""
        self._frames.append(frame)

    def stop(self) -> Optional[str]:
        """Stop recording and save debug HDF5 in the background."""
        self.recording = False
        if not self._frames:
            logger.warning("DebugEpisodeRecorder: 0 frames — skipping save")
            return None
        path = self._save_dir / f"episode_{self._episode_id}_debug.hdf5"
        frames = self._frames
        self._frames = []
        self.saving = True
        self.save_progress = 0.0
        self._save_thread = threading.Thread(
            target=self._save_worker, args=(frames, str(path)), daemon=True
        )
        self._save_thread.start()
        return str(path)

    def _save_worker(self, frames: list[dict], path: str) -> None:
        try:
            data = _recursive_np_stack(frames)

            def on_progress(p: float) -> None:
                self.save_progress = p

            _save_dict_with_progress(data, path, on_progress)
            logger.info(f"DebugEpisodeRecorder: {len(frames)} frames → {path}")
        except Exception:
            logger.exception("DebugEpisodeRecorder: save failed")
        finally:
            self.saving = False


class VRReader:
    """Quest 3 WebXR pose reader + IK solver; publishes VRJointData via Zenoh."""

    def __init__(
        self,
        namespace: str = "",
        robot_name: str = "vega_no_effector",
        head_link: str = "zed_depth_frame",
        left_arm_link: str = "L_ee",
        right_arm_link: str = "R_ee",
        stick_max_vx: float = 0.3,
        stick_max_vy: float = 0.2,
        stick_max_wz: float = 0.5,
        stick_deadzone: float = 0.1,
        publish_rate: float = 15.0,
        record_rate: float = 15.0,
        debug: bool = False,
        visualize: bool = False,
        urdf_path: Optional[str] = None,
        save_dir: str = "/home/yixuan/omniteleop/Dexmate/data/raw_data",
        debug_save_dir: str = "/home/yixuan/omniteleop/Dexmate/debug/debug_data",
        start_mode: StartMode = "follow_hand",
        save_debug: bool = True,
        workspace_check: bool = workspace_check,
        cameras: Optional[list[CameraStream]] = None,
    ) -> None:
        self.stick_max_vx = stick_max_vx
        self.stick_max_vy = stick_max_vy
        self.stick_max_wz = stick_max_wz
        self.stick_deadzone = stick_deadzone
        self.publish_rate = publish_rate
        if record_rate <= 0:
            raise ValueError(f"record_rate must be > 0, got {record_rate}")
        if record_rate > publish_rate:
            raise ValueError(
                f"record_rate ({record_rate}) must be ≤ publish_rate ({publish_rate}); "
                "raise publish_rate or lower record_rate."
            )
        self.record_rate = record_rate
        self._record_period_s = 1.0 / record_rate
        # Monotonic timestamp of the next ideal record. 0.0 means "anchor on the
        # next call to _handle_recording" (set on episode start).
        self._next_record_t: float = 0.0
        self.running = False
        self.start_mode: StartMode = start_mode
        self.save_debug = save_debug
        self._workspace_check_enabled = workspace_check
        if not self._workspace_check_enabled:
            logger.warning("Right-arm workspace check disabled.")

        self._debug_display = (
            get_debug_display("VRReader", publish_rate, refresh_rate=10) if debug else None
        )

        # Zenoh
        self.node = Node(name="vr_reader", namespace=namespace)
        config = get_config()
        vr_topic = config.get_topic("vr_joints", "vr/joints")
        self.vr_pub = self.node.create_publisher(vr_topic, encoder=DictDataCodec.encode)

        # WebXR reader
        self.quest = WebXRVRReader(
            port=5067,
            ssl_certfile="/home/yixuan/omniteleop/tests/cert.pem",
            ssl_keyfile="/home/yixuan/omniteleop/tests/key.pem",
        )

        # Camera selection. Both head_camera (ZED-X Mini, GMSL) and wrist_zedm
        # (ZED-M, USB) come from dexsensor via the dexcontrol Robot API.
        # head_camera runs the ZED SDK inside dexsensor → rectified L/R + depth.
        # wrist_zedm uses dexsensor's raw V4L2 usb_camera driver → a single
        # 2W*H side-by-side BGR frame, which we slice in the loop. No depth
        # is available for wrist this way; wrist_depth in --cameras is rejected.
        selected = cameras if cameras else _DEFAULT_CAMERAS
        if not selected:
            raise ValueError("`cameras` must select at least one stream.")
        if "wrist_depth" in selected:
            raise ValueError(
                "wrist_depth is not supported: dexsensor's usb_camera driver "
                "does not run the ZED SDK, so no depth is computed for the USB "
                "ZED-M. Remove wrist_depth from --cameras."
            )
        self.cameras: list[CameraStream] = list(selected)
        self._head_keys: list[str] = [
            _HEAD_STREAM_TO_OBS_KEY[c]
            for c in self.cameras
            if c in _HEAD_STREAM_TO_OBS_KEY
        ]
        self._wrist_halves: list[str] = [
            _WRIST_STREAM_TO_HALF[c]
            for c in self.cameras
            if c in _WRIST_STREAM_TO_HALF
        ]
        self._wrist_enabled: bool = bool(self._wrist_halves)
        logger.info(
            f"Cameras selected: {self.cameras}  "
            f"(head_keys={self._head_keys}, wrist_halves={self._wrist_halves})"
        )

        # Camera polling via Robot API — also provides arm/torso/head joint
        # feedback used by gripper monitors and recording, so we always
        # construct it. Sensors are only enabled when actually selected.
        configs = get_robot_config()
        configs.sensors["head_camera"].enabled = bool(self._head_keys)
        if self._wrist_enabled and _WRIST_SENSOR_ID not in configs.sensors:
            # Variant has no wrist_zedm; inject a USB CameraConfig pointing at
            # the dexsensor topic for it. dexsensor must be launched with
            # `--sensor wrist_zedm` for this subscriber to receive frames.
            configs.sensors[_WRIST_SENSOR_ID] = CameraConfig(
                name=_WRIST_SENSOR_ID,
                topic=_WRIST_TOPIC,
                rtc_channel=f"{_WRIST_TOPIC}_rtc",
            )
        if _WRIST_SENSOR_ID in configs.sensors:
            configs.sensors[_WRIST_SENSOR_ID].enabled = self._wrist_enabled
        self._cam_robot = _Robot(configs=configs)
        if self._head_keys:
            logger.info(f"Head camera streaming started (keys={self._head_keys})")
        if self._wrist_enabled:
            logger.info(
                f"Wrist camera streaming started (halves={self._wrist_halves}) "
                f"— requires `dexsensor launch --sensor {_WRIST_SENSOR_ID}`"
            )

        # KinHelper (head IK only)
        logger.info(f"Loading KinHelper for '{robot_name}' ...")
        self.kin = KinHelper(robot_name)
        self._build_ik_config(head_link, left_arm_link, right_arm_link)

        # MotionManager + ArmProcessors
        logger.info("Initialising MotionManager ...")
        robot_info = RobotInfo()
        self.mm = MotionManager(
            init_visualizer=False, visualizer_type="viser", joint_regions_to_lock=["BASE"]
        )
        self.mm.left_arm.set_joint_pos(INIT_LEFT_ARM_JOINTS)
        self.mm.right_arm.set_joint_pos(INIT_RIGHT_ARM_JOINTS)
        self.mm.torso.set_joint_pos(INIT_TORSO_JOINTS)
        self.mm.head.set_joint_pos(INIT_HEAD_JOINTS)
        self.left_proc = ArmProcessor("left", config, self.mm, robot_info, "vr")
        self.right_proc = ArmProcessor("right", config, self.mm, robot_info, "vr")

        # Warm-start qpos for head IK
        init_qpos = np.zeros(self.kin.sapien_robot.dof)
        for name, val in INIT_JOINTS_DICT.items():
            init_qpos[self.joint_name_to_idx[name]] = val
        self.current_qpos = init_qpos

        self._trigger_start: Optional[float] = None

        # Per-arm EEF workspace status — set per frame, read by _camera_poll.
        self._left_eef_oob: bool = False
        self._left_eef_xyz: np.ndarray = np.full(3, np.nan, dtype=np.float32)
        self._right_eef_oob: bool = False
        self._right_eef_xyz: np.ndarray = np.full(3, np.nan, dtype=np.float32)

        # Recording
        self.recorder = EpisodeRecorder(save_dir)
        self.debug_recorder = DebugEpisodeRecorder(debug_save_dir)
        # Per-camera caches. Keys match canonical obs names (left_rgb /
        # right_rgb / depth) — head_* and wrist_* are kept in separate dicts
        # because they originate from different sources (Robot API vs Zenoh).
        self._last_head_imgs: dict[str, np.ndarray] = {}
        self._last_head_depth_u16: Optional[np.ndarray] = None
        self._last_wrist_imgs: dict[str, np.ndarray] = {}
        self._prev_a: bool = False
        self._prev_b: bool = False
        self._prev_x: bool = False
        self._prev_y: bool = False
        # Left-arm freeze toggle: A-press during recording flips it. While set,
        # the left arm holds the pose captured at freeze time. Transient state —
        # never written to the recorded HDF5.
        self._left_arm_frozen: bool = False
        self._frozen_left_pos: Optional[list[float]] = None

        # IK / timing telemetry — populated by _solve_and_apply_arm_ik (or
        # set to NaN/skip-reason on every bypass path) and read by
        # _handle_recording when building a debug frame.
        self._last_ik_target_left: np.ndarray = np.full((4, 4), np.nan, dtype=np.float32)
        self._last_ik_target_right: np.ndarray = np.full((4, 4), np.nan, dtype=np.float32)
        self._last_ik_raw_left: np.ndarray = np.full(7, np.nan, dtype=np.float32)
        self._last_ik_raw_right: np.ndarray = np.full(7, np.nan, dtype=np.float32)
        self._last_ik_success: bool = False
        self._last_ik_in_collision: bool = False
        self._last_ik_within_limits: bool = False
        self._last_ik_failure_reason: str = "skipped_stage"
        self._last_ik_solve_ms: float = float("nan")
        self._last_publish_ms: float = float("nan")

        # Gripper FC03 status cache. `_last_obs_grip_*` holds gPO/255 ∈ [0,1].
        # Monitors queue FC03 replies separately from FC16 ACKs so recording
        # never depends on dexcontrol's single latest-response slot.
        self._last_obs_grip_left: float = float("nan")
        self._last_obs_grip_right: float = float("nan")
        self._grip_monitors: dict[str, RobotiqStatusMonitor] = {}
        self._obs_grip_event_count_left: int = 0
        self._obs_grip_event_count_right: int = 0
        self._recorded_obs_grip_event_count_left: int = 0
        self._recorded_obs_grip_event_count_right: int = 0

        # Reset interpolation state (set when entering "resetting" stage).
        # Reset is two-phase: phase 1 brings arms to SAFE_*_ARM_JOINTS, phase 2
        # brings everything (incl. arms) to INIT_*_JOINTS.
        self._reset_start_qpos: np.ndarray = np.eye(1)
        self._reset_target_qpos: np.ndarray = np.eye(1)
        self._reset_total_steps: int = 1
        self._reset_step_idx: int = 0
        self._reset_phase: int = 0

        # Loop state (set in run() before main loop)
        self._calib_stage: str = "static"
        self._robot_base_t_vr_base: np.ndarray = np.eye(4)
        self._curr_left_target: np.ndarray = np.eye(4)
        self._curr_right_target: np.ndarray = np.eye(4)

        # Per-arm calibration for fixed_pose mode. Position and orientation are
        # locked separately so controller tilt at lock time does not skew
        # subsequent Cartesian translation.
        self._vr_to_robot_left: Optional[np.ndarray] = None
        self._vr_to_robot_right: Optional[np.ndarray] = None
        self._fixed_left_robot0: Optional[np.ndarray] = None
        self._fixed_right_robot0: Optional[np.ndarray] = None
        self._fixed_left_vr_pos0: Optional[np.ndarray] = None
        self._fixed_right_vr_pos0: Optional[np.ndarray] = None
        self._fixed_left_vr_rot0: Optional[np.ndarray] = None
        self._fixed_right_vr_rot0: Optional[np.ndarray] = None

        self.visualize = visualize
        if visualize:
            self._setup_sapien_viewer(urdf_path)

    # ── Setup helpers ──────────────────────────────────────────────────────────

    def _build_ik_config(self, head_link: str, left_arm_link: str, right_arm_link: str) -> None:
        active_joints = self.kin.sapien_robot.get_active_joints()
        self.joint_names = [j.name for j in active_joints]
        self.joint_name_to_idx = {n: i for i, n in enumerate(self.joint_names)}

        lni = self.kin.link_name_to_idx
        self.head_eef_idx = lni.get(head_link)
        self.left_arm_eef_idx = lni.get(left_arm_link)
        self.right_arm_eef_idx = lni.get(right_arm_link)

        missing = [
            n
            for n, i in [
                (head_link, self.head_eef_idx),
                (left_arm_link, self.left_arm_eef_idx),
                (right_arm_link, self.right_arm_eef_idx),
            ]
            if i is None
        ]
        if missing:
            raise ValueError(f"EEF links not found in robot model: {missing}")

        self.head_qmask = np.array([n in _HEAD_IK_JOINTS for n in self.joint_names], dtype=bool)
        self.torso_indices = [
            self.joint_name_to_idx[n] for n in _TORSO_JOINTS if n in self.joint_name_to_idx
        ]
        self.head_motor_indices = [
            self.joint_name_to_idx[n] for n in _HEAD_MOTOR_JOINTS if n in self.joint_name_to_idx
        ]

        logger.info(
            f"IK config — head_eef={self.head_eef_idx}  "
            f"L_arm_eef={self.left_arm_eef_idx}  R_arm_eef={self.right_arm_eef_idx}  "
            f"head_DOF={self.head_qmask.sum()}"
        )

    def _setup_sapien_viewer(self, urdf_path: Optional[str]) -> None:
        scene = sapien.Scene()
        scene.add_ground(-0.1)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
        self._scene = scene

        self._viewer = scene.create_viewer()
        self._viewer.set_camera_xyz(x=-2, y=0, z=1)
        self._viewer.set_camera_rpy(r=0, p=-0.3, y=0)

        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        self._viz_robot = loader.load(urdf_path)
        for link in self._viz_robot.get_links():
            for shape in link.get_collision_shapes():
                shape.set_collision_groups([1, 1, 17, 0])
        self._viz_robot.set_qpos(self.current_qpos)

        self._l_target_mrk = make_frame_markers(scene, [0.2, 0.5, 1.0])  # blue
        self._r_target_mrk = make_frame_markers(scene, [1.0, 0.5, 0.2])  # orange
        self._l_eef_mrk = make_frame_markers(scene, [0.5, 0.85, 1.0])  # cyan
        self._r_eef_mrk = make_frame_markers(scene, [1.0, 0.85, 0.3])  # yellow
        logger.info("Sapien viewer ready.")

    # ── Per-frame helpers ──────────────────────────────────────────────────────

    def _trigger_held(self, transforms: VRFrame, duration: float = 1.0) -> bool:
        """True once right index trigger continuously held for `duration` s."""
        val = transforms["right_hand_trigger"]
        now = time.perf_counter()
        if val > 0.7:
            if self._trigger_start is None:
                self._trigger_start = now
            elif now - self._trigger_start >= duration:
                self._trigger_start = None
                return True
        else:
            self._trigger_start = None
        return False

    def _thumbstick_to_chassis(self, transforms: VRFrame) -> tuple[float, float, float]:
        def dz(v: float) -> float:
            return v if abs(v) > self.stick_deadzone else 0.0

        left_thumbstick = transforms["left_thumbstick"]
        right_thumbstick = transforms["right_thumbstick"]
        # stick y: up = negative on Quest → negate for forward = positive
        vx = -dz(left_thumbstick[1]) * self.stick_max_vx
        vy = -dz(left_thumbstick[0]) * self.stick_max_vy
        wz = -dz(right_thumbstick[0]) * self.stick_max_wz
        return vx, vy, wz

    # ── Camera polling ─────────────────────────────────────────────────────────

    def _all_selected_streams_ready(self) -> bool:
        """All selected streams have at least one frame cached.

        Recording requires every selected stream to be present so the saved
        HDF5 has consistent keys across frames.
        """
        for stream in self.cameras:
            if stream == "head_left_rgb" and "left_rgb" not in self._last_head_imgs:
                return False
            if stream == "head_right_rgb" and "right_rgb" not in self._last_head_imgs:
                return False
            if stream == "head_depth" and self._last_head_depth_u16 is None:
                return False
            if stream == "left_wrist_rgb" and "left_rgb" not in self._last_wrist_imgs:
                return False
            if stream == "right_wrist_rgb" and "right_rgb" not in self._last_wrist_imgs:
                return False
        return True

    def _render_tile(self, stream: CameraStream) -> Optional[np.ndarray]:
        """Render one 320×180 preview tile for the given selected stream.

        Returns None if no frame for this stream is available yet (subscriber
        hasn't received anything, or head poll returned no key).
        """
        if stream in _HEAD_STREAM_TO_OBS_KEY:
            key = _HEAD_STREAM_TO_OBS_KEY[stream]
            img = self._last_head_imgs.get(key)
        else:
            key = _WRIST_STREAM_TO_HALF[stream]
            img = self._last_wrist_imgs.get(key)
        if img is None:
            return None

        if key == "depth":
            finite = img[np.isfinite(img) & (img > 0)]
            if len(finite) == 0:
                normalized = np.zeros(img.shape[:2], dtype=np.uint8)
            else:
                mn, mx = finite.min(), np.percentile(finite, 95)
                normalized = np.clip((img - mn) / (mx - mn + 1e-6) * 255, 0, 255).astype(np.uint8)
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
            return cv2.resize(colored, (320, 180))
        # Both head_camera and dexsensor usb_camera publish RGB. cv2 wants BGR.
        return cv2.resize(img[:, :, ::-1], (320, 180))

    def _camera_poll(self) -> None:
        vis_imgs: list[np.ndarray] = []
        for stream in self.cameras:
            tile = self._render_tile(stream)
            if tile is not None:
                vis_imgs.append(tile)
        if not vis_imgs:
            return

        vis_img = concat_img_h(vis_imgs)
        episode_id: int = self.recorder.episode_id
        text: str = f"Episode: {episode_id}"
        if self.recorder.recording:
            text += f", Recording! Step: {self.recorder.num_frames()}"
        elif self.recorder.saving:
            text += f", Saving {_progress_bar(self.recorder.save_progress)}"
        elif self.debug_recorder.saving:
            text += f", Saving debug {_progress_bar(self.debug_recorder.save_progress)}"
        cv2.putText(
            vis_img,
            text,
            (10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            thickness=2,
            color=(255, 255, 255),
        )
        cv2.putText(
            vis_img,
            f"Stage: {self._calib_stage}",
            (10, vis_img.shape[0] - 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            thickness=2,
            color=(0, 255, 255),
        )
        if self._last_ik_in_collision:
            cv2.putText(
                vis_img,
                "Left/Right Arm Self-collision!",
                (10, 65),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                thickness=2,
                color=(0, 0, 255),
            )
        if self._left_eef_oob:
            xyz = self._left_eef_xyz
            cv2.putText(
                vis_img,
                f"WARNING: L out of workspace xyz=[{xyz[0]:+.2f},{xyz[1]:+.2f},{xyz[2]:+.2f}]",
                (10, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                thickness=2,
                color=(0, 0, 255),
            )
        if self._right_eef_oob:
            xyz = self._right_eef_xyz
            cv2.putText(
                vis_img,
                f"WARNING: R out of workspace xyz=[{xyz[0]:+.2f},{xyz[1]:+.2f},{xyz[2]:+.2f}]",
                (10, 135),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                thickness=2,
                color=(0, 0, 255),
            )
        _, buf = cv2.imencode(".jpg", vis_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        self.quest.set_frame_vis("img", base64.b64encode(buf).decode())
        cv2.imshow("VR see-through", vis_img)
        cv2.waitKey(1)

    # ── Per-step methods ───────────────────────────────────────────────────────

    def _head_ik_step(self, vr_head: np.ndarray) -> list[float]:
        if self._calib_stage == "static":
            return []
        head_qpos = self.kin.compute_ik_from_mat(
            self.current_qpos,
            self._robot_base_t_vr_base @ vr_head,
            eef_idx=self.head_eef_idx,
            active_qmask=self.head_qmask,
            damp=1000.0,
        )
        # self.mm.torso.set_joint_pos(head_qpos[self.torso_indices].tolist())
        for i in np.where(self.head_qmask)[0]:
            self.current_qpos[i] = head_qpos[i]
        return [float(head_qpos[i]) for i in self.head_motor_indices]

    def _approach_step(
        self, robot_l: np.ndarray, robot_r: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        for curr, tgt in [
            (self._curr_left_target, robot_l),
            (self._curr_right_target, robot_r),
        ]:
            curr[:3, 3] += _INTERP_ALPHA * (tgt[:3, 3] - curr[:3, 3])
            curr[:3, :3] = Slerp(
                [0, 1],
                Rotation.concatenate(
                    [Rotation.from_matrix(curr[:3, :3]), Rotation.from_matrix(tgt[:3, :3])]
                ),
            )(_INTERP_ALPHA).as_matrix()
        dist_l = np.linalg.norm(self._curr_left_target[:3, 3] - robot_l[:3, 3])
        dist_r = np.linalg.norm(self._curr_right_target[:3, 3] - robot_r[:3, 3])
        print(f"Approaching  L:{dist_l:.3f}  R:{dist_r:.3f} m", end="\r")
        if dist_l < 0.02 and dist_r < 0.02:
            self._calib_stage = "whole_body"
            console.print("\n[bold green]Live tracking active.[/]")
        return self._curr_left_target, self._curr_right_target

    def _solve_and_apply_arm_ik(
        self, l_target: np.ndarray, r_target: np.ndarray
    ) -> tuple[list[float], list[float]]:
        # Stash the canonical robot/world-frame EEF targets that were passed to
        # IK. These are used only by the optional debug recorder.
        self._last_ik_target_left = l_target.astype(np.float32, copy=True)
        self._last_ik_target_right = r_target.astype(np.float32, copy=True)

        t0 = time.perf_counter()
        with suppress_loguru_module("dexmotion", enabled=True):
            arm_solution, in_collision, within_limits = self.mm.ik(
                target_pose={"L_ee": l_target, "R_ee": r_target},
                type="pink",
            )
        self._last_ik_solve_ms = (time.perf_counter() - t0) * 1000.0
        self._last_ik_in_collision = bool(in_collision)
        self._last_ik_within_limits = bool(within_limits)

        if not arm_solution or in_collision or not within_limits:
            if in_collision:
                console.print("[red]Self-collision — holding arm position[/]")
            self._last_ik_success = False
            self._last_ik_raw_left = np.full(7, np.nan, dtype=np.float32)
            self._last_ik_raw_right = np.full(7, np.nan, dtype=np.float32)
            self._last_ik_failure_reason = (
                "in_collision"
                if in_collision
                else ("limits" if not within_limits else "no_solution")
            )
            return [], []

        self._last_ik_success = True
        self._last_ik_failure_reason = "ok"
        self._last_ik_raw_left = np.asarray(
            [arm_solution[f"L_arm_j{i}"] for i in range(1, 8)], dtype=np.float32
        )
        self._last_ik_raw_right = np.asarray(
            [arm_solution[f"R_arm_j{i}"] for i in range(1, 8)], dtype=np.float32
        )

        safe_left = self.left_proc.limit_joint_step(
            [arm_solution[f"L_arm_j{i}"] for i in range(1, 8)]
        )
        safe_right = self.right_proc.limit_joint_step(
            [arm_solution[f"R_arm_j{i}"] for i in range(1, 8)]
        )
        self.left_proc.apply_positions(safe_left)
        self.right_proc.apply_positions(safe_right)
        for i, n in enumerate(self.left_proc.joint_names):
            self.current_qpos[self.joint_name_to_idx[n]] = safe_left[i]
        for i, n in enumerate(self.right_proc.joint_names):
            self.current_qpos[self.joint_name_to_idx[n]] = safe_right[i]
        return safe_left.tolist(), safe_right.tolist()

    def _update_left_eef_workspace_status(self, left_pos: list[float]) -> None:
        """FK the commanded left-arm joints and check against workspace bounds.

        Mirrors the gate in vr_robot_controller — when this flag is True the
        follower will freeze the left arm, so the headset overlay matches.
        """
        if not left_pos or self._calib_stage not in (
            "resetting",
            "whole_body",
            "whole_body_alignment",
        ):
            self._left_eef_oob = False
            self._left_eef_xyz = np.full(3, np.nan, dtype=np.float32)
            return
        eef_xyz = self.kin.compute_fk_from_link_idx(
            self.current_qpos, [self.left_arm_eef_idx]
        )[0][:3, 3]
        bx, by, bz = (
            DEFAULT_LEFT_BOUNDS["x"],
            DEFAULT_LEFT_BOUNDS["y"],
            DEFAULT_LEFT_BOUNDS["z"],
        )
        in_bounds = (
            bx[0] <= eef_xyz[0] <= bx[1]
            and by[0] <= eef_xyz[1] <= by[1]
            and bz[0] <= eef_xyz[2] <= bz[1]
        )
        self._left_eef_oob = not in_bounds
        self._left_eef_xyz = np.asarray(eef_xyz, dtype=np.float32)

    def _update_right_eef_workspace_status(self, right_pos: list[float]) -> None:
        """FK the commanded right-arm joints and check against workspace bounds.

        Mirrors the gate in vr_robot_controller — when this flag is True the
        follower will freeze the right arm, so the headset overlay matches.
        """
        if not right_pos or self._calib_stage not in (
            "resetting",
            "whole_body",
            "whole_body_alignment",
        ):
            self._right_eef_oob = False
            self._right_eef_xyz = np.full(3, np.nan, dtype=np.float32)
            return
        eef_xyz = self.kin.compute_fk_from_link_idx(
            self.current_qpos, [self.right_arm_eef_idx]
        )[0][:3, 3]
        bx, by, bz = (
            DEFAULT_RIGHT_BOUNDS["x"],
            DEFAULT_RIGHT_BOUNDS["y"],
            DEFAULT_RIGHT_BOUNDS["z"],
        )
        in_bounds = (
            bx[0] <= eef_xyz[0] <= bx[1]
            and by[0] <= eef_xyz[1] <= by[1]
            and bz[0] <= eef_xyz[2] <= bz[1]
        )
        self._right_eef_oob = not in_bounds
        self._right_eef_xyz = np.asarray(eef_xyz, dtype=np.float32)

    def _poll_gripper_status_step(self) -> None:
        """Drain queued FC03 replies, then send next status requests.

        Co-exists with the follower's FC16 writes on the shared EE pass-through
        Zenoh topic. RobotiqStatusMonitor has its own subscriber and queues
        FC03 replies so FC16 ACKs cannot overwrite statuses before this loop
        reads them.
        """
        for side, monitor in self._grip_monitors.items():
            events = monitor.drain_status_events()
            if events:
                latest = events[-1]
                setattr(self, f"_last_obs_grip_{side}", float(latest["actual"]))
                count = getattr(self, f"_obs_grip_event_count_{side}") + len(events)
                setattr(self, f"_obs_grip_event_count_{side}", count)
            monitor.expire_timeouts(max(0.5, 2.0 / self.publish_rate))
            monitor.send_status_request()

    def _recordable_obs_grip(self, side: str) -> np.float32:
        """Return the latest gripper status only if it is fresh for this record."""
        count = getattr(self, f"_obs_grip_event_count_{side}")
        recorded_count = getattr(self, f"_recorded_obs_grip_event_count_{side}")
        if count <= recorded_count:
            return np.float32(np.nan)
        setattr(self, f"_recorded_obs_grip_event_count_{side}", count)
        return np.float32(getattr(self, f"_last_obs_grip_{side}"))

    def _mark_ik_skipped(self, reason: str) -> None:
        """Reset IK telemetry to NaN/skipped when no IK call was made this frame."""
        self._last_ik_target_left = np.full((4, 4), np.nan, dtype=np.float32)
        self._last_ik_target_right = np.full((4, 4), np.nan, dtype=np.float32)
        self._last_ik_raw_left = np.full(7, np.nan, dtype=np.float32)
        self._last_ik_raw_right = np.full(7, np.nan, dtype=np.float32)
        self._last_ik_success = False
        self._last_ik_in_collision = False
        self._last_ik_within_limits = False
        self._last_ik_failure_reason = reason
        self._last_ik_solve_ms = float("nan")

    def _clear_fixed_pose_calibration(self) -> None:
        self._vr_to_robot_left = None
        self._vr_to_robot_right = None
        self._fixed_left_robot0 = None
        self._fixed_right_robot0 = None
        self._fixed_left_vr_pos0 = None
        self._fixed_right_vr_pos0 = None
        self._fixed_left_vr_rot0 = None
        self._fixed_right_vr_rot0 = None

    def _has_fixed_pose_calibration(self) -> bool:
        return (
            self._fixed_left_robot0 is not None
            and self._fixed_right_robot0 is not None
            and self._fixed_left_vr_pos0 is not None
            and self._fixed_right_vr_pos0 is not None
            and self._fixed_left_vr_rot0 is not None
            and self._fixed_right_vr_rot0 is not None
        )

    def _arm_ik_step(
        self, vr_l: np.ndarray, vr_r: np.ndarray
    ) -> tuple[list[float], list[float], Optional[np.ndarray], Optional[np.ndarray]]:
        if self._calib_stage not in ("whole_body_alignment", "whole_body"):
            self._mark_ik_skipped("skipped_stage")
            return [], [], None, None
        if np.allclose(vr_l, INVALID_LEFT_POSE) or np.allclose(vr_r, INVALID_RIGHT_POSE):
            self._mark_ik_skipped("skipped_invalid_pose")
            return [], [], None, None
        if self.start_mode == "fixed_pose":
            return self._fixed_pose_arm_step(vr_l, vr_r)
        robot_l = self._robot_base_t_vr_base @ vr_l
        robot_r = self._robot_base_t_vr_base @ vr_r
        if self._calib_stage == "whole_body_alignment":
            ik_l, ik_r = self._approach_step(robot_l, robot_r)
        else:
            ik_l, ik_r = robot_l, robot_r
        left_pos, right_pos = self._solve_and_apply_arm_ik(ik_l, ik_r)
        return left_pos, right_pos, ik_l, ik_r

    def _fixed_pose_arm_step(
        self, vr_l: np.ndarray, vr_r: np.ndarray
    ) -> tuple[list[float], list[float], Optional[np.ndarray], Optional[np.ndarray]]:
        if self._calib_stage == "whole_body_alignment":
            # Arm is held at FIXED (set by resetting). Recompute per-arm lock
            # state each frame so it reflects the user's current hand pose;
            # locked when the user trigger-advances to whole_body.
            safe_left = self.left_proc.limit_joint_step(list(INIT_LEFT_ARM_JOINTS))
            safe_right = self.right_proc.limit_joint_step(list(INIT_RIGHT_ARM_JOINTS))
            self.left_proc.apply_positions(safe_left)
            self.right_proc.apply_positions(safe_right)
            for i, n in enumerate(self.left_proc.joint_names):
                self.current_qpos[self.joint_name_to_idx[n]] = safe_left[i]
            for i, n in enumerate(self.right_proc.joint_names):
                self.current_qpos[self.joint_name_to_idx[n]] = safe_right[i]
            left_fk = self.kin.compute_fk_from_link_idx(
                self.current_qpos, [self.left_arm_eef_idx]
            )[0]
            right_fk = self.kin.compute_fk_from_link_idx(
                self.current_qpos, [self.right_arm_eef_idx]
            )[0]
            self._fixed_left_robot0 = left_fk.copy()
            self._fixed_right_robot0 = right_fk.copy()
            self._fixed_left_vr_pos0 = vr_l[:3, 3].copy()
            self._fixed_right_vr_pos0 = vr_r[:3, 3].copy()
            self._fixed_left_vr_rot0 = vr_l[:3, :3].copy()
            self._fixed_right_vr_rot0 = vr_r[:3, :3].copy()
            # Kept for debug/backward compatibility; live fixed_pose tracking
            # uses the split lock state above.
            self._vr_to_robot_left = left_fk @ np.linalg.inv(vr_l)
            self._vr_to_robot_right = right_fk @ np.linalg.inv(vr_r)
            self._mark_ik_skipped("skipped_alignment_fixed")
            return safe_left.tolist(), safe_right.tolist(), left_fk, right_fk

        # whole_body: use locked per-arm calibration. Translation follows the
        # VR position delta in the calibrated VR/base frame; rotation follows
        # relative controller orientation.
        if not self._has_fixed_pose_calibration():
            self._mark_ik_skipped("skipped_no_calibration")
            return [], [], None, None
        assert self._fixed_left_robot0 is not None
        assert self._fixed_right_robot0 is not None
        assert self._fixed_left_vr_pos0 is not None
        assert self._fixed_right_vr_pos0 is not None
        assert self._fixed_left_vr_rot0 is not None
        assert self._fixed_right_vr_rot0 is not None
        robot_base_r_vr_base = self._robot_base_t_vr_base[:3, :3]
        ik_l = self._fixed_left_robot0.copy()
        ik_r = self._fixed_right_robot0.copy()
        ik_l[:3, 3] = self._fixed_left_robot0[:3, 3] + robot_base_r_vr_base @ (
            vr_l[:3, 3] - self._fixed_left_vr_pos0
        )
        ik_r[:3, 3] = self._fixed_right_robot0[:3, 3] + robot_base_r_vr_base @ (
            vr_r[:3, 3] - self._fixed_right_vr_pos0
        )
        ik_l[:3, :3] = (
            self._fixed_left_robot0[:3, :3] @ self._fixed_left_vr_rot0.T @ vr_l[:3, :3]
        )
        ik_r[:3, :3] = (
            self._fixed_right_robot0[:3, :3] @ self._fixed_right_vr_rot0.T @ vr_r[:3, :3]
        )
        left_pos, right_pos = self._solve_and_apply_arm_ik(ik_l, ik_r)
        return left_pos, right_pos, ik_l, ik_r

    def _resetting_step(self, transforms: VRFrame) -> tuple[list[float], list[float], list[float]]:
        """Interpolate joints toward reset target.

        On phase 1 completion advance
        to phase 2 (INIT); on phase 2 completion compute new base transform.
        """
        self._reset_step_idx += 1
        alpha = min(1.0, self._reset_step_idx / self._reset_total_steps)
        qpos = self._reset_start_qpos + alpha * (self._reset_target_qpos - self._reset_start_qpos)
        self.current_qpos[:] = qpos

        left_pos = [float(qpos[self.joint_name_to_idx[n]]) for n in self.left_proc.joint_names]
        right_pos = [float(qpos[self.joint_name_to_idx[n]]) for n in self.right_proc.joint_names]
        head_pos = [float(qpos[i]) for i in self.head_motor_indices]
        torso_pos = [float(qpos[i]) for i in self.torso_indices]

        self.mm.left_arm.set_joint_pos(left_pos)
        self.mm.right_arm.set_joint_pos(right_pos)
        self.mm.torso.set_joint_pos(torso_pos)
        self.mm.head.set_joint_pos(head_pos)

        if alpha >= 1.0:
            if self._reset_phase == 1:
                # Phase 1 (arms → SAFE) done; queue phase 2 (all → INIT).
                target = self.current_qpos.copy()
                for i, n in enumerate(self.left_proc.joint_names):
                    target[self.joint_name_to_idx[n]] = INIT_LEFT_ARM_JOINTS[i]
                for i, n in enumerate(self.right_proc.joint_names):
                    target[self.joint_name_to_idx[n]] = INIT_RIGHT_ARM_JOINTS[i]
                for i, idx in enumerate(self.torso_indices):
                    target[idx] = INIT_TORSO_JOINTS[i]
                for i, idx in enumerate(self.head_motor_indices):
                    target[idx] = INIT_HEAD_JOINTS[i]
                self._reset_start_qpos = self.current_qpos.copy()
                self._reset_target_qpos = target
                max_dist = float(np.max(np.abs(target - self.current_qpos)))
                self._reset_total_steps = max(1, int(max_dist / (_RESET_SPEED / self.publish_rate)))
                self._reset_step_idx = 0
                self._reset_phase = 2
                console.rule("[bold cyan]Safe pose reached → resetting to init pose…")
                return head_pos, left_pos, right_pos

            self._robot_base_t_vr_base = self.kin.compute_fk_from_link_idx(
                self.current_qpos, [self.head_eef_idx]
            )[0] @ np.linalg.inv(transforms["head"])
            self._curr_left_target = self.kin.compute_fk_from_link_idx(
                self.current_qpos, [self.left_arm_eef_idx]
            )[0].copy()
            self._curr_right_target = self.kin.compute_fk_from_link_idx(
                self.current_qpos, [self.right_arm_eef_idx]
            )[0].copy()
            self._calib_stage = "whole_body_alignment"
            console.rule("[bold cyan]Reset complete → whole_body_alignment")

        return head_pos, left_pos, right_pos

    def _arm_at_fixed(self, atol: float = 0.05) -> bool:
        """Check arm joints in current_qpos are within `atol` rad of INIT."""
        left = np.array(
            [self.current_qpos[self.joint_name_to_idx[n]] for n in self.left_proc.joint_names]
        )
        right = np.array(
            [self.current_qpos[self.joint_name_to_idx[n]] for n in self.right_proc.joint_names]
        )
        return bool(
            np.allclose(left, INIT_LEFT_ARM_JOINTS, atol=atol)
            and np.allclose(right, INIT_RIGHT_ARM_JOINTS, atol=atol)
        )

    def _handle_stage_transition(self, transforms: VRFrame) -> None:
        x_now = transforms["left_x_button"]
        y_now = transforms["left_y_button"]

        if x_now and not self._prev_x:
            self._calib_stage = "static"
            self._clear_fixed_pose_calibration()
            console.rule("[bold red]Stage reset → static (left X)")

        if y_now and not self._prev_y:
            # Phase 1: arms → SAFE (torso/head stay put). Phase 2 (all → INIT)
            # is queued by _resetting_step when phase 1 completes.
            target = self.current_qpos.copy()
            # NOTE: skip arms → SAFE on Y-press. Leaving the arm targets at the
            # current pose makes phase 1 a no-op for the arms (it completes in one
            # step), so the reset goes straight to phase 2 (→ INIT).
            # for i, n in enumerate(self.left_proc.joint_names):
            #     target[self.joint_name_to_idx[n]] = SAFE_LEFT_ARM_JOINTS[i]
            # for i, n in enumerate(self.right_proc.joint_names):
            #     target[self.joint_name_to_idx[n]] = SAFE_RIGHT_ARM_JOINTS[i]
            self._reset_start_qpos = self.current_qpos.copy()
            self._reset_target_qpos = target
            max_dist = float(np.max(np.abs(target - self.current_qpos)))
            self._reset_total_steps = max(1, int(max_dist / (_RESET_SPEED / self.publish_rate)))
            self._reset_step_idx = 0
            self._reset_phase = 1
            self._calib_stage = "resetting"
            self._clear_fixed_pose_calibration()
            console.rule("[bold cyan]Resetting to safe pose… (left Y)")

        self._prev_x = x_now
        self._prev_y = y_now

        if not self._trigger_held(transforms):
            return
        if self._calib_stage == "static":
            self._calib_stage = "head"
            console.rule("[bold cyan]Stage head — hold trigger → arm approach")
        elif self._calib_stage == "head":
            if self.start_mode == "fixed_pose" and not self._arm_at_fixed():
                console.print(
                    "[bold red]fixed_pose: press Y first to reset arms to FIXED before "
                    "advancing to whole_body_alignment[/]"
                )
                return
            self._calib_stage = "whole_body_alignment"
            self._clear_fixed_pose_calibration()
            self._curr_left_target = self.kin.compute_fk_from_link_idx(
                self.current_qpos, [self.left_arm_eef_idx]
            )[0].copy()
            self._curr_right_target = self.kin.compute_fk_from_link_idx(
                self.current_qpos, [self.right_arm_eef_idx]
            )[0].copy()
            if self.start_mode == "fixed_pose":
                console.rule(
                    "[bold cyan]Stage whole_body_alignment — position your hands, "
                    "hold trigger to lock calibration"
                )
            else:
                console.rule("[bold cyan]Stage whole_body_alignment — approaching arm targets…")
        elif self._calib_stage == "whole_body_alignment" and self.start_mode == "fixed_pose":
            if not self._has_fixed_pose_calibration():
                console.print(
                    "[bold red]fixed_pose: calibration not yet computed — wait one frame[/]"
                )
                return
            self._calib_stage = "whole_body"
            console.rule("[bold green]Calibration locked → whole_body")

    def _toggle_left_arm_freeze(self) -> None:
        """A-press during recording: freeze the left arm at its current pose, or
        release it. Freeze state is transient and never written to the HDF5."""
        self._left_arm_frozen = not self._left_arm_frozen
        if self._left_arm_frozen:
            self._frozen_left_pos = [
                float(self.current_qpos[self.joint_name_to_idx[n]])
                for n in self.left_proc.joint_names
            ]
            console.print("[bold cyan]Left arm frozen — press A to release[/]")
        else:
            self._frozen_left_pos = None
            console.print("[bold cyan]Left arm released — tracking resumed[/]")

    def _pin_left_arm(self, left_joints: list[float]) -> None:
        """Hold the leader's left-arm tracking state at ``left_joints``.

        Pinning the processor and current_qpos (not just the published value)
        keeps the command static while frozen and lets limit_joint_step ramp
        smoothly when the arm is released — the follower applies commands with
        no step limit, so the smoothing must happen here.
        """
        self.left_proc.apply_positions(np.asarray(left_joints, dtype=float))
        for i, n in enumerate(self.left_proc.joint_names):
            self.current_qpos[self.joint_name_to_idx[n]] = left_joints[i]

    def _handle_recording(
        self,
        transforms: VRFrame,
        vr_l: np.ndarray,
        vr_r: np.ndarray,
        head_pos: list[float],
        left_pos: list[float],
        right_pos: list[float],
        vx: float,
        vy: float,
        wz: float,
    ) -> list[float]:
        a_now = transforms["right_a_button"]
        b_now = transforms["right_b_button"]

        if (
            a_now
            and not self._prev_a
            and self._calib_stage == "whole_body"
            and not self.recorder.recording
            and not self.recorder.saving
            and not self.debug_recorder.saving
        ):
            if self.save_debug:
                self.debug_recorder.start(self.recorder.episode_id)
            self.recorder.start()
            self._left_arm_frozen = False
            self._frozen_left_pos = None
            self._recorded_obs_grip_event_count_left = self._obs_grip_event_count_left
            self._recorded_obs_grip_event_count_right = self._obs_grip_event_count_right
            self._next_record_t = 0.0  # anchor cadence to the first recorded frame
            console.print(
                f"[bold green]Recording started at {self.record_rate:g} Hz "
                "(press A to freeze/release left arm, B to stop)[/]"
            )
        elif a_now and not self._prev_a and self.recorder.recording:
            self._toggle_left_arm_freeze()
        if b_now and not self._prev_b and self.recorder.recording:
            path = self.recorder.stop()
            console.print(f"[bold yellow]Saving in background → {path}[/]")
            if self.save_debug:
                debug_path = self.debug_recorder.stop()
                console.print(f"[bold yellow]Debug saving in background → {debug_path}[/]")
            self._left_arm_frozen = False
            self._frozen_left_pos = None

        self._prev_a = a_now
        self._prev_b = b_now

        # While frozen, hold the left arm at the captured pose for both the
        # published command and the recorded action.
        if (
            self.recorder.recording
            and self._left_arm_frozen
            and self._frozen_left_pos is not None
        ):
            self._pin_left_arm(self._frozen_left_pos)
            left_pos = list(self._frozen_left_pos)

        if not self.recorder.recording or self._calib_stage != "whole_body":
            return left_pos
        if not self._all_selected_streams_ready():
            return left_pos

        # Throttle to record_rate so saved HDF5 has a consistent FPS independent
        # of publish_rate. Cadence is locked to ideal timestamps; if the loop
        # falls behind by >1 period we re-anchor to avoid burst catch-up.
        now_t = time.monotonic()
        if self._next_record_t == 0.0:
            self._next_record_t = now_t
        if now_t < self._next_record_t:
            return left_pos
        self._next_record_t += self._record_period_s
        if now_t > self._next_record_t:
            self._next_record_t = now_t + self._record_period_s

        # Sample observed joints once and FK them to the head camera link to
        # get world_t_cam (matches scripts/compute_extrinsics.py output).
        obs_torso = np.array(self._cam_robot.torso.get_joint_pos(), dtype=np.float32)
        obs_left_arm = np.array(self._cam_robot.left_arm.get_joint_pos(), dtype=np.float32)
        obs_right_arm = np.array(self._cam_robot.right_arm.get_joint_pos(), dtype=np.float32)
        obs_head = np.array(self._cam_robot.head.get_joint_pos(), dtype=np.float32)
        obs_qpos = np.zeros(self.kin.sapien_robot.dof, dtype=np.float64)
        for i, n in enumerate(_TORSO_JOINTS):
            if n in self.joint_name_to_idx:
                obs_qpos[self.joint_name_to_idx[n]] = obs_torso[i]
        for i, n in enumerate(self.left_proc.joint_names):
            obs_qpos[self.joint_name_to_idx[n]] = obs_left_arm[i]
        for i, n in enumerate(self.right_proc.joint_names):
            obs_qpos[self.joint_name_to_idx[n]] = obs_right_arm[i]
        for i, n in enumerate(_HEAD_MOTOR_JOINTS):
            if n in self.joint_name_to_idx:
                obs_qpos[self.joint_name_to_idx[n]] = obs_head[i]
        extrinsic = self.kin.compute_fk_from_link_idx(obs_qpos, [self.head_eef_idx])[0].astype(
            np.float32
        )
        obs_grip_left = self._recordable_obs_grip("left")
        obs_grip_right = self._recordable_obs_grip("right")

        images: dict[str, np.ndarray] = {}
        for stream in self.cameras:
            if stream == "head_left_rgb":
                images["head_left_rgb"] = self._last_head_imgs["left_rgb"]
            elif stream == "head_right_rgb":
                images["head_right_rgb"] = self._last_head_imgs["right_rgb"]
            elif stream == "head_depth":
                assert self._last_head_depth_u16 is not None
                images["head_depth"] = self._last_head_depth_u16
            elif stream == "left_wrist_rgb":
                images["left_wrist_rgb"] = self._last_wrist_imgs["left_rgb"]
            elif stream == "right_wrist_rgb":
                images["right_wrist_rgb"] = self._last_wrist_imgs["right_rgb"]
        # intrinsic/extrinsic are head_camera-specific — only emit when head
        # is being recorded. Wrist intrinsics not yet plumbed.
        if self._head_keys:
            images["intrinsic"] = ZED_K.astype(np.float32)
            images["extrinsic"] = extrinsic

        frame = {
            "timestamp_ns": np.int64(time.time_ns()),
            "action": {
                "joint": {
                    "left_arm": np.array(left_pos, dtype=np.float32),
                    "right_arm": np.array(right_pos, dtype=np.float32),
                    "head": np.array(head_pos, dtype=np.float32),
                    "chassis_vx": np.float32(vx),
                    "chassis_vy": np.float32(vy),
                    "chassis_wz": np.float32(wz),
                },
                "gripper": {
                    "left": np.float32(transforms["left_index_trigger"]),
                    "right": np.float32(transforms["right_index_trigger"]),
                },
            },
            "obs": {
                "joint": {
                    "left_arm": obs_left_arm,
                    "right_arm": obs_right_arm,
                    "head": obs_head,
                    "torso": obs_torso,
                },
                "gripper": {
                    "left": obs_grip_left,
                    "right": obs_grip_right,
                },
                "images": images,
            },
        }
        self.recorder.record(frame)

        if not self.save_debug:
            return left_pos

        # Debug keeps the robot/world-frame EEF poses that were passed to IK.
        # Raw VR wrist poses stay under debug vr_raw/*.
        eef_left = self._last_ik_target_left.astype(np.float32, copy=True)
        eef_right = self._last_ik_target_right.astype(np.float32, copy=True)

        nan44 = np.full((4, 4), np.nan, dtype=np.float32)
        nan7 = np.full(7, np.nan, dtype=np.float32)
        nan3 = np.full(3, np.nan, dtype=np.float32)
        vr_l_log = (
            vr_l.astype(np.float32) if not np.allclose(vr_l, INVALID_LEFT_POSE) else nan44.copy()
        )
        vr_r_log = (
            vr_r.astype(np.float32) if not np.allclose(vr_r, INVALID_RIGHT_POSE) else nan44.copy()
        )
        vr_to_l = (
            self._vr_to_robot_left.astype(np.float32)
            if self._vr_to_robot_left is not None
            else nan44.copy()
        )
        vr_to_r = (
            self._vr_to_robot_right.astype(np.float32)
            if self._vr_to_robot_right is not None
            else nan44.copy()
        )

        debug_frame = {
            "timestamp_ns": np.int64(time.time_ns()),
            "timing": {
                "monotonic_ns": np.int64(time.monotonic_ns()),
                "ik_solve_ms": np.float32(self._last_ik_solve_ms),
                "publish_ms": np.float32(self._last_publish_ms),
            },
            "calib_stage": np.bytes_(self._calib_stage),
            "vr_raw": {
                "head": transforms["head"].astype(np.float32),
                "left_wrist": vr_l_log,
                "right_wrist": vr_r_log,
                "left_thumbstick": np.asarray(transforms["left_thumbstick"], dtype=np.float32),
                "right_thumbstick": np.asarray(transforms["right_thumbstick"], dtype=np.float32),
                "left_trigger": np.float32(transforms["left_index_trigger"]),
                "right_trigger": np.float32(transforms["right_index_trigger"]),
            },
            "calib": {
                "robot_base_t_vr_base": self._robot_base_t_vr_base.astype(np.float32),
                "vr_to_robot_left": vr_to_l,
                "vr_to_robot_right": vr_to_r,
            },
            "target": {
                "eef_used_by_ik": {
                    "left": eef_left,
                    "right": eef_right,
                },
            },
            "ik": {
                "left_arm_raw": self._last_ik_raw_left,
                "right_arm_raw": self._last_ik_raw_right,
                "status": {
                    "success": np.bool_(self._last_ik_success),
                    "in_collision": np.bool_(self._last_ik_in_collision),
                    "within_limits": np.bool_(self._last_ik_within_limits),
                    "failure_reason": np.bytes_(self._last_ik_failure_reason),
                },
            },
            "publish": {
                "payload": {
                    "head_pos": (
                        np.asarray(head_pos, dtype=np.float32) if head_pos else nan3.copy()
                    ),
                    "left_arm_pos": (
                        np.asarray(left_pos, dtype=np.float32) if left_pos else nan7.copy()
                    ),
                    "right_arm_pos": (
                        np.asarray(right_pos, dtype=np.float32) if right_pos else nan7.copy()
                    ),
                    "left_gripper": np.float32(transforms["left_index_trigger"]),
                    "right_gripper": np.float32(transforms["right_index_trigger"]),
                    "chassis_vx": np.float32(vx),
                    "chassis_vy": np.float32(vy),
                    "chassis_wz": np.float32(wz),
                    "estop": np.bool_(self._calib_stage in ("static", "head")),
                },
            },
        }
        self.debug_recorder.record(debug_frame)
        return left_pos

    def _publish(
        self,
        head_pos: list[float],
        left_arm_pos: list[float],
        right_arm_pos: list[float],
        transforms: VRFrame,
        chassis_vx: float,
        chassis_vy: float,
        chassis_wz: float,
    ) -> None:
        estop = self._calib_stage in ("static", "head")
        data = VRJointData(
            timestamp_ns=time.time_ns(),
            head_pos=head_pos,
            left_arm_pos=left_arm_pos,
            right_arm_pos=right_arm_pos,
            left_gripper=float(transforms["left_index_trigger"]),
            right_gripper=float(transforms["right_index_trigger"]),
            chassis_vx=chassis_vx,
            chassis_vy=chassis_vy,
            chassis_wz=chassis_wz,
            estop=estop,
            calib_stage=self._calib_stage,
        )
        # logger.info(f"left_arm: {left_arm_pos}")
        t0 = time.perf_counter()
        self.vr_pub.publish(asdict(data))
        self._last_publish_ms = (time.perf_counter() - t0) * 1000.0

    def _update_visualization(self, ik_l: Optional[np.ndarray], ik_r: Optional[np.ndarray]) -> bool:
        """Update Sapien viewer. Returns False if viewer was closed."""
        self._viz_robot.set_qpos(self.current_qpos)
        curr_l_fk = self.kin.compute_fk_from_link_idx(self.current_qpos, [self.left_arm_eef_idx])[0]
        curr_r_fk = self.kin.compute_fk_from_link_idx(self.current_qpos, [self.right_arm_eef_idx])[
            0
        ]
        update_frame_markers(self._l_eef_mrk, curr_l_fk)
        update_frame_markers(self._r_eef_mrk, curr_r_fk)
        if ik_l is not None:
            update_frame_markers(self._l_target_mrk, ik_l)
            update_frame_markers(self._r_target_mrk, ik_r)
        self._scene.update_render()
        self._viewer.render()
        return not self._viewer.closed

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Main loop"""
        self.running = True
        self.quest.start()
        logger.info("Waiting for Quest data ...")
        self.quest.wait_for_data()
        logger.info("Quest connected.")

        init_tf = self.quest.get_latest_transformation()
        assert init_tf is not None
        self._robot_base_t_vr_base = self.kin.compute_fk_from_link_idx(
            self.current_qpos, [self.head_eef_idx]
        )[0] @ np.linalg.inv(init_tf["head"])
        self._curr_left_target = self.kin.compute_fk_from_link_idx(
            self.current_qpos, [self.left_arm_eef_idx]
        )[0].copy()
        self._curr_right_target = self.kin.compute_fk_from_link_idx(
            self.current_qpos, [self.right_arm_eef_idx]
        )[0].copy()
        self._calib_stage = "static"

        # Warm up gripper FC03 reads. Seeds the cache so obs.gripper is valid
        # from frame 0, and surfaces a clear warning if EE pass-through is
        # not enabled on the leader's Robot config.
        for side, arm in (
            ("left", self._cam_robot.left_arm),
            ("right", self._cam_robot.right_arm),
        ):
            parsed = poll_gripper_status(arm, function_code=0x03, timeout_s=0.5)
            if parsed is None:
                logger.warning(
                    f"{side} gripper FC03 warmup failed — obs.gripper.{side} "
                    "will record NaN. Verify enable_ee_pass_through=True for "
                    "the leader's Robot config."
                )
            else:
                setattr(self, f"_last_obs_grip_{side}", float(parsed["actual"]))
        self._grip_monitors = {
            "left": RobotiqStatusMonitor(
                self._cam_robot.left_arm, side="left", function_code=0x03
            ),
            "right": RobotiqStatusMonitor(
                self._cam_robot.right_arm, side="right", function_code=0x03
            ),
        }
        for monitor in self._grip_monitors.values():
            monitor.send_status_request()

        rate_limiter = RateLimiter(self.publish_rate)
        if self._debug_display:
            self._debug_display.start()
        console.rule("[bold cyan]Stage static — hold right trigger ≥ 1 s to start head tracking")

        try:
            step = 0
            while self.running:
                transforms = self.quest.get_latest_transformation()
                if transforms is None:
                    rate_limiter.sleep()
                    continue

                if self._head_keys:
                    head_imgs = self._cam_robot.sensors.head_camera.get_obs(
                        obs_keys=self._head_keys
                    )
                    self._last_head_imgs = head_imgs
                    if "depth" in head_imgs:
                        self._last_head_depth_u16 = np.clip(
                            head_imgs["depth"] * 1000, 0, 65535
                        ).astype(np.uint16)
                if self._wrist_enabled:
                    # dexsensor usb_camera publishes the raw ZED-M output as
                    # one (H, 2W, 3) RGB frame; slice into the eyes we need.
                    wrist_frame = self._cam_robot.sensors.wrist_zedm.get_obs()
                    if wrist_frame is not None:
                        if (
                            wrist_frame.ndim != 3
                            or wrist_frame.shape[2] != 3
                            or wrist_frame.shape[1] % 2 != 0
                        ):
                            raise ValueError(
                                f"Unexpected wrist frame shape {wrist_frame.shape}; "
                                "expected (H, 2W, 3) side-by-side stereo RGB."
                            )
                        half_w = wrist_frame.shape[1] // 2
                        # .copy() each half: get_obs() hands back a reused capture
                        # buffer the driver overwrites in place, so a bare view
                        # tears in the live preview and corrupts recorded frames
                        # (which are only np.stack-copied at save time).
                        for half in self._wrist_halves:
                            if half == "left_rgb":
                                self._last_wrist_imgs["left_rgb"] = wrist_frame[
                                    :, :half_w, :
                                ].copy()
                            elif half == "right_rgb":
                                self._last_wrist_imgs["right_rgb"] = wrist_frame[
                                    :, half_w:, :
                                ].copy()
                if step % 4 == 0:
                    self._camera_poll()
                vr_head = transforms["head"]
                vr_l = transforms["left_wrist"]
                vr_r = transforms["right_wrist"]

                self._poll_gripper_status_step()

                if self._calib_stage == "resetting":
                    head_pos, left_pos, right_pos = self._resetting_step(transforms)
                    self._mark_ik_skipped("skipped_resetting")
                    ik_l, ik_r = None, None
                else:
                    head_pos = self._head_ik_step(vr_head)
                    left_pos, right_pos, ik_l, ik_r = self._arm_ik_step(vr_l, vr_r)
                if self._workspace_check_enabled:
                    self._update_left_eef_workspace_status(left_pos)
                    self._update_right_eef_workspace_status(right_pos)
                chassis_vx, chassis_vy, chassis_wz = self._thumbstick_to_chassis(transforms)

                self._handle_stage_transition(transforms)
                left_pos = self._handle_recording(
                    transforms,
                    vr_l,
                    vr_r,
                    head_pos,
                    left_pos,
                    right_pos,
                    chassis_vx,
                    chassis_vy,
                    chassis_wz,
                )
                self._publish(
                    head_pos, left_pos, right_pos, transforms, chassis_vx, chassis_vy, chassis_wz
                )

                if left_pos and self._debug_display:
                    joints = {f"L_arm_j{i+1}": left_pos[i] for i in range(7)}
                    joints.update({f"R_arm_j{i+1}": right_pos[i] for i in range(7)})
                    self._debug_display.print_leader_arm(joints)

                if self.visualize and not self._update_visualization(ik_l, ik_r):
                    break

                rate_limiter.sleep()

        except KeyboardInterrupt:
            logger.info("VRReader stopped by user.")
        finally:
            self.running = False
            if self.recorder.recording:
                self.recorder.stop()
            if self.debug_recorder.recording:
                self.debug_recorder.stop()
            for r in (self.recorder, self.debug_recorder):
                save_thread = r._save_thread  # noqa: SLF001 - recorder exposes no join helper
                if save_thread is not None and save_thread.is_alive():
                    logger.info(f"Waiting for {type(r).__name__} save to finish ...")
                    save_thread.join()
            for monitor in self._grip_monitors.values():
                monitor.close()
            if self._debug_display:
                self._debug_display.stop()
            self.quest.close()
            self.node.shutdown()


# ── CLI entry point ────────────────────────────────────────────────────────────


def main() -> None:
    setup_logging()

    @dataclasses.dataclass
    class Args:
        robot_name: str = "vega_no_effector"
        """KinHelper robot model name"""

        head_link: str = "zed_depth_frame"
        """URDF link used as head end-effector"""

        left_arm_link: str = "L_ee"
        """URDF link used as left arm end-effector"""

        right_arm_link: str = "R_ee"
        """URDF link used as right arm end-effector"""

        stick_max_vx: float = 0.3
        """Max forward speed from left thumbstick (m/s)"""

        stick_max_vy: float = 0.2
        """Max lateral speed from left thumbstick (m/s)"""

        stick_max_wz: float = 0.5
        """Max yaw rate from right thumbstick (rad/s)"""

        stick_deadzone: float = 0.1
        """Thumbstick deadzone"""

        publish_rate: float = 15.0
        """Main loop rate (Hz). VR-pose poll, IK solve and command publish all
        run at this rate. Default matches record_rate so each loop iter records
        one frame (no throttle skipping). Must be ≥ record_rate; pick an integer
        multiple of record_rate for zero-jitter recording (e.g. 30 → 15)."""

        record_rate: float = 15.0
        """Episode recording rate (Hz). Frames are throttled so saved HDF5
        files have a consistent FPS independent of publish_rate."""

        namespace: str = ""
        """Zenoh topic namespace prefix"""

        debug: bool = False
        """Print calculated arm joint positions to terminal"""

        visualize: bool = True
        """Open Sapien viewer to visualize IK in real time"""

        urdf_path: str = "/home/yixuan/yixuan_utilities/src/yixuan_utilities/assets/robot/vega-urdf/vega_no_effector.urdf"  # noqa
        """Path to robot URDF for Sapien visualizer (required if --visualize)"""

        save_dir: str = "/home/yixuan/Dexmate/data/raw_data"
        """Directory to save episode_<N>.hdf5 files (A=start, B=stop)"""

        debug_save_dir: str = "/home/yixuan/omniteleop/Dexmate/debug/raw_data"
        """Directory to save episode_<N>_debug.hdf5 files"""

        start_mode: StartMode = "fixed_pose"
        """Episode start behaviour. 'follow_hand': arm interpolates to current
        controller pose after Y-reset (existing behaviour). 'fixed_pose': arm
        holds at FIXED_*_ARM_JOINTS so every recording begins from an identical
        configuration; user trigger-advances out of whole_body_alignment to
        lock the per-arm calibration."""

        save_debug: bool = False
        """If False, skip building and saving episode_<N>_debug.hdf5 to reduce
        per-frame overhead and shorten save time."""

        workspace_check: bool = workspace_check
        """Compute right-arm EEF FK each frame and overlay 'WARNING: out of
        workspace' on the headset view when the commanded EEF leaves the
        configured Cartesian bounds. Pass --workspace-check to enable."""

        cameras: list[CameraStream] = dataclasses.field(
            default_factory=lambda: list(_DEFAULT_CAMERAS)
        )
        """Camera streams to read and record. Choose any subset of:
        head_left_rgb, head_right_rgb, head_depth, left_wrist_rgb,
        right_wrist_rgb. Default: head_left_rgb head_depth left_wrist_rgb.
        Wrist streams require `dexsensor launch --sensor wrist_zedm` to be
        running. wrist_depth is rejected — dexsensor's usb_camera driver
        does not run the ZED SDK and cannot compute depth from USB ZED-M."""

    args = tyro.cli(Args)

    reader = VRReader(
        namespace=args.namespace,
        robot_name=args.robot_name,
        head_link=args.head_link,
        left_arm_link=args.left_arm_link,
        right_arm_link=args.right_arm_link,
        stick_max_vx=args.stick_max_vx,
        stick_max_vy=args.stick_max_vy,
        stick_max_wz=args.stick_max_wz,
        stick_deadzone=args.stick_deadzone,
        publish_rate=args.publish_rate,
        record_rate=args.record_rate,
        debug=args.debug,
        visualize=args.visualize,
        urdf_path=args.urdf_path,
        save_dir=args.save_dir,
        debug_save_dir=args.debug_save_dir,
        start_mode=args.start_mode,
        save_debug=args.save_debug,
        workspace_check=args.workspace_check,
        cameras=args.cameras,
    )
    reader.run()


if __name__ == "__main__":
    main()
