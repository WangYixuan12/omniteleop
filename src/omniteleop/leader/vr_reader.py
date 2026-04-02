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
        ↓  4x4 transforms in robot-global frame
    VRReader._run_loop()
        ├─ KinHelper.compute_ik_from_mat()      (head/torso)
        ├─ MotionManager.ik(type="pink")         (arms, self-collision avoidance)
        ├─ thumbstick → chassis velocity
        └─ Zenoh publisher → "vr/joints"         (VRJointData)
                                                      ↓
                                              VRRobotController (follower)

Calibration stages
------------------
static              Initial — hold right index trigger ≥ 1 s → head.
head                Head tracks VR headset.
                    Hold right index trigger ≥ 1 s → whole_body_alignment.
whole_body_alignment  Arms interpolate toward current controller positions.
                    Auto-advances to whole_body when both distances < 0.02 m.
whole_body          Live tracking with self-collision avoidance.

Published calib_stage (VRJointData):
  internal static/head          → "A",  estop=True
  internal whole_body_alignment → "B",  estop=True
  internal whole_body           → "C",  estop=False

Usage::

    omni-vr
    # or with SSL for WiFi:
    python -m omniteleop.leader.vr_reader --ssl-certfile cert.pem --ssl-keyfile key.pem
"""

from __future__ import annotations

import base64
import dataclasses
import pathlib
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, Optional

import cv2
import numpy as np
import sapien
import tyro
from dexbot_utils import RobotInfo
from dexcomm import Node, RateLimiter
from dexcomm.codecs import DictDataCodec
from dexmotion.motion_manager import MotionManager
from loguru import logger
from rich.console import Console
from scipy.spatial.transform import Rotation, Slerp
from yixuan_utilities.hdf5_utils import save_dict_to_hdf5
from yixuan_utilities.kinematics_helper import KinHelper

from omniteleop.common import get_config
from omniteleop.common.debug_display import get_debug_display
from omniteleop.common.log_utils import suppress_loguru_module
from omniteleop.common.logging import setup_logging
from omniteleop.common.schemas import VRJointData
from omniteleop.common.vr_mode_const import (
    INIT_HEAD_JOINTS,
    INIT_JOINTS_DICT,
    INIT_LEFT_ARM_JOINTS,
    INIT_RIGHT_ARM_JOINTS,
    INIT_TORSO_JOINTS,
)
from omniteleop.follower.component_processors import ArmProcessor
from omniteleop.leader.communication.webxr_vr_reader import VRFrame, WebXRVRReader

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
    import sapien

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


def _recursive_np_stack(list_of_dicts: list[dict]) -> dict:
    """Recursively np.stack a list of nested dicts into a single nested dict of arrays."""
    result: dict = {}
    for key in list_of_dicts[0]:
        val = list_of_dicts[0][key]
        if isinstance(val, dict):
            result[key] = _recursive_np_stack([d[key] for d in list_of_dicts])
        else:
            result[key] = np.stack([d[key] for d in list_of_dicts])
    return result


class EpisodeRecorder:
    """Accumulates per-frame data and saves to HDF5 on stop()."""

    def __init__(self, save_dir: str) -> None:
        self._save_dir = pathlib.Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._frames: list[dict] = []
        self.recording = False
        self.episode_id = len(list(self._save_dir.glob("episode_*.hdf5")))

    def start(self) -> None:
        """Start a new episode recording."""
        self._frames = []
        self.recording = True
        logger.info("EpisodeRecorder: recording started")

    def record(self, frame: dict) -> None:
        """Record a frame (VRJointData as dict)."""
        self._frames.append(frame)

    def stop(self) -> Optional[str]:
        """Stop recording and save to HDF5. Returns path if saved, else None."""
        self.recording = False
        if not self._frames:
            logger.warning("EpisodeRecorder: 0 frames — skipping save")
            return None
        path = self._save_dir / f"episode_{self.episode_id}.hdf5"
        data = _recursive_np_stack(self._frames)
        save_dict_to_hdf5(data, {}, str(path))
        logger.info(f"EpisodeRecorder: {len(self._frames)} frames → {path}")
        self._frames = []
        self.episode_id += 1
        return str(path)

    def num_frames(self) -> int:
        """Return number of frames recorded so far"""
        return len(self._frames)


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
        publish_rate: float = 10.0,
        debug: bool = False,
        visualize: bool = False,
        urdf_path: Optional[str] = None,
        save_dir: str = "data",
    ) -> None:
        self.stick_max_vx = stick_max_vx
        self.stick_max_vy = stick_max_vy
        self.stick_max_wz = stick_max_wz
        self.stick_deadzone = stick_deadzone
        self.publish_rate = publish_rate
        self.running = False

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

        # Camera polling via Robot API
        from dexcontrol.core.config import get_robot_config
        from dexcontrol.robot import Robot as _Robot

        configs = get_robot_config()
        configs.sensors["head_camera"].enabled = True
        self._cam_robot = _Robot(configs=configs)
        logger.info("Camera streaming started")

        # KinHelper (head IK only)
        logger.info(f"Loading KinHelper for '{robot_name}' ...")
        self.kin = KinHelper(robot_name)
        self._build_ik_config(head_link, left_arm_link, right_arm_link)

        # MotionManager + ArmProcessors
        logger.info("Initialising MotionManager ...")
        robot_info = RobotInfo()
        self.mm = MotionManager(init_visualizer=False, joint_regions_to_lock=["BASE"])
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

        # Recording
        self.recorder = EpisodeRecorder(save_dir)
        self._last_imgs: dict[str, np.ndarray] = {}
        self._last_depth_u16: Optional[np.ndarray] = None
        self._prev_a: bool = False
        self._prev_b: bool = False
        self._prev_x: bool = False
        self._prev_y: bool = False

        # Reset interpolation state (set when entering "resetting" stage)
        self._reset_start_qpos: np.ndarray = np.eye(1)
        self._reset_target_qpos: np.ndarray = np.eye(1)
        self._reset_total_steps: int = 1
        self._reset_step_idx: int = 0

        # Loop state (set in run() before main loop)
        self._calib_stage: str = "static"
        self._robot_base_t_vr_base: np.ndarray = np.eye(4)
        self._curr_left_target: np.ndarray = np.eye(4)
        self._curr_right_target: np.ndarray = np.eye(4)

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
        import sapien

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
        wz = dz(right_thumbstick[0]) * self.stick_max_wz
        return vx, vy, wz

    # ── Camera polling ─────────────────────────────────────────────────────────

    def _camera_poll(self) -> None:
        imgs = self._cam_robot.sensors.head_camera.get_obs(
            obs_keys=["left_rgb", "right_rgb", "depth"]
        )
        self._last_imgs = imgs
        self._last_depth_u16 = np.clip(imgs["depth"] * 1000, 0, 65535).astype(np.uint16)

        for key in ("left_rgb", "right_rgb"):
            img = imgs[key]
            episode_id: int = self.recorder.episode_id
            text: str = f"Episode: {episode_id}"
            if self.recorder.recording:
                text += f", Recording! Step: {self.recorder.num_frames()}"
            cv2.putText(
                img,
                text,
                (10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                thickness=2,
                color=(255, 255, 255),
            )
            small = cv2.resize(img[:, :, ::-1], (320, 180))
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 40])
            self.quest.set_frame_vis(key, base64.b64encode(buf).decode())
        depth = imgs["depth"]
        finite = depth[np.isfinite(depth) & (depth > 0)]
        if len(finite) == 0:
            normalized = np.zeros(depth.shape[:2], dtype=np.uint8)
        else:
            mn, mx = finite.min(), np.percentile(finite, 95)
            normalized = np.clip((depth - mn) / (mx - mn + 1e-6) * 255, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        _, buf = cv2.imencode(".jpg", colored, [cv2.IMWRITE_JPEG_QUALITY, 60])
        self.quest.set_frame_vis("depth", base64.b64encode(buf).decode())

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
        self.mm.torso.set_joint_pos(head_qpos[self.torso_indices].tolist())
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
        with suppress_loguru_module("dexmotion", enabled=True):
            arm_solution, in_collision, within_limits = self.mm.ik(
                target_pose={"L_ee": l_target, "R_ee": r_target},
                type="pink",
            )
        if not arm_solution or in_collision or not within_limits:
            if in_collision:
                console.print("[red]Self-collision — holding arm position[/]")
            return [], []
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

    def _arm_ik_step(
        self, vr_l: np.ndarray, vr_r: np.ndarray
    ) -> tuple[list[float], list[float], Optional[np.ndarray], Optional[np.ndarray]]:
        if self._calib_stage not in ("whole_body_alignment", "whole_body"):
            return [], [], None, None
        if np.allclose(vr_l, INVALID_LEFT_POSE) or np.allclose(vr_r, INVALID_RIGHT_POSE):
            return [], [], None, None
        robot_l = self._robot_base_t_vr_base @ vr_l
        robot_r = self._robot_base_t_vr_base @ vr_r
        if self._calib_stage == "whole_body_alignment":
            ik_l, ik_r = self._approach_step(robot_l, robot_r)
        else:
            ik_l, ik_r = robot_l, robot_r
        left_pos, right_pos = self._solve_and_apply_arm_ik(ik_l, ik_r)
        return left_pos, right_pos, ik_l, ik_r

    def _resetting_step(self, transforms: VRFrame) -> tuple[list[float], list[float], list[float]]:
        """Interpolate joints toward init pose; on completion compute new base transform."""
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

    def _handle_stage_transition(self, transforms: VRFrame) -> None:
        x_now = transforms["left_x_button"]
        y_now = transforms["left_y_button"]

        if x_now and not self._prev_x:
            self._calib_stage = "static"
            console.rule("[bold red]Stage reset → static (left X)")

        if y_now and not self._prev_y:
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
            self._calib_stage = "resetting"
            console.rule("[bold cyan]Resetting to init pose… (left Y)")

        self._prev_x = x_now
        self._prev_y = y_now

        if not self._trigger_held(transforms):
            return
        if self._calib_stage == "static":
            self._calib_stage = "head"
            console.rule("[bold cyan]Stage head — hold trigger → arm approach")
        elif self._calib_stage == "head":
            self._calib_stage = "whole_body_alignment"
            self._curr_left_target = self.kin.compute_fk_from_link_idx(
                self.current_qpos, [self.left_arm_eef_idx]
            )[0].copy()
            self._curr_right_target = self.kin.compute_fk_from_link_idx(
                self.current_qpos, [self.right_arm_eef_idx]
            )[0].copy()
            console.rule("[bold cyan]Stage whole_body_alignment — approaching arm targets…")

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
    ) -> None:
        a_now = transforms["right_a_button"]
        b_now = transforms["right_b_button"]

        if (
            a_now
            and not self._prev_a
            and self._calib_stage == "whole_body"
            and not self.recorder.recording
        ):
            self.recorder.start()
            console.print("[bold green]Recording started (press B to stop)[/]")
        if b_now and not self._prev_b and self.recorder.recording:
            path = self.recorder.stop()
            console.print(f"[bold yellow]Recording saved → {path}[/]")

        self._prev_a = a_now
        self._prev_b = b_now

        if not self.recorder.recording or self._calib_stage != "whole_body":
            return
        if not self._last_imgs or self._last_depth_u16 is None:
            return

        frame = {
            "timestamp_ns": np.int64(time.time_ns()),
            "action": {
                "eef": {
                    "left": (self._robot_base_t_vr_base @ vr_l).astype(np.float32),
                    "right": (self._robot_base_t_vr_base @ vr_r).astype(np.float32),
                },
                "joint": {
                    "left_arm": np.array(left_pos, dtype=np.float32),
                    "right_arm": np.array(right_pos, dtype=np.float32),
                    "head": np.array(head_pos, dtype=np.float32),
                    "chassis_vx": np.float32(vx),
                    "chassis_vy": np.float32(vy),
                    "chassis_wz": np.float32(wz),
                },
            },
            "obs": {
                "joint": {
                    "left_arm": np.array(
                        self._cam_robot.left_arm.get_joint_pos(), dtype=np.float32
                    ),
                    "right_arm": np.array(
                        self._cam_robot.right_arm.get_joint_pos(), dtype=np.float32
                    ),
                    "head": np.array(self._cam_robot.head.get_joint_pos(), dtype=np.float32),
                    "torso": np.array(self._cam_robot.torso.get_joint_pos(), dtype=np.float32),
                },
                "images": {
                    "left_rgb": self._last_imgs["left_rgb"],
                    "right_rgb": self._last_imgs["right_rgb"],
                    "depth": self._last_depth_u16,
                },
            },
        }
        self.recorder.record(frame)

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
        estop = self._calib_stage in ("static", "head", "whole_body_alignment")
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
        self.vr_pub.publish(asdict(data))

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

        rate_limiter = RateLimiter(self.publish_rate)
        if self._debug_display:
            self._debug_display.start()
        console.rule("[bold cyan]Stage static — hold right trigger ≥ 1 s to start head tracking")

        try:
            while self.running:
                transforms = self.quest.get_latest_transformation()
                if transforms is None:
                    rate_limiter.sleep()
                    continue

                self._camera_poll()
                vr_head = transforms["head"]
                vr_l = transforms["left_wrist"]
                vr_r = transforms["right_wrist"]

                if self._calib_stage == "resetting":
                    head_pos, left_pos, right_pos = self._resetting_step(transforms)
                    ik_l, ik_r = None, None
                else:
                    head_pos = self._head_ik_step(vr_head)
                    left_pos, right_pos, ik_l, ik_r = self._arm_ik_step(vr_l, vr_r)
                chassis_vx, chassis_vy, chassis_wz = self._thumbstick_to_chassis(transforms)

                self._handle_stage_transition(transforms)
                self._handle_recording(
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

        namespace: str = ""
        """Zenoh topic namespace prefix"""

        debug: bool = False
        """Print calculated arm joint positions to terminal"""

        visualize: bool = True
        """Open Sapien viewer to visualize IK in real time"""

        urdf_path: str = "/home/yixuan/yixuan_utilities/src/yixuan_utilities/assets/robot/vega-urdf/vega_no_effector.urdf"  # noqa
        """Path to robot URDF for Sapien visualizer (required if --visualize)"""

        save_dir: str = "data"
        """Directory to save HDF5 episodes (A=start, B=stop)"""

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
        debug=args.debug,
        visualize=args.visualize,
        urdf_path=args.urdf_path,
        save_dir=args.save_dir,
    )
    reader.run()


if __name__ == "__main__":
    main()
