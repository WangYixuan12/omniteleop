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
        ↓  4×4 transforms in robot-global frame
    VRReader._run_loop()
        ├─ KinHelper.compute_ik_from_mat()      (head/torso)
        ├─ MotionManager.ik(type="pink")         (arms, self-collision avoidance)
        ├─ thumbstick → chassis velocity
        └─ Zenoh publisher → "vr/joints"         (VRJointData)
                                                      ↓
                                              VRRobotController (follower)

Calibration stages
------------------
0           Initial — hold right index trigger ≥ 1 s → Stage A.
A           Head tracks VR headset.
            Hold right index trigger ≥ 1 s → Stage B_approach.
B_approach  Arms interpolate toward current controller positions.
            Auto-advances to Stage B when both distances < 0.02 m.
B           Live tracking with self-collision avoidance.

Published calib_stage (VRJointData):
  internal 0/A    → "A",  estop=True
  internal B_approach → "B",  estop=True
  internal B       → "C",  estop=False

Usage::

    omni-vr
    # or with SSL for WiFi:
    python -m omniteleop.leader.vr_reader --ssl-certfile cert.pem --ssl-keyfile key.pem
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from loguru import logger
from rich.console import Console
import tyro

from dexcomm import Node, RateLimiter
from dexcomm.codecs import DictDataCodec
from yixuan_utilities.kinematics_helper import KinHelper

from omniteleop.common import get_config
from omniteleop.common.logging import setup_logging
from omniteleop.common.schemas import VRJointData
from omniteleop.common.debug_display import get_debug_display
from omniteleop.common.log_utils import suppress_loguru_module
from omniteleop.leader.communication.webxr_vr_reader import WebXRVRReader
from dexmotion.motion_manager import MotionManager
from omniteleop.follower.component_processors import ArmProcessor
from dexbot_utils import RobotInfo

console = Console()

# Sentinel poses emitted by WebXRVRReader when a controller is not tracked
INVALID_LEFT_POSE = np.array([
    [ 0.,  0.,  1.,  0.],
    [-1.,  0.,  0.,  0.],
    [ 0., -1.,  0.,  0.],
    [ 0.,  0.,  0.,  1.],
])
INVALID_RIGHT_POSE = np.array([
    [ 0.,  0.,  1.,  0.],
    [ 1.,  0.,  0.,  0.],
    [ 0.,  1.,  0.,  0.],
    [ 0.,  0.,  0.,  1.],
])

_HEAD_IK_JOINTS  = {"head_j1", "head_j2", "head_j3"}
_HEAD_MOTOR_JOINTS = ["head_j1", "head_j2", "head_j3"]
_TORSO_JOINTS    = ["torso_j1", "torso_j2", "torso_j3"]
_LEFT_ARM_JOINTS  = [f"L_arm_j{i}" for i in range(1, 8)]
_RIGHT_ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]

_INTERP_ALPHA = 0.1   # approach interpolation step per frame


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
        publish_rate: float = 40.0,
        debug: bool = False,
    ) -> None:
        self.stick_max_vx   = stick_max_vx
        self.stick_max_vy   = stick_max_vy
        self.stick_max_wz   = stick_max_wz
        self.stick_deadzone = stick_deadzone
        self.publish_rate   = publish_rate
        self.running        = False

        self._debug_display = get_debug_display("VRReader", publish_rate, refresh_rate=10) if debug else None

        # Zenoh
        self.node    = Node(name="vr_reader", namespace=namespace)
        config       = get_config()
        vr_topic     = config.get_topic("vr_joints", "vr/joints")
        self.vr_pub  = self.node.create_publisher(vr_topic, encoder=DictDataCodec.encode)

        # WebXR reader
        self.quest = WebXRVRReader(port=5067, ssl_certfile="/home/yixuan/omniteleop/tests/cert.pem", ssl_keyfile="/home/yixuan/omniteleop/tests/key.pem")

        # KinHelper (head IK only)
        logger.info(f"Loading KinHelper for '{robot_name}' ...")
        self.kin = KinHelper(robot_name)
        self._build_ik_config(head_link, left_arm_link, right_arm_link)

        # MotionManager + ArmProcessors
        logger.info("Initialising MotionManager ...")
        robot_info      = RobotInfo()
        init_pos_config = config.get("init_pos", {})
        self.mm = MotionManager(init_visualizer=False, joint_regions_to_lock=["BASE"])
        self.mm.left_arm.set_joint_pos(init_pos_config.get("left_arm",  [0.0] * 7))
        self.mm.right_arm.set_joint_pos(init_pos_config.get("right_arm", [0.0] * 7))
        if robot_info.has_torso:
            self.mm.torso.set_joint_pos(init_pos_config.get("torso", [0.0] * 3))
        self.left_proc  = ArmProcessor("left",  config, self.mm, robot_info, "vr")
        self.right_proc = ArmProcessor("right", config, self.mm, robot_info, "vr")

        # Warm-start qpos for head IK
        self.current_qpos = self._build_init_qpos(init_pos_config)

        self._trigger_start: Optional[float] = None

    # ── Setup helpers ──────────────────────────────────────────────────────────

    def _build_ik_config(
        self, head_link: str, left_arm_link: str, right_arm_link: str
    ) -> None:
        active_joints         = self.kin.sapien_robot.get_active_joints()
        self.joint_names      = [j.name for j in active_joints]
        self.joint_name_to_idx = {n: i for i, n in enumerate(self.joint_names)}

        lni = self.kin.link_name_to_idx
        self.head_eef_idx      = lni.get(head_link)
        self.left_arm_eef_idx  = lni.get(left_arm_link)
        self.right_arm_eef_idx = lni.get(right_arm_link)

        missing = [
            n for n, i in [
                (head_link, self.head_eef_idx),
                (left_arm_link, self.left_arm_eef_idx),
                (right_arm_link, self.right_arm_eef_idx),
            ] if i is None
        ]
        if missing:
            raise ValueError(f"EEF links not found in robot model: {missing}")

        self.head_qmask = np.array(
            [n in _HEAD_IK_JOINTS for n in self.joint_names], dtype=bool
        )
        self.torso_indices = [
            self.joint_name_to_idx[n] for n in _TORSO_JOINTS
            if n in self.joint_name_to_idx
        ]
        self.head_motor_indices = [
            self.joint_name_to_idx[n] for n in _HEAD_MOTOR_JOINTS
            if n in self.joint_name_to_idx
        ]

        logger.info(
            f"IK config — head_eef={self.head_eef_idx}  "
            f"L_arm_eef={self.left_arm_eef_idx}  R_arm_eef={self.right_arm_eef_idx}  "
            f"head_DOF={self.head_qmask.sum()}"
        )

    def _build_init_qpos(self, init_pos_config: dict) -> np.ndarray:
        qpos = np.zeros(len(self.joint_names))
        for comp, joints in [
            ("left_arm",  _LEFT_ARM_JOINTS),
            ("right_arm", _RIGHT_ARM_JOINTS),
            ("torso",     _TORSO_JOINTS),
        ]:
            vals = init_pos_config.get(comp, [])
            for idx, jname in enumerate(joints):
                if idx < len(vals) and jname in self.joint_name_to_idx:
                    qpos[self.joint_name_to_idx[jname]] = vals[idx]
        return qpos

    # ── Per-frame helpers ──────────────────────────────────────────────────────

    def _trigger_held(self, transforms: dict, duration: float = 1.0) -> bool:
        """True once right index trigger continuously held for `duration` s."""
        val = transforms.get("right_hand_trigger", 0.0)
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

    def _thumbstick_to_chassis(
        self, transforms: dict
    ) -> tuple[float, float, float]:
        def dz(v: float) -> float:
            return v if abs(v) > self.stick_deadzone else 0.0

        l = transforms.get("left_thumbstick",  [0.0, 0.0])
        r = transforms.get("right_thumbstick", [0.0, 0.0])
        # stick y: up = negative on Quest → negate for forward = positive
        vx = -dz(l[1]) * self.stick_max_vx
        vy = -dz(l[0]) * self.stick_max_vy
        wz =  dz(r[0]) * self.stick_max_wz
        return vx, vy, wz

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.running = True
        self.quest.start()
        logger.info("Waiting for Quest data ...")
        self.quest.wait_for_data()
        logger.info("Quest connected.")

        # Capture VR→robot calibration transform from first frame
        init_tf = self.quest.get_latest_transformation()
        robot_base_t_vr_base = (
            self.kin.compute_fk_from_link_idx(self.current_qpos, [self.head_eef_idx])[0]
            @ np.linalg.inv(init_tf["head"])
        )

        # Approach targets (updated when B_approach stage begins)
        curr_left_target = self.kin.compute_fk_from_link_idx(
            self.current_qpos, [self.left_arm_eef_idx]
        )[0].copy()
        curr_right_target = self.kin.compute_fk_from_link_idx(
            self.current_qpos, [self.right_arm_eef_idx]
        )[0].copy()

        calib_stage  = "0"
        rate_limiter = RateLimiter(self.publish_rate)

        if self._debug_display:
            self._debug_display.start()

        console.rule("[bold cyan]Stage 0 — hold right trigger ≥ 1 s to start head tracking")

        try:
            while self.running:
                transforms = self.quest.get_latest_transformation()
                if transforms is None:
                    rate_limiter.sleep()
                    continue

                vr_head    = transforms["head"]
                vr_l_wrist = transforms["left_wrist"]
                vr_r_wrist = transforms["right_wrist"]

                # ── Head IK ───────────────────────────────────────────────────
                head_pos: list[float] = []
                if calib_stage != "0":
                    head_qpos = self.kin.compute_ik_from_mat(
                        self.current_qpos,
                        robot_base_t_vr_base @ vr_head,
                        eef_idx=self.head_eef_idx,
                        active_qmask=self.head_qmask,
                        damp=1000.0,
                    )
                    # Sync torso in MotionManager for warm-start arm IK
                    self.mm.torso.set_joint_pos(
                        head_qpos[self.torso_indices].tolist()
                    )
                    for i in np.where(self.head_qmask)[0]:
                        self.current_qpos[i] = head_qpos[i]
                    head_pos = [float(head_qpos[i]) for i in self.head_motor_indices]

                # ── Arm IK ────────────────────────────────────────────────────
                valid_wrists = (
                    not np.allclose(vr_l_wrist, INVALID_LEFT_POSE) and
                    not np.allclose(vr_r_wrist, INVALID_RIGHT_POSE)
                )
                left_arm_pos:  list[float] = []
                right_arm_pos: list[float] = []

                if calib_stage in ("B_approach", "B") and valid_wrists:
                    robot_l = robot_base_t_vr_base @ vr_l_wrist
                    robot_r = robot_base_t_vr_base @ vr_r_wrist

                    if calib_stage == "B_approach":
                        for curr, tgt in [
                            (curr_left_target,  robot_l),
                            (curr_right_target, robot_r),
                        ]:
                            curr[:3, 3] += _INTERP_ALPHA * (tgt[:3, 3] - curr[:3, 3])
                            curr[:3, :3] = Slerp(
                                [0, 1],
                                Rotation.concatenate([
                                    Rotation.from_matrix(curr[:3, :3]),
                                    Rotation.from_matrix(tgt[:3, :3]),
                                ]),
                            )(_INTERP_ALPHA).as_matrix()

                        dist_l = np.linalg.norm(curr_left_target[:3, 3]  - robot_l[:3, 3])
                        dist_r = np.linalg.norm(curr_right_target[:3, 3] - robot_r[:3, 3])
                        print(f"Approaching  L:{dist_l:.3f}  R:{dist_r:.3f} m", end="\r")
                        if dist_l < 0.02 and dist_r < 0.02:
                            calib_stage = "B"
                            console.print("\n[bold green]Live tracking active.[/]")
                        ik_left_target  = curr_left_target
                        ik_right_target = curr_right_target
                    else:
                        ik_left_target  = robot_l
                        ik_right_target = robot_r

                    with suppress_loguru_module("dexmotion", enabled=True):
                        arm_solution, in_collision, within_limits = self.mm.ik(
                            target_pose={
                                "L_ee": ik_left_target,
                                "R_ee": ik_right_target,
                            },
                            type="pink",
                        )

                    if arm_solution and not in_collision and within_limits:
                        safe_left  = self.left_proc.limit_joint_step(
                            [arm_solution[f"L_arm_j{i}"] for i in range(1, 8)]
                        )
                        safe_right = self.right_proc.limit_joint_step(
                            [arm_solution[f"R_arm_j{i}"] for i in range(1, 8)]
                        )
                        self.left_proc.apply_positions(safe_left)
                        self.right_proc.apply_positions(safe_right)
                        left_arm_pos  = safe_left.tolist()
                        right_arm_pos = safe_right.tolist()
                        if self._debug_display:
                            joints = {f"L_arm_j{i+1}": left_arm_pos[i]  for i in range(7)}
                            joints.update({f"R_arm_j{i+1}": right_arm_pos[i] for i in range(7)})
                            self._debug_display.print_leader_arm(joints)
                    elif in_collision:
                        console.print("[red]Self-collision — holding arm position[/]")

                # ── Chassis from thumbsticks ───────────────────────────────────
                chassis_vx, chassis_vy, chassis_wz = self._thumbstick_to_chassis(transforms)

                # ── Stage transitions ─────────────────────────────────────────
                if self._trigger_held(transforms):
                    if calib_stage == "0":
                        calib_stage = "A"
                        console.rule(
                            "[bold cyan]Stage A — head tracking. Hold trigger → arm approach"
                        )
                    elif calib_stage == "A":
                        calib_stage = "B_approach"
                        curr_left_target = self.kin.compute_fk_from_link_idx(
                            self.current_qpos, [self.left_arm_eef_idx]
                        )[0].copy()
                        curr_right_target = self.kin.compute_fk_from_link_idx(
                            self.current_qpos, [self.right_arm_eef_idx]
                        )[0].copy()
                        console.rule("[bold cyan]Stage B — approaching arm targets…")

                # ── Publish ───────────────────────────────────────────────────
                published_stage = "C" if calib_stage == "B" else (
                    "B" if calib_stage == "B_approach" else "A"
                )
                estop = calib_stage != "B"

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
                    calib_stage=published_stage,
                )
                self.vr_pub.publish(asdict(data))

                rate_limiter.sleep()

        except KeyboardInterrupt:
            logger.info("VRReader stopped by user.")
        finally:
            self.running = False
            if self._debug_display:
                self._debug_display.stop()
            self.quest.close()
            self.node.shutdown()


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()

    config = get_config()
    vr_cfg = config.get("input_handlers", {}).get("vr", {})
    chassis_cfg = vr_cfg.get("chassis", {})

    import dataclasses

    @dataclasses.dataclass
    class Args:
        robot_name: str = vr_cfg.get("robot_name", "vega_no_effector")
        """KinHelper robot model name"""

        head_link: str = vr_cfg.get("head_link", "zed_depth_frame")
        """URDF link used as head end-effector"""

        left_arm_link: str = vr_cfg.get("left_arm_link", "L_ee")
        """URDF link used as left arm end-effector"""

        right_arm_link: str = vr_cfg.get("right_arm_link", "R_ee")
        """URDF link used as right arm end-effector"""

        stick_max_vx: float = chassis_cfg.get("max_vx", 0.3)
        """Max forward speed from left thumbstick (m/s)"""

        stick_max_vy: float = chassis_cfg.get("max_vy", 0.2)
        """Max lateral speed from left thumbstick (m/s)"""

        stick_max_wz: float = chassis_cfg.get("max_wz", 0.5)
        """Max yaw rate from right thumbstick (rad/s)"""

        stick_deadzone: float = chassis_cfg.get("deadzone", 0.1)
        """Thumbstick deadzone"""

        namespace: str = ""
        """Zenoh topic namespace prefix"""

        debug: bool = False
        """Print calculated arm joint positions to terminal"""

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
    )
    reader.run()


if __name__ == "__main__":
    main()
