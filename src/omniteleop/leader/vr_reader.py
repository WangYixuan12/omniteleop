#!/usr/bin/env python3
"""VR leader node for teleoperation via Oculus Quest 3.

Architecture
------------
IK is solved **here on the leader side** using KinHelper (yixuan_utilities /
Sapien+Pinocchio backend).  The follower VRHandler receives ready-made joint
positions and issues direct JOINT commands – the same path used by the
exoskeleton pipeline.

Data flow::

    Quest3 (Unity SocketIO)
        ↓  protobuf  (controller mode: head + wrist poses + triggers/clicks)
    OpenXRUnitySocketClient
        ↓  4×4 transforms already in robot-global frame
    VRReader._run_stage_*
        ├─ KinHelper.compute_ik_from_mat()  (head, left arm, right arm)
        └─ Zenoh publisher → "vr/joints"   (VRJointData dict)
                                                ↓
                                        VRHandler (follower side)

Calibration workflow (console prompts)
--------------------------------------
Stage A  Robot head tracks VR headset orientation.
         Press RIGHT INDEX TRIGGER (> 0.7) to confirm → Stage B.
Stage B  Both arms preview tracking (head-relative).
         Press RIGHT INDEX TRIGGER to confirm → Stage C (live).
Stage C  Active teleoperation.
         Both thumbstick-clicks held ≥ 1.5 s  → toggle emergency-stop.
         Ctrl-C                                → clean shutdown.

Usage::

    # With ROBOT_CONFIG pointing at the quest3 profile:
    ROBOT_CONFIG=vega_1_quest3 omni-vr

    # Or with explicit args:
    python src/omniteleop/leader/vr_reader.py --server-url http://10.0.0.1:5066
"""

from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import asdict
from typing import Optional

import numpy as np
import tyro
from loguru import logger
from rich.console import Console

from dexcomm import Node, RateLimiter
from dexcomm.codecs import DictDataCodec

from omniteleop.common import get_config
from omniteleop.common.logging import setup_logging
from omniteleop.common.schemas import VRJointData
from omniteleop.leader.communication.openxr_socket_client import OpenXRUnitySocketClient

console = Console()

# ── Joint name sets for active_qmask construction ────────────────────────────
_HEAD_JOINTS = frozenset(
    {"torso_j1", "torso_j2", "torso_j3", "head_j1", "head_j2", "head_j3"}
)
_LEFT_ARM_JOINTS = frozenset(f"L_arm_j{i}" for i in range(1, 8))
_RIGHT_ARM_JOINTS = frozenset(f"R_arm_j{i}" for i in range(1, 8))

# Map component name → ordered joint names (matches robot/joints feedback format)
_COMP_JOINT_NAMES: dict[str, list[str]] = {
    "left_arm":  [f"L_arm_j{i}" for i in range(1, 8)],
    "right_arm": [f"R_arm_j{i}" for i in range(1, 8)],
    "torso":     ["torso_j1", "torso_j2", "torso_j3"],
    "head":      ["head_j1", "head_j2", "head_j3"],
}


class VRReader:
    """Connect to Quest3, solve IK with KinHelper, publish VRJointData.

    Parameters
    ----------
    server_url:      Quest Unity SocketIO server URL.
    namespace:       Zenoh topic namespace prefix.
    robot_name:      KinHelper robot model name (e.g. "vega_no_effector").
    head_link:       URDF link name used as head end-effector.
    left_arm_link:   URDF link name used as left arm end-effector.
    right_arm_link:  URDF link name used as right arm end-effector.
    arm_scale:       Scale applied to the head-relative wrist translation so that
                     the robot's shorter/longer arms still reach natural poses.
                     Set to (robot_arm_reach / human_arm_reach); default 1.0.
    publish_rate:    Hz for the Zenoh publishing loop.
    estop_hold_s:    Seconds both thumbstick-clicks must be held to toggle estop.
    """

    def __init__(
        self,
        server_url: str,
        namespace: str = "",
        robot_name: str = "vega_no_effector",
        head_link: str = "head_l1",
        left_arm_link: str = "L_ee",
        right_arm_link: str = "R_ee",
        arm_scale: float = 1.0,
        publish_rate: float = 40.0,
        estop_hold_s: float = 1.5,
        debug: bool = False,
    ) -> None:
        self.arm_scale = arm_scale
        self.publish_rate = publish_rate
        self.estop_hold_s = estop_hold_s
        self.debug = debug

        # ── Zenoh ────────────────────────────────────────────────────────────
        self.node = Node(name="vr_reader", namespace=namespace)
        config = get_config()
        vr_topic = config.get_topic("vr_joints", "vr/joints")
        robot_topic = config.get_topic("robot_joints", "robot/joints")

        self.vr_pub = self.node.create_publisher(vr_topic, encoder=DictDataCodec.encode)
        self.robot_sub = self.node.create_subscriber(
            robot_topic, self._on_robot_joints, decoder=DictDataCodec.decode
        )

        # ── KinHelper (lazy import keeps non-VR imports fast) ────────────────
        try:
            from yixuan_utilities.kinematics_helper import KinHelper  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "yixuan_utilities is required for VR mode.\n"
                "Install: pip install git+https://github.com/WangYixuan12/yixuan_utilities"
            ) from exc

        logger.info(f"Loading KinHelper for robot '{robot_name}' ...")
        self.kin = KinHelper(robot_name)
        self._build_ik_config(head_link, left_arm_link, right_arm_link)

        # ── Quest3 SocketIO client ────────────────────────────────────────────
        logger.info(f"Connecting to Quest server: {server_url} ...")
        self.quest = OpenXRUnitySocketClient(
            server_url, rate_limiter_freq=publish_rate
        )

        # ── Warm-start qpos (updated from robot feedback + IK results) ───────
        self._qpos_lock = threading.Lock()
        self.current_qpos: np.ndarray = self._qpos_from_config(config)

        # ── Run-time state ────────────────────────────────────────────────────
        self.calib_stage: str = "A"
        self.estop: bool = True
        self._confirm_cooldown: float = 0.0
        self._estop_btn_t: float = 0.0
        self._estop_btn_held: bool = False
        self.rate_limiter = RateLimiter(publish_rate)
        self.running = False

    # ── Initialisation helpers ─────────────────────────────────────────────────

    def _build_ik_config(
        self, head_link: str, left_arm_link: str, right_arm_link: str
    ) -> None:
        """Resolve EEF link indices and build active_qmask arrays from joint names."""
        active_joints = self.kin.sapien_robot.get_active_joints()
        self.joint_names: list[str] = [j.name for j in active_joints]
        n = len(self.joint_names)

        # EEF link index lookup
        lni = self.kin.link_name_to_idx
        self.head_eef_idx = lni.get(head_link)
        self.left_arm_eef_idx = lni.get(left_arm_link)
        self.right_arm_eef_idx = lni.get(right_arm_link)

        missing = [
            name
            for name, idx in [
                (head_link, self.head_eef_idx),
                (left_arm_link, self.left_arm_eef_idx),
                (right_arm_link, self.right_arm_eef_idx),
            ]
            if idx is None
        ]
        if missing:
            avail = sorted(lni.keys())
            raise ValueError(
                f"EEF link(s) not found in robot model: {missing}\n"
                f"Available links: {avail}\n"
                "Update head_link / left_arm_link / right_arm_link in the config."
            )

        def _mask(joint_set: frozenset) -> np.ndarray:
            return np.array(
                [name in joint_set for name in self.joint_names], dtype=bool
            )

        self.head_qmask = _mask(_HEAD_JOINTS)
        self.left_arm_qmask = _mask(_LEFT_ARM_JOINTS)
        self.right_arm_qmask = _mask(_RIGHT_ARM_JOINTS)

        # Per-component index lists for slicing full qpos into named arrays
        self.head_indices = [
            i for i, n in enumerate(self.joint_names)
            if n in {"head_j1", "head_j2", "head_j3"}
        ]
        self.left_arm_indices = [
            i for i, n in enumerate(self.joint_names) if n in _LEFT_ARM_JOINTS
        ]
        self.right_arm_indices = [
            i for i, n in enumerate(self.joint_names) if n in _RIGHT_ARM_JOINTS
        ]

        # joint name → (component, index-within-component) for warm-start updates
        self._joint_to_comp: dict[str, tuple[str, int]] = {}
        for comp, names in _COMP_JOINT_NAMES.items():
            for idx, jname in enumerate(names):
                self._joint_to_comp[jname] = (comp, idx)

        logger.info(
            f"IK config ready — head_eef={self.head_eef_idx}, "
            f"L_arm_eef={self.left_arm_eef_idx}, R_arm_eef={self.right_arm_eef_idx}"
        )
        logger.debug(
            "head_qmask joints: "
            f"{[self.joint_names[i] for i in np.where(self.head_qmask)[0]]}"
        )

    def _qpos_from_config(self, config) -> np.ndarray:
        """Build an initial warm-start qpos from config init_pos values."""
        n = len(self.joint_names)
        qpos = np.zeros(n)
        init_pos = config.get("init_pos", {})
        for i, jname in enumerate(self.joint_names):
            comp, idx = self._joint_to_comp.get(jname, (None, None))
            if comp is not None:
                comp_vals = init_pos.get(comp, [])
                if idx < len(comp_vals):
                    qpos[i] = comp_vals[idx]
        return qpos

    def _on_robot_joints(self, data: dict) -> None:
        """Update warm-start qpos from live robot joint feedback (Zenoh callback)."""
        with self._qpos_lock:
            for i, jname in enumerate(self.joint_names):
                comp, idx = self._joint_to_comp.get(jname, (None, None))
                if comp is not None and comp in data:
                    positions = data[comp]
                    if isinstance(positions, list) and idx < len(positions):
                        self.current_qpos[i] = positions[idx]

    # ── IK helpers ───────────────────────────────────────────────────────────

    def _compute_head_ik(self, T_head_world: np.ndarray) -> np.ndarray:
        """Solve IK for the head/torso chain. Returns a full qpos array."""
        with self._qpos_lock:
            qpos_init = self.current_qpos.copy()
        return self.kin.compute_ik_from_mat(
            qpos_init,
            T_head_world,
            eef_idx=self.head_eef_idx,
            active_qmask=self.head_qmask,
        )

    def _compute_arm_ik(
        self,
        T_arm_target: np.ndarray,
        side: str,
        head_qpos: np.ndarray,
    ) -> np.ndarray:
        """Solve IK for one arm, warm-started from head_qpos for consistency."""
        qmask = self.left_arm_qmask if side == "left" else self.right_arm_qmask
        eef_idx = self.left_arm_eef_idx if side == "left" else self.right_arm_eef_idx
        return self.kin.compute_ik_from_mat(
            head_qpos,
            T_arm_target,
            eef_idx=eef_idx,
            active_qmask=qmask,
        )

    def _arm_target_from_head_relative(
        self,
        T_vr_head: np.ndarray,
        T_vr_wrist: np.ndarray,
        T_robot_head: np.ndarray,
    ) -> np.ndarray:
        """Compute robot-frame arm target using a head-relative mapping.

        The wrist pose is expressed in the VR head frame, then transported to
        the robot head frame.  ``arm_scale`` scales the translation so that the
        robot's (shorter) arm can still reach natural positions.
        """
        T_head2wrist_vr = np.linalg.inv(T_vr_head) @ T_vr_wrist
        T_scaled = T_head2wrist_vr.copy()
        T_scaled[:3, 3] *= self.arm_scale
        return T_robot_head @ T_scaled

    def _update_qpos_from_ik(
        self,
        masks_and_qpos: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Thread-safe update of current_qpos from IK results."""
        with self._qpos_lock:
            for mask, qpos in masks_and_qpos:
                for i in np.where(mask)[0]:
                    self.current_qpos[i] = qpos[i]

    # ── Button helpers ────────────────────────────────────────────────────────

    def _is_confirm_pressed(self, transforms: dict) -> bool:
        """Return True when right index trigger > 0.7, debounced by 1 s."""
        val = transforms.get("right_index_trigger", 0.0)
        now = time.monotonic()
        if val > 0.7 and now > self._confirm_cooldown:
            self._confirm_cooldown = now + 1.0
            return True
        return False

    def _handle_estop_toggle(self, transforms: dict) -> None:
        """Toggle estop when both thumbstick-clicks are held for estop_hold_s."""
        both = transforms.get("left_thumbstick_click", False) and transforms.get(
            "right_thumbstick_click", False
        )
        now = time.monotonic()
        if both:
            if not self._estop_btn_held:
                self._estop_btn_t = now
                self._estop_btn_held = True
            elif now - self._estop_btn_t >= self.estop_hold_s:
                self.estop = not self.estop
                self._estop_btn_held = False
                state = "ACTIVATED" if self.estop else "RELEASED"
                color = "red" if self.estop else "green"
                console.print(f"[bold {color}]Emergency stop {state}[/]")
        else:
            self._estop_btn_held = False

    # ── Stage runners ─────────────────────────────────────────────────────────

    def _run_stage_a(self, transforms: dict) -> None:
        """Stage A: head-only IK, arms frozen. Confirm → Stage B."""
        T_head = transforms.get("head")
        if T_head is None:
            return

        head_qpos = self._compute_head_ik(T_head)
        head_pos = [float(head_qpos[i]) for i in self.head_indices]
        self._update_qpos_from_ik([(self.head_qmask, head_qpos)])

        data = VRJointData(
            timestamp_ns=time.time_ns(),
            head_pos=head_pos,
            estop=True,
            calib_stage="A",
        )
        self.vr_pub.publish(asdict(data))

        if self._is_confirm_pressed(transforms):
            self.calib_stage = "B"
            console.print(
                "\n[bold]Stage B:[/] Arms now track VR controllers (head-relative).\n"
                "  Position your controllers, then press "
                "[yellow]RIGHT INDEX TRIGGER[/] to go live."
            )

    def _run_stage_b(self, transforms: dict) -> None:
        """Stage B: head + arm IK preview, estop still on. Confirm → Stage C."""
        T_head = transforms.get("head")
        T_Lw = transforms.get("left_wrist")
        T_Rw = transforms.get("right_wrist")
        if T_head is None or T_Lw is None or T_Rw is None:
            return

        head_qpos = self._compute_head_ik(T_head)
        T_robot_head = self.kin.compute_fk_from_link_idx(
            head_qpos, [self.head_eef_idx]
        )[0]

        T_L_target = self._arm_target_from_head_relative(T_head, T_Lw, T_robot_head)
        T_R_target = self._arm_target_from_head_relative(T_head, T_Rw, T_robot_head)

        left_qpos = self._compute_arm_ik(T_L_target, "left", head_qpos)
        right_qpos = self._compute_arm_ik(T_R_target, "right", head_qpos)

        self._update_qpos_from_ik([
            (self.head_qmask, head_qpos),
            (self.left_arm_qmask, left_qpos),
            (self.right_arm_qmask, right_qpos),
        ])

        data = VRJointData(
            timestamp_ns=time.time_ns(),
            head_pos=[float(head_qpos[i]) for i in self.head_indices],
            left_arm_pos=[float(left_qpos[i]) for i in self.left_arm_indices],
            right_arm_pos=[float(right_qpos[i]) for i in self.right_arm_indices],
            left_gripper=0.0,
            right_gripper=0.0,
            estop=True,
            calib_stage="B",
        )
        self.vr_pub.publish(asdict(data))

        if self._is_confirm_pressed(transforms):
            self.calib_stage = "C"
            self.estop = False
            console.print(
                "\n[bold green]► Live teleoperation active![/]  "
                "Hold [yellow]both thumbstick-clicks ≥ 1.5 s[/] to toggle e-stop."
            )

    def _run_stage_c(self, transforms: dict) -> None:
        """Stage C: active teleoperation — head + both arms + gripper."""
        T_head = transforms.get("head")
        T_Lw = transforms.get("left_wrist")
        T_Rw = transforms.get("right_wrist")
        if T_head is None or T_Lw is None or T_Rw is None:
            return

        self._handle_estop_toggle(transforms)

        head_qpos = self._compute_head_ik(T_head)
        T_robot_head = self.kin.compute_fk_from_link_idx(
            head_qpos, [self.head_eef_idx]
        )[0]

        T_L_target = self._arm_target_from_head_relative(T_head, T_Lw, T_robot_head)
        T_R_target = self._arm_target_from_head_relative(T_head, T_Rw, T_robot_head)

        left_qpos = self._compute_arm_ik(T_L_target, "left", head_qpos)
        right_qpos = self._compute_arm_ik(T_R_target, "right", head_qpos)

        self._update_qpos_from_ik([
            (self.head_qmask, head_qpos),
            (self.left_arm_qmask, left_qpos),
            (self.right_arm_qmask, right_qpos),
        ])

        # Proportional gripper control from hand trigger [0 = open, 1 = closed]
        left_gripper = float(np.clip(transforms.get("left_hand_trigger", 0.0), 0.0, 1.0))
        right_gripper = float(np.clip(transforms.get("right_hand_trigger", 0.0), 0.0, 1.0))

        data = VRJointData(
            timestamp_ns=time.time_ns(),
            head_pos=[float(head_qpos[i]) for i in self.head_indices],
            left_arm_pos=[float(left_qpos[i]) for i in self.left_arm_indices],
            right_arm_pos=[float(right_qpos[i]) for i in self.right_arm_indices],
            left_gripper=left_gripper,
            right_gripper=right_gripper,
            estop=self.estop,
            calib_stage="C",
        )
        self.vr_pub.publish(asdict(data))

    # ── Main run loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """Block until Quest data are available, then run calibration + teleop."""
        self.running = True
        logger.info("Waiting for Quest data stream ...")
        self.quest.wait_for_data()
        logger.info("Quest stream active.")

        console.rule("[bold cyan]VR Teleop Calibration")
        console.print(
            "[bold]Stage A:[/] Robot head tracks VR headset orientation.\n"
            "         Press [yellow]RIGHT INDEX TRIGGER[/] (squeeze fully) to confirm."
        )

        try:
            while self.running:
                transforms = self.quest.get_latest_transformation()
                if transforms is None:
                    time.sleep(0.005)
                    continue

                if self.calib_stage == "A":
                    self._run_stage_a(transforms)
                elif self.calib_stage == "B":
                    self._run_stage_b(transforms)
                else:
                    self._run_stage_c(transforms)

                self.rate_limiter.sleep()

        except KeyboardInterrupt:
            logger.info("VRReader stopped by user.")
        finally:
            self.running = False
            self.quest.close()
            self.node.shutdown()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()

    config = get_config()
    vr_cfg = config.get("input_handlers", {}).get("vr", {})
    socket_cfg = vr_cfg.get("socket", {})

    @dataclasses.dataclass
    class Args:
        server_url: str = socket_cfg.get("server_url", "http://localhost:5066")
        """Quest Unity SocketIO server URL"""

        robot_name: str = vr_cfg.get("robot_name", "vega_no_effector")
        """KinHelper robot model name (e.g. 'vega_no_effector')"""

        head_link: str = vr_cfg.get("head_link", "head_l1")
        """URDF link used as head end-effector (run test_vr_ik_sapien.py to discover)"""

        left_arm_link: str = vr_cfg.get("left_arm_link", "L_ee")
        """URDF link used as left arm end-effector"""

        right_arm_link: str = vr_cfg.get("right_arm_link", "R_ee")
        """URDF link used as right arm end-effector"""

        arm_scale: float = vr_cfg.get("arm_scale", 1.0)
        """Reach scale factor: robot_arm_reach / human_arm_reach"""

        namespace: str = ""
        """Zenoh topic namespace prefix"""

        debug: bool = False
        """Enable verbose debug logging"""

    args = tyro.cli(Args)

    reader = VRReader(
        server_url=args.server_url,
        namespace=args.namespace,
        robot_name=args.robot_name,
        head_link=args.head_link,
        left_arm_link=args.left_arm_link,
        right_arm_link=args.right_arm_link,
        arm_scale=args.arm_scale,
        debug=args.debug,
    )
    reader.run()


if __name__ == "__main__":
    main()
