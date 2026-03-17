#!/usr/bin/env python3
"""Sapien live-VR IK simulation for Quest 3 teleoperation.

This is the simulation counterpart of ``vr_reader.py``.  It connects to the
Quest 3 Unity app, applies the same staged calibration (A → B → C), runs
real-time IK, and drives a **Sapien** robot model so you can validate the
full tracking pipeline before touching real hardware.

No Zenoh, no robot hardware — Sapien is the only output.

━━━ How to connect the Quest 3 to the local machine ━━━━━━━━━━━━━━━━━━━━━━━━

The Quest 3 runs the **Unity OpenXR Teleop** companion app, which exposes a
SocketIO server on port 5066.

Option A — USB (recommended, lowest latency)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. On the headset: Settings → General → Developer Mode → ON
2. Connect the Quest 3 to the PC via USB-C.
3. On the PC, run once per USB session::

       adb reverse tcp:5066 tcp:5066

   This tunnels port 5066 from the headset to ``localhost:5066`` on the PC.
4. Start the Unity app on the headset (it will wait for a connection).
5. Run the sim::

       python tests/test_vr_ik_sapien.py          # uses http://localhost:5066

Option B — Wi-Fi (same network)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Both PC and Quest 3 on the same Wi-Fi network.
2. On the headset: Settings → Wi-Fi → tap the connected network → note the IP.
3. Start the Unity app on the headset.
4. Run the sim::

       python tests/test_vr_ik_sapien.py --server-url http://<QUEST_IP>:5066

━━━ Calibration stages ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage A  Robot head tracks VR headset.
         Squeeze RIGHT INDEX TRIGGER fully to confirm and advance.
Stage B  Arms preview (head-relative IK). Confirm again to go live.
Stage C  Full live tracking in Sapien.  Close the viewer to exit.

━━━ CLI options ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  --arm-scale FLOAT    Head-relative wrist translation scale (default: 1.0)
  --inspect-only       Print link/joint tables and exit without connecting VR
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import sapien
from rich.console import Console

from yixuan_utilities.kinematics_helper import KinHelper
from omniteleop.leader.communication.webxr_vr_reader import WebXRVRReader
from dexmotion.motion_manager import MotionManager
from omniteleop.common import get_config
from omniteleop.common.log_utils import suppress_loguru_module
from omniteleop.follower.component_processors import ArmProcessor
from dexbot_utils import RobotInfo

# ── Robot and EEF link name configuration ─────────────────────────────────────
# Update these if the link names differ in your URDF.  Run with --inspect-only
# to print all available link names.
ROBOT_NAME = "vega_no_effector"
HEAD_LINK = "zed_depth_frame"
LEFT_ARM_EEF_LINK = "L_ee"
RIGHT_ARM_EEF_LINK = "R_ee"

DEFAULT_SERVER_URL = "http://localhost:5066"

# Joint name sets (should not need changing for standard Vega builds)
HEAD_JOINTS = ["torso_j1", "torso_j2", "torso_j3", "head_j2", "head_j3"]
LEFT_ARM_JOINTS = [f"L_arm_j{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]
TORSO_JOINTS = ["torso_j1", "torso_j2", "torso_j3"]

URDF_PATH = "/home/yixuan/yixuan_utilities/src/yixuan_utilities/assets/robot/vega-urdf/vega_no_effector.urdf"

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_mask(joint_names: list[str], active_set: frozenset) -> np.ndarray:
    return np.array([n in active_set for n in joint_names], dtype=bool)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1-5: inspect robot (always runs)
# ─────────────────────────────────────────────────────────────────────────────

def inspect_and_resolve(kin: KinHelper) -> dict | None:
    """Print link/joint tables and return the resolved IK config dict.

    Returns None if any EEF link name is not found — caller should abort.
    """
    print("=" * 64)
    print(f"Robot: {ROBOT_NAME}")

    # ── All links ─────────────────────────────────────────────────────────────
    print("\n── ALL LINKS (index → name) ─────────────────────────────────")
    for name, idx in sorted(kin.link_name_to_idx.items(), key=lambda x: x[1]):
        marker = " ◄" if name in {HEAD_LINK, LEFT_ARM_EEF_LINK, RIGHT_ARM_EEF_LINK} else ""
        print(f"  [{idx:3d}]  {name}{marker}")

    # ── EEF resolution ────────────────────────────────────────────────────────
    lni = kin.link_name_to_idx
    head_eef_idx = lni.get(HEAD_LINK)
    left_arm_eef_idx = lni.get(LEFT_ARM_EEF_LINK)
    right_arm_eef_idx = lni.get(RIGHT_ARM_EEF_LINK)

    print("\nEEF indices resolved:")
    print(f"  head      ({HEAD_LINK!r})          → {head_eef_idx}")
    print(f"  left_arm  ({LEFT_ARM_EEF_LINK!r}) → {left_arm_eef_idx}")
    print(f"  right_arm ({RIGHT_ARM_EEF_LINK!r}) → {right_arm_eef_idx}")

    missing = [
        n for n, i in [
            (HEAD_LINK, head_eef_idx),
            (LEFT_ARM_EEF_LINK, left_arm_eef_idx),
            (RIGHT_ARM_EEF_LINK, right_arm_eef_idx),
        ]
        if i is None
    ]
    if missing:
        print(
            f"\n⚠  Link(s) not found: {missing}\n"
            "   Update the *_LINK constants at the top of this file and re-run."
        )
        return None

    # ── Active joints ─────────────────────────────────────────────────────────
    active_joints = kin.sapien_robot.get_active_joints()
    joint_names = [j.name for j in active_joints]
    n_dof = len(joint_names)

    print(f"\n── ACTIVE JOINTS  (total DOF = {n_dof}) ──────────────────────")
    for i, name in enumerate(joint_names):
        marker = (
            " [HEAD]" if name in HEAD_JOINTS
            else " [L_ARM]" if name in LEFT_ARM_JOINTS
            else " [R_ARM]" if name in RIGHT_ARM_JOINTS
            else ""
        )
        print(f"  [{i:3d}]  {name}{marker}")

    # ── Mask verification ─────────────────────────────────────────────────────
    head_qmask = build_mask(joint_names, HEAD_JOINTS)
    left_arm_qmask = build_mask(joint_names, LEFT_ARM_JOINTS)
    right_arm_qmask = build_mask(joint_names, RIGHT_ARM_JOINTS)

    print("\n── ACTIVE QMASK SUMMARY ─────────────────────────────────────")
    print(f"  head      ({head_qmask.sum()} joints): "
          f"{[joint_names[i] for i in np.where(head_qmask)[0]]}")
    print(f"  left_arm  ({left_arm_qmask.sum()} joints): "
          f"{[joint_names[i] for i in np.where(left_arm_qmask)[0]]}")
    print(f"  right_arm ({right_arm_qmask.sum()} joints): "
          f"{[joint_names[i] for i in np.where(right_arm_qmask)[0]]}")

    ok = left_arm_qmask.sum() == 7 and right_arm_qmask.sum() == 7
    print(f"\n{'✓' if ok else '✗'}  Arm DOF check: expected 7 per arm")
    print("=" * 64)

    return {
        "head_eef_idx": head_eef_idx,
        "left_arm_eef_idx": left_arm_eef_idx,
        "right_arm_eef_idx": right_arm_eef_idx,
        "head_qmask": head_qmask,
        "left_arm_qmask": left_arm_qmask,
        "right_arm_qmask": right_arm_qmask,
        "head_indices": [
            i for i, n in enumerate(joint_names)
            if n in {"head_j1", "head_j2", "head_j3"}
        ],
        "left_arm_indices": [
            i for i, n in enumerate(joint_names) if n in LEFT_ARM_JOINTS
        ],
        "right_arm_indices": [
            i for i, n in enumerate(joint_names) if n in RIGHT_ARM_JOINTS
        ],
        "joint_names": joint_names,
        "n_dof": n_dof,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EEF / target visualisation helpers  — proper coordinate frames
# ─────────────────────────────────────────────────────────────────────────────

_AXIS_LEN    = 0.15    # full axis length in metres
_AXIS_RADIUS = 0.007   # capsule radius

# Pre-computed rotations: capsule local axis is X, so we need to rotate
# the Y-axis actor by +90° around Z and the Z-axis actor by -90° around Y.
_ROT_Z_P90 = Rotation.from_euler("z",  90, degrees=True)  # X → Y
_ROT_Y_N90 = Rotation.from_euler("y", -90, degrees=True)  # X → Z


def _make_capsule(scene: sapien.Scene, color_rgb, name):
    builder = scene.create_actor_builder()
    builder.add_capsule_visual(
        sapien.Pose(), radius=_AXIS_RADIUS, half_length=_AXIS_LEN / 2, material=color_rgb
    )
    entity = builder.build_kinematic(name=name)
    return entity


def _make_origin_sphere(scene: sapien.Scene, color_rgb, name):
    builder = scene.create_actor_builder()
    builder.add_sphere_visual(radius=_AXIS_RADIUS * 2.0, material=color_rgb)
    entity = builder.build_kinematic(name=name)
    return entity


def make_frame_markers(scene, origin_color, name_prefix):
    """Return a 4-tuple (origin_sphere, X_capsule, Y_capsule, Z_capsule).

    Axes use canonical RGB colouring (X=red, Y=green, Z=blue).
    The origin sphere uses *origin_color* to distinguish different frames.
    """
    return (
        _make_origin_sphere(scene, origin_color, name=f"{name_prefix}_origin"),  # origin
        _make_capsule(scene, [1.0, 0.15, 0.15], name=f"{name_prefix}_x"),   # X — red
        _make_capsule(scene, [0.15, 1.0, 0.15], name=f"{name_prefix}_y"),   # Y — green
        _make_capsule(scene, [0.15, 0.15, 1.0], name=f"{name_prefix}_z"),   # Z — blue
    )


def update_frame_markers(markers, mat4):
    p = mat4[:3, 3]
    frame_rot = Rotation.from_matrix(mat4[:3, :3])
    half = _AXIS_LEN / 2

    def _pose(center, rot):
        q = rot.as_quat()  # (x, y, z, w) → Sapien wants (w, x, y, z)
        return sapien.Pose(p=center, q=[q[3], q[0], q[1], q[2]])

    R = mat4[:3, :3]
    markers[0].set_pose(sapien.Pose(p=p))                                            # origin
    markers[1].set_pose(_pose(p + R[:, 0] * half, frame_rot))                       # X axis
    markers[2].set_pose(_pose(p + R[:, 1] * half, frame_rot * _ROT_Z_P90))          # Y axis
    markers[3].set_pose(_pose(p + R[:, 2] * half, frame_rot * _ROT_Y_N90))          # Z axis


# ─────────────────────────────────────────────────────────────────────────────
# Live VR simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_live_sim(
    kin: KinHelper,
    cfg: dict,
    arm_scale: float,
) -> None:
    """Connect to Quest 3, run staged calibration, and drive the Sapien viewer.

    Replicates the exact same IK pipeline as ``vr_reader.py`` so what you see
    here is what the robot will do — no surprises on hardware.
    """
    # ── Sapien scene ──────────────────────────────────────────────────────────
    scene = sapien.Scene()
    scene.add_ground(-0.1)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=-2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True
    robot = loader.load(URDF_PATH)

    for link in robot.get_links():
        for shape in link.get_collision_shapes():
            shape.set_collision_groups([1, 1, 17, 0])

    # ── Joint name map (needed to write arm solution back to Sapien qpos) ──────
    active_robot_joints = robot.get_active_joints()
    sapien_joint_names = [j.name for j in active_robot_joints]
    joint_name_to_idx  = {name: i for i, name in enumerate(sapien_joint_names)}

    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
    init_qpos_dict = {
        "L_arm_j1": np.pi / 2.0,
        "L_arm_j4": -np.pi / 2.0,
        "R_arm_j1": -np.pi / 2.0,
        "R_arm_j4": -np.pi / 2.0,
    }
    init_qpos = np.zeros(robot.dof)
    for name, val in init_qpos_dict.items():
        init_qpos[joint_name_to_idx[name]] = val
    robot.set_qpos(init_qpos)

    for ji, joint in enumerate(active_robot_joints):
        joint.set_drive_property(stiffness=4000, damping=500, force_limit=1000, mode="force")
        joint.set_drive_target(init_qpos[ji])

    # ── MotionManager + ArmProcessors (mirrors command_processor.py) ──────────
    console.print("Initialising MotionManager (pink IK + self-collision avoidance)…")
    mm = MotionManager(init_visualizer=False,joint_regions_to_lock=["BASE"])
    left_init = init_qpos[[joint_name_to_idx[n] for n in LEFT_ARM_JOINTS]]
    right_init = init_qpos[[joint_name_to_idx[n] for n in RIGHT_ARM_JOINTS]]
    torso_init = init_qpos[[joint_name_to_idx[n] for n in TORSO_JOINTS]]
    mm.left_arm.set_joint_pos(left_init.tolist())
    mm.right_arm.set_joint_pos(right_init.tolist())
    mm.torso.set_joint_pos(torso_init.tolist())
    config      = get_config()
    robot_info  = RobotInfo()
    left_proc   = ArmProcessor("left",  config, mm, robot_info, "vr")
    right_proc  = ArmProcessor("right", config, mm, robot_info, "vr")
    console.print("[bold green]MotionManager ready.[/]")

    # ── EEF / target pose visualisation markers ───────────────────────────────
    l_target_mrk = make_frame_markers(scene, [0.2, 0.5, 1.0], "l_tgt")    # blue   — left  IK target
    r_target_mrk = make_frame_markers(scene, [1.0, 0.5, 0.2], "r_tgt")    # orange — right IK target
    l_eef_mrk    = make_frame_markers(scene, [0.5, 0.85, 1.0], "l_curr") # cyan   — left  EEF current
    r_eef_mrk    = make_frame_markers(scene, [1.0, 0.85, 0.3], "r_curr") # yellow — right EEF current

    # ── Connect to Quest 3 ────────────────────────────────────────────────────
    console.print("(Make sure the Unity OpenXR Teleop app is running on the headset)")
    quest = WebXRVRReader(port=5067, ssl_certfile="/home/yixuan/omniteleop/tests/cert.pem", ssl_keyfile="/home/yixuan/omniteleop/tests/key.pem")
    quest.start()
    console.print("Waiting for VR data stream ...")
    quest.wait_for_data()
    console.print("[bold green]Quest stream active![/]\n")

    # ── Unpack IK config ──────────────────────────────────────────────────────
    head_eef_idx = cfg["head_eef_idx"]
    left_arm_eef_idx = cfg["left_arm_eef_idx"]
    right_arm_eef_idx = cfg["right_arm_eef_idx"]
    head_qmask = cfg["head_qmask"]

    current_qpos = init_qpos.copy()
    calib_stage = "0"  # 0 -> A -> B -> C

    # Interpolated wrist targets — initialised from FK, updated each frame
    curr_left_target  = kin.compute_fk_from_link_idx(current_qpos, [left_arm_eef_idx])[0]
    curr_right_target = kin.compute_fk_from_link_idx(current_qpos, [right_arm_eef_idx])[0]
    _INTERP_ALPHA = 0.1  # fraction to move toward new target each frame [0=frozen, 1=no interp]

    # ── Calibration prompts ───────────────────────────────────────────────────
    console.rule("[bold cyan]VR Sim Calibration")
    console.print(
        "[bold]Stage A:[/] Robot head tracks the VR headset orientation.\n"
        "         When satisfied, [yellow]squeeze the RIGHT INDEX TRIGGER fully[/] to confirm."
    )

    _confirm_start: float | None = None

    def _is_confirm(transforms: dict) -> bool:
        nonlocal _confirm_start
        val = transforms.get("right_index_trigger", 0.0)
        if val > 0.7:
            if _confirm_start is None:
                _confirm_start = time.perf_counter()
            elif time.perf_counter() - _confirm_start >= 1.0:
                _confirm_start = None  # reset so next stage requires a fresh hold
                return True
        else:
            _confirm_start = None  # trigger released — reset
        return False

    init_tf = quest.get_latest_transformation()
    vr_base_t_init_head = init_tf["head"]
    robot_base_t_init_head = \
        kin.compute_fk_from_link_idx(current_qpos, [head_eef_idx])[0]
    robot_base_t_vr_base = robot_base_t_init_head @ np.linalg.inv(vr_base_t_init_head) 

    # ── Main render/IK loop ───────────────────────────────────────────────────
    while not viewer.closed:
        transforms = quest.get_latest_transformation()
        print(f"right index trigger: {transforms["right_index_trigger"]}", end="\r")
        if transforms is None:
            scene.update_render()
            viewer.render()
            continue

        vr_base_t_curr_head = transforms["head"]
        vr_base_t_left_wrist = transforms["left_wrist"]
        vr_base_t_right_wrist = transforms["right_wrist"]
        

        # ── Head IK (all stages) ──────────────────────────────────
        if calib_stage != "0":
            robot_base_t_curr_head = robot_base_t_vr_base @ vr_base_t_curr_head
            # robot_t_curr_head[1,3] = 0.0
            # robot_t_curr_head[0,3] = init_robot_head[0,3]
            head_qpos = kin.compute_ik_from_mat(
                current_qpos, robot_base_t_curr_head,
                eef_idx=head_eef_idx, active_qmask=head_qmask,
                damp=1000.0,
            )
            mm.torso.set_joint_pos(head_qpos[[joint_name_to_idx[n] for n in TORSO_JOINTS]])
            for i in np.where(head_qmask)[0]:
                current_qpos[i] = head_qpos[i]

        # ── Arm IK (stage B_approach + B live) ───────────────────
        invalid_right_pose = np.array([[ 0.,  0.,  1.,  0.],
                                   [ 1.,  0.,  0.,  0.],
                                   [0.,  1.,  0.,  0.],
                                   [ 0.,  0.,  0.,  1.]])
        invalid_left_pose = np.array([[ 0.,  0.,  1.,  0.],
                                   [-1.,  0.,  0.,  0.],
                                   [ 0., -1.,  0.,  0.],
                                   [ 0.,  0.,  0.,  1.]])
        valid_wrists = (
            not np.allclose(vr_base_t_left_wrist,  invalid_left_pose) and
            not np.allclose(vr_base_t_right_wrist, invalid_right_pose)
        )

        if calib_stage in ("B_approach", "B") and valid_wrists:
            robot_base_t_left_wrist  = robot_base_t_vr_base @ vr_base_t_left_wrist
            robot_base_t_right_wrist = robot_base_t_vr_base @ vr_base_t_right_wrist

            if calib_stage == "B_approach":
                # Interpolate curr_*_target toward VR target
                for curr_target, new_target in [
                    (curr_left_target,  robot_base_t_left_wrist),
                    (curr_right_target, robot_base_t_right_wrist),
                ]:
                    curr_target[:3, 3] += _INTERP_ALPHA * (new_target[:3, 3] - curr_target[:3, 3])
                    r_curr = Rotation.from_matrix(curr_target[:3, :3])
                    r_new  = Rotation.from_matrix(new_target[:3, :3])
                    curr_target[:3, :3] = Slerp([0, 1], Rotation.concatenate([r_curr, r_new]))([_INTERP_ALPHA]).as_matrix()[0]

                
                dist_left  = np.linalg.norm(curr_left_target[:3, 3]  - robot_base_t_left_wrist[:3, 3])
                dist_right = np.linalg.norm(curr_right_target[:3, 3] - robot_base_t_right_wrist[:3, 3])
                print(f"Approaching target — L: {dist_left:.3f} m   R: {dist_right:.3f} m")

                if dist_left < 0.02 and dist_right < 0.02:
                    calib_stage = "B"
                    console.print("[bold green]Reached target — live tracking active.[/]")

                ik_left_target  = curr_left_target
                ik_right_target = curr_right_target
            else:
                # Stage B: direct tracking, no interpolation
                ik_left_target  = robot_base_t_left_wrist
                ik_right_target = robot_base_t_right_wrist

            # ── Arm IK with self-collision avoidance (MotionManager / pink) ──
            with suppress_loguru_module("dexmotion", enabled=True):
                arm_solution, in_collision, within_limits = mm.ik(
                    target_pose={LEFT_ARM_EEF_LINK: ik_left_target, RIGHT_ARM_EEF_LINK: ik_right_target},
                    type="pink",
                )

            if not arm_solution or in_collision or not within_limits:
                if in_collision:
                    console.print("[bold red]Self-collision — holding arm position[/]")
                elif not within_limits:
                    console.print("[yellow]Arm IK out of limits — holding arm position[/]")
            else:
                left_positions  = [arm_solution[f"L_arm_j{i}"] for i in range(1, 8)]
                right_positions = [arm_solution[f"R_arm_j{i}"] for i in range(1, 8)]
                safe_left  = left_proc.limit_joint_step(left_positions)
                safe_right = right_proc.limit_joint_step(right_positions)
                left_proc.apply_positions(safe_left)
                right_proc.apply_positions(safe_right)
                for i, name in enumerate(left_proc.joint_names):
                    if name in joint_name_to_idx:
                        current_qpos[joint_name_to_idx[name]] = safe_left[i]
                for i, name in enumerate(right_proc.joint_names):
                    if name in joint_name_to_idx:
                        current_qpos[joint_name_to_idx[name]] = safe_right[i]

            # ── EEF visualisation ─────────────────────────────────────────────
            curr_l_fk = kin.compute_fk_from_link_idx(current_qpos, [left_arm_eef_idx])[0]
            curr_r_fk = kin.compute_fk_from_link_idx(current_qpos, [right_arm_eef_idx])[0]
            update_frame_markers(l_target_mrk, ik_left_target)
            update_frame_markers(r_target_mrk, ik_right_target)
            update_frame_markers(l_eef_mrk,    curr_l_fk)
            update_frame_markers(r_eef_mrk,    curr_r_fk)
            dist_l = np.linalg.norm(curr_l_fk[:3, 3] - ik_left_target[:3, 3])
            dist_r = np.linalg.norm(curr_r_fk[:3, 3] - ik_right_target[:3, 3])
            print(f"EEF→target  L: {dist_l:.3f} m   R: {dist_r:.3f} m")

        # ── Stage transitions ─────────────────────────────────────────────
        if calib_stage == "0" and _is_confirm(transforms):
            calib_stage = "A"
            console.print(
                "\n[bold]Stage A:[/] Robot head now tracks the VR headset orientation.\n"
                "         When satisfied, [yellow]squeeze the RIGHT INDEX TRIGGER fully[/] to confirm and advance."
            )
        elif calib_stage == "A" and _is_confirm(transforms):
            calib_stage = "B_approach"
            curr_left_target  = kin.compute_fk_from_link_idx(current_qpos, [left_arm_eef_idx])[0].copy()
            curr_right_target = kin.compute_fk_from_link_idx(current_qpos, [right_arm_eef_idx])[0].copy()
            console.print(
                "\n[bold]Stage B (approach):[/] Moving arms toward VR controller positions…"
            )

        # ── Apply to Sapien display robot ─────────────────────────────────
        robot.set_qpos(current_qpos)
        for ji, joint in enumerate(active_robot_joints):
            joint.set_drive_target(current_qpos[ji])

        for _ in range(4):
            qf = robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            robot.set_qf(qf)
            scene.step()

        scene.update_render()
        viewer.render()

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--arm-scale",
        type=float,
        default=1.0,
        help="Head-relative wrist translation scale (default: 1.0)",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print link/joint tables and exit without connecting to the VR headset",
    )
    args = parser.parse_args()

    print(f"Loading KinHelper('{ROBOT_NAME}') ...")
    kin = KinHelper(ROBOT_NAME)

    cfg = inspect_and_resolve(kin)
    if cfg is None:
        return  # Link name mismatch — user must fix constants and re-run

    if args.inspect_only:
        print("\n--inspect-only: skipping VR connection and Sapien viewer.")
        return

    run_live_sim(kin, cfg, arm_scale=args.arm_scale)


if __name__ == "__main__":
    main()
