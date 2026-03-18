#!/usr/bin/env python3
"""Sapien live-VR IK simulation — Quest 3 → Sapien viewer.

Connect the Quest 3 WebXR streamer (vr_client.html), then run this script.
Hold the right index trigger for 1 s to advance each calibration stage:
  A          — head tracks VR headset
  B_approach — arms interpolate toward controller positions
  B          — arms track controllers live (self-collision avoidance active)
"""

from __future__ import annotations

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

ROBOT_NAME         = "vega_no_effector"
HEAD_LINK          = "zed_depth_frame"
LEFT_ARM_EEF_LINK  = "L_ee"
RIGHT_ARM_EEF_LINK = "R_ee"

HEAD_JOINTS      = ["torso_j1", "torso_j2", "torso_j3", "head_j2", "head_j3"]
LEFT_ARM_JOINTS  = [f"L_arm_j{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]
TORSO_JOINTS     = ["torso_j1", "torso_j2", "torso_j3"]

URDF_PATH = "/home/yixuan/yixuan_utilities/src/yixuan_utilities/assets/robot/vega-urdf/vega_no_effector.urdf"

console = Console()

# Sentinel poses sent when a controller is not tracked
INVALID_LEFT_POSE  = np.array([[ 0.,  0.,  1.,  0.], [-1.,  0.,  0.,  0.], [ 0., -1.,  0.,  0.], [ 0.,  0.,  0.,  1.]])
INVALID_RIGHT_POSE = np.array([[ 0.,  0.,  1.,  0.], [ 1.,  0.,  0.,  0.], [ 0.,  1.,  0.,  0.], [ 0.,  0.,  0.,  1.]])


# ── Robot inspection ──────────────────────────────────────────────────────────

def resolve_ik_config(kin: KinHelper) -> dict | None:
    lni = kin.link_name_to_idx
    head_eef_idx      = lni.get(HEAD_LINK)
    left_arm_eef_idx  = lni.get(LEFT_ARM_EEF_LINK)
    right_arm_eef_idx = lni.get(RIGHT_ARM_EEF_LINK)

    missing = [n for n, i in [(HEAD_LINK, head_eef_idx), (LEFT_ARM_EEF_LINK, left_arm_eef_idx), (RIGHT_ARM_EEF_LINK, right_arm_eef_idx)] if i is None]
    if missing:
        console.print(f"[red]Links not found: {missing}. Update *_LINK constants and re-run.[/]")
        return None

    joint_names = [j.name for j in kin.sapien_robot.get_active_joints()]
    head_qmask  = np.array([n in HEAD_JOINTS for n in joint_names], dtype=bool)
    console.print(f"EEF idx — head:{head_eef_idx}  L_arm:{left_arm_eef_idx}  R_arm:{right_arm_eef_idx}  head_DOF:{head_qmask.sum()}")
    return {
        "head_eef_idx":      head_eef_idx,
        "left_arm_eef_idx":  left_arm_eef_idx,
        "right_arm_eef_idx": right_arm_eef_idx,
        "head_qmask":        head_qmask,
    }


# ── Coordinate-frame visualisation markers ────────────────────────────────────

_AXIS_LEN    = 0.15
_AXIS_RADIUS = 0.007
_ROT_Z_P90   = Rotation.from_euler("z",  90, degrees=True)  # capsule X → world Y
_ROT_Y_N90   = Rotation.from_euler("y", -90, degrees=True)  # capsule X → world Z


def _kinematic(scene, build_fn):
    b = scene.create_actor_builder()
    build_fn(b)
    return b.build_kinematic()


def make_frame_markers(scene, origin_color):
    """Return (origin_sphere, X_capsule, Y_capsule, Z_capsule) as kinematic actors."""
    def sphere(b):
        b.add_sphere_visual(radius=_AXIS_RADIUS * 2, material=origin_color)

    def axis(color):
        def build(b):
            b.add_capsule_visual(sapien.Pose(), radius=_AXIS_RADIUS, half_length=_AXIS_LEN / 2, material=color)
        return build

    return (
        _kinematic(scene, sphere),
        _kinematic(scene, axis([1.0, 0.15, 0.15])),   # X — red
        _kinematic(scene, axis([0.15, 1.0, 0.15])),   # Y — green
        _kinematic(scene, axis([0.15, 0.15, 1.0])),   # Z — blue
    )


def update_frame_markers(markers, mat4):
    p, R = mat4[:3, 3], mat4[:3, :3]
    rot  = Rotation.from_matrix(R)
    half = _AXIS_LEN / 2

    def pose(center, r):
        q = r.as_quat()
        return sapien.Pose(p=center, q=[q[3], q[0], q[1], q[2]])

    markers[0].set_pose(sapien.Pose(p=p))
    markers[1].set_pose(pose(p + R[:, 0] * half, rot))
    markers[2].set_pose(pose(p + R[:, 1] * half, rot * _ROT_Z_P90))
    markers[3].set_pose(pose(p + R[:, 2] * half, rot * _ROT_Y_N90))


# ── Live simulation ───────────────────────────────────────────────────────────

def run_live_sim(kin: KinHelper, cfg: dict) -> None:
    # Scene
    scene = sapien.Scene()
    scene.add_ground(-0.1)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=-2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    # Robot
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True
    robot = loader.load(URDF_PATH)
    for link in robot.get_links():
        for shape in link.get_collision_shapes():
            shape.set_collision_groups([1, 1, 17, 0])

    active_joints     = robot.get_active_joints()
    joint_name_to_idx = {j.name: i for i, j in enumerate(active_joints)}

    init_qpos = np.zeros(robot.dof)
    for name, val in {"L_arm_j1": np.pi/2, "L_arm_j4": -np.pi/2, "R_arm_j1": -np.pi/2, "R_arm_j4": -np.pi/2}.items():
        init_qpos[joint_name_to_idx[name]] = val
    robot.set_qpos(init_qpos)
    for ji, joint in enumerate(active_joints):
        joint.set_drive_property(stiffness=4000, damping=500, force_limit=1000, mode="force")
        joint.set_drive_target(init_qpos[ji])

    # MotionManager + ArmProcessors
    console.print("Initialising MotionManager…")
    mm = MotionManager(init_visualizer=False, joint_regions_to_lock=["BASE"])
    mm.left_arm.set_joint_pos(init_qpos[[joint_name_to_idx[n] for n in LEFT_ARM_JOINTS]].tolist())
    mm.right_arm.set_joint_pos(init_qpos[[joint_name_to_idx[n] for n in RIGHT_ARM_JOINTS]].tolist())
    mm.torso.set_joint_pos(init_qpos[[joint_name_to_idx[n] for n in TORSO_JOINTS]].tolist())
    config     = get_config()
    robot_info = RobotInfo()
    left_proc  = ArmProcessor("left",  config, mm, robot_info, "vr")
    right_proc = ArmProcessor("right", config, mm, robot_info, "vr")
    console.print("[bold green]MotionManager ready.[/]")

    # Visualisation markers: target (solid colour) + current EEF (lighter)
    l_target_mrk = make_frame_markers(scene, [0.2, 0.5, 1.0])   # blue
    r_target_mrk = make_frame_markers(scene, [1.0, 0.5, 0.2])   # orange
    l_eef_mrk    = make_frame_markers(scene, [0.5, 0.85, 1.0])  # cyan
    r_eef_mrk    = make_frame_markers(scene, [1.0, 0.85, 0.3])  # yellow

    # Quest 3
    quest = WebXRVRReader(port=5067, ssl_certfile="/home/yixuan/omniteleop/tests/cert.pem", ssl_keyfile="/home/yixuan/omniteleop/tests/key.pem")
    quest.start()
    console.print("Waiting for VR data…")
    quest.wait_for_data()
    console.print("[bold green]Quest connected.[/]")

    head_eef_idx      = cfg["head_eef_idx"]
    left_arm_eef_idx  = cfg["left_arm_eef_idx"]
    right_arm_eef_idx = cfg["right_arm_eef_idx"]
    head_qmask        = cfg["head_qmask"]

    current_qpos      = init_qpos.copy()
    calib_stage       = "0"
    curr_left_target  = kin.compute_fk_from_link_idx(current_qpos, [left_arm_eef_idx])[0].copy()
    curr_right_target = kin.compute_fk_from_link_idx(current_qpos, [right_arm_eef_idx])[0].copy()
    INTERP_ALPHA      = 0.1

    init_tf              = quest.get_latest_transformation()
    robot_base_t_vr_base = (
        kin.compute_fk_from_link_idx(current_qpos, [head_eef_idx])[0]
        @ np.linalg.inv(init_tf["head"])
    )

    _confirm_start: float | None = None
    def _is_confirm(transforms):
        nonlocal _confirm_start
        val = transforms.get("right_index_trigger", 0.0)
        if val > 0.7:
            if _confirm_start is None:
                _confirm_start = time.perf_counter()
            elif time.perf_counter() - _confirm_start >= 1.0:
                _confirm_start = None
                return True
        else:
            _confirm_start = None
        return False

    console.rule("[bold cyan]Stage 0 — hold right trigger to start head tracking")

    while not viewer.closed:
        transforms = quest.get_latest_transformation()
        if transforms is None:
            scene.update_render(); viewer.render()
            continue

        vr_base_t_curr_head   = transforms["head"]
        vr_base_t_left_wrist  = transforms["left_wrist"]
        vr_base_t_right_wrist = transforms["right_wrist"]

        # Head IK
        if calib_stage != "0":
            head_qpos = kin.compute_ik_from_mat(
                current_qpos, robot_base_t_vr_base @ vr_base_t_curr_head,
                eef_idx=head_eef_idx, active_qmask=head_qmask, damp=1000.0,
            )
            mm.torso.set_joint_pos(head_qpos[[joint_name_to_idx[n] for n in TORSO_JOINTS]])
            for i in np.where(head_qmask)[0]:
                current_qpos[i] = head_qpos[i]

        # Arm IK
        valid_wrists = (
            not np.allclose(vr_base_t_left_wrist,  INVALID_LEFT_POSE) and
            not np.allclose(vr_base_t_right_wrist, INVALID_RIGHT_POSE)
        )
        ik_left_target  = None
        ik_right_target = None

        if calib_stage in ("B_approach", "B") and valid_wrists:
            robot_base_t_left_wrist  = robot_base_t_vr_base @ vr_base_t_left_wrist
            robot_base_t_right_wrist = robot_base_t_vr_base @ vr_base_t_right_wrist

            if calib_stage == "B_approach":
                for curr, tgt in [(curr_left_target, robot_base_t_left_wrist), (curr_right_target, robot_base_t_right_wrist)]:
                    curr[:3, 3] += INTERP_ALPHA * (tgt[:3, 3] - curr[:3, 3])
                    curr[:3, :3] = Slerp([0, 1], Rotation.concatenate([
                        Rotation.from_matrix(curr[:3, :3]), Rotation.from_matrix(tgt[:3, :3])
                    ]))(INTERP_ALPHA).as_matrix()
                dist_l = np.linalg.norm(curr_left_target[:3, 3]  - robot_base_t_left_wrist[:3, 3])
                dist_r = np.linalg.norm(curr_right_target[:3, 3] - robot_base_t_right_wrist[:3, 3])
                print(f"Approaching  L:{dist_l:.3f}  R:{dist_r:.3f} m", end="\r")
                if dist_l < 0.02 and dist_r < 0.02:
                    calib_stage = "B"
                    console.print("\n[bold green]Live tracking active.[/]")
                ik_left_target  = curr_left_target
                ik_right_target = curr_right_target
            else:
                ik_left_target  = robot_base_t_left_wrist
                ik_right_target = robot_base_t_right_wrist

            with suppress_loguru_module("dexmotion", enabled=True):
                arm_solution, in_collision, within_limits = mm.ik(
                    target_pose={LEFT_ARM_EEF_LINK: ik_left_target, RIGHT_ARM_EEF_LINK: ik_right_target},
                    type="pink",
                )

            if not arm_solution or in_collision or not within_limits:
                if in_collision:      console.print("[red]Self-collision — holding[/]")
                if not within_limits: console.print("[yellow]Out of limits — holding[/]")
            else:
                safe_left  = left_proc.limit_joint_step([arm_solution[f"L_arm_j{i}"] for i in range(1, 8)])
                safe_right = right_proc.limit_joint_step([arm_solution[f"R_arm_j{i}"] for i in range(1, 8)])
                left_proc.apply_positions(safe_left)
                right_proc.apply_positions(safe_right)
                for i, n in enumerate(left_proc.joint_names):
                    current_qpos[joint_name_to_idx[n]] = safe_left[i]
                for i, n in enumerate(right_proc.joint_names):
                    current_qpos[joint_name_to_idx[n]] = safe_right[i]

        # Visualisation
        curr_l_fk = kin.compute_fk_from_link_idx(current_qpos, [left_arm_eef_idx])[0]
        curr_r_fk = kin.compute_fk_from_link_idx(current_qpos, [right_arm_eef_idx])[0]
        update_frame_markers(l_eef_mrk, curr_l_fk)
        update_frame_markers(r_eef_mrk, curr_r_fk)
        if ik_left_target is not None:
            update_frame_markers(l_target_mrk, ik_left_target)
            update_frame_markers(r_target_mrk, ik_right_target)
            dist_l = np.linalg.norm(curr_l_fk[:3, 3] - ik_left_target[:3, 3])
            dist_r = np.linalg.norm(curr_r_fk[:3, 3] - ik_right_target[:3, 3])
            print(f"EEF→target  L:{dist_l:.3f}  R:{dist_r:.3f} m", end="\r")

        # Stage transitions
        if calib_stage == "0" and _is_confirm(transforms):
            calib_stage = "A"
            console.rule("[bold cyan]Stage A — head tracking. Hold trigger again → arms")
        elif calib_stage == "A" and _is_confirm(transforms):
            calib_stage = "B_approach"
            curr_left_target  = kin.compute_fk_from_link_idx(current_qpos, [left_arm_eef_idx])[0].copy()
            curr_right_target = kin.compute_fk_from_link_idx(current_qpos, [right_arm_eef_idx])[0].copy()
            console.rule("[bold cyan]Stage B — approaching arm targets…")

        # Sapien step
        robot.set_qpos(current_qpos)
        for ji, joint in enumerate(active_joints):
            joint.set_drive_target(current_qpos[ji])
        for _ in range(4):
            robot.set_qf(robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
            scene.step()
        scene.update_render()
        viewer.render()


def main() -> None:
    kin = KinHelper(ROBOT_NAME)
    cfg = resolve_ik_config(kin)
    if cfg is None:
        return
    run_live_sim(kin, cfg)


if __name__ == "__main__":
    main()
