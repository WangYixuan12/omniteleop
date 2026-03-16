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

# ── Robot and EEF link name configuration ─────────────────────────────────────
# Update these if the link names differ in your URDF.  Run with --inspect-only
# to print all available link names.
ROBOT_NAME = "vega_no_effector"
HEAD_LINK = "zed_depth_frame"
LEFT_ARM_EEF_LINK = "L_ee"
RIGHT_ARM_EEF_LINK = "R_ee"

DEFAULT_SERVER_URL = "http://localhost:5066"

# Joint name sets (should not need changing for standard Vega builds)
HEAD_JOINTS = frozenset(
    {"torso_j1", "torso_j2", "torso_j3", "head_j2", "head_j3"}
)
LEFT_ARM_JOINTS = frozenset(f"L_arm_j{i}" for i in range(1, 8))
RIGHT_ARM_JOINTS = frozenset(f"R_arm_j{i}" for i in range(1, 8))

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

    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
    init_qpos = np.zeros(robot.dof)
    robot.set_qpos(init_qpos)

    active_robot_joints = robot.get_active_joints()
    for ji, joint in enumerate(active_robot_joints):
        joint.set_drive_property(stiffness=4000, damping=500, force_limit=1000, mode="force")
        joint.set_drive_target(init_qpos[ji])

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
    left_arm_qmask = cfg["left_arm_qmask"]
    right_arm_qmask = cfg["right_arm_qmask"]

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
            for i in np.where(head_qmask)[0]:
                current_qpos[i] = head_qpos[i]

        # ── Arm IK (stage B_approach + B live) ───────────────────
        invalid_pose = np.array([[ 0.,  0.,  1.,  0.],
                                   [ 0.,  1.,  0.,  0.],
                                   [-1.,  0.,  0.,  0.],
                                   [ 0.,  0.,  0.,  1.]])
        valid_wrists = (
            not np.allclose(vr_base_t_left_wrist,  invalid_pose) and
            not np.allclose(vr_base_t_right_wrist, invalid_pose)
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
                print(f"robot_base_t_left_wrist:\n{robot_base_t_left_wrist}")
                print(f"robot_base_t_right_wrist:\n{robot_base_t_right_wrist}")

            left_qpos = kin.compute_ik_from_mat(
                head_qpos, ik_left_target,
                eef_idx=left_arm_eef_idx, active_qmask=left_arm_qmask,
                damp=1000.0,
            )
            right_qpos = kin.compute_ik_from_mat(
                head_qpos, ik_right_target,
                eef_idx=right_arm_eef_idx, active_qmask=right_arm_qmask,
                damp=1000.0,
            )
            for i in np.where(left_arm_qmask)[0]:
                current_qpos[i] = left_qpos[i]
            for i in np.where(right_arm_qmask)[0]:
                current_qpos[i] = right_qpos[i]

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
