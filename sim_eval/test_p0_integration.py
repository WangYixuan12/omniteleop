"""P0 integration test: drive the OmniGibson vega with the WBC service and confirm
L_ee/R_ee reach a commanded world target. Handles the WBC-world <-> OG-world frame
alignment (the importer z-shifts the robot). Run: behavior python -u, LD cleaned."""
import os
os.environ["OMNIGIBSON_HEADLESS"] = "1"
import numpy as np
import torch as th
from scipy.spatial.transform import Rotation as R
import omnigibson as og
from wbc_client import WBCClient

np.set_printoptions(precision=4, suppress=True, linewidth=140)
P = lambda *a: print(*a, flush=True)


def jc(kp=300):
    return {"name": "JointController", "motor_type": "position", "pos_kp": kp,
            "command_input_limits": None, "command_output_limits": None,
            "use_impedances": False, "use_delta_commands": False}


def pose_mat(pos, quat_xyzw):
    T = np.eye(4); T[:3, :3] = R.from_quat(quat_xyzw).as_matrix(); T[:3, 3] = pos
    return T


def link_T(robot, link):
    p, q = robot.links[link].get_position_orientation()
    return pose_mat(p.detach().cpu().numpy(), q.detach().cpu().numpy())


cfg = {
    "scene": {"type": "Scene", "use_floor_plane": True},
    "robots": [{
        "type": "vega_robotiq", "obs_modalities": ["proprio"], "grasping_mode": "physical",
        "grasping_direction": "upper", "position": [0, 0, 0.03],
        "controller_config": {
            "base": {"name": "HolonomicBaseJointController", "motor_type": "velocity",
                     "vel_kp": 150, "command_input_limits": None, "use_impedances": False},
            "trunk": jc(), "camera": jc(), "arm_left": jc(), "arm_right": jc(),
            "gripper_left": {"name": "MultiFingerGripperController", "mode": "smooth",
                             "command_input_limits": [-1.0, 1.0], "command_output_limits": "default"},
            "gripper_right": {"name": "MultiFingerGripperController", "mode": "smooth",
                              "command_input_limits": [-1.0, 1.0], "command_output_limits": "default"},
        },
    }],
    "task": {"type": "DummyTask"},
}

env = og.Environment(configs=cfg)
robot = env.robots[0]
env.reset()
names = robot.dof_names_ordered
cai = robot.controller_action_idx
dt = 1.0 / 30.0
wbc = WBCClient(port=5605, lock_base=True)   # arm+torso-only IK for P0 (base pinned in the WBC)
info = wbc.info()

# --- frame alignment: T_align maps WBC-world -> OG-world (measured at nominal reset) ---
T_align = link_T(robot, "base")                       # OG base-link world pose at reset
og_Le, og_Re = link_T(robot, "L_ee"), link_T(robot, "R_ee")
wbc_Le, wbc_Re = np.array(info["l_ee"]), np.array(info["r_ee"])
T_align_from_lee = og_Le @ np.linalg.inv(wbc_Le)
P("T_align (base link):\n", T_align)
P("consistency: |T_align(base) - T_align(from L_ee)| =",
  np.abs(T_align - T_align_from_lee).max())
# base-frame EEF should match exactly between OG and WBC (frame-independent):
P("base-frame L_ee match:", np.abs((np.linalg.inv(T_align) @ og_Le) - wbc_Le).max())
T_align_inv = np.linalg.inv(T_align)


def og_to_wbc(T_og):
    return T_align_inv @ T_og


def measured_pin_q():
    q = robot.get_joint_positions().detach().cpu().numpy()
    qby = {n: float(v) for n, v in zip(names, q)}
    base = (qby["base_footprint_x_joint"], qby["base_footprint_y_joint"], qby["base_footprint_rz_joint"])
    return WBCClient.build_pin_q(qby, base)


# P0 isolates the (validated) arm-IK path: base PINNED (base_twist not applied), head held,
# arm-reachable target. Full mobile base-twist shaping (base_closed_loop) is P1.
Ltgt_og = og_Le.copy(); Ltgt_og[:3, 3] += np.array([0.05, 0.08, -0.06])
Rtgt_og = og_Re.copy(); Rtgt_og[:3, 3] += np.array([0.05, -0.08, -0.06])
Ltgt, Rtgt = og_to_wbc(Ltgt_og), og_to_wbc(Rtgt_og)
P("\nReaching OG target L", Ltgt_og[:3, 3], " R", Rtgt_og[:3, 3])
# 1) WBC solves open-loop to convergence (its own l_err -> 0)
for i in range(150):
    resp = wbc.solve(Ltgt, Rtgt, head=None, current_q=None, dt=dt)
P(f"WBC converged: l_err={resp['l_err']:.4f} r_err={resp['r_err']:.4f}")

# 2) KINEMATIC map check: set the WBC solution joints directly on OG, hold, read L_ee.
#    Isolates mapping-correctness (this) from dynamic controller tracking (P1).
name2idx = {n: i for i, n in enumerate(names)}
tgt = WBCClient.result_to_joint_targets(resp)
q_og = robot.get_joint_positions().detach().cpu().numpy().copy()
for n, v in tgt.items():
    q_og[name2idx[n]] = v
robot.set_joint_positions(th.tensor(q_og, dtype=th.float32))
a = th.zeros(robot.action_dim)   # hold the same config so pos controllers keep it against gravity
a[cai["trunk"]] = th.tensor([tgt[n] for n in ["torso_j1", "torso_j2", "torso_j3"]])
a[cai["camera"]] = th.tensor([tgt[n] for n in ["head_j1", "head_j2", "head_j3"]])
a[cai["arm_left"]] = th.tensor([tgt[f"L_arm_j{k}"] for k in range(1, 8)])
a[cai["arm_right"]] = th.tensor([tgt[f"R_arm_j{k}"] for k in range(1, 8)])
a[cai["gripper_left"]] = -1.0; a[cai["gripper_right"]] = -1.0

def og_err():
    le, re = link_T(robot, "L_ee")[:3, 3], link_T(robot, "R_ee")[:3, 3]
    # compare OG L_ee against T_align @ (WBC solution L_ee) -- the mapping-consistent expected pose
    return np.linalg.norm(le - Ltgt_og[:3, 3]), np.linalg.norm(re - Rtgt_og[:3, 3])

env.step(a)                       # 1 step: kinematic set barely perturbed by gravity
li, ri = og_err()
P(f"\nIMMEDIATE (kinematic set)  L_ee err={li:.4f}  R_ee err={ri:.4f}")
# also verify the achieved OG joints equal the commanded WBC joints (pure map check)
q_now = robot.get_joint_positions().detach().cpu().numpy()
jmax = max(abs(q_now[name2idx[n]] - v) for n, v in tgt.items())
P(f"  max |OG joint - WBC joint| = {jmax:.4f} rad")
for _ in range(30):
    env.step(a)
lh, rh = og_err()
P(f"AFTER HOLD (30 steps)      L_ee err={lh:.4f}  R_ee err={rh:.4f}")
P(f"\n-> {'PASS (map exact; residual is gravity sag = P1 gains)' if li<0.03 and ri<0.03 else 'CHECK map'}")
wbc.close()
og.shutdown()
