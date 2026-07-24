"""Diagnostic: measure the Vega left gripper geometry (open vs closed) so the pick-place
grasp can target the fingertip grasp center, not the finger-base midpoint. Prints link origins,
AABB centers/extents, and confirms which command closes the fingers."""
import os
os.environ["OMNIGIBSON_HEADLESS"] = "1"
import numpy as np

from vega_og_env import VegaOGEnv
from tasks import PickPlaceTask

P = lambda *a: print(*a, flush=True)


def linfo(env, name):
    T = env.link_pose(name); p = T[:3, 3]
    lk = env.robot.links[name]
    ac = ext = None
    for attr in ("aabb",):
        try:
            aabb = getattr(lk, attr)
            amin = aabb[0].detach().cpu().numpy() if hasattr(aabb[0], "detach") else np.asarray(aabb[0])
            amax = aabb[1].detach().cpu().numpy() if hasattr(aabb[1], "detach") else np.asarray(aabb[1])
            ac = (amin + amax) / 2; ext = amax - amin
        except Exception as e:
            ext = f"<{type(e).__name__}>"
    P(f"  {name:16s} origin={np.round(p,3)} aabb_ctr={None if ac is None else np.round(ac,3)} "
      f"ext={ext if ac is None else np.round(ext,3)}")


def main():
    task = PickPlaceTask(rng=np.random.default_rng(0))
    env = VegaOGEnv(task=task, lock_base=True, wbc_port=5640, pos_kp=4000, obs_hw=(240, 320),
                    grasping_mode="physical", grasping_direction="upper",
                    robot_pos=task.ROBOT_POS, robot_yaw=task.ROBOT_YAW)
    env.reset(seed=0)

    def gjoints():
        q = env.robot.get_joint_positions().detach().cpu().numpy()
        return tuple(q[env.name2idx[name]] for name in env.robot.finger_joint_names["left"])

    finger_joints = env.robot.finger_joint_names["left"] + env.robot.finger_joint_names["right"]
    P("finger DOF idx:", {name: env.name2idx.get(name) for name in finger_joints})
    P("=== OPEN (rest) ===  L fingers =", np.round(gjoints(), 4), " grip_close=", env.grip_close)
    for n in ["L_ee", *env.robot.finger_link_names["left"]]:
        linfo(env, n)

    # definitively determine the close command: drive raw -1 then +1 and see which -> 0 (closed)
    for cmd in (-1.0, 1.0):
        for _ in range(80):
            a = env._hold_action()
            a[env.cai["gripper_left"]] = cmd
            env.env.step(a)
        P(f"=== after raw cmd={cmd:+.0f} ===  L fingers =", np.round(gjoints(), 4))
        for n in env.robot.finger_link_names["left"]:
            linfo(env, n)
    env.close()


if __name__ == "__main__":
    main()
