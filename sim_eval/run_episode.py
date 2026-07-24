"""Run one scripted-expert episode of a task and render the head camera to a video.
Usage: behavior python -u run_episode.py --task carry --seed 0 --out /tmp/carry.mp4"""
import os
os.environ["OMNIGIBSON_HEADLESS"] = "1"
import argparse
import numpy as np
import omnigibson as og

from vega_og_env import VegaOGEnv
from tasks import TASKS

P = lambda *a: print(*a, flush=True)


def head_rgb(env):
    obs, _ = env.robot.get_obs()
    for key, val in obs.items():
        if isinstance(val, dict) and "rgb" in val:
            a = val["rgb"]
            a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
            return a[..., :3].astype(np.uint8)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="carry", choices=list(TASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-ticks", type=int, default=800)
    ap.add_argument("--port", type=int, default=5620)
    ap.add_argument("--frame-every", type=int, default=5)
    ap.add_argument("--debug-every", type=int, default=0,
                    help="log measured vs WBC-reference base pose + head pose every N ticks")
    ap.add_argument("--head-cost", type=float, default=None,
                    help="override head_world_position_cost on all 3 axes (default: env's own)")
    args = ap.parse_args()
    out = args.out or f"/tmp/claude-1000/-home-yixuan-BEHAVIOR-1K/9901aed5-d602-4345-b36f-c4b1542046a9/scratchpad/{args.task}_seed{args.seed}.mp4"

    task = TASKS[args.task](rng=np.random.default_rng(args.seed))
    mobile = getattr(task, "MOBILE", False)
    ovr = None
    if args.head_cost is not None:
        ovr = {"head_world_position_cost": [args.head_cost] * 3,
               "enable_torso_top_x_anchor": False}
    env = VegaOGEnv(task=task, lock_base=not mobile, mobile=mobile, wbc_port=args.port, pos_kp=4000,
                    obs_hw=(480, 640), grasping_mode=task.GRASPING_MODE, wbc_overrides=ovr,
                    robot_pos=task.ROBOT_POS, robot_yaw=task.ROBOT_YAW)
    if mobile:
        env.base_x_max = getattr(task, "BASE_X_MAX", None)   # forward-park clamp near the table
    env.reset(seed=args.seed)
    # Park the head camera at zed_depth_frame + add the wrist camera, exactly as the data
    # collector does, so the head video the user reviews IS the policy's recorded view.
    from obs_pipeline import SimObsRecorder
    SimObsRecorder(env).setup()
    src, dst = task.objects_of_interest(env)                    # [source, target] world XYZ (generic)
    P(f"[run] task={args.task} seed={args.seed} grasp={task.GRASPING_MODE}; "
      f"src@{np.round(src,3)} dst@{np.round(dst,3)}")

    # aim the 3rd-person review camera at the work area (midpoint of the objects), from front-left
    # and above -- looks down at the table + arms rather than off at a wall (the old default did).
    work = (src + dst) / 2.0
    env._set_cam_lookat(work + np.array([-0.55, 1.15, 0.95]), work)
    for _ in range(3):
        og.sim.render()

    frames, thirds = [], []
    last_phase = None
    resp = {}
    for i in range(args.max_ticks):
        cmd = task.expert_step(env, {})
        if cmd.base_vel is not None:
            env.drive_base(cmd.base_vel)                        # navigate
            ee_err = None
        else:
            resp = env.wbc_tick(cmd.left_target, cmd.right_target, head_og=cmd.head_target,
                                 grip_l=cmd.gripper_left, grip_r=cmd.gripper_right)   # manipulate
            ee_err = resp.get("l_err")
        if args.debug_every and i % args.debug_every == 0:
            ref = np.asarray(resp.get("base_pose", [0, 0, 0]), dtype=float)
            meas = np.array(env.base_xyyaw())
            hl = getattr(task, "_head_link", None)
            he = ""
            if hl is not None and cmd.head_target is not None:
                Th = env.link_pose(hl)
                q = env.robot.get_joint_positions().detach().cpu().numpy()
                pitch = lambda T: np.degrees(np.arctan2(-T[2, 2], np.hypot(T[0, 2], T[1, 2])))
                he = (f" head@{np.round(Th[:3,3],3)} tgt@{np.round(cmd.head_target[:3,3],3)}"
                      f" pitch={pitch(Th):+.1f}/{pitch(cmd.head_target):+.1f}"
                      f" hj3={np.degrees(q[env.name2idx['head_j3']]):+.1f}"
                      f" torso={np.round(np.degrees([q[env.name2idx[n]] for n in ('torso_j1','torso_j2','torso_j3')]),1)}")
            P(f"    [dbg {i:3d}] base_meas={np.round(meas,3)} wbc_ref={np.round(ref,3)}{he}")
        if cmd.phase != last_phase:
            src = task.objects_of_interest(env)[0]
            ag = env.is_grasping("left")
            extra = ""
            if hasattr(env, "finger_qpos"):
                gp = env.finger_grasp_point("left")
                extra = f" fq={env.finger_qpos('left')} d(gp,src)={round(float(np.linalg.norm(gp - src)),3)}"
            bx, by, bwz = env.base_xyyaw()
            byaw = round(float(np.degrees(bwz)), 1)
            bpos = env.link_pose('base')[:3, 3]
            P(f"  tick {i:3d} phase={cmd.phase:9s} base=({bpos[0]:.2f},{bpos[1]:.2f}) yaw={byaw:+.1f} "
              f"src@{np.round(src,3)} ee_err={ee_err if ee_err is None else round(float(ee_err),4)} "
              f"grasped={ag.name if ag is not None else None}{extra}")
            last_phase = cmd.phase
        if i % args.frame_every == 0:
            f = head_rgb(env)                                   # head-cam view (the policy's view)
            if f is not None:
                frames.append(f)
            try:
                thirds.append(env.capture_third_person())       # 3rd-person review view
            except Exception:
                pass
        if cmd.done:
            break

    succ = task.success(env)
    src, dst = task.objects_of_interest(env)
    P(f"[run] SUCCESS={succ}  final src@{np.round(src,3)} dst@{np.round(dst,3)}  frames={len(frames)}")
    import imageio
    for tag, buf in [("", frames), ("_3rd", thirds)]:
        if not buf:
            continue
        path = out.rsplit(".", 1)[0] + tag + ".mp4"
        try:
            imageio.mimwrite(path, buf, fps=20, macro_block_size=None)
            P(f"[run] wrote {path}")
        except Exception as e:
            gif = path.rsplit(".", 1)[0] + ".gif"
            imageio.mimsave(gif, buf, fps=20)
            P(f"[run] mp4 failed ({e}); wrote {gif}")
    env.close()


if __name__ == "__main__":
    main()
