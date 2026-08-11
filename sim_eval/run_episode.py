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


def head_rgb(rec):
    """The PUBLISHED head view (post crop/resize) -- exactly the frame that gets recorded."""
    return rec._head_images()[0]


#: (label, commanded-pose attribute on the ExpertCommand, link whose pose is the ACTUAL, BGR colour)
TARGETS = (("L_ee", "left_target", "L_ee", (255, 220, 40)),      # cyan-ish
           ("R_ee", "right_target", "R_ee", (255, 90, 220)),     # magenta
           ("head", "head_target", None, (60, 230, 255)))        # yellow; link comes from the task


def draw_targets(env, task, frame, cmd, tick):
    """Draw every commanded EEF/head target on the review frame, next to where the robot has
    actually got to.

    A third-person video shows the robot but not what it was ASKED to do, which is exactly the
    thing under review here -- the whole plan is feed-forward world-frame waypoints, so a pose
    that looks wrong is either a bad target or bad tracking and the picture alone cannot say
    which. Each target is drawn as a filled dot with a crosshair, the measured frame as a hollow
    ring, and the two are joined by a line whose length IS the tracking error (printed in cm).
    The faint polyline is the rest of the planned waypoint list, so the intended path is visible
    before the robot gets there.
    """
    import cv2
    frame = np.ascontiguousarray(frame)
    h, w = frame.shape[:2]

    # gather every world point first, project in ONE pass (the intrinsic query is not free)
    segs = getattr(task, "_segs", None) or []
    world = [np.asarray(s[1], dtype=float) for s in segs]
    pairs = []
    eef_y = {}
    for label, attr, link, colour in TARGETS:
        T = getattr(cmd, attr, None)
        link = link or getattr(task, "_head_link", None)
        if T is None or link is None:
            continue
        tgt = np.asarray(T, dtype=float)[:3, 3]
        act = env.link_pose(link)[:3, 3]
        pairs.append((label, colour, len(world), float(np.linalg.norm(tgt - act))))
        if label in ("L_ee", "R_ee"):
            eef_y[label] = (float(tgt[1]), float(act[1]))
        world += [tgt, act]
    uv, ok = env.project_third_person(np.stack(world)) if world else (np.zeros((0, 2)), [])

    def px(i):
        if not ok[i] or not np.all(np.isfinite(uv[i])):
            return None
        u, v = int(round(uv[i][0])), int(round(uv[i][1]))
        return (u, v) if -w < u < 2 * w and -h < v < 2 * h else None

    # the planned waypoints (grasp centres), faint, so the intended path is visible up front
    chain = [px(i) for i in range(len(segs))]
    for a, b in zip(chain, chain[1:]):
        if a and b:
            cv2.line(frame, a, b, (170, 170, 170), 1, cv2.LINE_AA)
    for i, a in enumerate(chain):
        if a:
            cv2.circle(frame, a, 2, (170, 170, 170), -1, cv2.LINE_AA)
            if i == getattr(task, "_seg_i", -1):
                cv2.circle(frame, a, 7, (255, 255, 255), 1, cv2.LINE_AA)

    y = 22
    for label, colour, base, err in pairs:
        pt, pa = px(base), px(base + 1)
        if pt:
            cv2.line(frame, (pt[0] - 9, pt[1]), (pt[0] + 9, pt[1]), colour, 1, cv2.LINE_AA)
            cv2.line(frame, (pt[0], pt[1] - 9), (pt[0], pt[1] + 9), colour, 1, cv2.LINE_AA)
            cv2.circle(frame, pt, 4, colour, -1, cv2.LINE_AA)
        if pa:
            cv2.circle(frame, pa, 6, colour, 2, cv2.LINE_AA)
        if pt and pa:
            cv2.line(frame, pt, pa, colour, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{label} target  err {err * 100:5.1f} cm", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        y += 20
    cv2.putText(frame, f"t={tick}  {cmd.phase}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    y += 20
    if "L_ee" in eef_y and "R_ee" in eef_y:
        lt, lm = eef_y["L_ee"]
        rt, rm = eef_y["R_ee"]
        cv2.putText(
            frame,
            f"world y  target L {lt:+.3f} R {rt:+.3f} d {lt - rt:+.3f}"
            f"  measured L {lm:+.3f} R {rm:+.3f} d {lm - rm:+.3f}",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
        )
    cv2.putText(frame, "+ commanded    o measured", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="carry", choices=list(TASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-ticks", type=int, default=0,
                    help="0 = the task's own budget (long-horizon tasks size it from the route)")
    ap.add_argument("--port", type=int, default=5620)
    ap.add_argument("--frame-every", type=int, default=5)
    ap.add_argument("--debug-every", type=int, default=0,
                    help="log measured vs WBC-reference base pose + head pose every N ticks")
    ap.add_argument("--head-cost", type=float, default=None,
                    help="override head_world_position_cost on all 3 axes (default: env's own)")
    ap.add_argument("--no-overlay", action="store_true",
                    help="do not draw the commanded EEF/head targets on the 3rd-person video")
    import zed_sim
    zed_sim.ZedSimOptions.add_cli(ap)
    args = ap.parse_args()
    zed_opts = zed_sim.ZedSimOptions.from_args(args)
    out = args.out or f"/tmp/claude-1000/-home-yixuan-BEHAVIOR-1K/9901aed5-d602-4345-b36f-c4b1542046a9/scratchpad/{args.task}_seed{args.seed}.mp4"

    task = TASKS[args.task](rng=np.random.default_rng(args.seed))
    mobile = getattr(task, "MOBILE", False)
    ovr = None
    if args.head_cost is not None:
        ovr = {"head_world_position_cost": [args.head_cost] * 3}
    env = VegaOGEnv(task=task, lock_base=not mobile, mobile=mobile, wbc_port=args.port, pos_kp=4000,
                    obs_hw=zed_sim.head_render_hw(zed_opts), grasping_mode=task.GRASPING_MODE,
                    wbc_overrides=ovr, robot_pos=task.ROBOT_POS, robot_yaw=task.ROBOT_YAW)
    if mobile:
        env.base_x_max = getattr(task, "BASE_X_MAX", None)   # forward-park clamp near the table
    env.reset(seed=args.seed)
    max_ticks = args.max_ticks or int(getattr(task, "MAX_TICKS", 800))
    # Park the head camera at zed_depth_frame + add the wrist camera, exactly as the data
    # collector does, so the head video the user reviews IS the policy's recorded view.
    from obs_pipeline import SimObsRecorder
    rec = SimObsRecorder(env, seed=args.seed, zed=zed_opts)
    rec.setup()
    src, dst = task.objects_of_interest(env)                    # [source, target] world XYZ (generic)
    P(f"[run] task={args.task} seed={args.seed} grasp={task.GRASPING_MODE}; "
      f"src@{np.round(src,3)} dst@{np.round(dst,3)}")

    # the review camera CHASES the robot (see follow_third_person): a long-horizon episode covers
    # several metres, so anything anchored to the work area loses the robot within a second or two
    env.follow_third_person()
    for _ in range(3):
        og.sim.render()

    frames, thirds = [], []
    last_phase = None
    resp = {}
    tilt, yaws = [], []           # chassis roll/pitch and heading, for the motion-quality report
    for i in range(max_ticks):
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
            lim = env.at_limits([f"L_arm_j{k}" for k in range(1, 8)] + ["torso_j1", "torso_j2", "torso_j3"])
            P(f"    [dbg {i:3d}] base_meas={np.round(meas,3)} wbc_ref={np.round(ref,3)}{he}"
              + (f" AT-LIMIT={lim}" if lim else ""))
        if cmd.phase != last_phase:
            src = task.objects_of_interest(env)[0]
            ag = env.is_grasping("left")
            extra = ""
            if hasattr(env, "finger_qpos"):
                gp = env.finger_grasp_point("left")
                extra = f" fq={env.finger_qpos('left')} d(gp,src)={round(float(np.linalg.norm(gp - src)),3)}"
            carried = getattr(task, "apple", None)
            if carried is not None and hasattr(carried, "get_linear_velocity"):
                lv = carried.get_linear_velocity()
                av = carried.get_angular_velocity()
                lv = lv.detach().cpu().numpy() if hasattr(lv, "detach") else np.asarray(lv)
                av = av.detach().cpu().numpy() if hasattr(av, "detach") else np.asarray(av)
                extra += f" obj_v={np.linalg.norm(lv):.3f} obj_w={np.linalg.norm(av):.3f}"
            bx, by, bwz = env.base_xyyaw()
            byaw = round(float(np.degrees(bwz)), 1)
            bpos = env.link_pose('base')[:3, 3]
            tz, trx, tryy = env.chassis_tilt()          # 0,0,0 unless the chassis is tipping
            tilt.append(max(abs(trx), abs(tryy)))
            extra += f" tilt=({tz:+.3f},{np.degrees(trx):+.1f},{np.degrees(tryy):+.1f})"
            P(f"  tick {i:3d} phase={cmd.phase:9s} base=({bpos[0]:.2f},{bpos[1]:.2f}) yaw={byaw:+.1f} "
              f"src@{np.round(src,3)} ee_err={ee_err if ee_err is None else round(float(ee_err),4)} "
              f"grasped={ag.name if ag is not None else None}{extra}")
            last_phase = cmd.phase
        if i % args.frame_every == 0:
            yaws.append(env.base_xyyaw()[2])
            tilt.append(max(abs(v) for v in env.chassis_tilt()[1:]))
            f = head_rgb(rec)                                   # head-cam view (the policy's view)
            if f is not None:
                frames.append(f)
            try:
                env.follow_third_person()                       # keep the robot in frame
                # Viewer-camera observation is a render-product buffer. Updating the camera pose
                # alone leaves the next capture showing the previous pose while the overlay is
                # projected through the new one, so render twice just as camera setup does.
                for _ in range(2):
                    og.sim.render()
                third = env.capture_third_person()              # 3rd-person review view
                if not args.no_overlay:
                    third = draw_targets(env, task, third, cmd, i)
                thirds.append(third)
            except Exception as exc:
                if not getattr(main, "_third_warned", False):
                    main._third_warned = True
                    P(f"[run] third-person capture failed: {exc!r}")
        if cmd.done:
            break

    succ = task.success(env)
    quality = getattr(env, "_quality", {})
    quality_reasons = []
    if quality.get("hold_ticks", 0):
        quality_reasons.append(f"WBC holds={quality['hold_ticks']}")
    if quality.get("keepout_interventions", 0):
        quality_reasons.append(f"keepout edits={quality['keepout_interventions']}")
    if quality.get("max_chassis_tilt_deg", 0.0) > 2.0:
        quality_reasons.append(f"tilt={quality['max_chassis_tilt_deg']:.2f}deg")
    if quality.get("max_arm_target_jump_m", 0.0) > 0.03:
        quality_reasons.append(f"arm target jump={quality['max_arm_target_jump_m']:.3f}m")
    if quality.get("max_target_jump_deg", 0.0) > 10.0:
        quality_reasons.append(f"target turn={quality['max_target_jump_deg']:.1f}deg")
    succ = bool(succ and not quality_reasons)
    src, dst = task.objects_of_interest(env)
    # Motion quality, the thing the review video is actually judged on: total yaw variation (the
    # real robot manages 11.7 deg over a 3.5 m path and never commands wz at all) and the worst
    # chassis roll/pitch, which is 0 unless the body is tipping.
    y = np.unwrap(np.asarray(yaws, dtype=float)) if yaws else np.zeros(1)
    P(f"[run] SUCCESS={succ}  final src@{np.round(src,3)} dst@{np.round(dst,3)}  frames={len(frames)}")
    P(f"[run] motion: yaw net {np.degrees(y[-1] - y[0]):+.1f} deg, total variation "
      f"{np.degrees(np.abs(np.diff(y)).sum()):.1f} deg; max chassis tilt "
      f"{np.degrees(max(tilt) if tilt else 0.0):.2f} deg")
    P(f"[run] trajectory quality: {quality if quality else 'not instrumented'}"
      + (f" REJECT={quality_reasons}" if quality_reasons else ""))
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
