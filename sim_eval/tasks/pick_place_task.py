"""PickPlaceTask: tabletop single-arm pick-and-place (apple -> bowl), base fixed.

Mirrors the BEHAVIOR-DemoGen apple2bowl tabletop task, but drives the Vega with the
whole-body IK (WBC) service and a *physical* grasp instead of R1 + CuRobo. The left arm
picks an `apple` off the existing Rs_int dining table and drops it into a `bowl`; the
right arm holds at nominal and the base does not move (tabletop). SceneDiff objects of
interest are [apple, bowl]; success = apple Inside/OnTop the bowl.

Grasp is *physical*: the actuated Robotiq 2F-85 fingers physically close on the apple and
hold it by contact/friction -- no magnetization, no teleport. The expert first drives the
open finger gap onto the apple, then closes, so the pads pinch it for real.
"""
from __future__ import annotations
import numpy as np

from .base_task import SimTask, ExpertCommand

APPLE = {"category": "apple", "model": "agveuv"}
BOWL = {"category": "bowl", "model": "ajzltc"}


def T_from(pos, rot3x3):
    T = np.eye(4); T[:3, :3] = rot3x3; T[:3, 3] = np.asarray(pos)
    return T


class PickPlaceTask(SimTask):
    name = "pickplace"
    # Physical grasp: the actuated Robotiq 2F-85 fingers close on the apple and hold it by contact
    # + friction (no magnetization/teleport). 0=open .. 0.785=closed (upper-limit direction).
    GRASPING_MODE = "sticky"
    # tabletop: spawn the base already at the table (no navigation), facing +x.
    ROBOT_POS = (0.58, 0.40, 0.03)
    ROBOT_YAW = 0.0
    APPLE_XY = (1.20, 0.74)   # object (x,y); side by side (same x, offset y). Subclasses override.
    BOWL_XY = (1.20, 0.56)
    # ticks for the pre-contact segments. Mobile subclasses lengthen PREGRASP_TICKS so the
    # whole-body IK has time to drive the (unlocked) base up to the table before the grasp, and
    # SETTLE_TICKS so the base/arm converge on the apple before the fingers close (tighter pinch).
    PREGRASP_TICKS = 45
    APPROACH_TICKS = 40
    SETTLE_TICKS = 30

    def object_configs(self):
        # co-located on the near edge of the existing Rs_int dining table (bound in bind()).
        # bowl is fixed_base: it's a stationary drop target, so it can't be knocked off the table.
        return [
            # apple + bowl SIDE BY SIDE (same distance from the robot, offset laterally in y) so they
            # read as "next to" each other in the head view for SceneDiff. Apple scaled to ~0.06 m so
            # it clears the gripper opening.
            {"type": "DatasetObject", "name": "apple", **APPLE, "position": [*self.APPLE_XY, 0.90],
             "scale": [0.8, 0.8, 0.8]},
            {"type": "DatasetObject", "name": "bowl", **BOWL, "position": [*self.BOWL_XY, 0.90],
             "fixed_base": True},
        ]

    def bind(self, env):
        reg = env.env.scene.object_registry
        objs = env.env.scene.objects
        tables = [o for o in objs if o.category == "breakfast_table"]
        self.table = min(tables, key=lambda o: np.linalg.norm(env.obj_pos(o)[:2] - np.array([1.47, 0.41])))
        chairs = [o for o in objs if o.category == "straight_chair"]
        self.block_chair = min(chairs, key=lambda o: np.linalg.norm(env.obj_pos(o)[:2] - np.array([0.97, 0.38])))
        self.apple = reg("name", "apple")
        self.bowl = reg("name", "bowl")

    def reset(self, env):
        # clear the chair that sits between the robot and the table's near edge
        cp = env.obj_pos(self.block_chair)
        self.block_chair.set_position_orientation(position=[cp[0], cp[1] - 1.7, cp[2]], orientation=[0, 0, 0, 1])
        self.ztop = float(self.table.aabb[1][2])   # existing table already rests correctly (~0.76)
        r = self.rng
        ax = self.APPLE_XY[0] + r.uniform(-0.02, 0.02); ay = self.APPLE_XY[1] + r.uniform(-0.02, 0.02)
        bx = self.BOWL_XY[0] + r.uniform(-0.02, 0.02); by = self.BOWL_XY[1] + r.uniform(-0.02, 0.02)
        az = self.ztop + float(self.apple.aabb_extent[2]) / 2 + 0.01
        bz = self.ztop + float(self.bowl.aabb_extent[2]) / 2 + 0.005
        self.apple.set_position_orientation(position=[ax, ay, az], orientation=[0, 0, 0, 1])
        self.bowl.set_position_orientation(position=[bx, by, bz], orientation=[0, 0, 0, 1])

    def objects_of_interest(self, env):
        return np.stack([env.obj_pos(self.apple), env.obj_pos(self.bowl)])  # (2,3)

    def success(self, env):
        from omnigibson.object_states import Inside, OnTop
        try:
            if bool(self.apple.states[Inside].get_value(self.bowl)):
                return True
            return bool(self.apple.states[OnTop].get_value(self.bowl))
        except Exception:
            ap, bp = env.obj_pos(self.apple), env.obj_pos(self.bowl)
            return np.linalg.norm(ap[:2] - bp[:2]) < 0.08 and abs(ap[2] - bp[2]) < 0.12

    # ---- scripted expert (single left arm, base fixed) ----
    def expert_reset(self, env):
        # ONE consistent control law for the whole episode: a single world-space waypoint list for
        # the fingertip grasp CENTER, mapped to an L_ee target by the SAME `_grasp_target` at every
        # tick. Only the interpolated center + the gripper command change across the episode -- there
        # is no per-phase target math (no wrist servo, no post-grasp offset recalibration).
        self._t = 0
        self._L0 = env.link_pose("L_ee")
        # right arm holds its nominal pose RELATIVE TO THE BASE, so it rides a moving (mobile) base
        # instead of being left behind at a fixed world pose.
        self._Ree_base = np.linalg.inv(env.link_pose("base")) @ env.link_pose("R_ee")
        self._apple0 = env.obj_pos(self.apple).copy()
        self._bowl0 = env.obj_pos(self.bowl).copy()
        self._bowl_top = float(self.bowl.aabb[1][2])
        self._apple_half = float(self.apple.aabb_extent[2]) / 2
        # Grasp geometry computed ONCE: the fingertip grasp center as an offset in the L_ee frame,
        # plus the fixed grasp orientation (the arm's natural rest orientation -- its approach axis
        # points forward-and-slightly-down, reachable at the table; a top-down wrist blows up the IK).
        Tee = self._L0
        fm = env.finger_grasp_point("left")
        self._grasp_off_local = Tee[:3, :3].T @ (fm - Tee[:3, 3])
        self._Rgrasp = Tee[:3, :3].copy()
        # Waypoints for the grasp center: (name, end_center_world, gripper, n_ticks). Each tick lerps
        # the center from the previous waypoint's end to the current one; the gripper closes at the
        # apple and re-opens over the bowl. The center->L_ee mapping is identical throughout.
        go, gc = env.grip_open, env.grip_close
        a, b = self._apple0, self._bowl0
        drop_z = self._bowl_top + self._apple_half + 0.01
        self._start_center = fm
        self._center = fm.copy()   # the tick's commanded grasp center (subclasses read it)
        self._segs = [
            ("pregrasp", a + np.array([0.0, 0.0, 0.08]),        go, self.PREGRASP_TICKS),
            ("approach", a.copy(),                              go, self.APPROACH_TICKS),
            ("settle",   a.copy(),                              go, self.SETTLE_TICKS),
            ("close",    a.copy(),                              gc, 45),
            ("lift",     a + np.array([0.0, 0.0, 0.12]),        gc, 35),
            ("carry",    np.array([b[0], b[1], drop_z + 0.05]), gc, 50),
            ("place",    np.array([b[0], b[1], drop_z]),        gc, 45),
            ("release",  np.array([b[0], b[1], drop_z]),        go, 25),
            ("done",     np.array([b[0], b[1], drop_z + 0.12]), go, 15),
        ]
        # Head target: the camera is HEAD_FRAME == zed_depth_frame, so its target is built in the
        # optical convention (local z = optical axis). Position defaults to the reset head position
        # (mobile overrides it); orientation looks along the spawn HEADING, pitched down so the
        # commanded work point sits on the optical axis. At nominal posture the camera looks only
        # 19 deg down and the objects fall just BELOW the frame (wbik.yaml calls head_j1 the
        # "camera-view knob" and it is pinned at -0.40 for a table that is farther away than ours);
        # head_j3 (tilt) is the one free head IK dof, so the extra pitch resolves there. Only the
        # PITCH is taken from the work point -- a full look-at would also demand a base yaw (11 deg
        # at spawn) and turn the robot off its straight-in approach.
        self._head_link = next((n for n in ("zed_depth_frame", "eyes", "head_l3")
                                if n in env.robot.links), None)
        self._head_pos0 = env.link_pose(self._head_link)[:3, 3].copy()
        self._head_heading = float(self.ROBOT_YAW)
        print(f"[pickplace] apple0={np.round(self._apple0,3)} bowl0={np.round(self._bowl0,3)} "
              f"bowl_top={self._bowl_top:.3f} L_ee0={np.round(Tee[:3,3],3)} "
              f"finger_ctr={np.round(fm,3)} head0={np.round(self._head_pos0,3)}", flush=True)

    @staticmethod
    def _finger_center(env, link_name):
        """World-frame center of a finger link's AABB (proxy for the finger mesh middle)."""
        amin, amax = env.robot.links[link_name].aabb
        amin = amin.detach().cpu().numpy() if hasattr(amin, "detach") else np.asarray(amin)
        amax = amax.detach().cpu().numpy() if hasattr(amax, "detach") else np.asarray(amax)
        return (amin + amax) / 2

    def _head_target_pos(self, env):
        """Commanded head world position. Base is fixed here, so it stays at the reset pose."""
        return self._head_pos0

    @staticmethod
    def _gaze_R(head_pos, work_pos, heading):
        """Optical-frame R (columns: x right, y DOWN, z forward) looking along `heading` and
        pitched down so `work_pos` lands on the optical axis. Yaw stays at `heading`."""
        d = np.asarray(work_pos, dtype=float) - np.asarray(head_pos, dtype=float)
        dh = float(np.hypot(d[0], d[1]))
        f = np.array([np.cos(heading) * dh, np.sin(heading) * dh, d[2]])
        f /= max(np.linalg.norm(f), 1e-9)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(f, up)
        right /= max(np.linalg.norm(right), 1e-9)
        u = np.cross(right, f)
        return np.stack([right, -u, f], axis=1)

    def _head_target(self, env):
        pos = self._head_target_pos(env)
        return T_from(pos, self._gaze_R(pos, self._center, self._head_heading))

    def _grasp_target(self, grasp_center_world, Rmat, offset_local=None):
        """L_ee pose so that its fingertip grasp center lands at grasp_center_world (orientation Rmat)."""
        offset = self._grasp_off_local if offset_local is None else offset_local
        Lee_pos = np.asarray(grasp_center_world, dtype=float) - Rmat @ offset
        return T_from(Lee_pos, Rmat)

    def expert_step(self, env, obs) -> ExpertCommand:
        # Consistent control every tick: walk the waypoint list, lerp the grasp center, map it to an
        # L_ee target with the SAME `_grasp_target`. The right arm holds its nominal pose relative to
        # the (possibly moving) base. Only the center + gripper change -- no phase math, no navigate.
        Rt = env.link_pose("base") @ self._Ree_base
        t = self._t
        self._t += 1
        acc = 0
        prev = self._start_center
        for name, end_c, grip, n in self._segs:
            if t < acc + n:
                f = (t - acc) / n
                self._center = center = (1.0 - f) * prev + f * end_c
                L = self._grasp_target(center, self._Rgrasp)
                return ExpertCommand(left_target=L, right_target=Rt, head_target=self._head_target(env),
                                     gripper_left=grip, gripper_right=env.grip_open,
                                     done=False, phase=name)
            acc += n
            prev = end_c
        self._center = np.asarray(self._segs[-1][1], dtype=float)
        L = self._grasp_target(self._segs[-1][1], self._Rgrasp)
        return ExpertCommand(left_target=L, right_target=Rt, head_target=self._head_target(env),
                             gripper_left=self._segs[-1][2], gripper_right=env.grip_open,
                             done=True, phase="done")
