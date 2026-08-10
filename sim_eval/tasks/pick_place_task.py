"""PickPlaceTask: tabletop single-arm pick-and-place (apple -> bowl), base fixed.

Mirrors the BEHAVIOR-DemoGen apple2bowl tabletop task, but drives the Vega with the
whole-body IK (WBC) service instead of R1 + CuRobo. The left arm
picks an `apple` off the existing Rs_int dining table and drops it into a `bowl`; the
right arm holds at nominal and the base does not move (tabletop). SceneDiff objects of
interest are [apple, bowl]; success = apple Inside/OnTop the bowl.

Grasp is OmniGibson **sticky** (`GRASPING_MODE = "sticky"`), i.e. contact-triggered
magnetization: the object is attached once a finger touches it WHILE CLOSING, and
`vega_og_env` sets `GRASP_WINDOW = 0` so that happens on the first contact. It is not
friction holding the object. What makes it legitimate rather than a teleport is that the
expert drives the open finger gap onto the object and closes there, so the attachment only
ever fires with the pads already in contact -- nothing is grabbed across a gap and nothing
is moved into the hand. The non-prehensile tasks use "physical" instead, because a closed
fist under sticky would magnetize the door or chair it leans on.
"""
from __future__ import annotations
import numpy as np

from .base_task import SimTask, ExpertCommand

APPLE = {"category": "apple", "model": "agveuv"}
BOWL = {"category": "bowl", "model": "ajzltc"}


def T_from(pos, rot3x3):
    T = np.eye(4); T[:3, :3] = rot3x3; T[:3, 3] = np.asarray(pos)
    return T


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class PickPlaceTask(SimTask):
    name = "pickplace"
    # Sticky = contact-triggered magnetization, fired only with the pads already closing on the
    # object (see the module docstring). 0=open .. 0.785=closed (upper-limit direction).
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
        # Both action targets must be commands, never functions of measured state. Preserve the
        # nominal right EEF in the base frame, then map it through the commanded planar-body frame
        # derived from the head target on every tick (the same head-yaw reference used by the real
        # leader). Feeding env.link_pose("base") back into only the right action made the recorded
        # L/R streams semantically asymmetric and made the left target appear to stutter in a
        # base-following review camera.
        Tbase0 = env.link_pose("base")
        Ree = env.link_pose("R_ee")
        self._Ree_base = np.linalg.inv(Tbase0) @ Ree
        # Grasp geometry computed ONCE: the fingertip grasp center as an offset in the L_ee frame,
        # plus the fixed grasp orientation (the arm's natural rest orientation -- its approach axis
        # points forward-and-slightly-down, reachable at the table; a top-down wrist blows up the IK).
        Tee = self._L0
        fm = env.finger_grasp_point("left")
        self._grasp_off_local = Tee[:3, :3].T @ (fm - Tee[:3, 3])
        self._Rgrasp = Tee[:3, :3].copy()
        # ...and the SAME orientation expressed in the BASE frame. `_Rgrasp` is a world
        # rotation captured once, which is correct only while the body never turns. Long-horizon
        # tasks turn ~100 deg, and a world-fixed wrist command then means "keep the hand pointing
        # the way it pointed at spawn no matter which way you are facing" -- the hand visibly does
        # not follow the body, and the arm contorts to hold it. See `_grasp_rotation_now`.
        self._Rgrasp_base = rot_z(-float(self.ROBOT_YAW)) @ self._Rgrasp
        self._start_center = fm
        self._center = fm.copy()   # the tick's commanded grasp center (subclasses read it)
        # Ordinary tasks leave the right hand at its nominal body-frame pose. A bimanual task may
        # provide an aligned `_right_segs` waypoint list; capture its grasp geometry once here.
        fm_right = env.finger_grasp_point("right")
        self._right_grasp_off_local = Ree[:3, :3].T @ (fm_right - Ree[:3, 3])
        self._Rgrasp_right_base = rot_z(-float(self.ROBOT_YAW)) @ Ree[:3, :3]
        self._right_start_center = fm_right
        self._right_center = fm_right.copy()
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
        self._head_pos_base = (
            np.linalg.inv(Tbase0) @ env.link_pose(self._head_link)
        )[:3, 3].copy()
        pair0 = (self._start_center + self._right_start_center) / 2.0
        self._head_off_bimanual_base = rot_z(-self._head_heading) @ (self._head_pos0 - pair0)
        self._base_cmd_z = float(Tbase0[2, 3])
        # built LAST, so a subclass may lay its waypoints out using the head geometry above
        # (long_horizon plans its route to where the head command will actually park the base)
        # ``collect_demos`` reuses one task object.  A bimanual subclass populates this while
        # building its plan; clearing it first prevents a previous episode's right-arm plan from
        # leaking into a later reset (ordinary tasks intentionally leave it as ``None``).
        self._right_segs = None
        self._right_grasp_rotations = None
        self._right_segment_starts = None
        self._segs = self._build_segments(env)
        print(f"[{self.name}] L_ee0={np.round(Tee[:3,3],3)} finger_ctr={np.round(fm,3)} "
              f"head0={np.round(self._head_pos0,3)} segs={len(self._segs)} "
              f"ticks={sum(n for _,_,_,n in self._segs)}", flush=True)

    def _build_segments(self, env):
        """Waypoints for the grasp center: (name, end_center_world, gripper, n_ticks).

        Each tick lerps the center from the previous waypoint's end to this one; the gripper
        closes at the apple and re-opens over the bowl. The center->L_ee mapping is identical
        throughout, so subclasses change WHERE the hand goes by returning a different list --
        never by adding per-phase target math.
        """
        self._apple0 = env.obj_pos(self.apple).copy()
        self._bowl0 = env.obj_pos(self.bowl).copy()
        self._bowl_top = float(self.bowl.aabb[1][2])
        self._apple_half = float(self.apple.aabb_extent[2]) / 2
        go, gc = env.grip_open, env.grip_close
        a, b = self._apple0, self._bowl0
        drop_z = self._bowl_top + self._apple_half + 0.01
        print(f"[pickplace] apple0={np.round(a,3)} bowl0={np.round(b,3)} "
              f"bowl_top={self._bowl_top:.3f}", flush=True)
        return [
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

    @staticmethod
    def _finger_center(env, link_name):
        """World-frame center of a finger link's AABB (proxy for the finger mesh middle)."""
        amin, amax = env.robot.links[link_name].aabb
        amin = amin.detach().cpu().numpy() if hasattr(amin, "detach") else np.asarray(amin)
        amax = amax.detach().cpu().numpy() if hasattr(amax, "detach") else np.asarray(amax)
        return (amin + amax) / 2

    def _head_target_pos(self, env, heading):
        """Commanded head world position. Base is fixed here, so it stays at the reset pose."""
        return self._head_pos0

    def _head_heading_now(self, env):
        """Commanded gaze/base heading. Fixed here -- see the `_head_target` note on why a full
        look-at is wrong for a straight-in tabletop approach. Long-horizon tasks override it,
        because head yaw IS base yaw (head_j1/j2 are pinned) so this is the only way to turn."""
        return self._head_heading

    def _grasp_rotation_now(self, env):
        """Commanded wrist orientation this tick. Frozen in WORLD here, which is right for the
        fixed-base tabletop task; long-horizon overrides it to ride the commanded heading."""
        return self._Rgrasp

    def _smooth_center(self, env, raw):
        """Keep the waypoint target piecewise linear and its speed constant within each segment.

        Segment duration already sets the commanded speed. A first-order filter here made every
        segment accelerate and decay instead of advancing by equal increments.
        """
        return np.asarray(raw, dtype=float)

    def _smooth_right_center(self, env, raw):
        """Right-hand counterpart of :meth:`_smooth_center` for bimanual subclasses."""
        return np.asarray(raw, dtype=float)

    def _clock_step(self, env):
        """How far the plan clock advances this tick. 1.0 means one tick of wall-clock time."""
        return 1.0

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
        heading = self._head_heading_now(env)
        pos = self._head_target_pos(env, heading)
        return T_from(pos, self._gaze_R(pos, self._head_work_point(), heading))

    def _head_work_point(self):
        """Work point used by the head command.

        The default is the left grasp centre.  A bimanual plan overrides this semantic by making
        the midpoint of both commanded hands the work point, so the base does not park under one
        arm while asking the other to reach across the whole body.
        """
        return self._center

    def _commanded_base_target(self, head_target, heading):
        """Planar body command implied by the commanded head position and yaw.

        The WBC receives measured q separately through ``current_q``. Keeping measurement out of
        action construction matches the real follower, where both EEF targets are interpolated
        commands expressed through one head-yaw frame.
        """
        Rz = rot_z(heading)
        T = np.eye(4)
        T[:3, :3] = Rz
        T[:2, 3] = np.asarray(head_target)[:2, 3] - (Rz @ self._head_pos_base)[:2]
        T[2, 3] = self._base_cmd_z
        return T

    def _right_target(self, head_target, heading):
        """Nominal right EEF mapped through the smooth commanded planar-body frame."""
        return self._commanded_base_target(head_target, heading) @ self._Ree_base

    def _grasp_target(self, grasp_center_world, Rmat, offset_local=None):
        """L_ee pose so that its fingertip grasp center lands at grasp_center_world (orientation Rmat)."""
        offset = self._grasp_off_local if offset_local is None else offset_local
        Lee_pos = np.asarray(grasp_center_world, dtype=float) - Rmat @ offset
        return T_from(Lee_pos, Rmat)

    @staticmethod
    def _segment_grip(segs, i, f, default):
        """Interpolate a gripper command on the same clock as its Cartesian segment."""
        current = default if segs[i] is None else float(segs[i][2])
        previous = default if i == 0 or segs[i - 1] is None else float(segs[i - 1][2])
        return (1.0 - f) * previous + f * current

    @staticmethod
    def _interp_rotation(R0, R1, f):
        """Shortest-path SO(3) interpolation without linearly blending matrix entries."""
        from scipy.spatial.transform import Rotation as R

        R0 = np.asarray(R0, dtype=float)
        relative = R.from_matrix(R0.T @ np.asarray(R1, dtype=float)).as_rotvec()
        return R0 @ R.from_rotvec(float(f) * relative).as_matrix()

    def _segment_rotation(self, rotations, i, f):
        if rotations is None:
            return np.eye(3)
        previous = np.eye(3) if i == 0 else rotations[i - 1]
        return self._interp_rotation(previous, rotations[i], f)

    def expert_step(self, env, obs) -> ExpertCommand:
        # Consistent control every tick: walk the waypoint list, lerp the grasp center, map it to an
        # L_ee target with the SAME `_grasp_target`. The right arm rides the commanded planar-body
        # frame, not the measured base. Only the center + gripper change -- no measured-state action.
        t = self._t
        self._t += self._clock_step(env)
        acc = 0
        prev = self._start_center
        for i, (name, end_c, grip, n) in enumerate(self._segs):
            if t < acc + n:
                f = (t - acc) / n
                # where we are in the waypoint list, so subclasses can interpolate per-waypoint
                # quantities (the long-horizon gaze heading) on exactly the same schedule
                self._seg_i, self._seg_f = i, f
                self._center = center = self._smooth_center(env, (1.0 - f) * prev + f * end_c)
                right_segs = getattr(self, "_right_segs", None)
                right_active = right_segs is not None and right_segs[i] is not None
                if right_active:
                    rname, rend, _right_grip, rn = right_segs[i]
                    assert rname == name and rn == n
                    if i == 0 or right_segs[i - 1] is None:
                        starts = getattr(self, "_right_segment_starts", None)
                        rprev = self._right_start_center if starts is None else starts[i]
                    else:
                        rprev = right_segs[i - 1][1]
                    right_raw = (1.0 - f) * np.asarray(rprev) + f * np.asarray(rend)
                    self._right_center = self._smooth_right_center(env, right_raw)
                H = self._head_target(env)
                L = self._grasp_target(center, self._grasp_rotation_now(env))
                if not right_active:
                    Rt = self._right_target(H, self._head_heading)
                    right_grip = env.grip_open
                else:
                    right_grip = self._segment_grip(right_segs, i, f, env.grip_open)
                    extra = self._segment_rotation(
                        getattr(self, "_right_grasp_rotations", None), i, f
                    )
                    Rright = rot_z(self._head_heading_now(env)) @ extra @ self._Rgrasp_right_base
                    Rt = self._grasp_target(
                        self._right_center, Rright, self._right_grasp_off_local
                    )
                return ExpertCommand(left_target=L, right_target=Rt, head_target=H,
                                     gripper_left=self._segment_grip(
                                         self._segs, i, f, env.grip_open
                                     ), gripper_right=right_grip,
                                     done=False, phase=name)
            acc += n
            prev = end_c
        self._seg_i, self._seg_f = len(self._segs) - 1, 1.0
        self._center = self._smooth_center(env, self._segs[-1][1])
        right_segs = getattr(self, "_right_segs", None)
        if right_segs is not None and right_segs[-1] is not None:
            self._right_center = np.asarray(right_segs[-1][1], dtype=float)
        H = self._head_target(env)
        L = self._grasp_target(self._center, self._grasp_rotation_now(env))
        if right_segs is None or right_segs[-1] is None:
            Rt = self._right_target(H, self._head_heading)
            right_grip = env.grip_open
        else:
            extra = (np.eye(3) if getattr(self, "_right_grasp_rotations", None) is None
                     else self._right_grasp_rotations[-1])
            Rright = rot_z(self._head_heading_now(env)) @ extra @ self._Rgrasp_right_base
            Rt = self._grasp_target(self._right_center, Rright, self._right_grasp_off_local)
            right_grip = right_segs[-1][2]
        return ExpertCommand(left_target=L, right_target=Rt, head_target=H,
                             gripper_left=self._segs[-1][2], gripper_right=right_grip,
                             done=True, phase="done")
