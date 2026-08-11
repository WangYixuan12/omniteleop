"""CarryTask: bimanual carry-and-place (box -> tray), base fixed (tabletop).

BOTH arms grasp a storage_box by its two sides, lift it together, carry it to a nearby tray,
and set it down. Mirrors the box2cloth ManiFlow task. SceneDiff objects of interest = [box, tray];
success = OnTop(box, tray).

Same design as PickPlaceTask, extended to two arms and applied CONSISTENTLY every tick: a single
world-space waypoint list for the box CENTER; each arm's grasp center = box_center +/- the side
offset, mapped to an L_ee/R_ee target by the SAME `_grasp_target` at every tick. Only the
interpolated center + the gripper commands change -- no per-phase control math. Grasp is real
(sticky attaches when the pads contact the box; no kinematic hold/teleport).
"""
from __future__ import annotations
import numpy as np

from .base_task import SimTask, ExpertCommand

BOX = {"category": "storage_box", "model": "aqdbdu"}
TRAY = {"category": "tray", "model": "avotsj"}


def T_from(pos, rot3x3):
    T = np.eye(4); T[:3, :3] = rot3x3; T[:3, 3] = np.asarray(pos)
    return T


class CarryTask(SimTask):
    name = "carry"
    GRASPING_MODE = "sticky"
    ROBOT_POS = (0.58, 0.40, 0.03)   # tabletop: base already at the table, facing +x
    ROBOT_YAW = 0.0
    BOX_XY = (1.18, 0.40)            # box centered on the midline so BOTH arms reach a side
    TRAY_XY = (1.05, 0.40)           # drop target, pulled in toward the robot (easy reach)

    def object_configs(self):
        return [
            {"type": "DatasetObject", "name": "box", **BOX, "position": [*self.BOX_XY, 0.90]},
            {"type": "DatasetObject", "name": "tray", **TRAY, "position": [*self.TRAY_XY, 0.90],
             "fixed_base": True},
        ]

    def bind(self, env):
        reg = env.env.scene.object_registry
        objs = env.env.scene.objects
        tables = [o for o in objs if o.category == "breakfast_table"]
        self.table = min(tables, key=lambda o: np.linalg.norm(env.obj_pos(o)[:2] - np.array([1.47, 0.41])))
        chairs = [o for o in objs if o.category == "straight_chair"]
        self.block_chair = min(chairs, key=lambda o: np.linalg.norm(env.obj_pos(o)[:2] - np.array([0.97, 0.38])))
        self.box = reg("name", "box")
        self.tray = reg("name", "tray")

    def reset(self, env):
        cp = env.obj_pos(self.block_chair)
        self.block_chair.set_position_orientation(position=[cp[0], cp[1] - 1.7, cp[2]], orientation=[0, 0, 0, 1])
        self.ztop = float(self.table.aabb[1][2])
        r = self.rng
        bx = self.BOX_XY[0] + r.uniform(-0.02, 0.02); by = self.BOX_XY[1] + r.uniform(-0.02, 0.02)
        tx = self.TRAY_XY[0] + r.uniform(-0.02, 0.02); ty = self.TRAY_XY[1] + r.uniform(-0.02, 0.02)
        bz = self.ztop + float(self.box.aabb_extent[2]) / 2 + 0.01
        tz = self.ztop + float(self.tray.aabb_extent[2]) / 2 + 0.005
        self.box.set_position_orientation(position=[bx, by, bz], orientation=[0, 0, 0, 1])
        self.tray.set_position_orientation(position=[tx, ty, tz], orientation=[0, 0, 0, 1])

    def objects_of_interest(self, env):
        return np.stack([env.obj_pos(self.box), env.obj_pos(self.tray)])  # (2,3)

    def success(self, env):
        from omnigibson.object_states import OnTop
        try:
            return bool(self.box.states[OnTop].get_value(self.tray))
        except Exception:
            bp, tp = env.obj_pos(self.box), env.obj_pos(self.tray)
            return np.linalg.norm(bp[:2] - tp[:2]) < 0.12 and abs(bp[2] - tp[2]) < 0.15

    # ---- scripted expert (both arms, base fixed) ----
    def expert_reset(self, env):
        # ONE consistent control law: a single waypoint list for the box CENTER; each arm's grasp
        # center = box_center +/- side_offset, mapped to an eef target by `_grasp_target` every tick.
        self._t = 0
        self._box0 = env.obj_pos(self.box).copy()
        self._tray0 = env.obj_pos(self.tray).copy()
        self._box_top = float(self.box.aabb[1][2])
        # per-arm grasp geometry, computed ONCE: fingertip grasp center offset (in eef frame) + the
        # fixed grasp orientation (each arm's natural rest orientation).
        self._Rg = {}
        self._off = {}
        for arm, eef in (("left", "L_ee"), ("right", "R_ee")):
            Tee = env.link_pose(eef)
            fm = env.finger_grasp_point(arm)
            self._Rg[arm] = Tee[:3, :3].copy()
            self._off[arm] = Tee[:3, :3].T @ (fm - Tee[:3, 3])
        # side offset: half the box y-extent + a margin, so the two grasp centers straddle the box.
        gap = float(self.box.aabb_extent[1]) / 2 + 0.02
        self._side = {"left": np.array([0.0, gap, 0.0]), "right": np.array([0.0, -gap, 0.0])}
        # where each arm's grasp center starts (current pad centers), so segment 1 has no jump.
        self._start = {arm: env.finger_grasp_point(arm) for arm in ("left", "right")}
        # box-CENTER waypoints: (name, end_center_world, gripper, n_ticks). Approach from above the
        # box, close on the sides, lift, carry to the tray, lower, release.
        go, gc = env.grip_open, env.grip_close
        b, tr = self._box0, self._tray0
        lift_z = self._box_top + 0.14
        drop_z = float(self.tray.aabb[1][2]) + float(self.box.aabb_extent[2]) / 2 + 0.01
        self._segs = [
            ("pregrasp", b + np.array([0.0, 0.0, 0.10]),         go, 55),
            ("approach", b.copy(),                               go, 45),
            ("close",    b.copy(),                               gc, 45),
            ("lift",     np.array([b[0], b[1], lift_z]),         gc, 40),
            ("carry",    np.array([tr[0], tr[1], lift_z]),       gc, 55),
            ("place",    np.array([tr[0], tr[1], drop_z]),       gc, 45),
            ("release",  np.array([tr[0], tr[1], drop_z]),       go, 25),
            ("done",     np.array([tr[0], tr[1], drop_z + 0.12]), go, 15),
        ]
        print(f"[carry] box0={np.round(self._box0,3)} tray0={np.round(self._tray0,3)} "
              f"gap={gap:.3f} L_start={np.round(self._start['left'],3)} "
              f"R_start={np.round(self._start['right'],3)}", flush=True)

    def _grasp_target(self, arm, grasp_center_world):
        """eef pose so this arm's fingertip grasp center lands at grasp_center_world."""
        R = self._Rg[arm]
        pos = np.asarray(grasp_center_world, dtype=float) - R @ self._off[arm]
        return T_from(pos, R)

    def expert_step(self, env, obs) -> ExpertCommand:
        # Consistent every tick: lerp the box center along the waypoints, place each arm's grasp
        # center at box_center +/- side, map to an eef target with the same `_grasp_target`.
        t = self._t
        self._t += 1
        acc = 0
        prev = None
        seg = self._segs[-1]
        f = 1.0
        for s in self._segs:
            name, end_c, grip, n = s
            if t < acc + n:
                f = (t - acc) / n
                seg = s
                # first segment lerps each arm's grasp center from its own start; later segments
                # lerp the shared box center (the +/- side offset is added below).
                break
            acc += n
            prev = end_c
        name, end_c, grip, n = seg
        first = prev is None
        L = self._arm_target("left", first, prev, end_c, f)
        R = self._arm_target("right", first, prev, end_c, f)
        done = (name == "done" and f > 0.9)
        grip_action = 0.0 if np.isclose(grip, env.grip_open) else 1.0
        return ExpertCommand(left_target=L, right_target=R,
                             gripper_left=grip, gripper_right=grip,
                             gripper_action_left=grip_action,
                             gripper_action_right=grip_action,
                             done=done, phase=name)

    def _arm_target(self, arm, first, prev_center, end_center, f):
        if first:
            # segment 1: interpolate from the arm's current pad center to the box side (above).
            center = (1.0 - f) * self._start[arm] + f * (np.asarray(end_center) + self._side[arm])
        else:
            box_center = (1.0 - f) * np.asarray(prev_center) + f * np.asarray(end_center)
            center = box_center + self._side[arm]
        return self._grasp_target(arm, center)
