"""The two non-pick-and-place tasks: close a fridge door, walk a chair forward.

Both use the same control law as everything else here: one flat list of world waypoints for the
hand, lerped and mapped to an L_ee pose by `_grasp_target` on every tick. Each task spawns only the
station it operates; the station poses still come from the shared `tasks.layout` catalog.

`CloseFridgeTask` is non-prehensile. The gripper is commanded CLOSED for the whole episode (it is a
fist, not a jaw) and the waypoints run THROUGH the door rather than up to it -- the arm is
position-controlled, so commanding the hand into the panel is what generates the force that swings
it. Nothing is magnetised or teleported, which is why `GRASPING_MODE` is "physical" here: under
"sticky" the fist would grab the door on first touch. The arc is MEASURED, not derived: at reset the
door joint is driven through a sweep and the door link's world centre recorded at each angle, so the
waypoints follow the hinge exactly without needing the joint frame.

`PushChairTask` is NOT non-prehensile any more -- it grasps the backrest with both hands and tucks
the chair under the dining table. See its docstring.
"""
from __future__ import annotations

import numpy as np

from . import layout
from .long_horizon import TRANSIT_SPEED, TURN_RATE, LHSpec, LongHorizonPickPlace, _manhattan
from .pick_place_task import rot_z


def _link_center(obj, link_name):
    lo, hi = obj.links[link_name].aabb
    lo = lo.detach().cpu().numpy() if hasattr(lo, "detach") else np.asarray(lo)
    hi = hi.detach().cpu().numpy() if hasattr(hi, "detach") else np.asarray(hi)
    return (lo + hi) / 2


def _span_along(obj, axis):
    """(near, far) extent of an object's live world AABB projected onto a horizontal unit axis.

    Written as a projection rather than "take the x range" so the tuck geometry keeps meaning the
    same thing if a future layout works the chair from a heading other than world +x. At the
    axis-aligned headings this layout actually uses it IS the x range, exactly.
    """
    lo, hi = obj.aabb
    lo = lo.detach().cpu().numpy() if hasattr(lo, "detach") else np.asarray(lo)
    hi = hi.detach().cpu().numpy() if hasattr(hi, "detach") else np.asarray(hi)
    corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]], float)
    d = corners @ np.asarray(axis, dtype=float)[:2]
    return float(d.min()), float(d.max())


class _StationTask(LongHorizonPickPlace):
    """Shared scaffolding for the tasks that work a station without carrying anything to it."""

    @property
    def scene_station_keys(self):
        return (self.SPEC.station,)

    def object_configs(self):
        """This task's station, and nothing else -- these tasks carry no object from the table."""
        return layout.station_configs(self.scene_station_keys)

    def bind(self, env):
        objs = env.env.scene.objects
        self.table = min([o for o in objs if o.category == "breakfast_table"],
                         key=lambda o: np.linalg.norm(env.obj_pos(o)[:2]
                                                      - np.array(layout.SRC_TABLE_XY)))
        # The dining chairs are displaced here too, even though these tasks never touch the table:
        # "the indoor layout should remain same across all tasks" means the room has to LOOK the
        # same in all five, and the planner's `_clear_boxes` re-opens that floor unconditionally.
        self._chairs = layout.side_chairs(env)
        self.target = env.env.scene.object_registry("name", self.station.name)
        if self.target is None:
            raise ValueError(f"{self.name}: layout station {self.station.name!r} not in the scene")
        self.apple = self.bowl = self.target     # base-class handles, unused here

    def _drive_segments(self, env, carry_local, hand_z, heading, gripper, goal_xy):
        """Return ``(segments, headings)`` for the shared "get there" half of the plan.

        A short roll segment carries the reset heading, then the turn segment interpolates to the
        travel heading on the same waypoint clock as the hand. This matters because the head target
        is a full pose and head yaw is the chassis-yaw command; tagging every segment with the final
        heading made its orientation jump at tick zero instead of rotating with the base.
        """
        hand_off = rot_z(heading) @ carry_local
        park = np.asarray(goal_xy, dtype=float) - hand_off[:2]
        nav = self._nav(env, hand_off=hand_off, hand_z=hand_z)
        start = env.link_pose("base")[:2, 3]
        route = _manhattan(nav, nav.route(start, park), heading)
        legs, prev = [], np.asarray(route[0], dtype=float)
        for pt in route[1:]:
            dist = float(np.linalg.norm(pt - prev))
            hand = np.append(pt[:2], hand_z) + rot_z(heading) @ carry_local
            hand[2] = hand_z
            legs.append((f"transit{len(legs)}", hand, gripper,
                         max(40, int(round(dist / TRANSIT_SPEED * env.action_hz)))))
            prev = pt
        dpsi = float((heading - self.ROBOT_YAW + np.pi) % (2 * np.pi) - np.pi)
        roll_hand = np.append(start, hand_z) + rot_z(self.ROBOT_YAW) @ carry_local
        roll_hand[2] = hand_z
        drive = [("roll", roll_hand, gripper, 40)]
        headings = [float(self.ROBOT_YAW)]
        if abs(dpsi) > 1e-8:
            turn_hand = np.append(start, hand_z) + rot_z(heading) @ carry_local
            turn_hand[2] = hand_z
            drive.append(("turn", turn_hand, gripper,
                          max(60, int(round(abs(dpsi) / TURN_RATE * env.action_hz)))))
            headings.append(float(heading))
        drive.extend(legs)
        headings.extend([float(heading)] * len(legs))
        self._route_dbg = (route, park, np.degrees(dpsi), nav.free(*park))
        assert len(drive) == len(headings)
        return drive, headings


class CloseFridgeTask(_StationTask):
    """Drive to a fridge whose door was left open and push it shut."""

    name = "closefridge"
    GRASPING_MODE = "physical"      # a fist that pushes; sticky would magnetise on contact
    TRANSIT_Z = 1.00
    #: How far ajar the door starts. Modest on purpose: the recorded push points sit on the door
    #: PANEL, so at 60 deg open the panel centre is 0.64 m out from the fridge and the robot has to
    #: stand off even further to face it -- measured, the arm held 0.3-0.5 m of tracking error and
    #: never actually touched the door. At 40 deg the panel stays inside the arm's working volume.
    OPEN_ANGLE = 0.70               # rad (~40 deg)
    CLOSED_TOL = 0.15               # rad below which the door counts as shut
    ARC_STEPS = 6
    PUSH_TICKS = 520                # slow enough that the door swings with the hand
    APPROACH_TICKS = 260            # settle the base in front of the panel before contact
    CONTACT_OFFSET = 0.06           # door panel is thin; just clear its face
    BODY_MASS = 400.0               # see `_anchor_body`
    MAX_BODY_DRIFT = 0.10           # appliance must stay put; see `success`
    SPEC = LHSpec(name="closefridge", obj={}, station="fridge")

    def reset(self, env):
        layout.place_stations(env, self.scene_station_keys)
        layout.place_side_chairs(env, self._chairs)
        self._anchor_body()
        self._free_door()
        self._set_door(env, self.OPEN_ANGLE)
        for _ in range(5):
            env.env.step(env._hold_action())

    def bind(self, env):
        super().bind(env)
        self.fridge = self.target
        self._door_joint = self._pick_door_joint()

    def _pick_door_joint(self):
        """The revolute joint of the MAIN (largest) openable door."""
        cands = [(n, j) for n, j in self.fridge.joints.items() if j.joint_type == "RevoluteJoint"]
        if not cands:
            raise ValueError(f"{self.name}: fridge {self.fridge.model} has no revolute door joint")

        def area(item):
            link = item[1].body1.split("/")[-1]
            lo, hi = self.fridge.links[link].aabb
            lo = lo.detach().cpu().numpy() if hasattr(lo, "detach") else np.asarray(lo)
            hi = hi.detach().cpu().numpy() if hasattr(hi, "detach") else np.asarray(hi)
            return float(np.prod(hi - lo))
        name, joint = max(cands, key=area)
        self._door_link = joint.body1.split("/")[-1]
        return name

    def _anchor_body(self):
        """Make the cabinet heavy enough to stay put while its door swings.

        The fridge cannot be `fixed_base` (that locks its joints outright -- probed directly on
        this asset, the door would not rotate at all under a commanded joint velocity with both
        drive gains zeroed and the body woken, while the identical fridge spawned free moved
        straight away), so it is a free body and a shove on the door slides the whole appliance:
        measured, the door link travelled 0.73 m, far more than a 40 deg swing accounts for.
        Loading the base link up substitutes for the anchoring `fixed_base` would have given.
        """
        try:
            self.fridge.root_link.mass = self.BODY_MASS
        except Exception as exc:
            print(f"[{self.name}] could not set fridge mass: {exc}", flush=True)
        self._body0 = self.fridge.get_position_orientation()[0]
        self._body0 = (self._body0.detach().cpu().numpy()
                       if hasattr(self._body0, "detach") else np.asarray(self._body0))

    def body_drift(self, env):
        """How far the appliance itself slid -- a demo where it slides is not a door-closing demo."""
        p = env.obj_pos(self.fridge)
        return float(np.linalg.norm(p[:2] - self._body0[:2]))

    def _free_door(self):
        """Let the door swing when pushed.

        A DatasetObject's joints come with a position drive, so the door is actively held at
        whatever angle it was set to: the fist could lean on it all episode without the angle
        moving a milliradian, which is exactly what the first runs measured. Zeroing the drive
        stiffness leaves a passive hinge, with light damping so it does not swing wildly.
        """
        joint = self.fridge.joints[self._door_joint]
        for attr, value in (("stiffness", 0.0), ("damping", 0.5), ("friction", 0.0)):
            try:
                setattr(joint, attr, value)
            except Exception as exc:
                print(f"[{self.name}] could not set door {attr}: {exc}", flush=True)
        print(f"[{self.name}] door joint {self._door_joint} link={self._door_link} "
              f"driven={joint.driven} stiffness={joint.stiffness} damping={joint.damping} "
              f"limits={joint.lower_limit:.2f}..{joint.upper_limit:.2f}", flush=True)

    def _set_door(self, env, angle):
        import torch as th
        idx = list(self.fridge.joints).index(self._door_joint)
        q = self.fridge.get_joint_positions()
        q[idx] = angle
        self.fridge.set_joint_positions(th.as_tensor(q))
        qd = self.fridge.get_joint_velocities()
        qd[idx] = 0.0
        self.fridge.set_joint_velocities(th.as_tensor(qd))

    def _door_angle(self, env):
        idx = list(self.fridge.joints).index(self._door_joint)
        q = self.fridge.get_joint_positions()
        return float(q[idx].detach().cpu().numpy() if hasattr(q[idx], "detach") else q[idx])

    def _push_path(self, env):
        """Measure the door centre through the closing sweep; those points ARE the waypoints."""
        here = self._door_angle(env)
        self._door_start = here      # success requires it to have STARTED open
        pts = []
        for ang in np.linspace(self.OPEN_ANGLE, 0.0, self.ARC_STEPS):
            self._set_door(env, float(ang))
            env.env.step(env._hold_action())
            pts.append(_link_center(self.fridge, self._door_link))
        self._set_door(env, here)                     # leave the door as the episode found it
        env.env.step(env._hold_action())
        path = np.stack(pts)
        path[:, 2] = np.clip(path[:, 2], 0.85, 1.15)  # push at a comfortable working height
        return path

    def _build_segments(self, env):
        self._last_head_cmd = None
        self._clock = 1.0
        gc = env.grip_close
        path = self._push_path(env)
        self._obj0, self._goal0 = path[0].copy(), path[-1].copy()
        self._door_progress_during_push = 0.0
        base0 = env.link_pose("base")[:3, 3]
        carry_local = rot_z(-self.ROBOT_YAW) @ (self._start_center - base0)

        # Direction the push travels, and therefore the heading the push phase needs. The park MUST
        # be derived from the push, not from a nominal station park: routing to the latter left the
        # base 0.7 m from where the contact actually happens, so it was still driving when the hand
        # swung in -- measured, the door never moved at all.
        d = path[1] - path[0]
        d = d / max(float(np.linalg.norm(d)), 1e-9)
        push_heading = float(np.arctan2(d[1], d[0]))
        drive, drive_headings = self._drive_segments(
            env, carry_local, self.TRANSIT_Z, push_heading, gc, path[0][:2]
        )

        path = path - self.CONTACT_OFFSET * d      # touch the near face, not the centre
        stage = path[0] - 0.18 * d
        per = max(30, int(round(self.PUSH_TICKS / max(len(path) - 1, 1))))
        pushes = [(f"push{i}", p, gc, per) for i, p in enumerate(path[1:], start=1)]
        segs = [
            *drive,
            ("stage",  stage,   gc, self.APPROACH_TICKS),
            ("engage", path[0], gc, 60),
            *pushes,
            ("back",   path[-1] - 0.20 * d, gc, 60),
        ]
        # Roll at the reset heading, rotate head and body together through the turn, then hold the
        # push heading while approaching and closing the door.
        self._headings = drive_headings + [push_heading] * (len(segs) - len(drive))
        self._standoffs = [1.0] * len(segs)
        assert len(self._headings) == len(segs) == len(self._standoffs)
        route, park, dpsi, free = self._route_dbg
        print(f"[{self.name}] route {np.round(route, 2).tolist()}; park {np.round(park, 2).tolist()}"
              f" free={free}; turn {dpsi:+.0f} deg; push {np.round(path[0], 3)} -> "
              f"{np.round(path[-1], 3)} in {len(path) - 1} steps", flush=True)
        return segs

    def objects_of_interest(self, env):
        return np.stack([_link_center(self.fridge, self._door_link), self._goal0])

    def _release_point(self, env):
        return self._goal0

    def keepouts(self):
        return [(layout.SRC_TABLE_XY[0], layout.SRC_TABLE_XY[1], self.SRC_KEEPOUT),
                (self.station.xy[0], self.station.xy[1], self.station.keepout)]

    def success(self, env):
        """Door shut AND the appliance still where it started.

        The angle alone is not enough: with the fridge free to slide, barging into it also swings
        the door shut, which would score a success for a demonstration nobody wants to train on.
        """
        drift = self.body_drift(env)
        # abs(): the comparison is on a signed joint angle, so any sufficiently NEGATIVE angle
        # would otherwise read as "closed". And the door must have been open to begin with --
        # if it settled shut before contact the episode demonstrates nothing.
        closed = bool(abs(self._door_angle(env)) < self.CLOSED_TOL)
        started_open = float(getattr(self, "_door_start", 0.0)) >= self.OPEN_ANGLE - self.CLOSED_TOL
        if drift > self.MAX_BODY_DRIFT:
            print(f"[{self.name}] fridge slid {drift:.3f} m (limit {self.MAX_BODY_DRIFT}) -- "
                  f"pushed the appliance, not the door", flush=True)
            return False
        demonstrated = self._door_progress_during_push >= 0.5 * self.OPEN_ANGLE
        if not demonstrated:
            print(f"[{self.name}] door closed without enough recorded push progress: "
                  f"{self._door_progress_during_push:.3f} rad", flush=True)
        return bool(started_open and closed and demonstrated)

    def expert_step(self, env, obs):
        cmd = super().expert_step(env, obs)
        if cmd.phase.startswith("push") or cmd.phase == "engage":
            self._door_progress_during_push = max(
                self._door_progress_during_push,
                float(self._door_start - self._door_angle(env)),
            )
        return cmd


class PushChairTask(_StationTask):
    """Tuck a dining chair in: take its backrest with both grippers and push it under the table.

    The chair stands pulled out in front of the room's own dining table (see `tasks.layout`), the
    robot drives up behind it, comes straight down onto the top rail of the backrest with both
    hands, closes, and walks the chair forward until it is under the tabletop.

    It grasps rather than shoves, and the real data is why: over 41 box2cloth episodes the base
    drives exactly ONE chassis axis at a time and never yaws, with exactly two gripper transitions
    per episode. A non-prehensile push cannot produce that -- contact repeatedly breaks and
    re-forms, the reaction torque rings the chassis (measured 210-478 deg of yaw variation) and the
    outcome flipped between identical runs. Holding the chair removes the contact-chatter loop
    entirely: once grasped, the chair is part of the arm and moving it is the same carry the
    pick-and-place tasks already do cleanly.

    The chair is turned backrest-first toward the approach, so the hand comes straight DOWN onto
    the rail and the push is then a pure forward strafe -- one chassis axis for the whole tuck,
    exactly like the hardware. And the push distance is not a number: it is read off the table and
    the chair every episode, so the task means "put this chair under that table" rather than "move
    this chair 35 cm".
    """

    name = "pushchair"
    GRASPING_MODE = "assisted"      # requires two-pad contact and a ray hit on each handle
    #: How far above the rail the hand starts its descent. The grasp itself is a straight vertical
    #: drop onto the rail from here -- "from above" is the whole approach, not a wrist orientation:
    #: a top-down wrist blows up the IK (see `PickPlaceTask.expert_reset`), so the hand keeps its
    #: natural forward-and-slightly-down approach axis and simply comes down on the rail.
    #: 0.12 rather than a rounder number because of the head command's own ceiling: the hand rides
    #: this high through the drive, the head target is the hand plus a fixed 0.53 m, and
    #: `LongHorizonPickPlace._head_target_pos` clamps that at the reset standing height + 0.04
    #: (~1.003 m of hand height). This chair's rail is at 0.877, so 0.14 would ask for 1.017 and
    #: leave a permanent clamped residual for the solver to trade the base against; 0.12 asks for
    #: 0.997 and clears the rail by four times its thickness.
    PREGRASP_LIFT = 0.12
    #: Distance BELOW the chair's top the fingers close at -- on the rail, not above it.
    RAIL_DROP = 0.03
    #: Inset from the backrest's back face toward the seat, so the pads straddle the rail rather
    #: than closing on air behind or in front of it. MEASURED as solid occupancy at grasp height on
    #: the chair the layout spawns: its rail is a straight bar spanning the full width, 20-30 mm
    #: thick, and both pads land in material for RAIL_INSET 0.030-0.055 with 30 mm runs. 0.042 is
    #: the middle of that band, so the ~0 mm the chair settles by has 13 mm of margin either side.
    #: (This constant only works at all because the rail is straight -- see the layout's note on
    #: why a bowed backrest cannot be grasped from one x.)
    RAIL_INSET = 0.042
    #: How far each grasp is inset from the lateral AABB edge, so the two targets sit near the ends
    #: of the rail rather than on top of each other. 0.05 puts the hands 0.36 m apart and asks the
    #: right arm for a 0.29 m cross-body reach; widening to 0.08 costs another 6 cm of reach and
    #: thins the run from 30 mm to 22 mm, so it buys nothing.
    HANDLE_INSET = 0.05
    #: How far short of the table's near face the backrest is driven. The tuck has a PHYSICAL stop:
    #: the backrest stands taller than the tabletop, so it meets the table's near edge and the chair
    #: goes no further. Probed by teleport-and-settle in 2 cm steps with the table re-pinned each
    #: step (it is not fixed_base, and letting one collision shove it invalidates every later step):
    #: the last pose that held had its back at 1.008 and the next was blocked at 1.016, against a
    #: table face at 1.080 -- so the jam is 0.068 m short of it. 0.10 keeps ~3 cm off that stop, so
    #: the expert tucks the chair rather than driving it into the table.
    BACK_STANDOFF = 0.10
    #: Commanded speed of the final descent onto the rail, and how long to dwell there before the
    #: fingers close. The descent is the only segment that ENDS in contact, and the grasp is what
    #: depends on that contact, so it is the one place a flat tick count was buying nothing: at the
    #: old 150 ticks (0.08 m/s) the pads arrived as an impulse and knocked the chair 4 cm sideways
    #: and 6 deg over in the last centimetres, which is what cost the right hand its rail.
    DESCENT_SPEED = 0.03
    SETTLE_ON_RAIL = 30
    #: Commanded speed of the push. The push is a BASE move, not an arm move -- the hand target
    #: travels the whole tuck and the head command drags the chassis after it -- and the follower
    #: has a 0.06 m/s single-axis dead-band (see `long_horizon.TRANSIT_SPEED` for the measurement).
    #: Below that the base can only average a speed by duty-cycling, which is the one thing a
    #: grasped chair cannot tolerate. The old fixed 700 ticks over 0.35 m was 0.05 m/s, under it.
    PUSH_SPEED = 0.08
    MIN_PUSH_TICKS = 400
    #: How much of the chair has to end up under the tabletop for the tuck to count, and how far it
    #: has to have travelled to get there. The planned push is 0.59 m and leaves 0.40 m of the chair
    #: under the tabletop, so these sit well inside what a clean run achieves while still failing a
    #: chair merely nudged toward the table.
    MIN_UNDER = 0.25
    MIN_ADVANCE = 0.40
    MAX_DRIFT = 0.15
    MAX_Z_DROP = 0.08
    MAX_TILT = 0.10
    #: The Rs_int dining table is NOT fixed_base, so a chair barged into it slides the TABLE out of
    #: the way instead of going under it -- and would then satisfy every geometric test while
    #: demonstrating the opposite of the task. Same guard, and same reason, as the fridge's
    #: `MAX_BODY_DRIFT`.
    MAX_TABLE_DRIFT = 0.05
    SPEC = LHSpec(name="pushchair", obj={}, station="chair")

    def bind(self, env):
        super().bind(env)
        self.chair = self.target

    def reset(self, env):
        layout.place_stations(env, self.scene_station_keys)
        layout.place_side_chairs(env, self._chairs)
        self.table.keep_still()      # the tuck is measured against it; it must start at rest
        # Settle, then restore the ROBOT: stepping the sim to settle the chair also droops the arm,
        # and expert_reset would otherwise derive every offset from a sagged pose.
        q_robot = env.robot.get_joint_positions().clone()
        for _ in range(60):
            env.env.step(env._hold_action())
        env.robot.set_joint_positions(q_robot)
        env.env.step(env._hold_action())
        self._chair0 = env.obj_pos(self.chair).copy()
        self._table0 = env.obj_pos(self.table).copy()
        self._upright0 = self._upright(env)

    def _upright(self, env):
        q = self.chair.get_position_orientation()[1]
        q = q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)
        return 1.0 - 2.0 * (float(q[0]) ** 2 + float(q[1]) ** 2)

    def keepouts(self):
        # only the source table -- the chair is what we drive AT and then hold
        return [(layout.SRC_TABLE_XY[0], layout.SRC_TABLE_XY[1], self.SRC_KEEPOUT)]

    def objects_of_interest(self, env):
        return np.stack([env.obj_pos(self.chair),
                         np.array([*self._goal_xy, self._chair0[2]])])

    def _grasp_points(self, env):
        """World points on the two side handles, nearest the approaching robot.

        Taken from the chair's live AABB rather than a hand-measured offset, so it follows the
        asset and the layout's spawn yaw instead of encoding both. The rail is the only part of
        this chair narrow enough for the 85 mm jaw -- the seat and legs are not graspable.
        """
        lo, hi = self.chair.aabb
        lo = lo.detach().cpu().numpy() if hasattr(lo, "detach") else np.asarray(lo)
        hi = hi.detach().cpu().numpy() if hasattr(hi, "detach") else np.asarray(hi)
        c = env.obj_pos(self.chair)
        heading = float(self.station.approach)
        back = -np.array([np.cos(heading), np.sin(heading)])   # from the chair toward the robot
        left = np.array([-np.sin(heading), np.cos(heading)])
        half = (hi[:2] - lo[:2]) / 2
        edge = c[:2] + back * (np.abs(back) @ half - self.RAIL_INSET)
        handle_half_span = float(np.abs(left) @ half - self.HANDLE_INSET)
        z = float(hi[2]) - self.RAIL_DROP
        return (
            np.array([*(edge + handle_half_span * left), z]),
            np.array([*(edge - handle_half_span * left), z]),
        )

    def _grasp_point(self, env):
        """Compatibility helper: the left-handle point."""
        return self._grasp_points(env)[0]

    def _build_segments(self, env):
        """Descend onto the rail, close, walk the chair under the table, release."""
        self._last_head_cmd = None
        self._clock = 1.0
        go, gc = env.grip_open, env.grip_close
        heading = float(self.station.approach)
        grasp, right_grasp = self._grasp_points(env)
        fwd = np.array([np.cos(heading), np.sin(heading), 0.0])   # the robot's own forward
        # How far to push is READ OFF THE ROOM, not stored: drive the backrest up to BACK_STANDOFF
        # short of the table's near face. Deriving it means the plan still tucks the chair if the
        # layout mark, the chair model or the table ever moves, instead of silently pushing a fixed
        # distance to somewhere that is no longer under the table.
        table_face, _ = _span_along(self.table, fwd[:2])
        chair_back, chair_front = _span_along(self.chair, fwd[:2])
        push_dist = (table_face - self.BACK_STANDOFF) - chair_back
        tucked = chair_front + push_dist - table_face
        if push_dist <= 0.0:
            raise ValueError(
                f"{self.name}: the chair already stands at the table (back {chair_back:.3f}, "
                f"table near face {table_face:.3f}) -- there is nothing to push"
            )
        if tucked < self.MIN_UNDER:
            raise ValueError(
                f"{self.name}: driving the backrest to {self.BACK_STANDOFF:.3f} m short of the "
                f"table would leave only {tucked:.3f} m of the chair under the tabletop, below "
                f"MIN_UNDER={self.MIN_UNDER}; this chair is too shallow for this table"
            )
        push = push_dist * fwd
        push_ticks = max(self.MIN_PUSH_TICKS,
                         int(round(push_dist / self.PUSH_SPEED * env.action_hz)))
        self._push_dist = push_dist
        self._goal_xy = env.obj_pos(self.chair)[:2] + push[:2]
        self._push_axis = fwd[:2].copy()
        drag_end = grasp + push
        right_drag_end = right_grasp + push

        base0 = env.link_pose("base")[:3, 3]
        carry_local = rot_z(-self.ROBOT_YAW) @ (self._start_center - base0)
        right_carry_local = rot_z(-self.ROBOT_YAW) @ (self._right_start_center - base0)
        above = grasp + np.array([0.0, 0.0, self.PREGRASP_LIFT])
        right_above = right_grasp + np.array([0.0, 0.0, self.PREGRASP_LIFT])
        # The descent is timed off a speed, not given a flat tick count, because it ENDS in contact
        # and the contact is what the grasp depends on. Instrumented tick by tick at the old flat
        # 150 ticks (0.08 m/s), the chair sat untouched all the way to t=500 and was then knocked
        # -2.7 cm in x, +4.0 cm in y and +2.9 cm in z and pitched to +7.5 deg in the last
        # centimetres -- carrying the right-hand rail out from between the right pads, which then
        # closed on air while the left grasped normally.
        descend_ticks = max(200, int(round(self.PREGRASP_LIFT / self.DESCENT_SPEED
                                           * env.action_hz)))
        drive, drive_headings = self._drive_segments(
            env, carry_local, float(above[2]), heading, go, grasp[:2]
        )
        right_drive = []
        carry_delta = right_carry_local - carry_local
        for (name, end, _grip, ticks), drive_heading in zip(drive, drive_headings):
            right_end = np.asarray(end) + rot_z(drive_heading) @ carry_delta
            right_drive.append((name, right_end, go, ticks))

        segs = [
            *drive,
            ("descend",  grasp.copy(),   go, descend_ticks),
            ("settle",   grasp.copy(),   go, self.SETTLE_ON_RAIL),
            ("close",    grasp.copy(),   gc, 120),
            ("push",     drag_end,       gc, push_ticks),
            ("release",  drag_end,       go, 60),
            ("retreat",  drag_end + np.array([0.0, 0.0, self.PREGRASP_LIFT]), go, 60),
        ]
        self._right_segs = [
            *right_drive,
            ("descend",  right_grasp.copy(), go, descend_ticks),
            ("settle",   right_grasp.copy(), go, self.SETTLE_ON_RAIL),
            ("close",    right_grasp.copy(), gc, 120),
            ("push",     right_drag_end,     gc, push_ticks),
            ("release",  right_drag_end,     go, 60),
            ("retreat",  right_drag_end + np.array([0.0, 0.0, self.PREGRASP_LIFT]), go, 60),
        ]
        self._ever_grasped = {"left": False, "right": False}
        self._simultaneously_grasped = False
        # The chair is now directly ahead, so these are all the reset heading. For any future
        # turned layout, `_drive_segments` still rotates the head target with the turn.
        self._headings = drive_headings + [heading] * (len(segs) - len(drive))
        self._standoffs = [1.0] * len(segs)
        assert len(self._headings) == len(segs) == len(self._standoffs) == len(self._right_segs)
        assert [(name, ticks) for name, _end, _grip, ticks in segs] == [
            (name, ticks) for name, _end, _grip, ticks in self._right_segs
        ]
        route, park, dpsi, free = self._route_dbg
        print(f"[{self.name}] route {np.round(route, 2).tolist()}; park {np.round(park, 2).tolist()}"
              f" free={free}; turn {dpsi:+.0f} deg; grasps {np.round(grasp, 3)}, "
              f"{np.round(right_grasp, 3)} -> push to {np.round(drag_end, 3)}, "
              f"{np.round(right_drag_end, 3)}", flush=True)
        print(f"[{self.name}] tuck: table face {table_face:.3f}; chair {chair_back:.3f}.."
              f"{chair_front:.3f} -> push {push_dist:.3f} m in {push_ticks} ticks "
              f"({push_dist / (push_ticks / env.action_hz):.3f} m/s), leaving {tucked:.3f} m of "
              f"the chair under the tabletop (need {self.MIN_UNDER})", flush=True)
        return segs

    def expert_step(self, env, obs):
        cmd = super().expert_step(env, obs)
        ever_grasped = getattr(self, "_ever_grasped", None)
        if ever_grasped is not None:
            for arm in ("left", "right"):
                ever_grasped[arm] |= env.is_grasping(arm) is self.chair
        self._simultaneously_grasped |= (
            env.is_grasping("left") is self.chair and env.is_grasping("right") is self.chair
        )
        return cmd

    def _release_point(self, env):
        return np.array([*self._goal_xy, self._chair0[2]])

    def _tuck_depth(self, env):
        """How far the chair's leading edge has passed the table's near face, along the push."""
        table_face, _ = _span_along(self.table, self._push_axis)
        _, chair_front = _span_along(self.chair, self._push_axis)
        return chair_front - table_face

    def _calibrate_grasp_delta(self, env):
        """Keep the rail trajectory fixed after attachment.

        The inherited pick-and-place correction shifts destination hand waypoints so an object's
        centre lands at a receptacle centre. Here the waypoint is already the physical rail point:
        moving it by the push distance is exactly how far the attached chair should move. Applying
        the chair-COM-to-finger offset would instead put a large jump into release and retreat.
        """

    def success(self, env):
        """Both handles grasped and released, and the chair now stands UNDER the table.

        The tuck depth is the task; `advance` is kept alongside it as the guard that the chair was
        actually walked in rather than having started there. The table drift is the guard against
        the degenerate way to satisfy both: it is not fixed_base, so a chair barged into it slides
        the TABLE, which passes every geometric test while demonstrating the opposite of the task.
        """
        if env.is_grasping("left") is not None or env.is_grasping("right") is not None:
            return False
        if not all(self._ever_grasped.values()):
            print(f"[{self.name}] missing handle grasp(s): {self._ever_grasped}", flush=True)
            return False
        if not self._simultaneously_grasped:
            print(f"[{self.name}] handles were never held simultaneously", flush=True)
            return False
        p = env.obj_pos(self.chair)
        moved = p[:2] - self._chair0[:2]
        axis = self._push_axis / max(float(np.linalg.norm(self._push_axis)), 1e-9)
        advance = float(np.dot(moved, axis))
        drift = float(abs(moved[0] * axis[1] - moved[1] * axis[0]))
        upright = self._upright(env)
        under = self._tuck_depth(env)
        table_moved = float(np.linalg.norm(env.obj_pos(self.table)[:2] - self._table0[:2]))
        print(f"[{self.name}] under table={under:.3f} m (need {self.MIN_UNDER}) "
              f"advance={advance:.3f} (need {self.MIN_ADVANCE}) "
              f"drift={drift:.3f} (max {self.MAX_DRIFT}) upright={upright:.3f}/{self._upright0:.3f} "
              f"table moved={table_moved:.3f} (max {self.MAX_TABLE_DRIFT})", flush=True)
        return bool(under >= self.MIN_UNDER and advance >= self.MIN_ADVANCE
                    and drift <= self.MAX_DRIFT and table_moved <= self.MAX_TABLE_DRIFT
                    and abs(float(p[2] - self._chair0[2])) < self.MAX_Z_DROP
                    and upright > self._upright0 - self.MAX_TILT)


NP_TASKS = [CloseFridgeTask, PushChairTask]
