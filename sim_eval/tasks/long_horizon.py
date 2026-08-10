"""Long-horizon mobile pick-and-place: the sim counterparts of the real-robot task list.

The real plan is dish2rack / towel2rack / book2shelf, i.e. pick an object off a work surface,
DRIVE somewhere else, and place it. That is the same skill as `MobilePickPlaceTask` with a much
longer middle, so this module reuses that task wholesale and changes only two things:

1. **The waypoint list gains transit legs.** `_build_segments` still returns one flat list of
   world grasp-centre waypoints, and `expert_step` still lerps between them and maps each to an
   L_ee pose with the same `_grasp_target`. Transit is not a mode: it is waypoints that happen to
   be metres apart, so the whole-body IK drags the base along exactly as it already does over the
   0.9 m table approach. Waypoint time advances by one tick on every simulation step; measured
   tracking error never pauses or accelerates the scripted target.

2. **The gaze heading remains the spawn heading.** The wall stations face the robot, so no turn is
   needed. `head_j1`/`head_j2` are pinned by `head_ik_pinned_joints`, making head yaw a base-yaw
   command; keeping it at zero reproduces the real box2cloth data, which never commands wz.

**Everything about the drive is shaped by what the hardware chassis can actually do**, measured on
41 real box2cloth episodes (`action/joint/chassis_*`, 10 Hz): while the base is moving, exactly ONE
of vx / vy / wz is live **100.0 %** of the time, and wz is never commanded at all -- the operator's
paths are axis-aligned strafes. `base_ctrl.project_single_axis` reproduces that arbitration in sim,
which means a leg that mixes vx and vy makes the projection flap between them at loop rate. That is
the stutter. So:

  * the carried route is explicitly back -> right strafe -> forward, matching box2cloth rather
    than allowing a shortest-path planner to choose a different ordering;
  * the heading stays fixed for the whole episode;
  * waypoint time advances by exactly one tick per simulation tick; segment durations, rather
    than measured tracking error, set a steady commanded speed;
  * the commanded hand position is continuous and piecewise linear, so positions never jump at
    segment boundaries and every segment advances at its prescribed constant speed.

Routes are planned on the scene's own traversability map (`nav_map.NavMap`), not hand-traced. This
matters more than it sounds: Rs_int reads as open-plan in the object list, but `walls_exiopd_0`
seals the kitchen off from the living room with one doorway at x = -2.6, so every "drive to the
counter/fridge/trash can" route a human would sketch is unwalkable. The stations therefore live in
the living-room region, laid out once in `tasks.layout` and IDENTICAL for every task.

Source is always the `breakfast_table` the tabletop tasks already use, so the grasp half of every
task is the trajectory that is already validated end to end; only the transit and place are new.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from . import layout
from .layout import CLEAR_CHAIRS, DECLUTTER, SRC_PARK, STATIONS
from .mobile_pick_place_task import MobilePickPlaceTask
from .nav_map import NavMap
from .pick_place_task import rot_z


def rot_y(pitch):
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

#: Fixed commanded transit speed of the grasp-centre waypoint. The measured-error clock is gone
#: (it made the ACTION target visibly speed up and pause, a human-teleoperation artifact a
#: scripted expert should not have), but the 0.08 m/s that replaced it put the same artifact back
#: one level down, in the PLANT: 0.08 m/s is below the speed the chassis can hold continuously,
#: so the base could only average it by duty-cycling. Instrumented over a book2shelf transit, the
#: dispatched command was ZERO on 36 % of ticks and surged to 0.24 m/s in between -- a 0.5 s
#: sawtooth (measured base speed 0.081 +- 0.068 m/s). The cycle is structural: the follower slews
#: at 0.4 m/s^2, so it needs 15 ticks to climb from rest to the 0.06 m/s single-axis deadband and
#: the base stands still for all of them; the burst that follows overshoots, the solver's own
#: single-axis projection then hands one tick to vx, and `deadband_by_reference` zeroes the y
#: slew anchor outright, restarting the ramp from zero.
#: 0.17 m/s is the real robot's own transit speed, so it clears that floor by construction: over
#: 41 box2cloth episodes the commanded chassis speed while driving has median 0.168 m/s (deciles
#: 0.076 .. 0.276) and NOT ONE moving tick drives a linear axis below 0.06 m/s. Lowering the
#: follower deadbands instead would have been the wrong repair -- it would buy smoothness in sim
#: that the hardware cannot reproduce, in demonstrations meant to transfer to it.
TRANSIT_SPEED = 0.17
#: Follower speed cap for the whole episode, replacing the 0.45 m/s wbik.yaml teleop value.
#:
#: Raising TRANSIT_SPEED alone does not make the base move steadily, because the chassis command
#: never carries the plan's speed in the first place. Measured over a transit leg: the base tracks
#: its commanded pose to within 4.5 cm (max 9.8 cm), yet the QP's `base_twist` sits pinned at its
#: own 0.900 m/s cap on 91 % of ticks -- 2 cm of lag already saturates it. The solver twist is
#: therefore a BANG-BANG "go / stop" signal, and the follower's 0.4 m/s^2 slew limiter integrates
#: that square wave into a triangle: with a 0.45 m/s ceiling the base accelerated to 0.36 m/s,
#: overtook its own target, and the y demand then collapsed far enough that a leftover 4 cm x
#: error won the solver's single-axis argmax for one tick -- which reads to
#: `deadband_by_reference` as "y went quiet" and zeroes the y slew anchor outright. 0.2 s of
#: standstill, then the whole ramp again, 13 times per leg.
#: Capping the follower just above the planned speed removes that mechanism rather than damping
#: it: the saturated twist is clamped to a speed the plan actually wants, so the base does not run
#: away from its target and the demand does not collapse. The margin over TRANSIT_SPEED is what
#: lets a corner's residual lag be recovered.
#: Measured on the same leg, the two together take the dispatched command from zero on 36 % of
#: ticks to 7.8 %, and the base speed from 0.081 +- 0.068 m/s to 0.168 +- 0.058.
#: What is LEFT is not the same fault and is not a task parameter: driving one axis at a time
#: means the ~4 cm of cross-axis drift a strafe accumulates can only be corrected by taking ticks
#: away from the travel axis, and each handover costs one dead-zone crossing (~0.15 s, about once
#: every 1.75 s here). That is the shipped controller stack doing what wbik.yaml says it does --
#: the same "post-slew tail-cut at axis-switch" the hardware config documents -- so it is left
#: alone, for the reason given under TRANSIT_SPEED.
BASE_MAX_LIN = round(1.2 * TRANSIT_SPEED, 3)
#: Retained for the non-prehensile station tasks, whose existing layouts may still require a turn.
TURN_RATE = 0.35
#: World x of the post-pick retreat lane. From SRC_PARK this creates the real-data order
#: -vx (back away from the source), -vy (strafe right), +vx (approach the wall station).
RETREAT_X = 0.20
#: How far the commanded head may sit below its own reset standing height. The head command is
#: `work_point + rest_stand_off`, so a work point ABOVE the hands' rest height asks for a camera
#: position the neck and torso cannot reach -- and the head-position cost is 10000 on all three
#: axes, so an infeasible z is a permanent residual the solver trades the base against. Measured:
#: carrying at 1.02 m put the head command at 1.645 m (reset height is 1.497 m) and the base froze
#: for the whole 1.8 m transit, `base_pose` moving 1 mm while the target swept past it.
HEAD_Z_DROP = 0.36
HEAD_Z_RISE = 0.04
#: Fraction of the rest camera-to-hand stand-off actually commanded, horizontally. At 1.0 the base
#: parks 0.70 m from the work point, and reaching a 0.76 m tabletop from there straightens the
#: elbow: measured, `L_arm_j4` sat pinned at its +14 deg upper limit (URDF range [-176, +14],
#: nominal -89) for whole episodes, so the arm had no redundancy left for the grasp or the place.
#: Parking closer does NOT pay for itself, though, so this stays at 1.0. Tried at 0.72: the elbow
#: came off its stop, but `L_arm_j2` then sat on ITS -26 deg limit instead, the shortened stand-off
#: put a 0.17 m step into the head command at t=0 (the reset pose no longer satisfies it), the arm
#: held 0.20 m of tracking error through the whole approach and knocked the bowl onto the floor.
#: The straight elbow is therefore a REAL property of working a 0.76 m table at this stand-off --
#: it is the same on hardware, same URDF -- and not the thing that was breaking the grasps.
STANDOFF_SCALE = 1.0
#: Horizontal distance the head command parks the base at from its work point. It is just
#: |head_off_base[:2]|, the robot's own rest camera-to-hand offset, so it is a constant of the
#: hardware (measured 0.692 m) rather than a tuning knob.
NOMINAL_PARK_DIST = 0.692
#: Preserve the reset camera-to-hand geometry at the destination. The explicit route already ends
#: at the park implied by that geometry. Shortening this to 0.62 made the head ask the base to move
#: after arrival while the left hand stayed fixed on the rack and the right hand followed the
#: implied body pose; those mutually strained targets drove the torso / arms into their limits and
#: left 16 cm of EEF error at release. Full stand-off keeps head, both hands, and park consistent.
PLACE_STANDOFF = 1.0
#: Legs shorter than this are dropped: a few-centimetre leg is numerical noise on the shortcut,
#: and each one costs a chassis axis switch for no travel.
MIN_LEG = 0.15
#: Clearance around each shifted hand point in the route map. Unlike the chassis boxes, these
#: boxes are applied after chassis-radius erosion (see NavMap.point_block_boxes), so this is the
#: actual gripper / payload margin rather than an accidental extra 0.33 m chassis radius.
HAND_SWEEP_PAD = 0.06


def _manhattan(nav, route, yaw, min_leg=MIN_LEG):
    """Rewrite a planned polyline so every leg is a pure single-axis move in the `yaw` frame.

    The chassis arbitrates: `base_ctrl.project_single_axis` keeps the dominant one of vx / vy / wz
    and zeroes the rest, faithfully reproducing the follower (and, measured, the real robot, which
    drives exactly one axis 100 % of the time). A diagonal leg asks for vx AND vy at once, so the
    projection alternates between them at loop rate and the base crabs in visible steps.

    Each planned leg therefore becomes two: one along the body's forward axis and one along its
    lateral axis. Both orderings are tried and the one whose corner stays on traversable floor
    wins; if neither does, the diagonal is kept rather than driven through a wall.
    """
    R = rot_z(yaw)[:2, :2]
    out = [np.asarray(route[0], dtype=float)]
    for pt in np.asarray(route, dtype=float)[1:]:
        a = out[-1]
        d = R.T @ (pt - a)                       # displacement in the body frame
        for first in (0, 1):
            step = np.zeros(2)
            step[first] = d[first]
            mid = a + R @ step
            if min(abs(d[0]), abs(d[1])) < min_leg:
                break                            # already single-axis; nothing to split
            if nav.visible(a, mid) and nav.visible(mid, pt):
                out.append(mid)
                break
        out.append(pt)
    keep = [out[0]]
    for p in out[1:]:
        if float(np.linalg.norm(p - keep[-1])) >= min_leg:
            keep.append(p)
    if len(keep) == 1 or float(np.linalg.norm(out[-1] - keep[-1])) > 1e-6:
        keep.append(out[-1])
    return np.stack(keep)


def _box2cloth_route(nav, start, goal, retreat_x=RETREAT_X):
    """Validated back -> right-strafe -> forward route, all at fixed world yaw zero."""
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    route = np.stack([
        start,
        np.array([retreat_x, start[1]], dtype=float),
        np.array([retreat_x, goal[1]], dtype=float),
        goal,
    ])
    for a, b in zip(route, route[1:]):
        d = b - a
        if np.count_nonzero(np.abs(d) > 1e-8) != 1:
            raise ValueError(f"box2cloth route contains a non-axis-aligned leg: {a} -> {b}")
        if not nav.visible(a, b):
            raise ValueError(f"box2cloth route is blocked: {a} -> {b}")
    return route


@dataclasses.dataclass(frozen=True)
class LHSpec:
    """One long-horizon pick-and-place task, as data."""
    name: str
    #: the carried object: DatasetObject kwargs (category/model/scale)
    obj: dict
    #: which piece of the SHARED layout it must end up on/in (`tasks.layout.STATIONS`)
    station: str
    #: extra clearance above the receptacle's top when releasing
    release_dz: float = 0.02
    #: place ON TOP of the receptacle rather than requiring Inside
    on_top: bool = False
    #: where on the object to close the fingers, as an offset from its centre (object frame).
    #: The 2F-85 opens 85 mm, so anything wider than that has to be taken by an edge: a plate is
    #: 110 mm across but 8 mm at the rim, and the pads would otherwise collide with both rims and
    #: never close. (0,0,0) closes on the centre, which is right for the compact objects.
    grasp_offset: tuple = (0.0, 0.0, 0.0)
    #: spawn orientation (xyzw). Flat objects stand on edge so the pinch spans their thickness.
    obj_quat: tuple = (0.0, 0.0, 0.0, 1.0)
    #: Object-centre placement offset in the station's local frame. A wire rack does not have a
    #: solid centre, so the dish target sits between two tested supporting rails.
    place_offset: tuple = (0.0, 0.0, 0.0)
    #: Optional support surface height above the receptacle's live AABB bottom. Solid furniture
    #: uses its AABB top; the wire rack's AABB crown is raised side-members, 12 cm above the rails.
    support_z_from_bottom: float | None = None
    #: For a hollow shelf, stage this far outside its front at shelf height and insert
    #: horizontally. Zero retains the ordinary vertical place used by top surfaces.
    insert_from_front: float = 0.0
    #: World-frame roll, then yaw, applied to the carried object before a front insertion. The
    #: final rotation is ``Rz(reorient_yaw) @ Rx(reorient_roll)`` composed with ``obj_quat``.
    reorient_roll: float = 0.0
    reorient_yaw: float = 0.0
    #: Translate destination waypoints by the measured object-to-finger attachment transform.
    #: This is useful for top-surface drops, but an incidental off-centre pinch must not rewrite
    #: the book's insertion path after its 90-degree wrist turn.  That path stays feed-forward and
    #: its stocked cubby provides the physical clearance for contact-transform variation.
    attachment_reaim: bool = True
    #: Body-frame downward pitch at the source. ``rotate_book`` applies the object reorientation
    #: on top of this same alignment, so the attached object's relative rotation stays exact.
    #: A full top-down 90 degrees is not reachable on Vega at the table: measured, it saturated
    #: L_arm_j3/j6, retained 93 mm planar error, and shoved the book 178 mm before closing.
    grasp_pitch: float = 0.0
    #: Optional object-local surface normal that must face the station's outward/front direction
    #: at success.  Containment plus uprightness alone cannot distinguish a sideways book.
    place_outward_axis: tuple | None = None
    #: Constant body-frame wrist yaw used for an optional side approach.
    grasp_yaw: float = 0.0
    #: Optional independent yaw for the right wrist.
    right_grasp_yaw: float | None = None
    #: Roll about the gripper's own forward / approach axis at the source and destination.  This
    #: leaves the tool approaching horizontally while turning its jaw axis from horizontal to
    #: vertical, which permits a top/bottom pinch on a thin object whose edge overhangs a table.
    grasp_tool_roll: float = 0.0
    #: Optional independent tool-axis roll for the right wrist.
    right_grasp_tool_roll: float | None = None
    #: Half the separation between two grasp points along ``bimanual_axis`` in the object frame.
    #: Zero is the ordinary single-left-arm plan.
    bimanual_half_span: float = 0.0
    #: Object-frame axis separating the two grasp points.  A folded towel is pinched at its two
    #: side edges (local +/-y) while both wrists stay in the robot's validated forward pose.
    bimanual_axis: tuple = (1.0, 0.0, 0.0)
    #: Where on the source table this object starts, overriding the shared `SRC_OBJ_XY` mark.
    #: That mark was placed for a 73 mm bowl and sits 115 mm in from the tabletop's near face
    #: (solid top measured at x = 1.085), so a LONG object centred there hangs over the edge --
    #: measured, a 300 mm towel overhung it by 35 mm, and a gripper closing on a towel whose end
    #: is over thin air tips it instead of holding it. Only the object moves; the park, the
    #: keepout and the whole approach are unchanged.
    obj_xy: tuple | None = None
    #: An object that RESTS ON the station and becomes the actual receptacle -- DatasetObject
    #: kwargs, plus `liner_z_from_bottom` for where its underside sits above the station's live
    #: AABB bottom. For a slatted station this is the difference between a task and a coin flip:
    #: the dish rack's bare rails support the bowl over ~30 mm lanes separated by a gap that
    #: swallows it, against a control stack whose end-effector floor is ~20 mm. Laying a rimmed
    #: tray across the rails turns that into one continuous surface -- swept, a bowl is supported
    #: AND `OnTop` the tray at 25 of 25 points over +/-50 mm x by +/-100 mm y -- and the rim also
    #: stops the post-release roll that moved the bowl 74 mm on the bare rails. The dataset has
    #: exactly ONE dish_rack model, so a better rack is not an option; a drying tray in a dish rack
    #: is the real-world answer anyway. `self.bowl` becomes the liner, so the place target, the
    #: support height and the success predicate all follow it.
    liner: dict | None = None
    liner_z_from_bottom: float | None = None
    #: The receptacle is SLATTED, so judge the placement by contact rather than by `OnTop`.
    #: `OnTop` is Touching AND vertical adjacency, and that adjacency casts ONE ray straight down
    #: from the object's centre (`compute_adjacencies(use_aabb_center=False)`). On an airer the ray
    #: either finds a rail or drops between two, so the predicate combs on and off with the rail
    #: pitch while the object never moves. Measured over a 120 mm sweep of the towel: `Touching` is
    #: true at every one of 13 positions and the drift is 0.0-0.4 mm, while `OnTop` alternates
    #: (-50 yes, -40/-30/-20 no, -10/0/+10/+20 yes, +30/+40 no, +50 yes). That is an artifact of
    #: the raycast, not a placement quality signal, so contact plus the AABB-bottom height guard
    #: `success` already applies is the honest test -- it accepts exactly the physically correct
    #: cases and still rejects a towel on the floor.
    slatted: bool = False


#: The pick-and-place tasks. WHERE each receptacle stands is `tasks.layout`, not here -- the room
#: is furnished once and identically for every task.
#:
#: Every carried object is sized so its SHORT horizontal axis fits the 2F-85's 85 mm opening while
#: lying flat in its natural pose, and is grasped at its own centre. The first cut tried to keep
#: full-size assets by taking a plate at the rim and standing a textbook on edge; both failed --
#: the book toppled before the hand arrived (a book on its edge is not a stable rest pose, so the
#: sim was right to drop it) and the rim pinch on a 110 mm plate closed on a chord rather than the
#: object. Shrinking to a side plate and a notebook is the honest fix: the grasp is then an
#: ordinary centre pinch of the kind the tabletop task already proved out.
SPECS = [
    # a bowl, not a plate: a 9 mm disc lying on the table cannot be taken at all, because the pads
    # would have to close through the tabletop to get either side of it.
    LHSpec(name="dish2rack",
           obj={"category": "bowl", "model": "ajzltc", "scale": [0.7, 0.7, 0.7]},   # 73 x 82 x 42
           # Static drop sweep: this model's centre gap lets the bowl fall through, and there is a
           # supporting lane on EITHER side of it. The far one (45 mm toward world +x) was used
           # first and is out of reach -- measured at the place pose, the left hand cannot pass
           # world x 1.764 because `L_arm_j4` pins at its +14 deg upper limit (URDF range
           # [-176, +14]) and beyond that the tracking error grows 1:1 with the command. The drop
           # it implies is 1.783, so the place was asking for 19 mm more arm than exists.
           #
           # It nonetheless passed for a long time, which is worth understanding because the reason
           # is an accident: `_calibrate_grasp_delta` aims the destination at the OBJECT's target
           # minus the measured object-to-finger offset, and that offset happened to be 15-19 mm in
           # the helpful direction, pulling the hand back to roughly where it could go. Improve the
           # grasp and the slack disappears -- with the offset at 1-4 mm the same place left 43-48
           # mm of EEF error and dropped the bowl through the gap. A place that only works when the
           # grasp is a bit off is not a working place.
           #
           # Both of those are moot now that a tray spans the rails (see `liner`): the receptacle
           # is the tray's own flat top, so there is no lane to hit and no gap to miss.
           # `place_offset` is back to zero and `support_z_from_bottom` is None, which makes
           # `_support_surface_z` the liner's AABB top -- measured 0.8132, with the bowl resting at
           # 0.793. The near lane the bare rack needed is kept in the note above because it is the
           # reason the tray is there at all.
           #
           # The tray sits at the rack's own tested rail height, 0.408 above its AABB bottom: the
           # number that used to place the BOWL now places the tray, and the bowl rides on top.
           station="dish_rack", on_top=True, release_dz=0.004,
           liner={"category": "tray", "model": "hbjdlb"},      # 177 x 390 x 42 mm, fits the deck
           liner_z_from_bottom=0.408),
    # A folded towel onto the clothes airer.  Asset survey: the old bath_towel/thmepr was the only
    # bath_towel model and needed [0.2212, 0.0999, 1.7921] to become a rigid 240 x 57 x 50 mm stick.
    # dishtowel/ltydgg is naturally 157 x 218 x 26 mm; the scale below preserves that folded-towel
    # thickness and gives a compact 220 x 261 x 31 mm object.  Both wrists remain in the
    # nominal forward approach direction while rolling about that direction by 90 degrees. Their
    # jaws then close vertically around the towel's thin overhanging front edge.  The grasp centres
    # stay 190 mm apart: close to the arms' natural 230 mm separation, but 20 mm in from each
    # rounded side edge. At 230 mm the right ray crossed the towel but only one pad contacted the
    # irregular folded corner; 120 mm, at the other extreme, produced 55-60 mm cross-body error.
    #
    # It deliberately remains rigid.  OmniGibson's assisted/sticky grasp candidate search accepts
    # only RigidDynamicPrim links, so a ClothPrim cannot attach to either 2F-85 without a separate
    # cloth-particle grasp mechanism.  Touching / OnTop can evaluate cloth, but that does not solve
    # the missing grasp.  `support_z_from_bottom` is the airer's deck, 0.635 above its AABB bottom.
    LHSpec(name="towel2rack",
           # A folded dishtowel with a near-native aspect ratio.  At this scale its live bbox is
           # 220 x 261 x 31 mm.  Its near edge overhangs the table collision by about 58 mm,
           # leaving room for the rolled grippers' lower pads while roughly three quarters of the
           # towel remains supported. At 28 mm only the upper pad touched and the lower pad's
           # table collision made closure shove the towel. This remains rigid because OmniGibson
           # assisted grasping only considers RigidDynamicPrim links (a ClothPrim needs a new
           # attachment mechanism, not a prim_type flag).
           obj={"category": "dishtowel", "model": "ltydgg",
                "scale": [1.40, 1.20, 1.20]},
           station="towel_rack", on_top=True, obj_xy=(1.14, 0.40), slatted=True,
           release_dz=0.045, grasp_offset=(-0.110, 0.0, 0.006),
           # The grasp point is 110 mm behind the object centre. Move the object that far along the
           # rack's local -y (world +x), so the HAND and chassis retain the verified x corridor;
           # the 220 mm towel still has over 100 mm of rack deck beyond either x edge.
           place_offset=(0.0, -0.110, 0.0), support_z_from_bottom=0.635,
           grasp_tool_roll=np.pi / 2,
           right_grasp_tool_roll=np.pi / 2,
           bimanual_half_span=0.095, bimanual_axis=(0.0, 1.0, 0.0)),
    # A closed dark hardback lying flat: 104 x 75 mm footprint, 58 mm tall. The non-uniform scale is
    # deliberate. Standing a full-size book on its edge is not a stable rest pose, and a full-size
    # flat book is wider than the jaws. This model replaces textbook/nloqia, which is an OPEN-book
    # mesh: its binding surface is normal to the broad face, so making that surface face outward
    # required a 120-degree compound wrist turn. Measured, that pinned L_arm_j7, swung the grasp
    # point down 33 cm, and dropped the book on the floor. ajpulw has a true closed-book spine on
    # local -y. Spawn yaw -90 degrees and the reachable -90-degree wrist roll together map its
    # local +x spine axis to world +z and its local -y spine face to the shelf's world -x outward
    # normal. The source wrist pitches 30 degrees downward. The cover-width scale is 0.65 rather
    # than 0.46: it does not change the 75 mm axis the jaws span, but gives the upright book a
    # 104 mm base in shelf depth so the 17 mm-outward release contact remains inside its support
    # polygon. (See
    # `layout.book_shelf`, where the geometry below was measured.) `support_z_from_bottom` is the
    # top row's
    # interior floor, 0.716 m above the case's live AABB bottom. `place_offset` is in the
    # station's own frame, and both of its terms matter: local +x 0.071 puts the wider book far
    # from the front face, deep enough that it is unambiguously inside the compartment and still
    # 0.14 m clear of the back panel; local -y 0.1984 picks the cubby COLUMN and centres the
    # carried book between its two filler books, because the case's
    # centre line is the vertical divider and a book released onto it slides off (measured drift
    # 0.06-0.24 m, against 0.001 m in either column). Which column costs the ARM nothing either
    # way -- the park is derived from the target, so the hand ends at its rest offset from the
    # base whichever one is chosen -- but it moves the CHASSIS 0.19 m, so take the northern one
    # and keep the park that much further off the south wall. The 35 mm release clearance keeps
    # the diagonal open linkage above the cubby floor; the flanking books guide the short final
    # gravity settle after the fingers open.
    LHSpec(name="book2shelf",
           obj={"category": "hardback", "model": "ajpulw", "scale": [0.29, 0.65, 1.25]},
           station="book_shelf", on_top=False, release_dz=0.035,
           obj_quat=(0.0, 0.0, -0.70710678, 0.70710678),
           place_offset=(0.071, -0.1984, 0.0),
           support_z_from_bottom=0.716, insert_from_front=0.20,
           reorient_roll=-np.pi / 2,
           grasp_pitch=np.deg2rad(30.0),
           attachment_reaim=False,
           place_outward_axis=(0.0, -1.0, 0.0)),
]

# Two slim, fixed books flank the insertion target in the same top-right cubby.  They are guides,
# not just decoration: the carried hardback is 58 mm thick after it is stood up, so their inner
# faces leave a 130 mm slot centred on the commanded release y=-1.628. A static drop sweep found
# that 57 mm wedges the 57.5 mm book above the cubby floor. Even the earlier 75 mm slot was too
# prescriptive: two valid assisted grasps at different source-table y positions produced measured
# attachment vectors [30, 4, -1] and [7, 0, -30] mm. Rotating those incidental vectors into a
# destination correction rewrote the insertion laterally by up to 30 mm, drove L_arm_j7 to its
# stop, and left 7--9 cm of EEF error. Keeping the insertion feed-forward reduced that error to
# 1--2 mm; the 130 mm lane then accepted the physically held book with 6 mm worst-side clearance
# in the reproduced low-source rollout. The carried book's 104 mm shelf-depth footprint supplies
# the tip resistance; the fillers stock the compartment and guide it without being a precision
# funnel.
#
# They also have to cover the whole settle corridor.  With their old 83 mm depth and world-x
# centre 1.603, their rear edge ended at x=1.645.  Quaternion logging showed the held book was
# correctly upright and outward-facing at release (both direction cosines 0.990), then it slid
# from x=1.583 to 1.662 -- beyond that edge -- and pitched 86 degrees before the episode ended.
# These near-native depth scales make both guides about 150 mm deep and centre them at x=1.643,
# spanning x=1.567..1.718: they overlap the book before detachment and continue to flank it while
# it drops onto the cubby floor. Their height and thickness are unchanged.
BOOK_FILLERS = (
    {"category": "notebook", "model": "aanuhi", "scale": [1.00, 0.65, 0.75],
     "local_xy": (0.060, -0.1231)},
    {"category": "notebook", "model": "djtnsj", "scale": [0.68, 0.59, 0.72],
     "local_xy": (0.060, -0.2737)},
)
BOOK_FILLER_QUAT = (0.70710678, 0.0, 0.0, 0.70710678)  # local +y points world +z


class LongHorizonPickPlace(MobilePickPlaceTask):
    """Pick off the source table, drive to a station, place. Configured by an `LHSpec`."""

    SPEC: LHSpec = None
    ARM = "left"
    MOBILE = True
    # The real box2cloth takes contain no commanded wz. This is whole_body_ik.py's supported
    # translation-only root mode: yaw is pinned as a QP equality, not hidden by teleporting rz.
    BASE_DOFS = "xy"
    # BASE_YAW_HOLD_IN_XY is deliberately NOT set here, so it inherits the shared
    # `vr_teleop.base_yaw_hold_in_xy: true`. It used to be forced False, on the reading that a
    # task whose demonstrations contain no commanded wz should never command wz. That conflates
    # two different things, and wbik.yaml says so on the line itself: with `base_dofs: "xy"` the
    # TELEOP yaw is still hard-locked in the QP -- what the flag enables is closed-loop odometry
    # trimming a small wz "to hold the robot's measured yaw against wheel/ground drift". It is a
    # regulator, not a commanded motion.
    #
    # Forcing it off was survivable only while the env fed `current_q` back into the QP: the
    # solver then saw the drifted body yaw and the arm quietly reached around it. With the sim
    # running the robot's own architecture (IK integrates its own configuration, the base loop
    # closes outside it) nothing sees the drift at all. Measured on dish2rack: the chassis reached
    # -0.047 rad (2.7 deg) against a pinned reference of 0, which over the 0.6 m place reach is
    # 28 mm of lateral error -- the bowl landed 25 mm off its lane and fell through the rack.
    # "assisted", not "sticky", and NOT "physical". All three create the same fixed joint or none;
    # they differ only in what has to be true first (OmniGibson robot.py):
    #   sticky   -- `touching_at_least_two_fingers = True` unconditionally (3165-3169). One pad
    #               brushing the object anywhere welds it.
    #   assisted -- >=2 FINGER links in contact AND the object intersects a ray cast between the
    #               gripper's AG start/end points, i.e. it really is between the pads (3128-3130).
    #   physical -- no joint at all; friction only (`_handle_assisted_grasping` is skipped, 832).
    #
    # On these three tasks sticky and assisted are indistinguishable in outcome -- all succeed,
    # attach deltas agree to 1 mm (bowl [0.010,-0.001,-0.006] vs [0.010,-0.001,-0.006], book
    # [0.008,0.005,-0.009] vs [0.007,0.005,-0.009]) -- because each object was already being
    # pinched by both pads. The difference is what happens when one is NOT, which is not
    # hypothetical: the towel shipped for a while at `grasp_offset` z = 0.050, a 3 mm graze on its
    # side, and sticky welded it, ran the fingers on to 0.785 rad (fully closed, a 1 mm gap) around
    # a 57 mm towel, and still reported success. Replayed, assisted REFUSES that grasp -- no joint,
    # towel left on the table -- and takes the corrected 0.020 pinch normally. It is the mode whose
    # trigger condition is the no-unreal-artifacts rule, so a bad grasp fails loudly instead of
    # passing as a magnetised carry.
    #
    # "physical" was measured and does not work, for a reason that is about the DRIVE rather than
    # the contact: the knuckle joints are position-servoed at stiffness 1e7 / damping 1e5 with a
    # 16.5 N.m effort cap (~330 N at the pads), and `grip_close` commands the fully-closed limit.
    # There is no equilibrium against a rigid object -- the pads reached 42.4 mm around the 57 mm
    # towel with both in contact, then kept ramping and extruded it; the fingers ended at 0.785 and
    # the towel never left the table (it slid 431 mm out of the hand frame). Making it work needs
    # either a width-aware close (which is just telling the robot the answer) or a force-limited
    # gripper drive; neither is a mode switch.
    GRASPING_MODE = "assisted"
    ROBOT_POS = layout.ROBOT_POS
    ROBOT_YAW = layout.ROBOT_YAW
    APPLE_XY = layout.SRC_OBJ_XY
    BASE_X_MAX = None            # superseded by keepout circles at both work surfaces
    BASE_MAX_LIN = BASE_MAX_LIN  # see the module constant: the chassis command is bang-bang
    SRC_KEEPOUT = layout.SRC_KEEPOUT
    SCENE_STATION_KEYS = layout.RECEPTACLE_KEYS
    #: heading is degenerate when the waypoint is right on top of the base; hold the last one
    HEADING_MIN_RANGE = 0.20
    #: `expert_step` calibrates this every tick, but only THIS class's `_build_segments` clears it
    #: per episode -- the non-prehensile subclasses override that method and so never created the
    #: attribute at all, crashing on their first tick. A class-level default makes the calibration
    #: an early-return for them (they never grasp `self.apple`) instead of an AttributeError.
    _grasp_delta = None
    #: Accumulated pick-side servo displacement for the current episode / cycle.
    _reach_delta = None
    _right_reach_delta = None
    _right_grasp_delta = None

    #: The settle waypoint is a feedback target, but it is still a smooth trajectory: move either
    #: hand's endpoint by at most 0.10 m/s and stop integrating once the measured residual is 3 mm.
    #: This gives a 200-tick settle enough authority to close the observed 81 mm chained residual
    #: without putting an 81 mm discontinuity into one frame.
    REACH_SERVO_SPEED = 0.10
    REACH_SERVO_TOL = 0.003
    REACH_SERVO_MAX = 0.12
    REACH_SERVO_AT = 0.45
    REACH_SERVO_MARGIN = 0.005

    # ---- scene ----
    @property
    def station(self):
        return STATIONS[self.SPEC.station]

    @property
    def scene_station_keys(self):
        return self.SCENE_STATION_KEYS

    def _is_bimanual(self, spec=None):
        return float((self.SPEC if spec is None else spec).bimanual_half_span) > 0.0

    @staticmethod
    def _spawn_R(spec):
        from scipy.spatial.transform import Rotation as R

        return R.from_quat(list(spec.obj_quat)).as_matrix()

    @staticmethod
    def _reorient_R(spec):
        c, s = np.cos(float(spec.reorient_roll)), np.sin(float(spec.reorient_roll))
        roll = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        return rot_z(float(spec.reorient_yaw)) @ roll

    @staticmethod
    def _has_reorientation(spec):
        return (abs(float(spec.reorient_roll)) > 1e-8
                or abs(float(spec.reorient_yaw)) > 1e-8)

    def _grasp_world_offset(self, arm="left", placed=False):
        """Object-centre to commanded finger point, respecting the object's orientation."""
        spec = self.SPEC
        local = np.asarray(spec.grasp_offset, dtype=float).copy()
        if self._is_bimanual(spec):
            axis = np.asarray(spec.bimanual_axis, dtype=float)
            axis /= max(float(np.linalg.norm(axis)), 1e-9)
            local += (1.0 if arm == "left" else -1.0) * float(spec.bimanual_half_span) * axis
        Robj = self._spawn_R(spec)
        if placed:
            Robj = self._reorient_R(spec) @ Robj
        return Robj @ local

    def _placed_attachment_delta(self, delta):
        """Rotate a measured object-to-finger vector with the in-air book reorientation."""
        return self._reorient_R(self.SPEC) @ np.asarray(delta, dtype=float)

    def _grasp_alignment(self, arm="left"):
        """Body-frame adjustment that composes optional yaw with local tool-axis roll."""
        spec = self.SPEC
        if arm == "left":
            yaw = float(spec.grasp_yaw)
            roll = float(spec.grasp_tool_roll)
            base_R = getattr(self, "_Rgrasp_base", None)
        else:
            yaw = float(spec.grasp_yaw if spec.right_grasp_yaw is None
                        else spec.right_grasp_yaw)
            roll = float(spec.grasp_tool_roll if spec.right_grasp_tool_roll is None
                         else spec.right_grasp_tool_roll)
            base_R = getattr(self, "_Rgrasp_right_base", None)
        alignment = rot_y(float(spec.grasp_pitch)) @ rot_z(yaw)
        if abs(roll) <= 1e-12:
            return alignment
        if base_R is None:
            raise RuntimeError("tool-axis grasp roll requires the reset wrist orientation")
        return alignment @ base_R @ rot_z(roll) @ base_R.T

    def _right_hand_start(self):
        return getattr(self, "_right_start_center", self._start_center)

    def object_configs(self):
        """The carried object plus the three wall receptacles (no fridge or chair).

        Load the fixed receptacles above the room, then move them onto their marks in ``reset``.
        At their final poses the dish rack overlaps an original dining chair; OmniGibson advances
        physics while constructing / restoring the scene, before ``reset`` can tuck that chair
        away.  The collision launched the chair into the non-fixed source table, so every episode
        began with the table already displaced and tilted and the bowl rolled off before grasp.
        """
        stations = layout.station_configs(self.scene_station_keys)
        for cfg in stations:
            assert cfg["fixed_base"]
            cfg["position"][2] += 3.0
        return ([{"type": "DatasetObject", "name": "lh_obj", **self.SPEC.obj,
                  "position": [*self.APPLE_XY, 0.90]}]
                + self._liner_configs() + self._book_filler_configs() + stations)

    #: Name of the liner instance for a spec, or None. Keyed by station so a chained rollout that
    #: serves several stations gets one liner each rather than a shared handle.
    @staticmethod
    def _liner_name(spec):
        return None if spec.liner is None else f"lh_liner_{spec.station}"

    def _liner_configs(self, specs=None):
        """Spawn each liner high, like the stations; `reset` drops it onto its rails."""
        out = []
        for spec in (specs if specs is not None else [self.SPEC]):
            if spec.liner is None:
                continue
            out.append({"type": "DatasetObject", "name": self._liner_name(spec), **spec.liner,
                        "position": [*STATIONS[spec.station].xy, 3.9]})
        return out

    def _place_liner(self, env, spec):
        """Settle the liner on its station's support rails, at the reset pose the spec names."""
        if spec.liner is None:
            return
        obj = env.env.scene.object_registry("name", self._liner_name(spec))
        if obj is None:
            raise ValueError(f"{self.name}: liner {self._liner_name(spec)!r} not in the scene")
        if spec.liner_z_from_bottom is None:
            raise ValueError(f"{self.name}: spec.liner needs liner_z_from_bottom")
        station = env.env.scene.object_registry("name", STATIONS[spec.station].name)
        rails = float(station.aabb[0][2]) + float(spec.liner_z_from_bottom)
        obj.set_position_orientation(
            position=[*STATIONS[spec.station].xy, rails + float(obj.aabb_extent[2]) / 2 + 0.004],
            orientation=[0.0, 0.0, 0.0, 1.0])
        obj.keep_still()

    @staticmethod
    def _book_filler_name(i):
        return f"lh_book_filler_{i}"

    def _book_filler_configs(self):
        """Spawn stocked-cubby books high; reset places them after the case is on its mark."""
        return [
            {"type": "DatasetObject", "name": self._book_filler_name(i),
             "category": cfg["category"], "model": cfg["model"], "scale": cfg["scale"],
             "fixed_base": True, "position": [*STATIONS["book_shelf"].xy, 3.9]}
            for i, cfg in enumerate(BOOK_FILLERS)
        ]

    def _place_book_fillers(self, env):
        """Place two upright books beside, but not across, the carried book's insertion lane."""
        from scipy.spatial.transform import Rotation as R

        station = env.env.scene.object_registry("name", STATIONS["book_shelf"].name)
        if station is None:
            raise ValueError(f"{self.name}: stocked bookcase is not in the scene")
        floor = float(station.aabb[0][2]) + float(SPEC_BY_NAME["book2shelf"].support_z_from_bottom)
        Rstation = rot_z(float(STATIONS["book_shelf"].yaw))
        Rfill = R.from_quat(BOOK_FILLER_QUAT).as_matrix()
        for i, cfg in enumerate(BOOK_FILLERS):
            obj = env.env.scene.object_registry("name", self._book_filler_name(i))
            if obj is None:
                raise ValueError(f"{self.name}: filler {self._book_filler_name(i)!r} not in scene")
            local = np.array([*cfg["local_xy"], 0.0])
            xy = np.asarray(STATIONS["book_shelf"].xy) + (Rstation @ local)[:2]
            ext = layout.bbox_size(cfg["category"], cfg["model"], cfg["scale"])
            half_height = float((np.abs(Rfill) @ ext)[2]) / 2.0
            obj.set_position_orientation(
                position=[xy[0], xy[1], floor + half_height + 0.003],
                orientation=list(BOOK_FILLER_QUAT))
            obj.keep_still()

    def bind(self, env):
        objs = env.env.scene.objects
        reg = env.env.scene.object_registry

        def nearest(category, xy):
            cands = [o for o in objs if o.category == category]
            if not cands:
                raise ValueError(f"{self.name}: scene has no {category!r}")
            return min(cands, key=lambda o: np.linalg.norm(env.obj_pos(o)[:2] - np.array(xy)))

        self.table = nearest("breakfast_table", layout.SRC_TABLE_XY)
        self._chairs = layout.side_chairs(env)
        self.apple = reg("name", "lh_obj")          # `apple`/`bowl` are the base class's handles
        # `bowl` is the RECEPTACLE the place aims at and success is judged against -- the liner
        # where a spec has one, otherwise the station itself.
        name = self._liner_name(self.SPEC) or self.station.name
        self.bowl = reg("name", name)
        if self.bowl is None:
            raise ValueError(f"{self.name}: receptacle {name!r} not in the scene")

    def declutter(self, env):
        """Delete objects the tasks never touch. Called once, before the first reset.

        Each footprint is recorded BEFORE removal so the route planner can treat that floor as
        open -- the shipped traversability map still shows the object standing there.
        """
        self._removed_boxes = []
        gone = [o for o in list(env.env.scene.objects) if o.category in DECLUTTER]
        for obj in gone:
            try:
                lo, hi = obj.aabb
                lo = lo.detach().cpu().numpy() if hasattr(lo, "detach") else np.asarray(lo)
                hi = hi.detach().cpu().numpy() if hasattr(hi, "detach") else np.asarray(hi)
                self._removed_boxes.append((float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])))
                env.env.scene.remove_object(obj)
            except Exception as exc:                  # keep going; clutter is cosmetic
                print(f"[{self.name}] could not remove {obj.name}: {exc}", flush=True)
        if gone and not getattr(self, "_declutter_logged", False):
            self._declutter_logged = True
            print(f"[{self.name}] decluttered {len(gone)} objects: "
                  f"{sorted({o.category for o in gone})}", flush=True)
        left = sorted({o.category for o in env.env.scene.objects} & set(DECLUTTER))
        if left:      # the reset restores the initial object set, so verify it actually took
            print(f"[{self.name}] WARNING declutter left {left} in the scene", flush=True)

    def reset(self, env):
        spec = self.SPEC
        layout.place_stations(env, self.scene_station_keys)
        self._place_liner(env, spec)                 # ...after the station it rests on is on its mark
        self._place_book_fillers(env)
        layout.place_side_chairs(env, self._chairs)  # ...including the two chairs in the way
        self.table.keep_still()
        self.ztop = float(self.table.aabb[1][2])
        r = self.rng
        ax = self.APPLE_XY[0] + r.uniform(-0.02, 0.02)
        ay = self.APPLE_XY[1] + r.uniform(-0.02, 0.02)
        # Standing height computed ANALYTICALLY from the rotated bounding box, in one shot. The
        # first version dropped the object, stepped physics, then re-placed it using the AABB it
        # had after that step -- but a book on edge has already toppled by then, so it was re-placed
        # at its FLAT height while still commanded upright, and simply fell over again.
        az = self.ztop + self._upright_half_height(spec) + 0.004
        self.apple.set_position_orientation(position=[ax, ay, az],
                                            orientation=list(spec.obj_quat))
        self.apple.keep_still()

    @staticmethod
    def _upright_half_height(spec):
        """Half the object's vertical extent in its SPAWN orientation, from the asset's own bbox."""
        from scipy.spatial.transform import Rotation as R
        ext = layout.bbox_size(spec.obj["category"], spec.obj["model"],
                               spec.obj.get("scale"))
        rot = np.abs(R.from_quat(list(spec.obj_quat)).as_matrix())
        return float((rot @ ext)[2]) / 2.0

    def keepouts(self):
        """(x, y, r) circles the chassis must stay out of -- see `VegaOGEnv.base_keepouts`.

        With PLACE_STANDOFF leading the head command INSIDE the destination circle, this radius is
        what actually sets the working distance: the base drives in until the circle cancels its
        inward motion and then sits there. That is stable rather than a limit cycle, because only
        the inward radial component is removed -- nothing pushes back out. So each station's
        `keepout` is the working distance that receptacle wants: far enough that the chassis clears
        its footprint, near enough that the arm still reaches over it.

        Only the SOURCE table and the station being worked are listed. The rest of the layout is
        avoided by the route planner instead (`NavMap(block_boxes=...)`), because a circle big
        enough to contain a 1.07 m bookcase would also swallow the neighbouring station's park.
        """
        tp = layout.SRC_TABLE_XY
        return [(tp[0], tp[1], self.SRC_KEEPOUT),
                (self.station.xy[0], self.station.xy[1], self.station.keepout)] \
            + layout.keepouts(target=self.SPEC.station, keys=self.scene_station_keys)

    # ---- success ----
    def _at_target_support_height(self, tolerance=0.08):
        """Whether the object's lower face is at this spec's intended shelf / support."""
        bottom = float(self.apple.aabb[0][2])
        support = self._support_surface_z()
        return abs(bottom - support) <= float(tolerance)

    def success(self, env):
        from omnigibson.object_states import Inside, OnTop, Touching
        required_arms = ("left", "right") if self._is_bimanual() else (self.ARM,)
        if any(env.is_grasping(arm) is not None for arm in required_arms):
            return False        # still holding it: nothing has been placed
        grasped = getattr(self, "_ever_grasped", {})
        if not all(bool(grasped.get(arm, False)) for arm in required_arms):
            print(f"[{self.name}] missing required grasp(s): {grasped}", flush=True)
            return False
        try:
            if not self.SPEC.on_top:
                placed = bool(self.apple.states[Inside].get_value(self.bowl))
                if placed and self.SPEC.support_z_from_bottom is not None:
                    # Inside(bookcase) is true in every compartment. It therefore accepted a
                    # rollout where the carried book fell into the lower cubby at z=0.427 even
                    # though the scripted top-cubby floor is roughly 0.33 m higher. Check the
                    # object's actual lower face against that specific floor, while allowing the
                    # 35 mm scripted release clearance and a short gravity settle.
                    if not self._at_target_support_height():
                        bottom = float(self.apple.aabb[0][2])
                        support = self._support_surface_z()
                        print(f"[{self.name}] object is inside the receptacle but in the wrong "
                              f"compartment: bottom={bottom:.3f}, target floor={support:.3f}",
                              flush=True)
                        placed = False
            else:
                # A slatted receptacle is judged by contact, not by OnTop's single downward ray --
                # see `LHSpec.slatted` for the sweep that shows the ray combing on and off the rails
                # while the object sits dead still. The height guard below does the rest of the work.
                resting = Touching if self.SPEC.slatted else OnTop
                placed = bool(self.apple.states[resting].get_value(self.bowl))
                if placed:
                    # OnTop only requires touching plus vertical adjacency. On the wire rack a bowl
                    # that has fallen through can rest on the floor while touching a low rail.
                    placed = float(self.apple.aabb[0][2]) > self._support_surface_z() - 0.06
            if not placed:
                return False
            if self._has_reorientation(self.SPEC):
                q = self.apple.get_position_orientation()[1]
                q = q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)
                from scipy.spatial.transform import Rotation as R
                # Object orientation composes the commanded wrist reorientation with the
                # object's non-identity spawn attitude.  Looking only at the wrist rotation
                # checks the wrong local axis for books spawned with a tabletop yaw.
                intended = self._reorient_R(self.SPEC) @ self._spawn_R(self.SPEC)
                spine_axis = int(np.argmax(np.abs(intended[2])))
                spine_sign = float(np.sign(intended[2, spine_axis]))
                world_spine = spine_sign * R.from_quat(q).as_matrix()[:, spine_axis]
                if float(world_spine[2]) < np.cos(np.deg2rad(25.0)):
                    print(f"[{self.name}] book is contained but not upright: "
                          f"spine-up dot={world_spine[2]:.3f}", flush=True)
                    return False
                outward_axis = self.SPEC.place_outward_axis
                if outward_axis is not None:
                    outward_axis = np.asarray(outward_axis, dtype=float)
                    outward_axis /= max(float(np.linalg.norm(outward_axis)), 1e-9)
                    actual_R = R.from_quat(q).as_matrix()
                    front = rot_z(float(self.station.yaw)) @ np.array([1.0, 0.0, 0.0])
                    outward_dot = float((actual_R @ outward_axis) @ front)
                    if outward_dot < np.cos(np.deg2rad(25.0)):
                        print(f"[{self.name}] book is upright but its spine faces sideways: "
                              f"outward dot={outward_dot:.3f}", flush=True)
                        return False
            return True
        except Exception as exc:
            # Deliberately NOT a distance fallback. Proximity of two centres establishes neither
            # contact, support, containment nor release, so a state-evaluation fault would have
            # been silently converted into a success. Fail loudly instead. (Codex review.)
            raise RuntimeError(f"{self.name}: could not evaluate placement") from exc

    #: Rate the final 8 cm onto the object is flown at, m/s. The inherited flat APPROACH_TICKS=40
    #: made it 0.200 m/s, against the ~0.016 m/s the `pregrasp` leg descends at while it drives in
    #: -- an 11x jump partway down, and the visible "slow, then suddenly fast" of the review video.
    #: Removing the step outright needs the drive and the descent to be separate legs, which is
    #: measurably worse (see `_build_segments`), so what is left is to close the gap from the other
    #: end. 0.074 m/s is the slower rate of the two the 2x2 grasped at, so it is a tested value
    #: rather than a guess, and it takes the step from 11x to 4x for 0.7 s more per pick.
    GRASP_DESCENT_SPEED = 0.074
    #: Ticks held at the grasp point, fingers open, before closing -- 2.0 s against the 0.5 s
    #: inherited from the tabletop task. Derived from the measured step response of the aligned
    #: loop, not from a time constant: holding the settle for 8 s and logging |finger grasp point -
    #: commanded centre| gives
    #:     ticks into settle   +0    +50   +100   +150   +200   +400   +800
    #:     hand error (mm)    34.7   35.2  26.2   20.0   19.8   19.0   17.3
    #: i.e. the knee is at ~160 ticks and everything after 200 is worth ~1 mm/s. 200 buys the
    #: convergence there is to buy; 800 would buy 2 mm more for 6 s an episode.
    SETTLE_TICKS = 200
    # ---- where a pick STARTS from ----
    # Two quantities `_build_segments` needs that are constants for a single pick and variables for
    # a chained one, so they are named rather than inlined. `LongHorizonSequence` picks three times
    # in one rollout: only its FIRST pick begins at the spawn pose with the base back from the
    # table, and only its first pick reaches the table from `SRC_PARK`. The later two begin parked
    # at whichever station was just served, with the hand up at carry height.
    def _src_park(self):
        """Base pose the drive to the receptacle starts from -- i.e. where this pick happens."""
        return SRC_PARK

    def _hand_start(self):
        """Where the hand is when this pick's `pregrasp` leg begins. Only sizes that leg."""
        return self._start_center

    # ---- expert ----
    def _build_segments(self, env):
        """Grasp at the source table -> back, right-strafe, forward -> place without yaw.

        One flat waypoint list, mapped to L_ee by the same `_grasp_target` as every other tick.
        """
        # Cleared per episode. `collect_demos` reuses ONE env and task across episodes.
        self._last_head_cmd = None
        self._last_base_cmd = None
        self._last_left_cmd = None
        self._right_segs = None
        self._right_grasp_rotations = None
        self._right_segment_starts = None
        self._grasp_delta = None
        self._reach_delta = None
        self._right_grasp_delta = None
        self._right_reach_delta = None
        self._orient_left_eef = None
        self._orient_right_eef = None
        self._orient_head_center = None
        self._reach_servo_updates = {"left": 0, "right": 0}
        self._reach_servo_started = {"left": False, "right": False}
        self._reach_targets = {}
        self._reach_limits = {}
        self._reach_object0 = None
        self._reach_servo_logged = False
        self._second_hand_reaimed = False
        self._ever_grasped = {"left": False, "right": not self._is_bimanual()}
        go, gc = env.grip_open, env.grip_close
        obj0 = env.obj_pos(self.apple).copy()
        if not hasattr(self, "_right_start_center"):
            self._right_start_center = np.asarray(env.finger_grasp_point("right"), dtype=float)
        if not hasattr(self, "_head_pos0"):
            self._head_pos0 = env.link_pose("zed_depth_frame")[:3, 3].copy()
        a = obj0 + self._grasp_world_offset("left")
        right_a = obj0 + self._grasp_world_offset("right") if self._is_bimanual() else None
        self._obj0 = a
        drop = self._release_point(env)
        right_drop = (self._release_point(env, arm="right")
                      if self._is_bimanual() else None)

        base0 = env.link_pose("base")[:3, 3]
        # hand offset from the base at the reset pose, in the BASE frame: a transit waypoint is
        # "the hand, held where it rests, at this point on the route", so driving the waypoint
        # along the route is a base command that the arm can actually hold. Carrying at the rest
        # height (rather than an arbitrary clearance) also keeps the derived head command at
        # exactly the standing height -- the transit command is then the reset pose, translated.
        if self._is_bimanual():
            start_pair = (self._start_center + self._right_start_center) / 2.0
            carry_local = rot_z(-self.ROBOT_YAW) @ (start_pair - base0)
            self._head_off_bimanual_base = (
                rot_z(-self.ROBOT_YAW) @ (self._head_pos0 - start_pair)
            )
        else:
            carry_local = rot_z(-self.ROBOT_YAW) @ (self._start_center - base0)
        # Carry height: high enough to clear what the hand passes over, and no higher.
        #
        # The old floor was the hand's own REST height, which is not a clearance requirement at
        # all -- it was there so the transit command would be the reset pose, translated -- and on
        # a task that does not arrive over its receptacle that floor is what welded L_arm_j4 to
        # its +14 deg stop for the whole carry. Measured on book2shelf: carrying at the rest
        # height 0.963 pins the elbow on 98 % of lift+transit ticks and folds torso_j2 to 139 deg;
        # carrying at 0.82 never reaches the stop (elbow mean -42.6, max +9.5) and leaves the
        # torso at 89 deg, its nominal. It is the hand HEIGHT specifically -- the base stand-off
        # does nothing, 0.70 and 0.85 both still pin the elbow on 98 % of ticks -- and the
        # threshold is sharp, 0.86 still being 90 % pinned. So the clearances are taken
        # explicitly, and the rest height is kept only where it is doing real work:
        carry_z = max(float(drop[2]) + 0.06,                  # stay above the place itself
                      float(a[2]) + 0.03)                     # lift clear of the source surface
        if not self.SPEC.insert_from_front:
            # A top-surface place ARRIVES over the receptacle, so the carried object sweeps
            # through the space above it and the rest height is the validated clearance. Deriving
            # it from the receptacle's AABB crown instead was tried and is NOT equivalent: 3 cm
            # over the dish rack's crown is 0.934, only 3 cm below the rest height, and the bowl
            # still ended up at z 0.402 inside the frame -- an AABB top is a poor proxy for a
            # wire rack whose crown is two thin raised side members.
            carry_z = max(carry_z, float(self._start_center[2]))
        # The robot works the station facing the heading the LAYOUT assigns it -- chosen (and
        # verified against the eroded traversability map) so that the base pose it implies is
        # somewhere the chassis actually fits. `park` is where the base must END UP, not where a
        # human would say it parks: the head command is `work_point + rest_stand_off`, and that
        # stand-off has a 0.34 m lateral term, so the pose the place implies sits ~0.4 m beyond a
        # naive park. It must NOT carry the PLACE_STANDOFF lead-in either: the transit leg's own
        # endpoint is `route_end + rest_hand_offset`, so leading the route in pushes that hand the
        # same distance PAST the receptacle -- measured, 0.28 m past the dish rack and into the
        # table behind it, which drove the arm through the tabletop and collapsed the whole body.
        # The lead-in is applied by the HEAD COMMAND after arrival instead, where the keepout
        # circle bounds it and no arm waypoint moves.
        place_heading = float(self.station.approach)
        # Where the carried hand has to END the drive. Over a top surface that is the receptacle
        # itself -- the hand rides above it and comes down. Into an ENCLOSED case it emphatically
        # is not: the case is 1.06 m tall, the carry height is 0.96 m, and a route that ends over
        # the station centre drives the held book through the front panel at transit speed. So a
        # containment task drives only as far as its own staging point, which the insertion then
        # starts from. Pulling the route endpoint BACK is the safe direction -- the warning above
        # is about leading it IN.
        front = (rot_z(float(self.station.yaw)) @ np.array([1.0, 0.0, 0.0]))[:3]
        stage = drop + self.SPEC.insert_from_front * front
        right_stage = (right_drop + self.SPEC.insert_from_front * front
                       if right_drop is not None else None)
        if self._is_bimanual():
            goal = ((stage + right_stage) / 2.0)[:2]
        else:
            goal = stage[:2] if self.SPEC.insert_from_front else np.asarray(self.station.xy, float)
        park = goal - (rot_z(place_heading) @ carry_local)[:2]
        #: Published for `LongHorizonSequence`, which has to route the NEXT pick's return drive
        #: from wherever this place left the base.
        self._dest_park = park

        hand_off = rot_z(place_heading) @ carry_local
        if self._is_bimanual():
            span = (self._grasp_world_offset("left", placed=True)
                    - self._grasp_world_offset("right", placed=True)) / 2.0
            nav_hand_off = np.stack([hand_off + span, hand_off - span])
        else:
            span = None
            nav_hand_off = hand_off
        nav = self._nav(env, hand_off=nav_hand_off, hand_z=carry_z)
        if abs(place_heading - float(self.ROBOT_YAW)) > 1e-8:
            raise ValueError(
                f"{self.name}: wall-station heading {place_heading} would rotate the base; "
                f"expected spawn heading {self.ROBOT_YAW}"
            )
        src_park = ((a + right_a) / 2.0)[:2] - (rot_z(place_heading) @ carry_local)[:2] \
            if self._is_bimanual() else self._src_park()
        route = _box2cloth_route(nav, src_park, park)
        legs, right_legs = [], []
        prev = np.asarray(route[0], dtype=float)
        for pt in route[1:]:
            dist = float(np.linalg.norm(pt - prev))
            control = np.append(pt[:2], carry_z) + rot_z(place_heading) @ carry_local
            control[2] = carry_z
            hand = control + span if span is not None else control
            ticks = max(40, int(round(dist / TRANSIT_SPEED * env.action_hz)))
            name = f"transit{len(legs)}"
            legs.append((name, hand, gc, ticks))
            if span is not None:
                right_legs.append((name, control - span, gc, ticks))
            prev = pt
        lift = np.array([a[0], a[1], carry_z])
        right_lift = (np.array([right_a[0], right_a[1], carry_z])
                      if right_a is not None else None)
        # Top-surface tasks arrive over the receptacle and descend. A containment task has already
        # driven to its staging point outside the front (see `goal`), so `arrive` only settles
        # there at carry height, `carry` descends to the compartment's own height, and `place`
        # inserts horizontally through the opening -- the gripper never passes under the roof.
        if self.SPEC.insert_from_front:
            arrive = np.array([stage[0], stage[1], carry_z])
            carry = stage
            retreat = stage
            right_arrive = (np.array([right_stage[0], right_stage[1], carry_z])
                            if right_stage is not None else None)
            right_carry = right_stage
            right_retreat = right_stage
            # The 20 cm insertion asks the mobile base and arm to advance together.  At 260 ticks
            # (77 mm/s) the commanded book centre reached 1.617 m but the measured centre reached
            # only 1.528 m, leaving it on the compartment lip and in front of both filler books.
            # Fly the same straight segment at 40 mm/s; this changes neither its endpoint nor its
            # piecewise-linear velocity, and avoids hiding the lag with an end-of-segment pause.
            carry_ticks = 260
            place_ticks = max(
                260,
                int(round(float(np.linalg.norm(drop - stage)) / 0.040 * env.action_hz)),
            )
            retreat_ticks = 100
        else:
            arrive = np.array([drop[0], drop[1], carry_z])
            carry = drop + np.array([0.0, 0.0, 0.10])
            retreat = drop + np.array([0.0, 0.0, 0.12])
            right_arrive = (np.array([right_drop[0], right_drop[1], carry_z])
                            if right_drop is not None else None)
            right_carry = (right_drop + np.array([0.0, 0.0, 0.10])
                           if right_drop is not None else None)
            right_retreat = (right_drop + np.array([0.0, 0.0, 0.12])
                             if right_drop is not None else None)
            carry_ticks, place_ticks = 160, 240
            # Two assisted-grasp joints clear later than one. In the chained rollout the second
            # towel joint disappeared on the old 30-tick retreat's final tick, so recording ended
            # before either the free towel or its settling motion was visible.
            retreat_ticks = 100 if self._is_bimanual() else 30
        # The approach to the source table is a 0.9 m DRIVE, so it has to be timed off the same
        # speed as the transit legs. At the inherited flat 300 ticks it ran at 0.30 m/s, well above
        # the follower's own cap, and the base would simply have arrived at the table late -- with
        # the fingers already commanded onto the book.
        #
        # The 8 cm hover is load-bearing and stays. This leg is a 0.9 m DRIVE that also descends,
        # which gives the approach two very different speeds -- measured 0.016 m/s for the first
        # 55 % of the drop and 0.200 m/s for the rest, an 11x step, and exactly the "moves slowly,
        # then descends fast" the review video shows. The clean repair is to separate them: fly
        # level at rest height and let `approach` own the whole 18 cm in one segment at one speed.
        # It was implemented and REJECTED on measurement. What makes the shipped leg robust is not
        # the hover height as such but that the hand converges LATERALLY while it descends
        # diagonally: entering the last 8 cm it is ~3 mm off the mark, where a level fly-over hands
        # the descent ~10 mm of error because the base is still catching up at the end of a 0.9 m
        # drive. Coming down 18 cm off-centre clips the 42 mm bowl and shoves it -- measured 41 mm,
        # after which the fingers shut to 0.785 rad on nothing and `assisted` rightly refuses. A
        # 2x2 over hover height x descent duration pinned it on the hover (both durations grasp
        # from 8 cm, neither does from rest height), and adding a dwell over the object to let the
        # error settle first only made it intermittent -- 3 of 6 trials grasped, one shoving the
        # bowl 74 mm. The step is a cosmetic fault; losing the bowl is not.
        # What IS safe is narrowing the step, see `GRASP_DESCENT_SPEED` on `approach` below.
        hand0 = np.asarray(self._hand_start(), dtype=float)
        pregrasp = a + np.array([0.0, 0.0, 0.08])
        # SQUARE UP IN Y, THEN DRIVE IN X -- the same treatment `_manhattan` and
        # `_box2cloth_route` already give the transit, which the approach never got. The chassis
        # arbitrates: `project_single_axis` keeps one of vx / vy at a time, so a waypoint that is
        # diagonal in the plane makes the projection alternate between the axes at loop rate, and
        # the base crabs toward the table instead of driving at it. Timing the leg for the longer
        # path (L1 rather than the straight line) fixed only the CLOCK -- the base still had to
        # discover the two moves for itself. Measured on the chained rollout's book, whose mark is
        # 0.45 m to the right of the spawn: the single diagonal leg left the base 0.23 m short in y
        # when the fingers were commanded shut, and the arm reached across to cover it. Two
        # explicit legs make the ordering the plan's, not the projection's.
        #
        # Each leg is still timed off its own L1 distance, so the descent that rides along with the
        # x leg is paid for in that leg's clock. Height is HELD through the strafe and given
        # entirely to the x leg, which keeps the property the grasp actually depends on: the hand
        # converges laterally while it descends diagonally, arriving at the 8 cm hover ~3 mm off
        # the mark (see the note above on why a level fly-over is worse).
        #
        # A leg shorter than `MIN_LEG` is dropped, exactly as `_manhattan` drops one: the single
        # tasks pick 14 mm off the spawn's y and the chained rollout's later cycles arrive with the
        # hand already over the object, so both keep the one x leg they have always had, and only a
        # genuinely offset mark earns the strafe.
        legs_in, right_legs_in = [], []
        right_hand0 = np.asarray(self._right_hand_start(), dtype=float)
        right_pregrasp = (right_a + np.array([0.0, 0.0, 0.08])
                          if right_a is not None else None)
        left_alignment = self._grasp_alignment("left")
        right_alignment = self._grasp_alignment("right")
        needs_orient = right_pregrasp is not None and (
            np.linalg.norm(left_alignment - np.eye(3)) > 1e-8
            or np.linalg.norm(right_alignment - np.eye(3)) > 1e-8
        )
        if needs_orient:
            # Rotate at fixed EEF positions. Keeping the FINGER centre fixed while its 129 mm
            # EEF-to-fingertip offset turned sideways pushed both EEFs 129 mm forward, loading the
            # torso until PhysX held it 1.19 rad away from the WBIK posture. During this segment
            # `_smooth_center` follows the exact circular fingertip arc implied by the slerped
            # wrist, so the EEF positions and head/base command remain fixed.
            left_final_R = rot_z(self.ROBOT_YAW) @ left_alignment @ self._Rgrasp_base
            right_final_R = rot_z(self.ROBOT_YAW) @ right_alignment @ self._Rgrasp_right_base
            # A sequence builds future cycles up front, so the robot's measured EEF poses here
            # still belong to episode reset. Derive the fixed EEFs from this cycle's planned
            # incoming fingertip centres and natural wrist rotations instead. At f=0 this makes
            # the circular fingertip path exactly continuous with the preceding waypoint.
            left_initial_R = rot_z(self.ROBOT_YAW) @ self._Rgrasp_base
            right_initial_R = rot_z(self.ROBOT_YAW) @ self._Rgrasp_right_base
            self._orient_left_eef = hand0 - left_initial_R @ self._grasp_off_local
            self._orient_right_eef = (
                right_hand0 - right_initial_R @ self._right_grasp_off_local
            )
            left_oriented = self._orient_left_eef + left_final_R @ self._grasp_off_local
            right_oriented = (
                self._orient_right_eef + right_final_R @ self._right_grasp_off_local
            )
            self._orient_head_center = (hand0 + right_hand0) / 2.0
            legs_in.append(("orient", left_oriented, go, 180))
            right_legs_in.append(("orient", right_oriented, go, 180))
        left_approach0 = legs_in[-1][1] if legs_in else hand0
        right_approach0 = right_legs_in[-1][1] if right_legs_in else right_hand0
        y_travel = abs(float(pregrasp[1] - left_approach0[1]))
        if right_pregrasp is not None:
            y_travel = max(y_travel, abs(float(right_pregrasp[1] - right_approach0[1])))
        if y_travel >= MIN_LEG:
            square_up = np.array([left_approach0[0], pregrasp[1], left_approach0[2]])
            square_dist = float(np.abs(square_up - left_approach0).sum())
            if right_pregrasp is not None:
                right_square = np.array([
                    right_approach0[0], right_pregrasp[1], right_approach0[2]
                ])
                square_dist = max(square_dist,
                                  float(np.abs(right_square - right_approach0).sum()))
            ticks = max(40, int(round(square_dist / TRANSIT_SPEED * env.action_hz)))
            legs_in.append(("pregrasp_y", square_up, go, ticks))
            if right_pregrasp is not None:
                right_legs_in.append(("pregrasp_y", right_square, go, ticks))
        drive = pregrasp - (legs_in[-1][1] if legs_in else hand0)
        drive_dist = float(np.abs(drive).sum())
        if right_pregrasp is not None:
            right_drive = right_pregrasp - (
                right_legs_in[-1][1] if right_legs_in else right_hand0
            )
            drive_dist = max(drive_dist, float(np.abs(right_drive).sum()))
        pregrasp_ticks = max(
            self.PREGRASP_TICKS,
            int(round(drive_dist / TRANSIT_SPEED * env.action_hz)),
        )
        rotate_segment = []
        if self._has_reorientation(self.SPEC):
            rotate_segment = [("rotate_book", arrive.copy(), gc, 180)]
        # The upright book is released between two close-fitting filler books. Its reorientation
        # keeps the jaw axis horizontal, so withdraw in that same orientation while the book
        # settles. Only once the open hand is outside the cubby is it safe to reset the wrist for
        # the next task. Resetting in place before retreat made the linkage sweep through the
        # release lane; waiting in place with the old vertical-jaw grasp tipped the book 94 degrees.
        post_release_segments = [("retreat", retreat, go, retreat_ticks)]
        if self._has_reorientation(self.SPEC):
            post_release_segments.append(("clear_book", retreat.copy(), go, 180))
        # Assisted grasping needs 0.30 s of continuous two-pad contact before it creates a fixed
        # joint. Closing both grippers together made the first completed joint snap the towel
        # 22 mm and reset the other arm's contact window. Secure the right edge first, give that
        # joint time to settle, linearly re-aim the still-open left hand at the now-constrained
        # towel, and only then close it. Every grip transition is still interpolated by
        # PickPlaceTask._segment_grip; ``reaim_left`` is a genuine Cartesian segment rather than
        # a measured-pose jump (its endpoint is captured on its first tick below).
        if self._is_bimanual():
            grasp_segments = [
                ("close_right", a.copy(), go, 45),
                ("right_hold", a.copy(), go, 60),
                ("reaim_left", a.copy(), go, 80),
                ("close_left", a.copy(), gc, 45),
                ("grasp_hold", a.copy(), gc, 60),
            ]
        else:
            grasp_segments = [("close", a.copy(), gc, 45)]
        segs = [
            *legs_in,
            ("pregrasp", pregrasp, go, pregrasp_ticks),
            ("approach", a.copy(), go,
             max(self.APPROACH_TICKS,
                 int(round(float(pregrasp[2] - a[2])
                           / self.GRASP_DESCENT_SPEED * env.action_hz)))),
            ("settle",   a.copy(),                       go, self.SETTLE_TICKS),
            *grasp_segments,
            # Timed off the transit speed like every other travelling segment: at a flat 60 ticks
            # this 0.177 m rise ran at 0.296 m/s, the fastest command in the episode -- 1.7x the
            # transit and 4x the place descent.
            # It is NOT why L_arm_j4 rides its +14 deg stop through the carry, though it is where
            # the elbow arrives there. Slowing the rise 1.7x was tried and moved that not at all
            # (48.6 % of ticks on the stop, against 46.4 %), so the saturation is redundancy
            # resolution inside the WBC rather than anything this waypoint list controls.
            ("lift",     lift,                           gc,
             max(60, int(round(float(np.linalg.norm(lift - a)) / TRANSIT_SPEED * env.action_hz)))),
            *legs,
            # Settle at the receptacle before descending, and no longer. 420 ticks was the time
            # the OLD arrive needed to drag the hand 0.20 m back to the staging point at 0.048
            # m/s; that motion is now part of the route (see `goal`), so all that was left was a
            # 4.2 s dwell in which the commanded hand moved 8 mm -- the robot visibly freezing
            # just before every place, and 4 s of stationary frames in every recorded episode.
            # What the segment still owes is convergence after the transit, and that is measured:
            # the base comes to rest 1.5 s in and the arm's joint rate is back to noise by 2.0 s.
            ("arrive",   arrive,  gc, 100),
            *rotate_segment,
            ("carry",    carry,   gc, carry_ticks),
            ("place",    drop,    gc, place_ticks),
            ("release",  drop,    go, 60),
            *post_release_segments,
        ]
        if self._is_bimanual():
            approach_ticks = max(
                self.APPROACH_TICKS,
                int(round(float(right_pregrasp[2] - right_a[2])
                          / self.GRASP_DESCENT_SPEED * env.action_hz)),
            )
            lift_ticks = max(
                60,
                int(round(float(np.linalg.norm(lift - a)) / TRANSIT_SPEED * env.action_hz)),
            )
            self._right_segs = [
                *right_legs_in,
                ("pregrasp", right_pregrasp, go, pregrasp_ticks),
                ("approach", right_a.copy(), go, approach_ticks),
                ("settle", right_a.copy(), go, self.SETTLE_TICKS),
                ("close_right", right_a.copy(), gc, 45),
                ("right_hold", right_a.copy(), gc, 60),
                ("reaim_left", right_a.copy(), gc, 80),
                ("close_left", right_a.copy(), gc, 45),
                ("grasp_hold", right_a.copy(), gc, 60),
                ("lift", right_lift, gc, lift_ticks),
                *right_legs,
                ("arrive", right_arrive, gc, 100),
                ("carry", right_carry, gc, carry_ticks),
                ("place", right_drop, gc, place_ticks),
                ("release", right_drop, go, 60),
                ("retreat", right_retreat, go, retreat_ticks),
            ]
            assert [(n, t) for n, _p, _g, t in self._right_segs] == [
                (n, t) for n, _p, _g, t in segs
            ]
        # One heading per waypoint, held fixed: the real chassis never yaws in box2cloth.
        self._headings = [float(self.ROBOT_YAW)] * len(segs)
        # stand-off eases from the full rest distance to the shortened destination one over the
        # `arrive` segment, so the base is led in only once it is actually at the receptacle.
        destination = {
            "arrive", "rotate_book", "carry", "place", "release", "clear_book", "retreat",
        }
        self._standoffs = [PLACE_STANDOFF if name in destination else 1.0
                           for name, _end, _grip, _ticks in segs]

        jaw_alignment = left_alignment
        reoriented = self._reorient_R(self.SPEC) @ jaw_alignment
        self._grasp_rotations = [
            jaw_alignment if name == "clear_book"
            else reoriented if (
                name == "rotate_book"
                or (
                    self._has_reorientation(self.SPEC)
                    and any(prev[0] == "rotate_book" for prev in segs[:i])
                )
            ) else jaw_alignment
            for i, (name, _end, _grip, _ticks) in enumerate(segs)
        ]
        self._right_grasp_rotations = (
            [right_alignment.copy() for _ in segs] if self._is_bimanual() else None
        )
        self._right_segment_starts = None
        assert len(self._headings) == len(segs) == len(self._standoffs) \
            == len(self._grasp_rotations)
        print(f"[{self.name}] route {np.round(route, 2).tolist()} -> back/strafe/forward "
              f"legs; park {np.round(park, 2).tolist()} free={nav.free(*park)} "
              f"heading {np.degrees(place_heading):+.0f} deg; "
              f"obj@{np.round(a, 3)} release@{np.round(drop, 3)}", flush=True)
        return segs

    def _calibrate_grasp_delta(self, env):
        """Measure each attachment and make its destination finger path place the object.

        For the book, the measured object-to-finger vector rotates with the wrist.  ``arrive``
        therefore uses the flat correction and ``rotate_book`` / insertion use its rotated value;
        interpolating those endpoints keeps the object centre fixed while the wrist turns.
        """
        owners = getattr(self, "_owners", None)
        cycle = getattr(self, "_cycle", None)
        arms = ("left", "right") if self._is_bimanual() else ("left",)
        for arm in arms:
            attr = "_grasp_delta" if arm == "left" else "_right_grasp_delta"
            if getattr(self, attr, None) is not None or env.is_grasping(arm) is not self.apple:
                continue
            delta = env.obj_pos(self.apple) - env.finger_grasp_point(arm)
            setattr(self, attr, delta)
            if not self.SPEC.attachment_reaim:
                print(f"[{self.name}] {self.SPEC.name} {arm} attachment delta="
                      f"{np.round(delta, 3)}; feed-forward destination retained", flush=True)
                continue
            initial = -delta - self._grasp_world_offset(arm, placed=False)
            final = -self._placed_attachment_delta(delta) - self._grasp_world_offset(
                arm, placed=True
            )
            seg_attr = "_segs" if arm == "left" else "_right_segs"
            source = getattr(self, seg_attr)
            shifted = []
            for j, entry in enumerate(source):
                if entry is None:
                    shifted.append(None)
                    continue
                name, end, grip, ticks = entry
                owned = owners is None or owners[j] == cycle
                correction = (initial if name == "arrive" else final)
                if owned and name in {"arrive", "rotate_book", "carry", "place",
                                      "release", "clear_book", "retreat"}:
                    end = np.asarray(end, dtype=float) + correction
                shifted.append((name, end, grip, ticks))
            setattr(self, seg_attr, shifted)
            print(f"[{self.name}] {self.SPEC.name} {arm} attachment delta="
                  f"{np.round(delta, 3)}; destination correction="
                  f"{np.round(final, 3)}", flush=True)

    def _calibrate_reach_delta(self, env):
        """Continuously and rate-limitedly close measured pick residual during ``settle``.

        The old one-shot endpoint jump was sampled 75% through settle.  In the current chain the
        towel still arrived 81 mm off and was physically shoved about 95 mm before attaching.
        Updating the endpoint every settle tick is the scripted analogue of the hardware operator
        visually trimming the leader target, while the speed and total-displacement limits keep it
        a demonstrable trajectory rather than a teleport.
        """
        if not self.SPEC.obj:
            return                         # station tasks have their own contact waypoint logic
        i = getattr(self, "_seg_i", None)
        if i is None or not getattr(self, "_segs", None):
            return
        phase = self._segs[i][0]
        if phase != "settle":
            if (phase in {"close", "close_right"}
                    and not getattr(self, "_reach_servo_logged", False)
                    and self._reach_delta is not None):
                fields = [f"left={np.round(self._reach_delta * 1000, 1)} mm"]
                if self._is_bimanual() and self._right_reach_delta is not None:
                    fields.append(f"right={np.round(self._right_reach_delta * 1000, 1)} mm")
                drift = (0.0 if self._reach_object0 is None else float(np.linalg.norm(
                    env.obj_pos(self.apple) - self._reach_object0
                )))
                print(f"[{self.name}] {self.SPEC.name} reach servo applied "
                      + ", ".join(fields) + f"; object drift={drift * 1000:.1f} mm", flush=True)
                self._reach_servo_logged = True
            return
        if float(getattr(self, "_seg_f", 0.0)) < self.REACH_SERVO_AT:
            return

        owners = getattr(self, "_owners", None)
        cycle = getattr(self, "_cycle", None)
        if not hasattr(self, "_reach_servo_started"):
            self._reach_servo_started = {"left": False, "right": False}
        if not hasattr(self, "_reach_servo_updates"):
            self._reach_servo_updates = {"left": 0, "right": 0}
        if not hasattr(self, "_reach_targets"):
            self._reach_targets = {}
        if not hasattr(self, "_reach_limits"):
            self._reach_limits = {}
        arms = ("left", "right") if self._is_bimanual() else ("left",)
        for arm in arms:
            attr = "_reach_delta" if arm == "left" else "_right_reach_delta"
            total = getattr(self, attr, None)
            if total is None:
                total = np.zeros(3, dtype=float)
                setattr(self, attr, total)
            if arm not in self._reach_targets:
                self._reach_object0 = env.obj_pos(self.apple).copy()
                self._reach_targets[arm] = (
                    self._reach_object0 + self._grasp_world_offset(arm, placed=False)
                )
            # Hold the first settle sample fixed.  Chasing the live object creates positive
            # feedback as soon as an open finger nudges it: the endpoint follows the displaced
            # object, nudges it again, and runs to the correction cap.
            target = self._reach_targets[arm]
            residual = target - env.finger_grasp_point(arm)
            # This correction exists for the x/y base deadband.  The open hand becomes vertically
            # table-constrained near the grasp; integrating that untrackable z residual winds the
            # endpoint down through the tabletop instead of improving the pinch.
            residual[2] = 0.0
            if arm not in self._reach_limits:
                self._reach_limits[arm] = np.minimum(
                    np.abs(residual) + self.REACH_SERVO_MARGIN,
                    np.full(3, self.REACH_SERVO_MAX),
                )
                self._reach_limits[arm][2] = 0.0
            if not getattr(self, "_reach_servo_started", {}).get(arm, False):
                self._reach_servo_started[arm] = True
                print(f"[{self.name}] {self.SPEC.name} {arm} settle residual="
                      f"{np.round(residual * 1000, 1)} mm", flush=True)
            norm = float(np.linalg.norm(residual))
            if norm <= self.REACH_SERVO_TOL:
                continue
            step = residual * min(1.0, self.REACH_SERVO_SPEED / env.action_hz / norm)
            proposed = np.clip(
                total + step, -self._reach_limits[arm], self._reach_limits[arm]
            )
            step = proposed - total
            proposed_norm = float(np.linalg.norm(proposed))
            if proposed_norm > self.REACH_SERVO_MAX:
                proposed *= self.REACH_SERVO_MAX / proposed_norm
                step = proposed - total
            if float(np.linalg.norm(step)) < 1e-9:
                continue
            setattr(self, attr, proposed)
            self._reach_servo_updates[arm] += 1
            seg_attr = "_segs" if arm == "left" else "_right_segs"
            shifted = []
            for j, entry in enumerate(getattr(self, seg_attr)):
                if entry is None:
                    shifted.append(None)
                    continue
                name, end, grip, ticks = entry
                if name in {"settle", "close", "close_right", "right_hold",
                            "reaim_left", "close_left", "grasp_hold", "lift"} and (
                        owners is None or owners[j] == cycle):
                    end = np.asarray(end, dtype=float) + step
                shifted.append((name, end, grip, ticks))
            setattr(self, seg_attr, shifted)

    def _reaim_second_hand(self, env, phase):
        """Capture one smooth left-hand segment after the right assisted grasp has settled."""
        if (not self._is_bimanual() or phase != "reaim_left"
                or getattr(self, "_second_hand_reaimed", False)
                or env.is_grasping("right") is not self.apple):
            return
        i = self._seg_i
        old_end = np.asarray(self._segs[i][1], dtype=float)
        new_end = env.obj_pos(self.apple) + self._grasp_world_offset("left", placed=False)
        correction = new_end - old_end
        owners = getattr(self, "_owners", None)
        cycle = getattr(self, "_cycle", None)
        shifted = []
        for j, (name, end, grip, ticks) in enumerate(self._segs):
            if (name in {"reaim_left", "close_left", "grasp_hold", "lift"}
                    and (owners is None or owners[j] == cycle)):
                end = np.asarray(end, dtype=float) + correction
            shifted.append((name, end, grip, ticks))
        self._segs = shifted
        self._second_hand_reaimed = True
        print(f"[{self.name}] {self.SPEC.name} left re-aim after right grasp="
              f"{np.round(correction * 1000, 1)} mm", flush=True)

    def expert_step(self, env, obs):
        # Calibrate before advancing the next waypoint: the pick re-aim during settle, then the
        # place re-aim once the object is actually attached.
        self._calibrate_reach_delta(env)
        self._calibrate_grasp_delta(env)
        cmd = super().expert_step(env, obs)
        # This runs after the first ``reaim_left`` sample (f=0), so changing that segment's
        # endpoint cannot discontinuously move the commanded hand. Subsequent samples traverse
        # the measured correction at one constant rate over the remaining segment clock.
        self._reaim_second_hand(env, cmd.phase)
        history = getattr(self, "_ever_grasped", None)
        if history is not None:
            for arm in history:
                history[arm] |= env.is_grasping(arm) is self.apple
        return cmd

    def _release_point(self, env, arm="left"):
        """Initial finger target for landing the carried object on the receptacle.

        The waypoints command the fingertip grasp centre, and the object is held `grasp_offset`
        away from it (a plate is taken by its rim, 45 mm off its own centre). Sending the fingers
        to the receptacle therefore lands the object 45 mm past it -- so the offset is subtracted
        back out here, exactly as it was added when picking. Once sticky grasp has attached,
        `_calibrate_grasp_delta` replaces this nominal offset with the one actual contact transform.
        """
        gp = env.obj_pos(self.bowl)
        # The live AABB preserves per-instance scaling (and test doubles); applying the scripted
        # relative rotation gives the post-rotation half-height. All reoriented specs currently spawn
        # axis-aligned, so this is exact rather than a metadata approximation.
        live_extent = np.asarray(self.apple.aabb_extent, dtype=float)
        half = float((np.abs(self._reorient_R(self.SPEC)) @ live_extent)[2]) / 2.0
        obj_target = np.array([
            gp[0], gp[1], self._support_surface_z() + half + self.SPEC.release_dz
        ])
        obj_target += rot_z(float(self.station.yaw)) @ np.asarray(
            self.SPEC.place_offset, dtype=float)
        return obj_target + self._grasp_world_offset(arm, placed=True)

    def _support_surface_z(self):
        """Live world z of the surface this spec intends to place onto."""
        if self.SPEC.support_z_from_bottom is None:
            return float(self.bowl.aabb[1][2])
        return float(self.bowl.aabb[0][2]) + float(self.SPEC.support_z_from_bottom)

    def _nav(self, env, hand_off=None, hand_z=None, exclude=None):
        """Planner that knows what this episode actually put on the floor -- and what the HAND
        will sweep through, which is not the same thing.

        `clear_boxes` re-opens what `declutter` removed and the chairs `reset` displaces (the
        shipped map still shows them standing). `block_boxes` closes the layout's own stations,
        which the map was baked before -- without it the route runs straight through the bookcase
        it is driving to. The station being WORKED is excluded: the base is meant to come right up
        to that one, and its keepout circle is what stops it.

        `hand_off` is the world offset from the chassis to the carried hand. The hand rides 0.36 m
        to one side and 0.63 m ahead of the base, so a route that clears the furniture by the
        chassis radius can still drag the object straight through it: measured, dish2rack drove
        the bowl into the towel shelf standing 0.63 m east of its own lane -- ee_err jumped from
        0.078 to 0.30 m, the body pitched 5 deg and the bowl was knocked out of the hand. A
        station is unsafe for the base at `p` exactly when `p + hand_off` is inside it, so each
        station is blocked TWICE: once where it stands, once shifted by `-hand_off`. Only stations
        taller than the carry height get the second box -- the hand passes over the rest.

        `exclude` names the stations to leave open, and defaults to the one being worked. A chained
        rollout also has to plan the drive BACK from a station it has just served, and the base is
        standing at that station's own park when it does, so that leg opens the station it is
        leaving instead of the one it is heading for.
        """
        exclude = (self.SPEC.station,) if exclude is None else tuple(exclude)
        boxes = layout.station_boxes(exclude=exclude, keys=self.scene_station_keys)
        hand_boxes = []
        if hand_off is not None:
            offsets = np.asarray(hand_off, dtype=float)
            offsets = offsets[None, :] if offsets.ndim == 1 else offsets
            for off in offsets:
                for key in self.scene_station_keys:
                    if key in exclude:
                        continue
                    stn = STATIONS[key]
                    if hand_z is not None and stn.top_z() < hand_z - 0.03:
                        continue
                    x0, y0, x1, y1 = stn.footprint()
                    pad = HAND_SWEEP_PAD
                    hand_boxes.append((x0 - off[0] - pad, y0 - off[1] - pad,
                                       x1 - off[0] + pad, y1 - off[1] + pad))
        return NavMap(env.scene_model, clear_boxes=self._clear_boxes(env), block_boxes=boxes,
                      point_block_boxes=hand_boxes)

    def _clear_boxes(self, env):
        """Floor the planner must treat as OPEN even though the shipped map shows it occupied:
        the chairs `reset` displaces, and everything `declutter` deleted.

        Each box is grown by `layout.DECLUTTER_PAD` -- see that constant for why the exact
        footprint is not enough. The chairs' DESTINATIONS need no matching block box: both sit
        within a chassis radius of the west wall, so that floor is not in the eroded free set to
        begin with and no route can be planned through it.
        """
        pad = layout.DECLUTTER_PAD
        boxes = [(at[0] - 0.40, at[1] - 0.40, at[0] + 0.40, at[1] + 0.40)   # the spot it VACATES
                 for _cat, at, _to in CLEAR_CHAIRS]
        return boxes + [(x0 - pad, y0 - pad, x1 + pad, y1 + pad)
                        for x0, y0, x1, y1 in getattr(self, "_removed_boxes", [])]

    # ---- plan clock ----
    def _clock_step(self, env):
        """One plan tick per simulation tick: tracking error never modulates the action speed."""
        return 1.0

    def expert_reset(self, env):
        super().expert_reset(env)
        # Reach / attachment calibration may trim hand endpoints after planning.  Keep an immutable
        # copy for the bimanual head reference: on hardware the headset and both hand targets are
        # independent WBIK inputs, so a hand re-aim must not silently become a base translation.
        self._head_plan_segs = [
            (name, np.asarray(end, dtype=float).copy(), grip, ticks)
            for name, end, grip, ticks in self._segs
        ]
        right = getattr(self, "_right_segs", None)
        self._right_head_plan_segs = None if right is None else [
            None if entry is None else (
                entry[0], np.asarray(entry[1], dtype=float).copy(), entry[2], entry[3]
            )
            for entry in right
        ]

    # ---- arm ----
    #: Body-frame shift of the IDLE right hand: 0.25 m back, 0.15 m down. Outside the bimanual
    #: towel segments, the right arm has no waypoints and holds this pose in the body frame. The
    #: places park the base right up against a wall
    #: of furniture, and measured at those parks the nominal pose leaves the right gripper just
    #: 3 mm from the bookcase during the towel place (and 173 mm during the book place). It is not
    #: reaching for anything, so it should be out of the way.
    #:
    #: DOWN alone does not do it, and this is the part worth remembering: at the dish park the
    #: right hand sits directly over the drying rack, clearing it only because it rides high. Drop
    #: it 0.25 m and it stops passing over and starts entering -- measured -180 mm, i.e. inside the
    #: rack's footprint, a worse collision than the one being fixed. So the descent stops at 0.15 m
    #: (lowest right link 0.816 world, 60 mm over the rack's 0.756 crown) and the clearance is
    #: bought by pulling BACK instead, which helps at every park at once. Measured worst-case gap
    #: to the row over the three parks: 3 mm nominal -> 252 mm here, with the WBC tracking the
    #: commanded pose to 1 mm.
    RIGHT_TUCK = (-0.25, 0.0, -0.15)

    def _right_target(self, head_target, heading):
        """The held right-hand pose, tucked. Still a pure command: body frame, no measured state."""
        held = self._Ree_base.copy()
        held[:3, 3] += np.asarray(self.RIGHT_TUCK, dtype=float)
        return self._commanded_base_target(head_target, heading) @ held

    def _grasp_rotation_now(self, env):
        """Wrist orientation carried in the BODY frame, so the hand turns with the robot.

        The tabletop task freezes this in the world, which is fine when the base never turns.
        Here the body swings ~100 deg between stations, and a world-fixed wrist command reads as
        "hold the hand at its spawn heading regardless of which way you face": the gripper keeps
        pointing the original direction after the robot has correctly turned, and the arm has to
        contort around the body to satisfy it -- which is what pushed `L_arm_j4` onto its stop.
        Rotating the reset orientation by the SAME commanded heading the position and stand-off
        use keeps the whole hand command in one consistent frame.
        """
        i = getattr(self, "_seg_i", 0)
        f = getattr(self, "_seg_f", 0.0)
        extra = self._segment_rotation(getattr(self, "_grasp_rotations", None), i, f)
        return rot_z(self._head_heading_now(env)) @ extra @ self._Rgrasp_base

    def _smooth_center(self, env, raw):
        """Follow the exact fingertip arc that keeps the left EEF fixed during ``orient``."""
        if (getattr(self, "_segs", None)
                and self._segs[getattr(self, "_seg_i", 0)][0] == "orient"
                and self._orient_left_eef is not None):
            return self._orient_left_eef + self._grasp_rotation_now(env) @ self._grasp_off_local
        return super()._smooth_center(env, raw)

    def _smooth_right_center(self, env, raw):
        """Right-hand mirror of :meth:`_smooth_center` for the bimanual wrist turn."""
        if (getattr(self, "_right_segs", None)
                and self._right_segs[getattr(self, "_seg_i", 0)] is not None
                and self._right_segs[getattr(self, "_seg_i", 0)][0] == "orient"
                and self._orient_right_eef is not None):
            i = getattr(self, "_seg_i", 0)
            f = getattr(self, "_seg_f", 0.0)
            extra = self._segment_rotation(self._right_grasp_rotations, i, f)
            rotation = rot_z(self._head_heading_now(env)) @ extra @ self._Rgrasp_right_base
            return self._orient_right_eef + rotation @ self._right_grasp_off_local
        return super()._smooth_right_center(env, raw)

    def _bimanual_active(self):
        if not self._is_bimanual():
            return False
        right = getattr(self, "_right_segs", None)
        i = getattr(self, "_seg_i", 0)
        return right is not None and 0 <= i < len(right) and right[i] is not None

    def _planned_center(self, segs, start, starts=None):
        """Interpolate an immutable pre-calibration waypoint list on the live plan clock."""
        i = getattr(self, "_seg_i", 0)
        f = getattr(self, "_seg_f", 0.0)
        entry = segs[i]
        if entry is None:
            raise RuntimeError("requested a planned centre for an inactive arm segment")
        if i == 0 or segs[i - 1] is None:
            previous = start if starts is None or starts[i] is None else starts[i]
        else:
            previous = segs[i - 1][1]
        return (1.0 - f) * np.asarray(previous, dtype=float) + f * np.asarray(entry[1], dtype=float)

    def _head_work_point(self):
        if (getattr(self, "_segs", None)
                and self._segs[getattr(self, "_seg_i", 0)][0] == "orient"
                and self._orient_head_center is not None):
            return np.asarray(self._orient_head_center)
        if self._bimanual_active():
            left = self._planned_center(self._head_plan_segs, self._start_center)
            right = self._planned_center(
                self._right_head_plan_segs,
                self._right_start_center,
                getattr(self, "_right_segment_starts", None),
            )
            return (left + right) / 2.0
        return self._center

    # ---- head ----
    def _head_target_pos(self, env, heading):
        """Rest stand-off from the work point, with the z held inside the reachable band.

        The clamp is applied on EVERY tick, not just when carrying -- it is part of the one rule
        ("stand off from the work point the way you do at rest, as far as your neck allows"),
        not a phase. See HEAD_Z_DROP for what an unclamped z does to the base.
        """
        pos = self._head_work_point() + rot_z(heading) @ self._standoff()
        pos[2] = float(np.clip(pos[2], self._head_pos0[2] - HEAD_Z_DROP,
                               self._head_pos0[2] + HEAD_Z_RISE))
        self._last_head_cmd = pos
        Rz = rot_z(heading)
        # Same planar body command `_commanded_base_target` will derive from this head pose.
        self._last_base_cmd = pos[:2] - (Rz @ self._head_pos_base)[:2]
        return pos

    def _standoff(self):
        """Rest camera-to-hand offset, shortened horizontally.

        The scale is per-waypoint and lerped on the same schedule as the position and the heading,
        so it is one continuous quantity rather than a mode: 1.0 while picking and driving, easing
        to PLACE_STANDOFF as the robot closes on the receptacle.
        """
        scales = getattr(self, "_standoffs", None)
        if not scales:
            s = STANDOFF_SCALE
        else:
            s0 = scales[max(self._seg_i - 1, 0)]
            s = s0 + (scales[self._seg_i] - s0) * self._seg_f
        base_off = (self._head_off_bimanual_base if self._bimanual_active()
                    else self._head_off_base)
        off = np.asarray(base_off, dtype=float).copy()
        off[:2] *= s
        return off

    # ---- heading ----
    def _head_heading_now(self, env):
        """Gaze/base heading, lerped along the SAME waypoint schedule as the hand position.

        It has to be feed-forward. The obvious rule -- look at whatever the hand is reaching for,
        measured from the live base pose -- is unstable, because the camera's rest stand-off from
        the hands has a 0.34 m LATERAL component (the left hand works to the left of the head).
        Commanding "stand off from the work point by a vector that rotates with the heading, and
        take the heading from where I currently am" closes a positive feedback loop through the
        base: measured, the chassis spun 340 deg during the grasp and never closed on the mug.
        Per-waypoint headings break the loop -- nothing in the gaze command reads the base pose.
        """
        h0 = self._headings[max(self._seg_i - 1, 0)]
        h1 = self._headings[self._seg_i]
        d = (h1 - h0 + np.pi) % (2 * np.pi) - np.pi          # shortest way round
        self._head_heading = float(h0 + d * self._seg_f)
        return self._head_heading

    @property
    def MAX_TICKS(self):
        """Wall ticks to allow, with shutdown / final-settle headroom."""
        if not getattr(self, "_segs", None):
            return 5000
        return sum(n for _, _, _, n in self._segs) + 200


def _make(spec):
    attrs = {"name": spec.name, "SPEC": spec,
             "__doc__": f"Long-horizon {spec.name} (see LHSpec)."}
    if spec.obj_xy is not None:
        attrs["APPLE_XY"] = tuple(spec.obj_xy)      # the documented subclass override point
    cls = type(
        "".join(p.capitalize() for p in spec.name.replace("2", "_to_").split("_")) + "Task",
        (LongHorizonPickPlace,),
        attrs,
    )
    return cls


LH_TASKS = [_make(s) for s in SPECS]
SPEC_BY_NAME = {s.name: s for s in SPECS}


class LongHorizonSequence(LongHorizonPickPlace):
    """All three long-horizon pick-and-places, back to back, in ONE rollout.

    The three shipped tasks are the same skill against three receptacles, and each is validated on
    its own. Chaining them is not three episodes stitched together: the robot picks a second time
    from a table it is no longer parked at, having just placed at a station metres away, and has to
    do it without a reset putting anything back. So exactly three things are new, and everything
    else -- the grasp, the transit, the place, the head/base coupling -- is the parent's, called
    once per cycle with its object and station swapped in.

    1. **Three objects on the table at once.** They cannot share `SRC_OBJ_XY`, so each gets its own
       mark (`SOURCE_MARKS`) and they are spread in y. Lateral is the right axis to spread on: the
       base is never commanded to a park, it follows the head command (`work_point +
       rest_stand_off`), so moving a mark in y moves the CHASSIS by the same amount and leaves the
       body-relative reach -- the thing the single tasks validated -- untouched. Moving them in x
       would not: x is depth, and the towel already sits at its own x because a 220 mm object on
       the shared mark overhangs the tabletop's near face.

    2. **A return drive.** `_box2cloth_route` is reused unchanged and simply run backwards -- from
       the station's park to the next mark's park -- which yields the same -vx / -vy / +vx pattern
       the outbound leg uses, because the retreat lane at `RETREAT_X` is between them either way.
       A `clear` segment lifts the hand to its rest height at the station first, so the gripper
       withdraws from the receptacle before the chassis moves.

    3. **Per-cycle bookkeeping.** `_owners` labels every segment with the cycle that built it, and
       `expert_step` swaps `SPEC` / `apple` / `bowl` (hence `station`, `keepouts`, `_release_point`,
       `success`) the tick the plan crosses into the next one. Grasp calibration is per cycle and
       per arm: every attachment shifts only its OWN destination waypoints -- without `_owners`,
       the second pick's measured offset would be applied to the third pick's place as well.

    Success is all three placed, evaluated with the parent's own per-spec predicate.
    """

    name = "lh_all"
    #: Which specs to chain, in order. Farthest station first: the row runs dish rack (y = -0.41),
    #: towel rack (-1.05), bookcase (-1.83), so book -> dish -> towel walks the robot out to the far
    #: end and back up it, and no return drive has to pass a station it has already loaded.
    ORDER = ("book2shelf", "dish2rack", "towel2rack")
    #: Where each object waits on the source table. Measured, the tabletop is solid over
    #: x 1.10-1.85, y -0.05..+0.87, so these sit well inside it.  At the nominal marks, book /
    #: towel / bowl occupy about y=0.10..0.18 / 0.31..0.49 / 0.74..0.82, leaving at least 0.13 m
    #: nominal clearance (at least 0.09 m under the +/-20 mm reset jitter).  The bimanual towel is
    #: centred near the body's midline; the bowl stays near the left hand's original mark.
    SOURCE_MARKS = {"book2shelf": (1.20, 0.14), "dish2rack": (1.20, 0.78),
                    "towel2rack": (1.14, 0.40)}
    #: Ticks to raise the hand from the receptacle back to its rest height before driving off.
    CLEAR_TICKS = 60
    #: Floor on a return leg's duration, matching the outbound legs' own `max(40, ...)`.
    RETURN_TICKS_MIN = 40

    # ---- which cycle is live ----
    def _select(self, k):
        """Point every per-task handle at cycle @k. One assignment set, used by build and by run."""
        task_name = self.ORDER[k]
        self._cycle = k
        self.SPEC = SPEC_BY_NAME[task_name]
        self.APPLE_XY = self.SOURCE_MARKS[task_name]
        self.apple = self._objs[task_name]
        self.bowl = self._stns[task_name]
        histories = getattr(self, "_grasp_histories", None)
        if histories is not None and len(histories) > k:
            self._ever_grasped = histories[k]

    def _mark_park(self, task_name):
        """Base pose an object's mark implies -- see the class note on spreading marks in y."""
        return np.asarray(self.SOURCE_MARKS[task_name], dtype=float) - self._carry_local[:2]

    # ---- scene ----
    def object_configs(self):
        stations = layout.station_configs(self.scene_station_keys)
        for cfg in stations:                       # see the parent: spawned high, dropped at reset
            assert cfg["fixed_base"]
            cfg["position"][2] += 3.0
        objs = [{"type": "DatasetObject", "name": f"lh_obj_{n}", **SPEC_BY_NAME[n].obj,
                 "position": [*self.SOURCE_MARKS[n], 0.90]} for n in self.ORDER]
        liners = self._liner_configs([SPEC_BY_NAME[n] for n in self.ORDER])
        return objs + liners + self._book_filler_configs() + stations

    def bind(self, env):
        reg = env.env.scene.object_registry
        objs = env.env.scene.objects
        cands = [o for o in objs if o.category == "breakfast_table"]
        if not cands:
            raise ValueError(f"{self.name}: scene has no breakfast_table")
        self.table = min(cands, key=lambda o: np.linalg.norm(
            env.obj_pos(o)[:2] - np.array(layout.SRC_TABLE_XY)))
        self._chairs = layout.side_chairs(env)
        self._objs, self._stns = {}, {}
        for task_name in self.ORDER:
            obj = reg("name", f"lh_obj_{task_name}")
            if obj is None:
                raise ValueError(f"{self.name}: carried object lh_obj_{task_name} not in the scene")
            spec = SPEC_BY_NAME[task_name]
            # As the single task: the receptacle is the liner where the spec has one.
            name = self._liner_name(spec) or STATIONS[spec.station].name
            stn = reg("name", name)
            if stn is None:
                raise ValueError(f"{self.name}: receptacle {name!r} for {task_name!r} not in the scene")
            self._objs[task_name], self._stns[task_name] = obj, stn
        self._select(0)

    def reset(self, env):
        layout.place_stations(env, self.scene_station_keys)
        for task_name in self.ORDER:
            self._place_liner(env, SPEC_BY_NAME[task_name])
        self._place_book_fillers(env)
        layout.place_side_chairs(env, self._chairs)
        self.table.keep_still()
        self.ztop = float(self.table.aabb[1][2])
        for task_name in self.ORDER:
            spec = SPEC_BY_NAME[task_name]
            mx, my = self.SOURCE_MARKS[task_name]
            self._objs[task_name].set_position_orientation(
                position=[mx + self.rng.uniform(-0.02, 0.02),
                          my + self.rng.uniform(-0.02, 0.02),
                          self.ztop + self._upright_half_height(spec) + 0.004],
                orientation=list(spec.obj_quat))
            self._objs[task_name].keep_still()
        self._select(0)

    # ---- success ----
    def success(self, env):
        """Every cycle placed, each judged by the parent's own predicate for its spec."""
        live = self._cycle
        try:
            out = {}
            for k, task_name in enumerate(self.ORDER):
                self._select(k)
                out[task_name] = bool(super().success(env))
        finally:
            self._select(live)
        print(f"[{self.name}] per-task success: {out}", flush=True)
        return all(out.values())

    # ---- expert ----
    def _src_park(self):
        return self._cycle_src_park

    def _hand_start(self):
        return self._cycle_hand_start

    def _right_hand_start(self):
        return self._cycle_right_hand_start

    def _tucked_right_center_at(self, base_xy):
        """Commanded right finger centre at a return route's endpoint."""
        Tbase = np.eye(4)
        Tbase[:3, :3] = rot_z(float(self.ROBOT_YAW))
        Tbase[:2, 3] = np.asarray(base_xy, dtype=float)
        Tbase[2, 3] = self._base_cmd_z
        held = self._Ree_base.copy()
        held[:3, 3] += np.asarray(self.RIGHT_TUCK, dtype=float)
        Tee = Tbase @ held
        return Tee[:3, 3] + Tee[:3, :3] @ self._right_grasp_off_local

    def _build_segments(self, env):
        base0 = env.link_pose("base")[:3, 3]
        #: rest hand-offset from the base, the body constant that turns a mark into a park
        self._carry_local = rot_z(-self.ROBOT_YAW) @ (self._start_center - base0)
        hand_off = rot_z(float(self.ROBOT_YAW)) @ self._carry_local
        rest_z = float(self._start_center[2])
        go = env.grip_open

        segs, headings, standoffs, owners = [], [], [], []
        grasp_rotations, right_rotations = [], []
        right_segs, right_starts, grasp_histories = [], [], []
        self._grasp_histories = []
        hand = self._start_center.copy()
        prev_park = None
        for k, task_name in enumerate(self.ORDER):
            self._select(k)
            self._cycle_src_park = self._mark_park(task_name)
            if prev_park is not None:
                segs.append(("clear", np.array([hand[0], hand[1], rest_z]), go, self.CLEAR_TICKS))
                nav = self._nav(env, hand_off=hand_off, hand_z=rest_z,
                                exclude=(SPEC_BY_NAME[self.ORDER[k - 1]].station,))
                route = _box2cloth_route(nav, prev_park, self._cycle_src_park)
                prev = np.asarray(route[0], dtype=float)
                for j, pt in enumerate(route[1:]):
                    dist = float(np.linalg.norm(pt - prev))
                    waypoint = np.append(pt[:2], rest_z) + hand_off
                    waypoint[2] = rest_z
                    segs.append((f"return{j}", waypoint, go,
                                 max(self.RETURN_TICKS_MIN,
                                     int(round(dist / TRANSIT_SPEED * env.action_hz)))))
                    prev = pt
                hand = np.asarray(segs[-1][1], dtype=float)
                n_new = len(segs) - len(owners)
                headings += [float(self.ROBOT_YAW)] * n_new     # nothing here yaws, as everywhere
                standoffs += [STANDOFF_SCALE] * n_new
                grasp_rotations += [np.eye(3) for _ in range(n_new)]
                right_rotations += [np.eye(3) for _ in range(n_new)]
                right_segs += [None] * n_new
                right_starts += [None] * n_new
                owners += [k] * n_new
                print(f"[{self.name}] return{k} route {np.round(route, 2).tolist()}", flush=True)
            self._cycle_hand_start = hand
            self._cycle_right_hand_start = (
                self._right_start_center.copy() if k == 0
                else self._tucked_right_center_at(self._cycle_src_park)
            )
            cycle = super()._build_segments(env)     # the validated pick / transit / place, as-is
            cycle_right = self._right_segs
            cycle_right_rotations = self._right_grasp_rotations
            grasp_histories.append(self._ever_grasped)
            segs += cycle
            headings += list(self._headings)
            standoffs += list(self._standoffs)
            grasp_rotations += list(self._grasp_rotations)
            if cycle_right is None:
                right_segs += [None] * len(cycle)
                right_rotations += [np.eye(3) for _ in cycle]
            else:
                right_segs += list(cycle_right)
                right_rotations += list(cycle_right_rotations)
            right_starts += [self._cycle_right_hand_start.copy() for _ in cycle]
            owners += [k] * len(cycle)
            hand = np.asarray(cycle[-1][1], dtype=float)
            prev_park = self._dest_park
        if not (len(segs) == len(headings) == len(standoffs) == len(owners)
                == len(grasp_rotations) == len(right_segs) == len(right_rotations)
                == len(right_starts)):
            raise ValueError(
                f"{self.name}: per-waypoint lists disagree -- {len(segs)} segments, "
                f"{len(headings)} headings, {len(standoffs)} standoffs, {len(owners)} owners")
        self._headings, self._standoffs, self._owners = headings, standoffs, owners
        self._grasp_rotations = grasp_rotations
        self._right_segs = right_segs if any(entry is not None for entry in right_segs) else None
        self._right_grasp_rotations = right_rotations
        self._right_segment_starts = right_starts
        self._grasp_histories = grasp_histories
        self._seg_i, self._seg_f = 0, 0.0
        self._select(0)
        per_cycle = [sum(n for (_, _, _, n), o in zip(segs, owners) if o == k)
                     for k in range(len(self.ORDER))]
        print(f"[{self.name}] {len(segs)} segments, {sum(per_cycle)} ticks: "
              + ", ".join(f"{n} {t}t" for n, t in zip(self.ORDER, per_cycle)), flush=True)
        return segs

    def _calibrate_grasp_delta(self, env):
        """Use the parent's owner-aware left/right and rotated-attachment correction."""
        return super()._calibrate_grasp_delta(env)

    def expert_step(self, env, obs):
        """Advance the cycle the tick the plan crosses into it, then run the parent unchanged."""
        k = self._owners[min(getattr(self, "_seg_i", 0), len(self._owners) - 1)]
        if k != self._cycle:
            self._select(k)
            # Both re-aims are per PICK, so both reset at the cycle boundary.
            self._grasp_delta = None            # see `_calibrate_grasp_delta`
            self._right_grasp_delta = None
            self._reach_delta = None            # see `_calibrate_reach_delta`
            self._right_reach_delta = None
            self._reach_servo_updates = {"left": 0, "right": 0}
            self._reach_servo_started = {"left": False, "right": False}
            self._reach_targets = {}
            self._reach_limits = {}
            self._reach_object0 = None
            self._reach_servo_logged = False
            self._second_hand_reaimed = False
            env.base_keepouts = self.keepouts()  # the worked station's circle moves with the cycle
            print(f"[{self.name}] --- cycle {k}: {self.SPEC.name} "
                  f"-> {self.SPEC.station} ---", flush=True)
        return super().expert_step(env, obs)
