"""Task-scoped furniture poses for the long-horizon suite.

Pick/place tasks spawn the three receptacles together; close-fridge and push-chair spawn only their
own target. This keeps the fridge out of dish2rack without leaving a phantom planner obstacle or
removing the fridge needed by CloseFridgeTask.

The robot starts at (-0.35, 0.40), facing world +x. The dish rack, the drying rack and the bookcase
stand along the front/east wall (inner face x=1.86). The two that have a front face west toward the
robot; the drying rack is turned 90 deg to stand along the wall, for the reason given on its
`Station`. Their backs are 2 cm from the wall and neighbouring AABBs have 7-18 cm gaps. Working
heading is zero for all three regardless: after picking, the robot backs straight out, strafes
right (-y), then approaches forward (+x), with no yaw. This is the same -vx -> -vy -> +vx chassis
pattern measured in the real box2cloth demonstrations.

The explicit root z values account for each scaled asset's root-to-AABB-center offset. The
bookcase and the drying rack stand on the floor. The dataset's sole dish-rack model is a wall rack;
its root is raised so the tested working rail is at 0.77 m, close to the source tabletop height,
without the sustained left-elbow saturation observed with the rail at 0.82 m.
"""
from __future__ import annotations

import dataclasses
import json
import os

import numpy as np

#: source work surface: the Rs_int dining table every task picks from
SRC_TABLE_XY = (1.47, 0.41)
#: where the carried object starts on it
SRC_OBJ_XY = (1.20, 0.74)
#: robot spawn (base centre) and heading -- straight-in at the table, the validated tabletop pose
ROBOT_POS = (-0.35, 0.40, 0.03)
ROBOT_YAW = 0.0
#: base pose at the source table -- the forward park the tabletop tasks converge to
SRC_PARK = (0.56, 0.40)
#: keepout radius around the source table (base centre -> table centre at that park)
SRC_KEEPOUT = 0.90

#: Scene objects deleted at load: everything no task touches, which only costs render time, crowds
#: the review video and gives the base something to snag on. Structure (walls, floors, ceilings,
#: doors, windows) and anything a task stands on or places onto is NOT in this list. Removing
#: furniture invalidates the shipped traversability map, so `declutter` records each removed
#: footprint and hands it back as a `clear_box`.
DECLUTTER = ("pot_plant", "picture", "carpet", "laptop", "loudspeaker", "floor_lamp",
             "standing_tv", "mirror", "table_lamp", "bed", "shower_stall", "toilet",
             "pedestal_sink", "ottoman", "sofa", "coffee_table", "swivel_chair",
             "bottom_cabinet")

#: Dining chairs that sit between the robot and the table's near edge, and where they are tucked
#: at reset: (category, (x, y) of the instance, (x, y) it is moved to). Against the west wall, out
#: of every route and clear of the robot's own spawn -- the old destination was a relative shove
#: (+0.9, -1.9) that now lands on top of the station row. Displaced by EVERY task, including the
#: two that never touch the table, because the room has to look the same in all of them.
CLEAR_CHAIRS = (("straight_chair", (0.97, 0.38), (-1.72, 0.55)),
                ("straight_chair", (1.57, -0.18), (-1.72, 0.02)))


#: Pad added to every re-opened footprint before it is handed to `NavMap(clear_boxes=...)`. The
#: shipped traversability map bakes a margin around each object, so re-opening the floor with the
#: asset's EXACT box leaves a rim still marked occupied. Measured on `bottom_cabinet_jhymlr_0`:
#: the map is blocked from x = 1.32 while the asset's own box starts at 1.38, and that 6 cm rim
#: ran 1.6 m down the room and cut the entire east side -- where the row now stands -- off the
#: planner, so nothing east of x = 0.96 was reachable at all.
DECLUTTER_PAD = 0.10

ASSETS = os.environ.get("OMNIGIBSON_DATASET_PATH",
                        os.path.expanduser("~/BEHAVIOR-1K/datasets/behavior-1k-assets"))


def bbox_size(category, model, scale=None):
    """The asset's canonical bounding box, from the dataset metadata (no sim needed)."""
    path = os.path.join(ASSETS, "objects", category, model, "misc", "metadata.json")
    with open(path) as fh:
        ext = np.asarray(json.load(fh)["bbox_size"], dtype=float)
    return ext * np.asarray(scale if scale is not None else [1.0, 1.0, 1.0], dtype=float)


def rot_z2(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


@dataclasses.dataclass(frozen=True)
class Station:
    """One fixed piece of furniture: where it stands and how the robot works it."""
    key: str
    category: str
    model: str
    xy: tuple
    #: spawn yaw about z, and also the direction the station's FRONT faces (the asset's front is
    #: its local +x). Every station is turned to face the robot, i.e. `yaw = approach + 180 deg`.
    #: The chair is the exception -- see the module docstring.
    yaw: float
    #: heading the robot holds while working this station -- also the heading the whole transit
    #: ramps to. Chosen so the park it implies is standable (see the module docstring).
    approach: float
    #: keepout radius the chassis stops at. See `LongHorizonPickPlace.keepouts`: the head command
    #: deliberately leads INSIDE this circle, so the circle -- not the command -- sets the
    #: working distance.
    keepout: float = 0.62
    fixed_base: bool = True
    #: spawn z. None => stand on the floor, from the asset's own bbox.
    z: float | None = None
    #: per-axis scale, in the asset's OWN frame. Used to fit the row and the arm, not for looks.
    #: Width (local y) is trimmed only where the row would
    #: otherwise not fit between the east wall and the westernmost standable park; height is
    #: trimmed so the receptacle top lands in the 0.5-1.0 m band the arm can actually work
    #: (measured -- above ~0.95 m the elbow pins at its +14 deg stop and the place misses).
    scale: tuple | None = None
    #: keepout radius enforced even when this station is NOT the one being worked. Only for
    #: furniture a passing chassis can actually knock into: measured, pushchair overshot its
    #: route endpoint by 0.2 m, grazed the fridge, and the resulting reaction spun the base 90 deg
    #: and swept the chair 1.2 m across the floor before the gripper ever closed. 0 = rely on the
    #: route planner alone, which is right for the low, wall-hugging receptacles (a circle big
    #: enough to matter there would also swallow the neighbouring station's park).
    avoid: float = 0.0
    #: kg forced onto the root link at reset. Free-standing appliances have to be heavy enough
    #: that a brush from the chassis does not send them across the room; `fixed_base` is not an
    #: option for the fridge because it locks the door joint outright.
    mass: float | None = None
    #: Optional multiplicative RGB tint applied to the asset material at every reset.
    tint: tuple | None = None
    #: Measured world z of the station's live AABB top. `spawn_z() + extent()[2] / 2` is only the
    #: top when the asset's root IS its bbox centre, and the drying rack's sits 0.14 m below it --
    #: which made a 0.75 m rack read as 0.89 m. That matters because `_nav` uses the height to
    #: decide whether the CARRIED object clears a station: over-estimate it and the task gets a
    #: clearance box it does not need, which -- eroded by the chassis radius like every other block
    #: box -- swallowed the neighbouring bookcase's park and made book2shelf unroutable.
    top: float | None = None

    def top_z(self):
        """World z of the top of this station, measured where the asset's root is offset."""
        if self.top is not None:
            return float(self.top)
        return self.spawn_z() + float(self.extent()[2]) / 2

    @property
    def name(self):
        return f"stn_{self.key}"

    def extent(self):
        """World-axis-aligned (dx, dy, dz) at the spawn yaw."""
        ext = bbox_size(self.category, self.model, self.scale)
        half = np.abs(rot_z2(self.yaw)) @ (ext[:2] / 2)
        return np.array([2 * half[0], 2 * half[1], ext[2]])

    def footprint(self, pad=0.0):
        """(xmin, ymin, xmax, ymax) the chassis must not drive into."""
        e = self.extent()
        return (self.xy[0] - e[0] / 2 - pad, self.xy[1] - e[1] / 2 - pad,
                self.xy[0] + e[0] / 2 + pad, self.xy[1] + e[1] / 2 + pad)

    def quat(self):
        return (0.0, 0.0, float(np.sin(self.yaw / 2)), float(np.cos(self.yaw / 2)))

    def spawn_z(self):
        return float(self.z) if self.z is not None else float(self.extent()[2]) / 2 + 0.005

    def config(self):
        cfg = {"type": "DatasetObject", "name": self.name, "category": self.category,
               "model": self.model, "position": [self.xy[0], self.xy[1], self.spawn_z()],
               "orientation": list(self.quat()), "fixed_base": self.fixed_base}
        if self.scale is not None:
            cfg["scale"] = list(self.scale)
        return cfg


#: The fridge retains its independently validated non-prehensile task pose.
ROW_Y = -1.60
ROW_APPROACH = -np.pi / 2
ROW_YAW = np.pi / 2

#: Task station catalog. Order matters only for readable deterministic object configs.
STATIONS = {s.key: s for s in [
    # West end. It is 0.74 m deep, twice anything else in the row, and its door sweeps 0.62 m out,
    # so it gets a 0.48 m gap to the dish rack instead of the row's usual 0.12 -- constraints 4, 6
    # and 7. Not `fixed_base`: a fixed-base articulation has immovable joints on this asset (probed
    # directly -- the door would not rotate at all under a commanded joint velocity with both drive
    # gains zeroed), which is what made the robot appear to push the door with nothing happening.
    # It is anchored by mass instead (`CloseFridgeTask._anchor_body`).
    Station("fridge", "fridge", "xyejdx", (-1.32, ROW_Y), yaw=ROW_YAW, approach=ROW_APPROACH,
            keepout=0.76, fixed_base=False, z=0.85, avoid=0.85, mass=400.0),
    # East-wall receptacle row. Root coordinates, rather than nominal AABB centres, are listed:
    # the metadata base_link_offset is nonzero for these assets. Actual AABB back x = 1.840.
    Station("dish_rack", "dish_rack", "kxuutl", (1.748438, -0.413814), yaw=np.pi,
            approach=0.0, keepout=0.60, scale=(1.0, 0.75, 1.0), z=0.482232),
    # An actual clothes airer, where a dark hollow bookcase used to stand in for one. Mapped as
    # solid occupancy of the COLLISION meshes: a slatted deck 0.750 above the AABB bottom, seven
    # rails running the rack's LONG axis between two end frames and spaced 50-125 mm across its
    # DEPTH, on a scissor stand, plus two fold-out wings that only reach full width at the crown.
    #
    # `yaw = +pi/2` breaks the row's "turn it to face the robot" rule deliberately, and the reason
    # is the GRIPPER, not the furniture. `_Rgrasp` freezes the wrist at the reset pose and the pads
    # close along world y (measured: the two fingertip links sit 100 mm apart in y), so the carried
    # towel is at most ~85 mm in y or there is no pinch at all. It therefore has to bridge the
    # rails along x, i.e. the rails must be spaced along x, i.e. they must RUN along y -- which is
    # the long axis lying along the wall. That is also how an airer is really parked.
    #
    # Every one of the three scales is a measured limit, not a look. This is a floor-standing
    # appliance dropped into a slot sized for a shallow wall receptacle, so it only fits at all
    # because each axis was cut to what its NEIGHBOURS allow -- each boundary found by planning
    # dish2rack, book2shelf and towel2rack through the real `NavMap` at a range of sizes:
    #
    #   x 0.42 (0.64 m long) -- the slot. 1.53 m of rack does not go between a dish rack and a
    #     bookcase 0.80 m apart; at native size it overlapped BOTH, by 1.63 and 0.37 m. This is the
    #     free axis: it is the rails' LENGTH, so cutting it leaves the pitch the towel bridges
    #     untouched. 0.42 leaves 7.5 cm to each neighbour.
    #   y 0.70 (0.41 m deep) -- dish2rack's park. `NavMap` erodes every block box by the 0.33 m
    #     chassis radius, and dish2rack parks at x = 1.084 with its y 14 mm inside this rack's y
    #     span, so the rack's front face has to stay east of 1.414. Blocked at 0.442 m deep, clear
    #     at 0.430; 0.407 keeps 23 mm of margin. Depth is the one scale that also compresses the
    #     rail pitch, 50-125 mm at native to 39-97 mm here, which the 240 mm towel still bridges.
    #   z 0.72 (0.76 m tall) -- book2shelf's carry height. `_nav` gives a station a carried-object
    #     clearance box when it stands taller than the carry, and book2shelf carries at 0.820, so
    #     anything over 0.790 earns a box that -- eroded like every other -- swallows book2shelf's
    #     own park. Blocked at 0.798 tall, clear at 0.787; 0.756 keeps 31 mm. It puts the deck at
    #     0.640, inside the 0.5-1.0 m band the arm works, and leaves the carried towel's underside
    #     (0.928) passing 17 cm over the crown.
    Station("towel_rack", "drying_rack", "rygebd", (1.6367, -1.047), yaw=np.pi / 2,
            approach=0.0, keepout=0.60, scale=(0.42, 0.70, 0.72), z=0.5018, top=0.756),
    # A genuinely ENCLOSED case: a 2 x 3 grid of cubbies with a solid back, sides, roof and
    # dividers, so placing the book is an insertion through one compartment's front opening and
    # not a drop onto an open plank. `ftzmow` -- what this station used to be, and still is next
    # door -- is two boards on wire brackets, open on every side; nothing about putting a book on
    # it reads as shelving a book. All 37 bookcase and 29 shelf models were rendered front-on and
    # probed by settling the scaled textbook on a (column, height) grid 0.10 m inside the front
    # face. This one is the best of the true cubby units: 0.311 m deep (the book is 0.072), and
    # its interior floors sit 0.045 / 0.383 / 0.716 m above the case bottom, so the TOP cubby
    # lands at 0.721 m -- inside the 0.5-1.0 m band the arm can work, next to the dish rack's
    # validated 0.77 m rail. Both columns hold the book to within 1 mm of where it is released and
    # evaluate `Inside`; the case CENTRE is the vertical divider and sheds it, hence the
    # `place_offset` in the spec. Native scale: every one of those numbers was measured at 1:1.
    # `z` stands the AABB bottom on the floor (the root sits 0.53 m above it).
    Station("book_shelf", "bookcase", "otwukr", (1.702668, -1.826439), yaw=np.pi,
            approach=0.0, keepout=0.60, z=0.533625),
    # Pulled out in front of the dining table, backrest toward the robot and directly ahead of the
    # reset left hand, so the whole task is one forward push with no navigation turn: descend onto
    # the backrest, grasp with both hands, and walk the chair in under the tabletop along world +x.
    #
    # The MODEL is chosen for the grasp, not for looks. `amgwaw` -- the room's own dining chair, and
    # what this station used to spawn -- has a BOWED back: mapped as solid occupancy at grasp
    # height, its crest sits 5 mm in front of the AABB back face at the centre and sweeps to 135 mm
    # at the horns, so at any span wide enough for two hands the limbs run roughly ALONG the jaw's
    # closing axis. Instrumented tick by tick, the descent then wedged instead of straddling: first
    # contact spun the chair through 25-33 deg of yaw before the grippers closed, and the right hand
    # was left holding air. `ocyvcb` has a STRAIGHT rail -- 20-30 mm thick at one constant x across
    # its whole 0.430 m width -- so both pads meet it square, at every y.
    # Measured at native scale it is 0.503 x 0.460 x 0.907, so from this mark its front edge starts
    # 0.19 m short of the table's near face at x=1.080 -- unmistakably a chair someone got up from
    # -- its rail sits at 0.877, a comfortable 0.118 m above the tabletop the hands follow it over,
    # and the tuck ends with 0.40 m of it under the tabletop. The lane the chair travels is clear of
    # table legs from the floor up to 0.70 m (probed as solid occupancy, not AABBs), so what stops
    # the chair is its own backrest meeting the table's near edge -- the stop a real tuck has.
    Station("chair", "straight_chair", "ocyvcb", (0.64, 0.40), yaw=ROBOT_YAW,
            approach=ROBOT_YAW, keepout=0.0, fixed_base=False),
]}

RECEPTACLE_KEYS = ("dish_rack", "towel_rack", "book_shelf")


def station_configs(keys=None):
    """DatasetObject configs for the requested station keys (all stations by default)."""
    selected = STATIONS.keys() if keys is None else keys
    return [STATIONS[key].config() for key in selected]


def station_boxes(exclude=(), pad=0.0, keys=None):
    """Footprints for `NavMap(block_boxes=...)`: floor the shipped map shows open but isn't."""
    selected = STATIONS.keys() if keys is None else keys
    return [STATIONS[key].footprint(pad) for key in selected if key not in exclude]


def keepouts(target=None, keys=None):
    """(x, y, r) circles the chassis must stay out of, for stations that are not being worked.

    Only stations with a non-zero `avoid` are listed -- see that field for why a blanket circle
    is wrong for the low wall-hugging receptacles.
    """
    selected = STATIONS.keys() if keys is None else keys
    return [(STATIONS[key].xy[0], STATIONS[key].xy[1], STATIONS[key].avoid)
            for key in selected if STATIONS[key].avoid > 0.0 and key != target]


def side_chairs(env):
    """The two dining-chair instances `CLEAR_CHAIRS` names, nearest first."""
    out = []
    for cat, at, _to in CLEAR_CHAIRS:
        cands = [o for o in env.env.scene.objects
                 if o.category == cat and not o.name.startswith("stn_")]
        if cands:
            out.append(min(cands, key=lambda o: np.linalg.norm(env.obj_pos(o)[:2] - np.array(at))))
    return out


def place_side_chairs(env, chairs=None):
    """Tuck the two dining chairs against the west wall, in EVERY task.

    They stand between the robot and the table's near edge -- the chassis parked at `SRC_PARK`
    overlaps one of them -- so they have to move. The destination is absolute rather than a shove
    relative to wherever the chair currently is, because a relative shove compounds across resets
    and, with the stations now in a row along the east side, landed one chair on top of them.
    `stn_chair` is excluded by name: it is the same category and must NOT be tucked away.
    """
    for chair, (_cat, _at, to) in zip(chairs if chairs is not None else side_chairs(env),
                                      CLEAR_CHAIRS):
        p = env.obj_pos(chair)
        chair.set_position_orientation(position=[to[0], to[1], float(p[2])],
                                       orientation=[0.0, 0.0, 0.0, 1.0])
        chair.keep_still()


def place_stations(env, keys=None):
    """Put every station back on its mark. Called at reset so the layout is bit-identical
    across tasks, seeds and episodes -- including any station a previous episode shoved."""
    selected = STATIONS.keys() if keys is None else keys
    for key in selected:
        s = STATIONS[key]
        obj = env.env.scene.object_registry("name", s.name)
        if obj is None:
            continue
        obj.set_position_orientation(position=[s.xy[0], s.xy[1], s.spawn_z()],
                                     orientation=list(s.quat()))
        if s.mass is not None:
            try:
                obj.root_link.mass = s.mass
            except Exception as exc:
                print(f"[layout] could not set {s.name} mass: {exc}", flush=True)
        if s.tint is not None:
            tint = np.asarray(s.tint, dtype=float)
            for material in obj.materials:
                material.diffuse_tint = tint
        try:
            obj.keep_still()
        except Exception:
            pass
