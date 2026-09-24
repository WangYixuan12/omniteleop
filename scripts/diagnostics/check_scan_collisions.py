#!/usr/bin/env python3
"""Vet every pose in a scan pose file for self-collision BEFORE --execute.

record_arm_scan.py commands robot.set_joint_pos() directly, so the whole-body QP's
SelfCollisionBarrier and the reactive SafetyGate are NOT in the loop. This checks the
same metric they use -- VegaWholeBodyIK.self_collision_distance() over the kept
cross-group sphere pairs -- at each commanded waypoint AND along the path the executor
actually takes between consecutive waypoints.

That path is NOT a straight line through joint space. record_arm_scan.interp_to()
iterates ``for name, target in resolved.items()`` over COMMANDABLE order
(torso, head, left_arm, right_arm), gliding each component to its target BEFORE
starting the next. The trajectory is therefore an axis-aligned staircase, and its
corners -- states where one component has arrived and the next has not yet moved --
are exactly the configurations a simultaneous "diagonal" interpolation never visits.
Vetting the diagonal and then executing the staircase checks a path the robot does
not take. ``--path`` selects which to sweep; ``sequential`` is the executor's and is
the default. Exit status is 1 when the swept worst distance is below the floor, so
this can gate --execute instead of only printing.

Uncommanded joints hold their MEASURED value, which is the point: the right arm is
never commanded by the left-arm-only pose file, so it stays where the hardware has it.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

ap = argparse.ArgumentParser()
ap.add_argument("--poses", default="scripts/scan_poses_left_arm.json")
ap.add_argument("--config", default="src/omniteleop/follower/wbik.yaml")
ap.add_argument("--steps", type=int, default=61, help="samples per segment")
ap.add_argument("--measured", default=None,
                help="JSON {comp: [q...]}; omit to read live from the robot")
ap.add_argument("--path", choices=("sequential", "diagonal", "both"), default="sequential",
                help="sequential = the component-at-a-time staircase interp_to actually "
                     "walks (default); diagonal = simultaneous interpolation (what this "
                     "script used to check, and what the robot never does)")
ap.add_argument("--tip-urdf", default="~/yixuan_utilities/src/yixuan_utilities/assets/"
                "robot/vega-urdf/vega_with_robotiq_wrist_cam.urdf",
                help="URDF carrying the L/R_fingertip_pos/neg links, used to measure the "
                     "printed fingertips from their real mesh instead of the swept-volume "
                     "proxy spheres. --no-tip-mesh falls back to the spheres.")
ap.add_argument("--no-tip-mesh", action="store_true",
                help="do not use the fingertip meshes; fall back to the tip_* proxy "
                     "spheres (a swept-volume envelope, far too conservative for a path "
                     "gate -- it reports overlap where the real clearance is ~60 mm)")
ap.add_argument("--strict", action="store_true",
                help="also exit nonzero when the worst distance is merely below "
                     "safe_dist (the 'tight, review' band)")
args = ap.parse_args()

GROUPS = {"torso": ["torso_j1", "torso_j2", "torso_j3"],
          "head": ["head_j1", "head_j2", "head_j3"],
          "left_arm": [f"L_arm_j{i}" for i in range(1, 8)],
          "right_arm": [f"R_arm_j{i}" for i in range(1, 8)]}

if args.measured:
    measured = json.loads(Path(args.measured).read_text())
else:
    from dexcontrol.robot import Robot
    r = Robot()
    try:
        measured = {c: np.asarray(getattr(r, c).get_joint_pos(), float).ravel().tolist()
                    for c in GROUPS}
    finally:
        r.shutdown()
print("measured now:")
for c, v in measured.items():
    print(f"  {c:10s} {np.round(v, 4).tolist()}")

import omniteleop.follower.whole_body_ik as _wbik
from omniteleop.follower.whole_body_ik import VegaWholeBodyIK

# ---- fingertip geometry ------------------------------------------------------------
# ROBOTIQ_PROXY_SPHERES contains two different kinds of thing. palm/finger_pos/finger_neg
# model the 2F-85 BODY and are unchanged and previously validated. The tip_* spheres added
# 2026-09-18 for the printed ALOHA/UMI fingertips are a SWEPT-VOLUME envelope: r = 55 mm at
# x = +/-41 mm, covering the tip at EVERY gripper opening, because the live WBC barrier does
# not know the commanded closure. That is correct for the barrier and wrong here, where the
# commanded configuration is fully determined -- used as a path gate it reported -40 mm of
# overlap on a segment whose real fingertip clearance is 60 mm.
#
# So: keep the body proxies in the sphere model, drop the tip_* ones, and measure the tips
# from their actual mesh, posed through the URDF that actually carries the fingertip links.
# The mesh is replaced by its CONVEX HULL (135 verts vs 5458): it encloses the part, so the
# distance is conservative in the safe direction, and it is ~40x cheaper.
_BODY_ONLY = tuple(s for s in _wbik.ROBOTIQ_PROXY_SPHERES
                   if not s[0].startswith("tip_"))
tip_model = None
if not args.no_tip_mesh:
    import trimesh
    tip_urdf = Path(args.tip_urdf).expanduser()
    try:
        _tm = pin.buildModelFromUrdf(str(tip_urdf))
        _frames = [f"{s}_fingertip_{g}" for s in "LR" for g in ("pos", "neg")]
        if not all(_tm.existFrame(f) for f in _frames):
            raise ValueError(f"{tip_urdf.name} has no L/R_fingertip_pos/neg links")
        _hull = trimesh.load(tip_urdf.parent / "meshes/collision/robotiq_fingertip_umi.obj",
                             force="mesh").convex_hull
        tip_model = dict(model=_tm, data=_tm.createData(), frames=_frames,
                         V=np.asarray(_hull.vertices, float),
                         side={f: ("left" if f[0] == "L" else "right") for f in _frames})
        _wbik.ROBOTIQ_PROXY_SPHERES = _BODY_ONLY      # before the IK builds its model
        print(f"\nfingertips measured from {tip_urdf.name} "
              f"({len(_hull.vertices)}-vertex convex hull); tip_* proxy spheres dropped")
    except Exception as exc:
        print(f"\nWARNING: fingertip meshes unavailable ({exc}); falling back to the "
              f"tip_* proxy spheres, which are a swept-volume envelope and will read far "
              f"tighter than the real geometry")

ik = VegaWholeBodyIK(config_path=args.config)
cfg = ik.config
floor, safe = cfg.self_collision_floor, cfg.self_collision_safe_dist
warn, keep = cfg.self_collision_warn_dist, cfg.nominal_pair_keep_dist
npairs = len(ik.collision_sphere_model.collisionPairs) if ik.collision_sphere_model else 0
print(f"\nurdf   {cfg.urdf_path}")
print(f"pairs  {npairs} kept cross-group sphere pairs "
      f"(dropped those < {keep} m at nominal)")
print(f"limits floor {floor} m | CBF safe_dist {safe} m | warn {warn} m")

idx = ik._idx_q
_sm = ik.collision_sphere_model
_sphere_data = _sm.createData() if _sm is not None else None
from omniteleop.follower.wbc_safety import collision_group as _cg
sphere_groups = [_cg(g.name) for g in _sm.geometryObjects] if _sm is not None else []
sphere_radii = np.array([g.geometry.radius for g in _sm.geometryObjects]) if _sm is not None else np.zeros(0)
sphere_centers = np.zeros((len(sphere_groups), 3))
if tip_model is not None and _sm is None:
    tip_model = None


def tip_clearance(joint_vals: dict) -> float:
    """Min distance from either fingertip hull to any other collision group."""
    tm, td = tip_model["model"], tip_model["data"]
    q = pin.neutral(tm)
    for name, val in joint_vals.items():
        jid = tm.getJointId(name)
        if jid < tm.njoints:
            q[tm.idx_qs[jid]] = val
    pin.forwardKinematics(tm, td, q)
    pin.updateFramePlacements(tm, td)
    from omniteleop.common.convex_clearance import hull, distance
    import coal
    if "hull" not in tip_model:
        tip_model["hull"] = hull(tip_model["V"])
    geometry = tip_model["hull"]
    transforms = {f: coal.Transform3s(td.oMf[tm.getFrameId(f)].rotation,
                                    td.oMf[tm.getFrameId(f)].translation)
                  for f in tip_model["frames"]}
    best = np.inf
    for f in tip_model["frames"]:
        for j, group in enumerate(sphere_groups):
            if group is not None and group != tip_model["side"][f]:
                best = min(best, distance(geometry, transforms[f], coal.Sphere(sphere_radii[j]),
                                         coal.Transform3s(sphere_centers[j])))
    for left in ("L_fingertip_pos", "L_fingertip_neg"):
        for right in ("R_fingertip_pos", "R_fingertip_neg"):
            best = min(best, distance(geometry, transforms[left], geometry, transforms[right]))

    return best


def distance(joint_vals: dict) -> float:
    """Self-collision distance: sphere model, plus the real fingertips when available."""
    q = q_from(joint_vals)
    d = ik.self_collision_distance(q)
    if tip_model is None:
        return d
    pin.updateGeometryPlacements(ik.model, ik.data, ik.collision_sphere_model,
                                 _sphere_data, q)
    global sphere_centers
    sphere_centers = np.array([_sphere_data.oMg[i].translation
                               for i in range(len(sphere_groups))])
    return min(d, tip_clearance(joint_vals))


def q_from(joint_vals: dict) -> np.ndarray:
    q = ik.nominal_q().copy()
    for name, val in joint_vals.items():
        if name not in idx:
            raise KeyError(f"{name} not in model")
        q[idx[name]] = val
    return q


# Start from the measured state; each pose overrides only the joints it names.
state = {}
for c, names in GROUPS.items():
    for n, v in zip(names, measured[c]):
        state[n] = float(v)

poses = json.loads(Path(args.poses).read_text())
waypoints = [("MEASURED START", dict(state), set())]
for p in poses:
    commanded = set()
    for comp, names in GROUPS.items():
        if comp in p:
            commanded.add(comp)
            vals = np.asarray(p[comp], float).ravel()
            if vals.size != len(names):
                raise ValueError(f"{p['label']}: {comp} has {vals.size}, need {len(names)}")
            for n, v in zip(names, vals):
                state[n] = float(v)
    waypoints.append((p["label"], dict(state), commanded))

names_sorted = [n for names in GROUPS.values() for n in names]
print(f"\n{'waypoint':18s} {'min dist (m)':>13s}   verdict")
worst = np.inf
worst_where = ""
for label, st, _cmd in waypoints:
    d = distance(st)
    verdict = ("COLLISION" if d <= 0 else "BELOW FLOOR" if d < floor
               else "below safe_dist" if d < safe else "below warn" if d < warn else "ok")
    print(f"{label:18s} {d:13.4f}   {verdict}")
    if d < worst:
        worst, worst_where = d, f"waypoint {label}"

COMMANDABLE = ("torso", "head", "left_arm", "right_arm")


def sweep(a_state, b_state, moved, mode):
    """Worst distance and where, along one segment, for one path convention."""
    dmin, where = np.inf, ""
    if mode == "diagonal":
        a = np.array([a_state[n] for n in names_sorted])
        b = np.array([b_state[n] for n in names_sorted])
        for t in np.linspace(0.0, 1.0, max(args.steps, int(np.ceil(np.max(np.abs(b - a)) / .01)) + 1)):
            st = dict(zip(names_sorted, a + t * (b - a)))
            d = distance(st)
            if d < dmin:
                dmin, where = d, f"t={t:.2f}"
        return dmin, where
    # Sequential: each commanded component completes before the next one starts, so the
    # state carries forward. Components the pose does not name never move at all.
    st = dict(a_state)
    for comp in COMMANDABLE:
        if comp not in moved:
            continue
        jn = GROUPS[comp]
        a = np.array([a_state[n] for n in jn])
        b = np.array([b_state[n] for n in jn])
        if np.allclose(a, b):
            continue
        for t in np.linspace(0.0, 1.0, max(args.steps, int(np.ceil(np.max(np.abs(b - a)) / .01)) + 1)):
            for n, v in zip(jn, a + t * (b - a)):
                st[n] = float(v)
            d = distance(st)
            if d < dmin:
                dmin, where = d, f"{comp} t={t:.2f}"
        for n, v in zip(jn, b):          # committed before the next component moves
            st[n] = float(v)
    return dmin, where


modes = ("sequential", "diagonal") if args.path == "both" else (args.path,)
for mode in modes:
    print(f"\n{'segment':40s} {'min dist (m)':>13s}   where            verdict"
          f"   [{mode} path]")
    for (la, sa, _ma), (lb, sb, mb) in zip(waypoints[:-1], waypoints[1:]):
        dmin, where = sweep(sa, sb, mb, mode)
        if not np.isfinite(dmin):
            print(f"{la + ' -> ' + lb:40s} {'(no motion)':>13s}")
            continue
        verdict = ("COLLISION" if dmin <= 0 else "BELOW FLOOR" if dmin < floor
                   else "below safe_dist" if dmin < safe else "below warn" if dmin < warn
                   else "ok")
        seg = f"{la} -> {lb}"
        print(f"{seg:40s} {dmin:13.4f}   {where:16s} {verdict}")
        if mode == args.path or args.path == "both":
            if dmin < worst:
                worst, worst_where = dmin, f"segment {seg} ({mode}, {where})"

print(f"\nWORST OVERALL {worst:.4f} m  ({worst_where})")
unsafe = worst < floor
tight = worst < safe
print("VERDICT:", "UNSAFE -- do not run --execute" if unsafe
      else "tight, review" if tight else "SAFE")
if unsafe or (tight and args.strict):
    raise SystemExit(1)
