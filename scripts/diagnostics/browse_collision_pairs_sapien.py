#!/usr/bin/env python3
"""Browse WBC self-collision sphere pairs in SAPIEN, sorted by current distance.

This is a read-only diagnostic for the pair set that ``VegaWholeBodyIK`` actually keeps
after its cross-group and nominal-overlap filters. The SAPIEN window shows the robot and
the collision spheres at one posture; press Left/Right to step through pairs from closest
to farthest. The current pair is overlaid in gold and printed in the terminal.
Use ``--origin-comparison`` to overlay the raw pre-correction sphere centers in cyan,
making the WBC mesh-origin correction visible.

Run in the dexmate environment::

    /home/yixuan/miniforge3/envs/dexmate/bin/python \
        scripts/diagnostics/browse_collision_pairs_sapien.py

Controls:
    Right / D / N        next farther pair
    Left  / A / P        previous closer pair
    Home / 0             closest pair
    End                  farthest pair
    Q / Esc              quit

Useful non-viewer check::

    /home/yixuan/miniforge3/envs/dexmate/bin/python \
        scripts/diagnostics/browse_collision_pairs_sapien.py --list-only --top 20

Live teleop mode (``--interactive``): instead of one frozen posture, subscribe to the
EEF/head targets ``scripts/wbc_vr_leader.py`` publishes on ``vr/joints``, run the same
whole-body IK the follower runs, and animate the robot + collision spheres in SAPIEN as
you teleoperate -- the closest cross-group pair stays gold and its clearance prints
live. Start this first, then run the leader in another shell and grip-hold to engage::

    /home/yixuan/miniforge3/envs/dexmate/bin/python \
        scripts/diagnostics/browse_collision_pairs_sapien.py --interactive
    # ... and in another shell:
    /home/yixuan/miniforge3/envs/dexmate/bin/python scripts/wbc_vr_leader.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from omniteleop.follower import wbc_safety
from omniteleop.follower.sapien_env import prepare_sapien_render_env
from omniteleop.follower.whole_body_ik import (
    BASE_FRAME,
    HEAD_FRAME,
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    VegaWholeBodyIK,
    WBCConfig,
)

_GROUP_COLOR = {
    "body": [0.45, 0.45, 0.45, 0.18],
    "left": [0.15, 0.35, 1.0, 0.18],
    "right": [1.0, 0.20, 0.15, 0.18],
    None: [0.7, 0.7, 0.7, 0.18],
}
_GOLD = [1.0, 0.78, 0.05, 0.75]
_LINE = [1.0, 0.95, 0.25, 1.0]
_RAW_ORIGIN = [0.0, 0.95, 1.0, 0.25]
_ORIGIN_SHIFT_LINE = [0.0, 0.95, 1.0, 1.0]


@dataclass(frozen=True)
class SphereInfo:
    """One collision sphere with live world placement and metadata."""

    index: int
    name: str
    group: Optional[str]
    parent_joint_id: int
    parent_joint_name: str
    center: np.ndarray
    radius: float
    raw_center: Optional[np.ndarray] = None


@dataclass(frozen=True)
class PairInfo:
    """One kept collision pair, sorted by live surface distance."""

    rank: int
    geom_i: int
    geom_j: int
    distance: float
    adjacent: bool


def _load_q(args: argparse.Namespace, ik: VegaWholeBodyIK) -> tuple[np.ndarray, str]:
    """Resolve the configuration vector to browse and a human label."""
    if args.worst_from_hdf5:
        import h5py  # noqa: PLC0415

        with h5py.File(args.worst_from_hdf5, "r") as f:
            q_all = f["ik/q"][:]
            msd = f["ik/min_self_distance"][:]
            track = f["leader/tracking"][:] if "leader/tracking" in f else np.ones(len(msd), bool)
            held = f["ik/held"][:] if "ik/held" in f else np.zeros(len(msd), bool)
            mask = track & (~held)
            cand = np.where(mask)[0] if mask.any() else np.arange(len(msd))
            n = int(cand[np.argmin(msd[cand])])
            return q_all[n].astype(float), (
                f"worst frame {n} of {args.worst_from_hdf5} "
                f"(min_self_distance={msd[n] * 100:.2f}cm)"
            )
    if args.q_from_hdf5:
        import h5py  # noqa: PLC0415

        with h5py.File(args.q_from_hdf5, "r") as f:
            return f["ik/q"][args.frame].astype(float), (
                f"frame {args.frame} of {args.q_from_hdf5}"
            )
    return ik.nominal_q(), "nominal posture"


def _is_adjacent_pair(ik: VegaWholeBodyIK, joint_i: int, joint_j: int) -> bool:
    """True when two geometry parent joints are equal or direct parent/child."""
    return (
        joint_i == joint_j
        or ik.model.parents[joint_i] == joint_j
        or ik.model.parents[joint_j] == joint_i
    )


def _link_name_from_geom(name: str) -> str:
    """Return the source link name for a sphere geometry name."""
    return name.rsplit("_", 1)[0]


def _raw_origin_center(
    ik: VegaWholeBodyIK,
    geom,
    corrected_center: np.ndarray,
    offsets_by_link: dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    """Return the raw pre-correction world center for ``geom`` when it was shifted."""
    offset = offsets_by_link.get(_link_name_from_geom(geom.name))
    if offset is None:
        return None
    parent_pose = ik.configuration.data.oMi[int(geom.parentJoint)]
    return corrected_center - parent_pose.rotation @ offset


def _collect_spheres_and_pairs(
    ik: VegaWholeBodyIK,
    *,
    include_origin_comparison: bool = False,
) -> tuple[list[SphereInfo], list[PairInfo]]:
    """Collect live sphere placements and kept collision pairs sorted by surface gap."""
    sm = ik.collision_sphere_model
    sd = ik.collision_sphere_data
    if sm is None or sd is None:
        raise RuntimeError("collision-sphere model is not loaded")

    offsets_by_link = (
        wbc_safety.collision_sphere_mesh_origin_offsets(
            ik.config.urdf_path, ik.config.collision_spheres_urdf
        )
        if include_origin_comparison
        else {}
    )
    spheres: list[SphereInfo] = []
    for idx, go in enumerate(sm.geometryObjects):
        parent = int(go.parentJoint)
        center = np.asarray(sd.oMg[idx].translation, dtype=float).copy()
        spheres.append(
            SphereInfo(
                index=idx,
                name=go.name,
                group=wbc_safety.collision_group(go.name),
                parent_joint_id=parent,
                parent_joint_name=ik.model.names[parent],
                center=center,
                radius=float(go.geometry.radius),
                raw_center=_raw_origin_center(ik, go, center, offsets_by_link),
            )
        )

    pairs = []
    for k, pair in enumerate(sm.collisionPairs):
        si = spheres[pair.first]
        sj = spheres[pair.second]
        pairs.append(
            PairInfo(
                rank=k,
                geom_i=pair.first,
                geom_j=pair.second,
                distance=float(sd.distanceResults[k].min_distance),
                adjacent=_is_adjacent_pair(ik, si.parent_joint_id, sj.parent_joint_id),
            )
        )
    pairs.sort(key=lambda p: p.distance)
    pairs = [
        PairInfo(rank=i, geom_i=p.geom_i, geom_j=p.geom_j, distance=p.distance, adjacent=p.adjacent)
        for i, p in enumerate(pairs)
    ]
    return spheres, pairs


def _print_pair(
    pair: PairInfo,
    spheres: list[SphereInfo],
    total: int,
    *,
    show_origin_comparison: bool = False,
) -> None:
    """Print one pair status line with names, gap and adjacency."""
    a = spheres[pair.geom_i]
    b = spheres[pair.geom_j]
    adj = " adjacent" if pair.adjacent else ""
    print(
        f"[{pair.rank + 1:4d}/{total}] {pair.distance * 100:+7.2f}cm  "
        f"{a.name} ({a.group}, {a.parent_joint_name}) <-> "
        f"{b.name} ({b.group}, {b.parent_joint_name}){adj}",
        flush=True,
    )
    if show_origin_comparison:
        shifts = []
        for sphere in (a, b):
            if sphere.raw_center is None:
                continue
            shift = float(np.linalg.norm(sphere.center - sphere.raw_center))
            shifts.append(f"{sphere.name} raw->corrected {shift * 100:.2f}cm")
        if shifts:
            print("    origin shift: " + "; ".join(shifts), flush=True)
        else:
            print("    origin shift: none for this pair", flush=True)


def _print_table(
    pairs: list[PairInfo],
    spheres: list[SphereInfo],
    top: int,
    *,
    show_origin_comparison: bool = False,
) -> None:
    """Print the top-N sorted pairs without opening SAPIEN."""
    for pair in pairs[: max(0, top)]:
        _print_pair(
            pair,
            spheres,
            len(pairs),
            show_origin_comparison=show_origin_comparison,
        )


def _print_origin_comparison_summary(spheres: list[SphereInfo]) -> None:
    """Print a short legend and per-link raw-to-corrected shift summary."""
    max_shift_by_link: dict[str, float] = {}
    for sphere in spheres:
        if sphere.raw_center is None:
            continue
        link_name = _link_name_from_geom(sphere.name)
        shift = float(np.linalg.norm(sphere.center - sphere.raw_center))
        max_shift_by_link[link_name] = max(max_shift_by_link.get(link_name, 0.0), shift)

    if not max_shift_by_link:
        print("origin comparison: no mesh-origin sphere offsets were applied")
        return

    print("origin comparison: corrected spheres use normal colors/gold; raw")
    print("pre-correction centers are cyan ghosts with cyan raw->corrected lines.")
    for link_name, shift in sorted(max_shift_by_link.items()):
        print(f"  {link_name}: max raw->corrected shift {shift * 100:.2f}cm")


def _sapien_qpos(robot, ik: VegaWholeBodyIK, q: np.ndarray) -> np.ndarray:
    """Map the IK reduced q vector to SAPIEN active-joint order."""
    active = robot.get_active_joints()
    qpos = np.zeros(robot.dof)
    if robot.dof != len(active):
        raise ValueError(f"SAPIEN robot.dof={robot.dof}, active joints={len(active)}")
    for idx, joint in enumerate(active):
        name = joint.get_name()
        qidx = ik._idx_q.get(name)  # noqa: SLF001
        if qidx is None:
            continue
        value = float(q[qidx])
        limits = np.asarray(joint.get_limits()[0], dtype=float)
        if limits.shape == (2,) and np.all(np.isfinite(limits)):
            value = float(np.clip(value, limits[0], limits[1]))
        qpos[idx] = value
    return qpos


def _make_sphere_actor(scene, sapien, radius: float, color: list[float]):
    builder = scene.create_actor_builder()
    builder.add_sphere_visual(radius=radius, material=color)
    return builder.build_kinematic()


def _quat_wxyz_from_x_to(vec: np.ndarray) -> list[float]:
    """Quaternion rotating local +x onto ``vec``."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    v = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    v /= n
    rot, _ = Rotation.align_vectors([v], [[1.0, 0.0, 0.0]])
    qx, qy, qz, qw = rot.as_quat()
    return [float(qw), float(qx), float(qy), float(qz)]


def _make_segment_actor(scene, sapien, p0: np.ndarray, p1: np.ndarray, color: list[float]):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    builder = scene.create_actor_builder()
    builder.add_capsule_visual(
        sapien.Pose(),
        radius=0.009,
        half_length=max(length / 2.0, 1e-4),
        material=color,
    )
    actor = builder.build_kinematic()
    actor.set_pose(
        sapien.Pose(
            p=list((p0 + p1) * 0.5),
            q=_quat_wxyz_from_x_to(delta),
        )
    )
    return actor


def _build_scene(ik: VegaWholeBodyIK, q: np.ndarray, spheres: list[SphereInfo], args):
    """Build the SAPIEN scene, robot and base sphere actors."""
    prepare_sapien_render_env()
    import sapien  # noqa: PLC0415

    scene = sapien.Scene()
    scene.add_ground(args.ground_z)
    scene.set_ambient_light([0.55, 0.55, 0.55])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
    scene.add_directional_light([1, -1, -1], [0.25, 0.25, 0.25])

    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=args.camera_xyz[0], y=args.camera_xyz[1], z=args.camera_xyz[2])
    viewer.set_camera_rpy(r=0.0, p=args.camera_pitch, y=args.camera_yaw)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True
    robot = loader.load(ik.config.urdf_path)
    robot.set_qpos(_sapien_qpos(robot, ik, q))

    sphere_actors = []
    if not args.no_all_spheres:
        for sphere in spheres:
            actor = _make_sphere_actor(
                scene,
                sapien,
                sphere.radius,
                _GROUP_COLOR.get(sphere.group, _GROUP_COLOR[None]),
            )
            actor.set_pose(sapien.Pose(p=list(sphere.center)))
            sphere_actors.append(actor)
            if args.origin_comparison and sphere.raw_center is not None:
                raw_actor = _make_sphere_actor(scene, sapien, sphere.radius, _RAW_ORIGIN)
                raw_actor.set_pose(sapien.Pose(p=list(sphere.raw_center)))
                sphere_actors.append(raw_actor)
    return sapien, scene, viewer, robot, sphere_actors


class PairHighlighter:
    """Owns the current highlighted pair actors."""

    def __init__(
        self,
        scene,
        sapien,
        spheres: list[SphereInfo],
        pairs: list[PairInfo],
        *,
        origin_comparison: bool,
    ) -> None:
        self._scene = scene
        self._sapien = sapien
        self._spheres = spheres
        self._pairs = pairs
        self._origin_comparison = origin_comparison
        self._actors: list[object] = []

    def clear(self) -> None:
        """Remove the previous highlight actors from the scene."""
        for actor in self._actors:
            actor.remove_from_scene()
        self._actors = []

    def show(self, index: int) -> None:
        """Highlight and print the pair at ``index``."""
        self.clear()
        pair = self._pairs[index]
        a = self._spheres[pair.geom_i]
        b = self._spheres[pair.geom_j]
        for sphere in (a, b):
            if self._origin_comparison and sphere.raw_center is not None:
                raw_actor = _make_sphere_actor(
                    self._scene, self._sapien, sphere.radius, _RAW_ORIGIN
                )
                raw_actor.set_pose(self._sapien.Pose(p=list(sphere.raw_center)))
                self._actors.append(raw_actor)
                self._actors.append(
                    _make_segment_actor(
                        self._scene,
                        self._sapien,
                        sphere.raw_center,
                        sphere.center,
                        _ORIGIN_SHIFT_LINE,
                    )
                )
            actor = _make_sphere_actor(self._scene, self._sapien, sphere.radius, _GOLD)
            actor.set_pose(self._sapien.Pose(p=list(sphere.center)))
            self._actors.append(actor)
        self._actors.append(
            _make_segment_actor(self._scene, self._sapien, a.center, b.center, _LINE)
        )
        _print_pair(
            pair,
            self._spheres,
            len(self._pairs),
            show_origin_comparison=self._origin_comparison,
        )


def _key_down(viewer, *names: str) -> bool:
    window = getattr(viewer, "window", None)
    if window is None:
        return False
    for name in names:
        try:
            if window.key_down(name):
                return True
        except Exception:
            continue
    return False


def _viewer_loop(sapien, scene, viewer, highlighter: PairHighlighter, pairs: list[PairInfo],
                 start_index: int, rate: float) -> None:
    """Render loop with edge-triggered pair browsing."""
    index = int(np.clip(start_index, 0, len(pairs) - 1))
    highlighter.show(index)
    print("\nControls: Left/A/P previous, Right/D/N next, Home/0 closest, End farthest, Q/Esc quit")
    if highlighter._origin_comparison:  # noqa: SLF001
        print("Origin comparison: cyan ghosts are raw pre-correction sphere centers.")
    prev_next = prev_prev = prev_home = prev_end = False
    dt = 1.0 / rate

    try:
        while not viewer.closed:
            next_down = _key_down(viewer, "right", "arrowright", "d", "n")
            prev_down = _key_down(viewer, "left", "arrowleft", "a", "p")
            home_down = _key_down(viewer, "home", "0")
            end_down = _key_down(viewer, "end")
            if _key_down(viewer, "q", "esc", "escape"):
                break

            changed = False
            if next_down and not prev_next:
                index = min(len(pairs) - 1, index + 1)
                changed = True
            if prev_down and not prev_prev:
                index = max(0, index - 1)
                changed = True
            if home_down and not prev_home:
                index = 0
                changed = True
            if end_down and not prev_end:
                index = len(pairs) - 1
                changed = True
            if changed:
                highlighter.show(index)

            prev_next, prev_prev = next_down, prev_down
            prev_home, prev_end = home_down, end_down
            scene.update_render()
            viewer.render()
            time.sleep(dt)
    finally:
        highlighter.clear()
        viewer.close()


def _se3_mat(pose) -> np.ndarray:
    """4x4 matrix from a ``pin.SE3`` (``.homogeneous``) or a 4x4 array."""
    return np.asarray(pose.homogeneous if hasattr(pose, "homogeneous") else pose, dtype=float)


def _live_centers(ik: VegaWholeBodyIK) -> np.ndarray:
    """World centers (N, 3) of every collision sphere at the current configuration.

    Reads the live ``Configuration`` collision data, refreshed on every solve's
    ``integrate``/``update`` (the same ``oMg`` coal/_min_self_distance read).
    """
    sd = ik.collision_sphere_data
    return np.asarray([sd.oMg[k].translation for k in range(len(sd.oMg))], dtype=float)


def _set_robot_pose(robot, sapien, ik: VegaWholeBodyIK, base_z: float) -> None:
    """Place the SAPIEN robot at the solved planar base pose + actuated joints.

    The mobile base is a pinocchio planar root joint (not a SAPIEN active joint, since
    the URDF is loaded ``fix_root_link=True``), so the chassis pose is applied with
    ``set_root_pose`` and the arm/torso/head joints with ``set_qpos`` -- mirroring
    ``wbc_sapien_sim.SapienSimRobot``.
    """
    base = _se3_mat(ik.frame_pose(BASE_FRAME))
    yaw = float(np.arctan2(base[1, 0], base[0, 0]))
    half = yaw / 2.0
    robot.set_root_pose(
        sapien.Pose(p=[base[0, 3], base[1, 3], base_z],
                    q=[float(np.cos(half)), 0.0, 0.0, float(np.sin(half))])
    )
    robot.set_qpos(_sapien_qpos(robot, ik, ik.configuration.q))


def _run_interactive(ik: VegaWholeBodyIK, args: argparse.Namespace) -> None:
    """Live-visualize the collision model while ``wbc_vr_leader.py`` teleoperates.

    Subscribes to the leader's ``vr/joints`` EEF/head targets and runs the SAME
    whole-body IK the follower (``wbc_vr_record.py``) runs, then each frame moves the
    SAPIEN robot and every collision sphere onto the solved configuration. The closest
    cross-group pair is highlighted in gold and its signed clearance is printed, so you
    can watch the self-collision margin shrink as you teleoperate. This is open-loop /
    kinematic: no real robot or wheel dynamics, just the WBC's commanded configuration
    -- the one its own collision model reasons about.

    Run the leader in another shell and grip-hold to engage::

        /home/yixuan/miniforge3/envs/dexmate/bin/python scripts/wbc_vr_leader.py
    """
    if args.origin_comparison:
        raise SystemExit("--interactive cannot be combined with --origin-comparison")
    # Lazy heavy deps (dexcomm/zenoh + the shared control-loop helpers): keep the
    # read-only --list-only / static-browse paths free of the teleop stack.
    from omniteleop.wbc_stream import (  # noqa: PLC0415
        HeadTargetLowPassFilter,
        HeadTargetPlanarDeadbandFilter,
        TargetInterpolator,
        vr_to_ee_targets,
        vr_to_head_target,
    )
    from omniteleop.wbc_teleop import VRJointSubscriber, VRTeleopConfig  # noqa: PLC0415

    cfg = ik.config
    teleop = VRTeleopConfig.from_yaml()
    base_dur = 1.0 / teleop.cmd_rate if teleop.cmd_rate > 0 else 0.0

    # Nominal EE/head poses seed the interpolator/filters (the targets held until the
    # leader engages), exactly as the follower seeds them.
    left0 = _se3_mat(ik.frame_pose(LEFT_EE_FRAME))
    right0 = _se3_mat(ik.frame_pose(RIGHT_EE_FRAME))
    head0 = _se3_mat(ik.frame_pose(HEAD_FRAME))

    spheres, _pairs = _collect_spheres_and_pairs(ik)
    sapien, scene, viewer, robot, sphere_actors = _build_scene(ik, ik.configuration.q, spheres, args)
    base_z = float(robot.get_root_pose().p[2])

    sm = ik.collision_sphere_model
    pair_i = np.array([int(p.first) for p in sm.collisionPairs])
    pair_j = np.array([int(p.second) for p in sm.collisionPairs])
    radii = np.array([float(g.geometry.radius) for g in sm.geometryObjects])

    sub = VRJointSubscriber(namespace=args.namespace, name="browse_collision_interactive")
    sub.start()

    interp = TargetInterpolator(base_dur, left0, right0, head0)
    head_lpf = HeadTargetLowPassFilter(teleop.head_lpf_tau, head0)
    head_db = HeadTargetPlanarDeadbandFilter(
        head0,
        position_deadband=teleop.head_planar_pos_deadband,
        yaw_deadband=teleop.head_planar_yaw_deadband,
    )
    left_cmd, right_cmd, head_cmd = left0.copy(), right0.copy(), head0.copy()
    last_cmd_ns = -1
    prev_estop = True

    safe = float(cfg.self_collision_safe_dist)
    print(f"\n[interactive] subscribed to '{sub.topic}' (namespace={args.namespace!r}). "
          f"Run scripts/wbc_vr_leader.py and grip-hold to teleoperate.")
    print(f"[interactive] safe_dist={safe * 100:.1f}cm. Closest pair is gold; "
          "press Q/Esc or close the window to quit.")

    highlight: list = []
    last_solve = time.perf_counter()
    last_print = 0.0
    dt_render = 1.0 / args.rate if args.rate > 0 else 0.0
    try:
        while not viewer.closed:
            now = time.perf_counter()
            # Real elapsed time drives the open-loop integration (clamped against stalls),
            # so the model advances at the wall-clock rate the viewer actually renders.
            dt = float(np.clip(now - last_solve, 1e-4, 0.1))
            last_solve = now
            if _key_down(viewer, "q", "esc", "escape"):
                break

            vr = sub.latest
            estop = bool(vr.estop) if vr is not None else True
            if prev_estop and not estop:  # leader (re)calibrated: re-anchor to nominal
                ik.reset()
                interp.reset(left0, right0, head0)
                head_lpf.reset(head0)
                head_db.reset(head0)
                left_cmd, right_cmd, head_cmd = left0.copy(), right0.copy(), head0.copy()
                last_cmd_ns = -1
            prev_estop = estop

            left_cmd, right_cmd = vr_to_ee_targets(vr, left_cmd, right_cmd)
            head_cmd = vr_to_head_target(vr, head_cmd)
            if (vr is not None and vr.timestamp_ns != last_cmd_ns
                    and vr.left_ee_pose and vr.right_ee_pose):
                last_cmd_ns = vr.timestamp_ns
                interp.push(left_cmd, right_cmd, head_cmd, now=now, duration=base_dur)
            left_target, right_target, head_target_interp = interp.at(now)
            head_target = head_db.filter(head_lpf.filter(head_target_interp, dt))

            # Hold the nominal posture while the leader is static/e-stopped; solve only
            # once it engages (mirrors the follower's hold-on-estop behavior).
            if not estop:
                if cfg.head_mode == "ik":
                    ik.solve(left_target, right_target, dt, head_target=head_target)
                else:
                    head_joints = ik.solve_head(head_target, dt)
                    ik.solve(left_target, right_target, dt, head_joints=head_joints)

            # Move the rendered robot + every sphere onto the solved configuration.
            centers = _live_centers(ik)
            _set_robot_pose(robot, sapien, ik, base_z)
            for k, actor in enumerate(sphere_actors):
                actor.set_pose(sapien.Pose(p=list(centers[k])))

            # Closest cross-group pair (vectorized coal sphere narrowphase over the same
            # centers/pairs the WBC uses), highlighted in gold + a connecting line.
            diff = centers[pair_i] - centers[pair_j]
            dist = np.sqrt(np.einsum("ij,ij->i", diff, diff)) - (radii[pair_i] + radii[pair_j])
            kmin = int(np.argmin(dist))
            mind = float(dist[kmin])
            gi, gj = int(pair_i[kmin]), int(pair_j[kmin])
            for actor in highlight:
                actor.remove_from_scene()
            highlight = []
            for g in (gi, gj):
                gold = _make_sphere_actor(scene, sapien, float(radii[g]), _GOLD)
                gold.set_pose(sapien.Pose(p=list(centers[g])))
                highlight.append(gold)
            highlight.append(_make_segment_actor(scene, sapien, centers[gi], centers[gj], _LINE))

            if now - last_print >= 0.2:
                last_print = now
                a, b = spheres[gi], spheres[gj]
                flag = " !! BELOW SAFE" if mind < safe else ""
                stage = "teleop" if not estop else "static"
                print(f"[{stage:6s}] min self-dist {mind * 100:+6.2f}cm{flag}  "
                      f"{a.name} ({a.group}) <-> {b.name} ({b.group})        ",
                      end="\r", flush=True)

            scene.update_render()
            viewer.render()
            sleep = dt_render - (time.perf_counter() - now)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        for actor in highlight:
            actor.remove_from_scene()
        sub.close()
        viewer.close()
        print("\n[interactive] stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--q-from-hdf5", default=None, help="debug HDF5 to read ik/q from.")
    parser.add_argument("--frame", type=int, default=0, help="frame index for --q-from-hdf5.")
    parser.add_argument(
        "--worst-from-hdf5",
        default=None,
        help="debug HDF5: choose the non-held frame with smallest ik/min_self_distance.",
    )
    parser.add_argument("--start", type=int, default=1,
                        help="1-based sorted pair index to show first (default: 1, closest).")
    parser.add_argument("--top", type=int, default=20, help="rows printed by --list-only.")
    parser.add_argument("--list-only", action="store_true",
                        help="print sorted pairs and exit without opening SAPIEN.")
    parser.add_argument("--interactive", action="store_true",
                        help="live mode: subscribe to wbc_vr_leader.py's vr/joints targets, "
                             "run the follower's whole-body IK, and animate the robot + "
                             "collision spheres in SAPIEN while you teleoperate (the closest "
                             "pair stays highlighted in gold). Ignores the static q sources "
                             "(--q-from-hdf5 / --worst-from-hdf5); incompatible with "
                             "--origin-comparison / --list-only.")
    parser.add_argument("--namespace", default="",
                        help="Zenoh namespace for --interactive (must match the leader; "
                             "default empty).")
    parser.add_argument("--no-all-spheres", action="store_true",
                        help="only draw the current highlighted pair, not every sphere.")
    parser.add_argument("--origin-comparison", action="store_true",
                        help="overlay raw pre-correction sphere centers from the sphere URDF "
                             "against the corrected WBC centers.")
    parser.add_argument("--rate", type=float, default=100.0, help="viewer render rate in Hz.")
    parser.add_argument("--ground-z", type=float, default=-0.08)
    parser.add_argument("--camera-xyz", type=float, nargs=3, default=[2.2, -1.4, 1.45])
    parser.add_argument("--camera-yaw", type=float, default=2.55)
    parser.add_argument("--camera-pitch", type=float, default=-0.42)
    args = parser.parse_args()

    if args.interactive and args.list_only:
        parser.error("--interactive and --list-only are mutually exclusive")

    ik = VegaWholeBodyIK(WBCConfig())
    if args.interactive:
        ik.reset()  # nominal start; the teleop stream then drives the configuration live
        _run_interactive(ik, args)
        return
    q, label = _load_q(args, ik)
    if q.shape != (ik.model.nq,):
        raise ValueError(f"q has shape {q.shape}, expected ({ik.model.nq},)")
    ik.reset(q)
    spheres, pairs = _collect_spheres_and_pairs(
        ik, include_origin_comparison=args.origin_comparison
    )
    if not pairs:
        raise RuntimeError("no kept collision pairs to browse")

    print(f"collision pairs for {label}")
    warn = getattr(ik.config, "self_collision_warn", float("nan"))
    warn_text = f"{warn * 100:.1f}cm" if np.isfinite(warn) else "n/a"
    print(
        f"spheres={len(spheres)} kept_pairs={len(pairs)} "
        f"nominal_pair_keep_dist={ik.config.nominal_pair_keep_dist * 100:.1f}cm "
        f"safe_dist={ik.config.self_collision_safe_dist * 100:.1f}cm "
        f"warn={warn_text}"
    )
    if args.origin_comparison:
        _print_origin_comparison_summary(spheres)

    if args.list_only:
        _print_table(
            pairs,
            spheres,
            args.top,
            show_origin_comparison=args.origin_comparison,
        )
        return

    sapien, scene, viewer, _robot, _sphere_actors = _build_scene(ik, q, spheres, args)
    highlighter = PairHighlighter(
        scene,
        sapien,
        spheres,
        pairs,
        origin_comparison=args.origin_comparison,
    )
    _viewer_loop(sapien, scene, viewer, highlighter, pairs, args.start - 1, args.rate)


if __name__ == "__main__":
    main()
