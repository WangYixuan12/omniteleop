"""Fast regression check for the scripted Robotiq pregrasp path."""

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as R

from sim_eval import base_ctrl
from sim_eval.tasks import layout
from sim_eval.tasks.long_horizon import (
    BOOK_FILLERS, BOOK_FILLER_QUAT, LH_TASKS, PLACE_STANDOFF, SPEC_BY_NAME,
    LongHorizonSequence, _box2cloth_route, rot_y)
from sim_eval.tasks.mobile_pick_place_task import MobilePickPlaceTask
from sim_eval.tasks.nonprehensile import CloseFridgeTask, PushChairTask
from sim_eval.tasks.pick_place_task import PickPlaceTask, rot_z


class _Obj:
    def __init__(self, pos, extent=(0.06, 0.06, 0.06), top=0.84):
        self.pos = np.asarray(pos, dtype=float)
        self.aabb_extent = np.asarray(extent, dtype=float)
        self.aabb = (np.zeros(3), np.array([0.0, 0.0, top]))


class _FakeEnv:
    grip_open = 1.0
    grip_close = -1.0
    action_hz = 100.0

    def __init__(self, apple):
        self._apple = apple
        self._left = np.eye(4)
        self._left[:3, 3] = [1.02, 0.72, 0.88]
        self._right = np.eye(4)
        self._right[:3, 3] = [1.02, 0.25, 0.88]
        self._base = np.eye(4)
        self._base[:3, 3] = [0.58, 0.40, 0.03]
        self._head = np.eye(4)
        self._head[:3, 3] = [0.20, 0.40, 1.40]
        self._poses = {
            "L_ee": self._left,
            "R_ee": self._right,
            "base": self._base,
            "zed_depth_frame": self._head,
        }
        self.robot = SimpleNamespace(links={"zed_depth_frame": object()})
        self._finger_offset = np.array([0.15, 0.0, 0.0])

    def link_pose(self, name):
        return self._poses[name].copy()

    def obj_pos(self, obj):
        return obj.pos.copy()

    def finger_grasp_point(self, arm):
        return self._left[:3, :3] @ self._finger_offset + self._left[:3, 3]

    def is_grasping(self, arm):
        return getattr(self, "_grasped", None)


def test_precontact_grasp_point_approaches_apple_monotonically():
    apple = _Obj([1.25, 0.72, 0.79])
    bowl = _Obj([1.22, 0.55, 0.79])
    env = _FakeEnv(apple)
    task = PickPlaceTask()
    task.apple = apple
    task.bowl = bowl
    task.expert_reset(env)

    finger_points = [env.finger_grasp_point("left")]
    while True:
        cmd = task.expert_step(env, None)
        if cmd.phase not in {"pregrasp", "approach"}:
            break
        finger_points.append(
            cmd.left_target[:3, :3] @ task._grasp_off_local + cmd.left_target[:3, 3]
        )

    distances = np.linalg.norm(np.asarray(finger_points) - apple.pos, axis=1)
    assert np.all(np.diff(distances) <= 1e-8)


def test_actions_do_not_feed_back_one_measured_base_pose():
    """Measured q belongs in the WBC solve, not in only one arm's action target."""
    apple_a = _Obj([1.25, 0.72, 0.79])
    apple_b = _Obj([1.25, 0.72, 0.79])
    env_a, env_b = _FakeEnv(apple_a), _FakeEnv(apple_b)
    tasks = [MobilePickPlaceTask(), MobilePickPlaceTask()]
    for task, env, apple in zip(tasks, (env_a, env_b), (apple_a, apple_b)):
        task.apple = apple
        task.bowl = _Obj([1.22, 0.55, 0.79])
        task.expert_reset(env)

    # Everything commanded is identical; only one measured chassis sample jumps.
    env_b._base[:2, 3] += [0.10, -0.07]
    cmd_a, cmd_b = (task.expert_step(env, None) for task, env in zip(tasks, (env_a, env_b)))
    np.testing.assert_allclose(cmd_a.left_target, cmd_b.left_target)
    np.testing.assert_allclose(cmd_a.right_target, cmd_b.right_target)
    np.testing.assert_allclose(cmd_a.head_target, cmd_b.head_target)


class _VisibleNav:
    @staticmethod
    def route(a, b):
        return np.stack([a, b])

    @staticmethod
    def visible(_a, _b):
        return True

    @staticmethod
    def free(_x, _y):
        return True


def test_box2cloth_route_is_back_then_right_then_forward_without_yaw():
    route = _box2cloth_route(_VisibleNav(), [0.56, 0.40], [1.12, -0.78])
    expected = np.array([
        [0.56, 0.40],
        [0.20, 0.40],
        [0.20, -0.78],
        [1.12, -0.78],
    ])
    np.testing.assert_allclose(route, expected)
    delta = np.diff(route, axis=0)
    assert delta[0, 0] < 0.0 and delta[0, 1] == 0.0
    assert delta[1, 0] == 0.0 and delta[1, 1] < 0.0
    assert delta[2, 0] > 0.0 and delta[2, 1] == 0.0


def test_translation_only_shaper_clears_stale_yaw_before_axis_selection():
    command, anchor, axis = base_ctrl.shape_project_twist(
        np.array([0.20, 0.05, 0.80]),
        np.array([0.00, 0.00, 0.30]),
        0.01,
        deadband_lin=0.02,
        deadband_ang=0.04,
        max_lin_speed=0.45,
        max_ang_speed=0.90,
        max_lin_accel=0.40,
        max_ang_accel=0.80,
        post_linear_deadband=0.02,
        post_angular_deadband=0.12,
        enable_single_axis=True,
        xy_max_vel=1.0,
        yaw_max_vel=1.0,
        single_axis_deadband=0.001,
        dispatch_single_axis_deadband=0.001,
        single_axis_hysteresis_ratio=0.2,
        prev_axis=2,
        base_dofs="xy",
        allow_yaw_hold=False,
    )
    assert anchor[2] == 0.0
    assert command[2] == 0.0
    assert axis in (0, 1)


def test_tasks_spawn_only_the_furniture_they_use():
    receptacles = {"stn_dish_rack", "stn_towel_rack", "stn_book_shelf"}
    stocked_books = {"lh_book_filler_0", "lh_book_filler_1"}
    for cls in LH_TASKS:
        task = cls()
        configs = task.object_configs()
        names = {cfg["name"] for cfg in configs}
        # ...plus a liner where the spec puts one on its station (dish2rack lays a tray across the
        # rack's rails and places onto that -- see LHSpec.liner).
        liner = {f"lh_liner_{task.SPEC.station}"} if task.SPEC.liner is not None else set()
        assert names == {"lh_obj", *receptacles} | liner | stocked_books
        assert "stn_fridge" not in names
        assert task.BASE_DOFS == "xy"
        # Must NOT be overridden: it has to inherit the shared vr_teleop.base_yaw_hold_in_xy so the
        # odometry yaw regulator runs, exactly as on hardware. See the note on the class.
        assert not hasattr(task, "BASE_YAW_HOLD_IN_XY")
        # Receptacles load above the room so they cannot collide with the original dining chairs
        # before task.reset() tucks those chairs away, then place_stations() uses the final poses.
        by_name = {cfg["name"]: cfg for cfg in configs}
        assert all(by_name[name].get("fixed_base") is True for name in stocked_books)
        for key in layout.RECEPTACLE_KEYS:
            station = layout.STATIONS[key]
            np.testing.assert_allclose(by_name[station.name]["position"][:2], station.xy)
            assert np.isclose(by_name[station.name]["position"][2], station.spawn_z() + 3.0)
    assert [cfg["name"] for cfg in CloseFridgeTask().object_configs()] == ["stn_fridge"]
    assert [cfg["name"] for cfg in PushChairTask().object_configs()] == ["stn_chair"]


def test_dish_segment_metadata_stays_aligned_with_no_turn():
    task = next(cls() for cls in LH_TASKS if cls.name == "dish2rack")
    task.apple = _Obj([1.20, 0.74, 0.80])
    station = layout.STATIONS["dish_rack"]
    task.bowl = _Obj([*station.xy, 0.60], top=0.80)
    task._start_center = np.array([0.28, 0.76, 0.88])
    task._nav = lambda _env, **_kwargs: _VisibleNav()
    env = _FakeEnv(task.apple)
    env._base[:3, 3] = [-0.35, 0.40, 0.03]

    segs = task._build_segments(env)
    phases = [seg[0] for seg in segs]
    assert phases == [
        "pregrasp", "approach", "settle", "close", "lift",
        "transit0", "transit1", "transit2",
        "arrive", "carry", "place", "release", "retreat",
    ]
    assert len(segs) == len(task._headings) == len(task._standoffs)
    assert task._headings == [0.0] * len(segs)
    assert task._standoffs[:8] == [1.0] * 8
    assert task._standoffs[8:] == [PLACE_STANDOFF] * 5
    assert task._last_base_cmd is None
    assert task._last_left_cmd is None
    assert task._grasp_delta is None
    drop = task._release_point(env)
    # A tray spans the rails and IS the receptacle (LHSpec.liner), so the drop is its centre with
    # no lane offset to hit -- `place_offset` is zero and `support_z_from_bottom` is None, which
    # makes the support the liner's own AABB top. `env.obj_pos(task.bowl)` is the liner here.
    np.testing.assert_allclose(drop[:2], np.asarray(station.xy))
    assert drop[0] < 1.764, "the dish drop must stay inside the measured left-arm reach limit"
    assert np.isclose(drop[2], 0.80 + 0.03 + 0.004)   # fake receptacle top + apple half + dz
    assert SPEC_BY_NAME["dish2rack"].liner["category"] == "tray"
    # The wall rack's working rail should be midway between the too-low 0.72 m
    # trial and the 0.82 m trial that saturated L_arm_j4 throughout transit.
    assert np.isclose(station.z, 0.482232)


def test_long_horizon_left_and_right_targets_have_identical_planar_increments():
    """Both EEF commands ride one commanded body frame; neither stream may stutter alone."""
    task = next(cls() for cls in LH_TASKS if cls.name == "dish2rack")
    task.apple = _Obj([1.20, 0.74, 0.80])
    station = layout.STATIONS["dish_rack"]
    task.bowl = _Obj([*station.xy, 0.60], top=0.80)
    task._nav = lambda _env, **_kwargs: _VisibleNav()
    env = _FakeEnv(task.apple)
    env._base[:3, 3] = [-0.35, 0.40, 0.03]
    task.expert_reset(env)
    task._clock_step = lambda _env: 1.0

    previous = task.expert_step(env, None)
    for _ in range(sum(seg[3] for seg in task._segs) + 1):
        current = task.expert_step(env, None)
        left_step = current.left_target[:2, 3] - previous.left_target[:2, 3]
        right_step = current.right_target[:2, 3] - previous.right_target[:2, 3]
        np.testing.assert_allclose(left_step, right_step, atol=1e-12)
        previous = current
        if current.done:
            break
    assert current.done


def test_long_horizon_clock_is_independent_of_tracking_error():
    """Scripted targets advance steadily instead of copying human move/pause teleoperation."""
    task = next(cls() for cls in LH_TASKS if cls.name == "dish2rack")
    task._last_head_cmd = np.array([1.0, 0.0, 1.4])
    task._last_base_cmd = np.array([1.0, 0.0])
    task._last_left_cmd = np.eye(4)
    task._head_link = "zed_depth_frame"
    task._segs = [("place", np.zeros(3), -1.0, 100)]
    task._seg_i = 0
    env = _FakeEnv(_Obj([1.20, 0.74, 0.80]))
    env._head[:3, 3] = task._last_head_cmd
    env._base[:2, 3] = [0.0, 0.0]
    env._left[:3, 3] = [5.0, 5.0, 5.0]

    rates = [task._clock_step(env) for _ in range(50)]

    np.testing.assert_allclose(rates, 1.0)


def test_long_horizon_calibrates_destination_to_the_attached_object():
    """Placement endpoints compensate the one observed sticky-grasp transform."""
    task = next(cls() for cls in LH_TASKS if cls.name == "dish2rack")
    task.apple = _Obj([1.20, 0.74, 0.80])
    task._grasp_delta = None
    task._segs = [
        ("transit0", np.array([0.5, 0.4, 0.9]), -1.0, 10),
        ("arrive", np.array([1.0, 0.0, 0.9]), -1.0, 10),
        ("place", np.array([1.0, 0.0, 0.8]), -1.0, 10),
        ("release", np.array([1.0, 0.0, 0.8]), 1.0, 10),
    ]
    env = _FakeEnv(task.apple)
    env._grasped = task.apple
    before = [seg[1].copy() for seg in task._segs]
    delta = env.obj_pos(task.apple) - env.finger_grasp_point("left")

    task._calibrate_grasp_delta(env)

    np.testing.assert_allclose(task._segs[0][1], before[0])
    for i in range(1, len(task._segs)):
        np.testing.assert_allclose(task._segs[i][1], before[i] - delta)


def test_book_attachment_does_not_rewrite_the_feed_forward_insertion():
    task = next(cls() for cls in LH_TASKS if cls.name == "book2shelf")
    task.apple = _Obj([1.20, 0.14, 0.80])
    task._grasp_delta = None
    task._segs = [
        ("arrive", np.array([1.43, -1.63, 0.82]), -1.0, 100),
        ("rotate_book", np.array([1.43, -1.63, 0.82]), -1.0, 180),
        ("place", np.array([1.63, -1.63, 0.79]), -1.0, 200),
        ("release", np.array([1.63, -1.63, 0.79]), 1.0, 60),
    ]
    env = _FakeEnv(task.apple)
    env._grasped = task.apple
    before = [seg[1].copy() for seg in task._segs]

    task._calibrate_grasp_delta(env)

    assert task._grasp_delta is not None
    for segment, expected in zip(task._segs, before):
        np.testing.assert_allclose(segment[1], expected)


def test_book_uses_measured_upper_shelf_and_inserts_from_the_front():
    task = next(cls() for cls in LH_TASKS if cls.name == "book2shelf")
    task.apple = _Obj([1.20, 0.74, 0.80], extent=(0.072, 0.077, 0.065))
    station = layout.STATIONS["book_shelf"]
    task.bowl = _Obj([*station.xy, station.spawn_z()], top=station.spawn_z() + 0.45)
    task._start_center = np.array([0.29, 0.75, 0.88])
    task._nav = lambda _env, **_kwargs: _VisibleNav()
    env = _FakeEnv(task.apple)
    env._base[:3, 3] = layout.ROBOT_POS
    task._Rgrasp_base = R.from_euler("y", np.pi / 2).as_matrix()
    task._Rgrasp_right_base = task._Rgrasp_base.copy()

    segs = task._build_segments(env)
    by_name = {name: np.asarray(end) for name, end, _grip, _ticks in segs}
    height = station.extent()[2]

    assert task.SPEC.support_z_from_bottom is not None
    assert 0.55 * height < task.SPEC.support_z_from_bottom < 0.90 * height
    # The case faces world -x. Carry descends outside that front, then place moves +x
    # into the upper compartment without changing y or z.
    delta = by_name["place"] - by_name["carry"]
    np.testing.assert_allclose(delta, [task.SPEC.insert_from_front, 0.0, 0.0])
    np.testing.assert_allclose(by_name["retreat"], by_name["carry"])


def test_book_rotates_upright_before_insertion_and_stocks_its_cubby():
    task = next(cls() for cls in LH_TASKS if cls.name == "book2shelf")
    task.apple = _Obj([1.20, 0.74, 0.80], extent=(0.072, 0.077, 0.060))
    station = layout.STATIONS["book_shelf"]
    task.bowl = _Obj([*station.xy, station.spawn_z()], top=station.spawn_z() + 0.45)
    task._start_center = np.array([0.29, 0.75, 0.88])
    task._nav = lambda _env, **_kwargs: _VisibleNav()
    env = _FakeEnv(task.apple)
    env._base[:3, 3] = layout.ROBOT_POS
    task._Rgrasp_base = R.from_euler("y", np.pi / 2).as_matrix()
    task._Rgrasp_right_base = task._Rgrasp_base.copy()
    task._grasp_off_local = env._finger_offset.copy()

    segs = task._build_segments(env)
    names = [seg[0] for seg in segs]
    assert names.index("close") < names.index("rotate_book") < names.index("place")
    assert (
        names.index("release") < names.index("retreat") < names.index("clear_book")
    )
    assert len(segs) == len(task._grasp_rotations)
    source_R = task._grasp_rotations[names.index("close")] @ task._Rgrasp_base
    placed_R = task._grasp_rotations[names.index("place")] @ task._Rgrasp_base
    # The source wrist is pitched downward, then rotates up for insertion. The placement
    # rotation is composed on the same source alignment, so the
    # attachment still undergoes exactly the desired object rotation.
    source_alignment = task._grasp_alignment("left")
    np.testing.assert_allclose(source_R, source_alignment @ task._Rgrasp_base, atol=1e-12)
    np.testing.assert_allclose(
        placed_R,
        task._reorient_R(task.SPEC) @ source_alignment @ task._Rgrasp_base,
        atol=1e-12,
    )
    assert source_R[2, 2] <= -0.49
    assert abs(placed_R[2, 2]) <= 1e-12
    assert placed_R[0, 2] >= 0.86
    np.testing.assert_allclose(task._grasp_world_offset(), [0.0, 0.0, 0.0], atol=1e-8)
    object_R = task._reorient_R(task.SPEC) @ task._spawn_R(task.SPEC)
    outward = rot_z(float(station.yaw)) @ np.array([1.0, 0.0, 0.0])
    # ajpulw's binding runs along local +x and its actual spine face is local -y.
    np.testing.assert_allclose(object_R @ [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], atol=1e-8)
    np.testing.assert_allclose(object_R @ [0.0, -1.0, 0.0], outward, atol=1e-8)
    clear_i = names.index("clear_book")
    retreat_i = names.index("retreat")
    np.testing.assert_allclose(task._grasp_rotations[clear_i], source_alignment)
    np.testing.assert_allclose(
        task._grasp_rotations[retreat_i],
        task._reorient_R(task.SPEC) @ source_alignment,
    )
    assert segs[clear_i][3] >= 180
    fillers = [cfg for cfg in task.object_configs()
               if cfg["name"].startswith("lh_book_filler_")]
    assert len(fillers) == 2
    assert all(cfg.get("fixed_base") is True for cfg in fillers)
    # The stocked books are also physical release guides. Their slot accepts the observed range
    # of valid assisted-grasp transforms, and their depth spans insertion and inward settle.
    Rstation = rot_z(float(station.yaw))
    Rfiller = R.from_quat(BOOK_FILLER_QUAT).as_matrix()
    filler_geometry = []
    for cfg in BOOK_FILLERS:
        extent = layout.bbox_size(cfg["category"], cfg["model"], cfg["scale"])
        world_extent = np.abs(Rfiller) @ extent
        center = np.asarray(station.xy) + (
            Rstation @ np.array([*cfg["local_xy"], 0.0])
        )[:2]
        filler_geometry.append((center, world_extent))
    filler_geometry.sort(key=lambda item: item[0][1])
    slot = (filler_geometry[1][0][1] - filler_geometry[0][0][1]
            - 0.5 * (filler_geometry[0][1][1] + filler_geometry[1][1][1]))
    carried_extent = layout.bbox_size(
        task.SPEC.obj["category"], task.SPEC.obj["model"], task.SPEC.obj["scale"]
    )
    upright_extent = np.abs(object_R) @ carried_extent
    assert upright_extent[1] + 0.068 < slot < upright_extent[1] + 0.078
    nominal = np.asarray(station.xy) + (
        Rstation @ np.asarray(task.SPEC.place_offset, dtype=float)
    )[:2]
    for center, extent in filler_geometry:
        assert center[0] - extent[0] / 2.0 < nominal[0] - 0.040
        assert center[0] + extent[0] / 2.0 > nominal[0] + 0.080


def test_book_success_height_rejects_a_different_cubby():
    task = next(cls() for cls in LH_TASKS if cls.name == "book2shelf")
    task.bowl = _Obj([1.90, -1.63, 0.40], top=1.20)
    support = task._support_surface_z()
    task.apple = _Obj([1.60, -1.63, support + 0.04])

    task.apple.aabb = (
        np.array([1.56, -1.67, support + 0.035]),
        np.array([1.64, -1.59, support + 0.115]),
    )
    assert task._at_target_support_height()

    task.apple.aabb = (
        np.array([1.56, -1.67, support - 0.30]),
        np.array([1.64, -1.59, support - 0.22]),
    )
    assert not task._at_target_support_height()


def test_pickplace_interpolates_gripper_close_on_the_waypoint_clock():
    apple = _Obj([1.25, 0.72, 0.79])
    env = _FakeEnv(apple)
    task = PickPlaceTask()
    task.apple = apple
    task.bowl = _Obj([1.22, 0.55, 0.79])
    task.expert_reset(env)

    commands, phases = [], []
    while True:
        cmd = task.expert_step(env, None)
        commands.append(cmd.gripper_left)
        phases.append(cmd.phase)
        if cmd.done:
            break
    close = phases.index("close")
    assert commands[close] == env.grip_open
    assert env.grip_close in commands
    assert np.max(np.abs(np.diff(commands))) <= 0.1


def test_long_horizon_reach_servo_updates_while_residual_remains():
    task = next(cls() for cls in LH_TASKS if cls.name == "book2shelf")
    task.apple = _Obj([1.20, 0.74, 0.80])
    task._segs = [
        ("settle", np.array([1.20, 0.74, 0.80]), 1.0, 200),
        ("close", np.array([1.20, 0.74, 0.80]), -1.0, 45),
        ("lift", np.array([1.20, 0.74, 0.95]), -1.0, 80),
    ]
    task._seg_i = 0
    task._seg_f = 0.50
    task._reach_delta = None
    env = _FakeEnv(task.apple)
    env._left[:3, 3] = [1.00, 0.74, 0.80]
    before = task._segs[0][1].copy()
    task._calibrate_reach_delta(env)
    once = task._segs[0][1].copy()
    env._left[:3, 3] += [0.04, 0.0, 0.0]
    task._calibrate_reach_delta(env)
    twice = task._segs[0][1].copy()
    assert not np.allclose(before, once)
    assert not np.allclose(once, twice)


def test_towel_plan_is_folded_aspect_and_bimanual():
    task = next(cls() for cls in LH_TASKS if cls.name == "towel2rack")
    spec = task.SPEC
    ext = layout.bbox_size(spec.obj["category"], spec.obj["model"], spec.obj.get("scale"))
    planar = np.sort(np.asarray(ext[:2]))
    assert planar[1] / planar[0] <= 1.5
    task.apple = _Obj([1.20, 0.40, 0.80], extent=ext)
    station = layout.STATIONS["towel_rack"]
    task.bowl = _Obj([*station.xy, station.spawn_z()], top=station.spawn_z() + 0.72)
    task._start_center = np.array([0.29, 0.75, 0.88])
    task._nav = lambda _env, **_kwargs: _VisibleNav()
    env = _FakeEnv(task.apple)
    env._base[:3, 3] = layout.ROBOT_POS
    task._Rgrasp_base = R.from_euler("y", np.pi / 2).as_matrix()
    task._Rgrasp_right_base = R.from_euler("y", np.pi / 2).as_matrix()
    task._grasp_off_local = env._finger_offset.copy()
    task._right_grasp_off_local = env._finger_offset.copy()
    segs = task._build_segments(env)
    task._segs = segs
    assert len(segs) == len(task._right_segs)
    assert [(n, t) for n, _p, _g, t in segs] == [
        (n, t) for n, _p, _g, t in task._right_segs
    ]
    by_name = {left[0]: (left, right) for left, right in zip(segs, task._right_segs)}
    for phase in {"close_right", "right_hold", "reaim_left"}:
        left, right = by_name[phase]
        assert left[2] == env.grip_open
        assert right[2] == env.grip_close
    for phase in {"close_left", "grasp_hold", "lift", "arrive", "carry", "place"}:
        left, right = by_name[phase]
        assert left[2] == right[2] == env.grip_close
    for left, right in zip(segs, task._right_segs):
        if left[0].startswith("transit"):
            assert left[2] == right[2] == env.grip_close
    names = [entry[0] for entry in segs]
    assert names.index("close_right") < names.index("reaim_left") < names.index("close_left")
    # Leave enough recorded time after opening for assisted-grasp joints to clear and for the
    # towel to settle visibly on the rack.  Thirty ticks ended the rollout on the detach tick.
    assert by_name["retreat"][0][3] >= 100
    assert segs[0][0] == task._right_segs[0][0] == "orient"
    # The two arms retain their natural separation and roll about their own forward axes, so the
    # tool approach remains horizontal while both jaw axes become vertical around the overhang.
    left_off = task._grasp_world_offset("left")
    right_off = task._grasp_world_offset("right")
    np.testing.assert_allclose(left_off - right_off, [0.0, 0.19, 0.0], atol=1e-12)
    left_R = task._grasp_rotations[0] @ task._Rgrasp_base
    right_R = task._right_grasp_rotations[0] @ task._Rgrasp_right_base
    np.testing.assert_allclose(left_R[:, 2], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(right_R[:, 2], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(left_R[:, 1], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(right_R[:, 1], [0.0, 0.0, 1.0], atol=1e-12)
    left_target = task._grasp_target(segs[0][1], left_R)
    right_target = task._grasp_target(
        task._right_segs[0][1], right_R, task._right_grasp_off_local
    )
    np.testing.assert_allclose(left_target[:3, 3], task._orient_left_eef, atol=1e-12)
    np.testing.assert_allclose(right_target[:3, 3], task._orient_right_eef, atol=1e-12)

    # The right joint's attachment snap is absorbed by a constant-speed left-only segment. Its
    # first sample stays at the old endpoint; later source waypoints move together, while the
    # immutable head plan does not inherit this measured correction.
    task._head_plan_segs = [
        (name, np.asarray(end).copy(), grip, ticks) for name, end, grip, ticks in segs
    ]
    reaim = names.index("reaim_left")
    old = task._segs[reaim][1].copy()
    head_old = task._head_plan_segs[reaim][1].copy()
    task._seg_i = reaim
    task._seg_f = 0.0
    env._grasped = task.apple
    task.apple.pos += [0.022, 0.005, 0.014]
    task._second_hand_reaimed = False
    task._reaim_second_hand(env, "reaim_left")
    correction = np.array([0.022, 0.005, 0.014])
    np.testing.assert_allclose(task._segs[reaim][1], old + correction)
    np.testing.assert_allclose(task._head_plan_segs[reaim][1], head_old)


def test_lh_sequence_keeps_every_per_waypoint_list_aligned():
    task = LongHorizonSequence()
    task._objs = {}
    task._stns = {}
    for task_name in task.ORDER:
        spec = SPEC_BY_NAME[task_name]
        extent = layout.bbox_size(spec.obj["category"], spec.obj["model"], spec.obj.get("scale"))
        task._objs[task_name] = _Obj([*task.SOURCE_MARKS[task_name], 0.80], extent=extent)
        station = layout.STATIONS[spec.station]
        task._stns[task_name] = _Obj(
            [*station.xy, station.spawn_z()], top=station.spawn_z() + station.extent()[2]
        )
    task._select(0)
    env = _FakeEnv(task.apple)
    env._base[:3, 3] = layout.ROBOT_POS
    task._start_center = np.array([0.314, 0.726, 0.963])
    task._right_start_center = np.array([0.315, 0.074, 0.963])
    task._head_pos0 = env._head[:3, 3].copy()
    task._Rgrasp_base = R.from_euler("y", np.pi / 2).as_matrix()
    task._Rgrasp_right_base = R.from_euler("y", np.pi / 2).as_matrix()
    task._grasp_off_local = env._finger_offset.copy()
    task._right_grasp_off_local = env._finger_offset.copy()
    task._base_cmd_z = layout.ROBOT_POS[2]
    task._Ree_base = np.linalg.inv(env._base) @ env._right
    task._nav = lambda _env, **_kwargs: _VisibleNav()

    segs = task._build_segments(env)
    parallel = (
        task._headings,
        task._standoffs,
        task._owners,
        task._grasp_rotations,
        task._right_segs,
        task._right_grasp_rotations,
        task._right_segment_starts,
    )
    assert all(len(values) == len(segs) for values in parallel)
    assert set(task._owners) == {0, 1, 2}
    for left, right in zip(segs, task._right_segs):
        if right is not None:
            assert (left[0], left[3]) == (right[0], right[3])

    # A later cycle is planned before it executes.  Its in-place wrist turn must derive the
    # fixed EEF positions from the planned incoming fingertip centres, not from the robot's
    # episode-reset measured EEF poses.  The latter caused the 20--33 cm target jump seen in the
    # chained towel cycle.
    orient = next(i for i, (name, _end, _grip, _ticks) in enumerate(segs)
                  if name == "orient")
    assert task._owners[orient] == task.ORDER.index("towel2rack")
    left_start = np.asarray(segs[orient - 1][1])
    left_R0 = rot_z(task._headings[orient - 1]) @ task._grasp_rotations[orient - 1] \
        @ task._Rgrasp_base
    np.testing.assert_allclose(
        task._orient_left_eef,
        left_start - left_R0 @ task._grasp_off_local,
        atol=1e-12,
    )
    right_start = np.asarray(task._right_segment_starts[orient])
    right_R0 = rot_z(task._headings[orient - 1]) @ task._right_grasp_rotations[orient - 1] \
        @ task._Rgrasp_right_base
    np.testing.assert_allclose(
        task._orient_right_eef,
        right_start - right_R0 @ task._right_grasp_off_local,
        atol=1e-12,
    )


def test_pushchair_is_directly_ahead_and_requires_no_turn():
    station = layout.STATIONS["chair"]
    delta = np.asarray(station.xy) - np.asarray(layout.ROBOT_POS[:2])
    fwd = np.array([np.cos(layout.ROBOT_YAW), np.sin(layout.ROBOT_YAW)])
    side = np.array([-fwd[1], fwd[0]])

    assert np.isclose(station.approach, layout.ROBOT_YAW)
    assert np.isclose(station.yaw, layout.ROBOT_YAW)
    assert 0.4 < np.dot(delta, fwd) < 1.2
    assert abs(np.dot(delta, side)) < 0.45


def test_station_turn_rotates_head_heading_on_the_same_segment():
    task = PushChairTask()
    task._nav = lambda _env, **_kwargs: _VisibleNav()
    env = _FakeEnv(_Obj([0.0, 0.0, 0.0]))
    heading = np.pi / 2

    drive, headings = task._drive_segments(
        env,
        carry_local=np.array([0.6, 0.3, 0.9]),
        hand_z=0.9,
        heading=heading,
        gripper=env.grip_open,
        goal_xy=np.array([0.5, 0.4]),
    )

    assert [seg[0] for seg in drive[:2]] == ["roll", "turn"]
    assert len(drive) == len(headings)
    assert np.isclose(headings[0], task.ROBOT_YAW)
    assert np.isclose(headings[1], heading)

    # Halfway through the turn, the commanded camera optical axis has also yawed halfway.
    task._headings = headings
    task._standoffs = [1.0] * len(drive)
    task._seg_i = 1
    task._seg_f = 0.5
    task._center = 0.5 * (drive[0][1] + drive[1][1])
    task._head_pos0 = env.link_pose("zed_depth_frame")[:3, 3]
    task._head_pos_base = (
        np.linalg.inv(env.link_pose("base")) @ env.link_pose("zed_depth_frame")
    )[:3, 3]
    task._head_off_base = task._head_pos0 - task._center
    head_target = task._head_target(env)
    optical_axis = head_target[:3, 2]
    commanded_yaw = np.arctan2(optical_axis[1], optical_axis[0])
    assert np.isclose(commanded_yaw, np.pi / 4)


def test_pushchair_does_not_retarget_the_rail_path_from_chair_com_offset():
    """A chair moves with its grasped rail; pick/place object-centering must not shift the hand."""
    task = PushChairTask()
    task.apple = _Obj([0.45, 0.75, 0.42])
    task._grasp_delta = None
    task._segs = [
        ("push", np.array([0.62, 0.75, 0.76]), -1.0, 10),
        ("release", np.array([0.62, 0.75, 0.76]), 1.0, 10),
        ("retreat", np.array([0.62, 0.75, 0.90]), 1.0, 10),
    ]
    env = _FakeEnv(task.apple)
    env._grasped = task.apple
    before = [(name, end.copy(), grip, ticks) for name, end, grip, ticks in task._segs]

    task._calibrate_grasp_delta(env)

    for actual, expected in zip(task._segs, before):
        assert actual[0] == expected[0]
        np.testing.assert_allclose(actual[1], expected[1])
        assert actual[2:] == expected[2:]
