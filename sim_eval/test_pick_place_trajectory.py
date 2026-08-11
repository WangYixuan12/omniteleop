"""Fast regression check for the scripted Robotiq pregrasp path."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as R

from sim_eval import base_ctrl, zed_sim
from sim_eval.tasks import layout
from sim_eval.tasks.long_horizon import (
    BOOK_FILLERS, BOOK_FILLER_QUAT, LH_TASKS, PLACE_STANDOFF, SPEC_BY_NAME,
    LongHorizonSequence, _box2cloth_route, _manhattan, rot_y)
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
        self._release_calls = []
        self.robot.release_grasp_immediately = self._release_grasp_immediately
        self._assisted_handling = []
        self._finger_offset = np.array([0.15, 0.0, 0.0])

    def _release_grasp_immediately(self, arm):
        self._release_calls.append(arm)
        self._grasped = None

    def set_assisted_grasp_handling(self, enabled):
        self._assisted_handling.append(bool(enabled))

    def link_pose(self, name):
        return self._poses[name].copy()

    def obj_pos(self, obj):
        return obj.pos.copy()

    def finger_grasp_point(self, arm):
        return self._left[:3, :3] @ self._finger_offset + self._left[:3, 3]

    def is_grasping(self, arm):
        return getattr(self, "_grasped", None)


def test_third_person_camera_tracks_base_without_chase_lag(monkeypatch):
    """Review-camera stabilization must not turn a base strafe into apparent arm teleportation."""
    monkeypatch.setitem(sys.modules, "omnigibson", SimpleNamespace())
    monkeypatch.syspath_prepend("/home/yixuan/omniteleop/sim_eval")
    sys.modules.pop("vega_og_env", None)
    try:
        module = importlib.import_module("vega_og_env")
        env = object.__new__(module.VegaOGEnv)
        env.task = SimpleNamespace(_head_heading=0.0)
        base = np.eye(4)
        env.link_pose = lambda _name: base.copy()
        env._cam_clear = lambda _eye: True
        calls = []
        env._set_cam_lookat = lambda eye, aim, up=(0, 0, 1): calls.append(
            (np.asarray(eye).copy(), np.asarray(aim).copy())
        )

        env.follow_third_person()
        base[1, 3] = -0.20
        env.follow_third_person()

        eye0, aim0 = calls[0]
        eye1, aim1 = calls[1]
        np.testing.assert_allclose(eye1 - base[:3, 3], eye0, atol=1e-12)
        np.testing.assert_allclose(aim1 - base[:3, 3], aim0, atol=1e-12)
    finally:
        sys.modules.pop("vega_og_env", None)


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


def _assert_forward_contact_then_backward_release(segs):
    """Contact at constant height, then open in place before withdrawing."""
    by_name = {name: np.asarray(end) for name, end, _grip, _ticks in segs}
    pickup = by_name["approach"] - by_name["pregrasp"]
    placement = by_name["place"] - by_name["carry"]
    withdrawal = by_name["retreat"] - by_name["release_hold"]
    for delta, sign in ((pickup, 1.0), (placement, 1.0), (withdrawal, -1.0)):
        assert sign * delta[0] >= 0.075
        np.testing.assert_allclose(delta[1:], 0.0, atol=1e-12)


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


def test_manhattan_keeps_small_orthogonal_correction_and_exact_goal():
    route = np.array([[0.0, 0.0], [1.0, 0.05]])
    result = _manhattan(_VisibleNav(), route, yaw=0.0, min_leg=0.15)
    np.testing.assert_allclose(result[-1], route[-1])
    delta = np.diff(result, axis=0)
    assert np.all(np.linalg.norm(delta, axis=1) > 0.0)
    assert np.all(np.count_nonzero(np.abs(delta) > 1e-9, axis=1) == 1)


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


def test_camera_default_adds_dropout_without_changing_fov():
    opts = zed_sim.ZedSimOptions()
    assert opts.depth_dropout
    assert not opts.match_zed_fov
    assert not opts.wrist_zedm_fov
    assert zed_sim.head_render_hw(opts) == zed_sim.DEFAULT_RENDER_HW
    assert zed_sim.wrist_render_hw(opts) == zed_sim.DEFAULT_WRIST_RENDER_HW


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
        assert not hasattr(task, "BASE_DOFS")
        assert base_ctrl.WBIK.base_dofs == "xy_yaw"
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
        "pregrasp_y", "pregrasp_x", "pregrasp", "approach", "settle", "close",
        "grasp_hold", "lift",
        "transit0", "transit1", "transit2",
        "carry", "place", "release", "release_hold", "retreat", "home_arms",
    ]
    assert len(segs) == len(task._headings) == len(task._standoffs)
    assert task._headings == [0.0] * len(segs)
    first_destination = phases.index("carry")
    assert task._standoffs[:first_destination] == [1.0] * first_destination
    assert task._standoffs[first_destination:phases.index("home_arms")] == [
        PLACE_STANDOFF
    ] * (phases.index("home_arms") - first_destination)
    assert task._standoffs[-1] == 1.0
    assert task._last_base_cmd is None
    assert task._last_left_cmd is None
    assert task._grasp_delta is None
    by_name = {name: (np.asarray(end), grip, ticks) for name, end, grip, ticks in segs}
    np.testing.assert_allclose(by_name["release_hold"][0], by_name["release"][0])
    assert by_name["release_hold"][1] == env.grip_open
    assert 20 <= by_name["release_hold"][2] < 100
    _assert_forward_contact_then_backward_release(segs)
    drop = task._release_point(env)
    # A tray spans the rails and IS the receptacle (LHSpec.liner), so the drop is its centre with
    # no lane offset to hit -- `place_offset` is zero and `support_z_from_bottom` is None, which
    # makes the support the liner's own AABB top. `env.obj_pos(task.bowl)` is the liner here.
    np.testing.assert_allclose(drop[:2], np.asarray(station.xy))
    assert drop[0] < 1.764, "the dish drop must stay inside the measured left-arm reach limit"
    assert np.isclose(drop[2], 0.80 + 0.03 + 0.004)   # gravity clearance above tray support
    assert SPEC_BY_NAME["dish2rack"].liner["category"] == "tray"
    assert SPEC_BY_NAME["dish2rack"].liner["model"] == "coqeme"
    liner_extent = layout.bbox_size("tray", "coqeme", None)
    assert max(liner_extent[:2]) <= 0.25
    dish_obj = SPEC_BY_NAME["dish2rack"].obj
    dish_extent = layout.bbox_size(dish_obj["category"], dish_obj["model"], dish_obj.get("scale"))
    assert dish_obj == {
        "category": "soda_cup", "model": "vicaqs", "scale": [0.78, 0.78, 1.0]
    }
    assert max(dish_extent[:2]) <= 0.065
    assert dish_extent[2] >= 0.12
    # The wall rack's working rail should be midway between the too-low 0.72 m
    # trial and the 0.82 m trial that saturated L_arm_j4 throughout transit.
    assert np.isclose(station.z, 0.482232)


def test_idle_right_arm_keeps_the_same_nominal_pose_during_single_arm_work():
    task = next(cls() for cls in LH_TASKS if cls.name == "dish2rack")
    p = np.zeros(3)
    task._segs = [
        ("pregrasp", p, 1.0, 10),
        ("settle", p, 1.0, 10),
        ("home_arms", p, 1.0, 10),
        ("clear", p, 1.0, 10),
        ("return0", p, 1.0, 10),
        ("pregrasp_y", p, 1.0, 10),
        ("pregrasp", p, 1.0, 10),
    ]
    for i in range(len(task._segs)):
        for fraction in (0.0, 0.5, 1.0):
            task._seg_i, task._seg_f = i, fraction
            np.testing.assert_allclose(task._idle_right_shift(), 0.0)


def test_long_horizon_idle_right_target_follows_head_and_parks_during_arm_only_windows():
    """The idle hand rides the commanded body, except that both park for local manipulation."""
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
        expected_right = task._right_target(current.head_target, task._head_heading)
        np.testing.assert_allclose(current.right_target, expected_right, atol=1e-12)
        right_step = current.right_target[:3, 3] - previous.right_target[:3, 3]
        arm_only_phases = (
            task.HEAD_LOCK_GRASP_PHASES | task.HEAD_LOCK_RELEASE_PHASES
        )
        if previous.phase in arm_only_phases and (
                current.phase in arm_only_phases or current.done):
            # With the head / implied base parked, the idle right hand must also stay parked.
            np.testing.assert_allclose(right_step, 0.0, atol=1e-12)
        else:
            assert np.linalg.norm(right_step) < 0.01
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


def test_assisted_grasp_is_released_once_at_start_of_smooth_opening():
    task = next(cls() for cls in LH_TASKS if cls.name == "dish2rack")
    task.apple = _Obj([1.20, 0.74, 0.80])
    task._assisted_release_done = set()
    task._assisted_handling_suspended = False
    env = _FakeEnv(task.apple)
    env._grasped = task.apple

    task._release_assisted_grasp(env, "place")
    assert env._release_calls == []
    task._release_assisted_grasp(env, "release")
    task._release_assisted_grasp(env, "release")
    assert env._release_calls == ["left"]
    assert env._assisted_handling == [False]
    task._release_assisted_grasp(env, "release_hold")
    task._release_assisted_grasp(env, "retreat")
    assert env._assisted_handling == [False, True]


def test_dish_close_target_matches_cup_width_instead_of_full_closure():
    spec = SPEC_BY_NAME["dish2rack"]
    assert 0.0 < spec.grip_close_cmd < 1.0
    assert spec.grasp_hold_ticks >= 30
    diameter = 0.0783588 * spec.obj["scale"][0]
    assert 0.060 < diameter < 0.063


def test_book_attachment_corrects_height_without_rewriting_xy_insertion():
    task = next(cls() for cls in LH_TASKS if cls.name == "book2shelf")
    task.apple = _Obj([1.20, 0.14, 0.80])
    task._grasp_delta = None
    task._segs = [
        ("arrive", np.array([1.43, -1.63, 0.82]), -1.0, 100),
        ("carry", np.array([1.43, -1.63, 0.82]), -1.0, 260),
        ("place", np.array([1.63, -1.63, 0.79]), -1.0, 200),
        ("release", np.array([1.63, -1.63, 0.79]), 1.0, 60),
    ]
    env = _FakeEnv(task.apple)
    env._grasped = task.apple
    before = [seg[1].copy() for seg in task._segs]
    delta = env.obj_pos(task.apple) - env.finger_grasp_point("left")

    task._calibrate_grasp_delta(env)

    assert task._grasp_delta is not None
    for segment, expected in zip(task._segs, before):
        np.testing.assert_allclose(segment[1][:2], expected[:2])
        assert np.isclose(segment[1][2], expected[2] - delta[2])


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
    task._grasp_off_local = env._finger_offset.copy()

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


def test_book_spawns_upright_and_inserts_without_an_in_air_wrist_turn():
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
    _assert_forward_contact_then_backward_release(segs)
    names = [seg[0] for seg in segs]
    assert names.index("close") < names.index("place")
    assert "rotate_book" not in names
    assert "clear_book" not in names
    assert not task._has_reorientation(task.SPEC)
    # The upright hardback cannot tolerate a vertical final approach. Stop behind it at the
    # incoming height, lower with x/y fixed, then enter horizontally between the covers.
    by_name = {name: np.asarray(end) for name, end, _grip, _ticks in segs}
    high = by_name["pregrasp_x"]
    low = by_name["pregrasp"]
    contact = by_name["approach"]
    np.testing.assert_allclose(high[:2], low[:2], atol=1e-12)
    assert high[2] > low[2] + 0.05
    np.testing.assert_allclose(low[1:], contact[1:], atol=1e-12)
    assert contact[0] > low[0] + 0.09
    assert len(segs) == len(task._grasp_rotations)
    source_R = task._grasp_rotations[names.index("close")] @ task._Rgrasp_base
    placed_R = task._grasp_rotations[names.index("place")] @ task._Rgrasp_base
    # The hardback starts in its final upright attitude, so the wrist keeps one alignment from
    # grasp through insertion rather than rotating the attached book in the air.
    source_alignment = task._grasp_alignment("left")
    np.testing.assert_allclose(source_alignment, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(source_R, source_alignment @ task._Rgrasp_base, atol=1e-12)
    np.testing.assert_allclose(placed_R, source_R, atol=1e-12)
    np.testing.assert_allclose(source_R, task._Rgrasp_base, atol=1e-12)
    np.testing.assert_allclose(task._grasp_world_offset(), [0.0, 0.0, 0.0], atol=1e-8)
    object_R = task._spawn_R(task.SPEC)
    outward = rot_z(float(station.yaw)) @ np.array([1.0, 0.0, 0.0])
    # ajpulw's binding runs along local +x and its actual spine face is local -y. Both directions
    # are already correct on the source table and stay correct throughout the sticky attachment.
    np.testing.assert_allclose(object_R @ [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], atol=1e-8)
    np.testing.assert_allclose(object_R @ [0.0, -1.0, 0.0], outward, atol=1e-8)
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
    assert upright_extent[1] <= 0.050
    assert upright_extent[1] + 0.079 < slot < upright_extent[1] + 0.089
    by_name = {name: (np.asarray(end), ticks) for name, end, _grip, ticks in segs}
    np.testing.assert_allclose(by_name["release_hold"][0], by_name["place"][0])
    assert 20 <= by_name["release_hold"][1] < 100
    nominal = np.asarray(station.xy) + (
        Rstation @ np.asarray(task.SPEC.place_offset, dtype=float)
    )[:2]
    for center, extent in filler_geometry:
        assert center[0] - extent[0] / 2.0 < nominal[0] - 0.040
        assert center[0] + extent[0] / 2.0 > nominal[0] + 0.080


def test_book_release_and_home_never_lower_the_left_eef_target():
    task = next(cls() for cls in LH_TASKS if cls.name == "book2shelf")
    spec = task.SPEC
    extent = layout.bbox_size(spec.obj["category"], spec.obj["model"], spec.obj["scale"])
    task.apple = _Obj([1.20, 0.14, 0.80], extent=extent)
    station = layout.STATIONS[spec.station]
    task.bowl = _Obj(
        [*station.xy, station.spawn_z()], top=station.spawn_z() + station.extent()[2]
    )
    task._nav = lambda _env, **_kwargs: _VisibleNav()
    env = _FakeEnv(task.apple)
    env._base[:3, 3] = layout.ROBOT_POS
    env._left[:3, :3] = rot_z(task.ROBOT_YAW) @ R.from_euler("y", np.pi / 2).as_matrix()
    task.expert_reset(env)
    task._calibrate_reach_delta = lambda _env: None
    task._calibrate_grasp_delta = lambda _env: None

    z, phases = [], []
    while True:
        cmd = task.expert_step(env, None)
        if cmd.phase in {"release", "release_hold", "retreat", "home_arms"}:
            z.append(cmd.left_target[2, 3])
            phases.append(cmd.phase)
        if cmd.done:
            break
    assert {"release", "release_hold", "retreat", "home_arms"} <= set(phases)
    assert np.min(np.diff(z)) >= -1e-10


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


def test_book_success_rejects_a_tilted_carry_even_if_the_shelf_straightens_it(monkeypatch):
    states = ModuleType("omnigibson.object_states")
    states.Inside = type("Inside", (), {})
    states.OnTop = type("OnTop", (), {})
    states.Touching = type("Touching", (), {})
    monkeypatch.setitem(sys.modules, "omnigibson.object_states", states)

    class _Value:
        def get_value(self, _other):
            return True

    class _PoseObj(_Obj):
        def __init__(self, pos, quat):
            super().__init__(pos)
            self._quat = np.asarray(quat, dtype=float)
            self.states = {
                states.Inside: _Value(), states.OnTop: _Value(), states.Touching: _Value()
            }

        def get_position_orientation(self):
            return self.pos.copy(), self._quat.copy()

    task = next(cls() for cls in LH_TASKS if cls.name == "book2shelf")
    intended = task._spawn_R(task.SPEC)
    tilted = R.from_matrix(rot_y(np.deg2rad(30.0)) @ intended).as_quat()
    task.apple = _PoseObj([1.60, -1.63, 0.80], tilted)
    task._grasp_quality = {"ok": True, "max_attitude_error_deg": 0.0}
    env = _FakeEnv(task.apple)
    env._grasped = task.apple

    task._update_grasp_quality(env)

    assert not task._grasp_quality["ok"]
    assert np.isclose(task._grasp_quality["max_attitude_error_deg"], 30.0)

    # Model the cubby physically straightening the object after release. Terminal containment and
    # quaternion are now good, but the trajectory must remain rejected for imitation learning.
    task.bowl = _Obj([1.90, -1.63, 0.40], top=1.20)
    support = task._support_surface_z()
    task.apple.pos = np.array([1.60, -1.63, support + 0.04])
    task.apple._quat = np.asarray(task.SPEC.obj_quat, dtype=float)
    task.apple.aabb = (
        np.array([1.56, -1.67, support]),
        np.array([1.64, -1.59, support + 0.08]),
    )
    task._ever_grasped = {"left": True, "right": True}
    env._grasped = None
    assert not task.success(env)


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


def test_partial_cup_close_is_recorded_as_close_intent():
    segs = [
        ("settle", np.zeros(3), 1.0, 30),
        ("close", np.zeros(3), SPEC_BY_NAME["dish2rack"].grip_close_cmd, 45),
    ]
    assert PickPlaceTask._segment_grip_action(segs, 1, 0.0, 1.0) == 0.0
    assert PickPlaceTask._segment_grip_action(segs, 1, 1.0, 1.0) == 1.0


def test_long_horizon_pick_path_is_not_rewritten_from_measured_reach_error():
    task = next(cls() for cls in LH_TASKS if cls.name == "book2shelf")
    task.apple = _Obj([1.20, 0.74, 0.80])
    task._segs = [
        ("approach", np.array([1.20, 0.74, 0.80]), 1.0, 200),
        ("settle", np.array([1.20, 0.74, 0.80]), 1.0, 200),
        ("close", np.array([1.20, 0.74, 0.80]), -1.0, 45),
        ("lift", np.array([1.20, 0.74, 0.95]), -1.0, 80),
    ]
    task._seg_i = 0
    task._seg_f = 0.50
    task._reach_delta = None
    env = _FakeEnv(task.apple)
    env._left[:3, 3] = [1.00, 0.74, 0.80]
    before = [seg[1].copy() for seg in task._segs]
    task._calibrate_reach_delta(env)
    env._left[:3, 3] += [0.04, 0.0, 0.0]
    task._calibrate_reach_delta(env)
    task._seg_i = 1
    task._seg_f = 0.75
    task._calibrate_reach_delta(env)
    for expected, segment in zip(before, task._segs):
        np.testing.assert_allclose(segment[1], expected)
    assert task._reach_delta is None


def test_towel_plan_is_folded_aspect_and_bimanual():
    task = next(cls() for cls in LH_TASKS if cls.name == "towel2rack")
    spec = task.SPEC
    assert np.isclose(spec.bimanual_half_span, 0.080)
    ext = layout.bbox_size(spec.obj["category"], spec.obj["model"], spec.obj.get("scale"))
    planar = np.sort(np.asarray(ext[:2]))
    assert planar[1] / planar[0] <= 1.5
    table_near_x = 1.088
    source_x = float(spec.obj_xy[0])
    near_edge = source_x - float(ext[0]) / 2.0
    grasp_x = source_x + float(spec.grasp_offset[0])
    assert table_near_x - near_edge >= 0.075
    assert grasp_x - near_edge >= 0.035
    assert table_near_x - grasp_x >= 0.045
    assert source_x - 0.020 > table_near_x
    assert np.isclose(LongHorizonSequence.SOURCE_MARKS["towel2rack"][0], source_x)
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
    _assert_forward_contact_then_backward_release(segs)
    _assert_forward_contact_then_backward_release(task._right_segs)
    assert len(segs) == len(task._right_segs)
    assert [(n, t) for n, _p, _g, t in segs] == [
        (n, t) for n, _p, _g, t in task._right_segs
    ]
    by_name = {left[0]: (left, right) for left, right in zip(segs, task._right_segs)}
    assert "close_both" in by_name
    assert by_name["grasp_hold"][0][3] >= 60
    assert not {"close_right", "right_hold", "reaim_left", "close_left"} & set(by_name)
    for left, right in zip(segs, task._right_segs):
        assert left[2] == right[2]
    for phase in {"close_both", "lift", "carry", "place"}:
        left, right = by_name[phase]
        assert left[2] == right[2] == env.grip_close
    for left, right in zip(segs, task._right_segs):
        if left[0].startswith("transit"):
            assert left[2] == right[2] == env.grip_close
    assert by_name["release_hold"][0][2] == by_name["release_hold"][1][2] == env.grip_open
    names = [entry[0] for entry in segs]
    assert names.index("settle") < names.index("close_both") < names.index("lift")
    # Stage outside the table, descend with x/y fixed, and only then advance horizontally into the
    # towel. This prevents either open gripper from landing on the towel during the descent.
    high_left, high_right = by_name["pregrasp_x"]
    low_left, low_right = by_name["pregrasp"]
    approach_left, approach_right = by_name["approach"]
    for high, low, approach in (
        (high_left, low_left, approach_left),
        (high_right, low_right, approach_right),
    ):
        np.testing.assert_allclose(high[1][:2], low[1][:2], atol=1e-12)
        assert high[1][2] > low[1][2] + 0.05
        np.testing.assert_allclose(low[1][1:], approach[1][1:], atol=1e-12)
        assert approach[1][0] > low[1][0] + 0.09
    # Leave enough recorded time after opening for assisted-grasp joints to clear and for the
    # towel to settle visibly on the rack.  Thirty ticks ended the rollout on the detach tick.
    assert by_name["retreat"][0][3] >= 100
    assert segs[0][0] == task._right_segs[0][0] == "orient"
    # The two arms retain their natural separation and roll about their own forward axes, so the
    # tool approach remains horizontal while both jaw axes become vertical around the overhang.
    left_off = task._grasp_world_offset("left")
    right_off = task._grasp_world_offset("right")
    np.testing.assert_allclose(left_off - right_off, [0.0, 0.16, 0.0], atol=1e-12)
    np.testing.assert_allclose((left_off + right_off)[1:] / 2.0, [0.0, 0.006], atol=1e-12)
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

    # Every demonstration ends in an explicit, symmetric two-arm neutral segment rather than
    # stopping with both wrists extended over the rack.
    assert names[-1] == "home_arms"
    assert task._right_segs[-1] is not None
    assert segs[-1][2] == task._right_segs[-1][2] == env.grip_open
    np.testing.assert_allclose(task._grasp_rotations[-1], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(task._right_grasp_rotations[-1], np.eye(3), atol=1e-12)


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


def test_lh_sequence_towel_entry_keeps_bimanual_targets_symmetric():
    """The chained towel skill must enter from the preceding natural two-arm command."""
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
    env._left[:3, 3] = [0.185, 0.725, 0.973]
    env._right[:3, 3] = [0.185, 0.075, 0.973]
    env._left[:3, :3] = R.from_euler("y", np.pi / 2).as_matrix()
    env._right[:3, :3] = R.from_euler("y", np.pi / 2).as_matrix()

    def finger_grasp_point(arm):
        eef = env._left if arm == "left" else env._right
        return eef[:3, :3] @ env._finger_offset + eef[:3, 3]

    env.finger_grasp_point = finger_grasp_point
    task._nav = lambda _env, **_kwargs: _VisibleNav()
    task.expert_reset(env)
    task._calibrate_reach_delta = lambda _env: None
    task._calibrate_grasp_delta = lambda _env: None

    orient = next(
        i for i, (name, _end, _grip, _ticks) in enumerate(task._segs)
        if name == "orient" and task._owners[i] == task.ORDER.index("towel2rack")
    )
    boundary = sum(segment[3] for segment in task._segs[:orient])
    task._t = boundary - 1
    task._seg_i = orient - 1
    task._seg_f = 1.0 - 1.0 / task._segs[orient - 1][3]
    task._select(task._owners[orient])

    before = task.expert_step(env, None)
    current = task.expert_step(env, None)
    assert before.phase == "return2"
    assert current.phase == "orient"
    assert np.linalg.norm(current.right_target[:3, 3] - before.right_target[:3, 3]) < 0.005

    owner = task.ORDER.index("towel2rack")
    owned = [(i, seg) for i, seg in enumerate(task._segs) if task._owners[i] == owner]
    assert "pregrasp_x" not in [seg[0] for _i, seg in owned]
    return_i = max(i for i, seg in owned if seg[0].startswith("return"))
    pregrasp_i = next(i for i, seg in owned if seg[0] == "pregrasp")
    np.testing.assert_allclose(
        task._segs[return_i][1][0], task._segs[pregrasp_i][1][0], atol=0.005
    )

    bimanual_lead_in = {
        "orient", "pregrasp_y", "pregrasp_x", "pregrasp", "approach", "settle",
        "close_both",
    }
    centers = {}
    while current.phase in bimanual_lead_in:
        np.testing.assert_allclose(
            current.left_target[[0, 2], 3],
            current.right_target[[0, 2], 3],
            atol=1e-12,
        )
        centers.setdefault(current.phase, []).append(
            current.left_target[:3, :3] @ task._grasp_off_local
            + current.left_target[:3, 3]
        )
        current = task.expert_step(env, None)

    descend = np.asarray(centers["pregrasp"])
    forward = np.asarray(centers["approach"])
    assert np.ptp(descend[:, 0]) < 1e-10
    assert descend[-1, 2] < descend[0, 2] - 0.05
    assert np.ptp(forward[:, 2]) < 1e-10
    assert forward[-1, 0] > forward[0, 0] + 0.09


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


def test_lh_head_is_fixed_during_local_grasp_and_release_arm_motion():
    """The head / implied base park while the hands descend, contact, and clear."""
    for task in (cls() for cls in LH_TASKS):
        spec = task.SPEC
        extent = layout.bbox_size(spec.obj["category"], spec.obj["model"], spec.obj.get("scale"))
        source = LongHorizonSequence.SOURCE_MARKS[spec.name]
        task.apple = _Obj([*source, 0.80], extent=extent)
        station = layout.STATIONS[spec.station]
        task.bowl = _Obj(
            [*station.xy, station.spawn_z()], top=station.spawn_z() + station.extent()[2]
        )
        env = _FakeEnv(task.apple)
        env._base[:3, 3] = layout.ROBOT_POS
        env._left[:3, :3] = R.from_euler("y", np.pi / 2).as_matrix()
        env._right[:3, :3] = R.from_euler("y", np.pi / 2).as_matrix()

        def finger_grasp_point(arm):
            eef = env._left if arm == "left" else env._right
            return eef[:3, :3] @ env._finger_offset + eef[:3, 3]

        env.finger_grasp_point = finger_grasp_point
        task._nav = lambda _env, **_kwargs: _VisibleNav()
        task.expert_reset(env)
        task._calibrate_reach_delta = lambda _env: None
        task._calibrate_grasp_delta = lambda _env: None

        windows = {"grasp": ([], [], []), "release": ([], [], [])}
        while True:
            cmd = task.expert_step(env, None)
            window = (
                "grasp" if cmd.phase in task.HEAD_LOCK_GRASP_PHASES
                else "release" if cmd.phase in task.HEAD_LOCK_RELEASE_PHASES
                else None
            )
            if window is not None:
                windows[window][0].append(cmd.head_target.copy())
                windows[window][1].append(cmd.left_target.copy())
                windows[window][2].append(cmd.right_target.copy())
            if cmd.done:
                break

        for window, (heads, left_targets, right_targets) in windows.items():
            heads = np.asarray(heads)
            left_targets = np.asarray(left_targets)
            right_targets = np.asarray(right_targets)
            assert len(heads), f"{spec.name} never entered its {window} window"
            np.testing.assert_allclose(heads - heads[0], 0.0, atol=1e-12)
            arm_travel = np.linalg.norm(left_targets[:, :3, 3] - left_targets[0, :3, 3], axis=1)
            assert np.max(arm_travel) > 0.05, f"{spec.name} {window} did not move its arm"
            right_travel = np.linalg.norm(
                right_targets[:, :3, 3] - right_targets[0, :3, 3], axis=1
            )
            if task._is_bimanual():
                assert np.max(right_travel) > 0.05
            else:
                np.testing.assert_allclose(right_travel, 0.0, atol=1e-12)
