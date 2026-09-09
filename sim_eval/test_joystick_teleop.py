"""Exercise the real follower loop and HDF5 writer with a CPU simulator stand-in."""
import argparse
import importlib.util
from pathlib import Path
import pickle
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from joystick_service import BehaviorDriver, GROUPS, load_reference
from robot_model import WBIK_YAML
from teleop_protocol import pack, unpack


def test_bridge_roundtrips_images_without_numpy_pickle_globals():
    frame = {"rgb": np.arange(60, dtype=np.uint8).reshape(4, 5, 3),
             "depth": np.array([[65535, 12]], dtype=np.uint16),
             "pose": np.eye(4, dtype=np.float32), "stamp": np.int64(123)}
    data = pickle.dumps(pack(frame), protocol=4)
    assert b"numpy" not in data
    actual = unpack(pickle.loads(data))
    for key in frame:
        np.testing.assert_array_equal(actual[key], frame[key])
    with pytest.raises(TypeError, match="Object arrays"):
        pack(np.array([{}], dtype=object))


class Clock:
    now = 0.0

    def __call__(self):
        return self.now

    def advance(self, dt):
        self.now += dt


class Source:
    def __init__(self, clock):
        self.clock = clock

    def start(self):
        pass

    @property
    def latest(self):
        now = self.clock()
        # Live packets stop after 0.2s, then the operator presses X and finally Y.
        estop = now < .02 or now >= .95
        return SimpleNamespace(
            timestamp_ns=int(min(int(now * 10) / 10, .2) * 1e9),
            estop=estop, calib_stage="static" if estop else "teleop",
            exit_requested=now >= 1.05, home_requested=False,
            left_ee_pose=np.eye(4).ravel().tolist() if not estop else [],
            right_ee_pose=np.eye(4).ravel().tolist() if not estop else [],
            head_ee_pose=np.eye(4).ravel().tolist(),
            left_gripper=.7, right_gripper=.2, chassis_vx=0., chassis_vy=0., chassis_wz=.3)


class IK:
    _idx_q = {name: i for i, name in enumerate(n for names in GROUPS.values() for n in names)}

    def nominal_q(self):
        return np.zeros(20)

    def reset(self):
        pass

    def frame_pose(self, name):
        return np.eye(4)

    def solve(self, left, right, dt):
        return SimpleNamespace(**{g: np.zeros(len(n)) for g, n in GROUPS.items()},
                               success=True, held=False, left_ee_error=0., right_ee_error=0.,
                               stability_margin=1., safety_status="OK")


class PlantDriver(BehaviorDriver):
    def __init__(self, *args, clock, **kwargs):
        self.clock = clock
        self.commands = []
        self.plant = {"base_pose": np.zeros(3), "joints": {g: np.zeros(len(n)) for g, n in GROUPS.items()}}
        super().__init__(*args, **kwargs)

    def rpc(self, command, **req):
        # Validate the process-boundary serialization on every simulated request/reply.
        req = unpack(pickle.loads(pickle.dumps(pack(req))))
        if command in ("initialize", "engage", "home", "stop"):
            if command == "engage":
                self.plant["base_pose"] = np.zeros(3)
            if command == "stop":
                self.commands.append((self.clock(), np.zeros(3), True))
            return self.plant
        if command == "step":
            self.plant["base_pose"] = self.plant["base_pose"] + np.asarray(req["base"]) * .01
            self.plant["joints"] = req["joints"]
            self.commands.append((self.clock(), np.asarray(req["base"]), self._held))
            return self.plant
        if command == "episode_start":
            return {"static": {"obs": {"images": {"intrinsic": np.eye(3)}}}}
        if command == "episode_result":
            return {"task_success": False, "success_evaluated": True}
        if command == "capture":
            frame = {
                "timestamp_ns": np.int64(0),
                "obs": {"base": {"pose": self.plant["base_pose"].astype(np.float32),
                                 "pose_sim": self.plant["base_pose"].astype(np.float32)},
                        "images": {"head_left_rgb": np.zeros((4, 6, 3), dtype=np.uint8),
                                   "head_depth": np.zeros((4, 6), dtype=np.uint16),
                                   "left_wrist_rgb": np.zeros((4, 6, 3), dtype=np.uint8),
                                   "right_wrist_rgb": np.zeros((4, 6, 3), dtype=np.uint8)}},
                "action": {"eef": {"left": req["effective_targets"]["left"],
                                   "right": req["effective_targets"]["right"]},
                           "head": req["effective_targets"]["head"],
                           "base": {"pose": req["base_pose"]}},
            }
            return {"frame": unpack(pickle.loads(pickle.dumps(pack(frame))))}
        raise AssertionError(command)


def test_real_loop_stops_stale_input_and_records_current_base_intent(tmp_path):
    from omniteleop.follower.whole_body_ik import WBCConfig
    from omniteleop.wbc_policy_format import JOYSTICK_SIM_EPISODE_SCHEMA

    reference = load_reference()
    args = argparse.Namespace(config=str(WBIK_YAML), record=True, record_rate=10., hud_rate=0.,
                              source_timeout=.5, save_dir=str(tmp_path), max_joint_step=.05,
                              arm_cmd_lpf_tau=.04, turn_90=False)
    reference._bind_vr_teleop(args, argparse.ArgumentParser())
    cfg = WBCConfig.from_yaml(WBIK_YAML)
    cfg.lock_base_in_ik, cfg.head_mode = True, "fixed"
    clock = Clock()
    source = Source(clock)
    driver = PlantDriver(None, source, IK(), cfg, args, reference, clock=clock)
    reference.run_loop(source, IK(), cfg, driver, args, replay=False, clock=clock,
                       realtime=False, enable=dict(torso=True, arms=True, head=True, base=True))
    driver.close()
    assert any(np.linalg.norm(cmd) > 0 for _, cmd, _ in driver.commands)
    assert all(np.array_equal(cmd, np.zeros(3)) for t, cmd, held in driver.commands if t > .72)
    assert any(held for t, _, held in driver.commands if .72 < t < .9)
    path = tmp_path / "episode_0.hdf5"
    with h5py.File(path) as f:
        assert f["meta/schema"][()].decode() == JOYSTICK_SIM_EPISODE_SCHEMA
        assert f["meta/eef_target_frame"][()] == b"current_base"
        intent = f["action/chassis/intent_body"][:]
        np.testing.assert_allclose(intent[:, 2], .3)
        assert np.all(f["action/base/pose"][:, 2] > 0)
        # Turning the base must not rotate the hand's current-base-frame policy target.
        np.testing.assert_allclose(f["action/eef/left"][:], np.tile(np.eye(4), (len(intent), 1, 1)))
        assert not f["sim_result/task_success"][()]
        assert f["obs/images/head_depth"].dtype == np.uint16
    assert not list(tmp_path.glob("*.partial"))

    path_to_porter = Path(__file__).resolve().parents[1] / "scripts/port_wbc_mobile_hdf5.py"
    spec = importlib.util.spec_from_file_location("_behavior_porter_test", path_to_porter)
    porter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(porter)
    assert porter.inspect_episode_policy_action_contract(path)["chassis_translation_speed_mps"] == .18
    assert porter.inspect_episode_base_pose_contract(path)["base_pose_source"] == "sim_ground_truth"


def test_sim_headset_cameras_subscribe_in_sim_namespace(monkeypatch):
    import dexcomm
    from omniteleop.leader.wbc_headset_hud import WBCHeadsetHUD

    callbacks = {}

    class Node:
        def __init__(self, *, name, namespace):
            assert namespace == "behavior"

        def create_subscriber(self, topic, callback, decoder):
            callbacks[topic] = callback

    monkeypatch.setattr(dexcomm, "Node", Node)
    hud = WBCHeadsetHUD(None, ["head_left_rgb", "left_wrist_rgb", "right_wrist_rgb"], sim_namespace="behavior")
    assert hud._cam_robot is None
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    for callback in callbacks.values():
        callback({"data": rgb})
    np.testing.assert_array_equal(hud._last_head_imgs["left_rgb"], rgb)
    assert set(hud._last_wrist_imgs) == {"left_wrist_rgb", "right_wrist_rgb"}


@pytest.mark.parametrize("disconnect", [False, True])
def test_interrupted_demo_is_saved_only_as_partial(tmp_path, disconnect):
    from omniteleop.follower.whole_body_ik import WBCConfig

    reference = load_reference()
    args = argparse.Namespace(config=str(WBIK_YAML), record=True, record_rate=10., hud_rate=0.,
                              source_timeout=.5, save_dir=str(tmp_path), max_joint_step=.05,
                              arm_cmd_lpf_tau=.04, turn_90=False)
    reference._bind_vr_teleop(args, argparse.ArgumentParser())
    cfg = WBCConfig.from_yaml(WBIK_YAML)
    cfg.lock_base_in_ik, cfg.head_mode = True, "fixed"
    clock = Clock()
    clock.advance(.1)
    source = Source(clock)
    driver = PlantDriver(None, source, IK(), cfg, args, reference, clock=clock)
    driver.latch_source_sample(source.latest)
    result = IK().solve(np.eye(4), np.eye(4), .01)
    driver.actuate(result, .7, .2, None, False, .01)
    driver.start_recording_if_teleop(source.latest)
    driver.record_tick(result, False, clock(), left_target=np.eye(4), right_target=np.eye(4),
                       head_target=np.eye(4), source_timestamp_ns=1, source_receive_wall_ns=2)
    if disconnect:
        def broken_rpc(*args, **kwargs):
            raise ConnectionError("simulator disconnected")
        driver.rpc = broken_rpc
        with pytest.raises(RuntimeError, match="connection failed"):
            driver.close()
    else:
        driver.close(RuntimeError("operator interrupted"))
    assert not list(tmp_path.glob("*.hdf5"))
    assert (tmp_path / "episode_0.hdf5.partial").is_file()


def test_sim_pose_reanchors_at_engage_after_translating_and_turning():
    from teleop_joystick import SimSession
    from scipy.spatial.transform import Rotation

    class Joints:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros(20)

    origin = np.eye(4)
    origin[:3, 3] = [2, 3, .1]
    origin[:3, :3] = Rotation.from_euler("z", .9).as_matrix()
    env = SimpleNamespace(
        og=SimpleNamespace(sim=SimpleNamespace(is_playing=lambda: True)),
        scene_options={},
        robot=SimpleNamespace(get_joint_positions=Joints),
        name2idx=IK._idx_q, T_align=origin, T_align_inv=np.linalg.inv(origin))
    world_pose = origin.copy()
    env.link_pose = lambda _: world_pose
    env.og_to_wbc = lambda pose: env.T_align_inv @ pose
    session = SimSession(env, None, SimpleNamespace(task="none", scene="empty", seed=0))
    np.testing.assert_allclose(session.snapshot()["base_pose"], np.zeros(3), atol=1e-12)
    movement = np.eye(4)
    movement[0, 3] = .4
    movement[:3, :3] = Rotation.from_euler("z", .5).as_matrix()
    world_pose = origin @ movement
    np.testing.assert_allclose(session.snapshot()["base_pose"], [.4, 0, .5], atol=1e-12)
    session.handle({"cmd": "engage"})
    np.testing.assert_allclose(session.snapshot()["base_pose"], np.zeros(3), atol=1e-12)
