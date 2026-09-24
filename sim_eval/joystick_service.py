"""Dexmate-side driver for BEHAVIOR; runs the real joystick follower's run_loop.

Started by teleop_joystick.py over an inherited local socket. No hardware driver
or dexcontrol Robot is constructed. IK, transport, status, base shaping, and the
streaming HDF5 recorder stay in the same Python environment as real teleoperation.
"""
from __future__ import annotations

import argparse
import importlib.util
import socket
import sys
import time

import numpy as np

from _wire import recv_msg, send_msg
from robot_model import OMNITELEOP_SRC, WBIK_YAML
from teleop_protocol import pack, unpack

sys.path.insert(0, str(OMNITELEOP_SRC))

GROUPS = {"torso": [f"torso_j{i}" for i in range(1, 4)],
          "left_arm": [f"L_arm_j{i}" for i in range(1, 8)],
          "right_arm": [f"R_arm_j{i}" for i in range(1, 8)],
          "head": [f"head_j{i}" for i in range(1, 4)]}
CAMERA_TOPICS = {"head_left_rgb": "head_camera", "left_wrist_rgb": "left_wrist_zedm",
                 "right_wrist_rgb": "right_wrist_zedm"}

#: BEHAVIOR drives the base faster than the hardware contract. wbik.yaml's
#: joystick_teleop speeds are the REAL robot's and stay that way; this scales only the
#: simulator's copy of them (see main()).
SIM_BASE_SPEED_SCALE = 2.0


def load_reference():
    path = OMNITELEOP_SRC.parent / "scripts/wbc_joystick_robot.py"
    spec = importlib.util.spec_from_file_location("_behavior_joystick_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BehaviorDriver:
    name = "BEHAVIOR joystick"

    def __init__(self, conn, source, ik, cfg, args, reference, recorder=None):
        from omniteleop.common.streaming_recorder import StreamingEpisodeRecorder
        from omniteleop.follower.joystick_base import JoystickBaseShaper

        self.conn, self.source, self.cfg, self.args = conn, source, cfg, args
        self.reference = reference
        self._shaper = JoystickBaseShaper.from_configs(cfg, args)
        self._episode = recorder if recorder is not None else StreamingEpisodeRecorder(args.save_dir)
        self._nominal = {group: np.array([ik.nominal_q()[ik._idx_q[n]] for n in names])
                         for group, names in GROUPS.items()}
        self._prev = {}
        self._arm_filt = {}
        self._overstep_ticks = 0
        self._vr = None
        self._held = True
        self._grippers = np.zeros(2)
        self._next_record = self._next_hud = 0.0
        self._publishers = {}
        if args.hud_rate > 0:
            from dexcomm.codecs import RGBImageCodec
            self._publishers = {key: source.node.create_publisher(
                f"sensors/{sensor}/left_rgb", encoder=RGBImageCodec.encode)
                for key, sensor in CAMERA_TOPICS.items()
                if getattr(args, "camera_mode", "all") == "all" or key == "head_left_rgb"}
        self.state = self.rpc("initialize", nominal=self._nominal, urdf=cfg.urdf_path)

    def rpc(self, command, **fields):
        if getattr(self, "_rpc_failed", False):
            raise ConnectionError("Simulator RPC failed; connection is no longer usable")
        try:
            send_msg(self.conn, pack({"cmd": command, **fields}))
            response = unpack(recv_msg(self.conn))
        except (OSError, ConnectionError, EOFError):
            self._rpc_failed = True
            raise
        if "error" in response:
            self._rpc_failed = True
            raise RuntimeError(response["error"])
        return response

    def latch_source_sample(self, vr):
        self._vr = vr

    def engage_reset(self, ik, left0, right0, head0):
        self.state = self.rpc("engage")
        self._shaper.reset(np.zeros(3))
        self._prev.clear()
        self._arm_filt.clear()
        self._overstep_ticks = 0

    def home_to_nominal(self):
        self.stop_all_motion()
        self.stop_recording_episode()
        self.state = self.rpc("home", nominal=self._nominal)

    def extra_hold(self, now, last_cmd_wall, estop):
        from omniteleop.wbc_robot_util import compute_hold_reason
        return compute_hold_reason(estop, last_cmd_wall, now, self.args.source_timeout)

    def stop_all_motion(self):
        self.state = self.rpc("stop")
        self._shaper.hold(np.asarray(self.state["base_pose"]))
        self._prev.clear()
        self._arm_filt.clear()

    def actuate(self, result, left_gripper, right_gripper, enable, hold, dt):
        from omniteleop.wbc_robot_util import clamp_joint_step

        self._held = bool(hold)
        pose = np.asarray(self.state["base_pose"])
        if hold:
            self._shaper.hold(pose)
            self._prev.clear()
            self._arm_filt.clear()
            joints = {g: np.asarray(self.state["joints"][g]) for g in GROUPS}
            base = np.zeros(3)
            self._chassis = {}
        else:
            joints, max_over = {}, 0.0
            for group in GROUPS:
                first = group not in self._prev
                previous = self._prev.get(group, np.asarray(self.state["joints"][group]))
                clamped, requested = clamp_joint_step(previous, np.asarray(getattr(result, group)),
                                                      self.args.max_joint_step)
                max_over = max(max_over, requested)
                self._prev[group] = clamped.copy()
                if group in ("left_arm", "right_arm") and not first:
                    previous_filtered = self._arm_filt.get(group, previous)
                    clamped = previous_filtered + (clamped - previous_filtered) * (
                        dt / (self.args.arm_cmd_lpf_tau + dt))
                    self._arm_filt[group] = clamped.copy()
                joints[group] = clamped
            self._overstep_ticks = self._overstep_ticks + 1 if max_over > 2 * self.args.max_joint_step else 0
            if self.args.max_joint_step > 0 and self._overstep_ticks > self.reference._base._JOINT_STEP_ABORT_TICKS:
                raise RuntimeError("Sustained IK joint step exceeded the real follower's clamp")
            raw = np.array([self._vr.chassis_vx, self._vr.chassis_vy, self._vr.chassis_wz])
            base, pd_raw, pd_error = self._shaper.step(raw, pose, dt)
            self._chassis = {
                "source_body": raw, "intent_body": self._shaper.last_projected_joystick.copy(),
                "command_pose": self._shaper.cmd_pose.copy(), "pd_raw_body": pd_raw,
                "pd_error_body": pd_error, "prev_shaped_body": self._shaper.prev_shaped.copy(),
                "selected_axis": np.int64(-1 if self._shaper.last_intent_axis is None
                                          else self._shaper.last_intent_axis),
            }
            self._grippers = np.clip([left_gripper, right_gripper], 0.0, 1.0)
        self._sent = {n: float(v) for g, names in GROUPS.items() for n, v in zip(names, joints[g])}
        self._base_command = np.asarray(base)
        self.state = self.rpc("step", joints=joints, base=base, grippers=self._grippers)

    def metadata(self):
        from omniteleop.wbc_policy_format import (
            JOYSTICK_SIM_EPISODE_SCHEMA, JOYSTICK_POLICY_ACTION_SCHEMA, JOYSTICK_FIXED_SPEED_MAPPING)
        text = {
            "schema": JOYSTICK_SIM_EPISODE_SCHEMA, "control_mode": "joystick",
            "demonstration_source": "human_teleoperation", "simulator": "OmniGibson",
            "policy_action_schema": JOYSTICK_POLICY_ACTION_SCHEMA,
            "action_target_frame": "current_base", "eef_target_frame": "current_base",
            "head_target_frame": "current_base", "base_reference_source": "joystick_integrator",
            "action_base_pose_semantics": "joystick_command_reference",
            "obs_base_pose_source": "sim_ground_truth", "base_control_pose_source": "sim_ground_truth",
            "chassis_intent_mapping": JOYSTICK_FIXED_SPEED_MAPPING,
            "chassis_intent_path": "action/chassis/intent_body", "chassis_intent_frame": "current_base_body",
            "chassis_intent_stage": "post_mask_projection_pre_controller",
            "chassis_intent_temporal_semantics": "zero_order_hold",
            "episode_timestamp_semantics": "record_commit_wall_ns",
            "urdf_path": self.cfg.urdf_path, "base_dofs": self.cfg.base_dofs, "head_mode": self.cfg.head_mode,
        }
        return {**{k: np.bytes_(v) for k, v in text.items()},
                "chassis_translation_speed_mps": np.float64(self._shaper.translation_speed),
                "chassis_rotation_speed_radps": np.float64(self._shaper.rotation_speed),
                "ik_rate": np.float64(self.args.ik_rate), "record_rate": np.float64(self.args.record_rate),
                "lock_base_in_ik": np.bool_(True), "lock_torso_in_ik": np.bool_(self.cfg.lock_torso_in_ik)}

    def start_recording_if_teleop(self, vr):
        if (vr is None or vr.estop or vr.calib_stage != "teleop" or self._held
                or self._episode.recording):
            return
        if self._episode.saving:
            raise RuntimeError("Previous demo is still saving; wait before engaging")
        static = self.rpc("episode_start")["static"]
        static["meta"] = {**static.get("meta", {}), **self.metadata()}
        self._static = static
        self._episode.set_static(static)
        self._episode.start()
        self._next_record = 0.0
        print(f"\n[behavior] recording episode_{self._episode.episode_id}.hdf5", flush=True)

    def stop_recording_episode(self):
        if self._episode.recording:
            # Complete means the operator ended the take, not that the task succeeded.
            self._episode.set_static({**self._static,
                                      "sim_result": self.rpc("episode_result")})
            path = self._episode.stop()
            if path:
                print(f"\n[behavior] saving {path}", flush=True)

    def record_tick(self, result, hold, now, *, left_target, right_target, head_target,
                    source_timestamp_ns, source_receive_wall_ns):
        if self._episode.last_save_error is not None:
            raise RuntimeError("Demo writer failed") from self._episode.last_save_error
        record = self._episode.recording and not hold and now >= self._next_record
        hud = bool(self._publishers) and now >= self._next_hud
        if not record and not hud:
            return
        frame = self.rpc("capture", effective_targets={"left": left_target, "right": right_target,
                                                       "head": head_target},
                         sent_joints=self._sent, sent_base_twist=self._base_command,
                         base_pose=self._shaper.cmd_pose, grippers=self._grippers)["frame"]
        stamp = time.time_ns()
        if hud:
            for key, publisher in self._publishers.items():
                rgb = frame["obs"]["images"][key]
                publisher.publish({"data": rgb, "timestamp_ns": stamp,
                                   "width": rgb.shape[1], "height": rgb.shape[0]})
            self._next_hud = now + 1 / self.args.hud_rate
        if record:
            frame["timestamp_ns"] = np.int64(stamp)
            frame["action"]["chassis"] = {k: np.asarray(v, dtype=np.float32) if k != "selected_axis" else v
                                           for k, v in self._chassis.items()}
            frame["timing"] = {"source_timestamp_ns": np.int64(source_timestamp_ns),
                               "source_receive_wall_ns": np.int64(source_receive_wall_ns)}
            self._episode.record(frame)
            self._next_record = now + 1 / self.args.record_rate

    def close(self, error=None):
        close_error = None
        try:
            self.stop_all_motion()
        except (ConnectionError, OSError, RuntimeError) as exc:
            close_error = exc
            error = error or exc
        if error is not None:
            self._episode.abort(str(error))
        else:
            try:
                self.stop_recording_episode()
            except Exception as exc:
                close_error = exc
                self._episode.abort(str(exc))
        deadline = time.monotonic() + 30.0
        while self._episode.saving and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._episode.saving:
            raise RuntimeError("Timed out waiting for demo writer")
        if self._episode.last_save_error is not None:
            raise RuntimeError("Demo writer failed") from self._episode.last_save_error
        if close_error is not None:
            raise RuntimeError("Simulator connection failed while closing the demo") from close_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--namespace", default="behavior")
    parser.add_argument("--config", default=str(WBIK_YAML))
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--record-rate", type=float, default=10.0)
    parser.add_argument("--hud-rate", type=float, default=10.0)
    parser.add_argument("--camera-mode", choices=("head", "all"), default="head")
    parser.add_argument("--source-timeout", type=float, default=0.5)
    args = parser.parse_args()
    if (not np.isfinite([args.record_rate, args.hud_rate, args.source_timeout]).all()
            or args.record_rate <= 0 or args.hud_rate < 0 or args.source_timeout <= 0):
        parser.error("record-rate and source-timeout must be positive; hud-rate must be nonnegative")
    args.record = True
    reference = load_reference()
    reference._bind_vr_teleop(args, parser)
    args.max_joint_step = reference._base.DEFAULT_MAX_JOINT_STEP
    args.arm_cmd_lpf_tau = reference._base.DEFAULT_ARM_CMD_LPF_TAU
    args.turn_90 = False
    # Only the two fixed speeds scale; direction selection, PD/slew shaping and the
    # recorded intent labels are the shared ones, and metadata records the scaled values.
    args.joystick_translation_speed *= SIM_BASE_SPEED_SCALE
    args.joystick_rotation_speed *= SIM_BASE_SPEED_SCALE
    if (args.joystick_translation_speed > args.base_max_speed
            or args.joystick_rotation_speed > 2 * args.base_max_speed):
        parser.error(
            f"scaled joystick speeds {args.joystick_translation_speed:.3g} m/s / "
            f"{args.joystick_rotation_speed:.3g} rad/s exceed base_max_speed "
            f"{args.base_max_speed:g} m/s (angular cap {2 * args.base_max_speed:g} rad/s); "
            "JoystickBaseShaper would silently clamp them")
    ik, cfg = reference._build_joystick_ik(args.config)
    source = reference.VRJointSubscriber(args.namespace, name="behavior_joystick_follower")
    conn = socket.socket(fileno=args.fd)
    driver, error = None, None
    try:
        driver = BehaviorDriver(conn, source, ik, cfg, args, reference)
        reference.run_loop(source, ik, cfg, driver, args, replay=False, clock=time.perf_counter,
                           realtime=True, enable=dict(torso=True, arms=True, head=True, base=True))
    except KeyboardInterrupt as exc:
        error = exc
    except BaseException as exc:
        error = exc
        reference._publish_abort_status(source, exc)
        raise
    finally:
        try:
            if driver is not None:
                driver.close(error)
        finally:
            source.close()
            conn.close()


if __name__ == "__main__":
    main()
