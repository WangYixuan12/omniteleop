"""Collect a human Vega demo in BEHAVIOR using scripts/wbc_joystick_leader.py.

Run in the behavior environment. The launcher starts the real joystick control
loop in --dexmate-python and keeps Isaac rendering/physics in this process.
Use --check to inspect prerequisites without starting either simulator or transport.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time
import traceback

from _wire import recv_msg, send_msg

# The follower publishes status at 10 Hz from inside its own control loop and blocks on
# every simulator RPC, so a simulator stage that takes this long freezes teleoperation and
# the leader HUD reports "follower status stale". Name the offending stage when it happens.
SLOW_STAGE_S = 0.5


@contextmanager
def report_slow(label):
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - start
        if elapsed >= SLOW_STAGE_S:
            print(f"[behavior] slow {label}: {elapsed:.2f}s", file=sys.stderr, flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="pickplace", help="Single-stage sim_eval task, or 'none' for an empty scene")
    p.add_argument("--scene", default="Rs_int", help="BEHAVIOR scene model; 'empty' loads a floor only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path.home() / "Dexmate/data/behavior_joystick")
    p.add_argument("--namespace", default="behavior", help="Must match the joystick leader")
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "src/omniteleop/follower/wbik.yaml")
    p.add_argument("--dexmate-python", type=Path,
                   default=Path(os.environ.get("DEXMATE_PYTHON", Path.home() / "miniconda3/envs/dexmate2/bin/python")))
    p.add_argument("--record-rate", type=float, default=10.0)
    p.add_argument("--hud-rate", type=float, default=10.0)
    p.add_argument("--viewer-rate", type=float, default=30.0, help="Maximum render rate; physics remains 200 Hz")
    p.add_argument("--camera-mode", choices=("head", "all"), default="head",
                   help="head renders/records only head RGB (no depth/wrists); all restores wrists, depth, and third-person view")
    p.add_argument("--full-scene", action="store_true", help="Keep all furniture for pickplace instead of the dining-area subset")
    p.add_argument("--source-timeout", type=float, default=0.5)
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--check", action="store_true")
    p.add_argument("--smoke-test", action="store_true", help="Load, home, exercise both grippers, render cameras, then exit")
    p.add_argument("--smoke-output", type=Path, default=Path("/tmp/vega_smoke"))
    return p


def prerequisites(args):
    """Report missing runtime inputs before importing OmniGibson (which launches Isaac)."""
    problems = []
    spec = importlib.util.find_spec("omnigibson")
    if spec is None:
        problems.append("omnigibson is not installed in this Python; run with the behavior environment")
    else:
        from runtime_check import isaac_version_problem
        mismatch = isaac_version_problem()
        if mismatch:
            problems.append(mismatch)
        data = Path(os.environ.get("OMNIGIBSON_DATA_PATH", Path(spec.origin).resolve().parents[2] / "datasets"))
        model = data.expanduser() / "omnigibson-robot-assets/models/vega_robotiq"
        if not (model / "vega_robotiq.yaml").is_file():
            problems.append(f"Missing robot definition: {model / 'vega_robotiq.yaml'} (see sim_eval/ROBOT_ASSETS.md)")
    if not args.config.expanduser().is_file():
        problems.append(f"Missing follower config: {args.config}")
    if not args.dexmate_python.expanduser().is_file():
        problems.append(f"Missing dexmate Python: {args.dexmate_python}")
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        if gpu.returncode:
            problems.append("NVIDIA driver/GPU is unavailable (nvidia-smi failed)")
    except (OSError, subprocess.TimeoutExpired):
        problems.append("Cannot query NVIDIA GPU with nvidia-smi")
    return problems


class SimSession:
    """Simulator-side RPC implementation; the shared real loop owns control and recording."""

    def __init__(self, env, task, args):
        import numpy as np
        self.env, self.task, self.args = env, task, args
        self.recorder = None
        self.grippers = np.zeros(2)
        self.sim_time = 0.0
        self.scene_static = {}
        self._next_render = 0.0

    def ensure_playing(self):
        if not self.env.og.sim.is_playing():
            raise RuntimeError("Simulator timeline was paused or stopped. Restart the collector; use the leader's X/Y buttons to save/exit.")

    def snapshot(self):
        import numpy as np
        from joystick_service import GROUPS
        e = self.env
        self.ensure_playing()
        q = e.robot.get_joint_positions().detach().cpu().numpy()
        base = e.og_to_wbc(e.link_pose("base"))
        return {"base_pose": np.array([base[0, 3], base[1, 3], np.arctan2(base[1, 0], base[0, 0])]),
                "joints": {g: np.array([q[e.name2idx[n]] for n in names]) for g, names in GROUPS.items()}}

    def step(self, joints, base, grippers):
        import numpy as np
        import torch
        from joystick_service import GROUPS
        e = self.env
        self.ensure_playing()
        action = torch.zeros(e.robot.action_dim)
        for group, controller in (("torso", "trunk"), ("head", "camera"),
                                  ("left_arm", "arm_left"), ("right_arm", "arm_right")):
            values = np.asarray(joints[group], dtype=float)
            if values.shape != (len(GROUPS[group]),) or not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid {group} command")
            action[e.cai[controller]] = torch.tensor(values, dtype=torch.float32)
        base, grippers = np.asarray(base), np.asarray(grippers)
        if base.shape != (3,) or grippers.shape != (2,) or not np.all(np.isfinite(np.r_[base, grippers])):
            raise ValueError("Invalid base/gripper command")
        action[e.cai["base"]] = torch.tensor(base, dtype=torch.float32)
        self.grippers = np.clip(grippers, 0.0, 1.0)
        grip = {side: float(1 - 2 * value) for side, value in zip(("left", "right"), self.grippers)}
        for side, value in grip.items():
            action[e.cai[f"gripper_{side}"]] = value
        e.release_on_opening_command(grip)
        # This collector uses a DummyTask and evaluates its own task on save. Run the
        # simulator lifecycle (including contacts and assisted grasping), without reading
        # and discarding every camera's observations on every control tick.
        e.robot.apply_action(action)
        now = time.monotonic()
        render = now >= self._next_render
        with e.og.sim.render_on_step(render):
            with report_slow("sim.step+render" if render else "sim.step"):
                e.og.sim.step()
        if render:
            self._next_render = time.monotonic() + 1.0 / self.args.viewer_rate
        self.ensure_playing()  # GUI events during render can stop physics and invalidate handles.
        with report_slow("note_grasps"):
            e.note_grasps()
        self.sim_time += e.dt
        return self.snapshot()

    def stop(self):
        return self.step(self.snapshot()["joints"], [0, 0, 0], self.grippers)

    def home(self, nominal):
        import numpy as np
        initial = self.snapshot()["joints"]
        for i in range(max(1, round(2.0 / self.env.dt))):
            fraction = min(1.0, (i + 1) * self.env.dt / 2.0)
            # Smoothstep, with zero velocity at each end, while objects remain in place.
            fraction = fraction * fraction * (3 - 2 * fraction)
            joints = {g: initial[g] + fraction * (np.asarray(nominal[g]) - initial[g]) for g in initial}
            self.step(joints, [0, 0, 0], [0, 0])
        return self.snapshot()

    def handle(self, req):
        import numpy as np
        cmd = req["cmd"]
        e = self.env
        if cmd == "initialize":
            from obs_pipeline import SimObsRecorder
            self.home(req["nominal"])
            self.recorder = SimObsRecorder(e, seed=self.args.seed, urdf=req["urdf"],
                                          wrist_cameras=self.args.camera_mode == "all")
            self.recorder.setup()
            return self.snapshot()
        if cmd == "home":
            return self.home(req["nominal"])
        if cmd == "engage":
            e.T_align = e.link_pose("base")
            e.T_align_inv = np.linalg.inv(e.T_align)
            self.scene_static = {"meta": {"task_name": np.bytes_(self.args.task),
                                           "scene_model": np.bytes_(self.args.scene),
                                           "scene_options": np.bytes_(json.dumps(e.scene_options, sort_keys=True)),
                                           "furniture_layout": np.bytes_(getattr(self.task, "LAYOUT_NAME", "scene_default")),
                                           "seed": np.int64(self.args.seed)}}
            self.scene_static["meta"]["camera_mode"] = np.bytes_(getattr(self.args, "camera_mode", "all"))
            if self.task is not None and not getattr(self.task, "UNANNOTATED", False):
                positions = np.asarray(self.task.objects_of_interest(e))
                world = (e.T_align_inv[:3, :3] @ positions.T).T + e.T_align_inv[:3, 3]
                self.scene_static["task"] = {"env_state": world.reshape(-1, 2, 3).astype(np.float32),
                                              "task_sequence": np.asarray(self.task.task_sequence, dtype=np.int64)}
            elif self.task is not None:
                self.scene_static["meta"]["annotation_status"] = np.bytes_("unannotated_human_demo")
                self.scene_static["meta"]["object_physics"] = np.bytes_("rigid_pillow")
            return self.snapshot()
        if cmd == "step":
            return self.step(req["joints"], req["base"], req["grippers"])
        if cmd == "stop":
            return self.stop()
        if cmd == "episode_start":
            return {"static": {**self.scene_static,
                               "obs": {"images": {"intrinsic": self.recorder.intrinsic()}}}}
        if cmd == "episode_result":
            evaluated = self.task is not None and not getattr(self.task, "UNANNOTATED", False)
            return {"task_success": bool(self.task.success(e)) if evaluated else False,
                    "success_evaluated": evaluated}
        if cmd == "capture":
            self.ensure_playing()
            # Fresh images must correspond to this frame's joint measurements.
            with report_slow("capture render"):
                e.og.sim.render()
            self.ensure_playing()
            from types import SimpleNamespace
            semantic = req["grippers"]
            command = SimpleNamespace(gripper_action_left=float(semantic[0]),
                                      gripper_action_right=float(semantic[1]))
            with report_slow("capture recorder.frame"):
                frame = self.recorder.frame(command, time.time_ns(), wbc_resp=req)
            frame["obs"]["base"]["pose"] = self.snapshot()["base_pose"].astype(np.float32)
            frame["obs"]["base"]["pose_sim"] = frame["obs"]["base"]["pose"].copy()
            frame["sim_time_s"] = np.float64(self.sim_time)
            if self.task is not None and not getattr(self.task, "UNANNOTATED", False):
                frame["progress_index"] = np.int64(0)
                frame["active_task_id"] = np.int64(self.task.task_sequence[0])
            return {"frame": frame}
        raise ValueError(f"Unknown simulator command: {cmd}")


def main():
    p = parser()
    args = p.parse_args()
    if (not 0 < args.record_rate <= 100 or not 0 <= args.hud_rate <= 100 or args.source_timeout <= 0):
        p.error("record-rate must be in (0,100], hud-rate in [0,100], and source-timeout positive")
    if not 0 < args.viewer_rate <= 100:
        p.error("viewer-rate must be in (0,100]")
    os.environ["OMNITELEOP_WBIK_YAML"] = str(args.config.expanduser().resolve())
    problems = prerequisites(args)
    if problems:
        for problem in problems:
            print(f"[behavior] {problem}", file=sys.stderr)
        return 2
    if args.check:
        print("BEHAVIOR model definition, config, dexmate Python, and NVIDIA driver found.")
        return 0
    os.environ["OMNIGIBSON_HEADLESS"] = "1" if args.headless else "0"
    import numpy as np
    from vega_og_env import VegaOGEnv
    from tasks import TASKS
    from teleop_protocol import pack, unpack
    import base_ctrl

    if args.task != "none" and args.task not in TASKS:
        p.error(f"Unknown task {args.task}; choose from {', '.join(TASKS)} or none")
    task = None if args.task == "none" else TASKS[args.task](rng=np.random.default_rng(args.seed))
    reference_room = args.task == "pickplace" and args.scene == "Rs_int" and not args.full_scene
    if args.task == "lh_demo" and (args.scene != "Rs_int" or args.full_scene):
        p.error("lh_demo requires --scene Rs_int without --full-scene")
    if reference_room:
        from tasks.episode7_room import Episode7PickPlaceTask
        task = Episode7PickPlaceTask(rng=np.random.default_rng(args.seed))
    if task is not None and len(task.task_sequence) != 1:
        p.error("Choose a single-stage task; human stage annotation is not yet supported")
    env = session = child = None
    interrupted = False
    parent_socket, child_socket = socket.socketpair()
    try:
        rate = int(base_ctrl.VR_TELEOP.ik_rate)
        env = VegaOGEnv(task=task, action_hz=rate, physics_hz=2 * rate, render_hz=rate,
                        scene_model=None if args.scene == "empty" else args.scene,
                        robot_pos=getattr(task, "ROBOT_POS", (0, 0, .03)),
                        robot_yaw=getattr(task, "ROBOT_YAW", 0.0),
                        grasping_mode=getattr(task, "GRASPING_MODE", "physical"),
                        mobile=True, connect_wbc=False, head_only=args.camera_mode == "head",
                        scene_options=({"load_object_categories": ["floors"],
                                        "load_room_instances": ["living_room_0"]}
                                       if reference_room or args.task == "lh_demo" else None))
        env.reset(seed=args.seed, expert=False)
        session = SimSession(env, task, args)
        if args.smoke_test:
            from verify_vega_assets import verify_session
            verify_session(session, args.smoke_output)
            return 0
        command = [str(args.dexmate_python.expanduser()), str(Path(__file__).with_name("joystick_service.py")),
                   "--fd", str(child_socket.fileno()), "--namespace", args.namespace,
                   "--config", str(args.config.expanduser().resolve()),
                   "--save-dir", str(args.out.expanduser().resolve() / "raw_data"),
                   "--record-rate", str(args.record_rate), "--hud-rate", str(args.hud_rate),
                   "--camera-mode", args.camera_mode,
                   "--source-timeout", str(args.source_timeout)]
        child = subprocess.Popen(command, pass_fds=(child_socket.fileno(),))
        child_socket.close()
        # OmniGibson installs an application-level SIGINT handler that immediately closes Isaac.
        # From this point onward the dexmate child owns a recorder, so route Ctrl-C through this
        # launcher's finally block and keep the simulator RPC endpoint alive during child cleanup.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        parent_socket.settimeout(120)  # initial IK construction; the simulator is parked
        print(f"[behavior] Run wbc_joystick_leader.py --namespace {args.namespace} --sim-cameras", flush=True)
        while True:
            try:
                req = unpack(recv_msg(parent_socket))
            except (ConnectionError, EOFError):
                break
            try:
                response = session.handle(req)
            except Exception:
                send_msg(parent_socket, {"error": traceback.format_exc()})
                raise
            send_msg(parent_socket, pack(response))
            if req["cmd"] == "initialize":
                parent_socket.settimeout(35)  # allows bounded recorder close; no physics steps during IPC waits
        if child.wait(timeout=5) != 0:
            raise RuntimeError("Joystick control process failed; inspect its traceback and any .partial demo")
    except KeyboardInterrupt:
        interrupted = True
    except Exception:
        # OmniGibson shutdown exits the interpreter; print failures before entering cleanup.
        traceback.print_exc()
        # Unblock the child's recorder cleanup before waiting for it. Physics may
        # already be invalid, so further stop RPCs must not enter the simulator.
        parent_socket.close()
        raise
    finally:
        # Let the dexmate process abort any active writer and request its final zero-motion
        # command while this simulator endpoint is still alive. Closing this socket first turns
        # an ordinary Ctrl-C into a connection-reset traceback and can interrupt partial-file save.
        if interrupted and child is not None and child.poll() is None:
            parent_socket.settimeout(0.1)
            deadline = time.monotonic() + 35.0
            signal_deadline = time.monotonic() + 1.0
            child_signaled = False
            while child.poll() is None and time.monotonic() < deadline:
                try:
                    req = unpack(recv_msg(parent_socket))
                except socket.timeout:
                    # Interactive Ctrl-C reaches the whole terminal process group, including the
                    # child. A direct signal to only this PID does not, so fall back after giving
                    # the child time to enter its own cleanup. This also avoids a harmful double
                    # SIGINT while the child's recorder is flushing.
                    if not child_signaled and time.monotonic() >= signal_deadline:
                        child.send_signal(signal.SIGINT)
                        child_signaled = True
                    continue
                except (ConnectionError, EOFError, OSError):
                    break
                try:
                    response = session.handle(req)
                except Exception:
                    response = {"error": traceback.format_exc()}
                try:
                    send_msg(parent_socket, pack(response))
                except OSError:
                    break
        if child is not None and child.poll() is None:
            try:
                child.wait(timeout=35)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
        if session is not None:
            try:
                session.stop()
            except Exception as exc:
                print(f"[behavior] Final motion stop unavailable: {exc}", file=sys.stderr)
        parent_socket.close()
        child_socket.close()
        if env is not None:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
