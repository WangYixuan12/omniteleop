"""WBC client — behavior-env side of the WBC bridge.

Talks to `wbc_service.py` (running under the dexmate conda python) over a stdlib
socket, and owns the OmniGibson<->pinocchio joint mapping. Import from the
OmniGibson (behavior env) side:

    from wbc_client import WBCClient
    wbc = WBCClient()                     # auto-spawns the dexmate service subprocess
    wbc.reset()
    resp = wbc.solve(left_T, right_T, head_T, current_q=wbc.build_pin_q(qpos, base_xyyaw))
    # resp["torso"|"left_arm"|"right_arm"|"head"|"base_twist"] -> OmniGibson controller targets

Joint groups (URDF names, shared by OmniGibson vega and the pinocchio model):
"""
import atexit
import os
import socket
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _wire import recv_msg, send_msg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEXMATE_PYTHON = "/home/yixuan/miniforge3/envs/dexmate/bin/python"
SERVICE_PATH = os.path.join(HERE, "wbc_service.py")

TORSO_JOINTS = ["torso_j1", "torso_j2", "torso_j3"]
LARM_JOINTS = [f"L_arm_j{i}" for i in range(1, 8)]
RARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]
HEAD_JOINTS = ["head_j1", "head_j2", "head_j3"]
# Controlled (non-gripper, non-base) joints in pinocchio-q order after the 4-dof planar root.
PIN_JOINT_ORDER = TORSO_JOINTS + LARM_JOINTS + RARM_JOINTS + HEAD_JOINTS  # 20


class WBCError(RuntimeError):
    pass


class WBCClient:
    def __init__(self, host="127.0.0.1", port=5599, auto_spawn=True,
                 dexmate_python=DEXMATE_PYTHON, config=None, lock_base=False, overrides=None,
                 connect_timeout=60.0):
        self.host, self.port = host, port
        self._proc = None
        if auto_spawn:
            self._spawn(dexmate_python, config, lock_base, overrides)
        self.sock = self._connect(connect_timeout)
        atexit.register(self.close)

    # -- process / connection ------------------------------------------------
    def _spawn(self, dexmate_python, config, lock_base=False, overrides=None):
        cmd = [dexmate_python, SERVICE_PATH, "--host", self.host, "--port", str(self.port)]
        if config:
            cmd += ["--config", config]
        if lock_base:
            cmd += ["--lock-base"]
        if overrides:
            import json
            cmd += ["--overrides", json.dumps(overrides)]
        print(f"[wbc_client] spawning WBC service: {' '.join(cmd)}", flush=True)
        self._proc = subprocess.Popen(cmd)

    def _connect(self, timeout):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise WBCError(f"WBC service exited early (code {self._proc.returncode})")
            try:
                s = socket.create_connection((self.host, self.port), timeout=5.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # readiness handshake
                send_msg(s, {"cmd": "ping"})
                if recv_msg(s).get("ok"):
                    print("[wbc_client] connected to WBC service", flush=True)
                    return s
            except (ConnectionRefusedError, OSError) as e:
                last = e
                time.sleep(0.3)
        raise WBCError(f"could not connect to WBC service on {self.host}:{self.port}: {last}")

    def _rpc(self, req):
        send_msg(self.sock, req)
        resp = recv_msg(self.sock)
        if isinstance(resp, dict) and resp.get("error"):
            raise WBCError(resp["error"] + "\n" + resp.get("traceback", ""))
        return resp

    def close(self):
        try:
            if getattr(self, "sock", None) is not None:
                self.sock.close()
                self.sock = None
        except OSError:
            pass
        if getattr(self, "_proc", None) is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    # -- RPCs ----------------------------------------------------------------
    def info(self):
        return self._rpc({"cmd": "info"})

    def reset(self, q=None):
        return self._rpc({"cmd": "reset", "q": None if q is None else np.asarray(q).tolist()})

    def solve(self, left, right, head=None, current_q=None, dt=0.1):
        return self._rpc({
            "cmd": "solve",
            "left": np.asarray(left, dtype=float).reshape(4, 4).tolist(),
            "right": np.asarray(right, dtype=float).reshape(4, 4).tolist(),
            "head": None if head is None else np.asarray(head, dtype=float).reshape(4, 4).tolist(),
            "current_q": None if current_q is None else np.asarray(current_q, dtype=float).tolist(),
            "dt": float(dt),
        })

    # -- joint mapping -------------------------------------------------------
    @staticmethod
    def build_pin_q(qpos_by_name: dict, base_xyyaw) -> np.ndarray:
        """OmniGibson measured joints (name->rad) + base (x,y,yaw) -> pinocchio q(24)."""
        x, y, yaw = base_xyyaw
        q = np.zeros(24)
        q[0], q[1], q[2], q[3] = x, y, np.cos(yaw), np.sin(yaw)
        for i, name in enumerate(PIN_JOINT_ORDER):
            q[4 + i] = qpos_by_name[name]
        return q

    @staticmethod
    def result_to_joint_targets(resp: dict) -> dict:
        """WBCResult dict -> {joint_name: target_rad} for trunk/arm_left/arm_right/camera."""
        out = {}
        for name, val in zip(TORSO_JOINTS, resp["torso"]):
            out[name] = float(val)
        for name, val in zip(LARM_JOINTS, resp["left_arm"]):
            out[name] = float(val)
        for name, val in zip(RARM_JOINTS, resp["right_arm"]):
            out[name] = float(val)
        for name, val in zip(HEAD_JOINTS, resp["head"]):
            out[name] = float(val)
        return out
