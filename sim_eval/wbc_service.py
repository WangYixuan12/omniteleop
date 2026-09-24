#!/usr/bin/env python
"""WBC service — wraps VegaWholeBodyIK behind a stdlib-socket REP server.

Run with the **dexmate conda python** (has pinocchio + pink + qpsolvers + daqp):
    /home/yixuan/miniforge3/envs/dexmate/bin/python sim_eval/wbc_service.py --port 5599

The OmniGibson-side `wbc_client.py` (behavior env) drives it. Only plain float lists
cross the wire (see _wire.py). Requests are dicts with a "cmd":
  - {"cmd":"info"}                              -> {joint_names, nq, nv, q, l_ee, r_ee, head}
  - {"cmd":"reset", "q": [24] or null}          -> {"q":[24]}
  - {"cmd":"solve", "left":[4x4], "right":[4x4], "head":[4x4]|null,
                    "current_q":[24]|null, "dt":float}
        -> {success, torso[3], left_arm[7], right_arm[7], head[3],
            base_twist[3], base_pose[3], l_err, r_err, q[24]}
  - {"cmd":"ping"}                              -> {"ok":True}

pinocchio q layout (confirmed): [0:4] planar root (x, y, cos(yaw), sin(yaw)) |
[4:7] torso_j1..3 | [7:14] L_arm_j1..7 | [14:21] R_arm_j1..7 | [21:24] head_j1..3.
"""
import argparse
import os
import socket
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _wire import recv_msg, send_msg  # noqa: E402
from robot_model import OMNITELEOP_SRC, WBIK_YAML  # noqa: E402

sys.path.insert(0, str(OMNITELEOP_SRC))

import dataclasses  # noqa: E402

from omniteleop.follower.whole_body_ik import VegaWholeBodyIK, WBCConfig  # noqa: E402

DEFAULT_CONFIG = str(WBIK_YAML)
# q index slices for the planar-root pinocchio model
Q_TORSO, Q_LARM, Q_RARM, Q_HEAD = slice(4, 7), slice(7, 14), slice(14, 21), slice(21, 24)


#: Lazily built on the first `policy_fk`, reusing the live solver's model (see that branch).
_POLICY_FK = None


def _frame_pose(ik, name):
    return ik.configuration.get_transform_frame_to_world(name).homogeneous.tolist()


def handle(ik, req):
    cmd = req.get("cmd")
    if cmd == "ping":
        return {"ok": True}
    if cmd == "info":
        return {
            "joint_names": list(ik.model.names),
            "nq": int(ik.model.nq),
            "nv": int(ik.model.nv),
            "q": ik.configuration.q.tolist(),
            "l_ee": _frame_pose(ik, "L_ee"),
            "r_ee": _frame_pose(ik, "R_ee"),
            "head": _frame_pose(ik, "zed_depth_frame"),
        }
    if cmd == "reset":
        q = np.asarray(req["q"], dtype=float) if req.get("q") is not None else None
        ik.reset(q)
        return {"q": ik.configuration.q.tolist()}
    if cmd == "solve":
        left = np.asarray(req["left"], dtype=float).reshape(4, 4)
        right = np.asarray(req["right"], dtype=float).reshape(4, 4)
        head = np.asarray(req["head"], dtype=float).reshape(4, 4) if req.get("head") is not None else None
        current_q = np.asarray(req["current_q"], dtype=float) if req.get("current_q") is not None else None
        res = ik.solve(left, right, dt=float(req["dt"]), head_target=head, current_q=current_q)
        return {
            "success": bool(res.success),
            "torso": np.asarray(res.torso).tolist(),
            "left_arm": np.asarray(res.left_arm).tolist(),
            "right_arm": np.asarray(res.right_arm).tolist(),
            "head": np.asarray(res.head).tolist(),
            "base_twist": np.asarray(res.base_twist).tolist(),
            "base_pose": np.asarray(res.base_pose).tolist(),
            "l_err": float(res.left_ee_error),
            "r_err": float(res.right_ee_error),
            "held": bool(getattr(res, "held", False)),
            "q": np.asarray(res.q).tolist(),
        }
    if cmd == "policy_fk":
        # base_T_{L_ee, R_ee, zed_depth_frame} for `observation.state`, on THIS solver's own
        # model. The porter builds the same block with `WBCPolicyFK` against the same URDF
        # (scripts/port_wbc_mobile_hdf5), and its docstring is explicit that the offline ==
        # online guarantee depends on never using a second model -- so the rollout asks the
        # service that already owns it rather than standing up its own. Stateless: `frame_pose`
        # is evaluated at the q passed in, leaving the solver's integrated configuration alone.
        global _POLICY_FK
        if _POLICY_FK is None:
            from omniteleop.wbc_policy_format import WBCPolicyFK
            _POLICY_FK = WBCPolicyFK(ik)
        poses = _POLICY_FK.base_frame_poses(
            req["torso"], req["left_arm"], req["right_arm"], req["head"])
        return {name: np.asarray(pose, dtype=float).tolist() for name, pose in poses.items()}
    raise ValueError(f"unknown cmd {cmd!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--exit-on-disconnect", action="store_true",
                    help="serve ONE client and exit when it goes away. `wbc_client` passes this "
                         "whenever it auto-spawns the service, because that service belongs to "
                         "exactly one simulator process and nothing will ever reconnect to it")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--lock-base", action="store_true",
                    help="pin all base DOFs (arm+torso-only IK; base_twist == 0)")
    ap.add_argument("--overrides", default=None,
                    help="JSON dict of WBCConfig field overrides (e.g. head costs)")
    args = ap.parse_args()

    import json
    overrides = {}
    if args.lock_base:
        overrides["lock_base_in_ik"] = True
    if args.overrides:
        overrides.update(json.loads(args.overrides))
    print(f"[wbc_service] loading VegaWholeBodyIK({args.config}) overrides={overrides} ...", flush=True)
    cfg = WBCConfig.from_yaml(args.config)
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    ik = VegaWholeBodyIK(config=cfg)
    print(f"[wbc_service] loaded: nq={ik.model.nq} nv={ik.model.nv} head_mode={ik.config.head_mode}", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[wbc_service] listening on {args.host}:{args.port} (READY)", flush=True)

    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[wbc_service] client connected: {addr}", flush=True)
        try:
            while True:
                req = recv_msg(conn)
                try:
                    resp = handle(ik, req)
                except Exception as e:  # send the error back rather than dropping the connection
                    resp = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
                send_msg(conn, resp)
        except (ConnectionError, EOFError):
            print("[wbc_service] client disconnected", flush=True)
        finally:
            conn.close()
        if args.exit_on_disconnect:
            # OmniGibson segfaults on shutdown often enough that the client's atexit hook cannot
            # be relied on to kill us, and the documented collection workflow is one episode per
            # PROCESS in a loop -- so without this every episode leaks a ~1.3 GB solver. Four of
            # them were left running during this session's debugging, holding 5.3 GB. The socket
            # is closed by the kernel when the simulator dies however it dies, so the disconnect
            # above is the one signal that survives a crash.
            print("[wbc_service] client gone; exiting", flush=True)
            return


if __name__ == "__main__":
    main()
