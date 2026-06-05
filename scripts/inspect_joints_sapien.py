#!/usr/bin/env python3
"""Interactively inspect the Vega URDF joints in SAPIEN: type a joint, watch it move.

Loads the Vega URDF into a SAPIEN viewer (root fixed, kinematic -- no physics) and
lets you set joint positions from this terminal one at a time, so you can confirm
which URDF joint name (e.g. ``L_arm_j1``) corresponds to which real-robot joint by
seeing exactly which part of the robot rotates.

It is two-threaded: a background thread does blocking line input while the main
thread owns the SAPIEN viewer (the GL context must be rendered from the main
thread) and applies every command, so all robot/state mutation and printing happen
on one thread (no races). The viewer stays responsive while you type.

Joint names / groups / nominal posture come from
:mod:`omniteleop.follower.whole_body_ik` (the canonical source); the joint *list*,
*limits*, and qpos *indices* are read directly from the loaded articulation, so the
tool is correct for whatever URDF is passed.

Commands (type here; watch the SAPIEN window):
    <joint> <value>          set one joint, e.g.  L_arm_j1 1.57   or by index  10 1.57
    <j1> <v1> <j2> <v2> ...  set several joints in one line
    larm/rarm v1 ... v7      set a whole arm (7 values)
    torso v1 v2 v3           set the torso (3 values)
    head  v1 v2 v3           set the head  (3 values)
    list | l                 reprint the joint table with current values
    zero | z                 set all joints to 0
    nominal | nom            set the nominal posture
    help | h | ?             show the command help
    quit | q                 exit (or close the viewer window, or Ctrl-C)

Values are RADIANS; append ``d`` or ``deg`` for degrees (e.g. ``head_j1 30d``).
Out-of-range values are clamped to the URDF joint limits (noted when it happens).

Run with the dexmate conda env, e.g.::

    /home/yixuan/miniforge3/envs/dexmate/bin/python scripts/inspect_joints_sapien.py
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

import numpy as np

from omniteleop.follower.whole_body_ik import (
    DEFAULT_NOMINAL_POSTURE,
    DEFAULT_URDF,
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    TORSO_JOINTS,
    WHEEL_JOINTS,
)

PROMPT = "joint> "
_QUIT = object()  # sentinel queued when stdin reaches EOF

# Logical groups for the joint table and the group-set commands ("larm", etc.).
_GROUP_TABLE = [
    ("torso", TORSO_JOINTS),
    ("head", HEAD_JOINTS),
    ("left arm", LEFT_ARM_JOINTS),
    ("right arm", RIGHT_ARM_JOINTS),
    ("base / wheels", WHEEL_JOINTS),
]
_GROUP_CMDS = {
    "larm": LEFT_ARM_JOINTS,
    "rarm": RIGHT_ARM_JOINTS,
    "torso": TORSO_JOINTS,
    "head": HEAD_JOINTS,
}


def parse_angle(token: str) -> float:
    """Parse an angle token to radians; ``d``/``deg`` suffix means degrees.

    Plain numbers are radians (``1.57``). ``90d`` / ``90deg`` are degrees; ``rad`` is
    accepted for explicitness. Longer suffixes are matched first so ``0.5rad`` is not
    mistaken for the ``d`` (degree) suffix.
    """
    t = token.strip().lower()
    for suffix, to_rad in (("deg", True), ("rad", False), ("d", True), ("r", False)):
        if t.endswith(suffix):
            value = float(t[: -len(suffix)])
            return np.radians(value) if to_rad else value
    return float(t)


class JointModel:
    """Holds the qpos buffer and per-joint metadata read from a SAPIEN articulation.

    Assumes every active joint is 1-DOF, so the active-joint order is exactly the
    qpos order (qpos[i] <-> active_joints[i]) -- the same convention the WBC sim sink
    relies on. Raises if that assumption is violated.
    """

    def __init__(self, robot) -> None:
        active = robot.get_active_joints()
        multi = [j.get_name() for j in active if j.dof != 1]
        if multi:
            raise ValueError(
                f"expected only 1-DOF active joints; got multi-DOF joints {multi}"
            )
        self.names = [j.get_name() for j in active]
        self.name_to_idx = {name: i for i, name in enumerate(self.names)}
        self.limits = np.array([j.get_limits()[0] for j in active], dtype=float)  # (n, 2)
        self.dof = robot.dof
        if self.dof != len(self.names):
            raise ValueError(
                f"robot.dof ({self.dof}) != #active 1-DOF joints ({len(self.names)})"
            )
        self.qpos = np.zeros(self.dof)

    # -- command resolution / mutation -----------------------------------------

    def resolve(self, token: str) -> int:
        """Resolve a joint token (name or qpos index) to a qpos index."""
        if token in self.name_to_idx:
            return self.name_to_idx[token]
        try:
            idx = int(token)
        except ValueError:
            raise ValueError(f"unknown joint {token!r} (not a name or index)") from None
        if not 0 <= idx < self.dof:
            raise ValueError(f"index {idx} out of range 0..{self.dof - 1}")
        return idx

    def _clamp(self, idx: int, value: float) -> tuple[float, bool]:
        lo, hi = self.limits[idx]
        clamped = value
        if np.isfinite(lo):
            clamped = max(clamped, lo)
        if np.isfinite(hi):
            clamped = min(clamped, hi)
        return clamped, clamped != value

    def apply(self, idx: int, value: float) -> None:
        """Set one joint (clamped to its limits) and print what changed."""
        clamped, was_clamped = self._clamp(idx, value)
        self.qpos[idx] = clamped
        name = self.names[idx]
        note = f"  (clamped from {value:+.4f})" if was_clamped else ""
        print(
            f"  {name:<12} [idx {idx:>2}] = {clamped:+.4f} rad "
            f"({np.degrees(clamped):+7.2f} deg){note}"
        )

    def set_zero(self) -> None:
        self.qpos[:] = 0.0
        print("  all joints -> 0")

    def set_nominal(self) -> None:
        """Apply the solver's nominal posture (zero elsewhere), clamped to limits."""
        self.qpos[:] = 0.0
        for name, value in DEFAULT_NOMINAL_POSTURE.items():
            if name in self.name_to_idx:
                idx = self.name_to_idx[name]
                self.qpos[idx] = self._clamp(idx, value)[0]
        print("  set nominal posture (use 'list' to see values)")

    def set_group(self, group: str, value_tokens: list[str]) -> None:
        """Set a whole joint group ("larm"/"rarm"/"torso"/"head") from value tokens."""
        names = [n for n in _GROUP_CMDS[group] if n in self.name_to_idx]
        if len(value_tokens) != len(names):
            print(f"  ! {group} needs {len(names)} values, got {len(value_tokens)}")
            return
        try:
            angles = [parse_angle(v) for v in value_tokens]
        except ValueError as exc:
            print(f"  ! could not parse a value: {exc}")
            return
        for name, angle in zip(names, angles, strict=True):
            self.apply(self.name_to_idx[name], angle)

    # -- table ------------------------------------------------------------------

    def _print_row(self, name: str) -> None:
        idx = self.name_to_idx[name]
        value = self.qpos[idx]
        lo, hi = self.limits[idx]
        limits = (
            f"[{lo:+.2f}, {hi:+.2f}]"
            if np.isfinite(lo) and np.isfinite(hi)
            else "[   free   ]"
        )
        print(
            f"  {name:<12} {idx:>3} {value:>11.4f} {np.degrees(value):>9.2f}  {limits:>18}"
        )

    def print_table(self) -> None:
        """Print the current joint values, grouped, with qpos indices and limits."""
        print()
        print(
            f"  {'joint':<12} {'idx':>3} {'value(rad)':>11} {'(deg)':>9}  "
            f"{'limits(rad)':>18}"
        )
        shown: set[str] = set()
        for title, names in _GROUP_TABLE:
            rows = [n for n in names if n in self.name_to_idx]
            if not rows:
                continue
            print(f"  -- {title} --")
            for name in rows:
                self._print_row(name)
                shown.add(name)
        others = [n for n in self.names if n not in shown]
        if others:
            print("  -- other --")
            for name in others:
                self._print_row(name)
        print()

    # -- one command line -------------------------------------------------------

    def handle(self, line: str) -> bool:
        """Process one command line. Returns True if the user asked to quit."""
        line = line.strip()
        if not line:
            return False
        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("q", "quit", "exit"):
            return True
        if cmd in ("h", "help", "?"):
            print_help()
            return False
        if cmd in ("l", "list", "ls"):
            self.print_table()
            return False
        if cmd in ("z", "zero"):
            self.set_zero()
            return False
        if cmd in ("nom", "nominal"):
            self.set_nominal()
            return False
        if cmd in _GROUP_CMDS:
            self.set_group(cmd, parts[1:])
            return False

        # Otherwise: one or more <joint> <value> pairs.
        if len(parts) % 2 != 0:
            print("  ! expected <joint> <value> pairs (or a command; type 'help')")
            return False
        try:
            updates = [
                (self.resolve(parts[k]), parse_angle(parts[k + 1]))
                for k in range(0, len(parts), 2)
            ]
        except ValueError as exc:
            print(f"  ! {exc}")
            return False
        for idx, value in updates:
            self.apply(idx, value)
        return False


def print_help() -> None:
    """Print the interactive command help (the tail of the module docstring)."""
    print(__doc__[__doc__.index("Commands (") :])


def _stdin_reader(cmd_q: "queue.Queue") -> None:
    """Background thread: push each typed line onto the queue (daemon)."""
    while True:
        try:
            line = input()
        except EOFError:  # piped stdin exhausted / Ctrl-D
            cmd_q.put(_QUIT)
            return
        cmd_q.put(line)


def build_scene(urdf_path: str, ground_z: float = -0.1):
    """Create a SAPIEN scene + viewer and load the (root-fixed, kinematic) robot.

    Mirrors the known-good setup in ``omniteleop.follower.wbc_sapien_sim``: the
    robot faces +x and the camera sits in front of it looking back.
    """
    import sapien  # noqa: PLC0415 - heavy, env-specific dep; imported on use

    scene = sapien.Scene()
    scene.add_ground(ground_z)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=2.5, y=0.0, z=1.5)
    viewer.set_camera_rpy(r=0.0, p=-0.4, y=np.pi)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True  # keep the base pinned so joints are easy to read
    loader.load_multiple_collisions_from_file = True
    robot = loader.load(urdf_path)
    return scene, viewer, robot


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="URDF to load.")
    parser.add_argument(
        "--init",
        choices=("zero", "nominal"),
        default="nominal",
        help="initial posture (default: nominal == DEFAULT_NOMINAL_POSTURE; "
             "use 'zero' for the all-zeros pose).",
    )
    parser.add_argument(
        "--rate", type=float, default=60.0, help="viewer render rate in Hz (default 60)."
    )
    args = parser.parse_args()

    scene, viewer, robot = build_scene(args.urdf)
    model = JointModel(robot)
    if args.init == "nominal":
        model.set_nominal()

    print(f"[inspect_joints] loaded {args.urdf}")
    print(f"[inspect_joints] {model.dof} joints. Type a joint + value; watch the window.")
    print_help()
    model.print_table()

    cmd_q: "queue.Queue" = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(cmd_q,), daemon=True).start()

    sys.stdout.write(PROMPT)
    sys.stdout.flush()
    dt = 1.0 / args.rate
    running = True
    try:
        while running and not viewer.closed:
            handled = False
            while True:
                try:
                    item = cmd_q.get_nowait()
                except queue.Empty:
                    break
                handled = True
                if item is _QUIT or model.handle(item):
                    running = False
                    break
            if handled and running:
                sys.stdout.write(PROMPT)
                sys.stdout.flush()
            robot.set_qpos(model.qpos)
            scene.update_render()
            viewer.render()
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        print("\n[inspect_joints] closed.")


if __name__ == "__main__":
    main()
