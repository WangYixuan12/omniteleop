"""MobilePickPlaceTask: apple->bowl pick-and-place with the robot starting AWAY from the table.

Identical to PickPlaceTask -- SAME consistent expert, SAME waypoints, SAME grasp -- with only: the
base is UNLOCKED (MOBILE=True), the robot spawns back from the table, and a head target is supplied.
There is NO navigate phase and no phase switching: the expert emits the grasp waypoints (and the
head target) from tick 0, and the whole-body IK drives the (free) base forward.

The head target is built the SAME way as the arm's grasp target, from the SAME waypoint list -- it
is a *command*, not a measurement. Only the POSITION rule differs from the fixed-base task (which
supplies the shared gaze orientation, see PickPlaceTask):

    arm : orientation self._Rgrasp  captured ONCE from the reset pose
          position    self._center - Rgrasp @ self._grasp_off_local   (fingertips AT the waypoint)
    head: orientation gaze          heading held, pitched down onto self._center
          position    self._center + self._head_off                   (camera STOOD BACK from it)

``_head_off = head0 - start_center`` is the reset camera-to-hands offset, so the command reads
"stand off from the work point the way you do at rest, and keep looking at it" -- the teleop
semantic wbik.yaml is written against, where the head target is the operator's HEADSET and the
operator's head stays a roughly fixed distance back from their own hands as they walk in and work.

Why it must come from the waypoints rather than be frozen at the spawn pose. Vega's torso is a
sagittal (pitch) chain and ``head_ik_pinned_joints: [head_j1, head_j2]`` pins neck pitch and pan, so
the head CANNOT translate laterally relative to the base: measured over a whole episode,
head_y == base_y + 0.42 to within 1 cm. The head world-position target therefore *is* a base-pose
command (as wbik.yaml says: "x/y track the headset's world position (this drives the
base-following)"). Freezing it at the spawn pose commands "base stays exactly where it spawned",
which directly contradicts what the arm needs -- reaching cross-body to the bowl wants the base
~0.2 m to the right -- and the arm's FrameTask wins the tug of war: measured, the base swept
0.40 -> 0.71 -> 0.18 in y anyway while the head sat 0.21 m off its target and the EE never closed
to better than 0.11 m at the bowl. Offsetting from the live waypoint instead makes the head command
AGREE with the body posture the reach needs, so both objectives are satisfiable at once.

The same offset also supplies the drive: at tick 0 the waypoint is at the fingertips, so the head
target is the spawn pose (no step input); as the waypoint sweeps to the apple the head target sweeps
~0.9 m forward with it, which is exactly the base-following drive. Head-z rides the waypoint too;
the canonical ``head_world_position_cost`` leaves world z free, so the torso retains vertical
reach while world x/y drive the chassis.
"""
from __future__ import annotations

import numpy as np

from .pick_place_task import PickPlaceTask, rot_z


class MobilePickPlaceTask(PickPlaceTask):
    name = "mobilepickplace"
    MOBILE = True
    GRASPING_MODE = "sticky"
    ROBOT_POS = (-0.35, 0.40, 0.03)   # start back from the table; the WBC drives the base up to it
    ROBOT_YAW = 0.0
    BASE_X_MAX = 0.56                 # forward-park clamp: base stops here (front clears table edge)
    PREGRASP_TICKS = 300             # extra ticks for the WBC to drive the base up before grasping
    SETTLE_TICKS = 50                # extra settle: the base is still converging when it parks

    def expert_reset(self, env):
        super().expert_reset(env)
        # Head geometry computed ONCE from the reset pose, exactly like the arm's _Rgrasp/_grasp_off_local.
        # Stored in the BASE frame rather than the world: the stand-back is a body-fixed quantity
        # ("camera this far behind my own hands"), so on a task that TURNS (long_horizon) it has to
        # rotate with the commanded heading or the head command would order the base sideways. At the
        # spawn heading the two coincide, so the straight-in tasks are bit-for-bit unchanged.
        self._head_off_base = rot_z(-self._head_heading) @ (self._head_pos0 - self._start_center)
        print(f"[mobile] head={self._head_link} head0={np.round(self._head_pos0,3)} "
              f"head_off_base={np.round(self._head_off_base,3)}", flush=True)

    def _head_target_pos(self, env, heading):
        """Unlike the fixed-base task the head POSITION matters: it is the base command."""
        return self._center + rot_z(heading) @ self._head_off_base
