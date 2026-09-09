"""SimTask ABC — a pluggable task for the Vega OmniGibson sim eval pipeline.

A task owns: the objects it spawns, per-episode randomization, the success predicate,
the two "objects of interest" (source, target) whose GT world XYZ become the 6-D
env_state SceneDiff condition, and a scripted expert that emits world-frame EEF targets.

The env (`vega_og_env.VegaOGEnv`) is task-agnostic: it builds the robot + scene, runs the
WBC control loop, and renders obs; the task supplies scene objects + expert behavior.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ExpertCommand:
    """One control-tick command from a task's scripted expert (world frame, OG coords).

    If `base_vel` is set the env DRIVES the base (navigate phase, arms held); otherwise it
    runs the WBC toward the EEF targets (manipulate phase, base held).
    """
    left_target: Optional[np.ndarray] = None    # (4,4) desired L_ee world pose (manipulate)
    right_target: Optional[np.ndarray] = None   # (4,4) desired R_ee world pose (manipulate)
    head_target: Optional[np.ndarray] = None    # (4,4) desired zed_depth_frame world pose, or None to hold
    base_vel: Optional[np.ndarray] = None        # (vx,vy,wz) base-frame twist for the navigate phase
    gripper_left: float = -1.0       # task actuator convention: +1 open, -1 close
    gripper_right: float = -1.0
    # Policy semantics, independent of actuator calibration: 0=open, 1=closed.  A task may
    # deliberately stop a rigid-object pinch at an intermediate Robotiq command, so equality with
    # ``env.grip_close`` cannot recover this label after the fact.
    gripper_action_left: float = 1.0
    gripper_action_right: float = 1.0
    done: bool = False               # expert signals episode end (reached terminal state)
    phase: str = ""                  # human-readable phase label (for logging/video overlay)


class SimTask(ABC):
    """Base class for a co-located-objects mobile-bimanual task."""

    #: short id used for --task selection and output naming
    name: str = "base"
    #: grasping mode for the robot ("physical" kinematic-grasp tasks, "sticky" for real AG grasp)
    GRASPING_MODE: str = "physical"
    #: True => the WBC drives the base (mobile manipulation); False => base locked (tabletop)
    MOBILE: bool = False
    #: base spawn pose the env should use (world x,y,z) and yaw; tabletop tasks put it at the table
    ROBOT_POS: tuple = (-0.5, 0.4, 0.03)
    ROBOT_YAW: float = 0.0

    #: category names spawned; the first two are the [source, target] objects of interest
    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng(0)
        self.objects: dict = {}      # name -> OmniGibson object handle, filled by build()

    # ---- scene ----
    @abstractmethod
    def object_configs(self) -> list:
        """Return OmniGibson object config dicts for this task (added at env creation)."""

    @abstractmethod
    def bind(self, env) -> None:
        """Grab object handles from the created env (called once, after env creation)."""

    @abstractmethod
    def reset(self, env) -> None:
        """Randomize object poses for a new episode (called each episode, after play)."""

    # ---- SceneDiff condition ----
    @abstractmethod
    def objects_of_interest(self, env) -> np.ndarray:
        """Return (2N,3) world XYZ, [source, target] per grasp cycle, for the env_state
        condition. N is 1 for a single-stage task."""

    # ---- stage ----
    @property
    def stage_index(self) -> int:
        """Which grasp cycle the expert is in RIGHT NOW, 0-based; always 0 for one stage.

        Recorded per frame as `progress_index`, which is both what the structured porter
        requires (`port_wbc_mobile_hdf5`: (T,) int64, monotonic 0->1->2) and what the policy
        learns to predict with its stage head. An evaluator needs it too: the actuator command
        behind a 0..1 close intent is per-stage (see `LHSpec.grip_close_cmd`), so without a
        stage label a recorded or predicted gripper action cannot be turned back into a command.
        """
        return 0

    @property
    def task_sequence(self) -> tuple:
        """Task id occupying each stage slot, in execution order.

        The model maps progress -> task_sequence[progress] and uses that to select the
        environment-state row, so this is identity whenever the env_state rows are already
        written in execution order.
        """
        return (0,)

    # ---- success ----
    @abstractmethod
    def success(self, env) -> bool:
        """True when the task goal is achieved (OnTop/Inside/moved-to-target)."""

    # ---- scripted expert ----
    @abstractmethod
    def expert_reset(self, env) -> None:
        """Reset the expert's internal state machine at episode start."""

    @abstractmethod
    def expert_step(self, env, obs: dict) -> ExpertCommand:
        """Return the next-tick world-frame EEF targets + gripper commands."""
