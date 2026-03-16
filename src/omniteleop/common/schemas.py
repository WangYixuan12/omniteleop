"""Simple data schemas for communication between components.

These dataclasses define the structure of messages passed via Zenoh.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class ExoJointData:
    """Joint positions and velocities from the exoskeleton."""

    timestamp_ns: int
    left_arm_pos: List[float] = field(
        default_factory=list
    )  # Left arm joint positions in radians
    left_arm_vel: List[float] = field(
        default_factory=list
    )  # Left arm joint velocities in rad/s
    right_arm_pos: List[float] = field(
        default_factory=list
    )  # Right arm joint positions in radians
    right_arm_vel: List[float] = field(
        default_factory=list
    )  # Right arm joint velocities in rad/s

@dataclass
class JoyConData:
    """JoyCon controller inputs."""

    timestamp_ns: int
    left: Dict[str, Any]  # Left controller data
    right: Dict[str, Any]  # Right controller data

@dataclass
class SafeJointCommand:
    """Safety-validated commands for the robot."""

    timestamp_ns: int
    components: Dict[str, Dict[str, List[float]]]  # Component commands
    safety_flags: Dict[str, bool] = field(default_factory=dict)

@dataclass
class PoseData:
    """Pose data from the VR headset."""

    timestamp_ns: int
    left: Dict[str, Any]
    right: Dict[str, Any]


@dataclass
class VRJointData:
    """IK-solved joint commands streamed by the VR leader (vr_reader.py).

    The leader computes IK from Quest 3 controller and headset poses and
    publishes ready-to-use joint positions. The follower VRHandler subscribes
    to this topic and emits direct JOINT commands, mirroring the flow of
    ExoJointData for the exoskeleton pipeline.
    """

    timestamp_ns: int
    # Arm joint positions (7-DOF each), radians. Empty = not yet calibrated.
    left_arm_pos: List[float] = field(default_factory=list)
    right_arm_pos: List[float] = field(default_factory=list)
    # Head joint positions (3-DOF), radians
    head_pos: List[float] = field(default_factory=list)
    # Gripper scale [0.0 = open, 1.0 = fully closed] mapped from controller trigger
    left_gripper: float = 0.0
    right_gripper: float = 0.0
    # Control flags
    estop: bool = True          # Default True (safe) until calibration completes
    exit_requested: bool = False
    recalibrate: bool = False
    # Calibration stage: A=head, B=arms preview, C=live teleop
    calib_stage: str = "A"
