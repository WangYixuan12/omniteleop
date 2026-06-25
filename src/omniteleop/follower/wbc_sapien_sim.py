"""Robot sinks for the whole-body-control SAPIEN demo.

Both backends present the **same high-level motion-control surface as dexmate's**
``dexcontrol.Robot`` -- so the WBC dispatch code in ``scripts/wbc_sim_sapien.py`` is
identical whether it drives a simulation or real hardware:

- arms / torso / head -> ``set_joint_pos(q)``      (joint position control)
- mobile base        -> ``chassis.set_velocity(vx, vy, wz)``  (velocity control)

``SapienSimRobot`` applies those commands to a SAPIEN articulation (joints via
``set_qpos``; base velocity integrated into a planar root pose via
``set_root_pose``) and renders with a SAPIEN viewer set up like
``omniteleop.leader.vr_reader``. It runs in pure simulation -- no robot, no Zenoh.

``DexcontrolRobotSink`` is a thin wrapper over the real ``dexcontrol.Robot`` so the
same demo drives hardware with ``--real``.

dexmate high-level API reference (from the installed ``dexcontrol`` package):
``core/arm.py:230`` ``Arm.set_joint_pos(joint_pos, wait_time=0.0, ...)`` (send-now
when ``wait_time=0``); ``core/torso.py:133`` / ``core/head.py:84`` analogous;
``core/chassis.py:236`` ``Chassis.set_velocity(vx, vy, wz, sequential_steering=...)``.
There is no absolute base-pose API, so the base uses velocity control.
"""

# Inner component classes (_SimJointGroup/_SimChassis) intentionally write to their
# parent SapienSimRobot's state, so flake8-self private-access checks are disabled here.
# ruff: noqa: SLF001

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Sequence

import numpy as np

from omniteleop.follower.sapien_env import prepare_sapien_render_env
from omniteleop.follower.sapien_video import (
    VideoRecordingConfig,
    make_video_recording_config,
)
from omniteleop.follower.whole_body_ik import (
    DEFAULT_URDF,
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    TORSO_JOINTS,
)

# ---- robot-sink interface ----------------------------------------------------

class JointComponent(Protocol):
    """A joint group controlled by position (subset of ``dexcontrol`` Arm/Torso/Head)."""

    def set_joint_pos(self, joint_pos: Sequence[float]) -> None:
        """Command the group's joint positions (radians)."""
        ...


class ChassisComponent(Protocol):
    """A mobile base controlled by planar velocity (subset of ``dexcontrol`` Chassis)."""

    def set_velocity(self, vx: float, vy: float, wz: float) -> None:
        """Command the base velocity (m/s, m/s, rad/s) in the base frame."""
        ...


class RobotSink(Protocol):
    """High-level robot interface the WBC demo dispatches through.

    Mirrors the subset of ``dexcontrol.Robot`` the whole-body controller drives.
    """

    left_arm: JointComponent
    right_arm: JointComponent
    torso: JointComponent
    head: JointComponent
    chassis: ChassisComponent

    def step(self, dt: float) -> None:
        """Advance the backend by ``dt`` seconds (apply commanded state)."""
        ...

    def reset_base(self) -> None:
        """Return the mobile base to the world origin (sim only; no-op on hardware)."""
        ...

    def render(self, record_video: bool = True) -> None:
        """Render the current state, if the backend has a viewer."""
        ...

    def close(self) -> None:
        """Release resources / shut down the backend."""
        ...


@dataclass
class BaseVelocityController:
    """The single, official-aligned base-velocity command path.

    Every base command is issued as one coordinated, send-now
    ``chassis.set_velocity(vx, vy, wz, wait_time=0.0, sequential_steering=False)`` -- the
    way the real Vega base is driven (``deps`` ``dexcontrol/core/chassis.py``;
    ``examples/advanced_examples/chassis_s_curve.py``). The full ``(vx, vy, wz)`` is
    always sent together as a coordinated command.

    On top of that it adds three hardware-safety behaviors applied to the velocity stream:

    - **deadband**: an axis below ``deadband_lin``/``deadband_ang`` is zeroed, so
      solver/sensor jitter doesn't make the wheels chatter and constantly re-steer;
    - **slew limiting**: each axis may change by at most ``max_*_accel * dt`` per tick, so
      a step change in the command becomes a bounded acceleration (no jerk);
    - **e-stop**: :meth:`estop` latches the base to zero immediately (no slew) until
      :meth:`release`; a per-call ``hold=True`` is a momentary stop for transient safety
      (e.g. a failed IK solve).

    The shaped command is returned by :meth:`command` (and cached on :attr:`commanded`),
    so callers can show the difference between the raw target and what the base was given.
    """

    dt: float
    deadband_lin: float = 0.01       # m/s   (below this, a translation axis is zeroed)
    deadband_ang: float = 0.02       # rad/s
    max_lin_accel: float = 2.0       # m/s^2 (slew limit on vx, vy)
    max_ang_accel: float = 4.0       # rad/s^2 (slew limit on wz)
    max_lin_vel: float = 1.0         # m/s   (envelope clamp)
    max_ang_vel: float = 1.0         # rad/s

    def __post_init__(self) -> None:
        self._prev = np.zeros(3)
        self._estopped = False

    @property
    def commanded(self) -> np.ndarray:
        """The last twist actually sent to the chassis (after shaping)."""
        return self._prev.copy()

    @property
    def estopped(self) -> bool:
        """True while the latched e-stop is engaged."""
        return self._estopped

    def estop(self) -> None:
        """Latch an emergency stop: the base is held at zero (no slew) until released."""
        self._estopped = True
        self._prev = np.zeros(3)

    def release(self) -> None:
        """Clear the latched e-stop."""
        self._estopped = False

    def command(
        self, robot: RobotSink, twist: np.ndarray, hold: bool = False
    ) -> np.ndarray:
        """Shape ``twist`` (deadband -> clamp -> slew, or zero on e-stop) and send it.

        Returns the twist actually commanded (after shaping), for logging/visualization.
        """
        twist = np.asarray(twist, dtype=float).ravel()
        if twist.shape != (3,):
            raise ValueError(f"twist must be (vx, vy, wz); got shape {twist.shape}")
        if hold or self._estopped:
            target = np.zeros(3)
        else:
            target = twist.copy()
            db = (self.deadband_lin, self.deadband_lin, self.deadband_ang)
            vmax = (self.max_lin_vel, self.max_lin_vel, self.max_ang_vel)
            dmax = (self.max_lin_accel * self.dt, self.max_lin_accel * self.dt,
                    self.max_ang_accel * self.dt)
            for i in range(3):
                if abs(target[i]) < db[i]:
                    target[i] = 0.0                                  # deadband
                target[i] = float(np.clip(target[i], -vmax[i], vmax[i]))  # envelope
                step = float(np.clip(target[i] - self._prev[i], -dmax[i], dmax[i]))
                target[i] = self._prev[i] + step                    # slew limit
        self._prev = target
        robot.chassis.set_velocity(
            vx=float(target[0]), vy=float(target[1]), wz=float(target[2]),
            wait_time=0.0, sequential_steering=False,
        )
        return target.copy()


@dataclass
class TwistLowPassFilter:
    """First-order low-pass for body twists, used before acceleration limiting.

    A sparse target stream can make the IK emit one-tick velocity bursts. A plain slew
    limiter clips those bursts and loses displacement. This filter spreads each burst over
    several ticks while preserving DC gain, so the downstream slew limiter can follow it.
    """

    dt: float
    tau: float

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.tau < 0.0:
            raise ValueError(f"tau must be >= 0, got {self.tau}")
        self._state = np.zeros(3)

    def reset(self, value: Optional[np.ndarray] = None) -> None:
        """Reset the filter state, defaulting to zero velocity."""
        if value is None:
            self._state = np.zeros(3)
            return
        state = np.asarray(value, dtype=float).ravel()
        if state.shape != (3,):
            raise ValueError(f"value must be (vx, vy, wz); got shape {state.shape}")
        self._state = state.copy()

    def filter(self, twist: np.ndarray) -> np.ndarray:
        """Return the low-passed twist for this tick."""
        twist = np.asarray(twist, dtype=float).ravel()
        if twist.shape != (3,):
            raise ValueError(f"twist must be (vx, vy, wz); got shape {twist.shape}")
        if self.tau == 0.0:
            self._state = twist.copy()
        else:
            alpha = float(np.exp(-self.dt / self.tau))
            self._state = alpha * self._state + (1.0 - alpha) * twist
        return self._state.copy()


# PD gains for the base-tracking controller (see base_tracking_twist).
BASE_KP_XY = 3.0
BASE_KP_YAW = 4.0
BASE_VEL_LIMIT = 1.0  # m/s and rad/s clamp on the commanded base twist


def base_tracking_twist(
    ideal_pose: np.ndarray, ideal_twist: np.ndarray, measured_pose: np.ndarray
) -> np.ndarray:
    """Body-frame base twist that makes the *rendered/physical* base track a reference.

    The whole-body solver runs in its own base frame and emits a base trajectory
    (``ideal_pose``, ``ideal_twist``); the velocity-driven base (wheel slip in physics, or
    deadband/slew shaping of the command in the kinematic sim) lags/drifts from it. This
    adds proportional feedback on the world-frame pose error (rotated into the measured
    base frame) to the feed-forward twist (whose xy is likewise rotated from the ideal into
    the measured base frame), so the base is steered back onto the reference -- which keeps
    the rendered end-effectors on the solver's targets. The result is scaled uniformly to
    the base velocity limit, preserving its direction. Feed it to
    :meth:`BaseVelocityController.command`.
    """
    ideal_pose = np.asarray(ideal_pose, dtype=float)
    ideal_twist = np.asarray(ideal_twist, dtype=float)
    measured_pose = np.asarray(measured_pose, dtype=float)
    err = ideal_pose - measured_pose
    err[2] = (err[2] + np.pi) % (2.0 * np.pi) - np.pi  # shortest-arc yaw error (ideal-meas)
    # Feedback: world-frame pose error -> measured base frame.
    c, s = np.cos(-measured_pose[2]), np.sin(-measured_pose[2])
    fb = np.array([
        BASE_KP_XY * (c * err[0] - s * err[1]),
        BASE_KP_XY * (s * err[0] + c * err[1]),
        BASE_KP_YAW * err[2],
    ])
    # Feed-forward: the solver's twist is in its IDEAL base frame; rotate its xy into the
    # measured base frame (by ideal_yaw - measured_yaw = err[2]) before adding, else it
    # points the wrong way whenever the two base yaws differ.
    cf, sf = np.cos(err[2]), np.sin(err[2])
    ff = np.array([
        cf * ideal_twist[0] - sf * ideal_twist[1],
        sf * ideal_twist[0] + cf * ideal_twist[1],
        ideal_twist[2],
    ])
    twist = ff + fb
    # Scale the whole twist uniformly so no component exceeds the limit (per-component
    # clipping would bend the commanded direction under saturation).
    m = float(np.max(np.abs(twist))) / BASE_VEL_LIMIT
    return twist / m if m > 1.0 else twist


# ---- SAPIEN simulation backend -----------------------------------------------

class _SimJointGroup:
    """A named joint group whose ``set_joint_pos`` writes into the sim qpos buffer."""

    def __init__(self, robot: "SapienSimRobot", joint_names: Sequence[str]) -> None:
        self._robot = robot
        self._names = list(joint_names)

    def set_joint_pos(self, joint_pos: Sequence[float]) -> None:
        """Write the group's joint positions into the parent's qpos buffer."""
        q = np.asarray(joint_pos, dtype=float).ravel()
        if q.shape[0] != len(self._names):
            raise ValueError(
                f"expected {len(self._names)} joint values, got {q.shape[0]}"
            )
        for name, val in zip(self._names, q, strict=True):
            self._robot._qpos[self._robot._sapien_idx[name]] = val


class _SimChassis:
    """Sim chassis storing a coordinated base twist integrated in ``step``."""

    def __init__(self, robot: "SapienSimRobot") -> None:
        self._robot = robot

    def set_velocity(self, vx: float, vy: float, wz: float, **_: float) -> None:
        """Store the commanded base velocity (integrated on the next step)."""
        self._robot._base_twist = np.array([vx, vy, wz], dtype=float)

    def stop(self) -> None:
        """Zero the commanded base velocity."""
        self._robot._base_twist = np.zeros(3)


class SapienSimRobot:
    """SAPIEN-backed robot sink (kinematic; no physics stepping).

    Args:
        urdf_path: Vega URDF (same one the solver loads).
        with_viewer: open an interactive SAPIEN viewer (needs a display). When
            False, the robot still tracks state kinematically (``set_qpos`` /
            integrated base pose) but does not render -- useful for headless
            verification.
        ground_z: ground plane height.
        show_targets: draw the dual-arm target markers (only when a viewer exists).
    """

    def __init__(
        self,
        urdf_path: str = DEFAULT_URDF,
        with_viewer: bool = True,
        ground_z: float = -0.1,
        show_targets: bool = True,
        record_video: Optional[str] = None,
        video_fps: float = 15.0,
        video_size: tuple[int, int] = (960, 540),
    ) -> None:
        prepare_sapien_render_env()
        import sapien  # noqa: PLC0415 - heavy, env-specific dep; imported on use

        self._sapien = sapien
        scene = sapien.Scene()
        scene.add_ground(ground_z)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
        self._scene = scene

        self._viewer = None
        if with_viewer:
            self._viewer = scene.create_viewer()
            # In front of the robot (it faces +x), looking back at it (yaw=pi).
            self._viewer.set_camera_xyz(x=2.5, y=0.0, z=1.5)
            self._viewer.set_camera_rpy(r=0.0, p=-0.4, y=np.pi)

        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        self._robot = loader.load(urdf_path)
        for link in self._robot.get_links():
            for shape in link.get_collision_shapes():
                shape.set_collision_groups([1, 1, 17, 0])
        _links = {lk.name: lk for lk in self._robot.get_links()}
        self._ee_links = {"L": _links["L_ee"], "R": _links["R_ee"]}

        # Active-joint order is interleaved, so always index by name.
        self._sapien_idx = {
            j.name: i for i, j in enumerate(self._robot.get_active_joints())
        }
        self._qpos = np.zeros(self._robot.dof)
        self._base_xy = np.zeros(2)
        self._base_yaw = 0.0
        self._base_twist = np.zeros(3)
        self._base_z = float(self._robot.get_root_pose().p[2])
        self._sim_time = 0.0  # accumulated simulated time, for video-frame decimation

        # Components mirroring dexcontrol.Robot.
        self.left_arm = _SimJointGroup(self, LEFT_ARM_JOINTS)
        self.right_arm = _SimJointGroup(self, RIGHT_ARM_JOINTS)
        self.torso = _SimJointGroup(self, TORSO_JOINTS)
        self.head = _SimJointGroup(self, HEAD_JOINTS)
        self.chassis = _SimChassis(self)

        # Markers/frames are worth drawing whenever something renders (viewer/video).
        render_active = self._viewer is not None or record_video is not None
        if render_active:
            self._add_origin_frame()  # world-origin XYZ triad (X red, Y green, Z blue)
        self._show_targets = show_targets and render_active
        if self._show_targets:
            # Targets are drawn as XYZ coordinate triads (full pose, not just position):
            # X red, Y green, Z blue, smaller than the world-origin frame. The head target
            # is optional; WBC VR uses it to show desired gaze orientation.
            self._l_target = self._make_target_frame()
            self._r_target = self._make_target_frame()
            self._head_target = self._make_target_frame(length=0.18, radius=0.006)

        # Optional offscreen camera + video recorder.
        self._writer = None
        self._camera = None
        self._video_config: Optional[VideoRecordingConfig] = None
        self._video_path: Optional[str] = None
        self._video_frame_count = 0
        if record_video is not None:
            self._video_config = make_video_recording_config(
                record_video, video_fps, video_size
            )
            self._video_path = self._video_config.path
            self._frame_interval = 1.0 / self._video_config.fps
            self._next_frame_sim_time = 0.0

        self._robot.set_qpos(self._qpos)
        self._apply_base_pose()

    # -- video recording -------------------------------------------------------

    def _setup_recording(self, config: VideoRecordingConfig) -> None:
        import cv2  # noqa: PLC0415 - optional dep, only needed when recording

        path = config.path
        width, height = config.size
        self._camera = self._scene.add_camera("recorder", width, height, 1.0, 0.05, 100.0)
        # Fixed world vantage point (does NOT follow the robot), in front of the robot
        # (it faces +x) so the base driving around the origin is clearly visible.
        self._camera.set_local_pose(self._look_at(eye=[3.0, 0.0, 1.6], target=[0.3, 0.0, 0.9]))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, config.fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open a video writer for {path!r}")
        self._video_path = path
        print(f"[sapien] recording video -> {path} ({width}x{height} @ {config.fps:g} fps)")

    def _look_at(self, eye, target, up=(0.0, 0.0, 1.0)):
        """Camera pose looking from ``eye`` to ``target`` (SAPIEN +x-forward frame)."""
        from scipy.spatial.transform import Rotation  # noqa: PLC0415

        eye = np.asarray(eye, dtype=float)
        fwd = np.asarray(target, dtype=float) - eye
        fwd /= np.linalg.norm(fwd)
        left = np.cross(np.asarray(up, dtype=float), fwd)
        left /= np.linalg.norm(left)
        up2 = np.cross(fwd, left)
        rot = np.column_stack([fwd, left, up2])  # +x forward, +y left, +z up
        qx, qy, qz, qw = Rotation.from_matrix(rot).as_quat()
        return self._sapien.Pose(p=eye, q=[qw, qx, qy, qz])

    # -- markers / frames ------------------------------------------------------

    def _add_triad_visuals(self, builder, length: float, radius: float) -> None:
        """Add three capsules forming an XYZ triad (X red, Y green, Z blue) to ``builder``."""
        sapien = self._sapien
        half = length / 2.0
        # A capsule's long axis is its local +x, so orient each to a body axis:
        # X = identity, Y = +90 deg about z, Z = -90 deg about y. Each spans [0, length].
        builder.add_capsule_visual(
            sapien.Pose(p=[half, 0.0, 0.0]),
            radius=radius, half_length=half, material=[1.0, 0.1, 0.1])
        builder.add_capsule_visual(
            sapien.Pose(p=[0.0, half, 0.0], q=[0.70710678, 0.0, 0.0, 0.70710678]),
            radius=radius, half_length=half, material=[0.1, 1.0, 0.1])
        builder.add_capsule_visual(
            sapien.Pose(p=[0.0, 0.0, half], q=[0.70710678, 0.0, -0.70710678, 0.0]),
            radius=radius, half_length=half, material=[0.1, 0.1, 1.0])

    def _make_target_frame(self, length: float = 0.15, radius: float = 0.008):
        """Build a kinematic XYZ coordinate triad used to mark an EEF target pose."""
        builder = self._scene.create_actor_builder()
        self._add_triad_visuals(builder, length, radius)
        return builder.build_kinematic()

    def _add_origin_frame(self, length: float = 0.3, radius: float = 0.01) -> None:
        """Draw an XYZ coordinate triad (X red, Y green, Z blue) at the world origin."""
        builder = self._scene.create_actor_builder()
        self._add_triad_visuals(builder, length, radius)
        self._origin_frame = builder.build_kinematic()
        self._origin_frame.set_pose(self._sapien.Pose())  # world origin

    def _set_marker_pose(self, marker, pose) -> None:
        """Set a marker's full pose from a 4x4 matrix or a ``pin.SE3``."""
        from scipy.spatial.transform import Rotation  # noqa: PLC0415

        mat = np.asarray(pose.homogeneous if hasattr(pose, "homogeneous") else pose, dtype=float)
        if mat.shape != (4, 4):
            raise ValueError(f"target pose has shape {mat.shape}, expected (4, 4)")
        qx, qy, qz, qw = Rotation.from_matrix(mat[:3, :3]).as_quat()
        marker.set_pose(self._sapien.Pose(p=list(mat[:3, 3]), q=[qw, qx, qy, qz]))

    def set_targets(self, left_pose, right_pose, head_pose=None) -> None:
        """Place target coordinate frames (world-frame 4x4 poses or ``pin.SE3``)."""
        if self._show_targets:
            self._set_marker_pose(self._l_target, left_pose)
            self._set_marker_pose(self._r_target, right_pose)
            if head_pose is not None:
                self._set_marker_pose(self._head_target, head_pose)

    # -- base pose -------------------------------------------------------------

    def _apply_base_pose(self) -> None:
        x, y = self._base_xy
        half = self._base_yaw / 2.0
        quat = [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]  # (w, x, y, z)
        self._robot.set_root_pose(self._sapien.Pose(p=[x, y, self._base_z], q=quat))

    @property
    def base_pose(self) -> np.ndarray:
        """Integrated mobile-base pose ``(x, y, yaw)`` in the world frame."""
        return np.array([self._base_xy[0], self._base_xy[1], self._base_yaw])

    def ee_positions(self) -> Dict[str, np.ndarray]:
        """World positions of the ``L_ee``/``R_ee`` links (the rendered end-effectors).

        When the base is driven through :class:`BaseVelocityController`, the integrated
        base no longer follows the solver's exact twist (deadband/slew), so these lag the
        raw-target markers -- comparing them shows the command-shaping effect.
        """
        return {
            k: np.asarray(link.get_pose().p, dtype=float)
            for k, link in self._ee_links.items()
        }

    def set_base_pose(self, x: float, y: float, yaw: float) -> None:
        """Teleport the mobile base to an absolute world pose ``(x, y, yaw)``.

        Sets the base pose directly and zeroes the pending velocity, so a subsequent
        ``step`` integrates no twist on top of it. Useful for **rendering an
        authoritative whole-body solution**: the solver already integrates the base
        twist into an exact planar pose (``WBCResult.base_pose``), so a demo can show
        that pose verbatim instead of re-integrating the velocity command.
        """
        self._base_xy = np.array([float(x), float(y)], dtype=float)
        self._base_yaw = float(yaw)
        self._base_twist = np.zeros(3)
        self._apply_base_pose()

    def reset_base(self) -> None:
        """Teleport the mobile base back to the world origin (zero pose and velocity).

        Used when the VR leader re-calibrates: the leader re-anchors its EEF targets
        to the nominal (origin) base frame, so the sim base must return to the origin
        to match, clearing any drift accumulated during the previous teleop session.
        The joint qpos buffer is left as-is (the next ``set_joint_pos`` from the solver
        overwrites it with the reset posture).
        """
        self.set_base_pose(0.0, 0.0, 0.0)

    # -- lifecycle -------------------------------------------------------------

    def step(self, dt: float) -> None:
        """Integrate the commanded base velocity and push joint state to SAPIEN.

        The base twist ``(vx, vy, wz)`` is body-frame (x fwd, y left). It is integrated
        with the **exact SE(2) exponential** -- the same planar-joint integration the
        whole-body solver uses (``pin.JointModelPlanar``) -- so when the full coordinated
        twist is commanded (``chassis.set_velocity``) the rendered base tracks the
        solver's base pose to machine precision, instead of the ~mm drift a semi-implicit
        Euler step accumulates while translating and turning at once.
        """
        self._sim_time += dt
        vx, vy, wz = self._base_twist
        theta = wz * dt
        if abs(theta) < 1e-9:  # no rotation this step: straight body-frame translation
            dx_body, dy_body = vx * dt, vy * dt
        else:                  # arc traced by a constant body twist over dt (SE(2) exp)
            sin_t, cos_t = np.sin(theta), np.cos(theta)
            dx_body = (vx * sin_t + vy * (cos_t - 1.0)) / wz
            dy_body = (vx * (1.0 - cos_t) + vy * sin_t) / wz
        c, s = np.cos(self._base_yaw), np.sin(self._base_yaw)  # rotate by the *start* yaw
        self._base_xy += np.array([c * dx_body - s * dy_body, s * dx_body + c * dy_body])
        self._base_yaw += theta
        self._robot.set_qpos(self._qpos)
        self._apply_base_pose()

    def render(self, record_video: bool = True) -> None:
        """Update the scene; render to the viewer and/or write a video frame.

        Video frames are decimated to the recording fps: the control loop runs much
        faster (e.g. 100 Hz), so only one frame per ``1/fps`` of simulated time is
        captured, giving a real-time video at the requested fps. The viewer (when
        present) still renders every tick for smoothness.
        """
        has_pending_video = self._video_config is not None
        if self._viewer is None and self._camera is None and not (
            record_video and has_pending_video
        ):
            return
        if has_pending_video and not record_video:
            self._next_frame_sim_time = self._sim_time
        if record_video and has_pending_video and self._writer is None:
            self._next_frame_sim_time = self._sim_time
        write_due = (
            record_video and
            has_pending_video and
            self._sim_time >= self._next_frame_sim_time - 1e-9
        )
        if write_due and self._writer is None:
            self._setup_recording(self._video_config)
        write_frame = (
            write_due and
            self._writer is not None
        )
        if self._viewer is None and not write_frame:
            return  # between video frames and no viewer: nothing to draw this tick
        self._scene.update_render()
        if self._viewer is not None:
            self._viewer.render()
        if write_frame:
            self._next_frame_sim_time += self._frame_interval
            self._camera.take_picture()
            rgba = np.asarray(self._camera.get_picture("Color"))
            frame = (np.clip(rgba[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)[..., ::-1]
            self._writer.write(np.ascontiguousarray(frame))
            self._video_frame_count += 1

    @property
    def viewer_closed(self) -> bool:
        """True once the user has closed the viewer window."""
        return self._viewer is not None and self._viewer.closed

    def close(self) -> None:
        """Release the video writer and close the viewer window, if any."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            if self._video_frame_count > 0:
                print(f"[sapien] saved video -> {self._video_path}")
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


# ---- real-hardware backend ---------------------------------------------------

class _DexJointComponent:
    """Forwards ``set_joint_pos`` to a dexcontrol component (send-now, wait_time=0)."""

    def __init__(self, component) -> None:
        self._component = component

    def set_joint_pos(self, joint_pos: Sequence[float]) -> None:
        """Send joint positions to the dexcontrol component immediately."""
        self._component.set_joint_pos(np.asarray(joint_pos, dtype=float), wait_time=0.0)


class _DexChassis:
    """Forwards coordinated velocity and stop commands to the real chassis."""

    def __init__(self, chassis) -> None:
        self._chassis = chassis

    def set_velocity(
        self, vx: float, vy: float, wz: float,
        wait_time: float = 0.0, sequential_steering: bool = False,
    ) -> None:
        """Forward a coordinated base velocity command to the official dexcontrol API.

        ``sequential_steering=False`` so a single call doesn't stall ~1 s adjusting
        steering; ``wait_time=0`` so it issues immediately in the high-rate loop. These
        are the kwargs :class:`BaseVelocityController` passes (the official non-sequential
        send-now path).
        """
        self._chassis.set_velocity(
            vx, vy, wz, wait_time=wait_time, sequential_steering=sequential_steering
        )

    def stop(self) -> None:
        """Stop the base."""
        self._chassis.stop()


class DexcontrolRobotSink:
    """Drives the real Vega via ``dexcontrol.Robot`` with the same high-level calls.

    Requires a live robot reachable over Zenoh (constructing ``dexcontrol.Robot``
    initializes/validates components against the hardware). Used by ``--real``.
    """

    def __init__(self) -> None:
        from dexcontrol.core.config import get_robot_config  # noqa: PLC0415 - optional dep
        from dexcontrol.robot import Robot  # noqa: PLC0415 - optional dep

        self._robot = Robot(configs=get_robot_config())
        self.left_arm = _DexJointComponent(self._robot.left_arm)
        self.right_arm = _DexJointComponent(self._robot.right_arm)
        self.torso = _DexJointComponent(self._robot.torso)
        self.head = _DexJointComponent(self._robot.head)
        self.chassis = _DexChassis(self._robot.chassis)

    def set_targets(self, left_pose, right_pose, head_pose=None) -> None:
        """No-op: the real robot has no target markers to draw."""

    @property
    def base_pose(self) -> np.ndarray:
        """Placeholder base pose (the real base pose is not read back here)."""
        return np.zeros(3)

    def step(self, dt: float) -> None:
        """No-op: the real robot integrates the velocity command itself."""

    def reset_base(self) -> None:
        """No-op: the real robot's physical base cannot be teleported to the origin."""

    def render(self, record_video: bool = True) -> None:
        """No-op: nothing to render for the hardware backend."""

    @property
    def viewer_closed(self) -> bool:
        """Always False: the hardware backend has no viewer."""
        return False

    def close(self) -> None:
        """Shut down the dexcontrol robot connection."""
        self._robot.shutdown()
