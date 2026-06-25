"""Physics-driven SAPIEN backend for the Vega whole-body demo (Option A).

Unlike :class:`omniteleop.follower.wbc_sapien_sim.SapienSimRobot` -- which is *kinematic*
(it teleports a fixed root by integrating the base twist, wheels inert) -- this backend
runs **SAPIEN PhysX** with a free-floating base and **actuates the wheel joints**: the two
steered+driven wheels are commanded via swerve-drive IK (steering position drives + drive
velocity drives) and the base pose *emerges* from wheel-ground contact, exactly as on the
real robot. Arms/torso/head are stiff position drives tracking the solver's joint targets.

Because the base is physical, it does **not** exactly reach the solver's commanded pose
(wheel slip, finite motor torque, controller lag), so EE tracking is *approximate* -- by
design. The driving script closes the loop with a PD **base controller** that makes the
wheel-driven base track the solver's reference trajectory (see ``base_tracking_twist``
and ``--physics`` in ``scripts/wbc_waypoints_record.py``).

Interface mirrors ``SapienSimRobot`` (``left_arm/right_arm/torso/head`` with
``set_joint_pos``; ``chassis`` with ``set_velocity``/``stop``;
``step``/``base_pose``/``set_targets``/``render``/``viewer_closed``/``reset_base``/``close``)
so the WBC dispatch code is unchanged.

Design choices (informed by a Codex design review):
- **Exact substepping**: the physics timestep is ``control_dt / n_substeps`` so each
  ``step(dt)`` advances physics by *exactly* ``dt`` (no ``round()`` time drift).
- **Force-limited motor model**: wheels use velocity drives capped at the URDF effort
  (drive 16 N*m, steer 6 N*m), so traction can slip rather than teleport.
- **Drive gating while steering**: drive speed is scaled by ``cos(steer_error)`` so a
  mis-pointed wheel does not shove the base sideways during a steer slew.
- **Gravity compensation** (``compute_passive_force`` feed-forward on the body joints) is
  available but **off by default** -- it had no measurable effect here, as the stiff
  position drives already hold the arms; **bumped solver iterations**; **free-rolling
  caster** (only its swivel is damped, against shimmy).
- Self-collisions among non-wheel links are disabled (the solver's own collision
  avoidance keeps commanded poses feasible); only wheel-ground contact is simulated.

Calibrated against real-robot ``scripts/drive_box_record.py`` logs (replay/overlay tool:
``scripts/diagnostics/replay_drive_box_sim.py``). Three findings drove the wheel-model
tuning: (1) the swerve IK's 180 deg steer ambiguity must be broken *consistently* across
the two wheels (see ``wbc_swerve.STEER_TIE_TOL``) or they wind onto opposite branches and
fight on straight legs; (2) the firmware clamps steering at ``STEER_OP_LIMIT`` = 2.35 rad,
inside the URDF stop; (3) the drive velocity loop must track tightly (``DRIVE_DAMPING``)
or it droops ~12% below commanded wheel speed, shrinking both translation and turn rate.

NOTE: the scene/viewer/recording/marker helpers below are duplicated from
``SapienSimRobot`` rather than shared, to keep the working kinematic backend (used by
three teleop scripts) at zero risk. They can be DRY-ed into a shared mixin later.
"""

# The inner _PhysChassis writes into its parent SapienPhysicsRobot's state, so ruff's
# private-access (SLF001) checks are disabled here (matches wbc_sapien_sim.py).
# ruff: noqa: SLF001

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from omniteleop.follower import wbc_swerve
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

__all__ = ["SapienPhysicsRobot"]

STEER_JOINTS = {"L": "L_wheel_j1", "R": "R_wheel_j1"}
DRIVE_JOINTS = {"L": "L_wheel_j2", "R": "R_wheel_j2"}
CASTER_STEER_JOINT = "B_wheel_j1"   # rear caster swivel (passive, lightly damped)
CASTER_ROLL_JOINT = "B_wheel_j2"    # rear caster wheel (passive, free-spinning)
CASTER_JOINTS = [CASTER_STEER_JOINT, CASTER_ROLL_JOINT]
BODY_JOINT_GROUPS = (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, TORSO_JOINTS, HEAD_JOINTS)

# Nominal posture (URDF convention) to settle into; matches the solver's nominal.
from omniteleop.follower.whole_body_ik import DEFAULT_NOMINAL_POSTURE  # noqa: E402

# URDF geometry: the wheel axle sits this far above the base-link origin (steer pivot
# 0.133 - axle drop 0.1005). Used to place the ground so the base rests at z~=0.
AXLE_ABOVE_BASE = 0.0325

# Real-robot steering operating limit (rad). The Vega-1 firmware clamps steering at
# 2.35 rad (Vega1ChassisConfig), well inside the URDF revolute stop (2.722); the box
# turn data shows the wheel saturating at exactly this value. The sim's swerve IK must
# use the same bound, else it steers ~10 deg further than the real wheel in tight turns.
STEER_OP_LIMIT = 2.35

# Drive tuning (force-limited PD); see module docstring.
# STEER_FORCE is an *effective* torque, not the URDF effort (6 N*m): steering a
# ground-loaded wheel in place must overcome contact-scrub friction, which SAPIEN's
# rigid contact overestimates, so 6 N*m stalls the steer at ~38 deg. Real hardware (with
# compliant tires) re-steers a 90 deg flip in ~0.25 s; 25 N*m reproduces that slew
# (~245 ms) and clears the scrub with margin. Too slow a steer makes one wheel align
# before the other on a re-steer, so only one drives and the base spins (drive_box back
# leg span +12 deg spurious yaw at 6 N*m).
STEER_STIFFNESS, STEER_DAMPING, STEER_FORCE = 2000.0, 200.0, 25.0
# Drive damping is the velocity-loop gain: a force-mode velocity drive settles where
# DRIVE_DAMPING*(target-actual) = load, so a low value droops ~12% below the commanded
# wheel speed under load -- which shrank BOTH the box translation (~5%) and the turn
# rate (~15%) vs the real chassis (drive_box data). 200 tracks the target to ~1%, like
# a real servo's tight velocity loop; force stays capped at the URDF effort (16 N*m).
DRIVE_DAMPING, DRIVE_FORCE, DRIVE_VEL_MAX = 200.0, 16.0, 12.0  # URDF: drive 16 N*m, 12 rad/s
BODY_STIFFNESS, BODY_DAMPING, BODY_FORCE = 3000.0, 300.0, 1000.0
# Rear caster: the wheel ROLL joint must spin free (a braked caster drags ~1.4 deg/s off
# the in-place turn); only the SWIVEL gets light damping to suppress shimmy.
CASTER_SWIVEL_DAMPING, CASTER_SWIVEL_FORCE = 0.5, 1.0


class _PhysJointGroup:
    """A joint group whose ``set_joint_pos`` writes position-drive targets immediately."""

    def __init__(self, joints_by_name: Dict[str, object], names: Sequence[str]) -> None:
        self._names = list(names)
        self._joints = [joints_by_name[n] for n in self._names]

    def set_joint_pos(self, joint_pos: Sequence[float]) -> None:
        """Command the group's joint positions (radians) as drive targets."""
        q = np.asarray(joint_pos, dtype=float).ravel()
        if q.shape[0] != len(self._names):
            raise ValueError(f"expected {len(self._names)} joint values, got {q.shape[0]}")
        for joint, val in zip(self._joints, q, strict=True):
            joint.set_drive_target(float(val))


class _PhysChassis:
    """Chassis storing the desired body twist; swerve IK is applied in ``robot.step``."""

    def __init__(self, robot: "SapienPhysicsRobot") -> None:
        self._robot = robot

    def set_velocity(self, vx: float, vy: float, wz: float, **_: float) -> None:
        """Store the commanded base velocity (m/s, m/s, rad/s); realized by the wheels."""
        self._robot._base_twist = np.array([vx, vy, wz], dtype=float)

    def stop(self) -> None:
        """Zero the commanded base velocity."""
        self._robot._base_twist = np.zeros(3)


class SapienPhysicsRobot:
    """Physics-backed Vega: the base is driven by actuating the swerve wheel joints."""

    def __init__(
        self,
        urdf_path: str = DEFAULT_URDF,
        with_viewer: bool = True,
        ground_z: Optional[float] = None,
        show_targets: bool = True,
        record_video: Optional[str] = None,
        video_fps: float = 15.0,
        video_size: tuple[int, int] = (960, 540),
        control_dt: float = 0.01,
        physics_hz: float = 400.0,
        wheel_friction: float = 1.2,
        wheel_radius: Optional[float] = None,
        settle_time: float = 0.6,
        gravity_comp: bool = False,
        steer_limit: float = STEER_OP_LIMIT,
    ) -> None:
        prepare_sapien_render_env()
        import sapien  # noqa: PLC0415 - heavy, env-specific dep; imported on use
        import sapien.physx as physx  # noqa: PLC0415

        self._sapien = sapien
        # Exact substepping: physics advances control_dt / n_substeps per substep, so
        # step(dt) covers exactly dt (no round()-based time drift; Codex BLOCKER).
        self._n_substeps = max(1, round(control_dt * physics_hz))
        self._physics_dt = control_dt / self._n_substeps
        self._control_dt = control_dt
        self._gravity_comp = gravity_comp
        self._steer_limit = steer_limit  # real firmware steering clamp (see STEER_OP_LIMIT)

        scene = sapien.Scene()
        scene.set_timestep(self._physics_dt)
        # Place the ground so the base link rests near z=0 -- the IK's planar (z=0) base
        # convention. Otherwise the wheel-supported base sits ~5 cm high and every EE
        # target is off by that resting height in z. Uses the nominal wheel radius for
        # placement; the exact radius is measured post-settle for the swerve map.
        r_est = wheel_radius if wheel_radius is not None else wbc_swerve.DEFAULT_WHEEL_RADIUS
        self._ground_z = ground_z if ground_z is not None else (AXLE_ABOVE_BASE - r_est)
        self._start_z = 0.05  # initial root height; settles down onto the wheels
        self._material = physx.PhysxMaterial(wheel_friction, wheel_friction, 0.0)
        scene.add_ground(self._ground_z, material=self._material)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
        self._scene = scene

        self._viewer = None
        if with_viewer:
            self._viewer = scene.create_viewer()
            self._viewer.set_camera_xyz(x=2.5, y=0.0, z=1.5)
            self._viewer.set_camera_rpy(r=0.0, p=-0.4, y=np.pi)

        loader = scene.create_urdf_loader()
        loader.fix_root_link = False  # MOBILE base: pose emerges from wheel contact
        loader.load_multiple_collisions_from_file = True
        self._robot = loader.load(urdf_path)

        # Collision: keep only wheel<->ground contact; disable non-wheel/self collisions
        # (the solver's own collision avoidance keeps commanded poses feasible).
        for link in self._robot.get_links():
            is_wheel = "wheel" in link.name.lower()
            for shape in link.get_collision_shapes():
                if is_wheel:
                    shape.set_physical_material(self._material)
                else:
                    shape.set_collision_groups([0, 0, 0, 0])

        self._sapien_idx = {
            j.name: i for i, j in enumerate(self._robot.get_active_joints())
        }
        self._joint_by_name = {j.name: j for j in self._robot.get_active_joints()}
        _links = {lk.name: lk for lk in self._robot.get_links()}
        self._ee_links = {"L": _links["L_ee"], "R": _links["R_ee"]}
        self._sim_time = 0.0
        self._base_twist = np.zeros(3)

        self._setup_drives()
        self._apply_initial_posture()
        self._settle(settle_time)

        # Rolling radius: measured from the settled wheel-axle height (auto-calibrated),
        # so the no-slip mapping omega = speed / R is correct for this URDF.
        if wheel_radius is None:
            links = {lk.name: lk for lk in self._robot.get_links()}
            wheel_radius = float(links["L_wheel_l2"].get_pose().p[2] - self._ground_z)
        self._wheel_radius = wheel_radius

        # Components mirroring dexcontrol.Robot.
        self.left_arm = _PhysJointGroup(self._joint_by_name, LEFT_ARM_JOINTS)
        self.right_arm = _PhysJointGroup(self._joint_by_name, RIGHT_ARM_JOINTS)
        self.torso = _PhysJointGroup(self._joint_by_name, TORSO_JOINTS)
        self.head = _PhysJointGroup(self._joint_by_name, HEAD_JOINTS)
        self.chassis = _PhysChassis(self)

        render_active = self._viewer is not None or record_video is not None
        if render_active:
            self._add_origin_frame()
        self._show_targets = show_targets and render_active
        if self._show_targets:
            self._l_target = self._make_target_frame()
            self._r_target = self._make_target_frame()
            # Optional head-target triad (smaller), drawn by callers that track a head
            # pose (e.g. wbc_vr_record). Mirrors SapienSimRobot's head marker.
            self._head_target = self._make_target_frame(length=0.18, radius=0.006)

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
        print(f"[physics] substeps={self._n_substeps} (physics {1/self._physics_dt:.0f} Hz)  "
              f"R_wheel={self._wheel_radius:.4f} m  friction={wheel_friction}  "
              f"gravity_comp={gravity_comp}")

    # -- drive setup / posture -------------------------------------------------

    def _setup_drives(self) -> None:
        steer = set(STEER_JOINTS.values())
        drive = set(DRIVE_JOINTS.values())
        body = {n for grp in BODY_JOINT_GROUPS for n in grp}
        for name, joint in self._joint_by_name.items():
            if name in drive:        # wheel drive: force-limited velocity drive
                joint.set_drive_property(0.0, DRIVE_DAMPING, DRIVE_FORCE, mode="force")
                joint.set_drive_velocity_target(0.0)
            elif name in steer:      # steering: force-limited position drive
                joint.set_drive_property(STEER_STIFFNESS, STEER_DAMPING, STEER_FORCE, mode="force")
                joint.set_drive_target(0.0)
            elif name == CASTER_ROLL_JOINT:  # passive caster wheel: free-spinning
                joint.set_drive_property(0.0, 0.0, 0.0, mode="force")
                joint.set_drive_velocity_target(0.0)
            elif name == CASTER_STEER_JOINT:  # caster swivel: light damping (resist shimmy)
                joint.set_drive_property(
                    0.0, CASTER_SWIVEL_DAMPING, CASTER_SWIVEL_FORCE, mode="force"
                )
                joint.set_drive_velocity_target(0.0)
            elif name in body:       # arm/torso/head: stiff position tracking
                joint.set_drive_property(BODY_STIFFNESS, BODY_DAMPING, BODY_FORCE, mode="force")
                joint.set_drive_target(0.0)
        # Stabilize the floating-base solve (Codex: bump iterations).
        try:
            self._robot.set_solver_iterations(20, 4)
        except Exception:  # API name varies across SAPIEN builds; non-fatal
            pass

    def _apply_initial_posture(self) -> None:
        q = np.zeros(self._robot.dof)
        for name, val in DEFAULT_NOMINAL_POSTURE.items():
            if name in self._sapien_idx:
                q[self._sapien_idx[name]] = val
                self._joint_by_name[name].set_drive_target(float(val))
        self._robot.set_qpos(q)
        # Start slightly above the resting height and let it settle onto the wheels.
        self._robot.set_root_pose(self._sapien.Pose(p=[0.0, 0.0, self._start_z]))

    def _settle(self, settle_time: float) -> None:
        n = max(1, round(settle_time / self._physics_dt))
        for _ in range(n):
            self._apply_gravity_comp()
            self._scene.step()

    # -- control step ----------------------------------------------------------

    def _apply_gravity_comp(self) -> None:
        if not self._gravity_comp:
            return
        try:
            qf = self._robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
        except TypeError:  # older signature
            qf = self._robot.compute_passive_force()
        qf = np.asarray(qf, dtype=float)
        # Only feed-forward the actuated body joints; leave wheels/caster to their drives.
        for name in (set(STEER_JOINTS.values()) | set(DRIVE_JOINTS.values())
                     | set(CASTER_JOINTS)):
            if name in self._sapien_idx:
                qf[self._sapien_idx[name]] = 0.0
        self._robot.set_qf(qf)

    def measured_steer(self) -> Dict[str, float]:
        """Measured steering joint angles (rad) keyed ``"L"``/``"R"``."""
        qpos = self._robot.get_qpos()
        return {k: float(qpos[self._sapien_idx[j]]) for k, j in STEER_JOINTS.items()}

    def _command_wheels(self) -> None:
        """Run swerve IK on the stored twist and write steer/drive targets."""
        steer_now = self.measured_steer()
        cmds = wbc_swerve.swerve_command(
            tuple(float(v) for v in self._base_twist),
            steer_now, wheel_radius=self._wheel_radius, steer_limit=self._steer_limit,
        )
        # Compute each wheel's steer target + gated drive rate, then scale BOTH drive
        # rates by one factor if either exceeds the envelope -- so the realized motion
        # keeps its direction under drive saturation (per-wheel clipping would bend it).
        omega: Dict[str, float] = {}
        for key in ("L", "R"):
            cmd = cmds[key]
            self._joint_by_name[STEER_JOINTS[key]].set_drive_target(cmd.steer)
            # Gate the drive by steer alignment: a mis-pointed wheel must not shove the
            # base sideways while the steer is still slewing.
            gate = max(0.0, float(np.cos(cmd.steer - steer_now[key])))
            omega[key] = cmd.drive_omega * gate
        peak = max(abs(omega["L"]), abs(omega["R"]))
        scale = DRIVE_VEL_MAX / peak if peak > DRIVE_VEL_MAX else 1.0
        for key in ("L", "R"):
            self._joint_by_name[DRIVE_JOINTS[key]].set_drive_velocity_target(omega[key] * scale)

    def step(self, dt: float) -> None:
        """Advance physics by ``dt`` (exactly), driving the wheels from the stored twist.

        ``dt`` must equal the ``control_dt`` the backend was built with: the substep count
        and physics timestep are fixed at construction so each call advances physics by
        exactly ``control_dt``. A differing ``dt`` would desync physics time from the
        ``_sim_time`` video clock, so it is rejected.
        """
        if abs(dt - self._control_dt) > 1e-9:
            raise ValueError(
                f"step(dt={dt}) must match control_dt={self._control_dt} (fixed "
                "substepping); construct SapienPhysicsRobot with this control rate"
            )
        self._command_wheels()
        for _ in range(self._n_substeps):
            self._apply_gravity_comp()
            self._scene.step()
        self._sim_time += dt

    # -- state readout ---------------------------------------------------------

    @property
    def base_pose(self) -> np.ndarray:
        """Measured mobile-base pose ``(x, y, yaw)`` from the physical root link."""
        p = self._robot.get_root_pose()
        w, x, y, z = p.q
        yaw = float(np.arctan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z)))
        return np.array([float(p.p[0]), float(p.p[1]), yaw])

    def ee_positions(self) -> Dict[str, np.ndarray]:
        """World positions of the ``L_ee``/``R_ee`` links -- the *physical* end-effectors.

        These ride the wheel-driven base, so comparing them to the targets gives the true
        (physics) EE tracking error, unlike the solver's ideal-base ``WBCResult`` error.
        """
        return {
            k: np.asarray(link.get_pose().p, dtype=float)
            for k, link in self._ee_links.items()
        }

    def reset_base(self) -> None:
        """Teleport the base to the origin (zero velocity) and re-settle on the wheels."""
        self._robot.set_root_pose(self._sapien.Pose(p=[0.0, 0.0, self._start_z]))
        self._robot.set_root_linear_velocity(np.zeros(3))
        self._robot.set_root_angular_velocity(np.zeros(3))
        self._base_twist = np.zeros(3)
        self._command_wheels()  # zero the wheel drive targets before settling
        self._settle(0.3)

    # -- markers / rendering (duplicated from SapienSimRobot; see module note) --

    def _setup_recording(self, config: VideoRecordingConfig) -> None:
        import cv2  # noqa: PLC0415 - optional dep, only needed when recording

        path = config.path
        width, height = config.size
        self._camera = self._scene.add_camera("recorder", width, height, 1.0, 0.05, 100.0)
        self._camera.set_local_pose(self._look_at(eye=[3.0, 0.0, 1.6], target=[0.3, 0.0, 0.9]))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, config.fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open a video writer for {path!r}")
        self._video_path = path
        print(f"[physics] recording video -> {path} ({width}x{height} @ {config.fps:g} fps)")

    def _look_at(self, eye, target, up=(0.0, 0.0, 1.0)):
        from scipy.spatial.transform import Rotation  # noqa: PLC0415

        eye = np.asarray(eye, dtype=float)
        fwd = np.asarray(target, dtype=float) - eye
        fwd /= np.linalg.norm(fwd)
        left = np.cross(np.asarray(up, dtype=float), fwd)
        left /= np.linalg.norm(left)
        up2 = np.cross(fwd, left)
        rot = np.column_stack([fwd, left, up2])
        qx, qy, qz, qw = Rotation.from_matrix(rot).as_quat()
        return self._sapien.Pose(p=eye, q=[qw, qx, qy, qz])

    def _add_triad_visuals(self, builder, length: float, radius: float) -> None:
        sapien = self._sapien
        half = length / 2.0
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
        builder = self._scene.create_actor_builder()
        self._add_triad_visuals(builder, length, radius)
        return builder.build_kinematic()

    def _add_origin_frame(self, length: float = 0.3, radius: float = 0.01) -> None:
        builder = self._scene.create_actor_builder()
        self._add_triad_visuals(builder, length, radius)
        self._origin_frame = builder.build_kinematic()
        self._origin_frame.set_pose(self._sapien.Pose())

    def _set_marker_pose(self, marker, pose) -> None:
        from scipy.spatial.transform import Rotation  # noqa: PLC0415

        mat = np.asarray(pose.homogeneous if hasattr(pose, "homogeneous") else pose, dtype=float)
        if mat.shape != (4, 4):
            raise ValueError(f"target pose has shape {mat.shape}, expected (4, 4)")
        qx, qy, qz, qw = Rotation.from_matrix(mat[:3, :3]).as_quat()
        marker.set_pose(self._sapien.Pose(p=list(mat[:3, 3]), q=[qw, qx, qy, qz]))

    def set_targets(self, left_pose, right_pose, head_pose=None) -> None:
        """Place target coordinate frames (world-frame 4x4 poses or ``pin.SE3``).

        ``head_pose`` is optional (WBC VR uses it to show desired gaze orientation);
        when ``None`` the head marker is left where it was.
        """
        if self._show_targets:
            self._set_marker_pose(self._l_target, left_pose)
            self._set_marker_pose(self._r_target, right_pose)
            if head_pose is not None:
                self._set_marker_pose(self._head_target, head_pose)

    def render(self, record_video: bool = True) -> None:
        """Render to the viewer and/or write a decimated real-time video frame."""
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
            return
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
                print(f"[physics] saved video -> {self._video_path}")
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
