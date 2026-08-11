"""VegaOGEnv — OmniGibson env wrapper that drives the Vega with the WBC service.

Owns the OmniGibson Environment (vega + scene + task objects), the WBC bridge client,
the WBC<->OG frame alignment, and the per-tick control loop. Task-agnostic: a SimTask
supplies scene objects + a scripted expert; this class runs the WBC control loop and
(P2) renders observations.

Run with behavior python, LD stripped of CoppeliaSim, OMNIGIBSON_HEADLESS=1.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
import numpy as np
import torch as th
from scipy.spatial.transform import Rotation as R
import omnigibson as og

from wbc_client import WBCClient
import base_ctrl
import zed_sim

TORSO = ["torso_j1", "torso_j2", "torso_j3"]
ARM_L = [f"L_arm_j{i}" for i in range(1, 8)]
ARM_R = [f"R_arm_j{i}" for i in range(1, 8)]
HEAD = ["head_j1", "head_j2", "head_j3"]


def pose_mat(pos, quat_xyzw):
    T = np.eye(4); T[:3, :3] = R.from_quat(quat_xyzw).as_matrix(); T[:3, 3] = pos
    return T


def mat_to_pos_quat(T):
    return T[:3, 3].copy(), R.from_matrix(T[:3, :3]).as_quat()


def _jc(kp):
    return {"name": "JointController", "motor_type": "position", "pos_kp": kp,
            "command_input_limits": None, "command_output_limits": None,
            "use_impedances": False, "use_delta_commands": False}


class VegaOGEnv:
    def __init__(self, task=None, action_hz=100, physics_hz=200, render_hz=100, pos_kp=1500,
                 lock_base=True, wbc_port=5610, obs_hw=None, wbc_overrides=None,
                 scene_model="Rs_int", robot_pos=(-0.5, 0.4, 0.03), robot_yaw=0.0,
                 grasping_mode="physical", grasping_direction="upper", mobile=False):
        overrides = {} if wbc_overrides is None else dict(wbc_overrides)
        # The simulator and the real follower share one controller configuration.  Tasks may plan
        # different paths, but they may not silently replace solver DOFs or controller gains.
        if "base_dofs" in overrides and overrides["base_dofs"] != base_ctrl.WBIK.base_dofs:
            raise ValueError(
                "sim base_dofs must match follower/wbik.yaml: "
                f"{base_ctrl.WBIK.base_dofs!r}, got {overrides['base_dofs']!r}"
            )
        self.base_dofs = base_ctrl.WBIK.base_dofs
        self.base_yaw_hold_in_xy = bool(base_ctrl.VR_TELEOP.base_yaw_hold_in_xy)
        wbc_overrides = overrides or None
        self.base_x_max = None   # mobile: forward-park clamp (set by run_episode from the task)
        # mobile: (x, y, radius) circles the chassis must not drive INTO. The general form of
        # base_x_max, needed once the robot approaches stations from arbitrary headings: the head
        # command parks the base at the reset camera-to-hands distance from the work point, which
        # for a deep counter/fridge/bookcase is INSIDE the furniture. Only the inward radial
        # component is removed, so the base can still slide along and back away.
        self.base_keepouts = []
        self.mobile = mobile
        # EVERY control constant below is READ from the follower config the hardware uses --
        # `base_ctrl.VR_TELEOP` is wbik.yaml's `vr_teleop:` block and `base_ctrl.WBIK` its solver
        # block. None of them is retyped here, because retyping them is precisely how the sim
        # drifted: an audit against `scripts/wbc_vr_robot.py` found this env running a dispatch
        # deadband of 0.02 where the yaml says 0.0, for no reason but a stale local copy. The
        # angular = 2x linear convention is the follower's own (`wbc_vr_robot._drive_base`).
        vr, wb = base_ctrl.VR_TELEOP, base_ctrl.WBIK
        self.base_kp_xy = float(vr.base_kp_xy)
        self.base_kp_yaw = float(vr.base_kp_yaw)
        self.base_max_lin = float(vr.base_max_speed)
        self.base_max_ang = 2.0 * self.base_max_lin
        self.base_deadband_lin = float(vr.base_deadband)
        self.base_deadband_ang = 2.0 * self.base_deadband_lin
        self.base_max_lin_accel = float(vr.base_accel)
        self.base_max_ang_accel = 2.0 * self.base_max_lin_accel
        self._prev_base_twist = np.zeros(3)
        self.base_post_linear_deadband = float(vr.base_post_linear_deadband)
        self.base_post_angular_deadband = float(vr.base_post_angular_deadband)
        self.enable_base_single_axis = bool(wb.enable_base_single_axis)
        self.base_single_axis_deadband = float(wb.base_single_axis_deadband)
        self.base_dispatch_single_axis_deadband = float(wb.base_dispatch_single_axis_deadband)
        self.base_single_axis_hysteresis_ratio = float(wb.base_single_axis_hysteresis_ratio)
        self.base_xy_max_vel = float(wb.base_xy_max_vel)
        self.base_yaw_max_vel = float(wb.base_yaw_max_vel)
        # Quiet-tick chassis dispatch, and the per-tick joint-step clamp. Both are hardware
        # bring-up knobs the yaml keeps script-local (see its `vr_teleop:` preamble), so the
        # follower's own defaults are mirrored here: `wbc_vr_robot.DEFAULT_BASE_QUIET_HOLD_S`
        # and `DEFAULT_MAX_JOINT_STEP` / `_JOINT_STEP_ABORT_TICKS`.
        self.base_quiet_hold_s = -1.0
        self.max_joint_step = 0.05
        self.joint_step_abort_ticks = 25
        self._base_quiet_elapsed = 0.0
        self._prev_joint_cmd = {}
        self._overstep_ticks = 0
        self._base_action = "drive"
        # Command path: the leader's rate, and the head filters that sit between it and the solver.
        self.cmd_rate = float(vr.cmd_rate)
        self.head_lpf_tau = float(vr.head_lpf_tau)
        self.head_planar_pos_deadband = float(vr.head_planar_pos_deadband)
        self.head_planar_yaw_deadband = float(vr.head_planar_yaw_deadband)
        self._tick = 0
        self._interp = self._interp_n = None
        self._head_lpf = self._head_deadband = None
        # The follower's loop rate is a shared constant, not a sim knob: `ik_rate` is what sizes
        # the interpolation segments, the slew limiter's dt and the head low-pass alpha, so a sim
        # running at a different rate is running a differently-tuned controller.
        if abs(action_hz - float(vr.ik_rate)) > 1e-9:
            raise ValueError(
                f"action_hz={action_hz} must equal the follower's vr_teleop.ik_rate="
                f"{vr.ik_rate}; the control loop is shared, so its rate is too")
        if self.cmd_rate <= 0 or abs(action_hz / self.cmd_rate - round(action_hz / self.cmd_rate)) > 1e-9:
            raise ValueError(
                f"vr_teleop.ik_rate={action_hz} must be an integer multiple of cmd_rate="
                f"{self.cmd_rate}")
        self._cmd_period_ticks = int(round(action_hz / self.cmd_rate))
        self._base_shaped = np.zeros(3)    # slew anchor: the UNPROJECTED shaped twist
        self._base_axis = None             # active single axis, threaded across ticks
        self._base_held = False   # mobile: park + hold the base after navigation (P-hold)
        self.base_hold_target = np.zeros(3)   # (x,y,yaw) the base is held at (updated by drive_base)
        self.robot_pos = tuple(robot_pos)
        # OmniGibson rate hierarchy: render_dt multiple of physics_dt, action_dt multiple of
        # render_dt  =>  render_hz % action_hz == 0  AND  physics_hz % render_hz == 0.
        assert render_hz % action_hz == 0 and physics_hz % render_hz == 0, \
            f"need render_hz({render_hz}) % action_hz({action_hz})==0 and physics_hz({physics_hz}) % render_hz==0"
        self.task = task
        self.scene_model = scene_model      # tasks plan their base routes on this scene's trav map
        self.action_hz = action_hz
        self.dt = 1.0 / action_hz
        self.lock_base = lock_base
        self.grasping_mode = grasping_mode
        # The 2F-85 closes toward its upper revolute limit, so the controller is inverted to keep
        # the task-level convention stable: command -1 => CLOSE, +1 => OPEN.
        self.grip_close = -1.0
        self.grip_open = 1.0
        # A VisionSensor's render product is sized at LOAD time -- headless, its image_width
        # setter is a no-op -- so the head render size has to be fixed here. Callers using the
        # ZED-parity path pass `zed_sim.head_render_hw(opts)` (the raw SVGA canvas, which
        # obs_pipeline then crops/resizes like the real publisher).
        obs_hw = zed_sim.DEFAULT_RENDER_HW if obs_hw is None else tuple(obs_hw)
        robot_cfg = {
            # depth_linear == Isaac distance_to_image_plane (z-depth), which is what the pinhole
            # unprojection in wbc_pointcloud wants; OmniGibson's "depth" is distance_to_camera
            # (euclidean range) and would bow the recorded cloud outward at the image edges.
            "model": "vega_robotiq", "obs_modalities": ["rgb", "depth_linear", "proprio"],
            "grasping_mode": grasping_mode, "grasping_direction": grasping_direction,
            # Holonomic-base robots cannot float; OmniGibson forces this on anyway and warns if
            # left at the default False, so state it explicitly.
            "fixed_base": True,
            # MUST be False. Every controller group below asks for `command_input_limits: None`,
            # i.e. "my commands are already in physical units, pass them through". OmniGibson's
            # default action_normalize=True DISCARDS that and rewrites the input limits to (-1, 1)
            # for every group (robots/robot.py `if self._action_normalize: cfg[...] = "default"`),
            # which silently mangled every command we sent. Measured on the dish2rack episode:
            #   base   -> output limits default to the joint velocity caps, so the twist was SCALED
            #             by 1.5 (x, y) and by pi (yaw). A 0.30 m/s command drove 0.450 m/s and a
            #             0.35 rad/s command drove 1.099 rad/s -- the base ran 3x its commanded
            #             turn rate, which is what the yaw thrash and the chassis tipping were.
            #   trunk  -> WBC asked for up to 2.06 rad and was CLIPPED at 1.0 rad on 100% of ticks.
            #   arms   -> clipped on 82% (left) / 86% (right) of ticks, targets up to 1.42 rad.
            # The grippers pass an explicit [-1, 1] input range and are unaffected either way.
            "action_normalize": False,
            "position": list(robot_pos),
            "orientation": R.from_euler("z", robot_yaw).as_quat().tolist(),
            "sensor_config": {"VisionSensor": {"sensor_kwargs": {"image_height": obs_hw[0], "image_width": obs_hw[1]}}},
            "controller_config": {
                "base": {"name": "HolonomicBaseJointController", "motor_type": "velocity",
                         "vel_kp": 150, "command_input_limits": None, "use_impedances": False},
                "trunk": _jc(pos_kp), "camera": _jc(pos_kp), "arm_left": _jc(pos_kp), "arm_right": _jc(pos_kp),
                "gripper_left": {"name": "MultiFingerGripperController", "mode": "smooth",
                                 "command_input_limits": [-1.0, 1.0], "command_output_limits": "default"},
                "gripper_right": {"name": "MultiFingerGripperController", "mode": "smooth",
                                  "command_input_limits": [-1.0, 1.0], "command_output_limits": "default"},
            },
        }
        cfg = {
            "env": {"action_frequency": action_hz, "physics_frequency": physics_hz,
                    "rendering_frequency": render_hz},
            "scene": ({"type": "InteractiveTraversableScene", "scene_model": scene_model}
                      if scene_model else {"type": "Scene", "use_floor_plane": True}),
            "robots": [robot_cfg],
            "objects": task.object_configs() if task is not None else [],
            "task": {"type": "DummyTask"},
        }
        self.env = og.Environment(configs=cfg)
        self.robot = self.env.robots[0]
        self.names = self.robot.dof_names_ordered
        self.name2idx = {n: i for i, n in enumerate(self.names)}
        self.cai = self.robot.controller_action_idx
        if task is not None:
            task.bind(self)
        self.wbc = WBCClient(port=wbc_port, lock_base=lock_base, overrides=wbc_overrides)
        self.T_align = None
        self.T_align_inv = None

    # ---- kinematics / frames ----
    def link_pose(self, link):
        p, q = self.robot.links[link].get_position_orientation()
        return pose_mat(p.detach().cpu().numpy(), q.detach().cpu().numpy())

    def obj_pos(self, obj):
        return obj.get_position_orientation()[0].detach().cpu().numpy()

    def is_grasping(self, arm="left"):
        """Object magnetized to @arm under assisted/sticky grasping, else None."""
        return self.robot._ag_obj_in_hand.get(arm)

    def set_assisted_grasp_handling(self, enabled):
        """Gate automatic reattachment while a deliberately released gripper opens."""
        if self.grasping_mode != "physical":
            self.robot._disable_grasp_handling = not bool(enabled)

    def finger_qpos(self, arm="left"):
        """The two driven gripper-joint positions for @arm."""
        q = self.robot.get_joint_positions().detach().cpu().numpy()
        return [round(float(q[self.name2idx[name]]), 4)
                for name in self.robot.finger_joint_names[arm]]

    def finger_position_normalized(self, arm="left"):
        """Mean achieved Robotiq position in hardware FC03 semantics (0=open, 1=closed)."""
        q = self.robot.get_joint_positions().detach().cpu().numpy()
        lo = self.robot.joint_lower_limits.detach().cpu().numpy()
        hi = self.robot.joint_upper_limits.detach().cpu().numpy()
        values = []
        for name in self.robot.finger_joint_names[arm]:
            i = self.name2idx[name]
            values.append((float(q[i]) - float(lo[i])) / max(float(hi[i] - lo[i]), 1e-9))
        return float(np.clip(np.mean(values), 0.0, 1.0))

    def base_body_twist(self):
        """Measured base velocity in the body frame, matching hardware wheel odometry."""
        qd = self.robot.get_joint_velocities().detach().cpu().numpy()
        world = np.array([
            qd[self.name2idx["base_footprint_x_joint"]],
            qd[self.name2idx["base_footprint_y_joint"]],
        ], dtype=float)
        yaw = self.base_xyyaw()[2]
        c, s = np.cos(yaw), np.sin(yaw)
        body = np.array([[c, s], [-s, c]]) @ world
        return np.array([
            body[0], body[1], qd[self.name2idx["base_footprint_rz_joint"]]
        ], dtype=float)

    def finger_center_world(self, arm="left"):
        """World midpoint between @arm's two fingertip links (where the grasp closes)."""
        a, b = self.robot.finger_link_names[arm]
        return (self.link_pose(a)[:3, 3] + self.link_pose(b)[:3, 3]) / 2

    def finger_grasp_point(self, arm="left"):
        """World center of the configured assisted-grasp rays for @arm.

        Multi-link grippers such as the 2F-85 do not have a useful center at either their link
        origins or whole-link AABB centers. Prefer the explicit pad contact rays from the robot
        definition; retain the AABB midpoint as a fallback for simpler parallel jaws.
        """
        starts = self.robot.assisted_grasp_start_points
        ends = self.robot.assisted_grasp_end_points
        if starts is not None and ends is not None and starts.get(arm) and ends.get(arm):
            world_points = []
            for point in (*starts[arm], *ends[arm]):
                local = point.position.detach().cpu().numpy()
                T = self.link_pose(point.link_name)
                world_points.append(T[:3, :3] @ local + T[:3, 3])
            return np.mean(world_points, axis=0)

        a, b = self.robot.finger_link_names[arm]

        def ctr(link):
            amin, amax = self.robot.links[link].aabb
            amin = amin.detach().cpu().numpy() if hasattr(amin, "detach") else np.asarray(amin)
            amax = amax.detach().cpu().numpy() if hasattr(amax, "detach") else np.asarray(amax)
            return (amin + amax) / 2
        return (ctr(a) + ctr(b)) / 2

    def joint_margins(self, names):
        """(name, q_deg, lo_deg, hi_deg, frac) for each joint, frac=0 at a limit, 1 mid-range."""
        q = self.robot.get_joint_positions().detach().cpu().numpy()
        lo, hi = self.robot.joint_lower_limits, self.robot.joint_upper_limits
        lo = lo.detach().cpu().numpy() if hasattr(lo, "detach") else np.asarray(lo)
        hi = hi.detach().cpu().numpy() if hasattr(hi, "detach") else np.asarray(hi)
        out = []
        for n in names:
            i = self.name2idx[n]
            span = max(float(hi[i] - lo[i]), 1e-9)
            frac = min(float(q[i] - lo[i]), float(hi[i] - q[i])) / span
            out.append((n, np.degrees(q[i]), np.degrees(lo[i]), np.degrees(hi[i]), frac))
        return out

    def at_limits(self, names, tol=0.03):
        """Joints within `tol` of their range ends -- the ones squeezing the IK."""
        return [(n, round(qd, 1), round(lod, 1), round(hid, 1))
                for n, qd, lod, hid, f in self.joint_margins(names) if f < tol]

    def base_xyyaw(self):
        q = self.robot.get_joint_positions().detach().cpu().numpy()
        return (float(q[self.name2idx["base_footprint_x_joint"]]),
                float(q[self.name2idx["base_footprint_y_joint"]]),
                float(q[self.name2idx["base_footprint_rz_joint"]]))

    def measured_pin_q(self):
        """Measured joints in pinocchio order, clamped into the model's own limit box.

        A joint resting ON a hard stop settles a hair past it -- L_arm_j4 reads 0.244109 against
        a 0.244000 upper limit, 1.1e-4 rad -- and pink's `Configuration.check_limits` then logs
        `Value ... is out of limits` on every tick of every episode that holds a saturated joint.
        The excursion is position-controller slop, not a real posture: OmniGibson enforces the
        SAME URDF box we are clamping to, so nothing physical is being hidden.
        """
        q = self.robot.get_joint_positions().detach().cpu().numpy()
        lo, hi = self.robot.joint_lower_limits, self.robot.joint_upper_limits
        lo = lo.detach().cpu().numpy() if hasattr(lo, "detach") else np.asarray(lo)
        hi = hi.detach().cpu().numpy() if hasattr(hi, "detach") else np.asarray(hi)
        qby = {n: float(np.clip(q[i], lo[i], hi[i]))
               for n, i in ((n, self.name2idx[n]) for n in (TORSO + ARM_L + ARM_R + HEAD))}
        return WBCClient.build_pin_q(qby, self.base_xyyaw())

    def og_to_wbc(self, T_og):
        return self.T_align_inv @ np.asarray(T_og)

    # ---- lifecycle ----
    def _hold_action(self):
        q = self.robot.get_joint_positions().detach().cpu().numpy()
        a = th.zeros(self.robot.action_dim)
        a[self.cai["trunk"]] = th.tensor([q[self.name2idx[n]] for n in TORSO])
        a[self.cai["camera"]] = th.tensor([q[self.name2idx[n]] for n in HEAD])
        a[self.cai["arm_left"]] = th.tensor([q[self.name2idx[n]] for n in ARM_L])
        a[self.cai["arm_right"]] = th.tensor([q[self.name2idx[n]] for n in ARM_R])
        a[self.cai["gripper_left"]] = self.grip_open
        a[self.cai["gripper_right"]] = self.grip_open
        return a

    def reset(self, seed=None):
        self.env.reset()
        self.set_assisted_grasp_handling(True)
        # AFTER env.reset(), never before: OmniGibson's `scene.reset(hard=True)` forces the live
        # object set back to the scene's initial file, so anything removed at construction is
        # restored on the first reset. Decluttering there looked like it worked (the removals
        # logged cleanly) while the furniture was quietly back before the episode started -- the
        # robot then drove into the "removed" sofa and toppled mid-transit.
        if self.task is not None and hasattr(self.task, "declutter"):
            self.task.declutter(self)
        self.lock_chassis_tilt()
        self._prev_base_twist = np.zeros(3)
        self._base_shaped = np.zeros(3)
        self._base_axis = None
        self._base_held = False
        # Per-episode command-path state. The follower re-engages from scratch each run: fresh
        # interpolation segments, head filters re-anchored on the new engage pose, and a clamp
        # anchor that re-seeds from measured position on the first tick.
        self._tick = 0
        self._base_quiet_elapsed = 0.0
        self._base_action = "drive"
        self._interp = self._interp_n = None
        self._head_lpf = self._head_deadband = None
        self._prev_joint_cmd = {}
        self._overstep_ticks = 0
        self._quality = {
            "hold_ticks": 0,
            "keepout_interventions": 0,
            "max_target_jump_m": 0.0,
            "max_target_jump_detail": None,
            "max_arm_target_jump_m": 0.0,
            "max_target_jump_deg": 0.0,
            "max_chassis_tilt_deg": 0.0,
        }
        self._previous_effective_targets = None
        if self.task is not None:
            self.task.reset(self)
            self.base_keepouts = list(self.task.keepouts()) if hasattr(self.task, "keepouts") else []
            for _ in range(10):
                self.env.step(self._hold_action())   # settle objects + robot at nominal
        self.T_align = self.link_pose("base")
        self.T_align_inv = np.linalg.inv(self.T_align)
        self.base_hold_target = np.array(self.base_xyyaw())   # ~0 at spawn
        self.base_yaw_target = float(self.base_hold_target[2])
        self._setup_third_person()
        self.wbc.reset()
        if self.task is not None:
            self.task.expert_reset(self)

    # ---- 3rd-person camera (for videos) ----
    def _set_cam_lookat(self, eye, target, up=(0, 0, 1)):
        eye = np.asarray(eye, float); target = np.asarray(target, float)
        zc = eye - target; zc /= np.linalg.norm(zc)          # OG camera looks along -z
        xc = np.cross(np.asarray(up, float), zc); xc /= np.linalg.norm(xc)
        yc = np.cross(zc, xc)
        quat = R.from_matrix(np.stack([xc, yc, zc], axis=1)).as_quat()
        og.sim.viewer_camera.set_position_orientation(position=eye.tolist(), orientation=quat.tolist())

    def _setup_third_person(self):
        # A reused env must not ease from the previous episode's final camera pose.
        for name in ("_cam_scale", "_cam_eye", "_cam_aim"):
            if hasattr(self, name):
                delattr(self, name)
        self.follow_third_person()
        for _ in range(2):
            og.sim.render()

    def follow_third_person(self, back=0.25, side=0.0, up=2.25, ahead=0.40):
        """Re-aim the review camera from almost overhead, directly behind the robot.

        A fixed camera pointed at the work area is useless once the task spans metres: over a
        long-horizon episode the robot spends most of its time nowhere near the midpoint of the
        two objects, so it is simply out of frame. Called before every capture, this keeps the
        robot and whatever it is reaching for both in view for the whole take.

        Zero lateral offset is deliberate: at the zero-yaw task heading, image horizontal is
        exactly world -y. The review video can therefore expose whether the two EEF targets have
        the same lateral timing instead of mixing forward motion into that comparison.

        The eye is then pulled back IN toward the robot until it sits over open floor. Without
        that the shot is buried in a wall for much of the episode -- the robot works close to
        furniture and walls by definition, so a fixed offset spends its time outside the room.

        The aim point sits well ahead of and above the base so both grippers stay in frame. The
        hands work in front of the body at roughly chest height, and forward is IMAGE-UP in this
        near-overhead shot, so aiming at the base itself pushed them off the top edge exactly
        during the grasp and place phases. Tilting is the only lever available: `up` cannot grow,
        because the Rs_int ceiling slab starts at z = 2.40 and the eye already sits ~10 cm under
        it -- raise it and every frame is the underside of the ceiling.
        """
        T = self.link_pose("base")
        p = T[:3, 3]
        measured_yaw = float(np.arctan2(T[1, 0], T[0, 0]))
        # Use the commanded heading when the scripted task exposes it. In dish2rack that is
        # exactly world +x even if the undriven rz joint drifts a few degrees, so image horizontal
        # remains exactly world -y and the two EEF lateral coordinates are visually comparable.
        yaw = float(getattr(self.task, "_head_heading", measured_yaw))
        fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        # The stand-off direction is fixed in the WORLD, not behind the robot: a body-relative
        # offset swings the whole shot around the robot every time it turns, which on these tasks
        # is most of the episode.
        offset = np.array([-back, side, 0.0])
        want = 1.0
        for scale in (1.0, 0.8, 0.6, 0.45, 0.3, 0.0):
            if scale == 0.0 or self._cam_clear(p + scale * offset + np.array([0.0, 0.0, up])):
                want = scale
                break
        # Ease only the discrete wall-clearance scale. The eye and aim must follow the base
        # exactly: low-passing their world poses made the whole robot drift sideways by ~65 pixels
        # at episode start, then appear to teleport back when the first strafe ended.
        prev = getattr(self, "_cam_scale", want)
        self._cam_scale = prev + np.clip(want - prev, -0.02, 0.02)
        eye = p + self._cam_scale * offset + np.array([0.0, 0.0, up])
        aim = p + ahead * fwd + np.array([0.0, 0.0, 0.75])
        self._cam_eye = eye
        self._cam_aim = aim
        self._set_cam_lookat(self._cam_eye, self._cam_aim)

    def _cam_clear(self, eye):
        """True when (x, y) is open floor, i.e. the camera is inside the room, not in a wall."""
        if not hasattr(self, "_cam_map"):
            try:
                from tasks.nav_map import NavMap
                self._cam_map = NavMap(self.scene_model, robot_radius=0.12)
            except Exception:
                self._cam_map = None
        return True if self._cam_map is None else self._cam_map.free(float(eye[0]), float(eye[1]))

    def capture_third_person(self):
        obs = og.sim.viewer_camera.get_obs()[0]
        rgb = obs["rgb"] if isinstance(obs, dict) else obs
        rgb = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
        return rgb[..., :3].astype(np.uint8)

    def project_third_person(self, points):
        """World XYZ -> (u, v) pixels in the review camera, with an in-front-of-the-lens mask.

        Lets the review video draw the things the controller is actually chasing -- the commanded
        EEF and head poses -- on top of the robot that is chasing them, which is the only way to
        see tracking error in a picture. Must be called with the camera where the captured frame
        had it, i.e. between `follow_third_person` and `capture_third_person`.

        A USD camera looks down its own -z with +y up; the pinhole model wants +z forward and +y
        down, and the two differ by `diag(1, -1, -1)` (`obs_pipeline._OPTICAL_TO_USD`, which is its
        own inverse).
        """
        cam = og.sim.viewer_camera
        k = cam.intrinsic_matrix
        k = k.detach().cpu().numpy() if hasattr(k, "detach") else np.asarray(k)
        p, q = cam.get_position_orientation()
        world_t_cam = pose_mat(p.detach().cpu().numpy(), q.detach().cpu().numpy()) \
            @ np.diag([1.0, -1.0, -1.0, 1.0])
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        cam_pts = (np.linalg.inv(world_t_cam) @ homo.T).T[:, :3]
        z = cam_pts[:, 2]
        ok = z > 1e-3
        uv = np.full((len(pts), 2), np.nan)
        uv[ok, 0] = k[0, 0] * cam_pts[ok, 0] / z[ok] + k[0, 2]
        uv[ok, 1] = k[1, 1] * cam_pts[ok, 1] / z[ok] + k[1, 2]
        return uv, ok

    # ---- control ----
    def _shape_base_twist(self, twist):
        # lock_base: the velocity controller commands 0 but doesn't hold POSITION (arm reaction
        # slides the base) -> P-hold at base_hold_target (updated by drive_base during approach).
        # TODO(mobile): port follower/base_closed_loop.shape_twist for full WBC-driven base motion.
        if self.lock_base or self._base_held:
            cur = np.array(self.base_xyyaw())
            return -10.0 * (cur - self.base_hold_target)
        return np.asarray(twist, dtype=float)

    def hold_base_here(self):
        """Freeze the base at its current pose (mobile: park after navigation, then manipulate)."""
        self._base_held = True
        self.base_hold_target = np.array(self.base_xyyaw())

    def _mobile_base_cmd(self, resp):
        """WBC-driven base (mobile), following `wbc_vr_robot._drive_base` step for step.

        Feed-forward + proportional pose feedback against the MEASURED base pose, then the shared
        shaping/projection, then the quiet-tick dispatch decision. The sim's odometry equivalent
        is `base_xyyaw()` (the base_footprint joints), which shares the spawn-origin frame with
        the WBC's own `base_pose` -- that is what makes the SE(2) error well-posed.

        This env previously ran the feed-forward ALONE and fed `current_q` into the QP instead, so
        the loop was closed inside the solver rather than around it. That is a different
        architecture from the robot's, and the measurement offered in its defence (PD term ~1 % of
        the command) was a measurement of the sim's own plant: hardware odometry lags and drifts,
        which is the entire reason the outer loop exists. A sim whose base tracks better than the
        robot's for structural reasons is not the one to validate a controller on.
        """
        allow_yaw_hold = self.base_dofs == "xy" and self.base_yaw_hold_in_xy
        measured = np.array(self.base_xyyaw())
        cmd, _err = base_ctrl.pd_twist(
            np.asarray(resp["base_pose"], dtype=float),
            np.asarray(resp["base_twist"], dtype=float),
            measured,
            kp_xy=self.base_kp_xy, kp_yaw=self.base_kp_yaw,
            max_lin_speed=self.base_max_lin, max_ang_speed=self.base_max_ang)
        if allow_yaw_hold:
            # WBC yaw is hard-pinned in `xy` mode, so the PD's yaw term has no reference to track;
            # hold the episode's fixed heading instead, which is what the flag is for on hardware.
            cmd[2] = np.clip(
                self.base_kp_yaw * base_ctrl.wrap_pi(self.base_yaw_target - measured[2]),
                -self.base_max_ang,
                self.base_max_ang,
            )
        cmd = base_ctrl.mask_planar_twist_for_base_dofs(
            cmd, base_dofs=self.base_dofs, allow_yaw_hold=allow_yaw_hold)
        # Shape AND single-axis project, exactly as both real followers do via
        # base_closed_loop.shape_project_twist. Previously this stopped at shape_twist, so the
        # sim could drive vx, vy and wz at once -- the very thing wbik.yaml's single-axis block
        # exists to stop ("a drive straight leans sideways, an in-place turn wanders, and a
        # meant-to-be-still base creeps"). The slew limiter is fed the UNPROJECTED shaped
        # signal so a zeroed axis does not have to re-accelerate from standstill on every
        # axis change.
        shaped, self._base_shaped, self._base_axis = base_ctrl.shape_project_twist(
            cmd, self._base_shaped, self.dt,
            deadband_lin=self.base_deadband_lin, deadband_ang=self.base_deadband_ang,
            max_lin_speed=self.base_max_lin, max_ang_speed=self.base_max_ang,
            max_lin_accel=self.base_max_lin_accel, max_ang_accel=self.base_max_ang_accel,
            post_linear_deadband=self.base_post_linear_deadband,
            post_angular_deadband=self.base_post_angular_deadband,
            enable_single_axis=self.enable_base_single_axis,
            xy_max_vel=self.base_xy_max_vel, yaw_max_vel=self.base_yaw_max_vel,
            single_axis_deadband=self.base_single_axis_deadband,
            dispatch_single_axis_deadband=self.base_dispatch_single_axis_deadband,
            single_axis_hysteresis_ratio=self.base_single_axis_hysteresis_ratio,
            prev_axis=self._base_axis,
            base_dofs=self.base_dofs,
            allow_yaw_hold=allow_yaw_hold)
        # Forward-park clamp: once the base reaches base_x_max, stop it driving further toward the
        # table (the WBC would otherwise keep pulling it in to shorten the arm's reach -> collision).
        # Robot faces +x (yaw~0), so base-frame vx == world +x velocity.
        if self.base_x_max is not None and self.link_pose("base")[0, 3] >= self.base_x_max and shaped[0] > 0.0:
            shaped[0] = 0.0
        shaped = self._apply_keepouts(shaped)
        # This is the dispatched command in both real followers: the multi-axis signal is slewed,
        # then the final command is projected to one axis. Slewing the projected output a second
        # time blends the old and new axes during a switch, defeating that invariant.
        #
        # The quiet-tick decision runs here too, through the follower's own `base_quiet_dispatch`.
        # Its PURPOSE does not survive the port -- it exists so a swerve chassis holds its current
        # steering through a brief command dip instead of snapping the wheels back to 0 deg, and
        # OmniGibson's holonomic base has no steering state to hold, so "hold" and "recenter" both
        # dispatch the same zero twist. It is wired up anyway so the branch, the quiet timer and
        # the invariant it asserts are the shared ones, and so a future steering model inherits the
        # right behaviour instead of re-deriving it. The real cost is not portable and is worth
        # stating plainly: on hardware an axis switch costs a physical re-steer that the sim gets
        # for free.
        action, self._base_quiet_elapsed = base_ctrl.base_quiet_dispatch(
            shaped, self._base_quiet_elapsed, self.dt, self.base_quiet_hold_s)
        if action != "drive":
            shaped = np.zeros(3)
        self._base_action = action
        self._prev_base_twist = shaped
        return shaped

    def lock_chassis_tilt(self):
        """Dynamically servo the non-planar virtual joints instead of teleporting them per tick.

        The real Vega root is structurally planar. OmniGibson's imported holonomic root exposes
        free z/roll/pitch joints, so give those three joints ordinary PhysX position drives and a
        persistent target.  This lets reaction forces flow through the simulated chassis while
        avoiding the energy/state discontinuity caused by writing positions and velocities after
        every physics step. Yaw remains controlled by the normal base controller, as required by
        wbik.yaml's ``base_dofs: xy_yaw``.
        """
        from omnigibson.controllers.controller_base import ControlType
        import omnigibson.lazy as lazy

        names = [f"base_footprint_{axis}_joint" for axis in ("z", "rx", "ry")]
        q = self.robot.get_joint_positions().detach().cpu().numpy()
        targets = th.tensor([
            float(q[self.name2idx[names[0]]]), 0.0, 0.0
        ], dtype=th.float32)
        indices = [self.name2idx[name] for name in names]
        for name, drive_type in zip(names, ("linear", "angular", "angular")):
            joint = self.robot.joints[name]
            with og.sim.editing_usd():
                lazy.pxr.UsdPhysics.DriveAPI.Apply(joint.prim, drive_type)
            joint._driven = True
            joint.set_control_type(ControlType.POSITION, kp=50000.0, kd=5000.0)
        self.robot.set_joint_positions(targets, indices=indices, drive=True)

    def chassis_tilt(self):
        """(z, roll, pitch) of the chassis -- 0 when level. Nonzero means it is tipping."""
        q = self.robot.get_joint_positions().detach().cpu().numpy()
        return tuple(float(q[self.name2idx[f"base_footprint_{c}_joint"]]) for c in ("z", "rx", "ry"))

    def _apply_keepouts(self, twist):
        """Stop any active chassis axis that would drive farther into a violated keepout.

        The real controller's final command is one pure base-frame axis. Orthogonally projecting a
        forward command onto a circular boundary creates a diagonal vx+vy command and defeats that
        invariant. Test each base-frame axis in world coordinates instead: inward axes stop,
        while an already-commanded tangential or outward axis survives unchanged.
        """
        if not self.base_keepouts:
            return twist
        Tb = self.link_pose("base")
        p = Tb[:2, 3]
        yaw = float(np.arctan2(Tb[1, 0], Tb[0, 0]))
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s], [s, c]])
        out = np.asarray(twist, dtype=float).copy()
        for kx, ky, kr in self.base_keepouts:
            d = p - np.array([kx, ky], dtype=float)
            r = float(np.linalg.norm(d))
            if r >= kr:
                continue
            if r < 1e-9:
                out[:2] = 0.0
                continue
            n = d / r                                         # outward radial unit vector
            for axis in range(2):
                axis_world = Rz[:, axis] * out[axis]
                if float(axis_world @ n) < 0.0:
                    out[axis] = 0.0
        if not np.allclose(out, twist, atol=1e-12):
            quality = getattr(self, "_quality", None)
            if quality is not None:
                quality["keepout_interventions"] += 1
        return out

    def drive_base(self, base_vel, capture=False):
        """Navigate: command base velocity (base frame) + hold arms at nominal, step.
        Updates base_hold_target so the manipulation phase holds where we drove to."""
        a = self._hold_action()
        a[self.cai["base"]] = th.tensor(np.asarray(base_vel, dtype=float), dtype=th.float32)
        self.env.step(a)
        self.base_hold_target = np.array(self.base_xyyaw())
        return self.capture_third_person() if capture else None

    def wbc_tick(self, left_og, right_og, head_og=None, grip_l=-1.0, grip_r=-1.0):
        """One WBC control tick, in the order `scripts/wbc_vr_robot.py` runs it.

        interpolate the leader-rate command -> low-pass + deadband the head target -> solve ->
        hold on a bad solve -> clamp the per-joint step -> shape/dispatch the base.
        """
        left = self.og_to_wbc(left_og)
        right = self.og_to_wbc(right_og)
        head = self.og_to_wbc(head_og) if head_og is not None else None
        left, right, head = self._interpolate_command(left, right, head)
        if head is not None:
            if self._head_lpf is None:
                # Anchored on the first commanded head pose, as the follower anchors on the head
                # pose at engage. They coincide here: the expert's tick-0 head target IS the reset
                # pose (see MobilePickPlaceTask -- the waypoint starts at the fingertips, so the
                # head command starts at the spawn pose with no step input).
                self._head_lpf = base_ctrl.HeadTargetLowPassFilter(self.head_lpf_tau, head)
                self._head_deadband = base_ctrl.HeadTargetPlanarDeadbandFilter(
                    head,
                    position_deadband=self.head_planar_pos_deadband,
                    yaw_deadband=self.head_planar_yaw_deadband)
            head = self._head_lpf.filter(head, self.dt)
            head = self._head_deadband.filter(head)
        effective = {"left": left.copy(), "right": right.copy(),
                     "head": None if head is None else head.copy()}
        # current_q=None: the follower never re-seeds the IK from measured state (the string does
        # not appear in wbc_vr_robot.py at all) -- it integrates its own configuration and closes
        # the base loop OUTSIDE the solver, in `_mobile_base_cmd`. Feeding measurement in here as
        # well would be a second, tighter loop the robot does not have.
        resp = self.wbc.solve(left, right, head=head, current_q=None, dt=self.dt)
        # `hold` is the follower's safety gate: a failed solve or a solver-side hold freezes the
        # joints and stops the chassis rather than actuating a target the QP could not reach.
        hold = (not bool(resp.get("success", True))) or bool(resp.get("held", False))
        a = th.zeros(self.robot.action_dim)
        if hold:
            self._prev_joint_cmd = {}          # re-seed the clamp anchor from measured on resume
            base_cmd = np.zeros(3)
            self._base_shaped = np.zeros(3)
            self._base_axis = None
            self._base_quiet_elapsed = 0.0
            joints = self._measured_joint_targets()   # "send nothing" == hold this pose
        else:
            base_cmd = (self._mobile_base_cmd(resp) if (self.mobile and not self._base_held)
                        else self._shape_base_twist(resp["base_twist"]))
            joints = self._clamp_joint_step(WBCClient.result_to_joint_targets(resp))
        a[self.cai["base"]] = th.tensor(np.asarray(base_cmd, dtype=float), dtype=th.float32)
        a[self.cai["trunk"]] = th.tensor([joints[n] for n in TORSO])
        a[self.cai["camera"]] = th.tensor([joints[n] for n in HEAD])
        a[self.cai["arm_left"]] = th.tensor([joints[n] for n in ARM_L])
        a[self.cai["arm_right"]] = th.tensor([joints[n] for n in ARM_R])
        a[self.cai["gripper_left"]] = float(grip_l)
        a[self.cai["gripper_right"]] = float(grip_r)
        self.env.step(a)
        self._tick += 1
        resp["hold"] = hold
        resp["effective_targets"] = effective
        resp["sent_joints"] = dict(joints)
        resp["sent_base_twist"] = np.asarray(base_cmd, dtype=float).copy()
        quality = getattr(self, "_quality", None)
        if quality is not None:
            quality["hold_ticks"] += int(hold)
            tilt = self.chassis_tilt()[1:]
            quality["max_chassis_tilt_deg"] = max(
                quality["max_chassis_tilt_deg"],
                float(np.degrees(max(abs(v) for v in tilt))),
            )
            previous = self._previous_effective_targets
            if previous is not None:
                from scipy.spatial.transform import Rotation as SciRot
                for side in ("left", "right", "head"):
                    if effective[side] is None or previous[side] is None:
                        continue
                    dp = float(np.linalg.norm(effective[side][:3, 3] - previous[side][:3, 3]))
                    dr = float(np.degrees(SciRot.from_matrix(
                        previous[side][:3, :3].T @ effective[side][:3, :3]
                    ).magnitude()))
                    if dp > quality["max_target_jump_m"]:
                        phase = None
                        if self.task is not None and getattr(self.task, "_segs", None):
                            index = min(getattr(self.task, "_seg_i", 0), len(self.task._segs) - 1)
                            phase = self.task._segs[index][0]
                        quality["max_target_jump_m"] = dp
                        quality["max_target_jump_detail"] = {
                            "tick": self._tick,
                            "side": side,
                            "phase": phase,
                        }
                    if side in {"left", "right"}:
                        quality["max_arm_target_jump_m"] = max(
                            quality["max_arm_target_jump_m"], dp
                        )
                    quality["max_target_jump_deg"] = max(quality["max_target_jump_deg"], dr)
            self._previous_effective_targets = effective
        return resp

    #: Joint groups the follower clamps independently -- its `cmds` dict in `Driver.actuate`.
    _JOINT_GROUPS = {"torso": TORSO, "left_arm": ARM_L, "right_arm": ARM_R, "head": HEAD}

    def _measured_joint_targets(self):
        """Where the joints are now -- what a HOLD tick commands, since the sim must send a target
        every step where the follower simply skips the write."""
        q = self.robot.get_joint_positions().detach().cpu().numpy()
        return {n: float(q[self.name2idx[n]]) for g in self._JOINT_GROUPS.values() for n in g}

    def _clamp_joint_step(self, tgt):
        """Per-tick joint-step clamp, per group, exactly as `Driver.actuate` applies it.

        The anchor is the previous CLAMPED COMMAND (not the measurement), re-seeded from measured
        position whenever it is missing -- after a reset or on resume from a hold. A sustained
        demand above twice the clamp is a runaway target and aborts, as it does on hardware.
        """
        q = None
        out, max_over = dict(tgt), 0.0
        for group, names in self._JOINT_GROUPS.items():
            cmd = np.array([tgt[n] for n in names], dtype=float)
            prev = self._prev_joint_cmd.get(group)
            if prev is None:
                if q is None:
                    q = self.robot.get_joint_positions().detach().cpu().numpy()
                prev = np.array([float(q[self.name2idx[n]]) for n in names], dtype=float)
            clamped, requested = base_ctrl.clamp_joint_step(prev, cmd, self.max_joint_step)
            max_over = max(max_over, float(requested))
            self._prev_joint_cmd[group] = clamped
            out.update(dict(zip(names, clamped.tolist())))
        if self.max_joint_step > 0 and max_over > 2.0 * self.max_joint_step:
            self._overstep_ticks += 1
            if self._overstep_ticks > self.joint_step_abort_ticks:
                raise RuntimeError(
                    f"IK joint step {max_over:.3f} rad exceeded 2x the {self.max_joint_step:g} rad "
                    f"clamp for {self._overstep_ticks} ticks -- aborting for safety.")
        else:
            self._overstep_ticks = 0
        return out

    def _interpolate_command(self, left, right, head):
        """Hold the expert's target at the LEADER rate and lerp/slerp it up to the IK rate.

        On hardware the leader publishes at `cmd_rate` (10 Hz) and the follower glides each
        command over 1/cmd_rate toward the next, at `ik_rate` (100 Hz). A scripted expert can
        trivially emit a fresh target every IK tick, and this env used to -- which quietly makes
        the sim a smoother plant than the robot, and skips the one piece of the command path a
        LEARNED policy will certainly meet: `collect_demos` records at 10 Hz, so a policy's
        actions arrive on hardware through exactly this interpolator.

        The stream count is fixed at the first tick; a task that starts supplying a head target
        halfway through would silently change the segment geometry, so it raises instead.
        """
        poses = [left, right] + ([head] if head is not None else [])
        if self._interp is None:
            self._interp_n = len(poses)
            self._interp = base_ctrl.TargetInterpolator(1.0 / self.cmd_rate, *poses)
        if len(poses) != self._interp_n:
            raise ValueError(
                f"command stream changed from {self._interp_n} to {len(poses)} poses mid-episode "
                "(head target appeared or vanished); the interpolator cannot be resized")
        now = self._tick * self.dt
        if self._tick % self._cmd_period_ticks == 0:
            self._interp.push(*poses, now=now)
        out = self._interp.at(now)
        return (out[0], out[1], out[2] if head is not None else None)

    def close(self):
        try:
            self.wbc.close()
        finally:
            og.shutdown()
