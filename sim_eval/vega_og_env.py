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
        # Base is LOCKED in the sim, so the teleop-safety terms that keep the upper body centered
        # over the (moving) chassis are moot and only shrink/contort the arm's reach:
        #   - head-world-position objective (base-following-head): off
        #   - torso-top-x anchor (hold arm mount over base): off
        #   - CoM-over-base centering (tip-over safety): off -- no tip risk with a locked base
        #   - torso posture pull: relaxed so the torso can LEAN toward table targets (extra reach)
        # Together these open up a much larger, cleaner reachable workspace for precise grasping.
        # Base is planar in sim (no tip risk), so relax the teleop-safety terms that otherwise
        # fight the arm / shrink reach. SAME config for tabletop AND mobile: mobile base motion is
        # NOT driven by a head-world-position cost -- it emerges from the whole-body IK moving the
        # (unlocked) base to reach the arm's far EE targets, so the head cost stays zero.
        if wbc_overrides is None:
            # Works under EITHER value, because the mobile task derives its head target from the SAME
            # waypoints as the arm target (see MobilePickPlaceTask): [10000] = the head command drives
            # the base and agrees with the reach posture, [0] = the head is ignored and the arm's far
            # EE target drives the base. Mobile defaults to 10000 (teleop-like base-follows-head).
            # head_world_position_cost 10000 when mobile is REQUIRED, not a preference: it is the
            # only base-position command in the stack (head world x/y == base x/y, see
            # MobilePickPlaceTask). At 0 the head target is ignored, the base wanders laterally
            # (measured y 0.40 -> 0.89) and, with the torso anchor also off, the upper body folds
            # (torso_j3 -0.3 -> -49 deg, head sinking 1.49 -> 1.11 m). Turning the anchor ON fixes
            # the fold but is NOT a rescue: it costs reach and at cost 10000 it wrecks the take
            # (base yaw -41 deg, ee_err 0.27), so it stays off and mobile stays at 10000.
            wbc_overrides = {"head_world_position_cost": [10000.0, 10000.0, 10000.0] if mobile else [0.0, 0.0, 0.0],
                             "enable_torso_top_x_anchor": False,
                            }
        self.base_x_max = None   # mobile: forward-park clamp (set by run_episode from the task)
        self.mobile = mobile
        # base-twist shaping gains/limits (wbik.yaml vr_teleop block); angular limits = 2x linear
        self.base_kp_xy, self.base_kp_yaw = 1.0, 1.5
        self.base_max_lin, self.base_max_ang = 0.45, 0.9
        self.base_deadband_lin, self.base_deadband_ang = 0.02, 0.04
        self.base_max_lin_accel, self.base_max_ang_accel = 0.4, 0.8
        self._prev_base_twist = np.zeros(3)
        self._base_held = False   # mobile: park + hold the base after navigation (P-hold)
        self.base_hold_target = np.zeros(3)   # (x,y,yaw) the base is held at (updated by drive_base)
        self.robot_pos = tuple(robot_pos)
        # OmniGibson rate hierarchy: render_dt multiple of physics_dt, action_dt multiple of
        # render_dt  =>  render_hz % action_hz == 0  AND  physics_hz % render_hz == 0.
        assert render_hz % action_hz == 0 and physics_hz % render_hz == 0, \
            f"need render_hz({render_hz}) % action_hz({action_hz})==0 and physics_hz({physics_hz}) % render_hz==0"
        self.task = task
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
            "type": "vega_robotiq", "obs_modalities": ["rgb", "depth_linear", "proprio"],
            "grasping_mode": grasping_mode, "grasping_direction": grasping_direction,
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
        if grasping_mode in ("sticky", "assisted"):
            # instant scripted grasp: magnetize as soon as a finger contacts while closing
            # (default 0.3s window is fragile under a subsampled scripted trajectory).
            from omnigibson.macros import macros
            with macros.unlocked():
                macros.robots.robot.GRASP_WINDOW = 0.0
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

    def finger_qpos(self, arm="left"):
        """The two driven gripper-joint positions for @arm."""
        q = self.robot.get_joint_positions().detach().cpu().numpy()
        return [round(float(q[self.name2idx[name]]), 4)
                for name in self.robot.finger_joint_names[arm]]

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

    def base_xyyaw(self):
        q = self.robot.get_joint_positions().detach().cpu().numpy()
        return (float(q[self.name2idx["base_footprint_x_joint"]]),
                float(q[self.name2idx["base_footprint_y_joint"]]),
                float(q[self.name2idx["base_footprint_rz_joint"]]))

    def measured_pin_q(self):
        q = self.robot.get_joint_positions().detach().cpu().numpy()
        qby = {n: float(q[self.name2idx[n]]) for n in (TORSO + ARM_L + ARM_R + HEAD)}
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
        self._prev_base_twist = np.zeros(3)
        self._base_held = False
        if self.task is not None:
            self.task.reset(self)
            for _ in range(10):
                self.env.step(self._hold_action())   # settle objects + robot at nominal
        self.T_align = self.link_pose("base")
        self.T_align_inv = np.linalg.inv(self.T_align)
        self.base_hold_target = np.array(self.base_xyyaw())   # ~0 at spawn
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
        rp = np.array(self.link_pose("base")[:3, 3])
        self._set_cam_lookat(rp + np.array([-1.4, 1.4, 1.5]), rp + np.array([1.0, -0.1, 0.5]))
        for _ in range(2):
            og.sim.render()

    def capture_third_person(self):
        obs = og.sim.viewer_camera.get_obs()[0]
        rgb = obs["rgb"] if isinstance(obs, dict) else obs
        rgb = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
        return rgb[..., :3].astype(np.uint8)

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
        """WBC-driven base (mobile): PD-track the WBC's desired base_pose with its base_twist as
        feed-forward, then deadband/clamp/slew -> (vx,vy,wz) chassis velocity in the base frame.
        base_pose (WBC world) and base_xyyaw (base_footprint joints) share the spawn-origin frame,
        since current_q is fed with base_xyyaw each tick."""
        ref = np.asarray(resp["base_pose"], dtype=float)      # WBC desired base pose (spawn frame)
        ff = np.asarray(resp["base_twist"], dtype=float)      # feed-forward (reference base frame)
        meas = np.array(self.base_xyyaw())                    # current base pose (spawn frame)
        cmd = base_ctrl.pd_twist(ref, ff, meas, self.base_kp_xy, self.base_kp_yaw,
                                 self.base_max_lin, self.base_max_ang)
        shaped = base_ctrl.shape_twist(cmd, self._prev_base_twist, self.dt,
                                       self.base_deadband_lin, self.base_deadband_ang,
                                       self.base_max_lin, self.base_max_ang,
                                       self.base_max_lin_accel, self.base_max_ang_accel)
        # Forward-park clamp: once the base reaches base_x_max, stop it driving further toward the
        # table (the WBC would otherwise keep pulling it in to shorten the arm's reach -> collision).
        # Robot faces +x (yaw~0), so base-frame vx == world +x velocity.
        if self.base_x_max is not None and self.link_pose("base")[0, 3] >= self.base_x_max and shaped[0] > 0.0:
            shaped[0] = 0.0
        self._prev_base_twist = shaped
        return shaped

    def drive_base(self, base_vel, capture=False):
        """Navigate: command base velocity (base frame) + hold arms at nominal, step.
        Updates base_hold_target so the manipulation phase holds where we drove to."""
        a = self._hold_action()
        a[self.cai["base"]] = th.tensor(np.asarray(base_vel, dtype=float), dtype=th.float32)
        self.env.step(a)
        self.base_hold_target = np.array(self.base_xyyaw())
        return self.capture_third_person() if capture else None

    def wbc_tick(self, left_og, right_og, head_og=None, grip_l=-1.0, grip_r=-1.0):
        """One WBC control tick: world targets (OG frame) -> WBC -> apply -> env.step."""
        left = self.og_to_wbc(left_og)
        right = self.og_to_wbc(right_og)
        head = self.og_to_wbc(head_og) if head_og is not None else None
        resp = self.wbc.solve(left, right, head=head, current_q=self.measured_pin_q(), dt=self.dt)
        tgt = WBCClient.result_to_joint_targets(resp)
        a = th.zeros(self.robot.action_dim)
        base_cmd = (self._mobile_base_cmd(resp) if (self.mobile and not self._base_held)
                    else self._shape_base_twist(resp["base_twist"]))
        a[self.cai["base"]] = th.tensor(base_cmd, dtype=th.float32)
        a[self.cai["trunk"]] = th.tensor([tgt[n] for n in TORSO])
        a[self.cai["camera"]] = th.tensor([tgt[n] for n in HEAD])
        a[self.cai["arm_left"]] = th.tensor([tgt[n] for n in ARM_L])
        a[self.cai["arm_right"]] = th.tensor([tgt[n] for n in ARM_R])
        a[self.cai["gripper_left"]] = float(grip_l)
        a[self.cai["gripper_right"]] = float(grip_r)
        self.env.step(a)
        return resp

    def close(self):
        try:
            self.wbc.close()
        finally:
            og.shutdown()
