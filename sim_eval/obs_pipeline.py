"""Sim observation pipeline: cameras + the RAW-HDF5 frame schema the real recorder writes.

Deliberately emits the SAME `episode_<N>.hdf5` layout as `wbc_vr_robot.py --record`, so the
existing porters run on sim takes unchanged:

    port_wbc_mobile_hdf5.py  -> LeRobot dataset  (diffusion policy / ACT)
    port_wbc_mobile_zarr.py  -> ManiFlow zarr    (point-cloud policy)
    vis_episode_processed_wbc.py                 (visualize either)

Nothing here recomputes `observation.state`/`action`: the porters do that themselves with
`WBCPolicyFK` (pinocchio FK on the same URDF the WBC solves with), so this module only has to
write the raw measurements + the raw `ik.solve` targets.

Two camera facts drive the setup:

* **The head camera must sit at `zed_depth_frame`, not at `eyes`.** The porters unproject the
  head depth with `world_t_cam` taken from the STATE's head block, which is FK of
  `whole_body_ik.HEAD_FRAME == "zed_depth_frame"`. OmniGibson's vega_robotiq instead carries its
  VisionSensor on the `eyes` link, so an unretargeted cloud would be rigidly offset. `setup()`
  therefore parks the camera prim at `zed_depth_frame` (constant offset -- both are fixed frames
  on the head).
* **Optical vs USD camera convention.** `unproject_head_frame` uses the pinhole optical frame
  (x right, y DOWN, z FORWARD along the ray); a USD/OmniGibson camera looks down its own -z with
  +y up. They differ by `diag(1, -1, -1)` = `_OPTICAL_TO_USD`, applied when placing the prim.

Depth is taken from the `depth_linear` modality (Isaac `distance_to_image_plane`, i.e. the
z-depth the pinhole model wants) -- NOT `depth`, which OmniGibson maps to `distance_to_camera`
(euclidean range) and would bow the cloud outward toward the image edges. It is stored as uint16
millimetres (`wbc_pointcloud.DEPTH_SCALE_M = 1e-3`); non-finite/out-of-range pixels become 0,
which falls below `MIN_DEPTH_M` and is dropped by the unprojection.

`zed_sim.ZedSimOptions` optionally makes the cameras behave like the real ZEDs -- real field of
view + the publisher's crop/resize, and the measured depth-dropout process. Every switch defaults
OFF, so unless one is passed the frames are plain OmniGibson renders as before.

vega_robotiq has no wrist camera prim, so `setup()` creates one under the eef link, pulled back
and raised above the measured eef->fingertip direction (`finger_grasp_point`), then aimed at the
grasp point so the fingers and grasped object fill the frame.

All poses written are in the WBC **engage-origin world** (`VegaOGEnv.og_to_wbc`), the frame
`obs/base/pose`, `action/*` and the env-state share on hardware.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

import omnigibson as og

import zed_sim

# optical (x right, y down, z forward)  ->  USD camera (x right, y up, z backward)
_OPTICAL_TO_USD = np.diag([1.0, -1.0, -1.0, 1.0])
DEPTH_SCALE_M = 1e-3      # wbc_pointcloud.DEPTH_SCALE_M -- uint16 millimetres
MAX_DEPTH_MM = 65535
_NEAR_CLIP_M = zed_sim.NEAR_CLIP_M   # culls the robot's own chest (see zed_sim.NEAR_CLIP_M)


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _T(pos, rot3):
    T = np.eye(4)
    T[:3, :3] = rot3
    T[:3, 3] = pos
    return T


class SimObsRecorder:
    """Camera setup + per-tick raw-HDF5 frames for one episode.

    Call `setup()` once after `env.reset()` (the camera offsets are read off the reset pose),
    then `frame(cmd)` on every recorded tick.
    """

    def __init__(self, env, arm="left", wrist_back=0.12, wrist_up=0.08, seed=0, zed=None):
        self.env = env
        self.arm = arm
        self.eef_link = {"left": "L_ee", "right": "R_ee"}[arm]
        self.wrist_back = wrist_back
        self.wrist_up = wrist_up
        self.head_cam = None
        self.wrist_cam = None
        self._head_key = None
        self._render_k = None                 # K of the raw render (pre crop/resize)
        self.zed = zed or zed_sim.ZedSimOptions()   # all realism switches default OFF
        self._rng = np.random.default_rng(seed)   # depth-dropout draws; per-episode reproducible

    # ---- cameras ----
    def setup(self):
        robot = self.env.robot
        self.head_cam = next(iter(robot.sensors.values()))
        self._retarget_head_camera()
        self._render_k = zed_sim.configure_head_camera(self.head_cam, self.zed)
        if self.wrist_cam is None:            # created once; re-aimed on every episode reset
            self.wrist_cam = self._make_wrist_camera()
            zed_sim.configure_wrist_camera(self.wrist_cam, self.zed)
        self._place_wrist_camera()
        for _ in range(4):
            og.sim.render()
        k = self.intrinsic()
        err = float(np.abs(k - zed_sim.ZED_K).max())
        print(f"[obs] zed realism: {self.zed.summary()}; head K fx={k[0,0]:.2f} fy={k[1,1]:.2f} "
              f"cx={k[0,2]:.2f} cy={k[1,2]:.2f} (dev from real ZED_K {err:.3f} px)", flush=True)
        if self.zed.match_zed_fov and err > 1.0:
            raise RuntimeError(
                f"--match-zed-fov is on but the sim head intrinsic is {err:.2f} px off the real "
                f"ZED_K, so the policy would see a different camera than the robot. "
                f"Got\n{k}\nwant\n{zed_sim.ZED_K}"
            )

    def _retarget_head_camera(self):
        """Park the head VisionSensor at `zed_depth_frame` with the optical convention.

        The prim's parent is the `eyes` LINK, and both frames are rigid on the head, so one
        constant parent-frame pose holds for the whole episode.
        """
        link = self.head_cam.prim_path.rstrip("/").split("/")[-2]     # ".../<link>/Camera"
        T_parent = self.env.link_pose(link)
        T_target = self.env.link_pose("zed_depth_frame") @ _OPTICAL_TO_USD
        T_local = np.linalg.inv(T_parent) @ T_target
        self.head_cam.set_position_orientation(
            position=T_local[:3, 3].tolist(),
            orientation=R.from_matrix(T_local[:3, :3]).as_quat().tolist(),
            frame="parent",
        )
        # `eyes` sits 1.1 cm FORWARD of zed_depth_frame, i.e. just outside the head shell, so at
        # the exact zed pose the shell fills the frame (measured: every pixel black at 5 mm), and
        # the chest yoke at ~0.106 m straddles a 0.10 m near plane and paints a black rectangle
        # into the lower-left corner. `zed_sim.NEAR_CLIP_M` clears both -- see the constant.
        self.head_cam.clipping_range = (_NEAR_CLIP_M, 1.0e7)

    def _make_wrist_camera(self):
        """Create a VisionSensor prim on the eef link (vega_robotiq ships no wrist camera)."""
        from omnigibson.sensors import create_sensor
        from omnigibson.utils.usd_utils import absolute_prim_path_to_scene_relative

        env, robot = self.env, self.env.robot
        h, w = zed_sim.wrist_render_hw(self.zed)
        path = f"{robot.prim_path}/{self.eef_link}/WristCam"
        sensor = create_sensor(
            sensor_type="Camera",
            relative_prim_path=absolute_prim_path_to_scene_relative(env.env.scene, path),
            name=f"{robot.name}:wrist_cam",
            modalities=["rgb"],
            sensor_kwargs={"image_height": h, "image_width": w},
        )
        sensor.load(env.env.scene)
        if not sensor.initialized:
            sensor.initialize()
        return sensor

    def _place_wrist_camera(self):
        """Mount the wrist camera above the approach axis and aim it at the grasp point."""
        env = self.env
        Tee = env.link_pose(self.eef_link)
        grasp_point = env.finger_grasp_point(self.arm)
        approach = grasp_point - Tee[:3, 3]                           # eef -> fingertips (world)
        approach /= max(np.linalg.norm(approach), 1e-9)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(approach, up)
        right /= max(np.linalg.norm(right), 1e-9)
        mount_up = np.cross(right, approach)
        camera_pos = Tee[:3, 3] - self.wrist_back * approach + self.wrist_up * mount_up
        view_fwd = grasp_point - camera_pos
        view_fwd /= max(np.linalg.norm(view_fwd), 1e-9)
        right = np.cross(view_fwd, up)
        right /= max(np.linalg.norm(right), 1e-9)
        view_up = np.cross(right, view_fwd)
        # optical frame: x right, y down, z forward
        T_opt = _T(camera_pos, np.stack([right, -view_up, view_fwd], axis=1))
        T_local = np.linalg.inv(Tee) @ T_opt @ _OPTICAL_TO_USD
        self.wrist_cam.set_position_orientation(
            position=T_local[:3, 3].tolist(),
            orientation=R.from_matrix(T_local[:3, :3]).as_quat().tolist(),
            frame="parent",
        )

    def intrinsic(self):
        """K of the PUBLISHED head view (the render's K put through any crop/resize)."""
        return zed_sim.head_intrinsic(self._render_k, self.zed)

    # ---- per-frame data ----
    def _head_images(self):
        obs, _ = self.head_cam.get_obs()
        return zed_sim.head_frame(_np(obs["rgb"]), _np(obs["depth_linear"]),
                                  self._rng, self._render_k, self.zed)

    def _wrist_rgb(self):
        obs, _ = self.wrist_cam.get_obs()
        return zed_sim.wrist_frame(_np(obs["rgb"]), self.zed)

    def frame(self, cmd, timestamp_ns, wbc_resp=None):
        """One raw-schema frame dict (see module docstring). `cmd` is the ExpertCommand."""
        env = self.env
        q = _np(env.robot.get_joint_positions())
        joints = {grp: np.array([q[env.name2idx[n]] for n in names], dtype=np.float32)
                  for grp, names in (("torso", _TORSO), ("left_arm", _ARM_L),
                                     ("right_arm", _ARM_R), ("head", _HEAD))}
        rgb, depth = self._head_images()
        head_tgt = cmd.head_target if cmd.head_target is not None else env.link_pose("zed_depth_frame")
        base = np.array(env.base_xyyaw(), dtype=np.float32)
        frame = {
            "timestamp_ns": np.int64(timestamp_ns),
            "obs": {
                "images": {"head_left_rgb": rgb, "head_depth": depth,
                           "left_wrist_rgb": self._wrist_rgb()},
                "joint": joints,
                "base": {"pose": base},
                # achieved gripper reading (hardware: raw FC03; sim: driven knuckle angle)
                "gripper": {side: np.float32(env.finger_qpos(side)[0]) for side in ("left", "right")},
            },
            "action": {
                # verbatim ik.solve targets, WBC engage-origin world -- the action the policy learns
                "eef": {"left": env.og_to_wbc(cmd.left_target).astype(np.float32),
                        "right": env.og_to_wbc(cmd.right_target).astype(np.float32)},
                "head": env.og_to_wbc(head_tgt).astype(np.float32),
                "gripper": {"left": _binary_grip(env, cmd.gripper_left),
                            "right": _binary_grip(env, cmd.gripper_right)},
            },
        }
        if wbc_resp is not None:   # parity with the hardware take; not read by the porters
            tgt = _joint_targets(wbc_resp)
            frame["action"]["joint"] = {grp: np.array([tgt[n] for n in names], dtype=np.float32)
                                        for grp, names in (("torso", _TORSO), ("left_arm", _ARM_L),
                                                           ("right_arm", _ARM_R), ("head", _HEAD))}
            frame["action"]["base"] = {"pose": np.asarray(wbc_resp["base_pose"], dtype=np.float32)}
        return frame


_TORSO = ["torso_j1", "torso_j2", "torso_j3"]
_ARM_L = [f"L_arm_j{i}" for i in range(1, 8)]
_ARM_R = [f"R_arm_j{i}" for i in range(1, 8)]
_HEAD = ["head_j1", "head_j2", "head_j3"]


def _binary_grip(env, cmd_value):
    """Task convention (-1 close / +1 open) -> the dataset's categorical 1 closed / 0 open."""
    return np.float32(1.0 if float(cmd_value) == float(env.grip_close) else 0.0)


def _joint_targets(resp):
    from wbc_client import WBCClient
    return WBCClient.result_to_joint_targets(resp)
