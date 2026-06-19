#!/usr/bin/env python3
"""Whole-body VR leader: publish live EEF target poses for ``wbc_vr_record.py``.

1. reads Quest 3 controller + headset poses (WebXR, via ``WebXRVRReader``);
2. on a trigger-hold, captures the headset pose and computes two calibration
   transforms: a full head-pose transform for gaze tracking, and a gravity-aligned
   yaw+translation transform for wrist/EEF target positions;
3. each frame publishes the **Cartesian L_ee / R_ee target poses**

       left_ee_pose  = robot_base_t_vr_base_eef @ vr_left_controller
       right_ee_pose = robot_base_t_vr_base_eef @ vr_right_controller

   on the ``vr/joints`` topic (``VRJointData``), exactly the field
   ``wbc_vr_record.py`` consumes to drive its whole-body IK (base + torso + arms);
4. each ``teleop`` frame also publishes the **calibrated headset pose**

       head_ee_pose[:3, :3] = (robot_base_t_vr_base @ vr_headset)[:3, :3]
       head_ee_pose[:3,  3] = (robot_base_t_vr_base_eef @ vr_headset)[:3, 3]

   -- the head-frame (zed_depth_frame) teleop target. Its orientation uses the full
   head calibration, while its position uses the gravity-aligned room calibration so
   horizontal headset motion stays horizontal even if the robot's nominal optical
   frame is pitched down. The follower either tracks this pose as a WBC head task
   (``head_mode: 'ik'``) or uses its orientation for dedicated pan/tilt head IK
   (``head_mode: 'track'``).

Commands are published at a low rate (``--rate``, default **10 Hz**); the follower
lerp/slerp-interpolates each command over ``1/rate`` up to its 100 Hz IK ticks
(``wbc_vr_record.py --cmd-rate`` must match this rate). This mirrors the reference
deps/rby1-wbc split: 10 Hz ``trajectory_frequency_hz`` commands, 100 Hz
``ik_frequency_hz`` with ``use_interpolation: true``.

This preserves the ``vr_reader`` controller convention (controller forward 10 cm ->
target forward 10 cm in the calibrated base frame), with two deliberate changes:

  **Gravity-aligned translational calibration.** The nominal ``zed_depth_frame`` is
  pitched downward, so using the full head frame to calibrate target positions would
  make a horizontal walk create a fake target-height change. EEF targets therefore
  use headset yaw + translation only. The head target uses that same translation
  calibration, while keeping the full head-frame orientation calibration.

  **No legacy ``INIT_JOINT`` constants.** ``vr_reader`` anchors the calibration on
  ``FK(head_link)`` evaluated at ``omniteleop.common.vr_mode_const.INIT_*`` (the
  real-robot/dexcontrol-convention init posture). Here the calibration reference
  head pose comes from the **whole-body IK model's own nominal posture**
  (``VegaWholeBodyIK`` / ``WBCConfig.nominal_posture``), so the leader and the
  ``wbc_vr_record`` follower agree on the head/base frames and nothing depends on
  the legacy init constants.

Because the follower derives base motion from the EE targets, ``chassis_*`` is
published for schema completeness / other followers but is **ignored** by
``wbc_vr_record.py``. ``head_pos`` / ``*_arm_pos`` are left empty (the follower
computes all joints itself); ``head_ee_pose`` carries the Cartesian head target
from step 4 (empty while ``static``).

Run in the dexmate conda env (has pinocchio + pink + dexcomm + aiohttp/socketio)::

    # USB: adb reverse tcp:5067 tcp:5067, then open http://localhost:5067 in the
    #      Quest browser and tap "Start XR".
    /home/yixuan/miniforge3/envs/dexmate/bin/python scripts/wbc_vr_leader.py

    # ... and in another shell, the follower:
    /home/yixuan/miniforge3/envs/dexmate/bin/python scripts/wbc_vr_record.py

Controls (mirrors ``vr_reader``):
  * **hold right grip trigger >= 1 s** in ``static`` -> capture calibration and
    begin streaming targets (``teleop``);
  * **left X button** -> back to ``static`` (re-calibrate on the next trigger-hold);
  * index triggers -> ``left_gripper`` / ``right_gripper`` [0, 1];
  * left thumbstick -> ``chassis_vx`` / ``chassis_vy``; right thumbstick x ->
    ``chassis_wz`` (published, but unused by ``wbc_vr_record``).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from typing import Optional

import numpy as np
from dexcomm import Node
from dexcomm.codecs import DictDataCodec

from omniteleop.common import get_config
from omniteleop.common.schemas import VRJointData
from omniteleop.follower.whole_body_ik import (
    HEAD_FRAME,
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    VegaWholeBodyIK,
    WBCConfig,
)
from omniteleop.leader.communication.webxr_vr_reader import WebXRVRReader

# Sentinel poses the WebXR reader emits (post-transform, in the robot frame) when a
# controller is not tracked. Copied from ``vr_reader.INVALID_{LEFT,RIGHT}_POSE`` so
# we can hold the last good target instead of snapping to garbage.
INVALID_LEFT_POSE = np.array(
    [[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)
INVALID_RIGHT_POSE = np.array(
    [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)

DEFAULT_CERT = "/home/yixuan/omniteleop/tests/cert.pem"
DEFAULT_KEY = "/home/yixuan/omniteleop/tests/key.pem"
_TRIGGER_PRESS = 0.7  # right grip value counted as a press
_TRACK_ATOL = 1e-6  # tolerance for matching the untracked sentinel poses


def _to_mat(pose) -> np.ndarray:
    """4x4 matrix from a pin.SE3 (``.homogeneous``) or a 4x4 array."""
    return np.asarray(pose.homogeneous if hasattr(pose, "homogeneous") else pose, dtype=float)


def _rotz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _heading_yaw(pose: np.ndarray) -> float:
    """Yaw of a head/optical frame's viewing direction projected onto the ground."""
    rot = np.asarray(pose, dtype=float)[:3, :3]
    fwd_xy = rot[:2, 2]
    norm = float(np.linalg.norm(fwd_xy))
    if norm > 1e-8:
        return float(np.arctan2(fwd_xy[1], fwd_xy[0]))

    # Degenerate only when the viewing axis is almost vertical. The optical x-axis
    # is camera-right, so heading is 90 deg counter-clockwise from that projection.
    right_xy = rot[:2, 0]
    norm = float(np.linalg.norm(right_xy))
    if norm <= 1e-8:
        return 0.0
    return float(np.arctan2(right_xy[1], right_xy[0]) + np.pi / 2.0)


def _gravity_aligned_calibration(
    base_t_head_nominal: np.ndarray, vr_t_head_calib: np.ndarray
) -> np.ndarray:
    """Room-to-base transform for EEF targets, using yaw + translation only.

    The robot's nominal ``zed_depth_frame`` is pitched downward. Using the full
    head frame as the wrist calibration therefore pitches the entire VR room, so a
    horizontal operator walk creates a fake EEF height change. This calibration
    keeps gravity aligned: headset position maps to the nominal head position and
    headset yaw maps to nominal head yaw, while pitch/roll are left out of the
    room-to-base transform.
    """
    base_t_head_nominal = np.asarray(base_t_head_nominal, dtype=float)
    vr_t_head_calib = np.asarray(vr_t_head_calib, dtype=float)
    yaw = _heading_yaw(base_t_head_nominal) - _heading_yaw(vr_t_head_calib)
    rot = _rotz(yaw)
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = base_t_head_nominal[:3, 3] - rot @ vr_t_head_calib[:3, 3]
    return out


def _is_tracked(pose: np.ndarray, sentinel: np.ndarray) -> bool:
    """True unless ``pose`` matches the untracked-controller sentinel."""
    return not np.allclose(pose, sentinel, atol=_TRACK_ATOL)


class WBCVRLeader:
    """Reads Quest poses, calibrates once, and streams Cartesian EEF targets."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.rate = args.rate
        self.hold_seconds = args.hold_seconds
        self.stick_max_vx = args.stick_max_vx
        self.stick_max_vy = args.stick_max_vy
        self.stick_max_wz = args.stick_max_wz
        self.stick_deadzone = args.stick_deadzone

        # Zenoh publisher on the same topic the follower subscribes to.
        self.node = Node(name="wbc_vr_leader", namespace=args.namespace)
        self.topic = get_config().get_topic("vr_joints", "vr/joints")
        self.pub = self.node.create_publisher(self.topic, encoder=DictDataCodec.encode)

        # Calibration reference head pose, from the whole-body IK model's *nominal*
        # posture (NOT the legacy INIT_JOINT constants). At nominal the planar base
        # is at the world origin, so frame_pose() values are already in the base
        # frame, and they match exactly what wbc_vr_record's VegaWholeBodyIK uses.
        cfg = WBCConfig()
        if args.urdf:
            cfg.urdf_path = args.urdf
        ik = VegaWholeBodyIK(cfg)
        ik.reset()
        self.T_base_head = _to_mat(ik.frame_pose(HEAD_FRAME))
        if self.T_base_head.shape != (4, 4):
            raise ValueError(
                f"nominal head FK has shape {self.T_base_head.shape}, expected (4, 4)"
            )
        # Nominal EE poses (base frame) — only used to report the calibration delta.
        self._nominal_left = _to_mat(ik.frame_pose(LEFT_EE_FRAME))
        self._nominal_right = _to_mat(ik.frame_pose(RIGHT_EE_FRAME))
        del ik  # solver no longer needed; we stream Cartesian EE + head targets
        # (The head IK itself runs follower-side -- VegaWholeBodyIK.solve_head against
        # the live whole-body configuration; this leader maps the headset through
        # the full calibration while EEF targets use gravity-aligned yaw.)

        self.quest = WebXRVRReader(
            host=args.host, port=args.port,
            ssl_certfile=args.cert, ssl_keyfile=args.key,
        )

        # State
        self.stage = "static"  # "static" (estop) or "teleop"
        self.robot_base_t_vr_base: Optional[np.ndarray] = None
        self.robot_base_t_vr_base_eef: Optional[np.ndarray] = None
        self.last_left_target: Optional[np.ndarray] = None
        self.last_right_target: Optional[np.ndarray] = None
        self._trigger_start: Optional[float] = None
        self._prev_x = False
        self.running = False

    # -- helpers ----------------------------------------------------------------

    def _trigger_held(self, transforms) -> bool:
        """True once the right grip trigger is continuously held for hold_seconds."""
        val = float(transforms["right_hand_trigger"])
        now = time.perf_counter()
        if val > _TRIGGER_PRESS:
            if self._trigger_start is None:
                self._trigger_start = now
            elif now - self._trigger_start >= self.hold_seconds:
                self._trigger_start = None
                return True
        else:
            self._trigger_start = None
        return False

    def _thumbstick_to_chassis(self, transforms) -> tuple[float, float, float]:
        def dz(v: float) -> float:
            return v if abs(v) > self.stick_deadzone else 0.0

        left = transforms["left_thumbstick"]
        right = transforms["right_thumbstick"]
        # Quest stick y is negative when pushed up -> negate so forward is positive.
        vx = -dz(left[1]) * self.stick_max_vx
        vy = -dz(left[0]) * self.stick_max_vy
        wz = -dz(right[0]) * self.stick_max_wz
        return vx, vy, wz

    def _calibrate(self, headset_pose: np.ndarray) -> None:
        """Anchor room-to-base transforms so the headset maps onto the nominal head."""
        headset_pose = np.asarray(headset_pose, dtype=float)
        if headset_pose.shape != (4, 4):
            raise ValueError(f"headset pose has shape {headset_pose.shape}, expected (4, 4)")
        self.robot_base_t_vr_base = self.T_base_head @ np.linalg.inv(headset_pose)
        self.robot_base_t_vr_base_eef = _gravity_aligned_calibration(
            self.T_base_head, headset_pose
        )

    def _map_head_target(self, vr_head: np.ndarray) -> np.ndarray:
        """Calibrated head target: the headset pose mapped into the robot base frame.

        Orientation uses the full head calibration. Translation uses the
        gravity-aligned transform, because in WBC ``head_mode: "ik"`` this
        translation is a real whole-body target; mapping it through the downward
        pitched optical frame would turn horizontal headset motion into a height
        command. At the calibration instant the mixed pose still equals the nominal
        head pose by construction.
        """
        assert self.robot_base_t_vr_base is not None
        assert self.robot_base_t_vr_base_eef is not None
        vr_head = np.asarray(vr_head, dtype=float)
        if vr_head.shape != (4, 4):
            raise ValueError(f"headset pose has shape {vr_head.shape}, expected (4, 4)")
        out = self.robot_base_t_vr_base @ vr_head
        out[:3, 3] = (self.robot_base_t_vr_base_eef @ vr_head)[:3, 3]
        return out

    def _map_targets(self, vr_l: np.ndarray, vr_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map controller poses to base-frame EEF targets, holding last-good if untracked."""
        assert self.robot_base_t_vr_base_eef is not None
        if _is_tracked(vr_l, INVALID_LEFT_POSE):
            self.last_left_target = self.robot_base_t_vr_base_eef @ vr_l
        if _is_tracked(vr_r, INVALID_RIGHT_POSE):
            self.last_right_target = self.robot_base_t_vr_base_eef @ vr_r
        return self.last_left_target, self.last_right_target

    def _publish(
        self,
        left_target: Optional[np.ndarray],
        right_target: Optional[np.ndarray],
        head_target: Optional[np.ndarray],
        left_gripper: float,
        right_gripper: float,
        chassis: tuple[float, float, float],
    ) -> None:
        left_flat = left_target.reshape(-1).tolist() if left_target is not None else []
        right_flat = right_target.reshape(-1).tolist() if right_target is not None else []
        head_flat = head_target.reshape(-1).tolist() if head_target is not None else []
        data = VRJointData(
            timestamp_ns=time.time_ns(),
            head_pos=[],
            left_arm_pos=[],
            right_arm_pos=[],
            left_gripper=float(left_gripper),
            right_gripper=float(right_gripper),
            chassis_vx=float(chassis[0]),
            chassis_vy=float(chassis[1]),
            chassis_wz=float(chassis[2]),
            estop=(self.stage == "static"),
            calib_stage=self.stage,
            left_ee_pose=left_flat,
            right_ee_pose=right_flat,
            head_ee_pose=head_flat,
        )
        # Publisher was created with encoder=DictDataCodec.encode, so pass the
        # plain dict and let it encode (mirrors vr_reader._publish).
        self.pub.publish(asdict(data))

    # -- main loop --------------------------------------------------------------

    def run(self) -> None:
        """Main loop: poll the Quest, calibrate on a grip-hold, then stream EEF targets."""
        self.running = True
        self.quest.start()
        print("[wbc_vr_leader] waiting for Quest data ...")
        self.quest.wait_for_data()
        print(f"[wbc_vr_leader] Quest connected. Publishing on '{self.topic}'.")
        print("[wbc_vr_leader] Stage static — hold right grip trigger "
              f">= {self.hold_seconds:.0f}s to calibrate and start streaming targets.")

        dt = 1.0 / self.rate
        last_print = 0.0
        t0 = time.perf_counter()
        try:
            while self.running:
                now = time.perf_counter()
                transforms = self.quest.get_latest_transformation()
                if transforms is None:
                    time.sleep(dt)
                    continue

                # left X -> reset to static (re-calibrate on next trigger-hold).
                x_now = bool(transforms["left_x_button"])
                if x_now and not self._prev_x:
                    self.stage = "static"
                    self.robot_base_t_vr_base = None
                    self.robot_base_t_vr_base_eef = None
                    self.last_left_target = None
                    self.last_right_target = None
                    print("\n[wbc_vr_leader] reset -> static (left X). "
                          "Hold right grip to re-calibrate.")
                self._prev_x = x_now

                vr_l = transforms["left_wrist"]
                vr_r = transforms["right_wrist"]

                if self.stage == "static":
                    if self._trigger_held(transforms):
                        if not (_is_tracked(vr_l, INVALID_LEFT_POSE)
                                and _is_tracked(vr_r, INVALID_RIGHT_POSE)):
                            print("\n[wbc_vr_leader] both controllers must be tracked "
                                  "to calibrate — hold them in view and retry.")
                        else:
                            self._calibrate(transforms["head"])
                            self._map_targets(vr_l, vr_r)  # seed last-good targets
                            self.stage = "teleop"
                            d = self.last_left_target[:3, 3] - self._nominal_left[:3, 3]
                            print(f"\n[wbc_vr_leader] calibrated -> teleop. "
                                  f"L target offset from nominal = "
                                  f"[{d[0]:+.3f} {d[1]:+.3f} {d[2]:+.3f}] m")
                    self._publish(None, None, None, 0.0, 0.0, (0.0, 0.0, 0.0))
                else:  # teleop
                    left_target, right_target = self._map_targets(vr_l, vr_r)
                    head_target = self._map_head_target(transforms["head"])
                    chassis = self._thumbstick_to_chassis(transforms)
                    self._publish(
                        left_target, right_target, head_target,
                        transforms["left_index_trigger"], transforms["right_index_trigger"],
                        chassis,
                    )

                t = now - t0
                if t - last_print >= 0.5:
                    last_print = t
                    if self.stage == "teleop" and self.last_left_target is not None:
                        lp = self.last_left_target[:3, 3]
                        rp = self.last_right_target[:3, 3]
                        print(f"t={t:6.1f}s  teleop  "
                              f"L=[{lp[0]:+.3f} {lp[1]:+.3f} {lp[2]:+.3f}]  "
                              f"R=[{rp[0]:+.3f} {rp[1]:+.3f} {rp[2]:+.3f}] (base frame)",
                              end="\r")
                    else:
                        print(f"t={t:6.1f}s  static (hold right grip to calibrate)    ",
                              end="\r")

                sleep = dt - (time.perf_counter() - now)
                if sleep > 0:
                    time.sleep(sleep)
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.quest.close()
            self.node.shutdown()
            print("\n[wbc_vr_leader] stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rate", type=float, default=10.0,
                        help="command publish rate in Hz (default: 10; the follower "
                             "lerp/slerp-interpolates the commands up to its 100 Hz "
                             "IK rate -- see wbc_vr_record.py --cmd-rate, which must "
                             "match this).")
    parser.add_argument("--namespace", default="",
                        help="Zenoh namespace (must match the follower; default empty).")
    parser.add_argument("--host", default="0.0.0.0", help="WebXR bind host (default 0.0.0.0).")
    parser.add_argument("--port", type=int, default=5067, help="WebXR port (default 5067).")
    parser.add_argument("--cert", default=DEFAULT_CERT, help="TLS cert for the WebXR server.")
    parser.add_argument("--key", default=DEFAULT_KEY, help="TLS key for the WebXR server.")
    parser.add_argument("--urdf", default=None, help="override the Vega URDF path.")
    parser.add_argument("--hold-seconds", type=float, default=1.0,
                        help="right-grip hold time to calibrate/start (default 1.0).")
    parser.add_argument("--stick-max-vx", type=float, default=0.3)
    parser.add_argument("--stick-max-vy", type=float, default=0.2)
    parser.add_argument("--stick-max-wz", type=float, default=0.5)
    parser.add_argument("--stick-deadzone", type=float, default=0.1)
    args = parser.parse_args()

    if args.rate <= 0:
        raise ValueError(f"--rate must be > 0, got {args.rate}")

    WBCVRLeader(args).run()


if __name__ == "__main__":
    main()
