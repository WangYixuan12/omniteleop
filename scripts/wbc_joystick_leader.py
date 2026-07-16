#!/usr/bin/env python3
"""Joystick-base VR leader: EEF targets in a FIXED base frame + direct chassis joysticks.

A variant of ``scripts/wbc_vr_leader.py`` for the joystick mobile-manipulation paradigm
(pairs with ``scripts/wbc_joystick_robot.py`` / ``scripts/wbc_joystick_record.py``). It
reuses ALL of ``WBCVRLeader`` -- calibration, the headset HUD, the reference-alignment
gate, the ``--calibrate-ee-offset`` helper, every button -- and changes only what the new
paradigm requires:

  * **Hand POSITION rides a FIXED gravity-aligned base frame**, not the live head-yaw
    frame. In the base leader the hand position rides the robot's head yaw so a head+hands
    turn keeps the hands centered while the base follows the head. Here the base is driven
    DIRECTLY by the joysticks and the head only pans/tilts (follower ``head_mode: track``),
    so coupling the hand to head yaw would make a head look-around drag the hands. Instead
    the hand maps through the same room calibration used for the head
    (``robot_base_t_vr_base_eef @ vr_controller``) -- exactly the classic ``vr_reader``
    base-frame mapping. Because the target is published in the BASE frame and the follower
    interprets it in the robot's CURRENT (joystick-rotated) base, the hands still ride the
    robot when you drive/turn, with no head coupling. The startup blend to the nominal EE
    (so the arm does not lunge at engage) and the absolute orientation map
    (``R_yaw @ vr @ C``) are unchanged.

  * **Thumbstick mapping: RIGHT stick drives, LEFT stick turns.** Right stick Y ->
    ``chassis_vx`` (forward/back), right stick X -> ``chassis_vy`` (strafe), left stick X
    -> ``chassis_wz`` (yaw). When configured, the leader projects this mapped twist to one
    dominant axis before publishing; the follower enforces the same policy again before
    integrating its base reference. Unlike the base follower, which ignores these fields
    and derives base motion from the whole-body QP, joystick mode drives from them directly.

  * Hands are DECOUPLED from head tracking: a lost/invalid headset frame no longer freezes
    the hand targets (the base leader holds them because its hand position needs the live
    head yaw; here it does not).

Run in the dexmate conda env, same as the base leader::

    /home/yixuan/miniforge3/envs/dexmate/bin/python scripts/wbc_joystick_leader.py
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path
from typing import Optional

import numpy as np

from omniteleop.follower.base_closed_loop import (
    mask_planar_twist_for_base_dofs,
    project_planar_twist_single_axis_for_base_dofs,
)
from omniteleop.follower.whole_body_ik import WBCConfig
from omniteleop.wbc_teleop import JoystickTeleopConfig, VRTeleopConfig

# Load the base leader as a module (scripts/ is not a package; the tests do the same via
# importlib). Executing it only binds module-level defaults -- main() is __main__-guarded --
# so this is a cheap, side-effect-free import of the class + helpers we extend.
_BASE_PATH = Path(__file__).resolve().parent / "wbc_vr_leader.py"
_spec = importlib.util.spec_from_file_location("wbc_vr_leader", _BASE_PATH)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

WBCVRLeader = _base.WBCVRLeader
INVALID_LEFT_POSE = _base.INVALID_LEFT_POSE
INVALID_RIGHT_POSE = _base.INVALID_RIGHT_POSE
_is_tracked = _base._is_tracked
DEFAULT_CERT = _base.DEFAULT_CERT
DEFAULT_KEY = _base.DEFAULT_KEY
DEFAULT_EE_OFFSET_FILE = _base.DEFAULT_EE_OFFSET_FILE
DEFAULT_HEADSET_HUD_RATE = _base.DEFAULT_HEADSET_HUD_RATE
_DEFAULT_HUD_CAMERAS = _base._DEFAULT_HUD_CAMERAS


class JoystickVRLeader(WBCVRLeader):
    """VR leader with fixed-base-frame hands and joystick-driven base (see module docstring)."""

    def __init__(self, args: argparse.Namespace) -> None:
        cfg = getattr(args, "wbc_config", None)
        if cfg is None:
            cfg = WBCConfig.from_yaml(args.config)
        joystick_cfg = getattr(args, "joystick_config", None)
        if joystick_cfg is None:
            joystick_cfg = JoystickTeleopConfig.from_yaml(args.config)
        teleop_cfg = getattr(args, "vr_teleop_config", None)
        if teleop_cfg is None:
            teleop_cfg = VRTeleopConfig.from_yaml(args.config)
        self._joystick_base_dofs = cfg.base_dofs
        self._joystick_single_axis = cfg.enable_base_single_axis
        self._joystick_xy_max_vel = teleop_cfg.stick_max_vx
        self._joystick_yaw_max_vel = teleop_cfg.stick_max_wz
        self._joystick_axis_deadband = cfg.base_single_axis_deadband
        self._joystick_axis_hysteresis = joystick_cfg.single_axis_hysteresis_ratio
        self._joystick_axis: Optional[int] = None
        super().__init__(args, wbc_config=cfg, teleop_config=teleop_cfg)

    def _calibrate(self, headset_pose: np.ndarray) -> None:
        # Engage is a fresh operator command sequence, matching the follower/IK latch reset.
        self._joystick_axis = None
        super()._calibrate(headset_pose)

    # -- hand mapping: fixed gravity-aligned base frame (NOT head-yaw relative) ------------
    def _bprime_pos(self, side, vr_now, vr_head=None, head_target=None) -> np.ndarray:
        """Hand POSITION in the calibrated base frame (room-anchored, head-independent).

        ``robot_base_t_vr_base_eef`` is the gravity-aligned room->base calibration (yaw +
        translation) captured at engage. Mapping the controller through it places the hand
        at its operator-space position expressed in the robot base frame; the follower
        applies that in the robot's CURRENT base, so the hand rides a joystick base turn
        without any dependence on where the operator is looking. ``vr_head``/``head_target``
        are accepted for signature compatibility with the base class and ignored.
        """
        assert self.robot_base_t_vr_base_eef is not None
        return (self.robot_base_t_vr_base_eef @ np.asarray(vr_now, dtype=float))[:3, 3]

    def _eef_target(self, side, vr_now, *, vr_head=None, head_target=None, now=0.0) -> np.ndarray:
        """Base-frame EEF target: fixed-frame position (+ engage blend) and absolute orientation.

        Same structure as ``WBCVRLeader._eef_target`` minus the head-geometry requirement:
        the position comes from the fixed-frame ``_bprime_pos`` above (with the per-side
        engage offset ``_hf_offset_*`` ramped to zero over ``_hf_blend_dur`` so the arm does
        not lunge), and the orientation is the unchanged absolute map
        ``R_yaw @ vr_now_rot @ C`` (``C`` = the calibrated controller->gripper offset).
        """
        vr_now = np.asarray(vr_now, dtype=float)
        ee_offset = getattr(self, f"_ee_offset_{side}", None)
        if ee_offset is None:
            ee_offset = np.eye(3)
        off = self._hf_offset_left if side == "left" else self._hf_offset_right
        blend = 1.0 if self._hf_engage_t is None else float(
            np.clip((now - self._hf_engage_t) / self._hf_blend_dur, 0.0, 1.0)
        )
        out = np.eye(4)
        out[:3, 3] = self._bprime_pos(side, vr_now) - (1.0 - blend) * off
        out[:3, :3] = self.robot_base_t_vr_base_eef[:3, :3] @ vr_now[:3, :3] @ ee_offset
        return out

    def _map_targets(self, vr_l, vr_r, *, vr_head=None, head_target=None, now=0.0):
        """Map controller poses to base-frame EEF targets, holding last-good if untracked.

        Unlike the base leader, the hands do NOT depend on ``head_target`` (the fixed-frame
        position needs no head geometry), so a lost headset frame updates the hands normally
        instead of freezing them. Each side holds its last-good target only when its own
        controller is untracked.
        """
        assert self.robot_base_t_vr_base_eef is not None
        if _is_tracked(vr_l, INVALID_LEFT_POSE):
            self.last_left_target = self._eef_target("left", vr_l, now=now)
        if _is_tracked(vr_r, INVALID_RIGHT_POSE):
            self.last_right_target = self._eef_target("right", vr_r, now=now)
        return self.last_left_target, self.last_right_target

    # -- base: RIGHT stick drives (fwd/strafe), LEFT stick turns --------------------------
    def _thumbstick_to_chassis(self, transforms) -> tuple[float, float, float]:
        def dz(v: float) -> float:
            return v if abs(v) > self.stick_deadzone else 0.0

        left = transforms["left_thumbstick"]
        right = transforms["right_thumbstick"]
        # Quest stick y is negative when pushed up -> negate so forward is positive.
        vx = -dz(right[1]) * self.stick_max_vx   # right stick Y  -> forward/back
        vy = -dz(right[0]) * self.stick_max_vy   # right stick X  -> strafe left/right
        wz = -dz(left[0]) * self.stick_max_wz    # left stick X   -> yaw (left stick Y unused)
        chassis = mask_planar_twist_for_base_dofs(
            np.array([vx, vy, wz], dtype=float),
            base_dofs=self._joystick_base_dofs,
            allow_yaw_hold=False,
        )
        if self._joystick_single_axis:
            chassis, self._joystick_axis = project_planar_twist_single_axis_for_base_dofs(
                chassis,
                base_dofs=self._joystick_base_dofs,
                allow_yaw_hold=False,
                xy_max_vel=self._joystick_xy_max_vel,
                yaw_max_vel=self._joystick_yaw_max_vel,
                deadband=self._joystick_axis_deadband,
                hysteresis_ratio=self._joystick_axis_hysteresis,
                prev_axis=self._joystick_axis,
            )
        else:
            self._joystick_axis = None
        vx, vy, wz = (float(v) for v in chassis)
        # DIAGNOSTIC (temporary): print the RAW sticks the leader receives from WebXR and the
        # mapped chassis, ~2 Hz. If the sticks read [0, 0] while you push them, the data is not
        # reaching the leader (browser/reader); if they are non-zero but chassis is 0, it is the
        # deadzone/mapping. Remove once the base drives.
        now = time.perf_counter()
        if now - getattr(self, "_dbg_stick_t", 0.0) > 0.5:
            self._dbg_stick_t = now
            print(f"\n[joystick dbg] L_stick={[round(float(x), 3) for x in left]} "
                  f"R_stick={[round(float(x), 3) for x in right]} "
                  f"click(L,R)=({transforms.get('left_thumbstick_click')},"
                  f"{transforms.get('right_thumbstick_click')}) "
                  f"-> vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f}")
        return vx, vy, wz


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--namespace", default="",
                        help="Zenoh namespace (must match the follower; default empty).")
    parser.add_argument("--host", default="0.0.0.0", help="WebXR bind host (default 0.0.0.0).")
    parser.add_argument("--port", type=int, default=5067, help="WebXR port (default 5067).")
    parser.add_argument("--cert", default=DEFAULT_CERT, help="TLS cert for the WebXR server.")
    parser.add_argument("--key", default=DEFAULT_KEY, help="TLS key for the WebXR server.")
    parser.add_argument("--config", default=None,
                        help="wbik.yaml path for publish rate, shared thumbstick mapping, and "
                             "base single-axis policy. Default: canonical follower/wbik.yaml.")
    parser.add_argument("--hold-seconds", type=float, default=1.0,
                        help="right-grip hold time to calibrate/start (default 1.0).")
    parser.add_argument("--headset-hud", action=argparse.BooleanOptionalAction, default=True,
                        help="show the robot camera/status HUD in the Quest (default: enabled).")
    parser.add_argument("--hud-rate", type=float, default=DEFAULT_HEADSET_HUD_RATE,
                        help=f"headset HUD refresh rate in Hz (default {DEFAULT_HEADSET_HUD_RATE:g}).")
    parser.add_argument("--hud-cameras", nargs="+", choices=(
        "head_left_rgb", "head_right_rgb", "head_depth", "left_wrist_rgb", "right_wrist_rgb"
    ), default=list(_DEFAULT_HUD_CAMERAS),
                        help="camera streams to read for the headset HUD.")
    parser.add_argument("--align-reference", default=None,
                        help="optional raw-data episode HDF5 defining the reference L/R EEF pose "
                             "for the post-calibration align gate (see wbc_vr_leader). Omit to "
                             "disable.")
    parser.add_argument("--ee-offset-file", default=DEFAULT_EE_OFFSET_FILE,
                        help="YAML of per-side controller->gripper orientation offsets "
                             f"(scipy [x,y,z,w] quats). Default: {DEFAULT_EE_OFFSET_FILE}.")
    parser.add_argument("--calibrate-ee-offset", action="store_true",
                        help="one-time helper: capture the neutral-grip offset C and exit "
                             "(same as wbc_vr_leader.py --calibrate-ee-offset).")
    parser.add_argument("--ee-capture-seconds", type=float, default=1.5,
                        help="averaging window for --calibrate-ee-offset (default 1.5).")
    parser.add_argument("--debug-vr", dest="debug_vr", default=None,
                        help="write a per-tick RAW-VR diagnostic HDF5 under this directory.")
    args = parser.parse_args()

    # Publish rate = follower cmd_rate from wbik.yaml (leader/follower cannot drift).
    cfg = WBCConfig.from_yaml(args.config)
    vt = VRTeleopConfig.from_yaml(args.config)
    joystick_cfg = JoystickTeleopConfig.from_yaml(args.config)
    args.wbc_config = cfg
    args.joystick_config = joystick_cfg
    args.vr_teleop_config = vt
    args.rate = vt.cmd_rate
    if args.rate <= 0:
        raise ValueError(f"wbik.yaml vr_teleop.cmd_rate must be > 0, got {args.rate}")
    if args.headset_hud and (not np.isfinite(args.hud_rate) or args.hud_rate <= 0.0):
        parser.error("--hud-rate must be finite and > 0 when --headset-hud is enabled")

    leader = JoystickVRLeader(args)
    if args.calibrate_ee_offset:
        leader.calibrate_ee_offset(args.ee_capture_seconds, args.ee_offset_file)
    else:
        leader.run()


if __name__ == "__main__":
    main()
