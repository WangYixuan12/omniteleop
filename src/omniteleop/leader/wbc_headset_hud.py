"""Quest headset HUD helpers for WBC VR teleop.

This module owns the leader-side visualization surface: robot camera tiles, follower
status text, and reference-pose alignment prompts. The alignment gate itself belongs in
the leader/recorder flow; this module only formats and draws the operator feedback.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from omniteleop.common.schemas import WBCFollowerStatus

CameraStream = str
_DEFAULT_HUD_CAMERAS: list[CameraStream] = ["head_left_rgb", "left_wrist_rgb"]
_HEAD_STREAM_TO_OBS_KEY: dict[CameraStream, str] = {
    "head_left_rgb": "left_rgb",
    "head_right_rgb": "right_rgb",
    "head_depth": "depth",
}
_WRIST_STREAM_TO_OBS_KEY: dict[CameraStream, str] = {
    "left_wrist_rgb": "left_rgb",
    "right_wrist_rgb": "right_rgb",
}
_WRIST_SENSOR_ID = "wrist_zedm"

_WHITE = (255, 255, 255)
_RED = (0, 0, 255)
_YELLOW = (0, 255, 255)
_GREEN = (0, 255, 0)


@dataclass(frozen=True)
class HUDOverlayLine:
    """One rendered headset HUD line."""

    text: str
    color: tuple[int, int, int]


@dataclass(frozen=True)
class HandAlignmentStatus:
    """Current hand-vs-reference pose error for pre-record alignment prompts."""

    left_delta_mm: np.ndarray
    right_delta_mm: np.ndarray
    left_rot_deg: float
    right_rot_deg: float
    pos_tolerance_mm: float
    rot_tolerance_deg: float
    stable_s: float = 0.0
    stable_required_s: float = 0.0

    @property
    def left_ok(self) -> bool:
        """Whether the left hand is within the configured alignment tolerance."""
        return _hand_within_tolerance(
            self.left_delta_mm, self.left_rot_deg, self.pos_tolerance_mm, self.rot_tolerance_deg
        )

    @property
    def right_ok(self) -> bool:
        """Whether the right hand is within the configured alignment tolerance."""
        return _hand_within_tolerance(
            self.right_delta_mm, self.right_rot_deg, self.pos_tolerance_mm, self.rot_tolerance_deg
        )

    @property
    def ready(self) -> bool:
        """Whether both hands stayed aligned for the required stable window."""
        return self.left_ok and self.right_ok and self.stable_s >= self.stable_required_s


def follower_status_overlay_lines(
    status: Optional[WBCFollowerStatus],
    status_age_s: Optional[float],
    *,
    stale_after_s: float = 1.0,
) -> list[str]:
    """Human-readable follower lines for the headset HUD."""
    if status is None:
        return ["Follower: no status"]

    lines = []
    if status.hold_reason:
        lines.append(f"Follower: HOLD:{status.hold_reason}")
    elif status.estop or status.hold:
        lines.append(f"Follower: hold({status.stage})")
    if status_age_s is not None and status_age_s > stale_after_s:
        lines.append(f"WARNING: follower status stale {status_age_s:.1f}s")
    if not status.success:
        lines.append("WARNING: IK solve failed")
    safety = str(status.safety_status or "")
    if safety and safety.lower() != "ok":
        lines.append(safety)
    lines.append(f"err L/R={status.left_ee_error_mm:.0f}/{status.right_ee_error_mm:.0f}mm")
    return lines


def follower_status_overlay_color(line: str) -> tuple[int, int, int]:
    """BGR color for a follower HUD line."""
    if line.startswith(("WARNING", "WARN:", "HELD:")):
        return _RED
    return _WHITE


def collision_banner(status: Optional[WBCFollowerStatus]) -> Optional[HUDOverlayLine]:
    """Prominent self-collision banner (text + color), or None when clear.

    Reads the follower's already-published ``safety_status`` -- no new data path and
    no added latency: the follower keys "self-collision" into that string each IK tick
    (wbc_safety), the leader forwards the status here, and we only decide how to draw
    it. Red + "HOLD" once the step was frozen (``held`` / a ``HELD:`` reason); yellow
    while the hands are merely approaching the collision floor (``WARN:``).
    """
    if status is None:
        return None
    reason = str(status.safety_status or "")
    if "collision" not in reason.lower():
        return None
    if status.held or reason.startswith("HELD"):
        return HUDOverlayLine("SELF-COLLISION - HOLD", _RED)
    return HUDOverlayLine("SELF-COLLISION NEAR", _YELLOW)


def _hand_within_tolerance(
    delta_mm: np.ndarray,
    rot_deg: float,
    pos_tolerance_mm: float,
    rot_tolerance_deg: float,
) -> bool:
    delta = np.asarray(delta_mm, dtype=float)
    return (
        delta.shape == (3,)
        and np.all(np.isfinite(delta))
        and np.isfinite(rot_deg)
        and float(np.linalg.norm(delta)) <= pos_tolerance_mm
        and abs(float(rot_deg)) <= rot_tolerance_deg
    )


def _hand_alignment_label(
    delta_mm: np.ndarray,
    rot_deg: float,
    pos_tolerance_mm: float,
    rot_tolerance_deg: float,
) -> str:
    delta = np.asarray(delta_mm, dtype=float)
    flags = []
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        flags.append("POS")
    elif float(np.linalg.norm(delta)) > pos_tolerance_mm:
        flags.append("POS")
    if not np.isfinite(rot_deg) or abs(float(rot_deg)) > rot_tolerance_deg:
        flags.append("ROT")
    return "OK" if not flags else "+".join(flags)


def _hand_alignment_line(
    side: str,
    delta_mm: np.ndarray,
    rot_deg: float,
    pos_tolerance_mm: float,
    rot_tolerance_deg: float,
) -> HUDOverlayLine:
    delta = np.asarray(delta_mm, dtype=float)
    if delta.shape == (3,) and np.all(np.isfinite(delta)):
        vec = " ".join(f"{v:+.0f}" for v in delta)
    else:
        vec = "nan nan nan"
    label = _hand_alignment_label(delta, rot_deg, pos_tolerance_mm, rot_tolerance_deg)
    color = _GREEN if label == "OK" else _YELLOW
    return HUDOverlayLine(
        f"{side} dxyz=[{vec}]mm rot={rot_deg:.0f}deg {label}",
        color,
    )


def alignment_overlay_lines(alignment: Optional[HandAlignmentStatus]) -> list[HUDOverlayLine]:
    """HUD lines that prompt the teleoperator to match a reference hand pose."""
    if alignment is None:
        return []

    lines = [
        HUDOverlayLine("Align ref: move hands to reference", _YELLOW),
        _hand_alignment_line(
            "L",
            alignment.left_delta_mm,
            alignment.left_rot_deg,
            alignment.pos_tolerance_mm,
            alignment.rot_tolerance_deg,
        ),
        _hand_alignment_line(
            "R",
            alignment.right_delta_mm,
            alignment.right_rot_deg,
            alignment.pos_tolerance_mm,
            alignment.rot_tolerance_deg,
        ),
    ]
    if alignment.ready:
        lines.append(HUDOverlayLine("Gate: ready", _GREEN))
    elif alignment.stable_required_s > 0.0:
        lines.append(
            HUDOverlayLine(
                f"Gate: align {alignment.stable_s:.1f}/{alignment.stable_required_s:.1f}s",
                _YELLOW,
            )
        )
    else:
        lines.append(HUDOverlayLine("Gate: align", _YELLOW))
    return lines


class WBCHeadsetHUD:
    """Poll robot camera streams and push a composed HUD frame to the Quest browser."""

    def __init__(self, quest, cameras: list[CameraStream]) -> None:
        self.quest = quest
        self.cameras = list(dict.fromkeys(cameras or _DEFAULT_HUD_CAMERAS))
        self._head_keys = [
            _HEAD_STREAM_TO_OBS_KEY[c] for c in self.cameras if c in _HEAD_STREAM_TO_OBS_KEY
        ]
        self._wrist_keys = [
            _WRIST_STREAM_TO_OBS_KEY[c] for c in self.cameras if c in _WRIST_STREAM_TO_OBS_KEY
        ]
        self._last_head_imgs: dict[str, np.ndarray] = {}
        self._last_wrist_imgs: dict[str, np.ndarray] = {}
        self._last_camera_warn_t = 0.0
        self._cam_robot = None

        try:
            from dexbot_utils.configs.components.sensors.cameras import (  # noqa: PLC0415
                ZedXCameraConfig,
            )
            from dexcontrol.core.config import get_robot_config  # noqa: PLC0415
            from dexcontrol.robot import Robot as _Robot  # noqa: PLC0415

            configs = get_robot_config()
            if self._head_keys:
                if "head_camera" in configs.sensors:
                    configs.sensors["head_camera"].enabled = True
                else:
                    print(
                        "[wbc_vr_leader] WARNING: headset HUD requested head_camera "
                        "but the robot config has no head_camera sensor."
                    )
                    self._head_keys = []

            if self._wrist_keys and _WRIST_SENSOR_ID not in configs.sensors:
                configs.sensors[_WRIST_SENSOR_ID] = ZedXCameraConfig(
                    name=_WRIST_SENSOR_ID,
                    enable_rgb=True,
                    enable_depth=False,
                )
            if _WRIST_SENSOR_ID in configs.sensors:
                configs.sensors[_WRIST_SENSOR_ID].enabled = bool(self._wrist_keys)

            if self._head_keys or self._wrist_keys:
                self._cam_robot = _Robot(configs=configs)
                print(f"[wbc_vr_leader] headset HUD cameras={self.cameras}")
        except Exception as exc:  # pragma: no cover - hardware/config dependent
            print(f"[wbc_vr_leader] WARNING: headset HUD cameras unavailable: {exc}")
            self._head_keys = []
            self._wrist_keys = []
            self._cam_robot = None

    def _warn_camera_poll(self, exc: Exception) -> None:
        now = time.monotonic()
        if now - self._last_camera_warn_t < 5.0:
            return
        self._last_camera_warn_t = now
        print(f"[wbc_vr_leader] WARNING: headset HUD camera poll failed: {exc}")

    def _poll_cameras(self) -> None:
        if self._cam_robot is None:
            return
        try:
            if self._head_keys:
                obs = self._cam_robot.sensors.head_camera.get_obs(obs_keys=self._head_keys)
                for key in self._head_keys:
                    frame = obs.get(key)
                    if frame is not None:
                        self._last_head_imgs[key] = np.asarray(frame)
            if self._wrist_keys and hasattr(self._cam_robot.sensors, _WRIST_SENSOR_ID):
                wrist = getattr(self._cam_robot.sensors, _WRIST_SENSOR_ID)
                obs = wrist.get_obs(obs_keys=self._wrist_keys)
                for key in self._wrist_keys:
                    frame = obs.get(key)
                    if frame is None:
                        for obs_key, obs_val in obs.items():
                            if str(obs_key).endswith(key):
                                frame = obs_val
                                break
                    if frame is not None:
                        self._last_wrist_imgs[key] = np.asarray(frame)
        except Exception as exc:  # pragma: no cover - hardware dependent
            self._warn_camera_poll(exc)

    def _render_tile(self, stream: CameraStream) -> Optional[np.ndarray]:
        if stream in _HEAD_STREAM_TO_OBS_KEY:
            key = _HEAD_STREAM_TO_OBS_KEY[stream]
            img = self._last_head_imgs.get(key)
        else:
            key = _WRIST_STREAM_TO_OBS_KEY[stream]
            img = self._last_wrist_imgs.get(key)
        if img is None:
            return None

        if key == "depth":
            finite = img[np.isfinite(img) & (img > 0)]
            if len(finite) == 0:
                normalized = np.zeros(img.shape[:2], dtype=np.uint8)
            else:
                mn, mx = finite.min(), np.percentile(finite, 95)
                normalized = np.clip((img - mn) / (mx - mn + 1e-6) * 255, 0, 255).astype(
                    np.uint8
                )
            return cv2.resize(cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO), (320, 180))

        if img.ndim != 3 or img.shape[2] != 3:
            return None
        return cv2.resize(img[:, :, ::-1], (320, 180))

    @staticmethod
    def _draw_line(img: np.ndarray, text: str, y: int, color: tuple[int, int, int]) -> None:
        if len(text) > 108:
            text = text[:105] + "..."
        cv2.putText(
            img,
            text,
            (8, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            thickness=1,
            color=color,
        )

    @staticmethod
    def _draw_banner(img: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
        """Draw a large, width-fitted alert banner across the bottom of the HUD frame."""
        h, w = img.shape[:2]
        thickness = 2
        scale = 1.1
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if tw > w - 8:  # shrink to fit narrow single-tile frames
            scale *= (w - 8) / tw
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x = max((w - tw) // 2, 4)
        y = h - 10
        cv2.rectangle(img, (0, y - th - 8), (w, y + base + 4), (0, 0, 0), thickness=-1)
        cv2.putText(
            img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness
        )

    def poll_and_send(
        self,
        *,
        stage: str,
        status: Optional[WBCFollowerStatus],
        status_age_s: Optional[float],
        alignment: Optional[HandAlignmentStatus] = None,
    ) -> None:
        """Refresh camera caches, compose the HUD frame, and send it to WebXR."""
        self._poll_cameras()
        tiles: list[np.ndarray] = []
        for stream in self.cameras:
            if stream == "head_depth":
                continue
            tile = self._render_tile(stream)
            if tile is not None:
                tiles.append(tile)
        vis_img = np.concatenate(tiles, axis=1) if tiles else np.zeros((180, 320, 3), np.uint8)

        y = 22
        self._draw_line(vis_img, f"Stage: {stage}", y, _YELLOW)
        y += 22
        for line in follower_status_overlay_lines(status, status_age_s):
            self._draw_line(vis_img, line, y, follower_status_overlay_color(line))
            y += 18
        for line in alignment_overlay_lines(alignment):
            self._draw_line(vis_img, line.text, y, line.color)
            y += 18

        # Prominent self-collision alert, drawn last so it sits on top. Uses the
        # follower status already polled above -- no extra data path, no added latency.
        banner = collision_banner(status)
        if banner is not None:
            self._draw_banner(vis_img, banner.text, banner.color)

        ok, buf = cv2.imencode(".jpg", vis_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if ok:
            self.quest.set_frame_vis("img", base64.b64encode(buf).decode())
