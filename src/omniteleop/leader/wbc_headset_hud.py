"""Quest headset HUD helpers for WBC VR teleop.

This module owns the leader-side visualization surface: robot camera tiles, follower
status text, and reference-pose alignment prompts. The alignment gate itself belongs in
the leader/recorder flow; this module only formats and draws the operator feedback.
"""

from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from omniteleop.common.schemas import WBC_FOLLOWER_STAGE_ABORTED, WBCFollowerStatus

CameraStream = str
# Operator-facing tile order: left wrist | head | right wrist, so the HUD reads like
# the robot's own point of view (each hand on its own side of the head view).
_DEFAULT_HUD_CAMERAS: list[CameraStream] = [
    "left_wrist_rgb",
    "head_left_rgb",
    "right_wrist_rgb",
]
_HEAD_STREAM_TO_OBS_KEY: dict[CameraStream, str] = {
    "head_left_rgb": "left_rgb",
    "head_right_rgb": "right_rgb",
    "head_depth": "depth",
}
# Each wrist is its OWN ZED-M with its own publisher, and "left"/"right" name the ARM
# it is mounted on -- matching wbc_vr_robot._WRIST_SENSOR_IDS. (These two stream names
# used to mean the two stereo EYES of a single shared wrist camera.)
_WRIST_STREAM_TO_SENSOR_ID: dict[CameraStream, str] = {
    "left_wrist_rgb": "left_wrist_zedm",
    "right_wrist_rgb": "right_wrist_zedm",
}
# Every wrist publisher sends only its stereo left eye (tests/test_wrist_zedm_depth.py
# defaults to --skip-right-rgb), so this is the obs key for BOTH wrist sensors.
_WRIST_OBS_KEY = "left_rgb"

_WHITE = (255, 255, 255)
_RED = (0, 0, 255)
_YELLOW = (0, 255, 255)
_GREEN = (0, 255, 0)

# Camera-tile geometry for the composed HUD frame. Tiles are concatenated horizontally
# in ``self.cameras`` order, so the default three-camera layout gives a 3:1-wide frame
# (left wrist | head | right wrist); ``camPanelH`` in web/vr_client.html sizes the panel
# in the headset and its width auto-follows this aspect. The published RGB/depth streams
# are 4:3 (head SVGA 960x600 -> 640x480; wrist HD720 -> 640x480), so each tile is
# rendered at 4:3 (_TILE_W:_TILE_H) to MATCH the source and not stretch it -- a 16:9 tile
# squashed the 4:3 frame horizontally. At _HUD_SCALE=2 the tile equals the published 640x480 (1:1,
# no resample); _HUD_SCALE=1 downsamples to 320x240, cutting pixel count (and JPEG
# encode/transfer cost) to a quarter -- traded for latency, per the README note.
# Raising _HUD_SCALE upsamples + scales the text overlay together. The APPARENT size of
# the panel in the headset is set separately by ``camPanelH`` in ``web/vr_client.html``
# (its width auto-follows the streamed aspect); this constant only controls the
# streamed image resolution.
_HUD_SCALE = 1.25
_TILE_W = round(320 * _HUD_SCALE)
_TILE_H = round(240 * _HUD_SCALE)

# Status/alignment text is drawn on the MIDDLE tile (the head view in the default
# left-wrist | head | right-wrist layout) rather than across the whole montage from the
# left edge, so it annotates the view the operator is actually looking at instead of
# covering the left wrist camera. Confining it to one tile shrinks the text budget from
# the full montage width to a single _TILE_W, so the font is reduced to match: at the
# previous 0.4 * _HUD_SCALE a long alignment line measures 427px and would overflow a
# 400px tile into the right wrist view; 0.32 brings that to 342px. Lines are additionally
# truncated to _OVERLAY_MAX_W by _fit_text, which is what actually guarantees containment
# for unbounded strings (e.g. a follower ABORTED reason).
_OVERLAY_FONT_SCALE = 0.32 * _HUD_SCALE
_OVERLAY_PAD = round(8 * _HUD_SCALE)
_OVERLAY_MAX_W = _TILE_W - 2 * _OVERLAY_PAD


@dataclass(frozen=True)
class HUDOverlayLine:
    """One rendered headset HUD line."""

    text: str
    color: tuple[int, int, int]


@dataclass(frozen=True)
class HandAlignmentStatus:
    """Current hand-vs-reference pose error for pre-record alignment prompts.

    When ``head_delta_mm`` is set the operator is ALSO gated on returning the robot's
    head to its nominal pose (so a recorded episode always starts from the same head
    look direction as the reference). ``head_delta_mm is None`` means the head is not
    gated -- ``head_ok`` is then vacuously True and no head line is drawn.
    """

    left_delta_mm: np.ndarray
    right_delta_mm: np.ndarray
    left_rot_deg: float
    right_rot_deg: float
    pos_tolerance_mm: float
    rot_tolerance_deg: float
    stable_s: float = 0.0
    stable_required_s: float = 0.0
    head_delta_mm: Optional[np.ndarray] = None
    head_rot_deg: Optional[float] = None
    head_pos_tolerance_mm: float = 0.0
    head_rot_tolerance_deg: float = 0.0

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
    def head_gated(self) -> bool:
        """Whether this status gates the head pose against nominal."""
        return self.head_delta_mm is not None

    @property
    def head_ok(self) -> bool:
        """Whether the head is at nominal within tolerance (True when not gated)."""
        if self.head_delta_mm is None:
            return True
        return _hand_within_tolerance(
            self.head_delta_mm,
            self.head_rot_deg,
            self.head_pos_tolerance_mm,
            self.head_rot_tolerance_deg,
        )

    @property
    def ready(self) -> bool:
        """Whether hands (and head, if gated) stayed aligned for the stable window."""
        return (
            self.left_ok
            and self.right_ok
            and self.head_ok
            and self.stable_s >= self.stable_required_s
        )


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
    # Which episode_<N>.hdf5 the current/next take saves as. Comes from the FOLLOWER's
    # recorder (the leader has no recorder of its own), and is -1 when the follower runs
    # without --record -- then there is no file to name, so the line is dropped.
    if status.episode_id >= 0:
        lines.append(f"Episode: {status.episode_id}")
    if status.scenediff_enabled:
        if status.scenediff_state == "ready":
            lines.append("Scan saved. Release + hold RIGHT grip.")
        elif not status.scenediff_state:
            lines.append("Hold RIGHT grip: park arms + scan.")
    if status.stage == WBC_FOLLOWER_STAGE_ABORTED:
        # Terminal frame: the follower process died. hold_reason carries the exception's
        # first line; this status never refreshes, so the line (and abort_banner) persist.
        reason = f": {status.hold_reason}" if status.hold_reason else ""
        lines.append(f"Follower: ABORTED{reason}")
    elif status.hold_reason:
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
    if line.startswith(("WARNING", "WARN:", "HELD:", "Follower: ABORTED")):
        return _RED
    return _WHITE


def abort_banner(status: Optional[WBCFollowerStatus]) -> Optional[HUDOverlayLine]:
    """Prominent banner when the follower published its terminal ABORTED frame.

    The frame is sent once from the follower's exception path (wbc_vr_robot), so the
    leader keeps re-rendering the last received status and the banner persists until
    the follower is restarted and publishes a live status again.
    """
    if status is None or status.stage != WBC_FOLLOWER_STAGE_ABORTED:
        return None
    return HUDOverlayLine("FOLLOWER ABORTED - CHECK TERMINAL", _RED)


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

    prompt = (
        "Align ref: move hands+head to ref"
        if alignment.head_gated
        else "Align ref: move hands to reference"
    )
    lines = [
        HUDOverlayLine(prompt, _YELLOW),
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
    if alignment.head_gated:
        lines.append(
            _hand_alignment_line(
                "H",
                alignment.head_delta_mm,
                alignment.head_rot_deg,
                alignment.head_pos_tolerance_mm,
                alignment.head_rot_tolerance_deg,
            )
        )
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

    def __init__(self, quest, cameras: list[CameraStream], *, sim_namespace=None) -> None:
        self.quest = quest
        self.cameras = list(dict.fromkeys(cameras or _DEFAULT_HUD_CAMERAS))
        self._head_keys = [
            _HEAD_STREAM_TO_OBS_KEY[c] for c in self.cameras if c in _HEAD_STREAM_TO_OBS_KEY
        ]
        # Wrist streams are tracked by STREAM NAME (one sensor each), not by obs key --
        # both wrist sensors publish under the same obs key (_WRIST_OBS_KEY).
        self._wrist_streams = [c for c in self.cameras if c in _WRIST_STREAM_TO_SENSOR_ID]
        self._last_head_imgs: dict[str, np.ndarray] = {}
        self._last_wrist_imgs: dict[CameraStream, np.ndarray] = {}
        self._last_camera_warn_t = 0.0
        self._cam_robot = None

        # Background render loop state. poll_and_send (camera get_obs + JPEG encode) runs on
        # this thread so it NEVER blocks the caller's teleop command loop; the control loop
        # only pushes cheap overlay state via update_state(). start()/stop() manage the
        # thread (see wbc_vr_leader.run()).
        self._state_lock = threading.Lock()
        self._latest_state: Optional[dict] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._period = float("inf")

        self._sim_node = None
        self._sim_subscribers = []
        if sim_namespace is not None:
            from dexcomm import Node
            from dexcomm.codecs import RGBImageCodec

            self._sim_node = Node(name="behavior_headset_cameras", namespace=sim_namespace)
            for stream in self.cameras:
                if stream == "head_left_rgb":
                    sensor, cache, key = "head_camera", self._last_head_imgs, "left_rgb"
                elif stream in _WRIST_STREAM_TO_SENSOR_ID:
                    sensor, cache, key = _WRIST_STREAM_TO_SENSOR_ID[stream], self._last_wrist_imgs, stream
                else:
                    raise ValueError(f"BEHAVIOR HUD does not publish {stream}")

                def receive(data, cache=cache, key=key):
                    cache[key] = np.asarray(data["data"])

                self._sim_subscribers.append(self._sim_node.create_subscriber(
                    f"sensors/{sensor}/left_rgb", receive, decoder=RGBImageCodec.decode))
            return

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

            # Enable exactly the wrist sensors this HUD asked for; each needs its own
            # ZED-SDK publisher running on sensors/<sensor_id>/*.
            wanted_wrist_ids = {
                _WRIST_STREAM_TO_SENSOR_ID[c] for c in self._wrist_streams
            }
            for sensor_id in _WRIST_STREAM_TO_SENSOR_ID.values():
                if sensor_id in wanted_wrist_ids and sensor_id not in configs.sensors:
                    configs.sensors[sensor_id] = ZedXCameraConfig(
                        name=sensor_id,
                        enable_rgb=True,
                        enable_depth=False,
                    )
                if sensor_id in configs.sensors:
                    configs.sensors[sensor_id].enabled = sensor_id in wanted_wrist_ids

            if self._head_keys or self._wrist_streams:
                self._cam_robot = _Robot(configs=configs)
                print(f"[wbc_vr_leader] headset HUD cameras={self.cameras}")
        except Exception as exc:  # pragma: no cover - hardware/config dependent
            # A failed Robot is disposed by its own __del__, which calls
            # Robot.shutdown() -> dexcomm.cleanup_session() and closes the
            # PROCESS-WIDE Zenoh session. Callers must therefore create their
            # publishers AFTER constructing the HUD, or they silently end up holding a
            # closed publisher (wbc_vr_leader does exactly this -- see its __init__).
            print(
                f"[wbc_vr_leader] WARNING: headset HUD cameras unavailable: {exc}\n"
                "[wbc_vr_leader] WARNING: teleop continues WITHOUT the headset HUD. "
                "Note this also tore down the shared Zenoh session; publishers created "
                "before now are dead."
            )
            self._head_keys = []
            self._wrist_streams = []
            self._cam_robot = None

    def _warn_camera_poll(self, exc: Exception) -> None:
        now = time.monotonic()
        if now - self._last_camera_warn_t < 5.0:
            return
        self._last_camera_warn_t = now
        print(f"[wbc_vr_leader] WARNING: headset HUD camera poll failed: {exc}")

    # -- background render loop (keeps the HUD off the teleop control path) ------

    def start(self, rate: float) -> None:
        """Start the background HUD loop at ``rate`` Hz (idempotent).

        The heavy work -- camera ``get_obs`` (a blocking network read) plus ``cv2`` resize
        and JPEG encode -- runs on this thread, OFF the caller's teleop command loop, so the
        HUD adds no latency to teleoperation. The control loop feeds overlay state cheaply
        via :meth:`update_state`.
        """
        if self._thread is not None:
            return
        self._period = 1.0 / rate if rate and rate > 0.0 else float("inf")
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, name="wbc_headset_hud", daemon=True
        )
        self._thread.start()

    def update_state(
        self,
        *,
        stage: str,
        status: Optional[WBCFollowerStatus],
        status_age_s: Optional[float],
        alignment: Optional[HandAlignmentStatus] = None,
    ) -> None:
        """Store the latest overlay state for the background loop (control-loop safe).

        Only takes a lock and stores references -- no camera or JPEG work -- so it costs
        microseconds and never stalls the command loop that calls it.
        """
        with self._state_lock:
            self._latest_state = {
                "stage": stage,
                "status": status,
                "status_age_s": status_age_s,
                "alignment": alignment,
            }

    def _run_loop(self) -> None:
        while self._running:
            t0 = time.perf_counter()
            with self._state_lock:
                state = self._latest_state
            if state is not None:
                self.poll_and_send(**state)
            sleep = self._period - (time.perf_counter() - t0)
            if sleep > 0.0:
                time.sleep(sleep)

    def stop(self) -> None:
        """Stop the background loop and join the thread (idempotent)."""
        self._running = False
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

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
            # One get_obs per wrist camera; a missing sensor is skipped so a single
            # down publisher never blocks the other wrist or the head tile.
            for stream in self._wrist_streams:
                sensor_id = _WRIST_STREAM_TO_SENSOR_ID[stream]
                if not hasattr(self._cam_robot.sensors, sensor_id):
                    continue
                obs = getattr(self._cam_robot.sensors, sensor_id).get_obs(
                    obs_keys=[_WRIST_OBS_KEY]
                )
                frame = obs.get(_WRIST_OBS_KEY)
                if frame is None:  # tolerate a sensor-name-prefixed key
                    for obs_key, obs_val in obs.items():
                        if str(obs_key).endswith(_WRIST_OBS_KEY):
                            frame = obs_val
                            break
                if frame is not None:
                    self._last_wrist_imgs[stream] = np.asarray(frame)
        except Exception as exc:  # pragma: no cover - hardware dependent
            self._warn_camera_poll(exc)

    def _render_tile(self, stream: CameraStream) -> Optional[np.ndarray]:
        if stream in _HEAD_STREAM_TO_OBS_KEY:
            key = _HEAD_STREAM_TO_OBS_KEY[stream]
            img = self._last_head_imgs.get(key)
        else:
            key = _WRIST_OBS_KEY
            img = self._last_wrist_imgs.get(stream)
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
            return cv2.resize(
                cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO), (_TILE_W, _TILE_H)
            )

        if img.ndim != 3 or img.shape[2] != 3:
            return None
        return cv2.resize(img[:, :, ::-1], (_TILE_W, _TILE_H))

    @staticmethod
    def _placeholder_tile(stream: CameraStream) -> np.ndarray:
        """Labeled dark tile for a configured stream with no frame yet.

        Keeps tile POSITIONS fixed: with the default left-wrist | head | right-wrist
        layout, dropping an unfilled tile would slide the head view sideways and leave
        the operator unsure which wrist they are looking at.
        """
        tile = np.zeros((_TILE_H, _TILE_W, 3), np.uint8)
        text = f"{stream}: no frame"
        scale = 0.45 * _HUD_SCALE
        thickness = max(1, round(_HUD_SCALE))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.putText(
            tile,
            text,
            (max((_TILE_W - tw) // 2, 2), (_TILE_H + th) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            _YELLOW,
            thickness,
        )
        return tile

    @staticmethod
    def _fit_text(text: str, thickness: int) -> str:
        """Truncate ``text`` with an ellipsis until it measures within one tile.

        Replaces the previous fixed 108-character cap, which was tuned for text spanning
        the FULL montage width. Now that the overlay lives on a single tile the budget is
        both narrower and font-dependent, so measure instead of guessing -- a character
        count cannot bound proportional-font strings such as a follower abort reason.
        """
        def width(s: str) -> int:
            return cv2.getTextSize(
                s, cv2.FONT_HERSHEY_SIMPLEX, _OVERLAY_FONT_SCALE, thickness
            )[0][0]

        if width(text) <= _OVERLAY_MAX_W:
            return text
        # Binary-search the longest prefix that still fits once the ellipsis is added.
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if width(text[:mid] + "...") <= _OVERLAY_MAX_W:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo] + "..."

    @staticmethod
    def _draw_line(
        img: np.ndarray, text: str, y: int, color: tuple[int, int, int], x0: int = 0
    ) -> None:
        """Draw one overlay line, left-aligned at ``x0`` (the middle tile's origin)."""
        thickness = max(1, round(_HUD_SCALE))
        cv2.putText(
            img,
            WBCHeadsetHUD._fit_text(text, thickness),
            (x0 + _OVERLAY_PAD, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=_OVERLAY_FONT_SCALE,
            thickness=thickness,
            color=color,
        )

    @staticmethod
    def _draw_banner(img: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
        """Draw a large, width-fitted alert banner across the bottom of the HUD frame."""
        h, w = img.shape[:2]
        thickness = max(1, round(2 * _HUD_SCALE))
        scale = 1.1 * _HUD_SCALE
        pad = round(8 * _HUD_SCALE)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if tw > w - pad:  # shrink to fit narrow single-tile frames
            scale *= (w - pad) / tw
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x = max((w - tw) // 2, round(4 * _HUD_SCALE))
        y = h - round(10 * _HUD_SCALE)
        cv2.rectangle(img, (0, y - th - pad), (w, y + base + round(4 * _HUD_SCALE)), (0, 0, 0), thickness=-1)
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
            tiles.append(tile if tile is not None else self._placeholder_tile(stream))
        vis_img = (
            np.concatenate(tiles, axis=1)
            if tiles
            else np.zeros((_TILE_H, _TILE_W, 3), np.uint8)
        )

        # Anchor the overlay on the MIDDLE tile -- the head view in the default
        # left-wrist | head | right-wrist layout -- so the status annotates what the
        # operator is looking at instead of covering the left wrist camera. Derived from
        # the rendered tile count (head_depth is skipped above), so it still lands on the
        # centre tile for any --hud-cameras selection, and on the only tile when there is
        # one. Guarded for the empty-tiles placeholder frame.
        x0 = (len(tiles) // 2) * _TILE_W if tiles else 0

        y = round(22 * _HUD_SCALE)
        self._draw_line(vis_img, f"Stage: {stage}", y, _YELLOW, x0)
        y += round(22 * _HUD_SCALE)
        for line in follower_status_overlay_lines(status, status_age_s):
            self._draw_line(vis_img, line, y, follower_status_overlay_color(line), x0)
            y += round(18 * _HUD_SCALE)
        for line in alignment_overlay_lines(alignment):
            self._draw_line(vis_img, line.text, y, line.color, x0)
            y += round(18 * _HUD_SCALE)

        # Prominent alert, drawn last so it sits on top. Uses the follower status
        # already polled above -- no extra data path, no added latency. A follower
        # abort (terminal, process dead) outranks a live self-collision warning.
        banner = abort_banner(status) or collision_banner(status)
        if banner is not None:
            self._draw_banner(vis_img, banner.text, banner.color)

        ok, buf = cv2.imencode(".jpg", vis_img, [cv2.IMWRITE_JPEG_QUALITY, 40])
        if ok:
            self.quest.set_frame_vis("img", base64.b64encode(buf).decode())
