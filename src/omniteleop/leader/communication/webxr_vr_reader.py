#!/usr/bin/env python3
"""WebXR VR pose reader using Socket.IO — no Unity APK required.

Runs a Socket.IO server (via aiohttp) that:
- Serves vr_client.html at GET /
- Receives pose frames from the Quest 3 browser (WebXR)
- Converts them to the same 4×4 robot-frame dict as
  OpenXRUnitySocketClient._process_transformations()

Usage::

    from omniteleop.leader.communication.webxr_vr_reader import WebXRVRReader

    reader = WebXRVRReader(port=5067)
    reader.start()
    reader.wait_for_data()

    data = reader.get_latest_transformation()
    # data["head"]               → 4×4 np.ndarray in robot frame
    # data["left_wrist"]         → 4×4 np.ndarray in robot frame
    # data["right_wrist"]        → 4×4 np.ndarray in robot frame
    # data["left_index_trigger"] → float [0, 1]
    # ... (same keys as _process_transformations controller mode)

    reader.close()

Connection setup
----------------
USB (recommended):
  1. adb reverse tcp:5067 tcp:5067
  2. Quest browser: http://localhost:5067
  3. Tap "Start XR" — that's it.

WiFi:
  Quest browser: http://<PC_IP>:5067

Note: WebXR does not expose elbow poses; left_elbow/right_elbow are identity.
"""

import os
import threading
import time
from typing import Optional

import numpy as np
from loguru import logger
from pytransform3d import transformations as pt

import socketio
import aiohttp.web

_HTML_PATH = os.path.join(os.path.dirname(__file__), "..", "web", "vr_client.html")

def head_transform(qworld_t_xrquest: np.ndarray) -> np.ndarray:
    qworld_t_robworld = np.array(
        [[0, -1, 0, 0],
         [0, 0, 1, 0],
         [-1, 0, 0, 0],
         [0, 0, 0, 1]]
    )
    xrquest_t_opencvquest = np.array(
        [[1, 0, 0, 0],
         [0, -1, 0, 0],
         [0, 0, -1, 0],
         [0, 0, 0, 1]]
    )
    robworld_t_opencvquest = np.linalg.inv(qworld_t_robworld) @ qworld_t_xrquest @ xrquest_t_opencvquest
    return robworld_t_opencvquest

def right_eef_transform(qworld_t_xrquest: np.ndarray) -> np.ndarray:
    qworld_t_robworld = np.array(
        [[0, -1, 0, 0],
         [0, 0, 1, 0],
         [-1, 0, 0, 0],
         [0, 0, 0, 1]]
    )
    xrquest_t_opencvquest = np.array(
        [[-1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, -1, 0],
         [0, 0, 0, 1]]
    )
    robworld_t_opencvquest = np.linalg.inv(qworld_t_robworld) @ qworld_t_xrquest @ xrquest_t_opencvquest
    return robworld_t_opencvquest

def left_eef_transform(qworld_t_xrquest: np.ndarray) -> np.ndarray:
    qworld_t_robworld = np.array(
        [[0, -1, 0, 0],
         [0, 0, 1, 0],
         [-1, 0, 0, 0],
         [0, 0, 0, 1]]
    )
    xrquest_t_opencvquest = np.array(
        [[1, 0, 0, 0],
         [0, -1, 0, 0],
         [0, 0, -1, 0],
         [0, 0, 0, 1]]
    )
    robworld_t_opencvquest = np.linalg.inv(qworld_t_robworld) @ qworld_t_xrquest @ xrquest_t_opencvquest
    return robworld_t_opencvquest

# ---------------------------------------------------------------------------
# Pose helpers (reused from previous implementation)
# ---------------------------------------------------------------------------

def _json_pose_to_matrix(pos: list, quat: list) -> np.ndarray:
    """JSON pose → 4×4 robot-frame SE(3) matrix.

    Mirrors process_pose(is_right_handed=False) + transform().
    """
    pq = [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]
    mat = pt.transform_from_pq(pq)
    return mat


def _process_frame(frame: dict) -> dict:
    """Convert raw JSON pose frame from browser to robot-frame matrices."""
    head_mat = head_transform(_json_pose_to_matrix(frame["head"]["pos"], frame["head"]["quat"]))
    left_wrist  = left_eef_transform(_json_pose_to_matrix(frame["left_wrist"]["pos"],  frame["left_wrist"]["quat"]))
    right_wrist = right_eef_transform(_json_pose_to_matrix(frame["right_wrist"]["pos"], frame["right_wrist"]["quat"]))
    return {
        "head":                   head_mat,
        "left_wrist":             left_wrist,
        "right_wrist":            right_wrist,
        "left_hand_trigger":      float(frame.get("left_squeeze",          0.0)),
        "right_hand_trigger":     float(frame.get("right_squeeze",         0.0)),
        "left_index_trigger":     float(frame.get("left_index_trigger",    0.0)),
        "right_index_trigger":    float(frame.get("right_index_trigger",   0.0)),
        "left_grip_trigger":      float(frame.get("left_squeeze",          0.0)),
        "right_grip_trigger":     float(frame.get("right_squeeze",         0.0)),
        "left_thumbstick":        list(frame.get("left_thumbstick",        [0.0, 0.0])),
        "right_thumbstick":       list(frame.get("right_thumbstick",       [0.0, 0.0])),
        "left_thumbstick_click":  bool(frame.get("left_thumbstick_click",  False)),
        "right_thumbstick_click": bool(frame.get("right_thumbstick_click", False)),
    }


# ---------------------------------------------------------------------------
# WebXRVRReader
# ---------------------------------------------------------------------------

class WebXRVRReader:
    """Receives Quest 3 WebXR poses via Socket.IO and exposes robot-frame 4×4 matrices.

    Drop-in replacement for OpenXRUnitySocketClient — same
    ``get_latest_transformation()`` and ``wait_for_data()`` interface.

    Args:
        host: Interface to bind on (default "0.0.0.0").
        port: TCP port for both the HTML page and Socket.IO (default 5067).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5067,
        ssl_certfile: Optional[str] = None,
        ssl_keyfile: Optional[str] = None,
    ):
        """
        Args:
            ssl_certfile: Path to PEM certificate file. Required for WiFi (HTTPS).
            ssl_keyfile:  Path to PEM private key file. Required for WiFi (HTTPS).

        WiFi usage (HTTPS required for WebXR over non-localhost):
            # Generate a self-signed cert once:
            openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem \\
                        -days 365 -nodes -subj "/CN=<PC_IP>"
            # Then:
            reader = WebXRVRReader(ssl_certfile="cert.pem", ssl_keyfile="key.pem")
            # Quest browser: https://<PC_IP>:5067  (accept the security warning once)
        """
        self.host = host
        self.port = port
        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile = ssl_keyfile

        self._latest: Optional[dict] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._start_error: Optional[Exception] = None
        self._runner: Optional[aiohttp.web.AppRunner] = None
        self._loop = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the Socket.IO server in a background thread.

        Blocks until the server is bound and accepting connections, then returns.

        Raises:
            OSError: If the port is already in use.
        """
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._start_error:
            raise self._start_error
        scheme = "https" if self.ssl_certfile else "http"
        logger.info(f"WebXRVRReader: serving on {scheme}://{self.host}:{self.port}")

    def get_latest_transformation(self) -> Optional[dict]:
        """Return the most recent processed pose frame, or None if no data yet."""
        with self._lock:
            return self._latest

    def wait_for_data(self) -> None:
        """Block until the first pose frame arrives from the Quest browser."""
        logger.info("WebXRVRReader: waiting for first frame from Quest browser …")
        while self._latest is None:
            time.sleep(0.05)
        logger.info("WebXRVRReader: first frame received.")

    def close(self) -> None:
        """Shut down the server."""
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("WebXRVRReader: closed.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_app(self) -> aiohttp.web.Application:
        sio = socketio.AsyncServer(async_mode="aiohttp", cors_allowed_origins="*")
        app = aiohttp.web.Application()
        sio.attach(app)

        @sio.on("pose_data")
        async def on_pose_data(sid, data):
            try:
                processed = _process_frame(data)
                with self._lock:
                    self._latest = processed
            except (KeyError, TypeError) as exc:
                logger.warning(f"WebXRVRReader: bad frame from {sid} — {exc}")

        @sio.event
        async def connect(sid, environ):
            logger.info(f"WebXRVRReader: browser connected  sid={sid}")

        @sio.event
        async def disconnect(sid):
            logger.info(f"WebXRVRReader: browser disconnected  sid={sid}")

        async def serve_html(request):
            html_path = os.path.abspath(_HTML_PATH)
            return aiohttp.web.FileResponse(html_path)

        app.router.add_get("/", serve_html)
        return app

    def _run(self) -> None:
        import asyncio
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_server())
        self._loop.run_forever()

    async def _start_server(self) -> None:
        try:
            app = self._build_app()
            self._runner = aiohttp.web.AppRunner(app)
            await self._runner.setup()
            ssl_ctx = None
            if self.ssl_certfile and self.ssl_keyfile:
                import ssl
                ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ssl_ctx.load_cert_chain(self.ssl_certfile, self.ssl_keyfile)
            site = aiohttp.web.TCPSite(
                self._runner, self.host, self.port,
                reuse_address=True, ssl_context=ssl_ctx,
            )
            await site.start()
            self._ready.set()
        except OSError as exc:
            self._start_error = exc
            self._ready.set()
