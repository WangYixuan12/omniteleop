#!/usr/bin/env python3
"""Drift-free base pose from the mount-rigid iPhone (ARKit), as an odometry drop-in.

The phone bolted to the E-stop tower publishes ``<robot>/tracking/base_pose`` from
``record3d/tracking/base_pose_pub.py`` (robot side, ~15 Hz): the base pose in the ARKit
SESSION world, already through the phone->base extrinsic from
``record3d/calibration/calib_solve.py``.

:class:`ARKitBaseTracker` mirrors the surface of ``wbc_vr_robot``'s ``OdometryThread``
(``pose`` / ``age`` / ``snapshot`` / ``reset_origin`` / ``close``) so the follower can
read a measured base pose from either source through the same calls. Like the wheel
odometry, :attr:`pose` is planar ``(x, y, yaw)`` in the ENGAGE-ORIGIN frame -- the ARKit
world pose at the last :meth:`reset_origin`. Unlike it, the underlying measurement does
not accumulate drift, which is the whole point: swerve odometry integrates steering and
wheel velocity, so its yaw walks away over a take and every world-frame quantity derived
from it (``obs/base/pose``, and the offline ``world_t_cam``) walks with it.

Two failure modes are surfaced rather than smoothed over, because both silently corrupt
every pose derived from them:

* **staleness** -- :attr:`age` grows when the phone stops publishing (USB drop, app
  closed, tracking stall); the follower holds exactly as it does on stale odometry.
* **relocalization** -- ARKit can teleport its world origin after a tracking loss. The
  publisher flags those samples and counts them; :attr:`jumped` latches True once one
  lands after the current origin was captured, and only :meth:`reset_origin` (i.e. a
  re-engage) clears it.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
from dexcomm import Node
from dexcomm.codecs import DictDataCodec

TOPIC = "tracking/base_pose"


def _wrap(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class ARKitBaseTracker:
    """Latest iPhone-tracked base pose, re-zeroed at engage like the wheel odometry.

    ``namespace`` is the robot's Zenoh namespace (``$ROBOT_NAME``, e.g.
    ``dm/vg3cfe65689d-1``) -- the publisher runs on the robot, so its topic lives under
    the ROBOT namespace, not the leader/follower one.
    """

    def __init__(self, namespace: str, *, name: str = "wbc_vr_arkit") -> None:
        if not namespace:
            raise ValueError(
                "ARKitBaseTracker needs the robot's Zenoh namespace (export ROBOT_NAME)"
            )
        self.topic = f"{namespace}/{TOPIC}"
        self._lock = threading.Lock()
        self._world = np.full(3, np.nan)     # (x, y, yaw) in the ARKit session world
        self._jumps = -1                     # publisher's cumulative jump count
        self._rp = np.full(2, np.nan)        # base roll/pitch (gravity-aligned world)
        self._seq = -1
        self._update_t = 0.0
        self._origin: Optional[np.ndarray] = None
        self._origin_jumps = 0
        self.node = Node(name=name, namespace=namespace)
        self.sub = self.node.create_subscriber(
            TOPIC, self._on_msg, decoder=DictDataCodec.decode
        )

    def _on_msg(self, msg: dict) -> None:
        try:
            planar = np.asarray(msg["planar"], dtype=np.float64)
            rp = np.asarray(msg["rp_rad"], dtype=np.float64)
            seq, jumps = int(msg["seq"]), int(msg["jumps"])
        except (KeyError, TypeError, ValueError):
            return                            # ignore malformed frames
        if planar.shape != (3,) or rp.shape != (2,) or not np.all(np.isfinite(planar)):
            return
        with self._lock:
            self._world, self._rp = planar, rp
            self._seq, self._jumps = seq, jumps
            self._update_t = time.perf_counter()

    def wait_first(self, timeout: float) -> bool:
        """Block until the first pose arrives (True) or ``timeout`` elapses (False)."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._lock:
                if self._seq >= 0:
                    return True
            time.sleep(0.05)
        return False

    def _rel(self, world: np.ndarray, origin: Optional[np.ndarray]) -> np.ndarray:
        if origin is None or not np.all(np.isfinite(world)):
            return np.full(3, np.nan)
        d = world[:2] - origin[:2]
        c, s = np.cos(-origin[2]), np.sin(-origin[2])
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1],
                         _wrap(world[2] - origin[2])])

    @property
    def pose(self) -> np.ndarray:
        """Base pose ``(x, y, yaw)`` in the engage-origin frame (NaN before the origin)."""
        with self._lock:
            return self._rel(self._world, self._origin)

    @property
    def world_pose(self) -> np.ndarray:
        """Base pose ``(x, y, yaw)`` in the ARKit session world -- unbroken across engages."""
        with self._lock:
            return self._world.copy()

    @property
    def age(self) -> float:
        """Seconds since the last accepted sample (inf before the first)."""
        with self._lock:
            return time.perf_counter() - self._update_t if self._update_t else float("inf")

    @property
    def has_origin(self) -> bool:
        """True once :meth:`reset_origin` has captured an engage origin."""
        with self._lock:
            return self._origin is not None

    @property
    def jumped(self) -> bool:
        """True if ARKit relocalized since the origin was captured (cleared by a re-zero)."""
        with self._lock:
            return self._origin is not None and self._jumps > self._origin_jumps

    def snapshot(self) -> dict:
        """Atomic copy of the tracker state (engage-origin + world pose, age, health).

        Taken under the lock so the pose a control tick uses and the health logged
        beside it come from the SAME sample.
        """
        with self._lock:
            age = (time.perf_counter() - self._update_t) if self._update_t else float("inf")
            return {
                "pose": self._rel(self._world, self._origin),
                "world_pose": self._world.copy(),
                "rp": self._rp.copy(),
                "age": age,
                "seq": int(self._seq),
                "jumps": int(self._jumps),
                "jumped": self._origin is not None and self._jumps > self._origin_jumps,
                "has_origin": self._origin is not None,
            }

    def reset_origin(self) -> bool:
        """Re-zero on the latest sample -- called on the engage edge, like the odometry.

        Returns False (leaving :attr:`pose` NaN and :attr:`has_origin` False) when there
        is no sample to zero on, rather than inventing an origin.
        """
        with self._lock:
            if self._seq < 0 or not np.all(np.isfinite(self._world)):
                self._origin = None
                return False
            self._origin = self._world.copy()
            self._origin_jumps = self._jumps
            return True

    def close(self) -> None:
        """Close the Zenoh node (best-effort; dexcomm builds differ on close vs shutdown)."""
        try:
            close = getattr(self.node, "close", None) or getattr(self.node, "shutdown", None)
            if close is not None:
                close()
        except Exception:
            pass
