"""Asynchronous, timestamped camera pose/IMU sidecars for robot episodes.

No pose is inserted into a robot observation by arrival order. The sidecar keeps
full-rate samples, independent tracker origins, and validity for offline joining.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np
from loguru import logger


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_default(value.item()) if isinstance(value.item(), bytes) else value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    raise TypeError(f"Unsupported motion metadata type {type(value).__name__}")


class CameraMotionRecorder:
    def __init__(self, node, save_dir, camera_info):
        from dexcomm.codecs import JsonDataCodec

        self.save_dir = Path(save_dir)
        self.camera_info = camera_info
        self.lock = threading.Lock()
        self.history = deque(maxlen=360)
        self.active = None
        self.jobs = []
        self.subs = [node.create_subscriber(
            f"sensors/{sensor}/motion", callback=self._callback(sensor),
            decoder=JsonDataCodec.decode, buffer_size=1,
        ) for sensor in camera_info]

    def _callback(self, sensor):
        def callback(message):
            packet = getattr(message, "data", message)
            received_ns = time.time_ns()
            row = dict(type="sample", receive_ns=received_ns, sensor_id=sensor, packet=packet)
            now = time.monotonic()
            with self.lock:
                self.history.append(row)
                for job in self.jobs:
                    if job["closed"] or now >= job["stop_at"]:
                        continue
                    try:
                        job["queue"].put_nowait(row)
                    except queue.Full:
                        job["dropped"] += 1
        return callback

    def start(self, episode_id, clock_calibration):
        """Enqueue episode creation and preroll; file opening runs in the writer."""
        with self.lock:
            if self.active is not None:
                raise RuntimeError("Camera motion episode already active")
            self.jobs = [job for job in self.jobs
                         if not job["closed"] or job["thread"].is_alive()]
            start_ns = time.time_ns()
            path = self.save_dir / f"episode_{int(episode_id)}_camera_motion_{uuid.uuid4().hex}.jsonl"
            job = dict(queue=queue.Queue(maxsize=4096), stop_at=float("inf"),
                       closed=False, dropped=0, errors=[], aborted=False, path=path,
                       start_ns=start_ns, stop_ns=None)
            for row in self.history:
                if row["receive_ns"] >= start_ns - 1_000_000_000:
                    job["queue"].put_nowait(row)
            header = dict(type="metadata", schema_version=1, episode_id=int(episode_id),
                          start_wall_ns=start_ns, camera_info=self.camera_info,
                          camera_clock_calibration=clock_calibration,
                          pre_roll_s=1, post_roll_s=.5,
                          association="join by sensor/image_timestamp_ns; poses have independent world origins")
            job["thread"] = threading.Thread(target=self._write, args=(job, header),
                                             daemon=True, name="camera_motion_writer")
            self.jobs.append(job)
            self.active = job
            job["thread"].start()
        return path.name

    def stop(self, *, aborted=False):
        """Schedule a half-second tail without waiting in the controller."""
        with self.lock:
            if self.active is None:
                return
            self.active.update(stop_at=time.monotonic() + .5,
                               stop_ns=time.time_ns(), aborted=bool(aborted))
            self.active = None

    def _write(self, job, header):
        path = job["path"]
        partial = Path(str(path) + ".partial")
        counts = {sensor: 0 for sensor in self.camera_info}
        gaps = {sensor: 0 for sensor in self.camera_info}
        previous = {}
        invalid_poses = {sensor: 0 for sensor in self.camera_info}
        last_receive = {}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
            with partial.open("x", buffering=65536) as stream:
                stream.write(json.dumps(header, allow_nan=False, default=_json_default) + "\n")
                last_flush = time.monotonic()
                while True:
                    with self.lock:
                        done = time.monotonic() >= job["stop_at"] and job["queue"].empty()
                        if done:
                            job["closed"] = True
                    if done:
                        break
                    try:
                        row = job["queue"].get(timeout=.1)
                    except queue.Empty:
                        continue
                    sensor, packet = row["sensor_id"], row["packet"]
                    if not isinstance(packet, dict) or packet.get("sensor_id") != sensor or packet.get("schema_version") != 1:
                        job["errors"].append(f"Malformed motion packet for {sensor}")
                        continue
                    identity = (packet["session_id"], int(packet["sequence"]))
                    prev = previous.get(sensor)
                    if prev is not None:
                        if identity[0] != prev[0]:
                            job["errors"].append(f"Publisher restarted: {sensor}")
                        elif identity[1] != prev[1] + 1:
                            gaps[sensor] += abs(identity[1] - prev[1] - 1)
                    previous[sensor] = identity
                    last_receive[sensor] = row["receive_ns"]
                    counts[sensor] += 1
                    invalid_poses[sensor] += (packet["pose"].get("available", True)
                                               and not packet["pose"]["valid"])
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    if time.monotonic() - last_flush > 1:
                        stream.flush()
                        last_flush = time.monotonic()
                for sensor in self.camera_info:
                    if last_receive.get(sensor, 0) < job["stop_ns"] - 500_000_000:
                        job["errors"].append(f"Missing motion stream at episode end: {sensor}")
                complete = not job["aborted"] and not job["errors"] and not job["dropped"] and all(counts.values()) and not any(gaps.values())
                footer = dict(type="summary", complete=bool(complete),
                              stop_wall_ns=job["stop_ns"], aborted=job["aborted"],
                              queue_drops=job["dropped"], errors=job["errors"],
                              packets=counts, sequence_gaps=gaps, invalid_poses=invalid_poses)
                stream.write(json.dumps(footer, allow_nan=False) + "\n")
            if complete:
                partial.rename(path)
            else:
                logger.error(f"Camera motion recording incomplete: {partial}; {footer}")
        except Exception as exc:
            job["errors"].append(str(exc))
            logger.error(f"Camera motion writer failed for {partial}: {exc}")
        finally:
            with self.lock:
                job["closed"] = True

    def close(self):
        self.stop()
        deadline = time.monotonic() + 5
        for job in self.jobs:
            job["thread"].join(timeout=max(0, deadline - time.monotonic()))
        for sub in self.subs:
            sub.shutdown()
