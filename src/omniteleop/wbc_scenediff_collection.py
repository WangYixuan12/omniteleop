"""Tick-driven pre-episode stereo capture, using the follower's sole Robot owner.

No hardware connection or motion occurs on import. Every motion tick is bounded;
the caller checks the live command before invoking it. Disk writes run separately.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import time
import uuid

import h5py
import numpy as np

from omniteleop.common.head_stereo_sweep import HeadStereoSweep

PAN_DEGREES = (-30., -20., -10., 0., 10., 20., 30.)
GROUP_WIDTHS = {"head": 3, "torso": 3, "left_arm": 7, "right_arm": 7}


def load_park_pose(path):
    """Use measured terminal joints; reject an unfinished/moving reference tail."""
    result = {}
    with h5py.File(path, "r") as data:
        if not bool(data.attrs.get("complete", True)):
            raise ValueError("parking reference is incomplete")
        for group, width in GROUP_WIDTHS.items():
            tail = np.asarray(data[f"obs/joint/{group}"][-10:], dtype=float)
            if tail.shape != (10, width) or not np.all(np.isfinite(tail)):
                raise ValueError(f"invalid parking reference {group}")
            if np.max(np.ptp(tail, axis=0)) > np.radians(.5):
                raise ValueError(f"parking reference {group} tail is not settled")
            result[group] = np.median(tail, axis=0)
    return result


def read_joints(driver):
    result = driver._read_measured_joints()
    for group, width in GROUP_WIDTHS.items():
        value = np.asarray(result.get(group), dtype=float)
        if value.shape != (width,) or not np.all(np.isfinite(value)):
            raise RuntimeError(f"invalid measured {group} during sweep")
        result[group] = value.copy()
    return result


def poll_pair(camera):
    obs = camera.get_obs(obs_keys=["left_rgb", "right_rgb"], include_timestamp=True)
    entries = [obs.get(key) for key in ("left_rgb", "right_rgb")]
    if any(not isinstance(e, dict) or e.get("data") is None for e in entries):
        return None
    stamps = [int(e.get("timestamp_ns", -1)) for e in entries]
    receives = [int(e.get("receive_time_ns", -1)) for e in entries]
    if min(*stamps, *receives) <= 0 or stamps[0] != stamps[1]:
        return None
    images = [np.asarray(e["data"]) for e in entries]
    if any(im.dtype != np.uint8 or im.ndim != 3 or im.shape[-1] != 3 for im in images):
        raise ValueError("sweep requires uint8 RGB stereo")
    if images[0].shape != images[1].shape:
        raise ValueError("stereo dimensions differ")
    return (*images, stamps[0], *receives)


class CollectionSweep:
    """Parking -> seven settled pairs -> head restoration -> save -> alignment.

    The second press hands the measured parked pose directly to normal alignment; it
    does not perform a separate return-to-nominal move. The torso and base stay at
    their measured starting positions throughout capture.
    """
    def __init__(self, driver, request_id, *, clock=time.monotonic):
        from omniteleop.wbc_policy_format import WBCPolicyFK

        self.driver = driver
        self.clock = clock
        self.request_id = request_id
        self.arm_speed = float(getattr(driver.args, "scenediff_arm_speed", 0.50))
        if not np.isfinite(self.arm_speed) or self.arm_speed <= 0:
            raise ValueError("scenediff_arm_speed must be finite and > 0")
        self.state = "parking"
        self.detail = "parking arms"
        self.rows = []
        self.view = 0
        self.future = None
        self.pool = None
        self.sha256 = ""
        self.original = read_joints(driver)
        self.park = load_park_pose(driver.args.scenediff_park_reference)
        self.fk = WBCPolicyFK(driver.ik)
        if not driver.enable["arms"] or not driver.enable["head"]:
            raise RuntimeError("SceneDiff sweep requires arms and head enabled")
        if not driver.ik.collision_enabled:
            raise RuntimeError("SceneDiff parking requires the self-collision model")
        # The taught clearance is valid for this torso posture, not an arbitrary lean.
        if np.max(np.abs(self.original["torso"] - self.park["torso"])) > np.radians(3):
            raise RuntimeError("torso differs from parking reference by >3deg; home first")
        self.calibration = driver._query_head_stereo_calibration()
        if "head" not in driver._camera_clock_offsets_ns:
            raise RuntimeError("sweep requires calibrated head publisher clock")
        self.camera = driver.robot.sensors.head_camera
        self.base_origin = self._base_pose()
        self.torso_origin = self.original["torso"].copy()
        root = Path(driver.args.scenediff_sweep_dir).expanduser()
        episode_id = driver._episode.episode_id
        self.path = root / f"episode_{episode_id}_{uuid.uuid4().hex}_sweep.hdf5"
        self.head_template = self.park["head"].copy()
        self.jobs = [(g, self.park[g]) for g in ("left_arm", "right_arm")]
        self.group = None
        self.target = None
        self.guard = None
        self.stable_since = None
        self.barrier = None
        self.alignment_joints = None
        self.deadline = self.clock() + 10
        # Prove cameras are advancing before any movement.
        self.state = "camera_check"
        self._startup_stamp = None
        for g, q in self.jobs + [("head", self.head_template)]:
            self._check_limits(g, q)
        for pan in PAN_DEGREES:
            q = self.head_template.copy()
            q[1] = np.radians(pan)
            self._check_limits("head", q)
        driver.stop_all_motion()

    def _base_pose(self):
        odom = self.driver._odom
        if odom is None:
            raise RuntimeError("sweep needs wheel odometry to verify a stationary base")
        if odom.age > self.driver.args.source_timeout:
            raise RuntimeError("stale base odometry during sweep")
        pose = np.asarray(odom.snapshot()["pose"], dtype=float)
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            raise RuntimeError("invalid base pose during sweep")
        return pose.copy()

    def _check_limits(self, group, q):
        limits = np.asarray(self.driver._comp(group).joint_pos_limit, dtype=float)
        if limits.shape != (len(q), 2) or not np.all(np.isfinite(limits)):
            raise RuntimeError(f"missing {group} hardware limits")
        margin = np.radians(1)
        if np.any(q < limits[:, 0] + margin) or np.any(q > limits[:, 1] - margin):
            raise RuntimeError(f"{group} sweep target violates hardware limits")

    def _begin_move(self, group, target):
        from omniteleop.wbc_robot_home import _make_home_guard
        self._check_limits(group, target)
        self.group, self.target = group, np.asarray(target).copy()
        self.guard = _make_home_guard(self.driver, group)
        self.command = read_joints(self.driver)[group].copy()
        self.deadline = self.clock() + 30
        self.last_move = self.clock()
        self.stable_since = None

    def _move_tick(self, now, measured):
        if now > self.deadline:
            raise TimeoutError(f"{self.group} did not reach sweep target")
        # Faster arm parking; retain the settled head sweep's original speed.
        # Clamp elapsed time so a stalled loop cannot issue a large catch-up step.
        speed = self.arm_speed if self.group in ("left_arm", "right_arm") else .10
        step = speed * min(max(now - self.last_move, 0), .03)
        self.last_move = now
        if np.max(np.abs(measured[self.group] - self.command)) > np.radians(5):
            raise RuntimeError(f"{self.group} lagged sweep command by >5deg")
        delta = np.clip(self.target - self.command, -step, step)
        if self.group in ("left_arm", "right_arm"):
            # Let responsive arms use the full ramp speed, but stop advancing ahead
            # of slower hardware. Scale the whole step to preserve its direction;
            # never pull the command backwards to a lagging measurement.
            moving = np.abs(delta) > 0
            if np.any(moving):
                lead = (self.command - measured[self.group]) * np.sign(delta)
                room = np.maximum(np.radians(3) - lead[moving], 0.)
                delta *= min(1., float(np.min(room / np.abs(delta[moving]))))
        self.command += delta
        self.guard(self.command)
        self.driver._comp(self.group).set_joint_pos(self.command.tolist(), wait_time=0.)
        at_target = (
            np.max(np.abs(measured[self.group] - self.target)) <= np.radians(.5)
            and np.max(np.abs(self.command - self.target)) < 1e-6
        )
        if not at_target:
            self.stable_since = None
        elif self.stable_since is None:
            self.stable_since = now
        return self.stable_since is not None and now - self.stable_since >= .35

    def _fresh_pair(self):
        pair = poll_pair(self.camera)
        if pair is None:
            return None
        stamp = pair[2]
        age = self.driver._camera_capture_ages_ns(time.time_ns(), head_ns=stamp, wrist_ns={})["head"]
        if age < -20_000_000 or age > 150_000_000:
            return None
        return pair

    def _save(self, artifact):
        artifact.save(self.path)
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def begin_alignment(self):
        if self.state != "ready":
            raise RuntimeError("alignment requested without a completed sweep")
        # Seed the normal controller from this exact measured pose. This removes the
        # separate park-to-nominal ramp and prevents alignment from seeing a nominal
        # solver state while the hardware is still parked.
        self.alignment_joints = read_joints(self.driver)
        self.jobs = []
        self.group = None
        self.state = "aligned"
        self.detail = "aligning from parked pose"
        self.driver.stop_all_motion()

    def tick(self):
        now = self.clock()
        d = self.driver
        d.robot.chassis.set_velocity(vx=0., vy=0., wz=0., wait_time=0.)
        base_delta = self._base_pose() - self.base_origin
        yaw_delta = np.arctan2(np.sin(base_delta[2]), np.cos(base_delta[2]))
        if np.linalg.norm(base_delta[:2]) > .01 or abs(yaw_delta) > np.radians(.5):
            raise RuntimeError("base moved during stationary sweep")
        measured = read_joints(d)
        if np.max(np.abs(measured["torso"] - self.torso_origin)) > np.radians(.25):
            raise RuntimeError("torso moved during stationary sweep")
        if self.state == "camera_check":
            pair = self._fresh_pair()
            if pair is not None:
                if self._startup_stamp is None:
                    self._startup_stamp = pair[2]
                elif pair[2] > self._startup_stamp:
                    self.state = "parking"
            if now > self.deadline:
                raise TimeoutError("head stereo did not advance; no parking motion issued")
            return
        if self.state == "parking":
            if self.group is None:
                if self.jobs:
                    self._begin_move(*self.jobs.pop(0))
                else:
                    self.state = "sweeping"
                return
            if self._move_tick(now, measured):
                self.group = None
            return
        if self.state == "sweeping":
            if self.group is None:
                target = self.head_template.copy()
                target[1] = np.radians(PAN_DEGREES[self.view])
                self._begin_move("head", target)
                self.detail = f"sweep view {self.view + 1}/7"
            if self._move_tick(now, measured):
                self.state = "capture"
                self.barrier = None
                self.deadline = now + 5
            return
        if self.state == "capture":
            if now > self.deadline:
                raise TimeoutError("no fresh settled stereo pair")
            if np.max(np.abs(measured["head"] - self.target)) > np.radians(.5):
                raise RuntimeError("head left capture target")
            for arm in ("left_arm", "right_arm"):
                if np.max(np.abs(measured[arm] - self.park[arm])) > np.radians(.5):
                    raise RuntimeError(f"{arm} left parking pose")
            pair = self._fresh_pair()
            if pair is None:
                return
            if self.barrier is None:
                self.barrier = pair[2]
                self.capture_before = measured
                return
            if pair[2] <= self.barrier:
                return
            after = read_joints(d)
            motion = max(np.degrees(np.max(np.abs(after[g] - self.capture_before[g])))
                         for g in GROUP_WIDTHS)
            if motion > .25:
                raise RuntimeError("robot joints moved around stereo capture")
            joints = {g: (after[g] + self.capture_before[g]) / 2 for g in GROUP_WIDTHS}
            transform = self.fk.base_frame_poses(**joints)["head"]
            self.rows.append((pair[0].copy(), pair[1].copy(), *pair[2:], joints, transform, motion))
            self.view += 1
            self.group = None
            if self.view == len(PAN_DEGREES):
                self._begin_move("head", self.original["head"])
                self.state = "restore_head"
                self.detail = "restoring head"
            else:
                self.state = "sweeping"
            return
        if self.state == "restore_head":
            if not self._move_tick(now, measured):
                return
            r = self.rows
            artifact = HeadStereoSweep(
                left_rgb=np.stack([v[0] for v in r]), right_rgb=np.stack([v[1] for v in r]),
                capture_timestamp_ns=np.array([v[2] for v in r]),
                left_receive_ns=np.array([v[3] for v in r]),
                right_receive_ns=np.array([v[4] for v in r]),
                **{g + "_joints": np.stack([v[5][g] for v in r]) for g in GROUP_WIDTHS},
                base_t_cam=np.stack([v[6] for v in r]), target_pan_deg=np.array(PAN_DEGREES),
                measured_pan_deg=np.array([np.degrees(v[5]["head"][1]) for v in r]),
                capture_joint_motion_deg=np.array([v[7] for v in r]),
                left_k=self.calibration["left_K"], right_k=self.calibration["right_K"],
                baseline_m=self.calibration["baseline_m"],
            )
            self.pool = ThreadPoolExecutor(max_workers=1)
            self.future = self.pool.submit(self._save, artifact)
            self.state, self.detail = "saving", "saving stereo sweep"
            return
        if self.state == "saving" and self.future.done():
            self.sha256 = self.future.result()
            self.pool.shutdown(wait=False)
            self.state, self.detail = "ready", "scan saved; release then hold RIGHT grip"
            print(f"\n[scenediff] sweep saved -> {self.path}")

    def close(self):
        self.driver.stop_all_motion()
        if self.pool is not None:
            self.pool.shutdown(wait=False)

    def episode_metadata(self):
        if self.state != "aligned" or not self.sha256:
            raise RuntimeError("episode cannot start without a completed pre-episode sweep")
        return {"path": np.bytes_(str(self.path)), "sha256": np.bytes_(self.sha256),
                "request_id": np.bytes_(self.request_id),
                "park_reference": np.bytes_(str(self.driver.args.scenediff_park_reference))}
