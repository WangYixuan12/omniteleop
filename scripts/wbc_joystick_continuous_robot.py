#!/usr/bin/env python3
# ruff: noqa: SLF001  (reuses the loaded joystick follower's private helpers by design)
"""Real-robot joystick follower for continuous collection: takes end and start mid-teleop.

Pairs with ``scripts/wbc_joystick_continuous_leader.py``. Hardware bring-up, homing, the
arm/torso IK, the joystick base, the head, the recording format and every CLI flag are
``scripts/wbc_joystick_robot.py``; only the episode boundaries move from the e-stop to the
leader's stage:

  * ``calib_stage == "teleop"`` records. Entering it parks the base, re-zeros the base
    world (odometry / ARKit origin and the joystick reference) so each take's
    ``obs/base/pose`` starts at the origin as after an engage, and starts the take. Arms,
    torso, head and IK carry on from where they are: no homing, no IK reset.
  * ``calib_stage == "teleop_idle"`` (left X) stops the take, which saves in the background
    while teleop continues. If the next take is requested before that save finished, the
    base parks and the robot pauses until it has, so a demo never starts before its
    recording does. ``--streaming-recorder`` keeps that save short.
  * An e-stop still ends a take the stock way (stop all motion, then save), and left Y
    still stops everything, saving a take in progress.

The base is parked before every step that blocks the control loop -- a pending save, the
next take's camera/SoC clock queries (~1 s), the synchronous ``--debug-dir`` flush -- so the
chassis never keeps its last twist while the loop is stalled.

``--replay`` (no episode buttons) and ``--scenediff-sweep`` (one sweep cannot describe later
takes, each of which re-zeros the base world) are rejected.

Run in the dexmate conda env, with the joystick follower's flags::

    python scripts/wbc_joystick_continuous_robot.py --record --streaming-recorder
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

import numpy as np

# Load the joystick follower as a module (scripts/ is not a package), the way it loads the
# VR follower. Its main() is __main__-guarded, so this only binds its definitions.
_JOY_PATH = Path(__file__).resolve().parent / "wbc_joystick_robot.py"
_spec = importlib.util.spec_from_file_location("wbc_joystick_robot", _JOY_PATH)
_joy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_joy)

JoystickHardwareDriver = _joy.JoystickHardwareDriver
_run_joystick_ik_mode = _joy._run_joystick_ik_mode
_wait_for_recorder_save = _joy._base._wait_for_recorder_save
_sha256_file = _joy._base._sha256_file
_WRIST_SENSOR_IDS = _joy._base._WRIST_SENSOR_IDS

_LOG = "[wbc_joystick_continuous_robot]"


class ContinuousJoystickHardwareDriver(JoystickHardwareDriver):
    """``JoystickHardwareDriver`` whose takes start and stop on the leader stage."""

    def __init__(self, *args, **kwargs) -> None:
        # A stopped take still saving in the background. Set BEFORE super().__init__ so a
        # partial-init teardown (close()) never sees it unset.
        self._saving_path: Optional[str] = None
        super().__init__(*args, **kwargs)

    def start_recording_if_teleop(self, vr) -> None:
        """Episode boundaries; run_loop calls this every tick, after actuation."""
        self._report_background_save()
        if self._episode is None or vr is None or vr.estop:
            return  # no --record, or e-stopped: run_loop's stop-then-save path owns it
        teleop = str(getattr(vr, "calib_stage", "")) == "teleop"
        if teleop and not self._episode.recording:
            if self._saving_path is None:
                self._start_episode()
            else:
                # The previous take was still saving at the check above. Neither recorder
                # can start meanwhile (and a start would reset the save error before it
                # is read), and teleop must not run ahead of a take that has not started:
                # pause, base parked, until it is on disk. The take starts next tick, on a
                # fresh loop clock for its startup gates.
                self._park_base()
                print(f"\n{_LOG} waiting for {self._saving_path} to finish saving "
                      "before the next episode ...")
                self._report_background_save(wait=True)
        elif not teleop and self._episode.recording:
            self._stop_episode_in_background()

    def _park_base(self) -> None:
        """Zero the chassis and park the joystick reference (``_drive_base``'s hold)."""
        self._drive_base(None, True, self.enable, 0.0)

    def _start_episode(self) -> None:
        """Start a take at the current posture, with the base world re-zeroed here."""
        self._park_base()  # start_recording_episode() blocks on camera/SoC clock queries
        # Each take's obs/base/pose (policy state) and joystick reference start at the
        # origin, as after an engage. Nothing else resets: the IK has the base locked out
        # and the arm/head targets are base-relative.
        if self._odom is not None:
            self._odom.reset_origin()
        if self._arkit is not None and not self._arkit.reset_origin():
            print(f"{_LOG} WARNING: no iPhone pose to zero the ARKit origin on -- base "
                  "tracking stays NaN until the publisher recovers.")
        self._world_frame_epoch += 1
        self._shaper.reset(np.zeros(3))
        self.start_recording_episode()
        print(f"\n{_LOG} recording episode_{self._episode.episode_id} "
              "(base origin re-zeroed here).")

    def _stop_episode_in_background(self) -> None:
        """End the take and let it save while teleop continues.

        ``HardwareDriver.stop_recording_episode`` without its blocking wait (the recorder
        already saves in a background thread); ``_report_background_save`` reports the
        outcome on a later tick.
        """
        saved_id = self._episode.episode_id
        n = self._episode.num_frames()
        for sidecar in (getattr(self, "_native_state_recorder", None),
                        getattr(self, "_camera_motion_recorder", None)):
            if sidecar is not None:
                sidecar.stop(aborted=n == 0)
        if n == 0:
            self._park_base()  # a zero-frame streaming stop joins its writer (up to 5 s)
        path = self._episode.stop()
        if path is None:
            print(f"\n{_LOG} recording: 0 frames -- nothing saved; teleop continues.")
            return
        self._saving_path = path
        print(f"\n{_LOG} saving {n} frames -> {path} in the background; teleop continues.")
        if self.args.head_right_rgb:
            paired = n - self._rec_right_unpaired
            print(f"{_LOG} head stereo pairing: {paired}/{n} frames "
                  f"({100.0 * paired / n:.1f}%) carry head_right_frame_ns == head_frame_ns.")
        if self.args.wrist_right_rgb:
            for arm in _WRIST_SENSOR_IDS:
                paired = n - self._rec_wrist_right_unpaired[arm]
                print(f"{_LOG} {arm} wrist stereo pairing: {paired}/{n} frames "
                      f"({100.0 * paired / n:.1f}%) carry {arm}_wrist_right_frame_ns == "
                      f"{arm}_wrist_frame_ns.")
        if self._traj is not None and len(self._traj):
            self._park_base()  # the paired /debug HDF5 is written synchronously
            self._traj.flush(saved_id)

    def _report_background_save(self, *, wait: bool = False) -> None:
        """Report the background save once it is done (``wait``: block until it is).

        A failed or timed-out save raises, ending the session like the stock follower. An
        interrupted wait (Ctrl-C) leaves the save pending, so close() still finishes it.
        """
        path = self._saving_path
        if path is None or (self._episode.saving and not wait):
            return
        try:
            _wait_for_recorder_save(self._episode, path=path)
        except Exception:
            self._saving_path = None  # a timed-out save is reported exactly once
            raise
        self._saving_path = None
        error = self._episode.last_save_error
        if error is not None:
            raise RuntimeError(f"{_LOG} episode save failed for {path}: {error}") from error
        print(f"\n{_LOG} episode saved -> {path}")

    def close(self) -> None:
        """Stock teardown (stop motion, save a take still recording), then any background save."""
        try:
            super().close()
        finally:
            # Even if the stock teardown raised: a buffered save runs in a daemon thread
            # that process exit would kill mid-write.
            self._report_background_save(wait=True)

    def _recording_provenance_metadata(self) -> dict:
        """Snapshot this entry point and its leader beside the joystick sources."""
        out = super()._recording_provenance_metadata()
        repo = _JOY_PATH.parents[1]
        for label, path in (
            ("wbc_joystick_continuous_robot", Path(__file__).resolve()),
            ("wbc_joystick_continuous_leader",
             repo / "scripts/wbc_joystick_continuous_leader.py"),
        ):
            out["source_snapshot"][label] = {
                "path": np.asarray(str(path.relative_to(repo)).encode()),
                "sha256": np.asarray(_sha256_file(path).encode()),
                "content": np.asarray(path.read_bytes()),
            }
        return out


def _run_continuous_mode(args, enable) -> None:
    """Reject flags continuous collection cannot honour, then run the joystick follower."""
    if args.replay is not None:
        raise SystemExit(f"{_LOG} --replay has no episode buttons; replay with "
                         "wbc_joystick_robot.py instead.")
    if args.scenediff_sweep:
        raise SystemExit(f"{_LOG} --scenediff-sweep is unsupported: one sweep cannot "
                         "describe later takes (each re-zeros the base world).")
    _run_joystick_ik_mode(args, enable)


def main() -> None:
    # Reuse the joystick follower's CLI and runner verbatim, with this driver and help
    # text. _joy is this script's private instance of that module; wbc_joystick_robot.py
    # is unchanged.
    _joy.__doc__ = __doc__
    _joy.JoystickHardwareDriver = ContinuousJoystickHardwareDriver
    _joy._run_joystick_ik_mode = _run_continuous_mode
    _joy.main()


if __name__ == "__main__":
    main()
