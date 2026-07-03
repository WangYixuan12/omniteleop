# WBC VR Episode Reset Design

## Goal

Allow repeated real-robot WBC VR recording episodes without exiting and relaunching
`scripts/wbc_vr_leader.py` and `scripts/wbc_vr_robot.py`.

After an episode, the operator should be able to stop recording, return the robot to
nominal, recalibrate/reference-align, and start the next episode in the same process.

## Controls

- Left X ends the current episode and returns the leader to `static`.
- Left trigger, while `static`, requests robot homing to nominal.
- Right grip hold, while `static`, keeps the existing calibration flow:
  calibrate VR, optionally enter reference `align`, then enter `teleop`.
- Left Y is the terminal emergency/program stop. It publishes `exit_requested=True`,
  stops robot motion, saves any active recording, and exits.

## Stage Semantics

`static` remains the reset stage and means `estop=True`.

When the leader enters `static`, it publishes empty EE/head targets and zero chassis
commands. The follower must hold motion and must not record frames.

The follower starts a new episode only when the leader reaches `calib_stage="teleop"`.
The existing optional `align` stage may move the robot but does not record.

## Episode Rollover

On left X:

1. The leader switches to `stage="static"` instead of exiting.
2. The leader publishes a static/e-stop frame immediately so the follower stops.
3. The follower stops and saves the active episode if one is recording.
4. The scripts keep running and wait for the next reset sequence.

The follower must support stopping an `EpisodeRecorder` mid-run without shutting down
hardware. The next time `teleop` starts, the same recorder object can start a new
episode ID.

## Homing Request

Left trigger in `static` publishes a non-terminal homing request by setting a new
`VRJointData.home_requested=True` field on a static/e-stop frame. The field defaults
to `False`, so existing messages and replay data remain valid.

The follower handles the request by running the same nominal homing routine used at
startup, `home_to_nominal(...)`, while staying in the live process. During homing,
recording remains stopped and the follower continues to publish a status frame suitable
for the headset HUD. After homing completes, the leader stays in `static`; the operator
then holds right grip to recalibrate and proceed through the existing alignment path.

Repeated homing requests while a home is already in progress should be ignored or
coalesced so homing does not run concurrently.

## Emergency Stop

Left Y keeps the old terminal behavior:

1. Publish `exit_requested=True`.
2. Stop robot motion immediately.
3. Save any active episode.
4. Shut down both scripts.

This is distinct from left X, which is now an episode rollover command.

## Data And Compatibility

Add `home_requested: bool = False` to `VRJointData`. This keeps `static` unambiguous:
`calib_stage="static"` still means e-stop/reset, while `home_requested=True` requests
the one-shot nominal homing action.

Existing replay behavior should not change. Recorded episodes should not include the
left-X static frame, homing frames, or alignment frames.

## Testing

Add focused tests for:

- left X returns the leader to `static` without setting `exit_requested`;
- left Y publishes terminal `exit_requested=True`;
- follower stops/saves an active episode when the source returns to `static`;
- follower can start a subsequent episode on the next `teleop`;
- homing request in `static` calls the same nominal homing path without starting
  recording.
