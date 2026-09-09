# Collect a joystick demonstration in BEHAVIOR

Run the simulator in the **behavior51** environment and the leader in
**dexmate2**. `teleop_joystick.py` automatically starts its control process using
`--dexmate-python` (default `~/miniconda3/envs/dexmate2/bin/python`). That process
uses `wbc_joystick_robot.py`'s actual shared `run_loop`, IK setup, config binding,
`JoystickBaseShaper`, follower-status publisher, and streaming episode recorder.
It constructs no hardware driver or dexcontrol Robot.

First check runtime inputs:

```bash
cd /home/yifan/workspace/omniteleop_yifan
~/miniconda3/envs/behavior51/bin/python sim_eval/teleop_joystick.py --check
```

This checkout uses the separate `behavior51` environment because the older
`behavior` environment contains Isaac Sim 4.5. The installed BEHAVIOR code requires
Isaac Sim 5.1. The Vega asset and updated collision exclusions are installed; the
check above should print that every prerequisite was found.

Start the simulator in one terminal:

```bash
YIXUAN_UTILITIES_ROOT="$HOME/yixuan_utilities/src/yixuan_utilities" \
  ~/miniconda3/envs/behavior51/bin/python sim_eval/teleop_joystick.py \
  --task pickplace --namespace behavior --no-headless \
  --out "$HOME/Dexmate/data/behavior_pickplace"
```

Start the existing leader in another terminal:

```bash
PYTHONPATH="$PWD/src" ~/miniconda3/envs/dexmate2/bin/python \
  scripts/wbc_joystick_leader.py --namespace behavior --sim-cameras --hud-cameras head_left_rgb
```

Open the leader's HTTPS WebXR address on the Quest, with the same certificate and
controller orientation calibration used on hardware. The simulator publishes head RGB
and, with `--camera-mode all`, left/right wrist RGB into the `behavior` namespace. `--sim-cameras`
subscribes directly to those streams for the existing headset HUD and avoids the
hardware camera client. The HUD also receives the normal follower status.

Controls are unchanged:

- Hold the right grip for calibration; complete the leader's alignment gate to engage.
- Move the controllers to move the hands. Index triggers close their grippers.
- Right thumbstick drives forward/back or strafes; left thumbstick X turns.
- Left **X** stops motion, saves the take, and returns to the static stage.
- Left index trigger while static homes the robot. Objects stay where you left them.
- Left **Y** stops motion, saves the take, and exits. Ctrl-C or errors preserve an
  active take as `.hdf5.partial`, rather than marking an interrupted take complete.

Use a fresh simulator process/seed to reset task objects for the next attempt.
Re-engaging within a process starts another numbered take of the current scene.
The task's scripted expert is never run. The initial implementation accepts
single-stage `sim_eval` tasks; it rejects multi-stage tasks rather than inventing
progress labels for human actions. `--task none --scene empty` gives an empty
floor for control bring-up. `--no-headless` opens the Isaac viewer.

For `pickplace` in `Rs_int`, the default furniture arrangement approximates
`/data/Dexmate/wm_data/episode_7.hdf5`: a bookshelf in front of the robot, a
table beside it on the robot's right, a table on the left side of the room,
and a tall open rack on the right side. There are no chairs. The apple and
bowl start on the right-hand table; drive and turn toward it to collect a take.
Only the living/dining floor and required building shell come from the original
scene. Use `--full-scene` to restore the original furnished scene and task spawn.
Dimensions and positions are editable in `tasks/episode7_room.py`; they are
approximate, not calibrated from the recording. Episodes record the layout ID.
The viewer renders at up to 30 Hz (`--viewer-rate 15` reduces that further);
camera capture remains at the requested recording/HUD rate. Control and physics
still advance at 100/200 Hz in simulation time; actual wall-clock speed depends
on the machine. Camera observations are no longer read and discarded every tick.

The default `--camera-mode head` creates only the head RGB sensor (no depth
render product) and shows that camera in the main Isaac viewport. Wrist sensors
and the third-person sensor are not created. Recorded HDF5 files omit wrist and
`head_depth` datasets and identify the camera mode in metadata; use a head-only
RGB training configuration for these takes. `--camera-mode all` restores depth,
wrists, and the previous third-person view at higher cost.

Save and exit with the leader's X/Y controls. Isaac's toolbar Stop invalidates
the articulation; the collector now reports that condition and closes the RPC
connection so an active recording can be finalized as incomplete without hanging.
Previously completed episodes remain unchanged.

## Control and recording semantics

For the pillow/basket/towel sequence inspired by `/data/Dexmate/lh_data/episode_0.hdf5`:

```bash
~/miniconda3/envs/behavior51/bin/python sim_eval/teleop_joystick.py \
  --task lh_demo --namespace behavior --no-headless \
  --out /data/Dexmate/behavior_lh_data
```

Use the same joystick leader with `--namespace behavior --sim-cameras`.
The pillow and basket start side by side on the right table, one movable chair
stands in front of it, and the folded towel rests on the right stand. The robot
starts facing the table. Pillow and towel use rigid proxies, not cloth dynamics.
X saves a take; Y saves and exits. Human long-horizon takes omit automatic stage
labels and set `success_evaluated=false`; annotate stages/outcome separately.
The object layout is in `tasks/lh_demo_room.py`. `--task pickplace` is unchanged.

The solver excludes the base, so hand/head targets stay in the **current robot
base frame** as the joystick moves it. The base shaper receives measured simulation
pose, re-zeroed at each engage. All gains and rates come from `wbik.yaml`, including
the joystick head policy (currently **fixed**), the torso lock, and fixed direction
speeds (currently **0.18 m/s** translation and **0.30 rad/s** yaw). Joint-step clamps
and the arm command low-pass filter use the real follower's defaults. Actuation is
through the existing OmniGibson joint/base controllers; simulated motor/contact
dynamics are not a measured model of the hardware's velocity feedforward or swerve.

Each take is streamed to `OUT/raw_data/episode_N.hdf5` at `--record-rate` (10 Hz by
default) while engaged and not held. Existing files are not overwritten. With
`--camera-mode all` it contains three RGB images, head optical depth in uint16
millimetres, and head intrinsics. With the default `--camera-mode head` it
contains only head RGB plus intrinsics (no depth or wrists). Every take also has:

- Measured torso/arm/head joints, grippers, body twist, and engage-relative base pose.
- Effective hand/head targets, trigger values, actually sent joint/base commands,
  and the shaper's pre-controller joystick intent in `action/chassis/intent_body`.
- Source/receive wall timestamps, recording wall timestamps, and `sim_time_s`.
- Pre-action task object positions, task/scene/seed metadata, and the terminal
  task-success result. An operator-ended take is retained even when success is false.

The raw schema is `omniteleop_joystick_sim_raw/v1`, with the existing 32-D joystick
policy-action contract. Canonical `obs/base/pose` is explicitly `sim_ground_truth`,
with an identical `obs/base/pose_sim` diagnostic stream. The HDF5 porter recognizes
this schema and checks its fixed-speed intent and ground-truth pose declaration;
hardware camera-clock/Modbus provenance is not fabricated for simulator frames.

Rendering and physics run synchronously with control. If the machine cannot sustain
the configured 100 Hz, simulation time advances slower than wall time; both clocks
are retained in the take. Check actual rendering/control performance before collecting
training data. The asset was live-tested on this workstation's RTX 4090 in both an
empty scene and the actual `Rs_int` pick-place setup.

To repeat the non-recording validation:

```bash
YIXUAN_UTILITIES_ROOT="$HOME/yixuan_utilities/src/yixuan_utilities" \
  ~/miniconda3/envs/behavior51/bin/python sim_eval/teleop_joystick.py \
  --task pickplace --smoke-test --smoke-output /tmp/vega_pickplace_smoke
```

CPU integration tests:

```bash
PYTHONPATH=src:sim_eval OMNIGIBSON_HEADLESS=1 \
  ~/miniconda3/envs/dexmate2/bin/python -m pytest -q \
  sim_eval/test_joystick_teleop.py sim_eval/test_robot_model.py
```
