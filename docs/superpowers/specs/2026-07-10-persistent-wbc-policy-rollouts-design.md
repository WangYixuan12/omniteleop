# Persistent WBC Policy Rollouts

## Goal

Allow `scripts/wbc_policy_rollout.py` to run multiple rollout episodes without
restarting the process, reloading the policy, or reconnecting to the robot. An
active rollout ends when `--max-seconds` is reached, a replay completes, or the
operator presses Ctrl-C. The operator can then press Enter to prepare and start
another rollout, or press Ctrl-C at the idle prompt to shut down the process.

This behavior applies to the shared policy/replay loop used by
`scripts/wbc_maniflow_rollout.py` as well as the LeRobot image-policy entrypoint.

## Session and Episode Lifetimes

`_run_rollout` will own one persistent session. It loads and retains the policy,
IK model, FK helper, hardware driver, cameras, odometry thread, and reusable
episode recorders for the life of the process. Hardware shutdown remains in the
outermost `finally` block and runs exactly once.

Inside that session, an episode loop will run the existing preparation and
control flow once per rollout. Episode-scoped state must be new or explicitly
reset on every iteration:

- action schedule buffer, stop event, inference worker, timers, and counters;
- target interpolator and head filters;
- IK world anchor, odometry origin, driver command history, and policy queues;
- replay source cursor, when `--replay-episode` is used;
- main episode recording and `policy_io` recording.

The first episode preserves the existing `--auto-start` behavior. Every later
episode waits at an idle prompt regardless of `--auto-start`:

```text
Press Enter to home and start the next rollout (Ctrl-C exits) ...
```

Pressing Enter first reuses the driver's startup gripper activation path for both
grippers. That path issues the existing Robotiq reset/activate sequence, after
which the driver polls its existing FC03 status monitors until both normalized
achieved positions are at most `0.05` (`0.0` is fully open). Only then does it
run the nominal homing routine
while the process and hardware connection remain live. If either gripper does not
confirm open within five seconds, motion is stopped and the session terminates
instead of homing with a potentially held object. The episode then performs the
same engage reset and optional reference alignment as the first rollout. The
initial episode does not repeat the activation or nominal home already performed
by `HardwareDriver` startup.

## Position-Conditioned Rollouts

For a policy that requires `--position-condition`, every episode recomputes its
condition. After the episode's engage reset and reference alignment, and before
recording or inference starts, the script will:

1. capture a new live head-camera frame at the rollout start pose;
2. rerun SceneDiff against the configured reference;
3. resolve the object order using the configured matching mode;
4. call `policy.set_env_state` with the new result;
5. only then start the main recorder, `policy_io` recorder, and inference worker.

SceneDiff output continues to use the main recorder's pending `episode_N` ID, so
each condition directory is paired with the rollout that consumes it. Because
the recorder increments its ID after every successful save, later iterations
naturally target the next episode directory.

## Ending an Episode

The inner episode boundary handles these normal completion reasons:

- `--max-seconds` elapsed;
- the replay source released its final frame and completed its final glide;
- Ctrl-C occurred during episode preparation or active motion.

All three take the same fail-safe cleanup path:

1. signal the inference worker to stop;
2. stop all robot motion immediately;
3. join the worker before touching the log it writes;
4. stop and fully save the main rollout episode through
   `HardwareDriver.stop_recording_episode()`;
5. stop and fully save the corresponding `policy_io` episode;
6. return to the idle prompt.

If a stopped inference worker cannot join within the existing timeout, the
session must not start another episode against a still-running worker. That case,
worker failures, camera failures, SceneDiff failures, and other unexpected
exceptions remain terminal: motion stops, any active recordings are flushed,
and the outer session closes the hardware.

Ctrl-C has state-dependent meaning. During preparation or motion it ends only
the current episode. While blocked at the idle prompt it exits the process. This
provides the same separate "end episode" and "terminal stop" semantics that the
VR leader exposes with its X and Y buttons.

## Recording and Numbering

Both existing `EpisodeRecorder` objects are reusable: `start()` clears their
in-memory frames, and `stop()` advances `episode_id` after a nonempty save. Each
new rollout starts both recorders together after all preparation succeeds.
Episode cleanup waits for both asynchronous HDF5 saves before offering the next
prompt, preventing a later episode from racing a prior save.

The driver's per-episode camera freshness and record cadence state will be reset
when the main recorder starts, matching `wbc_vr_robot.py`'s multi-episode start
behavior. A preparation interrupted before recording creates no empty HDF5 and
does not consume an episode ID.

## Testing

Tests will exercise the lifecycle without robot hardware by using injected fakes
at the episode/session boundary. They will verify:

- timeout and Ctrl-C both stop motion, stop/join the worker, and flush recordings;
- Ctrl-C in an active episode returns to the idle decision, while Ctrl-C at the
  idle prompt exits;
- two rollouts reuse the same persistent driver and policy but receive fresh
  per-episode scheduler, worker, reset calls, and recorder starts;
- the second rollout homes before engage and the first does not double-home;
- the second rollout reuses startup gripper activation, confirms both grippers
  open, and only then homes; confirmation timeout prevents homing;
- replay restarts from frame zero for each rollout;
- position-conditioned policies rerun SceneDiff and set a fresh environment
  state before each recorder/worker start, using successive pending episode IDs;
- a worker join timeout or unexpected preparation/runtime failure prevents a
  subsequent rollout and still triggers one final hardware shutdown.

Existing rollout decoding, alignment, scheduling, instrumentation, and replay
tests will remain green. The implementation will stay within
`scripts/wbc_policy_rollout.py` plus focused tests; no unrelated refactoring is
in scope.
