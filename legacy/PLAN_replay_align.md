# WBC Recorded-Episode Replay + Reference-Pose Alignment for `wbc_policy_rollout.py`

**Goal:** Extend `scripts/wbc_policy_rollout.py` with two independent additions on top of the
existing live-policy path:

1. A **recorded-episode replay mode** (`--replay-episode FILE`) where the SOURCE of each tick's
   EEF/head/gripper target is a static recording. It drives the robot through the exact same
   actuation path a live policy rollout uses, including `HardwareDriver.actuate()`'s existing
   closed-loop base PD (`_drive_base()` tracks `result.base_pose`/`result.base_twist` against
   live `OdometryThread` feedback every tick, unconditionally) and all other hardware safety/
   tracking layers (joint-step clamp, e-stop, hold) — none of that is open-loop, none of it
   changes, and none of it needs to be added or reimplemented for replay. Real-time playback
   only.
2. A **reference-pose alignment step** (`--align-reference FILE`) shared by BOTH the live-policy
   path and the replay path: after engage, the robot autonomously drives its arms to a fixed
   reference EEF pose (read from a small reference episode HDF5, e.g.
   `/home/yixuan/Dexmate/data/raw_data/reference.hdf5`), then prompts the operator to press Enter
   before the real replay/rollout actually starts. This makes every take (recorded or replayed)
   start from the same physical arm configuration, with the head already at its nominal pose.

Read before implementing: `PLAN.md` (WBC Conventions section), `scripts/wbc_policy_rollout.py`,
`scripts/wbc_vr_robot.py` (`record_tick`, `engage_reset`, `_build_ik`, the 100 Hz loop around
line 1386-1403), `src/omniteleop/wbc_stream.py` (`TargetInterpolator`, `HeadTargetLowPassFilter`,
`HeadTargetPlanarDeadbandFilter`), `scripts/wbc_vr_leader.py` (`--align-reference` CLI + how it's
wired), and `src/omniteleop/leader/wbc_reference_alignment.py` (`load_reference_ee_poses`,
`ReferenceAlignmentGate`, `pose_rotation_error_deg`, the tolerance constants). Do **not** copy
`wbc_record.ReplaySource` (see Task 1) — it is a different schema for a different consumer.

---

## Task 1: Recorded-episode replay source — real-time only, own schema (no `--speed`)

**Do not reuse or subclass `omniteleop.wbc_record.ReplaySource`.** That class replays the
`wbc_vr_stream/v1` `VRJointData` schema written by `wbc_vr_leader.py`'s `--debug-vr` /
`wbc_vr_robot.py --replay` path (raw pre-WBC leader joints), and its speed-scaled-clock design
exists specifically so a human can slow down a VR session for inspection. That is the wrong
schema and the wrong use case here: this task replays the REAL per-episode recorder schema
written by `wbc_vr_robot.py --record` (`action/eef/{left,right}`, `action/head`,
`action/gripper/{left,right}`, `timestamp_ns`, `obs/*`) at real-time (1x) speed — there is no
`--speed` flag for this mode.

- [x] **Step 1: Write a fresh, independent reader/scheduler** (new small class, e.g.
  `RecordedEpisodeSource` in `scripts/wbc_policy_rollout.py`, or
  `src/omniteleop/wbc_episode_replay.py` if that's cleaner):
  - Loads `action/eef/left`, `action/eef/right`, `action/head`, `action/gripper/left`,
    `action/gripper/right`, `timestamp_ns` from the given `episode_*.hdf5`. Validate the same way
    `port_wbc_mobile_hdf5.read_required_array` / `_validate_target_batch` do: RuntimeError naming
    the missing dataset path — never silently substitute or interpolate over missing data.
  - On `start()`, anchor a wall clock. `.advance(now)` (or similar) releases frame `t` once
    `timestamp_ns[t] - timestamp_ns[0]` seconds of REAL time have elapsed since `start()` — no
    scaling factor anywhere in this class.
  - Exposes, once a frame is released: `left (4,4)`, `right (4,4)`, `head (4,4)`,
    `grip_left (float)`, `grip_right (float)`, and `segment_duration` = the raw (unscaled)
    inter-frame timestamp gap, for `interp.push(..., duration=segment_duration)` — this keeps the
    interpolation glide spanning the actual recorded inter-command interval (see
    `TargetInterpolator`'s docstring on why `duration` must match the real inter-arrival gap).
  - `done` property once the last frame has been released.

- [x] **Step 2: Tests** (`tests/test_wbc_policy_rollout_replay.py`, TDD — write first):
  - Raises `RuntimeError` naming the missing dataset when `action/eef/left`, `action/eef/right`,
    `action/head`, `action/gripper/left`, or `action/gripper/right` is absent from a synthetic
    episode HDF5.
  - Given a synthetic 3-frame episode with known `timestamp_ns`, frame `t` is not released before
    `timestamp_ns[t] - timestamp_ns[0]` real seconds have elapsed (inject a fake clock), and
    `segment_duration` equals the raw inter-frame gap (no division by any speed).
  - Gripper values pulled from the source are bit-identical to the raw `action/gripper/*` arrays
    (no binarization, no clipping — replay must send exactly what was recorded).

## Task 2: Head-target correctness in replay (unchanged from prior design — do not lose this)

This remains the crux correctness requirement for Task 1's replay loop:

- `action/head` in the raw episode HDF5 is recorded **verbatim POST-`head_lpf.filter()` AND
  POST-`head_deadband.filter()`** (`wbc_vr_robot.py:1336-1338`, `1543`) — the exact value passed to
  `ik.solve(head_target=...)` at record time.
- The live policy branch's 100 Hz loop (`wbc_policy_rollout.py:379-381`) always runs a fresh
  interpolator output through `head_lpf.filter()` then `head_deadband.filter()` (both stateful:
  the LPF is a first-order IIR lag, the deadband holds a planar anchor) before `ik.solve()`.
- **In the replay branch, skip `head_lpf.filter()` and `head_deadband.filter()` entirely** — feed
  `interp.at(now)`'s head component straight into `ik.solve(head_target=...)`. Re-running an
  already-filtered signal through both filters again would double-process it (extra lag, a
  deadband applied to already-suppressed noise) and the replayed motion would NOT match the
  recording.
- `action/eef/{left,right}` and `action/gripper/{left,right}` were never filtered — nothing to
  skip for them beyond what's below.
- `TargetInterpolator` STAYS in the replay path for all three streams — it's stateless per segment
  (pure lerp/slerp) so replaying the recorded samples through it reconstructs the same continuous
  100 Hz glide the robot originally tracked. Only the stateful filters must not be re-run.
- `HardwareDriver.actuate()`'s clamp/hold/safety logic (per-tick joint-step clamp, e-stop,
  `extra_hold`, base shaping) stays fully active in replay — these are hardware safety layers, not
  target postprocessing, and must never be bypassed. In particular `actuate()` unconditionally
  calls `_drive_base()` (`wbc_vr_robot.py:909`), which is a **closed-loop** base PD controller:
  `base_cl.pd_twist(result.base_pose, result.base_twist, snap["pose"], ...)`.
  Replay/rollout call `driver.actuate(result, ...)`
  unchanged, so this closed loop is inherited automatically; there is nothing to add here.

- [x] **Regression test**: spy/mock `HeadTargetLowPassFilter.filter` and
  `HeadTargetPlanarDeadbandFilter.filter`; assert they are **never called** when driving from
  `RecordedEpisodeSource`, and **are** called (existing, unchanged behavior) when driving from a
  live policy's `select_action()` output. This protects against a future refactor accidentally
  reintroducing double-filtering or removing filtering from the live path.

## Task 3: Replay still builds full observation + camera frames, and logs them

Unlike an earlier draft of this plan, the replay branch must NOT skip observation assembly. At
every replay tick, still call `driver._grab_head_images()`, `driver._grab_wrist_image()`, and
`_build_state(driver, fk, state_frame="base")` — exactly as the live-policy branch does — even
though the replayed action comes from `RecordedEpisodeSource`, not from `policy.select_action()`.
This exercises the full camera/state pipeline during replay (not just actuation) and produces a
complete, directly-comparable `policy_io/episode_N.hdf5` log (state, images-derived fields,
targets, base pose, timing) against the original recording. `select_action()` / `_PolicyBundle`
are the only things replay skips — no `--policy-path` is needed or accepted together with
`--replay-episode`.

- `io_log.record({...})` in replay should carry the same fields the live branch logs (`t`,
  `timestamp_ns`, `state`, `target`, `base_pose`); for the `"action"` field, encode the replayed
  targets with `mat_to_pos6d` (+ the two gripper floats) into the same 29-D `ACTION_AXES` layout a
  policy would have produced, so replay and live logs are directly diffable; `inference_s` can be
  `0.0` (no model ran).

### `--save-dir` default depends on mode

- [x] Add `DEFAULT_REPLAY_SAVE_DIR = str(Path("~/Dexmate/data/replay_raw_data").expanduser())`
  (resolves to `/home/yixuan/Dexmate/data/replay_raw_data`) alongside the existing
  `DEFAULT_ROLLOUT_SAVE_DIR`. When `--replay-episode` is given, `--save-dir` defaults to
  `DEFAULT_REPLAY_SAVE_DIR`; when `--policy-path` is given, it defaults to
  `DEFAULT_ROLLOUT_SAVE_DIR` (unchanged). Implement this as a post-`parse_args()` default
  resolution (only apply the mode-specific default when the user didn't pass `--save-dir`
  explicitly — i.e. `default=None` on the argparse flag, then fill in after checking which mode
  was selected), matching this script's existing pattern of binding tunables onto `args` after
  parsing.

## Task 4: `--align-reference` — shared by replay AND live-policy rollout

Mirrors `wbc_vr_leader.py --align-reference`'s reference-loading (`load_reference_ee_poses` from
`omniteleop.leader.wbc_reference_alignment` — reuse it directly, do not reimplement HDF5 parsing),
but the *gating mechanism* must differ: `wbc_vr_leader.py`'s `ReferenceAlignmentGate` is a PASSIVE
gate that waits for a HUMAN operator to manually drive the VR controllers until the live targets
match the reference. `wbc_policy_rollout.py` has no human operator driving targets — instead the
robot must ACTIVELY glide its own arms to the reference pose, then hand control to a human
Enter-press before doing anything else.

- [x] **Step 1: CLI** — add `--align-reference FILE` (default `None`, same
  `optional_reference_path()` normalization `wbc_vr_leader.py` uses so `"none"`/`"null"`/empty all
  mean disabled) to `wbc_policy_rollout.py`'s argparse. Applies identically regardless of whether
  `--policy-path` or `--replay-episode` was given.

- [x] **Step 2: Sequencing** — insert the alignment step AFTER engage
  (`ik.reset()` + `interp.reset(...)` + `driver.engage_reset(...)`, i.e. after the world frame is
  established) and AFTER the existing wait-for-first-FC03-gripper-reply loop, but BEFORE the main
  policy/replay 100 Hz loop starts. Concretely, right where `_run_rollout()` currently prints
  `"[wbc_policy_rollout] engaged: ..."` and starts the tick loop:
  1. If `--align-reference` is set: load `left_ref, right_ref = load_reference_ee_poses(path, ik)`
     (reuse verbatim; do not add a separate head reference). Hold the head target at `head_current`
     = `head0`, the pose captured right after the just-completed `ik.reset()`. Do not treat this as
     "skipping" head alignment: `head0 = ik.frame_pose(fk.head_frame)` immediately post-`ik.reset()`
     is, bit-for-bit, the same nominal-posture computation `wbc_vr_leader.py` uses for its own
     nominal-head gate (`self.T_base_head = ik.frame_pose(HEAD_FRAME)` right after its own
     `ik.reset()`, `wbc_vr_leader.py:425`) — both scripts reset to the same nominal `q`. So holding
     `head_current` for the whole align glide already IS aligning the head to nominal; there is no
     separate reference to look up or converge to. (This holds only for the align step itself —
     once the main replay/rollout loop starts, the head goes back to being driven by
     `action/head`/the policy's head output, same as any other DOF; it is not held at nominal for
     the whole episode.)
  2. Push `(left_ref, right_ref, head_current)` into `interp` with a glide duration long enough to
     be reachable (DEFAULT_ALIGN_SECONDS=3.0), then keep running the
     SAME 100 Hz `interp.at(now) -> ik.solve() -> driver.actuate()` loop used elsewhere in this
     script (head_lpf/deadband stay ACTIVE here — this is a real live glide, not replay) until
     both position and rotation error (reuse `pose_rotation_error_deg` and the
     `REFERENCE_ALIGN_POS_TOL_MM` / `REFERENCE_ALIGN_ROT_TOL_DEG` / `REFERENCE_ALIGN_STABLE_S`
     constants from `wbc_reference_alignment.py` for consistency with the leader) hold within
     tolerance for the stable window. 
  3. Hard-abort on a borderline convergence check after timeout.
  4. Print `"[wbc_policy_rollout] at reference pose. Press Enter to actually start "
     "{replay,rollout} online ..."` (word choice depends on mode) and block on `input()`.
  -  `--align-reference` is required for both live rollout/playback.

- [x] **Step 3: Tests** (`tests/test_wbc_policy_rollout_align.py`):
  - `load_reference_ee_poses` is called with the given path (mock/stub) and its (left, right)
    result is what gets pushed into `interp` as the glide target, unchanged.
  - The alignment loop keeps `head_lpf.filter()` / `head_deadband.filter()` ACTIVE (contrast with
    Task 2's replay regression test, where they must be skipped) — this loop is a live glide, not
    a replay of a filtered value.
  - Convergence check: given a fake `ik.solve()`/pose readout that starts far from the reference
    and converges within a few ticks, the loop exits once error is within tolerance for the stable
    window, not before.

## Task 5: CLI summary for `scripts/wbc_policy_rollout.py`

- [x] `--policy-path` becomes optional; add `--replay-episode FILE`. Exactly one of
  `--policy-path` / `--replay-episode` must be given — `parser.error(...)` otherwise.
- [x] Remove `--speed` entirely (delete the flag, the `args.speed = mod.DEFAULT_SPEED` binding at
  the bottom of `main()`, and any speed-scaling math) — replay is always real-time now.
- [x] Add `--align-reference FILE` (default `None`).
- [x] `--save-dir` default becomes mode-dependent (Task 3): `DEFAULT_REPLAY_SAVE_DIR` for
  `--replay-episode`, `DEFAULT_ROLLOUT_SAVE_DIR` (unchanged) for `--policy-path`.

## Verification

```bash
/home/yixuan/miniforge3/envs/dexmate/bin/python -m pytest \
  tests/test_wbc_policy_rollout_replay.py \
  tests/test_wbc_policy_rollout_align.py \
  tests/test_wbc_policy_rollout_decode.py \
  -q
/home/yixuan/miniforge3/envs/dexmate/bin/python -m py_compile scripts/wbc_policy_rollout.py
```

Then, on the real robot:

```bash
# Replay sanity check, starting from a known reference pose every time:
python scripts/wbc_policy_rollout.py \
  --replay-episode ~/Dexmate/data/raw_data/episode_0.hdf5 \
  --align-reference ~/Dexmate/data/raw_data/reference.hdf5

# Live rollout, same reference-pose alignment before it actually starts:
python scripts/wbc_policy_rollout.py \
  --policy-path /path/to/checkpoint \
  --align-reference ~/Dexmate/data/raw_data/reference.hdf5
```

Confirm: (1) the robot glides to the reference pose and waits for Enter before doing anything
else, in both modes; (2) the replayed episode (now saved under
`~/Dexmate/data/replay_raw_data/episode_N.hdf5` and `replay_raw_data/policy_io/episode_N.hdf5`)
reproduces the source recording's motion, including head behavior (no extra lag from
double-filtering); (3) `policy_io/episode_N.hdf5` contains full `observation.state` + the
camera-derived fields even though replay never ran a policy.

## Summary to give back to the user afterward

- What changed vs. the original replay-only design (removed `--speed`/`wbc_record.ReplaySource`
  reuse, added `--align-reference`, replay now builds full state+camera observations).
- Confirm the Step 2.3 timeout-behavior decision (warn-and-proceed vs. hard-abort) was resolved
  one way or the other, and why.
- Any deviations from this plan and why.
