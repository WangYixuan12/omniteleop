# WBC Mobile Policy Dataset And Rollout Implementation Plan

> **For agentic workers:** Implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build new WBC-specific dataset porting and rollout scripts for mobile manipulation policies that run at 10 Hz while the WBC IK/control loop runs at 100 Hz.

**Architecture:** Keep the existing tabletop porter (`/home/yixuan/lerobot_original/examples/port_datasets/port_dexmate_hdf5.py`) and arm-only rollout (`src/omniteleop/follower/policy_rollout.py`) as references only. Add a WBC policy format helper, a WBC raw-HDF5 to LeRobot porter, and a WBC policy rollout script. The policy **action is world-frame at 10 Hz** — the verbatim targets given to `ik.solve()` — so rollout pushes policy actions straight into the existing `TargetInterpolator`/WBC path with **no decode step**. The policy **state is base-frame achieved FK plus explicit odometry base-pose dims**.

**Tech Stack:** Python, NumPy, h5py, Pinocchio/Pink through `VegaWholeBodyIK`, LeRobotDataset, dexcontrol hardware APIs, existing WBC modules (`whole_body_ik.py`, `wbc_stream.py`, `wbc_teleop.py`, `wbc_vr_robot.py`).

---



## Conventions

The policy-facing vector is 10 Hz. WBC IK/control is 100 Hz.

```text
observation.state (32,):
  left_eef:   pos3 + rot6 + gripper      base-frame achieved
  right_eef:  pos3 + rot6 + gripper      base-frame achieved
  head:       pos3 + rot6                base-frame achieved (zed_depth_frame)
  base:       x, y, yaw                  measured odometry, engage-origin world

action (29,):
  left_eef:   pos3 + rot6 + gripper      WORLD-frame target (verbatim ik.solve input)
  right_eef:  pos3 + rot6 + gripper      WORLD-frame target (verbatim ik.solve input)
  head:       pos3 + rot6                WORLD-frame target (verbatim ik.solve input)
```

Axis order:

```text
STATE_AXES (32):
left_tx,left_ty,left_tz,left_r00,left_r10,left_r20,left_r01,left_r11,left_r21,left_gripper,
right_tx,right_ty,right_tz,right_r00,right_r10,right_r20,right_r01,right_r11,right_r21,right_gripper,
head_tx,head_ty,head_tz,head_r00,head_r10,head_r20,head_r01,head_r11,head_r21,
base_x,base_y,base_yaw

ACTION_AXES (29): STATE_AXES[:29]
```

Frame convention:

```text
world frame = engage-origin frame: on the engage edge the follower calls ik.reset()
              AND OdometryThread.reset_origin(), so the IK solver world and the
              measured odometry pose share one frame zeroed at engage
              (scripts/wbc_vr_robot.py:1356-1365, 682).

observation.state.eef  = base_T_ee   from WBC FK on measured obs/joint, base at zero in q
observation.state.head = base_T_zed_depth_frame from the same FK
observation.state.base = obs/base/pose (measured swerve odometry x, y, yaw; yaw is
                         integrated continuously — no ±pi wrap)
action.eef             = world_T_{left,right}_target: the EXACT 4x4 passed to
                         ik.solve(left_target, right_target, ...) at the record tick
action.head            = world_T_head_target: the EXACT 4x4 passed as
                         ik.solve(head_target=...) — i.e. AFTER the head LPF and the
                         planar deadband (wbc_vr_robot.py:1386-1393). "Post-clamp"
                         head command = post-target-shaping solver input.
```

The world anchor for the policy comes from the three explicit `base_x, base_y, base_yaw` state dims (images are egocentric; without them a world-frame action
head would be ungrounded). Do NOT compose `world_T_base` into the state EEF/head
poses — the state stays base-frame by design.

Rollout decodes nothing. At each 10 Hz policy tick:

```python
targets = split_policy_action(action)   # pos6d_to_mat on the 3 blocks + 2 gripper floats
interp.push(targets["left"], targets["right"], targets["head"], now=now, duration=1.0 / cmd_rate)
```

and the 100 Hz WBC tick runs the SAME path as teleop (`wbc_vr_robot.py:1386-1393`):
interpolate, apply the head LPF + planar deadband to the interpolated head target
(protects the base from policy jitter exactly as it does from headset dither), then

```python
result = ik.solve(left_target, right_target, dt, head_target=head_target)
```

with `WBCConfig(head_mode="ik")` — never re-compose anything with the moving base:
the targets are already world-frame and never ride the base.

Head pose representation and frame:

```text
head frame       = zed_depth_frame (whole_body_ik.HEAD_FRAME): the frame the WBC head
                   FrameTask tracks in head_mode "ik", the frame solve_head aims in
                   "track" mode, the frame the leader's calibrated head_ee_pose targets,
                   and the frame vis_episode.py already uses for offline world_t_cam.
                   URDF: fixed to head_l3, optical convention (z = viewing axis);
                   zed_left_camera sits 11.5 mm away along the optical axis — only
                   relevant to pixel-precise backprojection in the calib sidecar,
                   not to the policy pose.
head position    = 3D translation
head orientation = 6D rotation (first two rotation-matrix columns)
```

Keep all 9 head dims even though the current `wbik.yaml` config makes two of them
passengers on the ACTION side: `head_world_position_cost: [1e4, 1e4, 0]` frees world
z and roll is unactuatable (left as orientation residual). On the STATE side head z
is genuinely meaningful (torso height moves the camera). Symmetric pos3+rot6 blocks
keep state/action layouts aligned with no per-dim slicing. Do not use Euler,
axis-angle, or rotation-vector head orientation.

Relative-action training note: `RelativeActionsProcessorStep` computes
`action -= state[..., :action_dim]` positionally. With world-frame actions and
base-frame state those subtractions cross frames, so for this dataset
`use_relative_actions` must stay `false` (the appended base dims 29..31 are beyond
`action_dim` and never enter the subtraction either way).

Gripper convention follows the tabletop porter:

```text
observation.state gripper = raw achieved FC03 value (NaN until the first FC03 reply —
                            the porter must drop leading NaN-gripper frames and raise
                            on a mid-take NaN)
action gripper            = clipped/binarized command
```

Do not use WBC FK from `result` or `action/joint/*` as the policy EEF/head action. WBC FK is useful for achieved state, camera calibration, and debug streams only.

Raw/demo file convention:

```text
episode_*.hdf5:
  action/joint/*      = values actually sent to the real robot after clamping (post-clamp
                        _dbg["sent_joints"]; Task 2)
  action/eef/{left,right} = required 10 Hz WORLD-frame policy EEF target, float32 (T,4,4)
  action/head         = required 10 Hz WORLD-frame zed_depth_frame target, float32 (T,4,4)
  action/base/pose    = result.base_pose (x, y, yaw) — the WBC solver's commanded base
                        pose in the same engage-origin world, float32 (T,3). Not a policy
                        field: recorded because it is free (already computed each solve)
                        and lets offline analysis derive either-anchor base-frame
                        variants and quantify IK-world vs odom-world tracking error.
  obs/base/pose       = measured odometry pose — REQUIRED for the state base dims
                        (recording therefore requires --enable base)

episode_*_debug.hdf5:
  debug/cmd_*    = WBC result.* before hardware joint clamp
  debug/sent_*   = actual post-clamp joint command
  debug/left_target, debug/right_target, debug/head_target = effective WBC world targets for inspection only
```

The porter must never substitute debug targets for missing main raw fields. If
`action/eef/{left,right}`, `action/head`, or `obs/base/pose` is absent from
`episode_*.hdf5`, raise an error and exit. Re-record with the updated recorder
instead of reading `episode_*_debug.hdf5`.

## File Structure

Create:

```text
src/omniteleop/wbc_policy_format.py
scripts/port_wbc_mobile_hdf5.py
scripts/wbc_policy_rollout.py
tests/test_wbc_policy_format.py
tests/test_port_wbc_mobile_hdf5.py
tests/test_wbc_policy_rollout_decode.py
```

Modify:

```text
scripts/wbc_vr_robot.py
tests/test_wbc_vr_robot_gripper_wrist.py
```

Do not modify the tabletop porter or existing arm-only rollout except for comments if absolutely necessary.

## Task 1: WBC Policy Format Helpers

**Files:**

- Create: `src/omniteleop/wbc_policy_format.py`
- Test: `tests/test_wbc_policy_format.py`

- [x] **Step 1: Write helper tests**

Create `tests/test_wbc_policy_format.py` with tests for:

```python
import numpy as np

from omniteleop.wbc_policy_format import (
    ACTION_AXES,
    STATE_AXES,
    base_pose_to_mat,
    mat_to_pos6d,
    pos6d_to_mat,
    transform_pose,
)


def test_axes_layout():
    assert len(STATE_AXES) == 32
    assert len(ACTION_AXES) == 29
    assert STATE_AXES[:29] == ACTION_AXES
    assert STATE_AXES[-3:] == ["base_x", "base_y", "base_yaw"]
    assert ACTION_AXES[-9:] == [
        "head_tx", "head_ty", "head_tz",
        "head_r00", "head_r10", "head_r20",
        "head_r01", "head_r11", "head_r21",
    ]


def test_pos6d_roundtrip_preserves_pose_columns():
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [0.2, -0.1, 0.4]
    T[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    v = mat_to_pos6d(T)
    out = pos6d_to_mat(v)
    np.testing.assert_allclose(out[:3, 3], T[:3, 3], atol=1e-7)
    np.testing.assert_allclose(out[:3, 0], T[:3, 0], atol=1e-7)
    np.testing.assert_allclose(out[:3, 1], T[:3, 1], atol=1e-7)


def test_base_pose_transform_between_world_and_base():
    world_T_base = base_pose_to_mat(np.array([1.0, 2.0, np.pi / 2]))
    base_T_target = np.eye(4)
    base_T_target[:3, 3] = [0.3, 0.0, 0.5]
    world_T_target = transform_pose(world_T_base, base_T_target)
    recovered = transform_pose(np.linalg.inv(world_T_base), world_T_target)
    np.testing.assert_allclose(recovered, base_T_target, atol=1e-7)
```

- [x] **Step 2: Run failing tests**

Run:

```bash
/home/yixuan/miniforge3/envs/dexmate/bin/python -m pytest tests/test_wbc_policy_format.py -q
```

Expected: import failure because `omniteleop.wbc_policy_format` does not exist.

- [x] **Step 3: Implement helpers**

Implement `src/omniteleop/wbc_policy_format.py` with:

```python
from __future__ import annotations

import numpy as np

LEFT_EEF_AXES = [
    "left_tx", "left_ty", "left_tz",
    "left_r00", "left_r10", "left_r20",
    "left_r01", "left_r11", "left_r21",
    "left_gripper",
]
RIGHT_EEF_AXES = [
    "right_tx", "right_ty", "right_tz",
    "right_r00", "right_r10", "right_r20",
    "right_r01", "right_r11", "right_r21",
    "right_gripper",
]
HEAD_AXES = [
    "head_tx", "head_ty", "head_tz",
    "head_r00", "head_r10", "head_r20",
    "head_r01", "head_r11", "head_r21",
]
BASE_AXES = ["base_x", "base_y", "base_yaw"]
ACTION_AXES = LEFT_EEF_AXES + RIGHT_EEF_AXES + HEAD_AXES
STATE_AXES = ACTION_AXES + BASE_AXES


def mat_to_pos6d(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"expected (4,4), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("pose contains non-finite values")
    return np.concatenate([arr[:3, 3], arr[:3, 0], arr[:3, 1]]).astype(np.float32)


def pos6d_to_mat(pos6d: np.ndarray) -> np.ndarray:
    v = np.asarray(pos6d, dtype=np.float64).reshape(-1)
    if v.shape != (9,):
        raise ValueError(f"expected 9 values, got {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError("pos6d contains non-finite values")
    a1, a2 = v[3:6], v[6:9]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = v[:3]
    out[:3, :3] = np.stack([b1, b2, b3], axis=1)
    return out


def base_pose_to_mat(base_xyyaw: np.ndarray) -> np.ndarray:
    b = np.asarray(base_xyyaw, dtype=np.float64).reshape(-1)
    if b.shape != (3,):
        raise ValueError(f"expected base pose (3,), got {b.shape}")
    c, s = np.cos(b[2]), np.sin(b[2])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:2, 3] = b[:2]
    return T


def transform_pose(left_T_right: np.ndarray, right_T_obj: np.ndarray) -> np.ndarray:
    a = np.asarray(left_T_right, dtype=np.float64)
    b = np.asarray(right_T_obj, dtype=np.float64)
    if a.shape != (4, 4) or b.shape != (4, 4):
        raise ValueError(f"expected two (4,4) poses, got {a.shape} and {b.shape}")
    return a @ b
```

- [x] **Step 4: Run helper tests**

Run:

```bash
/home/yixuan/miniforge3/envs/dexmate/bin/python -m pytest tests/test_wbc_policy_format.py -q
```

Expected: all tests pass.

## Task 2: Update Real-Robot Recorder Semantics

**Files:**

- Modify: `scripts/wbc_vr_robot.py`
- Test: `tests/test_wbc_vr_robot_gripper_wrist.py`

- [x] **Step 1: Add tests for post-clamp action recording**

Extend `tests/test_wbc_vr_robot_gripper_wrist.py` so the fake driver has
`_dbg["sent_joints"]` and verifies `record_tick()` writes `action/joint/head`
from `sent_joints["head"]`, not `result.head`. Add a second test that proves
recording raises when sent joints are missing.

Use this test body:

```python
def test_record_tick_prefers_post_clamp_sent_joints_for_action():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)
    driver._dbg = {
        "sent_joints": {
            "torso": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "left_arm": np.full(7, 0.4, dtype=np.float32),
            "right_arm": np.full(7, 0.5, dtype=np.float32),
            "head": np.array([0.6, 0.7, 0.8], dtype=np.float32),
        }
    }
    result = types.SimpleNamespace(
        torso=np.zeros(3, dtype=np.float32),
        left_arm=np.zeros(7, dtype=np.float32),
        right_arm=np.zeros(7, dtype=np.float32),
        head=np.zeros(3, dtype=np.float32),
    )

    mod.HardwareDriver.record_tick(driver, result, hold=False, now=100.0)

    frame = captured[0]
    np.testing.assert_allclose(frame["action"]["joint"]["head"], [0.6, 0.7, 0.8])
    np.testing.assert_allclose(frame["action"]["joint"]["torso"], [0.1, 0.2, 0.3])


def test_record_tick_requires_post_clamp_sent_joints_for_action():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)
    driver._dbg = {}
    result = types.SimpleNamespace(
        torso=np.zeros(3, dtype=np.float32),
        left_arm=np.zeros(7, dtype=np.float32),
        right_arm=np.zeros(7, dtype=np.float32),
        head=np.zeros(3, dtype=np.float32),
    )

    with pytest.raises(RuntimeError, match="sent_joints"):
        mod.HardwareDriver.record_tick(driver, result, hold=False, now=100.0)
```

(Adapt these calls to the Task 3 signature once Task 3 lands — pass valid world
targets; the two tasks may be implemented together.)

- [x] **Step 2: Run failing test**

Run:

```bash
/home/yixuan/miniforge3/envs/dexmate/bin/python -m pytest tests/test_wbc_vr_robot_gripper_wrist.py::test_record_tick_prefers_post_clamp_sent_joints_for_action -q
```

Expected: failures because `record_tick()` currently writes `result.*` and does
not require post-clamp sent joints.

- [x] **Step 3: Modify** `record_tick()`

In `scripts/wbc_vr_robot.py`, inside `record_tick()`, before assembling `frame`, read:

```python
sent_joints = self._dbg.get("sent_joints") if hasattr(self, "_dbg") else None
if not isinstance(sent_joints, dict):
    raise RuntimeError("record_tick requires post-clamp _dbg['sent_joints']")

def action_joint(name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name not in sent_joints:
        raise RuntimeError(f"record_tick missing post-clamp sent_joints[{name!r}]")
    arr = np.asarray(sent_joints[name], dtype=np.float32)
    if arr.shape != shape:
        raise RuntimeError(
            f"record_tick sent_joints[{name!r}] has shape {arr.shape}, expected {shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"record_tick sent_joints[{name!r}] contains non-finite values")
    return arr
```

Then write:

```python
"torso": action_joint("torso", (3,)),
"left_arm": action_joint("left_arm", (7,)),
"right_arm": action_joint("right_arm", (7,)),
"head": action_joint("head", (3,)),
```

Do not change gripper behavior; grippers already record clipped commands.

Notes verified against the current code: `actuate()` runs before `record_tick()`
in the loop and populates `_dbg["sent_joints"]` for every ENABLED group
(`wbc_vr_robot.py:734-756`); it is `None` on hold, and `record_tick()` skips hold
frames (line 1114) — so on recorded frames the dict exists, but a group missing
from `--enable` would be absent. That is exactly what the strict check must catch:
recording a policy take requires torso+arms+head enabled (see Task 3 Step 3 for
the startup validation).

- [x] **Step 4: Run recorder tests**

Run:

```bash
/home/yixuan/miniforge3/envs/dexmate/bin/python -m pytest tests/test_wbc_vr_robot_gripper_wrist.py -q
```

Expected: all tests pass.

## Task 3: Record Required 10 Hz World-Frame EEF And Head Targets In Main HDF5

**Files:**

- Modify: `scripts/wbc_vr_robot.py`
- Test: `tests/test_wbc_vr_robot_gripper_wrist.py`

- [x] **Step 1: Add a test for verbatim world-target recording**

Add a test that calls `record_tick()` with `left_target`, `right_target`, and
`head_target` 4x4 matrices plus a fake odom pose and a fake
`result.base_pose = [0.9, 1.9, np.pi / 2 - 0.05]`. The fake driver must include
valid `_dbg["sent_joints"]` for torso, arms, and head so strict action recording
passes. It should verify the targets are stored VERBATIM (world frame, no
composition):

```text
frame["action"]["eef"]["left"]  == left_target   (bitwise-equal after float32 cast)
frame["action"]["eef"]["right"] == right_target
frame["action"]["head"]         == head_target
frame["action"]["base"]["pose"] == result.base_pose
frame["obs"]["base"]["pose"]    == odom pose
```

Add two failure tests: `record_tick()` raises when the odometry pose is
unavailable (no `_odom`), and raises on a non-finite or non-(4,4) target.

- [x] **Step 2: Run failing test**

Run:

```bash
/home/yixuan/miniforge3/envs/dexmate/bin/python -m pytest tests/test_wbc_vr_robot_gripper_wrist.py::test_record_tick_writes_world_frame_policy_targets -q
```

Expected: failure because `record_tick()` does not accept or record EEF/head targets.

- [x] **Step 3: Extend** `record_tick()` **signature and validate**

Change:

```python
def record_tick(self, result, hold: bool, now: float) -> None:
```

to:

```python
def record_tick(
    self,
    result,
    hold: bool,
    now: float,
    left_target: np.ndarray,
    right_target: np.ndarray,
    head_target: np.ndarray,
) -> None:
```

Validate and store the targets VERBATIM (they are already world-frame solver
inputs; the head target is already post-LPF/post-deadband at the call site):

```python
left_target = np.asarray(left_target, dtype=np.float64)
right_target = np.asarray(right_target, dtype=np.float64)
head_target = np.asarray(head_target, dtype=np.float64)
if left_target.shape != (4, 4) or right_target.shape != (4, 4) or head_target.shape != (4, 4):
    raise RuntimeError(
        "record_tick expected 4x4 policy targets, "
        f"got left={left_target.shape}, right={right_target.shape}, head={head_target.shape}"
    )
if (
    not np.all(np.isfinite(left_target))
    or not np.all(np.isfinite(right_target))
    or not np.all(np.isfinite(head_target))
):
    raise RuntimeError("record_tick received non-finite policy target")
if self._odom is None:
    raise RuntimeError(
        "record_tick requires odometry (--enable base) so obs/base/pose anchors "
        "the world-frame action targets"
    )
```

Then add to `frame["action"]`:

```python
"eef": {
    "left": left_target.astype(np.float32),
    "right": right_target.astype(np.float32),
},
"head": head_target.astype(np.float32),
"base": {"pose": np.asarray(result.base_pose, dtype=np.float32)},
```

Also add a startup validation where `--record` is configured: recording requires
`--enable` to include torso, arms, head, AND base (fail fast at init, not on the
first recorded frame).

- [x] **Step 4: Pass targets from the WBC loop**

In the main loop, change:

```python
driver.record_tick(result, hold, now)
```

to:

```python
driver.record_tick(
    result,
    hold,
    now,
    left_target=left_target,
    right_target=right_target,
    head_target=head_target,
)
```

(`left_target`/`right_target`/`head_target` are in scope at the call site —
`wbc_vr_robot.py:1386-1403` — and `head_target` is the post-LPF/post-deadband
matrix that was passed to `ik.solve()`.)

- [x] **Step 5: Run recorder tests**

Run:

```bash
/home/yixuan/miniforge3/envs/dexmate/bin/python -m pytest tests/test_wbc_vr_robot_gripper_wrist.py -q
```

Expected: all tests pass.

## Task 4: WBC Mobile Dataset Porter

**Files:**

- Create: `scripts/port_wbc_mobile_hdf5.py`
- Test: `tests/test_port_wbc_mobile_hdf5.py`

- [x] **Step 1: Add unit tests with synthetic HDF5**

Ask me to run the code on real robot to collect real data.

Test that a pure conversion helper returns:

```text
observation.state.shape == (32,)
action.shape == (29,)
observation.state[20:29] == mat_to_pos6d(base_T_zed_depth_frame achieved from obs joints)
observation.state[29:32] == obs/base/pose
action[20:29] == mat_to_pos6d(action/head)          # verbatim, no composition
action[:9]    == mat_to_pos6d(action/eef/left)      # verbatim, no composition
```

Also test that `load_action_targets_world()` raises when `action/eef/*` or
`action/head` is absent, and that the porter raises when `obs/base/pose` is
absent. Test the NaN-gripper policy: leading frames with a NaN `obs/gripper/*`
are dropped; a NaN after the first finite value raises.
Do not create a synthetic debug HDF5 for this test; debug data is not an input to
the dataset porter.

- [x] **Step 2: Implement pure conversion helpers first**

In `scripts/port_wbc_mobile_hdf5.py`, implement these importable helpers with
the listed signatures:

```text
compute_frame_state_action(raw: h5py.File, t: int, fk: WBCPolicyFK, action_eef: Mapping[str, np.ndarray], action_head: np.ndarray) -> tuple[np.ndarray, np.ndarray]
load_action_targets_world(raw: h5py.File, t: int) -> tuple[dict[str, np.ndarray], np.ndarray]
```

`compute_frame_state_action()` must:

```text
1. Build base_T_left/right achieved using WBC FK with base set to zero and measured obs joints.
2. Build base_T_head achieved using WBC FK for zed_depth_frame with base set to zero and measured obs joints.
3. Read obs/base/pose[t] (x, y, yaw), validate finite (3,).
4. Return observation.state as [mat_to_pos6d(base_T_left), obs_gripper_left,
   mat_to_pos6d(base_T_right), obs_gripper_right, mat_to_pos6d(base_T_head),
   base_x, base_y, base_yaw].
5. Return action as [mat_to_pos6d(action_eef["left"]), action_gripper_left,
   mat_to_pos6d(action_eef["right"]), action_gripper_right,
   mat_to_pos6d(action_head)] — the world-frame matrices VERBATIM from raw.
```

Use `VegaWholeBodyIK(WBCConfig(enable_collision_avoidance=False))` for FK. Do not use `KinHelper`.

`load_action_targets_world()` must:

```text
1. Require raw action/eef/{left,right}; return those WORLD-frame 4x4 matrices verbatim.
2. Require raw action/head; return that WORLD-frame 4x4 matrix verbatim.
3. Raise RuntimeError with the missing dataset path when any required dataset is absent.
4. Raise RuntimeError on non-finite transforms, wrong shapes, or invalid frame index.
```

- [x] **Step 3: Implement** `WBCPolicyFK`

Inside the porter or in `src/omniteleop/wbc_policy_format.py`, implement a small class:

```python
from omniteleop.follower.whole_body_ik import (
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    TORSO_JOINTS,
    VegaWholeBodyIK,
    WBCConfig,
)
from omniteleop.wbc_robot_util import set_base_in_q


class WBCPolicyFK:
    def __init__(self):
        self.ik = VegaWholeBodyIK(WBCConfig(enable_collision_avoidance=False))

    def q_from_raw(self, torso, left_arm, right_arm, head, base_xyyaw=None):
        base = np.zeros(3, dtype=float) if base_xyyaw is None else np.asarray(base_xyyaw, dtype=float)
        q = set_base_in_q(self.ik.nominal_q(), base)
        groups = (
            (TORSO_JOINTS, torso),
            (LEFT_ARM_JOINTS, left_arm),
            (RIGHT_ARM_JOINTS, right_arm),
            (HEAD_JOINTS, head),
        )
        for names, values in groups:
            arr = np.asarray(values, dtype=float).reshape(len(names))
            for name, value in zip(names, arr, strict=True):
                q[self.ik._idx_q[name]] = float(value)  # noqa: SLF001
        return q

    def frame_pose(self, frame_name: str, q: np.ndarray) -> np.ndarray:
        return np.asarray(self.ik.frame_pose(frame_name, q).homogeneous, dtype=np.float32)
```

State FK always uses `base_xyyaw=None` (base at zero → base-frame poses). The
`base_xyyaw` path exists for the calib sidecar's world extrinsic only.

- [x] **Step 4: Implement LeRobotDataset writing**

Follow the structure of `/home/yixuan/lerobot_original/examples/port_datasets/port_dexmate_hdf5.py`, but only create the WBC EEF variant:

```text
features:
  observation.images.head_rgb
  observation.images.wrist_rgb
  observation.state: float32 shape (32,), axes STATE_AXES
  action: float32 shape (29,), axes ACTION_AXES
```

Save sidecars:

```text
debug/depth/episode_<i>.npz              key "depth"
debug/calib/episode_<i>.npz              keys "extrinsic", "base_extrinsic", "intrinsic"
```

For calibration:

```text
base_extrinsic = base_T_zed_depth_frame from WBC FK with base zero
extrinsic      = world_T_base(obs/base/pose) @ base_extrinsic
intrinsic      = static raw intrinsic broadcast to (T,3,3), with resize scaling applied if images are resized
```

- [x] **Step 5: CLI**

Implement:

```bash
scripts/port_wbc_mobile_hdf5.py \
  --raw-dir /home/yixuan/Dexmate/data/raw_data \
  --root /home/yixuan/Dexmate/data/processed_wbc \
  --fps 10 \
  --resize-h 240 \
  --resize-w 320 \
  --repo-id dexmate_wbc_eef_head
```

Support `--dry-run` that validates shapes and prints frame/action dims without creating a dataset.

- [x] **Step 6: Test the porter**

Run:

```bash
/home/yixuan/miniforge3/envs/dexmate/bin/python -m pytest tests/test_port_wbc_mobile_hdf5.py -q
```

Expected: all tests pass.

Then run on the real demo in dry-run mode:

```bash
/home/yixuan/miniforge3/envs/dexmate/bin/python scripts/port_wbc_mobile_hdf5.py \
  --raw-dir /home/yixuan/Dexmate/data/raw_data \
  --root /tmp/dexmate_wbc_processed \
  --fps 10 \
  --dry-run
```

Expected: exits non-zero with an error naming the missing
`action/eef/{left,right}` or `action/head` dataset in the current pre-update demo. Do not read
from `/home/yixuan/Dexmate/data/raw_data_debug`.

## Task 5: WBC Policy Rollout Script

**Files:**

- Create: `scripts/wbc_policy_rollout.py`
- Test: `tests/test_wbc_policy_rollout_decode.py`

- [x] **Step 1: Test action splitting (no base composition)**

Create `tests/test_wbc_policy_rollout_decode.py` to test a pure function:

```python
import numpy as np

from scripts.wbc_policy_rollout import split_policy_action
from omniteleop.wbc_policy_format import mat_to_pos6d


def test_split_policy_action_returns_world_targets_verbatim():
    world_T_left = np.eye(4)
    world_T_left[:3, 3] = [1.0, 2.3, 0.5]
    world_T_right = np.eye(4)
    world_T_right[:3, 3] = [1.0, 2.4, 0.5]
    world_T_head = np.eye(4)
    world_T_head[:3, 3] = [0.9, 2.2, 1.1]
    action = np.concatenate([
        mat_to_pos6d(world_T_left), [0.0],
        mat_to_pos6d(world_T_right), [1.0],
        mat_to_pos6d(world_T_head),
    ]).astype(np.float32)

    targets = split_policy_action(action)

    np.testing.assert_allclose(targets["left"][:3, 3], [1.0, 2.3, 0.5], atol=1e-6)
    np.testing.assert_allclose(targets["right"][:3, 3], [1.0, 2.4, 0.5], atol=1e-6)
    np.testing.assert_allclose(targets["head"][:3, 3], [0.9, 2.2, 1.1], atol=1e-6)
    np.testing.assert_allclose(targets["head"][:3, :3], np.eye(3), atol=1e-6)
    assert targets["left_gripper"] == 0.0
    assert targets["right_gripper"] == 1.0


def test_split_policy_action_rejects_wrong_dim():
    with np.testing.assert_raises(ValueError):
        split_policy_action(np.zeros(32, dtype=np.float32))
```

- [x] **Step 2: Implement pure split function**

In `scripts/wbc_policy_rollout.py`:

```python
def split_policy_action(action: np.ndarray) -> dict[str, np.ndarray | np.float32]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (29,):
        raise ValueError(f"expected 29-D WBC action, got {action.shape}")
    return {
        "left": pos6d_to_mat(action[0:9]),
        "right": pos6d_to_mat(action[10:19]),
        "head": pos6d_to_mat(action[20:29]),
        "left_gripper": np.float32(action[9]),
        "right_gripper": np.float32(action[19]),
    }
```

The outputs are ALREADY world-frame targets — never compose them with any base
pose, at the 10 Hz tick or the 100 Hz tick.

- [x] **Step 3: Implement rollout loop**

The script should:

```text
1. Load policy and LeRobot preprocess/postprocess exactly like reference policy_rollout.py
   (use_relative_actions must be false for this dataset — see Conventions).
2. Build a 32-D observation.state at 10 Hz:
   - base-frame EEF achieved from WBC FK on measured joints with base zero
   - raw gripper observations
   - base-frame head pose achieved from WBC FK on measured joints with base zero
   - measured odometry base pose (x, y, yaw) from OdometryThread (engage-origin)
3. Run policy at 10 Hz.
4. split_policy_action() -> world-frame left/right/head targets + gripper commands.
5. Push the world targets into TargetInterpolator (duration = 1/cmd_rate), exactly like
   the leader stream.
6. At each 100 Hz tick, interpolate, apply the head LPF + planar deadband to the
   interpolated head target (same shaping as wbc_vr_robot.py:1386-1388), then run
   VegaWholeBodyIK.solve(left_target, right_target, dt, head_target=head_target).
7. Send postprocessed commands through the same clamp/safety behavior as wbc_vr_robot.py
   (reuse HardwareDriver.actuate or equivalent clamping).
8. Record rollout debug HDF5 with obs/action vectors, world targets, sent joints,
   measured joints, base pose/twist, and timing.
```

Engage semantics must mirror teleop: on engage call `ik.reset()` AND
`OdometryThread.reset_origin()` so the policy's world frame is re-anchored at the
rollout start pose, matching how every training episode was anchored.

Use `VRTeleopConfig.from_yaml()` for `ik_rate=100` and `cmd_rate=10`. The policy FPS must match dataset FPS (`10`) unless the user explicitly overrides it.

- [x] **Step 4: Keep head control as a pose target while using WBC head IK**

The policy emits a 9-D head pose action: position plus 6D rotation for
`zed_depth_frame`. Do not switch the WBC policy IK to `head_mode="track"`.
Instantiate:

```python
ik = VegaWholeBodyIK(WBCConfig(head_mode="ik"))
```

This uses the existing `head_mode="ik"` path in `whole_body_ik.py`: `head_task`
tracks the head target, `head_ik_pinned_joints` pins `head_j1/head_j2`, and
`head_j3` remains a free IK DOF solved together with EEF tracking. Camera PAN
authority exists ONLY through this path (base yaw driven by the head task —
`head_world_position_cost: [1e4, 1e4, 0]` + orientation cost); a joint-space head
command could not pan the camera. The policy vector uses 6D rotation even though
roll is unactuatable and target-z is freed by config: those are solver-side cost
choices, not dataset-layout choices. Do not expose Euler, axis-angle, or
rotation-vector head orientation in the dataset.

- [x] **Step 5: Run decode tests**

Run:

```bash
/home/yixuan/miniforge3/envs/dexmate/bin/python -m pytest tests/test_wbc_policy_rollout_decode.py -q
```

Expected: all tests pass.

## Task 6: Verification On Recorded Demo

**Files:**

- No new files unless a test exposes a missing helper.

- [x] **Step 1: Compile changed files**

Run:

```bash
/home/yixuan/miniforge3/envs/dexmate/bin/python -m py_compile \
  src/omniteleop/wbc_policy_format.py \
  scripts/port_wbc_mobile_hdf5.py \
  scripts/wbc_policy_rollout.py \
  scripts/wbc_vr_robot.py
```

Expected: no output and exit code 0.

- [x] **Step 2: Run focused tests**

Run:

```bash
/home/yixuan/miniforge3/envs/dexmate/bin/python -m pytest \
  tests/test_wbc_policy_format.py \
  tests/test_port_wbc_mobile_hdf5.py \
  tests/test_wbc_policy_rollout_decode.py \
  tests/test_wbc_vr_robot_gripper_wrist.py \
  -q
```

Expected: all tests pass.

- [x] **Step 3: Dry-run current recorded demo**

Run:

```bash
/home/yixuan/miniforge3/envs/dexmate/bin/python scripts/port_wbc_mobile_hdf5.py \
  --raw-dir /home/yixuan/Dexmate/data/raw_data \
  --root /tmp/dexmate_wbc_processed \
  --fps 10 \
  --dry-run
```

Expected: exits non-zero and prints a clear error containing
`action/eef/left`, `action/eef/right`, or `action/head`. This confirms the
porter does not use debug data as a substitute for missing main raw actions.

## Self-Review Checklist

- [x] `observation.state` EEF/head is base-frame achieved FK; the world anchor is ONLY the explicit `base_x, base_y, base_yaw` dims from odometry.
- [x] `action` EEF/head is the VERBATIM world-frame target given to `ik.solve()` (head: post-LPF/post-deadband), not WBC result FK and not a base-frame composition.
- [x] Recorder stores targets with zero transformation (only float32 cast) — offline values are bit-identical to online solver inputs.
- [x] Rollout composes NOTHING with the base pose: policy world targets go straight into `TargetInterpolator`, and 100 Hz WBC interpolation operates on world-frame targets.
- [x] Rollout re-applies the head LPF + planar deadband to interpolated head targets, same as teleop.
- [x] Head policy dims are `zed_depth_frame` position + 6D rotation (world-frame in action, base-frame in state), not scalar `head_j3`.
- [x] Rollout uses `WBCConfig(head_mode="ik")` and passes the policy head pose as `head_target` (camera pan authority lives in the head task → base yaw path).
- [x] `action/joint/*` in future main raw HDF5 records post-clamp sent values; `--record` fails fast unless torso+arms+head+base are all enabled.
- [x] `use_relative_actions` stays false for this dataset (world action minus base-frame state would cross frames).
- [x] Current demos without `action/eef/*` or `action/head` fail immediately; the porter does not read debug data as a substitute.
- [x] Porter drops leading NaN-gripper frames and raises on mid-take NaN.
- [x] No code path depends on SAPIEN `KinHelper` for WBC mobile FK.