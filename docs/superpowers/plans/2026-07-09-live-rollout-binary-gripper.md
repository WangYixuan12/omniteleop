# Live Rollout Binary Gripper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make live WBC policy rollout execute binary open/closed gripper commands using the same `0.5` threshold as its training dataset.

**Architecture:** Move the binary threshold into the shared WBC policy schema, then apply it while decoding postprocessed live-policy action chunks. Preserve pose interpolation while the action scheduler selects gripper commands categorically from the latest due knot; replay continues to bypass this path and remains verbatim.

**Tech Stack:** Python 3.12, NumPy, pytest, ruff

## Global Constraints

- Apply thresholding only to postprocessed live-policy predictions.
- Keep `observation.state` gripper values as raw FC03 achieved positions in `[0, 1]`.
- Keep raw HDF5 episode replay bit-identical, including continuous gripper values.
- Do not change arm, head, base, policy normalization, or hardware-driver behavior.
- Use `GRIPPER_BINARY_THRESHOLD = 0.5` for both dataset conversion and rollout.

---

### Task 1: Share and apply the binary gripper threshold

**Files:**
- Modify: `src/omniteleop/wbc_policy_format.py:56-70`
- Modify: `scripts/port_wbc_mobile_hdf5.py:116-140`
- Modify: `scripts/wbc_policy_rollout.py:117-189`
- Test: `tests/test_wbc_policy_format.py`
- Test: `tests/test_wbc_policy_rollout_decode.py`

**Interfaces:**
- Produces: `omniteleop.wbc_policy_format.GRIPPER_BINARY_THRESHOLD: float`
- Changes: `split_policy_action(action: np.ndarray) -> dict[str, np.ndarray | np.float32]` returns binary `left_gripper` and `right_gripper` values.

- [ ] **Step 1: Write the failing shared-schema test**

Add a schema assertion:

```python
from omniteleop.wbc_policy_format import GRIPPER_BINARY_THRESHOLD


def test_gripper_binary_threshold_matches_dataset_contract():
    assert GRIPPER_BINARY_THRESHOLD == 0.5
```

- [ ] **Step 2: Run the schema test and verify RED**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yixuan/miniforge3/envs/dexmate/bin/python \
  -m pytest -q tests/test_wbc_policy_format.py::test_gripper_binary_threshold_matches_dataset_contract
```

Expected: collection fails because `GRIPPER_BINARY_THRESHOLD` does not exist.

- [ ] **Step 3: Add the shared constant and make the porter import it**

In `wbc_policy_format.py`, define:

```python
GRIPPER_BINARY_THRESHOLD = 0.5
```

Import it in `port_wbc_mobile_hdf5.py` and remove the porter's local duplicate.

- [ ] **Step 4: Run the schema test and verify GREEN**

Run the Step 2 command again.

Expected: `1 passed`.

- [ ] **Step 5: Write the failing live-decode test**

Add a decode regression using non-binary model predictions:

```python
def test_split_policy_action_binarizes_gripper_predictions_at_threshold():
    mod = _load_rollout()
    action = np.concatenate([
        mat_to_pos6d(np.eye(4)), [0.499],
        mat_to_pos6d(np.eye(4)), [0.5],
        mat_to_pos6d(np.eye(4)),
    ]).astype(np.float32)

    targets = mod.split_policy_action(action)

    assert targets["left_gripper"] == np.float32(0.0)
    assert targets["right_gripper"] == np.float32(1.0)
```

- [ ] **Step 6: Run the decode test and verify RED**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yixuan/miniforge3/envs/dexmate/bin/python \
  -m pytest -q \
  tests/test_wbc_policy_rollout_decode.py::test_split_policy_action_binarizes_gripper_predictions_at_threshold
```

Expected: assertion failure because `0.499` and `0.5` pass through unchanged.

- [ ] **Step 7: Threshold live decoding**

Import the shared constant in `wbc_policy_rollout.py` and decode with:

```python
"left_gripper": np.float32(action[9] >= GRIPPER_BINARY_THRESHOLD),
"right_gripper": np.float32(action[19] >= GRIPPER_BINARY_THRESHOLD),
```

- [ ] **Step 8: Run both focused tests and verify GREEN**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yixuan/miniforge3/envs/dexmate/bin/python \
  -m pytest -q tests/test_wbc_policy_format.py::test_gripper_binary_threshold_matches_dataset_contract \
  tests/test_wbc_policy_rollout_decode.py::test_split_policy_action_binarizes_gripper_predictions_at_threshold
```

Expected: `2 passed`.

- [ ] **Step 9: Commit Task 1 files only**

```bash
git add src/omniteleop/wbc_policy_format.py scripts/port_wbc_mobile_hdf5.py \
  scripts/wbc_policy_rollout.py tests/test_wbc_policy_format.py \
  tests/test_wbc_policy_rollout_decode.py
git commit --only src/omniteleop/wbc_policy_format.py scripts/port_wbc_mobile_hdf5.py \
  scripts/wbc_policy_rollout.py tests/test_wbc_policy_format.py \
  tests/test_wbc_policy_rollout_decode.py -m "fix: binarize live rollout grippers"
```

### Task 2: Schedule grippers as categorical commands

**Files:**
- Modify: `scripts/wbc_policy_rollout.py:277-380`
- Test: `tests/test_wbc_policy_schedule.py`

**Interfaces:**
- Consumes: binary gripper knots produced by `split_policy_action`.
- Changes: `ActionScheduleBuffer.sample(query_time: float)` returns the most recent due gripper knot while retaining existing pose interpolation.

- [ ] **Step 1: Change scheduler tests to require discrete gripper timing**

Replace the interpolation expectation with:

```python
def test_sample_interpolates_poses_but_holds_grippers_until_next_knot():
    mod = _load_rollout()
    buf = mod.ActionScheduleBuffer()
    buf.queue([
        _act(mod, 1.0, x=0.0, yaw=0.0, grip=0.0),
        _act(mod, 1.1, x=0.1, yaw=np.pi / 2, grip=1.0),
    ], now=0.0)
    between = buf.sample(1.05)
    assert between is not None
    assert between.left[0, 3] == pytest.approx(0.05)
    assert _yaw_of(between.left) == pytest.approx(np.pi / 4)
    assert float(between.grip_left) == 0.0
    assert float(between.grip_right) == 0.0

    at_knot = buf.sample(1.1)
    assert at_knot is not None
    assert float(at_knot.grip_left) == 1.0
    assert float(at_knot.grip_right) == 1.0
```

Extend the next-chunk test so a future knot does not move the gripper early:

```python
buf.queue([_act(mod, 1.2, x=0.2, grip=1.0)], now=1.05)
out = buf.sample(1.1)
assert float(out.grip_left) == 0.0
at_knot = buf.sample(1.2)
assert at_knot is not None
assert float(at_knot.grip_left) == 1.0
```

- [ ] **Step 2: Run scheduler tests and verify RED**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yixuan/miniforge3/envs/dexmate/bin/python \
  -m pytest -q tests/test_wbc_policy_schedule.py
```

Expected: failures showing interpolated gripper values such as `0.5` instead of categorical `0.0`.

- [ ] **Step 3: Preserve pose interpolation and select the latest due gripper knot**

Change `_interpolate_scheduled` so its fallback grippers come from `prev`, not a lerp. In `ActionScheduleBuffer.sample`, choose `passed` when a knot became due, otherwise the valid `prev`, otherwise `future`; use `dataclasses.replace` to put that source's binary grippers on the returned action:

```python
gripper_source = passed if passed is not None else (prev if prev is not None else future)
if out is None or gripper_source is None:
    return None
executed = replace(
    out,
    timestamp=float(query_time),
    grip_left=gripper_source.grip_left,
    grip_right=gripper_source.grip_right,
)
```

- [ ] **Step 4: Run scheduler tests and verify GREEN**

Run the Step 2 command again.

Expected: all scheduler tests pass.

- [ ] **Step 5: Run full focused regression and static checks**

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yixuan/miniforge3/envs/dexmate/bin/python \
  -m pytest -q tests/test_wbc_policy_format.py tests/test_port_wbc_mobile_hdf5.py \
  tests/test_wbc_policy_rollout_decode.py tests/test_wbc_policy_schedule.py \
  tests/test_wbc_policy_rollout_replay.py
/home/yixuan/miniforge3/envs/dexmate/bin/python -m ruff check \
  src/omniteleop/wbc_policy_format.py scripts/port_wbc_mobile_hdf5.py \
  scripts/wbc_policy_rollout.py tests/test_wbc_policy_format.py \
  tests/test_wbc_policy_rollout_decode.py tests/test_wbc_policy_schedule.py
/home/yixuan/miniforge3/envs/dexmate/bin/python -m py_compile \
  src/omniteleop/wbc_policy_format.py scripts/port_wbc_mobile_hdf5.py \
  scripts/wbc_policy_rollout.py
git diff --check
```

Expected: all focused tests pass; ruff, compilation, and diff checks exit `0`.

- [ ] **Step 6: Commit Task 2 files only**

```bash
git add scripts/wbc_policy_rollout.py tests/test_wbc_policy_schedule.py
git commit --only scripts/wbc_policy_rollout.py tests/test_wbc_policy_schedule.py \
  -m "fix: schedule grippers as discrete actions"
```
