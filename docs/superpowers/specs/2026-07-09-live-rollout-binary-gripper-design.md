# Live Rollout Binary Gripper Design

## Goal

Make live WBC policy rollout execute the same binary gripper action contract used by
`scripts/port_wbc_mobile_hdf5.py`: action dimensions 9 and 19 represent open (`0`) or
closed (`1`) commands separated by a `0.5` threshold.

## Scope

- Apply the threshold only to postprocessed live-policy predictions.
- Keep `observation.state` gripper values as raw FC03 achieved positions in `[0, 1]`.
- Keep raw HDF5 episode replay bit-identical, including continuous gripper values.
- Do not change arm, head, base, policy normalization, or hardware-driver behavior.

## Design

`split_policy_action` will convert each de-normalized live-policy gripper prediction to
`0.0` or `1.0` using the dataset threshold, `>= 0.5`. Scheduled policy actions will
therefore contain binary gripper knots.

The action scheduler will treat grippers as categorical values rather than interpolated
widths. Before the first scheduled knot it will use that first knot, between knots it will
hold the latest knot whose timestamp has passed, and at a knot it will switch to that
knot's value. Pose positions and rotations retain their existing interpolation behavior.
Because replay bypasses the scheduled-policy buffer, replay remains verbatim.

The threshold constant will live in `omniteleop.wbc_policy_format`, the shared schema
module, and the porter and rollout will both import it to prevent drift.

## Testing

- Action decoding maps values below `0.5` to `0` and values at or above `0.5` to `1`.
- Scheduled interpolation holds the preceding binary command between knots and switches
  at the next knot instead of producing intermediate widths.
- A new overlapping chunk preserves the last executed binary command until its first
  new knot becomes due.
- Existing replay tests continue to prove raw gripper values are unchanged.
- Run focused porter, format, decode, schedule, and replay tests, followed by lint and
  compilation checks for the touched Python files.
