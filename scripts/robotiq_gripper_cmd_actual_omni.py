"""Standalone Robotiq Hand-E cmd-vs-actual logger.

Drives one or both Hand-E grippers through a step sequence, polls the actual
encoder position via FC03 over a queued EE pass-through response monitor, and
writes a CSV plus PNG of cmd / actual plus latency summary per side.

Modes:
  * stepped: traverse the requested step sequence once and send a gripper
    command only when the command value changes.
  * continuous: repeat a 0.01-resolution ramp at the tick rate and resend the
    gripper command on every tick, matching VRRobotController's whole-body
    control_rate by default.
  * square / triangle / sine: resend a periodic command on every tick. Use
    these to probe command bandwidth, phase delay, and tracking error.

The collector uses a queued response demux:
  * Tick rate issues cmd writes and FC03 status requests; emits one "tick" row
    per side per tick with separate send timestamps for the FC16 command (when
    one is actually sent) and the FC03 request.
  * Drain rate (~500 Hz) reads each side's queued monitor nonblockingly and
    emits one "sample" row with read timestamps whenever a fresh, parseable
    FC03 reply lands.

This avoids the lossy single-slot response pattern: FC16 write ACKs are counted
and discarded by the monitor while FC03 status replies are queued. The CSV's
`kind` column distinguishes tick, sample, and timeout rows.
The plot uses dots only, so it shows actual send/read events rather than
interpolated curves.

Hardware preconditions:
  * Robot config has `enable_ee_pass_through=True` for each requested arm.
  * Detected EE type is UNKNOWN (otherwise dexcontrol does not publish
    pass-through replies and `get_ee_pass_through_response()` returns
    None — the warm-up check below will warn).

Run:
    python scripts/robotiq_gripper_cmd_actual_omni.py
    python scripts/robotiq_gripper_cmd_actual_omni.py --mode continuous
    python scripts/robotiq_gripper_cmd_actual_omni.py --mode square --signal-frequency 0.5
    python scripts/robotiq_gripper_cmd_actual_omni.py --mode triangle --signal-frequency 0.25
    python scripts/robotiq_gripper_cmd_actual_omni.py --mode sine --signal-frequency 0.5
    python scripts/robotiq_gripper_cmd_actual_omni.py --sides right --duration 15
"""

import argparse
import csv
import math
import signal
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from dexcontrol.core.arm import Arm
from dexcontrol.robot import Robot

from omniteleop.common import get_config
from omniteleop.follower.robotiq import (
    RobotiqStatusMonitor,
    build_hande_command,
    poll_gripper_status,
    send_activate,
    send_ee_pass_through_with_timestamps,
)

DEFAULT_STEPS = (0.0, 1.0, 0.5, 1.0, 0.0)
CONTINUOUS_STEP_SIZE = 0.01
CONTINUOUS_STEPS = tuple(round(i * CONTINUOUS_STEP_SIZE, 2) for i in range(101)) + tuple(
    round(i * CONTINUOUS_STEP_SIZE, 2) for i in range(99, 0, -1)
)
DEFAULT_STEPPED_RATE_HZ = 20.0
DEFAULT_WHOLE_BODY_RATE_HZ = 100.0
DEFAULT_HOLD_S = 3.0
DEFAULT_SIGNAL_FREQUENCY_HZ = 0.25
DEFAULT_SIGNAL_CYCLES = 5.0
DRAIN_PERIOD_S = 0.002  # ~500 Hz nonblocking drain
PRE_MOTION_WINDOW_S = 0.25
SETTLED_TOL = 1.5 / 255.0
ENDPOINT_TOL = 0.02
PERIODIC_MODES = {"square", "triangle", "sine"}
DYNAMIC_MODES = PERIODIC_MODES | {"continuous"}

CSV_FIELDS = [
    "kind",
    "tick_t_monotonic_s",
    "sample_t_monotonic_s",
    "event_t_monotonic_s",
    "event_wall_ns",
    "side",
    "status_result",
    "status_request_id",
    "cmd",
    "cmd_sent",
    "cmd_send_t_monotonic_s",
    "cmd_send_monotonic_ns",
    "cmd_send_wall_ns",
    "cmd_hex",
    "cmd_at_sample",
    "status_request_send_t_monotonic_s",
    "status_request_send_monotonic_ns",
    "status_request_send_wall_ns",
    "status_request_hex",
    "state_read_t_monotonic_s",
    "state_read_monotonic_ns",
    "state_read_wall_ns",
    "status_timeout_t_monotonic_s",
    "status_timeout_monotonic_ns",
    "status_timeout_wall_ns",
    "monitor_fc16_acks_discarded",
    "monitor_invalid_discarded",
    "monitor_other_status_discarded",
    "monitor_missing_data_discarded",
    "monitor_status_timeouts",
    "monitor_unmatched_statuses",
    "monitor_status_queue_overflow",
    "monitor_pending_status_requests",
    "cmd_echo",
    "actual",
    "current_ma",
    "gSTA",
    "gOBJ",
    "gFLT",
    "raw_hex",
    "response_timestamp_ns",
    "response_sequence",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--mode",
        choices=("stepped", "continuous", "square", "triangle", "sine"),
        default="continuous",
        help=(
            "stepped sends only when the command changes; continuous sends a "
            "0.00->1.00->0.00 ramp, one command per tick, at the whole-body "
            "control rate by default; square/triangle/sine send periodic "
            "commands at --signal-frequency"
        ),
    )
    p.add_argument(
        "--sides",
        default="right",
        help="comma-separated sides to drive (default: right)",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=None,
        help=(
            "tick rate in Hz (default: "
            f"{DEFAULT_STEPPED_RATE_HZ:.0f} in stepped mode; "
            "whole-body control_rate in continuous/square/triangle/sine modes)"
        ),
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help=(
            "run duration in seconds (default: stepped len(steps)*hold + 2.0; "
            "continuous one ramp cycle + 2.0; periodic --cycles/--signal-frequency)"
        ),
    )
    p.add_argument(
        "--hold",
        type=float,
        default=DEFAULT_HOLD_S,
        help=(
            f"seconds per step in stepped mode (default: {DEFAULT_HOLD_S:.1f}); "
            "continuous mode advances one command per tick"
        ),
    )
    p.add_argument(
        "--signal-frequency",
        type=float,
        default=DEFAULT_SIGNAL_FREQUENCY_HZ,
        help=(
            "periodic command frequency in Hz for square/triangle/sine modes "
            f"(default: {DEFAULT_SIGNAL_FREQUENCY_HZ:.2f})"
        ),
    )
    p.add_argument(
        "--cycles",
        type=float,
        default=DEFAULT_SIGNAL_CYCLES,
        help=(
            "number of periodic cycles to run when --duration is omitted for "
            f"square/triangle/sine modes (default: {DEFAULT_SIGNAL_CYCLES:g})"
        ),
    )
    p.add_argument(
        "--min-cmd",
        type=float,
        default=0.0,
        help="minimum command for default continuous ramp and periodic modes (default: 0.0)",
    )
    p.add_argument(
        "--max-cmd",
        type=float,
        default=1.0,
        help="maximum command for default continuous ramp and periodic modes (default: 1.0)",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Robotiq rSP speed command in [0,1] (default: 1.0)",
    )
    p.add_argument(
        "--force",
        type=float,
        default=1.0,
        help="Robotiq rFR force command in [0,1] (default: 1.0)",
    )
    p.add_argument(
        "--analysis-max-lag",
        type=float,
        default=None,
        help=(
            "maximum delay, in seconds, searched by dynamic response analysis "
            "(default: bounded by the stimulus period)"
        ),
    )
    p.add_argument(
        "--steps",
        default=None,
        help=(
            "comma-separated step sequence in [0,1] "
            f"(default: {DEFAULT_STEPS} in stepped; "
            "0.00->1.00->0.00 by 0.01 in continuous)"
        ),
    )
    p.add_argument(
        "--function-code",
        type=lambda s: int(s, 0),
        default=0x03,
        help="Modbus function code for status read (3 or 4; default 3)",
    )
    p.add_argument(
        "--status-timeout",
        type=float,
        default=0.2,
        help="seconds before a sent FC03/FC04 request is marked timed out (default: 0.2)",
    )
    p.add_argument(
        "--out-dir",
        default="/home/yixuan/omniteleop/Dexmate/debug/plots/gripper",
        help="output directory (default: Dexmate/debug/plots/gripper)",
    )
    p.add_argument("--no-plot", action="store_true", help="skip matplotlib output")
    return p.parse_args()


def whole_body_control_rate_hz() -> float:
    """Match VRRobotController's whole-body gripper command loop rate."""
    return float(get_config().get_rate("control_rate", DEFAULT_WHOLE_BODY_RATE_HZ))


def get_arm(robot: Robot, side: str) -> Arm:
    if side == "left":
        return robot.left_arm
    if side == "right":
        return robot.right_arm
    raise ValueError(f"unknown side {side!r}; expected 'left' or 'right'")


def step_for_time(t_s: float, steps: tuple, hold_s: float) -> float:
    """One-shot step traversal: hold the final step value once the sequence ends."""
    cycle_s = len(steps) * hold_s
    if t_s >= cycle_s:
        return float(steps[-1])
    return float(steps[int(t_s // hold_s)])


def repeating_step_for_tick(tick_index: int, steps: tuple) -> float:
    """Repeating one-command-per-tick traversal for continuous resend tests."""
    return float(steps[tick_index % len(steps)])


def make_continuous_steps(min_cmd: float, max_cmd: float) -> tuple:
    """Default discrete triangle ramp used by the legacy continuous mode."""
    unit_steps = CONTINUOUS_STEPS
    span = max_cmd - min_cmd
    return tuple(round(min_cmd + span * s, 4) for s in unit_steps)


def periodic_command(
    t_s: float,
    mode: str,
    frequency_hz: float,
    min_cmd: float,
    max_cmd: float,
) -> float:
    """Return a square/triangle/sine command bounded by [min_cmd, max_cmd]."""
    phase = (t_s * frequency_hz) % 1.0
    span = max_cmd - min_cmd
    mid = (min_cmd + max_cmd) * 0.5
    amp = span * 0.5

    if mode == "square":
        return max_cmd if phase >= 0.5 else min_cmd
    if mode == "triangle":
        if phase < 0.5:
            return min_cmd + span * (2.0 * phase)
        return max_cmd - span * (2.0 * (phase - 0.5))
    if mode == "sine":
        # Starts at min_cmd with zero slope, reaches max_cmd half a cycle later.
        return mid - amp * math.cos(2.0 * math.pi * phase)
    raise ValueError(f"periodic_command got non-periodic mode {mode!r}")


def command_for_tick(
    mode: str,
    elapsed_s: float,
    tick_index: int,
    steps: tuple,
    hold_s: float,
    signal_frequency_hz: float,
    min_cmd: float,
    max_cmd: float,
) -> float:
    if mode == "continuous":
        return repeating_step_for_tick(tick_index, steps)
    if mode == "stepped":
        return step_for_time(elapsed_s, steps, hold_s)
    return periodic_command(elapsed_s, mode, signal_frequency_hz, min_cmd, max_cmd)


def warmup_check(arm: Arm, side: str, function_code: int, timeout_s: float = 0.5) -> bool:
    parsed = poll_gripper_status(arm, function_code=function_code, timeout_s=timeout_s)
    if parsed is None:
        print(
            f"WARN: {side} gripper did not return a parseable status reply "
            f"within {timeout_s*1000:.0f} ms. Verify enable_ee_pass_through=True "
            "and EE type is UNKNOWN; the loop will produce zero sample rows."
        )
        return False
    print(
        f"{side} warmup OK: gPO={parsed['gPO']} ({parsed['actual']:.3f}), "
        f"gPR={parsed['gPR']} ({parsed['cmd_echo']:.3f}), "
        f"gOBJ={parsed['gOBJ']}, gFLT={parsed['gFLT']}"
    )
    return True


def _event_elapsed_s(event: Optional[Dict], t0_monotonic_ns: int) -> str | float:
    if event is None:
        return ""
    return (int(event["send_monotonic_ns"]) - t0_monotonic_ns) / 1e9


def _monitor_stat_fields(stats: Optional[Dict[str, int]]) -> Dict:
    stats = stats or {}
    return {
        "monitor_fc16_acks_discarded": stats.get("fc16_acks_discarded", ""),
        "monitor_invalid_discarded": stats.get("invalid_discarded", ""),
        "monitor_other_status_discarded": stats.get("other_status_discarded", ""),
        "monitor_missing_data_discarded": stats.get("missing_data_discarded", ""),
        "monitor_status_timeouts": stats.get("status_timeouts", ""),
        "monitor_unmatched_statuses": stats.get("unmatched_statuses", ""),
        "monitor_status_queue_overflow": stats.get("status_queue_overflow", ""),
        "monitor_pending_status_requests": stats.get("pending_status_requests", ""),
    }


def make_tick_row(
    tick_elapsed: float,
    side: str,
    cmd: float,
    t0_monotonic_ns: int,
    cmd_event: Optional[Dict] = None,
    status_request_event: Optional[Dict] = None,
    monitor_stats: Optional[Dict[str, int]] = None,
) -> Dict:
    event = cmd_event or status_request_event
    row = {
        "kind": "tick",
        "tick_t_monotonic_s": tick_elapsed,
        "sample_t_monotonic_s": "",
        "event_t_monotonic_s": _event_elapsed_s(event, t0_monotonic_ns),
        "event_wall_ns": "" if event is None else event["send_wall_ns"],
        "side": side,
        "status_result": "requested" if status_request_event is not None else "",
        "status_request_id": (
            "" if status_request_event is None else status_request_event["status_request_id"]
        ),
        "cmd": cmd,
        "cmd_sent": cmd_event is not None,
        "cmd_send_t_monotonic_s": _event_elapsed_s(cmd_event, t0_monotonic_ns),
        "cmd_send_monotonic_ns": "" if cmd_event is None else cmd_event["send_monotonic_ns"],
        "cmd_send_wall_ns": "" if cmd_event is None else cmd_event["send_wall_ns"],
        "cmd_hex": "" if cmd_event is None else cmd_event["message_hex"],
        "cmd_at_sample": "",
        "status_request_send_t_monotonic_s": _event_elapsed_s(
            status_request_event, t0_monotonic_ns
        ),
        "status_request_send_monotonic_ns": (
            "" if status_request_event is None else status_request_event["send_monotonic_ns"]
        ),
        "status_request_send_wall_ns": (
            "" if status_request_event is None else status_request_event["send_wall_ns"]
        ),
        "status_request_hex": (
            "" if status_request_event is None else status_request_event["message_hex"]
        ),
        "state_read_t_monotonic_s": "",
        "state_read_monotonic_ns": "",
        "state_read_wall_ns": "",
        "status_timeout_t_monotonic_s": "",
        "status_timeout_monotonic_ns": "",
        "status_timeout_wall_ns": "",
        "cmd_echo": "",
        "actual": "",
        "current_ma": "",
        "gSTA": "",
        "gOBJ": "",
        "gFLT": "",
        "raw_hex": "",
        "response_timestamp_ns": "",
        "response_sequence": "",
    }
    row.update(_monitor_stat_fields(monitor_stats))
    return row


def make_sample_row(
    side: str,
    cmd_at_sample: Optional[float],
    parsed: Dict,
    t0_monotonic_ns: int,
) -> Dict:
    read_monotonic_ns = parsed.get("read_monotonic_ns")
    sample_elapsed = (
        ""
        if read_monotonic_ns in ("", None)
        else (int(read_monotonic_ns) - t0_monotonic_ns) / 1e9
    )
    request_send_monotonic_ns = parsed.get("request_send_monotonic_ns")
    request_elapsed = (
        ""
        if request_send_monotonic_ns in ("", None)
        else (int(request_send_monotonic_ns) - t0_monotonic_ns) / 1e9
    )
    row = {
        "kind": "sample",
        "tick_t_monotonic_s": "",
        "sample_t_monotonic_s": sample_elapsed,
        "event_t_monotonic_s": sample_elapsed,
        "event_wall_ns": parsed.get("read_wall_ns", ""),
        "side": side,
        "status_result": "ok",
        "status_request_id": parsed.get("status_request_id", ""),
        "cmd": "",
        "cmd_sent": "",
        "cmd_send_t_monotonic_s": "",
        "cmd_send_monotonic_ns": "",
        "cmd_send_wall_ns": "",
        "cmd_hex": "",
        "cmd_at_sample": "" if cmd_at_sample is None else cmd_at_sample,
        "status_request_send_t_monotonic_s": request_elapsed,
        "status_request_send_monotonic_ns": parsed.get("request_send_monotonic_ns", ""),
        "status_request_send_wall_ns": parsed.get("request_send_wall_ns", ""),
        "status_request_hex": parsed.get("status_request_hex", ""),
        "state_read_t_monotonic_s": sample_elapsed,
        "state_read_monotonic_ns": parsed.get("read_monotonic_ns", ""),
        "state_read_wall_ns": parsed.get("read_wall_ns", ""),
        "status_timeout_t_monotonic_s": "",
        "status_timeout_monotonic_ns": "",
        "status_timeout_wall_ns": "",
        "cmd_echo": parsed["cmd_echo"],
        "actual": parsed["actual"],
        "current_ma": parsed["current_ma"],
        "gSTA": parsed["gSTA"],
        "gOBJ": parsed["gOBJ"],
        "gFLT": parsed["gFLT"],
        "raw_hex": parsed["raw_hex"],
        "response_timestamp_ns": parsed.get("response_timestamp_ns", ""),
        "response_sequence": parsed.get("response_sequence", ""),
    }
    row.update(_monitor_stat_fields(None))
    return row


def make_timeout_row(
    side: str,
    timeout_event: Dict,
    t0_monotonic_ns: int,
    monitor_stats: Optional[Dict[str, int]] = None,
) -> Dict:
    timeout_elapsed = (int(timeout_event["timeout_monotonic_ns"]) - t0_monotonic_ns) / 1e9
    request_elapsed = (int(timeout_event["send_monotonic_ns"]) - t0_monotonic_ns) / 1e9
    row = {
        "kind": "timeout",
        "tick_t_monotonic_s": "",
        "sample_t_monotonic_s": "",
        "event_t_monotonic_s": timeout_elapsed,
        "event_wall_ns": timeout_event["timeout_wall_ns"],
        "side": side,
        "status_result": "timeout",
        "status_request_id": timeout_event["status_request_id"],
        "cmd": "",
        "cmd_sent": "",
        "cmd_send_t_monotonic_s": "",
        "cmd_send_monotonic_ns": "",
        "cmd_send_wall_ns": "",
        "cmd_hex": "",
        "cmd_at_sample": "",
        "status_request_send_t_monotonic_s": request_elapsed,
        "status_request_send_monotonic_ns": timeout_event["send_monotonic_ns"],
        "status_request_send_wall_ns": timeout_event["send_wall_ns"],
        "status_request_hex": timeout_event["message_hex"],
        "state_read_t_monotonic_s": "",
        "state_read_monotonic_ns": "",
        "state_read_wall_ns": "",
        "status_timeout_t_monotonic_s": timeout_elapsed,
        "status_timeout_monotonic_ns": timeout_event["timeout_monotonic_ns"],
        "status_timeout_wall_ns": timeout_event["timeout_wall_ns"],
        "cmd_echo": "",
        "actual": "",
        "current_ma": "",
        "gSTA": "",
        "gOBJ": "",
        "gFLT": "",
        "raw_hex": "",
        "response_timestamp_ns": "",
        "response_sequence": "",
    }
    row.update(_monitor_stat_fields(monitor_stats))
    return row


def write_csv(rows: List[Dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def print_summary(
    rows: List[Dict],
    sides: List[str],
    duration_s: float,
    hold_s: float,
    mode: str,
) -> None:
    settle_s = min(0.3, hold_s * 0.3)
    sample_rows = [r for r in rows if r["kind"] == "sample"]
    tick_rows = [r for r in rows if r["kind"] == "tick"]
    timeout_rows = [r for r in rows if r["kind"] == "timeout"]
    cmd_rows = [r for r in tick_rows if r.get("cmd_sent")]
    print(
        f"Captured {len(cmd_rows)} command sends, {len(tick_rows)} status requests, "
        f"{len(sample_rows)} state reads, and {len(timeout_rows)} status timeouts."
    )
    for side in sides:
        side_samples = [r for r in sample_rows if r["side"] == side]
        if not side_samples:
            print(f"{side}: no samples captured")
            continue

        rate_hz = len(side_samples) / duration_s if duration_s > 0 else float("nan")

        if mode == "stepped":
            steady_err: List[float] = []
            last_cmd_v: Optional[float] = None
            last_change_t = -1e9
            for r in side_samples:
                cmd_at_sample = r["cmd_at_sample"]
                if cmd_at_sample == "" or cmd_at_sample is None:
                    continue
                if cmd_at_sample != last_cmd_v:
                    last_cmd_v = cmd_at_sample
                    last_change_t = r["sample_t_monotonic_s"]
                if r["sample_t_monotonic_s"] - last_change_t >= settle_s:
                    steady_err.append(r["actual"] - cmd_at_sample)

            err = np.asarray(steady_err, np.float64)
            if err.size:
                print(
                    f"{side}: {len(side_samples)} samples ({rate_hz:.1f} Hz); "
                    f"steady-state actual-cmd: "
                    f"p95(|.|)={np.percentile(np.abs(err), 95):.3f}, "
                    f"max(|.|)={np.abs(err).max():.3f}"
                )
            else:
                print(
                    f"{side}: {len(side_samples)} samples ({rate_hz:.1f} Hz); "
                    "no steady-state samples (all within settle window)"
                )
        else:
            actual = np.asarray([r["actual"] for r in side_samples], dtype=np.float64)
            print(
                f"{side}: {len(side_samples)} samples ({rate_hz:.1f} Hz); "
                f"actual range={actual.min():.3f}..{actual.max():.3f}"
            )

        flt_counts = Counter(r["gFLT"] for r in side_samples)
        if any(k != 0 for k in flt_counts):
            print(f"  fault breakdown (gFLT): {dict(flt_counts)}")
        obj_counts = Counter(r["gOBJ"] for r in side_samples)
        print(f"  gOBJ histogram: {dict(obj_counts)}")
        side_timeouts = [r for r in timeout_rows if r["side"] == side]
        if side_timeouts:
            print(f"  status timeouts: {len(side_timeouts)}")


def classify_pre_motion(
    samples: List[tuple],
    t_send: float,
    direction: str,
    window_s: float,
    settled_tol: float,
) -> Dict:
    prior = [(t, r) for t, r in samples if t_send - window_s <= t < t_send]
    if len(prior) < 2:
        return {
            "pre_motion": "unknown",
            "pre_settled": False,
            "pre_actual_span": None,
            "pre_samples": len(prior),
        }

    actuals = [float(r["actual"]) for _t, r in prior]
    actual_span = max(actuals) - min(actuals)
    if actual_span <= settled_tol:
        state = "settled"
    else:
        delta = actuals[-1] - actuals[0]
        if direction == "close":
            toward = delta > settled_tol
            away = delta < -settled_tol
        else:
            toward = delta < -settled_tol
            away = delta > settled_tol
        if toward:
            state = "moving_toward"
        elif away:
            state = "moving_away"
        else:
            state = "unsettled"

    return {
        "pre_motion": state,
        "pre_settled": state == "settled",
        "pre_actual_span": actual_span,
        "pre_samples": len(prior),
    }


def has_directional_motion(
    direction: str,
    actual: float,
    actual_at_send: float,
    motion_thresh: float,
) -> bool:
    if direction == "close":
        return actual - actual_at_send > motion_thresh
    return actual_at_send - actual > motion_thresh


def has_reached_target(
    direction: str,
    actual: float,
    target: float,
    endpoint_tol: float,
) -> bool:
    if direction == "close":
        return actual >= target - endpoint_tol
    return actual <= target + endpoint_tol


def compute_latencies(
    rows: List[Dict],
    sides: List[str],
    motion_thresh: float = 0.005,
    echo_tol: float = 1.5 / 255.0,
    pre_motion_window_s: float = PRE_MOTION_WINDOW_S,
    settled_tol: float = SETTLED_TOL,
    endpoint_tol: float = ENDPOINT_TOL,
) -> Dict[str, Dict[str, List[Dict]]]:
    """For each cmd transition, find latency to first echo and first motion.

    A "transition" is a tick whose `cmd` differs from the previous tick's cmd
    on the same side. The very first tick (no prior cmd) is excluded.

    For each transition, scan forward through this side's sample rows for:
      * `echo_ms` — the first sample where `cmd_echo` matches the new cmd
        within `echo_tol` (Modbus round-trip + status-poll scheduling).
      * `motion_ms` — the first sample where `actual` moves from its value
        immediately before the send by more than `motion_thresh` in the
        commanded direction.
      * `complete_ms` — the first sample before the next command where
        `actual` reaches the commanded target within `endpoint_tol`.

    The pre-command actual trace is marked `settled` only if its range over
    `pre_motion_window_s` is no larger than `settled_tol`. Motion means use
    settled transitions only so residual motion is not counted as response
    latency.

    Direction:
      * `close` if the new cmd > old cmd (gripper moves toward fully closed).
      * `open`  if the new cmd < old cmd (gripper moves toward fully open).
    """
    results: Dict[str, Dict[str, List[Dict]]] = {s: {"close": [], "open": []} for s in sides}
    for side in sides:
        side_rows: List[tuple] = []
        for r in rows:
            if r["side"] != side:
                continue
            if r["kind"] == "tick":
                if not r.get("cmd_sent"):
                    continue
                t = r.get("cmd_send_t_monotonic_s") or r["tick_t_monotonic_s"]
            elif r["kind"] == "sample":
                t = r["sample_t_monotonic_s"]
            else:
                continue
            side_rows.append((t, r))
        side_rows.sort(key=lambda x: x[0])

        prev_cmd: Optional[float] = None
        last_actual: Optional[float] = None
        transitions: List[Dict] = []
        for _t, r in side_rows:
            if r["kind"] == "tick":
                cmd = r["cmd"]
                if prev_cmd is not None and cmd != prev_cmd:
                    transitions.append(
                        {
                            "t_send": r.get("cmd_send_t_monotonic_s")
                            or r["tick_t_monotonic_s"],
                            "from": prev_cmd,
                            "to": cmd,
                            "actual_at_send": last_actual,
                        }
                    )
                prev_cmd = cmd
            else:
                last_actual = r["actual"]

        sample_only = [(t, r) for t, r in side_rows if r["kind"] == "sample"]
        for i, tr in enumerate(transitions):
            target = tr["to"]
            t_send = tr["t_send"]
            t_next = transitions[i + 1]["t_send"] if i + 1 < len(transitions) else float("inf")
            actual_at_send = tr["actual_at_send"]
            direction = "close" if tr["to"] > tr["from"] else "open"
            pre_motion = classify_pre_motion(
                sample_only,
                t_send=t_send,
                direction=direction,
                window_s=pre_motion_window_s,
                settled_tol=settled_tol,
            )

            echo_ms: Optional[float] = None
            motion_ms: Optional[float] = None
            complete_ms: Optional[float] = None
            for t, r in sample_only:
                if t < t_send:
                    continue
                if t >= t_next:
                    break
                if echo_ms is None and abs(r["cmd_echo"] - target) <= echo_tol:
                    echo_ms = (t - t_send) * 1000.0
                if (
                    motion_ms is None
                    and actual_at_send is not None
                    and has_directional_motion(
                        direction,
                        actual=r["actual"],
                        actual_at_send=actual_at_send,
                        motion_thresh=motion_thresh,
                    )
                ):
                    motion_ms = (t - t_send) * 1000.0
                if complete_ms is None and has_reached_target(
                    direction,
                    actual=r["actual"],
                    target=target,
                    endpoint_tol=endpoint_tol,
                ):
                    complete_ms = (t - t_send) * 1000.0
                if echo_ms is not None and motion_ms is not None and complete_ms is not None:
                    break

            results[side][direction].append(
                {
                    "t_send": t_send,
                    "from": tr["from"],
                    "to": tr["to"],
                    "echo_ms": echo_ms,
                    "motion_ms": motion_ms,
                    "complete_ms": complete_ms,
                    **pre_motion,
                }
            )
    return results


def print_latencies(latencies: Dict[str, Dict[str, List[Dict]]]) -> None:
    print()
    print(
        "Send→gripper latency "
        "(echo = command visible in status; motion = directional encoder threshold):"
    )
    for side, dirs in latencies.items():
        for direction in ("close", "open"):
            entries = dirs[direction]
            if not entries:
                continue
            print(f"  {side} {direction}:")
            for e in entries:
                echo_str = "n/a" if e["echo_ms"] is None else f"{e['echo_ms']:.1f} ms"
                motion_str = "n/a" if e["motion_ms"] is None else f"{e['motion_ms']:.1f} ms"
                complete_str = "n/a" if e["complete_ms"] is None else f"{e['complete_ms']:.1f} ms"
                span = e.get("pre_actual_span")
                span_str = "" if span is None else f", pre_span={span:.3f}"
                print(
                    f"    [{e['from']:.2f} → {e['to']:.2f} @ t={e['t_send']:.2f}s] "
                    f"echo={echo_str}, motion={motion_str}, complete={complete_str}, "
                    f"pre={e.get('pre_motion', 'unknown')}{span_str}"
                )
            echos = [e["echo_ms"] for e in entries if e["echo_ms"] is not None]
            motions = [
                e["motion_ms"]
                for e in entries
                if e["motion_ms"] is not None and e.get("pre_settled")
            ]
            if echos:
                print(
                    f"    echo  agg: mean={np.mean(echos):.1f}, "
                    f"median={np.median(echos):.1f}, "
                    f"max={np.max(echos):.1f} ms (n={len(echos)})"
                )
            if motions:
                print(
                    f"    motion agg (settled only): mean={np.mean(motions):.1f}, "
                    f"median={np.median(motions):.1f}, "
                    f"max={np.max(motions):.1f} ms (n={len(motions)})"
                )
            skipped = sum(not e.get("pre_settled") for e in entries)
            if skipped:
                print(
                    f"    motion agg skipped {skipped} transition(s) "
                    "without settled pre-command actual"
                )


def compute_continuous_peak_latencies(
    rows: List[Dict],
    sides: List[str],
    cmd_target: float = 1.0,
    cmd_tol: float = SETTLED_TOL,
) -> Dict[str, List[Dict]]:
    """Find latency from cmd=1.0 ticks to the later actual maximum per cycle."""
    results: Dict[str, List[Dict]] = {side: [] for side in sides}
    for side in sides:
        side_ticks = [
            r
            for r in rows
            if r["kind"] == "tick" and r["side"] == side and r.get("cmd_sent")
        ]
        side_samples = [r for r in rows if r["kind"] == "sample" and r["side"] == side]
        side_ticks.sort(key=lambda r: r.get("cmd_send_t_monotonic_s") or r["tick_t_monotonic_s"])
        side_samples.sort(key=lambda r: r["sample_t_monotonic_s"])

        cmd_peak_ticks: List[Dict] = []
        prev_cmd: Optional[float] = None
        for r in side_ticks:
            cmd = float(r["cmd"])
            is_target = abs(cmd - cmd_target) <= cmd_tol
            was_target = prev_cmd is not None and abs(prev_cmd - cmd_target) <= cmd_tol
            if is_target and not was_target:
                cmd_peak_ticks.append(r)
            prev_cmd = cmd

        for i, tick in enumerate(cmd_peak_ticks):
            t_send = tick.get("cmd_send_t_monotonic_s") or tick["tick_t_monotonic_s"]
            t_next_peak = (
                cmd_peak_ticks[i + 1].get("cmd_send_t_monotonic_s")
                or cmd_peak_ticks[i + 1]["tick_t_monotonic_s"]
                if i + 1 < len(cmd_peak_ticks)
                else float("inf")
            )
            window = [r for r in side_samples if t_send <= r["sample_t_monotonic_s"] < t_next_peak]
            if not window:
                results[side].append(
                    {
                        "t_send": t_send,
                        "peak_ms": None,
                        "actual_max": None,
                        "t_peak": None,
                    }
                )
                continue

            actual_max = max(float(r["actual"]) for r in window)
            peak_sample = next(r for r in window if float(r["actual"]) >= actual_max)
            t_peak = peak_sample["sample_t_monotonic_s"]
            results[side].append(
                {
                    "t_send": t_send,
                    "peak_ms": (t_peak - t_send) * 1000.0,
                    "actual_max": actual_max,
                    "t_peak": t_peak,
                }
            )
    return results


def print_continuous_peak_latencies(
    peak_latencies: Dict[str, List[Dict]],
) -> None:
    print()
    print("Continuous latency (cmd=1.0 → actual maximum in cycle):")
    for side, entries in peak_latencies.items():
        if not entries:
            print(f"  {side}: no cmd=1.0 ticks found")
            continue
        print(f"  {side}:")
        for e in entries:
            peak_str = "n/a" if e["peak_ms"] is None else f"{e['peak_ms']:.1f} ms"
            actual_str = "n/a" if e["actual_max"] is None else f"{e['actual_max']:.3f}"
            print(
                f"    [cmd=1.00 @ t={e['t_send']:.2f}s] "
                f"peak={peak_str}, actual_max={actual_str}"
            )
        values = [e["peak_ms"] for e in entries if e["peak_ms"] is not None]
        if values:
            print(
                f"    peak agg: mean={np.mean(values):.1f}, "
                f"median={np.median(values):.1f}, "
                f"max={np.max(values):.1f} ms (n={len(values)})"
            )


def _zoh_lookup(
    query_t: np.ndarray,
    cmd_t: np.ndarray,
    cmd_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order-hold command lookup at arbitrary query times."""
    idx = np.searchsorted(cmd_t, query_t, side="right") - 1
    valid = (idx >= 0) & (idx < cmd_v.size)
    out = np.full(query_t.shape, np.nan, dtype=np.float64)
    out[valid] = cmd_v[idx[valid]]
    return out, valid


def _fit_fundamental(
    t: np.ndarray,
    y: np.ndarray,
    frequency_hz: float,
) -> Optional[Dict[str, float]]:
    if frequency_hz <= 0.0 or t.size < 4:
        return None
    finite = np.isfinite(t) & np.isfinite(y)
    if finite.sum() < 4:
        return None
    t = t[finite]
    y = y[finite]
    w = 2.0 * math.pi * frequency_hz
    design = np.column_stack([np.ones_like(t), np.sin(w * t), np.cos(w * t)])
    try:
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    offset, sin_coef, cos_coef = (float(x) for x in coef)
    amplitude = math.hypot(sin_coef, cos_coef)
    phase_rad = math.atan2(cos_coef, sin_coef)
    return {"offset": offset, "amplitude": amplitude, "phase_rad": phase_rad}


def _wrap_to_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _default_analysis_max_lag(frequency_hz: float) -> float:
    if frequency_hz > 0.0:
        return min(2.0, 0.75 / frequency_hz)
    return 2.0


def _local_extrema(t: np.ndarray, v: np.ndarray) -> List[Dict]:
    """Return local extrema from a sampled command sequence.

    Consecutive nearly equal command values are treated as one plateau, which
    makes square waves analyzable while preserving single-point sine/triangle
    extrema.
    """
    if t.size < 3 or v.size != t.size:
        return []

    value_range = float(np.max(v) - np.min(v))
    eps = max(1e-6, value_range * 1e-4)
    runs: List[Dict] = []
    start = 0
    for i in range(1, v.size):
        if abs(float(v[i]) - float(v[start])) > eps:
            runs.append(
                {
                    "start": start,
                    "end": i - 1,
                    "value": float(v[start]),
                    "time": float((t[start] + t[i - 1]) * 0.5),
                }
            )
            start = i
    runs.append(
        {
            "start": start,
            "end": v.size - 1,
            "value": float(v[start]),
            "time": float((t[start] + t[-1]) * 0.5),
        }
    )

    extrema: List[Dict] = []
    for i in range(1, len(runs) - 1):
        prev_v = runs[i - 1]["value"]
        this_v = runs[i]["value"]
        next_v = runs[i + 1]["value"]
        if this_v > prev_v + eps and this_v > next_v + eps:
            extrema.append({"kind": "max", "time": runs[i]["time"], "value": this_v})
        elif this_v < prev_v - eps and this_v < next_v - eps:
            extrema.append({"kind": "min", "time": runs[i]["time"], "value": this_v})
    return extrema


def _compute_extrema_response(
    cmd_t: np.ndarray,
    cmd_v: np.ndarray,
    actual_t: np.ndarray,
    actual_v: np.ndarray,
) -> Dict[str, Optional[float]]:
    extrema = _local_extrema(cmd_t, cmd_v)
    values: Dict[str, List[float]] = {
        "local_max_delay_ms": [],
        "local_max_error": [],
        "local_min_delay_ms": [],
        "local_min_error": [],
    }

    for i, ext in enumerate(extrema):
        t_cmd = float(ext["time"])
        t_next = float(extrema[i + 1]["time"]) if i + 1 < len(extrema) else float("inf")
        in_window = (actual_t >= t_cmd) & (actual_t < t_next)
        if not np.any(in_window):
            continue
        window_idx = np.flatnonzero(in_window)
        if ext["kind"] == "max":
            actual_idx = window_idx[int(np.argmax(actual_v[window_idx]))]
            values["local_max_delay_ms"].append((float(actual_t[actual_idx]) - t_cmd) * 1000.0)
            values["local_max_error"].append(float(actual_v[actual_idx]) - float(ext["value"]))
        else:
            actual_idx = window_idx[int(np.argmin(actual_v[window_idx]))]
            values["local_min_delay_ms"].append((float(actual_t[actual_idx]) - t_cmd) * 1000.0)
            values["local_min_error"].append(float(actual_v[actual_idx]) - float(ext["value"]))

    result: Dict[str, Optional[float]] = {}
    for key, vals in values.items():
        result[key] = float(np.mean(vals)) if vals else None
        result[f"{key}_n"] = len(vals)
    return result


def _compute_square_edge_response(
    cmd_t: np.ndarray,
    cmd_v: np.ndarray,
    actual_t: np.ndarray,
    actual_v: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Measure square-wave edge time to the actual extremum in that half-cycle."""
    if cmd_t.size < 3 or actual_t.size < 3:
        return {}

    cmd_range = float(np.max(cmd_v) - np.min(cmd_v))
    eps = max(1e-6, cmd_range * 1e-4)
    unique_values = np.unique(np.round(cmd_v / eps).astype(int)) if eps > 0.0 else np.array([])
    if cmd_range <= eps or unique_values.size > 2:
        return {}

    change_indices = np.flatnonzero(np.abs(np.diff(cmd_v)) > eps) + 1
    values: Dict[str, List[float]] = {
        "square_min_to_max_ms": [],
        "square_max_to_min_ms": [],
    }
    for i, change_idx in enumerate(change_indices):
        old_cmd = float(cmd_v[change_idx - 1])
        new_cmd = float(cmd_v[change_idx])
        t_edge = float(cmd_t[change_idx])
        t_next = (
            float(cmd_t[change_indices[i + 1]])
            if i + 1 < len(change_indices)
            else float("inf")
        )
        in_window = (actual_t >= t_edge) & (actual_t < t_next)
        if not np.any(in_window):
            continue

        window_idx = np.flatnonzero(in_window)
        if new_cmd > old_cmd:
            actual_idx = window_idx[int(np.argmax(actual_v[window_idx]))]
            values["square_min_to_max_ms"].append(
                (float(actual_t[actual_idx]) - t_edge) * 1000.0
            )
        else:
            actual_idx = window_idx[int(np.argmin(actual_v[window_idx]))]
            values["square_max_to_min_ms"].append(
                (float(actual_t[actual_idx]) - t_edge) * 1000.0
            )

    result: Dict[str, Optional[float]] = {}
    for key, vals in values.items():
        result[key] = float(np.mean(vals)) if vals else None
        result[f"{key}_n"] = len(vals)
    return result


def compute_dynamic_response_metrics(
    rows: List[Dict],
    sides: List[str],
    stimulus_frequency_hz: float,
    analysis_max_lag_s: Optional[float] = None,
) -> Dict[str, Dict]:
    """Estimate delay/gain/error for continuously resent command waveforms.

    Delay is estimated by scanning nonnegative lags and maximizing correlation
    between the actual encoder samples and a zero-order-held command signal.
    Error is then computed after applying that best delay to the command.
    """
    results: Dict[str, Dict] = {}
    max_lag_s = (
        float(analysis_max_lag_s)
        if analysis_max_lag_s is not None
        else _default_analysis_max_lag(stimulus_frequency_hz)
    )
    max_lag_s = max(0.0, max_lag_s)

    for side in sides:
        side_cmds = [
            r
            for r in rows
            if r["kind"] == "tick" and r["side"] == side and r.get("cmd_sent")
        ]
        side_samples = [r for r in rows if r["kind"] == "sample" and r["side"] == side]
        side_cmds.sort(key=lambda r: r.get("cmd_send_t_monotonic_s") or r["tick_t_monotonic_s"])
        side_samples.sort(key=lambda r: r["sample_t_monotonic_s"])

        if len(side_cmds) < 3 or len(side_samples) < 5:
            results[side] = {
                "ok": False,
                "reason": "need at least 3 command sends and 5 samples",
                "cmd_sends": len(side_cmds),
                "samples": len(side_samples),
            }
            continue

        cmd_t = np.asarray(
            [r.get("cmd_send_t_monotonic_s") or r["tick_t_monotonic_s"] for r in side_cmds],
            dtype=np.float64,
        )
        cmd_v = np.asarray([r["cmd"] for r in side_cmds], dtype=np.float64)
        actual_t = np.asarray([r["sample_t_monotonic_s"] for r in side_samples], dtype=np.float64)
        actual_v = np.asarray([r["actual"] for r in side_samples], dtype=np.float64)

        sample_span_s = float(actual_t[-1] - actual_t[0]) if actual_t.size > 1 else 0.0
        cmd_span_s = float(cmd_t[-1] - cmd_t[0]) if cmd_t.size > 1 else 0.0
        sample_rate_hz = (len(actual_t) - 1) / sample_span_s if sample_span_s > 0 else float("nan")
        cmd_rate_hz = (len(cmd_t) - 1) / cmd_span_s if cmd_span_s > 0 else float("nan")

        sample_dt = np.diff(actual_t)
        cmd_dt = np.diff(cmd_t)
        dt_candidates = [
            float(np.median(x[x > 0]))
            for x in (sample_dt, cmd_dt)
            if np.any(x > 0)
        ]
        lag_step_s = min(dt_candidates) * 0.5 if dt_candidates else 0.005
        lag_step_s = min(max(lag_step_s, 0.001), 0.010)

        zero_cmd_v, zero_valid = _zoh_lookup(actual_t, cmd_t, cmd_v)
        common = zero_valid & np.isfinite(actual_v) & np.isfinite(zero_cmd_v)
        if common.sum() < 5:
            results[side] = {
                "ok": False,
                "reason": "no overlapping command/sample window",
                "cmd_sends": len(side_cmds),
                "samples": len(side_samples),
                "sample_rate_hz": sample_rate_hz,
                "cmd_rate_hz": cmd_rate_hz,
            }
            continue

        raw_err = actual_v[common] - zero_cmd_v[common]
        cmd_range = float(np.max(zero_cmd_v[common]) - np.min(zero_cmd_v[common]))
        actual_range = float(np.max(actual_v[common]) - np.min(actual_v[common]))
        range_gain = actual_range / cmd_range if cmd_range > 1e-9 else None
        extrema_response = _compute_extrema_response(cmd_t, cmd_v, actual_t, actual_v)
        square_edge_response = _compute_square_edge_response(cmd_t, cmd_v, actual_t, actual_v)

        best = None
        lag_count = int(max_lag_s / lag_step_s) + 1
        for lag_idx in range(max(1, lag_count)):
            lag_s = lag_idx * lag_step_s
            shifted_t = actual_t - lag_s
            lag_cmd_v, lag_valid = _zoh_lookup(shifted_t, cmd_t, cmd_v)
            mask = lag_valid & np.isfinite(actual_v) & np.isfinite(lag_cmd_v)
            if mask.sum() < 5:
                continue
            centered_cmd = lag_cmd_v[mask] - float(np.mean(lag_cmd_v[mask]))
            centered_actual = actual_v[mask] - float(np.mean(actual_v[mask]))
            denom = float(np.linalg.norm(centered_cmd) * np.linalg.norm(centered_actual))
            if denom <= 1e-12:
                continue
            corr = float(np.dot(centered_cmd, centered_actual) / denom)
            err = actual_v[mask] - lag_cmd_v[mask]
            rmse = float(np.sqrt(np.mean(err * err)))
            if (
                best is None
                or corr > best["corr"]
                or (abs(corr - best["corr"]) <= 1e-6 and rmse < best["rmse"])
            ):
                best = {
                    "lag_s": lag_s,
                    "corr": corr,
                    "rmse": rmse,
                    "mask": mask,
                    "lag_cmd_v": lag_cmd_v,
                }

        if best is None:
            lag_s = None
            lag_corr = None
            lag_err = raw_err
        else:
            lag_s = float(best["lag_s"])
            lag_corr = float(best["corr"])
            lag_err = actual_v[best["mask"]] - best["lag_cmd_v"][best["mask"]]

        cmd_fit = _fit_fundamental(actual_t[common], zero_cmd_v[common], stimulus_frequency_hz)
        actual_fit = _fit_fundamental(actual_t[common], actual_v[common], stimulus_frequency_hz)
        fundamental_gain = None
        fundamental_delay_ms = None
        if cmd_fit and actual_fit and cmd_fit["amplitude"] > 1e-9 and stimulus_frequency_hz > 0.0:
            fundamental_gain = actual_fit["amplitude"] / cmd_fit["amplitude"]
            phase_lag = _wrap_to_pi(cmd_fit["phase_rad"] - actual_fit["phase_rad"])
            fundamental_delay_ms = phase_lag / (2.0 * math.pi * stimulus_frequency_hz) * 1000.0

        results[side] = {
            "ok": True,
            "cmd_sends": len(side_cmds),
            "samples": len(side_samples),
            "sample_rate_hz": sample_rate_hz,
            "cmd_rate_hz": cmd_rate_hz,
            "stimulus_frequency_hz": stimulus_frequency_hz,
            "delay_ms": None if lag_s is None else lag_s * 1000.0,
            "delay_corr": lag_corr,
            "delay_resolution_ms": lag_step_s * 1000.0,
            "range_gain": range_gain,
            "cmd_range": cmd_range,
            "actual_range": actual_range,
            "zero_lag_rmse": float(np.sqrt(np.mean(raw_err * raw_err))),
            "zero_lag_mae": float(np.mean(np.abs(raw_err))),
            "lag_rmse": float(np.sqrt(np.mean(lag_err * lag_err))),
            "lag_mae": float(np.mean(np.abs(lag_err))),
            "lag_p95_abs": float(np.percentile(np.abs(lag_err), 95)),
            "lag_max_abs": float(np.max(np.abs(lag_err))),
            "lag_mean_error": float(np.mean(lag_err)),
            "fundamental_gain": fundamental_gain,
            "fundamental_delay_ms": fundamental_delay_ms,
            **extrema_response,
            **square_edge_response,
        }
    return results


def _fmt_optional(value: object, suffix: str = "", precision: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        if not np.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.{precision}f}{suffix}"
    except (TypeError, ValueError):
        return "n/a"


def print_dynamic_response_metrics(metrics: Dict[str, Dict]) -> None:
    print()
    print("Dynamic response (continuous command waveform -> actual encoder):")
    for side, m in metrics.items():
        if not m.get("ok"):
            print(
                f"  {side}: unavailable ({m.get('reason', 'unknown')}; "
                f"cmds={m.get('cmd_sends', 0)}, samples={m.get('samples', 0)})"
            )
            continue
        print(f"  {side}:")
        print(f"    stimulus={m['stimulus_frequency_hz']:.3f} Hz")
        print(f"    cmd_rate={m['cmd_rate_hz']:.1f} Hz")
        print(f"    sample_rate={m['sample_rate_hz']:.1f} Hz")
        print(f"    delay={_fmt_optional(m['delay_ms'], ' ms', 1)}")
        print(f"    range_gain={_fmt_optional(m['range_gain'], precision=3)}")
        print(f"    mae={m['lag_mae']:.3f}")
        print(
            "    local_max_delay="
            f"{_fmt_optional(m.get('local_max_delay_ms'), ' ms', 1)}"
        )
        print(
            "    local_max_error="
            f"{_fmt_optional(m.get('local_max_error'), precision=3)}"
        )
        print(
            "    local_min_delay="
            f"{_fmt_optional(m.get('local_min_delay_ms'), ' ms', 1)}"
        )
        print(
            "    local_min_error="
            f"{_fmt_optional(m.get('local_min_error'), precision=3)}"
        )
        if "square_min_to_max_ms" in m or "square_max_to_min_ms" in m:
            print(
                "    square_min_to_max_time="
                f"{_fmt_optional(m.get('square_min_to_max_ms'), ' ms', 1)}"
            )
            print(
                "    square_max_to_min_time="
                f"{_fmt_optional(m.get('square_max_to_min_ms'), ' ms', 1)}"
            )


def mean_latency_label(latencies: Dict[str, Dict[str, List[Dict]]], side: str) -> str:
    entries: List[Dict] = []
    for direction in ("close", "open"):
        entries.extend(latencies.get(side, {}).get(direction, []))

    def latency_values(key: str, settled_only: bool = False) -> List[float]:
        return [
            e[key]
            for e in entries
            if e[key] is not None and (not settled_only or e.get("pre_settled"))
        ]

    def fmt_mean(values: List[float]) -> str:
        if not values:
            return "n/a"
        return f"{np.mean(values):.1f} ms"

    def endpoint_values(from_value: float, to_value: float) -> List[float]:
        return [
            e["complete_ms"]
            for e in entries
            if e["complete_ms"] is not None
            and e.get("pre_settled")
            and abs(e["from"] - from_value) <= SETTLED_TOL
            and abs(e["to"] - to_value) <= SETTLED_TOL
        ]

    echo_values = latency_values("echo_ms")
    motion_values = latency_values("motion_ms", settled_only=True)
    n = len(motion_values) if motion_values else len(echo_values)

    return (
        f"mean latency (n={n})\n"
        f"echo: {fmt_mean(echo_values)}\n"
        f"motion: {fmt_mean(motion_values)}\n"
        f"1-0: {fmt_mean(endpoint_values(1.0, 0.0))}"
    )


def continuous_peak_latency_label(
    peak_latencies: Dict[str, List[Dict]],
    side: str,
) -> str:
    entries = peak_latencies.get(side, [])
    values = [e["peak_ms"] for e in entries if e["peak_ms"] is not None]
    actual_maxes = [e["actual_max"] for e in entries if e["actual_max"] is not None]

    peak_mean = "n/a" if not values else f"{np.mean(values):.1f} ms"
    actual_mean = "n/a" if not actual_maxes else f"{np.mean(actual_maxes):.3f}"
    return (
        f"continuous latency (n={len(values)})\n"
        f"cmd 1→actual max: {peak_mean}\n"
        f"mean actual max: {actual_mean}"
    )


def dynamic_response_label(
    response_metrics: Dict[str, Dict],
    side: str,
) -> str:
    m = response_metrics.get(side, {})
    if not m.get("ok"):
        return "dynamic_response=n/a"
    return (
        f"stimulus={m['stimulus_frequency_hz']:.3f} Hz\n"
        f"cmd_rate={m['cmd_rate_hz']:.1f} Hz\n"
        f"sample_rate={m['sample_rate_hz']:.1f} Hz\n"
        f"delay={_fmt_optional(m['delay_ms'], ' ms', 1)}\n"
        f"mae={m['lag_mae']:.3f}\n"
        f"local_max_delay={_fmt_optional(m.get('local_max_delay_ms'), ' ms', 1)}\n"
        f"local_max_error={_fmt_optional(m.get('local_max_error'), precision=3)}\n"
        f"local_min_delay={_fmt_optional(m.get('local_min_delay_ms'), ' ms', 1)}\n"
        f"local_min_error={_fmt_optional(m.get('local_min_error'), precision=3)}"
    )


def _annotation_anchor(
    point_sets: List[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, str, str]:
    """Pick the least occupied plot corner for a metrics annotation."""
    candidates = [
        ("upper_left", 0.02, 0.98, "left", "top", (0.00, 0.42), (0.55, 1.05), 0),
        ("upper_right", 0.98, 0.98, "right", "top", (0.58, 1.00), (0.55, 1.05), 20),
        ("lower_left", 0.02, 0.02, "left", "bottom", (0.00, 0.42), (-0.05, 0.45), 0),
        ("lower_right", 0.98, 0.02, "right", "bottom", (0.58, 1.00), (-0.05, 0.45), 0),
    ]

    all_t = np.concatenate([t for t, _v in point_sets if t.size]) if point_sets else np.array([])
    if all_t.size == 0:
        _name, x, y, ha, va, *_rest = candidates[0]
        return x, y, ha, va
    t_min = float(np.min(all_t))
    t_span = float(np.max(all_t) - t_min)
    if t_span <= 0.0:
        _name, x, y, ha, va, *_rest = candidates[0]
        return x, y, ha, va

    best = None
    for candidate in candidates:
        _name, x, y, ha, va, x_window, y_window, penalty = candidate
        score = penalty
        for t, v in point_sets:
            if t.size == 0 or v.size == 0:
                continue
            x_norm = (t - t_min) / t_span
            mask = (
                (x_window[0] <= x_norm)
                & (x_norm <= x_window[1])
                & (y_window[0] <= v)
                & (v <= y_window[1])
            )
            score += int(np.count_nonzero(mask))
        if best is None or score < best[0]:
            best = (score, x, y, ha, va)

    _score, x, y, ha, va = best
    return x, y, ha, va


def plot_results(
    rows: List[Dict],
    sides: List[str],
    path: Path,
    stamp: str,
    latencies: Dict[str, Dict[str, List[Dict]]],
    mode: str,
    peak_latencies: Optional[Dict[str, List[Dict]]] = None,
    response_metrics: Optional[Dict[str, Dict]] = None,
) -> None:
    n = len(sides)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(10, 3 * n))
    axes = np.atleast_1d(axes)
    for ax, side in zip(axes, sides, strict=True):
        side_ticks = [r for r in rows if r["kind"] == "tick" and r["side"] == side]
        side_samples = [r for r in rows if r["kind"] == "sample" and r["side"] == side]
        side_timeouts = [r for r in rows if r["kind"] == "timeout" and r["side"] == side]
        side_cmds = [r for r in side_ticks if r.get("cmd_sent")]
        plotted_points: List[tuple[np.ndarray, np.ndarray]] = []

        if side_cmds:
            cmd_t = np.asarray(
                [r.get("cmd_send_t_monotonic_s") or r["tick_t_monotonic_s"] for r in side_cmds]
            )
            cmd = np.asarray([r["cmd"] for r in side_cmds], dtype=float)
            plotted_points.append((cmd_t, cmd))
            ax.plot(
                cmd_t,
                cmd,
                linestyle="None",
                marker=".",
                label="cmd sent",
                color="tab:blue",
                ms=7,
            )

        if side_samples:
            sample_t = np.asarray(
                [
                    r.get("state_read_t_monotonic_s") or r["sample_t_monotonic_s"]
                    for r in side_samples
                ]
            )
            actual = np.asarray([r["actual"] for r in side_samples], dtype=float)
            plotted_points.append((sample_t, actual))
            ax.plot(
                sample_t,
                actual,
                linestyle="None",
                marker=".",
                label="state read (gPO/255)",
                color="tab:red",
                ms=7,
            )

        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(f"{side} pos")
        ax.set_title(
            f"{side} gripper "
            f"(cmds={len(side_cmds)}, status_requests={len(side_ticks)}, "
            f"states={len(side_samples)}, timeouts={len(side_timeouts)})"
        )
        x_text, y_text, ha_text, va_text = _annotation_anchor(plotted_points)
        ax.text(
            x_text,
            y_text,
            (
                dynamic_response_label(response_metrics or {}, side)
                if mode in DYNAMIC_MODES
                else continuous_peak_latency_label(peak_latencies or {}, side)
                if mode == "continuous"
                else mean_latency_label(latencies, side)
            ),
            transform=ax.transAxes,
            ha=ha_text,
            va=va_text,
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle("Robotiq Gripper latency")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    sides = [s.strip() for s in args.sides.split(",") if s.strip()]
    for s in sides:
        if s not in ("left", "right"):
            raise ValueError(f"--sides got {s!r}; expected 'left' and/or 'right'")
    if not sides:
        raise ValueError("--sides resolved to empty list")

    if not 0.0 <= args.min_cmd <= 1.0 or not 0.0 <= args.max_cmd <= 1.0:
        raise ValueError("--min-cmd and --max-cmd must be in [0,1]")
    if args.min_cmd >= args.max_cmd:
        raise ValueError("--min-cmd must be smaller than --max-cmd")
    if not 0.0 <= args.speed <= 1.0 or not 0.0 <= args.force <= 1.0:
        raise ValueError("--speed and --force must be in [0,1]")

    if args.steps is None:
        if args.mode == "continuous":
            steps = make_continuous_steps(args.min_cmd, args.max_cmd)
        else:
            steps = DEFAULT_STEPS
    else:
        steps = tuple(float(x) for x in args.steps.split(","))
    if not steps or any(not 0.0 <= s <= 1.0 for s in steps):
        raise ValueError(f"--steps must be a list of values in [0,1]; got {steps}")

    if args.mode in PERIODIC_MODES:
        if args.signal_frequency <= 0.0:
            raise ValueError("--signal-frequency must be positive for square/triangle/sine modes")
        if args.cycles <= 0.0:
            raise ValueError("--cycles must be positive for square/triangle/sine modes")
    if args.analysis_max_lag is not None and args.analysis_max_lag < 0.0:
        raise ValueError("--analysis-max-lag must be nonnegative")

    if args.rate is None:
        args.rate = (
            whole_body_control_rate_hz() if args.mode in DYNAMIC_MODES else DEFAULT_STEPPED_RATE_HZ
        )

    if args.rate <= 0 or args.hold <= 0:
        raise ValueError("--rate and --hold must be positive")
    period_s = 1.0 / args.rate

    if args.duration is None:
        if args.mode == "continuous":
            args.duration = len(steps) * period_s + 2.0
        elif args.mode in PERIODIC_MODES:
            args.duration = args.cycles / args.signal_frequency
        else:
            args.duration = len(steps) * args.hold + 2.0
    if args.duration <= 0:
        raise ValueError("--duration must be positive")

    if args.function_code not in (0x03, 0x04):
        raise ValueError(f"--function-code must be 3 or 4; got {args.function_code:#x}")
    if args.status_timeout <= 0:
        raise ValueError("--status-timeout must be positive")

    stimulus_frequency_hz = (
        1.0 / (len(steps) * period_s)
        if args.mode == "continuous"
        else args.signal_frequency
        if args.mode in PERIODIC_MODES
        else 0.0
    )

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"gripper_cmd_actual_{stamp}.csv"
    png_path = out_dir / f"gripper_cmd_actual_{stamp}.png"

    print("Initialising Robot()...")
    robot = Robot()
    arms: Dict[str, Arm] = {side: get_arm(robot, side) for side in sides}
    monitors: Dict[str, RobotiqStatusMonitor] = {}

    rows: List[Dict] = []
    interrupted = {"flag": False}

    def _on_sigint(signum, frame):
        interrupted["flag"] = True

    signal.signal(signal.SIGINT, _on_sigint)

    try:
        print(f"Activating grippers: {sides}...")
        for arm in arms.values():
            send_activate(arm)
        time.sleep(0.2)

        print("Warming up status poll...")
        for side, arm in arms.items():
            warmup_check(arm, side, args.function_code)
        monitors = {
            side: RobotiqStatusMonitor(arm, side=side, function_code=args.function_code)
            for side, arm in arms.items()
        }

        if args.mode == "continuous":
            mode_detail = (
                f"ramp_steps={len(steps)}, cycle={len(steps) * period_s:.2f}s, "
                f"range=[{args.min_cmd:.2f}, {args.max_cmd:.2f}]"
            )
        elif args.mode in PERIODIC_MODES:
            mode_detail = (
                f"frequency={args.signal_frequency:.3f}Hz, "
                f"cycles={args.duration * args.signal_frequency:.2f}, "
                f"range=[{args.min_cmd:.2f}, {args.max_cmd:.2f}]"
            )
        else:
            mode_detail = f"steps={steps}, hold={args.hold:.2f}s"
        print(
            f"Logging for {args.duration:.1f} s with tick {args.rate:.1f} Hz "
            f"and queued FC03 demux (~{1.0/DRAIN_PERIOD_S:.0f} Hz drain, "
            f"timeout={args.status_timeout:.3f}s) "
            f"(mode={args.mode}, {mode_detail}, "
            f"speed={args.speed:.2f}, force={args.force:.2f}) → {csv_path}"
        )

        # Persistent per-side state.
        last_cmd: Dict[str, Optional[float]] = {side: None for side in sides}

        t0_monotonic_ns = time.monotonic_ns()
        t0 = time.monotonic()
        next_tick = t0
        tick_index = 0
        overruns = 0

        while not interrupted["flag"]:
            now = time.monotonic()
            elapsed = now - t0
            if elapsed >= args.duration:
                break

            # 1. Drain queued status replies and explicit timeouts.
            for side, monitor in monitors.items():
                for parsed in monitor.drain_status_events():
                    rows.append(
                        make_sample_row(
                            side=side,
                            cmd_at_sample=last_cmd[side],
                            parsed=parsed,
                            t0_monotonic_ns=t0_monotonic_ns,
                        )
                    )
                for timeout_event in monitor.expire_timeouts(args.status_timeout):
                    rows.append(
                        make_timeout_row(
                            side=side,
                            timeout_event=timeout_event,
                            t0_monotonic_ns=t0_monotonic_ns,
                            monitor_stats=monitor.stats(),
                        )
                    )

            # 2. Tick: schedule cmd writes and status requests.
            if now >= next_tick:
                cmd = command_for_tick(
                    mode=args.mode,
                    elapsed_s=elapsed,
                    tick_index=tick_index,
                    steps=steps,
                    hold_s=args.hold,
                    signal_frequency_hz=args.signal_frequency,
                    min_cmd=args.min_cmd,
                    max_cmd=args.max_cmd,
                )
                for side, arm in arms.items():
                    monitor = monitors[side]
                    cmd_event = None
                    if args.mode in DYNAMIC_MODES or cmd != last_cmd[side]:
                        cmd_event = send_ee_pass_through_with_timestamps(
                            arm, build_hande_command(cmd, speed=args.speed, force=args.force)
                        )
                        last_cmd[side] = cmd
                    status_request_event = monitor.send_status_request()
                    rows.append(
                        make_tick_row(
                            tick_elapsed=elapsed,
                            side=side,
                            cmd=cmd,
                            t0_monotonic_ns=t0_monotonic_ns,
                            cmd_event=cmd_event,
                            status_request_event=status_request_event,
                            monitor_stats=monitor.stats(),
                        )
                    )
                next_tick += period_s
                tick_index += 1
                if next_tick <= time.monotonic():
                    overruns += 1
                    next_tick = time.monotonic() + period_s

            # 3. Sleep just long enough to keep CPU sane without missing replies.
            sleep_s = min(DRAIN_PERIOD_S, max(0.0, next_tick - time.monotonic()))
            if sleep_s > 0:
                time.sleep(sleep_s)

        if interrupted["flag"]:
            print("Interrupted — flushing partial log.")
        if overruns:
            print(f"NOTE: {overruns} tick overruns (consider lowering --rate).")
    finally:
        if monitors:
            final_t0_monotonic_ns = (
                t0_monotonic_ns if "t0_monotonic_ns" in locals() else time.monotonic_ns()
            )
            for side, monitor in monitors.items():
                for parsed in monitor.drain_status_events():
                    rows.append(
                        make_sample_row(
                            side=side,
                            cmd_at_sample=last_cmd.get(side) if "last_cmd" in locals() else None,
                            parsed=parsed,
                            t0_monotonic_ns=final_t0_monotonic_ns,
                        )
                    )
                for timeout_event in monitor.expire_timeouts(0.0):
                    rows.append(
                        make_timeout_row(
                            side=side,
                            timeout_event=timeout_event,
                            t0_monotonic_ns=final_t0_monotonic_ns,
                            monitor_stats=monitor.stats(),
                        )
                    )
        monitor_stats_by_side = {
            side: monitor.stats() for side, monitor in monitors.items()
        }
        if rows:
            print(f"Captured {len(rows)} rows; writing CSV...")
            write_csv(rows, csv_path)
            print(f"Wrote {csv_path}")
            print()
            print_summary(rows, sides, args.duration, args.hold, args.mode)
            if monitor_stats_by_side:
                print("Demux counters:")
                for side, stats in monitor_stats_by_side.items():
                    print(
                        f"  {side}: requests={stats.get('status_requests', 0)}, "
                        f"states={stats.get('statuses_received', 0)}, "
                        f"timeouts={stats.get('status_timeouts', 0)}, "
                        f"fc16_acks={stats.get('fc16_acks_discarded', 0)}, "
                        f"invalid={stats.get('invalid_discarded', 0)}, "
                        f"overflow={stats.get('status_queue_overflow', 0)}, "
                        f"pending={stats.get('pending_status_requests', 0)}"
                    )
            latencies = compute_latencies(rows, sides)
            peak_latencies = compute_continuous_peak_latencies(rows, sides)
            response_metrics = (
                compute_dynamic_response_metrics(
                    rows,
                    sides,
                    stimulus_frequency_hz=stimulus_frequency_hz,
                    analysis_max_lag_s=args.analysis_max_lag,
                )
                if args.mode in DYNAMIC_MODES
                else {}
            )
            if args.mode == "continuous":
                print_continuous_peak_latencies(peak_latencies)
                print_dynamic_response_metrics(response_metrics)
            elif args.mode in PERIODIC_MODES:
                print_dynamic_response_metrics(response_metrics)
                if args.mode == "square":
                    print_latencies(latencies)
            else:
                print_latencies(latencies)
            if not args.no_plot:
                plot_results(
                    rows,
                    sides,
                    png_path,
                    stamp,
                    latencies,
                    args.mode,
                    peak_latencies,
                    response_metrics,
                )
                print(f"Wrote {png_path}")
        else:
            print("No rows captured; nothing to write.")
        for monitor in monitors.values():
            monitor.close()
        try:
            print("Shutting down robot...")
            robot.shutdown()
        except Exception as e:
            print(f"WARN: robot.shutdown() raised {e!r}")


if __name__ == "__main__":
    main()
