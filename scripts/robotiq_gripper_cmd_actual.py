"""Standalone Robotiq Hand-E cmd-vs-actual logger.

Drives one or both Hand-E grippers through a step sequence, polls the actual
encoder position via FC03 over the existing dexcontrol EE pass-through reply
channel, and writes a CSV plus PNG of cmd / actual plus latency summary per side.

The collector uses two independent rates inside one thread:
  * Tick rate (default 20 Hz) issues cmd writes (when the value changes) and
    FC03 status requests; emits one "tick" row per side per tick.
  * Drain rate (~500 Hz) reads each side's response slot nonblockingly and
    emits one "sample" row whenever a fresh, parseable FC03 reply lands.

This avoids the lossy snapshot-then-timeout pattern: every reply that did
arrive is captured, regardless of which tick triggered it. The CSV's `kind`
column distinguishes the two streams; sample rows carry only sample columns
and tick rows carry only tick columns — no NaN-fill rows.

Hardware preconditions:
  * Robot config has `enable_ee_pass_through=True` for each requested arm.
  * Detected EE type is UNKNOWN (otherwise dexcontrol does not publish
    pass-through replies and `get_ee_pass_through_response()` returns
    None — the warm-up check below will warn).

Run:
    python scripts/robotiq_gripper_cmd_actual.py
    python scripts/robotiq_gripper_cmd_actual.py --sides right --duration 15
"""

import argparse
import csv
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

from omniteleop.follower.robotiq import (
    build_hande_command,
    build_hande_status_request,
    poll_gripper_status,
    read_gripper_status_event,
    response_token,
    send_activate,
)

DEFAULT_STEPS = (0.0, 1.0, 0.5, 1.0, 0.0)
DEFAULT_HOLD_S = 3.0
DRAIN_PERIOD_S = 0.002  # ~500 Hz nonblocking drain
PRE_MOTION_WINDOW_S = 0.25
SETTLED_TOL = 1.5 / 255.0
ENDPOINT_TOL = 0.02

CSV_FIELDS = [
    "kind",
    "tick_t_monotonic_s",
    "sample_t_monotonic_s",
    "side",
    "cmd",
    "cmd_at_sample",
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
        "--sides",
        default="right",
        help="comma-separated sides to drive (default: right)",
    )
    p.add_argument("--rate", type=float, default=20.0, help="tick rate in Hz")
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="run duration in seconds (default: len(steps)*hold + 2.0, one-shot)",
    )
    p.add_argument(
        "--hold",
        type=float,
        default=DEFAULT_HOLD_S,
        help=f"seconds per step (default: {DEFAULT_HOLD_S:.1f})",
    )
    p.add_argument(
        "--steps",
        default=",".join(str(s) for s in DEFAULT_STEPS),
        help="comma-separated step sequence in [0,1]",
    )
    p.add_argument(
        "--function-code",
        type=lambda s: int(s, 0),
        default=0x03,
        help="Modbus function code for status read (3 or 4; default 3)",
    )
    p.add_argument(
        "--out-dir",
        default="/home/yixuan/omniteleop/Dexmate/plots/gripper",
        help="output directory (default: Dexmate/plots/gripper)",
    )
    p.add_argument(
        "--no-plot", action="store_true", help="skip matplotlib output"
    )
    return p.parse_args()


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


def make_tick_row(tick_elapsed: float, side: str, cmd: float) -> Dict:
    return {
        "kind": "tick",
        "tick_t_monotonic_s": tick_elapsed,
        "sample_t_monotonic_s": "",
        "side": side,
        "cmd": cmd,
        "cmd_at_sample": "",
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


def make_sample_row(
    sample_elapsed: float,
    side: str,
    cmd_at_sample: Optional[float],
    parsed: Dict,
) -> Dict:
    return {
        "kind": "sample",
        "tick_t_monotonic_s": "",
        "sample_t_monotonic_s": sample_elapsed,
        "side": side,
        "cmd": "",
        "cmd_at_sample": "" if cmd_at_sample is None else cmd_at_sample,
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


def write_csv(rows: List[Dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def print_summary(
    rows: List[Dict], sides: List[str], duration_s: float, hold_s: float
) -> None:
    settle_s = min(0.3, hold_s * 0.3)
    sample_rows = [r for r in rows if r["kind"] == "sample"]
    tick_rows = [r for r in rows if r["kind"] == "tick"]
    print(
        f"Captured {len(tick_rows)} tick rows and {len(sample_rows)} sample rows."
    )
    for side in sides:
        side_samples = [r for r in sample_rows if r["side"] == side]
        if not side_samples:
            print(f"{side}: no samples captured")
            continue

        rate_hz = len(side_samples) / duration_s if duration_s > 0 else float("nan")

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
                f"steady-state actual-cmd: mean={err.mean():+.3f}, "
                f"p95(|.|)={np.percentile(np.abs(err), 95):.3f}, "
                f"max(|.|)={np.abs(err).max():.3f}"
            )
        else:
            print(
                f"{side}: {len(side_samples)} samples ({rate_hz:.1f} Hz); "
                "no steady-state samples (all within settle window)"
            )

        flt_counts = Counter(r["gFLT"] for r in side_samples)
        if any(k != 0 for k in flt_counts):
            print(f"  fault breakdown (gFLT): {dict(flt_counts)}")
        obj_counts = Counter(r["gOBJ"] for r in side_samples)
        print(f"  gOBJ histogram: {dict(obj_counts)}")


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
    motion_thresh: float = 0.02,
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
    results: Dict[str, Dict[str, List[Dict]]] = {
        s: {"close": [], "open": []} for s in sides
    }
    for side in sides:
        side_rows: List[tuple] = []
        for r in rows:
            if r["side"] != side:
                continue
            t = (
                r["tick_t_monotonic_s"]
                if r["kind"] == "tick"
                else r["sample_t_monotonic_s"]
            )
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
                            "t_send": r["tick_t_monotonic_s"],
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
            t_next = (
                transitions[i + 1]["t_send"]
                if i + 1 < len(transitions)
                else float("inf")
            )
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
                if (
                    echo_ms is not None
                    and motion_ms is not None
                    and complete_ms is not None
                ):
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
                motion_str = (
                    "n/a" if e["motion_ms"] is None else f"{e['motion_ms']:.1f} ms"
                )
                complete_str = (
                    "n/a" if e["complete_ms"] is None else f"{e['complete_ms']:.1f} ms"
                )
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


def plot_results(
    rows: List[Dict],
    sides: List[str],
    path: Path,
    stamp: str,
    latencies: Dict[str, Dict[str, List[Dict]]],
) -> None:
    n = len(sides)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(10, 3 * n))
    axes = np.atleast_1d(axes)
    for ax, side in zip(axes, sides, strict=True):
        side_ticks = [r for r in rows if r["kind"] == "tick" and r["side"] == side]
        side_samples = [r for r in rows if r["kind"] == "sample" and r["side"] == side]

        if side_ticks:
            tick_t = np.asarray([r["tick_t_monotonic_s"] for r in side_ticks])
            cmd = np.asarray([r["cmd"] for r in side_ticks])
            ax.step(tick_t, cmd, where="post", label="cmd", color="tab:blue", lw=1.5)

        if side_samples:
            sample_t = np.asarray([r["sample_t_monotonic_s"] for r in side_samples])
            actual = np.asarray([r["actual"] for r in side_samples], dtype=float)
            ax.plot(sample_t, actual, label="actual (gPO/255)", color="tab:red", lw=1)

        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(f"{side} pos")
        ax.set_title(
            f"{side} gripper "
            f"(ticks={len(side_ticks)}, samples={len(side_samples)})"
        )
        ax.text(
            0.02,
            0.98,
            mean_latency_label(latencies, side),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
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

    steps = tuple(float(x) for x in args.steps.split(","))
    if not steps or any(not 0.0 <= s <= 1.0 for s in steps):
        raise ValueError(f"--steps must be a list of values in [0,1]; got {steps}")

    if args.duration is None:
        args.duration = len(steps) * args.hold + 2.0

    if args.rate <= 0 or args.duration <= 0 or args.hold <= 0:
        raise ValueError("--rate, --duration, and --hold must be positive")
    period_s = 1.0 / args.rate

    if args.function_code not in (0x03, 0x04):
        raise ValueError(
            f"--function-code must be 3 or 4; got {args.function_code:#x}"
        )

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"gripper_cmd_actual_{stamp}.csv"
    png_path = out_dir / f"gripper_cmd_actual_{stamp}.png"

    print("Initialising Robot()...")
    robot = Robot()
    arms: Dict[str, Arm] = {side: get_arm(robot, side) for side in sides}

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

        print(
            f"Logging for {args.duration:.1f} s with tick {args.rate:.1f} Hz "
            f"and ~{1.0/DRAIN_PERIOD_S:.0f} Hz drain "
            f"(steps={steps}, hold={args.hold:.2f}s) → {csv_path}"
        )

        # Persistent per-side state.
        last_cmd: Dict[str, Optional[float]] = {side: None for side in sides}
        last_token: Dict[str, object] = {
            side: response_token(arm.get_ee_pass_through_response())
            for side, arm in arms.items()
        }

        t0 = time.monotonic()
        next_tick = t0
        overruns = 0

        while not interrupted["flag"]:
            now = time.monotonic()
            elapsed = now - t0
            if elapsed >= args.duration:
                break

            # 1. Drain replies (per side, nonblocking).
            for side, arm in arms.items():
                new_tok, parsed, _advanced = read_gripper_status_event(
                    arm,
                    last_token[side],
                    function_code=args.function_code,
                )
                last_token[side] = new_tok
                if parsed is not None:
                    rows.append(
                        make_sample_row(
                            sample_elapsed=time.monotonic() - t0,
                            side=side,
                            cmd_at_sample=last_cmd[side],
                            parsed=parsed,
                        )
                    )

            # 2. Tick: schedule cmd writes and status requests.
            if now >= next_tick:
                cmd = step_for_time(elapsed, steps, args.hold)
                for side, arm in arms.items():
                    if cmd != last_cmd[side]:
                        arm.send_ee_pass_through_message(build_hande_command(cmd))
                        last_cmd[side] = cmd
                    arm.send_ee_pass_through_message(
                        build_hande_status_request(args.function_code)
                    )
                    rows.append(
                        make_tick_row(tick_elapsed=elapsed, side=side, cmd=cmd)
                    )
                next_tick += period_s
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
        if rows:
            print(f"Captured {len(rows)} rows; writing CSV...")
            write_csv(rows, csv_path)
            print(f"Wrote {csv_path}")
            print()
            print_summary(rows, sides, args.duration, args.hold)
            latencies = compute_latencies(rows, sides)
            print_latencies(latencies)
            if not args.no_plot:
                plot_results(rows, sides, png_path, stamp, latencies)
                print(f"Wrote {png_path}")
        else:
            print("No rows captured; nothing to write.")
        try:
            print("Shutting down robot...")
            robot.shutdown()
        except Exception as e:
            print(f"WARN: robot.shutdown() raised {e!r}")


if __name__ == "__main__":
    main()
