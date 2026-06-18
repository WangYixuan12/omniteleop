#!/usr/bin/env python3
"""Merge & summarize client.csv + dexmate.csv from the arm-latency probe.

Joins the lambda-side log (``client.csv``) and the dexmate-side log (the CSV
written by ``probe_arm_latency_dexmate.py``) on the per-rep sequence number,
applies the NTP offset to convert dexmate timestamps into the lambda clock,
then prints a per-stage latency summary (mean / p50 / p95 / max / std, in ms)
and writes a merged CSV with all derived columns.

Usage::

    python scripts/diagnostics/probe_arm_latency_merge.py \
        --client-csv  /home/yixuan/.../arm_latency/client.csv \
        --dexmate-csv /tmp/dexmate_arm_latency.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    return cols, rows


def _to_int(s: str) -> Optional[int]:
    if s is None or s == "":
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return None


def _stats(label: str, deltas_ms: np.ndarray) -> dict:
    if deltas_ms.size == 0:
        return {"stage": label, "n": 0,
                "mean": float("nan"), "p50": float("nan"),
                "p95": float("nan"), "max": float("nan"), "std": float("nan")}
    return {
        "stage": label, "n": int(deltas_ms.size),
        "mean": float(np.mean(deltas_ms)),
        "p50": float(np.median(deltas_ms)),
        "p95": float(np.percentile(deltas_ms, 95)),
        "max": float(np.max(deltas_ms)),
        "std": float(np.std(deltas_ms)),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--client-csv", required=True)
    p.add_argument("--dexmate-csv", required=True)
    p.add_argument(
        "--out",
        default=None,
        help="Output merged CSV path. Defaults to <client>_merged.csv.",
    )
    args = p.parse_args()

    client_path = Path(args.client_csv)
    dexmate_path = Path(args.dexmate_csv)
    client_cols, client_rows = _read_csv(client_path)
    _dexmate_cols, dexmate_rows = _read_csv(dexmate_path)

    # Index dexmate rows by sequence (drop sequence=0 from the hold-loop).
    dexmate_by_seq: dict[int, dict[str, str]] = {}
    for row in dexmate_rows:
        seq = _to_int(row.get("sequence", "0"))
        if seq is None or seq <= 0:
            continue
        if seq in dexmate_by_seq:
            raise SystemExit(f"dexmate.csv has duplicate sequence={seq}")
        dexmate_by_seq[seq] = row

    if not client_rows:
        raise SystemExit("client.csv has no rows")

    pre_ns = _to_int(client_rows[0].get("ntp_offset_ns_pre")) or 0
    post_ns = _to_int(client_rows[0].get("ntp_offset_ns_post")) or 0
    drift_ms = abs(post_ns - pre_ns) / 1e6
    if drift_ms > 1.0:
        print(
            f"WARN: NTP offset drifted {drift_ms:.3f} ms between pre/post "
            f"(pre={pre_ns} ns, post={post_ns} ns). Using pre."
        )

    # Build merged rows.
    merged_rows: list[dict] = []
    missing_seqs: list[int] = []
    for crow in client_rows:
        seq = _to_int(crow.get("sequence")) or 0
        drow = dexmate_by_seq.get(seq)
        t_recv = _to_int(drow["t_recv_ns_dexmate"]) if drow else None
        client_ts = _to_int(drow["client_timestamp_ns"]) if drow else None
        if drow is None:
            missing_seqs.append(seq)

        out: dict = dict(crow)
        out["t_recv_ns_dexmate"] = "" if t_recv is None else t_recv
        out["client_timestamp_ns_echo"] = "" if client_ts is None else client_ts
        # `Robot.query_ntp()` returns offset = dexmate_clock - lambda_clock,
        # so to convert a dexmate timestamp to lambda time we *subtract*.
        out["t2_lambda_ns"] = "" if t_recv is None else t_recv - pre_ns
        merged_rows.append(out)

    if missing_seqs:
        print(
            f"WARN: {len(missing_seqs)} client rows have no matching dexmate row "
            f"(sequences: {missing_seqs}). Check that the sniffer was running "
            f"and that ROBOT_NAME matches on both hosts."
        )

    # Compute deltas, in ms, treating sentinel/missing as NaN.
    def delta_ms(end_key: str, start_key: str, rows: list[dict]) -> np.ndarray:
        vals = []
        for row in rows:
            end = _to_int(row.get(end_key))
            start = _to_int(row.get(start_key))
            if end is None or start is None or end <= 0 or start <= 0:
                vals.append(np.nan)
            else:
                vals.append((end - start) / 1e6)
        return np.array(vals, dtype=np.float64)

    stat_rows = [r for r in merged_rows if _to_int(r.get("is_warmup")) == 0]
    summaries = [
        ("T1->T2  send -> robot receipt (NTP-corrected, cross-machine)",
         delta_ms("t2_lambda_ns", "t1_send_ns", stat_rows)),
        ("T2->T4  receipt -> motion start (robot clock, skew-immune)",
         delta_ms("t4_robot_ns", "t_recv_ns_dexmate", stat_rows)),
        ("T4->T5  motion -> arrival (robot clock, skew-immune)",
         delta_ms("t5_robot_ns", "t4_robot_ns", stat_rows)),
        ("T1->T4_local  send -> motion observed on lambda",
         delta_ms("t4_local_ns", "t1_send_ns", stat_rows)),
        ("T1->T5_local  send -> arrival observed on lambda",
         delta_ms("t5_local_ns", "t1_send_ns", stat_rows)),
    ]

    # Write merged CSV.
    extra_cols = ["t_recv_ns_dexmate", "client_timestamp_ns_echo", "t2_lambda_ns"]
    delta_cols = ["t1_to_t2_ms", "t2_to_t4_robot_ms", "t4_to_t5_robot_ms",
                  "t1_to_t4_local_ms", "t1_to_t5_local_ms"]
    out_cols = client_cols + extra_cols + delta_cols

    # Per-row deltas (over all rows, not just stat).
    all_deltas = {
        "t1_to_t2_ms":      delta_ms("t2_lambda_ns",  "t1_send_ns",        merged_rows),
        "t2_to_t4_robot_ms":delta_ms("t4_robot_ns",   "t_recv_ns_dexmate", merged_rows),
        "t4_to_t5_robot_ms":delta_ms("t5_robot_ns",   "t4_robot_ns",       merged_rows),
        "t1_to_t4_local_ms":delta_ms("t4_local_ns",   "t1_send_ns",        merged_rows),
        "t1_to_t5_local_ms":delta_ms("t5_local_ns",   "t1_send_ns",        merged_rows),
    }
    for i, row in enumerate(merged_rows):
        for k, arr in all_deltas.items():
            v = arr[i]
            row[k] = "" if np.isnan(v) else f"{v:.6f}"

    out_path = (
        Path(args.out)
        if args.out
        else client_path.with_name(client_path.stem + "_merged.csv")
    )
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged_rows)
    print(f"Wrote {out_path}")

    # Summary table.
    print("\nSummary (excluding warm-up; all values in ms):")
    header = f"{'stage':<60} {'n':>4} {'mean':>9} {'p50':>9} {'p95':>9} {'max':>9} {'std':>9}"
    print(header)
    print("-" * len(header))
    for label, arr in summaries:
        finite = arr[~np.isnan(arr)]
        s = _stats(label, finite)
        print(
            f"{s['stage']:<60} {s['n']:>4} "
            f"{s['mean']:>9.2f} {s['p50']:>9.2f} {s['p95']:>9.2f} "
            f"{s['max']:>9.2f} {s['std']:>9.2f}"
        )

    print(
        "\nNote: T3 (robot dispatches command to motor driver) lives inside "
        "the closed-source dexmate-onboard process and is not directly "
        "observable. It sits between T2 and T4_robot."
    )


if __name__ == "__main__":
    main()
