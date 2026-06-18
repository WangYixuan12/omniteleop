#!/usr/bin/env python3
"""Lambda-side arm-command latency probe.

Ping-pongs a step on a single right-arm joint (default ``R_arm_j4``) around
``INIT_RIGHT_ARM_JOINTS`` and records per-rep timestamps:

* **T1** — ``time.time_ns()`` just before the *first* publish of a rep's target.
  Also embedded in the wire message as ``timestamp_ns`` so the dexmate-side
  sniffer can read it back.
* **T4** — first state sample where motion is detected (vel > threshold AND
  |Δpos| > threshold). Captured as both the robot-clock
  ``right_arm.get_timestamp_ns()`` (T4_robot) and the lambda wall-clock time
  we observed the sample (T4_local).
* **T5** — first sample of the stable-arrival run (|pos − target| < tol AND
  |vel| < quiet, for ``--arrival-stable-n`` consecutive samples). Same dual
  clock as T4.

T2 (robot-side command receipt) is captured by ``probe_arm_latency_dexmate.py``
on dexmate. We tag each rep's first publish with a non-zero ``sequence`` field;
the 100 Hz hold-loop publishes that follow use ``sequence=0`` so the sniffer
drops them. Post-run, ``probe_arm_latency_merge.py`` joins the two logs on
``sequence``.

T3 (robot dispatches command to motor driver) lives inside the closed-source
dexmate-onboard process and is intentionally omitted; the merge script reports
(T2, T4_robot) as the bounds.

Pre-flight (do all of these *before* running):

1. ``pgrep -af vr_robot_controller`` returns nothing on lambda.
2. ``dextop node`` is running on dexmate (``ssh dexmate pgrep -af dextop``).
3. The sniffer is up in a tmux pane on dexmate; you saw ``[ready]``.

Run::

    python scripts/diagnostics/probe_arm_latency_client.py --reps 20

Heads-up: ``Robot()`` instantiation moves the head to its predefined home pose
as part of ``_set_default_state``. Make sure the head can safely move there.
"""

from __future__ import annotations

import argparse
import csv
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from dexcomm import RateLimiter
from dexcontrol.robot import Robot
from loguru import logger

from omniteleop.common.vr_mode_const import INIT_RIGHT_ARM_JOINTS


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--joint-idx", type=int, default=3,
                        help="Right-arm joint index (0..6). Default 3 = R_arm_j4.")
    parser.add_argument("--step-rad", type=float, default=0.2,
                        help="Step amplitude per rep (rad).")
    parser.add_argument("--reps", type=int, default=20,
                        help="Number of measurement reps (plus 1 warm-up).")
    parser.add_argument("--out-dir", type=str,
                        default="/home/yixuan/omniteleop/Dexmate/debug/arm_latency",
                        help="Directory for client.csv.")
    parser.add_argument("--vel-thresh", type=float, default=0.05,
                        help="Motion-start velocity threshold (rad/s).")
    parser.add_argument("--pos-thresh", type=float, default=0.005,
                        help="Motion-start displacement gate (rad) — robust to "
                             "vel-estimator noise at rest.")
    parser.add_argument("--pos-tol", type=float, default=0.01,
                        help="Arrival position tolerance (rad).")
    parser.add_argument("--vel-quiet", type=float, default=0.02,
                        help="Arrival velocity threshold (rad/s).")
    parser.add_argument("--arrival-stable-n", type=int, default=5,
                        help="Consecutive samples required to declare arrival.")
    parser.add_argument("--rep-timeout-s", type=float, default=5.0)
    parser.add_argument("--hold-rate-hz", type=float, default=100.0,
                        help="Rate at which the held target is republished "
                             "(matches dexmate-onboard watchdog expectation).")
    parser.add_argument("--poll-rate-hz", type=float, default=500.0,
                        help="State-sample polling rate on lambda.")
    parser.add_argument("--quiescence-s", type=float, default=0.5,
                        help="Sleep between reps.")
    args = parser.parse_args()

    if not (0 <= args.joint_idx < 7):
        raise SystemExit(f"--joint-idx must be in [0, 7), got {args.joint_idx}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "client.csv"

    init = np.array(INIT_RIGHT_ARM_JOINTS, dtype=np.float64)
    j = args.joint_idx

    logger.info("Connecting to robot (heads-up: head will be moved to home) ...")
    robot = Robot()

    try:
        if not robot.right_arm.wait_for_active(5.0):
            raise SystemExit("right_arm did not become active within 5 s")

        logger.info("Moving right arm to INIT_RIGHT_ARM_JOINTS ...")
        robot.right_arm.set_joint_pos(
            init, wait_time=2.0, exit_on_reach=True,
            exit_on_reach_kwargs={"tolerance": 0.02},
        )
        time.sleep(0.5)

        logger.info("Querying NTP offset (pre-run) ...")
        ntp_pre = robot.query_ntp(sample_count=30, device="soc")
        offset_pre_ns = (
            int(round(ntp_pre["offset"] * 1e9)) if ntp_pre.get("success") else 0
        )
        rtt_pre_ms = float(ntp_pre.get("rtt", 0.0)) * 1e3
        logger.info(
            f"NTP pre: offset={offset_pre_ns} ns, rtt={rtt_pre_ms:.2f} ms"
            + ("" if ntp_pre.get("success") else "  [query failed, assuming 0]")
        )

        # ── State-poll thread ────────────────────────────────────────────────
        # The state subscriber inside `right_arm` caches the latest sample; we
        # poll it at high rate and dedupe on `state.timestamp_ns` so each
        # fresh server-side sample is counted exactly once.
        samples: deque[tuple[int, int, float, float]] = deque(maxlen=200_000)
        stop_event = threading.Event()

        def poll_loop() -> None:
            limiter = RateLimiter(args.poll_rate_hz)
            last_ts = -1
            while not stop_event.is_set():
                try:
                    pos = robot.right_arm.get_joint_pos()
                    vel = robot.right_arm.get_joint_vel()
                    ts_robot = int(robot.right_arm.get_timestamp_ns())
                except Exception as exc:
                    logger.debug(f"poll read error: {exc}")
                    limiter.sleep()
                    continue
                if ts_robot != last_ts:
                    samples.append(
                        (time.time_ns(), ts_robot, float(pos[j]), float(vel[j]))
                    )
                    last_ts = ts_robot
                limiter.sleep()

        poll_thread = threading.Thread(target=poll_loop, name="state-poll", daemon=True)
        poll_thread.start()
        # Let the poll thread fill at least one sample before the first publish.
        time.sleep(0.2)

        # ── Per-rep loop ─────────────────────────────────────────────────────
        client_rows: list[dict] = []
        total = args.reps + 1  # rep 0 is warm-up
        # Reach into the arm's dexcomm publisher so we can inject the
        # `sequence` and `timestamp_ns` fields the public `set_joint_pos`
        # interface doesn't expose. If `Arm` ever stops using a dexcomm
        # publisher this will break loudly. See plan doc.
        publisher = robot.right_arm._publisher  # noqa: SLF001 - see comment

        for rep in range(total):
            direction = +1 if rep % 2 == 0 else -1
            target = init.copy()
            target[j] = init[j] + direction * args.step_rad
            seq = rep + 1  # >0; sniffer drops sequence=0 hold-loop publishes

            # Snapshot sample-buffer length so we only consider new samples.
            anchor = len(samples)

            target_list = target.tolist()
            hold_msg = {"pos": target_list, "sequence": 0}

            limiter = RateLimiter(args.hold_rate_hz)
            t1 = time.time_ns()
            publisher.publish({
                "pos": target_list,
                "sequence": seq,
                "timestamp_ns": t1,
            })
            limiter.sleep()

            t4_local: int = -1
            t4_robot: int = -1
            t5_local: int = -1
            t5_robot: int = -1
            stable_count = 0
            deadline = time.monotonic() + args.rep_timeout_s

            while time.monotonic() < deadline:
                publisher.publish(hold_msg)

                cur_len = len(samples)
                for idx in range(anchor, cur_len):
                    ts_local, ts_robot, pos_j, vel_j = samples[idx]
                    if ts_local < t1:
                        continue

                    if t4_local < 0:
                        moving = (
                            abs(vel_j) > args.vel_thresh
                            and abs(pos_j - init[j]) > args.pos_thresh
                        )
                        if moving:
                            t4_local, t4_robot = ts_local, ts_robot

                    if t4_local > 0 and t5_local < 0:
                        at_target = (
                            abs(pos_j - target[j]) < args.pos_tol
                            and abs(vel_j) < args.vel_quiet
                        )
                        if at_target:
                            stable_count += 1
                            if stable_count >= args.arrival_stable_n:
                                first = idx - (args.arrival_stable_n - 1)
                                t5_local = samples[first][0]
                                t5_robot = samples[first][1]
                        else:
                            stable_count = 0

                anchor = cur_len
                if t5_local > 0:
                    break
                limiter.sleep()

            client_rows.append({
                "rep": rep,
                "is_warmup": int(rep == 0),
                "direction": direction,
                "sequence": seq,
                "t1_send_ns": t1,
                "t4_local_ns": t4_local,
                "t4_robot_ns": t4_robot,
                "t5_local_ns": t5_local,
                "t5_robot_ns": t5_robot,
                "target_j_rad": float(target[j]),
                "init_j_rad": float(init[j]),
                "ntp_offset_ns_pre": offset_pre_ns,
            })

            if t5_local < 0:
                logger.warning(
                    f"rep {rep}: arrival not detected within {args.rep_timeout_s:g} s"
                )
            else:
                logger.info(
                    f"rep {rep:>2}  dir={direction:+d}  "
                    f"T1->T4_local={(t4_local - t1) / 1e6:6.1f} ms  "
                    f"T4->T5_robot={(t5_robot - t4_robot) / 1e6:6.1f} ms  "
                    f"T1->T5_local={(t5_local - t1) / 1e6:6.1f} ms"
                )

            time.sleep(args.quiescence_s)

        # ── Tear-down ────────────────────────────────────────────────────────
        logger.info("Returning right arm to INIT ...")
        robot.right_arm.set_joint_pos(
            init, wait_time=2.0, exit_on_reach=True,
            exit_on_reach_kwargs={"tolerance": 0.02},
        )

        stop_event.set()
        poll_thread.join(timeout=2.0)

        logger.info("Querying NTP offset (post-run) ...")
        ntp_post = robot.query_ntp(sample_count=30, device="soc")
        offset_post_ns = (
            int(round(ntp_post["offset"] * 1e9)) if ntp_post.get("success") else 0
        )
        for row in client_rows:
            row["ntp_offset_ns_post"] = offset_post_ns
        drift_ms = abs(offset_post_ns - offset_pre_ns) / 1e6
        logger.info(
            f"NTP post: offset={offset_post_ns} ns  (drift vs pre: {drift_ms:.3f} ms)"
        )

        cols = [
            "rep", "is_warmup", "direction", "sequence",
            "t1_send_ns", "t4_local_ns", "t4_robot_ns",
            "t5_local_ns", "t5_robot_ns",
            "target_j_rad", "init_j_rad",
            "ntp_offset_ns_pre", "ntp_offset_ns_post",
        ]
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(client_rows)
        logger.info(f"Wrote {len(client_rows)} rows to {out_csv}")
        logger.info(
            "Next steps:\n"
            "  1) Ctrl-C the dexmate sniffer; it writes its CSV on exit.\n"
            "  2) If the sniffer wrote into /home/dexmate/Dexmate/... it's\n"
            "     already visible on lambda at /home/yixuan/Dexmate/... (shared\n"
            "     mount). Otherwise scp it over.\n"
            "  3) Merge & summarize:\n"
            f"       python scripts/diagnostics/probe_arm_latency_merge.py \\\n"
            f"           --client-csv {out_csv} \\\n"
            f"           --dexmate-csv <path-to-dexmate.csv>"
        )

    finally:
        try:
            if not robot.is_shutdown():
                robot.shutdown()
        except Exception as exc:
            logger.warning(f"Shutdown error: {exc}")


if __name__ == "__main__":
    main()
