#!/usr/bin/env bash
# Start the robot-side node + the three camera depth publishers, each in its own
# tmux session on the robot. Anything already running is left alone, so this is
# safe to re-run to fill in whatever died.
#
#   scripts/diagnostics/start_robot_cams_tmux.sh          # start what's missing
#   CAMS=wrists scripts/diagnostics/start_robot_cams_tmux.sh   # only the two wrists
#   ssh dexmate -t tmux attach -t head                    # watch one (C-b d to detach)
#   ssh dexmate tmux kill-session -t head                 # stop one
set -euo pipefail

HOST=${1:-dexmate}

# CAMERA_TRACKING=1 opts into pose + IMU telemetry. Disabled by default because
# live GEN_3 and loopback trials failed the strict image-latency gate.
# CAMERA_IMU=1 opts into IMU-only telemetry; validate under the collection load.

# WRIST_HW  published wrist size HxW. Default 360x640 = the teleop demo-recording profile
#           (see below); pass WRIST_HW=720x1280 for arm/room scans that want the raw frame.
# HEAD_PROFILE  hd1200_750 (default: HD1200 capture resized to 1200x750, the geometry pinned
#           in omniteleop/common/head_camera.py; full 16:10 field of view, 1.56x the pixels
#           of SVGA, measured 72 Mbps / max age 139 ms / zero recorder rejects on 2026-09-03)
#           or svga (raw 960x600, the legacy episode 1-11 profile -- its intrinsics no longer
#           match head_camera.py, so only for comparisons). Raw HD1080 FAILS the recorder
#           contract (8.5 Hz simulated, would abort) and must not be used for recording.
ssh "$HOST" "CAMS=${CAMS:-all} WRIST_HW=${WRIST_HW:-360x640} WRIST_RATE=${WRIST_RATE:-30} HEAD_PROFILE=${HEAD_PROFILE:-hd1200_750} CAMERA_TRACKING=${CAMERA_TRACKING:-0} CAMERA_IMU=${CAMERA_IMU:-0} bash -s" <<'REMOTE'
set -u
REPO=~/yixuan/omniteleop_yifan

# CAMS picks which publishers to launch. The node is ensured first either way --
# every publisher needs it -- and an unselected camera is left exactly as it is,
# running or not, so this never stops anything.
CAMS=${CAMS:-all}
case "$CAMS" in all|wrists|head) ;; *) echo "CAMS must be all, wrists or head, got '$CAMS'" >&2; exit 2 ;; esac

# start <session> <pgrep-pattern> <command> [timeout-seconds] [ready-text]
# An already-running process counts as success. With ready-text, a new process
# must print that text before its timeout; otherwise it must survive the entire
# timeout. In either case, failure prevents the next dependency from starting.
start() {
  local session=$1
  local pattern=$2
  local command=$3
  local timeout_s=${4:-5}
  local ready_text=${5:-}
  local deadline
  local failure
  local seen_process=0

  if pgrep -f "$pattern" >/dev/null; then
    echo "skip   $session (already running)"
    return 0
  fi
  tmux kill-session -t "$session" 2>/dev/null || true   # clear a stale same-named session
  tmux new-session -d -s "$session" bash -li            # -li: conda's dexmate env is set up by .bashrc
  tmux send-keys -t "$session" "cd $REPO && $command" C-m
  echo "start  $session: $command"

  if [[ -z "$ready_text" ]]; then
    sleep "$timeout_s"
    if pgrep -f "$pattern" >/dev/null; then
      echo "ready  $session (alive after ${timeout_s}s)"
      return 0
    fi
    failure="exited during its ${timeout_s}s launch check"
  else
    deadline=$((SECONDS + timeout_s))
    while (( SECONDS < deadline )); do
      if pgrep -f "$pattern" >/dev/null; then
        seen_process=1
      elif (( seen_process )); then
        failure="exited before reporting readiness"
        break
      else
        # tmux may not have dispatched the command yet; allow it to appear.
        sleep 1
        continue
      fi
      if tmux capture-pane -p -t "$session" -S -200 2>/dev/null \
          | grep -Fq -- "$ready_text"; then
        echo "ready  $session ($ready_text)"
        return 0
      fi
      sleep 1
    done
    if (( seen_process )); then
      failure=${failure:-"did not report '$ready_text' within ${timeout_s}s"}
    else
      failure="never started within ${timeout_s}s"
    fi
  fi

  echo "fail   $session $failure" >&2
  echo "--- last output from tmux session '$session' ---" >&2
  tmux capture-pane -p -t "$session" -S -80 2>/dev/null >&2 || true
  return 1
}

launch_or_abort() {
  local session=$1
  if ! start "$@"; then
    echo "abort  $session failed; no later camera will be launched" >&2
    exit 1
  fi
}

# A fresh node previously received two five-second waits: one in start(), then
# one before publishers connected. Keep that ten-second startup allowance.
launch_or_abort node "dextop node start" "dextop node start" 10

# Wait until each camera reaches its post-open publishing log before opening the
# next one. This also prevents simultaneous opens from racing the ZEDX_Daemon.
#
# sensor_id IS the topic: the publisher writes sensors/<sensor_id>/{left_rgb,right_rgb,
# info,clock}. Every consumer of the wrists -- wbc_vr_robot.py (_WRIST_SENSOR_IDS), the
# leader headset HUD (_WRIST_STREAM_TO_SENSOR_ID), record_arm_scan.py and check_cams.py --
# reads sensors/<left|right>_wrist_zedm/*. Launching under any other id (e.g. dexcontrol's
# built-in *_wrist_camera) publishes to a topic nothing listens on: the publisher reports a
# healthy 15 fps while the leader logs "Timeout waiting for response from
# .../sensors/left_wrist_zedm/info" and the follower sees zero wrist frames.
# --serial-number is pinned per id in test_wrist_zedm_depth.py (_ARM_SERIALS) and a
# mismatch is rejected, so the arms cannot silently swap.
#
# WRIST_HW selects the published wrist size as HxW (default: 360x640, the teleop profile).
#   * Arm/room scans (README section 2) want the raw frame (WRIST_HW=720x1280) -- 320x240
#     is too coarse for scene ICP, and any 4:3 target ANISOTROPICALLY stretches the 16:9
#     sensor.
#   * Teleop demo recording (README section 4) uses 360x640: same 16:9 aspect, no stretch,
#     a quarter of the pixels. wbc_vr_robot.py --record stores wrist frames uncompressed
#     at --record-rate, so raw 1280x720 x 4 streams costs ~110 MB/s of disk (~6.6 GB/min)
#     versus ~28 MB/s at 640x360. (The 2026-09-03 "3-15 ms of camera-age margin at 720p"
#     and the episode 12 abort were CLOCK DRIFT on the workstation, not camera latency:
#     with the clocks synced on 2026-09-04, real delivery age is ~40 ms head and
#     ~85-100 ms wrists even at raw 720p. Disk is the reason for 640x360, not the gate.)
# The pgrep pattern includes the size, so re-running with a different WRIST_HW replaces
# the running wrist publishers instead of skipping them.
WRIST_HW=${WRIST_HW:-360x640}
case "$WRIST_HW" in
  720x1280) WRIST_RESIZE="--resize-h 0 --resize-w 0" ;;
  *x*)      WRIST_RESIZE="--resize-h ${WRIST_HW%x*} --resize-w ${WRIST_HW#*x}" ;;
  *) echo "WRIST_HW must look like 360x640 (HxW), got '$WRIST_HW'" >&2; exit 2 ;;
esac
# WRIST_RATE: ZED Mini fps (15/30/60). 30 halves the wrist frame period the 10 Hz recorder
# selects against, so the chosen frame is on average ~17 ms fresher; +23 Mbps on the link,
# no disk cost. (The 2026-09-03 numbers "148 ms at 15 Hz vs 132 ms at 30 Hz" included
# ~100 ms of workstation clock drift; re-measure with scripts/diagnostics/check_robot_clocks.py
# reading OK before quoting camera ages again.)
WRIST_RATE=${WRIST_RATE:-30}
case "$WRIST_RATE" in 15|30|60) ;; *) echo "WRIST_RATE must be 15, 30 or 60, got '$WRIST_RATE'" >&2; exit 2 ;; esac
WRIST_ARGS="--rate $WRIST_RATE $WRIST_RESIZE"

case "${CAMERA_TRACKING:-0}" in
  1) TRACKING_ARGS="--enable-tracking --no-area-memory" ;;
  0) TRACKING_ARGS="--no-enable-tracking --no-area-memory" ;;
  *) echo "CAMERA_TRACKING must be 0 or 1" >&2; exit 2 ;;
esac
case "${CAMERA_IMU:-0}" in
  1) TRACKING_ARGS="$TRACKING_ARGS --enable-imu" ;;
  0) ;;
  *) echo "CAMERA_IMU must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "$CAMS" != head ]]; then
  echo "wrist size: $WRIST_HW @ ${WRIST_RATE} Hz  ($WRIST_ARGS)"
  # Left SN12417276 requires SDK self-calibration: live 640x360 tests reduced
  # median vertical stereo error from ~3.8 px to ~0.2 px (2026-09-12).
  # Record the resulting intrinsics per session; still run pixel-quality checks.
  launch_or_abort lwrist "test_wrist_zedm_depth.py --sensor-id left_wrist_zedm .*$WRIST_ARGS --enable-self-calib $TRACKING_ARGS" \
                           "python tests/test_wrist_zedm_depth.py --sensor-id left_wrist_zedm --serial-number 12417276 --no-skip-right-rgb $WRIST_ARGS --enable-self-calib $TRACKING_ARGS" \
                           15 "Publishing on"
  launch_or_abort rwrist "test_wrist_zedm_depth.py --sensor-id right_wrist_zedm .*$WRIST_ARGS $TRACKING_ARGS" \
                           "python tests/test_wrist_zedm_depth.py --sensor-id right_wrist_zedm --serial-number 12930616 --no-skip-right-rgb $WRIST_ARGS $TRACKING_ARGS" \
                           15 "Publishing on"
fi
# HEAD_PROFILE picks the head capture/publish geometry (see the header). The pgrep pattern
# includes the profile's arguments so switching profiles replaces the running publisher.
HEAD_PROFILE=${HEAD_PROFILE:-hd1200_750}
case "$HEAD_PROFILE" in
  hd1200_750) HEAD_ARGS="--resolution HD1200 --crop 0 1200 0 1920 --resize-h 750 --resize-w 1200" ;;
  svga)       HEAD_ARGS="--resolution SVGA --crop 0 600 0 960 --resize-h 0 --resize-w 0" ;;
  *) echo "HEAD_PROFILE must be svga or hd1200_750, got '$HEAD_PROFILE'" >&2; exit 2 ;;
esac
if [[ "$CAMS" != wrists ]]; then
  echo "head profile: $HEAD_PROFILE  ($HEAD_ARGS)"
  launch_or_abort head   "test_head_zedx_depth.py --rate 30 --no-enable-depth --no-skip-right-rgb $HEAD_ARGS $TRACKING_ARGS" \
                           "python tests/test_head_zedx_depth.py --rate 30 --no-enable-depth --no-skip-right-rgb $HEAD_ARGS $TRACKING_ARGS" \
                           15 "Publishing on"
fi

echo "--- tmux sessions ---"
tmux ls
REMOTE
