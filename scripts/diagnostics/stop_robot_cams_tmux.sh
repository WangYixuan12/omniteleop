#!/usr/bin/env bash
# Stop the three camera depth publishers + the robot-side node started by
# start_robot_cams_tmux.sh. Publishers go first so they let go of the cameras
# before the node disappears; anything already gone is skipped.
#
#   scripts/diagnostics/stop_robot_cams_tmux.sh           # stop all four
#   ssh dexmate tmux ls                                   # check what is left
set -euo pipefail

HOST=${1:-dexmate}

ssh "$HOST" 'bash -s' <<'REMOTE'
set -u

# stop <session>: C-c first so the ZED SDK closes the camera cleanly, then kill.
stop() {
  if ! tmux has-session -t "$1" 2>/dev/null; then
    echo "skip   $1 (not running)"
    return
  fi
  tmux send-keys -t "$1" C-c
  sleep 2
  tmux kill-session -t "$1" 2>/dev/null || true
  echo "stop   $1"
}

stop lwrist
stop rwrist
stop head
stop node

echo "--- tmux sessions ---"
tmux ls 2>/dev/null || echo "(none)"
REMOTE
