#!/usr/bin/env bash
# Restart the robot's base-pose publisher and check that it survives 5 seconds.
# Usage: bash scripts/diagnostics/start_basepose_tmux.sh [ssh-host]
set -euo pipefail

HOST=${1:-dexmate}
if ssh "$HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
trap 'echo "ERROR: basepose startup command failed (line $LINENO)" >&2' ERR

log=$(mktemp /tmp/basepose.XXXXXX.log)
tmux kill-session -t '=basepose' 2>/dev/null || true

if ! tmux new-session -d -s basepose \
  "exec >'$log' 2>&1; cd ~/record3d && exec env ROBOT_NAME=dm/vg7ae4b55f3e-1 ZENOH_CONFIG=\$HOME/.dexmate/comm/zenoh/dm_vg7ae4b55f3e-1.dzcfg ~/miniconda3/envs/arkit/bin/python -u tracking/base_pose_pub.py --calib calib_result.json"; then
  echo "ERROR: Could not create basepose session; log: $log" >&2
  cat "$log" >&2
  exit 1
fi

# Check the pane too: remain-on-exit can keep a session after Python exits.
for ((i = 0; i < 5; i++)); do
  sleep 1
  if ! dead=$(tmux display-message -p -t '=basepose:0.0' '#{pane_dead}' 2>/dev/null) || [[ "$dead" != 0 ]]; then
    echo "ERROR: basepose exited during startup; log: $log" >&2
    cat "$log" >&2
    exit 1
  fi
done

echo "basepose survived the 5-second startup check; log: $log"
REMOTE
then
  exit 0
else
  status=$?
  echo "ERROR: basepose launch on $HOST failed (exit $status)" >&2
  exit "$status"
fi
