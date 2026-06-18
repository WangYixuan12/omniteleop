#!/usr/bin/env bash
# Save the current slam_toolbox map under both formats:
#   <name>.posegraph/.data  (for slam_toolbox localization mode)
#   <name>.pgm/.yaml        (occupancy grid, e.g. for Nav2 / inspection)
# Usage: ./save_map.sh <name>   (runs inside the yixuan-slam-mapping container)
#
# The container writes to /slam/maps (= the local runtime dir; docker cannot
# mount the sshfs results tree). The localization profile keeps reading that
# copy; a second copy goes to $SLAM_RESULTS_DIR/maps for the results archive.
set -euo pipefail

NAME="${1:?usage: save_map.sh <map-name>}"
CONTAINER="${SLAM_CONTAINER:-yixuan-slam-mapping}"
RUNTIME_DIR="${SLAM_RUNTIME_DIR:-/home/dexmate/yixuan_slam_runtime}"
RESULTS_DIR="${SLAM_RESULTS_DIR:-/home/dexmate/yixuan/Dexmate/SLAM}"

docker exec "$CONTAINER" bash -c "source /opt/ros/humble/setup.bash && \
  ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph \
    \"{filename: /slam/maps/$NAME}\" && \
  timeout 25 ros2 run nav2_map_server map_saver_cli -f /slam/maps/$NAME \
    --ros-args -p map_subscribe_transient_local:=true -p save_map_timeout:=15.0"

mkdir -p "$RESULTS_DIR/maps"
copied=0
for ext in posegraph data pgm yaml; do
  src="$RUNTIME_DIR/maps/$NAME.$ext"
  if [[ -f "$src" ]]; then
    cp "$src" "$RESULTS_DIR/maps/"
    copied=$((copied + 1))
  else
    echo "WARN: expected $src missing" >&2
  fi
done
[[ $copied -gt 0 ]] || { echo "ERROR: no map files produced" >&2; exit 1; }

echo "working copy : $RUNTIME_DIR/maps/$NAME.*  (used by localization profile)"
echo "results copy : $RESULTS_DIR/maps/$NAME.*  ($copied files)"
