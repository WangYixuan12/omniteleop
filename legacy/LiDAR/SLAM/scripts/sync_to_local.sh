#!/usr/bin/env bash
# Sync the repo's SLAM/ dir (lives on an sshfs mount the docker daemon cannot
# bind-mount) to a local ext4 runtime dir that compose mounts as /slam.
# maps/ and debug/ are preserved on the local side (never deleted by sync).
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="${SLAM_RUNTIME_DIR:-/home/dexmate/yixuan_slam_runtime}"

mkdir -p "$DST/maps" "$DST/debug"
rsync -av --delete \
  --exclude maps/ --exclude debug/ --exclude docker/ --exclude '__pycache__' \
  "$SRC/" "$DST/"
echo "synced $SRC -> $DST"
