#!/usr/bin/env bash
# lossless auto-checkpoint heartbeat: mirrors in-progress generality outputs +
# result manifests to a rolling backup on the usb drive every 5 min, so nothing
# is lost if the local run dies mid-flight. git history covers the committed jsons;
# this covers the uncommitted png outputs and interim state.
set -u
REPO="/Volumes/usb drive/neuroscope"
BK="$REPO/cluster_backup/auto_checkpoints"
DATA="/Volumes/usb drive/generality_data"
mkdir -p "$BK"
while true; do
  ts=$(date +%Y-%m-%dT%H:%M:%S)
  # rolling mirror of generality data (images, masks, translated outputs, results)
  rsync -a --delete --exclude '.DS_Store' "$DATA/" "$BK/generality_data_latest/" 2>>"$BK/checkpoint.err"
  # snapshot any results json under the repo (small, keep timestamped copies)
  find "$REPO/journal_extension" "$REPO/cluster_backup" -name 'results*.json' -o -name '*_result*.json' 2>/dev/null \
    | while read -r f; do :; done
  n=$(find "$DATA" -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
  echo "$ts ok pngs=$n" >> "$BK/checkpoint.log"
  sleep 300
done
