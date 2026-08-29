#!/usr/bin/env bash
# self-healing supervisor for the local sdedit sweep on unreliable hardware. leaves a healthy run alone;
# if the driver process dies (e.g. the mac slept and the mps job wedged) or hard-stalls (no new image for
# 30 min while work remains), it relaunches the sweep with --resume so it continues from the last saved
# image instead of waiting for a human. exits cleanly once the sweep logs DONE.
set -u
REPO="/Volumes/usb drive/neuroscope"
DATA="/Volumes/usb drive/generality_data"
LOG="$DATA/results_n200/sdedit_sweep.log"
WLOG="$DATA/results_n200/watchdog.log"
TARGET=1000   # 5 sweep conditions x 200 imgs; sweep is complete at/above this
say(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$WLOG"; }

last=-1; stall=0
say "watchdog start"
while true; do
  if grep -q "sdedit sweep DONE" "$LOG" 2>/dev/null; then say "sweep DONE — watchdog exit"; break; fi
  cur=$(find "$DATA"/sdedit_s0* -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
  alive=$(pgrep -f run_sdedit_sweep.sh | wc -l | tr -d ' ')
  if [ "$cur" -le "$last" ]; then stall=$((stall+1)); else stall=0; fi
  last=$cur
  if { [ "$alive" -eq 0 ] || [ "$stall" -ge 3 ]; } && [ "$cur" -lt "$TARGET" ]; then
    say "recover: alive=$alive stall=$stall pngs=$cur -> restart (resume)"
    pkill -f sdedit_translate 2>/dev/null; pkill -f run_sdedit_sweep.sh 2>/dev/null; sleep 3
    caffeinate -dimsu bash "$REPO/scripts/run_sdedit_sweep.sh" >/dev/null 2>&1 &
    stall=0
  else
    say "ok: alive=$alive pngs=$cur stall=$stall"
  fi
  sleep 600
done
