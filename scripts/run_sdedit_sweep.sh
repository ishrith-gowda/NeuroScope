#!/usr/bin/env bash
# SDEdit diffusion arm: strength sweep + empty-prompt ablation, at N=200, evaluated with the same frozen
# SegFormer + clean-fid as the rest of the benchmark. headline strength (0.55) runs FIRST so the key
# diffusion point + its eval land before the long tail. --resume makes every condition restart-safe on the
# unreliable local machine. raw/colormatch/cyclegan jsons already exist in RES, so the aggregator emits the
# full cross-family table after each new point.
set -u
REPO="/Volumes/usb drive/neuroscope"; PY="$REPO/.venv/bin/python"
G="/Volumes/usb drive/generality_data"; GEN="$REPO/journal_extension/generality"; RES="$G/results_n200"
export HF_HOME="/Volumes/usb drive/hf_cache" HF_HUB_DISABLE_TELEMETRY=1 PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$RES"; LOG="$RES/sdedit_sweep.log"
log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
evalcond(){ log "eval $1"; "$PY" "$GEN/eval_generality.py" --images "$2" --masks "$G/gta5/masks" \
  --cityscapes "$G/cityscapes/images" --out "$RES/$1.json" --tag "$1" 2>&1 | tee -a "$LOG" || log "WARN eval $1 failed"; }

log "===== sdedit sweep start (N=200) ====="
for S in 55 30 70 40; do
  log "--- sdedit strength 0.$S ---"
  "$PY" "$GEN/sdedit_translate.py" --src "$G/gta5/images" --out "$G/sdedit_s0$S" \
    --strength "0.$S" --steps 30 --size 512 --resume 2>&1 | tee -a "$LOG" || log "WARN gen 0.$S failed"
  evalcond "sdedit_s0$S" "$G/sdedit_s0$S"
  "$PY" "$GEN/aggregate_generality.py" --results "$RES" 2>&1 | tail -3   # refresh table after each point
done
log "--- sdedit empty-prompt 0.55 (ablation) ---"
"$PY" "$GEN/sdedit_translate.py" --src "$G/gta5/images" --out "$G/sdedit_s055_empty" \
  --strength 0.55 --steps 30 --size 512 --prompt "" --resume 2>&1 | tee -a "$LOG" || log "WARN empty gen failed"
evalcond sdedit_s055_empty "$G/sdedit_s055_empty"
"$PY" "$GEN/aggregate_generality.py" --results "$RES" 2>&1 | tee -a "$LOG"
log "===== sdedit sweep DONE ====="
