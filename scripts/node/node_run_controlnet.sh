#!/usr/bin/env bash
# structure-preserving diffusion baseline run: ControlNet-Canny, N images x seeds, evaluated with the same
# frozen SegFormer + clean-fid as the rest of the benchmark. reuses the existing prepped data + reference.
# durable: --resume + per-seed jsons. env: N (default 1000), SEEDS (default "42 43 44"), CS (cond scale, 1.0).
set -u
source "$HOME/gvenv/bin/activate"
export HF_HOME="$HOME/hf_cache" HF_HUB_DISABLE_TELEMETRY=1
PY="$HOME/gvenv/bin/python"; GEN="$HOME/gen"; DATA="$HOME/gen/data"; RES="$HOME/gen/results"
N="${N:-1000}"; SEEDS="${SEEDS:-42 43 44}"; CS="${CS:-1.0}"
G="$DATA/gta5/images"; M="$DATA/gta5/masks"; C="$DATA/cityscapes/images"; LOG="$RES/controlnet.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "===== controlnet-canny run: N=$N seeds=[$SEEDS] cond_scale=$CS ====="
for seed in $SEEDS; do
  out="$DATA/controlnet_seed${seed}"
  log "controlnet seed ${seed}"
  "$PY" "$GEN/controlnet_translate.py" --src "$G" --out "$out" --seed "$seed" --steps 30 --size 512 \
    --cond_scale "$CS" --limit "$N" --resume 2>&1 | tee -a "$LOG" || log "WARN controlnet seed${seed} gen failed"
  if [ ! -f "$RES/controlnet_seed${seed}.json" ]; then
    "$PY" "$GEN/eval_generality.py" --images "$out" --masks "$M" --cityscapes "$C" \
      --out "$RES/controlnet_seed${seed}.json" --tag "controlnet_seed${seed}" 2>&1 | tee -a "$LOG" || log "WARN eval seed${seed} failed"
  fi
done
"$PY" "$GEN/aggregate_seeds.py" --results "$RES" 2>&1 | tee -a "$LOG" || log "WARN aggregate failed"
log "===== controlnet run DONE ====="
