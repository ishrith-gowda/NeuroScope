#!/usr/bin/env bash
# publication-grade generality sweep on the A30 node. all conditions at N=$N, one frozen SegFormer-b4 +
# clean-fid evaluator (cuda). deterministic conditions (raw/colormatch/cyclegan) once; SDEdit swept over
# STRENGTHS x SEEDS for error bars, plus an empty-prompt ablation per seed. durable: each condition writes
# its own results json immediately and --resume skips finished images, so the run is restart-safe.
#
# env knobs: N (default 1000), SEEDS (default "42 43 44"), STRENGTHS (default "20 30 40 50 60 70").
set -u
source "$HOME/gvenv/bin/activate"
export HF_HOME="$HOME/hf_cache" HF_HUB_DISABLE_TELEMETRY=1
PY="$HOME/gvenv/bin/python"
GEN="$HOME/gen"; DATA="$HOME/gen/data"; RES="$HOME/gen/results"; CKPT="$HOME/gen/ckpt/100_net_G_A.pth"
N="${N:-1000}"; SEEDS="${SEEDS:-42 43 44}"; STRENGTHS="${STRENGTHS:-20 30 40 50 60 70}"; EMPTY_S="${EMPTY_S:-50}"
mkdir -p "$RES"; LOG="$RES/run.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
G="$DATA/gta5/images"; M="$DATA/gta5/masks"; C="$DATA/cityscapes/images"
evalcond(){ # tag imagesdir
  [ -f "$RES/$1.json" ] && { log "skip eval $1 (exists)"; return; }
  log "eval $1"
  "$PY" "$GEN/eval_generality.py" --images "$2" --masks "$M" --cityscapes "$C" \
    --out "$RES/$1.json" --tag "$1" 2>&1 | tee -a "$LOG" || log "WARN eval $1 failed"
}

log "===== publication sweep start: N=$N seeds=[$SEEDS] strengths=[$STRENGTHS] ====="

# data (once)
if [ ! -d "$G" ] || [ "$(ls "$G"/*.png 2>/dev/null | wc -l)" -lt "$N" ]; then
  log "prep data N=$N"
  "$PY" "$GEN/prep_generality_data.py" --out "$DATA" --n_gta5 "$N" --n_city "$N" 2>&1 | tee -a "$LOG"
fi

# deterministic conditions (seed-independent)
evalcond raw "$G"
log "colormatch gen"; "$PY" "$GEN/color_match_rgb.py" --src "$G" --ref_dir "$C" --out "$DATA/colormatch/images" 2>&1 | tee -a "$LOG"
evalcond colormatch "$DATA/colormatch/images"
log "cyclegan gen"; "$PY" "$GEN/cyclegan_translate.py" --src "$G" --ckpt "$CKPT" --out "$DATA/cyclegan/images" 2>&1 | tee -a "$LOG"
evalcond cyclegan "$DATA/cyclegan/images"

# SDEdit strength x seed sweep (for the frontier + error bars)
for seed in $SEEDS; do
  for s in $STRENGTHS; do
    out="$DATA/sdedit_s${s}_seed${seed}"
    log "sdedit strength 0.${s} seed ${seed}"
    "$PY" "$GEN/sdedit_translate.py" --src "$G" --out "$out" --strength "0.${s}" --steps 30 --size 512 \
      --seed "$seed" --resume 2>&1 | tee -a "$LOG" || log "WARN sdedit s${s} seed${seed} gen failed"
    evalcond "sdedit_s${s}_seed${seed}" "$out"
  done
  # empty-prompt ablation at EMPTY_S, per seed
  out="$DATA/sdedit_s${EMPTY_S}_empty_seed${seed}"
  log "sdedit empty-prompt 0.${EMPTY_S} seed ${seed}"
  "$PY" "$GEN/sdedit_translate.py" --src "$G" --out "$out" --strength "0.${EMPTY_S}" --steps 30 --size 512 \
    --seed "$seed" --prompt "" --resume 2>&1 | tee -a "$LOG" || log "WARN sdedit empty seed${seed} gen failed"
  evalcond "sdedit_s${EMPTY_S}_empty_seed${seed}" "$out"
done

# aggregate across seeds -> mean +/- std frontier
"$PY" "$GEN/aggregate_seeds.py" --results "$RES" 2>&1 | tee -a "$LOG" || log "WARN aggregate_seeds failed"
log "===== publication sweep DONE ====="
