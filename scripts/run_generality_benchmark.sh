#!/usr/bin/env bash
# self-consistent GTA5->Cityscapes generality benchmark, all conditions at the SAME N and the SAME frozen
# SegFormer evaluator on ONE machine -> fair fidelity(FID)-vs-utility(mIoU) comparison across generative
# families. durable: each condition writes its own results json immediately, headline conditions first, so a
# partial/interrupted run still yields the core result. NOT set -e: a single condition failure is logged and
# the run continues to protect the rest.
set -u
REPO="/Volumes/usb drive/neuroscope"
PY="$REPO/.venv/bin/python"
G="/Volumes/usb drive/generality_data"
GEN="$REPO/journal_extension/generality"
CKPT="$REPO/cluster_backup/node5_generality/checkpoints/100_net_G_A.pth"
RES="$G/results_n200"
export HF_HOME="/Volumes/usb drive/hf_cache" HF_HUB_DISABLE_TELEMETRY=1 PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$RES"
LOG="$RES/run.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
evalcond() { # tag  imagesdir
  log "eval $1 <- $2"
  "$PY" "$GEN/eval_generality.py" --images "$2" --masks "$G/gta5/masks" \
    --cityscapes "$G/cityscapes/images" --out "$RES/$1.json" --tag "$1" 2>&1 | tee -a "$LOG" || log "WARN eval $1 failed"
}

log "===== generality benchmark start (N=200) ====="

# 1) raw baseline (no translation)
log "--- raw ---"
evalcond raw "$G/gta5/images"

# 2) non-learned baseline: rgb histogram / color match
log "--- colormatch ---"
"$PY" "$GEN/color_match_rgb.py" --src "$G/gta5/images" --ref_dir "$G/cityscapes/images" \
  --out "$G/colormatch/images" 2>&1 | tee -a "$LOG" || log "WARN colormatch gen failed"
evalcond colormatch "$G/colormatch/images"

# 3) learned-GAN: CycleGAN (100-epoch G_A)
log "--- cyclegan ---"
"$PY" "$GEN/cyclegan_translate.py" --src "$G/gta5/images" --ckpt "$CKPT" \
  --out "$G/cyclegan/images" 2>&1 | tee -a "$LOG" || log "WARN cyclegan gen failed"
evalcond cyclegan "$G/cyclegan/images"

# 4) learned-diffusion: SDEdit strength sweep (target prompt). order = headline first, then span the frontier
for S in 55 30 70 40; do
  log "--- sdedit strength 0.$S ---"
  "$PY" "$GEN/sdedit_translate.py" --src "$G/gta5/images" --out "$G/sdedit_s0$S" \
    --strength "0.$S" --steps 30 --size 512 2>&1 | tee -a "$LOG" || log "WARN sdedit 0.$S gen failed"
  evalcond "sdedit_s0$S" "$G/sdedit_s0$S"
done

# 5) prompt ablation: empty prompt at strength 0.55 (isolates img2img prior from text guidance)
log "--- sdedit empty-prompt 0.55 ---"
"$PY" "$GEN/sdedit_translate.py" --src "$G/gta5/images" --out "$G/sdedit_s055_empty" \
  --strength 0.55 --steps 30 --size 512 --prompt "" 2>&1 | tee -a "$LOG" || log "WARN sdedit empty gen failed"
evalcond sdedit_s055_empty "$G/sdedit_s055_empty"

# 6) assemble the headline table + fid/miou frontier
log "--- aggregate ---"
"$PY" "$GEN/aggregate_generality.py" --results "$RES" 2>&1 | tee -a "$LOG" || log "WARN aggregate failed"

log "===== generality benchmark DONE ====="
