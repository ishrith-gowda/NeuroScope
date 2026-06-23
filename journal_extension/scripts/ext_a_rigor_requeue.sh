#!/bin/bash
# ext a rigor (tier c): re-run the harmonization eval with per-subject ssim + a
# fixed seed, for the BASE sa-cyclegan baseline and all 5 lambda arms, after the
# ext c eval frees the gpu. this enables (a) the paired wilcoxon on per-subject
# struct ssim (hybrid vs cycle-only) and (b) a published-style base baseline row,
# all from one consistent, reproducible run.
cd ~/neuroscope/code
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

BR=~/neuroscope/preprocessed/brats
UP=~/neuroscope/preprocessed/upenn
CLF=~/neuroscope/checkpoints/domain_classifier.pth
OUT=~/neuroscope/experiments/ext_a_eval
LOG=~/neuroscope/logs/ext_a_rigor_eval.log

echo "=== $(date) waiting for ext c eval to free the gpu ===" | tee -a "$LOG"
while [ ! -f ~/neuroscope/experiments/ext_c/eval_full/matrix_fid.json ]; do sleep 120; done
sleep 30

run() {  # $1=label  $2=checkpoint
  if [ ! -f "$2" ]; then echo "=== skip $1 (no checkpoint $2) ===" | tee -a "$LOG"; return; fi
  echo "=== $(date) eval $1 ===" | tee -a "$LOG"
  python journal_extension/scripts/eval_harmonization_correct.py \
    --checkpoint "$2" --label "$1" \
    --brats_dir "$BR" --upenn_dir "$UP" --domain_clf "$CLF" --output_dir "$OUT" \
    >> "$LOG" 2>&1
}

run base ~/neuroscope/checkpoints/base_sa_cyclegan_best.pth
for L in 0.0 0.1 0.5 1.0 2.0; do
  run "lambda$L" ~/neuroscope/experiments/ext_a/ext_a_lambda$L/checkpoints/checkpoint_best.pth
done

echo "=== $(date) EXT A RIGOR RE-EVAL COMPLETE ===" | tee -a "$LOG"
