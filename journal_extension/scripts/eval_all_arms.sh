#!/bin/bash
# hands-off mechanism-study data collection: eval each remaining sweep arm with
# the corrected harmonization metrics as it finishes. waits for the downstream
# (lambda0.5) to finish first to avoid concurrent 93GB-cache memory pile-up.
# lambda0.5 is already evaluated (skipped).
cd ~/neuroscope/code
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

BR=~/neuroscope/preprocessed/brats
UP=~/neuroscope/preprocessed/upenn
DC=~/neuroscope/checkpoints/domain_classifier.pth
OUT=~/neuroscope/experiments/ext_a_eval
DS=~/neuroscope/experiments/ext_a_downstream/lambda0.5/downstream_corrected.json
mkdir -p "$OUT"

echo "=== $(date) waiting for downstream(lambda0.5) before arm-evals (max ~4h) ==="
for w in $(seq 1 120); do [ -f "$DS" ] && break; sleep 120; done
echo "=== $(date) starting arm evals ==="

for LAM in 0.0 1.0 2.0 0.1; do
  CK=~/neuroscope/experiments/ext_a/ext_a_lambda${LAM}/checkpoints/checkpoint_best.pth
  J="$OUT/eval_lambda${LAM}.json"
  [ -f "$J" ] && { echo "=== lambda${LAM} already evaluated, skip ==="; continue; }
  echo "=== $(date) waiting for lambda${LAM} training to finish ==="
  while ! grep -q "DONE ext_a_lambda${LAM}" ~/neuroscope/logs/sweep_master.log 2>/dev/null; do sleep 120; done
  echo "=== $(date) EVAL lambda${LAM} ==="
  python journal_extension/scripts/eval_harmonization_correct.py \
    --checkpoint "$CK" --label lambda${LAM} \
    --brats_dir "$BR" --upenn_dir "$UP" --domain_clf "$DC" \
    --output_dir "$OUT" >> ~/neuroscope/logs/eval_arms.log 2>&1
  echo "=== $(date) DONE lambda${LAM} (exit $?) ==="
done
echo "=== $(date) ALL ARM EVALS COMPLETE ==="
