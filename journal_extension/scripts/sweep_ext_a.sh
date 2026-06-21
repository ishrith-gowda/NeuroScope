#!/bin/bash
# sequential lambda-sweep fine-tune for ext a (sashimi mechanism study).
# each arm warm-starts from the base sa-cyclegan checkpoint (matched budget).
# run order puts the key models first: hybrid (0.5), cycle-only control (0.0),
# then the cut-default (1.0) and the rest of the sweep.
set -u
cd ~/neuroscope/code
source .venv/bin/activate

BASE=~/neuroscope/checkpoints/base_sa_cyclegan_best.pth
BR=~/neuroscope/preprocessed/brats
UP=~/neuroscope/preprocessed/upenn
OUT=~/neuroscope/experiments/ext_a
LOG=~/neuroscope/logs
mkdir -p "$LOG" "$OUT"
EPOCHS=30

for LAM in 0.5 0.0 1.0 2.0 0.1; do
  NAME=ext_a_lambda${LAM}
  echo "=== $(date) START $NAME (epochs=$EPOCHS) ==="
  python journal_extension/scripts/finetune_ext_a_from_base.py \
    --base_checkpoint "$BASE" --brats_dir "$BR" --upenn_dir "$UP" \
    --output_dir "$OUT" --experiment_name "$NAME" \
    --lambda_nce "$LAM" --epochs "$EPOCHS" --num_workers 16 \
    > "$LOG/${NAME}.log" 2>&1
  echo "=== $(date) DONE $NAME (exit $?) ==="
done
echo "=== ALL EXT A RUNS COMPLETE $(date) ==="
