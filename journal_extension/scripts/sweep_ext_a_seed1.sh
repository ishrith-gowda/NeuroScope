#!/bin/bash
# ext a multi-seed (gold-standard variance): re-run all 5 lambda arms with a 2nd
# seed (seed=1) at the IDENTICAL matched budget (30 ep warm-start), then eval each.
# combined with the original seed-42 arms this yields a 2-seed estimate / error
# bars on the sweep and resolves whether lambda=0.5 is THE fid optimum or part of
# a flat 0.5-2.0 good region (the optimum is a ~1-fid margin, marginal on kid).
# runs strictly sequentially (one 93gb cache at a time -> ram-safe, no oom).
set -u
cd ~/neuroscope/code
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

BASE=~/neuroscope/checkpoints/base_sa_cyclegan_best.pth
BR=~/neuroscope/preprocessed/brats
UP=~/neuroscope/preprocessed/upenn
OUT=~/neuroscope/experiments/ext_a
EVALOUT=~/neuroscope/experiments/ext_a_eval
CLF=~/neuroscope/checkpoints/domain_classifier.pth
LOG=~/neuroscope/logs
SEED=1
EPOCHS=30

for LAM in 0.5 0.0 1.0 2.0 0.1; do
  NAME=ext_a_lambda${LAM}_seed${SEED}
  echo "=== $(date) START $NAME (epochs=$EPOCHS seed=$SEED) ==="
  python journal_extension/scripts/finetune_ext_a_from_base.py \
    --base_checkpoint "$BASE" --brats_dir "$BR" --upenn_dir "$UP" \
    --output_dir "$OUT" --experiment_name "$NAME" \
    --lambda_nce "$LAM" --epochs "$EPOCHS" --seed "$SEED" --num_workers 16 \
    > "$LOG/${NAME}.log" 2>&1
  echo "=== $(date) DONE TRAIN $NAME (exit $?) ==="
  python journal_extension/scripts/eval_harmonization_correct.py \
    --checkpoint "$OUT/$NAME/checkpoints/checkpoint_best.pth" \
    --label "lambda${LAM}_seed${SEED}" \
    --brats_dir "$BR" --upenn_dir "$UP" --domain_clf "$CLF" \
    --output_dir "$EVALOUT" >> "$LOG/ext_a_seed${SEED}_eval.log" 2>&1
  echo "=== $(date) DONE EVAL $NAME ==="
done
echo "=== ALL EXT A SEED${SEED} RUNS COMPLETE $(date) ==="
