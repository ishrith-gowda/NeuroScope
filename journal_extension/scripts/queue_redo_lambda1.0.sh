#!/bin/bash
# lambda1.0 training OOM'd at epoch 0 (no checkpoint) due to ram contention with
# the concurrent downstream run. the sweep falsely logged it DONE. re-run it
# after the sweep finishes (gpu/ram free), then eval its harmonization metrics.
cd ~/neuroscope/code
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

echo "=== $(date) waiting for sweep to complete before lambda1.0 redo ==="
while ! grep -q "ALL EXT A RUNS COMPLETE" ~/neuroscope/logs/sweep_master.log 2>/dev/null; do sleep 180; done

echo "=== $(date) re-running lambda1.0 fine-tune ==="
python journal_extension/scripts/finetune_ext_a_from_base.py \
  --base_checkpoint ~/neuroscope/checkpoints/base_sa_cyclegan_best.pth \
  --brats_dir ~/neuroscope/preprocessed/brats --upenn_dir ~/neuroscope/preprocessed/upenn \
  --output_dir ~/neuroscope/experiments/ext_a --experiment_name ext_a_lambda1.0 \
  --lambda_nce 1.0 --epochs 30 --num_workers 16 \
  > ~/neuroscope/logs/ext_a_lambda1.0_redo.log 2>&1

echo "=== $(date) lambda1.0 redo done; evaluating harmonization metrics ==="
python journal_extension/scripts/eval_harmonization_correct.py \
  --checkpoint ~/neuroscope/experiments/ext_a/ext_a_lambda1.0/checkpoints/checkpoint_best.pth \
  --label lambda1.0 \
  --brats_dir ~/neuroscope/preprocessed/brats --upenn_dir ~/neuroscope/preprocessed/upenn \
  --domain_clf ~/neuroscope/checkpoints/domain_classifier.pth \
  --output_dir ~/neuroscope/experiments/ext_a_eval >> ~/neuroscope/logs/eval_arms.log 2>&1

echo "=== $(date) lambda1.0 redo + eval COMPLETE ==="
