#!/bin/bash
# end-to-end corrected downstream eval for one harmonization checkpoint.
# args: $1 = checkpoint path, $2 = output dir
# 1) harmonize upenn->brats (G_B2A) and brats->upenn (G_A2B) with the correct
#    training-consistent pipeline, 2) train segmenters on raw and evaluate
#    cross-site raw vs harmonized (dice/hd95).
set -e
cd ~/neuroscope/code
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

CKPT="$1"
OUT="$2"
BR=~/neuroscope/preprocessed/brats
UP=~/neuroscope/preprocessed/upenn
mkdir -p "$OUT"

echo "=== $(date) harmonize upenn -> brats (B2A) ==="
python journal_extension/scripts/harmonize_for_downstream.py \
  --checkpoint "$CKPT" --site_dir "$UP" --direction B2A \
  --output_dir "$OUT/harm_upenn_to_brats"

echo "=== $(date) harmonize brats -> upenn (A2B) ==="
python journal_extension/scripts/harmonize_for_downstream.py \
  --checkpoint "$CKPT" --site_dir "$BR" --direction A2B \
  --output_dir "$OUT/harm_brats_to_upenn"

echo "=== $(date) cross-site downstream segmentation eval ==="
python journal_extension/scripts/eval_downstream_corrected.py \
  --brats_dir "$BR" --upenn_dir "$UP" \
  --harm_upenn_to_brats "$OUT/harm_upenn_to_brats" \
  --harm_brats_to_upenn "$OUT/harm_brats_to_upenn" \
  --output_dir "$OUT" --seg_epochs 40

echo "=== $(date) DOWNSTREAM COMPLETE ==="
