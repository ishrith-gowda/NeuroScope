#!/bin/bash
# end-to-end corrected downstream eval for one harmonization checkpoint.
# args: $1 = checkpoint path, $2 = output dir
# 1) harmonize upenn->brats (G_B2A) and brats->upenn (G_A2B) with the correct
#    training-consistent pipeline, 2) train segmenters on raw and evaluate
#    cross-site raw vs harmonized (dice/hd95).
#
# num_workers is kept LOW (2): BraTSSliceDataset caches volumes per worker, so
# many workers x the upenn cache (8GB) OOMs when the sweep's 93GB cache is also
# resident. harmonization is skipped if already done (idempotent re-runs).
set -e
cd ~/neuroscope/code
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

CKPT="$1"
OUT="$2"
BR=~/neuroscope/preprocessed/brats
UP=~/neuroscope/preprocessed/upenn
mkdir -p "$OUT"

harm_if_needed() {  # $1=site_dir  $2=direction  $3=out_subdir
  local nd
  nd=$(ls "$OUT/$3" 2>/dev/null | wc -l)
  if [ "$nd" -lt 60 ]; then
    echo "=== $(date) harmonize $2 -> $3 ==="
    python journal_extension/scripts/harmonize_for_downstream.py \
      --checkpoint "$CKPT" --site_dir "$1" --direction "$2" --output_dir "$OUT/$3"
  else
    echo "=== $3 already harmonized ($nd subjects), skip ==="
  fi
}

harm_if_needed "$UP" B2A harm_upenn_to_brats
harm_if_needed "$BR" A2B harm_brats_to_upenn

echo "=== $(date) cross-site downstream segmentation eval ==="
python journal_extension/scripts/eval_downstream_corrected.py \
  --brats_dir "$BR" --upenn_dir "$UP" \
  --harm_upenn_to_brats "$OUT/harm_upenn_to_brats" \
  --harm_brats_to_upenn "$OUT/harm_brats_to_upenn" \
  --output_dir "$OUT" --seg_epochs 40 --num_workers 2

echo "=== $(date) DOWNSTREAM COMPLETE ==="
