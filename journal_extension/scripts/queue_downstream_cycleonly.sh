#!/bin/bash
# run the corrected downstream eval for the cycle-only arm (lambda0.0) -- the
# key comparator for the mechanism study (hybrid vs cycle-only). gated so it
# starts only after lambda0.0 finishes training AND the lambda0.5 downstream
# completes, to avoid concurrent gpu/93GB-cache pile-up.
DS05=~/neuroscope/experiments/ext_a_downstream/lambda0.5/downstream_corrected.json
echo "=== $(date) waiting for lambda0.0 training to finish ==="
while ! grep -q "DONE ext_a_lambda0.0" ~/neuroscope/logs/sweep_master.log 2>/dev/null; do sleep 120; done
echo "=== $(date) waiting for lambda0.5 downstream to finish ==="
while [ ! -f "$DS05" ]; do sleep 120; done
echo "=== $(date) starting lambda0.0 (cycle-only) downstream ==="
bash ~/neuroscope/code/journal_extension/scripts/downstream_run.sh \
  ~/neuroscope/experiments/ext_a/ext_a_lambda0.0/checkpoints/checkpoint_best.pth \
  ~/neuroscope/experiments/ext_a_downstream/lambda0.0
echo "=== $(date) lambda0.0 downstream complete ==="
