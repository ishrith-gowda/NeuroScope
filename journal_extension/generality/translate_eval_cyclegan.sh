#!/usr/bin/env bash
# translate GTA5 -> Cityscapes with the trained CycleGAN (G_A), then score the translations with the
# frozen SegFormer (mIoU) + clean-fid (FID) -- the LEARNED arm of the generality benchmark.
set -euo pipefail
CG="$HOME/cyclegan"
G="$HOME/data/generality"
PY="$HOME/neuroscope/.venv/bin/python"
export WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 PATH="$HOME/.local/bin:$PATH"

cd "$CG"
# cycle_gan test needs both testA (GTA5) and testB (Cityscapes) present
mkdir -p datasets/gta2city/testB
cp "$G"/cityscapes/images/*.png datasets/gta2city/testB/ 2>/dev/null || true

echo "== translating GTA5 (testA) -> Cityscapes style (fake_B) =="
"$PY" test.py --dataroot ./datasets/gta2city --name gta2city_cyclegan --model cycle_gan \
  --no_dropout --num_test 1000 --phase test >/dev/null 2>&1 || \
  "$PY" test.py --dataroot ./datasets/gta2city --name gta2city_cyclegan --model cycle_gan \
  --no_dropout --num_test 1000 --phase test

echo "== extracting fake_B translations =="
mkdir -p "$G/cyclegan/images"
n=0
for f in "$CG"/results/gta2city_cyclegan/test_latest/images/*_fake_B.png; do
  [ -e "$f" ] || continue
  base="$(basename "$f")"
  cp "$f" "$G/cyclegan/images/${base%_fake_B.png}.png"
  n=$((n + 1))
done
echo "extracted $n translated images"

echo "== eval (frozen SegFormer mIoU + FID) =="
"$PY" "$HOME/neuroscope/journal_extension/generality/eval_generality.py" \
  --images "$G/cyclegan/images" --masks "$G/gta5/masks" \
  --cityscapes "$G/cityscapes/images" --out "$G/results/cyclegan.json" --tag cyclegan
