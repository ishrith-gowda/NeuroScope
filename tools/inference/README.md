# inference pipeline for qualitative figure generation

comprehensive scripts for running inference on trained sa-cyclegan-2.5d model
and generating publication-grade qualitative visualizations.

## overview

this pipeline consists of three main steps:

1. **case selection**: identify representative test samples (best/worst/median/interesting cases)
2. **inference**: run model on selected cases to generate translations and reconstructions
3. **attention extraction**: capture attention weights from cbam and self-attention modules
4. **figure generation**: create publication-grade comparison grids and visualizations

## requirements

```bash
# install dependencies (from project root)
pip install torch torchvision numpy scipy matplotlib seaborn
```

## usage workflow

### step 1: select representative cases

analyzes evaluation results to identify interesting test cases for visualization.

```bash
cd /path/to/neuroscope

python tools/inference/select_cases.py \
    --eval_results results/evaluation/evaluation_results.json \
    --output tools/inference/case_ids.json \
    --n_best 5 \
    --n_worst 5 \
    --n_median 3 \
    --n_interesting 5 \
    --n_random 10 \
    --seed 42
```

**output**: `case_ids.json` containing:
- `best`: top 5 cases by ssim (a→b)
- `worst`: bottom 5 cases by ssim
- `median`: 3 cases around median performance
- `interesting`: 5 cases with metric disagreement (high ssim but low psnr, etc.)
- `random`: 10 random representative cases

**note**: this script uses gaussian approximation to simulate per-sample metrics
from aggregate statistics. for precise case selection, run full evaluation with
per-sample metric logging enabled.

### step 2: run inference

loads trained model and generates translations for selected cases.

```bash
python tools/inference/run_inference.py \
    --cases tools/inference/case_ids.json \
    --checkpoint experiments/sa_cyclegan_25d_rtx6000_resume_20260108_002543/checkpoints/checkpoint_best.pth \
    --output_dir results/inference \
    --device cuda \
    --categories best worst median interesting random
```

**parameters**:
- `--cases`: json file from step 1
- `--checkpoint`: path to trained model checkpoint
- `--output_dir`: where to save inference results
- `--device`: cuda or cpu
- `--categories`: which case categories to process

**output**: `results/inference/inference_{category}.npz` containing:
- `inputs_a`: input images from domain a [N, 12, H, W]
- `inputs_b`: input images from domain b [N, 12, H, W]
- `generated_b`: a→b translations [N, 4, H, W]
- `generated_a`: b→a translations [N, 4, H, W]
- `reconstructed_a`: a→b→a cycle reconstructions [N, 4, H, W]
- `reconstructed_b`: b→a→b cycle reconstructions [N, 4, H, W]
- `case_indices`: original test set indices
- `ssim_a2b`: ssim scores for each case
- `psnr_a2b`: psnr scores for each case

**note**: 2.5d format means inputs are [12 channels] = 3 adjacent slices × 4 modalities,
outputs are [4 channels] = center slice, 4 modalities.

### step 3: extract attention maps

captures attention weights during inference using pytorch forward hooks.

```bash
python tools/inference/extract_attention.py \
    --cases tools/inference/case_ids.json \
    --checkpoint experiments/sa_cyclegan_25d_rtx6000_resume_20260108_002543/checkpoints/checkpoint_best.pth \
    --output_dir results/inference \
    --device cuda \
    --categories best worst median
```

**parameters**: same as step 2, but typically run on subset of categories to save time/space.

**output**: `results/inference/attention_{category}.npz` containing:
- spatial attention maps from cbam modules [N, 1, H, W]
- channel attention weights from cbam modules [N, C, 1, 1]
- self-attention weights (if present) [N, heads, HW, HW]

**note**: this script automatically detects attention modules in the model.
if no attention modules found (e.g., baseline cyclegan), it will skip gracefully.

### step 4: generate qualitative figures

creates publication-grade visualizations from inference results.

```bash
python tools/inference/generate_qualitative_figures.py \
    --inference_dir results/inference \
    --output_dir figures/qualitative \
    --modality 1
```

**parameters**:
- `--inference_dir`: directory containing npz files from steps 2-3
- `--output_dir`: where to save generated figures
- `--modality`: which modality to visualize (0=t1, 1=t1ce, 2=t2, 3=flair)

**output figures**:
- `fig_best_comparison.pdf/png`: 5-row grid showing input | generated | reconstructed
- `fig_best_multimodality.pdf/png`: all 4 modalities for best case
- `fig_best_cycle_consistency.pdf/png`: bidirectional cycle demonstration with difference maps
- `fig_best_attention_overlay.pdf/png`: attention heatmap overlay on input
- `fig_worst_comparison.pdf/png`: worst case comparisons
- `fig_worst_multimodality.pdf/png`: worst case all modalities
- `fig_median_comparison.pdf/png`: median case comparisons

all figures use latex rendering, 300 dpi, colorblind-friendly palettes,
and ieee two-column format specifications.

## complete workflow example

```bash
# from project root
cd /Volumes/usb\ drive/neuroscope

# step 1: select cases (fast, ~5 seconds)
python tools/inference/select_cases.py \
    --output tools/inference/case_ids.json

# step 2: run inference (moderate, ~1-2 min on gpu for 28 cases)
python tools/inference/run_inference.py \
    --cases tools/inference/case_ids.json \
    --device cuda \
    --categories best worst median interesting

# step 3: extract attention (moderate, ~30 sec on gpu for subset)
python tools/inference/extract_attention.py \
    --cases tools/inference/case_ids.json \
    --device cuda \
    --categories best median

# step 4: generate figures (fast, ~10 seconds)
python tools/inference/generate_qualitative_figures.py \
    --inference_dir results/inference \
    --output_dir figures/qualitative \
    --modality 1  # t1ce

# repeat step 4 for other modalities if desired
for mod in 0 2 3; do
    python tools/inference/generate_qualitative_figures.py \
        --modality $mod
done
```

## output structure

```
results/inference/
├── inference_best.npz
├── inference_worst.npz
├── inference_median.npz
├── inference_interesting.npz
├── inference_random.npz
├── attention_best.npz
├── attention_worst.npz
└── attention_median.npz

figures/qualitative/
├── fig_best_comparison.pdf
├── fig_best_comparison.png
├── fig_best_multimodality.pdf
├── fig_best_multimodality.png
├── fig_best_cycle_consistency.pdf
├── fig_best_cycle_consistency.png
├── fig_best_attention_overlay.pdf
├── fig_best_attention_overlay.png
├── fig_worst_comparison.pdf
├── fig_worst_comparison.png
├── fig_worst_multimodality.pdf
├── fig_worst_multimodality.png
├── fig_median_comparison.pdf
└── fig_median_comparison.png
```

## customization

### selecting different cases

modify case selection parameters:

```python
python tools/inference/select_cases.py \
    --n_best 10 \          # top 10 instead of 5
    --n_worst 10 \
    --n_median 5 \
    --n_interesting 8 \
    --seed 123             # different random seed
```

### custom figure layouts

edit `generate_qualitative_figures.py` to create custom visualizations:

```python
# example: create 4x4 grid instead of 5x3
generate_comparison_grid(
    inputs_a, generated_b, reconstructed_a,
    ssim_scores, output_path,
    n_samples=4,  # show 4 cases
    modality_idx=1
)
```

### attention visualization options

the attention extraction script automatically detects:
- cbam channel attention
- cbam spatial attention
- self-attention (multi-head)

to visualize specific attention types, filter the attention keys:

```python
# in generate_qualitative_figures.py
channel_attention_keys = [k for k in attention_data.keys()
                         if 'channel' in k.lower()]
spatial_attention_keys = [k for k in attention_data.keys()
                         if 'spatial' in k.lower()]
```

## troubleshooting

### issue: "no attention modules found"

**cause**: model is baseline cyclegan without attention mechanisms.

**solution**: skip step 3 (attention extraction) or train model with attention.

### issue: "checkpoint format not recognized"

**cause**: checkpoint structure doesn't match expected format.

**solution**: check if checkpoint contains `generator_A2B_state_dict` or `model_state_dict`.
modify `load_model()` function in run_inference.py if needed.

### issue: "test data directory not found"

**cause**: step 1 can't find processed test data to get sample indices.

**solution**: this is a warning, not an error. script will use synthetic indices
based on `test_samples` count from evaluation results. inference will still work
if you use the correct data loader in step 2.

### issue: out of memory during inference

**solution**: reduce batch size or process fewer categories at once:

```bash
# process one category at a time
python tools/inference/run_inference.py --categories best
python tools/inference/run_inference.py --categories worst
python tools/inference/run_inference.py --categories median
```

or use cpu:

```bash
python tools/inference/run_inference.py --device cpu
```

## publication usage

these scripts generate figures suitable for:
- main paper: best case comparisons, attention overlays
- supplementary material: worst cases, additional modalities, full grids

figure standards:
- resolution: 300 dpi
- format: pdf (vector) + png (raster)
- typography: latex computer modern fonts
- color scheme: colorblind-friendly
- layout: ieee two-column compatible

## advanced: per-sample evaluation

for precise case selection (instead of gaussian approximation), run evaluation
with per-sample metric logging:

```python
# in evaluation script
results = {
    'per_sample_metrics': {
        'ssim_A2B': ssim_list,  # list of per-sample values
        'psnr_A2B': psnr_list,
        # ... other metrics
    }
}
```

then modify `select_cases.py` to load actual per-sample values instead of
simulating distributions.

## contact

for questions or issues with the inference pipeline, check:
1. main project documentation: `/Volumes/usb drive/neuroscope/README.md`
2. figure generation guide: `/Volumes/usb drive/neuroscope/PUBLICATION_FIGURES_COMPLETE.md`
3. next steps guide: `/Volumes/usb drive/neuroscope/NEXT_STEPS_FOR_USER.md`
