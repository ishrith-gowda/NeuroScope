# NeuroScope

Domain-aware standardization of multimodal glioma MRI; CycleGAN‑based framework for standardizing multi‑institutional glioblastoma MRI scans (T1, T1ce, T2, FLAIR) across different scanner protocols.

## Table of Contents

1. Overview
2. Features
3. Directory structure
4. Requirements
5. Installation
6. Data preparation
7. Preprocessing pipeline
8. Training
9. Evaluation & visualization
10. Usage examples
11. Contributing
12. License

## Overview

NeuroScope tackles scanner‑protocol heterogeneity in glioblastoma MRI by learning an unsupervised image‑to‑image translation between BraTS (TCGA‑GBM) and UPenn‑GBM datasets. The CycleGAN operates on four‑channel 2D axial slices (T1, T1ce, T2, FLAIR) to produce harmonized volumes for downstream radiomic analysis.

## Features

- Domain splitting into BraTS (domain A) and UPenn (domain B)
- Preprocessing: skull‑stripping, percentile intensity normalization, 1 mm isotropic resampling
- CycleGAN with ResNet‑based generators (nine residual blocks) and 70×70 PatchGAN discriminators
- Losses: adversarial (MSE), cycle‑consistency (L1), identity (L1)
- TensorBoard logging: loss curves, weight histograms, sample grids
- Visualization: sample translation montages; SSIM/PSNR histograms; model summary and parameter counts

## Requirements

- Python 3.9 or later
- PyTorch 1.11 or later (CUDA or MPS support)
- torchvision, SimpleITK, matplotlib, numpy, pandas
- TensorBoard
- ANTs or HD‑BET (for optional skull‑stripping)
- NBIA Data Retriever or Aspera Connect (for dataset download)

## Installation

1. Clone the repo: git clone [https://github.com/ishrith-gowda/NeuroScope.git](https://github.com/ishrith-gowda/NeuroScope.git) and cd NeuroScope
2. Create and activate a virtual environment: python -m venv venv; source venv/bin/activate
3. Install dependencies: pip install -r requirements.txt
4. (Optional) Install HD‑BET: pip install hd-bet
5. Place the MNI152 template at \~/Templates/MNI152_T1_1mm.nii.gz

## Data preparation

1. Download BraTS (TCGA‑GBM) and UPenn‑GBM data via TCIA.
2. Organize under data/: data/BraTS-TCGA-GBM and data/UPENN-GBM.

## Preprocessing Pipeline (Refactored & Orchestrated)

All preprocessing scripts now expose consistent CLI arguments (`--splits`, `--output`, plus task‑specific flags). A unified orchestrator chains the major stages with dependency checks and idempotent skipping.

Primary scripts (in execution order):

1. `01_fast_intensity_normalization_neuroscope.py` – Fast percentile normalization & optional resampling
2. `05_comprehensive_intensity_bias_assessment_neuroscope.py` – Pre‑N4 slice bias profiling
3. `06_n4_bias_correction_neuroscope.py` – Conservative + adaptive N4 bias correction with retry logic
4. `07_assess_n4_correction_effectiveness_neuroscope.py` – Before/after CV & residual bias evaluation
5. `08_diagnose_n4_issues_neuroscope.py` – Targeted diagnostics on sampled subjects
6. `09_verify_preprocessing_completeness_neuroscope.py` – Final readiness & quality audit

Each stage writes both a full JSON and a lightweight `_summary.json` (median / key metrics). Schema validation and minimal summaries accelerate large‑scale inspection.

### One‑Command Orchestrator

Run the full pipeline (default `train,val` splits):

```
python scripts/01_data_preparation_pipeline/run_preprocessing_pipeline.py --splits=train,val --verbose
```

Key flags:

- `--force` re-runs stages even if their outputs exist
- `--dry-run` prints planned actions only
- `--stop-on-fail` aborts at first failing stage

### Individual Script Usage Examples

Fast normalization (train + val):

```
python scripts/01_data_preparation_pipeline/01_fast_intensity_normalization_neuroscope.py \
	--splits=train,val --lower-pct 0.5 --upper-pct 99.5
```

N4 bias correction with custom threshold & workers:

```
python scripts/01_data_preparation_pipeline/06_n4_bias_correction_neuroscope.py \
	--splits=train,val --bias-threshold 0.16 --max-workers 6
```

Assess N4 effectiveness on validation only:

```
python scripts/01_data_preparation_pipeline/07_assess_n4_correction_effectiveness_neuroscope.py --splits=val
```

Diagnostics (sample 8 subjects per section):

```
python scripts/01_data_preparation_pipeline/08_diagnose_n4_issues_neuroscope.py --max-per-section 8
```

Final verification including test split (if prepared):

```
python scripts/01_data_preparation_pipeline/09_verify_preprocessing_completeness_neuroscope.py --splits=train,val,test
```

### Output Conventions

- Core outputs reside under `preprocessed/` with dataset + (optional) `_n4corrected_v2` suffixes.
- Assessment JSONs: `slice_bias_assessment.json`, `n4_correction_results_improved_v2.json`, `n4_effectiveness_assessment.json`, `n4_diagnostic_analysis.json`, `neuroscope_pipeline_verification_results.json` plus `_summary.json` companions.
- A pipeline run summary: `preprocessed/pipeline_run_summary.json`.

### Quality Safeguards

- Unified brain mask generation across scripts for metric consistency.
- Conservative N4 pass with adaptive retry if first result fails sanity checks.
- Schema‑aware JSON writer: guards against truncated or structurally invalid result files.
- Lightweight summaries enable rapid monitoring in remote / low‑bandwidth settings.

### Reproducibility

All randomness (subject sampling in diagnostics) uses fixed seeds. Re-run determinism is preserved unless inputs change or `--force` triggers regeneration.

## Training

Run:
python 02_model_development_pipeline/train_cyclegan.py<br/>
\--data_root /path/to/data/preprocessed<br/>
\--meta_json /path/to/scripts/neuroscope_dataset_metadata_splits.json<br/>
\--n_epochs 100<br/>
\--batch_size 8<br/>
\--lr 2e-4<br/>
\--decay_epoch 50<br/>
\--checkpoint_interval 10<br/>
\--sample_interval 200<br/>
\--log_interval 50<br/>
Models are saved in checkpoints/, sample images in samples/, and TensorBoard logs in \~/Downloads/neuroscope/runs. Launch TensorBoard with tensorboard --logdir \~/Downloads/neuroscope/runs.

### Modular Training Pipeline

To mirror the preprocessing pipeline, CycleGAN training is split into small, composable scripts under `scripts/02_model_development_pipeline/`:

- `01_prepare_training_manifest.py`: Verifies preprocessed subjects per split and writes `neuroscope_training_manifest.json` + `_summary.json`.
- `02_dataloader_smoke_test.py`: Quick loader check to confirm balanced, valid batches for domains A/B.
- `03_train_cyclegan_entry.py`: Thin wrapper around `train_cyclegan.py` using USB-aware defaults from `neuroscope_preprocessing_config.py`.
- `04_evaluate_cyclegan.py`: Computes SSIM/PSNR on a small validation subset and saves qualitative grids.
- `05_export_inference_package.py`: Exports a self-contained generator bundle `full_G_A2B.pt` with a tiny inference stub.
- `run_training_pipeline.py`: Orchestrates the above with idempotent behavior and sensible defaults.

One-command run (after preprocessing):

```
python scripts/02_model_development_pipeline/run_training_pipeline.py --verbose
```

You can also run individual steps, e.g.:

```
python scripts/02_model_development_pipeline/01_prepare_training_manifest.py
python scripts/02_model_development_pipeline/02_dataloader_smoke_test.py \
	--preprocessed_dir "/Volumes/usb drive/neuroscope/preprocessed" \
	--metadata_json "/Volumes/usb drive/neuroscope/scripts/01_data_preparation_pipeline/neuroscope_dataset_metadata_splits.json"
```

## Evaluation & visualization

After training, run:
python 02_model_development_pipeline/visualize_cyclegan_training.py
Generates loss curves, SSIM/PSNR histograms, translation grids, and model summaries in figures/.

## Usage examples

Inference on new scans:

- Load a full model: model = torch.load('checkpoints/full_G_A2B_100.pt')\['architecture']
- model.eval(); output = model(input_tensor) for each slice

Contributing

1. Fork the repo and create a feature branch
2. Commit and push changes
3. Open a pull request with tests and PEP8 compliance
