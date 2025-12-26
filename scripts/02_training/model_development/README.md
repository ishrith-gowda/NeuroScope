# NeuroScope CycleGAN Training Pipeline

This directory contains a comprehensive CycleGAN training pipeline for domain adaptation between BraTS and UPenn MRI datasets.

## Overview

The pipeline implements CycleGAN for unsupervised domain adaptation:
- **Domain A**: BraTS-TCGA-GBM (preprocessed to [0,1] → mapped to [-1,1])
- **Domain B**: UPenn-GBM (preprocessed to [0,1] → mapped to [-1,1])

## Pipeline Components

### 1. Data Preparation (`01_prepare_training_manifest.py`)
- Creates training manifest from metadata
- Validates preprocessed data availability
- Generates subject lists for each split

### 2. Comprehensive Validation (`06_comprehensive_pipeline_validation.py`)
- **NEW**: Validates entire preprocessing → training pipeline
- Checks tensor normalization ([0,1] → [-1,1])
- Verifies domain mapping (A=brats, B=upenn)
- Validates data consistency between preprocessing and training
- Generates detailed validation report

### 3. DataLoader Smoke Test (`02_dataloader_smoke_test.py`)
- Quick test of dataset loaders
- Validates tensor ranges and shapes
- Ensures both domains are accessible

### 4. Training Entry Point (`03_train_cyclegan_entry.py`)
- Wrapper for main training script
- Uses optimized training parameters
- Includes gradient clipping and AdamW optimizer

### 5. Main Training Script (`train_cyclegan.py`)
- **IMPROVED**: Enhanced with performance optimizations
- Gradient clipping for training stability
- AdamW optimizer with weight decay
- Comprehensive tensor range validation
- Better error handling and logging

### 6. Evaluation (`04_evaluate_cyclegan.py`)
- Evaluates trained models with SSIM/PSNR metrics
- Generates sample visualizations
- Creates evaluation reports

### 7. Export (`05_export_inference_package.py`)
- Exports trained models for inference
- Includes usage documentation

## Key Improvements Made

### Tensor Normalization Flow
1. **Preprocessing**: Normalizes brain tissue to [0,1], background at 0
2. **DataLoader**: Clamps to [0,1] (safety check) → maps to [-1,1] for CycleGAN
3. **Training**: Expects [-1,1] range, validates tensor ranges

### Error Handling & Validation
- Comprehensive input validation in preprocessing
- Fallback normalization strategies
- Non-finite value detection and cleaning
- Tensor range validation throughout pipeline
- Detailed error logging and reporting

### Performance Optimizations
- AdamW optimizer with weight decay (1e-4)
- Gradient clipping (max_norm=5.0) for training stability
- Better memory management
- Optimized data loading

### Domain Mapping
- **Consistent throughout**: A=brats, B=upenn
- Validated in dataset loader and training scripts
- Clear documentation and logging

## Usage

### Quick Start
```bash
# Run complete pipeline with validation
python run_training_pipeline.py

# Skip validation (faster)
python run_training_pipeline.py --skip-validation

# Force re-run all stages
python run_training_pipeline.py --force
```

### Individual Components
```bash
# Validate pipeline
python 06_comprehensive_pipeline_validation.py --verbose

# Test data loaders
python 02_dataloader_smoke_test.py \
    --preprocessed_dir /path/to/preprocessed \
    --metadata_json /path/to/metadata.json

# Train model
python 03_train_cyclegan_entry.py \
    --n_epochs 100 \
    --batch_size 8 \
    --lr 2e-4

# Evaluate model
python 04_evaluate_cyclegan.py \
    --generator_ckpt /path/to/checkpoint.pth \
    --data_root /path/to/preprocessed \
    --meta_json /path/to/metadata.json \
    --output_dir /path/to/output
```

## Validation Report

The comprehensive validation script generates a detailed report (`pipeline_validation_report.json`) that includes:

- **Preprocessing Validation**: Checks [0,1] normalization
- **DataLoader Validation**: Verifies [-1,1] mapping and domain assignment
- **Data Consistency**: Ensures preprocessing → training consistency
- **File Structure**: Validates all required files and directories

## Training Parameters

### Default Configuration
- **Epochs**: 100
- **Batch Size**: 8
- **Learning Rate**: 2e-4
- **Optimizer**: AdamW (β1=0.5, β2=0.999, weight_decay=1e-4)
- **Gradient Clipping**: 5.0
- **Loss Weights**: λ_cycle=10.0, λ_identity=5.0

### Model Architecture
- **Generator**: ResNet with 9 residual blocks, InstanceNorm
- **Discriminator**: PatchGAN with InstanceNorm
- **Input/Output**: 4-channel MRI modalities (T1, T1GD, T2, FLAIR)

## Output Structure

```
checkpoints/
├── G_A2B_10.pth          # Generator A→B checkpoints
├── G_B2A_10.pth          # Generator B→A checkpoints
├── D_A_10.pth            # Discriminator A checkpoints
├── D_B_10.pth            # Discriminator B checkpoints
└── full_models_final.pt  # Complete model checkpoint

samples/
├── debug_domain_A.png    # Domain A debug samples
├── debug_domain_B.png    # Domain B debug samples
├── sample_500.png        # Training samples
└── training_loss_log.json # Loss history

figures/
├── evaluation_report.json # Evaluation metrics
└── samples_val.png       # Validation samples
```

## Troubleshooting

### Common Issues

1. **Tensor Range Errors**
   - Check preprocessing outputs are in [0,1]
   - Verify data loader transformation
   - Run comprehensive validation

2. **Domain Mapping Issues**
   - Ensure A=brats, B=upenn consistently
   - Check metadata file structure
   - Validate dataset loader configuration

3. **Memory Issues**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

4. **Training Instability**
   - Check gradient clipping is enabled
   - Verify learning rate schedule
   - Monitor loss curves

### Validation Commands
```bash
# Full validation
python 06_comprehensive_pipeline_validation.py --verbose

# Quick smoke test
python 02_dataloader_smoke_test.py \
    --preprocessed_dir /path/to/preprocessed \
    --metadata_json /path/to/metadata.json \
    --verbose
```

## Dependencies

- PyTorch >= 1.9.0
- SimpleITK
- NumPy
- Matplotlib
- Seaborn
- scikit-image
- torchvision
- tensorboard (optional)

## Notes

- The pipeline assumes preprocessed data is available from the preprocessing pipeline
- All paths are configured via `neuroscope_preprocessing_config.py`
- Training is optimized for MRI domain adaptation
- Comprehensive logging and validation ensure robust training