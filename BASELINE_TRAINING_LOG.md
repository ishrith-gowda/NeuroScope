# Baseline CycleGAN 2.5D Training Log

**Purpose**: Train standard CycleGAN without attention mechanisms as comparison baseline
**Status**: Training in progress
**Start Time**: 2026-01-12 05:21:35 UTC
**Server**: Chameleon Cloud RTX 6000

---

## Model Architecture

**Baseline CycleGAN 2.5D (No Attention)**
- Total Parameters: 33.88M
- Generator A→B: 11.41M parameters
- Generator B→A: 11.41M parameters
- Discriminator A: 5.53M parameters
- Discriminator B: 5.53M parameters

**Key Difference from SA-CycleGAN**:
- NO self-attention modules
- NO CBAM attention mechanisms
- Standard residual blocks only
- Same 2.5D architecture (3-slice input, center slice output)

**Comparison to SA-CycleGAN**:
- SA-CycleGAN: 35.1M parameters
- Baseline: 33.88M parameters
- Difference: 1.22M parameters removed (attention mechanisms)

---

## Training Configuration

**Identical to SA-CycleGAN for fair comparison:**

```yaml
epochs: 100
batch_size: 8
image_size: 128
learning_rate: 0.0002
optimizer: Adam (beta1=0.5, beta2=0.999)

# Loss weights
lambda_cycle: 10.0
lambda_identity: 5.0
lambda_ssim: 1.0
lambda_gradient: 1.0

# Architecture
ngf: 64
ndf: 64
n_residual_blocks: 9
use_attention: false  # KEY DIFFERENCE
use_cbam: false       # KEY DIFFERENCE
```

---

## Dataset

Same as SA-CycleGAN:
- **Domain A (BraTS)**: 8,184 samples
- **Domain B (UPenn-GBM)**: 52,638 samples
- **Training**: 42,110 samples
- **Validation**: 5,263 samples
- **Test**: 5,265 samples
- **Epoch length**: 52,638 (larger domain)
- **Batches per epoch**: 5,264

---

## Early Training Metrics (First 44 Iterations)

**Generator Loss Progression:**
- Iteration 1: 13.612
- Iteration 10: 6.296
- Iteration 20: 5.039
- Iteration 30: 4.398
- Iteration 40: 3.535
- Iteration 44: 2.908

**Discriminator Loss:**
- Started: 1.895
- Stabilized around: 0.5-0.7

**Cycle Consistency Loss:**
- Iteration 1: 6.029
- Iteration 10: 2.374
- Iteration 20: 1.707
- Iteration 30: 1.450
- Iteration 40: 1.148
- Iteration 44: 1.024

**Observations:**
- Healthy convergence
- Generator and discriminator both improving
- Cycle loss decreasing rapidly (expected for CycleGAN)
- No signs of mode collapse or instability

---

## Purpose of This Baseline

This baseline training is **critical for publication** because it proves:

1. **Attention mechanisms improve performance**: By comparing SA-CycleGAN (with attention) vs this baseline (no attention), we can quantify the contribution of self-attention and CBAM.

2. **Ablation study**: Shows that our architectural innovations (attention) are not just adding parameters, but genuinely improving translation quality.

3. **Fair comparison**: Same training procedure, same hyperparameters, same dataset - only difference is attention mechanisms.

4. **Publication requirement**: Top-tier venues (MICCAI, IEEE TMI) require baseline comparisons to validate novel contributions.

---

## Expected Outcomes

Based on literature and our SA-CycleGAN results:

**SA-CycleGAN-2.5D (with attention):**
- FID A→B: 61.18
- LPIPS A→B: 0.233
- Cycle SSIM: 0.923

**Expected Baseline (no attention):**
- FID A→B: 70-90 (worse distribution matching)
- LPIPS A→B: 0.25-0.30 (worse perceptual quality)
- Cycle SSIM: 0.85-0.90 (worse reconstruction)

**Hypothesis:**
Attention mechanisms improve translation quality by:
- Better capturing long-range dependencies (self-attention)
- Focusing on anatomically important regions (CBAM)
- Improving feature selection in residual blocks

---

## Next Steps

1. ✅ Training launched successfully
2. ⏳ Monitor training progress (100 epochs)
3. ⏳ Evaluate baseline with same metrics as SA-CycleGAN
4. ⏳ Perform statistical comparisons
5. ⏳ Generate comparative figures
6. ⏳ Write paper with quantitative ablation results

---

## Files

**Model**: `neuroscope/models/architectures/baseline_cyclegan_25d.py`
**Training Script**: `scripts/02_training/train_baseline_cyclegan_25d.py`
**Config**: `neuroscope/config/experiments/train_baseline_cyclegan_25d.yaml`
**Launch Script**: Server `/home/cc/neuroscope/launch_baseline_training.sh`
**Training Log**: Server `/home/cc/neuroscope/baseline_training.log`
**Checkpoints**: Server `/home/cc/neuroscope/experiments/baseline_cyclegan_25d_full/checkpoints/`

---

**Status**: Training epoch 1 in progress. Will run for approximately 100 epochs (estimated 3-7 days on RTX 6000).
