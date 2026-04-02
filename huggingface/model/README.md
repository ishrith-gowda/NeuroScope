---
license: other
license_name: cc-by-nc-nd-4.0
license_link: https://creativecommons.org/licenses/by-nc-nd/4.0/
pipeline_tag: image-to-image
tags:
  - medical-imaging
  - mri-harmonization
  - brain-mri
  - cyclegan
  - self-attention
  - domain-adaptation
  - glioma
  - neuroimaging
  - unpaired-image-translation
  - radiomics
  - 2.5d
  - federated-learning
  - neural-compression
  - pytorch
datasets:
  - ishrith-gowda/MRI-Harmonization-BraTS-UPenn
language:
  - en
model-index:
  - name: SA-CycleGAN-2.5D
    results:
      - task:
          type: image-to-image
          name: MRI Harmonization
        dataset:
          name: BraTS + UPenn-GBM
          type: ishrith-gowda/MRI-Harmonization-BraTS-UPenn
          config: two-site
          split: test
        metrics:
          - type: mmd
            value: 0.015
            name: Maximum Mean Discrepancy (post-harmonization)
          - type: mmd_reduction
            value: 99.1
            name: MMD Reduction (%)
          - type: domain_classifier_accuracy
            value: 59.7
            name: Domain Classifier Accuracy (%)
          - type: ssim
            value: 0.970
            name: SSIM (compression variant)
          - type: ssim_federated
            value: 0.998
            name: SSIM (federated variant, 40 rounds)
---

# SA-CycleGAN-2.5D: Self-Attention CycleGAN with Tri-Planar Context for Multi-Site MRI Harmonization

[![arXiv](https://img.shields.io/badge/arXiv-2603.17219-b31b1b.svg)](https://arxiv.org/abs/2603.17219)
[![GitHub](https://img.shields.io/badge/GitHub-SA--CycleGAN--2.5D-black?logo=github)](https://github.com/ishrith-gowda/SA-CycleGAN-2.5D)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![MICCAI 2026](https://img.shields.io/badge/Under%20Review-MICCAI%202026-blue)](https://arxiv.org/abs/2603.17219)

**Authors:** [Ishrith Gowda](https://github.com/ishrith-gowda) (UC Berkeley EECS), Chunwei Liu (Purdue University)

**Paper:** [SA-CycleGAN-2.5D: Self-Attention CycleGAN with Tri-Planar Context for Multi-Site MRI Harmonization](https://arxiv.org/abs/2603.17219)

---

## overview

multi-site neuroimaging analysis is fundamentally confounded by scanner-induced covariate shifts, where the marginal distribution of voxel intensities P(x) varies non-linearly across acquisition protocols while the conditional anatomy P(y|x) remains constant. this is particularly detrimental to radiomic reproducibility, where acquisition variance often exceeds biological pathology variance.

SA-CycleGAN-2.5D is a domain adaptation framework that reduces Maximum Mean Discrepancy (MMD) by **99.1%** (1.729 → 0.015) across two institutional MRI domains (BraTS and UPenn-GBM) without paired training data, collapsing a trained scanner domain classifier to near-chance accuracy (**59.7%**).

---

## key results

| metric | value | detail |
|---|---|---|
| MMD reduction | **99.1%** | 1.729 → 0.015 |
| domain classifier accuracy | **59.7%** | vs. 50% random baseline |
| cohen's d (attention ablation) | **1.32** (p < 0.001) | global attention is statistically essential |
| SSIM (federated, 40 rounds) | **0.998** | FedAvg, zero raw data sharing |
| SSIM (neural compression) | **0.970** | factorized entropy model, 8.4 bpe |
| training cohort | **654 glioma patients** | BraTS + UPenn-GBM, 52K slices |

---

## architecture

SA-CycleGAN-2.5D integrates three architectural innovations:

### 1. 2.5D tri-planar manifold injection
- input: **12-channel volume** (3 adjacent axial slices × 4 MRI modalities: T1, T1ce, T2, FLAIR)
- preserves through-plane gradients ∇z at O(HW) complexity, bridging 2D efficiency and 3D spatial consistency
- eliminates the slice-discontinuity artifacts common in 2D-only harmonization approaches

### 2. U-ResNet generator with dense voxel-to-voxel self-attention
- surpasses the O(√L) receptive field limit of standard CNNs
- models global scanner field biases — critical for field-strength-induced intensity shifts
- CBAM (Convolutional Block Attention Module) applied at resolution layers [3, 4, 5]
- 9 residual blocks, ngf=64

### 3. spectrally-normalized discriminator
- constrains Lipschitz constant K_D ≤ 1 for stable adversarial optimization
- prevents mode collapse without gradient penalty overhead

### journal extension contributions (5 novel modules)

| contribution | description | key metric |
|---|---|---|
| **federated harmonization** | FedAvg across distributed sites, zero raw data transfer | SSIM = 0.998, 40 rounds |
| **neural compression** | joint harmonization + factorized entropy model bottleneck | SSIM = 0.970, 8.4 bpe |
| **multi-domain AdaIN** | 4-scanner-domain style transfer with domain embedding | 4 domains: BraTS, UPenn 3T TrioTim, UPenn 3T other, UPenn 1.5T |
| **PatchNCE loss** | patch-level contrastive loss for structure preservation | — |
| **downstream eval** | cross-site U-Net segmentation transfer (Dice, HD95) | — |

---

## intended use

**intended uses:**
- multi-site MRI harmonization for clinical/research neuroimaging pipelines
- pre-processing step before radiomic feature extraction, segmentation, or classification
- privacy-preserving harmonization via the federated variant (no raw data sharing)
- bandwidth-constrained deployment via the compression variant

**out-of-scope uses:**
- modalities other than brain MRI (not validated)
- clinical diagnosis (research use only)
- sites with fewer than ~50 subjects per scanner type (insufficient domain coverage)

---

## training details

| parameter | value |
|---|---|
| framework | PyTorch 2.6.0 + CUDA 12.4 |
| hardware | NVIDIA A100 80GB PCIe |
| batch size | 32 |
| image size | 128 × 128 |
| optimizer | Adam (β1=0.5, β2=0.999) |
| learning rate | 5e-5 (G), 5e-5 (D) |
| scheduler | cosine annealing with 5-epoch warmup |
| mixed precision | AMP (fp16) |
| data workers | 16-worker parallel I/O + full in-memory slice caching |
| throughput | 4× over standard PyTorch baseline |

---

## usage

```python
import torch
from neuroscope.models.generator import UResNetGenerator
from neuroscope.models.cyclegan import SACycleGAN25D

# load model
model = SACycleGAN25D.from_pretrained("ishrith-gowda/SA-CycleGAN-2.5D")
model.eval()

# input: [B, 12, H, W] — 3 adjacent axial slices × 4 modalities (T1, T1ce, T2, FLAIR)
# normalized to [-1, 1] per modality
input_25d = torch.randn(1, 12, 128, 128)  # replace with real volume

with torch.no_grad():
    harmonized = model.generator_A2B(input_25d)  # [B, 4, H, W] — center slice, all modalities
```

see the [official repository](https://github.com/ishrith-gowda/SA-CycleGAN-2.5D) for full preprocessing pipeline, dataset loading, and inference scripts.

---

## data

trained on:
- **[BraTS-TCGA-GBM](https://www.cancerimagingarchive.net/collection/tcga-gbm/)** — The Cancer Imaging Archive, multi-institutional glioblastoma cohort
- **[UPenn-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/)** — University of Pennsylvania GBM cohort (multiple scanner configurations)

see [`ishrith-gowda/MRI-Harmonization-BraTS-UPenn`](https://huggingface.co/datasets/ishrith-gowda/MRI-Harmonization-BraTS-UPenn) for the preprocessed dataset card.

---

## ethical considerations

- **data privacy:** all training data is de-identified per TCIA data use agreements
- **federated variant:** enables cross-site harmonization with zero raw data transfer, directly addressing patient privacy in multi-institutional settings
- **clinical use:** this is a research tool. outputs should not be used for clinical diagnosis without further validation
- **bias:** evaluated exclusively on adult glioma patients. performance on pediatric, non-glioma, or non-brain MRI is unknown

---

## citation

```bibtex
@article{gowda2026sacyclegan25d,
  title={SA-CycleGAN-2.5D: Self-Attention CycleGAN with Tri-Planar Context for Multi-Site MRI Harmonization},
  author={Gowda, Ishrith and Liu, Chunwei},
  journal={arXiv preprint arXiv:2603.17219},
  year={2026},
  url={https://arxiv.org/abs/2603.17219},
  doi={10.48550/arXiv.2603.17219}
}
```

---

## acknowledgments

data provided by The Cancer Imaging Archive (TCIA) under the BraTS-TCGA-GBM and UPenn-GBM collections. compute provided by Chameleon Cloud (NSF-funded research infrastructure). research conducted at MIT CSAIL under the supervision of Dr. Chunwei Liu.
