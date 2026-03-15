# SA-CycleGAN-2.5D Journal Extension: Research Plan

**Target venue**: Medical Image Analysis (MedIA) or IEEE Transactions on Medical Imaging (TMI)
**Requirement**: 30% extension beyond MICCAI 2026 submission
**Timeline**: Draft ready by MICCAI decision day (June 12, 2026)

---

## 1. Extension Overview

The journal extension adds five complementary contributions to the MICCAI paper:

| Extension | Novelty | Effort | Impact |
|-----------|---------|--------|--------|
| A. Contrastive hybrid loss (PatchNCE) | Moderate | Medium | High |
| B. Neural compression-harmonization | High | High | High |
| C. Multi-domain (N>2) architecture | Moderate | Medium | High |
| D. Downstream task evaluation | Low | Low | High |
| E. Federated learning feasibility | High | Medium | Medium |

### Content Budget (30% extension = ~3 additional pages)
- Extension A: 0.75 pages (new loss, ablation table, comparison)
- Extension B: 1.0 page (compression-harmonization architecture, rate-distortion curves)
- Extension C: 0.5 pages (multi-domain conditioning, N=4 experiment)
- Extension D: 0.5 pages (segmentation Dice, survival prediction)
- Extension E: 0.25 pages (federated simulation, convergence curves)

---

## 2. Extension A: Contrastive Hybrid Loss (SA-CycleGAN-2.5D + PatchNCE)

### Motivation
CUT (Park et al., ECCV 2020) showed that PatchNCE can replace cycle consistency for
unpaired image translation. We propose a HYBRID: keep cycle consistency AND add PatchNCE
as complementary content preservation at the feature level. This is novel -- no prior work
combines cycle consistency + PatchNCE in a medical harmonization context.

### Architecture Changes
1. Modify SAGenerator25D to support `encode_only` mode returning intermediate features
2. Add per-layer MLP projection heads (2-layer MLP, 256-dim output)
3. Feature extraction at 5 layers: initial conv, 2 downsampling, bottleneck blocks 3 and 6

### Loss Formulation
```
L_total = L_adv + L_cycle + L_identity + L_ssim + lambda_nce * L_PatchNCE
```
where L_PatchNCE = PatchNCE(enc(fake_B), enc(real_A)) + PatchNCE(enc(fake_A), enc(real_B))

### Experiments
- Ablation: lambda_nce in {0.1, 0.5, 1.0, 2.0}
- Comparison: SA-CycleGAN (ours), CUT (same backbone), SA-CycleGAN + PatchNCE (hybrid)
- Metrics: SSIM, PSNR, LPIPS, FID, MMD, domain classifier accuracy

### Key References
- Park et al., "Contrastive Learning for Unpaired I2I Translation", ECCV 2020
- Han et al., "Dual Contrastive Learning for Unsupervised I2I Translation", CVPR 2021
- Jung et al., "Exploring Patch-wise Semantic Relation for Contrastive Learning", CVPR 2022

---

## 3. Extension B: Neural Compression-Harmonization (Harmonize-and-Compress)

### Motivation
Multi-site MRI studies generate petabytes of data. If harmonization is a required
preprocessing step, why not compress simultaneously? We propose "harmonize-and-compress":
a single forward pass that both harmonizes the MRI to the target domain AND produces a
compressed bitstream, avoiding the need for separate harmonization + JPEG2000/lossless
compression steps.

### Architecture: Compression-Aware Bottleneck
The key insight: the CycleGAN bottleneck already learns a compact representation.
We add quantization + entropy coding to make this representation transmittable.

```
Input (12ch) -> Encoder -> [Quantize + Entropy Model] -> Decoder -> Output (4ch)
                              |
                         Compressed Bitstream (rate R)
```

Components:
1. **Quantization**: Uniform scalar quantization with straight-through estimator (STE)
   during training, actual rounding at inference
2. **Entropy model**: Factorized prior (Balle et al. 2018) or hyperprior for better
   rate modeling
3. **Rate-distortion loss**: L_total + lambda_rate * R where R = estimated bitrate
   from entropy model

### Rate-Distortion Formulation
```
L_total = L_harmonization + lambda_rate * R(z_hat)
        = (L_adv + L_cycle + L_idt + L_ssim + L_nce) + lambda_rate * H(z_hat)
```
where z_hat = Q(z) is the quantized bottleneck, H(z_hat) is the cross-entropy
with the learned prior (estimated bitrate).

### Tradeoff Sweep
- lambda_rate in {0.001, 0.005, 0.01, 0.05, 0.1}
- Plot: SSIM vs bits-per-voxel (rate-distortion curve)
- Compare: Harmonize-then-compress (sequential) vs Harmonize-and-compress (joint)
- Baselines: JPEG2000 at matched bitrates, raw neural compression without harmonization

### Clinical Feasibility
- DICOM standard supports JPEG2000 lossy (Transfer Syntax 1.2.840.10008.1.2.4.91)
- FDA guidance allows lossy compression if diagnostic equivalence demonstrated
- Our approach: demonstrate that harmonized+compressed images maintain equivalent
  downstream segmentation performance

### Key References
- Balle et al., "Variational Image Compression with a Scale Hyperprior", ICLR 2018
- Minnen et al., "Joint Autoregressive and Hierarchical Priors", NeurIPS 2018
- Cheng et al., "Learned Image Compression with Discretized Gaussian Mixture", CVPR 2020
- Agustsson et al., "Generative Adversarial Networks for Extreme Learned Compression", ICCV 2019

---

## 4. Extension C: Multi-Domain Architecture (N > 2 Sites)

### Motivation
The MICCAI paper's primary limitation is the 2-domain constraint. Real clinical scenarios
involve N >> 2 sites. We extend SA-CycleGAN-2.5D to handle N domains with a single model.

### Architecture: Domain-Conditioned SA-Generator
Use Adaptive Instance Normalization (AdaIN) to condition the generator on target domain:

```
class DomainConditionedSAGenerator(SAGenerator25D):
    # Replace InstanceNorm with AdaIN
    # Domain embedding: learnable N-dim -> style vector
    # FiLM modulation at each normalization layer
```

Key changes:
1. Domain embedding layer: nn.Embedding(N, d_style) -> MLP -> (gamma, beta) per layer
2. Replace InstanceNorm2d with AdaIN: y = gamma * (x - mu) / sigma + beta
3. Single generator, single discriminator with domain classification head (StarGAN-style)
4. Discriminator: D(x) -> (real/fake score, domain_class)

### Dataset Plan
Split BraTS by scanner metadata to create 4 domains:
- Domain A: BraTS-Siemens (estimated ~35 subjects)
- Domain B: BraTS-GE (estimated ~25 subjects)
- Domain C: BraTS-Philips (estimated ~28 subjects)
- Domain D: UPenn-GBM (566 subjects, all Siemens 3T)

### Experiments
- N=4 multi-domain model vs N=2 pairwise models
- Domain classification accuracy for each pair
- MMD reduction across all domain pairs
- Zero-shot transfer: train on 3 domains, test harmonization to held-out domain

### Key References
- Choi et al., "StarGAN: Unified Generative Adversarial Networks", CVPR 2018
- Choi et al., "StarGAN v2: Diverse Image Synthesis for Multiple Domains", CVPR 2020
- Zuo et al., "HACA3: Unified Approach for Multi-site MR Harmonization", CMIG 2023
- Huang & Belongie, "Arbitrary Style Transfer with AdaIN", ICCV 2017

---

## 5. Extension D: Downstream Task Evaluation

### Motivation
The strongest validation of harmonization quality is downstream task performance.
Show that models trained on harmonized data generalize better across sites.

### Experiments
1. **Tumor segmentation**: Train U-Net on Domain A, test on Domain B (and vice versa)
   - Compare: raw, ComBat-harmonized, SA-CycleGAN-harmonized
   - Metrics: Dice score, Hausdorff distance (HD95), sensitivity, specificity

2. **Domain generalization**: Train segmentation on mixed harmonized data
   - Show improved cross-site Dice vs training on raw mixed data

3. **Radiomics stability**: ICC of top-20 prognostic features before/after harmonization
   - Focus on features known to correlate with overall survival in GBM

### Key References
- Isensee et al., "nnU-Net", Nature Methods 2021
- Dinsdale et al., "Deep Learning-Based Unlearning of Dataset Bias", NeuroImage 2021
- Pomponio et al., "Harmonization of large MRI datasets", NeuroImage 2020

---

## 6. Extension E: Federated Learning Feasibility Study

### Motivation
Real multi-site studies cannot centralize data due to privacy regulations (HIPAA, GDPR).
We explore whether SA-CycleGAN-2.5D can be trained in a federated setting.

### Simulation Framework
- **FedAvg**: Average generator and discriminator weights across K clients
- **FedProx**: Add proximal term to prevent client drift
- **Local discriminator**: Each client keeps its own discriminator, only share generator

### Experiment Design
Simulate K=2 clients (BraTS site, UPenn site):
- Each client has local data only
- Aggregation every E local epochs (E in {1, 5, 10})
- Compare: centralized, FedAvg, FedProx, local-D + shared-G
- Metrics: convergence speed (communication rounds), final SSIM/MMD

### Privacy Analysis
- Gradient inversion attack feasibility on GAN gradients
- Differential privacy: DP-SGD with epsilon in {1, 10, infinity}
- Show tradeoff between privacy budget and harmonization quality

### Key References
- McMahan et al., "Communication-Efficient Learning of Deep Networks", AISTATS 2017
- Li et al., "Federated Optimization in Heterogeneous Networks" (FedProx), MLSys 2020
- Sheller et al., "Federated learning in medicine", Nature Medicine 2020
- Hardy et al., "MD-GAN: Multi-Discriminator Generative Adversarial Networks", NeurIPS 2019

---

## 7. Implementation Priority Order

1. **PatchNCE hybrid loss** (Extension A) -- builds on existing code, fastest to implement
2. **Downstream evaluation** (Extension D) -- uses existing U-Net, adds segmentation metrics
3. **Compression bottleneck** (Extension B) -- novel contribution, high-impact
4. **Multi-domain conditioning** (Extension C) -- architecture refactor
5. **Federated simulation** (Extension E) -- simulation framework, can use existing training loop

---

## 8. File Structure

```
journal_extension/
├── RESEARCH_PLAN.md          (this file)
├── configs/
│   ├── patchnce_hybrid.yaml  (Extension A config)
│   ├── compression.yaml      (Extension B config)
│   ├── multi_domain.yaml     (Extension C config)
│   └── federated.yaml        (Extension E config)
├── scripts/
│   ├── train_hybrid_nce.py   (Extension A training)
│   ├── train_compression.py  (Extension B training)
│   ├── train_multi_domain.py (Extension C training)
│   ├── eval_downstream.py    (Extension D evaluation)
│   └── train_federated.py    (Extension E simulation)
├── experiments/              (results go here)
├── results/                  (metrics, tables)
└── figures/                  (generated visualizations)
```

New model code goes in existing neuroscope/models/ hierarchy:
- neuroscope/models/losses/patchnce.py (multi-layer PatchNCE with MLP heads)
- neuroscope/models/compression/ (quantization, entropy model, compressed generator)
- neuroscope/models/architectures/sa_cyclegan_25d_multidomain.py
- neuroscope/training/federated/ (FedAvg, FedProx implementations)
