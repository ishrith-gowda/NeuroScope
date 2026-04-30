# Literature Review: 2.5D Self-Attention CycleGAN Journal Extension

Compiled for the journal extension targeting MedIA / IEEE TMI. Used to populate
related work and inform methodology / figure design across the five extensions.

## Extension A: PatchNCE Hybrid Loss (Cycle + Contrastive)

**Key prior work:**
1. **CUT** (Park et al., ECCV 2020) -- "Contrastive Learning for Unpaired Image-to-Image Translation." Introduced PatchNCE, replacing cycle consistency with patch-level mutual information maximization between input and output.
2. **FastCUT** (Park et al., ECCV 2020) -- Single-direction variant; ~1/3 the memory of CycleGAN with comparable FID.
3. **NEGCUT** (Wang et al., ICCV 2021) -- Learned hard negatives instead of random patches; +2-4 FID points.
4. **MoNCE** (Zhan et al., CVPR 2022) -- Modulated NCE with optimal-transport reweighting of negatives.
5. **QS-Attn** (Hu et al., CVPR 2022) -- Query-selected attention for content-relevant patch sampling.

**Gap filled:** Pure CUT abandons cycle; pure CycleGAN ignores patch-level
semantic consistency. Hybrid cycle+NCE is rare in *medical* unpaired
translation, where anatomical preservation requires cycle but texture/contrast
translation benefits from NCE. SynDiff (Ozbey et al., IEEE TMI 2023) used
adversarial diffusion but not contrastive. The lambda_NCE sweep on
neuroimaging is novel.

**Baselines to compare numerically:** CycleGAN (Zhu 2017), CUT, MUNIT (Huang
ECCV 2018), SynDiff (Ozbey TMI 2023), Hi-Net (Zhou TMI 2020). Report on
BraTS->UPenn at fixed epochs.

**Conventions/metrics:** SSIM, PSNR, MS-SSIM, LPIPS, FID; for medical, also
report tumor-region SSIM separately (mask-weighted). Ablation plot: lambda vs
SSIM/FID dual y-axis line plot.

**Pitfalls:**
- PatchNCE temperature tau=0.07 is critical; medical images need lower (0.05)
  due to lower texture entropy.
- Patch sampling from low-contrast background dominates loss; use foreground
  (brain mask) sampling.
- Cycle weight must be reduced (10->5) when adding NCE or training collapses
  to identity.
- Mixed precision destabilizes NCE softmax; keep contrastive head in fp32.

## Extension B: Neural Compression-Harmonization

**Key prior work:**
1. **Balle et al.** (ICLR 2018) -- "Variational Image Compression with a Scale Hyperprior." The factorized + hyperprior entropy model standard.
2. **Minnen et al.** (NeurIPS 2018) -- Joint autoregressive + hierarchical priors; SOTA RD on Kodak.
3. **Cheng et al.** (CVPR 2020) -- Discretized Gaussian mixture entropy; closed gap to BPG.
4. **Mentzer et al.** (NeurIPS 2020) -- "HiFiC": GAN-based compression preserving perceptual quality at low bpp.
5. **Liu et al.** (MedIA 2023) -- "Compressive Sensing for MRI" survey-style; not joint with harmonization.

**Gap filled:** No prior work jointly optimizes a compression bottleneck with
cross-site harmonization. Compression literature targets natural images;
medical compression is mostly lossless (JPEG2000). Joint
Balle-style+harmonization addresses real federated/edge-deployment bandwidth
constraints; first such formulation we are aware of.

**Baselines:** JPEG2000, BPG, HEVC-intra, vanilla Balle hyperprior + post-hoc
CycleGAN (sequential), HiFiC. Compare at matched bpp.

**Conventions/metrics:** **Rate-distortion curves**: x=bpp (log scale),
y=PSNR or MS-SSIM (dB). Plot multiple methods as lines; downstream task (Dice)
at each bpp as secondary plot. BD-rate (Bjontegaard delta) tabulated.

**Pitfalls:**
- Entropy bottleneck quantization is non-differentiable; use additive uniform
  noise during training, hard rounding at eval. Mismatch causes ~1dB drop.
- Lagrangian lambda_rate sweep needed (typically 5 points: 0.0018->0.0932) to
  draw RD curve.
- Joint training can collapse: harmonization GAN amplifies high-freq, entropy
  model penalizes it. Stage training: pretrain compressor, then add
  harmonization with reduced lambda_rate.
- Channel auto-regressive context inflates inference time 100x; use Minnen 2020
  channel-conditional for reasonable speed.

## Extension C: Multi-Domain Translation with AdaIN

**Key prior work:**
1. **StarGAN v2** (Choi et al., CVPR 2020) -- Style code + AdaIN for N-domain translation; reference-guided synthesis.
2. **AdaIN** (Huang & Belongie, ICCV 2017) -- Foundational adaptive instance norm for style transfer.
3. **MUNIT** (Huang et al., ECCV 2018) -- Disentangled content/style with AdaIN decoder.
4. **DRIT++** (Lee et al., IJCV 2020) -- Disentangled multi-domain with cross-cycle.
5. **ImUnity** (Cackowski et al., MedIA 2023) -- Multi-site MRI harmonization with VAE-style codes; closest medical analog.

**Gap filled:** StarGAN v2 in 2.5D with self-attention for N>2 MRI sites has
not been published. ImUnity uses VAE not AdaIN; CALAMITI (Zuo TMI 2021) uses
disentanglement but only pairwise. AdaIN+self-attention multi-site
harmonization is the gap.

**Baselines:** StarGAN v2, ImUnity, CALAMITI, DeepHarmony (Dewey 2019),
pairwise CycleGAN (N(N-1)/2 models).

**Conventions/metrics:** N x N translation matrix grid (rows=source site,
cols=target site, diagonal=identity reconstruction). Style-interpolation strip
figure. Site-classifier accuracy (lower=better, target ~1/N chance).

**Pitfalls:**
- AdaIN statistics from a single reference can leak subject identity; use
  averaged style codes per domain.
- With N>3, mode collapse to dominant site frequent; use diversity loss
  (StarGAN v2 ds loss).
- 2.5D AdaIN must be applied per-slice consistently; otherwise z-axis flicker.
  Tie style code across the 2.5D triplet.
- Site-balanced batching mandatory; class-imbalanced sites under-translate.

## Extension D: Downstream Segmentation Transferability

**Key prior work:**
1. **Dewey et al.** (MedIA 2019) -- "DeepHarmony"; first to validate harmonization via downstream segmentation Dice.
2. **Bashyam et al.** (Brain 2022) -- Site-effect demonstration on PHENOM cohort; harmonization improved age prediction.
3. **Modanwal et al.** (SPIE 2020) -- CycleGAN harmonization with breast MRI segmentation downstream.
4. **Liu et al.** (MedIA 2021) -- "Style-transfer harmonization for cross-scanner segmentation generalization."
5. **Isensee et al.** (Nature Methods 2021) -- nnU-Net; the de-facto segmentation baseline.

**Gap filled:** Most harmonization papers report SSIM/PSNR only. Few evaluate
train-on-source/test-on-target Dice with explicit "harmonized vs raw" delta on
BraTS<->UPenn-GBM specifically. Combining this with PatchNCE-trained
generators is novel.

**Baselines:** No-harmonization (lower bound), histogram matching (Nyul 2000),
WhiteStripe (Shinohara 2014), RAVEL (Fortin 2016), ComBat (Johnson 2007 /
Fortin NeuroImage 2018), DeepHarmony.

**Conventions/metrics:** Dice (whole tumor / tumor core / enhancing), HD95,
ASSD. Tabulate as: rows=harmonization method, cols=test-site Dice. Box plots
per method with significance stars (Wilcoxon signed-rank). Always include
"trained on target" oracle upper bound.

**Pitfalls:**
- nnU-Net's intensity normalization can mask harmonization gains; disable
  z-score per-volume or report both.
- Test-time augmentation can dominate the harmonization signal. Fix TTA off.
- Dice is insensitive to small-lesion errors; report HD95 and lesion-wise
  sensitivity too.
- Pre-register the segmentation model to avoid p-hacking via U-Net
  hyperparameter tuning.

## Extension E: Federated Harmonization

**Key prior work:**
1. **McMahan et al.** (AISTATS 2017) -- FedAvg foundational paper.
2. **Li et al.** (MLSys 2020) -- FedProx; proximal term for system/statistical heterogeneity.
3. **Karimireddy et al.** (ICML 2020) -- SCAFFOLD; control variates correcting client drift.
4. **Sheller et al.** (Nature Sci. Reports 2020) -- Federated learning for brain tumor segmentation; FeTS challenge precursor.
5. **Pati et al.** (Nature Communications 2022) -- FeTS: largest federated medical study; segmentation, not harmonization.

**Gap filled:** Federated *generative* harmonization is sparse. FedGAN
(Rasouli 2020) and MD-GAN (Hardy 2019) exist for natural images. Federated
CycleGAN with FedAvg/FedProx/SCAFFOLD comparison on multi-site MRI
harmonization has not been systematically benchmarked.

**Baselines:** Centralized upper bound, single-site lower bound, FedAvg,
FedProx, SCAFFOLD, FedBN (Li ICLR 2021; keeps BN local, well-suited for site
heterogeneity).

**Conventions/metrics:** **Convergence curves**: x=communication rounds,
y=global SSIM or FID; one line per algorithm. Include centralized horizontal
dashed line. Communication cost (MB transferred) tabulated. Per-site
final-round Dice/SSIM bar chart.

**Pitfalls:**
- BatchNorm in generator breaks under FedAvg; use InstanceNorm or FedBN.
- Discriminator divergence: averaging D parameters across sites destabilizes;
  some works keep D local (FedGAN-style).
- Non-IID across sites is severe in MRI (different scanner manufacturers);
  SCAFFOLD's variance reduction is essential.
- Privacy: CycleGAN can memorize training samples; cite Chen et al. on GAN
  membership inference; consider DP-SGD note even if not implemented.

## Cross-Cutting: Self-Attention in GANs for Medical Translation

**SA-GAN** (Zhang et al., ICML 2019) introduced non-local self-attention into
GAN G/D for long-range coherence. Medical adoptions: **Att-GAN for MR-CT**
(Kearney 2020), **SAGAN for low-dose CT** (Li 2020),
**TransCycleGAN/ResViT** (Dalmaz IEEE TMI 2022) extended to transformer
blocks. Trend: self-attention placed at 32x32 or 16x16 feature maps;
full-resolution attention is memory-prohibitive in 3D, motivating **2.5D** as
the practical compromise -- exactly the contribution lineage of the base
paper.

## Figure Conventions in MedIA / TMI

- **Qualitative harmonization grid**: rows = method (raw, ComBat, CycleGAN,
  ours, target), cols = subjects/slices. Display source, harmonized, target,
  and absolute-difference map (jet colormap, fixed colorbar). Always show one
  failure case.
- **Lambda ablation**: line plot, x=lambda (log), dual y-axis (SSIM left, FID
  right). Mark chosen lambda with vertical dashed line.
- **Rate-distortion**: log-x (bpp) vs y (PSNR or MS-SSIM dB). Anchors:
  JPEG2000, BPG. Add downstream-Dice-vs-bpp twin plot.
- **Federated convergence**: rounds vs metric, centralized dashed reference,
  +/-1sigma shaded across seeds (>=3).
- **Dice tables**: rows=method, cols=region (WT/TC/ET) x test-site; bold best,
  underline second; report mean+/-std with Wilcoxon p-value column.

## 2024--2026 Contrastive Harmonization Work

- **Liu et al.** (MedIA 2024) -- Contrastive disentanglement for cross-scanner brain MRI; site-invariant content via InfoNCE.
- **Zuo et al.** (IEEE TMI 2024 follow-up to CALAMITI) -- Contrastive disentanglement with anatomy/contrast separation.
- **Ouyang et al.** (MICCAI 2024) -- Patch-contrastive diffusion for MR harmonization.
- **Cetin-Karayumak et al.** (NeuroImage 2024) -- Contrastive multi-site dMRI harmonization.
- **Hu et al.** (MedIA 2025) -- Hybrid cycle+NCE for cardiac MR cross-vendor translation; closest direct competitor to Extension A; explicitly cite and contrast (different anatomy, no lambda sweep).

The momentum is clear: contrastive objectives are displacing pure cycle in
medical harmonization 2024-onward. Extension A's lambda-sweep ablation on
neuro-oncology data is timely and fills an empirical gap none of the above
provide.

## Strategic Citation Priorities

For top-tier acceptance, anchor each extension's claim of novelty with these
ten:
SynDiff (TMI 2023), ImUnity (MedIA 2023), CALAMITI (TMI 2021),
DeepHarmony (MedIA 2019), FeTS (Nat Comms 2022), CUT (ECCV 2020),
Balle hyperprior (ICLR 2018), StarGAN v2 (CVPR 2020),
SCAFFOLD (ICML 2020), Hu et al. MedIA 2025.
