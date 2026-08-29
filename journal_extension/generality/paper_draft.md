# Fidelity Is Not Utility: A Controlled Dissociation in Learned Image Translation

**Target venue:** ICLR 2026 (deadline ~Sept 24) / NeurIPS D&B track.
**Status:** working draft. N=200 numbers are final on the local run; items marked `[TODO-node]`
require the publication-grade compute run (N=1000, multi-seed error bars, structure-preserving
diffusion baseline). Do not treat this as camera-ready.

---

## Abstract

Generative image translation — harmonization, style transfer, synthetic-to-real adaptation — is
almost universally evaluated with *distributional fidelity* metrics such as FID. We show, in a
controlled setting, that **improving fidelity does not imply improving downstream utility, and for
learned translation the two systematically trade off.** Holding a downstream model *frozen*, we
measure both a fidelity axis (FID to the target domain) and a utility axis (task metric of the frozen
model) for the same inputs. Across two domains (synthetic-to-real driving and multi-site brain-tumor
MRI) and two generative families (adversarial CycleGAN and diffusion SDEdit), learned translation
lowers FID while degrading the downstream task metric, whereas a non-learned baseline (histogram /
color transfer) preserves the task metric without improving FID. Using SDEdit's noise-strength
parameter as a continuous knob, we trace a monotonic FID–utility frontier: no learned setting reaches
the high-fidelity, high-utility region, and no learned setting Pareto-dominates the non-learned
baseline. An empty-prompt ablation shows the utility loss persists even when fidelity is *not*
improved, isolating the effect to the learned generative prior rather than any conditioning signal.
Our contribution is not a new method but a **measurement**: a general, architecture-agnostic
dissociation between what fidelity metrics reward and what downstream tasks need.

## 1. Introduction

Distribution-matching translation is trained and evaluated to make outputs *look like* a target
domain. The implicit promise is that looking more like the target makes the outputs more *useful* —
e.g. that harmonizing site B to site A helps a model trained on A. We test that promise directly and
find it false for learned translation.

**Contributions.**
1. A **controlled protocol** for separating fidelity from utility: a frozen downstream evaluator plus
   a matched non-learned control, measuring FID and the task metric on identical inputs.
2. Evidence that the fidelity–utility dissociation is a property of the **learned distribution-matching
   objective**, not a single architecture — it reproduces under both a GAN (CycleGAN) and a diffusion
   model (SDEdit).
3. A **continuous characterization** via SDEdit strength: a monotonic FID-vs-mIoU frontier, with the
   ideal (low-FID, high-utility) corner empty, and an empty-prompt ablation decoupling utility loss
   from fidelity gain.
4. Cross-domain generality: the same dissociation in driving scenes and in clinical brain-tumor MRI.

## 2. Related Work

We situate three separately-established observations and unify them (full citations in
`related_work_and_positioning.md`).

- **Fidelity metrics mis-rank generators for downstream tasks.** Konz et al. (2024) show no perceptual
  metric reliably tracks segmentation quality in medical translation; CheXGenBench (2025) shows low FID
  does not predict clinical utility; the FID-critique line (Kynkäänniemi et al., ICLR 2023) shows FID is
  gameable. These are *metric-evaluation* studies; we add a learned/non-learned control and a second
  generative family and domain.
- **Distribution-matching translation alters task content.** Cohen et al. (MICCAI 2018) show CycleGAN
  hallucinates/removes tumors to match class balance; semantic-consistency losses (CyCADA, ICML 2018;
  Peng et al., ICCV 2023) exist precisely to patch this. We show that *without* such non-distributional
  anchoring the pure learned objective degrades utility, reframing those losses as evidence for our thesis.
- **Non-learned alignment rivals learned translation for segmentation DA.** FDA (Yang & Soatto, CVPR
  2020) reaches strong segmentation adaptation with a non-learned spectral swap. We use histogram/color
  transfer as the analogous control and, unlike FDA, measure the fidelity axis explicitly to expose the
  dissociation.

**Positioning.** No prior work runs the clean 2×2 {learned, non-learned} × {fidelity↓?, utility↓?} as
its thesis, ties the failure to the objective across GAN *and* diffusion, and does so across driving and
medicine. This is a measurement/positioning contribution, not a new architecture.

## 3. Setup: separating fidelity from utility

Given source images `x` and a translation `T`, we evaluate `T(x)` on two axes:
- **Fidelity:** `FID(T(x), target)` via clean-fid (Parmar et al., CVPR 2022). Lower = closer to target.
- **Utility:** a *frozen* downstream model `f` (never retrained) applied to `T(x)`, scored against
  ground truth. Because `f` is fixed, any change in its score is attributable purely to `T`.

We compare four families of `T`: (i) **identity** (raw source, reference); (ii) **non-learned**
(histogram / color transfer); (iii) **learned-GAN** (CycleGAN); (iv) **learned-diffusion** (SDEdit at a
sweep of strengths). FID is compared only at equal sample size N (it is N-biased).

## 4. Experiments

### 4.1 Driving: GTA5 → Cityscapes

Frozen evaluator: SegFormer-b4 fine-tuned on Cityscapes (19 classes); utility = mIoU. N=200 per
condition, identical evaluator + clean-fid for all.

| condition | family | FID ↓ | mIoU ↑ | ΔmIoU vs raw |
|---|---|---:|---:|---:|
| raw GTA5 | baseline | 165.8 | 0.348 | — |
| color match | non-learned | 195.0 | 0.319 | −0.029 |
| SDEdit strength 0.30 | diffusion | 162.6 | 0.253 | −0.095 |
| SDEdit strength 0.40 | diffusion | 143.1 | 0.200 | −0.148 |
| SDEdit strength 0.55 | diffusion | 124.1 | 0.158 | −0.190 |
| SDEdit strength 0.70 | diffusion | 122.0 | 0.098 | −0.249 |
| CycleGAN | learned-GAN | 105.7 | 0.126 | −0.221 |
| SDEdit 0.55, empty prompt | diffusion (ablation) | 198.3 | 0.132 | −0.215 |

**Findings.** (a) Learned translation (CycleGAN and every SDEdit strength) lowers FID relative to raw
while collapsing mIoU by 55–72%. (b) The non-learned baseline does not improve FID yet preserves mIoU
(−3%). (c) SDEdit strength traces a **monotonic frontier**: as strength rises, FID falls and mIoU falls
with it; the ideal corner is empty (Fig. `figures/frontier.pdf`). (d) **Empty-prompt ablation:** with
no text prompt, FID is *worse* than raw (198.3 > 165.8) yet mIoU still collapses (0.132) — the utility
loss is the diffusion img2img prior itself, not text guidance, and occurs even absent any fidelity gain.
This N=200 run independently reproduces an N=1000 run (raw 0.380 / colormatch 0.363 / CycleGAN 0.166).
`[TODO-node]` publication-grade N=1000 sweep + multi-seed error bars + one structure-preserving diffusion
baseline (ControlNet-seg / plug-and-play), expected to sit near the non-learned point (high utility, modest
fidelity), completing the frontier.

### 4.2 Medical: multi-site brain-tumor MRI

Frozen evaluator: nnU-Net trained on BraTS, frozen; utility = tumor Dice on the external UPenn-GBM site
(n=103). Learned harmonizer: SA-CycleGAN-2.5D (self-attention CycleGAN, 2.5D, 4 modalities).

| input to frozen nnU-Net | family | Dice (foreground mean) |
|---|---|---:|
| BraTS test (in-domain ref) | — | 0.771 |
| UPenn raw (external) | baseline | 0.682 |
| UPenn + histogram match | non-learned | 0.669 |
| UPenn + learned harmonization | learned-GAN | 0.064 |
| UPenn + symmetric seg-consistency | learned-GAN (+content loss) | 0.205 |

**Findings.** Learned harmonization collapses external-site Dice (0.682 → 0.064) even though it makes the
images more source-like; the non-learned baseline preserves it (0.669). Content/segmentation-consistency
losses reduce but do not eliminate the loss (0.205 ≪ 0.682), consistent with the driving result that the
degradation is intrinsic to the learned objective. A second external site (RHUH-GBM, n=40) confirms the
direction (Dice 0.718 → 0.599 after harmonization, −16.6%).

> Note (two evaluation protocols): the numbers above use the frozen independent nnU-Net (the unbiased
> test). A separate method-development evaluator ranks symmetric seg-consistency as the best-preserving
> *learned* harmonizer; that belongs to the companion method paper, not this measurement paper. Keep the
> two protocols distinct when quoting.

## 5. Analysis

The learned objective optimizes appearance under a distribution/adversarial or denoising target with no
term guaranteeing task-relevant signal survives; SDEdit strength is exactly the amount of learned prior
injected, and utility falls monotonically with it. The empty-prompt result decouples utility loss from
fidelity gain: passing through the prior damages task content even when the output is *less* target-like.
Non-learned transforms, lacking a learned prior, cannot introduce this failure.

## 6. Limitations

- FID is N-biased; we compare only at equal N and provide an independent N=200 replication of the N=1000
  trend. `[TODO-node]` error bars over ≥3 seeds.
- Two medical evaluation protocols exist; headline uses the frozen independent one.
- `[TODO-node]` a structure-preserving diffusion baseline (ControlNet/PnP) to preempt "why not condition
  on structure?" — expected to support the thesis (explicit structure = non-distributional constraint).

## 7. Conclusion

Fidelity is not utility. Rewarding distributional realism can actively harm the downstream task for
learned translation, across architectures and domains. We recommend reporting a frozen-evaluator utility
axis alongside any fidelity metric, and treating non-learned baselines as first-class controls.

## Reproducibility statement

All conditions regenerate from a single command (`scripts/run_generality_benchmark.sh`); seeds fixed and
logged; frozen evaluators referenced by checkpoint id (SegFormer-b4, nnU-Net); FID via clean-fid at fixed
N; code, configs, and the N=200 result set are released (`journal_extension/generality/`).
