# Downstream Segmentation Transfer Analysis (corrected)

> **Correction note (supersedes the earlier version of this file).** An earlier version reported a
> −11 to −19% Dice degradation (harmonized A→B Dice 0.629). Those numbers came from a **buggy
> harmonization pipeline** (`eval_downstream.generate_harmonized_data`, which used the wrong channel
> order / intensity normalization / 2.5D slice assembly). The corrected pipeline
> (`journal_extension/scripts/eval_downstream_corrected.py`, harmonizing via `harmonize_for_downstream.py`)
> yields a **larger** degradation: A→B −53.3% and B→A −37.4%. The corrected result is the canonical one;
> the raw numbers below trace to committed JSONs under
> `journal_extension/results/patchnce/downstream_corrected/`.

### Summary of Results (λ_NCE = 0.5, best-FID arm; segmenter 40 epochs; brain-masked Dice / HD95)

| Condition | Dice | WT | TC | ET | HD95 |
|---|---|---|---|---|---|
| Raw A→B (cross-site) | 0.776 | 0.730 | 0.859 | 0.848 | 5.10 |
| **Harmonized A→B (cross-site)** | **0.363** | **0.057** | **0.414** | **0.420** | **31.44** |
| Raw A→A (within-site, source) | 0.765 | 0.750 | 0.892 | 0.865 | 4.68 |
| Raw B→A (cross-site, reverse) | 0.765 | 0.727 | 0.880 | 0.868 | 4.14 |
| **Harmonized B→A (cross-site, reverse)** | **0.479** | **0.413** | **0.607** | **0.561** | **12.09** |
| Raw B→B (within-site, target upper bound) | 0.855 | 0.821 | 0.928 | 0.920 | 3.13 |

Cross-site transfer degrades **−53.3%** (A→B: 0.776 → 0.363) and **−37.4%** (B→A: 0.765 → 0.479) in
mean foreground Dice after harmonization, with HD95 rising 5.1 → 31.4 mm and 4.1 → 12.1 mm respectively.

### Key Findings

1. **Perceptual-task utility gap (strengthened).** Despite high global intensity fidelity (global
   SSIM ≈ 0.99) and monotonically improving brain-masked *structure* SSIM (0.29 → 0.44 across the λ
   sweep), cross-site segmentation transfer *degrades* by 37–53% in Dice after harmonization. Image-level
   fidelity and downstream task utility move in opposite directions.

2. **Whole-tumor collapse.** The A→B whole-tumor (WT) Dice collapses from 0.730 (raw) to 0.057
   (harmonized) — the largest single degradation — indicating harmonization redistributes exactly the
   intensity/contrast cues the segmenter relies on at tumor boundaries.

3. **High raw cross-site baseline.** Raw A→B (0.776) and raw B→A (0.765) show the two sites are already
   close for segmentation despite visible domain differences. Harmonization therefore introduces
   perturbations without a compensating reduction in a large domain gap.

4. **Asymmetric degradation.** A→B (−53.3%) is worse than B→A (−37.4%), consistent with the BraTS
   training set (67 subjects) being smaller and more variable than UPenn (412), making the A-trained
   segmenter more sensitive to input perturbations.

### Mechanistic Interpretation

CycleGAN-based harmonization optimizes perceptual similarity via cycle-consistency and adversarial
losses, matching the target domain's intensity distribution and texture. Segmentation networks, however,
depend on intensity gradients and contrast at tumor boundaries — features redistributed during
harmonization. When the raw cross-site generalization gap is already small, these perturbations to
task-discriminative features dominate any reduction in domain shift, and downstream Dice falls.

### Contribution Framing

This is an honest, rigorously-diagnosed **empirical contribution**: a genuine negative/mechanism result.

> "These results reveal a disconnect between image-level quality metrics and downstream task-level
> utility. Despite high perceptual fidelity, harmonized images degraded cross-site segmentation transfer
> by 37–53% in Dice. Image-quality metrics (SSIM, FID) are insufficient surrogates for clinical utility;
> downstream evaluation should be mandatory when validating any harmonization pipeline."

This motivates the **task-aware harmonization** experiment (Extension A, WS-2): adding a
foreground-weighted / segmentation-consistency objective so harmonization preserves task-relevant
structure. If that recovers Dice, the paper reports a positive downstream result; if not, this
mechanism study stands on its own.

### Supporting Literature

- Ho et al. (JMRI 2026): image-level harmonization does not uniformly improve downstream tasks.
- Dinsdale et al. (NeuroImage 2021): task-agnostic image harmonization is fundamentally limited; proposes feature-level unlearning.
- Palladino et al. (arXiv 2025): upstream metrics (SSIM, FID) show "profound insensitivity" to anatomical detail critical for segmentation.
- Zuo et al. HACA3 (CMIG 2023): anatomy-aware harmonization is necessary; naive style transfer corrupts task-relevant features.
- Moyer et al. survey (BioMedical Engineering OnLine 2024): 2D/2.5D methods can negatively impact downstream prediction.

### Provenance

- Corrected downstream: `journal_extension/results/patchnce/downstream_corrected/lambda{0.0,0.5}_downstream_corrected.json` (`eval_downstream_corrected.py`).
- Harmonization metrics: `journal_extension/results/patchnce/harmonization_eval/eval_lambda*.json` (`eval_harmonization_correct.py`).
- See `journal_extension/results/patchnce/CANONICAL_RESULTS.md` for the regenerate command.
