## Downstream Segmentation Transfer Analysis

### Summary of Results

| Condition | Dice | WT | TC | ET | HD95 |
|---|---|---|---|---|---|
| Raw A->B (cross-site) | 0.777 | 0.740 | 0.848 | 0.835 | 5.1 |
| Harmonized A->B (cross-site) | 0.629 | 0.532 | 0.776 | 0.747 | 8.9 |
| Raw A->A (within-site) | 0.757 | 0.726 | 0.875 | 0.843 | 4.8 |
| Harmonized A->A (within-site) | 0.745 | 0.765 | 0.855 | 0.824 | 5.0 |
| Raw B->A (cross-site, reverse) | 0.767 | 0.740 | 0.884 | 0.866 | 4.1 |
| Harmonized B->A (cross-site, reverse) | 0.679 | 0.700 | 0.821 | 0.734 | 6.0 |

### Key Findings

1. **Perceptual-task utility gap**: Despite achieving SSIM=0.998 for harmonized output quality,
   cross-site segmentation transfer degrades by 11-19% in mean Dice score after harmonization.

2. **High raw cross-site baseline**: Raw A->B Dice of 0.777 and raw B->A Dice of 0.767 indicate
   the two sites are already similar for segmentation despite visual domain differences. This means
   harmonization introduces perturbations without conferring distributional benefit for the
   downstream task.

3. **Within-site distortion**: Harmonized A->A (0.745) slightly underperforms raw A->A (0.757),
   confirming that harmonization introduces small but measurable distortions even when no domain
   shift is present. This supports the information corruption hypothesis.

4. **Asymmetric degradation**: A->B (-19.1%) is more affected than B->A (-11.5%), potentially
   because the BraTS training set (67 subjects) is smaller and more variable than UPenn (412
   subjects), making the A-trained segmenter more sensitive to input perturbations.

### Mechanistic Interpretation

CycleGAN-based harmonization optimizes perceptual similarity via cycle-consistency and adversarial
losses. This objective encourages matching the target domain's intensity distribution and texture
patterns. However, segmentation networks rely on intensity gradients and contrast patterns at tumor
boundaries -- features that may be redistributed during harmonization. When the raw cross-site
generalization gap is already small, these perturbations to task-discriminative features outweigh
any reduction in domain shift.

### Contribution Framing

This result constitutes an **empirical contribution** to the field. We recommend framing it as:

> "These results reveal a fundamental disconnect between image-level quality metrics and
> downstream task-level utility. Despite achieving near-perfect perceptual fidelity (SSIM=0.998),
> harmonized images degraded cross-site segmentation transfer by 11-19% in Dice score. This
> finding underscores that image-quality metrics (SSIM, PSNR) are insufficient surrogates for
> clinical utility and that downstream evaluation should be mandatory for validating any
> harmonization pipeline."

### Supporting Literature

- Ho et al. (JMRI 2026): systematic benchmark showing image-level harmonization does not uniformly
  improve downstream tasks
- Dinsdale et al. (NeuroImage 2021): theoretical argument that task-agnostic image harmonization is
  fundamentally limited; proposes feature-level unlearning
- Palladino et al. (arXiv 2025): demonstrates that upstream metrics (SSIM, FID) show "profound
  insensitivity" to anatomical details critical for segmentation
- Zuo et al. HACA3 (CMIG 2023): anatomy-aware harmonization necessary; naive style transfer
  corrupts task-relevant features
- Moyer et al. survey (BioMedical Engineering OnLine 2024): notes that 2D/2.5D methods can
  negatively impact downstream brain age prediction

### Proposed Discussion Section Outline

1. **State the finding directly** with exact numbers
2. **Explain the mechanism**: intensity redistribution corrupting task-discriminative features
3. **Contextualize with literature**: this is an increasingly recognized phenomenon
4. **Provide evidence**: within-site degradation, asymmetric direction effects, high raw baseline
5. **Position as contribution**: motivates task-aware or jointly-optimized harmonization
6. **Future directions**: segmentation-aware auxiliary objectives, feature-level harmonization,
   task-conditional style transfer

### Future Work Directions (Motivated by This Finding)

1. **Task-aware harmonization**: add downstream task loss (e.g., segmentation cross-entropy) as an
   auxiliary training objective alongside cycle-consistency
2. **Feature-level harmonization**: operate in latent space rather than image space to preserve
   task-relevant representations (cf. Dinsdale et al.)
3. **Two-tier evaluation protocol**: all future harmonization papers should report both image-level
   metrics (SSIM, PSNR) and at least one downstream task metric (Dice, accuracy)
4. **Conditional harmonization**: learn when harmonization is beneficial (large domain gap) vs.
   harmful (small domain gap) and apply adaptively
