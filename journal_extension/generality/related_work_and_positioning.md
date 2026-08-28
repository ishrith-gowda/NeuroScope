# related work & positioning — "fidelity is not utility" (generality paper)

research-agent synthesis (2026-08-28). grounds the related-work + reproducibility sections and,
crucially, fixes the experimental design: the headline artifact is the **fid-vs-miou pareto
frontier across sdedit strength**, with a **prompt ablation**, showing no learned setting
pareto-dominates the non-learned baseline.

## bottom line on novelty

the individual observations are each already documented; the **unification into a single controlled
law is the contribution**:

1. "fid / perceptual fidelity does not track downstream utility" — known, esp. medical
   (konz'24, metrics-that-matter'25, chexgenbench'25) and metric-critique (kynkaanniemi iclr'23).
2. "distribution-matching translation alters task-relevant content" — known since cohen'18
   (tumor hallucination); the explicit motivation for semantic-consistency losses (cycada'18, peng iccv'23).
3. "non-learned domain alignment rivals learned translation for segmentation DA" — known (fda cvpr'20).

**novel & publishable = the synthesis into one controlled dissociation:** a *learned-vs-non-learned*
contrast as the central claim, held against a **frozen** downstream model, shown **architecture-agnostic**
(reproduces under a GAN objective = cyclegan AND a diffusion objective = sdedit) and **cross-domain**
(driving + medical). no prior work runs the clean 2×2 {learned, non-learned} × {fidelity↓?, utility↓?}
as the *thesis*, ties the failure to the *learned distribution-matching objective* (not one architecture),
or traces the fidelity–utility pareto frontier via the sdedit strength knob.

**reviewer risk to neutralize up front:** "we already knew fid ≠ utility (konz/chexgenbench)" and
"content-drift is why semantic-consistency losses exist (cycada)". defense: those are *scattered,
single-architecture, single-domain* observations treated as bugs to patch; we establish a *general,
mechanism-level dissociation* with a matched non-learned control, and show it survives GAN→diffusion,
which none test as a controlled variable. frame as a **measurement/positioning contribution**
("what our metrics actually optimize"), not a new method; the sdedit arm is the evidence that it is
the *objective*, not the architecture.

## closest priors (cite + differentiate)

### tier A — closest (differentiate explicitly)
- **cohen, luck, honari — distribution matching losses can hallucinate features. MICCAI 2018 (arXiv:1805.08841).**
  mechanistic ancestor: cyclegan adds/removes tumors to match target distribution. but qualitative,
  single-architecture, safety-framed, *no non-learned control, no fidelity-vs-utility dissociation*.
- **konz et al. — rethinking perceptual metrics for medical image translation. arXiv:2404.07318 (2024; MIDL?).**
  closest statement of "fidelity ≠ utility": no perceptual metric reliably correlates w/ segmentation; fid
  "especially inconsistent". but a *metric-evaluation* paper, medical-only, GAN-only, no learned/non-learned
  contrast. **most dangerous "already known" cite** — differentiate on (learned/non-learned)+(gan∧diffusion)+(cross-domain).
- **metrics that matter (arXiv:2505.07175, 2025).** upstream metric rankings of VAE/GAN/DDPM misalign with
  downstream segmentation (brats/ixi). **most threatening to the cross-family claim** (already spans gan+ddpm),
  but evaluates *generation*, not *translation*, no non-learned baseline, no driving domain. cite proactively.
- **chexgenbench (arXiv:2505.10496, 2025).** low fid ⇏ downstream utility; same spirit but synthetic
  x-ray generation/augmentation, no translation, no non-learned control.

### tier B — "we already fix this" (frame as motivation we generalize)
- **cycada (hoffman et al., ICML 2018).** adds semantic-consistency loss *because* pixel translation may not
  preserve semantics → we explain *why* such losses are needed; unconstrained objective is what fails.
- **peng et al. — diffusion translation with label guidance. ICCV 2023 (arXiv:2308.12350).** states gans
  "change semantic structure … detrimental to pixel-wise labels"; adds label guidance. a segmentation-DA
  reviewer will name this ("label-guided diffusion preserves structure → your degradation is a weak-baseline
  artifact"). **preempt**: the fix requires injecting *non-distributional* (label/structure) constraints;
  the unconstrained learned objective is what fails. note: they picked strength≈0.3 by validation among
  {0.3,0.5,0.7}; above that "input category typically not retained" — evidence *for* our claim.
- **fda (yang & soatto, CVPR 2020).** non-learned (low-freq spectrum swap) reaches SOTA seg-DA; but its metric
  *is* downstream miou — never frames fidelity-vs-utility. corroborates our non-learned baselines as principled.
- **richter et al. — enhancing photorealism enhancement. T-PAMI 2022 (arXiv:2105.04619).** uses seg-based KPI
  (skvd) as guardrail realism must not break → field implicitly knows realism can break task content.

### tier C — metric-critique backbone
- **kynkaanniemi et al. — role of imagenet classes in fid. ICLR 2023 (oral).** fid gameable w/o quality gain.
- **parmar, zhang, zhu — clean-fid (aliased resizing subtleties). CVPR 2022.** justifies clean-fid + metric care.

**net positioning sentence:** *prior work has separately shown (i) fidelity metrics mis-rank generators for
downstream tasks [konz'24, metrics-that-matter'25, chexgenbench'25], (ii) distribution-matching translation
alters task-relevant content [cohen'18, cycada'18, peng'23], and (iii) non-learned alignment suffices for
segmentation DA [fda'20]. we unify these into a single controlled dissociation and show it is a property of
the learned distribution-matching objective itself — reproducing under both adversarial (cyclegan) and
diffusion (sdedit) generators across driving and medical domains — not an artifact of any architecture or metric.*

## diffusion arm (sdedit) — design implications

- **sdedit (meng et al., ICLR 2022, arXiv:2108.01073).** paper's own framing: "sdedit naturally finds a
  trade-off between realism and faithfulness". control knob = reverse-diffusion start time t0 = SD img2img
  **`strength` ∈ [0,1]**. this is *exactly* our fidelity-vs-utility axis, made continuous.
- **DO THE SWEEP (the money figure):** strengths ~{0.2,0.3,0.4,0.5,0.6,0.7}; plot **fid-vs-miou pareto
  frontier**. predicted: strength↑ ⇒ fid↓ (more cityscapes-real) while miou↓ (structure/labels destroyed),
  and **no strength pareto-dominates the non-learned baseline**. disarms "you used a bad strength".
- **prompt ablation:** run target-prompt AND empty-prompt (isolate img2img prior from text guidance);
  reviewers will ask whether degradation is text-driven or prior-driven. fix per-image seed so strength is
  the only varying factor.
- **top 3 "why not X?" a reviewer will demand** (run ≥1 to make the rebuttal empirical):
  1. **controlnet (zhang et al., ICCV 2023)** seg/edge/depth-conditioned → preserves structure, but that
     injects the label explicitly → *confirms* thesis (utility needs non-distributional constraint).
  2. **plug-and-play diffusion features (tumanyan et al., CVPR 2023)** training-free, preserves layout →
     the "structure-preserving diffusion" point on the frontier (high miou, less fid gain, near non-learned).
  3. **instructpix2pix (brooks et al., CVPR 2023)** the default generic edit baseline (expected: fid↓, miou↓).
  also name-check label-conditioned diffusion DA SOTA: peng'23, dginstyle (ECCV 2024), zodi (2024) — we don't
  beat them (not a method paper) but note they all rely on explicit label/structure guidance.

## reproducibility packet checklist (iclr 2026)

- **tracking/config:** w&b or mlflow per run (config+git-sha+metrics+samples); report exact commit per table;
  hydra-style yaml so every number maps to a committed config.
- **seeds/determinism:** fix & report seeds; ≥3 seeds mean±std for headline results (single-seed generative =
  known reviewer complaint); note nondeterministic ops. sdedit: fix per-image seed across the strength sweep.
- **env pinning:** exact torch/cuda/cudnn, lockfile (uv.lock/pip freeze), container ideal; pin pretrained
  checkpoints by hash (SD version, segformer-b4, nnunet).
- **metric repro (critical):** cite clean-fid; report resize/backbone/split/N/reference-set; freeze & hash the
  downstream evaluator (segformer-b4 / nnunet) — the whole claim rests on the evaluator being fixed.
- **data/model docs:** datasheet (gebru) per derived dataset (gta5/cityscapes/rhuh/brats); model card
  (mitchell) per frozen model; croissant metadata; papers-with-code entry.
- **compute disclosure:** hardware, wall-clock, gpu-hours, approx energy/co2.
- **artifact:** anonymized repo; one-command repro (`make reproduce`) regenerating the headline fid-vs-miou
  table from cached translations + a heavier path regenerating translations; REPRODUCE.md maps figure/table→cmd;
  checksums. one-page reproducibility statement pointing to appendix/code/datasheet/seed/compute (not restating).

## flags (verify before citing)
- konz venue: likely MIDL 2024 — safe cite arXiv:2404.07318.
- metrics-that-matter (2505.07175) & chexgenbench (2505.10496): venues unconfirmed (arXiv/under review).
- richter T-PAMI 2022 vs arXiv 2021 — confirm journal year.
- clean-fid arxiv id: title "on aliased resizing and surprising subtleties in gan evaluation" (parmar,zhang,zhu,
  CVPR 2022) correct; verify id (2104.11222 from memory).
- no paper found making our exact unifying claim as its thesis — keep checking openreview/arxiv near submission
  for concurrent 2026 preprints.
