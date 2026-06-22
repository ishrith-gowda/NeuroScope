# Progress Tracker — SASHIMI 2026 Sprint

**Updated**: 2026-06-22 (live cluster state)
**Deadline**: SASHIMI 2026 paper **2026-07-01** (~9 days)
**Strategy**: quality > quantity, **sequential** — finish Ext A to an exceptional
standard, then Ext C. Both target SASHIMI; the journal (IEEE TMI / NeuroImage) is
the larger primary deliverable that later incorporates the full suite.

## Extension roadmap (scope)

| Ext | Topic | Venue (current plan) | Sprint status |
|-----|-------|----------------------|---------------|
| **A** | cycle + PatchNCE hybrid | **SASHIMI 2026 — LEAD, in progress** | **~80%** |
| **C** | multi-domain N=4 AdaIN | SASHIMI 2026 / AAAI 2027 — next | **~12%** |
| D | downstream task eval | folded into A's validation; standalone → ISBI/TMI later | (inside A) |
| B | neural compression | ICML 2027 / MedIA — later | 0% |
| E | federated harmonization | FL@FM-NeurIPS / MedIA — later | 0% |

## EXT A (current lead) — ~80% (all experiments complete)

| Component | % | Status |
|-----------|---|--------|
| Infra (A100 env, crash-safe backup/resume, corrected eval suite, manuscript scaffold, results generator) | 100% | DONE — mostly reusable for Ext C |
| λ-sweep training {0.0, 0.1, 0.5, 1.0, 2.0} | 100% | all 5 arms trained (λ=1.0 redo recovered the OOM casualty) |
| Harmonization eval (FID/KID, masked SSIM, domain-clf, MMD) | 100% | 5/5 arms; 5-arm table + sweep figure generated + verified |
| Downstream Dice/HD95 (hybrid + cycle-only) | 100% | both arms done; finding diagnosed (real, not a bug) |
| Manuscript prose (intro/related/method/setup) | 60% | written, compiles (7pp), method diagram in; results/discussion to fill with final numbers |
| Figures/tables | 70% | sweep fig + downstream fig + method diagram + 5-arm/downstream tables done + verified; qualitative montage pending |
| Final assembly (compile, double-blind, submit) | 0% | pending |

**Core result (STRONG, positive):** the λ-sweep validates the mechanism thesis —
FID has a clean interior optimum at λ=0.5 (51.8→47.3→48.2), source-structure
preservation rises monotonically (0.29→0.44), cycle consistency holds (~0.71), and
domain confusion reaches chance. PatchNCE measurably improves BOTH distributional
match and content preservation over pure cycle-consistency.

**Downstream finding (HONEST nuance, diagnosed):** harmonization does NOT help
cross-site tumor segmentation here and uniformly degrades it (~−42% across all arms,
both directions). Diagnosed as genuine, not a pipeline bug (ruled out axis/channel
misalignment, verified contrast preserved): (i) the raw cross-site gap is already
negligible (cross-site Dice 0.776 ≈ within-site 0.765), and (ii) a real-trained
segmenter chokes on GAN-synthesized texture (train-real / test-synthetic mismatch).
Image-harmonization quality and downstream task utility are distinct axes — a
defensible methodological point, not an over-claim. (Cleaner train-on-harmonized
protocol = journal-version strengthening experiment.)

## EXT C (next) — ~12%

| Component | % | Status |
|-----------|---|--------|
| Scaffold + bib + lit-review (positioning; closest-prior MMH flagged) | 100% | DONE |
| Multi-domain training (N=4, from scratch — no warm-start shortcut) | 0% | not started |
| Eval (N×N matrix, per-pair FID/MMD, site-clf, downstream) | 0% | reuses Ext A eval infra |
| Manuscript prose | 0% | scaffold only |

Note: ~70% of Ext C's tooling (env, eval suite, results generator, crash-safety,
harmonization pipeline) is **shared with Ext A** → effective head start beyond 12%.

## Other extensions (later venues, out of current sprint)

- **Ext D** (downstream) — folded into Ext A's validation; standalone short paper → ISBI 2027 later.
- **Ext B** (compression) → ICML 2027 / MedIA — deferred.
- **Ext E** (federated) → FL@FM-NeurIPS / MedIA — deferred.

## Overall (SASHIMI sprint = Ext A + Ext C)

**~33% complete** (Ext A ~55%, Ext C ~12%; shared infra means Ext C accelerates
once Ext A's tooling is final). Critical path: Ext A results land (auto) →
assemble tables/figures/prose → compile + double-blind → submit → start Ext C.

## DEFERRED — repo-wide CI/lint hardening (END of timeline; only if/when time)

Pre-existing legacy debt: **291 files unformatted, 2,607 ruff errors, 2 broken
files** (`scripts/05_downstream_evaluation/downstream_evaluation.py`,
`scripts/train_sa_cyclegan_complete.py` — invalid Python), plus mypy/test debt.
**NOT bulk-fixed mid-sprint** — it's an unreviewable 300-file diff touching code
actively running experiments. New code IS gated clean (ruff/flake8/black/isort +
uv.lock + PR #25). Proper remediation = a dedicated, tested, reviewed pass
**after both SASHIMI papers**: pin CI tool versions / consolidate on ruff,
`ruff format` whole repo, fix/remove the 2 broken files, triage mypy/bandit/test.
