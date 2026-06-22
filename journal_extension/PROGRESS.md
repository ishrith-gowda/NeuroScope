# Progress Tracker — SASHIMI 2026 Sprint

**Updated**: 2026-06-22 (live cluster state)
**Deadline**: SASHIMI 2026 paper **2026-07-01** (~9 days)
**Strategy**: quality > quantity, **sequential** — finish Ext A to an exceptional
standard, then Ext C. Both target SASHIMI; the journal (IEEE TMI / NeuroImage) is
the larger primary deliverable that later incorporates the full suite.

## Extension roadmap (scope)

| Ext | Topic | Venue (current plan) | Sprint status |
|-----|-------|----------------------|---------------|
| **A** | cycle + PatchNCE hybrid | **SASHIMI 2026 — LEAD, in progress** | **~55%** |
| **C** | multi-domain N=4 AdaIN | SASHIMI 2026 / AAAI 2027 — next | **~12%** |
| D | downstream task eval | folded into A's validation; standalone → ISBI/TMI later | (inside A) |
| B | neural compression | ICML 2027 / MedIA — later | 0% |
| E | federated harmonization | FL@FM-NeurIPS / MedIA — later | 0% |

## EXT A (current lead) — ~55%

| Component | % | Status |
|-----------|---|--------|
| Infra (A100 env, crash-safe backup/resume, corrected eval suite, manuscript scaffold, results generator) | 100% | DONE — mostly reusable for Ext C |
| λ-sweep training {0.5, 0.0, 1.0, 2.0, 0.1} | ~55% | 0.5 ✓, 0.0 ✓; 2.0 finishing; 0.1 next; 1.0 redo gated (OOM casualty) |
| Harmonization eval (FID/KID, masked SSIM, domain-clf, MMD) | 40% | 2/5 arms done (0.0, 0.5); rest auto-collect |
| Downstream Dice/HD95 (hybrid + cycle-only) | ~25% | λ=0.5 re-running (segmenter [A] ep30/40); λ=0.0 queued |
| Manuscript prose (intro/related/method/setup) | 60% | written, compiles (7pp), method diagram in; results/discussion pending numbers |
| Figures/tables | 40% | generator done; preliminary comparison table+fig; need full sweep + downstream + qualitative |
| Final assembly (compile, double-blind, submit) | 0% | pending |

**Preliminary result (POSITIVE):** hybrid (λ=0.5) beats cycle-only (λ=0.0) on
FID (47.3 vs 51.8) and source-structure SSIM (0.42 vs 0.29), both driving the
domain classifier to chance. The "PatchNCE adds local content preservation
without sacrificing distribution match" mechanism thesis is holding in real numbers.

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
