# Parallel Submission Strategy: SA-CycleGAN-2.5D + Five Journal Extensions

**Compiled**: 2026-05-01
**Last revised**: 2026-06-08
**Project**: SA-CycleGAN-2.5D for multi-site MRI harmonization, plus a five-contribution journal extension (PatchNCE hybrid loss / neural compression-harmonization / multi-domain AdaIN / downstream task-aware harmonization / federated harmonization)
**Authors**: Ishrith Gowda et al.
**Goal**: Maximize impact factor, prestige, and citation footprint by framing each contribution toward the venue best suited for its angle, and submitting in parallel without violating dual-submission policies.

---

## 0. Finalized Timeline (locked 2026-06-08)

Single source of truth for dates. **FIRM** = verified against the official venue site. **EST** = CFP not posted yet; historical/predicted, do not lock. All workshop/conference deadlines are submission deadlines (not event dates); event dates are listed separately at the bottom.

| Date | Venue | Deliverable | Extension | Confidence |
|------|-------|-------------|-----------|------------|
| 2026-06-08 | — | MICCAI rejected; begin cluster re-lease + journal rebuild | — | now |
| **2026-06-19** | DGM4MICCAI 2026 (workshop) | paper, 8 pp LNCS (AoE — no TZ stated) | Ext A (PatchNCE hybrid) | **FIRM** · stretch (only if §11-clean) |
| **2026-07-01** | SASHIMI 2026 (workshop) | paper, 8 pp LNCS (23:59 PT) | Ext C (multi-domain N=4) | **FIRM** |
| 2026-07-07 | DGM4MICCAI 2026 | decision (if submitted) | Ext A | FIRM |
| 2026-07-17 | DGM4MICCAI 2026 | camera-ready (if accepted) | Ext A | FIRM |
| **2026-07-21** | AAAI 2027 | abstract (11:59pm UTC-12) | Ext C (conf-grade) | **FIRM** |
| **2026-07-28** | AAAI 2027 | paper (supp/code 07-31) | Ext C (conf-grade) | **FIRM** |
| 2026-07-31 | SASHIMI 2026 | decision | Ext C | FIRM |
| **~2026-08 (target)** | IEEE TMI → NeuroImage fallback | journal submission — **PRIMARY DELIVERABLE** (must clear §11; downstream-led) | Base + Ext D + Ext C + Ext A | self-set (rolling) |
| 2026-08-21 | WACV 2027 | Round 2 paper registration | Ext D / Ext C | FIRM |
| **2026-08-28** | WACV 2027 | Round 2 paper (AoE) | Ext D (or Ext C if not at AAAI) | **FIRM** |
| ~2026-09-15 | FL@FM-NeurIPS 2026 (workshop) | paper | Ext E (federated) | EST (CFP not posted) |
| ~late Sept 2026 | ICLR 2027 | paper | Ext A (general-domain) | EST (CFP not posted) |
| ~2026-10-30 | ISBI 2027 | short paper, 4 pp | Ext D | EST (deadline "Coming Soon") |
| ~2026-11-15 | CVPR 2027 | abstract | Ext C extended (K=10 sites) | EST (not posted) |
| ~early Dec 2026 | MIDL 2027 | paper | Ext E (backup) | EST (no host/CFP yet) |
| ~late Jan 2027 | ICML 2027 | paper | Ext B (compression) | EST (CFP not posted) |
| rolling | MedIA / JBHI / Radiology AI | journal backups | Ext E full / Ext D clinical | rolling |

**Conference event dates (attend/awareness, not deadlines)**: MICCAI 2026 — Sept 27–Oct 1, Strasbourg (rejected, n/a); DGM4MICCAI + SASHIMI workshops — Oct 1 2026, Strasbourg; BMVC 2026 — Nov 23–26, Lancaster (missed, n/a); WACV 2027 — Jan 5–9, Disney Springs; AAAI 2027 — Feb 16–23, Montréal; ISBI 2027 — May 25–28, Lausanne.

---

## Revision Log

- **2026-06-08 (MICCAI 2026 EARLY REJECT — decision arrived ahead of the 06-12 notification)**: Paper 6244 (the base SA-CycleGAN-2.5D paper) was **rejected without rebuttal**. Three reviewers scored 1 (Strong Reject) / 3 (Weak Reject) / 2 (Reject); meta-reviewer recommended Reject, explicitly noting the required fixes "fall outside the scope of a MICCAI rebuttal." This **invalidates the entire "MICCAI-accept gates TMI" assumption** the prior plan was built on. **The reviews are unanimous and actionable** — see new §11. Two concerns dominate every review: (1) **novelty is incremental** — the method is read as a combination of known components (CycleGAN + self-attention + 2.5D + CBAM + spectral norm) with no methodological breakthrough; (2) **no downstream clinical validation** — MMD ↓ and domain-classifier-accuracy ↓ only prove the domains became *harder to separate*, not that disease/biological signal is preserved; reviewers demand segmentation, radiomics stability on fixed ROIs, or cross-site prediction. Secondary: theory overstated + undefined notation, incomplete ablations (only self-attention ablated), unclear tumor-aware-loss annotation requirement, no method diagram, reproducibility, unfair ComBat comparison. **Strategic consequence**: the base paper does NOT go to another top *methods conference* (it will hit the same novelty wall). It becomes a **journal submission** (IEEE TMI / NeuroImage / MedIA) re-engineered as a thoroughly-validated system where downstream validation (Ext D) is the centerpiece — a journal rewards a complete validated system over a single new mechanism. The genuinely-novel pieces (Ext B harmonize-and-compress; Ext A cycle+PatchNCE hybrid) carry the *methods-novelty* load at workshops/conferences. **Every resubmission must clear the §11 checklist before going out** — that is the real lesson of this rejection, more than any single deadline.
- **2026-06-08**: **BMVC 2026 (Ext C) was missed.** The 2026-05-06 plan slotted BMVC (abstract 2026-05-22 / paper 2026-05-29 AOE, no extensions) as the CRITICAL primary near-term target, but no draft was started and nothing was submitted. The camera-ready compute queue (triple-seed Ext A, λ_rate sweep Ext B, FedProx/SCAFFOLD Ext E, task-aware loss Ext D, N=4 matrix Ext C) also did not land — `experiments/` is empty and the cluster lease appears to have lapsed (ssh was already timing out on 2026-05-06). No manuscript skeletons (§6) were built. **Impact is contained**: Ext C's exact content re-homes to SASHIMI 2026 (paper 2026-07-01) and AAAI 2027 (paper 2026-07-27), so no contribution is orphaned. **New critical path** collapses around three things, in order: (1) **DGM4MICCAI 2026** (paper **2026-06-19**, 8 pp LNCS, co-located w/ MICCAI) for **Ext A (PatchNCE)** — the most complete extension and now the nearest actionable deadline; (2) the **MICCAI 2026 decision (2026-06-12)** which gates the TMI journal extension; (3) **SASHIMI 2026** (paper **2026-07-01**) for **Ext C** — the venue topic literally *is* synthesis in medical imaging. BMVC's next cycle is ~May 2027 if Ext C is still unplaced after AAAI. All workshop/conference deadlines below have now been verified against official sites (2026-06-08).
- **2026-06-08 (verification audit)**: Every date in this document was independently re-checked against official venue sites (three parallel research passes + direct page fetches). Results:
  - **CONFIRMED from official sites**: MICCAI 2026 decision 06-12 / conf 09-27..10-01 Strasbourg; DGM4MICCAI 2026 paper **06-19** (note: official page states NO timezone — treat as AoE), reviews 07-03, decision 07-07, cam-ready 07-17, workshop 10-01; SASHIMI 2026 paper **07-01** 23:59 PT, notif 07-31; BMVC 2026 abstract **05-22** / paper **05-29** AoE (PASSED — these are the real main-track dates, directly verified on the official dates page; the "workshop proposal" deadline is a separate 06-05); NeurIPS 2026 abstract 05-04 / paper 05-06 AoE (PASSED); AAAI 2027 abstract **07-21** / paper **07-28** / supp 07-31 (11:59pm UTC-12); WACV 2027 two-round R1 paper **06-26** / R2 paper **08-28** AoE, conf 01-05..09 Disney Springs; ECCV 2026 paper 03-05 (PASSED; next cycle ECCV 2028); IEEE TMI + MedIA rolling/no deadline.
  - **CORRECTED in this pass**: AAAI was 07-20/07-27 -> now **07-21/07-28**; WACV was a single "09-01 abstract" -> now the real **two-round 06-26 / 08-28**; ICLR 2027 location was wrongly listed as "Sydney" (that is NeurIPS 2026) -> official only gives region "West Coast North America"; CVPR 2027 dates softened to official "Jun 20-24"; DGM4MICCAI timezone claim ("11:59pm PT") removed as unverified.
  - **NOT YET POSTED (deadlines are historical estimates only, do not lock)**: ICLR 2027 (region "West Coast North America"), ICML 2027 (region "South America"), CVPR 2027 deadline (~Nov 15 predicted), ISBI 2027 paper deadline ("Coming Soon"; conf 05-25..28 Lausanne), MIDL 2027 (no host/venue/CFP yet; target early July 2027 Europe), and all NeurIPS 2026 workshops (FL@FM, ML4H/AI4Health, neural-compression — CFPs typically appear ~August after workshop acceptance).
- **2026-05-06**: NeurIPS 2026 (Ext B) deadline (2026-05-06 AOE) passed without submission. Ext B is now slotted for **ICML 2027** (deadline late January 2027) as the next-tier ML conference, with **MedIA** (rolling, IF 10.9) as the parallel journal track. The freed May calendar consolidates around BMVC 2026 (Ext C, paper due 2026-05-29) as the highest-priority near-term target.

---

## 1. Strategic Framing

The project has six independently-publishable units. Each can be re-framed for a different reviewer audience by emphasizing a different methodological angle:

| ID | Unit | Methods angle | Application angle | Best audience |
|----|------|---------------|-------------------|---------------|
| 0 | Base SA-CycleGAN-2.5D (MICCAI 2026 submission) | self-attention CycleGAN, 2.5D context | scanner harmonization, glioma | medical imaging |
| A | PatchNCE Hybrid Loss | cycle + contrastive hybrid; lambda sweep | unpaired translation | ML loss-design conferences (ICLR / NeurIPS / DGM4MICCAI) |
| B | Neural Compression-Harmonization | Balle hyperprior + harmonization joint | bandwidth-constrained federated MR | NeurIPS / ICML / MedIA |
| C | Multi-Domain AdaIN (N=4) | StarGAN v2-style style codes | multi-site translation matrix | CVPR / AAAI / BMVC / SASHIMI |
| D | Downstream Task-Aware | mask-weighted cycle loss; segmentation transfer | clinical Dice transferability | TMI / Radiology AI / NeuroImage / ISBI |
| E | Federated Generative Harmonization | FedAvg / FedProx / SCAFFOLD comparison | privacy-preserving multi-site training | JBHI / FL@FM-NeurIPS / MedIA |

**Anti-overlap rules** every co-submission must satisfy:
1. ~~Base paper (MICCAI 2026) is under review until decision day 2026-06-12.~~ **MICCAI 2026 rejected the base paper (early reject, 2026-06-08).** The dual-submission hold is lifted — the base content is now free to go anywhere. The constraint is no longer policy but *quality*: nothing resubmits until it clears the §11 reviewer-driven checklist.
2. Conference dual-submission policies (NeurIPS, ICLR, ICML, CVPR, AAAI, ECCV, MICCAI) are strict; the same content may not be under review at two of these simultaneously. Differentiated *extensions* are permitted if the contribution is substantially new.
3. IEEE TMI / MedIA explicitly welcome conference-extension journal versions when the new content >= 30% and a cover letter explains the delta.
4. Workshops co-located with a venue (e.g., DGM4MICCAI inside MICCAI 2026) typically allow overlap with the parent venue.
5. Each individual extension (A, B, C, D, E) is distinct enough that A->ICLR and B->NeurIPS and C->BMVC etc. is *not* a dual submission of the same content.

---

## 2. Tier-by-Tier Venue Map

### Tier 1: Top medical imaging

| Venue | Type | Deadline | Format | Acceptance | Impact / h5 | Best framing | Notes |
|-------|------|----------|--------|------------|-------------|--------------|-------|
| **MICCAI 2026** | Conference | **REJECTED (early reject, 2026-06-08; scores 1/3/2, no rebuttal)** | 8 pages LNCS | ~30% | h5 ~106 | ~~Base paper~~ | Done. Lessons captured in §11; base content moves to the journal track below. |
| **IEEE TMI** | Journal (rolling) | rolling | 14 pp IEEE | ~20-25% | IF ~10.6 | **PRIMARY DELIVERABLE: Base + Ext D (downstream) + Ext C + Ext A, re-engineered against the MICCAI reviews** | Now the master target, no longer gated on a MICCAI accept. Lead with downstream validation (the #1 reviewer demand). Must clear §11. NeuroImage is the fallback if TMI bounces on novelty. |
| **Medical Image Analysis (MedIA)** | Journal (rolling) | rolling | ~30 pp | ~25% | IF ~10.9 | **Ext B (compression) or Ext E (federated)** | Methods-heavy MedIA reviewers love rate-distortion stories. |
| **IEEE J. Biomed. & Health Informatics** | Journal (rolling) | rolling | ~10 pp | ~14% | IF ~7.7 | **Ext E (federated)** | JBHI welcomes privacy-preserving + federated medical informatics. |
| **NeuroImage / NeuroImage: Clinical** | Journal (rolling) | rolling | ~15 pp | ~30% | IF 5.7 / 3.4 | **Ext D (downstream)** or **Ext C (N=4 site)** | Neuroscience audience; emphasize biological interpretation. |
| **Radiology: Artificial Intelligence** | Journal (rolling) | rolling | flexible | competitive | IF ~13.2 | **Ext D (clinical impact)** | Highest-prestige clinical-AI journal; demands clinical-impact narrative. |

### Tier 2: Top ML / AI conferences

| Venue | Deadline (2026/27 cycle) | Format | Acceptance | h5 | Best framing | Notes |
|-------|---------------------------|--------|------------|------|--------------|-------|
| **NeurIPS 2026** | **PASSED** (abstract 2026-05-04 / paper 2026-05-06) | -- | -- | 309 | ~~Ext B~~ | Not submitted. Ext B deferred to ICML 2027 / MedIA. |
| **AAAI 2027** | **abstract 2026-07-21 / paper 2026-07-28 / supp 2026-07-31** (CONFIRMED, official) | 7+2 pp | ~23% | 220 | **Ext C (multi-domain AdaIN)** | 11:59pm UTC-12 (AoE). Two-phase review (Phase-1 rejects 2026-09-24). Montréal, 2027-02-16..23. Ext C's strongest *conference* home post-BMVC-miss. |
| **WACV 2027** | **two-round: R1 paper 2026-06-26 / R2 paper 2026-08-28** (CONFIRMED, official; AoE) | 8 pp | ~40% | -- | **Ext C or Ext D** | R2 registration 2026-08-21. Disney Springs, 2027-01-05..09. Two-round system gives a buffer; target Round 2. |
| **ICLR 2027** | CFP **NOT posted**; est. ~late Sept 2026 (historical only) | 10 pp OpenReview | ~32% | 304 | **Ext A (PatchNCE hybrid, general-domain rewrite)** | Official lists only region "West Coast North America"; city/dates TBA. (Earlier "Sydney" was wrong — that is NeurIPS 2026.) Verify CFP when posted. |
| **CVPR 2027** | abstract ~2026-11-15 (**PREDICTED, not posted**) | 8 pp | ~25% | 440 | **Base + Ext C extended (K=10 sites)** | No deadline posted yet. Seattle, ~Jun 2027 (official CVF lists Jun 20-24). |
| **ICML 2027** | ~late Jan 2027 (**PREDICTED; CFP not posted**) | 8 pp | ~27% | 289 | **Ext B (deferred compression paper)** or **Ext E** | Official lists only region "South America"; city/dates TBA. Core ML audience. |
| **ICCV 2027** | ~March 2027 (VERIFY) | 8 pp | ~25% | 310 | **Ext C or Ext A backup** | Backup if CVPR rejects. |
| **ECCV 2026** | **PASSED** (deadline 2026-03-05) | -- | -- | -- | -- | Next cycle: ECCV 2028. |

### Tier 3: Specialized

| Venue | Deadline | Format | Acceptance | Best framing | Notes |
|-------|----------|--------|------------|--------------|-------|
| **DGM4MICCAI 2026** workshop | **paper 2026-06-19** (confirmed); notif 2026-07-07; cam-ready 2026-07-17 | 8 pp LNCS | ~50% | **Ext A (PatchNCE)** | **NEAREST DEADLINE — primary focus.** OpenReview, double-blind. Workshop 2026-10-01, Strasbourg. Deep-generative-models specific; Ext A is the most complete extension. |
| **SASHIMI 2026** workshop | **paper 2026-07-01** (confirmed); notif 2026-07-31 | 8 pp LNCS | ~50% | **Ext C (multi-domain translation)** | Workshop title literally matches Ext C — the natural re-home for the missed BMVC content. Workshop 2026-10-01, Strasbourg. |
| **ISBI 2027** | paper deadline **not yet posted** ("Coming Soon"); est. late Oct/Nov 2026 | 4 pp IEEE | ~50% | **Ext D (segmentation transfer short paper)** | Conf 2027-05-25..28, Lausanne. Low-cost shot; broadens citation footprint. |
| **MIDL 2027** | **no host/CFP yet** (board still selecting organizers); target early July 2027, Europe | 8-12 pp full / 3 pp short | ~40% full | **Ext E or Ext D** | Deadline TBD — historically ~early Dec of prior year. |
| **BMVC 2026** | **PASSED** (abstract 2026-05-22 / paper 2026-05-29 AOE) | -- | -- | ~~Ext C~~ | **MISSED — no draft started.** No extensions were granted. Ext C re-homed to SASHIMI 2026 / AAAI 2027. Next cycle BMVC 2027 (~May 2027). |

### Tier 4: Cross-cutting workshops

| Venue | Deadline | Best framing | Notes |
|-------|----------|--------------|-------|
| **FL@FM-NeurIPS 2026** | ~mid Sept 2026 | **Ext E (federated harmonization)** | Canonical FL workshop. |
| **NeurIPS 2026 Med Imaging / AI4Health workshops** | ~mid Sept 2026 | **Ext D** | Backup for clinical angle if main NeurIPS rejects. |
| **DistShift / Domain Generalization @ NeurIPS 2026** | ~mid Sept 2026 (VERIFY) | **Ext C (cross-scanner generalization)** | Frame as DG benchmark. |
| **Neural Compression Workshop @ NeurIPS or ICML** | ~mid Sept 2026 (VERIFY) | **Ext B** | Direct fit for joint compression-harmonization. |
| **MedNeurIPS / ML4H 2026** | ~Oct 2026 (VERIFY) | **Ext D or Ext E** | Backup workshop venues. |

---

## 3. Recommended Parallel-Submission Plan (Priority Ordered)

| # | Priority | Venue | Deadline | Framing | Page budget | Reuse % | Status |
|---|----------|-------|----------|---------|-------------|---------|--------|
| 1 | **NEAR-TERM (stretch)** | **DGM4MICCAI 2026** workshop | paper **2026-06-19** | **Ext A (PatchNCE Hybrid)** | 8 pp LNCS | λ=0.5 results + figures done; must add ≥1 downstream result per §11 | submit only if a clean §11-compliant draft is ready by 06-19; otherwise slip to SASHIMI |
| 2 | **HIGH** | **SASHIMI 2026** workshop | paper **2026-07-01** (23 days) | **Ext C (Multi-Domain AdaIN)** — the missed-BMVC content | 8 pp LNCS | reuse existing multi-domain training + N x N matrix figure | **draft starts 2026-06-20**, immediately after DGM4MICCAI |
| 3 | **HIGHEST-VALUE** | **IEEE TMI** journal (NeuroImage fallback) | rolling, target ~Aug 2026 | **Base + Ext D (downstream, led) + Ext C + Ext A**, re-engineered vs MICCAI reviews | 14 pp | the master deliverable; no longer gated on MICCAI (rejected). Must clear §11 — esp. positive downstream result via task-aware loss | needs cluster re-lease for ablations + task-aware loss |
| 4 | **HIGH** | **AAAI 2027** | abstract **2026-07-21** / paper **2026-07-28** | **Ext C** (conference-grade version of the SASHIMI paper) | 7+2 pp | upgrade SASHIMI Ext C with stronger baselines | abstract by 07-21 |
| 5 | **MEDIUM** | **FL@FM-NeurIPS 2026** workshop | ~mid Sept 2026 (VERIFY) | **Ext E (Federated)** | 4-9 pp | reuse Ext E results | August draft |
| 6 | **MEDIUM** | **WACV 2027** | R1 paper **2026-06-26** / R2 paper **2026-08-28** | **Ext D** (downstream/clinical) or **Ext C** if not at AAAI | 8 pp | application-friendly buffer venue; target Round 2 | August draft |
| 7 | **MEDIUM** | **MedIA** journal | rolling, target Q4 2026 | **Ext E full version** post-workshop | ~30 pp | extends FL workshop paper with FedAvg/FedProx/SCAFFOLD comparison | October draft |
| 8 | **MEDIUM** | **ICLR 2027** | est. ~late Sept 2026 (CFP TBA) | **Ext A as a general unpaired-translation method paper** | 10 pp | rewrite Ext A into method-first form, add CelebA/CMP-Facade non-medical experiments | September draft |
| 9 | **OPTIONAL** | **ISBI 2027** | ~late Oct / early Nov 2026 (VERIFY) | **Ext D short paper** | 4 pp | distill journal Sec D to 4 pages | September draft |
| 10 | **OPTIONAL** | **MIDL 2027** | ~early Dec 2026 (VERIFY) | **Ext E** if not accepted at FL workshop | 8-12 pp | reuse Ext E + JBHI-style analysis | November draft |
| 11 | **BACKUP** | **CVPR 2027** | abstract ~**2026-11-15** | **Ext C extended** with K=10 sites synthetic experiment | 8 pp | major extension of the AAAI/SASHIMI Ext C paper | October draft |
| 12 | **BACKUP** | **ICML 2027** | ~late Jan 2027 (VERIFY) | **Ext B** (deferred compression paper) | 8 pp | the long-deferred Ext B; first real submission attempt | December draft |
| 13 | **BACKUP** | **JBHI** | rolling | **Ext E full federated** | ~10 pp | if MedIA rejects | flexible |
| 14 | **BACKUP** | **Radiology AI** | rolling | **Ext D clinical-narrative version** | ~10 pp | if NeuroImage Clinical rejects | flexible |

**Realistic total**: 6-8 first-tier submissions, 3-4 backup submissions, 1-2 journal extensions. Expected accepts: 4-6 publications across the portfolio over the 2026-2027 cycle. **Note (2026-06-08): the BMVC miss removed one conference attempt but orphaned no contribution** — Ext C still has SASHIMI 2026, AAAI 2027, WACV 2027, and CVPR 2027 as live homes.

---

## 4. Conflict Matrix (Avoiding Dual-Submission Violations)

| Item | Cannot overlap with | Reason |
|------|--------------------|---------------------------|
| Base paper (MICCAI 2026) | Anywhere else until 2026-06-12 | Active dual-submission risk |
| TMI journal (Base + A + D) | NeurIPS / ICLR / ICML versions of A | Substantial overlap requires staggering; submit TMI *after* extension A appears at a workshop |
| NeurIPS Ext B | ICML / ICLR / MedIA versions of Ext B | Pick NeurIPS first; on reject, pivot to ICML 2027; on ICML reject, MedIA. Sequential, not parallel. |
| BMVC Ext C | AAAI / CVPR / ICCV / SASHIMI versions of Ext C | Sequential. BMVC -> if reject, AAAI; if accept, expand for CVPR with new K=10 experiment to differentiate. |
| ICLR Ext A (general) | TMI Ext A (medical-only) | OK if framed differently: ICLR = general method, TMI = medical specialization |
| FL@FM-NeurIPS Ext E | MedIA Ext E / JBHI Ext E | Workshops normally allow journal extension. Sequence: workshop first, then journal extension in Q4. |
| ISBI Ext D | TMI Ext D | OK: ISBI 4-pp short paper does not preclude TMI long-form journal version. |

**Master rule**: Each conference-quality extension goes to *one* main conference at a time. Workshops and journals are stacked sequentially.

---

## 5. Action Items by Date

### Immediate (now 2026-06-08 -> 2026-06-19; CRITICAL)
- **2026-06-08** (today): Stand up `dgm4miccai2026_ext_a.tex` (LNCS, 8 pp). Ext A is the most complete extension — λ=0.5 results, ablation table, statistical tests, and qualitative figures already exist. This is purely a write-up sprint, not a compute sprint.
- **2026-06-08 -> 2026-06-12**: Draft intro / related work (CUT, DCLGAN, MICCAI base) / method (cycle + PatchNCE hybrid) / experiments using on-disk results. Anonymize for double-blind.
- **2026-06-12** (MICCAI 2026 decision): record outcome. If accept, Ext A's DGM4MICCAI paper cites the accepted base paper; if reject, frame Ext A as standalone. Decision also unlocks the TMI plan below.
- **2026-06-13 -> 2026-06-18**: figures, supplementary, internal review pass.
- **2026-06-19** (DGM4MICCAI 2026 paper deadline; timezone not stated on the official page — treat as AoE to be safe): **submit Ext A.**

### Late June - July 2026 (HIGH priority)
- **2026-06-20 -> 2026-06-30**: Draft SASHIMI 2026 Ext C paper (the re-homed BMVC content). Use existing multi-domain training history + N x N translation-matrix figure from `journal_extension/figures/`. Outline: intro, related work (StarGAN v2 / ImUnity / CALAMITI / HACA3), method (AdaIN + self-attention), four-domain experiment, ablation, qualitative N x N matrix.
- **2026-07-01** (SASHIMI 2026 paper deadline, 23:59 PT): **submit Ext C.**
- **2026-06-13 onward, conditional on MICCAI accept** (decision 2026-06-12): begin TMI journal extension using `journal_extension/manuscript/journal_extension.tex` as scaffold. Target submission ~2026-07-31 (slipped from the original ~07-15 to clear the two workshop deadlines first). If MICCAI rejects, convert Base + Ext A + Ext D into a standalone 14-page TMI/MedIA paper.
- **2026-07-21** (AAAI 2027 abstract, 11:59pm UTC-12): register Ext C conference-grade abstract.
- **2026-07-28** (AAAI 2027 paper; supp/code due 07-31): **submit Ext C** upgraded from the SASHIMI version with stronger baselines.

### August - October 2026
- **2026-08-28** (WACV 2027 Round 2 paper; register by 2026-08-21, AoE): submit Ext D (or Ext C if not placed at AAAI).
- **~2026-09-15** (FL@FM-NeurIPS / AI4Health workshops, VERIFY): submit Ext E to FL@FM, Ext D to AI4Health.
- **~late Sept 2026** (ICLR 2027 deadline, CFP TBA): submit Ext A reframed as a general-domain method paper with non-medical experiments.
- **~2026-10-30** (ISBI 2027 deadline VERIFY): submit Ext D short paper.

### November 2026 - February 2027
- **~2026-11-15** (CVPR 2027 abstract): submit Ext C extended (K=10 sites synthetic experiment).
- **~2026-12** (MIDL 2027 deadline VERIFY): backup Ext E.
- **~2027-01-25** (ICML 2027 deadline VERIFY): submit the deferred Ext B compression paper.

---

## 6. Manuscript Skeletons to Build

For each priority venue listed above, the project should maintain a separate `.tex` skeleton in `journal_extension/manuscript/` so drafts can be assembled rapidly when training results land. Recommended skeletons:

1. **`dgm4miccai2026_ext_a.tex` (BUILD FIRST, due 2026-06-19)** -- PatchNCE Hybrid Loss for Cycle-Consistent Medical Translation (LNCS, 8 pages). Reuses lambda-sweep, statistical significance, and qualitative figures. Most complete extension; the current critical path.
2. **`sashimi2026_ext_c.tex` (BUILD SECOND, due 2026-07-01)** -- Multi-Domain Self-Attention CycleGAN for N-Site MRI Harmonization (LNCS, 8 pages). Reuses the N x N matrix figure and Sec C prose. This is the re-home for the missed `bmvc2026_ext_c.tex`; AAAI 2027 (2026-07-27) takes a conference-grade upgrade of it.
3. `neurips2026_ext_b.tex` -- Neural Compression-Harmonization for Multi-Site MRI. NeurIPS was missed; repurpose this skeleton for ICML 2027 (~Jan 2027) format, 8 pages. Reuses architecture diagram, rate-distortion curve figures, and the journal-extension Sec B prose. Blocked on the Ext B λ_rate sweep landing.
4. `tmi_journal_extension.tex` -- the existing `journal_extension.tex` is already this skeleton; tighten cover letter and overlap analysis when MICCAI decision lands.
5. `iclr2027_ext_a_general.tex` -- a general-domain rewrite of Ext A with non-medical experiments added (CelebA, CMP-Facade). Defer to August.
6. `flnips2026_ext_e.tex` -- federated workshop paper, 4-9 pp.
7. `medi_ext_e_full.tex` -- full MedIA paper extending the federated workshop.
8. `aaai2027_ext_c.tex` -- AAAI-format reframe.
9. `cvpr2027_ext_c_K10.tex` -- CVPR extension with K=10 sites.

---

## 7. Risk Register

| Risk | Probability | Mitigation |
|------|-------------|------------|
| ~~NeurIPS 2026 deadline too tight~~ + ~~BMVC 2026 sprint not executed~~ | **MATERIALIZED** | Both passed without submission. Recovery: Ext B -> ICML 2027; Ext C -> SASHIMI 2026 / AAAI 2027. Root cause was no draft-start discipline — mitigate going forward by standing up the `.tex` skeleton on day one of each sprint (Phase 1 starts 2026-06-08). |
| Workshop write-up sprints slip the same way BMVC did | HIGH | DGM4MICCAI (2026-06-19) and SASHIMI (2026-07-01) are write-up-only (no compute). Treat the skeleton-on-day-one rule as non-negotiable; both reuse already-generated figures/results. |
| MICCAI 2026 rejection blocks TMI extension narrative | MEDIUM | TMI accepts purely-journal submissions; convert to standalone 14-page paper if rejected. |
| Reviewer flags overlap between conference and workshop versions | MEDIUM | Always submit a "differentiation cover letter" describing what is new vs. the parent paper. |
| Ext D's downstream Dice regression weakens Radiology AI angle | MEDIUM | The new task-aware loss (camera-ready queue task #4) closes this gap; ensure Dice improves before submitting to Radiology AI. |
| Federated runs do not converge to centralized SSIM in 20 rounds | LOW | If FedProx/SCAFFOLD plateau early, extend to 40 rounds; communicate honestly that 20 was budget-driven. |
| Insufficient compute for triple-seed Ext A by ICLR deadline | MEDIUM | ICLR submission can use single-seed + bootstrap intervals on the per-slice metrics; flag in limitations. |
| Cluster lease expired before camera-ready queue completed | **MATERIALIZED** | Lease lapsed ~2026-05-06; `experiments/` is empty. Workshop sprints (Phase 1-2) do not need it. Re-lease in parallel to produce camera-ready data (triple-seed Ext A, Ext B RD curve, FedProx/SCAFFOLD, task-aware Ext D) for the TMI journal extension and AAAI upgrade. Save checkpoints frequently; renew weekly. |

---

## 8. Author Contributions and Authorship Order

For each submission, the canonical authorship line is:

> Ishrith Gowda, Chunwei Liu

with contribution split:
- Ishrith Gowda: implementation, experiments, analysis, manuscript drafting
- Chunwei Liu: research direction, methodology guidance, manuscript review

For workshop / shorter venues this is unchanged. For TMI / MedIA the authorship may expand if additional contributors join during the camera-ready window (e.g., clinical co-authors for Radiology AI submission).

---

## 9. Reproducibility Commitments

Each submission must include:
- A `code/` zip or GitHub link with the exact `journal_extension/scripts/` snapshot used to produce the figures.
- The `regenerate_all_artifacts.sh` orchestration script.
- All training history JSON files for the runs reported.
- Test-set evaluation JSON blobs.
- The literature-review notes at `journal_extension/research/literature_review.md` should be cited as the basis for the related-work section.

---

## 10. Updated Critical Path (post-BMVC-miss, 2026-06-08)

BMVC 2026 (Ext C) was missed — no draft was started and the paper deadline (2026-05-29) passed. Nothing was submitted since the NeurIPS skip, and the camera-ready compute queue never landed. The critical path now collapses around the **two MICCAI-co-located workshops** (whose deadlines are real and imminent) plus the post-MICCAI TMI extension. Ext C is not lost: it moves from BMVC to SASHIMI 2026 / AAAI 2027.

**Phase 1 (now -> 2026-06-19): DGM4MICCAI 2026 sprint, Ext A.**
- 2026-06-08 -> 2026-06-12: assemble the LNCS 8-page paper from on-disk Ext A artifacts (λ=0.5 wins all test metrics; ablation, Wilcoxon+Bonferroni+Cohen's d, qualitative FLAIR/T1/T1ce/T2 figures all already generated). Outline: intro, related work (CUT / DCLGAN / SA-CycleGAN base), method (cycle + PatchNCE hybrid), λ sweep, statistical significance, qualitative results. **This is a write-up sprint, not a compute sprint** — no cluster dependency.
- 2026-06-12: MICCAI 2026 decision lands; adjust framing (cite accepted base vs. standalone).
- 2026-06-13 -> 2026-06-18: figures, supplementary, double-blind anonymization, internal review.
- 2026-06-19: submit Ext A (timezone unstated on official page — treat as AoE).

**Phase 2 (2026-06-20 -> 2026-07-01): SASHIMI 2026 sprint, Ext C (the re-homed BMVC content).**
- 2026-06-20 -> 2026-06-27: assemble the LNCS 8-page paper from existing multi-domain training history + N x N translation-matrix figure. Outline: intro, related work (StarGAN v2 / ImUnity / CALAMITI / HACA3), method (AdaIN + self-attention), four-domain experiment, ablation, qualitative N x N matrix.
- 2026-06-28 -> 2026-07-01: revise, figures, supplementary. Submit by 2026-07-01 23:59 PT.

**Phase 3 (2026-06-13 -> ~2026-08, the highest-value track): standalone TMI/NeuroImage journal submission, re-engineered against the MICCAI reviews.**
- MICCAI rejected the base paper (2026-06-08), so this is now an unconditional standalone journal submission, not a conference extension. Use `journal_extension/manuscript/journal_extension.tex` as scaffold but **rebuild around the §11 checklist**: lead with downstream validation (Ext D), add the full ablation grid, add a method diagram, trim the theory, add fair image-space baselines, do subject-level eval.
- Gating dependency: re-lease the cluster, run the **task-aware loss** (to convert Ext D's current Dice regression into a positive/neutral result) + the missing ablations. Do NOT submit until the downstream result is non-embarrassing. Realistic target ~late July / August 2026.

**Phase 4 (2026-07 -> 2026-12): broader portfolio.**
- 2026-07-21 / 28: AAAI 2027 abstract / paper (supp 07-31); Ext C conference-grade upgrade of the SASHIMI paper.
- 2026-08-28: WACV 2027 Round 2 paper (register 08-21); Ext D or Ext C.
- ~2026-09-15: FL@FM-NeurIPS / AI4Health workshops for Ext E and Ext D (VERIFY).
- ~late Sept 2026: ICLR 2027 (CFP TBA); Ext A general-domain method paper with non-medical experiments.
- ~2026-10-30: ISBI 2027 short paper for Ext D.
- ~2026-11-15: CVPR 2027 abstract; Ext C extended (K=10 sites).

**Phase 5 (2027-01 onward): backups.**
- ~2027-01-25: ICML 2027 for the deferred Ext B compression paper.
- Rolling: MedIA full extension of Ext E once a federated workshop paper is accepted.

**Active leases / cluster status (as of 2026-06-08)**: The camera-ready compute queue (started 2026-05-01) never produced results — `journal_extension/experiments/` is empty and ssh to the `compute_gigaio` node was already timing out on 2026-05-06, so the lease has almost certainly lapsed. **The Phase 1-2 sprints do not depend on the camera-ready upgrades**: both Ext A (DGM4MICCAI) and Ext C (SASHIMI) build entirely on the legacy single-seed results already in `journal_extension/results/` and the figures already in `journal_extension/figures/`. The camera-ready additions (triple-seed Ext A error bars, full Ext B RD curve, FedProx/SCAFFOLD for Ext E, task-aware loss for Ext D) are needed for the **TMI journal extension and the AAAI conference upgrade**, not the workshop submissions — so a cluster re-lease should be scheduled to run in parallel during the Phase 1-2 write-up sprints, not block them.

This document should be revisited weekly as deadlines firm up and the cluster queue completes.

---

## 11. MICCAI 2026 Reviewer-Driven Resubmission Checklist (gating requirement for ALL future submissions)

The MICCAI 2026 early reject (paper 6244; scores 1/3/2; meta = Reject) produced a clear, unanimous diagnosis. No base-paper-derived submission should go out until it clears the items below. Mapped to the extensions we already have, most of these are *addressable with existing or near-term work* — the rejection is essentially a spec for the journal extension.

### Blocking concerns (every reviewer + meta)

| # | Concern (verbatim sense) | Raised by | Fix | Where it lives |
|---|--------------------------|-----------|-----|----------------|
| 1 | **No downstream clinical validation.** Lower MMD / domain-classifier accuracy only shows domain *mixing*, not that disease/biological signal is preserved. Need segmentation Dice/HD95, radiomics ICC on **fixed ROIs**, or cross-site prediction, before vs after harmonization. | R1, R2, R3, meta | This is **Extension D** — it is the single most important addition and the headline of any resubmission. Must show harmonization *helps* (or at least preserves) a real task. Note: current Ext D found a Dice **regression** (A→B −19%, B→A −11%); the **task-aware loss** (camera-ready item) is required to turn this into a positive/neutral result *before* submitting, otherwise it confirms the reviewers' fear. | `journal_extension` Ext D + task-aware loss |
| 2 | **Incremental novelty** — assembled from known components, no methodological breakthrough. | R1, R2, R3, meta | Two responses: (a) for **journals**, reframe as a *complete validated harmonization system*, where thoroughness > single new mechanism; (b) for **methods venues**, lead with the genuinely-novel pieces — **Ext B (joint harmonize-and-compress)** is the strongest novelty claim, **Ext A (cycle+PatchNCE hybrid)** second. Do not pitch the bare base method to another top methods conference. | Ext B / Ext A framing |
| 3 | **Incomplete ablations** — only self-attention ablated; nothing for 2.5D context, CBAM, spectral norm, tumor-aware loss. | R2, meta | Run the full ablation grid (camera-ready compute). Each claimed component needs an ablation row + training curves. | needs cluster re-lease |
| 4 | **Theory overstated and under-specified** — "preserve tumor pathophysiology / clinical fidelity / anatomical morphology" unsupported; undefined notation in Sec 2.1. | R1, R2, R3 | Either **cut** the domain-adaptation-bound theory to a short honest motivation, or make it rigorous and *tightly* tied to the implementation. Define every symbol. Drop unsupported clinical-claim adjectives. | manuscript rewrite |
| 5 | **Tumor-aware loss provenance unclear** — are tumor annotations needed at train/inference? If manual labels are used, state it as a limitation. | R2, meta | Document the mask source explicitly; if labels are train-only, say so; quantify sensitivity to mask quality. | manuscript + Ext D |

### Secondary (cheap, do all of them)

- **Add a key method/architecture diagram** (R1: "no key diagram to illustrate the methodology"). Non-negotiable.
- **Tighten the Methods section** — reviewers found it verbose/dense and hard to read (R1, R2).
- **Subject-level (not slice-level) evaluation**, and reconcile the sample-count discrepancies across tables (R2, R1).
- **Fair baseline comparison** — the ComBat comparison was called "unfair" (operates in a different space); add image-space harmonization baselines (e.g., a plain CycleGAN, an ImUnity/CALAMITI/HACA3-style method) (R1).
- **Reproducibility** — release code + config, or give a complete enough algorithmic spec (R1, R2 both said insufficient).
- **One-way harmonization quality** — cycle metrics (ABA/BAB) can't certify A→B quality (R3); add a direct one-way evaluation.

### What this means for sequencing

- **DGM4MICCAI 2026 (Ext A, 06-19)** and **SASHIMI 2026 (Ext C, 07-01)** are workshops (~50% accept, more forgiving on novelty) — they can go out as focused single-contribution papers, but each should still carry **at least one downstream/validation result** so it doesn't repeat the MICCAI failure mode. If a credible Ext A workshop paper cannot be assembled honestly by 06-19, **do not force it** — slip to SASHIMI and the journal rather than burn another rushed submission.
- **The journal extension (TMI/NeuroImage, primary)** is where the full §11 checklist gets satisfied. It is no longer gated on a MICCAI accept and is the highest-value deliverable. Realistic target: cluster re-lease → finish task-aware loss + ablations + downstream → submit ~late July / August 2026.
