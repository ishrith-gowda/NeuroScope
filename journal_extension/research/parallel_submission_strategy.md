# Parallel Submission Strategy: SA-CycleGAN-2.5D + Five Journal Extensions

**Compiled**: 2026-05-01
**Last revised**: 2026-05-06
**Project**: SA-CycleGAN-2.5D for multi-site MRI harmonization, plus a five-contribution journal extension (PatchNCE hybrid loss / neural compression-harmonization / multi-domain AdaIN / downstream task-aware harmonization / federated harmonization)
**Authors**: Ishrith Gowda et al.
**Goal**: Maximize impact factor, prestige, and citation footprint by framing each contribution toward the venue best suited for its angle, and submitting in parallel without violating dual-submission policies.

## Revision Log

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
1. Base paper (MICCAI 2026) is under review until decision day **2026-06-12**. No overlapping submission anywhere else until that date except clearly differentiated workshops and journal extensions of an *accepted* paper.
2. Conference dual-submission policies (NeurIPS, ICLR, ICML, CVPR, AAAI, ECCV, MICCAI) are strict; the same content may not be under review at two of these simultaneously. Differentiated *extensions* are permitted if the contribution is substantially new.
3. IEEE TMI / MedIA explicitly welcome conference-extension journal versions when the new content >= 30% and a cover letter explains the delta.
4. Workshops co-located with a venue (e.g., DGM4MICCAI inside MICCAI 2026) typically allow overlap with the parent venue.
5. Each individual extension (A, B, C, D, E) is distinct enough that A->ICLR and B->NeurIPS and C->BMVC etc. is *not* a dual submission of the same content.

---

## 2. Tier-by-Tier Venue Map

### Tier 1: Top medical imaging

| Venue | Type | Deadline | Format | Acceptance | Impact / h5 | Best framing | Notes |
|-------|------|----------|--------|------------|-------------|--------------|-------|
| **MICCAI 2026** | Conference | submitted; decision **2026-06-12** | 8 pages LNCS | ~30% | h5 ~106 | Base paper | Already in pipeline. |
| **IEEE TMI** | Journal (rolling) | rolling | 14 pp IEEE | ~20-25% | IF ~10.6 | **Base + Ext A + Ext D combined journal extension** | The natural home post-MICCAI; submit ~July 2026 after MICCAI decision. |
| **Medical Image Analysis (MedIA)** | Journal (rolling) | rolling | ~30 pp | ~25% | IF ~10.9 | **Ext B (compression) or Ext E (federated)** | Methods-heavy MedIA reviewers love rate-distortion stories. |
| **IEEE J. Biomed. & Health Informatics** | Journal (rolling) | rolling | ~10 pp | ~14% | IF ~7.7 | **Ext E (federated)** | JBHI welcomes privacy-preserving + federated medical informatics. |
| **NeuroImage / NeuroImage: Clinical** | Journal (rolling) | rolling | ~15 pp | ~30% | IF 5.7 / 3.4 | **Ext D (downstream)** or **Ext C (N=4 site)** | Neuroscience audience; emphasize biological interpretation. |
| **Radiology: Artificial Intelligence** | Journal (rolling) | rolling | flexible | competitive | IF ~13.2 | **Ext D (clinical impact)** | Highest-prestige clinical-AI journal; demands clinical-impact narrative. |

### Tier 2: Top ML / AI conferences

| Venue | Deadline (2026/27 cycle) | Format | Acceptance | h5 | Best framing | Notes |
|-------|---------------------------|--------|------------|------|--------------|-------|
| **NeurIPS 2026** | **abstract 2026-05-04 / paper 2026-05-06** | 9 pp + appendix | ~25% | 309 | **Ext B (compression-harmonization)** | URGENT (~3 days). Most ML-novel framing. |
| **ICLR 2027** | ~late Sept / early Oct 2026 (VERIFY) | 10 pp OpenReview | ~32% | 304 | **Ext A (PatchNCE hybrid)** | Loss-function papers with clean ablations are ICLR's bread and butter. |
| **AAAI 2027** | ~2026-08-01 (VERIFY) | 7+2 pp | ~23% | 220 | **Ext C (multi-domain AdaIN)** | Applied-ML systems audience; N x N matrix demos well. |
| **CVPR 2027** | ~early Nov 2026 (VERIFY) | 8 pp | ~25% | 440 | **Base + Ext C combined as multi-domain self-attention** | Highest-prestige CV venue; multi-domain attention is on-brand. |
| **ICML 2027** | ~late Jan 2027 (VERIFY) | 8 pp | ~27% | 289 | **Ext B (backup if NeurIPS rejects)** or **Ext E** | Core ML audience; both compression and FL fit. |
| **ICCV 2027** | ~March 2027 (VERIFY) | 8 pp | ~25% | 310 | **Ext C or Ext A backup** | Backup if CVPR rejects. |
| **ECCV 2026** | **PASSED** (deadline 2026-03-05) | -- | -- | -- | -- | Next cycle: ECCV 2028. |

### Tier 3: Specialized

| Venue | Deadline | Format | Acceptance | Best framing | Notes |
|-------|----------|--------|------------|--------------|-------|
| **ISBI 2027** | ~late Oct / early Nov 2026 (VERIFY) | 4 pp IEEE | ~50% | **Ext D (segmentation transfer short paper)** | Low-cost shot; broadens citation footprint. |
| **MIDL 2027** | ~early Dec 2026 (VERIFY); MIDL 2026 PASSED | 8-12 pp full / 3 pp short | ~40% full | **Ext E or Ext D** | Direct medical-imaging-deep-learning audience. |
| **DGM4MICCAI 2026** workshop | ~late June / early July 2026 (VERIFY) | 8 pp LNCS | ~50% | **Ext A (PatchNCE)** | Co-located w/ MICCAI 2026; deep-generative-models specific. |
| **SASHIMI 2026** workshop | ~July 1, 2026 (VERIFY) | 8 pp LNCS | ~50% | **Ext C (multi-domain translation)** | Workshop title literally matches Ext C. |
| **WACV 2027** | round 1 ~July 2026; round 2 ~Sept 2026 (VERIFY) | 8 pp | ~40% | **Ext C or D** | Application-friendly; 2-round submission gives a buffer. |
| **BMVC 2026** | **abstract 2026-05-22 / paper 2026-05-29** | 9 pp | ~30% | **Ext C (multi-domain AdaIN)** | Tight 4-week turnaround; manageable since Ext C is mostly coded. |

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
| 1 | ~~CRITICAL~~ **PASSED** | ~~NeurIPS 2026~~ | ~~abstract 2026-05-04 / paper 2026-05-06~~ | ~~Ext B (Neural Compression-Harmonization)~~ | -- | -- | **NOT SUBMITTED — slotted for ICML 2027** |
| 2 | **CRITICAL (new)** | **BMVC 2026** | abstract **2026-05-22** / paper **2026-05-29** | **Ext C (Multi-Domain AdaIN)** | 9 pp | new write-up | **PRIMARY FOCUS — draft now** |
| 3 | **HIGH** | **DGM4MICCAI 2026** workshop | ~late June 2026 (VERIFY) | **Ext A (PatchNCE Hybrid)** | 8 pp LNCS | tight extension of journal-ext Sec A | begins draft mid-May |
| 4 | **HIGH** | **SASHIMI 2026** workshop | ~July 1 2026 (VERIFY) | **Ext C** if BMVC rejects, otherwise **Ext D** | 8 pp LNCS | reuse from BMVC / journal Sec D | begins draft June |
| 5 | **HIGH** | **IEEE TMI** journal | rolling, target 2026-07 | **Base + Ext A + Ext D** combined | 14 pp | full journal extension; the master deliverable | post MICCAI decision |
| 6 | **MEDIUM** | **AAAI 2027** | ~2026-08-01 (VERIFY) | **Ext C** if not accepted at BMVC, **Ext B** otherwise | 7+2 pp | hybrid w/ minor refresh | September draft |
| 7 | **MEDIUM** | **FL@FM-NeurIPS 2026** workshop | ~mid Sept 2026 | **Ext E (Federated)** | 4-9 pp | reuse Ext E results | August draft |
| 8 | **MEDIUM** | **MedIA** journal | rolling, target Q4 2026 | **Ext E full version** post-workshop | ~30 pp | extends FL workshop paper with FedAvg/FedProx/SCAFFOLD comparison | October draft |
| 9 | **MEDIUM** | **ICLR 2027** | ~early Oct 2026 (VERIFY) | **Ext A as a general unpaired-translation method paper** | 10 pp | rewrite Ext A into method-first form, add CelebA/CMP-Facade non-medical experiments | September draft |
| 10 | **OPTIONAL** | **ISBI 2027** | ~late Oct / early Nov 2026 (VERIFY) | **Ext D short paper** | 4 pp | distill journal Sec D to 4 pages | September draft |
| 11 | **OPTIONAL** | **MIDL 2027** | ~early Dec 2026 (VERIFY) | **Ext E** if not accepted at FL workshop | 8-12 pp | reuse Ext E + JBHI-style analysis | November draft |
| 12 | **BACKUP** | **CVPR 2027** | ~early Nov 2026 (VERIFY) | **Ext C extended** with K=10 sites synthetic experiment | 8 pp | major extension of BMVC paper | October draft |
| 13 | **BACKUP** | **ICML 2027** | ~late Jan 2027 (VERIFY) | **Ext B** if NeurIPS rejects | 8 pp | re-paper if needed | December draft |
| 14 | **BACKUP** | **JBHI** | rolling | **Ext E full federated** | ~10 pp | if MedIA rejects | flexible |
| 15 | **BACKUP** | **Radiology AI** | rolling | **Ext D clinical-narrative version** | ~10 pp | if NeuroImage Clinical rejects | flexible |

**Realistic total**: 6-8 first-tier submissions, 3-4 backup submissions, 1-2 journal extensions. Expected accepts: 4-6 publications across the portfolio over the 2026-2027 cycle.

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

### Immediate (next 7 days; CRITICAL)
- **2026-05-02**: Decide whether to attempt NeurIPS 2026 Ext B abstract by 2026-05-04. Realistic only if the lambda_rate sweep shows enough finished runs by then; if not, defer Ext B to ICML 2027 (Jan 2027) and reallocate the next two weeks to BMVC Ext C.
- **2026-05-04** (NeurIPS 2026 abstract deadline AOE): submit abstract if attempting.
- **2026-05-06** (NeurIPS 2026 paper deadline AOE): submit paper if attempting.

### May 2026 (HIGH priority)
- **2026-05-08 -> 2026-05-22**: Draft BMVC 2026 Ext C paper (assume base submission). Use existing multi-domain training (60-epoch run completes in queue ~2026-05-04) and N x N matrix figure.
- **2026-05-29** (BMVC paper deadline): submit Ext C.

### June - July 2026
- **2026-06-12** (MICCAI 2026 decision): conditional on accept, begin TMI journal-extension drafting using `journal_extension/manuscript/journal_extension.tex` as scaffold. Target submission ~2026-07-15.
- **2026-06 / 07** (DGM4MICCAI 2026 deadline VERIFY): submit Ext A workshop paper.
- **2026-07-01** (SASHIMI 2026 deadline VERIFY): submit Ext D as workshop paper if not held back for BMVC.

### August - October 2026
- **2026-08-01** (AAAI 2027 paper VERIFY): submit Ext C or Ext B refined version.
- **2026-09-15** (NeurIPS workshops VERIFY): submit Ext E to FL@FM-NeurIPS, Ext D to AI4Health.
- **2026-10-01** (ICLR 2027 deadline VERIFY): submit Ext A reframed as general method paper.
- **2026-10-30** (ISBI 2027 deadline VERIFY): submit Ext D short paper.

### November 2026 - February 2027
- **2026-11** (CVPR 2027 deadline VERIFY): submit Ext C extended (with K=10 sites synthetic experiment).
- **2026-12** (MIDL 2027 deadline VERIFY): backup Ext E.
- **2027-01-25** (ICML 2027 deadline VERIFY): backup Ext B.

---

## 6. Manuscript Skeletons to Build

For each priority venue listed above, the project should maintain a separate `.tex` skeleton in `journal_extension/manuscript/` so drafts can be assembled rapidly when training results land. Recommended skeletons:

1. `neurips2026_ext_b.tex` -- Neural Compression-Harmonization for Multi-Site MRI (NeurIPS format, 9 pages). Reuses architecture diagram, rate-distortion curve figures, and the journal-extension Sec B prose.
2. `bmvc2026_ext_c.tex` -- Multi-Domain Self-Attention CycleGAN for N-Site MRI Harmonization (BMVC format, 9 pages). Reuses the N x N matrix figure and Sec C prose.
3. `dgm4miccai2026_ext_a.tex` -- PatchNCE Hybrid Loss for Cycle-Consistent Medical Translation (LNCS, 8 pages). Reuses lambda-sweep, statistical significance, and qualitative figures.
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
| NeurIPS 2026 deadline (~3 days) too tight given queue progress | HIGH | If lambda_rate sweep results don't land by 2026-05-04, defer to ICML 2027. |
| MICCAI 2026 rejection blocks TMI extension narrative | MEDIUM | TMI accepts purely-journal submissions; convert to standalone 14-page paper if rejected. |
| Reviewer flags overlap between conference and workshop versions | MEDIUM | Always submit a "differentiation cover letter" describing what is new vs. the parent paper. |
| Ext D's downstream Dice regression weakens Radiology AI angle | MEDIUM | The new task-aware loss (camera-ready queue task #4) closes this gap; ensure Dice improves before submitting to Radiology AI. |
| Federated runs do not converge to centralized SSIM in 20 rounds | LOW | If FedProx/SCAFFOLD plateau early, extend to 40 rounds; communicate honestly that 20 was budget-driven. |
| Insufficient compute for triple-seed Ext A by ICLR deadline | MEDIUM | ICLR submission can use single-seed + bootstrap intervals on the per-slice metrics; flag in limitations. |
| Cluster lease expires before queue completes (lease typically 1 week) | HIGH | Renew Chameleon lease weekly. Save checkpoints frequently. Have a contingency to migrate to a different node. |

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

## 10. Updated Critical Path (post-NeurIPS-skip, 2026-05-06)

NeurIPS 2026 has passed without submission as planned. The new critical path collapses around BMVC 2026 and the post-MICCAI TMI extension:

**Phase 1 (now -> 2026-05-29): BMVC 2026 sprint, Ext C.**
- 2026-05-08 -> 2026-05-15: assemble BMVC paper using existing multi-domain training history + N x N translation matrix from `journal_extension/figures/`. Outline: introduction, related work (StarGAN v2 / ImUnity / CALAMITI), method (AdaIN + self-attention), four-domain experiment, ablation, qualitative N x N matrix.
- 2026-05-15 -> 2026-05-22: revise. Submit abstract by 2026-05-22 23:59 UTC.
- 2026-05-22 -> 2026-05-29: final pass on figures, supplementary, anonymisation. Submit paper by 2026-05-29 23:59 UTC.

**Phase 2 (2026-05-30 -> 2026-06-30): MICCAI 2026 decision wait + workshop drafts.**
- 2026-06-12: MICCAI 2026 decision. If **accept**, immediately begin TMI journal extension drafting using `journal_extension/manuscript/journal_extension.tex` as scaffold. If **reject**, redirect Base + Ext A + Ext D into a standalone TMI / MedIA submission with 14-page format. Either way, journal extension goes out by ~2026-07-15.
- 2026-06-15 -> 2026-06-30: draft DGM4MICCAI 2026 workshop paper for Ext A (PatchNCE) targeting late-June deadline (VERIFY).
- 2026-06-15 -> 2026-07-01: draft SASHIMI 2026 workshop paper for Ext D or Ext C-backup targeting ~July 1 deadline (VERIFY).

**Phase 3 (2026-08 -> 2026-12): broader portfolio.**
- 2026-08-01: AAAI 2027 paper deadline (VERIFY); resubmit Ext C if BMVC rejected.
- 2026-09-15: NeurIPS 2026 workshops (FL@FM, AI4Health) for Ext E and Ext D.
- 2026-10-01: ICLR 2027 deadline (VERIFY); submit Ext A as a general-domain method paper with non-medical experiments added.
- 2026-10-30: ISBI 2027 short paper for Ext D.
- 2026-11: CVPR 2027 deadline (VERIFY); Ext C extended (K=10 sites).

**Phase 4 (2027-01 onward): backups.**
- 2027-01-25: ICML 2027 for Ext B (the deferred compression paper).
- Rolling: MedIA full extension of Ext E once federated workshop paper is accepted.

**Active leases / cluster status**: The Chameleon Cloud `compute_gigaio` lease is up for periodic renewal; ssh to 129.114.109.228 returned a connection timeout on 2026-05-06 19:00 UTC, indicating the lease may have expired or the node is rebooting. The camera-ready training queue (started 2026-05-01) was at FedProx round 8 of 20 as of last check. Action item: verify lease status, restart the queue if needed, and accept whatever subset of camera-ready data has landed for the BMVC 2026 sprint. **The BMVC paper does not depend on the camera-ready upgrades** — Ext C's multi-domain training is on the legacy results in `all_results.json` and the matrix figure can be rendered from any reasonable Ext C checkpoint, with seed-1 preliminary results sufficient for an initial submission and the camera-ready additions added in revision/rebuttal.

This document should be revisited weekly as deadlines firm up and the cluster queue completes.
