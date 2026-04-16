# MICCAI 2026 Review — Paper 2038

**Title:** An Artifact-based Agent Framework for Adaptive and Reproducible Medical Image Processing

---

## Q1. Reviewer Guide Confirmation
I agree. I have read and understood the MICCAI 2026 Reviewers' Guide and the Confidentiality section; no part of the paper has been shown to an LLM.

## Q2. Public Release Consent
I agree.

## Q3. Relevance Category
- MIC (Medical Image Computing)
- Clinical Translation of Methodology

## Q4. Paper Type
Methodological contribution (framework/systems paper)

## Q5. Main Contribution

The paper proposes an artifact contract-based agent framework that inserts a semantic planning layer above a deterministic medical image processing workflow engine (Snakemake), targeting the tension between dataset-aware adaptability and reproducible execution in real clinical deployments. The core idea is the "artifact contract": workflow outputs are represented as typed tuples of (type, structured attributes, provenance), giving the LLM agent a machine-readable state to reason over instead of raw files, and simultaneously recording provenance for auditing. The agent uses this structured state for two operations — assembling a configuration C = (π, θ) by selecting rules from a modular rule library conditioned on the artifact set and the analytical goal, and answering natural-language queries over contract-compliant artifacts — while execution remains deterministic. The framework is validated across three clinical cohorts (NLST, LungCaTrial, BrainICU) spanning CT and MRI with varied heterogeneity, on nine analytical goals (curation, lung lobe/nodule segmentation, risk estimation, harmonization, brain segmentation), and an ablation removing access to the artifact contract quantifies the contribution of the structured grounding.

## Q6. Major Strengths

**1. Important and under-addressed problem.** The paper targets a genuine bottleneck in clinical translation of imaging methods. Researchers who have deployed pipelines on real PACS-derived data are well aware that the gap between controlled benchmarks and heterogeneous clinical data is substantial, involving format heterogeneity, nested archives, planning/localizer images, inconsistent naming, and missing metadata. Most MICCAI work focuses on model architecture; this paper addresses the infrastructure layer where most practical failures actually occur.

**2. Clean architectural decomposition.** The separation of semantic planning (LLM-driven, stochastic) from deterministic execution (workflow engine) is a principled design choice that respects the need for both flexibility and scientific reproducibility. The artifact contract serves as the interface between these layers and is a genuinely clever abstraction: it simultaneously (a) provides a queryable state representation, (b) records provenance for auditing, and (c) constrains the LLM to reason over structured fields rather than hallucinating over raw filenames. This addresses a real risk with LLM-driven pipeline tools.

**3. Privacy-aware deployment.** The decision to operate fully locally via Ollama with DeepSeek-R1 14B is a non-trivial engineering commitment that reflects understanding of the regulatory constraints (HIPAA, GDPR, institutional data use agreements) under which clinical imaging research actually operates. This is critical for clinical translation and differentiates the work from cloud-dependent agentic systems.

**4. Strong ablation demonstrating the core claim.** The ablation removing access to the artifact registry (Section 3, Table 2) directly quantifies the value of the artifact contract. Reporting degradations from 100% to as low as 10–20% on filtering/counting queries when the contract is removed is a convincing demonstration that structured grounding, not LLM capability alone, is responsible for the observed accuracy. This is a well-designed ablation that isolates the specific contribution.

**5. Realistic evaluation on diverse clinical cohorts.** The three cohorts cover research CT (NLST), clinical CT (LungCaTrial) with high heterogeneity, and research MRI (BrainICU) with moderate heterogeneity. Nine analytical goals span curation, segmentation, risk estimation, and harmonization. This breadth is appropriate for a framework paper and avoids the common pitfall of evaluating only on a single curated dataset.

**6. Reproducibility considerations taken seriously.** The authors commit to releasing code and report using Snakemake for deterministic execution, which itself is a well-established, auditable tool in bioinformatics. DAG equivalence across repeated runs is reported at 100% across all settings.

## Q7. Major Weaknesses

**1. Absence of head-to-head comparison with existing workflow/adaptation tools.** Although the paper cites Nextflow [2], Snakemake [4], and BIDS [12], it does not directly compare against any of them (or against purely manual workflow assembly by a domain expert) in terms of adaptation time, workflow correctness, or user effort. The only baseline in Table 2 is a degraded version of the authors' own system (no artifact contract). A comparison against (i) pure manual Snakemake scripting, (ii) a standard LLM agent without the artifact contract but with raw file listings, and ideally (iii) a BIDS-based pipeline manager, would substantially strengthen the claim that the artifact contract architecture is uniquely beneficial.

**2. Small and potentially biased semantic query benchmark.** The semantic query evaluation uses only 20 questions per category (status, filter/counting, provenance), per dataset (3 datasets), yielding 180 questions total. This is a very small sample for drawing statistical conclusions about 70% accuracy gains. More problematic, the paper does not disclose who authored the questions. If the authors wrote both the system and the benchmark, the evaluation is vulnerable to confirmation bias: questions that the system can answer may have been subconsciously selected. At minimum, the paper should describe the question-authoring protocol and ideally include questions authored by an independent party (a collaborating clinician or radiologist).

**3. Lack of timing, latency, and cost analysis.** A framework paper claiming adaptability should quantify the human effort saved. How long does workflow assembly take with the framework versus manual configuration by an expert? What is the agent latency (LLM inference time)? How many tokens per query? Without these numbers, one cannot judge whether the framework improves practical deployment or merely provides a more principled abstraction of an equally slow process.

**4. Ground truth for adaptability metrics is constructed by a single domain expert.** The Initial Rule Matching (IRM), Planning Iterations (PL), and Final Output (FO) metrics are measured against a "domain-expert ground-truth configuration." The paper does not report inter-rater reliability or describe the expert's qualifications and role. Since medical imaging workflows can often be assembled in multiple valid ways (e.g., differing intensity normalization strategies), a single-expert ground truth risks conflating "different from expert" with "incorrect." Reporting agreement between two or more experts would strengthen these metrics considerably.

**5. Scalability of the rule library is unclear.** The evaluation uses 11 to 20 rules per analytical goal, totaling perhaps a few dozen rules overall. Real clinical pipelines may need hundreds of rules across many modalities, and retrieval/planning over a large rule library is known to degrade agent performance due to context window constraints and selection ambiguity. The paper does not discuss how the framework scales, whether the rule library would be modularized or hierarchically indexed, or how performance degrades as the rule count grows.

**6. The reproducibility claim is technically true but slightly overstated.** Reporting 100% DAG equivalence across repeated runs is expected when execution is delegated to Snakemake, which is deterministic by design. The novel contribution here is really the fact that the agent repeatedly produces the same configuration C given the same inputs — which depends on LLM sampling temperature and prompt stability, not Snakemake. The paper should clarify that the DAG equivalence result is conditional on the deterministic planner behavior, and report whether temperature was set to 0 for DeepSeek-R1 (the text does not state this).

**7. Limited LLM diversity.** Only two local LLMs are evaluated (DeepSeek-R1 14B and Qwen 2.5 14B). A discussion of how smaller models (e.g., 7B) or larger models (e.g., 70B, or proprietary GPT-4/Claude) would perform — or at least an acknowledgment of the trade-off between local privacy and model capability — would help clinical adopters select appropriate infrastructure.

**8. Figure 2 (system overview) is visually dense.** The system diagram packs user layer, agent layer, execution layer, rule library, memory, skills, and an example configuration snippet into one figure, and is difficult to parse on first reading. Splitting into two figures (one for architecture, one for an example workflow) would improve clarity. Similarly, the example dialog in Table 1 uses "[omit to save space]" several times, which reduces the transparency of the interaction loop.

**9. Missing discussion of failure modes beyond upstream metadata issues.** Section 3 (Adaptability) notes that most failures arise from missing or corrupted DICOM metadata. This is a useful observation, but the paper should also discuss agent-side failure modes: what happens when the LLM hallucinates a non-existent rule, selects contradictory rules, or fails to terminate planning? These scenarios are foreseeable in real deployment and the paper's constrained agent design should have mechanisms to handle them that are worth describing.

**10. The framework does not propose methodological innovation for image analysis itself.** This is a framework/systems paper, not a new model or algorithm for a specific medical imaging task. Some reviewers in a MIC-focused venue may find this positioning uncomfortable. However, for the Clinical Translation track this is actually a strength. The authors should clearly state during the rebuttal that they intend this submission for Clinical Translation, not for a pure methodology track.

## Q8. Clarity and Organization
**Satisfactory.** The paper is well-structured (Introduction -> Methods -> Experiments -> Discussion) and the motivation is clearly articulated. The problem setup (Section 2.1) uses clean notation. However, Figure 2 is visually overloaded, and the example goal-planning dialog in Table 1 is truncated in ways that obscure the interaction patterns. Some minor language polishing would help ("analytical workflow" is used slightly inconsistently; a few sentences in Section 2.2 are long and dense).

## Q9. Reproducibility
**The authors claimed to release the source code and/or dataset upon acceptance of the submission.** The paper explicitly states "Code is available at https://available.upon.acceptance" (Section 2.2). The description of the framework components — artifact contract, rule library, Snakemake execution, Ollama with DeepSeek-R1 14B — is sufficient for an informed reader to reconstruct the high-level design. However, the specific prompts, the exact rule library content, and the evaluation questions for the semantic query benchmark are not provided, which would be needed for exact reproduction of the reported numbers.

## Q10. Code of Ethics Check
**No** — The paper uses publicly accessible research cohorts (NLST [5], BrainICU [3]) and describes a clinical cohort (LungCaTrial) that appears to have been accessed under standard institutional agreements. No raw patient data is transmitted externally (local LLM deployment). No concerns regarding human subjects, consent, or data use are raised by the paper's design.

## Q11. Ethics Concern Description
N/A.

## Q12. Additional Comments to Authors

Thank you for submitting this work. I believe you are addressing a genuinely important and under-represented problem at MICCAI: the gap between controlled benchmarks and heterogeneous clinical deployment. The artifact contract is a clever abstraction, and the decision to separate semantic planning from deterministic execution is principled and practically important.

My main suggestions for strengthening the paper are:

(a) Add a direct comparison against a pure LLM agent (without the artifact contract) as a baseline in Tables 1 and 2, and ideally against a manual Snakemake pipeline authored by a domain expert. Your current ablation (Table 2) is a great start but it compares your system to a degraded version of itself, not to existing practice.

(b) Quantify the human effort/time saved. A plot of assembly time vs. workflow complexity, or at least a mean time-per-workflow number, would make the clinical translation value concrete.

(c) Disclose how the semantic query benchmark questions were authored, and ideally have some questions written by a clinician who is not a co-author.

(d) Clarify the reproducibility claim: DAG equivalence is a property of Snakemake, not of the agent. Your contribution is configuration stability across LLM sampling, which should be reported as such. Did you use temperature=0?

(e) Discuss scalability of the rule library and failure modes of the agent (hallucinated rules, non-termination, conflicting rule selection).

(f) Consider splitting Figure 2 into two figures for clarity, and revising Table 1's dialog example to be less truncated.

(g) Position the paper explicitly for the Clinical Translation track during rebuttal, because the contribution is really infrastructural rather than algorithmic.

(h) Please double-check citations: reference [3] (BLSA) appears cited as the source for BrainICU (Section 3) which may be a labelling inconsistency; verify.

Overall, the paper demonstrates thoughtful engineering and addresses a real clinical translation gap. With the above additions, it could be a very strong contribution.

## Q13. Recommendation
**4 — Weak Accept** (marginally above the acceptance threshold, dependent on rebuttal)

## Q14. Justification of Recommendation

I recommend Weak Accept (4). The paper addresses an important problem in clinical translation that is under-represented at MICCAI, proposes a principled architectural solution (artifact contract + constrained agent + deterministic executor), and validates it on three real clinical cohorts with a meaningful ablation. The artifact contract abstraction is a genuine contribution that goes beyond "LLM + tool use" framings: by constraining the agent to reason over structured artifact fields with explicit provenance, it directly addresses grounding and reproducibility concerns that have limited deployment of LLM-driven pipeline tools in clinical settings. The 70% accuracy gain in the ablation is compelling evidence that the contract is doing real work, not merely wrapping the LLM.

My main concerns are evaluative rather than conceptual. The baselines are too narrow (only a degraded self-baseline), the semantic query benchmark is small and may be author-constructed without disclosure, the reproducibility claim partially conflates Snakemake determinism with the agent's own stability, and the scalability of the rule library is unaddressed. A direct comparison to manual pipeline assembly or to an unconstrained LLM agent would significantly strengthen the claim that the artifact contract is the key enabler. Timing/effort measurements are absent and are important for a paper framed around adaptability and clinical translation.

On balance, the conceptual contribution is strong enough and the domain problem important enough that I believe this paper is worth accepting if the authors can address these evaluative concerns in the rebuttal. I would not mind if it were rejected, because the evaluation gaps are real, but I would prefer to see it accepted because MICCAI does not see enough clinical-translation infrastructure papers of this quality.

## Q15. Expertise Limitations
My expertise is primarily in medical image analysis, deep learning for neuroimaging, and multi-site MRI harmonization. I am familiar with Snakemake/Nextflow in a bioinformatics context and with LLM agent architectures (ReAct, Toolformer, tool-use RL). I am less familiar with the specific clinical workflow management literature in radiology beyond BIDS and DICOM metadata standards, so my assessment of novelty relative to industrial pipeline tools (e.g., commercial PACS workflow engines) may be incomplete.

## Q16. Review Confidence
**Confident but not absolutely certain (3).** I am confident in my methodological and evaluative critique. My uncertainty is around the novelty of the artifact contract concept relative to industrial imaging workflow standards that are outside the typical MICCAI literature.

## Q17. Role/Position
Ph.D. student / Undergraduate researcher at UC Berkeley EECS. (Note: if forced to choose one, select the closest available option per the form's categories.)

## Q18. Paper Highlighting
- Highlighted poster
- Journal (if the authors could provide the additional experiments suggested, this would be a strong extension-paper candidate for a venue like Medical Image Analysis or IEEE TMI)

## Q19. Highlighting Comment

The paper represents a thoughtful engineering contribution addressing a real clinical translation gap. The artifact contract abstraction creates opportunity for discussion at the intersection of LLM agents and reproducible medical AI, which is an emerging topic with limited representation at MICCAI. A highlighted poster would allow interactive discussion about scalability and deployment trade-offs, which are best explored in conversation rather than a short oral.

## Q20. Confidential Comments to ACs/PCs

The paper appears authored by humans familiar with practical clinical imaging workflows; the engineering detail about DICOM header problems, mixed sessions, and nested archives is specific and credible. I did not detect any signs of LLM-generated content or hallucinated references. All 19 cited references appear plausible and consistent with the claims they support (I did not exhaustively verify each one, but the key citations — Snakemake [4], Toolformer [13], ReAct [16], TotalSegmentator [15], UNesT [17], HACA3 [19] — are all real and correctly attributed).

The main concern I would flag to ACs is whether this paper fits the MIC/methodology track criteria or should be routed through the Clinical Translation track. In a MIC-focused assessment, the absence of a new model or algorithm may be viewed unfavorably. In a Clinical Translation assessment, the work is substantially stronger. I recommend the ACs explicitly verify the track placement during discussion.
