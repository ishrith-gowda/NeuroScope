# MICCAI 2026 Review — Paper 1527

**Title:** Guiding A Visual-Language Model to Generate Clinical Reasoning via Structured Mechanisms

---

## Q1. Reviewer Guide Confirmation
I agree.

## Q2. Public Release Consent
I agree.

## Q3. Relevance Category
MIC (Medical Image Computing)

## Q4. Paper Type
Methodological contribution

## Q5. Main Contribution

The paper proposes SARM, a medical VLM fine-tuned on Qwen2-VL-7B that enforces a three-stage `<DESCRIPTION>/<REASONING>/<ANSWER>` output format and trains with GRPO using a "correctness-aware Gaussian length reward" (R = 0.9·r + 0.1·exp(-((L-μ)/σ)²) for correct answers and 0 otherwise, with μ=200, σ=50), paired with an Easy/Hard/Auto adaptive mode controlled by difficulty-labeled prompts. Training is two-stage: SFT on a 60K Med-DRA dataset distilled from GPT-4o over PubMedVision/PMC-VQA plus 20K ShareGPT-4o-Image samples, then GRPO on 3K PMC-VQA samples. Results are reported on MMMU-Med, VQA-RAD, SLAKE, PMC-VQA, and Path-VQA, with a 49.0% average against several medical VLM baselines.

## Q6. Major Strengths

Adaptive reasoning length for medical VQA is a reasonable research direction — simple modality questions do not need long chains, harder cases do. The Gaussian length kernel is a clean choice that produces smoother gradients than hard truncation or L1 penalties. Enforcing a structured output via a regex-based format reward is practical and helps downstream parsing. Evaluation breadth across five benchmarks (closed/open, radiology/pathology) is appropriate.

## Q7. Major Weaknesses

1. **LLaVA-CoT is mischaracterized.** The paper describes LLaVA-CoT's four stages as "summarization, description, reasoning, conclusion" and criticizes it as "prompt engineering rather than learning-driven." Both are wrong. LLaVA-CoT's stages are `<SUMMARY>/<CAPTION>/<REASONING>/<CONCLUSION>`, and the method is primarily supervised fine-tuning on a ~100K stage-tagged dataset (LLaVA-CoT-100k), plus a stage-level beam search at inference. Once corrected, SARM's three-stage paradigm reads as LLaVA-CoT with summary and caption collapsed, not a distinct contribution.

2. **HuatuoGPT-Vision numbers are far below published values.** Table 3 reports HuatuoGPT-Vision at 46.0/53.0/59.8/32.0/32.0 on the five benchmarks. The original paper reports roughly 54.4/68.1/76.9/58.2/63.5 for its 34B checkpoint — 8–31 points higher. The SARM paper does not state which checkpoint or evaluation protocol was used. Without that disclosure the claim of outperforming HuatuoGPT-Vision is not supportable, and the published 34B numbers are above SARM's own 49.0 average.

3. **Med-R1 and MedVLM-R1 numbers are unlabeled reproductions.** Neither paper reports on MMMU-Med, VQA-RAD, SLAKE, PMC-VQA, or Path-VQA. Med-R1 evaluates only on OmniMedVQA modality splits; MedVLM-R1 reports on a combined 17.3K MCQ set split by MRI/CT/X-ray. Table 3's numbers are the SARM authors' own re-runs of 2B-scale models on broader benchmarks without stating this, which is misleading and likely produces unfavorable baselines.

4. **The length reward is very close to AALC (ref [7]).** AALC already proposes accuracy-gated length shaping with γ=0.9 as the accuracy weight — the same constant SARM uses. The Gaussian kernel over target length is a variation, not a new paradigm. AALC should be cited as the direct precedent and ablated against.

5. **The Auto mode fails on 2 of 5 benchmarks by the authors' own numbers.** Table 2 shows "Ours" underperforms Hard mode on PMC-VQA (40.9 vs 43.7) and Path-VQA (35.0 vs 36.2). For a paper whose central claim is adaptive length control, this is a direct failure of the main hypothesis. It is mentioned only in passing.

6. **No ablations.** Three contributions are claimed (structured format, adaptive strategy, length reward) with no experiment isolating any of them. There is also no description of how difficulty labels are assigned to training samples — a critical detail because the adaptive mechanism depends on it.

7. **Reference [8] is miscited.** The paper attributes "p-term prompting" to Zhang et al. (arXiv:2505.15400), but `p_term` in that paper is a variable name for a hard-coded No-Thinking prefix in the ASRR method, not a general technique.

8. **Overstated clinical claims.** The introduction claims "clinical trustworthiness" and real-world deployment readiness, but evaluation is exact-match VQA accuracy with no reader study, no error analysis, and no statistical reporting. Training also distills GPT-4o outputs, which the paper does not discuss as a limitation for clinical correctness.

## Q8. Clarity and Organization
Satisfactory. The overall structure is readable but several issues reduce clarity: the LLaVA-CoT mischaracterization, the missing difficulty-partitioning protocol, an overloaded stylized pipeline figure, grammar errors ("All implemented were worked on four NVIDIA H20 GPUs"), and Table 2's failure case going undiscussed.

## Q9. Reproducibility
The submission does not provide sufficient information for reproducibility. No code or data release is mentioned; the difficulty-partitioning protocol, Med-DRA construction prompts, and baseline evaluation protocols (checkpoint, MC vs open-ended) are not specified, which makes Table 3 not independently verifiable.

## Q10. Code of Ethics Check
Not sure / insufficient information.

## Q11. Ethics Concern Description
The Med-DRA SFT data is distilled from GPT-4o on medical image–question pairs, which raises two questions: whether this use is compliant with OpenAI's Terms of Service, and whether the synthetic reasoning chains were clinically verified before being used as training signal. The paper mentions "self-verification" but does not describe the protocol.

## Q12. Additional Comments to Authors

Thank you for the submission — the direction is worthwhile. My main suggestions: (a) correct the LLaVA-CoT description and reposition the three-stage paradigm honestly relative to it; (b) cite AALC as the precedent for accuracy-gated length shaping and ablate against it; (c) label Med-R1 / MedVLM-R1 / HuatuoGPT-Vision numbers as your reproductions, state the checkpoint and protocol, and include the published 34B HuatuoGPT-Vision numbers alongside; (d) add ablations isolating the structured format, adaptive strategy, and length reward; (e) diagnose why Auto underperforms Hard on PMC-VQA and Path-VQA; (f) describe how difficulty labels are assigned; (g) fix the ref [8] citation (p_term is a variable name, not a method); (h) report standard deviations and a hyperparameter sensitivity sweep for μ, σ, and the 0.9/0.1 weights; (i) disclose compliance with GPT-4o ToS and the clinical verification procedure for Med-DRA.

## Q13. Recommendation
**2 — Reject**

## Q14. Justification of Recommendation

The novelty claims rest on a mischaracterization of LLaVA-CoT; once corrected, the three-stage paradigm is a merged variant rather than a new design. The length reward is very close to AALC's accuracy-gated shaping with the same γ=0.9 coefficient, without citation. The Table 3 baselines are not credible as presented: HuatuoGPT-Vision numbers are 8–31 points below the published 34B values, and Med-R1/MedVLM-R1 are evaluated on benchmarks their original authors never reported on, without labeling as reproductions. The central adaptive-length claim fails on 2 of 5 benchmarks by the paper's own Table 2. There are no ablations isolating the three claimed contributions, the difficulty-partitioning protocol is undocumented, and the hyperparameters are unjustified. Any one of these would warrant substantial revision; together they place the paper below the acceptance threshold and beyond what an 800-word rebuttal can address.

## Q15. Expertise Limitations
My expertise is in medical image analysis, multi-modal deep learning, and RL fine-tuning of language/vision models (GRPO/PPO/DPO, reasoning and length control). I am less routinely tracking per-benchmark numbers across the five medical VQA datasets, though the gap to published HuatuoGPT-Vision values is large enough to remain a concern regardless of checkpoint variant.

## Q16. Review Confidence
Confident but not absolutely certain (3).

## Q17. Role/Position
Ph.D. student.

## Q18. Paper Highlighting
None.

## Q19. Highlighting Comment
N/A.

## Q20. Confidential Comments to ACs/PCs

A few issues I want to flag. (1) LLaVA-CoT is mischaracterized — actual tags are summary/caption/reasoning/conclusion, and the method is SFT-based on the LLaVA-CoT-100k dataset, not prompt engineering. (2) Reference [8] (arXiv:2505.15400) does not introduce "p-term prompting"; `p_term` is a variable name for a No-Thinking prefix in the ASRR method. (3) Med-R1 and MedVLM-R1 never reported on the five benchmarks used in Table 3 — numbers are the authors' own reproductions and not labeled. (4) HuatuoGPT-Vision numbers are 8–31 points below the published 34B values. I verified all four against the cited arXiv papers. These are not subtle errors and materially affect the paper's positioning.

The writing has some machine-translation-style phrasing consistent with non-native English authors; I do not see evidence of LLM-generated content or hallucinated references beyond the misattribution flagged above. The core technical content appears human-authored.

Recommend reject with an invitation to resubmit once baselines are relabeled, ablations are added, and the LLaVA-CoT / AALC framing is corrected.
