# MICCAI 2026 Review — Paper 1527

**Title:** Guiding A Visual-Language Model to Generate Clinical Reasoning via Structured Mechanisms

---

## Q1. Reviewer Guide Confirmation
I agree. I have read and understood the MICCAI 2026 Reviewers' Guide and the Confidentiality section; no part of the paper has been shown to an LLM.

## Q2. Public Release Consent
I agree.

## Q3. Relevance Category
- MIC (Medical Image Computing)

## Q4. Paper Type
Methodological contribution (with application study component)

## Q5. Main Contribution

The paper proposes SARM (Structured-Adaptive Reasoning Model), a medical vision-language model that aims to improve transparency and clinical interpretability of diagnostic reasoning. The authors argue that direct-response medical VLMs omit critical reasoning stages, and that existing chain-of-thought approaches (specifically LLaVA-CoT) are "excessive for clinical workflows." Their proposed solution has three components.

First, a three-stage structured thinking paradigm using `<DESCRIPTION>` (extraction of clinically relevant image observations), `<REASONING>` (intermediate inference grounded in domain knowledge), and `<ANSWER>` (final prediction), intended to decompose medical reasoning into interpretable stages. The output format is enforced via a regex-based multistage format reward.

Second, an "Adaptive Length-controlled strategy" with three modes — Easy (shallow reasoning via a short prompt), Hard (detailed step-by-step reasoning), and Auto (adaptive). Each training instance is pre-annotated with a difficulty label that selects the mode during GRPO optimization. The mode is instantiated by prepending a mode-specific natural-language prompt (p_term) to the multimodal input.

Third, a "Correctness-Aware Gaussian Length Reward" for GRPO: when the generated answer is correct, the reward is R = 0.9·r + 0.1·ℓ where r ∈ {0,1} is exact-match correctness and ℓ = exp(-((L-μ)/σ)^2) is a Gaussian kernel over generated token length L around a target μ = 200 with σ = 50. When the answer is incorrect, R = 0. The stated purpose is to suppress both overly short and overly long reasoning while being gated on correctness.

Training uses a two-stage pipeline: SFT cold-start on a 60K Med-DRA dataset (constructed by querying GPT-4o over PubMedVision and PMC-VQA) plus 20K ShareGPT-4o-Image samples, followed by GRPO on 3K difficulty-partitioned PMC-VQA samples. The backbone is Qwen2-VL-7B, trained on 4 NVIDIA H20 GPUs. Evaluation covers five benchmarks — MMMU-Med, VQA-RAD, SLAKE, PMC-VQA, Path-VQA — with the reported headline result being an average accuracy of 49.0% against a range of baselines, positioned as state-of-the-art on three of the five benchmarks.

## Q6. Major Strengths

**1. Reasonable motivation.** Adaptive reasoning length in medical VLMs is a legitimate problem. Simple modality-recognition tasks genuinely do not need long chains of thought, while complex differential diagnosis tasks benefit from detailed inference. Coupling a structured reasoning format with length-aware optimization is a reasonable research direction.

**2. The Gaussian length reward formulation is clean.** Using a Gaussian kernel over token length to penalize deviations from a target (rather than hard-threshold truncation or L1 penalties) is an elegant choice that produces smooth gradients during policy optimization.

**3. Multistage format reward with regex verification.** Enforcing the `<DESCRIPTION>...</DESCRIPTION><REASONING>...</REASONING><ANSWER>...</ANSWER>` format via a regex reward ensures structural consistency across training, which is helpful for downstream parsing and interpretability.

**4. Breadth of evaluation benchmarks.** Testing on five distinct medical VQA datasets (MMMU-Med, VQA-RAD, SLAKE, PMC-VQA, Path-VQA) covers multiple modalities and task types (closed-set, open-ended, radiology, pathology).

## Q7. Major Weaknesses

**1. Mischaracterization of prior work (LLaVA-CoT).** The paper criticizes LLaVA-CoT [3] as relying on "prompt engineering rather than learning-driven optimization" and describes its four stages as "summarization, description, reasoning, conclusion." Both characterizations are incorrect. LLaVA-CoT's actual stage tags, as defined in the paper and its training data, are `<SUMMARY>`, `<CAPTION>`, `<REASONING>`, and `<CONCLUSION>`, where "caption" specifically means a description of image regions relevant to the question and is optionally skipped in text-only cases — a design decision the SARM paper elides. More importantly, LLaVA-CoT is primarily learning-driven: the authors construct the LLaVA-CoT-100k dataset with stage-tagged supervision and perform supervised fine-tuning on it, and additionally propose a stage-level beam search at inference. Framing LLaVA-CoT as "prompt engineering" is misleading and undermines the stated differentiation from SARM. Once corrected, the novelty of SARM's three-stage paradigm (description, reasoning, answer) — which reads as a merged version of LLaVA-CoT with the summary and caption stages collapsed — is substantially reduced.

**2. Likely citation error in the p-term prompting reference.** The paper cites reference [8] (Zhang et al., "When to continue thinking: Adaptive thinking mode switching for efficient reasoning", arXiv:2505.15400, 2025) as the source of "p-term prompting." Reference [8] actually proposes ASRR (Adaptive Self-Recovery Reasoning) with an accuracy-aware length reward. The string `p_term` appears in that paper only as a variable name for a hard-coded No-Thinking prefix (the literal string "Okay, I have finished thinking."); it is not a general methodology called "p-term prompting." The SARM authors appear to have overloaded the variable name as if it were an established technique. The closer antecedents — the "Wait"/budget-forcing approach from s1 (Muennighoff et al., 2025), and planning-token work — are not cited.

**3. Baseline reproduction integrity.** Table 3 reports Med-R1 [19] at 14.7 MMMU-Med, 39.0 VQA-RAD, 48.0 SLAKE, 27.6 PMC-VQA, 15.3 Path-VQA, and MedVLM-R1 [18] at 13.3 MMMU-Med, 48.6 VQA-RAD, 56.0 SLAKE, 32.5 PMC-VQA, 17.2 Path-VQA. **Neither Med-R1 nor MedVLM-R1 report results on any of these five benchmarks in their original papers.** Med-R1 (Lai et al., arXiv:2503.13939, 2025) evaluates exclusively on OmniMedVQA split by 8 imaging modalities and question types, and MedVLM-R1 (Pan et al., arXiv:2502.19634, 2025) reports on a combined 17,300-MCQ set split by modality (MRI/CT/X-ray), not on MMMU-Med/VQA-RAD/SLAKE/PMC-VQA/Path-VQA. The numbers in Table 3 are therefore the SARM authors' own reproductions, but they are not labeled as such. Because Med-R1 and MedVLM-R1 are small Qwen2-VL-2B models RL-tuned on narrower closed-form question sets, running them on broader benchmarks without adapting the evaluation protocol is likely to produce unfavorable numbers. The paper must clearly state that these are reproductions, describe the evaluation protocol used, and ideally re-evaluate under conditions closer to the original papers' settings.

**4. HuatuoGPT-Vision numbers are dramatically lower than published.** Table 3 reports HuatuoGPT-Vision [14] at 46.0 MMMU-Med, 53.0 VQA-RAD, 59.8 SLAKE, 32.0 PMC-VQA, 32.0 Path-VQA. The HuatuoGPT-Vision paper (Chen et al., arXiv:2406.19280, 2024) reports, for its **34B checkpoint**, approximately MMMU-Med ~54.4, VQA-RAD 68.1, SLAKE 76.9, PMC-VQA 58.2, Path-VQA 63.5 — roughly 8–31 points above SARM's reported reproductions. The SARM paper does not state which HuatuoGPT-Vision checkpoint (7B vs. 34B) was evaluated or under what protocol (open-ended exact-match vs. multiple-choice, prompt format). Even if a 7B variant was used, the gap (and SARM's own ~49.0 average) cannot be interpreted without this disclosure. At minimum the authors must specify the checkpoint and evaluation protocol, and should include the published 34B numbers alongside their reproductions so readers can judge the comparison fairly. As presented, the claim that SARM outperforms HuatuoGPT-Vision is not supportable.

**5. Novelty of the length reward is overstated relative to AALC.** Reference [7] (Li et al., "AALC: Large Language Model Efficient Reasoning via Adaptive Accuracy-Length Control", arXiv:2506.20160, 2025) proposes an accuracy-aware length reward of the form R = Att_acc·R_raw + α·R_len with Att_acc = γ + (1-γ)(1-r_acc) and γ = 0.9. When r_acc = 1 (correct), AALC's accuracy weight equals γ = 0.9; when incorrect, it increases to 1.0 (AALC shifts weight onto accuracy for wrong answers). SARM's reward is the same 0.9 accuracy weight when correct, but zeroes the total reward when incorrect rather than re-weighting — a different gating policy that nonetheless shares AALC's core principle: never allow length shaping to trade against correctness, with the 0.9 coefficient as the accuracy anchor when correct. The Gaussian kernel over target length is a modest variation on AALC's length term (AALC uses a `1 - min(r_acc^β, r_len)` penalty); it is not a new class of method. The paper should cite AALC as the direct precedent for accuracy-gated length shaping and ideally ablate the Gaussian kernel against AALC's original reward. As written, the Correctness-Aware Length Reward reads as a variant of an existing technique rather than a novel contribution.

**6. The adaptive strategy fails on 2 of 5 benchmarks (by the paper's own numbers).** Table 2 shows that on PMC-VQA and Path-VQA, SARM's "Ours" mode (40.9 and 35.0) actually *underperforms* the authors' own Hard mode (43.7 and 36.2). This is a substantial failure of the adaptive mechanism: on the two benchmarks where the Auto mode is supposed to choose intelligently between Easy and Hard, it consistently chooses worse than manually fixing Hard. The paper mentions this briefly ("slightly lower than the Hard mode") but does not analyze *why* the adaptive policy fails on these benchmarks, does not reconcile it with the claimed contribution, and does not discuss whether the difficulty pre-annotation is accurate for these datasets. This failure mode directly undermines the central claim.

**7. Complete absence of ablation studies.** The paper claims three contributions — (1) the structured thinking paradigm, (2) the adaptive length-controlled strategy, and (3) the correctness-aware length reward — but provides no ablation studies isolating their individual contributions. There is no experiment showing what happens if (a) the structured paradigm is removed (i.e., direct answer with length reward), (b) the adaptive strategy is removed (i.e., only Hard mode with length reward), (c) the length reward is removed (i.e., structured paradigm with only correctness reward), or (d) only the format reward is used. Without these, one cannot determine which component is actually responsible for any observed gains, and the paper's framing of "three contributions" is unsupported.

**8. Difficulty partitioning protocol is not described.** The paper repeatedly refers to a "difficulty-partitioned signal" or "pre-annotated difficulty label" that drives the Easy/Hard/Auto mode selection, but does not explain how difficulty is determined. Is it based on sample-level accuracy of a baseline model, annotator judgment, question length, answer-set size, or some other criterion? This is a critical methodological detail because the entire adaptive strategy depends on the labeling being meaningful.

**9. Synthetic training data via GPT-4o is not disclosed as a limitation.** The Med-DRA SFT dataset (60K samples) is constructed by prompting GPT-4o to generate "detailed descriptions and reasoning content" for PubMedVision and PMC-VQA samples, then self-verifying. This is essentially distillation from GPT-4o. The model is therefore learning to replicate GPT-4o's reasoning style, not independent clinical reasoning. This is not disclosed as a limitation, and no analysis is provided of (a) hallucination rates in the generated reasoning, (b) clinical correctness of the synthetic chains, or (c) whether evaluation benchmarks overlap with GPT-4o's training data. Any claim of "clinical trustworthiness" (which the paper makes in the introduction) is difficult to support when training signal derives from an upstream LLM with unknown clinical reliability.

**10. Reward weighting is arbitrary and unjustified.** The paper uses R = 0.9·r + 0.1·ℓ without justification for the 0.9/0.1 split, and uses target μ = 200 tokens and σ = 50 tokens without sensitivity analysis. Given that the entire adaptive mechanism hinges on these hyperparameters, a sensitivity sweep over at least σ and μ is expected.

**11. Overstated clinical claims.** The introduction states that the work "enhances clinical trustworthiness" and "facilitates the deployment of AI-assisted healthcare systems in real-world clinical settings." The evaluation is on closed-form VQA benchmarks with exact-match accuracy — no clinical validation, no clinician reader study, no case presentations, no error analysis on clinically relevant failure modes. These claims are not warranted by the evaluation.

**12. Missing statistical reporting.** There are no standard deviations, confidence intervals, or significance tests. The average improvement over HuatuoGPT-Vision (49.0 vs. 44.6 = +4.4 points) could easily be within noise across five benchmarks, particularly given the concerns about the HuatuoGPT-Vision baseline in point 4.

**13. Only a single backbone.** SARM is implemented on Qwen2-VL-7B [21]. The paper does not test whether the method generalizes to other VLM backbones (InternVL, LLaVA-Med, BLIP-2), which would be important to claim methodological generality.

**14. Minor writing and figure issues.** The abstract and Section 3.2 contain several grammar issues ("All implemented were worked on four NVIDIA H20 GPUs", "enabling accurate, stable, and appropriately scaled reasoning across diverse tasks"). Figure 1 is stylized (cartoon robots, speech bubbles, hand-drawn annotations) rather than a formal pipeline diagram — while visually distinctive, it makes it harder to precisely understand the data flow between SFT and GRPO stages. Table 1 (token counts) and Table 2 (accuracy) would benefit from being merged or placed side-by-side for easier cross-reference.

**15. No code or data release commitment.** The paper does not mention releasing code, the Med-DRA dataset, the difficulty-partition labels, or model checkpoints. For an RL-trained model depending on custom synthetic data and difficulty labels, this is a significant reproducibility concern.

## Q8. Clarity and Organization
**Satisfactory to Poor.** The paper's high-level motivation and section structure are understandable, but several issues reduce clarity: grammar errors throughout, mischaracterization of LLaVA-CoT's architecture, missing description of the difficulty-partitioning protocol, a stylized pipeline figure that is difficult to parse precisely, and insufficient technical detail in the reward function description. The key claim about adaptive strategy effectiveness is partially contradicted by Table 2 without discussion. I lean toward Satisfactory primarily because a competent reader can still follow the overall approach, but the clarity is below the level expected for MICCAI.

## Q9. Reproducibility
**The submission does not provide sufficient information for reproducibility.** Specifically: (a) no code or data release is mentioned; (b) the difficulty-partitioning protocol is not described; (c) the exact Med-DRA construction prompts for GPT-4o are not given; (d) the HuatuoGPT-Vision and other baseline evaluation protocols are not described (checkpoint size, open-ended vs. multiple-choice, prompt format), making the reported Table 3 numbers not independently verifiable; (e) hyperparameter sensitivity for μ, σ, and the 0.9/0.1 reward weights is absent.

## Q10. Code of Ethics Check
**Not sure/insufficient information.** The training pipeline distills reasoning from GPT-4o on medical image-question pairs. Depending on the OpenAI Terms of Service at the time of data construction, using GPT-4o outputs to train a competing model may violate the terms (which prohibit using outputs to develop models that compete with OpenAI). The paper should disclose its compliance with the GPT-4o ToS, and should discuss whether the Med-DRA dataset can be released consistent with those terms. Additionally, the evaluation involves medical diagnostic questions where hallucinated or incorrect reasoning chains (distilled from an external LLM) could propagate clinically unsafe content — this is a responsible-AI concern worth flagging, though not necessarily an ethics violation.

## Q11. Ethics Concern Description
See Q10. My concern is twofold: (1) whether the use of GPT-4o to generate synthetic clinical reasoning (Med-DRA) is compliant with OpenAI's Terms of Service regarding training competing models, and (2) whether the synthetic reasoning chains were verified for clinical accuracy before being used as training signal. The paper mentions a "self-verification process" but does not describe what was verified, by whom, or with what accuracy. Given that the model is positioned for "real-world clinical settings," these disclosures are needed.

## Q12. Additional Comments to Authors

Thank you for the submission. The topic is timely and the research direction — structured, length-aware medical reasoning — is worthwhile. However, in its current form the paper has substantial issues that I believe must be addressed before it can be considered for MICCAI. My main suggestions are:

**Related work and novelty positioning.** Please re-read LLaVA-CoT carefully. The four stages are *summary, caption, reasoning, conclusion*, and the method is primarily supervised fine-tuning with stage tags, not prompt engineering. Your three-stage paradigm is closer to a merged version of LLaVA-CoT than a distinct paradigm, and this should be stated honestly. Similarly, AALC (ref [7]) already proposes accuracy-gated length shaping with γ = 0.9 — you should cite this as the direct precedent for your length reward, and your contribution should be positioned as a Gaussian-kernel variant rather than a new class of method.

**Citation correction.** Reference [8] (Zhang et al., arXiv:2505.15400) does not introduce "p-term prompting"; `p_term` in that paper is just a variable name for a hard-coded No-Thinking prefix used in their ASRR method. Please correct this citation or re-attribute the concept.

**Baseline reproductions.** Table 3 reports Med-R1 and MedVLM-R1 numbers on benchmarks their original authors did not evaluate on (both papers use OmniMedVQA/MCQ splits, not MMMU-Med/VQA-RAD/SLAKE/PMC-VQA/Path-VQA). Please clearly label these as your reproductions, describe the evaluation protocol (prompting, exact-match vs. multiple-choice), and ideally re-evaluate at matched settings. Additionally, your HuatuoGPT-Vision numbers are 15–31 points below the published values — please verify checkpoint size and evaluation protocol, and re-run if necessary. If the published numbers cannot be matched under a standardized protocol, please include the published numbers alongside your reproductions so readers can judge the comparison fairly.

**Ablations.** Please add ablations isolating the contribution of (a) the structured format, (b) the adaptive strategy, (c) the length reward, and (d) the correctness gating. Without these, the three-contribution framing cannot be justified.

**Adaptive strategy failure on PMC-VQA and Path-VQA.** Please explicitly analyze why the Auto mode underperforms the Hard mode on these two benchmarks. Is it a difficulty-labeling problem? A mode-collapse in the Auto policy? This is the most important empirical question the paper should answer.

**Difficulty partitioning.** Please describe the protocol used to assign Easy/Hard labels to training samples.

**Statistical reporting.** Please report standard deviations or confidence intervals on the benchmark results.

**Hyperparameter sensitivity.** Please show how performance depends on μ, σ, and the 0.9/0.1 reward weights.

**Ethics disclosure.** Please clarify compliance with GPT-4o ToS when using GPT-4o outputs for training, and describe the self-verification protocol used to check clinical accuracy of the generated reasoning.

**Code and data release.** Please commit to releasing the Med-DRA dataset, difficulty labels, and training code upon acceptance.

## Q13. Recommendation
**2 — Reject** (should be rejected, independent of rebuttal)

## Q14. Justification of Recommendation

I recommend Reject (2) for the following reasons, each of which is substantial on its own:

**First, the paper's novelty claims rest on a mischaracterization of LLaVA-CoT.** The central contribution of the "three-stage structured paradigm" is differentiated from LLaVA-CoT [3] by claiming LLaVA-CoT uses "prompt engineering rather than learning-driven optimization" and by listing its stages as "summarization, description, reasoning, conclusion." Both claims are incorrect. LLaVA-CoT is primarily a learning-driven method with supervised fine-tuning on stage-tagged data, and its four stages are summary, caption, reasoning, conclusion. Correcting this reveals that SARM's three-stage paradigm (description, reasoning, answer) is essentially a merged version of LLaVA-CoT with one stage dropped, not a fundamentally distinct contribution.

**Second, the length-reward formulation is very close to AALC (Li et al., 2025) without acknowledgment.** AALC already proposes accuracy-gated length shaping with γ = 0.9 as the accuracy weight, which is the same constant used by SARM's R = 0.9·r + 0.1·ℓ. SARM's Gaussian kernel over target length is a variation, not a new paradigm. Positioning this as a third core contribution without citing AALC as the direct precedent overstates the novelty.

**Third, the baseline numbers in Table 3 are not credible as presented.** Med-R1 [19] and MedVLM-R1 [18] never reported results on MMMU-Med, VQA-RAD, SLAKE, PMC-VQA, or Path-VQA in their original papers — Med-R1 uses OmniMedVQA and MedVLM-R1 uses a combined 17.3K MCQ set split by modality. The numbers in Table 3 are the authors' own reproductions but are not labeled as such. Furthermore, HuatuoGPT-Vision numbers are 15–31 points below the published values across all five benchmarks, which is a very large gap and either indicates a mis-evaluation (wrong checkpoint, wrong protocol) or a mis-reporting. Either way, the claim that SARM outperforms HuatuoGPT-Vision is not supportable from Table 3 as it stands.

**Fourth, the adaptive strategy — the central empirical contribution — fails on 2 of 5 benchmarks.** On PMC-VQA and Path-VQA, SARM's Auto mode (40.9 and 35.0) underperforms the authors' own Hard mode (43.7 and 36.2). The paper mentions this only briefly and does not diagnose why. For a paper whose main claim is that adaptive length control improves reasoning across diverse tasks, this is a direct failure of the central hypothesis that is not adequately addressed.

**Fifth, there are no ablation studies.** The paper claims three distinct contributions but provides no experiments isolating them. This is below the expected standard for methodological papers at MICCAI.

**Sixth, critical methodological details are missing.** The difficulty-partitioning protocol is not described, the hyperparameters (0.9/0.1 weights, μ = 200, σ = 50) are unjustified, the Med-DRA construction prompts are not shown, statistical reporting is absent, and no code/data release is committed.

Any one of these issues would warrant substantial revision. Together, they indicate that the paper is not yet ready for acceptance at MICCAI 2026. I do not believe these concerns can be adequately addressed in a rebuttal — the required changes include rerunning baselines, adding ablation studies, and correcting the related-work framing, which are beyond the scope of an 800-word rebuttal.

If the authors address all of the above in a revised submission — particularly correcting the LLaVA-CoT and AALC framing, properly labeling reproduced baselines, adding ablations, and diagnosing the PMC-VQA/Path-VQA failure mode — I would be happy to reconsider a future version at another venue.

## Q15. Expertise Limitations

My expertise is in medical image analysis, multi-modal deep learning, and generative models. I am familiar with GRPO/PPO/DPO and with recent work on reasoning in LLMs (chain-of-thought, budget forcing, length control). I am less immersed in the specific medical VQA benchmark landscape (I know the benchmarks exist and their high-level characteristics, but I do not routinely track reported numbers across all five). I am therefore confident in the methodological and novelty assessment, but slightly less confident on the exact published numbers for HuatuoGPT-Vision at various model sizes. My understanding is that the reported 15–31 point gap is large enough that it is likely real regardless of exact checkpoint variant.

## Q16. Review Confidence
**Confident but not absolutely certain (3).** I am confident in the methodological and novelty concerns (LLaVA-CoT, AALC, ablations, baseline labeling, PMC-VQA/Path-VQA failure). I am slightly less certain about the exact magnitude of the HuatuoGPT-Vision discrepancy because different checkpoint sizes and evaluation protocols can shift numbers meaningfully. However, even granting a generous interpretation, the other concerns independently support rejection.

## Q17. Role/Position
Ph.D. student / Undergraduate researcher at UC Berkeley EECS.

## Q18. Paper Highlighting
None of the available options apply. This paper is not recommended for oral, highlighted poster, best paper/young scientist award, or journal highlighting at this time.

## Q19. Highlighting Comment
N/A.

## Q20. Confidential Comments to ACs/PCs

I want to flag several issues for the ACs that are important for assessing this paper.

**Citation and baseline concerns.** In my review I document that (a) LLaVA-CoT is mischaracterized (actual stages are summary/caption/reasoning/conclusion, and the method is learning-driven, not prompt engineering), (b) reference [8] is miscited as introducing "p-term prompting" when `p_term` is a variable name for a No-Thinking prefix in the ASRR method, (c) Med-R1 [19] and MedVLM-R1 [18] never report on the five benchmarks used in Table 3 — the numbers are the SARM authors' own reproductions but are not labeled, and (d) HuatuoGPT-Vision numbers are 15–31 points below published values. I verified these by cross-checking the cited references. These are not subtle mistakes; they materially affect the credibility of the paper's positioning and empirical claims.

**Likely LLM-assisted writing without content generation.** The writing has some machine-translation-style phrasing ("All implemented were worked on four NVIDIA H20 GPUs", "enabling accurate, stable, and appropriately scaled reasoning") consistent with non-native-English authors, but I do not see signs of LLM-generated reasoning or hallucinated references beyond the misattribution flagged above. The core technical content appears authored by humans familiar with GRPO.

**Clinical safety concern.** The training pipeline distills reasoning from GPT-4o on medical image-question pairs to produce the Med-DRA dataset. This has two issues: (1) possible ToS violation, and (2) unverified clinical correctness of synthetic reasoning chains being used to train a model the authors claim is intended for real clinical settings. The paper's "clinical trustworthiness" framing is not supported by the evaluation methodology.

**Track placement.** The paper seems to assume a MIC methodology slot. Given the weaknesses in baselines and ablations, it would not be strengthened by Clinical Translation track assessment either.

Overall recommendation: reject, with an invitation to resubmit to a future venue once the baseline, ablation, and novelty-framing issues are addressed.
