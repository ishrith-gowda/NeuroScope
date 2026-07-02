# canonical ext a results (committed)

these json files are the canonical evaluation outputs backing the extension a (patchnce-hybrid
harmonization) manuscript. they were previously only present in the gitignored `cluster_backup/`,
which made the manuscript's headline numbers non-reproducible from the repo. they are committed here
so every reported number traces to a committed file + the script that produced it.

## harmonization_eval/eval_lambda{0.0,0.1,0.5,1.0,2.0}.json
produced by `journal_extension/scripts/eval_harmonization_correct.py` on the five λ_NCE arms
(single seed 42, 200 epochs). each file contains per-modality FID/KID (A2B, B2A × FLAIR/T1/T1ce/T2),
the site domain-classifier accuracy (raw vs harmonized, chance 0.5), feature-space MMD, and the
brain-masked windowed SSIM (structure + cycle). headline fields: `fid_A2B_avg`,
`masked_windowed_ssim_structure_A2B.mean`, `domain_classifier.acc_harmonized`.

| λ_NCE | fid_A2B_avg | struct_ssim (A2B) | domain-clf (harmonized) |
|------|-------------|-------------------|--------------------------|
| 0.0  | 51.77 | 0.2892 | 0.4047 |
| 0.1  | 51.52 | 0.3219 | 0.4843 |
| 0.5  | 47.28 | 0.4193 | 0.5050 |
| 1.0  | 48.52 | 0.4198 | 0.5032 |
| 2.0  | 48.17 | 0.4395 | 0.5008 |

interior FID optimum at λ=0.5 (51.8 → 47.3 → 48.2); structure SSIM rises monotonically 0.29 → 0.44.

## downstream_corrected/lambda{0.0,0.5}_downstream_corrected.json
produced by `journal_extension/scripts/eval_downstream_corrected.py` (the corrected pipeline that
harmonizes via `harmonize_for_downstream.py` with the right channel order / normalization / 2.5D
slice assembly — superseding the earlier buggy `eval_downstream.generate_harmonized_data`). keys:
`A_on_rawA`, `A_on_rawB`, `A_on_harmB`, `B_on_rawB`, `B_on_rawA`, `B_on_harmA`, each with
`dice_mean_foreground_mean`, per-region (wt/tc/et) means, and `hd95_mean`.

headline (λ=0.5): cross-site segmentation transfer **degrades** after harmonization —
A→B 0.776 → 0.363 Dice (−53.3%, HD95 5.1 → 31.4) and B→A 0.765 → 0.479 (−37.4%, HD95 4.1 → 12.1).
this is the honest, corrected result (see `journal_extension/analysis/downstream_eval_analysis.md`).

## regenerate
```
python journal_extension/scripts/generate_ext_a_results.py \
  --eval_dir journal_extension/results/patchnce/harmonization_eval \
  --downstream_json journal_extension/results/patchnce/downstream_corrected/lambda0.5_downstream_corrected.json
```
