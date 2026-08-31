# Reproducing the generality benchmark ("fidelity is not utility")

Every number in the paper regenerates from the commands below. Seeds are fixed and logged; the frozen
evaluators are referenced by checkpoint id; FID uses clean-fid at a fixed N.

## Environment
- **Local (Apple MPS or CPU):** `python -m venv .venv && .venv/bin/pip install torch torchvision diffusers==0.40.0
  transformers clean-fid opencv-python-headless matplotlib datasets scikit-image`. Set
  `HF_HOME` to a large disk before any download.
- **GPU node (the publication run):** `scripts/node/node_setup.sh` (installs the CUDA-12.6 torch build +
  the stack; the default wheel is cu130 and too new for a 12.6 driver — the script pins cu126).

## Data
`journal_extension/generality/prep_generality_data.py --out <dir> --n_gta5 N --n_city N` streams the
GTA5→Cityscapes subsets from the Hugging Face Hub (public, ungated). Cityscapes provides 500 validation
images, used as the fixed FID reference.

## Frozen evaluators (never retrained)
- Driving utility: `nvidia/segformer-b4-finetuned-cityscapes-1024-1024` (mIoU, 19 classes).
- Fidelity: clean-fid (Inception), CPU on Mac / CUDA on the node — features are architecture-deterministic.
- Medical utility: nnU-Net trained on BraTS, frozen (Dice); results in `cluster_backup/`.
- CycleGAN generator: `cluster_backup/node5_generality/checkpoints/100_net_G_A.pth` (100-epoch G_A).

## Regenerating the paper artifacts

### Table 1 + Figure 1 (driving, N=1000, 3 seeds) — the publication result
On the GPU node, with the repo scripts + checkpoint staged in `~/gen`:
```
N=1000 SEEDS="42 43 44" STRENGTHS="20 30 40 50 60 70" EMPTY_S=50 bash scripts/node/node_run_sweep.sh
N=1000 SEEDS="42 43 44" bash scripts/node/node_run_controlnet.sh   # ControlNet-Canny baseline
```
Then aggregate + plot (locally or on the node), pointing at the results dir with the 24+3 per-seed jsons:
```
python journal_extension/generality/aggregate_seeds.py --results <results_dir>     # -> benchmark_seeds.{md,json}
python journal_extension/generality/plot_frontier_seeds.py --seeds-json <results_dir>/benchmark_seeds.json \
    --out journal_extension/generality/figures                                      # -> frontier_seeds.{pdf,png}
```
Result set is version-controlled at `journal_extension/generality/results_n1000/`.

### Local N=200 replication (single machine, Apple MPS)
```
bash scripts/run_generality_benchmark.sh          # raw / colormatch / cyclegan / sdedit sweep + empty
bash scripts/run_sdedit_sweep.sh                  # sdedit strengths (resumable)
python journal_extension/generality/aggregate_generality.py --results <results_n200>
python journal_extension/generality/plot_frontier.py --results <results_n200> --out <figs>
```

### Table 2 (medical) — frozen nnU-Net
The nnU-Net downstream Dice results are in `cluster_backup/node3_experiments/ext_a_nnunet/results/`
(`upenn_raw.json`, `upenn_harm.json`, `histogram.json`, `symmetric.json`, `brats_test.json`) and the
RHUH external result in `cluster_backup/experiments/ext_a_rhuh/eval/rhuh_external.json`.

## The manuscript
`journal_extension/generality/iclr_paper/` — `main.tex` + `references.bib` + `arxiv.sty` + the figure.
Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Tests
`pytest tests/test_generality.py` covers the aggregator (family/ΔmIoU/table assembly), strength parsing,
and the CycleGAN generator reconstruction (shape + output range).
