# reproducibility

how to reproduce the SA-CycleGAN-2.5D results from a clean checkout.

## environment

the project targets python 3.11 (`.python-version`). dependencies are declared in
`pyproject.toml` and pinned exactly in `uv.lock`.

```bash
git clone https://github.com/ishrith-gowda/SA-CycleGAN-2.5D.git
cd SA-CycleGAN-2.5D

# reproducible environment (recommended)
uv sync
# or, with pip:
python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

## determinism

- the global random seed is **42** (numpy, torch, and cuda); see
  `neuroscope/tests/conftest.py` and the `seed` parameters on the datasets.
- gpu training is not bit-exact across different hardware/driver versions; reported
  metrics are means over a fixed test split using the seeds above.

## hardware

the reported results were produced on a single **NVIDIA A100 80GB** (PyTorch +
CUDA). cpu-only execution is supported for the unit tests and small smoke runs, but
not for full training.

## data

see [data availability](DATA.md). download and preprocess the two TCIA cohorts, then
point the configuration files at the preprocessed directories.

## canonical pipeline (journal extension)

the journal-extension experiments live under `journal_extension/`. the primary
harmonization model — **extension a**, a hybrid PatchNCE + cycle objective with a
`lambda_nce` sweep whose optimum is around 0.5 — is reproduced via:

```bash
# train (configuration selects the lambda_nce value and data paths)
python journal_extension/scripts/train_hybrid_nce.py --help

# evaluate harmonization: masked windowed ssim, mmd, fid/kid, domain-classifier
python journal_extension/scripts/eval_harmonization_correct.py --help
```

- configurations: `journal_extension/configs/*.yaml`
- aggregated result tables / figures: `journal_extension/results/`, `journal_extension/figures/`
- exact reported metrics and statistical tests: the manuscript sources under
  `journal_extension/manuscript/`.

## tests and local gate

```bash
pytest tests/   # metric unit tests (the ci gate)
make ci         # full local gate: ruff format + ruff check + mypy + tests
```
