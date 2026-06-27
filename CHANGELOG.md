# changelog

all notable changes to this project are documented here. the format is based on
[keep a changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased]

### added
- professional repository foundation: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, `CHANGELOG.md`, issue/pull-request templates, `CODEOWNERS`,
  `.python-version`, and a codeql analysis workflow.
- branch protection on `main` (pull request + green ci required, no direct pushes).

### changed
- relicensed the project from cc by-nc-nd 4.0 to the mit license.

## [0.2.0] - 2026-06

### added
- journal extensions: hybrid patchnce–cycle loss (ext a), neural compression
  (ext b), multi-domain adain harmonization (ext c), downstream segmentation
  evaluation (ext d), and federated harmonization (ext e).
- corrected harmonization evaluation suite (masked windowed ssim, mmd, fid/kid,
  domain-classifier confusion) with per-subject outputs and paired statistics.
- boundary-sharpness mechanism analysis and multi-seed error bars.

### changed
- consolidated linting/formatting on `ruff` (replacing black + isort + flake8);
  modernized ci (pinned ruff, metric test gate); migrated `setup.py` to a pep 621
  shim and fixed the editable install.
- repo-wide formatting and lint cleanup (14,155 -> 0 lint errors).

## [0.1.0] - 2026-03

### added
- initial release of SA-CycleGAN-2.5D: self-attention cyclegan with tri-planar
  2.5d context for multi-site brain mri harmonization (brats-tcga <-> upenn-gbm).
- preprocessing pipeline, training, and evaluation metrics (fid, kid, ssim, mmd).
- arxiv preprint (arXiv:2603.17219).
