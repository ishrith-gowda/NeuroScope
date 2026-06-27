# data availability

SA-CycleGAN-2.5D is trained and evaluated on two publicly available, de-identified
glioblastoma MRI cohorts from The Cancer Imaging Archive (TCIA). no proprietary or
private data is required to reproduce the results.

## datasets

| role | dataset | source | scanners |
|------|---------|--------|----------|
| site a (source) | BraTS / TCGA-GBM | [TCIA: TCGA-GBM](https://www.cancerimagingarchive.net/collection/tcga-gbm/) | multi-institutional |
| site b (target) | UPenn-GBM | [TCIA: UPENN-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/) | 3T TrioTim, 3T other, 1.5T |

- **modalities:** T1, T1ce (post-contrast T1), T2, FLAIR.
- **cohort:** 654 subjects total (88 site a + 566 site b), ~52k axial slices after preprocessing.
- all imaging data are de-identified and distributed under their respective TCIA
  data use agreements. please review and accept each collection's terms on TCIA
  before downloading or using the data.

## access

1. create a TCIA account and review the data use agreement for each collection
   (links above).
2. download the collections with the [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images).
3. place the raw downloads outside the repository (they are large and are excluded
   by `.gitignore`).

## preprocessing

the harmonization model consumes preprocessed 2.5D volumes. the preprocessing
pipeline (see `scripts/01_data_preparation_pipeline/` and the README) performs:

1. dicom → nifti conversion,
2. skull stripping,
3. N4 bias-field correction,
4. affine registration to MNI152 space,
5. per-modality percentile intensity normalization,
6. 2.5D axial slice extraction — 3 adjacent slices × 4 modalities = a 12-channel
   generator input; the 4-channel center slice is the harmonization target.

the train / validation / test split is fixed by the global seed (42) and recorded
in the `journal_extension/configs/` yaml files.

## note for reviewers

raw and preprocessed imaging data are **not** committed to this repository (they
are large and governed by TCIA agreements). the code, configuration files, and
aggregated result tables needed to reproduce the analysis are committed; see
[reproducibility](REPRODUCIBILITY.md).
