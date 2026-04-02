---
license: other
license_name: tcia-data-usage-agreement
license_link: https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/
task_categories:
  - image-to-image
tags:
  - medical-imaging
  - brain-mri
  - mri-harmonization
  - glioma
  - neuroimaging
  - multi-site
  - domain-adaptation
  - radiomics
  - brats
  - nifti
  - multi-modal
language:
  - en
size_categories:
  - 10K<n<100K
---

# MRI-Harmonization-BraTS-UPenn

[![arXiv](https://img.shields.io/badge/arXiv-2603.17219-b31b1b.svg)](https://arxiv.org/abs/2603.17219)
[![GitHub](https://img.shields.io/badge/GitHub-SA--CycleGAN--2.5D-black?logo=github)](https://github.com/ishrith-gowda/SA-CycleGAN-2.5D)
[![TCIA](https://img.shields.io/badge/Source-TCIA-orange)](https://www.cancerimagingarchive.net)

preprocessed multi-site brain MRI dataset used to train and evaluate [SA-CycleGAN-2.5D](https://arxiv.org/abs/2603.17219), a self-attention GAN for unpaired cross-scanner MRI harmonization.

**paper:** [SA-CycleGAN-2.5D: Self-Attention CycleGAN with Tri-Planar Context for Multi-Site MRI Harmonization](https://arxiv.org/abs/2603.17219)  
**model:** [ishrith-gowda/SA-CycleGAN-2.5D](https://huggingface.co/ishrith-gowda/SA-CycleGAN-2.5D)

---

## dataset overview

| property | value |
|---|---|
| subjects | 654 glioma patients |
| total 2D slices (triplets) | ~52,000 |
| modalities | T1, T1ce, T2, FLAIR (4 per subject) |
| input format | 12-channel 2.5D triplets [3 slices × 4 modalities] |
| image size | 128 × 128 px (center-cropped) |
| domains | 2 institutional sites (BraTS, UPenn-GBM) |
| annotation | unpaired — no cross-site slice correspondence |

---

## source collections

### domain A: BraTS-TCGA-GBM
- **source:** [The Cancer Imaging Archive — TCGA-GBM](https://www.cancerimagingarchive.net/collection/tcga-gbm/)
- **subjects:** 88 glioblastoma patients
- **scanners:** multi-institutional, heterogeneous (1.5T and 3T, multiple vendors)
- **citation:** Bakas et al., 2017; Menze et al., 2015
- **access:** public, TCIA data use agreement

### domain B: UPenn-GBM
- **source:** [The Cancer Imaging Archive — UPenn-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/)
- **subjects:** 566 glioblastoma patients
- **scanners:** 3T Siemens TrioTim (primary), 3T other, 1.5T
- **citation:** Bakas et al., 2022
- **access:** public, TCIA data use agreement

---

## multi-domain scanner split

for the multi-domain AdaIN journal extension, UPenn-GBM subjects are sub-divided by scanner:

| domain id | domain name | subjects | slices (~) |
|---|---|---|---|
| 0 | brats (multi-institutional) | 88 | 6,538 |
| 1 | upenn_3t_triotim | 434 | 32,266 |
| 2 | upenn_3t_other | 65 | 4,863 |
| 3 | upenn_15t | 67 | 4,990 |
| **total** | | **654** | **~48,657** |

---

## preprocessing pipeline

all volumes preprocessed using the following pipeline (see `neuroscope/preprocessing/` in the [GitHub repo](https://github.com/ishrith-gowda/SA-CycleGAN-2.5D)):

1. **skull stripping** — FSL BET or pre-stripped (BraTS volumes are pre-stripped)
2. **co-registration** — T1ce as reference; T1, T2, FLAIR rigidly registered
3. **MNI152 atlas registration** — affine registration to 1mm isotropic MNI152 space
4. **intensity normalization** — z-score per modality per subject, clipped at ±3σ, rescaled to [-1, 1]
5. **slice extraction** — axial slices, excluding top/bottom 10% (non-brain), minimum foreground threshold
6. **2.5D triplet construction** — for each valid center slice idx, stack [idx-1, idx, idx+1] across all 4 modalities → [12, H, W] tensor
7. **center crop** — 128 × 128 px

---

## data splits

| split | subjects (A) | subjects (B) | slices (~) |
|---|---|---|---|
| train | 70 | 453 | 41,870 |
| val | 9 | 57 | 5,230 |
| test | 9 | 56 | 5,230 |

splits are subject-level (no slice from a test subject appears in train/val). splits stored in `data/metadata/domain_split.json` and `data/metadata/multi_domain_split.json` in the repository.

---

## format

```
subject_dir/
├── t1.nii.gz          # T1-weighted
├── t1ce.nii.gz        # T1 post-contrast
├── t2.nii.gz          # T2-weighted
└── flair.nii.gz       # T2-FLAIR
```

preprocessed volumes: float32 NIfTI, shape [H, W, D], intensity range [-1, 1].

---

## usage

```python
from neuroscope.data.dataset import MRIHarmonizationDataset

dataset = MRIHarmonizationDataset(
    brats_dir="path/to/preprocessed/brats",
    upenn_dir="path/to/preprocessed/upenn",
    split="train",
    image_size=128,
)

sample = dataset[0]
# sample["input"]     — torch.Tensor [12, 128, 128], 2.5D triplet
# sample["target"]    — torch.Tensor [4, 128, 128], center slice
# sample["domain"]    — str, "A" or "B"
```

see the [official repository](https://github.com/ishrith-gowda/SA-CycleGAN-2.5D) for full preprocessing scripts and dataloader.

---

## access & license

the raw source data is publicly available from TCIA under the [TCIA Data Usage Policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/). users must agree to the TCIA terms before using this data.

this preprocessed dataset card and the SA-CycleGAN-2.5D model are released under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) for research use only.

---

## citation

if you use this dataset or the associated model, please cite:

```bibtex
@article{gowda2026sacyclegan25d,
  title={SA-CycleGAN-2.5D: Self-Attention CycleGAN with Tri-Planar Context for Multi-Site MRI Harmonization},
  author={Gowda, Ishrith and Liu, Chunwei},
  journal={arXiv preprint arXiv:2603.17219},
  year={2026},
  url={https://arxiv.org/abs/2603.17219},
  doi={10.48550/arXiv.2603.17219}
}
```

also cite the original TCIA source collections:
```bibtex
@article{bakas2017advancing,
  title={Advancing the cancer genome atlas glioma MRI collections with expert segmentation labels and radiomic features},
  author={Bakas, Spyridon and others},
  journal={Scientific data},
  volume={4},
  number={1},
  pages={1--13},
  year={2017}
}

@article{bakas2022university,
  title={The university of pennsylvania glioblastoma (UPenn-GBM) cohort: advanced MRI, clinical, genomics, \& radiomics},
  author={Bakas, Spyridon and others},
  journal={Scientific data},
  volume={9},
  number={1},
  pages={453},
  year={2022}
}
```
