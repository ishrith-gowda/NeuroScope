---
title: 'SA-CycleGAN-2.5D: Self-Attention CycleGAN with Tri-Planar Context for Multi-Site MRI Harmonization'
authors:
  - admin
  - Chunwei Liu
date: '2026-03-17T00:00:00Z'
doi: '10.48550/arXiv.2603.17219'
publishDate: '2026-03-17T00:00:00Z'
publication_types: ['article']
publication: '*arXiv preprint arXiv:2603.17219*'
publication_short: 'arXiv 2026'
abstract: >
  Multi-site brain MRI studies suffer from scanner-induced domain shifts that
  confound downstream analyses. We propose SA-CycleGAN-2.5D, a self-attention
  CycleGAN with tri-planar 2.5D context for multi-site MRI harmonization. Our
  architecture processes three adjacent axial slices simultaneously across four
  MRI modalities (T1, T1ce, T2, FLAIR), producing harmonized center slices that
  preserve anatomical structure while reducing inter-site variability. Key
  innovations include multi-scale self-attention in the generator bottleneck for
  long-range spatial dependencies, CBAM attention in encoder/decoder paths for
  channel and spatial feature selection, and a multi-scale spectral-normalized
  PatchGAN discriminator. Evaluated on BraTS-TCGA-GBM (84 subjects, multi-site)
  and UPenn-GBM (515 subjects, single-site), SA-CycleGAN-2.5D achieves SSIM of
  0.998 for harmonized outputs. We further demonstrate federated learning
  extensions enabling privacy-preserving multi-institutional training without
  sharing patient data.
summary: >
  Self-attention CycleGAN with 2.5D tri-planar context for multi-site brain MRI
  harmonization. Achieves SSIM 0.998 on BraTS/UPenn-GBM datasets with federated
  learning extensions.
tags:
  - MRI Harmonization
  - CycleGAN
  - Self-Attention
  - Domain Adaptation
  - Medical Imaging
  - Federated Learning
featured: true
links:
  - name: arXiv
    url: https://arxiv.org/abs/2603.17219
  - name: Code
    url: https://github.com/ishrith-gowda/SA-CycleGAN-2.5D
  - name: Model
    url: https://huggingface.co/ishrith-gowda/SA-CycleGAN-2.5D
  - name: Dataset
    url: https://huggingface.co/datasets/ishrith-gowda/MRI-Harmonization-BraTS-UPenn
  - name: Demo
    url: https://huggingface.co/spaces/ishrith-gowda/SA-CycleGAN-2.5D-demo
url_pdf: https://arxiv.org/pdf/2603.17219
url_code: https://github.com/ishrith-gowda/SA-CycleGAN-2.5D
---

## Citation

```bibtex
@article{gowda2026sacyclegan25d,
  title={SA-CycleGAN-2.5D: Self-Attention CycleGAN with Tri-Planar Context
         for Multi-Site MRI Harmonization},
  author={Gowda, Ishrith and Liu, Chunwei},
  journal={arXiv preprint arXiv:2603.17219},
  year={2026},
  doi={10.48550/arXiv.2603.17219}
}
```
