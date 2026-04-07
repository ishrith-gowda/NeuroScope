---
title: Multi-Site Brain MRI Harmonization
summary: >
  Developing deep learning methods to standardize brain MRI data acquired across
  different scanners and institutions, enabling reliable multi-site neuroimaging
  studies.
tags:
  - Medical Imaging
  - Deep Learning
  - Domain Adaptation
date: '2025-09-01T00:00:00Z'
external_link: ''
links:
  - icon: github
    icon_pack: fab
    name: Code
    url: https://github.com/ishrith-gowda/SA-CycleGAN-2.5D
  - icon: file-alt
    icon_pack: fas
    name: Paper
    url: https://arxiv.org/abs/2603.17219
---

## Problem

Multi-site brain MRI studies aggregate data from different scanners and acquisition protocols. These scanner-induced domain shifts introduce non-biological variability that confounds downstream analyses such as tumor segmentation, volumetric measurements, and longitudinal tracking. Harmonization aims to remove these technical variations while preserving the underlying anatomical and pathological information.

## Approach

**SA-CycleGAN-2.5D** is a self-attention CycleGAN architecture designed specifically for multi-modal brain MRI harmonization:

- **2.5D Processing**: Inputs 3 adjacent axial slices (12 channels: 3 slices x 4 modalities) and outputs the harmonized center slice (4 channels), capturing inter-slice anatomical context without the memory cost of full 3D processing.
- **Multi-Scale Self-Attention**: Self-attention layers in the generator bottleneck capture long-range spatial dependencies critical for preserving anatomical structure across large brain regions.
- **CBAM Attention**: Channel and spatial attention modules in the encoder/decoder paths for adaptive feature selection.
- **Multi-Scale Discriminator**: Spectral-normalized PatchGAN operating at two scales for multi-frequency analysis.

## Extensions (Journal)

The journal extension introduces four additional contributions:

1. **Compression-Aware Harmonization**: Joint harmonization and lossy compression at 8.4 bits per element, achieving SSIM 0.970 -- enabling efficient storage and transmission of harmonized multi-site data.
2. **Multi-Domain Translation**: StarGAN-inspired extension handling 4+ scanner domains simultaneously with domain classification and reconstruction losses.
3. **Downstream Segmentation Transfer**: Systematic evaluation of harmonization impact on cross-site U-Net tumor segmentation, revealing important gaps between perceptual quality and task-level utility.
4. **Federated Learning**: FedAvg-based training across institutional silos without sharing patient data, achieving SSIM 0.998 -- demonstrating that privacy-preserving harmonization does not sacrifice quality.

## Datasets

- **BraTS-TCGA-GBM**: 84 subjects from The Cancer Imaging Archive, multi-institutional acquisition
- **UPenn-GBM**: 515 subjects from the University of Pennsylvania, single-institution acquisition
- Both datasets include 4 MRI modalities (T1, T1ce, T2, FLAIR) with expert tumor segmentation labels

## Key Results

| Experiment | Metric | Value |
|---|---|---|
| Federated Harmonization | SSIM | 0.998 |
| Compression-Harmonization | SSIM | 0.970 |
| Compression Rate | Bits per Element | 8.4 |
| Multi-Domain | Classification Loss | 0.027 |
| Cross-Site Segmentation (Raw) | Mean Dice | 0.777 |
