# generality benchmark: fidelity (FID) vs downstream utility (frozen-SegFormer mIoU)

GTA5->Cityscapes, N per condition as listed, same frozen SegFormer-b4 evaluator + clean-fid.

| condition | family | FID (down) | mIoU (up) | dmIoU vs raw | N |
|---|---|---:|---:|---:|---:|
| CycleGAN (adversarial) | learned-GAN | 105.75 | 0.1263 | -0.2212 | 200 |
| SDEdit (diffusion) strength 0.70 | learned-diffusion | 122.02 | 0.0983 | -0.2491 | 200 |
| SDEdit (diffusion) strength 0.55 | learned-diffusion | 124.10 | 0.1575 | -0.1899 | 200 |
| SDEdit (diffusion) strength 0.40 | learned-diffusion | 143.08 | 0.1998 | -0.1477 | 200 |
| SDEdit (diffusion) strength 0.30 | learned-diffusion | 162.60 | 0.2529 | -0.0946 | 200 |
| raw GTA5 (no translation) | baseline | 165.79 | 0.3475 | +0.0000 | 200 |
| color match (histogram) | non-learned | 194.96 | 0.3185 | -0.0290 | 200 |
| SDEdit (diffusion) strength 0.55, empty prompt | learned-diffusion | 198.31 | 0.1324 | -0.2151 | 200 |

**reading:** learned translation (GAN + diffusion) lowers FID but drops mIoU; the non-learned baseline preserves mIoU. no diffusion strength should pareto-dominate the non-learned point.
