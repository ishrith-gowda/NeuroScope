# generality benchmark (multi-seed): fidelity vs utility, mean +/- std

GTA5->Cityscapes, N per condition, frozen SegFormer-b4 + clean-fid; SDEdit over seeds.

| condition | family | FID (down) | mIoU (up) | seeds |
|---|---|---:|---:|---:|
| raw | baseline | 137.15 | 0.3803 | 1 (det.) |
| colormatch | non-learned | 165.88 | 0.3626 | 1 (det.) |
| cyclegan | learned-GAN | 65.49 | 0.1666 | 1 (det.) |
| SDEdit 0.20 | learned-diffusion | 145.10 ± 1.82 | 0.3229 ± 0.0028 | 3 |
| SDEdit 0.30 | learned-diffusion | 133.14 ± 2.13 | 0.2887 ± 0.0024 | 3 |
| SDEdit 0.40 | learned-diffusion | 116.44 ± 1.06 | 0.2370 ± 0.0031 | 3 |
| SDEdit 0.50 | learned-diffusion | 100.25 ± 1.37 | 0.1936 ± 0.0048 | 3 |
| SDEdit 0.60 | learned-diffusion | 90.75 ± 2.01 | 0.1495 ± 0.0012 | 3 |
| SDEdit 0.70 | learned-diffusion | 88.01 ± 0.37 | 0.1062 ± 0.0007 | 3 |
| SDEdit 0.50 empty | learned-diffusion | 154.18 ± 1.85 | 0.1695 ± 0.0027 | 3 |
