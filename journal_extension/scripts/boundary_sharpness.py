#!/usr/bin/env python3
"""
boundary-sharpness analysis for ext a -- a direct test of the mechanism claim.

the paper argues cycle-consistency admits an over-smoothing equilibrium (blurred
tissue boundaries) that multi-layer patchnce regularizes. we test this directly:
on the SAME subjects, harmonized by the cycle-only (lambda=0.0) vs the hybrid
(lambda=0.5) model (identical 128x128 pipeline, so directly comparable), we
measure brain-masked edge sharpness per subject and run a paired wilcoxon test.

sharpness metrics (higher = sharper, i.e. less over-smoothing):
  - mean Sobel gradient magnitude within the brain
  - variance of the Laplacian (a standard focus/sharpness measure)

if the hybrid is significantly sharper than cycle-only, that is direct evidence
for the over-smoothing mechanism the paper proposes.

usage:
    python boundary_sharpness.py \
        --cycle_dir  ~/neuroscope/experiments/ext_a_downstream/lambda0.0/harm_upenn_to_brats \
        --hybrid_dir ~/neuroscope/experiments/ext_a_downstream/lambda0.5/harm_upenn_to_brats \
        --out ~/neuroscope/experiments/ext_a_results/boundary_sharpness.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.stats import wilcoxon

MODS = ["t1", "t1gd", "t2", "flair"]


def slice_sharpness(sl: np.ndarray) -> tuple[float, float]:
    m = sl > 0.01
    if m.sum() < 50:
        return np.nan, np.nan
    gx = ndimage.sobel(sl, axis=0)
    gy = ndimage.sobel(sl, axis=1)
    gmag = np.sqrt(gx * gx + gy * gy)
    lap = ndimage.laplace(sl)
    return float(gmag[m].mean()), float(lap[m].var())


def subject_sharpness(subj_dir: Path) -> tuple[float, float]:
    gms, lvs = [], []
    for mod in MODS:
        p = subj_dir / f"{mod}.nii.gz"
        if not p.exists():
            continue
        vol = nib.load(str(p)).get_fdata().astype(np.float32)
        sg, sl = [], []
        for s in range(vol.shape[2]):
            g, lv = slice_sharpness(vol[:, :, s])
            if not np.isnan(g):
                sg.append(g)
                sl.append(lv)
        if sg:
            gms.append(np.mean(sg))
            lvs.append(np.mean(sl))
    return (float(np.mean(gms)) if gms else np.nan, float(np.mean(lvs)) if lvs else np.nan)


def paired(hybrid: np.ndarray, cycle: np.ndarray) -> dict:
    mask = ~(np.isnan(hybrid) | np.isnan(cycle))
    h, c = hybrid[mask], cycle[mask]
    p = float(wilcoxon(h, c).pvalue) if len(h) > 1 and np.any(h != c) else float("nan")
    return {
        "cycle_only_mean": float(np.mean(c)),
        "cycle_only_std": float(np.std(c)),
        "hybrid_mean": float(np.mean(h)),
        "hybrid_std": float(np.std(h)),
        "hybrid_minus_cycle": float(np.mean(h - c)),
        "relative_pct": float(100 * np.mean(h - c) / max(abs(np.mean(c)), 1e-8)),
        "wilcoxon_p": p,
        "n_pairs": len(h),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle_dir", required=True, help="lambda0.0 harmonized dir")
    ap.add_argument("--hybrid_dir", required=True, help="lambda0.5 harmonized dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cyc, hyb = Path(args.cycle_dir), Path(args.hybrid_dir)
    subjects = sorted(
        {d.name for d in cyc.iterdir() if d.is_dir()}
        & {d.name for d in hyb.iterdir() if d.is_dir()}
    )
    cg, hg, cl, hl = [], [], [], []
    for s in subjects:
        a, b = subject_sharpness(cyc / s)
        c, d = subject_sharpness(hyb / s)
        cg.append(a)
        cl.append(b)
        hg.append(c)
        hl.append(d)

    res = {
        "n_subjects": len(subjects),
        "gradient_magnitude": paired(np.array(hg), np.array(cg)),
        "laplacian_variance": paired(np.array(hl), np.array(cl)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))

    g = res["gradient_magnitude"]
    lv = res["laplacian_variance"]
    print(f"n={res['n_subjects']} subjects (paired)")
    print(
        f"grad-mag: cycle {g['cycle_only_mean']:.4f} vs hybrid {g['hybrid_mean']:.4f} "
        f"(Δ={g['hybrid_minus_cycle']:+.4f}, {g['relative_pct']:+.1f}%, wilcoxon p={g['wilcoxon_p']:.2e})"
    )
    print(
        f"lap-var:  cycle {lv['cycle_only_mean']:.5f} vs hybrid {lv['hybrid_mean']:.5f} "
        f"(Δ={lv['hybrid_minus_cycle']:+.5f}, {lv['relative_pct']:+.1f}%, p={lv['wilcoxon_p']:.2e})"
    )
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
