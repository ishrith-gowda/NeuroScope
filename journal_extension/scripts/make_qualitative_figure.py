#!/usr/bin/env python3
"""
publication qualitative montage for ext a: raw vs harmonized appearance across
modalities, with the tumor mask, for a few representative cases.

shows the paper's qualitative claim: harmonization preserves anatomy and retains
tumor contrast (anatomically faithful) while changing per-modality appearance
(e.g. FLAIR de-emphasized, T1 boosted) -- i.e. a style change, not corruption.
reuses BraTSSliceDataset so panels are exactly what the segmenter consumes, and
reads the harmonized volumes produced by harmonize_for_downstream.py.

usage:
    python make_qualitative_figure.py \
        --raw_dir   ~/neuroscope/preprocessed/upenn \
        --harm_dir  ~/neuroscope/experiments/ext_a_downstream/lambda0.5/harm_upenn_to_brats \
        --out       ~/neuroscope/experiments/ext_a_results/fig_ext_a_qualitative \
        --n_cases 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_downstream import BraTSSliceDataset

# BraTSSliceDataset channel order: t1=0, t1gd=1, flair=2, t2=3
COL_TITLES = ["raw FLAIR", "harmonized FLAIR", "raw T1", "harmonized T1", "tumor (GT)"]
PANEL_CH = [2, 2, 0, 0]  # flair(raw), flair(harm), t1(raw), t1(harm)


def pick_cases(raw_dir: str, harm_dir: str, n_cases: int, min_tumor: int = 80):
    cases = []
    for s in sorted(os.listdir(harm_dir)):
        try:
            dr = BraTSSliceDataset(raw_dir, [s], require_tumor=False)
            dh = BraTSSliceDataset(harm_dir, [s], require_tumor=False)
        except Exception:
            continue
        n = min(len(dr), len(dh))
        if n == 0:
            continue
        k = max(range(n), key=lambda i: int(dr[i]["mask"].sum()))
        if int(dr[k]["mask"].sum()) < min_tumor:
            continue
        cases.append((s, dr[k], dh[k]))
        if len(cases) == n_cases:
            break
    return cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--harm_dir", required=True)
    p.add_argument("--out", required=True, help="output path stem (writes .pdf and .png)")
    p.add_argument("--n_cases", type=int, default=3)
    args = p.parse_args()

    cases = pick_cases(args.raw_dir, args.harm_dir, args.n_cases)
    if not cases:
        raise SystemExit("no suitable tumor-bearing cases found")

    fig, ax = plt.subplots(len(cases), 5, figsize=(12.5, 2.6 * len(cases)))
    if len(cases) == 1:
        ax = ax[None, :]
    for ri, (_s, r, h) in enumerate(cases):
        img_r = r["image"].numpy()
        img_h = h["image"].numpy()
        mask = r["mask"].numpy()
        panels = [img_r[PANEL_CH[0]], img_h[PANEL_CH[1]], img_r[PANEL_CH[2]], img_h[PANEL_CH[3]]]
        for ci, panel in enumerate(panels):
            ax[ri, ci].imshow(panel, cmap="gray", vmin=0, vmax=1)
            ax[ri, ci].axis("off")
            if ri == 0:
                ax[ri, ci].set_title(COL_TITLES[ci], fontsize=11)
        ax[ri, 4].imshow(img_r[2], cmap="gray", vmin=0, vmax=1)
        ax[ri, 4].imshow(np.ma.masked_where(mask == 0, mask), alpha=0.45, cmap="autumn")
        ax[ri, 4].axis("off")
        if ri == 0:
            ax[ri, 4].set_title(COL_TITLES[4], fontsize=11)
        ax[ri, 0].text(
            -0.08,
            0.5,
            f"case {ri + 1}",
            transform=ax[ri, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
        )
    plt.tight_layout()
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=90, bbox_inches="tight")
    print(f"saved {out}.pdf/.png with {len(cases)} cases")


if __name__ == "__main__":
    main()
