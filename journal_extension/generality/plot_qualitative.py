#!/usr/bin/env python3
"""qualitative figure: source GTA5 scene vs each translation, side by side.

image-translation papers are expected to show the actual outputs, not only metrics. this lays out a few
scenes (rows) against every condition (columns) so a reader can see what "lower FID, lower mIoU" looks
like: the learned translations get more Cityscapes-like while visibly discarding/altering the scene
structure the segmenter needs. expects files named <prefix>_<scene>.png in --samples.

usage: plot_qualitative.py --samples <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

COLS = [
    ("raw", "GTA5 (raw)"),
    ("colormatch", "color match\n(non-learned)"),
    ("cyclegan", "CycleGAN\n(learned-GAN)"),
    ("sdedit20", "SDEdit 0.20"),
    ("sdedit50", "SDEdit 0.50"),
    ("sdedit70", "SDEdit 0.70"),
    ("empty", "SDEdit 0.50\n(empty prompt)"),
    ("controlnet", "ControlNet-Canny"),
]
ROWS = ["00000", "00099", "00189"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or a.samples

    nr, nc = len(ROWS), len(COLS)
    fig, axes = plt.subplots(nr, nc, figsize=(1.55 * nc, 1.6 * nr))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"]})

    for i, scene in enumerate(ROWS):
        for j, (prefix, label) in enumerate(COLS):
            ax = axes[i, j]
            p = os.path.join(a.samples, f"{prefix}_{scene}.png")
            if os.path.exists(p):
                # downscale for a compact, repo-friendly figure (cells render small anyway)
                ax.imshow(Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(label, fontsize=8.5, pad=4)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.005, wspace=0.04, hspace=0.04)
    fig.suptitle(
        "GTA5→Cityscapes translations: learned outputs look more real while altering scene structure",
        fontsize=10.5,
        y=0.985,
    )
    # photographic grid -> jpeg keeps it compact (the pdf/png embed images losslessly and get large)
    fig.savefig(os.path.join(out, "qualitative.png"), bbox_inches="tight", dpi=110)
    fig.savefig(
        os.path.join(out, "qualitative.jpg"),
        bbox_inches="tight",
        dpi=110,
        pil_kwargs={"quality": 85},
    )
    print(f"wrote qualitative.png / .jpg -> {out}")


if __name__ == "__main__":
    main()
