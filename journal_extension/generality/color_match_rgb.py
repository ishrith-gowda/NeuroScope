#!/usr/bin/env python3
"""RGB color-matching baseline (non-learned) for the GTA5->Cityscapes generality benchmark.

the non-learned control: per-channel histogram-match each GTA5 image to a fixed Cityscapes reference,
so the colour/appearance moves toward Cityscapes (lowering FID) WITHOUT touching spatial structure --
the RGB analog of our medical masked per-modality histogram match. hypothesis (mirrors the medical
result): this preserves downstream mIoU while a learned CycleGAN degrades it.

usage: color_match_rgb.py --src <gta5_images> --ref_dir <cityscapes_images> --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from PIL import Image
from skimage.exposure import match_histograms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="GTA5 images dir")
    ap.add_argument("--ref_dir", required=True, help="Cityscapes images dir (reference)")
    ap.add_argument("--ref_id", default="", help="reference filename (default: first)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    refs = sorted(glob.glob(os.path.join(a.ref_dir, "*.png")))
    ref_path = os.path.join(a.ref_dir, a.ref_id) if a.ref_id else refs[0]
    ref = np.asarray(Image.open(ref_path).convert("RGB"))
    os.makedirs(a.out, exist_ok=True)
    print(f"reference: {os.path.basename(ref_path)}")

    n = 0
    for p in sorted(glob.glob(os.path.join(a.src, "*.png"))):
        img = np.asarray(Image.open(p).convert("RGB"))
        matched = match_histograms(img, ref, channel_axis=-1)
        Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8)).save(
            os.path.join(a.out, os.path.basename(p))
        )
        n += 1
        if n % 200 == 0:
            print(f"  matched {n}")
    print(f"color-matched {n} GTA5 images -> {a.out}")


if __name__ == "__main__":
    main()
