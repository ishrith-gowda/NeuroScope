#!/usr/bin/env python3
"""prep the GTA5 -> Cityscapes generality benchmark data (non-medical replication of the
fidelity!=utility finding for the top-ML paper).

pulls HF-hosted mirrors (no registration gates) into plain image folders that the CycleGAN training,
FID, color-match baseline, and frozen-SegFormer mIoU eval all consume:
  gta5/images/*.png     GTA5 RGB (source domain B, analog of upenn)
  gta5/masks/*.png      GTA5 Cityscapes-format 19-class trainId masks (downstream GT)
  cityscapes/images/*.png  Cityscapes RGB (target domain A / FID reference / SegFormer home domain)

datasets: guimCC/gta5-cityscapes-labeling (image uint8 HxWx3 + mask uint8 HxW), tanganke/cityscapes
(image float32). streaming so we only pull the requested subset.

usage: prep_generality_data.py --out ~/data/generality --n_gta5 1000 --n_city 500
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from datasets import load_dataset
from PIL import Image


def _save_png(arr: np.ndarray, path: str) -> None:
    Image.fromarray(arr).save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_gta5", type=int, default=1000)
    ap.add_argument("--n_city", type=int, default=500)
    args = ap.parse_args()

    gi = os.path.join(args.out, "gta5", "images")
    gm = os.path.join(args.out, "gta5", "masks")
    ci = os.path.join(args.out, "cityscapes", "images")
    for d in (gi, gm, ci):
        os.makedirs(d, exist_ok=True)

    print(f"== GTA5 (guimCC/gta5-cityscapes-labeling, validation) x{args.n_gta5} ==")
    ds = load_dataset("guimCC/gta5-cityscapes-labeling", split="validation", streaming=True)
    n = 0
    for ex in ds:
        img = np.asarray(ex["image"], dtype=np.uint8)
        msk = np.asarray(ex["mask"], dtype=np.uint8)
        _save_png(img, os.path.join(gi, f"{n:05d}.png"))
        _save_png(msk, os.path.join(gm, f"{n:05d}.png"))
        n += 1
        if n % 200 == 0:
            print(f"  gta5 {n}/{args.n_gta5}")
        if n >= args.n_gta5:
            break
    print(f"  gta5 done: {n}")

    print(f"== Cityscapes (tanganke/cityscapes, validation) x{args.n_city} ==")
    dc = load_dataset("tanganke/cityscapes", split="validation", streaming=True)
    m = 0
    for ex in dc:
        arr = np.asarray(ex["image"], dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        if arr.max() <= 1.5:  # normalized floats -> [0,255]
            arr = arr * 255.0
        _save_png(np.clip(arr, 0, 255).astype(np.uint8), os.path.join(ci, f"{m:05d}.png"))
        m += 1
        if m >= args.n_city:
            break
    print(f"  cityscapes done: {m}")
    print(f"prepared -> {args.out} (gta5 {n} imgs+masks, cityscapes {m} imgs)")


if __name__ == "__main__":
    main()
