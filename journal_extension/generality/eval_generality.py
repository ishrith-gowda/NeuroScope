#!/usr/bin/env python3
"""frozen-SegFormer mIoU + clean-fid for the GTA5->Cityscapes generality benchmark.

for a given image dir (one condition: raw / cyclegan-translated / color-matched GTA5), compute:
  - 19-class mIoU vs GTA5 Cityscapes-format masks, under a FROZEN Cityscapes-pretrained SegFormer
    (input varied, weights fixed -> mIoU delta attributable to the translation; mirrors exp #3/#4)
  - FID vs Cityscapes images (clean-fid) -> the distributional-fidelity axis

usage: eval_generality.py --images <dir> --masks <gta5_masks> --cityscapes <city_imgs> \
         --out <json> --tag raw
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MID = "nvidia/segformer-b4-finetuned-cityscapes-1024-1024"
NCLS = 19


@torch.no_grad()
def _predict(model, proc, img: Image.Image, device: str) -> np.ndarray:
    inp = proc(images=img, return_tensors="pt").to(device)
    logits = model(**inp).logits  # [1,19,h/4,w/4]
    up = torch.nn.functional.interpolate(
        logits, size=(img.size[1], img.size[0]), mode="bilinear", align_corners=False
    )
    return up.argmax(1)[0].cpu().numpy().astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--cityscapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = SegformerForSemanticSegmentation.from_pretrained(MID).to(device).eval()
    proc = SegformerImageProcessor.from_pretrained(MID)

    inter = np.zeros(NCLS)
    union = np.zeros(NCLS)
    imgs = sorted(glob.glob(os.path.join(a.images, "*.png")))
    n = 0
    for p in imgs:
        mpath = os.path.join(a.masks, os.path.basename(p))
        if not os.path.exists(mpath):
            continue
        img = Image.open(p).convert("RGB")
        gt = np.asarray(Image.open(mpath))
        pred = _predict(model, proc, img, device)
        if pred.shape != gt.shape:
            pred = np.asarray(Image.fromarray(pred).resize(gt.shape[::-1], Image.NEAREST))
        valid = gt != 255
        for c in range(NCLS):
            pc = (pred == c) & valid
            gc = (gt == c) & valid
            inter[c] += float((pc & gc).sum())
            union[c] += float((pc | gc).sum())
        n += 1
        if n % 50 == 0:
            print(f"  segmented {n}/{len(imgs)}", flush=True)

    iou = inter / np.maximum(union, 1e-8)
    present = union > 0
    miou = float(iou[present].mean()) if present.any() else 0.0

    fid_val = None
    if glob.glob(os.path.join(a.cityscapes, "*.png")):
        from cleanfid import fid

        # clean-fid defaults its InceptionV3 extractor to cuda; force cpu so FID runs on this mac
        # (no cuda; inception features are architecture-deterministic, so cpu is exact and reliable)
        fid_val = float(
            fid.compute_fid(
                a.images, a.cityscapes, mode="clean", device=torch.device("cpu"), verbose=False
            )
        )

    out = {
        "tag": a.tag,
        "mIoU": miou,
        "n_images": n,
        "n_classes_present": int(present.sum()),
        "per_class_iou": {int(c): float(iou[c]) for c in range(NCLS)},
        "FID": fid_val,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    msg = f"[{a.tag}] mIoU {miou:.4f}  n={n}  classes_present={int(present.sum())}"
    if fid_val is not None:
        msg += f"  FID {fid_val:.2f}"
    print(msg + f"  -> {a.out}")


if __name__ == "__main__":
    main()
