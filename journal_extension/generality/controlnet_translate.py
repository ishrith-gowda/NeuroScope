#!/usr/bin/env python3
"""structure-preserving diffusion baseline: ControlNet-Canny GTA5->Cityscapes translation.

the reviewer question our thesis invites is "why not condition the diffusion model on structure?". this is
the empirical answer: SD1.5 + ControlNet conditioned on the source image's Canny edges regenerates a
Cityscapes-styled image that must follow the source geometry. it injects an explicit, NON-distributional
(edge/structure) constraint -- exactly the ingredient our thesis says is required to retain utility -- so
it is expected to sit at higher mIoU (structure kept) and more modest FID gain than unconstrained SDEdit,
near the non-learned baseline. running it makes the argument empirical rather than rhetorical.

usage: controlnet_translate.py --src <gta5 imgs> --out <dir> [--seed 42 --steps 30 --size 512
       --cond_scale 1.0 --limit N --resume]
"""

from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image

PROMPT = "a photograph of a real urban street scene, dashcam view, natural daylight, high detail, realistic"
NEG = "cartoon, painting, cgi, rendered, video game, illustration, blurry, low quality"
SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CN = "lllyasviel/sd-controlnet-canny"


def canny(img: Image.Image, lo: int = 100, hi: int = 200) -> Image.Image:
    e = cv2.Canny(np.asarray(img), lo, hi)
    return Image.fromarray(np.stack([e, e, e], axis=-1))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ControlNet-Canny structure-preserving diffusion baseline"
    )
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cond_scale", type=float, default=1.0, help="controlnet conditioning scale")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    dev = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    dtype = torch.float16 if dev == "cuda" else torch.float32
    cn = ControlNetModel.from_pretrained(CN, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD, controlnet=cn, torch_dtype=dtype, safety_checker=None
    )
    pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)
    print(f"device={dev} dtype={dtype} cond_scale={a.cond_scale} seed={a.seed}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    gen = torch.Generator(device="cpu")
    files = sorted(glob.glob(os.path.join(a.src, "*.png")))
    if a.limit:
        files = files[: a.limit]

    n = skipped = 0
    for i, p in enumerate(files):
        dst = os.path.join(a.out, os.path.basename(p))
        if a.resume and os.path.exists(dst):
            skipped += 1
            continue
        img = Image.open(p).convert("RGB").resize((a.size, a.size), Image.BICUBIC)
        gen.manual_seed(a.seed + i)
        out = pipe(
            prompt=PROMPT,
            negative_prompt=NEG,
            image=canny(img),
            num_inference_steps=a.steps,
            guidance_scale=7.5,
            controlnet_conditioning_scale=a.cond_scale,
            height=a.size,
            width=a.size,
            generator=gen,
        ).images[0]
        out.save(dst)
        n += 1
        if n % 50 == 0:
            print(f"  controlnet {n}/{len(files)} (skipped {skipped})", flush=True)
    print(f"ControlNet-Canny translated {n} images ({skipped} skipped) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
