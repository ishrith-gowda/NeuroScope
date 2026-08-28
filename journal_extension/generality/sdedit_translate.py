#!/usr/bin/env python3
"""SDEdit diffusion translation (GTA5 -> photorealistic / Cityscapes-like) via a PRETRAINED Stable
Diffusion img2img model -- the DIFFUSION arm of the generality benchmark, i.e. a SECOND, non-GAN
generative family.

rationale: our CycleGAN result shows learned generative harmonization improves FID but destroys
downstream mIoU. if a completely different generative family (a diffusion model, via SDEdit) shows the
SAME dissociation, the failure tracks the learned-generative OBJECTIVE, not the CycleGAN architecture --
the multi-family evidence a top-ML (ICLR/NeurIPS) paper needs. (SDEdit: Meng et al., ICLR 2022;
inference-only, no training.)

usage: sdedit_translate.py --src <gta5_images> --out <dir> [--strength 0.55 --steps 30 --limit N]
"""

from __future__ import annotations

import argparse
import glob
import os

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

PROMPT = "a photograph of a real urban street scene, dashcam view, natural daylight, high detail, realistic"
NEG = "cartoon, painting, cgi, rendered, video game, illustration, blurry, low quality"


def main() -> None:
    ap = argparse.ArgumentParser(description="SDEdit diffusion translation for the generality benchmark")
    ap.add_argument("--src", required=True, help="GTA5 images dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=0.55, help="SDEdit noise strength (0..1)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if dev == "mps" else torch.float16
    print(f"device={dev} dtype={dtype} model={a.model} strength={a.strength} steps={a.steps}", flush=True)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(a.model, torch_dtype=dtype, safety_checker=None)
    pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)

    os.makedirs(a.out, exist_ok=True)
    gen = torch.Generator(device="cpu").manual_seed(42)
    files = sorted(glob.glob(os.path.join(a.src, "*.png")))
    if a.limit:
        files = files[: a.limit]

    n = 0
    for p in files:
        img = Image.open(p).convert("RGB").resize((a.size, a.size), Image.BICUBIC)
        out = pipe(
            prompt=PROMPT,
            negative_prompt=NEG,
            image=img,
            strength=a.strength,
            num_inference_steps=a.steps,
            guidance_scale=7.5,
            generator=gen,
        ).images[0]
        out.save(os.path.join(a.out, os.path.basename(p)))
        n += 1
        if n % 20 == 0:
            print(f"  sdedit {n}/{len(files)}", flush=True)
    print(f"SDEdit-translated {n} images -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
