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
    ap = argparse.ArgumentParser(
        description="SDEdit diffusion translation for the generality benchmark"
    )
    ap.add_argument("--src", required=True, help="GTA5 images dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=0.55, help="SDEdit noise strength (0..1)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="base seed; image i uses seed+i so the noise is "
        "identical across a strength sweep (strength is the only varying factor)",
    )
    ap.add_argument(
        "--prompt",
        default=PROMPT,
        help='target-domain prompt; pass "" for the empty-prompt '
        "ablation (isolates the img2img prior from text guidance)",
    )
    ap.add_argument("--neg", default=NEG, help="negative prompt")
    ap.add_argument("--resume", action="store_true", help="skip images whose output already exists")
    a = ap.parse_args()

    dev = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float32 if dev == "mps" else torch.float16
    print(
        f"device={dev} dtype={dtype} model={a.model} strength={a.strength} steps={a.steps}",
        flush=True,
    )
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        a.model, torch_dtype=dtype, safety_checker=None
    )
    pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)

    os.makedirs(a.out, exist_ok=True)
    gen = torch.Generator(device="cpu")
    files = sorted(glob.glob(os.path.join(a.src, "*.png")))
    if a.limit:
        files = files[: a.limit]
    print(f"prompt={a.prompt!r} neg={a.neg!r} seed={a.seed} n_files={len(files)}", flush=True)

    n = 0
    skipped = 0
    for i, p in enumerate(files):
        dst = os.path.join(a.out, os.path.basename(p))
        if a.resume and os.path.exists(dst):  # resumable: unreliable local compute can hang mid-run
            skipped += 1
            continue
        img = Image.open(p).convert("RGB").resize((a.size, a.size), Image.BICUBIC)
        gen.manual_seed(a.seed + i)  # image i gets identical noise across every strength/prompt run
        out = pipe(
            prompt=a.prompt,
            negative_prompt=a.neg,
            image=img,
            strength=a.strength,
            num_inference_steps=a.steps,
            guidance_scale=7.5,
            generator=gen,
        ).images[0]
        out.save(dst)
        n += 1
        if n % 20 == 0:
            print(f"  sdedit {n}/{len(files)} (skipped {skipped} existing)", flush=True)
    print(f"SDEdit-translated {n} images ({skipped} resumed/skipped) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
