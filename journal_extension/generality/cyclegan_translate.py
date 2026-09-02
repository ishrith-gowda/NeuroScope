#!/usr/bin/env python3
"""standalone CycleGAN GTA5->Cityscapes translation (the LEARNED-GAN arm of the generality benchmark).

reconstructs the canonical junyanz ResnetGenerator (9 blocks, instance-norm, reflection pad) and loads a
trained G_A checkpoint directly -- no dependency on the full pytorch-CycleGAN-and-pix2pix repo, so the whole
benchmark is self-contained and reproducible on any machine (mps / cuda / cpu).

usage: cyclegan_translate.py --src <gta5_images> --ckpt <..._net_G_A.pth> --out <dir> [--size 256 --limit N]
"""

from __future__ import annotations

import argparse
import functools
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_bias):
        super().__init__()
        block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(dim),
        ]
        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """exact junyanz ResnetGenerator (n_blocks=9), so a stock G_A state_dict loads strict=True."""

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super().__init__()
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        use_bias = True  # instance norm -> conv carries the bias
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
        n_down = 2
        for i in range(n_down):
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]
        mult = 2**n_down
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer, use_bias)]
        for i in range(n_down):
            mult = 2 ** (n_down - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def load_generator(ckpt: str, device: str) -> ResnetGenerator:
    net = ResnetGenerator()
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    # junyanz drops instance-norm running stats; our affine=False norm has none either -> strict must pass
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(
        description="standalone CycleGAN translation for the generality benchmark"
    )
    ap.add_argument("--src", required=True, help="GTA5 images dir")
    ap.add_argument("--ckpt", required=True, help="trained ..._net_G_A.pth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256, help="cyclegan train resolution")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    a = ap.parse_args()

    dev = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    net = load_generator(a.ckpt, dev)
    print(
        f"device={dev} ckpt={os.path.basename(a.ckpt)} size={a.size} -> loaded G_A strict=True",
        flush=True,
    )

    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.src, "*.png")))
    if a.limit:
        files = files[: a.limit]

    n = 0
    for p in files:
        img = Image.open(p).convert("RGB").resize((a.size, a.size), Image.BICUBIC)
        x = (
            torch.from_numpy(np.asarray(img).astype(np.float32) / 127.5 - 1.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(dev)
        )
        y = net(x)[0].clamp(-1, 1).cpu().permute(1, 2, 0).numpy()
        out = ((y + 1.0) * 127.5).round().astype(np.uint8)
        Image.fromarray(out).save(os.path.join(a.out, os.path.basename(p)))
        n += 1
        if n % 50 == 0:
            print(f"  cyclegan {n}/{len(files)}", flush=True)
    print(f"CycleGAN-translated {n} images -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
