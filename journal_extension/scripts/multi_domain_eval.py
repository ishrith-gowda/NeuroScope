#!/usr/bin/env python3
"""
quantitative multi-domain eval for ext c (n domains, adain/stargan model).

from a trained multi-domain checkpoint:
  (1) N x N FID matrix: FID(G(real_i -> j), real_j) over all modality slices
      (off-diagonal mean = cross-site harmonization quality).
  (2) domain-classifier confusion: a 4-class cnn trained on real center slices;
      after harmonizing every site to one reference domain, accuracy w.r.t. the
      ORIGINAL site falls toward chance (1/N) if site identity is removed.
  (3) zero-shot transfer: translate held-out-site images to a training domain and
      measure FID to that domain's real images.

generator output is tanh ([-1,1]); the dataset normalizes to [-1,1] too.

usage:
    python multi_domain_eval.py --checkpoint <ckpt> --config <yaml> \
        --output_dir <dir> --heldout_split <heldout_zeroshot.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_multi_domain import MultiDomainMRIDataset

from neuroscope.models.architectures.sa_cyclegan_25d_multidomain import (
    MultiDomainConfig,
    MultiDomainSACycleGAN25D,
)


def modality_uint8(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] [b,4,h,w] -> [b*4,3,h,w] uint8 (each modality as a grayscale rgb)."""
    x = ((x.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
    b, c, h, w = x.shape
    return x.reshape(b * c, 1, h, w).repeat(1, 3, 1, 1)


class DomainCNN(nn.Module):
    def __init__(self, in_ch: int = 4, n_domains: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, n_domains),
        )

    def forward(self, x):
        return self.net(x)


def load_generator(checkpoint: str, cfg: dict, n: int, dev: str):
    mcfg = MultiDomainConfig(
        n_domains=n,
        ngf=int(cfg.get("ngf", 64)),
        ndf=int(cfg.get("ndf", 64)),
        n_residual_blocks=int(cfg.get("n_residual_blocks", 9)),
        attention_layers=tuple(cfg.get("attention_layers", [3, 4, 5])),
        domain_embed_dim=int(cfg.get("domain_embed_dim", 64)),
        style_dim=int(cfg.get("style_dim", 256)),
    )
    model = MultiDomainSACycleGAN25D(mcfg).to(dev).eval()
    ck = torch.load(checkpoint, map_location=dev, weights_only=False)
    sd = ck.get("model_state_dict", ck)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded generator (missing={len(missing)} unexpected={len(unexpected)})")
    return model.generator


def collect(ds, max_per_domain: int):
    inp: dict[int, list] = defaultdict(list)
    cen: dict[int, list] = defaultdict(list)
    for s in ds:
        d = int(s["domain_id"])
        if len(cen[d]) >= max_per_domain:
            continue
        inp[d].append(s["input"])
        cen[d].append(s["target"])
    return {d: torch.stack(inp[d]) for d in inp if inp[d]}, {
        d: torch.stack(cen[d]) for d in cen if cen[d]
    }


@torch.no_grad()
def translate(G, x12: torch.Tensor, tgt: int, dev: str, bs: int = 32) -> torch.Tensor:
    outs = []
    for i in range(0, len(x12), bs):
        b = x12[i : i + bs].to(dev)
        t = torch.full((len(b),), tgt, dtype=torch.long, device=dev)
        outs.append(G(b, t).cpu())
    return torch.cat(outs)


@torch.no_grad()
def fid(real4: torch.Tensor, fake4: torch.Tensor, dev: str) -> float:
    from torchmetrics.image.fid import FrechetInceptionDistance

    m = FrechetInceptionDistance(feature=2048, normalize=False).to(dev)
    for i in range(0, len(real4), 64):
        m.update(modality_uint8(real4[i : i + 64]).to(dev), real=True)
    for i in range(0, len(fake4), 64):
        m.update(modality_uint8(fake4[i : i + 64]).to(dev), real=False)
    return float(m.compute())


def train_domain_clf(cen: dict, n: int, dev: str, epochs: int = 6) -> tuple[DomainCNN, float]:
    clf = DomainCNN(in_ch=4, n_domains=n).to(dev).train()
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    xs = torch.cat([cen[d] for d in sorted(cen)])
    ys = torch.cat([torch.full((len(cen[d]),), d, dtype=torch.long) for d in sorted(cen)])
    idx = torch.randperm(len(xs))
    cut = int(0.85 * len(xs))
    tr, va = idx[:cut], idx[cut:]
    for _ in range(epochs):
        for i in range(0, len(tr), 64):
            b = tr[i : i + 64]
            opt.zero_grad()
            loss = ce(clf(xs[b].to(dev)), ys[b].to(dev))
            loss.backward()
            opt.step()
    clf.eval()
    with torch.no_grad():
        pred = torch.cat(
            [clf(xs[va[i : i + 64]].to(dev)).argmax(1).cpu() for i in range(0, len(va), 64)]
        )
    acc = float((pred == ys[va]).float().mean())
    print(f"domain clf trained; real val acc {acc:.3f} (chance {1 / n:.3f})")
    return clf, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--heldout_split", default=None)
    ap.add_argument("--max_per_domain", type=int, default=400)
    ap.add_argument("--ref_domain", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    domain_names = cfg["domain_names"]
    n = len(domain_names)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    G = load_generator(args.checkpoint, cfg, n, dev)

    ds = MultiDomainMRIDataset(
        cfg["data_dirs"],
        domain_names,
        split="test",
        domain_split_file=cfg.get("domain_split_file"),
    )
    inp, cen = collect(ds, args.max_per_domain)
    for d in sorted(cen):
        print(f"  domain {domain_names[d]}: {len(cen[d])} eval slices")

    # (1) NxN FID matrix
    mat = np.full((n, n), np.nan)
    for j in sorted(cen):
        for i in sorted(inp):
            fake = translate(G, inp[i], j, dev)
            mat[j, i] = fid(cen[j], fake, dev)
        print(f"  FID matrix col tgt={domain_names[j]} done")
    off = mat[~np.eye(n, dtype=bool)]
    fid_res = {
        "domain_names": domain_names,
        "matrix_rows_target_cols_source": mat.tolist(),
        "offdiag_mean": float(np.nanmean(off)),
        "diag_mean": float(np.nanmean(np.diag(mat))),
    }
    (out / "matrix_fid.json").write_text(json.dumps(fid_res, indent=2))
    print(f"  FID off-diagonal mean (cross-site harmonization): {fid_res['offdiag_mean']:.1f}")

    # (2) domain-classifier confusion
    clf, real_acc = train_domain_clf(cen, n, dev)
    ref = args.ref_domain
    correct = total = 0
    with torch.no_grad():
        for i in sorted(inp):
            harm = translate(G, inp[i], ref, dev)
            for k in range(0, len(harm), 64):
                pred = clf(harm[k : k + 64].to(dev)).argmax(1).cpu()
                correct += int((pred == i).sum())
                total += len(pred)
    harm_acc = correct / max(total, 1)
    conf = {
        "real_val_acc": real_acc,
        "harmonized_to_ref_acc_wrt_original": harm_acc,
        "chance": 1.0 / n,
        "reference_domain": domain_names[ref],
    }
    (out / "domain_confusion.json").write_text(json.dumps(conf, indent=2))
    print(f"  domain-clf: real {real_acc:.3f} -> harmonized {harm_acc:.3f} (chance {1 / n:.3f})")

    # (3) zero-shot held-out -> reference domain
    if args.heldout_split and Path(args.heldout_split).exists():
        with open(args.heldout_split) as f:
            ho = json.load(f)
        ho_dirs = dict.fromkeys(ho, cfg["data_dirs"][domain_names[0]])
        ho_ds = MultiDomainMRIDataset(
            ho_dirs, list(ho.keys()), split="test", domain_split_file=args.heldout_split
        )
        ho_inp, _ = collect(ho_ds, args.max_per_domain)
        zs = {}
        for d in sorted(ho_inp):
            fake = translate(G, ho_inp[d], ref, dev)
            zs[list(ho.keys())[d]] = fid(cen[ref], fake, dev)
        (out / "zeroshot.json").write_text(json.dumps(zs, indent=2))
        print(f"  zero-shot FID (heldout -> {domain_names[ref]}): {zs}")

    print(f"done -> {out}")


if __name__ == "__main__":
    main()
