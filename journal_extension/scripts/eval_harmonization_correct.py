#!/usr/bin/env python3
"""
corrected, defensible harmonization evaluation for ext a (and reusable for ext c).

this REPLACES the flawed eval that headlined the trainer's global-moment ssim on
the cycle reconstruction (which overstated true ssim ~3x). here we report the
metrics that actually measure harmonization quality and that the miccai reviewers
demanded:

  - domain-classifier accuracy (raw vs harmonized): high -> chance means the two
    sites became indistinguishable (the harmonization-success signal).
  - fid + kid (torchmetrics), per modality, fake-B vs real-B (and fake-A vs real-A):
    distribution match in image space.
  - mmd in the domain-classifier feature space (raw vs harmonized).
  - brain-masked WINDOWED ssim (skimage 7x7): structure preservation (source vs
    harmonized) and the honest cycle-reconstruction ssim. we report windowed, not
    global-moment, and mask to the brain so background does not dominate.
  - subject-level aggregation (mean +- std over subjects), not just slice-level.

all numbers are written to a json keyed by checkpoint label. NEVER hand-edit;
figures/tables read from this json.

usage:
    python eval_harmonization_correct.py \
        --checkpoint ~/neuroscope/experiments/ext_a/ext_a_lambda0.5/.../checkpoint_best.pth \
        --label lambda0.5 \
        --brats_dir ~/neuroscope/preprocessed/brats \
        --upenn_dir ~/neuroscope/preprocessed/upenn \
        --domain_clf ~/neuroscope/checkpoints/domain_classifier.pth \
        --output_dir ~/neuroscope/experiments/ext_a_eval \
        [--max_batches N  # for a quick validation pass]
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
from skimage.metrics import structural_similarity as ssim_fn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from neuroscope.models.architectures.sa_cyclegan_25d import (  # noqa: E402
    SACycleGAN25DConfig, create_model,
)
from neuroscope.data.datasets.dataset_25d import create_dataloaders  # noqa: E402

MODALITIES = ["FLAIR", "T1", "T1ce", "T2"]  # 4 output channels


# --------------------------------------------------------------------------- #
# domain classifier (inlined from scripts/05_downstream_evaluation/domain_classifier.py
# so this script is self-contained on the node)
# --------------------------------------------------------------------------- #
class DomainClassifier(nn.Module):
    def __init__(self, in_channels=4, base_filters=32, n_domains=2, dropout=0.3):
        super().__init__()
        bf = base_filters
        def block(ci, co):
            return [
                nn.Conv2d(ci, co, 3, padding=1), nn.InstanceNorm2d(co), nn.LeakyReLU(0.2, True),
                nn.Conv2d(co, co, 3, padding=1), nn.InstanceNorm2d(co), nn.LeakyReLU(0.2, True),
            ]
        self.features = nn.Sequential(
            *block(in_channels, bf), nn.MaxPool2d(2),
            *block(bf, bf * 2), nn.MaxPool2d(2),
            *block(bf * 2, bf * 4), nn.MaxPool2d(2),
            *block(bf * 4, bf * 8), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(bf * 8, bf * 4),
            nn.LeakyReLU(0.2, True), nn.Dropout(dropout), nn.Linear(bf * 4, n_domains),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def extract_features(self, x):
        f = self.features(x)
        return f.view(f.size(0), -1)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def to_3ch_uint8(x_1ch: torch.Tensor) -> torch.Tensor:
    """normalize a (N,1,H,W) tensor per-image to [0,1] then replicate to 3ch float for fid (normalize=True)."""
    x = x_1ch.clone().float()
    n = x.size(0)
    flat = x.view(n, -1)
    mn = flat.min(dim=1)[0].view(n, 1, 1, 1)
    mx = flat.max(dim=1)[0].view(n, 1, 1, 1)
    x = (x - mn) / (mx - mn + 1e-8)
    return x.repeat(1, 3, 1, 1).clamp(0, 1)


def masked_windowed_ssim(src: np.ndarray, tgt: np.ndarray) -> float:
    """per-channel skimage windowed ssim within the brain bounding box, averaged over channels.

    src, tgt: (C,H,W) arrays. brain = nonzero region of src; ssim computed on the
    bbox crop so background does not inflate. each channel min-max normalized.
    """
    vals = []
    for c in range(src.shape[0]):
        a, b = src[c], tgt[c]
        mask = np.abs(a) > 1e-6
        if mask.sum() < 64:
            continue
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        ac, bc = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
        ac = (ac - ac.min()) / (ac.max() - ac.min() + 1e-8)
        bc = (bc - bc.min()) / (bc.max() - bc.min() + 1e-8)
        if min(ac.shape) < 7:
            continue
        vals.append(ssim_fn(ac, bc, data_range=1.0))
    return float(np.mean(vals)) if vals else float("nan")


def mmd_rbf(x: np.ndarray, y: np.ndarray, sigmas=(0.5, 1.0, 2.0, 4.0, 8.0)) -> float:
    if len(x) > 1024:
        x = x[np.random.choice(len(x), 1024, replace=False)]
    if len(y) > 1024:
        y = y[np.random.choice(len(y), 1024, replace=False)]

    def k(a, b, s):
        d = np.sum(a * a, 1, keepdims=True) - 2 * a @ b.T + np.sum(b * b, 1, keepdims=True).T
        return np.exp(-d / (2 * s * s))

    m = 0.0
    for s in sigmas:
        m += k(x, x, s).mean() + k(y, y, s).mean() - 2 * k(x, y, s).mean()
    return float(m / len(sigmas))


def replicate_to_3slice(x4: torch.Tensor) -> torch.Tensor:
    """4ch single-slice -> 12ch pseudo-2.5d input (replicate center 3x), matching the trainer."""
    return x4.unsqueeze(2).repeat(1, 1, 3, 1, 1).view(x4.size(0), -1, x4.size(2), x4.size(3))


# --------------------------------------------------------------------------- #
# checkpoint loading
# --------------------------------------------------------------------------- #
def load_generator_weights(model, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck)
    cleaned = {}
    for k, v in sd.items():
        nk = k
        for p in ("_orig_mod.", "module."):
            nk = nk.replace(p, "")
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"  loaded model: missing={len(missing)} unexpected={len(unexpected)} epoch={ck.get('epoch','?')}")


# --------------------------------------------------------------------------- #
# main eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    # model
    config = SACycleGAN25DConfig(ngf=64, ndf=64, n_residual_blocks=9,
                                 attention_layers=(3, 4, 5), nce_feature_layers=(2, 5))
    model = create_model(config).to(device).eval()
    print(f"[{args.label}] loading generator weights from {args.checkpoint}")
    load_generator_weights(model, args.checkpoint, device)

    # domain classifier
    clf = DomainClassifier(in_channels=4, n_domains=2).to(device).eval()
    clf_ck = torch.load(args.domain_clf, map_location=device, weights_only=False)
    clf.load_state_dict(clf_ck.get("model_state_dict", clf_ck), strict=True)

    _, _, test_loader = create_dataloaders(
        brats_dir=args.brats_dir, upenn_dir=args.upenn_dir,
        batch_size=args.batch_size, image_size=(128, 128), num_workers=args.num_workers,
    )

    # fid/kid per modality, per direction
    fid = {f"{d}_{m}": FrechetInceptionDistance(feature=2048, normalize=True).to(device)
           for d in ("A2B", "B2A") for m in MODALITIES}
    kid = {f"{d}_{m}": KernelInceptionDistance(subset_size=50, normalize=True).to(device)
           for d in ("A2B", "B2A") for m in MODALITIES}

    # domain-classifier acc accumulators
    clf_correct_raw = clf_total_raw = 0
    clf_correct_harm = clf_total_harm = 0
    feat_realB, feat_fakeB = [], []

    # subject-level masked-ssim accumulators
    subj_struct = defaultdict(list)   # source vs harmonized (A->B)
    subj_cycle = defaultdict(list)    # cycle recon (A->B->A)

    for bi, batch in enumerate(test_loader):
        if args.max_batches and bi >= args.max_batches:
            break
        real_A = batch["A"].to(device)
        real_B = batch["B"].to(device)
        cen_A = batch["A_center"].to(device)
        cen_B = batch["B_center"].to(device)

        fake_B = model.G_A2B(real_A)
        fake_A = model.G_B2A(real_B)
        rec_A = model.G_B2A(replicate_to_3slice(fake_B))  # A->B->A cycle (honest windowed ssim)

        # --- domain classifier: raw (cen_A=0, cen_B=1) vs harmonized (fake_B from A keeps label 0) ---
        raw_logits = clf(torch.cat([cen_A, cen_B], 0))
        raw_labels = torch.cat([torch.zeros(cen_A.size(0)), torch.ones(cen_B.size(0))]).to(device).long()
        clf_correct_raw += (raw_logits.argmax(1) == raw_labels).sum().item()
        clf_total_raw += raw_labels.size(0)
        # harmonized: fake_B (was domain A, label 0) + fake_A (was domain B, label 1)
        harm_logits = clf(torch.cat([fake_B, fake_A], 0))
        harm_labels = torch.cat([torch.zeros(fake_B.size(0)), torch.ones(fake_A.size(0))]).to(device).long()
        clf_correct_harm += (harm_logits.argmax(1) == harm_labels).sum().item()
        clf_total_harm += harm_labels.size(0)
        feat_realB.append(clf.extract_features(cen_B).cpu().numpy())
        feat_fakeB.append(clf.extract_features(fake_B).cpu().numpy())

        # --- fid/kid per modality ---
        for mi, m in enumerate(MODALITIES):
            fid[f"A2B_{m}"].update(to_3ch_uint8(cen_B[:, mi:mi+1]), real=True)
            fid[f"A2B_{m}"].update(to_3ch_uint8(fake_B[:, mi:mi+1]), real=False)
            kid[f"A2B_{m}"].update(to_3ch_uint8(cen_B[:, mi:mi+1]), real=True)
            kid[f"A2B_{m}"].update(to_3ch_uint8(fake_B[:, mi:mi+1]), real=False)
            fid[f"B2A_{m}"].update(to_3ch_uint8(cen_A[:, mi:mi+1]), real=True)
            fid[f"B2A_{m}"].update(to_3ch_uint8(fake_A[:, mi:mi+1]), real=False)
            kid[f"B2A_{m}"].update(to_3ch_uint8(cen_A[:, mi:mi+1]), real=True)
            kid[f"B2A_{m}"].update(to_3ch_uint8(fake_A[:, mi:mi+1]), real=False)

        # --- masked windowed ssim, subject-level (A->B direction) ---
        cen_A_np = cen_A.cpu().numpy()
        fake_B_np = fake_B.cpu().numpy()
        rec_A_np = rec_A.cpu().numpy()
        for i, subj in enumerate(batch["A_subject"]):
            subj_struct[subj].append(masked_windowed_ssim(cen_A_np[i], fake_B_np[i]))
            subj_cycle[subj].append(masked_windowed_ssim(cen_A_np[i], rec_A_np[i]))

        if bi % 50 == 0:
            print(f"  batch {bi}/{len(test_loader)}")

    # aggregate
    def subj_agg(d):
        per_subj = [np.nanmean(v) for v in d.values() if len(v)]
        return {"mean": float(np.nanmean(per_subj)), "std": float(np.nanstd(per_subj)),
                "n_subjects": len(per_subj)}

    fr = np.concatenate(feat_realB) if feat_realB else np.zeros((1, 1))
    ff = np.concatenate(feat_fakeB) if feat_fakeB else np.zeros((1, 1))

    results = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "domain_classifier": {
            "acc_raw": clf_correct_raw / max(clf_total_raw, 1),
            "acc_harmonized": clf_correct_harm / max(clf_total_harm, 1),
            "chance": 0.5,
            "note": "lower harmonized acc toward chance = better site confusion",
        },
        "mmd_feature_space_fakeB_vs_realB": mmd_rbf(ff, fr),
        "fid": {k: float(v.compute().item()) for k, v in fid.items()},
        "kid_mean": {k: float(v.compute()[0].item()) for k, v in kid.items()},
        "masked_windowed_ssim_structure_A2B": subj_agg(subj_struct),
        "masked_windowed_ssim_cycle_A2B": subj_agg(subj_cycle),
    }
    # convenient averages
    results["fid_A2B_avg"] = float(np.mean([results["fid"][f"A2B_{m}"] for m in MODALITIES]))
    results["fid_B2A_avg"] = float(np.mean([results["fid"][f"B2A_{m}"] for m in MODALITIES]))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_{args.label}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{args.label}] wrote {out_path}")
    print(f"  domain-clf acc raw={results['domain_classifier']['acc_raw']:.3f} "
          f"harmonized={results['domain_classifier']['acc_harmonized']:.3f}")
    print(f"  fid A2B avg={results['fid_A2B_avg']:.2f} | "
          f"struct-ssim={results['masked_windowed_ssim_structure_A2B']['mean']:.3f} | "
          f"cycle-ssim={results['masked_windowed_ssim_cycle_A2B']['mean']:.3f}")
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--brats_dir", required=True)
    p.add_argument("--upenn_dir", required=True)
    p.add_argument("--domain_clf", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--max_batches", type=int, default=0, help="0 = full test set; N for a quick validation pass")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
