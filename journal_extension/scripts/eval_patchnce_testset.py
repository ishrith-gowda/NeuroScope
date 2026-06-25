#!/usr/bin/env python3
"""
test-set evaluation for extension a (patchnce hybrid) checkpoints.

evaluates the four trained models (primary lambda=1.0 and ablation
lambda={0.1, 0.5, 2.0}) on the held-out test split. computes per-slice
ssim, psnr, mae, lpips on the cycle reconstruction (rec_a vs real_a,
rec_b vs real_b) plus harmonization-direction perceptual metrics
(fake_b distribution vs real_b distribution via mmd).

usage:
    python eval_patchnce_testset.py \\
        --checkpoint_root /data/experiments/journal_extension \\
        --brats_dir /data/preprocessed/brats \\
        --upenn_dir /data/preprocessed/upenn \\
        --output_dir /data/experiments/journal_extension/patchnce_testset_eval

writes one test_results.json per checkpoint with per-slice arrays and
summary statistics. saves a combined patchnce_testset_summary.json with
all four runs aggregated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from tqdm import tqdm

# ensure repo root is on path so we can import journal_extension/neuroscope packages
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from journal_extension.scripts.train_hybrid_nce import HybridNCETrainer  # noqa: E402
from neuroscope.models.architectures.sa_cyclegan_25d import SACycleGAN25DConfig  # noqa: E402


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def compute_ssim_skimage(x: torch.Tensor, y: torch.Tensor) -> float:
    """canonical ssim with default 7x7 window, mean across channels."""
    from skimage.metrics import structural_similarity as ssim_fn

    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    vals = []
    for c in range(x_np.shape[0]):
        a = x_np[c]
        b = y_np[c]
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        b = (b - b.min()) / (b.max() - b.min() + 1e-8)
        vals.append(ssim_fn(a, b, data_range=1.0))
    return float(np.mean(vals))


def compute_ssim_global(x: torch.Tensor, y: torch.Tensor) -> float:
    """trainer's global-moment ssim (single mu/sigma over all elements).

    matches the simplified ssim used inside HybridNCETrainer.compute_ssim so
    the test numbers can be compared directly against validation logs.
    """
    x_cpu = x.detach().cpu()
    y_cpu = y.detach().cpu()
    mu_x = x_cpu.mean()
    mu_y = y_cpu.mean()
    sigma_x = ((x_cpu - mu_x) ** 2).mean()
    sigma_y = ((y_cpu - mu_y) ** 2).mean()
    sigma_xy = ((x_cpu - mu_x) * (y_cpu - mu_y)).mean()
    C1, C2 = 0.01**2, 0.03**2
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return float((num / den).item())


def compute_psnr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """compute psnr in db for a single sample (c, h, w)."""
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    vals = []
    for c in range(x_np.shape[0]):
        a = x_np[c]
        b = y_np[c]
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        b = (b - b.min()) / (b.max() - b.min() + 1e-8)
        mse = float(np.mean((a - b) ** 2))
        if mse <= 0:
            vals.append(99.0)
        else:
            vals.append(10.0 * np.log10(1.0 / mse))
    return float(np.mean(vals))


def compute_mae_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """mean absolute error in normalized intensity space for one sample."""
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    vals = []
    for c in range(x_np.shape[0]):
        a = x_np[c]
        b = y_np[c]
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        b = (b - b.min()) / (b.max() - b.min() + 1e-8)
        vals.append(float(np.mean(np.abs(a - b))))
    return float(np.mean(vals))


def maximum_mean_discrepancy(
    x: np.ndarray, y: np.ndarray, sigmas=(0.5, 1.0, 2.0, 4.0, 8.0)
) -> float:
    """compute squared mmd between two flat feature arrays with rbf kernel sum."""
    if len(x) > 1024:
        idx = np.random.choice(len(x), 1024, replace=False)
        x = x[idx]
    if len(y) > 1024:
        idx = np.random.choice(len(y), 1024, replace=False)
        y = y[idx]

    def rbf(a, b, s):
        d = (
            np.sum(a * a, axis=1, keepdims=True)
            - 2 * a @ b.T
            + np.sum(b * b, axis=1, keepdims=True).T
        )
        return np.exp(-d / (2 * s * s))

    mmd2 = 0.0
    for s in sigmas:
        kxx = rbf(x, x, s).mean()
        kyy = rbf(y, y, s).mean()
        kxy = rbf(x, y, s).mean()
        mmd2 += kxx + kyy - 2 * kxy
    return float(mmd2 / len(sigmas))


# ---------------------------------------------------------------------------
# evaluation core
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_checkpoint(
    trainer: HybridNCETrainer,
    checkpoint_path: Path,
    label: str,
) -> Dict:
    """run full test-set inference and return aggregated metrics."""
    print(f"\n[{label}] loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    # strip torch.compile wrapper prefix (_orig_mod.) and dataparallel prefix (module.)
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in ("_orig_mod.", "module."):
            new_k = new_k.replace(prefix, "")
        cleaned[new_k] = v
    missing, unexpected = trainer.model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[{label}] missing keys: {len(missing)} (showing first 3): {missing[:3]}")
    if unexpected:
        print(f"[{label}] unexpected keys: {len(unexpected)} (showing first 3): {unexpected[:3]}")
    trainer.model.eval()

    saved_epoch = ckpt.get("epoch", "?")
    saved_lambda = ckpt.get("lambda_nce", "?")
    print(f"[{label}] checkpoint epoch={saved_epoch} lambda_nce={saved_lambda}")

    per_slice: Dict[str, List[float]] = {
        "ssim_A2B": [],
        "ssim_B2A": [],
        "ssim_global_A2B": [],
        "ssim_global_B2A": [],
        "psnr_A2B": [],
        "psnr_B2A": [],
        "mae_A2B": [],
        "mae_B2A": [],
    }
    fake_b_features = []
    real_b_features = []
    fake_a_features = []
    real_a_features = []

    n_slices = 0
    t_start = time.time()
    for batch in tqdm(trainer.test_loader, desc=f"test {label}", ncols=100):
        real_a = batch["A"].to(trainer.device)
        real_b = batch["B"].to(trainer.device)
        center_a = batch["A_center"].to(trainer.device)
        center_b = batch["B_center"].to(trainer.device)

        fake_b = trainer.model.G_A2B(real_a)
        fake_a = trainer.model.G_B2A(real_b)

        # build a 3-slice tensor from the single fake by replicating the channel-axis,
        # matching the trainer convention so cycle reconstruction is consistent.
        fake_b_3 = (
            fake_b.unsqueeze(2)
            .repeat(1, 1, 3, 1, 1)
            .view(fake_b.size(0), -1, fake_b.size(2), fake_b.size(3))
        )
        fake_a_3 = (
            fake_a.unsqueeze(2)
            .repeat(1, 1, 3, 1, 1)
            .view(fake_a.size(0), -1, fake_a.size(2), fake_a.size(3))
        )
        rec_a = trainer.model.G_B2A(fake_b_3)
        rec_b = trainer.model.G_A2B(fake_a_3)

        bs = center_a.size(0)
        for i in range(bs):
            per_slice["ssim_A2B"].append(compute_ssim_skimage(center_a[i], rec_a[i]))
            per_slice["ssim_B2A"].append(compute_ssim_skimage(center_b[i], rec_b[i]))
            per_slice["ssim_global_A2B"].append(compute_ssim_global(center_a[i], rec_a[i]))
            per_slice["ssim_global_B2A"].append(compute_ssim_global(center_b[i], rec_b[i]))
            per_slice["psnr_A2B"].append(compute_psnr_torch(center_a[i], rec_a[i]))
            per_slice["psnr_B2A"].append(compute_psnr_torch(center_b[i], rec_b[i]))
            per_slice["mae_A2B"].append(compute_mae_torch(center_a[i], rec_a[i]))
            per_slice["mae_B2A"].append(compute_mae_torch(center_b[i], rec_b[i]))
        n_slices += bs

        # collect global-average-pooled features for mmd estimation
        fb_pool = fake_b.mean(dim=(2, 3)).cpu().numpy()
        rb_pool = center_b.mean(dim=(2, 3)).cpu().numpy()
        fa_pool = fake_a.mean(dim=(2, 3)).cpu().numpy()
        ra_pool = center_a.mean(dim=(2, 3)).cpu().numpy()
        fake_b_features.append(fb_pool)
        real_b_features.append(rb_pool)
        fake_a_features.append(fa_pool)
        real_a_features.append(ra_pool)

    elapsed = time.time() - t_start
    throughput = n_slices / max(elapsed, 1e-6)

    # mmd between fake_b and real_b distributions, fake_a vs real_a
    fake_b_features = np.concatenate(fake_b_features, axis=0)
    real_b_features = np.concatenate(real_b_features, axis=0)
    fake_a_features = np.concatenate(fake_a_features, axis=0)
    real_a_features = np.concatenate(real_a_features, axis=0)
    mmd_a2b = maximum_mean_discrepancy(fake_b_features, real_b_features)
    mmd_b2a = maximum_mean_discrepancy(fake_a_features, real_a_features)

    summary: Dict = {
        "label": label,
        "checkpoint": str(checkpoint_path),
        "n_slices": n_slices,
        "elapsed_seconds": elapsed,
        "throughput_slices_per_sec": throughput,
        "saved_epoch": saved_epoch,
        "saved_lambda_nce": saved_lambda,
        "mmd_A2B": mmd_a2b,
        "mmd_B2A": mmd_b2a,
    }

    for key, vals in per_slice.items():
        a = np.asarray(vals, dtype=np.float64)
        summary[f"{key}_mean"] = float(a.mean())
        summary[f"{key}_std"] = float(a.std())
        summary[f"{key}_median"] = float(np.median(a))
        summary[f"{key}_p25"] = float(np.percentile(a, 25))
        summary[f"{key}_p75"] = float(np.percentile(a, 75))

    summary["per_slice"] = per_slice
    return summary


def build_trainer(
    config_path: Path, brats_dir: str, upenn_dir: str, num_workers: int
) -> HybridNCETrainer:
    """instantiate trainer once and re-use across checkpoints."""
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f) or {}

    model_cfg = SACycleGAN25DConfig(
        ngf=int(cfg_dict.get("ngf", 64)),
        ndf=int(cfg_dict.get("ndf", 64)),
        n_residual_blocks=int(cfg_dict.get("n_residual_blocks", 9)),
        attention_layers=tuple(cfg_dict.get("attention_layers", [3, 4, 5])),
        nce_feature_layers=tuple(cfg_dict.get("nce_feature_layers", [2, 5])),
    )

    trainer = HybridNCETrainer(
        config=model_cfg,
        brats_dir=brats_dir,
        upenn_dir=upenn_dir,
        output_dir="/tmp/eval_dummy_output",
        batch_size=int(cfg_dict.get("batch_size", 16)),
        image_size=int(cfg_dict.get("image_size", 128)),
        num_workers=num_workers,
        experiment_name="patchnce_testset_eval",
        lambda_nce=1.0,
        nce_num_patches=int(cfg_dict.get("nce_num_patches", 256)),
        nce_temperature=float(cfg_dict.get("nce_temperature", 0.07)),
        use_compile=False,
        epochs=1,
    )
    print(f"test batches: {len(trainer.test_loader)}")
    return trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_root",
        required=True,
        help="root dir containing patchnce_hybrid + patchnce_ablation_lambda* subdirs",
    )
    parser.add_argument("--brats_dir", required=True)
    parser.add_argument("--upenn_dir", required=True)
    parser.add_argument(
        "--config", default=None, help="path to patchnce_hybrid.yaml (defaults to repo path)"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(args.checkpoint_root)

    # canonical experiment-dir layout produced by train_hybrid_nce.py
    runs = [
        (
            "lambda1.0",
            ckpt_root
            / "patchnce_hybrid"
            / "sa_cyclegan_25d_patchnce_hybrid"
            / "checkpoints"
            / "checkpoint_best.pth",
        ),
        (
            "lambda0.1",
            ckpt_root
            / "patchnce_ablation_lambda0.1"
            / "sa_cyclegan_25d_patchnce_hybrid"
            / "checkpoints"
            / "checkpoint_best.pth",
        ),
        (
            "lambda0.5",
            ckpt_root
            / "patchnce_ablation_lambda0.5"
            / "sa_cyclegan_25d_patchnce_hybrid"
            / "checkpoints"
            / "checkpoint_best.pth",
        ),
        (
            "lambda2.0",
            ckpt_root
            / "patchnce_ablation_lambda2.0"
            / "sa_cyclegan_25d_patchnce_hybrid"
            / "checkpoints"
            / "checkpoint_best.pth",
        ),
    ]

    config_path = (
        Path(args.config)
        if args.config
        else (Path(__file__).parent.parent / "configs" / "patchnce_hybrid.yaml")
    )

    trainer = build_trainer(config_path, args.brats_dir, args.upenn_dir, args.num_workers)

    all_results = {}
    for label, ckpt_path in runs:
        if not ckpt_path.exists():
            print(f"[{label}] missing checkpoint at {ckpt_path}, skipping.")
            continue
        summary = evaluate_checkpoint(trainer, ckpt_path, label)
        all_results[label] = summary
        per_run_path = out_dir / f"test_results_{label}.json"
        with open(per_run_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{label}] saved {per_run_path}")

    summary_path = out_dir / "patchnce_testset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nfinal summary written to {summary_path}")


if __name__ == "__main__":
    main()
