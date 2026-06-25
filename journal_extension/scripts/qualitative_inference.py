"""
qualitative inference for extension a (patchnce hybrid).

loads the four best checkpoints (lambda in {0.1, 0.5, 1.0, 2.0}) and renders
side-by-side translations on a small set of representative test cases. saves
both the raw .npz tensors (for later figure rebuilds) and a publication-quality
multi-panel figure showing real-A / fake-B / cycle-A / |diff| / fake-A /
real-B for each lambda.

usage:
    python qualitative_inference.py \\
        --checkpoint_root /data/experiments/journal_extension \\
        --brats_dir /data/preprocessed/brats \\
        --upenn_dir /data/preprocessed/upenn \\
        --output_dir /data/experiments/journal_extension/patchnce_qualitative \\
        --num_cases 4 --modality_index 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from journal_extension.scripts.train_hybrid_nce import HybridNCETrainer  # noqa: E402
from neuroscope.models.architectures.sa_cyclegan_25d import SACycleGAN25DConfig  # noqa: E402

MODALITY_NAMES: list[str] = ["FLAIR", "T1", "T1ce", "T2"]
LAMBDA_VALUES: list[float] = [0.1, 0.5, 1.0, 2.0]


def build_trainer(
    config_path: Path, brats_dir: str, upenn_dir: str, num_workers: int
) -> HybridNCETrainer:
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
        output_dir="/tmp/qualitative_dummy",
        batch_size=int(cfg_dict.get("batch_size", 16)),
        image_size=int(cfg_dict.get("image_size", 128)),
        num_workers=num_workers,
        experiment_name="patchnce_qualitative",
        lambda_nce=1.0,
        nce_num_patches=int(cfg_dict.get("nce_num_patches", 256)),
        nce_temperature=float(cfg_dict.get("nce_temperature", 0.07)),
        use_compile=False,
        epochs=1,
    )
    return trainer


def load_checkpoint_into(trainer: HybridNCETrainer, ckpt_path: Path) -> int:
    ckpt = torch.load(ckpt_path, map_location=trainer.device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in ("_orig_mod.", "module."):
            new_k = new_k.replace(prefix, "")
        cleaned[new_k] = v
    trainer.model.load_state_dict(cleaned, strict=False)
    trainer.model.eval()
    return int(ckpt.get("epoch", -1))


@torch.no_grad()
def collect_cases(trainer: HybridNCETrainer, num_cases: int) -> list[dict]:
    """grab a few diverse test cases by picking spread-out batch indices."""
    cases: list[dict] = []
    target = max(1, num_cases)
    indices_seen = set()
    for batch_idx, batch in enumerate(trainer.test_loader):
        if len(cases) >= target:
            break
        # pick the centre slice from each batch
        idx_in_batch = 0
        if batch_idx in indices_seen:
            continue
        indices_seen.add(batch_idx)
        cases.append(
            {
                "batch_idx": batch_idx,
                "real_A": batch["A"][idx_in_batch].cpu(),
                "real_B": batch["B"][idx_in_batch].cpu(),
                "center_A": batch["A_center"][idx_in_batch].cpu(),
                "center_B": batch["B_center"][idx_in_batch].cpu(),
            }
        )
        # skip ahead a bit so cases aren't all from the front
        for _ in range(max(1, len(trainer.test_loader) // (target + 1))):
            try:
                next(iter(trainer.test_loader))
            except StopIteration:
                break
    return cases


@torch.no_grad()
def run_translations(trainer: HybridNCETrainer, cases: list[dict]) -> list[dict]:
    """add fake_B / fake_A / rec_A / rec_B for each case using current weights."""
    out = []
    for case in cases:
        real_a = case["real_A"].unsqueeze(0).to(trainer.device)
        real_b = case["real_B"].unsqueeze(0).to(trainer.device)
        fake_b = trainer.model.G_A2B(real_a)
        fake_a = trainer.model.G_B2A(real_b)

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

        out.append(
            {
                "batch_idx": case["batch_idx"],
                "real_A": case["real_A"].numpy(),
                "real_B": case["real_B"].numpy(),
                "center_A": case["center_A"].numpy(),
                "center_B": case["center_B"].numpy(),
                "fake_B": fake_b.squeeze(0).cpu().numpy(),
                "fake_A": fake_a.squeeze(0).cpu().numpy(),
                "rec_A": rec_a.squeeze(0).cpu().numpy(),
                "rec_B": rec_b.squeeze(0).cpu().numpy(),
            }
        )
    return out


def normalise(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def render_grid(per_lambda: dict[float, list[dict]], modality_index: int, out_path: Path) -> None:
    """render figure: rows = case, columns = sequence per lambda."""
    if not per_lambda:
        return
    lambdas = sorted(per_lambda.keys())
    n_cases = min(len(per_lambda[lambdas[0]]), 4)
    if n_cases == 0:
        return

    cols_per_lambda = 4
    fig_w = 0.85 * cols_per_lambda * len(lambdas) + 1.5
    fig_h = 2.0 * n_cases + 0.6
    fig, axes = plt.subplots(n_cases, cols_per_lambda * len(lambdas), figsize=(fig_w, fig_h))
    if n_cases == 1:
        axes = np.array([axes])

    for c in range(n_cases):
        for li, lam in enumerate(lambdas):
            payload = per_lambda[lam][c]
            real_a = normalise(payload["center_A"][modality_index])
            fake_b = normalise(payload["fake_B"][modality_index])
            real_b = normalise(payload["center_B"][modality_index])
            diff = np.abs(real_a - fake_b)

            base = li * cols_per_lambda
            for j, (img, ttl, cmap) in enumerate(
                [
                    (real_a, "Real A", "gray"),
                    (fake_b, "Fake B", "gray"),
                    (real_b, "Real B", "gray"),
                    (diff, "$|\\,$Real A $-$ Fake B$\\,|$", "magma"),
                ]
            ):
                ax = axes[c, base + j]
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    if j == 0:
                        ax.set_title(f"$\\lambda$={lam}\n{ttl}", fontsize=8)
                    else:
                        ax.set_title(ttl, fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"Case {c + 1}", fontsize=8)

    fig.suptitle(
        f"Qualitative Translation -- Modality {MODALITY_NAMES[modality_index]}", fontsize=12, y=1.02
    )
    fig.tight_layout()
    fig.savefig(str(out_path) + ".pdf")
    fig.savefig(str(out_path) + ".png")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--brats_dir", required=True)
    parser.add_argument("--upenn_dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_cases", type=int, default=4)
    parser.add_argument(
        "--modality_index",
        type=int,
        default=1,
        help="which channel to display in the figure (0-3, default T1=1)",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(args.checkpoint_root)

    runs: list[tuple[float, Path]] = [
        (
            1.0,
            ckpt_root
            / "patchnce_hybrid"
            / "sa_cyclegan_25d_patchnce_hybrid"
            / "checkpoints"
            / "checkpoint_best.pth",
        ),
        (
            0.1,
            ckpt_root
            / "patchnce_ablation_lambda0.1"
            / "sa_cyclegan_25d_patchnce_hybrid"
            / "checkpoints"
            / "checkpoint_best.pth",
        ),
        (
            0.5,
            ckpt_root
            / "patchnce_ablation_lambda0.5"
            / "sa_cyclegan_25d_patchnce_hybrid"
            / "checkpoints"
            / "checkpoint_best.pth",
        ),
        (
            2.0,
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

    # collect a fixed set of cases once (deterministic order)
    print(f"sampling {args.num_cases} test cases for qualitative comparison...")
    cases = []
    for batch_idx, batch in enumerate(trainer.test_loader):
        if len(cases) >= args.num_cases:
            break
        # pick first sample of every Kth batch to spread cases
        spread = max(1, len(trainer.test_loader) // max(args.num_cases, 1))
        if batch_idx % spread != 0:
            continue
        cases.append(
            {
                "batch_idx": batch_idx,
                "real_A": batch["A"][0].cpu(),
                "real_B": batch["B"][0].cpu(),
                "center_A": batch["A_center"][0].cpu(),
                "center_B": batch["B_center"][0].cpu(),
            }
        )
    print(f"collected {len(cases)} cases")

    per_lambda: dict[float, list[dict]] = {}
    for lam, ckpt_path in runs:
        if not ckpt_path.exists():
            print(f"[lambda={lam}] missing checkpoint, skipping.")
            continue
        epoch_loaded = load_checkpoint_into(trainer, ckpt_path)
        print(
            f"[lambda={lam}] loaded epoch={epoch_loaded}, running inference on {len(cases)} cases"
        )
        per_lambda[lam] = run_translations(trainer, cases)

    # save raw arrays for later figure remixes
    payload = {
        "lambdas": list(per_lambda.keys()),
        "case_count": len(cases),
        "modality_names": MODALITY_NAMES,
        "modality_index": args.modality_index,
    }
    arrays_dict: dict[str, np.ndarray] = {}
    for lam, val_list in per_lambda.items():
        for i, val in enumerate(val_list):
            for key in (
                "real_A",
                "real_B",
                "center_A",
                "center_B",
                "fake_B",
                "fake_A",
                "rec_A",
                "rec_B",
            ):
                arrays_dict[f"lambda{lam}_case{i}_{key}"] = np.asarray(val[key])
    np.savez_compressed(out_dir / "qualitative_arrays.npz", **arrays_dict)
    (out_dir / "qualitative_meta.json").write_text(json.dumps(payload, indent=2))

    # render the figure for the requested modality
    render_grid(per_lambda, args.modality_index, out_dir / "fig_patchnce_qualitative")

    # render figures for all modalities for completeness
    for m in range(4):
        render_grid(
            per_lambda, m, out_dir / f"fig_patchnce_qualitative_modality{m}_{MODALITY_NAMES[m]}"
        )

    print(f"done. wrote arrays + figures to {out_dir}")


if __name__ == "__main__":
    main()
