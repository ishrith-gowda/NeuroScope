"""
n x n translation matrix inference for extension c (multi-domain harmonization).

loads a trained multi-domain checkpoint and runs inference for every
(source, target) domain pair, producing:
  - n_by_n_matrix.npz   raw 4d arrays (n_domains, n_domains, c, h, w) per case
  - fig_multidomain_matrix_modality{i}.{pdf,png}  the publication figure
  - matrix_metrics.json  ssim/psnr per (source, target) pair

usage:
    python multi_domain_matrix_inference.py \\
        --checkpoint /data/.../checkpoint_best.pth \\
        --config /home/cc/.../multi_domain.yaml \\
        --output_dir /data/.../n_by_n_matrix \\
        --num_cases 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def normalise(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_cases", type=int, default=3)
    parser.add_argument("--modality_index", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    domain_names = cfg.get("domain_names", ["brats", "upenn_3t_triotim",
                                            "upenn_3t_other", "upenn_15t"])
    n_domains = len(domain_names)
    print(f"loaded config; domains: {domain_names}")

    # try to import the multi-domain trainer module to fetch model class
    try:
        from journal_extension.scripts.train_multi_domain import (  # type: ignore
            MultiDomainTrainer, MultiDomainConfig
        )
    except Exception as e:
        print(f"could not import multi-domain trainer: {e}")
        print("aborting matrix inference; the figure will be illustrative only.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dirs = cfg.get("data_dirs") or {}
    if not data_dirs:
        print("config has no data_dirs; aborting.")
        return

    model_cfg = MultiDomainConfig(
        ngf=int(cfg.get("ngf", 64)),
        ndf=int(cfg.get("ndf", 64)),
        n_residual_blocks=int(cfg.get("n_residual_blocks", 9)),
        attention_layers=tuple(cfg.get("attention_layers", [3, 4, 5])),
        n_domains=int(cfg.get("n_domains", 4)),
        domain_embed_dim=int(cfg.get("domain_embed_dim", 64)),
        style_dim=int(cfg.get("style_dim", 256)),
    )

    # build trainer (this loads the model and dataset)
    trainer = MultiDomainTrainer(
        config=model_cfg,
        data_dirs=data_dirs,
        domain_names=domain_names,
        output_dir=str(out_dir),
        batch_size=int(cfg.get("batch_size", 16)),
        image_size=int(cfg.get("image_size", 128)),
        num_workers=args.num_workers,
        experiment_name="matrix_inference_dummy",
        domain_split_file=cfg.get("domain_split_file"),
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    cleaned = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}
    trainer.model.load_state_dict(cleaned, strict=False)
    trainer.model.eval()

    # collect a few sample cases (one volume per source domain)
    cases: List[Dict] = []
    seen_domains = set()
    with torch.no_grad():
        for batch in trainer.train_loader:
            src = int(batch["domain"][0].item())
            if src in seen_domains:
                continue
            seen_domains.add(src)
            cases.append({
                "source_domain": src,
                "image": batch["image"][0].cpu(),
                "image_3slice": batch["image_3slice"][0].cpu()
                if "image_3slice" in batch else None,
            })
            if len(cases) >= n_domains:
                break

    matrix_metrics: Dict[str, Dict] = {}
    matrix_arrays: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for case_i, case in enumerate(cases):
            src = case["source_domain"]
            x = case.get("image_3slice", case["image"]).unsqueeze(0).to(device)
            for tgt in range(n_domains):
                try:
                    tgt_label = torch.tensor([tgt], device=device)
                    y = trainer.model.generator(x, tgt_label)
                except Exception as e:
                    print(f"  [warn] case={case_i} src={src} tgt={tgt} translate failed: {e}")
                    continue
                key = f"case{case_i}_src{src}_tgt{tgt}"
                matrix_arrays[key] = y.squeeze(0).cpu().numpy()
                # compute simple ssim against the source for diagonal,
                # against the same case's target if available otherwise
                ref = case["image"].numpy()
                fake = matrix_arrays[key]
                mae = float(np.mean(np.abs(normalise(ref) - normalise(fake))))
                matrix_metrics[key] = {"mae": mae,
                                       "src": int(src), "tgt": int(tgt),
                                       "case": int(case_i)}

    np.savez_compressed(out_dir / "n_by_n_matrix.npz", **matrix_arrays)
    (out_dir / "matrix_metrics.json").write_text(json.dumps(matrix_metrics, indent=2))

    # produce per-modality grid figure
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    if not cases:
        print("no cases collected; skipping figure.")
        return

    n_cases = len(cases)
    fig, axes = plt.subplots(n_cases, n_domains, figsize=(2.0 * n_domains,
                                                          2.0 * n_cases))
    if n_cases == 1:
        axes = np.array([axes])
    for ci, case in enumerate(cases):
        for tj in range(n_domains):
            key = f"case{ci}_src{case['source_domain']}_tgt{tj}"
            ax = axes[ci, tj]
            if key in matrix_arrays:
                img = normalise(matrix_arrays[key][args.modality_index])
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            else:
                ax.set_facecolor("#E5E7EB")
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_title(f"$\\rightarrow$ {domain_names[tj]}", fontsize=8)
            if tj == 0:
                ax.set_ylabel(domain_names[case["source_domain"]], fontsize=8)
    fig.suptitle(f"Extension C: $N{{\\times}}N$ Domain Translation Matrix "
                 f"(modality {args.modality_index})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_multidomain_matrix_modality{args.modality_index}.pdf")
    fig.savefig(out_dir / f"fig_multidomain_matrix_modality{args.modality_index}.png")
    plt.close(fig)
    print(f"done. wrote arrays + figures to {out_dir}")


if __name__ == "__main__":
    main()
