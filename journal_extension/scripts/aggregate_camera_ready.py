"""
aggregate camera-ready upgrade outputs into figures and tables. consumes:

  - patchnce λ=0.5 multi-seed runs (extension a triple-seed)
  - compression λ_rate sweep runs (extension b)
  - multi-domain n=4 training and n x n inference (extension c)
  - task-aware harmonization run (extension d)
  - federated fedprox / scaffold runs (extension e)

produces:

  - fig_ext_a_seeds.{pdf,png}                  per-seed val ssim curves and bar
  - fig_ext_b_rate_distortion.{pdf,png}        rate-distortion curve
  - fig_ext_c_matrix.{pdf,png}                 n x n translation matrix figure
  - fig_ext_d_taskaware_dice.{pdf,png}         task-aware vs vanilla downstream
  - fig_ext_e_federated_compare.{pdf,png}      fedavg vs fedprox vs scaffold
  - table_camera_ready_*.tex                   one latex table per extension
  - camera_ready_summary.json                  one machine-readable digest
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PUB_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
}


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def load_history(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def discover_seed_runs(exp_root: Path) -> List[Path]:
    return sorted(exp_root.glob("patchnce_lambda0.5_seed*_100epochs"))


def discover_compression_runs(exp_root: Path) -> List[Path]:
    return sorted(exp_root.glob("compression_lambda_rate_*"))


def discover_federated_runs(exp_root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for strat in ("fedavg", "fedprox", "scaffold"):
        cand = list(exp_root.glob(f"federated_{strat}*"))
        if cand:
            out[strat] = cand[0]
    return out


# ---------------------------------------------------------------------------
# extension a: seeds
# ---------------------------------------------------------------------------


def fig_ext_a_seeds(repo_root: Path, out_dir: Path) -> None:
    """combine the existing seed=42 lambda=0.5 run with any new seeds."""
    histories: Dict[str, Dict] = {}

    primary = repo_root / "journal_extension" / "results" / "patchnce" / "lambda0.5_history.json"
    h = load_history(primary)
    if h:
        histories["seed=42 (200 epochs)"] = h

    exp_root_remote = repo_root / "journal_extension" / "results" / "camera_ready"
    if exp_root_remote.exists():
        for path in exp_root_remote.glob("patchnce_lambda0.5_seed*_history.json"):
            label = path.stem.replace("patchnce_lambda0.5_", "").replace("_history", "")
            histories[label] = load_history(path)

    if len(histories) < 2:
        print("ext a seeds: only one seed available; skipping figure.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for label, hist in histories.items():
        ssim_a = np.asarray(hist["val"]["ssim_A2B"])
        ssim_b = np.asarray(hist["val"]["ssim_B2A"])
        mean = (ssim_a + ssim_b) / 2.0
        axes[0].plot(np.arange(len(mean)), mean, label=label)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Mean Validation SSIM")
    axes[0].set_title("(a) Per-Seed Validation SSIM, $\\lambda_{\\mathrm{NCE}}=0.5$")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_ylim(0.95, 1.0)

    means = []
    labels = []
    for label, hist in histories.items():
        ssim_a = np.asarray(hist["val"]["ssim_A2B"])
        ssim_b = np.asarray(hist["val"]["ssim_B2A"])
        mean = float(((ssim_a + ssim_b) / 2.0).max())
        means.append(mean); labels.append(label)
    axes[1].bar(range(len(means)), means, color="#2563EB")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylim(0.95, 1.0)
    axes[1].set_title("(b) Best Mean Val SSIM per Seed")
    axes[1].set_ylabel("Best Mean Val SSIM")
    for i, v in enumerate(means):
        axes[1].text(i, v + 0.0005, f"{v:.4f}", ha="center", fontsize=8)

    fig.suptitle("Extension A: Multi-Seed Robustness for $\\lambda_{\\mathrm{NCE}}=0.5$",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ext_a_seeds.pdf")
    fig.savefig(out_dir / "fig_ext_a_seeds.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# extension b: rate-distortion
# ---------------------------------------------------------------------------


def fig_ext_b_rate_distortion(repo_root: Path, out_dir: Path) -> None:
    """rate-distortion curve from the lambda_rate sweep + existing point."""
    points: List[Dict] = []

    legacy = json.load(open(repo_root / "journal_extension" / "results" / "all_results.json"))
    legacy_c = legacy.get("compression", {})
    if legacy_c.get("history", {}).get("train", {}).get("bpe"):
        bpe_final = float(legacy_c["history"]["train"]["bpe"][-1])
        ssim = float(legacy_c.get("best_val_ssim", 0.0))
        psnr = (float(legacy_c["history"]["val"]["psnr_A2B"][-1])
                + float(legacy_c["history"]["val"]["psnr_B2A"][-1])) / 2.0 \
            if legacy_c["history"]["val"].get("psnr_A2B") else None
        points.append({
            "lambda_rate": 0.01, "bpe": bpe_final, "ssim": ssim, "psnr": psnr,
            "source": "previous (200 epochs)",
        })

    sweep_dir = repo_root / "journal_extension" / "results" / "camera_ready"
    for h_path in sorted(sweep_dir.glob("compression_lambda_rate_*_history.json")):
        h = load_history(h_path)
        if not h:
            continue
        train = h.get("history", {}).get("train", {})
        val = h.get("history", {}).get("val", {})
        bpe_final = float(train.get("bpe", [np.nan])[-1]) if train.get("bpe") else float("nan")
        ssim = float(h.get("best_val_ssim", 0.0))
        psnr = ((float(val["psnr_A2B"][-1]) + float(val["psnr_B2A"][-1])) / 2.0
                if val.get("psnr_A2B") else None)
        try:
            lam = float(h_path.stem.split("_")[-2])
        except Exception:
            lam = float("nan")
        points.append({
            "lambda_rate": lam, "bpe": bpe_final, "ssim": ssim, "psnr": psnr,
            "source": "new (50 epochs)",
        })

    if not points:
        print("ext b: no compression runs available.")
        return

    points.sort(key=lambda p: (p["bpe"] if p["bpe"] is not None and not np.isnan(p["bpe"]) else 0))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    bpes = [p["bpe"] for p in points if p["bpe"] is not None and not np.isnan(p["bpe"])]
    ssims = [p["ssim"] for p in points if p["bpe"] is not None and not np.isnan(p["bpe"])]
    psnrs = [p["psnr"] for p in points if p["psnr"] is not None]

    axes[0].plot(bpes, ssims, "o-", color="#059669")
    for p in points:
        axes[0].annotate(f"$\\lambda$={p['lambda_rate']:.3f}",
                         (p["bpe"], p["ssim"]),
                         textcoords="offset points", xytext=(5, 5), fontsize=7)
    axes[0].set_xlabel("Bits Per Element (BPE)")
    axes[0].set_ylabel("Best Val SSIM")
    axes[0].set_title("(a) Rate vs SSIM")
    axes[0].grid(True, alpha=0.3)

    if psnrs:
        bpes2 = [p["bpe"] for p in points if p["psnr"] is not None]
        axes[1].plot(bpes2, psnrs, "s-", color="#2563EB")
        axes[1].set_xlabel("Bits Per Element (BPE)")
        axes[1].set_ylabel("Mean Val PSNR (dB)")
        axes[1].set_title("(b) Rate vs PSNR")
        axes[1].grid(True, alpha=0.3)

    fig.suptitle("Extension B: Rate-Distortion Curve",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ext_b_rate_distortion.pdf")
    fig.savefig(out_dir / "fig_ext_b_rate_distortion.png")
    plt.close(fig)

    # latex table
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Extension B rate-distortion sweep over $\\lambda_{\\mathrm{rate}}$. "
                 "Lower BPE means smaller compressed code; higher SSIM/PSNR means better "
                 "reconstruction quality at that rate.}")
    lines.append("\\label{tab:ext_b_rd}")
    lines.append("\\begin{tabular}{cccccl}")
    lines.append("\\toprule")
    lines.append("$\\lambda_{\\mathrm{rate}}$ & BPE & Val SSIM & Val PSNR & "
                 "Source \\\\")
    lines.append("\\midrule")
    for p in points:
        psnr_str = f"{p['psnr']:.2f}" if p["psnr"] is not None else "--"
        lines.append(
            f"{p['lambda_rate']:.3f} & {p['bpe']:.3f} & "
            f"{p['ssim']:.4f} & {psnr_str} & {p['source']} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    (out_dir / "table_camera_ready_ext_b.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# extension c: matrix figure (already produced by inference script; re-link)
# ---------------------------------------------------------------------------


def fig_ext_c_matrix(repo_root: Path, out_dir: Path) -> None:
    src = repo_root / "journal_extension" / "results" / "camera_ready" / "multidomain_matrix"
    if not src.exists():
        print("ext c: matrix inference outputs not found.")
        return
    for f in src.glob("fig_multidomain_matrix_modality*.pdf"):
        target = out_dir / f.name.replace("fig_multidomain_matrix",
                                          "fig_ext_c_matrix")
        target.write_bytes(f.read_bytes())
    for f in src.glob("fig_multidomain_matrix_modality*.png"):
        target = out_dir / f.name.replace("fig_multidomain_matrix",
                                          "fig_ext_c_matrix")
        target.write_bytes(f.read_bytes())


# ---------------------------------------------------------------------------
# extension d: task-aware downstream comparison
# ---------------------------------------------------------------------------


def fig_ext_d_taskaware(repo_root: Path, out_dir: Path) -> None:
    legacy = json.load(open(repo_root / "journal_extension" / "results" / "all_results.json"))
    legacy_d = legacy.get("downstream", {})
    if not legacy_d:
        return

    new_path = repo_root / "journal_extension" / "results" / "camera_ready" / "downstream_taskaware.json"
    new_d = json.load(open(new_path)) if new_path.exists() else {}

    a2b_old = float(legacy_d["improvement"]["dice_mean_foreground_relative_pct"])
    b2a_old = float(legacy_d["improvement_reverse"]["dice_mean_foreground_relative_pct"])

    a2b_new = (float(new_d["improvement"]["dice_mean_foreground_relative_pct"])
               if new_d.get("improvement") else None)
    b2a_new = (float(new_d["improvement_reverse"]["dice_mean_foreground_relative_pct"])
               if new_d.get("improvement_reverse") else None)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    labels = ["A$\\rightarrow$B", "B$\\rightarrow$A"]
    x = np.arange(len(labels))
    width = 0.36
    old = [a2b_old, b2a_old]
    new = [a2b_new if a2b_new is not None else float("nan"),
           b2a_new if b2a_new is not None else float("nan")]
    ax.bar(x - width / 2, old, width, color="#9CA3AF", label="Vanilla harmonization")
    ax.bar(x + width / 2, new, width, color="#2563EB", label="Task-aware harmonization (new)")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Relative $\\Delta$ Dice (\\%)")
    ax.set_title("Extension D: Task-Aware vs Vanilla Harmonization Effect on Cross-Site Dice")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ext_d_taskaware_dice.pdf")
    fig.savefig(out_dir / "fig_ext_d_taskaware_dice.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# extension e: fedavg / fedprox / scaffold convergence
# ---------------------------------------------------------------------------


def fig_ext_e_federated_compare(repo_root: Path, out_dir: Path) -> None:
    legacy = json.load(open(repo_root / "journal_extension" / "results" / "all_results.json"))
    fed_legacy = legacy.get("federated", {})

    series: Dict[str, Dict] = {}
    if fed_legacy.get("history", {}).get("global_metrics"):
        series["FedAvg (legacy 40r)"] = fed_legacy["history"]

    cr_root = repo_root / "journal_extension" / "results" / "camera_ready"
    for strat_name, strat_label in [
        ("federated_fedprox", "FedProx (20r)"),
        ("federated_scaffold", "SCAFFOLD (20r)"),
    ]:
        for h_path in cr_root.glob(f"{strat_name}*history.json"):
            series[strat_label] = load_history(h_path).get("history", {})

    if len(series) < 2:
        print("ext e: insufficient federated runs for comparison.")
        return

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for label, hist in series.items():
        gms = hist.get("global_metrics") or []
        if not gms:
            continue
        rs = [g["round"] for g in gms]
        sa = [g["metrics"]["ssim_A2B"] for g in gms]
        sb = [g["metrics"]["ssim_B2A"] for g in gms]
        mean = [(a + b) / 2.0 for a, b in zip(sa, sb)]
        ax.plot(rs, mean, "o-", label=label)
    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Mean Global SSIM")
    ax.set_title("Extension E: FedAvg vs FedProx vs SCAFFOLD Convergence")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ext_e_federated_compare.pdf")
    fig.savefig(out_dir / "fig_ext_e_federated_compare.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def main():
    plt.rcParams.update(PUB_STYLE)
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "journal_extension" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_ext_a_seeds(repo_root, out_dir)
    fig_ext_b_rate_distortion(repo_root, out_dir)
    fig_ext_c_matrix(repo_root, out_dir)
    fig_ext_d_taskaware(repo_root, out_dir)
    fig_ext_e_federated_compare(repo_root, out_dir)

    print(f"done. wrote camera-ready figures to {out_dir}")


if __name__ == "__main__":
    main()
