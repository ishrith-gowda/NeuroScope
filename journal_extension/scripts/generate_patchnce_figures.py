"""
publication-quality figures and tables for extension a (patchnce hybrid loss).

inputs:
  - per-lambda training_history.json files in results/patchnce/
  - patchnce_testset_summary.json from eval_patchnce_testset.py (optional)

outputs (figures/ dir):
  - fig_patchnce_val_curves.{pdf,png}     learning curves (val ssim/psnr) per lambda
  - fig_patchnce_loss_components.{pdf,png}  train losses per lambda, four panels
  - fig_patchnce_lambda_sweep.{pdf,png}   best val and test metrics vs lambda
  - fig_patchnce_test_metrics.{pdf,png}   test-set ssim/psnr/mae/mmd bars per lambda
  - fig_patchnce_lr_schedule.{pdf,png}    learning rate schedule (one panel, all overlap)
  - fig_patchnce_epoch_times.{pdf,png}    epoch time distribution
  - fig_patchnce_architecture.{pdf,png}   schematic of the hybrid loss flow
  - fig_patchnce_summary_panel.{pdf,png}  single 3x2 multi-panel summary
  - table_patchnce_ablation.tex            latex ablation table
  - table_patchnce_testset.tex             latex test-set results
  - patchnce_paper_results.json            machine-readable summary blob
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------


LAMBDA_VALUES: List[float] = [0.1, 0.5, 1.0, 2.0]
LAMBDA_LABELS: Dict[float, str] = {
    0.1: r"$\lambda_{\mathrm{NCE}}=0.1$",
    0.5: r"$\lambda_{\mathrm{NCE}}=0.5$",
    1.0: r"$\lambda_{\mathrm{NCE}}=1.0$",
    2.0: r"$\lambda_{\mathrm{NCE}}=2.0$",
}

LAMBDA_COLORS: Dict[float, str] = {
    0.1: "#9CA3AF",
    0.5: "#059669",
    1.0: "#2563EB",
    2.0: "#DC2626",
}


PUB_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_histories(results_dir: Path) -> Dict[float, Dict]:
    """load per-lambda training_history.json files."""
    histories: Dict[float, Dict] = {}
    for lam in LAMBDA_VALUES:
        path = results_dir / f"lambda{lam}_history.json"
        if not path.exists():
            print(f"warning: missing {path}")
            continue
        with open(path) as f:
            histories[lam] = json.load(f)
    return histories


def load_testset_summary(results_dir: Path) -> Dict[float, Dict]:
    """load test-set evaluation results if present."""
    summary_path = results_dir / "patchnce_testset_summary.json"
    if not summary_path.exists():
        print(f"info: {summary_path} not found, test plots will be skipped")
        return {}
    with open(summary_path) as f:
        raw = json.load(f)
    results: Dict[float, Dict] = {}
    for label, payload in raw.items():
        if not label.startswith("lambda"):
            continue
        try:
            lam = float(label.replace("lambda", ""))
        except ValueError:
            continue
        results[lam] = payload
    return results


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def smooth(values, window: int = 5) -> np.ndarray:
    """simple moving average for plotting; preserves length via reflect padding."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < window or window <= 1:
        return arr
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="reflect")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def best_val_metrics(history: Dict) -> Dict[str, float]:
    """extract best val ssim and the corresponding psnr."""
    ssim_a2b = np.asarray(history["val"]["ssim_A2B"], dtype=np.float64)
    ssim_b2a = np.asarray(history["val"]["ssim_B2A"], dtype=np.float64)
    psnr_a2b = np.asarray(history["val"]["psnr_A2B"], dtype=np.float64)
    psnr_b2a = np.asarray(history["val"]["psnr_B2A"], dtype=np.float64)

    mean_ssim = (ssim_a2b + ssim_b2a) / 2.0
    best_idx = int(np.argmax(mean_ssim))

    return {
        "best_epoch": best_idx,
        "best_mean_ssim": float(mean_ssim[best_idx]),
        "ssim_A2B": float(ssim_a2b[best_idx]),
        "ssim_B2A": float(ssim_b2a[best_idx]),
        "psnr_A2B": float(psnr_a2b[best_idx]),
        "psnr_B2A": float(psnr_b2a[best_idx]),
        "final_mean_ssim": float(mean_ssim[-1]),
        "n_epochs": int(len(mean_ssim)),
    }


def save_figure(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def fig_val_curves(histories: Dict[float, Dict], out_dir: Path) -> None:
    """validation ssim and psnr learning curves for all four lambdas."""
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0), sharex=True)

    metrics = [
        ("ssim_A2B", "Validation SSIM (Domain A $\\rightarrow$ B Cycle)", "SSIM", (0.95, 1.0)),
        ("ssim_B2A", "Validation SSIM (Domain B $\\rightarrow$ A Cycle)", "SSIM", (0.95, 1.0)),
        ("psnr_A2B", "Validation PSNR (Domain A $\\rightarrow$ B Cycle)", "PSNR (dB)", None),
        ("psnr_B2A", "Validation PSNR (Domain B $\\rightarrow$ A Cycle)", "PSNR (dB)", None),
    ]

    for ax, (key, title, ylabel, ylim) in zip(axes.ravel(), metrics):
        for lam in LAMBDA_VALUES:
            if lam not in histories:
                continue
            vals = histories[lam]["val"][key]
            epochs = np.arange(len(vals))
            ax.plot(
                epochs,
                smooth(vals, window=5),
                color=LAMBDA_COLORS[lam],
                label=LAMBDA_LABELS[lam],
                linewidth=1.6,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlim(0, 200)

    handles = [
        Line2D([], [], color=LAMBDA_COLORS[l], label=LAMBDA_LABELS[l], linewidth=2.0)
        for l in LAMBDA_VALUES
        if l in histories
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )
    fig.suptitle("PatchNCE Hybrid Loss: Validation Trajectories", fontsize=12, y=1.0)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.99))
    save_figure(fig, out_dir, "fig_patchnce_val_curves")


def fig_loss_components(histories: Dict[float, Dict], out_dir: Path) -> None:
    """four panel: G_loss, D_loss, cycle_loss, nce_loss across epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0), sharex=True)
    keys = [
        ("G_loss", "Generator Loss", "Loss"),
        ("D_loss", "Discriminator Loss", "Loss"),
        ("cycle_loss", "Cycle Reconstruction Loss", "Loss"),
        ("nce_loss", "PatchNCE Loss (Pre-$\\lambda$ Scale)", "Loss"),
    ]

    for ax, (key, title, ylabel) in zip(axes.ravel(), keys):
        for lam in LAMBDA_VALUES:
            if lam not in histories:
                continue
            vals = histories[lam]["train"].get(key, [])
            if not vals:
                continue
            epochs = np.arange(len(vals))
            ax.plot(
                epochs,
                smooth(vals, window=5),
                color=LAMBDA_COLORS[lam],
                label=LAMBDA_LABELS[lam],
                linewidth=1.6,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 200)
        if "nce" in key.lower() or "cycle" in key.lower():
            ax.set_yscale("log")

    handles = [
        Line2D([], [], color=LAMBDA_COLORS[l], label=LAMBDA_LABELS[l], linewidth=2.0)
        for l in LAMBDA_VALUES
        if l in histories
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )
    fig.suptitle("PatchNCE Hybrid Loss: Training Loss Components", fontsize=12, y=1.0)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.99))
    save_figure(fig, out_dir, "fig_patchnce_loss_components")


def fig_lambda_sweep(
    histories: Dict[float, Dict], testset: Dict[float, Dict], out_dir: Path
) -> None:
    """sweep plots: best mean ssim and psnr vs lambda."""
    lams = sorted(histories.keys())
    if not lams:
        return
    val_ssim_means = []
    val_psnr_means = []
    val_ssim_a2b = []
    val_ssim_b2a = []
    val_psnr_a2b = []
    val_psnr_b2a = []

    for lam in lams:
        m = best_val_metrics(histories[lam])
        val_ssim_means.append(m["best_mean_ssim"])
        val_psnr_means.append((m["psnr_A2B"] + m["psnr_B2A"]) / 2.0)
        val_ssim_a2b.append(m["ssim_A2B"])
        val_ssim_b2a.append(m["ssim_B2A"])
        val_psnr_a2b.append(m["psnr_A2B"])
        val_psnr_b2a.append(m["psnr_B2A"])

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))

    ax = axes[0]
    ax.plot(lams, val_ssim_a2b, "o-", color="#2563EB", label="A $\\rightarrow$ B Cycle")
    ax.plot(lams, val_ssim_b2a, "s-", color="#DC2626", label="B $\\rightarrow$ A Cycle")
    ax.plot(lams, val_ssim_means, "d--", color="#111827", label="Mean", linewidth=1.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"PatchNCE Weight $\lambda_{\mathrm{NCE}}$")
    ax.set_ylabel("Best Validation SSIM")
    ax.set_title("SSIM vs $\\lambda_{\\mathrm{NCE}}$")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(lams, val_psnr_a2b, "o-", color="#2563EB", label="A $\\rightarrow$ B Cycle")
    ax.plot(lams, val_psnr_b2a, "s-", color="#DC2626", label="B $\\rightarrow$ A Cycle")
    ax.plot(lams, val_psnr_means, "d--", color="#111827", label="Mean", linewidth=1.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"PatchNCE Weight $\lambda_{\mathrm{NCE}}$")
    ax.set_ylabel("Best Validation PSNR (dB)")
    ax.set_title("PSNR vs $\\lambda_{\\mathrm{NCE}}$")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("PatchNCE Hybrid Loss: $\\lambda_{\\mathrm{NCE}}$ Ablation", fontsize=12, y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_patchnce_lambda_sweep")


def fig_test_metrics(testset: Dict[float, Dict], out_dir: Path) -> None:
    """test-set bar charts: ssim, psnr, mae, mmd, all per lambda."""
    if not testset:
        return
    lams = sorted(testset.keys())
    metrics = [
        ("ssim", "ssim_A2B_mean", "ssim_B2A_mean", "Test SSIM", "SSIM", (0.93, 1.0)),
        ("psnr", "psnr_A2B_mean", "psnr_B2A_mean", "Test PSNR (dB)", "PSNR (dB)", None),
        ("mae", "mae_A2B_mean", "mae_B2A_mean", "Test MAE", "MAE", None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.4))

    for ax, (_, a_key, b_key, title, ylabel, ylim) in zip(axes[:3], metrics):
        a = [testset[lam][a_key] for lam in lams]
        b = [testset[lam][b_key] for lam in lams]
        x = np.arange(len(lams))
        width = 0.36
        ax.bar(x - width / 2, a, width, color="#2563EB", label="A $\\rightarrow$ B Cycle")
        ax.bar(x + width / 2, b, width, color="#DC2626", label="B $\\rightarrow$ A Cycle")
        ax.set_xticks(x)
        ax.set_xticklabels([f"$\\lambda$={l}" for l in lams])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend(loc="upper right" if "mae" in title.lower() else "lower right")
        ax.grid(True, axis="y", alpha=0.3)

    ax = axes[3]
    mmd_a2b = [testset[lam]["mmd_A2B"] for lam in lams]
    mmd_b2a = [testset[lam]["mmd_B2A"] for lam in lams]
    x = np.arange(len(lams))
    width = 0.36
    ax.bar(x - width / 2, mmd_a2b, width, color="#2563EB", label="fake-B vs real-B")
    ax.bar(x + width / 2, mmd_b2a, width, color="#DC2626", label="fake-A vs real-A")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\lambda$={l}" for l in lams])
    ax.set_title("MMD (Domain Alignment, Lower is Better)")
    ax.set_ylabel("MMD")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("PatchNCE Hybrid Loss: Held-Out Test-Set Evaluation", fontsize=12, y=1.04)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_patchnce_test_metrics")


def fig_lr_schedule(histories: Dict[float, Dict], out_dir: Path) -> None:
    """learning rate schedule visualization."""
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for lam in LAMBDA_VALUES:
        if lam not in histories:
            continue
        lrs = histories[lam]["learning_rate"]
        epochs = np.arange(len(lrs))
        ax.plot(epochs, lrs, color=LAMBDA_COLORS[lam], label=LAMBDA_LABELS[lam], linewidth=1.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Generator Learning Rate")
    ax.set_yscale("log")
    ax.set_title("Cosine Annealing with Linear Warmup ($\\eta_0 = 5\\times10^{-5}$)")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_patchnce_lr_schedule")


def fig_epoch_times(histories: Dict[float, Dict], out_dir: Path) -> None:
    """epoch wall-clock distribution per lambda."""
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    boxes = []
    labels = []
    colors = []
    for lam in LAMBDA_VALUES:
        if lam not in histories:
            continue
        times = histories[lam].get("epoch_times", [])
        if not times:
            continue
        boxes.append(times)
        labels.append(f"$\\lambda$={lam}")
        colors.append(LAMBDA_COLORS[lam])
    bp = ax.boxplot(
        boxes,
        tick_labels=labels,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="black", linewidth=1.0),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set_ylabel("Epoch Wall-Clock Time (s)")
    ax.set_title("Per-Epoch Wall-Clock Time on NVIDIA A100-PCIE")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_patchnce_epoch_times")


def fig_architecture(out_dir: Path) -> None:
    """schematic of sa-cyclegan-2.5d + patchnce hybrid loss."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def block(x, y, w, h, label, color):
        rect = Rectangle(
            (x, y), w, h, linewidth=1.2, edgecolor="#1F2937", facecolor=color, alpha=0.85
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9, weight="bold")

    def arrow(x1, y1, x2, y2, color="#1F2937", style="-|>"):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle=style, color=color, mutation_scale=12, linewidth=1.4
            )
        )

    # encoder/decoder paths
    block(0.2, 4.4, 1.4, 0.9, "Real A\n(BraTS)", "#DBEAFE")
    block(0.2, 0.7, 1.4, 0.9, "Real B\n(UPenn-GBM)", "#FEE2E2")

    block(2.2, 4.4, 1.6, 0.9, "Encoder $E_{A\\to B}$", "#BFDBFE")
    block(2.2, 0.7, 1.6, 0.9, "Encoder $E_{B\\to A}$", "#FCA5A5")

    block(4.2, 4.4, 1.6, 0.9, "Self-Attention\nBottleneck", "#A5B4FC")
    block(4.2, 0.7, 1.6, 0.9, "Self-Attention\nBottleneck", "#A5B4FC")

    block(6.2, 4.4, 1.6, 0.9, "Decoder $G_{A\\to B}$", "#BFDBFE")
    block(6.2, 0.7, 1.6, 0.9, "Decoder $G_{B\\to A}$", "#FCA5A5")

    block(8.2, 4.4, 1.5, 0.9, "Fake B", "#3B82F6")
    block(8.2, 0.7, 1.5, 0.9, "Fake A", "#EF4444")

    # forward arrows
    for y in (4.85, 1.15):
        arrow(1.6, y, 2.2, y)
        arrow(3.8, y, 4.2, y)
        arrow(5.8, y, 6.2, y)
        arrow(7.8, y, 8.2, y)

    # cycle arrows
    arrow(8.95, 4.4, 8.95, 1.6, color="#059669", style="->")
    arrow(8.95, 1.6, 8.95, 4.4, color="#059669", style="->")
    ax.text(
        9.1, 3.0, "Cycle\nReconstruction", color="#059669", fontsize=8.5, ha="left", va="center"
    )

    # patchnce branches
    block(2.4, 2.7, 1.5, 0.7, "PatchNCE Heads\n(per-layer MLP)", "#FCD34D")
    block(7.0, 2.7, 1.5, 0.7, "PatchNCE Heads\n(per-layer MLP)", "#FCD34D")

    arrow(2.95, 4.4, 2.95, 3.4, color="#B45309", style="->")
    arrow(7.55, 4.4, 7.55, 3.4, color="#B45309", style="->")
    arrow(3.9, 3.05, 7.0, 3.05, color="#B45309", style="-|>")
    ax.text(
        5.4,
        3.25,
        "Contrastive InfoNCE\n($\\lambda_{\\mathrm{NCE}}$)",
        color="#B45309",
        fontsize=8.5,
        ha="center",
        va="bottom",
    )

    # loss legend
    legend_handles = [
        mpatches.Patch(color="#3B82F6", label="Forward Path A $\\rightarrow$ B"),
        mpatches.Patch(color="#EF4444", label="Forward Path B $\\rightarrow$ A"),
        mpatches.Patch(color="#059669", label="Cycle Consistency"),
        mpatches.Patch(color="#B45309", label="PatchNCE Hybrid"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
    )
    ax.set_title("SA-CycleGAN-2.5D with PatchNCE Hybrid Loss", fontsize=12, pad=10)

    save_figure(fig, out_dir, "fig_patchnce_architecture")


def fig_summary_panel(
    histories: Dict[float, Dict], testset: Dict[float, Dict], out_dir: Path
) -> None:
    """3 by 2 multi-panel summary suitable for the journal main figure."""
    fig = plt.figure(figsize=(11.0, 8.0))
    gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.35, height_ratios=[1.0, 1.0, 0.9])

    ax = fig.add_subplot(gs[0, 0])
    for lam in LAMBDA_VALUES:
        if lam not in histories:
            continue
        ssim_a2b = np.asarray(histories[lam]["val"]["ssim_A2B"])
        ssim_b2a = np.asarray(histories[lam]["val"]["ssim_B2A"])
        mean_ssim = (ssim_a2b + ssim_b2a) / 2.0
        ax.plot(
            np.arange(len(mean_ssim)),
            smooth(mean_ssim, 5),
            color=LAMBDA_COLORS[lam],
            label=LAMBDA_LABELS[lam],
        )
    ax.set_xlim(0, 200)
    ax.set_ylim(0.95, 1.0)
    ax.set_title("(a) Validation Mean SSIM")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SSIM")
    ax.legend(loc="lower right", ncol=2, fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    for lam in LAMBDA_VALUES:
        if lam not in histories:
            continue
        psnr_a2b = np.asarray(histories[lam]["val"]["psnr_A2B"])
        psnr_b2a = np.asarray(histories[lam]["val"]["psnr_B2A"])
        mean_psnr = (psnr_a2b + psnr_b2a) / 2.0
        ax.plot(
            np.arange(len(mean_psnr)),
            smooth(mean_psnr, 5),
            color=LAMBDA_COLORS[lam],
            label=LAMBDA_LABELS[lam],
        )
    ax.set_xlim(0, 200)
    ax.set_title("(b) Validation Mean PSNR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
    ax.legend(loc="lower right", ncol=2, fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    for lam in LAMBDA_VALUES:
        if lam not in histories:
            continue
        cyc = histories[lam]["train"].get("cycle_loss", [])
        ax.plot(
            np.arange(len(cyc)), smooth(cyc, 5), color=LAMBDA_COLORS[lam], label=LAMBDA_LABELS[lam]
        )
    ax.set_xlim(0, 200)
    ax.set_yscale("log")
    ax.set_title("(c) Cycle Reconstruction Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", ncol=2, fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    for lam in LAMBDA_VALUES:
        if lam not in histories:
            continue
        nce = histories[lam]["train"].get("nce_loss", [])
        ax.plot(
            np.arange(len(nce)), smooth(nce, 5), color=LAMBDA_COLORS[lam], label=LAMBDA_LABELS[lam]
        )
    ax.set_xlim(0, 200)
    ax.set_yscale("log")
    ax.set_title("(d) PatchNCE Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", ncol=2, fontsize=8)

    ax = fig.add_subplot(gs[2, 0])
    lams = sorted(histories.keys())
    val_means = [best_val_metrics(histories[l])["best_mean_ssim"] for l in lams]
    if testset:
        test_means = [
            (testset[l]["ssim_A2B_mean"] + testset[l]["ssim_B2A_mean"]) / 2.0
            if l in testset
            else np.nan
            for l in lams
        ]
    else:
        test_means = [np.nan] * len(lams)

    x = np.arange(len(lams))
    width = 0.36
    ax.bar(x - width / 2, val_means, width, color="#2563EB", label="Validation")
    ax.bar(x + width / 2, test_means, width, color="#059669", label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\lambda$={l}" for l in lams])
    ax.set_title("(e) Mean SSIM by Split")
    ax.set_ylabel("Mean SSIM")
    ax.set_ylim(0.95, 1.0)
    ax.legend(loc="lower right")

    ax = fig.add_subplot(gs[2, 1])
    times = []
    for lam in lams:
        ets = histories[lam].get("epoch_times", [])
        times.append(np.mean(ets) if ets else 0.0)
    ax.bar([f"$\\lambda$={l}" for l in lams], times, color=[LAMBDA_COLORS[l] for l in lams])
    ax.set_title("(f) Mean Epoch Time")
    ax.set_ylabel("Seconds")
    for i, t in enumerate(times):
        ax.text(i, t + 5, f"{t:.0f}s", ha="center", fontsize=8.5)

    fig.suptitle(
        "Extension A: SA-CycleGAN-2.5D + PatchNCE Hybrid Loss --- Summary", fontsize=13, y=1.0
    )
    save_figure(fig, out_dir, "fig_patchnce_summary_panel")


# ---------------------------------------------------------------------------
# tables (latex)
# ---------------------------------------------------------------------------


def table_ablation(histories: Dict[float, Dict], testset: Dict[float, Dict], out_dir: Path) -> None:
    """ablation table: lambda vs val ssim/psnr (and test if available)."""
    lams = sorted(histories.keys())
    rows = []
    for lam in lams:
        m = best_val_metrics(histories[lam])
        row = {
            "lambda": lam,
            "epoch": m["best_epoch"],
            "val_ssim_a2b": m["ssim_A2B"],
            "val_ssim_b2a": m["ssim_B2A"],
            "val_ssim_mean": m["best_mean_ssim"],
            "val_psnr_a2b": m["psnr_A2B"],
            "val_psnr_b2a": m["psnr_B2A"],
            "val_psnr_mean": (m["psnr_A2B"] + m["psnr_B2A"]) / 2.0,
            "n_epochs": m["n_epochs"],
        }
        if lam in testset:
            t = testset[lam]
            row.update(
                {
                    "test_ssim_a2b": t["ssim_A2B_mean"],
                    "test_ssim_b2a": t["ssim_B2A_mean"],
                    "test_psnr_a2b": t["psnr_A2B_mean"],
                    "test_psnr_b2a": t["psnr_B2A_mean"],
                    "test_mae_a2b": t["mae_A2B_mean"],
                    "test_mae_b2a": t["mae_B2A_mean"],
                    "test_mmd_a2b": t["mmd_A2B"],
                    "test_mmd_b2a": t["mmd_B2A"],
                }
            )
        rows.append(row)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Extension A ablation. Held-out validation and test metrics for "
        "SA-CycleGAN-2.5D with PatchNCE hybrid loss across $\\lambda_{\\mathrm{NCE}}$ "
        "values. Mean SSIM averages forward and reverse cycle.}"
    )
    lines.append("\\label{tab:patchnce_ablation}")
    has_test = any("test_ssim_a2b" in r for r in rows)
    if has_test:
        lines.append("\\begin{tabular}{lcccccc}")
        lines.append("\\toprule")
        lines.append(
            "$\\lambda_{\\mathrm{NCE}}$ & Best Epoch & "
            "Val SSIM (mean) & Val PSNR (mean) & "
            "Test SSIM (mean) & Test PSNR (mean) & Test MMD (A$\\rightarrow$B) \\\\"
        )
        lines.append("\\midrule")
        best_idx = int(np.argmax([r["val_ssim_mean"] for r in rows]))
        for i, r in enumerate(rows):
            test_ssim_mean = (r["test_ssim_a2b"] + r["test_ssim_b2a"]) / 2.0
            test_psnr_mean = (r["test_psnr_a2b"] + r["test_psnr_b2a"]) / 2.0
            star = "$\\dagger$" if i == best_idx else ""
            lines.append(
                f"{r['lambda']:.1f}{star} & {r['epoch']} & "
                f"{r['val_ssim_mean']:.4f} & {r['val_psnr_mean']:.2f} & "
                f"{test_ssim_mean:.4f} & {test_psnr_mean:.2f} & "
                f"{r['test_mmd_a2b']:.4f} \\\\"
            )
    else:
        lines.append("\\begin{tabular}{lccccc}")
        lines.append("\\toprule")
        lines.append(
            "$\\lambda_{\\mathrm{NCE}}$ & Best Epoch & "
            "SSIM (A$\\rightarrow$B) & SSIM (B$\\rightarrow$A) & "
            "PSNR (A$\\rightarrow$B, dB) & PSNR (B$\\rightarrow$A, dB) \\\\"
        )
        lines.append("\\midrule")
        best_idx = int(np.argmax([r["val_ssim_mean"] for r in rows]))
        for i, r in enumerate(rows):
            star = "$\\dagger$" if i == best_idx else ""
            lines.append(
                f"{r['lambda']:.1f}{star} & {r['epoch']} & "
                f"{r['val_ssim_a2b']:.4f} & {r['val_ssim_b2a']:.4f} & "
                f"{r['val_psnr_a2b']:.2f} & {r['val_psnr_b2a']:.2f} \\\\"
            )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    (out_dir / "table_patchnce_ablation.tex").write_text("\n".join(lines) + "\n")


def table_testset(testset: Dict[float, Dict], out_dir: Path) -> None:
    """detailed test-set table including std and percentiles."""
    if not testset:
        return
    lams = sorted(testset.keys())

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Detailed test-set evaluation of Extension A across all four "
        "$\\lambda_{\\mathrm{NCE}}$ values. Means $\\pm$ standard deviations on "
        "5{,}265 held-out 2.5D slices. MMD computed on global-average-pooled "
        "spatial features (lower is better).}"
    )
    lines.append("\\label{tab:patchnce_testset}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append(
        "$\\lambda_{\\mathrm{NCE}}$ & SSIM (A$\\rightarrow$B) & SSIM (B$\\rightarrow$A) & "
        "PSNR (A$\\rightarrow$B) & PSNR (B$\\rightarrow$A) & "
        "MAE (A$\\rightarrow$B) & MMD (A$\\rightarrow$B) \\\\"
    )
    lines.append("\\midrule")
    for lam in lams:
        t = testset[lam]
        lines.append(
            f"{lam:.1f} & "
            f"{t['ssim_A2B_mean']:.4f} $\\pm$ {t['ssim_A2B_std']:.4f} & "
            f"{t['ssim_B2A_mean']:.4f} $\\pm$ {t['ssim_B2A_std']:.4f} & "
            f"{t['psnr_A2B_mean']:.2f} $\\pm$ {t['psnr_A2B_std']:.2f} & "
            f"{t['psnr_B2A_mean']:.2f} $\\pm$ {t['psnr_B2A_std']:.2f} & "
            f"{t['mae_A2B_mean']:.4f} $\\pm$ {t['mae_A2B_std']:.4f} & "
            f"{t['mmd_A2B']:.4f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    (out_dir / "table_patchnce_testset.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# machine-readable aggregate
# ---------------------------------------------------------------------------


def write_aggregate(
    histories: Dict[float, Dict], testset: Dict[float, Dict], out_dir: Path
) -> None:
    """write a single json blob with the canonical extension a numbers."""
    lams = sorted(histories.keys())
    summary = {"lambdas": lams, "runs": {}}
    for lam in lams:
        m = best_val_metrics(histories[lam])
        run: Dict = {"validation": m, "lambda_nce": lam, "n_epochs": m["n_epochs"]}
        epoch_times = histories[lam].get("epoch_times", [])
        if epoch_times:
            run["epoch_time_mean"] = float(np.mean(epoch_times))
            run["epoch_time_total"] = float(np.sum(epoch_times))
        if lam in testset:
            run["test"] = {k: v for k, v in testset[lam].items() if k != "per_slice"}
        summary["runs"][f"lambda{lam}"] = run

    if histories:
        best_lam = max(
            histories.keys(), key=lambda l: best_val_metrics(histories[l])["best_mean_ssim"]
        )
        summary["best_lambda_by_val_ssim"] = best_lam
    if testset:
        best_test_lam = max(
            testset.keys(),
            key=lambda l: (testset[l]["ssim_A2B_mean"] + testset[l]["ssim_B2A_mean"]) / 2.0,
        )
        summary["best_lambda_by_test_ssim"] = best_test_lam

    (out_dir / "patchnce_paper_results.json").write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def main():
    plt.rcParams.update(PUB_STYLE)

    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / "patchnce"
    out_dir = script_dir.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    histories = load_histories(results_dir)
    if not histories:
        print("no per-lambda histories found, aborting.")
        return

    testset = load_testset_summary(results_dir)

    print(f"loaded {len(histories)} training histories.")
    print(f"loaded {len(testset)} test-set summaries.")

    fig_val_curves(histories, out_dir)
    fig_loss_components(histories, out_dir)
    fig_lambda_sweep(histories, testset, out_dir)
    fig_test_metrics(testset, out_dir)
    fig_lr_schedule(histories, out_dir)
    fig_epoch_times(histories, out_dir)
    fig_architecture(out_dir)
    fig_summary_panel(histories, testset, out_dir)

    table_ablation(histories, testset, out_dir)
    table_testset(testset, out_dir)
    write_aggregate(histories, testset, out_dir.parent / "results")

    print(f"done. wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
