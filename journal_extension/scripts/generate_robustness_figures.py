"""
robustness analysis for extension a (patchnce hybrid).

uses the per-slice arrays in test_results_lambda*.json to produce:

  - fig_patchnce_per_slice_box.{pdf,png}  per-slice metric box plot per lambda
  - fig_patchnce_per_slice_cdf.{pdf,png}  empirical cdf per lambda
  - fig_patchnce_diff_violin.{pdf,png}    paired difference violin plots
  - fig_patchnce_scatter_pairs.{pdf,png}  pairwise per-slice scatter
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LAMBDAS: List[float] = [0.1, 0.5, 1.0, 2.0]
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
}


def load_per_slice(results_dir: Path) -> Dict[float, Dict[str, np.ndarray]]:
    out: Dict[float, Dict[str, np.ndarray]] = {}
    for lam in LAMBDAS:
        path = results_dir / f"test_results_lambda{lam}.json"
        if not path.exists():
            continue
        with open(path) as f:
            payload = json.load(f)
        out[lam] = {
            k: np.asarray(v, dtype=np.float64) for k, v in payload.get("per_slice", {}).items()
        }
    return out


def fig_box(per_slice: Dict[float, Dict[str, np.ndarray]], out_dir: Path) -> None:
    metrics = [
        ("ssim_global_A2B", "Global SSIM (A$\\to$B)", (0.93, 1.0)),
        ("ssim_global_B2A", "Global SSIM (B$\\to$A)", (0.93, 1.0)),
        ("psnr_A2B", "PSNR (A$\\to$B, dB)", None),
        ("psnr_B2A", "PSNR (B$\\to$A, dB)", None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.6))
    for ax, (key, title, ylim) in zip(axes, metrics):
        data = []
        labels = []
        cols = []
        for lam in sorted(per_slice.keys()):
            arr = per_slice[lam].get(key)
            if arr is None or len(arr) == 0:
                continue
            data.append(arr)
            labels.append(f"$\\lambda$={lam}")
            cols.append(LAMBDA_COLORS[lam])
        bp = ax.boxplot(
            data,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.0),
        )
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        "Per-Slice Test-Set Robustness Across $\\lambda_{\\mathrm{NCE}}$", fontsize=12, y=1.04
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig_patchnce_per_slice_box.pdf")
    fig.savefig(out_dir / "fig_patchnce_per_slice_box.png")
    plt.close(fig)


def fig_cdf(per_slice: Dict[float, Dict[str, np.ndarray]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    keys = [
        ("ssim_global_A2B", "Global SSIM (A$\\to$B)"),
        ("ssim_global_B2A", "Global SSIM (B$\\to$A)"),
    ]
    for ax, (key, title) in zip(axes, keys):
        for lam in sorted(per_slice.keys()):
            arr = per_slice[lam].get(key)
            if arr is None or len(arr) == 0:
                continue
            sorted_arr = np.sort(arr)
            cdf = np.linspace(0, 1, len(sorted_arr))
            ax.plot(sorted_arr, cdf, color=LAMBDA_COLORS[lam], label=f"$\\lambda$={lam}")
        ax.set_xlabel(title)
        ax.set_ylabel("Empirical CDF")
        ax.set_title(f"Empirical CDF: {title}")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Per-Slice CDF Comparison", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_patchnce_per_slice_cdf.pdf")
    fig.savefig(out_dir / "fig_patchnce_per_slice_cdf.png")
    plt.close(fig)


def fig_diff_violin(per_slice: Dict[float, Dict[str, np.ndarray]], out_dir: Path) -> None:
    """paired differences relative to lambda=1.0 baseline."""
    if 1.0 not in per_slice:
        return

    keys = [
        ("ssim_global_A2B", "$\\Delta$ SSIM A$\\to$B"),
        ("ssim_global_B2A", "$\\Delta$ SSIM B$\\to$A"),
        ("psnr_A2B", "$\\Delta$ PSNR A$\\to$B"),
        ("psnr_B2A", "$\\Delta$ PSNR B$\\to$A"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.6))
    for ax, (key, title) in zip(axes, keys):
        diffs = []
        labels = []
        cols = []
        baseline = per_slice[1.0].get(key)
        if baseline is None:
            continue
        for lam in sorted(per_slice.keys()):
            if lam == 1.0:
                continue
            arr = per_slice[lam].get(key)
            if arr is None or len(arr) != len(baseline):
                continue
            diffs.append(arr - baseline)
            labels.append(f"$\\lambda$={lam}")
            cols.append(LAMBDA_COLORS[lam])
        if not diffs:
            continue
        parts = ax.violinplot(diffs, showmeans=True, showmedians=True, widths=0.85)
        for body, c in zip(parts["bodies"], cols):
            body.set_facecolor(c)
            body.set_alpha(0.55)
            body.set_edgecolor("black")
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_title(f"{title} vs $\\lambda$=1.0")
        ax.set_ylabel(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        "Paired Per-Slice Differences Relative to $\\lambda$=1.0 Baseline", fontsize=12, y=1.04
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig_patchnce_diff_violin.pdf")
    fig.savefig(out_dir / "fig_patchnce_diff_violin.png")
    plt.close(fig)


def fig_scatter_pairs(per_slice: Dict[float, Dict[str, np.ndarray]], out_dir: Path) -> None:
    """4x4 scatter matrix of pairwise per-slice correlations on the global ssim."""
    keys = sorted(per_slice.keys())
    if len(keys) < 2:
        return
    n = len(keys)
    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n), sharex=True, sharey=True)
    for i, la in enumerate(keys):
        for j, lb in enumerate(keys):
            ax = axes[i, j]
            a = per_slice[la].get("ssim_global_A2B")
            b = per_slice[lb].get("ssim_global_A2B")
            if a is None or b is None:
                ax.axis("off")
                continue
            ax.scatter(a, b, s=4, alpha=0.25, color=LAMBDA_COLORS[lb])
            ax.plot([0.92, 1.0], [0.92, 1.0], "k--", linewidth=0.8, alpha=0.5)
            ax.set_xlim(0.92, 1.0)
            ax.set_ylim(0.92, 1.0)
            if i == n - 1:
                ax.set_xlabel(f"$\\lambda$={lb}")
            if j == 0:
                ax.set_ylabel(f"$\\lambda$={la}")
            if i == j:
                ax.set_facecolor("#F3F4F6")
    fig.suptitle("Per-Slice Global SSIM (A$\\to$B): Pairwise Scatter", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_patchnce_scatter_pairs.pdf")
    fig.savefig(out_dir / "fig_patchnce_scatter_pairs.png")
    plt.close(fig)


def main():
    plt.rcParams.update(PUB_STYLE)

    repo_root = Path(__file__).resolve().parents[2]
    res_dir = repo_root / "journal_extension" / "results" / "patchnce"
    fig_dir = repo_root / "journal_extension" / "figures"

    per_slice = load_per_slice(res_dir)
    if not per_slice:
        print("no per-slice arrays found.")
        return

    print(f"loaded per-slice arrays for {len(per_slice)} lambda runs.")
    fig_box(per_slice, fig_dir)
    fig_cdf(per_slice, fig_dir)
    fig_diff_violin(per_slice, fig_dir)
    fig_scatter_pairs(per_slice, fig_dir)
    print(f"done. wrote robustness figures to {fig_dir}")


if __name__ == "__main__":
    main()
