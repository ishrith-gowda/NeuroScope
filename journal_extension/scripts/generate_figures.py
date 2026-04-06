"""
generate all journal extension figures for sa-cyclegan-2.5d.

produces publication-quality figures for:
  1. federated learning convergence
  2. compression training and rate-distortion
  3. multi-domain training dynamics
  4. downstream segmentation evaluation
  5. cross-experiment summary
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# publication style
plt.rcParams.update({
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
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
})

COLORS = {
    "primary": "#2563EB",
    "secondary": "#DC2626",
    "tertiary": "#059669",
    "quaternary": "#D97706",
    "gray": "#6B7280",
    "light_blue": "#93C5FD",
    "light_red": "#FCA5A5",
    "light_green": "#6EE7B7",
}


def load_results(results_path):
    with open(results_path) as f:
        return json.load(f)


# =========================================================================
# figure 1: federated learning convergence
# =========================================================================
def fig_federated_convergence(data, out_dir):
    """federated ssim convergence over communication rounds."""
    fed = data["federated"]
    history = fed["history"]
    global_metrics = history["global_metrics"]

    rounds_with_metrics = [m["round"] for m in global_metrics]
    ssim_a2b = [m["metrics"]["ssim_A2B"] for m in global_metrics]
    ssim_b2a = [m["metrics"]["ssim_B2A"] for m in global_metrics]

    round_times = [r["time"] for r in history["rounds"]]
    round_ids = list(range(len(round_times)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    # panel a: ssim vs round
    ax1.plot(rounds_with_metrics, ssim_a2b, "o-", color=COLORS["primary"],
             label=r"SSIM $A \to B$", markersize=6)
    ax1.plot(rounds_with_metrics, ssim_b2a, "s--", color=COLORS["secondary"],
             label=r"SSIM $B \to A$", markersize=6)
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Structural Similarity (SSIM)")
    ax1.set_title("(a) Federated Harmonization Quality")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0.990, 1.0)
    ax1.axhline(y=fed["best_ssim"], color=COLORS["gray"], linestyle=":",
                alpha=0.5, label=f'Best: {fed["best_ssim"]:.4f}')

    # panel b: round time
    ax2.bar(round_ids, [t / 60 for t in round_times],
            color=COLORS["primary"], alpha=0.7, width=0.8)
    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("Round Duration (min)")
    ax2.set_title("(b) Per-Round Training Time")
    mean_time = np.mean(round_times) / 60
    ax2.axhline(y=mean_time, color=COLORS["secondary"], linestyle="--",
                label=f"Mean: {mean_time:.1f} min")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(out_dir / "fig_federated_convergence.pdf")
    fig.savefig(out_dir / "fig_federated_convergence.png")
    plt.close(fig)
    print("  saved: fig_federated_convergence")


# =========================================================================
# figure 2: compression training curves
# =========================================================================
def fig_compression_training(data, out_dir):
    """compression experiment training dynamics."""
    comp = data["compression"]
    history = comp["history"]
    train = history["train"]
    epochs = list(range(len(train["G_loss"])))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # panel a: generator and discriminator loss
    ax = axes[0, 0]
    ax.plot(epochs, train["G_loss"], color=COLORS["primary"], label="Generator", alpha=0.8)
    ax.plot(epochs, train["D_loss"], color=COLORS["secondary"], label="Discriminator", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("(a) Adversarial Losses")
    ax.legend()
    ax.set_yscale("log")

    # panel b: rate loss
    ax = axes[0, 1]
    ax.plot(epochs, train["rate_loss"], color=COLORS["tertiary"], alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Rate Loss")
    ax.set_title("(b) Rate Loss (Entropy Model)")
    ax.axvline(x=20, color=COLORS["gray"], linestyle="--", alpha=0.5,
               label="Compression Warmup End")
    ax.legend()

    # panel c: bits per element
    ax = axes[1, 0]
    ax.plot(epochs, train["bpe"], color=COLORS["quaternary"], alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bits per Element (bpe)")
    ax.set_title("(c) Bitrate Over Training")
    ax.axvline(x=20, color=COLORS["gray"], linestyle="--", alpha=0.5)

    # panel d: cycle loss or g_loss detail
    ax = axes[1, 1]
    if train.get("cycle_loss") and len(train["cycle_loss"]) > 0:
        ax.plot(epochs, train["cycle_loss"], color=COLORS["primary"], alpha=0.8)
        ax.set_ylabel("Cycle Consistency Loss")
        ax.set_title("(d) Cycle Consistency Loss")
    else:
        # show g_loss on linear scale as alternative
        ax.plot(epochs, train["G_loss"], color=COLORS["primary"], alpha=0.8)
        ax.set_ylabel("Generator Loss")
        ax.set_title("(d) Generator Loss (Linear Scale)")
    ax.set_xlabel("Epoch")

    plt.tight_layout()
    fig.savefig(out_dir / "fig_compression_training.pdf")
    fig.savefig(out_dir / "fig_compression_training.png")
    plt.close(fig)
    print("  saved: fig_compression_training")


# =========================================================================
# figure 3: multi-domain training curves
# =========================================================================
def fig_multidomain_training(data, out_dir):
    """multi-domain training dynamics with 4 scanner domains."""
    md = data["multi_domain"]
    history = md["history"]
    train = history["train"]
    epochs = list(range(len(train["G_loss"])))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # panel a: generator loss
    ax = axes[0, 0]
    ax.plot(epochs, train["G_loss"], color=COLORS["primary"], alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Generator Loss")
    ax.set_title("(a) Generator Loss")

    # panel b: discriminator loss
    ax = axes[0, 1]
    ax.plot(epochs, train["D_loss"], color=COLORS["secondary"], alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Discriminator Loss")
    ax.set_title("(b) Discriminator Loss")

    # panel c: domain classification loss
    ax = axes[1, 0]
    ax.plot(epochs, train["cls_loss"], color=COLORS["tertiary"], alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Classification Loss")
    ax.set_title("(c) Domain Classification Loss")

    # panel d: reconstruction loss
    ax = axes[1, 1]
    ax.plot(epochs, train["rec_loss"], color=COLORS["quaternary"], alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction Loss")
    ax.set_title("(d) Reconstruction Loss")

    plt.tight_layout()
    fig.savefig(out_dir / "fig_multidomain_training.pdf")
    fig.savefig(out_dir / "fig_multidomain_training.png")
    plt.close(fig)
    print("  saved: fig_multidomain_training")


# =========================================================================
# figure 4: downstream segmentation evaluation
# =========================================================================
def fig_downstream_segmentation(data, out_dir):
    """downstream segmentation evaluation: raw vs harmonized."""
    ds = data["downstream"]

    conditions = {
        r"Raw $A \to B$": ds["raw_a_to_raw_b"],
        r"Harm. $A \to B$": ds["harm_a_to_harm_b"],
        r"Raw $A \to A$": ds["raw_a_to_raw_a"],
        r"Harm. $A \to A$": ds["harm_a_to_harm_a"],
        r"Raw $B \to A$": ds["raw_b_to_raw_a"],
        r"Harm. $B \to A$": ds["harm_b_to_harm_a"],
    }

    metrics = ["dice_mean_foreground_mean", "region_wt_mean", "region_tc_mean", "region_et_mean"]
    metric_labels = ["Mean Dice", "Whole Tumor", "Tumor Core", "Enhancing Tumor"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    x = np.arange(len(conditions))
    labels = list(conditions.keys())
    raw_colors = [COLORS["primary"], COLORS["secondary"], COLORS["primary"],
                  COLORS["secondary"], COLORS["primary"], COLORS["secondary"]]
    bar_colors = [COLORS["primary"], COLORS["light_blue"],
                  COLORS["tertiary"], COLORS["light_green"],
                  COLORS["quaternary"], COLORS["light_red"]]

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        values = [cond_data[metric] for cond_data in conditions.values()]
        bars = ax.bar(x, values, color=bar_colors, width=0.7, edgecolor="white", linewidth=0.5)
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0.4, 1.0)
        ax.set_title(label)

    plt.tight_layout()
    fig.savefig(out_dir / "fig_downstream_segmentation.pdf")
    fig.savefig(out_dir / "fig_downstream_segmentation.png")
    plt.close(fig)
    print("  saved: fig_downstream_segmentation")


# =========================================================================
# figure 5: downstream hd95 comparison
# =========================================================================
def fig_downstream_hd95(data, out_dir):
    """hausdorff distance comparison across conditions."""
    ds = data["downstream"]

    conditions = [
        (r"Raw $A \to B$", ds["raw_a_to_raw_b"]),
        (r"Harm. $A \to B$", ds["harm_a_to_harm_b"]),
        (r"Raw $A \to A$", ds["raw_a_to_raw_a"]),
        (r"Harm. $A \to A$", ds["harm_a_to_harm_a"]),
        (r"Raw $B \to A$", ds["raw_b_to_raw_a"]),
        (r"Harm. $B \to A$", ds["harm_b_to_harm_a"]),
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    names = [c[0] for c in conditions]
    means = [c[1]["hd95_mean"] for c in conditions]
    stds = [c[1]["hd95_std"] for c in conditions]
    colors = [COLORS["primary"], COLORS["light_blue"],
              COLORS["tertiary"], COLORS["light_green"],
              COLORS["quaternary"], COLORS["light_red"]]

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, width=0.6,
                  edgecolor="white", linewidth=0.5, capsize=3, error_kw={"linewidth": 1})
    ax.set_ylabel("HD95 (mm)")
    ax.set_title("95th Percentile Hausdorff Distance by Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")

    # add value labels
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "fig_downstream_hd95.pdf")
    fig.savefig(out_dir / "fig_downstream_hd95.png")
    plt.close(fig)
    print("  saved: fig_downstream_hd95")


# =========================================================================
# figure 6: cross-experiment summary radar chart
# =========================================================================
def fig_experiment_summary(data, out_dir):
    """summary of all extension contributions."""
    fed = data["federated"]
    comp = data["compression"]
    md = data["multi_domain"]
    ds = data["downstream"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # panel a: federated ssim trajectory
    ax = axes[0]
    gm = fed["history"]["global_metrics"]
    rounds = [m["round"] for m in gm]
    ssim_avg = [(m["metrics"]["ssim_A2B"] + m["metrics"]["ssim_B2A"]) / 2 for m in gm]
    ax.plot(rounds, ssim_avg, "o-", color=COLORS["primary"], markersize=6)
    ax.fill_between(rounds, [s - 0.001 for s in ssim_avg], [s + 0.001 for s in ssim_avg],
                    color=COLORS["primary"], alpha=0.15)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Average SSIM")
    ax.set_title("(a) Federated Convergence")
    ax.set_ylim(0.993, 1.0)
    ax.annotate(f"Best: {fed['best_ssim']:.4f}", xy=(rounds[-1], ssim_avg[-1]),
                xytext=(-60, -20), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=COLORS["gray"]),
                fontsize=8, color=COLORS["primary"])

    # panel b: compression bpe trajectory
    ax = axes[1]
    bpe = comp["history"]["train"]["bpe"]
    comp_epochs = list(range(len(bpe)))
    ax.plot(comp_epochs, bpe, color=COLORS["tertiary"], alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bits per Element")
    ax.set_title("(b) Compression Bitrate")
    ax.axvline(x=20, color=COLORS["gray"], linestyle="--", alpha=0.5)
    ax.annotate(f"Final: {bpe[-1]:.2f} bpe\nSSIM: {comp['best_val_ssim']:.3f}",
                xy=(len(bpe) - 1, bpe[-1]),
                xytext=(-80, 20), textcoords="offset points",
                fontsize=8, color=COLORS["tertiary"])

    # panel c: multi-domain cls loss
    ax = axes[2]
    cls_loss = md["history"]["train"]["cls_loss"]
    md_epochs = list(range(len(cls_loss)))
    ax.plot(md_epochs, cls_loss, color=COLORS["quaternary"], alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Domain Classification Loss")
    ax.set_title("(c) Multi-Domain Classification")
    ax.annotate(f"Final: {cls_loss[-1]:.4f}", xy=(len(cls_loss) - 1, cls_loss[-1]),
                xytext=(-80, 20), textcoords="offset points",
                fontsize=8, color=COLORS["quaternary"])

    plt.tight_layout()
    fig.savefig(out_dir / "fig_experiment_summary.pdf")
    fig.savefig(out_dir / "fig_experiment_summary.png")
    plt.close(fig)
    print("  saved: fig_experiment_summary")


# =========================================================================
# figure 7: learning rate schedules
# =========================================================================
def fig_learning_rates(data, out_dir):
    """learning rate schedules across experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # federated (per-round, extract from round times as proxy)
    ax = axes[0]
    fed_rounds = data["federated"]["history"]["rounds"]
    round_losses = [r.get("avg_losses", {}).get("G_loss", 0) for r in fed_rounds]
    ax.plot(range(len(round_losses)), round_losses, color=COLORS["primary"], alpha=0.8)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Avg. Generator Loss")
    ax.set_title("(a) Federated: Per-Round G Loss")

    # compression
    ax = axes[1]
    lr = data["compression"]["history"]["learning_rate"]
    ax.plot(range(len(lr)), lr, color=COLORS["tertiary"], alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("(b) Compression: LR Schedule")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))

    # multi-domain
    ax = axes[2]
    lr = data["multi_domain"]["history"]["learning_rate"]
    ax.plot(range(len(lr)), lr, color=COLORS["quaternary"], alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("(c) Multi-Domain: LR Schedule")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))

    plt.tight_layout()
    fig.savefig(out_dir / "fig_learning_rates.pdf")
    fig.savefig(out_dir / "fig_learning_rates.png")
    plt.close(fig)
    print("  saved: fig_learning_rates")


# =========================================================================
# figure 8: epoch timing comparison
# =========================================================================
def fig_epoch_times(data, out_dir):
    """training time comparison across experiments."""
    fig, ax = plt.subplots(figsize=(8, 4))

    # federated
    fed_times = [r["time"] / 60 for r in data["federated"]["history"]["rounds"]]
    comp_times = [t / 60 for t in data["compression"]["history"]["epoch_times"]]
    md_times = [t / 60 for t in data["multi_domain"]["history"]["epoch_times"]]

    experiments = ["Federated\n(per round)", "Compression\n(per epoch)", "Multi-Domain\n(per epoch)"]
    all_times = [fed_times, comp_times, md_times]
    positions = [1, 2, 3]
    colors = [COLORS["primary"], COLORS["tertiary"], COLORS["quaternary"]]

    bp = ax.boxplot(all_times, positions=positions, widths=0.5, patch_artist=True,
                    showmeans=True, meanline=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xticklabels(experiments)
    ax.set_ylabel("Duration (minutes)")
    ax.set_title("Training Duration Distribution")

    # annotate means
    for i, times in enumerate(all_times):
        mean_val = np.mean(times)
        ax.annotate(f"{mean_val:.1f} min", xy=(positions[i], mean_val),
                    xytext=(20, 5), textcoords="offset points", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "fig_epoch_times.pdf")
    fig.savefig(out_dir / "fig_epoch_times.png")
    plt.close(fig)
    print("  saved: fig_epoch_times")


# =========================================================================
# table 1: downstream segmentation results (latex)
# =========================================================================
def table_downstream_results(data, out_dir):
    """generate latex table for downstream segmentation results."""
    ds = data["downstream"]

    conditions = [
        ("Raw $A \\to B$ (cross-site)", ds["raw_a_to_raw_b"]),
        ("Harmonized $A \\to B$ (cross-site)", ds["harm_a_to_harm_b"]),
        ("Raw $A \\to A$ (within-site)", ds["raw_a_to_raw_a"]),
        ("Harmonized $A \\to A$ (within-site)", ds["harm_a_to_harm_a"]),
        ("Raw $B \\to A$ (cross-site, reverse)", ds["raw_b_to_raw_a"]),
        ("Harmonized $B \\to A$ (cross-site, reverse)", ds["harm_b_to_harm_a"]),
    ]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Downstream Segmentation Transfer: Dice Scores and HD95 Across Conditions}")
    lines.append("\\label{tab:downstream}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Condition & Dice $\\uparrow$ & WT $\\uparrow$ & TC $\\uparrow$ & ET $\\uparrow$ & HD95 $\\downarrow$ \\\\")
    lines.append("\\midrule")

    for name, d in conditions:
        dice = d["dice_mean_foreground_mean"]
        wt = d["region_wt_mean"]
        tc = d["region_tc_mean"]
        et = d["region_et_mean"]
        hd = d["hd95_mean"]
        lines.append(f"{name} & {dice:.3f} & {wt:.3f} & {tc:.3f} & {et:.3f} & {hd:.1f} \\\\")
        if name.endswith("(cross-site)") and "Harmonized" in name:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    table_str = "\n".join(lines)
    with open(out_dir / "table_downstream_results.tex", "w") as f:
        f.write(table_str)
    print("  saved: table_downstream_results.tex")


# =========================================================================
# table 2: experiment summary (latex)
# =========================================================================
def table_experiment_summary(data, out_dir):
    """generate latex summary table for all extensions."""
    fed = data["federated"]
    comp = data["compression"]
    md = data["multi_domain"]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Summary of Journal Extension Contributions}")
    lines.append("\\label{tab:extensions}")
    lines.append("\\begin{tabular}{llll}")
    lines.append("\\toprule")
    lines.append("Extension & Method & Key Metric & Value \\\\")
    lines.append("\\midrule")
    lines.append(f"Federated & FedAvg (40 rounds) & Best SSIM & {fed['best_ssim']:.4f} \\\\")
    lines.append(f"Compression & Factorized Entropy & Best Val SSIM & {comp['best_val_ssim']:.4f} \\\\")
    bpe_final = comp["history"]["train"]["bpe"][-1]
    lines.append(f"Compression & Factorized Entropy & Final bpe & {bpe_final:.2f} \\\\")
    cls_final = md["history"]["train"]["cls_loss"][-1]
    rec_final = md["history"]["train"]["rec_loss"][-1]
    lines.append(f"Multi-Domain & AdaIN (4 domains) & Final cls loss & {cls_final:.4f} \\\\")
    lines.append(f"Multi-Domain & AdaIN (4 domains) & Final rec loss & {rec_final:.4f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    table_str = "\n".join(lines)
    with open(out_dir / "table_experiment_summary.tex", "w") as f:
        f.write(table_str)
    print("  saved: table_experiment_summary.tex")


# =========================================================================
# table 3: multi-domain dataset statistics (latex)
# =========================================================================
def table_domain_statistics(data, out_dir):
    """domain statistics table."""
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Multi-Domain Dataset Statistics by Scanner Configuration}")
    lines.append("\\label{tab:domains}")
    lines.append("\\begin{tabular}{llccc}")
    lines.append("\\toprule")
    lines.append("Domain & Scanner & Subjects & Train Slices & Val Slices \\\\")
    lines.append("\\midrule")
    lines.append("BraTS & Multi-institutional & 88 & 6,538 & 825 \\\\")
    lines.append("UPenn 3T TrioTim & Siemens Trio Tim & 434 & 32,266 & 4,046 \\\\")
    lines.append("UPenn 3T Other & 3T (non-TrioTim) & 65 & 4,863 & 601 \\\\")
    lines.append("UPenn 1.5T & 1.5T Scanner & 67 & 4,990 & 610 \\\\")
    lines.append("\\midrule")
    lines.append("\\textbf{Total} & & \\textbf{654} & \\textbf{48,657} & \\textbf{6,082} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    table_str = "\n".join(lines)
    with open(out_dir / "table_domain_statistics.tex", "w") as f:
        f.write(table_str)
    print("  saved: table_domain_statistics.tex")


# =========================================================================
# main
# =========================================================================
def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results"
    figures_dir = script_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "all_results.json"
    if not results_path.exists():
        print(f"error: {results_path} not found")
        return

    print("loading experiment results...")
    data = load_results(results_path)

    print("\ngenerating figures...")
    fig_federated_convergence(data, figures_dir)
    fig_compression_training(data, figures_dir)
    fig_multidomain_training(data, figures_dir)
    fig_downstream_segmentation(data, figures_dir)
    fig_downstream_hd95(data, figures_dir)
    fig_experiment_summary(data, figures_dir)
    fig_learning_rates(data, figures_dir)
    fig_epoch_times(data, figures_dir)

    print("\ngenerating tables...")
    table_downstream_results(data, figures_dir)
    table_experiment_summary(data, figures_dir)
    table_domain_statistics(data, figures_dir)

    print(f"\nall figures saved to: {figures_dir}")
    print(f"total: {len(list(figures_dir.glob('*.pdf')))} PDFs, "
          f"{len(list(figures_dir.glob('*.png')))} PNGs, "
          f"{len(list(figures_dir.glob('*.tex')))} LaTeX tables")


if __name__ == "__main__":
    main()
