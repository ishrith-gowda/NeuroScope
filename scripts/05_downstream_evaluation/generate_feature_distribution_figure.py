#!/usr/bin/env python3
"""
generate publication figure for feature distribution metrics.

shows fid, kid, mmd, swd distances before/after harmonization
with proper latex rendering.
"""

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def setup_latex_style():
    """configure matplotlib for proper latex rendering."""
    sns.set_theme(style='whitegrid')

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 14,
        "font.size": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 18,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3
    })


# custom color palette
COLORS = {
    'raw': '#F4A582',        # coral/salmon
    'harmonized': '#92C5DE', # light blue
    'positive': '#98D4BB',   # mint (improvement)
    'negative': '#F2A7B3',   # light pink (degradation)
}


def create_feature_distribution_figure(results: dict, output_path: Path):
    """
    create feature distribution metrics comparison figure.

    2-panel layout:
    - (a) distance metrics bar chart (raw vs harmonized)
    - (b) percent change bar chart
    """
    setup_latex_style()

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.25, top=0.85)

    # force square subplots
    for ax in axes:
        ax.set_box_aspect(1)

    raw = results['raw']
    harmonized = results.get('harmonized', None)

    metrics = ['fid', 'kid_mean', 'mmd_rbf', 'sliced_wasserstein']
    metric_labels = ['FID', 'KID', 'MMD (RBF)', 'SWD']

    raw_values = [raw.get(m, 0) for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35

    # (a) distance metrics comparison
    ax = axes[0]

    if harmonized:
        harm_values = [harmonized.get(m, 0) for m in metrics]

        bars1 = ax.bar(x - width/2, raw_values, width,
                      label='Raw (Before)', color=COLORS['raw'],
                      edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, harm_values, width,
                      label='Harmonized (After)', color=COLORS['harmonized'],
                      edgecolor='black', linewidth=0.5)

        # add value labels
        for bar, val in zip(bars1, raw_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10, rotation=30)
        for bar, val in zip(bars2, harm_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10, rotation=30)
    else:
        bars1 = ax.bar(x, raw_values, width, label='Raw',
                      color=COLORS['raw'], edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars1, raw_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10, rotation=30)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel('Distance')
    ax.set_ylim(0, max(raw_values + (harm_values if harmonized else [])) * 1.1)
    ax.set_title(r'(a) Feature Distribution Distances', fontsize=15)
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')

    # (b) percent change
    ax = axes[1]

    if harmonized:
        reductions = []
        for r, h in zip(raw_values, harm_values):
            if r > 0:
                reductions.append((r - h) / r * 100)
            else:
                reductions.append(0)

        colors = [COLORS['positive'] if r > 0 else COLORS['negative'] for r in reductions]
        bars = ax.bar(x, reductions, width * 1.5, color=colors,
                     edgecolor='black', linewidth=0.5)

        # add value labels
        for bar, val in zip(bars, reductions):
            height = bar.get_height()
            ypos = height + 1 if height >= 0 else height - 3
            va = 'bottom' if height >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                   f'{val:.1f}\\%', ha='center', va=va, fontsize=11)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel(r'Reduction (\%)')
    ax.set_title(r'(b) Harmonization Effect', fontsize=15)

    # main figure title
    fig.suptitle(r'\textbf{Feature Distribution Analysis: Raw vs. Harmonized}',
                 fontsize=18, fontweight='bold', y=0.96)

    # save
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

    print(f'[fig] saved feature distribution figure to {output_path}')


def main():
    # paths
    project_root = Path(__file__).parent.parent.parent
    results_path = project_root / 'experiments' / 'downstream_evaluation' / 'feature_distribution' / 'feature_distribution_results.json'
    output_dir = project_root / 'figures' / 'downstream'

    output_dir.mkdir(parents=True, exist_ok=True)

    print('[feature] loading results...')
    with open(results_path) as f:
        results = json.load(f)

    print('[feature] generating figure...')
    create_feature_distribution_figure(results, output_dir / 'fig_feature_distribution.pdf')

    print('=' * 60)
    print('[feature] figure generation complete')


if __name__ == '__main__':
    main()
