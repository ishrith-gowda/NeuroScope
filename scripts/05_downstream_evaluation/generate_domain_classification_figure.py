#!/usr/bin/env python3
"""
generate publication figure for domain classification results.

shows domain classification performance before/after harmonization
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


# custom color palette - varied and visually appealing
COLORS = {
    'raw': '#F4A582',       # coral/salmon
    'harmonized': '#92C5DE', # light blue
    'domain_a': '#B8A9C9',   # lavender
    'domain_b': '#98D4BB',   # mint
}


def create_domain_classification_figure(results: dict, output_path: Path):
    """
    create domain classification comparison figure.

    3-panel layout:
    - (a) classification metrics bar chart
    - (b) confusion matrix (raw)
    - (c) confusion matrix (harmonized)
    """
    setup_latex_style()

    fig, axes = plt.subplots(1, 3, figsize=(28, 9))
    plt.subplots_adjust(wspace=0.25, top=0.85)

    # force all subplots to be square and same size
    for ax in axes:
        ax.set_box_aspect(1)

    raw = results['raw']
    harmonized = results.get('harmonized', None)

    # (a) classification metrics
    ax = axes[0]
    metrics = ['accuracy', 'auc', 'f1']
    metric_labels = ['Accuracy', 'AUC-ROC', 'F1-Score']

    raw_values = [raw.get(m, 0) for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35

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
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11)
        for bar, val in zip(bars2, harm_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    else:
        bars1 = ax.bar(x, raw_values, width, label='Raw',
                      color=COLORS['raw'], edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars1, raw_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel('Score')
    ax.set_title(r'(a) Domain Classification Metrics', fontsize=15)
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')

    # add horizontal line at 0.5 (chance level)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    # dashed line only, no label

    # (b) confusion matrix - raw
    ax = axes[1]
    if 'confusion_matrix' in raw:
        cm_raw = np.array(raw['confusion_matrix'])
        im1 = ax.imshow(cm_raw, cmap='OrRd', vmin=0, aspect='auto')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['BraTS', 'UPenn'])
        ax.set_yticklabels(['BraTS', 'UPenn'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(r'(b) Confusion Matrix (Raw)', fontsize=15)

        # add text annotations
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm_raw[i, j]}',
                       ha='center', va='center', fontsize=16, fontweight='bold',
                       color='white' if cm_raw[i, j] > cm_raw.max()/2 else 'black')

        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    # (c) confusion matrix - harmonized
    ax = axes[2]
    if harmonized and 'confusion_matrix' in harmonized:
        cm_harm = np.array(harmonized['confusion_matrix'])
        im2 = ax.imshow(cm_harm, cmap='Blues', vmin=0, aspect='auto')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['BraTS', 'UPenn'])
        ax.set_yticklabels(['BraTS', 'UPenn'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(r'(c) Confusion Matrix (Harmonized)', fontsize=15)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm_harm[i, j]}',
                       ha='center', va='center', fontsize=16, fontweight='bold',
                       color='white' if cm_harm[i, j] > cm_harm.max()/2 else 'black')

        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.axis('off')

    # main figure title
    fig.suptitle(r'\textbf{Domain Classification: Evaluating Harmonization Effectiveness}',
                 fontsize=18, fontweight='bold', y=0.96)

    # save
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

    print(f'[fig] saved domain classification figure to {output_path}')


def main():
    # paths
    project_root = Path(__file__).parent.parent.parent
    results_path = project_root / 'experiments' / 'downstream_evaluation' / 'domain_classification' / 'domain_classification_results.json'
    output_dir = project_root / 'figures' / 'downstream'

    output_dir.mkdir(parents=True, exist_ok=True)

    print('[domain] loading results...')
    with open(results_path) as f:
        results = json.load(f)

    print('[domain] generating figure...')
    create_domain_classification_figure(results, output_dir / 'fig_domain_classification.pdf')

    print('=' * 60)
    print('[domain] figure generation complete')


if __name__ == '__main__':
    main()
