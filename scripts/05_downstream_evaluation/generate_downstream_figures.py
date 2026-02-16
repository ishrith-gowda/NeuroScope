#!/usr/bin/env python3
"""
generate publication-quality figures for downstream task evaluation.

creates visualizations for:
- domain classification performance (before/after harmonization)
- feature distribution analysis (t-sne, pca)
- metric comparison charts (fid, mmd, kid, swd)
- statistical significance visualization
- confusion matrices

all figures designed for top-tier venue submission (miccai, tmi, neuroimage).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats

# set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# color palette for consistency
COLORS = {
    'domain_a': '#2E86AB',  # blue - brats
    'domain_b': '#A23B72',  # magenta - upenn
    'raw': '#E94F37',       # red - raw/unharmonized
    'harmonized': '#44AF69',  # green - harmonized
    'neutral': '#6B7280',   # gray
}


def plot_domain_classification_comparison(
    raw_results: Dict,
    harmonized_results: Optional[Dict],
    output_path: Path
):
    """
    plot domain classification accuracy comparison.

    bar chart comparing raw vs harmonized classification metrics.
    lower values after harmonization indicate better domain adaptation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    metrics = ['accuracy', 'auc', 'f1']
    metric_labels = ['Accuracy', 'AUC-ROC', 'F1-Score']

    raw_values = [raw_results.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    if harmonized_results:
        harm_values = [harmonized_results.get(m, 0) for m in metrics]

        bars1 = axes[0].bar(x - width/2, raw_values, width,
                          label='Raw', color=COLORS['raw'], edgecolor='black', linewidth=0.5)
        bars2 = axes[0].bar(x + width/2, harm_values, width,
                          label='Harmonized', color=COLORS['harmonized'], edgecolor='black', linewidth=0.5)

        # add value labels
        for bar, val in zip(bars1, raw_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, harm_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        bars1 = axes[0].bar(x, raw_values, width,
                          label='Raw', color=COLORS['raw'], edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars1, raw_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Domain Classification Metrics')
    axes[0].legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')

    # add horizontal line at 0.5 (chance level)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[0].text(2.1, 0.51, 'Chance', fontsize=8, color='gray')

    # confusion matrix - raw
    if 'confusion_matrix' in raw_results:
        cm_raw = np.array(raw_results['confusion_matrix'])
        im1 = axes[1].imshow(cm_raw, cmap='Blues', vmin=0)
        axes[1].set_xticks([0, 1])
        axes[1].set_yticks([0, 1])
        axes[1].set_xticklabels(['BraTS', 'UPenn'])
        axes[1].set_yticklabels(['BraTS', 'UPenn'])
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        axes[1].set_title('Confusion Matrix (Raw)')

        # add text annotations
        for i in range(2):
            for j in range(2):
                axes[1].text(j, i, f'{cm_raw[i, j]}',
                           ha='center', va='center', fontsize=12,
                           color='white' if cm_raw[i, j] > cm_raw.max()/2 else 'black')

    # confusion matrix - harmonized
    if harmonized_results and 'confusion_matrix' in harmonized_results:
        cm_harm = np.array(harmonized_results['confusion_matrix'])
        im2 = axes[2].imshow(cm_harm, cmap='Greens', vmin=0)
        axes[2].set_xticks([0, 1])
        axes[2].set_yticks([0, 1])
        axes[2].set_xticklabels(['BraTS', 'UPenn'])
        axes[2].set_yticklabels(['BraTS', 'UPenn'])
        axes[2].set_xlabel('Predicted')
        axes[2].set_ylabel('Actual')
        axes[2].set_title('Confusion Matrix (Harmonized)')

        for i in range(2):
            for j in range(2):
                axes[2].text(j, i, f'{cm_harm[i, j]}',
                           ha='center', va='center', fontsize=12,
                           color='white' if cm_harm[i, j] > cm_harm.max()/2 else 'black')
    else:
        axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.savefig(output_path.with_suffix('.png'), format='png', dpi=300)
    plt.close()

    print(f'[fig] saved domain classification comparison to {output_path}')


# t-SNE visualization color palette
COLORS_TSNE = {
    'brats_raw': '#E8703C',      # vibrant orange
    'upenn_raw': '#6B5A9F',      # deep purple
    'brats_harm': '#2A9D8F',     # teal
    'upenn_harm': '#E76F51',     # burnt sienna
}


def plot_tsne_visualization(
    tsne_raw: np.ndarray,
    labels: np.ndarray,
    tsne_harmonized: Optional[np.ndarray],
    output_path: Path
):
    """
    plot publication-quality t-sne visualization of feature space.

    shows domain separation before/after harmonization with full latex rendering.
    """
    setup_latex_style_publication()
    
    n_cols = 2 if tsne_harmonized is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols + 2, 7))
    plt.subplots_adjust(wspace=0.25, top=0.85)

    if n_cols == 1:
        axes = [axes]

    # raw features
    mask_a = labels == 0
    mask_b = labels == 1

    axes[0].scatter(tsne_raw[mask_a, 0], tsne_raw[mask_a, 1],
                   c=COLORS_TSNE['brats_raw'], alpha=0.6, s=35, 
                   label='BraTS', edgecolors='black', linewidth=0.3)
    axes[0].scatter(tsne_raw[mask_b, 0], tsne_raw[mask_b, 1],
                   c=COLORS_TSNE['upenn_raw'], alpha=0.6, s=35, 
                   label='UPenn', edgecolors='black', linewidth=0.3)

    axes[0].set_xlabel(r't-SNE Dimension 1', fontsize=12)
    axes[0].set_ylabel(r't-SNE Dimension 2', fontsize=12)
    axes[0].set_title(r'(a) Raw Features', fontsize=13, pad=10)
    axes[0].legend(loc='best', frameon=True, fancybox=False, 
                  edgecolor='black', fontsize=11, markerscale=1.2)
    axes[0].grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    axes[0].set_axisbelow(True)

    # remove axis ticks (t-sne coordinates are not meaningful)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # harmonized features
    if tsne_harmonized is not None:
        axes[1].scatter(tsne_harmonized[mask_a, 0], tsne_harmonized[mask_a, 1],
                       c=COLORS_TSNE['brats_harm'], alpha=0.6, s=35, 
                       label='BraTS (Harmonized)', edgecolors='black', linewidth=0.3)
        axes[1].scatter(tsne_harmonized[mask_b, 0], tsne_harmonized[mask_b, 1],
                       c=COLORS_TSNE['upenn_harm'], alpha=0.6, s=35, 
                       label='UPenn (Harmonized)', edgecolors='black', linewidth=0.3)

        axes[1].set_xlabel(r't-SNE Dimension 1', fontsize=12)
        axes[1].set_ylabel(r't-SNE Dimension 2', fontsize=12)
        axes[1].set_title(r'(b) Harmonized Features', fontsize=13, pad=10)
        axes[1].legend(loc='best', frameon=True, fancybox=False, 
                      edgecolor='black', fontsize=11, markerscale=1.2)
        axes[1].grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        axes[1].set_axisbelow(True)
        axes[1].set_xticks([])
        axes[1].set_yticks([])

    # main figure title
    fig.suptitle(r'\textbf{Feature Space Visualization (t-SNE)}',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

    print(f'[fig] saved publication-quality t-sne visualization to {output_path}')


def plot_feature_distribution_metrics(
    raw_metrics: Dict,
    harmonized_metrics: Optional[Dict],
    output_path: Path
):
    """
    plot feature distribution metrics comparison.

    bar chart showing fid, kid, mmd, swd before/after harmonization.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # metrics to plot
    metrics = ['fid', 'kid_mean', 'mmd_rbf', 'sliced_wasserstein']
    metric_labels = ['FID', 'KID', 'MMD (RBF)', 'SWD']

    raw_values = [raw_metrics.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    if harmonized_metrics:
        harm_values = [harmonized_metrics.get(m, 0) for m in metrics]

        # bar chart
        bars1 = axes[0].bar(x - width/2, raw_values, width,
                          label='Raw', color=COLORS['raw'], edgecolor='black', linewidth=0.5)
        bars2 = axes[0].bar(x + width/2, harm_values, width,
                          label='Harmonized', color=COLORS['harmonized'], edgecolor='black', linewidth=0.5)

        # add value labels
        for bar, val in zip(bars1, raw_values):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=45)
        for bar, val in zip(bars2, harm_values):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=45)
    else:
        bars1 = axes[0].bar(x, raw_values, width,
                          label='Raw', color=COLORS['raw'], edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars1, raw_values):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=45)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_ylabel('Distance')
    axes[0].set_title('Feature Distribution Distances')
    axes[0].legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')

    # reduction chart (if harmonized)
    if harmonized_metrics:
        reductions = [(r - h) / r * 100 if r > 0 else 0
                     for r, h in zip(raw_values, harm_values)]

        colors = [COLORS['harmonized'] if r > 0 else COLORS['raw'] for r in reductions]
        bars = axes[1].bar(x, reductions, width * 1.5, color=colors, edgecolor='black', linewidth=0.5)

        # add value labels
        for bar, val in zip(bars, reductions):
            height = bar.get_height()
            ypos = height + 1 if height >= 0 else height - 5
            va = 'bottom' if height >= 0 else 'top'
            axes[1].text(bar.get_x() + bar.get_width()/2, ypos,
                        f'{val:.1f}%', ha='center', va=va, fontsize=8)

        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(metric_labels)
        axes[1].set_ylabel('Reduction (%)')
        axes[1].set_title('Harmonization Improvement')

        # add annotation
        axes[1].text(0.5, 0.95, 'Positive = Better',
                    transform=axes[1].transAxes, ha='center', fontsize=8, color='gray')
    else:
        axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.savefig(output_path.with_suffix('.png'), format='png', dpi=300)
    plt.close()

    print(f'[fig] saved feature distribution metrics to {output_path}')


def setup_latex_style_publication():
    """configure matplotlib for publication-quality latex rendering."""
    sns.set_theme(style='whitegrid')
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 14,
        "font.size": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.titlesize": 18,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3
    })


# vibrant professional color palette for harmonization summary
COLORS_ENHANCED = {
    'raw': '#D97F0F',        # burnt orange
    'harmonized': '#2E5090', # deep blue
    'positive': '#3FA796',   # teal green
    'negative': '#A05050',   # dusty red
}


def plot_harmonization_effect_summary(
    domain_results: Dict,
    feature_results: Dict,
    output_path: Path
):
    """
    create publication-quality summary figure showing overall harmonization effect.

    combines domain classification and feature distribution results with
    professional styling matching reference figures.
    """
    setup_latex_style_publication()
    
    fig = plt.figure(figsize=(16, 9))

    # grid spec for complex layout
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)

    # (a) domain classification metrics
    ax1 = fig.add_subplot(gs[0, 0:2])

    metrics = ['accuracy', 'auc', 'f1']
    labels = ['Accuracy', 'AUC-ROC', 'F1-Score']

    raw_vals = [domain_results['raw'].get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    if 'harmonized' in domain_results:
        harm_vals = [domain_results['harmonized'].get(m, 0) for m in metrics]
        bars1 = ax1.bar(x - width/2, raw_vals, width, label='Raw (Before)',
                       color=COLORS_ENHANCED['raw'],
                       edgecolor='black', linewidth=0.7)
        bars2 = ax1.bar(x + width/2, harm_vals, width, label='Harmonized (After)',
                       color=COLORS_ENHANCED['harmonized'],
                       edgecolor='black', linewidth=0.7)
        
        # add value labels
        for bar, val in zip(bars1, raw_vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='normal')
        for bar, val in zip(bars2, harm_vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='normal')
    else:
        bars1 = ax1.bar(x, raw_vals, width, label='Raw',
                       color=COLORS_ENHANCED['raw'],
                       edgecolor='black', linewidth=0.7)
        for bar, val in zip(bars1, raw_vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11)

    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.set_ylim(0, 1.2)
    ax1.set_ylabel('Score', fontsize=13)
    ax1.set_title(r'(a) Domain Classification Metrics', fontsize=14, pad=10)
    ax1.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black', fontsize=10)
    ax1.set_axisbelow(True)

    # (b) feature distribution metrics
    ax2 = fig.add_subplot(gs[0, 2:4])

    feat_metrics = ['fid', 'kid_mean', 'mmd_rbf']
    feat_labels = ['FID', 'KID', 'MMD']

    raw_feat = [feature_results['raw'].get(m, 0) for m in feat_metrics]

    if 'harmonized' in feature_results:
        harm_feat = [feature_results['harmonized'].get(m, 0) for m in feat_metrics]
        bars1 = ax2.bar(x[:3] - width/2, raw_feat, width, label='Raw (Before)',
                       color=COLORS_ENHANCED['raw'],
                       edgecolor='black', linewidth=0.7)
        bars2 = ax2.bar(x[:3] + width/2, harm_feat, width, label='Harmonized (After)',
                       color=COLORS_ENHANCED['harmonized'],
                       edgecolor='black', linewidth=0.7)
        
        # add value labels with rotation for readability
        for bar, val in zip(bars1, raw_feat):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, rotation=25)
        for bar, val in zip(bars2, harm_feat):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, rotation=25)
    else:
        bars1 = ax2.bar(x[:3], raw_feat, width, label='Raw',
                       color=COLORS_ENHANCED['raw'],
                       edgecolor='black', linewidth=0.7)
        for bar, val in zip(bars1, raw_feat):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, rotation=25)

    ax2.set_xticks(x[:3])
    ax2.set_xticklabels(feat_labels, fontsize=12)
    ax2.set_ylabel('Distance', fontsize=13)
    ax2.set_title(r'(b) Feature Distribution Distances', fontsize=14, pad=10)
    ax2.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', fontsize=10)
    ax2.set_axisbelow(True)
    # set y-axis limit with extra space for value labels
    max_feat = max(raw_feat + (harm_feat if 'harmonized' in feature_results else []))
    ax2.set_ylim(0, max_feat * 1.25)

    # (c) improvement summary (bottom row)
    ax3 = fig.add_subplot(gs[1, :])

    if 'improvement' in domain_results or 'improvement' in feature_results:
        improvements = []
        improvement_labels = []

        if 'improvement' in domain_results:
            imp = domain_results['improvement']
            improvements.extend([
                imp.get('accuracy_reduction', 0) * 100,
                imp.get('auc_reduction', 0) * 100,
            ])
            improvement_labels.extend(['Domain Accuracy', 'Domain AUC'])

        if 'improvement' in feature_results:
            imp = feature_results['improvement']
            improvements.extend([
                imp.get('fid_reduction_percent', 0),
                imp.get('kid_reduction', 0) * 100 if imp.get('kid_reduction', 0) else 0,
            ])
            improvement_labels.extend(['FID Reduction', 'KID Reduction'])

        colors = [COLORS_ENHANCED['positive'] if v > 0 else COLORS_ENHANCED['negative'] for v in improvements]

        bars = ax3.barh(range(len(improvements)), improvements, color=colors,
                       edgecolor='black', linewidth=0.7)

        ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax3.set_yticks(range(len(improvements)))
        ax3.set_yticklabels(improvement_labels, fontsize=12)
        ax3.set_xlabel(r'Improvement (\%)', fontsize=13)
        ax3.set_title(r'(c) Harmonization Effect Summary', fontsize=14, pad=10)
        
        # add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, improvements)):
            width_bar = bar.get_width()
            xpos = width_bar + 1.5 if width_bar >= 0 else width_bar - 1.5
            ha = 'left' if width_bar >= 0 else 'right'
            ax3.text(xpos, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}\\%', ha=ha, va='center', fontsize=11, fontweight='normal')

        ax3.set_axisbelow(True)
        ax3.set_xlim(min(improvements) - 5, max(improvements) + 5)
    else:
        ax3.text(0.5, 0.5, 'harmonized results not available',
                transform=ax3.transAxes, ha='center', va='center', fontsize=12, color='gray')
        ax3.axis('off')

    # main figure title
    fig.suptitle(r'\textbf{Harmonization Effect Summary}',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

    print(f'[fig] saved publication-quality harmonization summary to {output_path}')


# vibrant training curves color palette
COLORS_TRAINING = {
    'train': '#C94D3F',      # warm red
    'val': '#5A8FBA',        # slate blue
    'auc': '#6FBC5F',        # forest green
    'grid': '#E8E8E8',       # light gray
}


def plot_training_curves(
    history: Dict,
    output_path: Path
):
    """
    plot domain classifier training curves with publication-quality styling.
    """
    setup_latex_style_publication()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(wspace=0.25, top=0.85)

    epochs = range(1, len(history['train_loss']) + 1)

    # (a) loss curves
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', 
           color=COLORS_TRAINING['train'], linewidth=2.2, marker='o', 
           markersize=3, alpha=0.8)
    ax.plot(epochs, history['val_loss'], label='Validation Loss', 
           color=COLORS_TRAINING['val'], linewidth=2.2, marker='s', 
           markersize=3, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title(r'(a) Training Loss', fontsize=14, pad=10)
    ax.legend(loc='upper right', frameon=True, fancybox=False, 
             edgecolor='black', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # (b) accuracy curves
    ax = axes[1]
    ax.plot(epochs, history['train_acc'], label='Train Accuracy', 
           color=COLORS_TRAINING['train'], linewidth=2.2, marker='o', 
           markersize=3, alpha=0.8)
    ax.plot(epochs, history['val_acc'], label='Validation Accuracy', 
           color=COLORS_TRAINING['val'], linewidth=2.2, marker='s', 
           markersize=3, alpha=0.8)
    
    if 'val_auc' in history:
        ax.plot(epochs, history['val_auc'], label='Validation AUC-ROC', 
               color=COLORS_TRAINING['auc'], linewidth=2.2, marker='^', 
               markersize=3, alpha=0.8, linestyle='-')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title(r'(b) Training Accuracy \& AUC-ROC', fontsize=14, pad=10)
    ax.legend(loc='lower right', frameon=True, fancybox=False, 
             edgecolor='black', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0.4, 1.05)
    
    # main figure title
    fig.suptitle(r'\textbf{Domain Classifier Training Curves}',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

    print(f'[fig] saved publication-quality training curves to {output_path}')


def create_latex_table(
    domain_results: Dict,
    feature_results: Dict,
    output_path: Path
):
    """
    create latex table summarizing all downstream evaluation results.
    """
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Downstream Task Evaluation Results}',
        r'\label{tab:downstream_evaluation}',
        r'\begin{tabular}{lccc}',
        r'\toprule',
        r'Metric & Raw & Harmonized & Improvement \\',
        r'\midrule',
        r'\multicolumn{4}{l}{\textit{Domain Classification}} \\',
    ]

    # domain classification metrics
    for metric, label in [('accuracy', 'Accuracy'), ('auc', 'AUC-ROC'), ('f1', 'F1-Score')]:
        raw_val = domain_results['raw'].get(metric, 0)
        if 'harmonized' in domain_results:
            harm_val = domain_results['harmonized'].get(metric, 0)
            improvement = (raw_val - harm_val) / raw_val * 100 if raw_val > 0 else 0
            sign = '+' if improvement > 0 else ''
            lines.append(f'{label} & {raw_val:.3f} & {harm_val:.3f} & {sign}{improvement:.1f}\\% \\\\')
        else:
            lines.append(f'{label} & {raw_val:.3f} & -- & -- \\\\')

    lines.extend([
        r'\midrule',
        r'\multicolumn{4}{l}{\textit{Feature Distribution}} \\',
    ])

    # feature distribution metrics
    for metric, label in [('fid', 'FID'), ('kid_mean', 'KID'), ('mmd_rbf', 'MMD'),
                          ('sliced_wasserstein', 'SWD')]:
        raw_val = feature_results['raw'].get(metric, 0)
        if 'harmonized' in feature_results:
            harm_val = feature_results['harmonized'].get(metric, 0)
            improvement = (raw_val - harm_val) / raw_val * 100 if raw_val > 0 else 0
            sign = '+' if improvement > 0 else ''
            lines.append(f'{label} & {raw_val:.3f} & {harm_val:.3f} & {sign}{improvement:.1f}\\% \\\\')
        else:
            lines.append(f'{label} & {raw_val:.3f} & -- & -- \\\\')

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f'[table] saved latex table to {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='generate downstream evaluation figures'
    )
    parser.add_argument('--domain-results', type=str, required=True,
                       help='path to domain classification results json')
    parser.add_argument('--feature-results', type=str, required=True,
                       help='path to feature distribution results json')
    parser.add_argument('--tsne-dir', type=str, default=None,
                       help='directory containing t-sne embeddings')
    parser.add_argument('--training-history', type=str, default=None,
                       help='path to training history json')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='output directory for figures')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load results
    with open(args.domain_results) as f:
        domain_results = json.load(f)

    with open(args.feature_results) as f:
        feature_results = json.load(f)

    print('[figs] generating downstream evaluation figures...')

    # domain classification comparison
    plot_domain_classification_comparison(
        domain_results['raw'],
        domain_results.get('harmonized'),
        output_dir / 'fig_domain_classification.pdf'
    )

    # feature distribution metrics
    plot_feature_distribution_metrics(
        feature_results['raw'],
        feature_results.get('harmonized'),
        output_dir / 'fig_feature_distribution.pdf'
    )

    # t-sne visualization
    if args.tsne_dir:
        tsne_dir = Path(args.tsne_dir)
        if (tsne_dir / 'tsne_raw.npy').exists():
            tsne_raw = np.load(tsne_dir / 'tsne_raw.npy')
            labels = np.load(tsne_dir / 'tsne_labels.npy')

            tsne_harm = None
            if (tsne_dir / 'tsne_harmonized.npy').exists():
                tsne_harm = np.load(tsne_dir / 'tsne_harmonized.npy')

            plot_tsne_visualization(
                tsne_raw, labels, tsne_harm,
                output_dir / 'fig_tsne_visualization.pdf'
            )

    # training curves
    if args.training_history:
        with open(args.training_history) as f:
            history = json.load(f)
        plot_training_curves(history, output_dir / 'fig_training_curves.pdf')

    # harmonization effect summary
    plot_harmonization_effect_summary(
        domain_results, feature_results,
        output_dir / 'fig_harmonization_summary.pdf'
    )

    # latex table
    create_latex_table(
        domain_results, feature_results,
        output_dir / 'table_downstream_results.tex'
    )

    print('=' * 60)
    print(f'[figs] all figures saved to {output_dir}')
    print('[figs] generated:')
    for f in sorted(output_dir.glob('fig_*.pdf')):
        print(f'  - {f.name}')
    print('[figs] generated:')
    for f in sorted(output_dir.glob('table_*.tex')):
        print(f'  - {f.name}')


if __name__ == '__main__':
    main()
