#!/usr/bin/env python3
"""
Generate Quantitative Results Figures for Publication

Creates publication-grade LaTeX-rendered figures showing:
- Metric distribution box plots
- Quantitative comparison tables
- Cycle consistency comparisons
- Statistical visualizations

Author: NeuroScope Research Team
Date: January 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from latex_figure_config import (
    FIGURE_SIZES, COLORS, save_figure
)

# Load evaluation results
PROJECT_ROOT = Path(__file__).parent.parent.parent
EVAL_PATH = PROJECT_ROOT / 'results/evaluation/evaluation_results.json'
CYCLE_PATH = PROJECT_ROOT / 'results/evaluation/cycle_consistency_results.json'
OUTPUT_DIR = PROJECT_ROOT / 'figures/main'
TABLES_DIR = PROJECT_ROOT / 'figures/tables'


def load_evaluation_results():
    """Load evaluation results from JSON."""
    with open(EVAL_PATH) as f:
        eval_results = json.load(f)
    with open(CYCLE_PATH) as f:
        cycle_results = json.load(f)
    return eval_results, cycle_results


def generate_box_plots_figure(eval_results):
    """
    Figure: Metric Distribution Box Plots

    Shows distribution of evaluation metrics across test set.
    """
    print("Generating: Metric Distribution Box Plots...")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(r'\textbf{SA-CycleGAN-2.5D: Test Set Metric Distributions}', y=1.02)

    metrics = ['ssim', 'psnr', 'mae', 'lpips', 'mse', 'fid']
    titles = ['SSIM', 'PSNR (dB)', 'MAE', 'LPIPS', 'MSE', 'FID']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]

        if metric == 'fid':
            # FID is a single value, show as bar
            fid_a2b = eval_results['a2b']['fid']['value']
            fid_b2a = eval_results['b2a']['fid']['value']
            ax.bar([0, 1], [fid_a2b, fid_b2a],
                   color=[COLORS['primary'], COLORS['secondary']],
                   alpha=0.7, width=0.6)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([r'$A\!\rightarrow\!B$', r'$B\!\rightarrow\!A$'])
            ax.set_ylabel(title)
        else:
            # Other metrics have distributions
            data_a2b = eval_results['a2b'][metric].get('values', [])
            data_b2a = eval_results['b2a'][metric].get('values', [])

            if len(data_a2b) == 0:
                # Use mean ± std if values not available
                mean_a2b = eval_results['a2b'][metric]['mean']
                std_a2b = eval_results['a2b'][metric]['std']
                mean_b2a = eval_results['b2a'][metric]['mean']
                std_b2a = eval_results['b2a'][metric]['std']

                ax.bar([0, 1], [mean_a2b, mean_b2a],
                       yerr=[std_a2b, std_b2a],
                       color=[COLORS['primary'], COLORS['secondary']],
                       alpha=0.7, width=0.6, capsize=5)
                ax.set_xticks([0, 1])
                ax.set_xticklabels([r'$A\!\rightarrow\!B$', r'$B\!\rightarrow\!A$'])
            else:
                # Box plot with distributions
                bp = ax.boxplot([data_a2b, data_b2a],
                               labels=[r'$A\!\rightarrow\!B$', r'$B\!\rightarrow\!A$'],
                               patch_artist=True,
                               showmeans=True,
                               meanline=True)
                for patch, color in zip(bp['boxes'], [COLORS['primary'], COLORS['secondary']]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)

            ax.set_ylabel(title)

        ax.set_title(f'({chr(97+idx)}) {title}')
        ax.grid(True, alpha=0.3, axis='y')

    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure(fig, 'fig06_metric_distributions', OUTPUT_DIR)
    plt.close()


def generate_cycle_consistency_figure(cycle_results):
    """
    Figure: Cycle Consistency Comparison

    Compares cycle A and cycle B reconstruction quality.
    """
    print("Generating: Cycle Consistency Comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle(r'\textbf{Cycle Consistency Reconstruction Quality}', y=1.02)

    metrics = ['ssim', 'psnr', 'mae']
    titles = ['SSIM', 'PSNR (dB)', 'MAE']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]

        cycle_a_mean = cycle_results['cycle_a'][metric]['mean']
        cycle_a_std = cycle_results['cycle_a'][metric]['std']
        cycle_b_mean = cycle_results['cycle_b'][metric]['mean']
        cycle_b_std = cycle_results['cycle_b'][metric]['std']

        x = np.arange(2)
        width = 0.6

        bars = ax.bar(x, [cycle_a_mean, cycle_b_mean],
                      yerr=[cycle_a_std, cycle_b_std],
                      color=[COLORS['primary'], COLORS['success']],
                      alpha=0.7, width=width, capsize=5)

        ax.set_xticks(x)
        ax.set_xticklabels([r'Cycle A', r'Cycle B'])
        ax.set_ylabel(title)
        ax.set_title(f'({chr(97+idx)}) {title}')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure(fig, 'fig07_cycle_consistency', OUTPUT_DIR)
    plt.close()


def generate_quantitative_tables(eval_results, cycle_results):
    """
    Generate LaTeX tables for quantitative results.
    """
    print("Generating: Quantitative Results Tables...")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Table 1: Translation Quality Metrics
    table1 = []
    table1.append(r'\begin{table}[htbp]')
    table1.append(r'\centering')
    table1.append(r'\caption{Quantitative Evaluation of SA-CycleGAN-2.5D on Test Set}')
    table1.append(r'\label{tab:quantitative_results}')
    table1.append(r'\begin{tabular}{lcc}')
    table1.append(r'\hline')
    table1.append(r'Metric & $A \rightarrow B$ (BraTS $\rightarrow$ UPenn) & $B \rightarrow A$ (UPenn $\rightarrow$ BraTS) \\')
    table1.append(r'\hline')

    # Add metrics
    metrics = [
        ('SSIM', 'ssim', ''),
        ('PSNR (dB)', 'psnr', ''),
        ('MAE', 'mae', ''),
        ('MSE', 'mse', ''),
        ('LPIPS', 'lpips', ''),
        ('FID', 'fid', ' (single value)'),
    ]

    for name, key, note in metrics:
        if key == 'fid':
            val_a2b = eval_results['a2b'][key]['value']
            val_b2a = eval_results['b2a'][key]['value']
            table1.append(f'{name}{note} & {val_a2b:.2f} & {val_b2a:.2f} \\\\')
        else:
            mean_a2b = eval_results['a2b'][key]['mean']
            std_a2b = eval_results['a2b'][key]['std']
            mean_b2a = eval_results['b2a'][key]['mean']
            std_b2a = eval_results['b2a'][key]['std']
            table1.append(f'{name} & {mean_a2b:.4f} $\\pm$ {std_a2b:.4f} & {mean_b2a:.4f} $\\pm$ {std_b2a:.4f} \\\\')

    table1.append(r'\hline')
    table1.append(r'\end{tabular}')
    table1.append(r'\end{table}')

    # Save table 1
    with open(TABLES_DIR / 'table1_quantitative_results.tex', 'w') as f:
        f.write('\n'.join(table1))

    # Table 2: Cycle Consistency Metrics
    table2 = []
    table2.append(r'\begin{table}[htbp]')
    table2.append(r'\centering')
    table2.append(r'\caption{Cycle Consistency Reconstruction Quality}')
    table2.append(r'\label{tab:cycle_consistency}')
    table2.append(r'\begin{tabular}{lcc}')
    table2.append(r'\hline')
    table2.append(r'Metric & Cycle A ($A \rightarrow B \rightarrow A$) & Cycle B ($B \rightarrow A \rightarrow B$) \\')
    table2.append(r'\hline')

    for name, key in [('SSIM', 'ssim'), ('PSNR (dB)', 'psnr'), ('MAE', 'mae')]:
        mean_a = cycle_results['cycle_a'][key]['mean']
        std_a = cycle_results['cycle_a'][key]['std']
        mean_b = cycle_results['cycle_b'][key]['mean']
        std_b = cycle_results['cycle_b'][key]['std']
        table2.append(f'{name} & {mean_a:.4f} $\\pm$ {std_a:.4f} & {mean_b:.4f} $\\pm$ {std_b:.4f} \\\\')

    table2.append(r'\hline')
    table2.append(r'\end{tabular}')
    table2.append(r'\end{table}')

    # Save table 2
    with open(TABLES_DIR / 'table2_cycle_consistency.tex', 'w') as f:
        f.write('\n'.join(table2))

    print(f"  Saved: {TABLES_DIR}/table1_quantitative_results.tex")
    print(f"  Saved: {TABLES_DIR}/table2_cycle_consistency.tex")


def main():
    """Generate all quantitative figures."""
    print("="*60)
    print("Generating Quantitative Figures and Tables")
    print("="*60)

    # Load results
    eval_results, cycle_results = load_evaluation_results()
    print(f"Loaded evaluation results: {eval_results['test_samples']} test samples")
    print(f"Loaded cycle consistency results: {cycle_results['test_samples']} test samples")

    # Generate figures
    generate_box_plots_figure(eval_results)
    generate_cycle_consistency_figure(cycle_results)
    generate_quantitative_tables(eval_results, cycle_results)

    print("\n" + "="*60)
    print("Quantitative figures generation complete!")
    print(f"Figures saved to: {OUTPUT_DIR}/")
    print(f"Tables saved to: {TABLES_DIR}/")
    print("="*60)


if __name__ == '__main__':
    main()
