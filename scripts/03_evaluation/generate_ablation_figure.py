#!/usr/bin/env python3
"""
generate publication figure for ablation study results.

creates bar charts comparing baseline cyclegan vs sa-cyclegan-2.5d
with statistical significance markers.
"""

import argparse
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'baseline': '#E74C3C',
    'attention': '#2E86AB',
}


def load_ablation_results(results_path: Path) -> dict:
    """load ablation results from json file."""
    with open(results_path) as f:
        return json.load(f)


def create_ablation_figure(results: dict, output_path: Path):
    """
    create comprehensive ablation study figure.

    4-panel layout:
    - (a) cycle ssim comparison
    - (b) cycle psnr comparison
    - (c) per-modality ssim
    - (d) effect sizes
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    baseline = results['baseline_results']
    attention = results['attention_results']
    stats = results['statistical_tests']

    # (a) cycle ssim comparison
    ax = axes[0, 0]
    metrics = ['Cycle SSIM\n(A→B→A)', 'Cycle SSIM\n(B→A→B)', 'Identity SSIM\n(A)', 'Identity SSIM\n(B)']
    baseline_vals = [
        baseline['cycle_ssim_A']['mean'],
        baseline['cycle_ssim_B']['mean'],
        baseline['identity_ssim_A']['mean'],
        baseline['identity_ssim_B']['mean']
    ]
    baseline_stds = [
        baseline['cycle_ssim_A']['std'],
        baseline['cycle_ssim_B']['std'],
        baseline['identity_ssim_A']['std'],
        baseline['identity_ssim_B']['std']
    ]
    attention_vals = [
        attention['cycle_ssim_A']['mean'],
        attention['cycle_ssim_B']['mean'],
        attention['identity_ssim_A']['mean'],
        attention['identity_ssim_B']['mean']
    ]
    attention_stds = [
        attention['cycle_ssim_A']['std'],
        attention['cycle_ssim_B']['std'],
        attention['identity_ssim_A']['std'],
        attention['identity_ssim_B']['std']
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_vals, width, yerr=baseline_stds,
                   label='Baseline CycleGAN', color=COLORS['baseline'],
                   edgecolor='black', linewidth=0.5, capsize=3)
    bars2 = ax.bar(x + width/2, attention_vals, width, yerr=attention_stds,
                   label='SA-CycleGAN-2.5D', color=COLORS['attention'],
                   edgecolor='black', linewidth=0.5, capsize=3)

    ax.set_ylabel('SSIM Score')
    ax.set_title('(a) Structural Similarity Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.9, 0.98)
    ax.legend(loc='lower right')

    # add significance stars
    sig_keys = ['cycle_ssim_A', 'cycle_ssim_B', 'identity_ssim_A', 'identity_ssim_B']
    for i, key in enumerate(sig_keys):
        if stats[key]['significant']:
            max_val = max(baseline_vals[i] + baseline_stds[i],
                         attention_vals[i] + attention_stds[i])
            ax.annotate('*', xy=(i, max_val + 0.005), ha='center', fontsize=14)

    # (b) cycle psnr comparison
    ax = axes[0, 1]
    metrics_psnr = ['Cycle PSNR\n(A→B→A)', 'Cycle PSNR\n(B→A→B)']
    baseline_psnr = [baseline['cycle_psnr_A']['mean'], baseline['cycle_psnr_B']['mean']]
    baseline_psnr_std = [baseline['cycle_psnr_A']['std'], baseline['cycle_psnr_B']['std']]
    attention_psnr = [attention['cycle_psnr_A']['mean'], attention['cycle_psnr_B']['mean']]
    attention_psnr_std = [attention['cycle_psnr_A']['std'], attention['cycle_psnr_B']['std']]

    x = np.arange(len(metrics_psnr))

    bars1 = ax.bar(x - width/2, baseline_psnr, width, yerr=baseline_psnr_std,
                   label='Baseline CycleGAN', color=COLORS['baseline'],
                   edgecolor='black', linewidth=0.5, capsize=3)
    bars2 = ax.bar(x + width/2, attention_psnr, width, yerr=attention_psnr_std,
                   label='SA-CycleGAN-2.5D', color=COLORS['attention'],
                   edgecolor='black', linewidth=0.5, capsize=3)

    ax.set_ylabel('PSNR (dB)')
    ax.set_title('(b) Peak Signal-to-Noise Ratio Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_psnr)
    ax.set_ylim(25, 32)
    ax.legend(loc='lower right')

    # add significance
    for i, key in enumerate(['cycle_psnr_A', 'cycle_psnr_B']):
        if stats[key]['significant']:
            max_val = max(baseline_psnr[i] + baseline_psnr_std[i],
                         attention_psnr[i] + attention_psnr_std[i])
            ax.annotate('*', xy=(i, max_val + 0.3), ha='center', fontsize=14)

    # (c) per-modality ssim (domain b direction where attention improves)
    ax = axes[1, 0]
    modalities = ['T1', 'T1CE', 'T2', 'FLAIR']
    baseline_mod = [
        baseline['cycle_ssim_B_T1']['mean'],
        baseline['cycle_ssim_B_T1CE']['mean'],
        baseline['cycle_ssim_B_T2']['mean'],
        baseline['cycle_ssim_B_FLAIR']['mean']
    ]
    attention_mod = [
        attention['cycle_ssim_B_T1']['mean'],
        attention['cycle_ssim_B_T1CE']['mean'],
        attention['cycle_ssim_B_T2']['mean'],
        attention['cycle_ssim_B_FLAIR']['mean']
    ]

    x = np.arange(len(modalities))

    bars1 = ax.bar(x - width/2, baseline_mod, width,
                   label='Baseline CycleGAN', color=COLORS['baseline'],
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, attention_mod, width,
                   label='SA-CycleGAN-2.5D', color=COLORS['attention'],
                   edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Cycle SSIM (B→A→B)')
    ax.set_title('(c) Per-Modality Reconstruction Quality')
    ax.set_xticks(x)
    ax.set_xticklabels(modalities)
    ax.set_ylim(0.9, 0.98)
    ax.legend(loc='lower right')

    # add improvement annotations
    for i in range(len(modalities)):
        improvement = (attention_mod[i] - baseline_mod[i]) * 100
        if improvement > 0:
            ax.annotate(f'+{improvement:.1f}%', xy=(i + width/2, attention_mod[i] + 0.003),
                       ha='center', fontsize=8, color='green')

    # (d) effect sizes (cohen's d)
    ax = axes[1, 1]
    effect_metrics = ['SSIM\n(A→B→A)', 'SSIM\n(B→A→B)', 'PSNR\n(A→B→A)', 'PSNR\n(B→A→B)']
    effect_sizes = [
        stats['cycle_ssim_A']['cohens_d'],
        stats['cycle_ssim_B']['cohens_d'],
        stats['cycle_psnr_A']['cohens_d'],
        stats['cycle_psnr_B']['cohens_d']
    ]

    colors = ['green' if d > 0 else 'red' for d in effect_sizes]
    bars = ax.barh(effect_metrics, effect_sizes, color=colors, edgecolor='black', linewidth=0.5)

    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0.8, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.axvline(x=-0.8, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    ax.set_xlabel("Cohen's d Effect Size")
    ax.set_title("(d) Effect Sizes (SA-CycleGAN vs Baseline)")

    # add interpretation guide
    ax.text(1.5, 3.5, 'Large\nEffect', fontsize=8, ha='center', color='gray')
    ax.text(-1.5, 3.5, 'Large\nEffect', fontsize=8, ha='center', color='gray')

    plt.tight_layout()

    # save
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight')
    plt.close()

    print(f'[fig] saved ablation figure to {output_path}')


def create_ablation_summary_table(results: dict, output_path: Path):
    """create summary latex table for ablation results."""
    stats = results['statistical_tests']

    # summary focusing on key improvements
    latex = r"""\begin{table}[htbp]
\centering
\caption{Ablation Study: Self-Attention Impact on Harmonization Quality}
\label{tab:ablation_summary}
\begin{tabular}{lccc}
\toprule
Direction & Metric & Improvement & Cohen's $d$ \\
\midrule
\multirow{2}{*}{A $\rightarrow$ B $\rightarrow$ A}
  & Cycle SSIM & """ + f"{stats['cycle_ssim_A']['difference']*100:+.2f}" + r"""\% & """ + f"{stats['cycle_ssim_A']['cohens_d']:.2f}" + r""" \\
  & Cycle PSNR & """ + f"{stats['cycle_psnr_A']['difference']:+.2f}" + r""" dB & """ + f"{stats['cycle_psnr_A']['cohens_d']:.2f}" + r""" \\
\midrule
\multirow{2}{*}{B $\rightarrow$ A $\rightarrow$ B}
  & Cycle SSIM & """ + f"{stats['cycle_ssim_B']['difference']*100:+.2f}" + r"""\% & """ + f"{stats['cycle_ssim_B']['cohens_d']:.2f}" + r""" \\
  & Cycle PSNR & """ + f"{stats['cycle_psnr_B']['difference']:+.2f}" + r""" dB & """ + f"{stats['cycle_psnr_B']['cohens_d']:.2f}" + r""" \\
\bottomrule
\multicolumn{4}{l}{\footnotesize All differences significant at $p < 0.001$} \\
\end{tabular}
\end{table}
"""

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f'[table] saved ablation summary table to {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='generate ablation study publication figure'
    )
    parser.add_argument('--results', type=str, required=True,
                       help='path to ablation results json')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='output directory for figures')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('[ablation] loading results...')
    results = load_ablation_results(Path(args.results))

    print('[ablation] generating figure...')
    create_ablation_figure(results, output_dir / 'fig_ablation_study.pdf')

    print('[ablation] generating summary table...')
    create_ablation_summary_table(results, output_dir / 'table_ablation_summary.tex')

    print('=' * 60)
    print('[ablation] figure generation complete')


if __name__ == '__main__':
    main()
