#!/usr/bin/env python3
"""
generate publication figure for computational efficiency comparison.

compares inference time and throughput across harmonization methods
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
COLORS = ['#7EB5D6', '#F2A7B3', '#B8A9C9', '#98D4BB']  # light blue, light pink, lavender, mint


def load_efficiency_results(results_path: Path) -> list:
    """load efficiency results from json file."""
    with open(results_path) as f:
        return json.load(f)


def create_efficiency_figure(metrics_list: list, output_path: Path):
    """
    create computational efficiency comparison figure.

    shows inference time comparison across methods.
    """
    setup_latex_style()

    # extract data
    methods = [m['method_name'].replace('_', ' ').title() for m in metrics_list]
    times_per_slice = [m.get('inference_time_per_slice_ms', 0) for m in metrics_list]
    times_per_volume = [m.get('inference_time_per_volume_ms', 0) / 1000 for m in metrics_list]  # convert to seconds
    throughput_slices = [m.get('throughput_slices_per_sec', 0) for m in metrics_list]

    # calculate throughput volumes per hour from throughput slices per sec
    # assuming 155 slices per volume
    throughput_volumes = [t * 3600 / 155 if t > 0 else 0 for t in throughput_slices]

    # create figure with 2 subplots (only meaningful data)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plt.subplots_adjust(wspace=0.3, top=0.85)

    # (a) inference time per slice (log scale due to wide range)
    ax = axes[0]
    colors = [COLORS[i % len(COLORS)] for i in range(len(methods))]
    bars = ax.bar(methods, times_per_slice, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Inference Time (ms/slice)')
    ax.set_title(r'\textbf{(a) Inference Time per Slice}')
    ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=30)

    # add value labels
    for bar, t in zip(bars, times_per_slice):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{t:.1f}', ha='center', va='bottom', fontsize=10)

    # (b) throughput (volumes per hour)
    ax = axes[1]
    bars = ax.bar(methods, throughput_volumes, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Throughput (volumes/hour)')
    ax.set_title(r'\textbf{(b) Processing Throughput}')
    ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=30)

    # add value labels
    for bar, t in zip(bars, throughput_volumes):
        if t > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f'{t:.0f}', ha='center', va='bottom', fontsize=10)

    # main figure title
    fig.suptitle(r'\textbf{Computational Efficiency Comparison of Harmonization Methods}',
                 fontsize=18, fontweight='bold', y=0.96)

    # save
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

    print(f'[fig] saved efficiency comparison to {output_path}')


def create_efficiency_table(metrics_list: list, output_path: Path):
    """create latex table for efficiency comparison."""

    latex = r"""\begin{table}[htbp]
\centering
\renewcommand{\arraystretch}{1.3}
\caption{Computational Efficiency Comparison of Harmonization Methods}
\vspace{0.5em}
\label{tab:efficiency}
\begin{tabular}{lccc}
\toprule
Method & Time/Slice (ms) & Time/Volume (s) & Throughput (vol/hr) \\
\midrule
"""

    for m in metrics_list:
        name = m['method_name'].replace('_', ' ').title()
        time_slice = m.get('inference_time_per_slice_ms', 0)
        time_vol = m.get('inference_time_per_volume_ms', 0) / 1000
        throughput_slices = m.get('throughput_slices_per_sec', 0)
        throughput_vol = throughput_slices * 3600 / 155 if throughput_slices > 0 else 0

        latex += f"{name} & {time_slice:.2f} & {time_vol:.2f} & {throughput_vol:.0f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f'[table] saved efficiency table to {output_path}')


def main():
    # paths
    project_root = Path(__file__).parent.parent.parent
    results_path = project_root / 'experiments' / 'computational_analysis' / 'efficiency_results.json'
    output_dir = project_root / 'figures' / 'computational'

    output_dir.mkdir(parents=True, exist_ok=True)

    print('[efficiency] loading results...')
    metrics_list = load_efficiency_results(results_path)

    print('[efficiency] generating figure...')
    create_efficiency_figure(metrics_list, output_dir / 'fig_efficiency_comparison.pdf')

    print('[efficiency] generating table...')
    create_efficiency_table(metrics_list, output_dir / 'table_efficiency.tex')

    print('=' * 60)
    print('[efficiency] figure generation complete')


if __name__ == '__main__':
    main()
