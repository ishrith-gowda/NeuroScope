"""
generate comprehensive statistical analysis figures for publication

creates comparison charts, approximated distributions, and statistical
summaries from aggregate evaluation results.

note: per-sample correlation analysis requires running inference
to collect individual metric arrays - see tools/inference/ scripts.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# import latex configuration
from latex_figure_config import COLORS, save_figure, get_figure_size

# setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EVAL_PATH = PROJECT_ROOT / 'results/evaluation/evaluation_results.json'
OUTPUT_DIR = PROJECT_ROOT / 'figures/main'


def load_evaluation_data():
    """load and structure evaluation results for analysis"""
    with open(EVAL_PATH, 'r') as f:
        data = json.load(f)

    # extract aggregate statistics for both directions
    metrics_a2b = {}
    metrics_b2a = {}

    metric_names = ['ssim', 'psnr', 'mae', 'lpips', 'mse']

    for metric in metric_names:
        metrics_a2b[metric] = data['a2b'][metric]
        metrics_b2a[metric] = data['b2a'][metric]

    # fid scores
    fid_scores = {
        'a2b': data['a2b']['fid']['value'],
        'b2a': data['b2a']['fid']['value']
    }

    n_samples = data['test_samples']

    return metrics_a2b, metrics_b2a, fid_scores, n_samples


def generate_comprehensive_comparison(metrics_a2b, metrics_b2a, fid_scores):
    """
    figure 15: comprehensive metric comparison with error bars

    shows mean ± std for all metrics comparing both directions
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    metric_info = [
        ('ssim', 'SSIM', True),
        ('psnr', 'PSNR (dB)', True),
        ('mae', 'MAE', False),
        ('lpips', 'LPIPS', False),
        ('mse', 'MSE', False)
    ]

    for idx, (key, name, higher_better) in enumerate(metric_info):
        ax = axes[idx]

        # extract mean and std
        mean_a2b = metrics_a2b[key]['mean']
        std_a2b = metrics_a2b[key]['std']
        mean_b2a = metrics_b2a[key]['mean']
        std_b2a = metrics_b2a[key]['std']

        x = np.arange(2)
        means = [mean_a2b, mean_b2a]
        stds = [std_a2b, std_b2a]
        colors = [COLORS['primary'], COLORS['secondary']]

        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                     color=colors, error_kw={'linewidth': 2})

        ax.set_xticks(x)
        ax.set_xticklabels([r'$A \rightarrow B$', r'$B \rightarrow A$'])
        ax.set_ylabel(name)
        ax.set_title(f'({chr(97+idx)}) {name}')
        ax.grid(True, alpha=0.3, axis='y')

        # add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std,
                   f'{mean:.4f}' if key in ['ssim', 'mae', 'lpips', 'mse'] else f'{mean:.2f}',
                   ha='center', va='bottom', fontsize=7)

        # mark better direction with arrow
        if higher_better:
            better_idx = 0 if mean_a2b > mean_b2a else 1
        else:
            better_idx = 0 if mean_a2b < mean_b2a else 1

        # add subtle indicator
        ax.scatter(better_idx, means[better_idx] + stds[better_idx] * 1.2,
                  marker='v', s=50, color=colors[better_idx], alpha=0.8)

    # handle fid separately in last subplot
    ax = axes[5]
    x = np.arange(2)
    fid_vals = [fid_scores['a2b'], fid_scores['b2a']]
    colors = [COLORS['primary'], COLORS['secondary']]

    bars = ax.bar(x, fid_vals, alpha=0.8, color=colors)

    ax.set_xticks(x)
    ax.set_xticklabels([r'$A \rightarrow B$', r'$B \rightarrow A$'])
    ax.set_ylabel('FID')
    ax.set_title(r'(f) FID (lower better)')
    ax.grid(True, alpha=0.3, axis='y')

    # add value labels
    for bar, val in zip(bars, fid_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}',
               ha='center', va='bottom', fontsize=7)

    # mark better
    better_idx = 0 if fid_scores['a2b'] < fid_scores['b2a'] else 1
    ax.scatter(better_idx, fid_vals[better_idx] * 1.05,
              marker='v', s=50, color=colors[better_idx], alpha=0.8)

    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    plt.tight_layout()
    save_figure(fig, 'fig15_comprehensive_comparison', output_dir=OUTPUT_DIR)
    plt.close()

    print("generated figure 15: comprehensive metric comparison")


def generate_approximated_distributions(metrics_a2b, metrics_b2a):
    """
    figure 16: approximated box plots using quartile data

    uses min, q25, median, q75, max to approximate distributions
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    metric_info = [
        ('ssim', 'SSIM'),
        ('psnr', 'PSNR (dB)'),
        ('mae', 'MAE'),
        ('lpips', 'LPIPS'),
        ('mse', 'MSE')
    ]

    for idx, (key, name) in enumerate(metric_info):
        ax = axes[idx]

        # create box plot data structure
        # [min, q25, median, q75, max]
        box_data_a2b = [
            [metrics_a2b[key]['min']],
            [metrics_a2b[key]['q25']],
            [metrics_a2b[key]['median']],
            [metrics_a2b[key]['q75']],
            [metrics_a2b[key]['max']]
        ]

        box_data_b2a = [
            [metrics_b2a[key]['min']],
            [metrics_b2a[key]['q25']],
            [metrics_b2a[key]['median']],
            [metrics_b2a[key]['q75']],
            [metrics_b2a[key]['max']]
        ]

        # manual box plot
        positions = [1, 2]

        # a2b box
        q25_a2b = metrics_a2b[key]['q25']
        q75_a2b = metrics_a2b[key]['q75']
        median_a2b = metrics_a2b[key]['median']
        min_a2b = metrics_a2b[key]['min']
        max_a2b = metrics_a2b[key]['max']

        # b2a box
        q25_b2a = metrics_b2a[key]['q25']
        q75_b2a = metrics_b2a[key]['q75']
        median_b2a = metrics_b2a[key]['median']
        min_b2a = metrics_b2a[key]['min']
        max_b2a = metrics_b2a[key]['max']

        width = 0.4

        # draw boxes
        box_a2b = plt.Rectangle((positions[0] - width/2, q25_a2b),
                                width, q75_a2b - q25_a2b,
                                facecolor=COLORS['primary'], alpha=0.6,
                                edgecolor='black', linewidth=1.5)
        box_b2a = plt.Rectangle((positions[1] - width/2, q25_b2a),
                                width, q75_b2a - q25_b2a,
                                facecolor=COLORS['secondary'], alpha=0.6,
                                edgecolor='black', linewidth=1.5)

        ax.add_patch(box_a2b)
        ax.add_patch(box_b2a)

        # draw medians
        ax.plot([positions[0] - width/2, positions[0] + width/2],
               [median_a2b, median_a2b], 'k-', linewidth=2)
        ax.plot([positions[1] - width/2, positions[1] + width/2],
               [median_b2a, median_b2a], 'k-', linewidth=2)

        # draw whiskers
        ax.plot([positions[0], positions[0]], [q75_a2b, max_a2b], 'k-', linewidth=1.5)
        ax.plot([positions[0], positions[0]], [min_a2b, q25_a2b], 'k-', linewidth=1.5)
        ax.plot([positions[1], positions[1]], [q75_b2a, max_b2a], 'k-', linewidth=1.5)
        ax.plot([positions[1], positions[1]], [min_b2a, q25_b2a], 'k-', linewidth=1.5)

        # caps
        cap_width = width * 0.5
        ax.plot([positions[0] - cap_width/2, positions[0] + cap_width/2],
               [max_a2b, max_a2b], 'k-', linewidth=1.5)
        ax.plot([positions[0] - cap_width/2, positions[0] + cap_width/2],
               [min_a2b, min_a2b], 'k-', linewidth=1.5)
        ax.plot([positions[1] - cap_width/2, positions[1] + cap_width/2],
               [max_b2a, max_b2a], 'k-', linewidth=1.5)
        ax.plot([positions[1] - cap_width/2, positions[1] + cap_width/2],
               [min_b2a, min_b2a], 'k-', linewidth=1.5)

        ax.set_xticks(positions)
        ax.set_xticklabels([r'$A \rightarrow B$', r'$B \rightarrow A$'])
        ax.set_ylabel(name)
        ax.set_title(f'({chr(97+idx)}) {name}')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(0.5, 2.5)

        # add mean markers
        mean_a2b = metrics_a2b[key]['mean']
        mean_b2a = metrics_b2a[key]['mean']
        ax.plot(positions[0], mean_a2b, 'D', color=COLORS['danger'],
               markersize=6, label='Mean')
        ax.plot(positions[1], mean_b2a, 'D', color=COLORS['danger'],
               markersize=6)

        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')

    # remove extra subplot
    axes[5].remove()

    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    plt.tight_layout()
    save_figure(fig, 'fig16_approximated_distributions', output_dir=OUTPUT_DIR)
    plt.close()

    print("generated figure 16: approximated box plot distributions")


def generate_performance_radar(metrics_a2b, metrics_b2a):
    """
    figure 17: radar chart comparing normalized performance

    normalizes all metrics to 0-1 scale for visual comparison
    """
    fig = plt.figure(figsize=(10, 5))

    metric_names = ['SSIM', 'PSNR', 'MAE\n(inv)', 'LPIPS\n(inv)', 'MSE\n(inv)']
    metric_keys = ['ssim', 'psnr', 'mae', 'lpips', 'mse']

    # normalize metrics
    normalized_a2b = []
    normalized_b2a = []

    for key in metric_keys:
        mean_a2b = metrics_a2b[key]['mean']
        mean_b2a = metrics_b2a[key]['mean']

        if key in ['ssim']:
            # already 0-1, higher better
            normalized_a2b.append(mean_a2b)
            normalized_b2a.append(mean_b2a)
        elif key == 'psnr':
            # normalize using typical range [15, 25]
            min_val, max_val = 15, 25
            norm_a2b = (mean_a2b - min_val) / (max_val - min_val)
            norm_b2a = (mean_b2a - min_val) / (max_val - min_val)
            normalized_a2b.append(np.clip(norm_a2b, 0, 1))
            normalized_b2a.append(np.clip(norm_b2a, 0, 1))
        else:
            # invert: lower is better
            # use percentile approach
            combined = [mean_a2b, mean_b2a]
            max_val = max(combined)
            min_val = 0
            norm_a2b = 1 - (mean_a2b - min_val) / (max_val - min_val)
            norm_b2a = 1 - (mean_b2a - min_val) / (max_val - min_val)
            normalized_a2b.append(norm_a2b)
            normalized_b2a.append(norm_b2a)

    # setup radar chart
    angles = np.linspace(0, 2*np.pi, len(metric_names), endpoint=False).tolist()
    normalized_a2b += normalized_a2b[:1]  # complete the circle
    normalized_b2a += normalized_b2a[:1]
    angles += angles[:1]

    ax = fig.add_subplot(111, projection='polar')

    # plot
    ax.plot(angles, normalized_a2b, 'o-', linewidth=2,
           color=COLORS['primary'], label=r'$A \rightarrow B$')
    ax.fill(angles, normalized_a2b, alpha=0.25, color=COLORS['primary'])

    ax.plot(angles, normalized_b2a, 'o-', linewidth=2,
           color=COLORS['secondary'], label=r'$B \rightarrow A$')
    ax.fill(angles, normalized_b2a, alpha=0.25, color=COLORS['secondary'])

    # styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    save_figure(fig, 'fig17_performance_radar', output_dir=OUTPUT_DIR)
    plt.close()

    print("generated figure 17: performance radar chart")


def generate_statistical_summary_table():
    """
    figure 18: comprehensive statistical summary table

    latex table with all metrics and statistics
    """
    with open(EVAL_PATH, 'r') as f:
        data = json.load(f)

    output_path = OUTPUT_DIR.parent / 'tables' / 'table3_statistical_summary.tex'
    output_path.parent.mkdir(exist_ok=True, parents=True)

    metric_names = ['SSIM', 'PSNR', 'MAE', 'LPIPS', 'MSE', 'FID']
    metric_keys = ['ssim', 'psnr', 'mae', 'lpips', 'mse', 'fid']

    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Comprehensive Statistical Summary of Evaluation Metrics}')
    lines.append(r'\label{tab:statistical_summary}')
    lines.append(r'\begin{tabular}{lccccccc}')
    lines.append(r'\hline')
    lines.append(r'Metric & Direction & Mean & Std & Median & Q25 & Q75 & Range \\')
    lines.append(r'\hline')

    for metric_name, metric_key in zip(metric_names, metric_keys):
        # a2b row
        if metric_key == 'fid':
            val = data['a2b'][metric_key]['value']
            lines.append(f'{metric_name} & $A \\rightarrow B$ & {val:.4f} & - & - & - & - & - \\\\')
        else:
            stats_a2b = data['a2b'][metric_key]
            mean_val = stats_a2b['mean']
            std_val = stats_a2b['std']
            median_val = stats_a2b['median']
            q25_val = stats_a2b['q25']
            q75_val = stats_a2b['q75']
            range_val = f"[{stats_a2b['min']:.4f}, {stats_a2b['max']:.4f}]"

            if metric_key == 'psnr':
                lines.append(f'{metric_name} & $A \\rightarrow B$ & {mean_val:.2f} & {std_val:.2f} & {median_val:.2f} & {q25_val:.2f} & {q75_val:.2f} & {range_val} \\\\')
            else:
                lines.append(f'{metric_name} & $A \\rightarrow B$ & {mean_val:.4f} & {std_val:.4f} & {median_val:.4f} & {q25_val:.4f} & {q75_val:.4f} & {range_val} \\\\')

        # b2a row
        if metric_key == 'fid':
            val = data['b2a'][metric_key]['value']
            lines.append(f' & $B \\rightarrow A$ & {val:.4f} & - & - & - & - & - \\\\')
        else:
            stats_b2a = data['b2a'][metric_key]
            mean_val = stats_b2a['mean']
            std_val = stats_b2a['std']
            median_val = stats_b2a['median']
            q25_val = stats_b2a['q25']
            q75_val = stats_b2a['q75']
            range_val = f"[{stats_b2a['min']:.4f}, {stats_b2a['max']:.4f}]"

            if metric_key == 'psnr':
                lines.append(f' & $B \\rightarrow A$ & {mean_val:.2f} & {std_val:.2f} & {median_val:.2f} & {q25_val:.2f} & {q75_val:.2f} & {range_val} \\\\')
            else:
                lines.append(f' & $B \\rightarrow A$ & {mean_val:.4f} & {std_val:.4f} & {median_val:.4f} & {q25_val:.4f} & {q75_val:.4f} & {range_val} \\\\')

        if metric_key != metric_keys[-1]:
            lines.append(r'\hline')

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"generated table 3: statistical summary ({output_path})")


def generate_effect_size_analysis(metrics_a2b, metrics_b2a, n_samples):
    """
    figure 19: effect size analysis (cohen's d)

    shows standardized difference between directions
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    metric_names = ['SSIM', 'PSNR', 'MAE', 'LPIPS', 'MSE']
    metric_keys = ['ssim', 'psnr', 'mae', 'lpips', 'mse']

    cohens_d = []

    for key in metric_keys:
        mean_a2b = metrics_a2b[key]['mean']
        mean_b2a = metrics_b2a[key]['mean']
        std_a2b = metrics_a2b[key]['std']
        std_b2a = metrics_b2a[key]['std']

        # pooled standard deviation
        pooled_std = np.sqrt((std_a2b**2 + std_b2a**2) / 2)

        # cohen's d
        d = (mean_a2b - mean_b2a) / pooled_std
        cohens_d.append(d)

    # plot
    y = np.arange(len(metric_names))
    colors_list = [COLORS['primary'] if d > 0 else COLORS['secondary'] for d in cohens_d]

    bars = ax.barh(y, cohens_d, color=colors_list, alpha=0.8)

    # add vertical line at 0
    ax.axvline(0, color='black', linewidth=1.5, linestyle='-')

    # add effect size thresholds
    ax.axvline(0.2, color='gray', linewidth=1, linestyle='--', alpha=0.5, label='Small')
    ax.axvline(0.5, color='gray', linewidth=1, linestyle='--', alpha=0.5, label='Medium')
    ax.axvline(0.8, color='gray', linewidth=1, linestyle='--', alpha=0.5, label='Large')
    ax.axvline(-0.2, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.axvline(-0.5, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.axvline(-0.8, color='gray', linewidth=1, linestyle='--', alpha=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(metric_names)
    ax.set_xlabel("Cohen's $d$ (Effect Size)")
    ax.set_title("Effect Size: $A \\rightarrow B$ vs $B \\rightarrow A$")
    ax.grid(True, alpha=0.3, axis='x')

    # add value labels
    for i, (bar, d) in enumerate(zip(bars, cohens_d)):
        width = bar.get_width()
        ax.text(width + 0.05 if width > 0 else width - 0.05, bar.get_y() + bar.get_height()/2.,
               f'{d:.3f}',
               ha='left' if width > 0 else 'right', va='center', fontsize=9)

    # add interpretation text
    ax.text(0.02, 0.98, r'$A \rightarrow B$ favored $\rightarrow$',
           transform=ax.transAxes, fontsize=8,
           verticalalignment='top', horizontalalignment='left')
    ax.text(0.98, 0.98, r'$\leftarrow$ $B \rightarrow A$ favored',
           transform=ax.transAxes, fontsize=8,
           verticalalignment='top', horizontalalignment='right')

    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    save_figure(fig, 'fig19_effect_size_analysis', output_dir=OUTPUT_DIR)
    plt.close()

    print("generated figure 19: effect size analysis")


def main():
    """generate all statistical analysis figures"""
    print("loading evaluation data...")
    metrics_a2b, metrics_b2a, fid_scores, n_samples = load_evaluation_data()

    print(f"loaded evaluation results for {n_samples} test samples")
    print(f"fid a2b: {fid_scores['a2b']:.2f}")
    print(f"fid b2a: {fid_scores['b2a']:.2f}")

    print("\ngenerating statistical figures...")

    generate_comprehensive_comparison(metrics_a2b, metrics_b2a, fid_scores)
    generate_approximated_distributions(metrics_a2b, metrics_b2a)
    generate_performance_radar(metrics_a2b, metrics_b2a)
    generate_statistical_summary_table()
    generate_effect_size_analysis(metrics_a2b, metrics_b2a, n_samples)

    print("\nall statistical figures generated successfully")
    print(f"output directory: {OUTPUT_DIR}")
    print("\nnote: per-sample correlation analysis requires running")
    print("inference to collect individual metric arrays.")
    print("see tools/inference/ scripts for qualitative analysis.")


if __name__ == '__main__':
    main()
