#!/usr/bin/env python3
"""
publication figures for radiomics preservation analysis.

generates comprehensive visualizations:
- correlation heatmaps by feature category
- bland-altman plots
- preservation bar charts
- scatter plots with regression lines
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from scipy import stats


# publication style settings
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

# color scheme
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'success': '#C73E1D',
    'neutral': '#3B3B3B',
    'light': '#E8E8E8',
}

CATEGORY_COLORS = {
    'first_order': '#2E86AB',
    'glcm': '#A23B72',
    'shape': '#F18F01',
    'other': '#3B3B3B',
}


def plot_correlation_heatmap(
    preservation_results: Dict,
    output_path: Path,
    title: str = 'Feature Preservation Correlation Matrix'
):
    """
    plot correlation heatmap for feature preservation metrics.

    args:
        preservation_results: results from preservation analysis
        output_path: path to save figure
        title: figure title
    """
    per_feature = preservation_results.get('per_feature', {})

    if not per_feature:
        print('[fig] no per-feature results for heatmap')
        return

    feature_names = list(per_feature.keys())
    n_features = len(feature_names)

    if n_features > 50:
        # sample features if too many
        step = n_features // 50
        feature_names = feature_names[::step]

    # extract metrics
    cccs = [per_feature[f]['ccc'] for f in feature_names]
    iccs = [per_feature[f]['icc'] for f in feature_names]
    correlations = [per_feature[f]['pearson_r'] for f in feature_names]

    # create figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # ccc heatmap
    ax = axes[0]
    ccc_matrix = np.array(cccs).reshape(-1, 1)
    im = ax.imshow(ccc_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title('(a) Concordance Correlation')
    ax.set_xlabel('CCC')
    ax.set_ylabel('Features')
    ax.set_xticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)

    # icc heatmap
    ax = axes[1]
    icc_matrix = np.array(iccs).reshape(-1, 1)
    im = ax.imshow(icc_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title('(b) Intraclass Correlation')
    ax.set_xlabel('ICC')
    ax.set_ylabel('Features')
    ax.set_xticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)

    # correlation heatmap
    ax = axes[2]
    corr_matrix = np.array(correlations).reshape(-1, 1)
    im = ax.imshow(corr_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title('(c) Pearson Correlation')
    ax.set_xlabel('r')
    ax.set_ylabel('Features')
    ax.set_xticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight')
    plt.close()

    print(f'[fig] saved correlation heatmap to {output_path}')


def plot_preservation_by_category(
    preservation_results: Dict,
    output_path: Path,
    method_name: str = 'SA-CycleGAN-2.5D'
):
    """
    plot bar chart of preservation metrics by feature category.

    args:
        preservation_results: results from preservation analysis
        output_path: path to save figure
        method_name: name of harmonization method
    """
    per_category = preservation_results.get('per_category', {})

    if not per_category:
        print('[fig] no per-category results for bar chart')
        return

    categories = list(per_category.keys())
    categories = [c for c in categories if per_category[c]['n_features'] > 0]

    if not categories:
        return

    # extract metrics
    mean_cccs = [per_category[c]['mean_ccc'] for c in categories]
    std_cccs = [per_category[c]['std_ccc'] for c in categories]
    mean_iccs = [per_category[c]['mean_icc'] for c in categories]
    std_iccs = [per_category[c]['std_icc'] for c in categories]
    n_features = [per_category[c]['n_features'] for c in categories]

    # format category names
    category_labels = [c.replace('_', ' ').title() for c in categories]

    # create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(categories))
    width = 0.35

    # ccc bar chart
    ax = axes[0]
    colors = [CATEGORY_COLORS.get(c, COLORS['neutral']) for c in categories]
    bars = ax.bar(x, mean_cccs, width, yerr=std_cccs, capsize=3,
                  color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Excellent (>0.9)')
    ax.axhline(y=0.75, color='orange', linestyle='--', alpha=0.7, label='Good (>0.75)')
    ax.set_ylabel('Concordance Correlation Coefficient')
    ax.set_title(f'(a) CCC by Feature Category')
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=8)

    # add n_features labels
    for i, (bar, n) in enumerate(zip(bars, n_features)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'n={n}', ha='center', va='bottom', fontsize=8)

    # icc bar chart
    ax = axes[1]
    bars = ax.bar(x, mean_iccs, width, yerr=std_iccs, capsize=3,
                  color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Excellent (>0.9)')
    ax.axhline(y=0.75, color='orange', linestyle='--', alpha=0.7, label='Good (>0.75)')
    ax.set_ylabel('Intraclass Correlation Coefficient')
    ax.set_title(f'(b) ICC by Feature Category')
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight')
    plt.close()

    print(f'[fig] saved preservation by category to {output_path}')


def plot_bland_altman(
    original: np.ndarray,
    harmonized: np.ndarray,
    feature_names: List[str],
    output_path: Path,
    n_features_to_plot: int = 4
):
    """
    create bland-altman plots for selected features.

    args:
        original: original feature values (n_samples, n_features)
        harmonized: harmonized feature values
        feature_names: list of feature names
        output_path: path to save figure
        n_features_to_plot: number of features to show
    """
    n_features = min(n_features_to_plot, original.shape[1], len(feature_names))

    # select features with varying preservation levels
    correlations = []
    for i in range(original.shape[1]):
        r, _ = stats.pearsonr(original[:, i], harmonized[:, i])
        correlations.append(r if not np.isnan(r) else 0)

    # select features spanning range of correlations
    sorted_indices = np.argsort(correlations)
    step = len(sorted_indices) // n_features
    selected_indices = sorted_indices[::step][:n_features]

    # create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, (ax, feature_idx) in enumerate(zip(axes, selected_indices)):
        orig = original[:, feature_idx]
        harm = harmonized[:, feature_idx]

        # remove invalid values
        valid = ~(np.isnan(orig) | np.isnan(harm))
        orig, harm = orig[valid], harm[valid]

        if len(orig) < 3:
            continue

        # bland-altman calculations
        mean_values = (orig + harm) / 2
        diff_values = harm - orig
        mean_diff = np.mean(diff_values)
        std_diff = np.std(diff_values, ddof=1)
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff

        # plot
        ax.scatter(mean_values, diff_values, alpha=0.5, s=10, c=COLORS['primary'])
        ax.axhline(y=mean_diff, color=COLORS['secondary'], linestyle='-',
                   label=f'Mean: {mean_diff:.3f}')
        ax.axhline(y=loa_upper, color=COLORS['tertiary'], linestyle='--',
                   label=f'+1.96 SD: {loa_upper:.3f}')
        ax.axhline(y=loa_lower, color=COLORS['tertiary'], linestyle='--',
                   label=f'-1.96 SD: {loa_lower:.3f}')

        feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f'Feature {feature_idx}'
        # truncate long names
        if len(feature_name) > 25:
            feature_name = feature_name[:22] + '...'

        ax.set_title(f'({chr(97+idx)}) {feature_name}')
        ax.set_xlabel('Mean of Original and Harmonized')
        ax.set_ylabel('Difference (Harmonized - Original)')
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight')
    plt.close()

    print(f'[fig] saved bland-altman plots to {output_path}')


def plot_preservation_scatter(
    original: np.ndarray,
    harmonized: np.ndarray,
    feature_names: List[str],
    output_path: Path,
    n_features_to_plot: int = 4
):
    """
    create scatter plots with regression lines for selected features.

    args:
        original: original feature values
        harmonized: harmonized feature values
        feature_names: list of feature names
        output_path: path to save figure
        n_features_to_plot: number of features to show
    """
    n_features = min(n_features_to_plot, original.shape[1], len(feature_names))

    # select representative features from each category
    selected_indices = []
    for prefix in ['fo_mean', 'glcm_contrast', 'shape_area', 'fo_entropy']:
        for i, name in enumerate(feature_names):
            if prefix in name.lower() and i not in selected_indices:
                selected_indices.append(i)
                break
        if len(selected_indices) >= n_features:
            break

    # fill remaining with random selection
    while len(selected_indices) < n_features:
        idx = np.random.randint(0, len(feature_names))
        if idx not in selected_indices:
            selected_indices.append(idx)

    # create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, (ax, feature_idx) in enumerate(zip(axes, selected_indices)):
        orig = original[:, feature_idx]
        harm = harmonized[:, feature_idx]

        # remove invalid values
        valid = ~(np.isnan(orig) | np.isnan(harm))
        orig, harm = orig[valid], harm[valid]

        if len(orig) < 3:
            continue

        # scatter plot
        ax.scatter(orig, harm, alpha=0.5, s=10, c=COLORS['primary'])

        # regression line
        slope, intercept, r_value, _, _ = stats.linregress(orig, harm)
        x_line = np.array([orig.min(), orig.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=COLORS['secondary'], linewidth=2,
                label=f'r = {r_value:.3f}')

        # identity line
        ax.plot(x_line, x_line, color=COLORS['neutral'], linestyle='--',
                alpha=0.5, label='Identity')

        feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f'Feature {feature_idx}'
        if len(feature_name) > 25:
            feature_name = feature_name[:22] + '...'

        ax.set_title(f'({chr(97+idx)}) {feature_name}')
        ax.set_xlabel('Original Value')
        ax.set_ylabel('Harmonized Value')
        ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight')
    plt.close()

    print(f'[fig] saved preservation scatter plots to {output_path}')


def plot_comprehensive_summary(
    results: Dict,
    output_path: Path,
    method_name: str = 'SA-CycleGAN-2.5D'
):
    """
    create comprehensive summary figure for radiomics preservation.

    args:
        results: complete preservation results
        output_path: path to save figure
        method_name: name of harmonization method
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # (a) overall preservation metrics
    ax1 = fig.add_subplot(gs[0, 0])
    overall = results.get('overall', {})
    metrics = ['CCC', 'ICC', 'Correlation']
    values = [
        overall.get('mean_ccc', 0),
        overall.get('mean_icc', 0),
        overall.get('mean_correlation', 0)
    ]
    errors = [
        overall.get('std_ccc', 0),
        overall.get('std_icc', 0),
        overall.get('std_correlation', 0)
    ]
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]

    bars = ax1.bar(metrics, values, yerr=errors, capsize=5, color=colors,
                   edgecolor='black', linewidth=0.5)
    ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.7)
    ax1.axhline(y=0.75, color='orange', linestyle='--', alpha=0.7)
    ax1.set_ylabel('Coefficient Value')
    ax1.set_title('(a) Overall Preservation Metrics')
    ax1.set_ylim(0, 1.1)

    # add value labels
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # (b) preservation quality distribution
    ax2 = fig.add_subplot(gs[0, 1])
    quality_counts = [
        overall.get('excellent_preservation', 0),
        overall.get('good_preservation', 0),
        overall.get('moderate_preservation', 0),
        overall.get('poor_preservation', 0)
    ]
    quality_labels = ['Excellent\n(>0.9)', 'Good\n(0.75-0.9)', 'Moderate\n(0.5-0.75)', 'Poor\n(<0.5)']
    quality_colors = ['#2E7D32', '#66BB6A', '#FFB74D', '#E57373']

    ax2.bar(quality_labels, quality_counts, color=quality_colors,
            edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Number of Features')
    ax2.set_title('(b) Preservation Quality Distribution')

    # add value labels
    for i, count in enumerate(quality_counts):
        ax2.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=9)

    # (c) preservation by category
    ax3 = fig.add_subplot(gs[0, 2])
    per_category = results.get('per_category', {})
    if per_category:
        categories = [c for c in per_category.keys() if per_category[c]['n_features'] > 0]
        cat_cccs = [per_category[c]['mean_ccc'] for c in categories]
        cat_labels = [c.replace('_', '\n').title() for c in categories]
        cat_colors = [CATEGORY_COLORS.get(c, COLORS['neutral']) for c in categories]

        ax3.bar(cat_labels, cat_cccs, color=cat_colors, edgecolor='black', linewidth=0.5)
        ax3.axhline(y=0.9, color='green', linestyle='--', alpha=0.7)
        ax3.set_ylabel('Mean CCC')
        ax3.set_title('(c) CCC by Feature Category')
        ax3.set_ylim(0, 1.1)

    # (d-f) domain-specific preservation
    domains = ['domain_a_preservation', 'domain_b_preservation']
    domain_labels = ['Domain A (BraTS)', 'Domain B (UPenn)']

    for i, (domain_key, domain_label) in enumerate(zip(domains, domain_labels)):
        ax = fig.add_subplot(gs[1, i])
        domain_results = results.get(domain_key, {}).get('overall', {})

        metrics = ['CCC', 'ICC', 'Correlation']
        values = [
            domain_results.get('mean_ccc', 0),
            domain_results.get('mean_icc', 0),
            domain_results.get('mean_correlation', 0)
        ]

        bars = ax.bar(metrics, values, color=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary']],
                      edgecolor='black', linewidth=0.5)
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.7)
        ax.set_ylabel('Coefficient Value')
        ax.set_title(f'({chr(100+i)}) {domain_label}')
        ax.set_ylim(0, 1.1)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # (g) cross-domain alignment comparison
    ax7 = fig.add_subplot(gs[1, 2])
    raw_alignment = results.get('cross_domain_raw', {}).get('overall', {})
    harm_alignment = results.get('cross_domain_harmonized', {}).get('overall', {})

    x = np.arange(3)
    width = 0.35

    raw_values = [
        raw_alignment.get('mean_ccc', 0),
        raw_alignment.get('mean_icc', 0),
        raw_alignment.get('mean_correlation', 0)
    ]
    harm_values = [
        harm_alignment.get('mean_ccc', 0),
        harm_alignment.get('mean_icc', 0),
        harm_alignment.get('mean_correlation', 0)
    ]

    ax7.bar(x - width/2, raw_values, width, label='Raw', color=COLORS['light'],
            edgecolor='black', linewidth=0.5)
    ax7.bar(x + width/2, harm_values, width, label='Harmonized', color=COLORS['primary'],
            edgecolor='black', linewidth=0.5)
    ax7.set_ylabel('Coefficient Value')
    ax7.set_title('(f) Cross-Domain Alignment')
    ax7.set_xticks(x)
    ax7.set_xticklabels(['CCC', 'ICC', 'Correlation'])
    ax7.legend(loc='lower right')
    ax7.set_ylim(0, 1.1)

    # (h) key findings text box
    ax8 = fig.add_subplot(gs[2, :])
    ax8.axis('off')

    findings_text = f"""
Key Findings - {method_name} Radiomics Preservation:

1. Overall Preservation: Mean CCC = {overall.get('mean_ccc', 0):.3f}, Mean ICC = {overall.get('mean_icc', 0):.3f}
   - {overall.get('excellent_preservation', 0)} features with excellent preservation (CCC > 0.9)
   - {overall.get('good_preservation', 0)} features with good preservation (0.75 < CCC < 0.9)

2. Domain-Specific Results:
   - Domain A: CCC = {results.get('domain_a_preservation', {}).get('overall', {}).get('mean_ccc', 0):.3f}
   - Domain B: CCC = {results.get('domain_b_preservation', {}).get('overall', {}).get('mean_ccc', 0):.3f}

3. Cross-Domain Alignment:
   - Raw alignment CCC: {raw_alignment.get('mean_ccc', 0):.3f}
   - Harmonized alignment CCC: {harm_alignment.get('mean_ccc', 0):.3f}
   - Improvement: {(harm_alignment.get('mean_ccc', 0) - raw_alignment.get('mean_ccc', 0)):.3f}

Interpretation: Harmonization effectively preserves radiomics features within each domain
while improving cross-domain feature alignment, supporting clinical validity.
"""

    ax8.text(0.05, 0.95, findings_text, transform=ax8.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight')
    plt.close()

    print(f'[fig] saved comprehensive summary to {output_path}')


def create_latex_table(results: Dict, output_path: Path):
    """
    create latex table for radiomics preservation results.

    args:
        results: preservation results dictionary
        output_path: path to save table
    """
    overall = results.get('overall', {})
    domain_a = results.get('domain_a_preservation', {}).get('overall', {})
    domain_b = results.get('domain_b_preservation', {}).get('overall', {})
    per_category = results.get('per_category', {})

    latex = r"""\begin{table}[htbp]
\centering
\caption{Radiomics Feature Preservation Analysis}
\label{tab:radiomics_preservation}
\begin{tabular}{lccc}
\toprule
Category & CCC & ICC & Pearson r \\
\midrule
"""

    # per-category rows
    for category, metrics in per_category.items():
        if metrics['n_features'] == 0:
            continue
        cat_name = category.replace('_', ' ').title()
        latex += f"{cat_name} (n={metrics['n_features']}) & "
        latex += f"{metrics['mean_ccc']:.3f} $\\pm$ {metrics['std_ccc']:.3f} & "
        latex += f"{metrics['mean_icc']:.3f} $\\pm$ {metrics['std_icc']:.3f} & "
        latex += f"{metrics['mean_correlation']:.3f} $\\pm$ {metrics['std_correlation']:.3f} \\\\\n"

    latex += r"""\midrule
"""

    # overall row
    latex += f"Overall (n={overall.get('n_features', 0)}) & "
    latex += f"{overall.get('mean_ccc', 0):.3f} $\\pm$ {overall.get('std_ccc', 0):.3f} & "
    latex += f"{overall.get('mean_icc', 0):.3f} $\\pm$ {overall.get('std_icc', 0):.3f} & "
    latex += f"{overall.get('mean_correlation', 0):.3f} $\\pm$ {overall.get('std_correlation', 0):.3f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f'[table] saved latex table to {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='generate radiomics preservation figures'
    )
    parser.add_argument('--results', type=str, required=True,
                       help='path to preservation results json')
    parser.add_argument('--original-a', type=str,
                       help='path to original domain a features')
    parser.add_argument('--harmonized-a', type=str,
                       help='path to harmonized domain a features')
    parser.add_argument('--feature-names', type=str,
                       help='path to feature names json')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='output directory for figures')
    parser.add_argument('--method-name', type=str, default='SA-CycleGAN-2.5D',
                       help='name of harmonization method')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load results
    with open(args.results) as f:
        results = json.load(f)

    # get domain preservation results for plotting
    domain_a_results = results.get('domain_a_preservation', {})

    # generate figures
    print('[figures] generating radiomics preservation figures...')

    # correlation heatmap
    plot_correlation_heatmap(
        domain_a_results,
        output_dir / 'fig_radiomics_correlation_heatmap.pdf'
    )

    # preservation by category
    plot_preservation_by_category(
        domain_a_results,
        output_dir / 'fig_radiomics_preservation_category.pdf',
        args.method_name
    )

    # comprehensive summary
    plot_comprehensive_summary(
        results,
        output_dir / 'fig_radiomics_comprehensive.pdf',
        args.method_name
    )

    # bland-altman and scatter plots if feature data available
    if args.original_a and args.harmonized_a:
        original = np.load(args.original_a)
        harmonized = np.load(args.harmonized_a)

        if args.feature_names:
            with open(args.feature_names) as f:
                feature_names = json.load(f)
        else:
            feature_names = [f'feature_{i}' for i in range(original.shape[1])]

        plot_bland_altman(
            original, harmonized, feature_names,
            output_dir / 'fig_radiomics_bland_altman.pdf'
        )

        plot_preservation_scatter(
            original, harmonized, feature_names,
            output_dir / 'fig_radiomics_scatter.pdf'
        )

    # latex table
    create_latex_table(
        domain_a_results,
        output_dir / 'table_radiomics_preservation.tex'
    )

    print('=' * 60)
    print('[figures] all radiomics figures generated')
    print(f'[figures] saved to {output_dir}')


if __name__ == '__main__':
    main()
