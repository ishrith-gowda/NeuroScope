#!/usr/bin/env python3
"""
LaTeX Figure Configuration for Publication-Grade Figures

Configures matplotlib to generate publication-ready figures with LaTeX rendering
suitable for MICCAI, IEEE TMI, and other top-tier medical imaging venues.

Author: NeuroScope Research Team
Date: January 2026
"""

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Use pgf backend for LaTeX rendering
matplotlib.use('Agg')  # Non-interactive backend

# Configure matplotlib for LaTeX rendering
plt.rcParams.update({
    # LaTeX configuration
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',

    # Font configuration
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.sans-serif': ['Computer Modern Sans Serif'],
    'font.monospace': ['Computer Modern Typewriter'],

    # Font sizes (matching LaTeX document)
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,

    # Figure sizing (IEEE two-column format)
    'figure.figsize': (7, 5),  # Default size
    'figure.dpi': 300,  # High resolution
    'savefig.dpi': 300,
    'savefig.format': 'pdf',  # Vector format for publication
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,

    # Line widths and markers
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'patch.linewidth': 1.0,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,

    # Grid and spines
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True,

    # Legend
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.fancybox': True,
    'legend.edgecolor': '0.8',

    # Colors - using colorblind-friendly palette
    'axes.prop_cycle': plt.cycler('color', [
        '#0173B2',  # Blue
        '#DE8F05',  # Orange
        '#029E73',  # Green
        '#CC78BC',  # Purple
        '#CA9161',  # Brown
        '#949494',  # Gray
        '#ECE133',  # Yellow
        '#56B4E9',  # Light blue
    ]),
})

# Configure seaborn for statistical plots
sns.set_palette("colorblind")
sns.set_style("whitegrid", {
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
})


def get_figure_size(width='columnwidth', fraction=1.0, aspect_ratio='golden'):
    """
    Calculate figure dimensions for publication.

    Args:
        width: 'columnwidth' (3.5in), 'textwidth' (7in), or custom float
        fraction: Fraction of width to use
        aspect_ratio: 'golden' (1.618), 'square' (1.0), or custom float

    Returns:
        tuple: (width, height) in inches
    """
    # IEEE column widths
    widths = {
        'columnwidth': 3.5,  # Single column
        'textwidth': 7.0,     # Full width (two columns)
    }

    # Get width in inches
    if isinstance(width, str):
        fig_width = widths.get(width, 7.0)
    else:
        fig_width = width

    fig_width = fig_width * fraction

    # Calculate height based on aspect ratio
    if aspect_ratio == 'golden':
        fig_height = fig_width / 1.618  # Golden ratio
    elif aspect_ratio == 'square':
        fig_height = fig_width
    elif isinstance(aspect_ratio, (int, float)):
        fig_height = fig_width / aspect_ratio
    else:
        fig_height = fig_width / 1.618  # Default to golden

    return (fig_width, fig_height)


def save_figure(fig, filename, output_dir='figures/main', formats=['pdf', 'png']):
    """
    Save figure in multiple formats with consistent settings.

    Args:
        fig: matplotlib figure object
        filename: output filename (without extension)
        output_dir: directory to save figures
        formats: list of formats to save ['pdf', 'png', 'pgf']
    """
    import os
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        output_path = output_dir / f"{filename}.{fmt}"
        fig.savefig(
            output_path,
            format=fmt,
            dpi=300 if fmt == 'png' else None,
            bbox_inches='tight',
            pad_inches=0.05,
        )
        print(f"Saved: {output_path}")


def add_statistical_annotation(ax, x1, x2, y, h, text, color='black'):
    """
    Add statistical significance annotation to plot.

    Args:
        ax: matplotlib axes
        x1, x2: x positions for bracket
        y: y position for bracket base
        h: height of bracket
        text: annotation text (*, **, ***, ns)
        color: bracket color
    """
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=color)
    ax.text((x1+x2)*0.5, y+h, text, ha='center', va='bottom', color=color, fontsize=10)


def create_colormap(name='viridis', reverse=False):
    """
    Get colormap suitable for medical images.

    Args:
        name: colormap name ('gray', 'viridis', 'plasma', 'hot', 'jet')
        reverse: reverse the colormap

    Returns:
        matplotlib colormap
    """
    cmap = plt.get_cmap(name)
    if reverse:
        cmap = cmap.reversed()
    return cmap


# Column widths for IEEE format
COLUMNWIDTH = 3.5  # inches
TEXTWIDTH = 7.0    # inches

# Common figure sizes
FIGURE_SIZES = {
    'single': get_figure_size('columnwidth', 1.0),
    'single_tall': get_figure_size('columnwidth', 1.0, aspect_ratio=1.0),
    'double': get_figure_size('textwidth', 1.0),
    'double_tall': get_figure_size('textwidth', 1.0, aspect_ratio=1.0),
}

# Color schemes
COLORS = {
    'primary': '#0173B2',
    'secondary': '#DE8F05',
    'success': '#029E73',
    'danger': '#D55E00',
    'warning': '#F0E442',
    'info': '#56B4E9',
    'gray': '#949494',
}

# Statistical significance markers
SIGNIFICANCE = {
    'ns': 'n.s.',
    'p<0.05': r'$^*$',
    'p<0.01': r'$^{**}$',
    'p<0.001': r'$^{***}$',
}


if __name__ == '__main__':
    # Test LaTeX rendering
    print("Testing LaTeX configuration...")

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single'])
    x = np.linspace(0, 2*np.pi, 100)
    ax.plot(x, np.sin(x), label=r'$\sin(x)$')
    ax.plot(x, np.cos(x), label=r'$\cos(x)$')
    ax.set_xlabel(r'$x$ (radians)')
    ax.set_ylabel(r'$f(x)$')
    ax.set_title(r'Test Figure with LaTeX Rendering')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(fig, 'test_latex_rendering', output_dir='figures/main')
    plt.close()

    print("LaTeX configuration test complete!")
    print(f"Test figure saved to: figures/main/test_latex_rendering.pdf")
