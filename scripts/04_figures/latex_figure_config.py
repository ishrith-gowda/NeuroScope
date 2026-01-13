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
import matplotlib as mpl
import seaborn as sns
import numpy as np

# use agg backend for non-interactive rendering
matplotlib.use('Agg')

# use matplotlib default style, then configure
plt.style.use('default')

# configure matplotlib for latex rendering and publication quality
# matching your exact working configuration with proper font hierarchy
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16,

    # high resolution output
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,

    # aesthetics
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'lines.linewidth': 2,
    'axes.axisbelow': True,
    'savefig.format': 'pdf',
})

# define consistent color palette (colorblind-friendly)
palette = sns.color_palette('colorblind')
COLORS_LIST = palette

# maintain backward compatibility with existing scripts
COLORS = {
    'primary': palette[0],    # blue
    'secondary': palette[1],  # orange
    'success': palette[2],    # green
    'danger': palette[3],     # red
    'warning': palette[4],    # purple
    'info': palette[5],       # brown
    'gray': palette[7],       # gray
}


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


# column widths for ieee format
COLUMNWIDTH = 3.5  # inches
TEXTWIDTH = 7.0    # inches

# common figure sizes
FIGURE_SIZES = {
    'single': get_figure_size('columnwidth', 1.0),
    'single_tall': get_figure_size('columnwidth', 1.0, aspect_ratio=1.0),
    'double': get_figure_size('textwidth', 1.0),
    'double_tall': get_figure_size('textwidth', 1.0, aspect_ratio=1.0),
}

# statistical significance markers
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
