"""
Visualization Utilities.

Publication-quality visualization for medical imaging,
training curves, and statistical results.
"""

from typing import Optional, Dict, List, Tuple, Union, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


# Slice Visualization

def plot_slice(
    data: np.ndarray,
    slice_idx: int = None,
    axis: int = 2,
    title: str = None,
    cmap: str = 'gray',
    vmin: float = None,
    vmax: float = None,
    colorbar: bool = True,
    ax: plt.Axes = None,
    figsize: Tuple[int, int] = (6, 6)
) -> plt.Figure:
    """
    Plot a single slice from 3D volume.
    
    Args:
        data: 3D volume
        slice_idx: Slice index (default: middle)
        axis: Axis to slice along
        title: Plot title
        cmap: Colormap
        vmin, vmax: Value range
        colorbar: Show colorbar
        ax: Existing axes
        figsize: Figure size
        
    Returns:
        Figure object
    """
    if slice_idx is None:
        slice_idx = data.shape[axis] // 2
    
    # Extract slice
    if axis == 0:
        slice_data = data[slice_idx, :, :]
    elif axis == 1:
        slice_data = data[:, slice_idx, :]
    else:
        slice_data = data[:, :, slice_idx]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    im = ax.imshow(slice_data.T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    
    if title:
        ax.set_title(title)
    
    ax.axis('off')
    
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
    return fig


def plot_slice_comparison(
    images: List[np.ndarray],
    labels: List[str],
    slice_idx: int = None,
    axis: int = 2,
    cmap: str = 'gray',
    figsize: Tuple[int, int] = None,
    suptitle: str = None
) -> plt.Figure:
    """
    Compare multiple images side by side.
    
    Args:
        images: List of 3D volumes
        labels: Labels for each image
        slice_idx: Slice index
        axis: Slice axis
        cmap: Colormap
        figsize: Figure size
        suptitle: Super title
        
    Returns:
        Figure object
    """
    n_images = len(images)
    
    if figsize is None:
        figsize = (4 * n_images, 4)
    
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    if n_images == 1:
        axes = [axes]
    
    # Compute global vmin/vmax
    all_data = np.concatenate([img.flatten() for img in images])
    vmin, vmax = np.percentile(all_data, [1, 99])
    
    for ax, img, label in zip(axes, images, labels):
        if slice_idx is None:
            idx = img.shape[axis] // 2
        else:
            idx = slice_idx
        
        if axis == 0:
            slice_data = img[idx, :, :]
        elif axis == 1:
            slice_data = img[:, idx, :]
        else:
            slice_data = img[:, :, idx]
        
        ax.imshow(slice_data.T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(label)
        ax.axis('off')
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.02)
    
    plt.tight_layout()
    return fig


def plot_montage(
    volume: np.ndarray,
    n_slices: int = 9,
    axis: int = 2,
    cmap: str = 'gray',
    figsize: Tuple[int, int] = (12, 12),
    title: str = None
) -> plt.Figure:
    """
    Create montage of slices from volume.
    
    Args:
        volume: 3D volume
        n_slices: Number of slices to show
        axis: Slice axis
        cmap: Colormap
        figsize: Figure size
        title: Title
        
    Returns:
        Figure object
    """
    n_cols = int(np.ceil(np.sqrt(n_slices)))
    n_rows = int(np.ceil(n_slices / n_cols))
    
    slice_indices = np.linspace(0, volume.shape[axis] - 1, n_slices).astype(int)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    vmin, vmax = np.percentile(volume, [1, 99])
    
    for i, (ax, idx) in enumerate(zip(axes, slice_indices)):
        if axis == 0:
            slice_data = volume[idx, :, :]
        elif axis == 1:
            slice_data = volume[:, idx, :]
        else:
            slice_data = volume[:, :, idx]
        
        ax.imshow(slice_data.T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f'Slice {idx}', fontsize=8)
        ax.axis('off')
    
    # Hide unused axes
    for ax in axes[len(slice_indices):]:
        ax.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    return fig


# Difference Maps

def plot_difference_map(
    original: np.ndarray,
    modified: np.ndarray,
    slice_idx: int = None,
    axis: int = 2,
    cmap: str = 'RdBu_r',
    symmetric: bool = True,
    figsize: Tuple[int, int] = (15, 4),
    title: str = None
) -> plt.Figure:
    """
    Plot original, modified, and difference map.
    
    Args:
        original: Original image
        modified: Modified image
        slice_idx: Slice index
        axis: Slice axis
        cmap: Colormap for difference
        symmetric: Symmetric color range
        figsize: Figure size
        title: Title
        
    Returns:
        Figure object
    """
    diff = modified - original
    
    if slice_idx is None:
        slice_idx = original.shape[axis] // 2
    
    # Extract slices
    def get_slice(vol, idx, ax):
        if ax == 0:
            return vol[idx, :, :]
        elif ax == 1:
            return vol[:, idx, :]
        return vol[:, :, idx]
    
    orig_slice = get_slice(original, slice_idx, axis)
    mod_slice = get_slice(modified, slice_idx, axis)
    diff_slice = get_slice(diff, slice_idx, axis)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original
    vmin, vmax = np.percentile(original, [1, 99])
    axes[0].imshow(orig_slice.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Modified
    axes[1].imshow(mod_slice.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title('Modified')
    axes[1].axis('off')
    
    # Difference
    if symmetric:
        max_abs = np.max(np.abs(diff_slice))
        im = axes[2].imshow(
            diff_slice.T, cmap=cmap, vmin=-max_abs, vmax=max_abs, origin='lower'
        )
    else:
        im = axes[2].imshow(diff_slice.T, cmap=cmap, origin='lower')
    
    axes[2].set_title('Difference')
    axes[2].axis('off')
    
    # Colorbar for difference
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.tight_layout()
    return fig


def plot_attention_overlay(
    image: np.ndarray,
    attention: np.ndarray,
    slice_idx: int = None,
    axis: int = 2,
    alpha: float = 0.5,
    cmap: str = 'jet',
    figsize: Tuple[int, int] = (6, 6),
    title: str = None
) -> plt.Figure:
    """
    Overlay attention map on image.
    
    Args:
        image: Base image
        attention: Attention map
        slice_idx: Slice index
        axis: Slice axis
        alpha: Attention transparency
        cmap: Attention colormap
        figsize: Figure size
        title: Title
        
    Returns:
        Figure object
    """
    if slice_idx is None:
        slice_idx = image.shape[axis] // 2
    
    def get_slice(vol, idx, ax):
        if ax == 0:
            return vol[idx, :, :]
        elif ax == 1:
            return vol[:, idx, :]
        return vol[:, :, idx]
    
    img_slice = get_slice(image, slice_idx, axis)
    att_slice = get_slice(attention, slice_idx, axis)
    
    # Normalize
    img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
    att_norm = (att_slice - att_slice.min()) / (att_slice.max() - att_slice.min() + 1e-8)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.imshow(img_norm.T, cmap='gray', origin='lower')
    im = ax.imshow(att_norm.T, cmap=cmap, alpha=alpha, origin='lower')
    
    if title:
        ax.set_title(title)
    
    ax.axis('off')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='Attention')
    
    return fig


# Training Visualization

def plot_training_curves(
    history: Dict[str, List[float]],
    metrics: List[str] = None,
    figsize: Tuple[int, int] = (12, 4),
    title: str = 'Training Curves'
) -> plt.Figure:
    """
    Plot training curves.
    
    Args:
        history: Dict of metric name -> values
        metrics: Metrics to plot (None = all)
        figsize: Figure size
        title: Title
        
    Returns:
        Figure object
    """
    if metrics is None:
        metrics = list(history.keys())
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        values = history[metric]
        epochs = range(1, len(values) + 1)
        
        ax.plot(epochs, values, 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_loss_landscape(
    losses: np.ndarray,
    x_range: Tuple[float, float] = (-1, 1),
    y_range: Tuple[float, float] = (-1, 1),
    figsize: Tuple[int, int] = (8, 6),
    title: str = 'Loss Landscape'
) -> plt.Figure:
    """
    Plot 2D loss landscape visualization.
    
    Args:
        losses: 2D array of loss values
        x_range: X-axis range
        y_range: Y-axis range
        figsize: Figure size
        title: Title
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.linspace(x_range[0], x_range[1], losses.shape[0])
    y = np.linspace(y_range[0], y_range[1], losses.shape[1])
    X, Y = np.meshgrid(x, y)
    
    # Contour plot
    contour = ax.contourf(X, Y, losses.T, levels=50, cmap='viridis')
    ax.contour(X, Y, losses.T, levels=10, colors='white', linewidths=0.5, alpha=0.5)
    
    plt.colorbar(contour, ax=ax, label='Loss')
    
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_title(title)
    
    return fig


# Statistical Visualization

def plot_box_comparison(
    data: Dict[str, List[float]],
    xlabel: str = 'Method',
    ylabel: str = 'Score',
    title: str = None,
    figsize: Tuple[int, int] = (10, 6),
    colors: List[str] = None
) -> plt.Figure:
    """
    Box plot comparison of methods.
    
    Args:
        data: Dict of method name -> values
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Title
        figsize: Figure size
        colors: Box colors
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    methods = list(data.keys())
    values = [data[m] for m in methods]
    
    bp = ax.boxplot(values, labels=methods, patch_artist=True)
    
    if colors is None:
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)
    
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_violin_comparison(
    data: Dict[str, List[float]],
    xlabel: str = 'Method',
    ylabel: str = 'Score',
    title: str = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Violin plot comparison.
    
    Args:
        data: Dict of method name -> values
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Title
        figsize: Figure size
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    methods = list(data.keys())
    values = [data[m] for m in methods]
    positions = range(1, len(methods) + 1)
    
    parts = ax.violinplot(values, positions, showmeans=True, showmedians=True)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(methods)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)
    
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confidence_intervals(
    data: Dict[str, Tuple[float, float, float]],
    xlabel: str = 'Value',
    title: str = 'Confidence Intervals',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot confidence intervals.
    
    Args:
        data: Dict of label -> (mean, lower, upper)
        xlabel: X-axis label
        title: Title
        figsize: Figure size
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    labels = list(data.keys())
    n = len(labels)
    y_positions = range(n)
    
    for i, label in enumerate(labels):
        mean, lower, upper = data[label]
        
        ax.errorbar(
            mean, i,
            xerr=[[mean - lower], [upper - mean]],
            fmt='o', capsize=5, capthick=2, linewidth=2, markersize=8
        )
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_effect_size_forest(
    effects: Dict[str, Tuple[float, float, float]],
    title: str = 'Effect Size Forest Plot',
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Forest plot for effect sizes.
    
    Args:
        effects: Dict of comparison -> (effect, lower CI, upper CI)
        title: Title
        figsize: Figure size
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    labels = list(effects.keys())
    n = len(labels)
    y_positions = range(n)
    
    for i, label in enumerate(labels):
        effect, lower, upper = effects[label]
        
        # Effect size point and CI
        ax.errorbar(
            effect, i,
            xerr=[[effect - lower], [upper - effect]],
            fmt='s', capsize=5, capthick=2, linewidth=2, markersize=10,
            color='steelblue'
        )
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Effect Size (Cohen\'s d)')
    ax.set_title(title)
    
    # Reference lines
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small')
    ax.axvline(x=0.5, color='gray', linestyle='-.', alpha=0.5, label='Medium')
    ax.axvline(x=0.8, color='gray', linestyle=':', alpha=0.5, label='Large')
    
    ax.legend(loc='upper right')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


# Publication Figures

def create_figure_grid(
    n_rows: int,
    n_cols: int,
    figsize: Tuple[int, int] = None,
    width_ratios: List[float] = None,
    height_ratios: List[float] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create figure grid for publication.
    
    Args:
        n_rows: Number of rows
        n_cols: Number of columns
        figsize: Figure size
        width_ratios: Column width ratios
        height_ratios: Row height ratios
        
    Returns:
        Tuple of (figure, axes array)
    """
    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)
    
    fig = plt.figure(figsize=figsize)
    
    gs = gridspec.GridSpec(
        n_rows, n_cols,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        hspace=0.3,
        wspace=0.3
    )
    
    axes = np.empty((n_rows, n_cols), dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            axes[i, j] = fig.add_subplot(gs[i, j])
    
    return fig, axes


def save_publication_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    formats: List[str] = None,
    dpi: int = 300
):
    """
    Save figure in multiple formats for publication.
    
    Args:
        fig: Figure to save
        path: Base path (without extension)
        formats: List of formats (default: ['pdf', 'png', 'svg'])
        dpi: DPI for raster formats
    """
    if formats is None:
        formats = ['pdf', 'png', 'svg']
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        fig.savefig(
            f"{path}.{fmt}",
            format=fmt,
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0.1
        )
