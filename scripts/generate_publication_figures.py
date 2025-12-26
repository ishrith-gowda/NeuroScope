#!/usr/bin/env python3
"""
Publication Figure Generation Script.

Automatically generates all figures for the paper including:
- Architecture diagrams
- Training curves
- Qualitative comparisons
- Quantitative results (bar charts, heatmaps)
- Ablation study visualizations
- Statistical significance plots

Usage:
    python scripts/generate_publication_figures.py \
        --results-dir ./results \
        --output-dir ./figures/publication \
        --format pdf png \
        --dpi 300
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as scipy_stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set publication-ready style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper")
sns.set_palette("Set2")


class PublicationFigureGenerator:
    """Generates publication-quality figures for research paper."""

    def __init__(
        self,
        results_dir: Path,
        output_dir: Path,
        formats: List[str] = None,
        dpi: int = 300,
    ):
        """
        Initialize figure generator.

        Args:
            results_dir: Directory containing evaluation results
            output_dir: Output directory for figures
            formats: Output formats (e.g., ['pdf', 'png'])
            dpi: DPI for raster formats
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.formats = formats or ['pdf', 'png']
        self.dpi = dpi

        # Figure styling
        self.colors = {
            'sa_cyclegan': '#2E86AB',  # Blue
            'cyclegan': '#A23B72',      # Purple
            'combat': '#F18F01',        # Orange
            'cut': '#C73E1D',           # Red
            'histogram': '#6A994E',     # Green
        }

    def save_figure(self, fig: plt.Figure, filename: str):
        """Save figure in multiple formats."""
        for fmt in self.formats:
            filepath = self.output_dir / f"{filename}.{fmt}"
            fig.savefig(
                filepath,
                format=fmt,
                dpi=self.dpi,
                bbox_inches='tight',
                pad_inches=0.1
            )
            logger.info(f"Saved: {filepath}")

    def generate_training_curves(self, tensorboard_logs: Dict):
        """
        Generate training curves figure.

        Args:
            tensorboard_logs: Dictionary of training logs
        """
        logger.info("Generating training curves...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')

        # Loss curves
        ax = axes[0, 0]
        for model_name, logs in tensorboard_logs.items():
            if 'train/loss_G' in logs:
                ax.plot(
                    logs['epochs'],
                    logs['train/loss_G'],
                    label=model_name,
                    color=self.colors.get(model_name, '#000000'),
                    linewidth=2
                )
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Generator Loss', fontsize=12)
        ax.set_title('Generator Loss', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Discriminator loss
        ax = axes[0, 1]
        for model_name, logs in tensorboard_logs.items():
            if 'train/loss_D' in logs:
                ax.plot(
                    logs['epochs'],
                    logs['train/loss_D'],
                    label=model_name,
                    color=self.colors.get(model_name, '#000000'),
                    linewidth=2
                )
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Discriminator Loss', fontsize=12)
        ax.set_title('Discriminator Loss', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Cycle consistency loss
        ax = axes[1, 0]
        for model_name, logs in tensorboard_logs.items():
            if 'train/loss_cycle' in logs:
                ax.plot(
                    logs['epochs'],
                    logs['train/loss_cycle'],
                    label=model_name,
                    color=self.colors.get(model_name, '#000000'),
                    linewidth=2
                )
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Cycle Loss', fontsize=12)
        ax.set_title('Cycle Consistency Loss', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes[1, 1]
        for model_name, logs in tensorboard_logs.items():
            if 'lr/generator' in logs:
                ax.plot(
                    logs['epochs'],
                    logs['lr/generator'],
                    label=model_name,
                    color=self.colors.get(model_name, '#000000'),
                    linewidth=2
                )
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_figure(fig, 'training_curves')
        plt.close()

    def generate_quantitative_comparison(self, results: Dict):
        """
        Generate bar chart comparing quantitative metrics.

        Args:
            results: Dictionary of evaluation results
        """
        logger.info("Generating quantitative comparison...")

        metrics = ['SSIM', 'PSNR', 'NMI', 'LPIPS']
        models = list(results.keys())

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Quantitative Comparison', fontsize=16, fontweight='bold')

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]

            # Prepare data
            means = []
            stds = []
            colors_list = []

            for model in models:
                if metric.lower() in results[model]['metrics']:
                    means.append(results[model]['metrics'][metric.lower()]['mean'])
                    stds.append(results[model]['metrics'][metric.lower()]['std'])
                    colors_list.append(self.colors.get(model, '#808080'))
                else:
                    means.append(0)
                    stds.append(0)
                    colors_list.append('#808080')

            # Create bar chart
            x_pos = np.arange(len(models))
            bars = ax.bar(
                x_pos,
                means,
                yerr=stds,
                capsize=5,
                color=colors_list,
                edgecolor='black',
                linewidth=1.5,
                alpha=0.8
            )

            # Customize
            ax.set_xlabel('Model', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + std,
                    f'{mean:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

        plt.tight_layout()
        self.save_figure(fig, 'quantitative_comparison')
        plt.close()

    def generate_qualitative_comparison(
        self,
        sample_images: Dict[str, np.ndarray],
        slice_idx: Optional[int] = None
    ):
        """
        Generate qualitative comparison showing example outputs.

        Args:
            sample_images: Dictionary mapping model names to image arrays
            slice_idx: Slice index for 3D volumes (if None, uses middle slice)
        """
        logger.info("Generating qualitative comparison...")

        n_models = len(sample_images)
        fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 5))

        if n_models == 1:
            axes = [axes]

        for idx, (model_name, image) in enumerate(sample_images.items()):
            # Extract 2D slice if 3D
            if image.ndim == 3:
                if slice_idx is None:
                    slice_idx = image.shape[2] // 2
                image_2d = image[:, :, slice_idx]
            else:
                image_2d = image

            # Display
            axes[idx].imshow(image_2d, cmap='gray', interpolation='bilinear')
            axes[idx].set_title(model_name, fontsize=14, fontweight='bold')
            axes[idx].axis('off')

        plt.tight_layout()
        self.save_figure(fig, 'qualitative_comparison')
        plt.close()

    def generate_ablation_heatmap(self, ablation_results: pd.DataFrame):
        """
        Generate heatmap for ablation study results.

        Args:
            ablation_results: DataFrame with ablation configurations and metrics
        """
        logger.info("Generating ablation study heatmap...")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data for heatmap
        # Assuming ablation_results has columns: config, ssim, psnr, nmi, etc.
        data = ablation_results.set_index('config')
        data = data.select_dtypes(include=[np.number])

        # Normalize to [0, 1] for each metric
        data_norm = (data - data.min()) / (data.max() - data.min())

        # Create heatmap
        sns.heatmap(
            data_norm.T,
            annot=data.T,
            fmt='.3f',
            cmap='RdYlGn',
            cbar_kws={'label': 'Normalized Score'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )

        ax.set_title('Ablation Study Results', fontsize=16, fontweight='bold')
        ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metric', fontsize=12, fontweight='bold')

        plt.tight_layout()
        self.save_figure(fig, 'ablation_heatmap')
        plt.close()

    def generate_statistical_significance(self, pvalues: Dict):
        """
        Generate statistical significance visualization.

        Args:
            pvalues: Dictionary of p-values for pairwise comparisons
        """
        logger.info("Generating statistical significance plot...")

        models = sorted(set([k[0] for k in pvalues.keys()] + [k[1] for k in pvalues.keys()]))
        n_models = len(models)

        # Create matrix of p-values
        pvalue_matrix = np.ones((n_models, n_models))

        for (model1, model2), pval in pvalues.items():
            idx1 = models.index(model1)
            idx2 = models.index(model2)
            pvalue_matrix[idx1, idx2] = pval
            pvalue_matrix[idx2, idx1] = pval

        # Create custom colormap
        colors_cmap = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('pvalue', colors_cmap, N=n_bins)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(
            pvalue_matrix,
            cmap=cmap,
            vmin=0,
            vmax=0.05,
            aspect='auto'
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('p-value', fontsize=12, fontweight='bold')

        # Set ticks
        ax.set_xticks(np.arange(n_models))
        ax.set_yticks(np.arange(n_models))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_yticklabels(models)

        # Add text annotations
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    text = ax.text(
                        j, i, f'{pvalue_matrix[i, j]:.4f}',
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=9
                    )

        ax.set_title('Statistical Significance (p-values)', fontsize=16, fontweight='bold')

        plt.tight_layout()
        self.save_figure(fig, 'statistical_significance')
        plt.close()

    def generate_attention_visualization(
        self,
        attention_maps: Dict[str, np.ndarray],
        input_image: np.ndarray
    ):
        """
        Visualize attention maps from self-attention layers.

        Args:
            attention_maps: Dictionary of attention maps
            input_image: Input image
        """
        logger.info("Generating attention visualization...")

        n_maps = len(attention_maps)
        fig, axes = plt.subplots(1, n_maps + 1, figsize=(4 * (n_maps + 1), 4))

        # Show input image
        axes[0].imshow(input_image, cmap='gray')
        axes[0].set_title('Input', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Show attention maps
        for idx, (layer_name, attn_map) in enumerate(attention_maps.items(), start=1):
            axes[idx].imshow(attn_map, cmap='hot', interpolation='bilinear')
            axes[idx].set_title(layer_name, fontsize=12, fontweight='bold')
            axes[idx].axis('off')

        plt.tight_layout()
        self.save_figure(fig, 'attention_visualization')
        plt.close()

    def generate_all_figures(self):
        """Generate all publication figures automatically."""
        logger.info("Generating all publication figures...")

        # Load results
        results_file = self.results_dir / 'all_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)

            # Convert to dictionary keyed by model name
            results_dict = {r['model_name']: r for r in results}

            # Generate quantitative comparison
            self.generate_quantitative_comparison(results_dict)

        # Load training logs (if available)
        # This is a placeholder - actual implementation depends on log format
        tensorboard_logs = {}
        self.generate_training_curves(tensorboard_logs)

        logger.info("Figure generation complete!")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate publication figures'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing evaluation results'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for figures'
    )

    parser.add_argument(
        '--format',
        nargs='+',
        default=['pdf', 'png'],
        choices=['pdf', 'png', 'svg', 'eps'],
        help='Output formats'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for raster formats'
    )

    args = parser.parse_args()

    # Initialize generator
    generator = PublicationFigureGenerator(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
        formats=args.format,
        dpi=args.dpi,
    )

    # Generate all figures
    generator.generate_all_figures()

    logger.info(f"\nAll figures saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
