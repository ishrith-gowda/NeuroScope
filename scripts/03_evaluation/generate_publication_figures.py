#!/usr/bin/env python3
"""
publication-quality figure generation for sa-cyclegan-2.5d

generates all figures needed for top-tier venue submission:
- qualitative comparison grids
- training curves and convergence plots
- attention visualization overlays
- quantitative results tables
- per-modality performance breakdowns
- statistical significance visualizations
- failure case analysis

all figures are publication-ready (high-dpi, proper formatting, latex-compatible).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from PIL import Image

# setup plotting style for publications
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'font.family': 'serif',
    'text.usetex': False,  # set to True if latex is installed
})

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PublicationFigureGenerator:
    """generates publication-quality figures."""

    def __init__(self, output_dir: Path, dpi: int = 300):
        """
        initialize generator.

        args:
            output_dir: directory to save figures
            dpi: resolution for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def plot_training_curves(
        self,
        history_file: Path,
        save_name: str = 'training_curves'
    ):
        """
        plot training and validation curves.

        args:
            history_file: path to training history json
            save_name: name for saved figure
        """
        logger.info("generating training curves...")

        # load history
        with open(history_file) as f:
            history = json.load(f)

        # extract metrics
        epochs = list(range(1, len(history.get('train_loss_G', [])) + 1))

        # create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Progression', fontsize=14, fontweight='bold')

        # plot 1: generator loss
        ax = axes[0, 0]
        if 'train_loss_G' in history:
            ax.plot(epochs, history['train_loss_G'], label='generator loss', linewidth=2)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title('generator loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # plot 2: discriminator loss
        ax = axes[0, 1]
        if 'train_loss_D' in history:
            ax.plot(epochs, history['train_loss_D'], label='discriminator loss',
                   linewidth=2, color='orange')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title('discriminator loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # plot 3: ssim
        ax = axes[1, 0]
        if 'val_ssim_A2B' in history:
            ax.plot(epochs, history['val_ssim_A2B'], label='A→B', linewidth=2)
        if 'val_ssim_B2A' in history:
            ax.plot(epochs, history['val_ssim_B2A'], label='B→A', linewidth=2)
        ax.set_xlabel('epoch')
        ax.set_ylabel('SSIM')
        ax.set_title('validation SSIM')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # plot 4: psnr
        ax = axes[1, 1]
        if 'val_psnr_A2B' in history:
            ax.plot(epochs, history['val_psnr_A2B'], label='A→B', linewidth=2)
        if 'val_psnr_B2A' in history:
            ax.plot(epochs, history['val_psnr_B2A'], label='B→A', linewidth=2)
        ax.set_xlabel('epoch')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('validation PSNR')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # save in multiple formats
        for ext in ['png', 'pdf']:
            save_path = self.output_dir / f'{save_name}.{ext}'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"saved to {save_path}")

        plt.close()

    def plot_qualitative_comparison(
        self,
        samples: List[Dict[str, np.ndarray]],
        model_names: List[str],
        save_name: str = 'qualitative_comparison'
    ):
        """
        create qualitative comparison grid.

        args:
            samples: list of sample dicts with keys: 'input', 'ground_truth',
                    'model1', 'model2', etc.
            model_names: list of model names
            save_name: name for saved figure
        """
        logger.info("generating qualitative comparison...")

        n_samples = len(samples)
        n_cols = 2 + len(model_names)  # input + gt + models

        fig = plt.figure(figsize=(3 * n_cols, 3 * n_samples))
        gs = gridspec.GridSpec(n_samples, n_cols, figure=fig,
                              wspace=0.05, hspace=0.15)

        # column titles
        titles = ['input', 'ground truth'] + model_names

        for row_idx, sample in enumerate(samples):
            # input
            ax = fig.add_subplot(gs[row_idx, 0])
            ax.imshow(sample['input'], cmap='gray')
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(titles[0], fontweight='bold')

            # ground truth
            ax = fig.add_subplot(gs[row_idx, 1])
            ax.imshow(sample['ground_truth'], cmap='gray')
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(titles[1], fontweight='bold')

            # model outputs
            for col_idx, model_name in enumerate(model_names, start=2):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.imshow(sample[model_name], cmap='gray')
                ax.axis('off')
                if row_idx == 0:
                    ax.set_title(titles[col_idx], fontweight='bold')

        plt.suptitle('qualitative comparison', fontsize=14, fontweight='bold', y=0.995)

        # save
        for ext in ['png', 'pdf']:
            save_path = self.output_dir / f'{save_name}.{ext}'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"saved to {save_path}")

        plt.close()

    def plot_quantitative_results_table(
        self,
        results: Dict[str, Dict],
        save_name: str = 'quantitative_results'
    ):
        """
        create publication-ready quantitative results table.

        args:
            results: dict mapping model names to evaluation results
            save_name: name for saved figure/table
        """
        logger.info("generating quantitative results table...")

        # extract metrics
        metrics = ['ssim', 'psnr', 'mae', 'lpips', 'fid']
        rows = []

        for model_name, model_results in results.items():
            row = {'model': model_name}

            # a->b metrics
            for metric in metrics:
                if metric == 'fid':
                    if 'a2b' in model_results and 'fid' in model_results['a2b']:
                        value = model_results['a2b']['fid'].get('value', -1)
                        row[f'{metric}_a2b'] = f'{value:.3f}' if value != -1 else 'n/a'
                elif metric in model_results.get('a2b', {}):
                    mean = model_results['a2b'][metric].get('mean', 0)
                    std = model_results['a2b'][metric].get('std', 0)
                    row[f'{metric}_a2b'] = f'{mean:.4f}±{std:.4f}'

            # b->a metrics
            for metric in metrics:
                if metric == 'fid':
                    if 'b2a' in model_results and 'fid' in model_results['b2a']:
                        value = model_results['b2a']['fid'].get('value', -1)
                        row[f'{metric}_b2a'] = f'{value:.3f}' if value != -1 else 'n/a'
                elif metric in model_results.get('b2a', {}):
                    mean = model_results['b2a'][metric].get('mean', 0)
                    std = model_results['b2a'][metric].get('std', 0)
                    row[f'{metric}_b2a'] = f'{mean:.4f}±{std:.4f}'

            rows.append(row)

        df = pd.DataFrame(rows)

        # save as csv
        csv_path = self.output_dir / f'{save_name}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"saved csv to {csv_path}")

        # save as latex
        latex_path = self.output_dir / f'{save_name}.tex'
        latex_str = df.to_latex(index=False, escape=False, column_format='l' + 'c' * (len(df.columns) - 1))
        latex_path.write_text(latex_str)
        logger.info(f"saved latex to {latex_path}")

        # create visual table
        fig, ax = plt.subplots(figsize=(14, len(df) * 0.6 + 1))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        # style header row
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # alternating row colors
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        plt.title('quantitative results', fontsize=12, fontweight='bold', pad=20)

        # save
        for ext in ['png', 'pdf']:
            save_path = self.output_dir / f'{save_name}_table.{ext}'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"saved to {save_path}")

        plt.close()

    def plot_metric_distributions(
        self,
        results: Dict[str, Dict],
        save_name: str = 'metric_distributions'
    ):
        """
        plot distribution of metrics across test set.

        args:
            results: dict mapping model names to evaluation results
            save_name: name for saved figure
        """
        logger.info("generating metric distribution plots...")

        metrics = ['ssim', 'psnr', 'mae', 'lpips']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('metric distributions on test set', fontsize=14, fontweight='bold')

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]

            for model_name, model_results in results.items():
                if metric in model_results.get('a2b', {}):
                    values = model_results['a2b'][metric].get('values', [])
                    if len(values) > 0:
                        ax.hist(values, bins=50, alpha=0.5, label=model_name, density=True)

            ax.set_xlabel(metric.upper())
            ax.set_ylabel('density')
            ax.set_title(f'{metric.upper()} distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # save
        for ext in ['png', 'pdf']:
            save_path = self.output_dir / f'{save_name}.{ext}'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"saved to {save_path}")

        plt.close()

    def plot_box_comparison(
        self,
        results: Dict[str, Dict],
        metric: str = 'ssim',
        save_name: str = 'box_comparison'
    ):
        """
        create box plot comparing models on a metric.

        args:
            results: dict mapping model names to evaluation results
            metric: metric to compare
            save_name: name for saved figure
        """
        logger.info(f"generating box plot for {metric}...")

        # prepare data
        data = []
        labels = []

        for model_name, model_results in results.items():
            if metric in model_results.get('a2b', {}):
                values = model_results['a2b'][metric].get('values', [])
                if len(values) > 0:
                    data.append(values)
                    labels.append(model_name)

        if len(data) == 0:
            logger.warning(f"no data available for {metric}")
            return

        # create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True)

        # style boxes
        colors = sns.color_palette("husl", len(data))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} comparison across models', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        # save
        for ext in ['png', 'pdf']:
            save_path = self.output_dir / f'{save_name}_{metric}.{ext}'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"saved to {save_path}")

        plt.close()


def main():
    """main execution function."""
    parser = argparse.ArgumentParser(
        description='generate publication-quality figures'
    )

    parser.add_argument(
        '--history',
        type=str,
        help='path to training history json'
    )

    parser.add_argument(
        '--results',
        type=str,
        nargs='+',
        help='paths to evaluation results json files'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='output directory for figures'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='resolution for saved figures'
    )

    args = parser.parse_args()

    # create generator
    generator = PublicationFigureGenerator(
        output_dir=Path(args.output_dir),
        dpi=args.dpi
    )

    # plot training curves if history provided
    if args.history:
        generator.plot_training_curves(Path(args.history))

    # plot quantitative results if results provided
    if args.results:
        results_dict = {}

        for result_file in args.results:
            with open(result_file) as f:
                result = json.load(f)
                model_name = Path(result_file).stem
                results_dict[model_name] = result

        generator.plot_quantitative_results_table(results_dict)
        generator.plot_metric_distributions(results_dict)

        # box plots for key metrics
        for metric in ['ssim', 'psnr', 'mae', 'lpips']:
            generator.plot_box_comparison(results_dict, metric)

    logger.info("figure generation complete!")


if __name__ == '__main__':
    main()
