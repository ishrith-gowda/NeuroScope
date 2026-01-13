#!/usr/bin/env python3
"""
Generate Dataset and Preprocessing Figures for Publication

Creates comprehensive visualizations of:
- Dataset statistics and distributions
- Train/val/test splits
- Domain characteristics
- Preprocessing pipeline
- Data flow diagrams

Author: NeuroScope Research Team
Date: January 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from latex_figure_config import (
    FIGURE_SIZES, COLORS, save_figure
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
EVAL_PATH = PROJECT_ROOT / 'results/evaluation/evaluation_results.json'
OUTPUT_DIR = PROJECT_ROOT / 'figures/main'


def generate_dataset_statistics_figure():
    """
    Figure: Dataset Statistics and Splits

    Shows sample counts, domain distributions, and train/val/test splits.
    """
    print("Generating: Dataset Statistics...")

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle(r'\textbf{Dataset Statistics and Experimental Setup}', y=1.02)

    # Dataset sizes
    ax = axes[0]
    datasets = ['BraTS\n(Domain A)', 'UPenn-GBM\n(Domain B)']
    samples = [8184, 52638]
    colors_list = [COLORS['primary'], COLORS['secondary']]

    bars = ax.bar(datasets, samples, color=colors_list, alpha=0.7, width=0.6)
    ax.set_ylabel('Number of 2.5D Slices')
    ax.set_title(r'(a) Dataset Sizes')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, samples):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}',
                ha='center', va='bottom', fontsize=9)

    # Add imbalance ratio annotation
    ratio = samples[1] / samples[0]
    ax.text(0.5, 0.95, f'Imbalance ratio: {ratio:.1f}:1',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            fontsize=8)

    # Train/Val/Test splits
    ax = axes[1]
    splits = ['Train', 'Val', 'Test']
    split_samples = [42110, 5263, 5265]
    split_colors = [COLORS['success'], COLORS['warning'], COLORS['danger']]

    bars = ax.bar(splits, split_samples, color=split_colors, alpha=0.7, width=0.6)
    ax.set_ylabel('Number of Samples')
    ax.set_title(r'(b) Data Splits')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels and percentages
    total = sum(split_samples)
    for bar, val in zip(bars, split_samples):
        height = bar.get_height()
        pct = (val / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}\n({pct:.1f}\%)',
                ha='center', va='bottom', fontsize=8)

    # Epoch length comparison
    ax = axes[2]
    components = ['Smaller\nDomain', 'Larger\nDomain\n(Epoch Length)', 'Training\nBatches']
    values = [8184, 52638, 5264]  # batch_size=8, so 52638/8 ≈ 6580, but actual is 5264 after splits
    bar_colors = [COLORS['primary'], COLORS['secondary'], COLORS['info']]

    bars = ax.bar(components, values, color=bar_colors, alpha=0.7, width=0.6)
    ax.set_ylabel('Count')
    ax.set_title(r'(c) Training Configuration')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}',
                ha='center', va='bottom', fontsize=8)

    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure(fig, 'fig08_dataset_statistics', OUTPUT_DIR)
    plt.close()


def generate_preprocessing_pipeline_figure():
    """
    Figure: Preprocessing Pipeline Diagram

    Shows the data flow from raw NIfTI to 2.5D slice triplets.
    """
    print("Generating: Preprocessing Pipeline...")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    fig.suptitle(r'\textbf{Preprocessing Pipeline: Raw MRI to 2.5D Slice Triplets}',
                 y=0.98)

    # Define pipeline stages
    stages = [
        {'name': 'Raw NIfTI\nVolumes', 'y': 7, 'color': COLORS['primary']},
        {'name': 'Skull\nStripping', 'y': 6, 'color': COLORS['secondary']},
        {'name': 'Normalization\n(0-1)', 'y': 5, 'color': COLORS['success']},
        {'name': 'Intensity\nClipping', 'y': 4, 'color': COLORS['warning']},
        {'name': '2.5D Slice\nExtraction', 'y': 3, 'color': COLORS['info']},
        {'name': 'Triplet\nFormation', 'y': 2, 'color': COLORS['danger']},
        {'name': 'Output:\n[B, 12, H, W]', 'y': 1, 'color': COLORS['gray']},
    ]

    # Draw pipeline boxes and arrows
    for i, stage in enumerate(stages):
        # Box
        rect = mpatches.FancyBboxPatch((1, stage['y']-0.3), 3, 0.6,
                                       boxstyle="round,pad=0.1",
                                       facecolor=stage['color'],
                                       edgecolor='black',
                                       alpha=0.6,
                                       linewidth=1.5)
        ax.add_patch(rect)

        # Text
        ax.text(2.5, stage['y'], stage['name'],
                ha='center', va='center',
                fontsize=9, fontweight='bold')

        # Arrow to next stage
        if i < len(stages) - 1:
            ax.annotate('', xy=(2.5, stage['y']-0.4), xytext=(2.5, stage['y']-0.7),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Add details on the right
    details = [
        (7, r'4 modalities: T1, T1ce, T2, FLAIR'),
        (6, r'HD-BET for robust skull stripping'),
        (5, r'Per-volume min-max normalization'),
        (4, r'1st-99th percentile clipping'),
        (3, r'Extract adjacent axial slices'),
        (2, r'Stack 3 slices $\times$ 4 modalities'),
        (1, r'Batch $\times$ 12 channels $\times$ 128 $\times$ 128'),
    ]

    for y, text in details:
        ax.text(5.5, y, text, ha='left', va='center', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.3))

    # Add data shape evolution
    ax.text(8.5, 7, r'$[H, W, D, 4]$', ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    ax.text(8.5, 3, r'$[D-2, 4, 3, H, W]$', ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    ax.text(8.5, 1, r'$[B, 12, H, W]$', ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

    plt.tight_layout()
    save_figure(fig, 'fig09_preprocessing_pipeline', OUTPUT_DIR)
    plt.close()


def generate_25d_processing_figure():
    """
    Figure: 2.5D Processing Illustration

    Shows how 3 adjacent slices are processed to generate center slice.
    """
    print("Generating: 2.5D Processing Illustration...")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    fig.suptitle(r'\textbf{2.5D Processing: Context-Aware Slice Translation}',
                 y=0.96)

    # Input: 3 slices
    input_y = 4
    slice_positions = [1, 2, 3]
    slice_labels = [r'Slice $i-1$', r'Slice $i$', r'Slice $i+1$']

    for pos, label in zip(slice_positions, slice_labels):
        rect = mpatches.Rectangle((pos, input_y-0.4), 0.8, 2,
                                  facecolor=COLORS['primary'],
                                  edgecolor='black',
                                  alpha=0.5,
                                  linewidth=2)
        ax.add_patch(rect)
        ax.text(pos+0.4, input_y-0.7, label, ha='center', fontsize=8)

    # Stack annotation
    ax.text(2, input_y+1.8, r'Input: 3 slices $\times$ 4 modalities = 12 channels',
           ha='center', fontsize=9,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.4))

    # Arrow to generator
    ax.annotate('', xy=(5, input_y+0.5), xytext=(4, input_y+0.5),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    # Generator box
    gen_rect = mpatches.FancyBboxPatch((5, input_y-0.2), 2, 1.4,
                                       boxstyle="round,pad=0.1",
                                       facecolor=COLORS['success'],
                                       edgecolor='black',
                                       alpha=0.7,
                                       linewidth=2)
    ax.add_patch(gen_rect)
    ax.text(6, input_y+0.5, r'\textbf{Generator}', ha='center', va='center',
           fontsize=10, fontweight='bold')
    ax.text(6, input_y+0.1, r'(with attention)', ha='center', va='center',
           fontsize=8)

    # Arrow to output
    ax.annotate('', xy=(8, input_y+0.5), xytext=(7.2, input_y+0.5),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    # Output: center slice
    output_x = 8.5
    out_rect = mpatches.Rectangle((output_x, input_y-0.1), 1, 1.2,
                                  facecolor=COLORS['secondary'],
                                  edgecolor='black',
                                  alpha=0.6,
                                  linewidth=2)
    ax.add_patch(out_rect)
    ax.text(output_x+0.5, input_y-0.5, r'Translated Slice $i$', ha='center', fontsize=8)

    # Output annotation
    ax.text(9, input_y+1.3, r'Output: 1 slice $\times$ 4 modalities',
           ha='center', fontsize=9,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.4))

    # Add advantage boxes at bottom
    advantages = [
        r'\textbf{Advantages:}',
        r'$\bullet$ 3D context from adjacent slices',
        r'$\bullet$ Preserves anatomical consistency',
        r'$\bullet$ More efficient than full 3D',
        r'$\bullet$ Better than pure 2D slice-by-slice',
    ]

    y_pos = 2
    for adv in advantages:
        ax.text(2, y_pos, adv, ha='left', fontsize=8)
        y_pos -= 0.4

    # Add dimensions
    ax.text(10.5, input_y+0.5, r'Input: $[B, 12, H, W]$', ha='left', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    ax.text(10.5, input_y-0.2, r'Output: $[B, 4, H, W]$', ha='left', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    plt.tight_layout()
    save_figure(fig, 'fig10_25d_processing', OUTPUT_DIR)
    plt.close()


def generate_training_overview_figure():
    """
    Figure: Training Overview and Configuration

    Shows complete training setup: data flow, model, losses, optimization.
    """
    print("Generating: Training Overview...")

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    fig.suptitle(r'\textbf{SA-CycleGAN-2.5D Training Configuration}',
                 y=0.98)

    # Hyperparameters box
    hp_text = [
        r'\textbf{Hyperparameters:}',
        r'Epochs: 100',
        r'Batch size: 8',
        r'Image size: $128 \times 128$',
        r'Learning rate: $2 \times 10^{-4}$',
        r'Optimizer: Adam ($\beta_1=0.5$, $\beta_2=0.999$)',
        r'LR schedule: Cosine annealing',
        r'Warmup epochs: 5',
    ]
    y_pos = 9
    for text in hp_text:
        ax.text(0.5, y_pos, text, ha='left', fontsize=7.5)
        y_pos -= 0.4

    # Loss weights box
    loss_text = [
        r'\textbf{Loss Weights:}',
        r'$\lambda_{\text{cycle}} = 10.0$',
        r'$\lambda_{\text{identity}} = 5.0$',
        r'$\lambda_{\text{SSIM}} = 1.0$',
        r'$\lambda_{\text{gradient}} = 1.0$',
    ]
    y_pos = 9
    for text in loss_text:
        ax.text(5.5, y_pos, text, ha='left', fontsize=7.5)
        y_pos -= 0.5

    # Architecture specs
    arch_text = [
        r'\textbf{Model Architecture:}',
        r'Generator: ResNet-based (9 blocks)',
        r'Base filters: 64',
        r'Self-attention: in bottleneck',
        r'CBAM: after each ResBlock',
        r'Discriminator: Multi-scale PatchGAN',
        r'Total parameters: 35.1M',
    ]
    y_pos = 5.5
    for text in arch_text:
        ax.text(0.5, y_pos, text, ha='left', fontsize=7.5)
        y_pos -= 0.4

    # Training stats
    stats_text = [
        r'\textbf{Training Statistics:}',
        r'Training samples: 42,110',
        r'Validation samples: 5,263',
        r'Test samples: 5,265',
        r'Batches per epoch: 5,264',
        r'Total iterations: 526,400',
        r'Training time: $\sim$85 hours',
        r'Hardware: RTX 6000 (24GB)',
    ]
    y_pos = 5.5
    for text in stats_text:
        ax.text(5.5, y_pos, text, ha='left', fontsize=7.5)
        y_pos -= 0.4

    # Data augmentation
    aug_text = [
        r'\textbf{Data Augmentation:}',
        r'Random horizontal flip',
        r'Random rotation ($\pm 10°$)',
        r'Gaussian noise',
        r'Intensity jittering',
    ]
    y_pos = 2.5
    for text in aug_text:
        ax.text(0.5, y_pos, text, ha='left', fontsize=7.5)
        y_pos -= 0.4

    # Regularization
    reg_text = [
        r'\textbf{Regularization:}',
        r'Gradient clipping: L2 norm $\leq 1.0$',
        r'Instance normalization',
        r'Spectral normalization (discriminator)',
        r'Replay buffer size: 50',
    ]
    y_pos = 2.5
    for text in reg_text:
        ax.text(5.5, y_pos, text, ha='left', fontsize=7.5)
        y_pos -= 0.4

    plt.tight_layout()
    save_figure(fig, 'fig11_training_overview', OUTPUT_DIR)
    plt.close()


def main():
    """Generate all dataset and preprocessing figures."""
    print("="*60)
    print("Generating Dataset and Preprocessing Figures")
    print("="*60)

    generate_dataset_statistics_figure()
    generate_preprocessing_pipeline_figure()
    generate_25d_processing_figure()
    generate_training_overview_figure()

    print("\n" + "="*60)
    print("Dataset figures generation complete!")
    print(f"Figures saved to: {OUTPUT_DIR}/")
    print("="*60)


if __name__ == '__main__':
    main()
