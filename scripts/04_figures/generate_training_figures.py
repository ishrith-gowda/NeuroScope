#!/usr/bin/env python3
"""
Generate Training Progression Figures for Publication

Creates publication-grade LaTeX-rendered figures showing:
- Loss curves (Generator, Discriminator, Cycle, Identity)
- Validation metrics progression (SSIM, PSNR)
- Learning rate schedule
- Gradient norms

Author: NeuroScope Research Team
Date: January 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
from latex_figure_config import (
    FIGURE_SIZES, COLORS, save_figure
)

# Load training history
PROJECT_ROOT = Path(__file__).parent.parent.parent
HISTORY_PATH = PROJECT_ROOT / 'results/training/training_history.json'
OUTPUT_DIR = PROJECT_ROOT / 'figures/main'


def load_training_history():
    """Load training history from JSON."""
    with open(HISTORY_PATH) as f:
        history = json.load(f)
    return history


def smooth_curve(values, window=5):
    """Apply moving average smoothing."""
    if len(values) < window:
        return values
    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
    # Pad to original length
    pad_left = (len(values) - len(smoothed)) // 2
    pad_right = len(values) - len(smoothed) - pad_left
    return np.pad(smoothed, (pad_left, pad_right), mode='edge')


def generate_loss_curves_figure(history):
    """
    Figure 1: Training Loss Curves

    Four subplots showing loss progression over training epochs:
    - Generator loss
    - Discriminator loss
    - Cycle consistency loss
    - Identity loss
    """
    print("Generating Figure 1: Training Loss Curves...")

    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    fig.suptitle(r'\textbf{SA-CycleGAN-2.5D Training Loss Progression}', fontsize=12)

    epochs = np.arange(1, len(history['G_loss']) + 1)

    # Generator Loss
    ax = axes[0, 0]
    g_loss = history['G_loss']
    g_loss_smooth = smooth_curve(g_loss)
    ax.plot(epochs, g_loss, alpha=0.3, color=COLORS['primary'], linewidth=0.5)
    ax.plot(epochs, g_loss_smooth, color=COLORS['primary'], linewidth=2, label='Generator')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(r'\textbf{(a)} Generator Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Discriminator Loss
    ax = axes[0, 1]
    d_loss = history['D_loss']
    d_loss_smooth = smooth_curve(d_loss)
    ax.plot(epochs, d_loss, alpha=0.3, color=COLORS['secondary'], linewidth=0.5)
    ax.plot(epochs, d_loss_smooth, color=COLORS['secondary'], linewidth=2, label='Discriminator')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(r'\textbf{(b)} Discriminator Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Cycle Consistency Loss
    ax = axes[1, 0]
    cycle_loss = history['cycle_loss']
    cycle_loss_smooth = smooth_curve(cycle_loss)
    ax.plot(epochs, cycle_loss, alpha=0.3, color=COLORS['success'], linewidth=0.5)
    ax.plot(epochs, cycle_loss_smooth, color=COLORS['success'], linewidth=2, label='Cycle Consistency')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(r'\textbf{(c)} Cycle Consistency Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Identity Loss
    ax = axes[1, 1]
    identity_loss = history['identity_loss']
    identity_loss_smooth = smooth_curve(identity_loss)
    ax.plot(epochs, identity_loss, alpha=0.3, color=COLORS['info'], linewidth=0.5)
    ax.plot(epochs, identity_loss_smooth, color=COLORS['info'], linewidth=2, label='Identity')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(r'\textbf{(d)} Identity Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    save_figure(fig, 'fig01_training_losses', OUTPUT_DIR)
    plt.close()


def generate_validation_metrics_figure(history):
    """
    Figure 2: Validation Metrics Progression

    Two subplots showing validation metrics:
    - SSIM for A→B and B→A
    - PSNR for A→B and B→A
    """
    print("Generating Figure 2: Validation Metrics...")

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    fig.suptitle(r'\textbf{SA-CycleGAN-2.5D Validation Metrics}', fontsize=12)

    # Plot all epochs
    val_epochs = np.arange(1, len(history['val_ssim_A2B']) + 1)

    # SSIM
    ax = axes[0]
    ssim_a2b = history['val_ssim_A2B']
    ssim_b2a = history['val_ssim_B2A']
    ssim_a2b_smooth = smooth_curve(ssim_a2b, window=5)
    ssim_b2a_smooth = smooth_curve(ssim_b2a, window=5)
    ax.plot(val_epochs, ssim_a2b, alpha=0.2, color=COLORS['primary'], linewidth=0.5)
    ax.plot(val_epochs, ssim_a2b_smooth, linewidth=2, color=COLORS['primary'], label=r'$A \rightarrow B$')
    ax.plot(val_epochs, ssim_b2a, alpha=0.2, color=COLORS['secondary'], linewidth=0.5)
    ax.plot(val_epochs, ssim_b2a_smooth, linewidth=2, color=COLORS['secondary'], label=r'$B \rightarrow A$')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SSIM')
    ax.set_title(r'\textbf{(a)} Structural Similarity (SSIM)')
    ax.set_ylim([0.9, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend()

    # PSNR
    ax = axes[1]
    psnr_a2b = history['val_psnr_A2B']
    psnr_b2a = history['val_psnr_B2A']
    psnr_a2b_smooth = smooth_curve(psnr_a2b, window=5)
    psnr_b2a_smooth = smooth_curve(psnr_b2a, window=5)
    ax.plot(val_epochs, psnr_a2b, alpha=0.2, color=COLORS['primary'], linewidth=0.5)
    ax.plot(val_epochs, psnr_a2b_smooth, linewidth=2, color=COLORS['primary'], label=r'$A \rightarrow B$')
    ax.plot(val_epochs, psnr_b2a, alpha=0.2, color=COLORS['secondary'], linewidth=0.5)
    ax.plot(val_epochs, psnr_b2a_smooth, linewidth=2, color=COLORS['secondary'], label=r'$B \rightarrow A$')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(r'\textbf{(b)} Peak Signal-to-Noise Ratio (PSNR)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    save_figure(fig, 'fig02_validation_metrics', OUTPUT_DIR)
    plt.close()


def generate_combined_losses_figure(history):
    """
    Figure 3: Combined Loss Components

    Single plot showing all loss components stacked:
    - GAN loss
    - Cycle loss (weighted)
    - Identity loss (weighted)
    - SSIM loss (weighted)
    """
    print("Generating Figure 3: Combined Loss Components...")

    fig, ax = plt.subplots(figsize=(7, 4))

    epochs = np.arange(1, len(history['G_loss']) + 1)

    # Plot each component
    gan_loss = smooth_curve(history['gan_loss'])
    cycle_loss = smooth_curve(history['cycle_loss'])
    identity_loss = smooth_curve(history['identity_loss'])
    ssim_loss = smooth_curve(history['ssim_loss'])

    ax.plot(epochs, gan_loss, linewidth=2, label='GAN Loss', color=COLORS['primary'])
    ax.plot(epochs, cycle_loss, linewidth=2, label=r'Cycle Loss ($\lambda=10$)', color=COLORS['success'])
    ax.plot(epochs, identity_loss, linewidth=2, label=r'Identity Loss ($\lambda=5$)', color=COLORS['secondary'])
    ax.plot(epochs, ssim_loss, linewidth=2, label=r'SSIM Loss ($\lambda=1$)', color=COLORS['info'])

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Value')
    ax.set_title(r'\textbf{Loss Components During Training}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    save_figure(fig, 'fig03_loss_components', OUTPUT_DIR)
    plt.close()


def generate_learning_rate_figure(history):
    """
    Figure 4: Learning Rate Schedule

    Shows how learning rate changed during training.
    """
    print("Generating Figure 4: Learning Rate Schedule...")

    fig, ax = plt.subplots(figsize=(7, 3))

    epochs = np.arange(1, len(history['learning_rate']) + 1)
    lr = history['learning_rate']

    ax.plot(epochs, lr, linewidth=2, color=COLORS['danger'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title(r'\textbf{Learning Rate Schedule (Cosine Annealing)}')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    save_figure(fig, 'fig04_learning_rate', OUTPUT_DIR)
    plt.close()


def generate_gradient_norms_figure(history):
    """
    Figure 5: Gradient Norms

    Shows gradient magnitudes for monitoring training stability.
    """
    print("Generating Figure 5: Gradient Norms...")

    fig, ax = plt.subplots(figsize=(7, 3.5))

    epochs = np.arange(1, len(history['gradient_norm_G']) + 1)
    grad_g = smooth_curve(history['gradient_norm_G'])
    grad_d = smooth_curve(history['gradient_norm_D'])

    ax.plot(epochs, grad_g, linewidth=2, label='Generator', color=COLORS['primary'])
    ax.plot(epochs, grad_d, linewidth=2, label='Discriminator', color=COLORS['secondary'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient L2 Norm')
    ax.set_title(r'\textbf{Gradient Magnitudes (Training Stability Indicator)}')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    save_figure(fig, 'fig05_gradient_norms', OUTPUT_DIR)
    plt.close()


def main():
    """Generate all training figures."""
    print("="*60)
    print("Generating Training Progression Figures")
    print("="*60)

    # Load history
    history = load_training_history()
    print(f"Loaded training history: {len(history['G_loss'])} epochs")

    # Generate all figures (only those with available data)
    generate_loss_curves_figure(history)
    generate_validation_metrics_figure(history)

    # Skip figures requiring missing data fields
    if len(history.get('gan_loss', [])) > 0:
        generate_combined_losses_figure(history)
    else:
        print("Skipping Figure 3: Combined Loss Components (gan_loss not available)")

    generate_learning_rate_figure(history)

    if len(history.get('gradient_norm_G', [])) > 0:
        generate_gradient_norms_figure(history)
    else:
        print("Skipping Figure 5: Gradient Norms (gradient data not available)")

    print("\n" + "="*60)
    print("Training figures generation complete!")
    print(f"Figures saved to: {OUTPUT_DIR}/")
    print("="*60)


if __name__ == '__main__':
    main()
