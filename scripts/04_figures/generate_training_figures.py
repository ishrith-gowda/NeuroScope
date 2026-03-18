#!/usr/bin/env python3
"""
generate training progression figures for publication

creates publication-grade latex-rendered figures showing:
- loss curves (generator, discriminator, cycle, identity)
- validation metrics progression (ssim, psnr)
- learning rate schedule
- gradient norms

author: neuroscope research team
date: january 2026
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

# load training history
PROJECT_ROOT = Path(__file__).parent.parent.parent
HISTORY_PATH = PROJECT_ROOT / 'results/training/training_history.json'
OUTPUT_DIR = PROJECT_ROOT / 'figures/main'


def load_training_history():
    """load training history from json."""
    with open(HISTORY_PATH) as f:
        history = json.load(f)
    return history


def smooth_curve(values, window=5):
    """apply moving average smoothing."""
    if len(values) < window:
        return values
    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
    # pad to original length
    pad_left = (len(values) - len(smoothed)) // 2
    pad_right = len(values) - len(smoothed) - pad_left
    return np.pad(smoothed, (pad_left, pad_right), mode='edge')


def generate_loss_curves_figure(history):
    """
    figure 1: training loss curves

    four subplots showing loss progression over training epochs:
    - generator loss
    - discriminator loss
    - cycle consistency loss
    - identity loss
    """
    print("generating figure 1: training loss curves...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.93)

    epochs = np.arange(1, len(history['G_loss']) + 1)

    # distinct color palette: garnet, navy, emerald, amber
    color_gen = '#9B2335'      # garnet
    color_disc = '#1B3A5C'     # navy
    color_cycle = '#2E8B57'    # sea green
    color_identity = '#D4A017' # goldenrod

    # generator loss
    ax = axes[0, 0]
    g_loss = history['G_loss']
    g_loss_smooth = smooth_curve(g_loss)
    ax.plot(epochs, g_loss, alpha=0.2, color=color_gen, linewidth=0.7)
    ax.plot(epochs, g_loss_smooth, color=color_gen, linewidth=2.2, label='Generator Loss')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title(r'(a) Generator Loss', fontsize=14, pad=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
    ax.set_axisbelow(True)

    # discriminator loss
    ax = axes[0, 1]
    d_loss = history['D_loss']
    d_loss_smooth = smooth_curve(d_loss)
    ax.plot(epochs, d_loss, alpha=0.2, color=color_disc, linewidth=0.7)
    ax.plot(epochs, d_loss_smooth, color=color_disc, linewidth=2.2, label='Discriminator Loss')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title(r'(b) Discriminator Loss', fontsize=14, pad=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
    ax.set_axisbelow(True)

    # cycle consistency loss
    ax = axes[1, 0]
    cycle_loss = history['cycle_loss']
    cycle_loss_smooth = smooth_curve(cycle_loss)
    ax.plot(epochs, cycle_loss, alpha=0.2, color=color_cycle, linewidth=0.7)
    ax.plot(epochs, cycle_loss_smooth, color=color_cycle, linewidth=2.2, label='Cycle Consistency Loss')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title(r'(c) Cycle Consistency Loss', fontsize=14, pad=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
    ax.set_axisbelow(True)

    # identity loss
    ax = axes[1, 1]
    identity_loss = history['identity_loss']
    identity_loss_smooth = smooth_curve(identity_loss)
    ax.plot(epochs, identity_loss, alpha=0.2, color=color_identity, linewidth=0.7)
    ax.plot(epochs, identity_loss_smooth, color=color_identity, linewidth=2.2, label='Identity Loss')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title(r'(d) Identity Loss', fontsize=14, pad=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
    ax.set_axisbelow(True)

    fig.suptitle(r'\textbf{Training Loss Progression}', fontsize=16, y=0.98)
    save_figure(fig, 'fig01_training_losses', OUTPUT_DIR)
    plt.close()


def generate_validation_metrics_figure(history):
    """
    figure 2: validation metrics progression

    two subplots showing validation metrics:
    - ssim for a→b and b→a
    - psnr for a→b and b→a
    """
    print("generating figure 2: validation metrics...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(wspace=0.25, top=0.85)

    # plot all epochs
    val_epochs = np.arange(1, len(history['val_ssim_A2B']) + 1)

    # unique colors: raspberry, steel blue
    color_a2b = '#8B2252'      # raspberry
    color_b2a = '#4682B4'      # steel blue

    # ssim
    ax = axes[0]
    ssim_a2b = history['val_ssim_A2B']
    ssim_b2a = history['val_ssim_B2A']
    ssim_a2b_smooth = smooth_curve(ssim_a2b, window=5)
    ssim_b2a_smooth = smooth_curve(ssim_b2a, window=5)
    ax.plot(val_epochs, ssim_a2b, alpha=0.15, color=color_a2b, linewidth=0.6)
    ax.plot(val_epochs, ssim_a2b_smooth, linewidth=2.2, color=color_a2b, label=r'BraTS $\rightarrow$ UPenn')
    ax.plot(val_epochs, ssim_b2a, alpha=0.15, color=color_b2a, linewidth=0.6)
    ax.plot(val_epochs, ssim_b2a_smooth, linewidth=2.2, color=color_b2a, label=r'UPenn $\rightarrow$ BraTS')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('SSIM', fontsize=13)
    ax.set_title(r'(a) Structural Similarity Index (SSIM)', fontsize=14, pad=10)
    ax.set_ylim([0.9, 1.0])
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
    ax.set_axisbelow(True)

    # psnr
    ax = axes[1]
    psnr_a2b = history['val_psnr_A2B']
    psnr_b2a = history['val_psnr_B2A']
    psnr_a2b_smooth = smooth_curve(psnr_a2b, window=5)
    psnr_b2a_smooth = smooth_curve(psnr_b2a, window=5)
    ax.plot(val_epochs, psnr_a2b, alpha=0.15, color=color_a2b, linewidth=0.6)
    ax.plot(val_epochs, psnr_a2b_smooth, linewidth=2.2, color=color_a2b, label=r'BraTS $\rightarrow$ UPenn')
    ax.plot(val_epochs, psnr_b2a, alpha=0.15, color=color_b2a, linewidth=0.6)
    ax.plot(val_epochs, psnr_b2a_smooth, linewidth=2.2, color=color_b2a, label=r'UPenn $\rightarrow$ BraTS')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('PSNR (dB)', fontsize=13)
    ax.set_title(r'(b) Peak Signal-to-Noise Ratio (PSNR)', fontsize=14, pad=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
    ax.set_axisbelow(True)

    fig.suptitle(r'\textbf{Validation Metrics Progression}', fontsize=16, y=0.98)
    save_figure(fig, 'fig02_validation_metrics', OUTPUT_DIR)
    plt.close()


def generate_combined_losses_figure(history):
    """
    figure 3: combined loss components

    single plot showing all loss components stacked:
    - gan loss
    - cycle loss (weighted)
    - identity loss (weighted)
    - ssim loss (weighted)
    """
    print("generating figure 3: combined loss components...")

    fig, ax = plt.subplots(figsize=(16, 6))
    plt.subplots_adjust(top=0.85)

    epochs = np.arange(1, len(history['G_loss']) + 1)

    # unique colors: vermillion, jade, tangerine, cobalt
    color_gan = '#CC3311'       # vermillion
    color_cycle = '#228B22'     # forest green
    color_identity = '#E68A00'  # tangerine
    color_ssim = '#0047AB'      # cobalt

    # plot each component
    gan_loss = smooth_curve(history['gan_loss'])
    cycle_loss = smooth_curve(history['cycle_loss'])
    identity_loss = smooth_curve(history['identity_loss'])
    ssim_loss = smooth_curve(history['ssim_loss'])

    ax.plot(epochs, gan_loss, linewidth=2.2, label='GAN Loss', color=color_gan, marker='o', markersize=2, alpha=0.8)
    ax.plot(epochs, cycle_loss, linewidth=2.2, label=r'Cycle Loss ($\lambda=10$)', color=color_cycle, marker='s', markersize=2, alpha=0.8)
    ax.plot(epochs, identity_loss, linewidth=2.2, label=r'Identity Loss ($\lambda=5$)', color=color_identity, marker='^', markersize=2, alpha=0.8)
    ax.plot(epochs, ssim_loss, linewidth=2.2, label=r'SSIM Loss ($\lambda=1$)', color=color_ssim, marker='d', markersize=2, alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss Value', fontsize=13)
    ax.set_title(r'(a) Weighted Loss Components', fontsize=14, pad=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=11, frameon=True, fancybox=False, edgecolor='black')
    ax.set_axisbelow(True)

    fig.suptitle(r'\textbf{Loss Components During Training}', fontsize=16, y=0.98)
    save_figure(fig, 'fig03_loss_components', OUTPUT_DIR)
    plt.close()


def generate_learning_rate_figure(history):
    """
    figure 4: learning rate schedule

    shows how learning rate changed during training.
    """
    print("generating figure 4: learning rate schedule...")

    fig, ax = plt.subplots(figsize=(16, 6))
    plt.subplots_adjust(top=0.85)

    epochs = np.arange(1, len(history['learning_rate']) + 1)
    lr = history['learning_rate']

    color_lr = '#7D3C98'  # plum

    ax.plot(epochs, lr, linewidth=2.5, color=color_lr, marker='o', markersize=1.5, alpha=0.85)
    ax.fill_between(epochs, lr, alpha=0.15, color=color_lr)
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Learning Rate', fontsize=13)
    ax.set_title(r'(a) Cosine Annealing Schedule', fontsize=14, pad=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_axisbelow(True)

    fig.suptitle(r'\textbf{Learning Rate Schedule}', fontsize=16, y=0.98)
    save_figure(fig, 'fig04_learning_rate', OUTPUT_DIR)
    plt.close()


def generate_gradient_norms_figure(history):
    """
    figure 5: gradient norms

    shows gradient magnitudes for monitoring training stability.
    """
    print("generating figure 5: gradient norms...")

    fig, ax = plt.subplots(figsize=(16, 6))
    plt.subplots_adjust(top=0.85)

    epochs = np.arange(1, len(history['gradient_norm_G']) + 1)
    grad_g = smooth_curve(history['gradient_norm_G'])
    grad_d = smooth_curve(history['gradient_norm_D'])

    color_gen_grad = '#007BA7'   # cerulean
    color_disc_grad = '#CC5500'  # burnt orange

    ax.plot(epochs, grad_g, linewidth=2.2, label='Generator Gradient', color=color_gen_grad, marker='o', markersize=1.5, alpha=0.8)
    ax.plot(epochs, grad_d, linewidth=2.2, label='Discriminator Gradient', color=color_disc_grad, marker='s', markersize=1.5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Gradient L2 Norm', fontsize=13)
    ax.set_title(r'(a) Gradient Magnitudes', fontsize=14, pad=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
    ax.set_axisbelow(True)

    fig.suptitle(r'\textbf{Training Stability Monitoring}', fontsize=16, y=0.98)
    save_figure(fig, 'fig05_gradient_norms', OUTPUT_DIR)
    plt.close()


def main():
    """generate all training figures."""
    print("="*60)
    print("generating training progression figures")
    print("="*60)

    # load history
    history = load_training_history()
    print(f"loaded training history: {len(history['G_loss'])} epochs")

    # generate all figures (only those with available data)
    generate_loss_curves_figure(history)
    generate_validation_metrics_figure(history)

    # skip figures requiring missing data fields
    if len(history.get('gan_loss', [])) > 0:
        generate_combined_losses_figure(history)
    else:
        print("skipping figure 3: combined loss components (gan_loss not available)")

    generate_learning_rate_figure(history)

    if len(history.get('gradient_norm_G', [])) > 0:
        generate_gradient_norms_figure(history)
    else:
        print("skipping figure 5: gradient norms (gradient data not available)")

    print("\n" + "="*60)
    print("training figures generation complete!")
    print(f"figures saved to: {OUTPUT_DIR}/")
    print("="*60)


if __name__ == '__main__':
    main()
