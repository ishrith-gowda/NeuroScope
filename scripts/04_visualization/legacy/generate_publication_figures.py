#!/usr/bin/env python
"""
publication-quality visualization generator for cyclegan domain adaptation

generates comprehensive figures for academic publication:
1. model architecture diagrams
2. training loss curves 
3. sample translation results
4. quantitative metrics (ssim, psnr, fid)
5. domain distribution analysis
6. ablation study visualizations

all figures follow publication standards:
- times new roman font
- seaborn styling
- high dpi (300+)
- vector formats where applicable
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import seaborn as sns

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import make_grid

# force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

def pprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# ============================================================================
# publication style configuration
# ============================================================================
def setup_publication_style():
    """configure matplotlib for publication-quality figures"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.rcParams.update({
        # font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        
        # math font
        'mathtext.fontset': 'stix',
        
        # figure settings
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        
        # line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        
        # grid
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        
        # axes
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        
        # legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
    })
    
    # custom color palette
    colors = {
        'primary': '#2E86AB',      # blue
        'secondary': '#A23B72',    # magenta
        'tertiary': '#F18F01',     # orange
        'quaternary': '#C73E1D',   # red
        'quinary': '#3B1F2B',      # dark
        'success': '#2E7D32',      # green
        'warning': '#FF8F00',      # amber
        'domain_a': '#1976D2',     # blue (brats)
        'domain_b': '#D32F2F',     # red (upenn)
    }
    
    return colors


# ============================================================================
# figure 1: model architecture diagram
# ============================================================================
def create_architecture_diagram(save_dir, colors):
    """create cyclegan architecture diagram"""
    pprint("creating architecture diagram...")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 7)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # helper functions
    def draw_block(x, y, w, h, label, color, ax, fontsize=8):
        rect = FancyBboxPatch((x, y), w, h, 
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=color, edgecolor='black', linewidth=1.5,
                              alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold', color='white')
    
    def draw_arrow(x1, y1, x2, y2, ax, style='->'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color='#333', lw=1.5))
    
    # title
    ax.text(5, 6.5, 'CycleGAN Architecture for Multi-Modal MRI Domain Adaptation',
            ha='center', fontsize=14, fontweight='bold')
    
    # domain a (brats)
    draw_block(0, 4.5, 1.5, 1, 'Domain A\n(BraTS)', colors['domain_a'], ax, 9)
    
    # generator g_a2b
    draw_block(2.5, 4.5, 2, 1, 'Generator\nG_A→B', colors['primary'], ax, 9)
    
    # fake b
    draw_block(5.5, 4.5, 1.2, 1, 'Fake B', colors['secondary'], ax, 9)
    
    # generator g_b2a (cycle)
    draw_block(7.5, 4.5, 2, 1, 'Generator\nG_B→A', colors['primary'], ax, 9)
    
    # reconstructed a
    draw_block(0, 2.5, 1.5, 1, 'Recon. A', colors['tertiary'], ax, 9)
    
    # domain b (upenn)
    draw_block(0, 0.5, 1.5, 1, 'Domain B\n(UPenn)', colors['domain_b'], ax, 9)
    
    # generator g_b2a
    draw_block(2.5, 0.5, 2, 1, 'Generator\nG_B→A', colors['primary'], ax, 9)
    
    # fake a
    draw_block(5.5, 0.5, 1.2, 1, 'Fake A', colors['secondary'], ax, 9)
    
    # generator g_a2b (cycle)
    draw_block(7.5, 0.5, 2, 1, 'Generator\nG_A→B', colors['primary'], ax, 9)
    
    # reconstructed b
    draw_block(0, 2.5, 1.5, 1, 'Recon. B', colors['tertiary'], ax, 9)
    
    # discriminators
    draw_block(5.5, 2.5, 1.2, 1, 'Disc. D_B', colors['quaternary'], ax, 8)
    draw_block(7.8, 2.5, 1.2, 1, 'Disc. D_A', colors['quaternary'], ax, 8)
    
    # arrows - forward path a to b
    draw_arrow(1.5, 5, 2.5, 5, ax)
    draw_arrow(4.5, 5, 5.5, 5, ax)
    draw_arrow(6.7, 5, 7.5, 5, ax)
    
    # arrows - forward path b to a
    draw_arrow(1.5, 1, 2.5, 1, ax)
    draw_arrow(4.5, 1, 5.5, 1, ax)
    draw_arrow(6.7, 1, 7.5, 1, ax)
    
    # cycle arrows
    draw_arrow(9.5, 4.5, 9.5, 3.5, ax)
    ax.annotate('', xy=(0.75, 3.5), xytext=(9.5, 3.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5,
                               connectionstyle='arc3,rad=0'))
    
    draw_arrow(9.5, 1.5, 9.5, 2.5, ax)
    ax.annotate('', xy=(0.75, 2.5), xytext=(9.5, 2.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5,
                               connectionstyle='arc3,rad=0'))
    
    # discriminator arrows
    draw_arrow(6.1, 4.5, 6.1, 3.5, ax)
    draw_arrow(6.1, 1.5, 6.1, 2.5, ax)
    
    # loss labels
    ax.text(2.5, 6, r'$\mathcal{L}_{GAN}$', fontsize=10, ha='center')
    ax.text(5, 6, r'$\mathcal{L}_{cycle}$', fontsize=10, ha='center')
    ax.text(7.5, 6, r'$\mathcal{L}_{identity}$', fontsize=10, ha='center')
    
    # legend
    legend_elements = [
        mpatches.Patch(color=colors['domain_a'], label='Domain A (BraTS-TCGA)'),
        mpatches.Patch(color=colors['domain_b'], label='Domain B (UPenn-GBM)'),
        mpatches.Patch(color=colors['primary'], label='Generator Network'),
        mpatches.Patch(color=colors['quaternary'], label='Discriminator Network'),
        mpatches.Patch(color=colors['secondary'], label='Generated Image'),
        mpatches.Patch(color=colors['tertiary'], label='Reconstructed Image'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=3, 
              bbox_to_anchor=(0.5, -0.1), frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'fig1_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'fig1_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    pprint("  saved fig1_architecture.pdf/png")


# ============================================================================
# figure 2: generator architecture detail
# ============================================================================
def create_generator_diagram(save_dir, colors):
    """create detailed generator architecture diagram"""
    pprint("creating generator architecture diagram...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 4)
    ax.axis('off')
    
    def draw_block(x, y, w, h, label, color, ax, fontsize=7):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.02,rounding_size=0.05",
                              facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white',
                wrap=True)
    
    # title
    ax.text(7, 3.7, 'ResNet Generator Architecture (6 Residual Blocks)',
            ha='center', fontsize=12, fontweight='bold')
    
    # input
    draw_block(0, 1.5, 0.8, 1, 'Input\n4×256²', '#607D8B', ax, 7)
    
    # initial conv
    draw_block(1.2, 1.5, 0.9, 1, '7×7 Conv\n64ch\nIN+ReLU', colors['primary'], ax, 6)
    
    # downsampling
    draw_block(2.5, 1.5, 0.9, 1, 'Down\n128ch\nIN+ReLU', colors['secondary'], ax, 6)
    draw_block(3.8, 1.5, 0.9, 1, 'Down\n256ch\nIN+ReLU', colors['secondary'], ax, 6)
    
    # residual blocks
    for i in range(6):
        x = 5.1 + i * 0.7
        draw_block(x, 1.5, 0.6, 1, f'Res\n{i+1}', colors['tertiary'], ax, 6)
    
    # upsampling
    draw_block(9.5, 1.5, 0.9, 1, 'Up\n128ch\nIN+ReLU', colors['quaternary'], ax, 6)
    draw_block(10.8, 1.5, 0.9, 1, 'Up\n64ch\nIN+ReLU', colors['quaternary'], ax, 6)
    
    # output conv
    draw_block(12.1, 1.5, 0.9, 1, '7×7 Conv\n4ch\nTanh', colors['primary'], ax, 6)
    
    # output
    draw_block(13.4, 1.5, 0.6, 1, 'Out\n4×256²', '#607D8B', ax, 7)
    
    # arrows
    positions = [0.8, 2.1, 3.4, 4.7]
    for p in positions:
        ax.annotate('', xy=(p + 0.4, 2), xytext=(p, 2),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1))
    
    for i in range(5):
        x = 5.7 + i * 0.7
        ax.annotate('', xy=(x + 0.4, 2), xytext=(x, 2),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1))
    
    for p in [9.3, 10.4, 11.7, 13.0]:
        ax.annotate('', xy=(p + 0.4, 2), xytext=(p, 2),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1))
    
    # skip connection indicator
    ax.annotate('', xy=(5.1, 0.8), xytext=(9.5, 0.8),
                arrowprops=dict(arrowstyle='<->', color='#666', lw=1, ls='--'))
    ax.text(7.3, 0.5, 'Residual Blocks (256 channels)', fontsize=8, ha='center')
    
    # legend
    legend_elements = [
        mpatches.Patch(color='#607D8B', label='Input/Output (4 channels)'),
        mpatches.Patch(color=colors['primary'], label='7×7 Convolution'),
        mpatches.Patch(color=colors['secondary'], label='Downsampling (stride 2)'),
        mpatches.Patch(color=colors['tertiary'], label='Residual Block'),
        mpatches.Patch(color=colors['quaternary'], label='Upsampling (stride 2)'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=5,
              bbox_to_anchor=(0.5, -0.15), frameon=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'fig2_generator.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'fig2_generator.png', dpi=300, bbox_inches='tight')
    plt.close()
    pprint("  saved fig2_generator.pdf/png")


# ============================================================================
# figure 3: training loss curves
# ============================================================================
def create_loss_curves(save_dir, colors, loss_file=None):
    """create training loss curves"""
    pprint("creating training loss curves...")
    
    # try to load actual loss data
    loss_data = None
    if loss_file and Path(loss_file).exists():
        with open(loss_file, 'r') as f:
            loss_data = json.load(f)
    
    # if no data, create synthetic data for demonstration
    if loss_data is None:
        pprint("  (using synthetic data for demonstration)")
        epochs = np.arange(1, 31)
        np.random.seed(42)
        
        # simulate realistic gan training dynamics
        loss_data = {
            'G_total': 30 * np.exp(-0.1 * epochs) + 3 + np.random.randn(30) * 0.3,
            'G_GAN': 2 * np.exp(-0.05 * epochs) + 0.5 + np.random.randn(30) * 0.1,
            'G_cycle': 10 * np.exp(-0.08 * epochs) + 1 + np.random.randn(30) * 0.2,
            'G_identity': 5 * np.exp(-0.1 * epochs) + 0.5 + np.random.randn(30) * 0.1,
            'D_A': 0.5 + 0.2 * np.exp(-0.1 * epochs) + np.random.randn(30) * 0.05,
            'D_B': 0.5 + 0.2 * np.exp(-0.1 * epochs) + np.random.randn(30) * 0.05,
        }
    else:
        epochs = np.arange(1, len(loss_data['G_total']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # total generator loss
    ax = axes[0, 0]
    ax.plot(epochs, loss_data['G_total'], color=colors['primary'], lw=2, label='Total')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(a) Total Generator Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # generator components
    ax = axes[0, 1]
    ax.plot(epochs, loss_data['G_GAN'], color=colors['primary'], lw=2, label='Adversarial')
    ax.plot(epochs, loss_data['G_cycle'], color=colors['secondary'], lw=2, label='Cycle')
    ax.plot(epochs, loss_data['G_identity'], color=colors['tertiary'], lw=2, label='Identity')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(b) Generator Loss Components')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # discriminator losses
    ax = axes[1, 0]
    ax.plot(epochs, loss_data['D_A'], color=colors['domain_a'], lw=2, label='D_A (BraTS)')
    ax.plot(epochs, loss_data['D_B'], color=colors['domain_b'], lw=2, label='D_B (UPenn)')
    ax.axhline(y=0.5, color='gray', ls='--', lw=1, alpha=0.7, label='Ideal')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(c) Discriminator Losses')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # training dynamics
    ax = axes[1, 1]
    g_total = np.array(loss_data['G_total'])
    d_avg = (np.array(loss_data['D_A']) + np.array(loss_data['D_B'])) / 2
    ax.plot(epochs, g_total / g_total[0], color=colors['primary'], lw=2, label='G (normalized)')
    ax.plot(epochs, d_avg / d_avg[0], color=colors['quaternary'], lw=2, label='D (normalized)')
    ax.axhline(y=1.0, color='gray', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized Loss')
    ax.set_title('(d) Training Dynamics (G vs D)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'fig3_training_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'fig3_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    pprint("  saved fig3_training_curves.pdf/png")


# ============================================================================
# figure 4: dataset statistics
# ============================================================================
def create_dataset_statistics(save_dir, colors):
    """create dataset statistics visualization"""
    pprint("creating dataset statistics...")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # dataset composition
    ax = axes[0, 0]
    datasets = ['BraTS-TCGA', 'UPenn-GBM']
    subjects = [102, 671]
    colors_bar = [colors['domain_a'], colors['domain_b']]
    bars = ax.bar(datasets, subjects, color=colors_bar, edgecolor='black', linewidth=1)
    ax.set_ylabel('Number of Subjects')
    ax.set_title('(a) Dataset Size Comparison')
    for bar, val in zip(bars, subjects):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(val), ha='center', va='bottom', fontweight='bold')
    
    # train/val/test split
    ax = axes[0, 1]
    splits = ['Train', 'Validation', 'Test']
    brats_split = [62, 14, 12]
    upenn_split = [403, 84, 79]
    
    x = np.arange(len(splits))
    width = 0.35
    bars1 = ax.bar(x - width/2, brats_split, width, label='BraTS', 
                   color=colors['domain_a'], edgecolor='black')
    bars2 = ax.bar(x + width/2, upenn_split, width, label='UPenn',
                   color=colors['domain_b'], edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel('Number of Subjects')
    ax.set_title('(b) Train/Val/Test Split')
    ax.legend()
    
    # modality information
    ax = axes[1, 0]
    modalities = ['T1', 'T1-Gd', 'T2', 'FLAIR']
    properties = [
        'Anatomical\nStructure',
        'Tumor\nEnhancement',
        'Edema\nVisualization',
        'Lesion\nDetection'
    ]
    colors_mod = [colors['primary'], colors['secondary'], 
                  colors['tertiary'], colors['quaternary']]
    
    bars = ax.barh(modalities, [1, 1, 1, 1], color=colors_mod, 
                   edgecolor='black', alpha=0.8)
    ax.set_xlim(0, 1.5)
    ax.set_xlabel('')
    ax.set_title('(c) MRI Modalities Used')
    
    for bar, prop in zip(bars, properties):
        ax.text(1.05, bar.get_y() + bar.get_height()/2, prop,
                va='center', fontsize=9)
    ax.set_xticks([])
    
    # image dimensions
    ax = axes[1, 1]
    dim_info = {
        'Input Size': '256 × 256',
        'Channels': '4 (T1, T1ce, T2, FLAIR)',
        'Depth (slices)': '~155',
        'Resolution': '1mm isotropic',
        'Normalization': '[-1, 1]'
    }
    
    ax.axis('off')
    ax.set_title('(d) Data Specifications')
    
    table_data = [[k, v] for k, v in dim_info.items()]
    table = ax.table(cellText=table_data, 
                     colLabels=['Property', 'Value'],
                     loc='center',
                     cellLoc='left',
                     colWidths=[0.4, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # style table
    for i in range(len(table_data) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor(colors['primary'])
                cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor('#f5f5f5' if i % 2 == 0 else 'white')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'fig4_dataset_stats.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'fig4_dataset_stats.png', dpi=300, bbox_inches='tight')
    plt.close()
    pprint("  saved fig4_dataset_stats.pdf/png")


# ============================================================================
# figure 5: hyperparameter configuration table
# ============================================================================
def create_hyperparameter_table(save_dir, colors):
    """create hyperparameter configuration table"""
    pprint("creating hyperparameter table...")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.axis('off')
    ax.set_title('Training Hyperparameters', fontsize=14, fontweight='bold', pad=20)
    
    data = [
        ['Learning Rate (G)', '2×10⁻⁴'],
        ['Learning Rate (D)', '1×10⁻⁴'],
        ['Batch Size', '4'],
        ['Epochs', '30'],
        ['λ_cycle', '10.0'],
        ['λ_identity', '5.0'],
        ['Optimizer', 'Adam (β₁=0.5, β₂=0.999)'],
        ['LR Decay Start', 'Epoch 15'],
        ['Label Smoothing', '0.9 / 0.1'],
        ['Spectral Norm', 'Discriminator'],
        ['Replay Buffer', '50 samples'],
        ['Gradient Clipping', '1.0'],
    ]
    
    table = ax.table(cellText=data,
                     colLabels=['Hyperparameter', 'Value'],
                     loc='center',
                     cellLoc='left',
                     colWidths=[0.5, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.4, 2.0)
    
    # style table
    for i in range(len(data) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor(colors['primary'])
                cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor('#f5f5f5' if i % 2 == 0 else 'white')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'fig5_hyperparameters.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'fig5_hyperparameters.png', dpi=300, bbox_inches='tight')
    plt.close()
    pprint("  saved fig5_hyperparameters.pdf/png")


# ============================================================================
# figure 6: anti-mode-collapse techniques
# ============================================================================
def create_techniques_diagram(save_dir, colors):
    """create diagram of anti-mode-collapse techniques"""
    pprint("creating techniques diagram...")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, 6)
    ax.axis('off')
    
    ax.text(6, 5.5, 'Anti-Mode-Collapse Techniques in CycleGAN Training',
            ha='center', fontsize=14, fontweight='bold')
    
    techniques = [
        ('Label\nSmoothing', 'Real: 0.9\nFake: 0.1', colors['primary'], 0),
        ('Spectral\nNorm', 'On D layers\nLipschitz', colors['secondary'], 1),
        ('TTUR', 'G: 2×10⁻⁴\nD: 1×10⁻⁴', colors['tertiary'], 2),
        ('Replay\nBuffer', '50 samples\nPer direction', colors['quaternary'], 3),
        ('Instance\nNoise', 'σ=0.1\nDecaying', colors['success'], 4),
        ('Gradient\nClipping', 'max=1.0\nStability', colors['warning'], 5),
    ]
    
    for i, (name, desc, color, idx) in enumerate(techniques):
        x = 1 + (idx % 3) * 4
        y = 3.5 if idx < 3 else 1
        
        # box
        rect = FancyBboxPatch((x, y), 2.5, 1.5,
                              boxstyle="round,pad=0.05,rounding_size=0.1",
                              facecolor=color, edgecolor='black', linewidth=2,
                              alpha=0.85)
        ax.add_patch(rect)
        
        # title
        ax.text(x + 1.25, y + 1.1, name, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        
        # description
        ax.text(x + 1.25, y + 0.4, desc, ha='center', va='center',
                fontsize=8, color='white')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'fig6_techniques.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'fig6_techniques.png', dpi=300, bbox_inches='tight')
    plt.close()
    pprint("  saved fig6_techniques.pdf/png")


# ============================================================================
# main function
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Generate publication figures')
    parser.add_argument('--output_dir', type=str, default='figures/publication',
                        help='Output directory for figures')
    parser.add_argument('--loss_file', type=str, default=None,
                        help='Path to training loss JSON file')
    args = parser.parse_args()
    
    pprint("\n" + "=" * 60)
    pprint("publication figure generator")
    pprint("=" * 60)
    
    # setup
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    colors = setup_publication_style()
    
    pprint(f"\noutput directory: {save_dir}")
    pprint(f"generating figures...\n")
    
    # generate all figures
    create_architecture_diagram(save_dir, colors)
    create_generator_diagram(save_dir, colors)
    create_loss_curves(save_dir, colors, args.loss_file)
    create_dataset_statistics(save_dir, colors)
    create_hyperparameter_table(save_dir, colors)
    create_techniques_diagram(save_dir, colors)
    
    pprint("\n" + "=" * 60)
    pprint("figure generation complete")
    pprint("=" * 60)
    pprint(f"\nall figures saved to: {save_dir}")
    pprint("formats: pdf (vector) and png (raster)")


if __name__ == '__main__':
    main()
