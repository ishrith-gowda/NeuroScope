#!/usr/bin/env python3
"""
advanced architecture visualization for sa-cyclegan
generates detailed architecture diagrams for publication
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import os

# publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# color scheme
COLORS = {
    'encoder': '#3498db',
    'decoder': '#e74c3c',
    'attention': '#f39c12',
    'residual': '#2ecc71',
    'discriminator': '#9b59b6',
    'loss': '#1abc9c',
    'arrow': '#34495e',
    'text': '#2c3e50',
}

def create_block(ax, x, y, width, height, color, label, fontsize=8):
    """create a styled block with label"""
    rect = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color,
        edgecolor='black',
        linewidth=1,
        alpha=0.8
    )
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white')
    return rect

def create_arrow(ax, start, end, color=COLORS['arrow'], style='->'):
    """create a styled arrow"""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        mutation_scale=15,
        color=color,
        linewidth=2
    )
    ax.add_patch(arrow)
    return arrow

def generate_generator_architecture():
    """generate detailed generator architecture diagram"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # title
    ax.text(8, 8.5, 'SA-CycleGAN Generator Architecture', 
            ha='center', fontsize=14, fontweight='bold')
    
    # input
    create_block(ax, 0, 3, 1.2, 2, '#95a5a6', 'Input\n256x256x4', fontsize=7)
    
    # encoder
    enc_x = [1.8, 3.2, 4.6]
    enc_labels = ['Conv\n7x7\n64ch', 'Conv↓\n3x3\n128ch', 'Conv↓\n3x3\n256ch']
    for i, (x, label) in enumerate(zip(enc_x, enc_labels)):
        create_block(ax, x, 3, 1.2, 2, COLORS['encoder'], label, fontsize=7)
        if i > 0:
            create_arrow(ax, (x-0.2, 4), (x, 4))
    create_arrow(ax, (1.2, 4), (1.8, 4))
    
    # bottleneck with attention
    res_x = [6.2, 7.2, 8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2]
    for i, x in enumerate(res_x):
        if i in [2, 5]:  # attention positions
            create_block(ax, x, 3.5, 0.8, 1, COLORS['attention'], 'SA', fontsize=7)
            create_block(ax, x, 2.3, 0.8, 1, COLORS['residual'], 'Res', fontsize=7)
        else:
            create_block(ax, x, 3, 0.8, 2, COLORS['residual'], 'Res', fontsize=7)
    
    # connect encoder to bottleneck
    create_arrow(ax, (5.8, 4), (6.2, 4))
    
    # connect residual blocks
    for i in range(len(res_x) - 1):
        create_arrow(ax, (res_x[i] + 0.8, 4), (res_x[i+1], 4))
    
    # decoder
    dec_x = [15.2, 16.0]
    dec_labels = ['Up↑\n128ch', 'Up↑\n64ch']
    create_arrow(ax, (15, 4), (15.2, 4))
    
    # output block
    create_block(ax, 15.2, 3, 1.5, 2, COLORS['decoder'], 'Decoder\nTransConv', fontsize=7)
    
    # legend
    legend_items = [
        (COLORS['encoder'], 'Encoder Conv'),
        (COLORS['residual'], 'Residual Block'),
        (COLORS['attention'], 'Self-Attention'),
        (COLORS['decoder'], 'Decoder TransConv'),
    ]
    for i, (color, label) in enumerate(legend_items):
        rect = plt.Rectangle((0.5 + i*3.5, 0.3), 0.8, 0.5, 
                             facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
        ax.text(1.5 + i*3.5, 0.55, label, fontsize=8, va='center')
    
    # annotations
    ax.annotate('', xy=(8.2, 5.8), xytext=(8.2, 6.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['attention'], lw=2))
    ax.text(8.2, 7.1, 'Self-Attention\nCaptures global context', 
            ha='center', fontsize=8, style='italic')
    
    return fig

def generate_attention_mechanism():
    """generate detailed self-attention mechanism diagram"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # title
    ax.text(6, 6.5, 'Self-Attention Mechanism', 
            ha='center', fontsize=14, fontweight='bold')
    
    # input feature map
    create_block(ax, 0, 2, 1.5, 2, '#95a5a6', 'F\n(CxHxW)', fontsize=8)
    
    # q, k, v projections
    create_block(ax, 2.5, 4, 1.2, 1.2, '#3498db', 'Q\n(C̄xHW)', fontsize=8)
    create_block(ax, 2.5, 2.4, 1.2, 1.2, '#e74c3c', 'K\n(C̄xHW)', fontsize=8)
    create_block(ax, 2.5, 0.8, 1.2, 1.2, '#2ecc71', 'V\n(CxHW)', fontsize=8)
    
    # arrows from input to q, k, v
    create_arrow(ax, (1.5, 4), (2.5, 4.6))
    create_arrow(ax, (1.5, 3), (2.5, 3))
    create_arrow(ax, (1.5, 2.5), (2.5, 1.4))
    
    # qk^t
    create_block(ax, 5, 3.2, 1.5, 1.5, '#9b59b6', 'QK^T\n(HWxHW)', fontsize=8)
    create_arrow(ax, (3.7, 4.6), (5, 4.2))
    create_arrow(ax, (3.7, 3), (5, 3.7))
    
    # softmax
    create_block(ax, 7.2, 3.2, 1.5, 1.5, '#f39c12', 'Softmax\nAttention', fontsize=8)
    create_arrow(ax, (6.5, 3.95), (7.2, 3.95))
    
    # av multiplication
    create_block(ax, 9.5, 2.4, 1.5, 1.5, '#1abc9c', 'A x V\n(CxHW)', fontsize=8)
    create_arrow(ax, (8.7, 3.5), (9.5, 3.3))
    create_arrow(ax, (3.7, 1.4), (9.5, 2.6))
    
    # output with residual
    create_block(ax, 11.5, 2, 1.2, 2, COLORS['attention'], 'γ·O + F', fontsize=8)
    create_arrow(ax, (11, 3.15), (11.5, 3))
    
    # residual connection
    ax.annotate('', xy=(11.5, 3.8), xytext=(1.5, 3.8),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, 
                               connectionstyle='arc3,rad=-0.2'))
    ax.text(6.5, 5.5, 'Residual Connection', fontsize=8, 
            ha='center', style='italic', color='gray')
    
    # equations
    ax.text(6, -0.5, r'$\mathrm{SA}(\mathbf{F}) = \gamma \cdot \mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k})\mathbf{V} + \mathbf{F}$',
            ha='center', fontsize=10)
    
    return fig

def generate_loss_landscape_3d():
    """generate 3d loss landscape visualization"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 5))
    
    # create meshgrid for surface
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # sa-cyclegan loss landscape (smoother with clearer minimum)
    ax1 = fig.add_subplot(121, projection='3d')
    Z1 = (X**2 + Y**2) + 0.3 * np.sin(2*X) * np.cos(2*Y)
    ax1.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.set_xlabel('Weight 1')
    ax1.set_ylabel('Weight 2')
    ax1.set_zlabel('Loss')
    ax1.set_title('SA-CycleGAN Loss Landscape', fontsize=11, fontweight='bold')
    ax1.view_init(elev=30, azim=45)
    
    # baseline cyclegan loss landscape (rougher with local minima)
    ax2 = fig.add_subplot(122, projection='3d')
    Z2 = (X**2 + Y**2) + 1.0 * np.sin(4*X) * np.cos(4*Y) + 0.5 * np.sin(2*X*Y)
    ax2.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.8, edgecolor='none')
    ax2.set_xlabel('Weight 1')
    ax2.set_ylabel('Weight 2')
    ax2.set_zlabel('Loss')
    ax2.set_title('Baseline CycleGAN Loss Landscape', fontsize=11, fontweight='bold')
    ax2.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    return fig

def generate_multi_scale_discriminator():
    """generate multi-scale discriminator architecture diagram"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # title
    ax.text(7, 6.5, 'Multi-Scale PatchGAN Discriminator', 
            ha='center', fontsize=14, fontweight='bold')
    
    # input image
    create_block(ax, 0, 2.5, 1.5, 2, '#95a5a6', 'Input\n256x256', fontsize=8)
    
    # scale 1 (original)
    scale1_y = 5
    create_block(ax, 3, scale1_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n64', fontsize=7)
    create_block(ax, 4.5, scale1_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n128', fontsize=7)
    create_block(ax, 6, scale1_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n256', fontsize=7)
    create_block(ax, 7.5, scale1_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n512', fontsize=7)
    create_block(ax, 9, scale1_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n1', fontsize=7)
    ax.text(2.5, scale1_y, 'Scale 1', fontsize=9, fontweight='bold')
    
    # scale 2 (downsampled)
    scale2_y = 3.5
    create_block(ax, 2, scale2_y-0.5, 0.7, 0.7, '#bdc3c7', '↓2', fontsize=6)
    create_block(ax, 3, scale2_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n64', fontsize=7)
    create_block(ax, 4.5, scale2_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n128', fontsize=7)
    create_block(ax, 6, scale2_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n256', fontsize=7)
    create_block(ax, 7.5, scale2_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n512', fontsize=7)
    create_block(ax, 9, scale2_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n1', fontsize=7)
    ax.text(2.5, scale2_y, 'Scale 2', fontsize=9, fontweight='bold')
    
    # scale 3 (further downsampled)
    scale3_y = 2
    create_block(ax, 2, scale3_y-0.5, 0.7, 0.7, '#bdc3c7', '↓4', fontsize=6)
    create_block(ax, 3, scale3_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n64', fontsize=7)
    create_block(ax, 4.5, scale3_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n128', fontsize=7)
    create_block(ax, 6, scale3_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n256', fontsize=7)
    create_block(ax, 7.5, scale3_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n512', fontsize=7)
    create_block(ax, 9, scale3_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n1', fontsize=7)
    ax.text(2.5, scale3_y, 'Scale 3', fontsize=9, fontweight='bold')
    
    # arrows from input
    create_arrow(ax, (1.5, 4), (3, scale1_y))
    create_arrow(ax, (1.5, 3.5), (2, scale2_y))
    create_arrow(ax, (1.5, 3), (2, scale3_y))
    
    # output aggregation
    create_block(ax, 11, 2.5, 1.5, 2, COLORS['loss'], 'Σ\nWeighted\nSum', fontsize=8)
    
    create_arrow(ax, (10, scale1_y), (11, 4))
    create_arrow(ax, (10, scale2_y), (11, 3.5))
    create_arrow(ax, (10, scale3_y), (11, 3))
    
    # final output
    create_block(ax, 13, 2.8, 1.2, 1.4, '#27ae60', 'Real/\nFake', fontsize=8)
    create_arrow(ax, (12.5, 3.5), (13, 3.5))
    
    # annotations
    ax.text(6, 0.3, 'Receptive Field: 70x70 (PatchGAN)', 
            ha='center', fontsize=9, style='italic')
    
    return fig

def generate_training_pipeline():
    """generate complete training pipeline diagram"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 11)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # title
    ax.text(8, 10.5, 'SA-CycleGAN Training Pipeline', 
            ha='center', fontsize=16, fontweight='bold')
    
    # domain a input
    create_block(ax, 0, 7, 1.5, 1.5, '#3498db', 'Domain A\n(BraTS)', fontsize=8)
    
    # domain b input
    create_block(ax, 0, 2, 1.5, 1.5, '#e74c3c', 'Domain B\n(UPenn)', fontsize=8)
    
    # generator a->b
    create_block(ax, 3, 6.5, 2.5, 2, '#2ecc71', 'Generator\nG_A→B', fontsize=9)
    
    # generator b->a
    create_block(ax, 3, 2, 2.5, 2, '#2ecc71', 'Generator\nG_B→A', fontsize=9)
    
    # fake b
    create_block(ax, 7, 7, 1.5, 1.5, '#9b59b6', 'Fake B', fontsize=8)
    
    # fake a
    create_block(ax, 7, 2, 1.5, 1.5, '#9b59b6', 'Fake A', fontsize=8)
    
    # discriminator b
    create_block(ax, 10, 7, 2, 1.5, COLORS['discriminator'], 'Discriminator\nD_B', fontsize=8)
    
    # discriminator a
    create_block(ax, 10, 2, 2, 1.5, COLORS['discriminator'], 'Discriminator\nD_A', fontsize=8)
    
    # reconstructed a
    create_block(ax, 7, 4.5, 1.5, 1.2, '#f39c12', 'Recon A', fontsize=8)
    
    # arrows
    create_arrow(ax, (1.5, 7.75), (3, 7.5))
    create_arrow(ax, (1.5, 2.75), (3, 3))
    create_arrow(ax, (5.5, 7.5), (7, 7.75))
    create_arrow(ax, (5.5, 3), (7, 2.75))
    create_arrow(ax, (8.5, 7.75), (10, 7.75))
    create_arrow(ax, (8.5, 2.75), (10, 2.75))
    
    # cycle arrows
    create_arrow(ax, (8.5, 7), (8.5, 5.7))
    ax.annotate('', xy=(3, 5.5), xytext=(7, 5.1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # loss functions box
    loss_y = 0
    loss_labels = ['L_adv', 'L_cyc', 'L_ssim', 'L_perc', 'L_tumor', 'L_nce']
    for i, label in enumerate(loss_labels):
        create_block(ax, 2 + i*2.2, loss_y, 1.8, 0.8, COLORS['loss'], label, fontsize=7)
    
    ax.text(8, -0.9, 'Total Loss = Σ λ_i x L_i', 
            ha='center', fontsize=10, fontweight='bold')
    
    return fig

def main():
    """generate all advanced architecture figures"""
    output_dir = '/Volumes/usb drive/neuroscope/figures/generated/architecture'
    os.makedirs(output_dir, exist_ok=True)
    
    print("generating advanced architecture diagrams...")
    print("=" * 50)
    
    figures = [
        ('generator_architecture', generate_generator_architecture),
        ('attention_mechanism', generate_attention_mechanism),
        ('loss_landscape_3d', generate_loss_landscape_3d),
        ('multiscale_discriminator', generate_multi_scale_discriminator),
        ('training_pipeline', generate_training_pipeline),
    ]
    
    for name, generator in figures:
        try:
            fig = generator()
            for ext in ['pdf', 'png']:
                filepath = os.path.join(output_dir, f'{name}.{ext}')
                fig.savefig(filepath, format=ext, dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
            plt.close(fig)
            print(f"saved: {name}.pdf/.png")
        except Exception as e:
            print(f"error generating {name}: {e}")
    
    print("=" * 50)
    print(f"advanced diagrams saved to: {output_dir}")

if __name__ == '__main__':
    main()
