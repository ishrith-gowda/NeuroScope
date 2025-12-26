#!/usr/bin/env python3
"""
Advanced Architecture Visualization for SA-CycleGAN
Generates detailed architecture diagrams for publication
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import os

# Publication-quality settings
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

# Color scheme
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
    """Create a styled block with label"""
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
    """Create a styled arrow"""
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
    """Generate detailed generator architecture diagram"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(8, 8.5, 'SA-CycleGAN Generator Architecture', 
            ha='center', fontsize=14, fontweight='bold')
    
    # Input
    create_block(ax, 0, 3, 1.2, 2, '#95a5a6', 'Input\n256×256×4', fontsize=7)
    
    # Encoder
    enc_x = [1.8, 3.2, 4.6]
    enc_labels = ['Conv\n7×7\n64ch', 'Conv↓\n3×3\n128ch', 'Conv↓\n3×3\n256ch']
    for i, (x, label) in enumerate(zip(enc_x, enc_labels)):
        create_block(ax, x, 3, 1.2, 2, COLORS['encoder'], label, fontsize=7)
        if i > 0:
            create_arrow(ax, (x-0.2, 4), (x, 4))
    create_arrow(ax, (1.2, 4), (1.8, 4))
    
    # Bottleneck with attention
    res_x = [6.2, 7.2, 8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2]
    for i, x in enumerate(res_x):
        if i in [2, 5]:  # Attention positions
            create_block(ax, x, 3.5, 0.8, 1, COLORS['attention'], 'SA', fontsize=7)
            create_block(ax, x, 2.3, 0.8, 1, COLORS['residual'], 'Res', fontsize=7)
        else:
            create_block(ax, x, 3, 0.8, 2, COLORS['residual'], 'Res', fontsize=7)
    
    # Connect encoder to bottleneck
    create_arrow(ax, (5.8, 4), (6.2, 4))
    
    # Connect residual blocks
    for i in range(len(res_x) - 1):
        create_arrow(ax, (res_x[i] + 0.8, 4), (res_x[i+1], 4))
    
    # Decoder
    dec_x = [15.2, 16.0]
    dec_labels = ['Up↑\n128ch', 'Up↑\n64ch']
    create_arrow(ax, (15, 4), (15.2, 4))
    
    # Output block
    create_block(ax, 15.2, 3, 1.5, 2, COLORS['decoder'], 'Decoder\nTransConv', fontsize=7)
    
    # Legend
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
    
    # Annotations
    ax.annotate('', xy=(8.2, 5.8), xytext=(8.2, 6.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['attention'], lw=2))
    ax.text(8.2, 7.1, 'Self-Attention\nCaptures global context', 
            ha='center', fontsize=8, style='italic')
    
    return fig

def generate_attention_mechanism():
    """Generate detailed self-attention mechanism diagram"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(6, 6.5, 'Self-Attention Mechanism', 
            ha='center', fontsize=14, fontweight='bold')
    
    # Input feature map
    create_block(ax, 0, 2, 1.5, 2, '#95a5a6', 'F\n(C×H×W)', fontsize=8)
    
    # Q, K, V projections
    create_block(ax, 2.5, 4, 1.2, 1.2, '#3498db', 'Q\n(C̄×HW)', fontsize=8)
    create_block(ax, 2.5, 2.4, 1.2, 1.2, '#e74c3c', 'K\n(C̄×HW)', fontsize=8)
    create_block(ax, 2.5, 0.8, 1.2, 1.2, '#2ecc71', 'V\n(C×HW)', fontsize=8)
    
    # Arrows from input to Q, K, V
    create_arrow(ax, (1.5, 4), (2.5, 4.6))
    create_arrow(ax, (1.5, 3), (2.5, 3))
    create_arrow(ax, (1.5, 2.5), (2.5, 1.4))
    
    # QK^T
    create_block(ax, 5, 3.2, 1.5, 1.5, '#9b59b6', 'QK^T\n(HW×HW)', fontsize=8)
    create_arrow(ax, (3.7, 4.6), (5, 4.2))
    create_arrow(ax, (3.7, 3), (5, 3.7))
    
    # Softmax
    create_block(ax, 7.2, 3.2, 1.5, 1.5, '#f39c12', 'Softmax\nAttention', fontsize=8)
    create_arrow(ax, (6.5, 3.95), (7.2, 3.95))
    
    # AV multiplication
    create_block(ax, 9.5, 2.4, 1.5, 1.5, '#1abc9c', 'A × V\n(C×HW)', fontsize=8)
    create_arrow(ax, (8.7, 3.5), (9.5, 3.3))
    create_arrow(ax, (3.7, 1.4), (9.5, 2.6))
    
    # Output with residual
    create_block(ax, 11.5, 2, 1.2, 2, COLORS['attention'], 'γ·O + F', fontsize=8)
    create_arrow(ax, (11, 3.15), (11.5, 3))
    
    # Residual connection
    ax.annotate('', xy=(11.5, 3.8), xytext=(1.5, 3.8),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, 
                               connectionstyle='arc3,rad=-0.2'))
    ax.text(6.5, 5.5, 'Residual Connection', fontsize=8, 
            ha='center', style='italic', color='gray')
    
    # Equations
    ax.text(6, -0.5, r'$\mathrm{SA}(\mathbf{F}) = \gamma \cdot \mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k})\mathbf{V} + \mathbf{F}$',
            ha='center', fontsize=10)
    
    return fig

def generate_loss_landscape_3d():
    """Generate 3D loss landscape visualization"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 5))
    
    # Create meshgrid for surface
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # SA-CycleGAN loss landscape (smoother with clearer minimum)
    ax1 = fig.add_subplot(121, projection='3d')
    Z1 = (X**2 + Y**2) + 0.3 * np.sin(2*X) * np.cos(2*Y)
    ax1.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.set_xlabel('Weight 1')
    ax1.set_ylabel('Weight 2')
    ax1.set_zlabel('Loss')
    ax1.set_title('SA-CycleGAN Loss Landscape', fontsize=11, fontweight='bold')
    ax1.view_init(elev=30, azim=45)
    
    # Baseline CycleGAN loss landscape (rougher with local minima)
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
    """Generate multi-scale discriminator architecture diagram"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(7, 6.5, 'Multi-Scale PatchGAN Discriminator', 
            ha='center', fontsize=14, fontweight='bold')
    
    # Input image
    create_block(ax, 0, 2.5, 1.5, 2, '#95a5a6', 'Input\n256×256', fontsize=8)
    
    # Scale 1 (original)
    scale1_y = 5
    create_block(ax, 3, scale1_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n64', fontsize=7)
    create_block(ax, 4.5, scale1_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n128', fontsize=7)
    create_block(ax, 6, scale1_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n256', fontsize=7)
    create_block(ax, 7.5, scale1_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n512', fontsize=7)
    create_block(ax, 9, scale1_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n1', fontsize=7)
    ax.text(2.5, scale1_y, 'Scale 1', fontsize=9, fontweight='bold')
    
    # Scale 2 (downsampled)
    scale2_y = 3.5
    create_block(ax, 2, scale2_y-0.5, 0.7, 0.7, '#bdc3c7', '↓2', fontsize=6)
    create_block(ax, 3, scale2_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n64', fontsize=7)
    create_block(ax, 4.5, scale2_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n128', fontsize=7)
    create_block(ax, 6, scale2_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n256', fontsize=7)
    create_block(ax, 7.5, scale2_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n512', fontsize=7)
    create_block(ax, 9, scale2_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n1', fontsize=7)
    ax.text(2.5, scale2_y, 'Scale 2', fontsize=9, fontweight='bold')
    
    # Scale 3 (further downsampled)
    scale3_y = 2
    create_block(ax, 2, scale3_y-0.5, 0.7, 0.7, '#bdc3c7', '↓4', fontsize=6)
    create_block(ax, 3, scale3_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n64', fontsize=7)
    create_block(ax, 4.5, scale3_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n128', fontsize=7)
    create_block(ax, 6, scale3_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n256', fontsize=7)
    create_block(ax, 7.5, scale3_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n512', fontsize=7)
    create_block(ax, 9, scale3_y-0.5, 1, 1, COLORS['discriminator'], 'Conv\n1', fontsize=7)
    ax.text(2.5, scale3_y, 'Scale 3', fontsize=9, fontweight='bold')
    
    # Arrows from input
    create_arrow(ax, (1.5, 4), (3, scale1_y))
    create_arrow(ax, (1.5, 3.5), (2, scale2_y))
    create_arrow(ax, (1.5, 3), (2, scale3_y))
    
    # Output aggregation
    create_block(ax, 11, 2.5, 1.5, 2, COLORS['loss'], 'Σ\nWeighted\nSum', fontsize=8)
    
    create_arrow(ax, (10, scale1_y), (11, 4))
    create_arrow(ax, (10, scale2_y), (11, 3.5))
    create_arrow(ax, (10, scale3_y), (11, 3))
    
    # Final output
    create_block(ax, 13, 2.8, 1.2, 1.4, '#27ae60', 'Real/\nFake', fontsize=8)
    create_arrow(ax, (12.5, 3.5), (13, 3.5))
    
    # Annotations
    ax.text(6, 0.3, 'Receptive Field: 70×70 (PatchGAN)', 
            ha='center', fontsize=9, style='italic')
    
    return fig

def generate_training_pipeline():
    """Generate complete training pipeline diagram"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 11)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(8, 10.5, 'SA-CycleGAN Training Pipeline', 
            ha='center', fontsize=16, fontweight='bold')
    
    # Domain A input
    create_block(ax, 0, 7, 1.5, 1.5, '#3498db', 'Domain A\n(BraTS)', fontsize=8)
    
    # Domain B input
    create_block(ax, 0, 2, 1.5, 1.5, '#e74c3c', 'Domain B\n(UPenn)', fontsize=8)
    
    # Generator A->B
    create_block(ax, 3, 6.5, 2.5, 2, '#2ecc71', 'Generator\nG_A→B', fontsize=9)
    
    # Generator B->A
    create_block(ax, 3, 2, 2.5, 2, '#2ecc71', 'Generator\nG_B→A', fontsize=9)
    
    # Fake B
    create_block(ax, 7, 7, 1.5, 1.5, '#9b59b6', 'Fake B', fontsize=8)
    
    # Fake A
    create_block(ax, 7, 2, 1.5, 1.5, '#9b59b6', 'Fake A', fontsize=8)
    
    # Discriminator B
    create_block(ax, 10, 7, 2, 1.5, COLORS['discriminator'], 'Discriminator\nD_B', fontsize=8)
    
    # Discriminator A
    create_block(ax, 10, 2, 2, 1.5, COLORS['discriminator'], 'Discriminator\nD_A', fontsize=8)
    
    # Reconstructed A
    create_block(ax, 7, 4.5, 1.5, 1.2, '#f39c12', 'Recon A', fontsize=8)
    
    # Arrows
    create_arrow(ax, (1.5, 7.75), (3, 7.5))
    create_arrow(ax, (1.5, 2.75), (3, 3))
    create_arrow(ax, (5.5, 7.5), (7, 7.75))
    create_arrow(ax, (5.5, 3), (7, 2.75))
    create_arrow(ax, (8.5, 7.75), (10, 7.75))
    create_arrow(ax, (8.5, 2.75), (10, 2.75))
    
    # Cycle arrows
    create_arrow(ax, (8.5, 7), (8.5, 5.7))
    ax.annotate('', xy=(3, 5.5), xytext=(7, 5.1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # Loss functions box
    loss_y = 0
    loss_labels = ['L_adv', 'L_cyc', 'L_ssim', 'L_perc', 'L_tumor', 'L_nce']
    for i, label in enumerate(loss_labels):
        create_block(ax, 2 + i*2.2, loss_y, 1.8, 0.8, COLORS['loss'], label, fontsize=7)
    
    ax.text(8, -0.9, 'Total Loss = Σ λ_i × L_i', 
            ha='center', fontsize=10, fontweight='bold')
    
    return fig

def main():
    """Generate all advanced architecture figures"""
    output_dir = '/Volumes/usb drive/neuroscope/figures/generated/architecture'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating advanced architecture diagrams...")
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
            print(f"Saved: {name}.pdf/.png")
        except Exception as e:
            print(f"Error generating {name}: {e}")
    
    print("=" * 50)
    print(f"Advanced diagrams saved to: {output_dir}")

if __name__ == '__main__':
    main()
