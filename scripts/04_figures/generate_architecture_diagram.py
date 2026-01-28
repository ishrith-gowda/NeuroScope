#!/usr/bin/env python3
"""
generate architecture diagram for sa-cyclegan-2.5d paper.

creates publication-quality neural network architecture visualization
showing the generator, discriminator, and attention mechanisms.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

# publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 9,
    'axes.titlesize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# color scheme
COLORS = {
    'input': '#E3F2FD',
    'conv': '#BBDEFB',
    'residual': '#90CAF9',
    'attention': '#FFCDD2',
    'cbam': '#F8BBD9',
    'upsample': '#C8E6C9',
    'output': '#DCEDC8',
    'discriminator': '#FFE0B2',
    'arrow': '#424242',
    'text': '#212121',
    'border': '#757575',
}


def draw_block(ax, x, y, width, height, color, label, fontsize=7, edgecolor='black'):
    """draw a rounded rectangle block with label."""
    rect = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color, edgecolor=edgecolor, linewidth=0.8
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            color=COLORS['text'], wrap=True)
    return rect


def draw_arrow(ax, start, end, color='black', style='->', connectionstyle='arc3,rad=0'):
    """draw an arrow between two points."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        mutation_scale=8,
        color=color,
        connectionstyle=connectionstyle,
        linewidth=0.8
    )
    ax.add_patch(arrow)
    return arrow


def create_generator_diagram(output_dir: Path):
    """
    create detailed generator architecture diagram.

    shows the 2.5d slice encoding, residual blocks with attention,
    and decoder structure.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 9)
    ax.axis('off')
    ax.set_aspect('equal')

    # title
    ax.text(7, 8.5, 'SA-CycleGAN-2.5D Generator Architecture', fontsize=14,
           ha='center', fontweight='bold')

    # input block
    draw_block(ax, 0.5, 5, 1.2, 2.5, COLORS['input'],
              'Input\n3×4×H×W\n(3 slices\n4 mods)', fontsize=7)

    # 2.5d encoder
    draw_block(ax, 2.2, 5, 1.4, 2.5, COLORS['conv'],
              '2.5D Slice\nEncoder\n7×7 Conv\nINorm, ReLU', fontsize=7)

    # downsample blocks
    draw_block(ax, 3.8, 5, 1.2, 2.0, COLORS['conv'],
              'Down ×2\n3×3 Conv\nstride=2', fontsize=7)

    draw_block(ax, 5.2, 5, 1.2, 2.0, COLORS['conv'],
              'Down ×2\n3×3 Conv\nstride=2', fontsize=7)

    # residual blocks with attention
    for i, x_pos in enumerate([6.6, 7.8, 9.0]):
        color = COLORS['attention'] if i == 1 else COLORS['residual']
        label = f'ResBlock\n{i+1}' if i != 1 else 'ResBlock\n+ Self-Attn'
        draw_block(ax, x_pos, 5, 1.0, 2.0, color, label, fontsize=7)

    # more residual blocks
    draw_block(ax, 10.2, 5, 1.5, 2.0, COLORS['residual'],
              'ResBlocks\n4-9\n(+CBAM)', fontsize=7)

    # upsample blocks
    draw_block(ax, 11.6, 5, 1.2, 2.0, COLORS['upsample'],
              'Up ×2\nTransConv\nstride=2', fontsize=7)

    draw_block(ax, 12.8, 5, 1.2, 2.0, COLORS['upsample'],
              'Up ×2\nTransConv\nstride=2', fontsize=7)

    # output
    draw_block(ax, 14.2, 5, 1.2, 2.5, COLORS['output'],
              'Output\n4×H×W\n(center\nslice)', fontsize=7)

    # draw arrows
    positions = [0.5, 2.2, 3.8, 5.2, 6.6, 7.8, 9.0, 10.2, 11.6, 12.8, 14.2]
    for i in range(len(positions) - 1):
        x1 = positions[i] + 0.6
        x2 = positions[i+1] - 0.5
        draw_arrow(ax, (x1, 5), (x2, 5), COLORS['arrow'])

    # self-attention detail box
    ax.add_patch(FancyBboxPatch(
        (6.3, 1), 4.5, 2.5,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='white', edgecolor=COLORS['border'], linewidth=1.2,
        linestyle='--'
    ))
    ax.text(8.55, 3.2, 'Self-Attention Module', fontsize=9, ha='center', fontweight='bold')

    # q, k, v blocks
    draw_block(ax, 7.0, 2.0, 0.8, 0.6, '#E1BEE7', 'Q', fontsize=8)
    draw_block(ax, 8.0, 2.0, 0.8, 0.6, '#C5CAE9', 'K', fontsize=8)
    draw_block(ax, 9.0, 2.0, 0.8, 0.6, '#B2DFDB', 'V', fontsize=8)

    draw_block(ax, 8.0, 1.2, 1.5, 0.5, '#FFF9C4', 'Attention\nMap', fontsize=7)
    draw_block(ax, 9.8, 2.0, 1.0, 0.6, COLORS['attention'], 'Output', fontsize=8)

    # arrows in attention module
    draw_arrow(ax, (7.4, 2.0), (7.6, 1.45), COLORS['arrow'], connectionstyle='arc3,rad=0.2')
    draw_arrow(ax, (8.0, 1.7), (8.0, 1.5), COLORS['arrow'])
    draw_arrow(ax, (8.6, 1.45), (8.6, 2.0), COLORS['arrow'], connectionstyle='arc3,rad=-0.2')
    draw_arrow(ax, (8.8, 1.2), (9.3, 2.0), COLORS['arrow'], connectionstyle='arc3,rad=-0.3')

    # cbam detail box
    ax.add_patch(FancyBboxPatch(
        (11.5, 0.8), 3.2, 2.7,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='white', edgecolor=COLORS['border'], linewidth=1.2,
        linestyle='--'
    ))
    ax.text(13.1, 3.2, 'CBAM Module', fontsize=9, ha='center', fontweight='bold')

    draw_block(ax, 12.2, 2.0, 1.0, 0.6, '#FFCCBC', 'Channel\nAttn', fontsize=7)
    draw_block(ax, 12.2, 1.2, 1.0, 0.6, '#D1C4E9', 'Spatial\nAttn', fontsize=7)
    draw_block(ax, 13.8, 1.6, 0.8, 1.2, COLORS['cbam'], 'Refined\nFeatures', fontsize=7)

    draw_arrow(ax, (12.7, 2.0), (13.4, 1.8), COLORS['arrow'])
    draw_arrow(ax, (12.7, 1.2), (13.4, 1.4), COLORS['arrow'])

    # legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['input'], edgecolor='black', label='Input/Output'),
        mpatches.Patch(facecolor=COLORS['conv'], edgecolor='black', label='Convolution'),
        mpatches.Patch(facecolor=COLORS['residual'], edgecolor='black', label='Residual Block'),
        mpatches.Patch(facecolor=COLORS['attention'], edgecolor='black', label='Self-Attention'),
        mpatches.Patch(facecolor=COLORS['cbam'], edgecolor='black', label='CBAM'),
        mpatches.Patch(facecolor=COLORS['upsample'], edgecolor='black', label='Upsample'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', ncol=3, fontsize=8,
             framealpha=0.9)

    plt.tight_layout()

    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(output_dir / f'fig_architecture_generator.{fmt}', format=fmt)

    plt.close(fig)
    print('[architecture] saved generator diagram')


def create_full_cyclegan_diagram(output_dir: Path):
    """
    create full cyclegan system diagram showing both generators
    and discriminators with cycle consistency.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, 7)
    ax.axis('off')
    ax.set_aspect('equal')

    # title
    ax.text(5.75, 6.7, 'SA-CycleGAN-2.5D: Bidirectional MRI Harmonization', fontsize=14,
           ha='center', fontweight='bold')

    # domain A (brats)
    draw_block(ax, 1, 5, 1.8, 1.2, '#E3F2FD', 'Domain A\n(BraTS)', fontsize=9)

    # domain B (upenn)
    draw_block(ax, 1, 2, 1.8, 1.2, '#FFF3E0', 'Domain B\n(UPenn-GBM)', fontsize=9)

    # generator A->B
    draw_block(ax, 4, 5, 2.2, 1.2, COLORS['attention'], 'Generator\nG_A→B', fontsize=10)

    # generator B->A
    draw_block(ax, 4, 2, 2.2, 1.2, COLORS['attention'], 'Generator\nG_B→A', fontsize=10)

    # fake B
    draw_block(ax, 7.5, 5, 1.8, 1.2, '#FFF3E0', 'Fake B', fontsize=9)

    # fake A
    draw_block(ax, 7.5, 2, 1.8, 1.2, '#E3F2FD', 'Fake A', fontsize=9)

    # discriminator B
    draw_block(ax, 10.5, 5, 1.8, 1.2, COLORS['discriminator'], 'Discriminator\nD_B', fontsize=9)

    # discriminator A
    draw_block(ax, 10.5, 2, 1.8, 1.2, COLORS['discriminator'], 'Discriminator\nD_A', fontsize=9)

    # arrows forward direction
    draw_arrow(ax, (1.9, 5), (2.9, 5), COLORS['arrow'])
    draw_arrow(ax, (5.1, 5), (6.6, 5), COLORS['arrow'])
    draw_arrow(ax, (8.4, 5), (9.6, 5), COLORS['arrow'])

    draw_arrow(ax, (1.9, 2), (2.9, 2), COLORS['arrow'])
    draw_arrow(ax, (5.1, 2), (6.6, 2), COLORS['arrow'])
    draw_arrow(ax, (8.4, 2), (9.6, 2), COLORS['arrow'])

    # cycle consistency arrows (curved)
    # fake B -> G_B2A -> rec A
    ax.annotate('', xy=(4, 4.3), xytext=(7.5, 4.4),
               arrowprops=dict(arrowstyle='->', color='#1976D2', lw=1.5,
                              connectionstyle='arc3,rad=-0.3'))
    ax.text(5.75, 3.9, 'Cycle A', fontsize=8, color='#1976D2', ha='center')

    # fake A -> G_A2B -> rec B
    ax.annotate('', xy=(4, 2.7), xytext=(7.5, 2.6),
               arrowprops=dict(arrowstyle='->', color='#E64A19', lw=1.5,
                              connectionstyle='arc3,rad=0.3'))
    ax.text(5.75, 3.1, 'Cycle B', fontsize=8, color='#E64A19', ha='center')

    # reconstruction blocks
    draw_block(ax, 4, 3.5, 1.5, 0.7, '#C8E6C9', 'Rec A', fontsize=8)
    draw_block(ax, 7.5, 3.5, 1.5, 0.7, '#FFE0B2', 'Rec B', fontsize=8)

    # loss labels
    ax.text(10.5, 3.5, 'Adversarial\nLoss', fontsize=8, ha='center',
           color='#D32F2F', fontweight='bold')

    ax.text(5.75, 1.0, 'Cycle Consistency Loss', fontsize=9, ha='center',
           color='#1565C0', fontweight='bold')

    ax.text(5.75, 0.5, 'L_cycle = ||G_B→A(G_A→B(A)) - A|| + ||G_A→B(G_B→A(B)) - B||',
           fontsize=8, ha='center', family='monospace')

    # legend
    legend_elements = [
        mpatches.Patch(facecolor='#E3F2FD', edgecolor='black', label='Domain A (BraTS)'),
        mpatches.Patch(facecolor='#FFF3E0', edgecolor='black', label='Domain B (UPenn)'),
        mpatches.Patch(facecolor=COLORS['attention'], edgecolor='black', label='Generator (w/ Attention)'),
        mpatches.Patch(facecolor=COLORS['discriminator'], edgecolor='black', label='Discriminator'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', ncol=2, fontsize=8,
             framealpha=0.9)

    plt.tight_layout()

    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(output_dir / f'fig_architecture_cyclegan.{fmt}', format=fmt)

    plt.close(fig)
    print('[architecture] saved full cyclegan diagram')


def create_25d_concept_diagram(output_dir: Path):
    """
    create diagram illustrating the 2.5d slice concept.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 5)
    ax.axis('off')
    ax.set_aspect('equal')

    # title
    ax.text(5, 4.7, '2.5D Slice Processing Concept', fontsize=14,
           ha='center', fontweight='bold')

    # 3d volume representation
    for i, (y_offset, alpha) in enumerate([(0.3, 0.4), (0, 1.0), (-0.3, 0.4)]):
        rect = Rectangle((0.5, 1.5 + y_offset), 2, 2,
                         facecolor='#90CAF9' if i == 1 else '#E3F2FD',
                         edgecolor='black', alpha=alpha, linewidth=1)
        ax.add_patch(rect)

    ax.text(1.5, 0.8, 'Adjacent\nSlices', fontsize=9, ha='center')
    ax.text(1.5, 4.2, 'i-1, i, i+1', fontsize=8, ha='center', style='italic')

    # arrow
    draw_arrow(ax, (3, 2.5), (4, 2.5), COLORS['arrow'])

    # stacked input
    draw_block(ax, 5, 2.5, 2, 2, COLORS['conv'],
              '2.5D Input\n[3×4, H, W]\n12 channels', fontsize=9)

    # arrow
    draw_arrow(ax, (6.2, 2.5), (7.2, 2.5), COLORS['arrow'])

    # generator
    draw_block(ax, 8, 2.5, 1.5, 1.8, COLORS['attention'],
              'Generator\nG_A→B', fontsize=9)

    # arrow
    draw_arrow(ax, (8.9, 2.5), (9.8, 2.5), COLORS['arrow'])

    # output
    rect = Rectangle((9.8, 1.5), 1.5, 2,
                     facecolor=COLORS['output'], edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(10.55, 2.5, 'Output\n[4, H, W]\nCenter\nSlice', fontsize=8, ha='center')

    # modality labels
    ax.text(5, 0.8, '4 modalities: T1, T1CE, T2, FLAIR', fontsize=8,
           ha='center', style='italic')

    plt.tight_layout()

    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(output_dir / f'fig_architecture_25d_concept.{fmt}', format=fmt)

    plt.close(fig)
    print('[architecture] saved 2.5d concept diagram')


def main():
    output_dir = Path('/Volumes/usb drive/neuroscope/figures/architecture')
    output_dir.mkdir(parents=True, exist_ok=True)

    print('[architecture] generating architecture diagrams...')
    print('=' * 60)

    create_generator_diagram(output_dir)
    create_full_cyclegan_diagram(output_dir)
    create_25d_concept_diagram(output_dir)

    print('=' * 60)
    print(f'[architecture] all diagrams saved to: {output_dir}')


if __name__ == '__main__':
    main()
