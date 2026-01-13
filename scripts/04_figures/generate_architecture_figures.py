#!/usr/bin/env python3
"""
Generate Architecture and Model Diagrams for Publication

Creates comprehensive visualizations of:
- Model architecture comparison (SA-CycleGAN vs Baseline)
- Attention mechanism diagrams
- Network structure details
- Parameter breakdown

Author: NeuroScope Research Team
Date: January 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from latex_figure_config import (
    FIGURE_SIZES, COLORS, save_figure
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'figures/main'


def generate_architecture_comparison():
    """
    Figure: SA-CycleGAN vs Baseline Architecture Comparison

    Shows side-by-side comparison highlighting attention mechanisms.
    """
    print("Generating: Architecture Comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(r'\textbf{Model Architecture Comparison}', y=1.02)

    # Baseline CycleGAN (left)
    ax = axes[0]
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title(r'(a) Baseline CycleGAN-2.5D', fontsize=11)

    # Baseline components
    baseline_blocks = [
        {'name': 'Input\n[B,12,H,W]', 'y': 9, 'color': COLORS['gray']},
        {'name': 'Encoder\n(3 conv)', 'y': 7.5, 'color': COLORS['primary']},
        {'name': 'ResBlock 1-3', 'y': 6.5, 'color': COLORS['secondary']},
        {'name': 'ResBlock 4-6', 'y': 5.5, 'color': COLORS['secondary']},
        {'name': 'ResBlock 7-9', 'y': 4.5, 'color': COLORS['secondary']},
        {'name': 'Decoder\n(3 transconv)', 'y': 3, 'color': COLORS['primary']},
        {'name': 'Output\n[B,4,H,W]', 'y': 1.5, 'color': COLORS['gray']},
    ]

    for block in baseline_blocks:
        rect = mpatches.FancyBboxPatch((0.5, block['y']-0.3), 4, 0.6,
                                       boxstyle="round,pad=0.05",
                                       facecolor=block['color'],
                                       edgecolor='black',
                                       alpha=0.6,
                                       linewidth=1.5)
        ax.add_patch(rect)
        ax.text(2.5, block['y'], block['name'],
                ha='center', va='center', fontsize=8)

        # Arrows
        if block != baseline_blocks[-1]:
            ax.annotate('', xy=(2.5, block['y']-0.4), xytext=(2.5, block['y']-0.6),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Parameters
    ax.text(2.5, 0.5, r'Parameters: 33.88M', ha='center', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

    # SA-CycleGAN (right)
    ax = axes[1]
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title(r'(b) SA-CycleGAN-2.5D (Ours)', fontsize=11)

    # SA-CycleGAN components (with attention)
    sa_blocks = [
        {'name': 'Input\n[B,12,H,W]', 'y': 9, 'color': COLORS['gray']},
        {'name': 'Encoder\n(3 conv)', 'y': 7.5, 'color': COLORS['primary']},
        {'name': 'ResBlock+CBAM 1', 'y': 6.7, 'color': COLORS['success']},
        {'name': 'ResBlock+CBAM 2', 'y': 6.1, 'color': COLORS['success']},
        {'name': 'ResBlock+CBAM 3', 'y': 5.5, 'color': COLORS['success']},
        {'name': 'Self-Attention', 'y': 4.8, 'color': COLORS['danger'], 'highlight': True},
        {'name': 'ResBlock+CBAM 4-6', 'y': 4, 'color': COLORS['success']},
        {'name': 'ResBlock+CBAM 7-9', 'y': 3.2, 'color': COLORS['success']},
        {'name': 'Decoder\n(3 transconv)', 'y': 2, 'color': COLORS['primary']},
        {'name': 'Output\n[B,4,H,W]', 'y': 0.8, 'color': COLORS['gray']},
    ]

    for block in sa_blocks:
        if block.get('highlight'):
            # Highlight self-attention
            rect = mpatches.FancyBboxPatch((0.3, block['y']-0.25), 4.4, 0.5,
                                           boxstyle="round,pad=0.05",
                                           facecolor=block['color'],
                                           edgecolor='red',
                                           alpha=0.7,
                                           linewidth=2.5)
        else:
            rect = mpatches.FancyBboxPatch((0.5, block['y']-0.22), 4, 0.44,
                                           boxstyle="round,pad=0.05",
                                           facecolor=block['color'],
                                           edgecolor='black',
                                           alpha=0.6,
                                           linewidth=1.5)
        ax.add_patch(rect)
        ax.text(2.5, block['y'], block['name'],
                ha='center', va='center', fontsize=7.5)

        # Arrows
        if block != sa_blocks[-1]:
            ax.annotate('', xy=(2.5, block['y']-0.3), xytext=(2.5, block['y']-0.45),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Parameters and improvement
    ax.text(2.5, 0.2, r'Parameters: 35.1M (+3.6\%)', ha='center', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Add legend for attention
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['secondary'], alpha=0.6, label='Standard ResBlock'),
        mpatches.Patch(facecolor=COLORS['success'], alpha=0.6, label='ResBlock + CBAM'),
        mpatches.Patch(facecolor=COLORS['danger'], alpha=0.7, edgecolor='red',
                      linewidth=2, label='Self-Attention'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
              bbox_to_anchor=(0.5, -0.05), fontsize=8)

    fig.subplots_adjust(top=0.92)
    plt.tight_layout()
    save_figure(fig, 'fig12_architecture_comparison', OUTPUT_DIR)
    plt.close()


def generate_attention_mechanism_diagram():
    """
    Figure: Attention Mechanisms Detailed Diagram

    Shows CBAM and Self-Attention structure in detail.
    """
    print("Generating: Attention Mechanisms Diagram...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(r'\textbf{Attention Mechanisms in SA-CycleGAN-2.5D}', y=1.02)

    # CBAM (left)
    ax = axes[0]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title(r'(a) CBAM (Channel and Spatial)', fontsize=10)

    # CBAM flow
    cbam_blocks = [
        {'name': 'Input\nFeatures', 'y': 7, 'x': 3, 'w': 2, 'color': COLORS['gray']},
        {'name': 'Channel\nAttention', 'y': 5.5, 'x': 3, 'w': 2, 'color': COLORS['primary']},
        {'name': 'Refined\nFeatures', 'y': 4.5, 'x': 3, 'w': 2, 'color': COLORS['gray']},
        {'name': 'Spatial\nAttention', 'y': 3, 'x': 3, 'w': 2, 'color': COLORS['secondary']},
        {'name': 'Output\nFeatures', 'y': 1.5, 'x': 3, 'w': 2, 'color': COLORS['success']},
    ]

    for i, block in enumerate(cbam_blocks):
        rect = mpatches.FancyBboxPatch((block['x']-block['w']/2, block['y']-0.3),
                                       block['w'], 0.6,
                                       boxstyle="round,pad=0.05",
                                       facecolor=block['color'],
                                       edgecolor='black',
                                       alpha=0.6,
                                       linewidth=1.5)
        ax.add_patch(rect)
        ax.text(block['x'], block['y'], block['name'],
                ha='center', va='center', fontsize=8)

        if i < len(cbam_blocks) - 1:
            ax.annotate('', xy=(3, block['y']-0.4), xytext=(3, block['y']-0.7),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Add side annotations
    ax.text(0.5, 5.5, r'$\otimes$', fontsize=16, ha='center', va='center')
    ax.text(0.5, 3, r'$\otimes$', fontsize=16, ha='center', va='center')
    ax.text(0.3, 5.5, 'Multiply', fontsize=7, ha='right', va='center', rotation=90)
    ax.text(0.3, 3, 'Multiply', fontsize=7, ha='right', va='center', rotation=90)

    # Self-Attention (right)
    ax = axes[1]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title(r'(b) Self-Attention (Bottleneck)', fontsize=10)

    # Self-attention flow
    sa_y = 7
    # Input
    rect = mpatches.FancyBboxPatch((1.5, sa_y-0.3), 3, 0.6,
                                   boxstyle="round,pad=0.05",
                                   facecolor=COLORS['gray'],
                                   edgecolor='black', alpha=0.6, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(3, sa_y, 'Input [B,C,H,W]', ha='center', va='center', fontsize=8)

    # Q, K, V branches
    sa_y = 5.5
    for i, (label, x_pos) in enumerate([('Q', 1), ('K', 3), ('V', 5)]):
        rect = mpatches.FancyBboxPatch((x_pos-0.4, sa_y-0.2), 0.8, 0.4,
                                       boxstyle="round,pad=0.03",
                                       facecolor=COLORS['primary'],
                                       edgecolor='black', alpha=0.6, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x_pos, sa_y, label, ha='center', va='center', fontsize=9, fontweight='bold')

        # Arrow from input
        ax.annotate('', xy=(x_pos, sa_y+0.3), xytext=(x_pos, 6.5),
                   arrowprops=dict(arrowstyle='->', lw=1, color='black'))

    # Attention operation
    sa_y = 4
    rect = mpatches.FancyBboxPatch((1, sa_y-0.3), 4, 0.6,
                                   boxstyle="round,pad=0.05",
                                   facecolor=COLORS['danger'],
                                   edgecolor='black', alpha=0.7, linewidth=2)
    ax.add_patch(rect)
    ax.text(3, sa_y, r'Attention = softmax$(QK^T/\sqrt{d})V$',
            ha='center', va='center', fontsize=7.5)

    # Arrows to attention
    for x_pos in [1, 3, 5]:
        ax.annotate('', xy=(3, sa_y+0.4), xytext=(x_pos, sa_y+0.8),
                   arrowprops=dict(arrowstyle='->', lw=1, color='black'))

    # Output
    sa_y = 2.5
    ax.annotate('', xy=(3, sa_y+0.5), xytext=(3, 3.6),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    rect = mpatches.FancyBboxPatch((1.5, sa_y-0.3), 3, 0.6,
                                   boxstyle="round,pad=0.05",
                                   facecolor=COLORS['success'],
                                   edgecolor='black', alpha=0.6, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(3, sa_y, 'Attended Features', ha='center', va='center', fontsize=8)

    # Benefits box
    benefits_text = [
        r'\textbf{Benefits:}',
        r'$\bullet$ Long-range dependencies',
        r'$\bullet$ Global context modeling',
        r'$\bullet$ Adaptive feature weighting',
    ]
    y_pos = 1.2
    for text in benefits_text:
        ax.text(3, y_pos, text, ha='center', fontsize=7)
        y_pos -= 0.3

    fig.subplots_adjust(top=0.92)
    plt.tight_layout()
    save_figure(fig, 'fig13_attention_mechanisms', OUTPUT_DIR)
    plt.close()


def generate_parameter_breakdown():
    """
    Figure: Parameter Count and Model Complexity

    Shows parameter distribution across model components.
    """
    print("Generating: Parameter Breakdown...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(r'\textbf{Model Complexity Analysis}', y=1.02)

    # SA-CycleGAN parameter breakdown (pie chart)
    ax = axes[0]
    components = ['Generator A→B\n(11.68M)', 'Generator B→A\n(11.68M)',
                  'Discriminator A\n(5.87M)', 'Discriminator B\n(5.87M)']
    sizes = [11.68, 11.68, 5.87, 5.87]
    colors_list = [COLORS['primary'], COLORS['secondary'],
                   COLORS['success'], COLORS['warning']]

    wedges, texts, autotexts = ax.pie(sizes, labels=components, colors=colors_list,
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 8})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)

    ax.set_title(r'(a) SA-CycleGAN-2.5D (35.1M total)', fontsize=10)

    # Comparison bar chart
    ax = axes[1]
    models = ['Baseline\nCycleGAN', 'SA-CycleGAN\n(Ours)']
    params = [33.88, 35.1]
    colors_list = [COLORS['gray'], COLORS['danger']]

    bars = ax.barh(models, params, color=colors_list, alpha=0.7, height=0.5)
    ax.set_xlabel('Parameters (Millions)')
    ax.set_title(r'(b) Model Size Comparison', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, val in zip(bars, params):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                f'{val:.2f}M',
                ha='left', va='center', fontsize=9)

    # Add difference annotation
    diff = params[1] - params[0]
    pct_increase = (diff / params[0]) * 100
    ax.text(0.98, 0.05, f'+{diff:.2f}M (+{pct_increase:.1f}\%) for attention',
           transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.4))

    fig.subplots_adjust(top=0.92)
    plt.tight_layout()
    save_figure(fig, 'fig14_parameter_breakdown', OUTPUT_DIR)
    plt.close()


def generate_cyclegan_workflow():
    """
    Figure: Complete CycleGAN Training Workflow

    Shows the full training loop with forward/backward cycles.
    """
    print("Generating: CycleGAN Workflow...")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    fig.suptitle(r'\textbf{CycleGAN Training Workflow}', y=0.96)

    # Domain A (top)
    a_y = 8.5
    rect_a = mpatches.Rectangle((1, a_y-0.4), 2, 0.8,
                                facecolor=COLORS['primary'],
                                edgecolor='black', alpha=0.6, linewidth=2)
    ax.add_patch(rect_a)
    ax.text(2, a_y, r'Real $A$\\ (BraTS)', ha='center', va='center', fontsize=9, fontweight='bold')

    # Domain B (top)
    b_y = 8.5
    rect_b = mpatches.Rectangle((9, b_y-0.4), 2, 0.8,
                                facecolor=COLORS['secondary'],
                                edgecolor='black', alpha=0.6, linewidth=2)
    ax.add_patch(rect_b)
    ax.text(10, b_y, r'Real $B$\\ (UPenn)', ha='center', va='center', fontsize=9, fontweight='bold')

    # Generator A→B
    gen_ab_y = 6.5
    rect_gab = mpatches.FancyBboxPatch((4, gen_ab_y-0.4), 2, 0.8,
                                       boxstyle="round,pad=0.1",
                                       facecolor=COLORS['success'],
                                       edgecolor='black', alpha=0.7, linewidth=2)
    ax.add_patch(rect_gab)
    ax.text(5, gen_ab_y, r'$G_{A \rightarrow B}$', ha='center', va='center', fontsize=10)

    # Arrow A → G_AB
    ax.annotate('', xy=(4, gen_ab_y), xytext=(2.5, a_y-0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Fake B
    fake_b_y = 6.5
    rect_fb = mpatches.Rectangle((7, fake_b_y-0.3), 1.5, 0.6,
                                 facecolor=COLORS['secondary'],
                                 edgecolor='black', alpha=0.4, linewidth=2,
                                 linestyle='--')
    ax.add_patch(rect_fb)
    ax.text(7.75, fake_b_y, r'Fake $B$', ha='center', va='center', fontsize=8)

    # Arrow G_AB → Fake B
    ax.annotate('', xy=(7, fake_b_y), xytext=(6, gen_ab_y),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Generator B→A
    gen_ba_y = 4.5
    rect_gba = mpatches.FancyBboxPatch((4, gen_ba_y-0.4), 2, 0.8,
                                       boxstyle="round,pad=0.1",
                                       facecolor=COLORS['warning'],
                                       edgecolor='black', alpha=0.7, linewidth=2)
    ax.add_patch(rect_gba)
    ax.text(5, gen_ba_y, r'$G_{B \rightarrow A}$', ha='center', va='center', fontsize=10)

    # Arrow Fake B → G_BA (cycle)
    ax.annotate('', xy=(5, gen_ba_y+0.4), xytext=(7.75, fake_b_y-0.4),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text(6.5, 5.3, 'Cycle', fontsize=7, color='red', ha='center')

    # Reconstructed A
    rec_a_y = 4.5
    rect_ra = mpatches.Rectangle((1.5, rec_a_y-0.3), 1.5, 0.6,
                                 facecolor=COLORS['primary'],
                                 edgecolor='black', alpha=0.4, linewidth=2,
                                 linestyle='--')
    ax.add_patch(rect_ra)
    ax.text(2.25, rec_a_y, r'Rec. $A$', ha='center', va='center', fontsize=8)

    # Arrow G_BA → Rec A
    ax.annotate('', xy=(3, rec_a_y), xytext=(4, gen_ba_y),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Discriminators
    disc_a_y = 3
    rect_da = mpatches.Ellipse((1.5, disc_a_y), 1.2, 0.6,
                               facecolor=COLORS['info'],
                               edgecolor='black', alpha=0.6, linewidth=1.5)
    ax.add_patch(rect_da)
    ax.text(1.5, disc_a_y, r'$D_A$', ha='center', va='center', fontsize=9, fontweight='bold')

    disc_b_y = 3
    rect_db = mpatches.Ellipse((9.5, disc_b_y), 1.2, 0.6,
                               facecolor=COLORS['info'],
                               edgecolor='black', alpha=0.6, linewidth=1.5)
    ax.add_patch(rect_db)
    ax.text(9.5, disc_b_y, r'$D_B$', ha='center', va='center', fontsize=9, fontweight='bold')

    # Discriminator connections
    ax.annotate('', xy=(1.5, disc_a_y+0.35), xytext=(2, a_y-0.5),
               arrowprops=dict(arrowstyle='->', lw=1, color='green'))
    ax.annotate('', xy=(1.5, disc_a_y+0.35), xytext=(2.25, rec_a_y-0.4),
               arrowprops=dict(arrowstyle='->', lw=1, color='red', linestyle='--'))

    # Losses
    loss_y = 1.5
    losses = [
        r'$\mathcal{L}_{\text{GAN}}$: Adversarial',
        r'$\mathcal{L}_{\text{cycle}}$: Cycle consistency',
        r'$\mathcal{L}_{\text{identity}}$: Identity mapping',
        r'$\mathcal{L}_{\text{SSIM}}$: Perceptual quality',
    ]
    for i, loss in enumerate(losses):
        ax.text(1, loss_y - i*0.35, loss, ha='left', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.5))

    # Total loss
    ax.text(6, 0.5, r'$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GAN}} + \lambda_{\text{cycle}} \mathcal{L}_{\text{cycle}} + \lambda_{\text{identity}} \mathcal{L}_{\text{identity}} + \lambda_{\text{SSIM}} \mathcal{L}_{\text{SSIM}}$',
           ha='center', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))

    fig.subplots_adjust(top=0.92)
    plt.tight_layout()
    save_figure(fig, 'fig15_cyclegan_workflow', OUTPUT_DIR)
    plt.close()


def main():
    """Generate all architecture figures."""
    print("="*60)
    print("Generating Architecture and Workflow Figures")
    print("="*60)

    generate_architecture_comparison()
    generate_attention_mechanism_diagram()
    generate_parameter_breakdown()
    generate_cyclegan_workflow()

    print("\n" + "="*60)
    print("Architecture figures generation complete!")
    print(f"Figures saved to: {OUTPUT_DIR}/")
    print("="*60)


if __name__ == '__main__':
    main()
