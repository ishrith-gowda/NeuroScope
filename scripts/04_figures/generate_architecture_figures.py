#!/usr/bin/env python3
"""
generate architecture and model diagrams for publication

creates comprehensive visualizations of:
- model architecture comparison (sa-cyclegan vs baseline)
- attention mechanism diagrams
- network structure details
- parameter breakdown

author: neuroscope research team
date: january 2026
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
    figure: sa-cyclegan vs baseline architecture comparison

    shows side-by-side comparison highlighting attention mechanisms.
    """
    print("generating: architecture comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plt.subplots_adjust(wspace=0.25, top=0.88, bottom=0.12)

    # color palette: oxford blue, emerald, crimson, ash
    c_encoder = '#002147'   # oxford blue
    c_resblock = '#50C878'  # emerald
    c_cbam = '#2D8B4E'      # darker emerald for cbam
    c_attention = '#DC143C'  # crimson
    c_gray = '#8B8C89'      # ash gray

    # baseline cyclegan (left)
    ax = axes[0]
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title(r'(a) Baseline CycleGAN-2.5D', fontsize=14, pad=10)

    baseline_blocks = [
        {'name': 'Input\n[B,12,H,W]', 'y': 9, 'color': c_gray},
        {'name': 'Encoder\n(3 conv)', 'y': 7.5, 'color': c_encoder},
        {'name': 'ResBlock 1-3', 'y': 6.5, 'color': c_resblock},
        {'name': 'ResBlock 4-6', 'y': 5.5, 'color': c_resblock},
        {'name': 'ResBlock 7-9', 'y': 4.5, 'color': c_resblock},
        {'name': 'Decoder\n(3 transconv)', 'y': 3, 'color': c_encoder},
        {'name': 'Output\n[B,4,H,W]', 'y': 1.5, 'color': c_gray},
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
                ha='center', va='center', fontsize=10)

        if block != baseline_blocks[-1]:
            ax.annotate('', xy=(2.5, block['y']-0.4), xytext=(2.5, block['y']-0.6),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    ax.text(2.5, 0.5, r'Parameters: 33.88M', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

    # sa-cyclegan (right)
    ax = axes[1]
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title(r'(b) SA-CycleGAN-2.5D (Ours)', fontsize=14, pad=10)

    sa_blocks = [
        {'name': 'Input\n[B,12,H,W]', 'y': 9, 'color': c_gray},
        {'name': 'Encoder\n(3 conv)', 'y': 7.5, 'color': c_encoder},
        {'name': 'ResBlock+CBAM 1', 'y': 6.7, 'color': c_cbam},
        {'name': 'ResBlock+CBAM 2', 'y': 6.1, 'color': c_cbam},
        {'name': 'ResBlock+CBAM 3', 'y': 5.5, 'color': c_cbam},
        {'name': 'Self-Attention', 'y': 4.8, 'color': c_attention, 'highlight': True},
        {'name': 'ResBlock+CBAM 4-6', 'y': 4, 'color': c_cbam},
        {'name': 'ResBlock+CBAM 7-9', 'y': 3.2, 'color': c_cbam},
        {'name': 'Decoder\n(3 transconv)', 'y': 2, 'color': c_encoder},
        {'name': 'Output\n[B,4,H,W]', 'y': 0.8, 'color': c_gray},
    ]

    for block in sa_blocks:
        if block.get('highlight'):
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
                ha='center', va='center', fontsize=9)

        if block != sa_blocks[-1]:
            ax.annotate('', xy=(2.5, block['y']-0.3), xytext=(2.5, block['y']-0.45),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    ax.text(2.5, 0.2, r'Parameters: 35.1M (+3.6\%)', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # legend
    legend_elements = [
        mpatches.Patch(facecolor=c_resblock, alpha=0.6, label='Standard ResBlock'),
        mpatches.Patch(facecolor=c_cbam, alpha=0.6, label='ResBlock + CBAM'),
        mpatches.Patch(facecolor=c_attention, alpha=0.7, edgecolor='red',
                      linewidth=2, label='Self-Attention'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
              bbox_to_anchor=(0.5, 0.01), fontsize=11,
              frameon=True, fancybox=False, edgecolor='black')

    fig.suptitle(r'\textbf{Model Architecture Comparison}',
                 fontsize=16, y=0.98)
    save_figure(fig, 'fig12_architecture_comparison', OUTPUT_DIR)
    plt.close()


def generate_attention_mechanism_diagram():
    """
    figure: attention mechanisms detailed diagram

    shows cbam and self-attention structure in detail.
    """
    print("generating: attention mechanisms diagram...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plt.subplots_adjust(wspace=0.25, top=0.88)

    # color palette: midnight blue, byzantium, jade, flame
    c_input = '#8B8C89'      # ash
    c_channel = '#191970'    # midnight blue
    c_spatial = '#702963'    # byzantium
    c_output = '#00A86B'     # jade
    c_qkv = '#3F51B5'       # indigo blue
    c_attn = '#E25822'       # flame

    # cbam (left)
    ax = axes[0]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title(r'(a) CBAM (Channel and Spatial)', fontsize=14, pad=10)

    cbam_blocks = [
        {'name': 'Input\nFeatures', 'y': 7, 'x': 3, 'w': 2, 'color': c_input},
        {'name': 'Channel\nAttention', 'y': 5.5, 'x': 3, 'w': 2, 'color': c_channel},
        {'name': 'Refined\nFeatures', 'y': 4.5, 'x': 3, 'w': 2, 'color': c_input},
        {'name': 'Spatial\nAttention', 'y': 3, 'x': 3, 'w': 2, 'color': c_spatial},
        {'name': 'Output\nFeatures', 'y': 1.5, 'x': 3, 'w': 2, 'color': c_output},
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
                ha='center', va='center', fontsize=10)

        if i < len(cbam_blocks) - 1:
            ax.annotate('', xy=(3, block['y']-0.4), xytext=(3, block['y']-0.7),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    ax.text(0.5, 5.5, r'$\otimes$', fontsize=18, ha='center', va='center')
    ax.text(0.5, 3, r'$\otimes$', fontsize=18, ha='center', va='center')
    ax.text(0.3, 5.5, 'Multiply', fontsize=9, ha='right', va='center', rotation=90)
    ax.text(0.3, 3, 'Multiply', fontsize=9, ha='right', va='center', rotation=90)

    # self-attention (right)
    ax = axes[1]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title(r'(b) Self-Attention (Bottleneck)', fontsize=14, pad=10)

    sa_y = 7
    rect = mpatches.FancyBboxPatch((1.5, sa_y-0.3), 3, 0.6,
                                   boxstyle="round,pad=0.05",
                                   facecolor=c_input,
                                   edgecolor='black', alpha=0.6, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(3, sa_y, 'Input [B,C,H,W]', ha='center', va='center', fontsize=10)

    sa_y = 5.5
    for i, (label, x_pos) in enumerate([('Q', 1), ('K', 3), ('V', 5)]):
        rect = mpatches.FancyBboxPatch((x_pos-0.4, sa_y-0.2), 0.8, 0.4,
                                       boxstyle="round,pad=0.03",
                                       facecolor=c_qkv,
                                       edgecolor='black', alpha=0.6, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x_pos, sa_y, label, ha='center', va='center', fontsize=11, fontweight='bold')

        ax.annotate('', xy=(x_pos, sa_y+0.3), xytext=(x_pos, 6.5),
                   arrowprops=dict(arrowstyle='->', lw=1, color='black'))

    sa_y = 4
    rect = mpatches.FancyBboxPatch((1, sa_y-0.3), 4, 0.6,
                                   boxstyle="round,pad=0.05",
                                   facecolor=c_attn,
                                   edgecolor='black', alpha=0.7, linewidth=2)
    ax.add_patch(rect)
    ax.text(3, sa_y, r'Attention = softmax$(QK^T/\sqrt{d})V$',
            ha='center', va='center', fontsize=9)

    for x_pos in [1, 3, 5]:
        ax.annotate('', xy=(3, sa_y+0.4), xytext=(x_pos, sa_y+0.8),
                   arrowprops=dict(arrowstyle='->', lw=1, color='black'))

    sa_y = 2.5
    ax.annotate('', xy=(3, sa_y+0.5), xytext=(3, 3.6),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    rect = mpatches.FancyBboxPatch((1.5, sa_y-0.3), 3, 0.6,
                                   boxstyle="round,pad=0.05",
                                   facecolor=c_output,
                                   edgecolor='black', alpha=0.6, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(3, sa_y, 'Attended Features', ha='center', va='center', fontsize=10)

    benefits_text = [
        r'\textbf{Benefits:}',
        r'$\bullet$ Long-range dependencies',
        r'$\bullet$ Global context modeling',
        r'$\bullet$ Adaptive feature weighting',
    ]
    y_pos = 1.2
    for text in benefits_text:
        ax.text(3, y_pos, text, ha='center', fontsize=9)
        y_pos -= 0.35

    fig.suptitle(r'\textbf{Attention Mechanisms in SA-CycleGAN-2.5D}',
                 fontsize=16, y=0.98)
    save_figure(fig, 'fig13_attention_mechanisms', OUTPUT_DIR)
    plt.close()


def generate_parameter_breakdown():
    """
    figure: parameter count and model complexity

    shows parameter distribution across model components.
    """
    print("generating: parameter breakdown...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(wspace=0.35, top=0.85)

    # lighter, publication-quality colors
    c_gen_ab = '#7BB3D0'   # soft blue
    c_gen_ba = '#F0A860'   # warm peach
    c_disc_a = '#8DC58A'   # soft green
    c_disc_b = '#D4A0C0'   # soft mauve

    # (a) sa-cyclegan parameter breakdown (pie chart)
    ax = axes[0]
    components = [r'Generator A$\rightarrow$B' + '\n(11.68M)',
                  r'Generator B$\rightarrow$A' + '\n(11.68M)',
                  'Discriminator A\n(5.87M)',
                  'Discriminator B\n(5.87M)']
    sizes = [11.68, 11.68, 5.87, 5.87]
    colors_list = [c_gen_ab, c_gen_ba, c_disc_a, c_disc_b]

    wedges, texts, autotexts = ax.pie(sizes, labels=components, colors=colors_list,
                                       autopct=r'%1.1f\%%', startangle=90,
                                       textprops={'fontsize': 11},
                                       wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)

    ax.set_title(r'(a) SA-CycleGAN-2.5D (35.1M total)', fontsize=13, pad=12)

    # (b) comparison bar chart
    ax = axes[1]
    models = ['Baseline\nCycleGAN', 'SA-CycleGAN-2.5D']
    params = [33.88, 35.1]
    bar_colors = ['#B0B0B0', c_gen_ab]

    bars = ax.barh(models, params, color=bar_colors, height=0.65,
                   edgecolor='black', linewidth=0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('Parameters (Millions)', fontsize=13)
    ax.set_title(r'(b) Model Size Comparison', fontsize=13, pad=12)
    ax.set_xlim(0, 42)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)

    for bar, val in zip(bars, params):
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
                f'{val:.2f}M',
                ha='left', va='center', fontsize=11)

    diff = params[1] - params[0]
    pct_increase = (diff / params[0]) * 100
    ax.text(0.98, 0.05, r'+' + f'{diff:.2f}M (+{pct_increase:.1f}' + r'\%) for attention',
           transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='#D4ECD0', alpha=0.6))

    fig.suptitle(r'\textbf{Model Complexity Analysis}',
                 fontsize=15, y=0.97)
    fig.savefig(OUTPUT_DIR / 'fig14_parameter_breakdown.pdf',
                format='pdf', bbox_inches='tight', pad_inches=0.15)
    plt.close()


def generate_cyclegan_workflow():
    """
    figure: complete cyclegan training workflow

    shows the full training loop with forward/backward cycles.
    """
    print("generating: cyclegan workflow...")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    fig.suptitle(r'\textbf{CycleGAN Training Workflow}',
                 fontsize=16, y=0.98)

    # color palette: prussian blue, persian rose, chartreuse, amber, sky
    c_dom_a = '#003153'    # prussian blue
    c_dom_b = '#CC3366'    # persian rose
    c_gen = '#7FFF00'      # chartreuse
    c_gen2 = '#FFBF00'     # amber
    c_disc = '#87CEEB'     # sky blue

    # domain a (top)
    a_y = 8.5
    rect_a = mpatches.Rectangle((1, a_y-0.4), 2, 0.8,
                                facecolor=c_dom_a,
                                edgecolor='black', alpha=0.6, linewidth=2)
    ax.add_patch(rect_a)
    ax.text(2, a_y, r'Real $A$\\ (BraTS)', ha='center', va='center', fontsize=10, fontweight='bold')

    # domain b (top)
    b_y = 8.5
    rect_b = mpatches.Rectangle((9, b_y-0.4), 2, 0.8,
                                facecolor=c_dom_b,
                                edgecolor='black', alpha=0.6, linewidth=2)
    ax.add_patch(rect_b)
    ax.text(10, b_y, r'Real $B$\\ (UPenn)', ha='center', va='center', fontsize=10, fontweight='bold')

    # generator a→b
    gen_ab_y = 6.5
    rect_gab = mpatches.FancyBboxPatch((4, gen_ab_y-0.4), 2, 0.8,
                                       boxstyle="round,pad=0.1",
                                       facecolor=c_gen,
                                       edgecolor='black', alpha=0.7, linewidth=2)
    ax.add_patch(rect_gab)
    ax.text(5, gen_ab_y, r'$G_{A \rightarrow B}$', ha='center', va='center', fontsize=12)

    # arrow a → g_ab
    ax.annotate('', xy=(4, gen_ab_y), xytext=(2.5, a_y-0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # fake b
    fake_b_y = 6.5
    rect_fb = mpatches.Rectangle((7, fake_b_y-0.3), 1.5, 0.6,
                                 facecolor=c_dom_b,
                                 edgecolor='black', alpha=0.4, linewidth=2,
                                 linestyle='--')
    ax.add_patch(rect_fb)
    ax.text(7.75, fake_b_y, r'Fake $B$', ha='center', va='center', fontsize=10)

    # arrow g_ab → fake b
    ax.annotate('', xy=(7, fake_b_y), xytext=(6, gen_ab_y),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # generator b→a
    gen_ba_y = 4.5
    rect_gba = mpatches.FancyBboxPatch((4, gen_ba_y-0.4), 2, 0.8,
                                       boxstyle="round,pad=0.1",
                                       facecolor=c_gen2,
                                       edgecolor='black', alpha=0.7, linewidth=2)
    ax.add_patch(rect_gba)
    ax.text(5, gen_ba_y, r'$G_{B \rightarrow A}$', ha='center', va='center', fontsize=12)

    # arrow fake b → g_ba (cycle)
    ax.annotate('', xy=(5, gen_ba_y+0.4), xytext=(7.75, fake_b_y-0.4),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text(6.5, 5.3, 'Cycle', fontsize=9, color='red', ha='center')

    # reconstructed a
    rec_a_y = 4.5
    rect_ra = mpatches.Rectangle((1.5, rec_a_y-0.3), 1.5, 0.6,
                                 facecolor=c_dom_a,
                                 edgecolor='black', alpha=0.4, linewidth=2,
                                 linestyle='--')
    ax.add_patch(rect_ra)
    ax.text(2.25, rec_a_y, r'Rec. $A$', ha='center', va='center', fontsize=10)

    # arrow g_ba → rec a
    ax.annotate('', xy=(3, rec_a_y), xytext=(4, gen_ba_y),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # discriminators
    disc_a_y = 3
    rect_da = mpatches.Ellipse((1.5, disc_a_y), 1.2, 0.6,
                               facecolor=c_disc,
                               edgecolor='black', alpha=0.6, linewidth=1.5)
    ax.add_patch(rect_da)
    ax.text(1.5, disc_a_y, r'$D_A$', ha='center', va='center', fontsize=11, fontweight='bold')

    disc_b_y = 3
    rect_db = mpatches.Ellipse((9.5, disc_b_y), 1.2, 0.6,
                               facecolor=c_disc,
                               edgecolor='black', alpha=0.6, linewidth=1.5)
    ax.add_patch(rect_db)
    ax.text(9.5, disc_b_y, r'$D_B$', ha='center', va='center', fontsize=11, fontweight='bold')

    # discriminator connections
    ax.annotate('', xy=(1.5, disc_a_y+0.35), xytext=(2, a_y-0.5),
               arrowprops=dict(arrowstyle='->', lw=1, color='green'))
    ax.annotate('', xy=(1.5, disc_a_y+0.35), xytext=(2.25, rec_a_y-0.4),
               arrowprops=dict(arrowstyle='->', lw=1, color='red', linestyle='--'))

    # losses
    loss_y = 1.5
    losses = [
        r'$\mathcal{L}_{\text{GAN}}$: Adversarial',
        r'$\mathcal{L}_{\text{cycle}}$: Cycle consistency',
        r'$\mathcal{L}_{\text{identity}}$: Identity mapping',
        r'$\mathcal{L}_{\text{SSIM}}$: Perceptual quality',
    ]
    for i, loss in enumerate(losses):
        ax.text(1, loss_y - i*0.38, loss, ha='left', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.5))

    # total loss
    ax.text(6, 0.5, r'$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GAN}} + \lambda_{\text{cycle}} \mathcal{L}_{\text{cycle}} + \lambda_{\text{identity}} \mathcal{L}_{\text{identity}} + \lambda_{\text{SSIM}} \mathcal{L}_{\text{SSIM}}$',
           ha='center', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#E0ECFF', alpha=0.5))
    save_figure(fig, 'fig15_cyclegan_workflow', OUTPUT_DIR)
    plt.close()


def main():
    """generate all architecture figures."""
    print("="*60)
    print("generating architecture and workflow figures")
    print("="*60)

    generate_architecture_comparison()
    generate_attention_mechanism_diagram()
    generate_parameter_breakdown()
    generate_cyclegan_workflow()

    print("\n" + "="*60)
    print("architecture figures generation complete!")
    print(f"figures saved to: {OUTPUT_DIR}/")
    print("="*60)


if __name__ == '__main__':
    main()
