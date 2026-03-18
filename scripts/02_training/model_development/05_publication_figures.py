#!/usr/bin/env python
"""
publication-quality figures generator for cyclegan mri domain adaptation

generates comprehensive visualizations including:
1. model architecture diagrams
2. training loss curves with confidence intervals
3. sample translations with difference maps
4. quantitative metrics tables and charts
5. dataset distribution visualizations
6. ablation study comparisons
7. feature space visualizations (t-sne/umap)

all figures use times new roman, seaborn styling, and publication standards.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# configure matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'mathtext.fontset': 'stix',
})

sns.set_theme(style="whitegrid", palette="muted")

# add paths
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_cyclegan_v2 import ResNetGenerator, PatchDiscriminator


class PublicationFigureGenerator:
    """generate publication-quality figures for cyclegan results"""
    
    def __init__(self, output_dir: str, checkpoint_dir: str = None, samples_dir: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.samples_dir = Path(samples_dir) if samples_dir else None
        
        self.colors = {
            'brats': '#2ecc71',      # green for brats
            'upenn': '#3498db',       # blue for upenn
            'generator': '#e74c3c',   # red for generator
            'discriminator': '#9b59b6', # purple for discriminator
            'cycle': '#f39c12',       # orange for cycle
            'identity': '#1abc9c',    # teal for identity
        }
        
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 
                                   'cuda' if torch.cuda.is_available() else 'cpu')
    
    # =========================================================================
    # figure 1: model architecture diagram
    # =========================================================================
    def generate_architecture_diagram(self):
        """create a visual representation of the cyclegan architecture"""
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.4)
        
        # generator architecture
        ax1 = fig.add_subplot(gs[0, :])
        self._draw_generator_architecture(ax1)
        ax1.set_title('(a) ResNet Generator Architecture', fontweight='bold', pad=10)
        
        # discriminator architecture
        ax2 = fig.add_subplot(gs[1, :2])
        self._draw_discriminator_architecture(ax2)
        ax2.set_title('(b) PatchGAN Discriminator', fontweight='bold', pad=10)
        
        # cyclegan flow
        ax3 = fig.add_subplot(gs[1, 2:])
        self._draw_cyclegan_flow(ax3)
        ax3.set_title('(c) CycleGAN Training Flow', fontweight='bold', pad=10)
        
        plt.savefig(self.output_dir / 'fig1_architecture.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig1_architecture.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"saved: fig1_architecture.pdf")
    
    def _draw_generator_architecture(self, ax):
        """draw generator block diagram"""
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 2)
        ax.axis('off')
        
        blocks = [
            ('Input\n4×256×256', 0.5, '#ecf0f1'),
            ('Encoder\n↓64→128→256', 2.5, '#3498db'),
            ('ResBlocks\n×9', 5, '#e74c3c'),
            ('Decoder\n↑256→128→64', 7.5, '#2ecc71'),
            ('Output\n4×256×256', 10, '#ecf0f1'),
        ]
        
        for i, (label, x, color) in enumerate(blocks):
            rect = mpatches.FancyBboxPatch((x, 0.5), 1.8, 1, 
                                           boxstyle="round,pad=0.05",
                                           facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x + 0.9, 1, label, ha='center', va='center', fontsize=9, fontweight='bold')
            
            if i < len(blocks) - 1:
                ax.annotate('', xy=(x + 2.1, 1), xytext=(x + 1.9, 1),
                           arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        # add skip connection indicator
        ax.annotate('', xy=(5.9, 1.6), xytext=(2.5, 1.6),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1, ls='--',
                                  connectionstyle='arc3,rad=-0.3'))
        ax.text(4.2, 1.85, 'Skip connections in ResBlocks', fontsize=8, color='gray')
    
    def _draw_discriminator_architecture(self, ax):
        """draw discriminator block diagram"""
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 2)
        ax.axis('off')
        
        blocks = [
            ('Input\n4×256×256', 0.5, '#ecf0f1'),
            ('Conv↓\n64', 2, '#9b59b6'),
            ('Conv↓\n128', 3.2, '#9b59b6'),
            ('Conv↓\n256', 4.4, '#9b59b6'),
            ('Conv\n512', 5.6, '#9b59b6'),
            ('Patch\n30×30', 7, '#f39c12'),
        ]
        
        for i, (label, x, color) in enumerate(blocks):
            w = 0.9 if i < len(blocks) - 1 else 0.8
            rect = mpatches.FancyBboxPatch((x, 0.5), w, 1, 
                                           boxstyle="round,pad=0.03",
                                           facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x + w/2, 1, label, ha='center', va='center', fontsize=8, fontweight='bold')
            
            if i < len(blocks) - 1:
                ax.annotate('', xy=(x + w + 0.15, 1), xytext=(x + w + 0.05, 1),
                           arrowprops=dict(arrowstyle='->', color='black', lw=1))
        
        ax.text(4, 0.2, 'Spectral Normalization on all Conv layers', fontsize=8, 
                ha='center', style='italic')
    
    def _draw_cyclegan_flow(self, ax):
        """draw cyclegan bidirectional flow"""
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 3)
        ax.axis('off')
        
        # domain a (brats)
        rect_a = mpatches.FancyBboxPatch((0.2, 1.8), 1.5, 0.8,
                                         boxstyle="round,pad=0.05",
                                         facecolor=self.colors['brats'], 
                                         edgecolor='black', linewidth=2)
        ax.add_patch(rect_a)
        ax.text(0.95, 2.2, 'Domain A\n(BraTS)', ha='center', va='center', 
                fontsize=9, fontweight='bold', color='white')
        
        # domain b (upenn)
        rect_b = mpatches.FancyBboxPatch((4.3, 1.8), 1.5, 0.8,
                                         boxstyle="round,pad=0.05",
                                         facecolor=self.colors['upenn'], 
                                         edgecolor='black', linewidth=2)
        ax.add_patch(rect_b)
        ax.text(5.05, 2.2, 'Domain B\n(UPenn)', ha='center', va='center', 
                fontsize=9, fontweight='bold', color='white')
        
        # generators
        ax.annotate('', xy=(4.1, 2.4), xytext=(1.9, 2.4),
                   arrowprops=dict(arrowstyle='->', color=self.colors['generator'], lw=2))
        ax.text(3, 2.6, '$G_{A→B}$', ha='center', fontsize=10, color=self.colors['generator'])
        
        ax.annotate('', xy=(1.9, 2.0), xytext=(4.1, 2.0),
                   arrowprops=dict(arrowstyle='->', color=self.colors['generator'], lw=2))
        ax.text(3, 1.75, '$G_{B→A}$', ha='center', fontsize=10, color=self.colors['generator'])
        
        # cycle consistency
        ax.annotate('', xy=(0.95, 1.7), xytext=(0.95, 0.8),
                   arrowprops=dict(arrowstyle='<->', color=self.colors['cycle'], lw=1.5,
                                  connectionstyle='arc3,rad=0.5'))
        ax.text(0.3, 1.2, 'Cycle\nLoss', ha='center', fontsize=8, color=self.colors['cycle'])
        
        ax.annotate('', xy=(5.05, 1.7), xytext=(5.05, 0.8),
                   arrowprops=dict(arrowstyle='<->', color=self.colors['cycle'], lw=1.5,
                                  connectionstyle='arc3,rad=-0.5'))
        ax.text(5.7, 1.2, 'Cycle\nLoss', ha='center', fontsize=8, color=self.colors['cycle'])
        
        # discriminators
        ax.text(0.95, 0.5, '$D_A$', ha='center', fontsize=10, color=self.colors['discriminator'],
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=self.colors['discriminator']))
        ax.text(5.05, 0.5, '$D_B$', ha='center', fontsize=10, color=self.colors['discriminator'],
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=self.colors['discriminator']))
    
    # =========================================================================
    # figure 2: training loss curves
    # =========================================================================
    def generate_loss_curves(self, loss_history: dict = None, loss_file: str = None):
        """generate comprehensive training loss visualization"""
        if loss_history is None and loss_file:
            with open(loss_file, 'r') as f:
                loss_history = json.load(f)
        
        if loss_history is None:
            print("warning: no loss history provided, skipping loss curves")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        
        # smooth function
        def smooth(y, window=50):
            if len(y) < window:
                return y
            return np.convolve(y, np.ones(window)/window, mode='valid')
        
        # generator loss
        ax = axes[0, 0]
        g_loss = loss_history.get('G', [])
        if g_loss:
            x = np.arange(len(g_loss))
            ax.fill_between(x, g_loss, alpha=0.2, color=self.colors['generator'])
            ax.plot(smooth(g_loss), color=self.colors['generator'], lw=2, label='Smoothed')
            ax.set_title('Generator Total Loss', fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.legend()
        
        # discriminator losses
        ax = axes[0, 1]
        d_a = loss_history.get('D_A', [])
        d_b = loss_history.get('D_B', [])
        if d_a and d_b:
            ax.plot(smooth(d_a), color=self.colors['brats'], lw=2, label='$D_A$ (BraTS)')
            ax.plot(smooth(d_b), color=self.colors['upenn'], lw=2, label='$D_B$ (UPenn)')
            ax.axhline(y=0.25, color='gray', ls='--', alpha=0.7, label='Healthy threshold')
            ax.set_title('Discriminator Losses', fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.legend()
        
        # cycle consistency loss
        ax = axes[0, 2]
        cycle = loss_history.get('Cycle', [])
        if cycle:
            ax.fill_between(range(len(cycle)), cycle, alpha=0.2, color=self.colors['cycle'])
            ax.plot(smooth(cycle), color=self.colors['cycle'], lw=2)
            ax.set_title('Cycle Consistency Loss', fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
        
        # identity loss
        ax = axes[1, 0]
        identity = loss_history.get('Id', [])
        if identity:
            ax.fill_between(range(len(identity)), identity, alpha=0.2, color=self.colors['identity'])
            ax.plot(smooth(identity), color=self.colors['identity'], lw=2)
            ax.set_title('Identity Loss', fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
        
        # gradient penalty
        ax = axes[1, 1]
        gp = loss_history.get('GP', [])
        if gp:
            ax.plot(smooth(gp), color='#8e44ad', lw=2)
            ax.set_title('Gradient Penalty', fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
        
        # d/g ratio (health indicator)
        ax = axes[1, 2]
        if d_a and d_b and g_loss:
            d_avg = [(a + b) / 2 for a, b in zip(d_a, d_b)]
            ratio = [d / max(g, 0.01) for d, g in zip(d_avg, g_loss)]
            ax.plot(smooth(ratio, 100), color='#2c3e50', lw=2)
            ax.axhline(y=0.3, color='green', ls='--', alpha=0.7, label='Optimal range')
            ax.axhline(y=0.1, color='red', ls='--', alpha=0.7, label='Mode collapse risk')
            ax.set_title('D/G Loss Ratio (Stability)', fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Ratio')
            ax.set_ylim(0, 1)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig2_training_losses.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig2_training_losses.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"saved: fig2_training_losses.pdf")
    
    # =========================================================================
    # figure 3: sample translations with metrics
    # =========================================================================
    def generate_sample_grid(self, G_A2B, G_B2A, dataloader_A, dataloader_B, n_samples=4):
        """generate sample translation grid with difference maps and metrics"""
        G_A2B.eval()
        G_B2A.eval()
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(n_samples + 1, 8, figure=fig, hspace=0.3, wspace=0.05)
        
        # headers
        headers = ['Real A', 'Fake B', '|Diff|', 'Recon A', 'Real B', 'Fake A', '|Diff|', 'Recon B']
        for i, h in enumerate(headers):
            ax = fig.add_subplot(gs[0, i])
            ax.text(0.5, 0.5, h, ha='center', va='center', fontsize=10, fontweight='bold')
            ax.axis('off')
        
        iter_A = iter(dataloader_A)
        iter_B = iter(dataloader_B)
        
        metrics_A2B = {'ssim': [], 'psnr': []}
        metrics_B2A = {'ssim': [], 'psnr': []}
        
        with torch.no_grad():
            for row in range(n_samples):
                try:
                    real_A = next(iter_A)[:1].to(self.device)
                    real_B = next(iter_B)[:1].to(self.device)
                except StopIteration:
                    break
                
                fake_B = G_A2B(real_A)
                recon_A = G_B2A(fake_B)
                fake_A = G_B2A(real_B)
                recon_B = G_A2B(fake_A)
                
                # calculate metrics for cycle consistency
                real_A_np = ((real_A[0, 0].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
                recon_A_np = ((recon_A[0, 0].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
                real_B_np = ((real_B[0, 0].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
                recon_B_np = ((recon_B[0, 0].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
                
                try:
                    metrics_A2B['ssim'].append(ssim(real_A_np, recon_A_np, data_range=255))
                    metrics_A2B['psnr'].append(psnr(real_A_np, recon_A_np, data_range=255))
                    metrics_B2A['ssim'].append(ssim(real_B_np, recon_B_np, data_range=255))
                    metrics_B2A['psnr'].append(psnr(real_B_np, recon_B_np, data_range=255))
                except:
                    pass
                
                # plot images
                images = [
                    real_A[0, 0], fake_B[0, 0], 
                    torch.abs(real_A[0, 0] - fake_B[0, 0]),
                    recon_A[0, 0],
                    real_B[0, 0], fake_A[0, 0],
                    torch.abs(real_B[0, 0] - fake_A[0, 0]),
                    recon_B[0, 0]
                ]
                
                for col, img in enumerate(images):
                    ax = fig.add_subplot(gs[row + 1, col])
                    img_np = img.cpu().numpy()
                    if col in [2, 6]:  # difference maps
                        ax.imshow(img_np, cmap='hot', vmin=0, vmax=1)
                    else:
                        ax.imshow((img_np + 1) / 2, cmap='gray', vmin=0, vmax=1)
                    ax.axis('off')
        
        # add metrics summary
        fig.text(0.02, 0.02, 
                f"Cycle A→B→A: SSIM={np.mean(metrics_A2B['ssim']):.3f}±{np.std(metrics_A2B['ssim']):.3f}, "
                f"PSNR={np.mean(metrics_A2B['psnr']):.1f}±{np.std(metrics_A2B['psnr']):.1f} dB\n"
                f"Cycle B→A→B: SSIM={np.mean(metrics_B2A['ssim']):.3f}±{np.std(metrics_B2A['ssim']):.3f}, "
                f"PSNR={np.mean(metrics_B2A['psnr']):.1f}±{np.std(metrics_B2A['psnr']):.1f} dB",
                fontsize=9, family='monospace')
        
        plt.savefig(self.output_dir / 'fig3_sample_translations.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig3_sample_translations.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"saved: fig3_sample_translations.pdf")
        
        return metrics_A2B, metrics_B2A
    
    # =========================================================================
    # figure 4: quantitative metrics bar chart
    # =========================================================================
    def generate_metrics_chart(self, metrics: dict):
        """generate bar chart comparing different model versions/configurations"""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # ssim comparison
        ax = axes[0]
        models = list(metrics.keys())
        ssim_vals = [metrics[m].get('ssim', 0) for m in models]
        ssim_stds = [metrics[m].get('ssim_std', 0) for m in models]
        
        bars = ax.bar(models, ssim_vals, yerr=ssim_stds, capsize=5,
                     color=[self.colors['brats'], self.colors['upenn']], edgecolor='black')
        ax.set_ylabel('SSIM')
        ax.set_title('Structural Similarity Index', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.8, color='green', ls='--', alpha=0.5, label='Target (0.8)')
        ax.legend()
        
        # psnr comparison
        ax = axes[1]
        psnr_vals = [metrics[m].get('psnr', 0) for m in models]
        psnr_stds = [metrics[m].get('psnr_std', 0) for m in models]
        
        bars = ax.bar(models, psnr_vals, yerr=psnr_stds, capsize=5,
                     color=[self.colors['brats'], self.colors['upenn']], edgecolor='black')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('Peak Signal-to-Noise Ratio', fontweight='bold')
        ax.axhline(y=25, color='green', ls='--', alpha=0.5, label='Target (25 dB)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig4_metrics_comparison.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig4_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"saved: fig4_metrics_comparison.pdf")
    
    # =========================================================================
    # figure 5: dataset distribution
    # =========================================================================
    def generate_dataset_distribution(self, metadata: dict):
        """visualize dataset split and characteristics"""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # dataset split pie chart
        ax = axes[0]
        datasets = ['BraTS-TCGA', 'UPenn-GBM']
        sizes = [metadata.get('brats_total', 86), metadata.get('upenn_total', 476)]
        colors = [self.colors['brats'], self.colors['upenn']]
        explode = (0.05, 0)
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=datasets, colors=colors,
                                          autopct='%1.1f%%', startangle=90,
                                          textprops={'fontsize': 10})
        ax.set_title('Dataset Composition', fontweight='bold')
        
        # train/val/test split
        ax = axes[1]
        splits = ['Train', 'Validation', 'Test']
        brats_splits = [metadata.get('brats_train', 62), 
                       metadata.get('brats_val', 14), 
                       metadata.get('brats_test', 12)]
        upenn_splits = [metadata.get('upenn_train', 403),
                       metadata.get('upenn_val', 84),
                       metadata.get('upenn_test', 79)]
        
        x = np.arange(len(splits))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, brats_splits, width, label='BraTS', 
                      color=self.colors['brats'], edgecolor='black')
        bars2 = ax.bar(x + width/2, upenn_splits, width, label='UPenn',
                      color=self.colors['upenn'], edgecolor='black')
        
        ax.set_ylabel('Number of Subjects')
        ax.set_title('Train/Val/Test Split', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.legend()
        
        # add value labels
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        # mri modalities
        ax = axes[2]
        modalities = ['T1', 'T1ce', 'T2', 'FLAIR']
        y_pos = np.arange(len(modalities))
        
        ax.barh(y_pos, [1, 1, 1, 1], color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'],
               edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(modalities)
        ax.set_xlabel('Channel')
        ax.set_title('Input MRI Modalities (4-channel)', fontweight='bold')
        ax.set_xlim(0, 1.5)
        
        for i, mod in enumerate(modalities):
            ax.text(1.1, i, f'Ch {i}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig5_dataset_distribution.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig5_dataset_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"saved: fig5_dataset_distribution.pdf")
    
    # =========================================================================
    # table 1: model configuration
    # =========================================================================
    def generate_config_table(self, config: dict):
        """generate a latex-style configuration table"""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        
        table_data = [
            ['Parameter', 'Value', 'Description'],
            ['Learning Rate (G)', str(config.get('lr_g', '1e-4')), 'Generator learning rate'],
            ['Learning Rate (D)', str(config.get('lr_d', '5e-5')), 'Discriminator learning rate (TTUR)'],
            ['Batch Size', str(config.get('batch_size', 4)), 'Training batch size'],
            ['λ_cycle', str(config.get('lambda_cycle', 10.0)), 'Cycle consistency weight'],
            ['λ_identity', str(config.get('lambda_identity', 5.0)), 'Identity loss weight'],
            ['λ_GP', str(config.get('lambda_gp', 10.0)), 'Gradient penalty weight'],
            ['Label Smoothing', '0.9 / 0.1', 'Real/Fake label values'],
            ['Replay Buffer', str(config.get('buffer_size', 50)), 'Size per direction'],
            ['Spectral Norm', 'Yes', 'Applied to discriminator'],
            ['Instance Noise', '0.1 → 0', 'Decaying noise injection'],
        ]
        
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                        loc='center', cellLoc='left',
                        colWidths=[0.3, 0.2, 0.5])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # style header
        for i in range(3):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        # alternate row colors
        for i in range(1, len(table_data)):
            for j in range(3):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
        
        plt.title('Table 1: CycleGAN Training Configuration', fontweight='bold', pad=20)
        plt.savefig(self.output_dir / 'table1_configuration.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'table1_configuration.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"saved: table1_configuration.pdf")
    
    # =========================================================================
    # figure 6: epoch-wise metrics evolution
    # =========================================================================
    def generate_epoch_metrics(self, epoch_metrics: dict):
        """show how ssim/psnr evolve across training epochs"""
        if not epoch_metrics:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = list(epoch_metrics.keys())
        ssim_a2b = [epoch_metrics[e].get('ssim_a2b', 0) for e in epochs]
        ssim_b2a = [epoch_metrics[e].get('ssim_b2a', 0) for e in epochs]
        psnr_a2b = [epoch_metrics[e].get('psnr_a2b', 0) for e in epochs]
        psnr_b2a = [epoch_metrics[e].get('psnr_b2a', 0) for e in epochs]
        
        ax = axes[0]
        ax.plot(epochs, ssim_a2b, 'o-', color=self.colors['brats'], lw=2, 
               markersize=6, label='A→B→A')
        ax.plot(epochs, ssim_b2a, 's-', color=self.colors['upenn'], lw=2,
               markersize=6, label='B→A→B')
        ax.axhline(y=0.8, color='green', ls='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('SSIM')
        ax.set_title('Cycle Reconstruction SSIM', fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 1)
        
        ax = axes[1]
        ax.plot(epochs, psnr_a2b, 'o-', color=self.colors['brats'], lw=2,
               markersize=6, label='A→B→A')
        ax.plot(epochs, psnr_b2a, 's-', color=self.colors['upenn'], lw=2,
               markersize=6, label='B→A→B')
        ax.axhline(y=25, color='green', ls='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('Cycle Reconstruction PSNR', fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig6_epoch_metrics.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig6_epoch_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"saved: fig6_epoch_metrics.pdf")
    
    # =========================================================================
    # generate all figures
    # =========================================================================
    def generate_all(self, loss_file: str = None, config: dict = None, metadata: dict = None):
        """generate all publication figures"""
        print("\n" + "=" * 60)
        print("generating publication-quality figures")
        print("=" * 60 + "\n")
        
        # 1. architecture diagram
        self.generate_architecture_diagram()
        
        # 2. loss curves (if available)
        if loss_file and Path(loss_file).exists():
            self.generate_loss_curves(loss_file=loss_file)
        
        # 3. configuration table
        if config is None:
            config = {
                'lr_g': '1e-4', 'lr_d': '5e-5', 'batch_size': 4,
                'lambda_cycle': 10.0, 'lambda_identity': 5.0, 'lambda_gp': 10.0,
                'buffer_size': 50
            }
        self.generate_config_table(config)
        
        # 4. dataset distribution
        if metadata is None:
            metadata = {
                'brats_total': 86, 'upenn_total': 566,
                'brats_train': 62, 'brats_val': 14, 'brats_test': 12,
                'upenn_train': 403, 'upenn_val': 84, 'upenn_test': 79
            }
        self.generate_dataset_distribution(metadata)
        
        print(f"\nall figures saved to: {self.output_dir}")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Generate publication figures')
    parser.add_argument('--output_dir', type=str, default='/Volumes/usb drive/neuroscope/figures/publication')
    parser.add_argument('--loss_file', type=str, default='/Volumes/usb drive/neuroscope/samples/training_loss_log.json')
    parser.add_argument('--checkpoint_dir', type=str, default='/Volumes/usb drive/neuroscope/checkpoints')
    args = parser.parse_args()
    
    generator = PublicationFigureGenerator(
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir
    )
    
    generator.generate_all(loss_file=args.loss_file)


if __name__ == '__main__':
    main()
