"""
Figure Generator.

Creates publication-quality figures for training visualization
and research papers.

Author: NeuroScope Research Team
"""

from typing import Optional, Dict, List, Tuple, Union, Any
from pathlib import Path
import numpy as np
from datetime import datetime


class FigureGenerator:
    """
    Generates publication-quality figures for training analysis.
    
    Features:
    - Loss curves with multiple components
    - Metric progression plots
    - Learning rate schedules
    - Gradient norm tracking
    - Distribution comparisons
    - Statistical summaries
    - LaTeX-compatible output
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        style: str = 'publication',
        save_format: str = 'pdf',
        dpi: int = 300,
        figsize: Tuple[float, float] = (10, 6),
        font_size: int = 12
    ):
        """
        Initialize figure generator.
        
        Args:
            output_dir: Directory to save figures
            style: Figure style ('publication', 'presentation', 'notebook')
            save_format: Output format ('pdf', 'png', 'svg', 'eps')
            dpi: Resolution for raster formats
            figsize: Default figure size in inches
            font_size: Base font size
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.style = style
        self.save_format = save_format
        self.dpi = dpi
        self.figsize = figsize
        self.font_size = font_size
        
        # Create subdirectories
        self.losses_dir = self.output_dir / 'losses'
        self.metrics_dir = self.output_dir / 'metrics'
        self.analysis_dir = self.output_dir / 'analysis'
        self.publication_dir = self.output_dir / 'publication'
        
        for d in [self.losses_dir, self.metrics_dir, 
                  self.analysis_dir, self.publication_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        self._setup_style()
        
    def _setup_style(self):
        """Configure matplotlib style."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            
            if self.style == 'publication':
                # Clean, publication-ready style
                plt.rcParams.update({
                    'font.size': self.font_size,
                    'font.family': 'serif',
                    'axes.labelsize': self.font_size,
                    'axes.titlesize': self.font_size + 2,
                    'xtick.labelsize': self.font_size - 2,
                    'ytick.labelsize': self.font_size - 2,
                    'legend.fontsize': self.font_size - 2,
                    'figure.titlesize': self.font_size + 4,
                    'axes.grid': True,
                    'grid.alpha': 0.3,
                    'axes.spines.top': False,
                    'axes.spines.right': False,
                    'figure.dpi': self.dpi,
                    'savefig.dpi': self.dpi,
                    'savefig.bbox': 'tight',
                    'savefig.pad_inches': 0.1,
                })
            elif self.style == 'presentation':
                plt.rcParams.update({
                    'font.size': self.font_size + 4,
                    'font.family': 'sans-serif',
                    'axes.labelsize': self.font_size + 4,
                    'axes.titlesize': self.font_size + 6,
                    'xtick.labelsize': self.font_size + 2,
                    'ytick.labelsize': self.font_size + 2,
                    'legend.fontsize': self.font_size + 2,
                    'axes.grid': True,
                    'grid.alpha': 0.4,
                    'lines.linewidth': 2.5,
                })
        except ImportError:
            pass
            
    # =========================================================================
    # Loss Curves
    # =========================================================================
    
    def plot_training_losses(
        self,
        history: Dict[str, List[float]],
        title: str = 'Training Losses',
        smooth_window: int = 0
    ) -> Path:
        """
        Plot all training loss components.
        
        Args:
            history: Dictionary with loss histories
            title: Figure title
            smooth_window: Window size for smoothing (0 = no smoothing)
            
        Returns:
            Path to saved figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = {
            'G_loss': '#2ecc71',
            'D_loss': '#e74c3c',
            'cycle': '#3498db',
            'identity': '#9b59b6',
            'gan': '#f39c12',
            'ssim': '#1abc9c'
        }
        
        def smooth(y, window):
            if window <= 1 or len(y) < window:
                return y
            return np.convolve(y, np.ones(window)/window, mode='valid')
            
        epochs = None
        
        # Generator loss
        if 'G_loss' in history or 'train_G_loss' in history:
            data = history.get('G_loss', history.get('train_G_loss', []))
            epochs = range(1, len(data) + 1)
            y = smooth(data, smooth_window) if smooth_window else data
            x = range(1, len(y) + 1)
            axes[0, 0].plot(x, y, color=colors['G_loss'], linewidth=1.5, label='Generator')
        if 'D_loss' in history or 'train_D_loss' in history:
            data = history.get('D_loss', history.get('train_D_loss', []))
            y = smooth(data, smooth_window) if smooth_window else data
            x = range(1, len(y) + 1)
            axes[0, 0].plot(x, y, color=colors['D_loss'], linewidth=1.5, label='Discriminator')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Generator vs Discriminator Loss')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        
        # Cycle and Identity losses
        if 'cycle_loss' in history or 'train_cycle_loss' in history:
            data = history.get('cycle_loss', history.get('train_cycle_loss', []))
            y = smooth(data, smooth_window) if smooth_window else data
            x = range(1, len(y) + 1)
            axes[0, 1].plot(x, y, color=colors['cycle'], linewidth=1.5, label='Cycle')
        if 'identity_loss' in history or 'train_identity_loss' in history:
            data = history.get('identity_loss', history.get('train_identity_loss', []))
            y = smooth(data, smooth_window) if smooth_window else data
            x = range(1, len(y) + 1)
            axes[0, 1].plot(x, y, color=colors['identity'], linewidth=1.5, label='Identity')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Cycle & Identity Losses')
        axes[0, 1].legend()
        
        # GAN losses (if available)
        gan_keys = [k for k in history.keys() if 'gan' in k.lower()]
        for key in gan_keys:
            data = history[key]
            y = smooth(data, smooth_window) if smooth_window else data
            x = range(1, len(y) + 1)
            axes[1, 0].plot(x, y, linewidth=1.5, label=key)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('GAN Losses')
        if gan_keys:
            axes[1, 0].legend()
            
        # Learning rate
        if 'learning_rate' in history:
            data = history['learning_rate']
            axes[1, 1].plot(range(1, len(data) + 1), data, 
                          color='#34495e', linewidth=1.5)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_yscale('log')
            
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = self.losses_dir / f'training_losses.{self.save_format}'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return filepath
        
    def plot_generator_discriminator_balance(
        self,
        g_losses: List[float],
        d_losses: List[float],
        title: str = 'GAN Training Balance'
    ) -> Path:
        """
        Plot generator vs discriminator loss balance.
        
        Shows ratio over time to monitor training stability.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(g_losses) + 1)
        
        # Absolute losses
        axes[0].plot(epochs, g_losses, 'g-', linewidth=1.5, label='Generator', alpha=0.8)
        axes[0].plot(epochs, d_losses, 'r-', linewidth=1.5, label='Discriminator', alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Absolute Losses')
        axes[0].legend()
        axes[0].set_yscale('log')
        
        # G/D Ratio
        ratios = [g / (d + 1e-8) for g, d in zip(g_losses, d_losses)]
        axes[1].plot(epochs, ratios, 'b-', linewidth=1.5)
        axes[1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Balance (G=D)')
        axes[1].fill_between(epochs, 0.5, 2.0, alpha=0.2, color='green', label='Healthy Range')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('G/D Loss Ratio')
        axes[1].set_title('Generator/Discriminator Balance')
        axes[1].set_ylim(0, 5)
        axes[1].legend()
        
        fig.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        filepath = self.losses_dir / f'gd_balance.{self.save_format}'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return filepath
        
    # =========================================================================
    # Metric Curves
    # =========================================================================
    
    def plot_validation_metrics(
        self,
        history: Dict[str, List[float]],
        title: str = 'Validation Metrics'
    ) -> Path:
        """
        Plot validation metrics (SSIM, PSNR) over training.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # SSIM
        ssim_keys = [k for k in history.keys() if 'ssim' in k.lower()]
        for key in ssim_keys:
            data = history[key]
            epochs = range(1, len(data) + 1)
            label = key.replace('val_', '').replace('_', ' ').upper()
            axes[0].plot(epochs, data, linewidth=1.5, label=label, marker='o', markersize=3)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('SSIM')
        axes[0].set_title('Structural Similarity (SSIM)')
        axes[0].set_ylim(0.5, 1.0)
        axes[0].legend()
        
        # PSNR
        psnr_keys = [k for k in history.keys() if 'psnr' in k.lower()]
        for key in psnr_keys:
            data = history[key]
            epochs = range(1, len(data) + 1)
            label = key.replace('val_', '').replace('_', ' ').upper()
            axes[1].plot(epochs, data, linewidth=1.5, label=label, marker='o', markersize=3)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('PSNR (dB)')
        axes[1].set_title('Peak Signal-to-Noise Ratio (PSNR)')
        axes[1].legend()
        
        fig.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        filepath = self.metrics_dir / f'validation_metrics.{self.save_format}'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return filepath
        
    def plot_combined_metrics(
        self,
        train_history: Dict[str, List[float]],
        val_history: Dict[str, List[float]],
        title: str = 'Training Progress'
    ) -> Path:
        """
        Create comprehensive training progress figure.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            return None
            
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Generator Loss
        ax1 = fig.add_subplot(gs[0, 0])
        if 'G_loss' in train_history:
            ax1.plot(train_history['G_loss'], 'g-', linewidth=1.5, alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Generator Loss')
        ax1.set_yscale('log')
        
        # Discriminator Loss
        ax2 = fig.add_subplot(gs[0, 1])
        if 'D_loss' in train_history:
            ax2.plot(train_history['D_loss'], 'r-', linewidth=1.5, alpha=0.8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Discriminator Loss')
        ax2.set_yscale('log')
        
        # Cycle Loss
        ax3 = fig.add_subplot(gs[0, 2])
        if 'cycle_loss' in train_history:
            ax3.plot(train_history['cycle_loss'], 'b-', linewidth=1.5, alpha=0.8)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Cycle Consistency Loss')
        
        # SSIM
        ax4 = fig.add_subplot(gs[1, 0])
        ssim_keys = [k for k in val_history.keys() if 'ssim' in k.lower()]
        for key in ssim_keys:
            ax4.plot(val_history[key], linewidth=1.5, label=key, marker='o', markersize=3)
        ax4.set_xlabel('Validation Step')
        ax4.set_ylabel('SSIM')
        ax4.set_title('SSIM Progression')
        ax4.set_ylim(0.5, 1.0)
        if ssim_keys:
            ax4.legend()
        
        # PSNR
        ax5 = fig.add_subplot(gs[1, 1])
        psnr_keys = [k for k in val_history.keys() if 'psnr' in k.lower()]
        for key in psnr_keys:
            ax5.plot(val_history[key], linewidth=1.5, label=key, marker='o', markersize=3)
        ax5.set_xlabel('Validation Step')
        ax5.set_ylabel('PSNR (dB)')
        ax5.set_title('PSNR Progression')
        if psnr_keys:
            ax5.legend()
        
        # Learning Rate
        ax6 = fig.add_subplot(gs[1, 2])
        if 'learning_rate' in train_history:
            ax6.plot(train_history['learning_rate'], 'k-', linewidth=1.5)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Learning Rate')
        ax6.set_title('Learning Rate Schedule')
        ax6.set_yscale('log')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        filepath = self.metrics_dir / f'training_progress.{self.save_format}'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return filepath
        
    # =========================================================================
    # Gradient Analysis
    # =========================================================================
    
    def plot_gradient_norms(
        self,
        gradient_history: Dict[str, List[float]],
        title: str = 'Gradient Norms'
    ) -> Path:
        """
        Plot gradient norm history to detect vanishing/exploding gradients.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None
            
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        for idx, (name, values) in enumerate(gradient_history.items()):
            color = colors[idx % len(colors)]
            steps = range(1, len(values) + 1)
            ax.plot(steps, values, linewidth=1.5, label=name, color=color, alpha=0.8)
            
        ax.set_xlabel('Step')
        ax.set_ylabel('Gradient L2 Norm')
        ax.set_title(title)
        ax.legend()
        ax.set_yscale('log')
        
        # Add reference lines
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Reference (1.0)')
        ax.axhline(y=100.0, color='red', linestyle=':', alpha=0.5, label='Warning (100)')
        
        filepath = self.analysis_dir / f'gradient_norms.{self.save_format}'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return filepath
        
    # =========================================================================
    # Publication Figures
    # =========================================================================
    
    def create_publication_summary(
        self,
        history: Dict[str, List[float]],
        val_history: Dict[str, List[float]],
        final_metrics: Dict[str, float],
        experiment_name: str = 'SA-CycleGAN'
    ) -> Path:
        """
        Create a publication-ready summary figure.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            return None
            
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # Main title
        fig.suptitle(f'{experiment_name} Training Summary', fontsize=16, fontweight='bold', y=0.98)
        
        # (a) Training Losses
        ax_a = fig.add_subplot(gs[0, 0])
        if 'G_loss' in history:
            ax_a.plot(history['G_loss'], 'g-', label='G', alpha=0.8)
        if 'D_loss' in history:
            ax_a.plot(history['D_loss'], 'r-', label='D', alpha=0.8)
        ax_a.set_xlabel('Epoch')
        ax_a.set_ylabel('Loss')
        ax_a.set_title('(a) Training Losses')
        ax_a.legend(loc='upper right')
        ax_a.set_yscale('log')
        
        # (b) Cycle Consistency
        ax_b = fig.add_subplot(gs[0, 1])
        if 'cycle_loss' in history:
            ax_b.plot(history['cycle_loss'], 'b-', label='Cycle', alpha=0.8)
        if 'identity_loss' in history:
            ax_b.plot(history['identity_loss'], 'm-', label='Identity', alpha=0.8)
        ax_b.set_xlabel('Epoch')
        ax_b.set_ylabel('Loss')
        ax_b.set_title('(b) Reconstruction Losses')
        ax_b.legend(loc='upper right')
        
        # (c) Learning Rate
        ax_c = fig.add_subplot(gs[0, 2])
        if 'learning_rate' in history:
            ax_c.plot(history['learning_rate'], 'k-')
        ax_c.set_xlabel('Epoch')
        ax_c.set_ylabel('Learning Rate')
        ax_c.set_title('(c) Learning Rate Schedule')
        ax_c.set_yscale('log')
        
        # (d) SSIM Progression
        ax_d = fig.add_subplot(gs[1, 0])
        ssim_a2b = val_history.get('ssim_A2B', val_history.get('val_ssim_A2B', []))
        ssim_b2a = val_history.get('ssim_B2A', val_history.get('val_ssim_B2A', []))
        if ssim_a2b:
            ax_d.plot(ssim_a2b, 'b-o', markersize=4, label='A→B→A')
        if ssim_b2a:
            ax_d.plot(ssim_b2a, 'r-s', markersize=4, label='B→A→B')
        ax_d.set_xlabel('Validation Step')
        ax_d.set_ylabel('SSIM')
        ax_d.set_title('(d) SSIM Progression')
        ax_d.set_ylim(0.6, 1.0)
        ax_d.legend()
        
        # (e) PSNR Progression
        ax_e = fig.add_subplot(gs[1, 1])
        psnr_a2b = val_history.get('psnr_A2B', val_history.get('val_psnr_A2B', []))
        psnr_b2a = val_history.get('psnr_B2A', val_history.get('val_psnr_B2A', []))
        if psnr_a2b:
            ax_e.plot(psnr_a2b, 'b-o', markersize=4, label='A→B→A')
        if psnr_b2a:
            ax_e.plot(psnr_b2a, 'r-s', markersize=4, label='B→A→B')
        ax_e.set_xlabel('Validation Step')
        ax_e.set_ylabel('PSNR (dB)')
        ax_e.set_title('(e) PSNR Progression')
        ax_e.legend()
        
        # (f) Final Metrics Summary
        ax_f = fig.add_subplot(gs[1, 2])
        ax_f.axis('off')
        
        # Create text summary
        metrics_text = "Final Metrics:\n" + "-" * 25 + "\n"
        for name, value in final_metrics.items():
            if 'ssim' in name.lower():
                metrics_text += f"{name}: {value:.4f}\n"
            elif 'psnr' in name.lower():
                metrics_text += f"{name}: {value:.2f} dB\n"
            else:
                metrics_text += f"{name}: {value:.4f}\n"
                
        ax_f.text(0.1, 0.9, metrics_text, transform=ax_f.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_f.set_title('(f) Final Results')
        
        filepath = self.publication_dir / f'training_summary.{self.save_format}'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return filepath
        
    def generate_all_figures(
        self,
        history: Dict[str, List[float]],
        val_history: Optional[Dict[str, List[float]]] = None,
        gradient_history: Optional[Dict[str, List[float]]] = None,
        final_metrics: Optional[Dict[str, float]] = None,
        experiment_name: str = 'SA-CycleGAN'
    ) -> Dict[str, Path]:
        """
        Generate all standard training figures.
        
        Returns dictionary of figure name -> file path
        """
        results = {}
        
        # Training losses
        path = self.plot_training_losses(history)
        if path:
            results['training_losses'] = path
            
        # G/D balance
        if 'G_loss' in history and 'D_loss' in history:
            path = self.plot_generator_discriminator_balance(
                history['G_loss'],
                history['D_loss']
            )
            if path:
                results['gd_balance'] = path
                
        # Validation metrics
        if val_history:
            path = self.plot_validation_metrics(val_history)
            if path:
                results['validation_metrics'] = path
                
            # Combined progress
            path = self.plot_combined_metrics(history, val_history)
            if path:
                results['training_progress'] = path
                
        # Gradient norms
        if gradient_history:
            path = self.plot_gradient_norms(gradient_history)
            if path:
                results['gradient_norms'] = path
                
        # Publication summary
        if val_history and final_metrics:
            path = self.create_publication_summary(
                history, val_history, final_metrics, experiment_name
            )
            if path:
                results['publication_summary'] = path
                
        return results
        
    def close(self):
        """Cleanup."""
        pass
