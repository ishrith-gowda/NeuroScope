"""
Sample Generator.

Generates and saves visual samples during training for
monitoring and publication purposes.

Author: NeuroScope Research Team
"""

from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime


class SampleGenerator:
    """
    Generates and saves visual samples during training.
    
    Features:
    - Side-by-side comparison images
    - Multi-modality visualization
    - Image grids with labels
    - Difference maps
    - Histogram comparisons
    - Training progress tracking
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        save_format: str = 'png',
        dpi: int = 150,
        figsize: Tuple[int, int] = (16, 8),
        max_samples: int = 8,
        modality_names: Optional[List[str]] = None
    ):
        """
        Initialize sample generator.
        
        Args:
            output_dir: Directory to save samples
            save_format: Image format ('png', 'jpg', 'pdf')
            dpi: DPI for saved images
            figsize: Figure size in inches
            max_samples: Maximum number of samples per batch to visualize
            modality_names: Names of MRI modalities
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_format = save_format
        self.dpi = dpi
        self.figsize = figsize
        self.max_samples = max_samples
        self.modality_names = modality_names or ['T1', 'T1Gd', 'T2', 'FLAIR']
        
        # Create subdirectories
        self.comparisons_dir = self.output_dir / 'comparisons'
        self.modalities_dir = self.output_dir / 'modalities'
        self.difference_dir = self.output_dir / 'difference_maps'
        self.grids_dir = self.output_dir / 'grids'
        
        for d in [self.comparisons_dir, self.modalities_dir, 
                  self.difference_dir, self.grids_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        # Track saved samples for animation
        self.sample_history: List[Dict] = []
        
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
        
    def _normalize_for_display(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range for display."""
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min > 1e-8:
            return (img - img_min) / (img_max - img_min)
        return np.zeros_like(img)
        
    def save_comparison(
        self,
        real_A: torch.Tensor,
        fake_B: torch.Tensor,
        rec_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_A: torch.Tensor,
        rec_B: torch.Tensor,
        epoch: int,
        batch_idx: int = 0,
        modality_idx: int = 0,
        sample_idx: int = 0
    ) -> Path:
        """
        Save side-by-side comparison of A→B→A and B→A→B cycles.
        
        Args:
            real_A: Real domain A images [B, C, H, W]
            fake_B: Generated B images [B, C, H, W]
            rec_A: Reconstructed A images [B, C, H, W]
            real_B: Real domain B images [B, C, H, W]
            fake_A: Generated A images [B, C, H, W]
            rec_B: Reconstructed B images [B, C, H, W]
            epoch: Current epoch
            batch_idx: Batch index
            modality_idx: Which modality to visualize
            sample_idx: Which sample from batch
            
        Returns:
            Path to saved figure
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("Warning: matplotlib not available for sample generation")
            return None
            
        # Extract single sample and modality
        def get_img(tensor, mod_idx=modality_idx, samp_idx=sample_idx):
            t = self._to_numpy(tensor)
            if t.ndim == 4:  # [B, C, H, W]
                if samp_idx < t.shape[0] and mod_idx < t.shape[1]:
                    return self._normalize_for_display(t[samp_idx, mod_idx])
            return np.zeros((64, 64))
            
        # Get images
        img_real_A = get_img(real_A)
        img_fake_B = get_img(fake_B)
        img_rec_A = get_img(rec_A)
        img_real_B = get_img(real_B)
        img_fake_A = get_img(fake_A)
        img_rec_B = get_img(rec_B)
        
        # Compute difference maps
        diff_A = np.abs(img_real_A - img_rec_A)
        diff_B = np.abs(img_real_B - img_rec_B)
        
        # Create figure
        fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.1, wspace=0.1)
        
        modality_name = self.modality_names[modality_idx] if modality_idx < len(self.modality_names) else f'Mod{modality_idx}'
        
        # Row 1: A → B → A cycle
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_real_A, cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f'Real A ({modality_name})', fontsize=10)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img_fake_B, cmap='gray', vmin=0, vmax=1)
        ax2.set_title('Fake B (A→B)', fontsize=10)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(img_rec_A, cmap='gray', vmin=0, vmax=1)
        ax3.set_title('Rec A (A→B→A)', fontsize=10)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(diff_A, cmap='hot', vmin=0, vmax=0.5)
        ax4.set_title('|Real A - Rec A|', fontsize=10)
        ax4.axis('off')
        
        # Row 2: B → A → B cycle
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.imshow(img_real_B, cmap='gray', vmin=0, vmax=1)
        ax5.set_title(f'Real B ({modality_name})', fontsize=10)
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.imshow(img_fake_A, cmap='gray', vmin=0, vmax=1)
        ax6.set_title('Fake A (B→A)', fontsize=10)
        ax6.axis('off')
        
        ax7 = fig.add_subplot(gs[1, 2])
        ax7.imshow(img_rec_B, cmap='gray', vmin=0, vmax=1)
        ax7.set_title('Rec B (B→A→B)', fontsize=10)
        ax7.axis('off')
        
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.imshow(diff_B, cmap='hot', vmin=0, vmax=0.5)
        ax8.set_title('|Real B - Rec B|', fontsize=10)
        ax8.axis('off')
        
        # Add epoch info
        fig.suptitle(f'Epoch {epoch} - Batch {batch_idx} - {modality_name}', 
                    fontsize=12, fontweight='bold')
        
        # Save
        filename = f'comparison_epoch{epoch:04d}_batch{batch_idx:04d}_{modality_name}.{self.save_format}'
        filepath = self.comparisons_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return filepath
        
    def save_multi_modality_comparison(
        self,
        real_A: torch.Tensor,
        fake_B: torch.Tensor,
        real_B: torch.Tensor,
        fake_A: torch.Tensor,
        epoch: int,
        sample_idx: int = 0
    ) -> Path:
        """
        Save comparison across all modalities.
        
        Args:
            real_A: Real domain A images [B, C, H, W]
            fake_B: Generated B images [B, C, H, W]
            real_B: Real domain B images [B, C, H, W]
            fake_A: Generated A images [B, C, H, W]
            epoch: Current epoch
            sample_idx: Which sample from batch
            
        Returns:
            Path to saved figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None
            
        n_modalities = min(real_A.shape[1], len(self.modality_names))
        
        fig, axes = plt.subplots(4, n_modalities, figsize=(4*n_modalities, 12))
        
        for i in range(n_modalities):
            # Get images for this modality
            img_real_A = self._normalize_for_display(
                self._to_numpy(real_A[sample_idx, i])
            )
            img_fake_B = self._normalize_for_display(
                self._to_numpy(fake_B[sample_idx, i])
            )
            img_real_B = self._normalize_for_display(
                self._to_numpy(real_B[sample_idx, i])
            )
            img_fake_A = self._normalize_for_display(
                self._to_numpy(fake_A[sample_idx, i])
            )
            
            # Plot
            axes[0, i].imshow(img_real_A, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'{self.modality_names[i]}', fontsize=10)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Real A', fontsize=10)
                
            axes[1, i].imshow(img_fake_B, cmap='gray', vmin=0, vmax=1)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('Fake B', fontsize=10)
                
            axes[2, i].imshow(img_real_B, cmap='gray', vmin=0, vmax=1)
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_ylabel('Real B', fontsize=10)
                
            axes[3, i].imshow(img_fake_A, cmap='gray', vmin=0, vmax=1)
            axes[3, i].axis('off')
            if i == 0:
                axes[3, i].set_ylabel('Fake A', fontsize=10)
                
        fig.suptitle(f'Multi-Modality Comparison - Epoch {epoch}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'multimodality_epoch{epoch:04d}.{self.save_format}'
        filepath = self.modalities_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return filepath
        
    def save_sample_grid(
        self,
        images: torch.Tensor,
        epoch: int,
        name: str = 'samples',
        nrow: int = 4,
        normalize: bool = True,
        modality_idx: int = 0
    ) -> Path:
        """
        Save a grid of sample images.
        
        Args:
            images: Batch of images [B, C, H, W]
            epoch: Current epoch
            name: Name for the grid
            nrow: Number of images per row
            normalize: Whether to normalize images
            modality_idx: Which modality channel to visualize
            
        Returns:
            Path to saved figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None
            
        images_np = self._to_numpy(images)
        n_samples = min(images_np.shape[0], self.max_samples)
        ncol = nrow
        nrow_grid = (n_samples + ncol - 1) // ncol
        
        fig, axes = plt.subplots(nrow_grid, ncol, figsize=(3*ncol, 3*nrow_grid))
        axes = np.atleast_2d(axes)
        
        for idx in range(n_samples):
            row = idx // ncol
            col = idx % ncol
            
            img = images_np[idx, modality_idx]
            if normalize:
                img = self._normalize_for_display(img)
                
            axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[row, col].axis('off')
            
        # Hide unused subplots
        for idx in range(n_samples, nrow_grid * ncol):
            row = idx // ncol
            col = idx % ncol
            axes[row, col].axis('off')
            
        modality_name = self.modality_names[modality_idx] if modality_idx < len(self.modality_names) else f'Mod{modality_idx}'
        fig.suptitle(f'{name} - Epoch {epoch} - {modality_name}', fontsize=12)
        plt.tight_layout()
        
        filename = f'{name}_epoch{epoch:04d}_{modality_name}.{self.save_format}'
        filepath = self.grids_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return filepath
        
    def save_difference_map(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        epoch: int,
        name: str = 'difference',
        sample_idx: int = 0,
        modality_idx: int = 0
    ) -> Path:
        """
        Save a difference/error map between original and reconstructed.
        
        Args:
            original: Original images [B, C, H, W]
            reconstructed: Reconstructed images [B, C, H, W]
            epoch: Current epoch
            name: Name for the difference map
            sample_idx: Which sample from batch
            modality_idx: Which modality
            
        Returns:
            Path to saved figure
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            return None
            
        orig = self._to_numpy(original[sample_idx, modality_idx])
        recon = self._to_numpy(reconstructed[sample_idx, modality_idx])
        
        orig = self._normalize_for_display(orig)
        recon = self._normalize_for_display(recon)
        
        diff = np.abs(orig - recon)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original
        im0 = axes[0].imshow(orig, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Original')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Reconstructed
        im1 = axes[1].imshow(recon, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Difference (heat map)
        im2 = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[2].set_title(f'Absolute Difference\nMean: {diff.mean():.4f}')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Histogram of differences
        axes[3].hist(diff.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[3].set_xlabel('Difference Value')
        axes[3].set_ylabel('Frequency')
        axes[3].set_title('Difference Distribution')
        axes[3].axvline(diff.mean(), color='red', linestyle='--', label=f'Mean: {diff.mean():.4f}')
        axes[3].legend()
        
        modality_name = self.modality_names[modality_idx] if modality_idx < len(self.modality_names) else f'Mod{modality_idx}'
        fig.suptitle(f'{name} - Epoch {epoch} - {modality_name}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        filename = f'{name}_epoch{epoch:04d}_{modality_name}.{self.save_format}'
        filepath = self.difference_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return filepath
        
    def generate_all_samples(
        self,
        real_A: torch.Tensor,
        fake_B: torch.Tensor,
        rec_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_A: torch.Tensor,
        rec_B: torch.Tensor,
        epoch: int,
        batch_idx: int = 0
    ) -> Dict[str, Path]:
        """
        Generate all sample types for a batch.
        
        Returns dictionary of sample type -> file path
        """
        results = {}
        
        # Per-modality comparisons
        n_modalities = min(real_A.shape[1], len(self.modality_names))
        for mod_idx in range(n_modalities):
            path = self.save_comparison(
                real_A, fake_B, rec_A,
                real_B, fake_A, rec_B,
                epoch, batch_idx, mod_idx
            )
            if path:
                results[f'comparison_{self.modality_names[mod_idx]}'] = path
                
        # Multi-modality view
        path = self.save_multi_modality_comparison(
            real_A, fake_B, real_B, fake_A, epoch
        )
        if path:
            results['multi_modality'] = path
            
        # Difference maps
        for mod_idx in range(min(2, n_modalities)):  # Just first 2 modalities
            path = self.save_difference_map(
                real_A, rec_A, epoch,
                name='diff_A2B2A',
                modality_idx=mod_idx
            )
            if path:
                results[f'diff_A_{self.modality_names[mod_idx]}'] = path
                
        # Sample grids
        path = self.save_sample_grid(fake_B, epoch, name='fake_B')
        if path:
            results['grid_fake_B'] = path
            
        path = self.save_sample_grid(fake_A, epoch, name='fake_A')
        if path:
            results['grid_fake_A'] = path
            
        return results
        
    def close(self):
        """Cleanup."""
        pass
