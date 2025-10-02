"""CycleGAN training implementation.

This module provides the core training logic for CycleGAN models,
including loss computation, optimization, and training loops.
"""

import os
import sys
import logging
import itertools
import json
import random
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

from neuroscope.core.logging import get_logger
from neuroscope.models.architectures.cyclegan import CycleGAN
from neuroscope.training.optimizers.cyclegan_optimizer import CycleGANOptimizer
from neuroscope.training.callbacks.training_callbacks import TrainingCallbacks

logger = get_logger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class CycleGANTrainer:
    """CycleGAN trainer with comprehensive training capabilities."""
    
    def __init__(
        self,
        model: CycleGAN,
        optimizer: CycleGANOptimizer,
        device: torch.device,
        config: Dict[str, Any]
    ):
        """Initialize CycleGAN trainer.
        
        Args:
            model: CycleGAN model instance
            optimizer: CycleGAN optimizer instance
            device: Training device
            config: Training configuration
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        # Initialize callbacks
        self.callbacks = TrainingCallbacks(config)
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.loss_history = {
            'G_A2B': [],
            'G_B2A': [],
            'D_A': [],
            'D_B': [],
            'cycle_A': [],
            'cycle_B': [],
            'identity_A': [],
            'identity_B': []
        }
        
        # TensorBoard writer
        self.tb_writer = None
        if config.get('use_tensorboard', True):
            log_dir = config.get('log_dir', 'runs')
            self.tb_writer = SummaryWriter(log_dir)
    
    def train_epoch(
        self,
        train_loader_a,
        train_loader_b,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader_a: DataLoader for domain A
            train_loader_b: DataLoader for domain B
            epoch: Current epoch number
            
        Returns:
            Dictionary of average losses for the epoch
        """
        self.model.train()
        epoch_losses = {
            'G_A2B': 0.0,
            'G_B2A': 0.0,
            'D_A': 0.0,
            'D_B': 0.0,
            'cycle_A': 0.0,
            'cycle_B': 0.0,
            'identity_A': 0.0,
            'identity_B': 0.0
        }
        
        num_batches = min(len(train_loader_a), len(train_loader_b))
        
        for batch_idx, (real_a, real_b) in enumerate(zip(train_loader_a, train_loader_b)):
            real_a = real_a.to(self.device)
            real_b = real_b.to(self.device)
            
            # Train generators
            g_losses = self._train_generators(real_a, real_b)
            
            # Train discriminators
            d_losses = self._train_discriminators(real_a, real_b)
            
            # Update epoch losses
            for key in epoch_losses:
                if key in g_losses:
                    epoch_losses[key] += g_losses[key]
                if key in d_losses:
                    epoch_losses[key] += d_losses[key]
            
            # Update step counter
            self.current_step += 1
            
            # Log step losses
            if batch_idx % self.config.get('log_interval', 50) == 0:
                self._log_step_losses(g_losses, d_losses, batch_idx, epoch)
            
            # Sample images
            if batch_idx % self.config.get('sample_interval', 200) == 0:
                self._sample_images(real_a, real_b, epoch, batch_idx)
            
            # Callbacks
            self.callbacks.on_batch_end(
                batch_idx, epoch, g_losses, d_losses
            )
        
        # Average epoch losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Update loss history
        for key in epoch_losses:
            self.loss_history[key].append(epoch_losses[key])
        
        return epoch_losses
    
    def _train_generators(self, real_a: torch.Tensor, real_b: torch.Tensor) -> Dict[str, float]:
        """Train generator networks.
        
        Args:
            real_a: Real images from domain A
            real_b: Real images from domain B
            
        Returns:
            Dictionary of generator losses
        """
        # Forward pass
        fake_b = self.model.G_A2B(real_a)
        fake_a = self.model.G_B2A(real_b)
        
        # Reconstructed images
        rec_a = self.model.G_B2A(fake_b)
        rec_b = self.model.G_A2B(fake_a)
        
        # Identity images
        id_a = self.model.G_B2A(real_a)
        id_b = self.model.G_A2B(real_b)
        
        # Generator losses
        g_losses = self.model.compute_generator_losses(
            real_a, real_b, fake_a, fake_b, rec_a, rec_b, id_a, id_b
        )
        
        # Backward pass
        self.optimizer.zero_grad_generators()
        total_g_loss = g_losses['total']
        total_g_loss.backward()
        
        # Gradient clipping
        if self.config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(
                    self.model.G_A2B.parameters(),
                    self.model.G_B2A.parameters()
                ),
                self.config['grad_clip']
            )
        
        self.optimizer.step_generators()
        
        return g_losses
    
    def _train_discriminators(self, real_a: torch.Tensor, real_b: torch.Tensor) -> Dict[str, float]:
        """Train discriminator networks.
        
        Args:
            real_a: Real images from domain A
            real_b: Real images from domain B
            
        Returns:
            Dictionary of discriminator losses
        """
        # Generate fake images
        with torch.no_grad():
            fake_b = self.model.G_A2B(real_a)
            fake_a = self.model.G_B2A(real_b)
        
        # Discriminator losses
        d_losses = self.model.compute_discriminator_losses(
            real_a, real_b, fake_a, fake_b
        )
        
        # Backward pass for discriminator A
        self.optimizer.zero_grad_discriminator_a()
        d_losses['D_A'].backward()
        self.optimizer.step_discriminator_a()
        
        # Backward pass for discriminator B
        self.optimizer.zero_grad_discriminator_b()
        d_losses['D_B'].backward()
        self.optimizer.step_discriminator_b()
        
        return d_losses
    
    def _log_step_losses(
        self,
        g_losses: Dict[str, float],
        d_losses: Dict[str, float],
        batch_idx: int,
        epoch: int
    ):
        """Log losses for current step."""
        logger.info(
            f"Epoch [{epoch}/{self.config['n_epochs']}], "
            f"Batch [{batch_idx}], "
            f"G_A2B: {g_losses['G_A2B']:.4f}, "
            f"G_B2A: {g_losses['G_B2A']:.4f}, "
            f"D_A: {d_losses['D_A']:.4f}, "
            f"D_B: {d_losses['D_B']:.4f}"
        )
        
        # TensorBoard logging
        if self.tb_writer:
            self.tb_writer.add_scalar('Loss/G_A2B', g_losses['G_A2B'], self.current_step)
            self.tb_writer.add_scalar('Loss/G_B2A', g_losses['G_B2A'], self.current_step)
            self.tb_writer.add_scalar('Loss/D_A', d_losses['D_A'], self.current_step)
            self.tb_writer.add_scalar('Loss/D_B', d_losses['D_B'], self.current_step)
    
    def _sample_images(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
        epoch: int,
        batch_idx: int
    ):
        """Generate and save sample images."""
        self.model.eval()
        
        with torch.no_grad():
            fake_b = self.model.G_A2B(real_a)
            fake_a = self.model.G_B2A(real_b)
            
            # Create image grid
            images = torch.cat([
                real_a[:4], fake_b[:4],
                real_b[:4], fake_a[:4]
            ], dim=0)
            
            # Save image grid
            sample_dir = self.config.get('sample_dir', 'samples')
            os.makedirs(sample_dir, exist_ok=True)
            
            save_image(
                images,
                f"{sample_dir}/sample_{epoch}_{batch_idx}.png",
                nrow=4,
                normalize=True,
                value_range=(-1, 1)
            )
            
            # TensorBoard logging
            if self.tb_writer:
                self.tb_writer.add_image(
                    'Samples',
                    make_grid(images, nrow=4, normalize=True, value_range=(-1, 1)),
                    self.current_step
                )
        
        self.model.train()
    
    def save_checkpoint(self, epoch: int, checkpoint_dir: str):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            checkpoint_dir: Directory to save checkpoint
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_history = checkpoint['loss_history']
        self.current_epoch = checkpoint['epoch']
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    
    def plot_loss_curves(self, save_path: str):
        """Plot and save loss curves.
        
        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CycleGAN Training Losses', fontsize=16)
        
        # Generator losses
        axes[0, 0].plot(self.loss_history['G_A2B'], label='G_A2B')
        axes[0, 0].plot(self.loss_history['G_B2A'], label='G_B2A')
        axes[0, 0].set_title('Generator Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Discriminator losses
        axes[0, 1].plot(self.loss_history['D_A'], label='D_A')
        axes[0, 1].plot(self.loss_history['D_B'], label='D_B')
        axes[0, 1].set_title('Discriminator Losses')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Cycle consistency losses
        axes[1, 0].plot(self.loss_history['cycle_A'], label='Cycle A')
        axes[1, 0].plot(self.loss_history['cycle_B'], label='Cycle B')
        axes[1, 0].set_title('Cycle Consistency Losses')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Identity losses
        axes[1, 1].plot(self.loss_history['identity_A'], label='Identity A')
        axes[1, 1].plot(self.loss_history['identity_B'], label='Identity B')
        axes[1, 1].set_title('Identity Losses')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved loss curves: {save_path}")
    
    def close(self):
        """Close trainer and cleanup resources."""
        if self.tb_writer:
            self.tb_writer.close()