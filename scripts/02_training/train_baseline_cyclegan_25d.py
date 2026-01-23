#!/usr/bin/env python3
"""
baseline training script for 2.5d cyclegan (no attention) - ablation study.

standard cyclegan baseline without attention mechanisms for comparison.
identical architecture to sa-cyclegan except no self-attention and no cbam.

usage:
    python train_baseline_cyclegan_25d.py --config configs/training/baseline.yaml
    python train_baseline_cyclegan_25d.py --epochs 100 --batch_size 4 --image_size 128
"""

import os
import sys
import argparse
import time
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from neuroscope.models.architectures.baseline_cyclegan_25d import (
    BaselineCycleGAN25D, BaselineCycleGAN25DConfig, create_baseline_model
)
from neuroscope.data.datasets.dataset_25d import UnpairedMRIDataset25D, create_dataloaders
from neuroscope.models.losses.combined_losses import CombinedLoss


class ReplayBuffer:
    """
    Image buffer for discriminator training stability.
    
    Stores previously generated images and randomly samples from them
    to provide diverse training examples for the discriminator.
    """
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data = []
        
    def push_and_pop(self, data: torch.Tensor) -> torch.Tensor:
        result = []
        for element in data:
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                result.append(element)
            else:
                if np.random.random() > 0.5:
                    idx = np.random.randint(0, self.max_size)
                    result.append(self.data[idx].clone())
                    self.data[idx] = element
                else:
                    result.append(element)
        return torch.cat(result, dim=0)


class BaselineCycleGAN25DTrainer:
    """
    Comprehensive trainer for 2.5D SA-CycleGAN.
    
    Features:
    - Full training loop with validation
    - TensorBoard logging
    - Checkpoint management
    - Learning rate scheduling
    - Replay buffer for discriminator stability
    - Comprehensive metrics tracking
    """
    
    def __init__(
        self,
        config: BaselineCycleGAN25DConfig,
        brats_dir: str,
        upenn_dir: str,
        output_dir: str,
        batch_size: int = 4,
        image_size: int = 128,
        lr: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
        num_workers: int = 4,
        device: str = 'auto',
        experiment_name: str = None
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment naming
        if experiment_name is None:
            experiment_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.experiment_dir / "checkpoints").mkdir(exist_ok=True)
        (self.experiment_dir / "samples").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)
        
        # Device setup
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # Multi-GPU detection
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.use_multi_gpu = self.num_gpus > 1

        self._print_header()
        print(f"Device: {self.device}")
        if self.use_multi_gpu:
            print(f"Multi-GPU: {self.num_gpus} GPUs available (DataParallel enabled)")
        print(f"Experiment: {experiment_name}")
        print(f"Output: {self.experiment_dir}")
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(PROJECT_ROOT / "runs" / experiment_name))
        
        # Create model
        print("\n" + "=" * 60)
        print("Initializing Baseline 2.5D CycleGAN Model (No Attention)")
        print("=" * 60)
        self.model = create_baseline_model(config)
        self.model = self.model.to(self.device)

        # Wrap with DataParallel for multi-GPU training
        if self.use_multi_gpu:
            print(f"Wrapping model with DataParallel across {self.num_gpus} GPUs")
            self.model.G_A2B = DataParallel(self.model.G_A2B)
            self.model.G_B2A = DataParallel(self.model.G_B2A)
            self.model.D_A = DataParallel(self.model.D_A)
            self.model.D_B = DataParallel(self.model.D_B)
        
        # Create dataloaders
        print("\n" + "=" * 60)
        print("Creating Dataloaders")
        print("=" * 60)
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            brats_dir=brats_dir,
            upenn_dir=upenn_dir,
            batch_size=batch_size,
            image_size=(image_size, image_size),
            num_workers=num_workers
        )
        
        print(f"\nDataset Statistics:")
        print(f"  Training batches: {len(self.train_loader)}")
        print(f"  Validation batches: {len(self.val_loader)}")
        print(f"  Test batches: {len(self.test_loader)}")
        
        # Optimizers
        self.opt_G = optim.Adam(
            list(self.model.G_A2B.parameters()) + list(self.model.G_B2A.parameters()),
            lr=lr, betas=(beta1, beta2)
        )
        self.opt_D = optim.Adam(
            list(self.model.D_A.parameters()) + list(self.model.D_B.parameters()),
            lr=lr, betas=(beta1, beta2)
        )
        
        # Learning rate schedulers (linear decay after 50%)
        def lambda_rule(epoch, total_epochs=100):
            decay_start = total_epochs // 2
            if epoch < decay_start:
                return 1.0
            return 1.0 - (epoch - decay_start) / (total_epochs - decay_start + 1)
        
        self.scheduler_G = optim.lr_scheduler.LambdaLR(
            self.opt_G, lr_lambda=lambda e: lambda_rule(e, 100)
        )
        self.scheduler_D = optim.lr_scheduler.LambdaLR(
            self.opt_D, lr_lambda=lambda e: lambda_rule(e, 100)
        )
        
        # Loss functions
        self.losses = CombinedLoss(
            lambda_cycle=config.lambda_cycle,
            lambda_identity=config.lambda_identity,
            lambda_ssim=config.lambda_ssim,
            lambda_gradient=1.0
        ).to(self.device)
        
        # Replay buffers for discriminator
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()
        
        # Training history
        self.history = {
            'train': {'G_loss': [], 'D_loss': [], 'cycle_loss': [], 'identity_loss': []},
            'val': {'ssim_A2B': [], 'ssim_B2A': [], 'psnr_A2B': [], 'psnr_B2A': []},
            'learning_rate': [],
            'epoch_times': []
        }
        
        self.start_epoch = 0
        self.best_val_ssim = 0
        self.global_step = 0
        
        # Save config
        self._save_config()
        
    def _print_header(self):
        """Print training header."""
        print("\n" + "=" * 60)
        print("  NeuroScope: 2.5D SA-CycleGAN MRI Harmonization")
        print("  Cross-Site Brain MRI Translation")
        print("=" * 60)
        
    def _save_config(self):
        """Save experiment configuration."""
        config_dict = {
            'model': self.config.__dict__,
            'training': {
                'device': str(self.device),
                'experiment_name': self.experiment_name,
                'train_batches': len(self.train_loader),
                'val_batches': len(self.val_loader),
                'test_batches': len(self.test_loader)
            }
        }
        with open(self.experiment_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def compute_ssim(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute SSIM between two tensors."""
        x = x.detach().cpu()
        y = y.detach().cpu()
        
        mu_x = x.mean()
        mu_y = y.mean()
        sigma_x = ((x - mu_x) ** 2).mean()
        sigma_y = ((y - mu_y) ** 2).mean()
        sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
        
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        return ssim.item()
    
    def compute_psnr(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute PSNR between two tensors."""
        mse = ((x - y) ** 2).mean().item()
        if mse == 0:
            return 100.0
        return 10 * np.log10(1.0 / mse)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute single training step."""
        real_A = batch['A'].to(self.device)
        real_B = batch['B'].to(self.device)
        center_A = batch['A_center'].to(self.device)
        center_B = batch['B_center'].to(self.device)
        
        # ================================================================
        # Train Generators
        # ================================================================
        self.opt_G.zero_grad()
        
        # Generate fake images
        fake_B = self.model.G_A2B(real_A)
        fake_A = self.model.G_B2A(real_B)
        
        # Create 3-slice input for cycle consistency
        fake_B_3slice = fake_B.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        fake_B_3slice = fake_B_3slice.view(fake_B.size(0), -1, fake_B.size(2), fake_B.size(3))
        fake_A_3slice = fake_A.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        fake_A_3slice = fake_A_3slice.view(fake_A.size(0), -1, fake_A.size(2), fake_A.size(3))
        
        # Cycle consistency
        rec_A = self.model.G_B2A(fake_B_3slice)
        rec_B = self.model.G_A2B(fake_A_3slice)
        
        loss_cycle_A = self.losses.cycle_loss(center_A, rec_A)
        loss_cycle_B = self.losses.cycle_loss(center_B, rec_B)
        
        # Identity loss
        identity_A = self.model.G_B2A(real_A)
        identity_B = self.model.G_A2B(real_B)
        
        loss_identity_A = self.losses.identity_loss(center_A, identity_A)
        loss_identity_B = self.losses.identity_loss(center_B, identity_B)
        
        # GAN loss
        pred_fake_B = self.model.D_B(fake_B)
        pred_fake_A = self.model.D_A(fake_A)
        
        loss_gan_A2B = self.losses.gan_loss.generator_loss(pred_fake_B)
        loss_gan_B2A = self.losses.gan_loss.generator_loss(pred_fake_A)
        
        # SSIM loss
        loss_ssim = self.losses.ssim_loss(center_A, rec_A) + \
                    self.losses.ssim_loss(center_B, rec_B)
        
        # Total generator loss
        loss_G = (loss_gan_A2B + loss_gan_B2A + 
                  loss_cycle_A + loss_cycle_B + 
                  loss_identity_A + loss_identity_B +
                  loss_ssim)
        
        loss_G.backward()
        self.opt_G.step()
        
        # ================================================================
        # Train Discriminators
        # ================================================================
        self.opt_D.zero_grad()
        
        # Use replay buffer for stability
        fake_A_buffer = self.fake_A_buffer.push_and_pop(fake_A.detach())
        fake_B_buffer = self.fake_B_buffer.push_and_pop(fake_B.detach())
        
        # D_A loss
        pred_real_A = self.model.D_A(center_A)
        pred_fake_A = self.model.D_A(fake_A_buffer)
        loss_D_A = self.losses.gan_loss.discriminator_loss(pred_real_A, pred_fake_A)
        
        # D_B loss
        pred_real_B = self.model.D_B(center_B)
        pred_fake_B = self.model.D_B(fake_B_buffer)
        loss_D_B = self.losses.gan_loss.discriminator_loss(pred_real_B, pred_fake_B)
        
        loss_D = (loss_D_A + loss_D_B) * 0.5
        loss_D.backward()
        self.opt_D.step()
        
        return {
            'G_loss': loss_G.item(),
            'D_loss': loss_D.item(),
            'cycle_A': loss_cycle_A.item(),
            'cycle_B': loss_cycle_B.item(),
            'identity_A': loss_identity_A.item(),
            'identity_B': loss_identity_B.item(),
            'gan_A2B': loss_gan_A2B.item(),
            'gan_B2A': loss_gan_B2A.item(),
            'ssim_loss': loss_ssim.item()
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {
            'G_loss': 0, 'D_loss': 0,
            'cycle_A': 0, 'cycle_B': 0,
            'identity_A': 0, 'identity_B': 0,
            'gan_A2B': 0, 'gan_B2A': 0,
            'ssim_loss': 0
        }
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            ncols=120,
            leave=True
        )
        
        for batch_idx, batch in enumerate(pbar):
            losses = self.train_step(batch)
            
            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key]
            
            # TensorBoard logging
            self.global_step += 1
            if self.global_step % 100 == 0:
                self.writer.add_scalar('Train/G_loss', losses['G_loss'], self.global_step)
                self.writer.add_scalar('Train/D_loss', losses['D_loss'], self.global_step)
                self.writer.add_scalar('Train/Cycle_loss', 
                                       losses['cycle_A'] + losses['cycle_B'], 
                                       self.global_step)
            
            # Update progress bar
            pbar.set_postfix({
                'G': f'{losses["G_loss"]:.3f}',
                'D': f'{losses["D_loss"]:.3f}',
                'cyc': f'{losses["cycle_A"] + losses["cycle_B"]:.3f}'
            })
        
        # Average losses
        n_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
            
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        metrics = {
            'ssim_A2B': [], 'ssim_B2A': [],
            'psnr_A2B': [], 'psnr_B2A': []
        }
        
        for batch in tqdm(self.val_loader, desc="Validating", ncols=100, leave=False):
            real_A = batch['A'].to(self.device)
            real_B = batch['B'].to(self.device)
            center_A = batch['A_center'].to(self.device)
            center_B = batch['B_center'].to(self.device)
            
            # Generate and reconstruct
            fake_B = self.model.G_A2B(real_A)
            fake_A = self.model.G_B2A(real_B)
            
            fake_B_3slice = fake_B.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            fake_B_3slice = fake_B_3slice.view(fake_B.size(0), -1, fake_B.size(2), fake_B.size(3))
            fake_A_3slice = fake_A.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            fake_A_3slice = fake_A_3slice.view(fake_A.size(0), -1, fake_A.size(2), fake_A.size(3))
            
            rec_A = self.model.G_B2A(fake_B_3slice)
            rec_B = self.model.G_A2B(fake_A_3slice)
            
            # Compute metrics per sample
            for i in range(center_A.size(0)):
                metrics['ssim_A2B'].append(self.compute_ssim(center_A[i], rec_A[i]))
                metrics['ssim_B2A'].append(self.compute_ssim(center_B[i], rec_B[i]))
                metrics['psnr_A2B'].append(self.compute_psnr(center_A[i], rec_A[i]))
                metrics['psnr_B2A'].append(self.compute_psnr(center_B[i], rec_B[i]))
        
        return {
            'ssim_A2B': np.mean(metrics['ssim_A2B']),
            'ssim_B2A': np.mean(metrics['ssim_B2A']),
            'psnr_A2B': np.mean(metrics['psnr_A2B']),
            'psnr_B2A': np.mean(metrics['psnr_B2A']),
            'ssim_std_A2B': np.std(metrics['ssim_A2B']),
            'ssim_std_B2A': np.std(metrics['ssim_B2A'])
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'opt_G_state_dict': self.opt_G.state_dict(),
            'opt_D_state_dict': self.opt_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'history': self.history,
            'best_val_ssim': self.best_val_ssim,
            'global_step': self.global_step,
            'config': self.config.__dict__
        }
        
        ckpt_dir = self.experiment_dir / "checkpoints"
        
        # Save latest
        torch.save(checkpoint, ckpt_dir / 'checkpoint_latest.pth')
        
        # Save periodic
        if epoch % 10 == 0:
            torch.save(checkpoint, ckpt_dir / f'checkpoint_epoch_{epoch}.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, ckpt_dir / 'checkpoint_best.pth')
            # Also save to main checkpoints folder
            torch.save(checkpoint, PROJECT_ROOT / 'checkpoints' / 'best_model.pth')
            print(f"  [best] new best model saved (ssim: {self.best_val_ssim:.4f})")
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
        self.opt_D.load_state_dict(checkpoint['opt_D_state_dict'])
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        self.history = checkpoint['history']
        self.best_val_ssim = checkpoint['best_val_ssim']
        self.global_step = checkpoint.get('global_step', 0)
        self.start_epoch = checkpoint['epoch'] + 1
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def save_samples(self, epoch: int):
        """Save sample images for visualization."""
        self.model.eval()
        
        with torch.no_grad():
            batch = next(iter(self.val_loader))
            real_A = batch['A'][:4].to(self.device)
            real_B = batch['B'][:4].to(self.device)
            
            fake_B = self.model.G_A2B(real_A)
            fake_A = self.model.G_B2A(real_B)
            
            # Log to TensorBoard
            self.writer.add_images(f'Samples/Real_A', 
                                   batch['A_center'][:4, :1], epoch)
            self.writer.add_images(f'Samples/Fake_B', 
                                   fake_B[:4, :1], epoch)
            self.writer.add_images(f'Samples/Real_B', 
                                   batch['B_center'][:4, :1], epoch)
            self.writer.add_images(f'Samples/Fake_A', 
                                   fake_A[:4, :1], epoch)
    
    def train(self, epochs: int, validate_every: int = 5, save_every: int = 10):
        """Full training loop."""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Total epochs: {epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Validate every: {validate_every} epochs")
        print(f"Save every: {save_every} epochs")
        print("=" * 60 + "\n")
        
        training_start = time.time()
        
        for epoch in range(self.start_epoch, epochs):
            epoch_start = time.time()
            
            # Train
            train_losses = self.train_epoch(epoch + 1)
            
            # Update schedulers
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # Log training losses
            self.history['train']['G_loss'].append(train_losses['G_loss'])
            self.history['train']['D_loss'].append(train_losses['D_loss'])
            self.history['train']['cycle_loss'].append(
                train_losses['cycle_A'] + train_losses['cycle_B']
            )
            self.history['train']['identity_loss'].append(
                train_losses['identity_A'] + train_losses['identity_B']
            )
            self.history['learning_rate'].append(self.scheduler_G.get_last_lr()[0])
            
            epoch_time = time.time() - epoch_start
            self.history['epoch_times'].append(epoch_time)
            
            # Validate
            if (epoch + 1) % validate_every == 0:
                val_metrics = self.validate()
                
                self.history['val']['ssim_A2B'].append(val_metrics['ssim_A2B'])
                self.history['val']['ssim_B2A'].append(val_metrics['ssim_B2A'])
                self.history['val']['psnr_A2B'].append(val_metrics['psnr_A2B'])
                self.history['val']['psnr_B2A'].append(val_metrics['psnr_B2A'])
                
                # TensorBoard
                self.writer.add_scalar('Val/SSIM_A2B', val_metrics['ssim_A2B'], epoch + 1)
                self.writer.add_scalar('Val/SSIM_B2A', val_metrics['ssim_B2A'], epoch + 1)
                self.writer.add_scalar('Val/PSNR_A2B', val_metrics['psnr_A2B'], epoch + 1)
                self.writer.add_scalar('Val/PSNR_B2A', val_metrics['psnr_B2A'], epoch + 1)
                
                avg_ssim = (val_metrics['ssim_A2B'] + val_metrics['ssim_B2A']) / 2
                is_best = avg_ssim > self.best_val_ssim
                if is_best:
                    self.best_val_ssim = avg_ssim
                
                # Print epoch summary
                print(f"\n{'='*60}")
                print(f"Epoch {epoch + 1}/{epochs} Summary ({epoch_time:.1f}s)")
                print(f"{'='*60}")
                print(f"Train Losses:")
                print(f"  Generator: {train_losses['G_loss']:.4f}")
                print(f"  Discriminator: {train_losses['D_loss']:.4f}")
                print(f"  Cycle: {train_losses['cycle_A'] + train_losses['cycle_B']:.4f}")
                print(f"  Identity: {train_losses['identity_A'] + train_losses['identity_B']:.4f}")
                print(f"\nValidation Metrics:")
                print(f"  SSIM A→B→A: {val_metrics['ssim_A2B']:.4f} ± {val_metrics['ssim_std_A2B']:.4f}")
                print(f"  SSIM B→A→B: {val_metrics['ssim_B2A']:.4f} ± {val_metrics['ssim_std_B2A']:.4f}")
                print(f"  PSNR A→B→A: {val_metrics['psnr_A2B']:.2f} dB")
                print(f"  PSNR B→A→B: {val_metrics['psnr_B2A']:.2f} dB")
                print(f"\nLearning Rate: {self.scheduler_G.get_last_lr()[0]:.2e}")
                
                # Save samples
                self.save_samples(epoch + 1)
                
                # Save checkpoint
                self.save_checkpoint(epoch + 1, is_best)
            
            # Save periodic checkpoint
            elif (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1)
                print(f"\n  Checkpoint saved at epoch {epoch + 1}")
        
        total_time = time.time() - training_start
        
        # Final summary
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total training time: {total_time / 3600:.2f} hours")
        print(f"Average epoch time: {np.mean(self.history['epoch_times']):.1f}s")
        print(f"Best validation SSIM: {self.best_val_ssim:.4f}")
        print(f"Checkpoints saved to: {self.experiment_dir / 'checkpoints'}")
        print(f"TensorBoard logs: {PROJECT_ROOT / 'runs' / self.experiment_name}")
        print("=" * 60)
        
        # Save final history
        with open(self.experiment_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Close TensorBoard writer
        self.writer.close()
        
        return self.history


def load_config(config_path: str) -> dict:
    """load configuration from yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='train baseline 2.5d cyclegan for mri harmonization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # config file (takes precedence)
    parser.add_argument('--config', type=str, default=None,
                        help='path to yaml config file')

    # data arguments
    parser.add_argument('--brats_dir', type=str,
                        default=str(PROJECT_ROOT / 'preprocessed' / 'brats'),
                        help='path to brats data')
    parser.add_argument('--upenn_dir', type=str,
                        default=str(PROJECT_ROOT / 'preprocessed' / 'upenn'),
                        help='path to upenn data')
    parser.add_argument('--output_dir', type=str,
                        default=str(PROJECT_ROOT / 'experiments'),
                        help='output directory for experiments')

    # training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='training batch size')
    parser.add_argument('--image_size', type=int, default=128,
                        help='image size (height and width)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of data loader workers')

    # model arguments
    parser.add_argument('--ngf', type=int, default=64,
                        help='number of generator filters')
    parser.add_argument('--ndf', type=int, default=64,
                        help='number of discriminator filters')
    parser.add_argument('--n_residual', type=int, default=9,
                        help='number of residual blocks')

    # loss weights
    parser.add_argument('--lambda_cycle', type=float, default=10.0,
                        help='cycle consistency loss weight')
    parser.add_argument('--lambda_identity', type=float, default=5.0,
                        help='identity loss weight')
    parser.add_argument('--lambda_ssim', type=float, default=1.0,
                        help='ssim loss weight')

    # training settings
    parser.add_argument('--validate_every', type=int, default=5,
                        help='validate every n epochs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='save checkpoint every n epochs')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='experiment name (default: timestamp)')

    # resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume from')

    args = parser.parse_args()

    # load config from yaml if provided
    if args.config:
        print(f"[config] loading from {args.config}")
        cfg = load_config(args.config)

        # override args with config values
        for key, value in cfg.items():
            if hasattr(args, key):
                setattr(args, key, value)

        # handle special mappings
        if 'lr_G' in cfg:
            args.lr = cfg['lr_G']
        if 'n_residual_blocks' in cfg:
            args.n_residual = cfg['n_residual_blocks']

    # create config
    config = BaselineCycleGAN25DConfig(
        ngf=args.ngf,
        ndf=args.ndf,
        n_residual_blocks=args.n_residual
    )

    # store loss weights for later use
    config.lambda_cycle = args.lambda_cycle
    config.lambda_identity = args.lambda_identity
    config.lambda_ssim = args.lambda_ssim

    # create trainer
    trainer = BaselineCycleGAN25DTrainer(
        config=config,
        brats_dir=args.brats_dir,
        upenn_dir=args.upenn_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        num_workers=args.num_workers,
        experiment_name=args.experiment_name
    )

    # resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # train
    trainer.train(
        epochs=args.epochs,
        validate_every=args.validate_every,
        save_every=args.save_every
    )


if __name__ == '__main__':
    main()
