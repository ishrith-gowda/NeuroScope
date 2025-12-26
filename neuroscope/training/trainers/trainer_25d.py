"""
Full Training Script for 2.5D SA-CycleGAN.

Complete training pipeline for brain MRI harmonization:
- BraTS (multi-institutional) ↔ UPenn-GBM (single-institution)

Usage:
    python train.py --epochs 100 --batch_size 4 --image_size 128

Author: NeuroScope Research Team
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified.models.architectures.sa_cyclegan_25d import (
    SACycleGAN25D, SACycleGAN25DConfig, create_model
)
from unified.data.dataset_25d import UnpairedMRIDataset25D, create_dataloaders
from unified.models.losses import CombinedLoss, LSGANLoss


class ReplayBuffer:
    """Image buffer for discriminator training stability."""
    
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


class Trainer:
    """
    Complete training pipeline for 2.5D SA-CycleGAN.
    """
    
    def __init__(
        self,
        config: SACycleGAN25DConfig,
        brats_dir: str,
        upenn_dir: str,
        output_dir: str,
        batch_size: int = 4,
        image_size: int = 128,
        lr: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
        num_workers: int = 4,
        use_amp: bool = True,
        device: str = 'auto'
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
        print(f"Using device: {self.device}")
        
        # Mixed precision
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Create model
        print("\n" + "=" * 60)
        print("Creating 2.5D SA-CycleGAN Model")
        print("=" * 60)
        self.model = create_model(config)
        self.model = self.model.to(self.device)
        
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
        
        # Optimizers
        self.opt_G = optim.Adam(
            list(self.model.G_A2B.parameters()) + list(self.model.G_B2A.parameters()),
            lr=lr, betas=(beta1, beta2)
        )
        self.opt_D = optim.Adam(
            list(self.model.D_A.parameters()) + list(self.model.D_B.parameters()),
            lr=lr, betas=(beta1, beta2)
        )
        
        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_G, T_max=100, eta_min=1e-6
        )
        self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_D, T_max=100, eta_min=1e-6
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
            'train_G_loss': [], 'train_D_loss': [],
            'train_cycle_loss': [], 'train_identity_loss': [],
            'val_ssim_A2B': [], 'val_ssim_B2A': [],
            'val_psnr_A2B': [], 'val_psnr_B2A': [],
            'learning_rate': []
        }
        
        self.start_epoch = 0
        self.best_val_ssim = 0
        
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
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {
            'G_loss': 0, 'D_loss': 0,
            'cycle_A': 0, 'cycle_B': 0,
            'identity_A': 0, 'identity_B': 0,
            'gan_A2B': 0, 'gan_B2A': 0
        }
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            ncols=100,
            leave=True
        )
        
        for batch_idx, batch in enumerate(pbar):
            real_A = batch['A'].to(self.device)  # [B, 12, H, W]
            real_B = batch['B'].to(self.device)
            center_A = batch['A_center'].to(self.device)  # [B, 4, H, W]
            center_B = batch['B_center'].to(self.device)
            
            # ================================================================
            # Train Generators
            # ================================================================
            self.opt_G.zero_grad()
            
            # Generate fake images
            fake_B = self.model.G_A2B(real_A)  # [B, 4, H, W]
            fake_A = self.model.G_B2A(real_B)
            
            # For cycle consistency, we need 3 slices of the fake
            # Simplified: repeat the generated slice
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
            
            # SSIM loss for structural preservation
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
            
            # Accumulate losses
            epoch_losses['G_loss'] += loss_G.item()
            epoch_losses['D_loss'] += loss_D.item()
            epoch_losses['cycle_A'] += loss_cycle_A.item()
            epoch_losses['cycle_B'] += loss_cycle_B.item()
            epoch_losses['identity_A'] += loss_identity_A.item()
            epoch_losses['identity_B'] += loss_identity_B.item()
            epoch_losses['gan_A2B'] += loss_gan_A2B.item()
            epoch_losses['gan_B2A'] += loss_gan_B2A.item()
            
            # Update progress bar
            pbar.set_postfix({
                'G': f'{loss_G.item():.3f}',
                'D': f'{loss_D.item():.3f}',
                'cyc': f'{(loss_cycle_A + loss_cycle_B).item():.3f}'
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
        
        ssim_A2B = []
        ssim_B2A = []
        psnr_A2B = []
        psnr_B2A = []
        
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
            
            # Compute metrics
            for i in range(center_A.size(0)):
                ssim_A2B.append(self.compute_ssim(center_A[i], rec_A[i]))
                ssim_B2A.append(self.compute_ssim(center_B[i], rec_B[i]))
                psnr_A2B.append(self.compute_psnr(center_A[i], rec_A[i]))
                psnr_B2A.append(self.compute_psnr(center_B[i], rec_B[i]))
        
        return {
            'ssim_A2B': np.mean(ssim_A2B),
            'ssim_B2A': np.mean(ssim_B2A),
            'psnr_A2B': np.mean(psnr_A2B),
            'psnr_B2A': np.mean(psnr_B2A)
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
            'config': self.config.__dict__
        }
        
        # Save latest
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pth')
        
        # Save periodic
        if epoch % 10 == 0:
            torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{epoch}.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pth')
            print(f"  ⭐ New best model saved (SSIM: {self.best_val_ssim:.4f})")
    
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
        self.start_epoch = checkpoint['epoch'] + 1
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def train(self, epochs: int, validate_every: int = 5, save_every: int = 10):
        """Full training loop."""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Epochs: {epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Device: {self.device}")
        print("=" * 60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, epochs):
            epoch_start = time.time()
            
            # Train
            train_losses = self.train_epoch(epoch + 1)
            
            # Update schedulers
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # Log training losses
            self.history['train_G_loss'].append(train_losses['G_loss'])
            self.history['train_D_loss'].append(train_losses['D_loss'])
            self.history['train_cycle_loss'].append(
                train_losses['cycle_A'] + train_losses['cycle_B']
            )
            self.history['train_identity_loss'].append(
                train_losses['identity_A'] + train_losses['identity_B']
            )
            self.history['learning_rate'].append(self.scheduler_G.get_last_lr()[0])
            
            # Validate
            if (epoch + 1) % validate_every == 0:
                val_metrics = self.validate()
                
                self.history['val_ssim_A2B'].append(val_metrics['ssim_A2B'])
                self.history['val_ssim_B2A'].append(val_metrics['ssim_B2A'])
                self.history['val_psnr_A2B'].append(val_metrics['psnr_A2B'])
                self.history['val_psnr_B2A'].append(val_metrics['psnr_B2A'])
                
                avg_ssim = (val_metrics['ssim_A2B'] + val_metrics['ssim_B2A']) / 2
                is_best = avg_ssim > self.best_val_ssim
                if is_best:
                    self.best_val_ssim = avg_ssim
                
                # Print epoch summary
                epoch_time = time.time() - epoch_start
                print(f"\n  Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s)")
                print(f"  Train: G={train_losses['G_loss']:.4f}, D={train_losses['D_loss']:.4f}")
                print(f"  Val SSIM: A→B→A={val_metrics['ssim_A2B']:.4f}, B→A→B={val_metrics['ssim_B2A']:.4f}")
                print(f"  Val PSNR: A→B→A={val_metrics['psnr_A2B']:.2f}dB, B→A→B={val_metrics['psnr_B2A']:.2f}dB")
                print(f"  LR: {self.scheduler_G.get_last_lr()[0]:.2e}")
                
                # Save checkpoint
                self.save_checkpoint(epoch + 1, is_best)
            
            # Save periodic checkpoint
            elif (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1)
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Training Complete!")
        print(f"Total time: {total_time / 3600:.2f} hours")
        print(f"Best validation SSIM: {self.best_val_ssim:.4f}")
        print("=" * 60)
        
        # Save final history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train 2.5D SA-CycleGAN')
    
    # Data arguments
    parser.add_argument('--brats_dir', type=str, 
                        default='/Volumes/usb drive/neuroscope/preprocessed/brats',
                        help='Path to BraTS data')
    parser.add_argument('--upenn_dir', type=str,
                        default='/Volumes/usb drive/neuroscope/preprocessed/upenn',
                        help='Path to UPenn data')
    parser.add_argument('--output_dir', type=str,
                        default='/Volumes/usb drive/neuroscope/experiments',
                        help='Output directory for checkpoints')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--image_size', type=int, default=128, help='Image size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers')
    
    # Model arguments
    parser.add_argument('--ngf', type=int, default=64, help='Generator filters')
    parser.add_argument('--ndf', type=int, default=64, help='Discriminator filters')
    parser.add_argument('--n_residual', type=int, default=9, help='Residual blocks')
    
    # Loss weights
    parser.add_argument('--lambda_cycle', type=float, default=10.0)
    parser.add_argument('--lambda_identity', type=float, default=5.0)
    parser.add_argument('--lambda_ssim', type=float, default=1.0)
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create config
    config = SACycleGAN25DConfig(
        ngf=args.ngf,
        ndf=args.ndf,
        n_residual_blocks=args.n_residual,
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity,
        lambda_ssim=args.lambda_ssim
    )
    
    # Create trainer
    trainer = Trainer(
        config=config,
        brats_dir=args.brats_dir,
        upenn_dir=args.upenn_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        num_workers=args.num_workers
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(epochs=args.epochs)


if __name__ == '__main__':
    main()
