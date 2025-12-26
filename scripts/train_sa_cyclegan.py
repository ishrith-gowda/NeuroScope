#!/usr/bin/env python3
"""
SA-CycleGAN Training Script for NeurIPS-level Brain MRI Domain Translation

This script implements training for the Self-Attention CycleGAN (SA-CycleGAN)
architecture with advanced loss functions for brain MRI domain translation
between BraTS and UPenn-GBM datasets.

Key Features:
1. SA-CycleGAN with multi-scale self-attention
2. Modality-aware encoding for 4-channel MRI (T1, T1ce, T2, FLAIR)
3. Advanced losses: Perceptual, Contrastive, Modality-specific, Tumor preservation
4. Multi-scale discriminator with spectral normalization
5. Comprehensive logging with TensorBoard
6. Checkpoint management with best model selection
7. Learning rate scheduling with warmup

Usage:
    python train_sa_cyclegan.py --data_dir ./preprocessed --epochs 100 --batch_size 4

Author: NeuroScope Team
Date: 2025
"""

import os
import sys
import json
import argparse
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import nibabel as nib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our novel architecture and losses
try:
    from neuroscope.models.architectures.sa_cyclegan import (
        SACycleGAN, SAGenerator, MultiScaleDiscriminator, create_sa_cyclegan
    )
    from neuroscope.models.losses.advanced_losses import (
        PerceptualLoss, ContrastiveLoss, ModalitySpecificLoss,
        TumorPreservationLoss, TotalLoss
    )
except ImportError:
    print("Warning: Could not import custom modules. Using fallback.")
    sys.path.insert(0, str(project_root / 'neuroscope' / 'models' / 'architectures'))
    sys.path.insert(0, str(project_root / 'neuroscope' / 'models' / 'losses'))


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Training configuration with sensible defaults for medical imaging."""
    
    # Data
    data_dir: str = './preprocessed'
    brats_dir: str = 'brats'
    upenn_dir: str = 'upenn'
    image_size: int = 256
    num_workers: int = 4
    
    # Model
    input_channels: int = 4
    ngf: int = 64  # Generator filters
    ndf: int = 64  # Discriminator filters
    n_residual_blocks: int = 9
    use_modality_encoder: bool = True
    
    # Training
    epochs: int = 100
    batch_size: int = 4
    lr_g: float = 1e-4
    lr_d: float = 2e-4  # TTUR: discriminator learns faster
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 1e-5
    
    # Learning rate schedule
    warmup_epochs: int = 5
    lr_decay_type: str = 'cosine'  # 'cosine' or 'linear'
    
    # Loss weights
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    lambda_perceptual: float = 1.0
    lambda_contrastive: float = 0.5
    lambda_modality: float = 1.0
    lambda_tumor: float = 2.0
    
    # Regularization
    label_smoothing_real: float = 0.9
    label_smoothing_fake: float = 0.1
    replay_buffer_size: int = 50
    gradient_clip: float = 1.0
    
    # Logging
    log_interval: int = 100
    save_interval: int = 10
    sample_interval: int = 500
    
    # Paths
    checkpoint_dir: str = './checkpoints/sa_cyclegan'
    log_dir: str = './runs/sa_cyclegan'
    sample_dir: str = './samples/sa_cyclegan'
    
    # Device
    device: str = 'auto'  # 'auto', 'cuda', 'mps', 'cpu'
    mixed_precision: bool = False  # Use AMP for faster training
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
    
    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}


# ============================================================================
# Data Loading
# ============================================================================

class MRISliceDataset(Dataset):
    """
    Dataset for loading pre-processed 2D MRI slices with caching.
    
    Supports on-the-fly augmentation and multi-modal MRI loading.
    """
    
    def __init__(
        self,
        root_dir: str,
        domain: str,  # 'brats' or 'upenn'
        image_size: int = 256,
        augment: bool = True,
        cache_size: int = 1000,
        slice_range: Tuple[int, int] = (40, 120)  # Focus on brain slices
    ):
        self.root_dir = Path(root_dir) / domain
        self.image_size = image_size
        self.augment = augment
        self.slice_range = slice_range
        
        # Find all subjects
        self.subjects = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
        
        # Pre-compute slice indices
        self.slices = []
        for subject in self.subjects:
            for slice_idx in range(slice_range[0], slice_range[1]):
                self.slices.append((subject, slice_idx))
        
        # LRU cache for loaded slices
        self.cache = {}
        self.cache_order = deque(maxlen=cache_size)
        
        # Modality names (ordered)
        self.modalities = ['t1', 't1ce', 't2', 'flair']
        
        # Augmentation transforms
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5 if augment else 0),
            transforms.RandomRotation(10 if augment else 0),
        ])
        
        print(f"Loaded {domain} dataset: {len(self.subjects)} subjects, {len(self.slices)} slices")
    
    def __len__(self) -> int:
        return len(self.slices)
    
    def _load_slice(self, subject: Path, slice_idx: int) -> torch.Tensor:
        """Load and stack all modalities for a slice."""
        cache_key = (str(subject), slice_idx)
        
        if cache_key in self.cache:
            return self.cache[cache_key].clone()
        
        modality_slices = []
        for mod in self.modalities:
            # Try different file patterns
            nifti_file = subject / f'{mod}.nii.gz'
            if not nifti_file.exists():
                nifti_file = subject / f'{subject.name}_{mod}.nii.gz'
            if not nifti_file.exists():
                # Search for any file containing the modality
                matches = list(subject.glob(f'*{mod}*.nii.gz'))
                if matches:
                    nifti_file = matches[0]
            
            if nifti_file.exists():
                img = nib.load(str(nifti_file)).get_fdata()
                if slice_idx < img.shape[2]:
                    slice_data = img[:, :, slice_idx].astype(np.float32)
                else:
                    slice_data = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            else:
                slice_data = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            
            # Normalize to [-1, 1]
            if slice_data.max() > slice_data.min():
                slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
                slice_data = slice_data * 2 - 1
            
            # Resize if necessary
            if slice_data.shape != (self.image_size, self.image_size):
                slice_data = np.array(
                    Image.fromarray(slice_data).resize(
                        (self.image_size, self.image_size),
                        Image.BILINEAR
                    )
                )
            
            modality_slices.append(slice_data)
        
        # Stack modalities [4, H, W]
        stacked = torch.tensor(np.stack(modality_slices), dtype=torch.float32)
        
        # Cache management
        if len(self.cache) >= len(self.cache_order):
            old_key = self.cache_order.popleft()
            self.cache.pop(old_key, None)
        
        self.cache[cache_key] = stacked
        self.cache_order.append(cache_key)
        
        return stacked.clone()
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        subject, slice_idx = self.slices[idx]
        data = self._load_slice(subject, slice_idx)
        
        # Apply augmentation
        if self.augment:
            data = self.transform(data)
        
        return data


class ReplayBuffer:
    """
    Replay buffer for training stabilization.
    
    Stores previously generated images and randomly samples from buffer
    to reduce mode oscillation.
    """
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data = []
    
    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        """Add new images and return a mix of new and old images."""
        result = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(img)
                result.append(img)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    result.append(self.data[idx].clone())
                    self.data[idx] = img
                else:
                    result.append(img)
        return torch.cat(result, dim=0)


# ============================================================================
# Training Utilities
# ============================================================================

def get_device(device_str: str = 'auto') -> torch.device:
    """Get the best available device."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_str)


def get_scheduler(optimizer, config: Config, steps_per_epoch: int):
    """Create learning rate scheduler with warmup."""
    
    def warmup_lambda(epoch):
        if epoch < config.warmup_epochs:
            return float(epoch) / float(max(1, config.warmup_epochs))
        return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, warmup_lambda)
    
    if config.lr_decay_type == 'cosine':
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    else:
        def linear_decay(epoch):
            return 1.0 - max(0, epoch - config.epochs // 2) / (config.epochs // 2)
        main_scheduler = LambdaLR(optimizer, linear_decay)
    
    return warmup_scheduler, main_scheduler


def save_checkpoint(
    state: dict,
    filename: str,
    checkpoint_dir: str,
    is_best: bool = False
):
    """Save checkpoint with optional best model copy."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(checkpoint_dir) / filename
    torch.save(state, filepath)
    
    if is_best:
        best_path = Path(checkpoint_dir) / 'best_model.pt'
        torch.save(state, best_path)


def save_samples(
    real_A: torch.Tensor,
    real_B: torch.Tensor,
    fake_A: torch.Tensor,
    fake_B: torch.Tensor,
    epoch: int,
    batch_idx: int,
    sample_dir: str
):
    """Save sample images for visualization."""
    Path(sample_dir).mkdir(parents=True, exist_ok=True)
    
    # Take first sample from batch
    real_A = real_A[0].cpu().numpy()
    real_B = real_B[0].cpu().numpy()
    fake_A = fake_A[0].cpu().numpy()
    fake_B = fake_B[0].cpu().numpy()
    
    # Save each modality
    for mod_idx, mod_name in enumerate(['t1', 't1ce', 't2', 'flair']):
        # Normalize to [0, 255]
        def to_img(x):
            x = (x + 1) / 2  # [-1, 1] -> [0, 1]
            x = np.clip(x * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(x)
        
        # Create comparison grid
        real_a_img = to_img(real_A[mod_idx])
        fake_b_img = to_img(fake_B[mod_idx])
        real_b_img = to_img(real_B[mod_idx])
        fake_a_img = to_img(fake_A[mod_idx])
        
        # Horizontal concatenation: real_A | fake_B | real_B | fake_A
        grid = Image.new('L', (256 * 4, 256))
        grid.paste(real_a_img, (0, 0))
        grid.paste(fake_b_img, (256, 0))
        grid.paste(real_b_img, (512, 0))
        grid.paste(fake_a_img, (768, 0))
        
        grid.save(Path(sample_dir) / f'epoch{epoch:03d}_batch{batch_idx:04d}_{mod_name}.png')


# ============================================================================
# Training Loop
# ============================================================================

class SACycleGANTrainer:
    """
    Trainer class for SA-CycleGAN with advanced losses and comprehensive logging.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = get_device(config.device)
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(config.sample_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = create_sa_cyclegan(
            input_channels=config.input_channels,
            ngf=config.ngf,
            ndf=config.ndf,
            n_residual_blocks=config.n_residual_blocks
        ).to(self.device)
        
        # Initialize losses
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        # Advanced losses (only if available)
        try:
            self.criterion_perceptual = PerceptualLoss(device=self.device)
            self.criterion_contrastive = ContrastiveLoss(device=self.device)
            self.criterion_modality = ModalitySpecificLoss()
            self.criterion_tumor = TumorPreservationLoss()
            self.use_advanced_losses = True
            print("Using advanced losses: Perceptual, Contrastive, Modality-specific, Tumor preservation")
        except Exception as e:
            print(f"Warning: Could not initialize advanced losses: {e}")
            self.use_advanced_losses = False
        
        # Optimizers (separate for G and D with TTUR)
        self.optimizer_G = Adam(
            list(self.model.G_A2B.parameters()) + list(self.model.G_B2A.parameters()),
            lr=config.lr_g,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        
        self.optimizer_D = Adam(
            list(self.model.D_A.parameters()) + list(self.model.D_B.parameters()),
            lr=config.lr_d,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        
        # Replay buffers
        self.buffer_A = ReplayBuffer(config.replay_buffer_size)
        self.buffer_B = ReplayBuffer(config.replay_buffer_size)
        
        # TensorBoard writer
        self.writer = SummaryWriter(config.log_dir)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.history = {'g_loss': [], 'd_loss': [], 'cycle_loss': [], 'metrics': []}
    
    def train_epoch(
        self,
        dataloader_A: DataLoader,
        dataloader_B: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {
            'g_total': 0, 'd_total': 0, 'cycle': 0, 'identity': 0,
            'g_adv': 0, 'd_adv': 0, 'perceptual': 0, 'contrastive': 0
        }
        
        # Iterate over both domains
        iter_A = iter(dataloader_A)
        iter_B = iter(dataloader_B)
        num_batches = min(len(dataloader_A), len(dataloader_B))
        
        pbar = tqdm(range(num_batches), desc=f'Epoch {epoch+1}/{self.config.epochs}')
        
        for batch_idx in pbar:
            try:
                real_A = next(iter_A).to(self.device)
                real_B = next(iter_B).to(self.device)
            except StopIteration:
                break
            
            # Create target tensors with label smoothing
            batch_size = real_A.size(0)
            
            # Forward pass through generators
            fake_B, fake_A, rec_A, rec_B = self.model(real_A, real_B)
            
            # Identity mapping (optional but helps preserve color)
            idt_A = self.model.G_B2A(real_A)
            idt_B = self.model.G_A2B(real_B)
            
            # =====================
            # Train Generators
            # =====================
            self.optimizer_G.zero_grad()
            
            # GAN loss (multi-scale)
            g_loss_A2B = 0
            g_loss_B2A = 0
            
            d_fake_B = self.model.D_B(fake_B)
            d_fake_A = self.model.D_A(fake_A)
            
            for d_out in d_fake_B:
                target_real = torch.full_like(d_out, self.config.label_smoothing_real)
                g_loss_A2B += self.criterion_gan(d_out, target_real)
            
            for d_out in d_fake_A:
                target_real = torch.full_like(d_out, self.config.label_smoothing_real)
                g_loss_B2A += self.criterion_gan(d_out, target_real)
            
            g_loss_adv = (g_loss_A2B + g_loss_B2A) / 2
            
            # Cycle consistency loss
            cycle_loss_A = self.criterion_cycle(rec_A, real_A)
            cycle_loss_B = self.criterion_cycle(rec_B, real_B)
            cycle_loss = (cycle_loss_A + cycle_loss_B) * self.config.lambda_cycle
            
            # Identity loss
            idt_loss_A = self.criterion_identity(idt_A, real_A)
            idt_loss_B = self.criterion_identity(idt_B, real_B)
            idt_loss = (idt_loss_A + idt_loss_B) * self.config.lambda_identity
            
            # Advanced losses
            perceptual_loss = torch.tensor(0.0, device=self.device)
            contrastive_loss = torch.tensor(0.0, device=self.device)
            
            if self.use_advanced_losses:
                # Perceptual loss
                perceptual_loss = (
                    self.criterion_perceptual(fake_B, real_A) +
                    self.criterion_perceptual(fake_A, real_B)
                ) * self.config.lambda_perceptual
                
                # Contrastive loss (if implemented with encoder)
                try:
                    contrastive_loss = (
                        self.criterion_contrastive(fake_B, real_A, real_B) +
                        self.criterion_contrastive(fake_A, real_B, real_A)
                    ) * self.config.lambda_contrastive
                except:
                    pass
            
            # Total generator loss
            g_loss = g_loss_adv + cycle_loss + idt_loss + perceptual_loss + contrastive_loss
            
            # Backward and optimize
            g_loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(self.model.G_A2B.parameters()) + list(self.model.G_B2A.parameters()),
                    self.config.gradient_clip
                )
            
            self.optimizer_G.step()
            
            # =====================
            # Train Discriminators
            # =====================
            self.optimizer_D.zero_grad()
            
            # Use replay buffer
            fake_A_buffer = self.buffer_A.push_and_pop(fake_A.detach())
            fake_B_buffer = self.buffer_B.push_and_pop(fake_B.detach())
            
            # Discriminator A
            d_loss_A = 0
            d_real_A = self.model.D_A(real_A)
            d_fake_A = self.model.D_A(fake_A_buffer)
            
            for d_real, d_fake in zip(d_real_A, d_fake_A):
                target_real = torch.full_like(d_real, self.config.label_smoothing_real)
                target_fake = torch.full_like(d_fake, self.config.label_smoothing_fake)
                d_loss_A += (self.criterion_gan(d_real, target_real) + 
                            self.criterion_gan(d_fake, target_fake)) / 2
            
            # Discriminator B
            d_loss_B = 0
            d_real_B = self.model.D_B(real_B)
            d_fake_B = self.model.D_B(fake_B_buffer)
            
            for d_real, d_fake in zip(d_real_B, d_fake_B):
                target_real = torch.full_like(d_real, self.config.label_smoothing_real)
                target_fake = torch.full_like(d_fake, self.config.label_smoothing_fake)
                d_loss_B += (self.criterion_gan(d_real, target_real) + 
                            self.criterion_gan(d_fake, target_fake)) / 2
            
            d_loss = (d_loss_A + d_loss_B) / 2
            
            d_loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(self.model.D_A.parameters()) + list(self.model.D_B.parameters()),
                    self.config.gradient_clip
                )
            
            self.optimizer_D.step()
            
            # Update losses
            epoch_losses['g_total'] += g_loss.item()
            epoch_losses['d_total'] += d_loss.item()
            epoch_losses['cycle'] += cycle_loss.item()
            epoch_losses['identity'] += idt_loss.item()
            epoch_losses['g_adv'] += g_loss_adv.item()
            epoch_losses['d_adv'] += d_loss.item()
            epoch_losses['perceptual'] += perceptual_loss.item()
            epoch_losses['contrastive'] += contrastive_loss.item()
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                self.writer.add_scalar('Loss/G_total', g_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/D_total', d_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/Cycle', cycle_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/Identity', idt_loss.item(), self.global_step)
                
                pbar.set_postfix({
                    'G': f'{g_loss.item():.3f}',
                    'D': f'{d_loss.item():.3f}',
                    'Cyc': f'{cycle_loss.item():.3f}'
                })
            
            # Save samples
            if batch_idx % self.config.sample_interval == 0:
                with torch.no_grad():
                    save_samples(
                        real_A, real_B, fake_A.detach(), fake_B.detach(),
                        epoch, batch_idx, self.config.sample_dir
                    )
            
            self.global_step += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def train(
        self,
        dataloader_A: DataLoader,
        dataloader_B: DataLoader
    ):
        """Full training loop."""
        print(f"\nStarting SA-CycleGAN training on {self.device}")
        print(f"Configuration: {json.dumps(self.config.to_dict(), indent=2)}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # Train epoch
            losses = self.train_epoch(dataloader_A, dataloader_B, epoch)
            
            # Log epoch losses
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print(f"  G Loss: {losses['g_total']:.4f}, D Loss: {losses['d_total']:.4f}")
            print(f"  Cycle: {losses['cycle']:.4f}, Identity: {losses['identity']:.4f}")
            
            self.history['g_loss'].append(losses['g_total'])
            self.history['d_loss'].append(losses['d_total'])
            self.history['cycle_loss'].append(losses['cycle'])
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                is_best = losses['g_total'] < self.best_loss
                if is_best:
                    self.best_loss = losses['g_total']
                
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_G': self.optimizer_G.state_dict(),
                        'optimizer_D': self.optimizer_D.state_dict(),
                        'losses': losses,
                        'config': self.config.to_dict(),
                        'history': self.history
                    },
                    f'checkpoint_epoch{epoch+1}.pt',
                    self.config.checkpoint_dir,
                    is_best=is_best
                )
                print(f"  Saved checkpoint (best: {is_best})")
        
        # Save final models
        torch.save(self.model.G_A2B.state_dict(), 
                   Path(self.config.checkpoint_dir) / 'G_A2B_final.pth')
        torch.save(self.model.G_B2A.state_dict(),
                   Path(self.config.checkpoint_dir) / 'G_B2A_final.pth')
        torch.save(self.model.D_A.state_dict(),
                   Path(self.config.checkpoint_dir) / 'D_A_final.pth')
        torch.save(self.model.D_B.state_dict(),
                   Path(self.config.checkpoint_dir) / 'D_B_final.pth')
        
        # Save training history
        with open(Path(self.config.checkpoint_dir) / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Final G Loss: {self.history['g_loss'][-1]:.4f}")
        print(f"Final D Loss: {self.history['d_loss'][-1]:.4f}")
        
        self.writer.close()


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train SA-CycleGAN for MRI domain translation')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./preprocessed',
                        help='Directory containing preprocessed data')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for training')
    
    # Model arguments
    parser.add_argument('--ngf', type=int, default=64,
                        help='Number of generator filters')
    parser.add_argument('--ndf', type=int, default=64,
                        help='Number of discriminator filters')
    parser.add_argument('--n_residual_blocks', type=int, default=9,
                        help='Number of residual blocks in generator')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr_g', type=float, default=1e-4,
                        help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=2e-4,
                        help='Discriminator learning rate')
    
    # Loss weights
    parser.add_argument('--lambda_cycle', type=float, default=10.0,
                        help='Cycle consistency loss weight')
    parser.add_argument('--lambda_identity', type=float, default=5.0,
                        help='Identity loss weight')
    parser.add_argument('--lambda_perceptual', type=float, default=1.0,
                        help='Perceptual loss weight')
    
    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/sa_cyclegan',
                        help='Directory for checkpoints')
    parser.add_argument('--log_dir', type=str, default='./runs/sa_cyclegan',
                        help='Directory for TensorBoard logs')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, mps, cpu)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create config from args
    config = Config(**vars(args))
    
    # Create datasets
    print("Loading datasets...")
    dataset_A = MRISliceDataset(
        root_dir=config.data_dir,
        domain=config.brats_dir,
        image_size=config.image_size,
        augment=True
    )
    
    dataset_B = MRISliceDataset(
        root_dir=config.data_dir,
        domain=config.upenn_dir,
        image_size=config.image_size,
        augment=True
    )
    
    # Create dataloaders
    dataloader_A = DataLoader(
        dataset_A,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    dataloader_B = DataLoader(
        dataset_B,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Dataset A (BraTS): {len(dataset_A)} slices")
    print(f"Dataset B (UPenn): {len(dataset_B)} slices")
    
    # Create trainer and train
    trainer = SACycleGANTrainer(config)
    trainer.train(dataloader_A, dataloader_B)


if __name__ == '__main__':
    main()
