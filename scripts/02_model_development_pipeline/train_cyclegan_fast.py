#!/usr/bin/env python
"""
CycleGAN Fast Training Script

Optimizations for speed:
1. Pre-load and cache slices in memory
2. Simplified but effective architecture
3. Reduced iterations per epoch
4. Real-time output with proper flushing
"""

import os
import sys
import argparse
import logging
import itertools
import json
import random
from datetime import datetime
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

def pprint(*args, **kwargs):
    """Print with flush"""
    print(*args, **kwargs, flush=True)

# ============================================================================
# Memory-Cached Dataset - Pre-loads slices for speed
# ============================================================================
class CachedSliceDataset(Dataset):
    """Pre-caches 2D slices in memory for fast training"""
    
    def __init__(self, data_root, section, split, meta_json, 
                 max_subjects=None, slices_per_subject=8, 
                 slice_range=(50, 110)):
        """
        Args:
            data_root: Path to preprocessed data
            section: 'brats' or 'upenn'
            split: 'train', 'val', or 'test'
            meta_json: Path to metadata JSON
            max_subjects: Limit subjects for faster loading
            slices_per_subject: Number of slices to cache per subject
            slice_range: (min_z, max_z) to sample from (brain region)
        """
        self.slices = []
        self.section = section
        self.split = split
        
        # Modality names match preprocessed directory structure
        modalities = ['t1.nii.gz', 't1gd.nii.gz', 't2.nii.gz', 'flair.nii.gz']
        
        # Load metadata
        with open(meta_json, 'r') as f:
            meta = json.load(f)
        
        # Get valid_subjects for this section
        subjects_meta = meta.get(section, {}).get('valid_subjects', {})
        subject_ids = [sid for sid, info in subjects_meta.items() 
                       if info.get('split') == split]
        
        if max_subjects:
            subject_ids = subject_ids[:max_subjects]
        
        pprint(f"Loading {section}/{split}: {len(subject_ids)} subjects...")
        
        for idx, sid in enumerate(subject_ids):
            subject_dir = Path(data_root) / section / sid
            
            # Check all modalities exist
            mod_paths = [subject_dir / mod for mod in modalities]
            if not all(p.exists() for p in mod_paths):
                continue
            
            try:
                # Load all modalities
                vols = []
                for mp in mod_paths:
                    img = sitk.ReadImage(str(mp))
                    arr = sitk.GetArrayFromImage(img).astype(np.float32)
                    vols.append(arr)
                vol = np.stack(vols, axis=0)  # [4, D, H, W]
                
                # Sample slices from brain region
                depth = vol.shape[1]
                z_min = max(0, min(slice_range[0], depth - 1))
                z_max = min(depth - 1, slice_range[1])
                
                if z_max <= z_min:
                    z_min, z_max = 0, depth - 1
                
                # Sample random slices
                z_indices = np.random.choice(
                    range(z_min, z_max + 1), 
                    size=min(slices_per_subject, z_max - z_min + 1),
                    replace=False
                )
                
                for z in z_indices:
                    slice_4ch = vol[:, z, :, :]  # [4, H, W]
                    
                    # Normalize to [-1, 1]
                    slice_4ch = np.clip(slice_4ch, 0, 1)
                    slice_4ch = slice_4ch * 2 - 1
                    
                    self.slices.append(torch.from_numpy(slice_4ch).float())
                
                if (idx + 1) % 20 == 0:
                    pprint(f"  Loaded {idx + 1}/{len(subject_ids)} subjects, {len(self.slices)} slices")
                    
            except Exception as e:
                pprint(f"  Warning: Failed to load {sid}: {e}")
                continue
        
        pprint(f"  Cached {len(self.slices)} slices for {section}/{split}")
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        return self.slices[idx]


# ============================================================================
# Lightweight Generator - Faster but still effective
# ============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )
    
    def forward(self, x):
        return x + self.block(x)


class FastGenerator(nn.Module):
    """Generator with same architecture as original (9 residual blocks)"""
    
    def __init__(self, in_channels=4, out_channels=4, n_residual=9, ngf=64):
        super().__init__()
        
        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]
        
        # Downsampling (2 layers)
        in_ch = ngf
        for _ in range(2):
            out_ch = in_ch * 2
            model += [
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        
        # Residual blocks
        for _ in range(n_residual):
            model.append(ResidualBlock(in_ch))
        
        # Upsampling (2 layers)
        for _ in range(2):
            out_ch = in_ch // 2
            model += [
                nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7),
            nn.Tanh(),
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


class FastDiscriminator(nn.Module):
    """PatchGAN discriminator with spectral normalization"""
    
    def __init__(self, in_channels=4, ndf=64):
        super().__init__()
        
        def disc_block(in_ch, out_ch, normalize=True):
            layers = [nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1))]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *disc_block(in_channels, ndf, normalize=False),
            *disc_block(ndf, ndf * 2),
            *disc_block(ndf * 2, ndf * 4),
            *disc_block(ndf * 4, ndf * 8),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, 4, padding=1)),
        )
    
    def forward(self, x):
        return self.model(x)


# ============================================================================
# Replay Buffer
# ============================================================================
class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []
    
    def push_and_pop(self, data):
        result = []
        for element in data:
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element.detach().cpu())
                result.append(element)
            else:
                if random.random() > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    result.append(self.data[i].clone().to(element.device))
                    self.data[i] = element.detach().cpu()
                else:
                    result.append(element)
        return torch.cat(result, 0)


# ============================================================================
# Training Function
# ============================================================================
def train(args):
    pprint("\n" + "=" * 70)
    pprint("CYCLEGAN FAST TRAINING")
    pprint("=" * 70)
    
    # Device
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    pprint(f"Device: {device}")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    # Load datasets (cached in memory)
    pprint("\nLoading and caching datasets...")
    
    dataset_A = CachedSliceDataset(
        args.data_root, 'brats', 'train', args.meta_json,
        max_subjects=args.max_subjects,
        slices_per_subject=args.slices_per_subject
    )
    dataset_B = CachedSliceDataset(
        args.data_root, 'upenn', 'train', args.meta_json,
        max_subjects=args.max_subjects,
        slices_per_subject=args.slices_per_subject
    )
    
    loader_A = DataLoader(dataset_A, batch_size=args.batch_size, shuffle=True, 
                          num_workers=0, drop_last=True)
    loader_B = DataLoader(dataset_B, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, drop_last=True)
    
    pprint(f"\nDataset A (BraTS): {len(dataset_A)} slices")
    pprint(f"Dataset B (UPenn): {len(dataset_B)} slices")
    
    # Initialize models
    pprint("\nInitializing models...")
    G_A2B = FastGenerator(n_residual=6).to(device)
    G_B2A = FastGenerator(n_residual=6).to(device)
    D_A = FastDiscriminator().to(device)
    D_B = FastDiscriminator().to(device)
    
    # Count parameters
    n_params_G = sum(p.numel() for p in G_A2B.parameters())
    n_params_D = sum(p.numel() for p in D_A.parameters())
    pprint(f"Generator params: {n_params_G:,}")
    pprint(f"Discriminator params: {n_params_D:,}")
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    G_A2B.apply(init_weights)
    G_B2A.apply(init_weights)
    
    # Replay buffers
    fake_A_buffer = ReplayBuffer(50)
    fake_B_buffer = ReplayBuffer(50)
    
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    
    # Optimizers (TTUR: generator learns faster)
    opt_G = optim.Adam(
        itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
        lr=args.lr, betas=(0.5, 0.999)
    )
    opt_D = optim.Adam(
        itertools.chain(D_A.parameters(), D_B.parameters()),
        lr=args.lr * 0.5, betas=(0.5, 0.999)
    )
    
    # LR schedulers
    def lr_lambda(epoch):
        return 1.0 - max(0, epoch - args.decay_epoch) / (args.n_epochs - args.decay_epoch + 1)
    
    scheduler_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    scheduler_D = optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)
    
    # Training parameters
    lambda_cycle = args.lambda_cycle
    lambda_id = args.lambda_identity
    
    # Loss tracking
    loss_history = {
        'G_total': [], 'G_GAN': [], 'G_cycle': [], 'G_identity': [],
        'D_A': [], 'D_B': []
    }
    
    # Fixed samples for visualization
    fixed_A = next(iter(loader_A))[:4].to(device)
    fixed_B = next(iter(loader_B))[:4].to(device)
    
    pprint("\n" + "=" * 70)
    pprint("STARTING TRAINING")
    pprint("=" * 70)
    pprint(f"Epochs: {args.n_epochs}")
    pprint(f"Batch size: {args.batch_size}")
    pprint(f"Learning rate: {args.lr}")
    pprint(f"Lambda cycle: {lambda_cycle}, Lambda identity: {lambda_id}")
    pprint("=" * 70 + "\n")
    
    start_time = datetime.now()
    global_step = 0
    
    for epoch in range(1, args.n_epochs + 1):
        epoch_start = datetime.now()
        
        epoch_losses = {k: 0.0 for k in loss_history.keys()}
        n_batches = 0
        
        iter_A = iter(loader_A)
        iter_B = iter(loader_B)
        n_iters = min(len(loader_A), len(loader_B))
        
        for i in range(n_iters):
            try:
                real_A = next(iter_A).to(device)
                real_B = next(iter_B).to(device)
            except StopIteration:
                break
            
            batch_size = real_A.size(0)
            
            # Valid and fake labels (with smoothing)
            valid = torch.full((batch_size, 1, 15, 15), 0.9, device=device)
            fake_label = torch.full((batch_size, 1, 15, 15), 0.1, device=device)
            
            # ================
            # Train Generators
            # ================
            opt_G.zero_grad()
            
            # Identity loss
            same_A = G_B2A(real_A)
            loss_id_A = criterion_identity(same_A, real_A) * lambda_id
            
            same_B = G_A2B(real_B)
            loss_id_B = criterion_identity(same_B, real_B) * lambda_id
            
            # GAN loss
            fake_B = G_A2B(real_A)
            pred_fake_B = D_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake_B, valid)
            
            fake_A = G_B2A(real_B)
            pred_fake_A = D_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake_A, valid)
            
            # Cycle loss
            recovered_A = G_B2A(fake_B)
            loss_cycle_A = criterion_cycle(recovered_A, real_A) * lambda_cycle
            
            recovered_B = G_A2B(fake_A)
            loss_cycle_B = criterion_cycle(recovered_B, real_B) * lambda_cycle
            
            # Total generator loss
            loss_G = (loss_GAN_A2B + loss_GAN_B2A + 
                      loss_cycle_A + loss_cycle_B + 
                      loss_id_A + loss_id_B)
            
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(G_A2B.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(G_B2A.parameters(), 1.0)
            opt_G.step()
            
            # ====================
            # Train Discriminators
            # ====================
            opt_D.zero_grad()
            
            # D_A
            pred_real_A = D_A(real_A)
            loss_D_real_A = criterion_GAN(pred_real_A, valid)
            
            fake_A_buffered = fake_A_buffer.push_and_pop(fake_A.detach())
            pred_fake_A = D_A(fake_A_buffered)
            loss_D_fake_A = criterion_GAN(pred_fake_A, fake_label)
            
            loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
            
            # D_B
            pred_real_B = D_B(real_B)
            loss_D_real_B = criterion_GAN(pred_real_B, valid)
            
            fake_B_buffered = fake_B_buffer.push_and_pop(fake_B.detach())
            pred_fake_B = D_B(fake_B_buffered)
            loss_D_fake_B = criterion_GAN(pred_fake_B, fake_label)
            
            loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
            
            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            opt_D.step()
            
            # Track losses
            epoch_losses['G_total'] += loss_G.item()
            epoch_losses['G_GAN'] += (loss_GAN_A2B + loss_GAN_B2A).item()
            epoch_losses['G_cycle'] += (loss_cycle_A + loss_cycle_B).item()
            epoch_losses['G_identity'] += (loss_id_A + loss_id_B).item()
            epoch_losses['D_A'] += loss_D_A.item()
            epoch_losses['D_B'] += loss_D_B.item()
            n_batches += 1
            global_step += 1
            
            # Log progress
            if (i + 1) % args.log_interval == 0 or i == 0:
                pprint(f"  [{epoch}/{args.n_epochs}] [{i+1}/{n_iters}] "
                       f"G: {loss_G.item():.4f} D_A: {loss_D_A.item():.4f} D_B: {loss_D_B.item():.4f} "
                       f"Cycle: {(loss_cycle_A + loss_cycle_B).item():.4f}")
        
        # Epoch summary
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
            loss_history[k].append(epoch_losses[k])
        
        pprint(f"\n[Epoch {epoch}/{args.n_epochs}] Time: {epoch_time:.1f}s | "
               f"G: {epoch_losses['G_total']:.4f} | D_A: {epoch_losses['D_A']:.4f} | D_B: {epoch_losses['D_B']:.4f}")
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        # Save sample images
        if epoch % args.sample_interval == 0 or epoch == 1:
            G_A2B.eval()
            G_B2A.eval()
            with torch.no_grad():
                fake_B_sample = G_A2B(fixed_A)
                fake_A_sample = G_B2A(fixed_B)
                recovered_A = G_B2A(fake_B_sample)
                recovered_B = G_A2B(fake_A_sample)
            
            # Save grid
            img_sample = torch.cat([
                fixed_A[:, 0:1], fake_B_sample[:, 0:1], recovered_A[:, 0:1],
                fixed_B[:, 0:1], fake_A_sample[:, 0:1], recovered_B[:, 0:1]
            ], dim=0)
            
            save_image(img_sample, 
                       os.path.join(args.sample_dir, f'fast_epoch_{epoch:03d}.png'),
                       nrow=4, normalize=True, value_range=(-1, 1))
            pprint(f"  Saved sample: fast_epoch_{epoch:03d}.png")
            
            G_A2B.train()
            G_B2A.train()
        
        # Save checkpoints
        if epoch % args.checkpoint_interval == 0:
            torch.save(G_A2B.state_dict(), 
                       os.path.join(args.checkpoint_dir, f'fast_G_A2B_{epoch}.pth'))
            torch.save(G_B2A.state_dict(), 
                       os.path.join(args.checkpoint_dir, f'fast_G_B2A_{epoch}.pth'))
            torch.save(D_A.state_dict(), 
                       os.path.join(args.checkpoint_dir, f'fast_D_A_{epoch}.pth'))
            torch.save(D_B.state_dict(), 
                       os.path.join(args.checkpoint_dir, f'fast_D_B_{epoch}.pth'))
            pprint(f"  Saved checkpoint at epoch {epoch}")
        
        pprint("")
    
    # Training complete
    total_time = (datetime.now() - start_time).total_seconds()
    pprint("\n" + "=" * 70)
    pprint("TRAINING COMPLETE")
    pprint("=" * 70)
    pprint(f"Total time: {total_time/60:.1f} minutes")
    pprint(f"Final losses - G: {loss_history['G_total'][-1]:.4f}, "
           f"D_A: {loss_history['D_A'][-1]:.4f}, D_B: {loss_history['D_B'][-1]:.4f}")
    
    # Save final models
    torch.save(G_A2B.state_dict(), os.path.join(args.checkpoint_dir, 'fast_G_A2B_final.pth'))
    torch.save(G_B2A.state_dict(), os.path.join(args.checkpoint_dir, 'fast_G_B2A_final.pth'))
    torch.save(D_A.state_dict(), os.path.join(args.checkpoint_dir, 'fast_D_A_final.pth'))
    torch.save(D_B.state_dict(), os.path.join(args.checkpoint_dir, 'fast_D_B_final.pth'))
    
    # Save loss history
    with open(os.path.join(args.sample_dir, 'fast_training_loss.json'), 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    pprint("\n✓ Final models and loss history saved!")
    
    return loss_history


def main():
    parser = argparse.ArgumentParser(description='CycleGAN Fast Training')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--meta_json', type=str, required=True)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--decay_epoch', type=int, default=25)
    parser.add_argument('--lambda_cycle', type=float, default=10.0)
    parser.add_argument('--lambda_identity', type=float, default=5.0)
    parser.add_argument('--max_subjects', type=int, default=50,
                        help='Max subjects to load per domain (for speed)')
    parser.add_argument('--slices_per_subject', type=int, default=8)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--sample_interval', type=int, default=5)
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
