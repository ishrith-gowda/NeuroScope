"""
CycleGAN v2 Training Script - Enhanced with Anti-Mode-Collapse Techniques

Key improvements over v1:
1. Label smoothing for discriminator stability
2. Noise injection to prevent discriminator from becoming too strong
3. Two-timescale update rule (TTUR) - slower discriminator learning
4. Spectral normalization for discriminator
5. Feature matching loss for better gradient flow
6. Replay buffer to prevent oscillation
7. Gradient penalty for Lipschitz constraint
8. Progressive training with warmup
"""

import os
import sys
import argparse
import logging
import itertools
import json
import random
from datetime import datetime
from collections import deque

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import matplotlib as mpl

from neuroscope_dataset_loader import get_cycle_domain_loaders

# Styling
sns.set_theme(style="whitegrid")
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)


def configure_logging():
    # Force unbuffered output for real-time logging
    import functools
    print = functools.partial(__builtins__['print'] if isinstance(__builtins__, dict) else getattr(__builtins__, 'print'), flush=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Replay Buffer - prevents mode collapse by showing discriminator old fakes
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
                self.data.append(element)
                result.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    result.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    result.append(element)
        return torch.cat(result, 0)


# ============================================================================
# Spectral Normalization - stabilizes discriminator
# ============================================================================
def add_spectral_norm(module):
    """Add spectral normalization to all Conv layers in module"""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, nn.utils.spectral_norm(child))
        else:
            add_spectral_norm(child)
    return module


# ============================================================================
# Network Architectures (with improvements)
# ============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, n_residual=9, ngf=64):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_feat, out_feat = ngf, ngf * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_feat),
                nn.ReLU(inplace=True)
            ]
            in_feat, out_feat = out_feat, out_feat * 2
        
        # Residual blocks
        for _ in range(n_residual):
            model += [ResidualBlock(in_feat)]
        
        # Upsampling
        out_feat = in_feat // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_feat),
                nn.ReLU(inplace=True)
            ]
            in_feat, out_feat = out_feat, out_feat // 2
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, kernel_size=7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    """PatchGAN Discriminator with optional spectral normalization"""
    def __init__(self, in_ch=4, ndf=64, use_spectral_norm=True):
        super().__init__()
        
        # Define blocks separately for feature extraction
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.final = nn.Conv2d(ndf * 8, 1, kernel_size=4, padding=1)
        
        if use_spectral_norm:
            add_spectral_norm(self)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.final(x)
    
    def forward_features(self, x):
        """Get intermediate features for feature matching"""
        features = []
        x = self.block1(x)
        features.append(x)
        x = self.block2(x)
        features.append(x)
        x = self.block3(x)
        features.append(x)
        x = self.block4(x)
        features.append(x)
        return self.final(x), features


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


# ============================================================================
# Loss Functions
# ============================================================================
def compute_gradient_penalty(D, real, fake, device):
    """Gradient penalty for WGAN-GP style regularization"""
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def feature_matching_loss(real_features, fake_features):
    """Feature matching loss - matches intermediate discriminator features"""
    loss = 0
    for rf, ff in zip(real_features, fake_features):
        loss += F.l1_loss(ff.mean(dim=[2, 3]), rf.mean(dim=[2, 3]).detach())
    return loss / len(real_features)


# ============================================================================
# Training Utilities
# ============================================================================
def sample_images(step, G_A2B, G_B2A, loaders, output_dir, tb_writer=None):
    dev = next(G_A2B.parameters()).device
    G_A2B.eval()
    G_B2A.eval()
    
    with torch.no_grad():
        if 'train_A' in loaders and 'train_B' in loaders:
            real_A = next(iter(loaders['train_A'])).to(dev)
            fake_B = G_A2B(real_A)
            recov_A = G_B2A(fake_B)
            
            real_B = next(iter(loaders['train_B'])).to(dev)
            fake_A = G_B2A(real_B)
            recov_B = G_A2B(fake_A)

            # Normalize to [0, 1] for visualization
            imgs = torch.cat([
                (real_A + 1) / 2,
                (fake_B + 1) / 2,
                (recov_A + 1) / 2,
                (real_B + 1) / 2,
                (fake_A + 1) / 2,
                (recov_B + 1) / 2
            ], 0)
            
            grid = make_grid(imgs, nrow=real_A.size(0), normalize=False)
            save_image(grid, os.path.join(output_dir, f"sample_{step}.png"))
            
            if tb_writer:
                tb_writer.add_image('samples', grid, step)
    
    G_A2B.train()
    G_B2A.train()


def plot_loss_graph(loss_history, save_path, tb_writer=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Generator losses
    ax = axes[0, 0]
    ax.plot(loss_history['G'], label='Total G Loss', alpha=0.7)
    ax.set_title('Generator Loss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Discriminator losses
    ax = axes[0, 1]
    ax.plot(loss_history['D_A'], label='D_A', alpha=0.7)
    ax.plot(loss_history['D_B'], label='D_B', alpha=0.7)
    ax.set_title('Discriminator Losses')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cycle & Identity losses
    ax = axes[1, 0]
    ax.plot(loss_history['Cycle'], label='Cycle', alpha=0.7)
    ax.plot(loss_history['Id'], label='Identity', alpha=0.7)
    ax.set_title('Cycle & Identity Losses')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # D/G ratio (health indicator)
    ax = axes[1, 1]
    d_avg = [(a + b) / 2 for a, b in zip(loss_history['D_A'], loss_history['D_B'])]
    g_vals = loss_history['G']
    # Moving average
    window = 100
    if len(d_avg) > window:
        d_smooth = np.convolve(d_avg, np.ones(window)/window, mode='valid')
        g_smooth = np.convolve(g_vals, np.ones(window)/window, mode='valid')
        ratio = [d / max(g, 0.01) for d, g in zip(d_smooth, g_smooth)]
        ax.plot(ratio, label='D/G Ratio', alpha=0.7)
        ax.axhline(y=0.25, color='g', linestyle='--', label='Healthy ratio', alpha=0.5)
    ax.set_title('D/G Loss Ratio (Health Indicator)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_curves_v2.png'), dpi=150)
    plt.close()
    logging.info(f"Saved loss curves to {save_path}/loss_curves_v2.png")


# ============================================================================
# Main Training Function
# ============================================================================
def train(args, device: torch.device):
    configure_logging()
    
    print("\n" + "=" * 80)
    print("CYCLEGAN V2 TRAINING - ENHANCED ANTI-MODE-COLLAPSE")
    print("=" * 80 + "\n")
    
    start_time = datetime.now()
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    # TensorBoard
    run_name = f"v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb_writer = SummaryWriter(log_dir=os.path.join(args.run_dir, run_name))
    
    # Data loaders
    loaders = get_cycle_domain_loaders(
        preprocessed_dir=args.data_root,
        metadata_json=args.meta_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        slices_per_subject=args.slices_per_subject,
        seed=args.seed,
    )
    
    train_loader_A = loaders['train_A']
    train_loader_B = loaders['train_B']
    
    # Models with spectral norm on discriminators
    G_A2B = ResNetGenerator().to(device)
    G_B2A = ResNetGenerator().to(device)
    D_A = PatchDiscriminator(use_spectral_norm=True).to(device)
    D_B = PatchDiscriminator(use_spectral_norm=True).to(device)
    
    # Initialize weights
    G_A2B.apply(weights_init_normal)
    G_B2A.apply(weights_init_normal)
    # Don't apply to D since spectral norm handles it
    
    # Replay buffers
    fake_A_buffer = ReplayBuffer(max_size=50)
    fake_B_buffer = ReplayBuffer(max_size=50)
    
    # Loss functions
    L_GAN = nn.MSELoss()
    L_cycle = nn.L1Loss()
    L_id = nn.L1Loss()
    
    # Optimizers with TTUR (Two-Timescale Update Rule)
    # Generator learns faster than discriminator to prevent D from winning
    lr_G = args.lr
    lr_D = args.lr * 0.5  # Slower discriminator
    
    opt_G = optim.Adam(
        itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
        lr=lr_G, betas=(0.5, 0.999)
    )
    opt_D_A = optim.Adam(D_A.parameters(), lr=lr_D, betas=(0.5, 0.999))
    opt_D_B = optim.Adam(D_B.parameters(), lr=lr_D, betas=(0.5, 0.999))
    
    # Learning rate schedulers with warmup
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return epoch / warmup
        decay_start = args.decay_epoch
        if epoch >= decay_start:
            return max(0.01, 1 - (epoch - decay_start) / (args.n_epochs - decay_start))
        return 1.0
    
    sched_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D_A = optim.lr_scheduler.LambdaLR(opt_D_A, lr_lambda)
    sched_D_B = optim.lr_scheduler.LambdaLR(opt_D_B, lr_lambda)
    
    # Training config
    lambda_cycle = args.lambda_cycle
    lambda_id = args.lambda_identity
    lambda_gp = 10.0  # Gradient penalty weight
    lambda_fm = 1.0   # Feature matching weight
    
    # Label smoothing for discriminator (prevents overconfidence)
    real_label = 0.9  # Instead of 1.0
    fake_label_val = 0.1  # Instead of 0.0
    
    loss_history = {"G": [], "D_A": [], "D_B": [], "Cycle": [], "Id": [], "GP": [], "FM": []}
    
    logging.info("Starting training loop...")
    logging.info(f"Training for {args.n_epochs} epochs")
    logging.info(f"Lambda cycle: {lambda_cycle}, Lambda identity: {lambda_id}")
    logging.info(f"LR Generator: {lr_G}, LR Discriminator: {lr_D}")
    
    global_step = 0
    
    for epoch in range(1, args.n_epochs + 1):
        epoch_start = datetime.now()
        iter_A = iter(train_loader_A)
        iter_B = iter(train_loader_B)
        
        for i in range(min(len(train_loader_A), len(train_loader_B))):
            try:
                real_A = next(iter_A).to(device)
                real_B = next(iter_B).to(device)
            except StopIteration:
                break
            
            batch_size = real_A.size(0)
            
            # Labels with smoothing
            pred_shape = D_A(real_A).shape
            valid = torch.full(pred_shape, real_label, device=device)
            fake = torch.full(pred_shape, fake_label_val, device=device)
            
            # Add instance noise (decays over epochs) - helps early training
            noise_std = max(0, 0.1 * (1 - epoch / args.n_epochs))
            if noise_std > 0:
                real_A_noisy = real_A + noise_std * torch.randn_like(real_A)
                real_B_noisy = real_B + noise_std * torch.randn_like(real_B)
            else:
                real_A_noisy = real_A
                real_B_noisy = real_B
            
            # ==================
            # Train Generators
            # ==================
            opt_G.zero_grad()
            
            # Identity loss
            loss_id_A = L_id(G_B2A(real_A), real_A) * lambda_id
            loss_id_B = L_id(G_A2B(real_B), real_B) * lambda_id
            
            # GAN loss
            fake_B = G_A2B(real_A)
            fake_A = G_B2A(real_B)
            
            loss_GAN_A2B = L_GAN(D_B(fake_B), valid)
            loss_GAN_B2A = L_GAN(D_A(fake_A), valid)
            
            # Cycle consistency loss
            recov_A = G_B2A(fake_B)
            recov_B = G_A2B(fake_A)
            loss_cycle_A = L_cycle(recov_A, real_A) * lambda_cycle
            loss_cycle_B = L_cycle(recov_B, real_B) * lambda_cycle
            
            # Feature matching loss (optional, helps gradient flow)
            _, real_B_feats = D_B.forward_features(real_B_noisy)
            _, fake_B_feats = D_B.forward_features(fake_B)
            loss_fm = feature_matching_loss(real_B_feats, fake_B_feats) * lambda_fm
            
            # Total generator loss
            loss_G = (loss_GAN_A2B + loss_GAN_B2A + 
                     loss_cycle_A + loss_cycle_B + 
                     loss_id_A + loss_id_B + 
                     loss_fm)
            
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(G_A2B.parameters(), G_B2A.parameters()), 
                max_norm=5.0
            )
            opt_G.step()
            
            # ==================
            # Train Discriminator A
            # ==================
            opt_D_A.zero_grad()
            
            # Use replay buffer
            fake_A_replay = fake_A_buffer.push_and_pop(fake_A.detach())
            
            loss_D_real_A = L_GAN(D_A(real_A_noisy), valid)
            loss_D_fake_A = L_GAN(D_A(fake_A_replay), fake)
            
            # Gradient penalty
            gp_A = compute_gradient_penalty(D_A, real_A, fake_A_replay, device)
            
            loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5 + lambda_gp * gp_A
            loss_D_A.backward()
            torch.nn.utils.clip_grad_norm_(D_A.parameters(), max_norm=5.0)
            opt_D_A.step()
            
            # ==================
            # Train Discriminator B
            # ==================
            opt_D_B.zero_grad()
            
            fake_B_replay = fake_B_buffer.push_and_pop(fake_B.detach())
            
            loss_D_real_B = L_GAN(D_B(real_B_noisy), valid)
            loss_D_fake_B = L_GAN(D_B(fake_B_replay), fake)
            
            gp_B = compute_gradient_penalty(D_B, real_B, fake_B_replay, device)
            
            loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5 + lambda_gp * gp_B
            loss_D_B.backward()
            torch.nn.utils.clip_grad_norm_(D_B.parameters(), max_norm=5.0)
            opt_D_B.step()
            
            # Record losses
            loss_history["G"].append(loss_G.item())
            loss_history["D_A"].append(loss_D_A.item())
            loss_history["D_B"].append(loss_D_B.item())
            loss_history["Cycle"].append((loss_cycle_A + loss_cycle_B).item())
            loss_history["Id"].append((loss_id_A + loss_id_B).item())
            loss_history["GP"].append((gp_A + gp_B).item())
            loss_history["FM"].append(loss_fm.item())
            
            global_step += 1
            
            # TensorBoard logging
            if global_step % 10 == 0:
                tb_writer.add_scalar('Loss/G', loss_G.item(), global_step)
                tb_writer.add_scalar('Loss/D_A', loss_D_A.item(), global_step)
                tb_writer.add_scalar('Loss/D_B', loss_D_B.item(), global_step)
                tb_writer.add_scalar('Loss/Cycle', (loss_cycle_A + loss_cycle_B).item(), global_step)
                tb_writer.add_scalar('Loss/Identity', (loss_id_A + loss_id_B).item(), global_step)
                tb_writer.add_scalar('Loss/GradientPenalty', (gp_A + gp_B).item(), global_step)
                tb_writer.add_scalar('Loss/FeatureMatching', loss_fm.item(), global_step)
                tb_writer.add_scalar('LR/Generator', opt_G.param_groups[0]['lr'], global_step)
                tb_writer.add_scalar('Noise/InstanceNoise', noise_std, global_step)
            
            # Console logging
            if global_step % args.log_interval == 0:
                logging.info(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i+1}] "
                    f"G: {loss_G.item():.4f} D_A: {loss_D_A.item():.4f} D_B: {loss_D_B.item():.4f} "
                    f"Cyc: {(loss_cycle_A + loss_cycle_B).item():.4f}"
                )
            
            # Sample images
            if global_step % args.sample_interval == 0:
                sample_images(global_step, G_A2B, G_B2A, loaders, args.sample_dir, tb_writer)
        
        # End of epoch
        sched_G.step()
        sched_D_A.step()
        sched_D_B.step()
        
        epoch_time = (datetime.now() - epoch_start).total_seconds() / 60
        logging.info(f"Epoch {epoch} completed in {epoch_time:.2f} min")
        
        # Save checkpoints
        if epoch % args.checkpoint_interval == 0:
            torch.save(G_A2B.state_dict(), os.path.join(args.checkpoint_dir, f"G_A2B_v2_{epoch}.pth"))
            torch.save(G_B2A.state_dict(), os.path.join(args.checkpoint_dir, f"G_B2A_v2_{epoch}.pth"))
            torch.save(D_A.state_dict(), os.path.join(args.checkpoint_dir, f"D_A_v2_{epoch}.pth"))
            torch.save(D_B.state_dict(), os.path.join(args.checkpoint_dir, f"D_B_v2_{epoch}.pth"))
            
            # Full checkpoint
            ckpt = {
                'epoch': epoch,
                'G_A2B_state': G_A2B.state_dict(),
                'G_B2A_state': G_B2A.state_dict(),
                'D_A_state': D_A.state_dict(),
                'D_B_state': D_B.state_dict(),
                'opt_G_state': opt_G.state_dict(),
                'opt_D_A_state': opt_D_A.state_dict(),
                'opt_D_B_state': opt_D_B.state_dict(),
                'loss_history': loss_history,
            }
            torch.save(ckpt, os.path.join(args.checkpoint_dir, f"full_v2_epoch_{epoch}.pt"))
            logging.info(f"Saved checkpoint at epoch {epoch}")
    
    # Final saves
    torch.save(G_A2B.state_dict(), os.path.join(args.checkpoint_dir, "final_models", "G_A2B_v2_final.pth"))
    torch.save(G_B2A.state_dict(), os.path.join(args.checkpoint_dir, "final_models", "G_B2A_v2_final.pth"))
    torch.save(D_A.state_dict(), os.path.join(args.checkpoint_dir, "final_models", "D_A_v2_final.pth"))
    torch.save(D_B.state_dict(), os.path.join(args.checkpoint_dir, "final_models", "D_B_v2_final.pth"))
    
    # Save loss history
    with open(os.path.join(args.sample_dir, 'training_loss_log_v2.json'), 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    plot_loss_graph(loss_history, args.sample_dir, tb_writer)
    
    total_time = (datetime.now() - start_time).total_seconds() / 60
    logging.info(f"Training complete! Total time: {total_time:.2f} minutes")
    
    tb_writer.close()


def parse_args():
    ap = argparse.ArgumentParser(description='CycleGAN v2 Training with Anti-Mode-Collapse')
    ap.add_argument('--data_root', type=str, default='/Volumes/usb drive/neuroscope/preprocessed')
    ap.add_argument('--meta_json', type=str, default='/Volumes/usb drive/neuroscope/scripts/01_data_preparation_pipeline/neuroscope_dataset_metadata_splits.json')
    ap.add_argument('--n_epochs', type=int, default=150)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)  # Lower LR for stability
    ap.add_argument('--decay_epoch', type=int, default=75)
    ap.add_argument('--lambda_cycle', type=float, default=10.0)
    ap.add_argument('--lambda_identity', type=float, default=5.0)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--log_interval', type=int, default=50)
    ap.add_argument('--sample_interval', type=int, default=200)
    ap.add_argument('--checkpoint_interval', type=int, default=10)
    ap.add_argument('--checkpoint_dir', type=str, default='/Volumes/usb drive/neuroscope/checkpoints')
    ap.add_argument('--sample_dir', type=str, default='/Volumes/usb drive/neuroscope/samples')
    ap.add_argument('--run_dir', type=str, default='/Volumes/usb drive/neuroscope/runs')
    ap.add_argument('--slices_per_subject', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    train(args, device)
