"""
Comprehensive 2.5D SA-CycleGAN Trainer.

Professional-grade training pipeline with:
- Full logging (TensorBoard, CSV, JSON, Console)
- Sample and figure generation
- Mixed precision training (AMP)
- Gradient monitoring and clipping
- Learning rate scheduling with warmup
- Early stopping
- Checkpoint management
- Reproducibility features

Author: NeuroScope Research Team
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple, List, Union
from pathlib import Path
from datetime import datetime
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# NeuroScope imports
from neuroscope.models.architectures.sa_cyclegan_25d import (
    SACycleGAN25D, SACycleGAN25DConfig, create_model
)
from neuroscope.data.datasets.dataset_25d import create_dataloaders
from neuroscope.models.losses.combined_losses import CombinedLoss
from neuroscope.training.loggers import LoggerManager
from neuroscope.training.samplers import SampleGenerator
from neuroscope.training.figures import FigureGenerator
from neuroscope.training.callbacks.training_callbacks import EarlyStopping, CallbackState


@dataclass
class TrainingConfig:
    """Comprehensive training configuration."""
    
    # Experiment
    experiment_name: str = "sa_cyclegan_25d"
    seed: int = 42
    deterministic: bool = True
    
    # Data paths
    brats_dir: str = "/Volumes/usb drive/neuroscope/preprocessed/brats"
    upenn_dir: str = "/Volumes/usb drive/neuroscope/preprocessed/upenn"
    output_dir: str = "/Volumes/usb drive/neuroscope/experiments"
    
    # Model architecture
    ngf: int = 64
    ndf: int = 64
    n_residual_blocks: int = 9
    use_attention: bool = True
    use_cbam: bool = True
    input_channels: int = 12  # 3 slices * 4 modalities
    output_channels: int = 4  # 4 modalities
    
    # Training
    epochs: int = 100
    batch_size: int = 4
    image_size: int = 128
    num_workers: int = 4
    
    # Optimizer
    lr_G: float = 2e-4
    lr_D: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.0
    
    # Scheduler
    scheduler_type: str = "cosine"  # 'cosine', 'linear', 'step'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Loss weights
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    lambda_ssim: float = 1.0
    lambda_gradient: float = 1.0
    
    # Regularization
    gradient_clip_norm: float = 1.0
    gradient_clip_value: Optional[float] = None
    
    # Mixed precision
    use_amp: bool = False  # Disabled for MPS
    
    # Validation & Checkpointing
    validate_every: int = 5
    save_every: int = 10
    save_best_only: bool = False
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 0.001
    
    # Logging
    log_every_n_steps: int = 10
    sample_every: int = 10
    figure_every: int = 10
    verbose: int = 2  # 0=silent, 1=minimal, 2=normal, 3=detailed
    
    # Resume
    resume_from: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        
    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """Load from YAML file."""
        import yaml
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)
        
    def save_yaml(self, path: str):
        """Save to YAML file."""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


class ReplayBuffer:
    """Image replay buffer for discriminator training stability."""
    
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


class ComprehensiveTrainer:
    """
    Professional-grade trainer for 2.5D SA-CycleGAN.
    
    Features:
    - Complete training/validation loops
    - Comprehensive logging (TensorBoard, CSV, JSON, Console)
    - Sample and figure generation
    - Mixed precision training
    - Gradient monitoring and clipping
    - Learning rate scheduling with warmup
    - Early stopping
    - Checkpoint management
    - Full reproducibility
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_reproducibility()
        self.setup_directories()
        self.setup_device()
        self.setup_model()
        self.setup_data()
        self.setup_optimizers()
        self.setup_losses()
        self.setup_logging()
        self.setup_callbacks()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.history = self._init_history()
        
        # Replay buffers
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()
        
    def setup_reproducibility(self):
        """Set random seeds for reproducibility."""
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    def setup_directories(self):
        """Create output directories."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(self.config.output_dir) / f"{self.config.experiment_name}_{timestamp}"
        
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.samples_dir = self.run_dir / "samples"
        self.figures_dir = self.run_dir / "figures"
        self.logs_dir = self.run_dir / "logs"
        
        for d in [self.checkpoints_dir, self.samples_dir, 
                  self.figures_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        # Save config
        self.config.save_yaml(str(self.run_dir / "config.yaml"))
        
    def setup_device(self):
        """Setup compute device."""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.device_name = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.device_name = "Apple Silicon (MPS)"
        else:
            self.device = torch.device('cpu')
            self.device_name = "CPU"
            
        # AMP only for CUDA
        self.use_amp = self.config.use_amp and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
    def setup_model(self):
        """Create and initialize model."""
        model_config = SACycleGAN25DConfig(
            ngf=self.config.ngf,
            ndf=self.config.ndf,
            n_residual_blocks=self.config.n_residual_blocks,
            lambda_cycle=self.config.lambda_cycle,
            lambda_identity=self.config.lambda_identity,
            lambda_ssim=self.config.lambda_ssim
        )
        
        self.model = create_model(model_config)
        self.model = self.model.to(self.device)
        
        # Count parameters
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
    def setup_data(self):
        """Create data loaders."""
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            brats_dir=self.config.brats_dir,
            upenn_dir=self.config.upenn_dir,
            batch_size=self.config.batch_size,
            image_size=(self.config.image_size, self.config.image_size),
            num_workers=self.config.num_workers
        )
        
        self.train_samples = len(self.train_loader.dataset)
        self.val_samples = len(self.val_loader.dataset)
        self.test_samples = len(self.test_loader.dataset)
        
    def setup_optimizers(self):
        """Create optimizers and schedulers."""
        # Generator optimizer
        self.opt_G = optim.Adam(
            list(self.model.G_A2B.parameters()) + list(self.model.G_B2A.parameters()),
            lr=self.config.lr_G,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )
        
        # Discriminator optimizer
        self.opt_D = optim.Adam(
            list(self.model.D_A.parameters()) + list(self.model.D_B.parameters()),
            lr=self.config.lr_D,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )
        
        # Schedulers
        if self.config.scheduler_type == 'cosine':
            self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(
                self.opt_G, T_max=self.config.epochs, eta_min=self.config.min_lr
            )
            self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(
                self.opt_D, T_max=self.config.epochs, eta_min=self.config.min_lr
            )
        elif self.config.scheduler_type == 'linear':
            def linear_lambda(epoch):
                return 1 - max(0, epoch - self.config.epochs // 2) / (self.config.epochs // 2 + 1)
            self.scheduler_G = optim.lr_scheduler.LambdaLR(self.opt_G, linear_lambda)
            self.scheduler_D = optim.lr_scheduler.LambdaLR(self.opt_D, linear_lambda)
        else:
            self.scheduler_G = optim.lr_scheduler.StepLR(self.opt_G, step_size=30, gamma=0.5)
            self.scheduler_D = optim.lr_scheduler.StepLR(self.opt_D, step_size=30, gamma=0.5)
            
    def setup_losses(self):
        """Initialize loss functions."""
        self.losses = CombinedLoss(
            lambda_cycle=self.config.lambda_cycle,
            lambda_identity=self.config.lambda_identity,
            lambda_ssim=self.config.lambda_ssim,
            lambda_gradient=self.config.lambda_gradient
        ).to(self.device)
        
    def setup_logging(self):
        """Initialize all loggers."""
        self.logger = LoggerManager(
            log_dir=self.logs_dir,
            experiment_name=self.config.experiment_name,
            use_tensorboard=True,
            use_csv=True,
            use_json=True,
            use_console=True,
            console_verbose=self.config.verbose
        )
        
        self.sample_generator = SampleGenerator(
            output_dir=self.samples_dir,
            modality_names=['T1', 'T1Gd', 'T2', 'FLAIR']
        )
        
        self.figure_generator = FigureGenerator(
            output_dir=self.figures_dir,
            style='publication'
        )
        
    def setup_callbacks(self):
        """Initialize callbacks."""
        self.early_stopping = None
        if self.config.early_stopping:
            self.early_stopping = EarlyStopping(
                monitor='val_ssim',
                patience=self.config.patience,
                min_delta=self.config.min_delta,
                mode='max',
                verbose=self.config.verbose > 0
            )
            
    def _init_history(self) -> Dict[str, List]:
        """Initialize training history."""
        return {
            'G_loss': [], 'D_loss': [],
            'cycle_loss': [], 'identity_loss': [],
            'gan_loss': [], 'ssim_loss': [],
            'val_ssim_A2B': [], 'val_ssim_B2A': [],
            'val_psnr_A2B': [], 'val_psnr_B2A': [],
            'learning_rate': [],
            'gradient_norm_G': [], 'gradient_norm_D': []
        }
        
    # =========================================================================
    # Metrics
    # =========================================================================
    
    def compute_ssim(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute SSIM between two tensors."""
        x = x.detach().cpu().float()
        y = y.detach().cpu().float()
        
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
        if mse < 1e-10:
            return 100.0
        return 10 * np.log10(1.0 / mse)
        
    def compute_gradient_norm(self, model: nn.Module) -> float:
        """Compute gradient L2 norm."""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
        
    # =========================================================================
    # Training Loop
    # =========================================================================
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {
            'G_loss': 0, 'D_loss': 0,
            'cycle_A': 0, 'cycle_B': 0,
            'identity_A': 0, 'identity_B': 0,
            'gan_A2B': 0, 'gan_B2A': 0,
            'ssim': 0
        }
        
        n_batches = len(self.train_loader)
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}/{self.config.epochs}",
            ncols=120,
            leave=True,
            disable=self.config.verbose < 1
        )
        
        batch_start_time = time.time()
        
        for batch_idx, batch in enumerate(pbar):
            real_A = batch['A'].to(self.device)
            real_B = batch['B'].to(self.device)
            center_A = batch['A_center'].to(self.device)
            center_B = batch['B_center'].to(self.device)
            
            # ================================================================
            # Train Generators
            # ================================================================
            self.opt_G.zero_grad()
            
            # Generate
            fake_B = self.model.G_A2B(real_A)
            fake_A = self.model.G_B2A(real_B)
            
            # Cycle consistency (expand to 3-slice format)
            fake_B_3slice = fake_B.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            fake_B_3slice = fake_B_3slice.view(fake_B.size(0), -1, fake_B.size(2), fake_B.size(3))
            fake_A_3slice = fake_A.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            fake_A_3slice = fake_A_3slice.view(fake_A.size(0), -1, fake_A.size(2), fake_A.size(3))
            
            rec_A = self.model.G_B2A(fake_B_3slice)
            rec_B = self.model.G_A2B(fake_A_3slice)
            
            # Losses
            loss_cycle_A = self.losses.cycle_loss(center_A, rec_A)
            loss_cycle_B = self.losses.cycle_loss(center_B, rec_B)
            
            identity_A = self.model.G_B2A(real_A)
            identity_B = self.model.G_A2B(real_B)
            loss_identity_A = self.losses.identity_loss(center_A, identity_A)
            loss_identity_B = self.losses.identity_loss(center_B, identity_B)
            
            pred_fake_B = self.model.D_B(fake_B)
            pred_fake_A = self.model.D_A(fake_A)
            loss_gan_A2B = self.losses.gan_loss.generator_loss(pred_fake_B)
            loss_gan_B2A = self.losses.gan_loss.generator_loss(pred_fake_A)
            
            loss_ssim = self.losses.ssim_loss(center_A, rec_A) + self.losses.ssim_loss(center_B, rec_B)
            
            loss_G = (loss_gan_A2B + loss_gan_B2A + 
                      loss_cycle_A + loss_cycle_B + 
                      loss_identity_A + loss_identity_B +
                      loss_ssim)
            
            loss_G.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.G_A2B.parameters()) + list(self.model.G_B2A.parameters()),
                    self.config.gradient_clip_norm
                )
                
            grad_norm_G = self.compute_gradient_norm(self.model.G_A2B)
            self.opt_G.step()
            
            # ================================================================
            # Train Discriminators
            # ================================================================
            self.opt_D.zero_grad()
            
            fake_A_buffer = self.fake_A_buffer.push_and_pop(fake_A.detach())
            fake_B_buffer = self.fake_B_buffer.push_and_pop(fake_B.detach())
            
            pred_real_A = self.model.D_A(center_A)
            pred_fake_A = self.model.D_A(fake_A_buffer)
            loss_D_A = self.losses.gan_loss.discriminator_loss(pred_real_A, pred_fake_A)
            
            pred_real_B = self.model.D_B(center_B)
            pred_fake_B = self.model.D_B(fake_B_buffer)
            loss_D_B = self.losses.gan_loss.discriminator_loss(pred_real_B, pred_fake_B)
            
            loss_D = (loss_D_A + loss_D_B) * 0.5
            loss_D.backward()
            
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.D_A.parameters()) + list(self.model.D_B.parameters()),
                    self.config.gradient_clip_norm
                )
                
            grad_norm_D = self.compute_gradient_norm(self.model.D_A)
            self.opt_D.step()
            
            # Accumulate
            epoch_losses['G_loss'] += loss_G.item()
            epoch_losses['D_loss'] += loss_D.item()
            epoch_losses['cycle_A'] += loss_cycle_A.item()
            epoch_losses['cycle_B'] += loss_cycle_B.item()
            epoch_losses['identity_A'] += loss_identity_A.item()
            epoch_losses['identity_B'] += loss_identity_B.item()
            epoch_losses['gan_A2B'] += loss_gan_A2B.item()
            epoch_losses['gan_B2A'] += loss_gan_B2A.item()
            epoch_losses['ssim'] += loss_ssim.item()
            
            # Log batch
            self.global_step += 1
            if batch_idx % self.config.log_every_n_steps == 0:
                batch_time = time.time() - batch_start_time
                samples_per_sec = self.config.batch_size / batch_time
                
                self.logger.log_batch(
                    batch_idx + 1, n_batches,
                    {
                        'G': loss_G.item(),
                        'D': loss_D.item(),
                        'cyc': (loss_cycle_A + loss_cycle_B).item()
                    },
                    samples_per_sec=samples_per_sec
                )
                
                # TensorBoard batch logging
                self.logger.log_scalar('Batch/G_loss', loss_G.item(), self.global_step)
                self.logger.log_scalar('Batch/D_loss', loss_D.item(), self.global_step)
                self.logger.log_scalar('Batch/grad_norm_G', grad_norm_G, self.global_step)
                self.logger.log_scalar('Batch/grad_norm_D', grad_norm_D, self.global_step)
                
            batch_start_time = time.time()
            
            pbar.set_postfix({
                'G': f'{loss_G.item():.3f}',
                'D': f'{loss_D.item():.3f}',
                'cyc': f'{(loss_cycle_A + loss_cycle_B).item():.3f}'
            })
            
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
            
        return epoch_losses
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        ssim_A2B, ssim_B2A = [], []
        psnr_A2B, psnr_B2A = [], []
        
        for batch in tqdm(self.val_loader, desc="Validation", 
                         ncols=100, leave=False, disable=self.config.verbose < 1):
            real_A = batch['A'].to(self.device)
            real_B = batch['B'].to(self.device)
            center_A = batch['A_center'].to(self.device)
            center_B = batch['B_center'].to(self.device)
            
            fake_B = self.model.G_A2B(real_A)
            fake_A = self.model.G_B2A(real_B)
            
            fake_B_3slice = fake_B.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            fake_B_3slice = fake_B_3slice.view(fake_B.size(0), -1, fake_B.size(2), fake_B.size(3))
            fake_A_3slice = fake_A.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            fake_A_3slice = fake_A_3slice.view(fake_A.size(0), -1, fake_A.size(2), fake_A.size(3))
            
            rec_A = self.model.G_B2A(fake_B_3slice)
            rec_B = self.model.G_A2B(fake_A_3slice)
            
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
        
    # =========================================================================
    # Checkpointing
    # =========================================================================
    
    def save_checkpoint(self, is_best: bool = False, filename: str = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'opt_G_state_dict': self.opt_G.state_dict(),
            'opt_D_state_dict': self.opt_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'best_metric': self.best_metric,
            'history': self.history,
            'config': self.config.to_dict()
        }
        
        # Save latest
        latest_path = self.checkpoints_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save periodic
        if self.current_epoch % self.config.save_every == 0:
            epoch_path = self.checkpoints_dir / f'checkpoint_epoch_{self.current_epoch:04d}.pth'
            torch.save(checkpoint, epoch_path)
            
        # Save best
        if is_best:
            best_path = self.checkpoints_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            
        self.logger.log_checkpoint(
            self.current_epoch,
            str(latest_path),
            is_best,
            {'best_ssim': self.best_metric} if is_best else None
        )
        
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
        self.opt_D.load_state_dict(checkpoint['opt_D_state_dict'])
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        self.history = checkpoint['history']
        
        self.logger.log_info(f"Resumed from epoch {checkpoint['epoch']}")
        
    # =========================================================================
    # Sample Generation
    # =========================================================================
    
    @torch.no_grad()
    def generate_samples(self):
        """Generate and save sample images."""
        self.model.eval()
        
        # Get one batch
        batch = next(iter(self.val_loader))
        real_A = batch['A'].to(self.device)
        real_B = batch['B'].to(self.device)
        center_A = batch['A_center'].to(self.device)
        center_B = batch['B_center'].to(self.device)
        
        fake_B = self.model.G_A2B(real_A)
        fake_A = self.model.G_B2A(real_B)
        
        fake_B_3slice = fake_B.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        fake_B_3slice = fake_B_3slice.view(fake_B.size(0), -1, fake_B.size(2), fake_B.size(3))
        fake_A_3slice = fake_A.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        fake_A_3slice = fake_A_3slice.view(fake_A.size(0), -1, fake_A.size(2), fake_A.size(3))
        
        rec_A = self.model.G_B2A(fake_B_3slice)
        rec_B = self.model.G_A2B(fake_A_3slice)
        
        # Save comparison for each modality
        sample_paths = self.sample_generator.generate_all_samples(
            center_A, fake_B, rec_A,
            center_B, fake_A, rec_B,
            self.current_epoch
        )
        
        # Log to TensorBoard
        self.logger.log_sample_comparison(
            center_A, fake_B, rec_A,
            center_B, fake_A, rec_B,
            self.current_epoch
        )
        
        self.logger.log_sample_saved(str(self.samples_dir), self.current_epoch)
        
        return sample_paths
        
    # =========================================================================
    # Main Training
    # =========================================================================
    
    def train(self):
        """Main training loop."""
        # Log configuration
        self.logger.log_config({
            'experiment': self.config.experiment_name,
            'model': {
                'ngf': self.config.ngf,
                'ndf': self.config.ndf,
                'n_residual': self.config.n_residual_blocks
            },
            'training': {
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'lr_G': self.config.lr_G,
                'lr_D': self.config.lr_D
            },
            'device': self.device_name
        })
        
        self.logger.log_model_info(self.model, "SACycleGAN25D")
        self.logger.log_data_info(
            self.train_samples, self.val_samples, self.test_samples,
            self.config.batch_size
        )
        
        self.logger.on_training_start(self.config.epochs)
        
        # Resume if specified
        if self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)
            
        training_start = time.time()
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()
            
            self.logger.on_epoch_start(self.current_epoch)
            
            # Train
            train_losses = self.train_epoch()
            
            # Update schedulers
            old_lr = self.scheduler_G.get_last_lr()[0]
            self.scheduler_G.step()
            self.scheduler_D.step()
            new_lr = self.scheduler_G.get_last_lr()[0]
            
            if abs(new_lr - old_lr) > 1e-10:
                self.logger.log_lr_update(old_lr, new_lr, "scheduler step")
                
            # Update history
            self.history['G_loss'].append(train_losses['G_loss'])
            self.history['D_loss'].append(train_losses['D_loss'])
            self.history['cycle_loss'].append(train_losses['cycle_A'] + train_losses['cycle_B'])
            self.history['identity_loss'].append(train_losses['identity_A'] + train_losses['identity_B'])
            self.history['learning_rate'].append(new_lr)
            
            # Validate
            val_metrics = None
            is_best = False
            
            if self.current_epoch % self.config.validate_every == 0:
                val_metrics = self.validate()
                
                self.history['val_ssim_A2B'].append(val_metrics['ssim_A2B'])
                self.history['val_ssim_B2A'].append(val_metrics['ssim_B2A'])
                self.history['val_psnr_A2B'].append(val_metrics['psnr_A2B'])
                self.history['val_psnr_B2A'].append(val_metrics['psnr_B2A'])
                
                avg_ssim = (val_metrics['ssim_A2B'] + val_metrics['ssim_B2A']) / 2
                is_best = avg_ssim > self.best_metric
                if is_best:
                    self.best_metric = avg_ssim
                    
                # Early stopping check
                if self.early_stopping:
                    state = CallbackState(
                        epoch=self.current_epoch,
                        metrics={'val_ssim': avg_ssim},
                        model=self.model
                    )
                    self.early_stopping.on_epoch_end(state)
                    
                    if self.early_stopping.stopped:
                        self.logger.log_early_stop(
                            self.current_epoch,
                            self.config.patience,
                            self.best_metric
                        )
                        break
                        
            epoch_time = time.time() - epoch_start
            
            # Log epoch
            train_metrics = {
                'G_loss': train_losses['G_loss'],
                'D_loss': train_losses['D_loss'],
                'cycle': train_losses['cycle_A'] + train_losses['cycle_B']
            }
            
            self.logger.on_epoch_end(
                self.current_epoch,
                train_metrics,
                val_metrics,
                new_lr,
                epoch_time
            )
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # Generate samples
            if self.current_epoch % self.config.sample_every == 0:
                self.generate_samples()
                
            # Generate figures
            if self.current_epoch % self.config.figure_every == 0:
                val_history = {
                    'ssim_A2B': self.history['val_ssim_A2B'],
                    'ssim_B2A': self.history['val_ssim_B2A'],
                    'psnr_A2B': self.history['val_psnr_A2B'],
                    'psnr_B2A': self.history['val_psnr_B2A']
                }
                self.figure_generator.plot_training_losses(self.history)
                self.figure_generator.plot_validation_metrics(val_history)
                
        # Training complete
        total_time = time.time() - training_start
        
        final_metrics = {
            'best_ssim': self.best_metric,
            'final_G_loss': self.history['G_loss'][-1],
            'final_D_loss': self.history['D_loss'][-1]
        }
        
        if self.history['val_ssim_A2B']:
            final_metrics['final_ssim_A2B'] = self.history['val_ssim_A2B'][-1]
            final_metrics['final_ssim_B2A'] = self.history['val_ssim_B2A'][-1]
            
        self.logger.on_training_end(final_metrics)
        
        # Generate final figures
        val_history = {
            'ssim_A2B': self.history['val_ssim_A2B'],
            'ssim_B2A': self.history['val_ssim_B2A'],
            'psnr_A2B': self.history['val_psnr_A2B'],
            'psnr_B2A': self.history['val_psnr_B2A']
        }
        
        self.figure_generator.create_publication_summary(
            self.history, val_history, final_metrics,
            self.config.experiment_name
        )
        
        # Save final history
        with open(self.run_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
            
        self.logger.close()
        
        return final_metrics
        
    def close(self):
        """Cleanup."""
        self.logger.close()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train 2.5D SA-CycleGAN')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--verbose', type=int, default=2)
    
    args = parser.parse_args()
    
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr_G=args.lr,
            lr_D=args.lr,
            resume_from=args.resume,
            verbose=args.verbose
        )
        
    trainer = ComprehensiveTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
