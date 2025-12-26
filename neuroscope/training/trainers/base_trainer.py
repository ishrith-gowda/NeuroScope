"""
Training Loop Implementations.

Provides robust training infrastructure for GAN-based
image translation models with proper logging and checkpointing.
"""

from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast


@dataclass
class TrainerConfig:
    """Configuration for trainer."""
    # Training parameters
    num_epochs: int = 200
    start_epoch: int = 0
    
    # Learning rates
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    
    # Betas for Adam
    beta1: float = 0.5
    beta2: float = 0.999
    
    # Loss weights
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.5
    lambda_perceptual: float = 1.0
    lambda_adversarial: float = 1.0
    
    # Training schedule
    decay_epoch: int = 100
    warmup_epochs: int = 0
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_frequency: int = 10
    keep_last_n: int = 5
    
    # Logging
    log_frequency: int = 100
    sample_frequency: int = 500
    
    # Mixed precision
    use_amp: bool = True
    grad_clip: Optional[float] = 1.0
    
    # Device
    device: str = 'cuda'
    
    # Buffer pool size for discriminator
    pool_size: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__annotations__}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainerConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class TrainingState:
    """Mutable training state."""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float('inf')
    best_epoch: int = 0
    
    # Running statistics
    losses: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Training history
    history: Dict[str, List[float]] = field(default_factory=lambda: {
        'loss_g': [], 'loss_d': [], 'loss_cycle': [], 'loss_identity': []
    })
    
    def update_history(self, losses: Dict[str, float]):
        """Update training history."""
        for key, value in losses.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'history': self.history,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingState':
        state = cls()
        state.epoch = d.get('epoch', 0)
        state.global_step = d.get('global_step', 0)
        state.best_metric = d.get('best_metric', float('inf'))
        state.best_epoch = d.get('best_epoch', 0)
        state.history = d.get('history', {})
        return state


class ImagePool:
    """
    Buffer pool for discriminator training.
    
    Stores previously generated images to provide a more
    stable training signal for the discriminator.
    """
    
    def __init__(self, pool_size: int = 50):
        self.pool_size = pool_size
        self.images = []
    
    def query(self, images: torch.Tensor) -> torch.Tensor:
        """
        Return images from pool with 50% probability.
        
        Args:
            images: Current batch of generated images
            
        Returns:
            Mix of current and pooled images
        """
        if self.pool_size == 0:
            return images
        
        return_images = []
        for image in images:
            image = image.unsqueeze(0)
            
            if len(self.images) < self.pool_size:
                self.images.append(image.clone())
                return_images.append(image)
            else:
                if torch.rand(1).item() > 0.5:
                    # Return from pool
                    idx = torch.randint(0, len(self.images), (1,)).item()
                    tmp = self.images[idx].clone()
                    self.images[idx] = image.clone()
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        
        return torch.cat(return_images, dim=0)


class BaseTrainer(ABC):
    """
    Abstract base trainer class.
    
    Provides common training infrastructure including:
        - Training loop management
        - Checkpointing
        - Logging
        - Mixed precision training
    """
    
    def __init__(
        self,
        config: TrainerConfig,
        callbacks: Optional[List] = None
    ):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.callbacks = callbacks or []
        
        # Initialize state
        self.state = TrainingState()
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_amp and self.device.type == 'cuda' else None
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def build_models(self) -> Dict[str, nn.Module]:
        """Build and return model dictionary."""
        pass
    
    @abstractmethod
    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Build and return optimizer dictionary."""
        pass
    
    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute single training step."""
        pass
    
    @abstractmethod
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute single validation step."""
        pass
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        self._call_callbacks('on_train_begin')
        
        for epoch in range(self.config.start_epoch, self.config.num_epochs):
            self.state.epoch = epoch
            self._call_callbacks('on_epoch_begin', epoch=epoch)
            
            # Training phase
            epoch_losses = self._train_epoch(train_loader)
            self.state.update_history(epoch_losses)
            
            # Validation phase
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader)
                self.state.metrics.update(val_metrics)
            
            # Learning rate scheduling
            self._step_schedulers(epoch)
            
            # Checkpointing
            if (epoch + 1) % self.config.save_frequency == 0:
                self._save_checkpoint(epoch)
            
            self._call_callbacks('on_epoch_end', epoch=epoch, losses=epoch_losses)
        
        self._call_callbacks('on_train_end')
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        for model in self.models.values():
            model.train()
        
        epoch_losses = {}
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            self._call_callbacks('on_batch_begin', batch_idx=batch_idx)
            
            # Move to device
            batch = self._to_device(batch)
            
            # Training step
            losses = self.train_step(batch)
            
            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] = epoch_losses.get(key, 0) + value
            
            self.state.global_step += 1
            self._call_callbacks('on_batch_end', batch_idx=batch_idx, losses=losses)
            
            # Logging
            if (batch_idx + 1) % self.config.log_frequency == 0:
                self._log_progress(batch_idx, num_batches, losses)
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        for model in self.models.values():
            model.eval()
        
        val_metrics = {}
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)
                metrics = self.validate_step(batch)
                
                for key, value in metrics.items():
                    val_metrics[key] = val_metrics.get(key, 0) + value
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_metrics
    
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            else:
                result[key] = value
        return result
    
    def _step_schedulers(self, epoch: int):
        """Step learning rate schedulers."""
        if hasattr(self, 'schedulers'):
            for scheduler in self.schedulers.values():
                scheduler.step()
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'state': self.state.to_dict(),
            'config': self.config.to_dict(),
        }
        
        # Save model states
        for name, model in self.models.items():
            checkpoint[f'model_{name}'] = model.state_dict()
        
        # Save optimizer states
        for name, optimizer in self.optimizers.items():
            checkpoint[f'optimizer_{name}'] = optimizer.state_dict()
        
        # Save scheduler states
        if hasattr(self, 'schedulers'):
            for name, scheduler in self.schedulers.items():
                checkpoint[f'scheduler_{name}'] = scheduler.state_dict()
        
        # Save checkpoint
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        torch.save(checkpoint, path)
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints keeping only last N."""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        
        if len(checkpoints) > self.config.keep_last_n:
            for ckpt in checkpoints[:-self.config.keep_last_n]:
                ckpt.unlink()
    
    def load_checkpoint(self, path: Union[str, Path]):
        """Load checkpoint from file."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore state
        self.state = TrainingState.from_dict(checkpoint['state'])
        
        # Restore models
        for name, model in self.models.items():
            if f'model_{name}' in checkpoint:
                model.load_state_dict(checkpoint[f'model_{name}'])
        
        # Restore optimizers
        for name, optimizer in self.optimizers.items():
            if f'optimizer_{name}' in checkpoint:
                optimizer.load_state_dict(checkpoint[f'optimizer_{name}'])
        
        # Restore schedulers
        if hasattr(self, 'schedulers'):
            for name, scheduler in self.schedulers.items():
                if f'scheduler_{name}' in checkpoint:
                    scheduler.load_state_dict(checkpoint[f'scheduler_{name}'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def _log_progress(self, batch_idx: int, num_batches: int, losses: Dict[str, float]):
        """Log training progress."""
        loss_str = ' | '.join([f'{k}: {v:.4f}' for k, v in losses.items()])
        print(f"Epoch [{self.state.epoch + 1}/{self.config.num_epochs}] "
              f"Batch [{batch_idx + 1}/{num_batches}] | {loss_str}")
    
    def _call_callbacks(self, hook: str, **kwargs):
        """Call callback hooks."""
        for callback in self.callbacks:
            if hasattr(callback, hook):
                getattr(callback, hook)(self, **kwargs)


class GANTrainer(BaseTrainer):
    """
    Generic GAN trainer.
    
    Handles generator and discriminator training with proper
    scheduling and gradient management.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        config: TrainerConfig,
        gan_loss: Optional[nn.Module] = None,
        callbacks: Optional[List] = None
    ):
        super().__init__(config, callbacks)
        
        self.models = {
            'G': generator.to(self.device),
            'D': discriminator.to(self.device)
        }
        
        # Loss functions
        self.criterion_gan = gan_loss or nn.BCEWithLogitsLoss()
        
        # Build optimizers
        self.optimizers = self.build_optimizers()
    
    def build_models(self) -> Dict[str, nn.Module]:
        return self.models
    
    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {
            'G': torch.optim.Adam(
                self.models['G'].parameters(),
                lr=self.config.lr_g,
                betas=(self.config.beta1, self.config.beta2)
            ),
            'D': torch.optim.Adam(
                self.models['D'].parameters(),
                lr=self.config.lr_d,
                betas=(self.config.beta1, self.config.beta2)
            )
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        real = batch['real']
        batch_size = real.size(0)
        
        # Labels
        real_label = torch.ones(batch_size, 1, device=self.device)
        fake_label = torch.zeros(batch_size, 1, device=self.device)
        
        # Generate noise
        noise = torch.randn(batch_size, self.models['G'].latent_dim, device=self.device)
        
        losses = {}
        
        # Train Discriminator
        self.optimizers['D'].zero_grad()
        
        with autocast(enabled=self.scaler is not None):
            # Real images
            pred_real = self.models['D'](real)
            loss_d_real = self.criterion_gan(pred_real, real_label)
            
            # Fake images
            fake = self.models['G'](noise)
            pred_fake = self.models['D'](fake.detach())
            loss_d_fake = self.criterion_gan(pred_fake, fake_label)
            
            loss_d = (loss_d_real + loss_d_fake) * 0.5
        
        if self.scaler:
            self.scaler.scale(loss_d).backward()
            self.scaler.step(self.optimizers['D'])
        else:
            loss_d.backward()
            self.optimizers['D'].step()
        
        losses['loss_d'] = loss_d.item()
        
        # Train Generator
        self.optimizers['G'].zero_grad()
        
        with autocast(enabled=self.scaler is not None):
            pred_fake = self.models['D'](fake)
            loss_g = self.criterion_gan(pred_fake, real_label)
        
        if self.scaler:
            self.scaler.scale(loss_g).backward()
            self.scaler.step(self.optimizers['G'])
            self.scaler.update()
        else:
            loss_g.backward()
            self.optimizers['G'].step()
        
        losses['loss_g'] = loss_g.item()
        
        return losses
    
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return {}


class CycleGANTrainer(BaseTrainer):
    """
    CycleGAN trainer for unpaired image-to-image translation.
    
    Implements the full CycleGAN training procedure with:
        - Cycle consistency loss
        - Identity loss
        - Adversarial loss
        - Optional perceptual loss
    """
    
    def __init__(
        self,
        G_A2B: nn.Module,
        G_B2A: nn.Module,
        D_A: nn.Module,
        D_B: nn.Module,
        config: TrainerConfig,
        criterion_gan: Optional[nn.Module] = None,
        criterion_cycle: Optional[nn.Module] = None,
        criterion_identity: Optional[nn.Module] = None,
        criterion_perceptual: Optional[nn.Module] = None,
        callbacks: Optional[List] = None
    ):
        super().__init__(config, callbacks)
        
        # Models
        self.models = {
            'G_A2B': G_A2B.to(self.device),
            'G_B2A': G_B2A.to(self.device),
            'D_A': D_A.to(self.device),
            'D_B': D_B.to(self.device)
        }
        
        # Loss functions
        self.criterion_gan = criterion_gan or nn.MSELoss()
        self.criterion_cycle = criterion_cycle or nn.L1Loss()
        self.criterion_identity = criterion_identity or nn.L1Loss()
        self.criterion_perceptual = criterion_perceptual
        
        # Image pools
        self.fake_A_pool = ImagePool(config.pool_size)
        self.fake_B_pool = ImagePool(config.pool_size)
        
        # Build optimizers
        self.optimizers = self.build_optimizers()
    
    def build_models(self) -> Dict[str, nn.Module]:
        return self.models
    
    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {
            'G': torch.optim.Adam(
                list(self.models['G_A2B'].parameters()) + 
                list(self.models['G_B2A'].parameters()),
                lr=self.config.lr_g,
                betas=(self.config.beta1, self.config.beta2)
            ),
            'D': torch.optim.Adam(
                list(self.models['D_A'].parameters()) + 
                list(self.models['D_B'].parameters()),
                lr=self.config.lr_d,
                betas=(self.config.beta1, self.config.beta2)
            )
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        real_A = batch['real_A']
        real_B = batch['real_B']
        
        losses = {}
        
        # =====================
        # Train Generators
        # =====================
        self.optimizers['G'].zero_grad()
        
        with autocast(enabled=self.scaler is not None):
            # Identity loss
            if self.config.lambda_identity > 0:
                idt_A = self.models['G_B2A'](real_A)
                loss_idt_A = self.criterion_identity(idt_A, real_A)
                
                idt_B = self.models['G_A2B'](real_B)
                loss_idt_B = self.criterion_identity(idt_B, real_B)
                
                loss_identity = (loss_idt_A + loss_idt_B) * self.config.lambda_identity
            else:
                loss_identity = 0
            
            # GAN loss
            fake_B = self.models['G_A2B'](real_A)
            pred_fake_B = self.models['D_B'](fake_B)
            loss_gan_A2B = self.criterion_gan(
                pred_fake_B, 
                torch.ones_like(pred_fake_B)
            )
            
            fake_A = self.models['G_B2A'](real_B)
            pred_fake_A = self.models['D_A'](fake_A)
            loss_gan_B2A = self.criterion_gan(
                pred_fake_A,
                torch.ones_like(pred_fake_A)
            )
            
            # Cycle consistency loss
            rec_A = self.models['G_B2A'](fake_B)
            loss_cycle_A = self.criterion_cycle(rec_A, real_A)
            
            rec_B = self.models['G_A2B'](fake_A)
            loss_cycle_B = self.criterion_cycle(rec_B, real_B)
            
            loss_cycle = (loss_cycle_A + loss_cycle_B) * self.config.lambda_cycle
            
            # Perceptual loss
            if self.criterion_perceptual is not None:
                loss_perceptual = (
                    self.criterion_perceptual(fake_B, real_A) +
                    self.criterion_perceptual(fake_A, real_B)
                ) * self.config.lambda_perceptual
            else:
                loss_perceptual = 0
            
            # Total generator loss
            loss_G = (loss_gan_A2B + loss_gan_B2A + loss_cycle + 
                     loss_identity + loss_perceptual)
        
        if self.scaler:
            self.scaler.scale(loss_G).backward()
            if self.config.grad_clip:
                self.scaler.unscale_(self.optimizers['G'])
                torch.nn.utils.clip_grad_norm_(
                    list(self.models['G_A2B'].parameters()) + 
                    list(self.models['G_B2A'].parameters()),
                    self.config.grad_clip
                )
            self.scaler.step(self.optimizers['G'])
        else:
            loss_G.backward()
            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    list(self.models['G_A2B'].parameters()) + 
                    list(self.models['G_B2A'].parameters()),
                    self.config.grad_clip
                )
            self.optimizers['G'].step()
        
        losses['loss_G'] = loss_G.item()
        losses['loss_cycle'] = loss_cycle.item() if isinstance(loss_cycle, torch.Tensor) else loss_cycle
        losses['loss_identity'] = loss_identity.item() if isinstance(loss_identity, torch.Tensor) else loss_identity
        
        # =====================
        # Train Discriminators
        # =====================
        self.optimizers['D'].zero_grad()
        
        with autocast(enabled=self.scaler is not None):
            # Discriminator A
            fake_A_pooled = self.fake_A_pool.query(fake_A.detach())
            
            pred_real_A = self.models['D_A'](real_A)
            loss_D_A_real = self.criterion_gan(
                pred_real_A,
                torch.ones_like(pred_real_A)
            )
            
            pred_fake_A = self.models['D_A'](fake_A_pooled)
            loss_D_A_fake = self.criterion_gan(
                pred_fake_A,
                torch.zeros_like(pred_fake_A)
            )
            
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
            
            # Discriminator B
            fake_B_pooled = self.fake_B_pool.query(fake_B.detach())
            
            pred_real_B = self.models['D_B'](real_B)
            loss_D_B_real = self.criterion_gan(
                pred_real_B,
                torch.ones_like(pred_real_B)
            )
            
            pred_fake_B = self.models['D_B'](fake_B_pooled)
            loss_D_B_fake = self.criterion_gan(
                pred_fake_B,
                torch.zeros_like(pred_fake_B)
            )
            
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
            
            loss_D = loss_D_A + loss_D_B
        
        if self.scaler:
            self.scaler.scale(loss_D).backward()
            self.scaler.step(self.optimizers['D'])
            self.scaler.update()
        else:
            loss_D.backward()
            self.optimizers['D'].step()
        
        losses['loss_D'] = loss_D.item()
        losses['loss_D_A'] = loss_D_A.item()
        losses['loss_D_B'] = loss_D_B.item()
        
        return losses
    
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute validation metrics."""
        real_A = batch['real_A']
        real_B = batch['real_B']
        
        # Generate translations
        fake_B = self.models['G_A2B'](real_A)
        fake_A = self.models['G_B2A'](real_B)
        
        # Cycle reconstruction
        rec_A = self.models['G_B2A'](fake_B)
        rec_B = self.models['G_A2B'](fake_A)
        
        # Compute metrics
        cycle_loss_A = self.criterion_cycle(rec_A, real_A).item()
        cycle_loss_B = self.criterion_cycle(rec_B, real_B).item()
        
        return {
            'val_cycle_A': cycle_loss_A,
            'val_cycle_B': cycle_loss_B,
            'val_cycle': (cycle_loss_A + cycle_loss_B) / 2
        }
    
    def generate_samples(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Generate sample translations for visualization."""
        with torch.no_grad():
            fake_B = self.models['G_A2B'](real_A)
            rec_A = self.models['G_B2A'](fake_B)
            
            fake_A = self.models['G_B2A'](real_B)
            rec_B = self.models['G_A2B'](fake_A)
        
        return {
            'real_A': real_A,
            'fake_B': fake_B,
            'rec_A': rec_A,
            'real_B': real_B,
            'fake_A': fake_A,
            'rec_B': rec_B
        }


# Aliases for compatibility
TrainingConfig = TrainerConfig
Trainer = CycleGANTrainer
