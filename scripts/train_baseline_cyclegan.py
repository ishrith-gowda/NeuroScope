#!/usr/bin/env python3
"""
Baseline CycleGAN Training Script.

Trains a standard CycleGAN without self-attention mechanisms for comparison.
This serves as a baseline for the SA-CycleGAN experiments.

Usage:
    python scripts/train_baseline_cyclegan.py \
        --dataset-a ./data/processed/brats \
        --dataset-b ./data/processed/upenn_gbm \
        --output-dir ./experiments/baseline_cyclegan \
        --epochs 200 \
        --batch-size 4 \
        --mixed-precision
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuroscope.models.components.generators import ResnetGenerator
from neuroscope.models.components.discriminators import PatchGANDiscriminator
from neuroscope.models.losses.cycle_losses import CycleLoss, IdentityLoss
from neuroscope.data.medical_dataset import MedicalImageDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaselineCycleGAN(nn.Module):
    """Standard CycleGAN without self-attention."""

    def __init__(
        self,
        input_nc: int = 4,
        output_nc: int = 4,
        ngf: int = 64,
        ndf: int = 64,
        n_residual_blocks: int = 9,
        use_spectral_norm: bool = True,
    ):
        """
        Initialize baseline CycleGAN.

        Args:
            input_nc: Number of input channels
            output_nc: Number of output channels
            ngf: Number of generator filters
            ndf: Number of discriminator filters
            n_residual_blocks: Number of residual blocks
            use_spectral_norm: Use spectral normalization
        """
        super().__init__()

        # Generators
        self.G_AB = ResnetGenerator(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            n_blocks=n_residual_blocks,
            use_spectral_norm=use_spectral_norm,
        )

        self.G_BA = ResnetGenerator(
            input_nc=output_nc,
            output_nc=input_nc,
            ngf=ngf,
            n_blocks=n_residual_blocks,
            use_spectral_norm=use_spectral_norm,
        )

        # Discriminators
        self.D_A = PatchGANDiscriminator(
            input_nc=input_nc,
            ndf=ndf,
            n_layers=3,
            use_spectral_norm=use_spectral_norm,
        )

        self.D_B = PatchGANDiscriminator(
            input_nc=output_nc,
            ndf=ndf,
            n_layers=3,
            use_spectral_norm=use_spectral_norm,
        )

    def forward(self, x_a, x_b):
        """Forward pass."""
        # Generate fake images
        fake_b = self.G_AB(x_a)
        fake_a = self.G_BA(x_b)

        # Cycle consistency
        rec_a = self.G_BA(fake_b)
        rec_b = self.G_AB(fake_a)

        return {
            'fake_b': fake_b,
            'fake_a': fake_a,
            'rec_a': rec_a,
            'rec_b': rec_b,
        }


class BaselineCycleGANTrainer:
    """Trainer for baseline CycleGAN."""

    def __init__(
        self,
        config: Dict,
        output_dir: Path,
        use_wandb: bool = False,
        use_mlflow: bool = False,
        mixed_precision: bool = False,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            output_dir: Output directory for checkpoints and logs
            use_wandb: Enable Weights & Biases logging
            use_mlflow: Enable MLflow logging
            mixed_precision: Use mixed precision training
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = mixed_precision and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # Initialize model
        self.model = BaselineCycleGAN(
            input_nc=config.get('input_channels', 4),
            output_nc=config.get('output_channels', 4),
            ngf=config.get('ngf', 64),
            ndf=config.get('ndf', 64),
            n_residual_blocks=config.get('n_residual_blocks', 9),
            use_spectral_norm=config.get('use_spectral_norm', True),
        ).to(self.device)

        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        # Loss functions
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = CycleLoss(loss_type='l1')
        self.criterion_identity = IdentityLoss(loss_type='l1')

        # Loss weights
        self.lambda_cycle = config.get('lambda_cycle', 10.0)
        self.lambda_identity = config.get('lambda_identity', 0.5)

        # Optimizers
        lr = config.get('lr', 0.0002)
        betas = config.get('betas', (0.5, 0.999))

        self.optimizer_G = optim.Adam(
            list(self.model.G_AB.parameters()) + list(self.model.G_BA.parameters()),
            lr=lr,
            betas=betas
        )

        self.optimizer_D = optim.Adam(
            list(self.model.D_A.parameters()) + list(self.model.D_B.parameters()),
            lr=lr,
            betas=betas
        )

        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_G,
            T_max=config.get('epochs', 200),
            eta_min=1e-6
        )

        self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_D,
            T_max=config.get('epochs', 200),
            eta_min=1e-6
        )

        # Tensorboard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs' / 'tensorboard'))

        # Weights & Biases
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project='neuroscope-baseline-cyclegan',
                    config=config,
                    dir=str(self.output_dir)
                )
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed. Skipping W&B logging.")
                self.use_wandb = False

        # MLflow
        self.use_mlflow = use_mlflow
        if use_mlflow:
            try:
                import mlflow
                mlflow.set_tracking_uri(str(self.output_dir / 'mlruns'))
                mlflow.start_run()
                mlflow.log_params(config)
                self.mlflow = mlflow
            except ImportError:
                logger.warning("mlflow not installed. Skipping MLflow logging.")
                self.use_mlflow = False

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_fid = float('inf')

    def train_epoch(self, dataloader_a: DataLoader, dataloader_b: DataLoader) -> Dict:
        """Train for one epoch."""
        self.model.train()

        metrics = {
            'loss_G': 0.0,
            'loss_D': 0.0,
            'loss_cycle': 0.0,
            'loss_identity': 0.0,
            'loss_gan_g': 0.0,
        }

        # Iterate over both dataloaders
        for batch_a, batch_b in tqdm(
            zip(dataloader_a, dataloader_b),
            total=min(len(dataloader_a), len(dataloader_b)),
            desc=f"Epoch {self.epoch}"
        ):
            # Get images
            real_a = batch_a['image'].to(self.device)
            real_b = batch_b['image'].to(self.device)

            batch_size = real_a.size(0)

            # ===== Train Generators =====
            self.optimizer_G.zero_grad()

            with autocast(enabled=self.use_amp):
                # Forward pass
                outputs = self.model(real_a, real_b)
                fake_a = outputs['fake_a']
                fake_b = outputs['fake_b']
                rec_a = outputs['rec_a']
                rec_b = outputs['rec_b']

                # Identity loss
                idt_a = self.model.G_BA(real_a)
                idt_b = self.model.G_AB(real_b)
                loss_idt_a = self.criterion_identity(idt_a, real_a)
                loss_idt_b = self.criterion_identity(idt_b, real_b)
                loss_identity = (loss_idt_a + loss_idt_b) / 2

                # GAN loss
                pred_fake_a = self.model.D_A(fake_a)
                loss_gan_a = self.criterion_gan(
                    pred_fake_a,
                    torch.ones_like(pred_fake_a)
                )

                pred_fake_b = self.model.D_B(fake_b)
                loss_gan_b = self.criterion_gan(
                    pred_fake_b,
                    torch.ones_like(pred_fake_b)
                )

                loss_gan_g = (loss_gan_a + loss_gan_b) / 2

                # Cycle consistency loss
                loss_cycle_a = self.criterion_cycle(rec_a, real_a)
                loss_cycle_b = self.criterion_cycle(rec_b, real_b)
                loss_cycle = (loss_cycle_a + loss_cycle_b) / 2

                # Total generator loss
                loss_G = (
                    loss_gan_g +
                    self.lambda_cycle * loss_cycle +
                    self.lambda_identity * loss_identity
                )

            # Backward pass for generators
            if self.use_amp:
                self.scaler.scale(loss_G).backward()
                self.scaler.step(self.optimizer_G)
            else:
                loss_G.backward()
                self.optimizer_G.step()

            # ===== Train Discriminators =====
            self.optimizer_D.zero_grad()

            with autocast(enabled=self.use_amp):
                # Discriminator A
                pred_real_a = self.model.D_A(real_a)
                loss_real_a = self.criterion_gan(
                    pred_real_a,
                    torch.ones_like(pred_real_a)
                )

                pred_fake_a = self.model.D_A(fake_a.detach())
                loss_fake_a = self.criterion_gan(
                    pred_fake_a,
                    torch.zeros_like(pred_fake_a)
                )

                loss_D_a = (loss_real_a + loss_fake_a) / 2

                # Discriminator B
                pred_real_b = self.model.D_B(real_b)
                loss_real_b = self.criterion_gan(
                    pred_real_b,
                    torch.ones_like(pred_real_b)
                )

                pred_fake_b = self.model.D_B(fake_b.detach())
                loss_fake_b = self.criterion_gan(
                    pred_fake_b,
                    torch.zeros_like(pred_fake_b)
                )

                loss_D_b = (loss_real_b + loss_fake_b) / 2

                # Total discriminator loss
                loss_D = (loss_D_a + loss_D_b) / 2

            # Backward pass for discriminators
            if self.use_amp:
                self.scaler.scale(loss_D).backward()
                self.scaler.step(self.optimizer_D)
                self.scaler.update()
            else:
                loss_D.backward()
                self.optimizer_D.step()

            # Update metrics
            metrics['loss_G'] += loss_G.item()
            metrics['loss_D'] += loss_D.item()
            metrics['loss_cycle'] += loss_cycle.item()
            metrics['loss_identity'] += loss_identity.item()
            metrics['loss_gan_g'] += loss_gan_g.item()

            self.global_step += 1

        # Average metrics
        n_batches = min(len(dataloader_a), len(dataloader_b))
        for key in metrics:
            metrics[key] /= n_batches

        return metrics

    def save_checkpoint(self, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }

        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoints' / f'epoch_{self.epoch}.pth'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoints' / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def train(
        self,
        dataloader_a: DataLoader,
        dataloader_b: DataLoader,
        num_epochs: int,
        save_freq: int = 10,
    ):
        """
        Main training loop.

        Args:
            dataloader_a: DataLoader for domain A
            dataloader_b: DataLoader for domain B
            num_epochs: Number of epochs to train
            save_freq: Save checkpoint every N epochs
        """
        logger.info("Starting baseline CycleGAN training...")
        logger.info(f"Total epochs: {num_epochs}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train epoch
            start_time = time.time()
            metrics = self.train_epoch(dataloader_a, dataloader_b)
            epoch_time = time.time() - start_time

            # Update learning rate
            self.scheduler_G.step()
            self.scheduler_D.step()

            # Log metrics
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Loss G: {metrics['loss_G']:.4f}, "
                f"Loss D: {metrics['loss_D']:.4f}, "
                f"Loss Cycle: {metrics['loss_cycle']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )

            # TensorBoard logging
            for key, value in metrics.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)

            self.writer.add_scalar('lr/generator', self.scheduler_G.get_last_lr()[0], epoch)
            self.writer.add_scalar('lr/discriminator', self.scheduler_D.get_last_lr()[0], epoch)

            # W&B logging
            if self.use_wandb:
                self.wandb.log({**metrics, 'epoch': epoch})

            # MLflow logging
            if self.use_mlflow:
                for key, value in metrics.items():
                    self.mlflow.log_metric(key, value, step=epoch)

            # Save checkpoint
            if (epoch + 1) % save_freq == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(metrics)

        logger.info("Training complete!")

        # Close logging
        self.writer.close()
        if self.use_wandb:
            self.wandb.finish()
        if self.use_mlflow:
            self.mlflow.end_run()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Train baseline CycleGAN')

    # Data arguments
    parser.add_argument('--dataset-a', type=str, required=True,
                        help='Path to domain A dataset')
    parser.add_argument('--dataset-b', type=str, required=True,
                        help='Path to domain B dataset')

    # Model arguments
    parser.add_argument('--input-channels', type=int, default=4,
                        help='Number of input channels')
    parser.add_argument('--output-channels', type=int, default=4,
                        help='Number of output channels')
    parser.add_argument('--ngf', type=int, default=64,
                        help='Number of generator filters')
    parser.add_argument('--ndf', type=int, default=64,
                        help='Number of discriminator filters')
    parser.add_argument('--n-residual-blocks', type=int, default=9,
                        help='Number of residual blocks')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--lambda-cycle', type=float, default=10.0,
                        help='Cycle loss weight')
    parser.add_argument('--lambda-identity', type=float, default=0.5,
                        help='Identity loss weight')

    # System arguments
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--use-mlflow', action='store_true',
                        help='Use MLflow logging')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    # Create configuration
    config = {
        'input_channels': args.input_channels,
        'output_channels': args.output_channels,
        'ngf': args.ngf,
        'ndf': args.ndf,
        'n_residual_blocks': args.n_residual_blocks,
        'lr': args.lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lambda_cycle': args.lambda_cycle,
        'lambda_identity': args.lambda_identity,
        'use_spectral_norm': True,
    }

    # Create datasets
    logger.info("Loading datasets...")
    dataset_a = MedicalImageDataset(
        root_dir=args.dataset_a,
        split='train',
        transform=None,  # Add transforms as needed
    )

    dataset_b = MedicalImageDataset(
        root_dir=args.dataset_b,
        split='train',
        transform=None,
    )

    # Create dataloaders
    dataloader_a = DataLoader(
        dataset_a,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    dataloader_b = DataLoader(
        dataset_b,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Initialize trainer
    trainer = BaselineCycleGANTrainer(
        config=config,
        output_dir=Path(args.output_dir),
        use_wandb=args.use_wandb,
        use_mlflow=args.use_mlflow,
        mixed_precision=args.mixed_precision,
    )

    # Train
    trainer.train(
        dataloader_a=dataloader_a,
        dataloader_b=dataloader_b,
        num_epochs=args.epochs,
        save_freq=args.save_freq,
    )


if __name__ == '__main__':
    main()
