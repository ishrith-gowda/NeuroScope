#!/usr/bin/env python3
"""
training script for harmonize-and-compress sa-cyclegan-2.5d.

jointly trains the cyclegan for harmonization while learning a compressed
bottleneck representation. the model produces both a harmonized image and
a compressed bitstream in a single forward pass. a rate-distortion loss
balances reconstruction quality against bitrate.

extension b of the journal extension.

usage:
    python train_compression.py --config ../configs/compression.yaml
    python train_compression.py --lambda_rate 0.01 --entropy_model_type factorized
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
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

# add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from neuroscope.models.architectures.sa_cyclegan_25d import SACycleGAN25DConfig
from neuroscope.models.compression.compressed_generator import (
    CompressedSACycleGAN25D,
)
from neuroscope.data.datasets.dataset_25d import UnpairedMRIDataset25D, create_dataloaders
from neuroscope.models.losses.combined_losses import CombinedLoss


class ReplayBuffer:
    """image buffer for discriminator training stability."""

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


def setup_torch_performance():
    """configure torch for maximum gpu throughput."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(enabled=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)


class CompressionTrainer:
    """
    trainer for harmonize-and-compress sa-cyclegan-2.5d.

    extends the base trainer with:
    - compression-aware bottleneck with quantization + entropy model
    - rate-distortion loss: l_total = l_harmonization + lambda_rate * r
    - warmup phase where compression is disabled for the first n epochs
    - rate-distortion curve logging (ssim vs bits-per-voxel)
    """

    def __init__(
        self,
        config: SACycleGAN25DConfig,
        brats_dir: str,
        upenn_dir: str,
        output_dir: str,
        batch_size: int = 16,
        image_size: int = 128,
        lr_G: float = 5e-5,
        lr_D: float = 5e-5,
        beta1: float = 0.5,
        beta2: float = 0.999,
        num_workers: int = 4,
        device: str = "auto",
        experiment_name: str = None,
        # compression hyperparameters
        entropy_model_type: str = "factorized",
        num_hyper_channels: int = 128,
        lambda_rate: float = 0.01,
        compression_warmup_epochs: int = 20,
        # training hyperparameters
        scheduler_type: str = "cosine",
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        gradient_clip_norm: float = 1.0,
        use_amp: bool = True,
    ):
        setup_torch_performance()
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # experiment naming
        if experiment_name is None:
            experiment_name = f"compression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # create subdirectories
        (self.experiment_dir / "checkpoints").mkdir(exist_ok=True)
        (self.experiment_dir / "samples").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)

        # device setup
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.use_multi_gpu = self.num_gpus > 1

        print("=" * 60)
        print("  harmonize-and-compress sa-cyclegan-2.5d training")
        print("=" * 60)
        print(f"device: {self.device}")
        if self.use_multi_gpu:
            print(f"multi-gpu: {self.num_gpus} gpus (dataparallel)")
        print(f"experiment: {experiment_name}")
        print(f"entropy model: {entropy_model_type}")
        print(f"lambda_rate: {lambda_rate}")
        print(f"compression warmup: {compression_warmup_epochs} epochs")

        # tensorboard
        runs_dir = Path("/data/runs") if Path("/data/runs").exists() else PROJECT_ROOT / "runs"
        self.writer = SummaryWriter(
            log_dir=str(runs_dir / experiment_name)
        )

        # create compressed model
        self.model = CompressedSACycleGAN25D(
            config=config,
            entropy_model_type=entropy_model_type,
            num_hyper_channels=num_hyper_channels,
        )
        self.model = self.model.to(self.device)

        self.lambda_rate = lambda_rate
        self.compression_warmup_epochs = compression_warmup_epochs

        # wrap with dataparallel for multi-gpu
        if self.use_multi_gpu:
            self.model.G_A2B = DataParallel(self.model.G_A2B)
            self.model.G_B2A = DataParallel(self.model.G_B2A)
            self.model.D_A = DataParallel(self.model.D_A)
            self.model.D_B = DataParallel(self.model.D_B)

        # create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            brats_dir=brats_dir,
            upenn_dir=upenn_dir,
            batch_size=batch_size,
            image_size=(image_size, image_size),
            num_workers=num_workers,
        )

        print(f"training batches: {len(self.train_loader)}")
        print(f"validation batches: {len(self.val_loader)}")

        # optimizers
        self.opt_G = optim.Adam(
            list(self.model.G_A2B.parameters()) + list(self.model.G_B2A.parameters()),
            lr=lr_G,
            betas=(beta1, beta2),
        )
        self.opt_D = optim.Adam(
            list(self.model.D_A.parameters()) + list(self.model.D_B.parameters()),
            lr=lr_D,
            betas=(beta1, beta2),
        )

        # learning rate schedulers
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.base_lr_G = lr_G
        self.base_lr_D = lr_D

        if scheduler_type == "cosine":
            self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(
                self.opt_G, T_max=200, eta_min=min_lr
            )
            self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(
                self.opt_D, T_max=200, eta_min=min_lr
            )
        else:
            def lambda_rule(epoch, total_epochs=200):
                decay_start = total_epochs // 2
                if epoch < decay_start:
                    return 1.0
                return 1.0 - (epoch - decay_start) / (total_epochs - decay_start + 1)

            self.scheduler_G = optim.lr_scheduler.LambdaLR(
                self.opt_G, lr_lambda=lambda_rule
            )
            self.scheduler_D = optim.lr_scheduler.LambdaLR(
                self.opt_D, lr_lambda=lambda_rule
            )

        # loss functions
        self.losses = CombinedLoss(
            lambda_cycle=config.lambda_cycle,
            lambda_identity=config.lambda_identity,
            lambda_ssim=config.lambda_ssim,
            lambda_gradient=1.0,
        ).to(self.device)

        # replay buffers
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # automatic mixed precision
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler_G = GradScaler("cuda", enabled=self.use_amp)
        self.scaler_D = GradScaler("cuda", enabled=self.use_amp)
        self.gradient_clip_norm = gradient_clip_norm

        # training history
        self.history = {
            "train": {
                "G_loss": [],
                "D_loss": [],
                "cycle_loss": [],
                "rate_loss": [],
                "bpe": [],
            },
            "val": {
                "ssim_A2B": [],
                "ssim_B2A": [],
                "psnr_A2B": [],
                "psnr_B2A": [],
                "bpe": [],
            },
            "learning_rate": [],
            "epoch_times": [],
        }

        self.start_epoch = 0
        self.best_val_ssim = 0
        self.global_step = 0

        # save config
        self._save_config(locals())

    def _save_config(self, init_args: dict):
        """save experiment configuration."""
        config_dict = {
            "model": {k: v for k, v in self.config.__dict__.items()},
            "training": {
                "lambda_rate": self.lambda_rate,
                "compression_warmup_epochs": self.compression_warmup_epochs,
                "scheduler_type": self.scheduler_type,
                "warmup_epochs": self.warmup_epochs,
                "device": str(self.device),
                "experiment_name": self.experiment_name,
            },
        }
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    def _get_generator(self, name: str):
        """get the raw generator module (unwrap dataparallel if needed)."""
        gen = getattr(self.model, name)
        if isinstance(gen, DataParallel):
            return gen.module
        return gen

    def _compression_enabled(self, epoch: int) -> bool:
        """check if compression is enabled for this epoch."""
        return epoch >= self.compression_warmup_epochs

    def compute_ssim(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """compute ssim between two tensors."""
        x = x.detach().cpu()
        y = y.detach().cpu()

        mu_x = x.mean()
        mu_y = y.mean()
        sigma_x = ((x - mu_x) ** 2).mean()
        sigma_y = ((y - mu_y) ** 2).mean()
        sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

        C1, C2 = 0.01**2, 0.03**2
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
            (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
        )
        return ssim.item()

    def compute_psnr(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """compute psnr between two tensors."""
        mse = ((x - y) ** 2).mean().item()
        if mse == 0:
            return 100.0
        return 10 * np.log10(1.0 / mse)

    def train_step(
        self, batch: Dict[str, torch.Tensor], epoch: int
    ) -> Dict[str, float]:
        """execute single training step with rate-distortion loss."""
        real_A = batch["A"].to(self.device)
        real_B = batch["B"].to(self.device)
        center_A = batch["A_center"].to(self.device)
        center_B = batch["B_center"].to(self.device)

        use_compression = self._compression_enabled(epoch)

        # ================================================================
        # train generators
        # ================================================================
        self.opt_G.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=self.use_amp):
            G_A2B = self._get_generator("G_A2B")
            G_B2A = self._get_generator("G_B2A")

            # forward translation with optional compression
            if use_compression:
                fake_B, bits_A2B, bpe_A2B = G_A2B(real_A)
                fake_A, bits_B2A, bpe_B2A = G_B2A(real_B)
            else:
                fake_B = G_A2B.forward_uncompressed(real_A)
                fake_A = G_B2A.forward_uncompressed(real_B)
                bits_A2B = torch.tensor(0.0, device=self.device)
                bits_B2A = torch.tensor(0.0, device=self.device)
                bpe_A2B = torch.tensor(0.0, device=self.device)
                bpe_B2A = torch.tensor(0.0, device=self.device)

            # cycle consistency
            fake_B_3slice = (
                fake_B.unsqueeze(2)
                .repeat(1, 1, 3, 1, 1)
                .view(fake_B.size(0), -1, fake_B.size(2), fake_B.size(3))
            )
            fake_A_3slice = (
                fake_A.unsqueeze(2)
                .repeat(1, 1, 3, 1, 1)
                .view(fake_A.size(0), -1, fake_A.size(2), fake_A.size(3))
            )

            if use_compression:
                rec_A, bits_rec_A, _ = G_B2A(fake_B_3slice)
                rec_B, bits_rec_B, _ = G_A2B(fake_A_3slice)
            else:
                rec_A = G_B2A.forward_uncompressed(fake_B_3slice)
                rec_B = G_A2B.forward_uncompressed(fake_A_3slice)
                bits_rec_A = torch.tensor(0.0, device=self.device)
                bits_rec_B = torch.tensor(0.0, device=self.device)

            loss_cycle_A = self.losses.cycle_loss(center_A, rec_A)
            loss_cycle_B = self.losses.cycle_loss(center_B, rec_B)

            # identity loss
            if use_compression:
                identity_A, _, _ = G_B2A(real_A)
                identity_B, _, _ = G_A2B(real_B)
            else:
                identity_A = G_B2A.forward_uncompressed(real_A)
                identity_B = G_A2B.forward_uncompressed(real_B)

            loss_identity_A = self.losses.identity_loss(center_A, identity_A)
            loss_identity_B = self.losses.identity_loss(center_B, identity_B)

            # adversarial loss
            pred_fake_B = self.model.D_B(fake_B)
            pred_fake_A = self.model.D_A(fake_A)

            loss_gan_A2B = self.losses.gan_loss.generator_loss(pred_fake_B)
            loss_gan_B2A = self.losses.gan_loss.generator_loss(pred_fake_A)

            # ssim loss
            loss_ssim = self.losses.ssim_loss(center_A, rec_A) + self.losses.ssim_loss(
                center_B, rec_B
            )

            # rate loss (bitrate from entropy model)
            total_bits = bits_A2B + bits_B2A + bits_rec_A + bits_rec_B
            loss_rate = self.lambda_rate * total_bits if use_compression else torch.tensor(0.0, device=self.device)

            # total generator loss
            loss_G = (
                loss_gan_A2B
                + loss_gan_B2A
                + loss_cycle_A
                + loss_cycle_B
                + loss_identity_A
                + loss_identity_B
                + loss_ssim
                + loss_rate
            )

        self.scaler_G.scale(loss_G).backward()

        if self.gradient_clip_norm > 0:
            self.scaler_G.unscale_(self.opt_G)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.G_A2B.parameters())
                + list(self.model.G_B2A.parameters()),
                self.gradient_clip_norm,
            )

        self.scaler_G.step(self.opt_G)
        self.scaler_G.update()

        # ================================================================
        # train discriminators
        # ================================================================
        self.opt_D.zero_grad(set_to_none=True)

        fake_A_buffer = self.fake_A_buffer.push_and_pop(fake_A.detach())
        fake_B_buffer = self.fake_B_buffer.push_and_pop(fake_B.detach())

        with autocast("cuda", enabled=self.use_amp):
            pred_real_A = self.model.D_A(center_A)
            pred_fake_A_d = self.model.D_A(fake_A_buffer)
            loss_D_A = self.losses.gan_loss.discriminator_loss(pred_real_A, pred_fake_A_d)

            pred_real_B = self.model.D_B(center_B)
            pred_fake_B_d = self.model.D_B(fake_B_buffer)
            loss_D_B = self.losses.gan_loss.discriminator_loss(pred_real_B, pred_fake_B_d)

            loss_D = (loss_D_A + loss_D_B) * 0.5

        self.scaler_D.scale(loss_D).backward()

        if self.gradient_clip_norm > 0:
            self.scaler_D.unscale_(self.opt_D)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.D_A.parameters()) + list(self.model.D_B.parameters()),
                self.gradient_clip_norm,
            )

        self.scaler_D.step(self.opt_D)
        self.scaler_D.update()

        avg_bpe = ((bpe_A2B + bpe_B2A) / 2).item() if use_compression else 0.0

        return {
            "G_loss": loss_G.item(),
            "D_loss": loss_D.item(),
            "cycle_A": loss_cycle_A.item(),
            "cycle_B": loss_cycle_B.item(),
            "identity_A": loss_identity_A.item(),
            "identity_B": loss_identity_B.item(),
            "gan_A2B": loss_gan_A2B.item(),
            "gan_B2A": loss_gan_B2A.item(),
            "ssim_loss": loss_ssim.item(),
            "rate_loss": loss_rate.item() if isinstance(loss_rate, torch.Tensor) else loss_rate,
            "bpe": avg_bpe,
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """train for one epoch."""
        self.model.train()

        epoch_losses = {
            "G_loss": 0,
            "D_loss": 0,
            "cycle_A": 0,
            "cycle_B": 0,
            "rate_loss": 0,
            "bpe": 0,
            "ssim_loss": 0,
        }

        compression_status = "on" if self._compression_enabled(epoch) else "warmup"
        pbar = tqdm(
            self.train_loader,
            desc=f"epoch {epoch} [compress: {compression_status}]",
            ncols=130,
            leave=True,
        )

        for batch_idx, batch in enumerate(pbar):
            losses = self.train_step(batch, epoch)

            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key]

            # tensorboard logging
            self.global_step += 1
            if self.global_step % 100 == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f"train/{k}", v, self.global_step)

            pbar.set_postfix(
                {
                    "G": f'{losses["G_loss"]:.3f}',
                    "D": f'{losses["D_loss"]:.3f}',
                    "rate": f'{losses["rate_loss"]:.3f}',
                    "bpe": f'{losses["bpe"]:.3f}',
                    "cyc": f'{losses["cycle_A"] + losses["cycle_B"]:.3f}',
                }
            )

        n_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """validate the model."""
        self.model.eval()

        metrics = {
            "ssim_A2B": [],
            "ssim_B2A": [],
            "psnr_A2B": [],
            "psnr_B2A": [],
            "bpe": [],
        }

        use_compression = self._compression_enabled(epoch)

        for batch in tqdm(self.val_loader, desc="validating", ncols=100, leave=False):
            real_A = batch["A"].to(self.device)
            real_B = batch["B"].to(self.device)
            center_A = batch["A_center"].to(self.device)
            center_B = batch["B_center"].to(self.device)

            G_A2B = self._get_generator("G_A2B")
            G_B2A = self._get_generator("G_B2A")

            if use_compression:
                fake_B, _, bpe_A2B = G_A2B(real_A)
                fake_A, _, bpe_B2A = G_B2A(real_B)
                metrics["bpe"].append(((bpe_A2B + bpe_B2A) / 2).item())
            else:
                fake_B = G_A2B.forward_uncompressed(real_A)
                fake_A = G_B2A.forward_uncompressed(real_B)

            # cycle reconstruction
            fake_B_3slice = (
                fake_B.unsqueeze(2)
                .repeat(1, 1, 3, 1, 1)
                .view(fake_B.size(0), -1, fake_B.size(2), fake_B.size(3))
            )
            fake_A_3slice = (
                fake_A.unsqueeze(2)
                .repeat(1, 1, 3, 1, 1)
                .view(fake_A.size(0), -1, fake_A.size(2), fake_A.size(3))
            )

            if use_compression:
                rec_A, _, _ = G_B2A(fake_B_3slice)
                rec_B, _, _ = G_A2B(fake_A_3slice)
            else:
                rec_A = G_B2A.forward_uncompressed(fake_B_3slice)
                rec_B = G_A2B.forward_uncompressed(fake_A_3slice)

            for i in range(center_A.size(0)):
                metrics["ssim_A2B"].append(self.compute_ssim(center_A[i], rec_A[i]))
                metrics["ssim_B2A"].append(self.compute_ssim(center_B[i], rec_B[i]))
                metrics["psnr_A2B"].append(self.compute_psnr(center_A[i], rec_A[i]))
                metrics["psnr_B2A"].append(self.compute_psnr(center_B[i], rec_B[i]))

        result = {
            "ssim_A2B": np.mean(metrics["ssim_A2B"]),
            "ssim_B2A": np.mean(metrics["ssim_B2A"]),
            "psnr_A2B": np.mean(metrics["psnr_A2B"]),
            "psnr_B2A": np.mean(metrics["psnr_B2A"]),
        }
        if metrics["bpe"]:
            result["bpe"] = np.mean(metrics["bpe"])
        return result

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "opt_G_state_dict": self.opt_G.state_dict(),
            "opt_D_state_dict": self.opt_D.state_dict(),
            "scheduler_G_state_dict": self.scheduler_G.state_dict(),
            "scheduler_D_state_dict": self.scheduler_D.state_dict(),
            "history": self.history,
            "best_val_ssim": self.best_val_ssim,
            "global_step": self.global_step,
            "config": self.config.__dict__,
            "lambda_rate": self.lambda_rate,
        }

        ckpt_dir = self.experiment_dir / "checkpoints"
        torch.save(checkpoint, ckpt_dir / "checkpoint_latest.pth")

        if epoch % 10 == 0:
            torch.save(checkpoint, ckpt_dir / f"checkpoint_epoch_{epoch}.pth")

        if is_best:
            torch.save(checkpoint, ckpt_dir / "checkpoint_best.pth")

    def load_checkpoint(self, checkpoint_path: str):
        """load checkpoint for resuming training."""
        print(f"loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.opt_G.load_state_dict(checkpoint["opt_G_state_dict"])
        self.opt_D.load_state_dict(checkpoint["opt_D_state_dict"])
        self.scheduler_G.load_state_dict(checkpoint["scheduler_G_state_dict"])
        self.scheduler_D.load_state_dict(checkpoint["scheduler_D_state_dict"])
        self.history = checkpoint.get("history", self.history)
        self.best_val_ssim = checkpoint.get("best_val_ssim", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.start_epoch = checkpoint["epoch"] + 1

        print(f"resumed from epoch {self.start_epoch}")

    def train(self, epochs: int = 200, validate_every: int = 1, save_every: int = 5):
        """full training loop."""
        print(f"\nstarting training for {epochs} epochs")
        print(f"  validate every: {validate_every} epochs")
        print(f"  save every: {save_every} epochs")
        print(f"  lambda_rate: {self.lambda_rate}")
        print(f"  compression warmup: {self.compression_warmup_epochs} epochs")
        print()

        for epoch in range(self.start_epoch, epochs):
            epoch_start = time.time()

            # warmup learning rate
            if epoch < self.warmup_epochs:
                warmup_factor = (epoch + 1) / self.warmup_epochs
                for param_group in self.opt_G.param_groups:
                    param_group["lr"] = self.base_lr_G * warmup_factor
                for param_group in self.opt_D.param_groups:
                    param_group["lr"] = self.base_lr_D * warmup_factor

            # train
            train_losses = self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start

            # step schedulers (after warmup)
            if epoch >= self.warmup_epochs:
                self.scheduler_G.step()
                self.scheduler_D.step()

            # log
            lr = self.opt_G.param_groups[0]["lr"]
            self.writer.add_scalar("lr/generator", lr, epoch)

            compress_str = "on" if self._compression_enabled(epoch) else "warmup"
            print(
                f"  epoch {epoch} [{compress_str}]: G={train_losses['G_loss']:.4f} "
                f"D={train_losses['D_loss']:.4f} "
                f"rate={train_losses['rate_loss']:.4f} "
                f"bpe={train_losses['bpe']:.3f} "
                f"cycle={train_losses['cycle_A'] + train_losses['cycle_B']:.4f} "
                f"lr={lr:.2e} time={epoch_time:.1f}s"
            )

            # validate
            if (epoch + 1) % validate_every == 0:
                val_metrics = self.validate(epoch)
                mean_ssim = (val_metrics["ssim_A2B"] + val_metrics["ssim_B2A"]) / 2

                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)

                bpe_str = f" bpe={val_metrics.get('bpe', 0):.3f}" if "bpe" in val_metrics else ""
                print(
                    f"  val: ssim_A2B={val_metrics['ssim_A2B']:.4f} "
                    f"ssim_B2A={val_metrics['ssim_B2A']:.4f} "
                    f"psnr_A2B={val_metrics['psnr_A2B']:.2f} "
                    f"psnr_B2A={val_metrics['psnr_B2A']:.2f}{bpe_str}"
                )

                is_best = mean_ssim > self.best_val_ssim
                if is_best:
                    self.best_val_ssim = mean_ssim
                    print(f"  *** new best val ssim: {mean_ssim:.4f} ***")

                if (epoch + 1) % save_every == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)

            # update history
            self.history["train"]["G_loss"].append(train_losses["G_loss"])
            self.history["train"]["D_loss"].append(train_losses["D_loss"])
            self.history["train"]["rate_loss"].append(train_losses["rate_loss"])
            self.history["train"]["bpe"].append(train_losses["bpe"])
            self.history["epoch_times"].append(epoch_time)
            self.history["learning_rate"].append(lr)

        # save final
        self.save_checkpoint(epochs - 1, is_best=False)
        self.writer.close()

        # save training history
        with open(self.experiment_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        print("\ntraining complete!")
        print(f"best val ssim: {self.best_val_ssim:.4f}")
        print(f"checkpoints saved to: {self.experiment_dir / 'checkpoints'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="train harmonize-and-compress sa-cyclegan-2.5d"
    )

    # config file
    parser.add_argument("--config", type=str, default=None, help="path to yaml config")

    # data
    parser.add_argument("--brats_dir", type=str, default=None)
    parser.add_argument("--upenn_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    # training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--lr_G", type=float, default=5e-5)
    parser.add_argument("--lr_D", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=32)

    # compression
    parser.add_argument("--entropy_model_type", type=str, default="factorized",
                        choices=["factorized", "hyperprior"])
    parser.add_argument("--num_hyper_channels", type=int, default=128)
    parser.add_argument("--lambda_rate", type=float, default=0.01)
    parser.add_argument("--compression_warmup_epochs", type=int, default=20)

    # model
    parser.add_argument("--n_residual_blocks", type=int, default=9)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)

    # checkpointing
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to resume")
    parser.add_argument("--experiment_name", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # load config from yaml if provided
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k) and getattr(args, k) is None:
                setattr(args, k, v)
            elif not hasattr(args, k):
                setattr(args, k, v)

    # create model config
    config = SACycleGAN25DConfig(
        ngf=args.ngf,
        ndf=args.ndf,
        n_residual_blocks=args.n_residual_blocks,
        attention_layers=tuple(getattr(args, "attention_layers", [3, 4, 5])),
    )

    # create trainer
    trainer = CompressionTrainer(
        config=config,
        brats_dir=args.brats_dir,
        upenn_dir=args.upenn_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr_G=args.lr_G,
        lr_D=args.lr_D,
        num_workers=args.num_workers,
        experiment_name=getattr(args, "experiment_name", None),
        entropy_model_type=args.entropy_model_type,
        num_hyper_channels=args.num_hyper_channels,
        lambda_rate=args.lambda_rate,
        compression_warmup_epochs=args.compression_warmup_epochs,
    )

    # resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # train
    trainer.train(
        epochs=args.epochs,
        validate_every=getattr(args, "validate_every", 1),
        save_every=getattr(args, "save_every", 5),
    )


if __name__ == "__main__":
    main()
