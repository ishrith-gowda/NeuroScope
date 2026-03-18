#!/usr/bin/env python3
"""
training script for multi-domain sa-cyclegan-2.5d with adain conditioning.

trains a single generator to harmonize across n>2 scanner domains using
adaptive instance normalization. the discriminator includes a domain
classification head (stargan-style) providing auxiliary training signal.

extension c of the journal extension.

usage:
    python train_multi_domain.py --config ../configs/multi_domain.yaml
    python train_multi_domain.py --n_domains 4 --style_dim 256
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
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

# add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from neuroscope.models.architectures.sa_cyclegan_25d_multidomain import (
    MultiDomainConfig,
    MultiDomainSACycleGAN25D,
)


class MultiDomainMRIDataset(Dataset):
    """
    multi-domain mri dataset for n>2 site harmonization.

    loads 2.5d slices from multiple scanner domains. each sample includes
    the 3-slice input, center slice ground truth, and domain label.
    """

    def __init__(
        self,
        data_dirs: Dict[str, str],
        domain_names: List[str],
        image_size: Tuple[int, int] = (128, 128),
        split: str = "train",
    ):
        """
        args:
            data_dirs: mapping of domain_name -> data directory path
            domain_names: ordered list of domain names
            image_size: target spatial resolution
            split: train/val/test
        """
        self.data_dirs = data_dirs
        self.domain_names = domain_names
        self.image_size = image_size
        self.domain_to_id = {name: i for i, name in enumerate(domain_names)}

        # collect all samples with domain labels
        self.samples = []
        for domain_name in domain_names:
            domain_dir = Path(data_dirs[domain_name])
            if not domain_dir.exists():
                print(f"warning: domain directory not found: {domain_dir}")
                continue

            # look for preprocessed nifti or numpy files
            for ext in ["*.npy", "*.npz", "*.nii.gz"]:
                files = sorted(domain_dir.glob(f"**/{ext}"))
                for f in files:
                    self.samples.append({
                        "path": str(f),
                        "domain": domain_name,
                        "domain_id": self.domain_to_id[domain_name],
                    })

        print(f"loaded {len(self.samples)} samples across {len(domain_names)} domains")
        for name in domain_names:
            count = sum(1 for s in self.samples if s["domain"] == name)
            print(f"  {name}: {count} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample["path"]
        domain_id = sample["domain_id"]

        # load volume slice
        if path.endswith(".npy"):
            data = np.load(path)
        elif path.endswith(".npz"):
            data = np.load(path)["data"]
        else:
            import nibabel as nib
            data = nib.load(path).get_fdata()

        # convert to tensor and normalize to [-1, 1]
        data = torch.from_numpy(data).float()
        if data.max() > 0:
            data = data / data.max() * 2 - 1

        # ensure correct shape [c, h, w]
        if data.dim() == 2:
            data = data.unsqueeze(0)

        # resize if needed
        if data.shape[-2:] != tuple(self.image_size):
            data = torch.nn.functional.interpolate(
                data.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False
            ).squeeze(0)

        return {
            "data": data,
            "domain_id": torch.tensor(domain_id, dtype=torch.long),
            "domain_name": sample["domain"],
        }


def setup_torch_performance():
    """configure torch for maximum gpu throughput."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(enabled=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)


class MultiDomainTrainer:
    """
    trainer for multi-domain sa-cyclegan-2.5d.

    trains a single generator conditioned on target domain via adain,
    with a multi-task discriminator providing real/fake and domain
    classification signals.

    training loop:
    1. sample batch from random source domain
    2. sample random target domain for each sample
    3. translate source -> target
    4. cycle back target -> source
    5. compute adversarial + classification + cycle + identity losses
    """

    def __init__(
        self,
        config: MultiDomainConfig,
        data_dirs: Dict[str, str],
        domain_names: List[str],
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
        lambda_cls: float = 1.0,
        lambda_rec: float = 10.0,
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
        self.domain_names = domain_names
        self.n_domains = len(domain_names)

        # experiment naming
        if experiment_name is None:
            experiment_name = f"multidomain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

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
        print("  multi-domain sa-cyclegan-2.5d training")
        print("=" * 60)
        print(f"device: {self.device}")
        if self.use_multi_gpu:
            print(f"multi-gpu: {self.num_gpus} gpus (dataparallel)")
        print(f"experiment: {experiment_name}")
        print(f"domains: {domain_names}")
        print(f"lambda_cls: {lambda_cls}")
        print(f"lambda_rec: {lambda_rec}")

        # tensorboard
        runs_dir = Path("/data/runs") if Path("/data/runs").exists() else PROJECT_ROOT / "runs"
        self.writer = SummaryWriter(
            log_dir=str(runs_dir / experiment_name)
        )

        # create model
        self.model = MultiDomainSACycleGAN25D(config)
        self.model = self.model.to(self.device)

        self.lambda_cls = lambda_cls
        self.lambda_rec = lambda_rec

        # wrap with dataparallel
        if self.use_multi_gpu:
            self.model.generator = DataParallel(self.model.generator)
            self.model.discriminator = DataParallel(self.model.discriminator)

        # create dataset and dataloader
        self.dataset = MultiDomainMRIDataset(
            data_dirs=data_dirs,
            domain_names=domain_names,
            image_size=(image_size, image_size),
        )
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        print(f"training batches: {len(self.train_loader)}")

        # optimizers
        self.opt_G = optim.Adam(
            self.model.generator.parameters(),
            lr=lr_G,
            betas=(beta1, beta2),
        )
        self.opt_D = optim.Adam(
            self.model.discriminator.parameters(),
            lr=lr_D,
            betas=(beta1, beta2),
        )

        # schedulers
        self.base_lr_G = lr_G
        self.base_lr_D = lr_D
        self.warmup_epochs = warmup_epochs

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

            self.scheduler_G = optim.lr_scheduler.LambdaLR(self.opt_G, lr_lambda=lambda_rule)
            self.scheduler_D = optim.lr_scheduler.LambdaLR(self.opt_D, lr_lambda=lambda_rule)

        # loss functions
        self.adv_loss = nn.MSELoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.rec_loss = nn.L1Loss()

        # mixed precision
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler_G = GradScaler("cuda", enabled=self.use_amp)
        self.scaler_D = GradScaler("cuda", enabled=self.use_amp)
        self.gradient_clip_norm = gradient_clip_norm

        # training history
        self.history = {
            "train": {"G_loss": [], "D_loss": [], "cls_loss": [], "rec_loss": []},
            "learning_rate": [],
            "epoch_times": [],
        }

        self.start_epoch = 0
        self.best_val_metric = 0
        self.global_step = 0

        self._save_config(locals())

    def _save_config(self, init_args: dict):
        config_dict = {
            "model": {k: v for k, v in self.config.__dict__.items()},
            "training": {
                "lambda_cls": self.lambda_cls,
                "lambda_rec": self.lambda_rec,
                "domain_names": self.domain_names,
                "device": str(self.device),
                "experiment_name": self.experiment_name,
            },
        }
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    def _get_generator(self):
        gen = self.model.generator
        if isinstance(gen, DataParallel):
            return gen.module
        return gen

    def _get_discriminator(self):
        disc = self.model.discriminator
        if isinstance(disc, DataParallel):
            return disc.module
        return disc

    def _sample_target_domains(self, source_domains: torch.Tensor) -> torch.Tensor:
        """sample random target domains different from source."""
        targets = torch.zeros_like(source_domains)
        for i in range(len(source_domains)):
            choices = [d for d in range(self.n_domains) if d != source_domains[i].item()]
            targets[i] = np.random.choice(choices)
        return targets

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """single training step for multi-domain model."""
        real = batch["data"].to(self.device)
        src_domain = batch["domain_id"].to(self.device)

        # need 12-channel input (3 slices x 4 modalities)
        # if input doesn't match, repeat to simulate 2.5d
        if real.size(1) < 12:
            n_repeats = 12 // real.size(1)
            real = real.repeat(1, n_repeats, 1, 1)[:, :12]

        center = real[:, :4]  # center slice (first 4 channels)
        tgt_domain = self._sample_target_domains(src_domain)

        # ================================================================
        # train discriminator
        # ================================================================
        self.opt_D.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=self.use_amp):
            # real images
            adv_real, cls_real = self.model.discriminator(center)
            loss_D_real = self.adv_loss(adv_real, torch.ones_like(adv_real))
            loss_D_cls = self.cls_loss(cls_real, src_domain)

            # fake images
            with torch.no_grad():
                fake = self.model.generator(real, tgt_domain)
            adv_fake, _ = self.model.discriminator(fake.detach())
            loss_D_fake = self.adv_loss(adv_fake, torch.zeros_like(adv_fake))

            loss_D = loss_D_real + loss_D_fake + self.lambda_cls * loss_D_cls

        self.scaler_D.scale(loss_D).backward()
        if self.gradient_clip_norm > 0:
            self.scaler_D.unscale_(self.opt_D)
            torch.nn.utils.clip_grad_norm_(
                self.model.discriminator.parameters(), self.gradient_clip_norm
            )
        self.scaler_D.step(self.opt_D)
        self.scaler_D.update()

        # ================================================================
        # train generator
        # ================================================================
        self.opt_G.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=self.use_amp):
            # forward translation
            fake = self.model.generator(real, tgt_domain)

            # adversarial + classification loss
            adv_fake_g, cls_fake = self.model.discriminator(fake)
            loss_G_adv = self.adv_loss(adv_fake_g, torch.ones_like(adv_fake_g))
            loss_G_cls = self.cls_loss(cls_fake, tgt_domain)

            # cycle consistency
            fake_3slice = (
                fake.unsqueeze(2)
                .repeat(1, 1, 3, 1, 1)
                .view(fake.size(0), -1, fake.size(2), fake.size(3))
            )
            rec = self.model.generator(fake_3slice, src_domain)
            loss_rec = self.rec_loss(rec, center)

            # identity loss
            idt = self.model.generator(real, src_domain)
            loss_idt = self.rec_loss(idt, center)

            loss_G = (
                loss_G_adv
                + self.lambda_cls * loss_G_cls
                + self.lambda_rec * loss_rec
                + self.lambda_rec * 0.5 * loss_idt
            )

        self.scaler_G.scale(loss_G).backward()
        if self.gradient_clip_norm > 0:
            self.scaler_G.unscale_(self.opt_G)
            torch.nn.utils.clip_grad_norm_(
                self.model.generator.parameters(), self.gradient_clip_norm
            )
        self.scaler_G.step(self.opt_G)
        self.scaler_G.update()

        return {
            "G_loss": loss_G.item(),
            "D_loss": loss_D.item(),
            "G_adv": loss_G_adv.item(),
            "G_cls": loss_G_cls.item(),
            "D_cls": loss_D_cls.item(),
            "rec_loss": loss_rec.item(),
            "idt_loss": loss_idt.item(),
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_losses = {"G_loss": 0, "D_loss": 0, "G_cls": 0, "rec_loss": 0}

        pbar = tqdm(self.train_loader, desc=f"epoch {epoch}", ncols=120, leave=True)
        for batch in pbar:
            losses = self.train_step(batch)
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key]

            self.global_step += 1
            if self.global_step % 100 == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f"train/{k}", v, self.global_step)

            pbar.set_postfix({
                "G": f'{losses["G_loss"]:.3f}',
                "D": f'{losses["D_loss"]:.3f}',
                "cls": f'{losses["G_cls"]:.3f}',
                "rec": f'{losses["rec_loss"]:.3f}',
            })

        n = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n
        return epoch_losses

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "opt_G_state_dict": self.opt_G.state_dict(),
            "opt_D_state_dict": self.opt_D.state_dict(),
            "scheduler_G_state_dict": self.scheduler_G.state_dict(),
            "scheduler_D_state_dict": self.scheduler_D.state_dict(),
            "history": self.history,
            "global_step": self.global_step,
            "config": self.config.__dict__,
        }
        ckpt_dir = self.experiment_dir / "checkpoints"
        torch.save(checkpoint, ckpt_dir / "checkpoint_latest.pth")
        if epoch % 10 == 0:
            torch.save(checkpoint, ckpt_dir / f"checkpoint_epoch_{epoch}.pth")
        if is_best:
            torch.save(checkpoint, ckpt_dir / "checkpoint_best.pth")

    def load_checkpoint(self, checkpoint_path: str):
        print(f"loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.opt_G.load_state_dict(checkpoint["opt_G_state_dict"])
        self.opt_D.load_state_dict(checkpoint["opt_D_state_dict"])
        self.scheduler_G.load_state_dict(checkpoint["scheduler_G_state_dict"])
        self.scheduler_D.load_state_dict(checkpoint["scheduler_D_state_dict"])
        self.history = checkpoint.get("history", self.history)
        self.global_step = checkpoint.get("global_step", 0)
        self.start_epoch = checkpoint["epoch"] + 1
        print(f"resumed from epoch {self.start_epoch}")

    def train(self, epochs: int = 200, save_every: int = 5):
        print(f"\nstarting multi-domain training for {epochs} epochs")
        print(f"  n_domains: {self.n_domains}")
        print(f"  lambda_cls: {self.lambda_cls}")
        print(f"  lambda_rec: {self.lambda_rec}")
        print()

        for epoch in range(self.start_epoch, epochs):
            epoch_start = time.time()

            if epoch < self.warmup_epochs:
                warmup_factor = (epoch + 1) / self.warmup_epochs
                for pg in self.opt_G.param_groups:
                    pg["lr"] = self.base_lr_G * warmup_factor
                for pg in self.opt_D.param_groups:
                    pg["lr"] = self.base_lr_D * warmup_factor

            train_losses = self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start

            if epoch >= self.warmup_epochs:
                self.scheduler_G.step()
                self.scheduler_D.step()

            lr = self.opt_G.param_groups[0]["lr"]
            self.writer.add_scalar("lr/generator", lr, epoch)

            print(
                f"  epoch {epoch}: G={train_losses['G_loss']:.4f} "
                f"D={train_losses['D_loss']:.4f} "
                f"cls={train_losses['G_cls']:.4f} "
                f"rec={train_losses['rec_loss']:.4f} "
                f"lr={lr:.2e} time={epoch_time:.1f}s"
            )

            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch)

            self.history["train"]["G_loss"].append(train_losses["G_loss"])
            self.history["train"]["D_loss"].append(train_losses["D_loss"])
            self.history["train"]["cls_loss"].append(train_losses["G_cls"])
            self.history["train"]["rec_loss"].append(train_losses["rec_loss"])
            self.history["epoch_times"].append(epoch_time)
            self.history["learning_rate"].append(lr)

        self.save_checkpoint(epochs - 1)
        self.writer.close()

        with open(self.experiment_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        print("\ntraining complete!")
        print(f"checkpoints saved to: {self.experiment_dir / 'checkpoints'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="train multi-domain sa-cyclegan-2.5d with adain"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--lr_G", type=float, default=5e-5)
    parser.add_argument("--lr_D", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--n_domains", type=int, default=4)
    parser.add_argument("--style_dim", type=int, default=256)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_rec", type=float, default=10.0)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--n_residual_blocks", type=int, default=9)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k) and getattr(args, k) is None:
                setattr(args, k, v)
            elif not hasattr(args, k):
                setattr(args, k, v)

    config = MultiDomainConfig(
        n_domains=getattr(args, "n_domains", 4),
        domain_embed_dim=getattr(args, "domain_embed_dim", 64),
        style_dim=getattr(args, "style_dim", 256),
        ngf=args.ngf,
        ndf=args.ndf,
        n_residual_blocks=args.n_residual_blocks,
        attention_layers=tuple(getattr(args, "attention_layers", [3, 4, 5])),
    )

    data_dirs = getattr(args, "data_dirs", {})
    domain_names = getattr(args, "domain_names", [])

    trainer = MultiDomainTrainer(
        config=config,
        data_dirs=data_dirs,
        domain_names=domain_names,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr_G=args.lr_G,
        lr_D=args.lr_D,
        num_workers=args.num_workers,
        experiment_name=getattr(args, "experiment_name", None),
        lambda_cls=getattr(args, "lambda_cls", 1.0),
        lambda_rec=getattr(args, "lambda_rec", 10.0),
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(
        epochs=args.epochs,
        save_every=getattr(args, "save_every", 5),
    )


if __name__ == "__main__":
    main()
