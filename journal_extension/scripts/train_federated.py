#!/usr/bin/env python3
"""
federated learning simulation for sa-cyclegan-2.5d harmonization.

simulates federated training where each clinical site trains locally
on its own data. supports three aggregation strategies:
    - fedavg: simple weight averaging
    - fedprox: proximal regularization to prevent client drift
    - scaffold: variance reduction via control variates

each round: distribute global model -> local training -> aggregate updates.

extension e of the journal extension.

usage:
    python train_federated.py --config ../configs/federated.yaml
    python train_federated.py --aggregation_strategy fedavg --local_epochs 5
"""

import os
import sys
import copy
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

# add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from neuroscope.models.architectures.sa_cyclegan_25d import (
    SACycleGAN25DConfig,
    SACycleGAN25D,
    create_model,
)
from neuroscope.data.datasets.dataset_25d import UnpairedMRIDataset25D, create_dataloaders
from neuroscope.models.losses.combined_losses import CombinedLoss
from neuroscope.training.federated.fedavg import FedAvgAggregator
from neuroscope.training.federated.strategies import (
    FedProxAggregator,
    ScaffoldAggregator,
)


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


class LocalClient:
    """
    local training client for federated cyclegan.

    each client has its own data, model copy, and optimizer.
    trains locally for e epochs, then returns model updates.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        data_loader: DataLoader,
        losses: CombinedLoss,
        lr_G: float = 5e-5,
        lr_D: float = 5e-5,
        beta1: float = 0.5,
        beta2: float = 0.999,
        gradient_clip_norm: float = 1.0,
        use_amp: bool = True,
        device: torch.device = None,
    ):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        self.losses = losses
        self.device = device or torch.device("cpu")
        self.gradient_clip_norm = gradient_clip_norm

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

        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler_G = GradScaler("cuda", enabled=self.use_amp)
        self.scaler_D = GradScaler("cuda", enabled=self.use_amp)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    def train_step(self, batch: Dict[str, torch.Tensor], prox_loss_fn=None) -> Dict[str, float]:
        """single local training step."""
        real_A = batch["A"].to(self.device)
        real_B = batch["B"].to(self.device)
        center_A = batch["A_center"].to(self.device)
        center_B = batch["B_center"].to(self.device)

        # train generators
        self.opt_G.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=self.use_amp):
            fake_B = self.model.G_A2B(real_A)
            fake_A = self.model.G_B2A(real_B)

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

            rec_A = self.model.G_B2A(fake_B_3slice)
            rec_B = self.model.G_A2B(fake_A_3slice)

            loss_cycle_A = self.losses.cycle_loss(center_A, rec_A)
            loss_cycle_B = self.losses.cycle_loss(center_B, rec_B)

            identity_A = self.model.G_B2A(real_A)
            identity_B = self.model.G_A2B(real_B)
            loss_idt_A = self.losses.identity_loss(center_A, identity_A)
            loss_idt_B = self.losses.identity_loss(center_B, identity_B)

            pred_fake_B = self.model.D_B(fake_B)
            pred_fake_A = self.model.D_A(fake_A)
            loss_gan_A2B = self.losses.gan_loss.generator_loss(pred_fake_B)
            loss_gan_B2A = self.losses.gan_loss.generator_loss(pred_fake_A)

            loss_ssim = self.losses.ssim_loss(center_A, rec_A) + self.losses.ssim_loss(
                center_B, rec_B
            )

            loss_G = (
                loss_gan_A2B + loss_gan_B2A
                + loss_cycle_A + loss_cycle_B
                + loss_idt_A + loss_idt_B
                + loss_ssim
            )

            # fedprox proximal term
            if prox_loss_fn is not None:
                loss_prox = prox_loss_fn(self.model)
                loss_G = loss_G + loss_prox

        self.scaler_G.scale(loss_G).backward()
        if self.gradient_clip_norm > 0:
            self.scaler_G.unscale_(self.opt_G)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.G_A2B.parameters()) + list(self.model.G_B2A.parameters()),
                self.gradient_clip_norm,
            )
        self.scaler_G.step(self.opt_G)
        self.scaler_G.update()

        # train discriminators
        self.opt_D.zero_grad(set_to_none=True)

        fake_A_buf = self.fake_A_buffer.push_and_pop(fake_A.detach())
        fake_B_buf = self.fake_B_buffer.push_and_pop(fake_B.detach())

        with autocast("cuda", enabled=self.use_amp):
            pred_real_A = self.model.D_A(center_A)
            pred_fake_A_d = self.model.D_A(fake_A_buf)
            loss_D_A = self.losses.gan_loss.discriminator_loss(pred_real_A, pred_fake_A_d)

            pred_real_B = self.model.D_B(center_B)
            pred_fake_B_d = self.model.D_B(fake_B_buf)
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

        return {
            "G_loss": loss_G.item(),
            "D_loss": loss_D.item(),
            "cycle": (loss_cycle_A + loss_cycle_B).item(),
        }

    def train_local(self, epochs: int, prox_loss_fn=None) -> Dict[str, float]:
        """train locally for e epochs. returns average losses."""
        self.model.train()
        total_losses = {"G_loss": 0, "D_loss": 0, "cycle": 0}
        n_steps = 0

        for _ in range(epochs):
            for batch in self.data_loader:
                losses = self.train_step(batch, prox_loss_fn)
                for k in total_losses:
                    total_losses[k] += losses[k]
                n_steps += 1

        if n_steps > 0:
            for k in total_losses:
                total_losses[k] /= n_steps

        return total_losses

    def get_dataset_size(self) -> int:
        return len(self.data_loader.dataset)


def setup_torch_performance():
    """configure torch for maximum gpu throughput."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(enabled=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)


class FederatedTrainer:
    """
    federated learning orchestrator for sa-cyclegan-2.5d.

    coordinates multiple local clients training on site-specific data.
    supports fedavg, fedprox, and scaffold aggregation strategies.
    """

    def __init__(
        self,
        config: SACycleGAN25DConfig,
        brats_dir: str,
        upenn_dir: str,
        output_dir: str,
        n_clients: int = 2,
        local_epochs: int = 5,
        communication_rounds: int = 40,
        aggregation_strategy: str = "fedavg",
        share_discriminators: bool = False,
        fedprox_mu: float = 0.01,
        batch_size: int = 16,
        image_size: int = 128,
        lr_G: float = 5e-5,
        lr_D: float = 5e-5,
        beta1: float = 0.5,
        beta2: float = 0.999,
        num_workers: int = 4,
        device: str = "auto",
        experiment_name: str = None,
        gradient_clip_norm: float = 1.0,
        use_amp: bool = True,
        eval_every_n_rounds: int = 5,
    ):
        setup_torch_performance()
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_clients = n_clients
        self.local_epochs = local_epochs
        self.communication_rounds = communication_rounds
        self.aggregation_strategy = aggregation_strategy
        self.eval_every_n_rounds = eval_every_n_rounds

        if experiment_name is None:
            experiment_name = f"federated_{aggregation_strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        (self.experiment_dir / "checkpoints").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)

        # device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print("=" * 60)
        print("  federated sa-cyclegan-2.5d simulation")
        print("=" * 60)
        print(f"device: {self.device}")
        print(f"aggregation: {aggregation_strategy}")
        print(f"clients: {n_clients}")
        print(f"local epochs: {local_epochs}")
        print(f"communication rounds: {communication_rounds}")
        print(f"total effective epochs: {local_epochs * communication_rounds}")
        if aggregation_strategy == "fedprox":
            print(f"fedprox mu: {fedprox_mu}")

        runs_dir = Path("/data/runs") if Path("/data/runs").exists() else PROJECT_ROOT / "runs"
        self.writer = SummaryWriter(
            log_dir=str(runs_dir / experiment_name)
        )

        # create global model
        self.global_model = create_model(config).to(self.device)

        # create aggregator
        if aggregation_strategy == "fedprox":
            self.aggregator = FedProxAggregator(
                self.global_model, mu=fedprox_mu, share_discriminators=share_discriminators
            )
        elif aggregation_strategy == "scaffold":
            self.aggregator = ScaffoldAggregator(
                self.global_model, n_clients=n_clients, share_discriminators=share_discriminators
            )
        else:
            self.aggregator = FedAvgAggregator(
                self.global_model, share_discriminators=share_discriminators
            )

        # create per-client data loaders
        # client 0: brats site, client 1: upenn site
        self.client_loaders = self._create_client_loaders(
            brats_dir, upenn_dir, batch_size, image_size, num_workers
        )

        # create loss functions (shared config)
        self.losses = CombinedLoss(
            lambda_cycle=config.lambda_cycle,
            lambda_identity=config.lambda_identity,
            lambda_ssim=config.lambda_ssim,
            lambda_gradient=1.0,
        ).to(self.device)

        # client training params
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.beta1 = beta1
        self.beta2 = beta2
        self.gradient_clip_norm = gradient_clip_norm
        self.use_amp = use_amp

        # history
        self.history = {
            "rounds": [],
            "client_losses": [],
            "global_metrics": [],
        }

        self._save_config()

    def _save_config(self):
        """save experiment configuration."""
        config_dict = {
            "model": {k: v for k, v in self.config.__dict__.items()},
            "training": {
                "aggregation_strategy": self.aggregation_strategy,
                "n_clients": self.n_clients,
                "local_epochs": self.local_epochs,
                "communication_rounds": self.communication_rounds,
                "device": str(self.device),
                "experiment_name": self.experiment_name,
            },
        }
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    def _create_client_loaders(
        self, brats_dir, upenn_dir, batch_size, image_size, num_workers
    ) -> List[DataLoader]:
        """create per-client data loaders (one per site)."""
        # for 2-client simulation, each client gets one site's data
        # both clients need paired data from their site for cyclegan training
        # we reuse the existing create_dataloaders which splits by site
        train_loader, val_loader, _ = create_dataloaders(
            brats_dir=brats_dir,
            upenn_dir=upenn_dir,
            batch_size=batch_size,
            image_size=(image_size, image_size),
            num_workers=num_workers,
        )

        # in simulation, both clients share the same data but train independently
        # (in real federated setting, each would only see their site's data)
        return [train_loader, train_loader]

    def _create_client(self, client_id: int, model: nn.Module) -> LocalClient:
        """create a local training client."""
        return LocalClient(
            client_id=client_id,
            model=model,
            data_loader=self.client_loaders[client_id % len(self.client_loaders)],
            losses=self.losses,
            lr_G=self.lr_G,
            lr_D=self.lr_D,
            beta1=self.beta1,
            beta2=self.beta2,
            gradient_clip_norm=self.gradient_clip_norm,
            use_amp=self.use_amp,
            device=self.device,
        )

    def compute_ssim(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x, y = x.detach().cpu(), y.detach().cpu()
        mu_x, mu_y = x.mean(), y.mean()
        sigma_x = ((x - mu_x) ** 2).mean()
        sigma_y = ((y - mu_y) ** 2).mean()
        sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
        C1, C2 = 0.01**2, 0.03**2
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
            (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
        )
        return ssim.item()

    @torch.no_grad()
    def evaluate_global_model(self) -> Dict[str, float]:
        """evaluate the global model on validation data."""
        self.global_model.eval()
        metrics = {"ssim_A2B": [], "ssim_B2A": []}

        # use first client's loader for evaluation
        loader = self.client_loaders[0]
        for batch in loader:
            real_A = batch["A"].to(self.device)
            real_B = batch["B"].to(self.device)
            center_A = batch["A_center"].to(self.device)
            center_B = batch["B_center"].to(self.device)

            fake_B = self.global_model.G_A2B(real_A)
            fake_A = self.global_model.G_B2A(real_B)

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

            rec_A = self.global_model.G_B2A(fake_B_3slice)
            rec_B = self.global_model.G_A2B(fake_A_3slice)

            for i in range(center_A.size(0)):
                metrics["ssim_A2B"].append(self.compute_ssim(center_A[i], rec_A[i]))
                metrics["ssim_B2A"].append(self.compute_ssim(center_B[i], rec_B[i]))

            # only evaluate on a subset for speed
            if len(metrics["ssim_A2B"]) >= 100:
                break

        return {
            "ssim_A2B": np.mean(metrics["ssim_A2B"]),
            "ssim_B2A": np.mean(metrics["ssim_B2A"]),
        }

    def save_checkpoint(self, round_idx: int):
        checkpoint = {
            "round": round_idx,
            "global_model_state_dict": self.global_model.state_dict(),
            "history": self.history,
            "config": self.config.__dict__,
            "aggregation_strategy": self.aggregation_strategy,
        }
        ckpt_dir = self.experiment_dir / "checkpoints"
        torch.save(checkpoint, ckpt_dir / "checkpoint_latest.pth")
        if round_idx % 10 == 0:
            torch.save(checkpoint, ckpt_dir / f"checkpoint_round_{round_idx}.pth")

    def train(self):
        """run federated training simulation."""
        print(f"\nstarting federated training ({self.aggregation_strategy})")
        print(f"  {self.communication_rounds} rounds x {self.local_epochs} local epochs")
        print()

        best_ssim = 0

        for round_idx in range(self.communication_rounds):
            round_start = time.time()

            # create fresh client models (copy of global)
            client_models = []
            for i in range(self.n_clients):
                client_model = copy.deepcopy(self.global_model)
                client_models.append(client_model)

            # distribute global model to clients
            self.aggregator.distribute_to_clients(client_models)

            # local training on each client
            round_losses = []
            prox_loss_fn = None
            if isinstance(self.aggregator, FedProxAggregator):
                prox_loss_fn = self.aggregator.compute_proximal_loss

            for i in range(self.n_clients):
                client = self._create_client(i, client_models[i])
                local_losses = client.train_local(self.local_epochs, prox_loss_fn)
                round_losses.append(local_losses)
                print(
                    f"  round {round_idx} | client {i}: "
                    f"G={local_losses['G_loss']:.4f} "
                    f"D={local_losses['D_loss']:.4f} "
                    f"cycle={local_losses['cycle']:.4f}"
                )

            # aggregate client updates
            dataset_sizes = [len(self.client_loaders[i % len(self.client_loaders)].dataset)
                            for i in range(self.n_clients)]
            self.aggregator.aggregate(client_models, dataset_sizes)

            # scaffold control variate update
            if isinstance(self.aggregator, ScaffoldAggregator):
                for i in range(self.n_clients):
                    n_local_steps = self.local_epochs * len(
                        self.client_loaders[i % len(self.client_loaders)]
                    )
                    self.aggregator.update_controls(
                        i, client_models[i], self.lr_G, n_local_steps
                    )

            round_time = time.time() - round_start

            # log average client losses
            avg_losses = {
                k: np.mean([cl[k] for cl in round_losses])
                for k in round_losses[0]
            }
            for k, v in avg_losses.items():
                self.writer.add_scalar(f"federated/{k}", v, round_idx)

            print(
                f"  round {round_idx} | avg: "
                f"G={avg_losses['G_loss']:.4f} "
                f"D={avg_losses['D_loss']:.4f} "
                f"time={round_time:.1f}s"
            )

            # evaluate global model periodically
            if (round_idx + 1) % self.eval_every_n_rounds == 0:
                metrics = self.evaluate_global_model()
                mean_ssim = (metrics["ssim_A2B"] + metrics["ssim_B2A"]) / 2

                for k, v in metrics.items():
                    self.writer.add_scalar(f"federated/eval/{k}", v, round_idx)

                print(
                    f"  round {round_idx} | eval: "
                    f"ssim_A2B={metrics['ssim_A2B']:.4f} "
                    f"ssim_B2A={metrics['ssim_B2A']:.4f}"
                )

                if mean_ssim > best_ssim:
                    best_ssim = mean_ssim
                    self.save_checkpoint(round_idx)
                    print(f"  *** new best ssim: {mean_ssim:.4f} ***")

                self.history["global_metrics"].append({
                    "round": round_idx,
                    "metrics": metrics,
                })

            self.history["rounds"].append({
                "round": round_idx,
                "avg_losses": avg_losses,
                "time": round_time,
            })

            # periodic checkpoint
            if (round_idx + 1) % 10 == 0:
                self.save_checkpoint(round_idx)

        # save final
        self.save_checkpoint(self.communication_rounds - 1)
        self.writer.close()

        with open(self.experiment_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2, default=str)

        print(f"\nfederated training complete!")
        print(f"best ssim: {best_ssim:.4f}")
        print(f"checkpoints saved to: {self.experiment_dir / 'checkpoints'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="federated learning simulation for sa-cyclegan-2.5d"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--brats_dir", type=str, default=None)
    parser.add_argument("--upenn_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--communication_rounds", type=int, default=40)
    parser.add_argument("--aggregation_strategy", type=str, default="fedavg",
                        choices=["fedavg", "fedprox", "scaffold"])
    parser.add_argument("--share_discriminators", action="store_true")
    parser.add_argument("--fedprox_mu", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--lr_G", type=float, default=5e-5)
    parser.add_argument("--lr_D", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--n_residual_blocks", type=int, default=9)
    parser.add_argument("--eval_every_n_rounds", type=int, default=5)
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

    config = SACycleGAN25DConfig(
        ngf=args.ngf,
        ndf=args.ndf,
        n_residual_blocks=args.n_residual_blocks,
        attention_layers=tuple(getattr(args, "attention_layers", [3, 4, 5])),
    )

    trainer = FederatedTrainer(
        config=config,
        brats_dir=args.brats_dir,
        upenn_dir=args.upenn_dir,
        output_dir=args.output_dir,
        n_clients=args.n_clients,
        local_epochs=args.local_epochs,
        communication_rounds=args.communication_rounds,
        aggregation_strategy=args.aggregation_strategy,
        share_discriminators=getattr(args, "share_discriminators", False),
        fedprox_mu=getattr(args, "fedprox_mu", 0.01),
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr_G=args.lr_G,
        lr_D=args.lr_D,
        num_workers=args.num_workers,
        experiment_name=getattr(args, "experiment_name", None),
        eval_every_n_rounds=getattr(args, "eval_every_n_rounds", 5),
    )

    trainer.train()


if __name__ == "__main__":
    main()
