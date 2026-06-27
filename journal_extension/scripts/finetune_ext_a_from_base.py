#!/usr/bin/env python3
"""
warm-start fine-tune launcher for extension a (patchnce hybrid).

the original patchnce lambda-sweep checkpoints were lost when the chameleon
lease lapsed. this launcher reconstructs a usable hybrid model FAST by
warm-starting the generators/discriminators from the surviving base
sa-cyclegan-2.5d checkpoint (final_trained_model/checkpoints/checkpoint_best.pth,
epoch 84) and fine-tuning with the added multi-layer patchnce loss -- ~6h for
~30 epochs instead of ~13h from scratch.

two arms for the sashimi 2026 mechanism study (same warm-start, matched budget):
  - hybrid:     --lambda_nce 0.5            (cycle + patchnce)
  - cycle-only: --lambda_nce 0.0           (continued cycle training, the control)
  - cut-only:   --lambda_nce 0.5 --no_cycle (set cycle weight ~0 in config; ablation)

the warm-start loads ONLY the generator/discriminator weights (strict=False),
leaving the patchnce mlp heads and optimizers fresh. it unwraps torch.compile
(_orig_mod) and dataparallel (.module) so the base (unwrapped) keys land on the
raw submodules.

SMOKE TEST ON THE NODE FIRST (1 step, tiny subset) before the full run:
    python finetune_ext_a_from_base.py ... --epochs 1 --smoke_test

usage (hybrid arm):
    python finetune_ext_a_from_base.py \\
        --base_checkpoint /data/neuroscope/final_trained_model/checkpoints/checkpoint_best.pth \\
        --brats_dir /data/preprocessed/brats --upenn_dir /data/preprocessed/upenn \\
        --output_dir /data/experiments/sashimi/ext_a \\
        --experiment_name ext_a_hybrid_lambda0.5 --lambda_nce 0.5 --epochs 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from journal_extension.scripts.train_hybrid_nce import HybridNCETrainer
from neuroscope.models.architectures.sa_cyclegan_25d import SACycleGAN25DConfig


def _unwrap(module: nn.Module) -> nn.Module:
    """strip torch.compile (_orig_mod) and dataparallel (.module) wrappers."""
    target = module
    for _ in range(4):
        if isinstance(target, nn.DataParallel):
            target = cast("nn.Module", target.module)
        elif hasattr(target, "_orig_mod"):
            target = cast("nn.Module", target._orig_mod)
        else:
            break
    return target


def warm_start_from_base(trainer: HybridNCETrainer, base_ckpt_path: str) -> None:
    """load only G/D weights from a base sa-cyclegan checkpoint into the hybrid model.

    patchnce heads and optimizers stay fresh; training starts at epoch 0 with a
    converged backbone. base keys are unwrapped (no _orig_mod./module. prefixes),
    so we load them onto the unwrapped raw submodules.
    """
    print(f"warm-starting from base checkpoint: {base_ckpt_path}")
    ck = torch.load(base_ckpt_path, map_location=trainer.device, weights_only=False)
    sd = ck.get("model_state_dict", ck)

    # group flat state_dict by top-level submodule (G_A2B / G_B2A / D_A / D_B)
    by_mod: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        # also strip any stray prefixes present in the base checkpoint itself
        kk = k
        for p in ("_orig_mod.", "module."):
            kk = kk.replace(p, "")
        top, _, rest = kk.partition(".")
        if rest:
            by_mod.setdefault(top, {})[rest] = v

    report = {}
    for name in ("G_A2B", "G_B2A", "D_A", "D_B"):
        if name not in by_mod or not hasattr(trainer.model, name):
            report[name] = "absent in base or model"
            continue
        target = _unwrap(getattr(trainer.model, name))
        missing, unexpected = target.load_state_dict(by_mod[name], strict=False)
        report[name] = (
            f"loaded={len(by_mod[name]) - len(unexpected)} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    print("warm-start summary:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    base_epoch = ck.get("epoch", "?")
    print(f"base checkpoint epoch was {base_epoch}; fine-tune starts at epoch 0.")


def parse_args():
    p = argparse.ArgumentParser(description="warm-start fine-tune ext a from base checkpoint")
    p.add_argument(
        "--base_checkpoint",
        default=None,
        help="base sa-cyclegan checkpoint to warm-start from (fresh runs)",
    )
    p.add_argument(
        "--resume",
        default=None,
        help="checkpoint to resume an interrupted run (restores model/opt/sched/scaler/epoch)",
    )
    p.add_argument("--brats_dir", required=True)
    p.add_argument("--upenn_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--experiment_name", required=True)
    p.add_argument(
        "--lambda_nce", type=float, default=0.5, help="0.5 = hybrid arm; 0.0 = cycle-only control"
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument(
        "--lr_G", type=float, default=2e-5, help="lower lr for fine-tuning a converged backbone"
    )
    p.add_argument("--lr_D", type=float, default=2e-5)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nce_temperature", type=float, default=0.07)
    p.add_argument("--nce_num_patches", type=int, default=256)
    p.add_argument("--n_residual_blocks", type=int, default=9)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument(
        "--smoke_test",
        action="store_true",
        help="construct, warm-start, run a single train step, and exit",
    )
    return p.parse_args()


def main():
    args = parse_args()

    config = SACycleGAN25DConfig(
        ngf=args.ngf,
        ndf=args.ndf,
        n_residual_blocks=args.n_residual_blocks,
        attention_layers=(3, 4, 5),
        nce_feature_layers=(2, 5),
    )

    trainer = HybridNCETrainer(
        config=config,
        brats_dir=args.brats_dir,
        upenn_dir=args.upenn_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr_G=args.lr_G,
        lr_D=args.lr_D,
        num_workers=args.num_workers,
        experiment_name=args.experiment_name,
        lambda_nce=args.lambda_nce,
        nce_num_patches=args.nce_num_patches,
        nce_temperature=args.nce_temperature,
        warmup_epochs=args.warmup_epochs,
        epochs=args.epochs,
        seed=args.seed,
    )

    if args.resume:
        # disaster-recovery / preemption: continue an interrupted arm with full state
        trainer.load_checkpoint(args.resume)
    elif args.base_checkpoint:
        warm_start_from_base(trainer, args.base_checkpoint)
    else:
        raise SystemExit("provide --base_checkpoint (fresh warm-start) or --resume (continue)")

    if args.smoke_test:
        print("smoke test: running a single train step...")
        batch = next(iter(trainer.train_loader))
        losses = trainer.train_step(batch)
        print("smoke step losses:", {k: round(v, 4) for k, v in losses.items()})
        print("smoke test OK -- warm-start + one step succeeded. exiting.")
        return

    trainer.train(epochs=args.epochs, validate_every=1, save_every=5)


if __name__ == "__main__":
    main()
