#!/usr/bin/env python3
"""
Comprehensive Training Launch Script for 2.5D SA-CycleGAN.

This is the main entry point for launching professional-grade training
with the full logging, sampling, and figure generation infrastructure.

Usage:
    # Using YAML config
    python train_comprehensive.py --config ../../../neuroscope/config/experiments/train_sa_cyclegan_25d.yaml
    
    # Quick debug run
    python train_comprehensive.py --debug
    
    # Command line overrides
    python train_comprehensive.py --epochs 50 --batch_size 2 --lr 1e-4
    
    # Resume training
    python train_comprehensive.py --resume /path/to/checkpoint.pth

Author: NeuroScope Research Team
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import random


def set_environment():
    """Configure environment for optimal training."""
    # Reduce memory fragmentation
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

    # For reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Fix macOS malloc stack logging warnings
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    os.environ['MallocStackLogging'] = '0'

    # Suppress unhelpful warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    

def print_banner():
    """Print simple training header."""
    print("\n" + "="*60)
    print("neuroscope - 2.5d sa-cyclegan training")
    print("="*60)


def get_device_info():
    """Get device information."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return device, f"{name} ({memory:.1f}GB)"
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        return device, "Apple Silicon MPS"
    else:
        return torch.device('cpu'), "CPU"


def create_debug_config():
    """Create minimal debug configuration."""
    from neuroscope.training.trainers.comprehensive_trainer import TrainingConfig
    
    return TrainingConfig(
        experiment_name="debug_run",
        epochs=2,
        batch_size=2,
        validate_every=1,
        save_every=1,
        sample_every=1,
        figure_every=1,
        log_every_n_steps=1,
        verbose=3,
        early_stopping=False
    )


def create_config_from_args(args) -> 'TrainingConfig':
    """Create configuration from command line arguments."""
    from neuroscope.training.trainers.comprehensive_trainer import TrainingConfig
    
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
        print(f"loaded config from: {args.config}")
    elif args.debug:
        config = create_debug_config()
        print("debug mode enabled")
    else:
        config = TrainingConfig()
        print("using default configuration")
    
    # Command line overrides
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr_G = args.lr
        config.lr_D = args.lr
    if args.resume is not None:
        config.resume_from = args.resume
    if args.seed is not None:
        config.seed = args.seed
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    # macOS-specific fix: reduce num_workers to avoid fork issues
    import platform
    if platform.system() == 'Darwin':  # macOS
        if hasattr(config, 'num_workers') and config.num_workers > 2:
            print(f"macos detected: reducing num_workers from {config.num_workers} to 2")
            config.num_workers = 2

    return config


def verify_data_paths(config):
    """Verify data directories exist and contain valid MRI data."""
    brats_path = Path(config.brats_dir)
    upenn_path = Path(config.upenn_dir)

    print("\nverifying data paths...")

    if not brats_path.exists():
        print(f"   brats directory not found: {brats_path}")
        return False
    else:
        # Count subject folders and NIfTI files
        brats_subjects = [d for d in brats_path.iterdir() if d.is_dir()]
        brats_nifti = list(brats_path.glob("**/*.nii.gz"))
        print(f"   brats: {len(brats_subjects)} subjects, {len(brats_nifti)} nifti files")
        if len(brats_subjects) == 0:
            print(f"   warning: no subject folders found in brats directory")

    if not upenn_path.exists():
        print(f"   upenn directory not found: {upenn_path}")
        return False
    else:
        # Count subject folders and NIfTI files
        upenn_subjects = [d for d in upenn_path.iterdir() if d.is_dir()]
        upenn_nifti = list(upenn_path.glob("**/*.nii.gz"))
        print(f"   upenn: {len(upenn_subjects)} subjects, {len(upenn_nifti)} nifti files")
        if len(upenn_subjects) == 0:
            print(f"   warning: no subject folders found in upenn directory")

    return True


def print_config_summary(config):
    """Print configuration summary."""
    print("\n" + "="*60)
    print("configuration summary")
    print("="*60)

    print(f"\nexperiment: {config.experiment_name}")
    print(f"seed: {config.seed}")

    print("\nmodel:")
    print(f"   generator filters: {config.ngf}")
    print(f"   discriminator filters: {config.ndf}")
    print(f"   residual blocks: {config.n_residual_blocks}")
    print(f"   self-attention: {config.use_attention}")
    print(f"   cbam: {config.use_cbam}")
    print(f"   input channels: {config.input_channels} (3 slices x 4 modalities)")
    print(f"   output channels: {config.output_channels}")

    print("\ntraining:")
    print(f"   epochs: {config.epochs}")
    print(f"   batch size: {config.batch_size}")
    print(f"   image size: {config.image_size}x{config.image_size}")
    print(f"   learning rate g: {config.lr_G}")
    print(f"   learning rate d: {config.lr_D}")
    print(f"   scheduler: {config.scheduler_type}")
    print(f"   warmup epochs: {config.warmup_epochs}")

    print("\nloss weights:")
    print(f"   lambda_cycle: {config.lambda_cycle}")
    print(f"   lambda_identity: {config.lambda_identity}")
    print(f"   lambda_ssim: {config.lambda_ssim}")
    print(f"   lambda_gradient: {config.lambda_gradient}")

    print(f"\noutput: {config.output_dir}")
    print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train 2.5D SA-CycleGAN with comprehensive logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with YAML config
    python train_comprehensive.py --config path/to/config.yaml
    
    # Quick debug run
    python train_comprehensive.py --debug
    
    # Override settings
    python train_comprehensive.py --epochs 50 --batch_size 8 --lr 1e-4
    
    # Resume training
    python train_comprehensive.py --resume experiments/run_001/checkpoints/latest.pth
        """
    )
    
    # Config
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    parser.add_argument('--debug', action='store_true',
                       help='Run quick debug training (2 epochs)')
    
    # Overrides
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate for G and D')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Setup
    set_environment()
    print_banner()

    # Device info
    device, device_name = get_device_info()
    print(f"device: {device_name}")
    print(f"python: {sys.version.split()[0]}")
    print(f"pytorch: {torch.__version__}")

    # Create config
    config = create_config_from_args(args)

    # Verify paths
    if not verify_data_paths(config):
        print("\ndata verification failed. exiting.")
        sys.exit(1)

    # Print summary
    print_config_summary(config)

    # Confirm
    if not args.debug:
        response = input("start training? [y/n]: ").strip().lower()
        if response and response != 'y':
            print("training cancelled.")
            sys.exit(0)

    # Import trainer (after all checks to fail fast)
    from neuroscope.training.trainers.comprehensive_trainer import ComprehensiveTrainer

    # Create trainer
    print("\ninitializing trainer...")
    trainer = ComprehensiveTrainer(config)

    # Model summary
    print(f"\nmodel parameters: {trainer.total_params:,}")
    print(f"   trainable: {trainer.trainable_params:,}")

    # Data summary
    print(f"\ndataset splits:")
    print(f"   train: {trainer.train_samples:,} samples")
    print(f"   valid: {trainer.val_samples:,} samples")
    print(f"   test: {trainer.test_samples:,} samples")

    # Start training
    print("\n" + "="*60)
    print("starting training")
    print("="*60 + "\n")

    try:
        final_metrics = trainer.train()

        print("\n" + "="*60)
        print("training complete")
        print("="*60)
        print(f"\nfinal results:")
        for k, v in final_metrics.items():
            print(f"   {k}: {v:.4f}")
        print(f"\nresults saved to: {trainer.run_dir}")

    except KeyboardInterrupt:
        print("\n\ntraining interrupted by user")
        trainer.save_checkpoint()
        print(f"checkpoint saved to: {trainer.checkpoints_dir}")

    except Exception as e:
        print(f"\ntraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        trainer.close()


if __name__ == '__main__':
    main()
