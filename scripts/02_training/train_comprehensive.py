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
    
    # Suppress unhelpful warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    

def print_banner():
    """Print training banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ███████╗ ██████╗██████╗ ███████╗    ║
║     ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔════╝██╔════╝██╔══██╗██╔════╝    ║
║     ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║███████╗██║     ██████╔╝█████╗      ║
║     ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║╚════██║██║     ██╔═══╝ ██╔══╝      ║
║     ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝███████║╚██████╗██║     ███████╗    ║
║     ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚═════╝╚═╝     ╚══════╝    ║
║                                                                              ║
║              2.5D SA-CycleGAN MRI Harmonization Training                     ║
║                    Professional Research Pipeline                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


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
        print(f"📄 Loaded config from: {args.config}")
    elif args.debug:
        config = create_debug_config()
        print("🐛 Debug mode enabled")
    else:
        config = TrainingConfig()
        print("⚙️  Using default configuration")
    
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
        
    return config


def verify_data_paths(config):
    """Verify data directories exist and contain valid MRI data."""
    brats_path = Path(config.brats_dir)
    upenn_path = Path(config.upenn_dir)
    
    print("\n📂 Verifying data paths...")
    
    if not brats_path.exists():
        print(f"   ❌ BraTS directory not found: {brats_path}")
        return False
    else:
        # Count subject folders and NIfTI files
        brats_subjects = [d for d in brats_path.iterdir() if d.is_dir()]
        brats_nifti = list(brats_path.glob("**/*.nii.gz"))
        print(f"   ✅ BraTS: {len(brats_subjects)} subjects, {len(brats_nifti)} NIfTI files")
        if len(brats_subjects) == 0:
            print(f"   ⚠️  Warning: No subject folders found in BraTS directory")
        
    if not upenn_path.exists():
        print(f"   ❌ UPenn directory not found: {upenn_path}")
        return False
    else:
        # Count subject folders and NIfTI files
        upenn_subjects = [d for d in upenn_path.iterdir() if d.is_dir()]
        upenn_nifti = list(upenn_path.glob("**/*.nii.gz"))
        print(f"   ✅ UPenn: {len(upenn_subjects)} subjects, {len(upenn_nifti)} NIfTI files")
        if len(upenn_subjects) == 0:
            print(f"   ⚠️  Warning: No subject folders found in UPenn directory")
        
    return True


def print_config_summary(config):
    """Print configuration summary."""
    print("\n" + "="*60)
    print("📋 CONFIGURATION SUMMARY")
    print("="*60)
    
    print(f"\n🔬 Experiment: {config.experiment_name}")
    print(f"🎲 Seed: {config.seed}")
    
    print("\n📊 Model:")
    print(f"   • Generator filters: {config.ngf}")
    print(f"   • Discriminator filters: {config.ndf}")
    print(f"   • Residual blocks: {config.n_residual_blocks}")
    print(f"   • Self-attention: {config.use_attention}")
    print(f"   • CBAM: {config.use_cbam}")
    print(f"   • Input channels: {config.input_channels} (3 slices × 4 modalities)")
    print(f"   • Output channels: {config.output_channels}")
    
    print("\n🏋️ Training:")
    print(f"   • Epochs: {config.epochs}")
    print(f"   • Batch size: {config.batch_size}")
    print(f"   • Image size: {config.image_size}×{config.image_size}")
    print(f"   • Learning rate G: {config.lr_G}")
    print(f"   • Learning rate D: {config.lr_D}")
    print(f"   • Scheduler: {config.scheduler_type}")
    print(f"   • Warmup epochs: {config.warmup_epochs}")
    
    print("\n⚖️ Loss weights:")
    print(f"   • λ_cycle: {config.lambda_cycle}")
    print(f"   • λ_identity: {config.lambda_identity}")
    print(f"   • λ_ssim: {config.lambda_ssim}")
    print(f"   • λ_gradient: {config.lambda_gradient}")
    
    print("\n📁 Output: {config.output_dir}")
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
    print(f"🖥️  Device: {device_name}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    # Create config
    config = create_config_from_args(args)
    
    # Verify paths
    if not verify_data_paths(config):
        print("\n❌ Data verification failed. Exiting.")
        sys.exit(1)
        
    # Print summary
    print_config_summary(config)
    
    # Confirm
    if not args.debug:
        response = input("🚀 Start training? [Y/n]: ").strip().lower()
        if response and response != 'y':
            print("Training cancelled.")
            sys.exit(0)
    
    # Import trainer (after all checks to fail fast)
    from neuroscope.training.trainers.comprehensive_trainer import ComprehensiveTrainer
    
    # Create trainer
    print("\n🔧 Initializing trainer...")
    trainer = ComprehensiveTrainer(config)
    
    # Model summary
    print(f"\n📐 Model parameters: {trainer.total_params:,}")
    print(f"   Trainable: {trainer.trainable_params:,}")
    
    # Data summary
    print(f"\n📊 Dataset splits:")
    print(f"   Train: {trainer.train_samples:,} samples")
    print(f"   Valid: {trainer.val_samples:,} samples")
    print(f"   Test: {trainer.test_samples:,} samples")
    
    # Start training
    print("\n" + "="*60)
    print("🏁 STARTING TRAINING")
    print("="*60 + "\n")
    
    try:
        final_metrics = trainer.train()
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE")
        print("="*60)
        print(f"\n📊 Final Results:")
        for k, v in final_metrics.items():
            print(f"   • {k}: {v:.4f}")
        print(f"\n📁 Results saved to: {trainer.run_dir}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        trainer.save_checkpoint()
        print(f"💾 Checkpoint saved to: {trainer.checkpoints_dir}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
