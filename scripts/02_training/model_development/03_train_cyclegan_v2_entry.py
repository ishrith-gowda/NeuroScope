#!/usr/bin/env python
"""
cyclegan v2 training entry point

this script launches the improved cyclegan training with:
- anti-mode-collapse techniques (replay buffer, spectral norm)
- two-timescale update rule (ttur)
- gradient penalty regularization
- feature matching loss
- label smoothing
- instance noise injection with decay
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# add paths
HERE = Path(__file__).resolve().parent
PREP_DIR = HERE.parent / '01_data_preparation_pipeline'
if str(PREP_DIR) not in sys.path:
    sys.path.insert(0, str(PREP_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import neuroscope_preprocessing_config as npc
PATHS = npc.PATHS


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, 
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args():
    ap = argparse.ArgumentParser(description='CycleGAN v2 Training (Enhanced)')
    ap.add_argument('--data_root', type=str, default=str(PATHS['preprocessed_dir']))
    ap.add_argument('--meta_json', type=str, default=str(PATHS['metadata_splits']))
    ap.add_argument('--n_epochs', type=int, default=150)  # more epochs for better convergence
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)  # lower lr for stability
    ap.add_argument('--decay_epoch', type=int, default=75)
    ap.add_argument('--lambda_cycle', type=float, default=10.0)
    ap.add_argument('--lambda_identity', type=float, default=5.0)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--log_interval', type=int, default=50)
    ap.add_argument('--sample_interval', type=int, default=200)
    ap.add_argument('--checkpoint_interval', type=int, default=10)
    ap.add_argument('--checkpoint_dir', type=str, default=str(PATHS['checkpoints_dir']))
    ap.add_argument('--sample_dir', type=str, default=str(PATHS['samples_dir']))
    ap.add_argument('--run_dir', type=str, default=str(PATHS['logs_dir']))
    ap.add_argument('--slices_per_subject', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def main():
    setup_logging()
    args = parse_args()
    
    print("\n" + "=" * 80)
    print("cyclegan v2 training entry point")
    print("=" * 80)
    print(f"python: {sys.executable}")
    print(f"working directory: {os.getcwd()}")
    
    print("\nconfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    try:
        import torch
        print(f"pytorch version: {torch.__version__}")
        
        device = torch.device(
            'cuda' if torch.cuda.is_available() 
            else 'mps' if torch.backends.mps.is_available() 
            else 'cpu'
        )
        print(f"using device: {device}")
        
        # import and run training
        from train_cyclegan_v2 import train
        
        class ArgsHolder:
            pass
        
        args_obj = ArgsHolder()
        for key, value in vars(args).items():
            setattr(args_obj, key, value)
        
        print("\nstarting v2 training...\n")
        train(args_obj, device)
        print("\ntraining completed successfully\n")
        
    except Exception as e:
        import traceback
        print(f"\ntraining failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
