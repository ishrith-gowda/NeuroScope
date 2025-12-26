import argparse
import logging
import os
import sys
from pathlib import Path

# Reuse PATH defaults via preprocessing config
HERE = Path(__file__).resolve().parent
PREP_DIR = HERE.parent / '01_data_preparation_pipeline'
if str(PREP_DIR) not in sys.path:
    sys.path.insert(0, str(PREP_DIR))
import neuroscope_preprocessing_config as npc  # type: ignore
PATHS = npc.PATHS


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def parse_args():
    ap = argparse.ArgumentParser(description='CycleGAN training entrypoint (wraps train_cyclegan.py)')
    ap.add_argument('--data_root', type=str, default=str(PATHS['preprocessed_dir']))
    ap.add_argument('--meta_json', type=str, default=str(PATHS['metadata_splits']))
    ap.add_argument('--n_epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=4)  # Reduced from 8 to 4 for better memory usage
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--decay_epoch', type=int, default=50)
    ap.add_argument('--lambda_cycle', type=float, default=10.0)
    ap.add_argument('--lambda_identity', type=float, default=5.0)
    ap.add_argument('--num_workers', type=int, default=0)  # No parallel workers to reduce memory usage
    ap.add_argument('--log_interval', type=int, default=10)  # More frequent logging
    ap.add_argument('--sample_interval', type=int, default=100)  # More frequent samples
    ap.add_argument('--checkpoint_interval', type=int, default=5)  # More frequent checkpoints
    ap.add_argument('--checkpoint_dir', type=str, default=str(PATHS['checkpoints_dir']))
    ap.add_argument('--sample_dir', type=str, default=str(PATHS['samples_dir']))
    ap.add_argument('--run_dir', type=str, default=str(PATHS['logs_dir']))
    ap.add_argument('--slices_per_subject', type=int, default=2)  # Reduced from 4 to 2 for memory
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def main():
    setup_logging()
    args = parse_args()
    
    print("\n" + "="*80)
    print("CYCLEGAN TRAINING ENTRY POINT")
    print("="*80)
    print(f"Python interpreter: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Show all parsed arguments for debugging
    print("\nTraining configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    try:
        print("Importing train_cyclegan...")
        # defer import to avoid heavy deps at module import time
        from train_cyclegan import train
        print("Successfully imported train_cyclegan")
        
        print("Importing torch...")
        import torch
        print(f"Torch version: {torch.__version__}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {device}")
        logging.info('Using device: %s', device)
        # Create an instance of a simple class to pass to train()
        class ArgsHolder:
            pass
        
        args_obj = ArgsHolder()
        print("Creating args object...")
        for key, value in vars(args).items():
            setattr(args_obj, key, value)
            print(f"  Set {key} = {value}")
        
        print("\nStarting training process...\n")
        train(args_obj, device)
        print("\nTraining completed successfully!\n")
    except Exception as e:
        import traceback
        print(f"\nERROR: Training failed with exception: {e}")
        print("\nStack trace:")
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
