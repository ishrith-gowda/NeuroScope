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
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--decay_epoch', type=int, default=50)
    ap.add_argument('--lambda_cycle', type=float, default=10.0)
    ap.add_argument('--lambda_identity', type=float, default=5.0)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--log_interval', type=int, default=50)
    ap.add_argument('--sample_interval', type=int, default=500)
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
    # defer import to avoid heavy deps at module import time
    from train_cyclegan import train
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logging.info('Using device: %s', device)
    class A:  # assemble a simple args object compatible with train()
        pass
    A.__dict__.update(vars(args))
    train(A, device)


if __name__ == '__main__':
    main()
