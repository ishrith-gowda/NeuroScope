import argparse
import logging
from pathlib import Path
import sys

from neuroscope_dataset_loader import get_cycle_domain_loaders


def setup_logging(verbose: bool):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='[%(levelname)s] %(message)s')


def parse_args():
    ap = argparse.ArgumentParser(description='Quick dataloader smoke test for CycleGAN domains')
    ap.add_argument('--preprocessed_dir', type=str, required=True)
    ap.add_argument('--metadata_json', type=str, required=True)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--slices_per_subject', type=int, default=4)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--verbose', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    loaders = get_cycle_domain_loaders(
        preprocessed_dir=args.preprocessed_dir,
        metadata_json=args.metadata_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        slices_per_subject=args.slices_per_subject,
        seed=args.seed,
    )

    required = ['train_A', 'train_B']
    for k in required:
        if k not in loaders:
            logging.error('missing loader %s; ensure preprocessing produced data for both domains', k)
            sys.exit(1)

    for name, dl in loaders.items():
        batch = next(iter(dl))
        logging.info('%s -> batch %s in [%.3f, %.3f]', name, tuple(batch.shape), batch.min().item(), batch.max().item())

    logging.info('Smoke test OK.')


if __name__ == '__main__':
    main()
