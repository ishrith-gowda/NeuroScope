#!/usr/bin/env python
"""
Dataset preparation script for paired MRI data.

This script creates train/val/test splits for paired MRI data.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys

# Add the project root to the Python path if not already there
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from neuroscope.preprocessing import create_paired_dataset_splits
from neuroscope.core.logging import setup_logging, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create dataset splits for paired MRI data.')
    parser.add_argument('--domain-a-dir', type=str, required=True,
                        help='Directory with domain A volumes')
    parser.add_argument('--domain-b-dir', type=str, required=True,
                        help='Directory with domain B volumes')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for split files')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Fraction of data for training')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Fraction of data for validation')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Fraction of data for testing')
    parser.add_argument('--file-pattern', type=str, default='*.nii.gz',
                        help='Glob pattern for input files')
    parser.add_argument('--paired', action='store_true', default=True,
                        help='Whether the data is paired (same filenames in both domains)')
    parser.add_argument('--unpaired', action='store_false', dest='paired',
                        help='If provided, treat data as unpaired')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level=log_level)
    logger = get_logger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset splits
    splits = create_paired_dataset_splits(
        domain_a_dir=args.domain_a_dir,
        domain_b_dir=args.domain_b_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        file_pattern=args.file_pattern,
        paired=args.paired,
        seed=args.seed
    )
    
    logger.info("Dataset splits created successfully")
    logger.info(f"Splits saved to {os.path.join(args.output_dir, 'dataset_splits.json')}")


if __name__ == '__main__':
    main()