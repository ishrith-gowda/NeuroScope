#!/usr/bin/env python
"""
Image registration script using the MRIRegistration module.

This script applies registration between MRI volumes.
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

from neuroscope.preprocessing import MRIRegistration
from neuroscope.core.logging import setup_logging, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Register MRI volumes.')
    parser.add_argument('--fixed-path', type=str, required=True,
                        help='Path to fixed (target) image or directory')
    parser.add_argument('--moving-path', type=str, required=True,
                        help='Path to moving (source) image or directory')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Output directory for registered images')
    parser.add_argument('--file-pattern', type=str, default='*.nii.gz',
                        help='Glob pattern for input files when directories are provided')
    parser.add_argument('--fixed-mask-path', type=str,
                        help='Optional path to fixed image mask or directory')
    parser.add_argument('--moving-mask-path', type=str,
                        help='Optional path to moving image mask or directory')
    parser.add_argument('--registration-type', choices=['rigid', 'affine', 'deformable'],
                        default='rigid', help='Type of registration')
    parser.add_argument('--metric', choices=['mutual_information', 'mean_squares', 'correlation'],
                        default='mutual_information', help='Similarity metric')
    parser.add_argument('--optimizer', choices=['gradient_descent', 'lbfgs'],
                        default='gradient_descent', help='Optimizer for registration')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate for optimizer')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Maximum number of iterations')
    parser.add_argument('--save-transforms', action='store_true',
                        help='Save transformation files')
    parser.add_argument('--output-json', type=str,
                        help='Output JSON file for registration metrics')
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
    os.makedirs(args.output_path, exist_ok=True)
    
    # Initialize registration
    registration = MRIRegistration(
        registration_type=args.registration_type,
        metric=args.metric,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        number_of_iterations=args.iterations,
        verbose=args.verbose
    )
    
    # Process volumes
    results = registration.batch_register(
        fixed_path=args.fixed_path,
        moving_path=args.moving_path,
        output_path=args.output_path,
        file_pattern=args.file_pattern,
        fixed_mask_path=args.fixed_mask_path,
        moving_mask_path=args.moving_mask_path,
        save_transforms=args.save_transforms
    )
    
    # Save metrics to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Metrics saved to {args.output_json}")
    
    logger.info("Registration complete")
    
    # Compute summary statistics
    if results:
        similarities = [res['final_similarity'] for res in results.values()]
        avg_similarity = sum(similarities) / len(similarities)
        logger.info(f"Average similarity: {avg_similarity:.3f}")


if __name__ == '__main__':
    main()