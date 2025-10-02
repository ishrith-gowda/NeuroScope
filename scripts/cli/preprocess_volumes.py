#!/usr/bin/env python
"""
Volume preprocessing script using the VolumePreprocessor module.

This script applies preprocessing steps to MRI volumes.
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

from neuroscope.preprocessing import VolumePreprocessor
from neuroscope.core.logging import setup_logging, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess MRI volumes.')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing MRI volumes')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for preprocessed volumes')
    parser.add_argument('--mask-dir', type=str,
                        help='Optional directory containing masks')
    parser.add_argument('--file-pattern', type=str, default='*.nii.gz',
                        help='Glob pattern for input files')
    parser.add_argument('--normalize', choices=['minmax', 'zscore', 'percentile', 'histequal', 'whitestripe'],
                        default='percentile', help='Normalization method')
    parser.add_argument('--lower-pct', type=float, default=1.0,
                        help='Lower percentile for percentile normalization')
    parser.add_argument('--upper-pct', type=float, default=99.0,
                        help='Upper percentile for percentile normalization')
    parser.add_argument('--target-range', type=float, nargs=2, default=[0, 1],
                        help='Target intensity range [min, max]')
    parser.add_argument('--crop', type=int, nargs=3,
                        help='Crop size as three integers [x, y, z]')
    parser.add_argument('--crop-method', choices=['center', 'random'],
                        default='center', help='Cropping method')
    parser.add_argument('--rescale', type=float, nargs='+',
                        help='Scale factor for rescaling (single value or [x, y, z])')
    parser.add_argument('--target-shape', type=int, nargs=3,
                        help='Target shape for rescaling [x, y, z]')
    parser.add_argument('--output-json', type=str,
                        help='Output JSON file for preprocessing metadata')
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
    
    # Initialize preprocessor
    preprocessor = VolumePreprocessor()
    
    # Add normalization step
    if args.normalize == 'minmax':
        preprocessor.add_step('min_max_normalization', {
            'target_range': tuple(args.target_range)
        })
    elif args.normalize == 'zscore':
        preprocessor.add_step('z_score_normalization', {})
    elif args.normalize == 'percentile':
        preprocessor.add_step('percentile_normalization', {
            'low_percentile': args.lower_pct,
            'high_percentile': args.upper_pct,
            'target_range': tuple(args.target_range)
        })
    elif args.normalize == 'histequal':
        preprocessor.add_step('histogram_equalization', {
            'num_bins': 256
        })
    elif args.normalize == 'whitestripe':
        preprocessor.add_step('white_stripe_normalization', {
            'target_value': 1.0
        })
    
    # Add crop step if requested
    if args.crop:
        preprocessor.add_step('crop', {
            'crop_size': tuple(args.crop),
            'method': args.crop_method
        })
    
    # Add rescale step if requested
    if args.rescale:
        preprocessor.add_step('rescale', {
            'scale_factor': args.rescale if len(args.rescale) > 1 else args.rescale[0],
            'order': 1  # Linear interpolation
        })
    elif args.target_shape:
        preprocessor.add_step('rescale', {
            'target_shape': tuple(args.target_shape),
            'order': 1  # Linear interpolation
        })
    
    # Process volumes
    results = preprocessor.batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_pattern=args.file_pattern,
        mask_dir=args.mask_dir
    )
    
    # Save metadata to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Metadata saved to {args.output_json}")
    
    logger.info("Preprocessing complete")
    logger.info(f"Processed {len(results)} volumes")


if __name__ == '__main__':
    main()