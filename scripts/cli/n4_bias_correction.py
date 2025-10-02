#!/usr/bin/env python
"""
Bias field correction script using the N4BiasFieldCorrection module.

This script applies N4 bias field correction to MRI volumes.
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

from neuroscope.preprocessing import N4BiasFieldCorrection
from neuroscope.core.logging import setup_logging, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Apply N4 bias field correction to MRI volumes.')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing MRI volumes')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for corrected volumes')
    parser.add_argument('--mask-dir', type=str,
                        help='Optional directory containing masks')
    parser.add_argument('--file-pattern', type=str, default='*.nii.gz',
                        help='Glob pattern for input files')
    parser.add_argument('--save-bias', action='store_true',
                        help='Save estimated bias fields')
    parser.add_argument('--shrink-factor', type=int, default=4,
                        help='Shrink factor for downsampling')
    parser.add_argument('--iterations', type=int, nargs='+', default=[50, 50, 30, 20],
                        help='Number of iterations at each resolution level')
    parser.add_argument('--convergence-threshold', type=float, default=0.001,
                        help='Convergence threshold')
    parser.add_argument('--spline-order', type=int, default=3,
                        help='Order of B-spline used in the approximation')
    parser.add_argument('--spline-distance', type=float, default=200.0,
                        help='Distance between B-spline control points')
    parser.add_argument('--output-json', type=str,
                        help='Output JSON file for correction metrics')
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
    
    # Initialize N4 correction
    n4_correction = N4BiasFieldCorrection(
        shrink_factor=args.shrink_factor,
        iterations=args.iterations,
        convergence_threshold=args.convergence_threshold,
        spline_order=args.spline_order,
        spline_distance=args.spline_distance,
        save_bias_field=args.save_bias
    )
    
    # Process volumes
    results = n4_correction.batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_pattern=args.file_pattern,
        mask_dir=args.mask_dir,
        save_bias=args.save_bias
    )
    
    # Save metrics to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Metrics saved to {args.output_json}")
    
    logger.info("N4 bias field correction complete")
    
    # Compute summary statistics
    if results:
        cv_improvements = [res['cv_improvement_percent'] for res in results.values()]
        avg_improvement = sum(cv_improvements) / len(cv_improvements)
        logger.info(f"Average CV improvement: {avg_improvement:.2f}%")


if __name__ == '__main__':
    main()