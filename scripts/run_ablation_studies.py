#!/usr/bin/env python3
"""
Ablation Studies Script for SA-CycleGAN.

Systematically evaluates the contribution of each component:
1. Full SA-CycleGAN (baseline)
2. Without self-attention
3. Without perceptual loss
4. Without tumor preservation loss
5. Without anatomical consistency loss
6. Without cycle consistency loss (control)
7. Different attention mechanisms (CBAM, Multi-Head, etc.)
8. Different numbers of residual blocks
9. Different discriminator architectures

This script automates ablation experiments to justify design choices.

Usage:
    python scripts/run_ablation_studies.py \
        --data-dir ./data/processed \
        --output-dir ./experiments/ablations \
        --epochs 100 \
        --batch-size 4
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AblationStudyOrchestrator:
    """Orchestrates ablation study experiments."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        epochs: int = 100,
        batch_size: int = 4,
        gpu: int = 0,
    ):
        """
        Initialize orchestrator.

        Args:
            data_dir: Directory with preprocessed data
            output_dir: Output directory for ablation experiments
            epochs: Number of training epochs (fewer than main experiment)
            batch_size: Batch size
            gpu: GPU ID to use
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = epochs
        self.batch_size = batch_size
        self.gpu = gpu

        self.scripts_dir = Path(__file__).parent

        # Ablation configurations
        self.ablations = self._define_ablations()

        # Results tracking
        self.results = {}
        self.start_time = None

    def _define_ablations(self) -> Dict:
        """Define all ablation configurations."""
        return {
            'full_model': {
                'description': 'Full SA-CycleGAN (all components)',
                'config': {
                    'use_attention': True,
                    'attention_type': 'self_attention',
                    'use_perceptual_loss': True,
                    'use_tumor_loss': True,
                    'use_anatomical_loss': True,
                    'lambda_cycle': 10.0,
                    'lambda_perceptual': 1.0,
                    'lambda_tumor': 5.0,
                    'lambda_anatomical': 1.0,
                },
            },
            'no_attention': {
                'description': 'Without self-attention',
                'config': {
                    'use_attention': False,
                    'use_perceptual_loss': True,
                    'use_tumor_loss': True,
                    'use_anatomical_loss': True,
                    'lambda_cycle': 10.0,
                    'lambda_perceptual': 1.0,
                    'lambda_tumor': 5.0,
                    'lambda_anatomical': 1.0,
                },
            },
            'no_perceptual': {
                'description': 'Without perceptual loss',
                'config': {
                    'use_attention': True,
                    'attention_type': 'self_attention',
                    'use_perceptual_loss': False,
                    'use_tumor_loss': True,
                    'use_anatomical_loss': True,
                    'lambda_cycle': 10.0,
                    'lambda_perceptual': 0.0,
                    'lambda_tumor': 5.0,
                    'lambda_anatomical': 1.0,
                },
            },
            'no_tumor_loss': {
                'description': 'Without tumor preservation loss',
                'config': {
                    'use_attention': True,
                    'attention_type': 'self_attention',
                    'use_perceptual_loss': True,
                    'use_tumor_loss': False,
                    'use_anatomical_loss': True,
                    'lambda_cycle': 10.0,
                    'lambda_perceptual': 1.0,
                    'lambda_tumor': 0.0,
                    'lambda_anatomical': 1.0,
                },
            },
            'no_anatomical_loss': {
                'description': 'Without anatomical consistency loss',
                'config': {
                    'use_attention': True,
                    'attention_type': 'self_attention',
                    'use_perceptual_loss': True,
                    'use_tumor_loss': True,
                    'use_anatomical_loss': False,
                    'lambda_cycle': 10.0,
                    'lambda_perceptual': 1.0,
                    'lambda_tumor': 5.0,
                    'lambda_anatomical': 0.0,
                },
            },
            'cbam_attention': {
                'description': 'Using CBAM instead of self-attention',
                'config': {
                    'use_attention': True,
                    'attention_type': 'cbam',
                    'use_perceptual_loss': True,
                    'use_tumor_loss': True,
                    'use_anatomical_loss': True,
                    'lambda_cycle': 10.0,
                    'lambda_perceptual': 1.0,
                    'lambda_tumor': 5.0,
                    'lambda_anatomical': 1.0,
                },
            },
            'multihead_attention': {
                'description': 'Using multi-head attention',
                'config': {
                    'use_attention': True,
                    'attention_type': 'multihead',
                    'use_perceptual_loss': True,
                    'use_tumor_loss': True,
                    'use_anatomical_loss': True,
                    'lambda_cycle': 10.0,
                    'lambda_perceptual': 1.0,
                    'lambda_tumor': 5.0,
                    'lambda_anatomical': 1.0,
                },
            },
            'fewer_blocks': {
                'description': 'Fewer residual blocks (6 instead of 9)',
                'config': {
                    'use_attention': True,
                    'attention_type': 'self_attention',
                    'use_perceptual_loss': True,
                    'use_tumor_loss': True,
                    'use_anatomical_loss': True,
                    'n_residual_blocks': 6,
                    'lambda_cycle': 10.0,
                    'lambda_perceptual': 1.0,
                    'lambda_tumor': 5.0,
                    'lambda_anatomical': 1.0,
                },
            },
            'more_blocks': {
                'description': 'More residual blocks (12 instead of 9)',
                'config': {
                    'use_attention': True,
                    'attention_type': 'self_attention',
                    'use_perceptual_loss': True,
                    'use_tumor_loss': True,
                    'use_anatomical_loss': True,
                    'n_residual_blocks': 12,
                    'lambda_cycle': 10.0,
                    'lambda_perceptual': 1.0,
                    'lambda_tumor': 5.0,
                    'lambda_anatomical': 1.0,
                },
            },
        }

    def run_ablation(self, name: str, config: Dict) -> Dict:
        """
        Run a single ablation experiment.

        Args:
            name: Name of the ablation
            config: Configuration dictionary

        Returns:
            Results dictionary
        """
        logger.info("\n" + "="*80)
        logger.info(f"Running ablation: {name}")
        logger.info(f"Description: {self.ablations[name]['description']}")
        logger.info("="*80 + "\n")

        output_dir = self.output_dir / name
        log_file = output_dir / 'training.log'

        # Build command
        # Note: This assumes we have a configurable training script
        # You may need to adapt this based on your actual training script
        cmd = [
            'python',
            str(self.scripts_dir / 'train_sa_cyclegan_complete.py'),
            '--dataset-a', str(self.data_dir / 'brats'),
            '--dataset-b', str(self.data_dir / 'upenn_gbm'),
            '--output-dir', str(output_dir),
            '--epochs', str(self.epochs),
            '--batch-size', str(self.batch_size),
            '--mixed-precision',
        ]

        # Add configuration flags
        for key, value in config.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f'--{key.replace("_", "-")}')
            else:
                cmd.extend([f'--{key.replace("_", "-")}', str(value)])

        # Set environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.gpu)

        # Save configuration
        config_file = output_dir / 'ablation_config.json'
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # Run training
        start_time = time.time()

        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"Logging to: {log_file}\n")

        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env
            )
            process.wait()

        duration = time.time() - start_time

        # Collect results
        result = {
            'name': name,
            'description': self.ablations[name]['description'],
            'config': config,
            'status': 'success' if process.returncode == 0 else 'failed',
            'returncode': process.returncode,
            'duration': duration,
            'output_dir': str(output_dir),
            'log_file': str(log_file),
        }

        if process.returncode == 0:
            logger.info(f"✓ Ablation '{name}' completed in {duration/3600:.2f} hours")
        else:
            logger.error(f"✗ Ablation '{name}' failed!")

        return result

    def run_all_ablations(self, specific_ablations: Optional[List[str]] = None):
        """
        Run all ablation studies.

        Args:
            specific_ablations: List of specific ablations to run (if None, runs all)
        """
        self.start_time = time.time()

        logger.info("\n" + "="*80)
        logger.info("ABLATION STUDIES ORCHESTRATION")
        logger.info("="*80)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"GPU: {self.gpu}")
        logger.info("="*80 + "\n")

        # Determine which ablations to run
        if specific_ablations:
            ablations_to_run = {k: v for k, v in self.ablations.items() if k in specific_ablations}
        else:
            ablations_to_run = self.ablations

        logger.info(f"Running {len(ablations_to_run)} ablation studies:")
        for name in ablations_to_run.keys():
            logger.info(f"  - {name}: {self.ablations[name]['description']}")
        logger.info("")

        # Run each ablation
        for name, ablation_info in ablations_to_run.items():
            result = self.run_ablation(name, ablation_info['config'])
            self.results[name] = result

        # Save results
        self._save_results()

        # Print summary
        self._print_summary()

    def _save_results(self):
        """Save ablation results to JSON."""
        results_file = self.output_dir / 'ablation_results.json'

        results_data = {
            'start_time': self.start_time,
            'end_time': time.time(),
            'total_duration': time.time() - self.start_time,
            'config': {
                'data_dir': str(self.data_dir),
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'gpu': self.gpu,
            },
            'results': self.results,
        }

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"\nResults saved to: {results_file}")

    def _print_summary(self):
        """Print summary of ablation studies."""
        logger.info("\n" + "="*80)
        logger.info("ABLATION STUDIES SUMMARY")
        logger.info("="*80)

        # Count successes and failures
        successes = sum(1 for r in self.results.values() if r['status'] == 'success')
        failures = sum(1 for r in self.results.values() if r['status'] == 'failed')

        logger.info(f"Total ablations: {len(self.results)}")
        logger.info(f"Successful: {successes}")
        logger.info(f"Failed: {failures}")
        logger.info("")

        # List each ablation
        for name, result in self.results.items():
            status_symbol = "✓" if result['status'] == 'success' else "✗"
            logger.info(f"{status_symbol} {name}")
            logger.info(f"   {result['description']}")
            logger.info(f"   Duration: {result['duration']/3600:.2f} hours")
            logger.info(f"   Output: {result['output_dir']}")
            logger.info("")

        total_duration = time.time() - self.start_time
        logger.info(f"Total time: {total_duration/3600:.2f} hours")
        logger.info("="*80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run ablation studies for SA-CycleGAN'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory with preprocessed data'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for ablation experiments'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (fewer than main experiment)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size'
    )

    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU ID to use'
    )

    parser.add_argument(
        '--ablations',
        nargs='+',
        default=None,
        help='Specific ablations to run (default: all)'
    )

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = AblationStudyOrchestrator(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        gpu=args.gpu,
    )

    # Run ablations
    orchestrator.run_all_ablations(specific_ablations=args.ablations)


if __name__ == '__main__':
    main()
