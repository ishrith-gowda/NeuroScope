#!/usr/bin/env python3
"""
Orchestration Script for Running All Baseline Methods.

Automatically runs all baseline harmonization methods:
1. Baseline CycleGAN (no self-attention)
2. ComBat (statistical harmonization)
3. CUT (Contrastive Unpaired Translation)
4. Histogram Matching
5. UNIT (Unsupervised Image-to-Image Translation)

This script orchestrates the entire baseline experiment pipeline,
allowing for reproducible and automated comparison.

Usage:
    python scripts/run_all_baselines.py \
        --data-dir ./data/processed \
        --output-dir ./experiments \
        --epochs 200 \
        --batch-size 4 \
        --gpus 0,1
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


class BaselineOrchestrator:
    """Orchestrates execution of all baseline methods."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        epochs: int = 200,
        batch_size: int = 4,
        gpus: Optional[List[int]] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            data_dir: Directory with preprocessed data
            output_dir: Output directory for experiments
            epochs: Number of training epochs
            batch_size: Batch size for training
            gpus: List of GPU IDs to use
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = epochs
        self.batch_size = batch_size
        self.gpus = gpus or [0]

        self.scripts_dir = Path(__file__).parent

        # Experiment tracking
        self.experiments = {}
        self.start_time = None

    def run_command(
        self,
        cmd: List[str],
        env: Optional[Dict] = None,
        log_file: Optional[Path] = None
    ) -> int:
        """
        Execute a command and log output.

        Args:
            cmd: Command as list of strings
            env: Environment variables
            log_file: Optional log file path

        Returns:
            Return code
        """
        logger.info(f"Running: {' '.join(cmd)}")

        # Prepare environment
        if env is None:
            env = os.environ.copy()

        # Setup logging
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env
                )
                process.wait()
        else:
            process = subprocess.run(cmd, env=env)

        return process.returncode

    def run_baseline_cyclegan(self):
        """Run baseline CycleGAN (no self-attention)."""
        logger.info("\n" + "="*80)
        logger.info("Running Baseline CycleGAN")
        logger.info("="*80 + "\n")

        output_dir = self.output_dir / 'baseline_cyclegan'
        log_file = output_dir / 'training.log'

        cmd = [
            'python',
            str(self.scripts_dir / 'train_baseline_cyclegan.py'),
            '--dataset-a', str(self.data_dir / 'brats'),
            '--dataset-b', str(self.data_dir / 'upenn_gbm'),
            '--output-dir', str(output_dir),
            '--epochs', str(self.epochs),
            '--batch-size', str(self.batch_size),
            '--mixed-precision',
            '--save-freq', '10',
        ]

        # Set GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.gpus))

        start_time = time.time()
        returncode = self.run_command(cmd, env=env, log_file=log_file)
        duration = time.time() - start_time

        self.experiments['baseline_cyclegan'] = {
            'status': 'success' if returncode == 0 else 'failed',
            'duration': duration,
            'output_dir': str(output_dir),
            'log_file': str(log_file),
        }

        if returncode == 0:
            logger.info(f"+ Baseline CycleGAN completed in {duration/3600:.2f} hours")
        else:
            logger.error(f"✗ Baseline CycleGAN failed!")

    def run_combat(self):
        """Run ComBat statistical harmonization."""
        logger.info("\n" + "="*80)
        logger.info("Running ComBat Harmonization")
        logger.info("="*80 + "\n")

        output_dir = self.output_dir / 'combat'
        log_file = output_dir / 'harmonization.log'

        cmd = [
            'python',
            str(self.scripts_dir / 'run_combat_baseline.py'),
            '--source-dir', str(self.data_dir / 'brats'),
            '--target-dir', str(self.data_dir / 'upenn_gbm'),
            '--output-dir', str(output_dir),
            '--modalities', 'T1', 'T1CE', 'T2', 'FLAIR',
            '--parametric',
        ]

        start_time = time.time()
        returncode = self.run_command(cmd, log_file=log_file)
        duration = time.time() - start_time

        self.experiments['combat'] = {
            'status': 'success' if returncode == 0 else 'failed',
            'duration': duration,
            'output_dir': str(output_dir),
            'log_file': str(log_file),
        }

        if returncode == 0:
            logger.info(f"+ ComBat completed in {duration/60:.2f} minutes")
        else:
            logger.error(f"✗ ComBat failed!")

    def run_histogram_matching(self):
        """Run histogram matching baseline."""
        logger.info("\n" + "="*80)
        logger.info("Running Histogram Matching")
        logger.info("="*80 + "\n")

        output_dir = self.output_dir / 'histogram_matching'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Simple histogram matching implementation
        # This would be implemented as a separate script
        logger.info("Histogram matching implementation needed")
        logger.info("(Can be implemented as simple intensity transformation)")

        self.experiments['histogram_matching'] = {
            'status': 'pending',
            'note': 'Requires implementation',
            'output_dir': str(output_dir),
        }

    def run_cut(self):
        """Run CUT (Contrastive Unpaired Translation)."""
        logger.info("\n" + "="*80)
        logger.info("Running CUT")
        logger.info("="*80 + "\n")

        output_dir = self.output_dir / 'cut'
        logger.info("CUT baseline requires implementation or external repo")
        logger.info("Alternative: Use official CUT implementation")

        self.experiments['cut'] = {
            'status': 'pending',
            'note': 'Requires CUT implementation or external integration',
            'output_dir': str(output_dir),
        }

    def run_unit(self):
        """Run UNIT (Unsupervised Image-to-Image Translation)."""
        logger.info("\n" + "="*80)
        logger.info("Running UNIT")
        logger.info("="*80 + "\n")

        output_dir = self.output_dir / 'unit'
        logger.info("UNIT baseline requires implementation or external repo")

        self.experiments['unit'] = {
            'status': 'pending',
            'note': 'Requires UNIT implementation or external integration',
            'output_dir': str(output_dir),
        }

    def save_experiment_log(self):
        """Save experiment tracking log."""
        log_file = self.output_dir / 'baseline_experiments.json'

        log_data = {
            'start_time': self.start_time,
            'end_time': time.time(),
            'total_duration': time.time() - self.start_time,
            'config': {
                'data_dir': str(self.data_dir),
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'gpus': self.gpus,
            },
            'experiments': self.experiments,
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"\nExperiment log saved to: {log_file}")

    def run_all(self, methods: Optional[List[str]] = None):
        """
        Run all baseline methods.

        Args:
            methods: List of methods to run (if None, runs all)
        """
        self.start_time = time.time()

        if methods is None:
            methods = ['baseline_cyclegan', 'combat', 'histogram', 'cut', 'unit']

        logger.info("\n" + "="*80)
        logger.info("BASELINE METHODS ORCHESTRATION")
        logger.info("="*80)
        logger.info(f"Methods to run: {', '.join(methods)}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"GPUs: {self.gpus}")
        logger.info("="*80 + "\n")

        # Run each method
        if 'baseline_cyclegan' in methods:
            self.run_baseline_cyclegan()

        if 'combat' in methods:
            self.run_combat()

        if 'histogram' in methods:
            self.run_histogram_matching()

        if 'cut' in methods:
            self.run_cut()

        if 'unit' in methods:
            self.run_unit()

        # Save experiment log
        self.save_experiment_log()

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("BASELINE EXPERIMENTS SUMMARY")
        logger.info("="*80)

        for method, info in self.experiments.items():
            status_symbol = "+" if info['status'] == 'success' else "✗"
            logger.info(f"{status_symbol} {method}: {info['status']}")

            if 'duration' in info:
                logger.info(f"   Duration: {info['duration']/3600:.2f} hours")

            if 'note' in info:
                logger.info(f"   Note: {info['note']}")

        total_duration = time.time() - self.start_time
        logger.info(f"\nTotal time: {total_duration/3600:.2f} hours")
        logger.info("="*80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run all baseline harmonization methods'
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
        help='Output directory for experiments'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for training'
    )

    parser.add_argument(
        '--gpus',
        type=str,
        default='0',
        help='Comma-separated list of GPU IDs'
    )

    parser.add_argument(
        '--methods',
        nargs='+',
        default=None,
        choices=['baseline_cyclegan', 'combat', 'histogram', 'cut', 'unit'],
        help='Specific methods to run (default: all)'
    )

    args = parser.parse_args()

    # Parse GPU list
    gpus = [int(g) for g in args.gpus.split(',')]

    # Initialize orchestrator
    orchestrator = BaselineOrchestrator(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        gpus=gpus,
    )

    # Run all baselines
    orchestrator.run_all(methods=args.methods)


if __name__ == '__main__':
    main()
