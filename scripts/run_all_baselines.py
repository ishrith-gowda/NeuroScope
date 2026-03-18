#!/usr/bin/env python3
"""
orchestration script for running all baseline methods.

automatically runs all baseline harmonization methods:
1. baseline cyclegan (no self-attention)
2. combat (statistical harmonization)
3. cut (contrastive unpaired translation)
4. histogram matching
5. unit (unsupervised image-to-image translation)

this script orchestrates the entire baseline experiment pipeline,
allowing for reproducible and automated comparison.

usage:
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

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaselineOrchestrator:
    """orchestrates execution of all baseline methods."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        epochs: int = 200,
        batch_size: int = 4,
        gpus: Optional[List[int]] = None,
    ):
        """
        initialize orchestrator.

        args:
            data_dir: directory with preprocessed data
            output_dir: output directory for experiments
            epochs: number of training epochs
            batch_size: batch size for training
            gpus: list of gpu ids to use
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = epochs
        self.batch_size = batch_size
        self.gpus = gpus or [0]

        self.scripts_dir = Path(__file__).parent

        # experiment tracking
        self.experiments = {}
        self.start_time = None

    def run_command(
        self,
        cmd: List[str],
        env: Optional[Dict] = None,
        log_file: Optional[Path] = None
    ) -> int:
        """
        execute a command and log output.

        args:
            cmd: command as list of strings
            env: environment variables
            log_file: optional log file path

        returns:
            return code
        """
        logger.info(f"Running: {' '.join(cmd)}")

        # prepare environment
        if env is None:
            env = os.environ.copy()

        # setup logging
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
        """run baseline cyclegan (no self-attention)."""
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

        # set gpu
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
        """run combat statistical harmonization."""
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
        """run histogram matching baseline."""
        logger.info("\n" + "="*80)
        logger.info("Running Histogram Matching")
        logger.info("="*80 + "\n")

        output_dir = self.output_dir / 'histogram_matching'
        output_dir.mkdir(parents=True, exist_ok=True)

        # simple histogram matching implementation
        # this would be implemented as a separate script
        logger.info("Histogram matching implementation needed")
        logger.info("(Can be implemented as simple intensity transformation)")

        self.experiments['histogram_matching'] = {
            'status': 'pending',
            'note': 'Requires implementation',
            'output_dir': str(output_dir),
        }

    def run_cut(self):
        """run cut (contrastive unpaired translation)."""
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
        """run unit (unsupervised image-to-image translation)."""
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
        """save experiment tracking log."""
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
        run all baseline methods.

        args:
            methods: list of methods to run (if none, runs all)
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

        # run each method
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

        # save experiment log
        self.save_experiment_log()

        # print summary
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
    """main execution function."""
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

    # parse gpu list
    gpus = [int(g) for g in args.gpus.split(',')]

    # initialize orchestrator
    orchestrator = BaselineOrchestrator(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        gpus=gpus,
    )

    # run all baselines
    orchestrator.run_all(methods=args.methods)


if __name__ == '__main__':
    main()
