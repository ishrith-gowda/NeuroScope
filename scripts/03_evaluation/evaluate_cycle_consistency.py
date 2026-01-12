#!/usr/bin/env python3
"""
cycle consistency evaluation for sa-cyclegan-2.5d

evaluates the model's cycle consistency (A→B→A and B→A→B) which is what
cycleg

an optimizes for. this is the CORRECT metric for unpaired translation,
unlike direct SSIM which compares to random target anatomy.

cycle consistency SSIM should match the training validation SSIM (~0.98).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neuroscope.models.architectures.sa_cyclegan_25d import SACycleGAN25D, SACycleGAN25DConfig
from neuroscope.data.datasets.dataset_25d import create_dataloaders

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CycleConsistencyEvaluator:
    """evaluates cycle consistency of trained model."""

    def __init__(
        self,
        checkpoint_path: str,
        brats_dir: str,
        upenn_dir: str,
        output_dir: str,
        device: str = 'cuda',
        batch_size: int = 32,
        num_workers: int = 16,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.brats_dir = Path(brats_dir)
        self.upenn_dir = Path(upenn_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.num_workers = num_workers

        logger.info(f"device: {self.device}")
        logger.info(f"checkpoint: {self.checkpoint_path}")

    def load_model(self):
        """load trained model from checkpoint."""
        logger.info("loading checkpoint...")
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        config_dict = checkpoint.get('config', {})
        epoch = checkpoint['epoch']

        logger.info(f"checkpoint epoch: {epoch}")

        # create model
        model_config = SACycleGAN25DConfig(
            ngf=config_dict.get('ngf', 64),
            ndf=config_dict.get('ndf', 64),
            n_residual_blocks=config_dict.get('n_residual_blocks', 9),
        )

        model = SACycleGAN25D(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        logger.info(f"model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

        return model, epoch

    def load_test_data(self) -> DataLoader:
        """load test dataset."""
        logger.info("loading test dataset...")

        _, _, test_loader = create_dataloaders(
            brats_dir=str(self.brats_dir),
            upenn_dir=str(self.upenn_dir),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            image_size=(128, 128),
            train_split=0.7,
            val_split=0.15,
        )

        logger.info(f"test samples: {len(test_loader.dataset)}")

        return test_loader

    def compute_cycle_metrics(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray
    ) -> Dict[str, float]:
        """compute metrics between original and cycle-reconstructed images."""
        # compute per-channel metrics
        ssim_scores = []
        psnr_scores = []
        mae_scores = []

        for c in range(original.shape[0]):
            # ssim
            ssim_val = ssim_func(
                original[c],
                reconstructed[c],
                data_range=max(original[c].max() - original[c].min(),
                              reconstructed[c].max() - reconstructed[c].min()) + 1e-8
            )
            ssim_scores.append(ssim_val)

            # psnr
            psnr_val = psnr_func(
                original[c],
                reconstructed[c],
                data_range=max(original[c].max() - original[c].min(),
                              reconstructed[c].max() - reconstructed[c].min()) + 1e-8
            )
            psnr_scores.append(psnr_val)

            # mae
            mae_val = np.mean(np.abs(original[c] - reconstructed[c]))
            mae_scores.append(mae_val)

        return {
            'ssim': float(np.mean(ssim_scores)),
            'psnr': float(np.mean(psnr_scores)),
            'mae': float(np.mean(mae_scores)),
        }

    def evaluate(self):
        """run cycle consistency evaluation."""
        logger.info("starting cycle consistency evaluation...")

        model, epoch = self.load_model()
        test_loader = self.load_test_data()

        # storage for metrics
        cycle_a_metrics = {'ssim': [], 'psnr': [], 'mae': []}
        cycle_b_metrics = {'ssim': [], 'psnr': [], 'mae': []}

        logger.info("running cycle reconstructions...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="evaluating cycles")):
                real_a = batch['A'].to(self.device)  # [B, 12, H, W]
                real_b = batch['B'].to(self.device)  # [B, 12, H, W]
                center_a = batch['A_center'].to(self.device)  # [B, 4, H, W]
                center_b = batch['B_center'].to(self.device)  # [B, 4, H, W]

                # cycle A: A → B → A
                fake_b = model.G_A2B(real_a)  # [B, 4, H, W]
                fake_b_3slice = fake_b.unsqueeze(2).repeat(1, 1, 3, 1, 1)  # [B, 4, 3, H, W]
                fake_b_3slice = fake_b_3slice.view(fake_b.size(0), -1, fake_b.size(2), fake_b.size(3))  # [B, 12, H, W]
                rec_a = model.G_B2A(fake_b_3slice)  # [B, 4, H, W]

                # cycle B: B → A → B
                fake_a = model.G_B2A(real_b)  # [B, 4, H, W]
                fake_a_3slice = fake_a.unsqueeze(2).repeat(1, 1, 3, 1, 1)  # [B, 4, 3, H, W]
                fake_a_3slice = fake_a_3slice.view(fake_a.size(0), -1, fake_a.size(2), fake_a.size(3))  # [B, 12, H, W]
                rec_b = model.G_A2B(fake_a_3slice)  # [B, 4, H, W]

                # move to cpu
                center_a_cpu = center_a.cpu().numpy()
                center_b_cpu = center_b.cpu().numpy()
                rec_a_cpu = rec_a.cpu().numpy()
                rec_b_cpu = rec_b.cpu().numpy()

                # compute metrics for each sample
                for i in range(center_a.size(0)):
                    # cycle A metrics
                    metrics_a = self.compute_cycle_metrics(center_a_cpu[i], rec_a_cpu[i])
                    for key in metrics_a:
                        cycle_a_metrics[key].append(metrics_a[key])

                    # cycle B metrics
                    metrics_b = self.compute_cycle_metrics(center_b_cpu[i], rec_b_cpu[i])
                    for key in metrics_b:
                        cycle_b_metrics[key].append(metrics_b[key])

        # compute statistics
        logger.info("computing statistics...")
        results = {
            'checkpoint': str(self.checkpoint_path),
            'checkpoint_epoch': epoch,
            'test_samples': len(test_loader.dataset),
            'evaluation_timestamp': datetime.now().isoformat(),
            'cycle_a': self._compute_stats(cycle_a_metrics),
            'cycle_b': self._compute_stats(cycle_b_metrics),
        }

        # save results
        output_file = self.output_dir / 'cycle_consistency_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"results saved to {output_file}")

        # print summary
        self._print_summary(results)

        return results

    def _compute_stats(self, metrics: Dict) -> Dict:
        """compute statistics for metrics."""
        stats = {}

        for metric_name, values in metrics.items():
            stats[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
            }

        return stats

    def _print_summary(self, results: Dict):
        """print evaluation summary."""
        logger.info("\n" + "=" * 80)
        logger.info("cycle consistency evaluation results")
        logger.info("=" * 80)

        logger.info(f"\ncheckpoint: {results['checkpoint']}")
        logger.info(f"epoch: {results['checkpoint_epoch']}")
        logger.info(f"test samples: {results['test_samples']}")

        logger.info("\ncycle A (A→B→A) metrics:")
        logger.info("-" * 40)
        for metric, stats in results['cycle_a'].items():
            logger.info(f"  {metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        logger.info("\ncycle B (B→A→B) metrics:")
        logger.info("-" * 40)
        for metric, stats in results['cycle_b'].items():
            logger.info(f"  {metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        logger.info("\n" + "=" * 80)


def main():
    """main execution."""
    parser = argparse.ArgumentParser(
        description='cycle consistency evaluation'
    )

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--brats-dir', type=str, required=True)
    parser.add_argument('--upenn-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=16)

    args = parser.parse_args()

    evaluator = CycleConsistencyEvaluator(
        checkpoint_path=args.checkpoint,
        brats_dir=args.brats_dir,
        upenn_dir=args.upenn_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    evaluator.evaluate()

    logger.info("\nevaluation complete!")


if __name__ == '__main__':
    main()
