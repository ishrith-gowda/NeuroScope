#!/usr/bin/env python3
"""
comprehensive evaluation script for sa-cyclegan-2.5d

evaluates trained model on test set with multiple metrics:
- ssim (structural similarity index)
- psnr (peak signal-to-noise ratio)
- mae (mean absolute error)
- mse (mean squared error)
- lpips (learned perceptual image patch similarity)
- fid (fréchet inception distance)

this script provides publication-quality quantitative results.
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

# suppress warnings
warnings.filterwarnings('ignore')

# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neuroscope.models.architectures.sa_cyclegan_25d import SACycleGAN25D, SACycleGAN25DConfig
from neuroscope.data.datasets.dataset_25d import UnpairedMRIDataset25D, create_dataloaders

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """evaluates trained sa-cyclegan-2.5d model on test set."""

    def __init__(
        self,
        checkpoint_path: str,
        brats_dir: str,
        upenn_dir: str,
        output_dir: str,
        device: str = 'cuda',
        batch_size: int = 16,
        num_workers: int = 8,
    ):
        """
        initialize evaluator.

        args:
            checkpoint_path: path to trained model checkpoint
            brats_dir: brats dataset directory
            upenn_dir: upenn dataset directory
            output_dir: output directory for results
            device: computation device
            batch_size: batch size for evaluation
            num_workers: number of dataloader workers
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.brats_dir = Path(brats_dir)
        self.upenn_dir = Path(upenn_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.num_workers = num_workers

        # store checkpoint metadata
        self.checkpoint_epoch = None
        self.checkpoint_best_metric = None

        logger.info(f"device: {self.device}")
        logger.info(f"checkpoint: {self.checkpoint_path}")

        # load lpips model if available
        self.lpips_model = None
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            logger.info("lpips model loaded successfully")
        except ImportError:
            logger.warning("lpips not available. install with: pip install lpips")

    def load_model(self) -> Tuple[nn.Module, Dict]:
        """load trained model from checkpoint."""
        logger.info("loading checkpoint...")
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        # extract config
        config_dict = checkpoint.get('config', {})

        # store checkpoint metadata
        self.checkpoint_epoch = checkpoint['epoch']
        self.checkpoint_best_metric = checkpoint.get('best_metric', None)

        logger.info(f"checkpoint epoch: {self.checkpoint_epoch}")
        logger.info(f"best metric: {self.checkpoint_best_metric}")

        # create model config
        model_config = SACycleGAN25DConfig(
            ngf=config_dict.get('ngf', 64),
            ndf=config_dict.get('ndf', 64),
            n_residual_blocks=config_dict.get('n_residual_blocks', 9),
        )

        # instantiate model
        model = SACycleGAN25D(model_config)

        # load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        logger.info("model loaded successfully")
        logger.info(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

        return model, config_dict

    def load_test_data(self) -> DataLoader:
        """load test dataset."""
        logger.info("loading test dataset...")

        # create dataloaders with same split as training
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

    def compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """compute ssim between two images."""
        # handle multi-channel images (compute per channel and average)
        if img1.ndim == 3 and img1.shape[0] > 1:
            ssim_scores = []
            n_channels = min(img1.shape[0], img2.shape[0])  # use minimum to avoid index errors
            for c in range(n_channels):
                # ensure slices have correct shape
                slice1 = img1[c]
                slice2 = img2[c]
                if slice1.size == 0 or slice2.size == 0:
                    continue
                score = ssim_func(
                    slice1,
                    slice2,
                    data_range=max(slice1.max() - slice1.min(), slice2.max() - slice2.min()) + 1e-8
                )
                ssim_scores.append(score)
            return float(np.mean(ssim_scores)) if ssim_scores else 0.0
        else:
            return float(ssim_func(
                img1,
                img2,
                data_range=max(img1.max() - img1.min(), img2.max() - img2.min()) + 1e-8
            ))

    def compute_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """compute psnr between two images."""
        data_range = max(img1.max() - img1.min(), img2.max() - img2.min()) + 1e-8
        return float(psnr_func(img1, img2, data_range=data_range))

    def compute_mae(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """compute mean absolute error."""
        return float(np.mean(np.abs(img1 - img2)))

    def compute_mse(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """compute mean squared error."""
        return float(np.mean((img1 - img2) ** 2))

    def compute_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> List[float]:
        """compute lpips perceptual similarity for a batch."""
        if self.lpips_model is None:
            return [-1.0] * img1.size(0)

        with torch.no_grad():
            # lpips expects images in [-1, 1] range
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)

            # normalize to [-1, 1] per sample
            batch_size = img1.size(0)
            for i in range(batch_size):
                img1[i] = 2 * (img1[i] - img1[i].min()) / (img1[i].max() - img1[i].min() + 1e-8) - 1
                img2[i] = 2 * (img2[i] - img2[i].min()) / (img2[i].max() - img2[i].min() + 1e-8) - 1

            # for 4-channel MRI, compute lpips on each channel and average
            if img1.size(1) == 4:
                lpips_scores = []
                for c in range(4):
                    score = self.lpips_model(
                        img1[:, c:c+1].repeat(1, 3, 1, 1),
                        img2[:, c:c+1].repeat(1, 3, 1, 1)
                    )
                    # score is [batch_size, 1, 1, 1], squeeze and convert
                    if c == 0:
                        lpips_scores = score.squeeze().cpu().numpy()
                    else:
                        lpips_scores += score.squeeze().cpu().numpy()
                # average across channels
                lpips_scores = lpips_scores / 4.0
                return lpips_scores.tolist() if hasattr(lpips_scores, 'tolist') else [float(lpips_scores)]
            else:
                # for other channel counts
                lpips_score = self.lpips_model(img1, img2)
                return lpips_score.squeeze().cpu().numpy().tolist()

        return [-1.0] * img1.size(0)

    def run_evaluation(self) -> Dict:
        """run comprehensive evaluation on test set."""
        logger.info("starting comprehensive evaluation...")

        # load model and test data
        model, config = self.load_model()
        test_loader = self.load_test_data()

        # storage for metrics
        metrics_a2b = {
            'ssim': [],
            'psnr': [],
            'mae': [],
            'mse': [],
            'lpips': [],
        }

        metrics_b2a = {
            'ssim': [],
            'psnr': [],
            'mae': [],
            'mse': [],
            'lpips': [],
        }

        # storage for fid computation
        real_b_images = []
        fake_b_images = []
        real_a_images = []
        fake_a_images = []

        # evaluate on test set
        logger.info("running inference on test set...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="evaluating")):
                real_a = batch['A'].to(self.device)  # [B, 12, H, W] - 3 slices
                real_b = batch['B'].to(self.device)  # [B, 12, H, W] - 3 slices
                center_a = batch['A_center'].to(self.device)  # [B, 4, H, W] - center slice only
                center_b = batch['B_center'].to(self.device)  # [B, 4, H, W] - center slice only

                # forward pass (generators output center slice only)
                fake_b = model.G_A2B(real_a)  # [B, 4, H, W]
                fake_a = model.G_B2A(real_b)  # [B, 4, H, W]

                # move to cpu for metric computation
                center_a_cpu = center_a.cpu()
                center_b_cpu = center_b.cpu()
                fake_a_cpu = fake_a.cpu()
                fake_b_cpu = fake_b.cpu()

                # compute metrics for each sample in batch
                for i in range(center_a.size(0)):
                    # convert to numpy for ssim/psnr
                    ca_np = center_a_cpu[i].numpy()  # center slice of A
                    cb_np = center_b_cpu[i].numpy()  # center slice of B
                    fa_np = fake_a_cpu[i].numpy()    # generated A
                    fb_np = fake_b_cpu[i].numpy()    # generated B

                    # debug logging for first batch
                    if batch_idx == 0 and i == 0:
                        logger.info(f"debug shapes - ca: {ca_np.shape}, cb: {cb_np.shape}, fa: {fa_np.shape}, fb: {fb_np.shape}")

                    # a -> b metrics (brats -> upenn): compare generated B with real center B
                    metrics_a2b['ssim'].append(self.compute_ssim(cb_np, fb_np))
                    metrics_a2b['psnr'].append(self.compute_psnr(cb_np, fb_np))
                    metrics_a2b['mae'].append(self.compute_mae(cb_np, fb_np))
                    metrics_a2b['mse'].append(self.compute_mse(cb_np, fb_np))

                    # b -> a metrics (upenn -> brats): compare generated A with real center A
                    metrics_b2a['ssim'].append(self.compute_ssim(ca_np, fa_np))
                    metrics_b2a['psnr'].append(self.compute_psnr(ca_np, fa_np))
                    metrics_b2a['mae'].append(self.compute_mae(ca_np, fa_np))
                    metrics_b2a['mse'].append(self.compute_mse(ca_np, fa_np))

                # compute lpips on batch (more efficient)
                if self.lpips_model is not None:
                    lpips_a2b_batch = self.compute_lpips(center_b, fake_b)
                    lpips_b2a_batch = self.compute_lpips(center_a, fake_a)

                    metrics_a2b['lpips'].extend(lpips_a2b_batch)
                    metrics_b2a['lpips'].extend(lpips_b2a_batch)

                # collect images for fid (use first channel of center slices)
                for i in range(center_a.size(0)):
                    real_b_images.append(center_b_cpu[i, 0].numpy())
                    fake_b_images.append(fake_b_cpu[i, 0].numpy())
                    real_a_images.append(center_a_cpu[i, 0].numpy())
                    fake_a_images.append(fake_a_cpu[i, 0].numpy())

        logger.info("computing fid scores...")
        fid_a2b = self.compute_fid(real_b_images, fake_b_images)
        fid_b2a = self.compute_fid(real_a_images, fake_a_images)

        # compute statistics
        logger.info("computing statistics...")
        results = {
            'checkpoint': str(self.checkpoint_path),
            'checkpoint_epoch': self.checkpoint_epoch,
            'checkpoint_best_metric': self.checkpoint_best_metric,
            'test_samples': len(test_loader.dataset),
            'evaluation_timestamp': datetime.now().isoformat(),
            'a2b': self._compute_statistics(metrics_a2b, fid_a2b),
            'b2a': self._compute_statistics(metrics_b2a, fid_b2a),
        }

        # save results
        output_file = self.output_dir / 'evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"results saved to {output_file}")

        # print summary
        self._print_summary(results)

        return results

    def _compute_statistics(self, metrics: Dict[str, List[float]], fid: float) -> Dict:
        """compute statistics for metrics."""
        stats = {}

        for metric_name, values in metrics.items():
            if len(values) > 0 and values[0] != -1.0:
                stats[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'q25': float(np.percentile(values, 25)),
                    'q75': float(np.percentile(values, 75)),
                }

        stats['fid'] = {'value': fid}

        return stats

    def compute_fid(self, real_images: List[np.ndarray], fake_images: List[np.ndarray]) -> float:
        """compute fréchet inception distance."""
        try:
            from scipy.linalg import sqrtm
            from torchvision import models, transforms
            from PIL import Image

            # load inception model
            inception = models.inception_v3(pretrained=True, transform_input=False)
            inception.fc = nn.Identity()
            inception = inception.to(self.device)
            inception.eval()

            def extract_features(images):
                features = []

                for img in tqdm(images, desc="extracting inception features"):
                    # normalize to [0, 255]
                    img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

                    # convert to 3-channel rgb
                    img_rgb = np.stack([img] * 3, axis=-1)
                    img_pil = Image.fromarray(img_rgb)

                    # resize to inception input size
                    img_pil = img_pil.resize((299, 299), Image.BILINEAR)

                    # to tensor and normalize
                    img_tensor = transforms.ToTensor()(img_pil)
                    img_tensor = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img_tensor)
                    img_tensor = img_tensor.unsqueeze(0).to(self.device)

                    # extract features
                    with torch.no_grad():
                        feat = inception(img_tensor)

                    features.append(feat.cpu().numpy())

                return np.concatenate(features, axis=0)

            # extract features
            real_features = extract_features(real_images)
            fake_features = extract_features(fake_images)

            # compute statistics
            mu_real = np.mean(real_features, axis=0)
            sigma_real = np.cov(real_features, rowvar=False)

            mu_fake = np.mean(fake_features, axis=0)
            sigma_fake = np.cov(fake_features, rowvar=False)

            # compute fid
            diff = mu_real - mu_fake
            covmean = sqrtm(sigma_real @ sigma_fake)

            if np.iscomplexobj(covmean):
                covmean = covmean.real

            fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)

            return float(fid)

        except Exception as e:
            logger.error(f"fid computation failed: {e}")
            return -1.0

    def _print_summary(self, results: Dict):
        """print evaluation summary."""
        logger.info("\n" + "=" * 80)
        logger.info("evaluation results summary")
        logger.info("=" * 80)

        logger.info(f"\ncheckpoint: {results['checkpoint']}")
        logger.info(f"epoch: {results['checkpoint_epoch']}")
        logger.info(f"test samples: {results['test_samples']}")

        logger.info("\na -> b (brats -> upenn) metrics:")
        logger.info("-" * 40)
        for metric, stats in results['a2b'].items():
            if metric == 'fid':
                logger.info(f"  {metric.upper()}: {stats['value']:.4f}")
            else:
                logger.info(f"  {metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        logger.info("\nb -> a (upenn -> brats) metrics:")
        logger.info("-" * 40)
        for metric, stats in results['b2a'].items():
            if metric == 'fid':
                logger.info(f"  {metric.upper()}: {stats['value']:.4f}")
            else:
                logger.info(f"  {metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        logger.info("\n" + "=" * 80)


def main():
    """main execution function."""
    parser = argparse.ArgumentParser(
        description='comprehensive evaluation of sa-cyclegan-2.5d'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='path to trained model checkpoint'
    )

    parser.add_argument(
        '--brats-dir',
        type=str,
        required=True,
        help='brats dataset directory'
    )

    parser.add_argument(
        '--upenn-dir',
        type=str,
        required=True,
        help='upenn dataset directory'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='output directory for results'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='computation device'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='batch size for evaluation'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='number of dataloader workers'
    )

    args = parser.parse_args()

    # run evaluation
    evaluator = ComprehensiveEvaluator(
        checkpoint_path=args.checkpoint,
        brats_dir=args.brats_dir,
        upenn_dir=args.upenn_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    results = evaluator.run_evaluation()

    logger.info("\nevaluation complete!")


if __name__ == '__main__':
    main()
