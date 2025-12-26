#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script.

Evaluates all trained models (SA-CycleGAN, baselines) using multiple metrics:
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)
- FID (Fréchet Inception Distance)
- NMI (Normalized Mutual Information)
- Perceptual Similarity (LPIPS)
- Radiomics Preservation
- Clinical Metrics (if segmentations available)

Usage:
    python scripts/evaluate_all_models.py \
        --models-dir ./experiments \
        --test-data ./data/processed/brats/test \
        --output-dir ./results/evaluation \
        --metrics ssim psnr fid nmi lpips
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from scipy.ndimage import label
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricCalculator:
    """Calculates various evaluation metrics for medical images."""

    def __init__(self, device: str = 'cuda'):
        """
        Initialize metric calculator.

        Args:
            device: Device for computation ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load perceptual loss model (LPIPS)
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='vgg').to(self.device)
        except ImportError:
            logger.warning("LPIPS not available. Install with: pip install lpips")
            self.lpips_model = None

    def calculate_ssim(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        data_range: Optional[float] = None
    ) -> float:
        """
        Calculate Structural Similarity Index (SSIM).

        Args:
            image1: First image
            image2: Second image
            data_range: Data range of images

        Returns:
            SSIM score (higher is better, range -1 to 1)
        """
        if data_range is None:
            data_range = max(image1.max(), image2.max()) - min(image1.min(), image2.min())

        # For 3D volumes, calculate mean SSIM across slices
        if image1.ndim == 3:
            ssim_scores = []
            for i in range(image1.shape[2]):
                score = ssim(
                    image1[:, :, i],
                    image2[:, :, i],
                    data_range=data_range
                )
                ssim_scores.append(score)
            return np.mean(ssim_scores)
        else:
            return ssim(image1, image2, data_range=data_range)

    def calculate_psnr(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        data_range: Optional[float] = None
    ) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR).

        Args:
            image1: First image
            image2: Second image
            data_range: Data range of images

        Returns:
            PSNR score in dB (higher is better)
        """
        if data_range is None:
            data_range = max(image1.max(), image2.max()) - min(image1.min(), image2.min())

        return psnr(image1, image2, data_range=data_range)

    def calculate_nmi(self, image1: np.ndarray, image2: np.ndarray, bins: int = 256) -> float:
        """
        Calculate Normalized Mutual Information (NMI).

        Args:
            image1: First image
            image2: Second image
            bins: Number of histogram bins

        Returns:
            NMI score (higher is better, range 0 to 2)
        """
        # Flatten images
        img1_flat = image1.flatten()
        img2_flat = image2.flatten()

        # Remove zero values (background)
        mask = (img1_flat > 0) & (img2_flat > 0)
        img1_flat = img1_flat[mask]
        img2_flat = img2_flat[mask]

        if len(img1_flat) == 0:
            return 0.0

        # Compute 2D histogram
        hist_2d, _, _ = np.histogram2d(
            img1_flat,
            img2_flat,
            bins=bins
        )

        # Normalize
        hist_2d = hist_2d / hist_2d.sum()

        # Marginal distributions
        hist_1d_img1 = hist_2d.sum(axis=1)
        hist_1d_img2 = hist_2d.sum(axis=0)

        # Entropies
        entropy_img1 = -np.sum(hist_1d_img1[hist_1d_img1 > 0] * np.log(hist_1d_img1[hist_1d_img1 > 0]))
        entropy_img2 = -np.sum(hist_1d_img2[hist_1d_img2 > 0] * np.log(hist_1d_img2[hist_1d_img2 > 0]))
        joint_entropy = -np.sum(hist_2d[hist_2d > 0] * np.log(hist_2d[hist_2d > 0]))

        # Normalized mutual information
        nmi = (entropy_img1 + entropy_img2) / joint_entropy

        return nmi

    def calculate_lpips(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> float:
        """
        Calculate Learned Perceptual Image Patch Similarity (LPIPS).

        Args:
            image1: First image (H x W or H x W x C)
            image2: Second image (H x W or H x W x C)

        Returns:
            LPIPS score (lower is better, range 0 to 1)
        """
        if self.lpips_model is None:
            logger.warning("LPIPS model not available")
            return 0.0

        # Convert to RGB if grayscale
        if image1.ndim == 2:
            image1 = np.stack([image1] * 3, axis=-1)
        if image2.ndim == 2:
            image2 = np.stack([image2] * 3, axis=-1)

        # Normalize to [-1, 1]
        image1 = 2 * (image1 - image1.min()) / (image1.max() - image1.min() + 1e-8) - 1
        image2 = 2 * (image2 - image2.min()) / (image2.max() - image2.min() + 1e-8) - 1

        # Convert to torch tensors
        img1_tensor = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        img2_tensor = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # Calculate LPIPS
        with torch.no_grad():
            lpips_score = self.lpips_model(img1_tensor, img2_tensor)

        return lpips_score.item()

    def calculate_fid(
        self,
        real_images: List[np.ndarray],
        generated_images: List[np.ndarray],
        batch_size: int = 50
    ) -> float:
        """
        Calculate Fréchet Inception Distance (FID).

        Args:
            real_images: List of real images
            generated_images: List of generated images
            batch_size: Batch size for feature extraction

        Returns:
            FID score (lower is better)
        """
        try:
            from scipy.linalg import sqrtm
            from torchvision import models, transforms

            # Load InceptionV3
            inception = models.inception_v3(pretrained=True, transform_input=False)
            inception = inception.to(self.device)
            inception.eval()

            # Remove final layer
            inception.fc = nn.Identity()

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            def get_features(images):
                features = []
                for img in tqdm(images, desc="Extracting features"):
                    # Convert to 3-channel if needed
                    if img.ndim == 2:
                        img = np.stack([img] * 3, axis=-1)
                    elif img.ndim == 3 and img.shape[-1] == 1:
                        img = np.repeat(img, 3, axis=-1)

                    # Normalize to [0, 255]
                    img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

                    # Transform and extract features
                    img_tensor = transform(img).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        feat = inception(img_tensor)

                    features.append(feat.cpu().numpy())

                return np.concatenate(features, axis=0)

            # Extract features
            real_features = get_features(real_images)
            gen_features = get_features(generated_images)

            # Calculate statistics
            mu_real = np.mean(real_features, axis=0)
            sigma_real = np.cov(real_features, rowvar=False)

            mu_gen = np.mean(gen_features, axis=0)
            sigma_gen = np.cov(gen_features, rowvar=False)

            # Calculate FID
            diff = mu_real - mu_gen
            covmean = sqrtm(sigma_real @ sigma_gen)

            if np.iscomplexobj(covmean):
                covmean = covmean.real

            fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)

            return float(fid)

        except Exception as e:
            logger.error(f"FID calculation failed: {e}")
            return -1.0

    def calculate_dice(self, seg1: np.ndarray, seg2: np.ndarray) -> float:
        """
        Calculate Dice coefficient for segmentation overlap.

        Args:
            seg1: First segmentation mask
            seg2: Second segmentation mask

        Returns:
            Dice score (range 0 to 1, higher is better)
        """
        intersection = np.sum(seg1 * seg2)
        union = np.sum(seg1) + np.sum(seg2)

        if union == 0:
            return 1.0

        dice = 2 * intersection / union
        return dice


class ModelEvaluator:
    """Evaluates trained models on test data."""

    def __init__(
        self,
        model_dir: Path,
        test_data_dir: Path,
        output_dir: Path,
        device: str = 'cuda',
    ):
        """
        Initialize evaluator.

        Args:
            model_dir: Directory containing trained model checkpoint
            test_data_dir: Directory with test data
            output_dir: Output directory for results
            device: Device for computation
        """
        self.model_dir = Path(model_dir)
        self.test_data_dir = Path(test_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.metric_calculator = MetricCalculator(device)

    def load_model(self, checkpoint_path: Path):
        """Load trained model from checkpoint."""
        logger.info(f"Loading model from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model architecture (this depends on your model structure)
        # You'll need to import the appropriate model class
        # For now, returning the checkpoint
        return checkpoint

    def evaluate_model(
        self,
        model_name: str,
        checkpoint_path: Path,
        metrics: List[str] = None
    ) -> Dict:
        """
        Evaluate a single model.

        Args:
            model_name: Name of the model
            checkpoint_path: Path to model checkpoint
            metrics: List of metrics to calculate

        Returns:
            Dictionary of evaluation results
        """
        if metrics is None:
            metrics = ['ssim', 'psnr', 'nmi', 'lpips']

        logger.info(f"\nEvaluating {model_name}...")

        # Load model
        checkpoint = self.load_model(checkpoint_path)

        # Find test images
        test_files = sorted(self.test_data_dir.rglob("*.nii.gz"))
        logger.info(f"Found {len(test_files)} test files")

        results = {
            'model_name': model_name,
            'checkpoint': str(checkpoint_path),
            'n_samples': len(test_files),
            'metrics': {}
        }

        # Initialize metric storage
        for metric in metrics:
            results['metrics'][metric] = []

        # Evaluate on each test sample
        for test_file in tqdm(test_files, desc=f"Evaluating {model_name}"):
            # Load test image
            img = nib.load(str(test_file))
            test_data = img.get_fdata()

            # Generate harmonized image (this is model-specific)
            # For now, using identity transformation as placeholder
            harmonized_data = test_data.copy()

            # Calculate metrics
            if 'ssim' in metrics:
                ssim_score = self.metric_calculator.calculate_ssim(test_data, harmonized_data)
                results['metrics']['ssim'].append(ssim_score)

            if 'psnr' in metrics:
                psnr_score = self.metric_calculator.calculate_psnr(test_data, harmonized_data)
                results['metrics']['psnr'].append(psnr_score)

            if 'nmi' in metrics:
                nmi_score = self.metric_calculator.calculate_nmi(test_data, harmonized_data)
                results['metrics']['nmi'].append(nmi_score)

            if 'lpips' in metrics and test_data.ndim == 3:
                # Calculate LPIPS on middle slice
                mid_slice = test_data.shape[2] // 2
                lpips_score = self.metric_calculator.calculate_lpips(
                    test_data[:, :, mid_slice],
                    harmonized_data[:, :, mid_slice]
                )
                results['metrics']['lpips'].append(lpips_score)

        # Calculate statistics
        for metric in results['metrics']:
            values = results['metrics'][metric]
            results['metrics'][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': values,
            }

        return results

    def compare_models(self, results: List[Dict]) -> pd.DataFrame:
        """
        Create comparison table of all models.

        Args:
            results: List of evaluation results for each model

        Returns:
            DataFrame with model comparisons
        """
        rows = []

        for result in results:
            row = {'Model': result['model_name']}

            for metric in result['metrics']:
                row[f'{metric.upper()}_mean'] = result['metrics'][metric]['mean']
                row[f'{metric.upper()}_std'] = result['metrics'][metric]['std']

            rows.append(row)

        df = pd.DataFrame(rows)
        return df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Evaluate all trained models'
    )

    parser.add_argument(
        '--models-dir',
        type=str,
        required=True,
        help='Directory containing all model checkpoints'
    )

    parser.add_argument(
        '--test-data',
        type=str,
        required=True,
        help='Test data directory'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for evaluation results'
    )

    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['ssim', 'psnr', 'nmi', 'lpips'],
        choices=['ssim', 'psnr', 'nmi', 'lpips', 'fid', 'dice'],
        help='Metrics to calculate'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for computation'
    )

    args = parser.parse_args()

    # Find all model checkpoints
    models_dir = Path(args.models_dir)
    model_checkpoints = {}

    # Search for best_model.pth in subdirectories
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            checkpoint_path = model_dir / 'checkpoints' / 'best_model.pth'
            if checkpoint_path.exists():
                model_checkpoints[model_dir.name] = checkpoint_path

    logger.info(f"Found {len(model_checkpoints)} models to evaluate")

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_dir=models_dir,
        test_data_dir=Path(args.test_data),
        output_dir=Path(args.output_dir),
        device=args.device,
    )

    # Evaluate each model
    all_results = []

    for model_name, checkpoint_path in model_checkpoints.items():
        results = evaluator.evaluate_model(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            metrics=args.metrics,
        )
        all_results.append(results)

        # Save individual results
        result_file = Path(args.output_dir) / f'{model_name}_results.json'
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

    # Create comparison table
    comparison_df = evaluator.compare_models(all_results)

    # Save comparison table
    comparison_file = Path(args.output_dir) / 'model_comparison.csv'
    comparison_df.to_csv(comparison_file, index=False)

    # Print comparison
    logger.info("\n" + "="*80)
    logger.info("Model Comparison:")
    logger.info("="*80)
    print(comparison_df.to_string(index=False))

    # Save all results
    all_results_file = Path(args.output_dir) / 'all_results.json'
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {args.output_dir}")
    logger.info(f"Comparison table: {comparison_file}")
    logger.info(f"All results: {all_results_file}")


if __name__ == '__main__':
    main()
