#!/usr/bin/env python3
"""
ComBat Harmonization Baseline.

Implements the ComBat statistical harmonization method as a baseline
for comparison with deep learning approaches.

ComBat (Empirical Bayes) harmonization removes batch effects while
preserving biological variation using location and scale adjustments.

Reference:
    Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects
    in microarray expression data using empirical Bayes methods. Biostatistics.

Usage:
    python scripts/run_combat_baseline.py \
        --source-dir ./data/processed/brats \
        --target-dir ./data/processed/upenn_gbm \
        --output-dir ./experiments/combat_baseline \
        --modalities T1 T1CE T2 FLAIR
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComBatHarmonizer:
    """
    ComBat harmonization using Empirical Bayes.

    Removes batch effects from multi-site neuroimaging data while
    preserving biological variation.
    """

    def __init__(
        self,
        parametric: bool = True,
        eb: bool = True,
        ref_batch: Optional[int] = None,
    ):
        """
        Initialize ComBat harmonizer.

        Args:
            parametric: Use parametric adjustments (True) or non-parametric (False)
            eb: Use Empirical Bayes for parameter estimation
            ref_batch: Reference batch (if None, uses grand mean)
        """
        self.parametric = parametric
        self.eb = eb
        self.ref_batch = ref_batch
        self.batch_info = None
        self.stand_mean = None

    def _standardize_across_features(
        self,
        data: np.ndarray,
        batch: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Standardize data across features.

        Args:
            data: Data matrix (n_samples x n_features)
            batch: Batch labels (n_samples,)

        Returns:
            Standardized data and standardization parameters
        """
        n_batch = len(np.unique(batch))
        n_samples, n_features = data.shape

        # Calculate batch-specific means and variances
        batch_design = np.zeros((n_samples, n_batch))
        for i, b in enumerate(np.unique(batch)):
            batch_design[batch == b, i] = 1

        # Grand mean
        grand_mean = np.mean(data, axis=0)

        # Variance pooled across batches
        var_pooled = np.var(data, axis=0)

        # Standardize
        stand_mean = np.dot(batch_design.T, data) / np.sum(batch_design, axis=0)[:, np.newaxis]

        # Compute batch-specific statistics
        batch_info = {}
        for i, b in enumerate(np.unique(batch)):
            batch_mask = batch == b
            batch_data = data[batch_mask]

            batch_info[b] = {
                'mean': np.mean(batch_data, axis=0),
                'var': np.var(batch_data, axis=0, ddof=1),
                'n_samples': np.sum(batch_mask),
            }

        return stand_mean, batch_info, grand_mean, var_pooled

    def _aprior(self, gamma_hat: np.ndarray) -> float:
        """Compute prior variance for gamma (location shift)."""
        m = np.mean(gamma_hat)
        s2 = np.var(gamma_hat, ddof=1)
        return (2 * s2 + m ** 2) / s2

    def _bprior(self, gamma_hat: np.ndarray) -> float:
        """Compute prior mean for gamma."""
        m = np.mean(gamma_hat)
        s2 = np.var(gamma_hat, ddof=1)
        return (m * s2 + m ** 3) / s2

    def _it_sol(
        self,
        sdat: np.ndarray,
        g_hat: np.ndarray,
        d_hat: np.ndarray,
        g_bar: float,
        t2: float,
        a: float,
        b: float,
        conv: float = 0.0001
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Iteratively compute posterior estimates.

        Args:
            sdat: Standardized data
            g_hat: Initial gamma estimates
            d_hat: Initial delta estimates
            g_bar: Prior mean for gamma
            t2: Prior variance for gamma
            a: Prior shape parameter for delta
            b: Prior scale parameter for delta
            conv: Convergence threshold

        Returns:
            Posterior gamma and delta estimates
        """
        n = sdat.shape[0]
        g_old = g_hat.copy()
        d_old = d_hat.copy()

        change = 1
        count = 0

        while change > conv:
            # Update gamma
            g_new = self._postmean(g_hat, g_bar, n, d_old, t2)

            # Update delta
            sum2 = np.sum((sdat - np.outer(g_new, np.ones(n))) ** 2, axis=1)
            d_new = self._postvar(sum2, n, a, b)

            # Check convergence
            change = max(
                np.max(np.abs(g_new - g_old) / g_old),
                np.max(np.abs(d_new - d_old) / d_old)
            )

            g_old = g_new.copy()
            d_old = d_new.copy()
            count += 1

            if count > 1000:
                logger.warning("Maximum iterations reached in iterative solution")
                break

        return g_new, d_new

    def _postmean(
        self,
        g_hat: np.ndarray,
        g_bar: float,
        n: int,
        d_star: np.ndarray,
        t2: float
    ) -> np.ndarray:
        """Compute posterior mean for gamma."""
        return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)

    def _postvar(
        self,
        sum2: np.ndarray,
        n: int,
        a: float,
        b: float
    ) -> np.ndarray:
        """Compute posterior variance for delta."""
        return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

    def fit_transform(
        self,
        data: np.ndarray,
        batch: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fit ComBat harmonization and transform data.

        Args:
            data: Data matrix (n_samples x n_features)
            batch: Batch labels (n_samples,)
            covariates: Optional covariates to preserve (n_samples x n_covariates)

        Returns:
            Harmonized data
        """
        logger.info("Fitting ComBat harmonization...")

        # Standardize data
        stand_mean, batch_info, grand_mean, var_pooled = \
            self._standardize_across_features(data, batch)

        n_batch = len(batch_info)
        n_features = data.shape[1]

        # Compute batch effects
        gamma_hat = np.zeros((n_batch, n_features))
        delta_hat = np.zeros((n_batch, n_features))

        for i, b in enumerate(batch_info.keys()):
            batch_data = data[batch == b]
            gamma_hat[i] = batch_info[b]['mean'] - grand_mean
            delta_hat[i] = batch_info[b]['var']

        # Empirical Bayes estimation
        if self.eb:
            logger.info("Computing empirical Bayes estimates...")

            gamma_star = np.zeros_like(gamma_hat)
            delta_star = np.zeros_like(delta_hat)

            for j in range(n_features):
                # Prior parameters for gamma
                g_bar = np.mean(gamma_hat[:, j])
                t2 = np.var(gamma_hat[:, j], ddof=1)

                # Prior parameters for delta
                a = self._aprior(delta_hat[:, j])
                b = self._bprior(delta_hat[:, j])

                # Posterior estimates
                if self.parametric:
                    # Parametric adjustment
                    gamma_star[:, j] = self._postmean(
                        gamma_hat[:, j],
                        g_bar,
                        np.array([batch_info[b]['n_samples'] for b in batch_info.keys()]).mean(),
                        delta_hat[:, j],
                        t2
                    )

                    delta_star[:, j] = self._postvar(
                        np.array([batch_info[b]['var'][j] * batch_info[b]['n_samples']
                                  for b in batch_info.keys()]),
                        np.array([batch_info[b]['n_samples'] for b in batch_info.keys()]).mean(),
                        a,
                        b
                    )
                else:
                    # Non-parametric adjustment
                    gamma_star[:, j] = gamma_hat[:, j]
                    delta_star[:, j] = delta_hat[:, j]
        else:
            gamma_star = gamma_hat
            delta_star = delta_hat

        # Apply harmonization
        logger.info("Applying harmonization...")
        harmonized = data.copy()

        for i, b in enumerate(batch_info.keys()):
            batch_mask = batch == b

            # Adjust location and scale
            harmonized[batch_mask] = (
                (data[batch_mask] - gamma_star[i]) /
                np.sqrt(delta_star[i]) *
                np.sqrt(var_pooled) +
                grand_mean
            )

        return harmonized

    def harmonize_volumes(
        self,
        source_files: List[Path],
        target_files: List[Path],
        output_dir: Path,
        modality: str,
    ) -> Dict:
        """
        Harmonize a set of volumetric MRI scans.

        Args:
            source_files: List of source domain files
            target_files: List of target domain files
            output_dir: Output directory for harmonized scans
            modality: MRI modality name

        Returns:
            Dictionary with harmonization statistics
        """
        logger.info(f"Harmonizing {modality} scans...")
        logger.info(f"Source: {len(source_files)} scans")
        logger.info(f"Target: {len(target_files)} scans")

        # Load all volumes
        all_files = source_files + target_files
        all_data = []
        all_affines = []

        for filepath in tqdm(all_files, desc="Loading volumes"):
            img = nib.load(str(filepath))
            data = img.get_fdata()
            all_data.append(data.flatten())
            all_affines.append(img.affine)

        # Create batch labels
        batch = np.array(
            [0] * len(source_files) + [1] * len(target_files)
        )

        # Stack data
        data_matrix = np.stack(all_data, axis=0)

        # Harmonize
        harmonized_matrix = self.fit_transform(data_matrix, batch)

        # Save harmonized volumes
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = {'modality': modality, 'n_source': len(source_files), 'n_target': len(target_files)}

        # Get original shape from first volume
        original_shape = nib.load(str(all_files[0])).shape

        for i, filepath in enumerate(tqdm(all_files, desc="Saving harmonized volumes")):
            # Reshape to original volume shape
            harmonized_volume = harmonized_matrix[i].reshape(original_shape)

            # Determine output path
            if i < len(source_files):
                subject_id = filepath.parent.name
                output_path = output_dir / 'source' / subject_id / f"{modality}_harmonized.nii.gz"
            else:
                subject_id = filepath.parent.name
                output_path = output_dir / 'target' / subject_id / f"{modality}_harmonized.nii.gz"

            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img = nib.Nifti1Image(harmonized_volume.astype(np.float32), all_affines[i])
            nib.save(img, str(output_path))

        logger.info(f"Harmonization complete for {modality}")
        return stats


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run ComBat harmonization baseline'
    )

    parser.add_argument(
        '--source-dir',
        type=str,
        required=True,
        help='Source dataset directory'
    )

    parser.add_argument(
        '--target-dir',
        type=str,
        required=True,
        help='Target dataset directory'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for harmonized data'
    )

    parser.add_argument(
        '--modalities',
        nargs='+',
        default=['T1', 'T1CE', 'T2', 'FLAIR'],
        help='MRI modalities to harmonize'
    )

    parser.add_argument(
        '--parametric',
        action='store_true',
        default=True,
        help='Use parametric adjustments'
    )

    parser.add_argument(
        '--no-eb',
        action='store_true',
        help='Disable Empirical Bayes estimation'
    )

    args = parser.parse_args()

    # Initialize harmonizer
    harmonizer = ComBatHarmonizer(
        parametric=args.parametric,
        eb=not args.no_eb,
    )

    # Process each modality
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    output_dir = Path(args.output_dir)

    all_stats = []

    for modality in args.modalities:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {modality}")
        logger.info(f"{'='*80}\n")

        # Find all files for this modality
        source_files = sorted(source_dir.rglob(f"*/{modality}.nii.gz"))
        target_files = sorted(target_dir.rglob(f"*/{modality}.nii.gz"))

        if not source_files or not target_files:
            logger.warning(f"No files found for {modality}. Skipping.")
            continue

        # Harmonize
        stats = harmonizer.harmonize_volumes(
            source_files=source_files,
            target_files=target_files,
            output_dir=output_dir,
            modality=modality,
        )

        all_stats.append(stats)

    # Save statistics
    stats_file = output_dir / 'combat_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info("ComBat harmonization complete!")
    logger.info(f"Harmonized data saved to: {output_dir}")
    logger.info(f"Statistics saved to: {stats_file}")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
