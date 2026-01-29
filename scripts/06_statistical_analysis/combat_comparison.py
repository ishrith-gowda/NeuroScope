#!/usr/bin/env python3
"""
combat harmonization baseline for comparison with sa-cyclegan.

implements neuroCombat statistical harmonization method as a baseline
for comparing against deep learning-based harmonization.

combat (combating batch effects when combining batches) is a widely-used
statistical method for harmonization in neuroimaging studies.

references:
- johnson et al. (2007): adjusting batch effects in microarray data
- fortin et al. (2017): harmonization of cortical thickness using combat
- fortin et al. (2018): harmonization of multi-site diffusion mri
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
from scipy import stats
import warnings

# add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class ComBatConfig:
    """configuration for combat harmonization."""
    parametric: bool = True  # use parametric adjustments
    eb: bool = True  # use empirical bayes estimation
    mean_only: bool = False  # only adjust means (not variances)
    ref_batch: Optional[str] = None  # reference batch for adjustment


class ComBatHarmonizer:
    """
    combat harmonization implementation.

    uses empirical bayes framework to estimate and remove batch effects
    while preserving biological variability.
    """

    def __init__(self, config: Optional[ComBatConfig] = None):
        self.config = config or ComBatConfig()
        self.gamma_hat = None  # batch mean adjustments
        self.delta_hat = None  # batch variance adjustments
        self.gamma_star = None  # eb-estimated means
        self.delta_star = None  # eb-estimated variances
        self.grand_mean = None
        self.var_pooled = None
        self.batch_info = None
        self.fitted = False

    def _standardize_across_features(
        self,
        data: np.ndarray,
        batch_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        standardize data across features.

        returns standardized data, feature means, and feature variances.
        """
        n_samples, n_features = data.shape

        # compute batch info
        unique_batches = np.unique(batch_labels)
        n_batches = len(unique_batches)

        # compute grand mean (weighted by batch size)
        grand_mean = np.zeros(n_features)
        batch_counts = {}
        batch_means = {}

        for batch in unique_batches:
            mask = batch_labels == batch
            batch_counts[batch] = np.sum(mask)
            batch_means[batch] = np.mean(data[mask], axis=0)
            grand_mean += batch_counts[batch] * batch_means[batch]

        grand_mean /= n_samples

        # compute pooled variance
        var_pooled = np.zeros(n_features)
        for batch in unique_batches:
            mask = batch_labels == batch
            batch_data = data[mask]
            var_pooled += np.sum((batch_data - batch_means[batch]) ** 2, axis=0)

        var_pooled /= (n_samples - n_batches)
        var_pooled = np.maximum(var_pooled, 1e-10)  # prevent division by zero

        # standardize
        data_std = (data - grand_mean) / np.sqrt(var_pooled)

        return data_std, grand_mean, var_pooled

    def _compute_batch_effects(
        self,
        data_std: np.ndarray,
        batch_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        compute batch-specific location (gamma) and scale (delta) parameters.
        """
        unique_batches = np.unique(batch_labels)
        n_batches = len(unique_batches)
        n_features = data_std.shape[1]

        gamma_hat = np.zeros((n_batches, n_features))
        delta_hat = np.zeros((n_batches, n_features))

        for i, batch in enumerate(unique_batches):
            mask = batch_labels == batch
            batch_data = data_std[mask]

            gamma_hat[i] = np.mean(batch_data, axis=0)
            delta_hat[i] = np.var(batch_data, axis=0, ddof=1)

        return gamma_hat, delta_hat

    def _empirical_bayes_estimate(
        self,
        gamma_hat: np.ndarray,
        delta_hat: np.ndarray,
        batch_counts: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        compute empirical bayes estimates for batch effects.

        shrinks batch-specific estimates toward overall mean using
        prior distribution estimated from data.
        """
        n_batches, n_features = gamma_hat.shape

        # prior for gamma (location)
        gamma_bar = np.mean(gamma_hat, axis=0)
        tau_sq = np.var(gamma_hat, axis=0, ddof=1)

        # prior for delta (scale) - inverse gamma
        delta_bar = np.mean(delta_hat, axis=0)
        lambda_bar = np.mean(delta_hat, axis=0)
        theta_bar = np.var(delta_hat, axis=0, ddof=1)

        # eb estimates
        gamma_star = np.zeros_like(gamma_hat)
        delta_star = np.zeros_like(delta_hat)

        batch_list = list(batch_counts.keys())

        for i in range(n_batches):
            n_i = batch_counts[batch_list[i]]

            # posterior mean for gamma
            gamma_star[i] = (
                (n_i * tau_sq * gamma_hat[i] + delta_hat[i] * gamma_bar) /
                (n_i * tau_sq + delta_hat[i] + 1e-10)
            )

            # posterior mean for delta (using moment matching)
            if self.config.mean_only:
                delta_star[i] = np.ones(n_features)
            else:
                # simplified eb estimate
                delta_star[i] = (
                    (theta_bar + delta_hat[i] * (n_i - 1)) /
                    (theta_bar / lambda_bar + n_i - 1 + 1e-10)
                )

        return gamma_star, delta_star

    def fit(
        self,
        data: np.ndarray,
        batch_labels: np.ndarray
    ) -> 'ComBatHarmonizer':
        """
        fit combat model to data.

        args:
            data: array of shape (n_samples, n_features)
            batch_labels: array of batch labels for each sample
        """
        # standardize
        data_std, grand_mean, var_pooled = self._standardize_across_features(
            data, batch_labels
        )

        # compute batch effects
        gamma_hat, delta_hat = self._compute_batch_effects(data_std, batch_labels)

        # batch info
        unique_batches = np.unique(batch_labels)
        batch_counts = {b: np.sum(batch_labels == b) for b in unique_batches}

        # empirical bayes estimation
        if self.config.eb:
            gamma_star, delta_star = self._empirical_bayes_estimate(
                gamma_hat, delta_hat, batch_counts
            )
        else:
            gamma_star = gamma_hat
            delta_star = delta_hat

        # store
        self.gamma_hat = gamma_hat
        self.delta_hat = delta_hat
        self.gamma_star = gamma_star
        self.delta_star = delta_star
        self.grand_mean = grand_mean
        self.var_pooled = var_pooled
        self.batch_info = {
            'unique_batches': list(unique_batches),
            'batch_counts': batch_counts
        }
        self.fitted = True

        return self

    def transform(
        self,
        data: np.ndarray,
        batch_labels: np.ndarray
    ) -> np.ndarray:
        """
        apply combat harmonization to data.

        args:
            data: array of shape (n_samples, n_features)
            batch_labels: array of batch labels for each sample

        returns:
            harmonized data of same shape
        """
        if not self.fitted:
            raise RuntimeError("combat model must be fitted before transform")

        # standardize with fitted parameters
        data_std = (data - self.grand_mean) / np.sqrt(self.var_pooled)

        # apply batch corrections
        data_combat = np.zeros_like(data_std)
        unique_batches = self.batch_info['unique_batches']

        for i, batch in enumerate(unique_batches):
            mask = batch_labels == batch
            if not np.any(mask):
                continue

            # remove batch effect
            batch_data = data_std[mask]

            if self.config.mean_only:
                data_combat[mask] = batch_data - self.gamma_star[i]
            else:
                data_combat[mask] = (
                    (batch_data - self.gamma_star[i]) /
                    np.sqrt(self.delta_star[i] + 1e-10)
                )

        # transform back to original scale
        data_harmonized = data_combat * np.sqrt(self.var_pooled) + self.grand_mean

        return data_harmonized

    def fit_transform(
        self,
        data: np.ndarray,
        batch_labels: np.ndarray
    ) -> np.ndarray:
        """fit and transform in one step."""
        return self.fit(data, batch_labels).transform(data, batch_labels)


def harmonize_mri_with_combat(
    domain_a_data: np.ndarray,
    domain_b_data: np.ndarray,
    config: Optional[ComBatConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    harmonize two mri domains using combat.

    args:
        domain_a_data: array of shape (n_a, n_features) for domain a
        domain_b_data: array of shape (n_b, n_features) for domain b
        config: combat configuration

    returns:
        (harmonized_a, harmonized_b) tuple
    """
    # combine data
    combined_data = np.vstack([domain_a_data, domain_b_data])

    # create batch labels
    batch_labels = np.array(
        ['A'] * len(domain_a_data) + ['B'] * len(domain_b_data)
    )

    # fit and transform
    combat = ComBatHarmonizer(config)
    harmonized = combat.fit_transform(combined_data, batch_labels)

    # split back
    n_a = len(domain_a_data)
    harmonized_a = harmonized[:n_a]
    harmonized_b = harmonized[n_a:]

    return harmonized_a, harmonized_b


def evaluate_combat_harmonization(
    raw_features_a: np.ndarray,
    raw_features_b: np.ndarray,
    output_dir: Path
) -> Dict:
    """
    evaluate combat harmonization and compare with deep learning methods.
    """
    print('[combat] fitting combat model...')

    # apply combat
    config = ComBatConfig(parametric=True, eb=True)
    harmonized_a, harmonized_b = harmonize_mri_with_combat(
        raw_features_a, raw_features_b, config
    )

    print('[combat] computing metrics...')

    # compute distribution metrics before/after
    def compute_mmd(x, y, sigma=1.0):
        """compute maximum mean discrepancy."""
        n_samples = min(500, len(x), len(y))
        idx_x = np.random.choice(len(x), n_samples, replace=False)
        idx_y = np.random.choice(len(y), n_samples, replace=False)
        x_sub, y_sub = x[idx_x], y[idx_y]

        def rbf_kernel(a, b):
            diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
            return np.exp(-np.sum(diff**2, axis=2) / (2 * sigma**2))

        k_xx = rbf_kernel(x_sub, x_sub).mean()
        k_yy = rbf_kernel(y_sub, y_sub).mean()
        k_xy = rbf_kernel(x_sub, y_sub).mean()

        return k_xx + k_yy - 2 * k_xy

    def compute_mean_diff(x, y):
        """compute mean difference between distributions."""
        return np.linalg.norm(np.mean(x, axis=0) - np.mean(y, axis=0))

    def compute_cosine_sim(x, y):
        """compute cosine similarity of distribution means."""
        mu_x = np.mean(x, axis=0)
        mu_y = np.mean(y, axis=0)
        return np.dot(mu_x, mu_y) / (np.linalg.norm(mu_x) * np.linalg.norm(mu_y) + 1e-10)

    # metrics before combat
    raw_mmd = compute_mmd(raw_features_a, raw_features_b)
    raw_mean_diff = compute_mean_diff(raw_features_a, raw_features_b)
    raw_cosine = compute_cosine_sim(raw_features_a, raw_features_b)

    # metrics after combat
    combat_mmd = compute_mmd(harmonized_a, harmonized_b)
    combat_mean_diff = compute_mean_diff(harmonized_a, harmonized_b)
    combat_cosine = compute_cosine_sim(harmonized_a, harmonized_b)

    results = {
        'raw': {
            'mmd': float(raw_mmd),
            'mean_difference': float(raw_mean_diff),
            'cosine_similarity': float(raw_cosine)
        },
        'combat': {
            'mmd': float(combat_mmd),
            'mean_difference': float(combat_mean_diff),
            'cosine_similarity': float(combat_cosine)
        },
        'improvement': {
            'mmd_reduction': float(raw_mmd - combat_mmd),
            'mmd_reduction_percent': float(100 * (raw_mmd - combat_mmd) / (raw_mmd + 1e-10)),
            'mean_diff_reduction': float(raw_mean_diff - combat_mean_diff),
            'cosine_improvement': float(combat_cosine - raw_cosine)
        }
    }

    # save harmonized features
    np.save(output_dir / 'combat_harmonized_a.npy', harmonized_a)
    np.save(output_dir / 'combat_harmonized_b.npy', harmonized_b)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='combat harmonization baseline comparison'
    )
    parser.add_argument('--features-a', type=str, required=True,
                       help='path to domain a features (npy file)')
    parser.add_argument('--features-b', type=str, required=True,
                       help='path to domain b features (npy file)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='output directory')
    parser.add_argument('--parametric', action='store_true', default=True,
                       help='use parametric combat')
    parser.add_argument('--no-eb', action='store_true',
                       help='disable empirical bayes')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('[combat] loading features...')
    features_a = np.load(args.features_a)
    features_b = np.load(args.features_b)

    print(f'[combat] domain a: {features_a.shape}')
    print(f'[combat] domain b: {features_b.shape}')

    # run evaluation
    results = evaluate_combat_harmonization(
        features_a, features_b, output_dir
    )

    # save results
    with open(output_dir / 'combat_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('=' * 60)
    print('[combat] results summary:')
    print(f"  raw mmd: {results['raw']['mmd']:.6f}")
    print(f"  combat mmd: {results['combat']['mmd']:.6f}")
    print(f"  mmd reduction: {results['improvement']['mmd_reduction_percent']:.1f}%")
    print(f"  raw cosine: {results['raw']['cosine_similarity']:.4f}")
    print(f"  combat cosine: {results['combat']['cosine_similarity']:.4f}")
    print('=' * 60)
    print(f'[combat] results saved to {output_dir}')


if __name__ == '__main__':
    main()
