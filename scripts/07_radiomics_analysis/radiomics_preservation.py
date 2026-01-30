#!/usr/bin/env python3
"""
radiomics feature preservation analysis.

evaluates how well harmonization methods preserve clinically
relevant radiomics features while reducing domain shift.

metrics computed:
- pearson correlation coefficient
- concordance correlation coefficient (ccc)
- intraclass correlation coefficient (icc)
- bland-altman analysis

references:
- lin (1989): concordance correlation coefficient
- shrout & fleiss (1979): intraclass correlations
- bland & altman (1986): statistical methods for assessing agreement
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class PreservationMetrics:
    """container for feature preservation metrics."""
    pearson_r: float
    pearson_p: float
    ccc: float
    icc: float
    mean_diff: float
    std_diff: float
    loa_lower: float  # limits of agreement
    loa_upper: float


def compute_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """
    compute concordance correlation coefficient.

    measures the agreement between two variables, combining
    precision (pearson r) and accuracy (bias correction).

    args:
        x: first measurement array
        y: second measurement array

    returns:
        concordance correlation coefficient [-1, 1]
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    # remove nan values
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    if len(x) < 3:
        return 0.0

    # means and variances
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]

    # ccc formula
    numerator = 2 * cov_xy
    denominator = var_x + var_y + (mean_x - mean_y) ** 2

    if denominator < 1e-10:
        return 1.0 if numerator < 1e-10 else 0.0

    return float(numerator / denominator)


def compute_icc(x: np.ndarray, y: np.ndarray, icc_type: str = '2,1') -> float:
    """
    compute intraclass correlation coefficient.

    icc(2,1): two-way random effects, single measures, absolute agreement

    args:
        x: first measurement array
        y: second measurement array
        icc_type: type of icc ('2,1' for absolute agreement)

    returns:
        icc value [0, 1]
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    # remove nan values
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    n = len(x)
    if n < 3:
        return 0.0

    # combine into matrix (n subjects x 2 raters)
    data = np.column_stack([x, y])
    k = 2  # number of raters

    # anova components
    grand_mean = np.mean(data)
    subject_means = np.mean(data, axis=1)
    rater_means = np.mean(data, axis=0)

    # sum of squares
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_subjects = k * np.sum((subject_means - grand_mean) ** 2)
    ss_raters = n * np.sum((rater_means - grand_mean) ** 2)
    ss_error = ss_total - ss_subjects - ss_raters

    # mean squares
    ms_subjects = ss_subjects / (n - 1)
    ms_raters = ss_raters / (k - 1) if k > 1 else 0
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 1

    # icc(2,1) formula
    numerator = ms_subjects - ms_error
    denominator = ms_subjects + (k - 1) * ms_error + (k / n) * (ms_raters - ms_error)

    if denominator < 1e-10:
        return 1.0 if abs(numerator) < 1e-10 else 0.0

    icc = numerator / denominator
    return float(np.clip(icc, 0, 1))


def bland_altman_analysis(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """
    perform bland-altman analysis for method agreement.

    args:
        x: first measurement (reference)
        y: second measurement (test)

    returns:
        (mean_diff, std_diff, loa_lower, loa_upper)
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    # remove nan values
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    if len(x) < 3:
        return 0.0, 0.0, 0.0, 0.0

    # differences and means
    diff = y - x
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    # 95% limits of agreement
    loa_lower = mean_diff - 1.96 * std_diff
    loa_upper = mean_diff + 1.96 * std_diff

    return float(mean_diff), float(std_diff), float(loa_lower), float(loa_upper)


def compute_preservation_metrics(
    original: np.ndarray,
    harmonized: np.ndarray
) -> PreservationMetrics:
    """
    compute all preservation metrics between original and harmonized features.

    args:
        original: original feature values
        harmonized: harmonized feature values

    returns:
        PreservationMetrics dataclass
    """
    original = np.asarray(original).flatten()
    harmonized = np.asarray(harmonized).flatten()

    # remove invalid values
    valid = ~(np.isnan(original) | np.isnan(harmonized) |
              np.isinf(original) | np.isinf(harmonized))
    original = original[valid]
    harmonized = harmonized[valid]

    if len(original) < 3:
        return PreservationMetrics(
            pearson_r=0.0, pearson_p=1.0, ccc=0.0, icc=0.0,
            mean_diff=0.0, std_diff=0.0, loa_lower=0.0, loa_upper=0.0
        )

    # pearson correlation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pearson_r, pearson_p = stats.pearsonr(original, harmonized)

    # concordance correlation
    ccc = compute_ccc(original, harmonized)

    # intraclass correlation
    icc = compute_icc(original, harmonized)

    # bland-altman
    mean_diff, std_diff, loa_lower, loa_upper = bland_altman_analysis(original, harmonized)

    return PreservationMetrics(
        pearson_r=float(pearson_r) if not np.isnan(pearson_r) else 0.0,
        pearson_p=float(pearson_p) if not np.isnan(pearson_p) else 1.0,
        ccc=ccc,
        icc=icc,
        mean_diff=mean_diff,
        std_diff=std_diff,
        loa_lower=loa_lower,
        loa_upper=loa_upper
    )


class RadiomicsPreservationAnalyzer:
    """
    analyzer for radiomics feature preservation across harmonization.
    """

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.feature_categories = self._categorize_features(feature_names)

    def _categorize_features(self, names: List[str]) -> Dict[str, List[int]]:
        """categorize features by type."""
        categories = {
            'first_order': [],
            'glcm': [],
            'shape': [],
            'other': []
        }

        for i, name in enumerate(names):
            if 'fo_' in name or 'first' in name.lower():
                categories['first_order'].append(i)
            elif 'glcm' in name.lower():
                categories['glcm'].append(i)
            elif 'shape' in name.lower():
                categories['shape'].append(i)
            else:
                categories['other'].append(i)

        return categories

    def analyze_preservation(
        self,
        original_features: np.ndarray,
        harmonized_features: np.ndarray
    ) -> Dict:
        """
        analyze feature preservation for all features.

        args:
            original_features: (n_samples, n_features) array
            harmonized_features: (n_samples, n_features) array

        returns:
            dictionary of preservation metrics by feature and category
        """
        n_samples, n_features = original_features.shape
        assert harmonized_features.shape == (n_samples, n_features)

        results = {
            'per_feature': {},
            'per_category': {},
            'overall': {}
        }

        # per-feature metrics
        all_cccs = []
        all_iccs = []
        all_correlations = []

        for i, name in enumerate(self.feature_names):
            metrics = compute_preservation_metrics(
                original_features[:, i],
                harmonized_features[:, i]
            )
            results['per_feature'][name] = {
                'pearson_r': metrics.pearson_r,
                'pearson_p': metrics.pearson_p,
                'ccc': metrics.ccc,
                'icc': metrics.icc,
                'mean_diff': metrics.mean_diff,
                'std_diff': metrics.std_diff,
                'loa_lower': metrics.loa_lower,
                'loa_upper': metrics.loa_upper
            }

            if not np.isnan(metrics.ccc):
                all_cccs.append(metrics.ccc)
            if not np.isnan(metrics.icc):
                all_iccs.append(metrics.icc)
            if not np.isnan(metrics.pearson_r):
                all_correlations.append(metrics.pearson_r)

        # per-category metrics
        for category, indices in self.feature_categories.items():
            if not indices:
                continue

            cat_cccs = []
            cat_iccs = []
            cat_correlations = []

            for i in indices:
                metrics = results['per_feature'][self.feature_names[i]]
                if not np.isnan(metrics['ccc']):
                    cat_cccs.append(metrics['ccc'])
                if not np.isnan(metrics['icc']):
                    cat_iccs.append(metrics['icc'])
                if not np.isnan(metrics['pearson_r']):
                    cat_correlations.append(metrics['pearson_r'])

            results['per_category'][category] = {
                'n_features': len(indices),
                'mean_ccc': float(np.mean(cat_cccs)) if cat_cccs else 0.0,
                'std_ccc': float(np.std(cat_cccs)) if cat_cccs else 0.0,
                'mean_icc': float(np.mean(cat_iccs)) if cat_iccs else 0.0,
                'std_icc': float(np.std(cat_iccs)) if cat_iccs else 0.0,
                'mean_correlation': float(np.mean(cat_correlations)) if cat_correlations else 0.0,
                'std_correlation': float(np.std(cat_correlations)) if cat_correlations else 0.0,
            }

        # overall metrics
        results['overall'] = {
            'n_features': n_features,
            'n_samples': n_samples,
            'mean_ccc': float(np.mean(all_cccs)) if all_cccs else 0.0,
            'std_ccc': float(np.std(all_cccs)) if all_cccs else 0.0,
            'mean_icc': float(np.mean(all_iccs)) if all_iccs else 0.0,
            'std_icc': float(np.std(all_iccs)) if all_iccs else 0.0,
            'mean_correlation': float(np.mean(all_correlations)) if all_correlations else 0.0,
            'std_correlation': float(np.std(all_correlations)) if all_correlations else 0.0,
            'excellent_preservation': sum(1 for c in all_cccs if c > 0.9),
            'good_preservation': sum(1 for c in all_cccs if 0.75 <= c <= 0.9),
            'moderate_preservation': sum(1 for c in all_cccs if 0.5 <= c < 0.75),
            'poor_preservation': sum(1 for c in all_cccs if c < 0.5),
        }

        return results


def compare_methods_preservation(
    original_a: np.ndarray,
    original_b: np.ndarray,
    harmonized_a: np.ndarray,
    harmonized_b: np.ndarray,
    feature_names: List[str],
    method_name: str = 'sa-cyclegan'
) -> Dict:
    """
    compare feature preservation between domains after harmonization.

    for harmonization, we want features to be:
    1. preserved within each domain (original vs harmonized)
    2. aligned between domains (harmonized_a vs harmonized_b)

    args:
        original_a: original features from domain a
        original_b: original features from domain b
        harmonized_a: harmonized features from domain a
        harmonized_b: harmonized features from domain b
        feature_names: list of feature names
        method_name: name of harmonization method

    returns:
        comprehensive comparison results
    """
    analyzer = RadiomicsPreservationAnalyzer(feature_names)

    results = {
        'method': method_name,
        'domain_a_preservation': analyzer.analyze_preservation(original_a, harmonized_a),
        'domain_b_preservation': analyzer.analyze_preservation(original_b, harmonized_b),
    }

    # cross-domain alignment before harmonization
    # sample to match sizes
    n_common = min(len(original_a), len(original_b))
    results['cross_domain_raw'] = analyzer.analyze_preservation(
        original_a[:n_common],
        original_b[:n_common]
    )

    # cross-domain alignment after harmonization
    results['cross_domain_harmonized'] = analyzer.analyze_preservation(
        harmonized_a[:n_common],
        harmonized_b[:n_common]
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description='radiomics feature preservation analysis'
    )
    parser.add_argument('--original-a', type=str, required=True,
                       help='path to original domain a features')
    parser.add_argument('--original-b', type=str, required=True,
                       help='path to original domain b features')
    parser.add_argument('--harmonized-a', type=str, required=True,
                       help='path to harmonized domain a features')
    parser.add_argument('--harmonized-b', type=str, required=True,
                       help='path to harmonized domain b features')
    parser.add_argument('--feature-names', type=str,
                       help='path to feature names json')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='output directory')
    parser.add_argument('--method-name', type=str, default='sa-cyclegan',
                       help='name of harmonization method')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'[preservation] loading features...')

    original_a = np.load(args.original_a)
    original_b = np.load(args.original_b)
    harmonized_a = np.load(args.harmonized_a)
    harmonized_b = np.load(args.harmonized_b)

    print(f'[preservation] original a: {original_a.shape}')
    print(f'[preservation] original b: {original_b.shape}')
    print(f'[preservation] harmonized a: {harmonized_a.shape}')
    print(f'[preservation] harmonized b: {harmonized_b.shape}')

    # load or generate feature names
    if args.feature_names and Path(args.feature_names).exists():
        with open(args.feature_names) as f:
            feature_names = json.load(f)
    else:
        feature_names = [f'feature_{i}' for i in range(original_a.shape[1])]

    # run analysis
    print(f'[preservation] analyzing feature preservation...')
    results = compare_methods_preservation(
        original_a, original_b,
        harmonized_a, harmonized_b,
        feature_names,
        args.method_name
    )

    # save results
    with open(output_dir / 'preservation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # print summary
    print('=' * 60)
    print(f'[preservation] {args.method_name} results:')
    print('=' * 60)
    print(f"  domain a preservation:")
    print(f"    mean ccc: {results['domain_a_preservation']['overall']['mean_ccc']:.4f}")
    print(f"    mean icc: {results['domain_a_preservation']['overall']['mean_icc']:.4f}")
    print(f"  domain b preservation:")
    print(f"    mean ccc: {results['domain_b_preservation']['overall']['mean_ccc']:.4f}")
    print(f"    mean icc: {results['domain_b_preservation']['overall']['mean_icc']:.4f}")
    print(f"  cross-domain alignment (raw):")
    print(f"    mean ccc: {results['cross_domain_raw']['overall']['mean_ccc']:.4f}")
    print(f"  cross-domain alignment (harmonized):")
    print(f"    mean ccc: {results['cross_domain_harmonized']['overall']['mean_ccc']:.4f}")
    print('=' * 60)


if __name__ == '__main__':
    main()
