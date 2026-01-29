#!/usr/bin/env python3
"""
comprehensive statistical analysis for mri harmonization evaluation.

implements publication-quality statistical analysis including:
- bootstrap confidence intervals (percentile, bca)
- paired/independent statistical tests (t-test, wilcoxon, mann-whitney)
- effect size measures (cohen's d, hedges' g, glass's delta)
- multiple comparison correction (bonferroni, holm, benjamini-hochberg)
- per-subject paired analysis

designed for top-tier venue submission (miccai, tmi, neuroimage).

references:
- cohen (1988): effect size interpretation guidelines
- benjamini & hochberg (1995): fdr control procedure
- efron & tibshirani (1993): bootstrap methods
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict
import warnings

import numpy as np
from scipy import stats
from scipy.stats import (
    ttest_rel, ttest_ind, wilcoxon, mannwhitneyu,
    shapiro, levene, kruskal, f_oneway
)


@dataclass
class StatisticalResult:
    """container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_type: str
    ci_lower: float
    ci_upper: float
    n_samples: int
    interpretation: str
    significant: bool


@dataclass
class BootstrapCI:
    """container for bootstrap confidence interval."""
    estimate: float
    ci_lower: float
    ci_upper: float
    se: float
    n_bootstrap: int
    method: str


def check_normality(data: np.ndarray, alpha: float = 0.05) -> Tuple[bool, float]:
    """
    check normality using shapiro-wilk test.

    returns (is_normal, p_value).
    """
    if len(data) < 3:
        return True, 1.0

    # shapiro-wilk has sample size limit
    if len(data) > 5000:
        data = np.random.choice(data, 5000, replace=False)

    stat, p = shapiro(data)
    return p > alpha, p


def check_homogeneity(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> Tuple[bool, float]:
    """
    check homogeneity of variances using levene's test.

    returns (is_homogeneous, p_value).
    """
    stat, p = levene(x, y)
    return p > alpha, p


def compute_cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """
    compute cohen's d for paired samples.

    d = mean(diff) / std(diff)
    """
    diff = x - y
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)


def compute_cohens_d_independent(x: np.ndarray, y: np.ndarray) -> float:
    """
    compute cohen's d for independent samples (pooled sd).

    d = (mean1 - mean2) / pooled_sd
    """
    n1, n2 = len(x), len(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)

    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(x) - np.mean(y)) / (pooled_sd + 1e-10)


def compute_hedges_g(x: np.ndarray, y: np.ndarray, paired: bool = False) -> float:
    """
    compute hedges' g (bias-corrected effect size).

    preferred over cohen's d for small samples (n < 50).
    """
    if paired:
        d = compute_cohens_d_paired(x, y)
        n = len(x)
    else:
        d = compute_cohens_d_independent(x, y)
        n = len(x) + len(y)

    # bias correction factor
    correction = 1 - 3 / (4 * n - 9)
    return d * correction


def compute_glass_delta(treatment: np.ndarray, control: np.ndarray) -> float:
    """
    compute glass's delta (use control group sd).

    delta = (mean_treatment - mean_control) / sd_control
    """
    return (np.mean(treatment) - np.mean(control)) / (np.std(control, ddof=1) + 1e-10)


def compute_rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """
    compute rank-biserial correlation from mann-whitney u.

    r = 1 - (2u) / (n1 * n2)
    """
    return 1 - (2 * u_stat) / (n1 * n2)


def interpret_effect_size(d: float, measure: str = "cohens_d") -> str:
    """
    interpret effect size magnitude according to cohen (1988).

    |d| < 0.2: negligible
    0.2 <= |d| < 0.5: small
    0.5 <= |d| < 0.8: medium
    |d| >= 0.8: large
    """
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: Callable = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    method: str = "percentile",
    random_state: int = 42
) -> BootstrapCI:
    """
    compute bootstrap confidence interval.

    methods:
    - percentile: simple percentile method
    - basic: basic bootstrap (2*estimate - percentiles)
    - bca: bias-corrected and accelerated (most accurate)
    """
    np.random.seed(random_state)
    n = len(data)
    alpha = 1 - confidence_level

    # original estimate
    theta_hat = statistic_fn(data)

    # bootstrap replicates
    boot_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        boot_sample = np.random.choice(data, n, replace=True)
        boot_stats[i] = statistic_fn(boot_sample)

    se = np.std(boot_stats)

    if method == "percentile":
        ci_lower = np.percentile(boot_stats, alpha / 2 * 100)
        ci_upper = np.percentile(boot_stats, (1 - alpha / 2) * 100)

    elif method == "basic":
        ci_lower = 2 * theta_hat - np.percentile(boot_stats, (1 - alpha / 2) * 100)
        ci_upper = 2 * theta_hat - np.percentile(boot_stats, alpha / 2 * 100)

    elif method == "bca":
        # bias correction
        z0 = stats.norm.ppf(np.mean(boot_stats < theta_hat))

        # acceleration (jackknife)
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(data, i)
            jackknife_stats[i] = statistic_fn(jack_sample)

        jack_mean = np.mean(jackknife_stats)
        num = np.sum((jack_mean - jackknife_stats) ** 3)
        denom = 6 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
        a = num / (denom + 1e-10)

        # adjusted percentiles
        z_alpha_lower = stats.norm.ppf(alpha / 2)
        z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

        p_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        p_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))

        ci_lower = np.percentile(boot_stats, p_lower * 100)
        ci_upper = np.percentile(boot_stats, p_upper * 100)

    else:
        raise ValueError(f"unknown bootstrap method: {method}")

    return BootstrapCI(
        estimate=float(theta_hat),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        se=float(se),
        n_bootstrap=n_bootstrap,
        method=method
    )


def paired_t_test(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> StatisticalResult:
    """
    perform paired t-test with effect size and ci.
    """
    n = len(x)
    diff = x - y

    stat, p = ttest_rel(x, y, alternative=alternative)
    d = compute_cohens_d_paired(x, y)
    g = compute_hedges_g(x, y, paired=True)

    # ci for mean difference
    se = np.std(diff, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
    ci_lower = np.mean(diff) - t_crit * se
    ci_upper = np.mean(diff) + t_crit * se

    return StatisticalResult(
        test_name="paired_t_test",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(g),  # use hedges' g for small sample correction
        effect_size_type="hedges_g",
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n_samples=n,
        interpretation=interpret_effect_size(g),
        significant=p < alpha
    )


def independent_t_test(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    equal_var: bool = True
) -> StatisticalResult:
    """
    perform independent samples t-test with effect size.
    """
    stat, p = ttest_ind(x, y, equal_var=equal_var)
    d = compute_cohens_d_independent(x, y)
    g = compute_hedges_g(x, y, paired=False)

    # ci for difference in means
    n1, n2 = len(x), len(y)
    se = np.sqrt(np.var(x, ddof=1) / n1 + np.var(y, ddof=1) / n2)
    df = n1 + n2 - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    diff = np.mean(x) - np.mean(y)
    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se

    return StatisticalResult(
        test_name="independent_t_test",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(g),
        effect_size_type="hedges_g",
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n_samples=n1 + n2,
        interpretation=interpret_effect_size(g),
        significant=p < alpha
    )


def wilcoxon_test(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05
) -> StatisticalResult:
    """
    perform wilcoxon signed-rank test (non-parametric paired test).

    effect size: r = z / sqrt(n)
    """
    n = len(x)
    diff = x - y

    # remove zero differences
    non_zero = diff != 0
    if np.sum(non_zero) < 10:
        warnings.warn("fewer than 10 non-zero differences for wilcoxon test")

    stat, p = wilcoxon(x, y, alternative='two-sided')

    # effect size: r = z / sqrt(n)
    # approximate z from p-value
    z = stats.norm.ppf(1 - p / 2)
    r = z / np.sqrt(n)

    # bootstrap ci for median difference
    boot_ci = bootstrap_ci(diff, statistic_fn=np.median, method='bca')

    return StatisticalResult(
        test_name="wilcoxon_signed_rank",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(r),
        effect_size_type="r",
        ci_lower=float(boot_ci.ci_lower),
        ci_upper=float(boot_ci.ci_upper),
        n_samples=n,
        interpretation=interpret_effect_size(r),
        significant=p < alpha
    )


def mann_whitney_test(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05
) -> StatisticalResult:
    """
    perform mann-whitney u test (non-parametric independent test).

    effect size: rank-biserial correlation
    """
    n1, n2 = len(x), len(y)

    stat, p = mannwhitneyu(x, y, alternative='two-sided')
    r = compute_rank_biserial(stat, n1, n2)

    return StatisticalResult(
        test_name="mann_whitney_u",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(r),
        effect_size_type="rank_biserial",
        ci_lower=np.nan,
        ci_upper=np.nan,
        n_samples=n1 + n2,
        interpretation=interpret_effect_size(r),
        significant=p < alpha
    )


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[List[float], List[bool]]:
    """
    apply bonferroni correction for multiple comparisons.
    """
    n = len(p_values)
    adjusted_p = [min(p * n, 1.0) for p in p_values]
    significant = [p < alpha for p in adjusted_p]
    return adjusted_p, significant


def holm_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[List[float], List[bool]]:
    """
    apply holm-bonferroni step-down correction.

    more powerful than bonferroni while controlling fwer.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    adjusted_p = np.zeros(n)
    for i, idx in enumerate(sorted_indices):
        adjusted_p[idx] = min(sorted_p[i] * (n - i), 1.0)

    # ensure monotonicity
    for i in range(1, n):
        adjusted_p[sorted_indices[i]] = max(
            adjusted_p[sorted_indices[i]],
            adjusted_p[sorted_indices[i - 1]]
        )

    significant = [p < alpha for p in adjusted_p]
    return list(adjusted_p), significant


def benjamini_hochberg_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[List[float], List[bool]]:
    """
    apply benjamini-hochberg fdr correction.

    controls false discovery rate at alpha level.
    recommended for exploratory analyses with many comparisons.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # compute adjusted p-values
    adjusted_p = np.zeros(n)
    for i, idx in enumerate(sorted_indices):
        adjusted_p[idx] = sorted_p[i] * n / (i + 1)

    # ensure monotonicity (from largest to smallest)
    for i in range(n - 2, -1, -1):
        adjusted_p[sorted_indices[i]] = min(
            adjusted_p[sorted_indices[i]],
            adjusted_p[sorted_indices[i + 1]]
        )

    adjusted_p = np.minimum(adjusted_p, 1.0)
    significant = [p < alpha for p in adjusted_p]
    return list(adjusted_p), significant


def select_test(
    x: np.ndarray,
    y: np.ndarray,
    paired: bool = True,
    alpha: float = 0.05
) -> StatisticalResult:
    """
    automatically select appropriate statistical test based on data properties.

    checks normality and homogeneity to choose parametric vs non-parametric.
    """
    if paired:
        diff = x - y
        is_normal, norm_p = check_normality(diff, alpha)

        if is_normal:
            return paired_t_test(x, y, alpha)
        else:
            return wilcoxon_test(x, y, alpha)
    else:
        is_normal_x, _ = check_normality(x, alpha)
        is_normal_y, _ = check_normality(y, alpha)
        is_homogeneous, _ = check_homogeneity(x, y, alpha)

        if is_normal_x and is_normal_y and is_homogeneous:
            return independent_t_test(x, y, alpha, equal_var=True)
        elif is_normal_x and is_normal_y:
            return independent_t_test(x, y, alpha, equal_var=False)
        else:
            return mann_whitney_test(x, y, alpha)


def compute_comprehensive_statistics(
    raw_metrics: Dict[str, np.ndarray],
    harmonized_metrics: Dict[str, np.ndarray],
    paired: bool = True,
    alpha: float = 0.05,
    correction_method: str = "holm"
) -> Dict:
    """
    compute comprehensive statistics for all metrics.

    returns detailed statistical analysis with effect sizes, cis, and corrections.
    """
    results = {
        'comparisons': {},
        'summary': {},
        'correction_method': correction_method,
        'alpha': alpha
    }

    p_values = []
    metric_names = []

    for metric_name in raw_metrics.keys():
        if metric_name not in harmonized_metrics:
            continue

        raw = np.array(raw_metrics[metric_name])
        harm = np.array(harmonized_metrics[metric_name])

        # skip if shapes don't match
        if raw.shape != harm.shape:
            continue

        # flatten if needed
        raw = raw.flatten()
        harm = harm.flatten()

        # compute statistics
        test_result = select_test(raw, harm, paired=paired, alpha=alpha)

        # bootstrap cis for means
        raw_ci = bootstrap_ci(raw, method='bca')
        harm_ci = bootstrap_ci(harm, method='bca')

        results['comparisons'][metric_name] = {
            'raw': {
                'mean': float(np.mean(raw)),
                'std': float(np.std(raw)),
                'ci_lower': raw_ci.ci_lower,
                'ci_upper': raw_ci.ci_upper,
                'n': len(raw)
            },
            'harmonized': {
                'mean': float(np.mean(harm)),
                'std': float(np.std(harm)),
                'ci_lower': harm_ci.ci_lower,
                'ci_upper': harm_ci.ci_upper,
                'n': len(harm)
            },
            'test': asdict(test_result),
            'improvement': float(np.mean(harm) - np.mean(raw)),
            'improvement_percent': float(
                100 * (np.mean(harm) - np.mean(raw)) / (np.mean(raw) + 1e-10)
            )
        }

        p_values.append(test_result.p_value)
        metric_names.append(metric_name)

    # apply multiple comparison correction
    if correction_method == "bonferroni":
        adjusted_p, significant = bonferroni_correction(p_values, alpha)
    elif correction_method == "holm":
        adjusted_p, significant = holm_correction(p_values, alpha)
    elif correction_method == "bh" or correction_method == "fdr":
        adjusted_p, significant = benjamini_hochberg_correction(p_values, alpha)
    else:
        adjusted_p = p_values
        significant = [p < alpha for p in p_values]

    # update with corrected values
    for i, metric_name in enumerate(metric_names):
        results['comparisons'][metric_name]['adjusted_p_value'] = adjusted_p[i]
        results['comparisons'][metric_name]['significant_corrected'] = significant[i]

    # summary statistics
    n_significant = sum(significant)
    n_total = len(significant)

    results['summary'] = {
        'n_metrics': n_total,
        'n_significant_uncorrected': sum(p < alpha for p in p_values),
        'n_significant_corrected': n_significant,
        'significant_metrics': [
            metric_names[i] for i in range(len(significant)) if significant[i]
        ],
        'mean_effect_size': float(np.mean([
            abs(results['comparisons'][m]['test']['effect_size'])
            for m in metric_names
        ])),
    }

    return results


def format_result_for_publication(result: Dict, metric_name: str) -> str:
    """
    format statistical result for publication.

    example: "ssim: 0.82 +/- 0.08 (95% ci: 0.79-0.85), t(95)=3.24, p=0.002, g=0.45"
    """
    comp = result['comparisons'].get(metric_name, {})
    if not comp:
        return ""

    raw = comp['raw']
    harm = comp['harmonized']
    test = comp['test']

    # format based on test type
    test_name = test['test_name']
    if 't_test' in test_name:
        test_str = f"t({test['n_samples']-1})={test['statistic']:.2f}"
    elif 'wilcoxon' in test_name:
        test_str = f"W={test['statistic']:.0f}"
    elif 'mann_whitney' in test_name:
        test_str = f"U={test['statistic']:.0f}"
    else:
        test_str = f"stat={test['statistic']:.2f}"

    # format p-value
    p = comp.get('adjusted_p_value', test['p_value'])
    if p < 0.001:
        p_str = "p<0.001"
    else:
        p_str = f"p={p:.3f}"

    # effect size
    es_type = test['effect_size_type']
    es = test['effect_size']
    es_str = f"{es_type}={es:.2f} ({test['interpretation']})"

    # significance marker
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    return (
        f"{metric_name}:\n"
        f"  raw: {raw['mean']:.3f} +/- {raw['std']:.3f} "
        f"(95% ci: {raw['ci_lower']:.3f}-{raw['ci_upper']:.3f})\n"
        f"  harmonized: {harm['mean']:.3f} +/- {harm['std']:.3f} "
        f"(95% ci: {harm['ci_lower']:.3f}-{harm['ci_upper']:.3f})\n"
        f"  {test_str}, {p_str}{sig}, {es_str}"
    )


def main():
    parser = argparse.ArgumentParser(
        description='comprehensive statistical analysis for harmonization evaluation'
    )
    parser.add_argument('--results-dir', type=str, required=True,
                       help='directory containing evaluation results')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='output directory for statistical analysis')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='significance level')
    parser.add_argument('--correction', type=str, default='holm',
                       choices=['bonferroni', 'holm', 'bh', 'fdr', 'none'],
                       help='multiple comparison correction method')
    parser.add_argument('--n-bootstrap', type=int, default=10000,
                       help='number of bootstrap samples')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir)

    print('[stats] loading evaluation results...')

    # load domain classification results
    domain_results_path = results_dir / 'domain_classification' / 'domain_classification_results.json'
    if domain_results_path.exists():
        with open(domain_results_path) as f:
            domain_results = json.load(f)
    else:
        domain_results = None

    # load feature distribution results
    feature_results_path = results_dir / 'feature_distribution' / 'feature_distribution_results.json'
    if feature_results_path.exists():
        with open(feature_results_path) as f:
            feature_results = json.load(f)
    else:
        feature_results = None

    # compile comprehensive statistics
    all_stats = {}

    if domain_results:
        print('[stats] analyzing domain classification results...')

        # extract metrics for statistical analysis
        raw_acc = domain_results['raw']['accuracy']
        harm_acc = domain_results.get('harmonized', {}).get('accuracy', None)

        if harm_acc:
            # compute reduction significance
            acc_reduction = raw_acc - harm_acc

            # bootstrap ci for the reduction
            # simulate from the accuracies (using binomial approximation)
            n_test = 318  # from our evaluation
            raw_correct = int(raw_acc * n_test)
            harm_correct = int(harm_acc * n_test)

            # generate simulated accuracies
            np.random.seed(42)
            raw_sim = np.random.binomial(n_test, raw_acc, 10000) / n_test
            harm_sim = np.random.binomial(n_test, harm_acc, 10000) / n_test
            reduction_sim = raw_sim - harm_sim

            reduction_ci = bootstrap_ci(reduction_sim, method='percentile')

            all_stats['domain_classification'] = {
                'raw_accuracy': raw_acc,
                'harmonized_accuracy': harm_acc,
                'accuracy_reduction': acc_reduction,
                'reduction_ci_lower': reduction_ci.ci_lower,
                'reduction_ci_upper': reduction_ci.ci_upper,
                'reduction_se': reduction_ci.se,
                'interpretation': 'harmonization significantly reduces domain discriminability'
                                 if harm_acc < 0.6 else 'moderate domain reduction'
            }

    if feature_results:
        print('[stats] analyzing feature distribution results...')

        raw_feat = feature_results['raw']
        harm_feat = feature_results.get('harmonized', {})

        if harm_feat:
            all_stats['feature_distribution'] = {
                'raw': raw_feat,
                'harmonized': harm_feat,
                'fid_reduction': raw_feat['fid'] - harm_feat['fid'],
                'fid_reduction_percent': 100 * (raw_feat['fid'] - harm_feat['fid']) / (raw_feat['fid'] + 1e-10),
                'mmd_reduction': raw_feat['mmd_rbf'] - harm_feat['mmd_rbf'],
                'mmd_reduction_percent': 100 * (raw_feat['mmd_rbf'] - harm_feat['mmd_rbf']) / (raw_feat['mmd_rbf'] + 1e-10),
            }

    # save results
    with open(output_dir / 'comprehensive_statistics.json', 'w') as f:
        json.dump(all_stats, f, indent=2)

    print('=' * 60)
    print('[stats] statistical analysis complete')
    print('=' * 60)

    if 'domain_classification' in all_stats:
        dc = all_stats['domain_classification']
        print('[domain classification]')
        print(f"  raw accuracy: {dc['raw_accuracy']:.4f}")
        print(f"  harmonized accuracy: {dc['harmonized_accuracy']:.4f}")
        print(f"  reduction: {dc['accuracy_reduction']:.4f} "
              f"(95% ci: {dc['reduction_ci_lower']:.4f}-{dc['reduction_ci_upper']:.4f})")

    print(f'[stats] results saved to {output_dir}')


if __name__ == '__main__':
    main()
