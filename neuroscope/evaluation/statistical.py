"""
Statistical Analysis Module.

Comprehensive statistical testing and analysis
for medical image harmonization evaluation.
"""

from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
from scipy import stats
from scipy.stats import (
    ttest_rel, ttest_ind, wilcoxon, mannwhitneyu,
    f_oneway, kruskal, shapiro, levene
)


@dataclass
class TestResult:
    """Result from statistical test."""
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    test_name: str = ""
    significant: bool = False
    alpha: float = 0.05
    
    def __post_init__(self):
        self.significant = self.p_value < self.alpha


@dataclass
class ConfidenceInterval:
    """Confidence interval result."""
    lower: float
    upper: float
    point_estimate: float
    confidence_level: float = 0.95
    method: str = "bootstrap"


def paired_t_test(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> TestResult:
    """
    Paired t-test for related samples.
    
    Args:
        x: First sample
        y: Second sample (paired with x)
        alpha: Significance level
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        TestResult with statistics
    """
    statistic, p_value = ttest_rel(x, y, alternative=alternative)
    
    # Effect size (Cohen's d for paired samples)
    diff = x - y
    effect_size = np.mean(diff) / np.std(diff, ddof=1)
    
    # Confidence interval for mean difference
    se = np.std(diff, ddof=1) / np.sqrt(len(diff))
    t_crit = stats.t.ppf(1 - alpha / 2, len(diff) - 1)
    ci = (np.mean(diff) - t_crit * se, np.mean(diff) + t_crit * se)
    
    return TestResult(
        statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        confidence_interval=ci,
        test_name="Paired t-test",
        alpha=alpha
    )


def independent_t_test(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    equal_var: bool = True,
    alternative: str = 'two-sided'
) -> TestResult:
    """
    Independent samples t-test.
    
    Args:
        x: First sample
        y: Second sample
        alpha: Significance level
        equal_var: Assume equal variances
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        TestResult with statistics
    """
    statistic, p_value = ttest_ind(
        x, y, equal_var=equal_var, alternative=alternative
    )
    
    # Cohen's d for independent samples
    pooled_std = np.sqrt(
        ((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) /
        (len(x) + len(y) - 2)
    )
    effect_size = (np.mean(x) - np.mean(y)) / pooled_std
    
    return TestResult(
        statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        test_name="Independent t-test",
        alpha=alpha
    )


def wilcoxon_test(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> TestResult:
    """
    Wilcoxon signed-rank test for paired samples.
    
    Non-parametric alternative to paired t-test.
    
    Args:
        x: First sample
        y: Second sample (paired)
        alpha: Significance level
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        TestResult with statistics
    """
    statistic, p_value = wilcoxon(x, y, alternative=alternative)
    
    # Effect size: r = Z / sqrt(N)
    n = len(x)
    z = stats.norm.ppf(p_value / 2)
    effect_size = abs(z) / np.sqrt(n)
    
    return TestResult(
        statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        test_name="Wilcoxon signed-rank",
        alpha=alpha
    )


def mann_whitney_test(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> TestResult:
    """
    Mann-Whitney U test for independent samples.
    
    Non-parametric alternative to independent t-test.
    
    Args:
        x: First sample
        y: Second sample
        alpha: Significance level
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        TestResult with statistics
    """
    statistic, p_value = mannwhitneyu(x, y, alternative=alternative)
    
    # Effect size: rank-biserial correlation
    n1, n2 = len(x), len(y)
    effect_size = 1 - (2 * statistic) / (n1 * n2)
    
    return TestResult(
        statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        test_name="Mann-Whitney U",
        alpha=alpha
    )


def anova_test(
    *groups: np.ndarray,
    alpha: float = 0.05
) -> TestResult:
    """
    One-way ANOVA for multiple group comparison.
    
    Args:
        *groups: Variable number of sample groups
        alpha: Significance level
        
    Returns:
        TestResult with F-statistic
    """
    statistic, p_value = f_oneway(*groups)
    
    # Effect size: eta-squared
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    
    ss_between = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2 
        for g in groups
    )
    ss_total = np.sum((all_data - grand_mean) ** 2)
    
    effect_size = ss_between / ss_total if ss_total > 0 else 0
    
    return TestResult(
        statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        test_name="One-way ANOVA",
        alpha=alpha
    )


def kruskal_wallis_test(
    *groups: np.ndarray,
    alpha: float = 0.05
) -> TestResult:
    """
    Kruskal-Wallis H test for multiple group comparison.
    
    Non-parametric alternative to ANOVA.
    
    Args:
        *groups: Variable number of sample groups
        alpha: Significance level
        
    Returns:
        TestResult with H-statistic
    """
    statistic, p_value = kruskal(*groups)
    
    # Effect size: epsilon-squared
    n = sum(len(g) for g in groups)
    effect_size = (statistic - len(groups) + 1) / (n - len(groups))
    
    return TestResult(
        statistic=statistic,
        p_value=p_value,
        effect_size=max(0, effect_size),
        test_name="Kruskal-Wallis H",
        alpha=alpha
    )


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    method: str = 'percentile'
) -> ConfidenceInterval:
    """
    Bootstrap confidence interval.
    
    Args:
        data: Input data
        statistic_fn: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95)
        method: 'percentile', 'bca', or 'basic'
        
    Returns:
        ConfidenceInterval result
    """
    n = len(data)
    point_estimate = statistic_fn(data)
    
    # Generate bootstrap samples
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic_fn(sample)
    
    alpha = 1 - confidence_level
    
    if method == 'percentile':
        lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    
    elif method == 'basic':
        lower = 2 * point_estimate - np.percentile(
            bootstrap_stats, (1 - alpha / 2) * 100
        )
        upper = 2 * point_estimate - np.percentile(
            bootstrap_stats, alpha / 2 * 100
        )
    
    elif method == 'bca':
        # Bias-corrected and accelerated
        z0 = stats.norm.ppf(np.mean(bootstrap_stats < point_estimate))
        
        # Jackknife for acceleration
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jackknife_sample = np.delete(data, i)
            jackknife_stats[i] = statistic_fn(jackknife_sample)
        
        jackknife_mean = np.mean(jackknife_stats)
        a = np.sum((jackknife_mean - jackknife_stats) ** 3) / (
            6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
        )
        
        z_alpha = stats.norm.ppf(alpha / 2)
        z_1_alpha = stats.norm.ppf(1 - alpha / 2)
        
        alpha1 = stats.norm.cdf(
            z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
        )
        alpha2 = stats.norm.cdf(
            z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha))
        )
        
        lower = np.percentile(bootstrap_stats, alpha1 * 100)
        upper = np.percentile(bootstrap_stats, alpha2 * 100)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return ConfidenceInterval(
        lower=lower,
        upper=upper,
        point_estimate=point_estimate,
        confidence_level=confidence_level,
        method=method
    )


def compute_effect_size(
    x: np.ndarray,
    y: np.ndarray,
    method: str = 'cohens_d'
) -> float:
    """
    Compute effect size between two samples.
    
    Args:
        x: First sample
        y: Second sample
        method: 'cohens_d', 'hedges_g', or 'glass_delta'
        
    Returns:
        Effect size value
    """
    mean_diff = np.mean(x) - np.mean(y)
    
    if method == 'cohens_d':
        # Pooled standard deviation
        pooled_std = np.sqrt(
            ((len(x) - 1) * np.var(x, ddof=1) + 
             (len(y) - 1) * np.var(y, ddof=1)) /
            (len(x) + len(y) - 2)
        )
        return mean_diff / pooled_std
    
    elif method == 'hedges_g':
        # Small sample correction to Cohen's d
        cohens_d = compute_effect_size(x, y, 'cohens_d')
        n = len(x) + len(y)
        correction = 1 - 3 / (4 * n - 9)
        return cohens_d * correction
    
    elif method == 'glass_delta':
        # Use control group SD only
        return mean_diff / np.std(y, ddof=1)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def interpret_effect_size(
    d: float,
    method: str = 'cohens_d'
) -> str:
    """
    Interpret effect size using standard thresholds.
    
    Args:
        d: Effect size value
        method: Effect size type
        
    Returns:
        Interpretation string
    """
    d = abs(d)
    
    if method in ['cohens_d', 'hedges_g']:
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    return "unknown"


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[List[float], List[bool]]:
    """
    Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values
        alpha: Significance level
        
    Returns:
        Tuple of (adjusted p-values, significance flags)
    """
    n = len(p_values)
    adjusted = [min(p * n, 1.0) for p in p_values]
    significant = [p < alpha for p in adjusted]
    
    return adjusted, significant


def benjamini_hochberg(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[List[float], List[bool]]:
    """
    Benjamini-Hochberg procedure for FDR control.
    
    Args:
        p_values: List of p-values
        alpha: Significance level
        
    Returns:
        Tuple of (adjusted p-values, significance flags)
    """
    n = len(p_values)
    
    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    # Compute adjusted p-values
    adjusted = np.zeros(n)
    for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
        rank = i + 1
        adjusted[idx] = p * n / rank
    
    # Ensure monotonicity
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    
    adjusted = np.clip(adjusted, 0, 1)
    significant = [p < alpha for p in adjusted]
    
    return list(adjusted), significant


def check_normality(
    data: np.ndarray,
    alpha: float = 0.05
) -> TestResult:
    """
    Shapiro-Wilk test for normality.
    
    Args:
        data: Input data
        alpha: Significance level
        
    Returns:
        TestResult (significant = not normal)
    """
    statistic, p_value = shapiro(data)
    
    return TestResult(
        statistic=statistic,
        p_value=p_value,
        test_name="Shapiro-Wilk",
        alpha=alpha
    )


def check_homogeneity(
    *groups: np.ndarray,
    alpha: float = 0.05
) -> TestResult:
    """
    Levene's test for homogeneity of variances.
    
    Args:
        *groups: Variable number of sample groups
        alpha: Significance level
        
    Returns:
        TestResult (significant = not homogeneous)
    """
    statistic, p_value = levene(*groups)
    
    return TestResult(
        statistic=statistic,
        p_value=p_value,
        test_name="Levene's",
        alpha=alpha
    )


class StatisticalAnalysis:
    """
    Comprehensive statistical analysis framework.
    
    Provides methods for complete statistical analysis
    of experimental results.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level for all tests
        """
        self.alpha = alpha
        self.results: Dict[str, TestResult] = {}
    
    def compare_methods(
        self,
        method_results: Dict[str, np.ndarray],
        baseline: str = None,
        paired: bool = True
    ) -> Dict[str, TestResult]:
        """
        Compare multiple methods against baseline.
        
        Args:
            method_results: Dict of method name -> scores
            baseline: Baseline method name
            paired: Whether samples are paired
            
        Returns:
            Dict of comparison name -> TestResult
        """
        comparisons = {}
        methods = list(method_results.keys())
        
        if baseline is None:
            baseline = methods[0]
        
        baseline_data = method_results[baseline]
        
        for method in methods:
            if method == baseline:
                continue
            
            method_data = method_results[method]
            
            # Check normality
            norm_baseline = check_normality(baseline_data, self.alpha)
            norm_method = check_normality(method_data, self.alpha)
            
            # Use parametric or non-parametric test
            if norm_baseline.significant or norm_method.significant:
                # Non-normal: use non-parametric
                if paired:
                    result = wilcoxon_test(
                        baseline_data, method_data, self.alpha
                    )
                else:
                    result = mann_whitney_test(
                        baseline_data, method_data, self.alpha
                    )
            else:
                # Normal: use t-test
                if paired:
                    result = paired_t_test(
                        baseline_data, method_data, self.alpha
                    )
                else:
                    result = independent_t_test(
                        baseline_data, method_data, self.alpha
                    )
            
            comparisons[f"{baseline}_vs_{method}"] = result
        
        # Apply multiple comparison correction
        p_values = [r.p_value for r in comparisons.values()]
        adjusted_p, significant = bonferroni_correction(p_values, self.alpha)
        
        for i, key in enumerate(comparisons.keys()):
            comparisons[key].p_value = adjusted_p[i]
            comparisons[key].significant = significant[i]
        
        self.results.update(comparisons)
        return comparisons
    
    def compute_summary_statistics(
        self,
        data: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute summary statistics.
        
        Args:
            data: Input data
            
        Returns:
            Dict of statistic name -> value
        """
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data, ddof=1)),
            'median': float(np.median(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'q1': float(np.percentile(data, 25)),
            'q3': float(np.percentile(data, 75)),
            'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
            'n': len(data)
        }
    
    def generate_report(self) -> Dict:
        """Generate comprehensive statistical report."""
        report = {
            'alpha': self.alpha,
            'tests': {}
        }
        
        for name, result in self.results.items():
            report['tests'][name] = {
                'test_name': result.test_name,
                'statistic': result.statistic,
                'p_value': result.p_value,
                'effect_size': result.effect_size,
                'significant': result.significant,
                'interpretation': interpret_effect_size(
                    result.effect_size
                ) if result.effect_size else None
            }
        
        return report
