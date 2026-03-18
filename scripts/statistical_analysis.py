#!/usr/bin/env python3
"""
statistical analysis script for model comparison.

performs rigorous statistical tests to validate experimental results:
- paired t-tests
- wilcoxon signed-rank tests
- anova with post-hoc tests
- effect size calculations (cohen's d)
- bonferroni correction for multiple comparisons
- bootstrap confidence intervals

usage:
    python scripts/statistical_analysis.py \
        --results-dir ./results/evaluation \
        --output-dir ./results/statistics \
        --alpha 0.05 \
        --n-bootstrap 10000
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """performs statistical analysis for model comparisons."""

    def __init__(
        self,
        alpha: float = 0.05,
        correction_method: str = 'bonferroni',
        n_bootstrap: int = 10000,
    ):
        """
        initialize statistical analyzer.

        args:
            alpha: significance level
            correction_method: method for multiple testing correction
            n_bootstrap: number of bootstrap samples
        """
        self.alpha = alpha
        self.correction_method = correction_method
        self.n_bootstrap = n_bootstrap

    def paired_t_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray
    ) -> Tuple[float, float]:
        """
        perform paired t-test.

        args:
            data1: first dataset
            data2: second dataset

        returns:
            (t-statistic, p-value)
        """
        t_stat, p_value = stats.ttest_rel(data1, data2)
        return t_stat, p_value

    def wilcoxon_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray
    ) -> Tuple[float, float]:
        """
        perform wilcoxon signed-rank test (non-parametric).

        args:
            data1: first dataset
            data2: second dataset

        returns:
            (statistic, p-value)
        """
        stat, p_value = stats.wilcoxon(data1, data2)
        return stat, p_value

    def cohens_d(
        self,
        data1: np.ndarray,
        data2: np.ndarray
    ) -> float:
        """
        calculate cohen's d effect size.

        args:
            data1: first dataset
            data2: second dataset

        returns:
            cohen's d value
        """
        mean_diff = np.mean(data1) - np.mean(data2)
        pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
        return mean_diff / pooled_std

    def bootstrap_ci(
        self,
        data: np.ndarray,
        confidence_level: float = 0.95,
        statistic=np.mean
    ) -> Tuple[float, float]:
        """
        calculate bootstrap confidence interval.

        args:
            data: dataset
            confidence_level: confidence level (default 0.95 for 95% ci)
            statistic: statistic function to apply

        returns:
            (lower_bound, upper_bound)
        """
        rng = np.random.default_rng(seed=42)

        res = bootstrap(
            (data,),
            statistic=statistic,
            n_resamples=self.n_bootstrap,
            confidence_level=confidence_level,
            random_state=rng
        )

        return res.confidence_interval.low, res.confidence_interval.high

    def anova_with_posthoc(
        self,
        data_dict: Dict[str, np.ndarray]
    ) -> Tuple[float, float, Dict]:
        """
        perform one-way anova with post-hoc pairwise tests.

        args:
            data_dict: dictionary mapping group names to data arrays

        returns:
            (f-statistic, p-value, pairwise_results)
        """
        # prepare data for anova
        groups = list(data_dict.values())

        # perform anova
        f_stat, p_value = stats.f_oneway(*groups)

        # post-hoc pairwise comparisons
        pairwise_results = {}
        model_names = list(data_dict.keys())

        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]

                # paired t-test
                t_stat, p_val = self.paired_t_test(
                    data_dict[model1],
                    data_dict[model2]
                )

                # effect size
                effect_size = self.cohens_d(
                    data_dict[model1],
                    data_dict[model2]
                )

                pairwise_results[(model1, model2)] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_val),
                    'cohens_d': float(effect_size),
                }

        # apply multiple testing correction
        p_values = [v['p_value'] for v in pairwise_results.values()]
        _, corrected_p, _, _ = multipletests(
            p_values,
            alpha=self.alpha,
            method=self.correction_method
        )

        # update with corrected p-values
        for idx, key in enumerate(pairwise_results.keys()):
            pairwise_results[key]['p_value_corrected'] = float(corrected_p[idx])
            pairwise_results[key]['significant'] = corrected_p[idx] < self.alpha

        return f_stat, p_value, pairwise_results

    def normality_test(self, data: np.ndarray) -> Tuple[float, float]:
        """
        test for normality using shapiro-wilk test.

        args:
            data: dataset

        returns:
            (statistic, p-value)
        """
        stat, p_value = stats.shapiro(data)
        return stat, p_value

    def analyze_metric(
        self,
        results: Dict[str, Dict],
        metric: str
    ) -> Dict:
        """
        perform comprehensive statistical analysis for a single metric.

        args:
            results: dictionary of model results
            metric: metric name to analyze

        returns:
            dictionary with analysis results
        """
        logger.info(f"Analyzing {metric}...")

        # extract data for each model
        data_dict = {}
        for model_name, model_results in results.items():
            if metric in model_results['metrics']:
                values = model_results['metrics'][metric]['values']
                data_dict[model_name] = np.array(values)

        if len(data_dict) < 2:
            logger.warning(f"Not enough models for {metric} comparison")
            return {}

        analysis = {
            'metric': metric,
            'n_models': len(data_dict),
            'models': list(data_dict.keys()),
        }

        # descriptive statistics
        descriptive = {}
        for model_name, data in data_dict.items():
            # normality test
            shapiro_stat, shapiro_p = self.normality_test(data)

            # bootstrap ci
            ci_low, ci_high = self.bootstrap_ci(data)

            descriptive[model_name] = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data, ddof=1)),
                'median': float(np.median(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'ci_95_low': float(ci_low),
                'ci_95_high': float(ci_high),
                'is_normal': shapiro_p > self.alpha,
                'shapiro_p': float(shapiro_p),
            }

        analysis['descriptive'] = descriptive

        # anova and post-hoc tests
        f_stat, anova_p, pairwise = self.anova_with_posthoc(data_dict)

        analysis['anova'] = {
            'f_statistic': float(f_stat),
            'p_value': float(anova_p),
            'significant': anova_p < self.alpha,
        }

        analysis['pairwise_comparisons'] = pairwise

        # find best performing model
        means = {k: v['mean'] for k, v in descriptive.items()}

        # for metrics where higher is better (ssim, psnr, nmi)
        if metric.upper() in ['SSIM', 'PSNR', 'NMI']:
            best_model = max(means, key=means.get)
        else:  # for metrics where lower is better (lpips, fid)
            best_model = min(means, key=means.get)

        analysis['best_model'] = best_model
        analysis['best_model_mean'] = means[best_model]

        # check if best model is significantly better than others
        significantly_better_than = []
        for other_model in means.keys():
            if other_model != best_model:
                pair_key = (best_model, other_model)
                reverse_key = (other_model, best_model)

                if pair_key in pairwise:
                    pair_result = pairwise[pair_key]
                elif reverse_key in pairwise:
                    pair_result = pairwise[reverse_key]
                else:
                    continue

                if pair_result['significant']:
                    significantly_better_than.append(other_model)

        analysis['significantly_better_than'] = significantly_better_than

        return analysis

    def generate_summary_table(self, analyses: List[Dict]) -> pd.DataFrame:
        """
        generate summary table of statistical results.

        args:
            analyses: list of analysis results for each metric

        returns:
            dataframe with summary
        """
        rows = []

        for analysis in analyses:
            if not analysis:
                continue

            metric = analysis['metric']
            best_model = analysis['best_model']
            best_mean = analysis['best_model_mean']
            n_sig_better = len(analysis['significantly_better_than'])

            row = {
                'Metric': metric.upper(),
                'Best Model': best_model,
                'Mean': f"{best_mean:.4f}",
                'Significantly Better Than': f"{n_sig_better}/{analysis['n_models']-1} models",
                'ANOVA p-value': f"{analysis['anova']['p_value']:.4f}",
            }

            rows.append(row)

        return pd.DataFrame(rows)

    def generate_pairwise_table(
        self,
        analyses: List[Dict],
        model1: str,
        model2: str
    ) -> pd.DataFrame:
        """
        generate pairwise comparison table for two models.

        args:
            analyses: list of analysis results
            model1: first model name
            model2: second model name

        returns:
            dataframe with pairwise comparisons
        """
        rows = []

        for analysis in analyses:
            if not analysis:
                continue

            metric = analysis['metric']
            pairwise = analysis.get('pairwise_comparisons', {})

            pair_key = (model1, model2)
            reverse_key = (model2, model1)

            if pair_key in pairwise:
                result = pairwise[pair_key]
            elif reverse_key in pairwise:
                result = pairwise[reverse_key]
            else:
                continue

            row = {
                'Metric': metric.upper(),
                't-statistic': f"{result['t_statistic']:.4f}",
                'p-value': f"{result['p_value']:.4f}",
                'Corrected p-value': f"{result['p_value_corrected']:.4f}",
                "Cohen's d": f"{result['cohens_d']:.4f}",
                'Significant': '+' if result['significant'] else '✗',
            }

            rows.append(row)

        return pd.DataFrame(rows)


def main():
    """main execution function."""
    parser = argparse.ArgumentParser(
        description='Perform statistical analysis on evaluation results'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing evaluation results'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for statistical analysis'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level'
    )

    parser.add_argument(
        '--correction',
        type=str,
        default='bonferroni',
        choices=['bonferroni', 'holm', 'fdr_bh'],
        help='Multiple testing correction method'
    )

    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=10000,
        help='Number of bootstrap samples'
    )

    args = parser.parse_args()

    # load results
    results_file = Path(args.results_dir) / 'all_results.json'
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return

    with open(results_file, 'r') as f:
        results_list = json.load(f)

    # convert to dictionary
    results = {r['model_name']: r for r in results_list}

    # initialize analyzer
    analyzer = StatisticalAnalyzer(
        alpha=args.alpha,
        correction_method=args.correction,
        n_bootstrap=args.n_bootstrap,
    )

    # perform analysis for each metric
    metrics = set()
    for model_results in results.values():
        metrics.update(model_results['metrics'].keys())

    analyses = []
    for metric in sorted(metrics):
        analysis = analyzer.analyze_metric(results, metric)
        if analysis:
            analyses.append(analysis)

    # save detailed results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_file = output_dir / 'detailed_analysis.json'
    with open(detailed_file, 'w') as f:
        json.dump(analyses, f, indent=2)

    logger.info(f"Detailed analysis saved to: {detailed_file}")

    # generate summary table
    summary_table = analyzer.generate_summary_table(analyses)
    summary_file = output_dir / 'summary_table.csv'
    summary_table.to_csv(summary_file, index=False)

    logger.info("\n" + "="*80)
    logger.info("Summary Table:")
    logger.info("="*80)
    print(summary_table.to_string(index=False))

    # generate pairwise comparison tables for sa-cyclegan vs baselines
    model_names = list(results.keys())
    if 'sa_cyclegan' in model_names:
        for other_model in model_names:
            if other_model != 'sa_cyclegan':
                pairwise_table = analyzer.generate_pairwise_table(
                    analyses,
                    'sa_cyclegan',
                    other_model
                )

                pairwise_file = output_dir / f'pairwise_sa_cyclegan_vs_{other_model}.csv'
                pairwise_table.to_csv(pairwise_file, index=False)

                logger.info(f"\nPairwise comparison: SA-CycleGAN vs {other_model}")
                print(pairwise_table.to_string(index=False))

    logger.info("\n" + "="*80)
    logger.info(f"Statistical analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
