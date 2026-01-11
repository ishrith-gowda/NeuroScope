#!/usr/bin/env python3
"""
statistical significance testing for model comparisons

performs rigorous statistical tests to validate performance claims:
- paired t-tests for metric comparisons
- wilcoxon signed-rank test (non-parametric)
- bootstrap confidence intervals
- effect size (cohen's d)
- multiple comparison correction (bonferroni/holm)
- per-modality significance testing

this ensures publication-quality statistical rigor for top-tier venues.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StatisticalComparator:
    """performs statistical comparisons between models."""

    def __init__(self, alpha: float = 0.05):
        """
        initialize comparator.

        args:
            alpha: significance level for hypothesis tests
        """
        self.alpha = alpha

    def paired_ttest(
        self,
        samples1: np.ndarray,
        samples2: np.ndarray
    ) -> Dict:
        """
        perform paired t-test.

        args:
            samples1: scores from model 1
            samples2: scores from model 2

        returns:
            dict with test results
        """
        t_stat, p_value = ttest_rel(samples1, samples2)

        result = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'mean_diff': float(np.mean(samples1 - samples2)),
            'std_diff': float(np.std(samples1 - samples2)),
        }

        return result

    def wilcoxon_test(
        self,
        samples1: np.ndarray,
        samples2: np.ndarray
    ) -> Dict:
        """
        perform wilcoxon signed-rank test (non-parametric).

        args:
            samples1: scores from model 1
            samples2: scores from model 2

        returns:
            dict with test results
        """
        try:
            stat, p_value = wilcoxon(samples1, samples2)

            result = {
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'median_diff': float(np.median(samples1 - samples2)),
            }
        except ValueError as e:
            logger.warning(f"wilcoxon test failed: {e}")
            result = {
                'statistic': None,
                'p_value': None,
                'significant': None,
                'median_diff': float(np.median(samples1 - samples2)),
            }

        return result

    def bootstrap_ci(
        self,
        samples: np.ndarray,
        n_bootstrap: int = 10000,
        ci_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        compute bootstrap confidence interval.

        args:
            samples: sample data
            n_bootstrap: number of bootstrap iterations
            ci_level: confidence level

        returns:
            (lower_bound, upper_bound)
        """
        bootstrap_means = []

        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(samples, size=len(samples), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))

        bootstrap_means = np.array(bootstrap_means)

        alpha_half = (1 - ci_level) / 2
        lower = np.percentile(bootstrap_means, alpha_half * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha_half) * 100)

        return float(lower), float(upper)

    def cohens_d(
        self,
        samples1: np.ndarray,
        samples2: np.ndarray
    ) -> float:
        """
        compute cohen's d effect size.

        args:
            samples1: scores from model 1
            samples2: scores from model 2

        returns:
            effect size (cohen's d)
        """
        n1, n2 = len(samples1), len(samples2)
        var1, var2 = np.var(samples1, ddof=1), np.var(samples2, ddof=1)

        # pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # cohen's d
        d = (np.mean(samples1) - np.mean(samples2)) / pooled_std

        return float(d)

    def bonferroni_correction(
        self,
        p_values: List[float],
        alpha: Optional[float] = None
    ) -> Tuple[List[bool], float]:
        """
        apply bonferroni correction for multiple comparisons.

        args:
            p_values: list of p-values
            alpha: significance level (uses self.alpha if not provided)

        returns:
            (list of significance decisions, adjusted alpha)
        """
        if alpha is None:
            alpha = self.alpha

        n = len(p_values)
        adjusted_alpha = alpha / n

        significant = [p < adjusted_alpha for p in p_values]

        return significant, adjusted_alpha

    def holm_bonferroni_correction(
        self,
        p_values: List[float],
        alpha: Optional[float] = None
    ) -> List[bool]:
        """
        apply holm-bonferroni correction (more powerful than bonferroni).

        args:
            p_values: list of p-values
            alpha: significance level

        returns:
            list of significance decisions
        """
        if alpha is None:
            alpha = self.alpha

        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]

        significant = np.zeros(n, dtype=bool)

        for i, p in enumerate(sorted_p):
            adjusted_alpha = alpha / (n - i)
            if p < adjusted_alpha:
                significant[sorted_indices[i]] = True
            else:
                break

        return significant.tolist()

    def compare_models(
        self,
        model1_results: Dict,
        model2_results: Dict,
        model1_name: str = "model 1",
        model2_name: str = "model 2"
    ) -> Dict:
        """
        comprehensive statistical comparison between two models.

        args:
            model1_results: evaluation results for model 1
            model2_results: evaluation results for model 2
            model1_name: name of model 1
            model2_name: name of model 2

        returns:
            dict with comprehensive comparison results
        """
        logger.info(f"comparing {model1_name} vs {model2_name}")

        comparison = {
            'model1': model1_name,
            'model2': model2_name,
            'metrics': {}
        }

        # get common metrics
        metrics1 = set(model1_results.get('a2b', {}).keys())
        metrics2 = set(model2_results.get('a2b', {}).keys())
        common_metrics = metrics1 & metrics2

        for metric in common_metrics:
            if metric == 'fid':
                # fid is computed globally, not per-sample
                continue

            logger.info(f"  testing {metric}...")

            # extract values
            values1_a2b = np.array(model1_results['a2b'][metric].get('values', []))
            values2_a2b = np.array(model2_results['a2b'][metric].get('values', []))

            if len(values1_a2b) == 0 or len(values2_a2b) == 0:
                continue

            # paired t-test
            ttest_result = self.paired_ttest(values1_a2b, values2_a2b)

            # wilcoxon test
            wilcoxon_result = self.wilcoxon_test(values1_a2b, values2_a2b)

            # effect size
            effect_size = self.cohens_d(values1_a2b, values2_a2b)

            # bootstrap ci for model 1
            ci_lower1, ci_upper1 = self.bootstrap_ci(values1_a2b)

            # bootstrap ci for model 2
            ci_lower2, ci_upper2 = self.bootstrap_ci(values2_a2b)

            comparison['metrics'][metric] = {
                'model1_mean': float(np.mean(values1_a2b)),
                'model1_std': float(np.std(values1_a2b)),
                'model1_ci': [ci_lower1, ci_upper1],
                'model2_mean': float(np.mean(values2_a2b)),
                'model2_std': float(np.std(values2_a2b)),
                'model2_ci': [ci_lower2, ci_upper2],
                'ttest': ttest_result,
                'wilcoxon': wilcoxon_result,
                'cohens_d': effect_size,
                'interpretation': self._interpret_effect_size(effect_size)
            }

        return comparison

    def _interpret_effect_size(self, d: float) -> str:
        """interpret cohen's d effect size."""
        abs_d = abs(d)

        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'

    def generate_comparison_table(
        self,
        comparisons: List[Dict],
        output_path: Path
    ):
        """
        generate publication-ready comparison table.

        args:
            comparisons: list of comparison dicts
            output_path: path to save table
        """
        rows = []

        for comp in comparisons:
            model1 = comp['model1']
            model2 = comp['model2']

            for metric, results in comp['metrics'].items():
                row = {
                    'comparison': f"{model1} vs {model2}",
                    'metric': metric.upper(),
                    f'{model1}_mean': f"{results['model1_mean']:.4f}",
                    f'{model1}_std': f"{results['model1_std']:.4f}",
                    f'{model2}_mean': f"{results['model2_mean']:.4f}",
                    f'{model2}_std': f"{results['model2_std']:.4f}",
                    'p_value': f"{results['ttest']['p_value']:.4e}",
                    'significant': '✓' if results['ttest']['significant'] else '✗',
                    'cohens_d': f"{results['cohens_d']:.3f}",
                    'effect': results['interpretation']
                }

                rows.append(row)

        df = pd.DataFrame(rows)

        # save as csv
        df.to_csv(output_path.with_suffix('.csv'), index=False)

        # save as latex
        latex_str = df.to_latex(index=False, escape=False)
        output_path.with_suffix('.tex').write_text(latex_str)

        logger.info(f"comparison table saved to {output_path}")

    def plot_comparison(
        self,
        model1_results: Dict,
        model2_results: Dict,
        model1_name: str,
        model2_name: str,
        output_dir: Path
    ):
        """
        create visualization comparing two models.

        args:
            model1_results: results for model 1
            model2_results: results for model 2
            model1_name: name of model 1
            model2_name: name of model 2
            output_dir: directory to save plots
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = ['ssim', 'psnr', 'mae', 'lpips']

        for metric in metrics:
            if metric not in model1_results.get('a2b', {}):
                continue

            values1 = model1_results['a2b'][metric].get('values', [])
            values2 = model2_results['a2b'][metric].get('values', [])

            if len(values1) == 0 or len(values2) == 0:
                continue

            # create violin plot
            fig, ax = plt.subplots(figsize=(8, 6))

            data = pd.DataFrame({
                'model': [model1_name] * len(values1) + [model2_name] * len(values2),
                'score': list(values1) + list(values2)
            })

            sns.violinplot(data=data, x='model', y='score', ax=ax)
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_ylabel(metric.upper())

            plt.tight_layout()
            plt.savefig(output_dir / f'{metric}_comparison.png', dpi=300)
            plt.close()

        logger.info(f"comparison plots saved to {output_dir}")


def main():
    """main execution function."""
    parser = argparse.ArgumentParser(
        description='statistical significance testing for model comparisons'
    )

    parser.add_argument(
        '--model1',
        type=str,
        required=True,
        help='path to model 1 evaluation results (json)'
    )

    parser.add_argument(
        '--model2',
        type=str,
        required=True,
        help='path to model 2 evaluation results (json)'
    )

    parser.add_argument(
        '--model1-name',
        type=str,
        default='model 1',
        help='name for model 1'
    )

    parser.add_argument(
        '--model2-name',
        type=str,
        default='model 2',
        help='name for model 2'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='output directory for results'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='significance level'
    )

    args = parser.parse_args()

    # load results
    logger.info(f"loading {args.model1}...")
    with open(args.model1) as f:
        model1_results = json.load(f)

    logger.info(f"loading {args.model2}...")
    with open(args.model2) as f:
        model2_results = json.load(f)

    # create comparator
    comparator = StatisticalComparator(alpha=args.alpha)

    # perform comparison
    comparison = comparator.compare_models(
        model1_results,
        model2_results,
        args.model1_name,
        args.model2_name
    )

    # save comparison
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_file = output_dir / 'statistical_comparison.json'
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"comparison saved to {comparison_file}")

    # generate table
    comparator.generate_comparison_table(
        [comparison],
        output_dir / 'comparison_table'
    )

    # generate plots
    comparator.plot_comparison(
        model1_results,
        model2_results,
        args.model1_name,
        args.model2_name,
        output_dir / 'plots'
    )

    # print summary
    logger.info("\n" + "=" * 80)
    logger.info("statistical comparison summary")
    logger.info("=" * 80)

    for metric, results in comparison['metrics'].items():
        logger.info(f"\n{metric.upper()}:")
        logger.info(f"  {args.model1_name}: {results['model1_mean']:.4f} ± {results['model1_std']:.4f}")
        logger.info(f"  {args.model2_name}: {results['model2_mean']:.4f} ± {results['model2_std']:.4f}")
        logger.info(f"  p-value: {results['ttest']['p_value']:.4e}")
        logger.info(f"  significant: {results['ttest']['significant']}")
        logger.info(f"  cohen's d: {results['cohens_d']:.3f} ({results['interpretation']})")

    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    main()
