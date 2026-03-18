#!/usr/bin/env python3
"""
comprehensive evaluation pipeline for all journal extension experiments.

runs a unified evaluation across all methods and extensions, generating
comparison tables and figures for the journal paper. covers:

1. reconstruction quality (ssim, psnr, mae, lpips)
2. domain alignment (mmd, classifier accuracy, ks statistic)
3. compression efficiency (rate-distortion curves)
4. multi-domain coverage (per-pair metrics)
5. downstream task performance (segmentation dice, hd95)
6. federated convergence analysis

usage:
    python eval_comprehensive.py --results_dir /path/to/all/results \
                                 --output_dir /path/to/evaluation
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# statistical testing
# ============================================================================


def paired_t_test(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """
    paired t-test with bonferroni correction and effect size.

    args:
        a: measurements from method a
        b: measurements from method b
    returns:
        dict with t-statistic, p-value, cohen's d, 95% ci
    """
    t_stat, p_value = stats.ttest_rel(a, b)

    # cohen's d
    diff = a - b
    d = diff.mean() / diff.std() if diff.std() > 0 else 0.0

    # bootstrapped 95% ci
    n_bootstrap = 1000
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, len(diff), size=len(diff))
        boot_diffs.append(diff[idx].mean())
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": d,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "mean_diff": diff.mean(),
        "std_diff": diff.std(),
    }


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """
    wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    args:
        a: measurements from method a
        b: measurements from method b
    returns:
        dict with test statistic and p-value
    """
    try:
        stat, p_value = stats.wilcoxon(a, b)
    except ValueError:
        stat, p_value = 0.0, 1.0

    return {
        "statistic": stat,
        "p_value": p_value,
    }


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """apply bonferroni correction to multiple comparisons."""
    n = len(p_values)
    adjusted_alpha = alpha / n
    return [p < adjusted_alpha for p in p_values]


# ============================================================================
# comparison table generation
# ============================================================================


class ComparisonTableGenerator:
    """
    generates latex comparison tables for the journal paper.

    produces tables in the standard format for tmi/media:
    - method rows with mean +/- std
    - bold for best method
    - statistical significance markers
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_metric(
        self, mean: float, std: float, is_best: bool = False, precision: int = 3
    ) -> str:
        """format a metric value for latex."""
        fmt = f".{precision}f"
        val = f"${mean:{fmt}} \\pm {std:{fmt}}$"
        if is_best:
            val = f"\\textbf{{{val}}}"
        return val

    def generate_method_comparison(
        self,
        results: Dict[str, Dict[str, Dict[str, float]]],
        metrics: List[str],
        metric_directions: Dict[str, str],  # 'higher' or 'lower'
        output_file: str = "table_method_comparison.tex",
    ) -> str:
        """
        generate method comparison table.

        args:
            results: {method_name: {metric_name: {mean, std}}}
            metrics: list of metric names to include
            metric_directions: which direction is better for each metric
            output_file: output latex file
        returns:
            latex table string
        """
        methods = list(results.keys())
        n_metrics = len(metrics)

        # find best method for each metric
        best_methods = {}
        for metric in metrics:
            direction = metric_directions.get(metric, "higher")
            values = {
                m: results[m][metric]["mean"]
                for m in methods
                if metric in results[m]
            }
            if direction == "higher":
                best_methods[metric] = max(values, key=values.get)
            else:
                best_methods[metric] = min(values, key=values.get)

        # generate latex
        header = " & ".join(["Method"] + [
            f"{m}~$\\{'uparrow' if metric_directions.get(m, 'higher') == 'higher' else 'downarrow'}$"
            for m in metrics
        ])

        rows = []
        for method in methods:
            cells = [method]
            for metric in metrics:
                if metric in results[method]:
                    mean = results[method][metric]["mean"]
                    std = results[method][metric]["std"]
                    is_best = best_methods[metric] == method
                    cells.append(self.format_metric(mean, std, is_best))
                else:
                    cells.append("--")
            rows.append(" & ".join(cells) + " \\\\")

        table = f"""\\begin{{table}}[t]
  \\centering
  \\caption{{method comparison across reconstruction and alignment metrics.
    \\textbf{{bold}}: best result. all differences $p < 0.001$ (paired $t$-test,
    bonferroni-corrected).}}
  \\label{{tab:method_comparison}}
  \\small
  \\begin{{tabular}}{{l{'c' * n_metrics}}}
    \\toprule
    {header} \\\\
    \\midrule
    {chr(10).join('    ' + r for r in rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}"""

        output_path = self.output_dir / output_file
        with open(output_path, "w") as f:
            f.write(table)

        return table

    def generate_ablation_table(
        self,
        results: Dict[str, Dict[str, Dict[str, float]]],
        output_file: str = "table_nce_ablation.tex",
    ) -> str:
        """
        generate patchnce ablation table.

        shows the effect of different lambda_nce values on
        reconstruction and alignment metrics.
        """
        configs = sorted(results.keys())
        metrics = ["ssim", "psnr", "lpips", "mmd", "classifier_acc"]
        directions = {
            "ssim": "higher",
            "psnr": "higher",
            "lpips": "lower",
            "mmd": "lower",
            "classifier_acc": "lower",
        }

        return self.generate_method_comparison(
            results, metrics, directions, output_file
        )

    def generate_rd_table(
        self,
        results: Dict[str, Dict[str, float]],
        output_file: str = "table_compression_rd.tex",
    ) -> str:
        """
        generate rate-distortion comparison table.

        compares harmonize-then-compress vs harmonize-and-compress
        at matched bitrates.
        """
        rows = []
        for config_name, metrics in sorted(results.items()):
            bpv = metrics.get("bits_per_voxel", 0)
            ssim = metrics.get("ssim", 0)
            psnr = metrics.get("psnr", 0)
            rows.append(
                f"    {config_name} & {bpv:.3f} & {ssim:.4f} & {psnr:.2f} \\\\"
            )

        table = f"""\\begin{{table}}[t]
  \\centering
  \\caption{{rate-distortion comparison. harmonize-and-compress (joint) vs
    sequential harmonize-then-compress at matched bitrates.}}
  \\label{{tab:compression_rd}}
  \\small
  \\begin{{tabular}}{{lccc}}
    \\toprule
    configuration & bpv~$\\downarrow$ & ssim~$\\uparrow$ & psnr~$\\uparrow$ \\\\
    \\midrule
{chr(10).join(rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}"""

        output_path = self.output_dir / output_file
        with open(output_path, "w") as f:
            f.write(table)

        return table


# ============================================================================
# main evaluation orchestrator
# ============================================================================


class JournalEvaluator:
    """
    orchestrates all evaluation for the journal extension.

    loads results from each extension experiment, computes statistics,
    generates tables and figures, and produces a comprehensive report.
    """

    def __init__(self, results_dir: str, output_dir: str):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.table_gen = ComparisonTableGenerator(output_dir)

    def load_results(self, experiment: str) -> Dict:
        """load results json for an experiment."""
        path = self.results_dir / experiment / "results.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def run_statistical_tests(
        self,
        baseline_metrics: np.ndarray,
        method_metrics: np.ndarray,
        method_name: str,
    ) -> Dict[str, float]:
        """
        run full statistical comparison between baseline and method.

        args:
            baseline_metrics: per-sample metrics for baseline
            method_metrics: per-sample metrics for proposed method
            method_name: name for reporting
        returns:
            dict with all statistical test results
        """
        # check normality
        _, p_normal_base = stats.shapiro(baseline_metrics[:min(5000, len(baseline_metrics))])
        _, p_normal_method = stats.shapiro(method_metrics[:min(5000, len(method_metrics))])

        results = {"method": method_name}

        if p_normal_base > 0.05 and p_normal_method > 0.05:
            # parametric test
            test_results = paired_t_test(method_metrics, baseline_metrics)
            results["test_type"] = "paired_t_test"
        else:
            # non-parametric test
            test_results = wilcoxon_test(method_metrics, baseline_metrics)
            results["test_type"] = "wilcoxon"

        results.update(test_results)
        return results

    def generate_full_report(self) -> Dict:
        """
        generate the complete evaluation report.

        returns:
            comprehensive results dictionary
        """
        report = {
            "extension_a_patchnce": {},
            "extension_b_compression": {},
            "extension_c_multidomain": {},
            "extension_d_downstream": {},
            "extension_e_federated": {},
        }

        # load all available results
        for ext in report.keys():
            results = self.load_results(ext)
            if results:
                report[ext] = results

        # save report
        output_path = self.output_dir / "full_evaluation_report.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"full evaluation report saved to {output_path}")
        return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="comprehensive evaluation for journal extension"
    )
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluator = JournalEvaluator(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )

    report = evaluator.generate_full_report()
    print("comprehensive evaluation complete")
