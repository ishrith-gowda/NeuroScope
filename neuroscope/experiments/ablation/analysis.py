"""
Experiment Analysis.

Compare experiments and generate publication-ready
reports and visualizations.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import numpy as np


class ExperimentAnalyzer:
    """
    Analyze and compare multiple experiments.
    
    Generates statistical comparisons, visualizations,
    and publication-ready reports.
    """
    
    def __init__(self, experiments: Dict[str, Dict]):
        """
        Args:
            experiments: Dict of experiment name -> results
        """
        self.experiments = experiments
        self.metrics = ['ssim', 'psnr', 'fid', 'lpips']
    
    def compute_statistics(self) -> Dict:
        """
        Compute statistics for all experiments.
        
        Returns:
            Statistics dictionary
        """
        stats = {}
        
        for name, results in self.experiments.items():
            metrics = results.get('test_metrics', {})
            
            stats[name] = {}
            for metric in self.metrics:
                if metric in metrics:
                    value = metrics[metric]
                    if isinstance(value, list):
                        stats[name][metric] = {
                            'mean': float(np.mean(value)),
                            'std': float(np.std(value)),
                            'median': float(np.median(value)),
                            'min': float(np.min(value)),
                            'max': float(np.max(value))
                        }
                    else:
                        stats[name][metric] = {'value': value}
        
        return stats
    
    def rank_experiments(
        self,
        metric: str = 'ssim',
        higher_is_better: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Rank experiments by metric.
        
        Args:
            metric: Metric to rank by
            higher_is_better: Whether higher values are better
            
        Returns:
            Sorted list of (name, value) tuples
        """
        rankings = []
        
        for name, results in self.experiments.items():
            metrics = results.get('test_metrics', {})
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, (list, np.ndarray)):
                    value = np.mean(value)
                rankings.append((name, value))
        
        rankings.sort(key=lambda x: x[1], reverse=higher_is_better)
        return rankings
    
    def compute_improvements(
        self,
        baseline_name: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute improvements over baseline.
        
        Args:
            baseline_name: Name of baseline experiment
            
        Returns:
            Dict of experiment -> metric -> improvement
        """
        improvements = {}
        
        baseline = self.experiments.get(baseline_name, {})
        baseline_metrics = baseline.get('test_metrics', {})
        
        for name, results in self.experiments.items():
            if name == baseline_name:
                continue
            
            metrics = results.get('test_metrics', {})
            improvements[name] = {}
            
            for metric in self.metrics:
                if metric in metrics and metric in baseline_metrics:
                    base_val = baseline_metrics[metric]
                    curr_val = metrics[metric]
                    
                    if isinstance(base_val, list):
                        base_val = np.mean(base_val)
                    if isinstance(curr_val, list):
                        curr_val = np.mean(curr_val)
                    
                    # Calculate relative improvement
                    if base_val != 0:
                        rel_improvement = (curr_val - base_val) / abs(base_val) * 100
                    else:
                        rel_improvement = 0
                    
                    improvements[name][metric] = {
                        'absolute': float(curr_val - base_val),
                        'relative_percent': float(rel_improvement)
                    }
        
        return improvements
    
    def generate_comparison_table(
        self,
        metrics: List[str] = None,
        format: str = 'latex'
    ) -> str:
        """
        Generate comparison table.
        
        Args:
            metrics: Metrics to include
            format: 'latex', 'markdown', or 'csv'
            
        Returns:
            Formatted table string
        """
        metrics = metrics or self.metrics
        stats = self.compute_statistics()
        
        if format == 'latex':
            return self._latex_table(stats, metrics)
        elif format == 'markdown':
            return self._markdown_table(stats, metrics)
        else:
            return self._csv_table(stats, metrics)
    
    def _latex_table(
        self,
        stats: Dict,
        metrics: List[str]
    ) -> str:
        """Generate LaTeX table."""
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Quantitative Comparison of Methods}",
            "\\label{tab:comparison}",
            "\\begin{tabular}{l" + "c" * len(metrics) + "}",
            "\\toprule",
            "Method & " + " & ".join(m.upper() for m in metrics) + " \\\\",
            "\\midrule"
        ]
        
        for name, data in stats.items():
            row = [name.replace('_', ' ').title()]
            for metric in metrics:
                if metric in data:
                    if 'mean' in data[metric]:
                        val = f"{data[metric]['mean']:.4f}"
                        std = f"±{data[metric]['std']:.4f}"
                        row.append(f"${val}$ {std}")
                    else:
                        row.append(f"${data[metric]['value']:.4f}$")
                else:
                    row.append("-")
            
            lines.append(" & ".join(row) + " \\\\")
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return '\n'.join(lines)
    
    def _markdown_table(
        self,
        stats: Dict,
        metrics: List[str]
    ) -> str:
        """Generate Markdown table."""
        lines = [
            "| Method | " + " | ".join(m.upper() for m in metrics) + " |",
            "|--------|" + "|".join("-" * 8 for _ in metrics) + "|"
        ]
        
        for name, data in stats.items():
            row = [name.replace('_', ' ').title()]
            for metric in metrics:
                if metric in data:
                    if 'mean' in data[metric]:
                        val = f"{data[metric]['mean']:.4f}±{data[metric]['std']:.4f}"
                    else:
                        val = f"{data[metric]['value']:.4f}"
                    row.append(val)
                else:
                    row.append("-")
            
            lines.append("| " + " | ".join(row) + " |")
        
        return '\n'.join(lines)
    
    def _csv_table(
        self,
        stats: Dict,
        metrics: List[str]
    ) -> str:
        """Generate CSV table."""
        lines = ["Method," + ",".join(metrics)]
        
        for name, data in stats.items():
            row = [name]
            for metric in metrics:
                if metric in data:
                    if 'mean' in data[metric]:
                        row.append(str(data[metric]['mean']))
                    else:
                        row.append(str(data[metric]['value']))
                else:
                    row.append("")
            
            lines.append(",".join(row))
        
        return '\n'.join(lines)


def compare_experiments(
    experiment_dirs: List[Path],
    output_dir: Path
) -> Dict:
    """
    Compare multiple experiments.
    
    Args:
        experiment_dirs: List of experiment directories
        output_dir: Output directory for reports
        
    Returns:
        Comparison results
    """
    experiments = {}
    
    for exp_dir in experiment_dirs:
        results_file = exp_dir / 'results' / 'final_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                experiments[exp_dir.name] = json.load(f)
    
    analyzer = ExperimentAnalyzer(experiments)
    
    # Generate comparison
    comparison = {
        'statistics': analyzer.compute_statistics(),
        'rankings': {
            'ssim': analyzer.rank_experiments('ssim', True),
            'psnr': analyzer.rank_experiments('psnr', True),
            'fid': analyzer.rank_experiments('fid', False),
            'lpips': analyzer.rank_experiments('lpips', False)
        }
    }
    
    # Save comparison
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Generate tables
    with open(output_dir / 'comparison_table.tex', 'w') as f:
        f.write(analyzer.generate_comparison_table(format='latex'))
    
    with open(output_dir / 'comparison_table.md', 'w') as f:
        f.write(analyzer.generate_comparison_table(format='markdown'))
    
    return comparison


def generate_comparison_report(
    experiments: Dict[str, Dict],
    output_path: Path,
    baseline_name: str = 'baseline'
) -> str:
    """
    Generate comprehensive comparison report.
    
    Args:
        experiments: Dict of experiment results
        output_path: Output path for report
        baseline_name: Name of baseline experiment
        
    Returns:
        Report content
    """
    analyzer = ExperimentAnalyzer(experiments)
    
    stats = analyzer.compute_statistics()
    improvements = analyzer.compute_improvements(baseline_name)
    
    lines = [
        "# Experiment Comparison Report",
        "",
        "## Overview",
        "",
        f"Total experiments compared: {len(experiments)}",
        f"Baseline: {baseline_name}",
        "",
        "## Performance Summary",
        "",
        analyzer.generate_comparison_table(format='markdown'),
        "",
        "## Improvements Over Baseline",
        ""
    ]
    
    for exp_name, imp in improvements.items():
        lines.append(f"### {exp_name}")
        lines.append("")
        for metric, values in imp.items():
            lines.append(
                f"- **{metric.upper()}**: "
                f"{values['absolute']:+.4f} "
                f"({values['relative_percent']:+.2f}%)"
            )
        lines.append("")
    
    # Rankings
    lines.extend([
        "## Rankings",
        "",
        "### By SSIM (higher is better)",
        ""
    ])
    
    for i, (name, value) in enumerate(analyzer.rank_experiments('ssim'), 1):
        lines.append(f"{i}. **{name}**: {value:.4f}")
    
    lines.extend([
        "",
        "### By PSNR (higher is better)",
        ""
    ])
    
    for i, (name, value) in enumerate(analyzer.rank_experiments('psnr'), 1):
        lines.append(f"{i}. **{name}**: {value:.2f} dB")
    
    content = '\n'.join(lines)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)
    
    return content
