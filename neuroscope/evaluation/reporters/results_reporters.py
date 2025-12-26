"""
Report Generators.

Generate publication-ready reports in various formats
including LaTeX, CSV, and JSON.
"""

from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import csv
from datetime import datetime


@dataclass
class EvaluationReport:
    """Container for evaluation report data."""
    title: str
    methods: List[str]
    metrics: Dict[str, Dict[str, float]]  # method -> metric -> value
    statistical_tests: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.metadata['timestamp'] = datetime.now().isoformat()


@dataclass
class AblationReport:
    """Container for ablation study report."""
    title: str
    baseline: str
    configurations: Dict[str, Dict[str, float]]
    component_contributions: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonReport:
    """Container for method comparison report."""
    title: str
    methods: List[str]
    metrics: List[str]
    results_table: List[List[Any]]  # 2D table
    best_per_metric: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class LaTeXReporter:
    """
    Generate LaTeX tables and figures for publication.
    
    Creates publication-ready LaTeX code for tables
    with proper formatting and statistical annotations.
    """
    
    def __init__(
        self,
        decimal_places: int = 4,
        bold_best: bool = True,
        include_std: bool = True
    ):
        """
        Args:
            decimal_places: Number of decimal places
            bold_best: Bold the best value in each column
            include_std: Include standard deviation
        """
        self.decimal_places = decimal_places
        self.bold_best = bold_best
        self.include_std = include_std
    
    def format_value(
        self,
        value: float,
        std: Optional[float] = None,
        is_best: bool = False
    ) -> str:
        """Format a value with optional std and bolding."""
        fmt = f"{{:.{self.decimal_places}f}}"
        
        if std is not None and self.include_std:
            text = f"{fmt.format(value)} $\\pm$ {fmt.format(std)}"
        else:
            text = fmt.format(value)
        
        if is_best and self.bold_best:
            text = f"\\textbf{{{text}}}"
        
        return text
    
    def generate_comparison_table(
        self,
        report: EvaluationReport,
        caption: str = "Quantitative comparison of methods",
        label: str = "tab:comparison"
    ) -> str:
        """
        Generate comparison table.
        
        Args:
            report: Evaluation report
            caption: Table caption
            label: Table label for referencing
            
        Returns:
            LaTeX table code
        """
        methods = report.methods
        metrics = list(list(report.metrics.values())[0].keys())
        
        # Find best values per metric
        best_values = {}
        for metric in metrics:
            values = [report.metrics[m].get(metric, 0) for m in methods]
            # Assume higher is better (adjust for specific metrics)
            best_idx = max(range(len(values)), key=lambda i: values[i])
            best_values[metric] = methods[best_idx]
        
        # Generate LaTeX
        n_cols = len(metrics) + 1
        col_spec = 'l' + 'c' * len(metrics)
        
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule"
        ]
        
        # Header
        header = "Method & " + " & ".join(metrics) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")
        
        # Data rows
        for method in methods:
            row_values = [method]
            for metric in metrics:
                value = report.metrics[method].get(metric, 0)
                std = report.metrics[method].get(f"{metric}_std", None)
                is_best = best_values[metric] == method
                row_values.append(self.format_value(value, std, is_best))
            
            lines.append(" & ".join(row_values) + " \\\\")
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(lines)
    
    def generate_ablation_table(
        self,
        report: AblationReport,
        caption: str = "Ablation study results",
        label: str = "tab:ablation"
    ) -> str:
        """
        Generate ablation study table.
        
        Args:
            report: Ablation report
            caption: Table caption
            label: Table label
            
        Returns:
            LaTeX table code
        """
        configs = list(report.configurations.keys())
        metrics = list(list(report.configurations.values())[0].keys())
        
        n_cols = len(metrics) + 1
        col_spec = 'l' + 'c' * len(metrics)
        
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule"
        ]
        
        # Header
        header = "Configuration & " + " & ".join(metrics) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")
        
        # Full model first
        if report.baseline in configs:
            row = [f"\\textbf{{{report.baseline}}}"]
            for metric in metrics:
                value = report.configurations[report.baseline].get(metric, 0)
                row.append(self.format_value(value, is_best=True))
            lines.append(" & ".join(row) + " \\\\")
            lines.append("\\midrule")
        
        # Ablated configurations
        for config in configs:
            if config == report.baseline:
                continue
            row = [config]
            for metric in metrics:
                value = report.configurations[config].get(metric, 0)
                row.append(self.format_value(value))
            lines.append(" & ".join(row) + " \\\\")
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(lines)
    
    def generate_statistical_table(
        self,
        test_results: Dict[str, Dict],
        caption: str = "Statistical comparison",
        label: str = "tab:stats"
    ) -> str:
        """
        Generate table with statistical test results.
        
        Args:
            test_results: Dict of test name -> results
            caption: Table caption
            label: Table label
            
        Returns:
            LaTeX table code
        """
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "Comparison & Test & Statistic & p-value & Effect Size & Sig. \\\\",
            "\\midrule"
        ]
        
        for name, result in test_results.items():
            stat = f"{result.get('statistic', 0):.3f}"
            p_val = result.get('p_value', 1)
            
            # Format p-value
            if p_val < 0.001:
                p_str = "$<$0.001"
            else:
                p_str = f"{p_val:.3f}"
            
            effect = result.get('effect_size', 0)
            effect_str = f"{effect:.3f}" if effect else "-"
            
            sig = "$\\checkmark$" if result.get('significant', False) else ""
            
            row = f"{name} & {result.get('test_name', '')} & {stat} & {p_str} & {effect_str} & {sig} \\\\"
            lines.append(row)
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(lines)
    
    def save(self, content: str, path: Union[str, Path]):
        """Save LaTeX content to file."""
        with open(path, 'w') as f:
            f.write(content)


class CSVReporter:
    """
    Generate CSV reports for data analysis.
    
    Creates machine-readable CSV files for further processing.
    """
    
    def __init__(self, delimiter: str = ','):
        self.delimiter = delimiter
    
    def generate_comparison_csv(
        self,
        report: EvaluationReport,
        include_metadata: bool = True
    ) -> List[List[str]]:
        """
        Generate comparison CSV data.
        
        Args:
            report: Evaluation report
            include_metadata: Include metadata rows
            
        Returns:
            2D list of CSV rows
        """
        rows = []
        
        if include_metadata:
            rows.append(['# Title', report.title])
            rows.append(['# Timestamp', report.metadata.get('timestamp', '')])
            rows.append([])
        
        # Header
        methods = report.methods
        metrics = list(list(report.metrics.values())[0].keys())
        rows.append(['Method'] + metrics)
        
        # Data
        for method in methods:
            row = [method]
            for metric in metrics:
                row.append(str(report.metrics[method].get(metric, '')))
            rows.append(row)
        
        return rows
    
    def save(
        self,
        rows: List[List[str]],
        path: Union[str, Path]
    ):
        """Save CSV data to file."""
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=self.delimiter)
            writer.writerows(rows)


class JSONReporter:
    """
    Generate JSON reports for programmatic access.
    
    Creates structured JSON files with complete results.
    """
    
    def __init__(self, indent: int = 2):
        self.indent = indent
    
    def generate_full_report(
        self,
        report: Union[EvaluationReport, AblationReport, ComparisonReport]
    ) -> Dict:
        """
        Generate complete JSON report.
        
        Args:
            report: Any report type
            
        Returns:
            Dict representation
        """
        if isinstance(report, EvaluationReport):
            return {
                'type': 'evaluation',
                'title': report.title,
                'methods': report.methods,
                'metrics': report.metrics,
                'statistical_tests': report.statistical_tests,
                'metadata': report.metadata
            }
        
        elif isinstance(report, AblationReport):
            return {
                'type': 'ablation',
                'title': report.title,
                'baseline': report.baseline,
                'configurations': report.configurations,
                'component_contributions': report.component_contributions,
                'metadata': report.metadata
            }
        
        elif isinstance(report, ComparisonReport):
            return {
                'type': 'comparison',
                'title': report.title,
                'methods': report.methods,
                'metrics': report.metrics,
                'results_table': report.results_table,
                'best_per_metric': report.best_per_metric,
                'metadata': report.metadata
            }
        
        return {}
    
    def save(
        self,
        data: Dict,
        path: Union[str, Path]
    ):
        """Save JSON data to file."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=self.indent)


class ReportGenerator:
    """
    Unified report generator.
    
    Generates reports in multiple formats simultaneously.
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        formats: List[str] = None
    ):
        """
        Args:
            output_dir: Output directory for reports
            formats: List of formats ('latex', 'csv', 'json')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.formats = formats or ['latex', 'csv', 'json']
        
        self.latex = LaTeXReporter()
        self.csv = CSVReporter()
        self.json = JSONReporter()
    
    def generate_evaluation_report(
        self,
        report: EvaluationReport,
        name: str = 'evaluation'
    ) -> Dict[str, Path]:
        """
        Generate evaluation report in all formats.
        
        Args:
            report: Evaluation report
            name: Base filename
            
        Returns:
            Dict of format -> file path
        """
        paths = {}
        
        if 'latex' in self.formats:
            latex_content = self.latex.generate_comparison_table(report)
            latex_path = self.output_dir / f"{name}.tex"
            self.latex.save(latex_content, latex_path)
            paths['latex'] = latex_path
        
        if 'csv' in self.formats:
            csv_data = self.csv.generate_comparison_csv(report)
            csv_path = self.output_dir / f"{name}.csv"
            self.csv.save(csv_data, csv_path)
            paths['csv'] = csv_path
        
        if 'json' in self.formats:
            json_data = self.json.generate_full_report(report)
            json_path = self.output_dir / f"{name}.json"
            self.json.save(json_data, json_path)
            paths['json'] = json_path
        
        return paths
    
    def generate_ablation_report(
        self,
        report: AblationReport,
        name: str = 'ablation'
    ) -> Dict[str, Path]:
        """Generate ablation report in all formats."""
        paths = {}
        
        if 'latex' in self.formats:
            latex_content = self.latex.generate_ablation_table(report)
            latex_path = self.output_dir / f"{name}.tex"
            self.latex.save(latex_content, latex_path)
            paths['latex'] = latex_path
        
        if 'json' in self.formats:
            json_data = self.json.generate_full_report(report)
            json_path = self.output_dir / f"{name}.json"
            self.json.save(json_data, json_path)
            paths['json'] = json_path
        
        return paths
