"""
experiments module.

experiment runners, ablation studies, and
reproducible experiment management.
"""

from .ablation import AblationConfig, AblationStudy, run_ablation_suite
from .analysis import ExperimentAnalyzer, compare_experiments, generate_comparison_report
from .runner import AblationRunner, BaselineRunner, ExperimentRunner

__all__ = [
    "AblationConfig",
    "AblationRunner",
    # ablation
    "AblationStudy",
    "BaselineRunner",
    # analysis
    "ExperimentAnalyzer",
    # runners
    "ExperimentRunner",
    "compare_experiments",
    "generate_comparison_report",
    "run_ablation_suite",
]
