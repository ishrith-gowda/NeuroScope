"""
experiments module.

experiment runners, ablation studies, and
reproducible experiment management.
"""

from .runner import (
    ExperimentRunner,
    AblationRunner,
    BaselineRunner
)

from .ablation import (
    AblationStudy,
    AblationConfig,
    run_ablation_suite
)

from .analysis import (
    ExperimentAnalyzer,
    compare_experiments,
    generate_comparison_report
)


__all__ = [
    # runners
    'ExperimentRunner',
    'AblationRunner',
    'BaselineRunner',
    
    # ablation
    'AblationStudy',
    'AblationConfig',
    'run_ablation_suite',
    
    # analysis
    'ExperimentAnalyzer',
    'compare_experiments',
    'generate_comparison_report',
]
