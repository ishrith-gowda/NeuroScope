"""
Experiments Module.

Experiment runners, ablation studies, and
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
    # Runners
    'ExperimentRunner',
    'AblationRunner',
    'BaselineRunner',
    
    # Ablation
    'AblationStudy',
    'AblationConfig',
    'run_ablation_suite',
    
    # Analysis
    'ExperimentAnalyzer',
    'compare_experiments',
    'generate_comparison_report',
]
