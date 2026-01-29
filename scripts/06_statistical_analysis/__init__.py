"""
statistical analysis and state-of-the-art comparison module.

provides comprehensive statistical validation for mri harmonization:
- bootstrap confidence intervals
- effect size calculations
- significance testing with multiple comparison correction
- combat baseline comparison
- publication-ready figures and tables
"""

from .comprehensive_statistics import (
    BootstrapCI,
    EffectSizeCalculator,
    StatisticalTests,
    MultipleComparisonCorrection,
    HarmonizationStatistics,
)

from .combat_comparison import (
    ComBatConfig,
    ComBatHarmonizer,
    harmonize_mri_with_combat,
    evaluate_combat_harmonization,
)

__all__ = [
    'BootstrapCI',
    'EffectSizeCalculator',
    'StatisticalTests',
    'MultipleComparisonCorrection',
    'HarmonizationStatistics',
    'ComBatConfig',
    'ComBatHarmonizer',
    'harmonize_mri_with_combat',
    'evaluate_combat_harmonization',
]
