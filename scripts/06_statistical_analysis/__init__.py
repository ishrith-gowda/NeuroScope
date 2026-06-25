"""
statistical analysis and state-of-the-art comparison module.

provides comprehensive statistical validation for mri harmonization:
- bootstrap confidence intervals
- effect size calculations
- significance testing with multiple comparison correction
- combat baseline comparison
- publication-ready figures and tables
"""

from .combat_comparison import (
    ComBatConfig,
    ComBatHarmonizer,
    evaluate_combat_harmonization,
    harmonize_mri_with_combat,
)
from .comprehensive_statistics import (
    BootstrapCI,
    EffectSizeCalculator,
    HarmonizationStatistics,
    MultipleComparisonCorrection,
    StatisticalTests,
)

__all__ = [
    "BootstrapCI",
    "ComBatConfig",
    "ComBatHarmonizer",
    "EffectSizeCalculator",
    "HarmonizationStatistics",
    "MultipleComparisonCorrection",
    "StatisticalTests",
    "evaluate_combat_harmonization",
    "harmonize_mri_with_combat",
]
