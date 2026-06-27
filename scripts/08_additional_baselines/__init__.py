"""
additional baseline harmonization methods module.

provides classical normalization methods for comparison
with deep learning-based harmonization.
"""

from .baseline_methods import (
    HistogramMatcher,
    IntensityRangeNormalizer,
    NormalizationConfig,
    NyulNormalizer,
    WhiteStripeNormalizer,
    ZScoreNormalizer,
    apply_baseline_harmonization,
    evaluate_baseline_method,
)

__all__ = [
    "HistogramMatcher",
    "IntensityRangeNormalizer",
    "NormalizationConfig",
    "NyulNormalizer",
    "WhiteStripeNormalizer",
    "ZScoreNormalizer",
    "apply_baseline_harmonization",
    "evaluate_baseline_method",
]
