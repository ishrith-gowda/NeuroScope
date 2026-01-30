"""
additional baseline harmonization methods module.

provides classical normalization methods for comparison
with deep learning-based harmonization.
"""

from .baseline_methods import (
    NormalizationConfig,
    ZScoreNormalizer,
    IntensityRangeNormalizer,
    HistogramMatcher,
    NyulNormalizer,
    WhiteStripeNormalizer,
    apply_baseline_harmonization,
    evaluate_baseline_method,
)

__all__ = [
    'NormalizationConfig',
    'ZScoreNormalizer',
    'IntensityRangeNormalizer',
    'HistogramMatcher',
    'NyulNormalizer',
    'WhiteStripeNormalizer',
    'apply_baseline_harmonization',
    'evaluate_baseline_method',
]
