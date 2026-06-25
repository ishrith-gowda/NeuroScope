"""
neuroscope evaluation package.

comprehensive evaluation framework for medical image harmonization
including image quality metrics, statistical analysis, and reporting.

modules:
    - metrics: image quality and similarity metrics
    - statistical: statistical testing and confidence intervals
    - analyzers: result analysis and interpretation
    - reporters: report generation (latex, csv, json)
    - validators: cross-validation and model validation
"""

from .analyzers import (
    AblationAnalyzer,
    # analysis results
    AnalysisResult,
    CrossDatasetAnalyzer,
    # analyzers
    ModalityAnalyzer,
    RegionAnalyzer,
)
from .metrics import (
    FID,
    LPIPS,
    PSNR,
    # core metrics
    SSIM,
    # metric collections
    ImageQualityMetrics,
    MedicalImageMetrics,
    MultiScaleSSIM,
    TissueContrastRatio,
    # medical metrics
    TumorPreservationScore,
    VolumePreservation,
    compute_all_metrics,
    compute_psnr,
    # convenience functions
    compute_ssim,
)
from .reporters import (
    AblationReport,
    ComparisonReport,
    CSVReporter,
    # report types
    EvaluationReport,
    JSONReporter,
    # reporters
    LaTeXReporter,
)
from .statistical import (
    # statistical summary
    StatisticalAnalysis,
    anova_test,
    benjamini_hochberg,
    # multiple comparisons
    bonferroni_correction,
    # confidence intervals
    bootstrap_ci,
    compute_effect_size,
    # hypothesis tests
    paired_t_test,
    wilcoxon_test,
)
from .validators import (
    # validators
    CrossValidator,
    HoldoutValidator,
    TemporalValidator,
    # validation results
    ValidationResult,
)

# aliases for compatibility
SSIMMetric = SSIM
PSNRMetric = PSNR
FIDMetric = FID
LPIPSMetric = LPIPS

__all__ = [
    "FID",
    "LPIPS",
    "PSNR",
    # metrics
    "SSIM",
    "AblationAnalyzer",
    "AblationReport",
    "AnalysisResult",
    "CSVReporter",
    "ComparisonReport",
    "CrossDatasetAnalyzer",
    # validators
    "CrossValidator",
    "EvaluationReport",
    "FIDMetric",
    "HoldoutValidator",
    "ImageQualityMetrics",
    "JSONReporter",
    "LPIPSMetric",
    # reporters
    "LaTeXReporter",
    "MedicalImageMetrics",
    # analyzers
    "ModalityAnalyzer",
    "MultiScaleSSIM",
    "PSNRMetric",
    "RegionAnalyzer",
    "SSIMMetric",
    "StatisticalAnalysis",
    "TemporalValidator",
    "TissueContrastRatio",
    "TumorPreservationScore",
    "ValidationResult",
    "VolumePreservation",
    "anova_test",
    "benjamini_hochberg",
    "bonferroni_correction",
    "bootstrap_ci",
    "compute_all_metrics",
    "compute_effect_size",
    "compute_psnr",
    "compute_ssim",
    # statistical
    "paired_t_test",
    "wilcoxon_test",
]
