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

from .metrics import (
    # core metrics
    SSIM,
    MultiScaleSSIM,
    PSNR,
    LPIPS,
    FID,
    
    # medical metrics
    TumorPreservationScore,
    TissueContrastRatio,
    VolumePreservation,
    
    # metric collections
    ImageQualityMetrics,
    MedicalImageMetrics,
    
    # convenience functions
    compute_ssim,
    compute_psnr,
    compute_all_metrics,
)

from .statistical import (
    # hypothesis tests
    paired_t_test,
    wilcoxon_test,
    anova_test,
    
    # confidence intervals
    bootstrap_ci,
    compute_effect_size,
    
    # multiple comparisons
    bonferroni_correction,
    benjamini_hochberg,
    
    # statistical summary
    StatisticalAnalysis,
)

from .analyzers import (
    # analyzers
    ModalityAnalyzer,
    RegionAnalyzer,
    AblationAnalyzer,
    CrossDatasetAnalyzer,
    
    # analysis results
    AnalysisResult,
)

from .reporters import (
    # reporters
    LaTeXReporter,
    CSVReporter,
    JSONReporter,
    
    # report types
    EvaluationReport,
    AblationReport,
    ComparisonReport,
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
    # metrics
    'SSIM',
    'SSIMMetric',
    'MultiScaleSSIM',
    'PSNR',
    'PSNRMetric',
    'LPIPS',
    'LPIPSMetric',
    'FID',
    'FIDMetric',
    'TumorPreservationScore',
    'TissueContrastRatio',
    'VolumePreservation',
    'ImageQualityMetrics',
    'MedicalImageMetrics',
    'compute_ssim',
    'compute_psnr',
    'compute_all_metrics',
    
    # statistical
    'paired_t_test',
    'wilcoxon_test',
    'anova_test',
    'bootstrap_ci',
    'compute_effect_size',
    'bonferroni_correction',
    'benjamini_hochberg',
    'StatisticalAnalysis',
    
    # analyzers
    'ModalityAnalyzer',
    'RegionAnalyzer',
    'AblationAnalyzer',
    'CrossDatasetAnalyzer',
    'AnalysisResult',
    
    # reporters
    'LaTeXReporter',
    'CSVReporter',
    'JSONReporter',
    'EvaluationReport',
    'AblationReport',
    'ComparisonReport',
    
    # validators
    'CrossValidator',
    'HoldoutValidator',
    'TemporalValidator',
    'ValidationResult',
]
