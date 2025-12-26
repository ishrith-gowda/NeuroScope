"""
NeuroScope Evaluation Package.

Comprehensive evaluation framework for medical image harmonization
including image quality metrics, statistical analysis, and reporting.

Modules:
    - metrics: Image quality and similarity metrics
    - statistical: Statistical testing and confidence intervals
    - analyzers: Result analysis and interpretation
    - reporters: Report generation (LaTeX, CSV, JSON)
    - validators: Cross-validation and model validation
"""

from .metrics import (
    # Core metrics
    SSIM,
    MultiScaleSSIM,
    PSNR,
    LPIPS,
    FID,
    
    # Medical metrics
    TumorPreservationScore,
    TissueContrastRatio,
    VolumePreservation,
    
    # Metric collections
    ImageQualityMetrics,
    MedicalImageMetrics,
    
    # Convenience functions
    compute_ssim,
    compute_psnr,
    compute_all_metrics,
)

from .statistical import (
    # Hypothesis tests
    paired_t_test,
    wilcoxon_test,
    anova_test,
    
    # Confidence intervals
    bootstrap_ci,
    compute_effect_size,
    
    # Multiple comparisons
    bonferroni_correction,
    benjamini_hochberg,
    
    # Statistical summary
    StatisticalAnalysis,
)

from .analyzers import (
    # Analyzers
    ModalityAnalyzer,
    RegionAnalyzer,
    AblationAnalyzer,
    CrossDatasetAnalyzer,
    
    # Analysis results
    AnalysisResult,
)

from .reporters import (
    # Reporters
    LaTeXReporter,
    CSVReporter,
    JSONReporter,
    
    # Report types
    EvaluationReport,
    AblationReport,
    ComparisonReport,
)

from .validators import (
    # Validators
    CrossValidator,
    HoldoutValidator,
    TemporalValidator,
    
    # Validation results
    ValidationResult,
)

# Aliases for compatibility
SSIMMetric = SSIM
PSNRMetric = PSNR
FIDMetric = FID
LPIPSMetric = LPIPS

__all__ = [
    # Metrics
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
    
    # Statistical
    'paired_t_test',
    'wilcoxon_test',
    'anova_test',
    'bootstrap_ci',
    'compute_effect_size',
    'bonferroni_correction',
    'benjamini_hochberg',
    'StatisticalAnalysis',
    
    # Analyzers
    'ModalityAnalyzer',
    'RegionAnalyzer',
    'AblationAnalyzer',
    'CrossDatasetAnalyzer',
    'AnalysisResult',
    
    # Reporters
    'LaTeXReporter',
    'CSVReporter',
    'JSONReporter',
    'EvaluationReport',
    'AblationReport',
    'ComparisonReport',
    
    # Validators
    'CrossValidator',
    'HoldoutValidator',
    'TemporalValidator',
    'ValidationResult',
]
