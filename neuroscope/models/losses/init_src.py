"""
Loss Functions Module.

This module provides comprehensive loss functions for image-to-image translation,
particularly optimized for medical imaging and MRI domain adaptation.
"""

# Adversarial losses
from .adversarial import (
    GANLoss,
    LSGANLoss,
    WassersteinLoss,
    WassersteinGANLoss,
    HingeLoss,
    HingeGANLoss,
    VanillaGANLoss,
    RelativisticLoss,
    RelativisticAverageLoss,
    SoftplusLoss,
    MultiScaleGANLoss,
)

# Perceptual and style losses
from .perceptual import (
    VGGFeatureExtractor,
    PerceptualLoss,
    StyleLoss,
    ContentStyleLoss,
)

# Reconstruction losses
from .reconstruction import (
    L1Loss,
    L2Loss,
    SSIMLoss,
    MultiScaleSSIMLoss,
    GradientLoss,
    CharbonnierLoss,
    FocalFrequencyLoss,
)

# Consistency losses
from .consistency import (
    CycleConsistencyLoss,
    CycleLoss,
    IdentityLoss,
    FeatureMatchingLoss,
    ContrastiveConsistencyLoss,
    SemanticConsistencyLoss,
    TemporalConsistencyLoss,
    ModeSeekingLoss,
)

# Medical imaging losses
from .medical import (
    TumorPreservationLoss,
    RadiomicsPreservationLoss,
    ModalityConsistencyLoss,
    AnatomicalConsistencyLoss,
    ContrastEnhancementLoss,
    NormalizedCrossCorrelationLoss,
)

# Regularization losses
from .regularization import (
    GradientPenalty,
    SpectralRegularization,
    R1Regularization,
    R2Regularization,
    PathLengthRegularization,
    OrthogonalRegularization,
    LatentRegularization,
    ConsistencyRegularization,
    CutoutRegularization,
)

# Volumetric (3D) losses
from .volumetric import (
    VolumetricSSIM,
    VolumetricMultiScaleSSIM,
    VolumetricCycleConsistencyLoss,
    VolumetricGradientLoss,
    VolumetricPerceptualLoss,
    VolumetricNCELoss,
    VolumetricIdentityLoss,
    AnatomicalConsistencyLoss as VolumetricAnatomicalLoss,
    TissuePreservationLoss,
    CombinedVolumetricLoss,
)

__all__ = [
    # Adversarial
    'GANLoss',
    'LSGANLoss',
    'WassersteinLoss',
    'WassersteinGANLoss',
    'HingeLoss',
    'HingeGANLoss',
    'VanillaGANLoss',
    'RelativisticLoss',
    'RelativisticAverageLoss',
    'SoftplusLoss',
    'MultiScaleGANLoss',
    
    # Perceptual
    'VGGFeatureExtractor',
    'PerceptualLoss',
    'StyleLoss',
    'ContentStyleLoss',
    
    # Reconstruction
    'L1Loss',
    'L2Loss',
    'SSIMLoss',
    'MultiScaleSSIMLoss',
    'GradientLoss',
    'CharbonnierLoss',
    'FocalFrequencyLoss',
    
    # Consistency
    'CycleConsistencyLoss',
    'CycleLoss',
    'IdentityLoss',
    'FeatureMatchingLoss',
    'ContrastiveConsistencyLoss',
    'SemanticConsistencyLoss',
    'TemporalConsistencyLoss',
    'ModeSeekingLoss',
    
    # Medical
    'TumorPreservationLoss',
    'RadiomicsPreservationLoss',
    'ModalityConsistencyLoss',
    'AnatomicalConsistencyLoss',
    'ContrastEnhancementLoss',
    'NormalizedCrossCorrelationLoss',
    
    # Regularization
    'GradientPenalty',
    'SpectralRegularization',
    'R1Regularization',
    'R2Regularization',
    'PathLengthRegularization',
    'OrthogonalRegularization',
    'LatentRegularization',
    'ConsistencyRegularization',
    'CutoutRegularization',
    
    # Volumetric (3D)
    'VolumetricSSIM',
    'VolumetricMultiScaleSSIM',
    'VolumetricCycleConsistencyLoss',
    'VolumetricGradientLoss',
    'VolumetricPerceptualLoss',
    'VolumetricNCELoss',
    'VolumetricIdentityLoss',
    'VolumetricAnatomicalLoss',
    'TissuePreservationLoss',
    'CombinedVolumetricLoss',
]
