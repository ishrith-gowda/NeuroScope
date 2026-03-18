"""
loss functions module.

this module provides comprehensive loss functions for image-to-image translation,
particularly optimized for medical imaging and mri domain adaptation.
"""

# adversarial losses
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

# perceptual and style losses
from .perceptual import (
    VGGFeatureExtractor,
    PerceptualLoss,
    StyleLoss,
    ContentStyleLoss,
)

# reconstruction losses
from .reconstruction import (
    L1Loss,
    L2Loss,
    SSIMLoss,
    MultiScaleSSIMLoss,
    GradientLoss,
    CharbonnierLoss,
    FocalFrequencyLoss,
)

# consistency losses
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

# medical imaging losses
from .medical import (
    TumorPreservationLoss,
    RadiomicsPreservationLoss,
    ModalityConsistencyLoss,
    AnatomicalConsistencyLoss,
    ContrastEnhancementLoss,
    NormalizedCrossCorrelationLoss,
)

# regularization losses
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

# volumetric (3d) losses
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
    # adversarial
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
    
    # perceptual
    'VGGFeatureExtractor',
    'PerceptualLoss',
    'StyleLoss',
    'ContentStyleLoss',
    
    # reconstruction
    'L1Loss',
    'L2Loss',
    'SSIMLoss',
    'MultiScaleSSIMLoss',
    'GradientLoss',
    'CharbonnierLoss',
    'FocalFrequencyLoss',
    
    # consistency
    'CycleConsistencyLoss',
    'CycleLoss',
    'IdentityLoss',
    'FeatureMatchingLoss',
    'ContrastiveConsistencyLoss',
    'SemanticConsistencyLoss',
    'TemporalConsistencyLoss',
    'ModeSeekingLoss',
    
    # medical
    'TumorPreservationLoss',
    'RadiomicsPreservationLoss',
    'ModalityConsistencyLoss',
    'AnatomicalConsistencyLoss',
    'ContrastEnhancementLoss',
    'NormalizedCrossCorrelationLoss',
    
    # regularization
    'GradientPenalty',
    'SpectralRegularization',
    'R1Regularization',
    'R2Regularization',
    'PathLengthRegularization',
    'OrthogonalRegularization',
    'LatentRegularization',
    'ConsistencyRegularization',
    'CutoutRegularization',
    
    # volumetric (3d)
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
