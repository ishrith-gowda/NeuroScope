"""
loss functions module.

this module provides comprehensive loss functions for image-to-image translation,
particularly optimized for medical imaging and mri domain adaptation.
"""

# adversarial losses
from .adversarial import (
    GANLoss,
    HingeGANLoss,
    HingeLoss,
    LSGANLoss,
    MultiScaleGANLoss,
    RelativisticAverageLoss,
    RelativisticLoss,
    SoftplusLoss,
    VanillaGANLoss,
    WassersteinGANLoss,
    WassersteinLoss,
)

# consistency losses
from .consistency import (
    ContrastiveConsistencyLoss,
    CycleConsistencyLoss,
    CycleLoss,
    FeatureMatchingLoss,
    IdentityLoss,
    ModeSeekingLoss,
    SemanticConsistencyLoss,
    TemporalConsistencyLoss,
)

# medical imaging losses
from .medical import (
    AnatomicalConsistencyLoss,
    ContrastEnhancementLoss,
    ModalityConsistencyLoss,
    NormalizedCrossCorrelationLoss,
    RadiomicsPreservationLoss,
    TumorPreservationLoss,
)

# perceptual and style losses
from .perceptual import (
    ContentStyleLoss,
    PerceptualLoss,
    StyleLoss,
    VGGFeatureExtractor,
)

# reconstruction losses
from .reconstruction import (
    CharbonnierLoss,
    FocalFrequencyLoss,
    GradientLoss,
    L1Loss,
    L2Loss,
    MultiScaleSSIMLoss,
    SSIMLoss,
)

# regularization losses
from .regularization import (
    ConsistencyRegularization,
    CutoutRegularization,
    GradientPenalty,
    LatentRegularization,
    OrthogonalRegularization,
    PathLengthRegularization,
    R1Regularization,
    R2Regularization,
    SpectralRegularization,
)
from .volumetric import (
    AnatomicalConsistencyLoss as VolumetricAnatomicalLoss,
)

# volumetric (3d) losses
from .volumetric import (
    CombinedVolumetricLoss,
    TissuePreservationLoss,
    VolumetricCycleConsistencyLoss,
    VolumetricGradientLoss,
    VolumetricIdentityLoss,
    VolumetricMultiScaleSSIM,
    VolumetricNCELoss,
    VolumetricPerceptualLoss,
    VolumetricSSIM,
)

__all__ = [
    "AnatomicalConsistencyLoss",
    "CharbonnierLoss",
    "CombinedVolumetricLoss",
    "ConsistencyRegularization",
    "ContentStyleLoss",
    "ContrastEnhancementLoss",
    "ContrastiveConsistencyLoss",
    "CutoutRegularization",
    # consistency
    "CycleConsistencyLoss",
    "CycleLoss",
    "FeatureMatchingLoss",
    "FocalFrequencyLoss",
    # adversarial
    "GANLoss",
    "GradientLoss",
    # regularization
    "GradientPenalty",
    "HingeGANLoss",
    "HingeLoss",
    "IdentityLoss",
    # reconstruction
    "L1Loss",
    "L2Loss",
    "LSGANLoss",
    "LatentRegularization",
    "ModalityConsistencyLoss",
    "ModeSeekingLoss",
    "MultiScaleGANLoss",
    "MultiScaleSSIMLoss",
    "NormalizedCrossCorrelationLoss",
    "OrthogonalRegularization",
    "PathLengthRegularization",
    "PerceptualLoss",
    "R1Regularization",
    "R2Regularization",
    "RadiomicsPreservationLoss",
    "RelativisticAverageLoss",
    "RelativisticLoss",
    "SSIMLoss",
    "SemanticConsistencyLoss",
    "SoftplusLoss",
    "SpectralRegularization",
    "StyleLoss",
    "TemporalConsistencyLoss",
    "TissuePreservationLoss",
    # medical
    "TumorPreservationLoss",
    # perceptual
    "VGGFeatureExtractor",
    "VanillaGANLoss",
    "VolumetricAnatomicalLoss",
    "VolumetricCycleConsistencyLoss",
    "VolumetricGradientLoss",
    "VolumetricIdentityLoss",
    "VolumetricMultiScaleSSIM",
    "VolumetricNCELoss",
    "VolumetricPerceptualLoss",
    # volumetric (3d)
    "VolumetricSSIM",
    "WassersteinGANLoss",
    "WassersteinLoss",
]
