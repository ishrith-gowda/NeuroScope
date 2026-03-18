"""
discriminator architectures module.

this module provides various discriminator architectures for
adversarial training in image-to-image translation.
"""

# base classes
from .base import (
    BaseDiscriminator,
    PatchDiscriminator,
    MultiScaleDiscriminatorBase,
    ConditionalDiscriminator,
    ProjectionDiscriminator,
    FeatureMatchingDiscriminator,
)

# patch discriminators
from .patch import (
    NLayerPatchDiscriminator,
    PixelDiscriminator,
    DeepPatchDiscriminator,
    ResidualPatchDiscriminator,
    DilatedPatchDiscriminator,
    AttentionPatchDiscriminator,
)

# multi-scale discriminators
from .multiscale import (
    MultiScaleDiscriminator,
    PyramidDiscriminator,
    SharedEncoderMultiScaleDiscriminator,
    AdaptiveMultiScaleDiscriminator,
    ProgressiveMultiScaleDiscriminator,
    DualScaleDiscriminator,
)

# spectral normalization discriminators
from .spectral import (
    SpectralNormDiscriminator,
    SNResNetDiscriminator,
    SNProjectionDiscriminator,
    SNMultiScaleDiscriminator,
    SNSelfAttentionDiscriminator,
    SNUNetDiscriminator,
)

# aliases for compatibility
PatchGANDiscriminator = NLayerPatchDiscriminator

__all__ = [
    # base
    'BaseDiscriminator',
    'PatchDiscriminator',
    'MultiScaleDiscriminatorBase',
    'ConditionalDiscriminator',
    'ProjectionDiscriminator',
    'FeatureMatchingDiscriminator',
    
    # patch
    'NLayerPatchDiscriminator',
    'PatchGANDiscriminator',
    'PixelDiscriminator',
    'DeepPatchDiscriminator',
    'ResidualPatchDiscriminator',
    'DilatedPatchDiscriminator',
    'AttentionPatchDiscriminator',
    
    # multi-scale
    'MultiScaleDiscriminator',
    'PyramidDiscriminator',
    'SharedEncoderMultiScaleDiscriminator',
    'AdaptiveMultiScaleDiscriminator',
    'ProgressiveMultiScaleDiscriminator',
    'DualScaleDiscriminator',
    
    # spectral
    'SpectralNormDiscriminator',
    'SNResNetDiscriminator',
    'SNProjectionDiscriminator',
    'SNMultiScaleDiscriminator',
    'SNSelfAttentionDiscriminator',
    'SNUNetDiscriminator',
]
