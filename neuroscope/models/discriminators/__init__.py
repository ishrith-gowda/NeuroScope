"""
Discriminator Architectures Module.

This module provides various discriminator architectures for
adversarial training in image-to-image translation.
"""

# Base classes
from .base import (
    BaseDiscriminator,
    PatchDiscriminator,
    MultiScaleDiscriminatorBase,
    ConditionalDiscriminator,
    ProjectionDiscriminator,
    FeatureMatchingDiscriminator,
)

# Patch discriminators
from .patch import (
    NLayerPatchDiscriminator,
    PixelDiscriminator,
    DeepPatchDiscriminator,
    ResidualPatchDiscriminator,
    DilatedPatchDiscriminator,
    AttentionPatchDiscriminator,
)

# Multi-scale discriminators
from .multiscale import (
    MultiScaleDiscriminator,
    PyramidDiscriminator,
    SharedEncoderMultiScaleDiscriminator,
    AdaptiveMultiScaleDiscriminator,
    ProgressiveMultiScaleDiscriminator,
    DualScaleDiscriminator,
)

# Spectral normalization discriminators
from .spectral import (
    SpectralNormDiscriminator,
    SNResNetDiscriminator,
    SNProjectionDiscriminator,
    SNMultiScaleDiscriminator,
    SNSelfAttentionDiscriminator,
    SNUNetDiscriminator,
)

# Aliases for compatibility
PatchGANDiscriminator = NLayerPatchDiscriminator

__all__ = [
    # Base
    'BaseDiscriminator',
    'PatchDiscriminator',
    'MultiScaleDiscriminatorBase',
    'ConditionalDiscriminator',
    'ProjectionDiscriminator',
    'FeatureMatchingDiscriminator',
    
    # Patch
    'NLayerPatchDiscriminator',
    'PatchGANDiscriminator',
    'PixelDiscriminator',
    'DeepPatchDiscriminator',
    'ResidualPatchDiscriminator',
    'DilatedPatchDiscriminator',
    'AttentionPatchDiscriminator',
    
    # Multi-scale
    'MultiScaleDiscriminator',
    'PyramidDiscriminator',
    'SharedEncoderMultiScaleDiscriminator',
    'AdaptiveMultiScaleDiscriminator',
    'ProgressiveMultiScaleDiscriminator',
    'DualScaleDiscriminator',
    
    # Spectral
    'SpectralNormDiscriminator',
    'SNResNetDiscriminator',
    'SNProjectionDiscriminator',
    'SNMultiScaleDiscriminator',
    'SNSelfAttentionDiscriminator',
    'SNUNetDiscriminator',
]
