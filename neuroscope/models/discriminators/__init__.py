"""
discriminator architectures module.

this module provides various discriminator architectures for
adversarial training in image-to-image translation.
"""

# base classes
from .base import (
    BaseDiscriminator,
    ConditionalDiscriminator,
    FeatureMatchingDiscriminator,
    MultiScaleDiscriminatorBase,
    PatchDiscriminator,
    ProjectionDiscriminator,
)

# multi-scale discriminators
from .multiscale import (
    AdaptiveMultiScaleDiscriminator,
    DualScaleDiscriminator,
    MultiScaleDiscriminator,
    ProgressiveMultiScaleDiscriminator,
    PyramidDiscriminator,
    SharedEncoderMultiScaleDiscriminator,
)

# patch discriminators
from .patch import (
    AttentionPatchDiscriminator,
    DeepPatchDiscriminator,
    DilatedPatchDiscriminator,
    NLayerPatchDiscriminator,
    PixelDiscriminator,
    ResidualPatchDiscriminator,
)

# spectral normalization discriminators
from .spectral import (
    SNMultiScaleDiscriminator,
    SNProjectionDiscriminator,
    SNResNetDiscriminator,
    SNSelfAttentionDiscriminator,
    SNUNetDiscriminator,
    SpectralNormDiscriminator,
)

# aliases for compatibility
PatchGANDiscriminator = NLayerPatchDiscriminator

__all__ = [
    "AdaptiveMultiScaleDiscriminator",
    "AttentionPatchDiscriminator",
    # base
    "BaseDiscriminator",
    "ConditionalDiscriminator",
    "DeepPatchDiscriminator",
    "DilatedPatchDiscriminator",
    "DualScaleDiscriminator",
    "FeatureMatchingDiscriminator",
    # multi-scale
    "MultiScaleDiscriminator",
    "MultiScaleDiscriminatorBase",
    # patch
    "NLayerPatchDiscriminator",
    "PatchDiscriminator",
    "PatchGANDiscriminator",
    "PixelDiscriminator",
    "ProgressiveMultiScaleDiscriminator",
    "ProjectionDiscriminator",
    "PyramidDiscriminator",
    "ResidualPatchDiscriminator",
    "SNMultiScaleDiscriminator",
    "SNProjectionDiscriminator",
    "SNResNetDiscriminator",
    "SNSelfAttentionDiscriminator",
    "SNUNetDiscriminator",
    "SharedEncoderMultiScaleDiscriminator",
    # spectral
    "SpectralNormDiscriminator",
]
