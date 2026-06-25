"""
generator architectures module.

this module provides various generator architectures for
image-to-image translation, particularly optimized for
medical imaging and mri domain adaptation.
"""

# base classes
from .base import (
    BaseGenerator,
    ConditionalGenerator,
    EncoderDecoderGenerator,
    MultiScaleGenerator,
    ResidualGenerator,
    StyleGenerator,
)

# decoder modules
from .decoder import (
    AttentionDecoder,
    ConvDecoder,
    PixelShuffleDecoder,
    ProgressiveDecoder,
    ResidualDecoder,
    UNetDecoder,
)

# encoder modules
from .encoder import (
    ConvEncoder,
    DenseEncoder,
    HierarchicalEncoder,
    MultiModalEncoder,
    ResidualEncoder,
)

# resnet-based generators
from .resnet import (
    DeepResNetGenerator,
    FastResNetGenerator,
    ResNetGenerator,
    ResNetGeneratorWithAttention,
)

# self-attention generators
from .sa_generator import (
    DenseSAGenerator,
    MultiScaleSAGenerator,
    SABottleneck,
    SADecoder,
    SAEncoder,
    SAGenerator,
)

# u-net generators
from .unet import (
    AttentionUNetGenerator,
    ResUNetGenerator,
    UNetGenerator,
    UNetPlusPlusGenerator,
)

# aliases for compatibility
CycleGANGenerator = ResNetGenerator

__all__ = [
    "AttentionDecoder",
    "AttentionUNetGenerator",
    # base
    "BaseGenerator",
    "ConditionalGenerator",
    # decoders
    "ConvDecoder",
    # encoders
    "ConvEncoder",
    "CycleGANGenerator",
    "DeepResNetGenerator",
    "DenseEncoder",
    "DenseSAGenerator",
    "EncoderDecoderGenerator",
    "FastResNetGenerator",
    "HierarchicalEncoder",
    "MultiModalEncoder",
    "MultiScaleGenerator",
    "MultiScaleSAGenerator",
    "PixelShuffleDecoder",
    "ProgressiveDecoder",
    # resnet
    "ResNetGenerator",
    "ResNetGeneratorWithAttention",
    "ResUNetGenerator",
    "ResidualDecoder",
    "ResidualEncoder",
    "ResidualGenerator",
    "SABottleneck",
    "SADecoder",
    "SAEncoder",
    # self-attention
    "SAGenerator",
    "StyleGenerator",
    "UNetDecoder",
    # u-net
    "UNetGenerator",
    "UNetPlusPlusGenerator",
]
