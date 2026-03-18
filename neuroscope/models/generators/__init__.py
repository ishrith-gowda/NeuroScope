"""
generator architectures module.

this module provides various generator architectures for
image-to-image translation, particularly optimized for
medical imaging and mri domain adaptation.
"""

# base classes
from .base import (
    BaseGenerator,
    EncoderDecoderGenerator,
    ResidualGenerator,
    MultiScaleGenerator,
    ConditionalGenerator,
    StyleGenerator,
)

# encoder modules
from .encoder import (
    ConvEncoder,
    ResidualEncoder,
    DenseEncoder,
    MultiModalEncoder,
    HierarchicalEncoder,
)

# decoder modules
from .decoder import (
    ConvDecoder,
    ResidualDecoder,
    UNetDecoder,
    AttentionDecoder,
    ProgressiveDecoder,
    PixelShuffleDecoder,
)

# resnet-based generators
from .resnet import (
    ResNetGenerator,
    ResNetGeneratorWithAttention,
    FastResNetGenerator,
    DeepResNetGenerator,
)

# self-attention generators
from .sa_generator import (
    SAGenerator,
    SAEncoder,
    SABottleneck,
    SADecoder,
    MultiScaleSAGenerator,
    DenseSAGenerator,
)

# u-net generators
from .unet import (
    UNetGenerator,
    AttentionUNetGenerator,
    ResUNetGenerator,
    UNetPlusPlusGenerator,
)

# aliases for compatibility
CycleGANGenerator = ResNetGenerator

__all__ = [
    # base
    'BaseGenerator',
    'EncoderDecoderGenerator',
    'ResidualGenerator',
    'MultiScaleGenerator',
    'ConditionalGenerator',
    'StyleGenerator',
    
    # encoders
    'ConvEncoder',
    'ResidualEncoder',
    'DenseEncoder',
    'MultiModalEncoder',
    'HierarchicalEncoder',
    
    # decoders
    'ConvDecoder',
    'ResidualDecoder',
    'UNetDecoder',
    'AttentionDecoder',
    'ProgressiveDecoder',
    'PixelShuffleDecoder',
    
    # resnet
    'ResNetGenerator',
    'ResNetGeneratorWithAttention',
    'FastResNetGenerator',
    'DeepResNetGenerator',
    'CycleGANGenerator',
    
    # self-attention
    'SAGenerator',
    'SAEncoder',
    'SABottleneck',
    'SADecoder',
    'MultiScaleSAGenerator',
    'DenseSAGenerator',
    
    # u-net
    'UNetGenerator',
    'AttentionUNetGenerator',
    'ResUNetGenerator',
    'UNetPlusPlusGenerator',
]
