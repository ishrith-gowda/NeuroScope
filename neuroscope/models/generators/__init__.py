"""
Generator Architectures Module.

This module provides various generator architectures for
image-to-image translation, particularly optimized for
medical imaging and MRI domain adaptation.
"""

# Base classes
from .base import (
    BaseGenerator,
    EncoderDecoderGenerator,
    ResidualGenerator,
    MultiScaleGenerator,
    ConditionalGenerator,
    StyleGenerator,
)

# Encoder modules
from .encoder import (
    ConvEncoder,
    ResidualEncoder,
    DenseEncoder,
    MultiModalEncoder,
    HierarchicalEncoder,
)

# Decoder modules
from .decoder import (
    ConvDecoder,
    ResidualDecoder,
    UNetDecoder,
    AttentionDecoder,
    ProgressiveDecoder,
    PixelShuffleDecoder,
)

# ResNet-based generators
from .resnet import (
    ResNetGenerator,
    ResNetGeneratorWithAttention,
    FastResNetGenerator,
    DeepResNetGenerator,
)

# Self-Attention generators
from .sa_generator import (
    SAGenerator,
    SAEncoder,
    SABottleneck,
    SADecoder,
    MultiScaleSAGenerator,
    DenseSAGenerator,
)

# U-Net generators
from .unet import (
    UNetGenerator,
    AttentionUNetGenerator,
    ResUNetGenerator,
    UNetPlusPlusGenerator,
)

# Aliases for compatibility
CycleGANGenerator = ResNetGenerator

__all__ = [
    # Base
    'BaseGenerator',
    'EncoderDecoderGenerator',
    'ResidualGenerator',
    'MultiScaleGenerator',
    'ConditionalGenerator',
    'StyleGenerator',
    
    # Encoders
    'ConvEncoder',
    'ResidualEncoder',
    'DenseEncoder',
    'MultiModalEncoder',
    'HierarchicalEncoder',
    
    # Decoders
    'ConvDecoder',
    'ResidualDecoder',
    'UNetDecoder',
    'AttentionDecoder',
    'ProgressiveDecoder',
    'PixelShuffleDecoder',
    
    # ResNet
    'ResNetGenerator',
    'ResNetGeneratorWithAttention',
    'FastResNetGenerator',
    'DeepResNetGenerator',
    'CycleGANGenerator',
    
    # Self-Attention
    'SAGenerator',
    'SAEncoder',
    'SABottleneck',
    'SADecoder',
    'MultiScaleSAGenerator',
    'DenseSAGenerator',
    
    # U-Net
    'UNetGenerator',
    'AttentionUNetGenerator',
    'ResUNetGenerator',
    'UNetPlusPlusGenerator',
]
