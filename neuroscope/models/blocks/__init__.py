"""
neural network building blocks.

this module provides reusable building blocks for constructing
generator and discriminator architectures.

modules:
    - residual: residual block implementations
    - conv: convolution block patterns
    - normalization: normalization layer implementations
"""

import torch.nn as nn

# residual blocks
from .residual import (
    ResidualBlock,
    PreActResidualBlock,
    BottleneckResidualBlock,
    SEResidualBlock,
    DenseResidualBlock,
    DenseBlock,
)

# convolution blocks
from .conv import (
    ConvBlock,
    ConvTransposeBlock,
    UpsampleConvBlock,
    DownsampleConvBlock,
    SeparableConvBlock,
)

# aliases
UpsampleBlock = UpsampleConvBlock
DownsampleBlock = DownsampleConvBlock

# normalization layers
from .normalization import (
    AdaptiveInstanceNorm2d,
    LayerNorm2d,
    GroupNorm2d,
    SPADE,
    ConditionalBatchNorm2d,
)

# aliases for compatibility
AdaptiveInstanceNorm = AdaptiveInstanceNorm2d
# denseblock is now imported from residual.py
PixelShuffleBlock = UpsampleBlock  # alias

__all__ = [
    # residual
    'ResidualBlock',
    'PreActResidualBlock', 
    'BottleneckResidualBlock',
    'SEResidualBlock',
    'DenseResidualBlock',
    'DenseBlock',
    # convolution
    'ConvBlock',
    'ConvTransposeBlock',
    'DownsampleBlock',
    'UpsampleBlock',
    'SeparableConvBlock',
    'PixelShuffleBlock',
    # normalization
    'AdaptiveInstanceNorm2d',
    'AdaptiveInstanceNorm',
    'LayerNorm2d',
    'GroupNorm2d',
    'SPADE',
    'ConditionalBatchNorm2d',
]
