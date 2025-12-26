"""
Neural Network Building Blocks.

This module provides reusable building blocks for constructing
generator and discriminator architectures.

Modules:
    - residual: Residual block implementations
    - conv: Convolution block patterns
    - normalization: Normalization layer implementations
"""

import torch.nn as nn

# Residual blocks
from .residual import (
    ResidualBlock,
    PreActResidualBlock,
    BottleneckResidualBlock,
    SEResidualBlock,
    DenseResidualBlock,
    DenseBlock,
)

# Convolution blocks
from .conv import (
    ConvBlock,
    ConvTransposeBlock,
    UpsampleConvBlock,
    DownsampleConvBlock,
    SeparableConvBlock,
)

# Aliases
UpsampleBlock = UpsampleConvBlock
DownsampleBlock = DownsampleConvBlock

# Normalization layers
from .normalization import (
    AdaptiveInstanceNorm2d,
    LayerNorm2d,
    GroupNorm2d,
    SPADE,
    ConditionalBatchNorm2d,
)

# Aliases for compatibility
AdaptiveInstanceNorm = AdaptiveInstanceNorm2d
# DenseBlock is now imported from residual.py
PixelShuffleBlock = UpsampleBlock  # Alias

__all__ = [
    # Residual
    'ResidualBlock',
    'PreActResidualBlock', 
    'BottleneckResidualBlock',
    'SEResidualBlock',
    'DenseResidualBlock',
    'DenseBlock',
    # Convolution
    'ConvBlock',
    'ConvTransposeBlock',
    'DownsampleBlock',
    'UpsampleBlock',
    'SeparableConvBlock',
    'PixelShuffleBlock',
    # Normalization
    'AdaptiveInstanceNorm2d',
    'AdaptiveInstanceNorm',
    'LayerNorm2d',
    'GroupNorm2d',
    'SPADE',
    'ConditionalBatchNorm2d',
]
