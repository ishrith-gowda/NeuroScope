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

# convolution blocks
from .conv import (
    ConvBlock,
    ConvTransposeBlock,
    DownsampleConvBlock,
    SeparableConvBlock,
    UpsampleConvBlock,
)

# residual blocks
from .residual import (
    BottleneckResidualBlock,
    DenseBlock,
    DenseResidualBlock,
    PreActResidualBlock,
    ResidualBlock,
    SEResidualBlock,
)

# aliases
UpsampleBlock = UpsampleConvBlock
DownsampleBlock = DownsampleConvBlock

# normalization layers
from .normalization import (
    SPADE,
    AdaptiveInstanceNorm2d,
    ConditionalBatchNorm2d,
    GroupNorm2d,
    LayerNorm2d,
)

# aliases for compatibility
AdaptiveInstanceNorm = AdaptiveInstanceNorm2d
# denseblock is now imported from residual.py
PixelShuffleBlock = UpsampleBlock  # alias

__all__ = [
    "SPADE",
    "AdaptiveInstanceNorm",
    # normalization
    "AdaptiveInstanceNorm2d",
    "BottleneckResidualBlock",
    "ConditionalBatchNorm2d",
    # convolution
    "ConvBlock",
    "ConvTransposeBlock",
    "DenseBlock",
    "DenseResidualBlock",
    "DownsampleBlock",
    "GroupNorm2d",
    "LayerNorm2d",
    "PixelShuffleBlock",
    "PreActResidualBlock",
    # residual
    "ResidualBlock",
    "SEResidualBlock",
    "SeparableConvBlock",
    "UpsampleBlock",
]
