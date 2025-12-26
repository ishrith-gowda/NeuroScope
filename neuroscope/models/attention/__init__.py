"""
Attention Mechanisms for Neural Networks.

This package provides various attention mechanisms for capturing
long-range dependencies and adaptive feature recalibration.

Modules:
    self_attention: Self-attention for spatial dependencies
    channel_attention: Channel-wise attention (SE, ECA)
    spatial_attention: Spatial attention (CBAM-style)
    multi_head: Multi-head attention variants
"""

from .self_attention import (
    SelfAttention2d,
    EfficientSelfAttention2d,
    MultiScaleSelfAttention,
    EfficientSelfAttention,
    SelfAttention,
)

from .channel_attention import (
    ChannelAttention,
    SqueezeExcitation,
    EfficientChannelAttention,
    GlobalContextBlock,
)

from .spatial_attention import (
    SpatialAttention,
    CBAM,
    CoordinateAttention,
    PolarizedSelfAttention,
)

from .multi_head import (
    MultiHeadSelfAttention2d,
    CrossAttention2d,
    WindowedMultiHeadAttention,
)

__all__ = [
    # Self-attention
    "SelfAttention2d",
    "SelfAttention",
    "EfficientSelfAttention2d",
    "EfficientSelfAttention",
    "MultiScaleSelfAttention",
    # Channel attention
    "ChannelAttention",
    "SqueezeExcitation",
    "EfficientChannelAttention",
    "GlobalContextBlock",
    # Spatial attention
    "SpatialAttention",
    "CBAM",
    "CoordinateAttention",
    "PolarizedSelfAttention",
    # Multi-head
    "MultiHeadSelfAttention2d",
    "CrossAttention2d",
    "WindowedMultiHeadAttention",
]
