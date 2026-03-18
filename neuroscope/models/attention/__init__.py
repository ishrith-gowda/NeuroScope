"""
attention mechanisms for neural networks.

this package provides various attention mechanisms for capturing
long-range dependencies and adaptive feature recalibration.

modules:
    self_attention: self-attention for spatial dependencies
    channel_attention: channel-wise attention (se, eca)
    spatial_attention: spatial attention (cbam-style)
    multi_head: multi-head attention variants
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
    # self-attention
    "SelfAttention2d",
    "SelfAttention",
    "EfficientSelfAttention2d",
    "EfficientSelfAttention",
    "MultiScaleSelfAttention",
    # channel attention
    "ChannelAttention",
    "SqueezeExcitation",
    "EfficientChannelAttention",
    "GlobalContextBlock",
    # spatial attention
    "SpatialAttention",
    "CBAM",
    "CoordinateAttention",
    "PolarizedSelfAttention",
    # multi-head
    "MultiHeadSelfAttention2d",
    "CrossAttention2d",
    "WindowedMultiHeadAttention",
]
