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

from .channel_attention import (
    ChannelAttention,
    EfficientChannelAttention,
    GlobalContextBlock,
    SqueezeExcitation,
)
from .multi_head import (
    CrossAttention2d,
    MultiHeadSelfAttention2d,
    WindowedMultiHeadAttention,
)
from .self_attention import (
    EfficientSelfAttention,
    EfficientSelfAttention2d,
    MultiScaleSelfAttention,
    SelfAttention,
    SelfAttention2d,
)
from .spatial_attention import (
    CBAM,
    CoordinateAttention,
    PolarizedSelfAttention,
    SpatialAttention,
)

__all__ = [
    "CBAM",
    # channel attention
    "ChannelAttention",
    "CoordinateAttention",
    "CrossAttention2d",
    "EfficientChannelAttention",
    "EfficientSelfAttention",
    "EfficientSelfAttention2d",
    "GlobalContextBlock",
    # multi-head
    "MultiHeadSelfAttention2d",
    "MultiScaleSelfAttention",
    "PolarizedSelfAttention",
    "SelfAttention",
    # self-attention
    "SelfAttention2d",
    # spatial attention
    "SpatialAttention",
    "SqueezeExcitation",
    "WindowedMultiHeadAttention",
]
