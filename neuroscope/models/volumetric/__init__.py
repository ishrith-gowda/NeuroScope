"""
volumetric (3d) architecture components.

this module provides 3d convolutional architectures for volumetric
medical image processing, enabling full 3d context for brain mri
harmonization with superior inter-slice consistency.
"""

from .blocks_3d import (
    CBAM3D,
    ChannelAttention3D,
    DownsampleBlock3D,
    ResidualBlock3D,
    SelfAttention3D,
    SpatialAttention3D,
    UpsampleBlock3D,
)
from .cyclegan_3d import CycleGAN3D, SACycleGAN3D
from .discriminator_3d import Discriminator3D, MultiScaleDiscriminator3D
from .generator_3d import Generator3D, SAGenerator3D

__all__ = [
    "CBAM3D",
    "ChannelAttention3D",
    "CycleGAN3D",
    "Discriminator3D",
    "DownsampleBlock3D",
    "Generator3D",
    "MultiScaleDiscriminator3D",
    "ResidualBlock3D",
    "SACycleGAN3D",
    "SAGenerator3D",
    "SelfAttention3D",
    "SpatialAttention3D",
    "UpsampleBlock3D",
]
