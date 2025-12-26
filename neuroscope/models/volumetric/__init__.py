"""
Volumetric (3D) Architecture Components.

This module provides 3D convolutional architectures for volumetric
medical image processing, enabling full 3D context for brain MRI
harmonization with superior inter-slice consistency.
"""

from .generator_3d import Generator3D, SAGenerator3D
from .discriminator_3d import Discriminator3D, MultiScaleDiscriminator3D
from .blocks_3d import (
    ResidualBlock3D,
    DownsampleBlock3D,
    UpsampleBlock3D,
    SelfAttention3D,
    ChannelAttention3D,
    SpatialAttention3D,
    CBAM3D
)
from .cyclegan_3d import CycleGAN3D, SACycleGAN3D

__all__ = [
    'Generator3D',
    'SAGenerator3D',
    'Discriminator3D',
    'MultiScaleDiscriminator3D',
    'ResidualBlock3D',
    'DownsampleBlock3D',
    'UpsampleBlock3D',
    'SelfAttention3D',
    'ChannelAttention3D',
    'SpatialAttention3D',
    'CBAM3D',
    'CycleGAN3D',
    'SACycleGAN3D',
]
