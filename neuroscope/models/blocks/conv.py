"""
convolution block implementations.

this module provides various convolution block patterns used throughout
the architecture.
"""

import torch
import torch.nn as nn
from typing import Optional, Type, Tuple, Union


class ConvBlock(nn.Module):
    """
    standard convolution block: conv -> norm -> activation.
    
    args:
        in_channels: input channels
        out_channels: output channels
        kernel_size: convolution kernel size
        stride: convolution stride
        padding: convolution padding
        norm_layer: normalization layer class
        activation: activation function
        bias: whether to use bias
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_layer: Optional[Type[nn.Module]] = nn.InstanceNorm2d,
        activation: Optional[nn.Module] = None,
        bias: bool = False
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            )
        ]
        
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
            
        if activation is not None:
            layers.append(activation)
        else:
            layers.append(nn.ReLU(inplace=True))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvTransposeBlock(nn.Module):
    """
    transposed convolution block for upsampling.
    
    args:
        in_channels: input channels
        out_channels: output channels
        kernel_size: convolution kernel size
        stride: convolution stride
        padding: convolution padding
        output_padding: output padding for transposed conv
        norm_layer: normalization layer class
        activation: activation function
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 0,
        norm_layer: Optional[Type[nn.Module]] = nn.InstanceNorm2d,
        activation: Optional[nn.Module] = None
    ):
        super().__init__()
        
        layers = [
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=False
            )
        ]
        
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
            
        if activation is not None:
            layers.append(activation)
        else:
            layers.append(nn.ReLU(inplace=True))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleConvBlock(nn.Module):
    """
    upsample + conv block (avoids checkerboard artifacts from transposed conv).
    
    args:
        in_channels: input channels
        out_channels: output channels
        scale_factor: upsampling scale factor
        mode: upsampling mode ('nearest', 'bilinear')
        norm_layer: normalization layer class
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        mode: str = 'bilinear',
        norm_layer: Optional[Type[nn.Module]] = nn.InstanceNorm2d
    ):
        super().__init__()
        
        self.upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode=mode,
            align_corners=True if mode == 'bilinear' else None
        )
        
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=False),
        ]
        
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
            
        layers.append(nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.conv(x)


class DownsampleConvBlock(nn.Module):
    """
    strided convolution block for downsampling.
    
    args:
        in_channels: input channels
        out_channels: output channels
        norm_layer: normalization layer class
        use_spectral_norm: whether to apply spectral normalization
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Optional[Type[nn.Module]] = nn.InstanceNorm2d,
        use_spectral_norm: bool = False
    ):
        super().__init__()
        
        conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=4, stride=2, padding=1, bias=False
        )
        
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)
            
        layers = [conv]
        
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
            
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SeparableConvBlock(nn.Module):
    """
    depthwise separable convolution block for efficiency.
    
    args:
        in_channels: input channels
        out_channels: output channels
        kernel_size: convolution kernel size
        stride: convolution stride
        padding: convolution padding
        norm_layer: normalization layer class
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_layer: Optional[Type[nn.Module]] = nn.InstanceNorm2d
    ):
        super().__init__()
        
        # depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        
        # pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False
        )
        
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.activation(x)


# aliases for backward compatibility
DownsampleBlock = DownsampleConvBlock
UpsampleBlock = UpsampleConvBlock
PixelShuffleBlock = UpsampleConvBlock
