"""
resnet-based generator architecture.

classic generator architecture using residual blocks.
"""

import torch
import torch.nn as nn
from typing import Optional, List

from ..blocks.residual import ResidualBlock
from ..attention.self_attention import SelfAttention2d
from .base import BaseGenerator


class ResNetGenerator(BaseGenerator):
    """
    resnet-style generator for image-to-image translation.
    
    architecture: conv -> downsample -> resblocks -> upsample -> conv
    
    args:
        in_channels: input channels
        out_channels: output channels
        ngf: base number of filters
        n_residual: number of residual blocks
        n_downsampling: number of downsampling layers
        norm_type: normalization type ('instance', 'batch')
        use_dropout: whether to use dropout in residual blocks
        padding_type: padding type ('reflect', 'replicate', 'zero')
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        ngf: int = 64,
        n_residual: int = 9,
        n_downsampling: int = 2,
        norm_type: str = 'instance',
        use_dropout: bool = False,
        padding_type: str = 'reflect'
    ):
        super().__init__(in_channels, out_channels, ngf)
        
        self.n_residual = n_residual
        self.n_downsampling = n_downsampling
        
        # normalization layer
        if norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d
        elif norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
            
        # padding layer
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        elif padding_type == 'replicate':
            padding_layer = nn.ReplicationPad2d
        else:
            padding_layer = nn.ZeroPad2d
            
        # initial convolution
        self.initial = nn.Sequential(
            padding_layer(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(inplace=True)
        )
        
        # downsampling
        self.downsample = nn.ModuleList()
        mult = 1
        for i in range(n_downsampling):
            self.downsample.append(
                nn.Sequential(
                    nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(inplace=True)
                )
            )
            mult *= 2
            
        # residual blocks
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(
                ngf * mult,
                norm_type=norm_type,
                use_dropout=use_dropout,
                padding_type=padding_type
            )
            for _ in range(n_residual)
        ])
        
        # upsampling
        self.upsample = nn.ModuleList()
        for i in range(n_downsampling):
            self.upsample.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        ngf * mult, ngf * mult // 2,
                        3, stride=2, padding=1, output_padding=1
                    ),
                    norm_layer(ngf * mult // 2),
                    nn.ReLU(inplace=True)
                )
            )
            mult //= 2
            
        # final convolution
        self.final = nn.Sequential(
            padding_layer(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass."""
        # initial
        out = self.initial(x)
        
        # downsample
        for down in self.downsample:
            out = down(out)
            
        # residual blocks
        out = self.residual_blocks(out)
        
        # upsample
        for up in self.upsample:
            out = up(out)
            
        # final
        out = self.final(out)
        
        return out


class ResNetGeneratorWithAttention(ResNetGenerator):
    """
    resnet generator with self-attention.
    
    adds self-attention layers after residual blocks.
    
    args:
        same as resnetgenerator, plus:
        attention_layers: list of layer indices to add attention
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        ngf: int = 64,
        n_residual: int = 9,
        n_downsampling: int = 2,
        norm_type: str = 'instance',
        use_dropout: bool = False,
        padding_type: str = 'reflect',
        attention_layers: Optional[List[int]] = None
    ):
        super().__init__(
            in_channels, out_channels, ngf, n_residual,
            n_downsampling, norm_type, use_dropout, padding_type
        )
        
        # default: add attention after middle residual blocks
        if attention_layers is None:
            attention_layers = [n_residual // 2]
            
        self.attention_layers = attention_layers
        
        # compute channels at bottleneck
        mult = 2 ** n_downsampling
        bottleneck_channels = ngf * mult
        
        # create attention modules
        self.attention_modules = nn.ModuleDict()
        for idx in attention_layers:
            self.attention_modules[str(idx)] = SelfAttention2d(bottleneck_channels)
            
        # rebuild residual blocks as list for attention insertion
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                bottleneck_channels,
                norm_type=norm_type,
                use_dropout=use_dropout,
                padding_type=padding_type
            )
            for _ in range(n_residual)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass with attention."""
        # initial
        out = self.initial(x)
        
        # downsample
        for down in self.downsample:
            out = down(out)
            
        # residual blocks with attention
        for i, block in enumerate(self.residual_blocks):
            out = block(out)
            if str(i) in self.attention_modules:
                out = self.attention_modules[str(i)](out)
                
        # upsample
        for up in self.upsample:
            out = up(out)
            
        # final
        out = self.final(out)
        
        return out


class FastResNetGenerator(BaseGenerator):
    """
    fast resnet generator with fewer parameters.
    
    optimized for speed while maintaining quality.
    
    args:
        in_channels: input channels
        out_channels: output channels
        ngf: base number of filters (smaller default)
        n_residual: number of residual blocks (fewer)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        ngf: int = 32,
        n_residual: int = 6
    ):
        super().__init__(in_channels, out_channels, ngf)
        
        # initial convolution - smaller kernel
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels, ngf, kernel_size=5, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        # single downsample
        self.downsample = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        )
        
        # lightweight residual blocks
        self.residual_blocks = nn.Sequential(*[
            LightweightResidualBlock(ngf * 2)
            for _ in range(n_residual)
        ])
        
        # single upsample
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        # final convolution
        self.final = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(ngf, out_channels, kernel_size=5, padding=0),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass."""
        out = self.initial(x)
        out = self.downsample(out)
        out = self.residual_blocks(out)
        out = self.upsample(out)
        out = self.final(out)
        return out


class LightweightResidualBlock(nn.Module):
    """lightweight residual block with depthwise separable convolutions."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            # depthwise
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            # pointwise
            nn.Conv2d(channels, channels, 1),
            nn.InstanceNorm2d(channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class DeepResNetGenerator(BaseGenerator):
    """
    deep resnet generator with more capacity.
    
    for high-quality generation with more parameters.
    
    args:
        in_channels: input channels
        out_channels: output channels
        ngf: base number of filters
        n_residual: number of residual blocks (more)
        n_downsampling: number of downsampling layers
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        ngf: int = 64,
        n_residual: int = 12,
        n_downsampling: int = 3
    ):
        super().__init__(in_channels, out_channels, ngf)
        
        # initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        # progressive downsampling
        self.downsample = nn.ModuleList()
        mult = 1
        for i in range(n_downsampling):
            out_mult = min(mult * 2, 8)  # cap at 8x base
            self.downsample.append(
                nn.Sequential(
                    nn.Conv2d(ngf * mult, ngf * out_mult, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(ngf * out_mult),
                    nn.ReLU(inplace=True)
                )
            )
            mult = out_mult
            
        # deep residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(n_residual):
            self.residual_blocks.append(
                PreActResidualBlock(ngf * mult)
            )
            
        # progressive upsampling
        self.upsample = nn.ModuleList()
        for i in range(n_downsampling):
            out_mult = max(mult // 2, 1)
            self.upsample.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        ngf * mult, ngf * out_mult,
                        3, stride=2, padding=1, output_padding=1
                    ),
                    nn.InstanceNorm2d(ngf * out_mult),
                    nn.ReLU(inplace=True)
                )
            )
            mult = out_mult
            
        # final convolution
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass."""
        out = self.initial(x)
        
        # store for potential skip connections
        skip_features = [out]
        
        for down in self.downsample:
            out = down(out)
            skip_features.append(out)
            
        for block in self.residual_blocks:
            out = block(out)
            
        for i, up in enumerate(self.upsample):
            out = up(out)
            
        out = self.final(out)
        return out


class PreActResidualBlock(nn.Module):
    """pre-activation residual block (he et al.)."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, padding=0)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)
