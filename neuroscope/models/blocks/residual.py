"""
residual block implementations for neural network architectures.

this module provides various residual block implementations used in
generator and discriminator architectures.
"""

import torch
import torch.nn as nn
from typing import Optional, Type


def get_norm_layer(norm_type: str) -> Type[nn.Module]:
    """get normalization layer class from string."""
    norm_layers = {
        'instance': nn.InstanceNorm2d,
        'batch': nn.BatchNorm2d,
        'layer': nn.LayerNorm,
        'group': lambda c: nn.GroupNorm(8, c),
        'none': nn.Identity,
    }
    return norm_layers.get(norm_type, nn.InstanceNorm2d)


class ResidualBlock(nn.Module):
    """
    standard residual block with two convolutions and skip connection.
    
    architecture:
        x -> conv -> norm -> relu -> conv -> norm -> + -> relu
        |                                           |
        +-------------------------------------------+
    
    args:
        channels: number of input/output channels
        norm_layer: normalization layer class (or none to use norm_type)
        norm_type: string specifying norm type ('instance', 'batch', etc.)
        use_dropout: whether to use dropout
        dropout_prob: dropout probability
    """
    
    def __init__(
        self,
        channels: int,
        norm_layer: Optional[Type[nn.Module]] = None,
        norm_type: str = 'instance',
        use_dropout: bool = False,
        dropout_prob: float = 0.5
    ):
        super().__init__()
        
        # get norm layer - prefer explicit class over string
        if norm_layer is None:
            norm_layer = get_norm_layer(norm_type)
        
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False),
            norm_layer(channels),
            nn.ReLU(inplace=True),
        ]
        
        if use_dropout:
            layers.append(nn.Dropout(dropout_prob))
            
        layers.extend([
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False),
            norm_layer(channels),
        ])
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass with residual connection."""
        return x + self.block(x)


class PreActResidualBlock(nn.Module):
    """
    pre-activation residual block (he et al., 2016).
    
    architecture:
        x -> norm -> relu -> conv -> norm -> relu -> conv -> +
        |                                                    |
        +----------------------------------------------------+
    
    args:
        channels: number of input/output channels
        norm_layer: normalization layer class
    """
    
    def __init__(
        self,
        channels: int,
        norm_layer: Type[nn.Module] = nn.InstanceNorm2d
    ):
        super().__init__()
        
        self.block = nn.Sequential(
            norm_layer(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False),
            norm_layer(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass with residual connection."""
        return x + self.block(x)


class BottleneckResidualBlock(nn.Module):
    """
    bottleneck residual block for deeper networks.
    
    architecture:
        x -> 1x1 conv -> norm -> relu -> 3x3 conv -> norm -> relu -> 1x1 conv -> norm -> +
        |                                                                                |
        +--------------------------------------------------------------------------------+
    
    args:
        in_channels: input channels
        bottleneck_channels: bottleneck channels (typically in_channels // 4)
        out_channels: output channels
        norm_layer: normalization layer class
    """
    
    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        norm_layer: Type[nn.Module] = nn.InstanceNorm2d
    ):
        super().__init__()
        
        bottleneck_channels = bottleneck_channels or in_channels // 4
        out_channels = out_channels or in_channels
        
        self.block = nn.Sequential(
            # 1x1 reduce
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            norm_layer(bottleneck_channels),
            nn.ReLU(inplace=True),
            # 3x3 conv
            nn.ReflectionPad2d(1),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=0, bias=False),
            norm_layer(bottleneck_channels),
            nn.ReLU(inplace=True),
            # 1x1 expand
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
        )
        
        # skip connection with projection if dimensions change
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels),
            )
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass with residual connection."""
        return self.relu(self.skip(x) + self.block(x))


class SEResidualBlock(nn.Module):
    """
    residual block with squeeze-and-excitation attention.
    
    args:
        channels: number of channels
        reduction: se reduction ratio
        norm_layer: normalization layer class
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        norm_layer: Type[nn.Module] = nn.InstanceNorm2d
    ):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False),
            norm_layer(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False),
            norm_layer(channels),
        )
        
        # squeeze-and-excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass with se attention and residual connection."""
        out = self.block(x)
        out = out * self.se(out)
        return x + out


class DenseResidualBlock(nn.Module):
    """
    dense residual block with growth rate.
    
    implements a densenet-style block with residual connection.
    
    args:
        channels: number of input/output channels
        growth_rate: number of channels added per dense layer
        n_layers: number of dense layers in the block
        norm_layer: normalization layer class
    """
    
    def __init__(
        self,
        channels: int,
        growth_rate: int = 32,
        n_layers: int = 4,
        norm_layer: Type[nn.Module] = nn.InstanceNorm2d
    ):
        super().__init__()
        
        self.channels = channels
        self.growth_rate = growth_rate
        
        layers = []
        for i in range(n_layers):
            in_ch = channels + i * growth_rate
            layers.append(nn.Sequential(
                norm_layer(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, growth_rate, kernel_size=3, padding=1, bias=False),
            ))
        self.layers = nn.ModuleList(layers)
        
        # transition layer to match output channels
        final_channels = channels + n_layers * growth_rate
        self.transition = nn.Sequential(
            norm_layer(final_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_channels, channels, kernel_size=1, bias=False),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass with dense connections and residual."""
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        
        out = torch.cat(features, dim=1)
        out = self.transition(out)
        return x + out


# alias for compatibility
DenseBlock = DenseResidualBlock
