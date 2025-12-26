"""
Residual Block implementations for neural network architectures.

This module provides various residual block implementations used in
generator and discriminator architectures.
"""

import torch
import torch.nn as nn
from typing import Optional, Type


def get_norm_layer(norm_type: str) -> Type[nn.Module]:
    """Get normalization layer class from string."""
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
    Standard Residual Block with two convolutions and skip connection.
    
    Architecture:
        x -> Conv -> Norm -> ReLU -> Conv -> Norm -> + -> ReLU
        |                                           |
        +-------------------------------------------+
    
    Args:
        channels: Number of input/output channels
        norm_layer: Normalization layer class (or None to use norm_type)
        norm_type: String specifying norm type ('instance', 'batch', etc.)
        use_dropout: Whether to use dropout
        dropout_prob: Dropout probability
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
        
        # Get norm layer - prefer explicit class over string
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
        """Forward pass with residual connection."""
        return x + self.block(x)


class PreActResidualBlock(nn.Module):
    """
    Pre-activation Residual Block (He et al., 2016).
    
    Architecture:
        x -> Norm -> ReLU -> Conv -> Norm -> ReLU -> Conv -> +
        |                                                    |
        +----------------------------------------------------+
    
    Args:
        channels: Number of input/output channels
        norm_layer: Normalization layer class
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
        """Forward pass with residual connection."""
        return x + self.block(x)


class BottleneckResidualBlock(nn.Module):
    """
    Bottleneck Residual Block for deeper networks.
    
    Architecture:
        x -> 1x1 Conv -> Norm -> ReLU -> 3x3 Conv -> Norm -> ReLU -> 1x1 Conv -> Norm -> +
        |                                                                                |
        +--------------------------------------------------------------------------------+
    
    Args:
        in_channels: Input channels
        bottleneck_channels: Bottleneck channels (typically in_channels // 4)
        out_channels: Output channels
        norm_layer: Normalization layer class
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
        
        # Skip connection with projection if dimensions change
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels),
            )
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return self.relu(self.skip(x) + self.block(x))


class SEResidualBlock(nn.Module):
    """
    Residual Block with Squeeze-and-Excitation attention.
    
    Args:
        channels: Number of channels
        reduction: SE reduction ratio
        norm_layer: Normalization layer class
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
        
        # Squeeze-and-Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SE attention and residual connection."""
        out = self.block(x)
        out = out * self.se(out)
        return x + out


class DenseResidualBlock(nn.Module):
    """
    Dense Residual Block with growth rate.
    
    Implements a DenseNet-style block with residual connection.
    
    Args:
        channels: Number of input/output channels
        growth_rate: Number of channels added per dense layer
        n_layers: Number of dense layers in the block
        norm_layer: Normalization layer class
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
        
        # Transition layer to match output channels
        final_channels = channels + n_layers * growth_rate
        self.transition = nn.Sequential(
            norm_layer(final_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_channels, channels, kernel_size=1, bias=False),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dense connections and residual."""
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        
        out = torch.cat(features, dim=1)
        out = self.transition(out)
        return x + out


# Alias for compatibility
DenseBlock = DenseResidualBlock
