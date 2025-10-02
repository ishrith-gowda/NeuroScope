"""ResNet-based generator for CycleGAN."""

import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple


class ResidualBlock(nn.Module):
    """Residual block with reflection padding for CycleGAN generator."""
    
    def __init__(self, dim: int):
        """Initialize residual block.
        
        Args:
            dim: Number of input and output channels.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of residual block.
        
        Args:
            x: Input tensor of shape [B, C, H, W].
            
        Returns:
            Output tensor of shape [B, C, H, W].
        """
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """ResNet-based generator for CycleGAN.
    
    Architecture:
    - Initial reflection padding and 7x7 convolution
    - 2 downsampling layers with stride 2
    - 9 residual blocks
    - 2 upsampling layers with stride 2
    - Final reflection padding and 7x7 convolution
    """
    
    def __init__(self, in_channels: int = 4, out_channels: int = 4, n_residual: int = 9):
        """Initialize ResNet generator.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            n_residual: Number of residual blocks.
        """
        super().__init__()
        
        # Initial block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling blocks
        in_feat, out_feat = 64, 128
        for _ in range(2):
            model += [
                nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_feat),
                nn.ReLU(inplace=True)
            ]
            in_feat, out_feat = out_feat, out_feat * 2
        
        # Residual blocks
        for _ in range(n_residual):
            model += [ResidualBlock(in_feat)]
        
        # Upsampling blocks
        out_feat = in_feat // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_feat),
                nn.ReLU(inplace=True)
            ]
            in_feat, out_feat = out_feat, out_feat // 2
        
        # Output block
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, kernel_size=7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of ResNet generator.
        
        Args:
            x: Input tensor of shape [B, C, H, W].
            
        Returns:
            Output tensor of shape [B, C, H, W].
        """
        return self.model(x)
    
    def get_num_params(self) -> int:
        """Get number of trainable parameters.
        
        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def weights_init_normal(m: nn.Module):
    """Initialize weights with normal distribution.
    
    Args:
        m: Module to initialize.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)