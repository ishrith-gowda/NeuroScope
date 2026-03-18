"""resnet-based generator for cyclegan."""

import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple


class ResidualBlock(nn.Module):
    """residual block with reflection padding for cyclegan generator."""
    
    def __init__(self, dim: int):
        """initialize residual block.
        
        args:
            dim: number of input and output channels.
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
        """forward pass of residual block.
        
        args:
            x: input tensor of shape [b, c, h, w].
            
        returns:
            output tensor of shape [b, c, h, w].
        """
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """resnet-based generator for cyclegan.
    
    architecture:
    - initial reflection padding and 7x7 convolution
    - 2 downsampling layers with stride 2
    - 9 residual blocks
    - 2 upsampling layers with stride 2
    - final reflection padding and 7x7 convolution
    """
    
    def __init__(self, in_channels: int = 4, out_channels: int = 4, n_residual: int = 9):
        """initialize resnet generator.
        
        args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            n_residual: number of residual blocks.
        """
        super().__init__()
        
        # initial block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # downsampling blocks
        in_feat, out_feat = 64, 128
        for _ in range(2):
            model += [
                nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_feat),
                nn.ReLU(inplace=True)
            ]
            in_feat, out_feat = out_feat, out_feat * 2
        
        # residual blocks
        for _ in range(n_residual):
            model += [ResidualBlock(in_feat)]
        
        # upsampling blocks
        out_feat = in_feat // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_feat),
                nn.ReLU(inplace=True)
            ]
            in_feat, out_feat = out_feat, out_feat // 2
        
        # output block
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, kernel_size=7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass of resnet generator.
        
        args:
            x: input tensor of shape [b, c, h, w].
            
        returns:
            output tensor of shape [b, c, h, w].
        """
        return self.model(x)
    
    def get_num_params(self) -> int:
        """get number of trainable parameters.
        
        returns:
            number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def weights_init_normal(m: nn.Module):
    """initialize weights with normal distribution.
    
    args:
        m: module to initialize.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)