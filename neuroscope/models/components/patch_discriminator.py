"""PatchGAN discriminator for CycleGAN."""

import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for CycleGAN.
    
    70x70 PatchGAN discriminator architecture that classifies 70x70 overlapping patches
    as real or fake. This creates a fully-convolutional network that can be applied to 
    images of arbitrary size.
    """
    
    def __init__(self, in_channels: int = 4, base_features: int = 64, n_layers: int = 3):
        """Initialize PatchGAN discriminator.
        
        Args:
            in_channels: Number of input channels.
            base_features: Number of features in the first layer.
            n_layers: Number of downsampling layers.
        """
        super().__init__()
        
        def discriminator_block(in_filters: int, 
                               out_filters: int, 
                               stride: int = 2, 
                               normalize: bool = True) -> List[nn.Module]:
            """Create a discriminator block.
            
            Args:
                in_filters: Number of input filters.
                out_filters: Number of output filters.
                stride: Stride for convolution.
                normalize: Whether to apply instance normalization.
                
            Returns:
                List of layers in the block.
            """
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        # Initial layer without normalization
        sequence = discriminator_block(in_channels, base_features, normalize=False)
        
        # Downsampling layers
        in_features = base_features
        for i in range(n_layers - 1):
            out_features = min(base_features * (2 ** (i + 1)), 512)
            sequence.extend(discriminator_block(in_features, out_features))
            in_features = out_features
        
        # Final layer with stride=1
        sequence.extend(discriminator_block(in_features, out_features * 2, stride=1))
        
        # Output layer
        sequence.append(nn.Conv2d(out_features * 2, 1, kernel_size=4, padding=1))
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of PatchGAN discriminator.
        
        Args:
            x: Input tensor of shape [B, C, H, W].
            
        Returns:
            Output tensor of shape [B, 1, H', W'].
        """
        return self.model(x)
    
    def get_num_params(self) -> int:
        """Get number of trainable parameters.
        
        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)