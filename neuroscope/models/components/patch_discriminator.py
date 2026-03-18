"""patchgan discriminator for cyclegan."""

import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple


class PatchDiscriminator(nn.Module):
    """patchgan discriminator for cyclegan.
    
    70x70 patchgan discriminator architecture that classifies 70x70 overlapping patches
    as real or fake. this creates a fully-convolutional network that can be applied to 
    images of arbitrary size.
    """
    
    def __init__(self, in_channels: int = 4, base_features: int = 64, n_layers: int = 3):
        """initialize patchgan discriminator.
        
        args:
            in_channels: number of input channels.
            base_features: number of features in the first layer.
            n_layers: number of downsampling layers.
        """
        super().__init__()
        
        def discriminator_block(in_filters: int, 
                               out_filters: int, 
                               stride: int = 2, 
                               normalize: bool = True) -> List[nn.Module]:
            """create a discriminator block.
            
            args:
                in_filters: number of input filters.
                out_filters: number of output filters.
                stride: stride for convolution.
                normalize: whether to apply instance normalization.
                
            returns:
                list of layers in the block.
            """
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        # initial layer without normalization
        sequence = discriminator_block(in_channels, base_features, normalize=False)
        
        # downsampling layers
        in_features = base_features
        for i in range(n_layers - 1):
            out_features = min(base_features * (2 ** (i + 1)), 512)
            sequence.extend(discriminator_block(in_features, out_features))
            in_features = out_features
        
        # final layer with stride=1
        sequence.extend(discriminator_block(in_features, out_features * 2, stride=1))
        
        # output layer
        sequence.append(nn.Conv2d(out_features * 2, 1, kernel_size=4, padding=1))
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass of patchgan discriminator.
        
        args:
            x: input tensor of shape [b, c, h, w].
            
        returns:
            output tensor of shape [b, 1, h', w'].
        """
        return self.model(x)
    
    def get_num_params(self) -> int:
        """get number of trainable parameters.
        
        returns:
            number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)