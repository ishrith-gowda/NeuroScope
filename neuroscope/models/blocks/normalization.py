"""
normalization layer implementations.

this module provides various normalization layers used in gan architectures.
"""

import torch
import torch.nn as nn
from typing import Optional


class AdaptiveInstanceNorm2d(nn.Module):
    """
    adaptive instance normalization (adain) for style transfer.
    
    aligns the mean and variance of content features with style features.
    
    args:
        num_features: number of feature channels
        eps: small constant for numerical stability
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor
    ) -> torch.Tensor:
        """
        apply adaptive instance normalization.
        
        args:
            x: input tensor [b, c, h, w]
            gamma: scale parameters [b, c] or [b, c, 1, 1]
            beta: shift parameters [b, c] or [b, c, 1, 1]
            
        returns:
            normalized tensor
        """
        # compute instance statistics
        b, c, h, w = x.size()
        x_reshaped = x.view(b, c, -1)
        mean = x_reshaped.mean(dim=2, keepdim=True)
        std = x_reshaped.std(dim=2, keepdim=True) + self.eps
        
        # normalize
        x_norm = (x_reshaped - mean) / std
        x_norm = x_norm.view(b, c, h, w)
        
        # apply style
        if gamma.dim() == 2:
            gamma = gamma.view(b, c, 1, 1)
            beta = beta.view(b, c, 1, 1)
            
        return gamma * x_norm + beta


class LayerNorm2d(nn.Module):
    """
    layer normalization for 2d feature maps.
    
    normalizes over c, h, w dimensions (used in transformers).
    
    args:
        num_features: number of feature channels
        eps: small constant for numerical stability
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """apply layer normalization."""
        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        std = x.std(dim=[1, 2, 3], keepdim=True) + self.eps
        
        x_norm = (x - mean) / std
        
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        
        return gamma * x_norm + beta


class GroupNorm2d(nn.Module):
    """
    group normalization wrapper with configurable groups.
    
    args:
        num_features: number of feature channels
        num_groups: number of groups (default: 32 or num_features if smaller)
        eps: small constant for numerical stability
    """
    
    def __init__(
        self,
        num_features: int,
        num_groups: int = 32,
        eps: float = 1e-5
    ):
        super().__init__()
        num_groups = min(num_groups, num_features)
        self.norm = nn.GroupNorm(num_groups, num_features, eps=eps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class SPADE(nn.Module):
    """
    spatially-adaptive normalization (spade) for semantic image synthesis.
    
    modulates normalized activations based on semantic layout.
    
    args:
        norm_channels: number of channels to normalize
        label_channels: number of label/condition channels
        hidden_channels: hidden layer channels in modulation network
    """
    
    def __init__(
        self,
        norm_channels: int,
        label_channels: int,
        hidden_channels: int = 128
    ):
        super().__init__()
        
        self.norm = nn.InstanceNorm2d(norm_channels, affine=False)
        
        # modulation network
        self.shared = nn.Sequential(
            nn.Conv2d(label_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.gamma = nn.Conv2d(hidden_channels, norm_channels, kernel_size=3, padding=1)
        self.beta = nn.Conv2d(hidden_channels, norm_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        """
        apply spade normalization.
        
        args:
            x: input features [b, c, h, w]
            segmap: semantic layout [b, label_channels, h', w']
            
        returns:
            modulated features
        """
        # normalize
        x_norm = self.norm(x)
        
        # resize segmap to match feature size
        segmap = nn.functional.interpolate(
            segmap, size=x.shape[2:], mode='nearest'
        )
        
        # compute modulation parameters
        shared_out = self.shared(segmap)
        gamma = self.gamma(shared_out)
        beta = self.beta(shared_out)
        
        return x_norm * (1 + gamma) + beta


class ConditionalBatchNorm2d(nn.Module):
    """
    conditional batch normalization for class-conditional generation.
    
    args:
        num_features: number of feature channels
        num_classes: number of conditioning classes
    """
    
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        
        # class-conditional parameters
        self.embed_gamma = nn.Embedding(num_classes, num_features)
        self.embed_beta = nn.Embedding(num_classes, num_features)
        
        # initialize
        self.embed_gamma.weight.data.fill_(1.0)
        self.embed_beta.weight.data.zero_()
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        apply conditional batch normalization.
        
        args:
            x: input features [b, c, h, w]
            y: class labels [b] (longtensor)
            
        returns:
            normalized features
        """
        out = self.bn(x)
        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        return gamma * out + beta
