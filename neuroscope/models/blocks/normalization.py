"""
Normalization layer implementations.

This module provides various normalization layers used in GAN architectures.
"""

import torch
import torch.nn as nn
from typing import Optional


class AdaptiveInstanceNorm2d(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) for style transfer.
    
    Aligns the mean and variance of content features with style features.
    
    Args:
        num_features: Number of feature channels
        eps: Small constant for numerical stability
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
        Apply adaptive instance normalization.
        
        Args:
            x: Input tensor [B, C, H, W]
            gamma: Scale parameters [B, C] or [B, C, 1, 1]
            beta: Shift parameters [B, C] or [B, C, 1, 1]
            
        Returns:
            Normalized tensor
        """
        # Compute instance statistics
        b, c, h, w = x.size()
        x_reshaped = x.view(b, c, -1)
        mean = x_reshaped.mean(dim=2, keepdim=True)
        std = x_reshaped.std(dim=2, keepdim=True) + self.eps
        
        # Normalize
        x_norm = (x_reshaped - mean) / std
        x_norm = x_norm.view(b, c, h, w)
        
        # Apply style
        if gamma.dim() == 2:
            gamma = gamma.view(b, c, 1, 1)
            beta = beta.view(b, c, 1, 1)
            
        return gamma * x_norm + beta


class LayerNorm2d(nn.Module):
    """
    Layer Normalization for 2D feature maps.
    
    Normalizes over C, H, W dimensions (used in Transformers).
    
    Args:
        num_features: Number of feature channels
        eps: Small constant for numerical stability
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization."""
        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        std = x.std(dim=[1, 2, 3], keepdim=True) + self.eps
        
        x_norm = (x - mean) / std
        
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        
        return gamma * x_norm + beta


class GroupNorm2d(nn.Module):
    """
    Group Normalization wrapper with configurable groups.
    
    Args:
        num_features: Number of feature channels
        num_groups: Number of groups (default: 32 or num_features if smaller)
        eps: Small constant for numerical stability
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
    Spatially-Adaptive Normalization (SPADE) for semantic image synthesis.
    
    Modulates normalized activations based on semantic layout.
    
    Args:
        norm_channels: Number of channels to normalize
        label_channels: Number of label/condition channels
        hidden_channels: Hidden layer channels in modulation network
    """
    
    def __init__(
        self,
        norm_channels: int,
        label_channels: int,
        hidden_channels: int = 128
    ):
        super().__init__()
        
        self.norm = nn.InstanceNorm2d(norm_channels, affine=False)
        
        # Modulation network
        self.shared = nn.Sequential(
            nn.Conv2d(label_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.gamma = nn.Conv2d(hidden_channels, norm_channels, kernel_size=3, padding=1)
        self.beta = nn.Conv2d(hidden_channels, norm_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        """
        Apply SPADE normalization.
        
        Args:
            x: Input features [B, C, H, W]
            segmap: Semantic layout [B, label_channels, H', W']
            
        Returns:
            Modulated features
        """
        # Normalize
        x_norm = self.norm(x)
        
        # Resize segmap to match feature size
        segmap = nn.functional.interpolate(
            segmap, size=x.shape[2:], mode='nearest'
        )
        
        # Compute modulation parameters
        shared_out = self.shared(segmap)
        gamma = self.gamma(shared_out)
        beta = self.beta(shared_out)
        
        return x_norm * (1 + gamma) + beta


class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization for class-conditional generation.
    
    Args:
        num_features: Number of feature channels
        num_classes: Number of conditioning classes
    """
    
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        
        # Class-conditional parameters
        self.embed_gamma = nn.Embedding(num_classes, num_features)
        self.embed_beta = nn.Embedding(num_classes, num_features)
        
        # Initialize
        self.embed_gamma.weight.data.fill_(1.0)
        self.embed_beta.weight.data.zero_()
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Apply conditional batch normalization.
        
        Args:
            x: Input features [B, C, H, W]
            y: Class labels [B] (LongTensor)
            
        Returns:
            Normalized features
        """
        out = self.bn(x)
        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        return gamma * out + beta
