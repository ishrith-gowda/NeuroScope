"""
Channel Attention mechanisms.

Implements various channel attention modules for adaptive feature recalibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM).
    
    Computes channel-wise attention weights using global pooling and MLPs.
    Based on CBAM (Woo et al., 2018).
    
    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio for bottleneck
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        
        reduced = max(in_channels // reduction, 8)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_channels, bias=False)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute channel attention.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Channel attention weights [B, C, 1, 1]
        """
        B, C, _, _ = x.size()
        
        # Global average pooling
        avg_out = self.avg_pool(x).view(B, C)
        avg_out = self.mlp(avg_out)
        
        # Global max pooling
        max_out = self.max_pool(x).view(B, C)
        max_out = self.mlp(max_out)
        
        # Combine and sigmoid
        attention = torch.sigmoid(avg_out + max_out)
        return attention.view(B, C, 1, 1)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al., 2018).
    
    Adaptively recalibrates channel-wise feature responses.
    
    Args:
        in_channels: Number of input channels
        reduction: Reduction ratio for bottleneck
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        
        reduced = max(in_channels // reduction, 8)
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SE attention and return recalibrated features.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Recalibrated tensor [B, C, H, W]
        """
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale


class EfficientChannelAttention(nn.Module):
    """
    Efficient Channel Attention (ECA).
    
    Uses 1D convolution instead of MLP for efficiency.
    Based on ECA-Net (Wang et al., 2020).
    
    Args:
        in_channels: Number of input channels
        kernel_size: 1D convolution kernel size (adaptive if None)
    """
    
    def __init__(
        self,
        in_channels: int,
        kernel_size: Optional[int] = None
    ):
        super().__init__()
        
        # Adaptive kernel size
        if kernel_size is None:
            # k = |log2(C) / gamma + b / gamma|_odd
            t = int(abs((torch.log2(torch.tensor(in_channels)).item() + 1) / 2))
            kernel_size = t if t % 2 else t + 1
            kernel_size = max(3, kernel_size)
            
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.conv = nn.Conv1d(
            1, 1, 
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ECA attention.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Recalibrated tensor [B, C, H, W]
        """
        B, C, H, W = x.size()
        
        # Global average pooling
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.view(B, 1, C)   # [B, 1, C]
        
        # 1D convolution
        y = self.conv(y)      # [B, 1, C]
        y = torch.sigmoid(y)
        y = y.view(B, C, 1, 1)
        
        return x * y


class GlobalContextBlock(nn.Module):
    """
    Global Context Block (GCNet).
    
    Efficient global context modeling using a simplified attention.
    Based on GCNet (Cao et al., 2019).
    
    Args:
        in_channels: Number of input channels
        reduction: Reduction ratio for transform
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        
        reduced = max(in_channels // reduction, 8)
        
        # Context modeling (1x1 conv + softmax)
        self.context = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )
        
        # Transform (bottleneck)
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=1),
            nn.LayerNorm([reduced, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_channels, kernel_size=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply global context block.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.size()
        
        # Context modeling: weighted sum of all positions
        context = self.context(x)  # [B, 1, H, W]
        context = context.view(B, 1, H * W)
        context = F.softmax(context, dim=-1)  # [B, 1, HW]
        context = context.view(B, 1, H, W)
        
        # Aggregate global context
        x_pooled = (x * context).sum(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        
        # Transform and add residual
        transform = self.transform(x_pooled)
        
        return x + transform
