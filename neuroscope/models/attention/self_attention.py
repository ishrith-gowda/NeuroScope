"""
Self-Attention mechanism implementations.

This module provides self-attention layers for capturing long-range
dependencies in feature maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SelfAttention2d(nn.Module):
    """
    Self-Attention layer for 2D feature maps.
    
    Implements the self-attention mechanism from "Self-Attention GAN" (Zhang et al., 2019).
    Captures long-range dependencies by computing attention between all spatial positions.
    
    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio for query/key projections
        use_spectral_norm: Whether to apply spectral normalization
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction: int = 8,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.reduced_channels = max(in_channels // reduction, 1)
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Output projection
        self.out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Learnable scaling parameter (gamma)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Apply spectral normalization
        if use_spectral_norm:
            self.query = nn.utils.spectral_norm(self.query)
            self.key = nn.utils.spectral_norm(self.key)
            self.value = nn.utils.spectral_norm(self.value)
            self.out = nn.utils.spectral_norm(self.out)
            
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            return_attention: Whether to return attention maps
            
        Returns:
            Output tensor and optionally attention maps
        """
        batch_size, C, H, W = x.size()
        
        # Compute Q, K, V
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # [B, HW, C']
        key = self.key(x).view(batch_size, -1, H * W)  # [B, C', HW]
        value = self.value(x).view(batch_size, -1, H * W)  # [B, C, HW]
        
        # Attention: softmax(Q @ K^T / sqrt(d)) @ V
        attention = self.softmax(torch.bmm(query, key) / math.sqrt(self.reduced_channels))  # [B, HW, HW]
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(batch_size, C, H, W)
        
        # Output projection with residual
        out = self.out(out)
        out = self.gamma * out + x
        
        if return_attention:
            return out, attention.view(batch_size, H, W, H, W)
        return out, None


class EfficientSelfAttention2d(nn.Module):
    """
    Efficient Self-Attention with linear complexity.
    
    Uses kernel approximation to reduce O(n²) complexity to O(n).
    Based on "Efficient Attention" (Shen et al., 2021).
    
    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio
    """
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        
        self.reduced_channels = max(in_channels // reduction, 1)
        
        self.query = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with linear complexity attention.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor
        """
        batch_size, C, H, W = x.size()
        N = H * W
        
        # Compute Q, K, V
        q = self.query(x).view(batch_size, self.reduced_channels, N)  # [B, C', N]
        k = self.key(x).view(batch_size, self.reduced_channels, N)    # [B, C', N]
        v = self.value(x).view(batch_size, C, N)                       # [B, C, N]
        
        # Apply softmax to Q and K separately
        q = F.softmax(q, dim=-1)  # Normalize over spatial
        k = F.softmax(k, dim=1)   # Normalize over channels
        
        # Efficient attention: V @ (K^T @ Q)
        # Instead of (Q @ K^T) @ V which is O(n²)
        context = torch.bmm(k, v.permute(0, 2, 1))  # [B, C', C]
        out = torch.bmm(q.permute(0, 2, 1), context)  # [B, N, C]
        out = out.permute(0, 2, 1).view(batch_size, C, H, W)
        
        return self.gamma * out + x


class MultiScaleSelfAttention(nn.Module):
    """
    Multi-scale self-attention for hierarchical feature processing.
    
    Applies attention at multiple resolutions and fuses results.
    
    Args:
        in_channels: Number of input channels
        scales: List of downsampling scales
        reduction: Channel reduction ratio
    """
    
    def __init__(
        self,
        in_channels: int,
        scales: Tuple[int, ...] = (1, 2, 4),
        reduction: int = 8
    ):
        super().__init__()
        
        self.scales = scales
        
        # Attention at each scale
        self.attention_layers = nn.ModuleList([
            SelfAttention2d(in_channels, reduction, use_spectral_norm=True)
            for _ in scales
        ])
        
        # Fusion
        self.fusion = nn.Conv2d(in_channels * len(scales), in_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale attention.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor
        """
        B, C, H, W = x.size()
        outputs = []
        
        for scale, attn in zip(self.scales, self.attention_layers):
            if scale > 1:
                # Downsample
                x_scaled = F.avg_pool2d(x, kernel_size=scale, stride=scale)
            else:
                x_scaled = x
                
            # Apply attention
            out, _ = attn(x_scaled)
            
            if scale > 1:
                # Upsample back
                out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
                
            outputs.append(out)
            
        # Fuse all scales
        fused = torch.cat(outputs, dim=1)
        return self.fusion(fused)


# Aliases for compatibility
EfficientSelfAttention = EfficientSelfAttention2d
SelfAttention = SelfAttention2d
