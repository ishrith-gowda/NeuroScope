"""
Spatial Attention mechanisms.

Implements various spatial attention modules for adaptive feature selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM).
    
    Computes spatial attention using channel pooling and convolution.
    Based on CBAM (Woo et al., 2018).
    
    Args:
        kernel_size: Convolution kernel size (must be odd)
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial attention.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Spatial attention weights [B, 1, H, W]
        """
        # Channel pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Concatenate and convolve
        combined = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        attention = self.conv(combined)  # [B, 1, H, W]
        
        return attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Combines channel and spatial attention sequentially.
    Based on Woo et al., 2018.
    
    Args:
        in_channels: Number of input channels
        reduction: Channel attention reduction ratio
        spatial_kernel: Spatial attention kernel size
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7
    ):
        super().__init__()
        
        from .channel_attention import ChannelAttention
        
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CBAM attention.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Attended tensor [B, C, H, W]
        """
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        sa = self.spatial_attention(x)
        x = x * sa
        
        return x


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention.
    
    Embeds positional information into channel attention.
    Based on Hou et al., 2021.
    
    Args:
        in_channels: Number of input channels
        reduction: Reduction ratio
    """
    
    def __init__(self, in_channels: int, reduction: int = 32):
        super().__init__()
        
        reduced = max(in_channels // reduction, 8)
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv1 = nn.Conv2d(in_channels, reduced, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(reduced, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(reduced, in_channels, kernel_size=1, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply coordinate attention.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Attended tensor [B, C, H, W]
        """
        B, C, H, W = x.size()
        
        # Pool along each direction
        x_h = self.pool_h(x)  # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, W, 1] -> [B, C, 1, W] permuted
        
        # Concatenate
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # Generate attention
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        return x * a_h * a_w


class PolarizedSelfAttention(nn.Module):
    """
    Polarized Self-Attention.
    
    Combines channel and spatial attention in a parallel manner.
    Based on Liu et al., 2021.
    
    Args:
        in_channels: Number of input channels
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.in_channels = in_channels
        
        # Channel-only branch
        self.ch_wv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.ch_wq = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.ch_softmax = nn.Softmax(dim=1)
        self.ch_wz = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        self.ln = nn.LayerNorm([in_channels, 1, 1])
        
        # Spatial-only branch
        self.sp_wv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.sp_wq = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.sp_softmax = nn.Softmax(dim=-1)
        self.sp_wz = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply polarized self-attention.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Attended tensor [B, C, H, W]
        """
        B, C, H, W = x.size()
        
        # Channel-only branch
        ch_wv = self.ch_wv(x)  # [B, C//2, H, W]
        ch_wq = self.ch_wq(x)  # [B, 1, H, W]
        ch_wv = ch_wv.view(B, C // 2, -1)  # [B, C//2, HW]
        ch_wq = ch_wq.view(B, -1, 1)  # [B, HW, 1]
        ch_wq = self.ch_softmax(ch_wq)  # softmax over spatial
        
        ch_wz = torch.matmul(ch_wv, ch_wq).unsqueeze(-1)  # [B, C//2, 1, 1]
        ch_out = self.ch_wz(ch_wz)  # [B, C, 1, 1]
        ch_out = self.ln(ch_out).sigmoid()
        
        # Spatial-only branch
        sp_wv = self.sp_wv(x).view(B, C // 2, -1)  # [B, C//2, HW]
        sp_wq = self.sp_wq(x).view(B, C // 2, -1)  # [B, C//2, HW]
        
        sp_attn = torch.matmul(sp_wq.permute(0, 2, 1), sp_wv)  # [B, HW, HW]
        sp_attn = self.sp_softmax(sp_attn)
        
        sp_out = torch.matmul(sp_wv, sp_attn.permute(0, 2, 1))  # [B, C//2, HW]
        sp_out = sp_out.view(B, C // 2, H, W)
        sp_out = self.sp_wz(sp_out).sigmoid()
        
        return x * ch_out * sp_out
