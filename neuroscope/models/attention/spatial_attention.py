"""
spatial attention mechanisms.

implements various spatial attention modules for adaptive feature selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpatialAttention(nn.Module):
    """
    spatial attention module (sam).
    
    computes spatial attention using channel pooling and convolution.
    based on cbam (woo et al., 2018).
    
    args:
        kernel_size: convolution kernel size (must be odd)
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
        compute spatial attention.
        
        args:
            x: input tensor [b, c, h, w]
            
        returns:
            spatial attention weights [b, 1, h, w]
        """
        # channel pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [b, 1, h, w]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [b, 1, h, w]
        
        # concatenate and convolve
        combined = torch.cat([avg_out, max_out], dim=1)  # [b, 2, h, w]
        attention = self.conv(combined)  # [b, 1, h, w]
        
        return attention


class CBAM(nn.Module):
    """
    convolutional block attention module (cbam).
    
    combines channel and spatial attention sequentially.
    based on woo et al., 2018.
    
    args:
        in_channels: number of input channels
        reduction: channel attention reduction ratio
        spatial_kernel: spatial attention kernel size
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
        apply cbam attention.
        
        args:
            x: input tensor [b, c, h, w]
            
        returns:
            attended tensor [b, c, h, w]
        """
        # channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # spatial attention
        sa = self.spatial_attention(x)
        x = x * sa
        
        return x


class CoordinateAttention(nn.Module):
    """
    coordinate attention.
    
    embeds positional information into channel attention.
    based on hou et al., 2021.
    
    args:
        in_channels: number of input channels
        reduction: reduction ratio
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
        apply coordinate attention.
        
        args:
            x: input tensor [b, c, h, w]
            
        returns:
            attended tensor [b, c, h, w]
        """
        B, C, H, W = x.size()
        
        # pool along each direction
        x_h = self.pool_h(x)  # [b, c, h, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [b, c, w, 1] -> [b, c, 1, w] permuted
        
        # concatenate
        y = torch.cat([x_h, x_w], dim=2)  # [b, c, h+w, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # split
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # generate attention
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        return x * a_h * a_w


class PolarizedSelfAttention(nn.Module):
    """
    polarized self-attention.
    
    combines channel and spatial attention in a parallel manner.
    based on liu et al., 2021.
    
    args:
        in_channels: number of input channels
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.in_channels = in_channels
        
        # channel-only branch
        self.ch_wv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.ch_wq = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.ch_softmax = nn.Softmax(dim=1)
        self.ch_wz = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        self.ln = nn.LayerNorm([in_channels, 1, 1])
        
        # spatial-only branch
        self.sp_wv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.sp_wq = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.sp_softmax = nn.Softmax(dim=-1)
        self.sp_wz = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        apply polarized self-attention.
        
        args:
            x: input tensor [b, c, h, w]
            
        returns:
            attended tensor [b, c, h, w]
        """
        B, C, H, W = x.size()
        
        # channel-only branch
        ch_wv = self.ch_wv(x)  # [b, c//2, h, w]
        ch_wq = self.ch_wq(x)  # [b, 1, h, w]
        ch_wv = ch_wv.view(B, C // 2, -1)  # [b, c//2, hw]
        ch_wq = ch_wq.view(B, -1, 1)  # [b, hw, 1]
        ch_wq = self.ch_softmax(ch_wq)  # softmax over spatial
        
        ch_wz = torch.matmul(ch_wv, ch_wq).unsqueeze(-1)  # [b, c//2, 1, 1]
        ch_out = self.ch_wz(ch_wz)  # [b, c, 1, 1]
        ch_out = self.ln(ch_out).sigmoid()
        
        # spatial-only branch
        sp_wv = self.sp_wv(x).view(B, C // 2, -1)  # [b, c//2, hw]
        sp_wq = self.sp_wq(x).view(B, C // 2, -1)  # [b, c//2, hw]
        
        sp_attn = torch.matmul(sp_wq.permute(0, 2, 1), sp_wv)  # [b, hw, hw]
        sp_attn = self.sp_softmax(sp_attn)
        
        sp_out = torch.matmul(sp_wv, sp_attn.permute(0, 2, 1))  # [b, c//2, hw]
        sp_out = sp_out.view(B, C // 2, H, W)
        sp_out = self.sp_wz(sp_out).sigmoid()
        
        return x * ch_out * sp_out
