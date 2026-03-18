"""
channel attention mechanisms.

implements various channel attention modules for adaptive feature recalibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ChannelAttention(nn.Module):
    """
    channel attention module (cam).
    
    computes channel-wise attention weights using global pooling and mlps.
    based on cbam (woo et al., 2018).
    
    args:
        in_channels: number of input channels
        reduction: channel reduction ratio for bottleneck
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        
        reduced = max(in_channels // reduction, 8)
        
        # shared mlp
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_channels, bias=False)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        compute channel attention.
        
        args:
            x: input tensor [b, c, h, w]
            
        returns:
            channel attention weights [b, c, 1, 1]
        """
        B, C, _, _ = x.size()
        
        # global average pooling
        avg_out = self.avg_pool(x).view(B, C)
        avg_out = self.mlp(avg_out)
        
        # global max pooling
        max_out = self.max_pool(x).view(B, C)
        max_out = self.mlp(max_out)
        
        # combine and sigmoid
        attention = torch.sigmoid(avg_out + max_out)
        return attention.view(B, C, 1, 1)


class SqueezeExcitation(nn.Module):
    """
    squeeze-and-excitation block (hu et al., 2018).
    
    adaptively recalibrates channel-wise feature responses.
    
    args:
        in_channels: number of input channels
        reduction: reduction ratio for bottleneck
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
        apply se attention and return recalibrated features.
        
        args:
            x: input tensor [b, c, h, w]
            
        returns:
            recalibrated tensor [b, c, h, w]
        """
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale


class EfficientChannelAttention(nn.Module):
    """
    efficient channel attention (eca).
    
    uses 1d convolution instead of mlp for efficiency.
    based on eca-net (wang et al., 2020).
    
    args:
        in_channels: number of input channels
        kernel_size: 1d convolution kernel size (adaptive if none)
    """
    
    def __init__(
        self,
        in_channels: int,
        kernel_size: Optional[int] = None
    ):
        super().__init__()
        
        # adaptive kernel size
        if kernel_size is None:
            # k = |log2(c) / gamma + b / gamma|_odd
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
        apply eca attention.
        
        args:
            x: input tensor [b, c, h, w]
            
        returns:
            recalibrated tensor [b, c, h, w]
        """
        B, C, H, W = x.size()
        
        # global average pooling
        y = self.avg_pool(x)  # [b, c, 1, 1]
        y = y.view(B, 1, C)   # [b, 1, c]
        
        # 1d convolution
        y = self.conv(y)      # [b, 1, c]
        y = torch.sigmoid(y)
        y = y.view(B, C, 1, 1)
        
        return x * y


class GlobalContextBlock(nn.Module):
    """
    global context block (gcnet).
    
    efficient global context modeling using a simplified attention.
    based on gcnet (cao et al., 2019).
    
    args:
        in_channels: number of input channels
        reduction: reduction ratio for transform
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        
        reduced = max(in_channels // reduction, 8)
        
        # context modeling (1x1 conv + softmax)
        self.context = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )
        
        # transform (bottleneck)
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=1),
            nn.LayerNorm([reduced, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_channels, kernel_size=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        apply global context block.
        
        args:
            x: input tensor [b, c, h, w]
            
        returns:
            output tensor [b, c, h, w]
        """
        B, C, H, W = x.size()
        
        # context modeling: weighted sum of all positions
        context = self.context(x)  # [b, 1, h, w]
        context = context.view(B, 1, H * W)
        context = F.softmax(context, dim=-1)  # [b, 1, hw]
        context = context.view(B, 1, H, W)
        
        # aggregate global context
        x_pooled = (x * context).sum(dim=[2, 3], keepdim=True)  # [b, c, 1, 1]
        
        # transform and add residual
        transform = self.transform(x_pooled)
        
        return x + transform
