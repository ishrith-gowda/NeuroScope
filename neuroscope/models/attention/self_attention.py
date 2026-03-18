"""
self-attention mechanism implementations.

this module provides self-attention layers for capturing long-range
dependencies in feature maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SelfAttention2d(nn.Module):
    """
    self-attention layer for 2d feature maps.
    
    implements the self-attention mechanism from "self-attention gan" (zhang et al., 2019).
    captures long-range dependencies by computing attention between all spatial positions.
    
    args:
        in_channels: number of input channels
        reduction: channel reduction ratio for query/key projections
        use_spectral_norm: whether to apply spectral normalization
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
        
        # query, key, value projections
        self.query = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # output projection
        self.out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # learnable scaling parameter (gamma)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # apply spectral normalization
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
        forward pass.
        
        args:
            x: input tensor [b, c, h, w]
            return_attention: whether to return attention maps
            
        returns:
            output tensor and optionally attention maps
        """
        batch_size, C, H, W = x.size()
        
        # compute q, k, v
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # [b, hw, c']
        key = self.key(x).view(batch_size, -1, H * W)  # [b, c', hw]
        value = self.value(x).view(batch_size, -1, H * W)  # [b, c, hw]
        
        # attention: softmax(q @ k^t / sqrt(d)) @ v
        attention = self.softmax(torch.bmm(query, key) / math.sqrt(self.reduced_channels))  # [b, hw, hw]
        
        # apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [b, c, hw]
        out = out.view(batch_size, C, H, W)
        
        # output projection with residual
        out = self.out(out)
        out = self.gamma * out + x
        
        if return_attention:
            return out, attention.view(batch_size, H, W, H, W)
        return out, None


class EfficientSelfAttention2d(nn.Module):
    """
    efficient self-attention with linear complexity.
    
    uses kernel approximation to reduce o(n²) complexity to o(n).
    based on "efficient attention" (shen et al., 2021).
    
    args:
        in_channels: number of input channels
        reduction: channel reduction ratio
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
        forward pass with linear complexity attention.
        
        args:
            x: input tensor [b, c, h, w]
            
        returns:
            output tensor
        """
        batch_size, C, H, W = x.size()
        N = H * W
        
        # compute q, k, v
        q = self.query(x).view(batch_size, self.reduced_channels, N)  # [b, c', n]
        k = self.key(x).view(batch_size, self.reduced_channels, N)    # [b, c', n]
        v = self.value(x).view(batch_size, C, N)                       # [b, c, n]
        
        # apply softmax to q and k separately
        q = F.softmax(q, dim=-1)  # normalize over spatial
        k = F.softmax(k, dim=1)   # normalize over channels
        
        # efficient attention: v @ (k^t @ q)
        # instead of (q @ k^t) @ v which is o(n²)
        context = torch.bmm(k, v.permute(0, 2, 1))  # [b, c', c]
        out = torch.bmm(q.permute(0, 2, 1), context)  # [b, n, c]
        out = out.permute(0, 2, 1).view(batch_size, C, H, W)
        
        return self.gamma * out + x


class MultiScaleSelfAttention(nn.Module):
    """
    multi-scale self-attention for hierarchical feature processing.
    
    applies attention at multiple resolutions and fuses results.
    
    args:
        in_channels: number of input channels
        scales: list of downsampling scales
        reduction: channel reduction ratio
    """
    
    def __init__(
        self,
        in_channels: int,
        scales: Tuple[int, ...] = (1, 2, 4),
        reduction: int = 8
    ):
        super().__init__()
        
        self.scales = scales
        
        # attention at each scale
        self.attention_layers = nn.ModuleList([
            SelfAttention2d(in_channels, reduction, use_spectral_norm=True)
            for _ in scales
        ])
        
        # fusion
        self.fusion = nn.Conv2d(in_channels * len(scales), in_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass with multi-scale attention.
        
        args:
            x: input tensor [b, c, h, w]
            
        returns:
            output tensor
        """
        B, C, H, W = x.size()
        outputs = []
        
        for scale, attn in zip(self.scales, self.attention_layers):
            if scale > 1:
                # downsample
                x_scaled = F.avg_pool2d(x, kernel_size=scale, stride=scale)
            else:
                x_scaled = x
                
            # apply attention
            out, _ = attn(x_scaled)
            
            if scale > 1:
                # upsample back
                out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
                
            outputs.append(out)
            
        # fuse all scales
        fused = torch.cat(outputs, dim=1)
        return self.fusion(fused)


# aliases for compatibility
EfficientSelfAttention = EfficientSelfAttention2d
SelfAttention = SelfAttention2d
