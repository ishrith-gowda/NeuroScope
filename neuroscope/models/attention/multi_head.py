"""
multi-head attention mechanisms.

implements multi-head attention variants for transformer-style processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadSelfAttention2d(nn.Module):
    """
    multi-head self-attention for 2d feature maps.
    
    extends self-attention with multiple attention heads for diverse
    representation learning.
    
    args:
        in_channels: number of input channels
        num_heads: number of attention heads
        head_dim: dimension per head (computed if none)
        dropout: dropout probability
        use_bias: whether to use bias in projections
    """
    
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_bias: bool = False
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = head_dim or (in_channels // num_heads)
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        
        # q, k, v projections
        self.qkv = nn.Conv2d(in_channels, self.inner_dim * 3, kernel_size=1, bias=use_bias)
        
        # output projection
        self.proj = nn.Conv2d(self.inner_dim, in_channels, kernel_size=1, bias=use_bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # learnable scaling
        self.gamma = nn.Parameter(torch.zeros(1))
        
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
        B, C, H, W = x.size()
        N = H * W
        
        # compute q, k, v
        qkv = self.qkv(x)  # [b, 3*inner_dim, h, w]
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, N)  # [b, 3, heads, head_dim, n]
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, b, heads, n, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [b, heads, n, head_dim]
        
        # attention: softmax(q @ k^t / sqrt(d)) @ v
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [b, heads, n, n]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # apply attention
        out = torch.matmul(attn, v)  # [b, heads, n, head_dim]
        out = out.transpose(2, 3).contiguous()  # [b, heads, head_dim, n]
        out = out.view(B, self.inner_dim, H, W)  # [b, inner_dim, h, w]
        
        # output projection with residual
        out = self.proj(out)
        out = self.gamma * out + x
        
        if return_attention:
            return out, attn
        return out, None


class CrossAttention2d(nn.Module):
    """
    cross-attention for feature alignment between different modalities.
    
    args:
        query_channels: query tensor channels
        key_channels: key/value tensor channels
        num_heads: number of attention heads
        dropout: dropout probability
    """
    
    def __init__(
        self,
        query_channels: int,
        key_channels: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = query_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Conv2d(query_channels, query_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(key_channels, query_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(key_channels, query_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(query_channels, query_channels, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        cross-attention forward pass.
        
        args:
            query: query tensor [b, c_q, h_q, w_q]
            context: key/value tensor [b, c_k, h_k, w_k]
            
        returns:
            attended query tensor [b, c_q, h_q, w_q]
        """
        B, C, H_q, W_q = query.size()
        _, _, H_k, W_k = context.size()
        N_q = H_q * W_q
        N_k = H_k * W_k
        
        # projections
        q = self.q_proj(query).view(B, self.num_heads, self.head_dim, N_q)
        k = self.k_proj(context).view(B, self.num_heads, self.head_dim, N_k)
        v = self.v_proj(context).view(B, self.num_heads, self.head_dim, N_k)
        
        # transpose for attention
        q = q.transpose(2, 3)  # [b, heads, n_q, head_dim]
        k = k.transpose(2, 3)  # [b, heads, n_k, head_dim]
        v = v.transpose(2, 3)  # [b, heads, n_k, head_dim]
        
        # attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [b, heads, n_q, n_k]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)  # [b, heads, n_q, head_dim]
        out = out.transpose(2, 3).contiguous()  # [b, heads, head_dim, n_q]
        out = out.view(B, C, H_q, W_q)
        
        return self.out_proj(out) + query


class WindowedMultiHeadAttention(nn.Module):
    """
    windowed multi-head attention (swin transformer style).
    
    computes attention within local windows for efficiency.
    
    args:
        in_channels: number of input channels
        num_heads: number of attention heads
        window_size: window size for local attention
    """
    
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        window_size: int = 8
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.proj = nn.Linear(in_channels, in_channels)
        
        # relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # create relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = coords.view(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass with windowed attention.
        
        args:
            x: input tensor [b, c, h, w]
            
        returns:
            output tensor [b, c, h, w]
        """
        B, C, H, W = x.size()
        ws = self.window_size
        
        # pad if necessary
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.size()
        
        # reshape to windows
        x = x.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # [b, nh, nw, ws, ws, c]
        nH, nW = Hp // ws, Wp // ws
        x = x.view(B * nH * nW, ws * ws, C)  # [b*nh*nw, ws*ws, c]
        
        # qkv
        qkv = self.qkv(x).reshape(-1, ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, b*nw*nh, heads, ws*ws, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # attention with relative position bias
        attn = (q @ k.transpose(-2, -1)) * self.scale
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(ws * ws, ws * ws, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(-1, ws * ws, C)
        out = self.proj(out)
        
        # reshape back
        out = out.view(B, nH, nW, ws, ws, C)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()  # [b, c, nh, ws, nw, ws]
        out = out.view(B, C, Hp, Wp)
        
        # remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W].contiguous()
            
        return out
