"""
3D Convolutional Building Blocks.

Advanced 3D blocks for volumetric medical image processing including:
- 3D Residual blocks with various normalization options
- 3D Downsampling/Upsampling blocks
- 3D Self-Attention mechanisms
- 3D Channel and Spatial Attention (CBAM)
- Memory-efficient implementations for large volumes
"""

from typing import Optional, Tuple, List, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SpectralNorm3d(nn.Module):
    """Spectral normalization for 3D convolutions."""
    
    def __init__(self, module: nn.Module, name: str = 'weight', n_power_iterations: int = 1):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        
        if not self._made_params():
            self._make_params()
    
    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        
        height = w.data.shape[0]
        for _ in range(self.n_power_iterations):
            v.data = F.normalize(torch.mv(w.view(height, -1).t(), u.data), dim=0)
            u.data = F.normalize(torch.mv(w.view(height, -1), v.data), dim=0)
        
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))
    
    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False
    
    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = F.normalize(u.data, dim=0)
        v.data = F.normalize(v.data, dim=0)
        w_bar = nn.Parameter(w.data)
        
        del self.module._parameters[self.name]
        
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)
    
    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class GroupNorm3D(nn.GroupNorm):
    """Group Normalization for 3D tensors with automatic group calculation."""
    
    def __init__(self, num_channels: int, num_groups: int = 32, eps: float = 1e-5):
        # Adjust num_groups if channels are too few
        num_groups = min(num_groups, num_channels)
        while num_channels % num_groups != 0:
            num_groups -= 1
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps)


class ResidualBlock3D(nn.Module):
    """
    3D Residual Block with flexible normalization.
    
    Supports:
    - Instance normalization
    - Group normalization  
    - Layer normalization
    - Spectral normalization
    - Gradient checkpointing for memory efficiency
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        norm_type: str = 'instance',
        use_spectral_norm: bool = False,
        use_dropout: bool = False,
        dropout_rate: float = 0.5,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # First conv block
        conv1 = nn.Conv3d(channels, channels, kernel_size, padding=padding, bias=False)
        if use_spectral_norm:
            conv1 = SpectralNorm3d(conv1)
        
        # Second conv block  
        conv2 = nn.Conv3d(channels, channels, kernel_size, padding=padding, bias=False)
        if use_spectral_norm:
            conv2 = SpectralNorm3d(conv2)
        
        # Normalization layers
        norm1 = self._get_norm_layer(norm_type, channels)
        norm2 = self._get_norm_layer(norm_type, channels)
        
        layers = [
            conv1,
            norm1,
            nn.ReLU(inplace=True),
        ]
        
        if use_dropout:
            layers.append(nn.Dropout3d(dropout_rate))
        
        layers.extend([
            conv2,
            norm2,
        ])
        
        self.block = nn.Sequential(*layers)
    
    def _get_norm_layer(self, norm_type: str, channels: int) -> nn.Module:
        if norm_type == 'instance':
            return nn.InstanceNorm3d(channels, affine=True)
        elif norm_type == 'batch':
            return nn.BatchNorm3d(channels)
        elif norm_type == 'group':
            return GroupNorm3D(channels)
        elif norm_type == 'layer':
            return nn.Identity()  # Layer norm handled differently for 3D
        else:
            return nn.Identity()
    
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class DownsampleBlock3D(nn.Module):
    """
    3D Downsampling block with strided convolution.
    
    Reduces spatial dimensions by factor of 2 in all dimensions.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        norm_type: str = 'instance',
        use_spectral_norm: bool = False
    ):
        super().__init__()
        
        conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size, stride, padding, bias=False
        )
        
        if use_spectral_norm:
            conv = SpectralNorm3d(conv)
        
        if norm_type == 'instance':
            norm = nn.InstanceNorm3d(out_channels, affine=True)
        elif norm_type == 'batch':
            norm = nn.BatchNorm3d(out_channels)
        elif norm_type == 'group':
            norm = GroupNorm3D(out_channels)
        else:
            norm = nn.Identity()
        
        self.block = nn.Sequential(
            conv,
            norm,
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock3D(nn.Module):
    """
    3D Upsampling block.
    
    Supports transposed convolution or interpolation + conv.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        norm_type: str = 'instance',
        use_spectral_norm: bool = False,
        mode: str = 'transpose'  # 'transpose' or 'interpolate'
    ):
        super().__init__()
        self.mode = mode
        
        if mode == 'transpose':
            conv = nn.ConvTranspose3d(
                in_channels, out_channels,
                kernel_size, stride, padding, bias=False
            )
        else:
            conv = nn.Conv3d(
                in_channels, out_channels,
                3, 1, 1, bias=False
            )
        
        if use_spectral_norm:
            conv = SpectralNorm3d(conv)
        
        if norm_type == 'instance':
            norm = nn.InstanceNorm3d(out_channels, affine=True)
        elif norm_type == 'batch':
            norm = nn.BatchNorm3d(out_channels)
        elif norm_type == 'group':
            norm = GroupNorm3D(out_channels)
        else:
            norm = nn.Identity()
        
        self.conv = conv
        self.norm = norm
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'interpolate':
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class SelfAttention3D(nn.Module):
    """
    3D Self-Attention Module.
    
    Implements self-attention for volumetric data with:
    - Query, Key, Value projections
    - Multi-head attention support
    - Learnable scaling parameter
    - Memory-efficient chunked computation
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        num_heads: int = 1,
        use_checkpoint: bool = True
    ):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // (reduction * num_heads)
        self.use_checkpoint = use_checkpoint
        
        # Projections
        self.query = nn.Conv3d(channels, channels // reduction, 1, bias=False)
        self.key = nn.Conv3d(channels, channels // reduction, 1, bias=False)
        self.value = nn.Conv3d(channels, channels, 1, bias=False)
        
        # Output projection
        self.output = nn.Conv3d(channels, channels, 1, bias=False)
        
        # Learnable scale parameter (initialized to 0 for stable training)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Scale factor
        self.scale = (channels // reduction) ** -0.5
    
    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        
        # Compute Q, K, V
        q = self.query(x).view(B, -1, D * H * W)  # [B, C', N]
        k = self.key(x).view(B, -1, D * H * W)    # [B, C', N]
        v = self.value(x).view(B, -1, D * H * W)  # [B, C, N]
        
        # Attention weights
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale  # [B, N, N]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(v, attn.transpose(1, 2))  # [B, C, N]
        out = out.view(B, C, D, H, W)
        
        # Output projection
        out = self.output(out)
        
        # Residual with learned scale
        return self.gamma * out + x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._attention, x, use_reentrant=False)
        return self._attention(x)


class ChannelAttention3D(nn.Module):
    """
    3D Channel Attention Module (SE-Net style).
    
    Learns channel-wise attention weights.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class SpatialAttention3D(nn.Module):
    """
    3D Spatial Attention Module.
    
    Learns spatial attention weights across D, H, W dimensions.
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(concat))
        return x * attn


class CBAM3D(nn.Module):
    """
    3D Convolutional Block Attention Module.
    
    Combines channel and spatial attention for volumetric data.
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel_size: int = 7
    ):
        super().__init__()
        
        self.channel_attention = ChannelAttention3D(channels, reduction)
        self.spatial_attention = SpatialAttention3D(spatial_kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class MultiHeadSelfAttention3D(nn.Module):
    """
    Multi-Head 3D Self-Attention.
    
    Provides enhanced attention with multiple heads for
    capturing different types of spatial relationships.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_checkpoint: bool = True
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_checkpoint = use_checkpoint
        
        self.qkv = nn.Conv3d(channels, channels * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv3d(channels, channels, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        N = D * H * W
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, N)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply to values
        out = (attn @ v).transpose(2, 3).reshape(B, C, D, H, W)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return self.gamma * out + x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._attention, x, use_reentrant=False)
        return self._attention(x)


class AxialAttention3D(nn.Module):
    """
    Axial Attention for 3D volumes.
    
    Applies attention along each axis separately for
    memory-efficient processing of large volumes.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        use_checkpoint: bool = True
    ):
        super().__init__()
        
        self.depth_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.height_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.width_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        
        self.norm_d = nn.LayerNorm(channels)
        self.norm_h = nn.LayerNorm(channels)
        self.norm_w = nn.LayerNorm(channels)
        
        self.use_checkpoint = use_checkpoint
    
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        
        # Depth attention
        x_d = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, C)
        x_d = self.norm_d(x_d)
        x_d, _ = self.depth_attn(x_d, x_d, x_d)
        x_d = x_d.reshape(B, H, W, D, C).permute(0, 4, 3, 1, 2)
        x = x + x_d
        
        # Height attention
        x_h = x.permute(0, 2, 4, 3, 1).reshape(B * D * W, H, C)
        x_h = self.norm_h(x_h)
        x_h, _ = self.height_attn(x_h, x_h, x_h)
        x_h = x_h.reshape(B, D, W, H, C).permute(0, 4, 1, 3, 2)
        x = x + x_h
        
        # Width attention
        x_w = x.permute(0, 2, 3, 4, 1).reshape(B * D * H, W, C)
        x_w = self.norm_w(x_w)
        x_w, _ = self.width_attn(x_w, x_w, x_w)
        x_w = x_w.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        x = x + x_w
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding for Vision Transformer style processing.
    
    Divides volume into non-overlapping patches and projects them.
    """
    
    def __init__(
        self,
        volume_size: Tuple[int, int, int] = (64, 256, 256),
        patch_size: Tuple[int, int, int] = (4, 16, 16),
        in_channels: int = 1,
        embed_dim: int = 768
    ):
        super().__init__()
        
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.num_patches = (
            (volume_size[0] // patch_size[0]) *
            (volume_size[1] // patch_size[1]) *
            (volume_size[2] // patch_size[2])
        )
        
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, D', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        x = self.norm(x)
        return x
