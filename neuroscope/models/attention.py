"""
self-attention module for medical image translation

this module implements self-attention mechanisms for capturing long-range 
dependencies in brain mri, which is crucial for preserving structural 
coherence across distant brain regions during domain translation.

reference:
- zhang et al., "self-attention generative adversarial networks" (icml 2019)
- adapted for 4-channel medical imaging with domain-specific modifications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SelfAttention(nn.Module):
    """
    self-attention layer for feature maps.
    
    computes attention weights between all spatial positions, allowing
    the model to capture long-range dependencies that are crucial for
    maintaining anatomical consistency in brain mri translation.
    
    args:
        in_channels: number of input channels
        reduction_ratio: channel reduction ratio for query/key (default: 8)
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super().__init__()
        
        self.in_channels = in_channels
        self.reduced_channels = max(in_channels // reduction_ratio, 1)
        
        # query, key, value projections
        self.query = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # learnable scaling parameter (gamma)
        # initialized to 0 so attention is gradually introduced during training
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # softmax for attention weights
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        forward pass with attention.
        
        args:
            x: input tensor of shape (b, c, h, w)
            
        returns:
            out: attended output tensor of shape (b, c, h, w)
            attention: attention weights of shape (b, h*w, h*w)
        """
        batch_size, C, H, W = x.size()
        
        # query: (b, c', h, w) -> (b, c', h*w) -> (b, h*w, c')
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        
        # key: (b, c', h, w) -> (b, c', h*w)
        key = self.key(x).view(batch_size, -1, H * W)
        
        # attention: (b, h*w, c') x (b, c', h*w) -> (b, h*w, h*w)
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)
        
        # value: (b, c, h, w) -> (b, c, h*w)
        value = self.value(x).view(batch_size, -1, H * W)
        
        # attended features: (b, c, h*w) x (b, h*w, h*w) -> (b, c, h*w)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # residual connection with learnable weight
        out = self.gamma * out + x
        
        return out, attention


class ChannelAttention(nn.Module):
    """
    channel attention module (squeeze-and-excitation style).
    
    recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between channels. useful for emphasizing 
    modality-specific features in multi-modal mri.
    
    args:
        in_channels: number of input channels
        reduction_ratio: channel reduction ratio (default: 16)
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        
        reduced_channels = max(in_channels // reduction_ratio, 1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass.
        
        args:
            x: input tensor of shape (b, c, h, w)
            
        returns:
            recalibrated tensor of shape (b, c, h, w)
        """
        batch_size, C, _, _ = x.size()
        
        # global average pooling
        avg_out = self.avg_pool(x).view(batch_size, C)
        avg_out = self.fc(avg_out)
        
        # global max pooling
        max_out = self.max_pool(x).view(batch_size, C)
        max_out = self.fc(max_out)
        
        # combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)
        attention = attention.view(batch_size, C, 1, 1)
        
        return x * attention


class SpatialAttention(nn.Module):
    """
    spatial attention module.
    
    generates a spatial attention map that highlights important regions
    (e.g., tumor areas, ventricles) while suppressing irrelevant background.
    
    args:
        kernel_size: size of the convolution kernel (default: 7)
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass.
        
        args:
            x: input tensor of shape (b, c, h, w)
            
        returns:
            spatially attended tensor of shape (b, c, h, w)
        """
        # channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # concatenate and convolve
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        
        return x * attention


class CBAM(nn.Module):
    """
    convolutional block attention module (cbam).
    
    combines channel and spatial attention for comprehensive feature
    refinement. particularly effective for medical imaging where both
    modality-specific (channel) and region-specific (spatial) features matter.
    
    reference:
    - woo et al., "cbam: convolutional block attention module" (eccv 2018)
    
    args:
        in_channels: number of input channels
        reduction_ratio: channel reduction ratio (default: 16)
        kernel_size: spatial attention kernel size (default: 7)
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass with sequential channel and spatial attention.
        
        args:
            x: input tensor of shape (b, c, h, w)
            
        returns:
            attended tensor of shape (b, c, h, w)
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    multi-head self-attention for feature maps.
    
    extends self-attention with multiple attention heads, allowing the model
    to jointly attend to information from different representation subspaces.
    
    args:
        in_channels: number of input channels
        num_heads: number of attention heads (default: 4)
        reduction_ratio: channel reduction ratio (default: 8)
    """
    
    def __init__(self, in_channels: int, num_heads: int = 4, reduction_ratio: int = 8):
        super().__init__()
        
        self.num_heads = num_heads
        self.reduced_channels = max(in_channels // reduction_ratio, num_heads)
        self.head_dim = self.reduced_channels // num_heads
        
        assert self.reduced_channels % num_heads == 0, \
            f"reduced_channels ({self.reduced_channels}) must be divisible by num_heads ({num_heads})"
        
        self.query = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass with multi-head attention.
        
        args:
            x: input tensor of shape (b, c, h, w)
            
        returns:
            attended tensor of shape (b, c, h, w)
        """
        B, C, H, W = x.size()
        
        # project to q, k, v
        q = self.query(x)  # (b, reduced_c, h, w)
        k = self.key(x)
        v = self.value(x)  # (b, c, h, w)
        
        # reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, H * W)  # (b, heads, head_dim, n)
        k = k.view(B, self.num_heads, self.head_dim, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)
        
        # attention scores
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale  # (b, heads, n, n)
        attn = F.softmax(attn, dim=-1)
        
        # apply attention to values
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)  # (b, heads, c/heads, n)
        out = out.reshape(B, C, H, W)
        
        # output projection and residual
        out = self.out_proj(out)
        out = self.gamma * out + x
        
        return out


class CrossModalAttention(nn.Module):
    """
    cross-modal attention for multi-channel mri.
    
    computes attention between different mri modalities (t1, t1ce, t2, flair)
    to capture complementary information and ensure modality consistency
    during domain translation.
    
    this is a novel contribution specific to multi-modal medical imaging.
    
    args:
        in_channels: number of input channels (typically 4 for t1/t1ce/t2/flair)
        hidden_dim: hidden dimension for attention computation
    """
    
    def __init__(self, in_channels: int = 4, hidden_dim: int = 64):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # per-modality feature extraction
        self.modality_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
                nn.InstanceNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(in_channels)
        ])
        
        # cross-modal attention
        self.query = nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1)
        self.key = nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1)
        self.value = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
        # output projection back to original channels
        self.out_proj = nn.Conv2d(hidden_dim * in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass with cross-modal attention.
        
        args:
            x: input tensor of shape (b, 4, h, w) representing 4 mri modalities
            
        returns:
            cross-modality attended tensor of shape (b, 4, h, w)
        """
        B, C, H, W = x.size()
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        
        # encode each modality separately
        modality_features = []
        for i, encoder in enumerate(self.modality_encoders):
            feat = encoder(x[:, i:i+1, :, :])  # (b, hidden_dim, h, w)
            modality_features.append(feat)
        
        # stack modalities: (b, num_modalities, hidden_dim, h, w)
        stacked = torch.stack(modality_features, dim=1)
        
        # compute cross-modal attention
        attended_features = []
        for i in range(self.in_channels):
            query_feat = self.query(modality_features[i])  # (b, hidden_dim/4, h, w)
            
            # attend to all other modalities
            attended = modality_features[i].clone()
            for j in range(self.in_channels):
                if i != j:
                    key_feat = self.key(modality_features[j])
                    value_feat = self.value(modality_features[j])
                    
                    # simplified attention (could be made more sophisticated)
                    attn = torch.sum(query_feat * key_feat, dim=1, keepdim=True)
                    attn = torch.sigmoid(attn)
                    attended = attended + attn * value_feat
            
            attended_features.append(attended)
        
        # concatenate and project
        out = torch.cat(attended_features, dim=1)  # (b, hidden_dim * 4, h, w)
        out = self.out_proj(out)  # (b, 4, h, w)
        
        # residual connection
        out = self.gamma * out + x
        
        return out
