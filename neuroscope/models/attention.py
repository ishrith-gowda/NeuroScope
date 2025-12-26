"""
Self-Attention Module for Medical Image Translation

This module implements self-attention mechanisms for capturing long-range 
dependencies in brain MRI, which is crucial for preserving structural 
coherence across distant brain regions during domain translation.

Reference:
- Zhang et al., "Self-Attention Generative Adversarial Networks" (ICML 2019)
- Adapted for 4-channel medical imaging with domain-specific modifications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SelfAttention(nn.Module):
    """
    Self-Attention layer for feature maps.
    
    Computes attention weights between all spatial positions, allowing
    the model to capture long-range dependencies that are crucial for
    maintaining anatomical consistency in brain MRI translation.
    
    Args:
        in_channels: Number of input channels
        reduction_ratio: Channel reduction ratio for query/key (default: 8)
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super().__init__()
        
        self.in_channels = in_channels
        self.reduced_channels = max(in_channels // reduction_ratio, 1)
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Learnable scaling parameter (gamma)
        # Initialized to 0 so attention is gradually introduced during training
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            out: Attended output tensor of shape (B, C, H, W)
            attention: Attention weights of shape (B, H*W, H*W)
        """
        batch_size, C, H, W = x.size()
        
        # Query: (B, C', H, W) -> (B, C', H*W) -> (B, H*W, C')
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        
        # Key: (B, C', H, W) -> (B, C', H*W)
        key = self.key(x).view(batch_size, -1, H * W)
        
        # Attention: (B, H*W, C') x (B, C', H*W) -> (B, H*W, H*W)
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)
        
        # Value: (B, C, H, W) -> (B, C, H*W)
        value = self.value(x).view(batch_size, -1, H * W)
        
        # Attended features: (B, C, H*W) x (B, H*W, H*W) -> (B, C, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        
        return out, attention


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (Squeeze-and-Excitation style).
    
    Recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between channels. Useful for emphasizing 
    modality-specific features in multi-modal MRI.
    
    Args:
        in_channels: Number of input channels
        reduction_ratio: Channel reduction ratio (default: 16)
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
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Recalibrated tensor of shape (B, C, H, W)
        """
        batch_size, C, _, _ = x.size()
        
        # Global average pooling
        avg_out = self.avg_pool(x).view(batch_size, C)
        avg_out = self.fc(avg_out)
        
        # Global max pooling
        max_out = self.max_pool(x).view(batch_size, C)
        max_out = self.fc(max_out)
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)
        attention = attention.view(batch_size, C, 1, 1)
        
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Generates a spatial attention map that highlights important regions
    (e.g., tumor areas, ventricles) while suppressing irrelevant background.
    
    Args:
        kernel_size: Size of the convolution kernel (default: 7)
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Spatially attended tensor of shape (B, C, H, W)
        """
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Combines channel and spatial attention for comprehensive feature
    refinement. Particularly effective for medical imaging where both
    modality-specific (channel) and region-specific (spatial) features matter.
    
    Reference:
    - Woo et al., "CBAM: Convolutional Block Attention Module" (ECCV 2018)
    
    Args:
        in_channels: Number of input channels
        reduction_ratio: Channel reduction ratio (default: 16)
        kernel_size: Spatial attention kernel size (default: 7)
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sequential channel and spatial attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Attended tensor of shape (B, C, H, W)
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention for feature maps.
    
    Extends self-attention with multiple attention heads, allowing the model
    to jointly attend to information from different representation subspaces.
    
    Args:
        in_channels: Number of input channels
        num_heads: Number of attention heads (default: 4)
        reduction_ratio: Channel reduction ratio (default: 8)
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
        Forward pass with multi-head attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Attended tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.size()
        
        # Project to Q, K, V
        q = self.query(x)  # (B, reduced_C, H, W)
        k = self.key(x)
        v = self.value(x)  # (B, C, H, W)
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, H * W)  # (B, heads, head_dim, N)
        k = k.view(B, self.num_heads, self.head_dim, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)
        
        # Attention scores
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale  # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)  # (B, heads, C/heads, N)
        out = out.reshape(B, C, H, W)
        
        # Output projection and residual
        out = self.out_proj(out)
        out = self.gamma * out + x
        
        return out


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention for multi-channel MRI.
    
    Computes attention between different MRI modalities (T1, T1ce, T2, FLAIR)
    to capture complementary information and ensure modality consistency
    during domain translation.
    
    This is a novel contribution specific to multi-modal medical imaging.
    
    Args:
        in_channels: Number of input channels (typically 4 for T1/T1ce/T2/FLAIR)
        hidden_dim: Hidden dimension for attention computation
    """
    
    def __init__(self, in_channels: int = 4, hidden_dim: int = 64):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Per-modality feature extraction
        self.modality_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
                nn.InstanceNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(in_channels)
        ])
        
        # Cross-modal attention
        self.query = nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1)
        self.key = nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1)
        self.value = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
        # Output projection back to original channels
        self.out_proj = nn.Conv2d(hidden_dim * in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with cross-modal attention.
        
        Args:
            x: Input tensor of shape (B, 4, H, W) representing 4 MRI modalities
            
        Returns:
            Cross-modality attended tensor of shape (B, 4, H, W)
        """
        B, C, H, W = x.size()
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        
        # Encode each modality separately
        modality_features = []
        for i, encoder in enumerate(self.modality_encoders):
            feat = encoder(x[:, i:i+1, :, :])  # (B, hidden_dim, H, W)
            modality_features.append(feat)
        
        # Stack modalities: (B, num_modalities, hidden_dim, H, W)
        stacked = torch.stack(modality_features, dim=1)
        
        # Compute cross-modal attention
        attended_features = []
        for i in range(self.in_channels):
            query_feat = self.query(modality_features[i])  # (B, hidden_dim/4, H, W)
            
            # Attend to all other modalities
            attended = modality_features[i].clone()
            for j in range(self.in_channels):
                if i != j:
                    key_feat = self.key(modality_features[j])
                    value_feat = self.value(modality_features[j])
                    
                    # Simplified attention (could be made more sophisticated)
                    attn = torch.sum(query_feat * key_feat, dim=1, keepdim=True)
                    attn = torch.sigmoid(attn)
                    attended = attended + attn * value_feat
            
            attended_features.append(attended)
        
        # Concatenate and project
        out = torch.cat(attended_features, dim=1)  # (B, hidden_dim * 4, H, W)
        out = self.out_proj(out)  # (B, 4, H, W)
        
        # Residual connection
        out = self.gamma * out + x
        
        return out
