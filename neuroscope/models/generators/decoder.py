"""
Decoder Modules for Generator Architectures.

This module provides various decoder implementations for image-to-image translation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from ..blocks.conv import ConvBlock, UpsampleBlock
from ..blocks.residual import ResidualBlock
from ..attention.self_attention import SelfAttention2d
from ..attention.spatial_attention import CBAM


class ConvDecoder(nn.Module):
    """
    Standard Convolutional Decoder.
    
    Progressive upsampling with optional skip connections and attention.
    
    Args:
        in_channels: Input channels (from encoder)
        out_channels: Output channels
        base_channels: Base channel count
        n_upsample: Number of upsampling layers
        norm_type: Normalization type
        activation: Activation type
        use_attention: Whether to use self-attention
        attention_layers: Which layers to add attention to
        use_skip: Whether to use skip connections
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        n_upsample: int = 2,
        norm_type: str = 'instance',
        activation: str = 'relu',
        use_attention: bool = False,
        attention_layers: Optional[List[int]] = None,
        use_skip: bool = False
    ):
        super().__init__()
        
        self.n_upsample = n_upsample
        self.use_attention = use_attention
        self.attention_layers = attention_layers or []
        self.use_skip = use_skip
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        self.attention_modules = nn.ModuleDict()
        
        current_channels = in_channels
        for i in range(n_upsample):
            out_ch = max(current_channels // 2, base_channels)
            
            # If using skip connections, input channels are doubled
            skip_channels = out_ch if use_skip else 0
            
            self.upsample_layers.append(
                UpsampleBlock(
                    current_channels + skip_channels if i > 0 and use_skip else current_channels,
                    out_ch,
                    norm_type=norm_type,
                    activation=activation
                )
            )
            
            if use_attention and i in attention_layers:
                self.attention_modules[f'attn_{i}'] = SelfAttention2d(out_ch)
                
            current_channels = out_ch
            
        # Final convolution
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(current_channels, out_channels, 7),
            nn.Tanh()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        skip_features: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor from encoder
            skip_features: Optional skip connection features (in reverse order)
            
        Returns:
            Decoded output
        """
        for i, up in enumerate(self.upsample_layers):
            # Concatenate skip features if available
            if self.use_skip and skip_features is not None and i > 0:
                skip_idx = len(skip_features) - i - 1
                if skip_idx >= 0 and skip_idx < len(skip_features):
                    skip = skip_features[skip_idx]
                    # Resize skip if necessary
                    if skip.shape[-2:] != x.shape[-2:]:
                        skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
                    x = torch.cat([x, skip], dim=1)
                    
            x = up(x)
            
            if self.use_attention and f'attn_{i}' in self.attention_modules:
                x = self.attention_modules[f'attn_{i}'](x)
                
        return self.final(x)


class ResidualDecoder(nn.Module):
    """
    Residual Decoder with residual blocks.
    
    Uses residual blocks between upsampling stages.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        base_channels: Base channel count
        n_upsample: Number of upsampling layers
        n_residual: Residual blocks per scale
        norm_type: Normalization type
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        n_upsample: int = 3,
        n_residual: int = 2,
        norm_type: str = 'instance'
    ):
        super().__init__()
        
        # Decoder stages
        self.stages = nn.ModuleList()
        
        current_channels = in_channels
        for i in range(n_upsample):
            out_ch = max(current_channels // 2, base_channels)
            
            stage = nn.ModuleList([
                # Upsampling
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(current_channels, out_ch, 3, padding=1),
                    nn.InstanceNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ),
                # Residual blocks
                nn.Sequential(*[
                    ResidualBlock(out_ch, norm_type=norm_type)
                    for _ in range(n_residual)
                ])
            ])
            
            self.stages.append(stage)
            current_channels = out_ch
            
        # Final output
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(current_channels, out_channels, 7),
            nn.Tanh()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        skip_features: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Forward pass."""
        for i, (upsample, residual) in enumerate(self.stages):
            x = upsample(x)
            
            # Optional skip connection
            if skip_features is not None:
                skip_idx = len(skip_features) - i - 1
                if skip_idx >= 0:
                    skip = skip_features[skip_idx]
                    if skip.shape[-2:] == x.shape[-2:]:
                        x = x + skip
                        
            x = residual(x)
            
        return self.final(x)


class UNetDecoder(nn.Module):
    """
    U-Net style Decoder with skip connections.
    
    Designed to work with matching encoder for U-Net architecture.
    
    Args:
        in_channels: Input channels from bottleneck
        out_channels: Output channels
        skip_channels: List of skip connection channel counts
        base_channels: Base channel count
        norm_type: Normalization type
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: List[int],
        base_channels: int = 64,
        norm_type: str = 'batch'
    ):
        super().__init__()
        
        self.skip_channels = skip_channels
        n_stages = len(skip_channels)
        
        # Decoder blocks
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        
        current_channels = in_channels
        for i in range(n_stages):
            # Upsampling convolution
            out_ch = skip_channels[i]
            self.up_convs.append(
                nn.ConvTranspose2d(current_channels, out_ch, 2, stride=2)
            )
            
            # Decoder block (after concatenation with skip)
            self.dec_blocks.append(
                nn.Sequential(
                    nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch) if norm_type == 'batch' else nn.InstanceNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch) if norm_type == 'batch' else nn.InstanceNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
            
            current_channels = out_ch
            
        # Final convolution
        self.final = nn.Conv2d(current_channels, out_channels, 1)
        
    def forward(
        self,
        x: torch.Tensor,
        skip_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Bottleneck features
            skip_features: List of encoder features for skip connections
                          (in order from deep to shallow)
        """
        # Reverse skip features to match decoder order
        skips = skip_features[::-1]
        
        for i, (up_conv, dec_block) in enumerate(zip(self.up_convs, self.dec_blocks)):
            x = up_conv(x)
            
            # Handle size mismatch
            skip = skips[i]
            if skip.shape[-2:] != x.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)
            
        return self.final(x)


class AttentionDecoder(nn.Module):
    """
    Attention-enhanced Decoder.
    
    Uses attention gates for skip connections.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        skip_channels: Skip connection channels
        base_channels: Base channel count
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: List[int],
        base_channels: int = 64
    ):
        super().__init__()
        
        n_stages = len(skip_channels)
        
        # Decoder stages with attention gates
        self.stages = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        
        current_channels = in_channels
        for i in range(n_stages):
            skip_ch = skip_channels[i]
            out_ch = skip_ch
            
            # Attention gate
            self.attention_gates.append(
                AttentionGate(skip_ch, current_channels, skip_ch // 2)
            )
            
            # Decoder stage
            self.stages.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(current_channels + skip_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
            
            current_channels = out_ch
            
        self.final = nn.Conv2d(current_channels, out_channels, 1)
        
    def forward(
        self,
        x: torch.Tensor,
        skip_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass with attention-gated skip connections."""
        skips = skip_features[::-1]
        
        for i, (attn_gate, stage) in enumerate(zip(self.attention_gates, self.stages)):
            skip = skips[i]
            
            # Apply attention gate
            attended_skip = attn_gate(skip, x)
            
            # Upsample and concatenate
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, attended_skip], dim=1)
            x = stage(x)
            
        return self.final(x)


class AttentionGate(nn.Module):
    """
    Attention Gate for skip connections.
    
    Focuses on relevant spatial regions using gating signal.
    """
    
    def __init__(
        self,
        skip_channels: int,
        gating_channels: int,
        inter_channels: int
    ):
        super().__init__()
        
        self.W_skip = nn.Conv2d(skip_channels, inter_channels, 1)
        self.W_gate = nn.Conv2d(gating_channels, inter_channels, 1)
        self.psi = nn.Conv2d(inter_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(
        self,
        skip: torch.Tensor,
        gating: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention gate.
        
        Args:
            skip: Skip connection features
            gating: Gating signal from decoder
        """
        # Resize gating to match skip
        gating_resized = F.interpolate(
            gating, size=skip.shape[-2:],
            mode='bilinear', align_corners=False
        )
        
        # Compute attention coefficients
        skip_proj = self.W_skip(skip)
        gate_proj = self.W_gate(gating_resized)
        
        attention = self.relu(skip_proj + gate_proj)
        attention = self.sigmoid(self.psi(attention))
        
        return skip * attention


class ProgressiveDecoder(nn.Module):
    """
    Progressive Decoder for multi-scale output.
    
    Generates outputs at multiple resolutions.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        base_channels: Base channel count
        n_scales: Number of output scales
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        n_scales: int = 3
    ):
        super().__init__()
        
        self.n_scales = n_scales
        
        # Scale-specific decoders
        self.scale_decoders = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        
        current_channels = in_channels
        for i in range(n_scales):
            out_ch = max(current_channels // 2, base_channels)
            
            self.scale_decoders.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(current_channels, out_ch, 3, padding=1),
                    nn.InstanceNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    CBAM(out_ch)
                )
            )
            
            self.to_rgb.append(
                nn.Conv2d(out_ch, out_channels, 1)
            )
            
            current_channels = out_ch
            
    def forward(
        self,
        x: torch.Tensor,
        skip_features: Optional[List[torch.Tensor]] = None,
        return_all_scales: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features
            skip_features: Optional skip connections
            return_all_scales: Whether to return outputs at all scales
        """
        outputs = []
        
        for i, (decoder, to_rgb) in enumerate(zip(self.scale_decoders, self.to_rgb)):
            x = decoder(x)
            
            # Add skip connection if available
            if skip_features is not None:
                skip_idx = len(skip_features) - i - 1
                if 0 <= skip_idx < len(skip_features):
                    skip = skip_features[skip_idx]
                    if skip.shape[-2:] == x.shape[-2:] and skip.shape[1] == x.shape[1]:
                        x = x + skip
                        
            outputs.append(torch.tanh(to_rgb(x)))
            
        if return_all_scales:
            return outputs
        else:
            return outputs[-1]


class PixelShuffleDecoder(nn.Module):
    """
    Decoder using Pixel Shuffle for upsampling.
    
    Sub-pixel convolution for efficient upsampling.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        base_channels: Base channel count
        n_upsample: Number of upsampling stages
        upscale_factor: Factor per upsampling stage
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        n_upsample: int = 2,
        upscale_factor: int = 2
    ):
        super().__init__()
        
        self.upscale_factor = upscale_factor
        
        # Upsampling stages
        self.stages = nn.ModuleList()
        
        current_channels = in_channels
        for i in range(n_upsample):
            out_ch = max(current_channels // 2, base_channels)
            
            self.stages.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, out_ch * (upscale_factor ** 2), 3, padding=1),
                    nn.PixelShuffle(upscale_factor),
                    nn.InstanceNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
            
            current_channels = out_ch
            
        self.final = nn.Sequential(
            nn.Conv2d(current_channels, out_channels, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        skip_features: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Forward pass using pixel shuffle upsampling."""
        for i, stage in enumerate(self.stages):
            x = stage(x)
            
            if skip_features is not None:
                skip_idx = len(skip_features) - i - 1
                if 0 <= skip_idx < len(skip_features):
                    skip = skip_features[skip_idx]
                    if skip.shape == x.shape:
                        x = x + skip
                        
        return self.final(x)
