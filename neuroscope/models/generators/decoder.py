"""
decoder modules for generator architectures.

this module provides various decoder implementations for image-to-image translation.
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
    standard convolutional decoder.
    
    progressive upsampling with optional skip connections and attention.
    
    args:
        in_channels: input channels (from encoder)
        out_channels: output channels
        base_channels: base channel count
        n_upsample: number of upsampling layers
        norm_type: normalization type
        activation: activation type
        use_attention: whether to use self-attention
        attention_layers: which layers to add attention to
        use_skip: whether to use skip connections
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
        
        # upsampling layers
        self.upsample_layers = nn.ModuleList()
        self.attention_modules = nn.ModuleDict()
        
        current_channels = in_channels
        for i in range(n_upsample):
            out_ch = max(current_channels // 2, base_channels)
            
            # if using skip connections, input channels are doubled
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
            
        # final convolution
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
        forward pass.
        
        args:
            x: input tensor from encoder
            skip_features: optional skip connection features (in reverse order)
            
        returns:
            decoded output
        """
        for i, up in enumerate(self.upsample_layers):
            # concatenate skip features if available
            if self.use_skip and skip_features is not None and i > 0:
                skip_idx = len(skip_features) - i - 1
                if skip_idx >= 0 and skip_idx < len(skip_features):
                    skip = skip_features[skip_idx]
                    # resize skip if necessary
                    if skip.shape[-2:] != x.shape[-2:]:
                        skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
                    x = torch.cat([x, skip], dim=1)
                    
            x = up(x)
            
            if self.use_attention and f'attn_{i}' in self.attention_modules:
                x = self.attention_modules[f'attn_{i}'](x)
                
        return self.final(x)


class ResidualDecoder(nn.Module):
    """
    residual decoder with residual blocks.
    
    uses residual blocks between upsampling stages.
    
    args:
        in_channels: input channels
        out_channels: output channels
        base_channels: base channel count
        n_upsample: number of upsampling layers
        n_residual: residual blocks per scale
        norm_type: normalization type
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
        
        # decoder stages
        self.stages = nn.ModuleList()
        
        current_channels = in_channels
        for i in range(n_upsample):
            out_ch = max(current_channels // 2, base_channels)
            
            stage = nn.ModuleList([
                # upsampling
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(current_channels, out_ch, 3, padding=1),
                    nn.InstanceNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ),
                # residual blocks
                nn.Sequential(*[
                    ResidualBlock(out_ch, norm_type=norm_type)
                    for _ in range(n_residual)
                ])
            ])
            
            self.stages.append(stage)
            current_channels = out_ch
            
        # final output
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
        """forward pass."""
        for i, (upsample, residual) in enumerate(self.stages):
            x = upsample(x)
            
            # optional skip connection
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
    u-net style decoder with skip connections.
    
    designed to work with matching encoder for u-net architecture.
    
    args:
        in_channels: input channels from bottleneck
        out_channels: output channels
        skip_channels: list of skip connection channel counts
        base_channels: base channel count
        norm_type: normalization type
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
        
        # decoder blocks
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        
        current_channels = in_channels
        for i in range(n_stages):
            # upsampling convolution
            out_ch = skip_channels[i]
            self.up_convs.append(
                nn.ConvTranspose2d(current_channels, out_ch, 2, stride=2)
            )
            
            # decoder block (after concatenation with skip)
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
            
        # final convolution
        self.final = nn.Conv2d(current_channels, out_channels, 1)
        
    def forward(
        self,
        x: torch.Tensor,
        skip_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        forward pass.
        
        args:
            x: bottleneck features
            skip_features: list of encoder features for skip connections
                          (in order from deep to shallow)
        """
        # reverse skip features to match decoder order
        skips = skip_features[::-1]
        
        for i, (up_conv, dec_block) in enumerate(zip(self.up_convs, self.dec_blocks)):
            x = up_conv(x)
            
            # handle size mismatch
            skip = skips[i]
            if skip.shape[-2:] != x.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)
            
        return self.final(x)


class AttentionDecoder(nn.Module):
    """
    attention-enhanced decoder.
    
    uses attention gates for skip connections.
    
    args:
        in_channels: input channels
        out_channels: output channels
        skip_channels: skip connection channels
        base_channels: base channel count
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
        
        # decoder stages with attention gates
        self.stages = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        
        current_channels = in_channels
        for i in range(n_stages):
            skip_ch = skip_channels[i]
            out_ch = skip_ch
            
            # attention gate
            self.attention_gates.append(
                AttentionGate(skip_ch, current_channels, skip_ch // 2)
            )
            
            # decoder stage
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
        """forward pass with attention-gated skip connections."""
        skips = skip_features[::-1]
        
        for i, (attn_gate, stage) in enumerate(zip(self.attention_gates, self.stages)):
            skip = skips[i]
            
            # apply attention gate
            attended_skip = attn_gate(skip, x)
            
            # upsample and concatenate
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, attended_skip], dim=1)
            x = stage(x)
            
        return self.final(x)


class AttentionGate(nn.Module):
    """
    attention gate for skip connections.
    
    focuses on relevant spatial regions using gating signal.
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
        apply attention gate.
        
        args:
            skip: skip connection features
            gating: gating signal from decoder
        """
        # resize gating to match skip
        gating_resized = F.interpolate(
            gating, size=skip.shape[-2:],
            mode='bilinear', align_corners=False
        )
        
        # compute attention coefficients
        skip_proj = self.W_skip(skip)
        gate_proj = self.W_gate(gating_resized)
        
        attention = self.relu(skip_proj + gate_proj)
        attention = self.sigmoid(self.psi(attention))
        
        return skip * attention


class ProgressiveDecoder(nn.Module):
    """
    progressive decoder for multi-scale output.
    
    generates outputs at multiple resolutions.
    
    args:
        in_channels: input channels
        out_channels: output channels
        base_channels: base channel count
        n_scales: number of output scales
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
        
        # scale-specific decoders
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
        forward pass.
        
        args:
            x: input features
            skip_features: optional skip connections
            return_all_scales: whether to return outputs at all scales
        """
        outputs = []
        
        for i, (decoder, to_rgb) in enumerate(zip(self.scale_decoders, self.to_rgb)):
            x = decoder(x)
            
            # add skip connection if available
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
    decoder using pixel shuffle for upsampling.
    
    sub-pixel convolution for efficient upsampling.
    
    args:
        in_channels: input channels
        out_channels: output channels
        base_channels: base channel count
        n_upsample: number of upsampling stages
        upscale_factor: factor per upsampling stage
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
        
        # upsampling stages
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
        """forward pass using pixel shuffle upsampling."""
        for i, stage in enumerate(self.stages):
            x = stage(x)
            
            if skip_features is not None:
                skip_idx = len(skip_features) - i - 1
                if 0 <= skip_idx < len(skip_features):
                    skip = skip_features[skip_idx]
                    if skip.shape == x.shape:
                        x = x + skip
                        
        return self.final(x)
