"""
Self-Attention Generator (SA-Generator).

Advanced generator with multi-scale self-attention for
improved long-range dependency modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple

from ..blocks.residual import ResidualBlock, DenseResidualBlock
from ..attention.self_attention import SelfAttention2d, MultiScaleSelfAttention, EfficientSelfAttention
from ..attention.channel_attention import SqueezeExcitation
from ..attention.spatial_attention import CBAM
from .base import BaseGenerator


class SAGenerator(BaseGenerator):
    """
    Self-Attention Generator for image-to-image translation.
    
    Incorporates multi-scale self-attention for capturing long-range
    dependencies, essential for medical image translation.
    
    Architecture:
    - Initial Conv
    - Multi-scale Encoder with Attention
    - Attention-enhanced Residual Bottleneck
    - Multi-scale Decoder with Attention
    - Final Conv
    
    Args:
        in_channels: Input channels (e.g., 4 for multi-modal MRI)
        out_channels: Output channels
        ngf: Base number of generator filters
        n_residual: Number of residual blocks in bottleneck
        n_downsampling: Number of downsampling layers
        attention_type: Type of attention ('self', 'cbam', 'multi_scale')
        use_spectral_norm: Whether to use spectral normalization
        norm_type: Normalization type
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        ngf: int = 64,
        n_residual: int = 9,
        n_downsampling: int = 2,
        attention_type: str = 'multi_scale',
        use_spectral_norm: bool = False,
        norm_type: str = 'instance'
    ):
        super().__init__(in_channels, out_channels, ngf)
        
        self.n_residual = n_residual
        self.n_downsampling = n_downsampling
        self.attention_type = attention_type
        
        # Normalization layer
        if norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d
        elif norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = nn.Identity
            
        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            self._apply_sn(nn.Conv2d(in_channels, ngf, 7), use_spectral_norm),
            norm_layer(ngf),
            nn.ReLU(inplace=True)
        )
        
        # Encoder with attention
        self.encoder = SAEncoder(
            ngf, n_downsampling, norm_layer,
            attention_type, use_spectral_norm
        )
        
        # Bottleneck channels
        bottleneck_channels = ngf * (2 ** n_downsampling)
        
        # Attention-enhanced residual bottleneck
        self.bottleneck = SABottleneck(
            bottleneck_channels, n_residual,
            norm_layer, attention_type, use_spectral_norm
        )
        
        # Decoder with attention
        self.decoder = SADecoder(
            bottleneck_channels, ngf, n_downsampling,
            norm_layer, attention_type, use_spectral_norm
        )
        
        # Final convolution
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            self._apply_sn(nn.Conv2d(ngf, out_channels, 7), use_spectral_norm),
            nn.Tanh()
        )
        
    def _apply_sn(self, module: nn.Module, use_sn: bool) -> nn.Module:
        """Apply spectral normalization if enabled."""
        if use_sn:
            return nn.utils.spectral_norm(module)
        return module
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Initial
        out = self.initial(x)
        
        # Encode
        encoded, skip_features = self.encoder(out)
        
        # Bottleneck
        bottleneck = self.bottleneck(encoded)
        
        # Decode with skip connections
        decoded = self.decoder(bottleneck, skip_features)
        
        # Final
        output = self.final(decoded)
        
        return output
        
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get intermediate feature maps for visualization."""
        features = {}
        
        out = self.initial(x)
        features['initial'] = out
        
        encoded, skip_features = self.encoder(out)
        features['encoded'] = encoded
        for i, skip in enumerate(skip_features):
            features[f'skip_{i}'] = skip
            
        bottleneck = self.bottleneck(encoded)
        features['bottleneck'] = bottleneck
        
        decoded = self.decoder(bottleneck, skip_features)
        features['decoded'] = decoded
        
        output = self.final(decoded)
        features['output'] = output
        
        return features


class SAEncoder(nn.Module):
    """Self-Attention Encoder."""
    
    def __init__(
        self,
        in_channels: int,
        n_downsampling: int,
        norm_layer,
        attention_type: str,
        use_spectral_norm: bool
    ):
        super().__init__()
        
        self.n_downsampling = n_downsampling
        
        self.down_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        mult = 1
        for i in range(n_downsampling):
            in_ch = in_channels * mult
            out_ch = in_channels * mult * 2
            
            # Downsampling block
            down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                norm_layer(out_ch),
                nn.ReLU(inplace=True)
            )
            if use_spectral_norm:
                down[0] = nn.utils.spectral_norm(down[0])
                
            self.down_blocks.append(down)
            
            # Attention block
            if attention_type == 'multi_scale':
                attn = MultiScaleSelfAttention(out_ch)
            elif attention_type == 'cbam':
                attn = CBAM(out_ch)
            else:
                attn = SelfAttention2d(out_ch)
                
            self.attention_blocks.append(attn)
            
            mult *= 2
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass returning encoded features and skip connections."""
        skip_features = []
        
        for down, attn in zip(self.down_blocks, self.attention_blocks):
            x = down(x)
            x = attn(x)
            skip_features.append(x)
            
        return x, skip_features


class SABottleneck(nn.Module):
    """Self-Attention Bottleneck."""
    
    def __init__(
        self,
        channels: int,
        n_residual: int,
        norm_layer,
        attention_type: str,
        use_spectral_norm: bool
    ):
        super().__init__()
        
        # Residual blocks with periodic attention
        self.blocks = nn.ModuleList()
        
        for i in range(n_residual):
            # Residual block
            self.blocks.append(
                ResidualBlock(channels, norm_type='instance')
            )
            
            # Add attention every 3 blocks
            if (i + 1) % 3 == 0:
                if attention_type == 'multi_scale':
                    self.blocks.append(MultiScaleSelfAttention(channels))
                elif attention_type == 'cbam':
                    self.blocks.append(CBAM(channels))
                else:
                    self.blocks.append(SelfAttention2d(channels))
                    
        # Final attention
        self.final_attention = EfficientSelfAttention(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through bottleneck."""
        for block in self.blocks:
            x = block(x)
            
        x = self.final_attention(x)
        
        return x


class SADecoder(nn.Module):
    """Self-Attention Decoder."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_upsampling: int,
        norm_layer,
        attention_type: str,
        use_spectral_norm: bool
    ):
        super().__init__()
        
        self.n_upsampling = n_upsampling
        
        self.up_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for i in range(n_upsampling):
            in_ch = in_channels // (2 ** i)
            out_ch = in_channels // (2 ** (i + 1))
            
            # Upsampling block
            up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
                norm_layer(out_ch),
                nn.ReLU(inplace=True)
            )
            if use_spectral_norm:
                up[0] = nn.utils.spectral_norm(up[0])
                
            self.up_blocks.append(up)
            
            # Skip connection processing - match skip channels to decoder channels
            # Skip from encoder at this level has in_ch channels (same as decoder input at this level)
            self.skip_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1),  # Project from skip channels to output channels
                    norm_layer(out_ch)
                )
            )
            
            # Attention block
            if attention_type == 'multi_scale':
                attn = MultiScaleSelfAttention(out_ch)
            elif attention_type == 'cbam':
                attn = CBAM(out_ch)
            else:
                attn = SelfAttention2d(out_ch)
                
            self.attention_blocks.append(attn)
            
    def forward(
        self,
        x: torch.Tensor,
        skip_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass with skip connections."""
        # Reverse skip features for decoder order
        skips = skip_features[::-1]
        
        for i, (up, skip_conv, attn) in enumerate(
            zip(self.up_blocks, self.skip_convs, self.attention_blocks)
        ):
            x = up(x)
            
            # Add skip connection if available
            if i < len(skips):
                skip = skips[i]
                if skip.shape[-2:] != x.shape[-2:]:
                    skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
                skip = skip_conv(skip)
                x = x + skip
                
            x = attn(x)
            
        return x


class MultiScaleSAGenerator(BaseGenerator):
    """
    Multi-Scale Self-Attention Generator.
    
    Processes input at multiple scales and fuses results.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        ngf: Base number of filters
        n_scales: Number of scales to process
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        ngf: int = 64,
        n_scales: int = 3
    ):
        super().__init__(in_channels, out_channels, ngf)
        
        self.n_scales = n_scales
        
        # Scale-specific generators
        self.scale_generators = nn.ModuleList()
        for i in range(n_scales):
            scale_ngf = ngf // (2 ** i)  # Smaller networks for coarser scales
            self.scale_generators.append(
                SAGenerator(
                    in_channels, out_channels, scale_ngf,
                    n_residual=6, n_downsampling=2
                )
            )
            
        # Multi-scale fusion
        self.fusion = MultiScaleFusion(out_channels, n_scales)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale processing."""
        scale_outputs = []
        
        current_input = x
        for i, generator in enumerate(self.scale_generators):
            # Generate at this scale
            output = generator(current_input)
            scale_outputs.append(output)
            
            # Downsample for next scale
            if i < self.n_scales - 1:
                current_input = F.avg_pool2d(x, 2 ** (i + 1))
                
        # Fuse multi-scale outputs
        fused = self.fusion(scale_outputs)
        
        return fused


class MultiScaleFusion(nn.Module):
    """Fuses outputs from multiple scales."""
    
    def __init__(self, channels: int, n_scales: int):
        super().__init__()
        
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, scale_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Fuse multi-scale outputs."""
        target_size = scale_outputs[0].shape[-2:]
        
        # Normalize weights
        weights = self.softmax(self.scale_weights)
        
        # Weighted sum with upsampling
        fused = torch.zeros_like(scale_outputs[0])
        for i, output in enumerate(scale_outputs):
            if output.shape[-2:] != target_size:
                output = F.interpolate(output, size=target_size, mode='bilinear', align_corners=False)
            fused += weights[i] * output
            
        return fused


class DenseSAGenerator(BaseGenerator):
    """
    Dense Self-Attention Generator.
    
    Uses dense connections in addition to attention for maximum
    feature reuse.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        ngf: Base number of filters
        n_dense_blocks: Number of dense blocks
        growth_rate: Growth rate for dense connections
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        ngf: int = 64,
        n_dense_blocks: int = 4,
        growth_rate: int = 32
    ):
        super().__init__(in_channels, out_channels, ngf)
        
        # Initial
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        # Downsample
        self.downsample = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        
        # Dense blocks with attention
        self.dense_blocks = nn.ModuleList()
        current_channels = ngf * 4
        
        for i in range(n_dense_blocks):
            block = DenseResidualBlock(
                current_channels, growth_rate, n_layers=4
            )
            self.dense_blocks.append(block)
            current_channels += growth_rate * 4
            
            # Add attention after each dense block
            self.dense_blocks.append(SelfAttention2d(current_channels))
            
        # Compress channels
        self.compress = nn.Conv2d(current_channels, ngf * 4, 1)
        
        # Upsample
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        # Final
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.initial(x)
        out = self.downsample(out)
        
        for block in self.dense_blocks:
            if isinstance(block, DenseResidualBlock):
                out = block(out)
            else:
                out = block(out)  # Attention
                
        out = self.compress(out)
        out = self.upsample(out)
        out = self.final(out)
        
        return out
