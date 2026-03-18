"""
self-attention generator (sa-generator).

advanced generator with multi-scale self-attention for
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
    self-attention generator for image-to-image translation.
    
    incorporates multi-scale self-attention for capturing long-range
    dependencies, essential for medical image translation.
    
    architecture:
    - initial conv
    - multi-scale encoder with attention
    - attention-enhanced residual bottleneck
    - multi-scale decoder with attention
    - final conv
    
    args:
        in_channels: input channels (e.g., 4 for multi-modal mri)
        out_channels: output channels
        ngf: base number of generator filters
        n_residual: number of residual blocks in bottleneck
        n_downsampling: number of downsampling layers
        attention_type: type of attention ('self', 'cbam', 'multi_scale')
        use_spectral_norm: whether to use spectral normalization
        norm_type: normalization type
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
        
        # normalization layer
        if norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d
        elif norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = nn.Identity
            
        # initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            self._apply_sn(nn.Conv2d(in_channels, ngf, 7), use_spectral_norm),
            norm_layer(ngf),
            nn.ReLU(inplace=True)
        )
        
        # encoder with attention
        self.encoder = SAEncoder(
            ngf, n_downsampling, norm_layer,
            attention_type, use_spectral_norm
        )
        
        # bottleneck channels
        bottleneck_channels = ngf * (2 ** n_downsampling)
        
        # attention-enhanced residual bottleneck
        self.bottleneck = SABottleneck(
            bottleneck_channels, n_residual,
            norm_layer, attention_type, use_spectral_norm
        )
        
        # decoder with attention
        self.decoder = SADecoder(
            bottleneck_channels, ngf, n_downsampling,
            norm_layer, attention_type, use_spectral_norm
        )
        
        # final convolution
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            self._apply_sn(nn.Conv2d(ngf, out_channels, 7), use_spectral_norm),
            nn.Tanh()
        )
        
    def _apply_sn(self, module: nn.Module, use_sn: bool) -> nn.Module:
        """apply spectral normalization if enabled."""
        if use_sn:
            return nn.utils.spectral_norm(module)
        return module
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass."""
        # initial
        out = self.initial(x)
        
        # encode
        encoded, skip_features = self.encoder(out)
        
        # bottleneck
        bottleneck = self.bottleneck(encoded)
        
        # decode with skip connections
        decoded = self.decoder(bottleneck, skip_features)
        
        # final
        output = self.final(decoded)
        
        return output
        
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """get intermediate feature maps for visualization."""
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
    """self-attention encoder."""
    
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
            
            # downsampling block
            down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                norm_layer(out_ch),
                nn.ReLU(inplace=True)
            )
            if use_spectral_norm:
                down[0] = nn.utils.spectral_norm(down[0])
                
            self.down_blocks.append(down)
            
            # attention block
            if attention_type == 'multi_scale':
                attn = MultiScaleSelfAttention(out_ch)
            elif attention_type == 'cbam':
                attn = CBAM(out_ch)
            else:
                attn = SelfAttention2d(out_ch)
                
            self.attention_blocks.append(attn)
            
            mult *= 2
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """forward pass returning encoded features and skip connections."""
        skip_features = []
        
        for down, attn in zip(self.down_blocks, self.attention_blocks):
            x = down(x)
            x = attn(x)
            skip_features.append(x)
            
        return x, skip_features


class SABottleneck(nn.Module):
    """self-attention bottleneck."""
    
    def __init__(
        self,
        channels: int,
        n_residual: int,
        norm_layer,
        attention_type: str,
        use_spectral_norm: bool
    ):
        super().__init__()
        
        # residual blocks with periodic attention
        self.blocks = nn.ModuleList()
        
        for i in range(n_residual):
            # residual block
            self.blocks.append(
                ResidualBlock(channels, norm_type='instance')
            )
            
            # add attention every 3 blocks
            if (i + 1) % 3 == 0:
                if attention_type == 'multi_scale':
                    self.blocks.append(MultiScaleSelfAttention(channels))
                elif attention_type == 'cbam':
                    self.blocks.append(CBAM(channels))
                else:
                    self.blocks.append(SelfAttention2d(channels))
                    
        # final attention
        self.final_attention = EfficientSelfAttention(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass through bottleneck."""
        for block in self.blocks:
            x = block(x)
            
        x = self.final_attention(x)
        
        return x


class SADecoder(nn.Module):
    """self-attention decoder."""
    
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
            
            # upsampling block
            up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
                norm_layer(out_ch),
                nn.ReLU(inplace=True)
            )
            if use_spectral_norm:
                up[0] = nn.utils.spectral_norm(up[0])
                
            self.up_blocks.append(up)
            
            # skip connection processing - match skip channels to decoder channels
            # skip from encoder at this level has in_ch channels (same as decoder input at this level)
            self.skip_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1),  # project from skip channels to output channels
                    norm_layer(out_ch)
                )
            )
            
            # attention block
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
        """forward pass with skip connections."""
        # reverse skip features for decoder order
        skips = skip_features[::-1]
        
        for i, (up, skip_conv, attn) in enumerate(
            zip(self.up_blocks, self.skip_convs, self.attention_blocks)
        ):
            x = up(x)
            
            # add skip connection if available
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
    multi-scale self-attention generator.
    
    processes input at multiple scales and fuses results.
    
    args:
        in_channels: input channels
        out_channels: output channels
        ngf: base number of filters
        n_scales: number of scales to process
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
        
        # scale-specific generators
        self.scale_generators = nn.ModuleList()
        for i in range(n_scales):
            scale_ngf = ngf // (2 ** i)  # smaller networks for coarser scales
            self.scale_generators.append(
                SAGenerator(
                    in_channels, out_channels, scale_ngf,
                    n_residual=6, n_downsampling=2
                )
            )
            
        # multi-scale fusion
        self.fusion = MultiScaleFusion(out_channels, n_scales)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass with multi-scale processing."""
        scale_outputs = []
        
        current_input = x
        for i, generator in enumerate(self.scale_generators):
            # generate at this scale
            output = generator(current_input)
            scale_outputs.append(output)
            
            # downsample for next scale
            if i < self.n_scales - 1:
                current_input = F.avg_pool2d(x, 2 ** (i + 1))
                
        # fuse multi-scale outputs
        fused = self.fusion(scale_outputs)
        
        return fused


class MultiScaleFusion(nn.Module):
    """fuses outputs from multiple scales."""
    
    def __init__(self, channels: int, n_scales: int):
        super().__init__()
        
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, scale_outputs: List[torch.Tensor]) -> torch.Tensor:
        """fuse multi-scale outputs."""
        target_size = scale_outputs[0].shape[-2:]
        
        # normalize weights
        weights = self.softmax(self.scale_weights)
        
        # weighted sum with upsampling
        fused = torch.zeros_like(scale_outputs[0])
        for i, output in enumerate(scale_outputs):
            if output.shape[-2:] != target_size:
                output = F.interpolate(output, size=target_size, mode='bilinear', align_corners=False)
            fused += weights[i] * output
            
        return fused


class DenseSAGenerator(BaseGenerator):
    """
    dense self-attention generator.
    
    uses dense connections in addition to attention for maximum
    feature reuse.
    
    args:
        in_channels: input channels
        out_channels: output channels
        ngf: base number of filters
        n_dense_blocks: number of dense blocks
        growth_rate: growth rate for dense connections
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
        
        # initial
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        # downsample
        self.downsample = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        
        # dense blocks with attention
        self.dense_blocks = nn.ModuleList()
        current_channels = ngf * 4
        
        for i in range(n_dense_blocks):
            block = DenseResidualBlock(
                current_channels, growth_rate, n_layers=4
            )
            self.dense_blocks.append(block)
            current_channels += growth_rate * 4
            
            # add attention after each dense block
            self.dense_blocks.append(SelfAttention2d(current_channels))
            
        # compress channels
        self.compress = nn.Conv2d(current_channels, ngf * 4, 1)
        
        # upsample
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        # final
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass."""
        out = self.initial(x)
        out = self.downsample(out)
        
        for block in self.dense_blocks:
            if isinstance(block, DenseResidualBlock):
                out = block(out)
            else:
                out = block(out)  # attention
                
        out = self.compress(out)
        out = self.upsample(out)
        out = self.final(out)
        
        return out
