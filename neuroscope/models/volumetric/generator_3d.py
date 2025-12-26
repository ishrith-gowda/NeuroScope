"""
3D Generator Architectures.

Volumetric generators for 3D medical image translation including:
- Standard 3D ResNet Generator
- Self-Attention 3D Generator
- Memory-efficient variants with gradient checkpointing
- U-Net style 3D generators with skip connections
"""

from typing import Optional, List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .blocks_3d import (
    ResidualBlock3D,
    DownsampleBlock3D,
    UpsampleBlock3D,
    SelfAttention3D,
    MultiHeadSelfAttention3D,
    AxialAttention3D,
    CBAM3D
)


class Generator3D(nn.Module):
    """
    Standard 3D Generator with ResNet architecture.
    
    Architecture: Encoder -> Bottleneck (ResBlocks) -> Decoder
    Suitable for volumetric brain MRI translation.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        ngf: int = 32,  # Reduced for 3D memory constraints
        n_downsampling: int = 2,
        n_residual: int = 6,
        norm_type: str = 'instance',
        use_dropout: bool = False,
        use_checkpoint: bool = True,
        padding_mode: str = 'reflect'
    ):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad3d(3) if padding_mode == 'reflect' else nn.ConstantPad3d(3, 0),
            nn.Conv3d(in_channels, ngf, 7, 1, 0, bias=False),
            nn.InstanceNorm3d(ngf, affine=True) if norm_type == 'instance' else nn.BatchNorm3d(ngf),
            nn.ReLU(inplace=True)
        )
        
        # Encoder (downsampling)
        self.encoder = nn.ModuleList()
        mult = 1
        for i in range(n_downsampling):
            in_ch = ngf * mult
            out_ch = ngf * mult * 2
            self.encoder.append(
                DownsampleBlock3D(in_ch, out_ch, norm_type=norm_type)
            )
            mult *= 2
        
        # Bottleneck (residual blocks)
        self.bottleneck = nn.ModuleList()
        for i in range(n_residual):
            self.bottleneck.append(
                ResidualBlock3D(
                    ngf * mult,
                    norm_type=norm_type,
                    use_dropout=use_dropout,
                    use_checkpoint=use_checkpoint
                )
            )
        
        # Decoder (upsampling)
        self.decoder = nn.ModuleList()
        for i in range(n_downsampling):
            in_ch = ngf * mult
            out_ch = ngf * mult // 2
            self.decoder.append(
                UpsampleBlock3D(in_ch, out_ch, norm_type=norm_type)
            )
            mult //= 2
        
        # Final convolution
        self.final = nn.Sequential(
            nn.ReflectionPad3d(3) if padding_mode == 'reflect' else nn.ConstantPad3d(3, 0),
            nn.Conv3d(ngf, out_channels, 7, 1, 0),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial
        x = self.initial(x)
        
        # Encoder
        for enc in self.encoder:
            x = enc(x)
        
        # Bottleneck
        for res in self.bottleneck:
            x = res(x)
        
        # Decoder
        for dec in self.decoder:
            x = dec(x)
        
        # Final
        x = self.final(x)
        
        return x


class SAGenerator3D(nn.Module):
    """
    Self-Attention Enhanced 3D Generator.
    
    Integrates self-attention mechanisms at multiple scales for
    capturing long-range dependencies in volumetric medical images.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        ngf: int = 32,
        n_downsampling: int = 2,
        n_residual: int = 6,
        attention_positions: Optional[List[int]] = None,
        attention_type: str = 'self',  # 'self', 'multi_head', 'axial', 'cbam'
        norm_type: str = 'instance',
        use_dropout: bool = False,
        use_checkpoint: bool = True
    ):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        attention_positions = attention_positions or [2, 4]  # Default attention positions
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReplicationPad3d(3),
            nn.Conv3d(in_channels, ngf, 7, 1, 0, bias=False),
            nn.InstanceNorm3d(ngf, affine=True),
            nn.ReLU(inplace=True)
        )
        
        # Encoder with attention
        self.encoder = nn.ModuleList()
        self.encoder_attention = nn.ModuleDict()
        
        mult = 1
        for i in range(n_downsampling):
            in_ch = ngf * mult
            out_ch = ngf * mult * 2
            self.encoder.append(
                DownsampleBlock3D(in_ch, out_ch, norm_type=norm_type)
            )
            # Add attention after encoder
            self.encoder_attention[f'enc_{i}'] = self._build_attention(
                out_ch, attention_type, use_checkpoint
            )
            mult *= 2
        
        # Bottleneck with attention at specified positions
        self.bottleneck = nn.ModuleList()
        self.bottleneck_attention = nn.ModuleDict()
        
        bottleneck_channels = ngf * mult
        for i in range(n_residual):
            self.bottleneck.append(
                ResidualBlock3D(
                    bottleneck_channels,
                    norm_type=norm_type,
                    use_dropout=use_dropout,
                    use_checkpoint=use_checkpoint
                )
            )
            if i in attention_positions:
                self.bottleneck_attention[f'res_{i}'] = self._build_attention(
                    bottleneck_channels, attention_type, use_checkpoint
                )
        
        # Decoder with attention
        self.decoder = nn.ModuleList()
        self.decoder_attention = nn.ModuleDict()
        
        for i in range(n_downsampling):
            in_ch = ngf * mult
            out_ch = ngf * mult // 2
            self.decoder.append(
                UpsampleBlock3D(in_ch, out_ch, norm_type=norm_type)
            )
            self.decoder_attention[f'dec_{i}'] = self._build_attention(
                out_ch, attention_type, use_checkpoint
            )
            mult //= 2
        
        # Final convolution
        self.final = nn.Sequential(
            nn.ReplicationPad3d(3),
            nn.Conv3d(ngf, out_channels, 7, 1, 0),
            nn.Tanh()
        )
    
    def _build_attention(
        self,
        channels: int,
        attention_type: str,
        use_checkpoint: bool
    ) -> nn.Module:
        """Build attention module based on type."""
        if attention_type == 'self':
            return SelfAttention3D(channels, use_checkpoint=use_checkpoint)
        elif attention_type == 'multi_head':
            return MultiHeadSelfAttention3D(
                channels, num_heads=8, use_checkpoint=use_checkpoint
            )
        elif attention_type == 'axial':
            return AxialAttention3D(channels, use_checkpoint=use_checkpoint)
        elif attention_type == 'cbam':
            return CBAM3D(channels)
        else:
            return nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial
        x = self.initial(x)
        
        # Encoder with attention
        encoder_features = []
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            x = self.encoder_attention[f'enc_{i}'](x)
            encoder_features.append(x)
        
        # Bottleneck with attention
        for i, res in enumerate(self.bottleneck):
            x = res(x)
            if f'res_{i}' in self.bottleneck_attention:
                x = self.bottleneck_attention[f'res_{i}'](x)
        
        # Decoder with attention
        for i, dec in enumerate(self.decoder):
            x = dec(x)
            x = self.decoder_attention[f'dec_{i}'](x)
        
        # Final
        x = self.final(x)
        
        return x


class UNetGenerator3D(nn.Module):
    """
    3D U-Net Generator with Skip Connections.
    
    Provides better preservation of fine anatomical details
    through skip connections between encoder and decoder.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        ngf: int = 32,
        n_levels: int = 4,
        use_attention: bool = True,
        attention_type: str = 'self',
        norm_type: str = 'instance',
        use_checkpoint: bool = True
    ):
        super().__init__()
        
        self.n_levels = n_levels
        self.use_attention = use_attention
        self.use_checkpoint = use_checkpoint
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv3d(in_channels, ngf, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(ngf, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(ngf, ngf, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(ngf, affine=True),
            nn.ReLU(inplace=True)
        )
        
        # Encoder path
        self.down_convs = nn.ModuleList()
        self.down_pools = nn.ModuleList()
        self.down_attns = nn.ModuleList() if use_attention else None
        
        mult = 1
        for i in range(n_levels):
            in_ch = ngf * mult
            out_ch = ngf * mult * 2
            
            self.down_convs.append(nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, 1, 1, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.ReLU(inplace=True)
            ))
            self.down_pools.append(nn.MaxPool3d(2))
            
            if use_attention:
                self.down_attns.append(
                    SelfAttention3D(out_ch, use_checkpoint=use_checkpoint)
                )
            
            mult *= 2
        
        # Bottleneck
        bottleneck_ch = ngf * mult
        self.bottleneck = nn.Sequential(
            ResidualBlock3D(bottleneck_ch, use_checkpoint=use_checkpoint),
            SelfAttention3D(bottleneck_ch, use_checkpoint=use_checkpoint) if use_attention else nn.Identity(),
            ResidualBlock3D(bottleneck_ch, use_checkpoint=use_checkpoint),
        )
        
        # Decoder path
        self.up_convs = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_attns = nn.ModuleList() if use_attention else None
        
        for i in range(n_levels):
            in_ch = ngf * mult + ngf * mult // 2  # Skip connection
            out_ch = ngf * mult // 2
            
            self.up_samples.append(
                nn.ConvTranspose3d(ngf * mult, ngf * mult // 2, 2, 2)
            )
            self.up_convs.append(nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, 1, 1, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.ReLU(inplace=True)
            ))
            
            if use_attention:
                self.up_attns.append(
                    SelfAttention3D(out_ch, use_checkpoint=use_checkpoint)
                )
            
            mult //= 2
        
        # Final convolution
        self.final = nn.Sequential(
            nn.Conv3d(ngf, ngf, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(ngf, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(ngf, out_channels, 1),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial
        x = self.initial(x)
        
        # Encoder - save features for skip connections
        skip_features = [x]
        for i in range(self.n_levels):
            x = self.down_pools[i](x)
            x = self.down_convs[i](x)
            if self.use_attention:
                x = self.down_attns[i](x)
            skip_features.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        for i in range(self.n_levels):
            x = self.up_samples[i](x)
            
            # Handle size mismatch
            skip = skip_features[-(i + 2)]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = self.up_convs[i](x)
            
            if self.use_attention:
                x = self.up_attns[i](x)
        
        # Final
        x = self.final(x)
        
        return x


class HybridGenerator2_5D(nn.Module):
    """
    2.5D Hybrid Generator.
    
    Processes volumetric data by combining 2D slices with
    inter-slice context, offering a memory-efficient alternative
    to full 3D processing.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        ngf: int = 64,
        context_slices: int = 3,  # Number of adjacent slices to consider
        n_residual: int = 9,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.context_slices = context_slices
        total_in_channels = in_channels * context_slices
        
        # 2D processing with context
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(total_in_channels, ngf, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.enc1 = self._make_encoder_block(ngf, ngf * 2)
        self.enc2 = self._make_encoder_block(ngf * 2, ngf * 4)
        
        # Bottleneck with 3D context
        self.bottleneck_3d = nn.Conv3d(
            ngf * 4, ngf * 4, (3, 3, 3), padding=1, bias=False
        )
        self.bottleneck_norm = nn.InstanceNorm2d(ngf * 4, affine=True)
        
        # Residual blocks
        self.residual = nn.Sequential(*[
            self._make_residual_block(ngf * 4) for _ in range(n_residual)
        ])
        
        # Self-attention
        if use_attention:
            from ..attention.self_attention import SelfAttention
            self.attention = SelfAttention(ngf * 4)
        else:
            self.attention = nn.Identity()
        
        # Decoder
        self.dec1 = self._make_decoder_block(ngf * 4, ngf * 2)
        self.dec2 = self._make_decoder_block(ngf * 2, ngf)
        
        # Final
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7, 1, 0),
            nn.Tanh()
        )
    
    def _make_encoder_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def _make_decoder_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )
    
    def _make_residual_block(self, channels: int) -> nn.Module:
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(channels, affine=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 2.5D generator.
        
        Args:
            x: Input tensor [B, C, D, H, W] or [B, context*C, H, W]
        
        Returns:
            Output tensor [B, C, H, W]
        """
        # Handle 3D input by extracting context slices
        if x.dim() == 5:
            B, C, D, H, W = x.shape
            center_idx = D // 2
            start_idx = max(0, center_idx - self.context_slices // 2)
            end_idx = min(D, start_idx + self.context_slices)
            
            context = x[:, :, start_idx:end_idx, :, :]
            x = context.view(B, -1, H, W)
        
        # Encoding
        x = self.initial(x)
        x = self.enc1(x)
        x = self.enc2(x)
        
        # Bottleneck
        x = self.residual(x)
        x = self.attention(x)
        
        # Decoding
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.final(x)
        
        return x
