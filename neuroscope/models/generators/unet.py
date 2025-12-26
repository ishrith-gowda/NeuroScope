"""
U-Net Generator Architecture.

U-Net style generators with skip connections for
image-to-image translation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from ..blocks.residual import ResidualBlock
from ..attention.self_attention import SelfAttention2d
from ..attention.spatial_attention import CBAM
from .base import BaseGenerator


class UNetGenerator(BaseGenerator):
    """
    U-Net Generator for image-to-image translation.
    
    Classic U-Net architecture with encoder-decoder and skip connections.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        ngf: Base number of filters
        n_downsampling: Number of downsampling layers
        norm_type: Normalization type
        use_dropout: Whether to use dropout
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        ngf: int = 64,
        n_downsampling: int = 4,
        norm_type: str = 'batch',
        use_dropout: bool = True
    ):
        super().__init__(in_channels, out_channels, ngf)
        
        self.n_downsampling = n_downsampling
        
        # Normalization
        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.Identity
            
        # Build U-Net
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        # Encoder path
        in_ch = in_channels
        encoder_channels = []
        
        for i in range(n_downsampling):
            out_ch = min(ngf * (2 ** i), 512)
            encoder_channels.append(out_ch)
            
            self.encoder.append(
                UNetEncoderBlock(in_ch, out_ch, norm_layer, use_dropout if i > 0 else False)
            )
            in_ch = out_ch
            
        # Bottleneck
        bottleneck_ch = min(ngf * (2 ** n_downsampling), 512)
        self.bottleneck = UNetEncoderBlock(in_ch, bottleneck_ch, norm_layer, use_dropout)
        
        # Decoder path
        in_ch = bottleneck_ch
        for i in range(n_downsampling - 1, -1, -1):
            out_ch = encoder_channels[i]
            # Input includes skip connection
            self.decoder.append(
                UNetDecoderBlock(in_ch + out_ch, out_ch, norm_layer, use_dropout if i > 0 else False)
            )
            in_ch = out_ch
            
        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_channels, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connections."""
        # Encode
        encoder_features = []
        out = x
        
        for encoder_block in self.encoder:
            out = encoder_block(out)
            encoder_features.append(out)
            
        # Bottleneck
        out = self.bottleneck(out)
        
        # Decode with skip connections
        for i, decoder_block in enumerate(self.decoder):
            skip = encoder_features[-(i + 1)]
            
            # Handle size mismatch
            if skip.shape[-2:] != out.shape[-2:]:
                out = F.interpolate(out, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                
            out = torch.cat([out, skip], dim=1)
            out = decoder_block(out)
            
        # Final
        out = self.final(out)
        
        return out


class UNetEncoderBlock(nn.Module):
    """U-Net encoder block (conv + norm + activation + pool)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer,
        use_dropout: bool = False
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        if use_dropout:
            layers.append(nn.Dropout(0.5))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetDecoderBlock(nn.Module):
    """U-Net decoder block (upsample + conv + norm + activation)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer,
        use_dropout: bool = False
    ):
        super().__init__()
        
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        if use_dropout:
            layers.append(nn.Dropout(0.5))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionUNetGenerator(BaseGenerator):
    """
    Attention U-Net Generator.
    
    U-Net with attention gates for skip connections.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        ngf: Base number of filters
        n_downsampling: Number of downsampling layers
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        ngf: int = 64,
        n_downsampling: int = 4
    ):
        super().__init__(in_channels, out_channels, ngf)
        
        self.n_downsampling = n_downsampling
        
        # Encoder
        self.encoder = nn.ModuleList()
        in_ch = in_channels
        encoder_channels = []
        
        for i in range(n_downsampling):
            out_ch = min(ngf * (2 ** i), 512)
            encoder_channels.append(out_ch)
            
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                )
            )
            in_ch = out_ch
            
        # Bottleneck
        bottleneck_ch = min(ngf * (2 ** n_downsampling), 512)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch, bottleneck_ch, 3, padding=1),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_ch, bottleneck_ch, 3, padding=1),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with attention gates
        self.decoder = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        in_ch = bottleneck_ch
        for i in range(n_downsampling - 1, -1, -1):
            out_ch = encoder_channels[i]
            
            # Attention gate
            self.attention_gates.append(
                AttentionGate(out_ch, in_ch, out_ch // 2)
            )
            
            # Upsample
            self.upsample.append(
                nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
            )
            
            # Decoder block
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
            
            in_ch = out_ch
            
        # Final
        self.final = nn.Conv2d(in_ch, out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention gates."""
        # Encode
        encoder_features = []
        out = x
        
        for enc_block in self.encoder:
            out = enc_block(out)
            encoder_features.append(out)
            
        # Bottleneck
        out = self.bottleneck(out)
        
        # Decode with attention
        for i, (attn_gate, up, dec_block) in enumerate(
            zip(self.attention_gates, self.upsample, self.decoder)
        ):
            skip = encoder_features[-(i + 1)]
            
            # Apply attention gate
            attended_skip = attn_gate(skip, out)
            
            # Upsample
            out = up(out)
            
            # Handle size mismatch
            if out.shape[-2:] != attended_skip.shape[-2:]:
                out = F.interpolate(out, size=attended_skip.shape[-2:], mode='bilinear', align_corners=False)
                
            # Concatenate and decode
            out = torch.cat([out, attended_skip], dim=1)
            out = dec_block(out)
            
        return self.final(out)


class AttentionGate(nn.Module):
    """Attention gate for skip connections."""
    
    def __init__(
        self,
        skip_channels: int,
        gating_channels: int,
        inter_channels: int
    ):
        super().__init__()
        
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.W_gate = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(
        self,
        skip: torch.Tensor,
        gating: torch.Tensor
    ) -> torch.Tensor:
        """Apply attention to skip connection."""
        # Resize gating signal
        gating_resized = F.interpolate(
            gating, size=skip.shape[-2:],
            mode='bilinear', align_corners=False
        )
        
        skip_proj = self.W_skip(skip)
        gate_proj = self.W_gate(gating_resized)
        
        attention = self.relu(skip_proj + gate_proj)
        attention = self.psi(attention)
        
        return skip * attention


class ResUNetGenerator(BaseGenerator):
    """
    Residual U-Net Generator.
    
    U-Net with residual blocks for improved gradient flow.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        ngf: Base number of filters
        n_downsampling: Number of downsampling layers
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        ngf: int = 64,
        n_downsampling: int = 4
    ):
        super().__init__(in_channels, out_channels, ngf)
        
        # Initial conv
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, ngf, 3, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        
        in_ch = ngf
        encoder_channels = [ngf]
        
        for i in range(n_downsampling):
            out_ch = min(in_ch * 2, 512)
            self.encoder.append(ResidualEncoderBlock(in_ch, out_ch))
            encoder_channels.append(out_ch)
            in_ch = out_ch
            
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(in_ch),
            ResidualBlock(in_ch)
        )
        
        # Decoder
        self.decoder = nn.ModuleList()
        
        for i in range(n_downsampling):
            skip_ch = encoder_channels[-(i + 2)]
            out_ch = skip_ch
            self.decoder.append(ResidualDecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch
            
        # Final
        self.final = nn.Sequential(
            nn.Conv2d(in_ch, out_channels, 1),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Initial
        out = self.initial(x)
        
        # Encode
        encoder_features = [out]
        for enc_block in self.encoder:
            out = self.pool(out)
            out = enc_block(out)
            encoder_features.append(out)
            
        # Bottleneck
        out = self.bottleneck(out)
        
        # Decode
        for i, dec_block in enumerate(self.decoder):
            skip = encoder_features[-(i + 2)]
            out = dec_block(out, skip)
            
        # Final
        return self.final(out)


class ResidualEncoderBlock(nn.Module):
    """Residual encoder block."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


class ResidualDecoderBlock(nn.Module):
    """Residual decoder block with skip connection."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int
    ):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 2, 2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Conv2d(in_channels + skip_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor
    ) -> torch.Tensor:
        # Upsample
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            
        # Concatenate
        combined = torch.cat([x, skip], dim=1)
        
        # Residual connection
        return self.relu(self.conv(combined) + self.shortcut(combined))


class UNetPlusPlusGenerator(BaseGenerator):
    """
    U-Net++ (Nested U-Net) Generator.
    
    Dense skip connections for multi-scale feature fusion.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        ngf: Base number of filters
        depth: Network depth
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        ngf: int = 64,
        depth: int = 4
    ):
        super().__init__(in_channels, out_channels, ngf)
        
        self.depth = depth
        
        # Create all conv blocks
        self.conv = nn.ModuleDict()
        
        for i in range(depth + 1):
            for j in range(depth + 1 - i):
                # Input channels
                if i == 0:
                    in_ch = in_channels if j == 0 else ngf * (2 ** j)
                else:
                    # Sum of all previous columns in same row + decoder from below
                    in_ch = ngf * (2 ** j) * (i + 1) + ngf * (2 ** (j + 1)) if j < depth - i else ngf * (2 ** j) * i
                    
                out_ch = ngf * (2 ** j)
                
                self.conv[f'x_{i}_{j}'] = ConvBlock(in_ch if i > 0 or j > 0 else in_channels, out_ch)
                
        # Final output
        self.final = nn.Conv2d(ngf * depth, out_channels, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through nested U-Net."""
        # Store all intermediate outputs
        outputs = {}
        
        # First column (encoder path)
        for j in range(self.depth + 1):
            if j == 0:
                outputs[f'x_0_{j}'] = self.conv[f'x_0_{j}'](x)
            else:
                outputs[f'x_0_{j}'] = self.conv[f'x_0_{j}'](self.pool(outputs[f'x_0_{j-1}']))
                
        # Remaining columns (nested paths)
        for i in range(1, self.depth + 1):
            for j in range(self.depth + 1 - i):
                # Collect all previous outputs from same row
                prev_outputs = [outputs[f'x_{k}_{j}'] for k in range(i)]
                
                # Add upsampled output from below
                up_output = self.up(outputs[f'x_{i-1}_{j+1}'])
                if up_output.shape[-2:] != prev_outputs[0].shape[-2:]:
                    up_output = F.interpolate(up_output, size=prev_outputs[0].shape[-2:], mode='bilinear', align_corners=False)
                    
                # Concatenate all
                combined = torch.cat(prev_outputs + [up_output], dim=1)
                outputs[f'x_{i}_{j}'] = self.conv[f'x_{i}_{j}'](combined)
                
        # Collect final outputs from top row
        final_outputs = [outputs[f'x_{i}_0'] for i in range(1, self.depth + 1)]
        final = torch.cat(final_outputs, dim=1)
        
        return self.final(final)


class ConvBlock(nn.Module):
    """Double convolution block."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
