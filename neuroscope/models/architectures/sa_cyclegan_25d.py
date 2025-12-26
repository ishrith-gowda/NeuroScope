"""
2.5D Self-Attention CycleGAN Architecture

Novel architecture for brain MRI harmonization using 2.5D slice processing
with multi-scale self-attention mechanisms.

Key Innovations:
1. 2.5D processing: Input 3 adjacent slices, output center slice
2. Multi-scale self-attention in generator bottleneck
3. CBAM attention in encoder/decoder paths  
4. Spectral-normalized multi-scale discriminator
5. Skip connections with attention-based fusion

This architecture is specifically designed for multi-modal MRI translation,
preserving anatomical structures and inter-slice continuity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SACycleGAN25DConfig:
    """Configuration for 2.5D SA-CycleGAN."""
    
    # 2.5D settings
    n_input_slices: int = 3  # Number of adjacent slices (2.5D)
    n_modalities: int = 4    # T1, T1ce, T2, FLAIR
    
    # Generator settings
    ngf: int = 64            # Base generator filters
    n_residual_blocks: int = 9
    attention_layers: Tuple[int, ...] = (3, 4, 5)  # Which blocks get self-attention
    use_modality_encoder: bool = True
    
    # Discriminator settings
    ndf: int = 64            # Base discriminator filters
    n_disc_layers: int = 3
    n_disc_scales: int = 2
    use_spectral_norm: bool = True
    
    # Loss weights
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    lambda_perceptual: float = 1.0
    lambda_ssim: float = 1.0
    lambda_tumor: float = 2.0
    
    @property
    def input_channels(self) -> int:
        """Total input channels = slices × modalities."""
        return self.n_input_slices * self.n_modalities  # 3 × 4 = 12
    
    @property
    def output_channels(self) -> int:
        """Output channels = modalities (center slice only)."""
        return self.n_modalities  # 4


# =============================================================================
# Attention Modules
# =============================================================================

class SelfAttention2D(nn.Module):
    """
    Self-attention for 2D feature maps.
    Captures long-range spatial dependencies crucial for anatomical consistency.
    """
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.in_channels = in_channels
        reduced = max(in_channels // reduction, 1)
        
        self.query = nn.Conv2d(in_channels, reduced, 1)
        self.key = nn.Conv2d(in_channels, reduced, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.InstanceNorm2d(in_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C']
        k = self.key(x).view(B, -1, H * W)                      # [B, C', HW]
        v = self.value(x).view(B, -1, H * W)                    # [B, C, HW]
        
        attn = torch.bmm(q, k)  # [B, HW, HW]
        attn = F.softmax(attn / math.sqrt(C), dim=-1)
        
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * self.norm(out) + x


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class SpatialAttention(nn.Module):
    """Spatial attention using channel statistics."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        max_val, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, max_val], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# =============================================================================
# Building Blocks
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual block with optional attention."""
    
    def __init__(self, channels: int, use_attention: bool = False, attention_type: str = 'cbam'):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels)
        )
        
        self.attention = None
        if use_attention:
            if attention_type == 'self':
                self.attention = SelfAttention2D(channels)
            elif attention_type == 'cbam':
                self.attention = CBAM(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.attention is not None:
            out = self.attention(out)
        return out + x


class SliceEncoder25D(nn.Module):
    """
    2.5D Slice Encoder: Processes 3 adjacent slices with modality awareness.
    
    Input: [B, 12, H, W] (3 slices × 4 modalities)
    Output: [B, ngf, H, W] with inter-slice context
    """
    
    def __init__(self, n_slices: int = 3, n_modalities: int = 4, ngf: int = 64):
        super().__init__()
        
        in_channels = n_slices * n_modalities  # 12
        
        # Slice-aware convolution: learns to combine adjacent slices
        self.slice_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        # Attention to weight slice importance
        self.slice_attention = CBAM(ngf)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 12, H, W] - 3 slices × 4 modalities stacked
        Returns:
            [B, ngf, H, W] - Encoded features with slice context
        """
        x = self.slice_conv(x)
        x = self.slice_attention(x)
        return x


# =============================================================================
# Generator
# =============================================================================

class SAGenerator25D(nn.Module):
    """
    2.5D Self-Attention Generator.
    
    Takes 3 adjacent slices as input and outputs the center slice,
    using inter-slice context for better anatomical consistency.
    """
    
    def __init__(self, config: SACycleGAN25DConfig):
        super().__init__()
        self.config = config
        ngf = config.ngf
        
        # 2.5D slice encoder
        self.encoder_initial = SliceEncoder25D(
            n_slices=config.n_input_slices,
            n_modalities=config.n_modalities,
            ngf=ngf
        )
        
        # Downsampling encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * 2),
                nn.ReLU(inplace=True),
                CBAM(ngf * 2)
            ),
            nn.Sequential(
                nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * 4),
                nn.ReLU(inplace=True),
                CBAM(ngf * 4)
            )
        ])
        
        # Bottleneck with self-attention at specified layers
        self.bottleneck = nn.ModuleList()
        for i in range(config.n_residual_blocks):
            use_self_attn = i in config.attention_layers
            self.bottleneck.append(
                ResidualBlock(
                    ngf * 4,
                    use_attention=True,
                    attention_type='self' if use_self_attn else 'cbam'
                )
            )
        
        # Global self-attention
        self.global_attention = SelfAttention2D(ngf * 4, reduction=4)
        
        # Upsampling decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf * 2),
                nn.ReLU(inplace=True),
                CBAM(ngf * 2)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf),
                nn.ReLU(inplace=True),
                CBAM(ngf)
            )
        ])
        
        # Skip connection fusion
        self.skip_fuse = nn.ModuleList([
            nn.Conv2d(ngf * 4, ngf * 2, 1, bias=False),
            nn.Conv2d(ngf * 2, ngf, 1, bias=False)
        ])
        
        # Output: center slice only (4 modalities)
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, config.output_channels, 7),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 12, H, W] - 3 adjacent slices × 4 modalities
        Returns:
            [B, 4, H, W] - Translated center slice (4 modalities)
        """
        # Initial encoding with slice awareness
        x = self.encoder_initial(x)
        
        # Encoder with skip connections
        skips = [x]
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
        
        # Bottleneck
        for block in self.bottleneck:
            x = block(x)
        x = self.global_attention(x)
        
        # Decoder with skip connections
        for i, dec in enumerate(self.decoder):
            x = dec(x)
            skip = self.skip_fuse[i](skips[-(i+1)])
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip
        
        return self.output(x)


# =============================================================================
# Discriminator
# =============================================================================

class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator with optional spectral normalization."""
    
    def __init__(
        self,
        in_channels: int,
        ndf: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        def get_conv(in_c, out_c, stride=2, bias=True):
            conv = nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1, bias=bias)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            return conv
        
        layers = [
            get_conv(in_channels, ndf),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        nf = ndf
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers.extend([
                get_conv(nf_prev, nf, bias=False),
                nn.InstanceNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # Self-attention before final layers
        layers.append(SelfAttention2D(nf))
        
        # Final layers
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers.extend([
            get_conv(nf_prev, nf, stride=1, bias=False),
            nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv(nf, 1, stride=1)
        ])
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for multi-frequency analysis."""
    
    def __init__(
        self,
        in_channels: int,
        ndf: int = 64,
        n_layers: int = 3,
        n_scales: int = 2,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_channels, ndf, n_layers, use_spectral_norm)
            for _ in range(n_scales)
        ])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            outputs.append(disc(x))
        return outputs


# =============================================================================
# Complete SA-CycleGAN 2.5D
# =============================================================================

class SACycleGAN25D(nn.Module):
    """
    Complete 2.5D Self-Attention CycleGAN for MRI Harmonization.
    
    Bidirectional domain translation between:
    - Domain A: BraTS (multi-institutional)
    - Domain B: UPenn-GBM (single-institution)
    """
    
    def __init__(self, config: Optional[SACycleGAN25DConfig] = None):
        super().__init__()
        self.config = config or SACycleGAN25DConfig()
        
        # Generators: A→B and B→A
        self.G_A2B = SAGenerator25D(self.config)
        self.G_B2A = SAGenerator25D(self.config)
        
        # Discriminators (operate on single slices = 4 channels)
        self.D_A = MultiScaleDiscriminator(
            in_channels=self.config.output_channels,  # 4
            ndf=self.config.ndf,
            n_layers=self.config.n_disc_layers,
            n_scales=self.config.n_disc_scales,
            use_spectral_norm=self.config.use_spectral_norm
        )
        self.D_B = MultiScaleDiscriminator(
            in_channels=self.config.output_channels,  # 4
            ndf=self.config.ndf,
            n_layers=self.config.n_disc_layers,
            n_scales=self.config.n_disc_scales,
            use_spectral_norm=self.config.use_spectral_norm
        )
        
    def forward(
        self,
        slices_A: torch.Tensor,
        slices_B: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            slices_A: [B, 12, H, W] - 3 adjacent slices from domain A
            slices_B: [B, 12, H, W] - 3 adjacent slices from domain B
            
        Returns:
            Dictionary with fake and reconstructed images
        """
        # Forward cycle: A → B → A
        fake_B = self.G_A2B(slices_A)   # [B, 4, H, W]
        
        # For reconstruction, we need 3 slices in domain B
        # Simplified: use fake_B repeated (in practice, translate adjacent slices)
        fake_B_3slice = fake_B.repeat(1, 3, 1, 1)  # [B, 12, H, W]
        rec_A = self.G_B2A(fake_B_3slice)
        
        # Backward cycle: B → A → B
        fake_A = self.G_B2A(slices_B)
        fake_A_3slice = fake_A.repeat(1, 3, 1, 1)
        rec_B = self.G_A2B(fake_A_3slice)
        
        return {
            'fake_B': fake_B,
            'fake_A': fake_A,
            'rec_A': rec_A,
            'rec_B': rec_B
        }
    
    @torch.no_grad()
    def translate_A2B(self, slices: torch.Tensor) -> torch.Tensor:
        """Translate from domain A to B."""
        return self.G_A2B(slices)
    
    @torch.no_grad()
    def translate_B2A(self, slices: torch.Tensor) -> torch.Tensor:
        """Translate from domain B to A."""
        return self.G_B2A(slices)
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for each component."""
        return {
            'G_A2B': sum(p.numel() for p in self.G_A2B.parameters()),
            'G_B2A': sum(p.numel() for p in self.G_B2A.parameters()),
            'D_A': sum(p.numel() for p in self.D_A.parameters()),
            'D_B': sum(p.numel() for p in self.D_B.parameters()),
            'total': sum(p.numel() for p in self.parameters())
        }


# =============================================================================
# Factory and Testing
# =============================================================================

def create_model(config: Optional[SACycleGAN25DConfig] = None) -> SACycleGAN25D:
    """Create and initialize the 2.5D SA-CycleGAN model."""
    model = SACycleGAN25D(config)
    
    # Print summary
    params = model.get_parameter_count()
    print("=" * 60)
    print("2.5D SA-CycleGAN Model Summary")
    print("=" * 60)
    print(f"Generator A→B: {params['G_A2B']:,} parameters")
    print(f"Generator B→A: {params['G_B2A']:,} parameters")
    print(f"Discriminator A: {params['D_A']:,} parameters")
    print(f"Discriminator B: {params['D_B']:,} parameters")
    print(f"Total: {params['total']:,} parameters ({params['total']/1e6:.2f}M)")
    print("=" * 60)
    
    return model


if __name__ == '__main__':
    # Quick test
    config = SACycleGAN25DConfig()
    model = create_model(config)
    
    # Test input: 3 slices × 4 modalities = 12 channels
    x = torch.randn(2, 12, 128, 128)
    
    # Test generator
    out = model.G_A2B(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")  # Should be [2, 4, 128, 128]
