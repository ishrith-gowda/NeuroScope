"""
2.5d self-attention cyclegan architecture

novel architecture for brain mri harmonization using 2.5d slice processing
with multi-scale self-attention mechanisms.

key innovations:
1. 2.5d processing: input 3 adjacent slices, output center slice
2. multi-scale self-attention in generator bottleneck
3. cbam attention in encoder/decoder paths  
4. spectral-normalized multi-scale discriminator
5. skip connections with attention-based fusion

this architecture is specifically designed for multi-modal mri translation,
preserving anatomical structures and inter-slice continuity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math


# =============================================================================
# configuration
# =============================================================================

@dataclass
class SACycleGAN25DConfig:
    """configuration for 2.5d sa-cyclegan."""
    
    # 2.5d settings
    n_input_slices: int = 3  # number of adjacent slices (2.5d)
    n_modalities: int = 4    # t1, t1ce, t2, flair
    
    # generator settings
    ngf: int = 64            # base generator filters
    n_residual_blocks: int = 9
    attention_layers: Tuple[int, ...] = (3, 4, 5)  # which blocks get self-attention
    use_modality_encoder: bool = True
    
    # discriminator settings
    ndf: int = 64            # base discriminator filters
    n_disc_layers: int = 3
    n_disc_scales: int = 2
    use_spectral_norm: bool = True
    use_disc_attention: bool = True  # self-attention in discriminator

    # loss weights
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    lambda_perceptual: float = 1.0
    lambda_ssim: float = 1.0
    lambda_tumor: float = 2.0
    
    @property
    def input_channels(self) -> int:
        """total input channels = slices x modalities."""
        return self.n_input_slices * self.n_modalities  # 3 x 4 = 12
    
    @property
    def output_channels(self) -> int:
        """output channels = modalities (center slice only)."""
        return self.n_modalities  # 4


# =============================================================================
# attention modules
# =============================================================================

class SelfAttention2D(nn.Module):
    """
    self-attention for 2d feature maps.
    captures long-range spatial dependencies crucial for anatomical consistency.
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

        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # [b, hw, c']
        k = self.key(x).view(B, -1, H * W)                      # [b, c', hw]
        v = self.value(x).view(B, -1, H * W)                    # [b, c, hw]

        # compute attention with numerical stability
        C_reduced = q.size(-1)  # reduced channel dimension
        attn = torch.bmm(q, k)  # [b, hw, hw]

        # clamp attention logits to prevent overflow in softmax
        attn = torch.clamp(attn / math.sqrt(C_reduced), min=-50, max=50)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)

        # clamp gamma to prevent amplification of unstable outputs
        gamma_clamped = torch.clamp(self.gamma, min=-1.0, max=1.0)
        return gamma_clamped * self.norm(out) + x


class ChannelAttention(nn.Module):
    """squeeze-and-excitation style channel attention."""

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
        # clamp attention weights to prevent complete suppression
        attn = torch.clamp(self.fc(x), min=0.01, max=1.0)
        return x * attn


class SpatialAttention(nn.Module):
    """spatial attention using channel statistics."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        max_val, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, max_val], dim=1)))
        # clamp attention weights to prevent complete suppression
        attn = torch.clamp(attn, min=0.01, max=1.0)
        return x * attn


class CBAM(nn.Module):
    """convolutional block attention module."""
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# =============================================================================
# building blocks
# =============================================================================

class ResidualBlock(nn.Module):
    """residual block with optional attention."""
    
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
    2.5d slice encoder: processes 3 adjacent slices with modality awareness.
    
    input: [b, 12, h, w] (3 slices x 4 modalities)
    output: [b, ngf, h, w] with inter-slice context
    """
    
    def __init__(self, n_slices: int = 3, n_modalities: int = 4, ngf: int = 64):
        super().__init__()
        
        in_channels = n_slices * n_modalities  # 12
        
        # slice-aware convolution: learns to combine adjacent slices
        self.slice_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        # attention to weight slice importance
        self.slice_attention = CBAM(ngf)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: [b, 12, h, w] - 3 slices x 4 modalities stacked
        returns:
            [b, ngf, h, w] - encoded features with slice context
        """
        x = self.slice_conv(x)
        x = self.slice_attention(x)
        return x


# =============================================================================
# generator
# =============================================================================

class SAGenerator25D(nn.Module):
    """
    2.5d self-attention generator.
    
    takes 3 adjacent slices as input and outputs the center slice,
    using inter-slice context for better anatomical consistency.
    """
    
    def __init__(self, config: SACycleGAN25DConfig):
        super().__init__()
        self.config = config
        ngf = config.ngf
        
        # 2.5d slice encoder
        self.encoder_initial = SliceEncoder25D(
            n_slices=config.n_input_slices,
            n_modalities=config.n_modalities,
            ngf=ngf
        )
        
        # downsampling encoder
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
        
        # bottleneck with self-attention at specified layers
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
        
        # global self-attention
        self.global_attention = SelfAttention2D(ngf * 4, reduction=4)
        
        # upsampling decoder
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
        
        # skip connection fusion
        self.skip_fuse = nn.ModuleList([
            nn.Conv2d(ngf * 4, ngf * 2, 1, bias=False),
            nn.Conv2d(ngf * 2, ngf, 1, bias=False)
        ])
        
        # output: center slice only (4 modalities)
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, config.output_channels, 7),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: [b, 12, h, w] - 3 adjacent slices x 4 modalities
        returns:
            [b, 4, h, w] - translated center slice (4 modalities)
        """
        # initial encoding with slice awareness
        x = self.encoder_initial(x)
        
        # encoder with skip connections
        skips = [x]
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
        
        # bottleneck
        for block in self.bottleneck:
            x = block(x)
        x = self.global_attention(x)
        
        # decoder with skip connections
        for i, dec in enumerate(self.decoder):
            x = dec(x)
            skip = self.skip_fuse[i](skips[-(i+1)])
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip
        
        return self.output(x)


# =============================================================================
# discriminator
# =============================================================================

class PatchDiscriminator(nn.Module):
    """patchgan discriminator with optional spectral normalization."""

    def __init__(
        self,
        in_channels: int,
        ndf: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
        use_attention: bool = True
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

        # optional self-attention before final layers
        if use_attention:
            layers.append(SelfAttention2D(nf))

        # final layers
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
    """multi-scale discriminator for multi-frequency analysis."""

    def __init__(
        self,
        in_channels: int,
        ndf: int = 64,
        n_layers: int = 3,
        n_scales: int = 2,
        use_spectral_norm: bool = True,
        use_attention: bool = True
    ):
        super().__init__()

        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_channels, ndf, n_layers, use_spectral_norm, use_attention)
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
# complete sa-cyclegan 2.5d
# =============================================================================

class SACycleGAN25D(nn.Module):
    """
    complete 2.5d self-attention cyclegan for mri harmonization.
    
    bidirectional domain translation between:
    - domain a: brats (multi-institutional)
    - domain b: upenn-gbm (single-institution)
    """
    
    def __init__(self, config: Optional[SACycleGAN25DConfig] = None):
        super().__init__()
        self.config = config or SACycleGAN25DConfig()
        
        # generators: a→b and b→a
        self.G_A2B = SAGenerator25D(self.config)
        self.G_B2A = SAGenerator25D(self.config)
        
        # discriminators (operate on single slices = 4 channels)
        self.D_A = MultiScaleDiscriminator(
            in_channels=self.config.output_channels,  # 4
            ndf=self.config.ndf,
            n_layers=self.config.n_disc_layers,
            n_scales=self.config.n_disc_scales,
            use_spectral_norm=self.config.use_spectral_norm,
            use_attention=self.config.use_disc_attention
        )
        self.D_B = MultiScaleDiscriminator(
            in_channels=self.config.output_channels,  # 4
            ndf=self.config.ndf,
            n_layers=self.config.n_disc_layers,
            n_scales=self.config.n_disc_scales,
            use_spectral_norm=self.config.use_spectral_norm,
            use_attention=self.config.use_disc_attention
        )
        
    def forward(
        self,
        slices_A: torch.Tensor,
        slices_B: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        training forward pass.
        
        args:
            slices_a: [b, 12, h, w] - 3 adjacent slices from domain a
            slices_b: [b, 12, h, w] - 3 adjacent slices from domain b
            
        returns:
            dictionary with fake and reconstructed images
        """
        # forward cycle: a → b → a
        fake_B = self.G_A2B(slices_A)   # [b, 4, h, w]
        
        # for reconstruction, we need 3 slices in domain b
        # simplified: use fake_b repeated (in practice, translate adjacent slices)
        fake_B_3slice = fake_B.repeat(1, 3, 1, 1)  # [b, 12, h, w]
        rec_A = self.G_B2A(fake_B_3slice)
        
        # backward cycle: b → a → b
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
        """translate from domain a to b."""
        return self.G_A2B(slices)
    
    @torch.no_grad()
    def translate_B2A(self, slices: torch.Tensor) -> torch.Tensor:
        """translate from domain b to a."""
        return self.G_B2A(slices)
    
    def get_parameter_count(self) -> Dict[str, int]:
        """get parameter counts for each component."""
        return {
            'G_A2B': sum(p.numel() for p in self.G_A2B.parameters()),
            'G_B2A': sum(p.numel() for p in self.G_B2A.parameters()),
            'D_A': sum(p.numel() for p in self.D_A.parameters()),
            'D_B': sum(p.numel() for p in self.D_B.parameters()),
            'total': sum(p.numel() for p in self.parameters())
        }


# =============================================================================
# factory and testing
# =============================================================================

def create_model(config: Optional[SACycleGAN25DConfig] = None) -> SACycleGAN25D:
    """create and initialize the 2.5d sa-cyclegan model."""
    model = SACycleGAN25D(config)
    
    # print summary
    params = model.get_parameter_count()
    print("=" * 60)
    print("2.5d sa-cyclegan model summary")
    print("=" * 60)
    print(f"generator a→b: {params['G_A2B']:,} parameters")
    print(f"generator b→a: {params['G_B2A']:,} parameters")
    print(f"discriminator a: {params['D_A']:,} parameters")
    print(f"discriminator b: {params['D_B']:,} parameters")
    print(f"total: {params['total']:,} parameters ({params['total']/1e6:.2f}m)")
    print("=" * 60)
    
    return model


if __name__ == '__main__':
    # quick test
    config = SACycleGAN25DConfig()
    model = create_model(config)
    
    # test input: 3 slices x 4 modalities = 12 channels
    x = torch.randn(2, 12, 128, 128)
    
    # test generator
    out = model.G_A2B(x)
    print(f"input shape: {x.shape}")
    print(f"output shape: {out.shape}")  # should be [2, 4, 128, 128]
