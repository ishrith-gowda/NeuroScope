"""
Self-Attention CycleGAN (SA-CycleGAN) Architecture

Novel architecture combining CycleGAN with multi-scale self-attention for
brain MRI domain translation between BraTS and UPenn-GBM datasets.

Key innovations:
1. Multi-scale self-attention in generator bottleneck
2. CBAM attention in encoder/decoder paths
3. Spectral-normalized discriminator with attention
4. Modality-aware feature extraction

Reference: This is a novel contribution for NeurIPS-level publication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class SpectralNorm(nn.Module):
    """Spectral normalization for stabilized training."""
    
    def __init__(self, module: nn.Module, name: str = 'weight', n_power_iterations: int = 1):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        self._make_params()
    
    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = F.normalize(u.data, dim=0)
        v.data = F.normalize(v.data, dim=0)
        
        delattr(self.module, self.name)
        self.module.register_parameter(self.name + "_bar", nn.Parameter(w.data))
        self.module.register_buffer(self.name + "_u", u.data)
        self.module.register_buffer(self.name + "_v", v.data)
    
    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        
        height = w.data.shape[0]
        for _ in range(self.n_power_iterations):
            v.data = F.normalize(torch.mv(w.view(height, -1).data.t(), u.data), dim=0)
            u.data = F.normalize(torch.mv(w.view(height, -1).data, v.data), dim=0)
        
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))
    
    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class SelfAttention2D(nn.Module):
    """
    Self-attention mechanism for 2D feature maps.
    
    Computes attention between all spatial positions, allowing the model
    to capture long-range dependencies crucial for anatomical consistency.
    """
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        # Reduced channel dimension for efficiency
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # Learnable attention scaling
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Layer norm for stability
        self.norm = nn.InstanceNorm2d(in_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Attention-weighted output [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Compute query, key, value projections
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C']
        key = self.key_conv(x).view(B, -1, H * W)  # [B, C', HW]
        value = self.value_conv(x).view(B, -1, H * W)  # [B, C, HW]
        
        # Scaled dot-product attention
        attention = torch.bmm(query, key)  # [B, HW, HW]
        attention = F.softmax(attention / math.sqrt(C // self.reduction), dim=-1)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(B, C, H, W)
        
        # Residual connection with learnable scale
        out = self.gamma * self.norm(out) + x
        
        return out


class ChannelAttention(nn.Module):
    """Channel attention module (squeeze-and-excitation style)."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    """Convolutional Block Attention Module combining channel and spatial attention."""
    
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class ResidualBlockWithAttention(nn.Module):
    """Residual block with optional attention mechanism."""
    
    def __init__(
        self,
        channels: int,
        use_attention: bool = False,
        attention_type: str = 'cbam'
    ):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels)
        )
        
        self.use_attention = use_attention
        if use_attention:
            if attention_type == 'cbam':
                self.attention = CBAM(channels)
            elif attention_type == 'self':
                self.attention = SelfAttention2D(channels)
            else:
                self.attention = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_attention:
            out = self.attention(out)
        return out + x


class ModalityEncoder(nn.Module):
    """
    Modality-specific encoder that processes each MRI modality separately
    before combining them, preserving modality-specific features.
    """
    
    def __init__(self, out_channels: int = 16):
        super().__init__()
        
        # Separate encoder for each modality
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(1, out_channels, 3, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(4)  # T1, T1ce, T2, FLAIR
        ])
        
        # Fusion layer with attention
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels * 4, 1, bias=False),
            nn.InstanceNorm2d(out_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.fusion_attention = CBAM(out_channels * 4)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 4, H, W] (4 MRI modalities)
        Returns:
            Fused features [B, out_channels*4, H, W]
        """
        # Process each modality separately
        modality_features = []
        for i, encoder in enumerate(self.encoders):
            modality_features.append(encoder(x[:, i:i+1, :, :]))
        
        # Concatenate and fuse with attention
        fused = torch.cat(modality_features, dim=1)
        fused = self.fusion(fused)
        fused = self.fusion_attention(fused)
        
        return fused


class SAGenerator(nn.Module):
    """
    Self-Attention Generator (SA-Generator) for MRI domain translation.
    
    Architecture:
    1. Modality-aware encoder with separate processing per modality
    2. Multi-scale encoder with CBAM attention
    3. Self-attention bottleneck for long-range dependencies
    4. Decoder with skip connections and CBAM attention
    
    This architecture is specifically designed for multi-modal MRI translation,
    preserving anatomical structures and modality-specific information.
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        output_channels: int = 4,
        ngf: int = 64,
        n_residual_blocks: int = 9,
        use_modality_encoder: bool = True,
        attention_layers: List[int] = [3, 4, 5]  # Which residual blocks get self-attention
    ):
        super().__init__()
        
        self.use_modality_encoder = use_modality_encoder
        
        # Initial modality-aware encoding (optional)
        if use_modality_encoder:
            self.modality_encoder = ModalityEncoder(out_channels=16)
            first_layer_in = 64  # 16 * 4 modalities
        else:
            first_layer_in = input_channels
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(first_layer_in, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        # Encoder (downsampling with attention)
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
        for i in range(n_residual_blocks):
            use_self_attention = i in attention_layers
            self.bottleneck.append(
                ResidualBlockWithAttention(
                    ngf * 4,
                    use_attention=True,
                    attention_type='self' if use_self_attention else 'cbam'
                )
            )
        
        # Global self-attention in bottleneck
        self.global_attention = SelfAttention2D(ngf * 4, reduction=4)
        
        # Decoder (upsampling with attention)
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
        
        # Output layer
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_channels, 7),
            nn.Tanh()
        )
        
        # Skip connections fusion
        self.skip_fuse1 = nn.Conv2d(ngf * 4, ngf * 2, 1, bias=False)
        self.skip_fuse2 = nn.Conv2d(ngf * 2, ngf, 1, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SA-Generator.
        
        Args:
            x: Input MRI [B, 4, H, W]
        Returns:
            Translated MRI [B, 4, H, W]
        """
        # Modality-aware encoding
        if self.use_modality_encoder:
            x = self.modality_encoder(x)
        
        # Initial convolution
        x = self.initial(x)
        
        # Encoder with skip connections
        skips = [x]
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
        
        # Bottleneck with self-attention
        for block in self.bottleneck:
            x = block(x)
        
        # Global attention
        x = self.global_attention(x)
        
        # Decoder with skip connections
        x = self.decoder[0](x)
        # Add skip connection from encoder
        skip = self.skip_fuse1(skips[-1])
        x = x + F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x = self.decoder[1](x)
        skip = self.skip_fuse2(skips[-2])
        x = x + F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Output
        x = self.output(x)
        
        return x


class SADiscriminator(nn.Module):
    """
    Self-Attention PatchGAN Discriminator.
    
    Multi-scale discriminator with spectral normalization and self-attention
    for improved training stability and realistic outputs.
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        ndf: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        def get_norm(layer):
            if use_spectral_norm:
                return SpectralNorm(layer)
            return layer
        
        # Initial layer (no normalization)
        sequence = [
            get_norm(nn.Conv2d(input_channels, ndf, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Hidden layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            
            sequence.extend([
                get_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=2, padding=1, bias=False)),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # Self-attention layer
        sequence.append(SelfAttention2D(ndf * nf_mult))
        
        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        sequence.extend([
            get_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # Output layer (PatchGAN)
        sequence.append(
            get_norm(nn.Conv2d(ndf * nf_mult, 1, 4, stride=1, padding=1))
        )
        
        self.model = nn.Sequential(*sequence)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image [B, C, H, W]
        Returns:
            Patch-wise discrimination scores [B, 1, H', W']
        """
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for better handling of different frequency content.
    
    Processes images at multiple scales to capture both fine details and
    global structure.
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        ndf: int = 64,
        n_layers: int = 3,
        n_scales: int = 2
    ):
        super().__init__()
        
        self.n_scales = n_scales
        self.discriminators = nn.ModuleList([
            SADiscriminator(input_channels, ndf, n_layers)
            for _ in range(n_scales)
        ])
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Input image [B, C, H, W]
        Returns:
            List of discrimination scores at each scale
        """
        outputs = []
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            outputs.append(disc(x))
        return outputs


class SACycleGAN(nn.Module):
    """
    Complete SA-CycleGAN model for bidirectional domain translation.
    
    Combines:
    - Two SA-Generators (A->B and B->A)
    - Two Multi-scale Discriminators (for domain A and B)
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        ngf: int = 64,
        ndf: int = 64,
        n_residual_blocks: int = 9,
        n_discriminator_layers: int = 3,
        n_discriminator_scales: int = 2,
        use_modality_encoder: bool = True
    ):
        super().__init__()
        
        # Generators
        self.G_A2B = SAGenerator(
            input_channels=input_channels,
            output_channels=input_channels,
            ngf=ngf,
            n_residual_blocks=n_residual_blocks,
            use_modality_encoder=use_modality_encoder
        )
        
        self.G_B2A = SAGenerator(
            input_channels=input_channels,
            output_channels=input_channels,
            ngf=ngf,
            n_residual_blocks=n_residual_blocks,
            use_modality_encoder=use_modality_encoder
        )
        
        # Discriminators
        self.D_A = MultiScaleDiscriminator(
            input_channels=input_channels,
            ndf=ndf,
            n_layers=n_discriminator_layers,
            n_scales=n_discriminator_scales
        )
        
        self.D_B = MultiScaleDiscriminator(
            input_channels=input_channels,
            ndf=ndf,
            n_layers=n_discriminator_layers,
            n_scales=n_discriminator_scales
        )
        
    def forward(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            real_A: Real images from domain A [B, C, H, W]
            real_B: Real images from domain B [B, C, H, W]
            
        Returns:
            fake_B: Generated images in domain B
            fake_A: Generated images in domain A
            rec_A: Reconstructed images in domain A
            rec_B: Reconstructed images in domain B
        """
        # Forward cycle: A -> B -> A
        fake_B = self.G_A2B(real_A)
        rec_A = self.G_B2A(fake_B)
        
        # Backward cycle: B -> A -> B
        fake_A = self.G_B2A(real_B)
        rec_B = self.G_A2B(fake_A)
        
        return fake_B, fake_A, rec_A, rec_B
    
    @torch.no_grad()
    def translate_A2B(self, x: torch.Tensor) -> torch.Tensor:
        """Translate from domain A to domain B."""
        return self.G_A2B(x)
    
    @torch.no_grad()
    def translate_B2A(self, x: torch.Tensor) -> torch.Tensor:
        """Translate from domain B to domain A."""
        return self.G_B2A(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_sa_cyclegan(
    input_channels: int = 4,
    ngf: int = 64,
    ndf: int = 64,
    n_residual_blocks: int = 9
) -> SACycleGAN:
    """
    Factory function to create SA-CycleGAN model.
    
    Args:
        input_channels: Number of input channels (4 for multi-modal MRI)
        ngf: Number of generator filters
        ndf: Number of discriminator filters
        n_residual_blocks: Number of residual blocks in generator
        
    Returns:
        SACycleGAN model instance
    """
    model = SACycleGAN(
        input_channels=input_channels,
        ngf=ngf,
        ndf=ndf,
        n_residual_blocks=n_residual_blocks
    )
    
    # Print parameter counts
    g_params = count_parameters(model.G_A2B)
    d_params = count_parameters(model.D_A)
    total = count_parameters(model)
    
    print(f"SA-CycleGAN Architecture Summary:")
    print(f"  Generator parameters: {g_params:,}")
    print(f"  Discriminator parameters: {d_params:,}")
    print(f"  Total parameters: {total:,}")
    
    return model


if __name__ == '__main__':
    # Test the architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_sa_cyclegan()
    model = model.to(device)
    
    # Test forward pass
    batch_size = 2
    x_a = torch.randn(batch_size, 4, 256, 256).to(device)
    x_b = torch.randn(batch_size, 4, 256, 256).to(device)
    
    fake_b, fake_a, rec_a, rec_b = model(x_a, x_b)
    
    print(f"\nInput shape: {x_a.shape}")
    print(f"Output shape: {fake_b.shape}")
    print(f"Reconstruction shape: {rec_a.shape}")
    
    # Test discriminator
    d_out = model.D_A(x_a)
    print(f"Discriminator outputs: {[o.shape for o in d_out]}")
