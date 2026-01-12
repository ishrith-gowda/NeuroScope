"""
standard cyclegan baseline for 2.5d medical image translation

this is a baseline implementation WITHOUT attention mechanisms for comparison.
identical to sa-cyclegan-2.5d except:
- no self-attention modules
- no cbam modules
- standard residual blocks

this baseline proves whether attention mechanisms improve performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class BaselineCycleGAN25DConfig:
    """configuration for baseline 2.5d cyclegan (no attention)."""

    # 2.5d settings
    n_input_slices: int = 3
    n_modalities: int = 4

    # generator settings
    ngf: int = 64
    n_residual_blocks: int = 9

    # discriminator settings
    ndf: int = 64
    n_disc_layers: int = 3
    n_disc_scales: int = 2
    use_spectral_norm: bool = True

    @property
    def input_channels(self) -> int:
        return self.n_input_slices * self.n_modalities  # 12

    @property
    def output_channels(self) -> int:
        return self.n_modalities  # 4


# =============================================================================
# basic building blocks
# =============================================================================

class ResidualBlock(nn.Module):
    """standard residual block without attention."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return residual + out


class DownsampleBlock(nn.Module):
    """downsampling block for encoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.conv(x)))


class UpsampleBlock(nn.Module):
    """upsampling block for decoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.conv(x)))


# =============================================================================
# generator
# =============================================================================

class BaselineGenerator25D(nn.Module):
    """baseline 2.5d generator without attention mechanisms."""

    def __init__(self, config: BaselineCycleGAN25DConfig):
        super().__init__()
        self.config = config

        # initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(config.input_channels, config.ngf, 7, padding=3),
            nn.InstanceNorm2d(config.ngf),
            nn.ReLU(inplace=True)
        )

        # encoder (downsampling)
        self.down1 = DownsampleBlock(config.ngf, config.ngf * 2)
        self.down2 = DownsampleBlock(config.ngf * 2, config.ngf * 4)

        # residual blocks (bottleneck)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(config.ngf * 4)
            for _ in range(config.n_residual_blocks)
        ])

        # decoder (upsampling)
        self.up1 = UpsampleBlock(config.ngf * 4, config.ngf * 2)
        self.up2 = UpsampleBlock(config.ngf * 2, config.ngf)

        # output convolution
        self.output = nn.Sequential(
            nn.Conv2d(config.ngf, config.output_channels, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass.

        args:
            x: [B, 12, H, W] - 3 slices x 4 modalities

        returns:
            [B, 4, H, W] - center slice, 4 modalities
        """
        # initial conv
        x = self.initial(x)

        # encoder
        x = self.down1(x)
        x = self.down2(x)

        # residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # decoder
        x = self.up1(x)
        x = self.up2(x)

        # output
        x = self.output(x)

        return x


# =============================================================================
# discriminator
# =============================================================================

class ConvBlock(nn.Module):
    """convolutional block for discriminator."""

    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True,
                 use_spectral_norm: bool = False):
        super().__init__()

        conv = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)

        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)

        layers = [conv]

        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineDiscriminator(nn.Module):
    """baseline patchgan discriminator without attention."""

    def __init__(self, in_channels: int, ndf: int = 64, n_layers: int = 3,
                 use_spectral_norm: bool = True):
        super().__init__()

        layers = []

        # first layer (no normalization)
        layers.append(ConvBlock(in_channels, ndf, use_norm=False,
                               use_spectral_norm=use_spectral_norm))

        # intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(ConvBlock(ndf * nf_mult_prev, ndf * nf_mult,
                                   use_spectral_norm=use_spectral_norm))

        # final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        conv = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, padding=1)
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)

        layers.append(conv)
        layers.append(nn.InstanceNorm2d(ndf * nf_mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # output layer
        conv_out = nn.Conv2d(ndf * nf_mult, 1, 4, padding=1)
        if use_spectral_norm:
            conv_out = nn.utils.spectral_norm(conv_out)
        layers.append(conv_out)

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """multi-scale discriminator for better feature matching."""

    def __init__(self, in_channels: int, ndf: int = 64, n_layers: int = 3,
                 n_scales: int = 2, use_spectral_norm: bool = True):
        super().__init__()

        self.n_scales = n_scales

        # create discriminators for each scale
        self.discriminators = nn.ModuleList([
            BaselineDiscriminator(in_channels, ndf, n_layers, use_spectral_norm)
            for _ in range(n_scales)
        ])

        # downsampling for multi-scale
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> list:
        """returns list of predictions from each scale."""
        outputs = []

        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            outputs.append(disc(x))

        return outputs


# =============================================================================
# full baseline cyclegan model
# =============================================================================

class BaselineCycleGAN25D(nn.Module):
    """baseline 2.5d cyclegan without attention mechanisms."""

    def __init__(self, config: Optional[BaselineCycleGAN25DConfig] = None):
        super().__init__()

        if config is None:
            config = BaselineCycleGAN25DConfig()

        self.config = config

        # generators
        self.G_A2B = BaselineGenerator25D(config)
        self.G_B2A = BaselineGenerator25D(config)

        # discriminators
        self.D_A = MultiScaleDiscriminator(
            in_channels=config.output_channels,
            ndf=config.ndf,
            n_layers=config.n_disc_layers,
            n_scales=config.n_disc_scales,
            use_spectral_norm=config.use_spectral_norm
        )
        self.D_B = MultiScaleDiscriminator(
            in_channels=config.output_channels,
            ndf=config.ndf,
            n_layers=config.n_disc_layers,
            n_scales=config.n_disc_scales,
            use_spectral_norm=config.use_spectral_norm
        )

    def forward(self, slices_a: torch.Tensor, slices_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        training forward pass.

        args:
            slices_a: [B, 12, H, W] - 3 slices from domain a
            slices_b: [B, 12, H, W] - 3 slices from domain b

        returns:
            dict with fake and reconstructed images
        """
        # forward cycle: a → b → a
        fake_b = self.G_A2B(slices_a)
        fake_b_3slice = fake_b.repeat(1, 3, 1, 1)
        rec_a = self.G_B2A(fake_b_3slice)

        # backward cycle: b → a → b
        fake_a = self.G_B2A(slices_b)
        fake_a_3slice = fake_a.repeat(1, 3, 1, 1)
        rec_b = self.G_A2B(fake_a_3slice)

        return {
            'fake_b': fake_b,
            'fake_a': fake_a,
            'rec_a': rec_a,
            'rec_b': rec_b
        }

    @torch.no_grad()
    def translate_a2b(self, slices: torch.Tensor) -> torch.Tensor:
        """translate from domain a to b."""
        return self.G_A2B(slices)

    @torch.no_grad()
    def translate_b2a(self, slices: torch.Tensor) -> torch.Tensor:
        """translate from domain b to a."""
        return self.G_B2A(slices)

    def get_parameter_count(self) -> Dict[str, int]:
        """get parameter counts for each component."""
        return {
            'g_a2b': sum(p.numel() for p in self.G_A2B.parameters()),
            'g_b2a': sum(p.numel() for p in self.G_B2A.parameters()),
            'd_a': sum(p.numel() for p in self.D_A.parameters()),
            'd_b': sum(p.numel() for p in self.D_B.parameters()),
            'total': sum(p.numel() for p in self.parameters())
        }


# =============================================================================
# factory function
# =============================================================================

def create_baseline_model(config: Optional[BaselineCycleGAN25DConfig] = None) -> BaselineCycleGAN25D:
    """create baseline cyclegan model."""
    model = BaselineCycleGAN25D(config)

    params = model.get_parameter_count()
    print("=" * 60)
    print("baseline cyclegan 2.5d (no attention)")
    print("=" * 60)
    print(f"generator a→b: {params['g_a2b']:,} parameters")
    print(f"generator b→a: {params['g_b2a']:,} parameters")
    print(f"discriminator a: {params['d_a']:,} parameters")
    print(f"discriminator b: {params['d_b']:,} parameters")
    print(f"total: {params['total']:,} parameters ({params['total']/1e6:.2f}M)")
    print("=" * 60)

    return model


if __name__ == '__main__':
    # quick test
    config = BaselineCycleGAN25DConfig()
    model = create_baseline_model(config)

    # test input
    x = torch.randn(2, 12, 128, 128)

    # test generator
    out = model.G_A2B(x)
    print(f"\ninput shape: {x.shape}")
    print(f"output shape: {out.shape}")
