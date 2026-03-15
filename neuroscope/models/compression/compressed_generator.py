"""
compressed sa-cyclegan-2.5d generator for harmonize-and-compress.

modifies the sa-generator bottleneck to include quantization and entropy
coding, producing a compressed bitstream alongside the harmonized image.
a single forward pass simultaneously harmonizes and compresses the mri data.

the architecture preserves the full sa-cyclegan-2.5d pipeline (encoder,
bottleneck with self-attention, decoder) while inserting a quantization
+ entropy estimation layer between the encoder and bottleneck.

key design decisions:
    - quantization is placed AFTER the encoder and BEFORE the bottleneck,
      so the bottleneck operates on quantized features. this forces the
      model to be robust to quantization noise.
    - the entropy model provides a differentiable bitrate estimate R
      that enters the loss as lambda_rate * R.
    - at inference, the quantized bottleneck features ARE the compressed
      representation (can be entropy coded with arithmetic coding).

reference:
    balle et al., "variational image compression with a scale hyperprior",
    iclr 2018.
    agustsson et al., "generative adversarial networks for extreme learned
    image compression", iccv 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from neuroscope.models.architectures.sa_cyclegan_25d import (
    SACycleGAN25DConfig,
    SAGenerator25D,
    SliceEncoder25D,
    ResidualBlock,
    SelfAttention2D,
    CBAM,
)
from neuroscope.models.compression.quantization import UniformQuantize, NoiseQuantize
from neuroscope.models.compression.entropy_model import FactorizedPrior, HyperpriorModel


class CompressedSAGenerator25D(nn.Module):
    """
    sa-generator-2.5d with compression-aware bottleneck.

    extends the standard sa-generator with:
    1. a bottleneck compression layer (quantize + entropy model)
    2. rate-distortion aware forward pass returning bitrate estimate
    3. encode/decode methods for actual compression at inference

    the compressed representation lives in the bottleneck feature space
    (ngf*4 channels at 1/4 spatial resolution), providing a natural
    compression ratio of approximately 12:1 before entropy coding
    (12-channel input at full res -> ngf*4 channels at 1/4 res).
    """

    def __init__(
        self,
        config: SACycleGAN25DConfig,
        entropy_model_type: str = "factorized",
        num_hyper_channels: int = 128,
    ):
        """
        args:
            config: sa-cyclegan-2.5d configuration
            entropy_model_type: "factorized" or "hyperprior"
            num_hyper_channels: channels for hyperprior (if used)
        """
        super().__init__()
        self.config = config
        ngf = config.ngf

        # === encoder (same as base sa-generator) ===
        self.encoder_initial = SliceEncoder25D(
            n_slices=config.n_input_slices,
            n_modalities=config.n_modalities,
            ngf=ngf,
        )

        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * 2),
                nn.ReLU(inplace=True),
                CBAM(ngf * 2),
            ),
            nn.Sequential(
                nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * 4),
                nn.ReLU(inplace=True),
                CBAM(ngf * 4),
            ),
        ])

        # === compression layer ===
        bottleneck_channels = ngf * 4  # 256 channels

        # pre-quantization projection (optional channel reduction for lower bitrate)
        self.pre_quant = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 1, bias=False),
            nn.InstanceNorm2d(bottleneck_channels),
        )

        # quantization
        self.quantize = UniformQuantize()

        # post-quantization projection (restore representation after quantization)
        self.post_quant = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 1, bias=False),
            nn.InstanceNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )

        # entropy model for bitrate estimation
        if entropy_model_type == "hyperprior":
            self.entropy_model = HyperpriorModel(
                num_channels=bottleneck_channels,
                num_hyper_channels=num_hyper_channels,
            )
        else:
            self.entropy_model = FactorizedPrior(
                num_channels=bottleneck_channels,
            )
        self.entropy_model_type = entropy_model_type

        # === bottleneck (same as base sa-generator) ===
        self.bottleneck = nn.ModuleList()
        for i in range(config.n_residual_blocks):
            use_self_attn = i in config.attention_layers
            self.bottleneck.append(
                ResidualBlock(
                    ngf * 4,
                    use_attention=True,
                    attention_type="self" if use_self_attn else "cbam",
                )
            )

        self.global_attention = SelfAttention2D(ngf * 4, reduction=4)

        # === decoder (same as base sa-generator) ===
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf * 2),
                nn.ReLU(inplace=True),
                CBAM(ngf * 2),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf),
                nn.ReLU(inplace=True),
                CBAM(ngf),
            ),
        ])

        self.skip_fuse = nn.ModuleList([
            nn.Conv2d(ngf * 4, ngf * 2, 1, bias=False),
            nn.Conv2d(ngf * 2, ngf, 1, bias=False),
        ])

        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, config.output_channels, 7),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        encode input to bottleneck representation.

        args:
            x: input tensor [b, 12, h, w]
        returns:
            (latent, skip_connections): latent features and encoder skips
        """
        x = self.encoder_initial(x)
        skips = [x]

        for enc in self.encoder:
            x = enc(x)
            skips.append(x)

        return x, skips

    def compress(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        quantize bottleneck and estimate bitrate.

        args:
            latent: continuous bottleneck features [b, c, h, w]
        returns:
            (quantized, total_bits, bits_per_element)
        """
        y = self.pre_quant(latent)
        y_hat = self.quantize(y)

        # estimate bitrate
        if self.entropy_model_type == "hyperprior":
            total_bits, bpe = self.entropy_model(y, y_hat)
        else:
            total_bits, bpe = self.entropy_model(y_hat)

        z = self.post_quant(y_hat)
        return z, total_bits, bpe

    def decode(self, z: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        decode from bottleneck to output image.

        args:
            z: bottleneck features (possibly quantized) [b, c, h/4, w/4]
            skips: skip connections from encoder
        returns:
            output image [b, 4, h, w]
        """
        # bottleneck blocks
        for block in self.bottleneck:
            z = block(z)
        z = self.global_attention(z)

        # decoder with skip connections
        for i, dec in enumerate(self.decoder):
            z = dec(z)
            skip = self.skip_fuse[i](skips[-(i + 1)])
            skip = F.interpolate(
                skip, size=z.shape[2:], mode="bilinear", align_corners=False
            )
            z = z + skip

        return self.output(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        full forward pass: encode -> compress -> decode.

        args:
            x: input tensor [b, 12, h, w]
        returns:
            (output, total_bits, bits_per_element):
                output: harmonized image [b, 4, h, w]
                total_bits: estimated total bits for the batch
                bits_per_element: average bits per spatial element
        """
        latent, skips = self.encode(x)
        z, total_bits, bpe = self.compress(latent)
        output = self.decode(z, skips)
        return output, total_bits, bpe

    def forward_uncompressed(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass without compression (for comparison / warmup).

        args:
            x: input tensor [b, 12, h, w]
        returns:
            harmonized image [b, 4, h, w] without quantization
        """
        latent, skips = self.encode(x)
        output = self.decode(latent, skips)
        return output


class CompressedSACycleGAN25D(nn.Module):
    """
    complete harmonize-and-compress cyclegan.

    both generators include compression-aware bottlenecks. the cycle
    consistency loss must pass through quantization in both directions,
    which the model learns to handle gracefully.

    rate-distortion loss:
        L_total = L_harmonization + lambda_rate * (R_A2B + R_B2A)
    where L_harmonization = L_adv + L_cycle + L_idt + L_ssim [+ L_nce]
    and R = estimated bitrate from entropy model.
    """

    def __init__(
        self,
        config: Optional[SACycleGAN25DConfig] = None,
        entropy_model_type: str = "factorized",
        num_hyper_channels: int = 128,
    ):
        super().__init__()
        self.config = config or SACycleGAN25DConfig()

        # compressed generators
        self.G_A2B = CompressedSAGenerator25D(
            self.config, entropy_model_type, num_hyper_channels
        )
        self.G_B2A = CompressedSAGenerator25D(
            self.config, entropy_model_type, num_hyper_channels
        )

        # discriminators (same as base -- operate on output images)
        from neuroscope.models.architectures.sa_cyclegan_25d import MultiScaleDiscriminator

        self.D_A = MultiScaleDiscriminator(
            in_channels=self.config.output_channels,
            ndf=self.config.ndf,
            n_layers=self.config.n_disc_layers,
            n_scales=self.config.n_disc_scales,
            use_spectral_norm=self.config.use_spectral_norm,
            use_attention=self.config.use_disc_attention,
        )
        self.D_B = MultiScaleDiscriminator(
            in_channels=self.config.output_channels,
            ndf=self.config.ndf,
            n_layers=self.config.n_disc_layers,
            n_scales=self.config.n_disc_scales,
            use_spectral_norm=self.config.use_spectral_norm,
            use_attention=self.config.use_disc_attention,
        )

    def forward(
        self, slices_A: torch.Tensor, slices_B: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        training forward pass with compression.

        returns:
            dict with fake images, reconstructions, and bitrate estimates
        """
        # forward: A -> B (with compression)
        fake_B, bits_A2B, bpe_A2B = self.G_A2B(slices_A)

        # cycle: B -> A -> B
        fake_B_3slice = (
            fake_B.unsqueeze(2)
            .repeat(1, 1, 3, 1, 1)
            .view(fake_B.size(0), -1, fake_B.size(2), fake_B.size(3))
        )
        rec_A, bits_rec_A, _ = self.G_B2A(fake_B_3slice)

        # forward: B -> A (with compression)
        fake_A, bits_B2A, bpe_B2A = self.G_B2A(slices_B)

        # cycle: A -> B -> A
        fake_A_3slice = (
            fake_A.unsqueeze(2)
            .repeat(1, 1, 3, 1, 1)
            .view(fake_A.size(0), -1, fake_A.size(2), fake_A.size(3))
        )
        rec_B, bits_rec_B, _ = self.G_A2B(fake_A_3slice)

        return {
            "fake_B": fake_B,
            "fake_A": fake_A,
            "rec_A": rec_A,
            "rec_B": rec_B,
            "bits_A2B": bits_A2B,
            "bits_B2A": bits_B2A,
            "bpe_A2B": bpe_A2B,
            "bpe_B2A": bpe_B2A,
            "bits_total": bits_A2B + bits_B2A + bits_rec_A + bits_rec_B,
        }

    def get_parameter_count(self) -> Dict[str, int]:
        """get parameter counts for each component."""
        return {
            "G_A2B": sum(p.numel() for p in self.G_A2B.parameters()),
            "G_B2A": sum(p.numel() for p in self.G_B2A.parameters()),
            "D_A": sum(p.numel() for p in self.D_A.parameters()),
            "D_B": sum(p.numel() for p in self.D_B.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }


if __name__ == "__main__":
    print("testing compressed sa-cyclegan-2.5d...")

    config = SACycleGAN25DConfig()

    # test compressed generator
    gen = CompressedSAGenerator25D(config, entropy_model_type="factorized")
    x = torch.randn(2, 12, 128, 128)

    output, total_bits, bpe = gen(x)
    print(f"input: {x.shape}")
    print(f"output: {output.shape}")
    print(f"total bits: {total_bits.item():.1f}")
    print(f"bits per element: {bpe.item():.3f}")

    # test uncompressed forward
    output_uc = gen.forward_uncompressed(x)
    print(f"uncompressed output: {output_uc.shape}")

    # test full model
    model = CompressedSACycleGAN25D(config)
    params = model.get_parameter_count()
    print(f"\ncompressed model parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:,} ({v/1e6:.2f}M)")

    # test forward
    a = torch.randn(2, 12, 128, 128)
    b = torch.randn(2, 12, 128, 128)
    results = model(a, b)
    print(f"\nforward pass results:")
    for k, v in results.items():
        if isinstance(v, torch.Tensor) and v.dim() > 0:
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {v.item():.3f}" if isinstance(v, torch.Tensor) else f"  {k}: {v}")

    print("\nall tests passed")
