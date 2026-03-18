"""
multi-domain sa-cyclegan-2.5d with adaptive instance normalization.

extends the 2-domain sa-cyclegan-2.5d to handle n>2 scanner domains
using a single generator conditioned on target domain via adain.
this avoids the o(n^2) scaling of pairwise cyclegan training.

key architecture changes:
    - instancenorm replaced with adaptive instance normalization (adain)
    - domain embedding layer maps domain id to affine parameters
    - single generator handles all n domains via domain conditioning
    - discriminator includes domain classification head (stargan-style)

reference:
    choi et al., "stargan: unified generative adversarial networks for
    multi-domain image-to-image translation", cvpr 2018.
    huang & belongie, "arbitrary style transfer in real-time with
    adaptive instance normalization", iccv 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math

from neuroscope.models.architectures.sa_cyclegan_25d import (
    SACycleGAN25DConfig,
    SelfAttention2D,
    ChannelAttention,
    SpatialAttention,
    CBAM,
)


@dataclass
class MultiDomainConfig(SACycleGAN25DConfig):
    """configuration for multi-domain sa-cyclegan-2.5d."""

    n_domains: int = 4
    domain_embed_dim: int = 64
    style_dim: int = 256


class AdaIN(nn.Module):
    """
    adaptive instance normalization.

    normalizes features to zero mean and unit variance, then applies
    domain-specific affine transformation (gamma, beta) computed from
    the domain embedding.

    reference:
        huang & belongie, "arbitrary style transfer in real-time with
        adaptive instance normalization", iccv 2017.
    """

    def __init__(self, num_features: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.style_fc = nn.Linear(style_dim, num_features * 2)

        # initialize affine transform to identity
        self.style_fc.bias.data[:num_features] = 1.0  # gamma = 1
        self.style_fc.bias.data[num_features:] = 0.0  # beta = 0

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: feature tensor [b, c, h, w]
            style: domain style vector [b, style_dim]
        returns:
            adain-normalized features [b, c, h, w]
        """
        # compute affine parameters from style
        affine = self.style_fc(style)
        gamma, beta = affine.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # normalize and apply domain-specific affine
        x = self.norm(x)
        return gamma * x + beta


class AdaINResidualBlock(nn.Module):
    """residual block with adaptive instance normalization."""

    def __init__(
        self,
        channels: int,
        style_dim: int,
        use_attention: bool = False,
        attention_type: str = "cbam",
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
        )
        self.adain1 = AdaIN(channels, style_dim)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
        )
        self.adain2 = AdaIN(channels, style_dim)

        self.attention = None
        if use_attention:
            if attention_type == "self":
                self.attention = SelfAttention2D(channels)
            elif attention_type == "cbam":
                self.attention = CBAM(channels)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: feature tensor [b, c, h, w]
            style: domain style vector [b, style_dim]
        returns:
            output features [b, c, h, w]
        """
        out = self.conv1(x)
        out = self.adain1(out, style)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.adain2(out, style)

        if self.attention is not None:
            out = self.attention(out)

        return out + x


class DomainEncoder(nn.Module):
    """
    domain embedding encoder.

    maps discrete domain id to a continuous style vector via learnable
    embedding + mlp. the style vector conditions the generator's adain
    layers for domain-specific normalization.
    """

    def __init__(self, n_domains: int, embed_dim: int = 64, style_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(n_domains, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, style_dim),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim, style_dim),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim, style_dim),
        )

    def forward(self, domain_id: torch.Tensor) -> torch.Tensor:
        """
        args:
            domain_id: integer domain labels [b]
        returns:
            style vector [b, style_dim]
        """
        emb = self.embedding(domain_id)
        return self.mlp(emb)


class MultiDomainSAGenerator25D(nn.Module):
    """
    multi-domain sa-generator-2.5d with adain conditioning.

    single generator that handles all n domains. the target domain is
    specified via domain_id, which is converted to a style vector and
    injected into the bottleneck via adain layers.

    architecture:
        encoder (shared) -> adain bottleneck (domain-conditioned) -> decoder (shared)

    the encoder and decoder are shared across all domains (they extract
    and reconstruct domain-invariant spatial features). only the bottleneck
    normalization is domain-specific.
    """

    def __init__(self, config: MultiDomainConfig):
        super().__init__()
        self.config = config
        ngf = config.ngf

        # domain encoder
        self.domain_encoder = DomainEncoder(
            n_domains=config.n_domains,
            embed_dim=config.domain_embed_dim,
            style_dim=config.style_dim,
        )

        # 2.5d slice encoder (shared, same as base)
        from neuroscope.models.architectures.sa_cyclegan_25d import SliceEncoder25D

        self.encoder_initial = SliceEncoder25D(
            n_slices=config.n_input_slices,
            n_modalities=config.n_modalities,
            ngf=ngf,
        )

        # downsampling encoder (shared)
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

        # adain bottleneck (domain-conditioned)
        self.bottleneck = nn.ModuleList()
        for i in range(config.n_residual_blocks):
            use_self_attn = i in config.attention_layers
            self.bottleneck.append(
                AdaINResidualBlock(
                    ngf * 4,
                    style_dim=config.style_dim,
                    use_attention=True,
                    attention_type="self" if use_self_attn else "cbam",
                )
            )

        self.global_attention = SelfAttention2D(ngf * 4, reduction=4)

        # upsampling decoder (shared)
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

    def forward(
        self, x: torch.Tensor, target_domain: torch.Tensor
    ) -> torch.Tensor:
        """
        args:
            x: input tensor [b, 12, h, w]
            target_domain: target domain ids [b] (integer labels)
        returns:
            harmonized image [b, 4, h, w] in target domain
        """
        # get domain style vector
        style = self.domain_encoder(target_domain)

        # encode
        x = self.encoder_initial(x)
        skips = [x]
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)

        # domain-conditioned bottleneck
        for block in self.bottleneck:
            x = block(x, style)
        x = self.global_attention(x)

        # decode
        for i, dec in enumerate(self.decoder):
            x = dec(x)
            skip = self.skip_fuse[i](skips[-(i + 1)])
            skip = F.interpolate(
                skip, size=x.shape[2:], mode="bilinear", align_corners=False
            )
            x = x + skip

        return self.output(x)


class MultiDomainDiscriminator(nn.Module):
    """
    multi-domain discriminator with domain classification head.

    outputs both a real/fake score (patchgan) and a domain classification
    vector (stargan-style). the domain classification provides an auxiliary
    training signal for the generator's domain conditioning.
    """

    def __init__(
        self,
        in_channels: int,
        n_domains: int,
        ndf: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        def get_conv(in_c, out_c, stride=2, bias=True):
            conv = nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1, bias=bias)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            return conv

        # shared feature extractor
        layers = [
            get_conv(in_channels, ndf),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf = ndf
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers.extend([
                get_conv(nf_prev, nf, bias=False),
                nn.InstanceNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        layers.append(SelfAttention2D(nf))

        nf_prev = nf
        nf = min(nf * 2, 512)
        layers.extend([
            get_conv(nf_prev, nf, stride=1, bias=False),
            nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
        ])

        self.features = nn.Sequential(*layers)

        # real/fake classification head (patchgan)
        self.adv_head = nn.Sequential(
            get_conv(nf, 1, stride=1),
        )

        # domain classification head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(nf, n_domains),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        args:
            x: input image [b, c, h, w]
        returns:
            (adv_out, cls_out):
                adv_out: real/fake patch scores [b, 1, h', w']
                cls_out: domain classification logits [b, n_domains]
        """
        feat = self.features(x)
        adv_out = self.adv_head(feat)
        cls_out = self.cls_head(feat)
        return adv_out, cls_out


class MultiDomainSACycleGAN25D(nn.Module):
    """
    complete multi-domain sa-cyclegan-2.5d.

    single generator + single discriminator for n-domain harmonization.
    training uses:
        - adversarial loss (real/fake)
        - domain classification loss (generator fools discriminator about domain)
        - cycle consistency loss (translate a->b->a)
        - identity loss (translate a->a should be identity)
    """

    def __init__(self, config: Optional[MultiDomainConfig] = None):
        super().__init__()
        self.config = config or MultiDomainConfig()

        self.generator = MultiDomainSAGenerator25D(self.config)
        self.discriminator = MultiDomainDiscriminator(
            in_channels=self.config.output_channels,
            n_domains=self.config.n_domains,
            ndf=self.config.ndf,
            n_layers=self.config.n_disc_layers,
            use_spectral_norm=self.config.use_spectral_norm,
        )

    def forward(
        self,
        x: torch.Tensor,
        source_domain: torch.Tensor,
        target_domain: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        training forward pass.

        args:
            x: input slices [b, 12, h, w]
            source_domain: source domain ids [b]
            target_domain: target domain ids [b]
        returns:
            dict with fake images, reconstructions, and discriminator outputs
        """
        # forward translation: source -> target
        fake = self.generator(x, target_domain)

        # cycle reconstruction: target -> source
        fake_3slice = (
            fake.unsqueeze(2)
            .repeat(1, 1, 3, 1, 1)
            .view(fake.size(0), -1, fake.size(2), fake.size(3))
        )
        rec = self.generator(fake_3slice, source_domain)

        # identity: source -> source (should be identity)
        idt = self.generator(x, source_domain)

        return {
            "fake": fake,
            "rec": rec,
            "idt": idt,
        }

    def get_parameter_count(self) -> Dict[str, int]:
        return {
            "generator": sum(p.numel() for p in self.generator.parameters()),
            "discriminator": sum(p.numel() for p in self.discriminator.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }


if __name__ == "__main__":
    print("testing multi-domain sa-cyclegan-2.5d...")

    config = MultiDomainConfig(n_domains=4)
    model = MultiDomainSACycleGAN25D(config)

    params = model.get_parameter_count()
    print(f"model parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:,} ({v/1e6:.2f}m)")

    # test forward
    x = torch.randn(2, 12, 128, 128)
    src = torch.tensor([0, 1])
    tgt = torch.tensor([2, 3])

    results = model(x, src, tgt)
    print(f"\nforward pass:")
    for k, v in results.items():
        print(f"  {k}: {v.shape}")

    # test discriminator
    adv, cls = model.discriminator(results["fake"])
    print(f"\ndiscriminator:")
    print(f"  adv: {adv.shape}")
    print(f"  cls: {cls.shape}")

    print("\nall tests passed")
