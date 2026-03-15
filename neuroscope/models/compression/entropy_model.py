"""
entropy models for learned image compression.

estimates the bitrate of quantized latent representations using learned
probability models. the estimated bitrate is used as a differentiable
rate term in the rate-distortion loss for joint optimization.

reference:
    balle et al., "variational image compression with a scale hyperprior",
    iclr 2018.
    balle et al., "end-to-end optimized image compression", iclr 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class FactorizedPrior(nn.Module):
    """
    factorized entropy model for latent compression.

    models p(z_hat) as a product of independent per-channel distributions,
    each parameterized by a flexible density model (piecewise linear cdf).
    the estimated bitrate is -log2(p(z_hat)) summed over all elements.

    this is the simplest entropy model from balle et al. (2017). for better
    rate-distortion performance, use the hyperprior model.
    """

    def __init__(self, num_channels: int, num_filters: int = 3):
        """
        args:
            num_channels: number of channels in the quantized latent
            num_filters: number of filters per layer in the density network
        """
        super().__init__()
        self.num_channels = num_channels

        # learnable parameters for cumulative density function
        # parameterized as a composition of softplus-activated linear layers
        self.matrices = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.factors = nn.ParameterList()

        channels = [1, num_filters, num_filters, num_filters, 1]
        for i in range(len(channels) - 1):
            self.matrices.append(
                nn.Parameter(torch.randn(num_channels, channels[i + 1], channels[i]))
            )
            self.biases.append(
                nn.Parameter(torch.randn(num_channels, channels[i + 1], 1))
            )
            if i < len(channels) - 2:
                self.factors.append(
                    nn.Parameter(torch.zeros(num_channels, channels[i + 1], 1))
                )

        self._initialize_parameters()

    def _initialize_parameters(self):
        """initialize parameters for stable training."""
        for matrix in self.matrices:
            nn.init.xavier_uniform_(matrix)
        for bias in self.biases:
            nn.init.zeros_(bias)

    def _logits_cumulative(self, x: torch.Tensor) -> torch.Tensor:
        """
        evaluate the log-cumulative density function.

        args:
            x: input tensor of shape (num_channels, 1, n)
        returns:
            log-cdf values of same shape
        """
        for i, (matrix, bias) in enumerate(zip(self.matrices, self.biases)):
            x = torch.bmm(F.softplus(matrix), x) + bias
            if i < len(self.factors):
                x = x + torch.tanh(x) * torch.tanh(self.factors[i])
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        estimate bitrate of quantized latent.

        args:
            x: quantized latent tensor of shape (b, c, h, w)
        returns:
            tuple of (total_bits, bits_per_element):
                total_bits: estimated total bits for the batch
                bits_per_element: average bits per spatial element
        """
        b, c, h, w = x.shape

        # reshape for per-channel processing: (c, 1, b*h*w)
        x_flat = x.permute(1, 0, 2, 3).reshape(c, 1, -1)

        # compute probability mass in quantization bin [x - 0.5, x + 0.5]
        upper = self._logits_cumulative(x_flat + 0.5)
        lower = self._logits_cumulative(x_flat - 0.5)

        # log-probability via log-sigmoid difference
        log_probs = torch.log(torch.sigmoid(upper) - torch.sigmoid(lower) + 1e-10)

        # sum over channels, reshape back
        log_probs = log_probs.reshape(c, b, h, w).permute(1, 0, 2, 3)

        # total bits = -sum(log2(p))
        total_bits = -log_probs.sum() / np.log(2)
        bits_per_element = total_bits / (b * h * w)

        return total_bits, bits_per_element


class HyperpriorModel(nn.Module):
    """
    scale hyperprior entropy model.

    uses a secondary latent (hyperprior) to estimate per-element scale
    parameters for a gaussian entropy model of the primary latent. this
    provides spatially adaptive rate allocation -- more bits for complex
    regions, fewer for smooth regions.

    reference:
        balle et al., "variational image compression with a scale
        hyperprior", iclr 2018.
    """

    def __init__(self, num_channels: int, num_hyper_channels: int = 128):
        """
        args:
            num_channels: channels in the primary latent (from bottleneck)
            num_hyper_channels: channels in the hyperprior latent
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_hyper_channels = num_hyper_channels

        # hyper-encoder: primary latent -> hyperprior latent
        self.hyper_encoder = nn.Sequential(
            nn.Conv2d(num_channels, num_hyper_channels, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_hyper_channels, num_hyper_channels, 3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_hyper_channels, num_hyper_channels, 3, stride=2, padding=1),
        )

        # hyper-decoder: hyperprior latent -> scale parameters
        self.hyper_decoder = nn.Sequential(
            nn.ConvTranspose2d(num_hyper_channels, num_hyper_channels, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(num_hyper_channels, num_hyper_channels, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_hyper_channels, num_channels, 3, padding=1),
            nn.Softplus(),  # scale must be positive
        )

        # factorized prior for the hyperprior latent itself
        self.hyper_prior = FactorizedPrior(num_hyper_channels)

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        estimate bitrate of quantized latent using hyperprior.

        args:
            y: continuous latent (before quantization)
            y_hat: quantized latent
        returns:
            tuple of (total_bits, bits_per_element)
        """
        b, c, h, w = y_hat.shape

        # encode hyperprior
        z = self.hyper_encoder(y.detach() if not self.training else y)

        # quantize hyperprior
        if self.training:
            z_hat = z + torch.empty_like(z).uniform_(-0.5, 0.5)
        else:
            z_hat = torch.round(z)

        # decode scale parameters from hyperprior
        sigma = self.hyper_decoder(z_hat)

        # ensure sigma matches y_hat spatial dimensions
        if sigma.shape[2:] != y_hat.shape[2:]:
            sigma = F.interpolate(
                sigma, size=y_hat.shape[2:], mode="bilinear", align_corners=False
            )

        # gaussian likelihood for primary latent
        # p(y_hat | sigma) = N(0, sigma^2) integrated over quantization bin
        sigma = sigma.clamp(min=0.01)
        upper = (y_hat + 0.5) / sigma
        lower = (y_hat - 0.5) / sigma

        # use normal cdf difference for probability mass
        from torch.distributions import Normal

        normal = Normal(0, 1)
        probs = normal.cdf(upper) - normal.cdf(lower)
        probs = probs.clamp(min=1e-10)

        bits_y = -torch.log2(probs).sum()

        # bits for hyperprior
        bits_z, _ = self.hyper_prior(z_hat)

        total_bits = bits_y + bits_z
        bits_per_element = total_bits / (b * h * w)

        return total_bits, bits_per_element


if __name__ == "__main__":
    print("testing entropy models...")

    # test factorized prior
    fp = FactorizedPrior(num_channels=256)
    z = torch.randn(2, 256, 8, 8)
    total_bits, bpe = fp(z)
    print(f"factorized prior: total_bits={total_bits.item():.1f}, bpe={bpe.item():.3f}")

    # test hyperprior
    hp = HyperpriorModel(num_channels=256, num_hyper_channels=128)
    y = torch.randn(2, 256, 32, 32)
    y_hat = torch.round(y)
    total_bits, bpe = hp(y, y_hat)
    print(f"hyperprior: total_bits={total_bits.item():.1f}, bpe={bpe.item():.3f}")

    print("all tests passed")
