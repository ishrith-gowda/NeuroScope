"""
quantization modules for neural compression.

implements differentiable quantization for training compressed representations
in the cyclegan bottleneck. during training, uses either additive uniform noise
or straight-through estimator (ste) to approximate discrete quantization.
at inference, uses actual rounding.

reference:
    balle et al., "variational image compression with a scale hyperprior",
    iclr 2018.
"""

import torch
import torch.nn as nn


class UniformQuantize(nn.Module):
    """
    uniform scalar quantization with straight-through estimator (ste).

    during training: adds uniform noise U(-0.5, 0.5) as a differentiable
    proxy for rounding. during inference: rounds to nearest integer.

    this is the standard approach from balle et al. (2018) for end-to-end
    trained compression.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: continuous latent tensor of any shape
        returns:
            quantized tensor (same shape)
        """
        if self.training:
            # additive uniform noise as differentiable proxy for rounding
            noise = torch.empty_like(x).uniform_(-0.5, 0.5)
            return x + noise
        else:
            # actual rounding at inference
            return torch.round(x)


class NoiseQuantize(nn.Module):
    """
    noise-based quantization with learnable scale.

    extends uniform quantization with a learnable scale parameter that
    controls the quantization step size. this provides an additional
    degree of freedom for the rate-distortion tradeoff.
    """

    def __init__(self, num_channels: int, init_scale: float = 1.0):
        """
        args:
            num_channels: number of channels in the latent representation
            init_scale: initial quantization step size
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.full((1, num_channels, 1, 1), init_scale)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: continuous latent tensor of shape (b, c, h, w)
        returns:
            quantized tensor (same shape)
        """
        # scale to quantization grid
        scale = self.scale.abs().clamp(min=0.01)
        x_scaled = x / scale

        if self.training:
            noise = torch.empty_like(x_scaled).uniform_(-0.5, 0.5)
            x_quantized = x_scaled + noise
        else:
            x_quantized = torch.round(x_scaled)

        # scale back
        return x_quantized * scale


class StraightThroughQuantize(nn.Module):
    """
    straight-through estimator (ste) quantization.

    applies rounding in the forward pass but passes gradients through
    unchanged in the backward pass. simpler than noise-based approaches
    but can have higher gradient variance.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # ste: round in forward, identity gradient in backward
            return x + (torch.round(x) - x).detach()
        else:
            return torch.round(x)
