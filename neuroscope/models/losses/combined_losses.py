"""
Loss Functions for 2.5D SA-CycleGAN.

Comprehensive loss functions for medical image translation:
- Adversarial (LSGAN)
- Cycle consistency
- Identity
- Perceptual (VGG-based)
- SSIM
- Gradient difference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math


class LSGANLoss(nn.Module):
    """Least Squares GAN Loss for improved training stability."""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(
        self,
        pred: torch.Tensor,
        target_is_real: bool
    ) -> torch.Tensor:
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.mse(pred, target)
    
    def discriminator_loss(
        self,
        real_pred: List[torch.Tensor],
        fake_pred: List[torch.Tensor]
    ) -> torch.Tensor:
        """Discriminator loss for multi-scale outputs."""
        loss = 0
        for real, fake in zip(real_pred, fake_pred):
            loss += self.forward(real, True) + self.forward(fake, False)
        return loss / len(real_pred)
    
    def generator_loss(self, fake_pred: List[torch.Tensor]) -> torch.Tensor:
        """Generator loss for multi-scale outputs."""
        loss = 0
        for fake in fake_pred:
            loss += self.forward(fake, True)
        return loss / len(fake_pred)


class CycleLoss(nn.Module):
    """Cycle consistency loss."""
    
    def __init__(self, lambda_cycle: float = 10.0):
        super().__init__()
        self.lambda_cycle = lambda_cycle
        
    def forward(
        self,
        real: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> torch.Tensor:
        return self.lambda_cycle * F.l1_loss(reconstructed, real)


class IdentityLoss(nn.Module):
    """Identity preservation loss."""
    
    def __init__(self, lambda_identity: float = 5.0):
        super().__init__()
        self.lambda_identity = lambda_identity
        
    def forward(
        self,
        real: torch.Tensor,
        same_domain_output: torch.Tensor
    ) -> torch.Tensor:
        return self.lambda_identity * F.l1_loss(same_domain_output, real)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss.
    
    Encourages structural preservation during translation.
    """
    
    def __init__(self, window_size: int = 11, lambda_ssim: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.lambda_ssim = lambda_ssim
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        
    def _create_window(self, channels: int, device: torch.device) -> torch.Tensor:
        """Create Gaussian window for SSIM computation."""
        sigma = 1.5
        gauss = torch.tensor([
            math.exp(-(x - self.window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(self.window_size)
        ], device=device)
        gauss = gauss / gauss.sum()
        
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d @ window_1d.t()
        window = window_2d.expand(channels, 1, self.window_size, self.window_size).contiguous()
        return window
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss (1 - SSIM)."""
        channels = x.size(1)
        window = self._create_window(channels, x.device)
        
        mu_x = F.conv2d(x, window, padding=self.window_size // 2, groups=channels)
        mu_y = F.conv2d(y, window, padding=self.window_size // 2, groups=channels)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(x * x, window, padding=self.window_size // 2, groups=channels) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, window, padding=self.window_size // 2, groups=channels) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=self.window_size // 2, groups=channels) - mu_xy

        # Clamp variances to prevent numerical instability (variance should never be negative)
        sigma_x_sq = torch.clamp(sigma_x_sq, min=0)
        sigma_y_sq = torch.clamp(sigma_y_sq, min=0)

        ssim = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / \
               ((mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2))
        
        return self.lambda_ssim * (1 - ssim.mean())


class GradientDifferenceLoss(nn.Module):
    """
    Gradient Difference Loss for edge preservation.
    
    Compares image gradients to preserve anatomical boundaries.
    """
    
    def __init__(self, lambda_grad: float = 1.0):
        super().__init__()
        self.lambda_grad = lambda_grad
        
        # Sobel filters
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def _compute_gradients(self, x: torch.Tensor) -> torch.Tensor:
        """Compute image gradients using Sobel filters."""
        grads = []
        for c in range(x.size(1)):
            gx = F.conv2d(x[:, c:c+1], self.sobel_x, padding=1)
            gy = F.conv2d(x[:, c:c+1], self.sobel_y, padding=1)
            grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
            grads.append(grad)
        return torch.cat(grads, dim=1)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        grad_x = self._compute_gradients(x)
        grad_y = self._compute_gradients(y)
        return self.lambda_grad * F.l1_loss(grad_x, grad_y)


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss.
    
    Uses pretrained VGG features for high-level similarity.
    Note: VGG expects 3-channel input, so we use first 3 modalities.
    """
    
    def __init__(self, lambda_perceptual: float = 1.0, layers: List[int] = [3, 8, 15]):
        super().__init__()
        self.lambda_perceptual = lambda_perceptual
        self.layers = layers
        
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        except:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True).features
        
        self.blocks = nn.ModuleList()
        prev = 0
        for layer in layers:
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev:layer]))
            prev = layer
        
        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False
            
        # Normalization for VGG
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input for VGG (expects 3 channels)."""
        # Use first 3 channels (T1, T1ce, T2) or repeat if single channel
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) > 3:
            x = x[:, :3]
        return (x - self.mean) / self.std
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._normalize(x)
        y = self._normalize(y)
        
        loss = 0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        
        return self.lambda_perceptual * loss / len(self.blocks)


class CombinedLoss(nn.Module):
    """
    Combined loss for SA-CycleGAN training.
    
    Aggregates all loss components with configurable weights.
    """
    
    def __init__(
        self,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
        lambda_ssim: float = 1.0,
        lambda_gradient: float = 1.0,
        lambda_perceptual: float = 0.0,  # Set to 0 by default (expensive)
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.gan_loss = LSGANLoss()
        self.cycle_loss = CycleLoss(lambda_cycle)
        self.identity_loss = IdentityLoss(lambda_identity)
        self.ssim_loss = SSIMLoss(lambda_ssim=lambda_ssim)
        self.gradient_loss = GradientDifferenceLoss(lambda_grad=lambda_gradient)
        
        self.use_perceptual = lambda_perceptual > 0
        if self.use_perceptual:
            self.perceptual_loss = PerceptualLoss(lambda_perceptual=lambda_perceptual)
            
    def to(self, device):
        """Move all losses to device."""
        super().to(device)
        return self


if __name__ == '__main__':
    # Test losses
    print("Testing loss functions...")
    
    x = torch.randn(2, 4, 128, 128)
    y = torch.randn(2, 4, 128, 128)
    
    losses = CombinedLoss()
    
    print(f"Cycle loss: {losses.cycle_loss(x, y).item():.4f}")
    print(f"SSIM loss: {losses.ssim_loss(x, y).item():.4f}")
    print(f"Gradient loss: {losses.gradient_loss(x, y).item():.4f}")
    
    print("\nloss functions test passed")
