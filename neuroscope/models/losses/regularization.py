"""
Regularization Losses for training stability.

This module provides regularization losses to improve
training stability and prevent mode collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math


class GradientPenalty(nn.Module):
    """
    Gradient Penalty for WGAN-GP training.
    
    Enforces Lipschitz constraint on discriminator.
    
    Args:
        weight: Weight for gradient penalty
        target_norm: Target norm for gradients (default 1.0)
    """
    
    def __init__(self, weight: float = 10.0, target_norm: float = 1.0):
        super().__init__()
        self.weight = weight
        self.target_norm = target_norm
        
    def forward(
        self,
        discriminator: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient penalty.
        
        Args:
            discriminator: Discriminator network
            real: Real samples
            fake: Fake samples
            
        Returns:
            Gradient penalty loss
        """
        batch_size = real.size(0)
        device = real.device
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)
        
        # Discriminator output
        d_interpolated = discriminator(interpolated)
        
        if isinstance(d_interpolated, (list, tuple)):
            d_interpolated = d_interpolated[0]
            
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient norm
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        
        # Gradient penalty
        penalty = ((gradient_norm - self.target_norm) ** 2).mean()
        
        return self.weight * penalty


class SpectralRegularization(nn.Module):
    """
    Spectral Regularization for discriminator.
    
    Applies soft spectral normalization as a loss term.
    
    Args:
        weight: Regularization weight
        n_power_iterations: Number of power iterations
    """
    
    def __init__(self, weight: float = 1.0, n_power_iterations: int = 1):
        super().__init__()
        self.weight = weight
        self.n_power_iterations = n_power_iterations
        
    def forward(self, model: nn.Module) -> torch.Tensor:
        """Compute spectral regularization loss."""
        reg_loss = 0.0
        count = 0
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight
                
                # Compute spectral norm
                with torch.no_grad():
                    weight_mat = weight.view(weight.size(0), -1)
                    
                    # Power iteration
                    u = torch.randn(1, weight_mat.size(0), device=weight.device)
                    u = F.normalize(u, dim=1)
                    
                    for _ in range(self.n_power_iterations):
                        v = F.normalize(torch.mm(u, weight_mat), dim=1)
                        u = F.normalize(torch.mm(v, weight_mat.t()), dim=1)
                        
                    sigma = torch.mm(torch.mm(u, weight_mat), v.t())
                    
                # Regularize towards unit spectral norm
                reg_loss += (sigma - 1.0) ** 2
                count += 1
                
        return self.weight * reg_loss / max(count, 1)


class R1Regularization(nn.Module):
    """
    R1 Gradient Regularization.
    
    Zero-centered gradient penalty for real samples only.
    Used in StyleGAN-style training.
    
    Args:
        weight: Regularization weight
    """
    
    def __init__(self, weight: float = 10.0):
        super().__init__()
        self.weight = weight
        
    def forward(
        self,
        discriminator: nn.Module,
        real: torch.Tensor
    ) -> torch.Tensor:
        """Compute R1 regularization."""
        real = real.requires_grad_(True)
        d_real = discriminator(real)
        
        if isinstance(d_real, (list, tuple)):
            d_real = d_real[0]
            
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_real.sum(),
            inputs=real,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Squared gradient norm
        r1_penalty = gradients.view(gradients.size(0), -1).pow(2).sum(dim=1).mean()
        
        return self.weight * r1_penalty


class R2Regularization(nn.Module):
    """
    R2 Gradient Regularization.
    
    Zero-centered gradient penalty for fake samples only.
    
    Args:
        weight: Regularization weight
    """
    
    def __init__(self, weight: float = 10.0):
        super().__init__()
        self.weight = weight
        
    def forward(
        self,
        discriminator: nn.Module,
        fake: torch.Tensor
    ) -> torch.Tensor:
        """Compute R2 regularization."""
        fake = fake.requires_grad_(True)
        d_fake = discriminator(fake)
        
        if isinstance(d_fake, (list, tuple)):
            d_fake = d_fake[0]
            
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_fake.sum(),
            inputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Squared gradient norm
        r2_penalty = gradients.view(gradients.size(0), -1).pow(2).sum(dim=1).mean()
        
        return self.weight * r2_penalty


class PathLengthRegularization(nn.Module):
    """
    Path Length Regularization from StyleGAN2.
    
    Encourages smooth mappings from latent space to image space.
    
    Args:
        weight: Regularization weight
        decay: EMA decay for path length mean
    """
    
    def __init__(self, weight: float = 2.0, decay: float = 0.01):
        super().__init__()
        self.weight = weight
        self.decay = decay
        self.register_buffer('path_length_mean', torch.zeros([]))
        
    def forward(
        self,
        fake_images: torch.Tensor,
        latents: torch.Tensor
    ) -> torch.Tensor:
        """Compute path length regularization."""
        batch_size = fake_images.size(0)
        
        # Random noise for gradient computation
        noise = torch.randn_like(fake_images) / math.sqrt(
            fake_images.size(2) * fake_images.size(3)
        )
        
        # Compute Jacobian
        gradients = torch.autograd.grad(
            outputs=(fake_images * noise).sum(),
            inputs=latents,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Path length
        path_lengths = torch.sqrt(gradients.pow(2).sum(dim=list(range(1, gradients.dim()))) + 1e-8)
        
        # Update mean with EMA
        path_length_mean = self.path_length_mean + self.decay * (
            path_lengths.mean() - self.path_length_mean
        )
        self.path_length_mean.copy_(path_length_mean.detach())
        
        # Regularization
        path_penalty = (path_lengths - self.path_length_mean).pow(2).mean()
        
        return self.weight * path_penalty


class OrthogonalRegularization(nn.Module):
    """
    Orthogonal Regularization for convolutional kernels.
    
    Encourages orthogonal filters for better feature diversity.
    
    Args:
        weight: Regularization weight
    """
    
    def __init__(self, weight: float = 1e-4):
        super().__init__()
        self.weight = weight
        
    def forward(self, model: nn.Module) -> torch.Tensor:
        """Compute orthogonal regularization."""
        reg_loss = 0.0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight
                weight_flat = weight.view(weight.size(0), -1)
                
                # Orthogonality constraint: W^T W should be identity
                wTw = torch.mm(weight_flat, weight_flat.t())
                identity = torch.eye(wTw.size(0), device=wTw.device)
                
                reg_loss += ((wTw - identity) ** 2).sum()
                
        return self.weight * reg_loss


class LatentRegularization(nn.Module):
    """
    Latent Space Regularization.
    
    Encourages smooth and well-structured latent space.
    
    Args:
        weight: Regularization weight
        prior: Prior distribution ('gaussian' or 'uniform')
    """
    
    def __init__(self, weight: float = 0.01, prior: str = 'gaussian'):
        super().__init__()
        self.weight = weight
        self.prior = prior
        
    def forward(
        self,
        mu: torch.Tensor,
        logvar: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute latent regularization (KL divergence).
        
        For VAE-style regularization towards standard normal.
        """
        if logvar is not None:
            # KL divergence for VAE
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / mu.size(0)  # Normalize by batch size
        else:
            # Simple L2 regularization towards zero mean
            kl_loss = mu.pow(2).mean()
            
        return self.weight * kl_loss


class ConsistencyRegularization(nn.Module):
    """
    Consistency Regularization for semi-supervised learning.
    
    Enforces consistent predictions under different augmentations.
    
    Args:
        weight: Regularization weight
        temperature: Temperature for sharpening
    """
    
    def __init__(self, weight: float = 1.0, temperature: float = 0.5):
        super().__init__()
        self.weight = weight
        self.temperature = temperature
        
    def forward(
        self,
        pred1: torch.Tensor,
        pred2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency regularization.
        
        Args:
            pred1: Predictions from original input
            pred2: Predictions from augmented input
            
        Returns:
            Consistency loss
        """
        # Sharpen pred1 (from original)
        pred1_sharp = F.softmax(pred1 / self.temperature, dim=1).detach()
        
        # KL divergence
        log_pred2 = F.log_softmax(pred2, dim=1)
        consistency_loss = F.kl_div(log_pred2, pred1_sharp, reduction='batchmean')
        
        return self.weight * consistency_loss


class CutoutRegularization(nn.Module):
    """
    Cutout Regularization.
    
    Applies random cutout during training for robustness.
    Returns augmented tensor (not a loss).
    
    Args:
        n_holes: Number of cutout regions
        length: Size of each cutout region
    """
    
    def __init__(self, n_holes: int = 1, length: int = 16):
        super().__init__()
        self.n_holes = n_holes
        self.length = length
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cutout augmentation."""
        h, w = x.size(-2), x.size(-1)
        mask = torch.ones_like(x)
        
        for _ in range(self.n_holes):
            y = torch.randint(h, size=(1,)).item()
            x_pos = torch.randint(w, size=(1,)).item()
            
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x_pos - self.length // 2)
            x2 = min(w, x_pos + self.length // 2)
            
            mask[:, :, y1:y2, x1:x2] = 0
            
        return x * mask
