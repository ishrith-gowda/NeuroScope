"""
Adversarial Loss functions for GAN training.

This module provides various adversarial loss formulations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


class VanillaGANLoss(nn.Module):
    """
    Vanilla GAN loss using binary cross entropy.
    
    L_D = -E[log(D(x))] - E[log(1 - D(G(z)))]
    L_G = -E[log(D(G(z)))]
    
    Args:
        label_smoothing: Smoothing factor for real labels
    """
    
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.bce = nn.BCEWithLogitsLoss()
        
    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor
    ) -> torch.Tensor:
        """Compute discriminator loss."""
        real_label = torch.ones_like(real_pred) * (1.0 - self.label_smoothing)
        fake_label = torch.zeros_like(fake_pred)
        
        real_loss = self.bce(real_pred, real_label)
        fake_loss = self.bce(fake_pred, fake_label)
        
        return (real_loss + fake_loss) * 0.5
        
    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """Compute generator loss."""
        real_label = torch.ones_like(fake_pred)
        return self.bce(fake_pred, real_label)


class LSGANLoss(nn.Module):
    """
    Least Squares GAN loss.
    
    L_D = 0.5 * E[(D(x) - 1)²] + 0.5 * E[D(G(z))²]
    L_G = 0.5 * E[(D(G(z)) - 1)²]
    
    More stable training compared to vanilla GAN.
    """
    
    def __init__(self):
        super().__init__()
        
    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor
    ) -> torch.Tensor:
        """Compute discriminator loss."""
        real_loss = torch.mean((real_pred - 1) ** 2)
        fake_loss = torch.mean(fake_pred ** 2)
        return (real_loss + fake_loss) * 0.5
        
    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """Compute generator loss."""
        return torch.mean((fake_pred - 1) ** 2) * 0.5


class WassersteinGANLoss(nn.Module):
    """
    Wasserstein GAN loss with gradient penalty.
    
    L_D = E[D(G(z))] - E[D(x)] + λ * GP
    L_G = -E[D(G(z))]
    
    Args:
        gp_weight: Weight for gradient penalty term
    """
    
    def __init__(self, gp_weight: float = 10.0):
        super().__init__()
        self.gp_weight = gp_weight
        
    def gradient_penalty(
        self,
        discriminator: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP."""
        batch_size = real.size(0)
        device = real.device
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolated = alpha * real + (1 - alpha) * fake.detach()
        interpolated.requires_grad_(True)
        
        # Discriminator output
        d_interpolated = discriminator(interpolated)
        if isinstance(d_interpolated, list):
            d_interpolated = d_interpolated[-1]  # Use last scale
            
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty
        
    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
        discriminator: nn.Module = None,
        real: torch.Tensor = None,
        fake: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute discriminator loss with gradient penalty."""
        loss = fake_pred.mean() - real_pred.mean()
        
        if discriminator is not None and real is not None and fake is not None:
            gp = self.gradient_penalty(discriminator, real, fake)
            loss = loss + self.gp_weight * gp
            
        return loss
        
    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """Compute generator loss."""
        return -fake_pred.mean()


class HingeGANLoss(nn.Module):
    """
    Hinge GAN loss.
    
    L_D = E[ReLU(1 - D(x))] + E[ReLU(1 + D(G(z)))]
    L_G = -E[D(G(z))]
    
    Used in SAGAN and BigGAN.
    """
    
    def __init__(self):
        super().__init__()
        
    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor
    ) -> torch.Tensor:
        """Compute discriminator loss."""
        real_loss = F.relu(1.0 - real_pred).mean()
        fake_loss = F.relu(1.0 + fake_pred).mean()
        return real_loss + fake_loss
        
    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """Compute generator loss."""
        return -fake_pred.mean()


class MultiScaleGANLoss(nn.Module):
    """
    Multi-scale adversarial loss for multi-scale discriminators.
    
    Aggregates losses from multiple discriminator scales.
    
    Args:
        loss_type: Base loss type ('lsgan', 'hinge', 'vanilla')
        weights: Weights for each scale (uniform if None)
    """
    
    def __init__(
        self,
        loss_type: str = 'lsgan',
        weights: List[float] = None
    ):
        super().__init__()
        
        if loss_type == 'lsgan':
            self.base_loss = LSGANLoss()
        elif loss_type == 'hinge':
            self.base_loss = HingeGANLoss()
        elif loss_type == 'vanilla':
            self.base_loss = VanillaGANLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        self.weights = weights
        
    def discriminator_loss(
        self,
        real_preds: List[torch.Tensor],
        fake_preds: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute multi-scale discriminator loss."""
        weights = self.weights or [1.0] * len(real_preds)
        
        total_loss = 0.0
        for w, real_pred, fake_pred in zip(weights, real_preds, fake_preds):
            total_loss += w * self.base_loss.discriminator_loss(real_pred, fake_pred)
            
        return total_loss / sum(weights)
        
    def generator_loss(
        self,
        fake_preds: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute multi-scale generator loss."""
        weights = self.weights or [1.0] * len(fake_preds)
        
        total_loss = 0.0
        for w, fake_pred in zip(weights, fake_preds):
            total_loss += w * self.base_loss.generator_loss(fake_pred)
            
        return total_loss / sum(weights)


# Aliases for compatibility
GANLoss = LSGANLoss  # Default GAN loss
WassersteinLoss = WassersteinGANLoss
HingeLoss = HingeGANLoss
RelativisticLoss = VanillaGANLoss  # Placeholder
RelativisticAverageLoss = VanillaGANLoss  # Placeholder
SoftplusLoss = VanillaGANLoss  # Placeholder
