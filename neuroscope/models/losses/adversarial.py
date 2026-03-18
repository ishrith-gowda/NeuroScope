"""
adversarial loss functions for gan training.

this module provides various adversarial loss formulations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


class VanillaGANLoss(nn.Module):
    """
    vanilla gan loss using binary cross entropy.
    
    l_d = -e[log(d(x))] - e[log(1 - d(g(z)))]
    l_g = -e[log(d(g(z)))]
    
    args:
        label_smoothing: smoothing factor for real labels
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
        """compute discriminator loss."""
        real_label = torch.ones_like(real_pred) * (1.0 - self.label_smoothing)
        fake_label = torch.zeros_like(fake_pred)
        
        real_loss = self.bce(real_pred, real_label)
        fake_loss = self.bce(fake_pred, fake_label)
        
        return (real_loss + fake_loss) * 0.5
        
    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """compute generator loss."""
        real_label = torch.ones_like(fake_pred)
        return self.bce(fake_pred, real_label)


class LSGANLoss(nn.Module):
    """
    least squares gan loss.
    
    l_d = 0.5 * e[(d(x) - 1)²] + 0.5 * e[d(g(z))²]
    l_g = 0.5 * e[(d(g(z)) - 1)²]
    
    more stable training compared to vanilla gan.
    """
    
    def __init__(self):
        super().__init__()
        
    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor
    ) -> torch.Tensor:
        """compute discriminator loss."""
        real_loss = torch.mean((real_pred - 1) ** 2)
        fake_loss = torch.mean(fake_pred ** 2)
        return (real_loss + fake_loss) * 0.5
        
    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """compute generator loss."""
        return torch.mean((fake_pred - 1) ** 2) * 0.5


class WassersteinGANLoss(nn.Module):
    """
    wasserstein gan loss with gradient penalty.
    
    l_d = e[d(g(z))] - e[d(x)] + λ * gp
    l_g = -e[d(g(z))]
    
    args:
        gp_weight: weight for gradient penalty term
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
        """compute gradient penalty for wgan-gp."""
        batch_size = real.size(0)
        device = real.device
        
        # random interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolated = alpha * real + (1 - alpha) * fake.detach()
        interpolated.requires_grad_(True)
        
        # discriminator output
        d_interpolated = discriminator(interpolated)
        if isinstance(d_interpolated, list):
            d_interpolated = d_interpolated[-1]  # use last scale
            
        # compute gradients
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
        """compute discriminator loss with gradient penalty."""
        loss = fake_pred.mean() - real_pred.mean()
        
        if discriminator is not None and real is not None and fake is not None:
            gp = self.gradient_penalty(discriminator, real, fake)
            loss = loss + self.gp_weight * gp
            
        return loss
        
    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """compute generator loss."""
        return -fake_pred.mean()


class HingeGANLoss(nn.Module):
    """
    hinge gan loss.
    
    l_d = e[relu(1 - d(x))] + e[relu(1 + d(g(z)))]
    l_g = -e[d(g(z))]
    
    used in sagan and biggan.
    """
    
    def __init__(self):
        super().__init__()
        
    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor
    ) -> torch.Tensor:
        """compute discriminator loss."""
        real_loss = F.relu(1.0 - real_pred).mean()
        fake_loss = F.relu(1.0 + fake_pred).mean()
        return real_loss + fake_loss
        
    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """compute generator loss."""
        return -fake_pred.mean()


class MultiScaleGANLoss(nn.Module):
    """
    multi-scale adversarial loss for multi-scale discriminators.
    
    aggregates losses from multiple discriminator scales.
    
    args:
        loss_type: base loss type ('lsgan', 'hinge', 'vanilla')
        weights: weights for each scale (uniform if none)
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
        """compute multi-scale discriminator loss."""
        weights = self.weights or [1.0] * len(real_preds)
        
        total_loss = 0.0
        for w, real_pred, fake_pred in zip(weights, real_preds, fake_preds):
            total_loss += w * self.base_loss.discriminator_loss(real_pred, fake_pred)
            
        return total_loss / sum(weights)
        
    def generator_loss(
        self,
        fake_preds: List[torch.Tensor]
    ) -> torch.Tensor:
        """compute multi-scale generator loss."""
        weights = self.weights or [1.0] * len(fake_preds)
        
        total_loss = 0.0
        for w, fake_pred in zip(weights, fake_preds):
            total_loss += w * self.base_loss.generator_loss(fake_pred)
            
        return total_loss / sum(weights)


# aliases for compatibility
GANLoss = LSGANLoss  # default gan loss
WassersteinLoss = WassersteinGANLoss
HingeLoss = HingeGANLoss
RelativisticLoss = VanillaGANLoss  # placeholder
RelativisticAverageLoss = VanillaGANLoss  # placeholder
SoftplusLoss = VanillaGANLoss  # placeholder
