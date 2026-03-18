"""
regularization losses for training stability.

this module provides regularization losses to improve
training stability and prevent mode collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math


class GradientPenalty(nn.Module):
    """
    gradient penalty for wgan-gp training.
    
    enforces lipschitz constraint on discriminator.
    
    args:
        weight: weight for gradient penalty
        target_norm: target norm for gradients (default 1.0)
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
        compute gradient penalty.
        
        args:
            discriminator: discriminator network
            real: real samples
            fake: fake samples
            
        returns:
            gradient penalty loss
        """
        batch_size = real.size(0)
        device = real.device
        
        # random interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)
        
        # discriminator output
        d_interpolated = discriminator(interpolated)
        
        if isinstance(d_interpolated, (list, tuple)):
            d_interpolated = d_interpolated[0]
            
        # compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # compute gradient norm
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        
        # gradient penalty
        penalty = ((gradient_norm - self.target_norm) ** 2).mean()
        
        return self.weight * penalty


class SpectralRegularization(nn.Module):
    """
    spectral regularization for discriminator.
    
    applies soft spectral normalization as a loss term.
    
    args:
        weight: regularization weight
        n_power_iterations: number of power iterations
    """
    
    def __init__(self, weight: float = 1.0, n_power_iterations: int = 1):
        super().__init__()
        self.weight = weight
        self.n_power_iterations = n_power_iterations
        
    def forward(self, model: nn.Module) -> torch.Tensor:
        """compute spectral regularization loss."""
        reg_loss = 0.0
        count = 0
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight
                
                # compute spectral norm
                with torch.no_grad():
                    weight_mat = weight.view(weight.size(0), -1)
                    
                    # power iteration
                    u = torch.randn(1, weight_mat.size(0), device=weight.device)
                    u = F.normalize(u, dim=1)
                    
                    for _ in range(self.n_power_iterations):
                        v = F.normalize(torch.mm(u, weight_mat), dim=1)
                        u = F.normalize(torch.mm(v, weight_mat.t()), dim=1)
                        
                    sigma = torch.mm(torch.mm(u, weight_mat), v.t())
                    
                # regularize towards unit spectral norm
                reg_loss += (sigma - 1.0) ** 2
                count += 1
                
        return self.weight * reg_loss / max(count, 1)


class R1Regularization(nn.Module):
    """
    r1 gradient regularization.
    
    zero-centered gradient penalty for real samples only.
    used in stylegan-style training.
    
    args:
        weight: regularization weight
    """
    
    def __init__(self, weight: float = 10.0):
        super().__init__()
        self.weight = weight
        
    def forward(
        self,
        discriminator: nn.Module,
        real: torch.Tensor
    ) -> torch.Tensor:
        """compute r1 regularization."""
        real = real.requires_grad_(True)
        d_real = discriminator(real)
        
        if isinstance(d_real, (list, tuple)):
            d_real = d_real[0]
            
        # compute gradients
        gradients = torch.autograd.grad(
            outputs=d_real.sum(),
            inputs=real,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # squared gradient norm
        r1_penalty = gradients.view(gradients.size(0), -1).pow(2).sum(dim=1).mean()
        
        return self.weight * r1_penalty


class R2Regularization(nn.Module):
    """
    r2 gradient regularization.
    
    zero-centered gradient penalty for fake samples only.
    
    args:
        weight: regularization weight
    """
    
    def __init__(self, weight: float = 10.0):
        super().__init__()
        self.weight = weight
        
    def forward(
        self,
        discriminator: nn.Module,
        fake: torch.Tensor
    ) -> torch.Tensor:
        """compute r2 regularization."""
        fake = fake.requires_grad_(True)
        d_fake = discriminator(fake)
        
        if isinstance(d_fake, (list, tuple)):
            d_fake = d_fake[0]
            
        # compute gradients
        gradients = torch.autograd.grad(
            outputs=d_fake.sum(),
            inputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # squared gradient norm
        r2_penalty = gradients.view(gradients.size(0), -1).pow(2).sum(dim=1).mean()
        
        return self.weight * r2_penalty


class PathLengthRegularization(nn.Module):
    """
    path length regularization from stylegan2.
    
    encourages smooth mappings from latent space to image space.
    
    args:
        weight: regularization weight
        decay: ema decay for path length mean
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
        """compute path length regularization."""
        batch_size = fake_images.size(0)
        
        # random noise for gradient computation
        noise = torch.randn_like(fake_images) / math.sqrt(
            fake_images.size(2) * fake_images.size(3)
        )
        
        # compute jacobian
        gradients = torch.autograd.grad(
            outputs=(fake_images * noise).sum(),
            inputs=latents,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # path length
        path_lengths = torch.sqrt(gradients.pow(2).sum(dim=list(range(1, gradients.dim()))) + 1e-8)
        
        # update mean with ema
        path_length_mean = self.path_length_mean + self.decay * (
            path_lengths.mean() - self.path_length_mean
        )
        self.path_length_mean.copy_(path_length_mean.detach())
        
        # regularization
        path_penalty = (path_lengths - self.path_length_mean).pow(2).mean()
        
        return self.weight * path_penalty


class OrthogonalRegularization(nn.Module):
    """
    orthogonal regularization for convolutional kernels.
    
    encourages orthogonal filters for better feature diversity.
    
    args:
        weight: regularization weight
    """
    
    def __init__(self, weight: float = 1e-4):
        super().__init__()
        self.weight = weight
        
    def forward(self, model: nn.Module) -> torch.Tensor:
        """compute orthogonal regularization."""
        reg_loss = 0.0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight
                weight_flat = weight.view(weight.size(0), -1)
                
                # orthogonality constraint: w^t w should be identity
                wTw = torch.mm(weight_flat, weight_flat.t())
                identity = torch.eye(wTw.size(0), device=wTw.device)
                
                reg_loss += ((wTw - identity) ** 2).sum()
                
        return self.weight * reg_loss


class LatentRegularization(nn.Module):
    """
    latent space regularization.
    
    encourages smooth and well-structured latent space.
    
    args:
        weight: regularization weight
        prior: prior distribution ('gaussian' or 'uniform')
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
        compute latent regularization (kl divergence).
        
        for vae-style regularization towards standard normal.
        """
        if logvar is not None:
            # kl divergence for vae
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / mu.size(0)  # normalize by batch size
        else:
            # simple l2 regularization towards zero mean
            kl_loss = mu.pow(2).mean()
            
        return self.weight * kl_loss


class ConsistencyRegularization(nn.Module):
    """
    consistency regularization for semi-supervised learning.
    
    enforces consistent predictions under different augmentations.
    
    args:
        weight: regularization weight
        temperature: temperature for sharpening
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
        compute consistency regularization.
        
        args:
            pred1: predictions from original input
            pred2: predictions from augmented input
            
        returns:
            consistency loss
        """
        # sharpen pred1 (from original)
        pred1_sharp = F.softmax(pred1 / self.temperature, dim=1).detach()
        
        # kl divergence
        log_pred2 = F.log_softmax(pred2, dim=1)
        consistency_loss = F.kl_div(log_pred2, pred1_sharp, reduction='batchmean')
        
        return self.weight * consistency_loss


class CutoutRegularization(nn.Module):
    """
    cutout regularization.
    
    applies random cutout during training for robustness.
    returns augmented tensor (not a loss).
    
    args:
        n_holes: number of cutout regions
        length: size of each cutout region
    """
    
    def __init__(self, n_holes: int = 1, length: int = 16):
        super().__init__()
        self.n_holes = n_holes
        self.length = length
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """apply cutout augmentation."""
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
