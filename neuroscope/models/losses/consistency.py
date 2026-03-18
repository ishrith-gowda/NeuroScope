"""
consistency losses for cycle-consistent image translation.

this module provides cycle consistency and identity preservation losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class CycleConsistencyLoss(nn.Module):
    """
    cycle consistency loss.
    
    ensures that x -> g(x) -> f(g(x)) ≈ x
    and y -> f(y) -> g(f(y)) ≈ y
    
    args:
        loss_type: 'l1' or 'l2'
        weight: loss weight
    """
    
    def __init__(self, loss_type: str = 'l1', weight: float = 10.0):
        super().__init__()
        
        self.weight = weight
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        else:
            self.loss_fn = nn.MSELoss()
            
    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor
    ) -> torch.Tensor:
        """
        compute cycle consistency loss.
        
        args:
            reconstructed: f(g(x)) or g(f(y))
            original: x or y
            
        returns:
            weighted cycle loss
        """
        return self.weight * self.loss_fn(reconstructed, original)


class IdentityLoss(nn.Module):
    """
    identity loss.
    
    ensures that g(y) ≈ y and f(x) ≈ x for color/content preservation.
    
    args:
        loss_type: 'l1' or 'l2'
        weight: loss weight
    """
    
    def __init__(self, loss_type: str = 'l1', weight: float = 5.0):
        super().__init__()
        
        self.weight = weight
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        else:
            self.loss_fn = nn.MSELoss()
            
    def forward(
        self,
        identity_output: torch.Tensor,
        original: torch.Tensor
    ) -> torch.Tensor:
        """
        compute identity loss.
        
        args:
            identity_output: g(y) or f(x)
            original: y or x
            
        returns:
            weighted identity loss
        """
        return self.weight * self.loss_fn(identity_output, original)


class FeatureMatchingLoss(nn.Module):
    """
    feature matching loss.
    
    matches intermediate discriminator features between real and fake.
    helps stabilize gan training.
    
    args:
        n_layers: number of discriminator layers to match
        loss_type: 'l1' or 'l2'
    """
    
    def __init__(self, n_layers: int = 3, loss_type: str = 'l1'):
        super().__init__()
        
        self.n_layers = n_layers
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        else:
            self.loss_fn = nn.MSELoss()
            
    def forward(
        self,
        fake_features: list,
        real_features: list
    ) -> torch.Tensor:
        """
        compute feature matching loss.
        
        args:
            fake_features: list of discriminator features for fake images
            real_features: list of discriminator features for real images
            
        returns:
            feature matching loss
        """
        loss = 0.0
        n_layers = min(len(fake_features), len(real_features), self.n_layers)
        
        for i in range(n_layers):
            loss += self.loss_fn(fake_features[i], real_features[i].detach())
            
        return loss / n_layers


class ContrastiveConsistencyLoss(nn.Module):
    """
    contrastive consistency loss.
    
    enforces consistency using contrastive learning principles.
    positive pairs: (x, f(g(x))) and (y, g(f(y)))
    negative pairs: other samples in batch
    
    args:
        temperature: temperature for softmax
        batch_size: batch size for negative sampling
    """
    
    def __init__(self, temperature: float = 0.07, batch_size: int = 4):
        super().__init__()
        
        self.temperature = temperature
        self.batch_size = batch_size
        
    def forward(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """compute contrastive consistency loss."""
        batch_size = original.size(0)
        
        # flatten and normalize
        orig_flat = original.view(batch_size, -1)
        recon_flat = reconstructed.view(batch_size, -1)
        
        orig_norm = F.normalize(orig_flat, dim=1)
        recon_norm = F.normalize(recon_flat, dim=1)
        
        # compute similarity matrix
        similarity = torch.matmul(orig_norm, recon_norm.T) / self.temperature
        
        # labels: positive pairs are on diagonal
        labels = torch.arange(batch_size, device=similarity.device)
        
        # cross-entropy loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class SemanticConsistencyLoss(nn.Module):
    """
    semantic consistency loss.
    
    ensures semantic content is preserved through translation.
    uses a pretrained feature extractor.
    
    args:
        feature_extractor: feature extraction network
        layers: which layers to use
    """
    
    def __init__(
        self,
        feature_extractor: nn.Module,
        layers: Optional[list] = None
    ):
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.layers = layers or ['conv1', 'conv2', 'conv3']
        
        # freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def forward(
        self,
        input_img: torch.Tensor,
        output_img: torch.Tensor
    ) -> torch.Tensor:
        """compute semantic consistency loss."""
        input_features = self.feature_extractor(input_img)
        output_features = self.feature_extractor(output_img)
        
        loss = 0.0
        count = 0
        
        if isinstance(input_features, dict):
            for layer in self.layers:
                if layer in input_features and layer in output_features:
                    loss += F.l1_loss(output_features[layer], input_features[layer].detach())
                    count += 1
        else:
            loss = F.l1_loss(output_features, input_features.detach())
            count = 1
            
        return loss / max(count, 1)


class TemporalConsistencyLoss(nn.Module):
    """
    temporal consistency loss.
    
    for video/sequential data, ensures temporal coherence.
    
    args:
        weight: loss weight
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        
    def forward(
        self,
        current_frame: torch.Tensor,
        prev_frame: torch.Tensor,
        current_translated: torch.Tensor,
        prev_translated: torch.Tensor,
        flow: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        compute temporal consistency loss.
        
        if flow is provided, uses optical flow warping.
        otherwise, uses simple frame difference.
        """
        if flow is not None:
            # warp previous translated frame using flow
            warped_prev = self._warp(prev_translated, flow)
            loss = F.l1_loss(current_translated, warped_prev)
        else:
            # simple frame difference consistency
            input_diff = current_frame - prev_frame
            output_diff = current_translated - prev_translated
            loss = F.l1_loss(output_diff, input_diff)
            
        return self.weight * loss
        
    def _warp(
        self,
        img: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        """warp image using optical flow."""
        B, C, H, W = img.size()
        
        # create mesh grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=img.device),
            torch.arange(W, device=img.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float()
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        
        # add flow to grid
        new_grid = grid + flow.permute(0, 2, 3, 1)
        
        # normalize grid to [-1, 1]
        new_grid[:, :, :, 0] = 2.0 * new_grid[:, :, :, 0] / (W - 1) - 1.0
        new_grid[:, :, :, 1] = 2.0 * new_grid[:, :, :, 1] / (H - 1) - 1.0
        
        # sample
        warped = F.grid_sample(img, new_grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        return warped


class ModeSeekingLoss(nn.Module):
    """
    mode seeking regularization loss.
    
    encourages diversity in generated outputs for different inputs.
    prevents mode collapse.
    
    args:
        weight: loss weight
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        output1: torch.Tensor,
        output2: torch.Tensor
    ) -> torch.Tensor:
        """
        compute mode seeking loss.
        
        args:
            z1, z2: different latent codes
            output1, output2: corresponding outputs
            
        returns:
            mode seeking loss (maximize output distance / latent distance)
        """
        # flatten
        output1_flat = output1.view(output1.size(0), -1)
        output2_flat = output2.view(output2.size(0), -1)
        z1_flat = z1.view(z1.size(0), -1)
        z2_flat = z2.view(z2.size(0), -1)
        
        # compute distances
        output_dist = torch.norm(output1_flat - output2_flat, dim=1)
        latent_dist = torch.norm(z1_flat - z2_flat, dim=1) + 1e-8
        
        # maximize ratio
        loss = -torch.mean(output_dist / latent_dist)
        
        return self.weight * loss


# aliases for compatibility
CycleLoss = CycleConsistencyLoss
