"""
Consistency Losses for cycle-consistent image translation.

This module provides cycle consistency and identity preservation losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class CycleConsistencyLoss(nn.Module):
    """
    Cycle Consistency Loss.
    
    Ensures that x -> G(x) -> F(G(x)) ≈ x
    and y -> F(y) -> G(F(y)) ≈ y
    
    Args:
        loss_type: 'l1' or 'l2'
        weight: Loss weight
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
        Compute cycle consistency loss.
        
        Args:
            reconstructed: F(G(x)) or G(F(y))
            original: x or y
            
        Returns:
            Weighted cycle loss
        """
        return self.weight * self.loss_fn(reconstructed, original)


class IdentityLoss(nn.Module):
    """
    Identity Loss.
    
    Ensures that G(y) ≈ y and F(x) ≈ x for color/content preservation.
    
    Args:
        loss_type: 'l1' or 'l2'
        weight: Loss weight
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
        Compute identity loss.
        
        Args:
            identity_output: G(y) or F(x)
            original: y or x
            
        Returns:
            Weighted identity loss
        """
        return self.weight * self.loss_fn(identity_output, original)


class FeatureMatchingLoss(nn.Module):
    """
    Feature Matching Loss.
    
    Matches intermediate discriminator features between real and fake.
    Helps stabilize GAN training.
    
    Args:
        n_layers: Number of discriminator layers to match
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
        Compute feature matching loss.
        
        Args:
            fake_features: List of discriminator features for fake images
            real_features: List of discriminator features for real images
            
        Returns:
            Feature matching loss
        """
        loss = 0.0
        n_layers = min(len(fake_features), len(real_features), self.n_layers)
        
        for i in range(n_layers):
            loss += self.loss_fn(fake_features[i], real_features[i].detach())
            
        return loss / n_layers


class ContrastiveConsistencyLoss(nn.Module):
    """
    Contrastive Consistency Loss.
    
    Enforces consistency using contrastive learning principles.
    Positive pairs: (x, F(G(x))) and (y, G(F(y)))
    Negative pairs: Other samples in batch
    
    Args:
        temperature: Temperature for softmax
        batch_size: Batch size for negative sampling
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
        """Compute contrastive consistency loss."""
        batch_size = original.size(0)
        
        # Flatten and normalize
        orig_flat = original.view(batch_size, -1)
        recon_flat = reconstructed.view(batch_size, -1)
        
        orig_norm = F.normalize(orig_flat, dim=1)
        recon_norm = F.normalize(recon_flat, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(orig_norm, recon_norm.T) / self.temperature
        
        # Labels: positive pairs are on diagonal
        labels = torch.arange(batch_size, device=similarity.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class SemanticConsistencyLoss(nn.Module):
    """
    Semantic Consistency Loss.
    
    Ensures semantic content is preserved through translation.
    Uses a pretrained feature extractor.
    
    Args:
        feature_extractor: Feature extraction network
        layers: Which layers to use
    """
    
    def __init__(
        self,
        feature_extractor: nn.Module,
        layers: Optional[list] = None
    ):
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.layers = layers or ['conv1', 'conv2', 'conv3']
        
        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def forward(
        self,
        input_img: torch.Tensor,
        output_img: torch.Tensor
    ) -> torch.Tensor:
        """Compute semantic consistency loss."""
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
    Temporal Consistency Loss.
    
    For video/sequential data, ensures temporal coherence.
    
    Args:
        weight: Loss weight
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
        Compute temporal consistency loss.
        
        If flow is provided, uses optical flow warping.
        Otherwise, uses simple frame difference.
        """
        if flow is not None:
            # Warp previous translated frame using flow
            warped_prev = self._warp(prev_translated, flow)
            loss = F.l1_loss(current_translated, warped_prev)
        else:
            # Simple frame difference consistency
            input_diff = current_frame - prev_frame
            output_diff = current_translated - prev_translated
            loss = F.l1_loss(output_diff, input_diff)
            
        return self.weight * loss
        
    def _warp(
        self,
        img: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        """Warp image using optical flow."""
        B, C, H, W = img.size()
        
        # Create mesh grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=img.device),
            torch.arange(W, device=img.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float()
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Add flow to grid
        new_grid = grid + flow.permute(0, 2, 3, 1)
        
        # Normalize grid to [-1, 1]
        new_grid[:, :, :, 0] = 2.0 * new_grid[:, :, :, 0] / (W - 1) - 1.0
        new_grid[:, :, :, 1] = 2.0 * new_grid[:, :, :, 1] / (H - 1) - 1.0
        
        # Sample
        warped = F.grid_sample(img, new_grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        return warped


class ModeSeekingLoss(nn.Module):
    """
    Mode Seeking Regularization Loss.
    
    Encourages diversity in generated outputs for different inputs.
    Prevents mode collapse.
    
    Args:
        weight: Loss weight
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
        Compute mode seeking loss.
        
        Args:
            z1, z2: Different latent codes
            output1, output2: Corresponding outputs
            
        Returns:
            Mode seeking loss (maximize output distance / latent distance)
        """
        # Flatten
        output1_flat = output1.view(output1.size(0), -1)
        output2_flat = output2.view(output2.size(0), -1)
        z1_flat = z1.view(z1.size(0), -1)
        z2_flat = z2.view(z2.size(0), -1)
        
        # Compute distances
        output_dist = torch.norm(output1_flat - output2_flat, dim=1)
        latent_dist = torch.norm(z1_flat - z2_flat, dim=1) + 1e-8
        
        # Maximize ratio
        loss = -torch.mean(output_dist / latent_dist)
        
        return self.weight * loss


# Aliases for compatibility
CycleLoss = CycleConsistencyLoss
