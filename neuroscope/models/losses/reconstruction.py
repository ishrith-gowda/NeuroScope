"""
Reconstruction Losses for image-to-image translation.

This module provides various reconstruction loss formulations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class L1Loss(nn.Module):
    """L1 (Mean Absolute Error) loss."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.loss = nn.L1Loss(reduction=reduction)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class L2Loss(nn.Module):
    """L2 (Mean Squared Error) loss."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss.
    
    SSIM measures structural similarity between images.
    Loss = 1 - SSIM
    
    Args:
        window_size: Size of the Gaussian window
        sigma: Standard deviation of Gaussian
        size_average: Whether to average over batch
        channel: Number of channels
    """
    
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        size_average: bool = True,
        channel: int = 1
    ):
        super().__init__()
        
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        
        # Create Gaussian window
        gaussian = self._create_gaussian_window(window_size, sigma)
        self.register_buffer('window', gaussian.unsqueeze(0).unsqueeze(0))
        
    def _create_gaussian_window(self, window_size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian window."""
        coords = torch.arange(window_size) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.outer(g)
        return window
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SSIM loss.
        
        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
            
        Returns:
            SSIM loss (1 - SSIM)
        """
        channel = pred.size(1)
        
        # Expand window to all channels
        window = self.window.expand(channel, 1, -1, -1).to(pred.device, pred.dtype)
        
        # Compute means
        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=channel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute variances and covariance
        sigma1_sq = F.conv2d(pred ** 2, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        
        # Constants for stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
                   
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean([1, 2, 3])


class MultiScaleSSIMLoss(nn.Module):
    """
    Multi-Scale SSIM (MS-SSIM) loss.
    
    Computes SSIM at multiple scales for better structure preservation.
    
    Args:
        window_size: Size of the Gaussian window
        levels: Number of scales
    """
    
    def __init__(self, window_size: int = 11, levels: int = 5):
        super().__init__()
        
        self.levels = levels
        self.ssim = SSIMLoss(window_size=window_size)
        
        # Default weights for each level
        self.weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333][:levels]
        self.weights = torch.tensor(self.weights)
        self.weights = self.weights / self.weights.sum()
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute MS-SSIM loss."""
        msssim_values = []
        
        for i in range(self.levels):
            ssim_val = 1 - self.ssim(pred, target)  # Convert loss back to SSIM
            msssim_values.append(ssim_val)
            
            if i < self.levels - 1:
                # Downsample
                pred = F.avg_pool2d(pred, 2)
                target = F.avg_pool2d(target, 2)
                
        # Weighted product
        msssim = torch.stack(msssim_values)
        weights = self.weights.to(msssim.device)
        msssim = (msssim ** weights.view(-1, 1)).prod(dim=0).mean()
        
        return 1 - msssim


class GradientLoss(nn.Module):
    """
    Gradient (edge) loss for preserving image structure.
    
    Computes L1/L2 loss on image gradients.
    
    Args:
        loss_type: 'l1' or 'l2'
    """
    
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        
        self.loss_type = loss_type
        
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
    def _compute_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute image gradient magnitude."""
        B, C, H, W = x.size()
        
        # Expand Sobel filters to all channels
        sobel_x = self.sobel_x.expand(C, 1, 3, 3).to(x.device, x.dtype)
        sobel_y = self.sobel_y.expand(C, 1, 3, 3).to(x.device, x.dtype)
        
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=C)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=C)
        
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient loss."""
        pred_grad = self._compute_gradient(pred)
        target_grad = self._compute_gradient(target)
        
        if self.loss_type == 'l1':
            return F.l1_loss(pred_grad, target_grad)
        else:
            return F.mse_loss(pred_grad, target_grad)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (smooth L1).
    
    L = sqrt((x - y)² + ε²)
    
    More robust to outliers than L2, smoother than L1.
    
    Args:
        epsilon: Small constant for numerical stability
    """
    
    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.epsilon ** 2).mean()


class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss for frequency-domain supervision.
    
    Emphasizes hard frequency components using focal weighting.
    
    Args:
        loss_weight: Weight for the loss
        alpha: Focal weight parameter
    """
    
    def __init__(self, loss_weight: float = 1.0, alpha: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute focal frequency loss."""
        # 2D FFT
        pred_freq = torch.fft.fft2(pred, norm='ortho')
        target_freq = torch.fft.fft2(target, norm='ortho')
        
        # Frequency distance
        freq_distance = torch.abs(pred_freq - target_freq)
        
        # Focal weight (emphasize hard frequencies)
        if weight_matrix is None:
            weight_matrix = freq_distance.detach() ** self.alpha
            weight_matrix = weight_matrix / weight_matrix.max()
            
        # Weighted loss
        loss = (weight_matrix * freq_distance).mean()
        
        return self.loss_weight * loss
