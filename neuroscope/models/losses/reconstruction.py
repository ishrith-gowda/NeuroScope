"""
reconstruction losses for image-to-image translation.

this module provides various reconstruction loss formulations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class L1Loss(nn.Module):
    """l1 (mean absolute error) loss."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.loss = nn.L1Loss(reduction=reduction)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class L2Loss(nn.Module):
    """l2 (mean squared error) loss."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class SSIMLoss(nn.Module):
    """
    structural similarity index (ssim) loss.
    
    ssim measures structural similarity between images.
    loss = 1 - ssim
    
    args:
        window_size: size of the gaussian window
        sigma: standard deviation of gaussian
        size_average: whether to average over batch
        channel: number of channels
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
        
        # create gaussian window
        gaussian = self._create_gaussian_window(window_size, sigma)
        self.register_buffer('window', gaussian.unsqueeze(0).unsqueeze(0))
        
    def _create_gaussian_window(self, window_size: int, sigma: float) -> torch.Tensor:
        """create 2d gaussian window."""
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
        compute ssim loss.
        
        args:
            pred: predicted tensor [b, c, h, w]
            target: target tensor [b, c, h, w]
            
        returns:
            ssim loss (1 - ssim)
        """
        channel = pred.size(1)
        
        # expand window to all channels
        window = self.window.expand(channel, 1, -1, -1).to(pred.device, pred.dtype)
        
        # compute means
        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=channel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # compute variances and covariance
        sigma1_sq = F.conv2d(pred ** 2, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        
        # constants for stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # ssim formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
                   
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean([1, 2, 3])


class MultiScaleSSIMLoss(nn.Module):
    """
    multi-scale ssim (ms-ssim) loss.
    
    computes ssim at multiple scales for better structure preservation.
    
    args:
        window_size: size of the gaussian window
        levels: number of scales
    """
    
    def __init__(self, window_size: int = 11, levels: int = 5):
        super().__init__()
        
        self.levels = levels
        self.ssim = SSIMLoss(window_size=window_size)
        
        # default weights for each level
        self.weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333][:levels]
        self.weights = torch.tensor(self.weights)
        self.weights = self.weights / self.weights.sum()
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """compute ms-ssim loss."""
        msssim_values = []
        
        for i in range(self.levels):
            ssim_val = 1 - self.ssim(pred, target)  # convert loss back to ssim
            msssim_values.append(ssim_val)
            
            if i < self.levels - 1:
                # downsample
                pred = F.avg_pool2d(pred, 2)
                target = F.avg_pool2d(target, 2)
                
        # weighted product
        msssim = torch.stack(msssim_values)
        weights = self.weights.to(msssim.device)
        msssim = (msssim ** weights.view(-1, 1)).prod(dim=0).mean()
        
        return 1 - msssim


class GradientLoss(nn.Module):
    """
    gradient (edge) loss for preserving image structure.
    
    computes l1/l2 loss on image gradients.
    
    args:
        loss_type: 'l1' or 'l2'
    """
    
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        
        self.loss_type = loss_type
        
        # sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
    def _compute_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """compute image gradient magnitude."""
        B, C, H, W = x.size()
        
        # expand sobel filters to all channels
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
        """compute gradient loss."""
        pred_grad = self._compute_gradient(pred)
        target_grad = self._compute_gradient(target)
        
        if self.loss_type == 'l1':
            return F.l1_loss(pred_grad, target_grad)
        else:
            return F.mse_loss(pred_grad, target_grad)


class CharbonnierLoss(nn.Module):
    """
    charbonnier loss (smooth l1).
    
    l = sqrt((x - y)² + ε²)
    
    more robust to outliers than l2, smoother than l1.
    
    args:
        epsilon: small constant for numerical stability
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
    focal frequency loss for frequency-domain supervision.
    
    emphasizes hard frequency components using focal weighting.
    
    args:
        loss_weight: weight for the loss
        alpha: focal weight parameter
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
        """compute focal frequency loss."""
        # 2d fft
        pred_freq = torch.fft.fft2(pred, norm='ortho')
        target_freq = torch.fft.fft2(target, norm='ortho')
        
        # frequency distance
        freq_distance = torch.abs(pred_freq - target_freq)
        
        # focal weight (emphasize hard frequencies)
        if weight_matrix is None:
            weight_matrix = freq_distance.detach() ** self.alpha
            weight_matrix = weight_matrix / weight_matrix.max()
            
        # weighted loss
        loss = (weight_matrix * freq_distance).mean()
        
        return self.loss_weight * loss
