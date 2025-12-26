"""
Medical Imaging-Specific Losses.

This module provides loss functions designed specifically for
medical image analysis, particularly MRI domain adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import math


class TumorPreservationLoss(nn.Module):
    """
    Tumor Preservation Loss.
    
    Ensures that tumor regions are preserved during domain translation.
    Critical for maintaining diagnostic integrity.
    
    Args:
        intensity_weight: Weight for intensity preservation
        boundary_weight: Weight for boundary preservation
        feature_weight: Weight for feature preservation
    """
    
    def __init__(
        self,
        intensity_weight: float = 1.0,
        boundary_weight: float = 0.5,
        feature_weight: float = 0.5
    ):
        super().__init__()
        
        self.intensity_weight = intensity_weight
        self.boundary_weight = boundary_weight
        self.feature_weight = feature_weight
        
        # Sobel filters for boundary detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
    def _compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge map using Sobel operator."""
        B, C, H, W = x.size()
        
        sobel_x = self.sobel_x.expand(C, 1, 3, 3).to(x.device, x.dtype)
        sobel_y = self.sobel_y.expand(C, 1, 3, 3).to(x.device, x.dtype)
        
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=C)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=C)
        
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
    def forward(
        self,
        input_img: torch.Tensor,
        output_img: torch.Tensor,
        tumor_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute tumor preservation loss.
        
        Args:
            input_img: Original image
            output_img: Translated image
            tumor_mask: Binary tumor segmentation mask (optional)
            
        Returns:
            Total tumor preservation loss
        """
        if tumor_mask is None:
            # Auto-detect high-intensity regions as potential tumor
            # This is a simplification; real implementation would use segmentation
            tumor_mask = (input_img > input_img.mean() + 2 * input_img.std()).float()
            
        # Intensity preservation in tumor regions
        intensity_loss = F.l1_loss(
            output_img * tumor_mask,
            input_img * tumor_mask
        )
        
        # Boundary preservation
        input_edges = self._compute_edges(input_img)
        output_edges = self._compute_edges(output_img)
        boundary_loss = F.l1_loss(
            output_edges * tumor_mask,
            input_edges * tumor_mask
        )
        
        # Feature preservation (local statistics)
        feature_loss = self._compute_feature_loss(input_img, output_img, tumor_mask)
        
        total_loss = (
            self.intensity_weight * intensity_loss +
            self.boundary_weight * boundary_loss +
            self.feature_weight * feature_loss
        )
        
        return total_loss
        
    def _compute_feature_loss(
        self,
        input_img: torch.Tensor,
        output_img: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute local feature statistics loss."""
        # Local mean and variance preservation
        mask_sum = mask.sum() + 1e-8
        
        input_mean = (input_img * mask).sum() / mask_sum
        output_mean = (output_img * mask).sum() / mask_sum
        
        input_var = ((input_img - input_mean) ** 2 * mask).sum() / mask_sum
        output_var = ((output_img - output_mean) ** 2 * mask).sum() / mask_sum
        
        mean_loss = (input_mean - output_mean) ** 2
        var_loss = (input_var - output_var) ** 2
        
        return mean_loss + var_loss


class RadiomicsPreservationLoss(nn.Module):
    """
    Radiomics Feature Preservation Loss.
    
    Preserves radiomic features important for clinical analysis.
    Includes texture features, shape features, and intensity statistics.
    
    Args:
        texture_weight: Weight for texture features
        shape_weight: Weight for shape features
        intensity_weight: Weight for intensity statistics
    """
    
    def __init__(
        self,
        texture_weight: float = 1.0,
        shape_weight: float = 0.5,
        intensity_weight: float = 1.0
    ):
        super().__init__()
        
        self.texture_weight = texture_weight
        self.shape_weight = shape_weight
        self.intensity_weight = intensity_weight
        
    def forward(
        self,
        input_img: torch.Tensor,
        output_img: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute radiomics preservation loss.
        
        Returns dict with individual loss components.
        """
        if roi_mask is None:
            roi_mask = torch.ones_like(input_img)
            
        losses = {}
        
        # Intensity statistics
        losses['intensity'] = self._intensity_statistics_loss(input_img, output_img, roi_mask)
        
        # Texture features (GLCM-inspired)
        losses['texture'] = self._texture_loss(input_img, output_img, roi_mask)
        
        # Shape preservation
        losses['shape'] = self._shape_loss(input_img, output_img, roi_mask)
        
        # Combined loss
        total = (
            self.intensity_weight * losses['intensity'] +
            self.texture_weight * losses['texture'] +
            self.shape_weight * losses['shape']
        )
        
        losses['total'] = total
        
        return losses
        
    def _intensity_statistics_loss(
        self,
        input_img: torch.Tensor,
        output_img: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Preserve first-order intensity statistics."""
        mask_sum = mask.sum() + 1e-8
        
        # Mean
        in_mean = (input_img * mask).sum() / mask_sum
        out_mean = (output_img * mask).sum() / mask_sum
        
        # Variance
        in_var = ((input_img - in_mean) ** 2 * mask).sum() / mask_sum
        out_var = ((output_img - out_mean) ** 2 * mask).sum() / mask_sum
        
        # Skewness
        in_skew = ((input_img - in_mean) ** 3 * mask).sum() / (mask_sum * (in_var ** 1.5 + 1e-8))
        out_skew = ((output_img - out_mean) ** 3 * mask).sum() / (mask_sum * (out_var ** 1.5 + 1e-8))
        
        # Kurtosis
        in_kurt = ((input_img - in_mean) ** 4 * mask).sum() / (mask_sum * (in_var ** 2 + 1e-8))
        out_kurt = ((output_img - out_mean) ** 4 * mask).sum() / (mask_sum * (out_var ** 2 + 1e-8))
        
        loss = (
            (in_mean - out_mean) ** 2 +
            (in_var - out_var) ** 2 +
            (in_skew - out_skew) ** 2 +
            (in_kurt - out_kurt) ** 2
        )
        
        return loss
        
    def _texture_loss(
        self,
        input_img: torch.Tensor,
        output_img: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Preserve texture features using local gradient statistics."""
        # Local gradient patterns
        kernel_size = 3
        
        # Compute local variance (approximates texture)
        def local_variance(x):
            x_unfold = F.unfold(x, kernel_size, padding=kernel_size // 2)
            local_mean = x_unfold.mean(dim=1, keepdim=True)
            local_var = ((x_unfold - local_mean) ** 2).mean(dim=1)
            B, C_H_W = local_var.shape[0], local_var.shape[1]
            H = W = int(math.sqrt(C_H_W))
            return local_var.view(B, 1, H, W)
            
        in_texture = local_variance(input_img)
        out_texture = local_variance(output_img)
        
        # Resize mask if needed
        if mask.shape[-2:] != in_texture.shape[-2:]:
            mask = F.interpolate(mask, size=in_texture.shape[-2:], mode='nearest')
            
        return F.l1_loss(out_texture * mask, in_texture * mask)
        
    def _shape_loss(
        self,
        input_img: torch.Tensor,
        output_img: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Preserve shape boundaries."""
        # Use Laplacian for shape
        laplacian_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=input_img.dtype, device=input_img.device
        ).view(1, 1, 3, 3)
        
        C = input_img.size(1)
        laplacian_kernel = laplacian_kernel.expand(C, 1, 3, 3)
        
        in_lap = F.conv2d(input_img, laplacian_kernel, padding=1, groups=C)
        out_lap = F.conv2d(output_img, laplacian_kernel, padding=1, groups=C)
        
        return F.l1_loss(out_lap * mask, in_lap * mask)


class ModalityConsistencyLoss(nn.Module):
    """
    Multi-Modal Consistency Loss.
    
    Ensures consistency across different MRI modalities during translation.
    
    Args:
        modality_weights: Dictionary mapping modality names to weights
    """
    
    def __init__(self, modality_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        # Default weights for standard MRI modalities
        self.modality_weights = modality_weights or {
            't1': 1.0,
            't1ce': 1.5,  # Higher weight for contrast-enhanced
            't2': 1.0,
            'flair': 1.2  # Higher weight for FLAIR (tumor visibility)
        }
        
    def forward(
        self,
        input_modalities: Dict[str, torch.Tensor],
        output_modalities: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute modality consistency loss.
        
        Args:
            input_modalities: Dict of input modality tensors
            output_modalities: Dict of output modality tensors
            
        Returns:
            Weighted modality consistency loss
        """
        total_loss = 0.0
        total_weight = 0.0
        
        for modality, weight in self.modality_weights.items():
            if modality in input_modalities and modality in output_modalities:
                loss = F.l1_loss(output_modalities[modality], input_modalities[modality])
                total_loss += weight * loss
                total_weight += weight
                
        return total_loss / (total_weight + 1e-8)


class AnatomicalConsistencyLoss(nn.Module):
    """
    Anatomical Structure Preservation Loss.
    
    Preserves anatomical structures using learned or handcrafted features.
    
    Args:
        structure_encoder: Optional pretrained anatomy encoder
    """
    
    def __init__(self, structure_encoder: Optional[nn.Module] = None):
        super().__init__()
        
        self.structure_encoder = structure_encoder
        
        if structure_encoder is not None:
            for param in self.structure_encoder.parameters():
                param.requires_grad = False
                
    def forward(
        self,
        input_img: torch.Tensor,
        output_img: torch.Tensor,
        anatomical_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute anatomical consistency loss."""
        if self.structure_encoder is not None:
            # Use learned features
            with torch.no_grad():
                input_features = self.structure_encoder(input_img)
            output_features = self.structure_encoder(output_img)
            
            loss = F.l1_loss(output_features, input_features)
        else:
            # Use gradient-based structure preservation
            loss = self._gradient_structure_loss(input_img, output_img, anatomical_mask)
            
        return loss
        
    def _gradient_structure_loss(
        self,
        input_img: torch.Tensor,
        output_img: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Preserve gradient structures."""
        # Second-order gradients for structure
        def second_order_gradient(x):
            dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            dy = x[:, :, 1:, :] - x[:, :, :-1, :]
            
            dxx = dx[:, :, :, 1:] - dx[:, :, :, :-1]
            dyy = dy[:, :, 1:, :] - dy[:, :, :-1, :]
            
            return dxx, dyy
            
        in_dxx, in_dyy = second_order_gradient(input_img)
        out_dxx, out_dyy = second_order_gradient(output_img)
        
        loss = F.l1_loss(out_dxx, in_dxx) + F.l1_loss(out_dyy, in_dyy)
        
        return loss


class ContrastEnhancementLoss(nn.Module):
    """
    Contrast Enhancement Preservation Loss.
    
    Specifically for T1ce modality - preserves contrast agent effects.
    
    Args:
        enhancement_threshold: Threshold for detecting enhanced regions
    """
    
    def __init__(self, enhancement_threshold: float = 0.7):
        super().__init__()
        self.enhancement_threshold = enhancement_threshold
        
    def forward(
        self,
        input_t1: torch.Tensor,
        input_t1ce: torch.Tensor,
        output_t1: torch.Tensor,
        output_t1ce: torch.Tensor
    ) -> torch.Tensor:
        """
        Preserve contrast enhancement patterns.
        
        Computes enhancement map as T1ce - T1 and ensures consistency.
        """
        # Enhancement maps
        input_enhancement = input_t1ce - input_t1
        output_enhancement = output_t1ce - output_t1
        
        # Detect enhanced regions
        enhancement_mask = (input_enhancement > self.enhancement_threshold * input_enhancement.max()).float()
        
        # Preserve enhancement in detected regions
        loss = F.l1_loss(
            output_enhancement * enhancement_mask,
            input_enhancement * enhancement_mask
        )
        
        return loss


class NormalizedCrossCorrelationLoss(nn.Module):
    """
    Normalized Cross-Correlation (NCC) Loss.
    
    Common in medical image registration, useful for intensity-invariant matching.
    
    Args:
        window_size: Size of local window for NCC computation
    """
    
    def __init__(self, window_size: int = 9):
        super().__init__()
        self.window_size = window_size
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute NCC loss (1 - NCC for minimization)."""
        # Local means
        ndims = len(pred.shape) - 2
        
        if ndims == 2:
            sum_filt = torch.ones(1, 1, self.window_size, self.window_size, device=pred.device)
        else:
            sum_filt = torch.ones(1, 1, self.window_size, self.window_size, self.window_size, device=pred.device)
            
        pad_no = self.window_size // 2
        
        if ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)
            
        conv_fn = F.conv2d if ndims == 2 else F.conv3d
        
        # Cross correlation
        pred_flat = pred.view(pred.shape[0], 1, *pred.shape[2:])
        target_flat = target.view(target.shape[0], 1, *target.shape[2:])
        
        pred_sum = conv_fn(pred_flat, sum_filt, stride=stride, padding=padding)
        target_sum = conv_fn(target_flat, sum_filt, stride=stride, padding=padding)
        pred2_sum = conv_fn(pred_flat ** 2, sum_filt, stride=stride, padding=padding)
        target2_sum = conv_fn(target_flat ** 2, sum_filt, stride=stride, padding=padding)
        pred_target_sum = conv_fn(pred_flat * target_flat, sum_filt, stride=stride, padding=padding)
        
        win_size = self.window_size ** ndims
        
        pred_mean = pred_sum / win_size
        target_mean = target_sum / win_size
        
        cross = pred_target_sum - target_mean * pred_sum - pred_mean * target_sum + pred_mean * target_mean * win_size
        pred_var = pred2_sum - 2 * pred_mean * pred_sum + pred_mean ** 2 * win_size
        target_var = target2_sum - 2 * target_mean * target_sum + target_mean ** 2 * win_size
        
        cc = cross ** 2 / (pred_var * target_var + 1e-8)
        
        return 1 - cc.mean()
