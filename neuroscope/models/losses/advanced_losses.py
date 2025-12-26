"""
Advanced Loss Functions for Medical Image Domain Adaptation

This module implements state-of-the-art loss functions for improving
the quality and clinical validity of domain-translated brain MRI:

1. Perceptual Loss - VGG-based feature matching
2. Contrastive Loss - PatchNCE-style feature preservation  
3. Tumor Preservation Loss - Domain-specific loss for GBM imaging
4. Multi-Scale SSIM Loss - Structure-preserving loss
5. Gradient Correlation Loss - Edge preservation

These losses address the limitations of standard L1/L2 reconstruction
losses by capturing perceptual and structural similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
import torchvision.models as models


class VGGFeatureExtractor(nn.Module):
    """
    VGG-19 based feature extractor for perceptual loss computation.
    
    Extracts features from multiple layers to capture both low-level
    textures and high-level semantic information.
    
    Adapted for single-channel/grayscale images by replicating channels.
    """
    
    def __init__(self, 
                 layers: List[int] = [3, 8, 17, 26],  # relu1_2, relu2_2, relu3_4, relu4_4
                 requires_grad: bool = False):
        super().__init__()
        
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg.features[:max(layers) + 1]
        
        self.layer_indices = layers
        self.layer_names = [f'relu{i}' for i in range(len(layers))]
        
        if not requires_grad:
            for param in self.features.parameters():
                param.requires_grad = False
                
        # Normalization parameters (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input for VGG.
        Handles 1-channel (grayscale) and 4-channel (multi-modal MRI) inputs.
        """
        if x.shape[1] == 1:
            # Grayscale: replicate to 3 channels
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 4:
            # Multi-modal MRI: use first 3 channels or average
            x = x[:, :3, :, :]  # T1, T1ce, T2
        elif x.shape[1] != 3:
            raise ValueError(f"Expected 1, 3, or 4 channels, got {x.shape[1]}")
        
        # Scale from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        
        # Normalize
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        
        return x
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from multiple VGG layers.
        
        Args:
            x: Input tensor of shape (B, C, H, W), range [-1, 1]
            
        Returns:
            Dictionary mapping layer names to feature tensors
        """
        x = self._preprocess(x)
        
        features = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_indices:
                idx = self.layer_indices.index(i)
                features[self.layer_names[idx]] = x
                
        return features


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG features.
    
    Computes L1 distance between VGG features of input and target images,
    capturing perceptual similarity beyond pixel-level differences.
    
    Reference:
    - Johnson et al., "Perceptual Losses for Real-Time Style Transfer" (ECCV 2016)
    
    Args:
        layers: VGG layer indices to use (default: relu1_2, relu2_2, relu3_4, relu4_4)
        weights: Per-layer weights (default: equal weights)
        criterion: Loss criterion (default: L1)
    """
    
    def __init__(self,
                 layers: List[int] = [3, 8, 17, 26],
                 weights: Optional[List[float]] = None,
                 criterion: str = 'l1'):
        super().__init__()
        
        self.vgg = VGGFeatureExtractor(layers=layers)
        self.weights = weights or [1.0 / len(layers)] * len(layers)
        
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
            
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between x and y.
        
        Args:
            x: Generated/reconstructed image, shape (B, C, H, W)
            y: Target/original image, shape (B, C, H, W)
            
        Returns:
            Scalar perceptual loss
        """
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        
        loss = 0.0
        for i, (name, weight) in enumerate(zip(self.vgg.layer_names, self.weights)):
            loss += weight * self.criterion(x_features[name], y_features[name])
            
        return loss


class PatchNCELoss(nn.Module):
    """
    Patch-based Contrastive Loss (PatchNCE).
    
    Encourages corresponding patches in input and output to have similar
    features while being dissimilar to other patches. This preserves
    local structure during translation.
    
    Reference:
    - Park et al., "Contrastive Learning for Unpaired Image-to-Image Translation" (ECCV 2020)
    
    Args:
        num_patches: Number of patches to sample per image
        temperature: Temperature for softmax scaling
    """
    
    def __init__(self, num_patches: int = 256, temperature: float = 0.07):
        super().__init__()
        
        self.num_patches = num_patches
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self, 
                feat_q: torch.Tensor, 
                feat_k: torch.Tensor,
                feat_k_neg: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute PatchNCE loss.
        
        Args:
            feat_q: Query features (from generated image), shape (B, C, H, W)
            feat_k: Key features (from input image, positive), shape (B, C, H, W)
            feat_k_neg: Negative key features (optional, from other images)
            
        Returns:
            Scalar contrastive loss
        """
        B, C, H, W = feat_q.shape
        
        # Reshape to (B, C, N) where N = H*W
        feat_q = feat_q.view(B, C, -1)  # (B, C, N)
        feat_k = feat_k.view(B, C, -1)  # (B, C, N)
        
        # Sample patch locations
        N = H * W
        if N > self.num_patches:
            indices = torch.randperm(N, device=feat_q.device)[:self.num_patches]
            feat_q = feat_q[:, :, indices]  # (B, C, num_patches)
            feat_k = feat_k[:, :, indices]  # (B, C, num_patches)
        
        # L2 normalize
        feat_q = F.normalize(feat_q, dim=1)
        feat_k = F.normalize(feat_k, dim=1)
        
        # Positive logits: (B, num_patches)
        l_pos = torch.sum(feat_q * feat_k, dim=1) / self.temperature
        
        # Negative logits from same image, different locations
        # (B, num_patches, num_patches)
        l_neg = torch.bmm(feat_q.transpose(1, 2), feat_k) / self.temperature
        
        # Mask out positive pairs on diagonal
        mask = torch.eye(feat_q.shape[2], device=feat_q.device).bool()
        l_neg = l_neg.masked_fill(mask.unsqueeze(0), float('-inf'))
        
        # Logits: positive + negatives
        # (B, num_patches, 1 + num_patches - 1)
        logits = torch.cat([l_pos.unsqueeze(-1), l_neg], dim=-1)
        
        # Labels: positive is always index 0
        labels = torch.zeros(B, feat_q.shape[2], dtype=torch.long, device=feat_q.device)
        
        # Cross-entropy loss
        loss = self.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        
        return loss


class MultiScaleSSIMLoss(nn.Module):
    """
    Multi-Scale Structural Similarity (MS-SSIM) Loss.
    
    Computes SSIM at multiple scales to capture both fine details
    and coarse structures, which is important for medical imaging.
    
    Args:
        scales: Number of scales to use (default: 3)
        weights: Per-scale weights
    """
    
    def __init__(self, scales: int = 3, weights: Optional[List[float]] = None):
        super().__init__()
        
        self.scales = scales
        self.weights = weights or [0.5 ** i for i in range(scales)]
        self.weights = [w / sum(self.weights) for w in self.weights]  # Normalize
        
    def _ssim(self, x: torch.Tensor, y: torch.Tensor, 
              window_size: int = 11, C1: float = 0.01**2, C2: float = 0.03**2) -> torch.Tensor:
        """Compute SSIM between x and y."""
        
        # Create Gaussian window
        def gaussian_window(size, sigma=1.5):
            coords = torch.arange(size, dtype=torch.float32) - size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            return g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)
        
        window = gaussian_window(window_size).to(x.device)
        C = x.shape[1]
        window = window.expand(C, 1, window_size, window_size)
        
        mu_x = F.conv2d(x, window, padding=window_size//2, groups=C)
        mu_y = F.conv2d(y, window, padding=window_size//2, groups=C)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(x * x, window, padding=window_size//2, groups=C) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, window, padding=window_size//2, groups=C) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=window_size//2, groups=C) - mu_xy
        
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        return ssim_map.mean()
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute MS-SSIM loss.
        
        Args:
            x: Generated image, shape (B, C, H, W)
            y: Target image, shape (B, C, H, W)
            
        Returns:
            Scalar MS-SSIM loss (1 - MS-SSIM for minimization)
        """
        ms_ssim = 0.0
        
        for i in range(self.scales):
            ssim = self._ssim(x, y)
            ms_ssim += self.weights[i] * ssim
            
            if i < self.scales - 1:
                x = F.avg_pool2d(x, kernel_size=2)
                y = F.avg_pool2d(y, kernel_size=2)
        
        return 1 - ms_ssim


class GradientCorrelationLoss(nn.Module):
    """
    Gradient Correlation Loss for edge preservation.
    
    Ensures that edges and boundaries in the translated image
    match those in the original, which is crucial for preserving
    anatomical structures in brain MRI.
    
    Args:
        loss_type: 'correlation' or 'l1'
    """
    
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        
        self.loss_type = loss_type
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
    def _compute_gradients(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute image gradients using Sobel operators."""
        B, C, H, W = x.shape
        
        # Apply Sobel to each channel
        grad_x = []
        grad_y = []
        for c in range(C):
            gx = F.conv2d(x[:, c:c+1, :, :], self.sobel_x, padding=1)
            gy = F.conv2d(x[:, c:c+1, :, :], self.sobel_y, padding=1)
            grad_x.append(gx)
            grad_y.append(gy)
            
        return torch.cat(grad_x, dim=1), torch.cat(grad_y, dim=1)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient correlation loss.
        
        Args:
            x: Generated image, shape (B, C, H, W)
            y: Target image, shape (B, C, H, W)
            
        Returns:
            Scalar gradient loss
        """
        grad_x_x, grad_y_x = self._compute_gradients(x)
        grad_x_y, grad_y_y = self._compute_gradients(y)
        
        if self.loss_type == 'l1':
            loss = F.l1_loss(grad_x_x, grad_x_y) + F.l1_loss(grad_y_x, grad_y_y)
        else:
            # Correlation-based
            grad_x = torch.sqrt(grad_x_x ** 2 + grad_y_x ** 2 + 1e-8)
            grad_y = torch.sqrt(grad_x_y ** 2 + grad_y_y ** 2 + 1e-8)
            
            # Normalize
            grad_x = grad_x / (grad_x.mean() + 1e-8)
            grad_y = grad_y / (grad_y.mean() + 1e-8)
            
            loss = 1 - F.cosine_similarity(grad_x.view(-1), grad_y.view(-1), dim=0)
            
        return loss


class TumorPreservationLoss(nn.Module):
    """
    Tumor Region Preservation Loss.
    
    A domain-specific loss for glioblastoma MRI that emphasizes
    preservation of high-intensity regions (likely tumor areas)
    during domain translation.
    
    This is a NOVEL contribution specific to medical imaging GAN.
    
    Args:
        intensity_threshold: Threshold for identifying potential tumor regions
        weight_tumor: Weight for tumor region loss
        weight_normal: Weight for normal tissue loss
    """
    
    def __init__(self, 
                 intensity_threshold: float = 0.5,
                 weight_tumor: float = 3.0,
                 weight_normal: float = 1.0):
        super().__init__()
        
        self.intensity_threshold = intensity_threshold
        self.weight_tumor = weight_tumor
        self.weight_normal = weight_normal
        
    def forward(self, 
                x: torch.Tensor, 
                y: torch.Tensor,
                modality_idx: int = 1) -> torch.Tensor:
        """
        Compute tumor-weighted preservation loss.
        
        Args:
            x: Generated/reconstructed image, shape (B, 4, H, W)
            y: Original image, shape (B, 4, H, W)
            modality_idx: Which modality to use for tumor detection (default: 1 = T1ce)
            
        Returns:
            Scalar weighted loss
        """
        # Use T1ce (contrast-enhanced) for tumor detection
        reference = y[:, modality_idx, :, :]
        
        # Create tumor mask (high intensity regions)
        tumor_mask = (reference > self.intensity_threshold).float()
        normal_mask = 1 - tumor_mask
        
        # Expand mask to all channels
        tumor_mask = tumor_mask.unsqueeze(1).expand_as(x)
        normal_mask = normal_mask.unsqueeze(1).expand_as(x)
        
        # Compute weighted L1 loss
        tumor_loss = F.l1_loss(x * tumor_mask, y * tumor_mask)
        normal_loss = F.l1_loss(x * normal_mask, y * normal_mask)
        
        total_loss = self.weight_tumor * tumor_loss + self.weight_normal * normal_loss
        
        return total_loss


class CombinedAdvancedLoss(nn.Module):
    """
    Combined loss function with all advanced components.
    
    Aggregates multiple loss terms with learnable or fixed weights
    for comprehensive image quality optimization.
    
    Args:
        lambda_perceptual: Weight for perceptual loss
        lambda_contrastive: Weight for contrastive loss  
        lambda_ssim: Weight for MS-SSIM loss
        lambda_gradient: Weight for gradient loss
        lambda_tumor: Weight for tumor preservation loss
    """
    
    def __init__(self,
                 lambda_perceptual: float = 1.0,
                 lambda_contrastive: float = 0.5,
                 lambda_ssim: float = 1.0,
                 lambda_gradient: float = 0.5,
                 lambda_tumor: float = 1.0,
                 use_perceptual: bool = True,
                 use_contrastive: bool = True,
                 use_ssim: bool = True,
                 use_gradient: bool = True,
                 use_tumor: bool = True):
        super().__init__()
        
        self.lambdas = {
            'perceptual': lambda_perceptual,
            'contrastive': lambda_contrastive,
            'ssim': lambda_ssim,
            'gradient': lambda_gradient,
            'tumor': lambda_tumor,
        }
        
        self.losses = nn.ModuleDict()
        
        if use_perceptual:
            self.losses['perceptual'] = PerceptualLoss()
        if use_contrastive:
            self.losses['contrastive'] = PatchNCELoss()
        if use_ssim:
            self.losses['ssim'] = MultiScaleSSIMLoss()
        if use_gradient:
            self.losses['gradient'] = GradientCorrelationLoss()
        if use_tumor:
            self.losses['tumor'] = TumorPreservationLoss()
            
    def forward(self, 
                generated: torch.Tensor, 
                target: torch.Tensor,
                features_gen: Optional[torch.Tensor] = None,
                features_tar: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all loss terms.
        
        Args:
            generated: Generated image
            target: Target image
            features_gen: Features from generator (for contrastive loss)
            features_tar: Features from target (for contrastive loss)
            
        Returns:
            Dictionary of loss terms and total loss
        """
        loss_dict = {}
        total = 0.0
        
        for name, loss_fn in self.losses.items():
            if name == 'contrastive' and features_gen is not None:
                loss = loss_fn(features_gen, features_tar)
            else:
                loss = loss_fn(generated, target)
                
            loss_dict[name] = loss
            total += self.lambdas[name] * loss
            
        loss_dict['total'] = total
        
        return loss_dict
