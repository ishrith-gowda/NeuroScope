"""
advanced loss functions for medical image domain adaptation

this module implements state-of-the-art loss functions for improving
the quality and clinical validity of domain-translated brain mri:

1. perceptual loss - vgg-based feature matching
2. contrastive loss - patchnce-style feature preservation  
3. tumor preservation loss - domain-specific loss for gbm imaging
4. multi-scale ssim loss - structure-preserving loss
5. gradient correlation loss - edge preservation

these losses address the limitations of standard l1/l2 reconstruction
losses by capturing perceptual and structural similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
import torchvision.models as models


class VGGFeatureExtractor(nn.Module):
    """
    vgg-19 based feature extractor for perceptual loss computation.
    
    extracts features from multiple layers to capture both low-level
    textures and high-level semantic information.
    
    adapted for single-channel/grayscale images by replicating channels.
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
                
        # normalization parameters (imagenet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        preprocess input for vgg.
        handles 1-channel (grayscale) and 4-channel (multi-modal mri) inputs.
        """
        if x.shape[1] == 1:
            # grayscale: replicate to 3 channels
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 4:
            # multi-modal mri: use first 3 channels or average
            x = x[:, :3, :, :]  # t1, t1ce, t2
        elif x.shape[1] != 3:
            raise ValueError(f"Expected 1, 3, or 4 channels, got {x.shape[1]}")
        
        # scale from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        
        # normalize
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        
        return x
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        extract features from multiple vgg layers.
        
        args:
            x: input tensor of shape (b, c, h, w), range [-1, 1]
            
        returns:
            dictionary mapping layer names to feature tensors
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
    perceptual loss using vgg features.
    
    computes l1 distance between vgg features of input and target images,
    capturing perceptual similarity beyond pixel-level differences.
    
    reference:
    - johnson et al., "perceptual losses for real-time style transfer" (eccv 2016)
    
    args:
        layers: vgg layer indices to use (default: relu1_2, relu2_2, relu3_4, relu4_4)
        weights: per-layer weights (default: equal weights)
        criterion: loss criterion (default: l1)
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
        compute perceptual loss between x and y.
        
        args:
            x: generated/reconstructed image, shape (b, c, h, w)
            y: target/original image, shape (b, c, h, w)
            
        returns:
            scalar perceptual loss
        """
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        
        loss = 0.0
        for i, (name, weight) in enumerate(zip(self.vgg.layer_names, self.weights)):
            loss += weight * self.criterion(x_features[name], y_features[name])
            
        return loss


class PatchNCELoss(nn.Module):
    """
    patch-based contrastive loss (patchnce).
    
    encourages corresponding patches in input and output to have similar
    features while being dissimilar to other patches. this preserves
    local structure during translation.
    
    reference:
    - park et al., "contrastive learning for unpaired image-to-image translation" (eccv 2020)
    
    args:
        num_patches: number of patches to sample per image
        temperature: temperature for softmax scaling
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
        compute patchnce loss.
        
        args:
            feat_q: query features (from generated image), shape (b, c, h, w)
            feat_k: key features (from input image, positive), shape (b, c, h, w)
            feat_k_neg: negative key features (optional, from other images)
            
        returns:
            scalar contrastive loss
        """
        B, C, H, W = feat_q.shape
        
        # reshape to (b, c, n) where n = h*w
        feat_q = feat_q.view(B, C, -1)  # (b, c, n)
        feat_k = feat_k.view(B, C, -1)  # (b, c, n)
        
        # sample patch locations
        N = H * W
        if N > self.num_patches:
            indices = torch.randperm(N, device=feat_q.device)[:self.num_patches]
            feat_q = feat_q[:, :, indices]  # (b, c, num_patches)
            feat_k = feat_k[:, :, indices]  # (b, c, num_patches)
        
        # l2 normalize
        feat_q = F.normalize(feat_q, dim=1)
        feat_k = F.normalize(feat_k, dim=1)
        
        # positive logits: (b, num_patches)
        l_pos = torch.sum(feat_q * feat_k, dim=1) / self.temperature
        
        # negative logits from same image, different locations
        # (b, num_patches, num_patches)
        l_neg = torch.bmm(feat_q.transpose(1, 2), feat_k) / self.temperature
        
        # mask out positive pairs on diagonal
        mask = torch.eye(feat_q.shape[2], device=feat_q.device).bool()
        l_neg = l_neg.masked_fill(mask.unsqueeze(0), float('-inf'))
        
        # logits: positive + negatives
        # (b, num_patches, 1 + num_patches - 1)
        logits = torch.cat([l_pos.unsqueeze(-1), l_neg], dim=-1)
        
        # labels: positive is always index 0
        labels = torch.zeros(B, feat_q.shape[2], dtype=torch.long, device=feat_q.device)
        
        # cross-entropy loss
        loss = self.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        
        return loss


class MultiScaleSSIMLoss(nn.Module):
    """
    multi-scale structural similarity (ms-ssim) loss.
    
    computes ssim at multiple scales to capture both fine details
    and coarse structures, which is important for medical imaging.
    
    args:
        scales: number of scales to use (default: 3)
        weights: per-scale weights
    """
    
    def __init__(self, scales: int = 3, weights: Optional[List[float]] = None):
        super().__init__()
        
        self.scales = scales
        self.weights = weights or [0.5 ** i for i in range(scales)]
        self.weights = [w / sum(self.weights) for w in self.weights]  # normalize
        
    def _ssim(self, x: torch.Tensor, y: torch.Tensor, 
              window_size: int = 11, C1: float = 0.01**2, C2: float = 0.03**2) -> torch.Tensor:
        """compute ssim between x and y."""
        
        # create gaussian window
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
        compute ms-ssim loss.
        
        args:
            x: generated image, shape (b, c, h, w)
            y: target image, shape (b, c, h, w)
            
        returns:
            scalar ms-ssim loss (1 - ms-ssim for minimization)
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
    gradient correlation loss for edge preservation.
    
    ensures that edges and boundaries in the translated image
    match those in the original, which is crucial for preserving
    anatomical structures in brain mri.
    
    args:
        loss_type: 'correlation' or 'l1'
    """
    
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        
        self.loss_type = loss_type
        
        # sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
    def _compute_gradients(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """compute image gradients using sobel operators."""
        B, C, H, W = x.shape
        
        # apply sobel to each channel
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
        compute gradient correlation loss.
        
        args:
            x: generated image, shape (b, c, h, w)
            y: target image, shape (b, c, h, w)
            
        returns:
            scalar gradient loss
        """
        grad_x_x, grad_y_x = self._compute_gradients(x)
        grad_x_y, grad_y_y = self._compute_gradients(y)
        
        if self.loss_type == 'l1':
            loss = F.l1_loss(grad_x_x, grad_x_y) + F.l1_loss(grad_y_x, grad_y_y)
        else:
            # correlation-based
            grad_x = torch.sqrt(grad_x_x ** 2 + grad_y_x ** 2 + 1e-8)
            grad_y = torch.sqrt(grad_x_y ** 2 + grad_y_y ** 2 + 1e-8)
            
            # normalize
            grad_x = grad_x / (grad_x.mean() + 1e-8)
            grad_y = grad_y / (grad_y.mean() + 1e-8)
            
            loss = 1 - F.cosine_similarity(grad_x.view(-1), grad_y.view(-1), dim=0)
            
        return loss


class TumorPreservationLoss(nn.Module):
    """
    tumor region preservation loss.
    
    a domain-specific loss for glioblastoma mri that emphasizes
    preservation of high-intensity regions (likely tumor areas)
    during domain translation.
    
    this is a novel contribution specific to medical imaging gan.
    
    args:
        intensity_threshold: threshold for identifying potential tumor regions
        weight_tumor: weight for tumor region loss
        weight_normal: weight for normal tissue loss
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
        compute tumor-weighted preservation loss.
        
        args:
            x: generated/reconstructed image, shape (b, 4, h, w)
            y: original image, shape (b, 4, h, w)
            modality_idx: which modality to use for tumor detection (default: 1 = t1ce)
            
        returns:
            scalar weighted loss
        """
        # use t1ce (contrast-enhanced) for tumor detection
        reference = y[:, modality_idx, :, :]
        
        # create tumor mask (high intensity regions)
        tumor_mask = (reference > self.intensity_threshold).float()
        normal_mask = 1 - tumor_mask
        
        # expand mask to all channels
        tumor_mask = tumor_mask.unsqueeze(1).expand_as(x)
        normal_mask = normal_mask.unsqueeze(1).expand_as(x)
        
        # compute weighted l1 loss
        tumor_loss = F.l1_loss(x * tumor_mask, y * tumor_mask)
        normal_loss = F.l1_loss(x * normal_mask, y * normal_mask)
        
        total_loss = self.weight_tumor * tumor_loss + self.weight_normal * normal_loss
        
        return total_loss


class CombinedAdvancedLoss(nn.Module):
    """
    combined loss function with all advanced components.
    
    aggregates multiple loss terms with learnable or fixed weights
    for comprehensive image quality optimization.
    
    args:
        lambda_perceptual: weight for perceptual loss
        lambda_contrastive: weight for contrastive loss  
        lambda_ssim: weight for ms-ssim loss
        lambda_gradient: weight for gradient loss
        lambda_tumor: weight for tumor preservation loss
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
        compute all loss terms.
        
        args:
            generated: generated image
            target: target image
            features_gen: features from generator (for contrastive loss)
            features_tar: features from target (for contrastive loss)
            
        returns:
            dictionary of loss terms and total loss
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
