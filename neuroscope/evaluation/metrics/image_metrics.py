"""
Image Quality Metrics.

Comprehensive collection of image quality and similarity metrics
for evaluating medical image harmonization.
"""

from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class MetricResult:
    """Result from metric computation."""
    value: float
    std: Optional[float] = None
    per_sample: Optional[List[float]] = None
    metadata: Dict = field(default_factory=dict)


class SSIM(nn.Module):
    """
    Structural Similarity Index (SSIM).
    
    Measures structural similarity between two images
    considering luminance, contrast, and structure.
    """
    
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channel: int = 1,
        size_average: bool = True,
        data_range: float = 1.0,
        K1: float = 0.01,
        K2: float = 0.03
    ):
        """
        Args:
            window_size: Size of Gaussian window
            sigma: Standard deviation of Gaussian
            channel: Number of channels
            size_average: Average over batch
            data_range: Data range (1.0 for normalized)
            K1: Stability constant for luminance
            K2: Stability constant for contrast
        """
        super().__init__()
        
        self.window_size = window_size
        self.sigma = sigma
        self.channel = channel
        self.size_average = size_average
        self.data_range = data_range
        
        # Stability constants
        self.C1 = (K1 * data_range) ** 2
        self.C2 = (K2 * data_range) ** 2
        
        # Create Gaussian window
        self.register_buffer('window', self._create_window(window_size, channel))
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create 2D Gaussian window."""
        gauss = torch.Tensor([
            math.exp(-(x - window_size // 2) ** 2 / (2 * self.sigma ** 2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        
        # Create 2D window
        window_2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
        
        return window
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute SSIM between x and y.
        
        Args:
            x: First image [B, C, H, W]
            y: Second image [B, C, H, W]
            mask: Optional mask for ROI computation
            
        Returns:
            SSIM value(s)
        """
        channel = x.size(1)
        
        if channel != self.channel:
            self.window = self._create_window(self.window_size, channel).to(x.device)
            self.channel = channel
        
        # Compute local means
        mu_x = F.conv2d(x, self.window, padding=self.window_size // 2, groups=channel)
        mu_y = F.conv2d(y, self.window, padding=self.window_size // 2, groups=channel)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        # Compute local variances
        sigma_x_sq = F.conv2d(
            x ** 2, self.window, padding=self.window_size // 2, groups=channel
        ) - mu_x_sq
        sigma_y_sq = F.conv2d(
            y ** 2, self.window, padding=self.window_size // 2, groups=channel
        ) - mu_y_sq
        sigma_xy = F.conv2d(
            x * y, self.window, padding=self.window_size // 2, groups=channel
        ) - mu_xy
        
        # SSIM formula
        numerator = (2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)
        denominator = (mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2)
        
        ssim_map = numerator / denominator
        
        if mask is not None:
            ssim_map = ssim_map * mask
            return ssim_map.sum() / mask.sum()
        
        if self.size_average:
            return ssim_map.mean()
        
        return ssim_map.mean(dim=[1, 2, 3])


class MultiScaleSSIM(nn.Module):
    """
    Multi-Scale Structural Similarity (MS-SSIM).
    
    Computes SSIM at multiple scales and combines them.
    """
    
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channel: int = 1,
        weights: Optional[List[float]] = None,
        data_range: float = 1.0
    ):
        """
        Args:
            window_size: Size of Gaussian window
            sigma: Standard deviation of Gaussian
            channel: Number of channels
            weights: Scale weights (default from MS-SSIM paper)
            data_range: Data range
        """
        super().__init__()
        
        self.window_size = window_size
        self.channel = channel
        self.weights = weights or [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        
        self.ssim = SSIM(
            window_size=window_size,
            sigma=sigma,
            channel=channel,
            size_average=False,
            data_range=data_range
        )
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MS-SSIM between x and y.
        
        Args:
            x: First image [B, C, H, W]
            y: Second image [B, C, H, W]
            
        Returns:
            MS-SSIM value
        """
        weights = torch.tensor(self.weights, device=x.device)
        levels = len(self.weights)
        
        msssim = []
        mcs = []
        
        for i in range(levels):
            ssim_val = self.ssim(x, y)
            msssim.append(ssim_val)
            
            if i < levels - 1:
                x = F.avg_pool2d(x, kernel_size=2)
                y = F.avg_pool2d(y, kernel_size=2)
        
        msssim = torch.stack(msssim, dim=-1)
        
        # Weighted product
        result = torch.prod(msssim ** weights, dim=-1)
        
        return result.mean()


class PSNR(nn.Module):
    """
    Peak Signal-to-Noise Ratio (PSNR).
    
    Measures reconstruction quality in decibels.
    """
    
    def __init__(self, data_range: float = 1.0, eps: float = 1e-8):
        """
        Args:
            data_range: Maximum value range
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.data_range = data_range
        self.eps = eps
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute PSNR between x and y.
        
        Args:
            x: First image
            y: Second image
            mask: Optional mask
            
        Returns:
            PSNR in dB
        """
        if mask is not None:
            mse = ((x - y) ** 2 * mask).sum() / mask.sum()
        else:
            mse = F.mse_loss(x, y)
        
        psnr = 10 * torch.log10(self.data_range ** 2 / (mse + self.eps))
        
        return psnr


class LPIPS(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS).
    
    Uses deep features for perceptual similarity.
    """
    
    def __init__(
        self,
        net: str = 'vgg',
        pretrained: bool = True,
        spatial: bool = False
    ):
        """
        Args:
            net: Network type ('vgg', 'alex', 'squeeze')
            pretrained: Use pretrained weights
            spatial: Return spatial map
        """
        super().__init__()
        
        self.net_type = net
        self.spatial = spatial
        
        # Build feature extractor
        if net == 'vgg':
            self._build_vgg()
        else:
            raise NotImplementedError(f"Network {net} not implemented")
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def _build_vgg(self):
        """Build VGG feature extractor."""
        from torchvision import models
        
        vgg = models.vgg16(weights='IMAGENET1K_V1' if True else None)
        
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.features.children())[:4]),   # relu1_2
            nn.Sequential(*list(vgg.features.children())[4:9]),  # relu2_2
            nn.Sequential(*list(vgg.features.children())[9:16]), # relu3_3
            nn.Sequential(*list(vgg.features.children())[16:23]),# relu4_3
            nn.Sequential(*list(vgg.features.children())[23:30]),# relu5_3
        ])
        
        # Learned weights for each layer
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.1) for _ in range(5)
        ])
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS distance.
        
        Args:
            x: First image [B, C, H, W]
            y: Second image [B, C, H, W]
            
        Returns:
            LPIPS distance
        """
        # Normalize to ImageNet range if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        
        feats_x = []
        feats_y = []
        
        hx, hy = x, y
        for slice_layer in self.slices:
            hx = slice_layer(hx)
            hy = slice_layer(hy)
            feats_x.append(hx)
            feats_y.append(hy)
        
        # Compute normalized differences
        diffs = []
        for i, (fx, fy) in enumerate(zip(feats_x, feats_y)):
            # Unit normalize
            fx_norm = fx / (fx.norm(dim=1, keepdim=True) + 1e-10)
            fy_norm = fy / (fy.norm(dim=1, keepdim=True) + 1e-10)
            
            diff = (fx_norm - fy_norm) ** 2
            diff = diff.mean(dim=[2, 3]) * self.weights[i]
            diffs.append(diff)
        
        return sum(diffs).mean()


class FID(nn.Module):
    """
    Fréchet Inception Distance (FID).
    
    Measures distance between feature distributions.
    """
    
    def __init__(self, dims: int = 2048):
        """
        Args:
            dims: Feature dimensions
        """
        super().__init__()
        self.dims = dims
        
        # Use InceptionV3 features
        from torchvision import models
        inception = models.inception_v3(weights='IMAGENET1K_V1', transform_input=False)
        
        # Remove final layers
        self.feature_extractor = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(3, 2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, 2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        for param in self.parameters():
            param.requires_grad = False
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from images."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Resize to Inception input size
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        features = self.feature_extractor(x)
        return features.view(features.size(0), -1)
    
    def forward(
        self,
        real: torch.Tensor,
        fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute FID between real and fake distributions.
        
        Args:
            real: Real images
            fake: Generated images
            
        Returns:
            FID score
        """
        with torch.no_grad():
            real_features = self._extract_features(real)
            fake_features = self._extract_features(fake)
        
        # Compute statistics
        mu_real = real_features.mean(dim=0)
        mu_fake = fake_features.mean(dim=0)
        
        sigma_real = torch.cov(real_features.T)
        sigma_fake = torch.cov(fake_features.T)
        
        # Fréchet distance
        diff = mu_real - mu_fake
        
        # Compute sqrt of product of covariances
        covmean = self._sqrtm(sigma_real @ sigma_fake)
        
        fid = diff @ diff + torch.trace(sigma_real + sigma_fake - 2 * covmean)
        
        return fid
    
    def _sqrtm(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute matrix square root."""
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        eigenvalues = torch.clamp(eigenvalues, min=0)
        return eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T


class TumorPreservationScore(nn.Module):
    """
    Tumor Preservation Score.
    
    Measures how well tumor regions are preserved
    during harmonization.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        region_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            threshold: Segmentation threshold
            region_weights: Weights for different tumor regions
        """
        super().__init__()
        
        self.threshold = threshold
        self.region_weights = region_weights or {
            'enhancing': 1.0,
            'necrotic': 0.8,
            'edema': 0.6
        }
    
    def forward(
        self,
        original: torch.Tensor,
        harmonized: torch.Tensor,
        segmentation: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute tumor preservation score.
        
        Args:
            original: Original image
            harmonized: Harmonized image
            segmentation: Tumor segmentation mask
            
        Returns:
            Preservation score
        """
        # Compute intensity correlation in tumor region
        tumor_mask = segmentation > self.threshold
        
        if tumor_mask.sum() == 0:
            return torch.tensor(1.0, device=original.device)
        
        orig_tumor = original[tumor_mask]
        harm_tumor = harmonized[tumor_mask]
        
        # Pearson correlation
        correlation = torch.corrcoef(
            torch.stack([orig_tumor.flatten(), harm_tumor.flatten()])
        )[0, 1]
        
        # SSIM in tumor region
        ssim = SSIM()(original * tumor_mask.float(), harmonized * tumor_mask.float())
        
        # Combined score
        score = 0.5 * (correlation + 1) + 0.5 * ssim
        
        return score


class TissueContrastRatio(nn.Module):
    """
    Tissue Contrast Ratio.
    
    Measures contrast between different tissue types.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        image: torch.Tensor,
        tissue_masks: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrast ratios between tissue types.
        
        Args:
            image: Input image
            tissue_masks: Dict of tissue type -> mask
            
        Returns:
            Dict of tissue pair -> contrast ratio
        """
        tissue_means = {}
        for name, mask in tissue_masks.items():
            if mask.sum() > 0:
                tissue_means[name] = (image * mask).sum() / mask.sum()
        
        contrasts = {}
        tissue_names = list(tissue_means.keys())
        
        for i, t1 in enumerate(tissue_names):
            for t2 in tissue_names[i + 1:]:
                m1, m2 = tissue_means[t1], tissue_means[t2]
                contrast = torch.abs(m1 - m2) / (m1 + m2 + 1e-8)
                contrasts[f'{t1}_vs_{t2}'] = contrast
        
        return contrasts


class VolumePreservation(nn.Module):
    """
    Volume Preservation Score.
    
    Ensures tumor volumes are preserved after harmonization.
    """
    
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
    
    def forward(
        self,
        original_seg: torch.Tensor,
        harmonized_seg: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute volume preservation score.
        
        Args:
            original_seg: Original segmentation
            harmonized_seg: Harmonized segmentation
            
        Returns:
            Volume preservation ratio
        """
        orig_volume = (original_seg > self.threshold).float().sum()
        harm_volume = (harmonized_seg > self.threshold).float().sum()
        
        if orig_volume == 0:
            return torch.tensor(1.0, device=original_seg.device)
        
        ratio = harm_volume / orig_volume
        
        # Penalize both shrinkage and expansion
        score = 1 - torch.abs(1 - ratio)
        
        return torch.clamp(score, 0, 1)


class ImageQualityMetrics:
    """
    Collection of image quality metrics.
    
    Provides convenient access to multiple metrics.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        self.ssim = SSIM().to(device)
        self.ms_ssim = MultiScaleSSIM().to(device)
        self.psnr = PSNR().to(device)
    
    def compute_all(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        """Compute all metrics."""
        x = x.to(self.device)
        y = y.to(self.device)
        
        return {
            'ssim': self.ssim(x, y).item(),
            'ms_ssim': self.ms_ssim(x, y).item(),
            'psnr': self.psnr(x, y).item()
        }


class MedicalImageMetrics:
    """
    Collection of medical image metrics.
    
    Specialized metrics for medical imaging.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        self.tumor_preservation = TumorPreservationScore().to(device)
        self.tissue_contrast = TissueContrastRatio().to(device)
        self.volume_preservation = VolumePreservation().to(device)
    
    def compute_all(
        self,
        original: torch.Tensor,
        harmonized: torch.Tensor,
        segmentation: torch.Tensor
    ) -> Dict[str, float]:
        """Compute all medical metrics."""
        original = original.to(self.device)
        harmonized = harmonized.to(self.device)
        segmentation = segmentation.to(self.device)
        
        return {
            'tumor_preservation': self.tumor_preservation(
                original, harmonized, segmentation
            ).item()
        }


def compute_ssim(x: torch.Tensor, y: torch.Tensor, **kwargs) -> float:
    """Convenience function for SSIM computation."""
    return SSIM(**kwargs)(x, y).item()


def compute_psnr(x: torch.Tensor, y: torch.Tensor, **kwargs) -> float:
    """Convenience function for PSNR computation."""
    return PSNR(**kwargs)(x, y).item()


def compute_all_metrics(
    x: torch.Tensor,
    y: torch.Tensor,
    segmentation: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute all available metrics.
    
    Args:
        x: First image
        y: Second image
        segmentation: Optional segmentation mask
        
    Returns:
        Dict of metric name -> value
    """
    metrics = {}
    
    # Image quality metrics
    iq_metrics = ImageQualityMetrics(device=x.device)
    metrics.update(iq_metrics.compute_all(x, y))
    
    # Medical metrics if segmentation provided
    if segmentation is not None:
        med_metrics = MedicalImageMetrics(device=x.device)
        metrics.update(med_metrics.compute_all(x, y, segmentation))
    
    return metrics
