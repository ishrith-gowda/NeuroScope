"""
volumetric loss functions for 3d brain mri harmonization.

this module implements 3d-specific loss functions optimized for
volumetric medical image processing:

- volumetriccycleconsistencyloss: 3d cycle consistency with spatial awareness
- volumetricperceptualloss: 3d perceptual loss using medicalnet features
- volumetricssimloss: 3d structural similarity index
- volumetricgradientloss: 3d gradient matching for edge preservation
- anatomicalconsistencyloss: preserves brain structure during harmonization
- volumetricnceloss: 3d contrastive learning for unpaired translation
- tissuepreservationloss: preserves gm/wm/csf boundaries
"""

from typing import Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VolumetricSSIM(nn.Module):
    """
    3d structural similarity index (ssim) for volumetric data.
    
    extends ssim to 3d using volumetric gaussian windows.
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
        initialize 3d ssim module.
        
        args:
            window_size: size of the gaussian window
            sigma: standard deviation of gaussian
            channel: number of input channels
            size_average: if true, return mean ssim
            data_range: dynamic range of input (1.0 for normalized)
            k1, k2: stability constants
        """
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = channel
        self.size_average = size_average
        self.data_range = data_range
        
        # ssim constants
        self.C1 = (K1 * data_range) ** 2
        self.C2 = (K2 * data_range) ** 2
        
        # create 3d gaussian window
        self.register_buffer('window', self._create_3d_window())
    
    def _create_3d_window(self) -> Tensor:
        """create 3d gaussian window."""
        # 1d gaussian
        gauss_1d = torch.tensor([
            math.exp(-(x - self.window_size // 2) ** 2 / (2 * self.sigma ** 2))
            for x in range(self.window_size)
        ], dtype=torch.float32)
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        # create 3d gaussian by outer product
        gauss_2d = gauss_1d.outer(gauss_1d)
        gauss_3d = gauss_2d.unsqueeze(-1) * gauss_1d.view(1, 1, -1)
        
        # reshape for conv3d: (out_channels, in_channels/groups, d, h, w)
        window = gauss_3d.unsqueeze(0).unsqueeze(0)
        window = window.expand(self.channel, 1, -1, -1, -1).contiguous()
        
        return window
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        compute 3d ssim between x and y.
        
        args:
            x: input tensor (b, c, d, h, w)
            y: target tensor (b, c, d, h, w)
            
        returns:
            ssim value (scalar if size_average, else per-sample)
        """
        channel = x.size(1)
        
        if channel != self.channel:
            window = self._create_3d_window()
            window = window.to(x.device)
            self.channel = channel
        else:
            window = self.window
        
        # ensure window is on same device
        window = window.to(x.device)
        
        # compute means
        mu_x = F.conv3d(x, window, padding=self.window_size // 2, groups=channel)
        mu_y = F.conv3d(y, window, padding=self.window_size // 2, groups=channel)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        # compute variances
        sigma_x_sq = F.conv3d(x * x, window, padding=self.window_size // 2, groups=channel) - mu_x_sq
        sigma_y_sq = F.conv3d(y * y, window, padding=self.window_size // 2, groups=channel) - mu_y_sq
        sigma_xy = F.conv3d(x * y, window, padding=self.window_size // 2, groups=channel) - mu_xy
        
        # ssim formula
        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / \
                   ((mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(dim=[1, 2, 3, 4])


class VolumetricMultiScaleSSIM(nn.Module):
    """
    multi-scale 3d ssim (ms-ssim) for volumetric data.
    
    computes ssim at multiple scales using average pooling.
    """
    
    def __init__(
        self,
        window_size: int = 11,
        channel: int = 1,
        weights: Optional[List[float]] = None,
        data_range: float = 1.0
    ):
        """
        initialize ms-ssim 3d.
        
        args:
            window_size: size of gaussian window
            channel: number of input channels
            weights: weights for each scale (default: 5 scales)
            data_range: dynamic range of input
        """
        super().__init__()
        
        self.weights = weights or [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.levels = len(self.weights)
        
        self.ssim = VolumetricSSIM(
            window_size=window_size,
            channel=channel,
            size_average=False,
            data_range=data_range
        )
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        compute ms-ssim between x and y.
        
        args:
            x: input tensor (b, c, d, h, w)
            y: target tensor (b, c, d, h, w)
            
        returns:
            ms-ssim value
        """
        weights = torch.tensor(self.weights, device=x.device)
        
        msssim_vals = []
        for i in range(self.levels):
            ssim_val = self.ssim(x, y)
            msssim_vals.append(ssim_val)
            
            if i < self.levels - 1:
                x = F.avg_pool3d(x, kernel_size=2, stride=2)
                y = F.avg_pool3d(y, kernel_size=2, stride=2)
        
        msssim = torch.stack(msssim_vals)
        msssim = (msssim ** weights.view(-1, 1)).prod(dim=0)
        
        return msssim.mean()


class VolumetricCycleConsistencyLoss(nn.Module):
    """
    volumetric cycle consistency loss with spatial awareness.
    
    extends cycle consistency to 3d with optional multi-scale
    and anatomical region weighting.
    """
    
    def __init__(
        self,
        loss_type: str = 'l1',  # 'l1', 'l2', 'huber', 'ssim', 'combined'
        lambda_ssim: float = 0.5,
        lambda_l1: float = 0.5,
        multi_scale: bool = True,
        num_scales: int = 3,
        spatial_weights: bool = False
    ):
        """
        initialize volumetric cycle consistency loss.
        
        args:
            loss_type: type of base loss function
            lambda_ssim: weight for ssim component
            lambda_l1: weight for l1 component
            multi_scale: use multi-scale computation
            num_scales: number of scales for multi-scale
            spatial_weights: apply spatial attention weighting
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.lambda_ssim = lambda_ssim
        self.lambda_l1 = lambda_l1
        self.multi_scale = multi_scale
        self.num_scales = num_scales
        self.spatial_weights = spatial_weights
        
        if 'ssim' in loss_type or loss_type == 'combined':
            self.ssim = VolumetricSSIM()
        
        # scale weights for multi-scale loss
        if multi_scale:
            self.scale_weights = [1.0 / (2 ** i) for i in range(num_scales)]
            total = sum(self.scale_weights)
            self.scale_weights = [w / total for w in self.scale_weights]
    
    def forward(
        self,
        real: Tensor,
        reconstructed: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        compute cycle consistency loss.
        
        args:
            real: original input (b, c, d, h, w)
            reconstructed: reconstructed input after cycle (b, c, d, h, w)
            mask: optional spatial weight mask
            
        returns:
            cycle consistency loss value
        """
        if self.multi_scale:
            return self._multi_scale_loss(real, reconstructed, mask)
        else:
            return self._single_scale_loss(real, reconstructed, mask)
    
    def _single_scale_loss(
        self,
        real: Tensor,
        reconstructed: Tensor,
        mask: Optional[Tensor]
    ) -> Tensor:
        """compute single-scale loss."""
        if self.loss_type == 'l1':
            loss = F.l1_loss(reconstructed, real, reduction='none')
        elif self.loss_type == 'l2':
            loss = F.mse_loss(reconstructed, real, reduction='none')
        elif self.loss_type == 'huber':
            loss = F.huber_loss(reconstructed, real, delta=1.0, reduction='none')
        elif self.loss_type == 'ssim':
            return 1 - self.ssim(reconstructed, real)
        elif self.loss_type == 'combined':
            l1_loss = F.l1_loss(reconstructed, real)
            ssim_loss = 1 - self.ssim(reconstructed, real)
            return self.lambda_l1 * l1_loss + self.lambda_ssim * ssim_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        
        return loss.mean()
    
    def _multi_scale_loss(
        self,
        real: Tensor,
        reconstructed: Tensor,
        mask: Optional[Tensor]
    ) -> Tensor:
        """compute multi-scale loss."""
        total_loss = 0.0
        
        for i, weight in enumerate(self.scale_weights):
            if i > 0:
                real = F.avg_pool3d(real, kernel_size=2, stride=2)
                reconstructed = F.avg_pool3d(reconstructed, kernel_size=2, stride=2)
                if mask is not None:
                    mask = F.avg_pool3d(mask.float(), kernel_size=2, stride=2)
            
            scale_loss = self._single_scale_loss(real, reconstructed, mask)
            total_loss = total_loss + weight * scale_loss
        
        return total_loss


class VolumetricGradientLoss(nn.Module):
    """
    3d gradient matching loss for edge preservation.
    
    computes gradient differences in all three spatial dimensions
    to preserve anatomical boundaries during harmonization.
    """
    
    def __init__(
        self,
        loss_type: str = 'l1',
        edge_weight: float = 1.0
    ):
        """
        initialize gradient loss.
        
        args:
            loss_type: 'l1' or 'l2' loss for gradients
            edge_weight: weight for edge regions
        """
        super().__init__()
        self.loss_type = loss_type
        self.edge_weight = edge_weight
        
        # sobel kernels for 3d
        self.register_buffer('sobel_x', self._create_sobel_kernel('x'))
        self.register_buffer('sobel_y', self._create_sobel_kernel('y'))
        self.register_buffer('sobel_z', self._create_sobel_kernel('z'))
    
    def _create_sobel_kernel(self, direction: str) -> Tensor:
        """create 3d sobel kernel for given direction."""
        # base 2d sobel kernel
        sobel_2d = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32)
        
        if direction == 'x':
            kernel = sobel_2d
        elif direction == 'y':
            kernel = sobel_2d.transpose(1, 2)
        elif direction == 'z':
            kernel = sobel_2d.transpose(0, 2)
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        # shape for conv3d: (out, in, d, h, w)
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def _compute_gradients(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """compute 3d gradients using sobel filters."""
        # apply to each channel
        B, C, D, H, W = x.shape
        
        grad_x = F.conv3d(
            x.view(B * C, 1, D, H, W),
            self.sobel_x.to(x.device),
            padding=1
        ).view(B, C, D, H, W)
        
        grad_y = F.conv3d(
            x.view(B * C, 1, D, H, W),
            self.sobel_y.to(x.device),
            padding=1
        ).view(B, C, D, H, W)
        
        grad_z = F.conv3d(
            x.view(B * C, 1, D, H, W),
            self.sobel_z.to(x.device),
            padding=1
        ).view(B, C, D, H, W)
        
        return grad_x, grad_y, grad_z
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        compute gradient matching loss.
        
        args:
            pred: predicted volume (b, c, d, h, w)
            target: target volume (b, c, d, h, w)
            
        returns:
            gradient loss value
        """
        pred_gx, pred_gy, pred_gz = self._compute_gradients(pred)
        target_gx, target_gy, target_gz = self._compute_gradients(target)
        
        if self.loss_type == 'l1':
            loss_x = F.l1_loss(pred_gx, target_gx)
            loss_y = F.l1_loss(pred_gy, target_gy)
            loss_z = F.l1_loss(pred_gz, target_gz)
        else:
            loss_x = F.mse_loss(pred_gx, target_gx)
            loss_y = F.mse_loss(pred_gy, target_gy)
            loss_z = F.mse_loss(pred_gz, target_gz)
        
        return (loss_x + loss_y + loss_z) / 3.0


class VolumetricPerceptualLoss(nn.Module):
    """
    3d perceptual loss using pretrained 3d feature extractors.
    
    options include:
    - medicalnet (pretrained 3d resnet on medical images)
    - custom 3d vgg-style network
    - 2.5d approach using pretrained 2d networks on orthogonal slices
    """
    
    def __init__(
        self,
        feature_extractor: str = 'resnet3d',  # 'resnet3d', 'vgg2.5d', 'custom'
        layer_weights: Optional[Dict[str, float]] = None,
        normalize_features: bool = True
    ):
        """
        initialize perceptual loss.
        
        args:
            feature_extractor: type of feature extractor
            layer_weights: weights for different layers
            normalize_features: whether to normalize features
        """
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.layer_weights = layer_weights or {'layer1': 1.0, 'layer2': 0.5}
        self.normalize_features = normalize_features
        
        # build feature extraction network
        self.encoder = self._build_encoder()
        
        # freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def _build_encoder(self) -> nn.Module:
        """build 3d feature extraction encoder."""
        if self.feature_extractor == 'resnet3d':
            return ResNet3DEncoder()
        elif self.feature_extractor == 'vgg2.5d':
            return VGG25DEncoder()
        else:
            return Custom3DEncoder()
    
    def forward(
        self,
        pred: Tensor,
        target: Tensor
    ) -> Tensor:
        """
        compute perceptual loss.
        
        args:
            pred: predicted volume (b, c, d, h, w)
            target: target volume (b, c, d, h, w)
            
        returns:
            perceptual loss value
        """
        # expand single channel to 3 for pretrained networks
        if pred.size(1) == 1:
            pred = pred.expand(-1, 3, -1, -1, -1)
            target = target.expand(-1, 3, -1, -1, -1)
        
        pred_features = self.encoder(pred)
        target_features = self.encoder(target)
        
        total_loss = 0.0
        
        for name, weight in self.layer_weights.items():
            if name in pred_features and name in target_features:
                p_feat = pred_features[name]
                t_feat = target_features[name]
                
                if self.normalize_features:
                    p_feat = F.normalize(p_feat, dim=1)
                    t_feat = F.normalize(t_feat, dim=1)
                
                total_loss = total_loss + weight * F.mse_loss(p_feat, t_feat)
        
        return total_loss


class ResNet3DEncoder(nn.Module):
    """simplified 3d resnet encoder for perceptual features."""
    
    def __init__(self, base_channels: int = 64):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(base_channels, base_channels, blocks=2)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, blocks=2, stride=2)
    
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """create residual layer."""
        layers = []
        
        # first block with potential downsampling
        layers.append(
            ResBlock3D(in_channels, out_channels, stride=stride)
        )
        
        for _ in range(1, blocks):
            layers.append(ResBlock3D(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """extract features at multiple levels."""
        features = {}
        
        x = self.conv1(x)
        
        x = self.layer1(x)
        features['layer1'] = x
        
        x = self.layer2(x)
        features['layer2'] = x
        
        x = self.layer3(x)
        features['layer3'] = x
        
        return features


class ResBlock3D(nn.Module):
    """basic 3d residual block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.InstanceNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.InstanceNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels)
            )
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out


class VGG25DEncoder(nn.Module):
    """
    2.5d vgg-style encoder using pretrained 2d vgg on orthogonal slices.
    
    extracts features from axial, sagittal, and coronal views.
    """
    
    def __init__(self):
        super().__init__()
        
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except ImportError:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)
        
        # extract feature layers
        self.features = vgg.features[:23]  # up to relu4_3
        
        # freeze
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        extract 2.5d features from orthogonal views.
        
        args:
            x: input volume (b, c, d, h, w)
            
        returns:
            dictionary of multi-view features
        """
        B, C, D, H, W = x.shape
        
        # get central slices from each view
        axial_idx = D // 2
        sagittal_idx = H // 2
        coronal_idx = W // 2
        
        # extract slices
        axial = x[:, :, axial_idx, :, :]  # (b, c, h, w)
        sagittal = x[:, :, :, sagittal_idx, :]  # (b, c, d, w)
        coronal = x[:, :, :, :, coronal_idx]  # (b, c, d, h)
        
        # expand to 3 channels if needed
        if C == 1:
            axial = axial.expand(-1, 3, -1, -1)
            sagittal = sagittal.expand(-1, 3, -1, -1)
            coronal = coronal.expand(-1, 3, -1, -1)
        
        # extract features
        axial_feat = self.features(axial)
        sagittal_feat = self.features(sagittal)
        coronal_feat = self.features(coronal)
        
        return {
            'layer1': axial_feat,
            'layer2': sagittal_feat,
            'layer3': coronal_feat,
        }


class Custom3DEncoder(nn.Module):
    """simple custom 3d encoder for perceptual features."""
    
    def __init__(self, base_channels: int = 32):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv3d(3, base_channels, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.InstanceNorm3d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.InstanceNorm3d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        features = self.encoder(x)
        return {'layer1': features}


class AnatomicalConsistencyLoss(nn.Module):
    """
    anatomical consistency loss for brain structure preservation.
    
    uses brain parcellation or tissue segmentation to ensure
    anatomical regions are preserved during harmonization.
    """
    
    def __init__(
        self,
        use_tissue_priors: bool = True,
        gm_weight: float = 1.0,
        wm_weight: float = 1.0,
        csf_weight: float = 0.5
    ):
        """
        initialize anatomical loss.
        
        args:
            use_tissue_priors: use tissue probability maps
            gm_weight: weight for gray matter preservation
            wm_weight: weight for white matter preservation
            csf_weight: weight for csf preservation
        """
        super().__init__()
        
        self.use_tissue_priors = use_tissue_priors
        self.weights = {
            'gm': gm_weight,
            'wm': wm_weight,
            'csf': csf_weight
        }
    
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        tissue_maps: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        """
        compute anatomical consistency loss.
        
        args:
            pred: predicted volume (b, c, d, h, w)
            target: target volume (b, c, d, h, w)
            tissue_maps: optional dictionary of tissue probability maps
            
        returns:
            anatomical consistency loss
        """
        if tissue_maps is None:
            # use intensity-based pseudo-segmentation
            tissue_maps = self._estimate_tissue_priors(target)
        
        total_loss = 0.0
        
        for tissue, weight in self.weights.items():
            if tissue in tissue_maps:
                mask = tissue_maps[tissue]
                
                # compute masked l1 loss
                diff = torch.abs(pred - target) * mask
                tissue_loss = diff.sum() / (mask.sum() + 1e-8)
                
                total_loss = total_loss + weight * tissue_loss
        
        return total_loss
    
    def _estimate_tissue_priors(self, volume: Tensor) -> Dict[str, Tensor]:
        """
        estimate tissue probability maps from intensity.
        
        uses simple thresholding as a rough approximation.
        real implementation should use proper segmentation.
        """
        # normalize to [0, 1]
        vol_min = volume.min()
        vol_max = volume.max()
        normalized = (volume - vol_min) / (vol_max - vol_min + 1e-8)
        
        # rough tissue estimation based on intensity
        # csf: ~0-0.3, gm: ~0.3-0.6, wm: ~0.6-1.0
        csf_mask = (normalized < 0.3).float()
        gm_mask = ((normalized >= 0.3) & (normalized < 0.6)).float()
        wm_mask = (normalized >= 0.6).float()
        
        return {
            'csf': csf_mask,
            'gm': gm_mask,
            'wm': wm_mask
        }


class TissuePreservationLoss(nn.Module):
    """
    tissue boundary preservation loss.
    
    preserves tissue boundaries (gm/wm, wm/csf) during harmonization
    by penalizing boundary changes.
    """
    
    def __init__(self, boundary_weight: float = 1.0):
        """
        initialize tissue preservation loss.
        
        args:
            boundary_weight: weight for boundary preservation
        """
        super().__init__()
        self.boundary_weight = boundary_weight
        
        # 3d laplacian for edge detection
        laplacian = torch.tensor([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ], dtype=torch.float32)
        self.register_buffer('laplacian', laplacian.unsqueeze(0).unsqueeze(0))
    
    def _compute_boundaries(self, x: Tensor) -> Tensor:
        """compute tissue boundaries using laplacian."""
        B, C, D, H, W = x.shape
        
        boundaries = F.conv3d(
            x.view(B * C, 1, D, H, W),
            self.laplacian.to(x.device),
            padding=1
        )
        
        return boundaries.view(B, C, D, H, W).abs()
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        compute tissue boundary preservation loss.
        
        args:
            pred: predicted volume
            target: target volume
            
        returns:
            boundary preservation loss
        """
        pred_boundaries = self._compute_boundaries(pred)
        target_boundaries = self._compute_boundaries(target)
        
        # encourage boundary preservation
        boundary_loss = F.mse_loss(pred_boundaries, target_boundaries)
        
        return self.boundary_weight * boundary_loss


class VolumetricNCELoss(nn.Module):
    """
    3d contrastive learning loss for unpaired volumetric translation.
    
    based on contrastive unpaired translation (cut) extended to 3d.
    uses patch-wise contrastive learning within volumes.
    """
    
    def __init__(
        self,
        num_patches: int = 64,
        temperature: float = 0.07,
        nce_layers: List[int] = [0, 4, 8, 12, 16]
    ):
        """
        initialize nce loss.
        
        args:
            num_patches: number of patches to sample
            temperature: contrastive temperature
            nce_layers: layers to use for nce
        """
        super().__init__()
        
        self.num_patches = num_patches
        self.temperature = temperature
        self.nce_layers = nce_layers
        
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
    
    def forward(
        self,
        feat_q: List[Tensor],
        feat_k: List[Tensor],
        feat_neg: Optional[List[Tensor]] = None
    ) -> Tensor:
        """
        compute nce loss.
        
        args:
            feat_q: query features from different layers
            feat_k: positive key features
            feat_neg: optional negative features (if none, use other patches)
            
        returns:
            nce loss value
        """
        total_loss = 0.0
        
        for idx in range(len(feat_q)):
            q = feat_q[idx]  # (b, c, d, h, w)
            k = feat_k[idx]
            
            B, C, D, H, W = q.shape
            
            # flatten spatial dimensions
            q_flat = q.view(B, C, -1)  # (b, c, dhw)
            k_flat = k.view(B, C, -1)
            
            # sample random patches
            num_locations = D * H * W
            patch_ids = torch.randperm(num_locations)[:self.num_patches]
            
            # extract patches
            q_patches = q_flat[:, :, patch_ids]  # (b, c, num_patches)
            k_patches = k_flat[:, :, patch_ids]
            
            # normalize
            q_patches = F.normalize(q_patches, dim=1)
            k_patches = F.normalize(k_patches, dim=1)
            
            # compute positive logits
            l_pos = (q_patches * k_patches).sum(dim=1, keepdim=True)  # (b, 1, num_patches)
            
            # compute negative logits
            l_neg = torch.bmm(q_patches.transpose(1, 2), k_patches)  # (b, num_patches, num_patches)
            
            # diagonal should be masked (positive pairs)
            diagonal_mask = torch.eye(self.num_patches, device=q.device).bool()
            l_neg.masked_fill_(diagonal_mask.unsqueeze(0), float('-inf'))
            
            # combine logits
            logits = torch.cat([l_pos.transpose(1, 2), l_neg], dim=2) / self.temperature
            
            # labels (positive is always at index 0)
            labels = torch.zeros(B, self.num_patches, dtype=torch.long, device=q.device)
            
            # compute loss
            loss = self.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss = total_loss + loss.mean()
        
        return total_loss / len(feat_q)


class VolumetricIdentityLoss(nn.Module):
    """
    identity loss for 3d volumes.
    
    encourages the generator to be identity when given
    target domain input, preserving content.
    """
    
    def __init__(
        self,
        loss_type: str = 'l1',
        lambda_ssim: float = 0.0
    ):
        """
        initialize identity loss.
        
        args:
            loss_type: base loss type ('l1', 'l2')
            lambda_ssim: optional ssim component weight
        """
        super().__init__()
        self.loss_type = loss_type
        self.lambda_ssim = lambda_ssim
        
        if lambda_ssim > 0:
            self.ssim = VolumetricSSIM()
    
    def forward(self, output: Tensor, input: Tensor) -> Tensor:
        """
        compute identity loss.
        
        args:
            output: generator output when given target domain input
            input: target domain input
            
        returns:
            identity loss value
        """
        if self.loss_type == 'l1':
            loss = F.l1_loss(output, input)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(output, input)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        if self.lambda_ssim > 0:
            ssim_loss = 1 - self.ssim(output, input)
            loss = loss + self.lambda_ssim * ssim_loss
        
        return loss


class CombinedVolumetricLoss(nn.Module):
    """
    combined volumetric loss for 3d cyclegan training.
    
    integrates all volumetric losses with configurable weights.
    """
    
    def __init__(
        self,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
        lambda_ssim: float = 1.0,
        lambda_gradient: float = 1.0,
        lambda_perceptual: float = 1.0,
        lambda_anatomical: float = 2.0,
        lambda_tissue: float = 1.0,
        lambda_nce: float = 1.0,
        use_nce: bool = False
    ):
        """
        initialize combined loss.
        
        args:
            lambda_*: weights for each loss component
            use_nce: whether to use nce loss
        """
        super().__init__()
        
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_ssim = lambda_ssim
        self.lambda_gradient = lambda_gradient
        self.lambda_perceptual = lambda_perceptual
        self.lambda_anatomical = lambda_anatomical
        self.lambda_tissue = lambda_tissue
        self.lambda_nce = lambda_nce
        
        # initialize component losses
        self.cycle_loss = VolumetricCycleConsistencyLoss(
            loss_type='combined',
            multi_scale=True
        )
        self.identity_loss = VolumetricIdentityLoss(lambda_ssim=0.5)
        self.ssim_loss = VolumetricMultiScaleSSIM()
        self.gradient_loss = VolumetricGradientLoss()
        self.perceptual_loss = VolumetricPerceptualLoss()
        self.anatomical_loss = AnatomicalConsistencyLoss()
        self.tissue_loss = TissuePreservationLoss()
        
        if use_nce:
            self.nce_loss = VolumetricNCELoss()
        else:
            self.nce_loss = None
    
    def forward(
        self,
        real_A: Tensor,
        real_B: Tensor,
        fake_A: Tensor,
        fake_B: Tensor,
        rec_A: Tensor,
        rec_B: Tensor,
        idt_A: Optional[Tensor] = None,
        idt_B: Optional[Tensor] = None,
        features_A: Optional[List[Tensor]] = None,
        features_B: Optional[List[Tensor]] = None
    ) -> Dict[str, Tensor]:
        """
        compute all losses.
        
        args:
            real_a, real_b: real images from domains a and b
            fake_a, fake_b: generated images
            rec_a, rec_b: reconstructed images after cycle
            idt_a, idt_b: identity outputs (optional)
            features_*: features for nce loss (optional)
            
        returns:
            dictionary of individual and total losses
        """
        losses = {}
        
        # cycle consistency
        loss_cycle_A = self.cycle_loss(real_A, rec_A)
        loss_cycle_B = self.cycle_loss(real_B, rec_B)
        losses['cycle_A'] = loss_cycle_A
        losses['cycle_B'] = loss_cycle_B
        
        # identity
        if idt_A is not None and idt_B is not None:
            loss_idt_A = self.identity_loss(idt_A, real_A)
            loss_idt_B = self.identity_loss(idt_B, real_B)
            losses['identity_A'] = loss_idt_A
            losses['identity_B'] = loss_idt_B
        else:
            loss_idt_A = loss_idt_B = 0
        
        # ssim
        loss_ssim_A = 1 - self.ssim_loss(fake_B, real_A)
        loss_ssim_B = 1 - self.ssim_loss(fake_A, real_B)
        losses['ssim_A'] = loss_ssim_A
        losses['ssim_B'] = loss_ssim_B
        
        # gradient
        loss_grad_A = self.gradient_loss(fake_B, real_A)
        loss_grad_B = self.gradient_loss(fake_A, real_B)
        losses['gradient_A'] = loss_grad_A
        losses['gradient_B'] = loss_grad_B
        
        # perceptual
        loss_perc_A = self.perceptual_loss(fake_B, real_A)
        loss_perc_B = self.perceptual_loss(fake_A, real_B)
        losses['perceptual_A'] = loss_perc_A
        losses['perceptual_B'] = loss_perc_B
        
        # anatomical
        loss_anat_A = self.anatomical_loss(fake_B, real_A)
        loss_anat_B = self.anatomical_loss(fake_A, real_B)
        losses['anatomical_A'] = loss_anat_A
        losses['anatomical_B'] = loss_anat_B
        
        # tissue boundary
        loss_tissue_A = self.tissue_loss(fake_B, real_A)
        loss_tissue_B = self.tissue_loss(fake_A, real_B)
        losses['tissue_A'] = loss_tissue_A
        losses['tissue_B'] = loss_tissue_B
        
        # nce
        if self.nce_loss is not None and features_A is not None:
            loss_nce = self.nce_loss(features_A, features_B)
            losses['nce'] = loss_nce
        else:
            loss_nce = 0
        
        # total
        total = (
            self.lambda_cycle * (loss_cycle_A + loss_cycle_B) +
            self.lambda_identity * (loss_idt_A + loss_idt_B) +
            self.lambda_ssim * (loss_ssim_A + loss_ssim_B) +
            self.lambda_gradient * (loss_grad_A + loss_grad_B) +
            self.lambda_perceptual * (loss_perc_A + loss_perc_B) +
            self.lambda_anatomical * (loss_anat_A + loss_anat_B) +
            self.lambda_tissue * (loss_tissue_A + loss_tissue_B) +
            self.lambda_nce * loss_nce
        )
        
        losses['total'] = total
        
        return losses
