"""
EfficientNet Feature Extraction Backbones.

Modern efficient feature extractors with compound scaling
for perceptual losses and feature matching.
"""

from typing import List, Optional, Dict, Tuple, Union
import torch
import torch.nn as nn

try:
    import torchvision.models as models
    from torchvision.models import efficientnet_b0, efficientnet_b4
    HAS_EFFICIENTNET = True
except ImportError:
    HAS_EFFICIENTNET = False


class EfficientNetNormalization(nn.Module):
    """Normalize input to EfficientNet expected range."""
    
    def __init__(
        self,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor."""
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) > 3:
            x = x[:, :3]
        
        if x.min() < 0:
            x = (x + 1) / 2
        
        return (x - self.mean) / self.std


class EfficientNetB0Features(nn.Module):
    """
    EfficientNet-B0 feature extractor.
    
    Efficient compound-scaled network for feature extraction.
    """
    
    # Stage output channels for EfficientNet-B0
    STAGE_CHANNELS = {
        0: 32,   # After stem
        1: 16,   # After MBConv1
        2: 24,   # After MBConv2
        3: 40,   # After MBConv3
        4: 80,   # After MBConv4
        5: 112,  # After MBConv5
        6: 192,  # After MBConv6
        7: 320,  # After MBConv7
        8: 1280, # After final conv
    }
    
    def __init__(
        self,
        pretrained: bool = True,
        requires_grad: bool = False,
        normalize_input: bool = True,
        feature_stages: Optional[List[int]] = None,
        weights: Optional[str] = 'IMAGENET1K_V1'
    ):
        super().__init__()
        
        if not HAS_EFFICIENTNET:
            raise ImportError("EfficientNet requires torchvision >= 0.11")
        
        self.feature_stages = feature_stages or [2, 4, 6, 8]
        self.normalize_input = normalize_input
        
        # Load pretrained EfficientNet-B0
        if pretrained:
            efficientnet = efficientnet_b0(weights=weights)
        else:
            efficientnet = efficientnet_b0(weights=None)
        
        # Extract features backbone
        self.features = efficientnet.features
        
        # Freeze parameters
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        # Input normalization
        if normalize_input:
            self.normalize = EfficientNetNormalization()
        else:
            self.normalize = nn.Identity()
        
        self.eval()
    
    @property
    def feature_channels(self) -> Dict[int, int]:
        """Get number of channels at each stage."""
        return {s: self.STAGE_CHANNELS[s] for s in self.feature_stages}
    
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Extract features from specified stages.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dictionary mapping stage number to feature tensors
        """
        x = self.normalize(x)
        
        features = {}
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx in self.feature_stages:
                features[idx] = x
        
        return features


class EfficientNetB4Features(nn.Module):
    """
    EfficientNet-B4 feature extractor.
    
    Higher capacity model with better accuracy-efficiency tradeoff.
    """
    
    # Stage output channels for EfficientNet-B4
    STAGE_CHANNELS = {
        0: 48,   # After stem
        1: 24,   # After MBConv1
        2: 32,   # After MBConv2
        3: 56,   # After MBConv3
        4: 112,  # After MBConv4
        5: 160,  # After MBConv5
        6: 272,  # After MBConv6
        7: 448,  # After MBConv7
        8: 1792, # After final conv
    }
    
    def __init__(
        self,
        pretrained: bool = True,
        requires_grad: bool = False,
        normalize_input: bool = True,
        feature_stages: Optional[List[int]] = None,
        weights: Optional[str] = 'IMAGENET1K_V1'
    ):
        super().__init__()
        
        if not HAS_EFFICIENTNET:
            raise ImportError("EfficientNet requires torchvision >= 0.11")
        
        self.feature_stages = feature_stages or [2, 4, 6, 8]
        self.normalize_input = normalize_input
        
        if pretrained:
            efficientnet = efficientnet_b4(weights=weights)
        else:
            efficientnet = efficientnet_b4(weights=None)
        
        self.features = efficientnet.features
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        if normalize_input:
            self.normalize = EfficientNetNormalization()
        else:
            self.normalize = nn.Identity()
        
        self.eval()
    
    @property
    def feature_channels(self) -> Dict[int, int]:
        return {s: self.STAGE_CHANNELS[s] for s in self.feature_stages}
    
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        x = self.normalize(x)
        
        features = {}
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx in self.feature_stages:
                features[idx] = x
        
        return features


class EfficientNetFeatureExtractor(nn.Module):
    """
    Flexible EfficientNet-based feature extractor.
    
    Supports B0-B4 variants with configurable stage weights.
    """
    
    def __init__(
        self,
        variant: str = 'b0',
        feature_stages: Optional[List[int]] = None,
        stage_weights: Optional[Dict[int, float]] = None,
        pretrained: bool = True,
        normalize_input: bool = True,
        normalize_features: bool = False
    ):
        super().__init__()
        
        self.variant = variant
        self.normalize_features = normalize_features
        
        # Default stage weights
        if stage_weights is None:
            stage_weights = {2: 1.0, 4: 1.0, 6: 1.0, 8: 1.0}
        self.stage_weights = stage_weights
        
        if feature_stages is None:
            feature_stages = list(stage_weights.keys())
        
        # Build backbone
        if variant == 'b0':
            self.backbone = EfficientNetB0Features(
                pretrained=pretrained,
                normalize_input=normalize_input,
                feature_stages=feature_stages
            )
        elif variant == 'b4':
            self.backbone = EfficientNetB4Features(
                pretrained=pretrained,
                normalize_input=normalize_input,
                feature_stages=feature_stages
            )
        else:
            raise ValueError(f"Unsupported variant: {variant}")
    
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Extract features from all stages."""
        features = self.backbone(x)
        
        if self.normalize_features:
            for key in features:
                features[key] = nn.functional.normalize(features[key], dim=1)
        
        return features
    
    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_type: str = 'l1'
    ) -> torch.Tensor:
        """Compute perceptual loss between prediction and target."""
        pred_features = self.forward(pred)
        target_features = self.forward(target)
        
        total_loss = 0
        for stage, weight in self.stage_weights.items():
            if stage in pred_features and stage in target_features:
                pred_feat = pred_features[stage]
                target_feat = target_features[stage]
                
                if loss_type == 'l1':
                    loss = torch.abs(pred_feat - target_feat).mean()
                elif loss_type == 'l2':
                    loss = torch.pow(pred_feat - target_feat, 2).mean()
                elif loss_type == 'cos':
                    loss = 1 - nn.functional.cosine_similarity(
                        pred_feat.flatten(2), target_feat.flatten(2), dim=2
                    ).mean()
                else:
                    loss = torch.abs(pred_feat - target_feat).mean()
                
                total_loss = total_loss + weight * loss
        
        return total_loss


class HybridFeatureExtractor(nn.Module):
    """
    Hybrid feature extractor combining multiple backbones.
    
    Combines features from VGG, ResNet, and EfficientNet for
    robust perceptual loss computation.
    """
    
    def __init__(
        self,
        backbones: Optional[List[str]] = None,
        backbone_weights: Optional[Dict[str, float]] = None,
        pretrained: bool = True,
        normalize_input: bool = True
    ):
        super().__init__()
        
        self.backbones_names = backbones or ['vgg19', 'resnet50']
        
        if backbone_weights is None:
            backbone_weights = {name: 1.0 for name in self.backbones_names}
        self.backbone_weights = backbone_weights
        
        # Import backbone classes
        from .vgg import VGG19Features
        from .resnet import ResNet50Features
        
        # Build backbones
        self.backbones = nn.ModuleDict()
        
        if 'vgg19' in self.backbones_names:
            self.backbones['vgg19'] = VGG19Features(
                pretrained=pretrained,
                normalize_input=normalize_input,
                feature_layers=['relu2_2', 'relu3_4', 'relu4_4']
            )
        
        if 'resnet50' in self.backbones_names:
            self.backbones['resnet50'] = ResNet50Features(
                pretrained=pretrained,
                normalize_input=normalize_input,
                feature_stages=[2, 3, 4]
            )
        
        if 'efficientnet' in self.backbones_names and HAS_EFFICIENTNET:
            self.backbones['efficientnet'] = EfficientNetB0Features(
                pretrained=pretrained,
                normalize_input=normalize_input,
                feature_stages=[4, 6, 8]
            )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Dict[str, Dict[Union[str, int], torch.Tensor]]:
        """
        Extract features from all backbones.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Nested dictionary: backbone -> layer -> features
        """
        all_features = {}
        for name, backbone in self.backbones.items():
            all_features[name] = backbone(x)
        return all_features
    
    def compute_hybrid_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_type: str = 'l1'
    ) -> torch.Tensor:
        """
        Compute hybrid perceptual loss from all backbones.
        
        Args:
            pred: Predicted image
            target: Target image
            loss_type: Loss function type
            
        Returns:
            Combined perceptual loss
        """
        pred_features = self.forward(pred)
        target_features = self.forward(target)
        
        total_loss = 0
        for backbone_name, weight in self.backbone_weights.items():
            if backbone_name in pred_features:
                pred_backbone = pred_features[backbone_name]
                target_backbone = target_features[backbone_name]
                
                backbone_loss = 0
                for layer in pred_backbone:
                    pred_feat = pred_backbone[layer]
                    target_feat = target_backbone[layer]
                    
                    if loss_type == 'l1':
                        loss = torch.abs(pred_feat - target_feat).mean()
                    elif loss_type == 'l2':
                        loss = torch.pow(pred_feat - target_feat, 2).mean()
                    else:
                        loss = torch.abs(pred_feat - target_feat).mean()
                    
                    backbone_loss = backbone_loss + loss
                
                total_loss = total_loss + weight * backbone_loss
        
        return total_loss
    
    def get_all_feature_channels(self) -> Dict[str, Dict]:
        """Get feature channels for all backbones."""
        channels = {}
        for name, backbone in self.backbones.items():
            if hasattr(backbone, 'feature_channels'):
                channels[name] = backbone.feature_channels
            elif hasattr(backbone, 'get_feature_channels'):
                channels[name] = backbone.get_feature_channels()
        return channels
