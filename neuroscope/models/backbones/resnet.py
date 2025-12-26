"""
ResNet Feature Extraction Backbones.

Provides ResNet-based feature extractors for multi-scale
perceptual loss and feature matching.
"""

from typing import List, Optional, Dict, Tuple
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetNormalization(nn.Module):
    """Normalize input to ResNet expected range."""
    
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


class ResNet18Features(nn.Module):
    """
    ResNet18 feature extractor.
    
    Extracts features from multiple stages for multi-scale analysis.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        requires_grad: bool = False,
        normalize_input: bool = True,
        feature_stages: Optional[List[int]] = None,
        weights: Optional[str] = 'IMAGENET1K_V1'
    ):
        super().__init__()
        
        self.feature_stages = feature_stages or [1, 2, 3, 4]
        self.normalize_input = normalize_input
        
        # Load pretrained ResNet18
        if pretrained:
            resnet = models.resnet18(weights=weights)
        else:
            resnet = models.resnet18(weights=None)
        
        # Build stages
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        # Freeze parameters
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        # Input normalization
        if normalize_input:
            self.normalize = ResNetNormalization()
        else:
            self.normalize = nn.Identity()
        
        self.eval()
    
    @property
    def feature_channels(self) -> Dict[int, int]:
        """Get number of channels at each stage."""
        return {1: 64, 2: 128, 3: 256, 4: 512}
    
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
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        if 1 in self.feature_stages:
            features[1] = x
        
        x = self.layer2(x)
        if 2 in self.feature_stages:
            features[2] = x
        
        x = self.layer3(x)
        if 3 in self.feature_stages:
            features[3] = x
        
        x = self.layer4(x)
        if 4 in self.feature_stages:
            features[4] = x
        
        return features


class ResNet34Features(nn.Module):
    """ResNet34 feature extractor with multi-stage outputs."""
    
    def __init__(
        self,
        pretrained: bool = True,
        requires_grad: bool = False,
        normalize_input: bool = True,
        feature_stages: Optional[List[int]] = None,
        weights: Optional[str] = 'IMAGENET1K_V1'
    ):
        super().__init__()
        
        self.feature_stages = feature_stages or [1, 2, 3, 4]
        self.normalize_input = normalize_input
        
        if pretrained:
            resnet = models.resnet34(weights=weights)
        else:
            resnet = models.resnet34(weights=None)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        if normalize_input:
            self.normalize = ResNetNormalization()
        else:
            self.normalize = nn.Identity()
        
        self.eval()
    
    @property
    def feature_channels(self) -> Dict[int, int]:
        return {1: 64, 2: 128, 3: 256, 4: 512}
    
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        x = self.normalize(x)
        features = {}
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        if 1 in self.feature_stages:
            features[1] = x
        
        x = self.layer2(x)
        if 2 in self.feature_stages:
            features[2] = x
        
        x = self.layer3(x)
        if 3 in self.feature_stages:
            features[3] = x
        
        x = self.layer4(x)
        if 4 in self.feature_stages:
            features[4] = x
        
        return features


class ResNet50Features(nn.Module):
    """
    ResNet50 feature extractor.
    
    Deeper features with bottleneck blocks for more
    expressive representations.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        requires_grad: bool = False,
        normalize_input: bool = True,
        feature_stages: Optional[List[int]] = None,
        include_stem: bool = False,
        weights: Optional[str] = 'IMAGENET1K_V1'
    ):
        super().__init__()
        
        self.feature_stages = feature_stages or [1, 2, 3, 4]
        self.normalize_input = normalize_input
        self.include_stem = include_stem
        
        if pretrained:
            resnet = models.resnet50(weights=weights)
        else:
            resnet = models.resnet50(weights=None)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        if normalize_input:
            self.normalize = ResNetNormalization()
        else:
            self.normalize = nn.Identity()
        
        self.eval()
    
    @property
    def feature_channels(self) -> Dict[int, int]:
        channels = {0: 64, 1: 256, 2: 512, 3: 1024, 4: 2048}
        return {s: channels[s] for s in self.feature_stages if s in channels}
    
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        x = self.normalize(x)
        features = {}
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        if self.include_stem and 0 in self.feature_stages:
            features[0] = x
        
        x = self.maxpool(x)
        
        x = self.layer1(x)
        if 1 in self.feature_stages:
            features[1] = x
        
        x = self.layer2(x)
        if 2 in self.feature_stages:
            features[2] = x
        
        x = self.layer3(x)
        if 3 in self.feature_stages:
            features[3] = x
        
        x = self.layer4(x)
        if 4 in self.feature_stages:
            features[4] = x
        
        return features


class ResNetPerceptualExtractor(nn.Module):
    """
    Flexible ResNet-based perceptual feature extractor.
    
    Supports ResNet18, 34, 50 with configurable stage weights.
    """
    
    def __init__(
        self,
        resnet_type: str = 'resnet50',
        feature_stages: Optional[List[int]] = None,
        stage_weights: Optional[Dict[int, float]] = None,
        pretrained: bool = True,
        normalize_input: bool = True,
        normalize_features: bool = False
    ):
        super().__init__()
        
        self.resnet_type = resnet_type
        self.normalize_features = normalize_features
        
        # Default stage weights
        if stage_weights is None:
            stage_weights = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
        self.stage_weights = stage_weights
        
        if feature_stages is None:
            feature_stages = list(stage_weights.keys())
        
        # Build backbone
        if resnet_type == 'resnet18':
            self.backbone = ResNet18Features(
                pretrained=pretrained,
                normalize_input=normalize_input,
                feature_stages=feature_stages
            )
        elif resnet_type == 'resnet34':
            self.backbone = ResNet34Features(
                pretrained=pretrained,
                normalize_input=normalize_input,
                feature_stages=feature_stages
            )
        else:  # resnet50
            self.backbone = ResNet50Features(
                pretrained=pretrained,
                normalize_input=normalize_input,
                feature_stages=feature_stages
            )
    
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
        """
        Compute perceptual loss between prediction and target.
        
        Args:
            pred: Predicted image tensor
            target: Target image tensor
            loss_type: 'l1', 'l2', or 'cos'
            
        Returns:
            Weighted perceptual loss
        """
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


class MultiScaleResNetFeatures(nn.Module):
    """
    Multi-scale feature extraction using ResNet.
    
    Processes input at multiple scales for robust features.
    """
    
    def __init__(
        self,
        resnet_type: str = 'resnet50',
        scales: List[float] = None,
        feature_stages: Optional[List[int]] = None,
        pretrained: bool = True,
        normalize_input: bool = True
    ):
        super().__init__()
        
        self.scales = scales or [1.0, 0.5, 0.25]
        self.feature_stages = feature_stages or [3, 4]
        
        # Build backbone
        if resnet_type == 'resnet18':
            self.backbone = ResNet18Features(
                pretrained=pretrained,
                normalize_input=normalize_input,
                feature_stages=feature_stages
            )
        elif resnet_type == 'resnet34':
            self.backbone = ResNet34Features(
                pretrained=pretrained,
                normalize_input=normalize_input,
                feature_stages=feature_stages
            )
        else:
            self.backbone = ResNet50Features(
                pretrained=pretrained,
                normalize_input=normalize_input,
                feature_stages=feature_stages
            )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Dict[float, Dict[int, torch.Tensor]]:
        """
        Extract features at multiple scales.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dictionary mapping scale to feature dictionaries
        """
        multi_scale_features = {}
        
        for scale in self.scales:
            if scale != 1.0:
                scaled_x = nn.functional.interpolate(
                    x, scale_factor=scale, mode='bilinear',
                    align_corners=False
                )
            else:
                scaled_x = x
            
            features = self.backbone(scaled_x)
            multi_scale_features[scale] = features
        
        return multi_scale_features
    
    def compute_multi_scale_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        scale_weights: Optional[Dict[float, float]] = None
    ) -> torch.Tensor:
        """Compute loss across multiple scales."""
        if scale_weights is None:
            scale_weights = {s: 1.0 / len(self.scales) for s in self.scales}
        
        pred_features = self.forward(pred)
        target_features = self.forward(target)
        
        total_loss = 0
        for scale, weight in scale_weights.items():
            if scale in pred_features and scale in target_features:
                for stage in self.feature_stages:
                    pred_feat = pred_features[scale][stage]
                    target_feat = target_features[scale][stage]
                    
                    loss = torch.abs(pred_feat - target_feat).mean()
                    total_loss = total_loss + weight * loss
        
        return total_loss
