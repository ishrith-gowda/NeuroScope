"""
VGG Feature Extraction Backbones.

Provides VGG-based feature extractors for perceptual loss
computation and feature matching in image translation.
"""

from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict


@dataclass
class VGGConfig:
    """Configuration for VGG feature extractor."""
    feature_layers: List[str] = None
    use_bn: bool = False
    pretrained: bool = True
    requires_grad: bool = False
    normalize_input: bool = True
    normalize_features: bool = False
    pool_type: str = 'max'  # 'max', 'avg', 'none'
    
    def __post_init__(self):
        if self.feature_layers is None:
            self.feature_layers = ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4']


class VGGNormalization(nn.Module):
    """Normalize input to VGG expected range."""
    
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
        # Handle single channel by repeating
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) > 3:
            x = x[:, :3]  # Take first 3 channels
        
        # Normalize from [-1, 1] to [0, 1] if needed
        if x.min() < 0:
            x = (x + 1) / 2
        
        return (x - self.mean) / self.std


class VGG16Features(nn.Module):
    """
    VGG16 feature extractor for perceptual loss.
    
    Extracts intermediate features from specified layers.
    """
    
    # Layer name mapping for VGG16
    LAYER_NAMES = {
        '0': 'conv1_1', '1': 'relu1_1', '2': 'conv1_2', '3': 'relu1_2', '4': 'pool1',
        '5': 'conv2_1', '6': 'relu2_1', '7': 'conv2_2', '8': 'relu2_2', '9': 'pool2',
        '10': 'conv3_1', '11': 'relu3_1', '12': 'conv3_2', '13': 'relu3_2',
        '14': 'conv3_3', '15': 'relu3_3', '16': 'pool3',
        '17': 'conv4_1', '18': 'relu4_1', '19': 'conv4_2', '20': 'relu4_2',
        '21': 'conv4_3', '22': 'relu4_3', '23': 'pool4',
        '24': 'conv5_1', '25': 'relu5_1', '26': 'conv5_2', '27': 'relu5_2',
        '28': 'conv5_3', '29': 'relu5_3', '30': 'pool5',
    }
    
    def __init__(
        self,
        feature_layers: Optional[List[str]] = None,
        pretrained: bool = True,
        requires_grad: bool = False,
        normalize_input: bool = True,
        weights: Optional[str] = 'IMAGENET1K_V1'
    ):
        super().__init__()
        
        self.feature_layers = feature_layers or ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        self.normalize_input = normalize_input
        
        # Load pretrained VGG16
        if pretrained:
            vgg = models.vgg16(weights=weights)
        else:
            vgg = models.vgg16(weights=None)
        
        # Build feature extraction layers
        self.features = vgg.features
        
        # Find the last layer we need
        name_to_idx = {v: int(k) for k, v in self.LAYER_NAMES.items()}
        max_idx = max(name_to_idx.get(layer, 0) for layer in self.feature_layers)
        
        # Truncate to only needed layers
        self.features = self.features[:max_idx + 1]
        
        # Store layer indices
        self.layer_indices = {
            layer: name_to_idx[layer] for layer in self.feature_layers
            if layer in name_to_idx
        }
        
        # Freeze parameters
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        # Input normalization
        if normalize_input:
            self.normalize = VGGNormalization()
        else:
            self.normalize = nn.Identity()
        
        self.eval()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from specified layers.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dictionary mapping layer names to feature tensors
        """
        x = self.normalize(x)
        
        features = {}
        for idx, layer in enumerate(self.features):
            x = layer(x)
            layer_name = self.LAYER_NAMES.get(str(idx))
            if layer_name in self.feature_layers:
                features[layer_name] = x
        
        return features
    
    def get_feature_channels(self) -> Dict[str, int]:
        """Get number of channels for each feature layer."""
        channels = {
            'relu1_1': 64, 'relu1_2': 64,
            'relu2_1': 128, 'relu2_2': 128,
            'relu3_1': 256, 'relu3_2': 256, 'relu3_3': 256,
            'relu4_1': 512, 'relu4_2': 512, 'relu4_3': 512,
            'relu5_1': 512, 'relu5_2': 512, 'relu5_3': 512,
        }
        return {layer: channels[layer] for layer in self.feature_layers if layer in channels}


class VGG19Features(nn.Module):
    """
    VGG19 feature extractor for perceptual loss.
    
    Standard choice for perceptual losses in image generation.
    """
    
    # Layer name mapping for VGG19
    LAYER_NAMES = {
        '0': 'conv1_1', '1': 'relu1_1', '2': 'conv1_2', '3': 'relu1_2', '4': 'pool1',
        '5': 'conv2_1', '6': 'relu2_1', '7': 'conv2_2', '8': 'relu2_2', '9': 'pool2',
        '10': 'conv3_1', '11': 'relu3_1', '12': 'conv3_2', '13': 'relu3_2',
        '14': 'conv3_3', '15': 'relu3_3', '16': 'conv3_4', '17': 'relu3_4', '18': 'pool3',
        '19': 'conv4_1', '20': 'relu4_1', '21': 'conv4_2', '22': 'relu4_2',
        '23': 'conv4_3', '24': 'relu4_3', '25': 'conv4_4', '26': 'relu4_4', '27': 'pool4',
        '28': 'conv5_1', '29': 'relu5_1', '30': 'conv5_2', '31': 'relu5_2',
        '32': 'conv5_3', '33': 'relu5_3', '34': 'conv5_4', '35': 'relu5_4', '36': 'pool5',
    }
    
    def __init__(
        self,
        feature_layers: Optional[List[str]] = None,
        pretrained: bool = True,
        requires_grad: bool = False,
        normalize_input: bool = True,
        weights: Optional[str] = 'IMAGENET1K_V1'
    ):
        super().__init__()
        
        # Default layers for perceptual loss
        self.feature_layers = feature_layers or [
            'relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4'
        ]
        self.normalize_input = normalize_input
        
        # Load pretrained VGG19
        if pretrained:
            vgg = models.vgg19(weights=weights)
        else:
            vgg = models.vgg19(weights=None)
        
        # Build feature extraction layers
        self.features = vgg.features
        
        # Find the last layer we need
        name_to_idx = {v: int(k) for k, v in self.LAYER_NAMES.items()}
        max_idx = max(name_to_idx.get(layer, 0) for layer in self.feature_layers)
        
        # Truncate to only needed layers
        self.features = self.features[:max_idx + 1]
        
        # Store layer indices
        self.layer_indices = {
            layer: name_to_idx[layer] for layer in self.feature_layers
            if layer in name_to_idx
        }
        
        # Freeze parameters
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        # Input normalization
        if normalize_input:
            self.normalize = VGGNormalization()
        else:
            self.normalize = nn.Identity()
        
        self.eval()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from specified layers."""
        x = self.normalize(x)
        
        features = {}
        for idx, layer in enumerate(self.features):
            x = layer(x)
            layer_name = self.LAYER_NAMES.get(str(idx))
            if layer_name in self.feature_layers:
                features[layer_name] = x
        
        return features
    
    def get_feature_channels(self) -> Dict[str, int]:
        """Get number of channels for each feature layer."""
        channels = {
            'relu1_1': 64, 'relu1_2': 64,
            'relu2_1': 128, 'relu2_2': 128,
            'relu3_1': 256, 'relu3_2': 256, 'relu3_3': 256, 'relu3_4': 256,
            'relu4_1': 512, 'relu4_2': 512, 'relu4_3': 512, 'relu4_4': 512,
            'relu5_1': 512, 'relu5_2': 512, 'relu5_3': 512, 'relu5_4': 512,
        }
        return {layer: channels[layer] for layer in self.feature_layers if layer in channels}


class VGGPerceptualExtractor(nn.Module):
    """
    Flexible VGG-based perceptual feature extractor.
    
    Supports both VGG16 and VGG19, with configurable layer weights
    for computing perceptual loss.
    """
    
    def __init__(
        self,
        vgg_type: str = 'vgg19',
        feature_layers: Optional[List[str]] = None,
        layer_weights: Optional[Dict[str, float]] = None,
        pretrained: bool = True,
        normalize_input: bool = True,
        normalize_features: bool = False,
        pool_type: str = 'none'
    ):
        super().__init__()
        
        self.vgg_type = vgg_type
        self.normalize_features = normalize_features
        self.pool_type = pool_type
        
        # Default layer weights
        if layer_weights is None:
            layer_weights = {
                'relu1_2': 1.0,
                'relu2_2': 1.0,
                'relu3_4': 1.0,
                'relu4_4': 1.0,
            }
        self.layer_weights = layer_weights
        
        # Feature layers from weights
        if feature_layers is None:
            feature_layers = list(layer_weights.keys())
        
        # Build backbone
        if vgg_type == 'vgg16':
            self.backbone = VGG16Features(
                feature_layers=feature_layers,
                pretrained=pretrained,
                normalize_input=normalize_input
            )
        else:
            self.backbone = VGG19Features(
                feature_layers=feature_layers,
                pretrained=pretrained,
                normalize_input=normalize_input
            )
        
        # Pooling layers
        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            self.pool = None
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract weighted features."""
        features = self.backbone(x)
        
        if self.normalize_features:
            for key in features:
                features[key] = nn.functional.normalize(features[key], dim=1)
        
        if self.pool is not None:
            for key in features:
                features[key] = self.pool(features[key])
        
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
        for layer, weight in self.layer_weights.items():
            if layer in pred_features and layer in target_features:
                pred_feat = pred_features[layer]
                target_feat = target_features[layer]
                
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


class MultiLayerVGG(nn.Module):
    """
    Multi-layer VGG extractor with Gram matrix computation.
    
    Computes both content and style features.
    """
    
    def __init__(
        self,
        content_layers: Optional[List[str]] = None,
        style_layers: Optional[List[str]] = None,
        pretrained: bool = True,
        normalize_input: bool = True
    ):
        super().__init__()
        
        self.content_layers = content_layers or ['relu4_4']
        self.style_layers = style_layers or [
            'relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4'
        ]
        
        all_layers = list(set(self.content_layers + self.style_layers))
        
        self.backbone = VGG19Features(
            feature_layers=all_layers,
            pretrained=pretrained,
            normalize_input=normalize_input
        )
    
    def gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix for style representation.
        
        Args:
            features: Feature tensor [B, C, H, W]
            
        Returns:
            Gram matrix [B, C, C]
        """
        B, C, H, W = features.size()
        features = features.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (C * H * W)
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Extract content and style features.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (content_features, style_features)
        """
        all_features = self.backbone(x)
        
        content_features = {
            layer: all_features[layer]
            for layer in self.content_layers
            if layer in all_features
        }
        
        style_features = {
            layer: self.gram_matrix(all_features[layer])
            for layer in self.style_layers
            if layer in all_features
        }
        
        return content_features, style_features
    
    def compute_content_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute content loss."""
        pred_content, _ = self.forward(pred)
        target_content, _ = self.forward(target)
        
        loss = 0
        for layer in self.content_layers:
            if layer in pred_content and layer in target_content:
                loss = loss + torch.pow(
                    pred_content[layer] - target_content[layer], 2
                ).mean()
        
        return loss
    
    def compute_style_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute style loss using Gram matrices."""
        _, pred_style = self.forward(pred)
        _, target_style = self.forward(target)
        
        loss = 0
        for layer in self.style_layers:
            if layer in pred_style and layer in target_style:
                loss = loss + torch.pow(
                    pred_style[layer] - target_style[layer], 2
                ).mean()
        
        return loss
