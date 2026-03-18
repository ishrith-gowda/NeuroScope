"""
perceptual loss functions.

computes losses in feature space of pretrained networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict


class VGGFeatureExtractor(nn.Module):
    """
    vgg feature extractor for perceptual loss computation.
    
    extracts features from intermediate vgg layers.
    
    args:
        layer_ids: layer indices to extract features from
        use_input_norm: whether to normalize input to vgg statistics
        requires_grad: whether to compute gradients for vgg
    """
    
    LAYER_NAMES = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
    ]
    
    def __init__(
        self,
        layer_ids: List[int] = [3, 8, 15, 22],  # relu1_2, relu2_2, relu3_3, relu4_3
        use_input_norm: bool = True,
        requires_grad: bool = False
    ):
        super().__init__()
        
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        except ImportError:
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True).features
            
        self.layer_ids = sorted(layer_ids)
        
        # extract only needed layers
        layers = OrderedDict()
        for i in range(max(self.layer_ids) + 1):
            layers[f'layer_{i}'] = vgg[i]
        self.vgg = nn.Sequential(layers)
        
        # freeze weights
        if not requires_grad:
            for param in self.vgg.parameters():
                param.requires_grad = False
                
        self.use_input_norm = use_input_norm
        
        # imagenet normalization
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        extract features from specified layers.
        
        args:
            x: input tensor [b, c, h, w] (can be 1-4 channels)
            
        returns:
            dictionary mapping layer ids to feature tensors
        """
        # convert to 3 channels if necessary
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) == 4:
            x = x[:, :3]  # use first 3 channels
        elif x.size(1) != 3:
            # average to 3 channels
            x = x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            
        # normalize to vgg input range
        if self.use_input_norm:
            x = (x - self.mean) / self.std
            
        features = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layer_ids:
                features[i] = x
                
        return features


class PerceptualLoss(nn.Module):
    """
    perceptual loss using vgg features.
    
    computes l1/l2 distance between feature representations.
    
    args:
        layer_weights: weights for each layer's contribution
        loss_type: 'l1' or 'l2'
        normalize: whether to normalize features
    """
    
    def __init__(
        self,
        layer_weights: Dict[int, float] = None,
        loss_type: str = 'l1',
        normalize: bool = True
    ):
        super().__init__()
        
        default_weights = {3: 1.0, 8: 1.0, 15: 1.0, 22: 1.0}
        self.layer_weights = layer_weights or default_weights
        
        self.feature_extractor = VGGFeatureExtractor(
            layer_ids=list(self.layer_weights.keys()),
            use_input_norm=True,
            requires_grad=False
        )
        
        self.loss_type = loss_type
        self.normalize = normalize
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        compute perceptual loss.
        
        args:
            pred: predicted tensor [b, c, h, w]
            target: target tensor [b, c, h, w]
            
        returns:
            perceptual loss value
        """
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        total_loss = 0.0
        
        for layer_id, weight in self.layer_weights.items():
            pred_feat = pred_features[layer_id]
            target_feat = target_features[layer_id]
            
            if self.normalize:
                pred_feat = F.normalize(pred_feat, dim=1)
                target_feat = F.normalize(target_feat, dim=1)
                
            if self.loss_type == 'l1':
                loss = F.l1_loss(pred_feat, target_feat)
            else:
                loss = F.mse_loss(pred_feat, target_feat)
                
            total_loss += weight * loss
            
        return total_loss / sum(self.layer_weights.values())


class LPIPSLoss(nn.Module):
    """
    learned perceptual image patch similarity (lpips).
    
    uses learned linear weights on vgg features.
    
    args:
        net: network type ('vgg', 'alex')
    """
    
    def __init__(self, net: str = 'vgg'):
        super().__init__()
        
        # vgg feature extractor
        self.feature_extractor = VGGFeatureExtractor(
            layer_ids=[3, 8, 15, 22, 29],  # all relu layers
            use_input_norm=True,
            requires_grad=False
        )
        
        # channel dimensions for each layer
        channels = [64, 128, 256, 512, 512]
        
        # learned linear layers (simplified - in practice, load pretrained weights)
        self.linear_layers = nn.ModuleDict({
            str(3): nn.Conv2d(64, 1, 1, bias=False),
            str(8): nn.Conv2d(128, 1, 1, bias=False),
            str(15): nn.Conv2d(256, 1, 1, bias=False),
            str(22): nn.Conv2d(512, 1, 1, bias=False),
            str(29): nn.Conv2d(512, 1, 1, bias=False),
        })
        
        # initialize with uniform weights
        for layer in self.linear_layers.values():
            nn.init.ones_(layer.weight)
            
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        compute lpips loss.
        
        args:
            pred: predicted tensor [b, c, h, w]
            target: target tensor [b, c, h, w]
            
        returns:
            lpips loss value
        """
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        total_loss = 0.0
        
        for layer_id in self.linear_layers.keys():
            pred_feat = pred_features[int(layer_id)]
            target_feat = target_features[int(layer_id)]
            
            # normalize features
            pred_feat = F.normalize(pred_feat, dim=1)
            target_feat = F.normalize(target_feat, dim=1)
            
            # squared difference
            diff = (pred_feat - target_feat) ** 2
            
            # apply learned weights and average
            weighted = self.linear_layers[layer_id](diff)
            total_loss += weighted.mean()
            
        return total_loss


class StyleLoss(nn.Module):
    """style loss based on gram matrix matching."""
    
    def __init__(
        self,
        feature_layers: List[str] = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],
        weights: Optional[List[float]] = None,
        normalize: bool = True
    ):
        """
        initialize style loss.
        
        args:
            feature_layers: vgg layers to use for style matching
            weights: weights for each layer
            normalize: whether to normalize gram matrices
        """
        super().__init__()
        self.feature_extractor = VGGFeatureExtractor(feature_layers)
        self.weights = weights or [1.0] * len(feature_layers)
        self.normalize = normalize
        
    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """compute gram matrix for style representation."""
        B, C, H, W = x.shape
        features = x.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))
        
        if self.normalize:
            gram = gram / (C * H * W)
            
        return gram
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        compute style loss.
        
        args:
            pred: predicted tensor [b, c, h, w]
            target: target tensor [b, c, h, w]
            
        returns:
            style loss value
        """
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        total_loss = 0.0
        
        for i, (pred_feat, target_feat) in enumerate(zip(pred_features, target_features)):
            pred_gram = self.gram_matrix(pred_feat)
            target_gram = self.gram_matrix(target_feat)
            
            layer_loss = F.mse_loss(pred_gram, target_gram)
            total_loss += self.weights[i] * layer_loss
            
        return total_loss / sum(self.weights)


class ContentStyleLoss(nn.Module):
    """combined content and style loss."""
    
    def __init__(
        self,
        content_weight: float = 1.0,
        style_weight: float = 1e-2,
        content_layers: List[str] = ['relu3_3'],
        style_layers: List[str] = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
    ):
        """
        initialize combined content-style loss.
        
        args:
            content_weight: weight for content loss
            style_weight: weight for style loss
            content_layers: vgg layers for content
            style_layers: vgg layers for style
        """
        super().__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.content_loss = PerceptualLoss(feature_layers=content_layers)
        self.style_loss = StyleLoss(feature_layers=style_layers)
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        compute combined content and style loss.
        
        args:
            pred: predicted tensor [b, c, h, w]
            target: target tensor [b, c, h, w]
            
        returns:
            tuple of (total_loss, content_loss, style_loss)
        """
        content = self.content_loss(pred, target)
        style = self.style_loss(pred, target)
        
        total = self.content_weight * content + self.style_weight * style
        
        return total, content, style
