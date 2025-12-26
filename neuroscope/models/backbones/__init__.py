"""
Backbone Feature Extraction Networks.

Pre-trained networks for feature extraction used in
perceptual losses and feature matching.
"""

from .vgg import (
    VGG16Features,
    VGG19Features,
    VGGPerceptualExtractor,
    MultiLayerVGG,
)

from .resnet import (
    ResNet18Features,
    ResNet34Features,
    ResNet50Features,
    ResNetPerceptualExtractor,
    MultiScaleResNetFeatures,
)

from .efficientnet import (
    EfficientNetB0Features,
    EfficientNetB4Features,
    EfficientNetFeatureExtractor,
    HybridFeatureExtractor,
)

__all__ = [
    # VGG
    'VGG16Features',
    'VGG19Features',
    'VGGPerceptualExtractor',
    'MultiLayerVGG',
    
    # ResNet
    'ResNet18Features',
    'ResNet34Features',
    'ResNet50Features',
    'ResNetPerceptualExtractor',
    'MultiScaleResNetFeatures',
    
    # EfficientNet
    'EfficientNetB0Features',
    'EfficientNetB4Features',
    'EfficientNetFeatureExtractor',
    'HybridFeatureExtractor',
]
