"""
backbone feature extraction networks.

pre-trained networks for feature extraction used in
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
    # vgg
    'VGG16Features',
    'VGG19Features',
    'VGGPerceptualExtractor',
    'MultiLayerVGG',
    
    # resnet
    'ResNet18Features',
    'ResNet34Features',
    'ResNet50Features',
    'ResNetPerceptualExtractor',
    'MultiScaleResNetFeatures',
    
    # efficientnet
    'EfficientNetB0Features',
    'EfficientNetB4Features',
    'EfficientNetFeatureExtractor',
    'HybridFeatureExtractor',
]
