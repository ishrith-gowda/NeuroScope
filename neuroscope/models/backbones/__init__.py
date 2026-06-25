"""
backbone feature extraction networks.

pre-trained networks for feature extraction used in
perceptual losses and feature matching.
"""

from .efficientnet import (
    EfficientNetB0Features,
    EfficientNetB4Features,
    EfficientNetFeatureExtractor,
    HybridFeatureExtractor,
)
from .resnet import (
    MultiScaleResNetFeatures,
    ResNet18Features,
    ResNet34Features,
    ResNet50Features,
    ResNetPerceptualExtractor,
)
from .vgg import (
    MultiLayerVGG,
    VGG16Features,
    VGG19Features,
    VGGPerceptualExtractor,
)

__all__ = [
    # efficientnet
    "EfficientNetB0Features",
    "EfficientNetB4Features",
    "EfficientNetFeatureExtractor",
    "HybridFeatureExtractor",
    "MultiLayerVGG",
    "MultiScaleResNetFeatures",
    # resnet
    "ResNet18Features",
    "ResNet34Features",
    "ResNet50Features",
    "ResNetPerceptualExtractor",
    # vgg
    "VGG16Features",
    "VGG19Features",
    "VGGPerceptualExtractor",
]
