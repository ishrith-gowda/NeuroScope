"""
neuroscope data pipeline package.

comprehensive data loading, preprocessing, and augmentation
for multi-modal medical image analysis.

modules:
    - datasets: dataset classes for brats, upenn-gbm, and custom data
    - transforms: image transformation and augmentation pipelines
    - samplers: custom sampling strategies for balanced training
    - loaders: dataloader factories with proper configuration
"""

from .datasets import (
    # utilities
    DATASET_REGISTRY,
    ABIDEDataset,
    ADNIDataset,
    # base classes
    BaseMedicalDataset,
    HCPDataset,
    # medical datasets
    IXIDataset,
    OASISDataset,
    TCGAGBMDataset,
    VolumetricDataset,
    create_medical_dataset,
)
from .loaders import (
    InfiniteDataLoader,
    PrefetchDataLoader,
    create_dataloader,
    create_test_loader,
    create_train_loader,
    create_val_loader,
)
from .samplers import (
    BalancedSampler,
    DomainBalancedSampler,
    StratifiedSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
)
from .transforms import (
    AdaptiveHistogramEqualization,
    # base transforms
    BaseTransform,
    BiasFieldAugmentation,
    CenterCrop,
    Compose,
    ElasticDeformation,
    HistogramEqualization,
    IntensityClipping,
    # intensity transforms
    IntensityNormalization,
    MinMaxNormalization,
    # medical-specific transforms
    N4BiasFieldCorrection,
    PercentileNormalization,
    RandomAffine,
    RandomBlur,
    RandomBrightnessContrast,
    # spatial transforms
    RandomCrop,
    RandomFlip,
    RandomGamma,
    # augmentation transforms
    RandomNoise,
    RandomRotation,
    Resize,
    SkullStripping,
    ZScoreNormalization,
    create_test_transforms,
    # pipeline builders
    create_train_transforms,
    create_val_transforms,
)

# aliases for compatibility
BraTSDataset = TCGAGBMDataset  # alias for backward compatibility
UPennGBMDataset = TCGAGBMDataset  # using same base class
MultiModalMRIDataset = VolumetricDataset  # alias

__all__ = [
    # dataset utilities
    "DATASET_REGISTRY",
    "ABIDEDataset",
    "ADNIDataset",
    "AdaptiveHistogramEqualization",
    # samplers
    "BalancedSampler",
    # datasets - base
    "BaseMedicalDataset",
    # transforms - base
    "BaseTransform",
    "BiasFieldAugmentation",
    "BraTSDataset",  # alias
    "CenterCrop",
    "Compose",
    "DomainBalancedSampler",
    "ElasticDeformation",
    "HCPDataset",
    "HistogramEqualization",
    # datasets - medical
    "IXIDataset",
    "InfiniteDataLoader",
    "IntensityClipping",
    # transforms - intensity
    "IntensityNormalization",
    "MinMaxNormalization",
    "MultiModalMRIDataset",  # alias
    # transforms - medical
    "N4BiasFieldCorrection",
    "OASISDataset",
    "PercentileNormalization",
    "PrefetchDataLoader",
    "RandomAffine",
    "RandomBlur",
    "RandomBrightnessContrast",
    # transforms - spatial
    "RandomCrop",
    "RandomFlip",
    "RandomGamma",
    # transforms - augmentation
    "RandomNoise",
    "RandomRotation",
    "Resize",
    "SkullStripping",
    "StratifiedSampler",
    "SubsetRandomSampler",
    "TCGAGBMDataset",
    "UPennGBMDataset",  # alias
    "VolumetricDataset",
    "WeightedRandomSampler",
    "ZScoreNormalization",
    # loaders
    "create_dataloader",
    "create_medical_dataset",
    "create_test_loader",
    "create_test_transforms",
    "create_train_loader",
    # transform builders
    "create_train_transforms",
    "create_val_loader",
    "create_val_transforms",
]
