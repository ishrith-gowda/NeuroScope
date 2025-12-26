"""
NeuroScope Data Pipeline Package.

Comprehensive data loading, preprocessing, and augmentation
for multi-modal medical image analysis.

Modules:
    - datasets: Dataset classes for BraTS, UPenn-GBM, and custom data
    - transforms: Image transformation and augmentation pipelines
    - samplers: Custom sampling strategies for balanced training
    - loaders: DataLoader factories with proper configuration
"""

from .datasets import (
    # Base classes
    BaseMedicalDataset,

    # Medical datasets
    IXIDataset,
    OASISDataset,
    ADNIDataset,
    ABIDEDataset,
    HCPDataset,
    TCGAGBMDataset,
    VolumetricDataset,

    # Utilities
    DATASET_REGISTRY,
    create_medical_dataset,
)

from .transforms import (
    # Base transforms
    BaseTransform,
    Compose,
    
    # Intensity transforms
    IntensityNormalization,
    ZScoreNormalization,
    MinMaxNormalization,
    PercentileNormalization,
    HistogramEqualization,
    AdaptiveHistogramEqualization,
    
    # Spatial transforms
    RandomCrop,
    CenterCrop,
    Resize,
    RandomFlip,
    RandomRotation,
    RandomAffine,
    ElasticDeformation,
    
    # Augmentation transforms
    RandomNoise,
    RandomBlur,
    RandomBrightnessContrast,
    RandomGamma,
    BiasFieldAugmentation,
    
    # Medical-specific transforms
    N4BiasFieldCorrection,
    SkullStripping,
    IntensityClipping,
    
    # Pipeline builders
    create_train_transforms,
    create_val_transforms,
    create_test_transforms,
)

from .samplers import (
    BalancedSampler,
    WeightedRandomSampler,
    DomainBalancedSampler,
    StratifiedSampler,
    SubsetRandomSampler,
)

from .loaders import (
    create_dataloader,
    create_train_loader,
    create_val_loader,
    create_test_loader,
    InfiniteDataLoader,
    PrefetchDataLoader,
)

# Aliases for compatibility
BraTSDataset = TCGAGBMDataset  # Alias for backward compatibility
UPennGBMDataset = TCGAGBMDataset  # Using same base class
MultiModalMRIDataset = VolumetricDataset  # Alias

__all__ = [
    # Datasets - Base
    'BaseMedicalDataset',

    # Datasets - Medical
    'IXIDataset',
    'OASISDataset',
    'ADNIDataset',
    'ABIDEDataset',
    'HCPDataset',
    'TCGAGBMDataset',
    'VolumetricDataset',
    'BraTSDataset',  # Alias
    'UPennGBMDataset',  # Alias
    'MultiModalMRIDataset',  # Alias

    # Dataset utilities
    'DATASET_REGISTRY',
    'create_medical_dataset',
    
    # Transforms - Base
    'BaseTransform',
    'Compose',
    
    # Transforms - Intensity
    'IntensityNormalization',
    'ZScoreNormalization',
    'MinMaxNormalization',
    'PercentileNormalization',
    'HistogramEqualization',
    'AdaptiveHistogramEqualization',
    
    # Transforms - Spatial
    'RandomCrop',
    'CenterCrop',
    'Resize',
    'RandomFlip',
    'RandomRotation',
    'RandomAffine',
    'ElasticDeformation',
    
    # Transforms - Augmentation
    'RandomNoise',
    'RandomBlur',
    'RandomBrightnessContrast',
    'RandomGamma',
    'BiasFieldAugmentation',
    
    # Transforms - Medical
    'N4BiasFieldCorrection',
    'SkullStripping',
    'IntensityClipping',
    
    # Transform builders
    'create_train_transforms',
    'create_val_transforms',
    'create_test_transforms',
    
    # Samplers
    'BalancedSampler',
    'WeightedRandomSampler',
    'DomainBalancedSampler',
    'StratifiedSampler',
    'SubsetRandomSampler',
    
    # Loaders
    'create_dataloader',
    'create_train_loader',
    'create_val_loader',
    'create_test_loader',
    'InfiniteDataLoader',
    'PrefetchDataLoader',
]
