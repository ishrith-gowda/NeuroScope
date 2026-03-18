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
    # base classes
    BaseMedicalDataset,

    # medical datasets
    IXIDataset,
    OASISDataset,
    ADNIDataset,
    ABIDEDataset,
    HCPDataset,
    TCGAGBMDataset,
    VolumetricDataset,

    # utilities
    DATASET_REGISTRY,
    create_medical_dataset,
)

from .transforms import (
    # base transforms
    BaseTransform,
    Compose,
    
    # intensity transforms
    IntensityNormalization,
    ZScoreNormalization,
    MinMaxNormalization,
    PercentileNormalization,
    HistogramEqualization,
    AdaptiveHistogramEqualization,
    
    # spatial transforms
    RandomCrop,
    CenterCrop,
    Resize,
    RandomFlip,
    RandomRotation,
    RandomAffine,
    ElasticDeformation,
    
    # augmentation transforms
    RandomNoise,
    RandomBlur,
    RandomBrightnessContrast,
    RandomGamma,
    BiasFieldAugmentation,
    
    # medical-specific transforms
    N4BiasFieldCorrection,
    SkullStripping,
    IntensityClipping,
    
    # pipeline builders
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

# aliases for compatibility
BraTSDataset = TCGAGBMDataset  # alias for backward compatibility
UPennGBMDataset = TCGAGBMDataset  # using same base class
MultiModalMRIDataset = VolumetricDataset  # alias

__all__ = [
    # datasets - base
    'BaseMedicalDataset',

    # datasets - medical
    'IXIDataset',
    'OASISDataset',
    'ADNIDataset',
    'ABIDEDataset',
    'HCPDataset',
    'TCGAGBMDataset',
    'VolumetricDataset',
    'BraTSDataset',  # alias
    'UPennGBMDataset',  # alias
    'MultiModalMRIDataset',  # alias

    # dataset utilities
    'DATASET_REGISTRY',
    'create_medical_dataset',
    
    # transforms - base
    'BaseTransform',
    'Compose',
    
    # transforms - intensity
    'IntensityNormalization',
    'ZScoreNormalization',
    'MinMaxNormalization',
    'PercentileNormalization',
    'HistogramEqualization',
    'AdaptiveHistogramEqualization',
    
    # transforms - spatial
    'RandomCrop',
    'CenterCrop',
    'Resize',
    'RandomFlip',
    'RandomRotation',
    'RandomAffine',
    'ElasticDeformation',
    
    # transforms - augmentation
    'RandomNoise',
    'RandomBlur',
    'RandomBrightnessContrast',
    'RandomGamma',
    'BiasFieldAugmentation',
    
    # transforms - medical
    'N4BiasFieldCorrection',
    'SkullStripping',
    'IntensityClipping',
    
    # transform builders
    'create_train_transforms',
    'create_val_transforms',
    'create_test_transforms',
    
    # samplers
    'BalancedSampler',
    'WeightedRandomSampler',
    'DomainBalancedSampler',
    'StratifiedSampler',
    'SubsetRandomSampler',
    
    # loaders
    'create_dataloader',
    'create_train_loader',
    'create_val_loader',
    'create_test_loader',
    'InfiniteDataLoader',
    'PrefetchDataLoader',
]
