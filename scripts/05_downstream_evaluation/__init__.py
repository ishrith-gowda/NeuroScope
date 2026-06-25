"""
downstream task evaluation module for mri harmonization.

this module implements comprehensive downstream evaluation to assess the
clinical impact of harmonization on machine learning tasks.

evaluation approaches:
1. domain classification - train classifier to distinguish domains,
   harmonization should reduce discriminability
2. feature distribution analysis - compute fid, mmd, kid, swd to
   quantify distribution shift before/after harmonization
3. segmentation (when labels available) - evaluate tumor segmentation
   performance on harmonized vs raw data

all methods designed for publication at top-tier venues.
"""

from .domain_classifier import (
    DomainClassifier,
    MRIDomainDataset,
    NiftiDomainDataset,
    evaluate_domain_classifier,
    train_domain_classifier,
)
from .feature_distribution_analysis import (
    FeatureExtractor,
    compute_fid,
    compute_kid,
    compute_mmd,
    compute_sliced_wasserstein,
)
from .unet_segmentation import (
    CombinedLoss,
    DiceLoss,
    UNet2D,
    compute_dice_score,
)

__all__ = [
    "CombinedLoss",
    "DiceLoss",
    "DomainClassifier",
    "FeatureExtractor",
    "MRIDomainDataset",
    "NiftiDomainDataset",
    "UNet2D",
    "compute_dice_score",
    "compute_fid",
    "compute_kid",
    "compute_mmd",
    "compute_sliced_wasserstein",
    "evaluate_domain_classifier",
    "train_domain_classifier",
]
