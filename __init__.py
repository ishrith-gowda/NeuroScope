"""NeuroScope: Domain-aware standardization of multimodal glioma MRI.

A comprehensive framework for CycleGAN-based domain adaptation between
multi-institutional glioblastoma MRI scans (T1, T1ce, T2, FLAIR).

This package provides:
- Advanced preprocessing pipelines for medical imaging data
- CycleGAN model implementations for domain adaptation
- Comprehensive training and evaluation tools
- Bias assessment and quality control utilities
- Professional visualization and reporting capabilities

Example:
    >>> import neuroscope
    >>> from neuroscope.config import get_default_training_config
    >>> config = get_default_training_config()
    >>> # Configure and run training pipeline
"""

__version__ = "0.1.0"
__author__ = "Ishrith Gowda"
__email__ = "your.email@example.com"

# Import core modules
from . import core
from . import data
from . import models
from . import preprocessing
from . import training
from . import evaluation
from . import visualization
from . import utils
from . import config

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "core",
    "data",
    "models",
    "preprocessing",
    "training",
    "evaluation",
    "visualization",
    "utils",
    "config"
]