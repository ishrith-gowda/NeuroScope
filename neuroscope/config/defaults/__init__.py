"""Default configurations for NeuroScope.

This module provides default configuration values for all NeuroScope components.
"""

from .training_config import (
    get_default_training_config,
    get_default_preprocessing_config,
    get_default_evaluation_config,
    merge_configs,
    validate_config
)

__all__ = [
    "get_default_training_config",
    "get_default_preprocessing_config",
    "get_default_evaluation_config", 
    "merge_configs",
    "validate_config"
]