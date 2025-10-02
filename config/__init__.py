"""Configuration management for NeuroScope.

This module provides comprehensive configuration management including
default configurations, validation, and dynamic configuration loading.
"""

from .defaults.training_config import (
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