"""
NeuroScope Training Infrastructure Package.

Comprehensive training utilities for deep learning models
including trainers, optimizers, schedulers, callbacks, and loggers.

Subpackages:
    - trainers: Main training loop implementations
    - optimizers: Optimizer factories and configurations
    - schedulers: Learning rate scheduling strategies
    - callbacks: Training callbacks for logging, checkpointing, etc.
    - loggers: Comprehensive logging infrastructure
    - samplers: Sample generation during training
    - figures: Publication-quality figure generation
"""

# Lazy imports to avoid circular dependencies and missing module errors
# Users should import from subpackages directly:
#   from neuroscope.training.trainers import ComprehensiveTrainer
#   from neuroscope.training.loggers import LoggerManager
#   from neuroscope.training.callbacks import EarlyStopping

__all__ = [
    'trainers',
    'optimizers', 
    'schedulers',
    'callbacks',
    'loggers',
    'samplers',
    'figures',
]
