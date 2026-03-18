"""
neuroscope training infrastructure package.

comprehensive training utilities for deep learning models
including trainers, optimizers, schedulers, callbacks, and loggers.

subpackages:
    - trainers: main training loop implementations
    - optimizers: optimizer factories and configurations
    - schedulers: learning rate scheduling strategies
    - callbacks: training callbacks for logging, checkpointing, etc.
    - loggers: comprehensive logging infrastructure
    - samplers: sample generation during training
    - figures: publication-quality figure generation
"""

# lazy imports to avoid circular dependencies and missing module errors
# users should import from subpackages directly:
#   from neuroscope.training.trainers import comprehensivetrainer
#   from neuroscope.training.loggers import loggermanager
#   from neuroscope.training.callbacks import earlystopping

__all__ = [
    'trainers',
    'optimizers', 
    'schedulers',
    'callbacks',
    'loggers',
    'samplers',
    'figures',
]
