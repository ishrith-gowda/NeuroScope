"""
Training Loggers Package.

Comprehensive logging infrastructure for 2.5D SA-CycleGAN training.

Includes:
- TensorBoardLogger: TensorBoard logging
- CSVLogger: Tabular metrics export
- JSONLogger: Structured JSON logs
- ConsoleLogger: Rich console output
- LoggerManager: Unified logging interface
- MetricsAggregator: Batch-level metrics aggregation

Author: NeuroScope Research Team
"""

from .tensorboard_logger import TensorBoardLogger
from .file_loggers import CSVLogger, JSONLogger, MetricsAggregator
from .console_logger import ConsoleLogger, Colors, colorize
from .manager import LoggerManager


__all__ = [
    'TensorBoardLogger',
    'CSVLogger',
    'JSONLogger',
    'MetricsAggregator',
    'ConsoleLogger',
    'Colors',
    'colorize',
    'LoggerManager',
]
