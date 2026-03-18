"""
training loggers package.

comprehensive logging infrastructure for 2.5d sa-cyclegan training.

includes:
- tensorboardlogger: tensorboard logging
- csvlogger: tabular metrics export
- jsonlogger: structured json logs
- consolelogger: rich console output
- loggermanager: unified logging interface
- metricsaggregator: batch-level metrics aggregation

author: neuroscope research team
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
