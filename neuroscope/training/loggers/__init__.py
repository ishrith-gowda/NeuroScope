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

from .console_logger import Colors, ConsoleLogger, colorize
from .file_loggers import CSVLogger, JSONLogger, MetricsAggregator
from .manager import LoggerManager
from .tensorboard_logger import TensorBoardLogger

__all__ = [
    "CSVLogger",
    "Colors",
    "ConsoleLogger",
    "JSONLogger",
    "LoggerManager",
    "MetricsAggregator",
    "TensorBoardLogger",
    "colorize",
]
