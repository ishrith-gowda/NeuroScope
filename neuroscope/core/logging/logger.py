"""setup logging for neuroscope."""

import logging
import sys
from pathlib import Path
from typing import Optional

from neuroscope.core.config import config


def setup_logging(
    log_file: Optional[Path] = None,
    log_level: int = logging.INFO,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
):
    """set up logging for neuroscope.
    
    args:
        log_file: path to log file. if none, logs will only be printed to console.
        log_level: logging level.
        log_format: logging format.
    """
    # create logger
    logger = logging.getLogger("neuroscope")
    logger.setLevel(log_level)
    
    # create formatter
    formatter = logging.Formatter(log_format)
    
    # create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # create file handler if log_file is provided
    if log_file is not None:
        # create directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """get logger with specified name.
    
    args:
        name: logger name.
    
    returns:
        logger instance.
    """
    return logging.getLogger(f"neuroscope.{name}")


# setup default logger
default_log_file = config.paths.get("logs_dir") / "neuroscope.log"
logger = setup_logging(default_log_file)

__all__ = ["setup_logging", "get_logger", "logger"]