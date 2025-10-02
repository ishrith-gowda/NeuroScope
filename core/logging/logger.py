"""Centralized logging configuration for NeuroScope.

This module provides a unified logging system with consistent formatting
and configuration across all NeuroScope modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class NeuroScopeLogger:
    """Centralized logger for NeuroScope with enhanced functionality."""
    
    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        log_file: Optional[Path] = None,
        format_string: Optional[str] = None
    ):
        """Initialize NeuroScope logger.
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Optional log file path
            format_string: Custom format string
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Default format
        if format_string is None:
            format_string = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(filename)s:%(lineno)d - %(message)s"
            )
        
        formatter = logging.Formatter(format_string)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, **kwargs)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics in structured format.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'metrics': metrics
        }
        
        self.info(f"METRICS: {json.dumps(log_data)}")
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information.
        
        Args:
            model_info: Dictionary containing model information
        """
        self.info(f"MODEL_INFO: {json.dumps(model_info)}")
    
    def log_training_step(
        self,
        epoch: int,
        batch_idx: int,
        losses: Dict[str, float],
        lr: Optional[float] = None
    ):
        """Log training step information.
        
        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            losses: Dictionary of losses
            lr: Optional learning rate
        """
        log_data = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'losses': losses,
            'learning_rate': lr
        }
        
        self.info(f"TRAINING_STEP: {json.dumps(log_data)}")


# Global logger registry
_loggers: Dict[str, NeuroScopeLogger] = {}


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None
) -> NeuroScopeLogger:
    """Get or create a logger instance.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        NeuroScopeLogger instance
    """
    if name not in _loggers:
        _loggers[name] = NeuroScopeLogger(name, level, log_file)
    
    return _loggers[name]


def configure_logging(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    format_string: Optional[str] = None
):
    """Configure global logging settings.
    
    Args:
        level: Default logging level
        log_dir: Directory for log files
        format_string: Custom format string
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log directory specified)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"neuroscope_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy loggers
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def set_log_level(level: int):
    """Set logging level for all loggers.
    
    Args:
        level: Logging level
    """
    for logger in _loggers.values():
        logger.logger.setLevel(level)
    
    # Also set root logger level
    logging.getLogger().setLevel(level)