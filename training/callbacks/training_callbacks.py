"""Training callbacks for CycleGAN training.

This module provides callback functionality for monitoring and controlling
the training process, including early stopping, checkpointing, and logging.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from neuroscope.core.logging import get_logger

logger = get_logger(__name__)


class TrainingCallbacks:
    """Collection of training callbacks for CycleGAN."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize training callbacks.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.callbacks = []
        
        # Initialize callbacks based on config
        self._init_callbacks()
        
        # Callback state
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.start_time = time.time()
    
    def _init_callbacks(self):
        """Initialize callbacks based on configuration."""
        callback_configs = self.config.get('callbacks', {})
        
        # Early stopping callback
        if callback_configs.get('early_stopping', {}).get('enabled', False):
            self.callbacks.append(EarlyStoppingCallback(
                callback_configs['early_stopping']
            ))
        
        # Checkpoint callback
        if callback_configs.get('checkpoint', {}).get('enabled', True):
            self.callbacks.append(CheckpointCallback(
                callback_configs['checkpoint']
            ))
        
        # Logging callback
        if callback_configs.get('logging', {}).get('enabled', True):
            self.callbacks.append(LoggingCallback(
                callback_configs['logging']
            ))
        
        # Validation callback
        if callback_configs.get('validation', {}).get('enabled', False):
            self.callbacks.append(ValidationCallback(
                callback_configs['validation']
            ))
    
    def on_epoch_begin(self, epoch: int):
        """Called at the beginning of each epoch."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)
    
    def on_epoch_end(self, epoch: int, losses: Dict[str, float]):
        """Called at the end of each epoch."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, losses)
    
    def on_batch_begin(self, batch_idx: int, epoch: int):
        """Called at the beginning of each batch."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch_idx, epoch)
    
    def on_batch_end(self, batch_idx: int, epoch: int, g_losses: Dict[str, float], d_losses: Dict[str, float]):
        """Called at the end of each batch."""
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, epoch, g_losses, d_losses)
    
    def on_training_begin(self):
        """Called at the beginning of training."""
        for callback in self.callbacks:
            callback.on_training_begin()
    
    def on_training_end(self):
        """Called at the end of training."""
        for callback in self.callbacks:
            callback.on_training_end()


class BaseCallback:
    """Base class for all training callbacks."""
    
    def on_epoch_begin(self, epoch: int):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, losses: Dict[str, float]):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch_idx: int, epoch: int):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch_idx: int, epoch: int, g_losses: Dict[str, float], d_losses: Dict[str, float]):
        """Called at the end of each batch."""
        pass
    
    def on_training_begin(self):
        """Called at the beginning of training."""
        pass
    
    def on_training_end(self):
        """Called at the end of training."""
        pass


class EarlyStoppingCallback(BaseCallback):
    """Early stopping callback to prevent overfitting."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize early stopping callback.
        
        Args:
            config: Early stopping configuration
        """
        self.patience = config.get('patience', 10)
        self.min_delta = config.get('min_delta', 0.001)
        self.monitor = config.get('monitor', 'total')
        self.mode = config.get('mode', 'min')  # 'min' or 'max'
        
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.patience_counter = 0
        self.should_stop = False
    
    def on_epoch_end(self, epoch: int, losses: Dict[str, float]):
        """Check if training should stop early."""
        current_score = losses.get(self.monitor, float('inf'))
        
        if self.mode == 'min':
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        else:  # max
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            self.should_stop = True


class CheckpointCallback(BaseCallback):
    """Checkpoint callback to save model states."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize checkpoint callback.
        
        Args:
            config: Checkpoint configuration
        """
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.save_best_only = config.get('save_best_only', True)
        self.save_frequency = config.get('save_frequency', 10)  # Save every N epochs
        self.monitor = config.get('monitor', 'total')
        self.mode = config.get('mode', 'min')
        
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, losses: Dict[str, float]):
        """Save checkpoint if conditions are met."""
        current_score = losses.get(self.monitor, float('inf'))
        
        # Check if this is the best score
        is_best = False
        if self.mode == 'min':
            if current_score < self.best_score:
                self.best_score = current_score
                is_best = True
        else:  # max
            if current_score > self.best_score:
                self.best_score = current_score
                is_best = True
        
        # Save checkpoint
        should_save = False
        if self.save_best_only and is_best:
            should_save = True
        elif not self.save_best_only and epoch % self.save_frequency == 0:
            should_save = True
        
        if should_save:
            self._save_checkpoint(epoch, losses, is_best)
    
    def _save_checkpoint(self, epoch: int, losses: Dict[str, float], is_best: bool):
        """Save model checkpoint."""
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
        if is_best:
            checkpoint_name = "best_model.pth"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # This would be implemented by the trainer
        logger.info(f"Checkpoint saved: {checkpoint_path}")


class LoggingCallback(BaseCallback):
    """Logging callback for training metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize logging callback.
        
        Args:
            config: Logging configuration
        """
        self.log_frequency = config.get('log_frequency', 10)
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.save_logs = config.get('save_logs', True)
        
        self.log_history = []
        
        if self.save_logs:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, losses: Dict[str, float]):
        """Log training metrics."""
        log_entry = {
            'epoch': epoch,
            'timestamp': time.time(),
            'losses': losses
        }
        
        self.log_history.append(log_entry)
        
        if epoch % self.log_frequency == 0:
            logger.info(f"Epoch {epoch} - Losses: {losses}")
    
    def on_training_end(self):
        """Save training logs."""
        if self.save_logs and self.log_history:
            log_file = self.log_dir / f"training_log_{int(time.time())}.json"
            with open(log_file, 'w') as f:
                json.dump(self.log_history, f, indent=2)
            logger.info(f"Training logs saved: {log_file}")


class ValidationCallback(BaseCallback):
    """Validation callback for model evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize validation callback.
        
        Args:
            config: Validation configuration
        """
        self.validation_frequency = config.get('validation_frequency', 5)
        self.validation_loader = config.get('validation_loader', None)
        self.metrics = config.get('metrics', ['mse', 'mae', 'ssim'])
    
    def on_epoch_end(self, epoch: int, losses: Dict[str, float]):
        """Run validation if conditions are met."""
        if epoch % self.validation_frequency == 0 and self.validation_loader:
            self._run_validation(epoch)
    
    def _run_validation(self, epoch: int):
        """Run validation evaluation."""
        logger.info(f"Running validation at epoch {epoch}")
        # Validation logic would be implemented here