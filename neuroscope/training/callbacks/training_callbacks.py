"""
Training Callbacks.

Comprehensive collection of training callbacks for monitoring,
checkpointing, and early stopping.
"""

from typing import Optional, Dict, Any, List, Callable, Union
from pathlib import Path
from dataclasses import dataclass, field
import time
import json
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


@dataclass
class CallbackState:
    """State passed to callbacks."""
    epoch: int = 0
    step: int = 0
    total_epochs: int = 0
    total_steps: int = 0
    train_loss: float = 0.0
    val_loss: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    model: Optional[nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class Callback(ABC):
    """Base callback class."""
    
    def on_train_begin(self, state: CallbackState) -> None:
        """Called at the start of training."""
        pass
    
    def on_train_end(self, state: CallbackState) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, state: CallbackState) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, state: CallbackState) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, state: CallbackState) -> None:
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, state: CallbackState) -> None:
        """Called at the end of each batch."""
        pass
    
    def on_validation_begin(self, state: CallbackState) -> None:
        """Called at the start of validation."""
        pass
    
    def on_validation_end(self, state: CallbackState) -> None:
        """Called at the end of validation."""
        pass


class CallbackList:
    """Container for multiple callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)
    
    def on_train_begin(self, state: CallbackState) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(state)
    
    def on_train_end(self, state: CallbackState) -> None:
        for callback in self.callbacks:
            callback.on_train_end(state)
    
    def on_epoch_begin(self, state: CallbackState) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(state)
    
    def on_epoch_end(self, state: CallbackState) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(state)
    
    def on_batch_begin(self, state: CallbackState) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin(state)
    
    def on_batch_end(self, state: CallbackState) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(state)


class EarlyStopping(Callback):
    """
    Early stopping callback.
    
    Stops training when a monitored metric stops improving.
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            monitor: Metric to monitor
            patience: Epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
            restore_best_weights: Whether to restore best model
            verbose: Whether to print messages
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.counter = 0
        self.stopped = False
        self.best_weights = None
    
    def _is_improvement(self, current: float) -> bool:
        if self.mode == 'min':
            return current < self.best - self.min_delta
        return current > self.best + self.min_delta
    
    def on_epoch_end(self, state: CallbackState) -> None:
        current = state.metrics.get(self.monitor, state.val_loss)
        
        if current is None:
            return
        
        if self._is_improvement(current):
            self.best = current
            self.best_epoch = state.epoch
            self.counter = 0
            
            if self.restore_best_weights and state.model is not None:
                self.best_weights = {
                    k: v.cpu().clone() 
                    for k, v in state.model.state_dict().items()
                }
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.stopped = True
                
                if self.verbose:
                    print(f"\nEarly stopping at epoch {state.epoch}")
                    print(f"Best {self.monitor}: {self.best:.6f} at epoch {self.best_epoch}")
    
    def on_train_end(self, state: CallbackState) -> None:
        if self.restore_best_weights and self.best_weights is not None:
            if state.model is not None:
                state.model.load_state_dict(self.best_weights)
                if self.verbose:
                    print(f"Restored best weights from epoch {self.best_epoch}")


class ModelCheckpoint(Callback):
    """
    Model checkpoint callback.
    
    Saves model checkpoints during training.
    """
    
    def __init__(
        self,
        filepath: Union[str, Path],
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_weights_only: bool = False,
        save_freq: Union[str, int] = 'epoch',
        verbose: bool = True
    ):
        """
        Args:
            filepath: Path template for checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save when improved
            save_weights_only: Only save weights
            save_freq: 'epoch' or number of batches
            verbose: Whether to print messages
        """
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.verbose = verbose
        
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def _is_improvement(self, current: float) -> bool:
        if self.mode == 'min':
            return current < self.best
        return current > self.best
    
    def _format_filepath(self, state: CallbackState) -> Path:
        name = self.filepath.name.format(
            epoch=state.epoch,
            step=state.step,
            **state.metrics
        )
        return self.filepath.parent / name
    
    def _save_checkpoint(self, state: CallbackState) -> None:
        filepath = self._format_filepath(state)
        
        if self.save_weights_only:
            checkpoint = state.model.state_dict()
        else:
            checkpoint = {
                'epoch': state.epoch,
                'step': state.step,
                'model_state_dict': state.model.state_dict(),
                'metrics': state.metrics
            }
            
            if state.optimizer is not None:
                checkpoint['optimizer_state_dict'] = state.optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        
        if self.verbose:
            print(f"Saved checkpoint to {filepath}")
    
    def on_epoch_end(self, state: CallbackState) -> None:
        if state.model is None:
            return
        
        if self.save_freq != 'epoch':
            return
        
        current = state.metrics.get(self.monitor, state.val_loss)
        
        if self.save_best_only:
            if current is not None and self._is_improvement(current):
                self.best = current
                self._save_checkpoint(state)
        else:
            self._save_checkpoint(state)
    
    def on_batch_end(self, state: CallbackState) -> None:
        if state.model is None:
            return
        
        if self.save_freq == 'epoch':
            return
        
        if state.step % self.save_freq == 0:
            self._save_checkpoint(state)


class ProgressLogger(Callback):
    """
    Progress logging callback.
    
    Logs training progress with metrics.
    """
    
    def __init__(
        self,
        log_freq: int = 100,
        metrics: Optional[List[str]] = None
    ):
        """
        Args:
            log_freq: Batches between logs
            metrics: Metrics to log
        """
        self.log_freq = log_freq
        self.metrics = metrics
        
        self.epoch_start_time = 0
        self.batch_count = 0
        self.running_loss = 0.0
    
    def on_epoch_begin(self, state: CallbackState) -> None:
        self.epoch_start_time = time.time()
        self.batch_count = 0
        self.running_loss = 0.0
    
    def on_batch_end(self, state: CallbackState) -> None:
        self.batch_count += 1
        self.running_loss += state.train_loss
        
        if self.batch_count % self.log_freq == 0:
            avg_loss = self.running_loss / self.batch_count
            elapsed = time.time() - self.epoch_start_time
            
            msg = f"Epoch {state.epoch} | Step {state.step} | "
            msg += f"Loss: {avg_loss:.4f} | "
            msg += f"Time: {elapsed:.1f}s"
            
            if self.metrics:
                for name in self.metrics:
                    if name in state.metrics:
                        msg += f" | {name}: {state.metrics[name]:.4f}"
            
            print(msg)
    
    def on_epoch_end(self, state: CallbackState) -> None:
        elapsed = time.time() - self.epoch_start_time
        avg_loss = self.running_loss / max(1, self.batch_count)
        
        msg = f"Epoch {state.epoch}/{state.total_epochs} completed | "
        msg += f"Train Loss: {avg_loss:.4f}"
        
        if state.val_loss is not None:
            msg += f" | Val Loss: {state.val_loss:.4f}"
        
        msg += f" | Time: {elapsed:.1f}s"
        
        print(msg)


class TensorBoardLogger(Callback):
    """
    TensorBoard logging callback.
    
    Logs metrics and images to TensorBoard.
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        log_freq: int = 100,
        log_images: bool = True,
        image_freq: int = 500
    ):
        """
        Args:
            log_dir: TensorBoard log directory
            log_freq: Steps between scalar logs
            log_images: Whether to log images
            image_freq: Steps between image logs
        """
        from torch.utils.tensorboard import SummaryWriter
        
        self.log_dir = Path(log_dir)
        self.log_freq = log_freq
        self.log_images = log_images
        self.image_freq = image_freq
        
        self.writer = SummaryWriter(log_dir)
    
    def on_batch_end(self, state: CallbackState) -> None:
        if state.step % self.log_freq == 0:
            self.writer.add_scalar('train/loss', state.train_loss, state.step)
            
            for name, value in state.metrics.items():
                self.writer.add_scalar(f'train/{name}', value, state.step)
        
        if self.log_images and state.step % self.image_freq == 0:
            if 'images' in state.extra:
                for name, images in state.extra['images'].items():
                    self.writer.add_images(name, images, state.step)
    
    def on_epoch_end(self, state: CallbackState) -> None:
        self.writer.add_scalar('epoch/train_loss', state.train_loss, state.epoch)
        
        if state.val_loss is not None:
            self.writer.add_scalar('epoch/val_loss', state.val_loss, state.epoch)
        
        for name, value in state.metrics.items():
            self.writer.add_scalar(f'epoch/{name}', value, state.epoch)
    
    def on_train_end(self, state: CallbackState) -> None:
        self.writer.close()


class LearningRateMonitor(Callback):
    """
    Learning rate monitoring callback.
    
    Logs learning rates during training.
    """
    
    def __init__(self, log_freq: int = 100):
        self.log_freq = log_freq
        self.lr_history: List[Dict[str, float]] = []
    
    def on_batch_end(self, state: CallbackState) -> None:
        if state.step % self.log_freq == 0:
            if state.optimizer is not None:
                lrs = {}
                for i, group in enumerate(state.optimizer.param_groups):
                    lrs[f'lr_group_{i}'] = group['lr']
                
                self.lr_history.append({
                    'step': state.step,
                    **lrs
                })


class GradientMonitor(Callback):
    """
    Gradient monitoring callback.
    
    Monitors gradient norms and statistics.
    """
    
    def __init__(
        self,
        model: nn.Module,
        log_freq: int = 100,
        log_histogram: bool = False
    ):
        """
        Args:
            model: Model to monitor
            log_freq: Steps between logs
            log_histogram: Whether to log histograms
        """
        self.model = model
        self.log_freq = log_freq
        self.log_histogram = log_histogram
        
        self.grad_norms: List[Dict[str, float]] = []
    
    def on_batch_end(self, state: CallbackState) -> None:
        if state.step % self.log_freq == 0:
            total_norm = 0.0
            param_norms = {}
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    param_norms[name] = param_norm
            
            total_norm = total_norm ** 0.5
            
            self.grad_norms.append({
                'step': state.step,
                'total_norm': total_norm,
                'param_norms': param_norms
            })


class MetricsHistory(Callback):
    """
    Metrics history callback.
    
    Records all metrics during training.
    """
    
    def __init__(self, save_path: Optional[Union[str, Path]] = None):
        """
        Args:
            save_path: Path to save history
        """
        self.save_path = Path(save_path) if save_path else None
        self.history: Dict[str, List[float]] = {}
    
    def on_epoch_end(self, state: CallbackState) -> None:
        if 'train_loss' not in self.history:
            self.history['train_loss'] = []
        self.history['train_loss'].append(state.train_loss)
        
        if state.val_loss is not None:
            if 'val_loss' not in self.history:
                self.history['val_loss'] = []
            self.history['val_loss'].append(state.val_loss)
        
        for name, value in state.metrics.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(value)
    
    def on_train_end(self, state: CallbackState) -> None:
        if self.save_path:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.save_path, 'w') as f:
                json.dump(self.history, f, indent=2)


class GANMonitor(Callback):
    """
    GAN-specific monitoring callback.
    
    Monitors generator and discriminator losses,
    as well as training stability indicators.
    """
    
    def __init__(
        self,
        log_freq: int = 100,
        track_d_g_ratio: bool = True,
        track_mode_collapse: bool = True
    ):
        """
        Args:
            log_freq: Steps between logs
            track_d_g_ratio: Track D/G loss ratio
            track_mode_collapse: Track mode collapse indicators
        """
        self.log_freq = log_freq
        self.track_d_g_ratio = track_d_g_ratio
        self.track_mode_collapse = track_mode_collapse
        
        self.g_losses: List[float] = []
        self.d_losses: List[float] = []
        self.d_g_ratios: List[float] = []
    
    def on_batch_end(self, state: CallbackState) -> None:
        if state.step % self.log_freq == 0:
            g_loss = state.metrics.get('g_loss', 0)
            d_loss = state.metrics.get('d_loss', 0)
            
            self.g_losses.append(g_loss)
            self.d_losses.append(d_loss)
            
            if self.track_d_g_ratio and g_loss > 0:
                ratio = d_loss / g_loss
                self.d_g_ratios.append(ratio)
    
    def get_stability_report(self) -> Dict[str, float]:
        """Get training stability report."""
        import numpy as np
        
        report = {}
        
        if self.g_losses:
            report['g_loss_mean'] = float(np.mean(self.g_losses))
            report['g_loss_std'] = float(np.std(self.g_losses))
        
        if self.d_losses:
            report['d_loss_mean'] = float(np.mean(self.d_losses))
            report['d_loss_std'] = float(np.std(self.d_losses))
        
        if self.d_g_ratios:
            report['d_g_ratio_mean'] = float(np.mean(self.d_g_ratios))
            report['d_g_ratio_std'] = float(np.std(self.d_g_ratios))
        
        return report


class TimingCallback(Callback):
    """
    Training timing callback.
    
    Tracks time for various training phases.
    """
    
    def __init__(self):
        self.train_start = 0
        self.epoch_times: List[float] = []
        self.batch_times: List[float] = []
        self._batch_start = 0
        self._epoch_start = 0
    
    def on_train_begin(self, state: CallbackState) -> None:
        self.train_start = time.time()
    
    def on_epoch_begin(self, state: CallbackState) -> None:
        self._epoch_start = time.time()
    
    def on_epoch_end(self, state: CallbackState) -> None:
        self.epoch_times.append(time.time() - self._epoch_start)
    
    def on_batch_begin(self, state: CallbackState) -> None:
        self._batch_start = time.time()
    
    def on_batch_end(self, state: CallbackState) -> None:
        self.batch_times.append(time.time() - self._batch_start)
    
    def get_timing_report(self) -> Dict[str, float]:
        """Get timing statistics."""
        import numpy as np
        
        return {
            'total_time': time.time() - self.train_start,
            'avg_epoch_time': float(np.mean(self.epoch_times)) if self.epoch_times else 0,
            'avg_batch_time': float(np.mean(self.batch_times)) if self.batch_times else 0,
            'epochs_completed': len(self.epoch_times)
        }
