"""
logger manager.

coordinates multiple loggers (tensorboard, csv, json, console)
for unified logging experience.

author: neuroscope research team
"""

from typing import Optional, Dict, Any, Union, List
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn

from .tensorboard_logger import TensorBoardLogger
from .file_loggers import CSVLogger, JSONLogger, MetricsAggregator
from .console_logger import ConsoleLogger


class LoggerManager:
    """
    unified logger manager that coordinates all logging backends.
    
    provides a single interface for logging to:
    - tensorboard (scalars, images, histograms)
    - csv (tabular metrics)
    - json (structured logs)
    - console (formatted output)
    
    usage:
        logger = loggermanager(
            log_dir="runs/experiment_1",
            experiment_name="sa_cyclegan_25d",
            use_tensorboard=true,
            use_csv=true,
            use_json=true,
            console_verbose=2
        )
        
        logger.on_training_start(total_epochs=100)
        
        for epoch in range(100):
            logger.on_epoch_start(epoch)
            
            for batch, data in enumerate(dataloader):
                # ... training code ...
                logger.log_batch(batch, total_batches, losses)
                
            logger.on_epoch_end(epoch, train_metrics, val_metrics)
            
        logger.on_training_end(final_metrics)
        logger.close()
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: str = "training",
        use_tensorboard: bool = True,
        use_csv: bool = True,
        use_json: bool = True,
        use_console: bool = True,
        console_verbose: int = 2,
        tensorboard_flush_secs: int = 30
    ):
        """
        initialize logger manager.
        
        args:
            log_dir: root directory for all logs
            experiment_name: name of the experiment
            use_tensorboard: enable tensorboard logging
            use_csv: enable csv logging
            use_json: enable json logging
            use_console: enable console logging
            console_verbose: console verbosity level (0-3)
            tensorboard_flush_secs: tensorboard flush frequency
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # create subdirectories
        self.tensorboard_dir = self.log_dir / "tensorboard"
        self.metrics_dir = self.log_dir / "metrics"
        self.samples_dir = self.log_dir / "samples"
        self.figures_dir = self.log_dir / "figures"
        self.checkpoints_dir = self.log_dir / "checkpoints"
        
        for d in [self.tensorboard_dir, self.metrics_dir, 
                  self.samples_dir, self.figures_dir, self.checkpoints_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # initialize loggers
        self.tb_logger: Optional[TensorBoardLogger] = None
        self.csv_logger: Optional[CSVLogger] = None
        self.json_logger: Optional[JSONLogger] = None
        self.console_logger: Optional[ConsoleLogger] = None
        
        if use_tensorboard:
            self.tb_logger = TensorBoardLogger(
                log_dir=self.tensorboard_dir,
                experiment_name=experiment_name,
                flush_secs=tensorboard_flush_secs
            )
            
        if use_csv:
            self.csv_logger = CSVLogger(
                log_dir=self.metrics_dir,
                filename="training_metrics.csv"
            )
            
        if use_json:
            self.json_logger = JSONLogger(
                log_dir=self.metrics_dir,
                filename="training_log.json"
            )
            
        if use_console:
            self.console_logger = ConsoleLogger(
                experiment_name=experiment_name,
                verbose=console_verbose,
                log_to_file=True,
                log_file=str(self.log_dir / "console.log")
            )
            
        # metrics aggregator for batch-level tracking
        self.train_aggregator = MetricsAggregator()
        
        # state tracking
        self.global_step = 0
        self.current_epoch = 0
        self.total_epochs = 0
        self.config: Dict[str, Any] = {}
        
    # =========================================================================
    # configuration & metadata
    # =========================================================================
    
    def log_config(self, config: Dict[str, Any]):
        """log training configuration."""
        self.config = config
        
        if self.tb_logger:
            self.tb_logger.log_config(config)
            self.tb_logger.log_hyperparameters(self._flatten_dict(config))
            
        if self.json_logger:
            self.json_logger.log_config(config)
            
        if self.console_logger:
            self.console_logger.print_config(config)
            
    def log_model_info(
        self,
        model: nn.Module,
        model_name: str = "SACycleGAN25D",
        input_shape: tuple = (12, 128, 128)
    ):
        """log model information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        if self.console_logger:
            self.console_logger.print_model_summary(
                model_name, total_params, trainable_params
            )
            
        if self.tb_logger:
            try:
                self.tb_logger.log_graph(model, input_shape)
            except Exception:
                pass  # graph logging can fail for complex models
                
        if self.json_logger:
            self.json_logger.log_metadata({
                'model_name': model_name,
                'total_params': total_params,
                'trainable_params': trainable_params
            })
            
    def log_data_info(
        self,
        train_samples: int,
        val_samples: int,
        test_samples: int,
        batch_size: int
    ):
        """log dataset information."""
        if self.console_logger:
            self.console_logger.print_data_summary(
                train_samples, val_samples, test_samples, batch_size
            )
            
        if self.json_logger:
            self.json_logger.log_metadata({
                'train_samples': train_samples,
                'val_samples': val_samples,
                'test_samples': test_samples,
                'batch_size': batch_size
            })
            
    # =========================================================================
    # training lifecycle
    # =========================================================================
    
    def on_training_start(self, total_epochs: int):
        """called at the start of training."""
        self.total_epochs = total_epochs
        
        if self.console_logger:
            self.console_logger.print_banner()
            self.console_logger.on_training_start(total_epochs)
            
        if self.json_logger:
            self.json_logger.log_event(
                'training_start',
                f'Starting training for {total_epochs} epochs'
            )
            
    def on_training_end(self, final_metrics: Optional[Dict[str, float]] = None):
        """called at the end of training."""
        if self.console_logger:
            self.console_logger.on_training_end(final_metrics)
            
        if self.json_logger:
            self.json_logger.log_event(
                'training_end',
                'Training completed',
                {'final_metrics': final_metrics}
            )
            
        self.flush()
        
    def on_epoch_start(self, epoch: int):
        """called at the start of each epoch."""
        self.current_epoch = epoch
        self.train_aggregator.reset()
        
        if self.console_logger:
            self.console_logger.on_epoch_start(epoch, self.total_epochs)
            
    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
        time_elapsed: Optional[float] = None
    ):
        """called at the end of each epoch."""
        # log to tensorboard
        if self.tb_logger:
            self.tb_logger.log_training_losses(train_metrics, epoch)
            if val_metrics:
                self.tb_logger.log_validation_metrics(val_metrics, epoch)
            if lr:
                self.tb_logger.log_learning_rates(lr, lr, epoch)
                
        # log to csv
        if self.csv_logger:
            self.csv_logger.log_epoch(epoch, train_metrics, val_metrics, lr)
            
        # log to json
        if self.json_logger:
            self.json_logger.log_epoch(epoch, train_metrics, val_metrics, lr, time_elapsed)
            
        # log to console
        if self.console_logger:
            self.console_logger.on_epoch_end(epoch, train_metrics, val_metrics, lr)
            
    # =========================================================================
    # batch-level logging
    # =========================================================================
    
    def log_batch(
        self,
        batch_idx: int,
        total_batches: int,
        losses: Dict[str, float],
        batch_size: int = 1,
        samples_per_sec: Optional[float] = None
    ):
        """log batch-level metrics."""
        self.global_step += 1
        self.train_aggregator.update(losses, batch_size)
        
        # tensorboard: log every n batches
        if self.tb_logger and batch_idx % 10 == 0:
            for name, value in losses.items():
                self.tb_logger.log_scalar(f"Batch/{name}", value, self.global_step)
                
        # console: handled by on_batch_end
        if self.console_logger:
            self.console_logger.on_batch_end(
                batch_idx, total_batches, losses, samples_per_sec
            )
            
    # =========================================================================
    # metrics & scalars
    # =========================================================================
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None
    ):
        """log a scalar value."""
        step = step if step is not None else self.global_step
        
        if self.tb_logger:
            self.tb_logger.log_scalar(tag, value, step)
            
    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: Optional[int] = None
    ):
        """log multiple scalars."""
        step = step if step is not None else self.global_step
        
        if self.tb_logger:
            self.tb_logger.log_scalars(main_tag, tag_scalar_dict, step)
            
    def log_learning_rates(self, lr_G: float, lr_D: float):
        """log learning rates."""
        if self.tb_logger:
            self.tb_logger.log_learning_rates(lr_G, lr_D, self.current_epoch)
            
    # =========================================================================
    # images & samples
    # =========================================================================
    
    def log_images(
        self,
        tag: str,
        images: torch.Tensor,
        step: Optional[int] = None
    ):
        """log images to tensorboard."""
        if self.tb_logger:
            self.tb_logger.log_images(tag, images, step)
            
    def log_sample_comparison(
        self,
        real_A: torch.Tensor,
        fake_B: torch.Tensor,
        rec_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_A: torch.Tensor,
        rec_B: torch.Tensor,
        step: Optional[int] = None,
        modality_idx: int = 0
    ):
        """log sample comparison images."""
        if self.tb_logger:
            self.tb_logger.log_sample_comparison(
                real_A, fake_B, rec_A,
                real_B, fake_A, rec_B,
                step, modality_idx
            )
            
    def log_sample_saved(self, path: str, epoch: int):
        """log that samples were saved."""
        if self.console_logger:
            self.console_logger.log_sample_saved(path, epoch)
            
    # =========================================================================
    # gradients & weights
    # =========================================================================
    
    def log_gradients(
        self,
        models: Dict[str, nn.Module],
        step: Optional[int] = None
    ):
        """log gradient norms for multiple models."""
        if self.tb_logger:
            self.tb_logger.log_gradient_norms(models, step)
            
    def log_model_histograms(
        self,
        model: nn.Module,
        prefix: str = "",
        step: Optional[int] = None
    ):
        """log weight and gradient histograms."""
        if self.tb_logger:
            self.tb_logger.log_model_weights(model, step, prefix)
            self.tb_logger.log_model_gradients(model, step, prefix)
            
    # =========================================================================
    # checkpoints & events
    # =========================================================================
    
    def log_checkpoint(
        self,
        epoch: int,
        path: str,
        is_best: bool = False,
        metrics: Optional[Dict[str, float]] = None
    ):
        """log checkpoint save."""
        if self.console_logger:
            self.console_logger.log_checkpoint(path, is_best, metrics)
            
        if self.json_logger:
            self.json_logger.log_checkpoint(epoch, path, is_best, metrics)
            
    def log_early_stop(self, epoch: int, patience: int, best_metric: float):
        """log early stopping."""
        if self.console_logger:
            self.console_logger.log_early_stop(epoch, patience, best_metric)
            
        if self.json_logger:
            self.json_logger.log_event(
                'early_stop',
                f'Early stopping at epoch {epoch}',
                {'patience': patience, 'best_metric': best_metric}
            )
            
    def log_lr_update(self, old_lr: float, new_lr: float, reason: str = ""):
        """log learning rate update."""
        if self.console_logger:
            self.console_logger.log_lr_update(old_lr, new_lr, reason)
            
        if self.json_logger:
            self.json_logger.log_event(
                'lr_update',
                f'LR updated: {old_lr:.2e} -> {new_lr:.2e}',
                {'old_lr': old_lr, 'new_lr': new_lr, 'reason': reason}
            )
            
    # =========================================================================
    # messages
    # =========================================================================
    
    def log_info(self, message: str):
        """log info message."""
        if self.console_logger:
            self.console_logger.log_info(message)
            
    def log_warning(self, message: str):
        """log warning message."""
        if self.console_logger:
            self.console_logger.log_warning(message)
            
    def log_error(self, message: str):
        """log error message."""
        if self.console_logger:
            self.console_logger.log_error(message)
            
    def log_success(self, message: str):
        """log success message."""
        if self.console_logger:
            self.console_logger.log_success(message)
            
    # =========================================================================
    # utility methods
    # =========================================================================
    
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = '',
        sep: str = '/'
    ) -> Dict[str, Any]:
        """flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
        
    def get_train_averages(self) -> Dict[str, float]:
        """get averaged training metrics for current epoch."""
        return self.train_aggregator.get_averages()
        
    def flush(self):
        """flush all loggers."""
        if self.tb_logger:
            self.tb_logger.flush()
        if self.csv_logger:
            self.csv_logger.flush()
        if self.json_logger:
            self.json_logger.flush()
            
    def close(self):
        """close all loggers."""
        if self.tb_logger:
            self.tb_logger.close()
        if self.csv_logger:
            self.csv_logger.close()
        if self.json_logger:
            self.json_logger.close()
        if self.console_logger:
            self.console_logger.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
