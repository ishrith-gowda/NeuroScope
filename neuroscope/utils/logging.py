"""
logging utilities.

structured logging and experiment tracking for
reproducible research.
"""

from typing import Optional, Dict, List, Any, Union
from pathlib import Path
from datetime import datetime
import logging
import json
import sys


# global logger registry
_loggers: Dict[str, logging.Logger] = {}


def setup_logger(
    name: str = 'neuroscope',
    level: int = logging.INFO,
    log_file: Union[str, Path] = None,
    console: bool = True,
    format_string: str = None
) -> logging.Logger:
    """
    set up a logger with file and console handlers.
    
    args:
        name: logger name
        level: logging level
        log_file: optional log file path
        console: enable console output
        format_string: custom format string
        
    returns:
        configured logger
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # clear existing handlers
    
    if format_string is None:
        format_string = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    _loggers[name] = logger
    return logger


def get_logger(name: str = 'neuroscope') -> logging.Logger:
    """
    get existing logger or create new one.
    
    args:
        name: logger name
        
    returns:
        logger instance
    """
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]


class MetricTracker:
    """
    track and aggregate metrics during training.
    
    supports running averages, best values, and history.
    """
    
    def __init__(self):
        self.history: Dict[str, List[float]] = {}
        self.running: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self.best: Dict[str, float] = {}
        self.best_epoch: Dict[str, int] = {}
    
    def update(
        self,
        metrics: Dict[str, float],
        n: int = 1
    ):
        """
        update with new metric values.
        
        args:
            metrics: dict of metric name -> value
            n: batch size / weight
        """
        for name, value in metrics.items():
            if name not in self.running:
                self.running[name] = 0.0
                self.counts[name] = 0
            
            self.running[name] += value * n
            self.counts[name] += n
    
    def average(self) -> Dict[str, float]:
        """get running averages."""
        return {
            name: self.running[name] / max(self.counts[name], 1)
            for name in self.running
        }
    
    def reset(self):
        """reset running averages."""
        self.running = {}
        self.counts = {}
    
    def end_epoch(self, epoch: int):
        """
        end epoch and record history.
        
        args:
            epoch: current epoch number
        """
        averages = self.average()
        
        for name, value in averages.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(value)
            
            # track best
            if name not in self.best or value > self.best[name]:
                self.best[name] = value
                self.best_epoch[name] = epoch
        
        self.reset()
    
    def get_history(self, name: str) -> List[float]:
        """get metric history."""
        return self.history.get(name, [])
    
    def get_best(self, name: str) -> Optional[float]:
        """get best value for metric."""
        return self.best.get(name)
    
    def is_best(self, name: str, value: float, mode: str = 'max') -> bool:
        """
        check if value is best.
        
        args:
            name: metric name
            value: current value
            mode: 'max' or 'min'
            
        returns:
            true if value is best
        """
        if name not in self.best:
            return True
        
        if mode == 'max':
            return value > self.best[name]
        return value < self.best[name]
    
    def to_dict(self) -> Dict:
        """convert to dictionary."""
        return {
            'history': self.history,
            'best': self.best,
            'best_epoch': self.best_epoch
        }
    
    def save(self, path: Union[str, Path]):
        """save to json file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'MetricTracker':
        """load from json file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        tracker = cls()
        tracker.history = data.get('history', {})
        tracker.best = data.get('best', {})
        tracker.best_epoch = data.get('best_epoch', {})
        
        return tracker


class ExperimentLogger:
    """
    comprehensive experiment logging.
    
    tracks configuration, metrics, artifacts, and provides
    tensorboard/w&b integration.
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: Union[str, Path],
        config: Dict = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False
    ):
        """
        args:
            experiment_name: name of experiment
            output_dir: output directory
            config: experiment configuration
            use_tensorboard: enable tensorboard
            use_wandb: enable weights & biases
        """
        self.name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {}
        self.start_time = datetime.now()
        
        # set up text logger
        self.logger = setup_logger(
            name=experiment_name,
            log_file=self.output_dir / 'experiment.log'
        )
        
        # metric tracking
        self.metrics = MetricTracker()
        
        # tensorboard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(self.output_dir / 'tensorboard')
            except ImportError:
                self.logger.warning("TensorBoard not available")
        
        # w&b
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project='neuroscope',
                    name=experiment_name,
                    config=config
                )
            except ImportError:
                self.logger.warning("Weights & Biases not available")
        
        # save config
        self._save_config()
        
        self.logger.info(f"Experiment '{experiment_name}' initialized")
    
    def _save_config(self):
        """save configuration to file."""
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump({
                'name': self.name,
                'start_time': self.start_time.isoformat(),
                'config': self.config
            }, f, indent=2)
    
    def log(self, message: str, level: str = 'info'):
        """log a message."""
        getattr(self.logger, level)(message)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int = None,
        prefix: str = ''
    ):
        """
        log metrics.
        
        args:
            metrics: dict of metric name -> value
            step: global step
            prefix: metric name prefix
        """
        self.metrics.update(metrics)
        
        # tensorboard
        if self.tb_writer is not None and step is not None:
            for name, value in metrics.items():
                tag = f"{prefix}/{name}" if prefix else name
                self.tb_writer.add_scalar(tag, value, step)
        
        # w&b
        if self.wandb_run is not None:
            import wandb
            log_dict = {
                f"{prefix}/{k}" if prefix else k: v 
                for k, v in metrics.items()
            }
            if step is not None:
                log_dict['step'] = step
            wandb.log(log_dict)
    
    def log_image(
        self,
        tag: str,
        image,
        step: int = None
    ):
        """
        log image.
        
        args:
            tag: image tag
            image: image (numpy or tensor)
            step: global step
        """
        import numpy as np
        
        if self.tb_writer is not None and step is not None:
            if hasattr(image, 'numpy'):
                image = image.numpy()
            
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                # chw -> hwc
                image = np.transpose(image, (1, 2, 0))
            
            self.tb_writer.add_image(tag, image, step, dataformats='HWC')
    
    def log_model_graph(self, model, input_tensor):
        """log model graph to tensorboard."""
        if self.tb_writer is not None:
            try:
                self.tb_writer.add_graph(model, input_tensor)
            except Exception as e:
                self.logger.warning(f"Failed to log model graph: {e}")
    
    def end_epoch(self, epoch: int):
        """end epoch and record metrics."""
        self.metrics.end_epoch(epoch)
        
        # log epoch summary
        averages = {k: v[-1] for k, v in self.metrics.history.items()}
        self.logger.info(f"Epoch {epoch} - {averages}")
    
    def save_checkpoint(
        self,
        state: Dict,
        filename: str = 'checkpoint.pth'
    ):
        """save checkpoint."""
        import torch
        
        path = self.output_dir / 'checkpoints' / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        self.logger.info(f"Saved checkpoint: {path}")
    
    def save_artifact(
        self,
        data: Any,
        filename: str,
        artifact_type: str = 'general'
    ):
        """save artifact."""
        path = self.output_dir / 'artifacts' / artifact_type / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if filename.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        elif filename.endswith('.npy'):
            import numpy as np
            np.save(path, data)
        else:
            with open(path, 'wb') as f:
                import pickle
                pickle.dump(data, f)
        
        self.logger.info(f"Saved artifact: {path}")
    
    def finish(self):
        """finish experiment and clean up."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        # save final metrics
        self.metrics.save(self.output_dir / 'metrics.json')
        
        # save summary
        summary = {
            'name': self.name,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'final_metrics': self.metrics.best,
            'best_epochs': self.metrics.best_epoch
        }
        
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Experiment completed in {duration}")
        
        # close writers
        if self.tb_writer is not None:
            self.tb_writer.close()
        
        if self.wandb_run is not None:
            import wandb
            wandb.finish()
