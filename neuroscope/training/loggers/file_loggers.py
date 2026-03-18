"""
csv and json loggers.

file-based logging for metrics export and reproducibility.

author: neuroscope research team
"""

from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from datetime import datetime
import csv
import json
import threading


class CSVLogger:
    """
    csv logger for tabular metrics export.
    
    creates clean, parseable csv files for metrics analysis
    with tools like pandas, excel, or r.
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        filename: str = "training_metrics.csv",
        append: bool = False,
        flush_every: int = 1
    ):
        """
        initialize csv logger.
        
        args:
            log_dir: directory to save csv files
            filename: name of the csv file
            append: whether to append to existing file
            flush_every: flush to disk every n writes
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / filename
        
        self.append = append
        self.flush_every = flush_every
        self._write_count = 0
        
        self._columns: Optional[List[str]] = None
        self._file = None
        self._writer = None
        self._lock = threading.Lock()
        
        if not append and self.filepath.exists():
            self.filepath.unlink()
            
    def _initialize_file(self, columns: List[str]):
        """initialize the csv file with headers."""
        mode = 'a' if self.append and self.filepath.exists() else 'w'
        self._file = open(self.filepath, mode, newline='', buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=columns, extrasaction='ignore')

        if mode == 'w':
            self._writer.writeheader()

        self._columns = columns
        
    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None
    ):
        """
        log metrics to csv.

        args:
            metrics: dictionary of metric_name -> value
            step: global step (optional)
            epoch: current epoch (optional)
        """
        with self._lock:
            # build row with optional step/epoch
            row = {}
            if step is not None:
                row['step'] = step
            if epoch is not None:
                row['epoch'] = epoch
            row['timestamp'] = datetime.now().isoformat()
            row.update(metrics)

            # initialize file if needed
            if self._columns is None:
                columns = list(row.keys())
                self._initialize_file(columns)

            # check if new columns appeared - if so, only write columns we have
            filtered_row = {k: v for k, v in row.items() if k in self._columns}

            # write row
            self._writer.writerow(filtered_row)
            self._write_count += 1

            # flush periodically
            if self._write_count % self.flush_every == 0:
                self._file.flush()
                
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None
    ):
        """
        log a complete epoch's metrics.
        
        args:
            epoch: current epoch
            train_metrics: training metrics
            val_metrics: validation metrics (optional)
            lr: learning rate (optional)
        """
        row = {'epoch': epoch}
        
        # add training metrics with prefix
        for k, v in train_metrics.items():
            row[f'train_{k}'] = v
            
        # add validation metrics with prefix
        if val_metrics:
            for k, v in val_metrics.items():
                row[f'val_{k}'] = v
                
        # add learning rate
        if lr is not None:
            row['learning_rate'] = lr
            
        self.log(row, epoch=epoch)
        
    def flush(self):
        """flush buffer to disk."""
        with self._lock:
            if self._file:
                self._file.flush()
                
    def close(self):
        """close the csv file."""
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None
                self._writer = None
                
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class JSONLogger:
    """
    json logger for structured metrics and metadata.
    
    creates json files with full training history,
    configuration, and metadata for reproducibility.
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        filename: str = "training_log.json",
        pretty_print: bool = True
    ):
        """
        initialize json logger.
        
        args:
            log_dir: directory to save json files
            filename: name of the json file
            pretty_print: whether to format json for readability
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / filename
        self.pretty_print = pretty_print
        
        self._lock = threading.Lock()
        
        # initialize log structure
        self._log = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            },
            'config': {},
            'history': {
                'train': [],
                'validation': [],
                'epochs': []
            },
            'checkpoints': [],
            'events': []
        }
        
        # load existing log if present
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    self._log = json.load(f)
            except json.JSONDecodeError:
                pass  # start fresh if file is corrupted
                
    def log_config(self, config: Dict[str, Any]):
        """
        log training configuration.
        
        args:
            config: configuration dictionary
        """
        with self._lock:
            self._log['config'] = self._serialize(config)
            self._save()
            
    def log_metadata(self, metadata: Dict[str, Any]):
        """
        log additional metadata.
        
        args:
            metadata: metadata dictionary
        """
        with self._lock:
            self._log['metadata'].update(self._serialize(metadata))
            self._save()
            
    def log_train_step(
        self,
        step: int,
        metrics: Dict[str, float],
        epoch: Optional[int] = None
    ):
        """
        log a training step.
        
        args:
            step: global step
            metrics: metrics dictionary
            epoch: current epoch (optional)
        """
        with self._lock:
            entry = {
                'step': step,
                'timestamp': datetime.now().isoformat(),
                **metrics
            }
            if epoch is not None:
                entry['epoch'] = epoch
                
            self._log['history']['train'].append(entry)
            
            # don't save every step (too slow)
            # save is done on epoch end or explicitly
            
    def log_validation(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """
        log validation results.
        
        args:
            epoch: current epoch
            metrics: validation metrics
        """
        with self._lock:
            entry = {
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                **metrics
            }
            self._log['history']['validation'].append(entry)
            self._save()
            
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
        time_elapsed: Optional[float] = None
    ):
        """
        log complete epoch summary.
        
        args:
            epoch: current epoch
            train_metrics: training metrics
            val_metrics: validation metrics (optional)
            lr: learning rate (optional)
            time_elapsed: time for this epoch (optional)
        """
        with self._lock:
            entry = {
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'train': train_metrics
            }
            
            if val_metrics:
                entry['validation'] = val_metrics
            if lr is not None:
                entry['learning_rate'] = lr
            if time_elapsed is not None:
                entry['time_elapsed_seconds'] = time_elapsed
                
            self._log['history']['epochs'].append(entry)
            self._log['metadata']['last_updated'] = datetime.now().isoformat()
            self._save()
            
    def log_checkpoint(
        self,
        epoch: int,
        path: str,
        is_best: bool = False,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        log checkpoint save event.
        
        args:
            epoch: current epoch
            path: checkpoint file path
            is_best: whether this is the best model
            metrics: associated metrics (optional)
        """
        with self._lock:
            entry = {
                'epoch': epoch,
                'path': str(path),
                'is_best': is_best,
                'timestamp': datetime.now().isoformat()
            }
            if metrics:
                entry['metrics'] = metrics
                
            self._log['checkpoints'].append(entry)
            self._save()
            
    def log_event(
        self,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        log a training event.
        
        args:
            event_type: type of event (e.g., 'early_stop', 'lr_update')
            message: event message
            data: additional event data (optional)
        """
        with self._lock:
            entry = {
                'type': event_type,
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
            if data:
                entry['data'] = self._serialize(data)
                
            self._log['events'].append(entry)
            self._save()
            
    def get_history(self, key: str = 'epochs') -> List[Dict]:
        """get training history."""
        return self._log['history'].get(key, [])
        
    def get_best_checkpoint(self) -> Optional[Dict]:
        """get the best checkpoint entry."""
        best_checkpoints = [c for c in self._log['checkpoints'] if c.get('is_best')]
        return best_checkpoints[-1] if best_checkpoints else None
        
    def _serialize(self, obj: Any) -> Any:
        """serialize object for json storage."""
        import numpy as np

        # handle numpy types
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # handle regular types
        elif isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return self._serialize(obj.__dict__)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
            
    def _save(self):
        """save log to disk."""
        with open(self.filepath, 'w') as f:
            # serialize to handle numpy types and other non-json-serializable objects
            serialized_log = self._serialize(self._log)
            if self.pretty_print:
                json.dump(serialized_log, f, indent=2)
            else:
                json.dump(serialized_log, f)
                
    def flush(self):
        """force save to disk."""
        with self._lock:
            self._save()
            
    def close(self):
        """close and save final state."""
        self.flush()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class MetricsAggregator:
    """
    aggregates metrics over batches for epoch-level reporting.
    
    computes running averages and stores per-batch values.
    """
    
    def __init__(self):
        self._values: Dict[str, List[float]] = {}
        self._counts: Dict[str, int] = {}
        
    def update(self, metrics: Dict[str, float], batch_size: int = 1):
        """
        update with new batch metrics.
        
        args:
            metrics: batch metrics
            batch_size: size of batch (for weighted averaging)
        """
        for key, value in metrics.items():
            if key not in self._values:
                self._values[key] = []
                self._counts[key] = 0
            self._values[key].append(value)
            self._counts[key] += batch_size
            
    def get_average(self, key: str) -> float:
        """get average value for a metric."""
        if key not in self._values or not self._values[key]:
            return 0.0
        return sum(self._values[key]) / len(self._values[key])
        
    def get_averages(self) -> Dict[str, float]:
        """get all average values."""
        return {key: self.get_average(key) for key in self._values}
        
    def get_values(self, key: str) -> List[float]:
        """get all values for a metric."""
        return self._values.get(key, [])
        
    def reset(self):
        """reset all accumulated values."""
        self._values.clear()
        self._counts.clear()
