"""
CSV and JSON Loggers.

File-based logging for metrics export and reproducibility.

Author: NeuroScope Research Team
"""

from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from datetime import datetime
import csv
import json
import threading


class CSVLogger:
    """
    CSV Logger for tabular metrics export.
    
    Creates clean, parseable CSV files for metrics analysis
    with tools like pandas, Excel, or R.
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        filename: str = "training_metrics.csv",
        append: bool = False,
        flush_every: int = 1
    ):
        """
        Initialize CSV logger.
        
        Args:
            log_dir: Directory to save CSV files
            filename: Name of the CSV file
            append: Whether to append to existing file
            flush_every: Flush to disk every N writes
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
        """Initialize the CSV file with headers."""
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
        Log metrics to CSV.

        Args:
            metrics: Dictionary of metric_name -> value
            step: Global step (optional)
            epoch: Current epoch (optional)
        """
        with self._lock:
            # Build row with optional step/epoch
            row = {}
            if step is not None:
                row['step'] = step
            if epoch is not None:
                row['epoch'] = epoch
            row['timestamp'] = datetime.now().isoformat()
            row.update(metrics)

            # Initialize file if needed
            if self._columns is None:
                columns = list(row.keys())
                self._initialize_file(columns)

            # Check if new columns appeared - if so, only write columns we have
            filtered_row = {k: v for k, v in row.items() if k in self._columns}

            # Write row
            self._writer.writerow(filtered_row)
            self._write_count += 1

            # Flush periodically
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
        Log a complete epoch's metrics.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics (optional)
            lr: Learning rate (optional)
        """
        row = {'epoch': epoch}
        
        # Add training metrics with prefix
        for k, v in train_metrics.items():
            row[f'train_{k}'] = v
            
        # Add validation metrics with prefix
        if val_metrics:
            for k, v in val_metrics.items():
                row[f'val_{k}'] = v
                
        # Add learning rate
        if lr is not None:
            row['learning_rate'] = lr
            
        self.log(row, epoch=epoch)
        
    def flush(self):
        """Flush buffer to disk."""
        with self._lock:
            if self._file:
                self._file.flush()
                
    def close(self):
        """Close the CSV file."""
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
    JSON Logger for structured metrics and metadata.
    
    Creates JSON files with full training history,
    configuration, and metadata for reproducibility.
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        filename: str = "training_log.json",
        pretty_print: bool = True
    ):
        """
        Initialize JSON logger.
        
        Args:
            log_dir: Directory to save JSON files
            filename: Name of the JSON file
            pretty_print: Whether to format JSON for readability
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / filename
        self.pretty_print = pretty_print
        
        self._lock = threading.Lock()
        
        # Initialize log structure
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
        
        # Load existing log if present
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    self._log = json.load(f)
            except json.JSONDecodeError:
                pass  # Start fresh if file is corrupted
                
    def log_config(self, config: Dict[str, Any]):
        """
        Log training configuration.
        
        Args:
            config: Configuration dictionary
        """
        with self._lock:
            self._log['config'] = self._serialize(config)
            self._save()
            
    def log_metadata(self, metadata: Dict[str, Any]):
        """
        Log additional metadata.
        
        Args:
            metadata: Metadata dictionary
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
        Log a training step.
        
        Args:
            step: Global step
            metrics: Metrics dictionary
            epoch: Current epoch (optional)
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
            
            # Don't save every step (too slow)
            # Save is done on epoch end or explicitly
            
    def log_validation(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """
        Log validation results.
        
        Args:
            epoch: Current epoch
            metrics: Validation metrics
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
        Log complete epoch summary.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics (optional)
            lr: Learning rate (optional)
            time_elapsed: Time for this epoch (optional)
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
        Log checkpoint save event.
        
        Args:
            epoch: Current epoch
            path: Checkpoint file path
            is_best: Whether this is the best model
            metrics: Associated metrics (optional)
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
        Log a training event.
        
        Args:
            event_type: Type of event (e.g., 'early_stop', 'lr_update')
            message: Event message
            data: Additional event data (optional)
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
        """Get training history."""
        return self._log['history'].get(key, [])
        
    def get_best_checkpoint(self) -> Optional[Dict]:
        """Get the best checkpoint entry."""
        best_checkpoints = [c for c in self._log['checkpoints'] if c.get('is_best')]
        return best_checkpoints[-1] if best_checkpoints else None
        
    def _serialize(self, obj: Any) -> Any:
        """Serialize object for JSON storage."""
        import numpy as np

        # Handle numpy types
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle regular types
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
        """Save log to disk."""
        with open(self.filepath, 'w') as f:
            # Serialize to handle numpy types and other non-JSON-serializable objects
            serialized_log = self._serialize(self._log)
            if self.pretty_print:
                json.dump(serialized_log, f, indent=2)
            else:
                json.dump(serialized_log, f)
                
    def flush(self):
        """Force save to disk."""
        with self._lock:
            self._save()
            
    def close(self):
        """Close and save final state."""
        self.flush()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class MetricsAggregator:
    """
    Aggregates metrics over batches for epoch-level reporting.
    
    Computes running averages and stores per-batch values.
    """
    
    def __init__(self):
        self._values: Dict[str, List[float]] = {}
        self._counts: Dict[str, int] = {}
        
    def update(self, metrics: Dict[str, float], batch_size: int = 1):
        """
        Update with new batch metrics.
        
        Args:
            metrics: Batch metrics
            batch_size: Size of batch (for weighted averaging)
        """
        for key, value in metrics.items():
            if key not in self._values:
                self._values[key] = []
                self._counts[key] = 0
            self._values[key].append(value)
            self._counts[key] += batch_size
            
    def get_average(self, key: str) -> float:
        """Get average value for a metric."""
        if key not in self._values or not self._values[key]:
            return 0.0
        return sum(self._values[key]) / len(self._values[key])
        
    def get_averages(self) -> Dict[str, float]:
        """Get all average values."""
        return {key: self.get_average(key) for key in self._values}
        
    def get_values(self, key: str) -> List[float]:
        """Get all values for a metric."""
        return self._values.get(key, [])
        
    def reset(self):
        """Reset all accumulated values."""
        self._values.clear()
        self._counts.clear()
