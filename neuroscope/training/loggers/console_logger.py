"""
Console Logger with Rich Formatting.

Beautiful, informative console output for training progress.

Author: NeuroScope Research Team
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import sys
import time


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    GRAY = '\033[90m'
    WHITE = '\033[97m'
    
    
def colorize(text: str, color: str) -> str:
    """Add color to text."""
    return f"{color}{text}{Colors.END}"


class ConsoleLogger:
    """
    Rich console logger for training progress visualization.
    
    Features:
    - Colored output
    - Progress bars
    - Formatted tables
    - Real-time metrics
    - Time estimates
    """
    
    def __init__(
        self,
        experiment_name: str = "Training",
        verbose: int = 2,
        use_color: bool = True,
        log_to_file: bool = False,
        log_file: Optional[str] = None
    ):
        """
        Initialize console logger.
        
        Args:
            experiment_name: Name of the experiment
            verbose: Verbosity level (0=silent, 1=minimal, 2=normal, 3=detailed)
            use_color: Whether to use colored output
            log_to_file: Whether to also log to file
            log_file: Path to log file (if log_to_file is True)
        """
        self.experiment_name = experiment_name
        self.verbose = verbose
        self.use_color = use_color
        self.log_to_file = log_to_file
        self.log_file = log_file
        
        self._start_time: Optional[float] = None
        self._epoch_start_time: Optional[float] = None
        self._current_epoch: int = 0
        self._total_epochs: int = 0
        
        if log_to_file and log_file:
            self._file = open(log_file, 'a')
        else:
            self._file = None
            
    def _color(self, text: str, color: str) -> str:
        """Apply color if enabled."""
        if self.use_color:
            return colorize(text, color)
        return text
        
    def _write(self, message: str, end: str = '\n'):
        """Write message to console and optionally to file."""
        print(message, end=end, flush=True)
        if self._file:
            # Strip color codes for file
            import re
            clean_message = re.sub(r'\033\[[0-9;]+m', '', message)
            self._file.write(clean_message + end)
            self._file.flush()
            
    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
            
    def _format_metric(self, name: str, value: float, precision: int = 4) -> str:
        """Format a metric for display."""
        if 'psnr' in name.lower():
            return f"{name}={value:.2f}dB"
        elif 'lr' in name.lower() or 'learning_rate' in name.lower():
            return f"{name}={value:.2e}"
        else:
            return f"{name}={value:.{precision}f}"
            
    # =========================================================================
    # Banner and Headers
    # =========================================================================
    
    def print_banner(self):
        """Print training banner."""
        if self.verbose < 1:
            return
            
        banner = f"""
{self._color('=' * 70, Colors.CYAN)}
{self._color('  neuroscope - 2.5d sa-cyclegan training', Colors.BOLD + Colors.CYAN)}
{self._color('  ' + '=' * 66, Colors.CYAN)}
{self._color(f'  experiment: {self.experiment_name}', Colors.WHITE)}
{self._color(f'  started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', Colors.GRAY)}
{self._color('=' * 70, Colors.CYAN)}
"""
        self._write(banner)
        
    def print_config(self, config: Dict[str, Any]):
        """Print configuration summary."""
        if self.verbose < 2:
            return
            
        self._write(f"\n{self._color('configuration:', Colors.BOLD)}")
        self._write(self._color('-' * 50, Colors.GRAY))
        
        for key, value in config.items():
            if isinstance(value, dict):
                self._write(f"  {self._color(key + ':', Colors.CYAN)}")
                for k, v in value.items():
                    self._write(f"    {k}: {v}")
            else:
                self._write(f"  {self._color(key + ':', Colors.CYAN)} {value}")
                
        self._write(self._color('-' * 50, Colors.GRAY) + "\n")
        
    def print_model_summary(
        self,
        model_name: str,
        total_params: int,
        trainable_params: int
    ):
        """Print model summary."""
        if self.verbose < 1:
            return
            
        self._write(f"\n{self._color('model summary:', Colors.BOLD)}")
        self._write(f"  name: {model_name}")
        self._write(f"  total parameters: {total_params:,}")
        self._write(f"  trainable parameters: {trainable_params:,}")
        self._write(f"  size: ~{total_params * 4 / 1e6:.1f} MB\n")
        
    def print_data_summary(
        self,
        train_samples: int,
        val_samples: int,
        test_samples: int,
        batch_size: int
    ):
        """Print data summary."""
        if self.verbose < 1:
            return
            
        train_batches = (train_samples + batch_size - 1) // batch_size
        val_batches = (val_samples + batch_size - 1) // batch_size
        
        self._write(f"\n{self._color('data summary:', Colors.BOLD)}")
        self._write(f"  training samples: {train_samples:,} ({train_batches} batches)")
        self._write(f"  validation samples: {val_samples:,} ({val_batches} batches)")
        self._write(f"  test samples: {test_samples:,}")
        self._write(f"  batch size: {batch_size}\n")
        
    # =========================================================================
    # Training Progress
    # =========================================================================
    
    def on_training_start(self, total_epochs: int):
        """Called at the start of training."""
        self._start_time = time.time()
        self._total_epochs = total_epochs
        
        if self.verbose >= 1:
            self._write(f"\n{self._color('starting training for', Colors.GREEN)} "
                       f"{self._color(str(total_epochs), Colors.BOLD)} "
                       f"{self._color('epochs', Colors.GREEN)}\n")
            
    def on_training_end(
        self,
        final_metrics: Optional[Dict[str, float]] = None
    ):
        """Called at the end of training."""
        total_time = time.time() - self._start_time if self._start_time else 0
        
        self._write(f"\n{self._color('=' * 70, Colors.GREEN)}")
        self._write(f"{self._color('training complete', Colors.BOLD + Colors.GREEN)}")
        self._write(f"  total time: {self._format_time(total_time)}")

        if final_metrics:
            self._write(f"  final metrics:")
            for name, value in final_metrics.items():
                self._write(f"    {self._format_metric(name, value)}")
                
        self._write(self._color('=' * 70, Colors.GREEN) + "\n")
        
    def on_epoch_start(self, epoch: int, total_epochs: int):
        """Called at the start of each epoch."""
        self._current_epoch = epoch
        self._total_epochs = total_epochs
        self._epoch_start_time = time.time()
        
        if self.verbose >= 1:
            progress = f"[{epoch}/{total_epochs}]"
            self._write(f"\n{self._color('epoch', Colors.BOLD)} "
                       f"{self._color(progress, Colors.CYAN)}")
            
    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None
    ):
        """Called at the end of each epoch."""
        epoch_time = time.time() - self._epoch_start_time if self._epoch_start_time else 0
        
        # Estimate remaining time
        elapsed_total = time.time() - self._start_time if self._start_time else 0
        avg_epoch_time = elapsed_total / epoch
        remaining_epochs = self._total_epochs - epoch
        eta = avg_epoch_time * remaining_epochs
        
        if self.verbose >= 1:
            # Progress bar
            progress = epoch / self._total_epochs
            bar_width = 30
            filled = int(bar_width * progress)
            bar = '█' * filled + '░' * (bar_width - filled)
            pct = progress * 100
            
            self._write(f"  {self._color(bar, Colors.GREEN)} {pct:.1f}%")
            self._write(f"  time: {self._format_time(epoch_time)} | "
                       f"eta: {self._format_time(eta)}")
            
        if self.verbose >= 2:
            # Training metrics
            train_str = " | ".join([
                self._format_metric(k, v)
                for k, v in train_metrics.items()
            ])
            self._write(f"  {self._color('train:', Colors.BLUE)} {train_str}")

            # Validation metrics
            if val_metrics:
                val_str = " | ".join([
                    self._format_metric(k, v)
                    for k, v in val_metrics.items()
                ])
                self._write(f"  {self._color('val:', Colors.YELLOW)} {val_str}")

            # Learning rate
            if lr is not None:
                self._write(f"  {self._color('lr:', Colors.GRAY)} {lr:.2e}")
                
    def on_batch_end(
        self,
        batch: int,
        total_batches: int,
        metrics: Dict[str, float],
        samples_per_sec: Optional[float] = None
    ):
        """Called at the end of each batch (for verbose >= 3)."""
        if self.verbose < 3:
            return
            
        metrics_str = " | ".join([
            f"{k}={v:.4f}" for k, v in metrics.items()
        ])
        
        speed_str = f" | {samples_per_sec:.1f} samples/s" if samples_per_sec else ""
        
        # Overwrite line for progress
        progress = f"[{batch}/{total_batches}]"
        line = f"\r  batch {progress}: {metrics_str}{speed_str}"
        self._write(line, end='')
        
        if batch == total_batches:
            self._write('')  # New line at end
            
    # =========================================================================
    # Events and Messages
    # =========================================================================
    
    def log_checkpoint(
        self,
        path: str,
        is_best: bool = False,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Log checkpoint save."""
        if self.verbose < 1:
            return
            
        if is_best:
            msg = self._color('best model saved: ', Colors.YELLOW)
        else:
            msg = self._color('checkpoint saved: ', Colors.GRAY)
            
        self._write(f"  {msg}{path}")
        
        if is_best and metrics and self.verbose >= 2:
            for name, value in metrics.items():
                self._write(f"    {self._format_metric(name, value)}")
                
    def log_early_stop(self, epoch: int, patience: int, best_metric: float):
        """Log early stopping."""
        self._write(f"\n{self._color('warning: early stopping triggered', Colors.YELLOW)}")
        self._write(f"  epoch: {epoch}")
        self._write(f"  patience: {patience} epochs without improvement")
        self._write(f"  best metric: {best_metric:.4f}\n")
        
    def log_lr_update(self, old_lr: float, new_lr: float, reason: str = ""):
        """Log learning rate update."""
        if self.verbose < 2:
            return
            
        msg = f"  {self._color('lr:', Colors.GRAY)} {old_lr:.2e} → {new_lr:.2e}"
        if reason:
            msg += f" ({reason})"
        self._write(msg)
        
    def log_sample_saved(self, path: str, epoch: int):
        """Log sample save."""
        if self.verbose < 2:
            return
            
        self._write(f"  {self._color('samples saved:', Colors.GRAY)} {path}")

    def log_info(self, message: str):
        """Log info message."""
        if self.verbose >= 1:
            self._write(f"  {self._color('info:', Colors.BLUE)} {message}")
            
    def log_warning(self, message: str):
        """Log warning message."""
        self._write(f"  {self._color('warning:', Colors.YELLOW)} {message}")

    def log_error(self, message: str):
        """Log error message."""
        self._write(f"  {self._color('error:', Colors.RED)} {message}")
        
    def log_success(self, message: str):
        """Log success message."""
        self._write(f"  {self._color('+', Colors.GREEN)} {message}")
        
    # =========================================================================
    # Tables
    # =========================================================================
    
    def print_metrics_table(
        self,
        metrics: Dict[str, Dict[str, float]],
        title: str = "Metrics Summary"
    ):
        """
        Print a formatted metrics table.
        
        Args:
            metrics: Dictionary of {row_name: {col_name: value}}
            title: Table title
        """
        if self.verbose < 1:
            return
            
        if not metrics:
            return
            
        # Get all column names
        all_cols = set()
        for row_metrics in metrics.values():
            all_cols.update(row_metrics.keys())
        cols = sorted(list(all_cols))
        
        # Calculate column widths
        col_width = max(12, max(len(c) for c in cols) + 2)
        row_width = max(12, max(len(r) for r in metrics.keys()) + 2)
        
        # Print title
        self._write(f"\n{self._color(title, Colors.BOLD)}")
        
        # Print header
        header = " " * row_width + "".join(f"{c:>{col_width}}" for c in cols)
        self._write(self._color(header, Colors.CYAN))
        self._write("-" * (row_width + col_width * len(cols)))
        
        # Print rows
        for row_name, row_metrics in metrics.items():
            row_str = f"{row_name:<{row_width}}"
            for col in cols:
                val = row_metrics.get(col, float('nan'))
                row_str += f"{val:>{col_width}.4f}"
            self._write(row_str)
            
        self._write("")
        
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def close(self):
        """Close the logger."""
        if self._file:
            self._file.close()
            self._file = None
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
