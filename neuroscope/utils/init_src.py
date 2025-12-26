"""
NeuroScope Utilities Package.

Common utilities for I/O, visualization, logging,
and configuration management.

Modules:
    - io: File I/O and checkpoint management
    - visualization: Plotting and visualization
    - logging: Structured logging
    - config: Configuration management
"""

from .io import (
    # NIfTI handling
    load_nifti,
    save_nifti,
    load_nifti_as_tensor,
    
    # Checkpoint management
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
    save_cyclegan_checkpoint,
    load_cyclegan_checkpoint,
    
    # Config handling
    load_config as load_config_io,
    save_config as save_config_io,
    merge_configs,
    
    # Utilities
    ensure_dir,
    list_files,
    copy_file,
)

from .visualization import (
    # Slice visualization
    plot_slice,
    plot_slice_comparison,
    plot_montage,
    plot_difference_map,
    plot_attention_overlay,
    
    # Training visualization
    plot_training_curves,
    plot_loss_landscape,
    
    # Statistical visualization
    plot_box_comparison,
    plot_violin_comparison,
    plot_confidence_intervals,
    plot_effect_size_forest,
    
    # Figure utilities
    create_figure_grid,
    save_publication_figure,
)

from .logging import (
    # Logger
    setup_logger,
    get_logger,
    
    # Metric tracking
    MetricTracker,
    ExperimentLogger,
)

from .config import (
    # Configuration classes
    ModelConfig,
    DataConfig,
    LossConfig,
    OptimizerConfig,
    TrainingConfig,
    EvaluationConfig,
    ExperimentConfig,
    
    # Configuration utilities
    ConfigManager,
    get_default_config,
    load_config,
    save_config,
)

__all__ = [
    # I/O
    'load_nifti',
    'save_nifti',
    'load_nifti_as_tensor',
    'save_checkpoint',
    'load_checkpoint',
    'get_latest_checkpoint',
    'save_cyclegan_checkpoint',
    'load_cyclegan_checkpoint',
    'ensure_dir',
    'list_files',
    'copy_file',
    'merge_configs',
    
    # Visualization
    'plot_slice',
    'plot_slice_comparison',
    'plot_montage',
    'plot_difference_map',
    'plot_attention_overlay',
    'plot_training_curves',
    'plot_loss_landscape',
    'plot_box_comparison',
    'plot_violin_comparison',
    'plot_confidence_intervals',
    'plot_effect_size_forest',
    'create_figure_grid',
    'save_publication_figure',
    
    # Logging
    'setup_logger',
    'get_logger',
    'MetricTracker',
    'ExperimentLogger',
    
    # Config
    'ModelConfig',
    'DataConfig',
    'LossConfig',
    'OptimizerConfig',
    'TrainingConfig',
    'EvaluationConfig',
    'ExperimentConfig',
    'ConfigManager',
    'get_default_config',
    'load_config',
    'save_config',
]
