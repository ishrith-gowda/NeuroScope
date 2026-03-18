"""
neuroscope utilities package.

common utilities for i/o, visualization, logging,
and configuration management.

modules:
    - io: file i/o and checkpoint management
    - visualization: plotting and visualization
    - logging: structured logging
    - config: configuration management
"""

from .io import (
    # nifti handling
    load_nifti,
    save_nifti,
    load_nifti_as_tensor,
    
    # checkpoint management
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
    save_cyclegan_checkpoint,
    load_cyclegan_checkpoint,
    
    # config handling
    load_config as load_config_io,
    save_config as save_config_io,
    merge_configs,
    
    # utilities
    ensure_dir,
    list_files,
    copy_file,
)

from .visualization import (
    # slice visualization
    plot_slice,
    plot_slice_comparison,
    plot_montage,
    plot_difference_map,
    plot_attention_overlay,
    
    # training visualization
    plot_training_curves,
    plot_loss_landscape,
    
    # statistical visualization
    plot_box_comparison,
    plot_violin_comparison,
    plot_confidence_intervals,
    plot_effect_size_forest,
    
    # figure utilities
    create_figure_grid,
    save_publication_figure,
)

from .logging import (
    # logger
    setup_logger,
    get_logger,
    
    # metric tracking
    MetricTracker,
    ExperimentLogger,
)

from .config import (
    # configuration classes
    ModelConfig,
    DataConfig,
    LossConfig,
    OptimizerConfig,
    TrainingConfig,
    EvaluationConfig,
    ExperimentConfig,
    
    # configuration utilities
    ConfigManager,
    get_default_config,
    load_config,
    save_config,
)

__all__ = [
    # i/o
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
    
    # visualization
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
    
    # logging
    'setup_logger',
    'get_logger',
    'MetricTracker',
    'ExperimentLogger',
    
    # config
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
