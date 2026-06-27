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

from .config import (
    # configuration utilities
    ConfigManager,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    LossConfig,
    # configuration classes
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    get_default_config,
    load_config,
    save_config,
)
from .io import (
    copy_file,
    # utilities
    ensure_dir,
    get_latest_checkpoint,
    list_files,
    load_checkpoint,
    load_cyclegan_checkpoint,
    # nifti handling
    load_nifti,
    load_nifti_as_tensor,
    # config handling
    merge_configs,
    # checkpoint management
    save_checkpoint,
    save_cyclegan_checkpoint,
    save_nifti,
)
from .logging import (
    ExperimentLogger,
    # metric tracking
    MetricTracker,
    get_logger,
    # logger
    setup_logger,
)
from .visualization import (
    # figure utilities
    create_figure_grid,
    plot_attention_overlay,
    # statistical visualization
    plot_box_comparison,
    plot_confidence_intervals,
    plot_difference_map,
    plot_effect_size_forest,
    plot_loss_landscape,
    plot_montage,
    # slice visualization
    plot_slice,
    plot_slice_comparison,
    # training visualization
    plot_training_curves,
    plot_violin_comparison,
    save_publication_figure,
)

__all__ = [
    "ConfigManager",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ExperimentLogger",
    "LossConfig",
    "MetricTracker",
    # config
    "ModelConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "copy_file",
    "create_figure_grid",
    "ensure_dir",
    "get_default_config",
    "get_latest_checkpoint",
    "get_logger",
    "list_files",
    "load_checkpoint",
    "load_config",
    "load_cyclegan_checkpoint",
    # i/o
    "load_nifti",
    "load_nifti_as_tensor",
    "merge_configs",
    "plot_attention_overlay",
    "plot_box_comparison",
    "plot_confidence_intervals",
    "plot_difference_map",
    "plot_effect_size_forest",
    "plot_loss_landscape",
    "plot_montage",
    # visualization
    "plot_slice",
    "plot_slice_comparison",
    "plot_training_curves",
    "plot_violin_comparison",
    "save_checkpoint",
    "save_config",
    "save_cyclegan_checkpoint",
    "save_nifti",
    "save_publication_figure",
    # logging
    "setup_logger",
]
