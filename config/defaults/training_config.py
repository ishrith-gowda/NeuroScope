"""Default training configuration for CycleGAN.

This module provides default configuration values for CycleGAN training,
including model parameters, optimizer settings, and training hyperparameters.
"""

from typing import Dict, Any, List


def get_default_training_config() -> Dict[str, Any]:
    """Get default training configuration.
    
    Returns:
        Dictionary containing default training configuration
    """
    return {
        # Model configuration
        'model': {
            'input_channels': 4,
            'output_channels': 4,
            'generator_channels': 64,
            'discriminator_channels': 64,
            'n_residual_blocks': 9,
            'lambda_cycle': 10.0,
            'lambda_identity': 5.0
        },
        
        # Training configuration
        'training': {
            'n_epochs': 100,
            'batch_size': 8,
            'log_interval': 50,
            'sample_interval': 200,
            'checkpoint_interval': 10,
            'grad_clip': 0.0,
            'use_tensorboard': True,
            'log_dir': 'runs',
            'sample_dir': 'samples',
            'checkpoint_dir': 'checkpoints'
        },
        
        # Optimizer configuration
        'generator_optimizer': {
            'type': 'adam',
            'lr': 0.0002,
            'betas': (0.5, 0.999),
            'weight_decay': 0.0
        },
        
        'discriminator_optimizer': {
            'type': 'adam',
            'lr': 0.0002,
            'betas': (0.5, 0.999),
            'weight_decay': 0.0
        },
        
        # Learning rate scheduler
        'scheduler': {
            'type': 'step',
            'step_size': 50,
            'gamma': 0.5
        },
        
        # Callbacks
        'callbacks': {
            'early_stopping': {
                'enabled': False,
                'patience': 10,
                'min_delta': 0.001,
                'monitor': 'total',
                'mode': 'min'
            },
            'checkpoint': {
                'enabled': True,
                'checkpoint_dir': 'checkpoints',
                'save_best_only': True,
                'save_frequency': 10,
                'monitor': 'total',
                'mode': 'min'
            },
            'logging': {
                'enabled': True,
                'log_frequency': 10,
                'log_dir': 'logs',
                'save_logs': True
            },
            'validation': {
                'enabled': False,
                'validation_frequency': 5,
                'metrics': ['mse', 'mae', 'ssim']
            }
        },
        
        # Data configuration
        'data': {
            'data_root': '/path/to/data',
            'metadata_json': '/path/to/metadata.json',
            'num_workers': 4,
            'pin_memory': True,
            'shuffle': True,
            'normalize_range': (-1, 1)
        },
        
        # Device configuration
        'device': {
            'use_cuda': True,
            'cuda_device': 0,
            'use_mps': False  # For Apple Silicon
        },
        
        # Reproducibility
        'reproducibility': {
            'seed': 42,
            'deterministic': True
        }
    }


def get_default_preprocessing_config() -> Dict[str, Any]:
    """Get default preprocessing configuration.
    
    Returns:
        Dictionary containing default preprocessing configuration
    """
    return {
        # Data paths
        'paths': {
            'raw_data_root': '/path/to/raw/data',
            'preprocessed_data_root': '/path/to/preprocessed/data',
            'metadata_file': '/path/to/metadata.json',
            'output_dir': '/path/to/output'
        },
        
        # Preprocessing steps
        'preprocessing_steps': [
            {
                'name': 'skull_stripping',
                'enabled': True,
                'method': 'hd_bet',
                'parameters': {
                    'device': 'cuda',
                    'mode': 'fast'
                }
            },
            {
                'name': 'bias_correction',
                'enabled': True,
                'method': 'n4',
                'parameters': {
                    'shrink_factor': 4,
                    'convergence_threshold': 0.001,
                    'maximum_iterations': 50
                }
            },
            {
                'name': 'normalization',
                'enabled': True,
                'method': 'percentile',
                'parameters': {
                    'lower_percentile': 0.5,
                    'upper_percentile': 99.5,
                    'target_range': (0, 1)
                }
            },
            {
                'name': 'resampling',
                'enabled': True,
                'method': 'isotropic',
                'parameters': {
                    'target_spacing': (1.0, 1.0, 1.0),
                    'interpolation': 'linear'
                }
            }
        ],
        
        # Quality control
        'quality_control': {
            'enabled': True,
            'check_file_integrity': True,
            'check_intensity_range': True,
            'check_spacing': True,
            'generate_visualizations': True
        },
        
        # Parallel processing
        'parallel_processing': {
            'enabled': True,
            'max_workers': 4,
            'chunk_size': 10
        }
    }


def get_default_evaluation_config() -> Dict[str, Any]:
    """Get default evaluation configuration.
    
    Returns:
        Dictionary containing default evaluation configuration
    """
    return {
        # Evaluation metrics
        'metrics': {
            'image_quality': ['mse', 'mae', 'ssim', 'psnr'],
            'bias_assessment': ['skewness', 'kurtosis', 'intensity_range'],
            'statistical': ['mean', 'std', 'median', 'percentiles']
        },
        
        # Visualization
        'visualization': {
            'enabled': True,
            'output_dir': 'evaluation_results',
            'formats': ['png', 'pdf'],
            'dpi': 300,
            'figure_size': (12, 8)
        },
        
        # Reporting
        'reporting': {
            'enabled': True,
            'output_format': 'json',
            'include_plots': True,
            'include_statistics': True
        }
    }


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    
    for config in configs:
        merged.update(config)
    
    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    required_keys = ['model', 'training', 'data']
    
    for key in required_keys:
        if key not in config:
            print(f"Missing required configuration key: {key}")
            return False
    
    # Validate model configuration
    model_config = config.get('model', {})
    required_model_keys = ['input_channels', 'output_channels', 'generator_channels']
    
    for key in required_model_keys:
        if key not in model_config:
            print(f"Missing required model configuration key: {key}")
            return False
    
    # Validate training configuration
    training_config = config.get('training', {})
    required_training_keys = ['n_epochs', 'batch_size']
    
    for key in required_training_keys:
        if key not in training_config:
            print(f"Missing required training configuration key: {key}")
            return False
    
    return True