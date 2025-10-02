# NeuroScope API Reference

This document provides comprehensive API documentation for the NeuroScope package.

## Core Modules

### neuroscope.core

Core utilities and configuration management.

#### neuroscope.core.logging

Centralized logging system for NeuroScope.

```python
from neuroscope.core.logging import get_logger, configure_logging

# Get logger instance
logger = get_logger(__name__)

# Configure global logging
configure_logging(level=logging.INFO, log_dir=Path('logs'))

# Log structured data
logger.log_metrics({'loss': 0.5, 'accuracy': 0.95}, step=100)
```

#### neuroscope.core.config

Configuration management system.

```python
from neuroscope.config import (
    get_default_training_config,
    get_default_preprocessing_config,
    validate_config
)

# Get default configurations
training_config = get_default_training_config()
preprocessing_config = get_default_preprocessing_config()

# Validate configuration
is_valid = validate_config(training_config)
```

## Data Handling

### neuroscope.data

Data loading, transformation, and management utilities.

#### neuroscope.data.loaders

Data loaders for different dataset types.

```python
from neuroscope.data.loaders import get_cycle_domain_loaders

# Create CycleGAN data loaders
train_loader_a, train_loader_b, val_loader_a, val_loader_b = get_cycle_domain_loaders(
    data_root='/path/to/data',
    metadata_json='/path/to/metadata.json',
    batch_size=8,
    num_workers=4
)
```

#### neuroscope.data.transforms

Data transformation utilities.

```python
from neuroscope.data.transforms import MRITransforms

# Create MRI-specific transforms
transforms = MRITransforms(
    normalize_range=(-1, 1),
    augment=True,
    crop_size=(256, 256)
)
```

## Model Architecture

### neuroscope.models

Model implementations including generators, discriminators, and complete architectures.

#### neuroscope.models.architectures

Complete model architectures.

```python
from neuroscope.models.architectures import CycleGAN

# Initialize CycleGAN model
model = CycleGAN(
    input_channels=4,
    output_channels=4,
    generator_channels=64,
    discriminator_channels=64,
    n_residual_blocks=9,
    lambda_cycle=10.0,
    lambda_identity=5.0
)

# Get model information
model_info = model.get_model_info()
print(f"Total parameters: {model_info['total_parameters']}")
```

#### neuroscope.models.generators

Generator network implementations.

```python
from neuroscope.models.generators import ResNetGenerator

# Initialize ResNet generator
generator = ResNetGenerator(
    input_channels=4,
    output_channels=4,
    channels=64,
    n_residual_blocks=9
)
```

#### neuroscope.models.discriminators

Discriminator network implementations.

```python
from neuroscope.models.discriminators import PatchDiscriminator

# Initialize PatchGAN discriminator
discriminator = PatchDiscriminator(
    input_channels=4,
    channels=64
)
```

## Preprocessing

### neuroscope.preprocessing

Comprehensive preprocessing pipeline for medical imaging data.

#### neuroscope.preprocessing.normalization

Intensity normalization and data augmentation.

```python
from neuroscope.preprocessing.normalization import (
    VolumeNormalization,
    DataAugmentation,
    VolumePreprocessor
)

# Normalize volume
normalized = VolumeNormalization.min_max_normalization(
    volume, target_range=(0, 1), mask=brain_mask
)

# Apply data augmentation
augmented = DataAugmentation.random_flip(volume, axes=[0, 1], p=0.5)

# Create preprocessing pipeline
preprocessor = VolumePreprocessor([
    ('min_max_normalization', {'target_range': (0, 1)}),
    ('percentile_normalization', {'low_percentile': 1.0, 'high_percentile': 99.0})
])

# Process volume
processed = preprocessor.preprocess(volume, mask=brain_mask)
```

#### neuroscope.preprocessing.bias_correction

Bias field correction utilities.

```python
from neuroscope.preprocessing.bias_correction import N4Correction

# Initialize N4 bias correction
n4_corrector = N4Correction(
    shrink_factor=4,
    convergence_threshold=0.001,
    maximum_iterations=50
)

# Apply bias correction
corrected = n4_corrector.correct_bias(volume, mask=brain_mask)
```

## Training

### neuroscope.training

Comprehensive training framework for CycleGAN models.

#### neuroscope.training.trainers

Training implementations.

```python
from neuroscope.training.trainers import CycleGANTrainer

# Initialize trainer
trainer = CycleGANTrainer(
    model=model,
    optimizer=optimizer,
    device=device,
    config=config
)

# Train for one epoch
epoch_losses = trainer.train_epoch(train_loader_a, train_loader_b, epoch=0)

# Save checkpoint
trainer.save_checkpoint(epoch=10, checkpoint_dir='checkpoints/')

# Plot loss curves
trainer.plot_loss_curves('loss_curves.png')
```

#### neuroscope.training.optimizers

Optimizers and learning rate schedulers.

```python
from neuroscope.training.optimizers import CycleGANOptimizer

# Initialize optimizer
optimizer = CycleGANOptimizer(
    generators={'G_A2B': model.G_A2B, 'G_B2A': model.G_B2A},
    discriminators={'D_A': model.D_A, 'D_B': model.D_B},
    config=config
)

# Get current learning rates
learning_rates = optimizer.get_lr()
```

#### neuroscope.training.callbacks

Training callbacks for monitoring and control.

```python
from neuroscope.training.callbacks import (
    EarlyStoppingCallback,
    CheckpointCallback,
    LoggingCallback
)

# Initialize callbacks
callbacks = TrainingCallbacks(config)

# Callbacks are automatically used during training
```

## Evaluation

### neuroscope.evaluation

Comprehensive evaluation tools for model assessment.

#### neuroscope.evaluation.analyzers

Analysis tools for bias assessment and quality control.

```python
from neuroscope.evaluation.analyzers import (
    analyze_dataset_bias,
    assess_subject_bias,
    compute_slice_wise_statistics
)

# Analyze dataset bias
bias_results = analyze_dataset_bias(metadata, splits_to_assess=['train', 'val'])

# Assess individual subject bias
subject_bias = assess_subject_bias(
    subject_id='subject_001',
    modality_files={'T1': '/path/to/t1.nii.gz'},
    mask_file='/path/to/mask.nii.gz'
)

# Compute slice-wise statistics
stats = compute_slice_wise_statistics(image, mask)
```

#### neuroscope.evaluation.reporters

Reporting and visualization utilities.

```python
from neuroscope.evaluation.reporters import (
    save_bias_assessment_results,
    create_bias_visualization,
    print_bias_assessment_summary
)

# Save bias assessment results
save_bias_assessment_results(bias_results, output_path='bias_results.json')

# Create visualizations
create_bias_visualization(bias_results, output_dir='visualizations/')

# Print summary
print_bias_assessment_summary(summary, bias_results)
```

## Visualization

### neuroscope.visualization

Professional visualization tools for medical imaging data.

#### neuroscope.visualization.plotters

Plotting utilities for various data types.

```python
from neuroscope.visualization.plotters import (
    plot_loss_curves,
    plot_metric_distributions,
    plot_bias_assessment
)

# Plot training loss curves
plot_loss_curves(loss_history, save_path='loss_curves.png')

# Plot metric distributions
plot_metric_distributions(metrics, save_path='metrics.png')
```

#### neuroscope.visualization.montages

Image montage creation utilities.

```python
from neuroscope.visualization.montages import create_translation_montage

# Create translation montage
montage = create_translation_montage(
    real_a=real_images,
    fake_b=fake_images,
    real_b=real_images_b,
    fake_a=fake_images_a
)
```

## Utilities

### neuroscope.utils

General utility functions for I/O, mathematics, and image processing.

#### neuroscope.utils.io

I/O utilities for various file formats.

```python
from neuroscope.utils.io import (
    load_volume,
    save_volume,
    load_metadata,
    save_metadata
)

# Load medical imaging volume
volume = load_volume('/path/to/volume.nii.gz')

# Save processed volume
save_volume(volume, '/path/to/output.nii.gz', reference_file='/path/to/reference.nii.gz')

# Load metadata
metadata = load_metadata('/path/to/metadata.json')
```

#### neuroscope.utils.math

Mathematical utilities.

```python
from neuroscope.utils.math import (
    compute_statistics,
    normalize_tensor,
    compute_metrics
)

# Compute statistical measures
stats = compute_statistics(data, measures=['mean', 'std', 'skewness', 'kurtosis'])

# Normalize tensor
normalized = normalize_tensor(tensor, method='min_max', range=(-1, 1))

# Compute evaluation metrics
metrics = compute_metrics(predicted, target, metrics=['mse', 'mae', 'ssim'])
```

## Command Line Interface

### CLI Commands

NeuroScope provides comprehensive CLI tools:

```bash
# Preprocess data
neuroscope preprocess --input-dir /path/to/raw --output-dir /path/to/processed

# Train model
neuroscope train --data-root /path/to/data --output-dir /path/to/results

# Evaluate model
neuroscope evaluate --model-path /path/to/model --data-path /path/to/test

# Run complete pipeline
neuroscope pipeline --input-dir /path/to/raw --output-dir /path/to/results

# Generate configuration
neuroscope config --generate training --output training_config.json

# Validate configuration
neuroscope config --validate training_config.json
```

## Configuration Reference

### Training Configuration

```python
training_config = {
    'model': {
        'input_channels': 4,
        'output_channels': 4,
        'generator_channels': 64,
        'discriminator_channels': 64,
        'n_residual_blocks': 9,
        'lambda_cycle': 10.0,
        'lambda_identity': 5.0
    },
    'training': {
        'n_epochs': 100,
        'batch_size': 8,
        'log_interval': 50,
        'sample_interval': 200,
        'checkpoint_interval': 10
    },
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
    }
}
```

### Preprocessing Configuration

```python
preprocessing_config = {
    'preprocessing_steps': [
        {
            'name': 'skull_stripping',
            'enabled': True,
            'method': 'hd_bet',
            'parameters': {'device': 'cuda', 'mode': 'fast'}
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
        }
    ]
}
```

## Error Handling

NeuroScope provides comprehensive error handling:

```python
from neuroscope.core.exceptions import (
    NeuroScopeError,
    ConfigurationError,
    DataError,
    ModelError,
    TrainingError
)

try:
    # NeuroScope operations
    pass
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except DataError as e:
    print(f"Data error: {e}")
except ModelError as e:
    print(f"Model error: {e}")
except TrainingError as e:
    print(f"Training error: {e}")
except NeuroScopeError as e:
    print(f"NeuroScope error: {e}")
```

## Best Practices

### Code Organization

1. **Use configuration files** for all parameters
2. **Implement proper logging** throughout your code
3. **Validate inputs** before processing
4. **Use type hints** for better code clarity
5. **Write comprehensive tests** for all functionality

### Performance Optimization

1. **Use appropriate batch sizes** for your hardware
2. **Enable mixed precision training** when available
3. **Use data loading optimizations** (pin_memory, num_workers)
4. **Profile your code** to identify bottlenecks
5. **Use GPU acceleration** when possible

### Data Management

1. **Validate data integrity** before processing
2. **Use consistent file naming** conventions
3. **Maintain metadata** for all datasets
4. **Implement proper data splitting** strategies
5. **Monitor data quality** throughout the pipeline