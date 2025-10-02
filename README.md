# NeuroScope

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Domain-aware standardization of multimodal glioma MRI using CycleGAN-based framework for standardizing multi-institutional glioblastoma MRI scans (T1, T1ce, T2, FLAIR) across different scanner protocols.**

## 🧠 Overview

NeuroScope tackles scanner-protocol heterogeneity in glioblastoma MRI by learning an unsupervised image-to-image translation between BraTS (TCGA-GBM) and UPenn-GBM datasets. The CycleGAN operates on four-channel 2D axial slices (T1, T1ce, T2, FLAIR) to produce harmonized volumes for downstream radiomic analysis.

## ✨ Key Features

- **Advanced Preprocessing Pipeline**: Comprehensive medical imaging preprocessing including skull stripping, bias correction, normalization, and resampling
- **CycleGAN Implementation**: State-of-the-art domain adaptation with ResNet-based generators and PatchGAN discriminators
- **Professional Architecture**: Modular, extensible design following best practices for medical imaging research
- **Comprehensive Evaluation**: Bias assessment, quality control, and statistical analysis tools
- **CLI Interface**: Command-line tools for all major operations
- **Extensive Documentation**: Professional documentation with examples and tutorials

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ishrith-gowda/NeuroScope.git
cd NeuroScope

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Install with optional dependencies
pip install -e ".[all]"
```

### Basic Usage

```bash
# Preprocess data
neuroscope preprocess --input-dir /path/to/raw --output-dir /path/to/processed

# Train CycleGAN model
neuroscope train --data-root /path/to/data --output-dir /path/to/results

# Run complete pipeline
neuroscope pipeline --input-dir /path/to/raw --output-dir /path/to/results
```

### Python API

```python
import neuroscope
from neuroscope.config import get_default_training_config
from neuroscope.models.architectures import CycleGAN
from neuroscope.training.trainers import CycleGANTrainer

# Load configuration
config = get_default_training_config()

# Initialize model
model = CycleGAN(**config['model'])

# Train model
trainer = CycleGANTrainer(model, optimizer, device, config)
trainer.train_epoch(train_loader_a, train_loader_b, epoch=0)
```

## 📁 Project Structure

```
neuroscope/
├── neuroscope/                    # Main package
│   ├── core/                      # Core utilities
│   │   ├── logging/              # Logging system
│   │   ├── config/               # Configuration management
│   │   ├── constants/            # Constants and enums
│   │   └── validators/           # Validation utilities
│   ├── data/                     # Data handling
│   │   ├── loaders/             # Data loaders
│   │   ├── transforms/          # Data transformations
│   │   ├── datasets/            # Dataset implementations
│   │   └── splits/              # Data splitting utilities
│   ├── models/                   # Model implementations
│   │   ├── generators/          # Generator networks
│   │   ├── discriminators/      # Discriminator networks
│   │   ├── losses/              # Loss functions
│   │   └── architectures/       # Complete architectures
│   ├── preprocessing/           # Preprocessing pipeline
│   │   ├── bias_correction/     # Bias field correction
│   │   ├── registration/        # Image registration
│   │   ├── normalization/      # Intensity normalization
│   │   └── skull_stripping/     # Skull stripping
│   ├── training/                # Training framework
│   │   ├── trainers/            # Training implementations
│   │   ├── optimizers/          # Optimizers and schedulers
│   │   ├── schedulers/          # Learning rate schedulers
│   │   └── callbacks/           # Training callbacks
│   ├── evaluation/              # Evaluation tools
│   │   ├── metrics/             # Evaluation metrics
│   │   ├── analyzers/           # Analysis tools
│   │   └── reporters/           # Reporting utilities
│   ├── visualization/           # Visualization tools
│   │   ├── plotters/            # Plotting utilities
│   │   ├── montages/            # Image montages
│   │   └── dashboards/          # Interactive dashboards
│   └── utils/                   # Utility functions
│       ├── io/                  # I/O utilities
│       ├── math/                # Mathematical utilities
│       └── image/               # Image processing utilities
├── scripts/                     # Command-line scripts
│   ├── cli/                     # CLI implementations
│   ├── pipeline/               # Pipeline scripts
│   ├── experiments/            # Experimental scripts
│   └── utilities/              # Utility scripts
├── tests/                       # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── fixtures/               # Test fixtures
├── docs/                        # Documentation
│   ├── api/                    # API documentation
│   ├── guides/                 # User guides
│   └── examples/               # Example notebooks
├── examples/                    # Example scripts
│   ├── basic/                  # Basic examples
│   └── advanced/               # Advanced examples
└── config/                      # Configuration files
    ├── defaults/               # Default configurations
    └── experiments/            # Experiment configurations
```

## 🔧 Configuration

NeuroScope uses a comprehensive configuration system with sensible defaults:

```python
from neuroscope.config import get_default_training_config

# Get default configuration
config = get_default_training_config()

# Customize configuration
config['training']['n_epochs'] = 200
config['training']['batch_size'] = 16
config['model']['lambda_cycle'] = 15.0

# Validate configuration
from neuroscope.config import validate_config
assert validate_config(config)
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=neuroscope

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

## 📊 Evaluation

NeuroScope provides comprehensive evaluation tools:

```python
from neuroscope.evaluation.analyzers import analyze_dataset_bias
from neuroscope.evaluation.reporters import create_bias_visualization

# Analyze dataset bias
bias_results = analyze_dataset_bias(metadata, splits_to_assess=['train', 'val'])

# Create visualizations
create_bias_visualization(bias_results, output_dir='results/')
```

## 📈 Monitoring

Track training progress with TensorBoard:

```bash
# Launch TensorBoard
tensorboard --logdir runs/

# Or use Weights & Biases
wandb login
neuroscope train --use-wandb
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Documentation

- [API Documentation](https://neuroscope.readthedocs.io/)
- [User Guide](docs/guides/user_guide.md)
- [Developer Guide](docs/guides/developer_guide.md)
- [Examples](examples/)

## 🏆 Citation

If you use NeuroScope in your research, please cite:

```bibtex
@software{neuroscope2024,
  title={NeuroScope: Domain-aware standardization of multimodal glioma MRI},
  author={Gowda, Ishrith},
  year={2024},
  url={https://github.com/ishrith-gowda/NeuroScope},
  version={0.1.0}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- BraTS dataset providers
- UPenn-GBM dataset contributors
- PyTorch and torchvision teams
- Medical imaging community

## 📞 Support

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/ishrith-gowda/NeuroScope/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/ishrith-gowda/NeuroScope/discussions)

---

**NeuroScope** - Advancing medical imaging through domain adaptation and standardization.