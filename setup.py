"""Setup script for the NeuroScope package.

This script provides comprehensive package configuration including
dependencies, entry points, and package metadata.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages, find_namespace_packages

# Read README for long description
def read_readme():
    """Read README file for long description."""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return "NeuroScope: Domain-aware standardization of multimodal glioma MRI"

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt."""
    requirements_path = Path(__file__).parent / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return []

# Package configuration
setup(
    name="neuroscope",
    version="0.1.0",
    author="Ishrith Gowda",
    author_email="your.email@example.com",
    description="Domain-aware standardization of multimodal glioma MRI",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ishrith-gowda/NeuroScope",
    project_urls={
        "Bug Reports": "https://github.com/ishrith-gowda/NeuroScope/issues",
        "Source": "https://github.com/ishrith-gowda/NeuroScope",
        "Documentation": "https://neuroscope.readthedocs.io/",
    },
    
    # Package discovery
    packages=find_packages(),
    package_dir={"neuroscope": "neuroscope"},
    include_package_data=True,
    package_data={
        "neuroscope": [
            "config/defaults/*.json",
            "config/defaults/*.yaml",
            "data/splits/*.json",
            "tests/fixtures/*.nii.gz",
            "tests/fixtures/*.json",
        ]
    },
    
    # Python version requirements
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=[
        "torch>=1.11.0",
        "torchvision>=0.12.0",
        "numpy>=1.23.0",
        "pandas>=1.4.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "simpleitk>=2.2.0",
        "nibabel>=4.0.0",
        "torchio>=0.18.84",
        "scipy>=1.9.0",
        "scikit-image>=0.19.0",
        "scikit-learn>=1.1.0",
        "pyyaml>=6.0",
        "jsonschema>=4.0.0",
        "tqdm>=4.64.0",
        "click>=8.0.0",
        "tensorboard>=2.8.0",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.8.0",
            "black>=22.1.0",
            "isort>=5.10.0",
            "pylint>=2.12.0",
            "mypy>=0.931",
            "flake8>=5.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "advanced": [
            "monai>=1.0.0",
            "dicom2nifti>=2.4.7",
            "hd-bet>=1.0.0",
            "antspyx>=0.3.0",
        ],
        "visualization": [
            "plotly>=5.0.0",
            "bokeh>=2.4.0",
            "dash>=2.0.0",
        ],
        "monitoring": [
            "wandb>=0.12.0",
            "mlflow>=2.3.0",
        ],
        "performance": [
            "numba>=0.56.0",
            "cupy-cuda11x>=10.0.0; platform_machine == 'x86_64'",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.4.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.8.0",
            "black>=22.1.0",
            "isort>=5.10.0",
            "pylint>=2.12.0",
            "mypy>=0.931",
            "flake8>=5.0.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
            "monai>=1.0.0",
            "dicom2nifti>=2.4.7",
            "hd-bet>=1.0.0",
            "antspyx>=0.3.0",
            "plotly>=5.0.0",
            "bokeh>=2.4.0",
            "dash>=2.0.0",
            "wandb>=0.12.0",
            "mlflow>=2.3.0",
            "numba>=0.56.0",
            "cupy-cuda11x>=10.0.0; platform_machine == 'x86_64'",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.4.0",
        ],
    },
    
    # Entry points for CLI
    entry_points={
        "console_scripts": [
            "neuroscope=neuroscope.scripts.cli.neuroscope_cli:main",
            "neuroscope-preprocess=neuroscope.scripts.cli.preprocess:main",
            "neuroscope-train=neuroscope.scripts.cli.train:main",
            "neuroscope-evaluate=neuroscope.scripts.cli.evaluate:main",
            "neuroscope-pipeline=neuroscope.scripts.cli.pipeline:main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: GPU",
        "Environment :: Console",
    ],
    
    # Keywords
    keywords=[
        "medical imaging",
        "mri",
        "cyclegan",
        "domain adaptation",
        "glioma",
        "brain tumor",
        "neuroimaging",
        "deep learning",
        "computer vision",
        "biomedical",
    ],
    
    # License
    license="MIT",
    
    # Minimum Python version
    python_requires=">=3.9",
    
    # Zip safe
    zip_safe=False,
    
    # Test suite
    test_suite="tests",
    
    # Command line options
    cmdclass={},
    
    # Data files
    data_files=[
        ("neuroscope/config", ["neuroscope/config/defaults/training_config.py"]),
        ("neuroscope/tests/fixtures", []),
    ],
    
    # Scripts
    scripts=[
        "scripts/cli/neuroscope_cli.py",
        "scripts/pipeline/data_preparation/preprocessing_pipeline.py",
        "scripts/pipeline/model_development/training_pipeline.py",
    ],
)