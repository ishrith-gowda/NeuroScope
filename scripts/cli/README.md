# NeuroScope CLI Tools

This directory contains command-line interface (CLI) tools for the NeuroScope package, built on top of the refactored modular code.

## Available Tools

### N4 Bias Field Correction

Apply N4 bias field correction to MRI volumes:

```bash
python n4_bias_correction.py --input-dir /path/to/input --output-dir /path/to/output
```

#### Options

- `--input-dir`: Input directory containing MRI volumes (required)
- `--output-dir`: Output directory for corrected volumes (required)
- `--mask-dir`: Optional directory containing masks
- `--file-pattern`: Glob pattern for input files (default: `*.nii.gz`)
- `--save-bias`: Save estimated bias fields
- `--shrink-factor`: Shrink factor for downsampling (default: 4)
- `--iterations`: Number of iterations at each resolution level (default: 50 50 30 20)
- `--convergence-threshold`: Convergence threshold (default: 0.001)
- `--spline-order`: Order of B-spline used in the approximation (default: 3)
- `--spline-distance`: Distance between B-spline control points (default: 200.0)
- `--output-json`: Output JSON file for correction metrics
- `--verbose`: Enable verbose output

### Volume Preprocessing

Preprocess MRI volumes with various operations:

```bash
python preprocess_volumes.py --input-dir /path/to/input --output-dir /path/to/output --normalize percentile
```

#### Options

- `--input-dir`: Input directory containing MRI volumes (required)
- `--output-dir`: Output directory for preprocessed volumes (required)
- `--mask-dir`: Optional directory containing masks
- `--file-pattern`: Glob pattern for input files (default: `*.nii.gz`)
- `--normalize`: Normalization method (`minmax`, `zscore`, `percentile`, `histequal`, `whitestripe`, default: `percentile`)
- `--lower-pct`: Lower percentile for percentile normalization (default: 1.0)
- `--upper-pct`: Upper percentile for percentile normalization (default: 99.0)
- `--target-range`: Target intensity range [min, max] (default: 0 1)
- `--crop`: Crop size as three integers [x, y, z]
- `--crop-method`: Cropping method (`center`, `random`, default: `center`)
- `--rescale`: Scale factor for rescaling (single value or [x, y, z])
- `--target-shape`: Target shape for rescaling [x, y, z]
- `--output-json`: Output JSON file for preprocessing metadata
- `--verbose`: Enable verbose output

### Registration

Register MRI volumes:

```bash
python register_volumes.py --fixed-path /path/to/fixed --moving-path /path/to/moving --output-path /path/to/output
```

#### Options

- `--fixed-path`: Path to fixed (target) image or directory (required)
- `--moving-path`: Path to moving (source) image or directory (required)
- `--output-path`: Output directory for registered images (required)
- `--file-pattern`: Glob pattern for input files when directories are provided (default: `*.nii.gz`)
- `--fixed-mask-path`: Optional path to fixed image mask or directory
- `--moving-mask-path`: Optional path to moving image mask or directory
- `--registration-type`: Type of registration (`rigid`, `affine`, `deformable`, default: `rigid`)
- `--metric`: Similarity metric (`mutual_information`, `mean_squares`, `correlation`, default: `mutual_information`)
- `--optimizer`: Optimizer for registration (`gradient_descent`, `lbfgs`, default: `gradient_descent`)
- `--learning-rate`: Learning rate for optimizer (default: 0.1)
- `--iterations`: Maximum number of iterations (default: 100)
- `--save-transforms`: Save transformation files
- `--output-json`: Output JSON file for registration metrics
- `--verbose`: Enable verbose output

### Dataset Preparation

Create train/val/test splits for paired MRI data:

```bash
python create_dataset_splits.py --domain-a-dir /path/to/domain-a --domain-b-dir /path/to/domain-b --output-dir /path/to/output
```

#### Options

- `--domain-a-dir`: Directory with domain A volumes (required)
- `--domain-b-dir`: Directory with domain B volumes (required)
- `--output-dir`: Output directory for split files (required)
- `--train-ratio`: Fraction of data for training (default: 0.8)
- `--val-ratio`: Fraction of data for validation (default: 0.1)
- `--test-ratio`: Fraction of data for testing (default: 0.1)
- `--file-pattern`: Glob pattern for input files (default: `*.nii.gz`)
- `--paired`: Whether the data is paired (same filenames in both domains, default: True)
- `--unpaired`: If provided, treat data as unpaired
- `--seed`: Random seed for reproducibility (default: 42)
- `--verbose`: Enable verbose output
