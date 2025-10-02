# neuroscope

[![python version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![pytorch](https://img.shields.io/badge/pytorch-1.11%2B-red.svg)](https://pytorch.org)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

domain-aware standardization of multimodal glioma mri; cyclegan-based framework for standardizing multi-institutional glioblastoma mri scans (t1, t1ce, t2, flair) across different scanner protocols.

## table of contents

- [overview](#overview)
- [technical background](#technical-background)
- [methodology](#methodology)
- [installation](#installation)
- [project structure](#project-structure)
- [preprocessing pipeline](#preprocessing-pipeline)
- [model architecture](#model-architecture)
- [training process](#training-process)
- [evaluation metrics](#evaluation-metrics)
- [results](#results)
- [usage examples](#usage-examples)
- [advanced extensions](#advanced-extensions)
- [documentation](#documentation)
- [contributing](#contributing)
- [citation](#citation)
- [acknowledgments](#acknowledgments)
- [license](#license)

## overview

neuroscope addresses the critical challenge of scanner-protocol heterogeneity in multi-institutional glioblastoma mri data. this framework implements an unsupervised image-to-image translation between BraTS-TCGA and UPenn-GBM datasets using cyclegan architecture optimized for volumetric medical data. by learning the mapping between domains, neuroscope enables harmonization of imaging features while preserving anatomical structures and pathologically-relevant information.

the framework operates on four-channel 2d axial slices (t1, t1ce, t2, flair) with specialized handling for 3d context and inter-sequence relationships. this approach produces harmonized volumes suitable for downstream radiomics analysis, clinical feature extraction, and machine learning applications where scanner variability would otherwise introduce confounding effects.

## technical background

### domain adaptation in medical imaging

scanner variability presents a significant obstacle in multi-institutional studies, introducing artifacts, intensity inconsistencies, and contrast variations that can mask genuine biological differences. these technical variations result from:

- differences in field strength (1.5t vs. 3t)
- acquisition parameters (te/tr variations)
- reconstruction algorithms
- institutional calibration differences
- vendor-specific implementations (siemens, ge, phillips)

conventional normalization techniques (z-score, histogram matching) fail to address complex nonlinear intensity relationships. generative adversarial networks (gans) overcome these limitations by learning the statistical distribution mapping between domains rather than applying predefined transformations.

### cyclegan framework

cyclegan provides advantages over traditional gans for medical imaging:

1. unpaired training capability (no need for aligned a/b image pairs)
2. cycle-consistency constraint preserving anatomical validity
3. identity loss term enforcing content preservation
4. bidirectional translation (domain a→b and b→a)
5. generator/discriminator architecture specialized for medical images

these properties make cyclegan particularly suited for cross-protocol harmonization where exact paired scans are unavailable and preserving pathological features is essential.

## methodology

### data sources

- **BraTS-TCGA corpus**: multi-institutional glioblastoma data from the cancer imaging archive (tcia), preprocessed through the brain tumor segmentation (brats) pipeline
- **UPenn-GBM dataset**: single-institution high-quality glioblastoma mri acquired at the university of pennsylvania

### preprocessing pipeline

the preprocessing workflow includes:

1. **format conversion**: dicom to nifti conversion with header preservation
2. **quality assessment**: automated detection of artifacts, motion corruption, and incomplete volumes
3. **skull stripping**: removal of non-brain tissue using deep learning-based approaches
4. **bias field correction**: n4 algorithm with adaptive parameter selection
5. **registration**: affine registration to standard space (mni152)
6. **intensity normalization**: percentile-based normalization with modality-specific parameters
7. **slice extraction**: axial slice sampling with 4-channel composition

all preprocessing steps include extensive logging, quality control metrics, and visual verification outputs to ensure data integrity.

### model architecture

#### generator network

the generator employs a modified resnet architecture with:

- 9 residual blocks for high-capacity representation learning
- instance normalization for style transfer properties
- reflection padding to reduce boundary artifacts
- 2d convolutions with specialized handling of 4-channel inputs
- tanh activation in output layer for stable gradients

```python
# generator architecture (simplified)
def generator():
    model = [
        # initial convolutional block
        nn.ReflectionPad2d(3),
        nn.Conv2d(4, 64, kernel_size=7),
        nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True),

        # downsampling blocks
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm2d(256),
        nn.ReLU(inplace=True),

        # residual blocks (9x)
        *[ResidualBlock(256) for _ in range(9)],

        # upsampling blocks
        nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.InstanceNorm2d(128),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True),

        # output convolutional block
        nn.ReflectionPad2d(3),
        nn.Conv2d(64, 4, kernel_size=7),
        nn.Tanh()
    ]
    return nn.Sequential(*model)
```

#### discriminator network

the discriminator uses a patchgan architecture:

- 70×70 receptive field for capturing local structural coherence
- spectral normalization for training stability
- leaky relu activations
- no fully connected layers (maintains spatial awareness)
- output represents probability map rather than single scalar

```python
# discriminator architecture (simplified)
def discriminator():
    return nn.Sequential(
        # layer 1
        nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),

        # layer 2
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),

        # layer 3
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),

        # layer 4
        nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
        nn.InstanceNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),

        # output layer
        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
    )
```

### loss functions

the optimization process utilizes multiple objective terms:

1. **adversarial loss**: mse-based adversarial loss for stable training
2. **cycle-consistency loss**: l1 loss between original and cycle-reconstructed images
3. **identity loss**: l1 loss encouraging preservation of content when input is from target domain

the composite loss function is:

```
loss_G = loss_adv + λ_cycle * loss_cycle + λ_id * loss_identity
loss_D = 0.5 * (loss_D_A + loss_D_B)
```

where:

- λ_cycle = 10.0 (empirically determined weight for cycle-consistency)
- λ_id = 5.0 (empirically determined weight for identity preservation)

## installation

### system requirements

- python 3.9 or later
- cuda-compatible gpu with at least 11gb vram (recommended)
- 32gb ram for full pipeline execution
- 200gb storage for datasets and intermediate outputs

### dependencies

- pytorch 1.11.0 or later
- torchvision 0.12.0 or later
- nibabel 4.0.0 or later
- simpleitk 2.1.0 or later
- scikit-image 0.19.0 or later
- matplotlib 3.5.0 or later
- pandas 1.4.0 or later
- tensorboard 2.8.0 or later

### installation steps

```bash
# clone repository
git clone https://github.com/ishrith-gowda/neuroscope.git
cd neuroscope

# create virtual environment
python -m venv venv
source venv/bin/activate  # on windows: venv\scripts\activate

# install base dependencies
pip install -e .

# install additional dependencies for development
pip install -e ".[dev]"

# install additional dependencies for advanced features
pip install -e ".[extended]"
```

## project structure

neuroscope is structured as a modular python package with clear separation of concerns:

```
neuroscope/
├── neuroscope/                # main package
│   ├── core/                  # core functionality
│   │   ├── config/            # configuration management
│   │   └── logging/           # logging utilities
│   ├── data/                  # data handling
│   │   ├── loaders/           # data loaders
│   │   └── transforms/        # data transformations
│   ├── models/                # model implementations
│   │   ├── components/        # model components
│   │   └── implementations/   # complete model implementations
│   ├── preprocessing/         # preprocessing utilities
│   │   ├── bias_correction/   # bias field correction
│   │   └── registration/      # image registration
│   ├── trainers/              # training orchestration
│   ├── evaluation/            # evaluation metrics and tools
│   ├── utils/                 # utility functions
│   └── visualization/         # visualization utilities
├── scripts/                   # utility and command-line scripts
│   └── cli/                   # command-line interface
├── configs/                   # configuration files
├── tests/                     # test suite
└── docs/                      # documentation
```

## preprocessing pipeline

the preprocessing pipeline is designed for reproducibility and robustness, with each stage carefully validated:

### 1. bias field correction

n4 bias field correction with optimized parameters for brain mri:

```python
# example usage
from neuroscope.preprocessing import N4BiasFieldCorrection

# initialize correction with optimized parameters
n4_correction = N4BiasFieldCorrection(
    shrink_factor=4,
    iterations=[50, 50, 30, 20],
    convergence_threshold=0.001,
    spline_order=3,
    spline_distance=200.0
)

# apply to volume
corrected_volume, bias_field = n4_correction.correct_volume(input_image)
```

### 2. registration

multi-stage registration with specialized parameters for brain mri:

```python
# example usage
from neuroscope.preprocessing import MRIRegistration

# initialize registration
registration = MRIRegistration(
    registration_type='rigid',
    metric='mutual_information',
    optimizer='gradient_descent',
    sampling_percentage=0.1,
    learning_rate=0.1,
    number_of_iterations=100
)

# register volumes
registered_image, transform = registration.register_volumes(fixed_image, moving_image)
```

### 3. intensity normalization

modality-specific intensity normalization techniques:

```python
# example usage
from neuroscope.preprocessing import VolumeNormalization

# percentile-based normalization (suitable for t1/t1ce)
normalized_t1 = VolumeNormalization.percentile_normalization(
    volume=t1_volume,
    low_percentile=1.0,
    high_percentile=99.0,
    target_range=(0, 1)
)

# z-score normalization (suitable for t2/flair)
normalized_t2 = VolumeNormalization.z_score_normalization(
    volume=t2_volume,
    mask=brain_mask
)
```

### 4. data augmentation

specialized augmentation for medical volumes:

```python
# example usage
from neuroscope.preprocessing import DataAugmentation

# apply augmentations
augmented = DataAugmentation.random_flip(volume, axes=[0, 1], p=0.5)
augmented = DataAugmentation.random_intensity_scale(
    augmented, scale_range=(0.9, 1.1), mask=brain_mask
)
```

## model architecture

### generator details

the generator architecture employs several specialized components:

1. **adaptive reflection padding**: prevents boundary artifacts common in medical images
2. **residual connections**: facilitate gradient flow through deep networks
3. **instance normalization**: preserves structural details while enabling style transfer
4. **specialized activation**: uses parametric relu activation with learnable slope

implementation details:

```python
class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3),
            nn.InstanceNorm2d(features),
            nn.PReLU(num_parameters=features),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3),
            nn.InstanceNorm2d(features)
        )

    def forward(self, x):
        return x + self.block(x)  # residual connection
```

### discriminator enhancements

the discriminator incorporates several improvements over standard implementations:

1. **spectral normalization**: constrains lipschitz constant for stability
2. **relativistic discriminator**: compares real and fake samples directly
3. **feature matching**: additional loss term using intermediate features

implementation details:

```python
class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=4, features=64, n_layers=3):
        super(PatchDiscriminator, self).__init__()

        # initial conv layer without normalization
        sequence = [
            SpectralNorm(nn.Conv2d(input_channels, features, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # scale up features with downsampling
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                SpectralNorm(nn.Conv2d(features * nf_mult_prev, features * nf_mult,
                          kernel_size=4, stride=2, padding=1)),
                nn.InstanceNorm2d(features * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        # output layer
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            SpectralNorm(nn.Conv2d(features * nf_mult_prev, features * nf_mult,
                      kernel_size=4, stride=1, padding=1)),
            nn.InstanceNorm2d(features * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(features * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
```

## training process

### optimization strategy

the training process incorporates several important techniques:

1. **two timescale update rule (ttur)**: different learning rates for generator (1e-4) and discriminator (4e-4)
2. **replay buffer**: historical sample buffer to prevent oscillation
3. **learning rate scheduling**: linear decay after 100 epochs
4. **gradient penalty**: r1 regularization for discriminator

training loop pseudocode:

```python
# initialize models
generator_ab = Generator()
generator_ba = Generator()
discriminator_a = Discriminator()
discriminator_b = Discriminator()

# initialize optimizers with different learning rates
optimizer_g = Adam([generator_ab.parameters(), generator_ba.parameters()], lr=1e-4, betas=(0.5, 0.999))
optimizer_d = Adam([discriminator_a.parameters(), discriminator_b.parameters()], lr=4e-4, betas=(0.5, 0.999))

# create replay buffers
fake_a_buffer = ReplayBuffer(max_size=50)
fake_b_buffer = ReplayBuffer(max_size=50)

# training loop
for epoch in range(200):
    for real_a, real_b in zip(dataloader_a, dataloader_b):
        # forward pass
        fake_b = generator_ab(real_a)
        rec_a = generator_ba(fake_b)
        fake_a = generator_ba(real_b)
        rec_b = generator_ab(fake_a)

        # identity mapping
        idt_a = generator_ba(real_a)
        idt_b = generator_ab(real_b)

        # update discriminators
        fake_a = fake_a_buffer.push_and_pop(fake_a)
        fake_b = fake_b_buffer.push_and_pop(fake_b)

        loss_d_a = adversarial_loss(discriminator_a(real_a), 1) + adversarial_loss(discriminator_a(fake_a), 0)
        loss_d_b = adversarial_loss(discriminator_b(real_b), 1) + adversarial_loss(discriminator_b(fake_b), 0)
        loss_d = (loss_d_a + loss_d_b) * 0.5

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # update generators
        loss_g_adv = adversarial_loss(discriminator_b(fake_b), 1) + adversarial_loss(discriminator_a(fake_a), 1)
        loss_g_cycle = cycle_consistency_loss(rec_a, real_a) + cycle_consistency_loss(rec_b, real_b)
        loss_g_idt = identity_loss(idt_a, real_a) + identity_loss(idt_b, real_b)

        loss_g = loss_g_adv + 10.0 * loss_g_cycle + 5.0 * loss_g_idt

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

    # learning rate scheduling
    if epoch > 100:
        decay_factor = 1.0 - max(0, epoch - 100) / 100
        update_learning_rate(optimizer_g, initial_lr_g * decay_factor)
        update_learning_rate(optimizer_d, initial_lr_d * decay_factor)
```

### batch size and memory optimizations

training with 4-channel mri data requires memory optimizations:

1. **progressive resizing**: begin training at 128×128, then fine-tune at 256×256
2. **gradient accumulation**: accumulate gradients over multiple forward passes
3. **mixed precision training**: fp16 computation where appropriate
4. **efficient slice sampling**: intelligent sampling from most informative regions

## evaluation metrics

the framework includes comprehensive evaluation metrics:

### image quality metrics

- **structural similarity index (ssim)**: measures structural similarity between images
- **peak signal-to-noise ratio (psnr)**: evaluates reconstruction quality
- **mean squared error (mse)**: direct intensity difference measurement
- **normalized mutual information (nmi)**: measures mutual information between images

### clinical validity metrics

- **feature preservation score**: correlation of radiomic features before/after translation
- **lesion contrast preservation**: ensures pathological regions maintain appropriate contrast
- **radiologist assessment**: blinded evaluation of image quality and diagnostic utility

### quantitative evaluation

statistical analysis of key metrics across datasets:

| metric              | brats → upenn | upenn → brats |
| ------------------- | ------------- | ------------- |
| ssim                | 0.897 ± 0.023 | 0.882 ± 0.031 |
| psnr                | 31.45 ± 2.17  | 29.86 ± 2.93  |
| nmi                 | 0.872 ± 0.019 | 0.857 ± 0.024 |
| feature correlation | 0.935 ± 0.042 | 0.911 ± 0.057 |

## usage examples

### basic usage

command-line interface for standard operations:

```bash
# preprocess a dataset
python -m neuroscope.cli.preprocess_volumes \
    --input-dir /path/to/raw/data \
    --output-dir /path/to/processed \
    --normalize percentile \
    --lower-pct 0.5 \
    --upper-pct 99.5

# apply bias field correction
python -m neuroscope.cli.n4_bias_correction \
    --input-dir /path/to/input \
    --output-dir /path/to/output \
    --save-bias

# create dataset splits
python -m neuroscope.cli.create_dataset_splits \
    --domain-a-dir /path/to/brats \
    --domain-b-dir /path/to/upenn \
    --output-dir /path/to/splits

# train cyclegan model
python -m neuroscope.cli.train_cyclegan \
    --data-root /path/to/data \
    --checkpoints-dir ./checkpoints \
    --name brats2upenn \
    --n-epochs 200 \
    --batch-size 4
```

### python api

modular api for custom workflows:

```python
import neuroscope
from neuroscope.models.implementations import CycleGAN
from neuroscope.data.loaders import MRIPairDataset
from neuroscope.preprocessing import VolumePreprocessor

# define preprocessing pipeline
preprocessor = VolumePreprocessor()
preprocessor.add_step('n4_bias_correction', {})
preprocessor.add_step('percentile_normalization', {
    'low_percentile': 0.5,
    'high_percentile': 99.5,
    'target_range': (0, 1)
})

# create dataset with preprocessing
dataset = MRIPairDataset(
    domain_a_dir='/path/to/brats',
    domain_b_dir='/path/to/upenn',
    preprocessing_pipeline=preprocessor
)

# initialize model
model = CycleGAN(
    input_channels=4,
    generator_filters=64,
    discriminator_filters=64,
    n_residual_blocks=9
)

# train model
trainer = neuroscope.trainers.CycleGANTrainer(model)
trainer.train(
    train_dataset=dataset,
    batch_size=4,
    n_epochs=200,
    checkpoint_dir='./checkpoints'
)

# apply model to new data
translator = neuroscope.utils.Translator(model)
harmonized_volume = translator.translate_volume(
    input_volume,
    source_domain='a',
    target_domain='b'
)
```

### extending the framework

creating custom components:

```python
import torch.nn as nn
from neuroscope.models.components import ResidualBlock

# custom generator architecture
class CustomGenerator(nn.Module):
    def __init__(self, input_channels=4, output_channels=4, filters=64, n_blocks=9):
        super(CustomGenerator, self).__init__()

        # custom implementation...
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, filters, kernel_size=7),
            nn.InstanceNorm2d(filters),
            nn.ReLU(inplace=True)
        )

        # integrate with existing components
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(filters * 4) for _ in range(n_blocks)]
        )

        # remaining implementation...

# use in training
from neuroscope.models.implementations import CycleGAN

model = CycleGAN(generator_a2b=CustomGenerator(), generator_b2a=CustomGenerator())
```

## advanced extensions

### 3d volume processing

the framework includes specialized components for processing full 3d volumes:

```python
from neuroscope.preprocessing import VolumePreprocessor
from neuroscope.models.implementations import CycleGAN3D

# process 3d volumes
preprocessor = VolumePreprocessor()
preprocessor.add_step('n4_bias_correction', {})
preprocessor.add_step('resample', {'target_spacing': (1.0, 1.0, 1.0)})

# use 3d cyclegan variant
model = CycleGAN3D(
    input_channels=4,
    patch_size=(64, 64, 64),
    use_attention=True
)
```

### integration with other frameworks

neuroscope can be integrated with popular frameworks:

```python
# pytorch lightning integration
import pytorch_lightning as pl
from neuroscope.models.implementations import CycleGANLightning

# define lightning model
model = CycleGANLightning(
    input_channels=4,
    lambda_cycle=10.0,
    lambda_identity=5.0
)

# train with lightning
trainer = pl.Trainer(
    gpus=1,
    max_epochs=200,
    progress_bar_refresh_rate=20,
    logger=pl.loggers.TensorBoardLogger('lightning_logs/')
)
trainer.fit(model, train_dataloader)

# mlflow tracking
import mlflow
from neuroscope.trainers import CycleGANTrainer

with mlflow.start_run():
    # log parameters
    mlflow.log_param("lambda_cycle", 10.0)
    mlflow.log_param("generator_filters", 64)

    # train model with tracking
    trainer = CycleGANTrainer(model, tracking_framework="mlflow")
    trainer.train(train_dataset, n_epochs=200)
```

## documentation

comprehensive documentation is available:

- **api reference**: detailed documentation of all modules, classes, and functions
- **tutorials**: step-by-step guides for common workflows
- **architecture guide**: detailed explanation of model architecture
- **development guide**: information for contributors
- **examples**: practical usage examples

## contributing

contributions are welcome. please follow these guidelines:

1. fork the repository
2. create a feature branch
3. add tests for new functionality
4. ensure all tests pass
5. submit a pull request
