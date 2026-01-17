#!/bin/bash
# =============================================================================
# chameleon cloud environment setup
# =============================================================================
# run this first to set up the training environment on chameleon
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# configuration
# -----------------------------------------------------------------------------
PROJECT_DIR="/home/cc/neuroscope"
DATA_DIR="${PROJECT_DIR}/data"
CONDA_ENV="neuroscope"

# -----------------------------------------------------------------------------
# system setup
# -----------------------------------------------------------------------------
echo "[setup] updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git \
    htop \
    tmux \
    screen \
    unzip \
    wget \
    curl

# -----------------------------------------------------------------------------
# miniconda installation (if not present)
# -----------------------------------------------------------------------------
if [ ! -d ~/miniconda3 ]; then
    echo "[setup] installing miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p ~/miniconda3
    rm ~/miniconda.sh
    ~/miniconda3/bin/conda init bash
    source ~/.bashrc
fi

# -----------------------------------------------------------------------------
# conda environment
# -----------------------------------------------------------------------------
source ~/miniconda3/etc/profile.d/conda.sh

if ! conda env list | grep -q ${CONDA_ENV}; then
    echo "[setup] creating conda environment..."
    conda create -n ${CONDA_ENV} python=3.10 -y
fi

conda activate ${CONDA_ENV}

# -----------------------------------------------------------------------------
# pytorch with cuda
# -----------------------------------------------------------------------------
echo "[setup] installing pytorch with cuda..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# -----------------------------------------------------------------------------
# project dependencies
# -----------------------------------------------------------------------------
echo "[setup] installing project dependencies..."
cd ${PROJECT_DIR}

pip install -r requirements.txt 2>/dev/null || pip install \
    numpy \
    scipy \
    scikit-learn \
    scikit-image \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    pyyaml \
    tensorboard \
    nibabel \
    SimpleITK \
    monai \
    lpips \
    pytorch-fid

# -----------------------------------------------------------------------------
# verify installation
# -----------------------------------------------------------------------------
echo "[setup] verifying installation..."

python -c "
import torch
print(f'pytorch version: {torch.__version__}')
print(f'cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'cuda version: {torch.version.cuda}')
    print(f'gpu device: {torch.cuda.get_device_name(0)}')
    print(f'gpu memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} gb')
"

# -----------------------------------------------------------------------------
# directory structure
# -----------------------------------------------------------------------------
echo "[setup] creating directory structure..."
mkdir -p ${DATA_DIR}/preprocessed/brats
mkdir -p ${DATA_DIR}/preprocessed/upenn
mkdir -p ${PROJECT_DIR}/experiments
mkdir -p ${PROJECT_DIR}/logs
mkdir -p ${PROJECT_DIR}/checkpoints

# -----------------------------------------------------------------------------
# done
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "chameleon setup complete!"
echo "=============================================="
echo ""
echo "next steps:"
echo "  1. upload preprocessed data to:"
echo "     - ${DATA_DIR}/preprocessed/brats/"
echo "     - ${DATA_DIR}/preprocessed/upenn/"
echo ""
echo "  2. start training:"
echo "     cd ${PROJECT_DIR}"
echo "     ./scripts/chameleon/train_on_chameleon.sh baseline"
echo ""
echo "  3. monitor with:"
echo "     nvidia-smi -l 1"
echo "     tail -f ${PROJECT_DIR}/logs/*.log"
echo "=============================================="
