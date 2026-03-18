#!/bin/bash
# chameleon cloud gpu cluster setup script
# run once after cloning the repo on the cluster

set -e

echo "setting up neuroscope on chameleon cloud..."

# create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install --upgrade pip
pip install -r requirements-cluster.txt

# create data directories
mkdir -p /home/cc/neuroscope/preprocessed/brats
mkdir -p /home/cc/neuroscope/preprocessed/upenn
mkdir -p /home/cc/neuroscope/preprocessed/brats_siemens
mkdir -p /home/cc/neuroscope/preprocessed/brats_ge
mkdir -p /home/cc/neuroscope/preprocessed/brats_philips

# create experiment output directories
mkdir -p /home/cc/neuroscope/experiments/journal_extension/patchnce_hybrid
mkdir -p /home/cc/neuroscope/experiments/journal_extension/compression
mkdir -p /home/cc/neuroscope/experiments/journal_extension/multi_domain
mkdir -p /home/cc/neuroscope/experiments/journal_extension/federated

# create tensorboard log directory
mkdir -p /home/cc/neuroscope/runs

# verify gpu availability
python3 -c "
import torch
print(f'pytorch version: {torch.__version__}')
print(f'cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'gpu count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  gpu {i}: {torch.cuda.get_device_name(i)}')
        mem = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f'    memory: {mem:.1f} gb')
else:
    print('warning: no gpu detected')
"

echo ""
echo "setup complete. activate environment with: source .venv/bin/activate"
echo "start training with: cd journal_extension/scripts && python train_hybrid_nce.py --config ../configs/patchnce_hybrid.yaml"
