#!/bin/bash
# cluster setup script for extension a (patchnce) experiments
# run this on a fresh chameleon cloud instance after ssh-ing in
#
# prerequisites: instance has nvidia gpu with cuda drivers installed
# usage: bash setup_cluster.sh

set -e

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

log "checking gpu..."
nvidia-smi

# =========================================================================
# 1. system packages
# =========================================================================
log "installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-venv python3-pip tmux htop

# =========================================================================
# 2. data disk setup (mount /dev/sdb or /dev/vdb to /data)
# =========================================================================
log "setting up data disk..."
DATA_DISK=""
for disk in /dev/sdb /dev/vdb /dev/nvme1n1; do
    if [ -b "$disk" ]; then
        DATA_DISK="$disk"
        break
    fi
done

if [ -n "$DATA_DISK" ] && ! mountpoint -q /data; then
    log "formatting and mounting $DATA_DISK -> /data"
    sudo mkfs.ext4 -q "$DATA_DISK" || true
    sudo mkdir -p /data
    sudo mount "$DATA_DISK" /data
    sudo chown cc:cc /data
else
    log "using existing /data mount or creating directory"
    sudo mkdir -p /data
    sudo chown cc:cc /data
fi

mkdir -p /data/experiments/journal_extension /data/experiments/logs /data/preprocessed

# =========================================================================
# 3. clone repo and setup venv
# =========================================================================
log "cloning repo..."
if [ ! -d /home/cc/neuroscope ]; then
    git clone https://github.com/ishrith-gowda/SA-CycleGAN-2.5D.git /home/cc/neuroscope
else
    cd /home/cc/neuroscope && git pull
fi

log "creating virtual environment..."
if [ ! -d /home/cc/neuroscope-env ]; then
    python3 -m venv /home/cc/neuroscope-env
fi
source /home/cc/neuroscope-env/bin/activate

log "installing pytorch and dependencies..."
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -q numpy scipy scikit-learn nibabel tqdm tensorboard pyyaml pillow

# =========================================================================
# 4. symlink preprocessed data
# =========================================================================
log "setting up data symlinks..."
cd /home/cc/neuroscope
ln -sfn /data/preprocessed preprocessed

# =========================================================================
# 5. verify setup
# =========================================================================
log "verifying setup..."
python3 -c "
import torch
print(f'pytorch {torch.__version__}')
print(f'cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'gpu: {torch.cuda.get_device_name(0)}')
    print(f'vram: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} gb')
"

log "========================================="
log " cluster setup complete"
log " "
log " next steps:"
log "   1. copy preprocessed data to /data/preprocessed/{brats,upenn}"
log "      scp -r preprocessed/brats cc@<ip>:/data/preprocessed/"
log "      scp -r preprocessed/upenn cc@<ip>:/data/preprocessed/"
log "   2. start training in tmux:"
log "      tmux new -s training"
log "      source /home/cc/neuroscope-env/bin/activate"
log "      cd /home/cc/neuroscope"
log "      bash journal_extension/scripts/run_patchnce_only.sh"
log "========================================="
