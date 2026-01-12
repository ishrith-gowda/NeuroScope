#!/bin/bash

# automated chameleon cloud instance setup script
# this script automates the setup process for a new rtx 6000 instance

set -e  # exit on error

# colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # no color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}neuroscope rtx 6000 setup script${NC}"
echo -e "${GREEN}================================${NC}"
echo

# get floating ip from user
read -p "enter the floating ip address of your new instance: " SERVER_IP

if [ -z "$SERVER_IP" ]; then
    echo -e "${RED}error: ip address cannot be empty${NC}"
    exit 1
fi

SSH_KEY="$HOME/Downloads/neuroscope-key.pem"
SSH_CMD="ssh -i $SSH_KEY -o StrictHostKeyChecking=no cc@$SERVER_IP"

echo -e "${YELLOW}using ip: $SERVER_IP${NC}"
echo -e "${YELLOW}testing connection...${NC}"

# test ssh connection
if ! $SSH_CMD "echo 'connection successful'" > /dev/null 2>&1; then
    echo -e "${RED}error: cannot connect to server${NC}"
    exit 1
fi

echo -e "${GREEN}✓ connection successful${NC}"
echo

# step 1: security verification
echo -e "${YELLOW}step 1: verifying security settings...${NC}"
$SSH_CMD "sudo grep 'PasswordAuthentication no' /etc/ssh/sshd_config" > /dev/null && echo -e "${GREEN}✓ ssh password auth disabled${NC}"
echo

# step 2: system update & cuda installation
echo -e "${YELLOW}step 2: installing cuda and system updates (this takes ~10 minutes)...${NC}"
$SSH_CMD << 'EOF'
set -e
sudo apt-get update -y > /dev/null 2>&1
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y > /dev/null 2>&1

# install cuda if not present
if ! command -v nvcc &> /dev/null; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb > /dev/null 2>&1
    sudo apt-get update -y > /dev/null 2>&1
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-12-1 > /dev/null 2>&1

    # set cuda paths
    echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi

echo "cuda installation complete"
EOF
echo -e "${GREEN}✓ cuda installed${NC}"
echo

# step 3: install python & pytorch
echo -e "${YELLOW}step 3: installing python and pytorch (this takes ~5 minutes)...${NC}"
$SSH_CMD << 'EOF'
set -e
sudo apt-get install -y python3-pip python3-dev python3-venv > /dev/null 2>&1

# create virtual environment
python3 -m venv ~/neuroscope_env
source ~/neuroscope_env/bin/activate

# install pytorch
pip install --upgrade pip > /dev/null 2>&1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 > /dev/null 2>&1

# install other dependencies
pip install numpy scipy matplotlib tensorboard pillow pyyaml tqdm nibabel > /dev/null 2>&1
pip install scikit-image scikit-learn pandas h5py > /dev/null 2>&1

# verify
python3 -c "import torch; assert torch.cuda.is_available(), 'cuda not available'"
echo "pytorch with cuda installed successfully"
EOF
echo -e "${GREEN}✓ pytorch installed${NC}"
echo

# step 4: transfer code
echo -e "${YELLOW}step 4: transferring neuroscope code...${NC}"
rsync -az --progress \
  --exclude='experiments/' \
  --exclude='experiments_from_gpu/' \
  --exclude='*.pth' \
  --exclude='*.pt' \
  --exclude='__pycache__/' \
  --exclude='.git/' \
  --exclude='*.pyc' \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  "/Volumes/usb drive/neuroscope/" cc@$SERVER_IP:~/neuroscope/
echo -e "${GREEN}✓ code transferred${NC}"
echo

# step 5: transfer data
echo -e "${YELLOW}step 5: transferring preprocessed data (this takes ~20 minutes for 7GB)...${NC}"
echo "  transferring brats dataset..."
rsync -az --progress \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  "/Volumes/usb drive/neuroscope/preprocessed/brats/" \
  cc@$SERVER_IP:~/neuroscope/preprocessed/brats/ 2>&1 | tail -1

echo "  transferring upenn dataset..."
rsync -az --progress \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  "/Volumes/usb drive/neuroscope/preprocessed/upenn/" \
  cc@$SERVER_IP:~/neuroscope/preprocessed/upenn/ 2>&1 | tail -1
echo -e "${GREEN}✓ data transferred${NC}"
echo

# step 6: transfer checkpoint
echo -e "${YELLOW}step 6: transferring checkpoint from epoch 2...${NC}"
rsync -az --progress \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  "/Volumes/usb drive/neuroscope/experiments_from_gpu/sa_cyclegan_25d_rtx6000_fast_20260106_230201/checkpoints/checkpoint_latest.pth" \
  cc@$SERVER_IP:~/neuroscope/checkpoint_epoch2.pth
echo -e "${GREEN}✓ checkpoint transferred (402 MB)${NC}"
echo

# step 7: install neuroscope package
echo -e "${YELLOW}step 7: installing neuroscope package...${NC}"
$SSH_CMD << 'EOF'
cd ~/neuroscope
source ~/neuroscope_env/bin/activate
pip install -e . > /dev/null 2>&1
python3 -c "import neuroscope; print('neuroscope installed')"
EOF
echo -e "${GREEN}✓ neuroscope installed${NC}"
echo

# step 8: create optimized config with dataloader optimizations
echo -e "${YELLOW}step 8: creating optimized training config...${NC}"
$SSH_CMD << 'EOF'
cat > ~/neuroscope/config_rtx6000_resume.yaml << 'CONFIG'
experiment_name: sa_cyclegan_25d_rtx6000_resume
seed: 42
deterministic: false
brats_dir: /home/cc/neuroscope/preprocessed/brats
upenn_dir: /home/cc/neuroscope/preprocessed/upenn
output_dir: /home/cc/neuroscope/experiments
ngf: 64
ndf: 64
n_residual_blocks: 9
use_attention: true
use_cbam: true
input_channels: 12
output_channels: 4
epochs: 100
batch_size: 16
num_workers: 32
lr_G: 0.00005
lr_D: 0.00005
beta1: 0.5
beta2: 0.999
weight_decay: 0.0
scheduler_type: cosine
warmup_epochs: 5
min_lr: 0.000001
lambda_cycle: 10.0
lambda_identity: 5.0
lambda_ssim: 1.0
lambda_gradient: 1.0
gradient_clip_norm: 1.0
use_amp: true
validate_every: 5
save_every: 5
save_best_only: false
early_stopping: false
patience: 20
min_delta: 0.001
log_every_n_steps: 100
sample_every: 1000
figure_every: 100
verbose: 1
resume_from_checkpoint: /home/cc/neuroscope/checkpoint_epoch2.pth
CONFIG
EOF
echo -e "${GREEN}✓ config created${NC}"
echo

# step 9: verify data
echo -e "${YELLOW}step 9: verifying data transfer...${NC}"
$SSH_CMD << 'EOF'
BRATS_SIZE=$(du -sh ~/neuroscope/preprocessed/brats/ 2>/dev/null | cut -f1)
UPENN_SIZE=$(du -sh ~/neuroscope/preprocessed/upenn/ 2>/dev/null | cut -f1)
BRATS_FILES=$(find ~/neuroscope/preprocessed/brats -name "*.nii.gz" 2>/dev/null | wc -l)
UPENN_FILES=$(find ~/neuroscope/preprocessed/upenn -name "*.nii.gz" 2>/dev/null | wc -l)

echo "brats: $BRATS_SIZE ($BRATS_FILES files)"
echo "upenn: $UPENN_SIZE ($UPENN_FILES files)"
EOF
echo -e "${GREEN}✓ data verified${NC}"
echo

# step 10: start training
echo -e "${YELLOW}step 10: starting training...${NC}"
$SSH_CMD << 'EOF'
cd ~/neuroscope
source ~/neuroscope_env/bin/activate
nohup bash -c 'yes | python3 -u scripts/02_training/train_comprehensive.py --config ~/neuroscope/config_rtx6000_resume.yaml' > ~/neuroscope/training_resume.log 2>&1 < /dev/null &
echo "training process started in background"
sleep 3
tail -20 ~/neuroscope/training_resume.log
EOF
echo -e "${GREEN}✓ training started${NC}"
echo

# step 11: setup auto-sync
echo -e "${YELLOW}step 11: setting up automatic checkpoint sync...${NC}"

# create updated sync script
cat > "/Volumes/usb drive/neuroscope/auto_sync_checkpoints_new.sh" << SYNCSCRIPT
#!/bin/bash
SERVER="$SERVER_IP"
SSH_KEY="\$HOME/Downloads/neuroscope-key.pem"
REMOTE_DIR="cc@\${SERVER}:~/neuroscope/experiments/"
LOCAL_DIR="/Volumes/usb drive/neuroscope/experiments_from_gpu/"
LOG_FILE="/Volumes/usb drive/neuroscope/logs/sync.log"

mkdir -p "\$(dirname "\$LOG_FILE")"
mkdir -p "\$LOCAL_DIR"

echo "[\$(date)] starting automatic checkpoint sync..." >> "\$LOG_FILE"
COUNTER=1

while true; do
    echo "[\$(date)] sync #\${COUNTER} starting..." >> "\$LOG_FILE"

    rsync -avz --progress \
        --include="*/" \
        --include="*.pth" \
        --include="*.pt" \
        --include="*.yaml" \
        --include="*.log" \
        --include="*.txt" \
        --include="*.png" \
        --include="*.jpg" \
        --exclude="*" \
        -e "ssh -i \$SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10" \
        "\$REMOTE_DIR" \
        "\$LOCAL_DIR" >> "\$LOG_FILE" 2>&1

    EXIT_CODE=\$?

    if [ \$EXIT_CODE -eq 0 ]; then
        echo "[\$(date)] sync #\${COUNTER} completed successfully" >> "\$LOG_FILE"
    else
        echo "[\$(date)] sync #\${COUNTER} failed with exit code: \$EXIT_CODE" >> "\$LOG_FILE"
    fi

    COUNTER=\$((COUNTER + 1))
    sleep 600
done
SYNCSCRIPT

chmod +x "/Volumes/usb drive/neuroscope/auto_sync_checkpoints_new.sh"

# kill old sync processes
pkill -f "auto_sync_checkpoints" 2>/dev/null || true

# start new sync
nohup bash "/Volumes/usb drive/neuroscope/auto_sync_checkpoints_new.sh" > /dev/null 2>&1 &
echo -e "${GREEN}✓ auto-sync started (syncs every 10 minutes)${NC}"
echo

# final summary
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}setup complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo
echo -e "${YELLOW}server ip:${NC} $SERVER_IP"
echo -e "${YELLOW}training status:${NC} resuming from epoch 2"
echo -e "${YELLOW}expected completion:${NC} 3.4 days (98 epochs remaining)"
echo
echo -e "${YELLOW}monitor training:${NC}"
echo "  ssh -i ~/Downloads/neuroscope-key.pem cc@$SERVER_IP 'tail -f ~/neuroscope/training_resume.log'"
echo
echo -e "${YELLOW}check gpu utilization:${NC}"
echo "  ssh -i ~/Downloads/neuroscope-key.pem cc@$SERVER_IP 'nvidia-smi dmon -c 20 -s u'"
echo
echo -e "${YELLOW}check sync status:${NC}"
echo "  bash '/Volumes/usb drive/neuroscope/check_sync_status.sh'"
echo
echo -e "${GREEN}checkpoints will be automatically backed up every 10 minutes!${NC}"
