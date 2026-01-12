#!/bin/bash
# Monitor baseline CycleGAN training progress on Chameleon Cloud

SSH_KEY=~/Downloads/neuroscope-key.pem
SERVER=cc@192.5.86.251
LOG_FILE=/home/cc/neuroscope/baseline_training.log
EXP_DIR=/home/cc/neuroscope/experiments/baseline_cyclegan_25d_full
LOCAL_DIR="/Volumes/usb drive/neuroscope/baseline_training_progress"

# Create local monitoring directory
mkdir -p "$LOCAL_DIR"

# Get current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=================================================="
echo "Baseline CycleGAN Training Monitor"
echo "Timestamp: $(date)"
echo "=================================================="
echo

# Check if training process is running
echo "1. Checking if training is running..."
PROCESS=$(ssh -i $SSH_KEY $SERVER "ps aux | grep train_baseline_cyclegan | grep -v grep")
if [ -n "$PROCESS" ]; then
    echo "✅ Training process is active"
    echo "$PROCESS"
else
    echo "⚠️  No training process found"
fi
echo

# Get last 50 lines of training log
echo "2. Recent training log (last 50 lines):"
echo "---"
ssh -i $SSH_KEY $SERVER "tail -50 $LOG_FILE" | tee "$LOCAL_DIR/latest_log_$TIMESTAMP.txt"
echo "---"
echo

# Check for checkpoints
echo "3. Checking for saved checkpoints..."
ssh -i $SSH_KEY $SERVER "ls -lth $EXP_DIR/checkpoints/ 2>/dev/null | head -10" | tee "$LOCAL_DIR/checkpoints_$TIMESTAMP.txt"
echo

# Extract training stats if available
echo "4. Training statistics (if available):"
STATS=$(ssh -i $SSH_KEY $SERVER "grep -E 'Epoch [0-9]+/[0-9]+' $LOG_FILE | tail -5")
if [ -n "$STATS" ]; then
    echo "$STATS"
else
    echo "No epoch statistics found yet"
fi
echo

# Check GPU utilization
echo "5. GPU status:"
ssh -i $SSH_KEY $SERVER "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits" 2>/dev/null
echo

# Disk usage
echo "6. Disk usage:"
ssh -i $SSH_KEY $SERVER "df -h /home/cc/neuroscope | tail -1"
echo

echo "=================================================="
echo "Monitor complete. Logs saved to:"
echo "$LOCAL_DIR/"
echo "=================================================="
