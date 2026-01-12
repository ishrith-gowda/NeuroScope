#!/bin/bash
SERVER="192.5.86.251"
SSH_KEY="$HOME/Downloads/neuroscope-key.pem"
REMOTE_DIR="cc@${SERVER}:~/neuroscope/experiments/"
LOCAL_DIR="/Volumes/usb drive/neuroscope/experiments_from_gpu/"
LOG_FILE="/Volumes/usb drive/neuroscope/logs/sync.log"

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$LOCAL_DIR"

echo "[$(date)] starting automatic checkpoint sync..." >> "$LOG_FILE"
COUNTER=1

while true; do
    echo "[$(date)] sync #${COUNTER} starting..." >> "$LOG_FILE"

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
        -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10" \
        "$REMOTE_DIR" \
        "$LOCAL_DIR" >> "$LOG_FILE" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] sync #${COUNTER} completed successfully" >> "$LOG_FILE"
    else
        echo "[$(date)] sync #${COUNTER} failed with exit code: $EXIT_CODE" >> "$LOG_FILE"
    fi

    COUNTER=$((COUNTER + 1))
    sleep 600
done
