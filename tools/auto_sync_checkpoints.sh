#!/bin/bash
# Automatic Checkpoint Sync Script
# Syncs training checkpoints and logs from GPU server to local machine every 10 minutes

SERVER="cc@192.5.86.245"
SSH_KEY="$HOME/Downloads/neuroscope-key.pem"
REMOTE_DIR="/home/cc/neuroscope/experiments/"
LOCAL_DIR="/Volumes/usb drive/neuroscope/experiments_from_gpu/"
LOG_FILE="/Volumes/usb drive/neuroscope/logs/sync.log"

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

echo "========================================" >> "$LOG_FILE"
echo "Automatic Checkpoint Sync Started" >> "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "Server: $SERVER" >> "$LOG_FILE"
echo "Sync Interval: 10 minutes" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Sync counter
SYNC_COUNT=0

while true; do
    SYNC_COUNT=$((SYNC_COUNT + 1))
    echo "" >> "$LOG_FILE"
    echo "[$(date)] Sync #$SYNC_COUNT starting..." >> "$LOG_FILE"

    # Run rsync with error handling
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
        "$SERVER:$REMOTE_DIR" \
        "$LOCAL_DIR" >> "$LOG_FILE" 2>&1

    RSYNC_EXIT=$?

    if [ $RSYNC_EXIT -eq 0 ]; then
        echo "[$(date)] Sync #$SYNC_COUNT completed successfully" >> "$LOG_FILE"

        # Show checkpoint summary
        CHECKPOINT_COUNT=$(find "$LOCAL_DIR" -name "*.pth" -o -name "*.pt" 2>/dev/null | wc -l | tr -d ' ')
        echo "[$(date)] Total checkpoints synced: $CHECKPOINT_COUNT" >> "$LOG_FILE"
    else
        echo "[$(date)] Sync #$SYNC_COUNT failed with exit code: $RSYNC_EXIT" >> "$LOG_FILE"
    fi

    # Wait 10 minutes before next sync
    echo "[$(date)] Waiting 10 minutes until next sync..." >> "$LOG_FILE"
    sleep 600
done
