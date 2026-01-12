#!/bin/bash
# Quick script to check sync status

echo "============================================"
echo "Checkpoint Sync Status"
echo "============================================"
echo ""

# Check if sync process is running
SYNC_PID=$(ps aux | grep "auto_sync_checkpoints.sh" | grep -v grep | awk '{print $2}')

if [ -n "$SYNC_PID" ]; then
    echo "✓ Sync process is RUNNING (PID: $SYNC_PID)"
else
    echo "✗ Sync process is NOT RUNNING"
    echo "  To restart: /Volumes/usb\ drive/neuroscope/auto_sync_checkpoints.sh &"
fi

echo ""
echo "Local checkpoint directory:"
echo "  /Volumes/usb drive/neuroscope/experiments_from_gpu/"
echo ""

# Count local checkpoints
CHECKPOINT_COUNT=$(find "/Volumes/usb drive/neuroscope/experiments_from_gpu/" -name "*.pth" -o -name "*.pt" 2>/dev/null | wc -l | tr -d ' ')
echo "Checkpoints synced: $CHECKPOINT_COUNT"

# Show recent sync activity
echo ""
echo "Last 10 sync events:"
echo "--------------------------------------------"
tail -n 30 "/Volumes/usb drive/neuroscope/logs/sync.log" 2>/dev/null | grep -E "\[(.*)\] Sync #|Total checkpoints" | tail -10

echo ""
echo "============================================"
echo "To view full sync log:"
echo "  tail -f /Volumes/usb\\ drive/neuroscope/logs/sync.log"
echo "============================================"
