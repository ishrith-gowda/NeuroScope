#!/bin/bash
# neuroscope cluster training monitor
# usage: bash tools/cluster_status.sh
#   or:  bash tools/cluster_status.sh --watch   (auto-refresh every 30s)
#   or:  bash tools/cluster_status.sh --logs     (show recent log output)
#   or:  bash tools/cluster_status.sh --attach   (attach to tmux session)

SSH_KEY="$HOME/.ssh/neuroscope-key"
HOST="cc@192.5.86.219"
SSH_CMD="ssh -i $SSH_KEY -o ConnectTimeout=5 -o StrictHostKeyChecking=no $HOST"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

show_status() {
    echo ""
    echo -e "${CYAN}==========================================${NC}"
    echo -e "${CYAN} neuroscope cluster training status${NC}"
    echo -e "${CYAN} $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${CYAN}==========================================${NC}"

    # gpu status
    echo ""
    echo -e "${YELLOW}[gpu]${NC}"
    $SSH_CMD "nvidia-smi --query-gpu=name,temperature.gpu,power.draw,power.limit,memory.used,memory.total,utilization.gpu --format=csv,noheader" 2>/dev/null | while IFS=',' read -r name temp power plimit memused memtotal util; do
        echo -e "  model:       ${GREEN}${name}${NC}"
        echo -e "  temperature: ${temp}"
        echo -e "  power:       ${power} /${plimit}"
        echo -e "  vram:        ${GREEN}${memused} /${memtotal}${NC}"
        echo -e "  utilization: ${GREEN}${util}${NC}"
    done

    # training process
    echo ""
    echo -e "${YELLOW}[training]${NC}"
    PROC=$($SSH_CMD "ps aux | grep 'python3.*train_.*\.py' | grep -v grep | head -1" 2>/dev/null)
    if [ -n "$PROC" ]; then
        SCRIPT=$(echo "$PROC" | grep -o 'train_[^ ]*\.py' | head -1)
        CPU=$(echo "$PROC" | awk '{print $3}')
        MEM=$(echo "$PROC" | awk '{print $4}')
        ELAPSED=$(echo "$PROC" | awk '{print $10}')
        echo -e "  status:  ${GREEN}running${NC}"
        echo -e "  script:  ${SCRIPT}"
        echo -e "  cpu:     ${CPU}%"
        echo -e "  ram:     ${MEM}%"
        echo -e "  elapsed: ${ELAPSED}"
    else
        echo -e "  status:  ${RED}not running${NC}"
    fi

    # latest progress from tmux
    echo ""
    echo -e "${YELLOW}[progress]${NC}"
    PROGRESS=$($SSH_CMD "tmux capture-pane -t training -p 2>/dev/null | grep -E 'epoch|loss|it/s' | tail -3" 2>/dev/null)
    if [ -n "$PROGRESS" ]; then
        echo "$PROGRESS" | while IFS= read -r line; do
            echo "  $line"
        done
    else
        echo "  no tmux session found"
    fi

    # disk usage
    echo ""
    echo -e "${YELLOW}[storage]${NC}"
    $SSH_CMD "df -h /data 2>/dev/null | tail -1" 2>/dev/null | awk '{printf "  data disk:   %s used / %s total (%s)\n", $3, $2, $5}'
    $SSH_CMD "du -sh /data/experiments/ 2>/dev/null" 2>/dev/null | awk '{printf "  experiments: %s\n", $1}'
    $SSH_CMD "du -sh /data/preprocessed/ 2>/dev/null" 2>/dev/null | awk '{printf "  data:        %s\n", $1}'

    # lease info
    echo ""
    echo -e "${YELLOW}[lease]${NC}"
    LEASE_END=$($SSH_CMD "grep 'Lease end' /var/lib/cloud/instance/user-data.txt 2>/dev/null || grep -r 'Lease end' /etc/ 2>/dev/null | head -1" 2>/dev/null)
    LEASE_EXP=$($SSH_CMD "grep 'expires in' /var/lib/cloud/instance/user-data.txt 2>/dev/null || grep -r 'expires' /etc/ 2>/dev/null | head -1" 2>/dev/null)
    if [ -n "$LEASE_END" ]; then
        echo "  $LEASE_END"
    fi
    echo "  end: 2026-03-25 17:38 UTC"

    # latest checkpoint
    echo ""
    echo -e "${YELLOW}[checkpoints]${NC}"
    CKPT=$($SSH_CMD "find /data/experiments -name '*.pth' -o -name '*.pt' 2>/dev/null | sort -t/ -k7 -n | tail -3" 2>/dev/null)
    if [ -n "$CKPT" ]; then
        echo "$CKPT" | while IFS= read -r line; do
            SIZE=$($SSH_CMD "du -sh '$line' 2>/dev/null" | awk '{print $1}')
            echo "  $SIZE  $(basename $line)"
        done
    else
        echo "  no checkpoints yet"
    fi

    echo ""
    echo -e "${CYAN}==========================================${NC}"
}

case "${1:-status}" in
    --watch|-w)
        while true; do
            clear
            show_status
            echo ""
            echo "refreshing in 30s... (ctrl+c to stop)"
            sleep 30
        done
        ;;
    --logs|-l)
        echo "latest training log output:"
        echo "---"
        $SSH_CMD "tmux capture-pane -t training -p -S -80 2>/dev/null | tail -40"
        ;;
    --attach|-a)
        echo "attaching to training session (ctrl+b then d to detach)..."
        ssh -t -i "$SSH_KEY" "$HOST" "tmux attach -t training"
        ;;
    --gpu|-g)
        $SSH_CMD "nvidia-smi"
        ;;
    *)
        show_status
        ;;
esac
