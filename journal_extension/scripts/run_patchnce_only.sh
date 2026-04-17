#!/bin/bash
# run only extension a (patchnce hybrid loss) experiments
# use this when extensions b-e are already complete
#
# runs: (1) primary lambda=1.0 training, (2) ablation sweep lambda={0.1, 0.5, 2.0}
# estimated time: ~50h primary + ~150h ablation = ~200h total on a100 80gb
# with existing b-e results already done, run: bash run_patchnce_only.sh
#
# for just the primary run without ablation: bash run_patchnce_only.sh primary
# for just the ablation sweep: bash run_patchnce_only.sh ablation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$(dirname "$SCRIPT_DIR")/configs"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
VENV="/home/cc/neuroscope-env"
LOG_DIR="/data/experiments/logs"
EXP_DIR="/data/experiments/journal_extension"

mkdir -p "$LOG_DIR" "$EXP_DIR"

# activate virtual environment
source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

timestamp() { date +%Y%m%d_%H%M%S; }
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

MODE="${1:-all}"

gpu_info() {
    # try nvidia first, then rocm for amd gpus
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
    elif command -v rocm-smi &>/dev/null; then
        rocm-smi --showproductname --showmeminfo vram 2>/dev/null | head -5
    else
        python3 -c "import torch; [print(f'{torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_memory//1024**3}gb') for i in range(torch.cuda.device_count())]" 2>/dev/null || echo "unknown"
    fi
}

log "========================================="
log " extension a: patchnce hybrid loss"
log " mode: $MODE"
log " gpu: $(gpu_info)"
log "========================================="

# =========================================================================
# primary run: lambda_nce = 1.0 (200 epochs)
# =========================================================================
run_primary() {
    log "--- primary patchnce training (lambda_nce=1.0) ---"

    local ckpt="$EXP_DIR/patchnce_hybrid/sa_cyclegan_25d_patchnce_hybrid/checkpoints/checkpoint_latest.pth"
    local resume_args=""
    if [ -f "$ckpt" ]; then
        log "found checkpoint — resuming from $ckpt"
        resume_args="--resume $ckpt"
    fi

    python3 -u "$SCRIPT_DIR/train_hybrid_nce.py" \
        --config "$CONFIG_DIR/patchnce_hybrid.yaml" \
        $resume_args \
        2>&1 | tee "$LOG_DIR/patchnce_primary_$(timestamp).log"

    log "primary patchnce training complete"
}

# =========================================================================
# ablation sweep: lambda_nce = {0.1, 0.5, 2.0}
# =========================================================================
run_ablation() {
    log "--- patchnce lambda ablation sweep ---"

    for lam in 0.1 0.5 2.0; do
        local name="patchnce_ablation_lambda${lam}"

        # check if already complete
        local history="$EXP_DIR/$name/training_history.json"
        if [ -f "$history" ]; then
            local n_epochs=$(python3 -c "import json; h=json.load(open('$history')); print(len(h.get('train',{}).get('G_loss',[])))" 2>/dev/null)
            if [ "$n_epochs" -ge 200 ]; then
                log "lambda=$lam already complete ($n_epochs/200 epochs) — skipping"
                continue
            fi
        fi

        # check for resume checkpoint
        local ckpt="$EXP_DIR/$name/checkpoints/checkpoint_latest.pth"
        local resume_args=""
        if [ -f "$ckpt" ]; then
            log "found checkpoint for lambda=$lam — resuming"
            resume_args="--resume $ckpt"
        fi

        log "training patchnce with lambda_nce=$lam"
        python3 -u "$SCRIPT_DIR/train_hybrid_nce.py" \
            --config "$CONFIG_DIR/patchnce_hybrid.yaml" \
            --lambda_nce "$lam" \
            --experiment_name "$name" \
            --output_dir "$EXP_DIR/$name" \
            $resume_args \
            2>&1 | tee "$LOG_DIR/${name}_$(timestamp).log"

        log "lambda=$lam complete"
    done

    log "ablation sweep complete"
}

# =========================================================================
# main
# =========================================================================
case "$MODE" in
    primary)
        run_primary
        ;;
    ablation)
        run_ablation
        ;;
    all)
        run_primary
        run_ablation
        ;;
    *)
        echo "usage: $0 [primary|ablation|all]"
        exit 1
        ;;
esac

log "========================================="
log " extension a complete"
log " $(date)"
log "========================================="
