#!/bin/bash
# full experiment pipeline for journal extension
# designed for week-long chameleon cloud lease
# handles resume from any interruption point
#
# usage: bash run_full_pipeline.sh [stage]
#   stages: federated, compression, multi_domain, downstream, figures, all
#   default: all (runs everything sequentially with auto-resume)

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

timestamp() {
    date +%Y%m%d_%H%M%S
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_with_resume() {
    local name="$1"
    local script="$2"
    local config="$3"
    local extra_args="${4:-}"

    log "========================================="
    log " experiment: $name"
    log " $(date)"
    log " gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    log " vram: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
    log "========================================="

    local logfile="$LOG_DIR/${name}_$(timestamp).log"
    local resume_args=""

    # auto-detect existing checkpoints
    local latest_ckpt=$(find "$EXP_DIR/$name" -name "checkpoint_latest.pth" 2>/dev/null | head -1)
    if [ -n "$latest_ckpt" ]; then
        log "found checkpoint: $latest_ckpt — resuming"
        if [ "$name" = "federated" ]; then
            resume_args="--resume"
        else
            resume_args="--resume $latest_ckpt"
        fi
    fi

    python3 -u "$SCRIPT_DIR/$script" \
        --config "$CONFIG_DIR/$config" \
        $resume_args $extra_args \
        2>&1 | tee "$logfile"

    log "completed: $name"
    log "log: $logfile"
}

# =========================================================================
# stage 1: federated learning (extension e)
# =========================================================================
run_federated() {
    log "=== stage 1: federated learning (extension e) ==="

    # check if already complete
    local history="$EXP_DIR/federated/sa_cyclegan_25d_federated/training_history.json"
    if [ -f "$history" ]; then
        local n_rounds=$(python3 -c "import json; h=json.load(open('$history')); print(len(h.get('rounds',[])))")
        if [ "$n_rounds" -ge 40 ]; then
            log "federated already complete ($n_rounds/40 rounds) — skipping"
            return 0
        fi
    fi

    run_with_resume "federated" "train_federated.py" "federated.yaml"
}

# =========================================================================
# stage 2: compression-harmonization (extension b)
# =========================================================================
run_compression() {
    log "=== stage 2: compression-harmonization (extension b) ==="

    # check if already complete
    local history="$EXP_DIR/compression/sa_cyclegan_25d_compression/training_history.json"
    if [ -f "$history" ]; then
        local n_epochs=$(python3 -c "import json; h=json.load(open('$history')); print(len(h.get('train',{}).get('G_loss',[])))")
        if [ "$n_epochs" -ge 200 ]; then
            log "compression already complete ($n_epochs/200 epochs) — skipping"
            return 0
        fi
    fi

    run_with_resume "compression" "train_compression.py" "compression.yaml"
}

# =========================================================================
# stage 3: multi-domain harmonization (extension c)
# =========================================================================
run_multi_domain() {
    log "=== stage 3: multi-domain harmonization (extension c) ==="

    # ensure domain split metadata is on cluster
    local split_file="$PROJECT_ROOT/data/metadata/multi_domain_split.json"
    if [ ! -f "$split_file" ]; then
        log "error: domain split file not found: $split_file"
        log "run the split generation script locally first"
        return 1
    fi

    # check if already complete
    local history="$EXP_DIR/multi_domain/sa_cyclegan_25d_multidomain/training_history.json"
    if [ -f "$history" ]; then
        local n_epochs=$(python3 -c "import json; h=json.load(open('$history')); print(len(h.get('train',{}).get('G_loss',[])))" 2>/dev/null)
        if [ "$n_epochs" -ge 200 ]; then
            log "multi-domain already complete ($n_epochs/200 epochs) — skipping"
            return 0
        fi
    fi

    run_with_resume "multi_domain" "train_multi_domain.py" "multi_domain.yaml"
}

# =========================================================================
# stage 4: downstream evaluation (extension d)
# =========================================================================
run_downstream() {
    log "=== stage 4: downstream evaluation (extension d) ==="

    # use best available harmonization checkpoint (federated best > compression best > federated latest)
    local harmonization_ckpt=""
    local fed_best="$EXP_DIR/federated/sa_cyclegan_25d_federated/checkpoints/checkpoint_best.pth"
    local comp_best="$EXP_DIR/compression/sa_cyclegan_25d_compressed/checkpoints/checkpoint_best.pth"
    local fed_latest="$EXP_DIR/federated/sa_cyclegan_25d_federated/checkpoints/checkpoint_latest.pth"

    if [ -f "$fed_best" ]; then
        harmonization_ckpt="$fed_best"
        log "using federated best checkpoint for harmonization"
    elif [ -f "$comp_best" ]; then
        harmonization_ckpt="$comp_best"
        log "using compression best checkpoint for harmonization"
    elif [ -f "$fed_latest" ]; then
        harmonization_ckpt="$fed_latest"
        log "using federated latest checkpoint for harmonization"
    else
        log "error: no harmonization checkpoint found — cannot run downstream eval"
        return 1
    fi

    local results="$EXP_DIR/downstream_eval/results.json"
    if [ -f "$results" ]; then
        log "downstream eval results already exist — skipping"
        return 0
    fi

    mkdir -p "$EXP_DIR/downstream_eval"

    python3 -u "$SCRIPT_DIR/eval_downstream.py" \
        --brats_dir "$PROJECT_ROOT/preprocessed/brats" \
        --upenn_dir "$PROJECT_ROOT/preprocessed/upenn" \
        --checkpoint_dir "$harmonization_ckpt" \
        --output_dir "$EXP_DIR/downstream_eval" \
        2>&1 | tee "$LOG_DIR/downstream_eval_$(timestamp).log"

    log "completed: downstream evaluation"
}

# =========================================================================
# main
# =========================================================================

STAGE="${1:-all}"

log "========================================="
log " neuroscope journal extension pipeline"
log " stage: $STAGE"
log " $(date)"
log "========================================="

case "$STAGE" in
    federated|fed|e)
        run_federated
        ;;
    compression|compress|b)
        run_compression
        ;;
    multi_domain|multidomain|c)
        run_multi_domain
        ;;
    downstream|eval|d)
        run_downstream
        ;;
    all)
        log "running full pipeline sequentially..."

        # extension e: federated (resume if interrupted)
        run_federated

        # extension b: compression
        run_compression

        # extension c: multi-domain
        run_multi_domain

        # extension d: downstream eval
        run_downstream

        log "========================================="
        log " all experiments completed!"
        log " $(date)"
        log "========================================="
        ;;
    *)
        echo "unknown stage: $STAGE"
        echo "usage: $0 [federated|compression|multi_domain|downstream|all]"
        exit 1
        ;;
esac
