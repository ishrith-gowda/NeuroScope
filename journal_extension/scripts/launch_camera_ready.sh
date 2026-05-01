#!/bin/bash
# orchestrates the camera-ready upgrade queue for all five extensions.
#
# extensions and approximate wallclock at the chosen training budget:
#   ext a: 1 additional seed for lambda_nce=0.5  (100 epochs, ~22h)
#   ext b: lambda_rate sweep at {0.001,0.005,0.05,0.1} (50 epochs, ~11h each, ~44h)
#   ext c: multi-domain n=4 training (60 epochs, ~12h)
#   ext d: task-aware harmonization for lambda_nce=0.5 (100 epochs, ~22h)
#   ext e: federated fedprox + scaffold (20 rounds each, ~17h total)
#
# run order is chosen so the shortest jobs finish first, giving early signal
# while longer runs continue in the background.
#
# usage:
#   bash launch_camera_ready.sh           # run everything
#   bash launch_camera_ready.sh ext_b     # run only ext b
#   bash launch_camera_ready.sh skip_a    # run all except ext a

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

VENV="/home/cc/neuroscope-env"
LOG_DIR="/data/experiments/logs"
EXP_DIR="/data/experiments/journal_extension"
DATA_BRATS="/data/preprocessed/brats"
DATA_UPENN="/data/preprocessed/upenn"

mkdir -p "$LOG_DIR" "$EXP_DIR"

source "$VENV/bin/activate"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

timestamp() { date +%Y%m%d_%H%M%S; }
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

MODE="${1:-all}"

# =========================================================================
# extension e: fedprox + scaffold (~17h total)
# =========================================================================
run_ext_e() {
    log "==========================================="
    log " extension e: federated fedprox + scaffold"
    log "==========================================="
    local config="$ROOT_DIR/journal_extension/configs/federated.yaml"

    for strat in fedprox scaffold; do
        local name="federated_${strat}_20rounds"
        local outdir="$EXP_DIR/$name"
        local logfile="$LOG_DIR/${name}_$(timestamp).log"

        log "  starting strategy=$strat"
        python3 -u "$ROOT_DIR/journal_extension/scripts/train_federated.py" \
            --config "$config" \
            --brats_dir "$DATA_BRATS" \
            --upenn_dir "$DATA_UPENN" \
            --output_dir "$outdir" \
            --aggregation_strategy "$strat" \
            --communication_rounds 20 \
            --local_epochs 5 \
            --num_workers 8 \
            --experiment_name "$name" \
            2>&1 | tee "$logfile"

        log "  $strat done."
    done
    log "extension e complete."
}

# =========================================================================
# extension c: multi-domain (60 epochs ~12h)
# =========================================================================
run_ext_c() {
    log "==========================================="
    log " extension c: multi-domain n=4"
    log "==========================================="
    local config="$ROOT_DIR/journal_extension/configs/multi_domain.yaml"
    local name="multidomain_n4_60epochs"
    local outdir="$EXP_DIR/$name"
    local logfile="$LOG_DIR/${name}_$(timestamp).log"

    python3 -u "$ROOT_DIR/journal_extension/scripts/train_multi_domain.py" \
        --config "$config" \
        --epochs 60 \
        --num_workers 8 \
        --experiment_name "$name" \
        --output_dir "$outdir" \
        2>&1 | tee "$logfile"

    log "extension c training done. running n x n inference..."
    python3 -u "$ROOT_DIR/journal_extension/scripts/multi_domain_matrix_inference.py" \
        --checkpoint "$outdir/$name/checkpoints/checkpoint_best.pth" \
        --config "$config" \
        --output_dir "$outdir/n_by_n_matrix" \
        2>&1 | tee "$LOG_DIR/${name}_inference_$(timestamp).log"
    log "extension c complete."
}

# =========================================================================
# extension b: lambda_rate sweep (4 new values * 50 epochs ~44h)
# =========================================================================
run_ext_b() {
    log "==========================================="
    log " extension b: lambda_rate sweep"
    log "==========================================="
    local config="$ROOT_DIR/journal_extension/configs/compression.yaml"

    for lam in 0.001 0.005 0.05 0.1; do
        local name="compression_lambda_rate_${lam}"
        local outdir="$EXP_DIR/$name"
        local logfile="$LOG_DIR/${name}_$(timestamp).log"

        log "  starting lambda_rate=$lam"
        python3 -u "$ROOT_DIR/journal_extension/scripts/train_compression.py" \
            --config "$config" \
            --brats_dir "$DATA_BRATS" \
            --upenn_dir "$DATA_UPENN" \
            --output_dir "$outdir" \
            --lambda_rate "$lam" \
            --epochs 50 \
            --num_workers 8 \
            --experiment_name "$name" \
            2>&1 | tee "$logfile"

        log "  lambda_rate=$lam done."
    done
    log "extension b complete."
}

# =========================================================================
# extension d: task-aware loss (100 epochs ~22h)
# =========================================================================
run_ext_d() {
    log "==========================================="
    log " extension d: task-aware harmonization"
    log "==========================================="
    local config="$ROOT_DIR/journal_extension/configs/patchnce_hybrid.yaml"
    local name="patchnce_taskaware_lambda0.5"
    local outdir="$EXP_DIR/$name"
    local logfile="$LOG_DIR/${name}_$(timestamp).log"

    python3 -u "$ROOT_DIR/journal_extension/scripts/train_hybrid_nce.py" \
        --config "$config" \
        --brats_dir "$DATA_BRATS" \
        --upenn_dir "$DATA_UPENN" \
        --output_dir "$outdir" \
        --num_workers 8 \
        --lambda_nce 0.5 \
        --task_aware_loss \
        --mask_weight_foreground 4.0 \
        --epochs 100 \
        --experiment_name "$name" \
        --seed 42 \
        2>&1 | tee "$logfile"

    log "extension d training done. running downstream segmentation eval..."
    python3 -u "$ROOT_DIR/journal_extension/scripts/eval_downstream.py" \
        --checkpoint "$outdir/$name/checkpoints/checkpoint_best.pth" \
        --output_dir "$outdir/downstream" \
        2>&1 | tee "$LOG_DIR/${name}_downstream_$(timestamp).log" || \
        log "  downstream eval encountered an error; continuing."
    log "extension d complete."
}

# =========================================================================
# extension a: 2nd seed for lambda_nce=0.5 (100 epochs ~22h)
# =========================================================================
run_ext_a() {
    log "==========================================="
    log " extension a: 2nd seed for lambda_nce=0.5"
    log "==========================================="
    local config="$ROOT_DIR/journal_extension/configs/patchnce_hybrid.yaml"
    local name="patchnce_lambda0.5_seed17_100epochs"
    local outdir="$EXP_DIR/$name"
    local logfile="$LOG_DIR/${name}_$(timestamp).log"

    python3 -u "$ROOT_DIR/journal_extension/scripts/train_hybrid_nce.py" \
        --config "$config" \
        --brats_dir "$DATA_BRATS" \
        --upenn_dir "$DATA_UPENN" \
        --output_dir "$outdir" \
        --num_workers 8 \
        --lambda_nce 0.5 \
        --epochs 100 \
        --experiment_name "$name" \
        --seed 17 \
        2>&1 | tee "$logfile"

    log "extension a 2nd seed complete."
}

# =========================================================================
# main
# =========================================================================
case "$MODE" in
    ext_e) run_ext_e ;;
    ext_c) run_ext_c ;;
    ext_b) run_ext_b ;;
    ext_d) run_ext_d ;;
    ext_a) run_ext_a ;;
    skip_a)
        run_ext_e
        run_ext_c
        run_ext_b
        run_ext_d
        ;;
    all|*)
        # order: shortest jobs first (e -> c -> a -> d -> b)
        run_ext_e
        run_ext_c
        run_ext_a
        run_ext_d
        run_ext_b
        ;;
esac

log "==========================================="
log " camera-ready queue complete"
log " $(date)"
log "==========================================="
