#!/bin/bash
# =============================================================================
# chameleon cloud training launcher
# =============================================================================
# usage:
#   ./train_on_chameleon.sh baseline   # train baseline cyclegan
#   ./train_on_chameleon.sh sa         # train sa-cyclegan with attention
#   ./train_on_chameleon.sh both       # train both sequentially
# =============================================================================

set -e  # exit on error

# -----------------------------------------------------------------------------
# configuration
# -----------------------------------------------------------------------------
PROJECT_DIR="/home/cc/neuroscope"
CONDA_ENV="neuroscope"
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# -----------------------------------------------------------------------------
# color output (optional)
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# -----------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------
log_info() {
    echo -e "${GREEN}[info]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[warn]${NC} $1"
}

log_error() {
    echo -e "${RED}[error]${NC} $1"
}

# -----------------------------------------------------------------------------
# environment setup
# -----------------------------------------------------------------------------
setup_environment() {
    log_info "setting up environment..."

    # activate conda
    source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
    conda activate ${CONDA_ENV}

    # verify gpu
    python -c "import torch; print(f'cuda available: {torch.cuda.is_available()}'); print(f'gpu: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

    # create log directory
    mkdir -p ${LOG_DIR}

    log_info "environment ready"
}

# -----------------------------------------------------------------------------
# training functions
# -----------------------------------------------------------------------------
train_baseline() {
    log_info "starting baseline cyclegan training..."

    cd ${PROJECT_DIR}

    # run baseline training
    python -m scripts.02_training.train_baseline_cyclegan_25d \
        --config neuroscope/config/experiments/chameleon_baseline_v100.yaml \
        2>&1 | tee ${LOG_DIR}/baseline_${TIMESTAMP}.log

    log_info "baseline training complete"
}

train_sa_cyclegan() {
    log_info "starting sa-cyclegan training..."

    cd ${PROJECT_DIR}

    # run sa-cyclegan training
    python -m scripts.02_training.train_sa_cyclegan_25d \
        --config neuroscope/config/experiments/chameleon_sa_cyclegan_v100.yaml \
        2>&1 | tee ${LOG_DIR}/sa_cyclegan_${TIMESTAMP}.log

    log_info "sa-cyclegan training complete"
}

# -----------------------------------------------------------------------------
# gpu monitoring (background)
# -----------------------------------------------------------------------------
start_gpu_monitor() {
    log_info "starting gpu monitoring..."

    # monitor gpu every 60 seconds
    while true; do
        nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv >> ${LOG_DIR}/gpu_stats_${TIMESTAMP}.csv
        sleep 60
    done &

    GPU_MONITOR_PID=$!
    echo ${GPU_MONITOR_PID} > ${LOG_DIR}/gpu_monitor.pid
}

stop_gpu_monitor() {
    if [ -f ${LOG_DIR}/gpu_monitor.pid ]; then
        kill $(cat ${LOG_DIR}/gpu_monitor.pid) 2>/dev/null || true
        rm ${LOG_DIR}/gpu_monitor.pid
    fi
}

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
main() {
    local mode=${1:-"baseline"}

    log_info "chameleon training launcher"
    log_info "mode: ${mode}"
    log_info "timestamp: ${TIMESTAMP}"

    setup_environment
    start_gpu_monitor

    # trap to stop monitor on exit
    trap stop_gpu_monitor EXIT

    case ${mode} in
        baseline)
            train_baseline
            ;;
        sa|sa-cyclegan)
            train_sa_cyclegan
            ;;
        both)
            train_baseline
            train_sa_cyclegan
            ;;
        *)
            log_error "unknown mode: ${mode}"
            echo "usage: $0 {baseline|sa|both}"
            exit 1
            ;;
    esac

    stop_gpu_monitor

    log_info "all training complete"
    log_info "logs saved to: ${LOG_DIR}"
}

main "$@"
