#!/bin/bash
# run all journal extension experiments sequentially on chameleon cloud
# usage: bash run_all_experiments.sh [experiment_name]
# if no argument, runs all experiments in order

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$(dirname "$SCRIPT_DIR")/configs"
VENV="/home/cc/neuroscope-env"
LOG_DIR="/data/experiments/logs"

mkdir -p "$LOG_DIR"

# activate virtual environment
source "$VENV/bin/activate"

# add project root to pythonpath
export PYTHONPATH="/home/cc/neuroscope:$PYTHONPATH"

# ensure experiment output dirs exist
mkdir -p /data/experiments/journal_extension/{patchnce_hybrid,compression,multi_domain,federated}
mkdir -p /data/runs

echo "========================================="
echo " neuroscope journal extension experiments"
echo " $(date)"
echo " gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo " vram: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "========================================="

run_experiment() {
    local name="$1"
    local script="$2"
    local config="$3"

    echo ""
    echo "-----------------------------------------"
    echo " starting: $name"
    echo " $(date)"
    echo "-----------------------------------------"

    local logfile="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"

    python3 "$SCRIPT_DIR/$script" --config "$CONFIG_DIR/$config" 2>&1 | tee "$logfile"

    echo ""
    echo " completed: $name at $(date)"
    echo " log: $logfile"
    echo "-----------------------------------------"
}

# determine which experiment to run
EXPERIMENT="${1:-all}"

case "$EXPERIMENT" in
    patchnce|hybrid|a)
        run_experiment "patchnce_hybrid" "train_hybrid_nce.py" "patchnce_hybrid.yaml"
        ;;
    compression|compress|b)
        run_experiment "compression" "train_compression.py" "compression.yaml"
        ;;
    multi_domain|multidomain|c)
        run_experiment "multi_domain" "train_multi_domain.py" "multi_domain.yaml"
        ;;
    federated|fed|e)
        run_experiment "federated" "train_federated.py" "federated.yaml"
        ;;
    all)
        echo "running all experiments sequentially..."
        echo ""

        # extension a: patchnce hybrid (baseline extension, run first)
        run_experiment "patchnce_hybrid" "train_hybrid_nce.py" "patchnce_hybrid.yaml"

        # extension b: neural compression
        run_experiment "compression" "train_compression.py" "compression.yaml"

        # extension c: multi-domain (requires vendor-split data)
        # run_experiment "multi_domain" "train_multi_domain.py" "multi_domain.yaml"

        # extension e: federated learning
        run_experiment "federated" "train_federated.py" "federated.yaml"

        echo ""
        echo "========================================="
        echo " all experiments completed at $(date)"
        echo "========================================="
        ;;
    *)
        echo "unknown experiment: $EXPERIMENT"
        echo "usage: $0 [patchnce|compression|multi_domain|federated|all]"
        exit 1
        ;;
esac
