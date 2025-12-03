#!/bin/bash

# ======================================================
# NeuroScope Training Pipeline Wrapper Script
# Enhanced version with real-time training metrics display
# ======================================================

# Set error handling
set -e

# Function to create a progress bar
function progress_bar {
    local progress=$1
    local total=$2
    local width=50
    local percentage=$((100 * progress / total))
    local filled=$((width * progress / total))
    local empty=$((width - filled))
    
    printf "\r[%${filled}s%${empty}s] %3d%% (%d/%d)" | tr ' ' '=' | tr '\n' ' '
    printf "%s" "${percentage}%" "${progress}" "${total}"
}

# Function to extract and display the latest metrics from TensorBoard logs
function display_training_metrics {
    local runs_dir="/Volumes/usb drive/neuroscope/runs"
    local latest_run=$(find "$runs_dir" -type f -name "events.out.tfevents.*" -print0 | xargs -0 ls -t | head -n 1)
    
    if [ -n "$latest_run" ]; then
        echo -e "\n=== CURRENT TRAINING METRICS (from TensorBoard logs) ==="
        
        # Use Python to extract and display metrics
        python -c "
import os
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import numpy as np

# Get latest TensorBoard log file
log_file = '$latest_run'
if not os.path.exists(log_file):
    print('No TensorBoard logs found yet')
    sys.exit(0)

# Load TensorBoard data
event_acc = EventAccumulator(log_file)
event_acc.Reload()

# Get available tags (metrics)
tags = event_acc.Tags()
scalar_tags = tags.get('scalars', [])

if not scalar_tags:
    print('No metrics available yet in the logs')
    sys.exit(0)

# Extract and display metrics
print('Latest training metrics:')
metrics_data = {}

for tag in scalar_tags:
    events = event_acc.Scalars(tag)
    if events:
        latest = events[-1]  # Get the most recent event
        step = latest.step
        value = latest.value
        
        # Format tag name for better display
        display_name = tag.replace('losses_', '').upper()
        metrics_data[display_name] = (step, value)

# Display metrics in a table format
if metrics_data:
    print(f'\n{"METRIC":<15} {"STEP":<10} {"VALUE":<15}')
    print('-' * 40)
    for metric, (step, value) in metrics_data.items():
        print(f'{metric:<15} {step:<10} {value:<15.6f}')

    # Calculate training progress based on steps
    steps = [step for _, (step, _) in metrics_data.items()]
    if steps:
        max_step = max(steps)
        target_steps = 10000  # Adjust based on your expected training length
        progress = min(100, (max_step / target_steps) * 100)
        print(f'\nEstimated progress: {progress:.1f}% (Step {max_step}/{target_steps})')
else:
    print('No metrics data available yet')
"
    else
        echo "No TensorBoard logs found yet. Training may still be in initialization phase."
    fi
}

# Function to display recent losses as a simple ASCII chart
function display_loss_chart {
    local samples_dir="/Volumes/usb drive/neuroscope/samples"
    
    # Check if loss_curves.png exists (for reference only - we'll generate our own)
    if [ -f "$samples_dir/loss_curves.png" ]; then
        echo -e "\n=== TRAINING LOSS CURVES (Updated) ==="
        echo "Loss curves image saved at: $samples_dir/loss_curves.png"
    fi
    
    # Generate ASCII loss chart from TensorBoard data
    local runs_dir="/Volumes/usb drive/neuroscope/runs"
    local latest_run=$(find "$runs_dir" -type f -name "events.out.tfevents.*" -print0 | xargs -0 ls -t | head -n 1)
    
    if [ -n "$latest_run" ]; then
        python -c "
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

# Get latest TensorBoard log file
log_file = '$latest_run'

# Load TensorBoard data
try:
    event_acc = EventAccumulator(log_file)
    event_acc.Reload()
    
    # Get available tags (metrics)
    tags = event_acc.Tags()
    scalar_tags = tags.get('scalars', [])
    
    if not scalar_tags or len(scalar_tags) < 2:
        print('Not enough loss data available for charting')
        sys.exit(0)
    
    # Focus on Generator and Discriminator losses
    gen_tag = next((tag for tag in scalar_tags if 'G' in tag), None)
    disc_tag = next((tag for tag in scalar_tags if 'D' in tag), None)
    
    if gen_tag and disc_tag:
        gen_events = event_acc.Scalars(gen_tag)
        disc_events = event_acc.Scalars(disc_tag)
        
        if len(gen_events) > 10 and len(disc_events) > 10:
            # Get loss values
            gen_steps = [e.step for e in gen_events]
            gen_values = [e.value for e in gen_events]
            disc_steps = [e.step for e in disc_events]
            disc_values = [e.value for e in disc_events]
            
            # Create simple ASCII chart (last 20 points)
            rows = 10  # height of the chart
            cols = 50  # width of the chart
            
            # Use only last N points for the chart
            n_points = min(50, len(gen_values))
            gen_values = gen_values[-n_points:]
            disc_values = disc_values[-n_points:]
            
            # Normalize values to chart height
            max_val = max(max(gen_values), max(disc_values))
            min_val = min(min(gen_values), min(disc_values))
            range_val = max_val - min_val or 1.0
            
            # Generate chart
            chart = [[' ' for _ in range(cols)] for _ in range(rows)]
            
            for i in range(min(cols, n_points)):
                gen_idx = int(rows - 1 - (rows - 1) * (gen_values[i * len(gen_values) // cols] - min_val) / range_val)
                disc_idx = int(rows - 1 - (rows - 1) * (disc_values[i * len(disc_values) // cols] - min_val) / range_val)
                
                gen_idx = max(0, min(rows - 1, gen_idx))
                disc_idx = max(0, min(rows - 1, disc_idx))
                
                chart[gen_idx][i] = 'G'
                chart[disc_idx][i] = 'D'
            
            # Print chart
            print('\\nLoss chart (G: Generator, D: Discriminator):')
            print(f'Max value: {max_val:.3f}')
            print('+' + '-' * cols + '+')
            for row in chart:
                print('|' + ''.join(row) + '|')
            print('+' + '-' * cols + '+')
            print(f'Min value: {min_val:.3f}')
            print(f'Steps: {gen_steps[0]} to {gen_steps[-1]}')
        else:
            print('Not enough data points for loss chart yet')
    else:
        print('Generator or Discriminator loss data not found')
except Exception as e:
    print(f'Error creating loss chart: {e}')
"
    fi
}

function check_checkpoint_progress {
    local ckpt_dir="/Volumes/usb drive/neuroscope/checkpoints"
    
    echo -e "\n=== CHECKPOINT PROGRESS ==="
    
    if [ -d "$ckpt_dir" ]; then
        # Count checkpoint files
        local g_a2b_count=$(ls -1 "$ckpt_dir"/G_A2B_*.pth 2>/dev/null | wc -l | tr -d ' ')
        local g_b2a_count=$(ls -1 "$ckpt_dir"/G_B2A_*.pth 2>/dev/null | wc -l | tr -d ' ')
        local d_a_count=$(ls -1 "$ckpt_dir"/D_A_*.pth 2>/dev/null | wc -l | tr -d ' ')
        local d_b_count=$(ls -1 "$ckpt_dir"/D_B_*.pth 2>/dev/null | wc -l | tr -d ' ')
        
        # Get latest epoch number
        local latest_epoch=0
        if [ "$g_a2b_count" -gt 0 ]; then
            latest_epoch=$(ls -1 "$ckpt_dir"/G_A2B_*.pth 2>/dev/null | sort -V | tail -1 | grep -o '[0-9]\+\.pth' | cut -d'.' -f1)
        fi
        
        echo "Latest checkpoint epoch: $latest_epoch"
        echo "Generator A2B checkpoints: $g_a2b_count"
        echo "Generator B2A checkpoints: $g_b2a_count"
        echo "Discriminator A checkpoints: $d_a_count"
        echo "Discriminator B checkpoints: $d_b_count"
        
        # Show sample images if they exist
        local latest_sample=$(ls -1t "$NEUROSCOPE_DIR/samples"/sample_*.png 2>/dev/null | head -1)
        if [ -n "$latest_sample" ]; then
            echo "Latest sample image: $latest_sample"
        fi
    else
        echo "No checkpoints directory found yet"
    fi
}

# Main script execution starts here
NEUROSCOPE_DIR="/Volumes/usb drive/neuroscope"

# Create a banner with script information
clear
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           NEUROSCOPE TRAINING PIPELINE - DETAILED METRICS        ║"
echo "║══════════════════════════════════════════════════════════════════║"
echo "║  This script runs the training pipeline with real-time metrics   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo "Starting at: $(date)"

# Activate virtual environment if it exists
if [ -f "$NEUROSCOPE_DIR/.venv/bin/activate" ]; then
  echo "Activating virtual environment..."
  source "$NEUROSCOPE_DIR/.venv/bin/activate"
fi

# Change to the project directory
cd "$NEUROSCOPE_DIR"

# Report Python version and environment
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"

# Install required packages if not already installed
echo -e "\n=== CHECKING DEPENDENCIES ==="
python -c "
import importlib.util
missing = []
for pkg in ['tensorboard', 'torch', 'matplotlib', 'seaborn', 'pandas', 'numpy']:
    if importlib.util.find_spec(pkg) is None:
        missing.append(pkg)
if missing:
    print(f'Missing packages: {\" \".join(missing)}')
    import sys
    sys.exit(1)
else:
    print('All required packages are installed')
" || {
    echo "Installing missing packages..."
    pip install tensorboard torch matplotlib seaborn pandas numpy
}

# Run the training pipeline with verbose flag and skip validation to save time
echo -e "\n=== STARTING TRAINING PIPELINE ==="
python scripts/02_model_development_pipeline/run_training_pipeline.py --verbose --skip-validation &
PID=$!

# Display training metrics every 30 seconds
echo -e "\nMonitoring training progress (Press Ctrl+C to stop)...\n"

iteration=0
while kill -0 $PID 2>/dev/null; do
    # Clear screen every iteration for a cleaner display
    clear
    
    # Print header
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║           NEUROSCOPE TRAINING PIPELINE - DETAILED METRICS        ║"
    echo "║══════════════════════════════════════════════════════════════════║"
    echo "║  Press Ctrl+C to stop monitoring (training will continue)        ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    
    # Print current time and iteration
    echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Training process running for: $(ps -o etime= -p $PID)"
    echo "Update #$iteration"
    
    # Display checkpoints progress
    check_checkpoint_progress
    
    # Display training metrics
    display_training_metrics
    
    # Display loss chart (every 3 iterations to avoid too much output)
    if [ $((iteration % 3)) -eq 0 ]; then
        display_loss_chart
    fi
    
    # Wait before next update
    iteration=$((iteration + 1))
    
    # Sleep with countdown
    for i in {30..1}; do
        echo -ne "\rNext update in $i seconds...   \r"
        sleep 1
    done
done

# Check if process exited normally or was terminated
wait $PID
EXIT_CODE=$?

# Final output
clear
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           NEUROSCOPE TRAINING PIPELINE - COMPLETED               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training pipeline completed successfully!"
    echo "Finished at: $(date)"
    
    # Show final metrics
    display_training_metrics
    display_loss_chart
    check_checkpoint_progress
    
    echo -e "\nTo view detailed training logs, run: tensorboard --logdir='$NEUROSCOPE_DIR/runs'"
else
    echo "Training pipeline failed with exit code: $EXIT_CODE"
    echo "Finished at: $(date)"
    echo "Check the logs for error messages."
    exit $EXIT_CODE
fi