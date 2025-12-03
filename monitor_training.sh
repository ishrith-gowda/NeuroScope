#!/bin/bash

# ======================================================
# NeuroScope Training Monitor
# This script displays real-time training metrics from TensorBoard logs
# without interfering with the training process
# ======================================================

NEUROSCOPE_DIR="/Volumes/usb drive/neuroscope"
RUNS_DIR="$NEUROSCOPE_DIR/runs"
CHECKPOINTS_DIR="$NEUROSCOPE_DIR/checkpoints"
SAMPLES_DIR="$NEUROSCOPE_DIR/samples"

# Activate virtual environment if it exists
if [ -f "$NEUROSCOPE_DIR/.venv/bin/activate" ]; then
  source "$NEUROSCOPE_DIR/.venv/bin/activate"
fi

# Check if Python has required packages
python -c "
import sys
missing = []
try:
    import tensorboard
except ImportError:
    missing.append('tensorboard')
try:
    import numpy
except ImportError:
    missing.append('numpy')
try:
    import pandas
except ImportError:
    missing.append('pandas')

if missing:
    print(f'Missing required packages: {\", \".join(missing)}')
    print('Please install them with: pip install ' + ' '.join(missing))
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install tensorboard numpy pandas
fi

function display_header {
    clear
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║             NEUROSCOPE TRAINING MONITOR - REAL-TIME METRICS            ║"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo "Monitor started at: $(date)"
    echo "Press Ctrl+C to exit monitoring (training will continue)"
    echo ""
}

function display_checkpoint_info {
    echo "═══ CHECKPOINT PROGRESS ═══"
    
    if [ -d "$CHECKPOINTS_DIR" ]; then
        # Count checkpoint files by type
        local g_a2b_files=($(ls -1 "$CHECKPOINTS_DIR"/G_A2B_*.pth 2>/dev/null || echo ""))
        local g_b2a_files=($(ls -1 "$CHECKPOINTS_DIR"/G_B2A_*.pth 2>/dev/null || echo ""))
        local d_a_files=($(ls -1 "$CHECKPOINTS_DIR"/D_A_*.pth 2>/dev/null || echo ""))
        local d_b_files=($(ls -1 "$CHECKPOINTS_DIR"/D_B_*.pth 2>/dev/null || echo ""))
        
        # Find latest epoch number
        local latest_epoch=0
        if [ ${#g_a2b_files[@]} -gt 0 ]; then
            latest=$(echo "${g_a2b_files[${#g_a2b_files[@]}-1]}" | grep -o '[0-9]\+\.pth' | cut -d'.' -f1)
            if [ -n "$latest" ]; then
                latest_epoch=$latest
            fi
        fi
        
        echo "Latest epoch: $latest_epoch"
        echo "Generator A→B: ${#g_a2b_files[@]} checkpoints"
        echo "Generator B→A: ${#g_b2a_files[@]} checkpoints"
        echo "Discriminator A: ${#d_a_files[@]} checkpoints"
        echo "Discriminator B: ${#d_b_files[@]} checkpoints"
        
        # Get most recent checkpoint and its timestamp
        if [ ${#g_a2b_files[@]} -gt 0 ]; then
            local latest_file="${g_a2b_files[${#g_a2b_files[@]}-1]}"
            local timestamp=$(date -r "$latest_file" "+%Y-%m-%d %H:%M:%S")
            local age=$(( $(date +%s) - $(date -r "$latest_file" +%s) ))
            local age_mins=$((age / 60))
            local age_secs=$((age % 60))
            echo "Latest checkpoint: $timestamp ($age_mins min $age_secs sec ago)"
        fi
    else
        echo "No checkpoints directory found yet"
    fi
    echo ""
}

function display_sample_images {
    echo "═══ SAMPLE IMAGES ═══"
    
    if [ -d "$SAMPLES_DIR" ]; then
        # Get the most recent sample images
        local sample_files=($(ls -1t "$SAMPLES_DIR"/sample_*.png 2>/dev/null | head -3))
        
        if [ ${#sample_files[@]} -gt 0 ]; then
            for file in "${sample_files[@]}"; do
                local basename=$(basename "$file")
                local timestamp=$(date -r "$file" "+%Y-%m-%d %H:%M:%S")
                local step=$(echo "$basename" | grep -o '[0-9]\+' || echo "unknown")
                echo "- $basename (Step $step, generated at $timestamp)"
            done
        else
            echo "No sample images generated yet"
        fi
    else
        echo "No samples directory found yet"
    fi
    echo ""
}

function display_metrics {
    echo "═══ TRAINING METRICS ═══"
    
    # Find the latest TensorBoard log file
    local latest_log=$(find "$RUNS_DIR" -type f -name "events.out.tfevents.*" -print0 2>/dev/null | xargs -0 ls -t 2>/dev/null | head -1)
    
    if [ -n "$latest_log" ]; then
        # Use Python to extract and display metrics
        python -c "
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import numpy as np
import time
from datetime import datetime

try:
    # Get latest TensorBoard log file
    log_file = '$latest_log'
    
    # Load TensorBoard data
    event_acc = EventAccumulator(log_file)
    event_acc.Reload()
    
    # Get available tags (metrics)
    tags = event_acc.Tags()
    scalar_tags = tags.get('scalars', [])
    
    if not scalar_tags:
        print('No metrics available yet in the logs')
        sys.exit(0)
    
    # Extract metrics data
    print(f'Latest metrics from {log_file.split(\"/\")[-1]}:')
    print(f'\\n{\"METRIC\":<20} {\"STEP\":<10} {\"VALUE\":<15} {\"CHANGE\":<10}')
    print('-' * 60)
    
    metrics_data = {}
    prev_values = {}
    
    for tag in scalar_tags:
        # Skip non-essential tags for cleaner output
        if 'histogram' in tag.lower() or 'image' in tag.lower():
            continue
            
        events = event_acc.Scalars(tag)
        if events:
            # Calculate stats
            values = [e.value for e in events]
            steps = [e.step for e in events]
            
            # Get the most recent event
            latest = events[-1]
            step = latest.step
            value = latest.value
            
            # Calculate trend (change over last 5 events if available)
            trend = '—'
            if len(events) > 5:
                prev_avg = np.mean([e.value for e in events[-6:-1]])
                change = ((value - prev_avg) / prev_avg) * 100 if prev_avg != 0 else 0
                trend = f'{change:+.2f}%'
            
            # Format tag name for better display
            display_name = tag.replace('losses_', '').upper()
            metrics_data[display_name] = (step, value, trend)
    
    # Sort metrics by name for consistent display
    for metric, (step, value, trend) in sorted(metrics_data.items()):
        print(f'{metric:<20} {step:<10d} {value:<15.6f} {trend:<10}')
    
    # Calculate training progress based on steps
    steps = [step for _, (step, _, _) in metrics_data.items()]
    if steps:
        max_step = max(steps)
        target_steps = 100000  # Adjust based on expected training length
        progress = min(100, (max_step / target_steps) * 100)
        print(f'\\nEstimated progress: {progress:.1f}% (Step {max_step}/{target_steps})')
        
        # Calculate estimated time to completion
        if max_step > 0 and max_step < target_steps:
            log_mtime = datetime.fromtimestamp(event_acc.FirstEventTimestamp() / 1e9)
            current_time = datetime.now()
            elapsed_seconds = (current_time - log_mtime).total_seconds()
            steps_per_second = max_step / elapsed_seconds if elapsed_seconds > 0 else 0
            
            if steps_per_second > 0:
                remaining_steps = target_steps - max_step
                remaining_seconds = remaining_steps / steps_per_second
                remaining_hours = int(remaining_seconds // 3600)
                remaining_minutes = int((remaining_seconds % 3600) // 60)
                
                print(f'Elapsed time: {elapsed_seconds/3600:.1f} hours')
                print(f'Estimated completion in: {remaining_hours} hours {remaining_minutes} minutes')
    
        # Training rate
        events_by_time = {}
        for tag in scalar_tags:
            if 'loss' in tag.lower():  # Focus on main loss metrics
                events = event_acc.Scalars(tag)
                if len(events) > 1:
                    first, last = events[0], events[-1]
                    first_time = first.wall_time
                    last_time = last.wall_time
                    elapsed = last_time - first_time
                    steps = last.step - first.step
                    
                    if elapsed > 0:
                        steps_per_hour = (steps / elapsed) * 3600
                        print(f'Training rate: {steps_per_hour:.1f} steps/hour')
                        break
except Exception as e:
    print(f'Error reading metrics: {e}')
"
    else
        echo "No TensorBoard logs found yet. Training may still be initializing."
    fi
    
    echo ""
}

function display_loss_chart {
    echo "═══ LOSS CHART ═══"
    
    # Find the latest TensorBoard log file
    local latest_log=$(find "$RUNS_DIR" -type f -name "events.out.tfevents.*" -print0 2>/dev/null | xargs -0 ls -t 2>/dev/null | head -1)
    
    if [ -n "$latest_log" ]; then
        # Use Python to generate ASCII chart
        python -c "
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

try:
    # Load TensorBoard data
    log_file = '$latest_log'
    event_acc = EventAccumulator(log_file)
    event_acc.Reload()
    
    # Get available tags (metrics)
    tags = event_acc.Tags()
    scalar_tags = tags.get('scalars', [])
    
    # Find generator and discriminator loss tags
    gen_tag = next((tag for tag in scalar_tags if 'G' in tag or 'generator' in tag.lower()), None)
    disc_tag = next((tag for tag in scalar_tags if 'D' in tag or 'discriminator' in tag.lower()), None)
    
    if gen_tag and disc_tag:
        gen_events = event_acc.Scalars(gen_tag)
        disc_events = event_acc.Scalars(disc_tag)
        
        if len(gen_events) > 10 and len(disc_events) > 10:
            # Get loss values (last 50 points for a cleaner chart)
            max_points = 50
            gen_values = [e.value for e in gen_events[-max_points:]]
            disc_values = [e.value for e in disc_events[-max_points:]]
            steps = [e.step for e in gen_events[-max_points:]]
            
            # Create ASCII chart
            chart_height = 10
            chart_width = 50
            
            # Determine min/max for scaling
            all_values = gen_values + disc_values
            min_val = min(all_values)
            max_val = max(all_values)
            value_range = max_val - min_val or 1.0
            
            # Generate chart
            chart = [[' ' for _ in range(chart_width)] for _ in range(chart_height)]
            
            # Plot points
            for i in range(min(chart_width, len(gen_values))):
                idx = int(i * len(gen_values) / chart_width)
                gen_row = int((chart_height - 1) * (1 - (gen_values[idx] - min_val) / value_range))
                disc_row = int((chart_height - 1) * (1 - (disc_values[idx] - min_val) / value_range))
                
                # Clamp values to chart bounds
                gen_row = max(0, min(chart_height - 1, gen_row))
                disc_row = max(0, min(chart_height - 1, disc_row))
                
                chart[gen_row][i] = 'G'
                chart[disc_row][i] = 'D'
            
            # Print chart with border
            print(f'Loss chart over last {len(gen_values)} steps (G: Generator, D: Discriminator):')
            print(f'Max value: {max_val:.4f}')
            print('┌' + '─' * chart_width + '┐')
            for row in chart:
                print('│' + ''.join(row) + '│')
            print('└' + '─' * chart_width + '┘')
            print(f'Min value: {min_val:.4f}')
            print(f'Step range: {steps[0]} to {steps[-1]}')
        else:
            print('Not enough data points for loss chart yet')
    else:
        if len(scalar_tags) > 0:
            print(f'Loss tags not found. Available tags: {scalar_tags}')
        else:
            print('No scalar metrics available yet')
except Exception as e:
    print(f'Error creating loss chart: {e}')
"
    else
        echo "No TensorBoard logs found yet"
    fi
    
    echo ""
}

function display_gpu_memory {
    echo "═══ SYSTEM RESOURCES ═══"
    
    # Check available memory
    total_mem=$(sysctl hw.memsize | awk '{print $2/1024/1024/1024 " GB"}')
    free_mem=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
    page_size=$(sysctl hw.pagesize | awk '{print $2}')
    free_mem_gb=$(echo "scale=2; $free_mem * $page_size / 1024 / 1024 / 1024" | bc)
    
    echo "Total system memory: $total_mem"
    echo "Free memory: $free_mem_gb GB"
    
    # Check for GPU if pytorch is available
    python -c "
import sys
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    
    if torch.cuda.is_available():
        print(f'CUDA available: Yes')
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU memory allocated: {torch.cuda.memory_allocated(0)/1024/1024:.1f} MB')
        print(f'GPU memory reserved: {torch.cuda.memory_reserved(0)/1024/1024:.1f} MB')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f'MPS available: Yes')
        print(f'Using Apple Silicon GPU')
    else:
        print(f'GPU: None available (using CPU)')
except ImportError:
    print('PyTorch not available')
except Exception as e:
    print(f'Error checking GPU status: {e}')
" 
    echo ""
}

# Main loop
display_header

iteration=1
while true; do
    display_checkpoint_info
    display_metrics
    display_loss_chart
    display_sample_images
    display_gpu_memory
    
    echo "Update #$iteration completed at $(date '+%H:%M:%S')"
    echo "Press Ctrl+C to exit monitoring (next update in 30 seconds)..."
    
    # Wait before next update
    sleep 30
    display_header
    iteration=$((iteration + 1))
done