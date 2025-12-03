#!/bin/bash

# ======================================================
# NeuroScope Training Launch Script
# Runs both training and monitoring in separate terminals
# ======================================================

NEUROSCOPE_DIR="/Volumes/usb drive/neuroscope"

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                NEUROSCOPE TRAINING LAUNCHER                            ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"

# Function to detect terminal application
get_terminal_app() {
    if [ -x "$(command -v osascript)" ]; then
        echo "osascript"
    elif [ -x "$(command -v gnome-terminal)" ]; then
        echo "gnome-terminal"
    elif [ -x "$(command -v konsole)" ]; then
        echo "konsole"
    elif [ -x "$(command -v xterm)" ]; then
        echo "xterm"
    else
        echo "unknown"
    fi
}

TERMINAL_APP=$(get_terminal_app)

echo "Detected terminal application: $TERMINAL_APP"
echo "Starting training pipeline and monitoring..."

if [ "$TERMINAL_APP" = "osascript" ]; then
    # For macOS, use AppleScript to open new terminal windows
    osascript -e "tell application \"Terminal\"
        do script \"cd '$NEUROSCOPE_DIR' && python scripts/02_model_development_pipeline/run_training_pipeline.py --verbose --skip-validation\"
        do script \"cd '$NEUROSCOPE_DIR' && ./monitor_training.sh\"
    end tell"
elif [ "$TERMINAL_APP" = "gnome-terminal" ]; then
    # For GNOME (Ubuntu, etc.)
    gnome-terminal -- bash -c "cd '$NEUROSCOPE_DIR' && python scripts/02_model_development_pipeline/run_training_pipeline.py --verbose --skip-validation; exec bash"
    gnome-terminal -- bash -c "cd '$NEUROSCOPE_DIR' && ./monitor_training.sh; exec bash"
elif [ "$TERMINAL_APP" = "konsole" ]; then
    # For KDE
    konsole -e bash -c "cd '$NEUROSCOPE_DIR' && python scripts/02_model_development_pipeline/run_training_pipeline.py --verbose --skip-validation; exec bash" &
    konsole -e bash -c "cd '$NEUROSCOPE_DIR' && ./monitor_training.sh; exec bash" &
elif [ "$TERMINAL_APP" = "xterm" ]; then
    # For basic X terminal
    xterm -e "cd '$NEUROSCOPE_DIR' && python scripts/02_model_development_pipeline/run_training_pipeline.py --verbose --skip-validation; bash" &
    xterm -e "cd '$NEUROSCOPE_DIR' && ./monitor_training.sh; bash" &
else
    echo "Unable to detect terminal application. Starting both processes in this terminal."
    echo "Please open another terminal and run the monitoring script manually with:"
    echo "./monitor_training.sh"
    echo ""
    echo "Press Enter to continue with training pipeline only..."
    read
    
    cd "$NEUROSCOPE_DIR"
    python scripts/02_model_development_pipeline/run_training_pipeline.py --verbose --skip-validation
fi

echo "Training and monitoring started in separate terminals."
echo "You can view detailed metrics in the monitoring terminal."