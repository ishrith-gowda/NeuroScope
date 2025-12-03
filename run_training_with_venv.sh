#!/bin/bash
# Script to run the training pipeline with the virtual environment

# Activate the virtual environment
source "/Volumes/usb drive/neuroscope/.venv/bin/activate"

# Print environment information
echo "============================================="
echo "RUNNING TRAINING PIPELINE WITH VIRTUAL ENVIRONMENT"
echo "============================================="
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo ""

# Install required packages if needed
echo "Ensuring required packages are installed..."
pip install tensorboard torch torchvision seaborn matplotlib torchinfo

# Run the training pipeline with verbose output
echo "Starting training pipeline..."
python scripts/02_model_development_pipeline/run_training_pipeline.py --verbose

# Return the exit code from the pipeline
exit $?