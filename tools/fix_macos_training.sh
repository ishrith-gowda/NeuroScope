#!/bin/bash
# Fix for macOS malloc stack logging warnings

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export MallocStackLogging=0

# Run training
python scripts/02_training/train_comprehensive.py "$@"
