#!/bin/bash

# Make all CLI scripts executable

echo "Making CLI scripts executable..."

chmod +x /Volumes/usb\ drive/neuroscope/scripts/cli/n4_bias_correction.py
chmod +x /Volumes/usb\ drive/neuroscope/scripts/cli/preprocess_volumes.py
chmod +x /Volumes/usb\ drive/neuroscope/scripts/cli/register_volumes.py
chmod +x /Volumes/usb\ drive/neuroscope/scripts/cli/create_dataset_splits.py
chmod +x /Volumes/usb\ drive/neuroscope/scripts/reorganize_legacy_scripts.sh

echo "CLI scripts are now executable."