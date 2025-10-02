#!/bin/bash

# Script to move legacy pipeline scripts to the legacy directory
# This helps reorganize the project while preserving the original scripts

echo "Moving legacy scripts to the legacy directory..."

# Create necessary directories
mkdir -p /Volumes/usb\ drive/neuroscope/scripts/legacy/01_data_preparation_pipeline
mkdir -p /Volumes/usb\ drive/neuroscope/scripts/legacy/02_model_development_pipeline
mkdir -p /Volumes/usb\ drive/neuroscope/scripts/legacy/03_advanced_extensions

# Move data preparation pipeline
echo "Moving data preparation pipeline scripts..."
mv /Volumes/usb\ drive/neuroscope/scripts/01_data_preparation_pipeline/* /Volumes/usb\ drive/neuroscope/scripts/legacy/01_data_preparation_pipeline/

# Move model development pipeline
echo "Moving model development pipeline scripts..."
mv /Volumes/usb\ drive/neuroscope/scripts/02_model_development_pipeline/* /Volumes/usb\ drive/neuroscope/scripts/legacy/02_model_development_pipeline/

# Move advanced extensions
echo "Moving advanced extensions scripts..."
mv /Volumes/usb\ drive/neuroscope/scripts/03_advanced_extensions/* /Volumes/usb\ drive/neuroscope/scripts/legacy/03_advanced_extensions/

# Remove empty directories
echo "Removing empty directories..."
rmdir /Volumes/usb\ drive/neuroscope/scripts/01_data_preparation_pipeline
rmdir /Volumes/usb\ drive/neuroscope/scripts/02_model_development_pipeline
rmdir /Volumes/usb\ drive/neuroscope/scripts/03_advanced_extensions

echo "Legacy scripts have been moved successfully."
echo "The new CLI tools are available in /Volumes/usb drive/neuroscope/scripts/cli/"