#!/bin/bash

# Full Medical Image Analysis Pipeline
# Usage: ./run_pipeline.sh [input_file]
#
# Supported inputs:
#   - DICOM files (.dcm, .dicom)
#   - Image files (.png, .jpg, .jpeg, .bmp, .tiff, .tif)
#
# NOTE: All configurations are now managed in config.py
# You only need to specify the input file path (optional)
# If no file is specified, uses the default from config.py

# Change to script directory
cd "$(dirname "$0")"

# Input file path - can be overridden from command line
if [ -n "$1" ]; then
    INPUT_FILE="$1"
    echo "Using input file from command line: $INPUT_FILE"
else
    # Uses default input file from config.py
    echo "No input file specified, using default from config.py"
    INPUT_FILE="/media/vbdi/ssd2t/Medical/dung/projects/vindr-xray/dataset/images/vin_png/1.2.392.200046.100.2.1.1.157685.20200716140330.1.1.1.png"
fi

echo ""
echo "================================================================"
echo "  Full Medical Image Analysis Pipeline"
echo "================================================================"
echo "Supported: DICOM (.dcm) or Images (.png/.jpg/.jpeg/.bmp/.tiff)"
echo "Configuration: config.py"
echo "  - Models: Loaded from config"
echo "  - Detection threshold: conf=0.1, iou=0.1 (from config)"
echo "  - Parallel workers: 8 (from config)"
echo "================================================================"
echo ""

# Run pipeline - uses config.py for all defaults
if [ -n "$INPUT_FILE" ]; then
    python3 main.py --dicom "$INPUT_FILE"
else
    python3 main.py
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Pipeline completed successfully!"
else
    echo ""
    echo "✗ Pipeline failed!"
    exit 1
fi
