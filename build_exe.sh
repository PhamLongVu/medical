#!/bin/bash
# Script to build standalone executable with Nuitka

echo "=========================================="
echo "Building Standalone Executable with Nuitka"
echo "=========================================="

# Check if nuitka is installed
if ! command -v nuitka3 &> /dev/null && ! python -c "import nuitka" &> /dev/null; then
    echo "❌ Nuitka is not installed!"
    echo "Install with: pip install nuitka"
    exit 1
fi

# Clean previous build
echo "Cleaning previous build..."
rm -rf build_exe dist_exe *.dist *.build *.onefile-build

# Build with Nuitka
echo ""
echo "Building executable..."
echo "This may take several minutes..."
echo ""

python -m nuitka \
    --standalone \
    --onefile \
    --static-libpython=no \
    --enable-plugin=numpy \
    --include-package=onnxruntime \
    --include-package=PIL \
    --include-package=cv2 \
    --include-data-dir=weights=weights \
    --include-data-dir=src=src \
    --output-dir=dist_exe \
    --output-filename=medical_xray_analyzer \
    --assume-yes-for-downloads \
    --show-progress \
    --show-memory \
    run.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Build successful!"
    echo "=========================================="
    echo "Executable location: dist_exe/medical_xray_analyzer"
    echo ""
    echo "Test with:"
    echo "  ./dist_exe/medical_xray_analyzer --image data/test/dicom/DICOM_HOANG\ VAN\ MINH_1760674525219.dcm"
else
    echo ""
    echo "❌ Build failed!"
    exit 1
fi

