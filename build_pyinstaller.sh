#!/bin/bash
# Script to build standalone executable with PyInstaller (Easier than Nuitka)

echo "=========================================="
echo "Building Standalone Executable with PyInstaller"
echo "=========================================="

# Check if pyinstaller is installed
if ! python -c "import PyInstaller" &> /dev/null; then
    echo "PyInstaller is not installed. Installing..."
    pip install pyinstaller
fi

# Clean previous build
echo "Cleaning previous build..."
rm -rf build dist *.spec

# Create spec file
echo ""
echo "Creating PyInstaller spec file..."

cat > medical_xray_analyzer.spec << 'EOF'
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('weights', 'weights'),
        ('src', 'src'),
    ],
    hiddenimports=[
        'onnxruntime',
        'PIL',
        'cv2',
        'numpy',
        'pydicom',
        'torchvision',
        'torchvision.transforms',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='medical_xray_analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
EOF

echo "✓ Spec file created"

# Build with PyInstaller
echo ""
echo "Building executable..."
echo "This may take several minutes..."
echo ""

pyinstaller --clean --noconfirm medical_xray_analyzer.spec

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Build successful!"
    echo "=========================================="
    echo "Executable location: dist/medical_xray_analyzer"
    echo ""
    echo "Size:"
    du -h dist/medical_xray_analyzer
    echo ""
    echo "Test with:"
    echo "  ./dist/medical_xray_analyzer --image data/test/dicom/DICOM_HOANG\ VAN\ MINH_1760674525219.dcm"
    echo ""
else
    echo ""
    echo "❌ Build failed!"
    exit 1
fi

