#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Setup script for macOS ARM64 build environment
# Creates symlinks in sw/ directory to connect submodules

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SDK_DIR="$SCRIPT_DIR/ai-ml-sdk-for-vulkan"

echo "Setting up ML SDK for Vulkan on macOS ARM64..."
echo "Root directory: $SCRIPT_DIR"
echo "SDK directory: $SDK_DIR"

# Create sw/ directory if it doesn't exist
mkdir -p "$SDK_DIR/sw"

echo ""
echo "Creating symlinks in sw/ directory..."

# Create symlinks to submodule directories
create_symlink() {
    local name="$1"
    local target="$2"
    local link_path="$SDK_DIR/sw/$name"

    if [ -L "$link_path" ]; then
        rm "$link_path"
    fi

    if [ -d "$target" ]; then
        ln -sf "$target" "$link_path"
        echo "  + $name -> $target"
    else
        echo "  ! Warning: $target not found, skipping $name"
    fi
}

# VGF Library
create_symlink "vgf-lib" "$SCRIPT_DIR/ai-ml-sdk-vgf-library"

# Model Converter
create_symlink "model-converter" "$SCRIPT_DIR/ai-ml-sdk-model-converter"

# Scenario Runner
create_symlink "scenario-runner" "$SCRIPT_DIR/ai-ml-sdk-scenario-runner"

# Emulation Layer
create_symlink "emulation-layer" "$SCRIPT_DIR/ai-ml-emulation-layer-for-vulkan"

echo ""
echo "Verifying symlinks..."
if [ -f "$SDK_DIR/sw/vgf-lib/CMakeLists.txt" ]; then
    echo "  + VGF Library CMakeLists.txt found"
else
    echo "  ! VGF Library CMakeLists.txt NOT found"
fi

echo ""
echo "macOS ARM64 CMake settings:"
echo "  - CMAKE_OSX_ARCHITECTURES=arm64"
echo "  - CMAKE_CXX_STANDARD=17"
echo "  - CMAKE_CXX_STANDARD_REQUIRED=ON"
echo ""
echo "Setup complete! You can now build with:"
echo "  cd ai-ml-sdk-for-vulkan"
echo "  python3 scripts/build.py --build-type Release --threads 8"
echo ""
