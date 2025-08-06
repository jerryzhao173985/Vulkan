#!/bin/bash
# Main Build Script for Vulkan ML SDK
# Consolidated from multiple build scripts

set -e

SDK_ROOT="/Users/jerry/Vulkan"
BUILD_TYPE="${1:-Release}"
THREADS="${2:-8}"

# Use the optimized build script as the main builder
exec "$SDK_ROOT/scripts/build/build_optimized.sh" "$BUILD_TYPE" "$THREADS"