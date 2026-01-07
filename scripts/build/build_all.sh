#!/bin/bash
# Full Build Script for ARM ML SDK for Vulkan
# Builds all dependencies and components from source
#
# Usage: ./build_all.sh [Release|Debug] [threads]

set -e

# Color codes for output (matching run_ml_demo.sh pattern)
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

# Script directory and SDK root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SDK_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Build configuration
BUILD_TYPE="${1:-Release}"
THREADS="${2:-8}"
BUILD_DIR="$SDK_ROOT/build-full"
DEPS_DIR="$BUILD_DIR/dependencies"
FINAL_SDK_DIR="$SDK_ROOT/builds/ARM-ML-SDK-Complete"

# SPIRV-Tools configuration
# IMPORTANT: Use arm-staging branch for ARM ML extension support
SPIRV_TOOLS_BRANCH="arm-staging"
SPIRV_TOOLS_REPO="https://github.com/KhronosGroup/SPIRV-Tools.git"

# Apple Silicon M-series optimization configuration
APPLE_SILICON_OPTIMIZATIONS=""
APPLE_SILICON_CMAKE_FLAGS=""
APPLE_SILICON_CFLAGS=""
APPLE_SILICON_CXXFLAGS=""

if [ "$(uname -m)" = "arm64" ] && [ "$(uname -s)" = "Darwin" ]; then
    # Detect Apple Silicon M-series and configure optimizations
    APPLE_SILICON_OPTIMIZATIONS="enabled"

    # CMAKE flags for Apple Silicon builds
    APPLE_SILICON_CMAKE_FLAGS="-DCMAKE_OSX_ARCHITECTURES=arm64"
    APPLE_SILICON_CMAKE_FLAGS="$APPLE_SILICON_CMAKE_FLAGS -DCMAKE_APPLE_SILICON_PROCESSOR=arm64"

    # Compiler optimization flags for Apple M-series (M1, M2, M3, M4, etc.)
    # -mcpu=apple-m1 enables baseline Apple Silicon optimizations
    # -mtune=native tunes for the specific M-series chip at runtime
    # -O3 enables aggressive optimizations for Release builds
    # -ffast-math enables floating point optimizations
    # -fvectorize enables auto-vectorization for NEON SIMD
    APPLE_SILICON_CFLAGS="-mcpu=apple-m1 -mtune=native -O3 -ffast-math -fvectorize"
    APPLE_SILICON_CXXFLAGS="-mcpu=apple-m1 -mtune=native -O3 -ffast-math -fvectorize"

    # ARM NEON SIMD is enabled by default on Apple Silicon, but we ensure it explicitly
    # The arm64 architecture on Apple Silicon always supports NEON
    APPLE_SILICON_CFLAGS="$APPLE_SILICON_CFLAGS -DARM_NEON_ENABLED=1"
    APPLE_SILICON_CXXFLAGS="$APPLE_SILICON_CXXFLAGS -DARM_NEON_ENABLED=1"
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}        ARM ML SDK - FULL BUILD SYSTEM                     ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo "  Build Type: $BUILD_TYPE"
echo "  Threads: $THREADS"
echo "  Build Dir: $BUILD_DIR"
echo "  SDK Output: $FINAL_SDK_DIR"
echo "  SPIRV-Tools Branch: $SPIRV_TOOLS_BRANCH"
if [ -n "$APPLE_SILICON_OPTIMIZATIONS" ]; then
    echo "  Apple Silicon: M-series optimizations enabled"
    echo "  CPU Target: apple-m1 (with native tuning)"
    echo "  SIMD: ARM NEON enabled"
fi
echo ""

# Helper functions
log_step() {
    echo -e "${CYAN}>>> $1${NC}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."

    local missing=0

    # Check for required tools
    for tool in cmake git python3 c++; do
        if ! command -v $tool &> /dev/null; then
            log_error "Missing required tool: $tool"
            missing=1
        else
            log_success "$tool found"
        fi
    done

    # Check for Vulkan SDK or MoltenVK
    if [ -n "$VULKAN_SDK" ] || [ -d "/usr/local/share/vulkan" ]; then
        log_success "Vulkan SDK/MoltenVK found"
    else
        log_warning "Vulkan SDK not found - will attempt to use system MoltenVK"
    fi

    # Check for Apple Silicon M-series
    if [ "$(uname -m)" = "arm64" ] && [ "$(uname -s)" = "Darwin" ]; then
        log_success "Apple Silicon M-series (ARM64) detected"
        log_success "M-series optimization flags enabled: -mcpu=apple-m1 -mtune=native"
        log_success "ARM NEON SIMD vectorization enabled"
    elif [ "$(uname -m)" = "arm64" ]; then
        log_success "ARM64 architecture detected (non-Apple)"
    else
        log_warning "Not running on Apple Silicon - M-series optimizations disabled"
    fi

    if [ $missing -eq 1 ]; then
        log_error "Prerequisites check failed"
        exit 1
    fi

    echo ""
}

# Build SPIRV-Tools from arm-staging branch
build_spirv_tools() {
    log_step "Building SPIRV-Tools from $SPIRV_TOOLS_BRANCH branch..."

    local spirv_dir="$DEPS_DIR/SPIRV-Tools"

    # Clone or update SPIRV-Tools
    if [ -d "$spirv_dir" ]; then
        log_warning "SPIRV-Tools directory exists, updating..."
        cd "$spirv_dir"
        git fetch origin
        git checkout $SPIRV_TOOLS_BRANCH
        git pull origin $SPIRV_TOOLS_BRANCH
    else
        mkdir -p "$DEPS_DIR"
        cd "$DEPS_DIR"
        git clone --branch $SPIRV_TOOLS_BRANCH --depth 1 "$SPIRV_TOOLS_REPO"
        cd SPIRV-Tools
    fi

    # Fetch SPIRV-Headers (required dependency)
    log_step "Fetching SPIRV-Headers..."
    python3 utils/git-sync-deps || {
        log_warning "git-sync-deps failed, trying manual fetch..."
        if [ ! -d "external/spirv-headers" ]; then
            git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Headers.git external/spirv-headers
        fi
    }

    # Configure and build
    mkdir -p build && cd build

    # Apply Apple Silicon M-series optimizations if available
    local cmake_flags=""
    if [ -n "$APPLE_SILICON_OPTIMIZATIONS" ]; then
        cmake_flags="$APPLE_SILICON_CMAKE_FLAGS"
        # Export compiler flags for Apple M-series optimization
        export CFLAGS="${CFLAGS:-} $APPLE_SILICON_CFLAGS"
        export CXXFLAGS="${CXXFLAGS:-} $APPLE_SILICON_CXXFLAGS"
        log_step "Applying Apple M-series optimization flags..."
    elif [ "$(uname -m)" = "arm64" ]; then
        cmake_flags="-DCMAKE_OSX_ARCHITECTURES=arm64"
    fi

    cmake .. \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_INSTALL_PREFIX="$DEPS_DIR/install" \
        -DSPIRV_WERROR=OFF \
        -DSPIRV_SKIP_TESTS=ON \
        $cmake_flags

    cmake --build . --config $BUILD_TYPE -j$THREADS
    cmake --install . --config $BUILD_TYPE

    log_success "SPIRV-Tools built successfully"
    cd "$SDK_ROOT"
    echo ""
}

# Build main SDK components
build_sdk() {
    log_step "Building ARM ML SDK main components..."

    # Check for main SDK build script
    if [ -f "$SDK_ROOT/ai-ml-sdk-for-vulkan/scripts/build.py" ]; then
        cd "$SDK_ROOT/ai-ml-sdk-for-vulkan"

        # Set environment for SPIRV-Tools
        export CMAKE_PREFIX_PATH="$DEPS_DIR/install:$CMAKE_PREFIX_PATH"
        export PKG_CONFIG_PATH="$DEPS_DIR/install/lib/pkgconfig:$PKG_CONFIG_PATH"

        python3 scripts/build.py \
            --build-type $BUILD_TYPE \
            --threads $THREADS \
            --build-dir "$BUILD_DIR/sdk-build"

        log_success "SDK build completed"
    else
        log_warning "Main SDK build script not found, attempting CMake build..."

        mkdir -p "$BUILD_DIR/sdk-build"
        cd "$BUILD_DIR/sdk-build"

        # Apply Apple Silicon M-series optimizations if available
        local sdk_cmake_flags=""
        if [ -n "$APPLE_SILICON_OPTIMIZATIONS" ]; then
            sdk_cmake_flags="$APPLE_SILICON_CMAKE_FLAGS"
            export CFLAGS="${CFLAGS:-} $APPLE_SILICON_CFLAGS"
            export CXXFLAGS="${CXXFLAGS:-} $APPLE_SILICON_CXXFLAGS"
            log_step "Applying Apple M-series optimization flags for SDK build..."
        fi

        cmake "$SDK_ROOT/ai-ml-sdk-for-vulkan" \
            -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
            -DCMAKE_PREFIX_PATH="$DEPS_DIR/install" \
            $sdk_cmake_flags

        cmake --build . --config $BUILD_TYPE -j$THREADS

        log_success "CMake build completed"
    fi

    cd "$SDK_ROOT"
    echo ""
}

# Collect and organize SDK components
collect_sdk() {
    log_step "Collecting SDK components to $FINAL_SDK_DIR..."

    mkdir -p "$FINAL_SDK_DIR/bin"
    mkdir -p "$FINAL_SDK_DIR/lib"
    mkdir -p "$FINAL_SDK_DIR/models"
    mkdir -p "$FINAL_SDK_DIR/shaders"
    mkdir -p "$FINAL_SDK_DIR/tools"
    mkdir -p "$FINAL_SDK_DIR/include"

    # Copy SPIRV libraries
    log_step "Copying SPIRV libraries..."
    if [ -d "$DEPS_DIR/install/lib" ]; then
        cp "$DEPS_DIR/install/lib"/libSPIRV*.a "$FINAL_SDK_DIR/lib/" 2>/dev/null || true
        local spirv_count=$(ls -1 "$FINAL_SDK_DIR/lib"/libSPIRV*.a 2>/dev/null | wc -l | tr -d ' ')
        log_success "Copied $spirv_count SPIRV libraries"
    fi

    # Copy from production build if available
    if [ -d "$SDK_ROOT/ai-ml-sdk-for-vulkan/arm-ml-sdk-vulkan-macos-production" ]; then
        log_step "Copying from production build..."

        # Libraries
        cp "$SDK_ROOT/ai-ml-sdk-for-vulkan/arm-ml-sdk-vulkan-macos-production/lib"/*.a "$FINAL_SDK_DIR/lib/" 2>/dev/null || true

        # Binaries
        if [ -f "$SDK_ROOT/ai-ml-sdk-for-vulkan/arm-ml-sdk-vulkan-macos-production/bin/scenario-runner" ]; then
            cp "$SDK_ROOT/ai-ml-sdk-for-vulkan/arm-ml-sdk-vulkan-macos-production/bin/scenario-runner" "$FINAL_SDK_DIR/bin/"
            chmod +x "$FINAL_SDK_DIR/bin/scenario-runner"
            log_success "Copied scenario-runner binary"
        fi
    fi

    # Copy models
    if [ -d "$SDK_ROOT/ai-ml-sdk-for-vulkan/unified-ml-sdk/models" ]; then
        cp "$SDK_ROOT/ai-ml-sdk-for-vulkan/unified-ml-sdk/models"/*.tflite "$FINAL_SDK_DIR/models/" 2>/dev/null || true
        local model_count=$(ls -1 "$FINAL_SDK_DIR/models"/*.tflite 2>/dev/null | wc -l | tr -d ' ')
        log_success "Copied $model_count TFLite models"
    fi

    # Copy shaders
    if [ -d "$SDK_ROOT/ai-ml-sdk-for-vulkan/unified-ml-sdk/shaders" ]; then
        cp "$SDK_ROOT/ai-ml-sdk-for-vulkan/unified-ml-sdk/shaders"/*.spv "$FINAL_SDK_DIR/shaders/" 2>/dev/null || true
        local shader_count=$(ls -1 "$FINAL_SDK_DIR/shaders"/*.spv 2>/dev/null | wc -l | tr -d ' ')
        log_success "Copied $shader_count SPIR-V shaders"
    fi

    # Copy Python tools
    if [ -d "$SDK_ROOT/ai-ml-sdk-for-vulkan/unified-ml-sdk/tools" ]; then
        cp "$SDK_ROOT/ai-ml-sdk-for-vulkan/unified-ml-sdk/tools"/*.py "$FINAL_SDK_DIR/tools/" 2>/dev/null || true
        local tool_count=$(ls -1 "$FINAL_SDK_DIR/tools"/*.py 2>/dev/null | wc -l | tr -d ' ')
        log_success "Copied $tool_count Python tools"
    fi

    echo ""
}

# Verify the build
verify_build() {
    log_step "Verifying build..."

    local errors=0

    # Check for scenario-runner
    if [ -f "$FINAL_SDK_DIR/bin/scenario-runner" ]; then
        log_success "scenario-runner binary present"

        # Try to run version check
        export DYLD_LIBRARY_PATH=/usr/local/lib:$FINAL_SDK_DIR/lib
        if "$FINAL_SDK_DIR/bin/scenario-runner" --version &>/dev/null; then
            log_success "scenario-runner executes successfully"
        else
            log_warning "scenario-runner binary may not be fully functional"
        fi
    else
        log_error "scenario-runner binary missing"
        errors=$((errors + 1))
    fi

    # Check libraries
    local lib_count=$(ls -1 "$FINAL_SDK_DIR/lib"/*.a 2>/dev/null | wc -l | tr -d ' ')
    if [ "$lib_count" -gt 0 ]; then
        log_success "Found $lib_count static libraries"
    else
        log_warning "No static libraries found"
    fi

    # Check SPIRV libraries specifically
    local spirv_count=$(ls -1 "$FINAL_SDK_DIR/lib"/libSPIRV*.a 2>/dev/null | wc -l | tr -d ' ')
    if [ "$spirv_count" -ge 6 ]; then
        log_success "Found $spirv_count SPIRV libraries (expected 7)"
    else
        log_warning "Only $spirv_count SPIRV libraries found (expected 7)"
    fi

    # Check models
    local model_count=$(ls -1 "$FINAL_SDK_DIR/models"/*.tflite 2>/dev/null | wc -l | tr -d ' ')
    if [ "$model_count" -ge 7 ]; then
        log_success "Found $model_count TFLite models"
    else
        log_warning "Only $model_count TFLite models found (expected 7)"
    fi

    # Check shaders
    local shader_count=$(ls -1 "$FINAL_SDK_DIR/shaders"/*.spv 2>/dev/null | wc -l | tr -d ' ')
    if [ "$shader_count" -ge 30 ]; then
        log_success "Found $shader_count SPIR-V shaders"
    else
        log_warning "Only $shader_count SPIR-V shaders found (expected 35+)"
    fi

    echo ""
    return $errors
}

# Print summary
print_summary() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}                    BUILD SUMMARY                          ${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${CYAN}SDK Location:${NC} $FINAL_SDK_DIR"
    echo ""
    echo -e "${CYAN}Components:${NC}"
    echo "  Binary: $FINAL_SDK_DIR/bin/scenario-runner"
    echo "  Libraries: $FINAL_SDK_DIR/lib/"
    echo "  Models: $FINAL_SDK_DIR/models/"
    echo "  Shaders: $FINAL_SDK_DIR/shaders/"
    echo "  Tools: $FINAL_SDK_DIR/tools/"
    echo ""
    echo -e "${CYAN}SPIRV-Tools Configuration:${NC}"
    echo "  Branch: $SPIRV_TOOLS_BRANCH (required for ARM ML extensions)"
    echo ""
    if [ -n "$APPLE_SILICON_OPTIMIZATIONS" ]; then
        echo -e "${CYAN}Apple Silicon M-series Optimizations:${NC}"
        echo "  CPU Target: apple-m1 (baseline for M1/M2/M3/M4)"
        echo "  Tuning: native (optimized for current M-series chip)"
        echo "  SIMD: ARM NEON vectorization enabled"
        echo "  Compiler: -O3 -ffast-math -fvectorize"
        echo ""
    fi
    echo -e "${CYAN}To use the SDK:${NC}"
    echo "  export DYLD_LIBRARY_PATH=/usr/local/lib:$FINAL_SDK_DIR/lib"
    echo "  $FINAL_SDK_DIR/bin/scenario-runner --version"
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
}

# Main execution
main() {
    echo "Starting full build at $(date)"
    echo ""

    check_prerequisites

    # Build SPIRV-Tools from arm-staging branch (required for ARM ML extensions)
    build_spirv_tools

    # Build main SDK
    build_sdk

    # Collect SDK components
    collect_sdk

    # Verify the build
    if verify_build; then
        echo -e "${GREEN}Build completed successfully!${NC}"
    else
        echo -e "${YELLOW}Build completed with warnings${NC}"
    fi

    print_summary

    echo "Build finished at $(date)"
}

# Run main function
main "$@"
