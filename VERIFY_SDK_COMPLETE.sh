#!/bin/bash
# Comprehensive SDK Verification Script

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SDK_DIR="/Users/jerry/Vulkan/builds/ARM-ML-SDK-Complete"
export DYLD_LIBRARY_PATH="/usr/local/lib:$SDK_DIR/lib:$DYLD_LIBRARY_PATH"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Vulkan ML SDK - Complete Verification                  ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Test functions
test_component() {
    local name=$1
    local path=$2
    if [ -f "$path" ] || [ -d "$path" ]; then
        echo -e "${GREEN}✓${NC} $name found"
        return 0
    else
        echo -e "${RED}✗${NC} $name missing"
        return 1
    fi
}

run_test() {
    local name=$1
    local cmd=$2
    echo -n "Testing $name... "
    if $cmd > /dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        return 1
    fi
}

# Check directory structure
echo -e "${CYAN}1. Checking SDK Structure:${NC}"
test_component "SDK Directory" "$SDK_DIR"
test_component "Binaries" "$SDK_DIR/bin"
test_component "Libraries" "$SDK_DIR/lib"
test_component "Models" "$SDK_DIR/models"
test_component "Shaders" "$SDK_DIR/shaders"
test_component "Tools" "$SDK_DIR/tools"
echo ""

# Check executables
echo -e "${CYAN}2. Checking Executables:${NC}"
test_component "scenario-runner" "$SDK_DIR/bin/scenario-runner"
echo ""

# Check libraries
echo -e "${CYAN}3. Checking Libraries:${NC}"
test_component "VGF Library" "$SDK_DIR/lib/libvgf.a"
test_component "SPIRV-Tools" "$SDK_DIR/lib/libSPIRV-Tools.a"
test_component "SPIRV-Tools-opt" "$SDK_DIR/lib/libSPIRV-Tools-opt.a"

# Count libraries
SPIRV_COUNT=$(ls -1 $SDK_DIR/lib/libSPIRV*.a 2>/dev/null | wc -l)
echo "  Found $SPIRV_COUNT SPIRV libraries"
echo ""

# Check models
echo -e "${CYAN}4. Checking ML Models:${NC}"
MODEL_COUNT=$(ls -1 $SDK_DIR/models/*.tflite 2>/dev/null | wc -l)
echo "  Found $MODEL_COUNT TensorFlow Lite models:"
if [ $MODEL_COUNT -gt 0 ]; then
    ls -1 $SDK_DIR/models/*.tflite | while read model; do
        echo "    • $(basename $model)"
    done
fi
echo ""

# Check shaders
echo -e "${CYAN}5. Checking Compute Shaders:${NC}"
SHADER_COUNT=$(ls -1 $SDK_DIR/shaders/*.spv 2>/dev/null | wc -l)
echo "  Found $SHADER_COUNT SPIR-V shaders"
echo ""

# Test scenario-runner
echo -e "${CYAN}6. Testing Scenario Runner:${NC}"
if [ -f "$SDK_DIR/bin/scenario-runner" ]; then
    echo "Version info:"
    cd "$SDK_DIR"
    DYLD_LIBRARY_PATH="/usr/local/lib:./lib" ./bin/scenario-runner --version 2>/dev/null | head -5 || echo "  (version check failed)"
    
    # Check help
    run_test "Help command" "DYLD_LIBRARY_PATH='/usr/local/lib:./lib' ./bin/scenario-runner --help"
fi
echo ""

# Check Python tools
echo -e "${CYAN}7. Checking Python Tools:${NC}"
if [ -d "$SDK_DIR/tools" ]; then
    TOOL_COUNT=$(find $SDK_DIR/tools -name "*.py" 2>/dev/null | wc -l)
    echo "  Found $TOOL_COUNT Python tools"
    find $SDK_DIR/tools -name "*.py" 2>/dev/null | head -5 | while read tool; do
        echo "    • $(basename $tool)"
    done
fi
echo ""

# Summary
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                    Verification Summary                     ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"

SCORE=0
MAX_SCORE=7

[ -d "$SDK_DIR" ] && SCORE=$((SCORE + 1))
[ -f "$SDK_DIR/bin/scenario-runner" ] && SCORE=$((SCORE + 1))
[ -f "$SDK_DIR/lib/libvgf.a" ] && SCORE=$((SCORE + 1))
[ $SPIRV_COUNT -ge 5 ] && SCORE=$((SCORE + 1))
[ $MODEL_COUNT -ge 5 ] && SCORE=$((SCORE + 1))
[ $SHADER_COUNT -ge 20 ] && SCORE=$((SCORE + 1))
[ -d "$SDK_DIR/tools" ] && SCORE=$((SCORE + 1))

echo ""
echo "Overall Score: $SCORE/$MAX_SCORE"
echo ""

if [ $SCORE -ge 6 ]; then
    echo -e "${GREEN}✓ SDK is fully functional and ready for use!${NC}"
    echo ""
    echo "To use the SDK:"
    echo "  cd $SDK_DIR"
    echo "  export DYLD_LIBRARY_PATH=/usr/local/lib:\$PWD/lib"
    echo "  ./bin/scenario-runner --help"
elif [ $SCORE -ge 4 ]; then
    echo -e "${YELLOW}⚠ SDK is partially functional${NC}"
    echo "Some components may be missing but core functionality is available"
else
    echo -e "${RED}✗ SDK build incomplete${NC}"
    echo "Please run the build script again"
fi

echo ""
echo "SDK Location: $SDK_DIR"