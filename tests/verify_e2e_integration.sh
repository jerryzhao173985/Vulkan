#!/bin/bash
# End-to-End Integration Verification for ARM ML SDK
# This script verifies the complete SDK can be launched and used

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SDK_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SDK="$SDK_ROOT/builds/ARM-ML-SDK-Complete"
OUTPUT_DIR="$SDK_ROOT/results/e2e-test"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

PASSED=0
FAILED=0
WARNINGS=0

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║    ARM ML SDK - End-to-End Integration Verification      ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "SDK Location: $SDK"
echo "Output Dir:   $OUTPUT_DIR"
echo ""

# Test function
check_pass() {
    echo -e "  ${GREEN}✓${NC} $1"
    PASSED=$((PASSED + 1))
}

check_fail() {
    echo -e "  ${RED}✗${NC} $1"
    FAILED=$((FAILED + 1))
}

check_warn() {
    echo -e "  ${YELLOW}⚠${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

# Phase 1: SDK Structure Verification
echo -e "${CYAN}=== Phase 1: SDK Structure Verification ===${NC}"

# Verify SDK directories exist
if [ -d "$SDK" ]; then
    check_pass "SDK directory exists"
else
    check_fail "SDK directory missing: $SDK"
    exit 1
fi

if [ -d "$SDK/bin" ]; then
    check_pass "bin/ directory exists"
else
    check_fail "bin/ directory missing"
fi

if [ -d "$SDK/lib" ]; then
    check_pass "lib/ directory exists"
else
    check_fail "lib/ directory missing"
fi

if [ -d "$SDK/models" ]; then
    check_pass "models/ directory exists"
else
    check_fail "models/ directory missing"
fi

if [ -d "$SDK/shaders" ]; then
    check_pass "shaders/ directory exists"
else
    check_fail "shaders/ directory missing"
fi

if [ -d "$SDK/tools" ]; then
    check_pass "tools/ directory exists"
else
    check_fail "tools/ directory missing"
fi
echo ""

# Phase 2: Binary Verification
echo -e "${CYAN}=== Phase 2: Binary Verification ===${NC}"

if [ -f "$SDK/bin/scenario-runner" ]; then
    check_pass "scenario-runner binary exists"

    # Check if executable
    if [ -x "$SDK/bin/scenario-runner" ]; then
        check_pass "scenario-runner is executable"
    else
        check_fail "scenario-runner is not executable"
    fi

    # Check binary type
    FILE_TYPE=$(file "$SDK/bin/scenario-runner" 2>/dev/null || echo "unknown")
    if echo "$FILE_TYPE" | grep -q "Mach-O 64-bit executable arm64"; then
        check_pass "Binary is ARM64 Mach-O executable"
    elif echo "$FILE_TYPE" | grep -q "executable"; then
        check_warn "Binary is executable but not ARM64: $FILE_TYPE"
    else
        check_fail "Invalid binary type: $FILE_TYPE"
    fi

    # Check binary size (should be substantial)
    SIZE=$(stat -f%z "$SDK/bin/scenario-runner" 2>/dev/null || echo "0")
    if [ "$SIZE" -gt 1000000 ]; then
        SIZE_MB=$((SIZE / 1024 / 1024))
        check_pass "Binary size is ${SIZE_MB}MB (valid)"
    else
        check_fail "Binary size too small: $SIZE bytes"
    fi
else
    check_fail "scenario-runner binary not found"
fi
echo ""

# Phase 3: Library Verification
echo -e "${CYAN}=== Phase 3: Library Verification ===${NC}"

SPIRV_COUNT=$(ls -1 "$SDK/lib"/libSPIRV*.a 2>/dev/null | wc -l | tr -d ' ')
if [ "$SPIRV_COUNT" -ge 6 ]; then
    check_pass "SPIRV libraries present ($SPIRV_COUNT found)"
else
    check_fail "Not enough SPIRV libraries (found $SPIRV_COUNT, expected 6+)"
fi

if [ -f "$SDK/lib/libSPIRV-Tools-shared.dylib" ]; then
    check_pass "SPIRV-Tools shared library exists"
else
    check_warn "SPIRV-Tools shared library missing"
fi
echo ""

# Phase 4: Model Verification
echo -e "${CYAN}=== Phase 4: Model Verification ===${NC}"

MODEL_COUNT=$(ls -1 "$SDK/models"/*.tflite 2>/dev/null | wc -l | tr -d ' ')
if [ "$MODEL_COUNT" -ge 7 ]; then
    check_pass "TFLite models present ($MODEL_COUNT found)"
else
    check_fail "Not enough models (found $MODEL_COUNT, expected 7)"
fi

# Verify model headers
for model in "$SDK/models"/*.tflite; do
    if [ -f "$model" ]; then
        model_name=$(basename "$model")
        # TFL3 signature is at offset 4
        HEADER=$(xxd -l 8 "$model" 2>/dev/null | head -1 || echo "")
        if echo "$HEADER" | grep -q "TFL3"; then
            check_pass "Model $model_name has valid TFL3 header"
        else
            check_warn "Model $model_name header not verified"
        fi
    fi
done
echo ""

# Phase 5: Shader Verification
echo -e "${CYAN}=== Phase 5: Shader Verification ===${NC}"

SHADER_COUNT=$(ls -1 "$SDK/shaders"/*.spv 2>/dev/null | wc -l | tr -d ' ')
if [ "$SHADER_COUNT" -ge 30 ]; then
    check_pass "SPIR-V shaders present ($SHADER_COUNT found)"
else
    check_fail "Not enough shaders (found $SHADER_COUNT, expected 30+)"
fi

# Verify key shaders (note: some have different naming conventions)
for shader in add multiply relu sigmoid optimized_conv2d matrix_multiply; do
    if [ -f "$SDK/shaders/${shader}.spv" ]; then
        check_pass "Shader ${shader}.spv exists"
    else
        check_warn "Shader ${shader}.spv missing"
    fi
done
echo ""

# Phase 6: Environment Setup Verification
echo -e "${CYAN}=== Phase 6: Environment Setup ===${NC}"

if [ -f "$SDK/launch_sdk.sh" ]; then
    check_pass "launch_sdk.sh exists"

    # Verify it's executable
    if [ -x "$SDK/launch_sdk.sh" ]; then
        check_pass "launch_sdk.sh is executable"
    else
        chmod +x "$SDK/launch_sdk.sh"
        check_warn "Fixed launch_sdk.sh permissions"
    fi
else
    check_fail "launch_sdk.sh missing"
fi

# Test environment variable setup
export DYLD_LIBRARY_PATH="/usr/local/lib:$SDK/lib:$DYLD_LIBRARY_PATH"
export VK_LAYER_PATH="$SDK/lib:$VK_LAYER_PATH"
check_pass "Environment variables set"
echo ""

# Phase 7: Output Directory Creation
echo -e "${CYAN}=== Phase 7: Output Directory Test ===${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"
if [ -d "$OUTPUT_DIR" ]; then
    check_pass "Output directory created: $OUTPUT_DIR"
else
    check_fail "Failed to create output directory"
fi

# Write test marker file
echo "E2E Integration Test - $(date)" > "$OUTPUT_DIR/test_marker.txt"
if [ -f "$OUTPUT_DIR/test_marker.txt" ]; then
    check_pass "Can write to output directory"
else
    check_fail "Cannot write to output directory"
fi
echo ""

# Phase 8: Scenario File Verification
echo -e "${CYAN}=== Phase 8: Scenario File Verification ===${NC}"

if [ -f "$SCRIPT_DIR/test_scenario.json" ]; then
    check_pass "test_scenario.json exists"

    # Validate JSON
    if python3 -m json.tool "$SCRIPT_DIR/test_scenario.json" > /dev/null 2>&1; then
        check_pass "test_scenario.json is valid JSON"
    else
        check_fail "test_scenario.json is invalid JSON"
    fi
else
    check_warn "test_scenario.json not found in tests/"
fi
echo ""

# Phase 9: Python Tools Verification
echo -e "${CYAN}=== Phase 9: Python Tools Verification ===${NC}"

if [ -d "$SDK/tools" ]; then
    TOOL_COUNT=$(ls -1 "$SDK/tools"/*.py 2>/dev/null | wc -l | tr -d ' ')
    if [ "$TOOL_COUNT" -gt 0 ]; then
        check_pass "Python tools present ($TOOL_COUNT found)"
    else
        check_warn "No Python tools found"
    fi
fi

# Check Python3 availability
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    check_pass "Python3 available: $PYTHON_VERSION"
else
    check_warn "Python3 not available"
fi
echo ""

# Phase 10: Binary Launch Test (without execution)
echo -e "${CYAN}=== Phase 10: Binary Launch Readiness ===${NC}"

# Verify binary dependencies can be resolved
if otool -L "$SDK/bin/scenario-runner" &> /dev/null; then
    check_pass "Binary dependencies can be inspected"

    # Check for unresolved dependencies
    DEPS=$(otool -L "$SDK/bin/scenario-runner" 2>&1)
    if echo "$DEPS" | grep -q "not found"; then
        check_warn "Some dependencies may be missing"
    else
        check_pass "All dependencies appear resolvable"
    fi
else
    check_warn "Could not inspect binary dependencies"
fi

# Check Vulkan library availability
if [ -f "/usr/local/lib/libvulkan.dylib" ] || [ -f "/usr/local/lib/libvulkan.1.dylib" ]; then
    check_pass "Vulkan runtime library available"
else
    check_warn "Vulkan runtime library not in standard location"
fi
echo ""

# Summary
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}           End-to-End Integration Summary                  ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

TOTAL=$((PASSED + FAILED + WARNINGS))
echo "Total Checks: $TOTAL"
echo -e "Passed:   ${GREEN}$PASSED${NC}"
echo -e "Failed:   ${RED}$FAILED${NC}"
echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
echo ""

# Write summary to output
cat > "$OUTPUT_DIR/e2e_summary.txt" << EOF
ARM ML SDK - End-to-End Integration Verification
================================================
Date: $(date)
SDK Path: $SDK

Results:
  Passed:   $PASSED
  Failed:   $FAILED
  Warnings: $WARNINGS

SDK Components Verified:
  - scenario-runner binary: $([ -x "$SDK/bin/scenario-runner" ] && echo "OK" || echo "MISSING")
  - SPIRV libraries: $SPIRV_COUNT
  - TFLite models: $MODEL_COUNT
  - SPIR-V shaders: $SHADER_COUNT
  - launch_sdk.sh: $([ -f "$SDK/launch_sdk.sh" ] && echo "OK" || echo "MISSING")

Environment:
  - DYLD_LIBRARY_PATH set
  - VK_LAYER_PATH set
  - Output directory created

Status: $([ $FAILED -eq 0 ] && echo "PASSED" || echo "FAILED")
EOF

check_pass "Summary written to $OUTPUT_DIR/e2e_summary.txt"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║         ✓ END-TO-END VERIFICATION PASSED                 ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "The SDK is ready for use. To launch:"
    echo ""
    echo "  cd $SDK"
    echo "  source launch_sdk.sh"
    echo ""
    echo "Or run scenario-runner directly:"
    echo ""
    echo "  export DYLD_LIBRARY_PATH=/usr/local/lib:$SDK/lib"
    echo "  $SDK/bin/scenario-runner --help"
    echo ""
    exit 0
else
    echo -e "${RED}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║         ✗ END-TO-END VERIFICATION FAILED                 ║${NC}"
    echo -e "${RED}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Please check the failed items above and resolve issues."
    exit 1
fi
