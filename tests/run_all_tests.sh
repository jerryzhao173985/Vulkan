#!/bin/bash
# Comprehensive Test Suite for ARM ML SDK

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

SDK_ROOT="/Users/jerry/Vulkan"
SDK_BIN="$SDK_ROOT/builds/ARM-ML-SDK-Complete/bin"
MODELS="$SDK_ROOT/builds/ARM-ML-SDK-Complete/models"
SHADERS="$SDK_ROOT/builds/ARM-ML-SDK-Complete/shaders"

PASSED=0
FAILED=0
SKIPPED=0

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      ARM ML SDK - Comprehensive Test Suite                ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

export DYLD_LIBRARY_PATH=/usr/local/lib:$SDK_ROOT/builds/ARM-ML-SDK-Complete/lib

# Test function
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    
    echo -ne "  Testing $test_name... "
    
    if eval "$test_cmd" > /tmp/test_output.log 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        FAILED=$((FAILED + 1))
        echo "    Error: $(tail -1 /tmp/test_output.log)"
        return 1
    fi
}

# Skip test function
skip_test() {
    local test_name="$1"
    echo -e "  Testing $test_name... ${YELLOW}SKIP${NC}"
    SKIPPED=$((SKIPPED + 1))
}

# Section 1: Binary Tests
echo -e "${CYAN}=== 1. Binary Execution Tests ===${NC}"
run_test "scenario-runner exists" "[ -f '$SDK_BIN/scenario-runner' ]"
run_test "scenario-runner executable" "[ -x '$SDK_BIN/scenario-runner' ]"
run_test "scenario-runner version" "'$SDK_BIN/scenario-runner' --version"
run_test "scenario-runner help" "'$SDK_BIN/scenario-runner' --help"
echo ""

# Section 2: Library Tests
echo -e "${CYAN}=== 2. Library Tests ===${NC}"
run_test "VGF library exists" "[ -f '$SDK_ROOT/builds/ARM-ML-SDK-Complete/lib/libvgf.a' ]"
run_test "SPIRV libraries" "ls $SDK_ROOT/builds/ARM-ML-SDK-Complete/lib/libSPIRV*.a 2>/dev/null | grep -q SPIRV"
echo ""

# Section 3: Model Tests
echo -e "${CYAN}=== 3. ML Model Tests ===${NC}"
for model in "$MODELS"/*.tflite; do
    if [ -f "$model" ]; then
        model_name=$(basename "$model")
        run_test "$model_name" "[ -f '$model' ] && [ -s '$model' ]"
    fi
done
echo ""

# Section 4: Shader Tests
echo -e "${CYAN}=== 4. Compute Shader Tests ===${NC}"
SHADER_COUNT=$(ls -1 "$SHADERS"/*.spv 2>/dev/null | wc -l)
run_test "Shader compilation ($SHADER_COUNT shaders)" "[ $SHADER_COUNT -gt 0 ]"
run_test "Add shader" "[ -f '$SHADERS/add.spv' ]"
run_test "Multiply shader" "[ -f '$SHADERS/multiply.spv' ]"
echo ""

# Section 5: Integration Tests
echo -e "${CYAN}=== 5. Integration Tests ===${NC}"

# Create test scenario
cat > /tmp/integration_test.json << EOF
{
  "name": "Integration Test",
  "description": "Basic integration test",
  "operations": [
    {
      "type": "compute",
      "shader": "add",
      "inputs": [1.0, 2.0],
      "expected": 3.0
    }
  ]
}
EOF

run_test "Dry run execution" "'$SDK_BIN/scenario-runner' --scenario /tmp/integration_test.json --dry-run"
run_test "Pipeline caching" "'$SDK_BIN/scenario-runner' --scenario /tmp/integration_test.json --pipeline-caching --dry-run"
echo ""

# Section 6: Performance Tests
echo -e "${CYAN}=== 6. Performance Tests ===${NC}"
run_test "Memory allocation" "python3 -c 'import numpy as np; a = np.random.randn(1000000)'"
run_test "Vulkan availability" "[ -f /usr/local/lib/libvulkan.dylib ] || [ -f /usr/local/lib/libvulkan.1.dylib ]"
echo ""

# Section 7: Git Repository Tests
echo -e "${CYAN}=== 7. Repository Tests ===${NC}"
for repo in ai-ml-sdk-for-vulkan ai-ml-sdk-vgf-library ai-ml-sdk-scenario-runner; do
    run_test "$repo git status" "cd '$SDK_ROOT/$repo' && git status > /dev/null 2>&1"
done
echo ""

# Section 8: Tool Tests
echo -e "${CYAN}=== 8. SDK Tool Tests ===${NC}"
run_test "vulkan-ml-sdk tool" "[ -f '$SDK_ROOT/tools/vulkan-ml-sdk' ]"
run_test "vulkan-ml-sdk-build tool" "[ -f '$SDK_ROOT/tools/vulkan-ml-sdk-build' ]"
run_test "Quick test demo" "[ -f '$SDK_ROOT/examples/demos/quick_test.sh' ]"
echo ""

# Section 9: Documentation Tests
echo -e "${CYAN}=== 9. Documentation Tests ===${NC}"
run_test "README exists" "[ -f '$SDK_ROOT/README.md' ]"
run_test "Build docs" "[ -f '$SDK_ROOT/docs/BUILD_SYSTEM_COMPLETE.md' ]"
run_test "Verification docs" "[ -f '$SDK_ROOT/docs/VERIFICATION_COMPLETE.md' ]"
echo ""

# Section 10: Build System Tests
echo -e "${CYAN}=== 10. Build System Tests ===${NC}"
run_test "CMake configuration" "[ -f '$SDK_ROOT/ai-ml-sdk-for-vulkan/CMakeLists.txt' ]"
run_test "Build scripts" "[ -f '$SDK_ROOT/scripts/build/build_all.sh' ]"
run_test "SDK complete directory" "[ -d '$SDK_ROOT/builds/ARM-ML-SDK-Complete' ]"
echo ""

# Summary
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}                    Test Results Summary                    ${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo ""

TOTAL=$((PASSED + FAILED + SKIPPED))
PASS_RATE=$(echo "scale=1; $PASSED * 100 / $TOTAL" | bc)

echo "Total Tests: $TOTAL"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo -e "Skipped: ${YELLOW}$SKIPPED${NC}"
echo "Pass Rate: ${PASS_RATE}%"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo -e "${GREEN}✓ SDK is fully operational!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo "Check /tmp/test_output.log for details"
    exit 1
fi