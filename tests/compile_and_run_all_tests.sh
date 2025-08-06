#!/bin/bash
# Comprehensive Test Suite Compilation and Execution
# Compiles and runs all C++ unit tests with proper dependencies

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Paths
SDK_ROOT="/Users/jerry/Vulkan"
SDK_BUILD="$SDK_ROOT/builds/ARM-ML-SDK-Complete"
TEST_ROOT="$SDK_ROOT/tests"
VGF_INCLUDE="$SDK_ROOT/ai-ml-sdk-vgf-library/include"
BIN_DIR="$TEST_ROOT/bin"

# Create bin directory
mkdir -p "$BIN_DIR"

# Export library paths
export DYLD_LIBRARY_PATH=/usr/local/lib:/opt/homebrew/lib:$SDK_BUILD/lib

# Compilation flags
CXX="c++"
CXXFLAGS="-std=c++17 -O2 -march=native -Wall"
INCLUDES="-I$SDK_BUILD/include -I$VGF_INCLUDE -I/usr/local/include -I/opt/homebrew/include"
LDFLAGS="-L$SDK_BUILD/lib -L/usr/local/lib -L/opt/homebrew/lib"
LIBS="-lvgf -lvulkan -lpthread"
FRAMEWORKS="-framework Metal -framework Foundation -framework IOKit"

# Test categories (using arrays instead of associative array for compatibility)
test_names=("VGF Core" "Scenario Runner" "ML Operations")
test_files=(
    "$TEST_ROOT/unit/vgf/test_vgf_core.cpp"
    "$TEST_ROOT/unit/runner/test_scenario_runner.cpp"
    "$TEST_ROOT/validation/ml_operations/test_ml_ops_validation.cpp"
)

# Statistics
TOTAL_TESTS=0
COMPILED_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Vulkan ML SDK - Comprehensive Test Compilation        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to compile a test
compile_test() {
    local test_name=$1
    local test_file=$2
    local executable="$BIN_DIR/$(basename $test_file .cpp)"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -ne "${CYAN}Compiling $test_name...${NC} "
    
    if [ ! -f "$test_file" ]; then
        echo -e "${YELLOW}SKIP${NC} - File not found"
        return 1
    fi
    
    # Check for special dependencies
    local extra_libs=""
    
    # Add jsoncpp for scenario runner tests
    if [[ "$test_file" == *"scenario_runner"* ]]; then
        extra_libs="$extra_libs -ljsoncpp"
    fi
    
    # Compile
    if $CXX $CXXFLAGS $INCLUDES "$test_file" -o "$executable" $LDFLAGS $LIBS $extra_libs $FRAMEWORKS 2>/dev/null; then
        echo -e "${GREEN}SUCCESS${NC}"
        COMPILED_TESTS=$((COMPILED_TESTS + 1))
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        # Show compilation error
        echo "  Compilation command:"
        echo "  $CXX $CXXFLAGS $INCLUDES $test_file -o $executable $LDFLAGS $LIBS $extra_libs $FRAMEWORKS"
        return 1
    fi
}

# Function to run a test
run_test() {
    local test_name=$1
    local test_file=$2
    local executable="$BIN_DIR/$(basename $test_file .cpp)"
    
    if [ ! -x "$executable" ]; then
        echo -e "  ${YELLOW}⚠ $test_name not compiled, skipping${NC}"
        return 1
    fi
    
    echo -e "\n${MAGENTA}▶ Running $test_name${NC}"
    
    if "$executable" 2>/dev/null; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo -e "  ${RED}✗ Test execution failed${NC}"
        return 1
    fi
}

# Compile all tests
echo -e "${CYAN}═══ Compilation Phase ═══${NC}\n"

for i in "${!test_names[@]}"; do
    compile_test "${test_names[$i]}" "${test_files[$i]}"
done

echo ""
echo -e "${CYAN}Compilation Summary:${NC}"
echo "  Total: $TOTAL_TESTS"
echo -e "  ${GREEN}Compiled: $COMPILED_TESTS${NC}"
echo -e "  ${RED}Failed: $((TOTAL_TESTS - COMPILED_TESTS))${NC}"

# Run all compiled tests
echo ""
echo -e "${CYAN}═══ Execution Phase ═══${NC}"

for i in "${!test_names[@]}"; do
    run_test "${test_names[$i]}" "${test_files[$i]}"
done

# Final summary
echo ""
echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║                    FINAL TEST SUMMARY                     ║${NC}"
echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Compilation Results:"
echo "  Total Tests: $TOTAL_TESTS"
echo -e "  ${GREEN}Successfully Compiled: $COMPILED_TESTS${NC}"
echo -e "  ${RED}Compilation Failed: $((TOTAL_TESTS - COMPILED_TESTS))${NC}"
echo ""
echo "Execution Results:"
echo "  Tests Run: $COMPILED_TESTS"
echo -e "  ${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "  ${RED}Failed: $FAILED_TESTS${NC}"

if [ $COMPILED_TESTS -gt 0 ]; then
    SUCCESS_RATE=$((100 * PASSED_TESTS / COMPILED_TESTS))
    echo ""
    echo "Success Rate: ${SUCCESS_RATE}%"
    
    if [ $SUCCESS_RATE -ge 95 ]; then
        echo -e "\n${GREEN}🎉 EXCELLENT - SDK tests are production ready!${NC}"
    elif [ $SUCCESS_RATE -ge 80 ]; then
        echo -e "\n${GREEN}✅ GOOD - Most tests passing${NC}"
    elif [ $SUCCESS_RATE -ge 60 ]; then
        echo -e "\n${YELLOW}⚠️ FAIR - Needs improvement${NC}"
    else
        echo -e "\n${RED}❌ POOR - Major issues detected${NC}"
    fi
fi

# Create detailed test report
REPORT_FILE="$TEST_ROOT/reports/test_report_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$TEST_ROOT/reports"

echo "Vulkan ML SDK Test Report" > "$REPORT_FILE"
echo "Generated: $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "Platform: macOS ARM64 (Apple M4 Max)" >> "$REPORT_FILE"
echo "SDK Path: $SDK_BUILD" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "Compilation Summary:" >> "$REPORT_FILE"
echo "  Total: $TOTAL_TESTS" >> "$REPORT_FILE"
echo "  Compiled: $COMPILED_TESTS" >> "$REPORT_FILE"
echo "  Failed: $((TOTAL_TESTS - COMPILED_TESTS))" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "Execution Summary:" >> "$REPORT_FILE"
echo "  Passed: $PASSED_TESTS" >> "$REPORT_FILE"
echo "  Failed: $FAILED_TESTS" >> "$REPORT_FILE"

echo ""
echo "Report saved to: $REPORT_FILE"

# Exit with appropriate code
if [ $FAILED_TESTS -gt 0 ] || [ $COMPILED_TESTS -eq 0 ]; then
    exit 1
else
    exit 0
fi